"""
LLM Prompt Engineering and API Integration

This module handles:
- Constructing structured prompts for the LLM
- Making API calls to Gemini (or other LLMs)
- Parsing and validating LLM responses
"""

import json
import os
from typing import Dict, Optional
from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_MAX_RETRIES,
    LLM_JSON_PREVIEW_CHARS,
    SUPPORTED_EFFECTS,
)

def _call_with_google_genai(full_prompt: str, api_key: str) -> str:
    """Primary Gemini call path using the new google.genai SDK."""
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=full_prompt,
        config={
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_TOKENS,
        },
    )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("google.genai returned an empty response text")
    return text


def _call_with_legacy_google_generativeai(full_prompt: str, api_key: str) -> str:
    """Fallback Gemini call path using deprecated google.generativeai SDK."""
    import google.generativeai as legacy_genai

    legacy_genai.configure(api_key=api_key)
    model = legacy_genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(
        full_prompt,
        generation_config=legacy_genai.types.GenerationConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        ),
    )

    if not getattr(response, "text", None):
        raise RuntimeError("google.generativeai returned an empty response text")
    return response.text


class LLMPromptGenerator:
    """
    Generates structured prompts for the LLM to produce time-variant audio parameters.
    """
    
    SYSTEM_PROMPT = """You are an expert audio effect engineer. Your task is to translate natural language descriptions of audio effects into precise, time-variant parameter sequences.

Given a user's description of desired audio effects, you must output a valid JSON object that specifies:
1. Which effects to apply (reverb, chorus, distortion, low_pass_filter)
2. When each effect starts and ends (in seconds)
3. How parameters change over time (using "start" and "end" values for smooth interpolation)

IMPORTANT CONSTRAINTS:
- All times must be in seconds and in ascending order
- Parameter values must be within the valid ranges specified
- Provide realistic, musical parameter values
- Ensure smooth transitions between parameter values

OUTPUT FORMAT (strict JSON):
{
  "effects": [
    {
      "type": "reverb",
      "start_time": 0.0,
      "end_time": 10.0,
      "decay_time": {"start": 0.5, "end": 2.0},
      "wet_dry": {"start": 0.2, "end": 0.8},
      "width": {"start": 1.0, "end": 1.0}
    },
    {
      "type": "distortion",
      "start_time": 5.0,
      "end_time": 12.0,
      "gain": {"start": 0.0, "end": 2.0},
      "tone": {"start": 0.5, "end": 0.3}
    }
  ],
  "total_duration_seconds": 15.0
}

VALID PARAMETER RANGES:
- reverb.decay_time: 0.1 to 5.0 seconds
- reverb.wet_dry: 0.0 to 1.0 (0=dry, 1=wet)
- reverb.width: 0.0 to 1.0
- chorus.rate: 0.1 to 10.0 Hz
- chorus.depth: 0.0 to 1.0
- chorus.wet_dry: 0.0 to 1.0
- distortion.gain: 0.0 to 10.0
- distortion.tone: 0.0 to 1.0
- low_pass_filter.cutoff_freq: 20 to 20000 Hz
- low_pass_filter.resonance: 0.0 to 1.0

Respond ONLY with valid JSON, no additional text."""

    def __init__(self):
        """Initialize the LLM prompt generator."""
        self.system_prompt = self.SYSTEM_PROMPT
        
    def generate_prompt(self, user_description: str, audio_duration: Optional[float] = None) -> str:
        """
        Generate the full prompt for the LLM including user instructions.
        
        Args:
            user_description: Natural language description of desired effects
            audio_duration: Length of the audio file in seconds (optional context)
            
        Returns:
            Formatted prompt string for LLM input
        """
        duration_context = ""
        if audio_duration:
            duration_context = f"\n\nThe input audio has a duration of {audio_duration:.1f} seconds. Ensure all effect timings fit within this duration."
        
        user_prompt = f"""Generate audio effect parameters for the following description:

"{user_description}"{duration_context}

Output ONLY valid JSON object."""
        
        return user_prompt


def call_gemini_api(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call the Gemini API to generate parameter sequences.
    
    Args:
        prompt: The user prompt describing desired effects
        system_prompt: System prompt for context (optional)
        
    Returns:
        JSON string from LLM output
        
    Raises:
        ValueError: If API key is not set or API call fails
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it with: export GEMINI_API_KEY='your-key'"
        )
    
    # Prepare messages with system prompt
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    try:
        return _call_with_google_genai(full_prompt, api_key)
    except Exception as new_sdk_error:
        # Keep backward compatibility as a fallback path.
        try:
            return _call_with_legacy_google_generativeai(full_prompt, api_key)
        except Exception as legacy_error:
            raise RuntimeError(
                "Gemini API call failed for both SDKs. "
                f"google.genai error: {new_sdk_error}; "
                f"google.generativeai fallback error: {legacy_error}"
            )


def extract_json_from_response(response_text: str) -> Dict:
    """
    Extract JSON object from LLM response text.
    Handles cases where the LLM includes extra text before/after JSON.
    
    Args:
        response_text: Raw text response from LLM
        
    Returns:
        Parsed JSON object as dictionary
        
    Raises:
        ValueError: If valid JSON cannot be extracted
    """
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # First try direct parse.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Robust parse: find the first complete JSON object inside any surrounding text.
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # Helpful diagnostics for truncated / malformed response.
    snippet = cleaned[:LLM_JSON_PREVIEW_CHARS]
    if "{" in cleaned and "}" not in cleaned:
        raise ValueError(
            "Failed to parse JSON: response appears truncated (missing closing braces). "
            f"Response preview: {snippet}"
        )

    raise ValueError(f"Failed to parse JSON: no valid JSON object found. Response preview: {snippet}")


def validate_effect_parameters(parameters: Dict) -> None:
    """
    Validate minimal schema of generated parameters.

    Args:
        parameters: Parsed JSON dictionary

    Raises:
        ValueError: If schema is invalid
    """
    if not isinstance(parameters, dict):
        raise ValueError("Output must be a JSON object")

    effects = parameters.get("effects")
    if not isinstance(effects, list):
        raise ValueError("Output JSON must contain an 'effects' list")

    for i, effect in enumerate(effects):
        if not isinstance(effect, dict):
            raise ValueError(f"effects[{i}] must be an object")

        for required in ["type", "start_time", "end_time"]:
            if required not in effect:
                raise ValueError(f"effects[{i}] missing required field '{required}'")

        if effect["type"] not in SUPPORTED_EFFECTS:
            raise ValueError(f"effects[{i}].type '{effect['type']}' is not supported")


def _build_retry_prompt(user_prompt: str, previous_response: str, error_message: str) -> str:
    """Build a stricter retry prompt when prior output was malformed."""
    preview = previous_response[:LLM_JSON_PREVIEW_CHARS]
    return (
        "Your previous output was invalid JSON and cannot be parsed.\n"
        f"Parse error: {error_message}\n"
        "Return ONLY one valid JSON object. No markdown. No explanation.\n"
        "JSON must include: {\"effects\": [...]} and each effect must have type/start_time/end_time.\n"
        f"Previous invalid response preview:\n{preview}\n\n"
        f"Original request:\n{user_prompt}"
    )


def generate_effect_parameters(
    user_description: str,
    audio_duration: Optional[float] = None,
) -> Dict:
    """
    Main function to generate effect parameters from text description.
    
    Args:
        user_description: Natural language description of desired effects
        audio_duration: Optional audio duration for context
        
    Returns:
        Dictionary containing effect sequence with time-variant parameters
    """
    generator = LLMPromptGenerator()
    full_prompt = generator.generate_prompt(user_description, audio_duration)

    current_prompt = full_prompt
    last_error: Optional[str] = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        response = call_gemini_api(current_prompt, system_prompt=generator.system_prompt)

        try:
            parameters = extract_json_from_response(response)
            validate_effect_parameters(parameters)
            return parameters
        except ValueError as e:
            last_error = str(e)
            if attempt == LLM_MAX_RETRIES:
                break
            current_prompt = _build_retry_prompt(full_prompt, response, str(e))

    raise ValueError(
        f"Failed to generate valid effect parameters after {LLM_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
