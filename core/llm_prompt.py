"""
LLM Prompt Engineering and API Integration

This module handles:
- Constructing structured prompts for the LLM (text-only and multimodal)
- Making API calls to Gemini (text and vision models)
- Parsing and validating LLM responses
"""

import json
import os
from io import BytesIO
from typing import Dict, Optional, Union

from config import (
    LLM_JSON_PREVIEW_CHARS,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_MODEL_VISION,
    LLM_TEMPERATURE,
    SUPPORTED_EFFECTS,
)
from utils.reference_audio import format_reference_context


EXTRACTABLE_EFFECTS = set(SUPPORTED_EFFECTS) | {"delay"}


def _normalize_effect_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lowpass": "low_pass_filter",
        "low_pass": "low_pass_filter",
        "lp_filter": "low_pass_filter",
    }
    return aliases.get(normalized, normalized)


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


def _call_with_google_genai_multimodal(
    prompt: str,
    image_buffer: BytesIO,
    api_key: str,
) -> str:
    """Multimodal Gemini call using google.genai SDK with vision model."""
    from google import genai

    client = genai.Client(api_key=api_key)
    
    # Read image bytes from buffer
    image_buffer.seek(0)
    image_bytes = image_buffer.read()

    # Create multimodal content with text + image
    response = client.models.generate_content(
        model=LLM_MODEL_VISION,
        contents=[
            prompt,
            {"inline_data": {"mime_type": "image/png", "data": image_bytes}},
        ],
        config={
            "temperature": LLM_TEMPERATURE,
            "max_output_tokens": LLM_MAX_TOKENS,
        },
    )

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("google.genai multimodal call returned an empty response text")
    return text


def _call_with_legacy_google_generativeai_multimodal(
    prompt: str,
    image_buffer: BytesIO,
    api_key: str,
) -> str:
    """Fallback multimodal Gemini call using deprecated google.generativeai SDK."""
    import google.generativeai as legacy_genai

    legacy_genai.configure(api_key=api_key)
    model = legacy_genai.GenerativeModel(LLM_MODEL_VISION)

    # Read image bytes
    image_buffer.seek(0)
    image_data = {"mime_type": "image/png", "data": image_buffer.read()}

    response = model.generate_content(
        [prompt, image_data],
        generation_config=legacy_genai.types.GenerationConfig(
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        ),
    )

    if not getattr(response, "text", None):
        raise RuntimeError("google.generativeai multimodal fallback returned empty response")
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

    def generate_prompt(
        self,
        user_description: str,
        audio_duration: Optional[float] = None,
        reference_context: Optional[Dict] = None,
    ) -> str:
        """
        Generate the full prompt for the LLM including user instructions.

        Args:
            user_description: Natural language description of desired effects
            audio_duration: Length of the audio file in seconds (optional context)
            reference_context: Optional reference-audio spectrogram summary

        Returns:
            Formatted prompt string for LLM input
        """
        duration_context = ""
        if audio_duration is not None:
            duration_context = (
                f"\n\nThe input audio has a duration of {audio_duration:.1f} seconds. "
                "Ensure all effect timings fit within this duration."
            )

        reference_context_block = ""
        if reference_context:
            reference_context_block = f"\n\n{format_reference_context(reference_context)}"

        user_prompt = f"""Generate audio effect parameters for the following description:

"{user_description}"{reference_context_block}{duration_context}

Output ONLY valid JSON object."""

        return user_prompt


class LLMVisionPromptGenerator:
    """
    Generates prompts for multimodal VLM to reverse-engineer effects from spectrograms.
    """

    SYSTEM_PROMPT_VISION = """You are a professional audio engineer specializing in effect reverse-engineering. Your task is to analyze a spectrogram visualization of reference audio and extract the audio effects and their time-varying parameters.

SPECTROGRAM ANALYSIS GUIDELINES:
1. Look for spectrogram "fingerprints" that indicate specific effects:
   - REVERB: Horizontal "tail" or decay pattern; high frequencies fade slowly
   - DELAY: Regular repeated echo patterns in the spectrogram
   - DISTORTION: Harmonic spread (vertical striations); increased high-frequency content
   - LOW-PASS FILTER: Gradual attenuation of high frequencies; brightening/darkening over time

2. Identify TIME-VARIANT BEHAVIOR:
   - Track parameter changes across the spectrogram timeline (X-axis)
   - Look for transitions: gradual increases/decreases vs. sudden changes
   - Note which frequency bands are affected most heavily

3. EXTRACT TIME SEGMENTS:
   - Divide the reference audio into logical segments (e.g., 2-5 second windows)
   - For each segment, estimate the effect intensity and parameters
   - Focus on Reverb, Delay, and Distortion primarily

OUTPUT FORMAT (strict JSON):
Your output must be a valid JSON object with time-segmented effect parameters. Use this structure:
{
  "effects": [
    {
      "name": "Reverb",
      "time_segments": [
        {
          "start_time": 0.0,
          "end_time": 5.0,
          "decay_time": 1.5,
          "wet_dry": 0.3,
          "width": 0.9
        },
        {
          "start_time": 5.0,
          "end_time": 10.0,
          "decay_time": 2.5,
          "wet_dry": 0.5,
          "width": 0.95
        }
      ]
    },
    {
      "name": "Delay",
      "time_segments": [
        {
          "start_time": 2.0,
          "end_time": 8.0,
          "delay_time": 0.3,
          "feedback": 0.4,
          "wet_dry": 0.2
        }
      ]
    },
    {
      "name": "Distortion",
      "time_segments": [
        {
          "start_time": 3.0,
          "end_time": 9.0,
          "gain": 1.5,
          "tone": 0.6
        }
      ]
    }
  ],
  "analysis_notes": "Brief description of observed effects and confidence level"
}

VALID PARAMETER RANGES:
- Reverb.decay_time: 0.1 to 5.0 seconds
- Reverb.wet_dry: 0.0 to 1.0 (0=dry, 1=wet)
- Reverb.width: 0.0 to 1.0
- Delay.delay_time: 0.1 to 2.0 seconds
- Delay.feedback: 0.0 to 0.95
- Delay.wet_dry: 0.0 to 1.0
- Distortion.gain: 0.0 to 10.0
- Distortion.tone: 0.0 to 1.0

IMPORTANT CONSTRAINTS:
- All times must be in seconds and in ascending order
- Estimate parameters conservatively; only include segments where you detect clear evidence
- If you are unsure about an effect, use lower intensity values
- Ensure smooth transitions between consecutive time segments
- Respond ONLY with valid JSON, no additional text"""

    def __init__(self):
        """Initialize the VLM prompt generator."""
        self.system_prompt = self.SYSTEM_PROMPT_VISION

    def generate_multimodal_prompt(
        self,
        user_description: str = "Reverse-engineer the effects from this reference audio spectrogram.",
        audio_duration: Optional[float] = None,
    ) -> str:
        """
        Generate a prompt for multimodal VLM analysis of spectrogram.

        Args:
            user_description: Custom instruction (default: reverse-engineering task)
            audio_duration: Optional audio duration for context

        Returns:
            Formatted prompt string for multimodal LLM input
        """
        duration_context = ""
        if audio_duration is not None:
            duration_context = (
                f"\n\nThe reference audio has a duration of {audio_duration:.1f} seconds. "
                "Ensure all effect timings fit within this duration."
            )

        user_prompt = f"""{user_description}{duration_context}

Analyze the attached spectrogram carefully. Output ONLY valid JSON with time-segmented effect parameters."""

        return user_prompt


def call_gemini_api(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call the Gemini API to generate parameter sequences (text-only).

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

    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"

    try:
        return _call_with_google_genai(full_prompt, api_key)
    except Exception as new_sdk_error:
        try:
            return _call_with_legacy_google_generativeai(full_prompt, api_key)
        except Exception as legacy_error:
            raise RuntimeError(
                "Gemini API call failed for both SDKs. "
                f"google.genai error: {new_sdk_error}; "
                f"google.generativeai fallback error: {legacy_error}"
            )


def call_gemini_vision_api(
    prompt: str,
    image_buffer: BytesIO,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Call the Gemini Vision API for multimodal spectrogram analysis.

    Args:
        prompt: The user prompt (e.g., reverse-engineering instruction)
        image_buffer: BytesIO object containing spectrogram PNG image
        system_prompt: System prompt for vision model (optional)

    Returns:
        JSON string from multimodal LLM output

    Raises:
        ValueError: If API key is not set or API call fails
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it with: export GEMINI_API_KEY='your-key'"
        )

    # Prepare full prompt with system instructions
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"

    try:
        return _call_with_google_genai_multimodal(full_prompt, image_buffer, api_key)
    except Exception as new_sdk_error:
        try:
            return _call_with_legacy_google_generativeai_multimodal(full_prompt, image_buffer, api_key)
        except Exception as legacy_error:
            raise RuntimeError(
                "Gemini Vision API call failed for both SDKs. "
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

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

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

        # Phase 2B schema: {name/type, time_segments:[...]}
        if "time_segments" in effect:
            effect_name = effect.get("name") or effect.get("type")
            if not isinstance(effect_name, str) or not effect_name.strip():
                raise ValueError(f"effects[{i}] with time_segments must include 'name' or 'type'")

            normalized = _normalize_effect_type(effect_name)
            if normalized not in EXTRACTABLE_EFFECTS:
                raise ValueError(f"effects[{i}] effect '{effect_name}' is not supported for extraction")

            segments = effect["time_segments"]
            if not isinstance(segments, list) or not segments:
                raise ValueError(f"effects[{i}].time_segments must be a non-empty list")

            previous_end = None
            for s_idx, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    raise ValueError(f"effects[{i}].time_segments[{s_idx}] must be an object")
                if "start_time" not in segment or "end_time" not in segment:
                    raise ValueError(
                        f"effects[{i}].time_segments[{s_idx}] missing required field start_time/end_time"
                    )

                start_time = float(segment["start_time"])
                end_time = float(segment["end_time"])
                if start_time >= end_time:
                    raise ValueError(
                        f"effects[{i}].time_segments[{s_idx}] start_time must be < end_time"
                    )
                if previous_end is not None and start_time < previous_end:
                    raise ValueError(
                        f"effects[{i}].time_segments must be ordered and non-overlapping"
                    )
                previous_end = end_time

            continue

        # Legacy schema: {type, start_time, end_time, ...}
        for required in ["type", "start_time", "end_time"]:
            if required not in effect:
                raise ValueError(f"effects[{i}] missing required field '{required}'")

        normalized = _normalize_effect_type(str(effect["type"]))
        if normalized not in SUPPORTED_EFFECTS:
            raise ValueError(f"effects[{i}].type '{effect['type']}' is not supported")


def _build_retry_prompt(user_prompt: str, previous_response: str, error_message: str) -> str:
    """Build a stricter retry prompt when prior output was malformed."""
    preview = previous_response[:LLM_JSON_PREVIEW_CHARS]
    return (
        "Your previous output was invalid JSON and cannot be parsed.\n"
        f"Parse error: {error_message}\n"
        "Return ONLY one valid JSON object. No markdown. No explanation.\n"
        "JSON must include {\"effects\": [...]} and use ONE valid schema:\n"
        "(A) legacy per-effect format: type/start_time/end_time + parameters\n"
        "(B) phase-2B format: name(or type) + time_segments[{start_time,end_time,...}]\n"
        f"Previous invalid response preview:\n{preview}\n\n"
        f"Original request:\n{user_prompt}"
    )


def generate_effect_parameters(
    user_description: str,
    audio_duration: Optional[float] = None,
    reference_context: Optional[Dict] = None,
) -> Dict:
    """
    Main function to generate effect parameters from text description.

    Args:
        user_description: Natural language description of desired effects
        audio_duration: Optional audio duration for context
        reference_context: Optional context extracted from a reference audio file

    Returns:
        Dictionary containing effect sequence with time-variant parameters
    """
    generator = LLMPromptGenerator()
    full_prompt = generator.generate_prompt(
        user_description,
        audio_duration,
        reference_context=reference_context,
    )

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


def extract_reference_effects(
    spectrogram_image_buffer: BytesIO,
    audio_duration: Optional[float] = None,
) -> Dict:
    """
    Extract effect parameters from a reference audio spectrogram using VLM.

    This function uses multimodal Gemini Vision to analyze a spectrogram
    visualization and reverse-engineer the effects and time-varying parameters.

    Args:
        spectrogram_image_buffer: BytesIO object containing spectrogram PNG
        audio_duration: Optional reference audio duration for context

    Returns:
        Dictionary containing extracted effect segments with time-variant parameters

    Raises:
        ValueError: If extraction fails after max retries
    """
    generator = LLMVisionPromptGenerator()
    full_prompt = generator.generate_multimodal_prompt(
        user_description="Reverse-engineer the effects from this reference audio spectrogram.",
        audio_duration=audio_duration,
    )

    current_prompt = full_prompt
    last_error: Optional[str] = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        response = call_gemini_vision_api(current_prompt, spectrogram_image_buffer, system_prompt=generator.system_prompt)

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
        f"Failed to extract reference effects after {LLM_MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
