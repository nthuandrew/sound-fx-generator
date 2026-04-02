"""
LLM Prompt Engineering and API Integration

This module handles:
- Constructing structured prompts for the LLM
- Making API calls to Gemini (or other LLMs)
- Parsing and validating LLM responses
"""

import json
import os
from typing import Dict, List, Optional
import google.generativeai as genai
from config import LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


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
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(LLM_MODEL)
    
    # Prepare messages with system prompt
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"
    
    try:
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            ),
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {str(e)}")


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
    # Remove markdown code block markers if present
    cleaned = response_text
    if cleaned.strip().startswith("```json"):
        cleaned = cleaned.strip()[7:]  # Remove ```json
    if cleaned.strip().startswith("```"):
        cleaned = cleaned.strip()[3:]  # Remove ```
    if cleaned.strip().endswith("```"):
        cleaned = cleaned.strip()[:-3]  # Remove trailing ```
    
    # First try direct JSON parsing
    try:
        return json.loads(cleaned.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in the response
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}") + 1
    
    if start_idx == -1 or end_idx == 0:
        raise ValueError(f"No JSON object found in response: {cleaned[:100]}")
    
    json_str = cleaned[start_idx:end_idx]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to repair by ensuring proper closing brackets
        depth = 0
        repaired = ""
        for char in json_str:
            if char == "{" or char == "[":
                depth += 1
            elif char == "}" or char == "]":
                depth -= 1
            repaired += char
        
        # Close any unclosed brackets
        while depth > 0:
            repaired += "}"
            depth -= 1
        
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON: {str(e)}\nJSON string: {json_str[:200]}")


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
    # Generate prompt
    generator = LLMPromptGenerator()
    full_prompt = generator.generate_prompt(user_description, audio_duration)
    
    # Call LLM API
    response = call_gemini_api(full_prompt, system_prompt=generator.system_prompt)
    
    # Parse and validate response
    parameters = extract_json_from_response(response)
    
    return parameters
