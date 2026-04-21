"""
Parameter Parser

This module is responsible for:
- Validating LLM-generated JSON parameters
- Parsing effect sequences into structured data
- Interpolating time-variant parameter envelopes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import PARAM_CONSTRAINTS


@dataclass
class TimeVaryingParameter:
    """Represents a parameter that changes over time."""
    name: str
    start_time: float
    end_time: float
    start_value: float
    end_value: float
    
    def interpolate(self, time: float) -> float:
        """
        Linear interpolation of parameter value at given time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Interpolated parameter value
        """
        if time <= self.start_time:
            return self.start_value
        if time >= self.end_time:
            return self.end_value
        
        # Linear interpolation
        progress = (time - self.start_time) / (self.end_time - self.start_time)
        return self.start_value + progress * (self.end_value - self.start_value)


@dataclass
class EffectInstance:
    """Represents a single effect instance with time-variant parameters."""
    effect_type: str
    start_time: float
    end_time: float
    parameters: Dict[str, TimeVaryingParameter]
    
    def get_parameters_at_time(self, time: float) -> Dict[str, float]:
        """
        Get all parameter values at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            Dictionary of parameter names to interpolated values
        """
        if time < self.start_time or time > self.end_time:
            return {}
        
        return {
            name: param.interpolate(time)
            for name, param in self.parameters.items()
        }


class ParameterParser:
    """
    Parses and validates LLM-generated effect parameters.
    """
    
    def __init__(self, constraints: Optional[Dict] = None):
        """
        Initialize the parser with parameter constraints.
        
        Args:
            constraints: Optional dictionary of parameter constraints
        """
        self.constraints = constraints or PARAM_CONSTRAINTS

    @staticmethod
    def _normalize_effect_type(raw_type: str) -> str:
        """Normalize effect names from model output to internal keys."""
        normalized = raw_type.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "lowpass": "low_pass_filter",
            "low_pass": "low_pass_filter",
            "lp_filter": "low_pass_filter",
            "reverb": "reverb",
            "chorus": "chorus",
            "distortion": "distortion",
            "delay": "delay",
        }
        return aliases.get(normalized, normalized)

    def _expand_time_segmented_effect(self, effect_dict: Dict) -> List[Dict]:
        """
        Convert phase-2B schema effect (`name` + `time_segments`) into legacy
        effect instances (`type`, `start_time`, `end_time`, param envelopes).
        """
        if "time_segments" not in effect_dict:
            return [effect_dict]

        effect_name = effect_dict.get("type") or effect_dict.get("name")
        if not isinstance(effect_name, str) or not effect_name.strip():
            raise ValueError("Time-segmented effect must include 'name' (or 'type')")

        segments = effect_dict.get("time_segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("time_segments must be a non-empty list")

        normalized_type = self._normalize_effect_type(effect_name)
        expanded: List[Dict] = []

        for idx, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise ValueError(f"time_segments[{idx}] must be an object")

            if "start_time" not in segment or "end_time" not in segment:
                raise ValueError(f"time_segments[{idx}] missing start_time/end_time")

            item = {
                "type": normalized_type,
                "start_time": float(segment["start_time"]),
                "end_time": float(segment["end_time"]),
            }

            for param_name, param_value in segment.items():
                if param_name in {"start_time", "end_time"}:
                    continue

                if isinstance(param_value, dict) and "start" in param_value and "end" in param_value:
                    item[param_name] = {
                        "start": float(param_value["start"]),
                        "end": float(param_value["end"]),
                    }
                elif isinstance(param_value, (int, float)):
                    value = float(param_value)
                    item[param_name] = {"start": value, "end": value}

            expanded.append(item)

        return expanded
    
    def validate_parameter_value(
        self,
        effect_type: str,
        param_name: str,
        value: float,
    ) -> bool:
        """
        Validate that a parameter value is within acceptable range.
        
        Args:
            effect_type: Type of effect (e.g., 'reverb')
            param_name: Parameter name (e.g., 'decay_time')
            value: Parameter value to validate
            
        Returns:
            True if valid, False otherwise
        """
        if effect_type not in self.constraints:
            return False
        
        if param_name not in self.constraints[effect_type]:
            return False
        
        min_val, max_val = self.constraints[effect_type][param_name]
        return min_val <= value <= max_val
    
    def clamp_parameter_value(
        self,
        effect_type: str,
        param_name: str,
        value: float,
    ) -> float:
        """
        Clamp parameter value to acceptable range, with warning.
        
        Args:
            effect_type: Type of effect
            param_name: Parameter name
            value: Original value
            
        Returns:
            Clamped value
        """
        if effect_type not in self.constraints or param_name not in self.constraints[effect_type]:
            return value
        
        min_val, max_val = self.constraints[effect_type][param_name]
        clamped = np.clip(value, min_val, max_val)
        
        if clamped != value:
            print(f"Warning: {effect_type}.{param_name} value {value} clamped to {clamped}")
        
        return clamped
    
    def parse_effect_instance(
        self,
        effect_dict: Dict,
        audio_duration: Optional[float] = None,
    ) -> EffectInstance:
        """
        Parse a single effect definition into an EffectInstance.
        
        Args:
            effect_dict: Dictionary containing effect parameters from LLM
            audio_duration: Optional max duration to clamp times
            
        Returns:
            EffectInstance object
            
        Raises:
            ValueError: If effect definition is invalid
        """
        # Validate required fields
        required_fields = ["type", "start_time", "end_time"]
        for field in required_fields:
            if field not in effect_dict:
                raise ValueError(f"Missing required field: {field}")
        
        effect_type = self._normalize_effect_type(str(effect_dict["type"]))
        start_time = float(effect_dict["start_time"])
        end_time = float(effect_dict["end_time"])
        
        # Validate time order
        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        # Validate effect support at parser level.
        if effect_type not in self.constraints:
            raise ValueError(f"Unsupported effect type for parsing: {effect_type}")
        
        # Clamp times if audio_duration provided
        if audio_duration:
            end_time = min(end_time, audio_duration)
            if start_time >= end_time:
                raise ValueError(f"Effect timing {start_time}-{end_time}s is invalid for audio duration {audio_duration}s")
        
        # Parse parameters
        time_varying_params = {}
        for param_name, param_value in effect_dict.items():
            if param_name in ["type", "start_time", "end_time"]:
                continue
            
            # Check if parameter is time-varying (dict with 'start' and 'end')
            if isinstance(param_value, dict) and "start" in param_value and "end" in param_value:
                start_val = float(param_value["start"])
                end_val = float(param_value["end"])
                
                # Clamp values
                start_val = self.clamp_parameter_value(effect_type, param_name, start_val)
                end_val = self.clamp_parameter_value(effect_type, param_name, end_val)
                
                time_varying_params[param_name] = TimeVaryingParameter(
                    name=param_name,
                    start_time=start_time,
                    end_time=end_time,
                    start_value=start_val,
                    end_value=end_val,
                )
            
            elif isinstance(param_value, (int, float)):
                # Convert constant value to time-varying (same start and end)
                value = float(param_value)
                value = self.clamp_parameter_value(effect_type, param_name, value)
                
                time_varying_params[param_name] = TimeVaryingParameter(
                    name=param_name,
                    start_time=start_time,
                    end_time=end_time,
                    start_value=value,
                    end_value=value,
                )
        
        return EffectInstance(
            effect_type=effect_type,
            start_time=start_time,
            end_time=end_time,
            parameters=time_varying_params,
        )
    
    def parse_parameters(
        self,
        llm_output: Dict,
        audio_duration: Optional[float] = None,
    ) -> List[EffectInstance]:
        """
        Parse complete LLM output into list of effect instances.
        
        Args:
            llm_output: Dictionary from LLM (should contain 'effects' key)
            audio_duration: Optional audio duration for validation
            
        Returns:
            List of parsed EffectInstance objects
            
        Raises:
            ValueError: If output structure is invalid
        """
        if "effects" not in llm_output:
            raise ValueError("LLM output missing 'effects' key")
        
        if not isinstance(llm_output["effects"], list):
            raise ValueError("'effects' must be a list")
        
        effect_instances = []
        for raw_effect in llm_output["effects"]:
            try:
                expanded = self._expand_time_segmented_effect(raw_effect)
                for effect_dict in expanded:
                    effect = self.parse_effect_instance(effect_dict, audio_duration)
                    effect_instances.append(effect)
            except ValueError as e:
                print(f"Warning: Failed to parse effect: {e}")
                continue
        
        return effect_instances
    
    def create_parameter_envelope(
        self,
        effect_instances: List[EffectInstance],
        audio_duration: float,
        sample_rate: int,
    ) -> Dict[int, Dict[str, float]]:
        """
        Create sample-level parameter envelopes for audio processing.
        
        Args:
            effect_instances: List of effect instances to process
            audio_duration: Total audio duration in seconds
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Dictionary mapping sample index to {effect_type: {param: value}}
        """
        num_samples = int(audio_duration * sample_rate)
        envelope = {}
        
        # Initialize envelope with empty dicts for each sample
        for i in range(num_samples):
            envelope[i] = {}
        
        # Fill in parameter values
        for effect in effect_instances:
            effect_key = effect.effect_type
            
            for i in range(num_samples):
                time = i / sample_rate
                
                if effect.start_time <= time <= effect.end_time:
                    if effect_key not in envelope[i]:
                        envelope[i][effect_key] = {}
                    
                    params = effect.get_parameters_at_time(time)
                    envelope[i][effect_key].update(params)
        
        return envelope
