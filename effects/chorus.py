"""
Chorus Effect

Implements a chorus effect using modulated delay lines.
"""

import numpy as np
from typing import Dict
from .base_effect import AudioEffect


class ChorusEffect(AudioEffect):
    """
    Chorus effect that creates a thickening, modulated sound.
    Uses multiple delay lines with modulated delays to create movement.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize chorus effect.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        super().__init__(sample_rate)
        self.num_taps = 3
        self.max_delay_samples = int(0.05 * sample_rate)  # 50ms max delay
    
    def process(
        self,
        audio: np.ndarray,
        parameters: Dict[str, float],
    ) -> np.ndarray:
        """
        Apply chorus effect with time-varying parameters.
        
        Args:
            audio: Input audio samples
            parameters: Dict containing:
                - rate: Modulation rate in Hz (0.1-10.0)
                - depth: Modulation depth 0.0-1.0
                - wet_dry: Wet/dry mix (0.0=dry, 1.0=wet)
                
        Returns:
            Processed audio with chorus effect
        """
        rate = parameters.get("rate", 1.0)
        depth = parameters.get("depth", 0.5)
        wet_dry = parameters.get("wet_dry", 0.5)
        
        # Validate parameters
        rate = np.clip(rate, 0.1, 10.0)
        depth = np.clip(depth, 0.0, 1.0)
        wet_dry = np.clip(wet_dry, 0.0, 1.0)
        
        audio_mono = self.ensure_mono(audio)
        wet_signal = np.zeros_like(audio_mono)
        
        # Process multiple taps
        for tap in range(self.num_taps):
            # Calculate base delay for this tap
            base_delay = (self.max_delay_samples * (tap + 1)) / (self.num_taps + 1)
            
            # Create modulation LFO (Low Frequency Oscillator)
            phase_offset = (2 * np.pi * tap) / self.num_taps
            time_array = np.arange(len(audio_mono)) / self.sample_rate
            lfo = np.sin(2 * np.pi * rate * time_array + phase_offset)
            
            # Modulate delay time
            modulated_delay = base_delay * (1.0 + depth * lfo)
            
            # Apply variable delay
            delayed = self._apply_variable_delay(audio_mono, modulated_delay)
            
            # Add to wet signal
            wet_signal += delayed / self.num_taps
        
        # Mix wet and dry
        output = (1.0 - wet_dry) * audio_mono + wet_dry * wet_signal
        
        return np.clip(output, -1.0, 1.0)
    
    def _apply_variable_delay(
        self,
        audio: np.ndarray,
        delay_samples: np.ndarray,
    ) -> np.ndarray:
        """
        Apply time-varying delay using linear interpolation.
        
        Args:
            audio: Input audio
            delay_samples: Delay in samples (can be array for time-varying)
            
        Returns:
            Delayed audio
        """
        if isinstance(delay_samples, (int, float)):
            delay_samples = np.full(len(audio), delay_samples)
        
        output = np.zeros_like(audio)
        padded_audio = np.pad(audio, (int(np.max(delay_samples)) + 1, 0), mode='constant')
        
        for i in range(len(audio)):
            delay = delay_samples[i]
            
            # Linear interpolation between samples
            integer_delay = int(np.floor(delay))
            fractional_delay = delay - integer_delay
            
            read_pos_1 = i + 1 + integer_delay
            read_pos_2 = read_pos_1 + 1
            
            if read_pos_2 < len(padded_audio):
                sample_1 = padded_audio[read_pos_1]
                sample_2 = padded_audio[read_pos_2]
                output[i] = sample_1 * (1 - fractional_delay) + sample_2 * fractional_delay
            elif read_pos_1 < len(padded_audio):
                output[i] = padded_audio[read_pos_1]
        
        return output
