"""
Distortion Effect

Implements a waveshaping distortion effect.
"""

import numpy as np
from typing import Dict
from .base_effect import AudioEffect
import config


class DistortionEffect(AudioEffect):
    """
    Distortion effect using waveshaping with tone control.
    Can create anything from subtle warmth to aggressive overdrive.
    """
    
    def process(
        self,
        audio: np.ndarray,
        parameters: Dict[str, float],
    ) -> np.ndarray:
        """
        Apply distortion effect with time-varying parameters.
        
        Args:
            audio: Input audio samples
            parameters: Dict containing:
                - gain: Gain/drive amount (0.0-10.0)
                - tone: Output filtering 0.0=dark, 1.0=bright
                
        Returns:
            Processed audio with distortion
        """
        gain = parameters.get("gain", 1.0)
        tone = parameters.get("tone", 0.5)
        
        # Validate parameters
        gain = np.clip(gain, 0.0, 10.0)
        tone = np.clip(tone, 0.0, 1.0)
        
        audio_mono = self.ensure_mono(audio)
        
        # Apply stronger drive mapping so high gain settings sound clearly aggressive.
        intensity = config.EFFECT_INTENSITY_MULTIPLIER
        drive = 1.0 + gain * 2.2 * intensity
        driven = audio_mono * drive
        
        # Apply heavier waveshaping with tanh.
        # The extra factor increases saturation audibility.
        distorted = np.tanh(1.4 * driven)
        
        # Apply tone control via simple filter
        if tone < 1.0:
            # Low-pass filtering for darker tone
            filtered = self._apply_tone_control(distorted, tone)
        else:
            filtered = distorted
        
        # Keep mostly wet for obvious distortion character.
        dry_mix = 0.05
        wet_mix = 0.95
        blended = dry_mix * audio_mono + wet_mix * filtered

        # Mild output shaping only, avoid over-attenuating high gain settings.
        output = np.tanh(1.15 * blended)
        
        return np.clip(output, -1.0, 1.0)
    
    def _apply_tone_control(self, audio: np.ndarray, tone: float) -> np.ndarray:
        """
        Apply tone control via simple one-pole filter.
        
        Args:
            audio: Input audio
            tone: Tone parameter (0.0-1.0)
            
        Returns:
            Filtered audio
        """
        # Tone ranges from 0 (dark) to 1 (bright)
        # We'll use a simple one-pole low-pass filter
        # Cutoff frequency inversely related to tone
        
        # Map tone to filter coefficient
        # Lower tone -> lower cutoff -> more filtering
        feedback = 0.5 * (1.0 - tone)
        
        output = np.zeros_like(audio)
        last_sample = 0.0
        
        for i, sample in enumerate(audio):
            output[i] = sample * (1 - feedback) + last_sample * feedback
            last_sample = output[i]
        
        return output
