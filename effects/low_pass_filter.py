"""
Low-Pass Filter Effect

Implements a low-pass filter with resonance control.
"""

import numpy as np
from typing import Dict
from scipy import signal
from .base_effect import AudioEffect
import config


class LowPassFilterEffect(AudioEffect):
    """
    Low-pass filter effect with variable cutoff frequency and resonance.
    Useful for sweeping filters and creating dynamic EQ effects.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize low-pass filter effect.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        super().__init__(sample_rate)
        self.nyquist = sample_rate / 2.0
    
    def process(
        self,
        audio: np.ndarray,
        parameters: Dict[str, float],
    ) -> np.ndarray:
        """
        Apply low-pass filter with time-varying parameters.
        
        Args:
            audio: Input audio samples
            parameters: Dict containing:
                - cutoff_freq: Cutoff frequency in Hz (20-20000)
                - resonance: Resonance/Q factor (0.0-1.0)
                
        Returns:
            Filtered audio
        """
        cutoff_freq = parameters.get("cutoff_freq", 5000.0)
        resonance = parameters.get("resonance", 0.5)
        
        # Validate parameters
        cutoff_freq = np.clip(cutoff_freq, 20.0, self.nyquist * 0.95)
        resonance = np.clip(resonance, 0.0, 1.0)
        
        audio_mono = self.ensure_mono(audio)
        
        # Design filter
        # Normalize cutoff frequency
        normalized_cutoff = cutoff_freq / self.nyquist
        intensity = config.EFFECT_INTENSITY_MULTIPLIER

        # Push cutoff lower for stronger audible filtering when intensity is high.
        effective_cutoff = np.clip(
            normalized_cutoff / (1.0 + 0.6 * (intensity - 1.0)),
            0.01,
            0.95,
        )
        
        # Map resonance to Q factor (1 to 10)
        Q = 1.0 + resonance * 9.0
        
        # Create Butterworth filter coefficients (steeper slope for clearer effect)
        try:
            output = self._apply_butterworth_filter(audio_mono, effective_cutoff, order=4)
        except Exception:
            # Fallback to simple low-pass if filter design fails
            output = self._apply_one_pole_filter(audio_mono, effective_cutoff, Q)

        # Wet-heavy blend to ensure the low-pass character is obvious.
        wet_mix = np.clip(0.75 + 0.15 * (intensity - 1.0), 0.7, 0.95)
        output = (1.0 - wet_mix) * audio_mono + wet_mix * output
        
        return np.clip(output, -1.0, 1.0)
    
    def _apply_one_pole_filter(
        self,
        audio: np.ndarray,
        normalized_cutoff: float,
        Q: float = 1.0,
    ) -> np.ndarray:
        """
        Apply simple one-pole low-pass filter.
        
        Args:
            audio: Input audio
            normalized_cutoff: Normalized cutoff frequency (0-1)
            Q: Quality factor for resonance
            
        Returns:
            Filtered audio
        """
        # Calculate filter coefficient from cutoff frequency
        # Coefficient ranges from 0 (sharp cutoff) to 1 (no filtering)
        coefficient = 1.0 - normalized_cutoff
        
        output = np.zeros_like(audio)
        last_sample = 0.0
        last_output = 0.0
        
        for i, sample in enumerate(audio):
            # Simple one-pole filter
            filtered = sample * (1 - coefficient) + last_output * coefficient
            
            # Add resonance by re-filtering
            if Q > 1.0:
                # Extra feedback for resonance
                resonance_gain = (Q - 1.0) * 0.1
                filtered += (filtered - last_sample) * resonance_gain
            
            output[i] = filtered
            last_output = filtered
            last_sample = sample
        
        return output
    
    def _apply_butterworth_filter(
        self,
        audio: np.ndarray,
        normalized_cutoff: float,
        order: int = 2,
    ) -> np.ndarray:
        """
        Apply Butterworth low-pass filter for high-quality filtering.
        
        Args:
            audio: Input audio
            normalized_cutoff: Normalized cutoff frequency (0-1)
            order: Butterworth order
            
        Returns:
            Filtered audio
        """
        try:
            # Design Butterworth filter
            b, a = signal.butter(order, normalized_cutoff, btype='low')
            
            # Apply filter
            filtered = signal.filtfilt(b, a, audio)
            
            return filtered
        except Exception as e:
            print(f"Warning: Butterworth filter failed: {str(e)}. Using one-pole filter.")
            return self._apply_one_pole_filter(audio, normalized_cutoff)
