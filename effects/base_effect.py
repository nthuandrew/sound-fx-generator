"""
Base Effect Class

Defines the interface and utility functions for all audio effects.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict


class AudioEffect(ABC):
    """
    Abstract base class for audio effects.
    All concrete effects must inherit from this class.
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the effect.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
    
    @abstractmethod
    def process(
        self,
        audio: np.ndarray,
        parameters: Dict[str, float],
    ) -> np.ndarray:
        """
        Process audio with given parameters.
        
        Args:
            audio: Audio samples (numpy array)
            parameters: Effect parameters as dictionary
            
        Returns:
            Processed audio samples
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, float]) -> bool:
        """
        Validate that parameters are within acceptable range.
        Can be overridden by subclasses for custom validation.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            True if valid
        """
        return True
    
    @staticmethod
    def ensure_mono(audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mono if stereo.
        
        Args:
            audio: Audio samples
            
        Returns:
            Mono audio
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        return audio
    
    @staticmethod
    def ensure_stereo(audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to stereo if mono.
        
        Args:
            audio: Audio samples
            
        Returns:
            Stereo audio
        """
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        return audio
    
    def smooth_clicks(self, audio: np.ndarray, crossfade_samples: int = 512) -> np.ndarray:
        """
        Apply crossfading to reduce clicking artifacts at parameter changes.
        
        Args:
            audio: Audio samples
            crossfade_samples: Number of samples for crossfade window
            
        Returns:
            Audio with reduced clicks
        """
        if crossfade_samples == 0 or len(audio) < crossfade_samples:
            return audio
        
        # Apply gentle envelope to reduce extreme clicks
        window = np.hanning(crossfade_samples * 2)
        fade_in = window[:crossfade_samples]
        fade_out = window[crossfade_samples:]
        
        result = audio.copy()
        
        # Fade in at start
        result[:crossfade_samples] *= fade_in
        
        # Fade out at end
        result[-crossfade_samples:] *= fade_out
        
        return result
