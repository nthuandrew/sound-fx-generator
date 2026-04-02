"""
Reverb Effect

Implements a simple reverb effect using delay lines and feedback.
"""

import numpy as np
from typing import Dict
from .base_effect import AudioEffect


class ReverbEffect(AudioEffect):
    """
    Reverb effect using delay-based processing.
    Creates spacious, echoing sound by combining delayed signals.
    """
    
    # Pre-defined delay times (in milliseconds) for different channels
    DELAY_TIMES = [50, 117, 233, 419]
    
    def __init__(self, sample_rate: int = 16000, max_decay_time: float = 5.0):
        """
        Initialize reverb effect.
        
        Args:
            sample_rate: Audio sample rate in Hz
            max_decay_time: Maximum decay time in seconds
        """
        super().__init__(sample_rate)
        self.max_decay_time = max_decay_time
        
        # Pre-allocate delay buffers
        self.max_delay_samples = int(max_decay_time * sample_rate)
        self.delay_buffers = [
            np.zeros(self.max_delay_samples)
            for _ in self.DELAY_TIMES
        ]
        self.delay_positions = [0] * len(self.DELAY_TIMES)
    
    def process(
        self,
        audio: np.ndarray,
        parameters: Dict[str, float],
    ) -> np.ndarray:
        """
        Apply reverb effect with time-varying parameters.
        
        Args:
            audio: Input audio samples
            parameters: Dict containing:
                - decay_time: Reverb decay time in seconds (0.1-5.0)
                - wet_dry: Wet/dry mix (0.0=dry, 1.0=wet)
                - width: Stereo width (0.0-1.0)
                
        Returns:
            Processed audio with reverb
        """
        decay_time = parameters.get("decay_time", 1.0)
        wet_dry = parameters.get("wet_dry", 0.3)
        width = parameters.get("width", 1.0)
        
        # Ensure valid parameters
        decay_time = np.clip(decay_time, 0.1, self.max_decay_time)
        wet_dry = np.clip(wet_dry, 0.0, 1.0)
        width = np.clip(width, 0.0, 1.0)
        
        # Convert to mono for processing
        audio_mono = self.ensure_mono(audio)
        
        # Calculate feedback coefficient
        feedback = 0.5 * (decay_time / self.max_decay_time)
        
        # Process through delay lines
        output = np.zeros_like(audio_mono)
        wet_signal = np.zeros_like(audio_mono)
        
        for ch_idx, delay_ms in enumerate(self.DELAY_TIMES):
            delay_samples = int((delay_ms / 1000.0) * self.sample_rate)
            delay_samples = max(1, min(delay_samples, self.max_delay_samples - 1))
            
            wet = self._process_delay_line(
                audio_mono,
                delay_samples,
                feedback,
                ch_idx,
            )
            
            # Apply phase shifting for stereo width
            if ch_idx % 2 == 1:
                wet = wet * width
            
            wet_signal += wet
        
        # Mix wet and dry signals
        wet_signal = wet_signal / len(self.DELAY_TIMES)
        output = (1.0 - wet_dry) * audio_mono + wet_dry * wet_signal
        
        # Prevent clipping
        output = np.clip(output, -1.0, 1.0)
        
        return output
    
    def _process_delay_line(
        self,
        audio: np.ndarray,
        delay_samples: int,
        feedback: float,
        buffer_idx: int,
    ) -> np.ndarray:
        """
        Process audio through a delay line with feedback.
        
        Args:
            audio: Input audio
            delay_samples: Number of samples delay
            feedback: Feedback coefficient
            buffer_idx: Which delay buffer to use
            
        Returns:
            Delayed audio output
        """
        output = np.zeros_like(audio)
        buffer = self.delay_buffers[buffer_idx]
        pos = self.delay_positions[buffer_idx]
        
        for i, sample in enumerate(audio):
            # Read delayed sample
            read_pos = (pos - delay_samples) % len(buffer)
            delayed_sample = buffer[read_pos]
            
            # Write new sample with feedback
            buffer[pos] = sample + feedback * delayed_sample
            
            output[i] = delayed_sample
            
            # Update position
            pos = (pos + 1) % len(buffer)
        
        # Update position for next call
        self.delay_positions[buffer_idx] = pos
        
        return output
