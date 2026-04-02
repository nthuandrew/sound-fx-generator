"""
Audio I/O Utilities

Functions for loading and saving audio files.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional


def load_audio(
    file_path: str,
    sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (None = native)
        mono: Convert to mono if True
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=mono)
        return audio, sample_rate
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {file_path}: {str(e)}")


def save_audio(
    audio: np.ndarray,
    file_path: str,
    sr: int = 16000,
    bit_depth: int = 16,
) -> None:
    """
    Save audio file using soundfile.
    
    Args:
        audio: Audio samples (numpy array)
        file_path: Output file path
        sr: Sample rate in Hz
        bit_depth: Bit depth (16 or 24)
        
    Raises:
        RuntimeError: If saving fails
    """
    try:
        # Ensure audio is in valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Determine subtype based on bit depth
        subtype = f"PCM_{bit_depth}"
        
        sf.write(file_path, audio, sr, subtype=subtype)
    except Exception as e:
        raise RuntimeError(f"Failed to save audio file {file_path}: {str(e)}")


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return len(audio) / sr
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {str(e)}")


def normalize_audio(audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio samples
        target_level: Target level in dB
        
    Returns:
        Normalized audio
    """
    # Measure current level
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Convert target from dB to linear
    target_linear = 10 ** (target_level / 20.0)
    
    # Scale
    gain = target_linear / rms
    normalized = audio * gain
    
    # Prevent clipping
    max_val = np.max(np.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val
    
    return normalized
