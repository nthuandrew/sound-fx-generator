"""
Utility module for audio I/O, evaluation metrics, and spectrogram rendering.
"""

from .audio_io import load_audio, save_audio
from .evaluation import compute_clapscore, compute_mse, compute_pearson_correlation
from .reference_audio import analyze_reference_audio, format_reference_context
from .spectrogram_renderer import (
    generate_reference_spectrogram_image,
    encode_spectrogram_to_base64,
    get_spectrogram_image_bytes,
)

__all__ = [
    "load_audio",
    "save_audio",
    "compute_clapscore",
    "compute_mse",
    "compute_pearson_correlation",
    "analyze_reference_audio",
    "format_reference_context",
    "generate_reference_spectrogram_image",
    "encode_spectrogram_to_base64",
    "get_spectrogram_image_bytes",
]
