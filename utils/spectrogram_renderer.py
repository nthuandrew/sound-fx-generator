"""
Spectrogram Visualization for VLM Input

This module generates linear frequency/amplitude spectrograms suitable for
multimodal LLM analysis. Follows ST-ITO ablation study specifications:
- Linear frequency scale (not mel)
- Linear amplitude (not dB)
- Viridis colormap
- No colorbar (prevents visual interference)
- Preserved X (time) and Y (frequency) axes
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend to avoid display issues
matplotlib.use("Agg")


def generate_reference_spectrogram_image(
    file_path: str,
    sr: Optional[int] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    output_format: str = "png",
) -> tuple[np.ndarray, BytesIO]:
    """
    Generate a linear spectrogram image from reference audio.

    This function creates a spectrogram visualization suitable for multimodal
    VLM analysis, following ST-ITO ablation study specifications:
    - Linear frequency scale (Y-axis in Hz)
    - Linear magnitude amplitude (not dB)
    - Viridis colormap for perceptual uniformity
    - No colorbar to avoid visual interference
    - Preserved and labeled X (time) and Y (frequency) axes

    Args:
        file_path: Path to reference audio file
        sr: Target sample rate (None keeps native)
        n_fft: FFT window size for STFT
        hop_length: Hop length for STFT frames
        output_format: Image format ('png', 'jpg', 'webp')

    Returns:
        Tuple of:
        - spectrogram_data: 2D numpy array of magnitude values (linear amplitude)
        - image_buffer: BytesIO object containing encoded image (PNG in memory)

    Raises:
        ValueError: If audio file is empty or cannot be loaded
        IOError: If image encoding fails
    """
    # Load audio
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    if len(audio) == 0:
        raise ValueError(f"Reference audio is empty: {file_path}")

    duration_seconds = len(audio) / sample_rate

    # Compute STFT (linear frequency output)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)  # Linear amplitude (not dB)

    # Prepare figure with strict dimensions for consistent VLM input
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100, tight_layout=True)

    # Display spectrogram with linear frequency scale
    # librosa.display.specshow expects linear_amplitude for linear color scale
    img = librosa.display.specshow(
        magnitude,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        ax=ax,
        cmap="viridis",
    )

    # Set axis labels
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency (Hz)", fontsize=12, fontweight="bold")
    ax.set_title(f"Reference Audio Spectrogram: {file_path}", fontsize=14, fontweight="bold")

    # Set Y-axis limits to audible range
    ax.set_ylim([0, sample_rate / 2])

    # ** CRITICAL: No colorbar (prevents visual interference with VLM) **
    # img.colorbar would be added here if needed, but we explicitly omit it

    # Render to BytesIO buffer
    image_buffer = BytesIO()
    fig.savefig(image_buffer, format=output_format, bbox_inches="tight", dpi=100, pad_inches=0.1)
    image_buffer.seek(0)
    plt.close(fig)

    return magnitude, image_buffer


def encode_spectrogram_to_base64(spectrogram_image_buffer: BytesIO) -> str:
    """
    Encode spectrogram image buffer to base64 string for API transmission.

    Args:
        spectrogram_image_buffer: BytesIO object containing PNG image data

    Returns:
        Base64-encoded string ready for multimodal API input
    """
    import base64

    spectrogram_image_buffer.seek(0)
    image_data = spectrogram_image_buffer.read()
    return base64.b64encode(image_data).decode("utf-8")


def get_spectrogram_image_bytes(spectrogram_image_buffer: BytesIO) -> bytes:
    """
    Get raw bytes from spectrogram image buffer.

    Args:
        spectrogram_image_buffer: BytesIO object containing image data

    Returns:
        Raw bytes of image
    """
    spectrogram_image_buffer.seek(0)
    return spectrogram_image_buffer.read()
