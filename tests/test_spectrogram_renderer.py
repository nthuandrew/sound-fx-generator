"""
Unit tests for spectrogram visualization module (Phase 2A).

Tests the spectrogram rendering functionality required for VLM-based
effect reverse-engineering.
"""

import unittest
from io import BytesIO
from unittest.mock import patch
import numpy as np
from utils.spectrogram_renderer import (
    generate_reference_spectrogram_image,
    encode_spectrogram_to_base64,
    get_spectrogram_image_bytes,
)


class TestSpectrogramRenderer(unittest.TestCase):
    """Test cases for spectrogram visualization."""

    def test_generate_spectrogram_from_reference_audio(self):
        """Test that spectrogram is generated with correct dimensions."""
        # Use reference.wav which should exist in the test environment
        try:
            mag, img_buf = generate_reference_spectrogram_image("reference.wav")
        except FileNotFoundError:
            self.skipTest("reference.wav not found; skipping test")

        # Verify spectrogram shape (n_fft//2 + 1 frequency bins)
        self.assertEqual(mag.shape[0], 1025)  # 2048//2 + 1
        self.assertGreater(mag.shape[1], 0)  # Should have time frames

        # Verify image buffer is valid
        self.assertIsInstance(img_buf, BytesIO)
        self.assertGreater(img_buf.getbuffer().nbytes, 0)

        # Verify magnitude values are positive (linear amplitude)
        self.assertGreaterEqual(mag.min(), 0.0)
        self.assertGreater(mag.max(), 0.0)

    def test_spectrogram_linear_amplitude(self):
        """Test that spectrogram uses linear amplitude, not dB."""
        try:
            mag, _ = generate_reference_spectrogram_image("reference.wav")
        except FileNotFoundError:
            self.skipTest("reference.wav not found")

        # Linear amplitude values should span a wide range (not compressed like dB)
        # We expect some very small values and some large values
        self.assertLess(mag.min(), 1.0)  # Should have small values
        self.assertGreater(mag.max(), 10.0)  # Should have large values

    def test_encode_spectrogram_to_base64(self):
        """Test base64 encoding of spectrogram image."""
        try:
            _, img_buf = generate_reference_spectrogram_image("reference.wav")
        except FileNotFoundError:
            self.skipTest("reference.wav not found")

        b64_str = encode_spectrogram_to_base64(img_buf)

        # Verify it's a valid base64 string
        self.assertIsInstance(b64_str, str)
        self.assertGreater(len(b64_str), 0)

        # Should not contain spaces or newlines (valid base64)
        self.assertNotIn(" ", b64_str)
        self.assertNotIn("\n", b64_str)

    def test_get_spectrogram_image_bytes(self):
        """Test retrieving raw bytes from spectrogram image."""
        try:
            _, img_buf = generate_reference_spectrogram_image("reference.wav")
        except FileNotFoundError:
            self.skipTest("reference.wav not found")

        img_bytes = get_spectrogram_image_bytes(img_buf)

        # Verify it's valid binary data
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 0)

        # PNG files start with specific magic bytes
        self.assertEqual(img_bytes[:8], b"\x89PNG\r\n\x1a\n")

    def test_spectrogram_image_format(self):
        """Test that generated image is in PNG format."""
        try:
            _, img_buf = generate_reference_spectrogram_image("reference.wav", output_format="png")
        except FileNotFoundError:
            self.skipTest("reference.wav not found")

        img_bytes = get_spectrogram_image_bytes(img_buf)

        # Verify PNG signature
        self.assertTrue(img_bytes.startswith(b"\x89PNG"))

    def test_spectrogram_with_custom_parameters(self):
        """Test spectrogram generation with custom STFT parameters."""
        try:
            mag1, _ = generate_reference_spectrogram_image(
                "reference.wav", n_fft=2048, hop_length=512
            )
            mag2, _ = generate_reference_spectrogram_image(
                "reference.wav", n_fft=1024, hop_length=256
            )
        except FileNotFoundError:
            self.skipTest("reference.wav not found")

        # Different FFT sizes should produce different frequency resolutions
        self.assertEqual(mag1.shape[0], 1025)  # 2048//2 + 1
        self.assertEqual(mag2.shape[0], 513)  # 1024//2 + 1

        # Time dimension should be different due to different hop lengths
        self.assertNotEqual(mag1.shape[1], mag2.shape[1])

    def test_spectrogram_reproducibility(self):
        """Test that spectrogram generation is reproducible."""
        try:
            mag1, _ = generate_reference_spectrogram_image("reference.wav")
            mag2, _ = generate_reference_spectrogram_image("reference.wav")
        except FileNotFoundError:
            self.skipTest("reference.wav not found")

        # Should produce identical results
        np.testing.assert_array_equal(mag1, mag2)

    @patch("utils.spectrogram_renderer.librosa.display.specshow")
    def test_spectrogram_uses_linear_axes_and_viridis(self, mock_specshow):
        """Ensure renderer calls specshow with linear frequency + viridis colormap."""
        try:
            generate_reference_spectrogram_image("reference.wav")
        except FileNotFoundError:
            self.skipTest("reference.wav not found")

        self.assertTrue(mock_specshow.called)
        _, kwargs = mock_specshow.call_args
        self.assertEqual(kwargs.get("y_axis"), "linear")
        self.assertEqual(kwargs.get("x_axis"), "time")
        self.assertEqual(kwargs.get("cmap"), "viridis")


if __name__ == "__main__":
    unittest.main()
