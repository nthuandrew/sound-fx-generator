import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.audio_processor import AudioProcessor
from main import process_single_file
from utils.audio_io import save_audio


class TestPhase2CExtractAndClone(unittest.TestCase):
    def _write_tone(self, path: str, freq: float = 440.0, sr: int = 16000, duration: float = 2.0):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.25 * np.sin(2 * np.pi * freq * t)
        save_audio(audio, path, sr=sr)

    def test_audio_processor_extract_and_clone_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            ref_path = os.path.join(tmpdir, "reference.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            self._write_tone(input_path, freq=330.0)
            self._write_tone(ref_path, freq=660.0)

            extracted = {
                "effects": [
                    {
                        "name": "Reverb",
                        "time_segments": [
                            {
                                "start_time": 0.0,
                                "end_time": 2.0,
                                "decay_time": 1.2,
                                "wet_dry": 0.35,
                                "width": 0.9,
                            }
                        ],
                    }
                ]
            }

            with patch.object(AudioProcessor, "extract_reference_effects", return_value=extracted):
                processor = AudioProcessor(sample_rate=16000)
                output_audio, info = processor.process(
                    input_path,
                    text_prompt="",
                    reference_audio_file=ref_path,
                    output_file=output_path,
                    verbose=False,
                    mode="extract_and_clone",
                )

            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(info["mode"], "extract_and_clone")
            self.assertIn("extracted_effects", info)
            self.assertGreaterEqual(len(info["extracted_effects"]), 1)
            self.assertEqual(output_audio.ndim, 1)

    def test_cli_extract_and_clone_requires_reference_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            self._write_tone(input_path)

            args = SimpleNamespace(
                mode="extract-and-clone",
                audio=input_path,
                prompt=None,
                output=None,
                reference_audio=None,
                sr=16000,
                no_normalize=False,
                verbose=False,
                quiet=True,
            )

            ok = process_single_file(args)
            self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
