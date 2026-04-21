import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.llm_prompt import LLMPromptGenerator
from utils.audio_io import save_audio
from utils.reference_audio import analyze_reference_audio, format_reference_context


class TestReferenceAudioPhase2(unittest.TestCase):
    def _write_tone(self, path: str, freq: float = 440.0, sr: int = 16000, duration: float = 2.0):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.25 * np.sin(2 * np.pi * freq * t)
        save_audio(audio, path, sr=sr)

    def test_analyze_reference_audio_returns_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "reference.wav")
            self._write_tone(ref_path, freq=880.0, duration=3.0)

            context = analyze_reference_audio(ref_path, sr=16000)
            context_dict = context.to_dict()

            self.assertGreater(context_dict["duration_seconds"], 0)
            self.assertIn("spectrogram_shape", context_dict)
            self.assertGreaterEqual(len(context_dict["segments"]), 1)

            formatted = format_reference_context(context)
            self.assertIn("Reference audio context", formatted)
            self.assertIn("coarse segments", formatted)

    def test_prompt_includes_reference_context(self):
        generator = LLMPromptGenerator()
        prompt = generator.generate_prompt(
            "Add a warm reverb",
            audio_duration=5.0,
            reference_context={
                "duration_seconds": 5.0,
                "sample_rate": 16000,
                "spectrogram_shape": [64, 100],
                "avg_rms_db": -18.2,
                "avg_centroid_hz": 1200.0,
                "avg_rolloff_hz": 2500.0,
                "onset_density": 0.2,
                "estimated_tempo_bpm": 120.0,
                "tempo_confidence": 0.7,
                "segments": [
                    {
                        "start_time": 0.0,
                        "end_time": 2.5,
                        "avg_energy_db": -20.0,
                        "avg_centroid_hz": 1000.0,
                        "avg_rolloff_hz": 2000.0,
                        "label": "warm / sparse / ambient",
                    }
                ],
            },
        )

        self.assertIn("Reference audio context", prompt)
        self.assertIn("warm / sparse / ambient", prompt)
        self.assertIn("The input audio has a duration of 5.0 seconds", prompt)

    def test_audio_processor_receives_reference_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.wav")
            ref_path = os.path.join(tmpdir, "reference.wav")
            output_path = os.path.join(tmpdir, "output.wav")
            self._write_tone(input_path, freq=440.0, duration=2.0)
            self._write_tone(ref_path, freq=880.0, duration=2.0)

            captured_kwargs = {}

            def fake_generate_effect_parameters(*args, **kwargs):
                captured_kwargs.update(kwargs)
                return {"effects": []}

            with patch("core.audio_processor.generate_effect_parameters", side_effect=fake_generate_effect_parameters):
                from core.audio_processor import AudioProcessor

                processor = AudioProcessor(sample_rate=16000)
                output_audio, info = processor.process(
                    input_path,
                    "Make it brighter",
                    reference_audio_file=ref_path,
                    output_file=output_path,
                    verbose=False,
                )

            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(len(output_audio.shape), 1)
            self.assertIn("reference_context", captured_kwargs)
            self.assertIsNotNone(captured_kwargs["reference_context"])
            self.assertIn("reference_audio_file", info)
            self.assertIn("reference_context", info)


if __name__ == "__main__":
    unittest.main()
