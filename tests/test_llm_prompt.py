import unittest
from io import BytesIO
from unittest.mock import patch

from core.llm_prompt import (
    extract_json_from_response,
    extract_reference_effects,
    generate_effect_parameters,
    validate_effect_parameters,
)


class TestLLMPromptRobustness(unittest.TestCase):
    def test_extract_json_with_markdown_wrapper(self):
        text = """```json
        {
          \"effects\": [
            {
              \"type\": \"distortion\",
              \"start_time\": 0.0,
              \"end_time\": 1.0,
              \"gain\": {\"start\": 2.0, \"end\": 5.0},
              \"tone\": {\"start\": 0.5, \"end\": 0.5}
            }
          ]
        }
        ```"""
        parsed = extract_json_from_response(text)
        self.assertIn("effects", parsed)
        self.assertEqual(parsed["effects"][0]["type"], "distortion")

    def test_validate_effect_parameters_rejects_missing_effects(self):
        with self.assertRaises(ValueError):
            validate_effect_parameters({"foo": 1})

    def test_validate_effect_parameters_accepts_time_segments_schema(self):
        payload = {
            "effects": [
                {
                    "name": "Reverb",
                    "time_segments": [
                        {
                            "start_time": 0.0,
                            "end_time": 2.0,
                            "decay_time": 1.2,
                            "wet_dry": 0.3,
                            "width": 0.9,
                        },
                        {
                            "start_time": 2.0,
                            "end_time": 4.0,
                            "decay_time": 2.0,
                            "wet_dry": 0.5,
                            "width": 0.95,
                        },
                    ],
                }
            ]
        }
        validate_effect_parameters(payload)

    def test_validate_effect_parameters_rejects_unordered_time_segments(self):
        payload = {
            "effects": [
                {
                    "name": "Distortion",
                    "time_segments": [
                        {"start_time": 2.0, "end_time": 3.0, "gain": 2.0, "tone": 0.4},
                        {"start_time": 1.0, "end_time": 2.0, "gain": 1.5, "tone": 0.5},
                    ],
                }
            ]
        }

        with self.assertRaises(ValueError):
            validate_effect_parameters(payload)

    @patch("core.llm_prompt.call_gemini_api")
    def test_generate_effect_parameters_retries_after_invalid_json(self, mock_call):
        # First response is malformed/truncated JSON, second is valid.
        mock_call.side_effect = [
            '{"effects": [{"type": "distortion", "start_time": 0.0, "end_time": 2.0,',
            '{"effects": [{"type": "distortion", "start_time": 0.0, "end_time": 2.0, "gain": {"start": 1.0, "end": 3.0}, "tone": {"start": 0.5, "end": 0.5}}]}'
        ]

        result = generate_effect_parameters("Add distortion")
        self.assertIn("effects", result)
        self.assertEqual(len(result["effects"]), 1)
        self.assertEqual(mock_call.call_count, 2)

    @patch("core.llm_prompt.call_gemini_vision_api")
    def test_extract_reference_effects_with_mocked_multimodal_call(self, mock_vision_call):
        mock_vision_call.return_value = (
            '{"effects": [{"name": "Reverb", "time_segments": [{"start_time": 0.0, '
            '"end_time": 2.0, "decay_time": 1.1, "wet_dry": 0.35, "width": 0.9}]}], '
            '"analysis_notes": "mocked"}'
        )

        fake_png = BytesIO(b"\x89PNG\r\n\x1a\nmock")
        result = extract_reference_effects(fake_png, audio_duration=2.0)

        self.assertIn("effects", result)
        self.assertEqual(result["effects"][0]["name"], "Reverb")
        self.assertEqual(mock_vision_call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
