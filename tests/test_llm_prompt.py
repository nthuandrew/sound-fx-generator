import unittest
from unittest.mock import patch

from core.llm_prompt import (
    extract_json_from_response,
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


if __name__ == "__main__":
    unittest.main()
