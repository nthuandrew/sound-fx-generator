import unittest

from core.parameter_parser import ParameterParser


class TestParameterParserPhase2B(unittest.TestCase):
    def setUp(self):
        self.parser = ParameterParser()

    def test_parse_time_segment_schema_to_effect_instances(self):
        llm_output = {
            "effects": [
                {
                    "name": "Reverb",
                    "time_segments": [
                        {
                            "start_time": 0.0,
                            "end_time": 2.0,
                            "decay_time": 1.0,
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

        effects = self.parser.parse_parameters(llm_output, audio_duration=4.0)
        self.assertEqual(len(effects), 2)
        self.assertEqual(effects[0].effect_type, "reverb")
        self.assertEqual(effects[0].start_time, 0.0)
        self.assertEqual(effects[0].end_time, 2.0)
        self.assertAlmostEqual(effects[1].parameters["decay_time"].start_value, 2.0)

    def test_parse_time_segment_schema_unsupported_effect_is_skipped(self):
        llm_output = {
            "effects": [
                {
                    "name": "Delay",
                    "time_segments": [
                        {
                            "start_time": 0.0,
                            "end_time": 2.0,
                            "delay_time": 0.3,
                            "feedback": 0.4,
                            "wet_dry": 0.2,
                        }
                    ],
                }
            ]
        }

        effects = self.parser.parse_parameters(llm_output, audio_duration=2.0)
        self.assertEqual(effects, [])

    def test_parse_legacy_schema_still_works(self):
        llm_output = {
            "effects": [
                {
                    "type": "distortion",
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "gain": {"start": 1.0, "end": 3.0},
                    "tone": {"start": 0.5, "end": 0.4},
                }
            ]
        }

        effects = self.parser.parse_parameters(llm_output, audio_duration=3.0)
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects[0].effect_type, "distortion")
        self.assertAlmostEqual(effects[0].parameters["gain"].start_value, 1.0)
        self.assertAlmostEqual(effects[0].parameters["gain"].end_value, 3.0)


if __name__ == "__main__":
    unittest.main()
