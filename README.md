# Language-Driven Time-Variant Sound Effect Generator

This project implements a language-driven, time-variant sound effect generator. The system uses an LLM to bridge the gap between natural language descriptions and low-level digital signal processing parameters.

 ## System Architecture

```
Text Prompt
    ↓
[LLM Prompt Engineering] 
    ↓ (LLM API Call)
JSON Parameter Sequence
    ↓
[Parameter Parser] (Validation & Parsing)
    ↓
Time-Varying Parameter Envelopes
    ↓
[Audio Processor + Effects]
    ↓
Output Audio + Evaluation Metrics
```

## Features

- **Natural Language Interface**: Users describe desired effects in plain English
- **Time-Variant Effects**: Support for dynamic parameter changes over time
- **Multiple Effects**: Reverb, Chorus, Distortion, Low-pass Filter
- **Evaluation Metrics**: Energy analysis, Dynamic range, Peak levels, Spectral changes

## Project Structure

```
sound-fx-generator/
├── core/                    # Core processing modules
│   ├── llm_prompt.py       # LLM prompt engineering and API calls
│   ├── parameter_parser.py # JSON parsing and validation
│   └── audio_processor.py  # Main audio processing engine
├── effects/                 # Audio effect implementations
│   ├── base_effect.py      # Base effect class
│   ├── reverb.py
│   ├── chorus.py
│   ├── distortion.py
│   └── low_pass_filter.py
├── utils/                   # Utilities
│   ├── audio_io.py         # Audio loading/saving
│   └── evaluation.py       # Evaluation metrics
├── output/                   # Processed audio outputs
├── main.py                 # CLI entry point
├── config.py               # Global configuration
└── input.wav                 # Example input audio
```

## Basic Usage

### Command Line (Single File)

```bash
python main.py --audio input.wav --prompt "Apply a super strong distortion effect" --output output/output.wav```
```

### Python API

```python
from core.audio_processor import AudioProcessor

# Create processor
processor = AudioProcessor(sample_rate=16000)

# Process audio
output_audio, info = processor.process(
    audio_file="input.wav",
    text_prompt="Add a slowly increasing reverb effect",
    output_file="output.wav",
    normalize=True,
    verbose=True,
)

print(f"Processing info: {info}")
```

## Tips for Better Results

### Text Prompts

Good prompts are specific about:
- **Effects**: reverb, chorus, distortion, low-pass filter
- **Timing**: "over 10 seconds", "middle part", "gradually", "starting at 2 seconds"
- **Intensity**: "gentle", "intense", "subtle", "aggressive"

For example: "Add a super strong distortion effect and apply a low pass filter, only in the middle part of the audio"

I attached 5 output audio files in the `output/` folder that were generated from `input.wav` using the following prompt. Listen to them to get a sense of how the system interprets the prompt and applies the effects over time.

Prompt 1: "Apply a mild distortion effect"
Prompt 2: "Add a slowly growing reverb effect"
Prompt 3: "Apply a super strong distortion effect and a high pass filter"
Prompt 4: "Apply a super strong distortion effect and a high pass filter"
Prompt 5: "Apply a super strong distortion effect and a low pass filter, only in the middle part of the audio"

Sample output logs:

```

Parsing and validating parameters...
  Effect 1: distortion [1.0s - 3.1s]
  Effect 2: low_pass_filter [1.0s - 3.1s]
Creating parameter envelopes...
Applying effects...
Normalizing output audio...
Saving output to output.wav...

Evaluation Metrics:
============================================================
QUANTITATIVE EVALUATION
============================================================

ENERGY ANALYSIS:
  Input RMS:          -18.61 dB
  Output RMS:          -6.56 dB
  RMS Change:          12.05 dB

DYNAMIC RANGE:
  Input:               17.77 dB
  Output:               6.56 dB

PEAK LEVELS:
  Input Peak:         0.9082
  Output Peak:        1.0000

SPECTRAL CHANGES:
  Spectral Distance:   0.0065

EFFECT PARAMETERS APPLIED:

  [DISTORTION] @ 1.0s - 3.1s
    • gain: 0.000 → 8.000
    • tone: 0.500

  [LOW_PASS_FILTER] @ 1.0s - 3.1s
    • cutoff_freq: 10000.000 → 1000.000
    • resonance: 0.500 → 0.700

============================================================
QUALITATIVE OBSERVATIONS
============================================================
  ✓ Volume increased significantly
  ✓ Dynamic range compressed (effects applied)
  ⚠ Warning: Some clipping detected

============================================================
Processing complete!

Success! Output saved to: output.wav
Processing info: {'audio_duration': 4.051882086167801, 'sample_rate': 44100, 'llm_output': {'effects': [{'type': 'distortion', 'start_time': 1.0, 'end_time': 3.1, 'gain': {'start': 0.0, 'end': 8.0}, 'tone': {'start': 0.5, 'end': 0.5}}, {'type': 'low_pass_filter', 'start_time': 1.0, 'end_time': 3.1, 'cutoff_freq': {'start': 10000.0, 'end': 1000.0}, 'resonance': {'start': 0.5, 'end': 0.7}}], 'total_duration_seconds': 4.1}, 'num_effects': 2, 'envelope_created': True, 'effects_applied': True, 'output_file': 'output.wav', 'evaluation': {'input_rms_db': np.float64(-18.60880960424388), 'output_rms_db': np.float64(-6.558343401554985), 'rms_change_db': np.float64(12.050466202688895), 'input_dynamic_range_db': 17.77217674255371, 'output_dynamic_range_db': 6.558343887329102, 'input_peak': 0.908172607421875, 'output_peak': 1.0, 'spectral_change': 0.006509455852210522, 'effect_summary': {'distortion': {'count': 1, 'time_ranges': [{'start': 1.0, 'end': 3.1, 'duration': 2.1}], 'param_ranges': {'gain': {'min': np.float64(0.0), 'max': np.float64(8.0)}, 'tone': {'min': np.float64(0.5), 'max': np.float64(0.5)}}}, 'low_pass_filter': {'count': 1, 'time_ranges': [{'start': 1.0, 'end': 3.1, 'duration': 2.1}], 'param_ranges': {'cutoff_freq': {'min': np.float64(1000.0), 'max': np.float64(10000.0)}, 'resonance': {'min': np.float64(0.5), 'max': np.float64(0.7)}}}}, 'observations': {'volume_increased': np.True_, 'volume_decreased': np.False_, 'dynamic_range_compressed': True, 'spectral_changed': False, 'no_clipping': False}}}
```


<!-- ### Audio Quality

- **Input format**: WAV, MP3, FLAC
- **Sample rate**: The processor automatically resamples to 44.1kHz -->

<!-- ## Testing

Run the basic test to verify everything is working:

```bash
python tests/test_basic.py
``` -->




## Resources

- Gemini API: https://ai.google.dev/
- Librosa (audio loading): https://librosa.org/
- Pedalboard (optional advanced effects): https://github.com/spotify/pedalboard
