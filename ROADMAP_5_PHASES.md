# Sound FX Generator — Next Steps Roadmap (5 Phases)

This roadmap organizes the next development stage into five practical phases for the final deliverable.

## Phase 1 — Stabilize Core Pipeline and Reliability

### Goal
Make the current text-to-parameter-to-audio pipeline robust and reproducible.

### Status
- [x] Phase 1 completed

### Tasks
- [x] Improve LLM output robustness (strict JSON validation, retry/fallback strategy).
- [x] Add better error handling for malformed responses and unsupported effect specs.
- [x] Lock core dependencies and environment setup for reproducible runs.
- [x] Build baseline test cases for common prompts (distortion, low-pass, chorus, reverb).

### Deliverables
- [x] Stable CLI workflow with reduced failure rate.
- [x] Regression tests and reproducibility notes.

---

## Phase 2 — Audio Effect Style Transfer & Reverse Engineering (VLM Spectrogram Analysis)

### Goal
Evolve from context-aware generation to **effect cloning**: Use VLM (Gemini 2.5 Pro) to reverse-engineer reference audio's effects and parameters from spectrogram visualization, then apply extracted parameters to input audio.

### Conceptual Shift
| Dimension | Previous (Heuristic Context) | **New (Spectrogram Reverse Engineering)** |
|-----------|------------------------------|---------------------------------------------|
| Reference Role | Background info (mood) | **Target blueprint** (effect analysis) |
| Analysis Method | DSP stats (RMS, centroids) → text | **Visual features** (spectrogram image) |
| LLM Task | "Create" parameters based on vibes | **"Reverse-engineer"** actual effects used |
| Goal | Prompt conditioning | **Effect cloning / style transfer** |

### Phase 2A — Spectrogram Visualization & VLM Integration

#### Tasks
- [x] **Remove legacy DSP module**: Delete heuristic context (RMS, Centroid, Rolloff calculations from `utils/reference_audio.py`). Keep audio loading and segmentation.
- [x] **Add spectrogram visualization**: New function `generate_reference_spectrogram_image()` that:
  - Reads reference.wav with librosa
  - Computes short-time Fourier transform (STFT) → magnitude spectrogram
  - Uses **linear frequency** and **linear magnitude** scales (per ST-ITO ablation study)
  - Applies **viridis colormap**
  - Preserves X (time) and Y (frequency) axes without colorbar (prevents visual interference)
  - Returns image as **byte buffer** (PNG in memory, ready for VLM API)
- [x] **Upgrade LLM client to support multimodal**: 
  - Switch to `Gemini 2.5 Pro` (or `Gemini 2.0 Flash` as fallback if needed)
  - Extend `core/llm_prompt.py` to accept image input alongside text
  - Support `genai.upload_file()` or direct base64 encoding for spectrogram image
  - Add robust error handling for multimodal API responses

#### Deliverables
- [x] `utils/spectrogram_renderer.py`: Pure visualization module (no analysis, just rendering)
- [x] Updated `core/llm_prompt.py` with VLM support and image input handling

### Phase 2B — Prompt Rewrite & Parameter Extraction

#### Tasks
- [x] **Rewrite prompt to reverse-engineering mode**:
  - Reposition LLM as "professional audio engineer" analyzing spectrogram
  - Explicitly ask for effect identification: Reverb (tail decay), Delay (echo patterns), Distortion (harmonic spread), EQ (spectral shaping)
  - Request **time-variant parameter extraction** in JSON:
    ```json
    {
      "effects": [
        {
          "name": "Reverb",
          "time_segments": [
            { "start_time": 0.0, "end_time": 5.0, "decay_time": 1.5, "wet_dry": 0.3, "width": 0.9 },
            { "start_time": 5.0, "end_time": 10.0, "decay_time": 2.5, "wet_dry": 0.5, "width": 0.95 }
          ]
        }
      ]
    }
    ```
  - Include reference image in multimodal prompt

- [x] **Extend parameter parser** `core/parameter_parser.py`:
  - Add schema validation for extracted time-variant parameters
  - Support per-effect time segmentation with (start_time, end_time) tuples
  - Validate constraint ranges align with effect definitions

#### Deliverables
- [x] Reverse-engineering prompt template in `core/llm_prompt.py`
- [x] Enhanced parameter parser with time-segment support

### Phase 2C — Effect Cloning Pipeline Integration

#### Tasks
- [ ] **Create effect extraction & application workflow**:
  - `extract_reference_effects()`: Call VLM with spectrogram image → receive extracted effect parameters + time segments
  - `apply_extracted_effects()`: Apply extracted parameters (time-varying curves) to input audio using existing effect chain
  - Return cloned audio with reference style applied

- [ ] **Update `core/audio_processor.py`**:
  - New mode: `mode="extract_and_clone"` (extract from reference, apply to input)
  - Accept both `reference_audio_file` and `input_audio_file`
  - Log extracted parameters for debugging and comparison
  - Output cloned audio with reference effect style

- [ ] **Add CLI support for effect cloning**:
  - New argument: `--mode {generate, extract-and-clone}` (default: generate)
  - When `extract-and-clone`: `--reference-audio reference.wav --input input.wav --output output_cloned.wav`

#### Deliverables
- [ ] `extract_reference_effects()` function with VLM analysis
- [ ] Updated `core/audio_processor.py` with cloning mode
- [ ] Updated `main.py` CLI with `--mode extract-and-clone`

### Phase 2D — Testing & Validation

#### Tasks
- [ ] **Update unit tests**:
  - `test_spectrogram_rendering()`: Verify spectrogram image generation (dimensions, colormap)
  - `test_vlm_multimodal_call()`: Mock VLM response with time-variant parameters
  - `test_parameter_extraction_time_segments()`: Validate parser handles (start_time, end_time) correctly
  - `test_effect_cloning_pipeline()`: End-to-end test: extract from reference → apply to input → verify output

- [ ] **Manual A/B testing**:
  - Compare **original reference** vs **extracted + re-applied cloned version** (should be ~90%+ similar)
  - Compare **text-only generation** vs **VLM reverse-engineered cloning** (quality, parameter fidelity)
  - Create example report with parameter tables

#### Deliverables
- [ ] `tests/test_spectrogram_renderer.py` (4+ tests)
- [ ] Updated `tests/test_llm_prompt.py` with multimodal tests
- [ ] A/B comparison report: reference.wav → extract → clone → compare

### Status
- [x] Phase 2A core: Spectrogram rendering + VLM integration
- [x] Phase 2B core: Prompt rewrite + parameter extraction
- [ ] Phase 2C core: Effect cloning pipeline
- [ ] Phase 2D testing: Validation + A/B experiments

### Overall Phase 2 Deliverables
- [x] Spectrogram visualization module (`utils/spectrogram_renderer.py`)
- [x] Multimodal VLM-enabled `core/llm_prompt.py`
- [x] Time-variant parameter extraction & parsing
- [ ] Effect cloning mode in `core/audio_processor.py` and CLI
- [ ] Comprehensive unit tests (8+ tests for Phase 2B+C+D)
- [ ] A/B comparison report demonstrating effect style transfer

---

## Phase 3 — Black-Box VST / Plugin-Oriented Control

### Goal
Move beyond fixed built-in effects by supporting black-box plugin parameter automation.

### Tasks
- Design a plugin adapter schema: parameter name, range, default, normalized domain.
- Map LLM outputs to plugin automation envelopes (time-varying curves).
- Add safety constraints (clamp, smoothing, max slope).
- Validate with at least one plugin-like target (or simulated plugin parameter set).

### Deliverables
- Flexible parameter automation layer for non-native effects.
- Demonstration of zero-shot-style control on plugin parameters.

---

## Phase 4 — Interactive Web Application

### Goal
Upgrade from CLI prototype to a usable application.

### Tasks
- Build backend API for upload, processing, and result retrieval.
- Build frontend for file upload, prompt input, progress feedback, and download.
- Expose generated parameter JSON and timeline visualization for interpretability.
- Add job management and basic request validation.

### Deliverables
- Functional web demo for end users.
- User flow: upload → prompt → process → preview/download.

---

## Phase 5 — Evaluation, Ablation, and Final Packaging

### Goal
Produce final-report-quality evidence and prepare final submission.

### Tasks
- Quantitative evaluation: alignment and audio-change metrics across settings.
- Ablation studies:
  - fixed vs flexible control
  - static vs time-varying parameters
  - text-only vs text+reference
- Qualitative listening study (small-scale user feedback if feasible).
- Consolidate final documentation, architecture diagram, and demo assets.

### Deliverables
- Final report-ready results section (tables + examples).
- Final demo package and reproducible instructions.

---

## Suggested Execution Order
1. Phase 1 (stability)
2. Phase 2 (reference conditioning)
3. Phase 4 (web demo)
4. Phase 3 (plugin-oriented flexibility)
5. Phase 5 (evaluation and final packaging)

> Note: If timeline is tight, prioritize Phases 1, 2, 4, and a lightweight version of Phase 3.
