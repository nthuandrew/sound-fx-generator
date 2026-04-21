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

## Phase 2 — Reference Audio Conditioning (Spectrogram Shortcut)

### Goal
Enable context-aware generation by conditioning on reference audio, starting from spectrogram inputs.

### Tasks
- Add a preprocessing module to convert reference audio into mel-spectrogram images.
- Extend prompt/template to include reference context cues from spectrogram summaries.
- Compare **text-only** vs **text + reference spectrogram** generation quality.
- Log generated parameter differences and resulting audio behavior.

### Deliverables
- End-to-end reference-audio-assisted generation prototype.
- A/B comparison examples (without and with reference input).

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
