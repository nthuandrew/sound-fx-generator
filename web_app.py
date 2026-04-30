"""
Flask web app for interactive Sound FX generation.

Features:
- Upload input wav + prompt + optional reference audio
- Generate effect parameters via LLM pipeline
- Display all effect parameter sliders (bars)
- Regenerate output audio directly from slider values
- Play/download output audio in browser
"""

import os
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory

import config
from core.audio_processor import AudioProcessor
from utils.audio_io import load_audio, normalize_audio, save_audio

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = WEB_DIR / "uploads"
GENERATED_DIR = WEB_DIR / "static" / "generated"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "static"),
)


@dataclass
class SessionData:
    session_id: str
    input_audio_file: str
    reference_audio_file: Optional[str]
    duration: float
    sample_rate: int
    last_output_file: Optional[str] = None


SESSION_STORE: Dict[str, SessionData] = {}
SESSION_LOCK = threading.Lock()


def _safe_filename(prefix: str, original_name: str) -> str:
    suffix = Path(original_name).suffix.lower() or ".wav"
    return f"{prefix}_{uuid.uuid4().hex[:10]}{suffix}"


def _build_default_control_values() -> Dict[str, Dict]:
    controls: Dict[str, Dict] = {}
    for effect_name, param_spec in config.PARAM_CONSTRAINTS.items():
        controls[effect_name] = {"enabled": False, "params": {}}
        for param_name, (min_v, max_v) in param_spec.items():
            midpoint = float((min_v + max_v) / 2.0)
            controls[effect_name]["params"][param_name] = {
                "start": midpoint,
                "end": midpoint,
            }
    return controls


def _extract_control_values_from_llm_output(
    processor: AudioProcessor,
    llm_output: Dict,
    duration: float,
) -> Dict[str, Dict]:
    controls = _build_default_control_values()
    parsed = processor.parser.parse_parameters(llm_output, audio_duration=duration)

    accumulator: Dict[Tuple[str, str], Dict[str, float]] = {}
    for instance in parsed:
        effect_name = instance.effect_type
        if effect_name not in controls:
            continue

        controls[effect_name]["enabled"] = True
        segment_duration = max(1e-6, float(instance.end_time - instance.start_time))
        for param_name, param in instance.parameters.items():
            key = (effect_name, param_name)
            item = accumulator.setdefault(
                key,
                {
                    "weight": 0.0,
                    "start_sum": 0.0,
                    "end_sum": 0.0,
                },
            )
            item["weight"] += segment_duration
            item["start_sum"] += float(param.start_value) * segment_duration
            item["end_sum"] += float(param.end_value) * segment_duration

    for (effect_name, param_name), item in accumulator.items():
        if item["weight"] > 0:
            controls[effect_name]["params"][param_name] = {
                "start": float(item["start_sum"] / item["weight"]),
                "end": float(item["end_sum"] / item["weight"]),
            }

    return controls


def _build_llm_output_from_controls(controls: Dict, duration: float) -> Dict:
    effects = []
    for effect_name, effect_data in controls.items():
        if not effect_data.get("enabled", False):
            continue

        # 先收集所有 param 的 envelope 長度
        param_envs = {}
        max_points = 0
        for param_name, value in effect_data.get("params", {}).items():
            if isinstance(value, list):
                env = value
            elif isinstance(value, dict) and "start" in value and "end" in value:
                env = [
                    {"t": 0.0, "v": float(value["start"])},
                    {"t": 1.0, "v": float(value["end"])}
                ]
            else:
                env = [
                    {"t": 0.0, "v": float(value)},
                    {"t": 1.0, "v": float(value)}
                ]
            param_envs[param_name] = env
            max_points = max(max_points, len(env))

        # For every parameter's envelope, segment the time according to its control points, and generate multiple automation segments.
        # assume all params share the same control point t (guaranteed by UI)
        if max_points < 2:
            continue
        for seg in range(max_points - 1):
            start_t = min(env[seg]["t"] for env in param_envs.values() if len(env) > seg)
            end_t = max(env[seg+1]["t"] for env in param_envs.values() if len(env) > seg+1)
            effect = {
                "type": effect_name,
                "start_time": float(start_t) * duration,
                "end_time": float(end_t) * duration,
            }
            for param_name, env in param_envs.items():
                if len(env) > seg+1:
                    effect[param_name] = {
                        "start": float(env[seg]["v"]),
                        "end": float(env[seg+1]["v"]),
                    }
            effects.append(effect)

    return {"effects": effects, "total_duration_seconds": float(duration)}


def _mode_from_form(value: Optional[str]) -> str:
    if not value:
        return "generate"
    return value.strip().lower().replace("-", "_")


def _to_json_safe(value):
    """Recursively convert numpy/scalar containers into JSON-serializable Python types."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summarize_generate_error(exc: Exception) -> str:
    raw = str(exc)
    upper = raw.upper()

    if any(token in upper for token in ["429", "503", "UNAVAILABLE", "QUOTA", "RESOURCE_EXHAUSTED"]):
        # Prefer server-provided retry hints when available.
        retry_seconds = None
        m = re.search(r"retry in\s+([0-9]+(?:\.[0-9]+)?)s", raw, flags=re.IGNORECASE)
        if m:
            retry_seconds = int(float(m.group(1)))
        else:
            m = re.search(r"retry_delay\s*\{[^}]*seconds:\s*([0-9]+)", raw, flags=re.IGNORECASE)
            if m:
                retry_seconds = int(m.group(1))

        if retry_seconds is not None:
            return (
                f"Gemini API is temporarily busy or quota-limited. "
                f"Please retry in about {retry_seconds} seconds."
            )

        return "Gemini API is temporarily busy or quota-limited. Please wait a moment and retry."

    return raw


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/effects-spec")
def effects_spec():
    return jsonify(
        {
            "effects": config.PARAM_CONSTRAINTS,
            "supported_effects": config.SUPPORTED_EFFECTS,
        }
    )


@app.post("/api/generate")
def generate():
    if "input_audio" not in request.files:
        return jsonify({"error": "input_audio file is required"}), 400

    input_file = request.files["input_audio"]
    if not input_file.filename:
        return jsonify({"error": "input_audio filename is empty"}), 400

    mode = _mode_from_form(request.form.get("mode"))
    prompt = request.form.get("prompt", "").strip()
    normalize = request.form.get("normalize", "true").lower() != "false"

    if mode not in {"generate", "extract_and_clone"}:
        return jsonify({"error": f"Unsupported mode: {mode}"}), 400

    reference_upload = request.files.get("reference_audio")
    reference_audio_path = None

    if mode == "extract_and_clone" and (reference_upload is None or not reference_upload.filename):
        return jsonify({"error": "reference_audio is required in extract-and-clone mode"}), 400

    input_filename = _safe_filename("input", input_file.filename)
    input_path = UPLOAD_DIR / input_filename
    input_file.save(input_path)

    if reference_upload and reference_upload.filename:
        reference_filename = _safe_filename("reference", reference_upload.filename)
        reference_path = UPLOAD_DIR / reference_filename
        reference_upload.save(reference_path)
        reference_audio_path = str(reference_path)

    processor = AudioProcessor()

    output_name = f"generated_{uuid.uuid4().hex[:10]}.wav"
    output_path = GENERATED_DIR / output_name

    try:
        _, info = processor.process(
            audio_file=str(input_path),
            text_prompt=prompt or "Apply suitable effects from the prompt.",
            reference_audio_file=reference_audio_path,
            output_file=str(output_path),
            normalize=normalize,
            verbose=False,
            mode=mode,
        )
    except Exception as exc:
        return jsonify({"error": _summarize_generate_error(exc), "details": str(exc)}), 500

    duration = float(info.get("audio_duration", 0.0))
    sample_rate = int(info.get("sample_rate", config.DEFAULT_SR))
    session_id = uuid.uuid4().hex

    with SESSION_LOCK:
        SESSION_STORE[session_id] = SessionData(
            session_id=session_id,
            input_audio_file=str(input_path),
            reference_audio_file=reference_audio_path,
            duration=duration,
            sample_rate=sample_rate,
            last_output_file=str(output_path),
        )

    llm_output = info.get("llm_output", {"effects": []})
    controls = _extract_control_values_from_llm_output(processor, llm_output, duration)

    payload = {
        "session_id": session_id,
        "mode": mode,
        "prompt": prompt,
        "llm_output": llm_output,
        "control_values": controls,
        "evaluation": info.get("evaluation", {}),
        "output_url": f"/api/audio/{output_name}",
        "output_file": output_name,
    }
    return jsonify(_to_json_safe(payload))


@app.post("/api/regenerate")
def regenerate_from_controls():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id")
    controls = payload.get("controls")
    normalize = bool(payload.get("normalize", True))

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not isinstance(controls, dict):
        return jsonify({"error": "controls must be an object"}), 400

    with SESSION_LOCK:
        session = SESSION_STORE.get(session_id)

    if not session:
        return jsonify({"error": "session not found; generate first"}), 404

    processor = AudioProcessor(sample_rate=session.sample_rate)

    try:
        input_audio, sr = load_audio(session.input_audio_file, sr=processor.sample_rate, mono=True)
    except Exception as exc:
        return jsonify({"error": f"Failed to load original input audio: {exc}"}), 500

    llm_output = _build_llm_output_from_controls(controls, duration=session.duration)

    try:
        effect_instances = processor.parser.parse_parameters(llm_output, audio_duration=session.duration)
        output_audio = processor.apply_extracted_effects(input_audio, sr, effect_instances, verbose=False)
        if normalize:
            output_audio = normalize_audio(output_audio, target_level=-3.0)

        output_name = f"regenerated_{uuid.uuid4().hex[:10]}.wav"
        output_path = GENERATED_DIR / output_name
        save_audio(output_audio, str(output_path), sr=sr)

        evaluation = processor._compute_evaluation_metrics(input_audio, output_audio, effect_instances, sr)
    except Exception as exc:
        return jsonify({"error": f"Failed to regenerate audio from controls: {exc}"}), 500

    with SESSION_LOCK:
        session.last_output_file = str(output_path)

    payload = {
        "session_id": session_id,
        "llm_output": llm_output,
        "control_values": controls,
        "evaluation": evaluation,
        "output_url": f"/api/audio/{output_name}",
        "output_file": output_name,
    }
    return jsonify(_to_json_safe(payload))


@app.get("/api/audio/<path:filename>")
def serve_generated_audio(filename: str):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=False)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
