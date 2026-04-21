"""
Reference Audio Analysis Utilities

This module extracts lightweight context from a reference audio file so the LLM
can make context-aware parameter suggestions.

The current MVP uses spectrogram-derived summaries instead of raw multimodal
image input, which keeps the pipeline simple and robust while still providing
song-structure and timbral cues.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union

import numpy as np
import librosa


@dataclass
class ReferenceSegment:
    """A coarse time segment extracted from the reference audio."""

    start_time: float
    end_time: float
    avg_energy_db: float
    avg_centroid_hz: float
    avg_rolloff_hz: float
    label: str


@dataclass
class ReferenceAudioContext:
    """Compact reference-audio summary to feed into the LLM prompt."""

    file_path: str
    duration_seconds: float
    sample_rate: int
    estimated_tempo_bpm: Optional[float]
    tempo_confidence: float
    avg_rms_db: float
    avg_centroid_hz: float
    avg_rolloff_hz: float
    onset_density: float
    estimated_boundaries: List[float]
    segments: List[ReferenceSegment]
    spectrogram_shape: List[int]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["segments"] = [asdict(segment) for segment in self.segments]
        return data


def _label_segment(avg_energy_db: float, avg_centroid_hz: float, onset_density: float) -> str:
    """Heuristic segment labeling for human-readable prompt context."""
    if avg_centroid_hz > 2500 and onset_density > 1.0:
        return "bright / dense / energetic"
    if avg_centroid_hz > 1800:
        return "clear / mid-high focused"
    if avg_energy_db > -20:
        return "full / steady / loud"
    if onset_density > 1.0:
        return "percussive / rhythmic"
    return "warm / sparse / ambient"


def analyze_reference_audio(
    file_path: str,
    sr: Optional[int] = None,
    n_mels: int = 64,
    hop_length: int = 512,
) -> ReferenceAudioContext:
    """
    Analyze reference audio and return a compact spectrogram-based summary.

    Args:
        file_path: Reference audio path
        sr: Target sample rate for analysis (None keeps native)
        n_mels: Number of mel bands for spectrogram summary
        hop_length: STFT hop length

    Returns:
        ReferenceAudioContext containing a compact summary
    """
    audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
    duration_seconds = len(audio) / sample_rate

    if len(audio) == 0:
        raise ValueError(f"Reference audio is empty: {file_path}")

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        power=2.0,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)

    avg_rms_db = float(20 * np.log10(np.mean(rms) + 1e-8))
    avg_centroid_hz = float(np.mean(centroid))
    avg_rolloff_hz = float(np.mean(rolloff))
    onset_density = float(np.count_nonzero(onset_env > np.mean(onset_env) + np.std(onset_env)) / max(duration_seconds, 1e-6))

    tempo_estimate: Optional[float]
    tempo_confidence: float
    try:
        tempo_estimate, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
        )
        tempo_estimate = float(tempo_estimate)
        tempo_confidence = float(min(1.0, len(beat_frames) / max(1.0, duration_seconds * 2.0)))
    except Exception:
        tempo_estimate = None
        tempo_confidence = 0.0

    # Derive coarse boundaries by looking for largest frame-wise changes in the
    # joint energy + spectral centroid curve.
    energy_curve = librosa.util.normalize(rms) if np.max(rms) > 0 else rms
    centroid_curve = librosa.util.normalize(centroid) if np.max(centroid) > 0 else centroid
    change_curve = np.abs(np.diff(energy_curve, prepend=energy_curve[:1])) + np.abs(
        np.diff(centroid_curve, prepend=centroid_curve[:1])
    )

    candidate_count = max(1, min(4, int(round(duration_seconds / 30.0)) + 1))
    candidate_frames = np.argsort(change_curve)[::-1]
    boundary_times: List[float] = [0.0, duration_seconds]
    min_separation = max(0.75, duration_seconds * 0.08)
    for frame_idx in candidate_frames:
        time_sec = float(frame_idx * hop_length / sample_rate)
        if time_sec <= 0.0 or time_sec >= duration_seconds:
            continue
        if all(abs(time_sec - existing) >= min_separation for existing in boundary_times):
            boundary_times.append(time_sec)
        if len(boundary_times) >= candidate_count + 1:
            break

    boundary_times = sorted(set(round(boundary, 2) for boundary in boundary_times))
    if boundary_times[-1] != round(duration_seconds, 2):
        boundary_times.append(round(duration_seconds, 2))

    segments: List[ReferenceSegment] = []
    for start_time, end_time in zip(boundary_times[:-1], boundary_times[1:]):
        if end_time <= start_time:
            continue

        start_frame = max(0, int(start_time * sample_rate / hop_length))
        end_frame = max(start_frame + 1, int(end_time * sample_rate / hop_length))
        end_frame = min(end_frame, mel_spec_db.shape[1])

        segment_mel = mel_spec_db[:, start_frame:end_frame]
        segment_rms = rms[start_frame:end_frame]
        segment_centroid = centroid[start_frame:end_frame]
        segment_rolloff = rolloff[start_frame:end_frame]
        segment_onset = onset_env[start_frame:end_frame]

        seg_energy_db = float(20 * np.log10(np.mean(segment_rms) + 1e-8)) if segment_rms.size else avg_rms_db
        seg_centroid_hz = float(np.mean(segment_centroid)) if segment_centroid.size else avg_centroid_hz
        seg_rolloff_hz = float(np.mean(segment_rolloff)) if segment_rolloff.size else avg_rolloff_hz
        seg_onset_density = float(np.count_nonzero(segment_onset > np.mean(onset_env)) / max(end_time - start_time, 1e-6)) if segment_onset.size else 0.0
        label = _label_segment(seg_energy_db, seg_centroid_hz, seg_onset_density)

        segments.append(
            ReferenceSegment(
                start_time=float(start_time),
                end_time=float(end_time),
                avg_energy_db=seg_energy_db,
                avg_centroid_hz=seg_centroid_hz,
                avg_rolloff_hz=seg_rolloff_hz,
                label=label,
            )
        )

    return ReferenceAudioContext(
        file_path=file_path,
        duration_seconds=float(duration_seconds),
        sample_rate=int(sample_rate),
        estimated_tempo_bpm=tempo_estimate,
        tempo_confidence=tempo_confidence,
        avg_rms_db=avg_rms_db,
        avg_centroid_hz=avg_centroid_hz,
        avg_rolloff_hz=avg_rolloff_hz,
        onset_density=onset_density,
        estimated_boundaries=boundary_times,
        segments=segments,
        spectrogram_shape=[int(mel_spec_db.shape[0]), int(mel_spec_db.shape[1])],
    )


def format_reference_context(context: Union[ReferenceAudioContext, Dict]) -> str:
    """Format a compact human-readable context block for the LLM prompt."""
    if isinstance(context, ReferenceAudioContext):
        context = context.to_dict()

    lines = [
        "Reference audio context (derived from spectrogram analysis):",
        f"- duration: {context['duration_seconds']:.2f}s",
        f"- sample_rate: {context['sample_rate']} Hz",
        f"- spectrogram_shape: {context['spectrogram_shape'][0]} mel bands x {context['spectrogram_shape'][1]} frames",
        f"- avg_rms_db: {context['avg_rms_db']:.2f}",
        f"- avg_centroid_hz: {context['avg_centroid_hz']:.1f}",
        f"- avg_rolloff_hz: {context['avg_rolloff_hz']:.1f}",
        f"- onset_density: {context['onset_density']:.2f}",
    ]

    if context.get("estimated_tempo_bpm") is not None:
        lines.append(
            f"- estimated_tempo_bpm: {float(context['estimated_tempo_bpm']):.1f} (confidence {float(context['tempo_confidence']):.2f})"
        )
    else:
        lines.append("- estimated_tempo_bpm: unavailable")

    lines.append("- coarse segments:")
    for idx, segment in enumerate(context.get("segments", []), start=1):
        lines.append(
            f"  {idx}. [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s] "
            f"{segment['label']} | energy {segment['avg_energy_db']:.2f} dB | centroid {segment['avg_centroid_hz']:.1f} Hz"
        )

    lines.append(
        "Use this reference context to make effect decisions more song-aware: "
        "apply stronger or denser effects in brighter/louder sections, and lighter effects in sparse sections."
    )
    return "\n".join(lines)
