"""Stage 2: transcribe audio with faster-whisper into word-level JSON."""
from __future__ import annotations

import logging
import os
import time
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from .utils import read_json, write_json

log = logging.getLogger("clipper.transcribe")

TRANSCRIPT_NAME = "transcript.json"
SAMPLE_RATE = 16000


def load_wav_float32(path: Path) -> np.ndarray:
    """Read a 16-bit PCM mono wav into a float32 array in [-1, 1].

    We pass samples directly to faster-whisper instead of a file path so the
    PyAV decoder is never used (some av releases are incompatible with
    faster-whisper 1.x); our audio.wav is already 16 kHz mono.
    """
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16-bit PCM wav")
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path.name}: expected {SAMPLE_RATE} Hz wav")
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _maybe_use_cached(cached: dict, cache: Path, cfg) -> dict:
    """Return a cached transcript, warning when it came from another model."""
    cached_model = cached.get("model", "?")
    if cached_model != cfg.whisper_model:
        log.warning(
            "cached transcript was made with model '%s' but --model is '%s'; "
            "delete %s to force re-transcription",
            cached_model, cfg.whisper_model, cache,
        )
    else:
        log.info("loading cached transcript %s", cache)
    return cached


def transcribe(audio_path: Path, run_dir: Path, cfg) -> dict:
    """Transcribe audio_path, caching the result in run_dir/transcript.json.

    Returns a dict: {"language": str, "duration": float,
                     "segments": [{"start", "end", "text", "words": [{"start","end","word"}]}]}
    """
    cache = run_dir / TRANSCRIPT_NAME
    if cache.exists():
        try:
            cached = read_json(cache)
        except (ValueError, OSError):
            # e.g. interrupted mid-write; transcribe again rather than crash
            log.warning("cached transcript is unreadable, re-transcribing")
            cache.unlink(missing_ok=True)
        else:
            return _maybe_use_cached(cached, cache, cfg)

    log.info("loading faster-whisper model '%s' (int8, CPU)", cfg.whisper_model)
    model = WhisperModel(
        cfg.whisper_model,
        device="cpu",
        compute_type="int8",
        cpu_threads=max(4, os.cpu_count() or 4),
    )

    samples = load_wav_float32(audio_path)
    segments_iter, info = model.transcribe(
        samples,
        language=None,
        word_timestamps=True,
        vad_filter=True,
        # avoids hallucination spiral loops on music/noise-heavy audio
        condition_on_previous_text=False,
    )
    total = info.duration or len(samples) / SAMPLE_RATE
    log.info("transcribing %.1f s of audio (language=%s)", total, info.language)

    segments: list[dict] = []
    started = time.monotonic()
    last_report = 0.0
    for seg in segments_iter:
        words = [
            {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
            for w in (seg.words or [])
        ]
        segments.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "words": words,
            }
        )
        # simple progress indication roughly every 5% of audio
        if total > 0 and seg.end / total - last_report >= 0.05:
            last_report = seg.end / total
            elapsed = time.monotonic() - started
            eta = elapsed / max(last_report, 1e-6) * (1 - last_report)
            log.info("progress %3d%%  (elapsed %4.0fs, eta %4.0fs)",
                     int(last_report * 100), elapsed, eta)

    transcript = {
        "language": info.language or "en",
        "duration": round(total, 3),
        "model": cfg.whisper_model,
        "segments": segments,
    }
    write_json(cache, transcript)
    log.info("wrote %s (%d segments)", cache.name, len(segments))
    return transcript
