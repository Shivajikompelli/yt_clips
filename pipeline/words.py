"""Helpers to turn transcript segments into flat word lists for captions."""
from __future__ import annotations

import logging

log = logging.getLogger("clipper.words")


def segments_to_words(segments: list[dict]) -> list[dict]:
    """Flatten transcript segments into a single word list."""
    return [w for seg in segments for w in seg.get("words", [])]


def words_between(words: list[dict], start: float, end: float) -> list[dict]:
    """Words overlapping the [start, end) window."""
    return [
        w for w in words
        if float(w["end"]) > start and float(w["start"]) < end
    ]
