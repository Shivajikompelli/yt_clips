"""Single pipeline orchestration path shared by the CLI and the API worker.

Executes download → transcribe → highlight → caption → render, reporting
progress as (stage, percent) so any caller (CLI logs, API job tracker) can
observe without the pipeline knowing about either.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .captions import build_ass, write_ass
from .download import get_video
from .highlights import CLIPS_NAME, LLMError, Highlight, find_highlights, make_llm_client
from .render import render_clip
from .transcribe import transcribe
from .utils import ToolError, slugify
from .words import segments_to_words, words_between

log = logging.getLogger("clipper.runner")


class RenderError(ToolError):
    """FFmpeg failed while rendering a clip (distinct exit code for the CLI)."""

# progress is distributed across the stages roughly by wall-clock cost
_DOWNLOAD_SPAN = (0, 20)
_TRANSCRIBE_SPAN = (20, 50)
_ANALYZE_SPAN = (50, 55)
_RENDER_SPAN = (55, 100)


@dataclass
class PipelineResult:
    clips: list[Highlight]
    paths: list[Path]
    video_path: Path


def _clamp_span(t: float, span: tuple[int, int]) -> int:
    lo, hi = span
    return int(lo + (hi - lo) * max(0.0, min(1.0, t)))


def run_pipeline(
    source: str,
    run_dir: Path,
    cfg,
    on_status: Callable[[str, int], None] | None = None,
    out_dir: Path | None = None,
) -> PipelineResult:
    """Run the full pipeline for one source, reporting (stage, percent) progress.

    Stage names match the API's status values: downloading, transcribing,
    analyzing, rendering. Raises ToolError / LLMError / FileNotFoundError /
    VideoTooLongError on failure; the caller decides how to surface them.
    """
    out_dir = out_dir or run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def report(stage: str, t: float, span: tuple[int, int]) -> None:
        if on_status:
            on_status(stage, _clamp_span(t, span))

    report("downloading", 0.0, _DOWNLOAD_SPAN)
    video_path = get_video(source, run_dir, cfg)
    report("downloading", 1.0, _DOWNLOAD_SPAN)

    report("transcribing", 0.0, _TRANSCRIBE_SPAN)
    transcript = transcribe(run_dir / "audio.wav", run_dir, cfg)
    report("transcribing", 1.0, _TRANSCRIBE_SPAN)

    report("analyzing", 0.0, _ANALYZE_SPAN)
    client = None
    if not (run_dir / CLIPS_NAME).exists():
        # only needed when clips are not cached; a resumed run with cached
        # clips.json can complete without any LLM key set
        client = make_llm_client(cfg)
    clips = find_highlights(transcript, run_dir, cfg, client)
    if not clips:
        raise LLMError("no clips were generated; nothing to render")
    report("analyzing", 1.0, _ANALYZE_SPAN)

    all_words = segments_to_words(transcript.get("segments", []))
    paths: list[Path] = []
    for i, clip in enumerate(clips, 1):
        report("rendering", (i - 1) / len(clips), _RENDER_SPAN)
        stem = f"{i:02d}_{slugify(clip.title)}"
        ass_text = build_ass(
            words_between(all_words, clip.start, clip.end),
            clip.start, clip.end, cfg.caption_style, cfg.out_w, cfg.out_h,
        )
        ass_path = run_dir / f"{stem}.ass"
        write_ass(ass_text, ass_path)
        out_path = out_dir / f"{stem}.mp4"
        try:
            render_clip(video_path, clip.start, clip.end, ass_path, out_path, cfg)
        except ToolError as exc:
            log.error("%s", exc)
            raise RenderError(str(exc)) from exc
        paths.append(out_path)
        report("rendering", i / len(clips), _RENDER_SPAN)

    return PipelineResult(clips=clips, paths=paths, video_path=video_path)
