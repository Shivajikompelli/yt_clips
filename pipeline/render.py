"""Stage 5: render vertical clips with burned-in captions via FFmpeg."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .utils import ToolError, find_tool

log = logging.getLogger("clipper.render")


def _filter_graph(ass_filename: str, width: int, height: int) -> str:
    """Blurred-fill background, width-fit foreground, burned ASS, yuv420p.

    `ass_filename` must be a bare filename relative to the process cwd (the ass
    file's own directory). Absolute Windows paths contain 'C:' and the colon is
    parsed as an option separator inside filtergraphs.
    """
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"boxblur=luma_radius=24:luma_power=2:chroma_radius=12,"
        f"eq=brightness=-0.05[bgb];"
        f"[fg]scale={width}:-2:flags=lanczos[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,"
        f"ass={ass_filename},"
        f"format=yuv420p[v]"
    )


def _quote_filter_arg(text: str) -> str:
    """Quote an argument for use inside a filtergraph option value."""
    return "'" + text.replace("\\", "/").replace("'", r"\'") + "'"


def render_clip(
    src: Path,
    start: float,
    end: float,
    ass_path: Path,
    out_path: Path,
    cfg,
) -> Path:
    """Render one vertical clip; `ass_path` timestamps are clip-relative.

    Uses input seeking (`-ss` before `-i`) which combined with re-encoding and
    the default `-accurate_seek` gives frame-accurate cuts and synced audio.
    """
    ffmpeg = find_tool("ffmpeg")
    # absolute paths: the subprocess runs with cwd=ass_dir (see below)
    src = Path(src).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)

    ass_file = Path(ass_path).resolve()
    ass_dir = ass_file.parent
    graph = _filter_graph(_quote_filter_arg(ass_file.name), cfg.out_w, cfg.out_h)

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src),
        "-t", f"{duration:.3f}",
        "-filter_complex", graph,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    log.info("rendering %s (%.1fs - %.1fs)", out_path.name, start, end)
    try:
        # cwd is the ass file's directory so the filtergraph can reference it
        # by bare filename (Windows drive-letter colons break filtergraphs)
        proc = subprocess.run(cmd, capture_output=True, cwd=str(ass_dir))
    except FileNotFoundError as exc:
        raise ToolError(f"could not execute {ffmpeg}: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        raise ToolError(f"ffmpeg failed for {out_path.name}:\n{stderr[-2000:]}")
    log.info("wrote %s", out_path)
    return out_path
