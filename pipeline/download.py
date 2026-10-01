"""Stage 1: acquire the source video and extract transcription audio."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .utils import (
    ToolError,
    ensure_ffmpeg_available,
    find_tool,
    probe_duration_seconds,
    probe_has_video_stream,
    run_cmd,
)

log = logging.getLogger("clipper.download")

# containers yt-dlp may legitimately produce; audio-only ones are caught by
# the video-stream probe right after the download
_MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".ts",
    ".m4a", ".mp3", ".aac", ".ogg", ".opus",
}


class VideoTooLongError(ValueError):
    """The source video exceeds max_input_minutes."""


def _source_is_url(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def get_video(source: str, run_dir: Path, cfg) -> Path:
    """Return a local video path for `source`, downloading URLs with yt-dlp.

    Raises VideoTooLongError when the source is longer than cfg.max_input_minutes.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    if _source_is_url(source):
        existing = sorted(run_dir.glob("source.*"))
        if existing:
            # resumed run: keep the previously downloaded source
            log.info("reusing existing %s from run dir", existing[0].name)
            video_path = existing[0]
        else:
            # yt-dlp shells out to ffmpeg for format merging and needs it on
            # PATH; this also fixes terminals opened before a winget install
            ensure_ffmpeg_available()
            video_path = _download_url(source, run_dir)
            if not probe_has_video_stream(video_path):
                video_path.unlink(missing_ok=True)
                raise ToolError(
                    "the URL points to an audio-only file; clipper needs a video"
                )
    else:
        local = Path(source).expanduser()
        if not local.is_file():
            raise FileNotFoundError(f"input file not found: {local}")
        video_path = _link_or_copy(local, run_dir)

    duration = probe_duration_seconds(video_path)
    log.info("source duration: %.1fs (%.1f min)", duration, duration / 60)
    max_seconds = cfg.max_input_minutes * 60
    if duration > max_seconds:
        video_path.unlink(missing_ok=True)
        raise VideoTooLongError(
            f"video is {duration / 60:.1f} min long, but max_input_minutes is "
            f"{cfg.max_input_minutes:g} (see config.yaml)"
        )

    extract_audio(video_path, run_dir / "audio.wav")
    return video_path


def _download_url(url: str, run_dir: Path) -> Path:
    """Download a single best-mp4 video with yt-dlp (no playlists)."""
    import yt_dlp

    # a resumed run dir may hold a stale source.* from a previous download
    for stale in run_dir.glob("source.*"):
        log.info("removing stale %s before download", stale.name)
        stale.unlink(missing_ok=True)

    out_base = run_dir / "source.%(ext)s"
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out_base),
        "noplaylist": True,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "socket_timeout": 30,
        "restrictfilenames": True,
    }
    log.info("downloading %s", url)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        for leftover in run_dir.glob("source.*"):
            leftover.unlink(missing_ok=True)
        raise ToolError(f"yt-dlp could not download {url}: {exc}") from exc

    candidates = sorted(run_dir.glob("source.*"))
    videos = [p for p in candidates if p.suffix.lower() in _MEDIA_EXTENSIONS]
    if not videos:
        raise ToolError(f"yt-dlp finished but no video file was produced in {run_dir}")
    if len(videos) > 1:
        log.warning("multiple source files found, using %s", videos[0].name)
    return videos[0]


def _link_or_copy(local: Path, run_dir: Path) -> Path:
    """Hard-link the local file into the run dir; fall back to copying."""
    target = run_dir / f"source{local.suffix.lower()}"
    if target.exists():
        return target
    try:
        target.hardlink_to(local.resolve())
        log.info("hard-linked %s into run dir", local.name)
    except OSError:
        log.info("copying %s into run dir", local.name)
        shutil.copy2(local, target)
    return target


def extract_audio(video_path: Path, out_wav: Path) -> None:
    """Extract mono 16 kHz PCM audio for transcription."""
    if out_wav.exists():
        log.info("audio.wav already exists, skipping extraction")
        return
    ffmpeg = find_tool("ffmpeg")
    try:
        run_cmd(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video_path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(out_wav),
            ],
            timeout=1800,
        )
    except ToolError as exc:
        if "does not contain any stream" in str(exc):
            raise ToolError(
                f"{video_path.name} has no audio track - captions require speech"
            ) from exc
        raise
    log.info("extracted audio: %s", out_wav.name)
