"""Shared helpers: config, logging, ffprobe, text utilities."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "whisper_model": "base",
    "max_input_minutes": 60,
    "clips_to_generate": 5,
    "min_clip_seconds": 20,
    "max_clip_seconds": 60,
    "llm_chunk_minutes": 15,
    "llm_max_retries": 3,
    "max_upload_mb": 1000,
    "caption_style": {
        "font": "Arial",
        "font_size": 56,
        "primary_color": "&H00FFFFFF",
        "highlight_color": "&H0000E5FF",
        "outline": 3,
        "position": "bottom",
    },
    "output_resolution": "1080x1920",
    "delete_source_after_render": True,
}

CAPTION_COLORS = {
    "white": "&H00FFFFFF",
    "yellow": "&H0000E5FF",
    "cyan": "&H00FFFF00",
    "green": "&H0000FF00",
    "red": "&H000000FF",
    "blue": "&H00FF0000",
}

log = logging.getLogger("clipper")


# ---------------------------------------------------------------- config


@dataclass
class CaptionStyle:
    font: str = "Arial"
    font_size: int = 56
    primary_color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000E5FF"
    outline: int = 3
    position: str = "bottom"


@dataclass
class Config:
    whisper_model: str = "base"
    max_input_minutes: float = 60
    clips_to_generate: int = 5
    min_clip_seconds: float = 20
    max_clip_seconds: float = 60
    llm_chunk_minutes: float = 15
    llm_max_retries: int = 3
    max_upload_mb: float = 1000
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)
    output_resolution: str = "1080x1920"
    delete_source_after_render: bool = True

    @property
    def out_w(self) -> int:
        return int(self.output_resolution.lower().split("x")[0])

    @property
    def out_h(self) -> int:
        return int(self.output_resolution.lower().split("x")[1])


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None) -> Config:
    cfg_path = path or (ROOT / "config.yaml")
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if isinstance(loaded, dict):
            data = loaded
    merged = deep_merge(DEFAULTS, data)
    style = {**DEFAULTS["caption_style"], **(merged.get("caption_style") or {})}
    style["primary_color"] = _color(style["primary_color"])
    style["highlight_color"] = _color(style["highlight_color"])
    merged["caption_style"] = style
    return Config(
        whisper_model=str(merged["whisper_model"]),
        max_input_minutes=float(merged["max_input_minutes"]),
        clips_to_generate=int(merged["clips_to_generate"]),
        min_clip_seconds=float(merged["min_clip_seconds"]),
        max_clip_seconds=float(merged["max_clip_seconds"]),
        llm_chunk_minutes=float(merged["llm_chunk_minutes"]),
        llm_max_retries=int(merged.get("llm_max_retries", 3)),
        max_upload_mb=float(merged.get("max_upload_mb", 1000)),
        caption_style=CaptionStyle(**style),
        output_resolution=str(merged["output_resolution"]),
        delete_source_after_render=bool(merged["delete_source_after_render"]),
    )


def _color(value: Any) -> str:
    """Normalize a caption color to ASS &HAABBGGRR form."""
    text = str(value).strip()
    if text.lower() in CAPTION_COLORS:
        return CAPTION_COLORS[text.lower()]
    if re.fullmatch(r"&H[0-9A-Fa-f]{1,8}", text):
        digits = text[2:].upper().rjust(8, "0")
        return f"&H{digits}"
    m = re.fullmatch(r"#?([0-9A-Fa-f]{6})", text)  # RRGGBB
    if m:
        r, g, b = m.group(1)[0:2], m.group(1)[2:4], m.group(1)[4:6]
        return f"&H00{b}{g}{r}".upper()
    raise ValueError(f"invalid caption color: {value!r}")


# ---------------------------------------------------------------- logging


def setup_logging(verbose: bool = False) -> None:
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. uvicorn --reload runs lifespan twice)
        return
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------- external tools


class ToolError(RuntimeError):
    """A required external tool is missing or failed."""


def find_tool(name: str) -> str:
    """Locate an executable: PATH, then CLIPPER_FFMPEG_DIR, then winget installs."""
    exe = f"{name}.exe" if os.name == "nt" else name
    found = shutil.which(name)
    if found:
        return found
    candidates: list[Path] = []
    custom = os.environ.get("CLIPPER_FFMPEG_DIR")
    if custom:
        candidates.append(Path(custom) / exe)
    candidates.extend(d / exe for d in _winget_package_dirs())
    for cand in candidates:
        if cand.exists():
            return str(cand)
    raise ToolError(
        f"'{name}' not found on PATH. Install FFmpeg (e.g. 'winget install Gyan.FFmpeg'), "
        "reopen your terminal, or set CLIPPER_FFMPEG_DIR to the folder containing "
        "ffmpeg/ffprobe."
    )


def _winget_package_dirs() -> list[Path]:
    """bin/ dirs of winget-installed portable packages (e.g. Gyan.FFmpeg).

    winget appends its package dirs to the *user* PATH, but terminals opened
    before the install never see it; this makes those shells work anyway.
    """
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return []
    base = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if not base.exists():
        return []
    dirs: list[Path] = []
    for pkg in base.glob("*FFmpeg*"):
        dirs.extend(d for d in pkg.rglob("bin") if d.is_dir())
    return dirs


def ensure_ffmpeg_available() -> str:
    """Resolve ffmpeg/ffprobe and prepend their dir to PATH for child processes
    (yt-dlp needs ffmpeg on PATH for format merging). Returns the ffmpeg path."""
    ffmpeg = find_tool("ffmpeg")
    find_tool("ffprobe")  # fail fast with the same clear error if missing
    bin_dir = str(Path(ffmpeg).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir.lower() not in (p.lower() for p in path_entries):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return ffmpeg


def run_cmd(args: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run a subprocess without a shell, raising ToolError with stderr on failure."""
    log.debug("run: %s", " ".join(args))
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ToolError(f"could not execute {args[0]}: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        raise ToolError(f"{args[0]} failed (exit {proc.returncode}): {stderr[-2000:]}")
    return proc


# ---------------------------------------------------------------- media probing


def probe_has_video_stream(path: Path) -> bool:
    """True if the media file contains at least one video stream."""
    ffprobe = find_tool("ffprobe")
    proc = run_cmd(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(path),
        ],
        timeout=120,
    )
    return bool(proc.stdout.decode(errors="replace").strip())


def probe_duration_seconds(path: Path) -> float:
    """Duration of a media file via ffprobe, in seconds."""
    ffprobe = find_tool("ffprobe")
    proc = run_cmd(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        timeout=120,
    )
    data = json.loads(proc.stdout.decode(errors="replace"))
    duration = data.get("format", {}).get("duration")
    if duration is None:
        raise ToolError(f"ffprobe could not determine duration of {path}")
    return float(duration)


# ---------------------------------------------------------------- misc


_SLUG_SAFE = re.compile(r"[^A-Za-z0-9_-]+")
_SLUG_DASH = re.compile(r"-{2,}")


def slugify(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug: lowercase ASCII alphanumerics and dashes."""
    text = (text or "").strip().lower()
    text = text.replace("'", "")
    text = _SLUG_SAFE.sub("-", text)
    text = _SLUG_DASH.sub("-", text).strip("-")
    if len(text) > max_len:
        text = text[:max_len].rsplit("-", 1)[0]
    return text or "clip"


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
