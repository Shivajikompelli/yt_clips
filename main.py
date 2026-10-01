"""clipper: turn a long video into short vertical clips with burned-in captions."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from pipeline.runner import RenderError, run_pipeline
from pipeline.download import VideoTooLongError
from pipeline.highlights import LLMError
from pipeline.utils import ROOT, ToolError, ensure_ffmpeg_available, load_config, setup_logging

log = logging.getLogger("clipper.main")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MISSING_TOOL = 3
EXIT_MISSING_KEY = 4
EXIT_BAD_INPUT = 5
EXIT_VIDEO_TOO_LONG = 6
EXIT_RENDER_FAILED = 7
EXIT_LLM_FAILED = 8

RUN_DIR = ROOT / "workdir"
OUT_DIR = ROOT / "output"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="Turn a long video into short vertical clips with captions.",
    )
    parser.add_argument("source", help="video URL or local file path")
    parser.add_argument("--out", default=str(OUT_DIR), help="output directory (default: output/)")
    parser.add_argument("--clips", type=int, default=None, help="number of clips to generate")
    parser.add_argument("--model", default=None, help="whisper model size (tiny|base|small|...)")
    parser.add_argument("--keep-source", action="store_true",
                        help="keep the source video after rendering")
    parser.add_argument("--resume", metavar="RUN_ID",
                        help="reuse an existing run's cached transcript and clips.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def make_run_dir(run_id: str | None) -> Path:
    """Create workdir/<run_id>/ (or reuse it when resuming)."""
    directory = RUN_DIR / (run_id or new_run_id())
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def new_run_id() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def output_paths(clips, out_dir: Path) -> list[Path]:
    """Numbered output paths: 01_slug.mp4, 02_slug.mp4, ..."""
    from pipeline.utils import slugify

    return [out_dir / f"{i:02d}_{slugify(clip.title)}.mp4"
            for i, clip in enumerate(sorted(clips, key=lambda c: c.start), 1)]


def print_summary(clips, paths: list[Path]) -> None:
    rows = [
        (p.name, f"{c.start:7.1f}-{c.end:7.1f}s", f"{c.score}/10", c.title)
        for c, p in zip(clips, paths)
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(3)] if rows else [0, 0, 0]
    print()
    print("=== clips ===")
    for name, span, score, title in rows:
        print(f"{name:<{widths[0]}}  {span:<{widths[1]}}  {score:<{widths[2]}}  {title}")
    if paths:
        print(f"\n{len(paths)} clip(s) in {paths[0].parent}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    setup_logging(args.verbose)

    cfg = load_config()
    if args.clips is not None:
        cfg.clips_to_generate = args.clips
    if args.model:
        cfg.whisper_model = args.model
    if args.keep_source:
        cfg.delete_source_after_render = False

    try:
        # resolve ffmpeg/ffprobe up front; also puts their dir on PATH so
        # yt-dlp can merge video+audio formats
        ensure_ffmpeg_available()
        run_dir = make_run_dir(args.resume)
        log.info("run dir: %s", run_dir)

        out_dir = Path(args.out).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        result = run_pipeline(
            args.source, run_dir, cfg,
            on_status=lambda stage, pct: log.info("[%3d%%] %s", pct, stage),
            out_dir=out_dir,
        )

        if cfg.delete_source_after_render:
            for stale in run_dir.glob("source.*"):
                log.info("deleting source %s (delete_source_after_render)", stale.name)
                stale.unlink(missing_ok=True)

        print_summary(result.clips, result.paths)
        return EXIT_OK

    except VideoTooLongError as exc:
        log.error("%s", exc)
        return EXIT_VIDEO_TOO_LONG
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return EXIT_BAD_INPUT
    except LLMError as exc:
        log.error("%s", exc)
        return EXIT_MISSING_KEY
    except ToolError as exc:
        log.error("%s", exc)
        # render failures get their own exit code (7); other tool failures are 3
        return EXIT_RENDER_FAILED if isinstance(exc, RenderError) else EXIT_MISSING_TOOL
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
