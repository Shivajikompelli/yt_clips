"""Single-worker job queue: one video processes at a time, extras wait queued.

A queue.Queue + one daemon thread keeps this dependency-free. Each job re-checks
the project status from the DB before running, so jobs that were deleted or
already retried while waiting are skipped cleanly, and `try_claim` makes state
transitions race-free against the API endpoints.
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from pathlib import Path

from pipeline.download import VideoTooLongError

from .db import Database

log = logging.getLogger("clipper.app.worker")


def _short_error(exc: BaseException) -> str:
    """Readable one-liner for the DB; full traceback goes to the log."""
    if isinstance(exc, FileNotFoundError):
        return f"input not found: {exc}"
    if isinstance(exc, VideoTooLongError):
        return str(exc)
    if isinstance(exc, RuntimeError):  # LLMError and ToolError are RuntimeErrors
        return str(exc).splitlines()[0][:300]
    return f"{type(exc).__name__}: {exc}"


class Worker:
    def __init__(self, db: Database, project_dir: Path):
        self.db = db
        self.project_dir = project_dir
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="clipper-worker", daemon=True)
        self._started = False

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True
            log.info("worker thread started (single-job, queue=%d)", self._queue.qsize())

    def enqueue(self, project_id: str) -> None:
        self._queue.put(project_id)
        log.info("queued project %s (queue depth %d)", project_id, self._queue.qsize())

    @property
    def queued_count(self) -> int:
        return self._queue.qsize()

    # ---------------------------------------------------------------- loop

    def _loop(self) -> None:
        while True:
            pid = self._queue.get()
            try:
                self._run_one(pid)
            except Exception:  # never let one job kill the worker
                log.error("worker loop error for %s\n%s", pid, traceback.format_exc())
            finally:
                self._queue.task_done()

    def _run_one(self, pid: str) -> None:
        project = self.db.get_project(pid)
        if project is None:
            log.info("project %s vanished before start; skipping", pid)
            return
        # the retry endpoint flips status back to queued; the DB row is the
        # source of truth, not the queue entry
        if project["status"] != "queued":
            log.info("project %s is %s (not queued); skipping", pid, project["status"])
            return
        if not self.db.try_claim(pid, "queued", "downloading"):
            return

        from pipeline.runner import run_pipeline
        from pipeline.utils import load_config

        cfg = load_config()
        if project.get("num_clips"):
            cfg.clips_to_generate = project["num_clips"]

        run_dir = self.project_dir / pid

        def on_status(stage: str, percent: int) -> None:
            # stage names come straight from the runner and match the status enum
            self.db.update_project(pid, status=stage, stage=stage, progress=percent)

        log.info("processing project %s (%s)", pid, project["source_type"])
        try:
            result = run_pipeline(project["source_value"], run_dir, cfg, on_status=on_status)
            self.db.replace_clips(pid, [
                {
                    "title": c.title,
                    "score": c.score,
                    "reason": c.reason,
                    "start_sec": c.start,
                    "end_sec": c.end,
                    "file_path": str(p),
                }
                for c, p in zip(result.clips, result.paths)
            ])
            self.db.update_project(pid, status="ready", progress=100, stage="ready", error=None)
            log.info("project %s ready: %d clip(s)", pid, len(result.paths))
        except Exception as exc:
            log.error("project %s failed\n%s", pid, traceback.format_exc())
            self.db.update_project(pid, status="failed", error=_short_error(exc))
