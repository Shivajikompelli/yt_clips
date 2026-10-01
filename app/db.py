"""SQLite persistence for Phase 2: stdlib sqlite3, no ORM.

sqlite3 (stdlib) over SQLAlchemy: the API is a single process with one worker
thread, so a process-local connection guarded by a lock is the right size. It
adds zero dependencies and keeps deployment to "python + a file".
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("clipper.app.db")

DB_NAME = "clipper.db"

PROJECTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source_type  TEXT NOT NULL CHECK (source_type IN ('upload', 'url', 'local')),
    source_value TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued',
    stage        TEXT,
    progress     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    client_id    TEXT,
    num_clips    INTEGER,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
)
"""

CLIPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id         TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    title      TEXT NOT NULL,
    score      INTEGER,
    reason     TEXT,
    start_sec  REAL,
    end_sec    REAL,
    file_path  TEXT,
    created_at TEXT NOT NULL
)
"""

SCHEMAS = [PROJECTS_SCHEMA, CLIPS_SCHEMA]

# statuses that mean "the worker may still be running this project"
IN_PROGRESS = ("queued", "downloading", "transcribing", "analyzing", "rendering")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex


class Database:
    """One shared connection, serialized by an RLock (writes are tiny)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            for schema in SCHEMAS:
                self._conn.execute(schema)
            self._conn.commit()
            self._migrate_source_type()

    def _migrate_source_type(self) -> None:
        """Rebuild pre-'local' projects tables so source_type allows 'local'.

        SQLite CHECK constraints are only evaluated on write, but an old table
        would reject new 'local' rows. The rename also repoints other tables'
        FK clauses at the renamed name, so `clips` is recreated as well —
        otherwise its REFERENCES clause can end up naming the temp table and
        every later clip write fails with 'no such table: projects_old'.
        Runs in one explicit transaction; skipped when new or already migrated.
        """
        constraint = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'projects'"
        ).fetchone()
        if constraint is None or "'local'" in (constraint[0] or ""):
            return
        log.info("migrating projects table: source_type now allows 'local'")
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("ALTER TABLE projects RENAME TO projects_old")
                self._conn.execute(PROJECTS_SCHEMA)
                self._conn.execute(
                    "INSERT INTO projects (id, title, source_type, source_value, status,"
                    " stage, progress, error, client_id, num_clips, created_at, updated_at)"
                    " SELECT id, title, source_type, source_value, status, stage, progress,"
                    " error, client_id, num_clips, created_at, updated_at FROM projects_old"
                )
                # recreate clips so its FK references `projects`, not projects_old
                self._conn.execute("DROP TABLE clips")
                self._conn.execute(CLIPS_SCHEMA)
                self._conn.execute("DROP TABLE projects_old")
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            log.info("migrating done; clips rows preserved: %d",
                     self._conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ helpers

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query_one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def _query_all(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------ projects

    def insert_project(self, *, title: str, source_type: str, source_value: str,
                       client_id: str | None = None, num_clips: int | None = None) -> dict:
        pid = new_id()
        now = _now()
        self._execute(
            "INSERT INTO projects (id, title, source_type, source_value, status,"
            " client_id, num_clips, created_at, updated_at) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
            (pid, title, source_type, source_value, client_id, num_clips, now, now),
        )
        return self.get_project(pid)  # type: ignore[return-value]

    def get_project(self, pid: str) -> dict[str, Any] | None:
        return self._query_one("SELECT * FROM projects WHERE id = ?", (pid,))

    def list_projects(self) -> list[dict[str, Any]]:
        # rowid = insertion order; created_at has second precision, which ties
        return self._query_all("SELECT * FROM projects ORDER BY rowid DESC")

    def update_project(self, pid: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = _now()
        cols = ", ".join(f"{name} = ?" for name in fields)
        self._execute(f"UPDATE projects SET {cols} WHERE id = ?",
                      (*fields.values(), pid))

    def try_claim(self, pid: str, from_status: str, to_status: str) -> bool:
        """Atomically move a project out of `from_status`; False if someone else did."""
        cur = self._execute(
            "UPDATE projects SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
            (to_status, _now(), pid, from_status),
        )
        return cur.rowcount == 1

    def mark_stale_failed(self) -> int:
        """Fail projects left mid-flight by a restart (queued jobs survive)."""
        cur = self._execute(
            "UPDATE projects SET status = 'failed', error = 'interrupted by restart',"
            " updated_at = ? WHERE status IN ('downloading', 'transcribing', 'analyzing', 'rendering')",
            (_now(),),
        )
        if cur.rowcount:
            log.warning("marked %d in-progress project(s) as failed after restart", cur.rowcount)
        return cur.rowcount

    # ------------------------------------------------------------ clips

    def replace_clips(self, pid: str, clips: list[dict[str, Any]]) -> None:
        now = _now()
        with self._lock:
            self._conn.execute("DELETE FROM clips WHERE project_id = ?", (pid,))
            for i, clip in enumerate(clips):
                self._conn.execute(
                    "INSERT INTO clips (id, project_id, idx, title, score, reason,"
                    " start_sec, end_sec, file_path, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (new_id(), pid, i, clip.get("title", ""), clip.get("score"),
                     clip.get("reason"), clip.get("start_sec"), clip.get("end_sec"),
                     clip.get("file_path"), now),
                )
            self._conn.commit()

    def clips_for_project(self, pid: str) -> list[dict[str, Any]]:
        return self._query_all(
            "SELECT * FROM clips WHERE project_id = ? ORDER BY idx", (pid,)
        )

    def get_clip(self, clip_id: str) -> dict[str, Any] | None:
        return self._query_one("SELECT * FROM clips WHERE id = ?", (clip_id,))

    # ------------------------------------------------------------ deletion

    TERMINAL_STATUSES = ("ready", "failed")

    def try_delete(self, pid: str) -> tuple[bool, dict[str, Any] | None]:
        """Atomically delete a project only when its status is terminal.

        Compare-and-set style, like try_claim: a single DELETE guarded on the
        status list, so a job that starts between a check and the delete can
        never be deleted mid-run. Returns (deleted, project): project is the
        row before deletion (for file cleanup), None when the row is missing;
        deleted is False when the status was non-terminal.
        """
        placeholders = ", ".join("?" for _ in self.TERMINAL_STATUSES)
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM projects WHERE id = ?", (pid,)
            ).fetchone()
            if row is None:
                return False, None
            cur = self._conn.execute(
                f"DELETE FROM projects WHERE id = ? AND status IN ({placeholders})",
                (pid, *self.TERMINAL_STATUSES),
            )
            if cur.rowcount != 1:
                self._conn.rollback()
                return False, dict(row)
            self._conn.execute("DELETE FROM clips WHERE project_id = ?", (pid,))
            self._conn.commit()
            return True, dict(row)

    def delete_project(self, pid: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM clips WHERE project_id = ?", (pid,))
            self._conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
            self._conn.commit()
