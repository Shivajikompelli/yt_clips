"""Phase 2 API: FastAPI app around the Phase 1 pipeline with SQLite job tracking."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

from pipeline.utils import ROOT, ensure_ffmpeg_available, find_tool, load_config, setup_logging

from .db import Database, IN_PROGRESS
from .models import (
    ClipOut,
    CreatedOut,
    DeletedOut,
    HealthOut,
    ProjectDetailOut,
    ProjectOut,
    RetryOut,
    URLRequest,
)
from .worker import Worker

log = logging.getLogger("clipper.app.main")

STORAGE_DIR = ROOT / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
PROJECTS_DIR = STORAGE_DIR / "projects"
DB_PATH = STORAGE_DIR / "clipper.db"

UPLOAD_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}

# DB and worker are constructed in the lifespan handler: storage dirs must
# exist first, and tests inject their own instances before that
db: Database
worker: Worker


def _init_runtime() -> tuple[Database, Worker]:
    database = Database(DB_PATH)
    return database, Worker(database, PROJECTS_DIR)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global db, worker
    setup_logging()  # INFO lines from worker/pipeline must reach the server log
    load_dotenv(ROOT / ".env")  # same as the CLI: keys for make_llm_client
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "static").mkdir(parents=True, exist_ok=True)

    db, worker = _init_runtime()

    # a killed server leaves jobs stuck mid-flight; only queued jobs survive
    db.mark_stale_failed()
    for project in db.list_projects():
        if project["status"] == "queued":
            worker.enqueue(project["id"])  # queued jobs survive restarts

    ensure_ffmpeg_available()
    worker.start()
    log.info("clipper API ready (db=%s, storage=%s)", DB_PATH, STORAGE_DIR)
    yield
    db.close()
    log.info("clipper API stopped")


app = FastAPI(title="clipper API", version="2.0.0", lifespan=lifespan)

# Phase 3 UI will live here; mounted now so the path exists from day one
app.mount("/static", StaticFiles(directory=str(ROOT / "static"), check_dir=False), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:8000", "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ helpers


def _require_project(pid: str) -> dict:
    project = db.get_project(pid)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _classify_source(url: str) -> tuple[str, str, str | None]:
    """Validate a JSON 'url' value and classify it.

    Returns (source_type, source_value, error_detail). source_type is
    'url' (remote URL), 'local' (existing file path or file:// URL), or None
    with an error_detail when the value is rejected. http URLs are accepted
    optimistically (they may exist only from the server's network POV);
    local paths and file:// URLs must exist on disk with an allowed extension.
    """
    lowered = url.lower()
    if lowered.startswith(("http://", "https://")):
        if re.match(r"^https?://[^\s/$.?#].[^\s]*$", url, flags=re.I):
            return "url", url, None
        return "", url, "invalid URL"
    if lowered.startswith("file://"):
        try:
            from urllib.parse import unquote, urlparse

            parsed = urlparse(url)
            if parsed.scheme != "file":
                return "", url, f"malformed file URL: {url}"
            path = unquote(parsed.path)
            if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
                path = path[1:]  # file:///C:/x -> C:/x
        except ValueError:
            return "", url, f"malformed file URL: {url}"
        return _classify_local_path(path, url)
    return _classify_local_path(url, url)


def _classify_local_path(path: str, original: str) -> tuple[str, str, str | None]:
    p = Path(path)
    if not p.is_file():
        return "", original, f"local file not found: {path}"
    suffix = p.suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        return "", original, (
            f"unsupported file type '{suffix}' (allowed: {', '.join(sorted(UPLOAD_EXTENSIONS))})"
        )
    return "local", str(p.resolve()), None


def _validate_url(url: str) -> None:
    _, _, error = _classify_source(url)
    if error:
        raise HTTPException(status_code=422, detail=error)


def _save_upload(upload: UploadFile, dest_dir: Path, max_bytes: int,
                 stem: str | None = None) -> Path:
    """Stream an upload to disk, refusing files that end up too big."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"unsupported file type '{suffix}' (allowed: {', '.join(sorted(UPLOAD_EXTENSIONS))})",
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stem or 'upload'}{suffix}"
    written = 0
    try:
        with open(dest, "wb") as fh:
            while chunk := upload.file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds the {max_bytes // (1024 * 1024)} MB limit",
                    )
                fh.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"could not store upload: {exc}") from exc
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="uploaded file is empty")
    return dest


# ------------------------------------------------------------------ routes


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    cfg = load_config()
    try:
        find_tool("ffmpeg")
        find_tool("ffprobe")
        ffmpeg_ok = True
    except Exception:
        ffmpeg_ok = False
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    configured = {
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
        "groq": bool(os.environ.get("GROQ_API_KEY")),
        "ollama": True,  # local; assumed reachable
    }.get(provider, False)
    return HealthOut(
        status="ok" if ffmpeg_ok and configured else "degraded",
        ffmpeg_available=ffmpeg_ok,
        llm_provider=provider,
        llm_configured=configured,
    )


@app.post("/projects", response_model=CreatedOut, status_code=202)
async def create_project(request: Request) -> CreatedOut:
    """Queue a new project from JSON {"url", "title?", "clips?"} or a multipart upload."""
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise HTTPException(status_code=422, detail="multipart request must include a 'file' field")
        return await _create_from_upload(upload)
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=422, detail="body must be JSON or multipart form data")
    if not isinstance(payload, dict) or not payload.get("url"):
        raise HTTPException(
            status_code=422,
            detail='JSON body must be {"url": "...", "title?": "...", "clips?": 5}',
        )
    body = URLRequest(**payload)
    source_type, source_value, error = _classify_source(body.url)
    if error:
        raise HTTPException(status_code=422, detail=error)
    row = db.insert_project(title=body.title or body.url, source_type=source_type,
                            source_value=source_value, num_clips=body.clips)
    worker.enqueue(row["id"])
    return CreatedOut(id=row["id"], status=row["status"])


async def _create_from_upload(upload: UploadFile) -> CreatedOut:
    cfg = load_config()
    # insert the row first so the upload file can be named <project_id>.<ext>;
    # unique names prevent concurrent uploads from overwriting each other
    row = db.insert_project(title=upload.filename or "upload", source_type="upload",
                            source_value="")
    try:
        dest = _save_upload(upload, UPLOADS_DIR, int(cfg.max_upload_mb * 1024 * 1024),
                            stem=row["id"])
    except HTTPException:
        db.delete_project(row["id"])  # failed upload leaves no orphan row
        raise
    title = Path(upload.filename or dest.stem).stem or dest.stem
    db.update_project(row["id"], title=title, source_value=str(dest))
    worker.enqueue(row["id"])
    return CreatedOut(id=row["id"], status=row["status"])


@app.get("/projects", response_model=list[ProjectOut])
def list_projects() -> list[ProjectOut]:
    return [ProjectOut(**p) for p in db.list_projects()]


@app.get("/projects/{pid}", response_model=ProjectDetailOut)
def get_project(pid: str) -> ProjectDetailOut:
    project = _require_project(pid)
    clips = [_clip_out(c) for c in db.clips_for_project(pid)]
    return ProjectDetailOut(**project, clips=clips)


def _clip_out(clip: dict) -> ClipOut:
    """Clip response: never expose server paths; expose route URLs instead."""
    return ClipOut(
        id=clip["id"], idx=clip["idx"], title=clip["title"], score=clip["score"],
        reason=clip["reason"], start_sec=clip["start_sec"], end_sec=clip["end_sec"],
        download_url=f"/clips/{clip['id']}/download",
        stream_url=f"/clips/{clip['id']}/stream",
    )


@app.get("/clips/{clip_id}/download")
def download_clip(clip_id: str) -> FileResponse:
    clip = db.get_clip(clip_id)
    if clip is None or not clip["file_path"] or not Path(clip["file_path"]).is_file():
        raise HTTPException(status_code=404, detail="clip file not found")
    name = Path(clip["file_path"]).name
    return FileResponse(clip["file_path"], media_type="video/mp4", filename=name)


@app.get("/clips/{clip_id}/stream")
def stream_clip(clip_id: str,
                range_header: str | None = Header(default=None, alias="Range")) -> StreamingResponse:
    """Serve a clip with HTTP Range support so browsers can seek previews."""
    clip = db.get_clip(clip_id)
    if clip is None or not clip["file_path"] or not Path(clip["file_path"]).is_file():
        raise HTTPException(status_code=404, detail="clip file not found")
    path = Path(clip["file_path"])
    size = path.stat().st_size

    range_value = range_header or ""
    match = re.match(r"bytes=(\d*)-(\d*)$", range_value.strip())
    if not match:
        return FileResponse(path, media_type="video/mp4",
                            headers={"Accept-Ranges": "bytes"})
    start_raw, end_raw = match.groups()
    start = int(start_raw) if start_raw else 0
    end = int(end_raw) if end_raw else min(start + 1024 * 1024, size - 1)  # 1 MB chunks
    end = min(end, size - 1)
    if start > end or start >= size:
        return JSONResponse(
            status_code=416,
            content={"detail": "requested range not satisfiable"},
            headers={"Content-Range": f"bytes */{size}"},
        )
    fh = open(path, "rb")
    fh.seek(start)
    chunk = fh.read(end - start + 1)
    fh.close()

    return StreamingResponse(
        iter([chunk]), media_type="video/mp4", status_code=206,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(len(chunk)),
        },
    )


@app.delete("/projects/{pid}", response_model=DeletedOut)
def delete_project(pid: str) -> DeletedOut:
    # atomic claim: only succeeds when the status is terminal (ready/failed),
    # so a job that starts between check and delete cannot be deleted mid-run
    deleted, project = db.try_delete(pid)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    if not deleted:
        raise HTTPException(
            status_code=409,
            detail=f"project is currently {project['status']}; wait for it to finish or fail",
        )
    _delete_project_files(project)
    return DeletedOut(id=pid, deleted=True)


def _delete_project_files(project: dict) -> None:
    import shutil

    run_dir = PROJECTS_DIR / project["id"]
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    if project["source_type"] == "upload":
        source = Path(project["source_value"])
        if source.is_file() and source.parent == UPLOADS_DIR:
            source.unlink(missing_ok=True)


@app.post("/projects/{pid}/retry", response_model=RetryOut)
def retry_project(pid: str) -> RetryOut:
    project = _require_project(pid)
    if project["status"] != "failed":
        raise HTTPException(status_code=409, detail=f"only failed projects can be retried (status: {project['status']})")
    # CAS-style claim: nobody else can flip it first; falls back to queued and
    # the worker reuses any cached transcript/clips.json in the run dir
    db.update_project(pid, status="queued", stage=None, progress=0, error=None)
    worker.enqueue(pid)
    return RetryOut(id=pid, status="queued")
