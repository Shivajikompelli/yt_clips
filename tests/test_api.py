"""Phase 2 API tests: job lifecycle with the pipeline mocked, no network, no ffmpeg."""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as app_main
import pipeline.runner as runner_mod
from app.db import Database
from pipeline.highlights import Highlight
from pipeline.runner import PipelineResult
from pipeline.utils import Config


# ---------------------------------------------------------------- helpers


def _fake_success(source, run_dir, cfg, on_status=None, out_dir=None):
    """Stand-in for run_pipeline: fast, deterministic, writes one clip file."""
    out_dir = out_dir or run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    for stage, pct in (("downloading", 20), ("transcribing", 50), ("analyzing", 55)):
        if on_status:
            on_status(stage, pct)
    time.sleep(0.02)
    clips = [Highlight(start=0.0, end=20.0, title="Fake Clip", score=9, reason="mocked")]
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(clips, 1):
        p = out_dir / f"{i:02d}_fake-clip.mp4"
        p.write_bytes(b"fake mp4 bytes")
        paths.append(p)
        if on_status:
            on_status("rendering", 55 + i)
    if on_status:
        on_status("rendering", 100)
    return PipelineResult(clips=clips, paths=paths, video_path=run_dir / "source.mp4")


def _cfg_with(**overrides) -> Config:
    cfg = Config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture()
def env(tmp_path: Path, monkeypatch):
    """Isolated storage + DB + a worker we can flush deterministically."""
    uploads = tmp_path / "uploads"
    projects = tmp_path / "projects"
    uploads.mkdir()
    projects.mkdir()

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(app_main, "DB_PATH", tmp_path / "clipper.db")
    monkeypatch.setattr(app_main, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(app_main, "PROJECTS_DIR", projects)

    db = Database(tmp_path / "clipper.db")
    worker = app_main.Worker(db, projects)
    # startup calls _init_runtime; make it return our isolated instances
    monkeypatch.setattr(app_main, "_init_runtime", lambda: (db, worker))
    monkeypatch.setattr(runner_mod, "run_pipeline", _fake_success)

    return {"db": db, "worker": worker, "uploads": uploads, "projects": projects}


@pytest.fixture()
def client(env):
    with TestClient(app_main.app) as c:
        yield c


def flush(worker, db, timeout: float = 5.0) -> None:
    """Block until the worker queue is fully drained."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if worker.queued_count == 0 and not any(
            p["status"] not in ("ready", "failed", "queued") for p in db.list_projects()
        ):
            time.sleep(0.05)
            return
        time.sleep(0.02)
    raise TimeoutError("worker did not drain in time")


# ---------------------------------------------------------------- create


class TestCreateProject:
    def test_url_returns_202_immediately(self, client, env):
        resp = client.post("/projects", json={"url": "https://youtube.com/watch?v=abc"})
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "queued"
        assert len(body["id"]) == 32

    def test_url_row_persisted(self, client, env):
        resp = client.post("/projects", json={"url": "https://x.test/v.mp4", "title": "T", "clips": 2})
        row = env["db"].get_project(resp.json()["id"])
        assert row["source_type"] == "url"
        assert row["title"] == "T"
        assert row["num_clips"] == 2
        # num_clips is exposed on both list and detail responses (Phase 3 UI)
        listed = client.get("/projects").json()
        assert listed[0]["num_clips"] == 2
        detail = client.get(f"/projects/{resp.json()['id']}").json()
        assert detail["num_clips"] == 2

    def test_upload_accepted(self, client, env):
        resp = client.post(
            "/projects",
            files={"file": ("talk.mp4", b"0" * 1024, "video/mp4")},
        )
        assert resp.status_code == 202
        row = env["db"].get_project(resp.json()["id"])
        assert row["source_type"] == "upload"
        assert Path(row["source_value"]).is_file()
        assert row["title"] == "talk"
        # stored under a unique, project-derived name (not a shared upload.mp4)
        assert Path(row["source_value"]).name == f"{row['id']}.mp4"

    def test_concurrent_uploads_do_not_collide(self, client, env):
        """Two uploads with different content must land in different files
        with their own bytes, even when the worker has not started yet."""
        r1 = client.post("/projects", files={"file": ("a.mp4", b"content-A", "video/mp4")})
        r2 = client.post("/projects", files={"file": ("b.mp4", b"content-BB", "video/mp4")})
        id1, id2 = r1.json()["id"], r2.json()["id"]
        assert id1 != id2
        p1 = Path(env["db"].get_project(id1)["source_value"])
        p2 = Path(env["db"].get_project(id2)["source_value"])
        assert p1 != p2
        assert p1.name == f"{id1}.mp4" and p2.name == f"{id2}.mp4"
        assert p1.read_bytes() == b"content-A"
        assert p2.read_bytes() == b"content-BB"
        flush(env["worker"], env["db"])

    def test_failed_upload_leaves_no_orphan_row(self, client, env):
        import pipeline.utils as utils

        monkeypatch_free = client
        assert monkeypatch_free is not None
        # empty file -> 422 -> the pre-inserted project row must be removed
        resp = client.post("/projects", files={"file": ("v.mp4", b"", "video/mp4")})
        assert resp.status_code == 422
        assert env["db"].list_projects() == []

    def test_invalid_url_rejected(self, client):
        for bad in ("not a url", "ftp://x", "https://"):
            assert client.post("/projects", json={"url": bad}).status_code == 422

    def test_file_url_valid(self, client, env, tmp_path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"local bytes")
        resp = client.post("/projects", json={"url": video.as_uri()})
        assert resp.status_code == 202
        row = env["db"].get_project(resp.json()["id"])
        assert row["source_type"] == "local"
        assert Path(row["source_value"]).is_file()

    def test_file_url_missing_file_rejected(self, client, tmp_path):
        resp = client.post("/projects", json={"url": (tmp_path / "nope.mp4").as_uri()})
        assert resp.status_code == 422
        assert "local file not found" in resp.json()["detail"]

    def test_file_url_malformed_rejected(self, client):
        # "file://host/path" form: no drive letter, resolves to a bogus path
        for bad in ("file://C:/does/not/exist.mp4", "file:///C:/no/such/file.txt"):
            resp = client.post("/projects", json={"url": bad})
            assert resp.status_code == 422, bad

    def test_plain_local_path_stored_as_local(self, client, env, tmp_path):
        video = tmp_path / "talk.mkv"
        video.write_bytes(b"x")
        resp = client.post("/projects", json={"url": str(video)})
        assert resp.status_code == 202
        row = env["db"].get_project(resp.json()["id"])
        assert row["source_type"] == "local"
        assert row["source_value"] == str(video.resolve())

    def test_local_path_bad_extension_rejected(self, client, tmp_path):
        audio = tmp_path / "song.mp3"
        audio.write_bytes(b"x")
        resp = client.post("/projects", json={"url": str(audio)})
        assert resp.status_code == 422
        assert "unsupported file type" in resp.json()["detail"]

    def test_bad_extension_rejected(self, client):
        resp = client.post("/projects", files={"file": ("song.mp3", b"x", "audio/mpeg")})
        assert resp.status_code == 422
        assert "unsupported file type" in resp.json()["detail"]

    def test_empty_upload_rejected(self, client):
        assert client.post("/projects", files={"file": ("v.mp4", b"", "video/mp4")}).status_code == 422

    def test_size_limit_rejected(self, client, env, monkeypatch):
        # patch where app.main uses it (from-import bound a local name)
        monkeypatch.setattr(app_main, "load_config", lambda: _cfg_with(max_upload_mb=1 / (1024 * 1024)))
        resp = client.post("/projects", files={"file": ("v.mp4", b"too big", "video/mp4")})
        assert resp.status_code == 413
        # the partial file must not linger in uploads/
        assert list(env["uploads"].glob("upload.*")) == []

    def test_multipart_to_root_routed_to_upload(self, client, env):
        resp = client.post("/projects", files={"file": ("v.mp4", b"x", "video/mp4")})
        assert resp.status_code == 202
        assert env["db"].get_project(resp.json()["id"])["source_type"] == "upload"


# ---------------------------------------------------------------- lifecycle


class TestLifecycle:
    def test_progresses_to_ready_with_clips(self, client, env):
        pid = client.post("/projects", json={"url": "https://x.test/v"}).json()["id"]
        flush(env["worker"], env["db"])
        detail = client.get(f"/projects/{pid}").json()
        assert detail["status"] == "ready"
        assert detail["progress"] == 100
        assert len(detail["clips"]) == 1
        clip = detail["clips"][0]
        assert clip["title"] == "Fake Clip"
        # num_clips echoes the request; None when unspecified (the UI applies
        # the config default of 5 client-side)
        assert detail["num_clips"] is None
        # server paths must not leak; route URLs are exposed instead
        assert "file_path" not in clip
        assert clip["download_url"] == f"/clips/{clip['id']}/download"
        assert clip["stream_url"] == f"/clips/{clip['id']}/stream"

    def test_list_newest_first(self, client, env):
        ids = [
            client.post("/projects", json={"url": f"https://x.test/{i}"}).json()["id"]
            for i in range(3)
        ]
        flush(env["worker"], env["db"])
        listed = client.get("/projects").json()
        assert [p["id"] for p in listed][:3] == ids[::-1]

    def test_single_worker_serializes_jobs(self, client, env, monkeypatch):
        """Jobs run one at a time: overlapping execution must never happen."""
        import pipeline.runner as rm

        active = threading.Semaphore(1)
        overlap = {"happened": False}

        def checking_pipeline(*args, **kwargs):
            if not active.acquire(blocking=False):
                overlap["happened"] = True  # two pipelines ran at once
            time.sleep(0.05)
            try:
                return _fake_success(*args, **kwargs)
            finally:
                active.release()

        monkeypatch.setattr(rm, "run_pipeline", checking_pipeline)
        first = client.post("/projects", json={"url": "https://x.test/1"}).json()["id"]
        second = client.post("/projects", json={"url": "https://x.test/2"}).json()["id"]
        flush(env["worker"], env["db"], timeout=10)
        assert client.get(f"/projects/{first}").json()["status"] == "ready"
        assert client.get(f"/projects/{second}").json()["status"] == "ready"
        assert not overlap["happened"]

    def test_pipeline_failure_marks_failed(self, client, env, monkeypatch):
        import pipeline.runner as runner_mod

        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("yt-dlp could not download https://x.test/v: video unavailable")
            return _fake_success(*a, **kw)

        monkeypatch.setattr(runner_mod, "run_pipeline", flaky)
        pid = client.post("/projects", json={"url": "https://x.test/v"}).json()["id"]
        flush(env["worker"], env["db"])
        detail = client.get(f"/projects/{pid}").json()
        assert detail["status"] == "failed"
        assert "yt-dlp" in detail["error"]
        # a failed job must not stop the worker: next job goes through
        pid2 = client.post("/projects", json={"url": "https://x.test/v2"}).json()["id"]
        flush(env["worker"], env["db"])
        assert client.get(f"/projects/{pid2}").json()["status"] == "ready"
        assert calls["n"] == 2

    def test_404_for_unknown_project(self, client):
        assert client.get(f"/projects/{uuid.uuid4().hex}").status_code == 404


# ---------------------------------------------------------------- retry


class TestRetry:
    def test_retry_requeues_failed(self, client, env, monkeypatch):
        import pipeline.runner as runner_mod

        calls = {"n": 0}

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient failure")
            return _fake_success(*a, **kw)

        monkeypatch.setattr(runner_mod, "run_pipeline", flaky)
        pid = client.post("/projects", json={"url": "https://x.test/v"}).json()["id"]
        flush(env["worker"], env["db"])
        assert client.get(f"/projects/{pid}").json()["status"] == "failed"

        resp = client.post(f"/projects/{pid}/retry")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"
        flush(env["worker"], env["db"])
        assert client.get(f"/projects/{pid}").json()["status"] == "ready"

    def test_retry_reuses_run_dir(self, client, env, monkeypatch):
        """Retry points at the same project dir so cached transcript survives."""
        pid = client.post("/projects", json={"url": "https://x.test/v"}).json()["id"]
        flush(env["worker"], env["db"])
        marker = env["projects"] / pid / "transcript.json"
        marker.write_text('{"cached": true}')
        env["db"].update_project(pid, status="failed", error="x")

        resp = client.post(f"/projects/{pid}/retry")
        flush(env["worker"], env["db"])
        assert resp.json()["status"] == "queued"
        assert marker.exists()  # run dir was not wiped

    def test_retry_ready_project_conflict(self, client, env):
        pid = client.post("/projects", json={"url": "https://x.test/v"}).json()["id"]
        flush(env["worker"], env["db"])
        assert client.post(f"/projects/{pid}/retry").status_code == 409


# ---------------------------------------------------------------- delete


class TestDelete:
    def test_delete_ready_project(self, client, env):
        pid = client.post("/projects", json={"url": "https://x.test/v"}).json()["id"]
        flush(env["worker"], env["db"])
        assert client.delete(f"/projects/{pid}").status_code == 200
        assert client.get(f"/projects/{pid}").status_code == 404
        assert not (env["projects"] / pid).exists()

    def test_delete_upload_removes_file(self, client, env):
        pid = client.post("/projects", files={"file": ("v.mp4", b"x", "video/mp4")}).json()["id"]
        flush(env["worker"], env["db"])
        source = env["db"].get_project(pid)["source_value"]
        assert client.delete(f"/projects/{pid}").status_code == 200
        assert not Path(source).exists()

    def test_delete_processing_conflict(self, client, env, monkeypatch):
        gate = threading.Event()

        def gated(*a, **kw):
            gate.wait(3)
            return _fake_success(*a, **kw)

        monkeypatch.setattr(runner_mod, "run_pipeline", gated)

        pid = client.post("/projects", json={"url": "https://x.test/v"}).json()["id"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if client.get(f"/projects/{pid}").json()["status"] == "transcribing":
                break
            time.sleep(0.02)
        assert client.delete(f"/projects/{pid}").status_code == 409
        gate.set()
        flush(env["worker"], env["db"], timeout=10)

    def test_delete_unknown_404(self, client):
        assert client.delete(f"/projects/{uuid.uuid4().hex}").status_code == 404

    def test_delete_race_is_atomic(self, client, env):
        """A status flip between check and delete must not delete a running job.

        delete_if_terminal guards the DELETE on the terminal-status list in a
        single statement; simulate the flip by re-statusing the row after the
        endpoint's read but proving the guard rejects non-terminal states.
        """
        pid = client.post("/projects", json={"url": "https://x.test/v"}).json()["id"]
        flush(env["worker"], env["db"])

        # flip to a non-terminal status *after* the project is visible: the
        # atomic delete must refuse even though the row exists
        env["db"].update_project(pid, status="queued")
        resp = client.delete(f"/projects/{pid}")
        assert resp.status_code == 409
        assert env["db"].get_project(pid) is not None  # row survived

        # the surviving row is still processable: enqueue and it completes
        env["worker"].enqueue(pid)
        flush(env["worker"], env["db"], timeout=10)
        assert client.get(f"/projects/{pid}").json()["status"] == "ready"

        # now terminal: delete succeeds
        assert client.delete(f"/projects/{pid}").status_code == 200
        assert env["db"].get_project(pid) is None

    def test_delete_if_terminal_missing_row(self, env):
        deleted, project = env["db"].try_delete(uuid.uuid4().hex)
        assert deleted is False
        assert project is None


# ---------------------------------------------------------------- clips


class TestClipEndpoints:
    @pytest.fixture()
    def ready(self, client, env):
        pid = client.post("/projects", json={"url": "https://x.test/v"}).json()["id"]
        flush(env["worker"], env["db"])
        return client.get(f"/projects/{pid}").json()

    def test_download(self, client, ready):
        clip = ready["clips"][0]
        resp = client.get(f"/clips/{clip['id']}/download")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("video/mp4")
        assert resp.headers["content-disposition"].startswith("attachment")
        assert resp.content == b"fake mp4 bytes"

    def test_stream_full(self, client, ready):
        clip = ready["clips"][0]
        resp = client.get(f"/clips/{clip['id']}/stream")
        assert resp.status_code == 200
        assert resp.headers["accept-ranges"] == "bytes"
        assert resp.content == b"fake mp4 bytes"

    def test_stream_range(self, client, ready):
        clip = ready["clips"][0]
        resp = client.get(f"/clips/{clip['id']}/stream", headers={"Range": "bytes=0-3"})
        assert resp.status_code == 206
        assert resp.headers["content-range"] == "bytes 0-3/14"
        assert resp.content == b"fake"

    def test_stream_unknown_404(self, client):
        assert client.get(f"/clips/{uuid.uuid4().hex}/stream").status_code == 404


# ---------------------------------------------------------------- health


class TestHealth:
    def test_health_shape(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body) == {"status", "ffmpeg_available", "llm_provider", "llm_configured"}
        assert body["llm_provider"] == "gemini"
        assert body["llm_configured"] is True


# ---------------------------------------------------------------- recovery


class TestStaleRecovery:
    def test_stale_in_progress_failed_on_startup(self, env, monkeypatch):
        db = env["db"]
        stuck = db.insert_project(title="stuck", source_type="url", source_value="https://x.test/v")
        db.update_project(stuck["id"], status="transcribing", stage="transcribing", progress=35)
        queued = db.insert_project(title="queued", source_type="url", source_value="https://x.test/v2")

        # re-run the startup recovery logic against this DB
        db.mark_stale_failed()
        stuck_row = db.get_project(stuck["id"])
        assert stuck_row["status"] == "failed"
        assert stuck_row["error"] == "interrupted by restart"
        queued_row = db.get_project(queued["id"])
        assert queued_row["status"] == "queued"  # queued jobs survive restarts

    def test_mark_stale_leaves_terminal_states(self, env):
        db = env["db"]
        ready = db.insert_project(title="r", source_type="url", source_value="https://x/v")
        db.update_project(ready["id"], status="ready")
        failed = db.insert_project(title="f", source_type="url", source_value="https://x/v")
        db.update_project(failed["id"], status="failed")
        db.mark_stale_failed()
        assert db.get_project(ready["id"])["status"] == "ready"
        assert db.get_project(failed["id"])["status"] == "failed"


# ---------------------------------------------------------------- phase 3 UI


class TestStaticUi:
    def test_root_serves_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert b"<!doctype html>" in resp.content.lower()
        assert b'app.js' in resp.content  # the UI entry point is wired up

    def test_static_assets_served(self, client):
        for path, needle in (("/static/styles.css", "text/css"),
                             ("/static/app.js", "javascript")):
            resp = client.get(path)
            assert resp.status_code == 200, path
            assert needle in resp.headers["content-type"], path
