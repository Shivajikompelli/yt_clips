"""Pydantic request/response schemas for the Phase 2 API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class URLRequest(BaseModel):
    """POST /projects body when the source is a URL."""
    url: str
    title: str | None = None
    clips: int | None = Field(default=None, ge=1, le=20)


class ClipOut(BaseModel):
    id: str
    idx: int
    title: str
    score: int | None = None
    reason: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    download_url: str
    stream_url: str


class ProjectOut(BaseModel):
    id: str
    title: str
    source_type: str
    source_value: str
    status: str
    stage: str | None = None
    progress: int
    error: str | None = None
    created_at: str
    updated_at: str


class ProjectDetailOut(ProjectOut):
    clips: list[ClipOut]


class CreatedOut(BaseModel):
    id: str
    status: str


class RetryOut(BaseModel):
    id: str
    status: str


class DeletedOut(BaseModel):
    id: str
    deleted: bool


class HealthOut(BaseModel):
    status: str
    ffmpeg_available: bool
    llm_provider: str
    llm_configured: bool
