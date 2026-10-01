# clipper

Turn a long video (local file or URL) into short vertical **1080x1920** clips with
burned-in, word-highlighted captions — fully local and free, except one optional
free-tier LLM call that picks the highlights.

```
long video ──▶ yt-dlp ──▶ faster-whisper ──▶ LLM highlight picker ──▶ ASS captions ──▶ FFmpeg ──▶ vertical clips
              (download)   (word timestamps)   (Gemini/Groq/Ollama)     (karaoke)      (libx264)
```

## Prerequisites

- **Python 3.11+**
- **FFmpeg** (includes `ffprobe`) on your PATH:
  - Windows: `winget install -e --id Gyan.FFmpeg` (then reopen your terminal)
  - macOS: `brew install ffmpeg`
  - Debian/Ubuntu: `sudo apt install ffmpeg`
  - Verify: `ffmpeg -version`

## Install

```bash
cd clipper
python -m pip install -r requirements.txt
cp .env.example .env        # then edit .env (see below)
```

## Get a free LLM key

Pick **one** provider — all have free tiers:

| Provider | Get a key | Notes |
|----------|-----------|-------|
| Gemini (default) | https://aistudio.google.com/apikey | Generous free tier |
| Groq | https://console.groq.com/keys | Very fast |
| Ollama (local, no key) | https://ollama.com , then `ollama pull llama3.1` | No network at all |

Put it in `.env`:

```ini
LLM_PROVIDER=gemini        # gemini | groq | ollama
GEMINI_API_KEY=AIza...     # only for gemini
GROQ_API_KEY=gsk_...       # only for groq
```

## Phase 2: API server (FastAPI + SQLite)

The pipeline is also exposed as a local REST API with SQLite job tracking. One video
processes at a time; extra jobs wait in `queued` status.

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

### curl examples

```bash
# health: ffmpeg + LLM provider status (no secrets)
curl -s localhost:8000/health

# queue a project from a URL (returns immediately, 202)
curl -s -X POST localhost:8000/projects -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=...", "title": "My talk", "clips": 3}'
# -> {"id": "a1b2...", "status": "queued"}

# ...or upload a video file
curl -s -X POST localhost:8000/projects -F "file=@talk.mp4"

# list projects (newest first)
curl -s localhost:8000/projects

# project detail incl. clips (poll while status moves through
# downloading -> transcribing -> analyzing -> rendering -> ready)
# each clip carries download_url and stream_url (server paths are not exposed)
curl -s localhost:8000/projects/<id>

# download a clip
curl -sL -o clip.mp4 localhost:8000/clips/<clip_id>/download

# stream a clip in the browser (Range supported; <video> element works)
curl -s -H "Range: bytes=0-1023" localhost:8000/clips/<clip_id>/stream

# retry a failed project (reuses cached transcript)
curl -s -X POST localhost:8000/projects/<id>/retry

# delete a finished/failed project (refused with 409 while processing)
curl -s -X DELETE localhost:8000/projects/<id>
```

Files: SQLite DB at `storage/clipper.db`, uploads in `storage/uploads/` (stored as
`<project_id>.<ext>`), per-project working files and clips under
`storage/projects/<id>/`. JSON `{"url": ...}` accepts http(s) URLs and local
file paths or `file://` URLs (stored as `source_type: "local"`; local files
must exist and have an allowed extension).

## Usage

```bash
# from a URL (YouTube etc., single video, no playlists)
python main.py "https://www.youtube.com/watch?v=..."

# from a local file
python main.py "C:\path\to\talk.mp4"

# options
python main.py <url_or_path> \
  --out output            # output directory (default: output/)
  --clips 5               # number of clips (default from config.yaml)
  --model small           # whisper model: tiny|base|small|medium|large-v3
  --keep-source           # keep the source video in workdir/ after rendering
  --resume <run_id>       # reuse a previous run's transcript + clips.json
```

Example output:

```
=== clips ===
01_the-one-question.mp4        0.0-  20.7s  8/10  The One Question
02_charge-money-early.mp4     30.2-  51.8s  9/10  Charge Money Early
```

Each run caches its work in `workdir/<run_id>/` (`transcript.json`, `clips.json`,
`.ass` files). Re-running with `--resume <run_id>` skips transcription and LLM
calls entirely. Final clips land in `output/`.

## Configuration

All knobs live in `config.yaml` (defaults are sane; delete any key to fall back):

```yaml
whisper_model: base          # tiny|base|small|medium|large-v3 (speed vs accuracy)
max_input_minutes: 60        # longer inputs are rejected
clips_to_generate: 5
min_clip_seconds: 20
max_clip_seconds: 60
llm_chunk_minutes: 15        # long transcripts are fed to the LLM in chunks
max_upload_mb: 1000          # API: reject larger uploads
caption_style:               # ASS colors are &HAABBGGRR; named colors also work
  font: Arial
  font_size: 56
  primary_color: "&H00FFFFFF"     # caption text (white)
  highlight_color: "&H0000E5FF"   # currently spoken word (yellow)
  outline: 3
  position: bottom
output_resolution: 1080x1920
delete_source_after_render: true
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 2 | Bad usage / missing input file |
| 3 | FFmpeg/ffprobe missing or failed |
| 4 | LLM not configured (missing API key, bad provider) |
| 5 | Invalid input (bad URL, unreadable file) |
| 6 | Video longer than `max_input_minutes` |
| 7 | FFmpeg render failure |
| 8 | LLM produced no usable clips |

## Troubleshooting

**`ffmpeg not found`** — Install it (see Prerequisites). clipper also
auto-detects winget installs (`Gyan.FFmpeg`) even in terminals opened before
the install, so you normally do not need to reopen the terminal. If detection
still fails, set `CLIPPER_FFMPEG_DIR` to the folder containing `ffmpeg.exe` /
`ffprobe.exe`.

**`GEMINI_API_KEY is not set`** — Copy `.env.example` to `.env` and add a key,
or set `LLM_PROVIDER=ollama` and install Ollama. Keys are only read from `.env`
or real environment variables.

**Download fails / `yt-dlp could not download`** — yt-dlp breaks when sites
change; update it: `python -m pip install -U yt-dlp`. Playlists are
intentionally rejected — pass a single-video URL.

**Transcription is slow** — Use `--model tiny` or `--model base`. Expect roughly
5-8x realtime on a modern CPU with `base` (a 10-minute video takes 1-2 minutes).

**Captions look wrong (font/colors)** — `Arial` must be installed for the
system font names to resolve; change `caption_style.font` in `config.yaml` to
any font on your machine. Colors are ASS `&HAABBGGRR` (note: **BGR**, not RGB).

**Clips start mid-thought / feel incomplete** — Clip edges are aligned to complete
sentences (detected from word punctuation in the transcript), so every clip opens
with its full setup and ends after its payoff. If a clip still feels truncated,
the LLM chose a weak window: delete the run's `clips.json` and re-run with
`--resume`, or adjust `prompts/highlight_prompt.txt`. A larger whisper model
(`--model small`) also yields cleaner punctuation and better sentence detection.

**Non-English or music-heavy videos** — The default `base` model is tuned for
English and produces poor transcripts for other languages or loud background
music (you will see very few transcript segments). Use a larger model, e.g.
`--model small` (slower but multilingual), and prefer clear talking-head
sources. If the LLM finds no usable clips, nothing is cached - just fix the
input and re-run.

**Transcription is interrupted** — A killed run leaves no transcript behind;
the next run starts transcription from scratch (downloads are kept and
reused with `--resume`).

**Windows path errors from ffmpeg** — Fixed internally by running ffmpeg from
the `.ass` file's directory; if you move `.ass` files manually, keep them next
to nothing else in a path with special characters.

**No clips were generated** — The LLM may have returned unusable JSON for every
chunk (warnings are logged per chunk). Re-run with `--resume` after checking
your key/quota, or lower `llm_chunk_minutes` so chunks are smaller.

## Development

```bash
python -m pytest tests/ -q     # 97 tests, LLM + pipeline mocked in API tests, no network
```

Layout:

```
main.py                    CLI entry point (wraps pipeline/runner.py)
pipeline/
  runner.py                shared orchestration: download → … → render + progress
  download.py              yt-dlp / local file + audio extraction
  transcribe.py            faster-whisper (int8, CPU, word timestamps)
  highlights.py            LLM adapters, parsing, snapping, dedupe
  captions.py              ASS karaoke caption builder
  render.py                FFmpeg vertical renderer
  utils.py                 config, logging, ffprobe helpers
prompts/
  highlight_prompt.txt     the LLM prompt (edit freely)
app/                       Phase 2 API
  main.py                  FastAPI routes + lifespan (init db, stale-job recovery)
  db.py                    sqlite3 persistence (storage/clipper.db)
  models.py                pydantic schemas
  worker.py                single-worker queue (one job at a time)
storage/
  uploads/                 uploaded videos
  projects/<id>/           per-project working files and clips
static/                    empty for now (UI comes in Phase 3)
workdir/<run_id>/          per-run cache (CLI)
output/                    final clips (CLI)
tests/                     pytest suite
```
