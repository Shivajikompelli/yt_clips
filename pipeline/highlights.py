"""Stage 3: find highlight clips with an LLM and post-process them."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .utils import read_json, write_json

log = logging.getLogger("clipper.highlights")

CLIPS_NAME = "clips.json"

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "highlight_prompt.txt"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={api_key}"
)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

REQUEST_TIMEOUT = 120


class LLMError(RuntimeError):
    """LLM call failed after retries, or returned unusable output."""


@dataclass
class Highlight:
    start: float
    end: float
    title: str
    score: int
    reason: str = ""


# ---------------------------------------------------------------- LLM providers


class LLMClient:
    """Minimal provider interface. Concrete adapters implement complete()."""

    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", max_retries: int = 3):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Copy .env.example to .env and add a key.")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries

    def complete(self, prompt: str) -> str:
        url = GEMINI_URL.format(model=self.model, api_key=self.api_key)
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
        }
        return _post_json(url, payload, self.max_retries, _extract_gemini_text)


class GroqClient(LLMClient):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile", max_retries: int = 3):
        if not api_key:
            raise LLMError("GROQ_API_KEY is not set. Copy .env.example to .env and add a key.")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 2048,
        }
        return _post_json(
            GROQ_URL,
            payload,
            self.max_retries,
            lambda data: data["choices"][0]["message"]["content"],
            headers={"Authorization": f"Bearer {self.api_key}"},
        )


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def complete(self, prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=300,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed (is it running at {self.host}?): {exc}") from exc
        return resp.json().get("response", "")


def _redact(text: str) -> str:
    """Mask API keys that may appear inside error text/URLs."""
    return re.sub(r"([?&]key=)[^&\s]+", r"\1***", text)


# status codes worth retrying with backoff (rate limits + transient overload)
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class _Transient(Exception):
    """A retryable HTTP-level failure."""


def _post_json(url: str, payload: dict, max_retries: int, extract, headers=None) -> str:
    """POST JSON with exponential backoff on 429/5xx, return extracted text."""
    delay = 4.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers or {}, timeout=REQUEST_TIMEOUT)
            if resp.status_code in _RETRYABLE_STATUSES:
                raise _Transient(f"HTTP {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:  # e.g. 401/403/404: retrying will not help
                raise LLMError(_redact(f"LLM HTTP {resp.status_code}: {resp.text[:300]}"))
            return extract(resp.json())
        except _Transient as exc:
            if attempt == max_retries:
                raise LLMError(
                    f"LLM unavailable after {max_retries} retries: {_redact(str(exc))}"
                ) from exc
            log.warning("transient error, backing off %.1fs (attempt %d/%d): %s",
                        delay, attempt, max_retries, _redact(str(exc)))
            time.sleep(delay)
            delay *= 2
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise LLMError(
                    f"LLM request failed after {max_retries} attempts: {_redact(str(exc))}"
                ) from exc
            log.warning("request error, retrying in %.1fs: %s", delay, _redact(str(exc)))
            time.sleep(delay)
            delay *= 2
    raise LLMError("unreachable")


def _extract_gemini_text(data: dict) -> str:
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError) as exc:
        raise LLMError(f"unexpected Gemini response shape: {json.dumps(data)[:300]}") from exc


def make_llm_client(cfg) -> LLMClient:
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        return GeminiClient(
            os.environ.get("GEMINI_API_KEY", ""),
            # "flash-lite-latest" is an alias that always resolves to the
            # current free-tier lite model: least rate-limited/loaded tier,
            # ample for strict-JSON clip picking. Override with GEMINI_MODEL.
            model=os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest"),
            max_retries=cfg.llm_max_retries,
        )
    if provider == "groq":
        return GroqClient(
            os.environ.get("GROQ_API_KEY", ""),
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            max_retries=cfg.llm_max_retries,
        )
    if provider == "ollama":
        return OllamaClient(
            model=os.environ.get("OLLAMA_MODEL", "llama3.1"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    raise LLMError(f"unknown LLM_PROVIDER '{provider}' (use gemini|groq|ollama)")


# ---------------------------------------------------------------- parsing


def parse_llm_clips(raw: str) -> list[Highlight]:
    """Parse the LLM response into validated Highlights.

    Strips markdown fences, extracts the first JSON array, validates every field.
    Raises ValueError on unusable output.
    """
    text = _strip_fences(raw)
    block = _extract_json_array(text)
    try:
        items = json.loads(block)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("expected a JSON array of clips")
    return [validate_highlight(item) for item in items]


def _strip_fences(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    return text


def _extract_json_array(text: str) -> str:
    """Return the most likely JSON array span in text.

    Prose brackets (e.g. citations "[1]") can produce false spans, so all
    balanced spans are scanned and the first one that parses to a list of
    objects (a clip array) wins; otherwise the first parseable list is used.
    """
    first_parseable: str | None = None
    pos = 0
    while True:
        start = text.find("[", pos)
        if start == -1:
            break
        span = _balanced_span(text, start)
        if span is not None:
            try:
                parsed = json.loads(span)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                if parsed and isinstance(parsed[0], dict):
                    return span
                if first_parseable is None:
                    first_parseable = span
        pos = start + 1
    if first_parseable is not None:
        return first_parseable
    raise ValueError("no JSON array found in response")


def _balanced_span(text: str, start: int) -> str | None:
    """Return text[start:end+1] of the balanced bracket span, if parseable."""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return candidate
    return None


def validate_highlight(item) -> Highlight:
    """Validate one clip dict, coercing types; raises ValueError with detail."""
    if not isinstance(item, dict):
        raise ValueError(f"clip is not an object: {item!r}")
    missing = [k for k in ("start", "end", "title", "score") if k not in item]
    if missing:
        raise ValueError(f"clip missing fields {missing}: {item!r}")
    start, end = _to_float(item["start"]), _to_float(item["end"])
    score = _to_int(item["score"])
    title = str(item["title"]).strip()
    if not title:
        raise ValueError(f"clip has empty title: {item!r}")
    if not (1 <= score <= 10):
        raise ValueError(f"score out of range 1-10: {item!r}")
    if end <= start:
        raise ValueError(f"end must be after start: {item!r}")
    return Highlight(start=start, end=end, title=title, score=score,
                     reason=str(item.get("reason", "")).strip())


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected a number, got {value!r}") from exc


def _to_int(value) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected an integer, got {value!r}") from exc


# ---------------------------------------------------------------- transcript handling


def build_timed_lines(segments: list[dict]) -> list[str]:
    """Format transcript segments as '[start-end] text' lines."""
    return [f"[{seg['start']}-{seg['end']}] {seg['text']}" for seg in segments]


def chunk_lines(lines: list[str], seconds: float, overlap: float = 10.0) -> list[list[str]]:
    """Group lines into chunks of about `seconds`, with a small time overlap."""
    if not lines:
        return []
    chunks: list[list[str]] = []
    current: list[str] = []
    chunk_start = 0.0

    def line_start(line: str) -> float:
        return float(line.split("]")[0].strip("[").split("-")[0])

    def line_end(line: str) -> float:
        return float(line.split("]")[0].strip("[").split("-")[1])

    for line in lines:
        if current and line_end(line) - chunk_start > seconds:
            chunks.append(current)
            # start next chunk with the overlap tail of the previous one
            current = [l for l in current if line_end(l) > chunk_start + seconds - overlap]
            chunk_start = line_start(current[0]) if current else line_start(line)
        current.append(line)
    if current:
        chunks.append(current)
    return chunks


def build_prompt(chunk: list[str], cfg) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    # token replacement (not str.format) so the prompt file may contain JSON braces
    header = template.replace(
        "{min_seconds}", str(int(cfg.min_clip_seconds))
    ).replace("{max_seconds}", str(int(cfg.max_clip_seconds)))
    return header + "\n" + "\n".join(chunk)


# ---------------------------------------------------------------- post-processing

# Common abbreviations whose period must not be read as a sentence end.
_NON_TERMINAL = {"mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "e.g.", "i.e."}


def _ends_sentence(word: str) -> bool:
    """True if a whisper word token ends a sentence (. ? ! or ellipsis)."""
    token = word.strip().lower()
    if token in _NON_TERMINAL:
        return False
    return token.endswith((".", "?", "!", "…"))


def sentence_spans(segments: list[dict]) -> list[tuple[float, float]]:
    """(start, end) spans of complete sentences, derived from word punctuation.

    Whisper word tokens carry their trailing punctuation, so a sentence ends
    where a word ends with . ? ! or ellipsis. Spans are non-overlapping and
    ordered; the gap after a sentence is natural silence, safe to cut on.
    """
    spans: list[tuple[float, float]] = []
    sent_start: float | None = None
    sent_end: float | None = None
    for seg in segments:
        for w in seg.get("words", []):
            try:
                w_start, w_end = float(w["start"]), float(w["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if sent_start is None:
                sent_start = w_start
            sent_end = w_end
            if _ends_sentence(str(w.get("word", ""))):
                spans.append((sent_start, sent_end))
                sent_start = None
                sent_end = None
    if sent_start is not None and sent_end is not None:
        spans.append((sent_start, sent_end))  # trailing words without punctuation
    return spans


def snap_to_segments(highlight: Highlight, segments: list[dict], cfg) -> Highlight:
    """Align clip edges to complete sentences so clips are self-contained.

    LLM timestamps are approximate; snapping to the nearest segment boundary
    routinely chopped off the setup or the payoff. Both edges now lock to
    sentence boundaries (punctuation from word timestamps): the opening
    sentence is always complete, the closing sentence always lands its point,
    and short clips grow outward whole-sentence at a time to reach the minimum.
    """
    if not segments:
        return highlight
    spans = sentence_spans(segments)
    if not spans:
        # transcript has no word punctuation: keep the LLM's raw window
        return highlight

    def clip_start(t: float) -> float:
        for s, e in spans:
            if s <= t < e:  # mid-sentence: back up to its beginning
                return s
        before = [s for s, _ in spans if s < t]  # on a boundary: next sentence owns t
        if before:
            return before[-1]
        return spans[0][0]  # t before all speech

    def clip_end(t: float) -> float:
        for s, e in spans:
            if s < t <= e:  # mid-sentence: carry through to its end
                return e
        after = [e for _, e in spans if e > t]  # on a boundary: previous payoff owns t
        if after:
            return after[0]
        return spans[-1][1]  # t beyond all speech

    start = clip_start(highlight.start)
    end = clip_end(highlight.end)

    # grow outward whole sentences (whichever side is cheaper) until min length
    later = [e for _, e in spans if e > end]
    earlier = [s for s, _ in spans if s < start]
    i = j = 0
    while end - start < cfg.min_clip_seconds and (i < len(later) or j < len(earlier)):
        grow_end = later[i] if i < len(later) else None
        grow_start = earlier[j] if j < len(earlier) else None
        add_end = grow_end - end if grow_end is not None else float("inf")
        add_start = start - grow_start if grow_start is not None else float("inf")
        if add_end <= add_start:
            end = grow_end
            i += 1
        else:
            start = grow_start
            j += 1

    # trim to max: drop trailing sentences first (keeps the hook), then leading
    if end - start > cfg.max_clip_seconds:
        fitting = [e for _, e in spans if start < e <= start + cfg.max_clip_seconds]
        end = max(fitting) if fitting else start + cfg.max_clip_seconds
        if end - start > cfg.max_clip_seconds:
            heads = [s for s, _ in spans
                     if start < s and end - s <= cfg.max_clip_seconds
                     and end - s >= cfg.min_clip_seconds]
            if heads:
                start = min(heads)

    return Highlight(start=round(start, 3), end=round(end, 3),
                     title=highlight.title, score=highlight.score, reason=highlight.reason)


def drop_overlaps(highlights: list[Highlight]) -> list[Highlight]:
    """Keep higher-scored clips when intervals overlap. Assumes sorted by score desc."""
    kept: list[Highlight] = []
    for clip in sorted(highlights, key=lambda h: h.score, reverse=True):
        if all(clip.end <= k.start or clip.start >= k.end for k in kept):
            kept.append(clip)
    return kept


def select_clips(raw_clips: list[Highlight], segments: list[dict], cfg) -> list[Highlight]:
    """Snap, enforce duration, dedupe overlaps, sort by score, keep top N."""
    snapped = [snap_to_segments(c, segments, cfg) for c in raw_clips]
    # a clip snapped against the video tail may be impossible to extend to the
    # minimum length; keep only clips that actually satisfy the bounds
    snapped = [c for c in snapped if c.end - c.start >= cfg.min_clip_seconds - 0.01]
    no_dupes = drop_overlaps(snapped)
    top = sorted(no_dupes, key=lambda h: (-h.score, h.start))[: cfg.clips_to_generate]
    return sorted(top, key=lambda h: h.start)


# ---------------------------------------------------------------- entry point


def find_highlights(transcript: dict, run_dir: Path, cfg, client: LLMClient | None = None) -> list[Highlight]:
    """Run the LLM over transcript chunks and return final selected clips.

    Caches clips.json; on resume the cached file is returned as-is.
    """
    cache = run_dir / CLIPS_NAME
    if cache.exists():
        log.info("loading cached %s", CLIPS_NAME)
        return [Highlight(**item) for item in read_json(cache)]

    client = client or make_llm_client(cfg)
    segments = transcript.get("segments", [])
    lines = build_timed_lines(segments)
    chunks = chunk_lines(lines, cfg.llm_chunk_minutes * 60)

    collected: list[Highlight] = []
    for i, chunk in enumerate(chunks, 1):
        prompt = build_prompt(chunk, cfg)
        log.info("LLM chunk %d/%d (%.0f s of transcript)", i, len(chunks),
                 cfg.llm_chunk_minutes * 60)
        try:
            highlights = parse_llm_clips(client.complete(prompt))
        except (LLMError, ValueError) as exc:
            log.warning("skipping chunk %d: %s", i, exc)
            continue
        log.info("chunk %d: %d candidate clip(s)", i, len(highlights))
        collected.extend(highlights)

    clips = select_clips(collected, segments, cfg)
    if not clips:
        # do not cache empty results: the next run should retry the LLM
        cache.unlink(missing_ok=True)
        log.warning("no usable clips were found (not caching; re-run to retry)")
        return []
    payload = [
        {"start": c.start, "end": c.end, "title": c.title, "score": c.score, "reason": c.reason}
        for c in clips
    ]
    write_json(cache, payload)
    log.info("wrote %s (%d clips)", CLIPS_NAME, len(payload))
    return clips
