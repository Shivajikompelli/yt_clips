"""Tests for LLM output parsing/validation, chunking, snapping and dedupe."""
from __future__ import annotations

import json

import pytest

from pipeline.highlights import (
    Highlight,
    _ends_sentence,
    _extract_json_array,
    build_timed_lines,
    chunk_lines,
    drop_overlaps,
    parse_llm_clips,
    select_clips,
    snap_to_segments,
    sentence_spans,
    validate_highlight,
)
from pipeline.utils import Config

# ---------------------------------------------------------------- fixtures


def make_cfg(**overrides) -> Config:
    cfg = Config()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_segment(i: int) -> dict:
    """One 5 s transcript segment whose words form a single sentence."""
    start, end = i * 5.0, (i + 1) * 5.0
    return {
        "start": start,
        "end": end,
        "text": f"point number {i} done.",
        "words": [
            {"start": start, "end": start + 3.0, "word": f"point{0}".replace("0", str(i))},
            {"start": start + 3.0, "end": start + 4.0, "word": "number"},
            {"start": start + 4.0, "end": end, "word": f"{i}."},
        ],
    }


SEGMENTS = [make_segment(i) for i in range(7)]  # sentences at (0,5) ... (30,35)


# ---------------------------------------------------------------- parsing


class TestParseLlmClips:
    def test_plain_array(self):
        raw = json.dumps([{"start": 1.0, "end": 30.0, "title": "T", "score": 7}])
        clips = parse_llm_clips(raw)
        assert len(clips) == 1
        assert clips[0] == Highlight(start=1.0, end=30.0, title="T", score=7, reason="")

    def test_fenced_json_block(self):
        inner = json.dumps([{"start": 1.0, "end": 30.0, "title": "T", "score": 7}])
        clips = parse_llm_clips(f"```json\n{inner}\n```")
        assert clips[0].title == "T"

    def test_fenced_without_language(self):
        inner = json.dumps([{"start": 1.0, "end": 30.0, "title": "T", "score": 7}])
        assert parse_llm_clips(f"```\n{inner}\n```")[0].score == 7

    def test_prose_around_array(self):
        inner = json.dumps([{"start": 1.0, "end": 30.0, "title": "T", "score": 7}])
        clips = parse_llm_clips(f"Here are the clips:\n{inner}\nHope this helps!")
        assert clips[0].end == 30.0

    def test_brackets_inside_strings(self):
        raw = '[{"start": 1.0, "end": 30.0, "title": "Top 3 [2026] tips", "score": 7}]'
        clips = parse_llm_clips(raw)
        assert clips[0].title == "Top 3 [2026] tips"

    def test_type_coercion(self):
        raw = '[{"start": "1.5", "end": "30.9", "title": 42, "score": "8.6"}]'
        clips = parse_llm_clips(raw)
        assert clips[0].start == 1.5
        assert clips[0].score == 9  # rounded

    @pytest.mark.parametrize(
        "raw",
        [
            "",                       # empty
            "no json here",           # no array
            "[{bad}",                 # unbalanced
            '{"start": 1.0}',         # object, not array
            '[{"end": 30.0}]',        # missing fields
            '[{"start": 30.0, "end": 1.0, "title": "T", "score": 5}]',   # end < start
            '[{"start": 1.0, "end": 30.0, "title": "", "score": 5}]',    # empty title
            '[{"start": 1.0, "end": 30.0, "title": "T", "score": 11}]',  # score too big
            '[{"start": 1.0, "end": 30.0, "title": "T", "score": 0}]',   # score too small
            '[{"start": null, "end": 30.0, "title": "T", "score": 5}]',  # null number
            '[]',                     # valid but empty
        ],
    )
    def test_malformed_rejected(self, raw):
        if raw == "[]":
            assert parse_llm_clips(raw) == []  # valid JSON, simply no clips
        else:
            with pytest.raises(ValueError):
                parse_llm_clips(raw)

    def test_clip_array_beats_citation_brackets(self):
        # prose citations like "[1]" must not swallow the real clip array
        text = 'Note: see [1] for methodology.\n'
        text += json.dumps([{"start": 1.0, "end": 30.0, "title": "T", "score": 7}])
        clips = parse_llm_clips(text)
        assert clips[0].title == "T"

    def test_nested_array_inside_object_is_ignored(self):
        text = 'metadata {"refs": [1, 2]} then '
        text += json.dumps([{"start": 1.0, "end": 30.0, "title": "T", "score": 7}])
        assert parse_llm_clips(text)[0].score == 7


class TestValidateHighlight:
    def test_extra_fields_ignored(self):
        h = validate_highlight({"start": 1, "end": 30, "title": "T", "score": 5, "junk": "x"})
        assert h.reason == ""

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="not an object"):
            validate_highlight([1, 2, 3])


# ---------------------------------------------------------------- chunking


class TestChunking:
    def test_timed_lines_format(self):
        lines = build_timed_lines([{"start": 123.4, "end": 130.2, "text": "hello"}])
        assert lines == ["[123.4-130.2] hello"]

    def test_short_transcript_single_chunk(self):
        lines = build_timed_lines(SEGMENTS)
        assert chunk_lines(lines, 900) == [lines]

    def test_split_with_overlap(self):
        lines = build_timed_lines(SEGMENTS)  # 35 s total
        chunks = chunk_lines(lines, 15, overlap=5)
        assert len(chunks) == 3
        # every chunk except the first starts before its predecessor ends (overlap)
        for prev, cur in zip(chunks, chunks[1:]):
            prev_end = float(prev[-1].split("]")[0].split("-")[1])
            cur_start = float(cur[0].split("]")[0].strip("[").split("-")[0])
            assert cur_start < prev_end

    def test_empty(self):
        assert chunk_lines([], 100) == []


# ---------------------------------------------------------------- snapping


class TestSentenceSpans:
    def test_single_sentence_per_segment(self):
        spans = sentence_spans(SEGMENTS)
        assert spans == [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0),
                         (20.0, 25.0), (25.0, 30.0), (30.0, 35.0)]

    def test_multi_sentence_segment_split(self):
        seg = {
            "start": 0.0, "end": 10.0, "text": "a b. c d?",
            "words": [
                {"start": 0.0, "end": 2.0, "word": "a"},
                {"start": 2.0, "end": 4.0, "word": "b."},   # sentence 1 ends
                {"start": 5.0, "end": 7.0, "word": "c"},
                {"start": 7.0, "end": 9.0, "word": "d?"},   # sentence 2 ends
            ],
        }
        assert sentence_spans([seg]) == [(0.0, 4.0), (5.0, 9.0)]

    def test_abbreviations_are_not_sentence_ends(self):
        seg = {
            "start": 0.0, "end": 10.0, "text": "dr. smith spoke here.",
            "words": [
                {"start": 0.0, "end": 1.0, "word": "Dr."},
                {"start": 1.0, "end": 4.0, "word": "Smith"},
                {"start": 4.0, "end": 7.0, "word": "spoke"},
                {"start": 7.0, "end": 9.0, "word": "here."},
            ],
        }
        assert sentence_spans([seg]) == [(0.0, 9.0)]

    def test_trailing_words_without_punctuation_form_final_span(self):
        seg = {
            "start": 0.0, "end": 10.0, "text": "one. two three",
            "words": [
                {"start": 0.0, "end": 2.0, "word": "one."},
                {"start": 3.0, "end": 5.0, "word": "two"},
                {"start": 6.0, "end": 9.0, "word": "three"},
            ],
        }
        assert sentence_spans([seg]) == [(0.0, 2.0), (3.0, 9.0)]

    def test_empty_and_malformed_words(self):
        assert sentence_spans([]) == []
        assert sentence_spans([{"start": 0.0, "end": 1.0, "text": "x"}]) == []
        seg = {
            "start": 0.0, "end": 4.0, "text": "x",
            "words": [
                {"start": "bad", "end": 1.0, "word": "x"},   # skipped
                {"start": 2.0, "end": 3.0, "word": "done."},
            ],
        }
        assert sentence_spans([seg]) == [(2.0, 3.0)]

    @pytest.mark.parametrize("word,expected", [
        ("done.", True), ("really?", True), ("wow!", True), ("wait…", True),
        ("Mr.", False), ("e.g.", False), ("hello", False), ("", False),
    ])
    def test_ends_sentence(self, word, expected):
        assert _ends_sentence(word) is expected


class TestSnapping:
    cfg = make_cfg(min_clip_seconds=5, max_clip_seconds=60)

    def test_snaps_inside_sentence_to_full_sentence(self):
        # LLM picked 6-14 (mid-sentence on both edges) -> whole sentences 5-15
        h = Highlight(start=6.0, end=14.0, title="T", score=5)
        snapped = snap_to_segments(h, SEGMENTS, self.cfg)
        assert (snapped.start, snapped.end) == (5.0, 15.0)

    def test_start_never_splits_a_sentence(self):
        h = Highlight(start=7.5, end=16.0, title="T", score=5)
        snapped = snap_to_segments(h, SEGMENTS, self.cfg)
        assert snapped.start == 5.0  # sentence (5,10) starts complete

    def test_end_always_lands_the_payoff(self):
        h = Highlight(start=5.0, end=17.0, title="T", score=5)
        snapped = snap_to_segments(h, SEGMENTS, self.cfg)
        assert snapped.end == 20.0  # sentence (15,20) is included whole

    def test_grows_by_whole_sentences_to_min(self):
        cfg = make_cfg(min_clip_seconds=12, max_clip_seconds=60)
        h = Highlight(start=0.0, end=5.0, title="T", score=5)
        snapped = snap_to_segments(h, SEGMENTS, cfg)
        assert snapped.end - snapped.start >= cfg.min_clip_seconds
        assert snapped.end in (10.0, 15.0)  # sentence boundary, not mid-speech

    def test_growth_prefers_cheaper_side(self):
        # end is 5 s short: extending end by one 5 s sentence beats a 10 s lead-in
        cfg = make_cfg(min_clip_seconds=10, max_clip_seconds=60)
        h = Highlight(start=5.0, end=10.0, title="T", score=5)
        snapped = snap_to_segments(h, SEGMENTS, cfg)
        assert (snapped.start, snapped.end) == (5.0, 15.0)

    def test_trims_long_clip_to_max_on_sentence_boundary(self):
        cfg = make_cfg(min_clip_seconds=5, max_clip_seconds=10)
        h = Highlight(start=0.0, end=35.0, title="T", score=5)
        snapped = snap_to_segments(h, SEGMENTS, cfg)
        assert snapped.end - snapped.start <= cfg.max_clip_seconds
        assert snapped.end == 10.0  # last sentence fully inside the window

    def test_no_sentence_punctuation_keeps_raw_window(self):
        bare = [{"start": 0.0, "end": 5.0, "text": "x"}]
        h = Highlight(start=3.0, end=9.0, title="T", score=5)
        assert snap_to_segments(h, bare, self.cfg) == h

    def test_no_segments_passthrough(self):
        h = Highlight(start=3.0, end=9.0, title="T", score=5)
        assert snap_to_segments(h, [], self.cfg) == h


# ---------------------------------------------------------------- dedupe / selection


class TestOverlaps:
    def test_keeps_higher_score(self):
        a = Highlight(start=0, end=20, title="A", score=5)
        b = Highlight(start=10, end=30, title="B", score=8)
        kept = drop_overlaps([a, b])
        assert [k.title for k in kept] == ["B"]

    def test_touching_intervals_kept(self):
        a = Highlight(start=0, end=20, title="A", score=5)
        b = Highlight(start=20, end=30, title="B", score=8)
        assert len(drop_overlaps([a, b])) == 2

    def test_disjoint_all_kept(self):
        a = Highlight(start=0, end=20, title="A", score=5)
        b = Highlight(start=25, end=30, title="B", score=8)
        c = Highlight(start=40, end=50, title="C", score=1)
        assert len(drop_overlaps([a, b, c])) == 3


class TestSelectClips:
    def test_top_n_sorted_by_start(self):
        cfg = make_cfg(min_clip_seconds=5, max_clip_seconds=60, clips_to_generate=2)
        raw = [
            Highlight(start=1, end=20, title="A", score=5),
            Highlight(start=6, end=25, title="B", score=9),   # overlaps A
            Highlight(start=30, end=34, title="C", score=8),  # too short, snaps longer
        ]
        clips = select_clips(raw, SEGMENTS, cfg)
        assert [c.title for c in clips] == ["B", "C"]
        assert clips[0].start < clips[1].start

    def test_snapped_clips_respect_bounds(self):
        cfg = make_cfg(min_clip_seconds=8, max_clip_seconds=15, clips_to_generate=5)
        raw = [Highlight(start=2, end=34, title="Long", score=9)]
        for clip in select_clips(raw, SEGMENTS, cfg):
            assert cfg.min_clip_seconds <= clip.end - clip.start <= cfg.max_clip_seconds

    def test_tail_clip_is_grown_not_dropped(self):
        # a late short pick now grows backward whole-sentences instead of being dropped
        cfg = make_cfg(min_clip_seconds=8, max_clip_seconds=60, clips_to_generate=5)
        raw = [Highlight(start=32.0, end=34.0, title="Tail", score=9)]
        clips = select_clips(raw, SEGMENTS, cfg)
        assert [c.title for c in clips] == ["Tail"]
        assert clips[0].end - clips[0].start >= cfg.min_clip_seconds
