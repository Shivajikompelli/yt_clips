"""Tests for ASS timestamp formatting, word shifting and dialogue building."""
from __future__ import annotations

import re

import pytest

from pipeline.captions import (
    build_ass,
    escape_ass_text,
    format_ass_time,
    group_words,
    shift_words,
)
from pipeline.utils import CaptionStyle

STYLE = CaptionStyle()


def make_words(spec: list[tuple[float, float, str]]) -> list[dict]:
    return [{"start": s, "end": e, "word": w} for s, e, w in spec]


class TestFormatAssTime:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "0:00:00.00"),
            (1.0, "0:00:01.00"),
            (61.246, "0:01:01.25"),      # rounds to centiseconds
            (3661.5, "1:01:01.50"),      # hours
            (-5.0, "0:00:00.00"),        # clamped
        ],
    )
    def test_format(self, seconds, expected):
        assert format_ass_time(seconds) == expected


class TestEscape:
    def test_override_braces_escaped(self):
        assert escape_ass_text("{\\b1}evil") == "(\\b1)evil"

    def test_newlines_flattened(self):
        assert escape_ass_text("two\nlines") == "two lines"


class TestShiftWords:
    def test_shifts_and_clips_window(self):
        words = make_words([
            (9.6, 10.3, "in"),      # straddles clip start -> clamped to 0
            (10.5, 11.5, "mid"),    # fully inside
            (11.8, 12.4, "edge"),   # straddles clip end -> clamped
            (13.0, 14.0, "out"),    # outside
        ])
        shifted = shift_words(words, clip_start=10.0, clip_end=12.0)
        assert [w["word"] for w in shifted] == ["in", "mid", "edge"]
        assert shifted[0]["start"] == 0.0
        assert shifted[1]["end"] == 1.5
        assert shifted[2]["end"] == 2.0

    def test_no_words_inside(self):
        words = make_words([(0.0, 1.0, "a")])
        assert shift_words(words, 10.0, 20.0) == []


class TestGroupWords:
    def test_max_words_per_line(self):
        words = make_words([(i * 0.5, i * 0.5 + 0.4, f"w{i}") for i in range(9)])
        lines = group_words(words)
        assert all(len(line["words"]) <= 4 for line in lines)
        assert len(lines) == 3

    def test_split_on_gap(self):
        words = make_words([(0.0, 0.4, "hi"), (5.0, 5.4, "again")])
        assert len(group_words(words)) == 2

    def test_monotonic_lines(self):
        words = make_words([(i * 0.5, i * 0.5 + 0.4, f"w{i}") for i in range(12)])
        lines = group_words(words)
        for prev, cur in zip(lines, lines[1:]):
            assert cur["start"] >= prev["end"]


class TestBuildAss:
    def dialogue_lines(self, ass: str) -> list[str]:
        return [l for l in ass.splitlines() if l.startswith("Dialogue:")]

    def test_contains_header_and_events(self):
        words = make_words([(10.0, 10.4, "hello"), (10.5, 11.0, "world")])
        ass = build_ass(words, clip_start=10.0, clip_end=20.0, style=STYLE)
        assert "PlayResX: 1080" in ass
        assert "PlayResY: 1920" in ass
        assert len(self.dialogue_lines(ass)) == 1

    def test_times_shifted_to_zero(self):
        words = make_words([(10.0, 10.4, "hello"), (10.5, 11.0, "world")])
        ass = build_ass(words, clip_start=10.0, clip_end=20.0, style=STYLE)
        event = self.dialogue_lines(ass)[0]
        start_ts = event.split(",")[1]
        assert start_ts == "0:00:00.00"

    def test_karaoke_tags_present(self):
        words = make_words([(10.0, 10.4, "hello"), (10.5, 11.0, "world")])
        ass = build_ass(words, clip_start=10.0, clip_end=20.0, style=STYLE)
        event = self.dialogue_lines(ass)[0]
        assert event.count("\\k") == 2

    def test_no_consecutive_overlap(self):
        # 9 fast words -> 3 lines; their display windows must not overlap
        words = make_words([(10.0 + i * 0.5, 10.0 + i * 0.5 + 0.4, f"w{i}") for i in range(9)])
        ass = build_ass(words, clip_start=10.0, clip_end=20.0, style=STYLE)

        def secs(ts: str) -> float:
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        prev_end = -1.0
        for event in self.dialogue_lines(ass):
            parts = event.split(",")
            start, end = secs(parts[1]), secs(parts[2])
            assert start >= prev_end
            prev_end = end

    def test_braces_in_words_sanitized(self):
        words = make_words([(10.0, 10.4, "{weird}")])
        ass = build_ass(words, clip_start=10.0, clip_end=20.0, style=STYLE)
        assert "(weird)" in self.dialogue_lines(ass)[0]
        assert "{\\{" not in self.dialogue_lines(ass)[0]

    def test_long_word_gets_smaller_font(self):
        words = make_words([(10.0, 10.4, "supercalifragilisticexpialidocious")])
        ass = build_ass(words, clip_start=10.0, clip_end=20.0, style=STYLE)
        event = self.dialogue_lines(ass)[0]
        m = re.search(r"\\fs(\d+)", event)
        assert m is not None and int(m.group(1)) < STYLE.font_size
