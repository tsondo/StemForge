"""Tests for POST /api/midi/events — Web MIDI event flattening.

Covers the flattening rules from docs/WEB_MIDI_OUT_SPEC.md:
event counts, ordering, overlap merging, clamping, zero-length drops,
multi-track shape, 404s, and the truncation-by-notes invariant.
"""

from __future__ import annotations

import pretty_midi
import pytest
from fastapi import HTTPException

from backend.api.midi import EventsRequest, get_midi_events, MAX_MIDI_OUT_EVENTS
from backend.services.session_store import SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_midi(notes, program=33, is_drum=False, name="bass"):
    """Build a single-instrument PrettyMIDI from (start, end, pitch, vel) tuples."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    for start, end, pitch, vel in notes:
        inst.notes.append(
            pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=end)
        )
    pm.instruments.append(inst)
    return pm


def session_with(stem_midi=None, merged=None):
    session = SessionStore(user="test")
    if stem_midi:
        session.stem_midi_data = stem_midi
    if merged is not None:
        session.merged_midi_data = merged
    return session


def events_of(result, track=0):
    return result["tracks"][track]["events"]


def call(session, label):
    return get_midi_events(EventsRequest(stem_label=label), session)


# ---------------------------------------------------------------------------
# Shape and counts
# ---------------------------------------------------------------------------

class TestShape:
    def test_two_events_per_note(self):
        pm = make_midi([(0.0, 0.5, 45, 96), (1.0, 1.5, 47, 88), (2.0, 2.5, 50, 70)])
        result = call(session_with({"bass": pm}), "bass")
        assert result["total_events"] == 6
        assert len(events_of(result)) == 6
        assert result["truncated"] is False
        assert result["label"] == "bass"
        assert result["duration"] == pytest.approx(2.5)

    def test_track_metadata(self):
        pm = make_midi([(0.0, 1.0, 60, 80)], program=33, name="bass")
        result = call(session_with({"bass": pm}), "bass")
        track = result["tracks"][0]
        assert track["name"] == "bass"
        assert track["program"] == 33
        assert track["is_drum"] is False

    def test_unnamed_instrument_falls_back_to_label(self):
        pm = make_midi([(0.0, 1.0, 60, 80)], name="")
        result = call(session_with({"other": pm}), "other")
        assert result["tracks"][0]["name"] == "other"


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_sorted_ascending_by_time(self):
        # Insert out of order; flattener must sort.
        pm = make_midi([(2.0, 2.5, 50, 70), (0.0, 0.5, 45, 96), (1.0, 1.5, 47, 88)])
        result = call(session_with({"bass": pm}), "bass")
        times = [e["t"] for e in events_of(result)]
        assert times == sorted(times)

    def test_off_before_on_at_equal_timestamps(self):
        # Two different pitches: one ends exactly where the other starts.
        pm = make_midi([(0.0, 1.0, 45, 96), (1.0, 2.0, 47, 88)])
        result = call(session_with({"bass": pm}), "bass")
        at_one = [e for e in events_of(result) if e["t"] == 1.0]
        assert [e["type"] for e in at_one] == ["off", "on"]

    def test_abutting_same_pitch_off_then_on(self):
        # Common after music21 Clean Up quantization: same pitch, end == start.
        pm = make_midi([(0.0, 1.0, 45, 96), (1.0, 2.0, 45, 88)])
        result = call(session_with({"bass": pm}), "bass")
        at_one = [e for e in events_of(result) if e["t"] == 1.0]
        assert [e["type"] for e in at_one] == ["off", "on"]
        assert all(e["note"] == 45 for e in at_one)


# ---------------------------------------------------------------------------
# Overlap merging
# ---------------------------------------------------------------------------

class TestOverlapMerge:
    def test_overlapping_same_pitch_truncates_earlier(self):
        # Second note of pitch 45 starts before the first ends: the first
        # note's off must land at the second note's start (abutting), never
        # interleaved on/on/off/off.
        pm = make_midi([(0.0, 1.5, 45, 96), (1.0, 2.0, 45, 88)])
        result = call(session_with({"bass": pm}), "bass")
        evs = events_of(result)
        assert len(evs) == 4
        assert [(e["t"], e["type"]) for e in evs] == [
            (0.0, "on"), (1.0, "off"), (1.0, "on"), (2.0, "off"),
        ]

    def test_identical_start_same_pitch_drops_earlier(self):
        # Same pitch, same start: the earlier (fully shadowed) note vanishes.
        pm = make_midi([(1.0, 1.5, 45, 96), (1.0, 2.0, 45, 88)])
        result = call(session_with({"bass": pm}), "bass")
        evs = events_of(result)
        assert len(evs) == 2
        assert [(e["t"], e["type"]) for e in evs] == [(1.0, "on"), (2.0, "off")]

    def test_overlapping_different_pitches_untouched(self):
        pm = make_midi([(0.0, 2.0, 45, 96), (1.0, 3.0, 47, 88)])
        result = call(session_with({"bass": pm}), "bass")
        assert len(events_of(result)) == 4
        offs = [e for e in events_of(result) if e["type"] == "off"]
        assert {e["t"] for e in offs} == {2.0, 3.0}

    def test_no_on_on_off_off_sequence_per_pitch(self):
        # Sounding-set safety: per pitch, ons and offs must strictly alternate.
        pm = make_midi([
            (0.0, 1.5, 45, 96), (1.0, 2.5, 45, 88), (2.0, 3.0, 45, 70),
        ])
        result = call(session_with({"bass": pm}), "bass")
        depth = 0
        for e in events_of(result):
            depth += 1 if e["type"] == "on" else -1
            assert depth in (0, 1)
        assert depth == 0


# ---------------------------------------------------------------------------
# Clamping and drops
# ---------------------------------------------------------------------------

class TestClamping:
    def test_note_on_velocity_clamped_to_floor_1(self):
        # Velocity 0 on a note-on would read as note-off on hardware.
        pm = make_midi([(0.0, 1.0, 60, 0)])
        result = call(session_with({"bass": pm}), "bass")
        on = [e for e in events_of(result) if e["type"] == "on"][0]
        assert on["vel"] == 1

    def test_off_events_carry_velocity_zero(self):
        pm = make_midi([(0.0, 1.0, 60, 100)])
        result = call(session_with({"bass": pm}), "bass")
        off = [e for e in events_of(result) if e["type"] == "off"][0]
        assert off["vel"] == 0

    def test_velocity_ceiling_127(self):
        pm = make_midi([(0.0, 1.0, 60, 200)])
        result = call(session_with({"bass": pm}), "bass")
        on = [e for e in events_of(result) if e["type"] == "on"][0]
        assert on["vel"] == 127

    def test_pitch_clamped_0_127(self):
        pm = make_midi([(0.0, 1.0, 200, 90), (2.0, 3.0, -5, 90)])
        result = call(session_with({"bass": pm}), "bass")
        pitches = {e["note"] for e in events_of(result)}
        assert pitches == {127, 0}

    def test_zero_length_notes_dropped(self):
        # pretty_midi validates end > start at construction, but session MIDI
        # can degrade to zero/negative length via mutation (e.g. quantization)
        # — build valid notes, then corrupt them the way the wild does.
        pm = make_midi([(0.0, 0.5, 60, 90), (1.0, 1.5, 62, 90), (2.0, 3.0, 64, 90)])
        pm.instruments[0].notes[0].end = 0.0    # zero-length
        pm.instruments[0].notes[1].end = 0.5    # negative-length
        result = call(session_with({"bass": pm}), "bass")
        assert result["total_events"] == 2
        assert {e["note"] for e in events_of(result)} == {64}


# ---------------------------------------------------------------------------
# Multi-track / merged
# ---------------------------------------------------------------------------

class TestMerged:
    def test_merged_returns_one_track_per_instrument(self):
        pm = pretty_midi.PrettyMIDI()
        for program, name in [(33, "bass"), (0, "piano")]:
            inst = pretty_midi.Instrument(program=program, name=name)
            inst.notes.append(
                pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0)
            )
            pm.instruments.append(inst)
        result = call(session_with(merged=pm), "merged")
        assert len(result["tracks"]) == 2
        assert [t["name"] for t in result["tracks"]] == ["bass", "piano"]
        assert result["total_events"] == 4

    def test_drum_instrument_reported(self):
        pm = make_midi([(0.0, 0.2, 36, 110)], is_drum=True, name="drums")
        result = call(session_with({"drums": pm}), "drums")
        assert result["tracks"][0]["is_drum"] is True


# ---------------------------------------------------------------------------
# 404s
# ---------------------------------------------------------------------------

class TestNotFound:
    def test_unknown_stem_404(self):
        with pytest.raises(HTTPException) as exc:
            call(session_with({"bass": make_midi([(0.0, 1.0, 60, 90)])}), "vocals")
        assert exc.value.status_code == 404

    def test_empty_merged_404(self):
        with pytest.raises(HTTPException) as exc:
            call(session_with(), "merged")
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def _dense_midi(self, note_count):
        # Non-overlapping notes, strictly increasing start times.
        return make_midi([
            (i * 0.1, i * 0.1 + 0.05, 40 + (i % 40), 90) for i in range(note_count)
        ])

    def test_under_cap_not_truncated(self):
        pm = self._dense_midi(100)
        result = call(session_with({"bass": pm}), "bass")
        assert result["truncated"] is False
        assert result["total_events"] == 200

    def test_over_cap_truncated_and_capped(self):
        pm = self._dense_midi(MAX_MIDI_OUT_EVENTS // 2 + 500)
        result = call(session_with({"bass": pm}), "bass")
        assert result["truncated"] is True
        assert result["total_events"] <= MAX_MIDI_OUT_EVENTS

    def test_truncation_keeps_earliest_notes(self):
        pm = self._dense_midi(MAX_MIDI_OUT_EVENTS // 2 + 500)
        result = call(session_with({"bass": pm}), "bass")
        evs = events_of(result)
        kept = MAX_MIDI_OUT_EVENTS // 2
        last_kept_start = (kept - 1) * 0.1
        assert max(e["t"] for e in evs) == pytest.approx(last_kept_start + 0.05)

    def test_truncation_invariant_no_hanging_ons(self):
        # The boundary case that matters: every emitted on has its off.
        pm = self._dense_midi(MAX_MIDI_OUT_EVENTS // 2 + 500)
        result = call(session_with({"bass": pm}), "bass")
        for track in result["tracks"]:
            sounding: set[tuple[int, int]] = set()
            depth: dict[int, int] = {}
            for e in track["events"]:
                delta = 1 if e["type"] == "on" else -1
                depth[e["note"]] = depth.get(e["note"], 0) + delta
                assert depth[e["note"]] >= 0
            assert all(v == 0 for v in depth.values()), "hanging note-on after truncation"

    def test_truncation_spans_tracks(self):
        # Notes dropped globally by start time, not per-track quotas.
        pm = pretty_midi.PrettyMIDI()
        early = pretty_midi.Instrument(program=0, name="early")
        late = pretty_midi.Instrument(program=1, name="late")
        half = MAX_MIDI_OUT_EVENTS // 2
        for i in range(half):
            early.notes.append(pretty_midi.Note(
                velocity=90, pitch=60, start=i * 0.01, end=i * 0.01 + 0.005))
        for i in range(500):
            late.notes.append(pretty_midi.Note(
                velocity=90, pitch=62, start=1000.0 + i, end=1000.5 + i))
        pm.instruments.extend([early, late])
        result = call(session_with(merged=pm), "merged")
        assert result["truncated"] is True
        # All late-track notes start after every early note — all dropped.
        assert result["tracks"][1]["events"] == []
        assert result["total_events"] == MAX_MIDI_OUT_EVENTS
