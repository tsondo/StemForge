"""Tests for utils.midi_io.flatten_to_format0()."""
from __future__ import annotations

import mido
import pytest

from utils.midi_io import flatten_to_format0, merge_tracks, read_midi, write_midi


@pytest.fixture
def multitrack_path(tmp_path):
    """A format-1 SMF with a melodic track and a drum track."""
    pm = merge_tracks({
        "vocals": [(0.0, 1.0, 60, 100), (1.0, 2.0, 64, 90)],
        "drums": [(0.0, 0.05, 35, 110), (0.5, 0.55, 38, 105)],
    })
    return write_midi(pm, tmp_path / "multi.mid")


def test_flatten_produces_format0(multitrack_path) -> None:
    flatten_to_format0(multitrack_path)
    mid = mido.MidiFile(str(multitrack_path))
    assert mid.type == 0
    assert len(mid.tracks) == 1


def test_flatten_preserves_notes(multitrack_path) -> None:
    before = read_midi(multitrack_path)
    notes_before = sorted(
        (n.pitch, round(n.start, 3)) for inst in before.instruments for n in inst.notes
    )
    flatten_to_format0(multitrack_path)
    after = read_midi(multitrack_path)
    notes_after = sorted(
        (n.pitch, round(n.start, 3)) for inst in after.instruments for n in inst.notes
    )
    assert notes_after == notes_before


def test_flatten_preserves_drum_channel(multitrack_path) -> None:
    flatten_to_format0(multitrack_path)
    mid = mido.MidiFile(str(multitrack_path))
    drum_notes = [
        msg for msg in mid.tracks[0]
        if msg.type == "note_on" and msg.channel == 9 and msg.velocity > 0
    ]
    assert len(drum_notes) == 2


def test_flatten_idempotent_on_format0(multitrack_path) -> None:
    flatten_to_format0(multitrack_path)
    first = multitrack_path.read_bytes()
    flatten_to_format0(multitrack_path)
    assert multitrack_path.read_bytes() == first
