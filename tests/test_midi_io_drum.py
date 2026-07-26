"""Tests for the is_drum parameter in utils.midi_io.notes_to_midi()."""
from __future__ import annotations

import pytest

from utils.midi_io import DRUM_STEM_LABELS, notes_to_midi


def test_drum_instrument_on_channel_10() -> None:
    pm = notes_to_midi([(0.5, 1.5, 35, 100)], is_drum=True)
    assert pm.instruments[0].is_drum is True
    assert pm.instruments[0].name == "StemForge Drums"


def test_default_behavior_unchanged() -> None:
    for kwargs in ({}, {"is_drum": False}):
        pm = notes_to_midi([(0.5, 1.5, 60, 100)], **kwargs)
        inst = pm.instruments[0]
        assert inst.is_drum is False
        assert inst.program == 0
        assert inst.name == "StemForge"
        assert inst.notes[0].end == pytest.approx(1.5)


def test_drum_note_capped_at_60ms() -> None:
    pm = notes_to_midi([(1.0, 2.0, 35, 100)], is_drum=True)
    assert pm.instruments[0].notes[0].end == pytest.approx(1.06)


def test_short_drum_note_passthrough() -> None:
    pm = notes_to_midi([(1.0, 1.03, 35, 100)], is_drum=True)
    assert pm.instruments[0].notes[0].end == pytest.approx(1.03)


def test_degenerate_drum_note_filtered() -> None:
    pm = notes_to_midi([(1.0, 0.5, 35, 100)], is_drum=True)
    assert pm.instruments[0].notes == []


def test_is_drum_survives_file_roundtrip(tmp_path) -> None:
    import pretty_midi

    pm = notes_to_midi([(0.5, 1.5, 35, 100)], is_drum=True)
    out = tmp_path / "drums.mid"
    pm.write(str(out))
    reloaded = pretty_midi.PrettyMIDI(str(out))
    assert reloaded.instruments[0].is_drum is True


def test_drum_stem_labels_cover_known_separators() -> None:
    assert "drums" in DRUM_STEM_LABELS
    assert "Drums & percussion" in DRUM_STEM_LABELS
