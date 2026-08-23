"""Tests for POST /api/midi/render — instrument handling and merged support.

The merged card previews through the soft synth like any other card, which
requires two things this endpoint did not originally do: resolve the
``merged`` label, and leave each instrument's own program alone so a
multi-instrument arrangement is not collapsed to a single voice.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pretty_midi
import pytest
from fastapi import HTTPException

from backend.api.midi import RenderRequest, render_midi_to_audio
from backend.services.session_store import SessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_multi_instrument():
    """Two instruments with distinct programs, one of them a drum kit."""
    pm = pretty_midi.PrettyMIDI()
    bass = pretty_midi.Instrument(program=33, is_drum=False, name="bass")
    bass.notes.append(pretty_midi.Note(velocity=90, pitch=40, start=0.0, end=0.5))
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.2))
    pm.instruments.append(bass)
    pm.instruments.append(drums)
    return pm


def session_with(stem_midi=None, merged=None):
    session = SessionStore(user="test")
    if stem_midi:
        session.stem_midi_data = stem_midi
    if merged is not None:
        session.merged_midi_data = merged
    return session


@pytest.fixture
def capture(tmp_path):
    """Patch FluidSynth and the output dir — no real synth, no disk churn.

    Yields a dict whose "instruments" key holds the (program, is_drum) pairs
    FluidSynth was actually handed, which is what these tests assert on.
    """
    captured = {}

    # A plain function, not a callable instance: only functions are
    # descriptors, so only a function gets `self` bound on attribute access.
    def fake_fluidsynth(self, fs=44100, **kwargs):
        captured["instruments"] = [
            (inst.program, inst.is_drum) for inst in self.instruments
        ]
        return np.zeros(int(fs * 0.1), dtype=np.float32)

    with patch.object(pretty_midi.PrettyMIDI, "fluidsynth", fake_fluidsynth), \
         patch("backend.api.midi.MIDI_DIR", tmp_path):
        yield captured


# ---------------------------------------------------------------------------
# Instrument preservation
# ---------------------------------------------------------------------------

def test_omitting_program_preserves_each_instrument(capture):
    """No program/is_drum given → the arrangement keeps its own voices."""
    session = session_with(merged=make_multi_instrument())
    render_midi_to_audio(RenderRequest(stem_label="merged"), session=session)
    assert capture["instruments"] == [(33, False), (0, True)]


def test_explicit_program_overrides_every_instrument(capture):
    """A stem card still forces its dropdown selection onto the render."""
    session = session_with(stem_midi={"bass": make_multi_instrument()})
    render_midi_to_audio(
        RenderRequest(stem_label="bass", program=81, is_drum=False), session=session
    )
    assert capture["instruments"] == [(81, False), (81, False)]


def test_program_and_is_drum_are_independent(capture):
    """is_drum can be set without disturbing each instrument's program."""
    session = session_with(stem_midi={"kit": make_multi_instrument()})
    render_midi_to_audio(
        RenderRequest(stem_label="kit", is_drum=True), session=session
    )
    assert capture["instruments"] == [(33, True), (0, True)]


def test_render_does_not_mutate_session_midi(capture):
    """Overrides act on a copy — the session's MIDI is untouched."""
    original = make_multi_instrument()
    session = session_with(stem_midi={"bass": original})
    before = [(i.program, i.is_drum) for i in original.instruments]
    render_midi_to_audio(
        RenderRequest(stem_label="bass", program=99, is_drum=True), session=session
    )
    after = [(i.program, i.is_drum) for i in session.stem_midi_data["bass"].instruments]
    assert after == before == [(33, False), (0, True)]


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------

def test_merged_label_resolves(capture):
    """'merged' is a valid render target, not just a stem name."""
    session = session_with(merged=make_multi_instrument())
    result = render_midi_to_audio(RenderRequest(stem_label="merged"), session=session)
    assert result["audio_path"].endswith(".wav")
    assert result["duration"] > 0


def test_merged_without_data_404s():
    session = session_with(stem_midi={"bass": make_multi_instrument()})
    with pytest.raises(HTTPException) as exc:
        render_midi_to_audio(RenderRequest(stem_label="merged"), session=session)
    assert exc.value.status_code == 404


def test_unknown_stem_404s():
    session = session_with(stem_midi={"bass": make_multi_instrument()})
    with pytest.raises(HTTPException) as exc:
        render_midi_to_audio(RenderRequest(stem_label="nope"), session=session)
    assert exc.value.status_code == 404
