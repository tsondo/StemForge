"""Tests for drum stem routing in MidiPipeline.run()."""
from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

from pipelines.midi_pipeline import MidiConfig, MidiPipeline


def _pipeline_with_mock_loader() -> tuple[MidiPipeline, MagicMock]:
    """MidiPipeline wired to a mock loader, bypassing load_model()."""
    pipeline = MidiPipeline()
    pipeline.configure(MidiConfig())
    loader = MagicMock()
    loader.convert_drum_to_midi.return_value = [(0.1, 0.16, 35, 100)]
    loader.convert_audio_to_midi.return_value = [(0.1, 0.4, 60, 80)]
    loader.convert_vocal_to_midi.return_value = ([(0.1, 0.5, 60, 80)], [])
    pipeline._loader = loader
    pipeline.is_loaded = True
    return pipeline, loader


def _stem(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    path = tmp_path / name
    path.write_bytes(b"\x00")
    return path


def test_drums_label_routes_to_adt(tmp_path) -> None:
    pipeline, loader = _pipeline_with_mock_loader()
    pipeline.run({"drums": _stem(tmp_path, "drums.wav")})
    assert loader.convert_drum_to_midi.called
    assert not loader.convert_audio_to_midi.called


def test_roformer_drum_label_routes_to_adt(tmp_path) -> None:
    pipeline, loader = _pipeline_with_mock_loader()
    pipeline.run({"Drums & percussion": _stem(tmp_path, "dp.wav")})
    assert loader.convert_drum_to_midi.called
    assert not loader.convert_audio_to_midi.called


def test_bass_label_still_uses_basicpitch(tmp_path) -> None:
    pipeline, loader = _pipeline_with_mock_loader()
    pipeline.run({"bass": _stem(tmp_path, "bass.wav")})
    assert loader.convert_audio_to_midi.called
    assert not loader.convert_drum_to_midi.called


def test_vocals_label_still_uses_vocal_path(tmp_path) -> None:
    pipeline, loader = _pipeline_with_mock_loader()
    pipeline.run({"vocals": _stem(tmp_path, "vocals.wav")})
    assert loader.convert_vocal_to_midi.called
    assert not loader.convert_drum_to_midi.called


def test_drum_stem_midi_is_drum(tmp_path) -> None:
    pipeline, _ = _pipeline_with_mock_loader()
    result = pipeline.run({"drums": _stem(tmp_path, "drums.wav")})
    assert result.stem_midi_data["drums"].instruments[0].is_drum is True


def test_non_drum_stem_midi_not_drum(tmp_path) -> None:
    pipeline, _ = _pipeline_with_mock_loader()
    result = pipeline.run({"bass": _stem(tmp_path, "bass.wav")})
    assert result.stem_midi_data["bass"].instruments[0].is_drum is False


def test_drum_model_evicted_after_run(tmp_path) -> None:
    pipeline, loader = _pipeline_with_mock_loader()
    pipeline.run({"drums": _stem(tmp_path, "drums.wav")})
    assert loader.evict_drum_model.called


def test_drum_model_not_evicted_without_drums(tmp_path) -> None:
    pipeline, loader = _pipeline_with_mock_loader()
    pipeline.run({"bass": _stem(tmp_path, "bass.wav")})
    assert not loader.evict_drum_model.called
