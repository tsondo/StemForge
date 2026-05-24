"""Shared types for transcription engines.

Each engine returns a :class:`TranscriptionResult` containing zero or
more :class:`TranscriptionSegment` items.  Word timestamps are optional
and indicated by ``has_word_timestamps`` — Whisper engines populate
``segments[].words``; Qwen produces a single segment with an empty
``words`` list and ``has_word_timestamps=False``.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class WordTiming:
    word: str
    start: float            # seconds
    end: float              # seconds
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    start: float            # seconds
    end: float              # seconds
    text: str
    words: list[WordTiming] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str               # full concatenated transcript
    language: str | None    # ISO-639-1, e.g. "en"
    segments: list[TranscriptionSegment]
    has_word_timestamps: bool
    engine_id: str          # which engine produced this result
    model_id: str           # specific model variant


@runtime_checkable
class TranscriptionEngine(Protocol):
    """Common interface for all transcription backends."""

    engine_id: str
    model_id: str
    supports_word_timestamps: bool
    requires_gpu: bool

    def load(self) -> None:
        ...

    def transcribe(
        self,
        audio_path: pathlib.Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        ...

    def clear(self) -> None:
        ...
