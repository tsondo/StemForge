"""Pluggable transcription engines for StemForge.

Each engine implements the :class:`TranscriptionEngine` protocol and is
registered in :data:`ENGINES` below.  The :class:`TranscribePipeline`
selects an engine by ID at ``configure()`` time.
"""
from __future__ import annotations

from .qwen3_asr_engine import Qwen3AsrEngine
from .types import (
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptionSegment,
    WordTiming,
)
from .whisper_engine import WhisperEngine

# Maps engine_id → engine class.  Engine IDs are flat strings, distinct
# from model_ids in models/registry.py.
ENGINES: dict[str, type[TranscriptionEngine]] = {
    "whisper": WhisperEngine,
    "qwen3-asr": Qwen3AsrEngine,
}

__all__ = [
    "ENGINES",
    "Qwen3AsrEngine",
    "TranscriptionEngine",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperEngine",
    "WordTiming",
]
