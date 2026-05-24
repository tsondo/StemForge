"""Pluggable transcription engines for StemForge.

Each engine implements the :class:`TranscriptionEngine` protocol and is
registered in :data:`ENGINES` below.  The :class:`TranscribePipeline`
selects an engine by ID at ``configure()`` time.
"""
from __future__ import annotations

from .qwen_engine import QwenEngine
from .types import (
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptionSegment,
    WordTiming,
)
from .whisper_engine import WhisperEngine

# Maps engine_id → engine class.  Engine IDs are flat strings, distinct
# from model_ids in models/registry.py.
ENGINES: dict[str, type] = {
    "whisper": WhisperEngine,
    "qwen": QwenEngine,
}

__all__ = [
    "ENGINES",
    "QwenEngine",
    "TranscriptionEngine",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperEngine",
    "WordTiming",
]
