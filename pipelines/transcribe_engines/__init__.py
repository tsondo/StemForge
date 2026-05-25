"""Pluggable transcription engines for StemForge.

Each engine implements the :class:`TranscriptionEngine` protocol and is
registered in :data:`ENGINES` below.  The :class:`TranscribePipeline`
selects an engine by ID at ``configure()`` time.
"""
from __future__ import annotations

from .qwen3_asr_engine import Qwen3AsrEngine
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
#
# Addendum 8 staged migration: qwen3-asr is added alongside qwen so users
# can A/B or fall back if validation surfaces a regression.  Stage 2
# removes the qwen entry once qwen3-asr is validated on the test stem.
ENGINES: dict[str, type[TranscriptionEngine]] = {
    "whisper": WhisperEngine,
    "qwen": QwenEngine,
    "qwen3-asr": Qwen3AsrEngine,
}

__all__ = [
    "ENGINES",
    "Qwen3AsrEngine",
    "QwenEngine",
    "TranscriptionEngine",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperEngine",
    "WordTiming",
]
