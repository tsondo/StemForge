"""Qwen3-ASR engine for multilingual lyric transcription.

License: Apache 2.0 — see licenses/LICENSE-Qwen3-ASR and ACKNOWLEDGMENTS.md.

Replaces the Qwen2-Audio engine that previously occupied this slot. Qwen3-ASR
is purpose-built for ASR (including music/song recognition), handles long
audio internally without external chunking, and exposes dedicated parameters
for language and vocabulary biasing — removing the prompt-engineering layer
that Addenda 5-7 spent considerable effort iterating on.

API note (verified against qwen-asr 0.0.6 in-tree docstring):
    Qwen3ASRModel.transcribe(audio, context="", language=None, return_time_stamps=False)
    - `context` is the vocabulary-biasing kwarg (not `prompt`); it becomes the
      chat template's system message.
    - `audio` and `language` accept scalar or list; we pass scalar for the
      single-audio case.
    - The ASRTranscription result has only .text / .language / .time_stamps —
      no .duration field, so segment.end is left at 0.0 by design.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any, Literal

import torch

from utils.cache import get_model_cache_dir
from utils.errors import ModelLoadError, PipelineExecutionError

from .types import (
    TranscriptionResult,
    TranscriptionSegment,
)

log = logging.getLogger(__name__)


Variant = Literal["1.7b", "0.6b"]

_VARIANT_REPOS: dict[str, str] = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
}

# Display metadata for the engines endpoint (mirrors pattern from Addendum 2).
QWEN3_ASR_VARIANTS: dict[str, dict] = {
    "qwen3-asr-1.7b": {
        "display_name": "Qwen3-ASR 1.7B",
        "approx_vram_gb": 7,
        "description": "Best quality — recommended for music",
    },
    "qwen3-asr-0.6b": {
        "display_name": "Qwen3-ASR 0.6B",
        "approx_vram_gb": 3,
        "description": "Faster, slight quality trade-off",
    },
}


# Map ISO codes to Qwen3-ASR's expected language names.
# The toolkit accepts both forms; human-readable names are the documented default.
_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "yue": "Cantonese",
    "ar": "Arabic",
    "hi": "Hindi",
    "ru": "Russian",
    # Add more from the Qwen3-ASR supported-language list as the UI exposes them.
}


def _resolve_language(language: str | None) -> str | None:
    if not language:
        return None
    return _LANGUAGE_NAMES.get(language.lower(), language)


class Qwen3AsrEngine:
    """Qwen3-ASR engine. Replaces the Qwen2-Audio engine."""

    engine_id = "qwen3-asr"
    supports_word_timestamps = False   # Becomes True if Qwen3-ForcedAligner is wired in (future)
    requires_gpu = True

    def __init__(self, model_id: str = "qwen3-asr-1.7b") -> None:
        if model_id not in _VARIANT_REPOS:
            raise ValueError(
                f"Unknown qwen3-asr model_id {model_id!r}. "
                f"Available: {sorted(_VARIANT_REPOS)}"
            )
        self.model_id = model_id
        self._repo = _VARIANT_REPOS[model_id]
        self._model: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not torch.cuda.is_available():
            raise ModelLoadError(
                "Qwen3-ASR requires CUDA. Use a Whisper engine for CPU transcription.",
                model_name=self.model_id,
            )
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ModelLoadError(
                "qwen-asr package is required. Install via `uv add qwen-asr`.",
                model_name=self.model_id,
            ) from exc

        cache_dir = str(get_model_cache_dir("qwen3-asr"))
        log.info("Loading %s on CUDA…", self._repo)
        try:
            self._model = Qwen3ASRModel.from_pretrained(
                self._repo,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Qwen3-ASR: {exc}",
                model_name=self.model_id,
            ) from exc
        log.info("Qwen3-ASR ready (%s).", self.model_id)

    def transcribe(
        self,
        audio_path: pathlib.Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        self.load()

        lang = _resolve_language(language)

        # Qwen3-ASR's transcribe() handles long audio internally — no chunking
        # needed. The kwarg names below match qwen-asr 0.0.6's live signature:
        # context= (NOT prompt=) is the vocabulary-biasing slot; scalar audio
        # and language values work for the single-file case.
        try:
            kwargs: dict = {"audio": str(audio_path)}
            if lang:
                kwargs["language"] = lang
            if prompt:
                kwargs["context"] = prompt
            results = self._model.transcribe(**kwargs)
        except Exception as exc:
            raise PipelineExecutionError(
                f"Qwen3-ASR transcription failed: {exc}",
                pipeline_name="transcribe",
            ) from exc

        if not results:
            raise PipelineExecutionError(
                "Qwen3-ASR returned no results.",
                pipeline_name="transcribe",
            )

        # The toolkit returns one result per input audio.
        r = results[0]
        text = (r.text or "").strip()
        detected_language = getattr(r, "language", None) or lang

        # ASRTranscription has no duration field, so segment.end stays 0.0.
        # That's a known limitation; word-level timestamps require the
        # Qwen3-ForcedAligner-0.6B companion (deferred per handoff).
        duration = float(getattr(r, "duration", 0.0))
        segment = TranscriptionSegment(
            start=0.0, end=duration, text=text, words=[],
        )

        log.info(
            "Qwen3-ASR transcribed %s (%d chars output, lang=%s).",
            audio_path.name, len(text), detected_language,
        )

        return TranscriptionResult(
            text=text,
            language=detected_language,
            segments=[segment],
            has_word_timestamps=False,
            engine_id=self.engine_id,
            model_id=self.model_id,
        )

    def clear(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Qwen3-ASR engine cleared.")
