"""Whisper transcription engine (wraps faster-whisper).

This is the ONLY place in StemForge that imports faster_whisper after
the refactor in Tasks 11 and 12.  Grep verification in Task 13.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch

from models.registry import get_spec, WhisperSpec
from utils.cache import get_model_cache_dir
from utils.errors import ModelLoadError, PipelineExecutionError

from .types import (
    TranscriptionResult,
    TranscriptionSegment,
    WordTiming,
)

log = logging.getLogger(__name__)


class WhisperEngine:
    engine_id = "whisper"
    supports_word_timestamps = True
    requires_gpu = False  # works on CPU

    def __init__(self, model_id: str = "whisper-base") -> None:
        spec = get_spec(model_id)
        if not isinstance(spec, WhisperSpec):
            raise ModelLoadError(
                f"{model_id!r} is not a Whisper model.", model_name=model_id,
            )
        self.model_id = model_id
        self._spec: WhisperSpec = spec
        self._model: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelLoadError(
                "faster-whisper is not installed.",
                model_name=self.model_id,
            ) from exc

        # Honor the spec's device field; "auto" defers to CUDA availability.
        # All current Whisper variants default to "cpu" because CTranslate2
        # ships CUDA-12 binaries and StemForge's venv runs CUDA 13.  When a
        # GPU-capable Whisper spec becomes practical, set its device="auto"
        # or "cuda" and this branch picks it up.
        spec_device = (self._spec.device or "cpu").lower()
        if spec_device == "cuda" or (spec_device == "auto" and torch.cuda.is_available()):
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = self._spec.compute_type or "int8"

        log.info(
            "Loading faster-whisper %r on %s (%s)…",
            self._spec.model_size, device, compute_type,
        )
        try:
            self._model = WhisperModel(
                self._spec.model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(get_model_cache_dir("whisper")),
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Whisper {self.model_id}: {exc}",
                model_name=self.model_id,
            ) from exc
        log.info("Whisper %s ready.", self.model_id)

    def transcribe(
        self,
        audio_path: pathlib.Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        self.load()
        try:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                word_timestamps=True,
                vad_filter=True,
                language=language,
                initial_prompt=prompt,
            )
        except Exception as exc:
            raise PipelineExecutionError(
                f"Whisper transcription failed for {audio_path.name}: {exc}",
                pipeline_name="transcribe",
            ) from exc

        segments: list[TranscriptionSegment] = []
        full_text_parts: list[str] = []
        for seg in segments_iter:
            words: list[WordTiming] = []
            if seg.words:
                for w in seg.words:
                    words.append(WordTiming(
                        word=str(w.word),
                        start=float(w.start),
                        end=float(w.end),
                        probability=float(w.probability) if w.probability is not None else None,
                    ))
            segments.append(TranscriptionSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=str(seg.text),
                words=words,
            ))
            full_text_parts.append(str(seg.text))

        return TranscriptionResult(
            text="".join(full_text_parts).strip(),
            language=str(info.language) if info.language else None,
            segments=segments,
            has_word_timestamps=True,
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
        log.info("Whisper engine cleared.")
