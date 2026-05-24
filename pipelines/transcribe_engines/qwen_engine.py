"""Qwen2-Audio engine for high-fidelity sung-lyrics transcription.

LICENSE NOTE: Apache 2.0.  See licenses/LICENSE-Qwen2-Audio and the
ACKNOWLEDGMENTS.md entry.  Re-verify the license before any future
upgrade — Qwen variants are not uniformly Apache 2.0 across sizes.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch

from utils.cache import get_model_cache_dir
from utils.errors import ModelLoadError, PipelineExecutionError

from .types import (
    TranscriptionResult,
    TranscriptionSegment,
)

log = logging.getLogger(__name__)

_QWEN_REPO = "Qwen/Qwen2-Audio-7B-Instruct"
_PROMPT = (
    "Transcribe the lyrics of this audio. Output only the lyrics text, "
    "preserving line breaks where the singer pauses. Do not add commentary, "
    "explanations, or section labels."
)


class QwenEngine:
    engine_id = "qwen"
    model_id = "qwen2-audio-7b-instruct"
    supports_word_timestamps = False
    requires_gpu = True

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str = "cuda",
    ) -> None:
        # model_id kwarg accepted for Protocol uniformity; ignored — Qwen
        # has exactly one variant in v1.  device may be "cuda" (default)
        # or "cuda:N" for multi-GPU scheduling.
        self._device = device
        self._model: Any | None = None
        self._processor: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not torch.cuda.is_available():
            raise ModelLoadError(
                "Qwen2-Audio requires CUDA — no GPU detected. "
                "Use a Whisper engine for CPU transcription.",
                model_name=self.model_id,
            )
        try:
            from transformers import (
                AutoProcessor,
                Qwen2AudioForConditionalGeneration,
            )
        except ImportError as exc:
            raise ModelLoadError(
                "transformers >= 4.45 with Qwen2-Audio support is required.",
                model_name=self.model_id,
            ) from exc

        cache_dir = str(get_model_cache_dir("qwen2-audio"))
        log.info("Loading %s on CUDA…", _QWEN_REPO)
        try:
            self._processor = AutoProcessor.from_pretrained(
                _QWEN_REPO, cache_dir=cache_dir,
            )
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                _QWEN_REPO,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map=self._device,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Qwen2-Audio: {exc}",
                model_name=self.model_id,
            ) from exc
        log.info("Qwen2-Audio ready.")

    def transcribe(
        self,
        audio_path: pathlib.Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        self.load()
        try:
            import librosa
        except ImportError as exc:
            raise ModelLoadError(
                "librosa is required.", model_name=self.model_id,
            ) from exc

        try:
            audio, _sr = librosa.load(str(audio_path), sr=16_000, mono=True)
        except Exception as exc:
            raise PipelineExecutionError(
                f"Failed to load audio {audio_path.name}: {exc}",
                pipeline_name="transcribe",
            ) from exc

        user_prompt = prompt or _PROMPT
        if language:
            user_prompt += f" The lyrics are in {language}."
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        inputs = self._processor(
            text=text, audios=[audio], sampling_rate=16_000,
            return_tensors="pt", padding=True,
        ).to(self._model.device)

        try:
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False,
                )
            generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
            output = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True,
            )[0].strip()
        except Exception as exc:
            raise PipelineExecutionError(
                f"Qwen2-Audio generation failed: {exc}",
                pipeline_name="transcribe",
            ) from exc

        duration = float(len(audio)) / 16_000.0
        segment = TranscriptionSegment(
            start=0.0, end=duration, text=output, words=[],
        )
        return TranscriptionResult(
            text=output,
            language=language,
            segments=[segment],
            has_word_timestamps=False,
            engine_id=self.engine_id,
            model_id=self.model_id,
        )

    def clear(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Qwen engine cleared.")
