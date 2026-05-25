"""Qwen2-Audio engine for high-fidelity sung-lyrics transcription.

LICENSE NOTE: Apache 2.0.  See licenses/LICENSE-Qwen2-Audio and the
ACKNOWLEDGMENTS.md entry.  Re-verify the license before any future
upgrade — Qwen variants are not uniformly Apache 2.0 across sizes.
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

_QWEN_REPO = "Qwen/Qwen2-Audio-7B-Instruct"
_PROMPT = (
    "Transcribe the lyrics of this audio. Output only the lyrics text, "
    "preserving line breaks where the singer pauses. Do not add commentary, "
    "explanations, or section labels."
)

Quantization = Literal["none", "nf4"]

# Variant metadata for the /api/transcribe/engines endpoint and the
# QwenEngine constructor.  Both variants share weights + chat template;
# they differ only in the from_pretrained quantization kwargs.
QWEN_VARIANTS: dict[str, dict] = {
    "qwen2-audio-7b-instruct": {
        "display_name": "Qwen2-Audio 7B",
        "quantization": "none",
        "approx_vram_gb": 16,
    },
    "qwen2-audio-7b-instruct-nf4": {
        "display_name": "Qwen2-Audio 7B (4-bit)",
        "quantization": "nf4",
        "approx_vram_gb": 9,
    },
}


class QwenEngine:
    engine_id = "qwen"
    supports_word_timestamps = False
    requires_gpu = True

    def __init__(
        self,
        model_id: str = "qwen2-audio-7b-instruct",
        *,
        device: str = "cuda",
    ) -> None:
        if model_id not in QWEN_VARIANTS:
            raise ValueError(
                f"Unknown qwen model_id {model_id!r}. "
                f"Available: {sorted(QWEN_VARIANTS)}"
            )
        self.model_id = model_id
        self._variant = QWEN_VARIANTS[model_id]
        self._quantization: Quantization = self._variant["quantization"]
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
        log.info(
            "Loading %s on CUDA (quantization=%s)…",
            _QWEN_REPO, self._quantization,
        )
        try:
            self._processor = AutoProcessor.from_pretrained(
                _QWEN_REPO, cache_dir=cache_dir,
            )
            from_pretrained_kwargs: dict = {
                "cache_dir": cache_dir,
                "device_map": self._device,
            }
            if self._quantization == "nf4":
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise ModelLoadError(
                        "BitsAndBytesConfig unavailable — install bitsandbytes.",
                        model_name=self.model_id,
                    ) from exc
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                from_pretrained_kwargs["quantization_config"] = bnb_config
                # device_map handled by bitsandbytes; do not pass torch_dtype.
            else:
                from_pretrained_kwargs["torch_dtype"] = torch.float16

            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                _QWEN_REPO, **from_pretrained_kwargs,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Qwen2-Audio ({self._quantization}): {exc}",
                model_name=self.model_id,
            ) from exc
        log.info("Qwen2-Audio ready (model_id=%s).", self.model_id)

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
        # NOTE: Qwen2AudioProcessor.__call__ takes `audio=` (singular).  The
        # old `audios=` kwarg (still mentioned in some HF docstrings) is
        # silently dropped — the processor then returns only input_ids and
        # the model responds with "no audio provided".  Sampling rate must
        # match proc.feature_extractor.sampling_rate (16 kHz for Qwen2-Audio).
        inputs = self._processor(
            text=text, audio=audio, sampling_rate=16_000,
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
