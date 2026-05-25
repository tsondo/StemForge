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

# Base task description used in turn 1. Stable across hint/no-hint cases.
_TURN1_BASE = (
    "I'll send you an audio clip in a moment. Your job is to transcribe "
    "the sung lyrics verbatim. Do not translate, summarize, or comment — "
    "output only the lyrics text, preserving line breaks where the singer "
    "pauses."
)

# Hint guidance, appended to turn 1 only when a hint is present.
# The final sentence is critical: it tells the model not to emit hint text
# unless it actually appears in the audio. Without this, the model treats
# the hint as content it must produce in every chunk (Phase 1 failure mode).
_TURN1_HINT = (
    "\n\nThe lyrics may contain these specific names and spellings: {hint}. "
    "When you hear those words sung, use those exact spellings. Otherwise "
    "ignore them — do not emit these names unless you actually hear them "
    "in the audio."
)

# Fabricated assistant turn. Same for both hint and no-hint cases — the
# constraint about only emitting heard text applies regardless.
_TURN2_ACKNOWLEDGMENT = (
    "Understood. I will transcribe the lyrics of the audio verbatim, "
    "preserving line breaks. I will only emit text I actually hear sung."
)

# Turn 3 base instruction. Minimal by design — the audio attached to this
# turn is the primary signal; the text is only here so the chat template
# is well-formed.
_TURN3_BASE = "Transcribe the lyrics."


def _sanitize_hint(hint: str | None) -> str:
    """Normalize a user-supplied hint to a single line with no double-quotes.

    Newlines and stray quotes inside the hint could disrupt the chat
    template's parse of the surrounding instruction text.
    """
    if not hint:
        return ""
    cleaned = " ".join(hint.split()).replace('"', "'")
    return cleaned


def _build_qwen_conversation(
    hint: str | None,
    language: str | None,
) -> list[dict]:
    """Construct Qwen's chat-template conversation for one chunk.

    Returns a three-turn structure that isolates hint context (turn 1) and
    fabricated assistant acknowledgment (turn 2) from the audio-bearing
    transcription request (turn 3). The audio placeholder is included in
    turn 3 only — the actual audio array is supplied separately to the
    processor at encoding time.

    The structural separation is the load-bearing design choice. Inline
    concatenation of hint into the same turn as the transcription request
    caused the model to treat hint text as content to emit (see Phase 1
    diagnostic for evidence). The fake-assistant-turn pattern is documented
    in Addendum 6 §5 as the escalation path for that failure mode and is
    implemented here.
    """
    hint_clean = _sanitize_hint(hint)

    # Turn 1: setup, optionally with hint guidance.
    turn1_text = _TURN1_BASE
    if hint_clean:
        turn1_text += _TURN1_HINT.format(hint=hint_clean)

    # Turn 3: minimal transcription request, optionally with language hint.
    turn3_text = _TURN3_BASE
    if language:
        turn3_text += f" The lyrics are in {language}."

    return [
        {"role": "user", "content": [
            {"type": "text", "text": turn1_text},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": _TURN2_ACKNOWLEDGMENT},
        ]},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "in_memory"},
            {"type": "text", "text": turn3_text},
        ]},
    ]


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

        from ._qwen_chunker import SAMPLE_RATE, slice_audio, stitch_chunks

        try:
            audio, _sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise PipelineExecutionError(
                f"Failed to load audio '{audio_path.name}': {exc}",
                pipeline_name="transcribe",
            ) from exc

        duration_s = float(len(audio)) / SAMPLE_RATE
        chunks = slice_audio(audio, sample_rate=SAMPLE_RATE)
        log.info(
            "Qwen transcribing %.1fs of audio in %d chunk(s).",
            duration_s, len(chunks),
        )

        chunk_texts: list[str] = []
        for chunk in chunks:
            try:
                text = self._transcribe_chunk(
                    chunk.samples,
                    language=language,
                    prompt=prompt,
                )
            except Exception as exc:
                raise PipelineExecutionError(
                    f"Qwen2-Audio generation failed on chunk {chunk.index} "
                    f"({chunk.start_s:.1f}-{chunk.end_s:.1f}s): {exc}",
                    pipeline_name="transcribe",
                ) from exc
            chunk_texts.append(text)
            preview = text.replace("\n", " ⏎ ").strip()
            if len(preview) > 80:
                preview = preview[:80] + "…"
            log.info(
                "Qwen chunk %d/%d (%.1f-%.1fs): %d chars | %s",
                chunk.index + 1, len(chunks),
                chunk.start_s, chunk.end_s, len(text), preview,
            )

        stitched = stitch_chunks(chunk_texts)

        # The pipeline expects segments. For Qwen we still emit a single
        # segment covering the whole duration — we don't have per-chunk
        # timestamps that align with the stitched text.
        segment = TranscriptionSegment(
            start=0.0, end=duration_s, text=stitched, words=[],
        )
        return TranscriptionResult(
            text=stitched,
            language=language,
            segments=[segment],
            has_word_timestamps=False,
            engine_id=self.engine_id,
            model_id=self.model_id,
        )

    def _transcribe_chunk(
        self,
        audio: "np.ndarray",
        *,
        language: str | None,
        prompt: str | None,
    ) -> str:
        """Run a single Qwen forward pass on one audio chunk."""
        conversation = _build_qwen_conversation(hint=prompt, language=language)
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

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )
        generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
        output = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0].strip()
        return output

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
