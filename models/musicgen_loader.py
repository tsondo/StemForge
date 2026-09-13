"""Stable Audio Open model loader for StemForge.

Wraps ``diffusers.StableAudioPipeline.from_pretrained`` to provide the same
load/evict interface as the other StemForge model loaders.

Returns a ``(pipeline, model_config)`` tuple where *model_config* is a plain
dict with ``"sample_rate"`` extracted from the pipeline's VAE
(always 44 100 Hz for ``stabilityai/stable-audio-open-1.0``).
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch

from utils.cache import get_model_cache_dir
from utils.device import get_device
from utils.errors import ModelLoadError


log = logging.getLogger("stemforge.models.musicgen_loader")


class MusicGenModelLoader:
    """Load and cache the Stable Audio Open generation pipeline.

    Parameters
    ----------
    cache_dir:
        Directory used to store downloaded model checkpoints.
        Defaults to ``~/.cache/stemforge/musicgen``.
    """

    def __init__(self, cache_dir: pathlib.Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = get_model_cache_dir("musicgen")
        self._cache_dir = pathlib.Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline: Any = None
        self._model_config: dict | None = None
        self._loaded_model_name: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, model_name: str, device: "torch.device | None" = None) -> tuple[Any, dict]:
        """Return ``(pipeline, model_config)`` for *model_name*, loading if needed.

        On first call downloads weights from HuggingFace Hub (~2 GB) and
        caches them locally.  Subsequent calls return the cached instance.
        The pipeline is placed on CUDA if available, otherwise CPU.
        fp16 is used on CUDA; fp32 on CPU.

        Parameters
        ----------
        model_name:
            HuggingFace repo ID, e.g. ``"stabilityai/stable-audio-open-1.0"``.

        Returns
        -------
        tuple[StableAudioPipeline, dict]
            Loaded pipeline and ``{"sample_rate": int}`` config dict.

        Raises
        ------
        ModelLoadError
            If ``diffusers`` is not installed, or if the model cannot be
            downloaded or loaded.
        """
        if self._loaded_model_name == model_name and self._pipeline is not None:
            return self._pipeline, self._model_config  # type: ignore[return-value]

        try:
            from diffusers import StableAudioPipeline  # type: ignore[import]
        except ImportError as exc:
            raise ModelLoadError(
                "diffusers is not installed.\n"
                "Install with: uv pip install 'diffusers>=0.30.0'",
                model_name=model_name,
            ) from exc

        device = device if device is not None else get_device()
        # MPS does not support float16 reliably; use float32 on MPS and CPU.
        dtype  = torch.float16 if device.type == "cuda" else torch.float32

        # Resolve HuggingFace auth token (needed for gated models).
        from utils.hf_hub import get_hf_token

        hf_token: str | None = get_hf_token()

        try:
            log.info("Loading %s (dtype=%s) …", model_name, dtype)
            pipeline = StableAudioPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                cache_dir=str(self._cache_dir),
                token=hf_token,
            )
            pipeline = pipeline.to(device)
        except Exception as exc:
            msg = str(exc)
            if "403" in msg or "gated" in msg.lower() or "authorized" in msg.lower():
                raise ModelLoadError(
                    f"{model_name} is a gated model — two steps required:\n"
                    "  1. Visit https://huggingface.co/stabilityai/stable-audio-open-1.0\n"
                    "     and accept the license (requires a free HuggingFace account).\n"
                    "  2. Authenticate locally:\n"
                    "       huggingface-cli login\n"
                    "     or set the HF_TOKEN environment variable.",
                    model_name=model_name,
                ) from exc
            raise ModelLoadError(
                f"Failed to download / load {model_name}: {exc}",
                model_name=model_name,
            ) from exc

        sample_rate: int = pipeline.vae.sampling_rate
        model_config = {"sample_rate": sample_rate}

        self._pipeline = pipeline
        self._model_config = model_config
        self._loaded_model_name = model_name
        log.info("Loaded %s on %s (sr=%d)", model_name, device, sample_rate)
        return pipeline, model_config

    def is_cached(self, model_name: str) -> bool:
        """Return True if *model_name* is already loaded in memory."""
        return self._loaded_model_name == model_name and self._pipeline is not None

    def evict(self, model_name: str | None = None) -> None:
        """Release the in-memory pipeline to free GPU/CPU memory.

        Parameters
        ----------
        model_name:
            If given, only evict if this name matches the currently loaded
            model.  Pass ``None`` to evict unconditionally.
        """
        if model_name is not None and self._loaded_model_name != model_name:
            return
        if self._pipeline is not None:
            try:
                self._pipeline.to("cpu")
            except Exception:
                pass
            del self._pipeline
        self._pipeline = None
        self._model_config = None
        self._loaded_model_name = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Evicted generation pipeline from memory")
