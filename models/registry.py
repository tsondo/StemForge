"""
StemForge Model Registry
========================

Single source of truth for every model's identity, audio/MIDI parameters,
device rules, capabilities, GUI metadata, and known quirks.

Usage
-----
::

    from models.registry import get_spec, list_specs, DemucsSpec
    spec = get_spec("htdemucs")
    demucs_ids = [s.model_id for s in list_specs(DemucsSpec)]

Registry vs. pipeline Config
-----------------------------
* :class:`ModelSpec` (and subclasses) — describe the *model itself*: its
  native sample rate, hardware constraints, capability tags, etc.  These
  are **frozen** and never change at runtime.
* Per-run Config classes (``DemucsConfig``, ``BasicPitchConfig``, etc.) —
  describe a specific *job*.  They remain unchanged by this module.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

from utils.cache import get_model_cache_dir


# ---------------------------------------------------------------------------
# Base spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Immutable descriptor for a single model variant.

    Parameters
    ----------
    model_id:
        Canonical registry key, e.g. ``"htdemucs"`` or ``"whisper-base"``.
    display_name:
        Human-readable label shown in GUI combos and tooltips.
    version:
        Upstream version string, e.g. ``"4.0.1"``.
    source:
        Where the weights come from: a Python module path, a Hugging Face
        repo ID, or the string ``"bundled"`` if shipped inside the package.
    device:
        Requested compute device: ``"cpu"``, ``"cuda"``, or ``"auto"``.
    gpu_capable:
        Whether the model can run on a GPU at all.
    device_fallback:
        Device to try if *device* is unavailable or raises a runtime error.
    device_quirks:
        Free-text description of known device-specific issues.
    sample_rate:
        Native audio sample rate in Hz (e.g. 44100, 22050, 16000).
    hop_size:
        Frame hop in samples; 0 means not applicable.
    chunk_size:
        Inference chunk size in samples; 0 means full-file inference.
    max_duration_seconds:
        Hard upper limit on input/output duration; 0.0 means unlimited.
    default_bpm:
        Default tempo used for MIDI output; 0.0 means not applicable.
    default_key:
        Default musical key string; empty string means not applicable.
    default_time_signature:
        Default time signature string, e.g. ``"4/4"``; empty = N/A.
    quantize_grid:
        Default quantisation grid: ``"sixteenth"``, ``"eighth"``, or
        ``"none"``.
    default_min_note_ms:
        Default minimum note length in milliseconds; 0.0 = N/A.
    capabilities:
        Frozen set of string tags describing what this model can do.
        Defined tags: ``"separate"``, ``"transcribe"``, ``"generate"``,
        ``"text_conditioning"``, ``"stem_input"``, ``"melody_conditioning"``,
        ``"batch_inference"``, ``"gpu_acceleration"``, ``"word_timestamps"``,
        ``"pitch_estimation"``.
    cache_subdir:
        Sub-directory name under ``~/.cache/stemforge/`` for model weights.
    description:
        One-line summary suitable for a GUI tooltip.
    preprocessing:
        Human-readable description of input preparation steps.
    postprocessing:
        Human-readable description of post-inference steps.
    """

    # Identity
    model_id: str
    display_name: str
    version: str
    source: str

    # Device
    device: str
    gpu_capable: bool
    device_fallback: str
    device_quirks: str

    # Audio
    sample_rate: int
    hop_size: int
    chunk_size: int
    max_duration_seconds: float

    # MIDI (0 / empty = N/A)
    default_bpm: float
    default_key: str
    default_time_signature: str
    quantize_grid: str
    default_min_note_ms: float

    # Capabilities
    capabilities: frozenset[str]

    # Cache
    cache_subdir: str

    # Description / quirks
    description: str
    preprocessing: str
    postprocessing: str

    # License (kw_only so subclass non-default fields are valid)
    license_warning: str = field(default="", kw_only=True)

    @property
    def cache_dir(self) -> pathlib.Path:
        """Absolute path to this model's weight cache directory."""
        return get_model_cache_dir(self.cache_subdir)


# ---------------------------------------------------------------------------
# Subclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DemucsSpec(ModelSpec):
    """Descriptor for a Demucs source-separation model variant.

    Additional fields
    -----------------
    available_stems:
        Tuple of stem names this model outputs, in Demucs source order.
    """

    available_stems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BasicPitchSpec(ModelSpec):
    """Descriptor for the BasicPitch audio-to-MIDI transcription model.

    Additional fields
    -----------------
    preferred_format:
        Preferred weight serialisation format: ``"savedmodel"`` or ``"onnx"``.
    onset_range:
        ``(min, max)`` slider range for the onset threshold parameter.
    frame_range:
        ``(min, max)`` slider range for the frame threshold parameter.
    min_note_range:
        ``(min_ms, max_ms)`` slider range for the minimum note length.
    default_onset:
        Default onset confidence threshold value.
    default_frame:
        Default frame confidence threshold value.
    """

    preferred_format: str
    onset_range: tuple[float, float]
    frame_range: tuple[float, float]
    min_note_range: tuple[float, float]
    default_onset: float
    default_frame: float


@dataclass(frozen=True, slots=True)
class WhisperSpec(ModelSpec):
    """Descriptor for a faster-whisper vocal transcription model.

    Additional fields
    -----------------
    model_size:
        Whisper model size string: ``"tiny"``, ``"base"``, ``"small"``,
        or ``"medium"``.
    compute_type:
        CTranslate2 compute type, e.g. ``"int8"`` or ``"float16"``.
    default_language:
        ISO-639-1 language hint, or ``None`` for auto-detection.
    word_timestamps:
        Whether word-level timestamps are requested from the transcription.
    vad_filter:
        Whether the Silero VAD pre-filter is applied before transcription.
    """

    model_size: str
    compute_type: str
    default_language: str | None
    word_timestamps: bool
    vad_filter: bool


@dataclass(frozen=True, slots=True)
class StableAudioSpec(ModelSpec):
    """Descriptor for the Stable Audio Open generation model.

    Additional fields
    -----------------
    hf_repo_id:
        Hugging Face Hub repository ID to load the model from.
    default_steps:
        Default number of diffusion sampling steps.
    default_cfg_scale:
        Default classifier-free guidance scale.
    conditioning_keys:
        Tuple of conditioning input names required by this model.
    """

    hf_repo_id: str
    default_steps: int
    default_cfg_scale: float
    conditioning_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoformerSpec(ModelSpec):
    """Descriptor for a BS-Roformer / MelBand-Roformer separation model.

    Additional fields
    -----------------
    architecture:
        Model class to instantiate: ``"bs_roformer"`` or
        ``"mel_band_roformer"``.
    checkpoint_url:
        Direct download URL for the ``.ckpt`` weight file.
    config_url:
        Direct download URL for the accompanying ``.yaml`` config file.
    target_instrument:
        The instrument this model is trained to isolate, e.g. ``"vocals"``.
    other_fix:
        When ``True``, the ``"other"`` stem is computed as
        ``mix - predicted`` rather than predicted directly.
    available_stems:
        Stems produced by the pipeline, e.g. ``("vocals", "other")``.
    default_chunk_size:
        Default inference chunk size in samples (352800 ≈ 8 s at 44.1 kHz).
    default_num_overlap:
        Default number of overlap divisions between consecutive chunks.
    """

    architecture: str
    checkpoint_url: str
    config_url: str
    target_instrument: str | None
    other_fix: bool
    available_stems: tuple[str, ...]
    default_chunk_size: int
    default_num_overlap: int


# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> ModelSpec:
    """Add *spec* to the registry and return it (for use as a module constant)."""
    if spec.model_id in _REGISTRY:
        raise ValueError(
            f"Duplicate model_id in registry: {spec.model_id!r}"
        )
    _REGISTRY[spec.model_id] = spec
    return spec


# ---------------------------------------------------------------------------
# Demucs variants
# ---------------------------------------------------------------------------

_DEMUCS_CAPS = frozenset({
    "separate",
    "stem_input",
    "gpu_acceleration",
    "batch_inference",
})
_DEMUCS_STEMS = ("vocals", "drums", "bass", "other")

DEMUCS_HTDEMUCS = _register(DemucsSpec(
    model_id="htdemucs",
    display_name="htdemucs (Hybrid Transformer)",
    version="4.0.1",
    source="demucs.pretrained",
    device="auto",
    gpu_capable=True,
    device_fallback="cpu",
    device_quirks="",
    sample_rate=44_100,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_DEMUCS_CAPS,
    cache_subdir="demucs",
    description="Best overall quality — good for most music.",
    preprocessing="Stereo 44.1 kHz; split into 10 s chunks for inference.",
    postprocessing="Overlap-add reconstruction; write per-stem WAV files.",
    available_stems=_DEMUCS_STEMS,
))

DEMUCS_HTDEMUCS_FT = _register(DemucsSpec(
    model_id="htdemucs_ft",
    display_name="htdemucs_ft (Fine-tuned)",
    version="4.0.1",
    source="demucs.pretrained",
    device="auto",
    gpu_capable=True,
    device_fallback="cpu",
    device_quirks="",
    sample_rate=44_100,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_DEMUCS_CAPS,
    cache_subdir="demucs",
    description="Fine-tuned variant — sharper on pop and rock.",
    preprocessing="Stereo 44.1 kHz; split into 10 s chunks for inference.",
    postprocessing="Overlap-add reconstruction; write per-stem WAV files.",
    available_stems=_DEMUCS_STEMS,
))

_MDX_QUIRKS = (
    "CUBLAS_STATUS_INVALID_VALUE on CUDA due to non-contiguous complex strides "
    "inside STFT layers; auto CPU fallback applied in _run_inference."
)

DEMUCS_MDX_EXTRA = _register(DemucsSpec(
    model_id="mdx_extra",
    display_name="mdx_extra (MDX architecture)",
    version="4.0.1",
    source="demucs.pretrained",
    device="auto",
    gpu_capable=True,
    device_fallback="cpu",
    device_quirks=_MDX_QUIRKS,
    sample_rate=44_100,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_DEMUCS_CAPS,
    cache_subdir="demucs",
    description="MDX architecture — excellent vocal isolation.",
    preprocessing="Stereo 44.1 kHz; .contiguous() applied; num_workers=0.",
    postprocessing="Overlap-add reconstruction; write per-stem WAV files.",
    available_stems=_DEMUCS_STEMS,
))

DEMUCS_MDX_EXTRA_Q = _register(DemucsSpec(
    model_id="mdx_extra_q",
    display_name="mdx_extra_q (MDX quality mode)",
    version="4.0.1",
    source="demucs.pretrained",
    device="auto",
    gpu_capable=True,
    device_fallback="cpu",
    device_quirks=_MDX_QUIRKS,
    sample_rate=44_100,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_DEMUCS_CAPS,
    cache_subdir="demucs",
    description="MDX quality mode — cleanest results, slowest.",
    preprocessing="Stereo 44.1 kHz; .contiguous() applied; num_workers=0.",
    postprocessing="Overlap-add reconstruction; write per-stem WAV files.",
    available_stems=_DEMUCS_STEMS,
))

# ---------------------------------------------------------------------------
# BasicPitch
# ---------------------------------------------------------------------------

BASICPITCH = _register(BasicPitchSpec(
    model_id="basicpitch",
    display_name="BasicPitch (ICASSP 2022)",
    version="0.4.0",
    source="bundled",
    device="cpu",
    gpu_capable=False,
    device_fallback="cpu",
    device_quirks=(
        "No CUDA kernels for CC 12.x; CUDA_VISIBLE_DEVICES=-1 forced "
        "before TF import."
    ),
    sample_rate=22_050,
    hop_size=256,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="sixteenth",
    default_min_note_ms=58.0,
    capabilities=frozenset({"transcribe"}),
    cache_subdir="basicpitch",
    description="Polyphonic audio-to-MIDI transcription (CPU-only).",
    preprocessing="Mono downmix; resample to 22 050 Hz internally.",
    postprocessing="Key snap; minimum-duration clip.",
    preferred_format="savedmodel",
    onset_range=(0.0, 1.0),
    frame_range=(0.0, 1.0),
    min_note_range=(20.0, 500.0),
    default_onset=0.5,
    default_frame=0.3,
))

# ---------------------------------------------------------------------------
# faster-whisper variants
# ---------------------------------------------------------------------------

_WHISPER_CAPS = frozenset({
    "transcribe",
    "word_timestamps",
    "pitch_estimation",
})

WHISPER_TINY = _register(WhisperSpec(
    model_id="whisper-tiny",
    display_name="Whisper tiny",
    version="1.1.0",
    source="openai/whisper-tiny",
    device="cpu",
    gpu_capable=False,
    device_fallback="cpu",
    device_quirks="",
    sample_rate=16_000,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_WHISPER_CAPS,
    cache_subdir="whisper",
    description="Smallest Whisper variant — fastest, least accurate.",
    preprocessing="Mono 16 kHz; VAD pre-filter.",
    postprocessing="Word-level timestamps; PYIN pitch estimation per word.",
    model_size="tiny",
    compute_type="int8",
    default_language=None,
    word_timestamps=True,
    vad_filter=True,
))

WHISPER_BASE = _register(WhisperSpec(
    model_id="whisper-base",
    display_name="Whisper base",
    version="1.1.0",
    source="openai/whisper-base",
    device="cpu",
    gpu_capable=False,
    device_fallback="cpu",
    device_quirks="",
    sample_rate=16_000,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_WHISPER_CAPS,
    cache_subdir="whisper",
    description="Default vocal transcription — good speed/accuracy balance.",
    preprocessing="Mono 16 kHz; VAD pre-filter.",
    postprocessing="Word-level timestamps; PYIN pitch estimation per word.",
    model_size="base",
    compute_type="int8",
    default_language=None,
    word_timestamps=True,
    vad_filter=True,
))

WHISPER_SMALL = _register(WhisperSpec(
    model_id="whisper-small",
    display_name="Whisper small",
    version="1.1.0",
    source="openai/whisper-small",
    device="cpu",
    gpu_capable=False,
    device_fallback="cpu",
    device_quirks="",
    sample_rate=16_000,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_WHISPER_CAPS,
    cache_subdir="whisper",
    description="Whisper small — better accuracy, slower.",
    preprocessing="Mono 16 kHz; VAD pre-filter.",
    postprocessing="Word-level timestamps; PYIN pitch estimation per word.",
    model_size="small",
    compute_type="int8",
    default_language=None,
    word_timestamps=True,
    vad_filter=True,
))

WHISPER_MEDIUM = _register(WhisperSpec(
    model_id="whisper-medium",
    display_name="Whisper medium",
    version="1.1.0",
    source="openai/whisper-medium",
    device="cpu",
    gpu_capable=False,
    device_fallback="cpu",
    device_quirks="",
    sample_rate=16_000,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_WHISPER_CAPS,
    cache_subdir="whisper",
    description="Whisper medium — highest accuracy, slowest.",
    preprocessing="Mono 16 kHz; VAD pre-filter.",
    postprocessing="Word-level timestamps; PYIN pitch estimation per word.",
    model_size="medium",
    compute_type="int8",
    default_language=None,
    word_timestamps=True,
    vad_filter=True,
))

WHISPER_LARGE_V3 = _register(WhisperSpec(
    model_id="whisper-large-v3",
    display_name="Whisper large-v3",
    version="1.1.0",
    source="openai/whisper-large-v3",
    device="cpu",
    gpu_capable=True,
    device_fallback="cpu",
    device_quirks="",
    sample_rate=16_000,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_WHISPER_CAPS,
    cache_subdir="whisper",
    description="Whisper large-v3 — best accuracy, GPU recommended.",
    preprocessing="Mono 16 kHz; VAD pre-filter.",
    postprocessing="Word-level timestamps; PYIN pitch estimation per word.",
    model_size="large-v3",
    compute_type="int8",
    default_language=None,
    word_timestamps=True,
    vad_filter=True,
))

# ---------------------------------------------------------------------------
# Stable Audio Open
# ---------------------------------------------------------------------------

STABLE_AUDIO_OPEN = _register(StableAudioSpec(
    model_id="stable-audio-open-1.0",
    display_name="Stable Audio Open 1.0",
    version="1.0",
    source="stabilityai/stable-audio-open-1.0",
    device="auto",
    gpu_capable=True,
    device_fallback="cpu",
    device_quirks="",
    sample_rate=44_100,
    hop_size=0,
    chunk_size=0,
    max_duration_seconds=47.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=frozenset({
        "generate",
        "text_conditioning",
        "gpu_acceleration",
    }),
    cache_subdir="stable_audio",
    description="Text-conditioned stereo audio generation up to 47 s.",
    preprocessing="Text prompt + seconds_start / seconds_total conditioning.",
    postprocessing="Write stereo WAV at 44.1 kHz.",
    hf_repo_id="stabilityai/stable-audio-open-1.0",
    default_steps=100,
    default_cfg_scale=7.0,
    conditioning_keys=("seconds_start", "seconds_total", "prompt"),
))

# ---------------------------------------------------------------------------
# BS-Roformer / MelBand-Roformer variants
# ---------------------------------------------------------------------------

_ROFORMER_CAPS = frozenset({
    "separate",
    "stem_input",
    "gpu_acceleration",
})

ROFORMER_VIPERX = _register(RoformerSpec(
    model_id="roformer-viperx-vocals",
    display_name="BS-Roformer (ViperX, SDR 12.97)",
    version="1.0",
    source="TRvlvr/model_repo",
    device="auto",
    gpu_capable=True,
    device_fallback="cpu",
    device_quirks="CUBLAS errors fall back to CPU per-chunk automatically.",
    sample_rate=44_100,
    hop_size=0,
    chunk_size=352_800,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_ROFORMER_CAPS,
    cache_subdir="roformer",
    description="cleaner, smoother vocals with fewer artifacts, but sometimes leaves more bleed from drums or guitars",
    preprocessing="Stereo 44.1 kHz; chunked overlap-add with linear fade.",
    postprocessing="other = mix − vocals; write per-stem WAV files.",
    architecture="bs_roformer",
    checkpoint_url=(
        "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    ),
    config_url=(
        "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/"
        "mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml"
    ),
    target_instrument="vocals",
    other_fix=True,
    available_stems=("vocals", "other"),
    default_chunk_size=352_800,
    default_num_overlap=4,
))

ROFORMER_ZFTURBO_4STEM = _register(RoformerSpec(
    model_id="roformer-zfturbo-4stem",
    display_name="BS-Roformer 4-Stem (ZFTurbo, SDR 9.66)",
    version="1.0",
    source="ZFTurbo/Music-Source-Separation-Training",
    device="auto",
    gpu_capable=True,
    device_fallback="cpu",
    device_quirks="CUBLAS errors fall back to CPU per-chunk automatically.",
    sample_rate=44_100,
    hop_size=0,
    chunk_size=485_100,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_ROFORMER_CAPS,
    cache_subdir="roformer",
    description="separates all four standard stems at once — best option when you need drums, bass, or other alongside vocals",
    preprocessing="Stereo 44.1 kHz; chunked overlap-add with linear fade.",
    postprocessing="Write per-stem WAV files (drums, bass, other, vocals).",
    architecture="bs_roformer",
    checkpoint_url=(
        "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
        "v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt"
    ),
    config_url=(
        "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/"
        "v1.0.12/config_bs_roformer_384_8_2_485100.yaml"
    ),
    target_instrument=None,
    other_fix=False,
    available_stems=("drums", "bass", "other", "vocals"),
    default_chunk_size=485_100,
    default_num_overlap=2,
))

ROFORMER_JARREDOU_6STEM = _register(RoformerSpec(
    model_id="roformer-jarredou-6stem",
    display_name="BS-Roformer 6-Stem (jarredou — bass/drums/other/vocals/guitar/piano)",
    version="1.0",
    source="jarredou/BS-ROFO-SW-Fixed",
    device="auto",
    gpu_capable=True,
    device_fallback="cpu",
    device_quirks="CUBLAS errors fall back to CPU per-chunk automatically.",
    sample_rate=44_100,
    hop_size=0,
    chunk_size=588_800,
    max_duration_seconds=0.0,
    default_bpm=0.0,
    default_key="",
    default_time_signature="",
    quantize_grid="none",
    default_min_note_ms=0.0,
    capabilities=_ROFORMER_CAPS,
    cache_subdir="roformer",
    description="separates six stems including guitar and piano — useful for mixed-instrument tracks, though guitar/piano quality varies by material",
    license_warning=(
        "These model weights have no license specified by the author. "
        "Under copyright law, absence of a license means all rights are reserved "
        "and no rights are granted to use, modify, or distribute the work. "
        "You use these weights at your own legal risk. "
        "MIT-licensed alternatives exist (Demucs, ViperX, ZFTurbo)."
    ),
    preprocessing="Stereo 44.1 kHz; chunked overlap-add with linear fade.",
    postprocessing="Write per-stem WAV files (bass, drums, other, vocals, guitar, piano).",
    architecture="bs_roformer",
    checkpoint_url=(
        "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt"
    ),
    config_url=(
        "https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml"
    ),
    target_instrument=None,
    other_fix=False,
    available_stems=("bass", "drums", "other", "vocals", "guitar", "piano"),
    default_chunk_size=588_800,
    default_num_overlap=2,
))

# ---------------------------------------------------------------------------
# Convenience constants
# ---------------------------------------------------------------------------

#: Default Whisper spec used by MidiModelLoader for vocal transcription.
DEFAULT_WHISPER_SPEC: WhisperSpec = WHISPER_BASE

#: Default Demucs spec used as the application default model.
DEFAULT_DEMUCS_SPEC: DemucsSpec = DEMUCS_HTDEMUCS

#: Default Roformer spec (ViperX BSRoformer).
DEFAULT_ROFORMER_SPEC: RoformerSpec = ROFORMER_VIPERX

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_spec(model_id: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for *model_id*.

    Raises
    ------
    KeyError
        With a helpful message listing all registered IDs if *model_id* is
        not found.
    """
    try:
        return _REGISTRY[model_id]
    except KeyError:
        registered = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Unknown model_id {model_id!r}.  "
            f"Registered IDs: {registered}"
        ) from None


def list_specs(kind: type[ModelSpec] | None = None) -> list[ModelSpec]:
    """Return all registered specs, optionally filtered by subclass type.

    Parameters
    ----------
    kind:
        If provided, only specs that are instances of *kind* are returned.
        Example: ``list_specs(DemucsSpec)`` returns all four Demucs variants.

    Returns
    -------
    list[ModelSpec]
        Ordered by registration order (insertion order of the module-level
        ``_register()`` calls).
    """
    specs = list(_REGISTRY.values())
    if kind is not None:
        specs = [s for s in specs if isinstance(s, kind)]
    return specs


def get_loader_kwargs(model_id: str) -> dict[str, Any]:
    """Return keyword arguments for constructing the appropriate model loader.

    The returned dict is suitable for unpacking into the loader constructor
    (e.g. ``DemucsModelLoader(**get_loader_kwargs("htdemucs"))``).

    Parameters
    ----------
    model_id:
        Registered model ID.

    Returns
    -------
    dict[str, Any]
        Loader constructor kwargs.

    Raises
    ------
    KeyError
        If *model_id* is not registered.
    NotImplementedError
        If no loader kwargs mapping is defined for this spec type.
    """
    spec = get_spec(model_id)

    if isinstance(spec, DemucsSpec):
        return {"cache_dir": spec.cache_dir}

    if isinstance(spec, BasicPitchSpec):
        return {
            "cache_dir": spec.cache_dir,
            "preferred_format": spec.preferred_format,
        }

    if isinstance(spec, WhisperSpec):
        return {
            "model_size_or_path": spec.model_size,
            "device": spec.device,
            "compute_type": spec.compute_type,
        }

    if isinstance(spec, StableAudioSpec):
        return {
            "cache_dir": spec.cache_dir,
            "hf_repo_id": spec.hf_repo_id,
        }

    if isinstance(spec, RoformerSpec):
        return {
            "cache_dir": spec.cache_dir,
            "architecture": spec.architecture,
            "checkpoint_url": spec.checkpoint_url,
            "config_url": spec.config_url,
        }

    raise NotImplementedError(
        f"No loader kwargs defined for spec type {type(spec).__name__!r}."
    )


def get_pipeline_defaults(model_id: str) -> dict[str, Any]:
    """Return default inference parameters for the model's pipeline Config.

    The returned dict is intended to seed pipeline Config constructors.
    Not all keys will apply to every Config class — callers should use
    ``**{k: v for k, v in defaults.items() if hasattr(config_cls, k)}``.

    Parameters
    ----------
    model_id:
        Registered model ID.

    Returns
    -------
    dict[str, Any]
        Default parameter values.

    Raises
    ------
    KeyError
        If *model_id* is not registered.
    NotImplementedError
        If no defaults mapping is defined for this spec type.
    """
    spec = get_spec(model_id)

    if isinstance(spec, DemucsSpec):
        return {
            "model_name": spec.model_id,
            "sample_rate": spec.sample_rate,
        }

    if isinstance(spec, BasicPitchSpec):
        return {
            "onset_threshold": spec.default_onset,
            "frame_threshold": spec.default_frame,
            "minimum_note_length": spec.default_min_note_ms,
            "minimum_frequency": None,
            "maximum_frequency": None,
        }

    if isinstance(spec, WhisperSpec):
        return {
            "model_size": spec.model_size,
            "device": spec.device,
            "compute_type": spec.compute_type,
            "language": spec.default_language,
            "word_timestamps": spec.word_timestamps,
            "vad_filter": spec.vad_filter,
        }

    if isinstance(spec, StableAudioSpec):
        return {
            "duration_seconds": 10.0,
            "steps": spec.default_steps,
            "cfg_scale": spec.default_cfg_scale,
        }

    if isinstance(spec, RoformerSpec):
        return {
            "model_id": spec.model_id,
            "sample_rate": spec.sample_rate,
            "chunk_size": spec.default_chunk_size,
            "num_overlap": spec.default_num_overlap,
        }

    raise NotImplementedError(
        f"No pipeline defaults defined for spec type {type(spec).__name__!r}."
    )


def get_gui_metadata(model_id: str) -> dict[str, Any]:
    """Return GUI-specific metadata for *model_id*.

    Includes slider ranges, allowed values, model selector lists, and
    tooltip text.  Keys vary by spec type:

    * **DemucsSpec** — ``available_stems``, ``model_choices``, ``tooltip``
    * **BasicPitchSpec** — ``onset_range``, ``frame_range``,
      ``min_note_range``, ``default_onset``, ``default_frame``,
      ``default_min_note_ms``, ``tooltip``
    * **WhisperSpec** — ``model_size``, ``tooltip``
    * **StableAudioSpec** — ``max_duration``, ``steps_range``,
      ``default_steps``, ``default_cfg_scale``, ``tooltip``

    Parameters
    ----------
    model_id:
        Registered model ID.

    Returns
    -------
    dict[str, Any]
        GUI metadata dict.

    Raises
    ------
    KeyError
        If *model_id* is not registered.
    NotImplementedError
        If no GUI metadata mapping is defined for this spec type.
    """
    spec = get_spec(model_id)

    if isinstance(spec, DemucsSpec):
        return {
            "available_stems": list(spec.available_stems),
            "model_choices": [s.model_id for s in list_specs(DemucsSpec)],
            "tooltip": spec.description,
        }

    if isinstance(spec, BasicPitchSpec):
        return {
            "onset_range": spec.onset_range,
            "frame_range": spec.frame_range,
            "min_note_range": spec.min_note_range,
            "default_onset": spec.default_onset,
            "default_frame": spec.default_frame,
            "default_min_note_ms": spec.default_min_note_ms,
            "tooltip": spec.description,
        }

    if isinstance(spec, WhisperSpec):
        return {
            "model_size": spec.model_size,
            "tooltip": spec.description,
        }

    if isinstance(spec, StableAudioSpec):
        return {
            "max_duration": spec.max_duration_seconds,
            "steps_range": (10, 200),
            "default_steps": spec.default_steps,
            "default_cfg_scale": spec.default_cfg_scale,
            "tooltip": spec.description,
        }

    if isinstance(spec, RoformerSpec):
        return {
            "available_stems": list(spec.available_stems),
            "model_choices": [s.model_id for s in list_specs(RoformerSpec)],
            "tooltip": spec.description,
            "target_instrument": spec.target_instrument,
        }

    raise NotImplementedError(
        f"No GUI metadata defined for spec type {type(spec).__name__!r}."
    )
