# Lyrics Transcription — Integration Spec

**Status:** Ready for implementation
**Target tab:** MIDI (add `Notes · Lyrics` sub-mode bar, mirroring the Enhance tab pattern)
**Estimated scope:** ~1 new pipeline, 1 new API router, 1 new engine module, 2 refactors, frontend additions
**Invocation:** `uv run stemforge`

---

## 0 · License Compliance

This integration adds **Qwen2-Audio-7B-Instruct** as an optional transcription engine alongside `faster-whisper` (MIT, already integrated).

**License:** Apache 2.0 (confirmed on the HuggingFace model card at `https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct`). Compatible with StemForge's Apache 2.0 stance — no commercial-use restrictions, no MAU thresholds, no field-of-use carve-outs.

**Required compliance steps** (Apache 2.0, §4):

1. **Copy the license text** into the repo as `licenses/LICENSE-Qwen2-Audio` (verbatim from the upstream `LICENSE` file at the Qwen2-Audio HF repo). If a `licenses/` directory doesn't exist, create it; this is where future third-party license texts will live too.
2. **Add an attribution entry** to `ACKNOWLEDGMENTS.md` (see §3.7 for the exact text).
3. **Preserve any `NOTICE` file** if one ships with the weights. As of the last check there is none in the Qwen2-Audio HF repo, but check at implementation time — if `NOTICE` exists, copy it to `licenses/NOTICE-Qwen2-Audio`.
4. **State changes** — N/A. This integration only runs inference; no fine-tuning, no weight modification. If that ever changes, Apache 2.0 §4(b) requires noting modifications.

**Verification step before merging:** open `https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct` and confirm the license badge still says `apache-2.0`. If it has changed (which would be unusual but not impossible), stop and surface the change to the user before proceeding.

**Note for future upgrades:** Qwen licensing varies across model families and sizes. Qwen2.5-VL 3B and 72B, for instance, ship under the custom "Qwen License" with commercial restrictions, while the 7B variants are Apache 2.0. If this engine is ever swapped for a newer Qwen model (Qwen2.5-Omni, Qwen3-Audio, etc.), repeat the license check from scratch rather than inferring from the family name.

---

## 1 · Goals

- Add a **standalone lyrics transcription feature** that takes any audio (vocal stem or full mix) and produces lyrics as `.txt`, `.lrc`, and `.srt` files.
- Make the **engine user-selectable** at request time: Whisper variants (tiny/base/small/medium/large-v3) vs. Qwen2-Audio.
- **Refactor** the two existing Whisper integration points (`models/midi_loader.py::_ensure_whisper` and `pipelines/vocal_midi_pipeline.py::load_model`) to consume the new engine interface instead of loading Whisper themselves. Net: one source of truth for transcription, less duplicated code.
- Results flow into the existing Export tab and Mix tab metadata, following the same pattern as `enhance_paths` and `voice_paths`.

**Out of scope for v1** — see §7.

---

## 2 · Architectural Decisions

1. **New pipeline `TranscribePipeline`** lives in `pipelines/transcribe_pipeline.py`, follows the same lifecycle pattern as `EnhancePipeline` (`configure → load_model → run → clear`).
2. **Pluggable engines** live in `pipelines/transcribe_engines/` as a sub-package. Each engine is a class implementing a common Protocol (Whisper-style duck-typed interface). New engines drop in without touching the pipeline.
3. **Whisper engine wraps `faster-whisper`** — already a dependency. Adds support for `large-v3` (currently missing from the registry).
4. **Qwen engine uses `transformers`** — already a dependency (used by Stable Audio Open and AceStep tooling). Loads `Qwen2-Audio-7B-Instruct` via `Qwen2AudioForConditionalGeneration`. Qwen does not emit word timestamps natively; the engine reports `supports_word_timestamps = False` and the pipeline degrades `.lrc`/`.srt` output to segment-level timing only when Qwen is selected.
5. **MIDI tab gets a mode bar** — `Notes · Lyrics`, mirroring the Enhance tab's `Clean Up · Tune · Effects` pattern. Default sub-mode is `Notes` so existing users see no behaviour change on first load.
6. **GPU scheduling** uses the existing `pipeline_manager.gpu_session(pipeline_hint="transcribe")`. Qwen requires GPU (≥10 GB VRAM); the engine raises a clean error if CUDA is unavailable. Whisper engines remain CPU-capable.
7. **Output directory** — new `LYRICS_DIR = OUTPUT_BASE / "lyrics"`, added to `utils/paths.py` and the allowlist in `backend/api/audio.py::_ALLOWED_ROOTS`.
8. **Session state** — new `_lyrics_paths: dict[str, pathlib.Path]` field on `SessionStore`, with `add_lyrics_path()` method and inclusion in `to_dict()` and `clear()`.

---

## 3 · New Files

### 3.1 · `pipelines/transcribe_engines/__init__.py`

```python
"""Pluggable transcription engines for StemForge.

Each engine implements the TranscriptionEngine protocol and is registered
in ENGINES below.  The TranscribePipeline selects an engine by ID at
configure() time.
"""
from __future__ import annotations

from .types import (
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptionSegment,
    WordTiming,
)
from .whisper_engine import WhisperEngine
from .qwen_engine import QwenEngine

# Maps engine_id → factory callable. Engine IDs are flat strings,
# distinct from model_ids in models/registry.py.
ENGINES: dict[str, type[TranscriptionEngine]] = {
    "whisper": WhisperEngine,
    "qwen": QwenEngine,
}

__all__ = [
    "ENGINES",
    "TranscriptionEngine",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WordTiming",
    "WhisperEngine",
    "QwenEngine",
]
```

### 3.2 · `pipelines/transcribe_engines/types.py`

```python
"""Shared types for transcription engines."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class WordTiming:
    word: str
    start: float            # seconds
    end: float              # seconds
    probability: float | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    start: float            # seconds
    end: float              # seconds
    text: str
    words: list[WordTiming] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str               # full concatenated transcript
    language: str | None    # ISO-639-1, e.g. "en"
    segments: list[TranscriptionSegment]
    has_word_timestamps: bool
    engine_id: str          # which engine produced this result
    model_id: str           # specific model variant


@runtime_checkable
class TranscriptionEngine(Protocol):
    """Common interface for all transcription backends."""

    engine_id: str          # e.g. "whisper", "qwen"
    model_id: str           # e.g. "whisper-large-v3", "qwen2-audio-7b-instruct"
    supports_word_timestamps: bool
    requires_gpu: bool

    def load(self) -> None:
        """Load model weights into memory.  Idempotent."""
        ...

    def transcribe(
        self,
        audio_path: pathlib.Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        """Run transcription on a single audio file."""
        ...

    def clear(self) -> None:
        """Release model weights and trigger GC."""
        ...
```

### 3.3 · `pipelines/transcribe_engines/whisper_engine.py`

Wraps `faster-whisper`. Replaces the inline Whisper loading currently duplicated in `models/midi_loader.py` and `pipelines/vocal_midi_pipeline.py`.

Key behaviour:
- Accepts a `model_id` parameter resolved via `models/registry.py::get_spec()` — supports all registered `WhisperSpec` entries.
- On CUDA: uses `device="cuda"`, `compute_type="float16"`. On CPU: `device="cpu"`, `compute_type="int8"`.
- Calls `model.transcribe(path, word_timestamps=True, vad_filter=True, language=language, initial_prompt=prompt)`.
- Converts faster-whisper's `Segment`/`Word` objects into the dataclasses in `types.py`.
- `supports_word_timestamps = True`, `requires_gpu = False`.

The implementation closely mirrors `models/midi_loader.py::_ensure_whisper` plus the transcription loop in `convert_vocal_to_midi` — but stripped of PYIN/MIDI concerns. Use `utils.cache.get_model_cache_dir("whisper")` for `download_root`.

### 3.4 · `pipelines/transcribe_engines/qwen_engine.py`

Wraps Qwen2-Audio-7B-Instruct via `transformers`.

Key behaviour:
- Uses `Qwen2AudioForConditionalGeneration` and `AutoProcessor` from `transformers`.
- Hard-fails with `ModelLoadError` if `torch.cuda.is_available()` is False (Qwen on CPU is impractical).
- `supports_word_timestamps = False`, `requires_gpu = True`.
- Prompt to the model:
  ```
  Transcribe the lyrics of this audio. Output only the lyrics text,
  preserving line breaks where the singer pauses. Do not add commentary,
  explanations, or section labels.
  ```
- Loads audio as mono 16 kHz numpy array via `librosa.load`.
- Caches the model under `utils.cache.get_model_cache_dir("qwen2-audio")`.
- Returns a single `TranscriptionSegment` covering the full audio duration with the full lyrics as `text` and an empty `words` list. The pipeline (§3.5) detects `has_word_timestamps = False` and adjusts output formats accordingly.

```python
"""Qwen2-Audio engine for high-fidelity sung-lyrics transcription.

LICENSE NOTE: Verify license at https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
before enabling this engine.  See LYRICS_TRANSCRIPTION_SPEC.md §0.
"""
from __future__ import annotations

import logging
import pathlib

import torch

from utils.cache import get_model_cache_dir
from utils.errors import ModelLoadError, PipelineExecutionError

from .types import (
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptionSegment,
    WordTiming,
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

    def __init__(self) -> None:
        self._model = None
        self._processor = None

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
                _QWEN_REPO, cache_dir=cache_dir
            )
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                _QWEN_REPO,
                cache_dir=cache_dir,
                torch_dtype=torch.float16,
                device_map="cuda",
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
                "librosa is required.", model_name=self.model_id
            ) from exc

        try:
            audio, _sr = librosa.load(str(audio_path), sr=16_000, mono=True)
        except Exception as exc:
            raise PipelineExecutionError(
                f"Failed to load audio '{audio_path.name}': {exc}",
                pipeline_name="transcribe",
            ) from exc

        # Qwen2-Audio chat-style prompt
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
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            text=text, audios=[audio], sampling_rate=16_000,
            return_tensors="pt", padding=True,
        ).to("cuda")

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
```

### 3.5 · `pipelines/transcribe_pipeline.py`

The orchestration layer. Selects an engine, runs it, writes outputs (`.txt`, `.lrc`, `.srt`).

```python
"""Lyrics transcription pipeline.

Selects a transcription engine (Whisper or Qwen), runs transcription on
a single audio file, and writes plain text, LRC, and SRT outputs.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Callable

from pipelines.transcribe_engines import (
    ENGINES,
    TranscriptionEngine,
    TranscriptionResult,
)
from utils.errors import InvalidInputError, PipelineExecutionError

log = logging.getLogger(__name__)

_SUPPORTED_EXTS = frozenset({".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"})


@dataclass(slots=True)
class TranscribeConfig:
    engine_id: str = "whisper"
    model_id: str = "whisper-base"       # only used by whisper engine
    language: str | None = None          # None = auto-detect (whisper only)
    prompt: str | None = None            # optional initial prompt
    output_dir: pathlib.Path | None = None
    formats: tuple[str, ...] = ("txt", "lrc", "srt")  # which files to write


@dataclass(slots=True)
class TranscribeResult:
    result: TranscriptionResult          # raw engine output
    output_paths: dict[str, pathlib.Path]  # format → file path
    label: str                           # display label for session


class TranscribePipeline:
    """Lifecycle: configure → load_model → run → clear."""

    def __init__(self) -> None:
        self._config: TranscribeConfig | None = None
        self._engine: TranscriptionEngine | None = None

    def configure(self, config: TranscribeConfig) -> None:
        if config.engine_id not in ENGINES:
            raise InvalidInputError(
                f"Unknown engine_id {config.engine_id!r}. "
                f"Available: {sorted(ENGINES)}",
                field="engine_id",
            )
        self._config = config

    def load_model(self) -> None:
        if self._config is None:
            raise PipelineExecutionError(
                "configure() must be called before load_model().",
                pipeline_name="transcribe",
            )
        engine_cls = ENGINES[self._config.engine_id]
        # Whisper engine takes model_id; Qwen does not.
        if self._config.engine_id == "whisper":
            self._engine = engine_cls(model_id=self._config.model_id)
        else:
            self._engine = engine_cls()
        self._engine.load()

    def run(
        self,
        audio_path: pathlib.Path,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> TranscribeResult:
        if self._config is None or self._engine is None:
            raise PipelineExecutionError(
                "load_model() must be called before run().",
                pipeline_name="transcribe",
            )
        if not audio_path.exists():
            raise InvalidInputError(
                f"Audio file not found: {audio_path}", field="audio_path",
            )
        if audio_path.suffix.lower() not in _SUPPORTED_EXTS:
            raise InvalidInputError(
                f"Unsupported audio format: {audio_path.suffix}", field="audio_path",
            )

        if progress_cb:
            progress_cb(0.1, "Transcribing…")

        result = self._engine.transcribe(
            audio_path,
            language=self._config.language,
            prompt=self._config.prompt,
        )

        if progress_cb:
            progress_cb(0.85, "Writing outputs…")

        out_dir = self._config.output_dir or audio_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = audio_path.stem
        engine_tag = result.engine_id
        base = f"{stem}-lyrics-{engine_tag}"

        output_paths: dict[str, pathlib.Path] = {}
        for fmt in self._config.formats:
            if fmt == "txt":
                p = out_dir / f"{base}.txt"
                p.write_text(result.text, encoding="utf-8")
                output_paths["txt"] = p
            elif fmt == "lrc":
                p = out_dir / f"{base}.lrc"
                p.write_text(_format_lrc(result), encoding="utf-8")
                output_paths["lrc"] = p
            elif fmt == "srt":
                p = out_dir / f"{base}.srt"
                p.write_text(_format_srt(result), encoding="utf-8")
                output_paths["srt"] = p

        if progress_cb:
            progress_cb(1.0, "Done")

        return TranscribeResult(
            result=result,
            output_paths=output_paths,
            label=f"{stem} ({engine_tag})",
        )

    def clear(self) -> None:
        if self._engine is not None:
            self._engine.clear()
            self._engine = None


# ── Format helpers ───────────────────────────────────────────────────

def _format_lrc(result: TranscriptionResult) -> str:
    """LRC karaoke format.

    With word timestamps: one line per word with [mm:ss.xx] prefix.
    Without word timestamps (Qwen): one line per segment with segment-start prefix.
    """
    lines: list[str] = []
    if result.has_word_timestamps:
        for seg in result.segments:
            for w in seg.words:
                lines.append(f"[{_lrc_timestamp(w.start)}]{w.word}")
    else:
        for seg in result.segments:
            for line in seg.text.splitlines():
                if line.strip():
                    lines.append(f"[{_lrc_timestamp(seg.start)}]{line.strip()}")
    return "\n".join(lines) + "\n"


def _format_srt(result: TranscriptionResult) -> str:
    """SRT subtitle format. Always segment-level — never per-word."""
    lines: list[str] = []
    for i, seg in enumerate(result.segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _lrc_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

### 3.6 · `backend/api/transcribe.py`

```python
"""Lyrics transcription endpoints."""
from __future__ import annotations

import pathlib
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.services.job_manager import job_manager
from backend.services.session_store import SessionStore, get_user_session
from backend.services import pipeline_manager
from utils.paths import LYRICS_DIR, user_dir

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


class TranscribeRequest(BaseModel):
    audio_path: str
    engine_id: str = "whisper"
    model_id: str = "whisper-base"      # ignored by qwen
    language: str | None = None
    prompt: str | None = None
    formats: list[str] = ["txt", "lrc", "srt"]


@router.get("/engines")
def list_engines() -> dict:
    """Return available engines and their capabilities for the UI."""
    import torch
    from pipelines.transcribe_engines import ENGINES
    from models.registry import list_specs, WhisperSpec

    cuda = torch.cuda.is_available()
    engines = []
    for engine_id, cls in ENGINES.items():
        # Instantiate without loading to read static attributes
        info = {
            "engine_id": engine_id,
            "supports_word_timestamps": cls.supports_word_timestamps,
            "requires_gpu": cls.requires_gpu,
            "available": (not cls.requires_gpu) or cuda,
            "models": [],
        }
        if engine_id == "whisper":
            info["models"] = [
                {"model_id": s.model_id, "display_name": s.display_name}
                for s in list_specs(WhisperSpec)
            ]
        elif engine_id == "qwen":
            info["models"] = [
                {"model_id": "qwen2-audio-7b-instruct",
                 "display_name": "Qwen2-Audio 7B Instruct"},
            ]
        engines.append(info)
    return {"engines": engines, "cuda_available": cuda}


def _run_transcribe(
    req: TranscribeRequest,
    job_id: str,
    session: SessionStore,
) -> dict:
    from pipelines.transcribe_pipeline import TranscribePipeline, TranscribeConfig

    progress_cb = job_manager.make_progress_callback(job_id)
    audio_path = pathlib.Path(req.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {req.audio_path}")

    out_dir = user_dir(LYRICS_DIR, session.user)
    config = TranscribeConfig(
        engine_id=req.engine_id,
        model_id=req.model_id,
        language=req.language,
        prompt=req.prompt,
        output_dir=out_dir,
        formats=tuple(req.formats),
    )

    pipeline_hint = "transcribe"
    with pipeline_manager.gpu_session(pipeline_hint=pipeline_hint) as ctx:
        pipeline = pipeline_manager.get_transcribe(ctx.gpu_index)
        pipeline.configure(config)
        try:
            progress_cb(0.05, "Loading model…")
            pipeline.load_model()
            result = pipeline.run(audio_path, progress_cb=progress_cb)
        finally:
            pipeline_manager.evict("transcribe", ctx.gpu_index)

    # Store in session by format
    for fmt, path in result.output_paths.items():
        label = f"{result.label} [{fmt}]"
        session.add_lyrics_path(label, path)

    return {
        "engine_id": result.result.engine_id,
        "model_id": result.result.model_id,
        "language": result.result.language,
        "has_word_timestamps": result.result.has_word_timestamps,
        "text": result.result.text,
        "segment_count": len(result.result.segments),
        "output_paths": {k: str(v) for k, v in result.output_paths.items()},
        "label": result.label,
    }


@router.post("")
def start_transcribe(
    req: TranscribeRequest,
    request: Request,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    user = getattr(request.state, "user", "local")
    job_id = job_manager.create_job("transcribe", user=user)
    job_manager.run_job(job_id, _run_transcribe, req, job_id, session)
    return {"job_id": job_id}
```

### 3.7 · `ACKNOWLEDGMENTS.md` addition

Append a new section under the existing Whisper entry:

```markdown
---

## Qwen2-Audio — Alibaba Cloud (Tongyi Lab)

Multimodal audio understanding model used as an optional engine in the
Lyrics transcription feature on the MIDI tab.

- **Model:** https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
- **Paper:** Chu et al. — *Qwen2-Audio Technical Report* (2024)
- **License:** Apache 2.0 (verified at integration time — see `licenses/LICENSE-Qwen2-Audio`)
```

Also copy the upstream `LICENSE` file from the Qwen2-Audio HF repo to `licenses/LICENSE-Qwen2-Audio`. Apache 2.0 §4 requires the license text to accompany any distribution that references the work; the `licenses/` directory keeps third-party license texts isolated from the project's own root `LICENSE` file.

---

## 4 · Modified Files

### 4.1 · `models/registry.py` — add `WHISPER_LARGE_V3`

Insert after `WHISPER_MEDIUM`:

```python
WHISPER_LARGE_V3 = _register(WhisperSpec(
    model_id="whisper-large-v3",
    display_name="Whisper large-v3",
    version="1.1.0",
    source="openai/whisper-large-v3",
    device="cpu",
    gpu_capable=True,                  # large-v3 benefits from GPU
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
```

### 4.2 · `utils/paths.py` — add `LYRICS_DIR`

```python
LYRICS_DIR   = OUTPUT_BASE / "lyrics"
```

### 4.3 · `backend/api/audio.py` — allowlist

Add `LYRICS_DIR` to the import block and to `_ALLOWED_ROOTS`.

### 4.4 · `backend/main.py` — wire router + ensure directory

```python
from backend.api import (
    system, audio, separate, midi, generate, mix, export,
    compose, sfx, voice, enhance,
    transcribe,                       # NEW
)
from utils.paths import (
    OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR,
    COMPOSE_DIR, SFX_DIR, VOICE_DIR, ENHANCE_DIR,
    LYRICS_DIR,                       # NEW
)

# …

app.include_router(transcribe.router)  # NEW (after enhance.router)

for d in (OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR,
          COMPOSE_DIR, SFX_DIR, VOICE_DIR, ENHANCE_DIR,
          LYRICS_DIR):                # NEW
    d.mkdir(parents=True, exist_ok=True)
```

### 4.5 · `backend/services/pipeline_manager.py` — add `get_transcribe`

Inside `_get_or_create`:

```python
        elif name == "transcribe":
            from pipelines.transcribe_pipeline import TranscribePipeline
            cache[name] = TranscribePipeline()
```

And add the accessor:

```python
def get_transcribe(gpu_index: int | None = None) -> Any:
    return _get_or_create("transcribe", gpu_index)
```

### 4.6 · `backend/services/session_store.py` — add `lyrics_paths`

Add to `__init__`:

```python
        self._lyrics_paths: dict[str, pathlib.Path] = {}
```

Add property + adder (after `enhance_paths`):

```python
    @property
    def lyrics_paths(self) -> dict[str, pathlib.Path]:
        with self._lock:
            return dict(self._lyrics_paths)

    def add_lyrics_path(self, label: str, path: pathlib.Path) -> None:
        with self._lock:
            self._lyrics_paths[label] = path
```

Include in `clear()`:

```python
            self._lyrics_paths = {}
```

Include in `to_dict()`:

```python
                "lyrics_paths": {k: str(v) for k, v in self._lyrics_paths.items()},
```

### 4.7 · `models/midi_loader.py` — refactor `convert_vocal_to_midi`

Replace the inline `_ensure_whisper` + transcription loop with a call to the new engine:

1. Remove `_ensure_whisper` method, `_whisper_model` field, and the `DEFAULT_WHISPER_SPEC` import.
2. In `convert_vocal_to_midi`, replace the Whisper section with:

```python
from pipelines.transcribe_engines import WhisperEngine

# Replace _ensure_whisper() + model.transcribe(...) block with:
engine = WhisperEngine(model_id="whisper-base")
engine.load()
try:
    transcription = engine.transcribe(path, language=language)
finally:
    engine.clear()

# Iterate over transcription.segments[].words instead of segment.words
```

The rest of `convert_vocal_to_midi` (PYIN + word→note mapping) stays the same — it now consumes `WordTiming` dataclasses instead of faster-whisper's native objects (same field names: `.word`, `.start`, `.end`, `.probability`).

### 4.8 · `pipelines/vocal_midi_pipeline.py` — refactor `load_model` and Whisper usage

1. Remove the `_whisper_model` field and the Whisper section of `load_model`.
2. Add `_transcribe_engine: WhisperEngine | None = None`.
3. In `load_model`, replace the Whisper block with:

```python
from pipelines.transcribe_engines import WhisperEngine

self._transcribe_engine = WhisperEngine(
    model_id=f"whisper-{self._config.whisper_model_size}",
)
self._transcribe_engine.load()
```

4. Replace any subsequent `self._whisper_model.transcribe(...)` calls with `self._transcribe_engine.transcribe(...)`, consuming `TranscriptionResult` instead of the raw iterator.
5. In `clear()`, call `self._transcribe_engine.clear()`.

These two refactors (4.7 and 4.8) are the consolidation goal: after them, **only `WhisperEngine` ever calls `faster_whisper.WhisperModel`** in the codebase. Grep `from faster_whisper` to confirm — should be one import site post-refactor.

---

## 5 · Frontend (`frontend/components/midi.js`)

The existing MIDI tab is a `two-col` layout. Add a **mode bar** above the left column controls, mirroring `enhance.js`. Default mode is `notes`; switching to `lyrics` swaps the left-column control set and the right-column result area.

### 5.1 · Mode bar

```js
// Near the top of initMidi(), before the existing controls:
const modeBar = el('div', { className: 'midi-mode-bar' },
  el('button', { className: 'midi-mode-btn active', 'data-mode': 'notes',
                 onClick: () => switchMidiMode('notes') }, 'Notes'),
  el('button', { className: 'midi-mode-btn', 'data-mode': 'lyrics',
                 onClick: () => switchMidiMode('lyrics') }, 'Lyrics'),
);
left.appendChild(modeBar);
```

Reuse Enhance's `.enhance-mode-bar` CSS by adding `.midi-mode-bar` with identical styles, or factor into a shared `.tab-mode-bar` class in `frontend/style.css`.

### 5.2 · Lyrics control panel

Hidden by default. Visible only when `_midiMode === 'lyrics'`.

Controls:
- **Stem selector** — reuse the existing stem checkbox group, but single-select (radio behaviour) in lyrics mode. Default to the vocal stem when present.
- **Engine** — `<select>` populated from `GET /api/transcribe/engines`. Options are grouped: Whisper variants (tiny → large-v3) and Qwen if available. Greyed out engines with a `(GPU required)` suffix when `available: false`.
- **Language hint** — same 12-language dropdown used in Compose (`compose.js::buildLanguageSelect`). First option is `Auto-detect`.
- **Output formats** — three checkboxes: `txt` (always checked, disabled), `lrc`, `srt`. When the selected engine has `supports_word_timestamps: false`, show an inline note: *"Qwen produces segment-level timing; .lrc/.srt will use coarse timestamps."*
- **Transcribe button** — primary action, calls `POST /api/transcribe` with the form payload and polls the job.

### 5.3 · Results panel

On job success, render a card in the right column with:
- The transcribed text in a `<textarea readonly>` (scrollable, monospace).
- Engine + model badges.
- Save buttons: `Save .txt`, `Save .lrc`, `Save .srt` (only those formats that were requested).
- A `Send to Compose` button that copies the text into `appState` and emits `lyricsReady` so the Compose tab can prefill its lyrics textarea when the user navigates there.

### 5.4 · Event bus

Add `lyricsReady` to `frontend/app.js::appState`:

```js
lyricsPaths: {},  // label → path
```

Compose tab's `initCompose` should listen for `lyricsReady` and offer to apply the text to the `compose-lyrics-text` textarea (with a small toast/banner asking for confirmation, since the user may not want to overwrite typed content).

### 5.5 · Loader hint

When entering Lyrics mode with no vocal stem loaded, show: *"Load audio or run separation first. Lyrics transcription works best on an isolated vocal stem."*

---

## 6 · Testing

### 6.1 · Smoke test — `tests/test_transcribe.py`

Modelled after `tests/test_faster_whisper.py`:

```python
"""Smoke test for TranscribePipeline with the Whisper engine.

Runs the full pipeline on tests/data/silence.wav and confirms:
  - Pipeline configures and loads without error
  - run() produces a TranscribeResult with output_paths for all formats
  - Output files exist and are non-empty (txt may be empty on silence)
  - .srt and .lrc are parseable
"""
import pathlib
import tempfile
from pipelines.transcribe_pipeline import TranscribePipeline, TranscribeConfig


def main() -> None:
    audio = pathlib.Path("tests/data/silence.wav")
    assert audio.exists(), "Missing test fixture"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = pathlib.Path(tmp)
        pipeline = TranscribePipeline()
        pipeline.configure(TranscribeConfig(
            engine_id="whisper",
            model_id="whisper-tiny",   # smallest for CI speed
            output_dir=out_dir,
        ))
        pipeline.load_model()
        try:
            result = pipeline.run(audio)
        finally:
            pipeline.clear()

        assert "txt" in result.output_paths
        assert "lrc" in result.output_paths
        assert "srt" in result.output_paths
        for p in result.output_paths.values():
            assert p.exists(), f"Missing output: {p}"
        print("transcribe pipeline OK")


if __name__ == "__main__":
    main()
```

Qwen is intentionally **not** smoke-tested in CI — it requires GPU, is large to download, and may have a license gate. Add a manual test note in `tests/README.md`.

### 6.2 · Manual checklist

- [ ] Separate a song → vocal stem appears in session.
- [ ] Switch to MIDI tab → Notes mode still works exactly as before (regression check on refactor §4.7).
- [ ] Switch to Lyrics mode → engine dropdown lists Whisper variants and Qwen.
- [ ] Run Whisper transcription on the vocal stem → all three output files appear in Export tab.
- [ ] On a GPU machine, run Qwen on the same stem → `.txt` populated; `.lrc`/`.srt` use segment-level timing; banner about coarse timing is shown.
- [ ] On a CPU-only machine, Qwen is greyed out in the dropdown with `(GPU required)` suffix.
- [ ] `Send to Compose` → switching to Compose shows the transcribed text in the lyrics textarea (with confirmation).
- [ ] `New Session` → `lyrics_paths` is cleared.

---

## 7 · Out of Scope (Not for v1)

- **Forced alignment hybrid mode** (Qwen text → Whisper word-timestamps via alignment-only pass). Worthwhile as a v2: gives accurate words + accurate timing in one combined output. Add as a third engine_id like `qwen+align` later.
- **Translation mode** (`task="translate"` in Whisper). Easy add later; controlled by a new field in `TranscribeConfig`.
- **Batch transcription** mirroring `enhance_batch`. Same shape as the existing batch endpoints, low-risk add once v1 is shipped.
- **In-place lyric editing** in the Lyrics result card. Currently the textarea is readonly; an edit mode + re-save would be nice but isn't critical for v1.
- **Sheet music sync** (using lyrics-with-timing to align with the MIDI sheet music view). Belongs in the music21 integration phase, not here.

---

## 8 · Definition of Done

1. Apache 2.0 verified on the Qwen2-Audio HF model card (§0); `licenses/LICENSE-Qwen2-Audio` populated; `ACKNOWLEDGMENTS.md` entry added.
2. `uv run stemforge` starts cleanly with no new warnings.
3. Manual checklist (§6.2) passes end-to-end on Tsondo's machine.
4. `grep -rn "from faster_whisper" --include='*.py'` returns exactly **one** match — inside `pipelines/transcribe_engines/whisper_engine.py`.
5. `tests/test_transcribe.py` runs green.
6. No regressions on `tests/test_faster_whisper.py` or the MIDI tab's existing vocal → MIDI flow.
7. `docs/CURRENT_STATE.md` is updated with one line under "What's working" describing the new Lyrics feature.
8. `docs/INSTRUCTIONS.md` section 4 (MIDI) gets a new sub-section describing the Notes/Lyrics mode bar.

*(Addendum 1 — Whisper hallucination mitigation)*

9. `condition_on_previous_text=False` is the default in `WhisperEngine`, surfaced through `TranscribeConfig` and the API request schema.
10. `_collapse_repetitions` helper exists, is wired into `TranscribePipeline.run()`, defaults to enabled with `max_run=4`.
11. Both Advanced toggles render in the MIDI tab's Lyrics mode and round-trip correctly (off-by-default for conditioning, on-by-default for collapse).
12. `tests/test_transcribe.py::test_collapse_repetitions` passes.
13. Re-running the test track shows no spurious `¡Oh, oh, oh!` tail.

*(Addendum 2 — Registry pruning + Qwen NF4 variant)*

14. `models/registry.py` contains exactly three Whisper specs: `WHISPER_TINY`, `WHISPER_SMALL`, `WHISPER_LARGE_V3`. `DEFAULT_WHISPER_SPEC` points to `WHISPER_LARGE_V3`.
15. `grep -rn "whisper-base\|whisper-medium" --include='*.py' --include='*.js' --include='*.md'` returns no functional references (only changelog / historical mentions, if any).
16. `pyproject.toml` lists `bitsandbytes>=0.43.0`. `uv.lock` is refreshed.
17. `pipelines/transcribe_engines/qwen_engine.py` accepts a `model_id` constructor arg and supports both `qwen2-audio-7b-instruct` (fp16) and `qwen2-audio-7b-instruct-nf4` (NF4).
18. `/api/transcribe/engines` returns five model entries total: 3 Whisper + 2 Qwen, each with `available` and (for Qwen) `approx_vram_gb` fields.
19. MIDI Lyrics dropdown shows exactly five annotated entries matching §3.1.
20. `ACKNOWLEDGMENTS.md` contains the new bitsandbytes entry.
21. Manual checklist (§5.3 of Addendum 2) passes on Tsondo's laptop, specifically the Qwen 4-bit transcription run.

*(Addendum 3 — Overlap-and-stitch chunking for Qwen)*

22. `pipelines/transcribe_engines/_qwen_chunker.py` exists with `slice_audio` and `stitch_chunks` implemented per §3.
23. `QwenEngine.transcribe()` uses the chunker; `_transcribe_chunk` helper extracted.
24. All six new unit tests in `tests/test_transcribe.py` pass.
25. Re-running the Catrina stem through Qwen 4-bit shows no chunk-boundary gibberish ("Eless con piernas"-type failures).
26. Whisper transcription is byte-for-byte unchanged — chunker code is reachable only via `QwenEngine`.
27. Logs show `Qwen transcribing N.Ns of audio in M chunk(s).` followed by per-pair `Stitched chunk X → X+1: matched Y tokens` lines on a multi-chunk run.

*(Addendum 4 — Pairwise matching + relaxed match threshold)*

28. `stitch_chunks` matches against `prev_tokens` (immediately previous chunk only), not against the full accumulated output.
29. `_is_acceptable_match` helper, `MIN_MATCH_TOKENS_STRICT`, `MIN_SINGLE_TOKEN_LENGTH`, and `_SINGLE_TOKEN_STOPLIST` are present in `_qwen_chunker.py`.
30. Old `MIN_MATCH_TOKENS` constant is removed.
31. Four new unit tests in `tests/test_transcribe.py` pass.
32. All existing tests still pass.
33. Re-running the Catrina stem through Qwen 4-bit produces a continuous second chorus with no `\n` fallback markers in that region.

*(Addendum 5 — Surface the Hint field in the Lyrics UI)*

34. MIDI Lyrics control panel contains a `Hint (optional)` text input between Language and Output formats, with the tooltip and placeholder text specified in §2.
35. `maxlength` of 224 characters enforced via HTML attribute.
36. Hint value persists in `appState.lyricsHint` across Notes/Lyrics mode switches.
37. Transcribe requests include the `prompt` field only when the hint is non-empty.
38. `New Session` clears `appState.lyricsHint`.
39. Manual hint test against the Catrina stem with Qwen 4-bit shows reduced proper-noun drift compared to the no-hint baseline from Addendum 4.

*(Addendum 6 — Qwen hint wrapper with anti-translation anchor)*

40. `qwen_engine.py` exposes a `_build_qwen_prompt(hint, language) -> str` helper that wraps non-empty hints with explicit vocabulary-guidance framing and anti-translation anchoring.
41. `_BASE_PROMPT` and `_HINT_WRAPPER` module constants present; old `_PROMPT` reference removed.
42. `_transcribe_chunk` calls `_build_qwen_prompt` instead of inline assembly.
43. `test_qwen_prompt_construction` unit test passes.
44. All existing tests in `tests/test_transcribe.py` continue to pass.
45. Manual hint test on Catrina stem with Qwen 4-bit shows verbatim Spanish lyrics (no "The lyrics translate to..." prefix) and uses the hint's spelling for proper nouns.
46. Whisper engine code is byte-for-byte unchanged.

*(Addendum 7 Phase 1 — Diagnostic logging for hint-induced truncation)*

47. `qwen_engine.py` emits one INFO-level log line per chunk in the format `Qwen chunk N/M (X.X-Y.Ys): Z chars | <preview>` where preview is the first 80 chars of the raw model output with newlines rendered as `⏎`.
48. `_qwen_chunker.py` emits one INFO-level log line per stitch decision, both for successful stitches (`Stitched chunk N → N+1: matched X tokens of A/B tail/head (...)`) and for fallback non-matches.
49. No other behavior changes — re-running the Catrina stem produces the same truncated output as before, with the addition of the diagnostic log lines.
50. Phase 2 of this addendum (the actual fix) is held pending review of the log output.

*(Addendum 7 Phase 2 — Fake-assistant-turn structural separation)*

51. `qwen_engine.py` exposes `_build_qwen_conversation(hint, language) -> list[dict]` returning a 3-turn user/assistant/user conversation structure.
52. Audio placeholder appears in turn 3 only; hint text appears in turn 1 only; language hint appears in turn 3 only.
53. `_TURN1_BASE`, `_TURN1_HINT`, `_TURN2_ACKNOWLEDGMENT`, `_TURN3_BASE` module constants are present.
54. Addendum 6 artifacts (`_BASE_PROMPT`, `_HINT_WRAPPER`, `_build_qwen_prompt`) are removed.
55. `_transcribe_chunk` consumes the new conversation list directly.
56. `test_qwen_conversation_construction` passes; old `test_qwen_prompt_construction` removed.
57. All other existing tests continue to pass.
58. Whisper engine, stitcher, pipeline, API, frontend all unchanged.
59. Re-running the Catrina diagnostic from Phase 1 shows per-chunk char counts substantially improved over the bimodal 26/196-char Phase 1 baseline.
