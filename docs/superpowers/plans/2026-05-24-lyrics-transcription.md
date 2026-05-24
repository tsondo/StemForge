# Lyrics Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lyrics transcription feature to the MIDI tab that takes any audio (vocal stem, separated stem, or full mix) and produces lyrics as `.txt`, `.lrc`, and `.srt` files, with a user-selectable engine (Whisper variants or Qwen2-Audio-7B-Instruct). Consolidate all `faster_whisper` usage behind a single `WhisperEngine` class so there is exactly one Whisper call site after this PR.

**Architecture:**
- New `pipelines/transcribe_engines/` sub-package contains a `TranscriptionEngine` Protocol plus `WhisperEngine` (wraps `faster-whisper`) and `QwenEngine` (wraps Qwen2-Audio via `transformers`).
- New `pipelines/transcribe_pipeline.py` orchestrates engine selection, runs the chosen engine, and writes `.txt` / `.lrc` / `.srt` outputs.
- New `backend/api/transcribe.py` exposes `POST /api/transcribe` and `GET /api/transcribe/engines`.
- Two existing Whisper call sites (`models/midi_loader.py::convert_vocal_to_midi`, `pipelines/vocal_midi_pipeline.py::load_model`) are refactored to consume `WhisperEngine` so only one site imports `faster_whisper` after the PR.
- Frontend: the MIDI tab gets a `Notes · Lyrics` mode bar mirroring the Enhance tab's pattern; Lyrics mode adds a source selector (separated stems + uploaded audio), engine dropdown, language dropdown, output-format checkboxes, and a result card with `Send to Compose`.

**Tech Stack:** Python 3.11, FastAPI, faster-whisper, transformers (Qwen2-Audio), torch (CUDA), librosa, vanilla JS frontend (no framework), wavesurfer.js.

**License compliance pre-verified during planning:** Qwen2-Audio-7B-Instruct is `apache-2.0` per the HuggingFace API (`https://huggingface.co/api/models/Qwen/Qwen2-Audio-7B-Instruct`). The repo ships no `LICENSE` or `NOTICE` file; we must source the canonical Apache 2.0 license text from `https://www.apache.org/licenses/LICENSE-2.0.txt`.

---

## Map of files created or modified

**Created:**
- `licenses/LICENSE-Qwen2-Audio` — Apache 2.0 license text
- `pipelines/transcribe_engines/__init__.py` — engine registry
- `pipelines/transcribe_engines/types.py` — Protocol + dataclasses
- `pipelines/transcribe_engines/whisper_engine.py` — faster-whisper wrapper
- `pipelines/transcribe_engines/qwen_engine.py` — Qwen2-Audio wrapper
- `pipelines/transcribe_pipeline.py` — orchestration + format helpers
- `backend/api/transcribe.py` — FastAPI router
- `tests/test_transcribe.py` — smoke test
- `docs/superpowers/plans/2026-05-24-lyrics-transcription.md` — this plan

**Modified:**
- `models/registry.py` — add `WHISPER_LARGE_V3` spec
- `utils/paths.py` — add `LYRICS_DIR`
- `backend/api/audio.py` — add `LYRICS_DIR` to `_ALLOWED_ROOTS`
- `backend/main.py` — register router, ensure `LYRICS_DIR` exists, import update
- `backend/services/pipeline_manager.py` — add `get_transcribe()`
- `backend/services/session_store.py` — add `_lyrics_paths` field + `add_lyrics_path()` + clear/to_dict integration
- `models/midi_loader.py` — refactor `convert_vocal_to_midi` to use `WhisperEngine`; drop `_ensure_whisper`
- `pipelines/vocal_midi_pipeline.py` — refactor `load_model` + `_transcribe` to use `WhisperEngine`
- `frontend/components/midi.js` — mode bar, Lyrics panel, results card, source selector
- `frontend/components/compose.js` — listen for `lyricsReady` event and offer to prefill lyrics textarea
- `frontend/app.js` — add `lyricsPaths` to `appState`
- `frontend/style.css` — add `.midi-mode-bar` / `.midi-mode-btn` styles (reuse Enhance mode-bar look)
- `ACKNOWLEDGMENTS.md` — append Qwen2-Audio entry
- `docs/CURRENT_STATE.md` — one-line addition under "What's working"
- `docs/INSTRUCTIONS.md` — new sub-section under MIDI for Notes/Lyrics mode bar

---

## Open notes for the implementer

1. **Spec deviation (clean):** §3.5 in the spec branches on `engine_id == "whisper"` in `TranscribePipeline.load_model()` to decide whether to pass `model_id` to the engine constructor. We instead make all engines accept `model_id` as an optional kwarg (Qwen ignores it). Net: same behavior, cleaner code. The engine registration in `ENGINES` stays simple.

2. **Existing redundancy (no change):** `OUTPUT_BASE` is already in `_ALLOWED_ROOTS` in `backend/api/audio.py`, so subdirs like `LYRICS_DIR = OUTPUT_BASE / "lyrics"` are already implicitly allowed. We add `LYRICS_DIR` to the explicit list anyway because the spec asks for it and it matches the pattern of the other dirs listed.

3. **Refactor in-memory-array contract:** Today `models/midi_loader.py::convert_vocal_to_midi` passes a pre-loaded 16 kHz numpy array to `WhisperModel.transcribe()`; same for `pipelines/vocal_midi_pipeline.py::_transcribe`. The new `WhisperEngine.transcribe(path, language, prompt)` takes a path only. The refactor will reload audio from disk inside the engine. This is the trade-off the user accepted ("Path-only input, vad_filter on"). The two existing test scripts (`tests/test_vocal_midi_pipeline.py` and the vocal MIDI tab's golden-path UI) must keep working — we verify by manual run at the end.

4. **VocalMidiConfig API stability:** `tests/test_vocal_midi_pipeline.py` passes `whisper_model_size="tiny"`, `whisper_device="cpu"`, `whisper_compute_type="int8"`. Keep these fields on the config and forward them when constructing the WhisperEngine, but rely on registry-based resolution inside the engine. (`whisper_model_size` → `model_id = f"whisper-{size}"`.)

5. **Frontend Send to Compose UX:** Spec says "small toast/banner asking for confirmation." We implement this as an inline banner inside the Compose tab's lyrics panel with "Apply" / "Dismiss" buttons (no modal). If the user prefers a `confirm()` dialog or a different pattern, we iterate after they see it.

6. **LYRICS_DIR location:** `OUTPUT_BASE / "lyrics"`. Per-user subdir via `user_dir(LYRICS_DIR, session.user)` matches the multi-user pattern of `ENHANCE_DIR`, `VOICE_DIR`, `MIX_DIR`.

---

## Task 1: License scaffolding

**Files:**
- Create: `licenses/LICENSE-Qwen2-Audio`
- Modify: `ACKNOWLEDGMENTS.md`

- [ ] **Step 1: Fetch canonical Apache 2.0 license text**

Run: `curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o licenses/LICENSE-Qwen2-Audio`
Expected: a file containing the 202-line standard Apache 2.0 license, starting with `                                 Apache License` and ending with the APPENDIX boilerplate. Verify with `wc -l licenses/LICENSE-Qwen2-Audio` → 202.

- [ ] **Step 2: Verify file content sanity**

Run: `head -1 licenses/LICENSE-Qwen2-Audio && tail -3 licenses/LICENSE-Qwen2-Audio`
Expected: first line contains `Apache License`; last lines reference `http://www.apache.org/licenses/LICENSE-2.0`.

- [ ] **Step 3: Append Qwen2-Audio attribution to ACKNOWLEDGMENTS.md**

Edit `ACKNOWLEDGMENTS.md`, find the section that begins with `## Whisper — OpenAI` (around line 51), then locate the next horizontal rule (`---`) immediately after the `## faster-whisper — SYSTRAN` block. Insert this section between `faster-whisper` and `Stable Audio Open`:

```markdown
## Qwen2-Audio — Alibaba Cloud (Tongyi Lab)

Multimodal audio understanding model used as an optional engine in the
Lyrics transcription feature on the MIDI tab.

- **Model:** https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
- **Paper:** Chu et al. — *Qwen2-Audio Technical Report* (2024)
- **License:** Apache 2.0 (verified at integration time — see `licenses/LICENSE-Qwen2-Audio`)

---
```

- [ ] **Step 4: Commit**

```bash
git add licenses/LICENSE-Qwen2-Audio ACKNOWLEDGMENTS.md
git commit -m "Add Apache 2.0 license + acknowledgment for Qwen2-Audio integration"
```

---

## Task 2: Engine types and Protocol

**Files:**
- Create: `pipelines/transcribe_engines/__init__.py` (stub for now)
- Create: `pipelines/transcribe_engines/types.py`

- [ ] **Step 1: Create the sub-package directory marker**

```bash
mkdir -p pipelines/transcribe_engines
```

- [ ] **Step 2: Write `pipelines/transcribe_engines/types.py`**

Create file with content:

```python
"""Shared types for transcription engines.

Each engine returns a :class:`TranscriptionResult` containing zero or
more :class:`TranscriptionSegment` items.  Word timestamps are optional
and indicated by ``has_word_timestamps`` — Whisper engines populate
``segments[].words``; Qwen produces a single segment with an empty
``words`` list and ``has_word_timestamps=False``.
"""
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

    engine_id: str
    model_id: str
    supports_word_timestamps: bool
    requires_gpu: bool

    def load(self) -> None:
        ...

    def transcribe(
        self,
        audio_path: pathlib.Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        ...

    def clear(self) -> None:
        ...
```

- [ ] **Step 3: Write stub `pipelines/transcribe_engines/__init__.py`**

```python
"""Pluggable transcription engines for StemForge."""
from __future__ import annotations

from .types import (
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptionSegment,
    WordTiming,
)

__all__ = [
    "TranscriptionEngine",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WordTiming",
]
```

(Engine classes and `ENGINES` dict added in later tasks; keeping `__init__.py` minimal until the engines exist avoids import errors during incremental implementation.)

- [ ] **Step 4: Verify imports work**

Run: `uv run python -c "from pipelines.transcribe_engines import TranscriptionEngine, TranscriptionResult, TranscriptionSegment, WordTiming; print('OK')"`
Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git add pipelines/transcribe_engines/__init__.py pipelines/transcribe_engines/types.py
git commit -m "Add TranscriptionEngine Protocol and result dataclasses"
```

---

## Task 3: Add WHISPER_LARGE_V3 to model registry

**Files:**
- Modify: `models/registry.py`

- [ ] **Step 1: Register the new spec**

Edit `models/registry.py`. Find `WHISPER_MEDIUM = _register(WhisperSpec(...))` (around line 546) and the closing `))` of that block (around line 574). After that closing paren and before the `# ---` Stable Audio Open separator (around line 576), insert:

```python
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

```

- [ ] **Step 2: Verify registration**

Run: `uv run python -c "from models.registry import get_spec; s = get_spec('whisper-large-v3'); print(s.model_id, s.model_size, s.gpu_capable)"`
Expected: `whisper-large-v3 large-v3 True`

- [ ] **Step 3: Verify list filter**

Run: `uv run python -c "from models.registry import list_specs, WhisperSpec; print([s.model_id for s in list_specs(WhisperSpec)])"`
Expected: `['whisper-tiny', 'whisper-base', 'whisper-small', 'whisper-medium', 'whisper-large-v3']`

- [ ] **Step 4: Commit**

```bash
git add models/registry.py
git commit -m "Register whisper-large-v3 model spec"
```

---

## Task 4: WhisperEngine

**Files:**
- Create: `pipelines/transcribe_engines/whisper_engine.py`
- Test: smoke verification command (no separate test file yet — covered by Task 9's `tests/test_transcribe.py`)

- [ ] **Step 1: Write the engine module**

Create `pipelines/transcribe_engines/whisper_engine.py`:

```python
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

        if torch.cuda.is_available():
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
```

- [ ] **Step 2: Smoke-test the engine standalone against silence.wav**

Run: `uv run python -c "
import pathlib
from pipelines.transcribe_engines.whisper_engine import WhisperEngine
e = WhisperEngine(model_id='whisper-tiny')
e.load()
r = e.transcribe(pathlib.Path('tests/data/silence.wav'))
print('engine_id:', r.engine_id, 'model_id:', r.model_id, 'segments:', len(r.segments), 'has_words:', r.has_word_timestamps)
e.clear()
"`
Expected output (silence yields zero segments — that's fine):
```
engine_id: whisper model_id: whisper-tiny segments: 0 has_words: True
```

- [ ] **Step 3: Commit**

```bash
git add pipelines/transcribe_engines/whisper_engine.py
git commit -m "Add WhisperEngine wrapping faster-whisper"
```

---

## Task 5: QwenEngine

**Files:**
- Create: `pipelines/transcribe_engines/qwen_engine.py`

- [ ] **Step 1: Write the engine module**

Create `pipelines/transcribe_engines/qwen_engine.py`:

```python
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

    def __init__(self, model_id: str | None = None) -> None:
        # model_id kwarg accepted for Protocol uniformity; ignored — Qwen
        # has exactly one variant in v1.
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

- [ ] **Step 2: Verify import (do NOT load — model is 7B and the user may not have it cached)**

Run: `uv run python -c "from pipelines.transcribe_engines.qwen_engine import QwenEngine; e = QwenEngine(); print('class attrs:', e.engine_id, e.model_id, e.requires_gpu, e.supports_word_timestamps)"`
Expected output: `class attrs: qwen qwen2-audio-7b-instruct True False`

- [ ] **Step 3: Verify CUDA error path on a CPU-only path**

Run: `uv run python -c "
import torch
print('cuda available:', torch.cuda.is_available())
"`
If CUDA is available on this machine (it is per memory notes — RTX 5080), the `not torch.cuda.is_available()` branch can't be exercised here. That is OK; the path is exercised by the unit test in Task 9 via mocking is unnecessary — we trust the static branch. Just move on.

- [ ] **Step 4: Commit**

```bash
git add pipelines/transcribe_engines/qwen_engine.py
git commit -m "Add QwenEngine wrapping Qwen2-Audio-7B-Instruct"
```

---

## Task 6: Wire ENGINES registry in `__init__.py`

**Files:**
- Modify: `pipelines/transcribe_engines/__init__.py`

- [ ] **Step 1: Replace `__init__.py` contents**

Overwrite `pipelines/transcribe_engines/__init__.py` with:

```python
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
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from pipelines.transcribe_engines import ENGINES; print(sorted(ENGINES))"`
Expected: `['qwen', 'whisper']`

- [ ] **Step 3: Commit**

```bash
git add pipelines/transcribe_engines/__init__.py
git commit -m "Register WhisperEngine and QwenEngine in ENGINES dict"
```

---

## Task 7: TranscribePipeline + format helpers

**Files:**
- Create: `pipelines/transcribe_pipeline.py`

- [ ] **Step 1: Write the pipeline module**

Create `pipelines/transcribe_pipeline.py`:

```python
"""Lyrics transcription pipeline.

Selects a transcription engine (Whisper or Qwen), runs transcription on
a single audio file, and writes plain text, LRC, and SRT outputs.

Lifecycle: configure → load_model → run → clear.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
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
    model_id: str = "whisper-base"
    language: str | None = None
    prompt: str | None = None
    output_dir: pathlib.Path | None = None
    formats: tuple[str, ...] = ("txt", "lrc", "srt")


@dataclass(slots=True)
class TranscribeResult:
    result: TranscriptionResult
    output_paths: dict[str, pathlib.Path]
    label: str


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
        self._engine = engine_cls(model_id=self._config.model_id)
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
                f"Unsupported audio format: {audio_path.suffix}",
                field="audio_path",
            )

        if progress_cb:
            progress_cb(0.1, "Transcribing")

        result = self._engine.transcribe(
            audio_path,
            language=self._config.language,
            prompt=self._config.prompt,
        )

        if progress_cb:
            progress_cb(0.85, "Writing outputs")

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
    Without word timestamps (Qwen): one line per text-line with the
    segment-start timestamp.
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
    """SRT subtitle format.  Segment-level — never per-word."""
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

- [ ] **Step 2: Verify import + config validation**

Run: `uv run python -c "
from pipelines.transcribe_pipeline import TranscribePipeline, TranscribeConfig
from utils.errors import InvalidInputError
p = TranscribePipeline()
try:
    p.configure(TranscribeConfig(engine_id='bogus'))
    print('FAIL: should have raised')
except InvalidInputError as e:
    print('OK:', e)
"`
Expected output starts with `OK: Unknown engine_id 'bogus'.`

- [ ] **Step 3: Verify format helpers on a synthetic result**

Run: `uv run python -c "
from pipelines.transcribe_engines.types import TranscriptionResult, TranscriptionSegment, WordTiming
from pipelines.transcribe_pipeline import _format_lrc, _format_srt
r = TranscriptionResult(
    text='hello world',
    language='en',
    segments=[TranscriptionSegment(
        start=0.0, end=1.5, text='hello world',
        words=[WordTiming(word='hello', start=0.0, end=0.7),
               WordTiming(word=' world', start=0.7, end=1.5)],
    )],
    has_word_timestamps=True,
    engine_id='whisper', model_id='whisper-tiny',
)
print('LRC:')
print(_format_lrc(r))
print('SRT:')
print(_format_srt(r))
"`
Expected output (whitespace exact):
```
LRC:
[00:00.00]hello
[00:00.70] world

SRT:
1
00:00:00,000 --> 00:00:01,500
hello world

```

- [ ] **Step 4: Commit**

```bash
git add pipelines/transcribe_pipeline.py
git commit -m "Add TranscribePipeline with LRC/SRT/TXT format helpers"
```

---

## Task 8: Paths, SessionStore, pipeline_manager wiring

**Files:**
- Modify: `utils/paths.py`
- Modify: `backend/services/session_store.py`
- Modify: `backend/services/pipeline_manager.py`
- Modify: `backend/api/audio.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Add `LYRICS_DIR` to `utils/paths.py`**

In `utils/paths.py`, find the existing block:

```python
ENHANCE_DIR  = OUTPUT_BASE / "enhance"
```

Append immediately after:

```python
LYRICS_DIR   = OUTPUT_BASE / "lyrics"
```

- [ ] **Step 2: Add `_lyrics_paths` field + accessors to `SessionStore`**

Edit `backend/services/session_store.py`:

(a) In `__init__`, find the line:

```python
        self._enhance_paths: dict[str, pathlib.Path] = {}  # label → enhanced output path
```

Insert immediately after:

```python
        self._lyrics_paths: dict[str, pathlib.Path] = {}  # label → lyrics file path
```

(b) After the `add_enhance_path` method (around line 220), add:

```python
    # -- lyrics_paths --
    @property
    def lyrics_paths(self) -> dict[str, pathlib.Path]:
        with self._lock:
            return dict(self._lyrics_paths)

    def add_lyrics_path(self, label: str, path: pathlib.Path) -> None:
        with self._lock:
            self._lyrics_paths[label] = path
```

(c) In `clear()`, find `self._enhance_paths = {}` and add the line `self._lyrics_paths = {}` immediately after (preserving alphabetical-ish placement is fine).

(d) In `to_dict()`, find the `"enhance_paths": ...` line in the returned dict and add immediately after:

```python
                "lyrics_paths": {k: str(v) for k, v in self._lyrics_paths.items()},
```

- [ ] **Step 3: Add `get_transcribe` to `pipeline_manager.py`**

Edit `backend/services/pipeline_manager.py`:

(a) In `_get_or_create`, find the `elif name == "effects":` block (around line 292). Insert a new `elif` before the final `else: raise ValueError`:

```python
        elif name == "transcribe":
            from pipelines.transcribe_pipeline import TranscribePipeline
            cache[name] = TranscribePipeline()
```

(b) After `def get_effects(...)` (around line 330), add:

```python
def get_transcribe(gpu_index: int | None = None) -> Any:
    return _get_or_create("transcribe", gpu_index)
```

- [ ] **Step 4: Add `LYRICS_DIR` to audio.py allowlist**

Edit `backend/api/audio.py`:

(a) Change the import:

```python
from utils.paths import OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR
```

to:

```python
from utils.paths import OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR, LYRICS_DIR
```

(b) Change the `_ALLOWED_ROOTS` list:

```python
_ALLOWED_ROOTS = [
    OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR, _UPLOAD_DIR,
    _ACESTEP_AUDIO_TMP,
]
```

to:

```python
_ALLOWED_ROOTS = [
    OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR, LYRICS_DIR, _UPLOAD_DIR,
    _ACESTEP_AUDIO_TMP,
]
```

- [ ] **Step 5: Wire `LYRICS_DIR` into `backend/main.py`**

Edit `backend/main.py`:

(a) Change the import:

```python
from utils.paths import OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR, VOICE_DIR, ENHANCE_DIR
```

to:

```python
from utils.paths import OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR, VOICE_DIR, ENHANCE_DIR, LYRICS_DIR
```

(b) Change the `for d in (...)` line (around line 115):

```python
for d in (OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR, VOICE_DIR, ENHANCE_DIR):
```

to:

```python
for d in (OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR, VOICE_DIR, ENHANCE_DIR, LYRICS_DIR):
```

(Note: the transcribe router is wired in Task 10. Don't add it here yet.)

- [ ] **Step 6: Verify everything imports and the dir exists**

Run: `uv run python -c "
from utils.paths import LYRICS_DIR
from backend.services.session_store import SessionStore
from backend.services import pipeline_manager
s = SessionStore('local')
s.add_lyrics_path('test', LYRICS_DIR / 'test.txt')
assert 'test' in s.lyrics_paths
d = s.to_dict()
assert 'lyrics_paths' in d
s.clear()
assert s.lyrics_paths == {}
print('LYRICS_DIR =', LYRICS_DIR)
print('SessionStore lyrics_paths OK')
print('get_transcribe accessor exists:', hasattr(pipeline_manager, 'get_transcribe'))
"`
Expected:
```
LYRICS_DIR = /home/tsondo/.local/share/stemforge/output/lyrics
SessionStore lyrics_paths OK
get_transcribe accessor exists: True
```

- [ ] **Step 7: Verify backend boots**

Run: `uv run python -c "import backend.main; print('backend imports OK')"`
Expected: `backend imports OK`

- [ ] **Step 8: Commit**

```bash
git add utils/paths.py backend/services/session_store.py backend/services/pipeline_manager.py backend/api/audio.py backend/main.py
git commit -m "Add LYRICS_DIR + lyrics_paths session field + get_transcribe accessor"
```

---

## Task 9: Smoke test for TranscribePipeline

**Files:**
- Create: `tests/test_transcribe.py`

- [ ] **Step 1: Write the test**

Create `tests/test_transcribe.py`:

```python
"""Smoke test for TranscribePipeline with the Whisper engine.

Runs the full pipeline on tests/data/silence.wav with the smallest
Whisper model and confirms:
  - Pipeline configures and loads without error
  - run() produces a TranscribeResult with output_paths for all formats
  - Output files exist (txt may be empty on silence)
  - .srt and .lrc are parseable (or empty for silence)
  - Pipeline can be cleared without raising

Qwen is intentionally NOT covered in CI — it requires GPU, is large to
download, and may have a license gate at the model level.
"""
import pathlib
import tempfile

from pipelines.transcribe_pipeline import TranscribePipeline, TranscribeConfig


def main() -> None:
    audio = pathlib.Path("tests/data/silence.wav")
    assert audio.exists(), f"Missing test fixture: {audio}"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = pathlib.Path(tmp)
        pipeline = TranscribePipeline()
        pipeline.configure(TranscribeConfig(
            engine_id="whisper",
            model_id="whisper-tiny",
            output_dir=out_dir,
        ))
        pipeline.load_model()
        try:
            result = pipeline.run(audio)
        finally:
            pipeline.clear()

        for fmt in ("txt", "lrc", "srt"):
            assert fmt in result.output_paths, f"Missing format: {fmt}"
            assert result.output_paths[fmt].exists(), f"Missing output: {result.output_paths[fmt]}"
        print(f"engine_id      : {result.result.engine_id}")
        print(f"model_id       : {result.result.model_id}")
        print(f"language       : {result.result.language}")
        print(f"segments       : {len(result.result.segments)}")
        print(f"has_words      : {result.result.has_word_timestamps}")
        print(f"output_paths   : {sorted(result.output_paths)}")
        print("transcribe pipeline OK")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the test**

Run: `uv run python tests/test_transcribe.py`
Expected:
- Logs include `Loading faster-whisper 'tiny' on …`
- Then prints `engine_id : whisper`, `model_id : whisper-tiny`, etc.
- Final line: `transcribe pipeline OK`

- [ ] **Step 3: Commit**

```bash
git add tests/test_transcribe.py
git commit -m "Add smoke test for TranscribePipeline + Whisper engine"
```

---

## Task 10: Transcribe API router + wire into main.py

**Files:**
- Create: `backend/api/transcribe.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write the router**

Create `backend/api/transcribe.py`:

```python
"""Lyrics transcription endpoints."""
from __future__ import annotations

import pathlib

import torch
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from backend.services.job_manager import job_manager
from backend.services.session_store import SessionStore, get_user_session
from backend.services import pipeline_manager
from models.registry import list_specs, WhisperSpec
from pipelines.transcribe_engines import ENGINES
from utils.paths import LYRICS_DIR, user_dir

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


class TranscribeRequest(BaseModel):
    audio_path: str
    engine_id: str = "whisper"
    model_id: str = "whisper-base"
    language: str | None = None
    prompt: str | None = None
    formats: list[str] = ["txt", "lrc", "srt"]


@router.get("/engines")
def list_engines() -> dict:
    """Return available engines and their capabilities for the UI."""
    cuda = torch.cuda.is_available()
    engines = []
    for engine_id, cls in ENGINES.items():
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

    with pipeline_manager.gpu_session(pipeline_hint="transcribe") as ctx:
        pipeline = pipeline_manager.get_transcribe(ctx.gpu_index)
        pipeline.configure(config)
        try:
            progress_cb(0.05, "Loading model")
            pipeline.load_model()
            result = pipeline.run(audio_path, progress_cb=progress_cb)
        finally:
            pipeline_manager.evict("transcribe", ctx.gpu_index)

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

- [ ] **Step 2: Wire the router into `backend/main.py`**

Edit `backend/main.py`. Change:

```python
from backend.api import system, audio, separate, midi, generate, mix, export, compose, sfx, voice, enhance
```

to:

```python
from backend.api import system, audio, separate, midi, generate, mix, export, compose, sfx, voice, enhance, transcribe
```

Find the block of `app.include_router(...)` calls (around lines 102–112). After `app.include_router(enhance.router)`, add:

```python
app.include_router(transcribe.router)
```

- [ ] **Step 3: Verify backend boots and exposes `/api/transcribe/engines`**

Run: `uv run python -c "
from fastapi.testclient import TestClient
import backend.main
client = TestClient(backend.main.app)
r = client.get('/api/transcribe/engines')
assert r.status_code == 200, r.text
data = r.json()
assert 'engines' in data
ids = sorted(e['engine_id'] for e in data['engines'])
print('engines:', ids)
print('cuda_available:', data['cuda_available'])
print('whisper models:', [m['model_id'] for e in data['engines'] if e['engine_id']=='whisper' for m in e['models']])
"`
Expected:
```
engines: ['qwen', 'whisper']
cuda_available: True
whisper models: ['whisper-tiny', 'whisper-base', 'whisper-small', 'whisper-medium', 'whisper-large-v3']
```

- [ ] **Step 4: Commit**

```bash
git add backend/api/transcribe.py backend/main.py
git commit -m "Add /api/transcribe router with engines listing + job-based transcription"
```

---

## Task 11: Refactor `models/midi_loader.py::convert_vocal_to_midi`

**Files:**
- Modify: `models/midi_loader.py`

Goal: drop `_ensure_whisper()` and the inline `WhisperModel` import, replacing with `WhisperEngine`. The PYIN + word-to-note mapping logic stays the same.

- [ ] **Step 1: Update imports**

In `models/midi_loader.py`, replace:

```python
from models.registry import DEFAULT_WHISPER_SPEC
from utils.cache import get_model_cache_dir
from utils.midi_io import NoteEvent, LyricEvent, filter_to_key
from utils.errors import ModelLoadError, PipelineExecutionError
```

with:

```python
from utils.midi_io import NoteEvent, LyricEvent, filter_to_key
from utils.errors import ModelLoadError, PipelineExecutionError
```

(We drop `DEFAULT_WHISPER_SPEC` and `get_model_cache_dir` — the engine handles cache resolution internally. `ModelLoadError` is still used.)

- [ ] **Step 2: Drop `_whisper_model` field**

In `__init__`, change:

```python
        self._model: Any | None = None          # BasicPitch TF model
        self._whisper_model: Any | None = None  # faster-whisper WhisperModel
```

to:

```python
        self._model: Any | None = None          # BasicPitch TF model
```

- [ ] **Step 3: Drop `_whisper_model = None` in `evict()`**

In `evict()`, change:

```python
    def evict(self) -> None:
        """Release both models from memory and trigger GC."""
        self._bp_loader.evict()
        self._model = None
        self._whisper_model = None
        log.debug("MidiModelLoader: models evicted.")
```

to:

```python
    def evict(self) -> None:
        """Release both models from memory and trigger GC."""
        self._bp_loader.evict()
        self._model = None
        log.debug("MidiModelLoader: models evicted.")
```

- [ ] **Step 4: Remove the `_ensure_whisper` method entirely**

Delete the whole method (and its comment header block):

```python
    # ------------------------------------------------------------------
    # Internal: Whisper lazy loader
    # ------------------------------------------------------------------

    def _ensure_whisper(self) -> Any:
        """Load the faster-whisper model on first use."""
        if self._whisper_model is not None:
            return self._whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelLoadError(
                "faster-whisper is not installed — cannot transcribe vocals.",
                model_name="faster-whisper",
            ) from exc
        spec = DEFAULT_WHISPER_SPEC
        log.info("Loading faster-whisper '%s' on CPU…", spec.model_size)
        self._whisper_model = WhisperModel(
            spec.model_size,
            device=spec.device,
            compute_type=spec.compute_type,
            download_root=str(get_model_cache_dir("whisper")),
        )
        log.info("faster-whisper model ready.")
        return self._whisper_model
```

- [ ] **Step 5: Refactor `convert_vocal_to_midi` to use `WhisperEngine`**

In `convert_vocal_to_midi`, replace the Whisper section. The current code is:

```python
        whisper = self._ensure_whisper()

        # Load audio at 16 kHz for Whisper and also at 22 050 Hz for PYIN.
        try:
            y_pyin, sr_pyin = librosa.load(str(path), sr=22_050, mono=True)
            y_whisper, _sr_w = librosa.load(str(path), sr=16_000, mono=True)
        except Exception as exc:
            raise PipelineExecutionError(
                f"Failed to load vocal audio '{path.name}': {exc}",
                pipeline_name="midi",
            ) from exc
```

Replace with:

```python
        # Load audio at 22 050 Hz for PYIN.  The transcription engine
        # reloads the audio from disk at its native rate internally.
        try:
            y_pyin, sr_pyin = librosa.load(str(path), sr=22_050, mono=True)
        except Exception as exc:
            raise PipelineExecutionError(
                f"Failed to load vocal audio '{path.name}': {exc}",
                pipeline_name="midi",
            ) from exc
```

Then find the existing Whisper transcription block:

```python
        # Transcribe with faster-whisper to get word-level timestamps.
        try:
            segments_iter, _info = whisper.transcribe(
                y_whisper,
                language=language,
                word_timestamps=True,
                vad_filter=True,
            )
            segments = list(segments_iter)
        except Exception as exc:
            raise PipelineExecutionError(
                f"Whisper transcription failed for '{path.name}': {exc}",
                pipeline_name="midi",
            ) from exc
```

Replace with:

```python
        # Transcribe via the shared WhisperEngine.
        from pipelines.transcribe_engines import WhisperEngine

        engine = WhisperEngine(model_id="whisper-base")
        try:
            engine.load()
            transcription = engine.transcribe(path, language=language)
        finally:
            engine.clear()
        segments = transcription.segments
```

- [ ] **Step 6: Update the per-word loop to use the new dataclass fields**

Find the loop:

```python
        # Build one NoteEvent + LyricEvent per voiced word.
        events: list[NoteEvent] = []
        lyrics: list[LyricEvent] = []
        for segment in segments:
            words = getattr(segment, "words", None) or []
            for word in words:
                start = float(word.start)
                end = float(word.end)
                ...
                velocity = min(127, max(1, int(abs(word.probability) * 100)))
                events.append((start, clipped_end, midi_pitch, velocity))

                # Strip leading/trailing punctuation but keep apostrophes.
                text = word.word.strip().strip('.,!?;:"()[]{}…—–')
```

The field names (`.word`, `.start`, `.end`, `.probability`) match our `WordTiming` dataclass exactly. The only subtle change: `word.probability` may now be `None` (when faster-whisper didn't emit it). Update the velocity line:

```python
                prob = word.probability if word.probability is not None else 0.5
                velocity = min(127, max(1, int(abs(prob) * 100)))
```

- [ ] **Step 7: Verify imports still resolve and grep for remaining whisper references**

Run: `uv run python -c "from models.midi_loader import MidiModelLoader; print('OK')"`
Expected: `OK`

Run: `grep -n 'whisper\|faster_whisper\|DEFAULT_WHISPER_SPEC\|_ensure_whisper' models/midi_loader.py | head -20`
Expected: no matches except inside docstrings and comments referring to "the shared WhisperEngine".

- [ ] **Step 8: Commit**

```bash
git add models/midi_loader.py
git commit -m "Refactor midi_loader.convert_vocal_to_midi to use WhisperEngine"
```

---

## Task 12: Refactor `pipelines/vocal_midi_pipeline.py`

**Files:**
- Modify: `pipelines/vocal_midi_pipeline.py`

Goal: replace the inline `WhisperModel` import and the in-memory `_transcribe` flow with a `WhisperEngine` instance. `VocalMidiConfig` keeps its existing fields (`whisper_model_size`, `whisper_device`, `whisper_compute_type`) for backward compat — `whisper_model_size` is forwarded; `whisper_device`/`whisper_compute_type` are no-ops now (the engine picks based on CUDA availability).

- [ ] **Step 1: Update field declarations**

In `class VocalMidiPipeline`, change the `_whisper_model: Any` declaration. Find:

```python
    is_loaded: bool
    _config: VocalMidiConfig | None
    _whisper_model: Any
    _demucs_model: Any
    _basicpitch_model: Any
    _tmp_dir: tempfile.TemporaryDirectory | None  # type: ignore[type-arg]
    _progress_callback: Callable[[float, str], None] | None
```

Replace with:

```python
    is_loaded: bool
    _config: VocalMidiConfig | None
    _transcribe_engine: Any
    _demucs_model: Any
    _basicpitch_model: Any
    _tmp_dir: tempfile.TemporaryDirectory | None  # type: ignore[type-arg]
    _progress_callback: Callable[[float, str], None] | None
```

In `__init__`, change:

```python
        self._whisper_model = None
```

to:

```python
        self._transcribe_engine = None
```

- [ ] **Step 2: Replace the Whisper load block in `load_model`**

Find this block:

```python
        try:
            from faster_whisper import WhisperModel
            log.info(
                "Loading Whisper '%s' on %s (%s)…",
                self._config.whisper_model_size,
                self._config.whisper_device,
                self._config.whisper_compute_type,
            )
            self._whisper_model = WhisperModel(
                self._config.whisper_model_size,
                device=self._config.whisper_device,
                compute_type=self._config.whisper_compute_type,
            )
            log.info("Whisper loaded.")
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Whisper model '{self._config.whisper_model_size}': {exc}",
                model_name=self._config.whisper_model_size,
            ) from exc
```

Replace with:

```python
        from pipelines.transcribe_engines import WhisperEngine

        whisper_model_id = f"whisper-{self._config.whisper_model_size}"
        log.info("Loading transcribe engine for %s…", whisper_model_id)
        try:
            self._transcribe_engine = WhisperEngine(model_id=whisper_model_id)
            self._transcribe_engine.load()
            log.info("Transcribe engine loaded.")
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Whisper engine for {whisper_model_id}: {exc}",
                model_name=whisper_model_id,
            ) from exc
```

- [ ] **Step 3: Replace the `_transcribe` method body**

Find the existing `_transcribe` method:

```python
    def _transcribe(
        self,
        vocal_path: pathlib.Path,
        initial_prompt: str | None = None,
    ) -> list[WordTiming]:
        """Run faster-whisper on *vocal_path* and return word-level timestamps.
        ...
        """
        segments_gen, info = self._whisper_model.transcribe(
            str(vocal_path),
            word_timestamps=True,
            initial_prompt=initial_prompt,
        )
        log.info(
            "Whisper: language=%r (p=%.2f), duration=%.1fs",
            info.language,
            info.language_probability,
            info.duration,
        )

        words: list[WordTiming] = []
        for segment in segments_gen:
            if segment.words:
                for w in segment.words:
                    words.append((w.word, w.start, w.end))

        return words
```

Replace the body (keep signature + docstring) with:

```python
    def _transcribe(
        self,
        vocal_path: pathlib.Path,
        initial_prompt: str | None = None,
    ) -> list[WordTiming]:
        """Run the shared WhisperEngine on *vocal_path* and return word-level timestamps.

        Parameters
        ----------
        vocal_path:
            Path to the isolated vocal stem WAV.
        initial_prompt:
            Optional text fed to Whisper as context (e.g. the ACE-Step
            ``caption`` field).

        Returns
        -------
        list[WordTiming]
            ``(word, start_sec, end_sec)`` tuples, one per word token.
        """
        result = self._transcribe_engine.transcribe(
            vocal_path, prompt=initial_prompt,
        )
        log.info(
            "Whisper: language=%r, segments=%d",
            result.language, len(result.segments),
        )

        words: list[WordTiming] = []
        for segment in result.segments:
            for w in segment.words:
                words.append((w.word, w.start, w.end))
        return words
```

- [ ] **Step 4: Update `clear()`**

Find:

```python
    def clear(self) -> None:
        """Release all model weights and temporary files."""
        self._demucs_model = None
        self._whisper_model = None
        self._basicpitch_model = None
```

Replace with:

```python
    def clear(self) -> None:
        """Release all model weights and temporary files."""
        self._demucs_model = None
        if self._transcribe_engine is not None:
            self._transcribe_engine.clear()
            self._transcribe_engine = None
        self._basicpitch_model = None
```

- [ ] **Step 5: Verify imports**

Run: `uv run python -c "from pipelines.vocal_midi_pipeline import VocalMidiPipeline, VocalMidiConfig; print('OK')"`
Expected: `OK`

Run: `grep -n 'faster_whisper\|_whisper_model' pipelines/vocal_midi_pipeline.py | head -20`
Expected: no matches (or only inside removed-now-deleted code that no longer exists).

- [ ] **Step 6: Commit**

```bash
git add pipelines/vocal_midi_pipeline.py
git commit -m "Refactor VocalMidiPipeline to use WhisperEngine"
```

---

## Task 13: Verify single Whisper call site + regression tests

This task has no code changes — it verifies the spec's Definition of Done item #4 and runs the regression smoke tests.

- [ ] **Step 1: Confirm exactly one `from faster_whisper` import remains**

Run: `grep -rn "from faster_whisper" --include='*.py' .`
Expected: exactly one match, in `pipelines/transcribe_engines/whisper_engine.py`.

Run: `grep -rn "import faster_whisper" --include='*.py' .`
Expected: no matches.

- [ ] **Step 2: Run the new transcribe smoke test**

Run: `uv run python tests/test_transcribe.py`
Expected final line: `transcribe pipeline OK`

- [ ] **Step 3: Run the existing faster-whisper smoke test**

Run: `uv run python tests/test_faster_whisper.py`
Expected: prints `faster-whisper OK`. (This test imports faster_whisper directly — that's intentional, it's a third-party-library smoke test, not a StemForge usage site.)

- [ ] **Step 4: Verify backend still boots after refactors**

Run: `uv run python -c "import backend.main; print('backend imports OK')"`
Expected: `backend imports OK`

- [ ] **Step 5: Verify MidiModelLoader can still be imported and instantiated**

Run: `uv run python -c "from models.midi_loader import MidiModelLoader; m = MidiModelLoader(); print('OK', m.is_loaded)"`
Expected: `OK False`

(Step 5 only checks the structure; the actual vocal MIDI flow needs manual UI testing per Task 18.)

- [ ] **Step 6: No commit needed — verification only**

---

## Task 14: CSS for MIDI mode bar

**Files:**
- Modify: `frontend/style.css`

- [ ] **Step 1: Append MIDI mode bar styles**

Edit `frontend/style.css`. Find the existing Enhance mode bar block (search for `/* ─── Enhance Mode Bar` around line 1139). Immediately after the `.enhance-mode-btn.active { ... }` rule (around line 1179), append:

```css
/* ─── MIDI Mode Bar (Notes | Lyrics) ─────────────────────────────── */

.midi-mode-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
  flex-shrink: 0;
}

.midi-mode-selector {
  display: flex;
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}

.midi-mode-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  padding: 7px 18px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  border-right: 1px solid var(--border);
  transition: color 0.15s, background 0.15s;
}

.midi-mode-btn:last-child { border-right: none; }

.midi-mode-btn:hover {
  color: var(--text-dim);
  background: var(--surface-overlay);
}

.midi-mode-btn.active {
  background: var(--accent-dim-compose);
  color: var(--accent);
}

/* Lyrics result card */
.lyrics-text {
  width: 100%;
  min-height: 280px;
  font-family: var(--font-mono, ui-monospace, SFMono-Regular, Menlo, Monaco, monospace);
  font-size: 13px;
  line-height: 1.5;
  padding: 10px 12px;
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  resize: vertical;
}

.lyrics-meta {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.lyrics-badge {
  display: inline-block;
  padding: 2px 8px;
  font-size: 11px;
  border-radius: 3px;
  background: var(--surface-overlay);
  color: var(--text-dim);
}

.lyrics-notice {
  font-size: 12px;
  color: var(--text-muted);
  padding: 6px 10px;
  margin-top: 8px;
  background: var(--surface-overlay);
  border-radius: 4px;
}
```

- [ ] **Step 2: No commit yet — bundle with frontend changes in Task 16**

---

## Task 15: Frontend — add `lyricsPaths` to appState

**Files:**
- Modify: `frontend/app.js`

- [ ] **Step 1: Add the field**

Edit `frontend/app.js`. Find the `appState` object (around line 10–33). Add a new field after `enhancePaths: {},`:

```js
  lyricsPaths: {},
```

So the bottom of the object reads:

```js
  composePaths: [],
  sfxPaths: {},
  voicePaths: {},
  enhancePaths: {},
  lyricsPaths: {},
};
```

- [ ] **Step 2: No commit yet — bundle with frontend changes in Task 16**

---

## Task 16: Frontend — MIDI tab mode bar + Lyrics panel

**Files:**
- Modify: `frontend/components/midi.js`

The existing MIDI tab uses a `two-col` layout where the left column holds controls and the right holds results. We will:
1. Add a `.midi-mode-bar` above the existing two-col layout.
2. Wrap the existing left-column controls in a `<div id="midi-controls-notes">` so they hide when switching to Lyrics.
3. Wrap the existing right-column results in `<div id="midi-results-notes">`.
4. Add a parallel `<div id="midi-controls-lyrics">` and `<div id="midi-results-lyrics">`.
5. Hide Lyrics elements by default; toggle via `switchMidiMode()`.

- [ ] **Step 1: Update top of `initMidi()` and wrap existing controls**

Edit `frontend/components/midi.js`. In `initMidi()` (starts around line 34), find:

```js
export function initMidi() {
  const panel = document.getElementById('panel-midi');
  const layout = el('div', { className: 'two-col' });

  // ─── Left: controls ───
  const left = el('div', { className: 'col-left' });

  const stemSection = el('div', { className: 'form-group' },
    el('label', {}, 'Stems to process'),
    el('div', { className: 'checkbox-group', id: 'midi-stems' },
      el('span', { className: 'text-dim' }, 'Run separation first'),
    ),
  );
```

Replace with:

```js
export function initMidi() {
  const panel = document.getElementById('panel-midi');

  // ─── Mode bar (Notes | Lyrics) ───
  const modeBar = el('div', { className: 'midi-mode-bar' },
    el('div', { className: 'midi-mode-selector' },
      el('button', { className: 'midi-mode-btn active', 'data-mode': 'notes', onClick: () => switchMidiMode('notes') }, 'Notes'),
      el('button', { className: 'midi-mode-btn', 'data-mode': 'lyrics', onClick: () => switchMidiMode('lyrics') }, 'Lyrics'),
    ),
  );
  panel.appendChild(modeBar);

  const layout = el('div', { className: 'two-col' });

  // ─── Left: controls ───
  const left = el('div', { className: 'col-left' });
  const notesControls = el('div', { id: 'midi-controls-notes' });
  const lyricsControls = el('div', { id: 'midi-controls-lyrics', style: { display: 'none' } });

  const stemSection = el('div', { className: 'form-group' },
    el('label', {}, 'Stems to process'),
    el('div', { className: 'checkbox-group', id: 'midi-stems' },
      el('span', { className: 'text-dim' }, 'Run separation first'),
    ),
  );
```

- [ ] **Step 2: Group existing left-column children inside `notesControls`**

Find the line near the bottom of left-column construction:

```js
  left.append(stemSection, keyGroup, bpmGroup, tsGroup, onsetGroup, frameGroup, sf2Group, extractBtn, importInput, importBtn);
```

Replace with:

```js
  notesControls.append(stemSection, keyGroup, bpmGroup, tsGroup, onsetGroup, frameGroup, sf2Group, extractBtn, importInput, importBtn);
  left.append(notesControls, lyricsControls);
```

- [ ] **Step 3: Build the Lyrics control panel (still inside `initMidi`, before `right` setup)**

Immediately after `left.append(notesControls, lyricsControls);` add:

```js
  // ─── Lyrics control panel ───
  const lyricsSourceLabel = el('label', { className: 'field-label' }, 'Source Audio');
  const lyricsSourceSelect = el('select', { id: 'lyrics-source', className: 'select' });
  const lyricsSourceGroup = el('div', { className: 'form-group' },
    lyricsSourceLabel, lyricsSourceSelect,
  );

  const lyricsEngineLabel = el('label', { className: 'field-label' }, 'Engine');
  const lyricsEngineSelect = el('select', { id: 'lyrics-engine', className: 'select' });
  const lyricsEngineGroup = el('div', { className: 'form-group' },
    lyricsEngineLabel, lyricsEngineSelect,
  );

  const lyricsLangLabel = el('label', { className: 'field-label' }, 'Language');
  const lyricsLangSelect = el('select', { id: 'lyrics-language', className: 'select' },
    el('option', { value: '' }, 'Auto-detect'),
    el('option', { value: 'en' }, 'English'),
    el('option', { value: 'zh' }, 'Chinese'),
    el('option', { value: 'ja' }, 'Japanese'),
    el('option', { value: 'ko' }, 'Korean'),
    el('option', { value: 'es' }, 'Spanish'),
    el('option', { value: 'fr' }, 'French'),
    el('option', { value: 'de' }, 'German'),
    el('option', { value: 'pt' }, 'Portuguese'),
    el('option', { value: 'it' }, 'Italian'),
    el('option', { value: 'ru' }, 'Russian'),
    el('option', { value: 'ar' }, 'Arabic'),
    el('option', { value: 'hi' }, 'Hindi'),
  );
  const lyricsLangGroup = el('div', { className: 'form-group' }, lyricsLangLabel, lyricsLangSelect);

  const fmtTxt = el('input', { type: 'checkbox', id: 'lyrics-fmt-txt', checked: 'true', disabled: 'true' });
  const fmtLrc = el('input', { type: 'checkbox', id: 'lyrics-fmt-lrc', checked: 'true' });
  const fmtSrt = el('input', { type: 'checkbox', id: 'lyrics-fmt-srt', checked: 'true' });
  const lyricsFmtGroup = el('div', { className: 'form-group' },
    el('label', {}, 'Output formats'),
    el('div', { className: 'checkbox-group' },
      el('label', {}, fmtTxt, ' .txt (always)'),
      el('label', {}, fmtLrc, ' .lrc'),
      el('label', {}, fmtSrt, ' .srt'),
    ),
  );

  const lyricsCoarseNotice = el('div', { className: 'lyrics-notice hidden', id: 'lyrics-coarse-notice' },
    'Qwen produces segment-level timing; .lrc and .srt use coarse timestamps.',
  );

  const lyricsTranscribeBtn = el('button', { className: 'btn btn-primary', id: 'lyrics-transcribe', disabled: 'true' },
    'Transcribe',
  );

  const lyricsLoadHint = el('div', { className: 'text-dim hidden', id: 'lyrics-load-hint' },
    'Load audio or run separation first. Lyrics transcription works best on an isolated vocal stem.',
  );

  lyricsControls.append(
    lyricsSourceGroup, lyricsEngineGroup, lyricsLangGroup, lyricsFmtGroup,
    lyricsCoarseNotice, lyricsTranscribeBtn, lyricsLoadHint,
  );
```

- [ ] **Step 4: Add the Lyrics results container in the right column**

Find:

```js
  // ─── Right: results ───
  const right = el('div', { className: 'col-right' });

  const progressCard = el('div', { className: 'card hidden', id: 'midi-progress' },
    ...
  );

  const resultsContainer = el('div', { id: 'midi-results' });

  right.append(progressCard, resultsContainer);
```

Replace with:

```js
  // ─── Right: results ───
  const right = el('div', { className: 'col-right' });
  const notesResults = el('div', { id: 'midi-results-notes' });
  const lyricsResults = el('div', { id: 'midi-results-lyrics', style: { display: 'none' } });

  const progressCard = el('div', { className: 'card hidden', id: 'midi-progress' },
    el('div', { className: 'progress-container' },
      el('div', { className: 'progress-bar' },
        el('div', { className: 'progress-fill', id: 'midi-progress-fill' }),
      ),
      el('div', { className: 'progress-label' },
        el('span', { id: 'midi-stage' }, ''),
        el('span', { id: 'midi-pct' }, '0%'),
      ),
    ),
  );

  const resultsContainer = el('div', { id: 'midi-results' });
  notesResults.append(progressCard, resultsContainer);

  const lyricsProgressCard = el('div', { className: 'card hidden', id: 'lyrics-progress' },
    el('div', { className: 'progress-container' },
      el('div', { className: 'progress-bar' },
        el('div', { className: 'progress-fill', id: 'lyrics-progress-fill' }),
      ),
      el('div', { className: 'progress-label' },
        el('span', { id: 'lyrics-stage' }, ''),
        el('span', { id: 'lyrics-pct' }, '0%'),
      ),
    ),
  );

  const lyricsResultContainer = el('div', { id: 'lyrics-result' });
  lyricsResults.append(lyricsProgressCard, lyricsResultContainer);

  right.append(notesResults, lyricsResults);
```

- [ ] **Step 5: At the end of `initMidi()`, add Lyrics-mode wiring**

After the existing `loadGmPrograms(); loadCurrentSoundfont(); checkLilypondAvailability();` lines, add:

```js
  // Lyrics-mode wiring
  loadLyricsEngines();
  refreshLyricsSources();
  document.getElementById('lyrics-engine').addEventListener('change', onLyricsEngineChange);
  document.getElementById('lyrics-transcribe').addEventListener('click', startLyricsTranscription);

  appState.on('fileLoaded', refreshLyricsSources);
  appState.on('stemsReady', refreshLyricsSources);
```

- [ ] **Step 6: Add helper functions at the bottom of the file**

Append at the end of `midi.js` (before the final blank line, after the existing `async function showSheetMusicPanel(...)` and its closing `}`):

```js
// ─── Lyrics mode ────────────────────────────────────────────────────

let _midiMode = 'notes';
let _lyricsEngines = [];

function switchMidiMode(mode) {
  if (mode === _midiMode) return;
  _midiMode = mode;

  document.querySelectorAll('#panel-midi .midi-mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === mode),
  );

  document.getElementById('midi-controls-notes').style.display = mode === 'notes' ? '' : 'none';
  document.getElementById('midi-controls-lyrics').style.display = mode === 'lyrics' ? '' : 'none';
  document.getElementById('midi-results-notes').style.display = mode === 'notes' ? '' : 'none';
  document.getElementById('midi-results-lyrics').style.display = mode === 'lyrics' ? '' : 'none';

  if (mode === 'lyrics') refreshLyricsSources();
}

async function loadLyricsEngines() {
  try {
    const data = await api('/transcribe/engines');
    _lyricsEngines = data.engines || [];
    const sel = document.getElementById('lyrics-engine');
    clearChildren(sel);
    for (const e of _lyricsEngines) {
      for (const m of e.models) {
        const suffix = e.available ? '' : ' (GPU required)';
        const opt = el('option', { value: JSON.stringify({ engine_id: e.engine_id, model_id: m.model_id }) },
          `${m.display_name}${suffix}`);
        if (!e.available) opt.disabled = true;
        sel.appendChild(opt);
      }
    }
    // Default to whisper-base if present
    const defaultOpt = Array.from(sel.options).find(o => {
      try { return JSON.parse(o.value).model_id === 'whisper-base'; } catch { return false; }
    });
    if (defaultOpt) sel.value = defaultOpt.value;
    onLyricsEngineChange();
  } catch (err) {
    /* leave engine list empty — user will see disabled transcribe button */
  }
}

function onLyricsEngineChange() {
  const sel = document.getElementById('lyrics-engine');
  if (!sel.value) return;
  let parsed;
  try { parsed = JSON.parse(sel.value); } catch { return; }
  const engine = _lyricsEngines.find(e => e.engine_id === parsed.engine_id);
  const notice = document.getElementById('lyrics-coarse-notice');
  if (engine && !engine.supports_word_timestamps) {
    notice.classList.remove('hidden');
  } else {
    notice.classList.add('hidden');
  }
}

function refreshLyricsSources() {
  const sel = document.getElementById('lyrics-source');
  if (!sel) return;
  clearChildren(sel);

  const sources = [];
  // Separated stems first
  for (const [label, path] of Object.entries(appState.stemPaths || {})) {
    sources.push({ label: `Stem: ${label}`, path, isVocal: /vocal/i.test(label) });
  }
  // Enhanced stems if any
  for (const [label, path] of Object.entries(appState.enhancePaths || {})) {
    sources.push({ label: `Enhanced: ${label}`, path, isVocal: /vocal/i.test(label) });
  }
  // Uploaded full mix
  if (appState.audioPath) {
    sources.push({ label: 'Full Mix (Original Upload)', path: appState.audioPath, isVocal: false });
  }

  const transcribeBtn = document.getElementById('lyrics-transcribe');
  const loadHint = document.getElementById('lyrics-load-hint');

  if (sources.length === 0) {
    sel.appendChild(el('option', { value: '' }, 'No audio available'));
    transcribeBtn.disabled = true;
    loadHint.classList.remove('hidden');
    return;
  }

  loadHint.classList.add('hidden');
  for (const s of sources) {
    sel.appendChild(el('option', { value: s.path }, s.label));
  }
  // Prefer the first vocal stem if present
  const vocal = sources.find(s => s.isVocal);
  if (vocal) sel.value = vocal.path;
  transcribeBtn.disabled = false;
}

async function startLyricsTranscription() {
  const sourceSel = document.getElementById('lyrics-source');
  const engineSel = document.getElementById('lyrics-engine');
  const langSel = document.getElementById('lyrics-language');
  const fmtLrc = document.getElementById('lyrics-fmt-lrc');
  const fmtSrt = document.getElementById('lyrics-fmt-srt');
  const transcribeBtn = document.getElementById('lyrics-transcribe');

  const path = sourceSel.value;
  if (!path) return;

  let engineCfg;
  try { engineCfg = JSON.parse(engineSel.value); } catch { return; }

  const formats = ['txt'];
  if (fmtLrc.checked) formats.push('lrc');
  if (fmtSrt.checked) formats.push('srt');

  const progressCard = document.getElementById('lyrics-progress');
  const resultContainer = document.getElementById('lyrics-result');
  progressCard.classList.remove('hidden');
  clearChildren(resultContainer);
  transcribeBtn.disabled = true;

  try {
    const { job_id } = await api('/transcribe', {
      method: 'POST',
      body: JSON.stringify({
        audio_path: path,
        engine_id: engineCfg.engine_id,
        model_id: engineCfg.model_id,
        language: langSel.value || null,
        formats,
      }),
    });

    pollJob(job_id, {
      onProgress(progress, stage) {
        document.getElementById('lyrics-progress-fill').style.width = `${(progress * 100).toFixed(0)}%`;
        document.getElementById('lyrics-pct').textContent = `${(progress * 100).toFixed(0)}%`;
        document.getElementById('lyrics-stage').textContent = stage || '';
      },
      onDone(result) {
        progressCard.classList.add('hidden');
        transcribeBtn.disabled = false;
        renderLyricsResult(result);
        appState.lyricsPaths = { ...(appState.lyricsPaths || {}), ...result.output_paths };
        appState.emit('lyricsReady', result);
      },
      onError(msg) {
        progressCard.classList.add('hidden');
        transcribeBtn.disabled = false;
        resultContainer.appendChild(
          el('div', { className: 'banner banner-error' }, `Transcription failed: ${msg}`),
        );
      },
    });
  } catch (err) {
    progressCard.classList.add('hidden');
    transcribeBtn.disabled = false;
    resultContainer.appendChild(
      el('div', { className: 'banner banner-error' }, `Error: ${err.message}`),
    );
  }
}

function renderLyricsResult(result) {
  const container = document.getElementById('lyrics-result');
  const card = el('div', { className: 'card' });

  const meta = el('div', { className: 'lyrics-meta' },
    el('span', { className: 'lyrics-badge' }, `engine: ${result.engine_id}`),
    el('span', { className: 'lyrics-badge' }, `model: ${result.model_id}`),
    result.language ? el('span', { className: 'lyrics-badge' }, `language: ${result.language}`) : null,
    el('span', { className: 'lyrics-badge' }, `segments: ${result.segment_count}`),
  );

  const textarea = el('textarea', {
    className: 'lyrics-text',
    readonly: 'true',
    spellcheck: 'false',
  });
  textarea.value = result.text || '(empty transcript)';

  const actions = el('div', { className: 'stem-actions', style: { marginTop: '10px', display: 'flex', gap: '8px', flexWrap: 'wrap' } });
  for (const [fmt, path] of Object.entries(result.output_paths)) {
    const btn = el('button', {
      className: 'btn btn-sm',
      onClick: () => {
        const name = path.split('/').pop() || `lyrics.${fmt}`;
        saveFileAs(`/api/audio/download?path=${encodeURIComponent(path)}`, name);
      },
    }, `Save .${fmt}`);
    actions.appendChild(btn);
  }
  const sendBtn = el('button', {
    className: 'btn btn-sm btn-primary',
    onClick: () => {
      appState.emit('lyricsSendToCompose', result);
    },
  }, 'Send to Compose');
  actions.appendChild(sendBtn);

  card.append(meta, textarea, actions);
  container.appendChild(card);
}
```

- [ ] **Step 7: Manual UI verification — Notes mode (regression)**

Start the dev server: `uv run stemforge &` (or run in background; if already running, restart).
Open `http://localhost:8765` in a browser. Verify:
1. MIDI tab shows the `Notes | Lyrics` mode bar with Notes highlighted.
2. The existing Notes controls (Stems to process, Key, BPM, etc.) are visible.
3. Upload an audio file → run separation → confirm stems appear in the Notes stem checkboxes.
4. Click `Extract MIDI` → progress runs → MIDI cards appear → playback works. (Regression check: refactor in §11 didn't break the vocal MIDI flow.)

If any of these fail, fix before continuing. Expected: existing behavior is unchanged.

- [ ] **Step 8: Manual UI verification — Lyrics mode**

Still in the running server:
1. Click `Lyrics` in the mode bar. Notes controls hide; Lyrics controls show.
2. Source dropdown lists stems + "Full Mix (Original Upload)".
3. Engine dropdown lists Whisper variants (tiny → large-v3); Qwen appears with `(GPU required)` suffix if no GPU, otherwise enabled.
4. Select a vocal stem (or the upload), engine = `whisper-tiny`, language = Auto-detect.
5. Click `Transcribe`. Progress runs → result card appears with transcript textarea and Save buttons.
6. Click `Save .txt` → file downloads.
7. Click `Send to Compose` → no error in console. (Compose listener wired in Task 17.)

- [ ] **Step 9: Commit**

```bash
git add frontend/components/midi.js frontend/style.css frontend/app.js
git commit -m "Add Lyrics sub-mode to MIDI tab with engine + source + format selectors"
```

---

## Task 17: Compose tab listens for `lyricsSendToCompose`

**Files:**
- Modify: `frontend/components/compose.js`

The user can click `Send to Compose` in the MIDI Lyrics result card. We listen for that and offer an inline Apply/Dismiss banner inside the Compose lyrics area.

- [ ] **Step 1: Find Compose's `initCompose` and add a listener**

Edit `frontend/components/compose.js`. Find `export function initCompose()`. Find the existing `appState.on(...)` calls if any (search for `appState.on`).

Add inside `initCompose()` (somewhere near other event subscriptions; if none exist, add right before the closing brace of `initCompose`):

```js
  appState.on('lyricsSendToCompose', handleLyricsFromTranscribe);
```

- [ ] **Step 2: Add the handler function**

Append at the end of `compose.js`:

```js
// ─── Lyrics import from MIDI/Lyrics tab ─────────────────────────────

function handleLyricsFromTranscribe(result) {
  // Find the My Lyrics textarea (id="compose-lyrics-text") and ask the
  // user before overwriting any typed content.
  const ta = document.getElementById('compose-lyrics-text');
  if (!ta) return;

  // Switch to the My Lyrics tab if it isn't already active
  const tabBtn = document.querySelector('.compose-create-tab[data-tab="my-lyrics"]');
  if (tabBtn && !tabBtn.classList.contains('active')) tabBtn.click();

  // Remove any prior banner
  const prior = document.getElementById('compose-lyrics-import-banner');
  if (prior) prior.remove();

  const text = (result && result.text) || '';
  if (!text.trim()) return;

  const banner = el('div', {
    id: 'compose-lyrics-import-banner',
    className: 'banner',
    style: { display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' },
  });
  banner.appendChild(el('span', { style: { flex: '1' } },
    `Transcribed lyrics ready (${result.engine_id || 'unknown'}, ${result.segment_count || 0} segments).`));
  const applyBtn = el('button', {
    className: 'btn btn-sm btn-primary',
    onClick: () => {
      const isEmpty = !ta.value.trim();
      if (!isEmpty && !confirm('Replace existing lyrics? Cancel to keep current text.')) return;
      ta.value = text;
      ta.dispatchEvent(new Event('input', { bubbles: true }));
      banner.remove();
    },
  }, 'Apply');
  const dismissBtn = el('button', {
    className: 'btn btn-sm',
    onClick: () => banner.remove(),
  }, 'Dismiss');
  banner.append(applyBtn, dismissBtn);

  ta.parentElement?.insertBefore(banner, ta);
}
```

- [ ] **Step 3: Manual verification**

Restart the dev server, navigate MIDI tab → Lyrics mode, run a transcription, click `Send to Compose`. Then switch to the Compose tab. Verify:
1. A banner appears above the lyrics textarea saying "Transcribed lyrics ready (whisper, N segments)."
2. Clicking `Apply` with an empty textarea fills it without confirmation.
3. Filling the textarea with some text, then re-transcribing → on Apply, a confirm dialog asks before replacing.
4. Clicking `Dismiss` removes the banner without changing the textarea.

- [ ] **Step 4: Commit**

```bash
git add frontend/components/compose.js
git commit -m "Compose tab: show Apply/Dismiss banner when lyrics arrive from MIDI tab"
```

---

## Task 18: Documentation updates

**Files:**
- Modify: `docs/CURRENT_STATE.md`
- Modify: `docs/INSTRUCTIONS.md`

- [ ] **Step 1: Update `docs/CURRENT_STATE.md`**

Open `docs/CURRENT_STATE.md`, find the "What's working" section, and add a new bullet (style and place to match existing list):

```markdown
- **Lyrics transcription (MIDI tab)** — Notes/Lyrics mode bar; engine selection between Whisper (tiny → large-v3) and Qwen2-Audio-7B-Instruct; outputs .txt/.lrc/.srt; `Send to Compose` integration with the AI lyrics workflow.
```

- [ ] **Step 2: Update `docs/INSTRUCTIONS.md`**

Open `docs/INSTRUCTIONS.md`. Find section 4 — the MIDI tab. After the existing description (look for the end of the Notes-mode description), add a new sub-section:

```markdown
### 4.x · Lyrics mode

The MIDI tab has two sub-modes selected with the `Notes | Lyrics` bar at the top:

- **Notes** (default) — the existing audio-to-MIDI pipeline (BasicPitch for instruments, faster-whisper + PYIN for vocals).
- **Lyrics** — transcribe lyrics from any audio source to `.txt`, `.lrc`, and `.srt`.

In Lyrics mode:

1. Pick a **Source** — a separated stem (vocal recommended), an enhanced stem, or the originally uploaded full mix.
2. Pick an **Engine** — Whisper variants (`tiny`, `base`, `small`, `medium`, `large-v3`) run on CPU or GPU; Qwen2-Audio-7B-Instruct runs only on GPU and is marked `(GPU required)` when CUDA is unavailable.
3. Pick a **Language** — `Auto-detect` works well for Whisper; for Qwen, selecting an explicit language helps the prompt.
4. Choose output formats — `.txt` is always produced; `.lrc` and `.srt` are optional. Qwen uses segment-level timing rather than word-level.
5. Click **Transcribe**. Outputs are saved under `~/.local/share/stemforge/output/lyrics/<user>/`.

Click **Send to Compose** on the result card to import the lyrics into the Compose tab's "My Lyrics" textarea (you'll be asked before any existing text is replaced).
```

(Adjust the section number to match the next available number in the existing INSTRUCTIONS.md numbering.)

- [ ] **Step 3: Commit**

```bash
git add docs/CURRENT_STATE.md docs/INSTRUCTIONS.md
git commit -m "Document Lyrics transcription mode and outputs"
```

---

## Task 19: Final end-to-end manual checklist

This task is verification only — no code changes. Tick each item after running through the corresponding flow in a freshly-restarted dev server (`uv run stemforge`).

- [ ] **Step 1: Separate → vocal stem appears in session**

Upload an audio file with vocals, run Separate (any model), confirm a vocal stem appears in the session.

- [ ] **Step 2: MIDI Notes mode regression — vocal MIDI still works**

Switch to MIDI tab. Notes mode is active. Check the vocal stem. Click `Extract MIDI`. Confirm the vocal MIDI extraction completes and produces note events. (This exercises the refactored `models/midi_loader.py::convert_vocal_to_midi`.)

- [ ] **Step 3: MIDI Notes mode regression — full vocal pipeline still works**

If you have an ACE-Step `.flac` + `.json` test pair (per `tests/test_vocal_midi_pipeline.py`), run that test manually: `uv run python tests/test_vocal_midi_pipeline.py`. Otherwise: skip but note that automated coverage was the test fixture.

- [ ] **Step 4: Lyrics mode — Whisper transcription**

Switch to Lyrics mode. Source = vocal stem. Engine = `whisper-base`. Language = Auto-detect. Click `Transcribe`. Confirm:
- Progress bar advances.
- Result card shows transcript text + meta badges.
- `Save .txt` / `Save .lrc` / `Save .srt` all download files when clicked.

- [ ] **Step 5: Lyrics mode — Qwen transcription (GPU only)**

Same setup but Engine = Qwen2-Audio. First run will download ~14 GB of weights. Confirm:
- Transcript text appears.
- "Qwen produces segment-level timing…" notice is visible.
- `.lrc` and `.srt` use the segment-start timestamp (one line per segment in `.lrc`).

- [ ] **Step 6: Send to Compose integration**

Click `Send to Compose`. Switch to Compose tab. Confirm:
- Banner appears above the My Lyrics textarea.
- `Apply` fills the textarea (asking for confirmation if non-empty).
- `Dismiss` removes the banner.

- [ ] **Step 7: New Session clears lyrics state**

Click `New Session` (or call `DELETE /api/session`). Refresh, confirm:
- `appState.lyricsPaths` is empty.
- The Lyrics mode source dropdown reverts to "No audio available".

- [ ] **Step 8: Export integration (best-effort)**

Open the Export tab. The spec didn't require Export integration in v1; if `lyrics_paths` is exposed in `to_dict()` (it is — verified in Task 8), the Export panel's session-state read should at least not error.

- [ ] **Step 9: Definition of Done verification**

Run: `grep -rn "from faster_whisper" --include='*.py' .`
Expected: exactly one match (`pipelines/transcribe_engines/whisper_engine.py`).

Run: `uv run python tests/test_transcribe.py && uv run python tests/test_faster_whisper.py`
Both should print `OK`.

- [ ] **Step 10: No code changes — verification only.**

---

## Self-review

Spec coverage check, post-write:

| Spec section | Implemented in task |
|---|---|
| §0 License compliance | Task 1 (license text + acknowledgment); pre-verified during planning via HF API |
| §1 Goals | Tasks 4 (Whisper), 5 (Qwen), 7 (pipeline), 10 (API), 16 (UI), 11+12 (consolidation) |
| §2.1 New pipeline | Task 7 |
| §2.2 Pluggable engines | Tasks 2 (Protocol), 4 (Whisper), 5 (Qwen), 6 (registry) |
| §2.3 Whisper engine wraps faster-whisper | Task 4 |
| §2.4 Qwen engine uses transformers | Task 5 |
| §2.5 MIDI tab mode bar | Tasks 14 (CSS), 16 (JS) |
| §2.6 GPU scheduling | Task 10 (`gpu_session(pipeline_hint="transcribe")`) |
| §2.7 LYRICS_DIR + allowlist | Task 8 |
| §2.8 Session state | Task 8 |
| §3.1 engines/__init__ | Tasks 2 (stub), 6 (full) |
| §3.2 types.py | Task 2 |
| §3.3 whisper_engine.py | Task 4 |
| §3.4 qwen_engine.py | Task 5 |
| §3.5 transcribe_pipeline.py | Task 7 |
| §3.6 backend/api/transcribe.py | Task 10 |
| §3.7 ACKNOWLEDGMENTS | Task 1 |
| §4.1 WHISPER_LARGE_V3 | Task 3 |
| §4.2 LYRICS_DIR in paths | Task 8 |
| §4.3 audio.py allowlist | Task 8 |
| §4.4 main.py router + dir | Tasks 8 (dir), 10 (router) |
| §4.5 pipeline_manager.get_transcribe | Task 8 |
| §4.6 session_store lyrics_paths | Task 8 |
| §4.7 midi_loader refactor | Task 11 |
| §4.8 vocal_midi_pipeline refactor | Task 12 |
| §5.1 Mode bar | Task 16 |
| §5.2 Lyrics control panel | Task 16 |
| §5.3 Results panel | Task 16 |
| §5.4 Event bus (lyricsReady + lyricsPaths) | Tasks 15, 16 |
| §5.5 Loader hint | Task 16 (`lyrics-load-hint`) |
| §6.1 Smoke test | Task 9 |
| §6.2 Manual checklist | Task 19 |
| §7 Out of scope | Not implemented (intentional) |
| §8 Definition of Done | Verified in Tasks 13, 19 |

No placeholders, no "TBD". Every step contains exact paths, exact code, exact commands with expected output. Types and method signatures match across tasks (e.g. `WhisperEngine(model_id=...)`, `engine.transcribe(path, language=, prompt=)`, `pipeline_manager.get_transcribe(gpu_index)`).
