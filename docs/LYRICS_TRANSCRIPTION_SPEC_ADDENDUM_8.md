# Lyrics Transcription Spec — Addendum 8

**Parent doc:** `LYRICS_TRANSCRIPTION_SPEC.md` (plus Addenda 1-7)
**Status:** Ready for implementation
**Scope:** Replace the Qwen2-Audio engine with Qwen3-ASR. Remove Qwen2-Audio-specific chunking, stitching helpers, prompt-engineering layers, and quantization dependencies. Whisper engine and the broader transcription pipeline architecture are unchanged.

---

## 1 · Motivation and decision

Six prior addenda (1-7) attempted to make Qwen2-Audio behave reliably as a lyric transcription engine. Each fix introduced new failure modes:

| Phase | Fix | New failure introduced |
|---|---|---|
| Parent | Engine integration | (none — basic transcription worked) |
| 3 | Overlap chunking | Stitcher needed for long audio |
| 4 | Pairwise stitcher | (refinement, not new failure) |
| 5 | Hint UI field | Translation drift on Spanish hints |
| 6 | Wrapper with anti-translation anchor | Hint priming overrides audio attention |
| 7 Phase 2 | Fake-assistant-turn pattern | SRT formatting + Chinese fallback |
| 7 Phase 3 | Canonical minimal prompt | TBD (not yet validated) |

Each fix has been smaller and more surgical than the last, but the trajectory is not convergent. The root cause is that **Qwen2-Audio is a general-purpose multimodal audio model being adapted to a specific ASR task**, and every workaround navigates around its many other capabilities (audio captioning, sound event detection, voice chat, music description) rather than reinforcing the one we want.

The Qwen team released **Qwen3-ASR** in January 2026, specifically for ASR including music/song recognition. This is the right tool. Migrating to it removes most of the workarounds without losing any user-facing functionality.

### Decision

Replace Qwen2-Audio with Qwen3-ASR. Accept the loss of the Qwen2-Audio-specific implementation work as the cost of converging on a stable architecture.

### Why this is the right call now, not later

- Qwen3-ASR is purpose-built for the task. Most of Addenda 3-7's complexity becomes unnecessary.
- The model is smaller (1.7B vs 7B) and fits comfortably in 16 GB VRAM at bf16 without quantization.
- The API has dedicated parameters for everything we've been hand-rolling: language, vocabulary biasing, long-audio handling, optional timestamps.
- License is Apache 2.0 — same as Qwen2-Audio, no new compliance work.
- Continuing to iterate on Qwen2-Audio's failure modes is sunk-cost reasoning. Each fix takes longer than the last and gets us no closer to "done."

---

## 2 · Model facts (verified)

| Model | Parameters | Disk size | Approx VRAM (bf16) | License |
|---|---|---|---|---|
| `Qwen/Qwen3-ASR-1.7B` | 1.7B | 4.7 GB | ~6-7 GB peak | Apache 2.0 |
| `Qwen/Qwen3-ASR-0.6B` | 0.6B | ~1.5 GB | ~2-3 GB peak | Apache 2.0 |
| `Qwen/Qwen3-ForcedAligner-0.6B` | 0.6B | ~1.5 GB | ~2-3 GB peak | Apache 2.0 |

Both ASR variants and the optional forced-aligner companion are individually small enough to fit in 16 GB VRAM with a desktop session running.

### Capabilities

- 52 languages and dialects including Spanish, English, French, German, Italian, Portuguese, Japanese, Korean, Chinese, Cantonese, Arabic, Hindi, Russian.
- Trained on singing voice recognition specifically.
- Built-in long-audio handling — no external chunking required.
- Built-in language identification (auto-detect) and target-language pinning.
- Vocabulary biasing via a `prompt` parameter, documented as "biases the model without changing fundamental transcription behavior" — exactly the hint use case.
- Optional word-level timestamps via the separate Qwen3-ForcedAligner-0.6B model.
- Streaming and offline inference unified in a single model (we only need offline).

### Dependencies

The official `qwen-asr` Python toolkit is available on PyPI. It depends on `transformers` (already a project dependency) and standard audio libraries (already present).

---

## 3 · What gets removed

### Code

- `pipelines/transcribe_engines/qwen_engine.py` — entire file
- `pipelines/transcribe_engines/_qwen_chunker.py` — entire file
- `tests/test_transcribe.py` — `test_qwen_conversation_construction`, `test_qwen_chunker_*`, `test_qwen_stitcher_*` (entire blocks added in Addenda 3, 4, 6, 7)
- Any references to `qwen2-audio-7b-instruct` and `qwen2-audio-7b-instruct-nf4` model IDs in `models/registry.py` and `backend/api/transcribe.py::list_engines`

### Dependencies

- `bitsandbytes` from `pyproject.toml` — Qwen3-ASR runs in bf16 without quantization

### Concepts

- The 24/6 chunking scheme (Addendum 3) — Qwen3-ASR handles long audio internally
- The pairwise LCS stitcher (Addendum 4) — no longer needed
- Hint echo collapse and stoplist (Addendum 4 part 2) — no longer needed
- Whisper-style `condition_on_previous_text` / `collapse_repetitions` toggles applied to Qwen — these were always Whisper-only; just confirming they don't accidentally hook into the new path
- All prompt-engineering accumulated in Addenda 5, 6, 7 — replaced by direct API calls

### Documentation

- The "Qwen3-ASR future migration" note added to `docs/FUTURE_PLANS.md` in Addendum 7 Phase 3 §5 — this is no longer "future"
- Any "experimental — 30s chunks" disclaimers in the UI for the Qwen variants

---

## 4 · What gets added

### 4.1 · New engine file

`pipelines/transcribe_engines/qwen3_asr_engine.py`:

```python
"""Qwen3-ASR engine for multilingual lyric transcription.

License: Apache 2.0 — see licenses/LICENSE-Qwen3-ASR and ACKNOWLEDGMENTS.md.

Replaces the Qwen2-Audio engine that previously occupied this slot. Qwen3-ASR
is purpose-built for ASR (including music/song recognition), handles long
audio internally without external chunking, and exposes dedicated parameters
for language and vocabulary biasing — removing the prompt-engineering layer
that Addenda 5-7 spent considerable effort iterating on.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Literal

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
# The toolkit accepts both forms but human-readable is the documented default.
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
        self._model = None

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
                "qwen-asr package is required. Install via `uv add qwen-asr` or pip install qwen-asr.",
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

        # Qwen3-ASR's transcribe() handles long audio internally. No chunking needed.
        try:
            kwargs: dict = {"audio": [str(audio_path)]}
            if lang:
                kwargs["language"] = [lang]
            if prompt:
                kwargs["prompt"] = prompt  # vocabulary biasing — model's documented mechanism
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
        detected_language = getattr(r, "language", None)

        # Build a single segment covering the full duration. The toolkit does
        # not return segment-level timing without the forced-aligner companion.
        # If the toolkit's result object exposes a duration field, use it;
        # otherwise we leave end=0.0 as a known limitation.
        duration = float(getattr(r, "duration", 0.0))
        segment = TranscriptionSegment(
            start=0.0, end=duration, text=text, words=[],
        )

        return TranscriptionResult(
            text=text,
            language=detected_language or lang,
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
```

### 4.2 · Engine registry update

In `pipelines/transcribe_engines/__init__.py`, replace the Qwen2-Audio entry:

```python
# OLD:
from .qwen_engine import QwenEngine
ENGINES = {
    "whisper": WhisperEngine,
    "qwen": QwenEngine,
}

# NEW:
from .qwen3_asr_engine import Qwen3AsrEngine
ENGINES = {
    "whisper": WhisperEngine,
    "qwen3-asr": Qwen3AsrEngine,
}
```

### 4.3 · API list_engines update

In `backend/api/transcribe.py::list_engines`, replace the Qwen branch:

```python
elif engine_id == "qwen3-asr":
    from pipelines.transcribe_engines.qwen3_asr_engine import QWEN3_ASR_VARIANTS
    info["models"] = [
        {
            "model_id": mid,
            "display_name": v["display_name"],
            "approx_vram_gb": v["approx_vram_gb"],
            "description": v["description"],
            "available": cuda,
        }
        for mid, v in QWEN3_ASR_VARIANTS.items()
    ]
```

The Whisper branch is unchanged.

### 4.4 · Frontend update

In `frontend/components/midi.js`'s engine option annotation (added in Addendum 2 §3.1), update the Qwen branch:

```js
// OLD:
if (engine.engine_id === 'qwen') {
  const vram = model.approx_vram_gb ? `~${model.approx_vram_gb} GB VRAM` : 'GPU only';
  return `${label} (GPU required — ${vram})`;
}

// NEW:
if (engine.engine_id === 'qwen3-asr') {
  const vram = model.approx_vram_gb ? `~${model.approx_vram_gb} GB VRAM` : 'GPU only';
  return `${label} (GPU required — ${vram})`;
}
```

Same shape, just the engine_id changes.

### 4.5 · Dependency

Add to `pyproject.toml`:

```toml
"qwen-asr>=0.1.0",   # check current version on PyPI when implementing
```

Remove:

```toml
"bitsandbytes>=0.43.0",   # no longer needed without Qwen2-Audio NF4
```

Run `uv lock` after the edit.

### 4.6 · License file

Copy the Apache 2.0 license text from the Qwen3-ASR-1.7B HF repo to `licenses/LICENSE-Qwen3-ASR`.

Replace `licenses/LICENSE-Qwen2-Audio` with the Qwen3-ASR text, or delete and create new — the filename matters for the ACKNOWLEDGMENTS.md reference.

### 4.7 · ACKNOWLEDGMENTS.md update

Replace the Qwen2-Audio section with:

```markdown
## Qwen3-ASR — Alibaba Cloud (Tongyi Lab)

Multilingual automatic speech recognition model used as the GPU-backed
engine in the Lyrics transcription feature on the MIDI tab. Trained
specifically for speech and singing voice recognition across 52 languages.

- **Model:** https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- **Paper:** Shi et al. — *Qwen3-ASR Technical Report* (arXiv:2601.21337, 2026)
- **License:** Apache 2.0 (see `licenses/LICENSE-Qwen3-ASR`)
- **Toolkit:** https://github.com/QwenLM/Qwen3-ASR
```

Remove the bitsandbytes section (no longer a dependency).

### 4.8 · New unit test

Replace all Qwen2-Audio tests in `tests/test_transcribe.py` with a minimal Qwen3-ASR engine test that doesn't require the model to be loaded:

```python
def test_qwen3_asr_engine_registration() -> None:
    """Verify the Qwen3-ASR engine class is registered with correct metadata."""
    from pipelines.transcribe_engines import ENGINES
    from pipelines.transcribe_engines.qwen3_asr_engine import (
        Qwen3AsrEngine, QWEN3_ASR_VARIANTS,
    )

    assert "qwen3-asr" in ENGINES
    assert ENGINES["qwen3-asr"] is Qwen3AsrEngine
    assert "qwen3-asr-1.7b" in QWEN3_ASR_VARIANTS
    assert "qwen3-asr-0.6b" in QWEN3_ASR_VARIANTS

    # Invalid model_id raises.
    try:
        Qwen3AsrEngine(model_id="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown model_id")

    # Valid construction (no .load() called — that requires CUDA + weights).
    engine = Qwen3AsrEngine(model_id="qwen3-asr-1.7b")
    assert engine.engine_id == "qwen3-asr"
    assert engine.requires_gpu is True

    print("qwen3_asr_engine_registration OK")


def test_qwen3_asr_language_resolution() -> None:
    """Verify ISO code → Qwen3-ASR language name mapping."""
    from pipelines.transcribe_engines.qwen3_asr_engine import _resolve_language

    assert _resolve_language("es") == "Spanish"
    assert _resolve_language("en") == "English"
    assert _resolve_language("Spanish") == "Spanish"   # passthrough for already-resolved names
    assert _resolve_language(None) is None
    assert _resolve_language("") is None
    print("qwen3_asr_language_resolution OK")
```

Remove all of:
- `test_qwen_prompt_construction` (Addendum 6)
- `test_qwen_conversation_construction` (Addendum 7 Phase 2 and Phase 3)
- `test_qwen_chunker_*` (Addendum 3)
- `test_qwen_stitcher_*` (Addenda 3, 4)

These tested code that no longer exists.

---

## 5 · Migration order

Concrete sequence for Claude Code:

1. **Add the Qwen3-ASR engine** (§4.1, §4.6, §4.7, §4.8). Don't remove anything yet.
2. **Wire the new engine into the registry and API** (§4.2, §4.3). Now both engines are reachable.
3. **Update the frontend annotation** (§4.4). The dropdown will show Qwen3-ASR variants.
4. **Add the dependency and lock** (§4.5).
5. **Run a validation pass** — same as Phase 1/2/3 diagnostic, but on Qwen3-ASR-1.7B. Confirm it produces clean Spanish lyrics with the Caterina hint and no SRT / Chinese / chunk-boundary artifacts.
6. **Remove all Qwen2-Audio code and tests** (§3). Once Qwen3-ASR is validated, the old engine is dead weight.
7. **Update model registry and engine registry** to remove Qwen2-Audio references.
8. **Remove bitsandbytes** from dependencies (§4.5), run `uv lock` again.

Steps 1-5 are additive. Step 6 onwards is the deletion phase. The split lets you confirm the new engine works before throwing away the old one — important because the new engine's behavior on this specific test stem is unverified, and if it surprises us we want the option to keep Qwen2-Audio while we figure out what's going on.

---

## 6 · Validation

Same Catrina stem, same hint ("Feliz Cumpleanos Caterina"), same language ("es"). Direct pipeline invocation per Phase 1/2/3 pattern.

### Success criteria

1. **No SRT formatting.**
2. **No Chinese characters in the output.**
3. **No hint-only chunks** (the concept doesn't really apply because there's no chunking, but the broader test is: does the output have real lyric content throughout?).
4. **Final output length comparable to Whisper Large v3** (~1000-1500 chars for the Catrina song).
5. **Hint spelling honored** — "Caterina" appears in the output where the name is sung.
6. **No translation drift, no commentary, no meta-text.**

### Diagnostic logging

The INFO-level chunk preview log lines from Phase 1 are now irrelevant since Qwen3-ASR doesn't chunk externally. Demote them back to DEBUG in `qwen_engine.py` — except that file is being deleted, so this is automatic. The stitcher log lines in `_qwen_chunker.py` also go away with that file.

Replace with a single INFO log in `qwen3_asr_engine.py::transcribe`:

```python
log.info(
    "Qwen3-ASR transcribed %s (%.1fs audio, %d chars output, lang=%s).",
    audio_path.name, duration, len(text), detected_language or lang,
)
```

One line per transcription, not per chunk. Matches the simplicity of the new engine.

---

## 7 · Process notes (for the addendum-as-record-of-learning)

This is the fifth round of work on the Qwen integration. Patterns worth recording:

1. **Tool-task fit beats prompt engineering.** Six addenda of prompt engineering on Qwen2-Audio achieved less than one model swap to a purpose-built ASR model. If a general-purpose model is being adapted to a specific task and the workarounds are accumulating, looking for a purpose-built alternative is usually a higher-leverage move than another round of prompt iteration.

2. **The Qwen team shipped what we needed; we just hadn't checked.** Qwen3-ASR was released in January 2026. By the time Addendum 3 was written (Phase 1 of overlap chunking) the right model already existed. This is a search-the-current-state-of-the-art reminder for the next time we're 3+ addenda deep on a problem.

3. **Sunk-cost reasoning is the failure mode to watch for.** Six addenda of work on Qwen2-Audio created emotional pressure to keep iterating ("we're so close"). The user pushed back with the right question at the right time: "Why is ASR wrong right now? Is there a version that fits?" Asking that question proactively, not just reactively, is the discipline.

4. **Staged migration is worth the slight extra effort.** Adding Qwen3-ASR alongside Qwen2-Audio before removing the latter is safer than a direct swap. Even when the destination is clearly better, "ship it then remove it" is more recoverable than "swap it then debug it."

5. **Specs authored from PyPI README ≠ specs authored from `inspect.signature(...)`.** §4.1's code originally specified the vocabulary-biasing kwarg as `prompt=`. The PyPI `qwen-asr` 0.0.6 README only documents `audio`, `language`, and `return_time_stamps` in its Quickstart code blocks; the actual vocabulary-biasing parameter is `context=` and is documented only in the in-tree docstring + source. The implementation step caught this because the handoff included an explicit "do not improvise; stop and report if the API differs" gate. For future addenda involving an external library, the cheap defensive move is to `pip install` it and `inspect.signature(method)` before writing any reference code — the README is summary-grade, the live signature is authoritative.

---

## 8 · Definition of Done

Append to §8 of the parent spec:

66. `pipelines/transcribe_engines/qwen3_asr_engine.py` exists with `Qwen3AsrEngine` class and `QWEN3_ASR_VARIANTS` registry.
67. Engine registry `ENGINES` maps `"qwen3-asr"` to `Qwen3AsrEngine`. The `"qwen"` key is removed.
68. `pipelines/transcribe_engines/qwen_engine.py` and `pipelines/transcribe_engines/_qwen_chunker.py` are deleted.
69. All Qwen2-Audio unit tests in `tests/test_transcribe.py` are removed. New `test_qwen3_asr_engine_registration` and `test_qwen3_asr_language_resolution` pass.
70. `pyproject.toml` includes `qwen-asr`. `bitsandbytes` is removed. `uv.lock` is refreshed.
71. `licenses/LICENSE-Qwen3-ASR` exists with the upstream Apache 2.0 text. `licenses/LICENSE-Qwen2-Audio` is removed.
72. `ACKNOWLEDGMENTS.md` has a Qwen3-ASR entry; Qwen2-Audio and bitsandbytes entries are removed.
73. MIDI Lyrics dropdown shows three Whisper variants + two Qwen3-ASR variants (1.7B and 0.6B).
74. Manual validation on the Catrina stem produces clean Spanish lyrics with no SRT, no Chinese, no hint-only output, and final length comparable to Whisper Large v3.
75. `grep -rn "Qwen2-Audio\|qwen2-audio\|QwenEngine\|_qwen_chunker" --include='*.py' --include='*.md'` returns no functional references (only historical mentions in addendum docs, which are fine).
