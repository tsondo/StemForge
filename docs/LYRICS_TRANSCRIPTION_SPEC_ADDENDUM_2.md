# Lyrics Transcription Spec — Addendum 2

**Parent doc:** `LYRICS_TRANSCRIPTION_SPEC.md` (plus Addendum 1)
**Status:** Ready for implementation
**Scope:** Two cleanups and one addition, in one Claude Code pass:
1. Prune the Whisper variant registry from five entries to three.
2. Add a bitsandbytes 4-bit NF4 model variant for the Qwen2-Audio engine.
3. Refresh the engine dropdown copy with the cut-down list and `(recommended)` / `(GPU required — N GB VRAM)` annotations.

---

## 1 · Whisper variant pruning

### 1.1 · Rationale

Five Whisper sizes (`tiny`, `base`, `small`, `medium`, `large-v3`) is more than the UI benefits from. Three is enough:

| Keep | Tier | Use case |
|---|---|---|
| `whisper-tiny` | Fastest | Tests, fast CPU jobs |
| `whisper-small` | Balanced | CPU fallback default |
| `whisper-large-v3` | Best | GPU default, music — recommended |

`whisper-base` is barely better than `tiny` and worse than `small` — an interpolation point with no clear use case. `whisper-medium` is worse than `large-v3` and not meaningfully smaller or faster on any GPU that can run either. Both are dropped.

### 1.2 · Remove from `models/registry.py`

Delete the two `_register(...)` blocks for `WHISPER_BASE` and `WHISPER_MEDIUM`. Keep `WHISPER_TINY`, `WHISPER_SMALL`, and `WHISPER_LARGE_V3` (added in the parent spec §4.1).

### 1.3 · Update `DEFAULT_WHISPER_SPEC`

In the "Convenience constants" section of `models/registry.py`:

```python
# OLD:
DEFAULT_WHISPER_SPEC: WhisperSpec = WHISPER_BASE

# NEW:
DEFAULT_WHISPER_SPEC: WhisperSpec = WHISPER_LARGE_V3
```

Note: this constant is referenced by `models/midi_loader.py` for the vocal→MIDI flow on the MIDI tab's Notes mode. Promoting `large-v3` to the default means Notes mode also gets the quality upgrade. If `large-v3` weights are not cached and CUDA is unavailable, faster-whisper will still run it on CPU — slow but functional. This is acceptable: users with no GPU were already not getting state-of-the-art results, and the registry default isn't the right knob to encode CPU-friendliness. The MIDI tab UI continues to expose all three sizes so a CPU user can pick `whisper-small` or `whisper-tiny` themselves.

### 1.4 · Update default in `pipelines/vocal_midi_pipeline.py`

In `VocalMidiConfig.__init__`:

```python
# OLD:
whisper_model_size: str = "base",

# NEW:
whisper_model_size: str = "small",
```

Why `small` and not `large-v3` here despite the registry default? `VocalMidiConfig` is consumed by the MIDI tab's *Notes* mode for vocal→MIDI conversion, where Whisper is used for word boundaries to drive PYIN pitch estimation. Word boundary quality is the relevant metric; lyric semantic accuracy is not. `small` is enough for word boundaries and substantially faster on CPU than `large-v3`. The Lyrics mode of the MIDI tab continues to default to `large-v3` via `TranscribeConfig` (no change there).

Also update the docstring in `VocalMidiConfig` — find:

```
        ``"tiny"``, ``"base"``,
        ``"small"``, or ``"medium"``).  Larger models are more accurate at the
```

and change to:

```
        ``"tiny"``, ``"small"``, or ``"large-v3"``).  Larger models are more
        accurate at the
```

Also update the `whisper_model_size` parameter docstring default annotation from `Default: "base"` to `Default: "small"`.

### 1.5 · Check for stale references

Grep for `whisper-base`, `whisper-medium`, `"base"`, and `"medium"` across the repo. Likely hits:

- `models/midi_loader.py` — confirm it uses `DEFAULT_WHISPER_SPEC` rather than a hardcoded `"base"` string.
- `pipelines/transcribe_engines/whisper_engine.py` — default `model_id="whisper-base"` from the parent spec needs updating to `"whisper-small"`.
- `pipelines/transcribe_pipeline.py` — `TranscribeConfig.model_id` default `"whisper-base"` from the parent spec needs updating to `"whisper-large-v3"` (this is the lyrics-mode default, distinct from the vocal-MIDI default).
- `backend/api/transcribe.py` — `TranscribeRequest.model_id` default `"whisper-base"` from the parent spec needs updating to `"whisper-large-v3"`.
- `tests/test_transcribe.py` — the smoke test uses `whisper-tiny`, no change needed.
- `tests/test_faster_whisper.py` — if it uses `whisper-base`, update to `whisper-small`.

If any other file references the removed model IDs as strings, the call site needs to be updated. The registry will raise `KeyError` at startup if a stale `get_spec("whisper-base")` call survives — which is the right behaviour, fail fast and visibly.

### 1.6 · Update `CLAUDE.md` model registry table

In the project root `CLAUDE.md`, the registry section currently reads:

```
| `WhisperSpec` | whisper-tiny, whisper-base, whisper-small, whisper-medium | `VocalMidiPipeline` |
```

Change to:

```
| `WhisperSpec` | whisper-tiny, whisper-small, whisper-large-v3 | `VocalMidiPipeline`, `TranscribePipeline` |
```

(The Pipeline column updates to reflect that Whisper specs now serve two pipelines, not one.)

### 1.7 · Update `docs/CURRENT_STATE.md` if it lists Whisper variants

If `docs/CURRENT_STATE.md` enumerates Whisper sizes, prune the list there too. If it only says "faster-whisper integration" without specifics, no change needed.

---

## 2 · Qwen2-Audio 4-bit NF4 model option

### 2.1 · Rationale

Qwen2-Audio-7B-Instruct at fp16 needs ~16 GB VRAM, uncomfortably tight on a 16 GB laptop with a desktop session running. NF4 4-bit quantization with double-quant via bitsandbytes drops this to ~9 GB at inference peak — comfortable headroom on 16 GB cards, and the only path that lets Tsondo's current laptop run the Qwen engine at all.

Same weights, same engine class, same chat-template prompt — only the `from_pretrained` call differs. This is a single new `model_id`, not a new engine.

**Quality:** NF4 retains roughly 95-98% of fp16 quality on Qwen-family models for language tasks. Audio transcription falls in a similar range. Acceptable for development use and likely acceptable for production use on this app.

### 2.2 · Add dependency

In `pyproject.toml` (and `pyproject.toml.MAC` if it carries the same dependency list), add to the main dependencies block, near `transformers` and `accelerate`:

```toml
    # 4-bit quantization for Qwen2-Audio (lyrics transcription)
    "bitsandbytes>=0.43.0",
```

`bitsandbytes` is Linux/CUDA only. On macOS it has no GPU implementation; the `pyproject.toml.MAC` variant should either omit this dependency or include it (it will install but the engine won't be reachable on Mac since `requires_gpu=True` already gates it). Cleanest: omit from `pyproject.toml.MAC`.

Run `uv lock` after the edit to refresh `uv.lock`.

### 2.3 · Register a new `QwenAudioSpec` variant (optional but clean)

The existing parent spec didn't add a `QwenAudioSpec` to `models/registry.py` — Qwen lives inside the engine code without a registry entry, mirroring how the engine module hard-codes the repo path. For two model variants this gets a little awkward but is still acceptable; adding a real spec class is optional cleanup.

**Recommendation:** keep it simple — no new spec class. Encode the two variants directly in the engine and the API listing. Skip ahead to §2.4.

### 2.4 · Engine code changes — `pipelines/transcribe_engines/qwen_engine.py`

Replace the existing single-variant class with a parameterized one. Constructor takes a `quantization` kwarg:

```python
"""Qwen2-Audio engine for high-fidelity sung-lyrics transcription.

LICENSE: Apache 2.0 — see licenses/LICENSE-Qwen2-Audio and ACKNOWLEDGMENTS.md.
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

_QWEN_REPO = "Qwen/Qwen2-Audio-7B-Instruct"
_PROMPT = (
    "Transcribe the lyrics of this audio. Output only the lyrics text, "
    "preserving line breaks where the singer pauses. Do not add commentary, "
    "explanations, or section labels."
)

Quantization = Literal["none", "nf4"]

# Variant metadata for the /api/transcribe/engines endpoint.
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

    def __init__(self, model_id: str = "qwen2-audio-7b-instruct") -> None:
        if model_id not in QWEN_VARIANTS:
            raise ValueError(
                f"Unknown qwen model_id {model_id!r}. "
                f"Available: {sorted(QWEN_VARIANTS)}"
            )
        self.model_id = model_id
        self._variant = QWEN_VARIANTS[model_id]
        self._quantization: Quantization = self._variant["quantization"]
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
                "device_map": "cuda",
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

    # transcribe() and clear() are unchanged from the parent spec §3.4.
```

Keep the `transcribe()` and `clear()` methods from the parent spec verbatim.

### 2.5 · Pipeline wiring — `pipelines/transcribe_pipeline.py`

In `TranscribePipeline.load_model()`, the Whisper branch passes a `model_id`; the Qwen branch currently passes nothing. Update the Qwen branch to forward `model_id` too:

```python
if self._config.engine_id == "whisper":
    self._engine = engine_cls(
        model_id=self._config.model_id,
        condition_on_previous_text=self._config.condition_on_previous_text,
    )
elif self._config.engine_id == "qwen":
    self._engine = engine_cls(model_id=self._config.model_id)
else:
    self._engine = engine_cls()
```

No other pipeline changes needed.

### 2.6 · API listing — `backend/api/transcribe.py`

Update the Qwen branch of `list_engines()` to enumerate both variants from `QWEN_VARIANTS`:

```python
        elif engine_id == "qwen":
            from pipelines.transcribe_engines.qwen_engine import QWEN_VARIANTS
            info["models"] = [
                {
                    "model_id": mid,
                    "display_name": v["display_name"],
                    "approx_vram_gb": v["approx_vram_gb"],
                }
                for mid, v in QWEN_VARIANTS.items()
            ]
```

The Whisper branch is already correct.

---

## 3 · Frontend updates — `frontend/components/midi.js`

The Lyrics control panel's **Engine** dropdown is built from the `/api/transcribe/engines` response. With the registry pruned and the Qwen variant added, the rendered list should look like:

```
Whisper Large v3            (recommended — GPU)
Whisper Small               (CPU-friendly)
Whisper Tiny                (fastest, lower quality)
Qwen2-Audio 7B              (GPU required — ~16 GB VRAM)
Qwen2-Audio 7B (4-bit)      (GPU required — ~9 GB VRAM)
```

### 3.1 · Annotation rendering

When building the `<option>` elements, append a contextual suffix to each `display_name`:

```js
function annotateEngineOption(engine, model, cudaAvailable) {
  const label = model.display_name;

  // Whisper variants
  if (engine.engine_id === 'whisper') {
    if (model.model_id === 'whisper-large-v3') return `${label} (recommended — GPU)`;
    if (model.model_id === 'whisper-small')    return `${label} (CPU-friendly)`;
    if (model.model_id === 'whisper-tiny')     return `${label} (fastest, lower quality)`;
    return label;
  }

  // Qwen variants — show VRAM
  if (engine.engine_id === 'qwen') {
    const vram = model.approx_vram_gb ? `~${model.approx_vram_gb} GB VRAM` : 'GPU only';
    return `${label} (GPU required — ${vram})`;
  }

  return label;
}
```

### 3.2 · Default selection

When the dropdown is first populated, pre-select `whisper-large-v3`. If the user later switches engines, remember their last choice in `appState.lyricsEngineId` and restore it on subsequent visits to the tab. (This mirrors how other tabs persist mode selections.)

### 3.3 · Greying out unavailable options

The `available` flag from the engines endpoint already drives this for Qwen on CPU-only hosts. For the 4-bit Qwen variant specifically: even with CUDA available, if `bitsandbytes` failed to import (older install, no CUDA toolkit), the option should be disabled with a `(bitsandbytes not installed)` suffix.

The cleanest way to surface this is to expose a per-model `available` field from the API:

```python
# In list_engines(), inside the qwen branch, replace the simple list comp with:
qwen_models = []
bnb_available = True
try:
    from transformers import BitsAndBytesConfig  # noqa: F401
    import bitsandbytes  # noqa: F401
except ImportError:
    bnb_available = False
for mid, v in QWEN_VARIANTS.items():
    model_available = cuda
    if v["quantization"] == "nf4" and not bnb_available:
        model_available = False
    qwen_models.append({
        "model_id": mid,
        "display_name": v["display_name"],
        "approx_vram_gb": v["approx_vram_gb"],
        "available": model_available,
    })
info["models"] = qwen_models
```

Then in the frontend, disabled options get a `(GPU required — N GB VRAM, unavailable)` suffix or similar. Keep this terse — the tooltip can explain why.

### 3.4 · Tooltip on the engine selector

Add a `title` attribute to the `<select>` itself (or its label) explaining the trade-off:

> Whisper is the standard speech recognition engine — fast, supports word-level timestamps, runs on CPU or GPU. Qwen2-Audio is a multimodal language model — slower, more accurate on sung or unusual vocals, no word timestamps, GPU only. The 4-bit Qwen variant fits in ~9 GB VRAM with minor quality loss.

---

## 4 · Documentation

### 4.1 · `ACKNOWLEDGMENTS.md`

The existing Qwen2-Audio entry from the parent spec §3.7 needs no change — it already covers both fp16 and 4-bit since they're the same upstream model.

Add a new entry for bitsandbytes, immediately after the Hugging Face Transformers entry:

```markdown
---

## bitsandbytes — Tim Dettmers et al.

4-bit and 8-bit quantization library used to load Qwen2-Audio at reduced
VRAM for the optional 4-bit lyrics transcription mode.

- **Repository:** https://github.com/bitsandbytes-foundation/bitsandbytes
- **Paper:** Dettmers et al. — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (NeurIPS 2022)
- **License:** MIT
```

### 4.2 · `docs/INSTRUCTIONS.md`

The MIDI tab section gained a Notes/Lyrics mode bar in the parent spec. Find the Lyrics-mode bullet that lists engine options and update it to match the new list of five (3 Whisper + 2 Qwen).

---

## 5 · Testing

### 5.1 · Existing tests

- `tests/test_transcribe.py` smoke test uses `whisper-tiny` — unchanged.
- `tests/test_faster_whisper.py` — if it references `whisper-base` or `whisper-medium`, update to `whisper-small`.

### 5.2 · No new automated test for Qwen NF4

Same rationale as the parent spec — Qwen requires GPU, the download is large, the test would be slow and flaky in CI. Skip.

### 5.3 · Manual checklist additions

- [ ] `uv lock && uv sync` succeeds with `bitsandbytes` added.
- [ ] `uv run stemforge` starts; no `KeyError` on startup from stale references to `whisper-base` / `whisper-medium`.
- [ ] MIDI Notes mode runs vocal→MIDI successfully with the new default `whisper-small`.
- [ ] MIDI Lyrics mode dropdown shows exactly 5 options with the suffixes specified in §3.1.
- [ ] On the 16 GB laptop, select **Qwen2-Audio 7B (4-bit)**, transcribe the Catrina stem. Confirm the model loads, VRAM peak stays under ~12 GB (watch `nvidia-smi`), output text matches the chat-template prompt format.
- [ ] On the same laptop, select **Qwen2-Audio 7B** (the fp16 variant). Confirm it either OOMs cleanly with a `ModelLoadError` surfaced in the UI, or — if it happens to fit — completes. Either outcome is correct; the test is that the system doesn't crash uncontrollably.
- [ ] Switch engines between calls (Whisper → Qwen 4-bit → Whisper) and confirm `pipeline_manager.evict("transcribe", ...)` is releasing VRAM between runs.

---

## 6 · Definition of Done (addendum)

Append to §8 of the parent spec:

14. `models/registry.py` contains exactly three Whisper specs: `WHISPER_TINY`, `WHISPER_SMALL`, `WHISPER_LARGE_V3`. `DEFAULT_WHISPER_SPEC` points to `WHISPER_LARGE_V3`.
15. `grep -rn "whisper-base\|whisper-medium" --include='*.py' --include='*.js' --include='*.md'` returns no functional references (only changelog / historical mentions, if any).
16. `pyproject.toml` lists `bitsandbytes>=0.43.0`. `uv.lock` is refreshed.
17. `pipelines/transcribe_engines/qwen_engine.py` accepts a `model_id` constructor arg and supports both `qwen2-audio-7b-instruct` (fp16) and `qwen2-audio-7b-instruct-nf4` (NF4).
18. `/api/transcribe/engines` returns five model entries total: 3 Whisper + 2 Qwen, each with `available` and (for Qwen) `approx_vram_gb` fields.
19. MIDI Lyrics dropdown shows exactly five annotated entries matching §3.1.
20. `ACKNOWLEDGMENTS.md` contains the new bitsandbytes entry.
21. Manual checklist (§5.3) passes on Tsondo's laptop, specifically the Qwen 4-bit transcription run.
