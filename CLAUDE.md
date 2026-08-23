# StemForge — Claude Context

## Workflow
- Break all multi-step tasks into a numbered plan before starting
- After each major step, commit and push, then pause and report: what was done, what changed, what's next
- Wait for explicit "continue" or "proceed" confirmation before moving to the next step

## What this project is

AI-powered audio processing web application with seven core pipelines:
- **Demucs** — source separation (vocals, drums, bass, other) — 4 models
- **BS-Roformer** — high-quality separation with 2-stem, 4-stem, and 6-stem (guitar + piano) models
- **BasicPitch** — polyphonic MIDI extraction from separated stems (instruments)
- **Vocal MIDI** — vocal pitch-to-MIDI via faster-whisper + PYIN pitch tracking
- **ADTOF** — drum stem transcription to GM channel-10 MIDI (kick, snare, tom, hi-hat, cymbal); drum-labelled stems auto-route here instead of BasicPitch. Weights are CC BY-NC-SA 4.0 — extraction is gated behind a UI license acknowledgment
- **Stable Audio Open** — text-conditioned audio generation with optional audio and MIDI conditioning (Synth tab)
- **AceStep** — full song generation from style descriptions + lyrics (Compose tab, runs as subprocess)

Additional systems:
- **Enhance** — three-mode vocal enhancement tab:
  - **Clean Up** — UVR denoise, dereverb, debleed via vendored `python-audio-separator` fork (8 curated presets across Roformer/MDXC/VR architectures)
  - **Tune** — auto-tune via CREPE neural pitch detection (`torchcrepe`) + selectable resynthesis method: WORLD vocoder (`pyworld`, best on lossless audio) or STFT phase vocoder with formant preservation (`stftpitchshift`, better on compressed/MP3); scale snapping with correction strength and humanization controls
  - **Effects** — per-stem channel strip with 4 effect types (EQ, Compressor, Noise Gate, Stereo Width), each offering DSP and/or ML methods: 3-band parametric EQ (`scipy.signal`), DSP/LA-2A neural compressor (`vendor/micro_tcn`, Apache 2.0), DSP/Spectral (`torchgating`)/DeepFilterNet(disabled, numpy<2.0 conflict) noise gate, Mid/Side stereo width
- **Model registry** (`models/registry.py`) — frozen `ModelSpec` descriptors for all models; single source of truth for device rules, sample rates, capabilities, metadata, and pipeline defaults
- **Audio profiler** (`utils/audio_profile.py`) — spectral analysis that recommends the best engine/model for a given audio file
- **Mix engine** — multi-track mixer combining audio stems and MIDI-rendered tracks with per-track instrument, volume, and FLAC render
- **SFX Stem Builder** (`backend/api/sfx.py`) — DAW-style canvas for placing audio clips on a timeline, aligned to a reference stem, with per-clip fades and volume; renders to a single stem for the Mix engine

**Architecture**: FastAPI backend (`backend/`) + vanilla HTML/CSS/JS frontend (`frontend/`) + AceStep subprocess.
Run with `python run.py` → open `http://localhost:8765` in browser.
AceStep runs on port 8001 by default. Disable with `--no-acestep`.

---

## Current state

All pipelines and the full web UI are implemented:

- Demucs separation — 4 models (htdemucs, htdemucs_ft, mdx_extra, mdx_extra_q), CUDA fallback for MDX-Net
- BS-Roformer separation — 6 models including ViperX vocals (SDR 12.97), KJ vocals, ZFTurbo 4-stem, jarredou 6-stem
- Automatic engine/model recommendation from spectral audio analysis
- MIDI extraction — BasicPitch for instruments, faster-whisper + PYIN pitch for vocals, ADTOF for drums (channel 10, license-gated)
- MIDI preview — server-side FluidSynth render, streamed to browser via wavesurfer.js
- Mix tab — per-track volume controls, audio/MIDI source types, FLAC render, multi-track preview
- Stable Audio Open generation (Synth tab) — text + audio + MIDI conditioning, up to 600 s (chunked at 47 s), Vocal Preservation Mode
- SFX Stem Builder (Synth tab) — DAW timeline, clip placement with fades, align-to reference waveform, render canvas to Mix
- AceStep generation (Compose tab) — full song creation/rework, AI lyrics, 3-column UI, cross-tab integration, LoRA adapter management, project save/load, seed recall, dismissable result cards
- AceStep LoRA training (Compose tab Train mode) — upload audio, scan/label/preprocess pipeline, LoRA/LoKR fine-tuning with loss chart, snapshot management, adapter export
- Batch separation — multi-file upload, single-stem extraction across all files, Save All zip download
- Upload supports audio (WAV, FLAC, MP3, OGG, AIFF) and video (MP4, MKV, WEBM, AVI, MOV) — video audio extracted via FFmpeg
- Export panel — all pipeline outputs, 6 audio formats (wav/flac/aiff/mp3/ogg/m4a), MIDI export (per-stem + merged multi-track, SMF format 0 or 1 for hardware arrangers), zip download
- Waveform visualization via wavesurfer.js with global transport bar
- Deterministic uv environment, Python 3.12, CUDA 13.0 wheels
- macOS support via MPS acceleration (separate `pyproject.toml.MAC`)

---

## Project structure

```
StemForge/
├── run.py                          # Launcher: uvicorn + AceStep subprocess management
├── pyproject.toml
├── pyproject.toml.MAC              # macOS variant (MPS, no CUDA index)
├── pyproject.toml.ROCM             # AMD variant (ROCm 7.1 torch index)
│
├── Ace-Step-Wrangler/              # Git submodule (independently runnable)
│   ├── vendor/ACE-Step-1.5/        # Nested submodule — upstream AceStep
│   ├── backend/                    # Wrangler's standalone backend (reference)
│   ├── frontend/                   # Wrangler's standalone frontend (reference)
│   └── run.py                      # Wrangler's standalone launcher (unused in StemForge)
│
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI app, router registration, static mount
│   ├── api/
│   │   ├── __init__.py
│   │   ├── system.py               # /api/health, /api/device, /api/models, /api/session
│   │   ├── audio.py                # /api/upload, /api/upload-batch, /api/audio/stream|download|waveform|info, /api/audio/profile
│   │   ├── separate.py             # /api/separate, /api/separate/batch, /api/separate/recommend, /api/jobs/{id}
│   │   ├── midi.py                 # /api/midi/extract|render|save|stems
│   │   ├── generate.py             # /api/generate (Synth tab)
│   │   ├── compose.py              # /api/compose/* (Compose tab — AceStep proxy)
│   │   ├── acestep_wrapper.py      # HTTP client for AceStep API
│   │   ├── enhance.py              # /api/enhance, /api/enhance/presets|stems|autotune (Enhance tab)
│   │   ├── mix.py                  # /api/mix/tracks|render|add-audio|add-midi
│   │   ├── sfx.py                  # /api/sfx/* (SFX Stem Builder — canvas, placements, render)
│   │   └── export.py               # /api/export, /api/export/download-zip
│   └── services/
│       ├── __init__.py
│       ├── job_manager.py          # Background thread runner + in-memory job store
│       ├── session_store.py        # Thread-safe session state (replaces old AppState)
│       ├── pipeline_manager.py     # Lazy-loaded pipeline singletons
│       └── acestep_state.py        # AceStep subprocess status (disabled/starting/running/crashed)
│
├── frontend/
│   ├── index.html                  # SPA shell — header, tab bar, tab panels, transport bar
│   ├── style.css                   # Design tokens + full layout (dark DAW aesthetic)
│   ├── app.js                      # State management, event bus, tab switching, poll helper
│   └── components/
│       ├── loader.js               # Drag-and-drop upload + file info + batch mode
│       ├── waveform.js             # wavesurfer.js wrapper
│       ├── separate.js             # Separation tab + batch mode
│       ├── enhance.js             # Enhance tab (Clean Up / Tune / Effects mode bar)
│       ├── waveform-diff.js       # Shared audio peak diff visualization
│       ├── midi.js                 # MIDI tab
│       ├── mix.js                  # Mix tab
│       ├── generate.js             # Synth tab (Stable Audio Open)
│       ├── compose.js              # Compose tab (AceStep — 3-col layout)
│       ├── export.js               # Export tab
│       ├── midi-viz.js             # Canvas piano roll
│       └── audio-player.js         # Global transport bar
│
├── pipelines/                      # All pipeline logic
│   ├── enhance_pipeline.py         # UVR denoise/dereverb via audio-separator
│   ├── autotune_pipeline.py        # CREPE pitch detection + WORLD/STFT resynthesis
│   ├── effects_pipeline.py        # Effects chain (EQ, compressor, gate, stereo width)
│   ├── demucs_pipeline.py
│   ├── roformer_pipeline.py
│   ├── midi_pipeline.py
│   ├── basicpitch_pipeline.py
│   ├── vocal_midi_pipeline.py
│   ├── musicgen_pipeline.py
│   └── resample.py
│
├── models/                         # UNCHANGED — model registry + loaders
│   ├── registry.py
│   ├── demucs_loader.py
│   ├── roformer_loader.py
│   ├── midi_loader.py
│   ├── basicpitch_loader.py
│   ├── adtof_loader.py
│   ├── basicpitch/
│   └── musicgen_loader.py
│
├── utils/
│   ├── cache.py                    # Model cache dir resolution (MODEL_LOCATION)
│   ├── paths.py                    # Output directory constants (shared across layers)
│   ├── audio_io.py                 # read_audio / write_audio
│   ├── audio_profile.py            # Spectral analysis + engine recommendation
│   ├── midi_io.py                  # MIDI read / write / helpers
│   ├── device.py                   # get_device / is_mps — platform-aware torch device
│   ├── platform.py                 # get_data_dir — OS-idiomatic data paths
│   ├── logging_utils.py            # configure_logging
│   └── errors.py                   # Custom exception hierarchy
```

---

## Import layer order (no circular imports)

```
utils/  →  models/  →  pipelines/  →  backend/services/  →  backend/api/  →  backend/main.py
```

`utils/paths.py` holds output directory constants — imported by pipelines, backend, and any layer.
`frontend/` is purely static (served by FastAPI's `StaticFiles`).

---

## Backend architecture

### Services (`backend/services/`)

| Service | Purpose |
|---|---|
| `job_manager.py` | `JobManager` — background thread runner, UUID-based job store, progress callback bridge |
| `session_store.py` | `SessionStore` — thread-safe singleton replacing old `AppState`; holds audio/stem/MIDI/mix/compose state |
| `pipeline_manager.py` | Lazy-loaded pipeline singletons with GPU memory lock; `get_demucs()`, `get_roformer()`, `get_effects()`, etc. |
| `acestep_state.py` | Thread-safe AceStep subprocess status: disabled / starting / running / crashed |

### API Endpoints (`backend/api/`)

| Method | Path | Type | Purpose |
|--------|------|------|---------|
| GET | /api/health | sync | Health check |
| GET | /api/device | sync | GPU/device info |
| GET | /api/models | sync | All registered models |
| GET | /api/session | sync | Current session state |
| DELETE | /api/session | sync | Clear session |
| POST | /api/upload | sync | Upload audio/video file (video → FFmpeg audio extraction) |
| POST | /api/upload-batch | sync | Upload multiple audio/video files for batch processing |
| GET | /api/audio/stream | sync | Stream audio (inline) |
| GET | /api/audio/download | sync | Download audio (attachment) |
| GET | /api/audio/waveform | sync | Downsampled peaks JSON |
| GET | /api/audio/info | sync | Audio metadata |
| POST | /api/audio/profile | sync | Audio profiler + recommendation |
| POST | /api/separate | job | Start separation |
| POST | /api/separate/batch | job | Batch separation — single stem from multiple files |
| POST | /api/separate/batch/save-all | sync | Zip batch results for download |
| GET | /api/separate/recommend | sync | Quick engine recommendation |
| GET | /api/jobs/{id} | sync | Poll any job's status |
| POST | /api/midi/extract | job | Start MIDI extraction |
| POST | /api/midi/render | sync | FluidSynth render to WAV |
| POST | /api/midi/save | sync | Save MIDI to disk |
| GET | /api/midi/stems | sync | Available MIDI stem labels |
| POST | /api/generate | job | Start audio generation (Synth) |
| GET | /api/compose/health | sync | AceStep subprocess status |
| POST | /api/compose/start | async | Launch AceStep on first use (idempotent) |
| POST | /api/compose/generate | async | Start AceStep generation (Compose) |
| GET | /api/compose/status/{id} | sync | Poll AceStep task status |
| GET | /api/compose/audio | sync | Audio proxy from AceStep |
| GET | /api/compose/download/{id}/{n}/audio | sync | Download compose result audio |
| GET | /api/compose/download/{id}/{n}/json | sync | Download compose result metadata |
| POST | /api/compose/generate-lyrics | sync | AI lyrics generation via AceStep LM |
| POST | /api/compose/estimate-duration | sync | Auto-duration estimation |
| POST | /api/compose/estimate-sections | sync | Section structure estimation |
| POST | /api/compose/upload-audio | sync | Upload audio for Rework mode |
| POST | /api/compose/send-to-session | sync | Save compose audio to session |
| POST | /api/compose/lora/load | async | Load LoRA/LoKR adapter |
| POST | /api/compose/lora/unload | async | Unload adapter, restore base model |
| POST | /api/compose/lora/toggle | async | Enable/disable loaded adapter |
| POST | /api/compose/lora/scale | async | Set adapter influence (0.0–1.0) |
| GET | /api/compose/lora/status | async | Current adapter state |
| GET | /api/compose/lora/browse | sync | List adapters in loras/ directory |
| POST | /api/compose/train/upload | async | Upload audio files for training |
| POST | /api/compose/train/clear | async | Delete audio + tensor files |
| GET | /api/compose/train/pipeline-state | async | Disk state for recovery |
| POST | /api/compose/train/scan | async | Load audio into AceStep dataset |
| POST | /api/compose/train/label | async | Start async auto-labeling |
| GET | /api/compose/train/label/status | async | Poll auto-label progress |
| GET | /api/compose/train/samples | async | List dataset samples |
| PUT | /api/compose/train/sample/{idx} | async | Update sample metadata |
| POST | /api/compose/train/save | async | Save dataset to disk |
| POST | /api/compose/train/load | async | Load saved dataset |
| POST | /api/compose/train/preprocess | async | Start tensor preprocessing |
| GET | /api/compose/train/preprocess/status | async | Poll preprocessing progress |
| POST | /api/compose/train/start | async | Start LoRA/LoKR training |
| GET | /api/compose/train/status | async | Poll training status + loss |
| POST | /api/compose/train/stop | async | Stop training |
| POST | /api/compose/train/export | async | Export adapter to loras/ |
| POST | /api/compose/train/reinitialize | async | Reload generation model |
| GET | /api/compose/train/snapshots | async | List saved snapshots |
| POST | /api/compose/train/snapshots/save | async | Save dataset + tensors snapshot |
| POST | /api/compose/train/snapshots/load | async | Load snapshot |
| DELETE | /api/compose/train/snapshots/{name} | async | Delete snapshot |
| GET | /api/mix/tracks | sync | Current track list |
| POST | /api/mix/tracks | sync | Update track state |
| POST | /api/mix/render | job | Render mix to FLAC |
| POST | /api/mix/add-audio | sync | Add manual audio track |
| POST | /api/mix/add-midi | sync | Add manual MIDI track |
| DELETE | /api/mix/tracks/{id} | sync | Remove track |
| POST | /api/sfx/create | sync | Create SFX canvas |
| GET | /api/sfx | sync | List all SFX canvases |
| GET | /api/sfx/available-clips | sync | Clips grouped by session/saved/imported |
| POST | /api/sfx/upload-clip | sync | Import external audio clip |
| POST | /api/sfx/rename-clip | sync | Rename clip file + update manifests |
| GET | /api/sfx/{id} | sync | Get canvas manifest + rendered path |
| POST | /api/sfx/{id}/placements | sync | Add clip placement |
| PUT | /api/sfx/{id}/placements/{pid} | sync | Update placement |
| DELETE | /api/sfx/{id}/placements/{pid} | sync | Remove placement |
| PATCH | /api/sfx/{id} | sync | Update canvas (duration, limiter, align) |
| POST | /api/sfx/{id}/send-to-mix | sync | Render canvas + add as Mix track |
| DELETE | /api/sfx/{id} | sync | Delete canvas |
| GET | /api/sfx/{id}/stream | sync | Stream rendered canvas audio |
| GET | /api/sfx/{id}/reference-waveform | sync | Downsampled peaks for align ref |
| GET | /api/enhance/presets | sync | Available enhancement presets |
| GET | /api/enhance/stems | sync | Stems available for enhancement |
| POST | /api/enhance | job | Start enhancement (denoise/dereverb) |
| POST | /api/enhance/batch | job | Batch enhancement — same preset across multiple files |
| POST | /api/enhance/batch/save-all | sync | Zip batch enhancement results for download |
| GET | /api/enhance/autotune-options | sync | Available keys and scales for auto-tune |
| POST | /api/enhance/autotune | job | Start auto-tune (CREPE + WORLD/STFT pitch correction) |
| GET | /api/enhance/effects-options | sync | Effects chain schema (types, methods, params) |
| POST | /api/enhance/effects | job | Start effects chain (EQ, compressor, gate, stereo width) |
| POST | /api/export | job | Start export |
| POST | /api/export/download-zip | sync | Zip download |

---

## Frontend architecture

### Event bus pattern (replaces DPG callback wiring)

```
Separate done  → appState.emit("stemsReady", stemPaths)
MIDI done      → appState.emit("midiReady", {labels, noteCounts})
Generate done  → appState.emit("generateReady", audioPath)
Compose done   → appState.emit("composeReady", {path, title, metadata})
Mix done       → appState.emit("mixReady", mixPath)
Enhance done   → appState.emit("enhanceReady", {output_path, preset, label})
Autotune done  → appState.emit("enhanceReady", {output_path, preset: "autotune", label})
SFX ready      → appState.emit("sfxReady", {id})
File loaded    → appState.emit("fileLoaded", {path, filename})
Batch loaded   → appState.emit("batchFilesLoaded", uploadedFiles)
Batch toggled  → appState.emit("batchModeChanged", boolean)
```

Downstream components subscribe in their `init*()` functions:
- MIDI listens to `stemsReady` → populate stem checkboxes
- Mix listens to `stemsReady` + `generateReady` + `composeReady` + `sfxReady` → add/refresh tracks
- Generate listens to `stemsReady` + `midiReady` + `mixReady` + `fileLoaded` → populate conditioning/align sources
- Export listens to all (including `composeReady`) → enable artifact checkboxes
- Separate listens to `fileLoaded` + `batchFilesLoaded` + `batchModeChanged` → enable separation, toggle batch UI

### Now Playing transport bar (`audio-player.js`)

The global transport bar at the bottom of every tab shows what audio is currently
playing and provides play/pause/stop/seek controls that work regardless of which
tab the user is on. **Every play button in every tab must call `transportLoad()`**
so the bar always reflects the active audio. Stop buttons and finish events must
call `transportStop()`.

**Contract — must be maintained across all components:**

| Component | Play → `transportLoad(url, label, false, source)` | Stop/Finish → `transportStop()` |
|---|---|---|
| `loader.js` | Upload preview | — |
| `separate.js` | Stem cards, batch cards | Stop btn, finish event |
| `enhance.js` | Result card play btn + auto-load on job done | Stop btn, finish event |
| `midi.js` | MIDI render play | Stop btn, finish event |
| `generate.js` | Synth result cards | Stop btn, finish event |
| `compose.js` | Compose result play, Voice result play | Stop btn, finish event |
| `mix.js` | Track play, MIDI track play, Play All, Master Mix | Stop btn, finish, stopPreview |

The `source` parameter (e.g. `'Separate'`, `'Mix'`, `'Enhance › Tune'`) is shown
as "Now Playing (source): label" so the user always knows which tab produced the
audio. When adding new playable components, wire them to the transport bar
following this same pattern.

### Job polling

Long-running pipeline jobs use `pollJob(jobId, {onProgress, onDone, onError, interval})` with 10s default interval.

---

## Model registry (`models/registry.py`)

Frozen `ModelSpec` subclasses describe every model variant. Spec types:

| Spec class | Models | Pipeline |
|---|---|---|
| `DemucsSpec` | htdemucs, htdemucs_ft, mdx_extra, mdx_extra_q | `DemucsPipeline` |
| `RoformerSpec` | roformer-viperx-vocals, roformer-kj-vocals, roformer-zfturbo-4stem, roformer-jarredou-6stem, + 2 more | `RoformerPipeline` |
| `BasicPitchSpec` | basicpitch | `BasicPitchPipeline` |
| `DrumMidiSpec` | adtof-drums | `MidiPipeline` (drum routing branch) |
| `WhisperSpec` | whisper-tiny, whisper-small, whisper-large-v3 | `VocalMidiPipeline`, `TranscribePipeline` |
| `StableAudioSpec` | stable-audio-open-1.0 | `MusicGenPipeline` |

Public API: `get_spec()`, `list_specs()`, `get_loader_kwargs()`, `get_pipeline_defaults()`, `get_gui_metadata()`.

---

## Pipeline interface (all pipelines follow this contract)

```python
pipeline.configure(config)   # supply Config dataclass
pipeline.load_model()        # load weights — raises ModelLoadError
result = pipeline.run(input) # run inference — raises PipelineExecutionError / InvalidInputError
pipeline.clear()             # release GPU memory
```

---

## Exception hierarchy (`utils/errors.py`)

```
StemForgeError
├── ModelLoadError(model_name=)            — weight loading / download failures
├── AudioProcessingError(path=)            — read / write / resample failures
├── PipelineExecutionError(pipeline_name=) — runtime inference failures
└── InvalidInputError(field=)             — pre-processing validation failures
```

---

## Output directories (`utils/paths.py`)

| Constant | Path |
|---|---|
| `STEMS_DIR` | `~/.local/share/stemforge/output/stems/` |
| `MIDI_DIR` | `~/Music/StemForge/` |
| `MUSICGEN_DIR` | `~/.local/share/stemforge/output/musicgen/` |
| `COMPOSE_DIR` | `~/.local/share/stemforge/output/compose/` |
| `MIX_DIR` | `~/.local/share/stemforge/output/mix/` |
| `SFX_DIR` | `~/.local/share/stemforge/output/sfx/` |
| `ENHANCE_DIR` | `~/.local/share/stemforge/output/enhance/` |
| `EXPORT_DIR` | `~/.local/share/stemforge/output/exports/` |

---

## AceStep subprocess

AceStep runs as a separate process, spawned lazily on first use (`POST
/api/compose/start`) via `acestep_state.launch()`:

- **Port:** 8001 (configurable via `--acestep-port` or `ACESTEP_PORT`); launch()
  pre-checks the port and reports a clear error if something already holds it
- **Model loading:** `ACESTEP_NO_INIT=false` is set at launch (overridable via env)
  — upstream defaults to lazy loading, which deadlocks the "running" status gate
- **Disable:** `--no-acestep` flag — Compose tab shows disabled state
- **GPU:** `--gpu N` sets `CUDA_VISIBLE_DEVICES` (and `HIP_VISIBLE_DEVICES` for
  ROCm) on the AceStep subprocess only
- **Deterministic:** `--deterministic` flag — sets near-greedy LM temperature (0.01) when seed is set + CUDA deterministic ops on AceStep subprocess. Useful for A/B testing LoRA vs base model.
- **State tracking:** `backend/services/acestep_state.py` — thread-safe status: disabled/starting/running/crashed
- **Graceful degradation:** StemForge stays alive if AceStep crashes. All other tabs work normally.
- **Compose router:** `backend/api/compose.py` proxies requests to AceStep's API via `backend/api/acestep_wrapper.py` — includes generation, LoRA management (6 endpoints), and training pipeline (20+ endpoints)
- **LoRA directory:** `Ace-Step-Wrangler/loras/` (configurable via `LORA_DIR` env var) — scanned for PEFT dirs and .safetensors files
- **Training directory:** `Ace-Step-Wrangler/training/` (configurable via `TRAIN_DIR` env var) — audio, tensors, output, snapshots subdirs
- **Submodule:** `Ace-Step-Wrangler/` (with nested `vendor/ACE-Step-1.5/`). To pull upstream changes: `cd Ace-Step-Wrangler && git pull origin main && cd .. && git add Ace-Step-Wrangler && git commit` — Dependabot also PRs pointer updates weekly

**Tab bar:** Separate · Enhance · MIDI · Synth · Compose · Mix · Export

---

## Platform notes

- **Linux (primary)**: CUDA 13.0 wheels, uv sync, Python 3.12
- **macOS (Apple Silicon)**: MPS acceleration via `pyproject.toml.MAC`; use `from utils.device import get_device`, never hardcode `"cuda"`
- **AMD (ROCm)**: `pyproject.toml.ROCM` — rocm7.1 torch index, same pins; resolution-verified but untested on hardware (issue #11)
  - ROCm's torch presents as CUDA (`torch.cuda.is_available()` is true, HIP translates), so hardcoded `"cuda"` is *not* a ROCm bug — the `get_device` rule exists for MPS
  - Installing overwrites `pyproject.toml` and re-resolves `uv.lock` (the committed lock is cu130); both show dirty afterwards and must not be committed
  - AceStep needs no separate ROCm setup: `acestep_state.launch()` spawns it with `sys.executable`, so it inherits this venv's torch. Its cu128 pins are neutralized by `override-dependencies` + `[tool.uv.sources]` in the ROCM pyproject
  - CPU-bound in this variant: UVR enhance (onnxruntime) and faster-whisper (ctranslate2). AceStep's LM is *not* an AMD regression — the prebuilt flash-attn wheels are marker-pinned to Python 3.11 and the project requires 3.12, so every platform (CUDA included) already runs the SDPA path
  - `Ace-Step-Wrangler` **standalone** is CUDA-only (its own pyproject pins cu128 indexes + a prebuilt CUDA flash-attn wheel); ROCm users must run AceStep through StemForge
- **FluidSynth**: Required for MIDI preview and Mix tab; GM soundfont auto-discovered
- **CI**: GitHub Actions (`.github/workflows/ci.yml`) — lock drift check, ROCm resolve-only check, and a CPU test job on every PR/push; Dependabot updates action versions and submodule pointers weekly

---

## Caches and logs

- Model weights: `~/.cache/stemforge/` (subdirs per model type) — override with `MODEL_LOCATION` env var or `--model-dir` flag
- Cache resolution: `utils/cache.py` → `get_model_cache_base()` / `get_model_cache_dir(subdir)`
- AceStep checkpoints: also reads `MODEL_LOCATION` (forwarded via `_PASSTHROUGH_VARS` in `acestep_state.py`)
- Logs: `~/.local/share/stemforge/logs/stemforge.log`
