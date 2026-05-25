Purpose & context
Tsondo is the developer and owner of StemForge, a source-available, GPU-accelerated local web application for AI-powered music production. The project chains multiple ML pipelines into a single workflow covering stem separation, MIDI extraction, audio generation, song composition, mixing, and export. It is hosted at github.com/tsondo/StemForge and has reached a v1.0.0 release with a substantially developed codebase.
Pipeline stack:

Demucs – stem separation (4 models)
BS-Roformer – stem separation (6 models including 4-stem and 6-stem)
BasicPitch / Vocal MIDI – MIDI extraction (instruments / vocals)
Stable Audio Open ("Synth" tab) – sound/texture generation
AceStep / ACE-Step-Wrangler ("Compose" tab) – AI music composition (Create, Rework, Lego, Complete, Voice, Train modes)
UVR / audio-separator ("Enhance" tab, Clean Up) – denoise/dereverb (8 presets)
Auto-Tune ("Enhance" tab, Tune) – CREPE pitch detection + Praat PSOLA resynthesis
RVC / Applio ("Compose" tab, Voice) – AI voice conversion (14 built-in voices + HuggingFace search)

Architecture: FastAPI backend ("thick backend" — pipelines run in-process) with vanilla HTML/CSS/JS frontend. This was migrated from a DearPyGUI desktop application. The frontend follows a dark DAW aesthetic with tab-based navigation (Separate → Enhance → MIDI → Synth → Compose → Mix → Export). ACE-Step-Wrangler is integrated as a git submodule in a vendor directory.
Design values: Clean architectural boundaries, graceful error handling, pragmatic technology choices, thorough documentation discipline.

Current state

StemForge v1.0.0 is published and actively developed post-release. All core pipelines are functional and tested.

**What's working:**
- Demucs + BS-Roformer stem separation (10 models total), batch mode, auto engine recommendation
- MIDI extraction for instruments (BasicPitch) and vocals (faster-whisper + PYIN)
- **Lyrics transcription (MIDI tab)** — Notes/Lyrics mode bar; engine selection between Whisper (tiny → large-v3) and Qwen2-Audio-7B-Instruct; outputs .txt/.lrc/.srt; `Send to Compose` integration with the AI lyrics workflow.
- Enhance tab with three-mode bar: Clean Up (8 UVR presets), Tune (CREPE + WORLD/STFT auto-tune), Effects (EQ, Compressor with DSP/LA-2A neural, Noise Gate with DSP/Spectral, Stereo Width)
- Stable Audio Open generation (Synth tab) — text + audio + MIDI conditioning, chunked to 600s
- SFX Stem Builder — DAW timeline with clip placement, fades, align-to reference
- AceStep composition (Compose tab) — 6 modes: Create, Rework, Lego, Complete, Voice, Train
- Compose result cards accumulate (newest first) with close buttons for dismissal
- `--deterministic` CLI flag for reproducible generation (near-greedy LM temperature + CUDA deterministic ops when seed is set)
- LoRA/LoKR adapter training pipeline with live loss chart, snapshots, export (correct field names per adapter type)
- LoRA browse detects nested `adapter/` subdirectory structure from training exports
- RVC voice conversion with 14 built-in voices + HuggingFace model search
- Mix tab with multi-track preview, per-track volume, FLAC render
- Export panel with 6 formats (WAV, FLAC, AIFF, MP3, OGG Opus, M4A), sample rate conversion (22050–96000 Hz), bit depth selection (16/24/32-bit for lossless), bitrate control (lossy), standalone file loader for format conversion, per-artifact source info badges, zip download
- Global transport bar with "Now Playing (source)" context labels
- Upload supports audio + video files (FFmpeg extraction)

**Integration:**
- Cross-tab event bus: all pipelines feed into Mix, Export, and each other
- MODEL_LOCATION environment variable shares model checkpoints across installations
- HuggingFace token: `huggingface-cli login` or `HF_TOKEN` in .env

Known issues / technical debt:

- Inadequate automated testing coverage.
- Documented but unsolved GPU contention between AceStep (subprocess) and in-process pipelines.
- Vanilla JS frontend is substantial (~8K lines across components) but well-organized with event bus pattern; evaluated React adoption and determined it's not yet justified.
- Potential scope creep from numerous planned features.


On the horizon

Effects Chain improvements: Convolution reverb (IR loading), delay, and chorus effects to complement the shipped EQ/compressor/gate/stereo-width chain.
RVC voice model training: Train custom voice models from audio samples within StemForge. Architecture evaluated, ~15–22 hours implementation. See FUTURE_PLANS.md.
DAW connectivity: Options ranging from drag-and-drop export to REST bridge plugins.
macOS support: MPS acceleration works via `pyproject.toml.MAC`. Further polish needed.
Native installable packages (RPM for Linux, MSI for Windows) — explored but not yet implemented.


Key learnings & principles

Thick backend vs. thin relay: StemForge requires in-process pipeline execution (thick backend), unlike ACE-Step-Wrangler's thin relay proxy pattern. This distinction is foundational to the architecture.
Stable Audio Open is a texture/sound generator, not a music generator — best used for ambient layers and sound effects, not full compositions.
Nested git submodules work well for this project structure; path dependencies in pyproject.toml mirror how Wrangler handles ACE-Step.
Non-destructive editing with JSON manifests is the right approach for the sound effects stem builder, enabling later adjustment without re-rendering.
Sequential pipelines don't benefit from multi-GPU load balancing in a single-user desktop context.
GitHub search indexing ≠ ranking: Zero-star/zero-fork repos with minimal READMEs rank poorly regardless of indexing; rich metadata and external mentions are the practical fix.
torchao is PyTorch's Architecture Optimization library for quantization/low-precision inference — a transitive dependency via AceStep, not needed directly.


Approach & patterns

Claude Code CLI is used for implementation tasks; Tsondo drafts detailed prompts/specs for Claude Code to execute, then refines UI interactively afterward.
Prefers complete, ready-to-use deliverables (full files, full specs) over partial suggestions.
Provides direct corrections when an approach falls short (e.g., supplying GitHub URLs directly when search failed).
Follows established codebase patterns (CLAUDE.md conventions, existing router/job patterns) when adding new features.
Backend specs are completed and tested via API before frontend UI work begins.
Documentation is maintained rigorously: CLAUDE.md, README.md, INSTRUCTIONS.md, FUTURE_PLANS.md are all actively kept current.
Job submission + polling pattern for long-running ML tasks in the FastAPI backend.


Tools & resources

Backend: Python 3.11, FastAPI, numpy, PyTorch (CUDA 13.0 / MPS), huggingface_hub, uv
ML models: Demucs, BS-Roformer, BasicPitch, Stable Audio Open, AceStep 1.5, torchcrepe, python-audio-separator (vendored), Applio/RVC (vendored)
Frontend: Vanilla HTML/CSS/JS, fetch() + polling, wavesurfer.js (CDN)
Dev tooling: Claude Code CLI, Git submodules
Dependency management: pyproject.toml with path dependencies for submodules (ace-step, nano-vllm)
Version control / hosting: GitHub (github.com/tsondo/StemForge)
