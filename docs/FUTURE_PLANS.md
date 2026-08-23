# Future Plans

Planned features and research directions for StemForge. Items are roughly
ordered by priority within each section. Nothing here is committed — this
is a living document for tracking ideas.

---

## Voice transformation — IMPLEMENTED

Voice conversion is now live in the **Compose tab** as a 5th mode
(**Create | Rework | Lego | Complete | Voice**), powered by RVC
(Retrieval-based Voice Conversion) via vendored Applio inference code.

### What shipped

- **RVC pipeline** (`pipelines/rvc_pipeline.py`) wrapping Applio's
  VoiceConverter — audio-in → audio-out, preserves lyrics, timing,
  and pitch contour of the source.
- **Vendored Applio** (`vendor/rvc/`) — inference-only subtree (no
  Gradio, no training code). MIT-licensed.
- **14 built-in voice models** auto-downloaded from HuggingFace on
  first use: Freddie Mercury, Adele, Frank Sinatra, Kurt Cobain,
  Ariana Grande, Taylor Swift, The Weeknd, Drake, Hatsune Miku,
  Donald Trump, SpongeBob, Peter Griffin, plus generic Male and Female.
- **Voice model browser** — search HuggingFace for RVC models by name,
  download with one click. Also supports uploading local .pth/.index files.
- **Controls**: pitch shift (-24 to +24 semitones), F0 method
  (RMVPE/CREPE/FCPE), voice character (index rate), consonant protection.
- **Source audio**: select from separated stems or load any audio file.
- **Cross-tab integration**: results auto-appear in Mix and Export tabs
  via the `transformReady` event. Transport bar picks up playback.
- **Backend**: `POST /api/voice/convert`, `GET /api/voice/models`,
  `POST /api/voice/models/import`, `POST /api/voice/models/upload`,
  `DELETE /api/voice/models/{name}`, `GET /api/voice/models/search`.
- Voice models cached at `~/.cache/stemforge/voice_models/`.

### Future improvements

- **Chatterbox / emotion control** — evaluate Resemble AI's voice
  conversion mode for emotion exaggeration (monotone → dramatic).
  Would complement RVC's identity-focused conversion.
- **GPT-SoVITS** — evaluate for cross-lingual singing voice conversion
  and cases where richer prosody control is needed.
- **Voice model training** — see dedicated section below.
- **Batch processing** — convert multiple stems through the same
  voice model in one operation.
- **Ethical safeguards** — watermarking, consent notices, or usage
  warnings when converting to a cloned voice.

---

## RVC voice model training

Train custom voice models from audio samples directly within StemForge.
The vendored Applio code already includes the model architectures
(Synthesizer, MultiPeriodDiscriminator) needed for training — only the
training loop, preprocessing, and feature extraction code needs to be
added. Evaluated as ~15–22 hours of implementation effort.

### Dependency impact

Three new packages, no conflicts with the existing venv:

- `tensorboard` — training loss and spectrogram logging
- `matplotlib` — spectrogram rendering for TensorBoard
- `scikit-learn` — MiniBatchKMeans for FAISS index size reduction

All other training deps (torch.distributed, noisereduce, torchcrepe,
faiss-cpu, transformers, librosa, scipy, soxr) are already present.

### Training workflow (4 steps)

1. **Preprocess** — load user's audio files (~2–10 min of voice),
   high-pass filter + optional denoise, slice into segments on silence
   boundaries, write full-SR WAVs + 16 kHz WAVs.
2. **Extract** — run F0 pitch extraction (RMVPE/CREPE/FCPE) on each
   segment, run HuBERT/ContentVec speaker embedding extraction,
   generate training config JSON + filelist.
3. **Train** — GAN training loop: Synthesizer (generator) vs
   MultiPeriodDiscriminator with mel-spectrogram, KL divergence,
   adversarial, and feature-matching losses. Saves periodic checkpoints
   + inference-ready `.pth`. Optional overtraining detection via
   smoothed EMA loss. Supports bf16 on Ampere+ GPUs.
4. **Index** — concatenate all extracted embeddings, optionally
   KMeans-reduce (for large datasets >200 k frames), build FAISS
   IVFFlat index, save as `.index` file.

**Output**: `{name}.pth` + `{name}.index` — the same format our
inference pipeline already consumes, ready to use in Voice mode.

### Code to vendor

~3,500 lines across 13 Python files + 4 JSON configs from Applio's
`rvc/train/` subtree. Key files:

| File | Purpose |
|------|---------|
| `train/train.py` | GAN training loop (~1,160 lines) |
| `train/data_utils.py` | Dataset, collate, bucket sampler |
| `train/losses.py` | Adversarial, feature-matching, KL losses |
| `train/mel_processing.py` | Spectrogram / mel computation |
| `train/preprocess/preprocess.py` | Audio slicing + normalization |
| `train/extract/extract.py` | F0 + embedding extraction |
| `train/process/extract_model.py` | Strip checkpoint to inference `.pth` |
| `train/process/extract_index.py` | Build FAISS `.index` |

All training code imports from `lib/algorithm/` (Synthesizer,
Discriminator) which is already vendored. Main refactoring work is
replacing Applio's `os.getcwd()` + sys.path hacks with StemForge
import conventions.

### GPU / VRAM requirements

| Config | VRAM | Notes |
|--------|------|-------|
| Minimum viable | 4 GB | batch_size=2, very slow |
| Practical | 6–8 GB | batch_size=4–8 |
| RTX 5080 (16 GB) | 16 GB | batch_size=8–16 + bf16 + GPU caching |

Training locks the GPU for minutes-to-hours. Should run as a managed
subprocess (like AceStep) with the pipeline_manager GPU lock ensuring
mutual exclusion with inference pipelines.

### Integration sketch

- **Backend**: ~6 new API endpoints (preprocess, extract, train, stop,
  status, build-index) under `/api/voice/train/*`
- **Frontend**: training panel in Voice mode — dataset upload/selection,
  model name, sample rate, epoch count, batch size, progress bars with
  loss curves
- **Pretrained models**: auto-download base G/D weights from HuggingFace
  (~200–400 MB per sample rate variant)

---

## AceStep LoRA / LoKR training — IMPLEMENTED

Custom adapter training is now live in the **Compose tab** as a 6th mode
(**Create | Rework | Lego | Complete | Voice | Train**).

### What shipped

- **Full training pipeline** in Compose tab Train mode: Upload → Scan →
  Auto-label → Preprocess → Train → Export.
- **Two adapter types**: LoRA (general purpose) and LoKR (compact).
- **Configurable hyperparameters**: rank, epochs, learning rate, batch size,
  warmup steps, gradient accumulation, save interval.
- **Live loss chart** with canvas 2D rendering and HiDPI support.
- **Named snapshots** — save/load/delete dataset + preprocessed tensors
  for iterating on training without re-running the pipeline.
- **Adapter export** to `loras/` directory for immediate use via the
  LoRA browser in generation modes.
- **Model reinitialization** after training to pick up new adapters.
- **Pipeline state recovery** — switching to Train mode checks disk state
  and in-progress tasks to resume where you left off.

### Also shipped (Compose tab additions)

- **LoRA adapter management** — browse, load, unload, scale (0–100%)
  adapters during generation. Post-generation warning if adapter is
  silently dropped.
- **Seed controls** — Last / Random buttons for reproducible generation.
- **Project save/load** — full Compose state serialized to JSON (~30 fields).

### Future improvements

- **Multi-GPU training** — currently single-GPU only.
- **Training presets** — save/recall hyperparameter configurations.
- **Adapter comparison** — A/B generation with different adapters loaded.
  Use `--deterministic` flag + fixed seed for reproducible A/B comparisons
  between base model and LoRA-adapted output.

---

## Audio enhancement — remaining phases

Phases 1, 2, and 3 are shipped. The Enhance tab has a three-mode bar:
**Clean Up** (Phase 1) · **Tune** (Phase 3) · **Effects** (Phase 2).

### Phase 1 — UVR Clean Up — IMPLEMENTED

8 curated presets (denoise, dereverb, debleed) via vendored
`python-audio-separator` fork across Roformer/MDXC/VR architectures.
Batch mode supported.

### Phase 3 — Auto-Tune — IMPLEMENTED

Pitch correction for vocal stems using CREPE neural pitch detection
(`torchcrepe`) with two user-selectable resynthesis methods:

- **WORLD Vocoder** (`pyworld`, MIT + Modified-BSD) — decomposes audio into
  F0, spectral envelope, and aperiodicity; modifies F0 and resynthesises.
  Formant-preserving by design. CPU-only. Best on lossless audio (WAV/FLAC).
- **Phase Vocoder (STFT)** (`stftpitchshift`, MIT) — spectral-domain pitch
  shifting with cepstral formant preservation. CPU-only. More robust on
  compressed audio (MP3/OGG) than WORLD.

Controls: key, scale (chromatic/major/minor/pentatonic/blues), correction
strength, humanization, and synthesis method dropdown.

#### Planned: Neural Vocoder (GPU) — third synthesis method

Add a GPU-accelerated neural vocoder as a third option in the method dropdown
for higher-fidelity resynthesis, especially on compressed or noisy audio where
WORLD and STFT show artifacts.

**Best candidate: SiFi-GAN**
- **License**: MIT — compatible with Apache 2.0
- **Architecture**: F0-conditioned source-filter neural vocoder. Takes F0
  contour + mel spectrogram as input, generates waveform. The F0 conditioning
  makes it a natural fit: feed CREPE's corrected F0 directly, no ratio mapping.
- **Pretrained**: 24 kHz model available. Would need to resample input down
  from 44.1 kHz, run inference, resample back up.
- **Repo**: `https://github.com/chomeyama/SiFiGAN`
- **Quality**: Produces natural-sounding speech/singing with fewer artifacts
  than traditional vocoders on degraded input, because the neural network
  learns to reconstruct clean waveforms from spectral features.

**Integration plan:**
1. Add `sifi-gan` as third entry in `AUTOTUNE_METHODS` tuple
2. New `utils/sifigan_shift.py` — download pretrained checkpoint on first use
   (via `huggingface_hub`), resample to 24 kHz, extract mel spectrogram,
   run SiFi-GAN inference on GPU, resample result back to original SR
3. Model weights cached at `~/.cache/stemforge/sifigan/` (~50 MB)
4. Requires GPU — method greyed out in dropdown when `torch.cuda.is_available()`
   is False, with tooltip explaining GPU requirement
5. Pipeline manager GPU lock ensures mutual exclusion with other GPU pipelines

**Dependencies to add:**
- `parallel-wavegan` (MIT) — contains SiFi-GAN model definitions and
  pretrained checkpoint loading utilities
- Or vendor the ~500-line model definition directly to avoid the full
  parallel-wavegan dependency tree

**Performance estimate (RTX 5080):**
- 3-minute vocal at 24 kHz ≈ 4.3 M samples
- SiFi-GAN inference: ~2–5 seconds (real-time factor ~30–50x on modern GPU)
- Resampling overhead: negligible (~100 ms each way via soxr)

### Phase 4 — Region Edit — planned

Manual region-based volume editing for cleaning up stems before voice
conversion or mixing. New **Edit** mode in the Enhance tab mode bar
(Clean Up · Tune · Edit · Effects).

- **Region selection** — wavesurfer.js Regions plugin for click-drag
  selection on the waveform. Multiple independent regions per stem.
- **Per-region controls** — inline popup with volume slider (0–100%,
  where 0% = silence), fade-in/out duration (ms), apply/delete buttons.
- **Apply All** — backend renders the modified audio (NumPy gain
  multiplication with cosine fade at region edges), emits `enhanceReady`
  for cross-tab integration.
- **Use case** — silence AceStep pre-lyrics vocalizations and other
  artifacts that don't voice-swap well, attenuate bleed in specific
  sections, manual cleanup that automated presets can't target.

### Phase 2 — Effects Chain — IMPLEMENTED

Per-stem channel strip with four effect types, each offering DSP and/or ML
methods:

- **3-band Parametric EQ** — low shelf, mid peak, high shelf with
  frequency/gain/Q via `scipy.signal`.
- **Compressor** — DSP feed-forward compressor and LA-2A neural
  optical compressor emulation (`vendor/micro_tcn`, Apache 2.0).
- **Noise Gate** — DSP spectral gating (`torchgating`) and Spectral
  method.
- **Stereo Width** — Mid/Side processing to narrow or widen the
  stereo image.

#### Future additions

- **Convolution reverb** — `scipy.signal.fftconvolve` with bundled
  impulse responses (IR files). Dry/wet mix control. Optionally load
  custom IR WAVs.
- **Delay** — circular buffer with feedback, mix, and tempo-sync
  option. Simple numpy array indexing.
- **Chorus** — modulated delay line with LFO (sine/triangle),
  depth, rate, and mix controls.
- **Draggable effect panels** — slidable panels for reordering effects in
  the chain via pointer events.

---

## SFX Stem Builder improvements

- **Draggable clip placement** — replace the current click-to-position
  workflow with direct drag-along-the-timeline via pointer events. Clips
  slide smoothly to new positions with visual snap feedback. Same
  interaction pattern as the Effects chain panels — pointer events
  with manual hit-testing for smooth sub-pixel control.
- **Drag-to-resize** — grab clip edges to adjust fade in/out duration
  visually on the timeline.

---

## Other ideas

_Add future feature ideas below this line._

### Native packaging
- RPM packages for Fedora/RHEL
- MSI installer for Windows
- .dmg for macOS
- Would make StemForge accessible to non-developers

### Batch processing (partially implemented)
- Batch stem separation is live — extract one stem type from multiple files
- Batch enhancement is live — apply one preset to multiple files
- Extend to other pipelines: batch MIDI extraction
- Useful for albums or sample libraries

### DAW integration
- Export stems + MIDI in a format that opens directly as a DAW project
  (e.g., Reaper project file, Ableton Live Set via ALS XML)
- Alternatively, a VST plugin wrapper for real-time stem separation
- Hardware MIDI output is already live (MIDI tab → MIDI Out, Web MIDI) —
  see `docs/WEB_MIDI_OUT_SPEC.md`; the items below are its deferred follow-ups

### Web MIDI follow-ups (deferred from the MIDI Out feature)
- **MIDI input** — record a hardware performance into StemForge as a new
  MIDI track, or as an alternative to BasicPitch extraction. The Web MIDI
  access plumbing in `frontend/components/webmidi.js` is reusable; the
  harder part is the return leg, since StemForge has no audio capture path.
  Needs its own spec.
- **Transport sync** — MIDI Clock / MTC / MMC so hardware follows the
  transport bar (meaningful only once there is a unified timeline).
- **Pitch bend and CC** — carry `PrettyMIDI.pitch_bends` and
  `control_changes` through `/api/midi/events`; waiting on a pipeline that
  actually produces them. The response shape has room for `bend`/`cc`
  event types without a breaking change.
- **Multi-port merged playback** — route each instrument of the merged
  MIDI to a different port/channel from a single card; the `tracks[]`
  response shape already supports it.
- **Seek / start-from-playhead** — a `startSec` argument to `play()` plus
  a rule for notes already sounding at the seek point.

### Improved audio generation
- Evaluate newer open-source generation models as they emerge
  (successors to Stable Audio Open, music-focused diffusion models)
- Explore LoRA fine-tuning of the generation model on specific genres
  or instruments for higher-quality, more controllable output
