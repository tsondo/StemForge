# Acknowledgments

StemForge is built on the shoulders of many outstanding open-source projects.
We are grateful to every team listed below for making their work freely available.

---

## Demucs — Meta (Facebook AI Research)

Hybrid Transformer source separation powering the Separate tab (htdemucs, htdemucs_ft, mdx_extra, mdx_extra_q).

- **Repository:** https://github.com/facebookresearch/demucs
- **Paper:** Rouard, Massa & Défossez — *Hybrid Transformers for Music Source Separation* (ICASSP 2023)
- **License:** MIT

---

## BS-Roformer / MelBand-Roformer — Community

High-quality separation models used alongside Demucs in the Separate tab.

### `bs-roformer` Python package — Lucidrains

- **Repository:** https://github.com/lucidrains/BS-RoFormer

### ViperX vocal model (SDR 12.97) — TRvlvr / UVR community

- **Model repository:** https://github.com/TRvlvr/model_repo

### ZFTurbo 4-stem BS-Roformer & Music-Source-Separation-Training — Roman Solovyev (ZFTurbo)

- **Repository:** https://github.com/ZFTurbo/Music-Source-Separation-Training
- **Paper:** Solovyev et al. — *Benchmarks and leaderboards for sound demixing tasks* (2023)

### jarredou 6-stem BS-Roformer (guitar + piano)

- **Model repository:** https://huggingface.co/jarredou/BS-ROFO-SW-Fixed

---

## Basic Pitch — Spotify

Polyphonic audio-to-MIDI transcription for instrument stems in the MIDI tab.

- **Repository:** https://github.com/spotify/basic-pitch
- **Paper:** Bittner et al. — *A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation* (ICASSP 2022)
- **License:** Apache 2.0

---

## ADTOF — Mickaël Zehren, Marco Alunno & Paolo Bientinesi

Automatic drum transcription powering the drum path in the MIDI tab
(kick, snare, tom, hi-hat, cymbal → GM channel 10).

- **Repository:** https://github.com/MZehren/ADTOF
- **Papers:** Zehren, Alunno & Bientinesi — *ADTOF: A large dataset of non-synthetic music for automatic drum transcription* (ISMIR 2021); *High-quality and reproducible automatic drum transcription from crowdsourced data* (2023)
- **License:** CC BY-NC-SA 4.0 (code, dataset, and pretrained models — see THIRD-PARTY-NOTICES.md)

### ADTOF-pytorch port — xavriley

PyTorch port with bundled weights converted from the official release, used by
StemForge via pip dependency.

- **Repository:** https://github.com/xavriley/ADTOF-pytorch

---

## Whisper — OpenAI

Speech recognition model used (via faster-whisper) for vocal pitch-to-MIDI extraction.

- **Repository:** https://github.com/openai/whisper
- **Paper:** Radford et al. — *Robust Speech Recognition via Large-Scale Weak Supervision* (2022)
- **License:** MIT

---

## faster-whisper — SYSTRAN

CTranslate2-accelerated Whisper inference powering the Vocal MIDI pipeline.

- **Repository:** https://github.com/SYSTRAN/faster-whisper
- **License:** MIT

---

## Qwen3-ASR — Alibaba Cloud (Tongyi Lab)

Multilingual automatic speech recognition model used as the GPU-backed
engine in the Lyrics transcription feature on the MIDI tab. Trained
specifically for speech and singing voice recognition across 52 languages.

- **Model:** https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- **Paper:** Shi et al. — *Qwen3-ASR Technical Report* (arXiv:2601.21337, 2026)
- **License:** Apache 2.0 (see `licenses/LICENSE-Qwen3-ASR`)
- **Toolkit:** https://github.com/QwenLM/Qwen3-ASR

---

## Stable Audio Open — Stability AI

Text-conditioned audio generation model powering the Synth tab.

- **Repository:** https://huggingface.co/stabilityai/stable-audio-open-1.0
- **Paper:** Evans et al. — *Stable Audio Open* (2024)
- **License:** Stability AI Community License

---

## ACE-Step — ACE Studio / Timedomain

Full song generation from lyrics and style descriptions, powering the Compose tab.

- **Repository:** https://github.com/AceStudioAI/ACE-Step
- **Paper:** *ACE-Step: A Step Towards Music Generation Foundation Model* (2025)
- **License:** Apache 2.0

---

## Applio / RVC — IAHispano & RVC-Project

Retrieval-based Voice Conversion inference code (vendored) powering the Voice mode in the Compose tab. StemForge vendors Applio's inference-only subtree for audio-in → audio-out voice transformation.

- **Applio repository:** https://github.com/IAHispano/Applio
- **RVC project:** https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- **License:** MIT

### RMVPE — lj1995

Robust pitch estimation model used as the default F0 extraction method for voice conversion.

- **Repository:** https://github.com/Dream-High/RMVPE
- **Paper:** Wei et al. — *RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music* (2023)

### FAISS — Meta (Facebook AI Research)

Similarity search library used for speaker embedding retrieval in the RVC pipeline.

- **Repository:** https://github.com/facebookresearch/faiss
- **License:** MIT

### ContentVec — auspicious3000

Self-supervised speech representation model used as the speaker embedding extractor in RVC.

- **Repository:** https://github.com/auspicious3000/contentvec
- **Paper:** Qian et al. — *ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers* (ICML 2022)

---

## music21 — MIT / Michael Scott Asato Cuthbert

Music analysis and notation toolkit powering MIDI cleanup, key detection, transposition, and sheet music export in the MIDI tab.

- **Repository:** https://github.com/cuthbertLab/music21
- **Paper:** Cuthbert & Ariza — *music21: A Toolkit for Computer-Aided Musicology* (2010)
- **License:** BSD 3-Clause

---

## OpenSheetMusicDisplay (OSMD)

Browser-based MusicXML rendering (via VexFlow) for in-app sheet music preview.

- **Repository:** https://github.com/opensheetmusicdisplay/opensheetmusicdisplay
- **License:** MIT

---

## LilyPond (optional)

Music engraving program used for PDF sheet music export via subprocess.

- **Website:** https://lilypond.org
- **License:** GPL 3.0 (external binary, not bundled)

---

## PyTorch — Meta (Facebook AI Research)

Deep learning framework underlying all inference pipelines.

- **Repository:** https://github.com/pytorch/pytorch
- **License:** BSD-3-Clause

---

## Hugging Face Diffusers

Diffusion pipeline framework used to load and run Stable Audio Open.

- **Repository:** https://github.com/huggingface/diffusers
- **License:** Apache 2.0

---

## Hugging Face Transformers

Tokenizer and model infrastructure used by the generation pipelines.

- **Repository:** https://github.com/huggingface/transformers
- **License:** Apache 2.0

---

## librosa

Audio analysis and feature extraction used in the audio profiler and resampling utilities.

- **Repository:** https://github.com/librosa/librosa
- **Paper:** McFee et al. — *librosa: Audio and Music Signal Analysis in Python* (SciPy 2015)
- **License:** ISC

---

## FluidSynth

Software synthesizer used for MIDI preview rendering and Mix tab audio.

- **Repository:** https://github.com/FluidSynth/fluidsynth
- **License:** LGPL-2.1

---

## wavesurfer.js — katspaugh

Waveform visualization in the browser, used for all audio players and the global transport bar.

- **Repository:** https://github.com/katspaugh/wavesurfer.js
- **License:** BSD-3-Clause

---

## FastAPI — Sebastián Ramírez (tiangolo)

Web framework powering the StemForge backend API.

- **Repository:** https://github.com/fastapi/fastapi
- **License:** MIT

---

## Uvicorn — Encode

ASGI server running the FastAPI application.

- **Repository:** https://github.com/encode/uvicorn
- **License:** BSD-3-Clause

---

## uv — Astral

Blazing-fast Python package manager and resolver used for deterministic environments.

- **Repository:** https://github.com/astral-sh/uv
- **License:** MIT / Apache 2.0

---

## Additional dependencies

StemForge also relies on many other excellent open-source libraries including
NumPy, SciPy, soundfile, mido, pretty_midi, einops, safetensors, accelerate,
pydub, soxr, ai-edge-litert (TFLite runtime), torchcrepe, torchfcpe,
noisereduce and stftpitchshift. Thank you to all their
maintainers and contributors.

---

If you believe your project should be listed here and is not, please
[open an issue](https://github.com/tsondo/StemForge/issues) and we will add it.
