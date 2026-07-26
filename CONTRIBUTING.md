# Contributing to StemForge

Thank you for your interest in contributing to StemForge!

---

## Contribution License

StemForge is licensed under the **Apache License 2.0**. By submitting a contribution
(pull request, patch, or other code/documentation), you agree that your contribution
is made under the same Apache 2.0 terms. You retain copyright ownership of your
contributions.

---

## Developer Certificate of Origin (DCO)

All contributions must be signed off under the
[Developer Certificate of Origin v1.1](https://developercertificate.org/):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### How to sign off

Add a `Signed-off-by` line to every commit message:

```bash
git commit -s -m "Add feature X"
```

This produces:

```
Add feature X

Signed-off-by: Your Name <your.email@example.com>
```

The `-s` flag uses the name and email from your Git configuration
(`user.name` and `user.email`).

---

## Prerequisites

### Python
StemForge requires **Python 3.12**.

If your system Python differs:

    uv python install 3.12

### uv
Install uv:

    curl -LsSf https://astral.sh/uv/install.sh | sh

### FFmpeg (required for PyAV)
You must have **FFmpeg >= 5.1** with development headers.

Ubuntu 22.04:

    sudo add-apt-repository -y ppa:ubuntuhandbook1/ffmpeg7
    sudo apt update
    sudo apt install ffmpeg libavcodec-dev libavformat-dev libavdevice-dev \
        libavfilter-dev libavutil-dev libswscale-dev libswresample-dev

Ubuntu 24.04+:

    sudo apt install ffmpeg libavcodec-dev libavformat-dev

Fedora:

    sudo dnf install ffmpeg-free ffmpeg-free-devel

Arch:

    sudo pacman -S ffmpeg

### FluidSynth (required for MIDI preview and Mix tab)

Fedora:

    sudo dnf install fluidsynth fluidsynth-devel fluid-soundfont-gm

Ubuntu / Debian:

    sudo apt install libfluidsynth3 libfluidsynth-dev fluid-soundfont-gm

Arch:

    sudo pacman -S fluidsynth soundfont-fluid

### GPU (optional)
For GPU acceleration:
- NVIDIA driver supporting **CUDA 13.0**
- PyTorch **2.10.0+cu130** (pinned in uv.lock)

CPU-only works everywhere.

---

## Getting Started

Clone (use `--recursive` for the AceStep submodule):

    git clone --recursive git@github.com:tsondo/StemForge.git
    cd StemForge

Sync environment:

    uv sync

Run the app:

    uv run stemforge

Then open http://localhost:8765 in your browser.

---

## Keeping Submodules in Sync

StemForge has a nested submodule chain: **StemForge → Ace-Step-Wrangler → vendor/ACE-Step-1.5**.
A plain `git pull` updates StemForge's submodule pointer in the index but does **not**
automatically check out the new commits inside the submodule directories. Left unattended
this causes confusing `modified: Ace-Step-Wrangler (new commits)` noise in `git status`.

### Recommended: one-time hook setup per workstation

StemForge ships a `post-merge` hook that runs `git submodule update --init --recursive`
automatically after every `git pull`. Activate it once on each workstation:

    git config core.hooksPath scripts/hooks

After that, pulling StemForge will always bring the full submodule chain into sync with
no extra steps.

### Manual sync (without the hook)

If you prefer not to use the hook, sync manually after every pull:

    git submodule update --init --recursive

### Submodule ownership

| Submodule | Owner | Policy |
|---|---|---|
| `Ace-Step-Wrangler/` | Project (tsondo) | First-party — can be freely modified |
| `Ace-Step-Wrangler/vendor/ACE-Step-1.5/` | Upstream | Read-only — pull upstream changes only |

If Ace-Step-Wrangler needs changes to integrate better with StemForge, make them in the
Wrangler repo and pull the updated submodule pointer into StemForge. Do not edit Wrangler
files directly from within the StemForge working tree.

---

## Code Quality

StemForge uses:
- **ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for tests

Run everything:

    make check

Run tests:

    make test

---

## Contribution Guidelines

1. **One feature or fix per PR** -- keep changes focused and reviewable
2. **Run `make check`** before submitting -- no ruff errors, no mypy errors, tests pass
3. **Follow existing code conventions** -- match the style of surrounding code
4. **Sign off your commits** with `git commit -s` (see DCO section above)
5. **Create a feature branch** -- do not commit directly to main
6. **Write clear PR descriptions** -- explain what changed and why

Small, incremental PRs are preferred over large, monolithic changes.

---

## Project Structure

    backend/           - FastAPI backend (API routers + services)
    backend/api/       - API endpoint routers
    backend/services/  - Job manager, session store, pipeline manager, AceStep state
    frontend/          - Vanilla HTML/CSS/JS SPA (served by FastAPI StaticFiles)
    frontend/components/ - Per-tab JS modules
    pipelines/         - ML pipeline implementations
    models/            - Model loaders, registry
    utils/             - Audio I/O, MIDI I/O, paths, device detection, logging, errors
    Ace-Step-Wrangler/ - Git submodule for AceStep (Compose tab)

Notes:
- Backend API endpoints must not block the event loop -- long operations run in background threads via `JobManager`
- All output paths come from `utils/paths.py`
- Import layer order: `utils/ -> models/ -> pipelines/ -> backend/services/ -> backend/api/ -> backend/main.py` (no circular imports)
- Frontend uses an event bus (`appState.on()`/`appState.emit()`) for cross-tab communication
- AceStep runs as a subprocess managed by `run.py`, proxied via `backend/api/compose.py`

---

## Model Weights

Models are cached under:

    ~/.cache/stemforge/

Do not commit model weights.

---

## Contributing from macOS

Copy the macOS pyproject file before syncing:

    cp pyproject.toml.MAC pyproject.toml
    uv sync

Install FluidSynth and set the library path:

    brew install fluid-synth
    export DYLD_LIBRARY_PATH="$(brew --prefix fluid-synth)/lib:$DYLD_LIBRARY_PATH"

**Rules for macOS-compatible code:**

- Always use `from utils.device import get_device` -- never hardcode `"cuda"`
- Always use `from utils.platform import get_data_dir` -- never hardcode `~/.local/share/`
- Avoid `torch.float16` unconditionally -- fall back to `float32` on MPS
- Do NOT commit a `pyproject.toml` derived from `pyproject.toml.MAC`

---

Thank you for helping build StemForge!
