"""Output-directory constants shared across all layers.

Extracted from ``gui/constants.py`` so that pipelines and backend code can
import these paths without depending on the GUI package.
"""

import pathlib
import re

from utils.platform import get_data_dir

_SAFE_RE = re.compile(r"[^a-zA-Z0-9_.\-@]")


def user_dir(base: pathlib.Path, user: str) -> pathlib.Path:
    """Return a per-user subdirectory under *base*, creating it if needed.

    Sanitizes *user* to prevent path traversal.  The ``"local"`` user
    (single-user dev mode) gets ``base/local/``.
    """
    safe = _SAFE_RE.sub("_", user)[:64] or "anonymous"
    d = base / safe
    d.mkdir(parents=True, exist_ok=True)
    return d

OUTPUT_BASE  = get_data_dir() / "output"
STEMS_DIR    = OUTPUT_BASE / "stems"
MIDI_DIR     = pathlib.Path.home() / "Music" / "StemForge"
MUSICGEN_DIR = OUTPUT_BASE / "musicgen"
MIX_DIR      = OUTPUT_BASE / "mix"
EXPORT_DIR   = OUTPUT_BASE / "exports"
COMPOSE_DIR  = OUTPUT_BASE / "compose"
SFX_DIR      = OUTPUT_BASE / "sfx"
VOICE_DIR    = OUTPUT_BASE / "voice"
ENHANCE_DIR  = OUTPUT_BASE / "enhance"
LYRICS_DIR   = OUTPUT_BASE / "lyrics"
