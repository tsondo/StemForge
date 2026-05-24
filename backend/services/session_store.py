"""Per-user session state with thread-safe registry.

Each user gets their own ``SessionStore`` instance, managed by a
``SessionRegistry`` singleton.  The registry is keyed by username
(from the ``x-auth-user`` header, defaulting to ``"local"`` for
single-user dev mode).

FastAPI endpoints resolve the current user's session via the
``get_user_session`` dependency.
"""

from __future__ import annotations

import pathlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request


@dataclass
class TrackState:
    track_id: str
    label: str
    source: str               # "audio" or "midi"
    path: pathlib.Path | None = None
    midi_data: Any = None     # PrettyMIDI object for MIDI tracks
    enabled: bool = True
    volume: float = 0.8
    program: int = 0          # GM program number
    is_drum: bool = False


class SessionStore:
    """Thread-safe per-user session state."""

    def __init__(self, user: str = "local") -> None:
        self._lock = threading.Lock()
        self.user = user
        self.last_seen: float = time.monotonic()
        self._audio_path: pathlib.Path | None = None
        self._audio_info: dict[str, Any] | None = None
        self._stem_paths: dict[str, pathlib.Path] = {}
        self._merged_midi_data: Any = None           # PrettyMIDI
        self._stem_midi_data: dict[str, Any] = {}    # label → PrettyMIDI
        self._musicgen_path: pathlib.Path | None = None
        self._mix_path: pathlib.Path | None = None
        self._mix_tracks: list[TrackState] = []
        self._compose_paths: list[dict[str, Any]] = []
        self._sfx_manifests: dict[str, dict] = {}  # sfx_id → manifest dict
        self._voice_paths: dict[str, pathlib.Path] = {}  # label → output path
        self._enhance_paths: dict[str, pathlib.Path] = {}  # label → enhanced output path
        self._lyrics_paths: dict[str, pathlib.Path] = {}  # label → lyrics file path
        self._kept_clips: set[str] = set()  # paths explicitly kept by user

    # -- audio_path --
    @property
    def audio_path(self) -> pathlib.Path | None:
        with self._lock:
            return self._audio_path

    @audio_path.setter
    def audio_path(self, value: pathlib.Path | None) -> None:
        with self._lock:
            self._audio_path = value

    # -- audio_info --
    @property
    def audio_info(self) -> dict[str, Any] | None:
        with self._lock:
            return self._audio_info

    @audio_info.setter
    def audio_info(self, value: dict[str, Any] | None) -> None:
        with self._lock:
            self._audio_info = value

    # -- stem_paths --
    @property
    def stem_paths(self) -> dict[str, pathlib.Path]:
        with self._lock:
            return dict(self._stem_paths)

    @stem_paths.setter
    def stem_paths(self, value: dict[str, pathlib.Path]) -> None:
        with self._lock:
            self._stem_paths = dict(value)

    # -- merged_midi_data --
    @property
    def merged_midi_data(self) -> Any:
        with self._lock:
            return self._merged_midi_data

    @merged_midi_data.setter
    def merged_midi_data(self, value: Any) -> None:
        with self._lock:
            self._merged_midi_data = value

    # -- stem_midi_data --
    @property
    def stem_midi_data(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._stem_midi_data)

    @stem_midi_data.setter
    def stem_midi_data(self, value: dict[str, Any]) -> None:
        with self._lock:
            self._stem_midi_data = dict(value)

    # -- musicgen_path --
    @property
    def musicgen_path(self) -> pathlib.Path | None:
        with self._lock:
            return self._musicgen_path

    @musicgen_path.setter
    def musicgen_path(self, value: pathlib.Path | None) -> None:
        with self._lock:
            self._musicgen_path = value

    # -- mix_path --
    @property
    def mix_path(self) -> pathlib.Path | None:
        with self._lock:
            return self._mix_path

    @mix_path.setter
    def mix_path(self, value: pathlib.Path | None) -> None:
        with self._lock:
            self._mix_path = value

    # -- mix_tracks --
    @property
    def mix_tracks(self) -> list[TrackState]:
        with self._lock:
            return list(self._mix_tracks)

    @mix_tracks.setter
    def mix_tracks(self, value: list[TrackState]) -> None:
        with self._lock:
            self._mix_tracks = list(value)

    # -- compose_paths --
    @property
    def compose_paths(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._compose_paths)

    @compose_paths.setter
    def compose_paths(self, value: list[dict[str, Any]]) -> None:
        with self._lock:
            self._compose_paths = list(value)

    def add_compose_path(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._compose_paths.append(entry)

    def add_track(self, track: TrackState) -> None:
        with self._lock:
            self._mix_tracks.append(track)

    def remove_track(self, track_id: str) -> bool:
        with self._lock:
            before = len(self._mix_tracks)
            self._mix_tracks = [t for t in self._mix_tracks if t.track_id != track_id]
            return len(self._mix_tracks) < before

    def get_track(self, track_id: str) -> TrackState | None:
        with self._lock:
            for t in self._mix_tracks:
                if t.track_id == track_id:
                    return t
            return None

    # -- sfx_manifests --
    def add_sfx_manifest(self, manifest: dict) -> None:
        with self._lock:
            self._sfx_manifests[manifest["id"]] = manifest

    def get_sfx_manifest(self, sfx_id: str) -> dict | None:
        with self._lock:
            m = self._sfx_manifests.get(sfx_id)
            return dict(m) if m else None

    def remove_sfx_manifest(self, sfx_id: str) -> bool:
        with self._lock:
            return self._sfx_manifests.pop(sfx_id, None) is not None

    @property
    def sfx_manifest_ids(self) -> list[str]:
        with self._lock:
            return list(self._sfx_manifests.keys())

    # -- voice_paths --
    @property
    def voice_paths(self) -> dict[str, pathlib.Path]:
        with self._lock:
            return dict(self._voice_paths)

    @voice_paths.setter
    def voice_paths(self, value: dict[str, pathlib.Path]) -> None:
        with self._lock:
            self._voice_paths = dict(value)

    def add_voice_path(self, label: str, path: pathlib.Path) -> None:
        with self._lock:
            self._voice_paths[label] = path

    # -- enhance_paths --
    @property
    def enhance_paths(self) -> dict[str, pathlib.Path]:
        with self._lock:
            return dict(self._enhance_paths)

    def add_enhance_path(self, label: str, path: pathlib.Path) -> None:
        with self._lock:
            self._enhance_paths[label] = path

    # -- lyrics_paths --
    @property
    def lyrics_paths(self) -> dict[str, pathlib.Path]:
        with self._lock:
            return dict(self._lyrics_paths)

    def add_lyrics_path(self, label: str, path: pathlib.Path) -> None:
        with self._lock:
            self._lyrics_paths[label] = path

    # -- kept_clips --
    @property
    def kept_clips(self) -> set[str]:
        with self._lock:
            return set(self._kept_clips)

    def keep_clip(self, path: str) -> None:
        with self._lock:
            self._kept_clips.add(path)

    def unkeep_clip(self, path: str) -> None:
        with self._lock:
            self._kept_clips.discard(path)

    def clear(self) -> None:
        with self._lock:
            self._audio_path = None
            self._audio_info = None
            self._stem_paths = {}
            self._merged_midi_data = None
            self._stem_midi_data = {}
            self._musicgen_path = None
            self._mix_path = None
            self._mix_tracks = []
            self._compose_paths = []
            self._sfx_manifests = {}
            self._voice_paths = {}
            self._enhance_paths = {}
            self._lyrics_paths = {}
            self._kept_clips = set()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "user": self.user,
                "audio_path": str(self._audio_path) if self._audio_path else None,
                "audio_info": self._audio_info,
                "stem_paths": {k: str(v) for k, v in self._stem_paths.items()},
                "has_merged_midi": self._merged_midi_data is not None,
                "stem_midi_labels": list(self._stem_midi_data.keys()),
                "musicgen_path": str(self._musicgen_path) if self._musicgen_path else None,
                "mix_path": str(self._mix_path) if self._mix_path else None,
                "compose_paths": list(self._compose_paths),
                "mix_tracks": [
                    {
                        "track_id": t.track_id,
                        "label": t.label,
                        "source": t.source,
                        "path": str(t.path) if t.path else None,
                        "enabled": t.enabled,
                        "volume": t.volume,
                        "program": t.program,
                        "is_drum": t.is_drum,
                    }
                    for t in self._mix_tracks
                ],
                "sfx_manifests": [
                    {"id": m["id"], "name": m["name"]}
                    for m in self._sfx_manifests.values()
                ],
                "voice_paths": {k: str(v) for k, v in self._voice_paths.items()},
                "enhance_paths": {k: str(v) for k, v in self._enhance_paths.items()},
                "lyrics_paths": {k: str(v) for k, v in self._lyrics_paths.items()},
            }


# ---------------------------------------------------------------------------
# Session registry — maps user IDs to SessionStore instances
# ---------------------------------------------------------------------------

_SAFE_USER_RE = re.compile(r"[^a-zA-Z0-9_.\-@]")


def _sanitize_user(user: str) -> str:
    """Sanitize username for use in filesystem paths."""
    return _SAFE_USER_RE.sub("_", user)[:64] or "anonymous"


class SessionRegistry:
    """Thread-safe registry of per-user sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionStore] = {}

    def get(self, user: str) -> SessionStore:
        """Get or create a session for *user*, updating last_seen."""
        with self._lock:
            if user not in self._sessions:
                self._sessions[user] = SessionStore(user=user)
            s = self._sessions[user]
            s.last_seen = time.monotonic()
            return s

    def remove(self, user: str) -> bool:
        """Remove a user's session. Returns True if it existed."""
        with self._lock:
            return self._sessions.pop(user, None) is not None

    def active_count(self, timeout_seconds: float) -> int:
        """Count sessions active within *timeout_seconds*."""
        now = time.monotonic()
        with self._lock:
            return sum(
                1 for s in self._sessions.values()
                if now - s.last_seen < timeout_seconds
            )

    def expire(self, timeout_seconds: float) -> list[str]:
        """Remove sessions inactive for longer than *timeout_seconds*.

        Returns list of expired usernames.
        """
        now = time.monotonic()
        with self._lock:
            expired = [
                u for u, s in self._sessions.items()
                if now - s.last_seen > timeout_seconds
            ]
            for u in expired:
                del self._sessions[u]
            return expired

    def try_admit(self, user: str, max_users: int, timeout_seconds: float) -> SessionStore | None:
        """Atomically check capacity and create/return session.

        Returns the user's SessionStore if admitted (or already known).
        Returns None if this is a new user and the server is at capacity.
        """
        with self._lock:
            if user not in self._sessions:
                if max_users > 0:
                    now = time.monotonic()
                    active = sum(
                        1 for s in self._sessions.values()
                        if now - s.last_seen < timeout_seconds
                    )
                    if active >= max_users:
                        return None
                self._sessions[user] = SessionStore(user=user)
            s = self._sessions[user]
            s.last_seen = time.monotonic()
            return s

    def list_users(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())


# Module-level singleton registry
registry = SessionRegistry()

# Backward-compatible alias — returns the "local" user session.
# Used by code not yet migrated to the Depends() pattern.
session = registry.get("local")


# ---------------------------------------------------------------------------
# FastAPI dependency for per-user session resolution
# ---------------------------------------------------------------------------

def get_user_session(request: Request) -> SessionStore:
    """FastAPI dependency: resolve the current user's session.

    The user is set by the ``inject_user`` middleware in ``main.py``
    from the ``x-auth-user`` header (or ``"local"`` in dev mode).
    """
    user = getattr(request.state, "user", "local")
    return registry.get(user)
