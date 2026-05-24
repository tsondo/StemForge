"""Audio endpoints: upload, stream, download, waveform, profile."""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import uuid

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse

from backend.services.session_store import SessionStore, get_user_session
from utils.paths import OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR, LYRICS_DIR

router = APIRouter(prefix="/api", tags=["audio"])

_UPLOAD_DIR = OUTPUT_BASE / "uploads"

# AceStep generates audio into its own cache tree inside the submodule.
# Allow streaming from there so compose results are accessible before
# the user explicitly sends them to session/mix.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ACESTEP_AUDIO_TMP = _PROJECT_ROOT / "Ace-Step-Wrangler" / "vendor" / "ACE-Step-1.5" / ".cache" / "acestep" / "tmp"

# Directories from which we allow file streaming (security)
_ALLOWED_ROOTS = [
    OUTPUT_BASE, STEMS_DIR, MIDI_DIR, MUSICGEN_DIR, MIX_DIR, EXPORT_DIR, COMPOSE_DIR, SFX_DIR, LYRICS_DIR, _UPLOAD_DIR,
    _ACESTEP_AUDIO_TMP,
]


def _validate_path(path_str: str) -> pathlib.Path:
    """Resolve and validate that path is within allowed directories."""
    p = pathlib.Path(path_str).resolve()
    if not any(str(p).startswith(str(root.resolve())) for root in _ALLOWED_ROOTS):
        raise HTTPException(403, "Access denied: path outside allowed directories")
    if not p.exists():
        raise HTTPException(404, f"File not found: {p.name}")
    return p


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session: SessionStore = Depends(get_user_session),
) -> dict:
    import subprocess
    from utils.audio_io import probe, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS

    ext = pathlib.Path(file.filename or "upload").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
        raise HTTPException(422, f"Unsupported format: {ext}")

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"

    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract audio from video files via FFmpeg (preserve source sample rate)
    if ext in VIDEO_EXTENSIONS:
        wav_dest = dest.with_suffix(".wav")
        try:
            subprocess.run(
                ["ffmpeg", "-i", str(dest), "-vn", "-acodec", "pcm_s24le",
                 str(wav_dest)],
                check=True, capture_output=True, timeout=120,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(422, f"Failed to extract audio from video: {exc}")
        dest.unlink(missing_ok=True)
        dest = wav_dest

    display_name = file.filename
    info = probe(dest)
    session.audio_path = dest
    session.audio_info = {
        "filename": display_name,
        "path": str(dest),
        "duration": info.duration,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "format": info.format,
        "bit_depth": info.bit_depth,
    }

    return {
        "filename": display_name,
        "path": str(dest),
        "duration": info.duration,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "bit_depth": info.bit_depth,
    }


@router.post("/upload-batch")
async def upload_batch(files: list[UploadFile] = File(...)) -> dict:
    """Upload multiple audio/video files for batch processing."""
    import subprocess
    from utils.audio_io import probe, SUPPORTED_EXTENSIONS, VIDEO_EXTENSIONS

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for file in files:
        ext = pathlib.Path(file.filename or "upload").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS and ext not in VIDEO_EXTENSIONS:
            results.append({"filename": file.filename, "error": f"Unsupported format: {ext}"})
            continue

        dest = _UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if ext in VIDEO_EXTENSIONS:
            wav_dest = dest.with_suffix(".wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-i", str(dest), "-vn", "-acodec", "pcm_s24le",
                     str(wav_dest)],
                    check=True, capture_output=True, timeout=120,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as exc:
                dest.unlink(missing_ok=True)
                results.append({"filename": file.filename, "error": f"Video extraction failed: {exc}"})
                continue
            dest.unlink(missing_ok=True)
            dest = wav_dest

        info = probe(dest)
        results.append({
            "filename": file.filename,
            "path": str(dest),
            "duration": info.duration,
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "bit_depth": info.bit_depth,
        })

    return {"files": results}


_MIME_MAP = {
    ".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg", ".aiff": "audio/aiff", ".aif": "audio/aiff",
    ".m4a": "audio/mp4",
}


@router.get("/audio/stream")
def stream_audio(path: str = Query(...)) -> FileResponse:
    p = _validate_path(path)
    mime = _MIME_MAP.get(p.suffix.lower(), "audio/wav")
    return FileResponse(p, media_type=mime)


@router.get("/audio/download")
def download_audio(path: str = Query(...)) -> FileResponse:
    p = _validate_path(path)
    return FileResponse(
        p,
        media_type="application/octet-stream",
        filename=p.name,
    )


@router.get("/audio/waveform")
def get_waveform(path: str = Query(...), points: int = Query(2000)) -> dict:
    """Return downsampled peak data for waveform visualization."""
    p = _validate_path(path)
    from utils.audio_io import read_audio

    waveform, sr = read_audio(p, mono=True)
    samples = waveform[0]  # shape: (samples,)

    # Downsample to requested number of points
    n = len(samples)
    if n <= points:
        peaks = samples.tolist()
    else:
        chunk_size = n // points
        peaks = []
        for i in range(points):
            chunk = samples[i * chunk_size : (i + 1) * chunk_size]
            peaks.append(float(np.max(np.abs(chunk))))

    return {"peaks": peaks, "duration": n / sr, "sample_rate": sr}


@router.get("/audio/info")
def audio_info(path: str = Query(...)) -> dict:
    p = _validate_path(path)
    from utils.audio_io import probe

    info = probe(p)
    return {
        "path": str(p),
        "duration": info.duration,
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "format": info.format,
        "bit_depth": info.bit_depth,
    }


@router.post("/audio/profile")
def profile_audio(session: SessionStore = Depends(get_user_session)) -> dict:
    """Run audio profiler on the current session audio."""
    audio_path = session.audio_path
    if not audio_path:
        raise HTTPException(400, "No audio file loaded")

    from utils.audio_profile import profile_audio as _profile, recommend_separator

    profile = _profile(audio_path)
    rec = recommend_separator(profile)

    return {
        "profile": {
            "spectral_flatness": profile.spectral_flatness,
            "transient_sharpness": profile.transient_sharpness,
            "transient_density": profile.transient_density,
            "dynamic_range": profile.dynamic_range,
            "noise_floor": profile.noise_floor,
            "stereo_correlation": profile.stereo_correlation,
            "harmonic_decay": profile.harmonic_decay,
            "vocal_naturalness": profile.vocal_naturalness,
            "drum_intrusion_risk": profile.drum_intrusion_risk,
        },
        "recommendation": {
            "engine": rec.engine,
            "model_id": rec.model_id,
            "reason": rec.reason,
            "confidence": rec.confidence,
        },
    }
