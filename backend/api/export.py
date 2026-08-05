"""Export and zip download endpoints."""

from __future__ import annotations

import io
import pathlib
import uuid
import zipfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.job_manager import job_manager
from utils.paths import EXPORT_DIR, user_dir

router = APIRouter(prefix="/api/export", tags=["export"])


class ExportRequest(BaseModel):
    items: list[str]          # list of file paths to export
    format: str = "wav"       # wav, flac, aiff, mp3, ogg (Opus), m4a (AAC)
    bitrate: int | None = None    # kbps for lossy formats (mp3/ogg/m4a)
    sample_rate: int | None = None  # target sample rate for lossless (e.g. 44100, 48000)
    bit_depth: int | None = None    # target bit depth for lossless (16, 24, 32)


class ZipRequest(BaseModel):
    items: list[str]          # list of file paths to zip


def _run_export(
    items: list[str],
    fmt: str,
    bitrate: int | None,
    sample_rate: int | None,
    bit_depth: int | None,
    job_id: str,
    user: str = "local",
) -> dict:
    """Convert selected items to target format."""
    from utils.audio_io import read_audio, write_audio

    progress_cb = job_manager.make_progress_callback(job_id)
    export_out = user_dir(EXPORT_DIR, user)

    exported = []
    for i, item_path in enumerate(items):
        progress_cb(i / len(items), f"Exporting {pathlib.Path(item_path).name}...")
        src = pathlib.Path(item_path)
        if not src.exists():
            continue

        # Non-audio artifacts — lyrics text (.txt/.lrc/.srt) and MIDI files
        # (.mid/.midi) — are copied verbatim into the export dir, bypassing
        # audio transcoding.  The selected audio format does not apply.
        if src.suffix.lower() in {".txt", ".lrc", ".srt", ".mid", ".midi"}:
            import shutil
            dest = export_out / src.name
            shutil.copy2(src, dest)
            exported.append(str(dest))
            continue

        dest = export_out / f"{src.stem}.{fmt}"

        # Fast path: same format with no parameter overrides — just copy
        same_fmt = src.suffix.lstrip(".").lower() == fmt
        no_overrides = bitrate is None and sample_rate is None and bit_depth is None
        if same_fmt and no_overrides:
            import shutil
            shutil.copy2(src, dest)
        else:
            from utils.audio_io import probe as audio_probe

            waveform, sr = read_audio(src, target_rate=sample_rate)
            out_sr = sample_rate or sr

            # Preserve source bit depth when not overridden
            out_bd = bit_depth
            if out_bd is None:
                src_info = audio_probe(src)
                out_bd = src_info.bit_depth or 16

            write_audio(
                waveform, out_sr, dest,
                fmt=fmt,
                bit_depth=out_bd,
                bitrate=bitrate,
            )

        exported.append(str(dest))

    progress_cb(1.0, "Done")
    return {"exported": exported}


@router.post("")
def start_export(req: ExportRequest, request: Request) -> dict:
    if not req.items:
        raise HTTPException(422, "No items to export")
    if req.format not in ("wav", "flac", "aiff", "mp3", "ogg", "m4a"):
        raise HTTPException(422, f"Unsupported format: {req.format}")

    if req.bit_depth is not None and req.bit_depth not in (16, 24, 32):
        raise HTTPException(422, f"Unsupported bit depth: {req.bit_depth}")

    user = getattr(request.state, "user", "local")
    job_id = job_manager.create_job("export", user=user)
    job_manager.run_job(
        job_id, _run_export,
        req.items, req.format, req.bitrate, req.sample_rate, req.bit_depth,
        job_id, user,
    )
    return {"job_id": job_id}


@router.post("/download-zip")
def download_zip(req: ZipRequest) -> StreamingResponse:
    if not req.items:
        raise HTTPException(422, "No items to zip")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item_path in req.items:
            p = pathlib.Path(item_path)
            if p.exists():
                zf.write(p, p.name)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=stemforge_export.zip"},
    )
