"""Lyrics transcription endpoints."""
from __future__ import annotations

import pathlib

import torch
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.services.job_manager import job_manager
from backend.services.session_store import SessionStore, get_user_session
from backend.services import pipeline_manager
from models.registry import list_specs, WhisperSpec
from pipelines.transcribe_engines import ENGINES
from utils.paths import ENHANCE_DIR, LYRICS_DIR, STEMS_DIR, user_dir

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


class TranscribeRequest(BaseModel):
    audio_path: str
    engine_id: str = "whisper"
    model_id: str = "whisper-large-v3"
    language: str | None = None
    prompt: str | None = None
    formats: list[str] = ["txt", "lrc", "srt"]
    condition_on_previous_text: bool = False   # Whisper only; off = suppress hallucination loops
    collapse_repetitions: bool = True          # collapse runs of duplicate segments
    max_repetition_run: int = 4               # max identical consecutive segments before collapse


@router.get("/engines")
def list_engines() -> dict:
    """Return available engines and their capabilities for the UI."""
    cuda = torch.cuda.is_available()
    engines = []
    for engine_id, cls in ENGINES.items():
        info = {
            "engine_id": engine_id,
            "supports_word_timestamps": cls.supports_word_timestamps,
            "requires_gpu": cls.requires_gpu,
            "available": (not cls.requires_gpu) or cuda,
            "models": [],
        }
        if engine_id == "whisper":
            info["models"] = [
                {"model_id": s.model_id, "display_name": s.display_name}
                for s in list_specs(WhisperSpec)
            ]
        elif engine_id == "qwen3-asr":
            from pipelines.transcribe_engines.qwen3_asr_engine import (
                QWEN3_ASR_VARIANTS,
            )
            info["models"] = [
                {
                    "model_id": mid,
                    "display_name": v["display_name"],
                    "approx_vram_gb": v["approx_vram_gb"],
                    "description": v["description"],
                    "available": cuda,
                }
                for mid, v in QWEN3_ASR_VARIANTS.items()
            ]
        engines.append(info)
    return {"engines": engines, "cuda_available": cuda}


def _run_transcribe(
    req: TranscribeRequest,
    job_id: str,
    session: SessionStore,
) -> dict:
    from pipelines.transcribe_pipeline import TranscribeConfig

    progress_cb = job_manager.make_progress_callback(job_id)
    audio_path = pathlib.Path(req.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {req.audio_path}")

    out_dir = user_dir(LYRICS_DIR, session.user)

    with pipeline_manager.gpu_session(pipeline_hint="transcribe") as ctx:
        config = TranscribeConfig(
            engine_id=req.engine_id,
            model_id=req.model_id,
            language=req.language,
            prompt=req.prompt,
            output_dir=out_dir,
            formats=tuple(req.formats),
            gpu_index=ctx.gpu_index,
            condition_on_previous_text=req.condition_on_previous_text,
            collapse_repetitions=req.collapse_repetitions,
            max_repetition_run=req.max_repetition_run,
        )
        pipeline = pipeline_manager.get_transcribe(ctx.gpu_index)
        pipeline.configure(config)
        try:
            progress_cb(0.05, "Loading model")
            pipeline.load_model()
            result = pipeline.run(audio_path, progress_cb=progress_cb)
        finally:
            pipeline_manager.evict("transcribe", ctx.gpu_index)

    for fmt, path in result.output_paths.items():
        label = f"{result.label} [{fmt}]"
        session.add_lyrics_path(label, path)

    return {
        "engine_id": result.result.engine_id,
        "model_id": result.result.model_id,
        "language": result.result.language,
        "has_word_timestamps": result.result.has_word_timestamps,
        "text": result.result.text,
        "segment_count": len(result.result.segments),
        "output_paths": {k: str(v) for k, v in result.output_paths.items()},
        "label": result.label,
    }


@router.post("")
def start_transcribe(
    req: TranscribeRequest,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    # Validate the audio path synchronously so a bad request fails fast
    # instead of surfacing as a job error after polling.  Allowed roots
    # match the source-selector options in the Lyrics UI: separated
    # stems, enhanced stems, and the originally uploaded audio.
    audio_path = pathlib.Path(req.audio_path).resolve()
    if not audio_path.exists():
        raise HTTPException(404, f"Audio file not found: {req.audio_path}")

    allowed_roots: list[pathlib.Path] = [STEMS_DIR.resolve(), ENHANCE_DIR.resolve()]
    if session.audio_path is not None:
        allowed_roots.append(session.audio_path.parent.resolve())
    if not any(
        str(audio_path).startswith(str(root)) for root in allowed_roots
    ):
        raise HTTPException(
            403,
            "Audio path is outside the allowed source directories. "
            "Choose a separated stem, enhanced stem, or your uploaded audio.",
        )

    job_id = job_manager.create_job("transcribe", user=session.user)
    job_manager.run_job(job_id, _run_transcribe, req, job_id, session)
    return {"job_id": job_id}
