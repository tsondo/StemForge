"""Lyrics transcription endpoints."""
from __future__ import annotations

import pathlib

import torch
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from backend.services.job_manager import job_manager
from backend.services.session_store import SessionStore, get_user_session
from backend.services import pipeline_manager
from models.registry import list_specs, WhisperSpec
from pipelines.transcribe_engines import ENGINES
from utils.paths import LYRICS_DIR, user_dir

router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


class TranscribeRequest(BaseModel):
    audio_path: str
    engine_id: str = "whisper"
    model_id: str = "whisper-base"
    language: str | None = None
    prompt: str | None = None
    formats: list[str] = ["txt", "lrc", "srt"]


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
        elif engine_id == "qwen":
            info["models"] = [
                {"model_id": "qwen2-audio-7b-instruct",
                 "display_name": "Qwen2-Audio 7B Instruct"},
            ]
        engines.append(info)
    return {"engines": engines, "cuda_available": cuda}


def _run_transcribe(
    req: TranscribeRequest,
    job_id: str,
    session: SessionStore,
) -> dict:
    from pipelines.transcribe_pipeline import TranscribePipeline, TranscribeConfig

    progress_cb = job_manager.make_progress_callback(job_id)
    audio_path = pathlib.Path(req.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {req.audio_path}")

    out_dir = user_dir(LYRICS_DIR, session.user)
    config = TranscribeConfig(
        engine_id=req.engine_id,
        model_id=req.model_id,
        language=req.language,
        prompt=req.prompt,
        output_dir=out_dir,
        formats=tuple(req.formats),
    )

    with pipeline_manager.gpu_session(pipeline_hint="transcribe") as ctx:
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
    request: Request,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    user = getattr(request.state, "user", "local")
    job_id = job_manager.create_job("transcribe", user=user)
    job_manager.run_job(job_id, _run_transcribe, req, job_id, session)
    return {"job_id": job_id}
