"""System endpoints: health, device info, models, session."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Request

from backend.services.session_store import (
    SessionStore, get_user_session, registry,
)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/device")
def device_info() -> dict:
    import asyncio
    import torch
    from utils.device import get_device, enumerate_gpus
    from backend.services.pipeline_manager import get_scheduler

    dev = get_device()
    info: dict = {"device": str(dev)}

    if dev.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(total / (1024 ** 3), 1)

        # Multi-GPU details
        gpus = enumerate_gpus()
        if len(gpus) > 1:
            scheduler = get_scheduler()
            info["gpus"] = [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_vram_mb": g.total_vram_mb,
                    "free_vram_mb": g.free_vram_mb,
                }
                for g in gpus
            ]
            info["scheduler"] = {
                "pool_slots": scheduler.slot_count,
                "excluded_indices": sorted(scheduler.excluded_indices),
                "slots": scheduler.slot_info(),
            }
    elif dev.type == "mps":
        info["gpu_name"] = "Apple Silicon (MPS)"

    # Compose backend GPU state (device-independent)
    try:
        from backend.compose_backend import get_compose_backend, claimed_gpu_indices
        from backend.compose_backend.protocol import BackendMode
        backend = get_compose_backend()
        if backend is not None:
            claimed = sorted(claimed_gpu_indices())
            loop = asyncio.new_event_loop()
            try:
                status = loop.run_until_complete(backend.get_status())
            finally:
                loop.close()
            info["compose"] = {
                "mode": status.mode.value,
                "status": status.status,
                "claimed_gpu_indices": claimed,
                "gpu_busy": bool(claimed),
            }
        else:
            info["compose"] = {"mode": BackendMode.DISABLED.value, "status": "disabled"}
    except Exception:
        pass

    return info


@router.get("/capabilities")
def capabilities() -> dict:
    """Report optional system capabilities (LilyPond, etc.)."""
    from utils.music21_bridge import check_lilypond
    return {"lilypond": check_lilypond()}


@router.get("/models")
def list_models() -> dict:
    from models.registry import (
        list_specs, DemucsSpec, RoformerSpec, BasicPitchSpec,
        DrumMidiSpec, WhisperSpec, StableAudioSpec,
    )

    def _serialize(spec) -> dict:
        d = {
            "model_id": spec.model_id,
            "display_name": spec.display_name,
            "description": spec.description,
            "device": spec.device,
            "sample_rate": spec.sample_rate,
            "available_stems": list(getattr(spec, "available_stems", [])),
        }
        if spec.license_warning:
            d["license_warning"] = spec.license_warning
        return d

    return {
        "demucs": [_serialize(s) for s in list_specs(DemucsSpec)],
        "roformer": [_serialize(s) for s in list_specs(RoformerSpec)],
        "basicpitch": [_serialize(s) for s in list_specs(BasicPitchSpec)],
        "drum_midi": [_serialize(s) for s in list_specs(DrumMidiSpec)],
        "whisper": [_serialize(s) for s in list_specs(WhisperSpec)],
        "stable_audio": [_serialize(s) for s in list_specs(StableAudioSpec)],
    }


@router.get("/session")
def get_session(
    request: Request,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    timeout = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "60")) * 60
    data = session.to_dict()
    data["active_users"] = registry.active_count(timeout)
    data["max_users"] = int(os.environ.get("MAX_USERS", "0"))
    return data


@router.delete("/session")
def clear_session(session: SessionStore = Depends(get_user_session)) -> dict:
    session.clear()
    return {"status": "cleared"}


# ── HuggingFace token management ──────────────────────────────────────────

from utils.platform import get_data_dir
from pydantic import BaseModel as _BaseModel

_HF_TOKEN_PATH = get_data_dir() / "hf_token"


class _HFTokenRequest(_BaseModel):
    token: str


@router.get("/hf-token")
def hf_token_status() -> dict:
    """Check whether a HuggingFace token is configured."""
    # Check our own stored token first, then fall back to huggingface_hub
    if _HF_TOKEN_PATH.is_file() and _HF_TOKEN_PATH.read_text().strip():
        return {"has_token": True, "source": "stemforge"}
    try:
        from huggingface_hub import get_token
        if get_token():
            return {"has_token": True, "source": "huggingface-cli"}
    except Exception:
        pass
    return {"has_token": False, "source": None}


@router.post("/hf-token")
def save_hf_token(req: _HFTokenRequest) -> dict:
    """Save a user-provided HuggingFace token."""
    token = req.token.strip()
    if not token.startswith("hf_"):
        from fastapi import HTTPException as _HE
        raise _HE(422, "Invalid token format — HuggingFace tokens start with 'hf_'")
    _HF_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _HF_TOKEN_PATH.write_text(token)
    _HF_TOKEN_PATH.chmod(0o600)
    return {"status": "saved"}


@router.delete("/hf-token")
def delete_hf_token() -> dict:
    """Remove the stored HuggingFace token."""
    if _HF_TOKEN_PATH.is_file():
        _HF_TOKEN_PATH.unlink()
    return {"status": "deleted"}
