"""Lazy-loaded pipeline singletons with multi-GPU scheduling.

Each GPU gets its own pipeline cache and lock.  The ``GpuScheduler``
assigns incoming jobs to GPUs using affinity (prefer a GPU that already
has the requested pipeline cached) then free-VRAM.  Single-GPU systems
degrade to one slot — identical to the old single-lock behaviour.

Vendored pipelines (Enhance/UVR, RVC/Applio) hardcode
``torch.device("cuda")`` without an index.  They always run on the
default GPU; the scheduler still serialises access via the slot lock.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Generator

import torch

log = logging.getLogger("stemforge.pipeline_manager")


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class GpuSlot:
    """One physical GPU with its own lock and pipeline cache."""

    index: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    pipelines: dict[str, Any] = field(default_factory=dict)


@dataclass
class GpuContext:
    """Yielded by ``gpu_session()`` — tells callers which GPU they got."""

    gpu_index: int | None  # None = CPU / MPS
    device: torch.device


# ── Scheduler ─────────────────────────────────────────────────────────

class GpuScheduler:
    """Per-GPU lock pool with affinity-aware acquisition."""

    def __init__(self) -> None:
        self._slots: list[GpuSlot] = []
        self._excluded: set[int] = set()
        self._fallback_lock = threading.Lock()
        self._initialized = False
        self._init_lock = threading.Lock()

    # -- Initialisation (lazy, once) ------------------------------------

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._do_init()
            self._initialized = True

    def _do_init(self) -> None:
        # All GPUs go into the pool.  GPU assignment to the compose backend
        # is handled dynamically by _get_compose_claimed_gpus() / _pick_slot()
        # rather than being statically excluded at init time.
        if not torch.cuda.is_available():
            log.info("GpuScheduler: no CUDA — using CPU fallback lock")
            return

        count = torch.cuda.device_count()
        for i in range(count):
            self._slots.append(GpuSlot(index=i))

        if self._slots:
            names = [torch.cuda.get_device_name(s.index) for s in self._slots]
            log.info("GpuScheduler: %d GPU slot(s): %s", len(self._slots), names)
        else:
            log.info("GpuScheduler: no CUDA GPUs found — CPU fallback")

    def _get_compose_claimed_gpus(self) -> set[int]:
        """Return GPU indices the compose backend is actively using right now.

        Reads a threading.Event in EmbeddedComposeBackend — no asyncio bridge
        needed.  Returns an empty set if the backend is idle, disabled, or
        if the import fails for any reason.
        """
        try:
            from backend.compose_backend import claimed_gpu_indices
            return claimed_gpu_indices()
        except Exception:
            return set()

    # -- Acquire / release ---------------------------------------------

    @contextlib.contextmanager
    def session(
        self, pipeline_hint: str | None = None,
    ) -> Generator[GpuContext, None, None]:
        """Acquire a GPU slot, yield a :class:`GpuContext`, release on exit."""
        self._ensure_init()

        if not self._slots:
            # CPU / MPS / all GPUs excluded
            self._fallback_lock.acquire()
            try:
                from utils.device import get_device

                yield GpuContext(gpu_index=None, device=get_device())
            finally:
                self._fallback_lock.release()
            return

        slot = self._pick_slot(pipeline_hint)
        slot.lock.acquire()
        try:
            device = torch.device(f"cuda:{slot.index}")
            yield GpuContext(gpu_index=slot.index, device=device)
        finally:
            slot.lock.release()

    def _pick_slot(self, hint: str | None) -> GpuSlot:
        """Choose the best GPU slot for the requested pipeline.

        Strategy:
        0. Filter — exclude GPUs actively claimed by the compose backend.
           If all are claimed, fall back to the full pool so pipelines are
           never permanently starved.
        1. Affinity — a non-locked GPU that already has *hint* cached.
        2. Free VRAM — among non-locked GPUs, pick the one with most free VRAM.
        3. Block — if all are locked, block on the affinity GPU (if any) or
           the first available slot.
        """
        if len(self._slots) == 1:
            return self._slots[0]

        # 0) Filter out GPUs actively used by the compose backend.
        claimed = self._get_compose_claimed_gpus()
        available = [s for s in self._slots if s.index not in claimed] or self._slots

        if len(available) == 1:
            return available[0]

        # 1) Affinity: try non-blocking acquire on a slot that has the hint cached
        if hint:
            for slot in available:
                if hint in slot.pipelines and slot.lock.acquire(timeout=0.1):
                    slot.lock.release()  # we'll re-acquire in session()
                    return slot

        # 2) Free VRAM: pick unlocked slot with most free VRAM
        best_free: GpuSlot | None = None
        best_vram = -1
        for slot in available:
            if slot.lock.acquire(blocking=False):
                slot.lock.release()
                try:
                    free, _ = torch.cuda.mem_get_info(slot.index)
                except Exception:
                    free = 0
                if free > best_vram:
                    best_vram = free
                    best_free = slot

        if best_free is not None:
            return best_free

        # 3) All busy — block on affinity slot or first available slot
        if hint:
            for slot in available:
                if hint in slot.pipelines:
                    return slot
        return available[0]

    # -- Pipeline cache per GPU ----------------------------------------

    def get_pipeline_cache(self, gpu_index: int | None) -> dict[str, Any]:
        """Return the pipeline cache for a specific GPU slot."""
        if gpu_index is None:
            # CPU fallback — use a module-level cache
            return _cpu_pipelines
        for slot in self._slots:
            if slot.index == gpu_index:
                return slot.pipelines
        return _cpu_pipelines

    # -- Info -----------------------------------------------------------

    @property
    def slot_count(self) -> int:
        self._ensure_init()
        return len(self._slots)

    @property
    def excluded_indices(self) -> set[int]:
        self._ensure_init()
        return set(self._excluded)

    def slot_info(self) -> list[dict]:
        """Return a summary of each slot for the /api/device endpoint."""
        self._ensure_init()
        info = []
        for slot in self._slots:
            try:
                free, total = torch.cuda.mem_get_info(slot.index)
            except Exception:
                free, total = 0, 0
            info.append({
                "index": slot.index,
                "name": torch.cuda.get_device_name(slot.index),
                "total_vram_mb": total >> 20,
                "free_vram_mb": free >> 20,
                "cached_pipelines": list(slot.pipelines.keys()),
                "busy": slot.lock.locked(),
            })
        return info


# ── Module-level singletons ──────────────────────────────────────────

_scheduler = GpuScheduler()
_create_lock = threading.Lock()
_cpu_pipelines: dict[str, Any] = {}


@contextlib.contextmanager
def gpu_session(
    pipeline_hint: str | None = None,
) -> Generator[GpuContext, None, None]:
    """Acquire a GPU, yield :class:`GpuContext`, release on exit.

    Usage::

        with pipeline_manager.gpu_session(pipeline_hint="demucs") as ctx:
            pipeline = pipeline_manager.get_demucs(ctx.gpu_index)
            pipeline.load_model(device=ctx.device)
            result = pipeline.run(path)
            pipeline_manager.evict("demucs", ctx.gpu_index)
    """
    with _scheduler.session(pipeline_hint) as ctx:
        yield ctx


# ── Pipeline accessors ───────────────────────────────────────────────

def _get_or_create(name: str, gpu_index: int | None = None) -> Any:
    """Return cached pipeline instance for the given GPU, creating on first call."""
    cache = _scheduler.get_pipeline_cache(gpu_index)

    if name in cache:
        return cache[name]

    with _create_lock:
        if name in cache:
            return cache[name]

        log.info("Creating pipeline: %s (gpu=%s)", name, gpu_index)

        if name == "demucs":
            from pipelines.demucs_pipeline import DemucsPipeline
            cache[name] = DemucsPipeline()

        elif name == "roformer":
            from pipelines.roformer_pipeline import RoformerPipeline
            cache[name] = RoformerPipeline()

        elif name == "midi":
            from pipelines.midi_pipeline import MidiPipeline
            cache[name] = MidiPipeline()

        elif name == "musicgen":
            from pipelines.musicgen_pipeline import MusicGenPipeline
            cache[name] = MusicGenPipeline()

        elif name == "rvc":
            from pipelines.rvc_pipeline import RvcPipeline
            cache[name] = RvcPipeline()

        elif name == "enhance":
            from pipelines.enhance_pipeline import EnhancePipeline
            cache[name] = EnhancePipeline()

        elif name == "autotune":
            from pipelines.autotune_pipeline import AutotunePipeline
            cache[name] = AutotunePipeline()

        elif name == "effects":
            from pipelines.effects_pipeline import EffectsPipeline
            cache[name] = EffectsPipeline()

        elif name == "transcribe":
            from pipelines.transcribe_pipeline import TranscribePipeline
            cache[name] = TranscribePipeline()

        else:
            raise ValueError(f"Unknown pipeline: {name!r}")

        return cache[name]


def get_demucs(gpu_index: int | None = None) -> Any:
    return _get_or_create("demucs", gpu_index)


def get_roformer(gpu_index: int | None = None) -> Any:
    return _get_or_create("roformer", gpu_index)


def get_midi(gpu_index: int | None = None) -> Any:
    return _get_or_create("midi", gpu_index)


def get_musicgen(gpu_index: int | None = None) -> Any:
    return _get_or_create("musicgen", gpu_index)


def get_rvc(gpu_index: int | None = None) -> Any:
    return _get_or_create("rvc", gpu_index)


def get_enhance(gpu_index: int | None = None) -> Any:
    return _get_or_create("enhance", gpu_index)


def get_autotune(gpu_index: int | None = None) -> Any:
    return _get_or_create("autotune", gpu_index)


def get_effects(gpu_index: int | None = None) -> Any:
    return _get_or_create("effects", gpu_index)


def get_transcribe(gpu_index: int | None = None) -> Any:
    return _get_or_create("transcribe", gpu_index)


def evict(name: str, gpu_index: int | None = None) -> None:
    """Release GPU memory for the named pipeline on a specific GPU."""
    cache = _scheduler.get_pipeline_cache(gpu_index)

    with _create_lock:
        pipeline = cache.pop(name, None)
        if pipeline:
            try:
                pipeline.clear()
                log.info("Evicted pipeline: %s (gpu=%s)", name, gpu_index)
            except Exception:
                log.exception("Error evicting pipeline %s", name)

    try:
        if torch.cuda.is_available() and gpu_index is not None:
            with torch.cuda.device(gpu_index):
                torch.cuda.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_scheduler() -> GpuScheduler:
    """Return the module-level scheduler (for /api/device reporting)."""
    return _scheduler
