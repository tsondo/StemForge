"""Lyrics transcription pipeline.

Selects a transcription engine (Whisper or Qwen), runs transcription on
a single audio file, and writes plain text, LRC, and SRT outputs.

Lifecycle: configure → load_model → run → clear.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Callable

from pipelines.transcribe_engines import (
    ENGINES,
    TranscriptionEngine,
    TranscriptionResult,
)
from utils.errors import AudioProcessingError, InvalidInputError, PipelineExecutionError

log = logging.getLogger(__name__)

_SUPPORTED_EXTS = frozenset({".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif", ".m4a"})


@dataclass(slots=True)
class TranscribeConfig:
    engine_id: str = "whisper"
    model_id: str = "whisper-base"
    language: str | None = None
    prompt: str | None = None
    output_dir: pathlib.Path | None = None
    formats: tuple[str, ...] = ("txt", "lrc", "srt")


@dataclass(slots=True)
class TranscribeResult:
    result: TranscriptionResult
    output_paths: dict[str, pathlib.Path]
    label: str


class TranscribePipeline:
    """Lifecycle: configure → load_model → run → clear."""

    def __init__(self) -> None:
        self._config: TranscribeConfig | None = None
        self._engine: TranscriptionEngine | None = None

    def configure(self, config: TranscribeConfig) -> None:
        if config.engine_id not in ENGINES:
            raise InvalidInputError(
                f"Unknown engine_id {config.engine_id!r}. "
                f"Available: {sorted(ENGINES)}",
                field="engine_id",
            )
        # If a different engine/model was already loaded, evict it so the
        # next load_model() instantiates fresh.  Matches DemucsPipeline's
        # configure() convention.
        if self._engine is not None and self._config is not None and (
            config.engine_id != self._config.engine_id
            or config.model_id != self._config.model_id
        ):
            self.clear()
        self._config = config

    def load_model(self) -> None:
        if self._config is None:
            raise PipelineExecutionError(
                "configure() must be called before load_model().",
                pipeline_name="transcribe",
            )
        engine_cls = ENGINES[self._config.engine_id]
        self._engine = engine_cls(model_id=self._config.model_id)
        self._engine.load()

    def run(
        self,
        audio_path: pathlib.Path,
        progress_cb: Callable[[float, str], None] | None = None,
    ) -> TranscribeResult:
        if self._config is None or self._engine is None:
            raise PipelineExecutionError(
                "load_model() must be called before run().",
                pipeline_name="transcribe",
            )
        if not audio_path.exists():
            raise InvalidInputError(
                f"Audio file not found: {audio_path}", field="audio_path",
            )
        if audio_path.suffix.lower() not in _SUPPORTED_EXTS:
            raise InvalidInputError(
                f"Unsupported audio format: {audio_path.suffix}",
                field="audio_path",
            )

        if progress_cb:
            progress_cb(0.1, "Transcribing")

        result = self._engine.transcribe(
            audio_path,
            language=self._config.language,
            prompt=self._config.prompt,
        )

        if progress_cb:
            progress_cb(0.85, "Writing outputs")

        out_dir = self._config.output_dir or audio_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = audio_path.stem
        engine_tag = result.engine_id
        base = f"{stem}-lyrics-{engine_tag}"

        output_paths: dict[str, pathlib.Path] = {}
        for fmt in self._config.formats:
            try:
                if fmt == "txt":
                    p = out_dir / f"{base}.txt"
                    p.write_text(result.text, encoding="utf-8")
                    output_paths["txt"] = p
                elif fmt == "lrc":
                    p = out_dir / f"{base}.lrc"
                    p.write_text(_format_lrc(result), encoding="utf-8")
                    output_paths["lrc"] = p
                elif fmt == "srt":
                    p = out_dir / f"{base}.srt"
                    p.write_text(_format_srt(result), encoding="utf-8")
                    output_paths["srt"] = p
            except OSError as exc:
                raise AudioProcessingError(
                    f"Failed to write {fmt} output to {out_dir}: {exc}",
                    path=str(out_dir),
                ) from exc

        if progress_cb:
            progress_cb(1.0, "Done")

        return TranscribeResult(
            result=result,
            output_paths=output_paths,
            label=f"{stem} ({engine_tag})",
        )

    def clear(self) -> None:
        if self._engine is not None:
            self._engine.clear()
            self._engine = None


# ── Format helpers ───────────────────────────────────────────────────


def _format_lrc(result: TranscriptionResult) -> str:
    """LRC karaoke format.

    With word timestamps: one line per word with [mm:ss.xx] prefix.
    Without word timestamps (Qwen): one line per text-line with the
    segment-start timestamp.
    """
    lines: list[str] = []
    if result.has_word_timestamps:
        for seg in result.segments:
            for w in seg.words:
                lines.append(f"[{_lrc_timestamp(w.start)}]{w.word}")
    else:
        for seg in result.segments:
            for line in seg.text.splitlines():
                if line.strip():
                    lines.append(f"[{_lrc_timestamp(seg.start)}]{line.strip()}")
    return "\n".join(lines) + "\n"


def _format_srt(result: TranscriptionResult) -> str:
    """SRT subtitle format.  Segment-level — never per-word."""
    lines: list[str] = []
    for i, seg in enumerate(result.segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def _lrc_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
