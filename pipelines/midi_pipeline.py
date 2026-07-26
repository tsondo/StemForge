"""
MIDI generation pipeline for StemForge.

Supports three generation modes, selected automatically based on the
inputs passed to :meth:`MidiPipeline.run`:

* **Stems only** — each audio stem is transcribed via BasicPitch and
  assembled into a multi-track MIDI file.
* **Text only** — a chord progression is generated from the supplied
  musical parameters (key, BPM, time signature, duration) without any
  audio input.
* **Hybrid** — stems are transcribed and the text prompt is used as
  optional conditioning metadata; key filtering is applied afterwards.

All three modes produce a single format-1 multi-track MIDI file in which
every track shares the same tempo map and time-signature, making the
output drop-in ready for any DAW.

Typical usage
-------------
::

    pipeline = MidiPipeline()
    pipeline.configure(MidiConfig(key="A minor", bpm=90.0, time_signature="3/4"))
    pipeline.load_model()
    result = pipeline.run({"Singing voice": vocals_path, "Bass": bass_path})
    pipeline.clear()
"""

import pathlib
import logging
import time
from typing import Any, Callable

from models.midi_loader import MidiModelLoader
from utils.midi_io import (
    NoteEvent, LyricEvent, MidiData, DRUM_STEM_LABELS,
    notes_to_midi, write_midi, merge_tracks, generate_chord_progression,
)
from utils.errors import (
    AudioProcessingError, InvalidInputError,
    ModelLoadError, PipelineExecutionError,
)


log = logging.getLogger("stemforge.pipelines.midi")

_TICKS_PER_BEAT = 480   # standard MIDI resolution

# Stem labels that represent sung vocals — routed to faster-whisper + PYIN
# instead of BasicPitch for better pitch tracking on the human voice.
_VOCAL_STEM_LABELS: frozenset[str] = frozenset({
    "vocals", "Singing voice",
})

# Drum/percussion stem labels are routed to ADTOF drum transcription
# instead of BasicPitch; the canonical label set lives in utils.midi_io.


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class MidiConfig:
    """Immutable configuration snapshot for a single MIDI generation job.

    Parameters
    ----------
    prompt:
        Optional natural-language description of the desired musical style.
        Used as conditioning metadata; applied as a key constraint when
        *key* is ``'Any'`` and a musical key can be inferred.
    duration_seconds:
        Target output length in seconds.  For stem-based mode, notes beyond
        this point are clipped.  For text-only mode, the progression is
        extended to fill this duration.
    key:
        Musical key for pitch snapping (e.g. ``'C major'``, ``'A minor'``).
        ``'Any'`` disables key filtering.
    time_signature:
        Time-signature string (e.g. ``'4/4'``, ``'3/4'``, ``'6/8'``).
    bpm:
        Tempo in beats per minute.  Clipped to ``[20, 300]``.
    onset_threshold:
        BasicPitch onset confidence threshold ``[0, 1]``.
        Higher = fewer false positives.
    frame_threshold:
        BasicPitch frame confidence threshold ``[0, 1]``.
        Higher = shorter detected note durations.
    minimum_note_length:
        Minimum note duration in milliseconds (BasicPitch post-filter).
    output_dir:
        Directory where generated MIDI files are written.  Created
        automatically if absent.
    """

    prompt: str | None
    duration_seconds: float
    key: str
    time_signature: str
    bpm: float
    onset_threshold: float
    frame_threshold: float
    minimum_note_length: float
    output_dir: pathlib.Path | None

    def __init__(
        self,
        prompt: str | None = None,
        duration_seconds: float = 10.0,
        key: str = "Any",
        time_signature: str = "4/4",
        bpm: float = 120.0,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length: float = 58.0,
        output_dir: pathlib.Path | None = None,
    ) -> None:
        self.prompt = prompt.strip() if prompt else None
        self.duration_seconds = max(1.0, float(duration_seconds)) if float(duration_seconds) > 0 else 0.0
        self.key = key
        self.time_signature = time_signature
        self.bpm = max(20.0, min(300.0, float(bpm)))
        self.onset_threshold = float(onset_threshold)
        self.frame_threshold = float(frame_threshold)
        self.minimum_note_length = float(minimum_note_length)
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class MidiResult:
    """Artefacts produced by a completed MIDI generation job.

    Parameters
    ----------
    merged_midi_data:
        In-memory PrettyMIDI object containing all tracks merged.  Not
        written to disk automatically; call write_stem_midi() to save.
    stem_midi_data:
        Per-stem PrettyMIDI objects ``{display_label: PrettyMIDI}``.
        Empty when operating in text-only mode.  Files are NOT written
        automatically; call write_stem_midi() on individual objects to save.
    note_counts:
        Number of notes detected (or generated) per track
        ``{stem_name: count}``.
    total_notes:
        Sum of all per-track note counts.
    """

    merged_midi_data: Any
    stem_midi_data: dict[str, Any]
    note_counts: dict[str, int]
    total_notes: int

    def __init__(
        self,
        merged_midi_data: Any,
        stem_midi_data: dict[str, Any],
        note_counts: dict[str, int],
        total_notes: int,
    ) -> None:
        self.merged_midi_data = merged_midi_data
        self.stem_midi_data = dict(stem_midi_data)
        self.note_counts = dict(note_counts)
        self.total_notes = int(total_notes)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MidiPipeline:
    """Complete MIDI generation pipeline.

    Orchestrates stem transcription (via :class:`~models.midi_loader.MidiModelLoader`),
    optional chord-progression generation, key filtering, multi-track
    merging, and file writing behind a uniform
    ``configure → load_model → run → clear`` lifecycle.

    The loader wraps BasicPitch and is shared across all ``run()`` calls;
    calling ``clear()`` releases TF memory without disturbing the PyTorch
    GPU context used by Demucs.
    """

    is_loaded: bool
    _config: MidiConfig | None
    _loader: MidiModelLoader | None
    _progress_callback: Callable[[float], None] | None

    def __init__(self) -> None:
        self.is_loaded = False
        self._config = None
        self._loader = None
        self._progress_callback = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, config: MidiConfig) -> None:
        """Store *config* for the next :meth:`run` call.

        Safe to call at any point; does not require reloading the model
        because all MidiConfig parameters are applied at inference time.
        """
        self._config = config

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the BasicPitch model into memory.

        Raises
        ------
        :class:`~utils.errors.PipelineExecutionError`
            If :meth:`configure` has not been called.
        :class:`~utils.errors.ModelLoadError`
            If TensorFlow or basic-pitch are unavailable.
        """
        if self._config is None:
            raise PipelineExecutionError(
                "configure() must be called before load_model().",
                pipeline_name="midi",
            )
        if self._loader is None:
            self._loader = MidiModelLoader()
        try:
            self._loader.load()
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Unexpected error loading MIDI model: {exc}",
                model_name="basicpitch",
            ) from exc
        self.is_loaded = True
        log.info("MidiPipeline: model ready.")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, stems: dict[str, pathlib.Path]) -> "MidiResult":
        """Generate MIDI from *stems* and/or a text prompt.

        Parameters
        ----------
        stems:
            ``{display_label: audio_path}`` for each stem to transcribe.
            Pass an empty dict to use text-only generation (requires
            ``config.prompt`` to be non-empty).

        Returns
        -------
        MidiResult
            Paths to written files and per-track note counts.

        Raises
        ------
        :class:`~utils.errors.PipelineExecutionError`
            If neither stems nor a prompt are provided, or if
            :meth:`load_model` was not called.
        :class:`~utils.errors.InvalidInputError`
            If a stem audio path does not exist.
        :class:`~utils.errors.AudioProcessingError`
            If a stem file cannot be read or output cannot be written.
        """
        if not self.is_loaded:
            raise PipelineExecutionError(
                "load_model() must be called before run().",
                pipeline_name="midi",
            )
        if self._config is None:
            raise PipelineExecutionError(
                "configure() must be called before run().",
                pipeline_name="midi",
            )

        cfg = self._config
        has_stems = bool(stems)
        has_prompt = bool(cfg.prompt)

        if not has_stems and not has_prompt:
            raise InvalidInputError(
                "Provide at least one stem or a text prompt.",
                field="stems / prompt",
            )

        # Validate all stem paths up-front — fail fast before any inference.
        for label, path in stems.items():
            if not path.exists():
                raise InvalidInputError(
                    f"Stem file not found: {path} ('{label}')",
                    field="stems",
                )

        self._report(2.0)

        track_notes: dict[str, list[NoteEvent]] = {}
        track_lyrics: dict[str, list[LyricEvent]] = {}
        stem_midi_data: dict[str, Any] = {}

        if has_stems:
            total = len(stems)
            for i, (label, path) in enumerate(stems.items()):
                base_pct = 5.0 + (i / total) * 70.0
                self._report(base_pct)
                log.info("MidiPipeline: transcribing '%s' (%s)…", label, path.name)

                if label in _VOCAL_STEM_LABELS:
                    log.info("MidiPipeline: routing '%s' to vocal path.", label)
                    notes, lyrics = self._loader.convert_vocal_to_midi(
                        path,
                        duration=cfg.duration_seconds,
                        key=cfg.key,
                        bpm=cfg.bpm,
                    )
                    if lyrics:
                        track_lyrics[label] = lyrics
                elif label in DRUM_STEM_LABELS:
                    log.info("MidiPipeline: routing '%s' to drum path.", label)
                    notes = self._loader.convert_drum_to_midi(
                        path,
                        duration=cfg.duration_seconds,
                    )
                else:
                    notes = self._loader.convert_audio_to_midi(
                        path,
                        prompt=cfg.prompt,
                        duration=cfg.duration_seconds,
                        key=cfg.key,
                        signature=cfg.time_signature,
                        bpm=cfg.bpm,
                        onset_threshold=cfg.onset_threshold,
                        frame_threshold=cfg.frame_threshold,
                        minimum_note_length=cfg.minimum_note_length,
                    )
                track_notes[label] = notes

                # Build per-stem MIDI object in memory (not written to disk yet)
                stem_lyrics = track_lyrics.get(label)
                stem_midi_data[label] = self._build_stem_midi(
                    label, notes, cfg, stem_lyrics,
                    is_drum=(label in DRUM_STEM_LABELS),
                )

                self._report(base_pct + (1.0 / total) * 70.0)

            # Free the ADTOF model's GPU memory once all drum stems are done.
            if any(label in DRUM_STEM_LABELS for label in stems):
                self._loader.evict_drum_model()

        else:
            # Text-only: generate a diatonic chord progression
            self._report(20.0)
            log.info(
                "MidiPipeline: text-only mode — generating chord progression "
                "(key=%r, bpm=%.1f, time_sig=%r, duration=%.1f s).",
                cfg.key, cfg.bpm, cfg.time_signature, cfg.duration_seconds,
            )
            notes = generate_chord_progression(
                cfg.key,
                cfg.bpm,
                cfg.time_signature,
                cfg.duration_seconds,
            )
            track_notes["generated"] = notes
            self._report(60.0)

        # Merge all tracks into a single format-1 MIDI file
        self._report(80.0)
        merged = merge_tracks(
            track_notes,
            bpm=cfg.bpm,
            time_signature=cfg.time_signature,
            ticks_per_beat=_TICKS_PER_BEAT,
            track_lyrics=track_lyrics or None,
        )

        note_counts = {name: len(notes) for name, notes in track_notes.items()}
        total_notes = sum(note_counts.values())

        self._report(100.0)
        log.info(
            "MidiPipeline: %d track(s), %d notes total (in memory)",
            len(track_notes), total_notes,
        )
        return MidiResult(
            merged_midi_data=merged,
            stem_midi_data=stem_midi_data,
            note_counts=note_counts,
            total_notes=total_notes,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Release the model from memory and reset pipeline state.

        Safe to call even if no model has been loaded.  Releases the TF
        session and triggers GC, freeing RAM without touching the PyTorch
        GPU context.
        """
        if self._loader is not None:
            self._loader.evict()
        self.is_loaded = False

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    def set_progress_callback(self, callback: Callable[[float], None]) -> None:
        """Register *callback* invoked with a ``[0, 100]`` percent value."""
        self._progress_callback = callback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _report(self, pct: float) -> None:
        if self._progress_callback is not None:
            self._progress_callback(pct)

    def _build_stem_midi(
        self,
        stem_name: str,
        notes: list[NoteEvent],
        cfg: MidiConfig,
        lyrics: list[LyricEvent] | None = None,
        is_drum: bool = False,
    ) -> Any:
        """Build and return a PrettyMIDI object for *stem_name* (no disk write)."""
        return notes_to_midi(
            notes, ticks_per_beat=_TICKS_PER_BEAT, tempo_bpm=cfg.bpm,
            lyrics=lyrics, is_drum=is_drum,
        )

    def write_stem_midi(
        self,
        stem_name: str,
        midi_obj: Any,
        cfg: MidiConfig,
    ) -> pathlib.Path:
        """Write *midi_obj* to *cfg.output_dir* and return the absolute path.

        Call this explicitly when the user requests a save (e.g. via the
        "Save as" button in the MIDI panel).  The pipeline itself no longer
        writes per-stem files automatically.
        """
        out_dir = cfg.output_dir
        if out_dir is None:
            out_dir = pathlib.Path.home() / ".local/share/stemforge/output/midi"
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_name = stem_name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_name}_{int(time.time())}.mid"
        path = out_dir / filename
        try:
            write_midi(midi_obj, path)
        except Exception as exc:
            raise AudioProcessingError(
                f"Failed to write stem MIDI for '{stem_name}': {exc}",
                path=str(path),
            ) from exc
        return path.resolve()

