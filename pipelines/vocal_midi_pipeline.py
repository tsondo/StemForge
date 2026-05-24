"""
Vocal MIDI pipeline for StemForge.

Converts an audio file (typically an ACE-Step render) into a quantised MIDI
file with embedded lyric meta-events by chaining four processing stages:

1. **Demucs** — isolates the vocal stem from the full mix.
2. **faster-whisper** — produces word-level timestamps from the vocal stem.
3. **BasicPitch** — extracts raw MIDI note events from the vocal stem.
4. **librosa + pretty_midi** — quantises notes to the BPM grid and assembles
   a Standard MIDI File with ``LYRIC`` meta-events aligned to the word timestamps.

When an ACE-Step JSON file is provided alongside the audio, BPM, key, and
structured lyrics are read directly from the JSON rather than being inferred.

Typical usage
-------------
::

    pipeline = VocalMidiPipeline()
    pipeline.configure(VocalMidiConfig(
        json_path=pathlib.Path("take_01.json"),
        whisper_model_size="base",
    ))
    pipeline.load_model()
    result = pipeline.run(pathlib.Path("take_01.wav"))
    pipeline.clear()
"""

import json
import logging
import pathlib
import tempfile
from typing import Any, Callable, TypeAlias

import numpy as np
import soundfile as sf
import torch

from utils.errors import (
    AudioProcessingError,
    InvalidInputError,
    ModelLoadError,
    PipelineExecutionError,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# (start_sec, end_sec, pitch_midi, velocity)
NoteEvent: TypeAlias = tuple[float, float, int, int]

# (word, start_sec, end_sec) — one entry per whisper word token
WordTiming: TypeAlias = tuple[str, float, float]

# Supported audio extensions for input validation
_SUPPORTED_EXTENSIONS = frozenset({".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"})


# ---------------------------------------------------------------------------
# ACE-Step metadata
# ---------------------------------------------------------------------------

class AceStepMetadata:
    """Structured representation of an ACE-Step generation JSON.

    ACE-Step JSON files contain production metadata that makes quantised MIDI
    assembly significantly more accurate than heuristic estimation:

    * ``bpm`` — exact tempo used during generation (integer field ``"bpm"``).
    * ``key`` — tonic and mode from ``"keyscale"`` (e.g. ``"C minor"``).
    * ``time_signature`` — resolved from ``"timesignature"`` with fallback to
      ``"cot_timesignature"``; bare numerator strings like ``"4"`` are
      normalised to ``"4/4"``.
    * ``lyrics`` — flat string from the ``"lyrics"`` field containing
      ``[section]`` markers and lyric lines separated by newlines.
    * ``caption`` — style description string from ``"caption"``, usable as a
      Whisper ``initial_prompt`` to improve transcription accuracy.
    * ``duration`` — total audio duration in seconds from ``"duration"``.

    Parameters
    ----------
    bpm:
        Beats per minute.
    key:
        Key string (e.g. ``"C minor"``), or ``None`` if absent.
    time_signature:
        Normalised time signature (e.g. ``"4/4"``), or ``None`` if absent.
    lyrics:
        Raw lyrics string with ``[section]`` markers.
    caption:
        Style/mood description; useful as a Whisper prompt.
    duration:
        Total duration of the generated audio in seconds, or ``None``.
    """

    bpm: float
    key: str | None
    time_signature: str | None
    lyrics: str
    caption: str | None
    duration: float | None

    def __init__(
        self,
        bpm: float,
        key: str | None = None,
        time_signature: str | None = None,
        lyrics: str = "",
        caption: str | None = None,
        duration: float | None = None,
    ) -> None:
        self.bpm = bpm
        self.key = key
        self.time_signature = time_signature
        self.lyrics = lyrics
        self.caption = caption
        self.duration = duration


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class VocalMidiConfig:
    """Immutable configuration snapshot for a single VocalMidiPipeline job.

    Parameters
    ----------
    json_path:
        Path to the ACE-Step JSON file that accompanies the audio.  When
        provided, BPM, key, and structured lyrics are read from the JSON
        rather than estimated.  ``None`` is allowed; in that case *bpm* must
        be supplied explicitly.
    output_path:
        Destination path for the assembled ``.mid`` file.  When ``None`` the
        file is written alongside the input audio with the same stem and a
        ``".mid"`` suffix.
    whisper_model_size:
        faster-whisper model size identifier (e.g. ``"tiny"``, ``"base"``,
        ``"small"``, ``"medium"``).  Larger models are more accurate at the
        cost of load time and memory.  Default: ``"base"``.
    whisper_device:
        Compute device for faster-whisper — ``"cpu"`` or ``"cuda"``.
        Default: ``"cpu"``.
    whisper_compute_type:
        CTranslate2 quantisation mode.  ``"int8"`` is the recommended default
        for CPU; ``"float16"`` is preferred on CUDA.
    demucs_model:
        Demucs model variant used for vocal isolation.  Must be one of the
        identifiers accepted by :func:`demucs.pretrained.get_model`.
        Default: ``"htdemucs"``.
    bpm:
        Override tempo in beats per minute.  Required when *json_path* is
        ``None``; ignored when a valid JSON is supplied.
    key:
        Override tonic and mode string (e.g. ``"A major"``).  Optional even
        when *json_path* is ``None`` — omitting it means no key signature is
        written to the MIDI file.
    min_note_length_ms:
        BasicPitch post-processing parameter: notes shorter than this value
        (in milliseconds) are discarded.  Default: ``58.0``.
    onset_threshold:
        BasicPitch onset detection confidence threshold in ``[0.0, 1.0]``.
        Default: ``0.5``.
    frame_threshold:
        BasicPitch frame sustain confidence threshold in ``[0.0, 1.0]``.
        Default: ``0.3``.
    """

    json_path: pathlib.Path | None
    output_path: pathlib.Path | None
    whisper_model_size: str
    whisper_device: str
    whisper_compute_type: str
    demucs_model: str
    bpm: float | None
    key: str | None
    min_note_length_ms: float
    onset_threshold: float
    frame_threshold: float

    def __init__(
        self,
        json_path: pathlib.Path | None = None,
        output_path: pathlib.Path | None = None,
        whisper_model_size: str = "base",
        whisper_device: str = "cpu",
        whisper_compute_type: str = "int8",
        demucs_model: str = "htdemucs",
        bpm: float | None = None,
        key: str | None = None,
        min_note_length_ms: float = 58.0,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
    ) -> None:
        self.json_path = json_path
        self.output_path = output_path
        self.whisper_model_size = whisper_model_size
        self.whisper_device = whisper_device
        self.whisper_compute_type = whisper_compute_type
        self.demucs_model = demucs_model
        self.bpm = bpm
        self.key = key
        self.min_note_length_ms = min_note_length_ms
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class VocalMidiResult:
    """Artefacts produced by a completed VocalMidiPipeline job.

    Parameters
    ----------
    midi_path:
        Absolute path of the written Standard MIDI File (``.mid``).
    bpm:
        Tempo used for quantisation (sourced from JSON or config override).
    key:
        Key signature string written to the MIDI file, or ``None`` if no key
        was available.
    note_count:
        Total number of note events in the assembled MIDI file.
    word_count:
        Total number of LYRIC meta-events embedded in the MIDI file.
    duration_seconds:
        Duration of the longest note event in seconds, as a proxy for the
        transcribed audio length.
    """

    midi_path: pathlib.Path
    bpm: float
    key: str | None
    note_count: int
    word_count: int
    duration_seconds: float

    def __init__(
        self,
        midi_path: pathlib.Path,
        bpm: float,
        key: str | None,
        note_count: int,
        word_count: int,
        duration_seconds: float,
    ) -> None:
        self.midi_path = midi_path
        self.bpm = bpm
        self.key = key
        self.note_count = note_count
        self.word_count = word_count
        self.duration_seconds = duration_seconds


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class VocalMidiPipeline:
    """Composite pipeline that converts an audio file to a lyric-annotated MIDI.

    Unlike the single-model pipelines, this class owns and manages *three*
    models simultaneously — Demucs, faster-whisper, and BasicPitch —
    coordinating them in a fixed four-stage execution graph.

    Lifecycle
    ---------
    1. ``pipeline = VocalMidiPipeline()``
    2. ``pipeline.configure(config)`` — supply a :class:`VocalMidiConfig`.
    3. ``pipeline.load_model()`` — load all three model weights.
    4. ``result = pipeline.run(audio_path)`` — process one audio file end-to-end.
    5. ``pipeline.clear()`` — release all model weights and temp files.

    Execution graph
    ---------------
    ``run(audio_path)``
      → :meth:`_parse_json`          parse ACE-Step metadata (if json_path set)
      → :meth:`_separate_vocals`     Demucs: extract vocal stem → temp WAV
      → :meth:`_transcribe`          faster-whisper: word-level timestamps
      → :meth:`_extract_notes`       BasicPitch: raw note events from vocal stem
      → :meth:`_quantize_to_grid`    librosa: snap note edges to BPM beat grid
      → :meth:`_assemble_midi`       pretty_midi: build MIDI + LYRIC meta-events
      → write ``.mid`` → return :class:`VocalMidiResult`
    """

    is_loaded: bool
    _config: VocalMidiConfig | None
    _transcribe_engine: Any
    _demucs_model: Any
    _basicpitch_model: Any
    _tmp_dir: tempfile.TemporaryDirectory | None  # type: ignore[type-arg]
    _progress_callback: Callable[[float, str], None] | None

    def __init__(self) -> None:
        self.is_loaded = False
        self._config = None
        self._transcribe_engine = None
        self._demucs_model = None
        self._basicpitch_model = None
        self._tmp_dir = None
        self._progress_callback = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, config: VocalMidiConfig) -> None:
        """Set or replace the pipeline configuration.

        Raises
        ------
        :class:`~utils.errors.InvalidInputError`
            If both ``config.json_path`` and ``config.bpm`` are ``None``
            (tempo cannot be determined for quantisation).
        """
        if config.json_path is None and config.bpm is None:
            raise InvalidInputError(
                "Either json_path or bpm must be set — tempo is required for "
                "quantisation.",
                field="bpm",
            )
        self._config = config

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load Demucs, faster-whisper, and BasicPitch weights into memory.

        Raises
        ------
        :class:`~utils.errors.PipelineExecutionError`
            If :meth:`configure` has not been called before this method.
        :class:`~utils.errors.ModelLoadError`
            If any of the three model checkpoints cannot be loaded.
        """
        if self._config is None:
            raise PipelineExecutionError(
                "configure() must be called before load_model().",
                pipeline_name="VocalMidiPipeline",
            )

        try:
            from demucs.pretrained import get_model
            log.info("Loading Demucs model '%s'…", self._config.demucs_model)
            self._demucs_model = get_model(self._config.demucs_model)
            self._demucs_model.eval()
            log.info("Demucs loaded — sources: %s", list(self._demucs_model.sources))
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Demucs model '{self._config.demucs_model}': {exc}",
                model_name=self._config.demucs_model,
            ) from exc

        from pipelines.transcribe_engines import WhisperEngine

        whisper_model_id = f"whisper-{self._config.whisper_model_size}"
        log.info("Loading transcribe engine for %s…", whisper_model_id)
        try:
            self._transcribe_engine = WhisperEngine(model_id=whisper_model_id)
            self._transcribe_engine.load()
            log.info("Transcribe engine loaded.")
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Whisper engine for {whisper_model_id}: {exc}",
                model_name=whisper_model_id,
            ) from exc

        # BasicPitch uses TensorFlow. Force CPU mode:
        # This TF build has no precompiled CUDA kernels for compute capability
        # ≥ 12.0 and ptxas is unavailable for JIT fallback, so GPU execution
        # fails. We apply both approaches so it works regardless of whether TF
        # has already been initialised transitively by another import:
        #   1. Env var (effective if TF has not yet been imported)
        #   2. tf.config.set_visible_devices (effective at runtime)
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
            log.debug("TensorFlow GPU devices hidden for BasicPitch CPU inference.")
        except Exception as tf_exc:
            log.debug("Could not configure TF devices (may already be CPU-only): %s", tf_exc)

        try:
            from basic_pitch import ICASSP_2022_MODEL_PATH
            self._basicpitch_model = ICASSP_2022_MODEL_PATH
            log.info("BasicPitch model path resolved: %s", self._basicpitch_model)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to resolve BasicPitch model path: {exc}",
                model_name="basic-pitch",
            ) from exc

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="stemforge_vocal_")
        self.is_loaded = True
        log.info("VocalMidiPipeline ready.")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, input_data: pathlib.Path) -> VocalMidiResult:
        """Process *input_data* end-to-end and return a lyric-annotated MIDI.

        Raises
        ------
        :class:`~utils.errors.PipelineExecutionError`
            If :meth:`load_model` has not been called successfully.
        :class:`~utils.errors.InvalidInputError`
            If *input_data* does not exist or has an unsupported file extension,
            or if ``config.json_path`` is set but the file cannot be parsed.
        :class:`~utils.errors.AudioProcessingError`
            If reading the audio or writing an intermediate stem file fails.
        """
        if not self.is_loaded:
            raise PipelineExecutionError(
                "load_model() must be called before run().",
                pipeline_name="VocalMidiPipeline",
            )

        if not input_data.exists():
            raise InvalidInputError(
                f"Audio file not found: {input_data}",
                field="input_data",
            )
        if input_data.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise InvalidInputError(
                f"Unsupported audio format '{input_data.suffix}'. "
                f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
                field="input_data",
            )

        # Resolve metadata and tempo ----------------------------------------
        metadata: AceStepMetadata | None = None
        if self._config.json_path is not None:
            metadata = self._parse_json(self._config.json_path)

        bpm: float = (
            metadata.bpm if metadata is not None else float(self._config.bpm)  # type: ignore[arg-type]
        )
        key: str | None = (
            metadata.key if metadata is not None else self._config.key
        )

        log.info("VocalMidiPipeline.run: bpm=%.1f key=%s file=%s", bpm, key, input_data)

        # Stage 1 — vocal separation ----------------------------------------
        self._report_progress(0.0, "Separating vocals")
        vocal_path = self._separate_vocals(input_data)

        # Stage 2 — transcription -------------------------------------------
        self._report_progress(25.0, "Transcribing")
        whisper_prompt = metadata.caption if metadata is not None else None
        words = self._transcribe(vocal_path, initial_prompt=whisper_prompt)
        log.info("Transcription: %d word(s) detected.", len(words))

        # Stage 3 — note extraction -----------------------------------------
        self._report_progress(50.0, "Extracting notes")
        notes = self._extract_notes(vocal_path)
        log.info("BasicPitch: %d raw note event(s).", len(notes))

        # Stage 4 — quantisation --------------------------------------------
        self._report_progress(75.0, "Quantising to grid")
        quantised = self._quantize_to_grid(notes, bpm)

        # Stage 5 — MIDI assembly -------------------------------------------
        self._report_progress(90.0, "Assembling MIDI")
        output_path = self._config.output_path or input_data.with_suffix(".mid")
        self._assemble_midi(quantised, words, metadata, output_path)

        self._report_progress(100.0, "Done")
        log.info("MIDI written to %s", output_path)

        duration = max((n[1] for n in quantised), default=0.0)
        return VocalMidiResult(
            midi_path=output_path.resolve(),
            bpm=bpm,
            key=key,
            note_count=len(quantised),
            word_count=len(words),
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Release all model weights and temporary files."""
        self._demucs_model = None
        if self._transcribe_engine is not None:
            self._transcribe_engine.clear()
            self._transcribe_engine = None
        self._basicpitch_model = None
        if self._tmp_dir is not None:
            try:
                self._tmp_dir.cleanup()
            except Exception:
                pass
            self._tmp_dir = None
        self.is_loaded = False
        log.info("VocalMidiPipeline cleared.")

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Register a callback invoked at the start of each pipeline stage.

        Parameters
        ----------
        callback:
            ``callback(percent: float, stage: str)`` where *percent* is in
            ``[0.0, 100.0]`` and *stage* is a human-readable label.
        """
        self._progress_callback = callback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _report_progress(self, percent: float, stage: str) -> None:
        log.debug("Stage [%.0f%%]: %s", percent, stage)
        if self._progress_callback is not None:
            self._progress_callback(percent, stage)

    def _parse_json(self, json_path: pathlib.Path) -> AceStepMetadata:
        """Parse an ACE-Step JSON file and return structured metadata.

        Raises
        ------
        :class:`~utils.errors.InvalidInputError`
            If *json_path* does not exist, cannot be decoded, or has no
            ``"bpm"`` field.
        """
        if not json_path.exists():
            raise InvalidInputError(
                f"ACE-Step JSON not found: {json_path}",
                field="json_path",
            )
        try:
            data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvalidInputError(
                f"Cannot parse JSON at {json_path}: {exc}",
                field="json_path",
            ) from exc

        if "bpm" not in data:
            raise InvalidInputError(
                f"JSON at {json_path} has no 'bpm' field.",
                field="bpm",
            )

        bpm = float(data["bpm"])

        key: str | None = data.get("keyscale") or None

        # time_signature may be empty string; fall back to cot_timesignature
        raw_ts = data.get("timesignature") or data.get("cot_timesignature") or None
        if raw_ts and str(raw_ts).isdigit():
            time_signature: str | None = f"{raw_ts}/4"
        else:
            time_signature = str(raw_ts) if raw_ts else None

        lyrics: str = data.get("lyrics") or ""
        caption: str | None = data.get("caption") or None
        duration: float | None = float(data["duration"]) if data.get("duration") else None

        log.info(
            "Parsed ACE-Step JSON: bpm=%.1f key=%r ts=%r duration=%s",
            bpm, key, time_signature, duration,
        )
        return AceStepMetadata(
            bpm=bpm,
            key=key,
            time_signature=time_signature,
            lyrics=lyrics,
            caption=caption,
            duration=duration,
        )

    def _separate_vocals(self, audio_path: pathlib.Path) -> pathlib.Path:
        """Run Demucs on *audio_path* and return the path to the vocal stem.

        The stem WAV is written into the pipeline's temporary directory.

        Raises
        ------
        :class:`~utils.errors.AudioProcessingError`
            If loading or writing audio fails.
        :class:`~utils.errors.PipelineExecutionError`
            If the model does not expose a ``vocals`` source.
        """
        from demucs.apply import apply_model
        from demucs.audio import convert_audio

        if "vocals" not in list(self._demucs_model.sources):
            raise PipelineExecutionError(
                f"Demucs model has no 'vocals' source. "
                f"Available: {list(self._demucs_model.sources)}",
                pipeline_name="VocalMidiPipeline",
            )

        # Load audio with soundfile → torch tensor
        try:
            audio_np, sr = sf.read(str(audio_path), always_2d=True)  # (samples, ch)
        except Exception as exc:
            raise AudioProcessingError(
                f"Cannot read audio file {audio_path}: {exc}",
                path=str(audio_path),
            ) from exc

        wav = torch.from_numpy(audio_np.T.astype(np.float32))  # (ch, samples)

        # Resample + channel-match to model's requirements
        wav = convert_audio(
            wav,
            sr,
            self._demucs_model.samplerate,
            self._demucs_model.audio_channels,
        )
        wav = wav.unsqueeze(0)  # (1, ch, samples)

        log.info(
            "Running Demucs on %s  [%d Hz, %.1fs]…",
            audio_path.name,
            self._demucs_model.samplerate,
            wav.shape[-1] / self._demucs_model.samplerate,
        )

        with torch.no_grad():
            sources = apply_model(
                self._demucs_model, wav, split=True, overlap=0.25, progress=False
            )
        # sources: (batch, n_sources, channels, samples)

        vocal_idx = list(self._demucs_model.sources).index("vocals")
        vocal = sources[0, vocal_idx].cpu()  # (channels, samples)

        vocal_path = pathlib.Path(self._tmp_dir.name) / "vocals.wav"  # type: ignore[union-attr]
        try:
            sf.write(
                str(vocal_path),
                vocal.numpy().T,  # (samples, channels)
                self._demucs_model.samplerate,
            )
        except Exception as exc:
            raise AudioProcessingError(
                f"Failed to write vocal stem to {vocal_path}: {exc}",
                path=str(vocal_path),
            ) from exc

        log.info("Vocal stem written to %s", vocal_path)
        return vocal_path

    def _transcribe(
        self,
        vocal_path: pathlib.Path,
        initial_prompt: str | None = None,
    ) -> list[WordTiming]:
        """Run the shared WhisperEngine on *vocal_path* and return word-level timestamps.

        Parameters
        ----------
        vocal_path:
            Path to the isolated vocal stem WAV.
        initial_prompt:
            Optional text fed to Whisper as context (e.g. the ACE-Step
            ``caption`` field).

        Returns
        -------
        list[WordTiming]
            ``(word, start_sec, end_sec)`` tuples, one per word token.
        """
        result = self._transcribe_engine.transcribe(
            vocal_path, prompt=initial_prompt,
        )
        log.info(
            "Whisper: language=%r, segments=%d",
            result.language, len(result.segments),
        )

        words: list[WordTiming] = []
        for segment in result.segments:
            for w in segment.words:
                words.append((w.word, w.start, w.end))
        return words

    def _extract_notes(self, vocal_path: pathlib.Path) -> list[NoteEvent]:
        """Run BasicPitch on *vocal_path* and return raw note events.

        Returns
        -------
        list[NoteEvent]
            ``(start_sec, end_sec, pitch_midi, velocity)`` tuples sorted by
            start time.

        Raises
        ------
        :class:`~utils.errors.PipelineExecutionError`
            If BasicPitch raises during inference.
        """
        from basic_pitch.inference import predict

        try:
            model_output, _midi, note_events = predict(
                vocal_path,
                model_or_model_path=self._basicpitch_model,
                onset_threshold=self._config.onset_threshold,  # type: ignore[union-attr]
                frame_threshold=self._config.frame_threshold,  # type: ignore[union-attr]
                minimum_note_length=self._config.min_note_length_ms,  # type: ignore[union-attr]
                minimum_frequency=None,
                maximum_frequency=None,
            )
        except Exception as exc:
            raise PipelineExecutionError(
                f"BasicPitch inference failed: {exc}",
                pipeline_name="VocalMidiPipeline",
            ) from exc

        # note_events: list of (start_s, end_s, pitch_midi, amplitude, pitch_bends)
        result: list[NoteEvent] = []
        for start, end, pitch, amplitude, _bends in note_events:
            velocity = max(1, min(127, int(round(float(amplitude) * 127))))
            result.append((float(start), float(end), int(pitch), velocity))

        return sorted(result, key=lambda n: n[0])

    def _quantize_to_grid(
        self,
        notes: list[NoteEvent],
        bpm: float,
    ) -> list[NoteEvent]:
        """Snap note start and end times to the nearest sixteenth-note boundary.

        Parameters
        ----------
        notes:
            Raw note events from :meth:`_extract_notes`.
        bpm:
            Tempo in beats per minute used to define the quantisation grid.

        Returns
        -------
        list[NoteEvent]
            Quantised note events.  Notes whose quantised duration would be
            zero or negative are extended to one sixteenth-note minimum.
        """
        if not notes:
            return []

        beat_s = 60.0 / bpm
        sixteenth_s = beat_s / 4.0

        def snap(t: float) -> float:
            return round(t / sixteenth_s) * sixteenth_s

        result: list[NoteEvent] = []
        for start, end, pitch, velocity in notes:
            q_start = snap(start)
            q_end = snap(end)
            if q_end <= q_start:
                q_end = q_start + sixteenth_s
            result.append((q_start, q_end, pitch, velocity))

        return result

    def _assemble_midi(
        self,
        notes: list[NoteEvent],
        words: list[WordTiming],
        metadata: AceStepMetadata | None,
        output_path: pathlib.Path,
    ) -> None:
        """Build and write a Standard MIDI File from notes and word timings.

        Creates a :class:`pretty_midi.PrettyMIDI` object at the target BPM,
        optionally sets a key-signature event, populates a vocal instrument
        track (GM program 53 — Voice Oohs), embeds each word as a ``LYRIC``
        meta-event, and writes the file to *output_path*.

        Raises
        ------
        :class:`~utils.errors.AudioProcessingError`
            If writing the MIDI file fails.
        """
        import pretty_midi

        bpm = metadata.bpm if metadata is not None else float(self._config.bpm)  # type: ignore[union-attr]
        key = metadata.key if metadata is not None else self._config.key  # type: ignore[union-attr]

        pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))

        # Key signature (optional)
        if key:
            try:
                key_num = pretty_midi.key_name_to_key_number(key)
                pm.key_signature_changes.append(
                    pretty_midi.KeySignature(key_num, 0.0)
                )
                log.debug("Key signature set: %s (key_number=%d)", key, key_num)
            except Exception as exc:
                log.warning("Could not set key signature '%s': %s", key, exc)

        # Instrument — GM 53 (Voice Oohs), 0-indexed = 52
        instrument = pretty_midi.Instrument(program=52, name="Vocals")
        for start, end, pitch, velocity in notes:
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch,
                    start=float(start),
                    end=float(end),
                )
            )
        pm.instruments.append(instrument)

        # Lyric meta-events
        for word, start, _end in words:
            clean = word.strip()
            if clean:
                pm.lyrics.append(pretty_midi.Lyric(text=clean, time=float(start)))

        # Write
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pm.write(str(output_path))
        except Exception as exc:
            raise AudioProcessingError(
                f"Failed to write MIDI file to {output_path}: {exc}",
                path=str(output_path),
            ) from exc
