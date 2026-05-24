"""
MIDI model loader for StemForge.

Wraps the BasicPitch audio-to-MIDI model and provides a high-level
``convert_audio_to_midi()`` facade that handles model loading, inference,
duration clipping, and optional key filtering in a single call.

Unlike the other loaders, this one intentionally stays slightly above the
raw-model level: it owns the BasicPitchModelLoader as a delegate and
encapsulates the BasicPitch inference protocol so the pipeline layer never
has to import ``basic_pitch.inference`` directly.

GPU / CPU note
--------------
BasicPitch runs on CPU only (the loader sets ``CUDA_VISIBLE_DEVICES=-1``
before TF is imported via ``BasicPitchModelLoader``).  The MIDI pipeline
is therefore purely CPU-bound; Demucs/MusicGen GPU memory is unaffected.
"""

import pathlib
import logging
from typing import Any

from models.basicpitch_loader import BasicPitchModelLoader
from utils.midi_io import NoteEvent, LyricEvent, filter_to_key
from utils.errors import ModelLoadError, PipelineExecutionError


log = logging.getLogger("stemforge.models.midi_loader")


class MidiModelLoader:
    """Facade that converts audio files to MIDI note events.

    Exposes two conversion paths:

    * :meth:`convert_audio_to_midi` — uses BasicPitch (good for instruments).
    * :meth:`convert_vocal_to_midi` — uses faster-whisper for word timing
      and librosa PYIN for pitch estimation (better for sung vocals).

    Lifecycle mirrors all other StemForge loaders::

        loader = MidiModelLoader()
        loader.load()
        notes = loader.convert_audio_to_midi(path, key="C major", bpm=120.0)
        loader.evict()

    Parameters
    ----------
    (none — delegates cache configuration to BasicPitchModelLoader defaults)
    """

    def __init__(self) -> None:
        self._bp_loader = BasicPitchModelLoader()
        self._model: Any | None = None          # BasicPitch TF model

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> Any:
        """Load the BasicPitch TF SavedModel.  Returns the model object.

        Idempotent: if already loaded, returns the cached instance
        without re-loading.

        Raises
        ------
        :class:`~utils.errors.ModelLoadError`
            If TensorFlow or basic-pitch are not installed, or the
            SavedModel cannot be deserialised.
        """
        if self._model is not None:
            return self._model
        try:
            self._model = self._bp_loader.load()
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(
                f"Unexpected error loading MIDI model: {exc}",
                model_name="basicpitch",
            ) from exc
        log.info("MidiModelLoader: BasicPitch model ready (CPU-only).")
        return self._model

    @property
    def is_loaded(self) -> bool:
        """``True`` if the model is currently in memory."""
        return self._model is not None

    def evict(self) -> None:
        """Release both models from memory and trigger GC."""
        self._bp_loader.evict()
        self._model = None
        log.debug("MidiModelLoader: models evicted.")

    # ------------------------------------------------------------------
    # High-level conversion
    # ------------------------------------------------------------------

    def convert_audio_to_midi(
        self,
        path: pathlib.Path,
        *,
        prompt: str | None = None,
        duration: float = 0.0,
        key: str = "Any",
        signature: str = "4/4",
        bpm: float = 120.0,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        minimum_note_length: float = 58.0,
    ) -> list[NoteEvent]:
        """Transcribe *path* to a list of note events.

        Parameters
        ----------
        path:
            Audio file to transcribe.  Any format supported by
            ``utils.audio_io.SUPPORTED_EXTENSIONS``.
        prompt:
            Optional text description.  Currently logged but not used for
            conditioning; reserved for future text-guided post-processing.
        duration:
            If positive, clip note events to ``[0, duration)`` seconds.
            Pass ``0`` (default) to keep all detected notes.
        key:
            Musical key for pitch snapping, e.g. ``'C major'`` or
            ``'A minor'``.  ``'Any'`` skips key filtering entirely.
        signature:
            Time signature string (informational; not passed to BasicPitch).
        bpm:
            Tempo in BPM (informational; used when building the merged MIDI
            tempo map in the pipeline layer, not by this method).
        onset_threshold:
            BasicPitch onset confidence threshold in ``[0, 1]``.
        frame_threshold:
            BasicPitch frame confidence threshold in ``[0, 1]``.
        minimum_note_length:
            Minimum note duration in milliseconds.

        Returns
        -------
        list[NoteEvent]
            ``(start_sec, end_sec, pitch_midi, velocity)`` tuples,
            sorted by ``start_sec``.

        Raises
        ------
        :class:`~utils.errors.ModelLoadError`
            If the model has not been loaded and cannot be loaded on demand.
        :class:`~utils.errors.PipelineExecutionError`
            If BasicPitch inference raises an unexpected error.
        """
        if self._model is None:
            self.load()

        if prompt:
            log.debug(
                "convert_audio_to_midi: text prompt received (%.60r) — "
                "reserved for future conditioning.",
                prompt,
            )

        try:
            note_events_raw = self._model.predict(
                path,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=minimum_note_length,
            )
        except PipelineExecutionError:
            raise
        except Exception as exc:
            raise PipelineExecutionError(
                f"Audio-to-MIDI transcription failed for '{path.name}': {exc}",
                pipeline_name="midi",
            ) from exc

        # Normalise BasicPitch raw output to NoteEvent tuples
        events: list[NoteEvent] = []
        for item in note_events_raw:
            start_t, end_t, pitch, amplitude, *_ = item
            velocity = max(1, min(127, int(float(amplitude) * 127)))
            events.append((float(start_t), float(end_t), int(pitch), velocity))
        events.sort(key=lambda e: e[0])

        # Optional duration clip
        if duration > 0.0:
            events = [
                (s, min(e, duration), p, v)
                for s, e, p, v in events
                if s < duration
            ]

        # Key snap: off-key pitches are moved to the nearest in-scale semitone
        if key and key != "Any":
            events = filter_to_key(events, key)

        log.debug(
            "convert_audio_to_midi: %s → %d notes (key=%r, onset=%.2f, frame=%.2f)",
            path.name, len(events), key, onset_threshold, frame_threshold,
        )
        return events

    def convert_vocal_to_midi(
        self,
        path: pathlib.Path,
        *,
        duration: float = 0.0,
        key: str = "Any",
        bpm: float = 120.0,
        language: str | None = None,
    ) -> tuple[list[NoteEvent], list[LyricEvent]]:
        """Transcribe a vocal stem to MIDI using faster-whisper + librosa PYIN.

        Word timing from faster-whisper is combined with probabilistic pitch
        estimation (librosa PYIN) to produce one note per word.  Pitches are
        snapped to *key* after estimation.

        Parameters
        ----------
        path:
            Vocal audio file (mono or stereo).
        duration:
            Clip output to this many seconds if positive.
        key:
            Musical key for pitch snapping (e.g. ``'A minor'``).
        bpm:
            Tempo (informational; used only in the pipeline's merge step).
        language:
            ISO-639-1 language code hint for Whisper (e.g. ``'en'``).
            ``None`` lets Whisper auto-detect.

        Returns
        -------
        tuple[list[NoteEvent], list[LyricEvent]]
            ``(notes, lyrics)`` — one note per transcribed word, and a
            parallel list of ``(time_seconds, word_text)`` lyric events
            suitable for embedding as MIDI type-0x05 meta messages.

        Raises
        ------
        :class:`~utils.errors.ModelLoadError`
            If faster-whisper or librosa is not installed.
        :class:`~utils.errors.PipelineExecutionError`
            If transcription or pitch estimation fails.
        """
        try:
            import numpy as np
            import librosa
        except ImportError as exc:
            raise ModelLoadError(
                "librosa is required for vocal pitch estimation — is it installed?",
                model_name="librosa",
            ) from exc

        # Load audio at 22 050 Hz for PYIN.  The transcription engine
        # reloads the audio from disk at its native rate internally.
        try:
            y_pyin, sr_pyin = librosa.load(str(path), sr=22_050, mono=True)
        except Exception as exc:
            raise PipelineExecutionError(
                f"Failed to load vocal audio '{path.name}': {exc}",
                pipeline_name="midi",
            ) from exc

        # Pre-compute PYIN pitch track for the whole file.
        try:
            f0, voiced_flag, _voiced_probs = librosa.pyin(
                y_pyin,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sr_pyin,
            )
        except Exception as exc:
            raise PipelineExecutionError(
                f"Pitch estimation failed for '{path.name}': {exc}",
                pipeline_name="midi",
            ) from exc

        frame_times = librosa.frames_to_time(
            range(len(f0)), sr=sr_pyin, hop_length=512
        )

        # Transcribe via the shared WhisperEngine.
        from pipelines.transcribe_engines import WhisperEngine

        engine = WhisperEngine(model_id="whisper-base")
        try:
            engine.load()
            transcription = engine.transcribe(path, language=language)
        finally:
            engine.clear()
        segments = transcription.segments

        # Build one NoteEvent + LyricEvent per voiced word.
        events: list[NoteEvent] = []
        lyrics: list[LyricEvent] = []
        for segment in segments:
            words = getattr(segment, "words", None) or []
            for word in words:
                start = float(word.start)
                end = float(word.end)
                if end <= start:
                    continue
                if duration > 0.0 and start >= duration:
                    continue

                # Find PYIN frames that fall within [start, end].
                mask = (frame_times >= start) & (frame_times < end) & voiced_flag
                voiced_f0 = f0[mask]

                if len(voiced_f0) == 0 or np.all(np.isnan(voiced_f0)):
                    # Unvoiced window — skip (rest / consonant).
                    continue

                median_hz = float(np.nanmedian(voiced_f0))
                if median_hz <= 0.0:
                    continue

                midi_pitch = int(round(librosa.hz_to_midi(median_hz)))
                midi_pitch = max(0, min(127, midi_pitch))

                clipped_end = min(end, duration) if duration > 0.0 else end
                prob = word.probability if word.probability is not None else 0.5
                velocity = min(127, max(1, int(abs(prob) * 100)))
                events.append((start, clipped_end, midi_pitch, velocity))

                # Strip leading/trailing punctuation but keep apostrophes.
                text = word.word.strip().strip('.,!?;:"()[]{}…—–')
                if text:
                    lyrics.append((start, text))

        events.sort(key=lambda e: e[0])
        lyrics.sort(key=lambda l: l[0])

        if key and key != "Any":
            events = filter_to_key(events, key)

        log.debug(
            "convert_vocal_to_midi: %s → %d notes, %d lyric events from %d segments",
            path.name, len(events), len(lyrics), len(segments),
        )
        return events, lyrics
