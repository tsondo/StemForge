"""
ADTOF drum transcription model loader for StemForge.

Wraps the ADTOF-pytorch Frame-RNN model (``xavriley/ADTOF-pytorch``) for
automatic drum transcription.  The model emits per-frame activations for
five drum classes; peak picking converts these to onset times, and each
class maps directly to a General MIDI percussion note (the library's
``LABELS_5`` is already the GM note list: kick 35, snare 38, tom 47,
hi-hat 42, cymbal 49).

Weights ship inside the pip package — nothing is downloaded at runtime.
Input audio of any sample rate is accepted; the library resamples to
44.1 kHz internally via librosa.
"""

import logging
import pathlib

import torch

from utils.device import get_device
from utils.errors import ModelLoadError, PipelineExecutionError
from utils.midi_io import NoteEvent

log = logging.getLogger("stemforge.models.adtof_loader")

# Percussion hits have no meaningful sustain — every event gets this length,
# matching the 60 ms drum cap in utils.midi_io.notes_to_midi().
NOTE_DURATION_S: float = 0.06

# ADTOF activations carry no amplitude information usable as velocity.
DEFAULT_VELOCITY: int = 100


class AdtofModelLoader:
    """Loads the ADTOF Frame-RNN model and transcribes drum audio.

    Lifecycle mirrors the other StemForge loaders::

        loader = AdtofModelLoader()
        loader.load()
        events = loader.predict(path)
        loader.evict()
    """

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> torch.nn.Module:
        """Load the Frame-RNN weights onto the best available device.

        Idempotent: if already loaded, returns the cached instance.

        Raises
        ------
        :class:`~utils.errors.ModelLoadError`
            If adtof-pytorch is not installed or the weights cannot be
            deserialised.
        """
        if self._model is not None:
            return self._model
        try:
            import adtof_pytorch

            model = adtof_pytorch.create_frame_rnn_model(
                adtof_pytorch.calculate_n_bins()
            )
            model.eval()
            weights_path = adtof_pytorch.get_default_weights_path()
            model = adtof_pytorch.load_pytorch_weights(
                model, str(weights_path), strict=False
            )
            self._device = get_device()
            model.to(self._device)
            self._model = model
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load ADTOF drum model: {exc}",
                model_name="adtof-drums",
            ) from exc
        log.info("AdtofModelLoader: model ready on %s.", self._device)
        return self._model

    @property
    def is_loaded(self) -> bool:
        """``True`` if the model is currently in memory."""
        return self._model is not None

    def evict(self) -> None:
        """Release the model from memory.  Safe to call when not loaded."""
        if self._model is None:
            return
        self._model = None
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()
        self._device = None
        log.debug("AdtofModelLoader: model evicted.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, path: pathlib.Path) -> list[NoteEvent]:
        """Transcribe a drum stem to GM percussion note events.

        Parameters
        ----------
        path:
            Drum audio file.  Any sample rate; resampled internally.

        Returns
        -------
        list[NoteEvent]
            ``(start_sec, end_sec, gm_note, velocity)`` tuples sorted by
            start time.  Pitches are GM channel-10 percussion notes.

        Raises
        ------
        :class:`~utils.errors.ModelLoadError`
            If the model is not loaded and cannot be loaded on demand.
        :class:`~utils.errors.PipelineExecutionError`
            If the forward pass or peak picking fails.
        """
        if self._model is None:
            self.load()

        try:
            from adtof_pytorch import PeakPicker, load_audio_for_model

            x = load_audio_for_model(str(path)).to(self._device)
            with torch.no_grad():
                activations = self._model(x).cpu().numpy()

            # Defaults are the library's trained per-class thresholds and
            # 100 fps frame rate; pick() returns {gm_note: [onset_sec, ...]}.
            peaks = PeakPicker().pick(activations)[0]
        except Exception as exc:
            raise PipelineExecutionError(
                f"Drum transcription failed for '{path.name}': {exc}",
                pipeline_name="midi",
            ) from exc

        events: list[NoteEvent] = [
            (float(onset), float(onset) + NOTE_DURATION_S, int(gm_note), DEFAULT_VELOCITY)
            for gm_note, onsets in peaks.items()
            for onset in onsets
        ]
        events.sort(key=lambda e: e[0])
        log.debug("AdtofModelLoader: %s → %d drum events", path.name, len(events))
        return events
