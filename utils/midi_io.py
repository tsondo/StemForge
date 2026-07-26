"""
MIDI I/O utilities for StemForge.

Provides lightweight functions for reading and writing Standard MIDI Files
(SMF) using only the Python standard library.  Supports single-track and
multi-track MIDI files (format 0 and format 1).

Backend
-------
All operations delegate to :mod:`pretty_midi`, which is already a
StemForge runtime dependency and is used by :mod:`pipelines.vocal_midi_pipeline`.
The :data:`MidiData` type alias is a :class:`pretty_midi.PrettyMIDI` object.
"""

import os
import pathlib
import logging
import struct
from typing import Any, TypeAlias

import pretty_midi


# A note event is a (start_sec, end_sec, pitch_midi, velocity) tuple.
NoteEvent: TypeAlias = tuple[float, float, int, int]

# A lyric event is a (time_seconds, word_text) pair.  Used to embed MIDI
# lyric meta-events (type 0x05) that singing synths such as Ace Studio,
# VOCALOID, and SynthV read on MIDI import.
LyricEvent: TypeAlias = tuple[float, str]

# A MIDI data object wraps tracks, tempo, and resolution.  The concrete
# type is determined at runtime by whichever MIDI library is available.
MidiData: TypeAlias = Any

log = logging.getLogger("stemforge.utils.midi_io")


def read_midi(path: pathlib.Path) -> MidiData:
    """Parse a Standard MIDI File and return a structured representation.

    Parameters
    ----------
    path:
        Path to the ``.mid`` or ``.midi`` file to read.

    Returns
    -------
    MidiData
        An object with ``tracks``, ``ticks_per_beat``, and ``format``
        attributes (concrete type determined at runtime).
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")
    pm = pretty_midi.PrettyMIDI(str(path))
    log.debug("read_midi: %s  instruments=%d", path.name, len(pm.instruments))
    return pm


def write_midi(midi_data: MidiData, path: pathlib.Path) -> pathlib.Path:
    """Serialise *midi_data* to a Standard MIDI File at *path*.

    Parameters
    ----------
    midi_data:
        Structured MIDI data as returned by :func:`read_midi` or constructed
        from note events via :func:`notes_to_midi`.
    path:
        Destination file path (will be created or overwritten).

    Returns
    -------
    pathlib.Path
        Resolved path of the written file.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi_data.write(str(path))
    log.debug("write_midi: %s", path)
    return path.resolve()


def notes_to_midi(
    note_events: list[NoteEvent],
    ticks_per_beat: int = 480,
    tempo_bpm: float = 120.0,
    lyrics: list[LyricEvent] | None = None,
    is_drum: bool = False,
) -> MidiData:
    """Convert a list of note events to a MIDI data object.

    Parameters
    ----------
    note_events:
        List of ``(start_sec, end_sec, pitch, velocity)`` tuples.
    ticks_per_beat:
        MIDI ticks per quarter-note (resolution).
    tempo_bpm:
        Tempo in beats per minute for the generated MIDI file.
    lyrics:
        Optional ``(time_sec, word)`` lyric events to embed.
    is_drum:
        When ``True``, the instrument is routed to the General MIDI
        percussion channel (10) and note durations are capped at 60 ms.

    Returns
    -------
    MidiData
        MIDI data object suitable for passing to :func:`write_midi`.
    """
    pm = pretty_midi.PrettyMIDI(
        resolution=int(ticks_per_beat),
        initial_tempo=float(tempo_bpm),
    )
    if is_drum:
        instrument = pretty_midi.Instrument(
            program=0,
            is_drum=True,
            name="StemForge Drums",
        )
    else:
        instrument = pretty_midi.Instrument(
            program=pretty_midi.instrument_name_to_program("Acoustic Grand Piano"),
            name="StemForge",
        )
    for start, end, pitch, velocity in note_events:
        # Guard against degenerate notes (zero or negative duration).
        if end <= start:
            continue
        if is_drum:
            # Percussion hits are instantaneous — long "notes" render badly.
            end = min(end, start + 0.06)
        note = pretty_midi.Note(
            velocity=max(1, min(127, int(velocity))),
            pitch=max(0, min(127, int(pitch))),
            start=float(start),
            end=float(end),
        )
        instrument.notes.append(note)
    instrument.notes.sort(key=lambda n: n.start)
    pm.instruments.append(instrument)
    if lyrics:
        for t, text in lyrics:
            if text:
                pm.lyrics.append(pretty_midi.Lyric(text=text, time=float(t)))
        pm.lyrics.sort(key=lambda l: l.time)
    return pm


def midi_to_notes(midi_data: MidiData) -> list[NoteEvent]:
    """Extract note events from *midi_data* as ``(start, end, pitch, velocity)`` tuples.

    Parameters
    ----------
    midi_data:
        MIDI data object as returned by :func:`read_midi`.
    """
    events: list[NoteEvent] = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            events.append((note.start, note.end, note.pitch, note.velocity))
    return sorted(events, key=lambda e: e[0])


def get_tempo(midi_data: MidiData) -> float:
    """Return the first tempo marking in *midi_data* as beats per minute."""
    _, tempos = midi_data.get_tempo_changes()
    if len(tempos) == 0:
        return 120.0
    return float(tempos[0])


def quantise_notes(
    note_events: list[NoteEvent],
    grid_seconds: float,
) -> list[NoteEvent]:
    """Snap note start and end times in *note_events* to the nearest *grid_seconds* boundary."""
    if grid_seconds <= 0:
        return list(note_events)

    def snap(t: float) -> float:
        return round(t / grid_seconds) * grid_seconds

    result: list[NoteEvent] = []
    for start, end, pitch, velocity in note_events:
        q_start = snap(start)
        q_end = snap(end)
        # Guarantee at least one grid cell of duration after snapping.
        if q_end <= q_start:
            q_end = q_start + grid_seconds
        result.append((q_start, q_end, pitch, velocity))
    return result


# ---------------------------------------------------------------------------
# Key / scale helpers
# ---------------------------------------------------------------------------

# Chromatic pitch-class for each note name (enharmonic equivalents share a value)
_NOTE_PC: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

_MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
_MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]


def _parse_key(key: str) -> tuple[int, list[int]] | None:
    """Return ``(root_pc, scale_intervals)`` for *key*, or ``None`` if unrecognised."""
    parts = key.strip().split()
    if len(parts) != 2:
        return None
    root = _NOTE_PC.get(parts[0])
    if root is None:
        return None
    intervals = _MAJOR_INTERVALS if parts[1].lower() == "major" else _MINOR_INTERVALS
    return root, intervals


def _parse_time_sig(sig: str) -> tuple[int, int]:
    """Parse ``'4/4'`` → ``(4, 4)``.  Returns ``(4, 4)`` on any error."""
    try:
        num_s, den_s = sig.split("/")
        return int(num_s), int(den_s)
    except Exception:
        return 4, 4


def filter_to_key(note_events: list[NoteEvent], key: str) -> list[NoteEvent]:
    """Snap pitches that are not in *key* to the nearest in-scale pitch.

    Parameters
    ----------
    note_events:
        Input note events.
    key:
        Key string such as ``'C major'`` or ``'A minor'``.
        Passing ``'Any'`` or an unrecognised string returns *note_events*
        unchanged.

    Returns
    -------
    list[NoteEvent]
        Same timing and velocity as input; pitches may be adjusted by ±1–6
        semitones to land on the nearest in-key pitch.
    """
    parsed = _parse_key(key)
    if parsed is None:
        return list(note_events)

    root, intervals = parsed
    scale_pcs = {(root + i) % 12 for i in intervals}

    result: list[NoteEvent] = []
    for start, end, pitch, velocity in note_events:
        if pitch % 12 in scale_pcs:
            result.append((start, end, pitch, velocity))
        else:
            # Scan ±6 semitones and pick the closest in-key candidate.
            best, best_dist = pitch, 13
            for delta in range(-6, 7):
                candidate = pitch + delta
                if 0 <= candidate <= 127 and candidate % 12 in scale_pcs:
                    if abs(delta) < best_dist:
                        best_dist = abs(delta)
                        best = candidate
            result.append((start, end, max(0, min(127, best)), velocity))
    return result


# ---------------------------------------------------------------------------
# Multi-track merge
# ---------------------------------------------------------------------------

# Default General MIDI program numbers for known stem names
_STEM_GM_PROGRAM: dict[str, int] = {
    "vocals":               52,   # Choir Aahs
    "Singing voice":        52,
    "drums":                 0,   # (is_drum=True overrides program)
    "Drums & percussion":    0,
    "bass":                 33,   # Electric Bass (finger)
    "Bass":                 33,
    "other":                48,   # String Ensemble 1
    "Everything else":      48,
    "generated":             0,   # Acoustic Grand Piano
}

# Canonical set of stem labels that represent drums/percussion.  Single
# source of truth — the MIDI pipeline (ADT routing) and the API layer
# (track defaults) both derive from this.
DRUM_STEM_LABELS: frozenset[str] = frozenset({
    "drums",                  # Demucs output label
    "Drums & percussion",     # BS-Roformer 4/6-stem output label
})

_STEM_IS_DRUM: dict[str, bool] = dict.fromkeys(DRUM_STEM_LABELS, True)


def merge_tracks(
    track_notes: dict[str, list[NoteEvent]],
    bpm: float = 120.0,
    time_signature: str = "4/4",
    ticks_per_beat: int = 480,
    stem_programs: dict[str, int] | None = None,
    stem_is_drum: dict[str, bool] | None = None,
    track_lyrics: dict[str, list[LyricEvent]] | None = None,
) -> MidiData:
    """Create a format-1 multi-track MIDI from per-stem note events.

    All tracks share a single tempo and time-signature map so they are
    perfectly time-aligned on any standards-compliant MIDI player or DAW.

    Parameters
    ----------
    track_notes:
        ``{stem_name: [NoteEvent, ...]}`` — one entry per track.
    bpm:
        Tempo in beats per minute (shared across all tracks).
    time_signature:
        String such as ``'4/4'`` or ``'3/4'`` (shared across all tracks).
    ticks_per_beat:
        MIDI resolution in ticks per quarter-note.
    stem_programs:
        Optional override map of stem name → GM program number.  Defaults
        to :data:`_STEM_GM_PROGRAM` for known names; piano (0) otherwise.
    stem_is_drum:
        Optional override map of stem name → ``True`` for percussion tracks.
        Defaults to :data:`_STEM_IS_DRUM` for known names.

    Returns
    -------
    MidiData
        ``pretty_midi.PrettyMIDI`` with one instrument per stem.
    """
    programs = {**_STEM_GM_PROGRAM, **(stem_programs or {})}
    drums = {**_STEM_IS_DRUM, **(stem_is_drum or {})}

    pm = pretty_midi.PrettyMIDI(
        resolution=int(ticks_per_beat),
        initial_tempo=float(bpm),
    )

    num, den = _parse_time_sig(time_signature)
    pm.time_signature_changes = [
        pretty_midi.TimeSignature(numerator=num, denominator=den, time=0.0)
    ]

    for stem_name, notes in track_notes.items():
        program = programs.get(stem_name, 0)
        is_drum = drums.get(stem_name, False)
        instrument = pretty_midi.Instrument(
            program=program,
            is_drum=is_drum,
            name=stem_name,
        )
        for start, end, pitch, velocity in notes:
            if end <= start:
                continue
            instrument.notes.append(pretty_midi.Note(
                velocity=max(1, min(127, int(velocity))),
                pitch=max(0, min(127, int(pitch))),
                start=float(start),
                end=float(end),
            ))
        instrument.notes.sort(key=lambda n: n.start)
        pm.instruments.append(instrument)

    if track_lyrics:
        for stem_name, lyrics in track_lyrics.items():
            for t, text in lyrics:
                if text:
                    pm.lyrics.append(pretty_midi.Lyric(text=text, time=float(t)))
        pm.lyrics.sort(key=lambda l: l.time)

    log.debug(
        "merge_tracks: %d tracks, bpm=%.1f, time_sig=%s, lyric_events=%d",
        len(track_notes), bpm, time_signature,
        sum(len(v) for v in (track_lyrics or {}).values()),
    )
    return pm


# ---------------------------------------------------------------------------
# Text-only chord generation
# ---------------------------------------------------------------------------

# Diatonic triads (semitone offsets from key root) for I–IV–V–I progressions.
# Each inner list is [root, third, fifth] relative to key root.
_MAJOR_PROGRESSION: list[list[int]] = [
    [0, 4, 7],    # I   major
    [5, 9, 12],   # IV  major
    [7, 11, 14],  # V   major
    [0, 4, 7],    # I   major (repeat)
]
_MINOR_PROGRESSION: list[list[int]] = [
    [0, 3, 7],    # i   minor
    [5, 8, 12],   # iv  minor
    [7, 11, 14],  # V   major (dominant)
    [0, 3, 7],    # i   minor (repeat)
]


def generate_chord_progression(
    key: str,
    bpm: float,
    time_signature: str,
    duration_seconds: float,
) -> list[NoteEvent]:
    """Generate a diatonic I–IV–V–I chord progression as note events.

    Used for text-only MIDI generation when no audio stems are available.
    The progression repeats as needed to fill *duration_seconds*.

    Parameters
    ----------
    key:
        Key string such as ``'C major'`` or ``'A minor'``.
        ``'Any'`` defaults to ``'C major'``.
    bpm:
        Tempo in beats per minute.
    time_signature:
        Time-signature string such as ``'4/4'``.
    duration_seconds:
        Total duration to fill.  At least one chord is always emitted.

    Returns
    -------
    list[NoteEvent]
        Chord tones as ``(start_sec, end_sec, pitch_midi, velocity)`` tuples,
        sorted by start time.
    """
    if not key or key == "Any":
        key = "C major"

    parsed = _parse_key(key)
    root = parsed[0] if parsed else 0
    is_major = "major" in key.lower()
    progression = _MAJOR_PROGRESSION if is_major else _MINOR_PROGRESSION

    num, _den = _parse_time_sig(time_signature)
    beats_per_bar = num
    bar_seconds = (beats_per_bar / max(1.0, bpm)) * 60.0
    # Each chord lasts 2 bars
    chord_seconds = bar_seconds * 2

    events: list[NoteEvent] = []
    t = 0.0
    velocity = 80
    # Middle-C octave (MIDI 48 = C3); keep within 36–84 for a pleasant range.
    base_midi = root + 48

    while t < duration_seconds:
        for triad in progression:
            if t >= duration_seconds:
                break
            chord_end = min(t + chord_seconds, duration_seconds)
            for offset in triad:
                pitch = max(36, min(84, base_midi + offset))
                events.append((t, chord_end - 0.02, pitch, velocity))
            t += chord_seconds

    events.sort(key=lambda e: e[0])
    return events
