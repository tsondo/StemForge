"""MIDI extraction, render, and save endpoints."""

from __future__ import annotations

import copy
import pathlib
import tempfile
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.services.job_manager import job_manager
from backend.services.session_store import SessionStore, TrackState, get_user_session
from backend.services import pipeline_manager
from utils.midi_io import DRUM_STEM_LABELS
from utils.paths import MIDI_DIR

router = APIRouter(prefix="/api/midi", tags=["midi"])

# ─── General MIDI program names (0-127) ──────────────────────────────────
GM_PROGRAMS: list[str] = [
    # Piano (0-7)
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    # Chromatic Percussion (8-15)
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    # Organ (16-23)
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion",
    # Guitar (24-31)
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
    "Distortion Guitar", "Guitar Harmonics",
    # Bass (32-39)
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    # Strings (40-47)
    "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    # Ensemble (48-55)
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2",
    "Choir Aahs", "Voice Oohs", "Synth Choir", "Orchestra Hit",
    # Brass (56-63)
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    # Reed (64-71)
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet",
    # Pipe (72-79)
    "Piccolo", "Flute", "Recorder", "Pan Flute",
    "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    # Synth Lead (80-87)
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)",
    "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    # Synth Pad (88-95)
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)",
    "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    # Synth Effects (96-103)
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    # Ethnic (104-111)
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    # Percussive (112-119)
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
    "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    # Sound Effects (120-127)
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]

# Smart defaults by stem label
STEM_DEFAULT_PROGRAM: dict[str, int] = {
    "vocals": 52,       # Voice Oohs
    "drums": 0,         # (is_drum=True overrides)
    "bass": 33,         # Electric Bass (finger)
    "guitar": 25,       # Acoustic Guitar (steel)
    "piano": 0,         # Acoustic Grand Piano
    "keyboard": 0,      # Acoustic Grand Piano
    "other": 48,        # String Ensemble 1
}

STEM_IS_DRUM: dict[str, bool] = dict.fromkeys(DRUM_STEM_LABELS, True)

# ─── SoundFont discovery & state ─────────────────────────────────────────

_SF2_SEARCH_PATHS = [
    "/usr/share/soundfonts/FluidR3_GM.sf2",       # Fedora
    "/usr/share/soundfonts/FluidR3_GM2-2.sf2",    # Fedora alt
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",       # Ubuntu
    "/usr/share/sounds/sf2/default-GM.sf2",       # Ubuntu alt
    "/usr/share/soundfonts/default.sf2",          # Arch
]


def _find_system_soundfont() -> str | None:
    """Return the first GM soundfont found on the system, or None."""
    for p in _SF2_SEARCH_PATHS:
        if pathlib.Path(p).is_file():
            return p
    return None


# Active soundfont path (None = let pretty_midi auto-discover)
_active_soundfont: str | None = _find_system_soundfont()


class ExtractRequest(BaseModel):
    stems: list[str]
    key: str = "Any"
    bpm: float = 120.0
    time_signature: str = "4/4"
    duration: float = 0.0
    prompt: str = ""
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_ms: float = 58.0


class RenderRequest(BaseModel):
    stem_label: str
    # None means "leave each instrument's own voice alone" — required for
    # merged MIDI, where forcing one program would collapse a multi-
    # instrument arrangement to a single sound.
    program: int | None = None
    is_drum: bool | None = None


class EventsRequest(BaseModel):
    stem_label: str


class SaveRequest(BaseModel):
    label: str = "merged"    # "merged" or a stem label
    smf_format: int = 1      # 1 = multi-track, 0 = single-track (hardware arrangers)


def _run_midi_extraction(
    stems: dict[str, pathlib.Path],
    config_kwargs: dict,
    job_id: str,
    session: SessionStore,
) -> dict:
    """Execute MIDI pipeline (runs in background thread)."""
    from pipelines.midi_pipeline import MidiConfig

    # MidiPipeline callback takes 1 arg (pct 0–100), not (pct, stage)
    def _midi_cb(pct):
        job_manager.update_progress(job_id, pct / 100.0, "Extracting MIDI...")

    with pipeline_manager.gpu_session(pipeline_hint="midi") as ctx:
        pipeline = pipeline_manager.get_midi(ctx.gpu_index)
        config = MidiConfig(**config_kwargs)
        pipeline.configure(config)

        job_manager.update_progress(job_id, 0.05, "Loading model...")
        pipeline.load_model()

        pipeline.set_progress_callback(_midi_cb)
        result = pipeline.run(stems)

    # Store in session and auto-add mix tracks
    session.merged_midi_data = result.merged_midi_data
    session.stem_midi_data = result.stem_midi_data or {}

    stem_info = {}
    for label, midi_data in (result.stem_midi_data or {}).items():
        note_count = sum(len(inst.notes) for inst in midi_data.instruments)
        stem_info[label] = {"note_count": note_count}

        track_id = f"midi-{label}"
        lower = label.lower()
        default_prog = next(
            (v for k, v in STEM_DEFAULT_PROGRAM.items() if k in lower), 0
        )
        default_drum = any(
            k.lower() in lower for k in STEM_IS_DRUM
        )
        if not session.get_track(track_id):
            session.add_track(TrackState(
                track_id=track_id,
                label=f"{label.replace('_', ' ').title()} (MIDI)",
                source="midi",
                midi_data=midi_data,
                program=default_prog,
                is_drum=default_drum,
            ))

    return {
        "labels": list((result.stem_midi_data or {}).keys()),
        "stem_info": stem_info,
        "has_merged": result.merged_midi_data is not None,
    }


@router.post("/extract")
def start_extraction(req: ExtractRequest, session: SessionStore = Depends(get_user_session)) -> dict:
    stem_paths = session.stem_paths
    if not stem_paths:
        raise HTTPException(400, "No stems available — run separation first")

    # Filter to requested stems
    stems = {k: v for k, v in stem_paths.items() if k in req.stems}
    if not stems:
        raise HTTPException(422, f"None of {req.stems} found in session stems")

    config_kwargs = {
        "key": req.key,
        "bpm": req.bpm,
        "time_signature": req.time_signature,
        "duration_seconds": req.duration,
        "prompt": req.prompt,
        "onset_threshold": req.onset_threshold,
        "frame_threshold": req.frame_threshold,
        "minimum_note_length": req.min_note_ms,
    }

    job_id = job_manager.create_job("midi", user=session.user)
    job_manager.run_job(job_id, _run_midi_extraction, stems, config_kwargs, job_id, session)
    return {"job_id": job_id}


@router.post("/render")
def render_midi_to_audio(req: RenderRequest, session: SessionStore = Depends(get_user_session)) -> dict:
    """Render a stem's MIDI to audio via FluidSynth.

    Accepts ``merged`` as well as a stem label, so the merged arrangement
    can be previewed through the soft synth like any other card.
    """
    midi_data = copy.deepcopy(_resolve_midi(req.stem_label, session))

    # Override every instrument only when the caller asked for it. Omitting
    # both preserves the arrangement's own voices.
    for inst in midi_data.instruments:
        if req.program is not None:
            inst.program = req.program
        if req.is_drum is not None:
            inst.is_drum = req.is_drum

    # Use source sample rate for MIDI renders
    audio_info = session.audio_info or {}
    sr = audio_info.get("sample_rate", 44100)

    # Render via FluidSynth (with active soundfont if set)
    sf2_kwargs = {"sf2_path": _active_soundfont} if _active_soundfont else {}
    audio = midi_data.fluidsynth(fs=sr, **sf2_kwargs)
    if audio is None or len(audio) == 0:
        raise HTTPException(500, "FluidSynth render produced no audio")

    # Write to temp file
    import numpy as np
    from utils.audio_io import write_audio

    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MIDI_DIR / f"render_{req.stem_label}_{uuid.uuid4().hex[:6]}.wav"
    waveform = audio.astype(np.float32)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]
    # Normalize
    peak = np.abs(waveform).max()
    if peak > 0:
        waveform = waveform / peak * 0.9
    write_audio(waveform, sr, out_path, bit_depth=24)

    return {
        "audio_path": str(out_path),
        "duration": len(audio) / sr,
    }


# Hard cap on events per /events response. A stem this dense is almost
# certainly a BasicPitch artefact and should go through Clean Up first.
MAX_MIDI_OUT_EVENTS = 20000


def _flatten_instrument_notes(inst) -> list[tuple[float, float, int, int]]:
    """Clean one instrument's notes for Web MIDI: (start, end, pitch, vel).

    Drops zero-length notes, clamps pitch to 0-127 and note-on velocity to
    1-127 (velocity 0 reads as note-off on hardware), and merges overlapping
    same-pitch notes by truncating the earlier note to the later one's start
    — hardware cannot represent the overlap, and the frontend's sounding-set
    would corrupt.
    """
    notes = [n for n in inst.notes if n.end > n.start]
    notes.sort(key=lambda n: (n.start, n.pitch))

    cleaned: list[tuple[float, float, int, int] | None] = []
    last_by_pitch: dict[int, int] = {}
    for n in notes:
        pitch = min(max(int(n.pitch), 0), 127)
        vel = min(max(int(n.velocity), 1), 127)
        start, end = float(n.start), float(n.end)
        prev_i = last_by_pitch.get(pitch)
        if prev_i is not None:
            prev = cleaned[prev_i]
            if prev is not None and prev[1] > start:
                if start > prev[0]:
                    cleaned[prev_i] = (prev[0], start, pitch, prev[3])
                else:
                    cleaned[prev_i] = None  # same start — earlier note vanishes
        cleaned.append((start, end, pitch, vel))
        last_by_pitch[pitch] = len(cleaned) - 1
    return [c for c in cleaned if c is not None]


@router.post("/events")
def get_midi_events(
    req: EventsRequest,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    """Flatten a session PrettyMIDI to a JSON note-event list for Web MIDI.

    Returns one entry in ``tracks`` per PrettyMIDI instrument.  Events
    within a track are sorted ascending by time, note-off before note-on
    at equal timestamps.  Read-only — session MIDI is never modified.
    """
    midi_data = _resolve_midi(req.stem_label, session)

    track_notes = [
        (inst, _flatten_instrument_notes(inst)) for inst in midi_data.instruments
    ]

    # Truncate by whole notes (never split an on from its off): drop the
    # latest-starting notes across all tracks until under the event cap.
    truncated = False
    total_notes = sum(len(notes) for _, notes in track_notes)
    if total_notes * 2 > MAX_MIDI_OUT_EVENTS:
        truncated = True
        keep = MAX_MIDI_OUT_EVENTS // 2
        indexed = [
            (note[0], ti, note)
            for ti, (_, notes) in enumerate(track_notes)
            for note in notes
        ]
        indexed.sort(key=lambda x: x[0])
        kept_per_track: list[list[tuple[float, float, int, int]]] = [
            [] for _ in track_notes
        ]
        for _, ti, note in indexed[:keep]:
            kept_per_track[ti].append(note)
        track_notes = [
            (inst, kept_per_track[ti]) for ti, (inst, _) in enumerate(track_notes)
        ]

    tracks = []
    total_events = 0
    for inst, notes in track_notes:
        events = []
        for start, end, pitch, vel in notes:
            events.append({"t": start, "type": "on", "note": pitch, "vel": vel})
            events.append({"t": end, "type": "off", "note": pitch, "vel": 0})
        events.sort(key=lambda e: (e["t"], 0 if e["type"] == "off" else 1))
        total_events += len(events)
        tracks.append({
            "name": inst.name or req.stem_label,
            "is_drum": bool(inst.is_drum),
            "program": int(inst.program),
            "events": events,
        })

    return {
        "label": req.stem_label,
        "duration": float(midi_data.get_end_time()),
        "total_events": total_events,
        "truncated": truncated,
        "tracks": tracks,
    }


@router.post("/save")
def save_midi(req: SaveRequest, session: SessionStore = Depends(get_user_session)) -> dict:
    if req.smf_format not in (0, 1):
        raise HTTPException(422, f"Unsupported SMF format: {req.smf_format}")
    MIDI_DIR.mkdir(parents=True, exist_ok=True)

    if req.label == "merged":
        midi_data = session.merged_midi_data
        if midi_data is None:
            raise HTTPException(404, "No merged MIDI available")
        out_path = MIDI_DIR / "merged.mid"
    else:
        stem_midi = session.stem_midi_data
        if req.label not in stem_midi:
            raise HTTPException(404, f"No MIDI for stem '{req.label}'")
        midi_data = stem_midi[req.label]
        out_path = MIDI_DIR / f"{req.label}.mid"

    from utils.midi_io import write_midi, flatten_to_format0
    write_midi(midi_data, out_path)
    if req.smf_format == 0:
        flatten_to_format0(out_path)
    return {"path": str(out_path)}


@router.post("/import")
async def import_midi_file(
    file: UploadFile = File(...),
    session: SessionStore = Depends(get_user_session),
) -> dict:
    """Import an external .mid/.midi file into the session as a MIDI track."""
    import shutil
    import pretty_midi

    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    dest = MIDI_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    midi_data = pretty_midi.PrettyMIDI(str(dest))
    label = pathlib.Path(file.filename or "imported").stem

    # Guess instrument from filename
    lower = label.lower()
    default_prog = next(
        (v for k, v in STEM_DEFAULT_PROGRAM.items() if k in lower), 0
    )
    default_drum = any(k.lower() in lower for k in STEM_IS_DRUM)

    # Store in session stem_midi_data so it appears in MIDI tab results
    smd = session.stem_midi_data
    smd[label] = midi_data
    session.stem_midi_data = smd

    # Also add as a Mix track
    track_id = f"midi-{label}"
    if not session.get_track(track_id):
        session.add_track(TrackState(
            track_id=track_id,
            label=f"{label.replace('_', ' ').title()} (MIDI)",
            source="midi",
            midi_data=midi_data,
            program=default_prog,
            is_drum=default_drum,
        ))

    note_count = sum(len(inst.notes) for inst in midi_data.instruments)
    return {
        "label": label,
        "note_count": note_count,
        "track_id": track_id,
    }


@router.get("/stems")
def get_midi_stems(session: SessionStore = Depends(get_user_session)) -> dict:
    stem_midi = session.stem_midi_data
    labels = list(stem_midi.keys())
    stem_info = {}
    for label, midi_data in stem_midi.items():
        note_count = sum(len(inst.notes) for inst in midi_data.instruments)
        stem_info[label] = {"note_count": note_count}
    return {"labels": labels, "stem_info": stem_info}


@router.get("/gm-programs")
def get_gm_programs() -> dict:
    """Return list of all 128 GM program names and smart defaults per stem label."""
    return {
        "programs": GM_PROGRAMS,
        "defaults": STEM_DEFAULT_PROGRAM,
        "drum_stems": STEM_IS_DRUM,
    }


@router.get("/soundfont")
def get_soundfont() -> dict:
    """Return the currently active soundfont path."""
    return {"path": _active_soundfont or ""}


class SoundfontRequest(BaseModel):
    path: str


@router.post("/soundfont")
def set_soundfont(req: SoundfontRequest) -> dict:
    """Set the active soundfont path. Empty string resets to auto-discovery."""
    global _active_soundfont
    if not req.path:
        _active_soundfont = _find_system_soundfont()
        return {"path": _active_soundfont or "", "status": "reset"}

    p = pathlib.Path(req.path)
    if not p.is_file():
        raise HTTPException(404, f"SoundFont file not found: {req.path}")
    if not p.suffix.lower() in (".sf2", ".sf3"):
        raise HTTPException(422, "File must be .sf2 or .sf3")

    _active_soundfont = str(p)
    return {"path": _active_soundfont, "status": "ok"}


# ─── music21 integration: request models ─────────────────────────────

class CleanUpRequest(BaseModel):
    stem_label: str
    quantize_divisors: list[int] = [4, 3]
    min_note_length: float = 0.125
    key: str | None = None
    time_signature: str | None = None


class TransposeRequest(BaseModel):
    stem_label: str
    semitones: int = 0
    interval: str | None = None
    key: str | None = None


class DetectKeyRequest(BaseModel):
    stem_label: str


class SheetMusicRequest(BaseModel):
    stem_label: str
    key: str | None = None
    time_signature: str | None = None
    quantize_divisors: list[int] = [4, 3]
    title: str | None = None


# ─── Helpers ──────────────────────────────────────────────────────────

def _resolve_midi(stem_label: str, session: SessionStore):
    """Look up PrettyMIDI from session by label."""
    if stem_label == "merged":
        midi_data = session.merged_midi_data
        if midi_data is None:
            raise HTTPException(404, "No merged MIDI available")
        return midi_data
    stem_midi = session.stem_midi_data
    if stem_label not in stem_midi:
        raise HTTPException(404, f"No MIDI for stem '{stem_label}'")
    return stem_midi[stem_label]


def _store_midi(stem_label: str, midi_data, session: SessionStore) -> None:
    """Write PrettyMIDI back into session (copy-out/mutate/copy-in for thread safety)."""
    if stem_label == "merged":
        session.merged_midi_data = midi_data
    else:
        data = session.stem_midi_data
        data[stem_label] = midi_data
        session.stem_midi_data = data


# ─── Tier 1: Clean Up ────────────────────────────────────────────────

@router.post("/clean")
def clean_stem_midi(
    req: CleanUpRequest,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    """Quantize and clean a stem's MIDI, replacing it in the session."""
    from utils.music21_bridge import clean_midi

    midi_data = _resolve_midi(req.stem_label, session)
    cleaned = clean_midi(
        midi_data,
        quarter_length_divisors=tuple(req.quantize_divisors),
        min_note_quarterLength=req.min_note_length,
        key=req.key,
        time_signature=req.time_signature,
    )
    _store_midi(req.stem_label, cleaned, session)
    note_count = sum(len(inst.notes) for inst in cleaned.instruments)
    return {"stem_label": req.stem_label, "note_count": note_count, "status": "cleaned"}


# ─── Tier 2: Analysis & Transform ────────────────────────────────────

@router.post("/detect-key")
def detect_stem_key(
    req: DetectKeyRequest,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    """Run key detection on a stem's MIDI."""
    from utils.music21_bridge import detect_key

    midi_data = _resolve_midi(req.stem_label, session)
    return detect_key(midi_data)


@router.post("/transpose")
def transpose_stem_midi(
    req: TransposeRequest,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    """Transpose a stem's MIDI, replacing it in the session."""
    from utils.music21_bridge import transpose_midi

    midi_data = _resolve_midi(req.stem_label, session)
    transposed = transpose_midi(
        midi_data,
        semitones=req.semitones,
        interval=req.interval,
        key=req.key,
    )
    _store_midi(req.stem_label, transposed, session)
    note_count = sum(len(inst.notes) for inst in transposed.instruments)
    return {"stem_label": req.stem_label, "note_count": note_count, "status": "transposed"}


# ─── Tier 3: Sheet Music ─────────────────────────────────────────────

@router.post("/sheet-music")
def get_sheet_music(
    req: SheetMusicRequest,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    """Return MusicXML string for in-browser rendering via OSMD."""
    from utils.music21_bridge import to_musicxml

    midi_data = _resolve_midi(req.stem_label, session)
    musicxml = to_musicxml(
        midi_data,
        quarter_length_divisors=tuple(req.quantize_divisors),
        key=req.key,
        time_signature=req.time_signature,
        title=req.title or req.stem_label,
    )
    return {"musicxml": musicxml, "stem_label": req.stem_label}


@router.post("/sheet-music/pdf")
def get_sheet_music_pdf(
    req: SheetMusicRequest,
    session: SessionStore = Depends(get_user_session),
):
    """Return PDF file via LilyPond rendering."""
    from utils.music21_bridge import to_pdf

    midi_data = _resolve_midi(req.stem_label, session)
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MIDI_DIR / f"sheet_{req.stem_label}_{uuid.uuid4().hex[:6]}.pdf"
    to_pdf(
        midi_data,
        out_path,
        quarter_length_divisors=tuple(req.quantize_divisors),
        key=req.key,
        time_signature=req.time_signature,
        title=req.title or req.stem_label,
    )
    return FileResponse(
        out_path,
        media_type="application/pdf",
        filename=f"{req.stem_label}_sheet_music.pdf",
    )


@router.post("/sheet-music/musicxml")
def save_musicxml(
    req: SheetMusicRequest,
    session: SessionStore = Depends(get_user_session),
):
    """Return MusicXML file for import into external notation software."""
    from utils.music21_bridge import to_musicxml_file

    midi_data = _resolve_midi(req.stem_label, session)
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MIDI_DIR / f"{req.stem_label}_{uuid.uuid4().hex[:6]}.musicxml"
    to_musicxml_file(
        midi_data,
        out_path,
        quarter_length_divisors=tuple(req.quantize_divisors),
        key=req.key,
        time_signature=req.time_signature,
        title=req.title or req.stem_label,
    )
    return FileResponse(
        out_path,
        media_type="application/vnd.recordare.musicxml+xml",
        filename=f"{req.stem_label}.musicxml",
    )
