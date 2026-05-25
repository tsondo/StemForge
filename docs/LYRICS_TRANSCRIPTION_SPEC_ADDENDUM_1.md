# Lyrics Transcription Spec — Addendum 1

**Parent doc:** `LYRICS_TRANSCRIPTION_SPEC.md`
**Status:** Ready for implementation
**Scope:** Whisper hallucination mitigation. Two changes — one engine-level (root cause), one pipeline-level (defence in depth).

---

## Motivation

Field testing with `whisper-large-v3` on a sung Spanish track revealed a classic hallucination loop: 13 consecutive `¡Oh, oh, oh!` segments emitted during instrumental tails and fade-outs. The VAD filter alone (`vad_filter=True`) does not suppress this — the loop is generative, not acoustic.

Root cause is `condition_on_previous_text=True` (faster-whisper's default). The model conditions each window on its own prior output, so once it commits to a repetitive token it self-reinforces. This is a known failure mode for music transcription specifically, where instrumental passages give the model no fresh acoustic content to anchor on.

Two changes address this:

1. **Engine-level (root cause):** disable `condition_on_previous_text` by default. Net effect on quality is small for music — most sung lyric lines are short and self-contained, so cross-window context contributes little anyway. The hallucination protection is worth the trade.
2. **Pipeline-level (defence in depth):** collapse runs of duplicate segments after the engine returns, regardless of engine. Catches hallucinations that slip through and also handles the case where a song legitimately repeats a hook but the user wants a clean transcript rather than a literal one.

---

## Change 1 — Engine-level: configurable conditioning, default off

### 1.1 · `pipelines/transcribe_engines/whisper_engine.py`

Add a constructor parameter and pass it through to `model.transcribe`:

```python
class WhisperEngine:
    engine_id = "whisper"
    supports_word_timestamps = True
    requires_gpu = False

    def __init__(
        self,
        model_id: str = "whisper-base",
        *,
        condition_on_previous_text: bool = False,   # NEW
    ) -> None:
        self.model_id = model_id
        self._condition_on_previous_text = condition_on_previous_text
        self._model = None
```

In the `transcribe()` method, pass it through:

```python
segments_iter, info = self._model.transcribe(
    str(audio_path),
    word_timestamps=True,
    vad_filter=True,
    language=language,
    initial_prompt=prompt,
    condition_on_previous_text=self._condition_on_previous_text,  # NEW
)
```

### 1.2 · `pipelines/transcribe_pipeline.py` — surface as a config option

Add to `TranscribeConfig`:

```python
@dataclass(slots=True)
class TranscribeConfig:
    engine_id: str = "whisper"
    model_id: str = "whisper-base"
    language: str | None = None
    prompt: str | None = None
    output_dir: pathlib.Path | None = None
    formats: tuple[str, ...] = ("txt", "lrc", "srt")
    condition_on_previous_text: bool = False        # NEW — Whisper only; ignored by other engines
    collapse_repetitions: bool = True               # NEW — see Change 2
    max_repetition_run: int = 4                     # NEW — see Change 2
```

In `load_model()`, pass the flag when instantiating the Whisper engine:

```python
if self._config.engine_id == "whisper":
    self._engine = engine_cls(
        model_id=self._config.model_id,
        condition_on_previous_text=self._config.condition_on_previous_text,
    )
else:
    self._engine = engine_cls()
```

### 1.3 · `backend/api/transcribe.py` — surface in the request schema

Add to `TranscribeRequest`:

```python
class TranscribeRequest(BaseModel):
    audio_path: str
    engine_id: str = "whisper"
    model_id: str = "whisper-base"
    language: str | None = None
    prompt: str | None = None
    formats: list[str] = ["txt", "lrc", "srt"]
    condition_on_previous_text: bool = False        # NEW
    collapse_repetitions: bool = True               # NEW
```

Pass through into `TranscribeConfig` in `_run_transcribe`. Standard plumbing.

### 1.4 · Frontend (`frontend/components/midi.js`)

Inside the Lyrics control panel, under **Engine** but only visible when the selected engine is a Whisper variant, add an **Advanced** disclosure (`<details>`) with one checkbox:

```
[ ] Cross-window conditioning (may improve coherence but can cause repetition loops on music)
```

Unchecked by default. Tooltip: *"Whisper's default is on, but it commonly causes hallucinated repetition on instrumental passages. StemForge disables it by default for music."*

This is intentionally tucked into Advanced because the right answer for music is almost always "off" — but the option exists so a user transcribing, say, an audiobook from the Voice tab gets a way to flip it back on without code changes.

---

## Change 2 — Pipeline-level: repetition collapse

A run of identical or near-identical segments longer than `max_repetition_run` (default 4) gets collapsed to that many copies followed by an ellipsis marker. This is engine-agnostic and runs on the `TranscriptionResult` before it gets formatted into `.txt`/`.lrc`/`.srt`.

### 2.1 · New helper in `pipelines/transcribe_pipeline.py`

Add a module-level function above `TranscribePipeline`:

```python
def _collapse_repetitions(
    result: TranscriptionResult,
    *,
    max_run: int,
) -> TranscriptionResult:
    """Collapse consecutive runs of duplicate segments.

    A "duplicate" is determined by case-folded, whitespace-stripped, punctuation-
    stripped text equality.  Runs longer than `max_run` are truncated to `max_run`
    segments and followed by a single ellipsis segment whose text is "[...]" and
    whose time span covers the dropped segments.

    Word timings inside dropped segments are discarded — the ellipsis segment has
    `words=[]`.  This is intentional: a hallucinated loop's word timings are
    themselves unreliable.
    """
    if len(result.segments) <= max_run:
        return result

    def _norm(text: str) -> str:
        # Case-fold, strip whitespace, strip a small set of repetition-prone punctuation.
        stripped = text.strip().lower()
        for ch in "¡!¿?.,;:":
            stripped = stripped.replace(ch, "")
        return " ".join(stripped.split())

    out: list[TranscriptionSegment] = []
    i = 0
    segs = result.segments
    while i < len(segs):
        j = i
        key = _norm(segs[i].text)
        if not key:
            out.append(segs[i])
            i += 1
            continue
        while j < len(segs) and _norm(segs[j].text) == key:
            j += 1
        run_len = j - i
        if run_len > max_run:
            # Keep first max_run segments verbatim
            out.extend(segs[i:i + max_run])
            # Append one ellipsis segment spanning the dropped tail
            dropped_start = segs[i + max_run].start
            dropped_end = segs[j - 1].end
            out.append(TranscriptionSegment(
                start=dropped_start,
                end=dropped_end,
                text="[...]",
                words=[],
            ))
        else:
            out.extend(segs[i:j])
        i = j

    # Rebuild `text` from collapsed segments so .txt output reflects the collapse.
    new_text = " ".join(s.text.strip() for s in out if s.text.strip())

    return TranscriptionResult(
        text=new_text,
        language=result.language,
        segments=out,
        has_word_timestamps=result.has_word_timestamps,
        engine_id=result.engine_id,
        model_id=result.model_id,
    )
```

### 2.2 · Wire it into `TranscribePipeline.run()`

After the engine call, before format writing:

```python
result = self._engine.transcribe(
    audio_path,
    language=self._config.language,
    prompt=self._config.prompt,
)

if self._config.collapse_repetitions:
    result = _collapse_repetitions(
        result,
        max_run=self._config.max_repetition_run,
    )

if progress_cb:
    progress_cb(0.85, "Writing outputs…")
# … existing format-writing code follows unchanged
```

### 2.3 · LRC/SRT formatter handling of `[...]`

The existing `_format_lrc` and `_format_srt` helpers in §3.5 already iterate segments and emit one entry per segment with its `text` field — `[...]` will pass through as a literal line. That's the correct behaviour for `.srt` (it shows visibly that content was elided) and acceptable for `.lrc` (karaoke players will display `[...]` as a one-line lyric, which is honest).

No formatter changes needed.

### 2.4 · Edge cases (already handled by the implementation above, noting for review)

- **Empty segments**: `_norm()` returns `""`, the empty-key branch leaves them in place rather than collapsing them — empty segments often represent legitimate pauses, not duplicates.
- **A legitimately-repeated hook** (e.g. a song chorus that genuinely repeats 8 times): also collapses to 4 + `[...]`. This is the intended behaviour — the `.txt` output is meant to be a usable lyric sheet, not a literal performance transcript. If a user wants the literal transcript, they set `collapse_repetitions=False` in the request.
- **Word timestamps inside kept segments**: preserved as-is. Only dropped-segment word timings are discarded.

### 2.5 · Frontend control

In the same **Advanced** disclosure as Change 1.4, add:

```
[x] Collapse repeated lines (recommended for music)
     Maximum consecutive repetitions before collapse: [4]
```

Checkbox checked by default. The number input accepts 2–20, default 4.

---

## Testing

### Smoke test addition — `tests/test_transcribe.py`

Append after the existing smoke test:

```python
def test_collapse_repetitions() -> None:
    """Verify the repetition collapse helper without invoking a model."""
    from pipelines.transcribe_pipeline import _collapse_repetitions
    from pipelines.transcribe_engines import TranscriptionResult, TranscriptionSegment

    # Build a synthetic result: 2 normal lines, then 10 identical "oh oh oh"
    # segments, then 1 normal line.
    segs = [
        TranscriptionSegment(start=0.0, end=2.0, text="Verse line one", words=[]),
        TranscriptionSegment(start=2.0, end=4.0, text="Verse line two", words=[]),
    ]
    for i in range(10):
        segs.append(TranscriptionSegment(
            start=4.0 + i * 0.5,
            end=4.5 + i * 0.5,
            text="¡Oh, oh, oh!",
            words=[],
        ))
    segs.append(TranscriptionSegment(start=9.0, end=11.0, text="Final line", words=[]))

    result = TranscriptionResult(
        text="(unused)",
        language="es",
        segments=segs,
        has_word_timestamps=False,
        engine_id="whisper",
        model_id="whisper-large-v3",
    )

    collapsed = _collapse_repetitions(result, max_run=4)

    # Expect: 2 verse + 4 oh + 1 ellipsis + 1 final = 8 segments
    assert len(collapsed.segments) == 8, f"got {len(collapsed.segments)} segments"
    assert collapsed.segments[6].text == "[...]"
    assert collapsed.segments[6].start == 6.0   # 5th oh started at 4.0 + 4*0.5
    assert collapsed.segments[7].text == "Final line"
    print("collapse_repetitions OK")


if __name__ == "__main__":
    main()
    test_collapse_repetitions()
```

### Manual checklist additions

- [ ] Re-run the Catrina song through `whisper-large-v3` with defaults. Confirm the trailing `¡Oh, oh, oh!` runs are gone (or collapsed to 4 + `[...]`).
- [ ] Toggle the **Cross-window conditioning** checkbox on in the UI and re-run. Confirm the hallucination loop returns — this is the regression test for the toggle actually wiring through.
- [ ] Toggle **Collapse repeated lines** off and re-run with conditioning still on. Confirm the literal repetition is preserved end-to-end (this proves both toggles work independently).
- [ ] Run on a song with a legitimately repeated chorus (>4 repeats). Confirm the collapse fires and the `.srt` output shows `[...]` for the elided run. This is expected behaviour, not a bug.

---

## Definition of Done (addendum)

Append to §8 of the parent spec:

9. `condition_on_previous_text=False` is the default in `WhisperEngine`, surfaced through `TranscribeConfig` and the API request schema.
10. `_collapse_repetitions` helper exists, is wired into `TranscribePipeline.run()`, defaults to enabled with `max_run=4`.
11. Both Advanced toggles render in the MIDI tab's Lyrics mode and round-trip correctly (off-by-default for conditioning, on-by-default for collapse).
12. `tests/test_transcribe.py::test_collapse_repetitions` passes.
13. Re-running the test track shows no spurious `¡Oh, oh, oh!` tail.
