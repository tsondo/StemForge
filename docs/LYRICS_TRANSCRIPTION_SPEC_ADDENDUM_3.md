# Lyrics Transcription Spec — Addendum 3

**Parent doc:** `LYRICS_TRANSCRIPTION_SPEC.md` (plus Addenda 1 and 2)
**Status:** Ready for implementation
**Scope:** Qwen-only. Replace naive sequential chunking with overlap-and-stitch chunking inside `QwenEngine.transcribe()`. Whisper is unaffected — faster-whisper handles long audio internally.

---

## 1 · Motivation

Qwen2-Audio has a 30-second audio context limit. Field testing on a 3-minute sung Spanish track with sequential non-overlapping 30s chunks produced:

- Bisected lyric lines at every chunk boundary ("Tu bondad es un rey..." cut off mid-phrase).
- Gibberish at chunk starts ("Eless con piernas" — the model recovering from a mid-syllable splice).
- Run-together words at chunk ends ("Katrina mi vida son de ser").

These are windowing failures, not model quality failures. The fix is a standard overlap-add scheme adapted for text rather than waveforms.

**Choice of parameters: 24-second chunks with 6-second overlap.** Step size 18 s. Selected because:
- 6 s reliably contains alignable text on sparse sung audio (verses with pauses, instrumental tags between phrases).
- 24 s leaves headroom inside Qwen's 30 s limit for processor padding and prompt tokens.
- 20% overlap ratio is the standard sweet spot for overlap-add windowed processing.
- A typical 3-minute song produces ~10 chunks vs. ~6 non-overlapping — acceptable compute cost.

Adaptive chunk sizing based on vocal density is rejected as over-engineering for v1.

---

## 2 · Architectural placement

The chunking logic lives in a new helper module **shared with no other engine**. Whisper's engine and the pipeline orchestrator remain untouched.

- New file: `pipelines/transcribe_engines/_qwen_chunker.py`
- Modified file: `pipelines/transcribe_engines/qwen_engine.py` — `transcribe()` is rewritten to call the chunker.

The chunker is a private module (leading underscore) because nothing outside `qwen_engine.py` should depend on it. If a second engine ever needs the same pattern, the right move is to promote it to a public module then; not now.

---

## 3 · New file: `pipelines/transcribe_engines/_qwen_chunker.py`

```python
"""Overlap-and-stitch chunking for Qwen2-Audio.

Qwen2-Audio has a hard 30-second audio context limit. For longer audio,
we slice the input into 24-second chunks with 6-second overlaps, transcribe
each chunk independently, and stitch the resulting text using longest-common-
substring matching on the overlapping tail/head regions.

Design constants
----------------
CHUNK_DURATION_S      : Audio window passed to the model per call. Must be
                        comfortably below Qwen's 30s limit to leave room for
                        processor padding and prompt tokens.
OVERLAP_DURATION_S    : Redundant audio at each chunk boundary. Chosen to
                        reliably contain 2+ alignable tokens on sparse sung
                        audio while keeping compute overhead acceptable.
MIN_MATCH_TOKENS      : Minimum LCS length (in tokens) to accept a stitch.
                        Below this, fall back to concatenation with no
                        deduplication — redundant text is better than dropped
                        text.
ALIGN_WINDOW_FRACTION : Fraction of each chunk's text searched for an
                        alignment match. 0.30 means the last 30% of chunk N
                        is matched against the first 30% of chunk N+1.
                        Larger than the overlap fraction (20%) to absorb
                        timing jitter — the text-time mapping is not linear.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

CHUNK_DURATION_S: float = 24.0
OVERLAP_DURATION_S: float = 6.0
STEP_DURATION_S: float = CHUNK_DURATION_S - OVERLAP_DURATION_S  # 18.0

MIN_MATCH_TOKENS: int = 2
ALIGN_WINDOW_FRACTION: float = 0.30

# Sample rate used by Qwen2-Audio's audio processor.
SAMPLE_RATE: int = 16_000


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A single audio slice scheduled for transcription."""
    index: int                  # 0-based chunk index
    start_s: float              # start time in source audio
    end_s: float                # end time in source audio
    samples: np.ndarray         # mono int16/float32 audio at SAMPLE_RATE


def slice_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> list[AudioChunk]:
    """Slice a mono audio array into overlapping chunks.

    The final chunk extends to the end of the audio regardless of size — it
    will be shorter than CHUNK_DURATION_S if the audio doesn't divide evenly.
    """
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}")
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}")

    total_samples = len(audio)
    chunk_samples = int(CHUNK_DURATION_S * SAMPLE_RATE)
    step_samples = int(STEP_DURATION_S * SAMPLE_RATE)

    # Single-chunk fast path: audio fits inside one window.
    if total_samples <= chunk_samples:
        return [AudioChunk(
            index=0,
            start_s=0.0,
            end_s=total_samples / SAMPLE_RATE,
            samples=audio,
        )]

    chunks: list[AudioChunk] = []
    pos = 0
    idx = 0
    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunks.append(AudioChunk(
            index=idx,
            start_s=pos / SAMPLE_RATE,
            end_s=end / SAMPLE_RATE,
            samples=audio[pos:end],
        ))
        # If this chunk reached the end, stop — don't emit a redundant final
        # chunk that's just the tail of the previous one.
        if end >= total_samples:
            break
        pos += step_samples
        idx += 1
    return chunks


# ── Text stitching ────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"\S+")


def _tokenize(text: str) -> list[str]:
    """Whitespace tokens, preserving original casing and punctuation.

    Normalization for *matching* happens in _norm_for_match. We keep the
    original tokens around so the final stitched output preserves the
    model's chosen capitalization and punctuation.
    """
    return _TOKEN_RE.findall(text)


def _norm_for_match(token: str) -> str:
    """Casefold and strip a small set of repetition-prone punctuation."""
    stripped = token.lower()
    for ch in "¡!¿?.,;:\"'":
        stripped = stripped.replace(ch, "")
    return stripped


def _longest_common_substring(a: list[str], b: list[str]) -> tuple[int, int, int]:
    """Find the longest contiguous token substring shared by a and b.

    Returns (length, a_start_idx, b_start_idx). Length 0 means no match.
    Token equality is determined under _norm_for_match.

    O(len(a) * len(b)) time, O(min(len(a), len(b))) space.
    """
    if not a or not b:
        return (0, 0, 0)
    na = [_norm_for_match(t) for t in a]
    nb = [_norm_for_match(t) for t in b]

    # Rolling 1-D DP — only need the previous row.
    prev = [0] * (len(nb) + 1)
    curr = [0] * (len(nb) + 1)
    best_len = 0
    best_a = 0
    best_b = 0
    for i in range(1, len(na) + 1):
        for j in range(1, len(nb) + 1):
            if na[i - 1] == nb[j - 1] and na[i - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_a = i - best_len
                    best_b = j - best_len
            else:
                curr[j] = 0
        prev, curr = curr, prev
        for j in range(len(curr)):
            curr[j] = 0
    return (best_len, best_a, best_b)


def stitch_chunks(chunk_texts: list[str]) -> str:
    """Stitch overlapping chunk transcripts into a single transcript.

    For each adjacent pair (N, N+1):
      1. Take the last ALIGN_WINDOW_FRACTION of N's tokens (tail).
      2. Take the first ALIGN_WINDOW_FRACTION of N+1's tokens (head).
      3. Find the longest common substring between tail and head.
      4. If LCS length >= MIN_MATCH_TOKENS:
           - Cut N at the *start* of its matched region.
           - Cut N+1 at the *end* of its matched region.
           - The matched region itself is kept from N (could be either; N
             tends to be slightly more reliable than the start of N+1, which
             can show splice artifacts).
      5. Otherwise: fall back to plain concatenation, accepting duplication.
         This is the correct degradation — redundant text is recoverable by
         the user; dropped text is not.
    """
    if not chunk_texts:
        return ""
    if len(chunk_texts) == 1:
        return chunk_texts[0].strip()

    # Tokenize all chunks once.
    tokens_per_chunk: list[list[str]] = [_tokenize(t) for t in chunk_texts]

    # Start with chunk 0's tokens in full.
    out_tokens: list[str] = list(tokens_per_chunk[0])

    for i in range(1, len(tokens_per_chunk)):
        next_tokens = tokens_per_chunk[i]
        if not next_tokens:
            continue
        if not out_tokens:
            out_tokens = list(next_tokens)
            continue

        # Tail of accumulated output, head of next chunk.
        tail_len = max(1, int(len(out_tokens) * ALIGN_WINDOW_FRACTION))
        head_len = max(1, int(len(next_tokens) * ALIGN_WINDOW_FRACTION))
        tail = out_tokens[-tail_len:]
        head = next_tokens[:head_len]

        length, tail_start, head_start = _longest_common_substring(tail, head)

        if length >= MIN_MATCH_TOKENS:
            # Absolute index in out_tokens where the match starts.
            cut_out_at = len(out_tokens) - tail_len + tail_start
            # Absolute index in next_tokens where the match ends.
            cut_next_at = head_start + length
            # Keep out_tokens[:cut_out_at] + tail-match + next_tokens[cut_next_at:]
            matched_region = tail[tail_start:tail_start + length]
            out_tokens = out_tokens[:cut_out_at] + matched_region + list(next_tokens[cut_next_at:])
            log.debug(
                "Stitched chunk %d → %d: matched %d tokens (%r)",
                i - 1, i, length, " ".join(matched_region),
            )
        else:
            # No reliable match — concatenate with a soft separator.
            # Use a newline so the unstitched join is visible in the .txt
            # output without inventing punctuation the model didn't emit.
            out_tokens = out_tokens + ["\n"] + list(next_tokens)
            log.info(
                "No alignment between chunk %d and %d (best match: %d tokens) "
                "— falling back to concatenation.",
                i - 1, i, length,
            )

    # Reassemble with single spaces, but preserve any "\n" sentinel we inserted.
    result_parts: list[str] = []
    for tok in out_tokens:
        if tok == "\n":
            # Strip trailing space from previous part, append newline.
            if result_parts and result_parts[-1].endswith(" "):
                result_parts[-1] = result_parts[-1].rstrip()
            result_parts.append("\n")
        else:
            if result_parts and not result_parts[-1].endswith("\n"):
                result_parts.append(" ")
            result_parts.append(tok)
    return "".join(result_parts).strip()
```

---

## 4 · Modified file: `pipelines/transcribe_engines/qwen_engine.py`

Replace the existing `transcribe()` method. The new version slices, calls the model per chunk, stitches, and returns a single `TranscriptionResult` covering the full duration.

```python
    def transcribe(
        self,
        audio_path: pathlib.Path,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        self.load()
        try:
            import librosa
        except ImportError as exc:
            raise ModelLoadError(
                "librosa is required.", model_name=self.model_id,
            ) from exc

        from ._qwen_chunker import SAMPLE_RATE, slice_audio, stitch_chunks

        try:
            audio, _sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        except Exception as exc:
            raise PipelineExecutionError(
                f"Failed to load audio '{audio_path.name}': {exc}",
                pipeline_name="transcribe",
            ) from exc

        duration_s = float(len(audio)) / SAMPLE_RATE
        chunks = slice_audio(audio, sample_rate=SAMPLE_RATE)
        log.info(
            "Qwen transcribing %.1fs of audio in %d chunk(s).",
            duration_s, len(chunks),
        )

        chunk_texts: list[str] = []
        for chunk in chunks:
            try:
                text = self._transcribe_chunk(
                    chunk.samples,
                    language=language,
                    prompt=prompt,
                )
            except Exception as exc:
                raise PipelineExecutionError(
                    f"Qwen2-Audio generation failed on chunk {chunk.index} "
                    f"({chunk.start_s:.1f}-{chunk.end_s:.1f}s): {exc}",
                    pipeline_name="transcribe",
                ) from exc
            chunk_texts.append(text)
            log.debug(
                "Chunk %d/%d (%.1f-%.1fs): %d chars",
                chunk.index + 1, len(chunks),
                chunk.start_s, chunk.end_s, len(text),
            )

        stitched = stitch_chunks(chunk_texts)

        # The pipeline expects segments. For Qwen we still emit a single
        # segment covering the whole duration — we don't have per-chunk
        # timestamps that align with the stitched text.
        segment = TranscriptionSegment(
            start=0.0, end=duration_s, text=stitched, words=[],
        )
        return TranscriptionResult(
            text=stitched,
            language=language,
            segments=[segment],
            has_word_timestamps=False,
            engine_id=self.engine_id,
            model_id=self.model_id,
        )

    def _transcribe_chunk(
        self,
        audio: "np.ndarray",
        *,
        language: str | None,
        prompt: str | None,
    ) -> str:
        """Run a single Qwen forward pass on one audio chunk."""
        user_prompt = prompt or _PROMPT
        if language:
            user_prompt += f" The lyrics are in {language}."
        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "in_memory"},
                {"type": "text", "text": user_prompt},
            ]},
        ]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )
        inputs = self._processor(
            text=text, audios=[audio], sampling_rate=16_000,
            return_tensors="pt", padding=True,
        ).to("cuda")

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )
        generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
        output = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0].strip()
        return output
```

Notes on the rewrite:

- `_transcribe_chunk` is split out so each forward pass is a single small method — easier to wrap with timing, retries, or per-chunk progress callbacks later if needed.
- `max_new_tokens` is reduced to 512 from the parent spec's 1024. With 24-second chunks at typical singing pace, 512 tokens is generous; the previous 1024 was sized for the full-track case that no longer happens.
- The in-memory audio path uses `"audio_url": "in_memory"` as a sentinel because the processor reads audio from the `audios=` kwarg rather than the conversation URL when both are provided. If the upstream processor version doesn't accept this pattern, the alternative is to dump each chunk to a temp `.wav` and pass the path — slower but bulletproof.
- The fp16 and 4-bit code paths use the same chunker — both have the same 30 s context limit.

---

## 5 · Pipeline integration

No changes to `pipelines/transcribe_pipeline.py`. The chunker is internal to `QwenEngine`; the pipeline still sees a single `transcribe()` call returning a single `TranscriptionResult`.

---

## 6 · Progress reporting

Per-chunk progress callbacks would be a useful UX improvement (a 10-chunk transcription currently appears frozen until the whole thing finishes), but the parent spec only plumbs one progress callback down to `TranscribePipeline.run()` — not into individual engines. Wiring per-chunk progress all the way to the UI would require changes to the engine Protocol, the pipeline, the API job state, and the frontend polling logic.

**Decision: out of scope for this addendum.** Note as a follow-up. For now, the existing `progress_cb(0.1, "Transcribing…")` covers the entire chunked run.

If chunking causes user-visible pauses long enough that the UI looks frozen, the cheapest interim fix is to bump the "Transcribing…" progress to 0.15, 0.20, 0.25, ... after each chunk inside the engine by reaching through to a module-level callback ref — ugly but functional. Skip this unless it actually becomes a problem.

---

## 7 · Testing

### 7.1 · New unit tests in `tests/test_transcribe.py`

The chunker and stitcher are pure functions of arrays and strings — perfect for unit testing without a model. Add these after `test_collapse_repetitions`:

```python
def test_qwen_chunker_single_chunk() -> None:
    """Audio shorter than chunk size should produce exactly one chunk."""
    import numpy as np
    from pipelines.transcribe_engines._qwen_chunker import (
        slice_audio, CHUNK_DURATION_S, SAMPLE_RATE,
    )
    # 10 seconds of silence
    audio = np.zeros(int(10 * SAMPLE_RATE), dtype=np.float32)
    chunks = slice_audio(audio)
    assert len(chunks) == 1, f"got {len(chunks)} chunks for 10s audio"
    assert chunks[0].start_s == 0.0
    assert abs(chunks[0].end_s - 10.0) < 0.01
    print("qwen_chunker single OK")


def test_qwen_chunker_overlap_layout() -> None:
    """Verify chunk boundaries match the documented step/overlap values."""
    import numpy as np
    from pipelines.transcribe_engines._qwen_chunker import (
        slice_audio, CHUNK_DURATION_S, OVERLAP_DURATION_S, STEP_DURATION_S,
        SAMPLE_RATE,
    )
    # 60 seconds → expect chunks at [0-24], [18-42], [36-60].
    audio = np.zeros(int(60 * SAMPLE_RATE), dtype=np.float32)
    chunks = slice_audio(audio)
    assert len(chunks) == 3, f"got {len(chunks)} chunks for 60s audio"
    assert abs(chunks[0].start_s - 0.0) < 0.01
    assert abs(chunks[1].start_s - STEP_DURATION_S) < 0.01    # 18.0
    assert abs(chunks[2].start_s - 2 * STEP_DURATION_S) < 0.01  # 36.0
    assert abs(chunks[0].end_s - CHUNK_DURATION_S) < 0.01       # 24.0
    assert abs(chunks[-1].end_s - 60.0) < 0.01
    print("qwen_chunker overlap layout OK")


def test_qwen_chunker_irregular_tail() -> None:
    """Final chunk should be truncated to actual audio length, not padded."""
    import numpy as np
    from pipelines.transcribe_engines._qwen_chunker import (
        slice_audio, SAMPLE_RATE,
    )
    # 50 seconds → [0-24], [18-42], [36-50]
    audio = np.zeros(int(50 * SAMPLE_RATE), dtype=np.float32)
    chunks = slice_audio(audio)
    assert len(chunks) == 3
    assert abs(chunks[-1].end_s - 50.0) < 0.01
    assert abs(chunks[-1].start_s - 36.0) < 0.01
    # The final chunk is 14 seconds, not 24.
    print("qwen_chunker irregular tail OK")


def test_qwen_stitcher_clean_overlap() -> None:
    """Two chunks with a clear shared substring should stitch cleanly."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    chunk_a = "feliz cumpleaños mi princesa eres mi sol mi fortaleza"
    chunk_b = "mi sol mi fortaleza con tu amor llenas mi vida"
    stitched = stitch_chunks([chunk_a, chunk_b])
    # The shared "mi sol mi fortaleza" should appear exactly once.
    assert stitched.lower().count("mi sol mi fortaleza") == 1, stitched
    assert "feliz cumpleaños" in stitched.lower()
    assert "llenas mi vida" in stitched.lower()
    print("qwen_stitcher clean overlap OK")


def test_qwen_stitcher_no_overlap_falls_back() -> None:
    """When chunks share no content, both are preserved with a separator."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    chunk_a = "uno dos tres cuatro cinco"
    chunk_b = "siete ocho nueve diez once"
    stitched = stitch_chunks([chunk_a, chunk_b])
    assert "uno dos tres" in stitched
    assert "nueve diez once" in stitched
    print("qwen_stitcher fallback OK")


def test_qwen_stitcher_single_chunk() -> None:
    """Single-chunk input should be returned unchanged."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    stitched = stitch_chunks(["hello world"])
    assert stitched == "hello world"
    print("qwen_stitcher single chunk OK")


# Add the new tests to the __main__ block:
if __name__ == "__main__":
    main()
    test_collapse_repetitions()
    test_qwen_chunker_single_chunk()
    test_qwen_chunker_overlap_layout()
    test_qwen_chunker_irregular_tail()
    test_qwen_stitcher_clean_overlap()
    test_qwen_stitcher_no_overlap_falls_back()
    test_qwen_stitcher_single_chunk()
```

These tests cost nothing (no model load, no audio I/O) and catch the most likely regressions in either component.

### 7.2 · Manual checklist

- [ ] Re-run the Catrina stem through `Qwen2-Audio 7B (4-bit)`. Compare:
    - Chunk-boundary gibberish from the prior run ("Eless con piernas") should be absent.
    - Lyric lines that previously spanned boundaries should be intact.
    - Total transcription time roughly proportional to chunk count (~10 chunks for 3 minutes).
- [ ] Confirm log lines show the expected chunk count and stitch outcomes (`Stitched chunk N → N+1: matched X tokens` or `No alignment between chunk N and N+1`).
- [ ] On a very short audio file (<24 s), confirm the single-chunk fast path runs (one model call, no stitching).
- [ ] On an instrumental track with very sparse vocals, confirm the fallback path triggers cleanly and the output contains the soft `\n` separator at unstitched joins rather than failing.

---

## 8 · UI copy update

Addendum 2 §3.1 suggested adding `(experimental, 30s chunks)` to the Qwen dropdown labels. With chunking now properly handled, downgrade that to just the VRAM annotation:

```
Qwen2-Audio 7B (4-bit)      (GPU required — ~9 GB VRAM)
Qwen2-Audio 7B              (GPU required — ~16 GB VRAM)
```

If you'd like to flag that Qwen output quality is still being evaluated relative to Whisper, the right place is the engine tooltip, not the per-option suffix. Suggested tooltip addition (append to the existing one from Addendum 2 §3.4):

> Long audio is automatically chunked into overlapping 24-second windows; transcripts are stitched together by matching text in the overlap regions.

---

## 9 · Definition of Done (addendum)

Append to §8 of the parent spec:

22. `pipelines/transcribe_engines/_qwen_chunker.py` exists with `slice_audio` and `stitch_chunks` implemented per §3.
23. `QwenEngine.transcribe()` uses the chunker; `_transcribe_chunk` helper extracted.
24. All six new unit tests in `tests/test_transcribe.py` pass.
25. Re-running the Catrina stem through Qwen 4-bit shows no chunk-boundary gibberish ("Eless con piernas"-type failures).
26. Whisper transcription is byte-for-byte unchanged — chunker code is reachable only via `QwenEngine`.
27. Logs show `Qwen transcribing N.Ns of audio in M chunk(s).` followed by per-pair `Stitched chunk X → X+1: matched Y tokens` lines on a multi-chunk run.
