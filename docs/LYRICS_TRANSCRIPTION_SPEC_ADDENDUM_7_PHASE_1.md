# Lyrics Transcription Spec — Addendum 7, Phase 1

**Parent doc:** `LYRICS_TRANSCRIPTION_SPEC.md` (plus Addenda 1-6)
**Status:** Ready for implementation
**Scope:** Diagnostic logging only. No functional behavior changes. Two files touched: `pipelines/transcribe_engines/qwen_engine.py` and `pipelines/transcribe_engines/_qwen_chunker.py`. No tests, no UI, no schema, no Whisper, no pipeline changes.

---

## 1 · Motivation

After Addendum 6, a manual test with hint `"Feliz Cumpleanos Caterina"` produced ~50 words of output for a 3-minute song that should have produced ~10 chunks of content. Two competing hypotheses explain the result:

1. **Stitcher over-collapse.** Each chunk produces good, diverse lyric content but every chunk's output is bookended by the hint phrase (Qwen echoing the hint as a meta-label). The pairwise stitcher matches the bookend phrases as "overlap" and discards the actual lyric content between them.
2. **Hint priming overrides audio attention.** Each chunk's output is itself short and converges on a hint-driven template, leaving the stitcher little material to work with regardless of its behavior.

The fixes for these two hypotheses are different and partially incompatible:

- **(1)** wants per-chunk output left intact and the stitcher hardened against bookend matches, or hint echoes stripped before stitching.
- **(2)** wants the prompt redesigned so the model attends to audio properly, with stitcher behavior left alone.

Implementing both as a precaution is possible but risks treating a symptom rather than the cause. A single round of diagnostic logging will determine which hypothesis is correct, after which Phase 2 of this addendum will land the targeted fix.

---

## 2 · Required diagnostic signals

To distinguish the hypotheses, three numbers per chunk are needed:

1. **Char count of raw model output.** Low and similar across chunks → Hypothesis 2. High and varying → Hypothesis 1.
2. **First 80 characters of raw model output.** Lets the user (and the next analysis turn) see whether the hint phrase appears verbatim at chunk starts, and whether the lyric content varies across chunks.
3. **LCS match length per stitch decision.** Anomalously large matches (e.g. >50% of a chunk's tokens) confirm over-collapse. Small or zero matches with truncated output confirm Hypothesis 2.

The chunker layout log line already exists per Addendum 3 (`Qwen transcribing N.Ns of audio in M chunk(s).`) and does not need changes.

---

## 3 · Implementation

### 3.1 · `pipelines/transcribe_engines/qwen_engine.py` — per-chunk INFO log

Locate the chunk iteration loop inside `transcribe()` (added in Addendum 3 §4). It currently emits a DEBUG-level message:

```python
chunk_texts.append(text)
log.debug(
    "Chunk %d/%d (%.1f-%.1fs): %d chars",
    chunk.index + 1, len(chunks),
    chunk.start_s, chunk.end_s, len(text),
)
```

Replace the existing `log.debug(...)` call with an INFO-level call that includes a text preview:

```python
chunk_texts.append(text)
preview = text.replace("\n", " ⏎ ").strip()
if len(preview) > 80:
    preview = preview[:80] + "…"
log.info(
    "Qwen chunk %d/%d (%.1f-%.1fs): %d chars | %s",
    chunk.index + 1, len(chunks),
    chunk.start_s, chunk.end_s, len(text), preview,
)
```

Three deliberate choices:

- **INFO, not DEBUG.** The user needs these lines visible in normal log output to paste them back. Promoting to INFO temporarily is acceptable; Phase 2 will demote back to DEBUG once the diagnosis is settled.
- **Newlines collapsed in the preview.** A `⏎` glyph makes the structure visible (chunk-internal line breaks are signal) without breaking log formatting.
- **80-character cap with ellipsis.** Long enough to see the hint-echo pattern at chunk starts, short enough not to flood the log with full lyric content.

### 3.2 · `pipelines/transcribe_engines/_qwen_chunker.py` — promote stitcher logs to INFO

The stitcher already logs match/no-match decisions, but the match case is at DEBUG and the no-match case is at INFO (per Addendum 3 §3). Promote the match log to INFO so the user sees both decision types in one log dump without enabling debug.

Find:

```python
log.debug(
    "Stitched chunk %d → %d: matched %d tokens (%r)",
    i - 1, i, length,
    " ".join(tail[tail_start:tail_start + length]),
)
```

Change `log.debug` to `log.info`. No other changes — the existing message content is exactly what's needed.

Also augment the message with the tail and head lengths so the user can see the ratio:

```python
log.info(
    "Stitched chunk %d → %d: matched %d tokens of %d/%d tail/head (%r)",
    i - 1, i, length, tail_len, head_len,
    " ".join(tail[tail_start:tail_start + length]),
)
```

`tail_len` and `head_len` are already locals in that scope from Addendum 4's rewrite. If the match is 5 tokens against a tail of 8 and a head of 10, that's 63% of the tail consumed — a number the user can eyeball.

### 3.3 · Reproducibility

To run the diagnostic, the user simply re-runs the same Catrina-stem transcription that produced the truncated output. No code changes outside this addendum, no test fixtures, no flag toggling.

---

## 4 · What the user does

1. Apply this addendum via Claude Code.
2. Restart `uv run stemforge`.
3. In the UI, run **Qwen2-Audio 7B (4-bit)** on the same Catrina stem with hint = `"Feliz Cumpleanos Caterina"` — exactly the run that produced the truncated output.
4. Copy the full server log output covering that transcription run. The relevant lines will start with `Qwen transcribing N.Ns of audio in M chunk(s).` and include M `Qwen chunk N/M (...)` lines plus M-1 `Stitched chunk N → N+1` (or `No alignment between chunk N and N+1`) lines.
5. Paste the log back. Phase 2 of this addendum will follow with a targeted fix.

---

## 5 · Expected output shapes

For the user's reference when reading the log:

**If Hypothesis 1 (stitcher over-collapse):**

```
Qwen chunk 1/10 (0.0-24.0s): 312 chars | Feliz Cumpleanos Caterina. Es mi Caterina la que brille siempre con su…
Qwen chunk 2/10 (18.0-42.0s): 295 chars | Feliz Cumpleanos Caterina. cada paso suyo es un poema la niña de mis…
Qwen chunk 3/10 (36.0-60.0s): 308 chars | Feliz Cumpleanos Caterina. Feliz cumpleaños mi princesa eres mi sol mi…
Stitched chunk 0 → 1: matched 4 tokens of 8/9 tail/head ('feliz cumpleanos caterina')
Stitched chunk 1 → 2: matched 4 tokens of 9/9 tail/head ('feliz cumpleanos caterina')
...
```

Each chunk is substantial (~300 chars) and lyric-diverse, but every match is exactly the hint phrase against itself. Big tail-fraction match.

**If Hypothesis 2 (hint priming overrides audio):**

```
Qwen chunk 1/10 (0.0-24.0s): 38 chars | Feliz Cumpleanos Caterina.
Qwen chunk 2/10 (18.0-42.0s): 42 chars | Feliz Cumpleanos Caterina. Una reina.
Qwen chunk 3/10 (36.0-60.0s): 36 chars | Feliz Cumpleanos Caterina.
Stitched chunk 0 → 1: matched 3 tokens of 1/2 tail/head ('feliz cumpleanos caterina')
No alignment between chunk 1 and 2 (best match: 0 tokens)
...
```

Char counts are tiny and dominated by the hint phrase. There is no actual lyric content to stitch — most of the song's audio went un-transcribed.

**If a hybrid:**

Char counts moderate (~150-200), match lengths inconsistent — partial priming where some chunks produce useful content and others don't.

---

## 6 · Definition of Done (addendum)

Append to §8 of the parent spec:

47. `qwen_engine.py` emits one INFO-level log line per chunk in the format `Qwen chunk N/M (X.X-Y.Ys): Z chars | <preview>` where preview is the first 80 chars of the raw model output with newlines rendered as `⏎`.
48. `_qwen_chunker.py` emits one INFO-level log line per stitch decision, both for successful stitches (`Stitched chunk N → N+1: matched X tokens of A/B tail/head (...)`) and for fallback non-matches (existing message, already INFO).
49. No other behavior changes — re-running the Catrina stem produces the same truncated output as before, with the addition of the diagnostic log lines.
50. Phase 2 of this addendum (the actual fix) is held pending review of the log output.
