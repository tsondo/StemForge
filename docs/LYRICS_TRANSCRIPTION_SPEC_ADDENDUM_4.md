# Lyrics Transcription Spec — Addendum 4

**Parent doc:** `LYRICS_TRANSCRIPTION_SPEC.md` (plus Addenda 1, 2, 3)
**Status:** Ready for implementation
**Scope:** Qwen-only. Two changes to `stitch_chunks()` in `pipelines/transcribe_engines/_qwen_chunker.py`. No changes to `slice_audio()`, the engine, the pipeline, the API, or the frontend.

---

## 1 · Motivation

After Addendum 3, chunk-boundary gibberish ("Eless con piernas") is gone, but a new failure mode emerged on chorus-heavy material: when the second chorus repeats the first chorus's lyrics, the LCS matcher can either match against the wrong (distant) instance of those lyrics or fail to match at all, dropping into the soft-newline fallback path. On the Catrina test track this produced three consecutive failed stitches at the second chorus.

Two root causes:

1. **The matcher operates on the entire accumulated output's tail.** After N chunks, the "tail" contains content from acoustically distant points in the song. On repetitive material this introduces ambiguous matches.
2. **`MIN_MATCH_TOKENS = 2` is too strict for sparse sung audio.** When the 6-second overlap region contains only one distinct content word (common in slow ballads or instrumental-heavy choruses), the matcher refuses to stitch and falls back to concatenation, producing visible artifacts.

Both fixes are local to `stitch_chunks()`.

---

## 2 · Change 1 — Pairwise positional matching

### 2.1 · Current behavior

For each new chunk, `stitch_chunks` searches the last `ALIGN_WINDOW_FRACTION` of *the accumulated output* against the first `ALIGN_WINDOW_FRACTION` of the new chunk. This means by chunk 5, the matcher can compare chunk 5's start against text that originated in chunks 1-4 — including lyrics from a previous chorus that's nearly identical to the current one.

### 2.2 · New behavior

For each new chunk, only the **immediately previous chunk's text** is searched. The accumulated output is appended to but never re-matched.

This guarantees the matcher compares texts that the chunker *intended* to overlap acoustically — chunk N's tail and chunk N+1's head correspond to the same 6 seconds of source audio. Distant repetitions are now structurally impossible to false-match against.

### 2.3 · Implementation

Replace the accumulator loop in `stitch_chunks` with this structure:

```python
def stitch_chunks(chunk_texts: list[str]) -> str:
    if not chunk_texts:
        return ""
    if len(chunk_texts) == 1:
        return chunk_texts[0].strip()

    tokens_per_chunk: list[list[str]] = [_tokenize(t) for t in chunk_texts]

    # Output is built chunk by chunk. For stitching, we only ever compare
    # chunk N's tail against chunk N+1's head — never the full accumulated
    # output. This avoids false matches against distant repetitions of
    # similar text (e.g. a chorus that repeats minutes apart).
    out_tokens: list[str] = list(tokens_per_chunk[0])
    prev_tokens: list[str] = list(tokens_per_chunk[0])

    for i in range(1, len(tokens_per_chunk)):
        next_tokens = tokens_per_chunk[i]
        if not next_tokens:
            continue
        if not prev_tokens:
            out_tokens.extend(next_tokens)
            prev_tokens = list(next_tokens)
            continue

        tail_len = max(1, int(len(prev_tokens) * ALIGN_WINDOW_FRACTION))
        head_len = max(1, int(len(next_tokens) * ALIGN_WINDOW_FRACTION))
        tail = prev_tokens[-tail_len:]
        head = next_tokens[:head_len]

        length, tail_start, head_start = _longest_common_substring(tail, head)

        if _is_acceptable_match(length, tail, tail_start):
            # Replace the tail of the previous chunk in out_tokens with the
            # matched region from prev_tokens, then append the unmatched
            # portion of next_tokens. The matched region itself is taken
            # from the previous chunk (slightly more reliable than chunk
            # starts, which can show splice artifacts).
            prev_match_end_in_prev = len(prev_tokens) - tail_len + tail_start + length
            tokens_to_drop_from_out = len(prev_tokens) - prev_match_end_in_prev
            if tokens_to_drop_from_out > 0:
                out_tokens = out_tokens[:-tokens_to_drop_from_out]
            # The matched region is already at the end of out_tokens at this
            # point — we just need to append the post-match part of next.
            cut_next_at = head_start + length
            appended = list(next_tokens[cut_next_at:])
            out_tokens.extend(appended)
            log.debug(
                "Stitched chunk %d → %d: matched %d tokens (%r)",
                i - 1, i, length,
                " ".join(tail[tail_start:tail_start + length]),
            )
        else:
            out_tokens.append("\n")
            out_tokens.extend(next_tokens)
            log.info(
                "No alignment between chunk %d and %d (best match: %d tokens) "
                "— falling back to concatenation.",
                i - 1, i, length,
            )

        prev_tokens = list(next_tokens)

    # Reassembly (unchanged from Addendum 3).
    result_parts: list[str] = []
    for tok in out_tokens:
        if tok == "\n":
            if result_parts and result_parts[-1].endswith(" "):
                result_parts[-1] = result_parts[-1].rstrip()
            result_parts.append("\n")
        else:
            if result_parts and not result_parts[-1].endswith("\n"):
                result_parts.append(" ")
            result_parts.append(tok)
    return "".join(result_parts).strip()
```

The accumulator (`out_tokens`) is still maintained because the final return value needs to be the full concatenated transcript. But the **matcher** only ever looks at `prev_tokens` (a single chunk's worth), which is rotated each iteration.

---

## 3 · Change 2 — Relaxed match threshold with safeguards

### 3.1 · Current behavior

`MIN_MATCH_TOKENS = 2`. A single matching word triggers the fallback path.

### 3.2 · New behavior

Single-token matches are accepted, but only when the token is "content-bearing" — long enough and not on a stop-word list. A new `_is_acceptable_match` helper centralizes this logic so it sits next to the matcher.

### 3.3 · Implementation

Add these module-level constants and the helper to `_qwen_chunker.py`, replacing the existing `MIN_MATCH_TOKENS = 2`:

```python
# Match acceptance rules. Two-or-more-token matches are always accepted.
# Single-token matches require the token to be substantive — long enough and
# not on the function-word stop list. This catches the common case of sparse
# sung audio where the overlap region contains only one content word, while
# rejecting accidental matches on Spanish/English function words.
MIN_MATCH_TOKENS_STRICT: int = 2
MIN_SINGLE_TOKEN_LENGTH: int = 4    # chars (after normalization)

# Function words to never accept as a single-token match. Lower-cased.
# Spanish + English coverage; both languages are realistic for this app.
_SINGLE_TOKEN_STOPLIST: frozenset[str] = frozenset({
    # Spanish articles, prepositions, common verbs
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "al", "en", "con", "por", "para", "sin",
    "que", "como", "cuando", "donde", "mi", "tu", "su",
    "es", "son", "era", "fue", "ser", "ir", "voy", "vas",
    "me", "te", "se", "le", "lo", "y", "o", "no", "si",
    # English equivalents
    "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "with", "and", "or", "but", "is", "are", "was", "were",
    "i", "you", "we", "they", "he", "she", "it", "my", "your",
    "this", "that", "these", "those",
    # Filler / interjection
    "oh", "ah", "uh", "mm", "yeah", "ay",
})


def _is_acceptable_match(
    length: int,
    tail: list[str],
    tail_start: int,
) -> bool:
    """Decide whether an LCS match is strong enough to stitch on.

    Rules:
      - Length >= MIN_MATCH_TOKENS_STRICT (2): always accept.
      - Length == 1: accept only if the matched token is substantive —
        at least MIN_SINGLE_TOKEN_LENGTH characters after normalization
        and not on the stop-word list.
      - Length == 0: reject.
    """
    if length >= MIN_MATCH_TOKENS_STRICT:
        return True
    if length == 1:
        matched = _norm_for_match(tail[tail_start])
        if len(matched) < MIN_SINGLE_TOKEN_LENGTH:
            return False
        if matched in _SINGLE_TOKEN_STOPLIST:
            return False
        return True
    return False
```

### 3.4 · Remove the old constant

Delete the line `MIN_MATCH_TOKENS: int = 2` from §3 of Addendum 3. It's superseded by `MIN_MATCH_TOKENS_STRICT` and the new helper.

---

## 4 · Testing

### 4.1 · New unit tests

Append these to `tests/test_transcribe.py` after the existing chunker tests:

```python
def test_qwen_stitcher_avoids_distant_repetition() -> None:
    """The matcher must only consider the immediately previous chunk.

    Simulates a song where chunk 1 contains a chorus, chunk 2 contains a
    verse, and chunk 3 contains the same chorus again. The chunk 2 → chunk 3
    stitch must use chunk 2's tail (not chunk 1's), even though chunk 1
    would match chunk 3 more completely.
    """
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    chunk_1 = "feliz cumpleaños mi princesa eres mi sol mi fortaleza"
    chunk_2 = "mi sol mi fortaleza con tu amor llenas mi vida cada sonrisa"
    chunk_3 = "cada sonrisa es la salida catrina mi corazón te pertenece"
    stitched = stitch_chunks([chunk_1, chunk_2, chunk_3])
    # "feliz cumpleaños" must appear exactly once — the second chorus is
    # not in any of these chunks.
    assert stitched.lower().count("feliz cumpleaños") == 1, stitched
    # "cada sonrisa" appears in both chunk 2 and chunk 3 and must be
    # deduplicated cleanly.
    assert stitched.lower().count("cada sonrisa") == 1, stitched
    # And the full transcript should read sensibly.
    assert "te pertenece" in stitched.lower()
    print("qwen_stitcher distant-repetition OK")


def test_qwen_stitcher_single_token_substantive() -> None:
    """A single substantive token in the overlap region should stitch."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    # Overlap: one long content word (catrina). Stop-list words and short
    # tokens are mixed in but shouldn't affect the acceptance decision.
    chunk_a = "una reina en su día catrina"
    chunk_b = "catrina mi corazón te pertenece"
    stitched = stitch_chunks([chunk_a, chunk_b])
    assert stitched.lower().count("catrina") == 1, stitched
    assert "te pertenece" in stitched.lower()
    print("qwen_stitcher single-token-substantive OK")


def test_qwen_stitcher_single_token_stopword_rejected() -> None:
    """A single stop-word match should fall back to concatenation."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    chunk_a = "uno dos tres cuatro de"
    chunk_b = "de cinco seis siete ocho"
    stitched = stitch_chunks([chunk_a, chunk_b])
    # "de" is a Spanish stop word — match should be refused, both
    # occurrences preserved.
    assert stitched.lower().count("de") == 2, stitched
    print("qwen_stitcher stopword-rejected OK")


def test_qwen_stitcher_single_token_short_rejected() -> None:
    """A single token shorter than MIN_SINGLE_TOKEN_LENGTH should be rejected."""
    from pipelines.transcribe_engines._qwen_chunker import stitch_chunks
    # "sol" is 3 chars, below the 4-char threshold.
    chunk_a = "eres mi gran sol"
    chunk_b = "sol que brilla mucho"
    stitched = stitch_chunks([chunk_a, chunk_b])
    assert stitched.lower().count("sol") == 2, stitched
    print("qwen_stitcher short-token-rejected OK")
```

Add to the `__main__` block at the bottom:

```python
    test_qwen_stitcher_avoids_distant_repetition()
    test_qwen_stitcher_single_token_substantive()
    test_qwen_stitcher_single_token_stopword_rejected()
    test_qwen_stitcher_single_token_short_rejected()
```

### 4.2 · Existing tests

Re-run all existing tests in `tests/test_transcribe.py`. They should all still pass — Change 1 alters internal bookkeeping but preserves the externally observable behavior of `test_qwen_stitcher_clean_overlap`, `test_qwen_stitcher_no_overlap_falls_back`, and `test_qwen_stitcher_single_chunk`.

### 4.3 · Manual checklist

- [ ] Re-run the Catrina stem through Qwen 4-bit. Specifically check the second-chorus area where the previous run produced three consecutive `\n` fallbacks ("...que tienes mi razón de ser oh / elis cum peyos mi princesa eres mi sol mi / La razón te pertenece..."). After Addendum 4, this region should read as a single continuous chorus.
- [ ] Confirm the log shows fewer "No alignment between chunk N and N+1" lines than the Addendum 3 baseline. A long song should now show mostly "Stitched chunk N → N+1: matched X tokens" entries.
- [ ] Test on a deliberately repetitive track if you have one — a song with multiple identical chorus repetitions is the case Change 1 was designed for.

---

## 5 · Out of scope (deferred)

The filler-token stripping idea from the comparison discussion (collapsing isolated `oh` / `ah` interjections adjacent to fallback boundaries) is not in this addendum. Reasoning:

- It's cosmetic, not corrective — it hides symptoms of failed stitches rather than fixing them.
- After Changes 1 and 2 land, the rate of failed stitches should drop enough that the remaining cosmetic artifacts are rare.
- If after manual testing the residual `oh` artifacts are still annoying, a 5-line post-processing pass over `stitch_chunks`'s output can be added in a future patch — no architectural changes needed.

Revisit only if the manual checklist's "fewer fallbacks" criterion is met but the transcripts still look messy.

---

## 6 · Definition of Done (addendum)

Append to §8 of the parent spec:

28. `stitch_chunks` matches against `prev_tokens` (immediately previous chunk only), not against the full accumulated output.
29. `_is_acceptable_match` helper, `MIN_MATCH_TOKENS_STRICT`, `MIN_SINGLE_TOKEN_LENGTH`, and `_SINGLE_TOKEN_STOPLIST` are present in `_qwen_chunker.py`.
30. Old `MIN_MATCH_TOKENS` constant is removed.
31. Four new unit tests in `tests/test_transcribe.py` pass.
32. All existing tests still pass.
33. Re-running the Catrina stem through Qwen 4-bit produces a continuous second chorus with no `\n` fallback markers in that region.
