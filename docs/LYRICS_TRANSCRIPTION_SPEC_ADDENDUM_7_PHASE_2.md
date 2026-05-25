# Lyrics Transcription Spec — Addendum 7, Phase 2

**Parent doc:** `LYRICS_TRANSCRIPTION_SPEC.md` (plus Addenda 1-6 and Phase 1 of Addendum 7)
**Status:** Ready for implementation
**Scope:** Qwen-only. Rewrite `_build_qwen_prompt` into `_build_qwen_conversation`, returning a 3-turn chat structure that isolates the hint from the audio-bearing transcription request. Update `_transcribe_chunk` to consume the new structure. Update the unit test. No changes to Whisper, the stitcher, the pipeline, the API, or the frontend.

---

## 1 · Motivation

Phase 1 diagnostic confirmed Hypothesis 2: when the hint and the transcription instruction are concatenated into a single user turn, Qwen treats the hint as foreground task content. Seven of ten chunks emitted only the hint phrase, producing nothing for those audio segments. The stitcher correctly deduplicated identical near-empty chunks; the data loss happened at the model, not in stitching.

Architectural rephrasing: **the hint must be background context the model has already processed, not part of the current turn's instructions.** The fake-assistant-turn pattern accomplishes this by putting the hint in turn 1, a fabricated assistant acknowledgment in turn 2, and the audio-bearing transcription request alone in turn 3. The model's attention on the audio turn is no longer competing with hint-as-directive text.

This is the escalation path documented in Addendum 6 §5, now needed in practice.

---

## 2 · Design

### 2.1 · Three-turn conversation structure

```
Turn 1 — user, text-only:
  Setup. Describes the upcoming transcription task and supplies the hint
  if present. Frames hint material as background spelling guidance, not
  as content to emit.

Turn 2 — assistant, text-only, fabricated:
  Acknowledgment. The model "agrees" to the format in its own voice,
  committing to verbatim transcription with the supplied spellings.

Turn 3 — user, audio + minimal text:
  The transcription request. Audio is here. Text instruction is short
  and focused — "Transcribe the lyrics." — so the audio dominates
  attention on this turn.
```

### 2.2 · Structural symmetry between hint and no-hint cases

Both branches use the same three-turn shape. With an empty hint, turn 1 omits the hint guidance and turn 2 stays the same. This keeps a single code path and single failure surface — important because two of the failure modes we've already debugged (translation drift in Addendum 6, hint-priming in Phase 1) were specifically caused by prompt structures that varied based on hint presence.

### 2.3 · Turn-by-turn text

**Turn 1, no hint:**

```
I'll send you an audio clip in a moment. Your job is to transcribe the
sung lyrics verbatim. Do not translate, summarize, or comment — output
only the lyrics text, preserving line breaks where the singer pauses.
```

**Turn 1, with hint:**

```
I'll send you an audio clip in a moment. Your job is to transcribe the
sung lyrics verbatim. Do not translate, summarize, or comment — output
only the lyrics text, preserving line breaks where the singer pauses.

The lyrics may contain these specific names and spellings: <HINT>.
When you hear those words sung, use those exact spellings. Otherwise
ignore them — do not emit these names unless you actually hear them in
the audio.
```

The "Otherwise ignore them — do not emit these names unless you actually hear them in the audio" sentence is the load-bearing addition compared to Addendum 6. It explicitly addresses the failure mode Phase 1 surfaced: the model emitting hint text in chunks where it doesn't acoustically appear.

**Turn 2, both cases:**

```
Understood. I will transcribe the lyrics of the audio verbatim,
preserving line breaks. I will only emit text I actually hear sung.
```

The second sentence pre-commits the model (in its own voice) to the constraint that prevented it from honoring instructions in Phase 1. This is the structural advantage of the fake-turn pattern — the model is more likely to honor commitments framed as its own prior statements than commitments framed as user instructions.

**Turn 3, with language hint:**

```
Transcribe the lyrics. The lyrics are in <LANGUAGE>.
```

**Turn 3, without language hint:**

```
Transcribe the lyrics.
```

Audio is attached to turn 3 alongside this text. Turn 3 contains no hint material — the audio is the only ambiguous signal in this turn, and the minimal text instruction reinforces "transcribe what you hear, nothing else."

### 2.4 · Hint sanitization

Same as Addendum 6 §3.1 — strip newlines, downgrade double-quotes. The hint enters turn 1 inline so character escaping still matters.

---

## 3 · Implementation

### 3.1 · `pipelines/transcribe_engines/qwen_engine.py` — rewrite prompt construction

Replace the Addendum 6 module-level constants and helper with the following. Note the function rename (`_build_qwen_prompt` → `_build_qwen_conversation`) because the return type changes from `str` to `list[dict]`.

```python
"""Qwen2-Audio engine for high-fidelity sung-lyrics transcription.

LICENSE: Apache 2.0 — see licenses/LICENSE-Qwen2-Audio and ACKNOWLEDGMENTS.md.
"""

# ... existing imports ...

# Base task description used in turn 1. Stable across hint/no-hint cases.
_TURN1_BASE = (
    "I'll send you an audio clip in a moment. Your job is to transcribe "
    "the sung lyrics verbatim. Do not translate, summarize, or comment — "
    "output only the lyrics text, preserving line breaks where the singer "
    "pauses."
)

# Hint guidance, appended to turn 1 only when a hint is present.
# The final sentence is critical: it tells the model not to emit hint text
# unless it actually appears in the audio. Without this, the model treats
# the hint as content it must produce in every chunk.
_TURN1_HINT = (
    "\n\nThe lyrics may contain these specific names and spellings: {hint}. "
    "When you hear those words sung, use those exact spellings. Otherwise "
    "ignore them — do not emit these names unless you actually hear them "
    "in the audio."
)

# Fabricated assistant turn. Same for both hint and no-hint cases — the
# constraint about only emitting heard text applies regardless.
_TURN2_ACKNOWLEDGMENT = (
    "Understood. I will transcribe the lyrics of the audio verbatim, "
    "preserving line breaks. I will only emit text I actually hear sung."
)

# Turn 3 base instruction. Minimal by design — the audio attached to this
# turn is the primary signal; the text is only here so the chat template
# is well-formed.
_TURN3_BASE = "Transcribe the lyrics."


def _sanitize_hint(hint: str | None) -> str:
    """Normalize a user-supplied hint to a single line with no double-quotes.

    Newlines and stray quotes inside the hint could disrupt the chat
    template's parse of the surrounding instruction text.
    """
    if not hint:
        return ""
    cleaned = " ".join(hint.split()).replace('"', "'")
    return cleaned


def _build_qwen_conversation(
    hint: str | None,
    language: str | None,
) -> list[dict]:
    """Construct Qwen's chat-template conversation for one chunk.

    Returns a three-turn structure that isolates hint context (turn 1) and
    fabricated assistant acknowledgment (turn 2) from the audio-bearing
    transcription request (turn 3). The audio placeholder is included in
    turn 3 only — the actual audio array is supplied separately to the
    processor at encoding time.

    The structural separation is the load-bearing design choice. Inline
    concatenation of hint into the same turn as the transcription request
    caused the model to treat hint text as content to emit (see Phase 1
    diagnostic for evidence). The fake-assistant-turn pattern is documented
    in Addendum 6 §5 as the escalation path for that failure mode and is
    implemented here.
    """
    hint_clean = _sanitize_hint(hint)

    # Turn 1: setup, optionally with hint guidance.
    turn1_text = _TURN1_BASE
    if hint_clean:
        turn1_text += _TURN1_HINT.format(hint=hint_clean)

    # Turn 3: minimal transcription request, optionally with language hint.
    turn3_text = _TURN3_BASE
    if language:
        turn3_text += f" The lyrics are in {language}."

    return [
        {"role": "user", "content": [
            {"type": "text", "text": turn1_text},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": _TURN2_ACKNOWLEDGMENT},
        ]},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "in_memory"},
            {"type": "text", "text": turn3_text},
        ]},
    ]
```

### 3.2 · `pipelines/transcribe_engines/qwen_engine.py` — update `_transcribe_chunk`

The existing method (per Addendum 6 §3.1) calls `_build_qwen_prompt` and constructs a single-turn conversation inline. Rewrite to consume the new function's structured output:

```python
def _transcribe_chunk(
    self,
    audio: "np.ndarray",
    *,
    language: str | None,
    prompt: str | None,
) -> str:
    """Run a single Qwen forward pass on one audio chunk."""
    conversation = _build_qwen_conversation(hint=prompt, language=language)
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

The processor call is unchanged — it still receives `text=<rendered conversation>` and `audios=[audio]`. The chat template rendering inside `apply_chat_template` handles the multi-turn structure; the audio placeholder in turn 3 is resolved by `audios=[audio]` at the same position it occupied in the single-turn version.

### 3.3 · Removal of Addendum 6 artifacts

Delete the following symbols from `qwen_engine.py`:

- `_BASE_PROMPT` (replaced by `_TURN1_BASE` plus `_TURN3_BASE`)
- `_HINT_WRAPPER` (replaced by `_TURN1_HINT`)
- `_build_qwen_prompt` (replaced by `_build_qwen_conversation`)

If any code outside `qwen_engine.py` imports these symbols, that's a bug to surface. They are leading-underscore module-private, so a grep should return only references inside this file plus the unit test in `tests/test_transcribe.py`.

### 3.4 · Diagnostic logging

Keep the INFO-level chunk preview logging added in Phase 1. The same diagnostic will be needed to verify the fix works — the success criterion is that per-chunk char counts become substantial and uniform, not bimodal (26 vs. 196 chars). Once the fix is validated, both Phase 1's INFO logs can be demoted to DEBUG in a follow-up housekeeping pass; for now leave them alone so the post-fix verification run is directly comparable to the pre-fix baseline.

---

## 4 · Testing

### 4.1 · Replace the Addendum 6 unit test

Remove `test_qwen_prompt_construction` from `tests/test_transcribe.py` — it tests `_build_qwen_prompt` which no longer exists.

Add `test_qwen_conversation_construction`:

```python
def test_qwen_conversation_construction() -> None:
    """Verify the Qwen conversation builder produces the expected 3-turn structure."""
    from pipelines.transcribe_engines.qwen_engine import (
        _build_qwen_conversation,
        _TURN1_BASE, _TURN2_ACKNOWLEDGMENT, _TURN3_BASE,
    )

    # No hint, no language → 3 turns, hint guidance absent, language suffix absent.
    conv = _build_qwen_conversation(None, None)
    assert len(conv) == 3
    assert conv[0]["role"] == "user"
    assert conv[1]["role"] == "assistant"
    assert conv[2]["role"] == "user"

    # Turn 1 contains base task only.
    turn1_text = conv[0]["content"][0]["text"]
    assert turn1_text == _TURN1_BASE
    assert "names and spellings" not in turn1_text

    # Turn 2 is the fabricated acknowledgment.
    assert conv[1]["content"][0]["text"] == _TURN2_ACKNOWLEDGMENT

    # Turn 3 has audio placeholder + minimal text.
    assert conv[2]["content"][0]["type"] == "audio"
    assert conv[2]["content"][1]["text"] == _TURN3_BASE

    # Empty-string and whitespace hints behave like None.
    assert _build_qwen_conversation("", None) == conv
    assert _build_qwen_conversation("   ", None) == conv

    # Hint only → hint guidance in turn 1, turn 2/3 unchanged.
    conv = _build_qwen_conversation("Catrina, Feliz cumpleaños", None)
    turn1_text = conv[0]["content"][0]["text"]
    assert turn1_text.startswith(_TURN1_BASE)
    assert "Catrina, Feliz cumpleaños" in turn1_text
    assert "do not emit these names unless you actually hear them" in turn1_text
    # Turn 3 must NOT contain the hint — that was the Phase 1 failure mode.
    assert "Catrina" not in conv[2]["content"][1]["text"]
    # Turn 2 must not vary based on hint presence.
    assert conv[1]["content"][0]["text"] == _TURN2_ACKNOWLEDGMENT

    # Language only → turn 3 has language suffix, turn 1 unchanged.
    conv = _build_qwen_conversation(None, "Spanish")
    assert conv[0]["content"][0]["text"] == _TURN1_BASE
    turn3_text = conv[2]["content"][1]["text"]
    assert "in Spanish" in turn3_text

    # Hint + language → hint in turn 1, language in turn 3, no crosstalk.
    conv = _build_qwen_conversation("Catrina", "Spanish")
    turn1_text = conv[0]["content"][0]["text"]
    turn3_text = conv[2]["content"][1]["text"]
    assert "Catrina" in turn1_text
    assert "Catrina" not in turn3_text
    assert "Spanish" in turn3_text
    assert "Spanish" not in turn1_text

    # Multi-line / quote-laden hint should be sanitized.
    conv = _build_qwen_conversation('Line one\nLine "two"\nLine three', None)
    turn1_text = conv[0]["content"][0]["text"]
    assert "\n\nThe lyrics may contain" in turn1_text  # the join we added
    # The hint itself, post-sanitization:
    assert "Line one Line 'two' Line three" in turn1_text

    print("qwen_conversation_construction OK")
```

Update the `__main__` block: replace `test_qwen_prompt_construction()` with `test_qwen_conversation_construction()`.

### 4.2 · Validation run

Same procedure as Phase 1 part B/C/D. Claude Code applies the changes, runs the same Catrina stem with the same hint, parses the resulting log, and reports per-chunk char counts plus stitch ratios.

**Success criteria for Phase 2:**

- Per-chunk char counts substantial (median > 150) and roughly uniform (max/min ratio < 3.0).
- Hint-only chunks (26-char outputs containing only the hint phrase) are absent or reduced to <2 of 10.
- Final stitched output length is in the range of 800-1500 chars (the Whisper baseline produced ~1100; Qwen should be in the same order of magnitude).
- Hint spelling appears in output where proper nouns are actually sung (e.g. "Caterina" in chorus lines that contain her name), not in chunks where the hint phrase isn't being sung.

**Inconclusive or failure indicators:**

- Bimodal char count distribution similar to Phase 1.
- Hint phrase still appears in chunks where no corresponding audio exists.
- Final stitched output substantially shorter than Whisper's.

If the fix fails, Addendum 8 (deterministic spelling substitution post-hoc rather than via prompt) is the documented next escalation. Phase 1's logging infrastructure remains in place to drive that diagnosis if needed.

---

## 5 · Process notes

The user has flagged this entire transcription work as breaking ground where the value is in the process, not just the outcome. Three patterns worth recording for future work:

1. **Diagnosis before fix paid off here.** Without Phase 1's automated diagnostic, the natural intuition (strip hint echoes from chunk output) would have addressed a symptom rather than the cause. The data revealed that 70% of chunks produced nothing but the hint — a model-attention problem, not an output-formatting problem.

2. **Architectural separation beats instructional steering.** Addendum 6 tried to fix instruction-level behavior with stronger instructions ("Do not translate or summarize"). That worked for translation drift but introduced the hint-priming problem. Phase 2 fixes it structurally by relocating the hint to a different conversation turn entirely. When instruction-following breaks down, structural separation is the next tool, not louder instructions.

3. **Symmetric code paths matter.** Multiple prior failures (Addendum 6's translation drift, Phase 1's hint priming) traced back to prompt structures that varied based on whether the hint was empty. Phase 2 uses the three-turn shape unconditionally; the only thing that varies is the content of turn 1. One code path, one failure surface, no asymmetric edge cases to debug.

---

## 6 · Definition of Done

Append to §8 of the parent spec:

51. `qwen_engine.py` exposes `_build_qwen_conversation(hint, language) -> list[dict]` returning a 3-turn user/assistant/user conversation structure.
52. Audio placeholder appears in turn 3 only; hint text appears in turn 1 only; language hint appears in turn 3 only.
53. `_TURN1_BASE`, `_TURN1_HINT`, `_TURN2_ACKNOWLEDGMENT`, `_TURN3_BASE` module constants are present.
54. Addendum 6 artifacts (`_BASE_PROMPT`, `_HINT_WRAPPER`, `_build_qwen_prompt`) are removed.
55. `_transcribe_chunk` consumes the new conversation list directly.
56. `test_qwen_conversation_construction` passes; old `test_qwen_prompt_construction` removed.
57. All other existing tests continue to pass.
58. Whisper engine, stitcher, pipeline, API, frontend all unchanged.
59. Re-running the Catrina diagnostic from Phase 1 shows per-chunk char counts substantially improved over the bimodal 26/196-char Phase 1 baseline.
