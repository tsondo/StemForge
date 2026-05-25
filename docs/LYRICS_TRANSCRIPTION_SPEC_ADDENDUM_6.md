# Lyrics Transcription Spec — Addendum 6

**Parent doc:** `LYRICS_TRANSCRIPTION_SPEC.md` (plus Addenda 1, 2, 3, 4, 5)
**Status:** Ready for implementation
**Scope:** Qwen-only. Reshape how the user's hint string is injected into Qwen's chat-template prompt to prevent the model from interpreting the hint as a task switch. No changes to the Whisper engine, the pipeline, the API, or the frontend.

---

## 1 · Motivation

Addendum 5 added a UI hint field that flows through `TranscribeRequest.prompt` to both engines.

For **Whisper** this works correctly. The hint flows to `faster-whisper.transcribe(initial_prompt=...)`, which is a token-conditioning input — it biases the decoder's vocabulary without being read as instruction. Manual test on the Catrina stem with hint = "Feliz Cumpleanos Caterina" produced clean verbatim transcription with "Caterina" spelled consistently.

For **Qwen** the same hint broke the task. Qwen's only prompt channel is the chat-template instruction, and appending a Spanish-language hint to a task instruction caused the model to reinterpret the task. Manual test produced output like:

```
The lyrics translate to 'Happy Birthday Caterina,' with the message
wishing Caterina a happy birthday...
```

Every chunk was prefixed with `"The lyrics translate to '...'"` — Qwen had silently shifted from transcription to translation-and-summarization, presumably because a Spanish hint in an English-instruction context cues the model that translation is part of the requested task.

Root cause is architectural, not a bug: Whisper and Qwen don't have comparable "prompt" semantics, so the same string can't be passed to both raw.

## 2 · Design

Three principles:

1. **The user's mental model stays one field.** They type a hint; it works for whatever engine is selected. The asymmetry is hidden inside the engine implementations.
2. **Whisper code is untouched.** It already works correctly per Addendum 5's manual test.
3. **Qwen wraps the hint with explicit anti-drift anchoring.** The wrapper does two things: (a) labels the hint as vocabulary guidance rather than task context, and (b) reasserts the original task in the same sentence to prevent the model from drifting into translation or summarization.

### 2.1 · The wrapper text

When the hint is non-empty, Qwen's prompt becomes:

```
Transcribe the lyrics of this audio. Output only the lyrics text,
preserving line breaks where the singer pauses. Do not add commentary,
explanations, or section labels. Names and key terms that may appear in
the lyrics: <HINT>. Use these spellings exactly. Do not translate or
summarize — output only the verbatim transcribed lyrics.
```

(Plus the existing `" The lyrics are in {language}."` suffix when language is set.)

Three specific phrases carry weight in this wrapper:

- **"Names and key terms that may appear in the lyrics:"** frames the hint as vocabulary, not instruction. The model is told what role the trailing text plays.
- **"Use these spellings exactly."** instructs the model to preserve the user's preferred orthography for the listed terms (Caterina vs. Katerina, Catrina, etc.).
- **"Do not translate or summarize — output only the verbatim transcribed lyrics."** is the load-bearing anti-drift anchor. It re-asserts the original task immediately after the hint, in the same paragraph, making it harder for the model to follow the hint's language into translation mode.

When the hint is empty, the wrapper is not added — the prompt is exactly the parent spec's `_PROMPT` constant, optionally with the language suffix. This avoids changing behavior for the no-hint code path that is already tested and working.

### 2.2 · No UI changes

The frontend hint field, its label, its placeholder, and its tooltip are all unchanged. From the user's perspective they paste names into a box; the box just works, regardless of selected engine. The asymmetry is invisible.

### 2.3 · No request schema changes

`TranscribeRequest.prompt`, `TranscribeConfig.prompt`, and the engine `transcribe()` signatures all stay as-is. The change is contained inside Qwen's `_transcribe_chunk` method.

---

## 3 · Implementation

### 3.1 · `pipelines/transcribe_engines/qwen_engine.py` — rewrite prompt construction

Locate the existing `_PROMPT` constant at module level and the chunk-level prompt construction in `_transcribe_chunk`. Currently (per parent spec §3.4 and Addendum 3 §4):

```python
_PROMPT = (
    "Transcribe the lyrics of this audio. Output only the lyrics text, "
    "preserving line breaks where the singer pauses. Do not add commentary, "
    "explanations, or section labels."
)

# Inside _transcribe_chunk:
user_prompt = prompt or _PROMPT
if language:
    user_prompt += f" The lyrics are in {language}."
```

Replace with:

```python
_BASE_PROMPT = (
    "Transcribe the lyrics of this audio. Output only the lyrics text, "
    "preserving line breaks where the singer pauses. Do not add commentary, "
    "explanations, or section labels."
)

_HINT_WRAPPER = (
    " Names and key terms that may appear in the lyrics: {hint}. "
    "Use these spellings exactly. Do not translate or summarize — "
    "output only the verbatim transcribed lyrics."
)


def _build_qwen_prompt(hint: str | None, language: str | None) -> str:
    """Construct Qwen's chat-template instruction text.

    The hint, when present, is wrapped with explicit framing so the model
    treats it as vocabulary guidance rather than as additional task
    instruction. The wrapper re-asserts the original task immediately after
    the hint to prevent the model from drifting into translation or
    summarization when the hint contains text in a different language than
    the surrounding instruction.

    A trailing language suffix is appended when language is set, mirroring
    the existing behavior from the parent spec.
    """
    prompt = _BASE_PROMPT
    hint_clean = (hint or "").strip()
    if hint_clean:
        # Sanitize hint to a single line — newlines or stray quotes inside the
        # user's input could affect the model's parse of the wrapper.
        hint_clean = " ".join(hint_clean.split()).replace('"', "'")
        prompt += _HINT_WRAPPER.format(hint=hint_clean)
    if language:
        prompt += f" The lyrics are in {language}."
    return prompt
```

Then in `_transcribe_chunk`, replace the inline prompt assembly with a call:

```python
def _transcribe_chunk(
    self,
    audio: "np.ndarray",
    *,
    language: str | None,
    prompt: str | None,
) -> str:
    """Run a single Qwen forward pass on one audio chunk."""
    user_prompt = _build_qwen_prompt(hint=prompt, language=language)
    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "in_memory"},
            {"type": "text", "text": user_prompt},
        ]},
    ]
    # ... rest of method unchanged
```

### 3.2 · Note on the rename

The old `_PROMPT` becomes `_BASE_PROMPT`. If any other code in `qwen_engine.py` imports or references `_PROMPT` by name, update those references too. Grep should be conclusive — it's a leading-underscore module-private constant, so the only legitimate consumers should be inside this file.

### 3.3 · No changes elsewhere

Whisper's engine is left exactly as-is. `WhisperEngine.transcribe` still passes the raw `prompt` argument to `model.transcribe(initial_prompt=prompt, ...)`. Whisper's token-conditioning mechanism handles unwrapped hints correctly; wrapping it would only degrade quality.

The pipeline, the API router, the request schema, the session store, and the frontend are all unchanged.

---

## 4 · Testing

### 4.1 · New unit test

Add to `tests/test_transcribe.py`:

```python
def test_qwen_prompt_construction() -> None:
    """Verify the Qwen prompt assembler wraps hints correctly."""
    from pipelines.transcribe_engines.qwen_engine import (
        _build_qwen_prompt, _BASE_PROMPT,
    )

    # No hint, no language → bare base prompt.
    assert _build_qwen_prompt(None, None) == _BASE_PROMPT
    assert _build_qwen_prompt("", None) == _BASE_PROMPT
    assert _build_qwen_prompt("   ", None) == _BASE_PROMPT

    # Hint only → wrapper applied, anti-translation language present.
    out = _build_qwen_prompt("Catrina, Feliz cumpleaños", None)
    assert "Catrina, Feliz cumpleaños" in out
    assert "Use these spellings exactly" in out
    assert "Do not translate or summarize" in out
    assert out.startswith(_BASE_PROMPT)

    # Language only, no hint → language suffix present, no wrapper.
    out = _build_qwen_prompt(None, "Spanish")
    assert "lyrics are in Spanish" in out
    assert "Use these spellings exactly" not in out

    # Hint + language → both present, hint wrapper precedes language suffix.
    out = _build_qwen_prompt("Catrina", "Spanish")
    assert "Catrina" in out
    assert "Use these spellings exactly" in out
    assert "lyrics are in Spanish" in out
    # Wrapper must come before the language suffix so the language hint
    # doesn't get absorbed into the anti-translation clause.
    assert out.index("Use these spellings exactly") < out.index("lyrics are in Spanish")

    # Multi-line / quote-laden hint should be sanitized to one line.
    out = _build_qwen_prompt('Line one\nLine "two"\nLine three', None)
    assert "\n" not in out
    assert '"' not in out
    assert "Line one Line 'two' Line three" in out

    print("qwen_prompt_construction OK")
```

Add to the `__main__` block:

```python
    test_qwen_prompt_construction()
```

### 4.2 · Manual checklist

- [ ] Re-run the Catrina stem through **Qwen 4-bit** with hint = `"Feliz Cumpleanos Caterina"`. The output must NOT contain `"The lyrics translate to"` or any equivalent translation/summary framing. Output must be verbatim Spanish lyrics.
- [ ] With the same hint, confirm Qwen now spells the name as `"Caterina"` (matching the hint) rather than `"katherine"` / `"patricia"` from Addendum 4's run.
- [ ] Re-run the same stem with **Whisper Large v3** and the same hint. Output should be byte-for-byte identical to the Addendum 5 manual-test result (Whisper is untouched by this addendum).
- [ ] Run Qwen 4-bit with hint left **empty**. Output should be identical to the pre-Addendum-5 / Addendum-4 baseline — the no-hint code path is unchanged.
- [ ] Try a deliberately weird hint: `"banana"` (no relation to song content). Qwen output should still be the actual lyrics, not anything containing "banana." The model should ignore vocabulary that doesn't acoustically match.
- [ ] Try a hint containing translation-mode trigger words like `"translation"` or `"meaning"`. Output should still be verbatim transcription thanks to the anti-translation anchor in the wrapper. (If this test fails, see §5.)

---

## 5 · Known residual risk

The wrapper relies on the model honoring the "Do not translate or summarize" anchor when the hint itself contains text suggestive of translation. If real-world testing surfaces hints that defeat the anchor — e.g. a user types `"Please translate"` as their hint — the model might still drift.

If that becomes a problem, the next escalation (out of scope for this addendum, noted here for future reference) is to inject the hint via a **fake assistant turn**:

```python
conversation = [
    {"role": "user", "content": [{"type": "text", "text":
        "I'll send an audio clip. The lyrics may contain these names: <HINT>. "
        "Use those exact spellings. Transcribe only — no translation, no summary."
    }]},
    {"role": "assistant", "content": [{"type": "text", "text":
        "Understood. I will transcribe the lyrics verbatim using those spellings."
    }]},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "in_memory"},
        {"type": "text", "text": _BASE_PROMPT},
    ]},
]
```

This pattern is more robust because the model commits to the format in its own (fake) voice before seeing the audio, making it harder to drift mid-task. It's also more expensive (more tokens per call) and harder to reason about. Reserve for when the simpler wrapper proves insufficient.

---

## 6 · Definition of Done (addendum)

Append to §8 of the parent spec:

40. `qwen_engine.py` exposes a `_build_qwen_prompt(hint, language) -> str` helper that wraps non-empty hints with explicit vocabulary-guidance framing and anti-translation anchoring.
41. `_BASE_PROMPT` and `_HINT_WRAPPER` module constants present; old `_PROMPT` reference removed.
42. `_transcribe_chunk` calls `_build_qwen_prompt` instead of inline assembly.
43. `test_qwen_prompt_construction` unit test passes.
44. All existing tests in `tests/test_transcribe.py` continue to pass.
45. Manual hint test on Catrina stem with Qwen 4-bit shows verbatim Spanish lyrics (no "The lyrics translate to..." prefix) and uses the hint's spelling for proper nouns.
46. Whisper engine code is byte-for-byte unchanged.
