# Lyrics Transcription Spec — Addendum 5

**Parent doc:** `LYRICS_TRANSCRIPTION_SPEC.md` (plus Addenda 1, 2, 3, 4)
**Status:** Ready for implementation
**Scope:** Frontend-only. Surface the already-plumbed `prompt` field as a user-visible text input in the MIDI tab's Lyrics control panel. Backend, pipeline, and engines are unchanged.

---

## 1 · Motivation

The parent spec plumbs an `initial_prompt` string from request → `TranscribeConfig` → both engines (Whisper passes it to `faster-whisper.transcribe(initial_prompt=...)`; Qwen appends it to the chat-template prompt). The plumbing works but the UI does not expose it, so users can never set it.

Proper-noun drift is the most visible class of transcription error on the test track:

- Qwen 4-bit renders "Catrina" as "katherine", "Katrin", and "patricia" on different chunks of the same song.
- Whisper Large v3 gets "Catrina" correct here but has historically mishead names on other material.

Both Whisper and Qwen bias toward vocabulary in the initial prompt. A single field where the user can paste the song title, performer name, and any known proper nouns will fix the dominant residual error class for both engines at once. Highest-leverage, lowest-cost UX change in the current backlog.

---

## 2 · Design

**One labelled text input** added to the Lyrics control panel, between the engine/language selectors and the Advanced disclosure. Not inside Advanced — this is meant to be used, not hidden.

### 2.1 · Label and placeholder

```
Hint (optional)
[ Song title, names, key phrases — e.g. "Catrina, Feliz cumpleaños"        ]
```

The placeholder text shows a concrete example so users understand both *what* to put there and *what format*. The label says "Hint (optional)" rather than "Initial prompt" or "Whisper prompt" because:

- It's optional and the UI should signal that.
- "Initial prompt" is jargon and is also conflated with LLM "system prompt" in many users' minds.
- "Hint" is honest about what the field does: it nudges the model, doesn't dictate output.

### 2.2 · Tooltip on the input (via `title=`)

> Words and phrases here will bias the transcription model toward similar vocabulary. Useful for proper nouns (names, places), song titles, recurring phrases the model gets wrong, and uncommon spellings. Works with both Whisper and Qwen engines. Keep it short — a single line of the most distinctive terms is enough.

### 2.3 · Length cap

Soft cap at **224 characters** on the input. This matches the practical limit faster-whisper imposes on `initial_prompt` (it's tokenized and prepended to each transcription window's context, and very long prompts can crowd out the actual transcription content). Qwen accepts longer prompts but doesn't benefit from them at this task.

Enforce by setting `maxlength="224"` on the `<input>`. No JavaScript validation needed — the HTML attribute does it.

### 2.4 · Persistence within session

The hint value persists in `appState.lyricsHint` for the lifetime of the page session, mirroring how other Lyrics-mode preferences are persisted per Addendum 2 §3.2. When the user reopens the MIDI tab or switches between Notes and Lyrics modes, the hint value is preserved. On `New Session` the hint is cleared.

The hint does NOT persist across page reloads. (Adding localStorage persistence is over-engineering for v1; this is a one-off-per-song field, not a setting.)

### 2.5 · Per-engine handling — no UI differentiation

Whisper consumes the hint as `initial_prompt`, biasing token probabilities. Qwen consumes it as text appended to the chat-template instruction. Both treatments are appropriate for their respective engines and the user does not need to know the difference. One field, one behavior model from the user's perspective.

### 2.6 · Empty state

When the hint is empty, the request payload omits the `prompt` field entirely (or sends `null`). The backend already handles `prompt=None` correctly per the parent spec — no change needed there.

---

## 3 · Implementation

### 3.1 · `frontend/components/midi.js` — add the input

Locate the Lyrics control panel construction (added in parent spec §5.2). The current order of controls is:

1. Stem selector
2. Engine dropdown (`<select>`)
3. Language hint dropdown (`<select>`)
4. Output formats (checkboxes)
5. Advanced disclosure (cross-window conditioning, collapse repetitions)
6. Transcribe button

Insert the new hint field as item 4 (between language and output formats):

```js
// Hint field — biases the transcription model toward specific vocabulary.
// Plumbed through TranscribeRequest.prompt for both engines.
const hintGroup = el('div', { className: 'form-group' },
  el('label', { htmlFor: 'midi-lyrics-hint' }, 'Hint (optional)'),
  el('input', {
    type: 'text',
    id: 'midi-lyrics-hint',
    className: 'midi-lyrics-hint-input',
    maxlength: '224',
    placeholder: 'Song title, names, key phrases — e.g. "Catrina, Feliz cumpleaños"',
    title: 'Words and phrases here will bias the transcription model toward '
         + 'similar vocabulary. Useful for proper nouns (names, places), '
         + 'song titles, recurring phrases the model gets wrong, and uncommon '
         + 'spellings. Works with both Whisper and Qwen engines. Keep it '
         + 'short — a single line of the most distinctive terms is enough.',
    onInput: (e) => { appState.lyricsHint = e.target.value; },
  }),
);
```

Append `hintGroup` to the Lyrics panel between the language group and the output-formats group.

### 3.2 · `frontend/components/midi.js` — wire into the transcribe call

In the existing `handleTranscribe` handler (or whatever the Transcribe button's click handler is called — name will be in the parent spec's implementation), add the hint to the request body. Find the existing `fetch('/api/transcribe', ...)` body construction and add:

```js
const hintValue = (_id('midi-lyrics-hint')?.value || '').trim();

const body = {
  audio_path: selectedStemPath,
  engine_id: selectedEngineId,
  model_id: selectedModelId,
  language: selectedLanguage || null,
  formats: selectedFormats,
  condition_on_previous_text: conditioningCheckbox.checked,
  collapse_repetitions: collapseCheckbox.checked,
};
if (hintValue) body.prompt = hintValue;  // only send when non-empty
```

The "only send when non-empty" pattern keeps the request payload clean and matches the Pydantic model's `prompt: str | None = None` shape.

### 3.3 · `frontend/components/midi.js` — restore on mode switch

If the parent spec's `switchMidiMode('lyrics')` function rebuilds the Lyrics panel rather than just toggling visibility, the hint input loses its value on every mode switch. Two options:

- **Preferred:** if the panel is built once and toggled (display: none / display: block), nothing extra is needed — the input keeps its value naturally.
- **Fallback:** if the panel is rebuilt, restore the value after construction:
  ```js
  if (appState.lyricsHint) {
    _id('midi-lyrics-hint').value = appState.lyricsHint;
  }
  ```

Check which pattern is in use and apply the matching fix.

### 3.4 · `frontend/app.js` — register `lyricsHint` in appState

Find the `appState` object definition. Add the new field alongside the other Lyrics-mode persistence fields added in Addendum 2:

```js
// In appState declaration:
lyricsEngineId: 'whisper',       // from Addendum 2
lyricsModelId: 'whisper-large-v3', // from Addendum 2
lyricsHint: '',                   // NEW — from Addendum 5
```

Also include it in whatever `clearSession()` / `New Session` logic resets these fields:

```js
appState.lyricsHint = '';
```

### 3.5 · `frontend/style.css` — minimal styling

Reuse the existing `.form-group` and the standard `<input type="text">` styling. No new CSS class is strictly required. If the input ends up visually different from other text inputs in the MIDI tab, add a single rule:

```css
.midi-lyrics-hint-input {
  width: 100%;
  font-family: var(--font-base);
  font-size: 13px;
}
```

Use existing CSS custom properties for color and border — no new tokens.

---

## 4 · Testing

### 4.1 · No new automated tests

This is a frontend wiring change. The backend `prompt` field is already covered by the existing pipeline; the input plumbing is too thin to warrant a dedicated test.

### 4.2 · Manual checklist

- [ ] Open MIDI tab → Lyrics mode → confirm the new "Hint (optional)" field is visible between Language and Output formats.
- [ ] Hover the field → tooltip appears with the documented text.
- [ ] Type 250 characters → input stops accepting at 224. (Browser-enforced via `maxlength`.)
- [ ] Type "Catrina, Feliz cumpleaños" → leave field → switch to Notes mode → switch back to Lyrics → field still shows the value.
- [ ] Hit Transcribe with the hint set → inspect the network request to `/api/transcribe` → confirm `prompt` field is present with the typed text.
- [ ] Clear the hint → hit Transcribe → confirm the request payload omits `prompt` (or sends `null`).
- [ ] On the Catrina test track, run **Qwen 4-bit** with hint = "Catrina, Feliz cumpleaños mi princesa". Confirm "katherine" / "patricia" no longer appear. The proper noun drift should be gone or substantially reduced.
- [ ] Run **Whisper Large v3** with the same hint. Output should be unchanged or marginally improved (Whisper got this song's names right without the hint, but the hint should not hurt).
- [ ] Click "New Session" → confirm hint field is cleared.

---

## 5 · Definition of Done (addendum)

Append to §8 of the parent spec:

34. MIDI Lyrics control panel contains a `Hint (optional)` text input between Language and Output formats, with the tooltip and placeholder text specified in §2.
35. `maxlength` of 224 characters enforced via HTML attribute.
36. Hint value persists in `appState.lyricsHint` across Notes/Lyrics mode switches.
37. Transcribe requests include the `prompt` field only when the hint is non-empty.
38. `New Session` clears `appState.lyricsHint`.
39. Manual hint test against the Catrina stem with Qwen 4-bit shows reduced proper-noun drift compared to the no-hint baseline from Addendum 4.
