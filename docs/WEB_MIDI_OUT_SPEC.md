# Web MIDI Output — Technical Specification (rev. 2)

**Feature:** Play extracted MIDI stems directly to external hardware synthesizers
**Location:** MIDI tab — per-stem card controls + a global MIDI Out panel in the left column
**Dependencies:** None. Web MIDI API is a W3C browser API; no Python or JS packages added.

> **Revision 2.** Supersedes the initial draft. Substantive changes:
> two-phase Stop (the queued-window hole), Web Worker clock (background-tab
> throttling), disconnect-scoped port-change handling, explicit multi-track
> merge rules, overlap merging in the backend flattener, velocity clamp scoped
> to note-ons, truncation-by-notes invariant, Program Change suppression keyed
> off drum-stem status, and reuse of the existing `_resolve_midi` helper.
> Each change is marked **[rev2]** where it matters.

---

## Overview

StemForge currently treats MIDI as a file format and a FluidSynth preview source.
This spec adds a third destination: the user's physical MIDI ports, so extracted
stems can be auditioned on hardware synths (the reference setup is a Korg modwave
and a Korg wavestate on separate MIDI ins).

The MIDI port lives on the **browser's** machine, not the server's. This is the
deciding architectural fact: a server-side `python-rtmidi` port is meaningless
when StemForge is deployed to HF Spaces or accessed from another machine on the
LAN. Web MIDI puts the port where the hardware actually is.

The backend keeps ownership of the MIDI data — it flattens the session
`PrettyMIDI` to a JSON event list. The frontend does no MIDI parsing, needs no
parsing library, and only schedules and sends. Thick backend / thin frontend is
preserved.

**Scope is output only.** MIDI input (recording from hardware into StemForge) is
explicitly out of scope and deferred — see *Deferred* at the end.

---

## User flow

```
Separate → MIDI → [Extract] → per-stem cards appear
                                  │
                                  ├─ Instrument: [Electric Bass ▼]   ← existing, FluidSynth only
                                  ├─ [▶ Play] [■ Stop]               ← existing, FluidSynth preview
                                  │
                                  └─ MIDI Out: [Port ▼] [Ch 1 ▼] [▶ Send] [■ Stop]   ← NEW
```

Left column gains a **MIDI Out** section: an availability banner, a
*Request access* button (see below), and a **Panic** button that silences every
known port regardless of which card is playing.

Typical session: Julio extracts bass and "other" from a track, sets bass →
wavestate ch 1, other → modwave ch 1, and hits Send on each. Hardware plays;
StemForge's own FluidSynth preview stays available and independent.

Imported MIDI (`POST /api/midi/import`) lands in `stem_midi_data` and gets a
card like any extracted stem, so it gets MIDI Out for free — no extra work, but
the test plan exercises it once.

---

## Backend

### New endpoint: `POST /api/midi/events`

Added to `backend/api/midi.py`, alongside `/render` and `/save`. Follows the
existing `RenderRequest` pattern.

```python
class EventsRequest(BaseModel):
    stem_label: str


@router.post("/events")
def get_midi_events(
    req: EventsRequest,
    session: SessionStore = Depends(get_user_session),
) -> dict:
    """Flatten a session PrettyMIDI to a JSON note-event list for Web MIDI.

    Returns one entry in ``tracks`` per PrettyMIDI instrument.  Events
    within a track are sorted ascending by time, note-off before note-on
    at equal timestamps.
    """
```

**[rev2] Resolution:** use the **existing** `_resolve_midi(stem_label, session)`
helper already defined in `backend/api/midi.py` for the music21 endpoints. It
already implements the `merged` / stem-label convention and the 404s. Do not
add a second copy.

**Response shape:**

```json
{
  "label": "bass",
  "duration": 187.34,
  "total_events": 1842,
  "truncated": false,
  "tracks": [
    {
      "name": "bass",
      "is_drum": false,
      "program": 33,
      "events": [
        {"t": 0.512, "type": "on",  "note": 45, "vel": 96},
        {"t": 0.998, "type": "off", "note": 45, "vel": 0},
        {"t": 1.004, "type": "on",  "note": 47, "vel": 88}
      ]
    }
  ]
}
```

`total_events` counts events across **all** tracks. `duration` is
`PrettyMIDI.get_end_time()`.

**Flattening rules (per track):**

1. Skip notes where `end <= start`.
2. **[rev2] Merge overlapping same-pitch notes.** Within one track, if a note
   of pitch P starts before an earlier note of pitch P has ended, hardware
   cannot represent the overlap: the first note-off would silence the second
   note, and the frontend's sounding-set would corrupt. Resolve at the source:
   truncate the earlier note's `end` to the later note's `start` (producing
   abutting notes, which rule 4 already handles). This is a few lines in the
   flattener and is covered by pytest — do not push this to the frontend.
3. Each surviving `pretty_midi.Note` becomes an `on` event at `note.start`
   (velocity clamped to **1–127** — velocity 0 would be interpreted as
   note-off by the receiver, which is why the floor is 1) and an `off` event
   at `note.end` with `vel: 0`. **[rev2]** The clamp applies to note-ons
   only; `off` events always carry velocity 0. Pitch is clamped to 0–127 on
   both.
4. Sort each track's events by `t`, with `off` sorting **before** `on` at
   identical timestamps. Abutting same-pitch notes (common after music21
   Clean Up quantization, and produced by rule 2) would otherwise leave a
   hanging note.
5. `program` and `is_drum` are reported for display only. **The frontend does
   not act on them by default** — see *Program Change* below.
6. **[rev2] Truncation invariant: truncate by notes, never by raw events.**
   Cap `total_events` at 20000 per response. If exceeded, drop whole notes
   from the end (latest `start` first) until under the cap — **never emit an
   `on` whose `off` was dropped** — set `"truncated": true`, and let the
   frontend surface a warning. A stem this dense is almost certainly a
   BasicPitch artefact and should be run through Clean Up first.

**Threading:** synchronous. This is a pure in-memory transform over data
already in the session — microseconds, no GPU, no job manager.

**Session mutation:** none. This endpoint is strictly read-only. It does not
touch `stem_midi_data`, unlike `/clean` and `/transpose`.

**Snapshot semantics.** The response is a snapshot: if the user runs Clean Up
or Transpose while a card is playing, the hardware keeps playing the pre-edit
events until the next Send. This is intentional — do not add invalidation
plumbing.

### No changes to `/api/capabilities`

Web MIDI availability is a property of the browser, not the server. The server
cannot report it and must not try. Detection is entirely frontend-side.

---

## Frontend

### New module: `frontend/components/webmidi.js`

A self-contained ES module in the existing `components/` directory (the
frontend is already ESM — `app.js` loads with `type="module"`). No CDN script
tag, no `index.html` change.

```js
/**
 * Web MIDI output — port discovery and lookahead event scheduling.
 *
 * Wraps navigator.requestMIDIAccess and provides a scheduler that streams
 * note events to a hardware port with sub-millisecond timing accuracy.
 *
 * Timing model: a Worker-driven tick (100 ms) scans a lookahead window
 * (250 ms) and hands each event to output.send() with an absolute
 * DOMHighResTimeStamp. The browser's MIDI subsystem does the precise
 * dispatch; the tick only needs to be roughly on time. This is the
 * standard "two clocks" pattern.
 */

export function isSupported()        // navigator.requestMIDIAccess exists
export async function initWebMidi()  // request access, wire statechange
export function getOutputs()         // [{id, name, manufacturer}]
export function getOutputById(id)
export function onPortsChanged(cb)   // subscribe to hot-plug events
export function createScheduler(outputId, channel)
export function panicAll()           // silence every known output
```

**`createScheduler(outputId, channel)` returns:**

```js
{
  play(events),   // begin streaming a flat, sorted event array from t=0
  stop(),         // two-phase halt — see Stop below
  isPlaying(),
  onFinish(cb),
  onTick(cb),     // fires with current playback seconds
}
```

**[rev2]** `channel` is fixed at creation. Changing the card's channel (or
port) dropdown during playback has no effect until the next Send — the UI does
not need to react, and the spec says so to keep the implementer from inventing
live-rebind behaviour.

**[rev2]** The initial draft's `startSec` parameter is dropped. Nothing in the
UI seeks; a start-from-playhead feature can reintroduce it alongside the rule
for notes already sounding at the seek point.

### [rev2] Track merging — who flattens `tracks[]`

`/api/midi/events` returns one entry per PrettyMIDI instrument so a future
multi-port feature can route them independently, but `play(events)` takes one
flat array and each card has exactly one port/channel pair. The rule:

- **`midi.js` merges.** Before calling `play()`, concatenate every track's
  events into one array and **re-sort with the same comparator the backend
  uses** (ascending `t`, `off` before `on` at equal `t`). The backend's
  per-track ordering does not survive concatenation; the tie-break must be
  re-established across tracks or the abutting-note guarantee is lost.
- Per-stem BasicPitch output is single-instrument, so this is usually a no-op
  copy; `merged` and multi-instrument imports are where it matters.
- Whether a **merged** card appears in the MIDI Out UI at all: yes, iff
  `session.merged_midi_data` exists — same visibility rule the Save controls
  use. All of its instruments play on the card's single port/channel;
  per-instrument routing is deferred (see *Deferred*).

### Scheduling design

This is the part that determines whether it feels tight or sloppy, so it is
specified precisely.

```
LOOKAHEAD_MS = 250    // how far ahead of the cursor we schedule
TICK_MS      = 100    // how often we scan
LEAD_MS      = 120    // startup delay before t=0, absorbs first-tick jitter
FLUSH_MS     = LOOKAHEAD_MS + 50   // [rev2] second-phase stop margin
```

**[rev2] The clock runs in a Web Worker.** Chrome throttles main-thread timers
in hidden tabs to ≥1 s (once per minute under intensive throttling). A user
auditioning hardware synths is *expected* to switch away from the tab — their
hands are on the modwave — so main-thread `setInterval` is disqualified, and
the 150 ms of slack in the original design is three orders of magnitude short.
Worker timers are not throttled this way. Mechanics:

- The worker is created from an inline `Blob` URL (`new Worker(URL.createObjectURL(...))`)
  containing only `setInterval(() => postMessage(0), TICK_MS)` — no new file to
  serve, no path coupling to `StaticFiles`.
- One shared worker instance for the module; each active scheduler subscribes
  to its message. `onmessage` runs `tick()` on the main thread, where
  `performance.now()`, the event cursor, and `output.send()` all live. The
  worker carries no state and no times — it is purely a metronome, so the
  worker/main-thread `timeOrigin` difference is irrelevant.
- Do **not** widen the lookahead on `visibilitychange` instead: a wider queued
  window directly worsens Stop latency (see below). The worker keeps both
  properties.

On `play(events)`:
1. Record `originMs = performance.now() + LEAD_MS`.
2. Reset the event cursor to index 0.
3. Subscribe to the worker tick.

Each `tick()`:
1. Compute `windowEndMs = performance.now() + LOOKAHEAD_MS`.
2. Advance the cursor, sending every event whose absolute time
   `originMs + e.t * 1000` falls before `windowEndMs`:
   ```js
   const status = (e.type === 'on' ? 0x90 : 0x80) | (channel - 1);
   output.send([status, e.note, e.vel], absMs);
   ```
   **[rev2]** Wrap sends in try/catch — `send()` throws `InvalidStateError` on
   a disconnected port; treat a throw as a disconnect (stop this scheduler,
   surface the error on the card's status field).
3. Track sounding pitches in a `Set` — add on `on`, remove on `off`. Because
   this is updated at *send* time, the set covers notes that are queued in the
   browser's MIDI subsystem but not yet audible — the two-phase Stop depends
   on this property.
4. If a tick arrives late (starved timer), send overdue events immediately —
   late is recoverable, silent is not. The worker clock makes multi-second
   starvation rare rather than routine.
5. When the cursor passes the last event, wait out the tail, then fire
   `onFinish` and unsubscribe.

**Why windowed rather than bulk-scheduling the whole file.** Sending 10000
timestamped messages up front works in Chrome, but cancelling them depends on
`MIDIOutput.clear()`, whose availability is inconsistent across
implementations. The windowed scheduler bounds the queue to ~250 ms, so Stop
can outwait it deterministically without `clear()`. **Do not use
`output.clear()`** — the design deliberately does not depend on it.

### [rev2] Stop — two phases, because the queue cannot be recalled

Silence is a correctness requirement, not a nicety. A stuck note on a
wavestate is the worst possible first impression of this feature.

Events already handed to `output.send()` with future timestamps **cannot be
recalled** without `clear()` — that is the reason `clear()` exists. So a Stop
that only sends immediate note-offs loses the race: up to `LOOKAHEAD_MS` of
already-queued note-**ons** fire *after* the panic messages and hang. The
initial draft had exactly this bug. `stop()` therefore runs everything twice:

**Phase 1 — immediate (timestamp 0):**
1. Unsubscribe from the tick; no further events are queued.
2. Explicit `note off` for every pitch in the sounding set (which, per tick
   step 3, includes queued-but-not-yet-sounding notes).
3. CC 120 (All Sound Off), CC 123 (All Notes Off), CC 64 = 0 (sustain off) on
   the scheduler's channel.

**Phase 2 — timestamped at `performance.now() + FLUSH_MS`,** i.e. after every
already-queued event has fired:
4. The same per-pitch note-offs and the same three CCs again.

Phase 1 exists so audible notes cut off instantly; phase 2 exists so queued
note-ons that fire in between are silenced within ~300 ms. The per-pitch
offs exist because not all hardware honours CC 123; the CCs exist because the
sounding set can drift if a tick is missed. All of it runs; none of it is
redundant. Perceived Stop latency for anything already sounding is still
immediate — phase 2 only mops up the queue tail.

`panicAll()` runs phases 1–2 across **all 16 channels on every known output**
(per-pitch offs replaced by CC 120/123/64 only — panic has no sounding set),
and is wired to:
- The Panic button in the left column — active even when no scheduler is
  playing.
- `window.addEventListener('pagehide', panicAll)` **and** `beforeunload`.
  **[rev2]** `beforeunload` alone is unreliable (bfcache, mobile, tab kill);
  `pagehide` catches more. Both are best-effort — sends during teardown may
  not flush on every platform, which is why the Panic button exists.

**[rev2] Port changes are disconnect-scoped.** The `statechange` handler fires
on *connect* as well as disconnect. Do **not** wire it to `panicAll()` — the
initial draft's wiring would have silenced everything the moment a third
device was plugged in. The rule: on `statechange`, refresh the port dropdowns;
if a **disconnected** port has an active scheduler, `stop()` that scheduler
(phase 1 will throw into the try/catch — that's fine, the notes died with the
port) and surface "port disconnected" on its card. Nothing else reacts.

### UI changes in `frontend/components/midi.js`

**Left column — new MIDI Out section**, placed after the soundfont picker in
`initMidi()`:

```
┌─ MIDI Out ─────────────────────────────────┐
│ ✓ 2 output ports found                     │
│ [Request access]            [⏻ Panic]      │
└────────────────────────────────────────────┘
```

The banner is one of four states:

| Condition | Message |
|---|---|
| `!isSupported()` | Web MIDI is not available in this browser. Chrome or Edge is required. |
| Not a secure context | Web MIDI requires HTTPS or localhost. Open StemForge at `http://localhost:8765`. |
| Access denied | MIDI access was denied. Click Request access and allow the prompt. |
| OK, zero ports | No MIDI outputs found. Connect an interface — the list updates automatically. |

Follow the existing `banner banner-error` / `banner-warn` class pattern
already used for the OSMD load failure.

**[rev2] "Request access", not "Rescan".** `MIDIAccess.outputs` is a live map
kept current by `statechange` — there is nothing to rescan, and a rescan
button would be a no-op. The button's real job is re-invoking
`requestMIDIAccess()` after a denial (or before first grant). Name it for what
it does; hide it once access is granted.

**Per-card — new MIDI Out row** in `buildMidiCard()`, inserted between
`instrumentRow` and `waveContainer`:

```js
const midiOutRow = el('div', { className: 'midi-out-row' },
  el('label', { className: 'text-dim' }, 'MIDI Out:'),
  portSelect,       // "None" + one option per output
  channelSelect,    // 1–16
  sendBtn,          // "▶ Send"
  outStopBtn,       // "■ Stop"
  progChangeLabel,  // checkbox "Send Program Change" — UNCHECKED by default
  outStatus,        // "Playing 0:42 / 3:07" or error text
);
```

The row is hidden entirely when `!isSupported()` — a dead control is worse
than no control.

**Exclusivity.** Module-level `const _outSchedulers = []`, mirroring the
existing `midiPlayers` / `stopOtherPlayers` convention. Starting a card's Send
stops any other scheduler *on the same port and channel* only. Two cards on
different ports (modwave + wavestate) must be able to play simultaneously —
that is the entire point of the feature.

**Independence from FluidSynth.** The hardware scheduler and the wavesurfer
preview are separate transports. Send does not start the waveform; Play does
not start the hardware. Do not attempt to sync them. Expect the global
transport bar to show FluidSynth audio while hardware plays something else —
that is correct behaviour, not a bug.

### Program Change

The instrument dropdown selects a **GM program for FluidSynth**. Sending it to
hardware would overwrite whatever patch the user has dialled in on the modwave
— hostile default behaviour.

The "Send Program Change" checkbox is **off by default**. When on, the
scheduler emits `[0xC0 | (channel-1), program]` once at `play()` before the
first note, using the card's current instrument selection.

**[rev2] Drum suppression keys off the stem, not a dropdown value.** The
initial draft suppressed PC when the instrument selection was "Drum Kit" — no
such dropdown entry exists; the selector is populated from
`/api/midi/gm-programs` (128 GM names), and drum-ness is a property of the
stem label (`STEM_IS_DRUM` / the ADTOF routing). The rule: when the card's
stem is a drum stem (or the track reports `is_drum`), no Program Change is
sent even with the checkbox on — GM percussion is a channel convention, not a
program, and neither Korg is a GM drum module.

Likewise, **`is_drum` must not force channel 10.** On hardware the channel is
entirely the user's choice via the dropdown.

### Persistence

Port name + channel per stem label, in `localStorage` under key
`stemforge.midiout` (the frontend has no existing localStorage usage; this
establishes the `stemforge.*` prefix). Restore by matching on **port name**,
not port ID — IDs are not stable across reboots. Two identical interfaces
produce duplicate names; first match wins, documented as a limitation.
Silently fall back to "None" when no name matches. This is the last
implementation step and can be dropped without affecting anything else.

### CSS — `frontend/style.css`

```css
.midi-out-row { }        /* flex, gap 8px, align center — mirror .midi-instrument-row */
.midi-out-port { }       /* min-width ~180px, truncate long port names */
.midi-out-status { }     /* var(--text-dim), monospace time display */
.midi-out-section { }    /* left column panel, mirror the soundfont picker block */
.btn-panic { }           /* var(--error) border, filled on hover */
```

---

## DO NOT

These are deliberate exclusions. Implementing any of them silently expands
scope or introduces a failure mode.

- **Do not add `python-rtmidi`, `mido`, or any Python MIDI dependency.** The
  port belongs to the browser. Adding a server-side path creates two divergent
  implementations.
- **Do not request SysEx** — `requestMIDIAccess({ sysex: false })`. SysEx
  triggers a heavier permission prompt and opens the door to device-specific
  messages that could alter synth state.
- **Do not use `MIDIOutput.clear()`.** The two-phase Stop exists specifically
  so nothing depends on it.
- **Do not run the tick on the main thread.** Hidden-tab timer throttling
  breaks playback by seconds, not milliseconds.
- **Do not wire `statechange` to `panicAll()`.** It fires on connect too;
  handle disconnects per-scheduler.
- **Do not sync hardware playback to the FluidSynth waveform.** Separate
  transports, deliberately.
- **Do not send Program Change by default.**
- **Do not force channel 10 for `is_drum` tracks.**
- **Do not add MIDI input, clock, MTC, or MMC.** Output only, no transport
  sync.
- **Do not modify session MIDI data.** `/api/midi/events` is read-only.
- **Do not duplicate `_resolve_midi`.** It already exists in
  `backend/api/midi.py`; reuse it.
- **Do not add Web MIDI status to `/api/capabilities`.** The server cannot
  know.
- **Do not touch `frontend/components/audio-player.js`.** The global transport
  bar is for audio; hardware MIDI has its own controls in the MIDI tab.

---

## File inventory

| File | Action | Description |
|------|--------|-------------|
| `frontend/components/webmidi.js` | **New** | Port discovery, worker clock, lookahead scheduler, two-phase stop/panic |
| `backend/api/midi.py` | **Edit** | Add `POST /api/midi/events` (reuses existing `_resolve_midi`) |
| `frontend/components/midi.js` | **Edit** | MIDI Out left-column section + per-card row + track merge |
| `frontend/style.css` | **Edit** | `.midi-out-*` classes, `.btn-panic` |
| `tests/test_midi_events.py` | **New** | Endpoint shape, ordering, overlap merge, clamping, truncation invariant, 404s |
| `docs/INSTRUCTIONS.md` | **Edit** | MIDI Out subsection: usage, browser support, secure-context workarounds, ALSA port contention |
| `README.md` | **Edit** | Feature bullet + browser requirement note |
| `docs/FUTURE_PLANS.md` | **Edit** | Move DAW/MIDI note; add MIDI input as deferred |

No `pyproject.toml` change. No `pyproject.toml.ROCM` / `.MAC` change. No
`uv.lock` change. No `index.html` change.

---

## Licensing impact

**None.** Web MIDI is a W3C specification implemented by the browser. No
package is added to either dependency tree, nothing enters
`THIRD-PARTY-NOTICES.md`, and no audit is required.

Given how much of this project's dependency work has been license triage —
parselmouth, KJ MelBandRoformer, Pedalboard, the unresolved ADTOF lineage — a
feature with zero licensing surface is worth noting explicitly.

---

## Testing

### Automated (pytest)

`tests/test_midi_events.py` covers the backend only:

1. Known `PrettyMIDI` → correct event count (`2 × note count` after merge/skip
   rules).
2. Events sorted ascending by `t`; `off` precedes `on` at equal timestamps.
3. Abutting same-pitch notes produce off-then-on, not on-then-off.
4. **[rev2]** Overlapping same-pitch notes are merged: the earlier note's off
   lands at the later note's start; no interleaved on/on/off/off survives.
5. Note-on velocity clamped to 1–127; `off` events carry `vel: 0` untouched;
   pitch clamped to 0–127.
6. Zero-length notes (`end <= start`) dropped.
7. Multi-instrument `merged` returns one `tracks` entry per instrument.
8. Unknown `stem_label` → 404; empty session `merged` → 404.
9. Over-cap input sets `truncated: true`, respects the 20000 cap, **and
   [rev2] contains no `on` without its matching `off`** — the truncation
   invariant, tested at the boundary.

Web MIDI itself cannot be exercised in pytest and should not be mocked into a
false sense of coverage.

### Manual — hardware validation protocol

For Julio (modwave + wavestate, Linux). Structured the same way as the AMD
testing protocol.

**Pre-flight**
1. Browser and version. Chrome/Edge required; confirm Firefox behaviour
   separately rather than assuming (Firefox gates Web MIDI behind a
   site-permission add-on flow).
2. StemForge opened at `http://localhost:8765`, not a LAN IP — Web MIDI needs
   a secure context.
3. `aconnect -l` — confirm both Korgs appear as ALSA sequencer clients before
   touching StemForge.
4. Close any DAW or sequencer holding the ports open.

**Functional**
5. Both devices listed in the port dropdown, named recognisably.
6. Send a bass stem to the wavestate on ch 1 → notes sound, pitches correct.
7. Send a second stem to the modwave concurrently → both play, neither
   interrupts the other.
8. Stop mid-playback → audible notes cut immediately; nothing hangs after the
   ~300 ms queue flush.
9. Panic during playback → all ports silent.
10. Close the browser tab mid-playback → no hanging notes (`pagehide` /
    `beforeunload`; best-effort — note platform if it fails).
11. Unplug an interface mid-playback → that card stops with "port
    disconnected", the other card keeps playing, port list updates, no
    console errors. Plug a third device in mid-playback → **nothing** stops.
12. Program Change off → synth patch unchanged. Toggle on → patch changes.
    On a drum stem, toggle on → patch still unchanged (suppression).
13. **[rev2]** Switch to another application for 2+ minutes while both cards
    play (tab hidden) → playback continues gapless. This validates the worker
    clock and is the single most likely real-world failure mode.
14. Import an external multi-instrument `.mid` via the Import button → its
    card plays all instruments on the selected port/channel.

**Timing**
15. Play a quantized drum stem for 3+ minutes. Listen for gaps or jitter
    against the FluidSynth render played separately. (Long-run *drift* is
    impossible by construction — every timestamp derives from one `originMs`
    — so what this test catches is starvation gaps and send jitter.)
16. Load a dense stem (>5000 notes) and confirm no stutter at bar lines under
    CPU load.

Report format: numbered item, pass/fail, and for failures the browser console
output plus `aconnect -l` at the time.

---

## Effort estimate

| Component | Estimate |
|-----------|----------|
| `POST /api/midi/events` + flattening rules (incl. overlap merge, truncation invariant) | 2–3 h |
| `webmidi.js` — access, port discovery, hot-plug | 2 h |
| `webmidi.js` — worker clock, lookahead scheduler, two-phase stop/panic | 5–6 h |
| `midi.js` — left-column section + banners | 2 h |
| `midi.js` — per-card row, track merge, exclusivity, status | 2–3 h |
| CSS | 1 h |
| pytest suite | 2 h |
| localStorage persistence | 1 h |
| Docs | 1 h |
| Hardware validation round-trip with Julio | 2 h |
| **Total** | **~20–23 h** |

---

## Risks and mitigations

**Browser coverage.** Chrome and Edge implement Web MIDI on Linux via ALSA.
Firefox's support is gated behind a site-permission add-on flow whose current
behaviour should be verified on real hardware rather than assumed.
*Mitigation:* feature-detect and show a clear message naming Chrome/Edge; hide
the controls rather than showing broken ones. Confirm Firefox empirically
during validation and document the finding.

**Secure-context requirement.** `navigator.requestMIDIAccess` is undefined on
plain `http://` to a non-localhost host. Anyone reaching StemForge over the
LAN — which the multi-user session work explicitly supports — loses the
feature with no obvious explanation. *Mitigation:* detect
`window.isSecureContext` and say so precisely, naming `localhost:8765`.
**[rev2]** `INSTRUCTIONS.md` documents the two real workarounds for LAN use:
serve StemForge behind an HTTPS reverse proxy, or (single-machine testing
only) Chrome's `#unsafely-treat-insecure-origin-as-secure` flag.

**Queue tail on Stop.** Events inside the lookahead window are irrevocable
once sent. *Mitigation:* the two-phase Stop; worst-case residual sound is one
queued note lasting ~`FLUSH_MS`, then silenced.

**Timer starvation.** *Mitigation:* the worker clock removes hidden-tab
throttling, the dominant cause; the 250 ms lookahead against a 100 ms tick
absorbs ordinary main-thread jank; a late tick sends overdue events
immediately rather than skipping them — late is recoverable, silent is not.

**Payload size.** A dense 4-minute stem can reach ~10000 events, roughly
500 KB of JSON. Fine over localhost, wasteful over a LAN. *Mitigation:* the
20000-event cap plus a UI nudge toward Clean Up. Compact encoding is a later
optimisation if it ever matters.

**Port contention.** ALSA ports held exclusively by a running DAW will not
appear. *Mitigation:* troubleshooting note in `INSTRUCTIONS.md` and step 4 of
the pre-flight.

**Dropped MIDI features.** `PrettyMIDI.pitch_bends` and `control_changes` are
not emitted. BasicPitch produces neither, but the vocal MIDI path may produce
pitch bends worth carrying later. *Mitigation:* documented limitation; the
`tracks` response shape has room to add `bend` and `cc` event types without a
breaking change.

**Stale snapshot.** Clean Up / Transpose during playback leaves hardware on
pre-edit data until the next Send. *Mitigation:* accepted and documented;
no invalidation plumbing.

---

## Implementation order

Each step is independently testable.

1. **`POST /api/midi/events`** + pytest suite. Verify by curl against a live
   session — correct counts, correct ordering, overlap merge, truncation
   invariant. No frontend yet.
2. **`webmidi.js` — access and discovery.** Wire nothing to the UI; confirm
   from the browser console that both Korgs enumerate.
3. **`webmidi.js` — worker clock, scheduler, two-phase stop.** Drive it from
   the console against a hand-written event array. Confirm notes sound, Stop
   silences (including a queued-tail test: Stop immediately after Send), and
   playback survives a hidden tab. This is the highest-risk step; do not
   proceed until timing feels right.
4. **`midi.js` — left-column section.** Banners, Request access, Panic. Panic
   must work before any card can start a note.
5. **`midi.js` — per-card row.** Port/channel selects, track merge, Send/Stop,
   status, exclusivity, disconnect handling.
6. **Program Change checkbox** (with drum-stem suppression).
7. **CSS pass.**
8. **localStorage persistence.**
9. **Docs** — `INSTRUCTIONS.md`, `README.md`, `FUTURE_PLANS.md`.
10. **Hardware validation** with Julio against the protocol above.

Steps 1–5 are the working feature. 6–8 are polish and can ship in a follow-up.

---

## Deferred

**MIDI input.** Recording from hardware into StemForge — capturing a
performance on the modwave as a new MIDI track, or as an alternative to
BasicPitch extraction. Web MIDI supports `MIDIInput` with the same permission
grant, so the access plumbing in `webmidi.js` is reusable. The harder part is
the return leg: StemForge has no audio capture path at all, only file upload,
so hardware audio still has to be recorded externally and dropped on the
Export or Mix zone. Input is a separate spec.

**Transport sync.** MIDI Clock, MTC, and MMC so the hardware follows
StemForge's transport bar, or vice versa. Meaningful only once there is a
unified timeline; currently each tab has its own transport.

**Pitch bend and CC.** Carry `PrettyMIDI.pitch_bends` and `control_changes`
through the event list. Waiting on a pipeline that actually produces them.

**Multi-port merged playback.** Route each instrument of the `merged` MIDI to
a different port/channel from a single card, rather than requiring per-stem
cards. The `tracks[]` response shape already supports it; only UI and
scheduler fan-out are missing.

**Seek / start-from-playhead.** Reintroduce a `startSec` argument to `play()`
along with a defined rule for notes already sounding at the seek point.
