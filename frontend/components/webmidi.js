/**
 * Web MIDI output — port discovery and lookahead event scheduling.
 *
 * Wraps navigator.requestMIDIAccess and provides a scheduler that streams
 * note events to a hardware port with sub-millisecond timing accuracy.
 *
 * Timing model: a Worker-driven tick (100 ms) scans a lookahead window
 * (250 ms) and hands each event to output.send() with an absolute
 * DOMHighResTimeStamp. The browser's MIDI subsystem does the precise
 * dispatch; the tick only needs to be roughly on time ("two clocks").
 * The tick runs in a Web Worker because main-thread timers are throttled
 * to >= 1 s in hidden tabs — and auditioning hardware synths with the tab
 * hidden is the expected use, not the edge case.
 *
 * Stop runs in two phases: events already handed to output.send() with
 * future timestamps cannot be recalled (MIDIOutput.clear() is not relied
 * on — inconsistent support), so note-offs and panic CCs are sent once
 * immediately and once timestamped after the lookahead queue has drained.
 */

const LOOKAHEAD_MS = 250;  // how far ahead of the cursor events are queued
const TICK_MS = 100;       // worker metronome period
const FLUSH_MS = LOOKAHEAD_MS + 50;  // second-phase stop margin

/** Startup delay before t=0, absorbing first-tick jitter. Exported so the
 *  UI can delay the waveform cursor by the same amount and stay in step. */
export const LEAD_MS = 120;

let midiAccess = null;
const portListeners = [];
const activeSchedulers = new Set();

// ─── Access & discovery ──────────────────────────────────────────────────

export function isSupported() {
  return typeof navigator !== 'undefined' && !!navigator.requestMIDIAccess;
}

/**
 * Request MIDI access (sysex: false — deliberately) and wire statechange.
 * Idempotent; rejects if unsupported or the user denies the prompt.
 */
export async function initWebMidi() {
  if (!isSupported()) throw new Error('Web MIDI is not supported in this browser');
  if (midiAccess) return midiAccess;
  midiAccess = await navigator.requestMIDIAccess({ sysex: false });
  midiAccess.onstatechange = (e) => {
    // A scheduler whose port vanished is stopped here, centrally — the
    // notes died with the port; sends would only throw. Connects are
    // ignored beyond notifying listeners (never panic on hot-plug).
    if (e.port && e.port.type === 'output' && e.port.state === 'disconnected') {
      for (const s of [...activeSchedulers]) {
        if (s.outputId === e.port.id) s.handleDisconnect();
      }
    }
    for (const cb of portListeners) {
      try { cb(e.port); } catch (err) { console.error('onPortsChanged handler failed', err); }
    }
  };
  return midiAccess;
}

/** @returns {{id: string, name: string, manufacturer: string}[]} */
export function getOutputs() {
  if (!midiAccess) return [];
  return [...midiAccess.outputs.values()].map((o) => ({
    id: o.id,
    name: o.name || o.id,
    manufacturer: o.manufacturer || '',
  }));
}

export function getOutputById(id) {
  return midiAccess ? midiAccess.outputs.get(id) || null : null;
}

/** Subscribe to hot-plug events. cb receives the MIDIPort that changed. */
export function onPortsChanged(cb) {
  portListeners.push(cb);
}

// ─── Worker clock ────────────────────────────────────────────────────────
// The worker carries no state and no times — it is purely a metronome, so
// the worker/main-thread timeOrigin difference is irrelevant.

let clockWorker = null;
const tickSubscribers = new Set();

function ensureClock() {
  if (clockWorker) return;
  const src = `setInterval(() => postMessage(0), ${TICK_MS});`;
  clockWorker = new Worker(URL.createObjectURL(new Blob([src], { type: 'application/javascript' })));
  clockWorker.onmessage = () => {
    for (const fn of [...tickSubscribers]) fn();
  };
}

// ─── Panic plumbing ──────────────────────────────────────────────────────

function sendPanicCCs(output, ch, ts) {
  output.send([0xB0 | ch, 120, 0], ts);  // All Sound Off
  output.send([0xB0 | ch, 123, 0], ts);  // All Notes Off
  output.send([0xB0 | ch, 64, 0], ts);   // Sustain off
}

function sendNoteOffs(output, ch, pitches, ts) {
  for (const note of pitches) output.send([0x80 | ch, note, 0], ts);
}

/**
 * Silence every known output on all 16 channels, immediately and again
 * after the lookahead queue drains. Safe to call at any time.
 */
export function panicAll() {
  for (const s of [...activeSchedulers]) s.stop();
  if (!midiAccess) return;
  const flushAt = performance.now() + FLUSH_MS;
  for (const output of midiAccess.outputs.values()) {
    for (let ch = 0; ch < 16; ch++) {
      try {
        sendPanicCCs(output, ch, 0);
        sendPanicCCs(output, ch, flushAt);
      } catch (err) { /* port gone — nothing left to silence */ }
    }
  }
}

if (typeof window !== 'undefined') {
  // Best-effort teardown silence; pagehide catches cases beforeunload
  // misses (bfcache, mobile). The Panic button is the reliable path.
  window.addEventListener('pagehide', panicAll);
  window.addEventListener('beforeunload', panicAll);
}

// ─── Scheduler ───────────────────────────────────────────────────────────

/**
 * Slice an event array to start at `startSec`.
 *
 * A note already sounding at the seek point is skipped rather than
 * re-triggered from the middle of its envelope — right for a pad, clearly
 * wrong for anything plucked. That leaves its note-off orphaned in the
 * remaining stream, and a bare off would cut a note the player happens to
 * be holding on the keyboard, so orphaned offs are dropped too. A
 * re-trigger of the same pitch clears the flag, keeping its own off.
 */
function sliceFrom(evts, startSec) {
  if (startSec <= 0) return evts;
  const hanging = new Set();  // sounding at startSec, note-off still ahead
  let i = 0;
  for (; i < evts.length && evts[i].t < startSec; i++) {
    const e = evts[i];
    if (e.type === 'on') hanging.add(e.note);
    else hanging.delete(e.note);
  }
  if (!hanging.size) return evts.slice(i);
  const out = [];
  for (; i < evts.length; i++) {
    const e = evts[i];
    if (hanging.has(e.note)) {
      hanging.delete(e.note);
      if (e.type === 'off') continue;
    }
    out.push(e);
  }
  return out;
}

/**
 * Create a scheduler bound to one output port and one channel (1-16).
 * Port and channel are fixed for the scheduler's lifetime — changing a
 * dropdown mid-playback takes effect on the next Send, by design.
 */
export function createScheduler(outputId, channel) {
  const output = getOutputById(outputId);
  if (!output) throw new Error('MIDI output not found');
  const ch = (channel - 1) & 0x0f;

  let events = [];
  let cursor = 0;
  let originMs = 0;
  let playing = false;
  const sounding = new Set();  // updated at *send* time → covers queued notes
  let finishCb = null;
  let tickCb = null;
  let errorCb = null;

  function cleanup() {
    playing = false;
    tickSubscribers.delete(tick);
    activeSchedulers.delete(scheduler);
  }

  function tick() {
    if (!playing) return;
    const now = performance.now();
    const windowEnd = now + LOOKAHEAD_MS;
    try {
      while (cursor < events.length) {
        const e = events[cursor];
        const absMs = originMs + e.t * 1000;
        if (absMs >= windowEnd) break;
        const status = (e.type === 'on' ? 0x90 : 0x80) | ch;
        // A starved tick sends overdue events immediately rather than
        // skipping them — late is recoverable, silent is not.
        output.send([status, e.note, e.vel], Math.max(absMs, now));
        if (e.type === 'on') sounding.add(e.note);
        else sounding.delete(e.note);
        cursor++;
      }
    } catch (err) {
      // send() throws InvalidStateError on a disconnected port.
      scheduler.handleDisconnect(err);
      return;
    }
    if (tickCb) tickCb(Math.max(0, (now - originMs) / 1000));
    if (cursor >= events.length) {
      const lastT = events.length ? events[events.length - 1].t : 0;
      if (now >= originMs + lastT * 1000) {
        cleanup();
        sounding.clear();
        if (finishCb) finishCb();
      }
    }
  }

  const scheduler = {
    outputId,
    channel,

    /**
     * Begin streaming a flat event array ({t, type, note, vel}, sorted
     * ascending, off-before-on at equal t).
     * opts.program: GM program 0-127 to send once before the first note
     * (the Program Change checkbox) — omit/null to leave the patch alone.
     * opts.startSec: seek position in seconds (default 0). Notes already
     * sounding there are skipped — see sliceFrom(). A startSec past the
     * last event is legal and finishes immediately.
     */
    play(evts, opts = {}) {
      if (playing) scheduler.stop();
      const startSec = Math.max(0, opts.startSec || 0);
      events = sliceFrom(evts, startSec);
      cursor = 0;
      sounding.clear();
      ensureClock();
      if (opts.program != null) {
        output.send([0xC0 | ch, opts.program & 0x7f]);
      }
      // Shifting the origin back by startSec keeps every downstream
      // `originMs + e.t * 1000` calculation working unchanged.
      originMs = performance.now() + LEAD_MS - startSec * 1000;
      playing = true;
      tickSubscribers.add(tick);
      activeSchedulers.add(scheduler);
      tick();  // prime the first window now, not TICK_MS from now
    },

    /**
     * Two-phase halt. Phase 1 (immediate): note-off every sounding pitch,
     * then CC 120/123/64. Phase 2 (after FLUSH_MS): the same again, so
     * note-ons already queued in the browser's MIDI subsystem — which
     * cannot be recalled — are silenced within ~300 ms.
     */
    stop() {
      if (!playing && sounding.size === 0) return;
      cleanup();
      const flushAt = performance.now() + FLUSH_MS;
      try {
        sendNoteOffs(output, ch, sounding, 0);
        sendPanicCCs(output, ch, 0);
        sendNoteOffs(output, ch, sounding, flushAt);
        sendPanicCCs(output, ch, flushAt);
      } catch (err) { /* port gone — notes died with it */ }
      sounding.clear();
    },

    /** Called centrally when this scheduler's port disconnects. */
    handleDisconnect(err) {
      cleanup();
      sounding.clear();
      if (errorCb) errorCb(err || new Error('MIDI port disconnected'));
    },

    isPlaying() {
      return playing;
    },

    onFinish(cb) { finishCb = cb; },
    onTick(cb) { tickCb = cb; },
    onError(cb) { errorCb = cb; },
  };

  return scheduler;
}
