/**
 * MIDI tab — stem selection, extraction, per-stem playback with waveform,
 * GM instrument selector, soundfont picker, preview/save.
 */

import { appState, api, pollJob, el, formatTime, saveFileAs } from '../app.js';
import { createWaveform } from './waveform.js';
import { transportLoad, transportStop } from './audio-player.js';
import { isSupported as webMidiSupported, initWebMidi, getOutputs, onPortsChanged, createScheduler, panicAll, LEAD_MS } from './webmidi.js';

function clearChildren(elem) {
  while (elem.firstChild) elem.removeChild(elem.firstChild);
}

/** GM program names (populated from backend on init). */
let gmPrograms = [];
let stemDefaults = {};
let drumStems = {};

/** ADTOF license warning text (from /api/models) and session acknowledgment. */
let drumLicenseWarning = '';
let drumLicenseAcknowledged = false;

/** LilyPond availability (checked on init). */
let _lilypondAvailable = false;

/** All active MIDI card players — for exclusive playback. */
const midiPlayers = [];

/** Web MIDI output state — access granted, and card-row refresh hooks. */
let midiOutReady = false;
const midiOutRefreshers = [];  // each cb receives getOutputs() on any port change

/**
 * Active per-card hardware schedulers, mirroring the midiPlayers /
 * stopOtherPlayers convention. Exclusivity is per port+channel only:
 * two cards on different ports (modwave + wavestate) play simultaneously.
 */
const _outSchedulers = [];

/**
 * Port + channel per stem label, persisted across sessions. Ports are
 * matched by NAME, not id — ids are not stable across reboots. Two
 * identical interfaces produce duplicate names; first match wins
 * (documented limitation). No match falls back silently to "None".
 */
const MIDI_OUT_LS_KEY = 'stemforge.midiout';

function loadMidiOutPrefs() {
  try {
    return JSON.parse(localStorage.getItem(MIDI_OUT_LS_KEY)) || {};
  } catch (err) {
    return {};
  }
}

function saveMidiOutPref(label, portName, channel) {
  try {
    const prefs = loadMidiOutPrefs();
    prefs[label] = { port: portName, channel };
    localStorage.setItem(MIDI_OUT_LS_KEY, JSON.stringify(prefs));
  } catch (err) { /* storage unavailable — persistence is best-effort */ }
}

function stopOtherPlayers(except) {
  for (const p of midiPlayers) {
    if (p.ws !== except && p.ws.isPlaying()) {
      p.ws.stop();
      p.playBtn.textContent = '\u25B6 Play';
    }
  }
}

export function initMidi() {
  const panel = document.getElementById('panel-midi');

  // ─── Mode bar (Notes | Lyrics) ───
  const modeBar = el('div', { className: 'midi-mode-bar' },
    el('div', { className: 'midi-mode-selector' },
      el('button', { className: 'midi-mode-btn active', 'data-mode': 'notes', onClick: () => switchMidiMode('notes') }, 'Notes'),
      el('button', { className: 'midi-mode-btn', 'data-mode': 'lyrics', onClick: () => switchMidiMode('lyrics') }, 'Lyrics'),
    ),
  );
  panel.appendChild(modeBar);

  const layout = el('div', { className: 'two-col' });

  // ─── Left: controls ───
  const left = el('div', { className: 'col-left' });
  const notesControls = el('div', { id: 'midi-controls-notes' });
  const lyricsControls = el('div', { id: 'midi-controls-lyrics', style: { display: 'none' } });

  const stemSection = el('div', { className: 'form-group' },
    el('label', {}, 'Stems to process'),
    el('div', { className: 'checkbox-group', id: 'midi-stems' },
      el('span', { className: 'text-dim' }, 'Run separation first'),
    ),
    el('div', { className: 'text-dim hidden', id: 'midi-drum-hint' },
      'Drum stems use ADTOF drum transcription (kick, snare, tom, hi-hat, cymbal → GM drum kit).',
    ),
  );

  const keyGroup = el('div', { className: 'form-group' },
    el('label', {}, 'Key'),
    el('select', { id: 'midi-key' },
      el('option', { value: 'Any' }, 'Any (auto-detect)'),
      ...['C major','C minor','D major','D minor','E major','E minor',
          'F major','F minor','G major','G minor','A major','A minor',
          'B major','B minor'].map(k => el('option', { value: k }, k)),
    ),
  );

  const bpmGroup = el('div', { className: 'form-group' },
    el('label', {}, 'BPM'),
    el('input', { type: 'number', id: 'midi-bpm', value: '120', min: '20', max: '300' }),
  );

  const tsGroup = el('div', { className: 'form-group' },
    el('label', {}, 'Time Signature'),
    el('select', { id: 'midi-ts' },
      el('option', { value: '4/4' }, '4/4'),
      el('option', { value: '3/4' }, '3/4'),
      el('option', { value: '6/8' }, '6/8'),
      el('option', { value: '2/4' }, '2/4'),
    ),
  );

  const onsetGroup = el('div', { className: 'form-group' },
    el('label', {}, 'Onset threshold'),
    el('div', { className: 'slider-row' },
      el('input', { type: 'range', id: 'midi-onset', min: '0', max: '1', step: '0.05', value: '0.5' }),
      el('span', { className: 'slider-value', id: 'midi-onset-val' }, '0.50'),
    ),
  );

  const frameGroup = el('div', { className: 'form-group' },
    el('label', {}, 'Frame threshold'),
    el('div', { className: 'slider-row' },
      el('input', { type: 'range', id: 'midi-frame', min: '0', max: '1', step: '0.05', value: '0.3' }),
      el('span', { className: 'slider-value', id: 'midi-frame-val' }, '0.30'),
    ),
  );

  // ─── SoundFont selector ───
  const sf2Group = el('div', { className: 'form-group' },
    el('label', {}, 'SoundFont'),
    el('div', { className: 'sf2-row' },
      el('input', { type: 'text', id: 'midi-sf2-path', readonly: 'true', placeholder: 'System default' }),
      el('button', { className: 'btn btn-sm', id: 'midi-sf2-browse', title: 'Browse for .sf2 file' }, 'Browse'),
      el('button', { className: 'btn btn-sm', id: 'midi-sf2-reset', title: 'Reset to system default' }, 'Reset'),
    ),
  );

  // ─── MIDI Out (Web MIDI hardware output) ───
  const midiOutSection = buildMidiOutSection();

  const extractBtn = el('button', { className: 'btn btn-primary', id: 'midi-start', disabled: 'true' },
    'Extract MIDI',
  );

  // ─── Import MIDI file ───
  const importInput = el('input', { type: 'file', accept: '.mid,.midi', style: { display: 'none' }, id: 'midi-import-input' });
  const importBtn = el('button', { className: 'btn btn-sm', id: 'midi-import' }, 'Import MIDI file');

  notesControls.append(stemSection, keyGroup, bpmGroup, tsGroup, onsetGroup, frameGroup, sf2Group, midiOutSection, extractBtn, importInput, importBtn);
  left.append(notesControls, lyricsControls);

  // ─── Lyrics control panel ───
  const lyricsSourceLabel = el('label', { className: 'field-label' }, 'Source Audio');
  const lyricsSourceSelect = el('select', { id: 'lyrics-source', className: 'select' });
  const lyricsSourceGroup = el('div', { className: 'form-group' },
    lyricsSourceLabel, lyricsSourceSelect,
  );

  const lyricsEngineLabel = el('label', { className: 'field-label' }, 'Engine');
  const lyricsEngineSelect = el('select', {
    id: 'lyrics-engine',
    className: 'select',
    title: 'Whisper is the standard speech recognition engine — fast, supports word-level timestamps, runs on CPU or GPU. Qwen2-Audio is a multimodal language model — slower, more accurate on sung or unusual vocals, no word timestamps, GPU only. The 4-bit Qwen variant fits in ~9 GB VRAM with minor quality loss. Long audio is automatically chunked into overlapping 24-second windows; transcripts are stitched together by matching text in the overlap regions.',
  });
  const lyricsEngineGroup = el('div', { className: 'form-group' },
    lyricsEngineLabel, lyricsEngineSelect,
  );

  const lyricsLangLabel = el('label', { className: 'field-label' }, 'Language');
  const lyricsLangSelect = el('select', { id: 'lyrics-language', className: 'select' },
    el('option', { value: '' }, 'Auto-detect'),
    el('option', { value: 'en' }, 'English'),
    el('option', { value: 'zh' }, 'Chinese'),
    el('option', { value: 'ja' }, 'Japanese'),
    el('option', { value: 'ko' }, 'Korean'),
    el('option', { value: 'es' }, 'Spanish'),
    el('option', { value: 'fr' }, 'French'),
    el('option', { value: 'de' }, 'German'),
    el('option', { value: 'pt' }, 'Portuguese'),
    el('option', { value: 'it' }, 'Italian'),
    el('option', { value: 'ru' }, 'Russian'),
    el('option', { value: 'ar' }, 'Arabic'),
    el('option', { value: 'hi' }, 'Hindi'),
  );
  const lyricsLangGroup = el('div', { className: 'form-group' }, lyricsLangLabel, lyricsLangSelect);

  // Hint field — biases the transcription model toward specific vocabulary.
  // Plumbed through TranscribeRequest.prompt for both engines.  The panel is
  // built once and only show/hidden by switchMidiMode, so the input value
  // survives mode switches naturally — no explicit restore needed.
  const lyricsHintGroup = el('div', { className: 'form-group' },
    el('label', { 'for': 'midi-lyrics-hint', className: 'field-label' }, 'Hint (optional)'),
    el('input', {
      type: 'text',
      id: 'midi-lyrics-hint',
      className: 'midi-lyrics-hint-input',
      maxlength: '224',
      placeholder: 'Song title, names, key phrases — e.g. "Catrina, Feliz cumpleaños"',
      title: 'Words and phrases here will bias the transcription model toward similar vocabulary. Useful for proper nouns (names, places), song titles, recurring phrases the model gets wrong, and uncommon spellings. Works with both Whisper and Qwen engines. Keep it short — a single line of the most distinctive terms is enough.',
      value: appState.lyricsHint || '',
      onInput: (e) => { appState.lyricsHint = e.target.value; },
    }),
  );

  const fmtTxt = el('input', { type: 'checkbox', id: 'lyrics-fmt-txt', checked: 'true', disabled: 'true' });
  const fmtLrc = el('input', { type: 'checkbox', id: 'lyrics-fmt-lrc', checked: 'true' });
  const fmtSrt = el('input', { type: 'checkbox', id: 'lyrics-fmt-srt', checked: 'true' });
  const lyricsFmtGroup = el('div', { className: 'form-group' },
    el('label', {}, 'Output formats'),
    el('div', { className: 'checkbox-group' },
      el('label', {}, fmtTxt, ' .txt (always)'),
      el('label', {}, fmtLrc, ' .lrc'),
      el('label', {}, fmtSrt, ' .srt'),
    ),
  );

  const lyricsCoarseNotice = el('div', { className: 'lyrics-notice hidden', id: 'lyrics-coarse-notice' },
    'Qwen produces segment-level timing; .lrc and .srt use coarse timestamps.',
  );

  const lyricsTranscribeBtn = el('button', { className: 'btn btn-primary', id: 'lyrics-transcribe', disabled: 'true' },
    'Transcribe',
  );

  const lyricsLoadHint = el('div', { className: 'text-dim hidden', id: 'lyrics-load-hint' },
    'Load audio or run separation first. Lyrics transcription works best on an isolated vocal stem.',
  );

  // ─── Advanced options ───
  const lyricsCondRow = el('div', {
    className: 'form-group',
    id: 'lyrics-cond-row',
    style: { display: 'none' },   // shown only for Whisper engines
  },
    el('label', {
      className: 'checkbox-label',
      title: "Whisper conditions each transcription window on its own previous output. For continuous speech this improves coherence across sentences. For music it commonly causes hallucinated repetition loops on instrumental passages and fade-outs, because the model has no fresh acoustic content and keeps predicting whatever it just emitted. Off by default. Turn on for spoken-word audio (audiobooks, podcasts, dictation).",
    },
      el('input', { type: 'checkbox', id: 'lyrics-condition-on-previous' }),
      ' Cross-window conditioning',
    ),
  );

  const lyricsCollapseRow = el('div', { className: 'form-group' },
    el('label', {
      className: 'checkbox-label',
      title: 'Collapses runs of identical lines longer than the maximum below into that many copies followed by "[...]". Useful when Whisper hallucinates a repetition loop, but will also collapse genuinely repeated lyrics (an outro that repeats a line ten times, a chorus with many "oh, oh, oh" repetitions, etc.). Turn off if your song has real, intentional repetition you want preserved verbatim.',
    },
      el('input', { type: 'checkbox', id: 'lyrics-collapse-reps', checked: 'true' }),
      ' Collapse repeated lines',
    ),
    el('div', { className: 'form-row', style: { marginTop: '4px', display: 'flex', alignItems: 'center', gap: '6px' } },
      el('label', { style: { fontSize: '0.85em' } }, 'Max repetitions before collapse:'),
      el('input', { type: 'number', id: 'lyrics-max-run', value: '4', min: '2', max: '20',
                    style: { width: '60px' } }),
    ),
  );

  const lyricsAdvanced = el('details', { className: 'lyrics-advanced' },
    el('summary', {}, 'Advanced'),
    lyricsCondRow,
    lyricsCollapseRow,
  );

  lyricsControls.append(
    lyricsSourceGroup, lyricsEngineGroup, lyricsLangGroup, lyricsHintGroup, lyricsFmtGroup,
    lyricsCoarseNotice, lyricsAdvanced, lyricsTranscribeBtn, lyricsLoadHint,
  );

  // ─── Right: results ───
  const right = el('div', { className: 'col-right' });
  const notesResults = el('div', { id: 'midi-results-notes' });
  const lyricsResults = el('div', { id: 'midi-results-lyrics', style: { display: 'none' } });

  const progressCard = el('div', { className: 'card hidden', id: 'midi-progress' },
    el('div', { className: 'progress-container' },
      el('div', { className: 'progress-bar' },
        el('div', { className: 'progress-fill', id: 'midi-progress-fill' }),
      ),
      el('div', { className: 'progress-label' },
        el('span', { id: 'midi-stage' }, ''),
        el('span', { id: 'midi-pct' }, '0%'),
      ),
    ),
  );

  const resultsContainer = el('div', { id: 'midi-results' });
  notesResults.append(progressCard, resultsContainer);

  const lyricsProgressCard = el('div', { className: 'card hidden', id: 'lyrics-progress' },
    el('div', { className: 'progress-container' },
      el('div', { className: 'progress-bar' },
        el('div', { className: 'progress-fill', id: 'lyrics-progress-fill' }),
      ),
      el('div', { className: 'progress-label' },
        el('span', { id: 'lyrics-stage' }, ''),
        el('span', { id: 'lyrics-pct' }, '0%'),
      ),
    ),
  );

  const lyricsResultContainer = el('div', { id: 'lyrics-result' });
  lyricsResults.append(lyricsProgressCard, lyricsResultContainer);

  right.append(notesResults, lyricsResults);
  layout.append(left, right);
  panel.appendChild(layout);

  // Hidden file input for SF2 browsing
  const sf2Input = el('input', { type: 'file', id: 'midi-sf2-input', accept: '.sf2,.sf3', style: { display: 'none' } });
  panel.appendChild(sf2Input);

  // ─── Wire events ───
  document.getElementById('midi-onset').addEventListener('input', (e) => {
    document.getElementById('midi-onset-val').textContent = parseFloat(e.target.value).toFixed(2);
  });
  document.getElementById('midi-frame').addEventListener('input', (e) => {
    document.getElementById('midi-frame-val').textContent = parseFloat(e.target.value).toFixed(2);
  });
  document.getElementById('midi-start').addEventListener('click', startExtraction);

  // Show the drum transcription hint whenever a drum stem is checked
  document.getElementById('midi-stems').addEventListener('change', syncDrumHint);

  // Import MIDI file
  document.getElementById('midi-import').addEventListener('click', () => {
    document.getElementById('midi-import-input').click();
  });
  document.getElementById('midi-import-input').addEventListener('change', handleMidiImport);

  // SoundFont controls
  document.getElementById('midi-sf2-browse').addEventListener('click', () => {
    document.getElementById('midi-sf2-input').click();
  });
  document.getElementById('midi-sf2-input').addEventListener('change', handleSf2Browse);
  document.getElementById('midi-sf2-reset').addEventListener('click', resetSoundfont);

  appState.on('stemsReady', (stemPaths) => {
    populateStemCheckboxes(stemPaths);
    document.getElementById('midi-start').disabled = false;
  });

  // Load GM programs, current soundfont, and check LilyPond on init
  loadGmPrograms();
  loadCurrentSoundfont();
  checkLilypondAvailability();

  // Lyrics-mode wiring
  loadLyricsEngines();
  refreshLyricsSources();
  document.getElementById('lyrics-engine').addEventListener('change', onLyricsEngineChange);
  document.getElementById('lyrics-transcribe').addEventListener('click', startLyricsTranscription);

  appState.on('fileLoaded', refreshLyricsSources);
  appState.on('stemsReady', refreshLyricsSources);
}

async function loadGmPrograms() {
  try {
    const data = await api('/midi/gm-programs');
    gmPrograms = data.programs || [];
    stemDefaults = data.defaults || {};
    drumStems = data.drum_stems || {};
  } catch { /* fail silently, will use defaults */ }
  try {
    const models = await api('/models');
    drumLicenseWarning = models.drum_midi?.[0]?.license_warning || '';
    syncDrumHint();
  } catch { /* no warning shown if models endpoint fails */ }
}

async function loadCurrentSoundfont() {
  try {
    const data = await api('/midi/soundfont');
    const input = document.getElementById('midi-sf2-path');
    if (data.path) {
      input.value = data.path;
    } else {
      input.value = '';
      input.placeholder = 'System default';
    }
  } catch { /* ignore */ }
}

async function checkLilypondAvailability() {
  try {
    const data = await api('/capabilities');
    _lilypondAvailable = data.lilypond?.available ?? false;
  } catch { /* assume unavailable */ }
}

async function handleSf2Browse() {
  const fileInput = document.getElementById('midi-sf2-input');
  const file = fileInput.files[0];
  if (!file) return;

  // We need the user to provide a server-side path, not upload the file.
  // The file input gives us the filename; prompt for the full path.
  const path = prompt(
    'Enter the full server path to the SoundFont file:\n\n' +
    `(Selected: ${file.name})`,
    `/usr/share/soundfonts/${file.name}`,
  );
  fileInput.value = '';
  if (!path) return;

  try {
    const res = await api('/midi/soundfont', {
      method: 'POST',
      body: JSON.stringify({ path }),
    });
    document.getElementById('midi-sf2-path').value = res.path;
  } catch (err) {
    alert(`SoundFont error: ${err.message}`);
  }
}

async function resetSoundfont() {
  try {
    const res = await api('/midi/soundfont', {
      method: 'POST',
      body: JSON.stringify({ path: '' }),
    });
    const input = document.getElementById('midi-sf2-path');
    input.value = res.path || '';
    if (!res.path) input.placeholder = 'System default';
  } catch (err) {
    alert(`Reset failed: ${err.message}`);
  }
}

function populateStemCheckboxes(stemPaths) {
  const container = document.getElementById('midi-stems');
  clearChildren(container);
  for (const label of Object.keys(stemPaths)) {
    container.appendChild(
      el('label', {},
        el('input', { type: 'checkbox', value: label, checked: 'true' }),
        label,
      ),
    );
  }
  syncDrumHint();
}

function syncDrumHint() {
  const checked = document.querySelectorAll('#midi-stems input[type="checkbox"]:checked');
  const hasDrum = Array.from(checked).some(cb => isDrumStem(cb.value));
  document.getElementById('midi-drum-hint').classList.toggle('hidden', !hasDrum);
  syncDrumLicenseBanner(hasDrum);
}

/** Show the ADTOF license warning banner when a drum stem is checked. */
function syncDrumLicenseBanner(hasDrum) {
  const existing = document.getElementById('midi-drum-license');
  if (existing) existing.remove();
  if (!hasDrum || !drumLicenseWarning) return;

  const hint = document.getElementById('midi-drum-hint');

  // Already acknowledged this session — show a brief reminder only
  if (drumLicenseAcknowledged) {
    hint.after(el('div', {
      className: 'banner banner-warn', id: 'midi-drum-license',
    }, 'License warning acknowledged. Drum transcription weights are non-commercial (CC BY-NC-SA 4.0).'));
    return;
  }

  const ackBtn = el('button', { className: 'btn btn-sm' }, 'I understand — proceed');
  const banner = el('div', {
    className: 'banner banner-warn', id: 'midi-drum-license',
  },
    el('strong', {}, 'License warning: '),
    drumLicenseWarning,
    el('div', { style: 'margin-top: 0.5rem' }, ackBtn),
  );
  ackBtn.addEventListener('click', () => {
    drumLicenseAcknowledged = true;
    syncDrumHint();
  });
  hint.after(banner);
}

/** Returns true if a drum stem is checked but the license is unacknowledged. */
function isDrumLicenseBlocked() {
  const checked = document.querySelectorAll('#midi-stems input[type="checkbox"]:checked');
  const hasDrum = Array.from(checked).some(cb => isDrumStem(cb.value));
  return hasDrum && !!drumLicenseWarning && !drumLicenseAcknowledged;
}

async function startExtraction() {
  const stemEls = document.querySelectorAll('#midi-stems input[type="checkbox"]:checked');
  const stems = Array.from(stemEls).map(e => e.value);
  if (!stems.length) return;
  if (isDrumLicenseBlocked()) return;  // banner above the stem list explains

  const progressCard = document.getElementById('midi-progress');
  const resultsContainer = document.getElementById('midi-results');
  progressCard.classList.remove('hidden');
  clearChildren(resultsContainer);
  midiPlayers.length = 0;
  document.getElementById('midi-start').disabled = true;

  try {
    const { job_id } = await api('/midi/extract', {
      method: 'POST',
      body: JSON.stringify({
        stems,
        key: document.getElementById('midi-key').value,
        bpm: parseFloat(document.getElementById('midi-bpm').value),
        time_signature: document.getElementById('midi-ts').value,
        onset_threshold: parseFloat(document.getElementById('midi-onset').value),
        frame_threshold: parseFloat(document.getElementById('midi-frame').value),
      }),
    });

    pollJob(job_id, {
      onProgress(progress, stage) {
        document.getElementById('midi-progress-fill').style.width = `${(progress * 100).toFixed(0)}%`;
        document.getElementById('midi-pct').textContent = `${(progress * 100).toFixed(0)}%`;
        document.getElementById('midi-stage').textContent = stage;
      },
      onDone(result) {
        progressCard.classList.add('hidden');
        document.getElementById('midi-start').disabled = false;
        showMidiResults(result);
      },
      onError(msg) {
        progressCard.classList.add('hidden');
        document.getElementById('midi-start').disabled = false;
        resultsContainer.appendChild(
          el('div', { className: 'banner banner-error' }, `MIDI extraction failed: ${msg}`),
        );
      },
    });
  } catch (err) {
    progressCard.classList.add('hidden');
    document.getElementById('midi-start').disabled = false;
    resultsContainer.appendChild(
      el('div', { className: 'banner banner-error' }, `Error: ${err.message}`),
    );
  }
}

async function handleMidiImport() {
  const fileInput = document.getElementById('midi-import-input');
  const file = fileInput.files[0];
  fileInput.value = '';
  if (!file) return;

  const importBtn = document.getElementById('midi-import');
  importBtn.disabled = true;
  importBtn.textContent = 'Importing...';

  try {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/midi/import', { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();

    // Build a result card for the imported MIDI
    buildMidiCard(data.label, { note_count: data.note_count });
    if (!appState.midiLabels.includes(data.label)) appState.midiLabels.push(data.label);
    appState.emit('midiReady', { labels: [data.label], stem_info: { [data.label]: { note_count: data.note_count } } });
  } catch (err) {
    alert(`MIDI import failed: ${err.message}`);
  } finally {
    importBtn.textContent = 'Import MIDI file';
    importBtn.disabled = false;
  }
}

function showMidiResults(result) {
  const container = document.getElementById('midi-results');
  midiPlayers.length = 0;
  // Old cards are about to be superseded — silence their hardware sends.
  for (const entry of _outSchedulers) entry.stopPlayback();
  _outSchedulers.length = 0;

  appState.midiLabels = result.labels || [];
  appState.midiHasMerged = !!result.has_merged;
  appState.emit('midiReady', result);

  // Merged MIDI buttons
  if (result.has_merged) {
    const mergedRow = el('div', { className: 'midi-merged-row' });

    mergedRow.appendChild(
      el('button', {
        className: 'btn',
        onClick: async () => {
          try {
            const res = await api('/midi/save', {
              method: 'POST',
              body: JSON.stringify({ label: 'merged' }),
            });
            alert(`Saved: ${res.path}`);
          } catch (err) {
            alert(`Save failed: ${err.message}`);
          }
        },
      }, 'Save merged MIDI'),
    );

    // Clean Up All merged
    const cleanAllBtn = el('button', { className: 'btn btn-sm' }, 'Clean Up All');
    cleanAllBtn.addEventListener('click', async () => {
      cleanAllBtn.disabled = true;
      cleanAllBtn.textContent = 'Cleaning...';
      try {
        const key = document.getElementById('midi-key').value;
        const ts = document.getElementById('midi-ts').value;
        await api('/midi/clean', {
          method: 'POST',
          body: JSON.stringify({
            stem_label: 'merged',
            key: key !== 'Any' ? key : null,
            time_signature: ts,
          }),
        });
        cleanAllBtn.textContent = 'Cleaned \u2713';
        setTimeout(() => { cleanAllBtn.textContent = 'Clean Up All'; cleanAllBtn.disabled = false; }, 2000);
      } catch (err) {
        alert(`Clean failed: ${err.message}`);
        cleanAllBtn.textContent = 'Clean Up All';
        cleanAllBtn.disabled = false;
      }
    });
    mergedRow.appendChild(cleanAllBtn);

    // Sheet Music (All)
    const sheetAllBtn = el('button', { className: 'btn btn-sm' }, 'Sheet Music (All)');
    sheetAllBtn.addEventListener('click', async () => {
      sheetAllBtn.disabled = true;
      sheetAllBtn.textContent = 'Loading...';
      try {
        const res = await api('/midi/sheet-music', {
          method: 'POST',
          body: JSON.stringify({ stem_label: 'merged', title: 'All Stems (Merged)' }),
        });
        showSheetMusicPanel(container, res.musicxml, 'merged');
        sheetAllBtn.textContent = 'Sheet Music (All)';
        sheetAllBtn.disabled = false;
      } catch (err) {
        alert(`Sheet music failed: ${err.message}`);
        sheetAllBtn.textContent = 'Sheet Music (All)';
        sheetAllBtn.disabled = false;
      }
    });
    mergedRow.appendChild(sheetAllBtn);

    container.appendChild(mergedRow);
    // The merged MIDI gets hardware output too — same visibility rule as
    // the Save controls. All instruments play on the row's single
    // port/channel; per-instrument routing is deferred. It is the one row
    // that keeps its own Send/Stop: there is no merged card, so it has no
    // transport to borrow and no waveform to follow.
    container.appendChild(buildMidiOutRow('merged', { standalone: true }).row);
  }

  // Per-stem result cards with full playback
  for (const [label, info] of Object.entries(result.stem_info || {})) {
    buildMidiCard(label, info);
  }
}

/**
 * Build the left-column MIDI Out section: availability banner, Request
 * access button, Panic button. Web MIDI availability is a browser
 * property — detection is entirely client-side, the server cannot know.
 */
function buildMidiOutSection() {
  const banner = el('div', { className: 'banner banner-info', id: 'midi-out-banner' });
  const requestBtn = el('button', { className: 'btn btn-sm', id: 'midi-out-request' }, 'Request access');
  const panicBtn = el('button', {
    className: 'btn btn-sm btn-panic', id: 'midi-out-panic', disabled: 'true',
    title: 'Silence every known MIDI output on all channels',
  }, '⏻ Panic');
  const actions = el('div', { className: 'midi-out-actions' }, requestBtn, panicBtn);
  const section = el('div', { className: 'form-group midi-out-section' },
    el('label', {}, 'MIDI Out'), banner, actions,
  );

  function setBanner(cls, text) {
    banner.className = `banner ${cls}`;
    banner.textContent = text;
  }

  function refreshPorts() {
    const outputs = getOutputs();
    if (outputs.length === 0) {
      setBanner('banner-warn', 'No MIDI outputs found. Connect an interface — the list updates automatically.');
    } else {
      setBanner('banner-success', `✓ ${outputs.length} output port${outputs.length === 1 ? '' : 's'} found`);
    }
    for (const cb of midiOutRefreshers) {
      try { cb(outputs); } catch (err) { console.error('MIDI Out refresh failed', err); }
    }
  }

  async function grantAccess() {
    try {
      await initWebMidi();
    } catch (err) {
      setBanner('banner-error', 'MIDI access was denied. Click Request access and allow the prompt.');
      return;
    }
    midiOutReady = true;
    // MIDIAccess.outputs is a live map kept current by statechange —
    // once granted there is nothing to rescan, so the button goes away.
    requestBtn.classList.add('hidden');
    panicBtn.removeAttribute('disabled');
    onPortsChanged(refreshPorts);
    refreshPorts();
  }

  if (!window.isSecureContext) {
    setBanner('banner-error', 'Web MIDI requires HTTPS or localhost. Open StemForge at http://localhost:8765.');
    actions.classList.add('hidden');
  } else if (!webMidiSupported()) {
    setBanner('banner-error', 'Web MIDI is not available in this browser. Chrome or Edge is required.');
    actions.classList.add('hidden');
  } else {
    setBanner('banner-info', 'Click Request access to enable hardware MIDI output.');
    requestBtn.addEventListener('click', grantAccess);
    panicBtn.addEventListener('click', panicAll);
    // If permission was already granted, initialize silently — query first
    // so first-time visitors are not hit with a prompt on tab load.
    navigator.permissions?.query({ name: 'midi', sysex: false })
      .then((status) => { if (status.state === 'granted') grantAccess(); })
      .catch(() => {});
  }

  return section;
}

/**
 * Build a per-card MIDI Out row — routing only. The card's own
 * Play/Stop/Rewind drive whichever destination `portSelect` names, so the
 * row deliberately carries no transport buttons (spec rev. 3): with two
 * sets of buttons a user could sound the soft synth and the hardware at
 * once, which is never wanted.
 *
 * opts.getProgram: () => GM program 0-127, or null when nothing sendable
 *   is selected (Drum Kit).
 * opts.standalone: render the row's own Send/Stop pair. Used only by the
 *   merged row, which has no card and therefore no transport to borrow.
 *
 * Returns a controller the card wires its transport to.
 */
function buildMidiOutRow(label, opts = {}) {
  const { getProgram = null, standalone = false } = opts;

  const portSelect = el('select', { className: 'midi-out-select midi-out-port' });
  const channelSelect = el('select', { className: 'midi-out-select' });
  for (let c = 1; c <= 16; c++) {
    channelSelect.appendChild(el('option', { value: String(c) }, `Ch ${c}`));
  }
  // Off by default: the instrument dropdown selects a FluidSynth patch, and
  // silently overwriting whatever the user dialled in on their synth would
  // be hostile. Drum stems never send PC even when checked — GM percussion
  // is a channel convention, not a program.
  const progChangeBox = el('input', { type: 'checkbox', className: 'midi-out-pc' });
  const progChangeLabel = el('label', { className: 'text-dim midi-out-pc-label' },
    progChangeBox, 'Send Program Change');
  const outStatus = el('span', { className: 'midi-out-status text-dim' });

  const sendBtn = standalone
    ? el('button', { className: 'btn btn-sm', disabled: 'true' }, '▶ Send') : null;
  const outStopBtn = standalone
    ? el('button', { className: 'btn btn-sm', disabled: 'true' }, '■ Stop') : null;

  const row = el('div', { className: 'midi-out-row' },
    el('label', { className: 'text-dim' }, 'MIDI Out:'),
    portSelect, channelSelect, sendBtn, outStopBtn, progChangeLabel, outStatus,
  );
  if (!getProgram) progChangeLabel.classList.add('hidden');

  const ctl = {
    row,
    isHardware: () => false,
    isPlaying: () => false,
    start: async () => false,
    stop: () => {},
    invalidate: () => {},
    onRoutingChange: () => {},
    onFinish: () => {},
    onError: () => {},
  };

  // A dead control is worse than no control. With Web MIDI unavailable the
  // card behaves exactly as it did before this feature existed.
  if (!window.isSecureContext || !webMidiSupported()) {
    row.classList.add('hidden');
    return ctl;
  }

  let sched = null;
  let cache = null;      // {events, duration, truncated, isDrum} — cleared on edit
  let routingCb = null;
  let finishCb = null;
  let errorCb = null;

  // Saved preference for this stem — the channel applies immediately, the
  // port by name-match once ports are known. Auto-restore stops as soon as
  // the user touches the port dropdown, so an explicit SoftSynth sticks.
  const pref = loadMidiOutPrefs()[label] || null;
  let autoRestorePort = !!(pref && pref.port);
  if (pref && pref.channel >= 1 && pref.channel <= 16) {
    channelSelect.value = String(pref.channel);
  }

  /** Channel and Program Change mean nothing to the FluidSynth render. */
  function syncRoutingState() {
    const hw = !!portSelect.value;
    channelSelect.disabled = !hw;
    progChangeBox.disabled = !hw;
    progChangeLabel.classList.toggle('text-disabled', !hw);
    if (sendBtn) {
      if (midiOutReady && hw) sendBtn.removeAttribute('disabled');
      else sendBtn.setAttribute('disabled', 'true');
    }
  }

  function refreshPortOptions(outputs) {
    const prev = portSelect.value;
    clearChildren(portSelect);
    // Not "None" — this entry routes to FluidSynth, a real destination.
    portSelect.appendChild(el('option', { value: '' },
      standalone ? 'None' : 'SoftSynth'));
    for (const o of outputs) {
      portSelect.appendChild(el('option', { value: o.id, title: o.name }, o.name));
    }
    if (Array.from(portSelect.options).some(op => op.value === prev && prev !== '')) {
      portSelect.value = prev;
    } else if (autoRestorePort) {
      const match = outputs.find(o => o.name === pref.port);  // first match wins
      portSelect.value = match ? match.id : '';
    } else {
      portSelect.value = '';
    }
    syncRoutingState();
    if (routingCb) routingCb();
  }

  function savePref() {
    const selected = portSelect.selectedOptions[0];
    saveMidiOutPref(label,
      portSelect.value ? selected.textContent : null,
      parseInt(channelSelect.value, 10));
  }

  function stopPlayback(msg = '') {
    if (sched) sched.stop();
    sched = null;
    entry.sched = null;
    if (outStopBtn) outStopBtn.setAttribute('disabled', 'true');
    outStatus.textContent = msg;
    syncRoutingState();
  }

  const entry = { sched: null, stopPlayback };
  _outSchedulers.push(entry);

  /** Fetch + flatten the event stream, cached until the MIDI is edited. */
  async function loadEvents() {
    if (cache) return cache;
    const data = await api('/midi/events', {
      method: 'POST',
      body: JSON.stringify({ stem_label: label }),
    });
    // Merge all tracks into one flat array, re-sorted with the backend's
    // comparator (ascending t, off before on at equal t) — per-track
    // ordering does not survive concatenation, and losing the tie-break
    // would break the abutting-note guarantee.
    const events = [];
    for (const track of data.tracks) events.push(...track.events);
    events.sort((a, b) =>
      a.t - b.t || (a.type === b.type ? 0 : a.type === 'off' ? -1 : 1));
    cache = {
      events,
      duration: data.duration,
      truncated: data.truncated,
      isDrum: data.tracks.some((tr) => tr.is_drum),
    };
    return cache;
  }

  async function start(startSec = 0) {
    const portId = portSelect.value;
    const channel = parseInt(channelSelect.value, 10);
    if (!portId) return false;

    if (sendBtn) sendBtn.setAttribute('disabled', 'true');
    let data;
    try {
      data = await loadEvents();
    } catch (err) {
      stopPlayback(`Failed: ${err.message}`);
      return false;
    }

    // Exclusivity: stop other schedulers on the same port AND channel only.
    for (const other of _outSchedulers) {
      if (other !== entry && other.sched && other.sched.isPlaying()
          && other.sched.outputId === portId && other.sched.channel === channel) {
        other.stopPlayback();
      }
    }
    stopPlayback();  // our own previous run, whatever port it was on

    try {
      sched = createScheduler(portId, channel);
    } catch (err) {
      stopPlayback('Port unavailable');
      return false;
    }
    entry.sched = sched;

    // Program Change: only when checked, and suppressed for drum stems —
    // keyed off the stem label / track flags, never off a dropdown value,
    // and never forcing channel 10 (the channel is the user's choice).
    let program = null;
    if (progChangeBox.checked && getProgram
        && !isDrumStem(label) && !data.isDrum) {
      program = getProgram();
    }

    outStatus.textContent = data.truncated ? 'Truncated — run Clean Up' : '';
    sched.onFinish(() => {
      sched = null;
      entry.sched = null;
      stopPlayback(outStatus.textContent);
      if (finishCb) finishCb();
    });
    sched.onError(() => {
      sched = null;
      entry.sched = null;
      stopPlayback('Port disconnected');
      if (errorCb) errorCb();
    });
    sched.play(data.events, { program, startSec });
    if (outStopBtn) outStopBtn.removeAttribute('disabled');
    return true;
  }

  if (sendBtn) sendBtn.addEventListener('click', () => start(0));
  if (outStopBtn) outStopBtn.addEventListener('click', () => stopPlayback());

  // Port/channel changes take effect on the next Send — the scheduler's
  // binding is fixed at creation, deliberately. Switching routing while
  // something is playing stops it, so neither destination is left hanging.
  portSelect.addEventListener('change', () => {
    autoRestorePort = false;
    stopPlayback();
    savePref();
    syncRoutingState();
    if (routingCb) routingCb();
  });
  channelSelect.addEventListener('change', () => {
    stopPlayback();
    savePref();
  });

  midiOutRefreshers.push(refreshPortOptions);
  refreshPortOptions(getOutputs());

  Object.assign(ctl, {
    isHardware: () => !!portSelect.value,
    isPlaying: () => !!(sched && sched.isPlaying()),
    start,
    stop: () => stopPlayback(),
    invalidate: () => { cache = null; },
    onRoutingChange: (cb) => { routingCb = cb; },
    onFinish: (cb) => { finishCb = cb; },
    onError: (cb) => { errorCb = cb; },
  });
  return ctl;
}

/**
 * Build a MIDI result card with waveform, playback controls, and instrument selector.
 * Mirrors the stem cards in the Separate tab.
 */
function buildMidiCard(label, info) {
  const container = document.getElementById('midi-results');
  const card = el('div', { className: 'stem-card' });

  // ─── Instrument selector ───
  const defaultProgram = getDefaultProgram(label);
  const defaultIsDrum = isDrumStem(label);

  const instrumentSelect = el('select', { className: 'midi-instrument-select' });
  // Add drum kit option at top
  instrumentSelect.appendChild(el('option', { value: 'drum' }, 'Drum Kit'));
  for (let i = 0; i < gmPrograms.length; i++) {
    instrumentSelect.appendChild(el('option', { value: String(i) }, `${i}: ${gmPrograms[i]}`));
  }
  // Set default
  if (defaultIsDrum) {
    instrumentSelect.value = 'drum';
  } else {
    instrumentSelect.value = String(defaultProgram);
  }

  // ─── Transport buttons ───
  const playBtn = el('button', { className: 'btn btn-sm' }, '\u25B6 Play');
  const stopBtn = el('button', { className: 'btn btn-sm' }, '\u25A0 Stop');
  const rewindBtn = el('button', { className: 'btn btn-sm' }, '\u23EA Rewind');
  const timeLabel = el('span', { className: 'stem-time' }, '0:00 / 0:00');

  const saveBtn = el('button', {
    className: 'btn btn-sm',
    onClick: async () => {
      try {
        const res = await api('/midi/save', {
          method: 'POST',
          body: JSON.stringify({ label }),
        });
        alert(`Saved: ${res.path}`);
      } catch (err) {
        alert(`Save failed: ${err.message}`);
      }
    },
  }, '\u2193 Save');

  const header = el('div', { className: 'stem-card-header' },
    el('span', { className: 'stem-label' }, `${label} (${info.note_count} notes)`),
    el('div', { className: 'stem-actions' },
      playBtn, stopBtn, rewindBtn, timeLabel, saveBtn,
    ),
  );

  // Instrument row
  const instrumentRow = el('div', { className: 'midi-instrument-row' },
    el('label', { className: 'text-dim' }, 'Instrument:'),
    instrumentSelect,
  );

  // MIDI Out row — routing only; the card's transport below drives it.
  const midiOut = buildMidiOutRow(label, {
    getProgram: () => {
      const val = instrumentSelect.value;
      return val === 'drum' ? null : parseInt(val, 10);
    },
  });
  const midiOutRow = midiOut.row;

  // Waveform container (initially empty — populated on first render)
  const waveContainer = el('div', { className: 'stem-waveform' });
  const renderHint = el('div', { className: 'midi-render-hint text-dim' }, 'Press Play to render audio preview');

  // ─── MIDI Tools row ───
  const noteCountLabel = header.querySelector('.stem-label');
  let transposeOffset = 0;

  const cleanBtn = el('button', { className: 'btn btn-sm' }, 'Clean Up');
  const detectKeyBtn = el('button', { className: 'btn btn-sm' }, 'Detect Key');
  const keyInfoSpan = el('span', { className: 'midi-key-info text-dim' });

  // Transpose controls — mode selector + [-] [+] buttons
  const transMode = el('select', { className: 'midi-sheet-select' },
    el('option', { value: '1' }, 'Semitone'),
    el('option', { value: 'm2' }, 'Minor 2nd'),
    el('option', { value: 'M2' }, 'Major 2nd'),
    el('option', { value: 'm3' }, 'Minor 3rd'),
    el('option', { value: 'M3' }, 'Major 3rd'),
    el('option', { value: 'P4' }, 'Perfect 4th'),
    el('option', { value: 'A4' }, 'Tritone'),
    el('option', { value: 'P5' }, 'Perfect 5th'),
    el('option', { value: 'm6' }, 'Minor 6th'),
    el('option', { value: 'M6' }, 'Major 6th'),
    el('option', { value: 'm7' }, 'Minor 7th'),
    el('option', { value: 'M7' }, 'Major 7th'),
    el('option', { value: 'P8' }, 'Octave'),
  );
  const transposeLabel = el('span', { className: 'midi-transpose-label text-dim' }, '0');
  const transDown = el('button', { className: 'btn btn-sm' }, '\u2212');
  const transUp = el('button', { className: 'btn btn-sm' }, '+');
  const transposeControls = el('div', { className: 'midi-transpose-controls' },
    transMode, transDown, transposeLabel, transUp,
  );

  // Sheet Music dropdown
  const sheetSelect = el('select', { className: 'btn btn-sm midi-sheet-select' },
    el('option', { value: '' }, 'Sheet Music...'),
    el('option', { value: 'preview' }, 'Preview'),
    el('option', { value: 'musicxml' }, 'Download MusicXML'),
  );
  if (_lilypondAvailable) {
    sheetSelect.appendChild(el('option', { value: 'pdf' }, 'Download PDF'));
  }

  // Save XML button
  const saveXmlBtn = el('button', { className: 'btn btn-sm' }, 'Save XML');

  const toolsRow = el('div', { className: 'midi-tools-row' },
    cleanBtn, detectKeyBtn, keyInfoSpan,
    el('span', { className: 'text-dim' }, 'Transpose:'), transposeControls,
    sheetSelect, saveXmlBtn,
  );

  // Sheet music panel placeholder
  const sheetPanel = el('div', { className: 'sheet-music-panel hidden' });

  card.append(header, instrumentRow, midiOutRow, waveContainer, renderHint, toolsRow, sheetPanel);
  container.appendChild(card);

  // ─── MIDI Tools event handlers ───

  cleanBtn.addEventListener('click', async () => {
    cleanBtn.disabled = true;
    cleanBtn.textContent = 'Cleaning...';
    try {
      const key = document.getElementById('midi-key').value;
      const ts = document.getElementById('midi-ts').value;
      const res = await api('/midi/clean', {
        method: 'POST',
        body: JSON.stringify({
          stem_label: label,
          key: key !== 'Any' ? key : null,
          time_signature: ts,
        }),
      });
      noteCountLabel.textContent = `${label} (${res.note_count} notes)`;
      cleanBtn.textContent = 'Cleaned \u2713';
      transposeOffset = 0;
      transposeLabel.textContent = '0';
      // Re-render waveform with cleaned MIDI
      renderedUrl = null;
      midiOut.invalidate();
      renderAndLoad(false);
      setTimeout(() => { cleanBtn.textContent = 'Clean Up'; cleanBtn.disabled = false; }, 2000);
    } catch (err) {
      alert(`Clean failed: ${err.message}`);
      cleanBtn.textContent = 'Clean Up';
      cleanBtn.disabled = false;
    }
  });

  detectKeyBtn.addEventListener('click', async () => {
    detectKeyBtn.disabled = true;
    detectKeyBtn.textContent = 'Detecting...';
    try {
      const res = await api('/midi/detect-key', {
        method: 'POST',
        body: JSON.stringify({ stem_label: label }),
      });
      const pct = Math.round(res.confidence * 100);
      keyInfoSpan.textContent = `Detected: ${res.key} (${pct}%)`;
      // Update the global key selector
      const keySelect = document.getElementById('midi-key');
      const matchOption = Array.from(keySelect.options).find(o => o.value === res.key);
      if (matchOption) keySelect.value = res.key;
      detectKeyBtn.textContent = 'Detect Key';
      detectKeyBtn.disabled = false;
    } catch (err) {
      alert(`Detection failed: ${err.message}`);
      detectKeyBtn.textContent = 'Detect Key';
      detectKeyBtn.disabled = false;
    }
  });

  // Semitone equivalents for offset tracking display
  const _intervalSemitones = {
    '1': 1, 'm2': 1, 'M2': 2, 'm3': 3, 'M3': 4,
    'P4': 5, 'A4': 6, 'P5': 7, 'm6': 8, 'M6': 9,
    'm7': 10, 'M7': 11, 'P8': 12,
  };

  async function doTranspose(direction) {
    transDown.disabled = true;
    transUp.disabled = true;
    const mode = transMode.value;
    const body = { stem_label: label };

    if (mode === '1') {
      // Semitone mode — send raw semitones
      body.semitones = direction;
    } else {
      // Named interval mode — prefix with - for down
      body.interval = direction < 0 ? `-${mode}` : mode;
    }

    try {
      const res = await api('/midi/transpose', {
        method: 'POST',
        body: JSON.stringify(body),
      });
      transposeOffset += direction * (_intervalSemitones[mode] || 1);
      transposeLabel.textContent = transposeOffset > 0 ? `+${transposeOffset}` : String(transposeOffset);
      noteCountLabel.textContent = `${label} (${res.note_count} notes)`;
      // Re-render waveform
      renderedUrl = null;
      midiOut.invalidate();
      renderAndLoad(false);
    } catch (err) {
      alert(`Transpose failed: ${err.message}`);
    } finally {
      transDown.disabled = false;
      transUp.disabled = false;
    }
  }

  transDown.addEventListener('click', () => doTranspose(-1));
  transUp.addEventListener('click', () => doTranspose(1));

  sheetSelect.addEventListener('change', async () => {
    const action = sheetSelect.value;
    sheetSelect.value = '';
    if (!action) return;

    if (action === 'preview') {
      sheetSelect.disabled = true;
      try {
        const res = await api('/midi/sheet-music', {
          method: 'POST',
          body: JSON.stringify({ stem_label: label, title: label }),
        });
        showSheetMusicPanel(sheetPanel, res.musicxml, label);
      } catch (err) {
        alert(`Sheet music failed: ${err.message}`);
      } finally {
        sheetSelect.disabled = false;
      }
    } else if (action === 'pdf') {
      try {
        const resp = await fetch('/api/midi/sheet-music/pdf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stem_label: label, title: label }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const blob = await resp.blob();
        saveFileAs(blob, `${label}_sheet_music.pdf`);
      } catch (err) {
        alert(`PDF export failed: ${err.message}`);
      }
    } else if (action === 'musicxml') {
      try {
        const resp = await fetch('/api/midi/sheet-music/musicxml', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stem_label: label, title: label }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const blob = await resp.blob();
        saveFileAs(blob, `${label}.musicxml`);
      } catch (err) {
        alert(`MusicXML export failed: ${err.message}`);
      }
    }
  });

  saveXmlBtn.addEventListener('click', async () => {
    try {
      const resp = await fetch('/api/midi/sheet-music/musicxml', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stem_label: label, title: label }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const blob = await resp.blob();
      saveFileAs(blob, `${label}.musicxml`);
    } catch (err) {
      alert(`Save MusicXML failed: ${err.message}`);
    }
  });

  // State for this card
  let ws = null;
  let renderedUrl = null;
  let lastProgram = instrumentSelect.value;

  /** Ensure wavesurfer instance exists. */
  function ensureWaveform() {
    if (ws) return;
    ws = createWaveform(waveContainer, { height: 50, color: 'midi' });
    midiPlayers.push({ ws, playBtn });

    ws.on('timeupdate', (time) => {
      const dur = ws.getDuration();
      timeLabel.textContent = `${formatTime(time)} / ${formatTime(dur)}`;
    });

    ws.on('finish', () => {
      // Under hardware routing this is the muted cursor reaching the end of
      // the FluidSynth render, which is only approximately the MIDI's end —
      // it must not be treated as the hardware run finishing.
      if (!midiOut.isHardware()) transportStop();
      updateTransportLabel();
    });

    ws.on('error', () => {
      playBtn.disabled = false;
      updateTransportLabel();
    });

    // Click-to-seek. 'interaction' fires only for user clicks, never for
    // our own setTime() calls, so restarting here cannot recurse.
    ws.on('interaction', (newTime) => {
      if (!midiOut.isHardware() || !midiOut.isPlaying()) return;
      hardwareStart(newTime);
    });
  }

  /** Render MIDI to audio with current instrument, then load into waveform. */
  async function renderAndLoad(autoplay) {
    const val = instrumentSelect.value;
    const isDrum = val === 'drum';
    const program = isDrum ? 0 : parseInt(val, 10);

    playBtn.disabled = true;
    playBtn.textContent = 'Rendering...';

    try {
      const res = await api('/midi/render', {
        method: 'POST',
        body: JSON.stringify({ stem_label: label, program, is_drum: isDrum }),
      });
      renderedUrl = `/api/audio/stream?path=${encodeURIComponent(res.audio_path)}`;
      lastProgram = val;

      // Hide hint
      renderHint.classList.add('hidden');

      ensureWaveform();
      ws.load(renderedUrl);

      if (autoplay) {
        ws.once('ready', () => {
          playBtn.disabled = false;
          stopOtherPlayers(ws);
          ws.play();
          playBtn.textContent = '\u23F8 Pause';
          transportLoad(renderedUrl, label, false, 'MIDI', { cardWs: ws });
        });
      } else {
        ws.once('ready', () => {
          playBtn.disabled = false;
          playBtn.textContent = '\u25B6 Play';
        });
      }
    } catch (err) {
      alert(`Render failed: ${err.message}`);
      playBtn.textContent = '\u25B6 Play';
      playBtn.disabled = false;
    }
  }

  // ─── Transport ───
  // One button, one destination. The MIDI Out dropdown decides whether it
  // drives the FluidSynth render or a hardware port; the two can never
  // sound together (spec rev. 3).

  /** Label the transport button for the current routing and play state. */
  function updateTransportLabel() {
    if (midiOut.isHardware()) {
      playBtn.textContent = midiOut.isPlaying() ? '⏸ Pause' : '▶ Send';
    } else {
      playBtn.textContent = (ws && ws.isPlaying()) ? '⏸ Pause' : '▶ Play';
    }
  }

  midiOut.onRoutingChange(() => {
    // The row already stopped any hardware run; stop the browser side too
    // so neither destination is left sounding, and reset the mute state.
    if (ws) {
      if (ws.isPlaying()) ws.pause();
      ws.setVolume(midiOut.isHardware() ? 0 : 1);
    }
    transportStop();
    updateTransportLabel();
  });
  midiOut.onFinish(() => {
    if (ws) { ws.pause(); ws.setTime(0); }
    updateTransportLabel();
  });
  midiOut.onError(() => {
    if (ws) ws.pause();
    updateTransportLabel();
  });

  /**
   * Start hardware playback with the waveform running alongside it, muted,
   * so the cursor moves and click-to-seek keeps working. The waveform
   * starts LEAD_MS late because the scheduler's origin includes that lead.
   */
  async function hardwareStart(startSec) {
    const ok = await midiOut.start(startSec);
    updateTransportLabel();
    if (!ok || !ws) return;
    ws.setVolume(0);
    ws.setTime(startSec);
    setTimeout(() => {
      if (midiOut.isPlaying() && ws && !ws.isPlaying()) ws.play();
    }, LEAD_MS);
  }

  function hardwareStop(rewind) {
    midiOut.stop();
    if (ws) {
      ws.pause();
      if (rewind) ws.setTime(0);
    }
    updateTransportLabel();
  }

  playBtn.addEventListener('click', async () => {
    if (midiOut.isHardware()) {
      // Pause holds position, so the next press resumes from the cursor.
      if (midiOut.isPlaying()) hardwareStop(false);
      else await hardwareStart(ws ? ws.getCurrentTime() : 0);
      return;
    }

    const needsRender = !renderedUrl || instrumentSelect.value !== lastProgram;
    if (needsRender) {
      renderAndLoad(true);
      return;
    }

    if (ws && ws.isPlaying()) {
      ws.pause();
      playBtn.textContent = '▶ Play';
    } else if (ws) {
      stopOtherPlayers(ws);
      ws.setVolume(1);
      ws.play();
      playBtn.textContent = '⏸ Pause';
      transportLoad(renderedUrl, label, false, 'MIDI', { cardWs: ws });
    }
  });

  // Stop
  stopBtn.addEventListener('click', () => {
    if (midiOut.isHardware()) {
      hardwareStop(true);
      return;
    }
    if (ws) {
      ws.stop();
      transportStop();
      playBtn.textContent = '▶ Play';
    }
  });

  // Rewind
  rewindBtn.addEventListener('click', () => {
    if (!ws) return;
    ws.setTime(0);
    // Restart from the top rather than leaving the synth playing from the
    // old position with the cursor sitting at zero.
    if (midiOut.isHardware() && midiOut.isPlaying()) hardwareStart(0);
  });

  // Re-render when instrument changes and audio was already rendered;
  // also sync the instrument to the corresponding Mix track.
  instrumentSelect.addEventListener('change', () => {
    const val = instrumentSelect.value;
    const isDrum = val === 'drum';
    const program = isDrum ? 0 : parseInt(val, 10);

    // Update Mix track
    const trackId = `midi-${label}`;
    api('/mix/tracks', {
      method: 'POST',
      body: JSON.stringify({ track_id: trackId, program, is_drum: isDrum }),
    }).then(() => {
      appState.emit('midiInstrumentChanged', { label, program, is_drum: isDrum });
    }).catch(() => { /* track may not exist yet */ });

    if (renderedUrl) {
      renderAndLoad(false);
    }
  });

  // A saved preference may already have restored a hardware port before the
  // routing callback above was wired, so set the button label once now.
  updateTransportLabel();

  // Auto-render waveform on card creation (no autoplay)
  renderAndLoad(false);
}

/** Get the default GM program for a stem label. */
function getDefaultProgram(label) {
  const lower = label.toLowerCase();
  for (const [key, prog] of Object.entries(stemDefaults)) {
    if (lower.includes(key)) return prog;
  }
  return 0;
}

/** Check if a stem label should default to drum kit. */
function isDrumStem(label) {
  const lower = label.toLowerCase();
  for (const key of Object.keys(drumStems)) {
    if (lower.includes(key.toLowerCase())) return true;
  }
  return false;
}

/**
 * Show OSMD-rendered sheet music in a collapsible panel.
 * @param {HTMLElement} panel - the .sheet-music-panel container
 * @param {string} musicxml - MusicXML string
 * @param {string} label - stem label for context
 */
async function showSheetMusicPanel(panel, musicxml, label) {
  clearChildren(panel);
  panel.classList.remove('hidden');

  const OSMD = window.opensheetmusicdisplay?.OpenSheetMusicDisplay;
  if (!OSMD) {
    panel.appendChild(el('div', { className: 'banner banner-error' },
      'OpenSheetMusicDisplay not loaded. Check your internet connection.'));
    return;
  }

  const details = el('details', { open: true },
    el('summary', {}, `Sheet Music: ${label}`),
  );
  const renderTarget = el('div', { className: 'sheet-music-container' });
  details.appendChild(renderTarget);

  // Download buttons below the notation
  const downloadRow = el('div', { className: 'midi-tools-row' });
  const dlXmlBtn = el('button', { className: 'btn btn-sm' }, 'Download MusicXML');
  dlXmlBtn.addEventListener('click', async () => {
    try {
      const resp = await fetch('/api/midi/sheet-music/musicxml', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stem_label: label, title: label }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const blob = await resp.blob();
      saveFileAs(blob, `${label}.musicxml`);
    } catch (err) { alert(`Download failed: ${err.message}`); }
  });
  downloadRow.appendChild(dlXmlBtn);

  if (_lilypondAvailable) {
    const dlPdfBtn = el('button', { className: 'btn btn-sm' }, 'Download PDF');
    dlPdfBtn.addEventListener('click', async () => {
      try {
        const resp = await fetch('/api/midi/sheet-music/pdf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stem_label: label, title: label }),
        });
        if (!resp.ok) throw new Error(await resp.text());
        const blob = await resp.blob();
        saveFileAs(blob, `${label}_sheet_music.pdf`);
      } catch (err) { alert(`PDF download failed: ${err.message}`); }
    });
    downloadRow.appendChild(dlPdfBtn);
  }

  details.appendChild(downloadRow);
  panel.appendChild(details);

  // Render with OSMD
  try {
    const osmd = new OSMD(renderTarget, {
      autoResize: true,
      drawTitle: true,
    });
    await osmd.load(musicxml);
    osmd.render();
  } catch (err) {
    renderTarget.appendChild(el('div', { className: 'banner banner-error' },
      `Notation render failed: ${err.message}`));
  }
}

// ─── Lyrics mode ────────────────────────────────────────────────────

let _midiMode = 'notes';
let _lyricsEngines = [];

function switchMidiMode(mode) {
  if (mode === _midiMode) return;
  _midiMode = mode;

  document.querySelectorAll('#panel-midi .midi-mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === mode),
  );

  document.getElementById('midi-controls-notes').style.display = mode === 'notes' ? '' : 'none';
  document.getElementById('midi-controls-lyrics').style.display = mode === 'lyrics' ? '' : 'none';
  document.getElementById('midi-results-notes').style.display = mode === 'notes' ? '' : 'none';
  document.getElementById('midi-results-lyrics').style.display = mode === 'lyrics' ? '' : 'none';

  if (mode === 'lyrics') refreshLyricsSources();
}

function annotateEngineOption(engine, model) {
  const label = model.display_name;

  // Whisper variants — fixed per-model suffixes.
  if (engine.engine_id === 'whisper') {
    if (model.model_id === 'whisper-large-v3') return `${label} (recommended — GPU)`;
    if (model.model_id === 'whisper-small')    return `${label} (CPU-friendly)`;
    if (model.model_id === 'whisper-tiny')     return `${label} (fastest, lower quality)`;
    return label;
  }

  // Qwen3-ASR variants — purpose-built ASR engine, replaces Qwen2-Audio.
  if (engine.engine_id === 'qwen3-asr') {
    const vram = model.approx_vram_gb ? `~${model.approx_vram_gb} GB VRAM` : 'GPU only';
    const modelAvail = (typeof model.available === 'boolean') ? model.available : engine.available;
    if (!modelAvail) return `${label} (GPU required — ${vram}, unavailable)`;
    return `${label} (GPU required — ${vram})`;
  }

  return label;
}

async function loadLyricsEngines() {
  const sel = document.getElementById('lyrics-engine');
  try {
    const data = await api('/transcribe/engines');
    _lyricsEngines = data.engines || [];
    clearChildren(sel);
    for (const e of _lyricsEngines) {
      for (const m of e.models) {
        const modelAvail = (typeof m.available === 'boolean') ? m.available : e.available;
        const opt = el(
          'option',
          { value: JSON.stringify({ engine_id: e.engine_id, model_id: m.model_id }) },
          annotateEngineOption(e, m),
        );
        if (!modelAvail) opt.disabled = true;
        sel.appendChild(opt);
      }
    }
    // Default selection priority:
    //   1. user's previous choice from this session (appState.lyricsEngineId)
    //   2. whisper-large-v3 (the spec-mandated default)
    //   3. first enabled option
    const remembered = appState.lyricsEngineId;
    let chosen = null;
    if (remembered) {
      chosen = Array.from(sel.options).find(o => {
        if (o.disabled) return false;
        try { return JSON.parse(o.value).model_id === remembered; } catch { return false; }
      });
    }
    if (!chosen) {
      chosen = Array.from(sel.options).find(o => {
        if (o.disabled) return false;
        try { return JSON.parse(o.value).model_id === 'whisper-large-v3'; } catch { return false; }
      });
    }
    if (!chosen) {
      chosen = Array.from(sel.options).find(o => !o.disabled);
    }
    if (chosen) sel.value = chosen.value;
    onLyricsEngineChange();
  } catch (err) {
    // Engine list failed to load — show a disabled placeholder option and
    // keep the Transcribe button disabled so the user gets a clear signal.
    _lyricsEngines = [];
    clearChildren(sel);
    const opt = el('option', { value: '' }, 'Engines unavailable — backend offline?');
    opt.disabled = true;
    sel.appendChild(opt);
    const transcribeBtn = document.getElementById('lyrics-transcribe');
    if (transcribeBtn) transcribeBtn.disabled = true;
  }
}

function onLyricsEngineChange() {
  const sel = document.getElementById('lyrics-engine');
  if (!sel.value) return;
  let parsed;
  try { parsed = JSON.parse(sel.value); } catch { return; }
  const engine = _lyricsEngines.find(e => e.engine_id === parsed.engine_id);
  const notice = document.getElementById('lyrics-coarse-notice');
  if (engine && !engine.supports_word_timestamps) {
    notice.classList.remove('hidden');
  } else {
    notice.classList.add('hidden');
  }
  // Show conditioning toggle only for Whisper — it's a Whisper-specific parameter.
  const condRow = document.getElementById('lyrics-cond-row');
  if (condRow) {
    condRow.style.display = (parsed.engine_id === 'whisper') ? '' : 'none';
  }
  // Remember the user's choice across mode/tab switches within this session.
  appState.lyricsEngineId = parsed.model_id;
}

function refreshLyricsSources() {
  const sel = document.getElementById('lyrics-source');
  if (!sel) return;
  clearChildren(sel);

  const sources = [];
  // Separated stems first
  for (const [label, path] of Object.entries(appState.stemPaths || {})) {
    sources.push({ label: `Stem: ${label}`, path, isVocal: /vocal/i.test(label) });
  }
  // Enhanced stems if any
  for (const [label, path] of Object.entries(appState.enhancePaths || {})) {
    sources.push({ label: `Enhanced: ${label}`, path, isVocal: /vocal/i.test(label) });
  }
  // Uploaded full mix
  if (appState.audioPath) {
    sources.push({ label: 'Full Mix (Original Upload)', path: appState.audioPath, isVocal: false });
  }

  const transcribeBtn = document.getElementById('lyrics-transcribe');
  const loadHint = document.getElementById('lyrics-load-hint');

  if (sources.length === 0) {
    sel.appendChild(el('option', { value: '' }, 'No audio available'));
    transcribeBtn.disabled = true;
    loadHint.classList.remove('hidden');
    return;
  }

  loadHint.classList.add('hidden');
  for (const s of sources) {
    sel.appendChild(el('option', { value: s.path }, s.label));
  }
  // Prefer the first vocal stem if present
  const vocal = sources.find(s => s.isVocal);
  if (vocal) sel.value = vocal.path;
  // Only enable Transcribe if engines also loaded.  If loadLyricsEngines()
  // failed, _lyricsEngines is empty and the engine select shows a disabled
  // placeholder; keep the button disabled until engines arrive.
  if (_lyricsEngines.length === 0) return;
  transcribeBtn.disabled = false;
}

async function startLyricsTranscription() {
  const sourceSel = document.getElementById('lyrics-source');
  const engineSel = document.getElementById('lyrics-engine');
  const langSel = document.getElementById('lyrics-language');
  const fmtLrc = document.getElementById('lyrics-fmt-lrc');
  const fmtSrt = document.getElementById('lyrics-fmt-srt');
  const transcribeBtn = document.getElementById('lyrics-transcribe');

  const path = sourceSel.value;
  if (!path) return;

  let engineCfg;
  try { engineCfg = JSON.parse(engineSel.value); } catch { return; }

  const formats = ['txt'];
  if (fmtLrc.checked) formats.push('lrc');
  if (fmtSrt.checked) formats.push('srt');

  const progressCard = document.getElementById('lyrics-progress');
  const resultContainer = document.getElementById('lyrics-result');
  progressCard.classList.remove('hidden');
  clearChildren(resultContainer);
  transcribeBtn.disabled = true;

  const condCheck = document.getElementById('lyrics-condition-on-previous');
  const collapseCheck = document.getElementById('lyrics-collapse-reps');
  const maxRunInput = document.getElementById('lyrics-max-run');
  const hintInput = document.getElementById('midi-lyrics-hint');
  const conditionOnPrevious = condCheck ? condCheck.checked : false;
  const collapseReps = collapseCheck ? collapseCheck.checked : true;
  const maxRun = maxRunInput ? Math.max(2, Math.min(20, parseInt(maxRunInput.value, 10) || 4)) : 4;
  const hintValue = (hintInput?.value || '').trim();

  try {
    const payload = {
      audio_path: path,
      engine_id: engineCfg.engine_id,
      model_id: engineCfg.model_id,
      language: langSel.value || null,
      formats,
      condition_on_previous_text: conditionOnPrevious,
      collapse_repetitions: collapseReps,
      max_repetition_run: maxRun,
    };
    // Only send `prompt` when non-empty; matches the backend's
    // `prompt: str | None = None` shape and keeps the wire payload clean.
    if (hintValue) payload.prompt = hintValue;

    const { job_id } = await api('/transcribe', {
      method: 'POST',
      body: JSON.stringify(payload),
    });

    pollJob(job_id, {
      onProgress(progress, stage) {
        document.getElementById('lyrics-progress-fill').style.width = `${(progress * 100).toFixed(0)}%`;
        document.getElementById('lyrics-pct').textContent = `${(progress * 100).toFixed(0)}%`;
        document.getElementById('lyrics-stage').textContent = stage || '';
      },
      onDone(result) {
        progressCard.classList.add('hidden');
        transcribeBtn.disabled = false;
        renderLyricsResult(result);
        appState.lyricsPaths = { ...(appState.lyricsPaths || {}), ...result.output_paths };
        appState.emit('lyricsReady', result);
      },
      onError(msg) {
        progressCard.classList.add('hidden');
        transcribeBtn.disabled = false;
        resultContainer.appendChild(
          el('div', { className: 'banner banner-error' }, `Transcription failed: ${msg}`),
        );
      },
    });
  } catch (err) {
    progressCard.classList.add('hidden');
    transcribeBtn.disabled = false;
    resultContainer.appendChild(
      el('div', { className: 'banner banner-error' }, `Error: ${err.message}`),
    );
  }
}

function renderLyricsResult(result) {
  const container = document.getElementById('lyrics-result');
  const card = el('div', { className: 'card' });

  const meta = el('div', { className: 'lyrics-meta' },
    el('span', { className: 'lyrics-badge' }, `engine: ${result.engine_id}`),
    el('span', { className: 'lyrics-badge' }, `model: ${result.model_id}`),
    result.language ? el('span', { className: 'lyrics-badge' }, `language: ${result.language}`) : null,
    el('span', { className: 'lyrics-badge' }, `segments: ${result.segment_count}`),
  );

  const textarea = el('textarea', {
    className: 'lyrics-text',
    readonly: 'true',
    spellcheck: 'false',
  });
  textarea.value = result.text || '(empty transcript)';

  const actions = el('div', { className: 'stem-actions', style: { marginTop: '10px', display: 'flex', gap: '8px', flexWrap: 'wrap' } });
  for (const [fmt, path] of Object.entries(result.output_paths)) {
    const btn = el('button', {
      className: 'btn btn-sm',
      onClick: () => {
        const name = path.split('/').pop() || `lyrics.${fmt}`;
        saveFileAs(`/api/audio/download?path=${encodeURIComponent(path)}`, name);
      },
    }, `Save .${fmt}`);
    actions.appendChild(btn);
  }
  const sendBtn = el('button', {
    className: 'btn btn-sm btn-primary',
    onClick: () => {
      appState.emit('lyricsSendToCompose', result);
    },
  }, 'Send to Compose');
  actions.appendChild(sendBtn);

  card.append(meta, textarea, actions);
  container.appendChild(card);
}
