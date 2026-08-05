/**
 * Export tab — preview artifacts, choose format, export with auto-download.
 */

import { appState, api, apiUpload, pollJob, el, formatTime, saveFileAs } from '../app.js';
import { createWaveform } from './waveform.js';
import { transportLoad, transportStop } from './audio-player.js';

function clearChildren(elem) {
  while (elem.firstChild) elem.removeChild(elem.firstChild);
}

const _LOSSY_FORMATS = new Set(['mp3', 'ogg', 'm4a']);
const _LOSSLESS_FORMATS = new Set(['wav', 'flac', 'aiff']);

/** Files uploaded directly in the Export tab (independent of session). */
let _exportFiles = [];

/** Active waveform players for exclusive playback. */
let _players = [];

function _stopOtherPlayers(except) {
  for (const p of _players) {
    if (p.ws !== except && p.ws.isPlaying()) {
      p.ws.stop();
      p.playBtn.textContent = '\u25B6 Play';
    }
  }
}

export function initExport() {
  const panel = document.getElementById('panel-export');

  // ─── File loader (convert any file without touching other tabs) ───
  const exportFileInput = el('input', {
    type: 'file',
    accept: '.wav,.flac,.mp3,.ogg,.aiff,.aif,.m4a,.mp4,.mkv,.webm,.avi,.mov,.m4v,.flv',
    multiple: 'true',
    style: { display: 'none' },
    id: 'export-file-input',
  });

  const dropZone = el('div', { className: 'export-drop-zone', id: 'export-drop-zone' },
    el('span', { className: 'drop-text' }, 'Drop files here to convert, or click to browse'),
    el('span', { className: 'drop-hint' }, 'WAV, FLAC, MP3, OGG, AIFF, M4A — or video'),
  );

  dropZone.addEventListener('click', () => exportFileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    if (files.length) _handleExportFiles(files);
  });

  exportFileInput.addEventListener('change', () => {
    const files = Array.from(exportFileInput.files);
    if (files.length) _handleExportFiles(files);
    exportFileInput.value = '';
  });

  // ─── Top bar: format, quality settings, export button ───
  const formatSelect = el('select', { id: 'export-format' },
    el('option', { value: 'wav' }, 'WAV (lossless)'),
    el('option', { value: 'flac' }, 'FLAC (lossless)'),
    el('option', { value: 'aiff' }, 'AIFF (lossless)'),
    el('option', { value: 'mp3' }, 'MP3'),
    el('option', { value: 'ogg' }, 'OGG Opus'),
    el('option', { value: 'm4a' }, 'M4A (AAC)'),
  );

  // Sample rate — applies to all formats
  const sampleRateSelect = el('select', { id: 'export-sample-rate' },
    el('option', { value: '' }, 'Original'),
    el('option', { value: '22050' }, '22050 Hz'),
    el('option', { value: '44100' }, '44100 Hz'),
    el('option', { value: '48000' }, '48000 Hz'),
    el('option', { value: '88200' }, '88200 Hz'),
    el('option', { value: '96000' }, '96000 Hz'),
  );

  // Lossy: bitrate slider
  const bitrateSlider = el('input', {
    type: 'range', id: 'export-bitrate', min: '64', max: '320', step: '32', value: '192',
  });
  const bitrateLabel = el('span', { id: 'export-bitrate-value' }, '192 kbps');
  const bitrateGroup = el('span', { className: 'export-bitrate-inline hidden', id: 'export-bitrate-group' },
    bitrateLabel, bitrateSlider,
  );

  // Lossless only: bit depth
  const bitDepthSelect = el('select', { id: 'export-bit-depth' },
    el('option', { value: '' }, 'Original'),
    el('option', { value: '16' }, '16-bit'),
    el('option', { value: '24' }, '24-bit'),
    el('option', { value: '32' }, '32-bit'),
  );

  const bitDepthGroup = el('span', { className: 'export-lossless-inline', id: 'export-bit-depth-group' },
    el('label', {}, 'Depth: ', bitDepthSelect),
  );

  // MIDI SMF format — shown only when MIDI artifacts are available.  Format 0
  // (single-track) is required by some hardware arrangers/keyboards.
  const midiFormatSelect = el('select', { id: 'export-midi-format' },
    el('option', { value: '1' }, 'Format 1 (multi-track)'),
    el('option', { value: '0' }, 'Format 0 (single-track)'),
  );
  const midiFormatGroup = el('span', { className: 'export-lossless-inline hidden', id: 'export-midi-format-group' },
    el('label', {}, 'MIDI: ', midiFormatSelect),
  );

  const exportBtn = el('button', { className: 'btn btn-primary', id: 'export-start', disabled: 'true' }, 'Export Selected');

  const topBar = el('div', { className: 'export-top-bar' },
    el('label', {}, 'Format: ', formatSelect),
    el('label', {}, 'Rate: ', sampleRateSelect),
    bitrateGroup,
    bitDepthGroup,
    midiFormatGroup,
    exportBtn,
  );

  // ─── Progress bar (hidden by default) ───
  const progressBar = el('div', { className: 'export-progress-bar hidden', id: 'export-progress' },
    el('div', { className: 'progress-fill', id: 'export-progress-fill' }),
  );

  // ─── Artifact preview area ───
  const previewArea = el('div', { id: 'export-previews', className: 'export-previews' },
    el('span', { className: 'text-dim' }, 'Load a file to get started'),
  );

  panel.append(exportFileInput, dropZone, topBar, progressBar, previewArea);

  // ─── Wire events ───
  exportBtn.addEventListener('click', startExport);

  formatSelect.addEventListener('change', _updateFormatSettings);
  bitrateSlider.addEventListener('input', _updateBitrateLabel);

  // Set initial visibility
  _updateFormatSettings();

  // Listen for all ready events
  appState.on('fileLoaded', refreshPreviews);
  appState.on('stemsReady', refreshPreviews);
  appState.on('midiReady', refreshPreviews);
  appState.on('generateReady', refreshPreviews);
  appState.on('composeReady', refreshPreviews);
  appState.on('mixReady', refreshPreviews);
  appState.on('sfxReady', refreshPreviews);
  appState.on('transformReady', refreshPreviews);
  appState.on('enhanceReady', refreshPreviews);
  appState.on('lyricsReady', refreshPreviews);
}

async function _handleExportFiles(fileList) {
  const dropZone = document.getElementById('export-drop-zone');

  clearChildren(dropZone);
  dropZone.appendChild(el('span', { className: 'drop-text' }, `Uploading ${fileList.length} file${fileList.length > 1 ? 's' : ''}...`));

  let added = 0;
  for (const file of fileList) {
    try {
      const info = await apiUpload('/upload', file);
      _exportFiles.push({ label: info.filename, path: info.path });
      added++;
    } catch (err) {
      console.error(`Export upload failed for ${file.name}:`, err);
    }
  }

  clearChildren(dropZone);
  if (added > 0) {
    dropZone.append(
      el('span', { className: 'drop-text' }, `\u2713 ${added} file${added > 1 ? 's' : ''} added`),
      el('span', { className: 'drop-hint' }, 'Drop more files to add'),
    );
    refreshPreviews();
  } else {
    dropZone.append(
      el('span', { className: 'drop-text', style: { color: 'var(--error)' } }, 'Upload failed'),
      el('span', { className: 'drop-hint' }, 'Try again'),
    );
  }
}

function _updateFormatSettings() {
  const fmt = document.getElementById('export-format').value;
  const bitrateGroup = document.getElementById('export-bitrate-group');
  const bitDepthGroup = document.getElementById('export-bit-depth-group');

  if (_LOSSY_FORMATS.has(fmt)) {
    bitrateGroup.classList.remove('hidden');
    bitDepthGroup.classList.add('hidden');
    const defaultBr = fmt === 'ogg' ? 128 : 192;
    document.getElementById('export-bitrate').value = defaultBr;
    _updateBitrateLabel();
  } else {
    bitrateGroup.classList.add('hidden');
    bitDepthGroup.classList.remove('hidden');
  }
}

function _updateBitrateLabel() {
  const v = document.getElementById('export-bitrate').value;
  document.getElementById('export-bitrate-value').textContent = v + ' kbps';
}

async function _fetchAudioInfo(path, badge) {
  try {
    const info = await api(`/audio/info?path=${encodeURIComponent(path)}`);
    const parts = [];
    if (info.sample_rate) parts.push(`${(info.sample_rate / 1000).toFixed(info.sample_rate % 1000 ? 1 : 0)} kHz`);
    if (info.bit_depth) parts.push(`${info.bit_depth}-bit`);
    if (parts.length) badge.textContent = parts.join(' · ');
  } catch { /* non-critical */ }
}

// ─── Preview players ──────────────────────────────────────────────────

function refreshPreviews() {
  const container = document.getElementById('export-previews');

  // Destroy old players
  for (const p of _players) { try { p.ws.destroy(); } catch {} }
  _players = [];
  clearChildren(container);

  const items = collectArtifacts();
  if (items.length === 0) {
    container.appendChild(el('span', { className: 'text-dim' }, 'No artifacts yet'));
    document.getElementById('export-start').disabled = true;
    return;
  }

  document.getElementById('export-start').disabled = false;

  // MIDI format selector only matters when MIDI artifacts exist
  document.getElementById('export-midi-format-group')
    .classList.toggle('hidden', !items.some(i => i.type === 'midi'));

  for (const item of items) {
    // MIDI artifacts live in the session (no file path until export time);
    // the checkbox carries the stem label instead of a path.
    const checkbox = item.type === 'midi'
      ? el('input', { type: 'checkbox', checked: 'true', 'data-midi-label': item.midiLabel })
      : el('input', { type: 'checkbox', checked: 'true', 'data-path': item.path });

    // Lyrics (.txt/.lrc/.srt) and MIDI outputs are non-audio \u2014 render a
    // simple row with a checkbox + label so they can be selected for export,
    // but skip the waveform/playback chrome that only makes sense for audio.
    if (item.type === 'lyrics' || item.type === 'midi') {
      const ext = item.type === 'midi'
        ? 'mid'
        : (item.path.split('.').pop() || '').toLowerCase();
      const header = el('div', { className: 'stem-card-header' },
        el('label', { className: 'export-check-label' }, checkbox,
          el('span', { className: 'stem-label' }, `${item.label}  `,
            el('span', { className: 'text-dim' }, `(${item.type}${ext ? ' \u00B7 .' + ext : ''})`),
          ),
        ),
      );
      const card = el('div', { className: 'stem-card' }, header);
      container.appendChild(card);
      continue;
    }

    const streamUrl = `/api/audio/stream?path=${encodeURIComponent(item.path)}`;

    const playBtn = el('button', { className: 'btn btn-sm' }, '\u25B6 Play');
    const stopBtn = el('button', { className: 'btn btn-sm' }, '\u25A0');
    const rewindBtn = el('button', { className: 'btn btn-sm' }, '\u23EA');
    const timeLabel = el('span', { className: 'stem-time' }, '0:00 / 0:00');
    const infoBadge = el('span', { className: 'export-info-badge text-dim' });

    const header = el('div', { className: 'stem-card-header' },
      el('label', { className: 'export-check-label' }, checkbox,
        el('span', { className: 'stem-label' }, `${item.label}  `, el('span', { className: 'text-dim' }, `(${item.type})`)),
        infoBadge,
      ),
      el('div', { className: 'stem-actions' },
        playBtn, stopBtn, rewindBtn, timeLabel,
      ),
    );

    // Fetch source audio info (rate + depth) asynchronously
    _fetchAudioInfo(item.path, infoBadge);

    const waveContainer = el('div', { className: 'stem-waveform' });
    const card = el('div', { className: 'stem-card' }, header, waveContainer);
    container.appendChild(card);

    const ws = createWaveform(waveContainer, { height: 50 });
    ws.load(streamUrl);
    _players.push({ ws, playBtn });

    playBtn.addEventListener('click', () => {
      if (ws.isPlaying()) {
        ws.pause();
        playBtn.textContent = '\u25B6 Play';
      } else {
        _stopOtherPlayers(ws);
        ws.play();
        playBtn.textContent = '\u23F8 Pause';
        transportLoad(streamUrl, item.label, false, 'Export', { cardWs: ws });
      }
    });

    stopBtn.addEventListener('click', () => {
      ws.stop();
      transportStop();
      playBtn.textContent = '\u25B6 Play';
    });

    rewindBtn.addEventListener('click', () => ws.setTime(0));

    ws.on('timeupdate', (time) => {
      const dur = ws.getDuration();
      timeLabel.textContent = `${formatTime(time)} / ${formatTime(dur)}`;
    });

    ws.on('finish', () => {
      playBtn.textContent = '\u25B6 Play';
      transportStop();
    });
  }
}

function collectArtifacts() {
  const items = [];

  // Original upload
  if (appState.audioPath) {
    const name = appState.audioInfo?.filename || appState.audioPath.split('/').pop();
    items.push({ label: name, path: appState.audioPath, type: 'original' });
  }

  // Stems
  for (const [label, path] of Object.entries(appState.stemPaths || {})) {
    items.push({ label, path, type: 'stem' });
  }

  // MIDI extractions — held in the session until saved; resolved to file
  // paths at export time via /api/midi/save (see startExport).
  if (appState.midiHasMerged) {
    items.push({ label: 'Merged MIDI (all tracks)', type: 'midi', midiLabel: 'merged' });
  }
  for (const label of appState.midiLabels || []) {
    items.push({ label: `${label} MIDI`, type: 'midi', midiLabel: label });
  }

  // Generated audio (Synth tab)
  if (appState.musicgenPath) {
    items.push({ label: 'Generated', path: appState.musicgenPath, type: 'generated' });
  }

  // Composed audio (Compose tab)
  for (const entry of appState.composePaths || []) {
    if (entry.path) {
      items.push({ label: entry.title || 'Composed', path: entry.path, type: 'composed' });
    }
  }

  // SFX stems
  for (const [label, path] of Object.entries(appState.sfxPaths || {})) {
    items.push({ label, path, type: 'sfx' });
  }

  // Voice transforms
  for (const [label, path] of Object.entries(appState.voicePaths || {})) {
    items.push({ label, path, type: 'voice' });
  }

  // Enhanced stems
  for (const [label, path] of Object.entries(appState.enhancePaths || {})) {
    items.push({ label, path, type: 'enhanced' });
  }

  // Lyrics transcription outputs (.txt/.lrc/.srt) — non-audio
  for (const [label, path] of Object.entries(appState.lyricsPaths || {})) {
    items.push({ label, path, type: 'lyrics' });
  }

  // Mix
  if (appState.mixPath) {
    items.push({ label: 'Mix', path: appState.mixPath, type: 'mix' });
  }

  // Files added directly in the Export tab
  for (const f of _exportFiles) {
    items.push({ label: f.label, path: f.path, type: 'convert' });
  }

  return items;
}

// ─── Export ───────────────────────────────────────────────────────────

async function startExport() {
  const checkedEls = document.querySelectorAll('#export-previews input[type="checkbox"]:checked');
  const items = [];
  const midiLabels = [];
  for (const cb of checkedEls) {
    if (cb.dataset.midiLabel) midiLabels.push(cb.dataset.midiLabel);
    else if (cb.dataset.path) items.push(cb.dataset.path);
  }
  const format = document.getElementById('export-format').value;

  if (!items.length && !midiLabels.length) return;

  // Materialize selected MIDI artifacts to disk first (they live in the
  // session until saved), honoring the chosen SMF format.
  if (midiLabels.length) {
    const smfFormat = parseInt(document.getElementById('export-midi-format').value, 10);
    for (const label of midiLabels) {
      try {
        const res = await api('/midi/save', {
          method: 'POST',
          body: JSON.stringify({ label, smf_format: smfFormat }),
        });
        items.push(res.path);
      } catch (err) {
        alert(`MIDI save failed for ${label}: ${err.message}`);
        return;
      }
    }
  }

  const body = { items, format };
  const sr = document.getElementById('export-sample-rate').value;
  if (sr) body.sample_rate = parseInt(sr, 10);

  if (_LOSSY_FORMATS.has(format)) {
    body.bitrate = parseInt(document.getElementById('export-bitrate').value, 10);
  } else if (_LOSSLESS_FORMATS.has(format)) {
    const bd = document.getElementById('export-bit-depth').value;
    if (bd) body.bit_depth = parseInt(bd, 10);
  }

  const progressBar = document.getElementById('export-progress');
  const fill = document.getElementById('export-progress-fill');
  progressBar.classList.remove('hidden');
  fill.style.width = '0%';

  try {
    const { job_id } = await api('/export', {
      method: 'POST',
      body: JSON.stringify(body),
    });

    pollJob(job_id, {
      onProgress(progress) {
        fill.style.width = `${(progress * 100).toFixed(0)}%`;
      },
      async onDone(result) {
        progressBar.classList.add('hidden');
        const exported = result.exported || [];
        // Auto-download: single file → save dialog, multiple → zip
        if (exported.length === 1) {
          const name = exported[0].split('/').pop();
          await saveFileAs(`/api/audio/download?path=${encodeURIComponent(exported[0])}`, name);
        } else if (exported.length > 1) {
          await _downloadZip(exported);
        }
      },
      onError(msg) {
        progressBar.classList.add('hidden');
        alert(`Export failed: ${msg}`);
      },
    });
  } catch {
    progressBar.classList.add('hidden');
  }
}

async function _downloadZip(paths) {
  const res = await fetch('/api/export/download-zip', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items: paths }),
  });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'stemforge_export.zip';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 1000);
}
