/**
 * StemForge — Main application module
 *
 * State management, event bus, tab switching, job polling,
 * and component initialization.
 */

// ─── Event Bus / App State ──────────────────────────────────────────────

export const appState = {
  _listeners: {},

  on(event, cb) {
    if (!this._listeners[event]) this._listeners[event] = [];
    this._listeners[event].push(cb);
  },

  emit(event, data) {
    (this._listeners[event] || []).forEach(cb => cb(data));
  },

  // Current session data (mirrors backend)
  audioPath: null,
  audioInfo: null,
  stemPaths: {},
  midiLabels: [],
  musicgenPath: null,
  mixPath: null,
  composePaths: [],
  sfxPaths: {},
  voicePaths: {},
  enhancePaths: {},
  lyricsPaths: {},
  lyricsEngineId: null,    // remembered Lyrics-mode engine selection (per session)
};

// ─── API Helpers ────────────────────────────────────────────────────────

export async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function apiUpload(path, file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`/api${path}`, { method: 'POST', body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ─── Job Polling ────────────────────────────────────────────────────────

/**
 * Poll a background job until completion.
 *
 * @param {string} jobId
 * @param {object} opts - { onProgress(progress, stage), onDone(result), onError(msg), interval }
 */
export function pollJob(jobId, { onProgress, onDone, onError, interval = 10000 } = {}) {
  const timer = setInterval(async () => {
    try {
      const job = await api(`/jobs/${jobId}`);

      if (job.status === 'running' || job.status === 'pending') {
        onProgress?.(job.progress, job.stage);
      } else if (job.status === 'done') {
        clearInterval(timer);
        onProgress?.(1.0, 'Done');
        onDone?.(job.result);
      } else if (job.status === 'error') {
        clearInterval(timer);
        onError?.(job.error || 'Unknown error');
      }
    } catch (err) {
      clearInterval(timer);
      onError?.(err.message);
    }
  }, interval);

  return () => clearInterval(timer);  // cancel function
}

// ─── Save-As Helper ─────────────────────────────────────────────────────

/**
 * Prompt the user to save a remote file with a "Save As" dialog.
 * Uses the File System Access API when available (Chromium),
 * falls back to a download link for Firefox/Safari.
 *
 * @param {string} url - URL to fetch the file from
 * @param {string} suggestedName - default filename shown in the dialog
 */
export async function saveFileAs(url, suggestedName = 'download') {
  // Always use blob + anchor download — showSaveFilePicker's WritableFileStream
  // is unreliable on Linux/Chrome (leaves .crdownload temp files).
  const res = await fetch(url);
  const blob = await res.blob();
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = blobUrl;
  a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(blobUrl);
}

// ─── Utility ────────────────────────────────────────────────────────────

export function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function el(tag, attrs = {}, ...children) {
  const elem = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'className') elem.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(elem.style, v);
    else if (k.startsWith('on')) elem.addEventListener(k.slice(2).toLowerCase(), v);
    else elem.setAttribute(k, v);
  }
  for (const child of children) {
    if (typeof child === 'string') elem.appendChild(document.createTextNode(child));
    else if (child) elem.appendChild(child);
  }
  return elem;
}

// ─── Tab Switching ──────────────────────────────────────────────────────

function initTabs() {
  const tabBar = document.getElementById('tab-bar');
  const buttons = tabBar.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.tab-panel');

  tabBar.addEventListener('click', (e) => {
    const btn = e.target.closest('.tab-btn');
    if (!btn) return;

    const tabName = btn.dataset.tab;

    buttons.forEach(b => b.classList.toggle('active', b === btn));
    panels.forEach(p => p.classList.toggle('hidden', p.id !== `panel-${tabName}`));
  });
}

// ─── Device Badge ───────────────────────────────────────────────────────

async function initDeviceBadge() {
  try {
    const info = await api('/device');
    const badge = document.getElementById('device-badge');
    if (info.gpu_name) {
      badge.textContent = info.gpu_name;
      badge.classList.add('gpu');
    } else {
      badge.textContent = info.device;
    }
  } catch {
    document.getElementById('device-badge').textContent = 'offline';
  }
}

// ─── Component Init ─────────────────────────────────────────────────────

async function initComponents() {
  const [
    { initLoader },
    { initSeparate },
    { initEnhance },
    { initMidi },
    { initMix },
    { initGenerate },
    { initCompose },
    { initExport },
    { initTransport },
  ] = await Promise.all([
    import('./components/loader.js'),
    import('./components/separate.js'),
    import('./components/enhance.js'),
    import('./components/midi.js'),
    import('./components/mix.js'),
    import('./components/generate.js'),
    import('./components/compose.js'),
    import('./components/export.js'),
    import('./components/audio-player.js'),
  ]);

  initLoader();
  initSeparate();
  initEnhance();
  initMidi();
  initMix();
  initGenerate();
  initCompose();
  initExport();
  initTransport();
}

// ─── New Session ─────────────────────────────────────────────────────────

async function initUserBadge() {
  try {
    const info = await api('/session');
    const user = info.user;
    if (user && user !== 'local') {
      const badge = document.getElementById('user-badge');
      badge.textContent = user;
      badge.style.display = '';
    }
  } catch { /* single-user mode, no badge */ }
}

function initNewSession() {
  document.getElementById('new-session-btn').addEventListener('click', async () => {
    if (!confirm('Start a new session? All tracks, stems, and canvases will be cleared.')) return;
    try {
      await api('/session', { method: 'DELETE' });
    } catch { /* server may be unreachable, reload anyway */ }
    window.location.reload();
  });
}

// ─── Boot ───────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initDeviceBadge();
  initUserBadge();
  initNewSession();
  initComponents();
});
