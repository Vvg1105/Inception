/**
 * EEG live decode → window.__dreamEEG
 *
 * **Blink**
 *   { "present": true, "blink": false }
 *   { "capture": true }  → opens place UI (dream-blink-up)
 *
 * **Emotion** (optional — drives 2D pad + focus when “EEG drives mood” is on):
 *   { "emotion": { "arousal": 0.7, "valence": 0.4, "focus": 0.6 } }  // all 0..1
 *   { "arousal": 0.7, "valence": 0.4 }  // partial updates keep previous values
 */
let ws = null;
let rafId = null;
let lastPayload = { present: false, blink: false };
let staleTimer = null;
const STALE_MS = 3000;

const CAPTURE_DEBOUNCE_MS = 320;
let lastCaptureAt = 0;

function clamp01(v, fallback) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(1, n));
}

function extractEmotion(d, prev) {
  const prevE = prev && prev.emotion
    ? prev.emotion
    : { arousal: 0.5, valence: 0.5, focus: 0.5 };

  if (d.emotion && typeof d.emotion === 'object') {
    return {
      arousal: clamp01(d.emotion.arousal, prevE.arousal),
      valence: clamp01(d.emotion.valence, prevE.valence),
      focus: clamp01(d.emotion.focus, prevE.focus),
      label: d.emotion.label || prevE.label || '',
    };
  }

  const a = Number.isFinite(Number(d.arousal)) ? clamp01(d.arousal, prevE.arousal) : null;
  const v = Number.isFinite(Number(d.valence)) ? clamp01(d.valence, prevE.valence) : null;
  const f = Number.isFinite(Number(d.focus)) ? clamp01(d.focus, prevE.focus) : null;
  if (a === null && v === null && f === null) return prev.emotion;

  return {
    arousal: a ?? prevE.arousal,
    valence: v ?? prevE.valence,
    focus: f ?? prevE.focus,
    label: d.label || prevE.label || '',
  };
}

function pushToWindow() {
  if (typeof window === 'undefined') return;
  const out = {
    present: lastPayload.present,
    blink: lastPayload.blink,
  };
  if (lastPayload.emotion) {
    out.emotion = { ...lastPayload.emotion };
  }
  window.__dreamEEG = out;
}

function scheduleStaleCheck() {
  if (staleTimer) clearTimeout(staleTimer);
  staleTimer = setTimeout(() => {
    lastPayload = { present: false, blink: false };
    pushToWindow();
  }, STALE_MS);
}

function isBlinkCapture(d) {
  if (!d || typeof d !== 'object') return false;
  if (d.capture === true) return true;
  if (d.blink_capture === true) return true;
  if (d.event === 'blink_capture' || d.type === 'blink_capture') return true;
  return false;
}

function dispatchBlinkCapture() {
  const now = Date.now();
  if (now - lastCaptureAt < CAPTURE_DEBOUNCE_MS) return;
  lastCaptureAt = now;
  window.dispatchEvent(
    new CustomEvent('dream-blink-up', { detail: { source: 'eeg', capture: true } })
  );
}

function loop() {
  pushToWindow();
  rafId = requestAnimationFrame(loop);
}

export function ensureEEGWindowSync() {
  if (rafId == null) rafId = requestAnimationFrame(loop);
}

/**
 * @param {{ url?: string, staleMs?: number }} [options]
 * @returns {Promise<void>}
 */
export function startEEGBridge(options = {}) {
  const url = options.url || 'ws://127.0.0.1:8765';
  const staleMs = options.staleMs ?? STALE_MS;

  stopEEGBridge();

  return new Promise((resolve, reject) => {
    try {
      ws = new WebSocket(url);
    } catch (e) {
      reject(e);
      return;
    }

    ws.onopen = () => {
      lastPayload = { present: true, blink: false };
      pushToWindow();
      ensureEEGWindowSync();
      if (staleMs > 0) scheduleStaleCheck();
      resolve();
    };

    ws.onmessage = (ev) => {
      try {
        const d = typeof ev.data === 'string' ? JSON.parse(ev.data) : JSON.parse(String(ev.data));

        const em = extractEmotion(d, lastPayload);
        const hasEmotion =
          em &&
          typeof em.arousal === 'number' &&
          typeof em.valence === 'number' &&
          typeof em.focus === 'number';

        if (isBlinkCapture(d)) {
          dispatchBlinkCapture();
          if (d.present !== undefined || d.blink !== undefined) {
            lastPayload = {
              ...lastPayload,
              present: d.present !== false,
              blink: d.blink !== undefined ? !!d.blink : lastPayload.blink,
              ...(hasEmotion ? { emotion: em } : {}),
            };
          } else if (hasEmotion) {
            lastPayload = { ...lastPayload, emotion: em };
          }
          pushToWindow();
          if (staleMs > 0) scheduleStaleCheck();
          return;
        }

        lastPayload = {
          present: d.present !== false,
          blink: !!d.blink,
          ...(hasEmotion ? { emotion: em } : lastPayload.emotion ? { emotion: lastPayload.emotion } : {}),
        };
        pushToWindow();
        if (staleMs > 0) scheduleStaleCheck();
      } catch (_) {
        /* ignore bad frames */
      }
    };

    ws.onerror = () => {
      lastPayload = { present: false, blink: false };
      pushToWindow();
    };

    ws.onclose = () => {
      if (staleTimer) clearTimeout(staleTimer);
      lastPayload = { present: false, blink: false };
      pushToWindow();
    };
  });
}

export function stopEEGBridge() {
  if (staleTimer) {
    clearTimeout(staleTimer);
    staleTimer = null;
  }
  if (rafId != null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  if (ws) {
    try {
      ws.close();
    } catch (_) {
      /* ignore */
    }
    ws = null;
  }
  lastPayload = { present: false, blink: false };
  pushToWindow();
}
