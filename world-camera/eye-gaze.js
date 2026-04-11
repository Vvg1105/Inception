/**
 * Eye tracking WebSocket consumer.
 *
 * Python (backend/eye_track.py) runs the webcam + MediaPipe iris tracking
 * and sends { gaze: {x,y}, blink: bool } over WebSocket.
 * This module just receives those values and writes window.__dreamGaze.
 */

let ws = null;
let staleTimer = null;
const STALE_MS = 2000;

function slider(id, fb) {
  const el = document.getElementById(id);
  if (!el) return fb;
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : fb;
}

function publish(x, y, blink, face) {
  window.__dreamGaze = {
    active: true,
    x, y,
    mix: slider('gaze-mix', 0.85),
    blinkingFromCamera: blink,
    faceDetected: face,
  };
}

function markStale() {
  window.__dreamGaze = { active: false };
}

function scheduleStale() {
  if (staleTimer) clearTimeout(staleTimer);
  staleTimer = setTimeout(markStale, STALE_MS);
}

export async function loadWebgazerScript() {}

export async function startEyeGazeBridge(url) {
  stopEyeGazeBridge();

  const wsUrl = url || 'ws://127.0.0.1:8766';
  window.__dreamGaze = { active: true, x: NaN, y: NaN, mix: slider('gaze-mix', 0.85), blinkingFromCamera: false, faceDetected: false };

  return new Promise((resolve, reject) => {
    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      reject(e);
      return;
    }

    ws.onopen = () => {
      console.log('[eye-gaze] connected to', wsUrl);
      scheduleStale();
      resolve();
    };

    ws.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        const gaze = d.gaze;
        if (!gaze) return;

        const x = typeof gaze.x === 'number' ? gaze.x : 0;
        const y = typeof gaze.y === 'number' ? gaze.y : 0;
        const blink = !!d.blink;
        const face = !d.no_face;

        publish(x, y, blink, face);
        scheduleStale();

        if (d.capture) {
          const srcEl = document.getElementById('blink-source');
          const mode = (srcEl && srcEl.value) || 'auto';
          const eeg = window.__dreamEEG;
          const eegActive = eeg && eeg.present;
          if (mode === 'camera' || (mode === 'auto' && !eegActive)) {
            window.dispatchEvent(
              new CustomEvent('dream-blink-up', { detail: { source: 'camera', capture: true } })
            );
          }
        }
      } catch (_) {}
    };

    ws.onerror = () => {
      markStale();
    };

    ws.onclose = () => {
      if (staleTimer) clearTimeout(staleTimer);
      markStale();
    };
  });
}

export function stopEyeGazeBridge() {
  if (staleTimer) { clearTimeout(staleTimer); staleTimer = null; }
  if (ws) { try { ws.close(); } catch (_) {} ws = null; }
  window.__dreamGaze = { active: false };
}
