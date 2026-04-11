/**
 * TRIBE v2 fMRI decoder bridge
 *
 * Connects to the TRIBE WebSocket server (backend/tribe_ws.py) and
 * exposes decoded object predictions on window.__dreamTribe.
 *
 * When a prediction arrives, dispatches a CustomEvent "tribe-decoded"
 * on window with the prediction detail so index.html can auto-place objects.
 *
 * Usage from index.html:
 *   import { startTribeBridge, stopTribeBridge } from './world-camera/tribe-bridge.js';
 *   startTribeBridge('ws://127.0.0.1:8766');
 *   window.addEventListener('tribe-decoded', (e) => { ... e.detail ... });
 */

let ws = null;
let reconnectTimer = null;
let url = '';

const state = {
  connected: false,
  object_class: '',
  object_idx: -1,
  size: '',
  confidence: 0,
  probabilities: {},
  lastPrediction: 0,
};

window.__dreamTribe = state;

function connect() {
  if (ws && ws.readyState <= 1) return;

  try {
    ws = new WebSocket(url);
  } catch { return; }

  ws.onopen = () => {
    state.connected = true;
    console.log('[TRIBE] connected to', url);
    if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (ev) => {
    let d;
    try { d = JSON.parse(ev.data); } catch { return; }
    if (!d.tribe) return;

    state.object_class = d.object_class || '';
    state.object_idx = d.object_idx ?? -1;
    state.size = d.size || '';
    state.confidence = d.confidence || 0;
    state.probabilities = d.probabilities || {};
    state.lastPrediction = Date.now();

    window.dispatchEvent(new CustomEvent('tribe-decoded', { detail: { ...d } }));
  };

  ws.onclose = () => {
    state.connected = false;
    console.log('[TRIBE] disconnected');
    if (!reconnectTimer) {
      reconnectTimer = setInterval(connect, 3000);
    }
  };

  ws.onerror = () => { ws?.close(); };
}

export function startTribeBridge(wsUrl = 'ws://127.0.0.1:8766') {
  url = wsUrl;
  connect();
}

export function stopTribeBridge() {
  if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
  if (ws) { ws.close(); ws = null; }
  state.connected = false;
}

export function triggerTribePrediction() {
  if (ws && ws.readyState === 1) {
    ws.send(JSON.stringify({ trigger: true }));
  }
}
