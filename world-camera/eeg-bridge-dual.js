/**
 * eeg-bridge-dual.js — Dual-user EEG WebSocket bridge (Neural Symbiosis demo).
 *
 * Connects to one WebSocket served by backend/eeg_decode_dual.py and parses
 * the combined dual-user JSON into three window objects:
 *
 *   window.__dreamEEG1       — User 1 (GTECH BCICore-8)
 *   window.__dreamEEG2       — User 2 (OpenBCI Cyton)
 *   window.__dreamSymbiosis  — Neural consensus / combined state
 *   window.__dreamActiveUser — 1 or 2 (who has creation permission)
 *   window.__dreamEEG        — mirror of the active user (backward compat)
 *
 * Window object shapes:
 *   __dreamEEG1 / __dreamEEG2 : {
 *       present : bool,
 *       blink   : bool,
 *       emotion : { arousal, valence, focus: 0..1, label: string }
 *   }
 *   __dreamSymbiosis : {
 *       arousal, valence, score: 0..1,
 *       label: "resonant"|"entangled"|"drifting"|"divergent"
 *   }
 *
 * Dispatches:
 *   CustomEvent('dream-blink-up')
 *     detail: { source: 'eeg', capture: true, user: 1|2 }
 *     — only fired when the active user has a blink rising-edge
 *
 * CustomEvent('dream-permission-change')
 *     detail: { activeUser: 1|2 }
 *     — fired whenever active_user changes
 */

let _ws            = null;
let _rafId         = null;
let _staleTimer    = null;
let _lastPayload   = {};
let _prevActive    = 1;

const STALE_MS           = 3000;
const CAPTURE_DEBOUNCE   = 320;
let   _lastCaptureAt     = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────

function _clamp(v, fb) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.max(0, Math.min(1, n)) : fb;
}

function _extractEmotion(obj, prev) {
  const pe = prev?.emotion ?? { arousal: 0.5, valence: 0.5, focus: 0.5, label: '' };
  if (!obj?.emotion || typeof obj.emotion !== 'object') return pe;
  return {
    arousal: _clamp(obj.emotion.arousal, pe.arousal),
    valence: _clamp(obj.emotion.valence, pe.valence),
    focus:   _clamp(obj.emotion.focus,   pe.focus),
    label:   obj.emotion.label || pe.label || '',
  };
}

const _deadState = () => ({
  present: false, blink: false,
  emotion: { arousal: 0.5, valence: 0.5, focus: 0.5, label: '' },
  ch_amplitudes: [],
});

// ── Push latest payload into window globals ───────────────────────────────────

function _push(d) {
  if (typeof window === 'undefined') return;

  window.__dreamEEG1 = {
    present:       !!(d.user1?.present),
    blink:         !!(d.user1?.blink),
    emotion:       _extractEmotion(d.user1, window.__dreamEEG1),
    ch_amplitudes: Array.isArray(d.user1?.ch_amplitudes) ? d.user1.ch_amplitudes : (window.__dreamEEG1?.ch_amplitudes ?? []),
  };

  window.__dreamEEG2 = {
    present:       !!(d.user2?.present),
    blink:         !!(d.user2?.blink),
    emotion:       _extractEmotion(d.user2, window.__dreamEEG2),
    ch_amplitudes: Array.isArray(d.user2?.ch_amplitudes) ? d.user2.ch_amplitudes : (window.__dreamEEG2?.ch_amplitudes ?? []),
  };

  window.__dreamSymbiosis = d.symbiosis
    ? { ...d.symbiosis }
    : { arousal: 0.5, valence: 0.5, score: 0.5, label: 'drifting' };

  const newActive = d.active_user || 1;
  window.__dreamActiveUser = newActive;

  // Backward-compat mirror
  window.__dreamEEG = newActive === 2 ? window.__dreamEEG2 : window.__dreamEEG1;

  // Fire permission-change event if active user switched
  if (newActive !== _prevActive) {
    _prevActive = newActive;
    window.dispatchEvent(
      new CustomEvent('dream-permission-change', { detail: { activeUser: newActive } })
    );
  }
}

function _markDead() {
  if (typeof window === 'undefined') return;
  window.__dreamEEG1       = _deadState();
  window.__dreamEEG2       = _deadState();
  window.__dreamEEG        = _deadState();
  window.__dreamSymbiosis  = { arousal: 0.5, valence: 0.5, score: 0, label: 'divergent' };
  window.__dreamActiveUser = 1;
}

// ── Stale detection ───────────────────────────────────────────────────────────

function _arm(staleMs) {
  if (_staleTimer) clearTimeout(_staleTimer);
  if (staleMs <= 0) return;
  _staleTimer = setTimeout(_markDead, staleMs);
}

// ── RAF loop — keeps window objects fresh every frame ────────────────────────

function _raf() {
  if (_lastPayload && _lastPayload.user1) _push(_lastPayload);
  _rafId = requestAnimationFrame(_raf);
}

// ── Blink capture ─────────────────────────────────────────────────────────────

function _fireBlink(user) {
  const now = Date.now();
  if (now - _lastCaptureAt < CAPTURE_DEBOUNCE) return;
  _lastCaptureAt = now;
  window.dispatchEvent(
    new CustomEvent('dream-blink-up', { detail: { source: 'eeg', capture: true, user } })
  );
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Connect to the dual-user EEG WebSocket server.
 *
 * @param {{ url?: string, staleMs?: number }} [options]
 * @returns {Promise<void>}
 */
export function startDualEEGBridge(options = {}) {
  const url     = options.url     || 'ws://127.0.0.1:8765';
  const staleMs = options.staleMs ?? STALE_MS;

  stopDualEEGBridge();

  return new Promise((resolve, reject) => {
    try {
      _ws = new WebSocket(url);
    } catch (e) {
      reject(e);
      return;
    }

    _ws.onopen = () => {
      if (_rafId == null) _rafId = requestAnimationFrame(_raf);
      _arm(staleMs);
      resolve();
    };

    _ws.onmessage = (ev) => {
      try {
        const d = typeof ev.data === 'string'
          ? JSON.parse(ev.data)
          : JSON.parse(String(ev.data));

        _lastPayload = d;
        _push(d);

        if (d.capture) {
          _fireBlink(d.active_user || 1);
        }

        _arm(staleMs);
      } catch (_) {
        /* ignore malformed frames */
      }
    };

    _ws.onerror = () => { _markDead(); };
    _ws.onclose = () => {
      if (_staleTimer) { clearTimeout(_staleTimer); _staleTimer = null; }
      _markDead();
    };
  });
}

export function stopDualEEGBridge() {
  if (_staleTimer) { clearTimeout(_staleTimer); _staleTimer = null; }
  if (_rafId != null) { cancelAnimationFrame(_rafId); _rafId = null; }
  if (_ws) {
    try { _ws.close(); } catch (_) {}
    _ws = null;
  }
  _lastPayload = {};
  _markDead();
}
