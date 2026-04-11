/**
 * Ground cursor: mouse NDC + optional window.__dreamGaze + gamepad,
 * smoothed onto the ground plane (point light + disc).
 */
import * as THREE from 'three';

export function createPointerGroundFollow({ scene, camera, ground, raycaster, canvas }) {
  const cursorLight = new THREE.PointLight(0xffffff, 2.6, 18, 2);
  cursorLight.position.set(0, 0.6, 0);
  scene.add(cursorLight);

  const cursorDisc = new THREE.Mesh(
    new THREE.CircleGeometry(0.18, 32),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.32 })
  );
  cursorDisc.rotation.x = -Math.PI / 2;
  cursorDisc.position.y = 0.003;
  scene.add(cursorDisc);

  const pointerNDC = new THREE.Vector2();
  const ndcBlended = new THREE.Vector2();
  const cursorGroundTarget = new THREE.Vector3();
  const cursorGroundSmooth = new THREE.Vector3();
  let cursorGroundReady = false;

  /** Ground position frozen while blink-hold is active. */
  let blinkHoldPos = null;
  /** Previous frame had blink-hold engaged (for edge events). */
  let prevBlinkHold = false;
  let lastBlinkSourceForEvent = 'camera';

  function readBlinkPauseEnabled() {
    const el = document.getElementById('blink-pause-cursor');
    return el ? el.checked : false;
  }

  function readBlinkSourceMode() {
    const el = document.getElementById('blink-source');
    return (el && el.value) || 'auto';
  }

  /**
   * Which signal drives “eyes closed”: EEG if configured, else MediaPipe camera.
   * window.__dreamEEG = { present: true, blink: true } from your EEG bridge each frame.
   */
  function resolveBlinkingHold() {
    const mode = readBlinkSourceMode();
    const eeg = typeof window !== 'undefined' ? window.__dreamEEG : null;
    const gz = typeof window !== 'undefined' ? window.__dreamGaze : null;
    if (mode === 'eeg') {
      return !!(eeg && eeg.present && eeg.blink);
    }
    if (mode === 'camera') {
      return !!(gz && gz.blinkingFromCamera);
    }
    if (eeg && eeg.present) return !!eeg.blink;
    return !!(gz && gz.blinkingFromCamera);
  }

  function activeBlinkSignalLabel() {
    const mode = readBlinkSourceMode();
    const eeg = typeof window !== 'undefined' ? window.__dreamEEG : null;
    if (mode === 'eeg') return eeg && eeg.present ? 'eeg' : 'none';
    if (mode === 'camera') return 'camera';
    if (eeg && eeg.present) return 'eeg';
    return 'camera';
  }

  canvas.addEventListener('mousemove', (e) => {
    pointerNDC.x = (e.clientX / window.innerWidth) * 2 - 1;
    pointerNDC.y = -(e.clientY / window.innerHeight) * 2 + 1;
  });

  function updatePointerFollow(dt) {
    const pauseOn = readBlinkPauseEnabled();
    const blinkNow = pauseOn && resolveBlinkingHold();

    if (pauseOn) {
      if (!prevBlinkHold && blinkNow) {
        lastBlinkSourceForEvent = activeBlinkSignalLabel();
        if (typeof window !== 'undefined') {
          window.dispatchEvent(
            new CustomEvent('dream-blink-down', { detail: { source: lastBlinkSourceForEvent } })
          );
        }
      }
      if (prevBlinkHold && !blinkNow) {
        if (typeof window !== 'undefined') {
          window.dispatchEvent(
            new CustomEvent('dream-blink-up', { detail: { source: lastBlinkSourceForEvent } })
          );
        }
      }
    }
    prevBlinkHold = !!blinkNow;
    if (!pauseOn) prevBlinkHold = false;

    if (blinkNow) {
      if (!blinkHoldPos && cursorGroundReady) {
        blinkHoldPos = cursorGroundSmooth.clone();
      }
      if (blinkHoldPos) {
        cursorLight.position.set(blinkHoldPos.x, 0.6, blinkHoldPos.z);
        cursorDisc.position.set(blinkHoldPos.x, 0.003, blinkHoldPos.z);
      }
      return;
    }
    blinkHoldPos = null;

    ndcBlended.copy(pointerNDC);
    const gz = typeof window !== 'undefined' ? window.__dreamGaze : null;
    // Blend when gaze NDC is valid; do not require gz.active (avoids flicker when gaze drops frames).
    if (gz && Number.isFinite(gz.x) && Number.isFinite(gz.y)) {
      const mix = gz.mix != null ? gz.mix : 0.78;
      ndcBlended.x = THREE.MathUtils.lerp(ndcBlended.x, gz.x, mix);
      ndcBlended.y = THREE.MathUtils.lerp(ndcBlended.y, gz.y, mix);
    }
    const gpEl = document.getElementById('gamepad-nudge');
    const gScale = gpEl ? parseFloat(gpEl.value) : 0.4;
    const gp = navigator.getGamepads ? navigator.getGamepads()[0] : null;
    if (gp && gp.axes && gp.axes.length >= 2) {
      const s = gScale * 0.13;
      ndcBlended.x += gp.axes[0] * s;
      ndcBlended.y -= gp.axes[1] * s;
    }
    ndcBlended.x = THREE.MathUtils.clamp(ndcBlended.x, -1.4, 1.4);
    ndcBlended.y = THREE.MathUtils.clamp(ndcBlended.y, -1.4, 1.4);

    raycaster.setFromCamera(ndcBlended, camera);
    const hits = raycaster.intersectObject(ground, false);
    if (hits.length === 0) return;
    cursorGroundTarget.copy(hits[0].point);
    if (!cursorGroundReady) {
      cursorGroundSmooth.copy(cursorGroundTarget);
      cursorGroundReady = true;
    }
    const inertiaEl = document.getElementById('cursor-inertia');
    const inertia = inertiaEl ? parseFloat(inertiaEl.value) : 0.72;
    const alpha = 1 - Math.exp(-(5.2 + inertia * 42) * Math.min(Math.max(dt, 0.001), 0.05));
    cursorGroundSmooth.lerp(cursorGroundTarget, alpha);

    cursorLight.position.set(cursorGroundSmooth.x, 0.6, cursorGroundSmooth.z);
    cursorDisc.position.set(cursorGroundSmooth.x, 0.003, cursorGroundSmooth.z);
  }

  return {
    cursorLight,
    cursorDisc,
    updatePointerFollow,
    pointerNDC,
  };
}
