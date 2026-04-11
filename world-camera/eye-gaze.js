/**
 * Webcam gaze → window.__dreamGaze (NDC + blinkingFromCamera), consumed by pointer-follow.js.
 * MediaPipe Face Landmarker — Apache-2.0 — https://developers.google.com/mediapipe
 *
 * Eye look: EYE_LOOK_* blendshapes. Blink hold (camera path): EYE_BLINK_* + hysteresis.
 *
 * For EEG-driven blink (no camera), the page reads window.__dreamEEG — see pointer-follow.js.
 */
const MP_PKG = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
const MP_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MP_MODEL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

let visionImportPromise = null;

function loadVisionModule() {
  if (!visionImportPromise) visionImportPromise = import(MP_PKG);
  return visionImportPromise;
}

function readGazeMix() {
  const el = document.getElementById('gaze-mix');
  if (!el) return 0.85;
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : 0.85;
}

/** One slider scales both X and Y gaze→cursor equally (world panel “gaze sens”). */
function readGazeSensitivity() {
  const el = document.getElementById('gaze-sens');
  if (!el) return 1;
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : 1;
}

function lerp2(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

/** If true, negate horizontal gaze→cursor. */
const FLIP_X = false;
/** Base scale before `gaze-sens` slider; same for horizontal and vertical. */
const GAIN_BASE = 1.6;
/** Lerp factor toward latest eye NDC each frame — higher = snappier (more jitter if too close to 1). */
const SMOOTH = 0.9;

/** MediaPipe EYE_BLINK_* → hysteresis “eyes closed” (for cursor hold when camera is the blink source). */
const BLINK_ON = 0.52;
const BLINK_OFF = 0.28;
let cameraBlinkClosed = false;

/** Normalize MediaPipe blendshape names to EYE_LOOK_IN_LEFT-style keys. */
function canonicalBlendshapeName(raw) {
  const s = (raw || '').trim();
  if (!s) return '';
  if (s.includes('_')) return s.toUpperCase();
  return s
    .replace(/([A-Z])/g, '_$1')
    .replace(/^_/, '')
    .toUpperCase();
}

/**
 * FaceLandmarker blendshape categories → { EYE_LOOK_IN_LEFT: score, ... }.
 */
function blendshapeScores(classifications) {
  const out = Object.create(null);
  if (!classifications || !classifications.categories) return out;
  for (const c of classifications.categories) {
    const name = canonicalBlendshapeName(c.categoryName || c.displayName || '');
    if (name) out[name] = typeof c.score === 'number' ? c.score : 0;
  }
  return out;
}

function getB(bs, key) {
  const v = bs[key];
  return typeof v === 'number' ? v : 0;
}

/**
 * Horizontal / vertical eye-look from ARKit-style weights (both eyes, conjugate gaze).
 * Looking viewer-right: left eye nasal (IN), right eye temporal (OUT).
 */
function eyeLookFromBlendshapes(bs) {
  const hx =
    (getB(bs, 'EYE_LOOK_IN_LEFT') -
      getB(bs, 'EYE_LOOK_OUT_LEFT') +
      (getB(bs, 'EYE_LOOK_OUT_RIGHT') - getB(bs, 'EYE_LOOK_IN_RIGHT'))) *
    0.5;
  const vy =
    (getB(bs, 'EYE_LOOK_UP_LEFT') -
      getB(bs, 'EYE_LOOK_DOWN_LEFT') +
      (getB(bs, 'EYE_LOOK_UP_RIGHT') - getB(bs, 'EYE_LOOK_DOWN_RIGHT'))) *
    0.5;
  return { hx, vy };
}

function updateCameraBlinkFromBlendshapes(bs) {
  const b = Math.max(getB(bs, 'EYE_BLINK_LEFT'), getB(bs, 'EYE_BLINK_RIGHT'));
  if (!cameraBlinkClosed && b > BLINK_ON) cameraBlinkClosed = true;
  else if (cameraBlinkClosed && b < BLINK_OFF) cameraBlinkClosed = false;
  return cameraBlinkClosed;
}

let gazeRafId = null;
let gazeBridgeRunning = false;
let smoothNdc = { x: 0, y: 0 };

let mediaStream = null;
let videoEl = null;
let faceLandmarker = null;

function applyNdcToDreamGaze(nx, ny, blinkingFromCamera) {
  window.__dreamGaze = {
    active: true,
    x: nx,
    y: ny,
    mix: readGazeMix(),
    blinkingFromCamera: !!blinkingFromCamera,
  };
}

function gazeLoop(ts) {
  if (!gazeBridgeRunning || !faceLandmarker || !videoEl) return;

  if (videoEl.readyState >= 2) {
    const result = faceLandmarker.detectForVideo(videoEl, ts);
    const bsRaw = result.faceBlendshapes && result.faceBlendshapes[0];
    if (bsRaw) {
      const bs = blendshapeScores(bsRaw);
      const eyesClosed = updateCameraBlinkFromBlendshapes(bs);
      if (!eyesClosed) {
        const { hx, vy } = eyeLookFromBlendshapes(bs);
        const sens = readGazeSensitivity();
        let nx = (FLIP_X ? -hx : hx) * GAIN_BASE * sens;
        let ny = vy * GAIN_BASE * sens;
        nx = Math.max(-1.35, Math.min(1.35, nx));
        ny = Math.max(-1.35, Math.min(1.35, ny));
        smoothNdc = lerp2(smoothNdc, { x: nx, y: ny }, SMOOTH);
      }
      applyNdcToDreamGaze(smoothNdc.x, smoothNdc.y, eyesClosed);
    }
  }

  gazeRafId = requestAnimationFrame(gazeLoop);
}

async function createFaceLandmarker(FilesetResolver, FaceLandmarker) {
  const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
  const base = {
    baseOptions: {
      modelAssetPath: MP_MODEL,
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
  };
  try {
    return await FaceLandmarker.createFromOptions(fileset, base);
  } catch (e) {
    console.warn('[eye-gaze] MediaPipe GPU init failed, trying CPU', e);
    return await FaceLandmarker.createFromOptions(fileset, {
      ...base,
      baseOptions: { ...base.baseOptions, delegate: 'CPU' },
    });
  }
}

/** Preload WASM / JS (optional). */
export async function loadWebgazerScript() {
  await loadVisionModule();
}

/**
 * Start webcam + MediaPipe eye blendshape gaze → window.__dreamGaze (blends with mouse in pointer-follow).
 */
export async function startEyeGazeBridge() {
  const { FilesetResolver, FaceLandmarker } = await loadVisionModule();

  if (gazeRafId != null) {
    cancelAnimationFrame(gazeRafId);
    gazeRafId = null;
  }
  stopEyeGazeBridgeInner();

  gazeBridgeRunning = true;
  cameraBlinkClosed = false;
  smoothNdc = { x: 0, y: 0 };
  window.__dreamGaze = { active: true, x: NaN, y: NaN, mix: readGazeMix(), blinkingFromCamera: false };

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });

    videoEl = document.createElement('video');
    videoEl.playsInline = true;
    videoEl.muted = true;
    videoEl.setAttribute('playsinline', '');
    videoEl.srcObject = mediaStream;
    await videoEl.play();

    faceLandmarker = await createFaceLandmarker(FilesetResolver, FaceLandmarker);

    gazeRafId = requestAnimationFrame(gazeLoop);
  } catch (e) {
    gazeBridgeRunning = false;
    stopEyeGazeBridgeInner();
    window.__dreamGaze = { active: false };
    throw e;
  }
}

function stopEyeGazeBridgeInner() {
  if (gazeRafId != null) {
    cancelAnimationFrame(gazeRafId);
    gazeRafId = null;
  }
  if (faceLandmarker) {
    try {
      faceLandmarker.close();
    } catch (e) {
      console.warn('[eye-gaze] close landmarker', e);
    }
    faceLandmarker = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (videoEl) {
    videoEl.srcObject = null;
    videoEl = null;
  }
}

/** Stop gaze updates and release camera. */
export function stopEyeGazeBridge() {
  gazeBridgeRunning = false;
  cameraBlinkClosed = false;
  stopEyeGazeBridgeInner();
  window.__dreamGaze = { active: false };
}
