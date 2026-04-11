/**
 * Orbit camera rig — perspective camera + damped orbit controls.
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export function createOrbitRig(canvas) {
  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 600);
  camera.position.set(10, 7, 14);
  camera.lookAt(0, 0, 0);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.055;
  /** Wheel / pinch zoom (r161+ fixes r160 wheel scaling; tune here if needed). */
  controls.zoomSpeed = 2.1;
  controls.minDistance = 2;
  controls.maxDistance = 150;
  controls.maxPolarAngle = Math.PI / 2 - 0.04;

  return { camera, controls };
}
