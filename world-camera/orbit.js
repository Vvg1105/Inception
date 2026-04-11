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
  controls.zoomSpeed = 3.5;
  controls.panSpeed = 2.5;
  controls.rotateSpeed = 1.4;
  controls.keyPanSpeed = 30;
  controls.minDistance = 1;
  controls.maxDistance = 250;
  controls.maxPolarAngle = Math.PI / 2 - 0.04;

  return { camera, controls };
}
