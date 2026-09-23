import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { activityDisplay } from './brain-activity.js';
import { t, getLang, setLang, applyStatic, fmtNum } from './i18n.js';
import { trainingMetrics, trainingCapabilities, appendTrainingLog, speedError, finite } from './training-series.js';

const $ = (id) => document.getElementById(id);
const theme = getComputedStyle(document.documentElement);
const colorToken = (name) => theme.getPropertyValue(name).trim();
const chartColor = colorToken('--chart-line'), gaitColor = colorToken('--chart-gait');
/** Writes only on change: these run at SSE rate, and a write breaks focus and selection. */
const setText = (el, value) => { if (el && el.textContent !== value) el.textContent = value; };
const setHTML = (el, value) => { if (el && el.innerHTML !== value) el.innerHTML = value; };
/** Sets the value AND re-points data-i18n, so a later applyStatic() renders it in the new language. */
const setKey = (el, key, vars) => { if (!el) return; el.dataset.i18n = key; el.textContent = t(key, vars); };

let state = null, meta = null, lastEvent = 0, vrm = null, avatarRoot = null;
let lastEpisode = null, activeView = 'studio', skeletonMode = false, displayPose = null;
const gait = [], trailPoints = [];
let toastTimer, pendingTarget = null;
let trainingData = null, trainingPending = false, trainingFetching = false, metricData = [], trainingEnabled = null, trainingGeneration = 0;
let logData = { run_id: null, cursor: 0, text: '' }, logFetching = false;
let evaluationData = null, evaluationActive = false, evaluationFetching = false, evaluationPending = false, evaluationGeneration = 0;
const escapeHTML = value => String(value ?? '--').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const number = (value, digits = 3) => finite(value) ? value.toLocaleString(getLang(), { maximumFractionDigits: digits }) : '--';
const phaseText = phase => { const key = `chart.phase.${phase}`; return t(key) === key ? t('train.unknown', { phase }) : t(key); };
const reasonText = reason => { const key = `train.reason.${reason}`; return t(key) === key ? t('train.reason.other', { reason }) : t(key); };
function toast(message) { $('toast').textContent = message; $('toast').classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.remove('show'), 4200); }
async function post(url, data = {}) {
  const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
  if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(typeof error.detail === 'string' ? error.detail : t('error.badRequest')); }
  return response.json();
}
async function control(action, extras = {}) { try { await post('/api/control', { action, ...extras }); } catch (error) { toast(error.message); } }
async function setTarget(change) {
  pendingTarget = { speed: pendingTarget?.speed ?? state?.target_speed ?? 0.5,
    yaw: pendingTarget?.yaw ?? state?.target_yaw ?? 0, ...change };
  try { await post('/api/control', { action: 'target', ...pendingTarget }); }
  catch (error) { pendingTarget = null; toast(error.message); }
}

const VIEW_KEYS = { studio: 'nav.studio', results: 'nav.results', research: 'nav.research' };
function view(name) {
  activeView = name;
  document.querySelectorAll('.view').forEach(el => el.classList.toggle('active', el.id === `${name}View`));
  document.querySelectorAll('.nav').forEach(el => el.classList.toggle('active', el.dataset.view === name));
  setKey($('breadcrumb'), VIEW_KEYS[name] || 'nav.studio');
  if (name === 'results') { loadEvaluation(); trainingStatus(); evaluationStatus(); trainingLogs(); drawTrainingCharts(); }
}
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => view(button.dataset.view)));
$('brandHome').onclick = () => view('studio');
$('playButton').onclick = () => control(state?.running ? 'pause' : 'play');
$('resetButton').onclick = () => { control('reset'); toast(t('ctrl.reset.done')); };
$('pushButton').onclick = () => { control('push'); toast(t('ctrl.push.done')); };
$('targetSpeed').oninput = (e) => { $('targetLabel').innerHTML = `${Number(e.target.value).toFixed(2)} <small>m/s</small>`; };
$('targetSpeed').onchange = () => setTarget({ speed: +$('targetSpeed').value });
$('heading').oninput = (e) => { $('headingLabel').textContent = `${e.target.value}°`; };
$('heading').onchange = () => setTarget({ yaw: +$('heading').value * Math.PI / 180 });
$('lesion').onchange = (e) => { control('lesion', { lesion: e.target.value }); toast(t(e.target.value === 'intact' ? 'lesion.intact.done' : 'lesion.other.done')); };
async function trainingAction(action) {
  if (trainingPending) return;
  if (action === 'start' && !$('trainConfig').reportValidity()) return;
  if (action === 'stop' && !confirm(t('chart.train.stopConfirm'))) return;
  const allowed = trainingCapabilities(trainingData, trainingEnabled);
  if (!allowed[action]) return;
  trainingGeneration++; trainingPending = true; renderTraining(); $('trainError').hidden = true;
  try {
    const options = { worlds: +$('trainWorlds').value, batch: +$('trainBatch').value,
      max_seconds: +$('trainDuration').value, warm_start: $('trainWarmStart').checked };
    if (meta?.training_method === 'ppo') Object.assign(options, { minibatch: +$('trainMinibatch').value, steps: +$('trainSteps').value, freeze_brain: $('trainFreeze').checked });
    const result = await post(action === 'start' ? '/api/train' : `/api/train/${action}`, action === 'start' ? options : {});
    // Keep the action disabled until the authoritative status arrives, even if a stale poll was in flight.
    trainingData = { ...trainingData, ...result, active: true, lifecycle: { start: 'starting', pause: 'pause_requested', resume: 'training', stop: 'stopping' }[action], phase: { start: 'starting', pause: 'pause_requested', resume: 'training', stop: 'stopping' }[action], can_pause: false, can_resume: false, can_stop: false };
    if (action === 'start') { trainingData.history = []; metricData = trainingMetrics([], meta?.training_method); }
    toast(t(`train.sent.${action}`));
  } catch (error) { $('trainError').hidden = false; setText($('trainError'), error.message); }
  finally { trainingPending = false; renderTraining(); await trainingStatus(); }
}
$('trainButton').onclick = () => trainingAction('start');
$('trainPause').onclick = () => trainingAction('pause');
$('trainResume').onclick = () => trainingAction('resume');
$('trainStop').onclick = () => trainingAction('stop');
$('trainConfig').onsubmit = event => { event.preventDefault(); trainingAction('start'); };
$('evaluateButton').onclick = async () => {
  if (evaluationActive || evaluationPending) return;
  evaluationGeneration++; evaluationPending = true; $('evaluateButton').disabled = true; $('evaluationError').hidden = true;
  try { await post('/api/evaluate', { quick: $('evaluationProtocol').value === 'quick', seconds: 20 }); evaluationActive = true; $('evaluationDetails').open = true; }
  catch (error) { $('evaluationError').hidden = false; setText($('evaluationError'), error.message); }
  finally { evaluationPending = false; $('evaluateButton').disabled = evaluationActive; }
  // Fetch status only after the pending flag clears: evaluationStatus() skips
  // while a launch is pending, so calling it inside try was a no-op.
  evaluationStatus();
};
$('evaluationDetails').addEventListener('toggle', () => { $('evaluationContent').hidden = !$('evaluationDetails').open; });
window.addEventListener('keydown', (event) => {
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON', 'SUMMARY', 'A'].includes(event.target.tagName)) return;
  if (activeView !== 'studio') return;
  if (event.code === 'Space') { event.preventDefault(); control(state?.running ? 'pause' : 'play'); }
  else if (event.key === 'r' || event.key === 'R') control('reset');
});

// Body switch: the server swaps its one live body; a full reload is the smallest
// correct re-init (VRM loader and brain scene are not re-entrant).
async function switchBody(robot) {
  if (!meta || meta.body?.robot === robot) return;
  try {
    await post('/api/body', { robot });
    location.reload();
  } catch (error) { toast(error.message); }
}
$('bodyG1').onclick = () => switchBody('g1');
$('bodyYumi').onclick = () => switchBody('yumi');

const stageOverlay = $('stageOverlay');
function showStageOverlay(message) { if (!message) { stageOverlay.hidden = true; return; } stageOverlay.textContent = message; stageOverlay.hidden = false; }
/** The pill keeps its dot: only the label span is rewritten. */
function setBadge(key) { setHTML($('liveBadge'), `<i></i><span data-i18n="${key}">${t(key)}</span>`); }

/** Only the WebGL context can fail here; the rest of the scene graph is plain JS. */
function makeRenderer(maxPixelRatio) {
  try {
    const created = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    created.setPixelRatio(Math.min(devicePixelRatio, maxPixelRatio));
    return created;
  } catch (error) { console.error(error); return null; }
}
const viewport = $('viewport');
const renderer = makeRenderer(1.6);
if (renderer) {
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.18;
  viewport.appendChild(renderer.domElement);
}
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2('#edf2f8', 0.05);
const camera = new THREE.PerspectiveCamera(33, 1, 0.05, 200);
const controls = renderer ? new OrbitControls(camera, renderer.domElement) : null;
if (controls) {
  controls.enableDamping = true;
  controls.minDistance = 1.6; controls.maxDistance = 10; controls.maxPolarAngle = Math.PI * 0.47;
  controls.target.set(0, 0.9, 0);
}
camera.position.set(2.15, 1.65, 3.45);
const lastRoot = new THREE.Vector3(), rootTarget = new THREE.Vector3();
$('cameraReset').onclick = () => {
  if (!controls) return;
  controls.target.copy(rootTarget).add(new THREE.Vector3(0, 0.9, 0));
  camera.position.copy(rootTarget).add(new THREE.Vector3(2.15, 1.65, 3.45));
  controls.update();
};
const hemi = new THREE.HemisphereLight('#ffffff', '#8a9bb1', 1.1); scene.add(hemi);
const key = new THREE.DirectionalLight('#fff2e7', 1.8); key.position.set(2, 5, 3); key.castShadow = true;
key.shadow.mapSize.set(1024, 1024); key.shadow.camera.left = -3; key.shadow.camera.right = 3; key.shadow.camera.top = 3; key.shadow.camera.bottom = -3; key.shadow.normalBias = 0.015; key.shadow.bias = -0.0001;
scene.add(key, key.target);
const rim = new THREE.DirectionalLight('#dae8ff', 1.3); rim.position.set(-3, 2.5, -2); scene.add(rim);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({ color: '#dce4ef', roughness: 0.93, metalness: 0.05 }));
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; ground.position.y = -0.015; scene.add(ground);
const grid = new THREE.GridHelper(400, 400, '#b9c4d6', '#cdd7e4'); grid.position.y = -0.009; grid.material.transparent = true; grid.material.opacity = 0.55; scene.add(grid);
const laneMaterial = new THREE.LineBasicMaterial({ color: '#a3b1c6', transparent: true, opacity: 0.45 });
for (const x of [-1.1, 1.1]) { const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, 0.002, -180), new THREE.Vector3(x, 0.002, 180)]); scene.add(new THREE.Line(geo, laneMaterial)); }
const trailGeo = new THREE.BufferGeometry();
const trailBuffer = new Float32Array(1000 * 3);
trailGeo.setAttribute('position', new THREE.BufferAttribute(trailBuffer, 3));
trailGeo.setDrawRange(0, 0);
const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: '#7d6ba8', transparent: true, opacity: 0.55 })); scene.add(trail);
const footMarkers = ['#1f8a70', '#4a3aa7'].map(color => { const marker = new THREE.Mesh(new THREE.RingGeometry(0.07, 0.078, 36), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.75, side: THREE.DoubleSide })); marker.rotation.x = -Math.PI / 2; marker.position.y = 0.005; scene.add(marker); return marker; });
const skeleton = new THREE.Group(); scene.add(skeleton); skeleton.visible = false;
const jointMaterial = new THREE.MeshStandardMaterial({ color: '#7a5cc9', emissive: '#3d2c68', emissiveIntensity: 0.3, roughness: 0.35 });
const jointMeshes = Array.from({ length: 13 }, () => { const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.024, 12, 10), jointMaterial); skeleton.add(mesh); return mesh; });
const skeletonGeometry = new THREE.BufferGeometry();
const skeletonLines = new THREE.LineSegments(skeletonGeometry, new THREE.LineBasicMaterial({ color: '#0e7a63' })); skeleton.add(skeletonLines);
function setMode(isSkeleton) {
  skeletonMode = isSkeleton; skeleton.visible = isSkeleton;
  if (vrm) vrm.scene.visible = !isSkeleton;
  $('avatarMode').classList.toggle('selected', !isSkeleton);
  $('skeletonMode').classList.toggle('selected', isSkeleton);
  $('avatarMode').setAttribute('aria-pressed', String(!isSkeleton));
  $('skeletonMode').setAttribute('aria-pressed', String(isSkeleton));
}
$('avatarMode').onclick = () => setMode(false); $('skeletonMode').onclick = () => setMode(true);
let avatarScale = 1.16, hipsRest = 0.9, avatarAxis = 1, physicsHips = 0.793;
const loader = new GLTFLoader(); loader.register(parser => new VRMLoaderPlugin(parser));
/** A VRM may not map every humanoid bone; rotateBone() already tolerates that, these must too. */
const bone = (name) => (vrm ? vrm.humanoid.getNormalizedBoneNode(name) : null) || null;
let avatarEpoch = 0, requestedAvatar = null;
async function loadAvatar(body) {
  const epoch = ++avatarEpoch;
  requestedAvatar = `${body.robot}:${body.avatar_sha256}`;
  $('modelLoading').hidden = false;
  setText($('modelLoading'), t('stage.loadingPct', { pct: 0 }));
  if (avatarRoot) { scene.remove(avatarRoot); VRMUtils.deepDispose(avatarRoot); }
  vrm = avatarRoot = null;
  delete viewport.dataset.loadedBody;
  delete viewport.dataset.avatarSha256;
  setText(document.querySelector('.avatar-name'), '—');
  let gltf;
  try {
    if (!body.avatar_url || body.avatar_pending) {
      setText($('modelLoading'), t('stage.noAvatar'));
      setText(document.querySelector('.avatar-name'), body.name || body.robot);
      setText($('avatarCredit'), body.credit || '');
      setMode(true);
      return;
    }
    const response = await fetch(body.avatar_url, { cache: 'no-cache' });
    if (!response.ok) throw new Error(`Avatar HTTP ${response.status}`);
    const bytes = await response.arrayBuffer();
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(v => v.toString(16).padStart(2, '0')).join('');
    if (body.avatar_sha256 && digest !== body.avatar_sha256) throw new Error('Avatar checksum mismatch');
    if (epoch !== avatarEpoch) return;
    gltf = await loader.parseAsync(bytes, new URL('.', new URL(body.avatar_url, location.href)).href);
    if (epoch !== avatarEpoch) { VRMUtils.deepDispose(gltf.scene); return; }
    const loaded = gltf.userData.vrm;
    const hips = loaded?.humanoid.getNormalizedBoneNode('hips');
    if (!hips) throw new Error('VRM has no mapped hips bone');
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);
    VRMUtils.rotateVRM0(loaded);
    vrm = loaded;
    avatarAxis = vrm.meta.metaVersion === '0' ? -1 : 1;
    vrm.scene.traverse(node => { if (node.isMesh) { node.castShadow = true; node.receiveShadow = true; node.frustumCulled = false; } });
    physicsHips = body.hips_height || physicsHips;
    hipsRest = hips.position.y;
    avatarScale = hipsRest / physicsHips;
    scene.add(vrm.scene); avatarRoot = vrm.scene;
    viewport.dataset.loadedBody = body.robot;
    viewport.dataset.avatarSha256 = digest;
    viewport.dataset.avatarTitle = vrm.meta.name || vrm.meta.title || '';
    setText(document.querySelector('.avatar-name'), body.avatar);
    setText($('avatarCredit'), body.credit);
    $('modelLoading').hidden = true;
    setMode(skeletonMode);
  } catch (error) {
    if (gltf && avatarRoot !== gltf.scene) VRMUtils.deepDispose(gltf.scene);
    if (epoch !== avatarEpoch) return;
    setText($('modelLoading'), t('stage.loadFailed'));
    console.error(error);
  }
}
const qX = new THREE.Quaternion(), qY = new THREE.Quaternion(), qZ = new THREE.Quaternion();
const axisX = new THREE.Vector3(1, 0, 0), axisY = new THREE.Vector3(0, 1, 0), axisZ = new THREE.Vector3(0, 0, 1);
function rotateBone(name, pitch = 0, roll = 0, yaw = 0) {
  const target = bone(name); if (!target) return;
  // VRM0 faces -Z before rotateVRM0; conjugate local rotations into the +Z display frame.
  qX.setFromAxisAngle(axisX, pitch * avatarAxis); qZ.setFromAxisAngle(axisZ, roll * avatarAxis); qY.setFromAxisAngle(axisY, yaw);
  target.quaternion.copy(qX).multiply(qZ).multiply(qY);
}
function updateBody(dt) {
  if (!state?.qpos) return;
  if (!displayPose) displayPose = state.qpos.slice();
  const blend = state.running ? 1 - Math.exp(-25 * Math.max(dt, 0)) : 1;
  displayPose = displayPose.map((v, i) => v + (state.qpos[i] - v) * blend);
  const q = displayPose;
  rootTarget.set(q[1] * avatarScale, 0, q[0] * avatarScale);
  const delta = rootTarget.clone().sub(lastRoot);
  camera.position.add(delta); if (controls) controls.target.add(delta); lastRoot.copy(rootTarget);
  key.position.copy(rootTarget).add(new THREE.Vector3(2, 5, 3)); key.target.position.copy(rootTarget);
  ground.position.x = rootTarget.x; ground.position.z = rootTarget.z;
  const hips = bone('hips');
  if (vrm && hips) {
    avatarRoot.position.copy(rootTarget); avatarRoot.position.y = q[2] * avatarScale - hipsRest;
    rotateBone('hips');
    hips.quaternion.set(q[5] * avatarAxis, q[6], q[4] * avatarAxis, q[3]).normalize();
    for (let side = 0; side < 2; side++) {
      const name = side ? 'right' : 'left', i = 7 + side * 6;
      rotateBone(`${name}UpperLeg`, q[i], q[i + 1], q[i + 2]);
      rotateBone(`${name}LowerLeg`, q[i + 3]);
      rotateBone(`${name}Foot`, q[i + 4], q[i + 5]);
      // Upper arms visually follow measured leg angles; the physics proxy has fixed arms.
      rotateBone(`${name}UpperArm`, -q[i] * 0.48, side ? 1.38 : -1.38);
      rotateBone(`${name}LowerArm`, 0, 0, side ? 0.13 : -0.13);
    }
    vrm.update(dt);
    vrm.scene.updateMatrixWorld(true);
    for (let side = 0; side < 2; side++) {
      const foot = bone(side ? 'rightFoot' : 'leftFoot');
      if (!foot) { footMarkers[side].visible = false; continue; }
      const position = foot.getWorldPosition(new THREE.Vector3());
      footMarkers[side].position.set(position.x, 0.007, position.z);
      footMarkers[side].visible = state.contacts[side];
    }
  }
  if (state.joints && skeletonMode) {
    const points = state.joints.map(p => new THREE.Vector3(p[1] * avatarScale, p[2] * avatarScale, p[0] * avatarScale));
    points.forEach((p, i) => jointMeshes[i]?.position.copy(p));
    const links = [];
    for (let i = 1; i < points.length; i++) { const parent = i === 7 ? 0 : i - 1; links.push(points[parent], points[i]); }
    skeletonGeometry.setFromPoints(links);
    skeletonGeometry.computeBoundingSphere();
  }
}
const brainHost = $('brainViewport');
const brainRenderer = makeRenderer(1.5);
if (brainRenderer) brainHost.appendChild(brainRenderer.domElement);
const brainScene = new THREE.Scene(), brainCamera = new THREE.PerspectiveCamera(37, 1, 0.1, 100);
brainScene.background = new THREE.Color('#050811');
brainCamera.position.set(0.25, 0.6, 5.3); brainCamera.lookAt(0, 0, 0);
const brainGroup = new THREE.Group(); brainScene.add(brainGroup);
let geometry, brainCurrent = null, brainPrevious = null, brainView = 'change';
function initBrain(data) {
  const positions = data.sample_coords.map(c => new THREE.Vector3(c[0], -c[2], c[1]));
  const box = new THREE.Box3().setFromPoints(positions), center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const scale = 2.9 / Math.max(size.x, size.y, size.z);
  positions.forEach(p => p.sub(center).multiplyScalar(scale));
  geometry = new THREE.BufferGeometry().setFromPoints(positions);
  geometry.setAttribute('signal', new THREE.BufferAttribute(new Float32Array(positions.length), 1));
  const material = new THREE.ShaderMaterial({
    uniforms: {
      pixelRatio: { value: brainRenderer ? brainRenderer.getPixelRatio() : 1 },
      negativeColor: { value: new THREE.Color('#55baff') },
      positiveColor: { value: new THREE.Color('#ffb36b') },
    },
    vertexShader: `
      attribute float signal;
      uniform float pixelRatio;
      uniform vec3 negativeColor;
      uniform vec3 positiveColor;
      varying vec3 pointColor;
      varying float pointOpacity;
      void main() {
        float strength = sqrt(abs(signal));
        pointColor = mix(vec3(0.48, 0.48, 0.48), signal >= 0.0 ? positiveColor : negativeColor, strength);
        pointOpacity = 0.8 + 0.2 * strength;
        gl_PointSize = pixelRatio * (3.5 + 4.5 * strength);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 pointColor;
      varying float pointOpacity;
      void main() {
        float radius = length(gl_PointCoord - 0.5) * 2.0;
        if (radius > 1.0) discard;
        // Keep a visible core even when activity is zero; only the halo fades out.
        float glow = 1.0 - smoothstep(0.3, 1.0, radius);
        gl_FragColor = vec4(pointColor, pointOpacity * glow);
        #include <colorspace_fragment>
      }
    `,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  brainGroup.add(new THREE.Points(geometry, material)); brainGroup.rotation.set(0.12, -0.12, 0);
}
function updateBrain() {
  if (!geometry || !state?.activity) return;
  if (!brainCurrent || state.time !== brainCurrent.time || state.episode !== brainCurrent.episode) {
    brainPrevious = brainCurrent;
    brainCurrent = state;
  }
  const display = activityDisplay(brainCurrent, brainPrevious, brainView);
  if (geometry.attributes.signal.array.length !== display.signals.length) return;
  geometry.attributes.signal.array.set(display.signals);
  geometry.attributes.signal.needsUpdate = true;
  setText($('activityRms'), `${display.rms.toFixed(3)}${brainView === 'change' ? ' /s' : ''}`);
  setKey($('brainLegendLow'), brainView === 'change' ? 'brain.legend.down' : 'brain.legend.negative');
  setKey($('brainLegendHigh'), brainView === 'change' ? 'brain.legend.up' : 'brain.legend.positive');
  setKey($('brainScale'), brainView === 'change' ? 'brain.scale.change' : 'brain.scale.strength');
  setKey($('activityLabel'), brainView === 'change' ? 'brain.label.change' : 'brain.label.strength');
  setKey($('brainReadout'), state.running ? 'brain.readout.live' : 'brain.readout.paused');
}
for (const mode of ['change', 'strength']) $('brain' + mode).addEventListener('click', () => {
  brainView = mode;
  for (const option of ['change', 'strength']) $('brain' + option).setAttribute('aria-pressed', String(option === mode));
  updateBrain();
});
const resizer = new ResizeObserver(() => {
  if (renderer && viewport.clientWidth && viewport.clientHeight) { renderer.setSize(viewport.clientWidth, viewport.clientHeight); camera.aspect = viewport.clientWidth / viewport.clientHeight; camera.updateProjectionMatrix(); }
  if (brainRenderer && brainHost.clientWidth && brainHost.clientHeight) { brainRenderer.setSize(brainHost.clientWidth, brainHost.clientHeight); brainCamera.aspect = brainHost.clientWidth / brainHost.clientHeight; brainCamera.updateProjectionMatrix(); }
}); resizer.observe(viewport); resizer.observe(brainHost);
let previousFrame = performance.now();
if (renderer) renderer.setAnimationLoop(now => {
  const dt = Math.min((now - previousFrame) / 1000, 0.05); previousFrame = now;
  if (activeView === 'studio') {
    updateBody(state?.running && performance.now() - lastEvent < 1500 ? dt : 0);
    if (controls) controls.update();
    renderer.render(scene, camera);
    if (brainRenderer) brainRenderer.render(brainScene, brainCamera);
  }
});
if (!renderer) showStageOverlay(t('stage.noWebgl'));

function chart(canvas, series, { max = 1, min = 0, log = false, color = chartColor, xLabel = '', baseline = null } = {}) {
  const ratio = Math.min(devicePixelRatio, 2), rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  // Assigning width/height reallocates the backing store, so only do it when the box changed.
  const width = Math.round(rect.width * ratio), height = Math.round(rect.height * ratio);
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height);
  const w = rect.width, h = rect.height, left = 36, right = 8, top = 18, bottom = 24, pw = w - left - right, ph = h - top - bottom;
  ctx.font = '10px system-ui, "Segoe UI", sans-serif'; ctx.fillStyle = '#596b80'; ctx.lineWidth = 1;
  const gridCount = log ? 6 : 4;
  const span = max - min;
  for (let i = 0; i < gridCount; i++) {
    const y = top + i / (gridCount - 1) * ph;
    ctx.strokeStyle = '#e3eaf2'; ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - right, y); ctx.stroke();
    const label = log ? `10${['⁰', '⁻¹', '⁻²', '⁻³', '⁻⁴', '⁻⁵'][i]}` : (min + span * (1 - i / 3)).toFixed(1);
    ctx.fillText(label, 3, y + 3);
  }
  const yScale = (v) => top + (1 - (log ? (Math.log10(Math.max(v, 1e-5)) + 5) / 5 : (v - min) / (span || 1))) * ph;
  if (baseline !== null) { ctx.save(); ctx.strokeStyle = '#8b93a3'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(left, yScale(baseline)); ctx.lineTo(w - right, yScale(baseline)); ctx.stroke(); ctx.restore(); }
  if (series.length > 1) {
    const minX = series[0][0], maxX = series.at(-1)[0];
    const points = series.map(([x, y]) => [left + (x - minX) / Math.max(maxX - minX, 1) * pw, Math.max(top, Math.min(top + ph, yScale(y)))]);
    const gradient = ctx.createLinearGradient(0, top, 0, top + ph); gradient.addColorStop(0, `${color}33`); gradient.addColorStop(1, `${color}00`);
    ctx.beginPath(); ctx.moveTo(points[0][0], top + ph); points.forEach(p => ctx.lineTo(...p)); ctx.lineTo(points.at(-1)[0], top + ph); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();
    ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.strokeStyle = color; ctx.lineWidth = 1.7; ctx.stroke();
    if (series.length < 30) { points.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); }); }
    ctx.fillStyle = '#9aa0b4'; ctx.fillText(String(Math.round(minX)), left, h - 6); ctx.textAlign = 'right'; ctx.fillText(`${Math.round(maxX)} ${xLabel}`, w - right, h - 6); ctx.textAlign = 'left';
  } else { ctx.fillStyle = '#9aa0b4'; ctx.fillText(t('chart.waiting'), left + 14, top + ph / 2); }
}
function updateMetrics() {
  if (!state?.qpos) return;
  const speed = state.velocity[0];
  setText($('speed'), speed.toFixed(2));
  setText($('upright'), Math.max(0, state.upright * 100).toFixed(1));
  $('balanceBar').style.width = `${Math.max(0, Math.min(100, state.upright * 100))}%`;
  setKey($('balanceState'), state.fallen ? (state.running ? 'metric.upright.fallen' : 'metric.upright.fallenPaused') : 'metric.upright.measured');
  const gap = speed - state.target_speed;
  setKey($('speedState'), Math.abs(gap) <= 0.05 ? 'metric.speed.ontrack' : gap < 0 ? 'metric.speed.below' : 'metric.speed.above',
    { error: Math.abs(gap).toFixed(2) });
  setText($('distance'), state.distance.toFixed(2)); setText($('simTime'), state.time.toFixed(1));
  if (pendingTarget && Math.abs(state.target_speed - pendingTarget.speed) < 1e-5 && Math.abs(state.target_yaw - pendingTarget.yaw) < 1e-5) pendingTarget = null;
  if (!pendingTarget && document.activeElement !== $('targetSpeed')) {
    $('targetSpeed').value = state.target_speed;
    setHTML($('targetLabel'), `${state.target_speed.toFixed(2)} <small>m/s</small>`);
  }
  if (!pendingTarget && document.activeElement !== $('heading')) {
    $('heading').value = Math.round(state.target_yaw * 180 / Math.PI);
    setText($('headingLabel'), `${Math.round(state.target_yaw * 180 / Math.PI)}°`);
  }
  if (document.activeElement !== $('lesion')) $('lesion').value = state.lesion;
  setText($('brainMs'), `${state.brain_ms.toFixed(1)} ms`);
  setText($('outputRms'), state.output_rms.toFixed(5));
  setText($('rtf'), `${Math.min(state.real_time_factor, 99).toFixed(2)} ×`);
  setText($('actualHeading'), `${(state.yaw * 180 / Math.PI).toFixed(1)}°`);
  const playKey = state.running ? 'ctrl.pause' : 'ctrl.play';
  setHTML($('playButton'), `<span class="glyph" aria-hidden="true">${state.running ? 'Ⅱ' : '▶'}</span><span data-i18n="${playKey}">${t(playKey)}</span>`);
  $('playButton').setAttribute('aria-label', t(playKey));
  setHTML($('liveBadge'), `<i></i><span>${t(state.fallen ? (state.running ? 'stage.fallen' : 'stage.fallenPaused') : state.running ? 'stage.live' : 'stage.paused')}</span>`);
  setText($('checkpointHash'), `${t('footer.checkpoint')} ${state.checkpoint ? String(state.checkpoint).slice(0, 10) : '—'}`);
  if (state.checkpoint) $('checkpointHash').title = t('footer.checkpointTitle', { hash: state.checkpoint, updates: state.checkpoint_updates ?? '—' });
  ['leftFoot', 'rightFoot'].forEach((id, i) => $(id).classList.toggle('contact', state.contacts[i]));
  setText($('steps'), t('chart.steps', { left: state.foot_strikes[0], right: state.foot_strikes[1] }));
  if (lastEpisode !== state.episode) { gait.length = 0; trailPoints.length = 0; trailGeo.setDrawRange(0, 0); displayPose = null; lastEpisode = state.episode; }
  if (state.running && (!gait.length || state.time > gait.at(-1)[0])) {
    gait.push([state.time, speed]); while (gait.length && state.time - gait[0][0] > 12) gait.shift();
    const actualRoot = new THREE.Vector3(state.qpos[1] * avatarScale, 0.01, state.qpos[0] * avatarScale);
    if (!trailPoints.length || actualRoot.distanceTo(trailPoints.at(-1)) > 0.1) {
      trailPoints.push(actualRoot); if (trailPoints.length > 1000) trailPoints.shift();
      trailPoints.forEach((p, i) => p.toArray(trailBuffer, i * 3));
      trailGeo.attributes.position.needsUpdate = true; trailGeo.setDrawRange(0, trailPoints.length); trailGeo.computeBoundingSphere();
    }
  }
  const spark = gait.slice(-70).map(([, y], i, arr) => `${i ? 'L' : 'M'}${i / Math.max(arr.length - 1, 1) * 100},${25 - Math.max(0, Math.min(1, y)) * 22}`).join(' ');
  $('speedSpark').setAttribute('d', spark);
  chart($('gaitChart'), gait, { max: 0.9, color: gaitColor, xLabel: 's', baseline: state.target_speed });
}
function setConnection(ok) {
  document.querySelector('.connection').classList.toggle('lost', !ok);
  $('railLed').classList.toggle('down', !ok);
  $('railLabel').title = t(ok ? 'stage.feedback' : 'stage.feedbackLost');
  $('feedbackState').firstElementChild.classList.toggle('down', !ok);
  setKey($('feedbackState').lastElementChild, ok ? 'stage.feedback' : 'stage.feedbackLost');
}
let bodySyncing = false;
async function syncBodyIdentity() {
  if (bodySyncing || !meta || !state?.robot) return;
  const identity = `${state.robot}:${state.avatar_sha256}`;
  if (requestedAvatar === identity) return;
  bodySyncing = true;
  try {
    const response = await fetch('/api/meta', { cache: 'no-cache' });
    if (!response.ok) throw new Error('Body metadata unavailable');
    const next = await response.json();
    if (next.body.robot !== state.robot || next.body.avatar_sha256 !== state.avatar_sha256) return;
    meta = next; trainingEnabled = meta.training_enabled;
    displayPose = null; lastEpisode = null; pendingTarget = null;
    $('bodyG1').setAttribute('aria-pressed', String(meta.body.robot === 'g1'));
    $('bodyYumi').setAttribute('aria-pressed', String(meta.body.robot === 'yumi'));
    applyModeCopy();
    await loadAvatar(meta.body);
    trainingStatus();
    if (activeView === 'results') loadEvaluation();
  } catch (error) { console.error(error); }
  finally { bodySyncing = false; }
}
const events = new EventSource('/api/events');
events.onmessage = (event) => {
  state = JSON.parse(event.data); lastEvent = performance.now();
  syncBodyIdentity();
  if (!state.qpos && !state.error) return;
  if (state.error) { setKey($('connection'), 'connection.error'); setConnection(false); toast(state.error); return; }
  setKey($('connection'), meta?.mode === 'full_connectome' ? 'connection.full' : 'connection.local');
  setConnection(true);
  showStageOverlay(null);
  updateMetrics(); updateBrain();
};
events.onerror = () => { setKey($('connection'), 'connection.lost'); setConnection(false); setBadge('stage.stale'); };
setInterval(() => {
  if (lastEvent && performance.now() - lastEvent > 3000) { setKey($('connection'), 'connection.stale'); setConnection(false); setBadge('stage.stale'); }
}, 1000);

function renderTraining() {
  const data = trainingData, caps = trainingCapabilities(data, trainingEnabled, trainingPending);
  for (const [action, id] of Object.entries({ start: 'trainButton', pause: 'trainPause', resume: 'trainResume', stop: 'trainStop' })) $(id).disabled = !caps[action];
  $('trainFields').disabled = trainingPending || data?.active === true || trainingEnabled !== true;
  const method = meta?.training_method || (meta?.body?.robot === 'yumi' ? 'ppo' : 'dagger');
  document.querySelectorAll('.ppo-option').forEach(el => { el.hidden = method !== 'ppo'; el.querySelector('input').disabled = method !== 'ppo'; });
  $('trainBatchOption').hidden = method === 'ppo'; $('trainBatch').disabled = method === 'ppo';
  setText($('trainIdentity'), `${data?.robot || meta?.body?.robot || '--'} / ${(data?.algorithm || method).toUpperCase()}`);
  const reason = meta?.training_unavailable_reason || meta?.reason;
  setText($('trainAvailability'), trainingEnabled === false ? reasonText(reason || 'unavailable') : t('train.controlHint'));
  $('trainWarmStart').disabled = meta?.training_checkpoint_available === false;
  if (meta?.training_checkpoint_available === false) $('trainWarmStart').checked = false;
  if (!data) return;
  const phase = data.lifecycle || data.phase || 'not_started';
  setText($('trainPhase'), phaseText(phase));
  $('trainPhase').removeAttribute('data-i18n');
  $('trainPhase').dataset.failed = String(['failed', 'timed_out'].includes(phase));
  const phaseDetail = data.lifecycle && data.phase !== data.lifecycle ? ` · ${phaseText(data.phase)}` : '';
  setText($('trainRun'), `${t('train.run')} ${data.run_id || '--'}${phaseDetail}`);
  setText($('trainStats'), [
    `${t('train.activeTime')} ${number(data.active_elapsed, 0)} / ${number(data.max_seconds, 0)} s`,
    `${t('train.pausedTime')} ${number(data.paused_seconds, 0)} s`,
    finite(data.iteration) ? t('chart.train.statIters', { count: number(data.iteration, 0) }) : '',
    finite(data.update) ? t('chart.train.statUpdates', { count: number(data.update, 0) }) : '',
    finite(data.peak_vram_gib) ? t('chart.train.statVram', { gib: number(data.peak_vram_gib, 2) }) : '',
  ].filter(Boolean).join(' · '));
  if (finite(data.active_elapsed) && finite(data.max_seconds) && data.max_seconds > 0) $('trainProgress').value = Math.min(100, 100 * data.active_elapsed / data.max_seconds);
  else $('trainProgress').value = 0;
  if (data.error) { $('trainError').hidden = false; setText($('trainError'), String(data.error)); }
  renderTrainingCharts();
}
async function trainingStatus() {
  if (trainingFetching || trainingPending) return;
  trainingFetching = true;
  const generation = trainingGeneration;
  try {
    const response = await fetch('/api/training'); if (!response.ok) throw new Error(t('chart.phase.unavailable'));
    const data = await response.json();
    if (trainingPending || generation !== trainingGeneration) return;
    trainingData = data; metricData = trainingMetrics(data.history, data.algorithm || meta?.training_method);
    renderTraining();
  } catch (error) {
    setText($('trainPhase'), t('chart.phase.unavailable')); $('trainPhase').dataset.failed = 'true';
    trainingData = null; for (const id of ['trainButton', 'trainPause', 'trainResume', 'trainStop']) $(id).disabled = true;
  } finally { trainingFetching = false; }
}
async function trainingLogs() {
  if (activeView !== 'results' || document.hidden || logFetching || !trainingData) return;
  logFetching = true;
  const run = trainingData.run_id || '';
  try {
    const query = new URLSearchParams({ cursor: String(logData.run_id === run ? logData.cursor ?? 0 : 0), run_id: run });
    const response = await fetch(`/api/train/logs?${query}`);
    if (!response.ok) throw new Error(t('train.logsFailed'));
    const incoming = await response.json();
    if (run !== (trainingData?.run_id || '')) return;
    logData = appendTrainingLog(logData, incoming);
    setText($('trainLogs'), logData.text || t('train.logsEmpty'));
    if ($('trainFollow').checked) $('trainLogs').scrollTop = $('trainLogs').scrollHeight;
    setText($('trainLogState'), t('train.logsLive'));
  } catch (error) { setText($('trainLogState'), t('train.logsFailed')); }
  finally { logFetching = false; }
}
function renderTrainingCharts() {
  const signature = metricData.map(m => m.key).join(',');
  const host = $('trainingCharts');
  if (host.dataset.metrics !== signature) {
    host.dataset.metrics = signature;
    setHTML(host, metricData.map(m => `<figure class="metric-chart"><figcaption><span id="metric-title-${m.key}"></span><b id="metric-value-${m.key}">--</b></figcaption><canvas id="metric-${m.key}" tabindex="0" role="img" aria-describedby="metric-tip-${m.key}"></canvas><output id="metric-tip-${m.key}" class="chart-readout"></output></figure>`).join(''));
    metricData.forEach(metric => {
      const canvas = $(`metric-${metric.key}`);
      canvas.onpointermove = event => { const rect = canvas.getBoundingClientRect(); canvas.hoverRatio = Math.max(0, Math.min(1, (event.clientX - rect.left - 54) / Math.max(rect.width - 70, 1))); drawTrainingCharts(); };
      canvas.onpointerleave = () => { canvas.hoverRatio = null; drawTrainingCharts(); };
      canvas.onfocus = () => { canvas.hoverRatio = 1; drawTrainingCharts(); };
      canvas.onkeydown = event => { if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return; event.preventDefault(); const current = metricData.find(m => m.key === metric.key); canvas.hoverRatio = event.key === 'Home' ? 0 : event.key === 'End' ? 1 : Math.max(0, Math.min(1, (canvas.hoverRatio ?? 1) + (event.key === 'ArrowLeft' ? -1 : 1) / Math.max(current.points.length - 1, 1))); drawTrainingCharts(); };
      canvas.onblur = () => { canvas.hoverRatio = null; drawTrainingCharts(); };
    });
  }
  for (const metric of metricData) {
    setText($(`metric-title-${metric.key}`), t(`train.metric.${metric.key}`));
    setText($(`metric-value-${metric.key}`), number(metric.points.at(-1)?.[1], 5));
    $(`metric-${metric.key}`).setAttribute('aria-label', `${t(`train.metric.${metric.key}`)}; ${t('train.chartKeys')}`);
  }
  if ($('trainingTableDetails').open) renderTrainingTable();
  drawTrainingCharts();
}
function renderTrainingTable() {
  setHTML($('trainingTable'), metricData.map(m => `<table><caption>${t(`train.metric.${m.key}`)}</caption><thead><tr><th>${t(`chart.axis.${m.axis}`)}</th><th>${t('train.value')}</th><th>${t('train.note')}</th></tr></thead><tbody>${m.points.map(([x, y, warmup]) => `<tr><td>${number(x, 0)}</td><td>${number(y, 6)}</td><td>${warmup ? t('train.warmup') : '--'}</td></tr>`).join('')}</tbody></table>`).join(''));
}
$('trainingTableDetails').addEventListener('toggle', () => { if ($('trainingTableDetails').open) renderTrainingTable(); });
function drawTrainingCharts() {
  if (activeView !== 'results') return;
  for (const metric of metricData) {
    const canvas = $(`metric-${metric.key}`); if (!canvas) continue;
    const rect = canvas.getBoundingClientRect(); if (!rect.width) continue;
    const ratio = Math.min(devicePixelRatio, 2);
    if (canvas.width !== Math.round(rect.width * ratio) || canvas.height !== Math.round(rect.height * ratio)) { canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio); }
    const ctx = canvas.getContext('2d'); if (!ctx) continue;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height);
    const left = 54, top = 14, width = rect.width - 70, height = rect.height - 43;
    const points = metric.points, real = points.filter(([, y]) => finite(y));
    const tip = $(`metric-tip-${metric.key}`);
    if (!real.length) { setText(tip, t('chart.waiting')); continue; }
    let minY = Math.min(...real.map(p => p[1])), maxY = Math.max(...real.map(p => p[1]));
    const padding = (maxY - minY || Math.abs(maxY) || 1) * .1; minY -= padding; maxY += padding;
    const minX = points[0][0], maxX = points.at(-1)[0], xSpan = maxX - minX || 1;
    const xAt = x => left + (x - minX) / xSpan * width, yAt = y => top + (maxY - y) / (maxY - minY) * height;
    ctx.font = '10px system-ui'; ctx.lineWidth = 1;
    for (let i = 0; i < 4; i++) { const y = top + height * i / 3; ctx.strokeStyle = colorToken('--line'); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + width, y); ctx.stroke(); ctx.fillStyle = colorToken('--muted'); ctx.textAlign = 'right'; const value = maxY - (maxY - minY) * i / 3; ctx.fillText(Math.abs(value) > 0 && (Math.abs(value) < .001 || Math.abs(value) >= 1e5) ? value.toExponential(1) : number(value, 3), left - 6, y + 3); }
    ctx.textAlign = 'left'; ctx.fillText(number(minX, 0), left, rect.height - 8); ctx.textAlign = 'right'; ctx.fillText(`${number(maxX, 0)} ${t(`chart.axis.${metric.axis}`)}`, left + width, rect.height - 8);
    ctx.strokeStyle = chartColor; ctx.lineWidth = 2; ctx.beginPath(); let connected = false;
    for (const [x, y] of points) { if (!finite(y)) { connected = false; continue; } if (connected) ctx.lineTo(xAt(x), yAt(y)); else ctx.moveTo(xAt(x), yAt(y)); connected = true; } ctx.stroke();
    if (real.length < 20) for (const [x, y, warmup] of real) { ctx.beginPath(); ctx.arc(xAt(x), yAt(y), 4, 0, Math.PI * 2); ctx.fillStyle = warmup ? colorToken('--panel') : chartColor; ctx.fill(); ctx.stroke(); }
    const target = canvas.hoverRatio == null ? maxX : minX + canvas.hoverRatio * (maxX - minX);
    const point = points.reduce((a, b) => Math.abs(b[0] - target) < Math.abs(a[0] - target) ? b : a);
    if (canvas.hoverRatio != null) { ctx.strokeStyle = colorToken('--muted'); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(xAt(point[0]), top); ctx.lineTo(xAt(point[0]), top + height); ctx.stroke(); ctx.setLineDash([]); }
    setText(tip, `${t(`chart.axis.${metric.axis}`)} ${number(point[0], 0)} · ${number(point[1], 6)}${point[2] ? ` · ${t('train.warmup')}` : ''}`);
  }
}
async function loadEvaluation() {
  const generation = evaluationGeneration;
  let response;
  try { response = await fetch('/api/evaluation'); if (generation !== evaluationGeneration) return; }
  catch (error) { console.error(error); setEvaluationState('failed', t('results.failed')); return; }
  // A 404 means the server has no evaluation matching the loaded checkpoint yet,
  // which is a normal state for a freshly trained checkpoint — not a read failure.
  if (!response.ok) {
    setEvaluationState(response.status === 404 ? 'absent' : 'failed', t(response.status === 404 ? 'eval.absent' : 'results.failed'));
    return;
  }
  try { evaluationData = await response.json(); if (generation !== evaluationGeneration) return; renderEvaluation(); }
  catch (error) { console.error(error); setEvaluationState('failed', t('results.failed')); }
}
/** `state` is 'absent' | 'failed': an empty evaluation collapses quietly, a real
    failure stays visible even while the details stay collapsed. */
function setEvaluationState(state, message) {
  evaluationData = null;
  setText($('evaluationState'), evaluationActive && state === 'absent' ? t('eval.running') : message);
  $('evaluationState').dataset.failed = String(state === 'failed');
  $('evaluationError').hidden = state !== 'failed';
  if (state === 'failed') setText($('evaluationError'), message);
  $('evaluationContent').hidden = true;
  setHTML($('evaluationSummary'), ''); setText($('evaluationRows'), '');
}
function renderEvaluation() {
  const data = evaluationData; if (!data) return setEvaluationState('absent', t('results.pending'));
  // While a job is active the phase pill is owned by evaluationStatus(); the report's
  // own state only shows when nothing is running, so the pill never flickers.
  if (!evaluationActive) {
    setText($('evaluationState'), t(data.complete === false ? 'eval.running' : 'eval.done'));
    $('evaluationState').dataset.failed = 'false';
  }
  $('evaluationError').hidden = true;
  $('evaluationContent').hidden = !$('evaluationDetails').open;
  const tests = data.tests || [], longWalks = data.long_walks || [], perturbations = data.perturbations || [];
  const controls = data.controls || {};
  const disconnected = Array.isArray(controls) ? controls : controls.disconnected || [];
  const rewired = Array.isArray(controls) ? [] : controls.rewired || [];
  setHTML($('evaluationSummary'),
    `<div><b>${escapeHTML(data.successes ?? '--')} / ${escapeHTML(data.attempts ?? '--')}</b><span>${t('results.summary.walk')}${data.complete === false ? t('results.summary.inProgress') : ''}</span></div>`
    + `<div><b>${escapeHTML(data.robot || '--')}</b><span>${t('eval.robot')}</span></div>`
    + `<div><b title="${escapeHTML(data.checkpoint_sha256 || '')}">${escapeHTML((data.checkpoint_sha256 || '').slice(0, 10) || '--')}</b><span>${t('eval.checkpoint')}</span></div>`);
  const strictGait = data.kind === 'held_out_locomotion';
  setKey($('resultsIntro'), strictGait ? 'results.pGait' : data.kind === 'held_out_native_mujoco' ? 'results.pFull' : meta?.mode === 'full_connectome' ? 'results.pFullScreening' : 'results.p');
  if (strictGait) setText($('resultsCriteria'), t('results.microGait', { lateral: number(data.long_lateral_m, 2), recovered: data.yaw_recovered, total: data.yaw_tests?.length || 0 }));
  else setKey($('resultsCriteria'), data.kind === 'held_out_native_mujoco' ? 'results.microFull' : meta?.mode === 'full_connectome' ? 'results.microFullScreening' : 'results.micro');
  const groups = [
    { key: 'walk', rows: tests, expect: 'pass' },
    { key: 'yaw', rows: data.yaw_tests || [], expect: 'pass' },
    { key: 'long', rows: longWalks, expect: 'pass' },
    { key: 'push', rows: perturbations, expect: 'pass' },
    { key: 'disconnected', rows: disconnected, expect: 'fall' },
    { key: 'rewired', rows: rewired, expect: 'fall' },
  ].filter(g => g.rows.length);
  const failed = ['survived', 'speed', 'both_feet', 'alternation', 'upright', 'direction', ...(strictGait ? ['swing'] : [])];
  setHTML($('evaluationRows'), groups.map(g => {
    const rows = g.rows.map(r => {
      const reason = r.criteria ? failed.filter(k => !r.criteria[k]).map(k => t(`results.fail.${k}`)).join(' / ') : t('results.noTarget');
      const ok = r.success || (g.expect === 'fall' && r.fallen);
      const cls = r.success ? 'pass' : ok ? 'neutral' : 'fail';
      const error = speedError(r);
      return `<tr><td>${escapeHTML(r.seed)} / ${escapeHTML(r.condition || t(`results.condition.${g.key}`))}</td><td>${number(r.target_speed, 2)} m/s</td>`
        + `<td>${number(r.mean_speed, 3)} m/s</td><td>${number(error, 3)}</td>`
        + `<td>${number(r.seconds, 1)} s</td><td>${number(r.foot_strikes?.[0], 0)} / ${number(r.foot_strikes?.[1], 0)}</td>`
        + `<td>${number(r.alternations, 0)}</td><td>${(r.contact_fraction || []).map(f => number(f, 2)).join(' / ') || '--'}</td>`
        + `<td>${number(r.minimum_upright, 2)}</td><td>${number(r.inference_ms_median, 2)} / ${number(r.inference_ms_p95, 2)} ms</td>`
        + `<td class="${cls}">${r.success ? t('results.pass') : r.fallen ? (g.expect === 'fall' ? t('results.expect.fall') : t('results.fallen')) : reason}</td></tr>`;
    }).join('');
    return `<tr class="group-row"><td colspan="11">${t(`results.group.${g.key}`)} · ${g.rows.filter(r => g.expect === 'fall' ? r.fallen : r.success).length} / ${g.rows.length}</td></tr>${rows}`;
  }).join(''));
  if (!groups.length) setHTML($('evaluationRows'), `<tr><td colspan="11">${t('results.pending')}</td></tr>`);
}
async function evaluationStatus() {
  if (evaluationFetching || evaluationPending) return;
  const generation = evaluationGeneration;
  evaluationFetching = true;
  try {
    const response = await fetch('/api/evaluate'); if (!response.ok) throw new Error(t('eval.statusFailed'));
    const job = await response.json();
    if (generation !== evaluationGeneration) return;
    const wasActive = evaluationActive;
    evaluationActive = job.active === true;
    $('evaluateButton').disabled = evaluationActive;
    $('evaluationJob').hidden = !evaluationActive;
    if (evaluationActive) {
      setText($('evaluationState'), phaseText(job.phase || 'evaluating'));
      setText($('evaluationJob'), `${phaseText(job.phase || 'evaluating')} · ${number(job.completed, 0)} / ${number(job.total, 0)} · ${job.robot || '--'} · ${String(job.checkpoint_sha256 || '--').slice(0, 10)}`);
      await loadEvaluation();
    } else if (wasActive) { await loadEvaluation(); }
    if (job.error || job.phase === 'failed') {
      $('evaluationError').hidden = false; setText($('evaluationError'), String(job.error || t('eval.failed')));
      setText($('evaluationState'), t('eval.failed')); $('evaluationState').dataset.failed = 'true';
    }
  } catch { $('evaluationError').hidden = false; setText($('evaluationError'), t('eval.statusFailed')); }
  finally { evaluationFetching = false; }
}

/** Everything below only differs between the layered prototype and the full connectome.
    Capabilities come from the server's probed hardware, never from a deployment label. */
function applyModeCopy() {
  if (!meta) return;
  const robot = meta.body?.robot === 'yumi' ? 'yumi' : 'g1';
  const bodyName = t(`stage.bodyName.${robot}`);
  setKey($('stageMeta'), 'stage.meta', { body: bodyName, joints: 12 });
  setKey($('integrityText'), 'integrity.text');
  setKey($('footerMode'), 'footer.b');
  setKey($('researchContext'), meta.mode === 'full_connectome' ? 'study.context.full' : 'study.context.legacy', {
    body: robot === 'yumi' ? 'Yumi' : 'G1' });
  const hardware = meta.hardware;
  if (hardware) setText($('hardwareText'), t('research.hw.text', {
    cpu: hardware.cpu, gpu: hardware.gpu ? t('research.hw.gpu', { gpu: hardware.gpu }) : '',
    total: hardware.ram_total_gb, available: hardware.ram_available_gb, rss: hardware.process_rss_mb, engine: hardware.engine }));
  setKey($('hardwareTitle'), 'research.hw.title');
  setKey($('hardwareNote'), 'research.hw.note');
  if (meta.body?.avatar_url && meta.body.avatar_pending) document.querySelector('.avatar-name').textContent = t('stage.waitingYumi');
  if (meta.mode !== 'full_connectome') return;

  setText($('datasetChip'), meta.dataset || '—');
  setKey($('footerMode'), 'footer.bFull');
  setKey($('researchReport'), 'research.studio.a');
  setKey($('trainButton'), 'train.start');
  setKey($('integrityText'), 'integrity.textFull', { body: bodyName });
  $('lesion').querySelector('[value="rewired"]')?.remove();
}

/** The button shows the language it switches TO, which is the action, not the state. */
function updateLanguageToggle() { setText($('langCurrent'), getLang() === 'zh' ? 'EN' : '中文'); }
function setLanguage(next) {
  setLang(next);
  updateLanguageToggle();
  applyStatic();
  applyModeCopy();
  if (activeView === 'results') loadEvaluation();
  trainingStatus();
  if (state) { updateMetrics(); updateBrain(); }
}
$('languageToggle').onclick = () => setLanguage(getLang() === 'zh' ? 'en' : 'zh');

try {
  const response = await fetch('/api/meta'); if (!response.ok) throw new Error('connectome metadata unavailable');
  meta = await response.json();
  physicsHips = meta.body?.hips_height || 0.793;
  loadAvatar(meta.body);
  setText($('datasetChip'), meta.dataset || '—');
  setText($('railLabel'), 'SERVER');
  setText($('neuronCount'), fmtNum(meta.neurons));
  setText($('edgeCount'), `${(meta.edges / 1e6).toFixed(2)} M`);
  setText($('sampleCount'), t('brain.sample', { shown: fmtNum(meta.sample_indices.length), total: fmtNum(meta.neurons) }));
  const hardware = meta.hardware;
  setText($('hardwareText'), t('research.hw.text', {
    cpu: hardware.cpu, gpu: hardware.gpu ? t('research.hw.gpu', { gpu: hardware.gpu }) : '',
    total: hardware.ram_total_gb, available: hardware.ram_available_gb, rss: hardware.process_rss_mb, engine: hardware.engine }));
  initBrain(meta);
  trainingEnabled = meta.training_enabled;
  applyStatic();
  applyModeCopy();
} catch (error) { console.error(error); toast(t('error.meta')); }
updateLanguageToggle();

// Body switcher visibility follows meta: local studio can hot-swap, cloud cannot.
if (meta?.body_switch) $('bodySwitch').hidden = false;
const currentRobot = meta?.body?.robot === 'yumi' ? 'yumi' : 'g1';
$('bodyG1').setAttribute('aria-pressed', String(currentRobot === 'g1'));
$('bodyYumi').setAttribute('aria-pressed', String(currentRobot === 'yumi'));

await trainingStatus();
setInterval(() => { if (activeView === 'results' && !document.hidden) trainingStatus(); }, 2500);
setInterval(() => { if (activeView === 'results' && !document.hidden) { trainingLogs(); evaluationStatus(); } }, 1000);
document.addEventListener('visibilitychange', () => { if (!document.hidden && activeView === 'results') { trainingStatus(); trainingLogs(); evaluationStatus(); } });
window.addEventListener('resize', drawTrainingCharts);
document.querySelector('.studio-gait').addEventListener('toggle', () => { if (state) chart($('gaitChart'), gait, { max: 0.9, color: gaitColor, xLabel: 's', baseline: state.target_speed }); });
