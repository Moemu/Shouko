// Standalone browser preview: ConnectomeBrain (WebGPU) + BodyWasm (MuJoCo-WASM)
// + PreviewLoop, rendered with the same VRM avatar and brain panel as the
// studio workbench, but with no server behind it.
import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { activityDisplay } from './brain-activity.js';
import { t, getLang, setLang, applyStatic } from './i18n.js';
import { $, setText, setHTML, setKey, showView } from './dom.js';
import * as evaluationView from './evaluation-view.js';
import { checkSupport, ConnectomeBrain } from './brain-webgpu.js';
import { BodyWasm } from './body-wasm.js';
import { PreviewLoop } from './preview-loop.js';
// The research view links the full report; vite emits it as a hashed asset so the
// static host needs no extra configuration for it.
import reportUrl from '../research/REPORT.md?url';

let toastTimer;
function toast(message) { $('toast').textContent = message; $('toast').classList.add('show'); clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').classList.remove('show'), 4200); }

// Deploy-time locations. The package base URL is a <meta> tag so a static host
// can point the page at the published copy without rebuilding it.
const setting = (name, fallback) => document.querySelector(`meta[name="${name}"]`)?.content || fallback;
const PACKAGE = new URL(setting('preview-package', '/artifacts/preview/'), document.baseURI);
const VRM_URL = setting('preview-vrm', '/yumi.vrm');

applyStatic();
$('languageToggle').onclick = () => {
  setLang(getLang() === 'zh' ? 'en' : 'zh');
  $('langCurrent').textContent = getLang() === 'zh' ? 'EN' : '中';
  applyStatic();
  if (bootDone) {
    setKey($('stageMeta'), 'stage.meta', {body: t('stage.bodyName.yumi'), joints: 12});
    setText($('connection'), t('preview.connected'));
  }
};
$('langCurrent').textContent = getLang() === 'zh' ? 'EN' : '中';

// The results and research views are read-only here: the build already removed the
// markup that would need a server (see viewFragments in vite.config.js).
let activeView = 'studio', packageMeta = null, evidenceRequested = false;

/** Show the package's held-out evidence — but only while it is bound to the package's
 *  checkpoint. The studio gates this the same way in /api/evaluation; a static host has
 *  no server to do it, so the check moves here and a mismatch is refused, not shown. */
function loadEvidence() {
  if (!packageMeta || evidenceRequested) return;
  evidenceRequested = true;
  fetch(new URL(packageMeta.evaluation.path, PACKAGE)).then((response) => response.json()).then((record) => {
    if (record.checkpoint_sha256 !== packageMeta.checkpoint_sha256) {
      evaluationView.setEvaluationState('failed', t('results.mismatch'));
      return;
    }
    evaluationView.renderEvaluation(record, { meta: packageMeta });
  }).catch((error) => {
    console.error(error);
    evaluationView.setEvaluationState('failed', t('results.failed'));
  });
}

function view(name) {
  activeView = name;
  showView(name);
  if (name === 'results') loadEvidence();
}
document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => view(button.dataset.view)));
// The build rewrote the shared views' API links to placeholders; point them at what
// this page actually has. The evidence file is content-addressed, so its name is only
// known once the package's meta.json is in.
$('researchReport').href = reportUrl;
setKey($('researchContext'), 'study.context.full', { body: 'Yumi' });
evaluationView.setEvaluationState('absent', t('eval.absent'));

function overlay(message) {
  const el = $('stageOverlay');
  if (!message) { el.hidden = true; return; }
  el.textContent = message; el.hidden = false;
}

/** The checkpoint records the observation interface it was trained against, and
 *  the browser body must be built from those same numbers. Comparing the pair
 *  catches a package shipped alongside the wrong body_config.json — the drift
 *  class that produced the 2026-09-16 interface mismatch. */
function physicsDrift(bodyConfig, physicsInterface) {
  for (const [key, recorded] of Object.entries(physicsInterface || {})) {
    // The body spec calls the default angles `default_angles`; the browser config
    // carries them as `home` (app/sim.py casts them to float32, hence the tolerance).
    const local = key === 'default_angles' ? 'home' : key;
    const want = [recorded].flat(), got = [bodyConfig[local]].flat();
    if (got.length !== want.length) return key;
    if (want.some((value, i) => typeof got[i] !== 'number' || Math.abs(value - got[i]) > 1e-6)) return key;
  }
  return null;
}

const boot = (async () => {
  const support = await checkSupport();
  if (!support.supported) throw new Error(t('preview.error.webgpu'));
  // Hash verification needs a secure context, so over plain HTTP the package
  // would load unchecked rather than fail.
  if (!crypto.subtle) toast(t('preview.error.insecure'));

  overlay(t('preview.loading.brain'));
  const brain = await ConnectomeBrain.load(PACKAGE, {
    support,
    progress: (stage, done, total) => {
      if (stage === 'weights') overlay(`${t('preview.loading.weights')} ${done}/${total}`);
    },
  });
  const meta = brain.meta;
  // The evidence lives in the package, so it can only be shown once the package's own
  // checkpoint is known to compare against.
  packageMeta = meta;
  if (activeView === 'results') loadEvidence();
  // Selected by its copy key: the shared markup is identical on both pages, so the
  // link has no id of its own.
  document.querySelector('[data-i18n="results.json"]').href = new URL(meta.evaluation.path, PACKAGE).href;
  const hash = meta.checkpoint_sha256 || '';
  $('neuronCount').textContent = meta.n.toLocaleString(getLang());
  $('edgeCount').textContent = `${meta.nnz.toLocaleString(getLang())} ${getLang() === 'zh' ? '突触' : 'synapses'}`;
  $('checkpointHash').textContent = `CHECKPOINT ${hash ? hash.slice(0, 8) : '—'}`;

  const config = await (await fetch(new URL('body_config.json', PACKAGE))).json();
  const drifted = physicsDrift(config, meta.physics_interface);
  if (drifted) throw new Error(`body_config.json disagrees with the checkpoint on ${drifted}`);

  overlay(t('preview.loading.body'));
  const body = await BodyWasm.load({
    glueUrl: new URL(meta.mujoco.glue.path, PACKAGE).href,
    wasmUrl: new URL(meta.mujoco.wasm.path, PACKAGE).href,
    sceneUrl: new URL(meta.scene.scene.path, PACKAGE).href,
    includeUrl: new URL(meta.scene.include.path, PACKAGE).href,
    config,
  });
  const initialQvel = meta.initial_qvel;
  body.reset(initialQvel);

  const loop = new PreviewLoop({brain, body, initialQvel});
  window.__preview = loop;
  return {brain, body, loop, config, coords: brain.buffers.coords};
})();

const scene3d = (async () => {
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(2);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  const viewport = $('viewport');
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2('#edf2f8', 0.025);
  const camera = new THREE.PerspectiveCamera(33, 1, 0.05, 200);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 1.6; controls.maxDistance = 10; controls.maxPolarAngle = Math.PI * 0.47;
  controls.target.set(0, 0.9, 0);
  camera.position.set(2.15, 1.65, 3.45);

  const hemi = new THREE.HemisphereLight('#ffffff', '#8a9bb1', 1.1); scene.add(hemi);
  const key = new THREE.DirectionalLight('#fff2e7', 1.8); key.position.set(2, 5, 3); key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024); key.shadow.camera.left = -3; key.shadow.camera.right = 3; key.shadow.camera.top = 3; key.shadow.camera.bottom = -3; key.shadow.normalBias = 0.015; key.shadow.bias = -0.0001;
  scene.add(key, key.target);
  const rim = new THREE.DirectionalLight('#dae8ff', 1.3); rim.position.set(-3, 2.5, -2); scene.add(rim);
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshStandardMaterial({color: '#dce4ef', roughness: 0.93, metalness: 0.05}));
  ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; ground.position.y = -0.015; scene.add(ground);
  const grid = new THREE.GridHelper(400, 400, '#b9c4d6', '#cdd7e4'); grid.position.y = -0.009; grid.material.transparent = true; grid.material.opacity = 0.55; scene.add(grid);

  return {renderer, scene, camera, controls, ground, key, viewport};
})();

const avatar3d = (async () => {
  const {scene} = await scene3d;
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  const bytes = await (await fetch(VRM_URL)).arrayBuffer();
  const gltf = await loader.parseAsync(bytes, new URL('.', new URL(VRM_URL, location.href)).href);
  const vrm = gltf.userData.vrm;
  const hips = vrm.humanoid.getNormalizedBoneNode('hips');
  if (!hips) throw new Error('VRM has no mapped hips bone');
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  VRMUtils.rotateVRM0(vrm);
  vrm.scene.traverse((node) => { if (node.isMesh) { node.castShadow = true; node.receiveShadow = true; node.frustumCulled = false; } });
  scene.add(vrm.scene);
  return vrm;
})();

const brain3d = (async () => {
  const {coords} = await boot;
  const brainRenderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
  brainRenderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  const host = $('brainViewport');
  host.appendChild(brainRenderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#050811');
  const camera = new THREE.PerspectiveCamera(37, 1, 0.1, 100);
  camera.position.set(0.25, 0.6, 5.3); camera.lookAt(0, 0, 0);
  const positions = [];
  for (let i = 0; i < coords.length; i += 3) positions.push(new THREE.Vector3(coords[i], -coords[i + 2], coords[i + 1]));
  const box = new THREE.Box3().setFromPoints(positions), center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const scale = 2.9 / Math.max(size.x, size.y, size.z);
  positions.forEach((p) => p.sub(center).multiplyScalar(scale));
  const geometry = new THREE.BufferGeometry().setFromPoints(positions);
  geometry.setAttribute('signal', new THREE.BufferAttribute(new Float32Array(positions.length), 1));
  const material = new THREE.ShaderMaterial({
    uniforms: {
      pixelRatio: {value: brainRenderer.getPixelRatio()},
      negativeColor: {value: new THREE.Color('#55baff')},
      positiveColor: {value: new THREE.Color('#ffb36b')},
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
        float glow = 1.0 - smoothstep(0.3, 1.0, radius);
        gl_FragColor = vec4(pointColor, pointOpacity * glow);
        #include <colorspace_fragment>
      }
    `,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const group = new THREE.Group();
  group.add(new THREE.Points(geometry, material));
  group.rotation.set(0.12, -0.12, 0);
  scene.add(group);
  return {brainRenderer, scene, camera, geometry, host};
})();

let state = null, displayPose = null;
const qX = new THREE.Quaternion(), qY = new THREE.Quaternion(), qZ = new THREE.Quaternion();
const axisX = new THREE.Vector3(1, 0, 0), axisY = new THREE.Vector3(0, 1, 0), axisZ = new THREE.Vector3(0, 0, 1);
let avatarScale = 1.16, hipsRest = 0.9, avatarAxis = 1, physicsHips = 0.97231;
const lastRoot = new THREE.Vector3(), rootTarget = new THREE.Vector3(), rootDelta = new THREE.Vector3();

function rotateBone(vrm, name, pitch = 0, roll = 0, yaw = 0) {
  const target = vrm ? vrm.humanoid.getNormalizedBoneNode(name) : null;
  if (!target) return;
  qX.setFromAxisAngle(axisX, pitch * avatarAxis); qZ.setFromAxisAngle(axisZ, roll * avatarAxis); qY.setFromAxisAngle(axisY, yaw);
  target.quaternion.copy(qX).multiply(qZ).multiply(qY);
}

function updateBody(dt, snapshot, {scene, camera, controls, ground, key}, vrm) {
  if (!snapshot?.qpos) return;
  if (!displayPose) displayPose = snapshot.qpos.slice();
  const blend = snapshot.running ? 1 - Math.exp(-25 * Math.max(dt, 0)) : 1;
  for (let i = 0; i < displayPose.length; i++) displayPose[i] += (snapshot.qpos[i] - displayPose[i]) * blend;
  const q = displayPose;
  rootTarget.set(q[1] * avatarScale, 0, q[0] * avatarScale);
  rootDelta.copy(rootTarget).sub(lastRoot);
  camera.position.add(rootDelta); controls.target.add(rootDelta); lastRoot.copy(rootTarget);
  key.position.copy(rootTarget).add(new THREE.Vector3(2, 5, 3)); key.target.position.copy(rootTarget);
  ground.position.x = rootTarget.x; ground.position.z = rootTarget.z;
  const hips = vrm ? vrm.humanoid.getNormalizedBoneNode('hips') : null;
  if (vrm && hips) {
    vrm.scene.position.copy(rootTarget);
    vrm.scene.position.y = q[2] * avatarScale - hipsRest;
    hips.quaternion.set(q[5] * avatarAxis, q[6], q[4] * avatarAxis, q[3]).normalize();
    for (let side = 0; side < 2; side++) {
      const name = side ? 'right' : 'left', i = 7 + side * 6;
      rotateBone(vrm, `${name}UpperLeg`, q[i], q[i + 1], q[i + 2]);
      rotateBone(vrm, `${name}LowerLeg`, q[i + 3]);
      rotateBone(vrm, `${name}Foot`, q[i + 4], q[i + 5]);
      rotateBone(vrm, `${name}UpperArm`, -q[i] * 0.48, side ? 1.38 : -1.38);
      rotateBone(vrm, `${name}LowerArm`, 0, 0, side ? 0.13 : -0.13);
    }
    vrm.update(dt);
    vrm.scene.updateMatrixWorld(true);
  }
}

let brainView = 'change', brainCurrent = null, brainPrevious = null, brainCache = null;
function updateBrain(snapshot) {
  if (!brainCache || !snapshot?.activity) return;
  const {geometry} = brainCache;
  if (!brainCurrent || snapshot.time !== brainCurrent.time || snapshot.episode !== brainCurrent.episode) {
    brainPrevious = brainCurrent;
    brainCurrent = snapshot;
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
  setKey($('brainReadout'), snapshot.running ? 'brain.readout.live' : 'brain.readout.paused');
}
for (const mode of ['change', 'strength']) $('brain' + mode).addEventListener('click', () => {
  brainView = mode;
  for (const option of ['change', 'strength']) $('brain' + option).setAttribute('aria-pressed', String(option === mode));
  updateBrain(state);
});

let avatar = null, bootDone = false;
boot.then(async ({loop, config}) => {
  avatar = await avatar3d;
  avatarAxis = avatar.meta.metaVersion === '0' ? -1 : 1;
  physicsHips = config.hips_height || physicsHips;
  hipsRest = avatar.humanoid.getNormalizedBoneNode('hips').position.y;
  avatarScale = hipsRest / physicsHips;
  brainCache = await brain3d;
  bootDone = true;
  $('modelLoading').hidden = true;
  setText(document.querySelector('.avatar-name'), 'Yumi');
  setText($('avatarCredit'), '原设：松酒 · 画师：7Apoi · 模型：星晨水影工作室 · 发布：墨海徽');
  setKey($('stageMeta'), 'stage.meta', {body: t('stage.bodyName.yumi'), joints: 12});
  $('connection').removeAttribute('data-i18n');
  setText($('connection'), t('preview.connected'));
  setText($('datasetChip'), 'MaleCNS v1.0');
  const stage = await scene3d;
  stage.controls.target.set(0, 0.9, 0);
  $('cameraReset').onclick = () => {
    stage.controls.target.copy(rootTarget).add(new THREE.Vector3(0, 0.9, 0));
    stage.camera.position.copy(rootTarget).add(new THREE.Vector3(2.15, 1.65, 3.45));
    stage.controls.update();
  };
  $('pushButton').onclick = () => { loop.push(); toast(t('ctrl.push.done')); };
  $('resetButton').onclick = () => {
    loop.reset();
    loop.running = true;
    displayPose = null; brainCurrent = null;
    toast(t('ctrl.reset.done'));
  };
  $('targetSpeed').oninput = (e) => { $('targetLabel').innerHTML = `${Number(e.target.value).toFixed(2)} <small>m/s</small>`; };
  $('targetSpeed').onchange = (e) => loop.setSpeed(+e.target.value);
  $('heading').oninput = (e) => { $('headingLabel').textContent = `${e.target.value}°`; };
  $('heading').onchange = (e) => loop.setYaw(+e.target.value * Math.PI / 180);
  $('lesion').onchange = (e) => {
    loop.setLesion(e.target.value !== 'intact');
    loop.reset();
    loop.running = true;
    displayPose = null; brainCurrent = null;
    toast(t(e.target.value === 'intact' ? 'lesion.intact.done' : 'lesion.other.done'));
  };

  let errorShown = false, pausedByHidden = false, playSync = null;
  const setPlaying = (playing) => {
    playSync = playing;
    setKey($('playLabel'), playing ? 'ctrl.pause' : 'ctrl.play');
    $('playButton').querySelector('.glyph').textContent = playing ? 'Ⅱ' : '▶';
  };
  // A hidden tab can lose the WebGPU device (mapAsync then rejects with
  // "external Instance reference"), so pause the sim while the page is hidden
  // and resume when it comes back — unless the user paused it themselves.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && loop.running) {
      pausedByHidden = true;
      loop.running = false;
      setPlaying(false);
    } else if (!document.hidden && pausedByHidden) {
      pausedByHidden = false;
      loop.running = true;
      setPlaying(true);
    }
  });
  $('playButton').onclick = () => {
    pausedByHidden = false;
    loop.running = !loop.running;
    setPlaying(loop.running);
  };
  loop.onState((snapshot) => {
    state = snapshot;
    if (snapshot.running !== playSync) setPlaying(snapshot.running);
    if (loop.lastError) {
      if (!errorShown) {
        errorShown = true;
        overlay(`${loop.lastError.message || loop.lastError}`);
        toast(loop.lastError.message || String(loop.lastError));
      }
    } else {
      if (errorShown) errorShown = false;
      overlay(null);
    }
    const speed = snapshot.velocity[0];
    setText($('speed'), speed.toFixed(2));
    setText($('upright'), Math.max(0, snapshot.upright * 100).toFixed(1));
    const balance = `${Math.max(0, Math.min(100, snapshot.upright * 100))}%`;
    if ($('balanceBar').style.width !== balance) $('balanceBar').style.width = balance;
    setText($('distance'), snapshot.distance.toFixed(2));
    setText($('simTime'), snapshot.time.toFixed(1));
    setText($('rtf'), `${(snapshot.real_time_factor * 100).toFixed(0)} %`);
    setText($('actualHeading'), `${(snapshot.yaw * 180 / Math.PI).toFixed(1)}°`);
    setText($('brainMs'), `${snapshot.brain_ms.toFixed(1)} ms`);
    setText($('outputRms'), snapshot.output_rms.toFixed(3));
    if (snapshot.fallen) {
      setKey($('speedState'), 'metric.upright.fallen');
    } else {
      const gap = speed - snapshot.target_speed;
      setKey($('speedState'), Math.abs(gap) <= 0.05 ? 'metric.speed.ontrack' : gap < 0 ? 'metric.speed.below' : 'metric.speed.above',
        {error: Math.abs(gap).toFixed(2)});
    }
    setKey($('balanceState'), snapshot.fallen ? (snapshot.running ? 'metric.upright.fallen' : 'metric.upright.fallenPaused') : 'metric.upright.measured');
    const slow = snapshot.running && snapshot.real_time_factor < 0.95;
    $('slowBadge').hidden = !slow;
    if (slow) setText($('slowLabel'), t('preview.slow', {rate: (1 / snapshot.real_time_factor).toFixed(1)}));
    updateBrain(snapshot);
  });
  setPlaying(true);
  await loop.start();
}).catch((error) => {
  console.error(error);
  overlay(error.message || String(error));
  toast(error.message || String(error));
});

let lastFrame = performance.now();
let stageCtx = null, brainCtx = null;
Promise.all([scene3d, brain3d]).then(([stage, brainPanel]) => { stageCtx = stage; brainCtx = brainPanel; })
  .catch((error) => { console.error(error); overlay(error.message || String(error)); });

// three.js setSize() reallocates the drawing buffer unconditionally, so it
// must only run when the panel actually changed size — not every frame.
const lastSizes = new WeakMap();
function resizeRenderer(renderer, camera, host) {
  const width = host.clientWidth, height = host.clientHeight;
  if (!width || !height) return;
  const last = lastSizes.get(renderer);
  if (last && last.width === width && last.height === height) return;
  lastSizes.set(renderer, {width, height});
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function frame() {
  requestAnimationFrame(frame);
  const now = performance.now();
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  if (!stageCtx || !brainCtx) return;
  resizeRenderer(stageCtx.renderer, stageCtx.camera, stageCtx.viewport);
  resizeRenderer(brainCtx.brainRenderer, brainCtx.camera, brainCtx.host);
  if (avatar) updateBody(dt, state, stageCtx, avatar);
  stageCtx.controls.update();
  stageCtx.renderer.render(stageCtx.scene, stageCtx.camera);
  brainCtx.brainRenderer.render(brainCtx.scene, brainCtx.camera);
}
frame();
