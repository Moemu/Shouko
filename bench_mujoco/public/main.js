import loadMujoco from './mujoco.js';

const log = document.getElementById('log');
const lines = [];
function say(msg) {
  lines.push(String(msg));
  log.textContent = lines.join('\n');
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

try {
  const t0 = performance.now();
  const mujoco = await loadMujoco();
  say(`module loaded in ${(performance.now() - t0).toFixed(0)}ms; FS present: ${typeof mujoco.FS}`);

  const sceneXml = await (await fetch('./scene.xml')).text();
  const yumiXml = await (await fetch('./yumi.xml')).text();
  const tLoad = performance.now();
  let model;
  if (mujoco.FS && mujoco.FS.writeFile) {
    mujoco.FS.writeFile('/scene.xml', sceneXml);
    mujoco.FS.writeFile('/yumi.xml', yumiXml);
    model = mujoco.MjModel.from_xml_path('/scene.xml');
    say('loaded via FS + from_xml_path');
  } else {
    const inlined = sceneXml.replace('<include file="yumi.xml"/>', yumiXml.replace(/<\?xml[^>]*\?>/, ''));
    model = mujoco.MjModel.from_xml_string(inlined);
    say('loaded via inline include + from_xml_string');
  }
  say(`model built in ${(performance.now() - tLoad).toFixed(0)}ms: nq=${model.nq} nv=${model.nv} nu=${model.nu}`);

  const data = new mujoco.MjData(model);

  const ref = await (await fetch('./parity_python.json')).json();
  const cfg = ref.config;
  say(`python model: nq=${ref.model.nq} nv=${ref.model.nv} nu=${ref.model.nu} timestep=${ref.model.opt_timestep}`);

  model.opt.timestep = cfg.timestep;
  if (model.opt.iterations !== undefined) {
    model.opt.iterations = ref.model.opt_iterations;
    model.opt.ls_iterations = ref.model.opt_ls_iterations;
    model.opt.tolerance = ref.model.opt_tolerance;
  }
  say(`opt: timestep=${model.opt.timestep} iterations=${model.opt.iterations} ls=${model.opt.ls_iterations} tol=${model.opt.tolerance} integrator=${model.opt.integrator}`);

  mujoco.mj_resetData(model, data);
  for (let j = 0; j < 12; j++) data.qpos[7 + j] = cfg.home[j];
  data.qpos[2] = cfg.initial_height;
  data.qvel[0] = 0.01;
  data.qvel[1] = -0.02;
  mujoco.mj_forward(model, data);

  const home = cfg.home, kp = cfg.kps, kd = cfg.kds, scale = cfg.action_scale;
  const target = new Float64Array(12);
  const torque = new Float64Array(12);
  const records = [];
  let simMs = 0;

  const tStart = performance.now();
  for (let k = 0; k < ref.steps; k++) {
    for (let j = 0; j < 12; j++) target[j] = home[j] + 0.5 * Math.sin(0.2 * k + j) * scale;
    const tStep = performance.now();
    for (let d = 0; d < cfg.decimation; d++) {
      for (let j = 0; j < 12; j++) {
        torque[j] = kp[j] * (target[j] - data.qpos[7 + j]) - kd[j] * data.qvel[6 + j];
        data.ctrl[j] = torque[j];
      }
      mujoco.mj_step(model, data);
    }
    simMs += performance.now() - tStep;
    if (k % ref.sample_every === 0 || k === ref.steps - 1) {
      records.push({ k, qpos: Array.from(data.qpos) });
    }
  }
  const totalMs = performance.now() - tStart;

  let worst = 0, worstK = -1;
  const perRecord = [];
  for (let i = 0; i < ref.records.length; i++) {
    const d = maxAbsDiff(records[i].qpos, ref.records[i].qpos);
    perRecord.push([records[i].k, Number(d.toExponential(2))]);
    if (d > worst) { worst = d; worstK = records[i].k; }
  }
  say(`control steps: ${ref.steps} in ${totalMs.toFixed(0)}ms wall (physics ${simMs.toFixed(0)}ms, ${(simMs / ref.steps).toFixed(3)}ms/control-step, ${(simMs / (ref.steps * cfg.decimation)).toFixed(3)}ms/mj_step)`);
  say(`max |qpos diff| vs python: ${worst.toExponential(3)} at k=${worstK}`);
  say('per-record (k, maxdiff): ' + JSON.stringify(perRecord));

  window.__result = { ok: true, worst, worstK, totalMs, simMs };
} catch (err) {
  say('ERROR: ' + (err && (err.stack || err.message || err)));
  window.__result = { ok: false, error: String(err) };
}
