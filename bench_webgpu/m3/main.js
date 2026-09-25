// M3 closed-loop parity: ConnectomeBrain (WebGPU) + BodyWasm (MuJoCo-WASM)
// vs the Python CPU reference (parity_closed_loop.py), seed 4242, 500 steps.
import {checkSupport, ConnectomeBrain} from '/web/brain-webgpu.js?v=2';
import {BodyWasm} from '/web/body-wasm.js';

const out = document.getElementById('results');
const lines = [];
const log = (msg) => { lines.push(String(msg)); out.textContent = lines.join('\n'); };

const maxAbs = (a, b) => {
  let m = 0, idx = -1;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) { m = d; idx = i; }
  }
  return [m, idx];
};

try {
  const [configR, parityR] = await Promise.all([
    fetch('./body_config.json').then((r) => r.json()),
    fetch('./parity_closed_loop.json').then((r) => r.json())]);
  const config = configR;
  const parity = parityR;
  log(`config: obs=${config.observation_size} decimation=${config.control_decimation} dt=${config.simulation_dt} gait=${config.gait_period_s}s`);
  log(`reference: ${parity.steps} steps, speed=${parity.speed}, initial_qvel=${JSON.stringify(parity.initial_qvel)}, final_height=${parity.final_height.toFixed(4)}, distance=${parity.distance.toFixed(3)}`);

  const support = await checkSupport();
  if (!support.supported) throw new Error(support.reason);

  const fetchBin = async (name) => new Uint8Array(await (await fetch(`/bench_webgpu/fixtures/${name}`)).arrayBuffer());
  const [refR, valuesD, preD, ptrD, inputsD, biasD, encD, roD, roBD, nwD, nbD, omD, osD, outD] =
    await Promise.all(['reference.json', 'values.f32.bin', 'pre.i32.bin', 'ptr.i32.bin',
      'inputs.i32.bin', 'bias.f32.bin', 'encoder.f32.bin', 'readout.f32.bin', 'readout_bias.f32.bin',
      'norm_w.f32.bin', 'norm_b.f32.bin', 'obs_mean.f32.bin', 'obs_std.f32.bin', 'outputs.i32.bin']
      .map(fetchBin));
  const ref = JSON.parse(new TextDecoder().decode(refR));

  const sampleCount = 2048;
  const sample = new Int32Array(sampleCount);
  for (let i = 0; i < sampleCount; i++) sample[i] = Math.floor(i * (ref.n - 1) / (sampleCount - 1));
  const meta = {
    n: ref.n, nnz: ref.nnz, neural_steps: ref.neural_steps,
    observation_size: ref.observation_size, action_size: ref.action_size,
    inputs_count: ref.inputs_count, outputs_count: ref.outputs_count,
    sample_count: sampleCount,
  };
  const buffers = {
    values: new Float32Array(valuesD.buffer), pre: new Int32Array(preD.buffer),
    ptr: new Int32Array(ptrD.buffer), inputs: new Int32Array(inputsD.buffer),
    outputs: new Int32Array(outD.buffer), bias: new Float32Array(biasD.buffer),
    encoder: new Float32Array(encD.buffer), readout: new Float32Array(roD.buffer),
    readout_bias: new Float32Array(roBD.buffer), norm_w: new Float32Array(nwD.buffer),
    norm_b: new Float32Array(nbD.buffer), obs_mean: new Float32Array(omD.buffer),
    obs_std: new Float32Array(osD.buffer), sample,
  };
  const brain = await ConnectomeBrain.fromBuffers(meta, buffers, support);
  window.__brain = brain;
  log(`brain ready; gpu errors: ${brain.errors.length}`);

  const tLoad = performance.now();
  const body = await BodyWasm.load({
    // No wasmUrl: this harness loads the glue's own sibling copy, as it did when
    // the parity numbers in the report were measured.
    glueUrl: '/bench_mujoco/public/mujoco.js',
    sceneUrl: '/bench_mujoco/public/scene.xml',
    includeUrl: '/bench_mujoco/public/yumi.xml',
    config,
  });
  body.reset(parity.initial_qvel);
  log(`body ready in ${(performance.now() - tLoad).toFixed(0)}ms; nq=${body.model.nq} nv=${body.model.nv} nu=${body.model.nu}`);
  window.__body = body;

  const SPEED = parity.speed, TARGET_YAW = parity.target_yaw;
  const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  let worstAction = 0, worstQpos = 0, worstK = -1;
  const actionWorstByK = [];
  const qposDiffs = [];
  const inferTimes = [], stepTimes = [];
  const tAll = performance.now();

  for (let k = 0; k < parity.steps; k++) {
    const [, , , yaw] = body.observation();
    const error = Math.atan2(Math.sin(TARGET_YAW - yaw), Math.cos(TARGET_YAW - yaw));
    const command = [SPEED, 0, clip(error * 1.4, -0.2, 0.2)];
    const obs = body.motorObservation(command);

    let t = performance.now();
    const r = await brain.infer(obs);
    inferTimes.push(performance.now() - t);

    t = performance.now();
    const state = body.stepJoints(r.action);
    stepTimes.push(performance.now() - t);

    const [aDiff] = maxAbs(r.action, parity.records[k].action);
    const qpos = Array.from(body.data.qpos);
    const [qDiff] = maxAbs(qpos, parity.records[k].qpos);
    if (aDiff > worstAction) worstAction = aDiff;
    if (qDiff > worstQpos) { worstQpos = qDiff; worstK = k; }
    qposDiffs.push(qDiff);
    if (!isFinite(aDiff) || !isFinite(qDiff)) { log(`non-finite diff at k=${k}`); break; }
    if (k % 100 === 0 || k === parity.steps - 1) {
      log(`k=${k} actionDiff=${aDiff.toExponential(2)} qposDiff=${qDiff.toExponential(2)} height=${state.height.toFixed(4)} fallen=${state.fallen}`);
    }
  }
  const firstExceed = qposDiffs.findIndex((d) => d > 1e-3);
  log(`first qposDiff > 1e-3 at k=${firstExceed}`);
  log(`diffs k=1..24: ${qposDiffs.slice(1, 25).map((d) => d.toExponential(2)).join(' ')}`);
  const totalMs = performance.now() - tAll;

  const stat = (arr) => {
    const s = [...arr].sort((a, b) => a - b);
    const mean = arr.reduce((a, b) => a + b) / arr.length;
    return `mean=${mean.toFixed(2)}ms p95=${s[Math.floor(arr.length * 0.95)].toFixed(2)}ms max=${s[arr.length - 1].toFixed(2)}ms`;
  };
  log(`wall: ${totalMs.toFixed(0)}ms for ${parity.steps} steps (${(totalMs / parity.steps).toFixed(1)}ms/step; real-time budget 20ms)`);
  log(`infer: ${stat(inferTimes)}`);
  log(`physics: ${stat(stepTimes)}`);
  log(`worst action diff: ${worstAction.toExponential(3)}`);
  log(`worst qpos diff: ${worstQpos.toExponential(3)} at k=${worstK}`);
  const passed = worstAction < 1e-3 && worstQpos < 1e-3;
  log(`RESULT: ${passed ? 'PASS' : 'FAIL'} (tolerance 1e-3)`);
  log('DONE');
  window.__m3done = true;
} catch (e) {
  log('ERROR: ' + (e && (e.stack || e.message || e)));
  window.__m3done = true;
}
