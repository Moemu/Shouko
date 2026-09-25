// Read back module GPU buffers and compare against fixtures.
import {ConnectomeBrain} from '/web/brain-webgpu.js';

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
  const fetchBin = async (name) => new Uint8Array(await (await fetch(`/bench_webgpu/fixtures/${name}`)).arrayBuffer());
  const [refR, valuesD, preD, ptrD, biasD] =
    await Promise.all(['reference.json', 'values.f32.bin', 'pre.i32.bin', 'ptr.i32.bin', 'bias.f32.bin'].map(fetchBin));
  const ref = JSON.parse(new TextDecoder().decode(refR));
  const meta = {
    n: ref.n, nnz: ref.nnz, neural_steps: ref.neural_steps,
    observation_size: ref.observation_size, action_size: ref.action_size,
    inputs_count: ref.inputs_count, outputs_count: ref.outputs_count, sample_count: 0,
  };
  const buffers = {
    values: new Float32Array(valuesD.buffer), pre: new Int32Array(preD.buffer),
    ptr: new Int32Array(ptrD.buffer), bias: new Float32Array(biasD.buffer),
    inputs: new Int32Array((await fetchBin('inputs.i32.bin')).buffer),
    outputs: new Int32Array((await fetchBin('outputs.i32.bin')).buffer),
    encoder: new Float32Array((await fetchBin('encoder.f32.bin')).buffer),
    readout: new Float32Array((await fetchBin('readout.f32.bin')).buffer),
    readout_bias: new Float32Array((await fetchBin('readout_bias.f32.bin')).buffer),
    norm_w: new Float32Array((await fetchBin('norm_w.f32.bin')).buffer),
    norm_b: new Float32Array((await fetchBin('norm_b.f32.bin')).buffer),
    obs_mean: new Float32Array((await fetchBin('obs_mean.f32.bin')).buffer),
    obs_std: new Float32Array((await fetchBin('obs_std.f32.bin')).buffer),
  };
  const brain = await ConnectomeBrain.fromBuffers(meta, buffers);
  window.__brain = brain;
  const d = brain.device;

  const readBack = async (gpuBuf, bytes) => {
    const st = d.createBuffer({size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    const enc = d.createCommandEncoder();
    enc.copyBufferToBuffer(gpuBuf, 0, st, 0, bytes);
    d.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const a = st.getMappedRange().slice(0);
    st.unmap(); st.destroy();
    return a;
  };

  const check = async (key, gpuBuf, fixtureName, Type) => {
    const host = buffers[key];
    const raw = await readBack(gpuBuf, host.byteLength);
    const gpu = new Type(raw);
    const [diff, idx] = maxAbs(gpu, host);
    log(`${key}: host-vs-gpu maxAbsDiff=${diff.toExponential(3)} @${idx}/${host.length} ` +
        `(host[0]=${host[0]} gpu[0]=${gpu[0]})`);
    if (fixtureName) {
      const fx = new Type((await fetchBin(fixtureName)).buffer);
      const [hDiff] = maxAbs(host, fx);
      log(`${key}: host-vs-fixture maxAbsDiff=${hDiff.toExponential(3)}`);
    }
  };

  await check('bias', brain.bBias, 'bias.f32.bin', Float32Array);
  await check('ptr', brain.bPtr, 'ptr.i32.bin', Int32Array);
  const preHost = buffers.pre;
  const preRaw = await readBack(brain.bPre, preHost.byteLength);
  const preGpu = new Int32Array(preRaw);
  let mism = -1, cnt = 0;
  for (let i = 0; i < preHost.length; i++) { if (preGpu[i] !== preHost[i]) { if (mism < 0) mism = i; cnt++; } }
  log(`pre: host-vs-gpu mismatches=${cnt} first@${mism}` +
      (mism >= 0 ? ` (host=${preHost[mism]} gpu=${preGpu[mism]})` : ''));
  const valHost = buffers.values;
  const valRaw = await readBack(brain.bValues, valHost.byteLength);
  const valGpu = new Float32Array(valRaw);
  const [vDiff, vIdx] = maxAbs(valGpu, valHost);
  log(`values: host-vs-gpu maxAbsDiff=${vDiff.toExponential(3)} @${vIdx}/${valHost.length} ` +
      `(host[0]=${valHost[0]} gpu[0]=${valGpu[0]})`);
  log('DIAG DONE');
  window.__diagDone = true;
} catch (e) {
  log('ERROR: ' + (e && (e.stack || e.message || e)));
  window.__diagDone = true;
}
