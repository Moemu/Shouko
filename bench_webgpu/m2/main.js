// M2 validation: exercise the productized web/brain-webgpu.js module against
// the same fixtures used by the bench_webgpu prototype (export_reference.py).
import {checkSupport, ConnectomeBrain} from '/web/brain-webgpu.js?v=2';

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
  const support = await checkSupport();
  log(`support: ${JSON.stringify(support)}`);
  if (!support.supported) throw new Error(support.reason);

  const fetchBin = async (name) => new Uint8Array(await (await fetch(`/bench_webgpu/fixtures/${name}`)).arrayBuffer());
  const [refR, valuesD, preD, ptrD, inputsD, biasD, encD, roD, roBD, nwD, nbD, omD, osD, outD] =
    await Promise.all(['reference.json', 'values.f32.bin', 'pre.i32.bin', 'ptr.i32.bin',
      'inputs.i32.bin', 'bias.f32.bin', 'encoder.f32.bin', 'readout.f32.bin', 'readout_bias.f32.bin',
      'norm_w.f32.bin', 'norm_b.f32.bin', 'obs_mean.f32.bin', 'obs_std.f32.bin', 'outputs.i32.bin']
      .map(fetchBin));
  const ref = JSON.parse(new TextDecoder().decode(refR));

  // 2048 evenly spaced sample indices, mirroring studio_server's display sampling.
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

  const t0 = performance.now();
  const brain = await ConnectomeBrain.fromBuffers(meta, buffers, support);
  window.__brain = brain;
  log(`brain ready in ${(performance.now() - t0).toFixed(0)}ms; gpu errors: ${brain.errors.length}`);

  for (const tag of ['zero', 'rand']) {
    const fx = ref.fixtures[tag];
    const r = await brain.infer(new Float32Array(fx.obs));
    const [oaDiff] = maxAbs(r.outputActivity, fx.output_activity);
    const [aDiff, aIdx] = maxAbs(r.action, fx.action);
    log(`[${tag}] output_activity maxAbsDiff=${oaDiff.toExponential(3)}; ` +
        `action maxAbsDiff=${aDiff.toExponential(3)} @${aIdx} ` +
        `(webgpu=${r.action[aIdx].toExponential(3)} torch=${fx.action[aIdx].toExponential(3)}); ` +
        `outputRms=${r.outputRms.toFixed(4)} gatherMs=${r.gatherMs.toFixed(2)}; ` +
        `sampleActivity len=${r.sampleActivity.length}`);
  }

  const fx = ref.fixtures.rand;
  const signal = new Float32Array(fx.obs);
  for (let i = 0; i < 10; i++) { await brain.infer(signal); }
  const times = [];
  for (let i = 0; i < 100; i++) {
    const t = performance.now();
    await brain.infer(signal);
    times.push(performance.now() - t);
  }
  times.sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b) / times.length;
  const p = (q) => times[Math.floor(q * times.length)].toFixed(2);
  log(`infer (4 steps + gather + readback): mean=${mean.toFixed(2)}ms p50=${p(0.5)} ` +
      `p95=${p(0.95)} p99=${p(0.99)} max=${times[times.length - 1].toFixed(2)} ` +
      `(prototype bench was mean 7.54 / p99 14.7 without gather)`);

  const lesionR = await brain.infer(signal, {lesion: true});
  log(`[lesion] action[0]=${lesionR.action[0].toExponential(3)} (intact ${fx.action[0].toExponential(3)}); ` +
      `outputRms=${lesionR.outputRms.toFixed(4)}`);

  // diagnostics: signal check + manual 4-step replay with full readback
  const sbD = await fetchBin('sbias.f32.bin');
  const sb = new Float32Array(sbD.buffer);
  let [sDiff] = maxAbs(brain._signal, sb);
  log(`[diag] _signal vs sbias fixture: maxAbsDiff=${sDiff.toExponential(3)}`);
  const as4 = new Float32Array((await fetchBin('act_step4.f32.bin')).buffer);
  const d = brain.device;
  const fullStaging = d.createBuffer({size: ref.n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
  const replay = async () => {
    d.queue.writeBuffer(brain.bU2, 0, new Uint32Array([ref.n, 0, 0, 0]));
    d.queue.writeBuffer(brain.bS, 0, brain._signal);
    d.queue.writeBuffer(brain.bActA, 0, brain.zero);
    const enc2 = d.createCommandEncoder();
    const pass2 = enc2.beginComputePass();
    let cur = 0;
    for (let step = 0; step < 4; step++) {
      pass2.setPipeline(brain.spPipeline);
      pass2.setBindGroup(0, brain.BG[cur][0]);
      pass2.dispatchWorkgroups(65535);
      pass2.setPipeline(brain.upPipeline);
      pass2.setBindGroup(0, brain.BG[cur][1]);
      pass2.dispatchWorkgroups(Math.ceil(ref.n / 64));
      cur = 1 - cur;
    }
    pass2.end();
    enc2.copyBufferToBuffer(brain.bActA, 0, fullStaging, 0, ref.n * 4);
    d.queue.submit([enc2.finish()]);
    await fullStaging.mapAsync(GPUMapMode.READ);
    const act = new Float32Array(fullStaging.getMappedRange().slice(0));
    fullStaging.unmap();
    return act;
  };
  const act = await replay();
  const [a4Diff, a4Idx] = maxAbs(act, as4);
  log(`[diag] manual replay vs act_step4: maxAbsDiff=${a4Diff.toExponential(3)} @${a4Idx} ` +
      `(webgpu=${act[a4Idx].toExponential(3)} torch=${as4[a4Idx].toExponential(3)})`);
  fullStaging.destroy();

  log('DONE');
  window.__m2done = true;
} catch (e) {
  log('ERROR: ' + (e && (e.stack || e.message || e)));
  window.__m2done = true;
}
