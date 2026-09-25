// Scratch benchmark: CSR SpMV + 4-step connectome recurrence in WebGPU,
// validated against fixtures exported from app.full_brain (export_reference.py).
const WG = 64;
const STRIDE = 65535;

const SHADER = /* wgsl */ `
struct U { n: u32, pad0: u32, pad1: u32, pad2: u32, };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read> pre: array<i32>;
@group(0) @binding(3) var<storage, read> ptr: array<i32>;
@group(0) @binding(4) var<storage, read> act_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> sp: array<f32>;
@group(0) @binding(6) var<uniform> u2: U;
@group(0) @binding(7) var<storage, read> sp2: array<f32>;
@group(0) @binding(8) var<storage, read> s_bias: array<f32>;
@group(0) @binding(9) var<storage, read> act_in2: array<f32>;
@group(0) @binding(10) var<storage, read_write> act_out: array<f32>;

var<workgroup> red: array<f32, ${WG}>;

@compute @workgroup_size(${WG})
fn spmv(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  for (var row = wg.x; row < u.n; row += ${STRIDE}u) {
    let start = u32(ptr[row]);
    let end = u32(ptr[row + 1u]);
    var acc = 0.0;
    var k = start + lid.x;
    loop {
      if (k >= end) { break; }
      acc = acc + values[k] * act_in[pre[k]];
      k = k + ${WG}u;
    }
    red[lid.x] = acc;
    workgroupBarrier();
    var s = ${WG}u;
    while (s > 1u) {
      s = s >> 1u;
      if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
      workgroupBarrier();
    }
    if (lid.x == 0u) { sp[row] = red[0]; }
    workgroupBarrier();
  }
}

@compute @workgroup_size(${WG})
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < u2.n) {
    act_out[i] = 0.2 * act_in2[i] + 0.8 * tanh(3.0 * sp2[i] + s_bias[i]);
  }
}
`;

const out = document.getElementById('results');
const lines = [];
function log(msg) {
  lines.push(msg);
  out.textContent = lines.join('\n');
}

function f32(view, name, expected) {
  const buf = view.getBuffer(new Float32Array(expected), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  return buf;
}

async function main() {
  if (!navigator.gpu) { log('FAIL: no navigator.gpu'); return; }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) { log('FAIL: no adapter'); return; }
  const info = adapter.info || {};
  log(`adapter: vendor=${info.vendor} arch=${info.architecture} desc=${info.description}`);
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(16, adapter.limits.maxStorageBuffersPerShaderStage) },
  });
  device.addEventListener('uncapturederror', (e) => log('GPU ERROR: ' + e.error.message));
  log(`maxStorageBufferBindingSize=${device.limits.maxStorageBufferBindingSize / 2**20}MB`);

  const fetchBin = async (name) => new Uint8Array(await (await fetch(`./fixtures/${name}`)).arrayBuffer());

  const [refR, valuesD, preD, ptrD, inputsD, biasD, encD, roD, roBD, nwD, nbD, omD, osD, outD] =
    await Promise.all(['reference.json', 'values.f32.bin', 'pre.i32.bin', 'ptr.i32.bin',
      'inputs.i32.bin', 'bias.f32.bin', 'encoder.f32.bin', 'readout.f32.bin', 'readout_bias.f32.bin',
      'norm_w.f32.bin', 'norm_b.f32.bin', 'obs_mean.f32.bin', 'obs_std.f32.bin', 'outputs.i32.bin']
      .map(fetchBin));
  const ref = JSON.parse(new TextDecoder().decode(refR));
  log(`n=${ref.n} nnz=${ref.nnz} obs=${ref.observation_size} inputs=${ref.inputs_count} outputs=${ref.outputs_count}`);

  const mkBuf = (data, usage) => {
    const b = device.createBuffer({ size: data.byteLength, usage });
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const bValues = mkBuf(valuesD, STORAGE);
  const bPre = mkBuf(preD, STORAGE);
  const bPtr = mkBuf(ptrD, STORAGE);
  const bBias = mkBuf(biasD, STORAGE);
  const n = ref.n;

  const bSp = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const bActA = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const bActB = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const bS = device.createBuffer({ size: n * 4, usage: STORAGE });

  const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uniform, 0, new Uint32Array([n, 0, 0, 0]));

  const module = device.createShaderModule({ code: SHADER });
  const spLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const upLayout = device.createBindGroupLayout({ entries: [
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
  ]});
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [spLayout] }),
    compute: { module, entryPoint: 'spmv' } });
  const upPipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [upLayout] }),
    compute: { module, entryPoint: 'update' } });

  const mkBG = (actIn, actOut) => [device.createBindGroup({ layout: spLayout, entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: bValues } },
      { binding: 2, resource: { buffer: bPre } },
      { binding: 3, resource: { buffer: bPtr } },
      { binding: 4, resource: { buffer: actIn } },
      { binding: 5, resource: { buffer: bSp } },
    ]}), device.createBindGroup({ layout: upLayout, entries: [
      { binding: 6, resource: { buffer: uniform } },
      { binding: 7, resource: { buffer: bSp } },
      { binding: 8, resource: { buffer: bS } },
      { binding: 9, resource: { buffer: actIn } },
      { binding: 10, resource: { buffer: actOut } },
    ]})];
  const BG = [mkBG(bActA, bActB), mkBG(bActB, bActA)];

  const zero = new Float32Array(n);
  device.queue.writeBuffer(bActA, 0, zero);

  const readBack = async (buf) => {
    const staging = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, n * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    return data;
  };

  const inference = (signalData) => {
    device.queue.writeBuffer(bS, 0, signalData);
    device.queue.writeBuffer(bActA, 0, zero);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    let cur = 0;
    for (let step = 0; step < ref.neural_steps; step++) {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, BG[cur][0]);
      pass.dispatchWorkgroups(STRIDE);
      pass.setPipeline(upPipeline);
      pass.setBindGroup(0, BG[cur][1]);
      pass.dispatchWorkgroups(Math.ceil(n / WG));
      cur = 1 - cur;
    }
    pass.end();
    device.queue.submit([enc.finish()]);
    return cur === 0 ? bActA : bActB;
  };

  // JS-side signal: s = bias; s[inputs[k]] = encoder_out[k]
  const encoderOut = (obs) => {
    const om = new Float32Array(omD.buffer), os = new Float32Array(osD.buffer);
    const W = new Float32Array(encD.buffer);
    const x = new Float32Array(ref.observation_size);
    for (let i = 0; i < ref.observation_size; i++) {
      x[i] = Math.min(10, Math.max(-10, (obs[i] - om[i]) / os[i]));
    }
    const sig = new Float32Array(ref.inputs_count);
    for (let j = 0; j < ref.inputs_count; j++) {
      let acc = 0, off = j * ref.observation_size;
      for (let i = 0; i < ref.observation_size; i++) acc += W[off + i] * x[i];
      sig[j] = acc;
    }
    return sig;
  };
  const makeSignal = (obs) => {
    const s = new Float32Array(biasD.buffer.slice(0));
    const sig = encoderOut(obs);
    const inputs = new Int32Array(inputsD.buffer);
    for (let k = 0; k < ref.inputs_count; k++) s[inputs[k]] += sig[k];
    return s;
  };

  const maxAbs = (a, b) => {
    let m = 0, idx = -1;
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      const d = Math.abs(a[i] - b[i]);
      if (d > m) { m = d; idx = i; }
    }
    return [m, idx];
  };
  const mismatchList = (a, b, tol, limit) => {
    const out = [];
    for (let i = 0; i < Math.min(a.length, b.length) && out.length < limit; i++) {
      if (Math.abs(a[i] - b[i]) > tol) out.push(i);
    }
    return out;
  };

  // ---- validation ----
  const [sbD, as1, as2, as3, as4] = await Promise.all(['sbias.f32.bin',
    'act_step1.f32.bin', 'act_step2.f32.bin', 'act_step3.f32.bin', 'act_step4.f32.bin'].map(fetchBin));
  const fx = ref.fixtures.rand;
  const sigJS = makeSignal(fx.obs);
  const [sDiff] = maxAbs(sigJS, new Float32Array(sbD.buffer));
  log(`[rand] s_bias maxAbsDiff=${sDiff.toExponential(3)}`);

  const oneStep = (cur) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, BG[cur][0]); pass.dispatchWorkgroups(STRIDE);
    pass.setPipeline(upPipeline); pass.setBindGroup(0, BG[cur][1]); pass.dispatchWorkgroups(Math.ceil(n / WG));
    pass.end();
    device.queue.submit([enc.finish()]);
  };
  device.queue.writeBuffer(bS, 0, sigJS);
  device.queue.writeBuffer(bActA, 0, zero);
  const stepRefs = [new Float32Array(as1.buffer), new Float32Array(as2.buffer),
    new Float32Array(as3.buffer), new Float32Array(as4.buffer)];
  for (let step = 0; step < ref.neural_steps; step++) {
    oneStep(step % 2);
    const act = await readBack(step % 2 === 0 ? bActB : bActA);
    const [d, idx] = maxAbs(act, stepRefs[step]);
    const mis = mismatchList(act, stepRefs[step], 1e-3, 24);
    log(`[rand] after step ${step + 1}: maxAbsDiff=${d.toExponential(3)} @${idx} ` +
        `(webgpu=${act[idx].toExponential(3)} torch=${stepRefs[step][idx].toExponential(3)}) ` +
        `firstMis=${JSON.stringify(mis)}`);
  }

  for (const tag of ['zero', 'rand']) {
    const fx2 = ref.fixtures[tag];
    const finalBuf = inference(makeSignal(fx2.obs));
    const act = await readBack(finalBuf);
    const outputs = new Int32Array(outD.buffer);
    const outAct = new Float32Array(ref.outputs_count);
    for (let k = 0; k < ref.outputs_count; k++) outAct[k] = act[outputs[k]];
    const [oaDiff, oaIdx] = maxAbs(outAct, fx.output_activity);
    // normalizer + readout in JS
    const nw = new Float32Array(nwD.buffer), nb = new Float32Array(nbD.buffer);
    let mean = 0; for (const v of outAct) mean += v; mean /= outAct.length;
    let varr = 0; for (const v of outAct) varr += (v - mean) ** 2;
    const std = Math.sqrt(varr / outAct.length + 1e-5);
    const motor = outAct.map((v, k) => (v - mean) / std * nw[k] + nb[k]);
    const rw = new Float32Array(roD.buffer), rb = new Float32Array(roBD.buffer);
    const action = new Float32Array(ref.action_size);
    for (let a = 0; a < ref.action_size; a++) {
      let acc = rb[a], off = a * ref.outputs_count;
      for (let k = 0; k < ref.outputs_count; k++) acc += rw[off + k] * motor[k];
      action[a] = acc;
    }
    const [aDiff, aIdx] = maxAbs(action, fx2.action);
    log(`[${tag}] output_activity maxAbsDiff=${oaDiff.toExponential(3)} @${oaIdx}; ` +
        `action maxAbsDiff=${aDiff.toExponential(3)} @${aIdx} (ref action[0]=${fx2.action[0].toExponential(3)})`);
  }

  // ---- timing ----
  const signal = makeSignal(fx.obs);
  const ITERS = 100;
  for (let i = 0; i < 10; i++) {
    inference(signal);
    await device.queue.onSubmittedWorkDone();
  }
  const times = [];
  for (let i = 0; i < ITERS; i++) {
    const t0 = performance.now();
    inference(signal);
    await device.queue.onSubmittedWorkDone();
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  const mean = times.reduce((a, b) => a + b) / ITERS;
  const p = (q) => times[Math.floor(q * ITERS)].toFixed(2);
  log(`inference(4 steps): mean=${mean.toFixed(2)}ms p50=${p(0.5)} p95=${p(0.95)} ` +
      `p99=${p(0.99)} min=${times[0].toFixed(2)} max=${times[ITERS - 1].toFixed(2)} ` +
      `iters=${ITERS} (1x realtime needs <20ms per control step)`);
  log('DONE');
  window.__benchDone = true;
}

main().catch((e) => { log('ERROR: ' + (e && e.message || e)); window.__benchDone = true; });
