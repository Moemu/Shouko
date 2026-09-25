// Browser-side connectome inference (WebGPU). Port of app/full_brain.py
// ConnectomePolicy.forward: 4 rate-update steps of CSR sparse matmul + tanh,
// then LayerNorm over output neurons and a linear readout to joint actions.
//
// Weights come from the preview package produced by app/export_preview_weights.py:
//   <baseUrl>/meta.json  + binary files listed in meta.files
// meta.json fields: n, nnz, neural_steps, observation_size, action_size,
//   inputs_count, outputs_count, graph_sha256, checkpoint_sha256,
//   physics_interface, files: { key: { path, sha256, kind } }
// file kinds: values/pre/ptr/inputs/outputs/bias/encoder/readout/
//   readout_bias/norm_w/norm_b/obs_mean/obs_std, optional sample (i32 indices).

const WG = 64;
const STRIDE = 65535;

const SHADER = /* wgsl */ `
struct U { n: u32, lesion: u32, pad0: u32, pad1: u32, };
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

@group(0) @binding(11) var<uniform> gu: U;
@group(0) @binding(12) var<storage, read> gidx: array<u32>;
@group(0) @binding(13) var<storage, read> gact: array<f32>;
@group(0) @binding(14) var<storage, read_write> gout: array<f32>;

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
    let drive = select(3.0 * sp2[i], 0.0, u2.lesion == 1u);
    act_out[i] = 0.2 * act_in2[i] + 0.8 * tanh(drive + s_bias[i]);
  }
}

@compute @workgroup_size(${WG})
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < gu.n) { gout[i] = gact[gidx[i]]; }
}
`;

// Exported so the package check (web/preview-package.test.js) validates a
// generated package against this same contract instead of a second copy of it.
export const FILE_KINDS = {
  values: {type: Float32Array, size: (m) => m.nnz},
  pre: {type: Int32Array, size: (m) => m.nnz},
  ptr: {type: Int32Array, size: (m) => m.n + 1},
  inputs: {type: Int32Array, size: (m) => m.inputs_count},
  outputs: {type: Int32Array, size: (m) => m.outputs_count},
  bias: {type: Float32Array, size: (m) => m.n},
  encoder: {type: Float32Array, size: (m) => m.inputs_count * m.observation_size},
  readout: {type: Float32Array, size: (m) => m.action_size * m.outputs_count},
  readout_bias: {type: Float32Array, size: (m) => m.action_size},
  norm_w: {type: Float32Array, size: (m) => m.outputs_count},
  norm_b: {type: Float32Array, size: (m) => m.outputs_count},
  obs_mean: {type: Float32Array, size: (m) => m.observation_size},
  obs_std: {type: Float32Array, size: (m) => m.observation_size},
  sample: {type: Int32Array, size: (m) => m.sample_count, optional: true},
  coords: {type: Float32Array, size: (m) => m.sample_count * 3, optional: true},
};

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** Run `worker` over `items` with at most `limit` in flight.
 *
 * The package is ~200MB in 15 files. Fetching them one at a time costs a full
 * round-trip each — at the 100-200ms RTT this page targets that is minutes of
 * idle wire — while unbounded fan-out would compete with the page's own
 * requests and hold every large buffer in memory at once. */
async function mapConcurrent(items, limit, worker) {
  const results = new Array(items.length);
  let next = 0;
  const run = async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await worker(items[index], index);
    }
  };
  await Promise.all(Array.from({length: Math.min(limit, items.length)}, run));
  return results;
}

export async function checkSupport() {
  if (!navigator.gpu) return {supported: false, reason: 'webgpu_unavailable'};
  // On Optimus laptops the default adapter can be the integrated GPU, where the
  // full-connectome SpMV is several times slower; always ask for the discrete GPU.
  const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) return {supported: false, reason: 'no_adapter'};
  if (adapter.limits.maxStorageBuffersPerShaderStage < 6) {
    return {supported: false, reason: 'storage_buffer_limit', limit: adapter.limits.maxStorageBuffersPerShaderStage};
  }
  // The SpMV dispatch saturates this dimension exactly, so a device reporting less
  // would reject it; refuse early with a readable reason instead of failing inside
  // pipeline creation.
  if (adapter.limits.maxComputeWorkgroupsPerDimension < STRIDE) {
    return {supported: false, reason: 'workgroup_limit', limit: adapter.limits.maxComputeWorkgroupsPerDimension};
  }
  return {
    supported: true,
    requiredLimits: {
      maxStorageBuffersPerShaderStage: Math.min(16, adapter.limits.maxStorageBuffersPerShaderStage),
    },
    adapterInfo: adapter.info ? {...adapter.info} : {},
    adapter,
  };
}

export class ConnectomeBrain {
  static async load(baseUrl, {progress, concurrency = 4, support} = {}) {
    const report = (stage, done, total) => progress && progress(stage, done, total);
    // Every package path resolves against this base, so it must end in a slash.
    // A deployer who omits it would otherwise silently resolve one directory up.
    if (!baseUrl.pathname.endsWith('/')) baseUrl = new URL(baseUrl.pathname + '/', baseUrl);
    report('meta');
    const meta = await (await fetch(new URL('meta.json', baseUrl))).json();
    const keys = Object.keys(meta.files);
    const buffers = {};
    let done = 0;
    await mapConcurrent(keys, concurrency, async (key) => {
      const entry = meta.files[key];
      const raw = await (await fetch(new URL(entry.path, baseUrl))).arrayBuffer();
      // Verification needs a secure context; over plain HTTP crypto.subtle is
      // absent and the package would load unchecked, so the page says so.
      if (entry.sha256 && crypto.subtle) {
        const hex = await sha256Hex(raw);
        if (hex !== entry.sha256) throw new Error(`sha256 mismatch for ${key}`);
      }
      buffers[key] = new FILE_KINDS[key].type(raw);
      report('weights', ++done, keys.length);
    });
    // Reuse the caller's capability probe so the adapter is only requested once.
    return ConnectomeBrain.fromBuffers(meta, buffers, support || {});
  }

  static fromBuffers(meta, buffers, {adapter, requiredLimits} = {}) {
    for (const [key, buffer] of Object.entries(buffers)) {
      const spec = FILE_KINDS[key];
      if (!spec) throw new Error(`unknown weight file: ${key}`);
      if (buffer.length !== spec.size(meta)) {
        throw new Error(`${key}: expected ${spec.size(meta)} elements, got ${buffer.length}`);
      }
    }

    const n = meta.n;
    const m = meta.sample_count || 0;
    const gatherCount = m + meta.outputs_count;

    const instance = new ConnectomeBrain();
    instance.meta = meta;
    instance.destroyed = false;

    return (async () => {
      if (!adapter) {
        const support = await checkSupport();
        if (!support.supported) throw new Error(`webgpu unsupported: ${support.reason}`);
        requiredLimits = support.requiredLimits;
        adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
      }
      instance.device = await adapter.requestDevice({requiredLimits});
      const device = instance.device;
      instance.errors = [];
      device.addEventListener('uncapturederror', (e) => instance.errors.push(e.error.message));

      const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
      const mkBuf = (data, usage) => {
        const b = device.createBuffer({size: data.byteLength, usage});
        device.queue.writeBuffer(b, 0, data);
        return b;
      };
      const actUsage = STORAGE | GPUBufferUsage.COPY_SRC;

      instance.bValues = mkBuf(buffers.values, STORAGE);
      instance.bPre = mkBuf(buffers.pre, STORAGE);
      instance.bPtr = mkBuf(buffers.ptr, STORAGE);
      instance.bS = device.createBuffer({size: n * 4, usage: STORAGE});
      device.queue.writeBuffer(instance.bS, 0, buffers.bias);
      instance.bSp = device.createBuffer({size: n * 4, usage: actUsage});
      instance.bActA = device.createBuffer({size: n * 4, usage: actUsage});
      instance.bActB = device.createBuffer({size: n * 4, usage: actUsage});
      instance.zero = new Float32Array(n);

      const uniformDesc = {size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC};
      instance.bU = device.createBuffer(uniformDesc);
      instance.bU2 = device.createBuffer(uniformDesc);
      instance.bGU = device.createBuffer(uniformDesc);
      device.queue.writeBuffer(instance.bU, 0, new Uint32Array([n, 0, 0, 0]));
      device.queue.writeBuffer(instance.bU2, 0, new Uint32Array([n, 0, 0, 0]));
      device.queue.writeBuffer(instance.bGU, 0, new Uint32Array([gatherCount, 0, 0, 0]));

      instance.gatherIndices = null;
      if (m > 0) {
        const all = new Uint32Array(gatherCount);
        all.set(new Uint32Array(buffers.sample.buffer, 0, m), 0);
        all.set(new Uint32Array(buffers.outputs.buffer, 0, meta.outputs_count), m);
        instance.gatherIndices = mkBuf(all, STORAGE);
      } else {
        instance.gatherIndices = mkBuf(new Uint32Array(buffers.outputs.buffer, 0, meta.outputs_count), STORAGE);
      }
      instance.bGathered = device.createBuffer({size: gatherCount * 4, usage: STORAGE | GPUBufferUsage.COPY_SRC});
      instance.staging = device.createBuffer({
        size: gatherCount * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});

      instance.module = device.createShaderModule({code: SHADER});
      const vis = GPUShaderStage.COMPUTE;
      const storage = (i) => ({binding: i, visibility: vis, buffer: {type: 'read-only-storage'}});
      const storageRW = (i) => ({binding: i, visibility: vis, buffer: {type: 'storage'}});
      const uniformB = (i) => ({binding: i, visibility: vis, buffer: {type: 'uniform'}});
      const spLayout = device.createBindGroupLayout({entries: [
        uniformB(0), storage(1), storage(2), storage(3), storage(4), storageRW(5)]});
      const upLayout = device.createBindGroupLayout({entries: [
        uniformB(6), storage(7), storage(8), storage(9), storageRW(10)]});
      const gaLayout = device.createBindGroupLayout({entries: [
        uniformB(11), storage(12), storage(13), storageRW(14)]});

      instance.spPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: [spLayout]}),
        compute: {module: instance.module, entryPoint: 'spmv'}});
      instance.upPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: [upLayout]}),
        compute: {module: instance.module, entryPoint: 'update'}});
      instance.gaPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({bindGroupLayouts: [gaLayout]}),
        compute: {module: instance.module, entryPoint: 'gather'}});

      const mkBG = (actIn, actOut) => [
        device.createBindGroup({layout: spLayout, entries: [
          {binding: 0, resource: {buffer: instance.bU}},
          {binding: 1, resource: {buffer: instance.bValues}},
          {binding: 2, resource: {buffer: instance.bPre}},
          {binding: 3, resource: {buffer: instance.bPtr}},
          {binding: 4, resource: {buffer: actIn}},
          {binding: 5, resource: {buffer: instance.bSp}},
        ]}),
        device.createBindGroup({layout: upLayout, entries: [
          {binding: 6, resource: {buffer: instance.bU2}},
          {binding: 7, resource: {buffer: instance.bSp}},
          {binding: 8, resource: {buffer: instance.bS}},
          {binding: 9, resource: {buffer: actIn}},
          {binding: 10, resource: {buffer: actOut}},
        ]}),
      ];
      instance.BG = [mkBG(instance.bActA, instance.bActB), mkBG(instance.bActB, instance.bActA)];
      instance.gaBG = [0, 1].map((k) => device.createBindGroup({layout: gaLayout, entries: [
        {binding: 11, resource: {buffer: instance.bGU}},
        {binding: 12, resource: {buffer: instance.gatherIndices}},
        {binding: 13, resource: {buffer: k === 0 ? instance.bActA : instance.bActB}},
        {binding: 14, resource: {buffer: instance.bGathered}},
      ]}));

      instance.n = n;
      instance.neuralSteps = meta.neural_steps;
      instance.observationSize = meta.observation_size;
      instance.actionSize = meta.action_size;
      instance.inputsCount = meta.inputs_count;
      instance.outputsCount = meta.outputs_count;
      instance.sampleCount = m;
      instance.gatherCount = gatherCount;

      instance.buffers = buffers;
      instance._weights = null;
      instance._obs = new Float32Array(meta.observation_size);
      instance._signal = new Float32Array(n);
      return instance;
    })();
  }

  // One decision: 4 rate updates from zero activity, then LayerNorm + readout.
  // obs: Float32Array(observationSize). Returns {action, outputActivity,
  //   sampleActivity, outputRms, gatherMs}.
  async infer(obs, {lesion = false} = {}) {
    if (this.destroyed) throw new Error('brain destroyed');
    const d = this.device;

    const om = this.buffers.obs_mean, os = this.buffers.obs_std;
    const x = this._obs;
    for (let i = 0; i < this.observationSize; i++) {
      x[i] = Math.min(10, Math.max(-10, (obs[i] - om[i]) / os[i]));
    }
    const enc = this.buffers.encoder;
    const sig = this._signal;
    sig.set(this.buffers.bias);
    const inputs = this.buffers.inputs;
    for (let j = 0; j < this.inputsCount; j++) {
      let acc = 0;
      const off = j * this.observationSize;
      for (let i = 0; i < this.observationSize; i++) acc += enc[off + i] * x[i];
      sig[inputs[j]] += acc;
    }

    d.queue.writeBuffer(this.bU2, 0, new Uint32Array([this.n, lesion ? 1 : 0, 0, 0]));
    d.queue.writeBuffer(this.bS, 0, sig);
    d.queue.writeBuffer(this.bActA, 0, this.zero);

    const cmd = d.createCommandEncoder();
    const pass = cmd.beginComputePass();
    let cur = 0;
    for (let step = 0; step < this.neuralSteps; step++) {
      if (!lesion) {
        pass.setPipeline(this.spPipeline);
        pass.setBindGroup(0, this.BG[cur][0]);
        pass.dispatchWorkgroups(STRIDE);
      }
      pass.setPipeline(this.upPipeline);
      pass.setBindGroup(0, this.BG[cur][1]);
      pass.dispatchWorkgroups(Math.ceil(this.n / WG));
      cur = 1 - cur;
    }
    pass.setPipeline(this.gaPipeline);
    pass.setBindGroup(0, this.gaBG[cur === 0 ? 0 : 1]);
    pass.dispatchWorkgroups(Math.ceil(this.gatherCount / WG));
    pass.end();
    cmd.copyBufferToBuffer(this.bGathered, 0, this.staging, 0, this.gatherCount * 4);
    d.queue.submit([cmd.finish()]);

    const t0 = performance.now();
    await this.staging.mapAsync(GPUMapMode.READ);
    const gathered = new Float32Array(this.staging.getMappedRange().slice(0));
    this.staging.unmap();
    const gatherMs = performance.now() - t0;

    const outAct = gathered.subarray(this.sampleCount);
    const nw = this.buffers.norm_w, nb = this.buffers.norm_b;
    let mean = 0;
    for (const v of outAct) mean += v;
    mean /= this.outputsCount;
    let varr = 0;
    for (const v of outAct) varr += (v - mean) ** 2;
    const std = Math.sqrt(varr / this.outputsCount + 1e-5);
    const rw = this.buffers.readout, rb = this.buffers.readout_bias;
    const action = new Float32Array(this.actionSize);
    for (let a = 0; a < this.actionSize; a++) {
      let acc = rb[a];
      const off = a * this.outputsCount;
      for (let k = 0; k < this.outputsCount; k++) acc += rw[off + k] * ((outAct[k] - mean) / std * nw[k] + nb[k]);
      action[a] = acc;
    }
    let rms = 0;
    for (const v of outAct) rms += v * v;
    rms = Math.sqrt(rms / this.outputsCount);

    return {
      action,
      outputActivity: outAct.slice(),
      sampleActivity: gathered.slice(0, this.sampleCount),
      outputRms: rms,
      gatherMs,
    };
  }

  destroy() {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const b of [this.bValues, this.bPre, this.bPtr, this.bS, this.bSp,
      this.bActA, this.bActB, this.bU, this.bU2, this.bGU, this.gatherIndices,
      this.bGathered, this.staging]) {
      if (b) b.destroy();
    }
    this.device.destroy();
  }
}
