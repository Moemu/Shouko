// Browser-side MuJoCo body (Yumi). Port of the app/sim.py preview path
// (Body with load_motor_policy=False): reset, motor_observation, step_joints,
// fall detection. The numeric contract (home, scales, decimation) comes from
// the checkpoint's physics_interface via the config argument, so the browser
// body cannot silently drift from the trained interface.
//
// config fields: simulation_dt, control_decimation, kps, kds, home,
//   action_scale, cmd_scale, ang_vel_scale, dof_vel_scale, gait_period_s,
//   linear_velocity_scale, height_reference, initial_height, fall_height,
//   observation_size (50 adds base velocity + height error fields).
//
// reset(initialQvel) takes the two base-velocity perturbations that Python
// draws with np.random.default_rng(seed).normal(0, 0.015, 2); the caller
// supplies them so both sides start from identical state.

const fround = Math.fround;

export class BodyWasm {
  static async load({glueUrl, wasmUrl, sceneUrl, includeUrl, config}) {
    if (!glueUrl) throw new Error('BodyWasm.load needs a glueUrl (the MuJoCo-WASM module)');
    const mod = await import(glueUrl);
    const factory = mod.default || mod;
    // The glue resolves its wasm through locateFile; pointing that at the
    // package's content-addressed copy keeps both files verifiable and cacheable
    // instead of forcing them to share a directory under fixed names.
    const mujoco = await factory(wasmUrl
      ? {locateFile: (path) => (path.endsWith('.wasm') ? wasmUrl : path)}
      : {});
    const sceneXml = await (await fetch(sceneUrl)).text();
    const includeXml = await (await fetch(includeUrl)).text();
    if (mujoco.FS && mujoco.FS.writeFile) {
      mujoco.FS.writeFile('/scene.xml', sceneXml);
      const includeName = sceneXml.match(/<include file="([^"]+)"/);
      const path = '/' + (includeName ? includeName[1] : 'body.xml');
      mujoco.FS.writeFile(path, includeXml);
      return new BodyWasm(mujoco, mujoco.MjModel.from_xml_path('/scene.xml'), config);
    }
    const inlined = sceneXml.replace(/<include file="[^"]*"\/>/, includeXml.replace(/<\?xml[^>]*\?>/, ''));
    return new BodyWasm(mujoco, mujoco.MjModel.from_xml_string(inlined), config);
  }

  constructor(mujoco, model, config) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = new mujoco.MjData(model);
    this.cfg = config;
    model.opt.timestep = config.simulation_dt;
    model.opt.iterations = 50;
    model.opt.ls_iterations = 50;
    model.opt.tolerance = 1e-6;
    this.home = Float64Array.from(config.home);
    this.kp = Float64Array.from(config.kps);
    this.kd = Float64Array.from(config.kds);
    this.target = new Float64Array(12);
    this.torque = new Float64Array(12);
    this.action = new Float32Array(12);
    this._obs = new Float32Array(config.observation_size);
    this._gravity = new Float64Array(3);
    this.steps = 0;
    this.distance = 0;
    this.lastXy = [0, 0];
    this.contacts = [false, false];
    this.footStrikes = [0, 0];
  }

  reset(initialQvel) {
    const {mujoco, model, data, cfg} = this;
    mujoco.mj_resetData(model, data);
    for (let j = 0; j < 12; j++) data.qpos[7 + j] = this.home[j];
    data.qpos[2] = cfg.initial_height;
    data.qvel[0] = initialQvel[0];
    data.qvel[1] = initialQvel[1];
    this.action.fill(0);
    this.target.set(this.home);
    this.steps = 0;
    this.distance = 0;
    this.lastXy = [data.qpos[0], data.qpos[1]];
    this.contacts = [false, false];
    this.footStrikes = [0, 0];
    mujoco.mj_forward(model, data);
  }

  observation() {
    const q = this.data.qpos;
    const w = q[3], x = q[4], y = q[5], z = q[6];
    const g = this._gravity;
    g[0] = 2 * (-z * x + w * y);
    g[1] = -2 * (z * y + w * x);
    g[2] = 1 - 2 * (w * w + z * z);
    const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    return [g[0], g[1], g[2], yaw];
  }

  motorObservation(command) {
    const {data, cfg} = this;
    this.observation();
    const phase = data.time / cfg.gait_period_s * 2 * Math.PI;
    const o = this._obs;
    let k = 0;
    for (let j = 0; j < 3; j++) o[k++] = fround(data.qvel[3 + j] * cfg.ang_vel_scale);
    o[k++] = fround(this._gravity[0]);
    o[k++] = fround(this._gravity[1]);
    o[k++] = fround(this._gravity[2]);
    for (let j = 0; j < 3; j++) o[k++] = fround(command[j] * cfg.cmd_scale[j]);
    for (let j = 0; j < 12; j++) o[k++] = fround(data.qpos[7 + j] - this.home[j]);
    for (let j = 0; j < 12; j++) o[k++] = fround(data.qvel[6 + j] * cfg.dof_vel_scale);
    for (let j = 0; j < 12; j++) o[k++] = this.action[j];
    o[k++] = fround(Math.sin(phase));
    o[k++] = fround(Math.cos(phase));
    if (cfg.observation_size === 50) {
      for (let j = 0; j < 2; j++) o[k++] = fround(data.qvel[j] * cfg.linear_velocity_scale);
      o[k++] = fround(data.qpos[2] - cfg.height_reference);
    }
    return o;
  }

  stepJoints(action) {
    const {data, cfg} = this;
    if (action.length !== 12) throw new Error('Expected twelve joint actions');
    for (let j = 0; j < 12; j++) {
      if (!Number.isFinite(action[j])) throw new Error(`Non-finite joint action ${j}`);
      this.action[j] = fround(Math.min(8, Math.max(-8, action[j])));
      this.target[j] = fround(this.home[j] + this.action[j] * cfg.action_scale);
    }
    for (let d = 0; d < cfg.control_decimation; d++) {
      for (let j = 0; j < 12; j++) {
        this.torque[j] = this.kp[j] * (this.target[j] - data.qpos[7 + j]) - this.kd[j] * data.qvel[6 + j];
        data.ctrl[j] = this.torque[j];
      }
      this.mujoco.mj_step(this.model, data);
    }
    this.steps += 1;
    const xy = [data.qpos[0], data.qpos[1]];
    this.distance += Math.hypot(xy[0] - this.lastXy[0], xy[1] - this.lastXy[1]);
    this.lastXy = xy;
    this.updateContacts();
    return this.snapshot();
  }

  updateContacts() {
    const {model, data} = this;
    const contacts = [false, false];
    const cs = data.contact;
    if (!cs || cs.length === undefined) { this.footStrikes = [0, 0]; this.contacts = contacts; return; }
    for (let i = 0; i < cs.length; i++) {
      const c = cs[i];
      if (c.geom1 !== 0 && c.geom2 !== 0) continue;
      const other = c.geom1 === 0 ? c.geom2 : c.geom1;
      const bodyId = model.geom_bodyid ? model.geom_bodyid[other] : -1;
      const name = this.bodyName(bodyId);
      if (!name) continue;
      const side = name.includes('left') ? 0 : name.includes('right') ? 1 : -1;
      if (side >= 0 && name.includes('ankle')) contacts[side] = true;
    }
    for (let i = 0; i < 2; i++) {
      if (contacts[i] && !this.contacts[i]) this.footStrikes[i] += 1;
    }
    this.contacts = contacts;
  }

  bodyName(bodyId) {
    if (bodyId < 0) return null;
    if (!this._bodyNames) {
      this._bodyNames = [];
      const m = this.model;
      const count = m.nbody;
      for (let b = 0; b < count; b++) {
        try { this._bodyNames[b] = (m.body(b) && m.body(b).name) || (m.body_names && m.body_names[b]) || ''; }
        catch { this._bodyNames[b] = (m.body_names && m.body_names[b]) || ''; }
      }
    }
    return this._bodyNames[bodyId] || null;
  }

  snapshot() {
    const [, , , yaw] = this.observation();
    const g = this._gravity;
    return {
      time: this.data.time,
      height: this.data.qpos[2],
      distance: this.distance,
      yaw,
      upright: -g[2],
      fallen: this.data.qpos[2] < this.cfg.fall_height || -g[2] < 0.45,
      contacts: this.contacts,
      footStrikes: this.footStrikes,
    };
  }

  destroy() {
    this.data && this.data.destroy && this.data.destroy();
    this.model && this.model.destroy && this.model.destroy();
  }
}
