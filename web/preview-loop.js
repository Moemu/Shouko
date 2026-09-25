// Browser-side closed control loop. Owns the 20ms control tick: observation →
// ConnectomeBrain inference → BodyWasm physics step, with local commands
// (speed/yaw/push/lesion/reset) and automatic slow-motion when inference plus
// physics exceed the real-time budget. Mirrors the app/studio_server.py loop.

// Fallback only. The real control period comes from the checkpoint's own interface
// (simulation_dt x control_decimation) so that a re-export with a different
// decimation cannot silently desync the browser loop from the server's semantics.
const DEFAULT_CONTROL_MS = 20;
const YAW_GAIN = 1.4;
const YAW_CLAMP = 0.2;

const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

export class PreviewLoop {
  constructor({brain, body, initialQvel, speed = 0.5, yaw = 0}) {
    this.brain = brain;
    this.body = body;
    // Take the control period from the body contract rather than a second copy of it.
    const contract = body && body.cfg;
    this.controlMs = contract ? contract.simulation_dt * contract.control_decimation * 1000 : DEFAULT_CONTROL_MS;
    this.initialQvel = initialQvel;
    this.speed = speed;
    this.yaw = yaw;
    this.lesion = false;
    this.running = false;
    this.episode = 1;
    this.activity = new Float32Array(brain.sampleCount || 0);
    this.outputRms = 0;
    this.brainMs = 0;
    this.rtf = 0;
    this.lastError = null;
    this._stopping = false;
    this._listeners = new Set();
  }

  onState(listener) {
    this._listeners.add(listener);
    return () => this._listeners.delete(listener);
  }

  _emit() {
    const body = this.body;
    const [, , , yaw] = body.observation();
    const g = body._gravity;
    const snapshot = {
      running: this.running,
      episode: this.episode,
      time: body.data.time,
      qpos: Array.from(body.data.qpos),
      velocity: [body.data.qvel[0], body.data.qvel[1], body.data.qvel[2]],
      yaw,
      upright: -g[2],
      height: body.data.qpos[2],
      distance: body.distance,
      contacts: body.contacts,
      foot_strikes: body.footStrikes,
      fallen: body.data.qpos[2] < body.cfg.fall_height || -g[2] < 0.45,
      activity: this.activity,
      output_rms: this.outputRms,
      brain_ms: this.brainMs,
      real_time_factor: this.rtf,
      target_speed: this.speed,
      target_yaw: this.yaw,
      lesion: this.lesion ? 'disconnected' : 'intact',
    };
    for (const listener of this._listeners) listener(snapshot);
  }

  setSpeed(v) { this.speed = v; }
  setYaw(rad) { this.yaw = rad; }
  setLesion(disconnected) { this.lesion = disconnected; }
  push() { this.body.data.qvel[1] += 0.25; }
  reset() {
    this.body.reset(this.initialQvel);
    this.episode += 1;
    this.rtf = 0;
  }

  async start() {
    this.running = true;
    this._stopping = false;
    let previous = performance.now();
    while (!this._stopping) {
      const t0 = performance.now();
      const dt = (t0 - previous) / 1000;
      previous = t0;
      if (this.running) {
        try {
          const [, , , yaw] = this.body.observation();
          const error = Math.atan2(Math.sin(this.yaw - yaw), Math.cos(this.yaw - yaw));
          const obs = this.body.motorObservation([this.speed, 0, clip(error * YAW_GAIN, -YAW_CLAMP, YAW_CLAMP)]);
          const tBrain = performance.now();
          const result = await this.brain.infer(obs, {lesion: this.lesion});
          this.brainMs = performance.now() - tBrain;
          if (result.sampleActivity) this.activity = result.sampleActivity;
          this.outputRms = result.outputRms;
          this.body.stepJoints(result.action);
          this.lastError = null;
        } catch (error) {
          this.lastError = error;
          this.running = false;
        }
      }
      const wall = performance.now() - t0;
      this.rtf = this.running ? this.controlMs / Math.max(wall, this.controlMs) : 0;
      this._emit();
      const wait = this.controlMs - (performance.now() - t0);
      // Over budget: yield a fixed slice so rendering and input stay responsive
      // instead of saturating the main thread with back-to-back ticks.
      await new Promise((resolve) => setTimeout(resolve, wait < 0 ? 6 : Math.max(0, wait)));
      if (wait < 0) previous = performance.now();
    }
  }

  stop() {
    this._stopping = true;
    this.running = false;
  }
}
