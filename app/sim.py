"""MuJoCo G1 motor plant. Adapted from Unitree RL Gym (BSD-3-Clause)."""
from pathlib import Path
import time
import json

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
UNITREE = ROOT / "vendor/unitree_rl_gym"
torch.set_num_threads(1)


ROBOTS = {
    "g1": dict(
        cfg=UNITREE / "deploy/deploy_mujoco/configs/g1.yaml",
        xml=UNITREE / "resources/robots/g1_description/scene.xml",
        initial_height=0.79, fall_height=0.45, hips_height=0.793),
    "yumi": dict(
        cfg=ROOT / "app/yumi_description/yumi.yaml",
        xml=ROOT / "app/yumi_description/scene.xml",
        initial_height=0.962, fall_height=0.55, hips_height=0.97231),
}

# 观测/动作接口键：这些值定义策略看到和输出的尺度（home 基准、缩放）。
# 它们是模型接口的一部分，与权重一起进检查点并在加载时比对，
# 防止 2026-09-16 那类 home 漂移静默错配（见 research/experiments/HOME_REFERENCE_MISMATCH_20260916.md）。
INTERFACE_KEYS = ("default_angles", "action_scale", "cmd_scale",
                  "ang_vel_scale", "dof_vel_scale", "control_decimation")


def observation_interface(cfg):
    """Extract the observation/action interface from a body config."""
    interface = {}
    for key in INTERFACE_KEYS:
        if key not in cfg:
            continue
        value = cfg[key]
        interface[key] = [float(v) for v in value] if isinstance(value, (list, tuple)) else float(value)
    return interface


def apply_interface(cfg, interface):
    """Return a copy of cfg with the interface values overridden (checkpoint-driven home etc.)."""
    merged = dict(cfg)
    merged.update(interface or {})
    return merged


class Body:
    def __init__(self, load_motor_policy=True, robot="g1", interface=None):
        spec = ROBOTS[robot]
        self.robot = robot
        self.initial_height = spec["initial_height"]
        self.fall_height = spec["fall_height"]
        self.hips_height = spec["hips_height"]
        self.cfg = apply_interface(yaml.safe_load(spec["cfg"].read_text(encoding="utf-8")), interface)
        self.interface = observation_interface(self.cfg)
        self.model = mujoco.MjModel.from_xml_path(str(spec["xml"]))
        self.model.opt.timestep = self.cfg["simulation_dt"]
        if not load_motor_policy:
            mesh = self.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
            self.model.geom_contype[mesh] = 0
            self.model.geom_conaffinity[mesh] = 0
            self.model.opt.iterations = 50
            self.model.opt.ls_iterations = 50
            self.model.opt.tolerance = 1e-6
        self.data = mujoco.MjData(self.model)
        self.policy = (torch.jit.load(str(UNITREE / "deploy/pre_train/g1/motion.pt"), map_location="cpu").eval()
                       if load_motor_policy else None)
        self.home = np.array(self.cfg["default_angles"], dtype=np.float32)
        self.kp = np.array(self.cfg["kps"])
        self.kd = np.array(self.cfg["kds"])
        self.reset()

    def reset(self, seed=0):
        if self.policy is not None:
            self.policy.hidden_state.zero_()
            self.policy.cell_state.zero_()
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:] = self.home
        self.data.qpos[2] = self.initial_height
        rng = np.random.default_rng(seed)
        self.data.qvel[:2] = rng.normal(0, 0.015, 2)
        self.action = np.zeros(12, dtype=np.float32)
        self.target = self.home.copy()
        self.steps = 0
        self.distance = 0.0
        self.last_xy = self.data.qpos[:2].copy()
        self.contacts = [False, False]
        self.foot_strikes = [0, 0]
        mujoco.mj_forward(self.model, self.data)

    def observation(self):
        w, x, y, z = self.data.qpos[3:7]
        gravity = np.array([2*(-z*x+w*y), -2*(z*y+w*x), 1-2*(w*w+z*z)])
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        return gravity, float(yaw)

    def motor_observation(self, command):
        gravity, _ = self.observation()
        phase = self.data.time / 1.0 * 2*np.pi
        return np.concatenate([
            self.data.qvel[3:6] * self.cfg["ang_vel_scale"], gravity,
            np.asarray(command) * self.cfg["cmd_scale"],
            self.data.qpos[7:] - self.home,
            self.data.qvel[6:] * self.cfg["dof_vel_scale"], self.action,
            [np.sin(phase), np.cos(phase)],
        ]).astype(np.float32)

    def step(self, command):
        if self.policy is None:
            raise RuntimeError('This body accepts direct joint actions through step_joints().')
        obs = self.motor_observation(command)
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(obs).unsqueeze(0)).numpy().reshape(-1)
        return self.step_joints(action)

    def step_joints(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (12,) or not np.all(np.isfinite(action)):
            raise ValueError('Expected twelve finite joint actions')
        self.action = np.clip(action, -8, 8)
        self.target = self.home + self.action * self.cfg["action_scale"]
        for _ in range(self.cfg["control_decimation"]):
            torque = self.kp * (self.target-self.data.qpos[7:]) - self.kd*self.data.qvel[6:]
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        xy = self.data.qpos[:2].copy()
        self.distance += float(np.linalg.norm(xy-self.last_xy))
        self.last_xy = xy
        contacts = [False, False]
        for c in self.data.contact:
            if 0 not in (c.geom1, c.geom2):
                continue
            body_id = self.model.geom_bodyid[c.geom2 if c.geom1 == 0 else c.geom1]
            name = self.model.body(body_id).name
            for i, side in enumerate(("left", "right")):
                if side in name and "ankle" in name:
                    contacts[i] = True
        self.foot_strikes = [n + int(now and not old) for n, now, old in zip(self.foot_strikes,contacts,self.contacts)]
        self.contacts = contacts
        return self.snapshot()

    def snapshot(self):
        gravity, yaw = self.observation()
        return {
            "time": float(self.data.time), "qpos": self.data.qpos.tolist(),
            "velocity": self.data.qvel[:3].tolist(), "yaw": yaw,
            "upright": float(-gravity[2]), "height": float(self.data.qpos[2]),
            "distance": self.distance, "contacts": self.contacts,
            "foot_strikes": self.foot_strikes,
            "fallen": bool(self.data.qpos[2] < self.fall_height or -gravity[2] < 0.45),
            "torque_rms": float(np.sqrt(np.mean(self.data.ctrl**2))),
            "joints": self.data.xpos[1:].tolist(),
        }


if __name__ == "__main__":
    body = Body()
    start = time.perf_counter()
    minimum_height = 10
    for _ in range(1500):
        state = body.step([0.5, 0, 0])
        minimum_height = min(minimum_height,state["height"])
        if state["fallen"]:
            break
    report = {**state, "wall_seconds": time.perf_counter()-start, "minimum_height": minimum_height,
              "baseline": "frozen Unitree G1 policy; no fly controller"}
    (ROOT / "runs/motor_baseline.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
