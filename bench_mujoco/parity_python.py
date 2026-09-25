"""Python-side reference for the MuJoCo-WASM spike: 1000 control steps of the
Yumi preview path (app/sim.py Body, load_motor_policy=False) with a fixed
action schedule and fixed initial qvel, recorded to parity_python.json.

qvel override replaces Body.reset's numpy RNG draw so the browser side can
reproduce the initial state without porting numpy's generator.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from app.sim import Body

STEPS = 1000
SAMPLE_EVERY = 20


def main():
    body = Body(load_motor_policy=False, robot="yumi")
    body.reset(4242)
    body.data.qvel[0] = 0.01
    body.data.qvel[1] = -0.02
    mujoco.mj_forward(body.model, body.data)

    records = []
    for k in range(STEPS):
        action = np.array([0.5 * np.sin(0.2 * k + j) for j in range(12)], dtype=np.float32)
        state = body.step_joints(action)
        if k % SAMPLE_EVERY == 0 or k == STEPS - 1:
            records.append({
                "k": k,
                "time": state["time"],
                "qpos": state["qpos"],
                "contacts": state["contacts"],
                "fallen": state["fallen"],
            })

    cfg = body.cfg
    out = {
        "steps": STEPS,
        "sample_every": SAMPLE_EVERY,
        "config": {
            "timestep": cfg["simulation_dt"],
            "decimation": cfg["control_decimation"],
            "kps": body.kp.tolist(),
            "kds": body.kd.tolist(),
            "home": body.home.tolist(),
            "action_scale": cfg["action_scale"],
            "initial_height": body.initial_height,
        },
        "model": {
            "nq": body.model.nq,
            "nv": body.model.nv,
            "nu": body.model.nu,
            "opt_timestep": float(body.model.opt.timestep),
            "opt_iterations": int(body.model.opt.iterations),
            "opt_ls_iterations": int(body.model.opt.ls_iterations),
            "opt_tolerance": float(body.model.opt.tolerance),
            "mesh_contype_zeroed": int(((body.model.geom_type == mujoco.mjtGeom.mjGEOM_MESH).sum())),
        },
        "records": records,
    }
    dest = Path(__file__).parent / "parity_python.json"
    dest.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {dest} with {len(records)} records; final time={records[-1]['time']:.3f}s "
          f"height={records[-1]['qpos'][2]:.4f} fallen={records[-1]['fallen']}")


if __name__ == "__main__":
    main()
