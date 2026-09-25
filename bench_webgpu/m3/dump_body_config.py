"""Dump the numeric body contract (checkpoint-driven interface + yaml kps/kds) for browser parity."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.sim import Body
from app.full_brain import checkpoint_configuration

configuration = checkpoint_configuration(ROOT / 'runs/yumi/best.pt')
body = Body(load_motor_policy=False, robot='yumi',
            interface=configuration['physics_interface'],
            observation_size=configuration['observation_size'])
cfg = body.cfg
out = dict(
    robot='yumi', initial_height=body.initial_height, fall_height=body.fall_height,
    simulation_dt=cfg['simulation_dt'], control_decimation=cfg['control_decimation'],
    kps=[float(v) for v in body.kp], kds=[float(v) for v in body.kd],
    home=[float(v) for v in body.home], action_scale=float(cfg['action_scale']),
    cmd_scale=[float(v) for v in cfg['cmd_scale']],
    ang_vel_scale=float(cfg['ang_vel_scale']), dof_vel_scale=float(cfg['dof_vel_scale']),
    gait_period_s=float(cfg['gait_period_s']),
    linear_velocity_scale=float(cfg['linear_velocity_scale']),
    height_reference=float(cfg['height_reference']),
    observation_size=body.observation_size)
target = Path(__file__).parent / 'body_config.json'
target.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
