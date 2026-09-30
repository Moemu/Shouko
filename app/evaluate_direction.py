"""Hash-bound native direction evaluation and bounded command/action probes."""
import argparse
from contextlib import nullcontext
import hashlib
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .direction_metrics import direction_summary
from .evaluate_locomotion import gait_summary, load_policy, summarize
from .full_brain import ROOT
from .ordered_inference import ordered_inference
from .sim import Body
from .train_full import atomic_json, file_sha256


@torch.inference_mode()
def evaluate(policy, cfg, kind, args, seeds, yaw):
    bodies = [Body(False, 'yumi', interface=cfg['physics_interface'], observation_size=cfg['observation_size']) for _ in seeds]
    for body, seed in zip(bodies, seeds):
        body.reset(seed)
        body.data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        mujoco.mj_forward(body.model, body.data)
    dt = bodies[0].model.opt.timestep*bodies[0].cfg['control_decimation']
    speeds = [.35, .5, .65]*3
    alive = np.ones(9, dtype=bool)
    upright = np.ones(9)
    alternations = np.zeros(9, dtype=int)
    previous_side, previous_contacts = [None]*9, [[False, False] for _ in seeds]
    series = [dict(contacts=[], clearance=[], feet_xy=[], yaw=[], y=[], height=[], knees=[]) for _ in seeds]
    traces = [dict(time=[0.0], qpos=[b.data.qpos.copy()], qvel=[b.data.qvel.copy()], yaw=[yaw], actions=[]) for b in bodies]
    feet = [[np.flatnonzero(b.model.geom_bodyid == b.model.body(n).id) for n in ['left_ankle_roll_link', 'right_ankle_roll_link']] for b in bodies]
    offset = np.zeros(12, dtype=np.float32)
    offset[[1, 7]] = args.hip_roll_offset
    offset[[5, 11]] = args.ankle_roll_offset
    digest, started = hashlib.sha256(), time.monotonic()
    last = [None]*9
    with ordered_inference() if kind == 'connectome' and args.sparse_backend == 'ordered' else nullcontext():
        for step in range(round(args.seconds/dt)):
            if args.push_velocity and step == round(args.push_time/dt):
                for body in bodies:
                    body.data.qvel[1] += args.push_velocity
            obs = np.stack([b.motor_observation([v, args.vy_command, float(np.clip(-b.observation()[1]*args.yaw_feedback_gain, -.2, .2))]) for b, v in zip(bodies, speeds)])
            actions = policy(torch.from_numpy(obs).cuda())[0].cpu().numpy() + offset
            for i, body in enumerate(bodies):
                if not alive[i]:
                    continue
                state = body.step_joints(actions[i]); last[i] = state
                upright[i] = min(upright[i], state['upright'])
                for side in range(2):
                    if state['contacts'][side] and not previous_contacts[i][side]:
                        alternations[i] += int(previous_side[i] is not None and previous_side[i] != side)
                        previous_side[i] = side
                previous_contacts[i] = state['contacts'].copy()
                alive[i] = not state['fallen']
                s, tr = series[i], traces[i]
                s['contacts'].append(state['contacts'])
                s['clearance'].append([float(np.min(body.data.geom_xpos[g, 2]-body.model.geom_size[g, 0])) for g in feet[i]])
                s['feet_xy'].append([body.data.geom_xpos[g, :2].mean(axis=0).tolist() for g in feet[i]])
                s['yaw'].append(state['yaw']); s['y'].append(state['qpos'][1])
                s['height'].append(state['height']); s['knees'].append([state['qpos'][10], state['qpos'][16]])
                tr['time'].append(state['time']); tr['qpos'].append(body.data.qpos.copy())
                tr['qvel'].append(body.data.qvel.copy()); tr['yaw'].append(state['yaw']); tr['actions'].append(actions[i].copy())
            digest.update(np.stack([b.data.qpos for b in bodies]).tobytes())
            digest.update(np.stack([b.data.qvel for b in bodies]).tobytes())
            if not alive.any():
                break
    tests = []
    for i, (seed, speed, state) in enumerate(zip(seeds, speeds, last)):
        duration = state['time']; velocity = state['qpos'][0]/duration
        lateral = abs(state['qpos'][1]); gait = gait_summary(series[i], dt)
        criteria = dict(survived=bool(alive[i] and duration >= args.seconds-.01), speed=bool(abs(velocity-speed) <= .15),
                        feet=bool(min(state['foot_strikes']) >= int(args.seconds*.5)), alternation=bool(alternations[i] >= int(args.seconds)),
                        upright=bool(upright[i] >= .7), direction=bool(lateral <= max(1, args.seconds*speed*.2)))
        tr = traces[i]
        direction = direction_summary(tr['time'], tr['qpos'], tr['yaw'], args.push_time if args.push_velocity else 0,
                                      bodies[i].cfg['gait_period_s'])
        strict = all(criteria.values()) and all(f['qualifying_swings_per_second'] >= 1 for f in gait['feet'])
        tests.append(dict(seed=seed, target_speed=speed, initial_yaw=yaw, seconds=duration, mean_speed=velocity, lateral_m=lateral,
                          fallen=not bool(alive[i]), criteria=criteria, success=all(criteria.values()), strict_success=strict,
                          direction_success=bool(strict and direction['maximum_lateral_m'] <= args.corridor), gait=gait, direction=direction))
    return dict(tests=tests, trajectory_sha256=digest.hexdigest(), wall_seconds=time.monotonic()-started), traces


def main(args):
    torch.set_num_threads(8)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(kind='native_direction_v1', arguments=vars(args), completed=False, conditions=[],
                  source_sha256={n: file_sha256(ROOT/'app'/n) for n in ['evaluate_direction.py', 'direction_metrics.py', 'evaluate_locomotion.py', 'sim.py', 'full_brain.py', 'ordered_inference.py']},
                  body_xml_sha256={str(p.relative_to(ROOT)): file_sha256(p) for p in (ROOT/'app/yumi_description').glob('*.xml')})
    atomic_json(output, report)
    for checkpoint in args.checkpoints:
        policy, cfg, kind = load_policy(checkpoint)
        condition = dict(checkpoint=checkpoint, checkpoint_sha256=file_sha256(checkpoint), policy_kind=kind,
                         physics_interface=cfg['physics_interface'], results=[])
        report['conditions'].append(condition)
        for cohort in range(args.cohorts):
            seeds = list(range(args.seed_base+cohort*10, args.seed_base+cohort*10+9))
            for yaw in args.yaws:
                row, traces = evaluate(policy, cfg, kind, args, seeds, yaw)
                if args.trace:
                    trace_path = output.with_name(f'{output.stem}_m{len(report["conditions"])-1}_c{cohort}_yaw{yaw}.npz')
                    np.savez_compressed(trace_path, **{f'episode{i}_{key}': np.asarray(value) for i, tr in enumerate(traces) for key, value in tr.items()})
                    row.update(trace=trace_path.name, trace_sha256=file_sha256(trace_path))
                condition['results'].append(dict(cohort=cohort, initial_yaw=yaw, **row))
                atomic_json(output, report)
        tests = [t for row in condition['results'] for t in row['tests']]
        condition['summary'] = dict(**summarize(tests), strict_passes=sum(t['strict_success'] for t in tests),
                                    direction_passes=sum(t['direction_success'] for t in tests))
        del policy
        torch.cuda.empty_cache()
        atomic_json(output, report)
    report['completed'] = True
    atomic_json(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--cohorts', type=int, default=1)
    parser.add_argument('--seed-base', type=int, default=310001)
    parser.add_argument('--yaws', nargs='+', type=float, default=[0.0])
    parser.add_argument('--push-velocity', type=float, default=0)
    parser.add_argument('--push-time', type=float, default=10)
    parser.add_argument('--vy-command', type=float, default=0)
    parser.add_argument('--yaw-feedback-gain', type=float, default=1.4)
    parser.add_argument('--hip-roll-offset', type=float, default=0)
    parser.add_argument('--ankle-roll-offset', type=float, default=0)
    parser.add_argument('--corridor', type=float, default=2)
    parser.add_argument('--sparse-backend', choices=['ordered', 'native'], default='ordered')
    parser.add_argument('--trace', action='store_true')
    main(parser.parse_args())
