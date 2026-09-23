"""Matched native MuJoCo evaluation with repeatable inference and gait diagnostics."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from .full_brain import ConnectomePolicy, ROOT, checkpoint_configuration
from .mlp_policy import MLPPolicy
from .ordered_inference import ordered_inference
from .sim import Body
from .train_full import atomic_json, file_sha256


def load_policy(path):
    configuration = checkpoint_configuration(path)
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    kind = saved.get('policy_kind', 'connectome')
    if kind == 'mlp':
        model = MLPPolicy(configuration['observation_size'], configuration['action_size'])
    else:
        model = ConnectomePolicy(observation_size=configuration['observation_size'],
                                 neural_steps=configuration['neural_steps'])
    model.load(path)
    return model.eval(), configuration, kind


def complete_runs(mask):
    padded = np.concatenate(([False], mask, [False])).astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [(a, b) for a, b in zip(starts, ends) if a > 0 and b < len(mask)]


def gait_summary(series, dt):
    contacts = np.asarray(series['contacts'], dtype=bool)
    clearance = np.asarray(series['clearance'])
    feet = np.asarray(series['feet_xy'])
    per_foot = []
    for side in range(2):
        swings = complete_runs(~contacts[:, side])
        durations = [(b-a)*dt for a, b in swings]
        heights = [float(clearance[a:b, side].max()) for a, b in swings]
        qualifying = [d >= 0.12 and h >= 0.025 for d, h in zip(durations, heights)]
        supported = contacts[1:, side] & contacts[:-1, side]
        speed = np.linalg.norm(np.diff(feet[:, side], axis=0), axis=1)/dt
        per_foot.append(dict(complete_swings=len(swings),
                             swing_seconds_median=float(np.median(durations)) if durations else None,
                             swing_clearance_median=float(np.median(heights)) if heights else None,
                             qualifying_swings=int(sum(qualifying)),
                             qualifying_swings_per_second=float(sum(qualifying)/(len(contacts)*dt)),
                             support_foot_center_speed=float(speed[supported].mean()) if supported.any() else None,
                             contact_fraction=float(contacts[:, side].mean())))
    yaw = np.asarray(series['yaw'])
    recovery = np.abs(yaw) <= 0.05
    window = max(1, round(0.5/dt))
    sustained = np.flatnonzero(np.convolve(recovery.astype(int), np.ones(window, dtype=int), 'valid') == window)
    return dict(feet=per_foot, mean_abs_yaw=float(np.abs(yaw).mean()),
                final_yaw=float(yaw[-1]), recovery_seconds=float(sustained[0]*dt) if len(sustained) else None,
                maximum_lateral_m=float(np.abs(series['y']).max()),
                mean_abs_lateral_m=float(np.abs(series['y']).mean()),
                mean_height=float(np.mean(series['height'])),
                mean_knees=np.mean(series['knees'], axis=0).tolist(),
                note='Qualifying swing screen: >=0.12 s airborne and >=0.025 m minimum-sole clearance; '
                     'support speed uses sole-geometry centers, not exact contact-point slip. No natural-gait claim.')


@torch.inference_mode()
def evaluate_batch(policy, bodies, seeds, speeds, yaws, seconds, zero_new=False, lesion=False, push=False):
    for body, seed, yaw in zip(bodies, seeds, yaws):
        body.reset(seed)
        if yaw:
            body.data.qpos[3:7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
            mujoco.mj_forward(body.model, body.data)
    dt = bodies[0].model.opt.timestep*bodies[0].cfg['control_decimation']
    alive = np.ones(len(bodies), dtype=bool)
    min_upright = np.ones(len(bodies))
    alternations = np.zeros(len(bodies), dtype=int)
    previous_side = [None]*len(bodies)
    previous_contacts = [[False, False] for _ in bodies]
    series = [dict(contacts=[], clearance=[], feet_xy=[], yaw=[], y=[], height=[], knees=[]) for _ in bodies]
    foot_geoms = []
    for body in bodies:
        foot_geoms.append([np.flatnonzero(body.model.geom_bodyid == body.model.body(name).id)
                           for name in ['left_ankle_roll_link', 'right_ankle_roll_link']])
    last = [None]*len(bodies)
    digest = hashlib.sha256()
    started = time.perf_counter()
    for step in range(round(seconds/dt)):
        if push and step == round(10/dt):
            for body in bodies:
                body.data.qvel[1] += 0.25
        obs = np.stack([b.motor_observation([v, 0, float(np.clip(-b.observation()[1]*1.4, -0.2, 0.2))])
                        for b, v in zip(bodies, speeds)])
        if zero_new:
            obs[:, 47:50] = 0
        tensor = torch.from_numpy(obs).cuda()
        actions = (policy(tensor, lesion=True) if lesion else policy(tensor))[0].cpu().numpy()
        for i, body in enumerate(bodies):
            if not alive[i]:
                continue
            state = body.step_joints(actions[i])
            min_upright[i] = min(min_upright[i], state['upright'])
            for side in range(2):
                if state['contacts'][side] and not previous_contacts[i][side]:
                    alternations[i] += int(previous_side[i] is not None and previous_side[i] != side)
                    previous_side[i] = side
            previous_contacts[i] = state['contacts'].copy()
            alive[i] = not state['fallen']
            last[i] = state
            s = series[i]
            s['contacts'].append(state['contacts'])
            s['clearance'].append([float(np.min(body.data.geom_xpos[g, 2]-body.model.geom_size[g, 0]))
                                   for g in foot_geoms[i]])
            s['feet_xy'].append([body.data.geom_xpos[g, :2].mean(axis=0).tolist() for g in foot_geoms[i]])
            s['yaw'].append(state['yaw']); s['y'].append(state['qpos'][1])
            s['height'].append(state['height']); s['knees'].append([state['qpos'][10], state['qpos'][16]])
        digest.update(np.stack([b.data.qpos.copy() for b in bodies]).tobytes())
        digest.update(np.stack([b.data.qvel.copy() for b in bodies]).tobytes())
        if not alive.any():
            break
    tests = []
    for i, (seed, speed, state) in enumerate(zip(seeds, speeds, last)):
        duration = state['time']
        velocity = state['qpos'][0]/duration
        lateral = abs(state['qpos'][1])
        criteria = dict(survived=bool(alive[i] and duration >= seconds-0.01),
                        speed=bool(abs(velocity-speed) <= 0.15),
                        feet=bool(min(state['foot_strikes']) >= int(seconds*0.5)),
                        alternation=bool(alternations[i] >= int(seconds)), upright=bool(min_upright[i] >= 0.7),
                        direction=bool(lateral <= max(1, seconds*speed*0.2)))
        tests.append(dict(seed=seed, initial_yaw=yaws[i], target_speed=speed, seconds=duration,
                          mean_speed=velocity, lateral_m=lateral, fallen=not bool(alive[i]),
                          foot_strikes=state['foot_strikes'], alternations=int(alternations[i]),
                          minimum_upright=float(min_upright[i]), criteria=criteria, success=all(criteria.values()),
                          gait=gait_summary(series[i], dt)))
    return dict(tests=tests, trajectory_sha256=digest.hexdigest(), wall_seconds=time.perf_counter()-started)


def summarize(tests):
    return dict(episodes=len(tests), successes=sum(t['success'] for t in tests), falls=sum(t['fallen'] for t in tests),
                criteria_passes={key: sum(t['criteria'][key] for t in tests) for key in tests[0]['criteria']},
                lateral_m=float(np.mean([t['lateral_m'] for t in tests])),
                maximum_lateral_mean=float(np.mean([t['gait']['maximum_lateral_m'] for t in tests])),
                mean_abs_yaw=float(np.mean([t['gait']['mean_abs_yaw'] for t in tests])),
                mean_height=float(np.mean([t['gait']['mean_height'] for t in tests])),
                speed_mae=float(np.mean([abs(t['mean_speed']-t['target_speed']) for t in tests])),
                qualifying_swings_per_second=float(np.mean([f['qualifying_swings_per_second']
                                                           for t in tests for f in t['gait']['feet']])))


def main(args):
    torch.set_num_threads(8)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in (ROOT/'app/yumi_description').glob('*.xml')}
    report = dict(kind='matched_native_locomotion', batch=9, seconds=args.seconds, seed_base=args.seed_base,
                  sparse_backend=args.sparse_backend,
                  yaws=args.yaws, body_xml_sha256=hashes, conditions=[], completed=False,
                  source_sha256={f: file_sha256(ROOT/'app'/f) for f in
                                 ['sim.py', 'full_brain.py', 'ordered_inference.py', 'evaluate_locomotion.py']},
                  runtime=dict(torch=torch.__version__, mujoco=mujoco.__version__, gpu=torch.cuda.get_device_name()),
                  limitation='Development unless explicitly reserved beforehand. Gait measurements are diagnostics, '
                             'not a validated human-naturalness criterion.')
    for checkpoint in args.checkpoints:
        model, config, kind = load_policy(checkpoint)
        bodies = [Body(False, 'yumi', interface=config['physics_interface'],
                       observation_size=config['observation_size']) for _ in range(9)]
        digest = file_sha256(checkpoint)
        modes = ['normal', 'repeat'] if args.repeat else ['normal']
        if args.zero_new:
            modes += ['zero_new']
        if args.lesion:
            if kind != 'connectome':
                raise ValueError('Disconnect control requires a connectome')
            modes += ['lesion']
        normal_hashes = {}
        with ordered_inference() if kind == 'connectome' and args.sparse_backend == 'ordered' else torch.inference_mode():
            for mode in modes:
                results = []
                for cohort in range(args.cohorts):
                    for yaw in args.yaws:
                        seeds = list(range(args.seed_base+cohort*10, args.seed_base+cohort*10+9))
                        row = evaluate_batch(model, bodies, seeds, [0.35, 0.5, 0.65]*3, [yaw]*9,
                                             args.seconds, mode == 'zero_new', mode == 'lesion', args.push)
                        key = (cohort, yaw)
                        if mode == 'normal':
                            normal_hashes[key] = row['trajectory_sha256']
                        elif mode == 'repeat':
                            assert normal_hashes[key] == row['trajectory_sha256'], 'Non-repeatable evaluation'
                        results.append(dict(cohort=cohort, initial_yaw=yaw, **row))
                tests = [t for r in results for t in r['tests']]
                summary = summarize(tests)
                report['conditions'].append(dict(checkpoint=str(checkpoint), checkpoint_sha256=digest,
                                                  policy_kind=kind, mode=mode, physics_interface=bodies[0].interface,
                                                  results=results, summary=summary))
                atomic_json(output, report)
                print(json.dumps(dict(checkpoint=str(checkpoint), mode=mode, **summary)), flush=True)
        assert file_sha256(checkpoint) == digest
        del model, bodies
        torch.cuda.empty_cache()
    report['completed'] = True
    atomic_json(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--seed-base', type=int, default=93001)
    parser.add_argument('--cohorts', type=int, default=3)
    parser.add_argument('--yaws', nargs='+', type=float, default=[0.0])
    parser.add_argument('--repeat', action='store_true')
    parser.add_argument('--zero-new', action='store_true')
    parser.add_argument('--lesion', action='store_true')
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--sparse-backend', choices=['ordered', 'native'], default='ordered',
                        help='Record the CSR inference path separately from native MuJoCo physics')
    main(parser.parse_args())
