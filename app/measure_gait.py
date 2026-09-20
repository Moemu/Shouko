"""Step-0 gait diagnostics on the native MuJoCo acceptance path, frozen checkpoints only.

Three measurements, no training (proposals of 2026-09-16, Step 0):
  gait      50 Hz full-state record -> cadence, duty factor, hip/knee/ankle
            amplitudes, left-right symmetry, pelvis height.
  sweep     frozen-checkpoint gait-clock sweep (0.5-1.667 s period, cmd 0.5):
            does the locked policy tolerate a clock change, and does speed
            respond monotonically to frequency?
  transfer  (cmd -> vx) curve extending runs/local/heldout_yumi.json's three
            anchor points (cmd 0.35/0.5/0.65, seeds 8001-8003).

Everything runs through sim.Body, the same native path as evaluate_full
acceptance (whose motor_observation hardcodes a 1.0 s clock). The training
environment gpu_body.GPUHumanoid hardcodes 0.8 s instead; both periods are
inside the sweep grid, so no code path is trusted unmeasured. Raw 50 Hz series
go to *_frames.npz (git-ignored); derived metrics go to small JSON files.

Checkpoint/interface pairing is explicit per lineage: neither Yumi checkpoint
embeds physics_interface, and the 2026-09-16 home incident showed what silent
pairing drift does (research/experiments/HOME_REFERENCE_MISMATCH_20260916.md).
"""
import argparse
import hashlib
import json
import subprocess
import time

import mujoco
import numpy as np
import torch
import yaml

from .full_brain import ConnectomePolicy, ROOT
from .sim import INTERFACE_KEYS, Body

LINEAGES = {
    'squat': dict(checkpoint=ROOT / 'runs/yumi/best.pt',
                  interface_yaml=ROOT / 'app/yumi_description/yumi.yaml',
                  expect_sha='a7a4281f'),
    'tall': dict(checkpoint=ROOT / 'runs/yumi/best_tall.pt',
                 interface_yaml=ROOT / 'runs/local/yumi_tall.yaml',
                 expect_sha='666134f2'),
}
GAIT_CMDS = [0.35, 0.5, 0.65]
TRANSFER_CMDS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
SWEEP_PERIODS = [0.5, 0.625, 0.8, 1.0, 1.25, 1.667]  # 2.0 .. 0.6 Hz
DEFAULT_SEEDS = [8001, 8002, 8003]
# sim.Body hardcodes these; gpu_body.GPUHumanoid hardcodes 0.8 (training path).
ACCEPTANCE_CLOCK_S = 1.0
TRAINING_CLOCK_S = 0.8
# Joint order in yumi.xml: [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll] x (left, right).
HIP_PITCH, KNEE, ANKLE_PITCH = (0, 6), (3, 9), (4, 10)


class ClockBody(Body):
    """sim.Body with a configurable gait-clock period.

    motor_observation mirrors sim.py:102-111 (default period 1.0 s = acceptance
    path); only the phase denominator differs. A subclass instead of a sim.py
    edit so the acceptance path itself stays untouched by diagnostics.
    """

    def __init__(self, *args, clock_period=ACCEPTANCE_CLOCK_S, **kwargs):
        super().__init__(*args, **kwargs)
        self.clock_period = clock_period

    def motor_observation(self, command):
        gravity, _ = self.observation()
        phase = self.data.time / self.clock_period * 2 * np.pi
        return np.concatenate([
            self.data.qvel[3:6] * self.cfg["ang_vel_scale"], gravity,
            np.asarray(command) * self.cfg["cmd_scale"],
            self.data.qpos[7:] - self.home,
            self.data.qvel[6:] * self.cfg["dof_vel_scale"], self.action,
            [np.sin(phase), np.cos(phase)],
        ]).astype(np.float32)


def lineage_interface(path):
    """Interface keys with native yaml types.

    sim.observation_interface float-converts everything, which breaks
    Body.step_joints (range() over control_decimation); that combination is
    unreachable today because every existing checkpoint has physics_interface
    null, but this script pairs interfaces by hand so it must not crash.
    """
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    return {key: cfg[key] for key in INTERFACE_KEYS if key in cfg}


def file_sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def interface_digest(interface):
    return hashlib.sha256(json.dumps(interface or {}, sort_keys=True).encode()).hexdigest()


def git_revision():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, timeout=10,
                              check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def dominant_hz(series, dt=0.02, low=0.5, high=8.0):
    x = np.asarray(series, dtype=float)
    if x.size < 50 or float(x.std()) < 1e-6:
        return None
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, dt)
    band = (freqs >= low) & (freqs <= high)
    if not band.any():
        return None
    return float(freqs[band][int(np.argmax(spec[band]))])


@torch.inference_mode()
def run_episode(model, body, seed, speed, seconds, record=False):
    """Same loop contract as evaluate_full.episode, plus optional 50 Hz series."""
    body.reset(seed)
    steps = round(seconds / 0.02)
    series = None
    if record:
        series = dict(t=np.zeros(steps), height=np.zeros(steps), vx=np.zeros(steps),
                      yaw=np.zeros(steps), joint=np.zeros((steps, 12)),
                      footz=np.zeros((steps, 2)), contact=np.zeros((steps, 2), dtype=bool))
    previous_contacts = [False, False]
    previous_side = None
    alternating = 0
    duty = np.zeros(2)
    both = 0
    height_sum = 0.0
    knee_sum = np.zeros(2)
    minimum_height, minimum_upright = 10.0, 1.0
    neural_ms = []
    started = time.perf_counter()
    for step in range(steps):
        _, yaw = body.observation()
        yaw_rate = float(np.clip(-yaw * 1.4, -0.2, 0.2))
        observation = body.motor_observation([speed, 0, yaw_rate])
        tick = time.perf_counter()
        action, _ = model(torch.as_tensor(observation, device=model.weight.device)[None])
        neural_ms.append((time.perf_counter() - tick) * 1000)
        state = body.step_joints(action[0].cpu().numpy())
        duty += state['contacts']
        both += int(all(state['contacts']))
        height_sum += state['height'] * 0.02
        knee_sum += [state['qpos'][7 + KNEE[0]], state['qpos'][7 + KNEE[1]]]
        minimum_height = min(minimum_height, state['height'])
        minimum_upright = min(minimum_upright, state['upright'])
        for side in range(2):
            if state['contacts'][side] and not previous_contacts[side]:
                alternating += int(previous_side is not None and previous_side != side)
                previous_side = side
        previous_contacts = state['contacts']
        if record:
            series['t'][step] = state['time']
            series['height'][step] = state['height']
            series['vx'][step] = state['velocity'][0]
            series['yaw'][step] = state['yaw']
            series['joint'][step] = state['qpos'][7:]
            series['footz'][step] = [state['joints'][body.foot_ids[0] - 1][2],
                                     state['joints'][body.foot_ids[1] - 1][2]]
            series['contact'][step] = state['contacts']
        if state['fallen']:
            break
    duration = state['time']
    steps_done = step + 1
    joints = np.asarray(state['qpos'][7:])
    summary = dict(seed=seed, target_speed=speed, clock_period_s=float(body.clock_period),
                   seconds=duration, fallen=state['fallen'],
                   mean_speed=state['qpos'][0] / max(duration, 1e-9), forward_m=state['qpos'][0],
                   lateral_m=abs(state['qpos'][1]),
                   foot_strikes=state['foot_strikes'],
                   strike_rate_hz=[round(n / duration, 3) for n in state['foot_strikes']] if duration > 0 else None,
                   alternations=alternating,
                   duty_factor=[round(float(d / steps_done), 4) for d in duty],
                   double_support_fraction=round(both / steps_done, 4),
                   mean_height=round(height_sum / max(duration, 1e-9), 4),
                   minimum_height=round(minimum_height, 4), minimum_upright=round(minimum_upright, 4),
                   final_knee_rad=[round(float(joints[KNEE[0]]), 4), round(float(joints[KNEE[1]]), 4)],
                   mean_knee_rad=[round(float(k / steps_done), 4) for k in knee_sum],
                   inference_ms_median=round(float(np.median(neural_ms)), 2),
                   wall_seconds=round(time.perf_counter() - started, 2))
    return summary, series


def gait_metrics(summary, series):
    """Derived gait metrics from one recorded episode (series arrays may be shorter than planned)."""
    n = series['t'].size
    joints = series['joint']
    per_leg = []
    for side, (hip, knee, ankle) in enumerate(zip(HIP_PITCH, KNEE, ANKLE_PITCH)):
        per_leg.append(dict(
            hip_pitch_mean_rad=round(float(joints[:n, hip].mean()), 4),
            hip_pitch_amplitude_rad=round(float(joints[:n, hip].max() - joints[:n, hip].min()), 4),
            knee_mean_rad=round(float(joints[:n, knee].mean()), 4),
            knee_range_rad=round(float(joints[:n, knee].max() - joints[:n, knee].min()), 4),
            ankle_pitch_mean_rad=round(float(joints[:n, ankle].mean()), 4),
            cadence_hz_knee=dominant_hz(joints[:n, knee]),
            cadence_hz_contact=dominant_hz(series['contact'][:n, side].astype(float)),
            foot_height_range_m=round(float(series['footz'][:n, side].max() - series['footz'][:n, side].min()), 4)))
    strikes = summary['foot_strikes']
    return dict(
        per_leg=per_leg,
        cadence_hz_knee_mean=None if None in (per_leg[0]['cadence_hz_knee'], per_leg[1]['cadence_hz_knee'])
        else round((per_leg[0]['cadence_hz_knee'] + per_leg[1]['cadence_hz_knee']) / 2, 3),
        strike_length_m=[round(summary['forward_m'] / n, 4) if n else None for n in strikes],
        pelvis_height_std_m=round(float(series['height'].std()), 4),
        strike_symmetry_l_over_r=round(strikes[0] / strikes[1], 3) if strikes[1] else None,
        yaw_total_deg=round(float(np.degrees(series['yaw'][-1] - series['yaw'][0])), 2))


def lineage_provenance(name):
    spec = LINEAGES[name]
    interface = lineage_interface(spec['interface_yaml'])
    return dict(lineage=name, checkpoint=str(spec['checkpoint'].relative_to(ROOT)),
                checkpoint_sha256=file_sha256(spec['checkpoint']),
                interface_yaml=str(spec['interface_yaml'].relative_to(ROOT)),
                interface_yaml_sha256=file_sha256(spec['interface_yaml']),
                physics_interface=interface, physics_interface_sha256=interface_digest(interface),
                expect_checkpoint_sha_prefix=spec['expect_sha'],
                native_clock_s=ACCEPTANCE_CLOCK_S, training_env_clock_s=TRAINING_CLOCK_S,
                git_revision=git_revision(), mujoco=mujoco.__version__, torch=torch.__version__)


def save_json(path, value):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2))
    temporary.replace(target)


def main(args):
    torch.set_num_threads(4)
    seeds = args.seeds
    out = ROOT / args.out_dir
    lineages = list(LINEAGES) if args.lineage == 'all' else [args.lineage]
    modes = ['gait', 'sweep', 'transfer'] if args.mode == 'all' else [args.mode]
    for name in lineages:
        spec = LINEAGES[name]
        actual = file_sha256(spec['checkpoint'])
        if not actual.startswith(spec['expect_sha']):
            raise SystemExit(f'{spec["checkpoint"]} sha256 {actual[:8]} does not match expected '
                             f'{spec["expect_sha"]} — checkpoint lineage changed, refusing to mislabel measurements')
        provenance = lineage_provenance(name)
        model = ConnectomePolicy(device=args.device).eval()
        model.load(spec['checkpoint'])
        body = ClockBody(load_motor_policy=False, robot='yumi',
                         interface=provenance['physics_interface'])
        body.foot_ids = [body.model.body(n).id for n in ['left_ankle_roll_link', 'right_ankle_roll_link']]
        print(json.dumps(dict(event='lineage_ready', **{k: provenance[k] for k in
              ('lineage', 'checkpoint', 'checkpoint_sha256', 'interface_yaml', 'physics_interface_sha256')})), flush=True)

        if 'gait' in modes:
            episodes, frames = [], []
            for cmd in GAIT_CMDS:
                for seed in seeds:
                    body.clock_period = ACCEPTANCE_CLOCK_S
                    summary, series = run_episode(model, body, seed, cmd, args.seconds, record=True)
                    episodes.append({**summary, **gait_metrics(summary, series)})
                    frames.append(series)
                    print(json.dumps(dict(event='gait', lineage=name, **summary)), flush=True)
            save_json(out / f'gait_{name}.json', dict(kind='gait_record_50hz', seconds=args.seconds,
                                                      seeds=seeds, cmds=GAIT_CMDS, provenance=provenance,
                                                      episodes=episodes))
            np.savez_compressed(out / f'gait_{name}_frames.npz',
                                **{f'{i}_{key}': value for i, series in enumerate(frames)
                                   for key, value in series.items()})

        if 'sweep' in modes:
            rows = []
            for period in SWEEP_PERIODS:
                body.clock_period = period
                per_seed = []
                for seed in seeds:
                    summary, _ = run_episode(model, body, seed, args.sweep_cmd, args.seconds)
                    per_seed.append(summary)
                    print(json.dumps(dict(event='sweep', lineage=name, **summary)), flush=True)
                rows.append(dict(clock_period_s=period, frequency_hz=round(1 / period, 3),
                                 survivors=sum(not s['fallen'] for s in per_seed), attempts=len(per_seed),
                                 mean_speed_survivors=round(float(np.mean([s['mean_speed'] for s in per_seed
                                                                           if not s['fallen']])), 4) if any(not s['fallen'] for s in per_seed) else None,
                                 mean_height_survivors=round(float(np.mean([s['mean_height'] for s in per_seed
                                                                           if not s['fallen']])), 4) if any(not s['fallen'] for s in per_seed) else None,
                                 episodes=per_seed))
            save_json(out / f'sweep_{name}.json', dict(kind='clock_sweep', sweep_cmd=args.sweep_cmd,
                                                       seconds=args.seconds, seeds=seeds,
                                                       provenance=provenance, rows=rows))

        if 'transfer' in modes:
            body.clock_period = ACCEPTANCE_CLOCK_S
            anchors = {}
            heldout_path = ROOT / 'runs/local/heldout_yumi.json'
            if name == 'squat' and heldout_path.exists():
                for test in json.loads(heldout_path.read_text())['tests']:
                    anchors[(test['seed'], round(test['target_speed'], 2))] = test['mean_speed']
            rows = []
            for cmd in TRANSFER_CMDS:
                per_seed = []
                for seed in seeds:
                    summary, _ = run_episode(model, body, seed, cmd, args.seconds)
                    per_seed.append(summary)
                    print(json.dumps(dict(event='transfer', lineage=name, **summary)), flush=True)
                anchor = [round(summary['mean_speed'] - anchors[(seed, round(cmd, 2))], 4)
                          for seed in seeds if (seed, round(cmd, 2)) in anchors] or None
                rows.append(dict(target_speed=cmd,
                                 mean_speed=round(float(np.mean([s['mean_speed'] for s in per_seed])), 4),
                                 speeds=[s['mean_speed'] for s in per_seed],
                                 survivors=sum(not s['fallen'] for s in per_seed),
                                 mean_height=round(float(np.mean([s['mean_height'] for s in per_seed])), 4),
                                 heldout_anchor_delta=anchor, episodes=per_seed))
            save_json(out / f'transfer_{name}.json', dict(kind='cmd_to_vx_transfer', seconds=args.seconds,
                                                          seeds=seeds, cmds=TRANSFER_CMDS,
                                                          provenance=provenance, rows=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--mode', choices=['gait', 'sweep', 'transfer', 'all'], default='all')
    parser.add_argument('--lineage', choices=['squat', 'tall', 'all'], default='all')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--sweep-cmd', type=float, default=0.5)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--out-dir', default='runs/local/gait_measure_20260918')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    main(parser.parse_args())
