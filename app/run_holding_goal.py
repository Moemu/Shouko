"""Frozen paired-seed holding experiment, conditional on verified braking evidence."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .diagnose_goal_d0 import write_json
from .evaluate_goal import file_sha256
from .evaluate_holding_goal import evaluate
from .free_body import ROOT
from .ppo_goal import PPOSettings


SOURCES = ('app/run_holding_goal.py', 'app/ppo_holding_goal.py', 'app/ppo_goal.py',
           'app/evaluate_holding_goal.py', 'app/braking_task.py', 'app/local_goal.py',
           'app/free_body.py', 'app/gpu_goal.py', 'app/goal_task.py', 'app/goal_checkpoint.py',
           'app/mlp_policy.py', 'app/ppo_yumi.py', 'app/recovery_curriculum.py',
           'app/value_diagnostics.py', 'app/diagnose_goal_d0.py', 'app/training_control.py')


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def run(diagnostic: Path, output: Path, wall_deadline: float) -> dict:
    evidence, verified = read(diagnostic/'summary.json'), read(diagnostic/'verification.json')
    if (evidence['branch'] != 'holding_control_positive' or not verified['verified']
            or verified['summary_sha256'] != file_sha256(diagnostic/'summary.json')
            or evidence['positive_forks'] != ['1402_145']):
        raise ValueError('This preregistered single-state course requires verified fork 1402_145')
    fork = diagnostic/'forks/fork_1402_145.npz'
    recorded_fork = next(r for r in read(diagnostic/'forks/manifest.json')['forks'] if r['id'] == '1402_145')
    if (file_sha256(fork) != recorded_fork['sha256']
            or any(r['fork_sha256'] != recorded_fork['sha256'] for r in evidence['probes'] if r['fork_id'] == '1402_145')):
        raise ValueError('Verified braking fork changed before learning')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(fork, output/'reset.npz')
    order = [(seed, group, scale) for seed in (1501, 1502, 1503)
             for group, scale in ((('baseline', 0.), ('potential', .5)) if seed != 1502
                                  else (('potential', .5), ('baseline', 0.)))]
    prereg = dict(schema='paired_holding_v1', source_forks=['1402_145'], independent_sources=1,
                 state_sha256=file_sha256(output/'reset.npz'),
                 diagnostic_summary_sha256=file_sha256(diagnostic/'summary.json'),
                 diagnostic_verification_sha256=file_sha256(diagnostic/'verification.json'),
                 order=[dict(seed=s, group=g, scale=p) for s, g, p in order],
                 settings=asdict(PPOSettings(seed=1501, initial_std=.15)), iterations=256,
                 per_run_training_seconds=420., total_training_process_seconds=2700.,
                 overall_wall_deadline_unix=wall_deadline, formal_transition_limit=1572864,
                 minimum_passing_seeds=2, evaluation_backends=['native', 'cuda'],
                 initial_evaluation='same frozen reset, deterministic initial checkpoint; no seed selection',
                 source_sha256={n: file_sha256(ROOT/n) for n in SOURCES})
    for name in SOURCES:
        dest = output/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    write_json(output/'preregistration.json', prereg)
    rows, training_seconds = [], 0.
    branch = 'running'
    for seed, group, scale in order:
        if prereg['source_sha256'] != {n: file_sha256(ROOT/n) for n in SOURCES}:
            raise RuntimeError('Source changed during the frozen holding comparison')
        if time.time() >= wall_deadline or training_seconds >= 2700.:
            branch = 'budget_stop'
            break
        folder = output/f'{group}_{seed}'
        started = time.perf_counter()
        budget = min(420., 2700.-training_seconds, wall_deadline-time.time()-30.)
        if budget <= 0:
            branch = 'budget_stop'
            break
        with (output/f'{group}_{seed}.log').open('w', encoding='utf-8') as console:
            process = subprocess.run([sys.executable, '-m', 'app.ppo_holding_goal', '--state', str(output/'reset.npz'),
                                      '--output', str(folder), '--seed', str(seed), '--potential-scale', str(scale),
                                      '--max-seconds', str(budget)], cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                                     timeout=max(1., min(budget+60., wall_deadline-time.time())))
        duration = time.perf_counter()-started
        training_seconds += duration
        row = dict(seed=seed, group=group, scale=scale, training_process_seconds=duration,
                   complete=False, actual_transitions=read(folder/'training.json').get('env_steps', 0))
        rows.append(row)
        if process.returncode:
            branch = 'route_uncertainty_stop'
            row['error'] = read(folder/'training.json').get('error', 'Training process failed')
            break
        report = read(folder/'summary.json')
        row.update(actual_transitions=report['launch_transitions'], checkpoint_sha256=report['checkpoint_sha256'])
        if report['launch_transitions'] != 262144:
            branch = 'budget_stop'
            break
        row['complete'] = True
        for phase, checkpoint in (('initial', 'initial.pt'), ('final', 'last.pt')):
            row[phase] = {b: evaluate(folder/checkpoint, output/'reset.npz',
                                      output/f'eval_{group}_{seed}_{phase}_{b}', b) for b in ('native', 'cuda')}
        row['passed'] = all(x['passed'] for x in row['final'].values())
        write_json(output/'progress.json', dict(runs=rows, training_process_seconds=training_seconds))
        print(json.dumps(dict(group=group, seed=seed, passed=row['passed'],
                              initial_passed={b: v['passed'] for b, v in row['initial'].items()})), flush=True)
        if any(not x['safe'] for phase in ('initial', 'final') for x in row[phase].values()):
            branch = 'route_uncertainty_stop'
            break
    counts = {g: sum(bool(r.get('passed')) for r in rows if r['group'] == g) for g in ('baseline', 'potential')}
    if len(rows) == 6 and all(r['complete'] for r in rows) and branch == 'running':
        disagreement = any(r['final']['native']['passed'] != r['final']['cuda']['passed'] for r in rows)
        branch = ('route_uncertainty_stop' if disagreement else
                  'baseline_preferred' if counts['baseline'] >= 2 else
                  'potential_course_positive' if counts['potential'] >= 2 else 'route_uncertainty_stop')
    result = dict(branch=branch, runs=rows, passed_seeds=counts, training_process_seconds=training_seconds,
                  actual_transitions=sum(r['actual_transitions'] for r in rows), independent_sources=1,
                  state_sha256=prereg['state_sha256'], preregistration_sha256=file_sha256(output/'preregistration.json'),
                  limitation='One selected near-field state; no evidence of full-task or source-state generalization.')
    write_json(output/'summary.json', result)
    print(json.dumps(dict(branch=branch, passed_seeds=counts, actual_transitions=result['actual_transitions'])), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.diagnostic, args.output, args.wall_deadline)


if __name__ == '__main__':
    main()
