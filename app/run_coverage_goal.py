"""Frozen three-source paired comparison of near-only and mixed reset coverage."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .coverage_goal import load_pool
from .diagnose_goal_d0 import write_json
from .evaluate_coverage_goal import evaluate
from .evaluate_goal import file_sha256
from .free_body import ROOT
from .ppo_goal import PPOSettings
from .verify_holding_goal import verify_evaluation


SOURCES = ('app/run_coverage_goal.py', 'app/ppo_coverage_goal.py', 'app/coverage_goal.py', 'app/ppo_goal.py',
           'app/evaluate_coverage_goal.py', 'app/evaluate_holding_goal.py', 'app/verify_holding_goal.py',
           'app/verify_local_goal.py', 'app/braking_task.py', 'app/local_goal.py', 'app/goal_checkpoint.py',
           'app/free_body.py', 'app/gpu_goal.py', 'app/goal_task.py', 'app/mlp_policy.py', 'app/ppo_yumi.py',
           'app/recovery_curriculum.py', 'app/value_diagnostics.py', 'app/diagnose_goal_d0.py', 'app/training_control.py')


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def run(screen: Path, pool: Path, range_check: Path, contracts: Path, output: Path, wall_deadline: float) -> dict:
    checked = read(range_check/'summary.json')
    verified = read(range_check/'verification.json')
    if (checked['branch'] != 'coverage_checked' or not verified['verified']
            or verified['summary_sha256'] != file_sha256(range_check/'summary.json')):
        raise ValueError('Coverage learning requires a complete verified range check')
    contract_checks = read(contracts/'checks.json')
    if not contract_checks['passed'] or contract_checks['engineering_attempted_transitions'] > 2048:
        raise ValueError('Engineering contracts or sample budget failed')
    states = load_pool(pool)
    pool_manifest = read(pool/'manifest.json')
    if pool_manifest['original_source_reset_sha256'] != file_sha256(screen/'reset.npz'):
        raise ValueError('Coverage pool original reset mismatch')
    range_prereg = read(range_check/'preregistration.json')
    for state in pool_manifest['states'][1:]:
        expected = next(c['fork_sha256'] for c in range_prereg['cases'] if c['fork_id'] == state['id'])
        if state['sha256'] != expected:
            raise ValueError('Coverage pool differs from checked range states')
    output.mkdir(parents=True, exist_ok=False)
    order = [(s, 1601+s-1401, arm) for s in (1401, 1402, 1403)
             for arm in (('mixed', 'near') if s == 1402 else ('near', 'mixed'))]
    source_hashes = {}
    for seed in (1401, 1402, 1403):
        checkpoint = screen/f'iid_zero_{seed}/last.pt'
        source_hashes[str(seed)] = file_sha256(checkpoint)
        if source_hashes[str(seed)] != read(checkpoint.parent/'summary.json')['checkpoint_sha256']:
            raise ValueError('Approach actor differs from training evidence')
    prereg = dict(schema='paired_coverage_learning_v1', order=[dict(source_seed=s, sampling_seed=n, arm=a) for s, n, a in order],
                 settings=asdict(PPOSettings(seed=1601, initial_std=.15)), iterations=256, training_work_cost=0.,
                 pool_sha256=file_sha256(pool/'manifest.json'), pool_manifest=pool_manifest, source_actors=source_hashes,
                 source_reset_sha256=file_sha256(screen/'reset.npz'), max_training_process_seconds=2700.,
                 per_run_training_seconds=420., formal_transition_limit=1572864, overall_wall_deadline_unix=wall_deadline,
                 range_summary_sha256=file_sha256(range_check/'summary.json'),
                 range_verification_sha256=file_sha256(range_check/'verification.json'),
                 contracts_sha256=file_sha256(contracts/'checks.json'),
                 source_sha256={n: file_sha256(ROOT/n) for n in SOURCES})
    for name in SOURCES:
        dest = output/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    write_json(output/'preregistration.json', prereg)
    rows, audits, training_seconds = [], [], 0.
    branch = 'running'
    for source_seed, sampling_seed, arm in order:
        if prereg['source_sha256'] != {n: file_sha256(ROOT/n) for n in SOURCES}:
            raise RuntimeError('Source changed during frozen coverage comparison')
        if time.time() >= wall_deadline or training_seconds >= 2700.:
            branch = 'budget_stop'
            break
        budget = min(420., 2700.-training_seconds-10., wall_deadline-time.time()-30.)
        if budget <= 0:
            branch = 'budget_stop'
            break
        folder = output/f'{arm}_{source_seed}'
        row = dict(source_seed=source_seed, sampling_seed=sampling_seed, arm=arm, complete=False, training_complete=False)
        rows.append(row)
        command = [sys.executable, '-m', 'app.ppo_coverage_goal', '--pool', str(pool), '--source-checkpoint',
                   str(screen/f'iid_zero_{source_seed}/last.pt'), '--source-state', str(screen/'reset.npz'),
                   '--arm', arm, '--seed', str(sampling_seed), '--output', str(folder), '--max-seconds', str(budget)]
        started = time.perf_counter()
        timed_out = False
        with (output/f'{arm}_{source_seed}.log').open('w', encoding='utf-8') as console:
            try:
                process = subprocess.run(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                                         timeout=max(1., min(budget+60., wall_deadline-time.time())))
            except subprocess.TimeoutExpired:
                timed_out = True
        duration = time.perf_counter()-started
        training_seconds += duration
        status = read(folder/'training.json')
        row.update(training_process_seconds=duration, actual_transitions=status.get('env_steps', 0),
                   attempted_transitions=status.get('attempted_local_transitions', 0))
        if timed_out:
            row['error'] = 'Training process exceeded wall budget'
            branch = 'budget_stop'
            break
        if process.returncode:
            row['error'] = status.get('error', 'Training initialization/process failed')
            branch = 'route_uncertainty_stop'
            break
        train = read(folder/'summary.json')
        row['checkpoint_sha256'] = train['checkpoint_sha256']
        if train['launch_transitions'] != 262144:
            branch = 'budget_stop'
            break
        row['training_complete'] = True
        for phase, filename in (('initial', 'initial.pt'), ('final', 'last.pt')):
            row[phase] = {}
            for start in ('original', 'near'):
                row[phase][start] = {}
                for backend in ('native', 'cuda'):
                    dest = output/f'eval_{arm}_{source_seed}_{phase}_{start}_{backend}'
                    result = evaluate(folder/filename, pool, dest, backend, start)
                    row[phase][start][backend] = result
                    state = states[0 if start == 'original' else 7].path
                    audits.append(verify_evaluation(dest, folder/filename, state, source_state=pool/'manifest.json'))
                    if not result['safe']:
                        branch = 'route_uncertainty_stop'
                        break
                if branch != 'running':
                    break
                if row[phase][start]['native']['passed'] != row[phase][start]['cuda']['passed']:
                    branch = 'route_uncertainty_stop'
                    break
            if branch != 'running':
                break
        if branch != 'running':
            break
        row['complete'] = True
        row['passed'] = all(r['passed'] for b in row['final'].values() for r in b.values())
        write_json(output/'progress.json', dict(runs=rows, training_process_seconds=training_seconds))
        print(json.dumps(dict(source_seed=source_seed, arm=arm, joint_passed=row['passed'],
                              final={s: {b: r['passed'] for b, r in v.items()} for s, v in row['final'].items()})), flush=True)
    counts = {a: sum(bool(r.get('passed')) for r in rows if r['arm'] == a) for a in ('near', 'mixed')}
    if branch == 'running':
        assert len(rows) == 6 and all(r['complete'] for r in rows)
        branch = ('near_preferred' if counts['near'] >= 2 else 'mixed_coverage_positive' if counts['mixed'] >= 2
                  else 'route_uncertainty_stop')
    result = dict(branch=branch, runs=rows, passed_sources=counts, training_process_seconds=training_seconds,
                  actual_transitions=sum(r['actual_transitions'] for r in rows),
                  attempted_transitions=sum(r['attempted_transitions'] for r in rows),
                  complete_comparison=len(rows) == 6 and all(r['complete'] for r in rows),
                  preregistration_sha256=file_sha256(output/'preregistration.json'))
    write_json(output/'summary.json', result)
    write_json(output/'trajectory_audits.json', dict(checks=audits, summary_sha256=file_sha256(output/'summary.json')))
    print(json.dumps(dict(branch=branch, passed_sources=counts, actual_transitions=result['actual_transitions'],
                          attempted_transitions=result['attempted_transitions'])), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-screen', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--range-check', type=Path, required=True)
    parser.add_argument('--contracts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.source_screen, args.pool, args.range_check, args.contracts, args.output, args.wall_deadline)


if __name__ == '__main__':
    main()
