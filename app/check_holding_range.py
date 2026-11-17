"""Frozen nine-state range check for paired initial/final holding actors."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .diagnose_goal_d0 import write_json
from .evaluate_goal import file_sha256
from .evaluate_holding_goal import evaluate
from .free_body import ROOT
from .verify_holding_goal import verify_evaluation


SOURCES = ('app/check_holding_range.py', 'app/evaluate_holding_goal.py', 'app/verify_holding_goal.py',
           'app/braking_task.py', 'app/goal_checkpoint.py', 'app/local_goal.py', 'app/free_body.py',
           'app/gpu_goal.py', 'app/goal_task.py', 'app/mlp_policy.py', 'app/diagnose_goal_d0.py')


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def run(source: Path, output: Path, wall_deadline: float) -> dict:
    torch.set_num_threads(1)
    started = time.time()
    deadline = min(started+1800., wall_deadline)
    evidence = read(source/'learning/summary.json')
    verification = read(source/'learning/verification.json')
    if not verification['verified'] or verification['summary_sha256'] != file_sha256(source/'learning/summary.json'):
        raise ValueError('Holding comparison evidence mismatch')
    forks = read(source/'diagnostic/forks/manifest.json')['forks']
    if len(forks) != 9 or evidence['passed_seeds']['baseline'] != 3:
        raise ValueError('Expected frozen nine-state, three-baseline evidence')
    for fork in forks:
        if file_sha256(source/'diagnostic/forks'/fork['filename']) != fork['sha256']:
            raise ValueError('Frozen fork changed')
    cases = []
    for seed in (1501, 1502, 1503):
        folder = source/'learning'/f'baseline_{seed}'
        training = read(folder/'summary.json')
        for phase, filename, key in (('initial', 'initial.pt', 'initial_checkpoint_sha256'),
                                     ('final', 'last.pt', 'checkpoint_sha256')):
            checkpoint = folder/filename
            if file_sha256(checkpoint) != training[key]:
                raise ValueError('Holding actor changed')
            for fork in forks:
                cases.append(dict(seed=seed, phase=phase, checkpoint=filename, checkpoint_sha256=training[key],
                                  fork_id=fork['id'], fork_filename=fork['filename'], fork_sha256=fork['sha256'],
                                  reuse=fork['id'] == '1402_145'))
    output.mkdir(parents=True, exist_ok=False)
    prereg = dict(schema='holding_range_check_v1', started_unix=started, deadline_unix=deadline,
                  max_new_native=48, max_new_cuda=48, max_new_control_steps=14400, weight_updates=0,
                  source_state_sha256=file_sha256(source/'learning/reset.npz'), cases=cases,
                  learning_summary_sha256=file_sha256(source/'learning/summary.json'),
                  learning_verification_sha256=file_sha256(source/'learning/verification.json'),
                  source_sha256={n: file_sha256(ROOT/n) for n in SOURCES})
    for name in SOURCES:
        target = output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    write_json(output/'preregistration.json', prereg)
    rows, checks = [], []
    branch = 'coverage_checked'
    new_counts = dict(native=0, cuda=0)
    new_steps = 0
    for case in cases:
        checkpoint = source/'learning'/f"baseline_{case['seed']}"/case['checkpoint']
        state = source/'diagnostic/forks'/case['fork_filename']
        row = dict(**case, evaluations={})
        rows.append(row)
        for backend in ('native', 'cuda'):
            if backend == 'cuda' and not row['evaluations']['native']['passed']:
                break
            if time.time() >= deadline:
                branch = 'budget_stop'
                break
            if case['reuse']:
                folder = source/'learning'/f"eval_baseline_{case['seed']}_{case['phase']}_{backend}"
                result = read(folder/'summary.json')
                audit = verify_evaluation(folder, checkpoint, source/'learning/reset.npz')
                if file_sha256(state) != prereg['source_state_sha256']:
                    raise ValueError('Reused state differs from the original holding course')
            else:
                folder = output/f"eval_{case['phase']}_{case['seed']}_{case['fork_id']}_{backend}"
                result = evaluate(checkpoint, source/'learning/reset.npz', folder, backend, transfer_state=state)
                audit = verify_evaluation(folder, checkpoint, state, source_state=source/'learning/reset.npz')
                with np.load(folder/'trace.npz') as trace:
                    new_steps += len(trace['work_j'])
                new_counts[backend] += 1
            assert result['checkpoint_sha256'] == case['checkpoint_sha256']
            assert result['state_sha256'] == case['fork_sha256']
            row['evaluations'][backend] = result
            checks.append(dict(seed=case['seed'], phase=case['phase'], fork_id=case['fork_id'],
                               backend=backend, reused=case['reuse'], **audit))
            write_json(output/'progress.json', dict(runs=rows, new_evaluations=new_counts, new_control_steps=new_steps))
            if not result['safe'] or (backend == 'cuda' and not result['passed']):
                branch = 'route_uncertainty_stop'
                break
        if branch != 'coverage_checked':
            break
        print(json.dumps(dict(seed=case['seed'], phase=case['phase'], fork_id=case['fork_id'],
                              native_passed=row['evaluations']['native']['passed'], reused=case['reuse'])), flush=True)
    if branch == 'coverage_checked':
        assert len(rows) == 54 and new_counts['native'] == 48
    assert new_counts['cuda'] <= 48 and new_steps <= 14400
    success = {p: [dict(seed=r['seed'], fork_id=r['fork_id']) for r in rows
                    if r['phase'] == p and r['evaluations'].get('cuda', {}).get('passed')]
               for p in ('initial', 'final')}
    result = dict(schema='holding_range_result_v1', branch=branch, runs=rows, confirmed_successes=success,
                  new_evaluations=new_counts, reused_evaluations=sum(c['reused'] for c in checks),
                  new_control_steps=new_steps, weight_updates=0, elapsed_seconds=time.time()-started,
                  preregistration_sha256=file_sha256(output/'preregistration.json'))
    write_json(output/'summary.json', result)
    # Recount from independent per-trajectory audits, not the producer's success list.
    independently_counted = {p: sorted((c['seed'], c['fork_id']) for c in checks
                                      if c['phase'] == p and c['backend'] == 'cuda' and c['passed'])
                              for p in ('initial', 'final')}
    assert independently_counted == {p: sorted((x['seed'], x['fork_id']) for x in success[p]) for p in success}
    assert new_counts == {b: sum(not c['reused'] and c['backend'] == b for c in checks) for b in new_counts}
    write_json(output/'verification.json', dict(verified=True, checks=checks, branch=branch,
                                              summary_sha256=file_sha256(output/'summary.json'),
                                              verifier_source_sha256=file_sha256(ROOT/'app/verify_holding_goal.py')))
    print(json.dumps(dict(branch=branch, new_evaluations=new_counts, confirmed_successes=success)), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.wall_deadline)


if __name__ == '__main__':
    main()
