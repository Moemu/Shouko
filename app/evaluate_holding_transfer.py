"""Read-only return from verified near-field training to the original fallen task."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .braking_task import FiniteLocalGoalTask, FiniteTaskContract, save_fork
from .diagnose_goal_d0 import write_json
from .evaluate_goal import file_sha256
from .evaluate_holding_goal import evaluate
from .free_body import FreeBody, ROOT
from .verify_holding_goal import verify_evaluation


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def run(learning: Path, original_reset: Path, output: Path, wall_deadline: float) -> dict:
    trained, verification = read(learning/'summary.json'), read(learning/'verification.json')
    if (trained['branch'] not in ('baseline_preferred', 'potential_course_positive') or not verification['verified']
            or verification['summary_sha256'] != file_sha256(learning/'summary.json')):
        raise ValueError('Transfer requires a positive, independently verified holding comparison')
    started = time.time()
    deadline = min(started+600., wall_deadline)
    output.mkdir(parents=True, exist_ok=False)
    save_fork(FiniteLocalGoalTask(FreeBody(), original_reset), output/'reset.npz')
    with np.load(original_reset) as original, np.load(output/'reset.npz') as target:
        for name in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'previous_action'):
            np.testing.assert_array_equal(target[name], original[name])
        assert float(target['elapsed']) == 0. and float(target['dwell']) == 0.
        np.testing.assert_allclose(target['target']-target['origin'], [2., 0.], atol=1e-12, rtol=0)
    cases = [dict(group='initial', seed=s, folder=f'baseline_{s}', checkpoint='initial.pt') for s in (1501, 1502, 1503)]
    cases += [dict(group=r['group'], seed=r['seed'], folder=f"{r['group']}_{r['seed']}", checkpoint='last.pt')
              for r in trained['runs']]
    for case in cases:
        case['checkpoint_sha256'] = file_sha256(learning/case['folder']/case['checkpoint'])
        training = read(learning/case['folder']/'summary.json')
        expected = training['initial_checkpoint_sha256'] if case['checkpoint'] == 'initial.pt' else training['checkpoint_sha256']
        if case['checkpoint_sha256'] != expected:
            raise ValueError('Verified holding weights changed before transfer')
    sources = ('app/evaluate_holding_transfer.py', 'app/evaluate_holding_goal.py', 'app/braking_task.py',
               'app/local_goal.py', 'app/free_body.py', 'app/gpu_goal.py', 'app/goal_task.py',
               'app/goal_checkpoint.py', 'app/mlp_policy.py', 'app/diagnose_goal_d0.py',
               'app/verify_holding_goal.py', 'app/verify_local_goal.py')
    prereg = dict(schema='holding_to_fallen_transfer_v1', cases=cases, max_seconds=600.,
                 actual_deadline_unix=deadline, source_state_sha256=file_sha256(learning/'reset.npz'),
                 original_reset_sha256=file_sha256(original_reset), target_state_sha256=file_sha256(output/'reset.npz'),
                 target_contract=asdict(FiniteTaskContract()), updates=0, max_transitions=4500,
                 learning_summary_sha256=file_sha256(learning/'summary.json'),
                 learning_verification_sha256=file_sha256(learning/'verification.json'),
                 source_sha256={n: file_sha256(ROOT/n) for n in sources})
    for name in sources:
        dest = output/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    write_json(output/'preregistration.json', prereg)
    rows, audits = [], []
    branch = 'running'
    transitions = 0
    for case in cases:
        checkpoint = learning/case['folder']/case['checkpoint']
        row = dict(**case, backends={})
        for backend in ('native', 'cuda'):
            if time.time() >= deadline:
                branch = 'budget_stop'
                break
            dest = output/f"eval_{case['group']}_{case['seed']}_{backend}"
            result = evaluate(checkpoint, learning/'reset.npz', dest, backend, transfer_state=output/'reset.npz')
            audits.append(verify_evaluation(dest, checkpoint, output/'reset.npz', source_state=learning/'reset.npz'))
            with np.load(dest/'trace.npz') as z:
                transitions += len(z['work_j'])
            assert file_sha256(checkpoint) == case['checkpoint_sha256']
            row['backends'][backend] = result
            if not result['safe']:
                branch = 'route_uncertainty_stop'
                break
        rows.append(row)
        write_json(output/'progress.json', dict(runs=rows, transitions=transitions))
        if branch != 'running':
            break
        row['passed'] = all(x['passed'] for x in row['backends'].values())
        print(json.dumps(dict(group=case['group'], seed=case['seed'], passed=row['passed'])), flush=True)
        if (any(not x['safe'] for x in row['backends'].values())
                or row['backends']['native']['passed'] != row['backends']['cuda']['passed']):
            branch = 'route_uncertainty_stop'
            break
    counts = {g: sum(bool(r.get('passed')) for r in rows if r['group'] == g) for g in ('initial', 'baseline', 'potential')}
    evaluated = {g: sum(len(r['backends']) == 2 for r in rows if r['group'] == g) for g in counts}
    if branch == 'running':
        unsafe = any(not x['safe'] for r in rows for x in r['backends'].values())
        disagreement = any(r['backends']['native']['passed'] != r['backends']['cuda']['passed'] for r in rows)
        branch = ('route_uncertainty_stop' if unsafe or disagreement else
                  'finite_fallen_baseline_positive' if counts['baseline'] >= 2 else 'route_uncertainty_stop')
    result = dict(schema='holding_transfer_report_v1', branch=branch, runs=rows, passed_seeds=counts,
                  evaluated_seeds=evaluated, full_deck=len(audits) == 18,
                  read_only_transitions=transitions, weight_updates=0, elapsed_seconds=time.time()-started,
                  preregistration_sha256=file_sha256(output/'preregistration.json'))
    write_json(output/'summary.json', result)
    audited = {a['folder']: a for a in audits}
    independently_counted = {g: 0 for g in counts}
    for row in rows:
        passed = len(row['backends']) == 2 and all(
            audited[f"eval_{row['group']}_{row['seed']}_{b}"]['passed'] for b in row['backends'])
        independently_counted[row['group']] += int(passed)
        if 'passed' in row:
            assert row['passed'] == passed
        for b in row['backends']:
            metadata = read(output/f"eval_{row['group']}_{row['seed']}_{b}"/'summary.json')
            for name, digest in metadata['source_sha256'].items():
                assert prereg['source_sha256'][name] == digest
    assert independently_counted == counts and transitions <= 4500
    assert file_sha256(original_reset) == prereg['original_reset_sha256']
    assert file_sha256(learning/'reset.npz') == prereg['source_state_sha256']
    write_json(output/'verification.json', dict(verified=True, evaluations=audits, weight_updates=0,
                                              independently_counted_seeds=independently_counted,
                                              summary_sha256=file_sha256(output/'summary.json'),
                                              source_sha256=prereg['source_sha256']))
    print(json.dumps(dict(branch=branch, passed_seeds=counts, read_only_transitions=transitions)), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--learning', type=Path, required=True)
    parser.add_argument('--original-reset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-deadline', type=float, required=True)
    args = parser.parse_args()
    run(args.learning, args.original_reset, args.output, args.wall_deadline)


if __name__ == '__main__':
    main()
