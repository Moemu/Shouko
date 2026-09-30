"""Cross data and sampling seeds, then match the additional replay budget."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from app.train_full import file_sha256

ROOT = Path('runs/sensory_stability_20260930')
OLD = Path('runs/sensory_access_20260930')
SOURCE = Path('runs/direction_control_20260929/transfer_main/round4/last.pt')
GROUPS = [('normal', ['--cohorts', '3'], 24),
          ('long', ['--seconds', '120', '--trace'], 9),
          ('yaw', ['--yaws', '-.2', '.2'], 16),
          ('push_positive', ['--push-velocity', '.25'], 8),
          ('push_negative', ['--push-velocity', '-.25'], 8)]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def summarize(path, checkpoint_hash):
    evidence = read(path)
    assert evidence['completed']
    condition = evidence['conditions'][0]
    assert condition['checkpoint_sha256'] == checkpoint_hash
    tests = [test for row in condition['results'] for test in row['tests']]
    by_speed = {}
    for speed in [.35, .5, .65]:
        selected = [test for test in tests if test['target_speed'] == speed]
        by_speed[str(speed)] = dict(episodes=len(selected),
            signed_endpoint_mean_m=sum(t['direction']['signed_end_y_m'] for t in selected)/len(selected),
            maximum_lateral_m=max(t['direction']['maximum_lateral_m'] for t in selected))
    return dict(**condition['summary'], by_speed=by_speed,
                evidence_path=str(path), evidence_sha256=file_sha256(path))


def accepted(name, result, threshold):
    return (result['strict_passes'] >= threshold and result['falls'] <= 1 and
            (name != 'long' or result['direction_passes'] == 9))


def cell(stage, dataset_name, seed, arm, recipes, previous=None, repeat=False):
    key = f'{dataset_name}_{seed}_{arm}' + ('_repeat' if repeat else '')
    out = ROOT/f'stage{stage}'/key
    out.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, stage=stage, dataset=dataset_name, seed=seed, arm=arm,
                  repeat=repeat, jobs=[], results={}, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    execution = out/'execution.json'
    def persist():
        save(execution, report)
    def run(name, arguments):
        command = [sys.executable, '-u', *arguments]
        row = dict(name=name, command=command)
        report['jobs'].append(row)
        persist()
        start = time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=2400,
                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'))
        row.update(returncode=process.returncode, wall_seconds=time.monotonic()-start)
        persist()
        if process.returncode:
            raise RuntimeError(f'{key}: {name}')
    try:
        training = recipes[dataset_name]['arguments']['train'].copy()
        validation = recipes['A']['arguments']['validation']
        source = SOURCE if stage == 1 else Path(previous['checkpoint'])
        optimization_seed = seed if stage == 1 else seed+1
        if stage == 2:
            training += recipes['extra']['arguments']['train'][-4:]
        reuse = None
        if stage == 1 and not repeat and (dataset_name, seed) in [('A', 3031), ('B', 4031)]:
            folder = OLD if dataset_name == 'A' else OLD/'replica'
            reuse = (folder/arm/'last.pt', folder, arm)
        if stage == 2 and dataset_name == 'B' and seed == 4031 and arm == 's1':
            reuse = (OLD/'replica/dagger/round1/last.pt', OLD/'replica/dagger', 'round1_eval')
        if reuse:
            checkpoint = reuse[0]
            report['reused'] = True
        else:
            run('train', ['-m', 'app.train_sensory_readout', '--source', str(source),
                '--train', *training, '--validation', *validation, '--output', str(out/'train'),
                '--seed', str(optimization_seed), *(['--sensory'] if arm == 's1' else []),
                *(['--calibrate-vy'] if stage == 1 else [])])
            checkpoint = out/'train/last.pt'
            report['reused'] = False
        fit_path = checkpoint.with_name('fit.json')
        fit = read(fit_path)
        assert fit['completed'] and file_sha256(checkpoint) == fit['checkpoint_sha256']
        assert fit['arguments']['train'] == training and fit['arguments']['validation'] == validation
        assert fit['arguments']['seed'] == optimization_seed
        assert fit['arguments']['updates'] == 500 and fit['arguments']['batch'] == 128
        assert fit['arguments']['sensory'] == (arm == 's1')
        assert fit['arguments']['calibrate_vy'] == (stage == 1)
        assert fit['source_checkpoint_sha256'] == file_sha256(source)
        assert fit['arguments']['readout_lr'] == 1e-5 and fit['arguments']['sensory_lr'] == 1e-4
        for name, digest in fit['source_sha256'].items():
            assert file_sha256(Path('app')/name) == digest, name
        report.update(checkpoint=str(checkpoint), checkpoint_sha256=fit['checkpoint_sha256'],
            fit_path=str(fit_path), fit_sha256=file_sha256(fit_path),
            sample_indices_sha256=fit['sample_indices_sha256'],
            vy_observation_std=fit['vy_observation_std'])
        persist()
        passed = True
        for name, extra, threshold in GROUPS:
            if reuse:
                evidence = reuse[1]/f'{reuse[2]}_{name}.json'
            else:
                evidence = out/f'eval_{name}.json'
                run('eval_'+name, ['-m', 'app.evaluate_direction', '--checkpoints', str(checkpoint),
                    '--output', str(evidence), '--seed-base', '310201', *extra])
            result = summarize(evidence, report['checkpoint_sha256'])
            report['results'][name] = result
            persist()
            if not accepted(name, result, threshold):
                passed = False
                break
        report.update(completed=True, development_passed=passed,
                      finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        persist()
        return key, report
    except Exception as error:
        report['error'] = repr(error)
        persist()
        raise


def main():
    torch.set_num_threads(8)
    output = ROOT/'matrix.json'
    if output.exists():
        raise FileExistsError(output)
    recipes = dict(A=read(OLD/'s1/fit.json'), B=read(OLD/'replica/s1/fit.json'),
                   extra=read(OLD/'replica/dagger/round1/fit.json'))
    assert file_sha256(SOURCE) == 'e550239157c4ed2bf1b67914377e4e77c7d38c9d29a64066097abe9369072a10'
    expected = {}
    for recipe in recipes.values():
        for path, digest in recipe['data_sha256'].items():
            if path in expected:
                assert expected[path] == digest
            expected[path] = digest
    for path, digest in expected.items():
        assert file_sha256(path) == digest, path
    report = dict(completed=False, stages={}, data_sha256=expected,
        source_sha256=file_sha256(__file__), max_concurrent_processes=2,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(output, report)
    configurations = [(data, seed, arm) for data in ['A', 'B'] for seed in [3031, 4031] for arm in ['c0', 's1']]
    for stage in [1, 2]:
        results = {}
        report['stages'][str(stage)] = results
        jobs = [(data, seed, arm, False) for data, seed, arm in configurations]
        if stage == 1:
            jobs.append(('A', 3031, 's1', True))
        errors = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for data, seed, arm, repeat in jobs:
                prior = report['stages']['1'][f'{data}_{seed}_{arm}'] if stage == 2 else None
                future = pool.submit(cell, stage, data, seed, arm, recipes, prior, repeat)
                futures[future] = (data, seed, arm, repeat)
            for future in as_completed(futures):
                try:
                    key, result = future.result()
                    results[key] = result
                    print(json.dumps(dict(stage=stage, cell=key, passed=result['development_passed'],
                        long=result['results'].get('long'))), flush=True)
                except Exception as error:
                    errors.append(dict(cell=futures[future], error=repr(error)))
                save(output, report)
        if errors:
            report['errors'] = errors
            save(output, report)
            raise RuntimeError(errors)
        for data in ['A', 'B']:
            for seed in [3031, 4031]:
                c0, s1 = (results[f'{data}_{seed}_{arm}'] for arm in ['c0', 's1'])
                assert c0['sample_indices_sha256'] == s1['sample_indices_sha256']
                assert c0['vy_observation_std'] == s1['vy_observation_std']
        if stage == 1:
            reference = torch.load(results['A_3031_s1']['checkpoint'], map_location='cpu', weights_only=True, mmap=True)['state_dict']
            repeated = torch.load(results['A_3031_s1_repeat']['checkpoint'], map_location='cpu', weights_only=True, mmap=True)['state_dict']
            report['repeat_parameter_max_difference'] = {name:float((repeated[name]-reference[name]).abs().max())
                for name in ['encoder.weight', 'readout.weight', 'readout.bias']}
            save(output, report)
    final = report['stages']['2']
    report['stability_screen_passed'] = all(final[f'{data}_{seed}_s1']['development_passed']
        for data in ['A', 'B'] for seed in [3031, 4031])
    report['c0_passed_cells'] = sum(final[f'{data}_{seed}_c0']['development_passed']
        for data in ['A', 'B'] for seed in [3031, 4031])
    report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(output, report)


if __name__ == '__main__':
    main()
