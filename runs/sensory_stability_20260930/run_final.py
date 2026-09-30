"""Consume new holdouts only after every preregistered S1 cell passes."""
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from app.train_full import file_sha256
from runs.sensory_stability_20260930.run_matrix import ROOT, read, save, summarize, accepted

GROUPS = [('normal', ['--cohorts', '3'], 24),
          ('yaw', ['--yaws', '-.2', '.2'], 16),
          ('long', ['--seconds', '120', '--trace'], 9),
          ('push_positive', ['--push-velocity', '.25'], 8),
          ('push_negative', ['--push-velocity', '-.25'], 8)]


def evaluate(label, cell, seed_base):
    out = ROOT/'final'/label
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(cell['checkpoint'])
    digest = cell['checkpoint_sha256']
    assert file_sha256(checkpoint) == digest
    report = dict(completed=False, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                  frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  selection_rule='Preregistered stage2 A/3031/S1 and B/4031/S1; no ranking or tuning.',
                  seed_base=seed_base, jobs=[], results={})
    execution = out/'execution.json'
    save(execution, report)
    for index, (name, extra, threshold) in enumerate(GROUPS):
        assert file_sha256(checkpoint) == digest
        evidence = out/(name+'.json')
        command = [sys.executable, '-u', '-m', 'app.evaluate_direction', '--checkpoints', str(checkpoint),
                   '--output', str(evidence), '--seed-base', str(seed_base+1000*index), *extra]
        row = dict(name=name, command=command)
        report['jobs'].append(row)
        save(execution, report)
        started = time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=2400,
                                     env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'))
        row.update(returncode=process.returncode, wall_seconds=time.monotonic()-started)
        save(execution, report)
        if process.returncode:
            raise RuntimeError(f'{label}: {name}')
        report['results'][name] = summarize(evidence, digest)
        save(execution, report)
        print(json.dumps(dict(label=label, group=name, summary=report['results'][name])), flush=True)
    report.update(completed=True, passed=all(accepted(name, report['results'][name], threshold)
                  for name, _, threshold in GROUPS),
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(execution, report)
    return label, report


def main():
    torch.set_num_threads(8)
    matrix = read(ROOT/'matrix.json')
    assert matrix['completed'] and matrix['stability_screen_passed']
    assert all(matrix['stages']['2'][f'{data}_{seed}_s1']['development_passed']
               for data in ['A', 'B'] for seed in [3031, 4031])
    used = set()
    for path, digest in matrix['data_sha256'].items():
        assert file_sha256(path) == digest
        dataset = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        used.update(dataset['episode_ids'].tolist())
    # Include all previous physical evaluation seeds, not only dataset episodes.
    for folder in ['runs/direction_control_20260929', 'runs/sensory_access_20260930', str(ROOT)]:
        for path in Path(folder).rglob('*.json'):
            data = read(path)
            if isinstance(data, dict) and data.get('kind') == 'native_direction_v1':
                for condition in data['conditions']:
                    for row in condition['results']:
                        used.update(test['seed'] for test in row['tests'])
    reserved = {base+1000*group+10*cohort+episode for base in [9000001, 9100001]
                for group in range(5) for cohort in range(3 if group == 0 else 1) for episode in range(9)}
    assert not used & reserved, sorted(used & reserved)
    report = dict(completed=False, matrix_sha256=file_sha256(ROOT/'matrix.json'),
                  source_sha256=file_sha256(__file__), seed_overlap=[], prior_unique_seeds=len(used), branches={})
    output = ROOT/'final.json'
    assert not output.exists()
    save(output, report)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(evaluate, label, matrix['stages']['2'][key], base)
                   for label, key, base in [('main', 'A_3031_s1', 9000001), ('verification', 'B_4031_s1', 9100001)]]
        for future in futures:
            label, result = future.result()
            report['branches'][label] = result
            save(output, report)
    report.update(completed=True, passed=all(item['passed'] for item in report['branches'].values()))
    save(output, report)


if __name__ == '__main__':
    main()
