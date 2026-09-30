"""Verify locally recovered matrix checkpoints and physical trajectory evidence."""
import json
from pathlib import Path
import numpy as np
import torch

from app.train_full import file_sha256
from app.train_sensory_readout import check_boundary
from runs.sensory_stability_20260930.run_matrix import ROOT, SOURCE, GROUPS, read, accepted

torch.set_num_threads(8)
matrix = read(ROOT/'matrix.json')
assert matrix['completed'] and not matrix.get('errors')
snapshot = read(ROOT/'preregistration.json')
for name, digest in snapshot.items():
    assert file_sha256(ROOT/name) == digest, name
assert matrix['source_sha256'] == file_sha256(ROOT/'run_matrix.py')
assert file_sha256(SOURCE) == 'e550239157c4ed2bf1b67914377e4e77c7d38c9d29a64066097abe9369072a10'
before = torch.load(SOURCE, map_location='cpu', weights_only=True, mmap=True)['state_dict']
used = set()
for path, digest in matrix['data_sha256'].items():
    assert file_sha256(path) == digest, path
    data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    used.update(data['episode_ids'].tolist())
report = dict(completed=False, matrix_sha256=file_sha256(ROOT/'matrix.json'),
              verified_data_files=len(matrix['data_sha256']), cells={}, final={})


def verify_evidence(result, checkpoint_hash, final=False):
    path = Path(result['evidence_path'])
    assert file_sha256(path) == result['evidence_sha256'], path
    evidence = read(path)
    assert evidence['completed']
    condition = evidence['conditions'][0]
    assert condition['checkpoint_sha256'] == checkpoint_hash
    for name, digest in evidence['source_sha256'].items():
        assert file_sha256(Path('app')/name) == digest, name
    for name, digest in evidence['body_xml_sha256'].items():
        assert file_sha256(name) == digest, name
    tests = [test for row in condition['results'] for test in row['tests']]
    assert result['strict_passes'] == sum(test['strict_success'] for test in tests)
    assert result['direction_passes'] == sum(test['direction_success'] for test in tests)
    assert result['falls'] == sum(test['fallen'] for test in tests)
    if final:
        assert not used & {test['seed'] for test in tests}
    for row in condition['results']:
        if 'trace' not in row:
            continue
        trace = path.parent/row['trace']
        assert file_sha256(trace) == row['trace_sha256']
        with np.load(trace) as arrays:
            for index, test in enumerate(row['tests']):
                maximum = float(np.abs(arrays[f'episode{index}_qpos'][:, 1]).max())
                assert abs(maximum-test['direction']['maximum_lateral_m']) < 1e-10
                assert test['direction_success'] == (test['strict_success'] and maximum <= 2)
    return dict(episodes=len(tests), strict_passes=result['strict_passes'],
                corridor_passes=result['direction_passes'], falls=result['falls'])


for stage, cells in matrix['stages'].items():
    for key, cell in cells.items():
        assert cell['completed']
        checkpoint = Path(cell['checkpoint'])
        assert file_sha256(checkpoint) == cell['checkpoint_sha256']
        assert file_sha256(cell['fit_path']) == cell['fit_sha256']
        fit = read(cell['fit_path'])
        assert fit['completed'] and fit['checkpoint_sha256'] == cell['checkpoint_sha256']
        assert fit['arguments']['updates'] == 500
        assert fit['source_checkpoint_sha256'] == file_sha256(fit['arguments']['source'])
        for name, digest in fit['source_sha256'].items():
            assert file_sha256(Path('app')/name) == digest, name
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)['state_dict']
        changes = check_boundary(before, state, cell['arm'] == 's1')
        del state
        groups = {name:verify_evidence(result, cell['checkpoint_sha256']) for name, result in cell['results'].items()}
        passed = len(groups) == 5 and all(accepted(name, cell['results'][name], threshold) for name, _, threshold in GROUPS)
        assert passed == cell['development_passed']
        report['cells'][stage+'/'+key] = dict(checkpoint_sha256=cell['checkpoint_sha256'],
            development_passed=passed, groups=groups, parameter_changes=changes)
    for data in ['A', 'B']:
        for seed in [3031, 4031]:
            pair = [cells[f'{data}_{seed}_{arm}'] for arm in ['c0', 's1']]
            assert pair[0]['sample_indices_sha256'] == pair[1]['sample_indices_sha256']
            assert pair[0]['vy_observation_std'] == pair[1]['vy_observation_std']
            stream = read(ROOT/'sampling_audit.json')[f'stage{stage}_{data}']['streams'][str(seed)]
            assert pair[0]['sample_indices_sha256'] == stream['sample_indices_sha256']
screen = all(matrix['stages']['2'][f'{data}_{seed}_s1']['development_passed']
             for data in ['A', 'B'] for seed in [3031, 4031])
assert screen == matrix['stability_screen_passed']
if (ROOT/'final.json').exists():
    final = read(ROOT/'final.json')
    assert final['completed'] and final['matrix_sha256'] == file_sha256(ROOT/'matrix.json')
    for label, branch in final['branches'].items():
        assert file_sha256(branch['checkpoint']) == branch['checkpoint_sha256']
        key = 'A_3031_s1' if label == 'main' else 'B_4031_s1'
        assert branch['checkpoint_sha256'] == matrix['stages']['2'][key]['checkpoint_sha256']
        groups = {name:verify_evidence(result, branch['checkpoint_sha256'], True) for name, result in branch['results'].items()}
        assert {name:group['episodes'] for name, group in groups.items()} == dict(normal=27, yaw=18, long=9, push_positive=9, push_negative=9)
        assert branch['passed'] == all(accepted(name, branch['results'][name], threshold) for name, _, threshold in GROUPS)
        report['final'][label] = dict(passed=branch['passed'], groups=groups)
report.update(completed=True, stability_screen_passed=screen, source_sha256=file_sha256(__file__))
(ROOT/'closeout.json').write_text(json.dumps(report, indent=2))
print(json.dumps(dict(completed=True, verified_cells=len(report['cells']),
                     stability_screen_passed=screen, final=report['final'])))
