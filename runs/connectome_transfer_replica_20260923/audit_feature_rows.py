"""Recompute ordered frozen features from hash-verified raw observation rows."""
import hashlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.imitate_yumi import dataset, predict
from app.evaluate_locomotion import load_policy

torch.set_num_threads(8)
root = Path('runs/connectome_transfer_ridge_20260923')
def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

source = Path('runs/source/best.pt')
assert sha(source) == '458fc465a6fb802a338c218fc4be414fc9153376c5d2f76f116518e6cc5aa475'
assert sha('runs/teacher/last.pt') == '9c5ca9339b26f5af2e8acf94e7edd3b9371ef5efb41d647bcbd7904243ac50e1'
original = torch.load(source, weights_only=True, map_location='cpu')['state_dict']
rows = []
for name in ['ridge1e4', 'replica']:
    candidate = root/name
    report = json.loads((candidate/'fit.json').read_text())
    reference = Path(report['reference'])
    assert sha(reference/'fit.json') == report['reference_fit_sha256']
    assert sha(reference/'features.pt') == report['feature_cache_sha256']
    assert sha(candidate/'last.pt') == report['checkpoint_sha256']
    fit = json.loads((reference/'fit.json').read_text())
    assert sha(reference/'last.pt') == fit['checkpoint_sha256']
    for path, digest in report['data_sha256'].items():
        assert sha(path) == digest
    state = torch.load(candidate/'last.pt', weights_only=True, map_location='cpu')['state_dict']
    refstate = torch.load(reference/'last.pt', weights_only=True, map_location='cpu')['state_dict']
    for key in original:
        if not key.startswith('readout.'):
            assert torch.equal(original[key], state[key]), key
            assert torch.equal(original[key], refstate[key]), key
    cache = torch.load(reference/'features.pt', weights_only=True, map_location='cpu')
    model, _, _ = load_policy(str(reference/'last.pt'))
    groups = []
    for group in ['train', 'validation']:
        data = dataset(fit['arguments'][group])
        features = predict(model, data['observations'], fit['arguments']['batch'], True)
        assert torch.equal(features, cache[group]), (name, group)
        digests = {}
        for field in ['observations', 'targets', 'episode_ids']:
            tensor = data[field].contiguous()
            header = json.dumps(dict(shape=list(tensor.shape), dtype=str(tensor.dtype)), sort_keys=True).encode()
            digests[field] = hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()
        groups.append(dict(group=group, rows=len(features), ordered_tensor_sha256=digests,
                           regenerated_features_bitwise_equal=True))
    rows.append(dict(candidate=name, checkpoint_sha256=report['checkpoint_sha256'],
                     all_non_readout_tensors_equal_fixed_source_and_reference=True, groups=groups))
    del model, cache
    torch.cuda.empty_cache()
result = dict(fixed_source_sha256=sha(source), fixed_teacher_sha256=sha('runs/teacher/last.pt'), verified=rows)
(root/'feature_row_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
