"""Verify sparse gradients, learned graph participation, and checkpoint-bound walking evidence."""
import hashlib
import json

import numpy as np
import torch

from .full_brain import ConnectomePolicy, ROOT
from .gpu_benchmark import sparse_check
from .sim import Body


def main():
    torch.set_num_threads(4)
    runs = ROOT/'runs/cloud'
    sparse_check('cuda')
    model = ConnectomePolicy()
    model.load(runs/'best.pt')
    assert model.n == 166700 and model.weight.numel() == 25582938
    assert not np.intersect1d(model.inputs.cpu(), model.outputs.cpu()).size
    assert bool((model.edge_delta != 0).any()), 'Brain connections were never trained'
    body = Body(load_motor_policy=False)
    assert body.policy is None
    observation = torch.tensor(body.motor_observation([0.5, 0, 0]), device='cuda')[None]
    action, _ = model(observation)
    action.square().mean().backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
        gradients[name] = float(parameter.grad.abs().max())
        assert gradients[name] > 0, name
    with (runs/'best.pt').open('rb') as handle:
        checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    result = json.loads((runs/'heldout.json').read_text())
    assert result['checkpoint_sha256'] == checkpoint_hash and result['complete']
    assert result['teacher_used'] is False
    assert result['graph_sha256'] == model.meta['graph_sha256']
    assert result['successes'] == result['attempts'] == 9
    assert all(row['success'] for row in result['long_walks'] + result['perturbations'])
    assert len(result['controls']) == 3 and not any(row['success'] for row in result['controls'])
    report = dict(passed=True, checkpoint_sha256=checkpoint_hash,
                  trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
                  gradient_maxima=gradients,
                  changed_edge_fraction=float((model.edge_delta != 0).float().mean()),
                  checks=['sparse forward and backward dense reference', 'complete source graph',
                          'disjoint sensory and motor neurons', 'gradients reach every parameter group',
                          'trained connectome weights', 'teacher-free inference body',
                          'checkpoint-bound independent walking and causal controls'])
    (runs/'checks_full.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
