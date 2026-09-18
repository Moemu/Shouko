"""Compare candidates on the fixed training-validation seed, before held-out tests."""
import json

import torch

from .full_brain import ConnectomePolicy, ROOT
from .gpu_body import GPUHumanoid
from .train_full import evaluate, atomic_json


def main():
    torch.set_num_threads(4)
    runs = ROOT/'runs/cloud'
    model = ConnectomePolicy().eval()
    env = GPUHumanoid(16)
    candidates = []
    for name in ['straight_v1.pt', 'best.pt', 'last.pt']:
        extra = model.load(runs/name)
        result = evaluate(model, env, seconds=30)
        candidates.append(dict(name=name, score=result['score'], evaluation=result, updates=extra['updates']))
        print(json.dumps(candidates[-1]), flush=True)
    selected = max(candidates, key=lambda row: row['score'])
    model.load(runs/selected['name'])
    model.save(runs/'best.pt', dict(evaluation=selected['evaluation'], updates=selected['updates']))
    atomic_json(runs/'evaluation.json', selected['evaluation'])
    atomic_json(runs/'selection.json', dict(selected=selected['name'], candidates=candidates))


if __name__ == '__main__':
    main()
