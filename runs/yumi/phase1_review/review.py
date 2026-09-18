"""Short native-physics validation; training seeds, not final held-out acceptance."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from app.evaluate_full import episode
from app.full_brain import ConnectomePolicy
from app.sim import Body


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    torch.set_num_threads(4)
    target = Path(__file__).parent
    checkpoint = ROOT / 'runs/yumi/best.pt'
    checkpoint_hash = digest(checkpoint)
    model = ConnectomePolicy(device='cuda').eval()
    extra = model.load(checkpoint)
    evaluation = extra['evaluation']
    gpu_tests = evaluation['tests']
    report = dict(kind='training_seed_native_diagnostic', checkpoint_sha256=checkpoint_hash,
                  checkpoint_updates=extra['updates'], robot='yumi', teacher_used=False,
                  final_acceptance=False, torch=torch.__version__,
                  body_sha256={name: digest(ROOT / 'app/yumi_description' / name)
                               for name in ['scene.xml', 'yumi.xml', 'yumi.yaml']},
                  gpu_screening={key: value for key, value in evaluation.items() if key != 'tests'},
                  gpu_survivors=[row for row in gpu_tests if not row['fallen']], tests=[])
    body = Body(load_motor_policy=False, robot='yumi')
    assert body.policy is None
    for index, speed in enumerate([0.35, 0.5, 0.65]):
        result, frames = episode(model, body, seed=1001, speed=speed, seconds=15)
        report['tests'].append(result)
        (target / f'replay_{index}.json').write_text(json.dumps(dict(result=result, frames=frames)))
        (target / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps(result), flush=True)
    assert digest(checkpoint) == checkpoint_hash, 'Checkpoint changed during review'
    report['complete'] = True
    (target / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
