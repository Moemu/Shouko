"""Verify the extracted package and run batch-one native CSR for one model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from app.evaluate_locomotion import load_policy, evaluate_batch
from app.sim import Body
import app.full_brain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['primary', 'replica'], default='primary')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert Path(app.full_brain.__file__).resolve().parent == root/'app', 'Runtime escaped package'
    for row in manifest['files']:
        path = root/row['path']
        with path.open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == row['sha256'], row['path']
    torch.set_num_threads(8)
    checkpoint = root/manifest['models'][args.role]['path']
    policy, cfg, kind = load_policy(str(checkpoint))
    assert cfg['observation_size'] == 50 and cfg['action_size'] == 12 and cfg['neural_steps'] == 4
    body = Body(False, 'yumi', interface=cfg['physics_interface'], observation_size=cfg['observation_size'])
    base = 9000001 if args.role == 'primary' else 9100001
    started = time.monotonic()
    rows = []
    for speed, seed, seconds in [(.35, base, 30), (.5, base+1, 30), (.65, base+2, 30), (.65, base+2002, 120)]:
        result = evaluate_batch(policy, [body], [seed], [speed], [0.], seconds)
        test = result['tests'][0]
        strict = test['success'] and len(test['gait']['feet']) == 2 and all(
            f['qualifying_swings_per_second'] >= 1 for f in test['gait']['feet'])
        corridor = test['gait']['maximum_lateral_m'] <= 2
        rows.append(dict(seconds=seconds, strict=bool(strict), corridor=bool(corridor), result=result))
        print(json.dumps(dict(role=args.role, seed=seed, speed=speed, seconds=seconds,
                              strict=bool(strict), maximum_lateral_m=test['gait']['maximum_lateral_m'])), flush=True)
    passed = all(r['strict'] and (r['seconds'] != 120 or r['corridor']) for r in rows)
    report = dict(completed=True, passed=passed, role=args.role,
                  checkpoint_sha256=manifest['models'][args.role]['sha256'],
                  source_head=manifest['source_head'], backend='native CSR', batch=1,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  wall_seconds=time.monotonic()-started, results=rows,
                  limitation='Known seeds: package and single-world reproduction, not new holdouts or browser validation.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    if not passed:
        raise SystemExit('Package smoke failed; see '+str(output))


if __name__ == '__main__':
    main()
