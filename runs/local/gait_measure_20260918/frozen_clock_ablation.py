import json
import sys

import numpy as np
import torch

sys.path.insert(0, 'D:/Project/Neuromechfly')
from app.measure_gait import LINEAGES, ClockBody, file_sha256, lineage_interface, lineage_provenance, run_episode  # noqa: E402
from app.full_brain import ConnectomePolicy  # noqa: E402

torch.set_num_threads(4)
FROZEN_PERIOD = 1.0e9  # phase = t * 2*pi / 1e9 stays ~0: sin~0, cos~1 for the whole episode

episodes = []
for name in ['squat', 'tall']:
    spec = LINEAGES[name]
    model = ConnectomePolicy(device='cuda').eval()
    model.load(spec['checkpoint'])
    body = ClockBody(load_motor_policy=False, robot='yumi', interface=lineage_interface(spec['interface_yaml']))
    body.clock_period = FROZEN_PERIOD
    for seed in [8001, 8002, 8003]:
        summary, _ = run_episode(model, body, seed, 0.5, 30.0)
        episodes.append(dict(event='frozen_clock', lineage=name, frozen_period_s=FROZEN_PERIOD, **summary))
        print(json.dumps(episodes[-1]), flush=True)

with open('runs/local/gait_measure_20260918/frozen_clock.json', 'w') as handle:
    json.dump(dict(kind='frozen_clock_ablation', cmd=0.5, seconds=30.0, seeds=[8001, 8002, 8003],
                   provenance={name: lineage_provenance(name) for name in ['squat', 'tall']},
                   episodes=episodes), handle, indent=2)
print('saved frozen_clock.json', flush=True)
