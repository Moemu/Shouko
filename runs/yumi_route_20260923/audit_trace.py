"""Describe support mode in an existing CPU trace; never replaces native batch acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('trace', type=Path)
args = parser.parse_args()
output = args.trace.with_name(args.trace.stem+'_audit.json')
if output.exists():
    raise FileExistsError(output)
trace = np.load(args.trace)
metadata = json.loads(args.trace.with_suffix('.json').read_text())
dt = float(np.median(np.diff(trace['time'])))
selected = trace['time'] >= 2
contacts = trace['contacts'].astype(bool)
feet = []
for side in range(2):
    transitions = np.diff(np.r_[False, ~contacts[:, side], False].astype(int))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    bouts = [dict(start_seconds=float(trace['time'][a]), seconds=round((b-a)*dt, 8),
                  peak_minimum_sole_clearance=float(trace['clearance'][a:b, side].max()))
             for a, b in zip(starts, ends) if a > 0 and b < len(contacts)]
    feet.append(dict(complete_bouts=bouts,
                     qualifying=sum(b['seconds'] >= .12 and b['peak_minimum_sole_clearance'] >= .025 for b in bouts)))
report = dict(trace=str(args.trace), trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
              checkpoint_sha256=metadata['sha256'], device=metadata['device'], seed=metadata['seed'],
              command=metadata['command'], duration=float(trace['time'][-1]),
              support_window='From 2 seconds to trace end',
              support_fractions={name: float((contacts[selected].sum(1) == count).mean())
                                 for count, name in enumerate(['flight', 'single', 'double'])},
              pelvis_height_range=[float(trace['qpos'][selected, 2].min()), float(trace['qpos'][selected, 2].max())],
              feet=feet, limitation='One supplemental CPU replay; no population or natural-gait claim.')
output.write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k != 'feet'}))
