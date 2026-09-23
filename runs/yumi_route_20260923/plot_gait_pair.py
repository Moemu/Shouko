"""Show old acceptance and actual foot clearance separately for the reward probe."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_route import summarize

root = Path(__file__).resolve().parent
pair = json.loads((root/'gait_paired_seed3026.json').read_text())
if not pair['completed']:
    raise ValueError('Wait for the paired native report')
initial = json.loads((root/'mlp_seed2026_native.json').read_text())['conditions'][0]
conditions = [initial, *pair['conditions']]
summary = [summarize(c) for c in conditions]
names = ['Initial MLP', 'Old reward', '+ Sole clearance']
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), constrained_layout=True)
ax = axes[0, 0]
x = np.arange(3)
ax.bar(x-.17, [c['successes'] for c in summary], width=.34, label='Old six criteria', color='#64748b')
ax.bar(x+.17, [c['strict_gait_successes'] for c in summary], width=.34, label='+ Sustained swing screen', color='#0d9488')
ax.set_xticks(x, names)
ax.set(ylabel='Episodes passing / 27', ylim=(0, 30), title='Matched native MuJoCo, 30 s')
ax.legend(fontsize=8, loc='upper left')
ax = axes[1, 0]
for i, condition in enumerate(pair['conditions']):
    tests = [t for g in condition['results'] for t in g['tests'] if t['criteria']['survived']]
    rates = [np.mean([t['gait']['feet'][side]['qualifying_swings_per_second'] for t in tests])
             if tests else np.nan for side in range(2)]
    ax.bar(np.arange(2)+(i-.5)*.32, rates, .32, label=f'{names[i+1]} (n={len(tests)})', color=['#2563eb', '#ea580c'][i])
ax.axhline(1, color='#dc2626', ls='--', lw=1, label='Predeclared screen')
ax.set_xticks([0, 1], ['Left', 'Right'])
ax.set(ylabel='Qualifying swings / s, survivors only', title='Airborne >=0.12 s AND clearance >=25 mm')
ax.legend(fontsize=8)
traces = [np.load(root/f) for f in ['gait_control_cpu_trace.npz', 'gait_phase_cpu_trace.npz']]
end = min(14, *(t['time'][-1] for t in traces))
for i, trace in enumerate(traces):
    selected = (trace['time'] >= max(0, end-4)) & (trace['time'] <= end)
    for side in range(2):
        axes[side, 1].plot(trace['time'][selected], trace['clearance'][selected, side]*1000,
                           label=names[i+1], color=['#2563eb', '#ea580c'][i], lw=1)
for side, name in enumerate(['Left sole', 'Right sole']):
    ax = axes[side, 1]
    ax.axhline(25, color='#dc2626', ls='--', lw=1)
    ax.set(xlabel='Time (s)', ylabel='Minimum sole clearance (mm)', title=name+' (supplemental CPU replay)')
    ax.set_ylim(min(-10, ax.get_ylim()[0]), max(65, ax.get_ylim()[1]))
    ax.legend(fontsize=8)
fig.suptitle('Reward probe: same MLP initialization, same seed and 400-update budget', fontsize=13)
fig.savefig(root/'gait_pair_evidence.png', dpi=180)
