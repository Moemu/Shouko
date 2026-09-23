"""First positive gait evidence, with the sample-budget difference explicit."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_route import summarize

root = Path(__file__).resolve().parent
conditions = [json.loads((root/name).read_text())['conditions'][0] for name in
              ['mlp_seed2026_native.json', 'support_extend1024_native.json']]
summaries = [summarize(c) for c in conditions]
labels = ['Old reward\n13.1M transitions', 'Phase support\n65.5M transitions']
colors = ['#94a3b8', '#0d9488']
fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.3), constrained_layout=True)
x = np.arange(2)
ax = axes[0, 0]
ax.bar(x-.17, [s['successes'] for s in summaries], .34, color='#94a3b8', label='Old six criteria')
ax.bar(x+.17, [s['strict_gait_successes'] for s in summaries], .34, color='#0d9488', label='Strict swing screen')
ax.set_xticks(x, labels)
ax.set(ylabel='Episodes passing / 27', ylim=(0, 34), title='Native MuJoCo, 30 seconds')
ax.legend(loc='upper left', fontsize=8)
for i, s in enumerate(summaries):
    ax.text(i+.17, s['strict_gait_successes']+.5, str(s['strict_gait_successes']), ha='center')
ax = axes[0, 1]
for i, c in enumerate(conditions):
    feet = [f for g in c['results'] for t in g['tests'] for f in t['gait']['feet']]
    ax.scatter([f['swing_seconds_median'] for f in feet],
               [f['swing_clearance_median']*1000 for f in feet],
               s=20, alpha=.65, color=colors[i], label=labels[i].replace('\n', ', '))
ax.axvline(.12, color='#dc2626', ls='--', lw=1)
ax.axhline(25, color='#dc2626', ls='--', lw=1)
ax.set(xlabel='Per-foot median airborne duration (s)', ylabel='Per-foot median peak clearance (mm)',
       xlim=(0, .42), ylim=(0, 70), title='Each point: one foot in one native episode')
ax.legend(loc='lower right', fontsize=8)
traces = [np.load(root/name) for name in ['mlp_cpu_trace.npz', 'support_extend1024_trace.npz']]
for side in range(2):
    ax = axes[1, side]
    for i, trace in enumerate(traces):
        selected = (trace['time'] >= 10) & (trace['time'] <= 12.4)
        ax.plot(trace['time'][selected], trace['clearance'][selected, side]*1000,
                color=colors[i], label=['Old reward', 'Phase support'][i], lw=1.5)
    ax.axhline(25, color='#dc2626', ls='--', lw=1, label='25 mm swing screen')
    ax.set(xlabel='Time (s)', ylabel='Minimum sole clearance (mm)', ylim=(-4, 70),
           title=['Left sole', 'Right sole'][side]+' - supplemental CPU replay')
    ax.legend(fontsize=8, loc='upper right')
fig.suptitle('First strict gait: same Yumi body and 50-to-12 interface, phase-driven MLP\n'
             'Independent training replication and equal-budget old-reward control are pending', fontsize=12)
fig.savefig(root/'first_strict_gait_evidence.png', dpi=180)
