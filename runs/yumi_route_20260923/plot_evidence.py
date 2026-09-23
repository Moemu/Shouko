"""Render diagnostics from saved evidence, without rerunning or changing physics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
conditions = [json.loads((root/p).read_text())['conditions'][0]
              for p in ['baseline_native.json', 'mlp_seed2026_native.json']]
names = ['Source connectome', 'MLP seed 2026']
axes[0].bar(names, [c['summary']['successes'] for c in conditions], color=['#64748b', '#2563eb'])
axes[0].set_ylim(0, 30)
axes[0].set_ylabel('Episodes passing the old six criteria / 27')
for i, c in enumerate(conditions):
    axes[0].text(i, c['summary']['successes']+0.7, str(c['summary']['successes']), ha='center')
axes[0].set_title('Tracking improved; gait remains unproven')
trace = np.load(root/'mlp_cpu_trace.npz')
selected = (trace['time'] >= 10) & (trace['time'] <= 14)
for side, name in enumerate(['Left sole', 'Right sole']):
    axes[1].plot(trace['time'][selected], trace['clearance'][selected, side]*1000, label=name, lw=1)
axes[1].axhline(25, color='#dc2626', ls='--', label='25 mm swing screen')
axes[1].set(xlabel='Time (s)', ylabel='Minimum sole clearance (mm)', ylim=(-10, 29),
            title='MLP remains near the floor')
axes[1].legend(loc='upper right', fontsize=8)
fig.suptitle('Same Yumi body and action interface; different policy and training history', fontsize=12)
fig.text(.53, -.025, 'Right: supplemental CPU replay, seed 93002, command 0.5 m/s. Acceptance uses CUDA batch 9.',
         ha='center', fontsize=8)
fig.savefig(root/'mlp_gait_evidence.png', dpi=180, bbox_inches='tight')
