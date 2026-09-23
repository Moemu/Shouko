"""Matched-budget and independent-validation evidence from completed native reports."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_route import summarize

root = Path(__file__).resolve().parent


def result(name):
    report = json.loads((root/name).read_text())
    assert report['completed']
    return summarize(report['conditions'][0])


development = [result(name) for name in ['old_reward_extend1024_native.json',
                'support_extend1024_native.json', 'support_replicate2027_native.json', 'support_refine2027_native.json']]
heldout = [result('support_refined_'+name+'.json') for name in
           ['holdout', 'yaw_holdout', 'long_holdout', 'push_holdout']]
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), constrained_layout=True)
ax = axes[0, 0]
x = np.arange(4)
ax.bar(x-.17, [r['successes'] for r in development], .34, color='#94a3b8', label='Old six criteria')
ax.bar(x+.17, [r['strict_gait_successes'] for r in development], .34, color='#0d9488', label='Strict swing screen')
for i, r in enumerate(development):
    ax.text(i+.17, r['strict_gait_successes']+.5, str(r['strict_gait_successes']), ha='center')
ax.set_xticks(x, ['Old reward\n2026, 65.5M', 'Support\n2026, 65.5M', 'Support\n2027, 65.5M', '+200 rounds\n2027, 91.8M'])
ax.set(ylabel='Episodes passing / 27', ylim=(0, 34), title='Matched 65.5M control + separate convergence extension')
ax.legend(loc='upper left', fontsize=8)
ax = axes[0, 1]
ax.bar(np.arange(4), [100*r['strict_gait_successes']/r['episodes'] for r in heldout], color='#0d9488')
for i, r in enumerate(heldout):
    ax.text(i, 100*r['strict_gait_successes']/r['episodes']+2,
            f"{r['strict_gait_successes']}/{r['episodes']}\nfalls {r['falls']}", ha='center', fontsize=9)
ax.set_xticks(np.arange(4), ['Nominal\n30 seconds', 'Yaw +/-0.2\n30 seconds', 'Long walk\n120 seconds', 'Lateral push\nat 10 seconds'])
ax.set(ylabel='Strict gait passes (%)', ylim=(0, 123), title='Unused cohorts, fixed seed-2027 refinement endpoint')
traces = [np.load(root/name) for name in ['old_reward_extend1024_trace.npz', 'support_extend1024_trace.npz']]
for side in range(2):
    ax = axes[1, side]
    for trace, name, color in zip(traces, ['Old reward', 'Phase support'], ['#94a3b8', '#0d9488']):
        selected = (trace['time'] >= 10) & (trace['time'] <= 12.4)
        ax.plot(trace['time'][selected], trace['clearance'][selected, side]*1000, color=color, label=name, lw=1.5)
    ax.axhline(25, color='#dc2626', ls='--', lw=1, label='25 mm swing screen')
    ax.set(xlabel='Time (s)', ylabel='Minimum sole clearance (mm)', ylim=(-5, 75),
           title=['Left sole', 'Right sole'][side]+' - supplemental CPU replay, seed-2026 policies')
    ax.legend(loc='upper right', fontsize=8)
fig.suptitle('Yumi gait evidence: ordinary phase-driven MLP, unchanged body and 50-to-12 interface\n'
             'Strict screen: old six criteria AND each foot >=1 swing/s, each swing >=0.12 s and >=25 mm', fontsize=12)
fig.supxlabel(f"Yaw settling: {heldout[1]['recovered']}/{heldout[1]['episodes']}; "
              f"120 s mean end-lateral displacement: {heldout[2]['lateral_m']:.2f} m. "
              'No connectome or natural-gait claim.', fontsize=10)
fig.savefig(root/'route_final_evidence.png', dpi=180)
