import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path('runs/connectome_transfer_replica_20260923')
primary = Path('runs/connectome_transfer_ridge_20260923')
def read(folder, name):
    condition = json.loads((folder/(name+'.json')).read_text())['conditions'][0]
    tests = [t for group in condition['results'] for t in group['tests']]
    return tests, condition['summary']
def strict(t):
    return t['success'] and all(f['qualifying_swings_per_second'] >= 1 for f in t['gait']['feet'])

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
names = ['holdout', 'yaw_holdout', 'long_holdout', 'push_holdout']
colors = ['#087f8c', '#cb7853']
for index, (folder, label) in enumerate([(primary, 'Primary'), (root, 'Independent data + one extra round')]):
    groups = [read(folder, name)[0] for name in names]
    values = [100*sum(map(strict, group))/len(group) for group in groups]
    positions = np.arange(4)+(index-.5)*.36
    axes[0,0].bar(positions, values, .34, color=colors[index], label=label)
    for x, group in zip(positions, groups):
        axes[0,0].text(x, 102, f'{sum(map(strict,group))}/{len(group)}', ha='center', fontsize=10)
axes[0,0].set(xticks=range(4), xticklabels=['Normal', 'Yaw +/-0.2', '120 seconds', 'Lateral impulse'], ylim=(0,130), ylabel='Strict passes (%)', title='Two fixed candidates, unused final cohorts; zero falls')
axes[0,0].legend(loc='upper left', fontsize=8)

axes[0,1].bar(range(5), [0,27,7,6,6], color=['#909b9d',colors[0],colors[1],colors[1],colors[1]])
axes[0,1].set(xticks=range(5), xticklabels=['Teacher-only\nnormal /27', 'DAgger 3\nnormal /27', 'DAgger 3\npush /9', 'Push DAgger\nlong /9', 'Same-schedule\nreplica push /9'], ylim=(0,31), ylabel='Strict passes (denominators differ)', title='Failures retained; success needed distribution coverage')
for i, value in enumerate([0,27,7,6,6]): axes[0,1].text(i,value+.4,str(value),ha='center')

trace = np.load(primary/'candidate_trace.npz')
mask = (trace['time']>=10)&(trace['time']<=12.5)
for side, label in enumerate(['Left sole', 'Right sole']):
    axes[1,0].plot(trace['time'][mask], 1000*trace['clearance'][mask,side],label=label)
axes[1,0].axhline(25,color='#888888',ls='--',label='25 mm swing screen')
axes[1,0].set(xlabel='Time (s)',ylabel='Minimum sole clearance (mm)',title='Primary representative trace: single support 90%, flight 0%')
axes[1,0].legend(fontsize=9)

axes[1,1].axis('off')
axes[1,1].text(0,1,'What this establishes',fontsize=16,weight='bold',va='top')
axes[1,1].text(0,.87,'166,700 neurons and 25,582,938 edges retained\nOnly 9,792 readout parameters changed\nNo teacher in evaluation; disconnected graph: 0/9\nNative CSR compatibility: both candidates 27/27',fontsize=12,va='top',linespacing=1.6)
axes[1,1].text(0,.48,'Boundaries that remain',fontsize=16,weight='bold',va='top')
axes[1,1].text(0,.35,'120 s mean lateral drift: 5.47 m / 4.90 m\nPrecise yaw recovery: 9/18 / 6/18\nExplicit phase input remains; no autonomous CPG claim\nSame source graph; no biological topology advantage shown',fontsize=12,va='top',linespacing=1.6)
fig.suptitle('Full-connectome closed-loop gait transfer | 2026-09-23',fontsize=18)
fig.tight_layout(rect=[0,.04,1,.95])
fig.text(.5,.018,'Native MuJoCo, fixed Yumi interface. Extra-data replication is not an equal-budget replication.',ha='center',fontsize=10)
fig.savefig(root/'final_evidence.png',dpi=150)
