import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('runs/connectome_transfer_20260923')
def rows(name):
    d=json.loads((root/(name+'.json')).read_text())
    return [t for g in d['conditions'][0]['results'] for t in g['tests']]
def strict(t):
    return t['success'] and all(f['qualifying_swings_per_second']>=1 for f in t['gait']['feet'])
fig,ax=plt.subplots(2,2,figsize=(16,10))
names=['readout_native','dagger1_native','dagger2_native','dagger3_native','replica3_native']
sets=[rows(n) for n in names];x=np.arange(len(names))
passes=[sum(strict(t) for t in ts) for ts in sets];falls=[sum(t['fallen'] for t in ts) for ts in sets]
ax[0,0].bar(x-.18,passes,.36,label='Strict gait passes',color='#08988b');ax[0,0].bar(x+.18,falls,.36,label='Falls',color='#be6b64')
for i,v in enumerate(passes): ax[0,0].text(i-.18,v+.4,str(v),ha='center')
ax[0,0].set_xticks(x,['Teacher-only\nreadout','DAgger 1','DAgger 2','DAgger 3','Independent\ndata replica'])
ax[0,0].set_ylim(0,33);ax[0,0].set_ylabel('Episodes / 27');ax[0,0].legend();ax[0,0].set_title('Only the 9,792 readout parameters change')
names=['student_holdout','student_yaw_holdout','student_long_holdout','student_push_holdout']
sets=[rows(n) for n in names]
ax[0,1].bar(np.arange(4),[100*sum(strict(t) for t in ts)/len(ts) for ts in sets],color='#08988b')
for i,ts in enumerate(sets): ax[0,1].text(i,103,f'{sum(strict(t) for t in ts)}/{len(ts)}\nfalls {sum(t["fallen"] for t in ts)}',ha='center')
ax[0,1].set_xticks(range(4),['New starts','Yaw +/-0.2','120 seconds','Lateral push']);ax[0,1].set_ylim(0,125)
ax[0,1].set_ylabel('Strict gait passes (%)');ax[0,1].set_title('Preselected DAgger-3 checkpoint: unused cohorts')
trace=np.load(root/'student_trace.npz');mask=(trace['time']>=10)&(trace['time']<=12.5)
for side,label in enumerate(['Left sole','Right sole']): ax[1,0].plot(trace['time'][mask],1000*trace['clearance'][mask,side],label=label)
ax[1,0].axhline(25,color='#be6b64',ls='--',label='25 mm swing screen');ax[1,0].legend();ax[1,0].set_xlabel('Time (s)');ax[1,0].set_ylabel('Minimum sole clearance (mm)');ax[1,0].set_title('Representative native trace: no teacher in control')
durations=[np.mean([t['seconds'] for t in ts]) for ts in [rows(n) for n in ['readout_native','dagger1_native','dagger2_native','dagger3_native']]]
ax[1,1].plot(range(4),durations,'o-',color='#08988b');ax[1,1].set_xticks(range(4),['Teacher-only','DAgger 1','DAgger 2','DAgger 3']);ax[1,1].set_ylim(0,33);ax[1,1].set_ylabel('Mean survival (s)');ax[1,1].set_title('Student-state labels close the rollout gap')
for i,v in enumerate(durations): ax[1,1].text(i,v+1,f'{v:.2f}',ha='center')
fig.suptitle('Full-connectome Yumi gait transfer: frozen graph, encoder, neuron biases and normalization\nExplicit phase input remains; no natural-gait or biological-topology-superiority claim',fontsize=15)
fig.tight_layout(rect=[0,0.025,1,.94]);fig.savefig(root/'transfer_evidence.png',dpi=140)
