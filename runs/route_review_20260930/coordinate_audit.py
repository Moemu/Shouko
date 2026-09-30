"""Compare displacement rate and steady world/body velocities on the same traces."""
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path('runs/direction_control_20260929')
paths=[root/'transfer_main/ridge1e5_long_m0_c0_yaw0.0.npz',root/'teacher_control3029_long_m0_c0_yaw0.0.npz']
results=[]
for path in paths:
    with path.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    data=np.load(path); rows=[]
    for i in range(9):
        t=data[f'episode{i}_time']; q=data[f'episode{i}_qpos']; v=data[f'episode{i}_qvel']; yaw=data[f'episode{i}_yaw']
        mask=t>=10
        vy_body=-np.sin(yaw)*v[:,0]+np.cos(yaw)*v[:,1]
        rows.append(dict(episode=i,target_speed=[.35,.5,.65][i%3],
            net_displacement_rate_mmps=float((q[-1,1]-q[0,1])/(t[-1]-t[0])*1000),
            steady_world_vy_mmps=float(v[mask,1].mean()*1000),steady_body_vy_mmps=float(vy_body[mask].mean()*1000),
            maximum_abs_y_m=float(np.abs(q[:,1]).max())))
    groups={str(speed):{key:float(np.mean([row[key] for row in rows if row['target_speed']==speed]))
        for key in ['net_displacement_rate_mmps','steady_world_vy_mmps','steady_body_vy_mmps','maximum_abs_y_m']}
        for speed in [.35,.5,.65]}
    results.append(dict(source=str(path),source_sha256=digest,episodes=rows,means_by_speed=groups))
report=dict(steady_window='t >= 10 s; arithmetic sample mean of recorded qvel, not finite-difference midpoint decomposition',
    paired_initial_states=False,limitation='These two files use different development seeds; this audit reconciles metric definitions and does not estimate a paired causal effect.',results=results)
Path('runs/route_review_20260930/coordinate_audit.json').write_text(json.dumps(report,indent=2))
for row in results:
    print(row['source'],json.dumps(row['means_by_speed']))
