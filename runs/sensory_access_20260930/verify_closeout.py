"""Check local checkpoint boundaries, reserved seeds and final trajectory evidence."""
import json
from pathlib import Path
import numpy as np
import torch
from app.train_full import file_sha256
from app.train_sensory_readout import check_boundary

torch.set_num_threads(8)
root=Path('runs/sensory_access_20260930')
source=Path('runs/direction_control_20260929/transfer_main/round4/last.pt')
before=torch.load(source,map_location='cpu',weights_only=True,mmap=True)['state_dict']
fits=[json.loads(p.read_text()) for p in [root/'s1/fit.json',root/'replica/s1/fit.json',root/'replica/dagger/round1/fit.json']]
paths=sorted({p for f in fits for p in f['arguments']['train']+f['arguments']['validation']})
training_ids=set()
for path in paths:
    expected={f['data_sha256'][path] for f in fits if path in f['data_sha256']}
    assert len(expected)==1 and file_sha256(path) in expected
    data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    training_ids.update(data['episode_ids'].tolist())
report=dict(completed=False,source_checkpoint_sha256=file_sha256(source),branches={},
    verified_data_files=len(paths),same_update_replication_passed=False,
    tests=['CPU selected-column gradient/initialization/folding/boundary',
           'CPU cache integrity and invalidation','CPU direction metrics',
           'Full-graph CUDA gradient/initialization/parameter boundary',
           'Native MuJoCo development and final holdouts'])
for label,folder in [('main',root/'final_validation'),('replica_extra',root/'replica/dagger/final_validation')]:
    execution=json.loads((folder/'execution.json').read_text())
    assert execution['completed'] and execution['passed']
    checkpoint=Path(execution['checkpoint'])
    assert file_sha256(checkpoint)==execution['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)['state_dict']
    changes=check_boundary(before,state,True)
    groups={};maxima=[]
    for name,count in [('normal',27),('yaw',18),('long',9),('push_positive',9),('push_negative',9)]:
        evidence=json.loads((folder/(name+'.json')).read_text());assert evidence['completed']
        condition=evidence['conditions'][0]
        assert condition['checkpoint_sha256']==execution['checkpoint_sha256']
        tests=[t for row in condition['results'] for t in row['tests']]
        assert len(tests)==count and all(t['strict_success'] and not t['fallen'] for t in tests)
        assert not {t['seed'] for t in tests} & training_ids
        if name=='long':
            for row in condition['results']:
                trace=folder/row['trace'];assert file_sha256(trace)==row['trace_sha256']
                arrays=np.load(trace)
                for i,test in enumerate(row['tests']):
                    maximum=float(np.abs(arrays[f'episode{i}_qpos'][:,1]).max())
                    assert abs(maximum-test['direction']['maximum_lateral_m'])<1e-10
                    assert maximum<=2 and test['direction_success']
                    maxima.append(maximum)
        groups[name]=dict(episodes=count,strict_passes=sum(t['strict_success'] for t in tests),falls=0,
                         mean_absolute_endpoint_lateral_m=condition['summary']['lateral_m'])
    fit=json.loads(checkpoint.with_name('fit.json').read_text())
    report['branches'][label]=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
        groups=groups,maximum_long_lateral_m=max(maxima),allowed_parameter_changes=changes,
        cumulative_updates=500 if label=='main' else 1000,
        final_training_samples=sum(torch.load(p,map_location='cpu',weights_only=True,mmap=True)['observations'].shape[0]
            for p in fit['arguments']['train']))
original_replica=json.loads((root/'replica/execution.json').read_text())
assert original_replica['completed'] and not original_replica['c0_development_passed'] and not original_replica['s1_development_passed']
dagger=json.loads((root/'replica/dagger/execution.json').read_text())
assert dagger['completed'] and dagger['rounds']==1
report.update(completed=True,extra_dagger_rounds=1,source_sha256=file_sha256(__file__),
    interpretation='Two fixed-source branches pass reserved holdouts; the second required extra convergence. No equal-update or biological topology superiority claim.')
(root/'closeout.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
