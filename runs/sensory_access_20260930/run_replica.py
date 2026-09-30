"""Repeat the matched update on newly sampled data from the same historical actors."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from app.train_full import file_sha256

root=Path('runs/sensory_access_20260930');out=root/'replica';out.mkdir(exist_ok=False)
pair=json.loads((root/'pair_execution.json').read_text())
assert pair['completed'] and (pair['s1_development_passed'] or pair['c0_development_passed'])
source=Path('runs/direction_control_20260929/transfer_main/round4/last.pt')
reference=json.loads(source.with_name('fit.json').read_text())
validation=reference['arguments']['validation']+[str(root/f'validation_{name}.pt') for name in ['normal','positive','negative']]
report=dict(completed=False,jobs=[],results={},started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    selection_rule='Prefer S1 if all development gates pass; otherwise C0 if all pass. No final holdout tuning.',
    limitation='New collection and optimizer sampling conditional on fixed historical actors and the shared round-four source; not independent training from scratch.')
output=out/'execution.json'
def persist():
    p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(output)
def run(name,args,timeout=1800):
    command=[sys.executable,'-u',*args];row=dict(name=name,command=command)
    report['jobs'].append(row);persist();start=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'))
    row.update(returncode=result.returncode,wall_seconds=time.monotonic()-start);persist()
    if result.returncode:raise RuntimeError(name)
torch.set_num_threads(8)
patch=torch.load(root/'history_readouts.pt',map_location='cpu',weights_only=True)
assert patch['base_sha256']==file_sha256(source)
manifest=json.loads((root/'history_readouts.json').read_text())
assert manifest['payload_sha256']==file_sha256(root/'history_readouts.pt')
base=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
report['actors']=[]
for i,actor in enumerate(patch['actors']):
    state=dict(base['state_dict']);state.update(actor['readout'])
    path=out/f'history{i}.pt';torch.save(dict(**actor['metadata'],state_dict=state),path)
    report['actors'].append(dict(path=str(path),checkpoint_sha256=file_sha256(path),
        original_checkpoint_sha256=actor['original_checkpoint_sha256'],frozen_tensors_verified_locally=True))
persist()
training=[];used_ids=set()
for path in reference['arguments']['train']+validation:
    used_ids.update(torch.load(path,map_location='cpu',weights_only=True,mmap=True)['episode_ids'].tolist())
report['collection_seeds']=[610001+i*1000 for i in range(len(reference['arguments']['train']))];persist()
for i,oldpath in enumerate(reference['arguments']['train']):
    recipe=json.loads(Path(oldpath).with_suffix('.json').read_text())['arguments']
    path=out/Path(oldpath).name
    args=['-m','app.imitate_yumi','collect','--teacher',recipe['teacher'],'--beta',str(recipe['beta']),
        '--seconds',str(recipe['seconds']),'--cohorts',str(recipe['cohorts']),
        '--yaws',*[str(v) for v in recipe['yaws']],'--seed-base',str(report['collection_seeds'][i]),'--output',str(path)]
    if recipe['student']:args+=['--student',str(out/f'history{(i-1)//3}.pt')]
    if recipe.get('push_velocity') is not None:args+=['--push-velocity',str(recipe['push_velocity'])]
    elif recipe.get('push'):args+=['--push']
    proposed={report['collection_seeds'][i]+cohort*10+j for cohort in range(recipe['cohorts']) for j in range(9)}
    assert not proposed & used_ids;used_ids.update(proposed)
    run('collect_'+path.stem,args)
    training.append(str(path))
report['training_samples']=sum(torch.load(p,map_location='cpu',weights_only=True,mmap=True)['observations'].shape[0] for p in training)
persist()
fits={}
for label in ['c0','s1']:
    run('fit_'+label,['-m','app.train_sensory_readout','--source',str(source),'--train',*training,
        '--validation',*validation,'--output',str(out/label),'--seed','4031','--calibrate-vy',*(['--sensory'] if label=='s1' else [])])
    fits[label]=json.loads((out/label/'fit.json').read_text());assert fits[label]['completed']
assert fits['c0']['sample_indices_sha256']==fits['s1']['sample_indices_sha256']
report['matched_samples']=True;persist()
for label in ['c0','s1']:
    passed=True
    for name,extra,threshold in [('normal',['--cohorts','3'],24),('long',['--seconds','120','--trace'],9),
        ('yaw',['--yaws','-.2','.2'],16),('push_positive',['--push-velocity','.25'],8),('push_negative',['--push-velocity','-.25'],8)]:
        path=out/f'{label}_{name}.json'
        run(label+'_'+name,['-m','app.evaluate_direction','--checkpoints',str(out/label/'last.pt'),
            '--output',str(path),'--seed-base','310201',*extra])
        data=json.loads(path.read_text());assert data['completed']
        s=data['conditions'][0]['summary'];report['results'][label+'_'+name]=s;persist()
        print(json.dumps(dict(stage=label+'_'+name,**s)),flush=True)
        if s['strict_passes']<threshold or s['falls']>1 or (name=='long' and s['direction_passes']!=9):
            passed=False;break
    report[label+'_development_passed']=passed;persist()
selected='s1' if report['s1_development_passed'] else ('c0' if report['c0_development_passed'] else None)
report.update(completed=True,selected=selected,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
if selected:report['candidate_sha256']=fits[selected]['checkpoint_sha256']
persist()
