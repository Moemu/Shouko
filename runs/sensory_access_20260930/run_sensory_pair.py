"""Run the preregistered matched pair only after the weighted control fails."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/sensory_access_20260930')
initial=json.loads((root/'initial_execution.json').read_text())
assert initial['completed'] and not initial['weighted_development_passed']
source=Path('runs/direction_control_20260929/transfer_main/round4/last.pt')
reference=json.loads(source.with_name('fit.json').read_text())
validation=reference['arguments']['validation']+[str(root/f'validation_{name}.pt') for name in ['normal','positive','negative']]
output=root/'pair_execution.json'
if output.exists():raise FileExistsError(output)
report=dict(completed=False,jobs=[],results={},started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def persist():
    p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(output)
def run(name,args,timeout=7200):
    command=[sys.executable,'-u',*args]
    row=dict(name=name,command=command);report['jobs'].append(row);persist();start=time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'))
    row.update(returncode=result.returncode,wall_seconds=time.monotonic()-start);persist()
    if result.returncode:raise RuntimeError(name)
def fit(label,updates,sensory,valid):
    run(label,['-m','app.train_sensory_readout','--source',str(source),
        '--train',*reference['arguments']['train'],'--validation',*valid,
        '--output',str(root/label),'--updates',str(updates),'--calibrate-vy',*(['--sensory'] if sensory else [])])
    data=json.loads((root/label/'fit.json').read_text());assert data['completed']
    report['results'][label]=dict(checkpoint_sha256=data['checkpoint_sha256'],
        trainable_parameters=data['trainable_parameters'],parameter_changes=data['parameter_changes'],
        final_input_response=data['final_input_response'],peak_memory_bytes=data['peak_memory_bytes'])
    persist();return data
def evaluate(label,checkpoint,extra,threshold):
    path=root/(label+'.json')
    run(label,['-m','app.evaluate_direction','--checkpoints',str(checkpoint),'--output',str(path),
        '--seed-base','310201',*extra],1500)
    data=json.loads(path.read_text());assert data['completed']
    s=data['conditions'][0]['summary'];report['results'][label]=s;persist();print(json.dumps(dict(stage=label,**s)),flush=True)
    return s['strict_passes']>=threshold and s['falls']<=1 and (not label.endswith('_long') or s['direction_passes']==9)
persist()
smoke=fit('sensory_smoke_v2',2,True,reference['arguments']['validation'])
assert smoke['gradient_update_1']['encoder.column']>0 and smoke['parameter_changes']['encoder.weight']>0
assert max(smoke['final_input_response'].values())>0
c0=fit('c0',500,False,validation)
s1=fit('s1',500,True,validation)
assert c0['sample_indices_sha256']==s1['sample_indices_sha256']
assert c0['vy_observation_std']==s1['vy_observation_std']
report['matched_samples']=True;persist()
for label in ['c0','s1']:
    checkpoint=root/label/'last.pt';passed=True
    for name,extra,threshold in [('normal',['--cohorts','3'],24),('long',['--seconds','120','--trace'],9),
        ('yaw',['--yaws','-.2','.2'],16),('push_positive',['--push-velocity','.25'],8),('push_negative',['--push-velocity','-.25'],8)]:
        if not evaluate(label+'_'+name,checkpoint,extra,threshold):
            passed=False;break
    report[label+'_development_passed']=passed;persist()
report.update(completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist()
