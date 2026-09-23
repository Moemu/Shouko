import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

torch.set_num_threads(4)
root=Path('runs/connectome_transfer_balance_20260923');root.mkdir(parents=True,exist_ok=True)
prior=Path('runs/prior');push=Path('runs/connectome_transfer_push_20260923');output=root/'execution.json'
if output.exists():raise FileExistsError(output)
report=dict(completed=False,jobs=[],development=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def persist():
    t=output.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2));t.replace(output)
def make(label,left,right,alpha):
    a=torch.load(left,map_location='cpu',weights_only=True);b=torch.load(right,map_location='cpu',weights_only=True)
    for n in a['state_dict']:
        if n.startswith('readout.'):
            a['state_dict'][n]=a['state_dict'][n]*(1-alpha)+b['state_dict'][n]*alpha
        else:assert torch.equal(a['state_dict'][n],b['state_dict'][n]),n
    a['extra']=dict(method='readout parameter interpolation',alpha=alpha,left_sha256=sha(left),right_sha256=sha(right))
    folder=root/label;folder.mkdir(exist_ok=False);path=folder/'last.pt';torch.save(a,path)
    (folder/'construction.json').write_text(json.dumps(dict(**a['extra'],checkpoint_sha256=sha(path)),indent=2))
    return path
def run(name,args,timeout=1200):
    row=dict(name=name,command=[sys.executable,'-u',*args]);report['jobs'].append(row);persist();start=time.monotonic()
    with (root/(name+'.log')).open('w') as handle:
        r=subprocess.run(row['command'],stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),timeout=timeout)
    row.update(returncode=r.returncode,wall_seconds=time.monotonic()-start);persist()
    if r.returncode:raise RuntimeError(name)
def evaluate(label,checkpoint,seed,extra=()):
    path=root/(label+'.json');run(label,['-m','app.evaluate_locomotion','--checkpoints',str(checkpoint),
        '--seed-base',str(seed),*extra,'--output',str(path)])
    c=json.loads(path.read_text())['conditions'][0];ts=[t for g in c['results'] for t in g['tests']]
    return dict(strict=sum(t['success'] and all(f['qualifying_swings_per_second']>=1 for f in t['gait']['feet']) for t in ts),falls=c['summary']['falls'],summary=c['summary'])
def development(label,checkpoint,seeds):
    normal=evaluate(label+'_normal',checkpoint,seeds[0]);pushed=evaluate(label+'_push',checkpoint,seeds[1],['--push','--cohorts','1'])
    long=evaluate(label+'_long',checkpoint,seeds[2],['--seconds','120','--cohorts','1'])
    passed=normal['strict']>=24 and pushed['strict']>=8 and long['strict']>=8 and max(normal['falls'],pushed['falls'],long['falls'])<=1
    return dict(normal=normal,push=pushed,long=long,passed=passed)
persist();deadline=time.monotonic()+2400
while not (push/'candidate_trace.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Push stage incomplete')
    time.sleep(10)
assert json.loads((push/'execution.json').read_text())['completed']
candidate=None
for alpha in [.5,.25,.75]:
    label='alpha'+str(alpha).replace('.','_');path=make(label,prior/'dagger3/last.pt',push/'push2/last.pt',alpha)
    gate=development(label,path,[103001,114001,115001]);report['development'].append(dict(alpha=alpha,**gate));persist()
    if gate['passed']:
        candidate=path;report.update(candidate=str(path),alpha=alpha,candidate_sha256=sha(path));persist();break
if candidate:
    replica=make('replica',prior/'replica3/last.pt',push/'replica_push2/last.pt',report['alpha'])
    report['replication']=development('replica',replica,[203001,214001,215001]);persist()
    for label,extra in [('holdout',['--seed-base','130001']),
        ('yaw_holdout',['--seed-base','131001','--cohorts','1','--yaws','-.2','.2']),
        ('long_holdout',['--seed-base','132001','--cohorts','1','--seconds','120']),
        ('push_holdout',['--seed-base','133001','--cohorts','1','--push']),
        ('lesion',['--seed-base','103001','--cohorts','1','--lesion','--repeat']),
        ('native_csr',['--seed-base','103001','--sparse-backend','native'])]:
        run(label,['-m','app.evaluate_locomotion','--checkpoints',str(candidate),*extra,'--output',str(root/(label+'.json'))])
report.update(completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist()
