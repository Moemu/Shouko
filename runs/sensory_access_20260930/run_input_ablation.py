"""Disable only the learned vy column and evaluate on development episodes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from app.train_full import file_sha256

root=Path('runs/sensory_access_20260930')
final=json.loads((root/'final_validation/execution.json').read_text())
assert final['completed'] and final['selected']=='s1'
source=Path(final['checkpoint']);assert file_sha256(source)==final['checkpoint_sha256']
out=root/'input_ablation';out.mkdir(exist_ok=False)
saved=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
old=saved['state_dict']['encoder.weight']
changed=old.clone();changed[:,48]=0
assert torch.equal(changed[:,:48],old[:,:48]) and torch.equal(changed[:,49:],old[:,49:])
saved['state_dict']['encoder.weight']=changed
saved['extra']=dict(method='development-only vy-column ablation',source_sha256=final['checkpoint_sha256'])
path=out/'last.pt';torch.save(saved,path)
report=dict(completed=False,source_sha256=final['checkpoint_sha256'],checkpoint_sha256=file_sha256(path),
    changed_tensor='encoder.weight[:, 48]',results={})
(out/'execution.json').write_text(json.dumps(report,indent=2))
for label,extra in [('normal',['--cohorts','3']),('long',['--seconds','120','--trace'])]:
    with (out/(label+'.log')).open('w') as log:
        subprocess.run([sys.executable,'-u','-m','app.evaluate_direction','--checkpoints',str(path),
            '--output',str(out/(label+'.json')),'--seed-base','310201',*extra],stdout=log,stderr=subprocess.STDOUT,
            timeout=2400,check=True,env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'))
    data=json.loads((out/(label+'.json')).read_text());assert data['completed']
    report['results'][label]=data['conditions'][0]['summary']
    (out/'execution.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['results'][label]),flush=True)
report['completed']=True;(out/'execution.json').write_text(json.dumps(report,indent=2))
