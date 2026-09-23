"""Install an extracted, checksum-verified v0.2.0 package into this studio."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from app.release_evidence import build_report

PRIMARY='f1a200718ebfec845cbf9c02ba743e937ea94c747fba53e7b12dfd12f8d3e1f9'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def install(package: Path):
    package=package.resolve()
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path=(package/row['path']).resolve()
        if not path.is_relative_to(package) or not path.is_file() or sha(path)!=row['sha256']:
            raise ValueError('Invalid package member: '+row['path'])
    primary=package/'models/primary.pt'
    if sha(primary)!=PRIMARY:
        raise ValueError('Unexpected primary checkpoint')
    report=build_report(package/'evidence/primary',PRIMARY)
    if report['successes']!=63 or report['attempts']!=63:
        raise ValueError('Incomplete v0.2.0 acceptance')
    for row in manifest['files']:
        if row['path'].startswith('app/'):
            local=ROOT/row['path']
            if not local.is_file() or sha(local)!=row['sha256']:
                raise ValueError('Checkout does not match package runtime: '+row['path'])
    for name in ['full_graph.npz','full_graph.json']:
        target=ROOT/'data'/name
        if target.exists() and sha(target)!=sha(package/'data'/name):
            raise ValueError('Existing graph differs; preserve it before installation')
    source=ROOT/'runs/yumi/upright-source.pt'
    if source.exists() and sha(source)!=sha(package/'models/upright-source.pt'):
        raise ValueError('Existing upright source differs')
    runs=ROOT/'runs/yumi'
    runs.mkdir(parents=True,exist_ok=True)
    current=runs/'best.pt'
    if current.exists() and sha(current)!=PRIMARY:
        backup=runs/'archive'/('before-v0.2.0-'+sha(current)[:12])
        backup.mkdir(parents=True,exist_ok=True)
        for name in ['best.pt','last.pt','ppo_state_best.pt','ppo_state.pt','training.json','evaluation.json','release_evaluation.json']:
            path=runs/name
            if path.exists():
                target=backup/name
                if target.exists() and sha(target)!=sha(path):
                    raise FileExistsError('Backup differs: '+name)
                if not target.exists(): shutil.copy2(path,target)
                if sha(target)!=sha(path): raise ValueError('Backup failed: '+name)
        for name in ['last.pt','ppo_state_best.pt','ppo_state.pt']:
            (runs/name).unlink(missing_ok=True)
    for name in ['full_graph.npz','full_graph.json']:
        target=ROOT/'data'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=sha(package/'data'/name):
            raise ValueError('Existing graph differs; preserve it before installation')
        if not target.exists(): shutil.copy2(package/'data'/name,target)
    shutil.copy2(primary,current)
    source=runs/'upright-source.pt'
    if source.exists() and sha(source)!=sha(package/'models/upright-source.pt'):
        raise ValueError('Existing upright source differs')
    if not source.exists(): shutil.copy2(package/'models/upright-source.pt',source)
    (runs/'release_evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    training=dict(phase='stopped',method='frozen-core ridge readout',checkpoint_sha256=PRIMARY,
                  history=[],note='Historical PPO records are archived; no matching PPO optimizer state for this endpoint.')
    (runs/'training.json').write_text(json.dumps(training,indent=2)+'\n',encoding='utf-8')
    (runs/'evaluation.json').write_text(json.dumps(dict(checkpoint_sha256=PRIMARY,kind='release_reference',
        independent_report='release_evaluation.json'))+'\n',encoding='utf-8')
    print(json.dumps(dict(installed=PRIMARY,strict_successes=report['successes'],training_started=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    install(parser.parse_args().package)
