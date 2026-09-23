"""Read immutable fit and native reports without changing historical success labels."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

def native(path):
    d=json.loads(path.read_text())
    if not d['completed']:
        raise ValueError('Incomplete: '+str(path))
    results=[]
    for c in d['conditions']:
        tests=[t for g in c['results'] for t in g['tests']]
        strict=[t['success'] and all(f['qualifying_swings_per_second']>=1 for f in t['gait']['feet']) for t in tests]
        yaw=[t for t in tests if t['initial_yaw']!=0]
        results.append(dict(checkpoint_sha256=c['checkpoint_sha256'], mode=c['mode'],
            **c['summary'], strict=sum(strict), mean_seconds=statistics.mean(t['seconds'] for t in tests),
            yaw_recovered=sum(t['gait']['recovery_seconds'] is not None for t in yaw), yaw_attempts=len(yaw),
            mean_lateral_fraction_of_command_distance=statistics.mean(t['lateral_m']/(d['seconds']*t['target_speed']) for t in tests),
            by_speed={str(v):dict(episodes=sum(t['target_speed']==v for t in tests),
                strict=sum(ok for t,ok in zip(tests,strict) if t['target_speed']==v),
                falls=sum(t['fallen'] for t in tests if t['target_speed']==v)) for v in [.35,.5,.65]}))
    return dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), conditions=results)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('inputs',nargs='+')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    result=dict(evidence=[native(Path(x)) for x in a.inputs],
        limitation='Explicit phase remains. Episode cohorts are not independent training seeds. '
        'Strict gait is an engineering screen, not naturalness or topology superiority.')
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for e in result['evidence']:
        print(e['source'],[(c['mode'],c['strict'],c['falls'],c['mean_seconds']) for c in e['conditions']])
