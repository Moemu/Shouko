"""Check strict gait semantics and fail-closed checkpoint/interface binding."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from .release_evidence import build_report, matches_checkpoint


def main():
    source=Path(__file__).resolve().parents[1]/'research/releases/v0.2.0/evidence'
    digest='f1a200718ebfec845cbf9c02ba743e937ea94c747fba53e7b12dfd12f8d3e1f9'
    report=build_report(source,digest)
    interface=json.loads((source/'holdout.json').read_text())['conditions'][0]['physics_interface']
    assert (report['successes'],report['attempts'])==(63,63)
    assert [len(report[g]) for g in ['tests','yaw_tests','long_walks','perturbations','controls']]==[27,18,9,9,9]
    assert matches_checkpoint(report,digest,'yumi',interface)
    assert not matches_checkpoint(report,'wrong','yumi',interface)
    assert not matches_checkpoint(report,digest,'g1',interface)
    assert not matches_checkpoint(report,digest,'yumi',dict(interface,action_scale=.99))
    assert not matches_checkpoint(dict(report,complete=False),digest,'yumi',interface)
    with TemporaryDirectory() as directory:
        target=Path(directory)
        for name in ['holdout','yaw_holdout','long_holdout','push_holdout','lesion']:
            data=json.loads((source/(name+'.json')).read_text())
            if name=='holdout':
                data['conditions'][0]['results'][0]['tests'][0]['gait']['feet'][0]['qualifying_swings_per_second']=0
            (target/(name+'.json')).write_text(json.dumps(data))
        assert build_report(target,digest)['successes']==62
        data['conditions'][0]['checkpoint_sha256']='wrong'
        # Modify all modes so lesion selection cannot hide a bad binding.
        for condition in data['conditions']: condition['checkpoint_sha256']='wrong'
        (target/'lesion.json').write_text(json.dumps(data))
        try: build_report(target,digest)
        except ValueError: pass
        else: raise AssertionError('Mismatched evidence accepted')
    print('PASS: strict swing screen, group counts, checkpoint and physics binding')


if __name__=='__main__':
    main()
