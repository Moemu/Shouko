"""Convert frozen locomotion evidence into the studio's grouped report schema."""
import hashlib
import json
from pathlib import Path


def interface_digest(interface):
    return hashlib.sha256(json.dumps(interface or {}, sort_keys=True).encode()).hexdigest()


def build_report(directory: Path, checkpoint_sha256: str) -> dict:
    groups = {'holdout': 'tests', 'yaw_holdout': 'yaw_tests',
              'long_holdout': 'long_walks', 'push_holdout': 'perturbations', 'lesion': 'controls'}
    report = dict(kind='held_out_locomotion', robot='yumi', complete=True,
                  checkpoint_sha256=checkpoint_sha256, protocol='strict_gait_v1', sources=[])
    interface = None
    for filename, group in groups.items():
        path = directory/(filename+'.json')
        raw = path.read_bytes()
        data = json.loads(raw)
        if not data.get('completed'):
            raise ValueError('Incomplete evidence: '+filename)
        mode = 'lesion' if group == 'controls' else 'normal'
        matches = [c for c in data['conditions'] if c['mode'] == mode]
        if len(matches) != 1:
            raise ValueError('Ambiguous evidence: '+filename)
        condition = matches[0]
        if condition['checkpoint_sha256'] != checkpoint_sha256:
            raise ValueError('Evidence checkpoint mismatch: '+filename)
        if interface is None:
            interface = condition['physics_interface']
        if condition['physics_interface'] != interface:
            raise ValueError('Evidence physics mismatch: '+filename)
        rows = []
        for cohort in condition['results']:
            for original in cohort['tests']:
                row = dict(original)
                criteria = dict(row['criteria'])
                criteria['both_feet'] = criteria.pop('feet')
                feet = row['gait']['feet']
                criteria['swing'] = len(feet) == 2 and all(f['qualifying_swings_per_second'] >= 1 for f in feet)
                row.update(criteria=criteria, success=all(criteria.values()),
                           contact_fraction=[f['contact_fraction'] for f in feet])
                if group == 'yaw_tests':
                    row['condition'] = f"yaw {row['initial_yaw']:+.2f} rad"
                rows.append(row)
        report[group] = rows
        report['sources'].append(dict(file=filename+'.json', sha256=hashlib.sha256(raw).hexdigest(),
                                      sparse_backend=data['sparse_backend']))
    walks = sum([report[g] for g in ['tests', 'yaw_tests', 'long_walks', 'perturbations']], [])
    report.update(successes=sum(r['success'] for r in walks), attempts=len(walks),
                  physics_interface_sha256=interface_digest(interface),
                  long_lateral_m=sum(r['lateral_m'] for r in report['long_walks'])/len(report['long_walks']),
                  yaw_recovered=sum(r['gait']['recovery_seconds'] is not None for r in report['yaw_tests']))
    return report


def matches_checkpoint(record: dict, checkpoint_sha256: str | None, robot: str, interface: dict) -> bool:
    return bool(checkpoint_sha256 and record.get('complete') is True
                and record.get('kind') == 'held_out_locomotion' and record.get('protocol') == 'strict_gait_v1'
                and record.get('checkpoint_sha256') == checkpoint_sha256 and record.get('robot') == robot
                and record.get('physics_interface_sha256') == interface_digest(interface))
