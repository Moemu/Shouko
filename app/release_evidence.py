"""Convert frozen locomotion evidence into the studio's grouped report schema."""
import hashlib
import json
import math
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


def body_xml_hashes(root: Path) -> dict:
    return {f'app/yumi_description/{name}': hashlib.sha256(
        (root/'app/yumi_description'/name).read_bytes()).hexdigest()
        for name in ['scene.xml', 'yumi.xml']}


def build_direction_report(directory: Path, checkpoint_sha256: str) -> dict:
    """Validate the five frozen direction cohorts; recompute strict/corridor passes."""
    groups = [('normal', 'tests', 30, 3, [0.0], 0),
              ('yaw', 'yaw_tests', 30, 1, [-.2, .2], 0),
              ('long', 'long_walks', 120, 1, [0.0], 0),
              ('push_positive', 'perturbations', 30, 1, [0.0], .25),
              ('push_negative', 'perturbations', 30, 1, [0.0], -.25)]
    report = dict(kind='held_out_locomotion', robot='yumi', complete=True,
                  checkpoint_sha256=checkpoint_sha256, protocol='strict_gait_v2_corridor2',
                  corridor_m=2, sources=[], controls=[])
    bindings = None
    for name, group, seconds, cohorts, yaws, push in groups:
        raw = (directory/(name+'.json')).read_bytes()
        data = json.loads(raw)
        if data.get('completed') is not True or data.get('kind') != 'native_direction_v1' or len(data['conditions']) != 1:
            raise ValueError('Incomplete or ambiguous direction evidence: '+name)
        args = data['arguments']
        expected = dict(seconds=seconds, cohorts=cohorts, yaws=yaws, push_velocity=push,
                        push_time=10, vy_command=0, yaw_feedback_gain=1.4,
                        hip_roll_offset=0, ankle_roll_offset=0, corridor=2, sparse_backend='ordered')
        if any(args.get(key) != value for key, value in expected.items()):
            raise ValueError('Direction protocol mismatch: '+name)
        condition = data['conditions'][0]
        if condition['checkpoint_sha256'] != checkpoint_sha256 or condition['policy_kind'] != 'connectome':
            raise ValueError('Direction checkpoint mismatch: '+name)
        current = (condition['physics_interface'], data['body_xml_sha256'], data['source_sha256'])
        if not all(current) or (bindings is not None and current != bindings):
            raise ValueError('Direction runtime binding mismatch: '+name)
        bindings = current
        results = condition['results']
        if len(results) != cohorts*len(yaws):
            raise ValueError('Missing direction cohorts: '+name)
        seen = set()
        rows = report.setdefault(group, [])
        for cohort in results:
            identity = (cohort['cohort'], cohort['initial_yaw'])
            if identity in seen or identity[0] not in range(cohorts) or identity[1] not in yaws or len(cohort['tests']) != 9:
                raise ValueError('Invalid direction cohort: '+name)
            seen.add(identity)
            for index, original in enumerate(cohort['tests']):
                if (original['seed'] != args['seed_base']+identity[0]*10+index
                        or original['target_speed'] != [.35, .5, .65][index % 3]
                        or original['initial_yaw'] != identity[1]):
                    raise ValueError('Invalid direction trial: '+name)
                maximum = original['direction']['maximum_lateral_m']
                if not math.isfinite(maximum) or maximum < 0:
                    raise ValueError('Invalid trajectory extent: '+name)
                criteria = {key: original['criteria'][key] for key in
                            ['survived', 'speed', 'feet', 'alternation', 'upright', 'direction']}
                if any(type(value) is not bool for value in criteria.values()):
                    raise ValueError('Invalid gait criteria: '+name)
                criteria['both_feet'] = criteria.pop('feet')
                feet = original['gait']['feet']
                criteria['swing'] = len(feet) == 2 and all(
                    math.isfinite(f['qualifying_swings_per_second']) and f['qualifying_swings_per_second'] >= 1 for f in feet)
                if group == 'long_walks':
                    criteria['corridor'] = maximum <= 2
                row = dict(original, criteria=criteria, success=all(criteria.values()),
                           contact_fraction=[f['contact_fraction'] for f in feet])
                if group == 'yaw_tests':
                    row['condition'] = f"yaw {identity[1]:+.2f} rad"
                elif group == 'perturbations':
                    row['condition'] = f'push {push:+.2f} m/s at 10 s'
                rows.append(row)
        report['sources'].append(dict(file=name+'.json', sha256=hashlib.sha256(raw).hexdigest(), sparse_backend=args['sparse_backend']))
    walks = sum([report[g] for g in ['tests', 'yaw_tests', 'long_walks', 'perturbations']], [])
    report.update(successes=sum(r['success'] for r in walks), attempts=len(walks),
                  physics_interface_sha256=interface_digest(bindings[0]), body_xml_sha256=bindings[1], source_sha256=bindings[2],
                  long_lateral_m=sum(r['lateral_m'] for r in report['long_walks'])/9,
                  long_maximum_lateral_m=max(r['direction']['maximum_lateral_m'] for r in report['long_walks']),
                  long_corridor_passes=sum(r['criteria']['corridor'] for r in report['long_walks']),
                  yaw_recovered=sum(r['gait']['recovery_seconds'] is not None for r in report['yaw_tests']))
    return report


def matches_checkpoint(record: dict, checkpoint_sha256: str | None, robot: str, interface: dict,
                       body_hashes: dict | None = None) -> bool:
    protocol = record.get('protocol')
    valid_protocol = protocol == 'strict_gait_v1' or (
        protocol == 'strict_gait_v2_corridor2' and bool(body_hashes)
        and record.get('body_xml_sha256') == body_hashes)
    return bool(checkpoint_sha256 and record.get('complete') is True
                and record.get('kind') == 'held_out_locomotion' and valid_protocol
                and record.get('checkpoint_sha256') == checkpoint_sha256 and record.get('robot') == robot
                and record.get('physics_interface_sha256') == interface_digest(interface))
