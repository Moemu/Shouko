"""Summarize immutable native reports with the predeclared stricter gait screen."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(condition):
    tests = [t for group in condition['results'] for t in group['tests']]
    strict = [t['success'] and all(f['qualifying_swings_per_second'] >= 1 for f in t['gait']['feet'])
              for t in tests]
    per_command = {}
    for speed in sorted({t['target_speed'] for t in tests}):
        rows = [t for t in tests if t['target_speed'] == speed]
        per_command[str(speed)] = dict(episodes=len(rows), successes=sum(t['success'] for t in rows),
            mean_speed=statistics.mean(t['mean_speed'] for t in rows),
            speed_mae=statistics.mean(abs(t['mean_speed']-speed) for t in rows),
            mean_height=statistics.mean(t['gait']['mean_height'] for t in rows))
    recovered = [t['gait']['recovery_seconds'] for t in tests if t['initial_yaw'] != 0
                 and t['gait']['recovery_seconds'] is not None]
    result = dict(checkpoint=condition['checkpoint'], checkpoint_sha256=condition['checkpoint_sha256'],
                  mode=condition['mode'], **condition['summary'], strict_gait_successes=sum(strict),
                  per_command=per_command,
                  recovered=len(recovered), nonzero_yaw_episodes=sum(t['initial_yaw'] != 0 for t in tests),
                  recovery_seconds_median=statistics.median(recovered) if recovered else None)
    surviving_feet = [f for t in tests if t['criteria']['survived'] for f in t['gait']['feet']]
    result['survivors_qualifying_swings_per_second'] = (statistics.mean(
        f['qualifying_swings_per_second'] for f in surviving_feet) if surviving_feet else None)
    feet = [f for t in tests for f in t['gait']['feet']]
    for metric in ['swing_seconds_median', 'swing_clearance_median', 'support_foot_center_speed']:
        values = [f[metric] for f in feet if f[metric] is not None]
        result[metric] = statistics.median(values) if values else None
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    report = dict(strict_gait_definition='Old six criteria AND both feet >=1 qualifying swing/s; '
                                         'each qualifying swing >=0.12s airborne and >=0.025m clearance.',
                  limitation='Engineering screen, not validated human naturalness; cohorts are not independent training seeds.',
                  evidence=[])
    for filename in args.inputs:
        source = Path(filename)
        data = json.loads(source.read_text())
        if not data['completed']:
            raise ValueError('Incomplete report: '+filename)
        entry = dict(source=filename, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     seconds=data['seconds'], seed_base=data['seed_base'], yaws=data['yaws'],
                     conditions=[summarize(c) for c in data['conditions']])
        report['evidence'].append(entry)
        for c in entry['conditions']:
            print(json.dumps({k:c[k] for k in ['checkpoint', 'mode', 'successes', 'strict_gait_successes',
                                               'falls', 'speed_mae', 'lateral_m', 'qualifying_swings_per_second']}))
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
