"""Summarize complete native diagnostics without pooling different backends."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read(name):
    result = json.loads((ROOT / name).read_text(encoding='utf-8'))
    assert result['completed'], name
    return result


def summarize_rows(rows):
    tests = [test for row in rows for test in row['tests']]
    return dict(episodes=len(tests), score=mean(row['score'] for row in rows),
                successes=sum(test['success'] for test in tests),
                falls=sum(test['fallen'] for test in tests),
                lateral_m=mean(test['lateral_m'] for test in tests),
                criteria_passes={key: sum(test['criteria'][key] for test in tests)
                                 for key in tests[0]['criteria']})


def main():
    feedback = read('native_feedback27.json')
    report = dict(native27={name: summarize_rows([row for row in feedback['conditions'] if row['name'] == name])
                            for name in dict.fromkeys(row['name'] for row in feedback['conditions'])})
    commands = read('command_factorial.json')
    report['command_factorial'] = []
    report['paired_turn_response'] = []
    for row in commands['conditions']:
        tests = row['tests']
        report['command_factorial'].append(dict(
            speed=row['forward_command'], turn=row['turn_command'],
            **{key: mean(test[key] for test in tests) for key in
               ['mean_world_vx', 'mean_body_vx', 'mean_body_vy', 'mean_yaw_rate']},
            positive_yaw_episodes=sum(test['mean_yaw_rate'] > 0 for test in tests),
            falls=sum(test['fallen'] for test in tests)))
    for speed in commands['speed_grid']:
        rows = [row for row in commands['conditions'] if row['forward_command'] == speed]
        low = next(row for row in rows if row['turn_command'] == -0.2)['tests']
        high = next(row for row in rows if row['turn_command'] == 0.2)['tests']
        assert [test['seed'] for test in low] == [test['seed'] for test in high]
        delta = [b['mean_yaw_rate'] - a['mean_yaw_rate'] for a, b in zip(low, high)]
        report['paired_turn_response'].append(dict(speed=speed, mean_delta=mean(delta),
                                                   positive_deltas=sum(x > 0 for x in delta), deltas=delta))
    rows = [row for row in commands['conditions'] if row['turn_command'] == 0]
    low = next(row for row in rows if row['forward_command'] == 0.35)['tests']
    high = next(row for row in rows if row['forward_command'] == 0.65)['tests']
    assert [test['seed'] for test in low] == [test['seed'] for test in high]
    delta = [b['mean_body_vx'] - a['mean_body_vx'] for a, b in zip(low, high)]
    report['paired_speed_response'] = dict(mean_delta=mean(delta), positive_deltas=sum(x > 0 for x in delta), deltas=delta)
    if (ROOT / 'heading_limits.json').exists():
        heading = read('heading_limits.json')
        report['heading_limits'] = {str(limit): summarize_rows([row for row in heading['conditions'] if row['yaw_limit'] == limit])
                                    for limit in heading['limits']}
    (ROOT / 'native_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
