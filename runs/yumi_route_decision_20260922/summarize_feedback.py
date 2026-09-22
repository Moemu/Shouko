"""Compare feedback interventions against same-seed intact repeat variation."""
import json
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
data = json.loads((root/'feedback.json').read_text())
assert data['completed'] and len(data['results']) == len(data['schedule'])
groups = {}
for row in data['results']:
    name = 'source' if row['checkpoint'].endswith('yumi_obs50/best.pt') else Path(row['checkpoint']).parent.name
    ev = row['evaluation']
    item = dict(score=ev['score'], mean_speed=ev['mean_speed'], mean_duration=ev['mean_duration'],
                successes=ev['successes'], falls=sum(t['fallen'] for t in ev['tests']),
                lateral_m=mean(t['lateral_m'] for t in ev['tests']))
    groups.setdefault(name, {}).setdefault(row['seed'], {})[row['mode']] = item
report = dict(kind='feedback_ablation_against_repeat_variation', conditions={},
              limitation='Three development seeds with two intact repeats each. Descriptive ranges, not confidence intervals.')
for name, seeds in groups.items():
    rows=[]
    for seed, modes in seeds.items():
        intact1, intact2, ablated = [modes[key] for key in ['intact_1','intact_2','zero_new']]
        metrics={}
        for key in intact1:
            baseline=mean([intact1[key],intact2[key]])
            metrics[key]=dict(intact_mean=baseline, zero_new=ablated[key],
                              zero_minus_intact=ablated[key]-baseline,
                              intact_repeat_abs_difference=abs(intact1[key]-intact2[key]),
                              zero_within_intact_range=min(intact1[key],intact2[key])<=ablated[key]<=max(intact1[key],intact2[key]))
        rows.append(dict(seed=seed,metrics=metrics,modes=modes))
    report['conditions'][name]=dict(per_seed=rows,mean_metrics={key:dict(
        intact=mean(row['metrics'][key]['intact_mean'] for row in rows),
        zero_new=mean(row['metrics'][key]['zero_new'] for row in rows),
        zero_minus_intact=mean(row['metrics'][key]['zero_minus_intact'] for row in rows),
        intact_repeat_abs_difference=mean(row['metrics'][key]['intact_repeat_abs_difference'] for row in rows)) for key in rows[0]['metrics']})
(root/'feedback_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({name:row['mean_metrics'] for name,row in report['conditions'].items()},indent=2))
