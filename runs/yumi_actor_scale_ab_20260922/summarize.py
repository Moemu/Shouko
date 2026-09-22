"""Summarize matched development evaluations without a significance claim."""
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
paired = json.loads((root/'paired.json').read_text())
groups = {}
for result in paired['results']:
    name = 'source' if result['checkpoint'].endswith('yumi_obs50/best.pt') else Path(result['checkpoint']).parent.name
    groups.setdefault(name, []).append(result)
summary = dict(kind='paired_development_screen_summary', conditions={}, comparisons={},
               limitation='One training seed per condition; evaluation seeds are not independent training repetitions.')
for name, records in groups.items():
    tests = [test for record in records for test in record['evaluation']['tests']]
    item = dict(checkpoint_sha256=records[0]['checkpoint_sha256'], attempts=len(tests),
                successes=sum(t['success'] for t in tests), falls=sum(t['fallen'] for t in tests),
                score=statistics.mean(r['evaluation']['score'] for r in records),
                mean_duration=statistics.mean(t['seconds'] for t in tests),
                mean_speed=statistics.mean(t['mean_speed'] for t in tests),
                speed_mae=statistics.mean(abs(t['mean_speed']-t['target_speed']) for t in tests),
                mean_lateral_m=statistics.mean(t['lateral_m'] for t in tests),
                mean_height=statistics.mean(r['evaluation']['mean_height'] for r in records),
                per_seed=[dict(seed=r['seed'], score=r['evaluation']['score'],
                               successes=r['evaluation']['successes'],
                               falls=sum(t['fallen'] for t in r['evaluation']['tests'])) for r in records],
                tiers={})
    for target in sorted(set(round(t['target_speed'], 2) for t in tests)):
        tier = [t for t in tests if round(t['target_speed'], 2) == target]
        item['tiers'][str(target)] = dict(attempts=len(tier), successes=sum(t['success'] for t in tier),
                                         falls=sum(t['fallen'] for t in tier),
                                         mean_speed=statistics.mean(t['mean_speed'] for t in tier),
                                         mean_lateral_m=statistics.mean(t['lateral_m'] for t in tier))
    status_path = root/name/'training.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        history = status['history']
        actor_rows = [h for h in history if not h['value_warmup']]
        item['training'] = dict(iterations=status['iteration'], actor_rounds=len(actor_rows),
                                optimizer_steps=sum(h['policy_optimizer_steps'] for h in history),
                                kl_stops=sum(h['kl_stop'] for h in history),
                                mean_actor_kl=statistics.mean(h['approx_kl'] for h in actor_rows),
                                initial_value_loss=history[0]['v_loss'], final_value_loss=history[-1]['v_loss'],
                                mean_actor_round_seconds=statistics.mean(h['rollout_seconds']+h['update_seconds'] for h in actor_rows),
                                peak_vram_gib=status['peak_vram_gib'], elapsed=status['elapsed'])
    summary['conditions'][name] = item
for left, right in [('source', 'control'), ('source', 'scaled'), ('control', 'scaled')]:
    rows = {name: {(r['seed'], t['environment']): t for r in groups[name] for t in r['evaluation']['tests']}
            for name in (left, right)}
    assert rows[left].keys() == rows[right].keys()
    pairs = [(rows[left][key], rows[right][key]) for key in rows[left]]
    summary['comparisons'][f'{right}_minus_{left}'] = dict(
        score=summary['conditions'][right]['score']-summary['conditions'][left]['score'],
        gained_success=sum(not a['success'] and b['success'] for a,b in pairs),
        lost_success=sum(a['success'] and not b['success'] for a,b in pairs),
        new_falls=sum(not a['fallen'] and b['fallen'] for a,b in pairs),
        prevented_falls=sum(a['fallen'] and not b['fallen'] for a,b in pairs))
(root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
