# -*- coding: utf-8 -*-
"""验证 best.pt (2abe0093, score 0.869) 的真实性。

对 best.pt 和 best_tall_obs50.pt (780f4f4f) 各跑 3 次独立评估，
量化同检查点重复评估的散布，判断 0.869 是真实水平还是瞬态幸运。

用法（云端）：
    PYTHONPATH=. python app/verify_best.py
"""
import json
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from app.full_brain import ConnectomePolicy
from app.gpu_body import GPUHumanoid, observation_interface
from app.train_full import evaluate

RUNS = ROOT / 'runs' / 'yumi_obs50'


def load_policy(path, obs_size, interface=None):
    policy = ConnectomePolicy(observation_size=obs_size).cuda()
    policy.load(path, interface=interface)
    return policy


def main():
    # 与 ppo_yumi.py 相同的 interface 加载逻辑
    interface_yaml = ROOT / 'runs' / 'local' / 'yumi_tall.yaml'
    interface = observation_interface(
        yaml.safe_load(interface_yaml.read_text(encoding='utf-8')))
    env = GPUHumanoid(32, robot='yumi', interface=interface, observation_size=50)
    results = {}

    for name, ckpt_name, obs_size in [
        ('best', 'best.pt', 50),
        ('source', 'best_tall_obs50.pt', 50),
    ]:
        path = RUNS / ckpt_name
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        policy = load_policy(path, obs_size, interface=env.interface)
        scores = []
        for trial in range(3):
            ev = evaluate(policy, env, seconds=30)
            scores.append({
                'successes': ev['successes'],
                'score': round(ev['score'], 4),
                'mean_speed': round(ev['mean_speed'], 4),
                'mean_duration': round(ev['mean_duration'], 2),
            })
            print(f"  {name} trial{trial}: score={scores[-1]['score']} successes={scores[-1]['successes']}")
        results[name] = scores

    out = RUNS / 'verify_best.json'
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nwritten: {out}")

    # 汇总
    for name, trials in results.items():
        s = [t['score'] for t in trials]
        print(f"{name}: mean={sum(s)/len(s):.4f} min={min(s):.4f} max={max(s):.4f} spread={max(s)-min(s):.4f}")


if __name__ == '__main__':
    main()
