#!/bin/bash
cd /root/autodl-tmp/neuromechfly
for i in $(seq 1 12); do
  echo "=== burst $i $(date +%H:%M:%S) ==="
  .venv/bin/python -m app.ppo_yumi --resume runs/yumi/best.pt --resume-state --freeze-brain --lr 1e-4 --std-override 0.08 --epochs 4 --target-kl 0.02 --value-warmup 6 --max-seconds 170 --eval-every 5
done
