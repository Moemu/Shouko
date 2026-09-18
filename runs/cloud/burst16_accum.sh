#!/bin/bash
cd /root/autodl-tmp/neuromechfly
for i in $(seq 1 8); do
  echo "=== clock1.0 accum burst $i $(date +%H:%M:%S) ===" >> runs/yumi/ppo16_clock10_accum.log
  .venv/bin/python -m app.ppo_yumi --resume runs/yumi/best.pt --resume-state --freeze-brain --lr 1e-4 --std-override 0.08 --epochs 4 --target-kl 0.02 --value-warmup 6 --max-seconds 170 --eval-every 5 >> runs/yumi/ppo16_clock10_accum.log 2>&1
done
