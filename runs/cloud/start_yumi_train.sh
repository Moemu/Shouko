#!/bin/bash
# Start persistent Yumi PPO training (survives SSH disconnect). 8h budget.
cd /root/autodl-tmp/neuromechfly
mkdir -p runs/yumi
nohup .venv/bin/python -u -m app.ppo_yumi --max-seconds 28800 --eval-every 5 \
  --resume runs/yumi/last.pt --resume-state > runs/yumi/ppo5.log 2>&1 &
echo $! > runs/yumi/train.pid
echo "started pid $(cat runs/yumi/train.pid)"
