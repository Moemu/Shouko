#!/bin/bash
# Yumi PPO cloud bootstrap: run once on a fresh AutoDL instance (PyTorch base image).
set -e
cd /root/autodl-tmp/neuromechfly
tar -xzf yumi_code.tgz
tar -xzf yumi_ckpt.tgz
PY=/root/miniconda3/bin/python
if [ ! -x .venv/bin/python ]; then
  $PY -m venv --system-site-packages .venv
fi
.venv/bin/pip install -q --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  mujoco==3.13.0 mujoco-warp==3.13.0 warp-lang==1.17.0 \
  numpy psutil PyYAML fastapi uvicorn pydantic scipy pyarrow glfw PyOpenGL click etils
.venv/bin/python - <<'EOF'
import torch, mujoco, warp, mujoco_warp
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
EOF
echo BOOTSTRAP_OK
