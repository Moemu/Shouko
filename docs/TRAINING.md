# 复现训练过程

本文档只覆盖**训练过程的复现**。环境安装、权重安装与验收见 [README](../README.zh.md)（[English](../README.md)）与 [STUDIO.md](../research/releases/v0.2.0/STUDIO.md)、[REPRODUCE.md](../research/releases/v0.2.0/REPRODUCE.md)。所有验收脚本都不训练、不选择检查点。

## 当前主线（v0.2.0，Yumi 下肢）

- 训练在云端 GPU 上按指南执行：本机流程见 [LOCAL.md](LOCAL.md)，云端流程见 [CLOUD.md](CLOUD.md)。复训需要相同的连接组数据与 vendor 环境。
- 方法回顾：Yumi MLP 教师为训练数据标注动作 → 学生自己执行、把学生实际走到的状态（含失败态）交回教师再标注，迭代若干轮（DAgger）→ 在冻结核心缓存的特征上用岭回归解析拟合 9,792 个读出参数。教师只参与训练，评估与预览由连接组策略独立控制。
- 本轮迁移实验的五阶段归档与逐项核验记录见[连接组迁移编年](../research/experiments/CONNECTOME_TRANSFER_20260923.md)。
- 训练结束后按 README「安装与验收 v0.2.0 模型」一节运行本地验收命令。

## 历史世系（G1 时代）

项目前两代在 G1 代理身体上完成，以下命令仍可复现。G1 是宇树（Unitree）的人形机器人模型：子图原型经冻结的官方步态策略驱动 G1，云端全图则整路径训练、直接输出 12 个关节目标。

### 子图原型（8,192 神经元）

需要 Python 3.12、uv、Node.js 20.19+ 或 22.12+、Git。`setup.ps1` 下载约 1.1 GB 连接组原始数据并建立 `.venv`（子图原型的旧版环境）：

```powershell
.\setup.ps1
.venv\Scripts\python.exe -m app.prepare
.venv\Scripts\python.exe -m app.train --samples 1536
.venv\Scripts\python.exe -m app.check
npm run build
```

服务器运行构建后的页面，改完前端先构建再刷新浏览器。

### 云端全图（G1 身体）

云实例复用基础 PyTorch 2.12.1+cu130，依赖锁定在 `requirements.cloud.lock.txt`，不要用本地 CPU 原型的安装脚本覆盖云环境。实例配置、基准数据与费用记录见 [CLOUD.md](CLOUD.md)。

```bash
cd /root/autodl-tmp/neuromechfly
.venv/bin/python -m app.launch_cloud train --seconds 600
.venv/bin/python -m app.evaluate_full
.venv/bin/python -m app.check_full
```

训练进程结束后再跑验收。

### Yumi PPO 世系

`app/ppo_yumi.py` 从 G1 检查点对 Yumi 身体做强化学习微调（感觉编码、神经元偏置、连接增益与读出整路径训练，约 2,658 万参数）。过程与调参记录见 `research/experiments/` 编年，路线起点见 [REPORT.md](../research/REPORT.md) 与 [路线决策记录](../research/experiments/ROUTE_DECISION_20260922.md)。
