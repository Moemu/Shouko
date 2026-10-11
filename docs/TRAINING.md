# 复现训练过程

本文档只覆盖**训练过程的复现**。环境安装、权重安装与验收见 [README](../README.zh.md)（[English](../README.md)）与 [STUDIO.md](../research/releases/v0.2.0/STUDIO.md)、[REPRODUCE.md](../research/releases/v0.2.0/REPRODUCE.md)。所有验收脚本都不训练、不选择检查点。

## 预览基线（v0.2.0，Yumi 下肢）

- 训练在云端 GPU 上按指南执行：本机流程见 [LOCAL.md](LOCAL.md)，云端流程见 [CLOUD.md](CLOUD.md)。复训需要相同的连接组数据与 vendor 环境。
- 方法回顾：Yumi MLP 教师为训练数据标注动作 → 学生自己执行、把学生实际走到的状态（含失败态）交回教师再标注，迭代若干轮（DAgger）→ 在冻结核心缓存的特征上用岭回归解析拟合 9,792 个读出参数。教师只参与训练，评估与预览由连接组策略独立控制。
- 本轮迁移实验的五阶段归档与逐项核验记录见[连接组迁移编年](../research/experiments/CONNECTOME_TRANSFER_20260923.md)。
- 训练结束后按 README「安装与验收 v0.2.0 模型」一节运行本地验收命令。

## 开发实验：单列感觉接入

2026-09-30 的训练接口为 `app.train_sensory_readout`。它保留完整连接组，只训练读出，或同时训练 50 维观测中世界系 vy 对应的 encoder 第 48 列。索引从 0 开始。实验记录与源权重哈希见[感觉接入编年](../research/experiments/SENSORY_ACCESS_20260930.md)；两份固定候选作为 v0.3.0 独立发行，默认预览仍为 v0.2.0。

以下为 CUDA 环境中的接口示例。将示例路径替换为自己的源权重和由 `app.imitate_yumi collect` 生成的数据。训练与验证必须使用相同教师、相同物理接口，且回合种子不重叠。

```bash
python -m app.train_sensory_readout --source source.pt --train train.pt --validation validation.pt --output runs/sensory_trial --sensory --calibrate-vy --updates 500 --batch 128 --seed 3031
```

省略 `--sensory` 即为只读出对照。`--calibrate-vy` 仅适用于源权重的 vy 列全零时；它用训练数据确定该项标准差，并核验初始动作不变。从已接通该列的权重继续训练时，省略该选项，保持尺度固定。每次调用使用新 Adam；输出目录必须尚不存在。保存前会检查冻结参数及标准检查点格式。

后续[稳定性验证](../research/experiments/SENSORY_STABILITY_20260930.md) 使用数据集 × 采样种子的交叉对照，再匹配两臂的补充数据和累计更新预算。第二段保留首段 vy 标准差，重新创建 Adam，并使用预定的新采样种子。先固定候选，再启用新留出；不能根据留出结果重选训练种子或调整参数。

`fit.json` 记录采样顺序哈希、梯度、参数变化、输入响应和分组验证误差。输入列改变后必须重新计算连接组特征，不能沿用旧冻结特征缓存。拟合结果仍需通过 `app.evaluate_direction` 的物理验收；动作误差下降不等于长走走廊通过。

采集接口支持 `--speeds .65` 生成纯高速队列，并在每个回合保存 `push_applied`。三档默认速度仍为 0.35、0.50、0.65 m/s。v0.3.0 包含固定候选、完整图、运行时和评估证据，见[复现说明](../research/releases/v0.3.0/REPRODUCE.md)。历史训练数据未全部打包；上述训练接口不承诺逐字节重建本轮实验。

## 开发实验：双臂目标任务

独立的 18 关节目标任务使用 69 维观测，允许低姿态移动，不加载步态教师。当前 `app.ppo_goal` 只支持 MLP；它不替换工作室默认模型，也不使用旧版 47/50 维 checkpoint。身体假设、测试证据及限制见 [本机闭环记录](../research/experiments/GOAL_PPO_LOCAL_20261011.md)。

使用已有 CUDA 环境。该实验额外需要 MuJoCo Warp；依赖安装如下，输出目录必须尚不存在：

```powershell
uv pip install --python .venv-gpu\Scripts\python.exe -r requirements.goal-gpu.lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
.venv-gpu\Scripts\python.exe -m app.test_goal_training --gpu --output runs/goal_contract_trial
.venv-gpu\Scripts\python.exe -m app.ppo_goal --output runs/goal_ppo_trial --iterations 32 --max-seconds 120
.venv-gpu\Scripts\python.exe -m app.ppo_goal --output runs/goal_ppo_resume_trial --iterations 2 --max-seconds 30 --resume runs/goal_ppo_trial/last.pt
```

默认 16 个世界、每轮 64 步。`--iterations` 为本次最多追加的轮数；`--max-seconds` 限制初始化完成后的训练时间，初始化和保存另计时。MLP 不需要连接组数据或旧权重。短测只验证训练链路，不承诺学会到达。

`initial.pt` 保存本次训练开始时的状态，`last.pt` 原子保存最终策略、价值网络、探索噪声和 Adam；`training.json` 记录进度，`summary.json` 记录哈希、参数变化、吞吐和分配器显存统计。恢复要求相同身体、观测、任务、关节与 PPO 参数，并重新开始环境。该入口只做优化器热启动，不恢复物理现场，也没有接入工作室页面的训练按钮。

固定预算筛查及初始/最终策略的独立评估示例：

```powershell
.venv-gpu\Scripts\python.exe -m app.test_evaluate_goal
.venv-gpu\Scripts\python.exe -m app.evaluate_goal --create-deck --deck runs/goal_screen_trial/deck.json
foreach ($seed in 101, 202, 303) {
    .venv-gpu\Scripts\python.exe -m app.ppo_goal --output "runs/goal_screen_trial/seed_$seed" --seed $seed --iterations 256 --max-seconds 420
    if ($LASTEXITCODE -ne 0) { throw "Training failed for seed $seed" }
    foreach ($stage in 'initial', 'last') {
        .venv-gpu\Scripts\python.exe -m app.evaluate_goal --deck runs/goal_screen_trial/deck.json --checkpoint "runs/goal_screen_trial/seed_$seed/$stage.pt" --output "runs/goal_screen_trial/eval_${seed}_$stage"
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed for seed $seed" }
    }
}
```

固定牌组含 16 个方向，每回合三个目标。评估使用原生 MuJoCo CPU 和确定性均值动作，按原任务的距离、速度、停留和超时规则判定；不使用训练探索噪声。每份结果绑定牌组、检查点、身体与代码哈希，并保存逐回合指标及 10 Hz 轨迹。创建牌组与评估输出目录均拒绝覆盖。评估前不得根据结果改变牌组或调参；完整样本预算须检查 `summary.json` 的实际转换数。首轮实测见 [多种子筛查](../research/experiments/GOAL_MLP_SCREEN_20261011.md)，它不代表已收敛。

若需定位长期接触轨迹的后端差异，可用同一牌组和检查点运行只读 CUDA 对照。它额外需要 MuJoCo Warp，记录到达、终止、距离进展、机械功和质心轨迹；不替代原生评估中的关节及接触身体明细。

```powershell
.venv-gpu\Scripts\python.exe -m app.evaluate_goal_gpu --deck runs/goal_screen_trial/deck.json --checkpoint runs/goal_ppo_trial/last.pt --output runs/goal_gpu_check_trial
```

控制与奖励诊断、三组单因素筛查入口：

```powershell
.venv-gpu\Scripts\python.exe -m app.diagnose_goal --output runs/goal_control_probes_trial
.venv-gpu\Scripts\python.exe -m app.test_goal_screen
.venv-gpu\Scripts\python.exe -m app.screen_goal_controls --output runs/goal_controls_trial --exploration-std 0.70
.venv-gpu\Scripts\python.exe -m app.verify_goal_controls --output runs/goal_controls_trial
```

先查看探针的 `summary.json`。仅当 `exploration_passed=true` 时，使用其中的 `exploration_std` 运行筛查。筛查固定种子 404、505、606，比较默认配置、零训练功代价和增大探索三组，每次 262,144 次转换。`campaign.json` 记录当前子任务，`summary.json` 汇总全部结果并给出 `next_branch`；输出目录拒绝覆盖，任何子任务失败或样本预算不足都会停止。

单次训练可用 `--initial-std` 设置初始探索标准差，用 `--work-cost` 设置训练奖励的机械功代价。物理任务和公共评估的功代价保持 0.001，因此不同组仍能按相同任务指标比较。非默认训练功代价写入检查点设置并参与恢复检查；默认配置保留旧检查点的设置结构。参数和观测接口不变。

阶段预算、分支及当前结果见 [本机诊断编年](../research/experiments/GOAL_LOCAL_CAMPAIGN_20261011.md)。九组正式筛查及十二份 GPU 复核均完成，没有到达，三组均为 0/3 种子通过持续推进门槛。当前分支为 `route_uncertainty_stop`：控制命名、奖励记账及后端能力结论已核验，训练预算是否不足、奖励/探索组合或信用分配问题仍不能区分。按项目停机门交回跨厂商路线讨论；尚未执行延长训练、新留出验收或连接组资源阶段，也未达到远端训练准备条件。

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
