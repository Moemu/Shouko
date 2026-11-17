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

前序阶段预算和结果见 [本机诊断编年](../research/experiments/GOAL_LOCAL_CAMPAIGN_20261011.md)。九组正式筛查及十二份 GPU 复核均完成，没有到达，三组均为 0/3 种子通过持续推进门槛。控制命名、奖励记账及后端能力结论已核验，随后交回跨厂商讨论。

用户定稿后的 [D0 持续动作与分布诊断](../research/experiments/GOAL_CONTROL_D0_20261116.md) 已完成。100 条菜单中 80 条完整完成，20 条按物理警戒线中止；8 条安全菜单在倒伏后有额外推进，但三条生成历史均落入同一姿态类别，未通过“两类低姿态”门。三个既有策略另完成合计 12,288 次采样/均值配对转换，没有更新权重。当前分支为 `coverage_or_local_only`：局部可动已有证据，探索、能耗和完整任务学习仍待路线定稿。尚未达到连接组或远端训练准备条件。

使用已复训的基线检查点，运行同一冻结诊断与证据核验：

```powershell
.venv-gpu\Scripts\python.exe -m app.diagnose_goal_d0 --checkpoint-root runs/goal_controls_trial --output runs/goal_d0_trial --max-seconds 3600
.venv-gpu\Scripts\python.exe -m app.verify_goal_d0 --checkpoint-root runs/goal_controls_trial --output runs/goal_d0_trial
```

检查点根目录应含 `baseline_404/last.pt`、`baseline_505/last.pt`、`baseline_606/last.pt`，可由上文 `app.screen_goal_controls` 生成。D0 输出目录必须不存在；核验只读取已有结果，并以少量轨迹回放检查控制实现。诊断入口不训练、不生成新权重，不把四秒位移计为原三目标任务验收。

### 自然倒伏后的局部学习对照

这是独立的 `fallen_local_goal_v1` 诊断任务：生成自然倒伏状态，保留实际速度和 warmstart，固定 +x 两米目标，每回合 5 秒。身体和 69/18 接口沿用双臂任务；它不等于原始直立开局、随机三目标的能力验收。

安装上文 `requirements.goal-gpu.lock.txt` 环境后，可直接从随机权重复训，无需私有 checkpoint 或连接组数据：

```powershell
.venv-gpu\Scripts\python.exe -m app.test_local_goal --output runs/local_goal_contract_trial
.venv-gpu\Scripts\python.exe -m app.screen_local_goal --output runs/local_goal_trial --max-seconds 10800
.venv-gpu\Scripts\python.exe -m app.verify_local_goal --output runs/local_goal_trial
```

筛查固定三个配对种子，比较独立/持续控制分布 × 原/零训练功代价，每组每种子 262,144 次转换。`campaign.json` 记录进度，`summary.json` 给出实际完成数和下一分支。任何输出目录都必须不存在；核验读取已经完成的目录，不重训。

局部权重使用独立 checkpoint schema，并绑定完整 reset 文件哈希、任务期限和 rho。旧评估/预览加载器会拒绝局部权重。持续组的训练 log-prob 和确定性评估都使用相同条件分布，不能绕过其前次动作项。

单独评估已完成的局部 checkpoint：

```powershell
.venv-gpu\Scripts\python.exe -m app.evaluate_local_goal --checkpoint runs/local_goal_trial/persistent_cost_1401/last.pt --state runs/local_goal_trial/reset.npz --output runs/local_goal_eval_trial --backend native
```

`--backend cuda` 可复核同一状态、执行规则和权重的后端结论。每份结果绑定 checkpoint、reset、源码和逐步轨迹，不将相同的确定性回合重复计为独立样本。筛查保存执行源码快照，核验器检查其哈希并独立重算原始 rollout 与评估。

阶段预算、固定门和当前结果见[局部学习编年](../research/experiments/LOCAL_GOAL_FACTORIAL_20261116.md)。十二个正式 run 已处理：九个完成固定预算，三个持续零功 run 因随机采样穿透警戒线中止，实际尝试 2,696,976 次转换。四组均为 0/3 种子通过持续推进门。独立零功组能接近目标，但末两秒明显回退；持续功组主要早期推进后停滞。中断权重另作只读诊断，不补成完整候选。

该局部因子阶段结束于 `route_uncertainty_stop`，当时没有执行原初态恢复课程或远端训练。后续已获定稿的有限终点与保持验证见下一节；旧任务与生产权重保留。

### 有限终点、制动诊断与近场保持

旧 `fallen_local_goal_v1` 保留其 timeout 截断语义。新 `fallen_finite_goal_v1` 将原 5 秒截止视为真正终止，超时不自举。近场学习另用 `nearfield_holding_v1`：恢复干预前的完整状态、原目标和已用时间；不会从分叉状态重新获得 5 秒。

使用上一节自行复训的 `runs/local_goal_trial`，可生成只读诊断。该阶段固定九份完整最终策略、九份分叉状态、54 条动作/目标续接，并记录实际终点：

```powershell
.venv-gpu\Scripts\python.exe -m app.test_braking_task --output runs/braking_contract_trial
.venv-gpu\Scripts\python.exe -m app.diagnose_goal_braking --source-screen runs/local_goal_trial --output runs/braking_trial/diagnostic --max-seconds 3600
.venv-gpu\Scripts\python.exe -m app.verify_goal_braking --source-screen runs/local_goal_trial --output runs/braking_trial/diagnostic
```

先检查 `summary.json` 的分支与 `verification.json`。只有原目标下安全完成连续 1 秒停留，才支持近场课程。重新训练受后端和采样影响，不承诺得到相同阳性状态；阴性结果不应通过搬位置、清速度或放宽停留规则修复。

本轮冻结的课程入口只接受已核验的单一阳性 `1402_145`，并拒绝其他状态集合。它取该探针的**干预前**快照，不读取脚本动作标签。两臂使用三份配对随机初始化、相同初始 std=0.15、rho=0、功系数 0.001；处理臂只加 `gamma*Phi(next)-Phi(current)`，其中 `Phi=0.5*dwell_fraction`，真实终点的 Phi=0。新 checkpoint schema 为 `yumi_holding_goal_ppo`，旧任务/预览加载器拒收。

若自己生成的诊断满足同一冻结条件，运行以下命令。输出目录均须不存在；工程短测使用 1024 次转换，正式对照上限 1,572,864 次转换、训练进程累计 45 分钟。

```powershell
.venv-gpu\Scripts\python.exe -m app.test_holding_goal --state runs/braking_trial/diagnostic/forks/fork_1402_145.npz --output runs/braking_trial/contracts
$holdingDeadline = [DateTimeOffset]::UtcNow.AddHours(1).ToUnixTimeSeconds()
.venv-gpu\Scripts\python.exe -m app.run_holding_goal --diagnostic runs/braking_trial/diagnostic --output runs/braking_trial/learning --wall-deadline $holdingDeadline
.venv-gpu\Scripts\python.exe -m app.verify_holding_goal --output runs/braking_trial/learning
```

每个训练种子的最终策略需在全部冻结状态、native/CUDA 两后端都安全完成原停留，组门为至少 2/3。初始策略也评估，防止将已有保持能力误记为学习收益。两臂均通过时优先简单基线；单来源课程不证明跨状态、两米移动或三目标能力。

通过近场门并完成独立核验后，按独立预登记只读回到原自然倒伏两米任务。源权重仍绑定近场状态，目标状态另行绑定，不重训、不替换 checkpoint 合同：

```powershell
.venv-gpu\Scripts\python.exe -m app.evaluate_holding_transfer --learning runs/braking_trial/learning --original-reset runs/local_goal_trial/reset.npz --output runs/braking_trial/transfer --wall-deadline $holdingDeadline
```

2026-11-17 执行结果：只读诊断完成 46,080 次转换和 54 条续接，只有一个来源状态能完整保持。两臂近场学习均为 3/3 种子通过，初始策略也全部通过，因此未证明塑形必要或训练创造了保持能力。回到原自然倒伏两米任务的 18 条只读轨迹全部安全，但两臂最终均为 0/3，净进展约为 −0.003～−0.014 米；单状态保持没有自动迁移。

当前分支为 `route_uncertainty_stop`。接近与保持的衔接方式需跨厂商讨论并由用户定稿；不追加训练或启动远端。预算、分支、哈希、复现命令和限制见[制动与保持编年](../research/experiments/GOAL_BRAKING_20261117.md)。这些入口可从自行生成的局部权重和状态运行，不依赖私有服务。它们是开发实验，未接入工作室或发行模型。

### 接近与保持联合对照

这是独立的 `approach_holding_v1` 实验。它从旧 `iid_zero` 复制 actor 全部参数与观测尺度，重建 critic、Adam 和探索状态。N 从同一门口状态训练；M 固定半数世界为原两米起点、四分之一为中间分叉、四分之一为门口。中间池按预定次序轮换；比例按控制转换计算，不按回合次数计算。

两臂共同 std=0.15、rho=0、训练功成本 0、势函数 0，原有限 5 秒截止和安全线不变。新 schema 为 `yumi_coverage_goal_ppo`，合同绑定源 actor、源 reset、完整池和分配规则；它是 actor 初始化，不是旧优化器恢复。

先从上述局部因子、制动与保持流程自行生成 `runs/local_goal_trial` 和 `runs/braking_trial`。只有前轮证据及新的范围检查通过，才进入本实验。重新训练可能产生不同分支，不承诺重复本轮数字；不能为了凑池搬位置、清速度或修改截止。输出目录均须不存在。

```powershell
$coverageDeadline = [DateTimeOffset]::UtcNow.AddHours(3).ToUnixTimeSeconds()
.venv-gpu\Scripts\python.exe -m app.check_holding_range --source runs/braking_trial --output runs/coverage_trial/range --wall-deadline $coverageDeadline
if ($LASTEXITCODE -ne 0) { throw 'C0 process failed' }
$rangeResult = Get-Content runs/coverage_trial/range/summary.json -Raw | ConvertFrom-Json
if ($rangeResult.branch -ne 'coverage_checked') { throw 'C0 did not permit learning' }
@'
from pathlib import Path
from app.coverage_goal import prepare_pool
prepare_pool(Path('runs/local_goal_trial'), Path('runs/braking_trial'), Path('runs/coverage_trial/pool'))
'@ | .venv-gpu\Scripts\python.exe -
if ($LASTEXITCODE -ne 0) { throw 'Pool preparation failed' }
.venv-gpu\Scripts\python.exe -m app.test_coverage_goal --pool runs/coverage_trial/pool --source-screen runs/local_goal_trial --output runs/coverage_trial/contracts
if ($LASTEXITCODE -ne 0) { throw 'Coverage contracts failed' }
.venv-gpu\Scripts\python.exe -m app.run_coverage_goal --source-screen runs/local_goal_trial --pool runs/coverage_trial/pool --range-check runs/coverage_trial/range --contracts runs/coverage_trial/contracts --output runs/coverage_trial/learning --wall-deadline $coverageDeadline
if ($LASTEXITCODE -ne 0) { throw 'Coverage process failed; inspect preserved evidence' }
.venv-gpu\Scripts\python.exe -m app.verify_coverage_goal --output runs/coverage_trial/learning --pool runs/coverage_trial/pool --source-screen runs/local_goal_trial --contracts runs/coverage_trial/contracts
```

先检查 C0 的 `summary.json` 和 `verification.json`；只有 `coverage_checked` 放行。工程 PPO 短测为 1536 次转换。正式比较三份源 actor，各 N/M 一份，每份最多 262,144 次转换，累计 1,572,864，训练子进程最多 45 分钟。源码与输入在正式前封存，运行期间变更源码会停止后续单元。

每个最终 actor 必须在原两米和门口两个起点、native/CUDA 四条轨迹全部完成停留，才算该来源通过。组门为至少 2/3 个相同来源通过四条，部分净位移不替代到达。初始快照也评估；中断单元不补成完整候选。

保存连续首四轮与最终轮，覆盖原起点第 250 步终点。核验器独立重算奖励/GAE、类别份额与重置，并在同一 CUDA 环境重放保存 RNG 的动作及 minibatch 索引；这些重放没有物理积分、梯度或参数更新。跨环境的历史训练不承诺逐字节一致。

本轮结果：C0 的 48 条新增原生轨迹全部安全，没有扩展保持范围；C1 全部固定预算及 48 条初始/最终评估完成。联合门 N/M 均 0/3；N/1402 在门口学会保持，但原起点推进下降。M 的原起点净进展约 0.82～1.04 米（两后端），没有完成停留；三个类别训练内到达均为 0。原起点实际转换和保存窗口优化批次已核实，不能再将本轮失败仅归于没有采样原起点。

所有 1536 轮都触发 KL 优化早停，首轮更新后的全窗口 Gaussian KL 约 3.19～8.32。它与日志的更新前 minibatch 近似 KL 不是同一指标；这是优化风险，尚未证明其为唯一根因。当前 `route_uncertainty_stop`，不补长训或改设置。完整预算、分支、哈希及解释边界见[接近保持编年](../research/experiments/APPROACH_HOLDING_20261117.md)。未启动远端、连接组训练或发行。

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
