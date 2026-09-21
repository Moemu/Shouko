# 下一步训练策略独立提案（2026-09-21）

提出方：Codex-independent。

## 独立性与范围

本提案未读取任何 `NEXT_STEP_PROPOSAL` 文件，也未读取 `research/experiments/` 编年结论。只读取了 `AGENTS.md`、`research/REPORT.md`、`research/guides/LOCAL.md`、`research/guides/CLOUD.md`，指定代码，以及 `runs/` 原始结果。

本轮只做分析和提案，没有运行 GPU 仿真、训练或云端连接，也没有改生产代码。文中的训练时长和算力是建议预算，不是承诺。

## 核心判断

1. 策略的稀疏反向传播路径成立。它还没有证明 PPO 的优势估计足够可靠。
2. 价值预热是合理的工程保护，但当前证据不能证明它是必要条件，也不能证明当前 `v_loss` 阈值有意义。
3. `runs/yumi_obs50/diagnose_value.py` 当前不能用于确认或否定价值门控。它比较了不同状态的值和回报，并且把保存的对数标准差错误地先 clamp 再 `exp`。
4. 下一轮应该从已验证会走的 squat 检查点开始，隔离“新增可观测量”和“价值预热”两个因素。继续对没有基础步态的 tall 检查点增加 burst，辨识力很低。

## 已确认事实

### 梯度路径可用，但这不是 PPO 成功证据

`app/full_brain.py:23-30` 用 `sampled_addmm` 计算实测边上的梯度。`runs/cloud/checks_full.json` 报告了稀疏前向/反向与 dense reference 一致，26,575,852 个参数都收到梯度；`edge_delta` 梯度最大值为 0.009106，更新边比例为 0.818946。这个结果支持“梯度能到达边、偏置、编码器和读出层”。

它没有回答优势符号是否正确、回报是否能由当前观测预测，或策略更新是否保留步态。后两项仍需控制实验。

### 现有 PPO 的价值误差没有校准尺度

`app/ppo_yumi.py:274-353` 在每个 128 步 rollout 中用当前值网络计算 GAE。`app/ppo_yumi.py:388-393` 用同一批 rollout 的 `returns` 更新值网络，并把训练批次的 MSE 记录为 `v_loss`。`app/ppo_yumi.py:414-435` 的可选门控仍使用这个训练批次误差；没有独立状态、独立噪声或 held-out return 的指标。

原始日志显示高 `v_loss` 与短期成功可以同时出现：

- `runs/yumi/ppo9.log` 在第 10 次评估有 31/32 个筛查回合成功、score 0.853。后续训练行的 `v_loss` 在 217.7–607.0 之间，第 15 次评估变为 0/32、score -0.112。
- `runs/yumi/cloud_ppo5_review_20260915/training.json` 的 151 次记录中，`v_loss` 为 62.0–228.5；配套摘要记录第 65 次筛查为 16/32、score 0.721，第 150 次为 0/32、score 0.487。

这些记录说明训练会漂移，也说明不能用一个未经校准的绝对 `v_loss` 值判断优势“有噪声”或“可用”。它们不是同一硬件和同一评估时长，不能直接比较绝对 score。

### 47 维观测缺少奖励直接使用的量

`app/gpu_body.py:109-122` 的 47 维观测包含角速度、重力、命令、关节位置/速度、上一动作和相位，但不包含 `qvel[:,0:2]` 或骨盆高度。奖励在 `app/ppo_yumi.py:289-331` 直接使用前向速度、侧向速度和骨盆高度。50 维分支才追加线速度和相对骨盆高度。

这使 50 维扩展有明确的可检验理由，但不是成功保证。当前 `runs/yumi_obs50/best_tall_obs50.pt` 的检查点 metadata 仍是 50 维、5 次更新、30 秒筛查 7/32、score 0.488、平均速度 0.235 m/s。它从 tall 检查点扩展而来，不能证明新增观测已经改善策略。

### tall 与 squat 的差异首先是基础步态差异

`runs/local/gait_measure_20260918/gait_squat.json` 的 9 个 30 秒回合均未跌倒。按命令分组的平均速度为：

| 目标速度 | 平均速度 | 平均骨盆高度 |
|---:|---:|---:|
| 0.35 m/s | 0.4429 m/s | 0.8286 m |
| 0.50 m/s | 0.5145 m/s | 0.8304 m |
| 0.65 m/s | 0.6084 m/s | 0.8319 m |

同目录的 `gait_tall.json` 也没有跌倒，但平均速度只有 0.0376、0.1380、0.0407 m/s，平均骨盆高度约 0.90 m。`transfer_tall.json` 在 0.35、0.50、0.65 m/s 命令下的平均速度只有 0.0076、0.1387、0.1601 m/s。此 tall lineage 没有可用于判断“保留步态”的可靠起点。

已验证的 squat 检查点 `runs/local/heldout_yumi.json`（checkpoint SHA-256 前缀 `a7a4281f`）在 9/9 个独立 30 秒回合通过，另有 3/3 次轻推回合和 120 秒回合通过。它适合作为下一轮的保留步态基线。

## 价值诊断的逐项审查

`runs/yumi_obs50/diagnose_value.py` 的思路是对的：固定初始状态，改变 action noise，估计回报的条件随机部分，再看值网络能解释多少变化。但当前实现有以下问题。

1. **值和回报不是同一状态。** `rollout()` 在第 32–68 行结束后，`env` 已经处于 rollout B 的末状态；第 73 行才计算 `v0`，却拿它与 rollout A 的回报比较。应在每个 rollout 开始前保存 `obs0`，或保存整段 `obs_t`，然后比较 `value(obs_t)` 与同一状态的 return-to-go。
2. **探索噪声的 clamp 顺序错误。** 第 36 行先对 `log_std` 做 `clamp(0.05, 1.5)`，第 44 行再 `exp()`。训练代码第 283–284 行是先 `exp()`，再把标准差限制在 0.05–1.5。保存的 `log_std` 若对应 0.08，诊断脚本会把它变成约 1.05 的标准差，改变了被测过程。
3. **诊断奖励不等于训练奖励。** 训练奖励在 `app/ppo_yumi.py:298-331` 包含 `switched` 足部交替奖励和上一动作差分惩罚。诊断第 56–63 行没有交替项，且把动作差分项写成 `(action * 0)`。训练还每步设置 yaw feedback 和 0.15–0.75 的速度命令；诊断第 41 行固定 yaw 命令为零，速度命令沿用初始化的 0.5。
4. **噪声与重置的混合没有标注。** 两次 rollout 虽然用同一 reset seed，但摔倒后会随机重置。不同 action noise 造成的摔倒时间差会进一步改变后续 reset 状态。因此 pair difference 不是纯粹的单状态动作噪声，除非逐步保存状态并单独分析首个 episode。
5. **指标名称不准确。** 第 83 行记录的是 `residual.std()`，不是 RMSE。第 84 行的 R² 也使用了错位状态的 residual。至少要同时报告 `sqrt(mean(residual**2))`、残差均值和 held-out explained variance。
6. **当前原始产物不足。** 在本次检查时，`runs/yumi_obs50` 可见的检查点只有 `best_tall_obs50.pt`；脚本硬编码读取 `last.pt` 和 `ppo_state.pt`，也没有现成的 `diagnose_value.json` 可供复核。

因此，当前结论应是：梯度实现有数值证据；价值预热只有工程直觉和历史相关日志；“不可约噪声导致高 `v_loss`，所以策略仍可更新”的诊断尚未成立。

## 建议的下一轮实验

### A. 先建立可复核的基线和 50 维等价性

从 `runs/yumi/best.pt` 建立隔离的 50 维副本，身体接口使用 `app/yumi_description/yumi.yaml`。`app/expand_observation.py` 已规定新增编码器列初始化为零、观测均值补零、标准差补 1。训练前做 16 个 world、10 秒的 47/50 维配对检查：动作、摔倒布尔值和每个 world 的速度差应在预先写明的浮点容差内相等。失败就停止，不进入 PPO。

这一步应使用已验证会走的 squat lineage。不要把 `best_tall_obs50.pt` 的非行走表现当作 50 维接口的基线。

### B. 用训练同构的只读诊断校准价值网络

建立一个临时诊断脚本，不改生产入口，内容应满足：

- 复用训练中的命令分布、yaw feedback、`switched` 奖励、上一动作惩罚、posture reward 和 terminal penalty。
- 按训练顺序计算 `std = exp(log_std).clamp(0.05, 1.5)`。
- 对相同 reset 状态运行至少 4 个 noise seed；保存 `obs_t`、即时奖励和每个 `t` 的 return-to-go。摔倒后的新 episode 单独分组，不与原状态混合。
- 用同一状态的 `value(obs_t)` 计算 NRMSE、残差均值和 held-out explained variance。另报 pair-difference 的条件方差估计和置信区间。
- 对 47 维和 50 维输入使用同一批状态，比较新增线速度/高度是否真的改善值预测。

诊断阶段不更新 actor，不选择 checkpoint，也不使用 `value_abort_vloss`。如果 50 维没有改善 held-out value explained variance，就没有理由把更多预算投入价值门控；应先重新审查奖励和可观测性。

### C. 只做两臂短 PPO 对照

只有在 B 产出可复核的状态对齐指标后，才建议做两臂。两臂从同一个 50 维 squat 副本开始，固定 seed、world 数、命令分布、奖励、学习率、参数冻结范围和评估种子：

| 臂 | `value_warmup` | 目的 |
|---|---:|---|
| W0 | 0 | 测试立即更新 actor 的基线风险 |
| W6 | 6 | 测试 6 个 value-only iteration 是否改善 actor 更新 |

建议沿用资源受限 runner 的 `worlds=16`、`std-override=0.08`、`lr=1e-4`、`epochs=4`、`target-kl=0.02` 和 `knee-gate=stance`，但每臂最多 6 次 rollout iteration。`--freeze-brain` 只冻结 `edge_delta`；它仍会更新 `neuron_bias`、encoder 和 readout。记录这一事实，不能把结果称为 readout-only。

每臂在第 0、3、6 次做短筛查，最终只对候选做 `measure_gait` 的 0.35/0.50/0.65 m/s 三档、3 个种子，并用未参与筛查的 8004–8009 种子复核。选择条件应同时满足：不牺牲已验证的生存、速度、交替触地和方向指标，并产生可重复的骨盆高度改善。建议把约 0.02 m 的高度增量作为首轮“有实际意义”的观察门槛；这是建议的检测尺度，不是用户必须接受的定稿标准。

若 W0 和 W6 在相同预算下都不能保持 squat 步态，停止继续加时。若两臂都保持步态但高度没有稳定改善，应优先改变 posture 目标或身体接口，而不是扩大 PPO 更新次数。若 W6 在 held-out 上重复优于 W0，再考虑把 value gate 接入生产训练流程。

## 建议预算与停止规则

- 诊断：4 个 noise seed × 16 worlds × 128 steps，建议只预留一个短窗口。
- PPO：两臂各不超过约 420 秒活动时间，合计约 14 分钟活动时间，另加配对评估时间。这个数字是建议上限，受机器负载和 CUDA 图捕获影响，不承诺完成次数。
- 不建议直接执行 `runs/yumi_obs50/run_burst_local.py` 的 8 个 burst。当前证据还不足以支持连续长时间更新。
- 任一臂出现 baseline 步态明显退化、值诊断无法复核、或 checkpoint/interface/hash 不一致，应立即停该臂并保留日志，不继续消耗算力。

## 待用户定稿的方向

本提案只建议先做“已验证 squat 基线 + 训练同构价值诊断 + W0/W6 短对照”。是否以 50 维观测、0.02 m 高度门槛和两臂预算作为正式实验标准，由用户在各独立提案比较后定稿。本文件不代用户选择最终训练方向。

## 交叉审查后的执行约束

1. 每臂最多 6 轮时，W6 的 6 轮全部用于 value-only 预热，没有 actor 更新。它只能作为冻结 actor 对照，不能单独判断“预热后学习”是否有效。若比较预热效果，预热后两臂必须再运行相同数量的 actor 更新，并另报预热增加的样本量和成本。
2. 8004–8009 已出现在 `runs/local/heldout_yumi.json`。它们不能再称为新的最终留出种子。新验收须另选未参与方案选择的种子。
3. 零扩维等价性先做同一输入下的前向输出对比，再用同一模型重复 rollout 建立数值噪声包络。两个 GPU 轨迹不要求逐位一致。
