# 奖励与更新机制审计

**审查者：** Codex independent
**日期：** 2026-09-22
**范围：** `app/ppo_yumi.py`、`app/gpu_body.py`、`app/sim.py`、`app/full_brain.py` 与 `runs/` 中已有实测数据。

本审查未读取其他提案或其结论。没有运行 GPU、云端实验，也没有修改生产代码。以下把代码事实、实测事实和待验证推断分开。

## 结论

1. 没有发现足以确认“观测、PD、摔倒判定或 PPO 更新接线错误”的直接缺陷。GPU 与 CPU 的观测字段、动作缩放和摔倒条件在 `app/gpu_body.py:118-141`、`app/sim.py:143-175,193-204` 中相互对应。
2. 奖励有两个需要继续验证的结构特征：解析的直立零速状态基线较高；横向位移没有直接惩罚，换脚奖励也不要求向前移动。这些是代码事实，不是已确认的激励缺陷，也没有证明它们造成当前停滞或横漂。
3. `ConnectomePolicy` 每次调用默认清空内部 activity。`app/full_brain.py:72-85` 与 `app/ppo_yumi.py:317,412` 表明训练调用没有传回 state。这是可验证的架构假设；相位和上一动作仍提供了部分时间信息，所以不能直接称为 bug。
4. rollout reward、Value loss 和单次评估都不足以单独决定是否长训。现有短训之间的评估差异说明需要固定配对评估；不要把 Value loss 差异解释为估值精度差异。

## 代码与数据证据

### 1. 解析零速状态基线

训练速度命令在 `app/ppo_yumi.py:289` 取 `0.15..0.75 m/s`。核心奖励在 `app/ppo_yumi.py:333-339`。对直立、零速度、零偏航、零动作且未换脚的状态，忽略姿态塑形时，单步奖励为：

`1.8 + 1.5 * exp(-4 * command_x^2)`。

该值约为 `3.17`（0.15 m/s）、`2.35`（0.50 m/s）和 `1.96`（0.75 m/s），命令区间平均约 `2.51`。这是一个解析状态基线，不代表存在一条能长期保持该状态的静止轨迹，也不能由常数项本身推出动作梯度偏爱静止。`runs/yumi_actor_scale_ab_20260922/control/training.json` 和 `scaled/training.json` 的第 1 轮平均 reward 约 `2.5246`、`2.5243`，只能说明 warmup rollout 的平均值接近该解析基线。

**可确认事实：** 奖励使用速度跟踪项，但没有单独的“不前进即失败”终止或惩罚项。

**待验证推断：** 可实现的完整轨迹回报可能仍让低速稳定行为占优。需要轨迹级对照，不能用这个瞬时解析基线代替比较。

### 2. 横漂与换脚奖励没有被直接约束

`app/ppo_yumi.py:334` 只惩罚瞬时 `vy^2`，没有累计横向位置、横向漂移距离或相对路径的项。`app/ppo_yumi.py:337` 对满足足底高度差的 `switched` 直接加 `0.4`，没有要求该换脚同时带来正向速度或减小横向误差。

因此，换脚项本身可以获得 credit，却没有把该 credit 与正向速度绑定。这个结构与横漂现象相容，但当前代码没有记录每回合 switch 次数，不能从现有日志量化其贡献，也不能据此确认激励因果。

现有配对评估数据也说明训练 reward 与行走质量不是同一个指标：`runs/yumi_actor_scale_ab_20260922/paired.json` 同时记录了 `mean_speed`、`lateral_m`、`success`。这些指标应作为选择标准；不能用 rollout reward 或 vloss 替代它们。

### 3. 神经状态每次推理都会重置

`app/full_brain.py:72-85` 在 `state is None` 时执行 `activity = zeros_like(signal)`。训练收集和 PPO 更新分别在 `app/ppo_yumi.py:317`、`app/ppo_yumi.py:412` 调用 `policy(obs)`，没有接收并传回第二个返回值 `activity`。所以每个控制步只运行 `neural_steps` 次内部更新，跨步没有 connectome activity 记忆。

这是当前实现的确定事实，不是训练失败的证明。`app/gpu_body.py:118-131` 仍输入 gait phase 和上一动作，策略可能用这些量形成有限的时间反馈。需要固定权重的 state carry/reset 对照，才能判断它是否影响速度或横漂。

### 4. 更新与动作路径的检查结果

- PPO 在 `app/ppo_yumi.py:411-435` 正常计算 ratio、clip、反向传播并分别更新 actor 与 Value；没有看到跳过 actor 更新的条件以外的接线错误。
- 环境在 `app/gpu_body.py:133-140` 和 `app/sim.py:166-175` 将动作裁剪到 `[-8,8]`，并按相同的 home、action scale、PD 结构执行。
- `app/ppo_yumi.py:320,368,415` 保存 raw Gaussian action，并对 raw action 计算 raw log-prob；环境随后对动作做确定性 `[-8,8]` clip。按当前调用路径，这可以是“潜在动作策略、裁剪后执行”的合法定义；没有发现动作存储或概率调用错用。只有在动作频繁触碰边界，或项目明确要求执行动作的概率模型时，才需要另行审查。现有 `runs/yumi_actor_scale_ab_20260922/control/replay.json` 和 `scaled/replay.json` 的已记录动作没有达到 `|a|=8`，因此不把它列为当前低速或横漂原因。

## 最小可证伪对照

只建议下面两个方向，每个对照保持 checkpoint、随机种子、worlds、rollout 长度、优化器和配对评估完全相同。

1. **换脚奖励消融。** 只把 `app/ppo_yumi.py:337` 的 `0.4 * switched.float()` 置零，其他奖励和训练参数不动。用同一组配对评估比较 `mean_speed`、`lateral_m`、success、跌倒率。若横漂和成功率没有稳定改善，则“换脚 credit 是横漂主要来源”的假设被削弱；若仅速度下降而横漂不变，则该项主要支持动作活跃度，不能继续当作行走质量奖励。
2. **固定权重 state carry/reset 对照（低优先级）。** 对同一 checkpoint 和相同初态，分别让相邻控制步传递 `activity`，以及保持当前的每步清零；episode reset 时清零，并记录相同的速度、`lateral_m`、偏航和跌倒时间。这个对照会直接改变为 reset 训练的既有权重的策略函数。若 carry 对这些指标没有稳定影响，只能削弱“已有 reset 权重可即插即用地受益于 carry”的假设，不能否定经过 state carry 训练的记忆策略价值。这个实验不应自动改生产训练路径。

在上述对照完成前，不建议仅凭 reward 上升或 vloss 下降继续长训并宣称路线有效。评估应使用同一配对初态，并把训练 reward、速度、横漂、存活和成功率分开报告。
