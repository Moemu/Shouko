# 在线 actor 输入尺度 A/B 的独立判断

日期：2026-09-22

作者：Codex-independent（`/root/input_audit`）

## 独立性与数据范围

本提案未读取本轮其他提案。任务消息提供了实验范围和指标摘要；我没有把摘要当作结论，而是自行读取以下原始运行结果和代码复核：

- `runs/yumi_actor_scale_ab_20260922/summary.json`
- `runs/yumi_actor_scale_ab_20260922/paired.json`
- `runs/yumi_actor_scale_ab_20260922/verification.json`
- `runs/yumi_actor_scale_ab_20260922/control/training.json`
- `runs/yumi_actor_scale_ab_20260922/scaled/training.json`
- `app/ppo_yumi.py`
- `app/value_observation.py`
- `app/full_brain.py`

本次独立审查没有运行 GPU、云端命令或物理仿真，也没有修改代码。数据来自今天已经完成的在线云端 A/B 任务。

## 实验设计检查

两组都使用训练 seed `2026`、128 个并行训练世界、同一源 checkpoint、1 次 value warmup 和 6 轮 actor 更新。两组各完成 `768` 个 policy optimizer steps、`114688` 个环境步，`kl_stop=0`。control 使用新增三列的 actor scale `[1, 1, 1]`；scaled 使用 `[0.0811730, 0.0643464, 0.05]`。

代码路径支持“只改 actor 输入尺度”的设计：`full_brain.py:72-85` 在 encoder 前使用 `policy.obs_std`；`value_observation.py:36-46` 只修改 actor 的 `obs_std[47:50]`，并要求新增 encoder 列权重仍为零；`ppo_yumi.py:193-201` 在切换 actor scale 后仍保留独立的 `ValueObservationStats`。运行记录也确认了这一点：control 和 scaled 的 `value_std[47:50]` 都是 `[1, 1, 1]`，scaled 的 `actor_std[47:50]` 才是候选尺度，三组的 `critic_coordinates_preserved` 均为 `true`。

因此，当前 control/scaled 差异不能简单归因于把新尺度误用到了已有 critic。它仍然是单个训练 seed 的短程在线实验。

## 配对评估结果

`paired.json` 是开发评估，使用同一组 3 个评估 seed、每组 32 个世界，汇总为每个条件 96 次尝试。`summary.json` 的结果如下：

| 条件 | 成功 | 跌倒 | 分数 | 平均持续时间 | 平均速度 | 平均横向偏移 | 平均高度 |
|---|---:|---:|---:|---:|---:|---:|---:|
| source | 10/96 | 5 | 0.412160 | 29.1765 s | 0.1790 | 3.7365 m | 0.8997 m |
| control | 12/96 | 8 | 0.398886 | 28.6155 s | 0.2074 | 3.9338 m | 0.8994 m |
| scaled | 10/96 | 5 | 0.329200 | 28.9542 s | 0.2228 | 5.2466 m | 0.8992 m |

相对于 control，scaled 的分数低 `0.069686`，成功少 2 次，平均速度高 `0.01535`，但平均横向偏移多 `1.31286 m`。分层结果也显示相同方向：scaled 在 `0.5` 目标速度层成功 `3/33`、横向偏移 `7.135 m`，control 为 `6/33`、`4.299 m`；scaled 在 `0.65` 层速度更高，但横向偏移仍更大。

三个配对评估 seed 中，scaled 相对 control 的分数差分别约为：

- seed `92001`：`-0.00968`；
- seed `92002`：`-0.08943`；
- seed `92003`：`-0.10995`。

方向在这三个评估 seed 上一致，但它们是评估重复，不是三个独立训练 seed。`summary.json` 也明确记录了这一限制。

## 训练过程与评估冲突

scaled 的训练内部指标没有显示明显崩溃：最终 rollout reward 为 `2.3194`，control 为 `2.2204`；最终 value loss 为 `97.56`，control 为 `146.16`；平均 actor KL 为 `0.00517`，control 为 `0.00504`。两组均未触发 KL 提前停止。

但是，配对开发评估的 96 次结果中 scaled 分数更低、横向漂移更大。说明短程 rollout reward、value loss 和小 KL 不能替代配对开发评估，也不能直接作为续训选择标准。

两个 `training.json` 还各自记录了另一套开发评估：control 为 `11/128`、分数 `0.383380`，scaled 为 `27/128`、分数 `0.427273`。这与 `paired.json` 的 96 次配对开发评估方向相反，说明两套评估使用了不同的评估样本或环境安排。做 A/B 判断时，应以同一 `paired.json` 评估集合为主，不能把两套数字混成一个结论。这个差异本身也提高了继续长训前复核的必要性。

## 新输入确实产生了在线效果

`verification.json` 证明两组都学习了新增 encoder 列：

- source 的 `new_encoder_rms` 为 `0`；
- control 为 `8.5961e-05`；
- scaled 为 `9.9103e-05`。

scaled 的新增列权重 RMS 约为 control 的 `1.15` 倍。更关键的是固定 source rollout 的敏感性检查：

- control 将新增输入置零后的动作 RMS 为 `1.0150e-05`；
- scaled 将新增输入置零后的动作 RMS 为 `1.8962e-04`，约为 control 的 `18.7` 倍；
- scaled 的 vx、vy、height 单位标准差动作扰动分别约为 control 的 `17.5`、`23.7`、`31.3` 倍。

因此可以排除“scaled 没有生效”以及“新输入仍然完全没有控制路径”。`verification.json` 同时标记 `critic_coordinates_preserved=true`，三组的 critic 新尺度均为 `[1,1,1]`。这些固定输入扰动是敏感性证据，不是步态成功证据。

## 是否支持直接续长训

### scaled

当前不支持把 scaled 直接作为默认路线长训。它确实提高了新输入对动作的敏感性，也提高了平均速度，但在相同训练 seed 和配对评估集合上分数下降、横向漂移明显增加，成功次数没有增加。现有结果更像“速度和横向稳定性之间发生了交换”，而不是已验证的行走改进。

### control

control 比 scaled 更适合作为保守基线，但也没有足够证据直接长训：它相对 source 的分数下降 `0.01327`，跌倒从 5 次增至 8 次，平均持续时间也略低。它可以作为后续复现实验的基线，不能称为已经改善。

所以本轮结果不支持无条件延长任一新分支。scaled 应保留为有明确假说的实验分支；control 应保留为比较基线。

## 被削弱和仍未否定的假说

### 已被削弱

1. **新输入完全没有学习路径。** 两组新增列都有非零权重和动作敏感性。
2. **输入尺度改动没有实际效果。** scaled 与 control 的新输入敏感性相差约一个数量级以上，说明改动已经到达动作输出。
3. **scaled 的主要问题是 critic 坐标被意外改变。** 本轮 `ValueObservationStats` 独立，记录中的 critic 坐标保持不变。
4. **新输入尺度必然带来更好的短期行走分数。** 配对评估相反，scaled 的分数低于 control 和 source。
5. **更高平均速度等于更好的行走。** scaled 速度更高，但横向漂移更大，综合分数更低。
6. **训练 reward/value loss 可以单独决定续训。** scaled 的内部指标较好，但配对评估较差。

### 尚未否定

1. 只有一个训练 seed，scaled 的劣势是否可重复仍未知。
2. 当前尺度候选是否过强，或三列是否应分别缩放，尚未测试。
3. scaled 的新路径是否需要更多 actor 更新才能形成稳定反馈，尚未测试；本轮只到 6 轮 actor 更新。
4. reward 对横向漂移的约束是否不足，尚未通过独立消融验证。
5. critic 的数值误差是否影响了两组长期学习，不能由最终 value loss 单独判断。
6. `neural_steps`、输入到输出的时间信用分配和物理接触稳定性没有在本轮 A/B 中分离。

## 最小可证伪的后续候选

下面是候选，不自动决定生产实现：

1. **训练 seed 复现。** 在完全相同的 7 轮预算下，只增加 2 个训练 seed，并使用同一组配对评估 seed。若 scaled 在多数训练 seed 上仍出现较低分数和较大横向偏移，可否定“当前尺度适合作为默认续训路线”。
2. **尺度剂量对照。** 保持 critic 坐标、奖励和训练预算不变，比较 control、当前候选和一个中间尺度。若中间尺度恢复分数且减少横向漂移，可否定“当前候选尺度是唯一合理尺度”。
3. **短程延长而非长训。** 只对 control/scaled 各追加少量相同 actor 更新，继续记录配对分数、横向偏移、跌倒率和新增列敏感性。若 scaled 的敏感性继续增大而横向漂移恶化，可否定“在这段追加预算内，scaled 会自动改善稳定性”；这不能否定更长训练预算的可能性。

在这些候选完成前，最稳妥的判断是：新输入和 actor scale 已被在线验证为有效，但当前尺度没有被验证为有利于行走。保留 control 作为基线，暂停 scaled 的默认长训推广。
