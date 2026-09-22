# 固定单步对照后的输入学习诊断

日期：2026-09-22

作者：Codex-independent（`/root/input_audit`）

## 独立性与范围

本提案未参考其他 2026-09-22 提案，也未参考主 Agent 的后续结论。分析只使用了以下生产代码和本次诊断数据：

- `app/full_brain.py`
- `app/ppo_yumi.py`
- `app/probe_observation_learning.py`
- `runs/yumi_input_diagnostics_20260922/update256_v2.json`
- `runs/yumi_input_diagnostics_20260922/update64_v2_20260921.json`
- `runs/yumi_input_diagnostics_20260922/update64_v2_20260922.json`
- `runs/yumi_input_diagnostics_20260922/update64_v2_20260923.json`
- `runs/yumi_input_diagnostics_20260922/reachability.json`
- `runs/yumi_input_diagnostics_20260922/routes256.json`
- `runs/yumi_input_diagnostics_20260922/routes64.json`

本独立审查没有另行运行 GPU、云端命令或物理仿真；所引用的固定单步、route 和 reachability 结果已经由今天的云端 GPU 任务完成并保存。下面的“确认”只表示这批固定离线数据直接支持的事实，不表示已经完成行走训练。

## 实验实际比较了什么

`ConnectomePolicy.forward()` 先按 `obs_mean/obs_std` 归一化观测，再将 50 维输入送入输入神经元，经过 `neural_steps` 次稀疏神经更新，最后由输出神经元和 `readout` 产生动作（`app/full_brain.py:72-85`）。当前输入 47、48、49 的检查点权重为零，因此改变它们的归一化尺度不会改变初始动作；它只会改变这些列在之后更新中的梯度。

探针用同一份固定 rollout、同一份 action/advantage 和同一套参数，比较 `original` 与 `new_inputs_scaled` 各一次 actor 更新（`app/probe_observation_learning.py:157-203`）。四份 `update*v2*.json` 使用相同的 checkpoint、rollout、state 哈希，学习率均为 `5e-6`，`source_optimizer_slots` 均为 0。它们是固定批次的离线单步，不是 on-policy 续训。

## 证据

### 1. 新输入确实有梯度，当前数据不支持“输入断路”

三个新输入的旧尺度都是 `1`。按本批数据计算的新尺度是：

| 输入列 | 原尺度 | 新尺度 | 原归一化标准差 | 新归一化标准差 |
|---|---:|---:|---:|---:|
| 47 | 1 | 0.0811730 | 0.0811730 | 1.0000 |
| 48 | 1 | 0.0643464 | 0.0643464 | 1.0000 |
| 49 | 1 | 0.0500000（下限） | 0.0413568 | 0.8271 |

四份结果中，47、48、49 三列的 encoder 梯度非零比例在两种条件下都为 `0.8501384258`。`reachability.json` 的四步结构上限也给出 `14738 / 17336 = 0.8501384402` 个输入行可达。两者在小数误差内相同。这支持以下解释：当前约 15% 的零梯度来自四步展开的结构可达范围，而不是这三个新输入完全没有反向路径。

它仍然不是生理可达性或控制质量证明。`reachability.json` 明确说明它是在零初始状态下的结构上界；非线性增益、输出抵消和时间上的信用分配仍可能削弱实际控制效果。

### 2. 尺度改变按预期放大了原始梯度，但没有按同样倍数放大 Adam 更新

以 `update64_v2_20260922.json` 为例，`new_inputs_scaled / original` 的新列梯度 RMS 比例为：

- 第 47 列：`12.31937`，对应 `1 / 0.0811730 = 12.31937`；
- 第 48 列：`15.54084`，对应 `1 / 0.0643464 = 15.54088`；
- 第 49 列：`20.00002`，对应 `1 / 0.05 = 20`。

这说明归一化尺度确实进入了 actor 的反向路径。它不能被解释为“新特征没有被学习”。

但是，`source_optimizer_slots=0`，探针从没有 Adam 历史槽位的状态开始。当前 optimizer 中 encoder 学习率为 `lr * 0.5`（`app/ppo_yumi.py:155-163`；探针在 `app/probe_observation_learning.py:43-57` 复现这一分组）。在这种首步 Adam 更新中，梯度的共同倍数大部分会被一阶/二阶矩归一化抵消；全局裁剪也发生在 Adam 之前。四份结果中新列 encoder 更新 RMS 为：

- `original`：约 `1.76e-6` 至 `1.83e-6`；
- `new_inputs_scaled`：约 `2.02e-6` 至 `2.06e-6`。

因此，当前数据支持“尺度使新列原始梯度放大约 12 至 20 倍”，不支持“首步参数更新也放大 12 至 20 倍”。将后者作为预期会误判 Adam 首步。

### 3. 总梯度主要由 readout 和 neuron_bias 组成

四份 v2 结果的参数组梯度范围（两种条件合并）为：

| 参数组 | 梯度范数范围 |
|---|---:|
| `readout` | `71.58`–`126.90` |
| `neuron_bias` | `17.99`–`31.39` |
| `encoder` | `10.56`–`22.06` |
| `log_std` | `0.385`–`0.641` |

总范数约为 `74.92`–`132.57`，全局裁剪倍率约为 `0.0377`–`0.0667`。切换新尺度后，`readout`、`neuron_bias` 和 `log_std` 几乎不变，encoder 范数只小幅增加。例如 `update64_v2_20260922.json` 中，encoder 为 `21.2270 -> 21.2632`，readout 为 `104.8216 -> 104.8216`，总范数为 `110.1835 -> 110.1904`。

这解释了为什么总动作变化没有明显改变：当前单步的主要更新方向来自旧路径和 readout。它不是说新列没有作用，也不能由参数组梯度范数直接推出长期步态贡献。

### 4. 新列的动作影响变大，但绝对量仍很小

在 `update64_v2_20260922.json` 中：

- 总 `action_rms_change`：`0.00073591 -> 0.00073926`；
- 新列单独造成的 `new_channel_action_rms`：`3.50e-7 -> 5.12e-6`；
- 旧路径动作变化：约 `0.00073569 -> 0.00073568`；
- 固定标准差下的 KL：`0.00050772 -> 0.00051234`。

`routes64.json` 和 `routes256.json` 的单参数组回放给出相同关系：readout 的动作 RMS 约 `5.61e-4`，旧 encoder 约 `1.55e-4` 至 `1.65e-4`，新 encoder 在缩放后约 `5.10e-6` 至 `6.24e-6`。这些 route 值是把同一次 Adam 增量单独施加到初始 actor 的结果；记录本身说明非线性作用不能相加，也不能当作步态贡献百分比。

所以当前最稳妥的表述是：新尺度让新列在这一步的动作扰动增加了约一个数量级，但扰动仍小于旧路径和 readout 的单组动作变化。它可能影响多步训练，也可能被信用分配、readout 或物理反馈淹没；本数据不能在两者之间定论。

### 5. 共享 `obs_std` 会让完整 PPO 对照混入 critic 变化

`ppo_yumi.py` 中 actor 和 `Value` 都使用 `(obs - policy.obs_mean) / policy.obs_std`（`app/ppo_yumi.py:305-310`、`418-422`）。探针的 critic 检查直接量化了这个混杂：

- 原尺度 critic MSE：`326.1148444`；
- 直接使用新共享尺度：MSE `111918.8053`，输出 RMS 改变 `335.0089`；
- 只把 `Value.net[0].weight` 乘以 `new_scale / old_scale` 后再用新尺度：MSE `326.1148459`，与原输出的最大绝对误差 `6.10e-5`。

这证明尺度坐标变换可以保持 critic 函数不变，但直接改共享 buffer 会改变已有 Value。因而，未来若在完整 PPO 中直接替换共享 `obs_std`，A/B 同时测试了 actor 和 critic，不能称为纯 actor 输入尺度实验。当前探针的 actor 单步使用固定 advantage，没有更新 Value；上面的 critic 结果是独立的解析检查，不能当作已经完成的 on-policy critic 训练。

## 可以排除、不能排除的瓶颈

### 当前证据可以排除或强烈反驳

1. **新列完全没有梯度。** 三列均有非零梯度，且非零比例与四步结构可达比例一致。
2. **新列没有进入归一化路径。** 梯度 RMS 按新尺度的倒数精确放大。
3. **encoder 没有被 optimizer 纳入。** 探针和生产 PPO 都把 `encoder.weight` 放入独立 optimizer 组，并且新列产生了非零 Adam 增量。
4. **直接共享新尺度仍能保持原 critic。** 数据相反：直接共享新尺度会严重改变 Value；需要做第一层权重重参数化或保持 critic 的旧坐标。

### 当前证据不能排除

1. **新列长期 on-policy 学习是否能提高行走。** 这里只有一个固定 rollout 和一次离线更新。
2. **四步神经展开是否足够支持行走控制。** 结构 reachability 只给上界，不能证明时间信用分配或动作质量。
3. **readout、旧 encoder、neuron_bias、critic 或物理奖励是否是长期瓶颈。** 单步 route 结果只说明当前初始化和学习率下的局部动作敏感度。
4. **新输入尺度是否在新 rollout 上泛化。** 新尺度来自这一个诊断批次，没有 held-out 数据。
5. **固定步的微小动作差异是否会经过多轮反馈累积。** `action_rms_change` 和 KL 很小，只能说明这一首步局部变化小。

## 最小下一步诊断

固定 rollout 的单步 paired probe 已经完成，不应重复运行。真正尚未完成的是多步 on-policy 学习和在线 A/B 闭环；下面只列出控制项，不自动决定生产实现：

1. 如果进入短程在线 A/B，使用同一初始 checkpoint、相同环境随机种子和相同世界数；只改变 actor 的输入尺度条件。
2. 对 new-scale 条件保持 critic 函数不变：要么 Value 继续使用旧尺度，要么在切换前按 `Value.net[0].weight *= new_scale / old_scale` 做等价重参数化。不要把共享 `obs_std` 的直接替换当作纯 actor 对照。
3. 先用少量 on-policy 更新观察 `mean_duration`、跌倒率、动作 RMS、Value 输出误差和 KL，再决定是否值得进入长训；这一步仍不能宣称完成行走。
4. 若要判断神经时间范围，再单独改变 `neural_steps` 做固定批次或短程对照，把它作为结构/信用分配诊断，不能与输入尺度效果混为一谈。

今天云端 GPU 已完成的 `update*v2*.json`、`reachability.json` 和 `routes*.json` 已足够说明：新输入能反向学习，输入尺度确实生效，但首步 Adam 后的新路径动作影响仍很小。下一步若继续，应先控制 critic 坐标，再用短程多步 on-policy A/B 检查反馈是否累积；这仍是实验提案，不是生产实现决定。
