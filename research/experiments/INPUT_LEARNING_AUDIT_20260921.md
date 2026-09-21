# 新增观测学习审计（2026-09-21）

## 范围与状态

用户批准先检查新增速度、高度输入的影响，再决定长训配置。随后用户明确选择：先完成低内存分析，完整连接组对照留待云端。本轮未开云实例，未保存新策略权重，未修改训练配方。

已完成 CPU 观测、encoder 更新、Adam 状态和价值网检查。完整图单步对照曾在本地触发内存保护退出，**没有完成结果**，不能把其部分输出当作 A/B 证据。

以下路径均相对于 `runs/yumi_input_audit_20260921/`。主基线 SHA-256 为 `458fc465a6fb802a338c218fc4be414fc9153376c5d2f76f116518e6cc5aa475`，rollout 与配套 PPO 状态的哈希校验通过。

## 已确认的结果

### 1. 通路接通，新列确实在更新

`gpu_body.py` 与 `sim.py` 都追加三个字段，`full_brain.py` 将全部 50 维送入 encoder。扩维初始新列为零，统计量为 mean=0 / std=1。源 PPO 状态的策略 Adam slot 数为零，但短训后的 encoder slot 已有 307 / 768 步记录；恢复参数组成功不等于源文件含有历史策略动量。

`encoder_updates.json` 对迁移基线与两个短训检查点做了小张量比较：

| 配置 | 旧 47 列权重变化 RMS | 新 3 列权重变化 RMS | 保存的 Adam 步数 |
|---|---:|---:|---:|
| normfix | 3.00e-4 | 3.20e-4 | 307 |
| low_lr | 8.95e-5 | 9.37e-5 | 768 |

**新增列的绝对参数变化并不比旧列小一个数量级。** 因此不能把当前结果解释为新列没有接入优化器，或仅凭输入幅度断言其梯度导致参数不更新。

### 2. 新通道在编码后的贡献仍很小

`statistics_verified.json` 中新增观测的标准差为 0.08117、0.06435、0.04136，旧 std=1 没有放大这些输入。三列没有触及策略的 ±10 输入裁剪边界。

在同一批 4096 个观测上，分别计算旧列和新列对 encoder 输出的贡献 RMS：

| 配置 | 旧通道贡献 | 新通道贡献 |
|---|---:|---:|
| normfix | 2.52662 | 4.13e-5 |
| low_lr | 2.52661 | 1.26e-5 |

来源为 `encoder_updates.json`，复算脚本为 `check_encoder_updates.py`。这里比较的是 47 列与 3 列各自的总贡献，列数和已有权重大小都不同；不能当作逐特征梯度或生物重要性排名。

推断：新通道从零开始，短训后仍远弱于成熟旧通道。它们对当前网络的直接影响较小，与前轮固定输入敏感度结果相符。但这仍不能确定放大输入会改善身体控制。

### 3. Adam 会改变对尺度的直觉

同一 CPU 检查中的解析示例：两个零初始参数使用梯度 0.1 和 2.0，Adam 学习率为 2.5e-6，首次更新均约为 -2.5e-6。实际保存的 low_lr encoder Adam 方向 RMS，旧列为 0.17272，新列为 0.17232。

因此不能预设“输入放大约 20 倍，权重更新也放大约 20 倍”。后续应测量梯度、Adam 后增量、输入到动作的敏感度和真实步态，不能只看 encoder 权重大小。

### 4. 直接修改共享归一化会破坏价值估计

`ppo_yumi.py` 将相同的 `policy.obs_mean/obs_std` 用于策略和价值网。本批次只将新尺度改为 `[0.0811730, 0.0643464, 0.05]`、保持均值为零，CPU 结果如下：

| 价值网条件 | 对同批 bootstrap 目标的 MSE |
|---|---:|
| 原尺度 | 326.11 |
| 直接改共享尺度 | 111,918.81 |
| 新尺度 + 第一层解析补偿 | 326.11 |

解析补偿采用 `W_new = W_old * new_std / old_std`，只改第一层输入坐标；预测最大绝对误差为 6.10e-5。见 `statistics_verified.json`。

这是初始函数等价检查。**只补偿第一层权重，不能保证后续 Adam 更新等价**，更不能任意重置价值优化器而继续称单变量对照。MSE 的目标仍来自同一批 bootstrap 回报，不代表真实长期回报误差。

## 代码与验证

- `app/probe_observation_learning.py`：默认完整图固定批次对照；`--statistics-only` 只读取 CPU 数据和价值网。检查来源哈希、零初始化前提、时间和内存预算；不运行物理，不保存权重。
- `app/test_observation_probe.py`：CPU 小图验证初始策略等价、价值网解析补偿、恢复后的学习率与 Adam 缩放示例，已通过。
- `check_encoder_updates.py`：读取已保存的 encoder 和 Adam 小张量，已执行成功。

CPU 复现命令（仓库根目录，环境已按本地指南安装）：

```powershell
.venv-gpu\Scripts\python.exe -m app.probe_observation_learning --checkpoint runs/yumi_calibration_20260921/migrated/runs/yumi_obs50/best.pt --state runs/yumi_calibration_20260921/migrated/runs/yumi_obs50/ppo_state_best.pt --rollout runs/yumi_calibration_20260921/rollout.pt --output runs/yumi_input_audit_20260921/statistics_recheck.json --statistics-only
.venv-gpu\Scripts\python.exe -m app.test_observation_probe
.venv-gpu\Scripts\python.exe -m runs.yumi_input_audit_20260921.check_encoder_updates
```

这些是对已保存实验产物的复核命令，需要前轮迁移包和 rollout；不是从空仓库开始训练的说明。大文件仍不入 Git，未建立公开发布渠道，也未推送仓库。

## 云端待执行：固定批次的单次更新对照

具体协议见[综合讨论](../proposals/2026-09-21/Codex-新增观测对照与价值网隔离.md)。已经准备命令，但未成功执行完整图部分：

```bash
.venv/bin/python -m app.probe_observation_learning --checkpoint runs/yumi_obs50/best.pt --state runs/yumi_obs50/ppo_state_best.pt --rollout runs/yumi_calibration_20260921/rollout.pt --output runs/yumi_input_audit_20260921/one_update_cloud.json --device cuda --samples 64 --microbatch 16 --max-seconds 180
```

A/B 使用相同的 64 条观测、动作和固定优势值，各从同一策略及优化器状态执行一次更新。B 只改变 actor 新输入尺度；由于固定优势值且不更新价值网，不引入 critic 重训差异。记录初始动作一致性、每列梯度与参数增量、新通道动作贡献及固定标准差下的 KL。它只检查局部学习机制；不检验行走成功，也不是恢复云端 PPO 训练。

若固定对照表明尺度会改变有意义的动作响应，再提出有独立 critic 输入统计量的在线短训。当前没有预设必须晋升尺度方案，亦不据本轮 CPU 结果批准长训。
