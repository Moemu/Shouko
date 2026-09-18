# Yumi 直立步态优化：2026-09-16 实验记录与当前状态

## 目标

解决 ppo9 最佳模型（score 0.853，16/16 验收）的"深蹲 < 形"步态：
骨盆高 0.82–0.84 m（直腿应 ~0.96 m）、膝角 0.7–1.3 rad、前倾、小碎步。

## 病因诊断（已验证）

1. **奖励无姿势项**：原奖励只含速度跟踪/直立/存活/换脚，无任何偏好直腿的项。
2. **home 姿态继承 G1 屈膝**：`yumi.yaml` 的 `default_angles` 膝 0.3 rad、髋 -0.1、踝 -0.2，
   PD 目标 = home + 0.25·action，零动作即半蹲；action 正则还在惩罚伸直所需的持续偏置。
3. **PPO 在最优点固有不稳定（关键发现）**：从收敛策略出发微调，无论学习率
   （2e-5~1e-4）、探索噪声（0.08~0.4）、有无姿势奖励，**10~15 轮后必崩**。
   ppo9 历史（第 10 轮达峰后再未超越）与此完全吻合。根因：2558 万突触边
   全量更新在小批量 PPO 下冲散循环网络吸引子（外部独立分析结论一致）。
4. **高斯探索噪声惩罚直立**：直腿高重心是窄鞍点，深蹲是宽势能谷，
   std 0.2~0.4 的噪声环境下 PPO 天然偏好深蹲自保。

## 实验编年（云端 4090D，runs/yumi/ppo*.log）

| 轮次 | 配置 | 结果 |
|---|---|---|
| ppo10 run1 | 姿势奖励（高斯高度项 σ=0.05）lr 1e-4 | 5 轮内 score 0.87→-0.17 |
| ppo10 run2 | +值预热 10 轮 lr 5e-5 | 撑到 iter 10 后崩至 0.3（站立不动 hack） |
| ppo10 run3 | +行走门控 walk_gate | 崩解变慢但仍崩（iter 25） |
| ppo10 run4 | +std 0.2 / lr 2e-5 / epochs 1 | 同样 iter 15~20 崩；发现高斯项在工作点梯度≈0 |
| ppo10 run5 | 高度项改线性斜坡 | 仍崩 → 怀疑非奖励问题 |
| ppo10 ctrl | **零姿势权重对照** | 同样 iter 10 峰、iter 25 崩 → 确诊固有漂移 |
| ppo10 burst | 短脉冲+棘轮（score+身高选择） | 质量锁在 0.87~0.88，但身高永远 0.83（抽奖无进展） |
| ppo11 | **冻结突触边+偏置**（仅 40k 读出层）std 0.08 | 40+ 轮不崩（score 稳定 0.85±0.05），但身高纹丝不动 |
| ppo12 | 冻结 + **home 直立化**（膝 0.08）| 从跌倒恢复：0.90 m 高走 29 s，但 score 卡 0.2~0.3（容量不足） |
| ppo13 | ppo12 成果为种子，**全网络**脉冲 | 慢升但被外部报告叫停（直立极限环脆弱，必崩风险） |
| ppo14 | 冻边 + **开 166k neuron_bias** lr 5e-5 | 稳定窗扩到 25~30 轮，iter 25 创 0.397 后崩（偏置 lr 过大） |
| ppo15 | 冻边 + 偏置 **lr 1e-5** + 纯净 Adam + 170 s 脉冲 | **稳定棘轮上升**：0.418→0.44→0.48→0.49，successes 7 |

## 当前最佳模型（ppo15c，截至 13:57）

- 云端 `runs/yumi/best.pt`（本地副本 `runs/yumi/best_tall.pt`，hash 666134f2…；
  云端另有 `best_08s_tall.pt` 同哈希备份）
- 指标：score 峰值 0.525、successes 最高 9/32、**骨盆高 0.90 m**（+7 cm）、走满 30 s
- 短板：速度仅 ~0.2 m/s（目标 0.35~0.65），成功率因此受限
- 选择指标：combined = score + 0.5·clamp((height−0.80)/0.13)，当前棘轮值 0.908

## 午后进展（13:00–14:00）

### 步频 1.0 s 实验（另一 Agent 发起，已淘汰）

另一 Agent 将 `gpu_body.py` 相位周期 0.8 s→1.0 s（ppo16_clock10），
并把 best.pt 重置为 1.0 s 检查点跑脉冲（burst16_accum，8 轮）。
最终对比：

| | 0.8 s 线（ppo15） | 1.0 s 线（ppo16） |
|---|---|---|
| 棘轮 combined | **0.871** | 0.802 |
| score | **0.48~0.49** | 0.17~0.39 |
| successes | **7/32** | 0~6/32 |
| 速度 | 0.18 m/s | 0.227 m/s |

结论：1.0 s 时钟仅换得 +0.05 m/s 速度，score 全面落后（观测分布重置代价过大），
**已淘汰并回退**：best.pt 恢复为 `best_08s_tall.pt`（哈希校验一致），
云端与本地 `gpu_body.py` 相位均改回 0.8 s。1.0 s 期间文件保留在
`ppo16_clock10*.log`，备份 `best_08s_tall.pt` / `ppo_state_08s_tall.pt`。

### ppo15c（回退后第三轮 12 脉冲）：平台期确认

- 第 1 轮即冲到 score 0.525 / successes 7（新棘轮 0.908），但后续 11 轮未能超越
- 各脉冲基线 0.31~0.47 震荡，successes 最高 9/32，身高稳定 0.90 m
- **结论：冻边 + 166k 偏置 + 当前奖励结构已榨干；姿势目标达成，速度跟踪需要新的梯度信号**

### 下一步候选（按风险排序）

1. **加重速度奖励**（推荐）：vx 跟踪项 1.5→2.5，训练指令速度收窄到 0.4~0.75
   （逼出快走，放弃慢速舒适区）；不动观测分布，脉冲机制直接复用
2. 解冻部分突触边 + 极小学习率（风险高，历史上 10~15 轮必崩，暂不建议）

## 代码改动（app/ppo_yumi.py、app/train_full.py、yumi.yaml）

- `ppo_yumi.py`：姿势奖励（行走门控 + 线性高度斜坡 + 膝角软惩罚）、
  `--posture-w/--knee-w/--height-target/--posture-select-w`、`--std-override`、
  `--freeze-brain`（冻突触边，训 bias lr·0.1 + readout lr + encoder lr·0.5）、
  优化器状态跨结构容错加载、日志记 mean_knee/mean_height、
  best 选择改用 combined 指标（含身高）
- `train_full.py`：evaluate 增加 mean_height 统计
- `yumi.yaml`：`default_angles` 直立化（±0.03/0/0/0.08/-0.05/0），initial_height 0.968
  （本地已备份 `yumi.yaml.bak`，云端备份 `yumi.yaml.bak`）

## 运行中状态（截至 13:57）

- 云端 AutoDL（`ssh -p <端口> root@<AutoDL 主机>（私有端点已脱敏，配置见 runs/cloud/target.json，不入库）`）：
  ppo15c 已跑完，当前无训练进程；实例仍开机计费中
- 本机 8743 预览：直立模型（score 0.49 版，与云端最新 best 差一轮 ppo15c 的 0.525，
  如需一致可重新拉回覆盖 `runs/yumi/best.pt`，服务器热加载自动生效）

## 下一步

1. 等待确认后执行"加重速度奖励"方案（vx 权重 1.5→2.5 + 指令速度 0.4~0.75），
   复用 burst15.sh 脉冲机制跑 12 轮
2. 若 successes 稳定 10+/32：拉回本机、重跑 `evaluate_full --robot yumi`
   完整 16 项验收（确认鲁棒性未退化）、更新 YUMI_BODY.md、预览对比
3. 全部达标后关闭云端实例（计费中）

## 复现命令

```bash
# 云端脉冲训练（当前配置）
.venv/bin/python -m app.ppo_yumi --resume runs/yumi/best.pt --resume-state \
  --freeze-brain --lr 1e-4 --std-override 0.08 --epochs 4 --target-kl 0.02 \
  --value-warmup 6 --max-seconds 170 --eval-every 5
```

## 注意事项

- 跨 home 变更后旧 best 分数门槛失效：需以新 home 的模型重设基准（ppo13 已处理）
- 改优化器参数结构后不要恢复旧 Adam 动量（代码已自动跳过并打印 policy_optimizer: false）
- SSH 远程杀进程勿用 pkill -f（会匹配自身 ssh 命令行）；用
  `ps -eo pid,comm,args | awk '$2=="python" && /ppo_yumi/ {print $1}' | xargs -r kill`
- 旧深蹲模型备份：本地 `runs/yumi/best.pt.bak` + `ppo_state_best.pt.bak`（hash a7a4281f…）
