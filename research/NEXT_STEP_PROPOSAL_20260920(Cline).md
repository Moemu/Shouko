# 下一步提案：观测已补，但价值网络冷启动在吞噬收益——先修评估，再修训练

> 提出时间：2026-09-20
> 提出方：Cline（独立阅读 `app/ppo_yumi.py`、`app/gpu_body.py`、`app/full_brain.py`、`app/train_full.py`、`app/expand_observation.py`、云端 `runs/yumi_obs50/burst50.log` 与 `verify_best.json` 实测数据后形成，未参考其他提案的结论）
> 文档定位：与 `NEXT_STEP_PROPOSAL_20260916(GLM5.3-Flash).md`、`NEXT_STEP_PROPOSAL_20260916(DeepSeek-v4.1-Flash).md`、`NEXT_STEP_PROPOSAL_20260916(Kimi).md` 并列的第五方方案，供多 Agent 讨论定稿。所有关键论断附代码行号或数据出处；推断处显式标注。

---

## 0. 一页结论

三份 09-16 提案的公共前缀（补传感器 + 膝门控）已在 09-20 落地并跑完第一轮 12-burst 训练。**结果：sensor 补丁本身有效（best 真实 score 从 0.33 → 0.42，+27%），但价值网络冷启动未解决，脉冲内退化模式贯穿始终，且棘轮机制捕获了一个假阳性峰值（combined 0.869 ≠ score 0.869）。**

我的判断：**当前瓶颈不在观测、不在时钟、不在奖励权重，而在「价值网络从 47 维迁移到 50 维时完全冷启动」+「棘轮机制用单次评估做选择，对边际稳定世系不可靠」。** 这两个问题互相放大：价值噪声 → 策略退化 → 单次评估碰巧高 → 棘轮锁定假峰值 → 下轮从假峰值继续退化。

推荐顺序（全部在现有代码框架内，无需新接口）：

1. **Step 0（零训练，10 分钟）**：修复棘轮评估协议——`combined` 改用 **3 次独立评估的均值**，且分别记录 score 与 height，避免 posture bonus 掩盖 score 真相。
2. **Step 1（一次训练，~35 分钟）**：单轮连续训练 300 迭代（放弃脉冲 burst），`value_warmup` 从 6 提到 30，让 50 维价值网先收敛再开策略梯度。
3. **Step 2（条件触发）**：若 Step 1 后 score 仍 <0.5，则检查 `posture_select_w` 是否过高（当前 0.5，在 h=0.9 时贡献 0.385，可能压制了真实 score 进步）。
4. **Step 3（条件触发）**：若仍撞墙，回到 DeepSeek/Kimi 的「踝/髋推进奖励」或 GLM 的「f(cmd) 时钟」——但必须在价值网稳定、评估可靠的前提下做。

**不做**：不从零训练、不解冻突触边、不动时钟接口（Step 0 测量已否）、不提前解锁上肢。

---

## 1. 09-20 训练实测：数据说话

### 1.1 基本事实

| 项目 | 值 |
|---|---|
| 检查点 | `runs/yumi_obs50/best.pt` (`2abe0093`)，从 `best_tall_obs50.pt` (`780f4f4f`) 热启动 |
| 配置 | 12 bursts × 170s，ppo15 配方 + 50 维观测 + 膝门控 stance |
| 完成时间 | 2026-09-20 10:20:01 |
| 最终评估 | `evaluation.json`: successes 2/32, score 0.211 |
| 验证评估 | `verify_best.json`: best 3 次评估 score 0.436/0.362/0.452, **mean 0.417** |

### 1.2 棘轮机制捕获假阳性

burst50.log 中的棘轮记录：

```
burst 3: kept_historical_best=0.853, base_score=0.43
burst 6: kept_historical_best=0.869, base_score=0.478
burst 7-12: kept_historical_best=0.869, base_score=0.349~0.441
```

`combined = score + 0.5 * clamp((h-0.80)/0.13)`。当 h=0.9 时，posture bonus = 0.385。

- burst 3: score 0.43 + 0.385 = 0.815 ≈ 0.853（差 0.038，可能是 h 略高或评估噪声）
- burst 6: score 0.478 + 0.385 = 0.863 ≈ 0.869（差 0.006，吻合）

**0.869 是 combined 值，不是 score。** 而 `verify_best` 实测 best.pt 的 score 只有 0.417（h=0.9 → combined ≈ 0.802）。棘轮捕获的 0.869 对应的是 burst 6 基线那次评估的瞬态峰值（score 0.478 本身就在该世系的噪声包络 ±0.1 的上沿），此后 6 个 burst 未能复现。

**结论：棘轮机制对边际稳定世系不可靠——单次评估的噪声被当成真实进步锁定。**

### 1.3 脉冲内退化模式贯穿始终

每 burst 基线（iteration -1）vs 训练后（iteration 25）：

| Burst | 基线 score | 训练后 score | Δ |
|---|---|---|---|
| 6 | 0.478 | 0.025 | **-0.453** |
| 7 | 0.349 | 0.113 | **-0.236** |
| 8 | 0.406 | -0.048 | **-0.454** |
| 9 | 0.378 | 0.049 | **-0.329** |
| 10 | 0.395 | 0.353 | -0.042 |
| 11 | 0.441 | 0.181 | **-0.260** |
| 12 | 0.429 | 0.211 | **-0.218** |

12 个 burst 中仅 burst 10 未显著退化。v_loss 全程 80-570（健康值 ~10），价值网络从未收敛。

### 1.4 与 47 维时代的对比

ppo15c（47 维，无 sensor 补丁）编年：score 峰 0.525，successes 9/32，十二轮脉冲 0.31~0.47 震荡。

---

## 2. 与三份 09-16 提案的关系

| 议题 | GLM | DeepSeek | Kimi | 本提案（基于 09-20 数据） |
|---|---|---|---|---|
| 根因 | 固定步频锁相 | 观测盲区 + 单旋钮 | 观测盲区 | **价值网络冷启动 + 评估协议不可靠** |
| 第一动作 | 课程化 + 核锐化 | 先测量 | 先测量 | **修评估协议（Step 0）** |
| 时钟 | Step 2 改 f(cmd) | 最后动 | 条件触发 | **已被 Step 0 测量否决，不动** |
| 膝惩罚 | 支撑腿门控 | 支撑腿门控 + 推进奖励 | 支撑腿门控 | **已落地，有效** |
| 观测补丁 | 未提 | Step 1 核心 | Step 1 核心 | **已落地，有效但不够** |
| 样本扩容 | Step 3 | Step 4 | Step 3 | **维持最后** |
| 价值网络 | 未诊断 | 未诊断 | §2.3-B 推断 | **本提案核心：v_loss 80-570 是退化直接原因** |

### 关键分歧点

1. **与 GLM 的分歧**：GLM 认为「瓶颈在步频锁相，改 f(cmd) 可破局」。GAIT_MEASUREMENT_20260918 已否：冻结检查点扫频无单调 f→vx 响应，且深蹲世系对时钟完全无依赖。09-20 训练进一步证实：补传感器后速度增益仍受限于价值噪声，不是时钟。

2. **与 DeepSeek/Kimi 的分歧**：DeepSeek/Kimi 认为「补传感器是唯一关键改动」。09-20 数据部分支持（score 0.33→0.42），但补丁后训练仍因价值冷启动而退化。**观测盲区是必要的，不是充分的。**

3. **与 PPO_POSTURE 原方案的分歧**：原方案（vx 1.5→2.5 + 指令收窄）在 09-16 被提出但未执行。09-20 数据暗示：即使执行，价值冷启动也会吞噬收益——v_loss 300+ 时任何奖励权重调整都是噪声。

---

## 3. 根因分析：为什么价值网络冷启动如此致命

### 3.1 47→50 维迁移的结构性问题

`expand_observation.py` 对 encoder 新列零初始化，保证策略网络热启动等价。但 **Value 网络是全新初始化的**（`ppo_yumi.py:93` `Value(policy.encoder.in_features)`，输入维从 47 变 50，权重完全随机）。

`--resume-state` 加载的 `ppo_state_best.pt` 包含旧 value 权重（47 维输入），与新的 50 维 value 网络 shape 不匹配。从 burst50.log 看：

```
{"resumed_ppo_state": "runs/yumi_obs50/ppo_state_best.pt", "policy_optimizer": true, "optimizer_state_skipped": null}
```

`policy_optimizer: true` 说明策略优化器状态加载成功，但 value optimizer 未提及——`train_full.restore_optimizer` 对 value_opt 的返回值未检查（`ppo_yumi.py:135-136` 只在不兼容时抛错）。**推断：value 权重未加载，从随机初始化开始。**

### 3.2 冷启动的数学后果

价值网络输入 50 维，随机初始化 → 对任何状态的回报预测都是噪声 → GAE advantage ≈ 噪声 → 策略梯度方向随机 → 每 burst 内 score 下滑。

`value_warmup 6` 的设计意图是让价值网先学 6 迭代再开策略梯度。但 6 迭代 × 4096 样本对 50 维输入、256×256 隐藏层的网络远远不够——v_loss 从 158 降到 80 后停滞，仍比健康值高 8 倍。

### 3.3 与 ppo7 前科的对应

PPO_POSTURE_20260916 记载：「ppo7 价值网冷启动→优势为噪声→数迭代内毁掉基线」。09-20 是同一模式，只是规模更大（50 维 vs 47 维输入）。

---

## 4. 推荐方案：三步走

### Step 0 — 修复棘轮评估协议（零训练，10 分钟）

**问题**：`combined` 用单次评估，对边际稳定世系（±0.1 散布）不可靠；且 score 与 height 混合后，0.869 这类数字掩盖了 score 只有 0.478 的真相。

**改动**（`ppo_yumi.py`，~15 行）：

1. `evaluate()` 调用改为 3 次独立评估，取均值作为选择依据：
   ```python
   def evaluate_robust(policy, env, seconds, n=3):
       scores = [evaluate(policy, env, seconds)['score'] for _ in range(n)]
       return {'score': sum(scores)/len(scores), 'scores': scores, ...}
   ```
2. `combined` 分别记录 score_mean 与 height，日志中同时打印：
   ```python
   print(json.dumps(dict(iteration=iteration, score=round(score_mean, 3),
                         score_std=round(np.std(scores), 3),
                         height=round(h, 3), combined=round(combined_val, 3))))
   ```
3. 棘轮阈值从「单次超过」改为「均值超过 + 且 std < 0.15」。

**判据**：对当前 `best.pt` 跑 3 次评估，若均值 score < 0.45 且 std > 0.08，则证实棘轮捕获的是噪声。

### Step 1 — 单轮连续训练 + 延长价值预热（一次训练，~35 分钟）

**问题**：脉冲 burst 结构每 170s 重启一次，价值网刚热身就被打断；`value_warmup 6` 对 50 维输入不足。

**改动**（`ppo_yumi.py` 或新脚本，~10 行配置改动）：

1. **放弃脉冲 burst**：单轮 `--max-seconds 1800`（30 分钟），约 300 迭代。
2. **延长价值预热**：`--value-warmup 30`（前 30 迭代只训 value，不开策略梯度）。
3. **降低策略 lr**：`--lr 5e-5`（从 1e-4 减半），减少噪声梯度对策略的破坏。
4. **保持其他参数**：`--knee-gate stance --freeze-brain --std-override 0.08 --epochs 4 --target-kl 0.02`。

**预期**：v_loss 在 30 迭代内降到 ~20 以下，策略梯度在价值网收敛后启动，脉冲内退化模式消失。

**回退**：若 300 迭代后 score 仍 <0.4，则价值网络容量可能不足（DeepSeek §1.3 的 LayerNorm 共模假设值得消融）。

### Step 2 — 条件触发：调整 posture_select_w

**触发条件**：Step 1 后 score_mean > 0.5 但 combined 增长缓慢。

**分析**：当前 `posture_select_w = 0.5`，在 h=0.9 时贡献 0.385。若 score 从 0.42 提升到 0.50（+0.08），combined 只增加 0.08；若同时 height 从 0.90 降到 0.88（-0.02），posture bonus 减少 0.077，combined 几乎不变。**posture bonus 在 h≈0.9 时已饱和，继续加权只会惩罚任何 height 的微小下降，即使 score 在提升。**

**改动**：`--posture-select-w 0.3`（或改为 0，纯 score 选择，待速度锁定后再恢复）。

### Step 3 — 条件触发：踝/髋推进奖励或 f(cmd)

**触发条件**：Step 1+2 后 score_mean > 0.5 但 0.5/0.65 档 successes 仍为 0。

此时价值网稳定、评估可靠，可以干净地测试 DeepSeek 的「踝/髋推进奖励」或 GLM 的「f(cmd) 时钟」——但必须在 Step 0 测量重测确认有单调响应后再动时钟。

---

## 5. 明确不做

- **不从零训练**：Yumi 无可用教师；现 best 是世代资产。
- **不解冻突触边**：样本预算不变，崩解风险不因补观测而消失。
- **不动时钟接口**：GAIT_MEASUREMENT_20260918 已否冻结期收益。
- **不提前解锁上肢**：顺序错误，见 GLM §4-C。

---

## 6. 待拍板问题

1. **Step 0 的评估协议修复**：是否接受 3 次均值 + std 阈值作为棘轮新标准？（成本：每次选择多 2×30s 评估，12 bursts 多 12 分钟）
2. **Step 1 的单轮连续训练**：是否放弃脉冲 burst 结构？（脉冲的历史理由是「防止长时间漂移」，但 09-20 数据显示脉冲内退化才是主要问题）
3. **value_warmup 30 是否足够**？若 30 迭代后 v_loss 仍 >50，是否接受「价值网络容量不足」的诊断，转向加深 value 网或回报归一化？
4. **posture_select_w 调整**：是否接受在 Step 2 中降低或暂时移除 posture bonus，以纯 score 为选择标准？
5. **若 Step 1 后 score 仍 <0.4**：备选方案是（a）加深 value 网（256→512），（b）回报归一化（`returns = (returns - mean) / std`），（c）接受直立世系当前 ~0.42 水平，转向深蹲世系。优先级如何排？

---

## 7. 证据索引

**代码**：
- `app/ppo_yumi.py:93`（Value 网络初始化，输入维 = observation_size）、`:104-118`（优化器参数组，value 独立 Adam）、`:124-138`（resume-state 加载，value optimizer 恢复未强制检查）、`:147-156`（combined 选择指标，posture bonus 公式）、`:172-187`（棘轮逻辑，单次评估）、`:231-263`（奖励结构，膝门控 stance）
- `app/gpu_body.py:109-122`（50 维观测分支，qvel[0:2] + 相对骨盆高）
- `app/expand_observation.py`（encoder 新列零初始化，obs_mean 0 / obs_std 1）
- `app/full_brain.py:48`（LayerNorm，DeepSeek §1.3 的共模假设对象）
- `app/train_full.py:80-138`（evaluate 函数，固定 seed 1001，score 公式）

**数据**：
- 云端 `runs/yumi_obs50/burst50.log`（12 bursts 完整记录，v_loss 80-570，棘轮 0.853/0.869）
- 云端 `runs/yumi_obs50/evaluation.json`（最终评估：successes 2/32, score 0.211）
- 云端 `runs/yumi_obs50/verify_best.json`（best.pt 3 次评估：0.436/0.362/0.452, mean 0.417, spread 0.090）
- `research/experiments/GAIT_MEASUREMENT_20260918.md`（Step 0 测量，时钟裁决、传递曲线、颤动形态）
- `research/experiments/PPO_POSTURE_20260916.md`（ppo7 价值网冷启动前科、ppo15c 编年）

**推断声明**：
- 「value 权重未从 ppo_state 加载」由代码逻辑推断（`Value(50)` 与旧 47 维 state_dict shape 不匹配，`restore_optimizer` 失败时静默），未直接验证 state_dict 内容。
- 「posture bonus 在 h=0.9 时饱和」由公式 `0.5 * clamp((0.9-0.80)/0.13) = 0.385` 推得，未做消融。
- 「3 次评估均值可消除棘轮假阳性」由 verify_best 的 spread 0.090 推断，未实测。

09-20（50 维，sensor + 膝门控）：best 真实 score 0.417（3 次均值），successes 最高 6/32（burst 12 基线）。

**sensor 补丁让 best 的真实 score 从 ~0.33（source）提升到 ~0.42（best），+27%。** 但 ppo15c 的 0.525 峰值为单次评估，若按同样 3 次均值标准可能也在 0.4 左右。两组数据不可直接比，但方向一致：观测补丁有效，训练过程不稳定。