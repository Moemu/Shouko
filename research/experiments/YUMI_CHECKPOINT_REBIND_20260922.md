# Yumi 预览默认检查点切换到直立 obs50 世系（2026-09-22）

## 决策与授权

用户在本轮对话中确认：仓库当前没有所谓「生产路径」，预览默认权重应随阶段成果切换；选定直立世系检查点作为阶段性成果（用户称 v0.2.0）。候选评估由本 Agent 完成：`458fc465` 是直立世系中唯一具备校准后可信评估（[ROUTE_DECISION_20260922](ROUTE_DECISION_20260922.md) 的固定顺序 CSR + 原生 MuJoCo 路径）、血缘档案完整（`666134f2` → `780f4f4f` 扩维 → obs50 续训）、且为下一轮课程训练基线的候选。A/B 子代（`8c41b0b3`/`19a93a1c`）按路线决策不晋升。

云端实例已由用户关闭；本条目全部操作在本机（RTX 4070 Laptop, CUDA）完成。未训练、未修改任何历史编年。

## 切换内容与哈希

`runs/yumi/` 下四件套整组替换为 obs50 世系（来源 `runs/yumi_calibration_20260921/migrated/runs/yumi_obs50/`，另 `runs/cloud/obs50_migration_20260921.tgz` 归档为独立副本）：

| 文件 | SHA-256 | 说明 |
|---|---|---|
| `best.pt` | `458fc465a6fb802a338c218fc4be414fc9153376c5d2f76f116518e6cc5aa475` | 50 维直立 obs50 best，本轮全部诊断的来源权重 |
| `last.pt` | `cbe3b3f173867f70400b1f82b31fa8ab590d1b0eac680bb1219c4e2de5c68f6b` | 迁移 last |
| `ppo_state_best.pt` / `ppo_state.pt` | 状态文件内嵌 `checkpoint_sha256` 分别绑定 `458fc465` / `cbe3b3f1`（实测校验） | 配对 PPO 状态；恢复按文件名自动配对（`ppo_yumi.py:181-183`） |

旧深蹲世系备份（切换前实测）：

| 文件 | SHA-256 | 说明 |
|---|---|---|
| `best.pt.bak` / `ppo_state_best.pt.bak` | `a7a4281f604c82831bf947fe599d490dd5d223bf479c9c44289e7b4d76508d97` | ppo9 深蹲，9/9 + 16/16 验收世系，原备份不动 |
| `last47.pt` / `ppo_state47.pt` | `last47` 为 `7f67b4d2de6dccfc1689234c3b25102804df31eea8f578b88c300a6164642205` | 切换前的旧 last 配对（改名以匹配 `.gitignore` 的 `runs/**/*.pt` 模式） |
| `training47.json` / `evaluation47.json` | — | 旧世系训练状态快照 |
| `best_tall.pt` | `666134f2b26e4bb5f585633fe22170a1f796e39a39edcd31ca339bd5e831488d` | 直立 home 原始世系，保留 |

`training.json` / `evaluation.json` 同步为 migrated obs50 的对应文件（该快照 phase 为训练中；页面活跃训练判定按进程扫描 `external_training`，不受影响）。home 配置风险不复存在：`load_checkpoint` 从 checkpoint 内嵌 `physics_interface` 与 `observation_size` 重建身体（`studio_server.py:151-163`），50 维接口自动加载。

## 加载验证

`start.ps1 -Body yumi -Device cuda` 启动后 `/api/state` 上报 `checkpoint = 458fc465…`、`checkpoint_updates = 40`、robot=yumi，无加载错误。

## 留出验收（真实数字，未达标即如实记录）

由内置 `/api/evaluate` 触发 `app.evaluate_full.py`（与历史验收同一路径：原生 MuJoCo、留出种子 8001–8009、六判据严格成功）。两份记录均在 `runs/yumi/evaluations/` 下与快照 checkpoint 哈希绑定，`/api/evaluation` 实测只对当前加载哈希出示（绑定机制验证通过）。

全量记录 `95111ceac1b4434696896fedbeb9c7bb/heldout.json`（16 回合，complete=true）：

| 检查 | 结果 |
|---|---|
| 9 新初态 × 30 秒（严格六判据） | **0/9**；判据逐项通过 4–5/6；存活 8/9（8004 跌倒）；横移 0.64–7.06 m |
| 120 秒持续行走 | 存活 120 秒，前进 10.99 m（0.092 m/s），横漂 15.43 m；触地 544/498 次，交替 933 次 |
| 侧向轻推 ×3 | 3/3 不倒（最低直立 0.91–0.92） |
| 断开脑内连接 ×3 | 3/3 约 1 秒跌倒（1.0/1.0/1.1 s） |

逐回合 30 秒测试（速度 m/s / 横移 m / 跌倒）：8001 −0.047/3.42/否；8002 0.311/7.06/否；8003 0.451/4.79/否；8004 0.212/0.86/是；8005 0.101/2.91/否；8006 0.459/0.98/否；8007 −0.045/0.64/否；8008 0.078/3.76/否；8009 0.358/6.35/否。

**同种子直接对比**：`a7a4281f`（深蹲）的历史留出记录（`runs/local/heldout_yumi.json`，同种子 8001–8009、同判据）为 9/9、120 秒 65.55 m 横漂 0.13 m。直立 obs50 世系在定向行走（速度与方向跟踪）上明确弱于旧记录；其成立的部分是直立姿态（配对评估平均根部高约 0.899 m，见 [ACTOR_SCALE_AB_20260922](ACTOR_SCALE_AB_20260922.md)）、存活、交替节律、轻推恢复与连接组依赖性（断连即倒）。这与 [ROUTE_DECISION_20260922](ROUTE_DECISION_20260922.md) 的诊断（速度 3/27、方向 11/27，主要缺口在速度与朝向跟踪）一致。

quick 预跑（`379e7ba831c74bea9071bd1352504a82`）与全量跑的同种子轨迹存在差异（如 8001 横移 1.78→3.42 m）：生产验收路径沿用历史 CUDA 推理，路线决策已记录其 ~1e-7 动作差会随接触放大；单次记录不构成逐位确定性复现，两份记录均保留。

## 范围与边界

- v0.2.0 定位：直立世系阶段成果 = 直立姿态 + 存活 + 节律 + 校准评估基线；**不宣称定向行走验收通过**，该目标仍开放（旧 `a7a4281f` 仍是最后一个通过完整行走验收的检查点，其记录保留且仍绑定其哈希）。
- 未打 git tag、未建立 Release（发布渠道见 AGENTS.md 待决事项）；权重发布需附 SHA-256 与训练出处后由用户决定。
- 实例已关机，无云端操作；来源 checkpoint 未改动。
