# research/ 目录索引

research/ 专门存放实验过程记录与提案：根目录放主报告与当前修正交接，其余按 提案讨论 / 实验记录 / 角色与身体 / 外部参考 / 溯源数据 / 发行准备 分层。面向使用者的复现与部署指南（LOCAL、CLOUD、PREVIEW、TRAINING）在仓库根目录的 `docs/`。

## 固定路径（代码与脚本依赖，移动前必须同步改引用）

| 路径 | 依赖方 |
|---|---|
| `REPORT.md` | `app/server.py` `/api/report` 直接服务 |
| `../docs/LOCAL.md`、`../docs/CLOUD.md` | `app/cloud_server.py` `/api/report`；`cloud.ps1` 打包上云；`start.ps1` 错误提示 |
| `provenance/provenance.json` | `setup.ps1` 资产哈希校验 |

## 根目录

- `REPORT.md` — 主报告：论文调研、方法、结果、算力、未完成范围（2026-09-15 基线）。
- [LOCAL_FIXES_20260921.md](LOCAL_FIXES_20260921.md) — 无卡修正、CPU 验证与待执行的 GPU 检查。

## proposals/ — 提案与综合讨论

本轮交接：[2026-09-22 进展汇总](experiments/ROUND_SUMMARY_20260922.md)，包含已完成代码、实测结果、验证及尚未执行的下一阶段。

- [提案索引](proposals/README.md) — 按 `YYYY-MM-DD/模型或Agent名-提案概要.md` 归档，保留各方原始观点。
- [2026-09-16](proposals/README.md#2026-09-16) — GLM5.3-Flash、DeepSeek-v4.1-Flash、Kimi 的步态与观测提案。
- [2026-09-20](proposals/README.md#2026-09-20) — Cline 的评估与价值网络提案。
- [2026-09-21](proposals/README.md#2026-09-21) — Codex 独立提案与综合讨论。

## releases/ — 发行准备记录

- `v0.2.0/` — v0.2.0 的决策记录、草稿、校验证据与打包脚本；其中 STUDIO / REPRODUCE / MODEL_CARD 随发行包分发，由 `prepare_v020.py` 按固定路径映射。

## experiments/ — 实验编年与评审（按时间序）

- `PPO5_REVIEW_20260915.md` — ppo5 前后评审。
- `PPO8_FIXES_20260915.md` — ppo8 修复记录。
- `POSTURE_AND_STABILITY_REVIEW_20260916.md` — 全景技术报告：深蹲病因四因素、ppo11~14 复盘、上肢与仿生前瞻。
- `PPO_POSTURE_20260916.md` — 姿态实验编年 ppo10~16（含"下一步候选"）。
- `HOME_REFERENCE_MISMATCH_20260916.md` — `home` 观测基准与检查点不匹配：11:41 改 `default_angles` 让此前所有检查点失效；含 A/B 证据、影响范围与待做的结构性修复。
- [GAIT_MEASUREMENT_20260918.md](experiments/GAIT_MEASUREMENT_20260918.md) — 步态、时钟与速度传递测量。
- [G1_CHECKPOINT_REBIND_20260918.md](experiments/G1_CHECKPOINT_REBIND_20260918.md) — G1 检查点与验收记录绑定。
- [PPO_OBS50_20260920.md](experiments/PPO_OBS50_20260920.md) — 50 维观测续训、最佳检查点筛选与价值预热。
- [GPU_CALIBRATION_20260921.md](experiments/GPU_CALIBRATION_20260921.md) — 迁移 PPO 状态、冻结梯度提速、价值诊断与四组短训复测；尚无稳定提升。
- [INPUT_LEARNING_AUDIT_20260921.md](experiments/INPUT_LEARNING_AUDIT_20260921.md) — 后续 CPU 审计：新列确实更新、共享归一化会扰动价值网；完整图固定对照待云端执行。
- [INPUT_SCALE_CLOUD_20260922.md](experiments/INPUT_SCALE_CLOUD_20260922.md) — 云端单步配对与分组回放完成，新输入有梯度但动作影响仍小。
- [ACTOR_SCALE_AB_20260922.md](experiments/ACTOR_SCALE_AB_20260922.md) — 隔离 critic 统计量后的在线尺度 A/B；新输入敏感度增强，配对行走未改善，未晋升权重。
- [ROUTE_DECISION_20260922.md](experiments/ROUTE_DECISION_20260922.md) — 最新编年：诊断已完成；最终决策保留连接组与 Yumi，下一阶段优先朝向恢复课程及原生多指标验收。尚无行走突破，实例保持开机。

## avatar/ — 角色与身体

- `YUMI.md` — Yumi VRM 资产来源、许可与哈希。
- `YUMI_BODY.md` — Yumi 物理身体建模、验收与遗留事项。
- `YUMEKA_REFERENCE.md` — 角色参考资料。

## references/ — 外部项目与研究资料

- [果蝇具身项目参考（2026-09-21）](references/fly-embodiment/README.md) — satorunet / hae 与 yakshawan / FFREP：进展、控制与学习实现、源码索引、固定版本引用、公开取回与校验方法和复现边界。

## provenance/ — 溯源数据

- `provenance.json` — 数据 / 模型 / 资产 SHA-256（setup.ps1 依赖）。
- `avatar-meta.json`、`yumi-meta.json`、`avatar-commit.txt` — VRM 内嵌元数据与源码版本。
- `licenses/` — 第三方许可副本（flycube、unitree、DOOMFLY 归属说明）。
