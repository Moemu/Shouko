# research/ 目录索引

按用途分层：根目录放主报告与当前活跃文档，其余按 运行指南 / 实验记录 / 角色与身体 / 溯源数据 分层。

## 固定路径（代码与脚本依赖，移动前必须同步改引用）

| 路径 | 依赖方 |
|---|---|
| `REPORT.md` | `app/server.py` `/api/report` 直接服务 |
| `guides/LOCAL.md`、`guides/CLOUD.md` | `app/cloud_server.py` `/api/report`；`cloud.ps1` 打包上云；`start.ps1` 错误提示 |
| `provenance/provenance.json` | `setup.ps1` 资产哈希校验 |

## 根目录

- `REPORT.md` — 主报告：论文调研、方法、结果、算力、未完成范围（2026-09-15 基线）。
- `NEXT_STEP_PROPOSAL_20260916(GLM5.3-Flash).md` — 直立步态提速破局提案（供多 Agent 讨论定稿，含另一 Agent 方案的对比与合并建议）。
- `NEXT_STEP_PROPOSAL_20260916(Observability).md` — 独立审查提案：观测盲区（缺线速度/骨盆高）与单旋钮锁死的诊断、实测速度-指令拟合、与另两份方案的分歧、建议先测量后训练。
- `NEXT_STEP_PROPOSAL_20260916(Kimi).md` — 独立审查提案：观测缺失（奖励要求的线速度/骨盆高不在 47 维观测内、策略无记忆）为真正断点；主张先测 (cmd, vx) 传递曲线，再零初始化补观测热启动 + 膝惩罚支撑腿门控，时钟改动条件触发。

## guides/ — 运行与复现

- `LOCAL.md` — 本机完整图预览：使用、资源、费用口径、复现。
- `CLOUD.md` — 云端实验：实测、费用、复现记录。

## experiments/ — 实验编年与评审（按时间序）

- `PPO5_REVIEW_20260915.md` — ppo5 前后评审。
- `PPO8_FIXES_20260915.md` — ppo8 修复记录。
- `POSTURE_AND_STABILITY_REVIEW_20260916.md` — 全景技术报告：深蹲病因四因素、ppo11~14 复盘、上肢与仿生前瞻。
- `PPO_POSTURE_20260916.md` — 姿态实验编年 ppo10~16 与当前状态（最新的实验记录，含"下一步候选"）。
- `HOME_REFERENCE_MISMATCH_20260916.md` — `home` 观测基准与检查点不匹配：11:41 改 `default_angles` 让此前所有检查点失效；含 A/B 证据、影响范围与待做的结构性修复。

## avatar/ — 角色与身体

- `YUMI.md` — Yumi VRM 资产来源、许可与哈希。
- `YUMI_BODY.md` — Yumi 物理身体建模、验收与遗留事项。
- `YUMEKA_REFERENCE.md` — 角色参考资料。

## provenance/ — 溯源数据

- `provenance.json` — 数据 / 模型 / 资产 SHA-256（setup.ps1 依赖）。
- `avatar-meta.json`、`yumi-meta.json`、`avatar-commit.txt` — VRM 内嵌元数据与源码版本。
- `licenses/` — 第三方许可副本（flycube、unitree、DOOMFLY 归属说明）。
