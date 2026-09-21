# 提案与讨论索引

按提出日期归档：`YYYY-MM-DD/模型或Agent名-提案概要.md`。综合讨论与独立提案分别保留。这里只整理文档位置，不将提案判断视为已验证结论或已批准实验。

名称沿用原文件中的模型或 Agent 标签。Cline、Codex 的记录没有明确底层模型版本，因此不猜测型号。GLM5.3-Flash 原文件的正文署名为 Claude，与文件标签不一致；此次保留原文及原文件标签，未重新归因。

## 2026-09-16

- [GLM5.3-Flash：步态接口与直立提速](2026-09-16/GLM5.3-Flash-步态接口与直立提速.md)
- [DeepSeek-v4.1-Flash：观测盲区与控制耦合诊断](2026-09-16/DeepSeek-v4.1-Flash-观测盲区与控制耦合诊断.md)
- [Kimi：补齐观测与闭环热启动](2026-09-16/Kimi-补齐观测与闭环热启动.md)

## 2026-09-20

- [Cline：评估校准与价值网络预热](2026-09-20/Cline-评估校准与价值网络预热.md)

## 2026-09-21

- [Codex-independent：深蹲基线与短程对照](2026-09-21/Codex-independent-深蹲基线与短程对照.md) — 独立审查，末尾另有交叉审查补记。
- [Codex：直立基线校准与策略对照](2026-09-21/Codex-直立基线校准与策略对照.md) — 已参考各方意见的综合讨论。
- [Codex-independent：新增观测学习诊断](2026-09-21/Codex-independent-新增观测学习诊断.md) — 独立审查，含 critic 混杂与 Adam 缩放审查补记。
- [Codex：新增观测对照与价值网隔离](2026-09-21/Codex-新增观测对照与价值网隔离.md) — 综合 CPU 证据；完整图对照留待云端。

后续实现见 [2026-09-21 无卡修正交接](../LOCAL_FIXES_20260921.md)。实验执行记录仍放在 [实验目录](../experiments/)。

## 旧名称对照

为保留实验编年的原始引用，旧名称在此提供迁移对照。

| 原文件名 | 新位置 |
|---|---|
| `NEXT_STEP_PROPOSAL_20260916(GLM5.3-Flash).md` | [GLM 提案](2026-09-16/GLM5.3-Flash-步态接口与直立提速.md) |
| `NEXT_STEP_PROPOSAL_20260916(DeepSeek-v4.1-Flash).md` | [DeepSeek 提案](2026-09-16/DeepSeek-v4.1-Flash-观测盲区与控制耦合诊断.md) |
| `NEXT_STEP_PROPOSAL_20260916(Observability).md`（旧索引别名） | [DeepSeek 提案](2026-09-16/DeepSeek-v4.1-Flash-观测盲区与控制耦合诊断.md) |
| `NEXT_STEP_PROPOSAL_20260916(Kimi).md` | [Kimi 提案](2026-09-16/Kimi-补齐观测与闭环热启动.md) |
| `NEXT_STEP_PROPOSAL_20260920(Cline).md` | [Cline 提案](2026-09-20/Cline-评估校准与价值网络预热.md) |
| `NEXT_STEP_PROPOSAL_20260921(Codex-independent).md` | [Codex 独立提案](2026-09-21/Codex-independent-深蹲基线与短程对照.md) |
| `NEXT_STEP_DISCUSSION_20260921(Codex).md` | [Codex 综合讨论](2026-09-21/Codex-直立基线校准与策略对照.md) |
