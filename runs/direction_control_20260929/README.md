# 2026-09-29 方向控制实验工件

结论、预注册和阶段边界见 `research/experiments/DIRECTION_CONTROL_20260929.md`。本目录保留成功与失败路径；没有替换已发布权重或默认预览。

## 执行顺序

1. `run_stage0.py`：三条来源基线、双向推扰及有限动作/命令探针。
2. `run_probe_validation.py`：通过短测的四个组合接受长走与联合门槛。
3. `run_teacher_pair.py`：同预算旧奖励与横向精度 60 对照。教师选择由 `teacher_selection.json` 固定。
4. `run_transfer.py`：最多四轮冻结核心读出迁移。主路径记录在 `transfer_main/execution.json`。
5. `run_refit_probe.py`：相同特征与标签下的一次 ridge 对照。
6. `run_readout_ppo.py`：critic 冷启动尝试；因预热门槛结束，actor 未更新。
7. `prepare_readout_critic.py`、`run_readout_ppo_warm.py`：显式派生的双源初始化与有界读出 PPO。
8. `run_yaw_gain_probe.py`：PPO 末轮的单点航向反馈诊断。
9. `run_teacher_control_replica.py`、`run_teacher_repro_stage.py`：旧奖励续训重复及条件执行的新初态验证。

这些驱动使用相对于仓库根目录的路径，已有结果文件通常会拒绝覆盖。不要在原实验目录中重复启动。各执行 JSON 保存实际命令、源码哈希、退出状态和耗时；它们比目录命名更能说明哪些步骤确实执行。

`run_calibrated_validation.py` 和 `run_transfer_holdout.py` 是条件方案。脚本存在不代表已运行；先检查对应执行记录。学生的最终留出只允许在开发联合门槛通过且候选固定后使用。

## 证据与备份

| 工件 | 内容 |
| --- | --- |
| `stage0_manifest.json` | 初始评估、探针、轨迹和源码包 |
| `teachers_manifest.json` | 首次教师配对的权重、PPO 状态、评估和日志 |
| `supervised_main_manifest.json` | 四轮迁移及固定数据正则对照，包含原始样本和每轮完整特征 |
| `readout_attempts_manifest.json` | 两次读出 PPO、派生初始化、源码与驱动 |
| `push_exposure_audit.json` | 根据回合时长推断实际推扰暴露，明确区别于配置值 |

压缩包、权重、轨迹与大特征文件按项目规则不入库。manifest 给出逐文件 SHA-256；`*_backup_verification.json` 记录本机校验结果。重复的逐数据集特征缓存仍留在云端，完整每轮特征与原始数据已经归档。

PPO 源码归档解压到本机 `readout_source/`，没有覆盖工作区应用源码。来源路径 `runs/source_teacher`、`runs/source_main` 和 `runs/source_replica` 在隔离云工作区指向原始检查点；来源检查点哈希见实验编年。本文没有提供新的公共权重下载渠道，实验工件尚未发行。

## 阶段完成

`teacher_repro_execution.json` 确认两位教师都通过新初态联合门槛。连接组学生仍未通过全部长走方向门槛；学生最终留出与独立学生迁移没有执行。

`phase_closeout_manifest.json` 保存重复教师、最终评估、轨迹和源码；`phase_closeout_backup_verification.json` 确认本机校验完成。此源码副本位于本机 `phase_closeout_source/`，同样不覆盖应用源码。配对来源评估为 `teacher_source_matched_long.json` 和 `main_source_matched_development_long.json`。
