# G1 检查点哈希重绑定：2026-09-18

## 背景

2026-09-17 云端续训后 `runs/cloud/best.pt` 被替换为新检查点，而页面评测接口 `/api/evaluation` 只服务与当前加载检查点哈希绑定的留出验收记录，导致 G1 身体评测 404。本条记录处置过程与结果。

## 事实核查

- 当前 `runs/cloud/best.pt` SHA-256：`ffcf9a9799ae54332cb5c07ec56d79145ba7e1b557100c78ef082e878713271b`（311,785,917 字节，2026-09-17 17:57 更新；`training.json` 显示 4800 次更新、四舍五入 DAgger 续训约 1008 秒，训练时筛评测 9/9）。
- 旧检查点 `f1d5147c…`（2026-09-15 验收 9/9 所绑定）在本地无任何备份：`runs/cloud/last.pt` 为 `a9fe74c6…`（同轮次末态，非旧检查点）。
- 既有验收记录 `runs/cloud/heldout.json` 与 `runs/local/heldout.json` 均绑定 `f1d5147c…`，且不含 `robot` 字段（早于该字段引入），无法直接匹配当前接口。
- 处置选择：不回滚（旧权重不可恢复，且新检查点是续训改进产物），改为对当前检查点补做独立留出验收。

## 处置

- 2026-09-18：本机 `.venv-gpu` 启动 `start.ps1 -Body g1 -Device cuda`，确认热加载 `ffcf9a97…`（`/api/state` 的 `checkpoint` 字段）。
- 通过 `POST /api/evaluate`（quick，9 项 × 30 s）执行独立留出原生 MuJoCo 验收，输出 `runs/cloud/evaluations/7ea830656e3e457ebfe5d274d6174658/heldout.json`，绑定 `ffcf9a97…`。
- 结果：**6/9 通过**。未达标项为两档高速（0.65 m/s）测试的朝向标准（lateral_m ≈ 4.3 m，超过 max(1 m, 前进距离 20%)）。低速两档（0.35 / 0.5 m/s）全部通过，无跌倒。
- 验证：`GET /api/evaluation` 返回 200，服务记录绑定 `ffcf9a97…`，哈希脱节消除。验证后已停止服务。

## 与训练时筛评测的差异说明

2026-09-17 训练时筛评测为 9/9（15 s），本次独立验收为 6/9（30 s、更严格朝向标准）。差异来自更长行程放大了高速档横向漂移；以独立验收为准。

## 同步更新

- `research/guides/LOCAL.md`、`research/guides/CLOUD.md` 检查点哈希行已更新为当前世系并注明旧检查点状态。
- 旧验收记录文件保留不动（2026-09-15 验收 9/9 仍指向 `f1d5147c…`，作为历史证据）。
