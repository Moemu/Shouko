# AGENTS.md — Neuromechfly 工作室

果蝇连接组（MaleCNS）驱动人形身体（G1/Yumi）实时行走的仿真工作室。完整连接组推理 + MuJoCo 物理 + VRM 角色展示 + 训练/评估工作台，本机与 AutoDL 云端共用一套服务端逻辑。

## 目录与固定路径

- `app/` — Python 服务端。`studio_server.py` 是唯一主线入口（FastAPI，`start.ps1` 启动于 8740）；`server.py`（8192 神经元子图）与 `cloud_server.py` 为 legacy/特殊用途，不再加功能。
- `web/` — 前端源码（Vite，无框架，扁平 i18n 字典）。`web/dist/` 是构建产物，不入库。
- `research/` — 报告、指南、实验编年与溯源。**以下路径被代码/脚本引用，移动前必须同步改引用**：`research/REPORT.md`（`/api/report`）、`research/guides/LOCAL.md`、`guides/CLOUD.md`（`cloud.ps1` 打包上云、错误提示）、`research/provenance/provenance.json`（`setup.ps1` 哈希校验）。
- `runs/` — 训练产物与运行日志；`data/` — 连接组数据。二者大体积文件均不入库（见 .gitignore）。
- `vendor/` — 上游依赖检出版（flycube、unitree_rl_gym），不改内部代码。

## 常用命令

- 前端：`npm run build`（Vite）、`npm test`（三个 node 测试：i18n 键奇偶、脑活动、训练曲线）。
- 后端测试（可直接运行，无 pytest）：`.venv-gpu\Scripts\python.exe -m app.test_studio_server`、`-m app.test_training_control`。
- 本地启动：`.\start.ps1 [-Body g1|yumi] [-Device cuda|cpu] [-Port 8740]`；停止 `.\stop.ps1`。双环境：`.venv`（CPU）/`.venv-gpu`（CUDA）。
- 云端：`.\cloud.ps1 Status|Preview|Train|Sync`，SSH 目标经 `-CloudHost/-CloudPort` 参数或 `runs/cloud/target.json`（git-ignored）传入。

## 硬约束

1. **隐私红线**：任何入库文件不得出现真实 SSH 主机/端口、用户名路径（`C:\Users\...`）、聊天记录路径等个人信息。云端目标只存在 `runs/cloud/target.json`（已 ignore）。写实验记录时用占位符（如 `root@<AutoDL 主机>`）。
2. **大文件不入库**：`*.pt`/`*.npz`/`*.tgz`/`*.log`/`*.pid`/`*.jsonl`、`data/`、`web/public/*.vrm` 均被 ignore，勿改 .gitignore 放行。
3. **i18n 双字典同步**：文案改动必须同时改 `web/i18n.js` 的 zh/en 两字典（当前 444 键，`npm test` 校验奇偶与占位符），页面用 `data-i18n` / `data-i18n-attr` 标记。
4. **实验数据不虚构**：报告与实验记录中的数字必须来自 `runs/` 实测文件，引用注明出处；推断要标注为推断。
5. **实验编年史不可篡改**：`research/experiments/` 是按时间序的历史记录，只允许追加或隐私脱敏，不改写历史结论。
6. **平台**：Windows + PowerShell（.ps1）+ Git Bash；Python 文件一律 UTF-8；控制台输出乱码多为 GBK 显示问题，勿误判为文件损坏。
7. **评估与检查点哈希绑定**：页面评测数据与 checkpoint 哈希绑定，替换权重需同步评估文件，注意 README「已知问题」中的哈希不一致记录。
8. **社区可复现红线**：推送到公开仓库前必须确认社区用户可独立复现——不依赖私有端点、私有文件或未写进文档的手动步骤；文档引用的脚本必须真实存在；需 GPU/云训练的产物（权重、评估文件）要有发布渠道（如 Release）并附 SHA-256 与训练出处，或提供可跑通的复训命令。

## 协作模式：观点碰撞（仅方向性决策）

适用于**训练策略、实验方向、架构取舍**等方向性决策；用户明确指定的修复/收尾任务直接执行，不强制碰撞。已有范例：`research/NEXT_STEP_PROPOSAL_*.md` 与 `experiments/PPO_POSTURE_*.md` 末尾的「下一步候选」。

流程：

1. **独立提案**：每个参与 Agent 独立阅读代码与 `runs/` 实测数据后各自写一份提案（`research/NEXT_STEP_PROPOSAL_<日期>(<提出方>).md`），明确声明**未参考其他提案的结论**，关键论断附代码/数据出处。
2. **对比与合并**：提案并列呈现，显式写出分歧点与各自依据；合并建议只做参考，不替用户拍板。
3. **用户定稿**：方向由用户选择，定稿后进入 `experiments/` 编年执行。
4. **范围纪律**：不为收尾旧账无限扩大当前任务范围；发现的旧账单独列出，等用户决定是否处理。

## 待决事项

- 仓库许可证已定为 MIT（2026-09-18 落地 LICENSE）；发布渠道（tag/Release 上传权重）待远程仓库建立后进行。

## 动敏感区域前先读

- 训练/评估逻辑：`research/REPORT.md`（方法与边界）、`research/guides/LOCAL.md`、`guides/CLOUD.md`。
- 身体与角色：`research/avatar/YUMI.md`、`YUMI_BODY.md`（VRM 许可约束见 `THIRD_PARTY.md`，署名不可去除）。
- 当前实验状态：`research/experiments/PPO_OBS50_20260920.md`（最新编年，ppo17：50 维观测热启动续训、棘轮统计缺陷发现）。
- 双语说明：`README.md` / `README.zh.md` 改动需保持一致。
