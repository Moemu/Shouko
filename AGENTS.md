# AGENTS.md — Neuromechfly 工作室

果蝇连接组（MaleCNS）驱动人形身体（G1/Yumi）实时行走的仿真工作室。完整连接组推理 + MuJoCo 物理 + VRM 角色展示 + 训练/评估工作台，本机与 AutoDL 云端共用一套服务端逻辑。

## 目录与固定路径

- `app/` — Python 服务端。`studio_server.py` 是唯一主线入口（FastAPI，`start.ps1` 启动于 8740）；`server.py`（8192 神经元子图）与 `cloud_server.py` 为 legacy/特殊用途，不再加功能。
- `web/` — 前端源码（Vite，无框架，扁平 i18n 字典）。`web/dist/` 是构建产物，不入库。
- `docs/` — 面向使用者的复现与部署指南（LOCAL / CLOUD / PREVIEW / TRAINING）。
- `research/` — 专门负责实验过程记录（`experiments/` 编年）与提案（`proposals/`），另含主报告、溯源、角色资料、外部参考与发行准备记录。**以下路径被代码/脚本引用，移动前必须同步改引用**：`research/REPORT.md`（`/api/report`）、`docs/LOCAL.md`、`docs/CLOUD.md`（`cloud.ps1` 打包上云、`start.ps1` 错误提示、`app/cloud_server.py`）、`research/provenance/provenance.json`（`setup.ps1` 哈希校验）。`docs/PREVIEW.md`（浏览器端预览的静态托管，仅文档引用）。
- `runs/` — 训练产物与运行日志；`data/` — 连接组数据。二者大体积文件均不入库（见 .gitignore）。
- `vendor/` — 上游依赖检出版（flycube、unitree_rl_gym），不改内部代码。

## 常用命令

- 前端：`npm run build`（Vite）、`npm test`（五个 node 测试：i18n 键奇偶、脑活动、训练曲线、评估视图展开、预览权重包契约——末项在未导出包时自动跳过，其余无 DOM 依赖）。
- 后端测试（可直接运行，无 pytest）：`.venv-gpu\Scripts\python.exe -m app.test_studio_server`、`-m app.test_training_control`。
- 预览权重包：`.venv-gpu\Scripts\python.exe -m app.export_preview_weights` → `artifacts/preview/`（发布工件，不入库；证据与检查点不符会拒绝导出）。部署见 `docs/PREVIEW.md`。
- 本地看预览（免服务器、免 Caddy）：`.venv\Scripts\python.exe -m app.serve_preview` → http://127.0.0.1:8741/；它是 PREVIEW.md §4 服务器规格的可执行版本，回归在 `-m app.test_serve_preview`。
- 本地启动：`.\start.ps1 [-Body g1|yumi] [-Device cuda|cpu] [-Port 8740]`；停止 `.\stop.ps1`。双环境：`.venv`（CPU）/`.venv-gpu`（CUDA）。
- 云端：`.\cloud.ps1 Status|Preview|Train|Sync`，SSH 目标经 `-CloudHost/-CloudPort` 参数或 `runs/cloud/target.json`（git-ignored）传入。

## 云实例保留

- GPU 资源可能长期无空闲。训练结束或暂时无任务时，默认保留实例，不自行关机、释放 GPU 或切换无卡模式；需要释放时由用户明确决定。
- 预算包含保留实例的空闲费用。接近预算上限前提醒用户，提前确定继续保留的预算或释放时间；停止训练不等于停止计费。

## 硬约束

1. **隐私红线**：任何入库文件不得出现真实 SSH 主机/端口、用户名路径（`C:\Users\...`）、聊天记录路径等个人信息。云端目标只存在 `runs/cloud/target.json`（已 ignore）。写实验记录时用占位符（如 `root@<AutoDL 主机>`）。
2. **大文件不入库**：`*.pt`/`*.npz`/`*.tgz`/`*.log`/`*.pid`/`*.jsonl`、`data/`、`web/public/*.vrm` 均被 ignore，勿改 .gitignore 放行。
   **`.gitignore` 的纪律**：它是入库文件，只写「社区用户确会产生」的产物（如 `artifacts/`：文档要求用户跑导出脚本）。仅由本机特定状态产生、别人复现不到的文件——私有部署目标、发布准备工作区、靠私有权重才能重建的二进制夹具、本机调试输出——**一律写进 `.git/info/exclude`**，不要写进 `.gitignore`。待决的未跟踪文件要么入库、要么明确排除，不要长期悬在工作区里。
3. **i18n 双字典同步**：文案改动必须同时改 `web/i18n.js` 的 zh/en 两字典（当前 463 键，`npm test` 校验奇偶与占位符），页面用 `data-i18n` / `data-i18n-attr` 标记。
4. **实验数据不虚构**：报告与实验记录中的数字必须来自 `runs/` 实测文件，引用注明出处；推断要标注为推断。
5. **实验编年史不可篡改**：`research/experiments/` 是按时间序的历史记录，只允许追加或隐私脱敏，不改写历史结论。
6. **平台**：Windows + PowerShell（.ps1）+ Git Bash；Python 文件一律 UTF-8；控制台输出乱码多为 GBK 显示问题，勿误判为文件损坏。
7. **评估与检查点哈希绑定**：页面评测数据与 checkpoint 哈希绑定，替换权重需同步评估文件，注意 README「已知问题」中的哈希不一致记录。
8. **社区可复现红线**：推送到公开仓库前必须确认社区用户可独立复现——不依赖私有端点、私有文件或未写进文档的手动步骤；文档引用的脚本必须真实存在；需 GPU/云训练的产物（权重、评估文件）要有发布渠道（如 Release）并附 SHA-256 与训练出处，或提供可跑通的复训命令。

## 协作模式：观点碰撞（仅方向性决策）

适用于**训练策略、实验方向、架构取舍**等方向性决策；用户明确指定的修复/收尾任务直接执行，不强制碰撞。已有范例见 `research/proposals/README.md` 与 `experiments/PPO_POSTURE_*.md` 末尾的「下一步候选」。

流程：

1. **独立提案**：每个参与 Agent 独立阅读代码与 `runs/` 实测数据后各自写一份提案（`research/proposals/YYYY-MM-DD/<模型或Agent名>-<提案概要>.md`），明确声明**未参考其他提案的结论**，关键论断附代码/数据出处。综合讨论放在对应日期下，并明确说明已参考其他提案。
2. **对比与合并**：提案并列呈现，显式写出分歧点与各自依据；合并建议只做参考，不替用户拍板。
3. **用户定稿**：方向由用户选择，定稿后进入 `experiments/` 编年执行。
4. **范围纪律**：不为收尾旧账无限扩大当前任务范围；发现的旧账单独列出，等用户决定是否处理。

## 待决事项

- 仓库许可证已定为 MIT；远程已有 v0.1.0 Release。v0.2.0 尚未打标签或发行，建议由连接组读出迁移主候选承接，旧直立权重保留为来源基线。发布准备见 `research/releases/v0.2.0/DECISION.md`；版本安排待用户定稿。

## 动敏感区域前先读

- 训练/评估逻辑：`research/REPORT.md`（方法与边界）、`docs/LOCAL.md`、`docs/CLOUD.md`。
- 浏览器端预览与静态托管：`docs/PREVIEW.md`（工件布局 + **服务器规格** + 本机 Caddy 参考实现，三层分开；首屏约 235 MiB、HTTPS 是权重哈希校验的前提、评估快照与检查点绑定）。推送实现**有意不入库**（站点特定、不构成社区复现路径），文档只给结果约束。
- 身体与角色：`research/avatar/YUMI.md`、`YUMI_BODY.md`（VRM 许可约束见 `THIRD_PARTY.md`，署名不可去除）。
- 当前实验状态：`research/experiments/CONNECTOME_TRANSFER_20260923.md`（完整连接组冻结核心、只训练读出，主候选及独立数据额外收敛分支均通过四组严格留出；同日程复现曾失败，精确朝向与横漂仍待解决。显式相位保留，不代表自主CPG或拓扑优势。下一阶段建议见 `research/proposals/2026-09-23/Codex-连接组读出迁移与下一阶段路线.md`，由用户定稿；实例保持开机）。
- 前序身体/任务对照：`research/experiments/ROUTE_EXECUTION_20260923.md`（普通MLP迈步、等预算旧奖励对照与四组留出，作为连接组迁移教师的来源）。
- 前序路线：`research/experiments/ROUTE_DECISION_20260922.md`；保留历史结论，当前证据以2026-09-23实验编年为准。
- 前序诊断：`research/experiments/ACTOR_SCALE_AB_20260922.md`、`GPU_CALIBRATION_20260921.md`、`INPUT_LEARNING_AUDIT_20260921.md`、`INPUT_SCALE_CLOUD_20260922.md`；方向讨论见 `research/proposals/2026-09-22/`。
- 双语说明：`README.md` / `README.zh.md` 改动需保持一致。
