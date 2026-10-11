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
3. **i18n 双字典同步**：文案改动必须同时改 `web/i18n.js` 的 zh/en 两字典（当前 465 键，`npm test` 校验奇偶与占位符），页面用 `data-i18n` / `data-i18n-attr` 标记。
4. **实验数据不虚构**：报告与实验记录中的数字必须来自 `runs/` 实测文件，引用注明出处；推断要标注为推断。
5. **实验编年史不可篡改**：`research/experiments/` 是按时间序的历史记录，只允许追加或隐私脱敏，不改写历史结论。
6. **平台**：Windows + PowerShell（.ps1）+ Git Bash；Python 文件一律 UTF-8；控制台输出乱码多为 GBK 显示问题，勿误判为文件损坏。
7. **评估与检查点哈希绑定**：页面评测数据与 checkpoint 哈希绑定，替换权重需同步评估文件，注意 README「已知问题」中的哈希不一致记录。
8. **社区可复现红线**：推送到公开仓库前必须确认社区用户可独立复现——不依赖私有端点、私有文件或未写进文档的手动步骤；文档引用的脚本必须真实存在；需 GPU/云训练的产物（权重、评估文件）要有发布渠道（如 Release）并附 SHA-256 与训练出处，或提供可跑通的复训命令。

## 自主执行与停机门

默认按**阶段**推进：一次对话连续执行多个阶段，直到发行就绪或触发停机门。承诺边界是「从已定稿方向出发，跑到发行就绪、或跑到下一个路线不确定点」——这两者之间不保证。

**阶段**：一段有固定预算与固定判定门的实验/工程块。开始前把预算、指标和**分支表**写进 `research/experiments/` 的预登记段；判定门由实测数据判定，不由用户判定。阶段结束按分支表选择下一阶段；表覆盖不到的分支算路线不确定。阶段长度以「能被一次对话装下」为准，超出就再切一段。

**自主执行域（不必询问）**：

- 在已定稿方向内选择下一个实验、超参数、种子与诊断项；
- 本机 CPU/GPU 训练与评估，累计机时在预算内；
- 新增 `app/` 下的入口、测试与记录脚本；跑 `npm run build`、`npm test` 与后端测试；
- 每阶段结果追加进 `research/experiments/` 编年（预登记、分支、哈希、复现命令），负结果同样入编年；
- `git commit`（不 push），含 `docs/TRAINING.md`、README 双语同步；
- 修硬约束内的既有问题（隐私、`.gitignore` 纪律、i18n 奇偶、哈希绑定）；
- spawn 同厂子 Agent 做**对抗性核验与局部方案对比**：独立重算数字、尝试反证主代理结论、复现轨迹与哈希、比较同一阶段内的两条实现路径。产出是核验结论，写入编年，不进 `proposals/`；**不得**用它生成路线提案。

**停机门（必须交回用户）**：

1. **发行动作本身**：`git push`、打 tag、发 Release、升级生产预览、替换 v0.3.0 权重或评估；
2. **预算耗尽**：本机机时、云端费用或 token 预算触顶（含云实例保留的空闲费用，见「云实例保留」）；
3. **路线不确定**：同一问题连续两轮负结果且诊断无法区分原因；两个以上分支都符合现有证据；需要改动已定稿的方向、身体或任务接口；实测与预登记预期矛盾。——停下并请用户召集跨厂商提案，**不自行选方向**；
4. **真阻塞**：所需权限、凭据或私有资源在信封外且无替代路径。

除这四条外一律自主继续——包括负结果、需要换实现方式、需要推翻本阶段自己的假设。**负结果 + 完整记录 = 该阶段合法完成，不等于需要询问。**

**状态落在文件里**：每阶段结束把当前分支、已排除项、下一步写入编年末段与 `docs/TRAINING.md`。一次对话能否走到发行，取决于记忆是否在文件里，不取决于上下文长度。

## 协作模式：观点碰撞（跨厂商，用户召集）

适用于**训练策略、实验方向、架构取舍**等方向性决策，由停机门第 3 条触发。路线讨论的价值来自**不同厂商的先验差异**；同一厂商的子 Agent 共享训练先验，其一致意见不是多份独立证据。已有范例见 `research/proposals/README.md` 与 `experiments/PPO_POSTURE_*.md` 末尾的「下一步候选」。

- **路线提案只能跨厂商**：由用户召集不同厂商的 Agent 各自独立成稿（`research/proposals/YYYY-MM-DD/<模型或Agent名>-<提案概要>.md`），明确声明**未参考其他提案的结论**，关键论断附代码/数据出处。综合讨论另写，并明确说明已参考其他提案。主代理可以写自己的一份，但不得代替其他厂商发起。
- **禁止同厂路线提案**：不得用同一厂商的子 Agent 自动生成路线提案；不得把同厂子 Agent 的一致意见写成「多方一致」；不得据此推进方向。
- **允许同厂核验**：见「自主执行与停机门」，核验结论进编年，不进 `proposals/`。
- **用户定稿**：方向由用户选择，定稿后进入 `experiments/` 编年继续自主执行。
- **范围纪律**：不为收尾旧账无限扩大当前任务范围；发现的旧账单独列出，等用户决定是否处理。

## 发行就绪（长跑的终止条件）

全部满足即「独立发行就绪」，主动停下报告；发行动作本身由用户执行（停机门第 1 条）。本节是硬约束 8 的正向清单。

- 硬约束 8 逐条可核对：不依赖私有端点、私有文件或未写进文档的手动步骤；文档引用的脚本真实存在；
- 权重与评估文件的发布渠道、SHA-256 与训练出处齐备，或复训命令可跑通；
- `npm run build`、`npm test`、后端测试、预览权重导出（证据与检查点哈希一致）全绿；
- README 双语一致、i18n 双字典奇偶通过、预览截图与当前检查点一致；
- `research/experiments/` 编年完整覆盖本对话全部阶段，含预登记、分支、负结果与复现命令；
- 工作区无悬空未跟踪文件（入库或写进 `.git/info/exclude`）。

## 待决事项

- 仓库许可证为 MIT。v0.1.0、v0.2.0 已公开发行；v0.3.0 为 2026-09-30 的方向稳定性检查点，用户已批准发行。生产在线预览已于 2026-10-01 升级到 v0.3.0 主模型，本地工作室默认安装仍为 v0.2.0；v0.3.0 模型卡、验收与复现见 `research/releases/v0.3.0/`。
- 长跑中仅有两个用户门：**方向定稿**（停机门第 3 条触发时）与**发行动作**（停机门第 1 条）。其余执行决策由 Agent 自主判定。

## 动敏感区域前先读

- 训练/评估逻辑：`research/REPORT.md`（方法与边界）、`docs/LOCAL.md`、`docs/CLOUD.md`。
- 浏览器端预览与静态托管：`docs/PREVIEW.md`（工件布局 + **服务器规格** + 本机 Caddy 参考实现，三层分开；首屏约 235 MiB、HTTPS 是权重哈希校验的前提、评估快照与检查点绑定）。推送实现**有意不入库**（站点特定、不构成社区复现路径），文档只给结果约束。
- 身体与角色：`research/avatar/YUMI.md`、`YUMI_BODY.md`（VRM 许可约束见 `THIRD_PARTY.md`，署名不可去除）。
- 当前实验状态：`research/experiments/SENSORY_STABILITY_20260930.md`（完整连接组，vy 编码列与读出共 27,128 个可训练参数；两份固定候选各通过 72/72，新留出长走全程 2 米走廊各 9/9。共享补充池、早期同预算复现失败和速度相关残余漂移均须保留说明；不证明自主 CPG 或拓扑优势。上肢接入另行决策，实例保持开机）。
- 前序身体/任务对照：`research/experiments/ROUTE_EXECUTION_20260923.md`（普通MLP迈步、等预算旧奖励对照与四组留出，作为连接组迁移教师的来源）。
- 前序路线：`research/experiments/ROUTE_DECISION_20260922.md`；保留历史结论，当前证据以2026-09-23实验编年为准。
- 前序诊断：`research/experiments/ACTOR_SCALE_AB_20260922.md`、`GPU_CALIBRATION_20260921.md`、`INPUT_LEARNING_AUDIT_20260921.md`、`INPUT_SCALE_CLOUD_20260922.md`；方向讨论见 `research/proposals/2026-09-22/`。
- 双语说明：`README.md` / `README.zh.md` 改动需保持一致。
