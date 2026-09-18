# Yumi 物理身体适配 · 交接记录

更新：2026-09-15 17:50。**最终更新：2026-09-15 23:05——任务已完成，见下方"最终状态"。**接手 Agent 请先读完本文件再动手。

## 最终状态（2026-09-15 23:05）：✅ Yumi 身体行走已通过完整验收

云端 ppo9（从 iter65_baseline 出发，lr=1e-4、target_kl=0.015、value 预热 8 轮、奖励前向/横向/航向独立加权；改动明细见 [PPO8_FIXES_20260915.md](PPO8_FIXES_20260915.md)）**第 10 轮**即达 score 0.853（31/32）成为 best.pt，后续至第 175 轮未再超越。该 best.pt 完整验收 **16/16 全部通过**：9/9 初态 30 秒、120 秒长走 65.6 m（横漂仅 0.13 m，此前为 27 m）、3/3 轻推恢复、3/3 断连约 1 秒跌倒。验收原件在云端 `runs/yumi/heldout_yumi_ppo9.json`，本机副本 `runs/local/heldout_yumi.json`（检查点哈希 a7a4281f… 与本地 `runs/yumi/best.pt` 一致）。训练已于 23:1x 手动停止，日志备份在本机 `runs/yumi/ppo9.log`，配套 PPO 状态在 `runs/yumi/ppo_state_best.pt`。

本机 GPU 实时预览已验证通过：`FLYBODY_LOCAL=1 FLYBODY_DEVICE=cuda FLYBODY_BODY=yumi FLYBODY_CHECKPOINT=D:\Project\Neuromechfly\runs\yumi\best.pt` + `.venv-gpu` 起 8743 端口，实测 1.0× 实时，截图确认脚部贴地、接触标记与物理一致、无穿地。注意 `cloud_server.py` 已支持 `FLYBODY_CHECKPOINT` 环境变量。

### 剩余事项

1. ppo9 训练已于 23:1x 停止，最终 best.pt（第 10 轮，score 0.853）及配套 `ppo_state_best.pt` 已拉回本机 `runs/yumi/`。**注意：云端实例仍按有卡模式计费（¥1.88/小时）直到在 AutoDL 控制台手动关机**；云端 8742 预览进程也还在。
2. **README.md / research/REPORT.md 已过时**：仍写"以 G1 物理代理驱动 VRM 角色"，需更新为 Yumi 身体已验收的状态。
3. 扩展方向：上肢平衡、起步停步、转向行走、复杂地形。
4. 本机 GPU 偶发 NVML 初始化异常（torch 仍可用），CUDA 启动失败重试即可。

---

## 历史记录：训练迁移到云端（2026-09-15 17:50）

- 本地训练已随下班关机计划全部停止，本地无训练进程。训练已迁移到**新 AutoDL 4090D 实例**（原实例无 GPU 配额）：SSH `ssh -p <端口> root@<AutoDL 主机>（私有端点已脱敏，配置见 runs/cloud/target.json，不入库）`，项目目录 `/root/autodl-tmp/neuromechfly`，环境 `.venv`（`--system-site-packages` 复用基础镜像 torch 2.12.1+cu130；bootstrap 脚本本地备份在 `runs/cloud/bootstrap_yumi.sh`、`start_yumi_train.sh`）。
- 云端 PPO 于 17:40 左右以 `nohup` 持久启动（PID 在云端 `runs/yumi/train.pid`），日志 `runs/yumi/ppo5.log`，**预算 28800 秒（8 小时），约凌晨 01:35 自动结束**。从本地第 115 轮 `last.pt` + `ppo_state.pt` 续训。**约 9 秒/迭代，是本地的 4 倍**，iteration 5 评估 score 0.054（mean_duration 8.7s），已超过本地历史最佳。
- **训练结束后实例仍按有卡模式计费（¥1.88/小时）直到手动关机**。请在 AutoDL 控制台关机或设定时关机。取回结果：`scp -P <端口> -r root@<AutoDL 主机>:/root/autodl-tmp/neuromechfly/runs/yumi runs/yumi-cloud`（整个目录含 best.pt/last.pt/ppo_state.pt/training.json/ppo5.log）。
- 云端 `app/ppo_yumi.py` 的 best.pt 保护曾打包进旧版（key 少了 `extra` 层），已用 scp 单独更新为正确版本；当前运行进程是旧代码载入的，但保护只在启动时生效且 best 已被 iter 5 合法超越，无实际影响；下次重启训练即为正确逻辑。
- 云端预览（8742，FLYBODY_BODY=yumi 已 nohup 启动）：`code.tgz` 只含 `web/dist`，**`web/public/yumi.vrm` 需单独上传**（已传），否则页面显示"等待 Yumi 文件"并回退到 pixiv 示例角色。本地隧道：`./cloud.ps1 Preview`（端点见本地 target.json，不入库）。

## 首次云端验收（2026-09-15 晚，iter-65 best.pt，SHA e11ff401e9ed0ab4…）

结果存 `runs/local/heldout_yumi.json`（云端原件 `runs/yumi/heldout_yumi.json`，日志 `runs/yumi/acceptance-cloud1.log`）：

| 测试 | 结果 | 说明 |
|---|---|---|
| 9 初态 30 秒 | 1/9 | **全部 9 个存活满 30 秒、步态/触地/直立全过**，败在 direction（横向漂移 3.7–12.2 m）和 0.65 档 speed（实测约 0.48） |
| 120 秒长走 | 0/1 | 存活且速度达标，方向漂移超限 |
| 扰动 3 次 | 1/3 | **3/3 存活**（抗推很好），2 次方向超限 |
| 断连对照 3 次 | 0/3 存活 | 符合预期的阴性对照：连接断开即跌倒，证明连接组在参与控制 |

结论：生存与步态已解决；**唯一瓶颈是航向保持**（缓慢弧形漂移）。训练评估 episode 只有 15 秒且横向阈值（20% 目标距离）比验收宽松，策略没学过纠正累积漂移。下一轮换账方向：加长训练/评估 episode 到 30 秒、奖励里加横向位置/航向误差罚项、训练时加航向扰动让策略学会纠偏；配合 lr/探索退火或策略 EMA 抑制 16/16↔0/16 的边界振荡（评估是确定性的，见 train_full.py:26 evaluate 用均值动作不采样）。

## 第一轮本地训练结果（已完成）
- 第一轮实际执行 119 迭代 / 5414 秒。`best.pt` 和 `last.pt` 保存的都是第 **115** 轮；现有代码只在每 5 轮评估时保存，116–119 轮未落盘。下一轮应补上退出时的最终保存；`--resume-state` 目前只恢复 value 和 log_std，不恢复 Adam 状态。
- 第一轮最佳权重 SHA-256：`0071562a3f3e53f2f7648b59146fc030985298c2748f1bdc2cb61903f5c64dc1`。GPU 筛查为 **0/32** 通过，平均存活 **9.58068 秒**，score **−0.0265262**。4/32 撑满 15 秒，但都没有同时满足速度和方向要求。存活提升不能替代行走验收。
- 历史最佳保护在首次修复后仍读取了错误层级。已于本次接手改为 `extra.evaluation.score`，并用真实检查点验证读取到 −0.0265262；此前读顶层字段会得到负无穷，仍可能覆盖历史最佳。修改不触发训练。

### 原生 MuJoCo 短程诊断（训练种子 1001，非最终留出验收）

使用同一第 115 轮权重、Yumi 身体、GPU 脑推理，目标每回合 15 秒，无教师。

| 目标速度 | 存活 | 平均前进速度 | 横向偏移 | 结果 |
| --- | --- | --- | --- | --- |
| 0.35 m/s | 7.74 s | 0.058 m/s | 0.223 m | 跌倒、速度不达标 |
| 0.50 m/s | 8.16 s | 0.203 m/s | 0.111 m | 跌倒、速度不达标 |
| 0.65 m/s | 4.78 s | 0.818 m/s | 0.276 m | 跌倒、速度不达标 |

三回合均记录到交替落脚，但接触切换计数不能单独证明自然步态。脑推理中位数 8.94–9.22 ms，P95 9.87–12.34 ms；本机 GPU 可以实时推理。原生与 GPU 筛查的初态随机数实现不同，这组数据不能单独判定物理引擎存在偏差。

可复现脚本：`runs/yumi/phase1_review/review.py`；完整结果、身体文件哈希和逐帧回放：`runs/yumi/phase1_review/report.json`、`replay_0.json` 至 `replay_2.json`。完整 9 初态 / 120 秒长走 / 扰动 / 断连验收仍未完成。

### 浏览器预览

预览已按用户要求改为**手动暂停**：跌倒后脑推理和物理继续运行，页面显示“已跌倒 · 继续模拟”，直到点击暂停。训练与正式评估仍按原来的跌倒条件结束回合；预览跌倒后的累计时间不能计入行走存活成绩。已实测第 8.02 秒跌倒后继续至第 18.24 秒，暂停冻结时间、继续恢复步进均正常；记录在 `runs/local/manual_pause_check.json`。

此前已在外部 Edge 检查实际画面：Yumi 正常加载，暂停位姿没有明显贴地或膝向异常，评估页正确显示未完成。此后已按用户录屏要求重启 8743，当前为 Yumi 身体 + CUDA 推理，加载第 115 轮权重 `u0000115`，哈希与上文一致。训练保持暂停。前端训练图仍按模仿学习的 `samples/action_mse` 读取，尚未适配 PPO 的 `iteration/reward`，训练面板的初始化文字是旧状态。

神经活动显示已按用户关于“运行时只有微小闪烁”的反馈更新：默认“变化”视图显示相邻快照的活动差 / 仿真时间差，固定满量程 ±0.5/s；“强度”视图固定 ±1。超范围的颜色与点大小饱和，RMS 数值保留实际幅度。点的大小、亮度、颜色均由实测信号驱动，没有时间驱动的装饰闪烁。保持约 10 Hz 采样及 2,048 个胞体显示点；未添加突触连线或脑内四步传播动画。重置、更换回合、超过 0.5 秒的采样间隔不计算跨段差值；暂停保留最后快照。

验证：`node --test web/brain-activity.test.js`、`npm run build` 通过；Edge 实际行走、两种视图切换和暂停状态已检查。新增前端计算位于 `web/brain-activity.js`，点绘制位于 `web/main.js`。真实采样记录位于 `runs/yumi/phase1_review/brain_activity.json`（约 6.4 MB，大体积仅本机保存、不入库；前端测试使用合成数据，不依赖该文件）。

## 任务背景

项目主线（见 README.md、research/REPORT.md）：完整果蝇连接组（166,700 神经元 / 2558 万连接）直接输出 12 关节动作，经 MuJoCo 物理驱动 VRM 角色行走。此前身体用的是 **G1 物理代理**（Yumi 只是外观跟随）。本阶段目标：**建立 Yumi 自己的物理身体并让其学会行走**。

## 已完成的工作

### 1. VRM 骨架解析（完成）

- 新文件 `app/yumi_skeleton.py`：解析 GLB/JSON，提取 VRM humanoid 骨骼世界坐标。
- Yumi 实测尺寸：髋高 0.97231 m，髋半宽 0.07229，大腿 0.39215，小腿 0.44377，踝高 0.09818，踝到脚尖 0.11655，肩宽 0.14578，臂长 0.47897，头顶 1.44344（整体约 1.55 m 身高）。

### 2. Yumi MuJoCo 物理身体（完成，站立/跌倒物理正常）

- `app/yumi_description/yumi.xml`：12 关节人形，**关节顺序、轴向、零位与 G1 完全一致**（hip pitch/roll/yaw → knee → ankle pitch/roll，每腿 6 个），因此脑网络 12 维动作 / 47 维观测接口不变，前端 VRM 骨骼映射数学上精确对应（旋转复合顺序已核对：MuJoCo R_y(pitch)·R_x(roll)·R_z(yaw) ≡ 前端 qX·qZ·qY）。
- 质量假设（非实测，文件内有注释）：总 38 kg，Dempster 节段比例；躯干+头+臂 25.7 kg 固定在骨盆（与 G1 代理一致，手臂不参与控制）。
- 碰撞：足底 4 球/脚（半径 0.022）；躯干与腿胶囊 `contype=0 conaffinity=1`（只与地面碰撞，避免自碰撞），跌倒后躺在地面（静止高度约 0.13 m）。
- `app/yumi_description/yumi.yaml`：PD 增益与 G1 相同（kps [100,100,100,150,40,40]×2），default_angles 与 G1 相同（保持观测基准 qpos-home 不变）。initial_height 0.962（含髋关节在骨盆下 0.039，注意：早前 0.923 是错的，脚会穿地），fall_height 0.55。
- 力矩上限已提高到不低于 G1：髋 pitch 120 / roll 150 / yaw 88，膝 150，踝 60 N·m（备份在 yumi.xml.bak / yumi.yaml.bak；原值更低，实测对结果无影响）。
- `app/sim.py`：`Body(load_motor_policy=False, robot='g1'|'yumi')`，ROBOTS 字典集中管理 cfg/xml/initial_height/fall_height/hips_height；跌倒阈值、初始高度已参数化。
- `app/gpu_body.py`：`GPUHumanoid(worlds, robot=...)` 同样支持 yumi（mujoco-warp 已装入 `.venv-gpu`：mujoco-warp==3.13.0, warp-lang==1.17.0，用 `uv pip install --python .venv-gpu/Scripts/python.exe ...` 安装）。

### 3. 控制链路接线（完成）

- `app/cloud_server.py`：环境变量 `FLYBODY_BODY=yumi` 切换身体；`/api/meta` 的 body 增加 `robot` 和 `hips_height`；LOCAL 评估文件按 robot 分名（`runs/local/heldout_yumi.json`）。
- `web/main.js`：avatarScale 改用 `meta.body.hips_height`（Yumi 时为 0.97231，显示比例≈1.0，物理米数=显示米数）；多处硬编码"G1 物理代理"文案改为按 robot 动态显示。前端已 `npm run build` 通过（npm 使用本机全局安装路径（个人环境，略））。
- `app/evaluate_full.py`：新增 `--robot` 参数，可做 Yumi 的 9 初态/长走/扰动/断连验收。

### 4. 现有权重测试（完成，结论：不迁移）

- 云端 G1 检查点 `runs/cloud/best.pt` 在 Yumi 身体上：三个速度均约 1.6–1.8 秒跌倒（有交替踏步，走约 1 m 后失去平衡）。
- **现有 G1 教师直接驱动 Yumi 身体只能撑约 1.7 秒**，其示范不能直接用于该身体。此结果不排除重新训练 Yumi 教师，再蒸馏 / DAgger 训练完整连接组。
- 提高力矩上限对直接迁移无效果（但保留更高上限供 PPO 使用）。

### 5. PPO 微调（第一轮结束，后续训练已暂停，尚未成功）

- 新文件 `app/ppo_yumi.py`：以 `runs/cloud/best.pt` 为起点，PPO 强化学习。奖励 = 1.5·速度跟踪 + 0.5·直立 + 0.3·朝向 + 0.2·存活 + 0.4·双足交替（踝部高度切换）− 小动作/动作变化惩罚 − 1 跌倒终止。独立 value MLP；可学 log_std（初值 0.4）；GAE 0.99/0.95；clip 0.2；32 worlds × 128 steps；约 37 秒/迭代（瓶颈是 2558 万边的稀疏反向传播）。`--resume-state` 可续 value/log_std（`runs/yumi/ppo_state.pt`）。
- **第一轮 5400s 已完成（119 迭代）**：mean_duration 前 90 轮在 1.2–2.8 秒波动，第 95 轮起突破：95→3.7，100→4.8，105→5.0，**110→9.8，115→9.6 秒**，score 升到 −0.027（评估上限 15 秒）。successes 为 0/32。详细结果见文首。
- **第二轮已按用户要求停止**。`train.pid`、`training.json` 中的 PID 只是旧记录，启动前应核对真实进程。
- 教训：不要让两个训练进程同时跑（曾经 DAgger 没杀干净抢 GPU 还互写 training.json；杀进程用 `"/c/Windows/System32/taskkill.exe" //F //PID <pid>`，Git Bash 的 kill 不可靠）。

### 6. 预览服务（正在运行）

- 8743 端口：Yumi 身体 + CPU 脑推理（GPU 被训练占用，CPU 约 187 ms/决策，慢动作但可看站姿对齐）。启动方式：`FLYBODY_LOCAL=1 FLYBODY_DEVICE=cpu FLYBODY_BODY=yumi .venv/Scripts/python.exe -m uvicorn app.cloud_server:app --host 127.0.0.1 --port 8743`，PID 在 `runs/local/yumi-server.pid`。
- 已验证 meta 正确报告 yumi/0.97231，物理状态流正常（t=0.9 站姿 h=0.864）。
- **画面一致性（数值验证已完成，2026-09-15 下午）**：脚本 `runs/yumi/vischeck.py` 离线复现 main.js updateBody() 的映射数学并做 FK 对比。结论：**一致**——旋转映射 600 随机样本足部朝向误差 0.000°；home 站姿物理足底 z=−0.0001（贴地不穿地）、渲染预测 +0.0044（±0.01 内）；膝向 600 样本 0 例不符。唯一系统差：VRM 踝骨相对物理踝轴恒定在胫骨后方 ~4 cm（VRM 文件自身几何，非映射 bug），垂直向仅差 0.8 mm；副作用是前端 footMarkers 相对物理接触点恒定偏后 ~4 cm（如需更准可改用 toe 骨）。浏览器截图级验证未做，但数值层面已无已知风险。

## 下一步建议（按优先级）

1. **保持暂停，先依据诊断确定下一轮方案**。已有平衡改善，继续完整连接组 PPO 有验证价值；下一轮应同时关注存活、速度误差、横向偏移和足部接触，不能仅用突破 5 秒决定加时。恢复前补上最终保存、检查点与 PPO 状态配对，以及预览状态显示。可能的训练改进包括：
   - 增大 worlds（32→64）提高样本多样性；epochs 2→4 提高样本利用；
   - 课程学习：先低速 0.15–0.35 m/s；
   - 冻结 edge_delta 只训 encoder/readout/bias（反向快 3–5 倍，牺牲"脑连接也学习"的叙事，可作为对照实验）；
   - 参考 G1 步态做 DeepMimic 式轨迹跟踪奖励（关节语义相同，可直接映射参考轨迹）。
2. **验收**：行走稳定后跑 `.venv-gpu/Scripts/python.exe -m app.evaluate_full --robot yumi --checkpoint runs/yumi/best.pt --output runs/local/heldout_yumi.json`（9 初态 30 s + 120 s 长走 + 3 扰动 + 3 断连）。
3. **画面一致性**：8743 预览（届时改 FLYBODY_DEVICE=cuda 达实时）截图检查贴地/穿地/膝向/起伏；然后更新 README、research/REPORT.md 与本文件。
4. 之后才是上肢平衡、起步停步、复杂地形。

## 注意事项

- `runs/yumi/` 下的 `train.log`/`ppo.log`/`ppo2.log` 是早期失败尝试（DAgger、无塑形 PPO）的日志，别看错；当前有效日志是 `ppo3.log`。
- ppo_yumi.py 的 best.pt 保护读取 **extra.evaluation.score**。此前首次修复读取了错误层级，已再次修正。长期训练后仍建议核对评估时长、身体版本和权重哈希，避免比较不同条件下的分数。
- cloud_server.py 已修复：RUNS 按 FLYBODY_BODY 选择（yumi → runs/yumi），预览可读到 yumi 检查点与 PPO 的 training.json 指标；8743 预览服务器需重启才生效。
- 前端构建产物在 `web/dist`，服务器只服务构建后的页面；改 main.js 后必须重新 build。
- Git Bash 中 powershell 不在 PATH，用全路径 `/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe`；后台启动训练用 Start-Process（启动模式见 `start.ps1`；8743 旧入口 `local-full.ps1` 已移除）。
