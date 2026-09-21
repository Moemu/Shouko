# 无卡修正交接：2026-09-21

本次按用户要求完成本地修正及 CPU 验证。未启动训练、完整连接组推理或 GPU 物理仿真，未改云端进程或检查点。已有实验编年和其他 Agent 的修改保留。

## 已修正

### 1. 观测与身体接口

- `app/full_brain.py` 提供内存映射的检查点配置读取，不为读取维度加载连接组。
- `app/evaluate_full.py` 和主入口 `app/studio_server.py` 按检查点创建 47/50 维策略及身体。预览切回旧模型时重新解析身体，避免沿用上一份模型的接口。
- `app/sim.py` 支持追加横向/前向线速度及相对髋高。原生与批量路径使用相同的字段顺序和配置尺度。
- 新训练检查点记录观测维度、步态时钟、线速度尺度、髋高参考、物理时间步及既有动作接口。GPU 路径现在使用配置中的动作尺度、控制步数和时间步。
- 修复检查点中 `control_decimation=10.0` 导致原生 `range()` 失败的问题；拒绝非整数控制步数。
- 原生评估与预览节奏采用实际控制时间步。评估报告记录维度与时钟；服务端检查报告中已有的身体接口哈希。

**历史兼容约定**：没有显式时钟字段的 47 维检查点，原生预览仍用 1.0 秒，并保留原有六字段接口哈希；同类旧检查点在 GPU 训练路径仍用历史 0.8 秒。已有 50 维实验的原生路径按其训练实现采用 0.8 秒。这是明确保留的历史差异，不代表过去两条路径已经一致。带完整接口的新检查点在两条路径均使用记录值。

真实本地文件仅做元数据读取及 CPU 身体创建。结果见 `runs/yumi_obs50/review_20260921/local_contract_check.json`：`best.pt`、`best_tall.pt` 为 47 维；`best_tall_obs50.pt` 为 50 维。该检查不证明完整网络行走成功。

### 2. 检查点选择与预热门控

- 启动时重复评估同一策略，不重写 `best.pt`、配套 PPO 状态或历史选择门槛。
- 训练中的候选分支也检查张量身份。冻结策略期间的评估波动不再被登记成新的最佳策略。
- 张量比较逐项搬到 CPU，减少同时保留完整 CPU 副本的内存需求。
- 价值门控要求两个完整、连续的五期窗口都低于阈值；最早在第六期通过。截止仍未通过则退出，不放行 actor 更新。通过后下一期即可更新 actor。
- 门控默认仍关闭。固定损失阈值、两次均值和选择 margin 都只是工程规则，尚未得到统计标定；本次没有清洗历史门槛或宣布其合理。
- 日志保留旧 `lr` 字段，同时增加所有优化器参数组的名称及学习率，避免把偏置组学习率误读为主学习率。

真正候选与基线的成对重复评估仍属于下一阶段实验，不能把本次身份保护解释为已消除全部筛选偏差。

### 3. 价值诊断使用实际训练样本

`app/value_diagnostics.py` 与 PPO 共用 GAE 计算。新增 `--dump-rollout` 保存当前启动的第一批真实 rollout，时间点在 PPO 更新之前。训练随后继续，不是“只采样后退出”开关。

导出包含实际观测、动作、奖励、终止掩码、价值预测及末步 bootstrap、GAE、value target、实际采样标准差、检查点哈希、身体接口、种子和奖励参数。没有全脑激活。

CPU 分析先核对张量形状、有限性和目标的一致性，再计算 MSE、RMSE、偏差、目标方差、R² 和解释方差，并按命令速度及终止状态拆分。常量目标的 R² 返回空值。这里是同批、带 bootstrap 的诊断，**不证明价值网泛化、不可约噪声占比或行走成功**。

旧的 `runs/yumi_obs50/diagnose_value.py` 已改为新入口的薄包装，避免误启动那套奖励、探索噪声和状态对齐均不一致的 GPU 诊断。此前讨论文档对旧脚本的引用描述的是修复前状态。

## 验证

以下检查在 CPU 上通过：

```powershell
.venv-gpu\Scripts\python.exe -X utf8 -m app.test_policy_contract
.venv-gpu\Scripts\python.exe -X utf8 -m app.test_value_diagnostics
.venv-gpu\Scripts\python.exe -X utf8 -m app.test_studio_server
.venv-gpu\Scripts\python.exe -X utf8 -m app.test_training_control
npm test
git diff --check
```

另对修改的 Python 文件执行了 `py_compile`。

策略测试使用四神经元合成连接组和真实 MuJoCo 身体，覆盖 47/50 维切换、历史时钟、实际批量观测函数的 CPU 计算、短步原生评估，以及报告接口哈希拒绝。服务端测试也改用这一小型连接组，避免回归测试加载完整脑网络。

PPO 测试覆盖常量回报 bootstrap、终止边界、R² 与解释方差的区别、门控窗口、导出往返，以及实际启动/训练中选择分支的同权重保护。合成测试数据不属于训练成果。

## GPU 可用后的顺序

1. 先检查实例剩余 RAM/VRAM，在小环境数下验证 Warp 内核及完整 50 维链路。本次未执行 CUDA 图、GPU 物理或完整连接组验收。
2. 在已经选定并限定预算的训练命令后添加 `--dump-rollout runs/yumi_obs50/diagnostics/rollout.pt`。不要仅为了这个参数擅自启动长训。若需要诊断续训后的 critic，应从对应且哈希匹配的 PPO 状态启动。
3. 将导出文件复制到本地，用 CPU 分析：

```powershell
.venv-gpu\Scripts\python.exe -X utf8 -m app.value_diagnostics --rollout runs/yumi_obs50/diagnostics/rollout.pt --output runs/yumi_obs50/diagnostics/value.json
```

4. 根据真实批次再决定固定数据拟合、按环境/完整轨迹留出检查和小规模策略对照。当前没有导出的真实批次，因此没有新增价值网质量结论。

本次资源快照与检查摘要保存在 `runs/yumi_obs50/review_20260921/local_fix_checks.json`。本地内存余量偏低且 GPU 正被使用，因此没有追加完整网络实验。无需为本次 CPU 修正支付云 GPU 费用。
