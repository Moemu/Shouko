# 与本项目的工程对照

本页把外部材料按本项目遇到的开发问题重新组织：每一行给出可查证的外部实现、对应的本项目代码入口，以及一个可以用小实验回答的问题。先读 [证据边界](VERIFICATION.md) 了解资料等级。页内所有本项目路径均相对于仓库根目录，链接可直接打开。

## 按开发问题查资料

| 本地问题 | 外部材料 | 本项目阅读入口 | 可验证的小问题 |
|---|---|---|---|
| 评估噪声把偶然高分固化成最佳权重 | hae [ningen/school/tune.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/tune.mjs) | [app/ppo_yumi.py](../../../app/ppo_yumi.py)、[app/test_training_control.py](../../../app/test_training_control.py)、[ppo17 编年](../../experiments/PPO_OBS50_20260920.md) | 同权重是否被误标为新候选；真实候选是否在配对种子上稳定改善 |
| 训练与预览的时间和观测契约 | hae [ningen/school/trial.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/trial.mjs) 与 [ningen/web/app.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/app.js) 的同步 / 异步区别 | [app/full_brain.py](../../../app/full_brain.py)、[app/sim.py](../../../app/sim.py)、[app/gpu_body.py](../../../app/gpu_body.py)、[app/evaluate_full.py](../../../app/evaluate_full.py) | 相同观测字段、时钟、动作尺度与身体配置是否一致 |
| 扩维后保留旧能力 | 在线 [migrate_reflex.mjs](https://hae.satoru.net/ningen/school/migrate_reflex.mjs) | [app/expand_observation.py](../../../app/expand_observation.py)、[app/test_policy_contract.py](../../../app/test_policy_contract.py) | 原输出逐点等价；新增参数是否有明确初值；旧权重与图能否追踪 |
| 局部学习的最小接口 | hae [brain.c](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/src/brain.c)、[reader.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/juku/reader.mjs)、[chooser.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/chooser.mjs)、[reflex.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/reflex.mjs) | [app/full_brain.py](../../../app/full_brain.py)、[app/ppo_yumi.py](../../../app/ppo_yumi.py) | 独立行为选择任务中局部更新是否有效；是否影响原行走基线 |
| 接入物理上肢 | hae [body-worker.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/body-worker.js)、[muscles.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/muscles.mjs)、[MyoSim](https://github.com/MyoHub/myo_sim) | [app/yumi_skeleton.py](../../../app/yumi_skeleton.py)、[app/yumi_description/](../../../app/yumi_description/)、[Yumi 身体说明](../../avatar/YUMI_BODY.md) | 上肢质量、关节轴和碰撞是否正确；视觉动作是否真的参与动力学 |
| 加入感官闭环与目标任务 | hae [body-worker.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/body-worker.js) 的 `senses` / `smelling`；[FFREP 目标追逐](YAKSHAWAN.md#适合借鉴的设计) | [app/sim.py](../../../app/sim.py)、[app/studio_server.py](../../../app/studio_server.py) | 输入来自场景真值还是传感模拟；打乱 / 延迟输入时行为如何变化 |
| 稀疏神经计算性能 | hae [brain.c](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/src/brain.c)、[export_graph.py](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/tools/export_graph.py)、[flybrain.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/flybrain.js) | [app/full_brain.py](../../../app/full_brain.py) | 同负载、同精度、同时间含义下是否减少延迟和内存 |
| 脑活动可视化 | [FFREP 三视图](YAKSHAWAN.md#适合借鉴的设计)、hae [life/README.md](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/life/README.md) | [web/](../../../web/)、[app/studio_server.py](../../../app/studio_server.py)、[YUMEKA 参考](../../avatar/YUMEKA_REFERENCE.md) | 显示点是否对应实际计算节点；活动、变化量与推断贡献是否分开 |
| 公共演示与回放 | hae [ningen/web/app.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/app.js)、[ningen/school/server.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/server.mjs) | [web/](../../../web/)、[app/studio_server.py](../../../app/studio_server.py) | 可否从一条记录查到模型、配置、输入事件与失败原因 |
| 连接组是否带来额外价值 | [FlyGM](https://arxiv.org/abs/2602.17997v3)、[FLYNN](https://arxiv.org/abs/2607.00025v2)；[Blank Lobe 消融构想](YAKSHAWAN.md#适合借鉴的设计) | [app/train_full.py](../../../app/train_full.py)、[app/ppo_yumi.py](../../../app/ppo_yumi.py)、[app/evaluate_full.py](../../../app/evaluate_full.py) | 同预算随机图、MLP、去除附加模块的结果是否不同 |

[app/server.py](../../../app/server.py) 与 [app/cloud_server.py](../../../app/cloud_server.py) 是历史或特殊用途入口，新增功能走 [app/studio_server.py](../../../app/studio_server.py)。

## 1. 评估与筛选：可借鉴的分离

hae 的参数搜索把候选积累到足够评估后再用于网页展示。其中一个工程分离值得注意：正在尝试的状态，与对外展示的已验证状态各自存放。

本项目 2026-09-21 的 CPU 修正已经加入同权重重评防护并更新了观测与身体契约：防护逻辑在 [app/ppo_yumi.py](../../../app/ppo_yumi.py)，对应测试在 [app/test_value_diagnostics.py](../../../app/test_value_diagnostics.py)；观测与身体契约的检查在 [app/test_policy_contract.py](../../../app/test_policy_contract.py)。本页汇总的是可继续研究的候选问题。

仍可研究的协议：

1. 冻结一组验证场景、种子、噪声模式与判据。
2. 基线与候选使用成对场景；同时报告个体结果、均值、散布和失败数。
3. 选择用验证集，最终测试集分离；预先限制反复筛选次数。
4. 记录检查点、图、身体、观测、时钟及奖励配置的标识。
5. 同一权重的追加评估扩充证据，不生成一次新的「学习进步」。

这些是候选协议。具体次数、置信区间方法与资源预算需要结合测得的噪声确定。hae 的固定 margin 对应它自己的奖励尺度，套用到本项目的奖励尺度没有依据。

## 2. 检查点迁移：按语义映射

在线 [migrate_reflex.mjs](https://hae.satoru.net/ningen/school/migrate_reflex.mjs) 用原图中边的位置把旧可塑增益迁到新通路，检查旧边保留、新增边初始化、布局长度一致。

可以借鉴的是不变量检查：图排序一旦变化，数组位置可能不再表示同一突触；更稳妥的做法是绑定图哈希，必要时用明确的前后神经元 ID 作为键。

本项目 47→50 维观测扩展已经完成（见 [app/expand_observation.py](../../../app/expand_observation.py) 与 [app/test_policy_contract.py](../../../app/test_policy_contract.py)）。后续扩展需要同时保持字段顺序、归一化、单位、动作接口与物理时钟不变。静态前向等价与长轨迹一致是两个不同判据；边际稳定的系统可能放大微小数值差异。

## 3. 局部可塑性：要学习的对象

hae 展示了三种不同用途的局部可塑：分类读出、预设行为方案选择、局部运动反射调节。它们的奖励来源和输出语义各不相同。

若未来开展相关实验，可以从少量明确的行为方案开始：冻结已有行走策略，测高层选择是否可学习，并保留随机选择、固定选择、普通小型可训练读出作为对照。

逐轮的突触范围、增益分布、奖励与轨迹都值得保存。在线学习开关关闭后需要确定性评估，并检查是否遗忘旧行为。「局部更新一定更稳定」与「多巴胺一定替代反向传播」都还是待验证的假设。

## 4. 身体复杂度：任务难度与生物相似度

FFREP 的四足比人形更容易产生目标运动，这是作者的观察。可能的解释包括支撑几何、动作维度、奖励密度和神经映射；现有材料无法确定哪一个是主因。

本项目已有 G1 与 Yumi，可以先用现有身体检查配置、质量与力矩变化带来的差异。新增四足或 416 肌肉人体会扩大模型、控制与验收范围，属于需要用户定稿的方向决策。

hae 的身体控制代码适合查 IK、支撑与肌肉分配的实现方式；它不提供把本项目当前关节策略直接迁成全身肌肉控制的证据。

## 5. 用户可理解的实验界面

可以围绕现有页面能力（暂停、重置、推扰、断连、训练曲线、原始评估）组织三个简短工作流：

- **跟随命令**：目标与实际速度、方向、脚触地和跌倒状态同屏显示。
- **扰动恢复**：标出施力时刻、方向和恢复过程，保留失败回合。
- **模型对照**：相同输入回放正常 / 断连，标明模型及身体版本。

以上是交互设计参考。文案改动仍需同步 zh / en 字典（仓库约定，见 [AGENTS.md](../../../AGENTS.md)）；角色资产沿用各自的原有许可。

若以后加入喂食、等级或用户反馈，参与次数、训练量与独立能力指标是三类不同的数据。视觉奖励与神经权重更新也是两回事，除非确有更新并且可以追踪。

## 6. 性能优化的比较单位

比较 WASM LIF、本项目的 GPU 稀疏活动率网络与小网络时，至少注明：

- 实际活跃节点与总节点、神经元对边与突触计数。
- 仿真神经时间、控制频率、批量环境数、是否反向传播。
- 冷启动与稳态、墙钟延迟的中位数 / 尾部、内存峰值。
- 输入分布、数值精度、线程数、硬件与版本。

浏览器下载体积、糖刺激推理速度与 PPO 训练吞吐是三个不同问题，各自的瓶颈不同；优化前先确定要解决的是哪一个。
