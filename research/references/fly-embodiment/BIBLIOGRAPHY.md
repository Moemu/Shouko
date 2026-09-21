# 论文、数据与工具入口

核查日期：2026-09-21。下表按开发问题组织，只收录与本次主题直接相关的一手研究、官方数据或上游项目。摘要是阅读导向，不是复现结论。

## 连接组与神经仿真

| 来源 | 本次核查层级 | 适合查什么 | 使用边界 |
|---|---|---|---|
| [MaleCNS 官方项目](https://male-cns.janelia.org/) / [下载](https://male-cns.janelia.org/download/) | 官方项目页面 | 脑与 VNC、注释、版本、数据许可、查询工具 | 结构数据不是完整生理参数；使用本地导入与哈希口径 |
| [FlyWire Codex](https://codex.flywire.ai/) | hae 的明确数据来源；未登录查询 | 神经元 ID、细胞类型与连接 | 与 MaleCNS 的 ID 和群组不能直接互换 |
| [Shiu 等，Nature 2024](https://doi.org/10.1038/s41586-024-07763-9) / [作者代码](https://github.com/philshiu/Drosophila_brain_model) | 作者仓库 README 与代码入口；期刊正文抓取受限 | 原始 LIF、激活 / 静默、传感运动路径、Brian2 参考 | hae 增加的可塑规则不属于原论文全部结论 |
| [hae 核心库](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/README.md) | 固定代码与说明 | WASM 实现、图压缩、仿真测试、API | 速度只对应指定任务和环境 |

MaleCNS 官方页面记录 v1.0 于 2026-06-08 发布，论文于 2026-09-03 发表。这与账号演示发布日期是不同时间线。

## 局部可塑性的生物依据

| 研究 | 核心问题 | 对实现的价值与边界 |
|---|---|---|
| Hige 等，2015，*Heterosynaptic Plasticity Underlies Aversive Olfactory Learning in Drosophila*，[论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068/)，DOI `10.1016/j.neuron.2015.11.003` | 嗅觉输入与多巴胺神经元激活如何形成分区、刺激特异的突触抑制 | 为 KC→MBON 局部更新提供生物背景；不证明任意人形关节奖励映射有效 |
| Handler 等，2019，*Distinct Dopamine Receptor Pathways Underlie the Temporal Sensitivity of Associative Learning*，[论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9012144/)，DOI `10.1016/j.cell.2019.05.040` | 刺激先后顺序与不同受体如何影响增强 / 抑制 | 提醒实现关注时间关系和受体机制；软件中的负调制量不是负浓度 |

本次核对论文题录、摘要及相关段落，未重做生物实验或逐式验证 hae 的简化规则。

## 直接相关的机器学习控制论文

### FlyGM

[Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly，v3](https://arxiv.org/abs/2602.17997v3)；[HTML 全文](https://arxiv.org/html/2602.17997v3)。作者 Zehao Jin、Yaoye Zhu、Chen Zhang、Yanan Sui。该版本日期为 2026-06-14。

作者以连接组图构造仿真果蝇控制器，并报告深度强化学习下的移动任务结果。对我们最有用的是方法与对照设计：结构先验、图 / 非图基线、匹配节点和边数的随机图，以及神经可视化是否受处理流程影响。

优先读 §3、§4.1、附录 A、E.1、F、G。E.1 将相同分析流程用于随机图；这比只展示连接组的活动图更能支持结构贡献讨论。本文对象仍是仿真果蝇，不能直接将结果推广至 Yumi。此次没有运行作者代码，也未确认可用的完整发行包。

### FLYNN

[FLYNN: Robust Neural Network for Robot Navigation using Fly Brain Topology，v2](https://arxiv.org/abs/2607.00025v2)；[HTML 全文](https://arxiv.org/html/2607.00025v2)。作者 Benquan Wang、Jingdao Chen。该版本日期为 2026-07-13。

作者研究连接组拓扑的 RNN 在 MuJoCo 视觉导航、分布外场景和感觉缺失下的表现。适合参考参数规模对照、小世界网络基线、视觉缺失条件和内部表征分析。优先读 §II 与 §III。

论文明确给出 [代码入口](https://github.com/ben-gitdev/fly-gym)；本次仅定位该链接，未审查仓库或核验发行产物。此 `fly-gym` 与 NeuroMechFly 的 `flygym` 是不同项目。作者报告的鲁棒性不等于我们的控制网络已经具有相同性质。

## 身体与物理

| 来源 | 核查层级 | 适合查什么 | 注意 |
|---|---|---|---|
| [NeuroMechFly / FlyGym](https://github.com/NeLy-EPFL/flygym) / [文档](https://neuromechfly.org/) | 官方仓库页面；根文档抓取未取得正文 | 果蝇身体、感觉闭环、CPG 与反射控制 | 按固定版本阅读，勿混用不同代 API；本项目人形运行不依赖 FlyGym 控制身体 |
| [MyoSim](https://github.com/MyoHub/myo_sim) | 官方 README | MuJoCo 肌骨模型；MyoFullBody 的身体组成 | 模型复杂度不等于学习难度已解决；hae 对模型有派生处理 |
| [MyoSuite 文档](https://myosuite.readthedocs.io/en/latest/suite.html) / [2022 论文](https://arxiv.org/abs/2205.13600) | 官方文档与论文摘要 | 肌肉驱动、接触丰富的动作任务 | 与当前 12 关节目标控制接口不同 |
| [MuscleMimic 2026](https://arxiv.org/abs/2603.25544) / [作者项目](https://github.com/amathislab/musclemimic) | 论文摘要，项目链接由摘要提供 | 全身肌骨运动学习、动作重定向、模型与检查点线索 | 未验证仓库、硬件预算或与 Yumi 的兼容性 |

## 既有邻近工程

[本项目早期 REPORT](../../REPORT.md)已记录 [FlyCube](https://github.com/lntegrals/flycube-public)、[fly-chess](https://github.com/tolatolatop/fly-chess) 与 Unitree 教师来源。它们在本次未重新做版本审计，仅作为既有线索保留。不要把旧 README 中的成绩或缺陷默认当成其当前版本状态。

## 术语对照

| 名称 | 含义 | 不要混淆 |
|---|---|---|
| KC | Kenyon cell，肯扬细胞 | 与任意人工隐藏层节点不同 |
| MBON | 蘑菇体输出神经元 | hae 某些任务读取其输入驱动，不直接读其发放数 |
| DAN | 多巴胺能神经元 | 人工奖励标量不自动成为真实 DAN 模型 |
| DN | 下行神经元 | 指定功能映射仍需要实证与接口设计 |
| VNC | 腹神经索 | FlyWire 脑图与 MaleCNS CNS 的覆盖范围不同 |
| LIF | 漏积分发放模型 | 不等于我们的连续活动率网络 |
| 活动率状态 | 连续模型中的神经活动变量 | 不应标成生物脉冲频率，除非确有定义与换算 |
| 可塑增益 | 原连接上的可训练倍率 | 与新增连接、人工脑叶、短时动态状态不同 |
| IK | 逆运动学 | 工程 IK 产生的动作不是网络自主学出的动作轨迹 |
