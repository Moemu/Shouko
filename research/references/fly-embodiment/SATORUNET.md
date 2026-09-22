# satorunet / hae：实现阅读笔记

hae（[网站](https://hae.satoru.net/) · [源码](https://github.com/satorunet/hae)）是 @satorunet 用 FlyWire 果蝇脑连接组做的一组浏览器实验：脉冲神经网络在 WASM 里运行，识字、人体角色、算盘、LIFE 等任务分别演示一种能力。这页按子系统记录本次定点阅读的发现，供工程对照或移植前评估取用。

阅读约定：下文文件路径相对 hae 固定提交 [`d4551c26`](https://github.com/satorunet/hae/tree/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d) 的仓库根目录，文件名即链接，也可从 [源码索引](SOURCE_INDEX.md) 逐个打开；标「在线」的指 [hae.satoru.net](https://hae.satoru.net/) 的部署文件，没有对应的公开提交。证据等级与版本差异见 [入口页](README.md#资料等级) 与 [证据边界](VERIFICATION.md)。源码经过定点阅读，未做完整安全审计或运行复现。

## 公开进展与任务划分

| 任务 | 可核实的材料 | 解读边界 |
|---|---|---|
| 平假名与数字识别 | [juku/reader.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/juku/reader.mjs)、[trainer.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/juku/trainer.mjs)；[识字原帖](https://x.com/satorunet/status/2098973056769384724) | 学习 KC→MBON 增益；地面写字另有预定笔画与 IK |
| 人体与猫耳角色 | 9 月 15 日 [原帖](https://x.com/satorunet/status/2099850928237285657)；[ningen/](https://github.com/satorunet/hae/tree/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen) | 四肢着地觅食；双足行走未在其中 |
| 声音反馈 | 9 月 16 日 [原帖](https://x.com/satorunet/status/2100019534937874824) | 角色表现层；认知能力的变化未在本次材料中 |
| 算盘 | 9 月 18 日 [原帖](https://x.com/satorunet/status/2100749866129580200)；[soroban/](https://github.com/satorunet/hae/tree/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/soroban) | 主要运算由新增人工脉冲电路实现 |
| LIFE | 9 月 15 日 [原帖](https://x.com/satorunet/status/2099738298331930680)；[life/README.md](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/life/README.md) | 主页面并非每只果蝇各跑一份全脑 |

## 1. LIF 与 WASM 核心

**源码核查入口**：[flybrain/src/brain.c](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/src/brain.c)、[flybrain/flybrain.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/flybrain.js)、[flybrain/README.md](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/README.md)、[flybrain/test/](https://github.com/satorunet/hae/tree/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/test)。

脑来自 Shiu 模型与 FlyWire v783，使用膜电位、突触状态、阈值、重置、不应期和延迟传播。它与本项目每次决策内的 `tanh` 活动率迭代是两种数值模型：状态、时间单位和梯度方法都不同，跨模型复用已有权重与检查点没有对应的状态映射。

值得读的实现：

- `may_cross`：用膜电位的未来上界判断神经元在新输入到达前是否可能发放。
- `advance`：对休眠神经元补算时间衰减，避免每步遍历所有细胞。
- `fb_run`：活动集合、延迟环形缓冲、Poisson 输入和可塑更新的时序。
- `fb_plastic_*`：只为指定突触分配增益和分区，不是默认训练全图。
- JS 包装：WASM 内存上的 TypedArray、64 位神经元 ID 对应、图加载与运行 API。

**作者报告**：指定糖感受刺激测试中，确定性发放时间与 Brian2 一致；随机刺激另比较发放率。此结果针对给定模型与测试，覆盖范围不延伸到其它输入、改动后的可塑模型或具身任务。参见 [核心库原文](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/README.md)。

**对我们的候选价值**：稀疏活动调度、数据压缩和跨运行时一致性测试值得研究。其糖刺激测试的延迟与本项目全图训练吞吐测的是不同负载，数字不可直接换算。切换 LIF 还会改变状态、时间单位、输入编码、梯度方法和检查点契约。

## 2. 图与注释导入

阅读 [flybrain/tools/export_graph.py](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/tools/export_graph.py)、[build_mb.py](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/tools/build_mb.py)、[build_groups.py](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/tools/build_groups.py)、[build_compartments.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/tools/build_compartments.mjs)、[export_positions.py](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/tools/export_positions.py)。

这里把连接表转换成按突触前节点组织的稀疏邻接格式，并建立 KC、MBON、DAN、感觉和下行群组。图结构、群组顺序、可塑边顺序共同决定保存权重的语义。本项目使用 `W[post, pre]`；移植时转置关系需要明确。

移植时需要记录原始 ID、图哈希、输入输出组、递质符号、静默名单和版本。图的神经元对边数与原始突触接触总数是两个不同的量；线上不同任务页面的边数口径有差别，本目录没有把它们统一成一份图的规模。

## 3. 识字：固定表征与局部读出学习

[juku/reader.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/juku/reader.mjs) 的主要路径是：图像像素 → 人工映射的嗅觉投射输入 → KC 发放计数 → 分区输入读出 → 字符类别。

源码明确静默不属于所选蘑菇体相关集合的神经元。类别读出使用 `driveByGroup`，并非简单挑选发放最多的 MBON。每次 `look` 会重置动态状态；保存的突触增益与短时神经状态是两种不同的记忆。

`feedback` 在误判后对正确、错误分区施加不同符号的调制。核心增益更新由突触前活动痕迹与调制量共同决定，并限制上下界。「负多巴胺」是有符号的软件控制量，与生物多巴胺浓度不是同一个概念。

[trainer.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/juku/trainer.mjs) 有独立于训练样本的固定试题、固定评估种子和按类别成绩。反复查看固定试题会影响开发选择，因此这类试题更接近验证集，最终测试需要另外留出。

在线 [juku/hand.mjs](https://hae.satoru.net/juku/hand.mjs) 支持手写与字体混合、按时间块留出，以及增益平均。按时间块分组能减少连续样本泄漏，同一书写者仍可能跨集合。该流程与本项目的全脑 PPO 权重平均是两个不同问题。

相关来源：源码索引中的固定版本 `juku/` 文件，以及在线 `juku/hand.mjs`。在线说明与 Git 的差异见 [证据边界](VERIFICATION.md#2-git-固定版本与在线部署的差异)。

## 4. 人体：三种学习与一个工程身体控制器

**源码入口**：[ningen/README.md](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/README.md)、[web/body-worker.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/body-worker.js)、[web/muscles.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/muscles.mjs)、[web/brain-worker.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/brain-worker.js)、[school/chooser.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/chooser.mjs)、[school/strategies.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/strategies.mjs)、[web/reflex.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/reflex.mjs)、[school/tune.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/tune.mjs)。

| 层 | 输入与输出 | 学习或控制责任 |
|---|---|---|
| 全脑反应 | 食物、触觉、声音与运动相关人工感觉 → 指定神经群活动 | 是否前进、速度、摄食及部分反射；静默失控相关细胞 |
| 蘑菇体选择器 | 食物类别 → 预设行为方案 | KC→MBON 局部学习；单独建立的子网络实例 |
| 运动局部可塑性 | 下行神经群活动与身体表现 → 调制量 | 指定运动通路输入增益；奖励由工程代码计算 |
| 基本身体控制 | 行为方案、姿态、接触 → 关节力矩 / 肌肉激活 | 静态步态、IK、重心转移、支撑力分配、肌肉分配和补充力矩 |
| 参数搜索 | 多轮身体仿真结果 → 控制配置 | 进化策略，不涉及生物突触学习 |

身体来自 MyoSim MyoFullBody。公开代码设定 `rootForce: 0`、`rootTorque: 0`，同时使用支撑与平衡求解和额外关节力矩。根部外力为零与纯肌肉自主控制是两回事：后文这些工程组件仍承担大部分平衡工作。

`body-worker.js` 的 `senses`、`smelling`、`hearing`、`moveSenses` 展示如何从场景状态构造感官量；这些量由工程映射生成，与逼真的视网膜图像是两类输入。

`behave`、`GAIT` 与 IK 负责很大一部分运动。作者也明确说明，纯神经转向曾绕圈或越过食物；前进期间的目标方向由身体侧机制辅助。移植评估时，这种辅助应显式列入基线。

### 局部学习的适用范围

`reflex.mjs` 的调制量结合「结果相对平常的变化」与「该神经群活动相对平常的变化」。源码注释承认：将蘑菇体式规则用于这些下行神经元属于工程借用，没有声称已经找到对应的真实多巴胺回路。

这一点可以启发小规模行为调节实验；它能否替代 PPO、解决双足平衡或避免策略崩解，没有公开证据。

### 候选筛选

[tune.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/tune.mjs) 会重评当前最佳设置，并以累积均值比较手设基线。固定版本中，发布标记要求足够的候选评估次数、基线次数与成绩差。该规则是工程门槛：它控制发布，不构成统计显著性检验。

「展示已验证设置」的分离值得借鉴；其固定次数与 margin 基于 hae 自己的奖励尺度与随机性，套用到本项目没有依据。

## 5. 在线与服务端的运行链

[ningen/web/app.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/app.js) 协调绘制、脑 Worker、身体 Worker 和 school Worker。[school/trial.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/trial.mjs) 提供服务端试验入口；[school/server.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/server.mjs) 汇总试验、记录和等级。阅读时关注 `trial`、`senses`、`frame`、`error` 和回放处理。

在线技术说明指出浏览器脑与身体异步运行，而服务端试验按步同步。同一份代码在这两种时序下会产生不同的闭环轨迹；控制步、感知延迟、动作保持和仿真实时倍率需要分别记录。

等级按试验次数增加。等级、累计训练量与能力提升是三个不同指标；公众提交的数据与独立测试的用途也不同。

## 6. 算盘与人工电路

[kakou/graft.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/kakou/graft.mjs) 提供扩展图与构造回路的基础。[hold](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/kakou/hold.mjs)、[erase](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/kakou/erase.mjs)、[gate](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/kakou/gate.mjs)、[sequence](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/kakou/sequence.mjs) 分别探索状态保持、复位、门控与时序。[soroban/soroban.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/soroban/soroban.mjs) 组合规则电路，[bake-soroban.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/soroban/bake-soroban.mjs) 生成较小的运行图。

作者解释：数字识别使用学到的读出，算盘运算规则则人工布线；算对后的奖励不训练算盘电路；浏览器运行图也经过裁剪。模块构造与裁剪前后的一致性检查值得借鉴；把结果记成原连接组学会算术则与代码不符。[作者原理页](https://hae.satoru.net/soroban/)

该一致性证明只覆盖选定刺激集合；裁剪图对未见输入的行为，是采用类似优化时需要补齐的测试边界。

## 7. LIFE 与可视化真实性

[life/README.md](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/life/README.md) 区分主雌蝇的实际脑仿真、脚本驱动的雄蝇和规则驱动的后代。神经活动气泡也区分仿真发放与按当前行为标示的相关细胞。这个区分对本项目的活动面板很有价值。

显示某神经群、显示实际活动、显示因果贡献是三个层次；点的位置来自真实胞体坐标，连线表示模型中的连接，与真实轴突形态是不同层面的信息。

## 阅读优先级

1. 控制与评估：[ningen/README.md](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/README.md) → [body-worker.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/body-worker.js) → [tune.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/tune.mjs) → [trial.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/trial.mjs)。
2. 局部学习：[brain.c](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/src/brain.c) 可塑部分 → [reader.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/juku/reader.mjs) → [chooser.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/chooser.mjs) → [reflex.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/reflex.mjs)。
3. 检查点扩展：在线 [migrate_reflex.mjs](https://hae.satoru.net/ningen/school/migrate_reflex.mjs)，同时看原图布局与长度检查。
4. 推理性能：[brain.c](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/src/brain.c) 活动调度 → [export_graph.py](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/tools/export_graph.py) → [flybrain.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/flybrain/flybrain.js)。
5. 体验与溯源：[app.js](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/web/app.js) 回放 / trial → [server.mjs](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/ningen/school/server.mjs) → [life/README.md](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/life/README.md)。

各模块可读性好；直接移植仍需重做 FlyWire 与 MaleCNS 的 ID、群组与 VNC 覆盖映射。
