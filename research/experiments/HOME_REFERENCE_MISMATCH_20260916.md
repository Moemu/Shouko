# `home` 观测基准与检查点不匹配（2026-09-16）

## 结论

`app/yumi_description/yumi.yaml` 在 **2026-09-16 11:41:38** 被改动过 `default_angles`，
而 `default_angles` 就是 `Body.home`，它同时是 PD 目标基线和**观测基准**
（`app/sim.py:83` 的 `qpos[7:] - self.home`）。改了它，喂给策略的 12 个关节位置通道
全部平移（髋 +0.07、膝 −0.22、踝 +0.15 rad），**11:41 之前训练或验收过的每个检查点
都失效**。

证据：同一个检查点 `runs/yumi/best.pt.bak`（`a7a4281f…`，那份 9/9 验收的模型），
只改 `home` 一个变量：

| 种子 | 旧 home（`yumi.yaml.bak`） | 新 home（11:41 之后的 `yumi.yaml`） |
|---|---|---|
| 8001 @0.35 | 走满 10.0 s · 0.434 m/s · 骨盆 0.799 · **通过** | 1.94 s 跌倒 · 0.066 m/s |
| 8002 @0.50 | 走满 10.0 s · 0.585 m/s · 骨盆 0.806 · **通过** | 2.02 s 跌倒 · 0.110 m/s |
| 8003 @0.65 | 走满 10.0 s · 0.693 m/s · 骨盆 0.808 · **通过** | 2.00 s 跌倒 · 0.127 m/s |

旧 home 的三个数字与 `runs/local/heldout_yumi.json` 记录的 9/9（30 s，0.425 / 0.527 /
0.683）一致。**那份验收一直是可复现的，只是在旧 home 下**。

**处置：已把 `yumi.yaml` 回退到 `yumi.yaml.bak` 的内容（`diff` 除行尾外为空）。**
回退后重跑 `evaluate_full.py`，9/9 恢复（见下）。

## 症状与发现过程

从「本机完整图预览看起来不对」出发，逐步排除：

1. **先否定了"权重加载错了"。** 运行时上报 `state.checkpoint =
   666134f2b26e4bb5f585633fe22170a1f796e39a39edcd31ca339bd5e831488d`，
   与 `runs/yumi/best.pt` 的 sha256 逐位相同；`body.robot=yumi`、`hips_height=0.97231`
   都对。加载路径没问题。
2. **实测加载中的模型**（21.5 s，SSE 采样 196 帧）：

   | | ppo15c `666134f2`（当时加载） |
   |---|---|
   | 骨盆高 mean / min / max | **0.900 / 0.864 / 0.934 m** |
   | 膝角 mean（取 L/R 较大者） | 0.820 rad |
   | 朝向 yaw 起 → 止 | −107.1° → +80.5°，**扫过 357.5°** |
   | 净位移 dx / dy | **−3.29 m / −0.13 m**（朝后） |
   | 平均速率 | **0.153 m/s**（目标 0.5） |
   | 实时倍率 | 1.00 |

   骨盆 0.900 m 说明姿态确实是直立世系；但它**在 21.5 s 内转了一整圈**，不构成可用步态。
3. **换 `runs/yumi/best.pt.bak`（`a7a4281f`）实测** —— 立刻倒下并躺平
   （骨盆 0.13 m，`upright 0.0003`）。但它在 `heldout_yumi.json` 里是 9/9。
4. **用 `evaluate_full.py` 直接跑同一个文件**（绕开预览）—— 8/9 在约 2 s 内跌倒。
   同文件、同代码路径、同机器人，与记录不符 → 说明**它下面是别的东西变了**。
5. **比对配置文件 mtime**，锁定 11:41:38 的 `yumi.yaml` 改动；再用
   `runs/local/probe_home.py` 在内存里只覆盖 `home`，一次变量得到上表的 A/B。

## 影响范围

| 检查点 | sha256 前缀 | 训练/验收时的 home | 回退后 |
|---|---|---|---|
| `runs/yumi/best.pt.bak` | `a7a4281f…` | 旧（已验证 9/9） | **可用** |
| `runs/yumi/best_tall.pt` = `runs/yumi/best.pt` | `666134f2…` | 新（11:41 那轮） | 失效（需要新 home） |
| `runs/yumi/last.pt` | `7f67b4d2…` | 新 | 失效 |
| `runs/cloud/best.pt` | — | G1 身体，不受影响 | — |

⚠️ **回退 home 之后，`runs/yumi/best.pt` 里那份 `666134f2` 是失效的**
（它按新 home 训练）。而 `local-full.ps1` 默认加载 `RUNS/best.pt`（补注：该 8743 旧入口脚本后已移除，本机完整图预览统一走 `start.ps1`，见 guides/LOCAL.md）。
`666134f2` 与 `best_tall.pt` 逐字节相同，所以内容没有丢失；但
`best.pt` 该指向哪一份是研究决定，尚未处置。

## 回退后的验证

`evaluate_full.py --robot yumi --checkpoint runs/yumi/best.pt.bak --seconds 10 --quick`，
9 个种子全部通过：

```
seed 8001 @0.35  10.0 s  0.437 m/s  pelvis_min 0.801  success
seed 8002 @0.50  10.0 s  0.522 m/s  pelvis_min 0.805  success
seed 8003 @0.65  10.0 s  0.644 m/s  pelvis_min 0.801  success
seed 8004 @0.35  10.0 s  0.449 m/s  pelvis_min 0.786  success
seed 8005 @0.50  10.0 s  0.507 m/s  pelvis_min 0.802  success
seed 8006 @0.65  10.0 s  0.660 m/s  pelvis_min 0.807  success
seed 8007 @0.35  10.0 s  0.484 m/s  pelvis_min 0.801  success
seed 8008 @0.50  10.0 s  0.516 m/s  pelvis_min 0.798  success
seed 8009 @0.65  10.0 s  0.636 m/s  pelvis_min 0.808  success
```

## 需要做的结构性修复（未做）

`home` 是模型接口的一部分，却存在一个全局可变的 yaml 里。检查点已经带
`graph_hash` 与 `feature_version` 校验（`app/brain.py:76`、`app/full_brain.py` 的
`load()`），**但不校验 `home`**，所以不匹配是静默的——策略只是变差，不报错。

建议：把 `default_angles`（以及 `action_scale` / `cmd_scale` 这类同样进观测或动作的
缩放）写进检查点，在 `load()` 里比对，不匹配就抛错。这样旧检查点读旧 home、
新检查点读新 home，两条世系可以共存而不互相污染。

同类问题的前两次见 `PPO_POSTURE_20260916.md` 与
`POSTURE_AND_STABILITY_REVIEW_20260916.md` 的记录方式：都是"界面/文档说的"与
"实际在跑的"不是同一个东西。

## 附带发现

- **`yumi.yaml` 的 `initial_height` 是死键。** 11:41 同时把它从 0.962 改成 0.968，
  但 `app/sim.py:32` 用的是 `ROBOTS['yumi']['initial_height']`（硬编码 0.962），
  `cfg` 里这个键从未被读取。**那次改动实际生效的只有 `default_angles`。**
- 同样没人读的死键：`dof_pos_scale`、`num_actions`、`num_obs`、`cmd_init`。
- 训练面板（`/api/training` → `runs/yumi/training.json`）的数字来自 **09:1x 那一轮**
  （14/32、mean_speed 0.640、min_height 0.796），与当时加载的 11:43 检查点不是同一轮。
  `/api/evaluation` 反而做了检查点哈希校验，所以会老实返回 404。

## 复现

```powershell
# A/B：在内存里只覆盖 home，不碰任何配置文件
$env:PYTHONPATH = 'D:\Project\Neuromechfly'
.venv-gpu\Scripts\python.exe runs/local/probe_home.py runs/yumi/best.pt.bak old
.venv-gpu\Scripts\python.exe runs/local/probe_home.py runs/yumi/best.pt.bak new

# 按记录的方式跑完整评估
.venv-gpu\Scripts\python.exe -m app.evaluate_full --robot yumi `
  --checkpoint runs/yumi/best.pt.bak --output runs/local/probe.json --seconds 10 --quick --device cuda
```

## 相关文件

- `app/yumi_description/yumi.yaml` / `yumi.yaml.bak` — 配置与改动前副本
- `runs/local/yumi_tall.yaml` — 11:41 那轮"直立实验"用的配置（= 当时的 `yumi.yaml`）
- `runs/local/probe_home.py` — 本次 A/B 脚本
- `runs/local/heldout_yumi.json` — 9/9 验收记录（`a7a4281f`）

## 处置补记（2026-09-18，追加）

`best.pt` 指向已定：**换回 9/9 验收的 `a7a4281f` 世系**（`best.pt.bak` → `best.pt`，`ppo_state_best.pt.bak` → `ppo_state_best.pt` 同步配对）。直立世系 `666134f2` 无丢失，继续以 `runs/yumi/best_tall.pt` 保留（与原 `best.pt` 逐字节相同）。依据：社区可复现红线——发布物必须自洽，`a7a4281f` 是当前 `yumi.yaml`（旧 home）下唯一有完整验收记录的检查点；`666134f2` 需要新 home 配置，待该方向重启时再随配置一起归档。

验证（2026-09-18）：`start.ps1 -Body yumi` 启动后 `/api/state` 上报 checkpoint `a7a4281f…`，`/api/evaluation` 返回 200 且 `checkpoint_sha256` 一致（此前为 404）。README「已知问题」同步改写为「检查点世系说明」。

建议的结构性修复（把 `default_angles` 写进检查点并在 `load()` 比对）仍待做，见上文「需要做的结构性修复」。

## 成因分析补记（2026-09-18，追加）

问题：**到底是云端模型没保存，还是被错误覆盖？** 核查结论：**都不是，没有任何权重丢失或损坏；错位是本地配置回退时「只回退一半」造成的配对漂移。**

证据链（文件 mtime + 编年交叉核对）：

1. **两条世系各自完整、各有备份。** `a7a4281f`（ppo9，旧 home，9/9+16/16 验收）有本地 `best.pt.bak`（9/16 09:06）+ `ppo_state_best.pt.bak`；`666134f2`（ppo15c，直立 home）有本地 `best_tall.pt` 逐字节副本（9/16 11:41:21，哈希已复核一致）+ 云端 `best_08s_tall.pt` 同哈希备份（见 PPO_POSTURE 编年）。
2. **9/16 11:41 的配置切换本身是成对的、一致的。** 当时为在本机 8743 预览云端直立模型：拉回 `best_tall.pt`（`666134f2`）、存 `yumi_tall.yaml`、备份 `yumi.yaml.bak`（11:41:38）后把 `yumi.yaml` 切到直立 home——配置与权重同步切换，运行配对正确。
3. **错位发生在 9/16 21:44。** 排查时把 `yumi.yaml` 回退旧 home 作为处置（让 `a7a4281f` 复活，`yumi.yaml` 当前 mtime 21:44:09 即此回退），但 `best.pt` 没有同步指回 `a7a4281f`，仍停在 `666134f2`——从这一刻起配置-权重对分离漂移。
4. **漂移是静默的。** 检查点 `load()` 只校验 `graph_hash` 与 `feature_version`，不校验 `home`（模型接口的一部分），所以不报错、只变差；页面评测靠哈希绑定诚实 404，才暴露问题。

结论：云端保存与拉回流程都正常，`best.pt` 被 `666134f2` 覆盖是 11:41 拉取直立模型的正常操作且有备份；错位的直接原因是 **21:44 回退配置时未同步回退权重指针**，深层原因是 home 观测基准无版本化、加载不校验（即上文「结构性修复」一节）。2026-09-18 的处置补记（`best.pt` 指回 `a7a4281f`）正是补上了当时缺的那半步。
