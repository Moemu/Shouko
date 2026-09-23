# Qoder 提案：浏览器端预览 + 静态托管（2026-09-23）

声明：本提案未参考同日其他提案的结论。性能数据来自 2026-09-23 本机实测（`bench_preview_cpu.py`、`bench_webgpu/`），引用处注明；推断处标注为推断。

## 1. 目标与边界

**目标**：web preview 上线到无 GPU 的香港 VPS（RTT 100-200ms），服务器完全不参与推理与物理仿真，退化为纯静态文件托管。

**实测依据**（2026-09-23，`runs/yumi/best.pt`，166,700 神经元 / 25.58M 突触）：

| 指标 | 数值 | 来源 |
|---|---|---|
| CPU 单步推理（8 线程） | 258ms（需 <20ms，不可行） | `bench_preview_cpu.py` |
| WebGPU 单次推理（4 神经步） | 均值 7.54ms / p99 14.7ms（RTX 4070 Laptop） | `bench_webgpu/` 实测 |
| WebGPU 数值 vs PyTorch | activity 全量 maxAbsDiff ~1e-5，动作 ~1.5e-5 | 同上 |
| MuJoCo 步进（CPU） | 0.81ms 均值 | `bench_preview_cpu.py` |

**不做**：服务器端多会话推理架构、CPU 推理优化（稀疏核替换）、训练控制面上云、studio_server 本地工作台的任何功能删减。

## 2. 架构

```
浏览器（预览页 preview.html）
  ├─ WGSL ConnectomeBrain    4 步神经循环（已验证，bench_webgpu/main.js 即原型）
  ├─ MuJoCo-WASM Body        物理 + PD 控制环（移植 app/sim.py 预览路径）
  ├─ PreviewLoop             20ms 控制环；测速不足自动降速（慢放）
  └─ 现有 UI（VRM 角色、脑活动面板、控制条）改读本地循环

VPS（nginx 静态托管）
  ├─ web/dist/ + research/ 静态文件（REPORT.md 等保持 /api/report 引用路径可静态化）
  ├─ artifacts/preview/     推理权重包 + SHA-256（Release 同一份）
  └─ 无 Python、无训练端点、无动态 API
```

延迟只影响首次下载（估算：图数据压缩后 ~100-150MB + VRM 25MB，浏览器缓存后二次进入秒开；数字为推断，实施时实测）。

## 3. 工件

### 3.1 推理权重包（新增导出脚本 `app/export_preview_weights.py`）

从 checkpoint 导出推理所需最小集合（`bench_webgpu/export_reference.py` 已验证字段齐全）：

- `values.f32`（= `weight*(1+0.9*tanh(edge_delta))` 预合成，98MB）、`pre.i32`（98MB）、`ptr.i32`（0.6MB）
- `inputs/outputs/bias/encoder/readout/normalizer/obs_mean/obs_std`（合计 <4MB）
- `meta.json`：graph_sha256、checkpoint_sha256、observation_size/action_size/neural_steps、physics_interface（供浏览器端 Body 使用，保证 home/尺度与训练一致——防止 2026-09-16 类接口错配）

可选 fp16/int8 量化把 `values+pre` 压到 ~100MB（仅预览用途，页面标注；**不**回写训练评估链路）。若量化，需在导出脚本内置与 f32 版的 maxAbsDiff 自检并记录进 meta。

发布：产物入 Release 附件 + SHA-256 + 训练出处（红线 8），仓库不入库（红线 2）。graph_sha256 绑定沿用 `full_brain.py` 的校验逻辑，浏览器端加载时校验。

### 3.2 MuJoCo-WASM（首要 Spike）

`app/sim.py` 预览路径（`load_motor_policy=False`）只依赖：`mj_resetData/mj_forward/mj_step`、qpos/qvel/ctrl/contact 读写、yaml 配置。**Spike 先行**：确认 mujoco 3.13 官方 WASM 构建或社区 `mujoco-wasm` 能加载 `app/yumi_description/scene.xml` 并跑通 `step_joints` 等价逻辑。失败兜底：VPS 上不开物理的"脑活动展示模式"，或换用 Three.js 侧现成 WASM 绑定。

## 4. 前端模块（web/，无框架扁平约定）

| 文件 | 职责 |
|---|---|
| `web/brain-webgpu.js` | WGSL SpMV + 4 步循环 + 读出头；设备能力检测（storage 上限、128MB 绑定限制）；输出 activity 供脑活动面板 |
| `web/body-wasm.js` | sim.py 移植：`motor_observation`（50 维，含 2026-09-16 防错配的接口字段）、`step_joints`（PD + decimation）、`snapshot`、跌倒判定 |
| `web/preview-loop.js` | 控制环：每 20ms 一拍；若推理+物理实测超预算则整环降速并显示"慢放 ×N"；命令（speed/yaw/push/lesion/reset）全部本地 |
| `web/preview.html` | 独立入口（studio 工作台保持本地不变），静态加载 |
| `web/i18n.js` | 新增文案 zh/en 双字典同步（红线 3），`npm test` 校验 |

脑活动面板：客户端已持全量 activity，采样逻辑对齐 studio_server 的 `self.activity[self.sample]` 显示口径（推断：直接用同一采样索引即可）。

**一致性验收**：浏览器端以 seed=4242 跑 500 控制拍，与 Python 同路径（CPU）逐拍比对 qpos/动作，容差按实测 1e-5 量级放大到 1e-3 定阈值；超阈值即 fail。预览页明确标注"浏览器端演示，精度以服务端评估为准"（红线 7：评估数据仍走服务端评估文件，哈希绑定不变）。

## 5. 服务端与部署

- `studio_server.py` 零改动（本地工作台照旧）；`server.py`/`cloud_server.py` 不动。
- 静态化：`/api/meta`、`/api/evaluation` 的数据改为预生成 JSON（导出脚本顺带产出）；`/api/report` 引用的 `research/REPORT.md` 改为静态路径（`research/` 直接随 dist 托管，路径引用同步改，属 AGENTS.md「移动前必须同步改引用」清单项）。
- VPS 部署：nginx 配置 + rsync 清单放 `guides/PREVIEW.md`（新文档，主机/端口用占位符，真实目标进 git-ignored 的 `runs/cloud/preview_target.json`，红线 1）。
- 预算：静态 VPS 为既有资源，无新增 GPU/保留费。

## 6. 里程碑

| 阶段 | 内容 | 验收 |
|---|---|---|
| M1 Spike | MuJoCo-WASM 加载 yumi scene.xml，跑通 1000 步 step_joints | 与 Python 数值容差内一致；失败则启用兜底再议 |
| M2 大脑 | `brain-webgpu.js` 产品化（能力检测/加载进度/哈希校验/慢放） | 复测 bench_webgpu 指标不劣化 |
| M3 闭环 | preview-loop + body-wasm + 一致性验收 | 4242 种子 500 拍比对通过 |
| M4 UI | preview.html + 现有 UI 接本地循环 + i18n | 浏览器手测金路径：play/reset/speed/yaw/push/lesion/脑活动面板 |
| M5 发布 | 导出脚本 + Release 附件 + guides/PREVIEW.md + nginx + 编年史追加 | 社区可复现自检（红线 8）通过；README/README.zh 同步 |

## 7. 风险与兜底

| 风险 | 概率 | 兜底 |
|---|---|---|
| MuJoCo-WASM 加载自定义 XML 失败 | 低-中 | M1 Spike 前置；兜底为脑活动展示模式或换绑定库 |
| 核显/旧设备推理超 20ms | 中 | preview-loop 自动慢放（CPU 基准 258ms 也可跑 ~0.07 倍速）；页面明示倍速 |
| Safari/无 WebGPU | 中 | WASM 慢放兜底或提示换浏览器 |
| 首次下载过大 | 低 | values fp16 量化（自检后发布）；pre 可尝试 delta/分块压缩 |
| 浏览器端与训练评估数字被社区混用 | 低 | 页面标注 + README 已知问题注明预览精度边界 |

## 8. 下一步

M1/M2 相互独立，可并行 Spike。定稿后进入 `research/experiments/` 编年执行；本提案仅为实施方案，方向与取舍（尤其 3.1 的量化选项、第 5 节静态化范围）待用户拍板。
