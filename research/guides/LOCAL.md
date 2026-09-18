# 完整连接组 · 本地推理验证

本机加载云端训练的同一检查点，执行完整 166,700 神经元、25,582,938 条连接的神经计算与 MuJoCo 物理。没有加载教师策略。训练曲线读取云端备份；本地验证写入 `runs/local`。

## 短测结果

2026-09-15，Windows 11，Ryzen 9 7940HX，RTX 4070 Laptop 8 GB，约 16 GB 系统内存。

| 项目 | 实测 |
|---|---:|
| CPU 单次决策中位数 | 187.0 ms |
| GPU 单次决策中位数 | 10.04 ms |
| GPU 单次决策 P95 | 12.00 ms |
| 短测张量显存峰值 | 506.5 MiB |
| CPU / GPU 最大关节输出差异 | 4.77 × 10⁻⁷ |

GPU 测量预热 10 次后采样 100 次，包含输入上传和结果取回。一次决策计算全图 4 次神经更新。CPU 结果为同一 CUDA 版 PyTorch 在 CPU 上运行的 8 次短测。单步控制预算为 20 ms；物理和页面同时运行的速度需看实时界面的实测值。显存数字是 PyTorch 张量峰值，不含驱动、CUDA 上下文、浏览器与其它程序。

短测进程工作集约 1.26 GiB。当时系统可用内存只有约 332 MiB；页面切换和其它应用会影响延迟。推理通过不等于本机适合持续全图训练。

## 行走与实时页面验收

`runs/local/heldout.json` 已完成，检查点哈希与云端验收一致：

| 测试 | 本机结果 |
|---|---|
| 9 个初态 × 30 秒，目标 0.35 / 0.50 / 0.65 m/s | 9/9 通过 |
| 120 秒行走 | 通过，前进 55.40 m |
| 3 次侧向轻推 | 3/3 恢复 |
| 断开脑内连接 | 0/3 通过，约 1.36–1.40 秒跌倒 |

长测与扰动均满足速度、直立、方向和交替触地标准。完整验收张量显存峰值约 597 MiB。

Yumi 页面开启后，读取真实仿真时间与墙钟时间：19.70 秒墙钟推进 19.70 秒仿真，实测 **1.00× 实时**。原始采样见 `runs/local/preview.json`。页面已验证播放、暂停、恢复连接、断连跌倒、速度、朝向、轻推和角色/骨架切换。

本轮验收经历两种干扰，因此不把全部回合墙钟耗时当成稳定性能基准：中途电池模式将 GPU 限到 10 W、210 MHz；接电后恢复约 2445 MHz。另一个早期 8740 子图预览占用了约 26 个逻辑核，已停止。1.00× 测量是在接电、停止旧预览后完成。之后交互检查中也观察到约 0.7–1.0× 的瞬时倍率；其它应用负载较高时仍会降速，不能把短测理解成始终稳定的 50 Hz。

## 打开本地预览

```powershell
cd D:\Project\Neuromechfly
.\start.ps1                # 默认 G1 身体 + CUDA；-Device cpu 为慢速 CPU 推理
```

访问 [本机工作室](http://127.0.0.1:8740)。它不依赖 SSH 或 AutoDL。服务需要已训练的检查点（`runs/cloud/best.pt` 或 `runs/yumi/best.pt`），页面顶栏可切换 G1 / Yumi。云端全图预览经 SSH 隧道使用 8742（`./cloud.ps1 Preview`），与本机服务相互独立。

停止本地服务：`.\stop.ps1`，启动日志在 `runs/cloud/server-error.log`（或 `runs/yumi/server-error.log`）。

## 环境与复现

GPU 环境隔离在 `.venv-gpu`，没有替换早期 `.venv`。使用官方 PyTorch CUDA wheel，版本为 2.12.1+cu130；依赖固定在 `requirements.local-gpu.lock.txt`。

```powershell
uv venv .venv-gpu --python .venv\Scripts\python.exe
uv pip install --python .venv-gpu\Scripts\python.exe -r requirements.local-gpu.lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
.venv-gpu\Scripts\python.exe -m app.check_local
.venv-gpu\Scripts\python.exe -m app.evaluate_full --device cuda --output runs/local/heldout.json
```

当前检查点 SHA-256：`ffcf9a9799ae54332cb5c07ec56d79145ba7e1b557100c78ef082e878713271b`（2026-09-17 云端续训 4800 次更新；2026-09-18 本机独立留出验收 6/9，记录于 `runs/cloud/evaluations/`，横向漂移为高速档未达标项）。旧检查点 `f1d5147c…`（2026-09-15 验收 9/9）已被覆盖、无本地备份，仅验收记录留存。图 SHA-256：`72a6a9ade0117063d3b2515b12ceec3633d9f429ae43f0f8b73b2066a9e51b89`。

## Yumi

已接入用户手动下载的免费小狗 Yumi，版本 1.3.0 / 20240715。文件约 24.6 MiB，37,586 三角形，完整下肢 Humanoid 骨骼。模型文件 SHA-256 和内嵌许可已记录。

VRM 位于 `web/public/yumi.vrm`。服务直接读取文件，不必为更换模型重建前端。文件缺失时保留原角色，并显示“等待 Yumi 文件”。模型使用 VRM0 朝向转换和归一化 Humanoid 骨骼；髋部高度决定物理代理与显示角色的比例。VRM0 的 X/Z 旋转经过方向修正，避免出现手臂举过头顶和腿部动作反向。已检查材质、手臂下垂和实际行走画面。

作者与条款见 [Yumi 来源记录](../avatar/YUMI.md)。模型成本为 0 JPY。模型不进入普通 Git 历史，禁止公开再分发。

## 费用口径

本次安装、推理和验证均在本机执行，没有启动云端训练或变更云实例模式。AutoDL 有卡模式按全部开机时长计费，即使 GPU 空闲也一样。按用户报价，费用为有卡小时 × ¥1.88；无卡时段与适用存储费用另计。

之前的 ¥0.414 仅是两轮训练进程耗时的折算值，不能表示实际账单。累计费用需用平台账单核对。迁移到本地不会自动停止仍开机的云实例。
