# docs/ — 使用者指南

面向「想跑起来」的读者；实验记录与提案见 [`research/`](../research/)，入口在 [README](../README.zh.md)。

- [LOCAL.md](LOCAL.md) — 本机完整图预览：使用、资源、费用口径、复现。
- [CLOUD.md](CLOUD.md) — 云端实验：实测、费用、复现记录。
- [PREVIEW.md](PREVIEW.md) — 浏览器端预览的静态托管：工件布局、服务器规格、本机 Caddy 参考实现。
- [TRAINING.md](TRAINING.md) — 训练过程复现：当前主线与 G1 时代各世系的命令。

`LOCAL.md` 与 `CLOUD.md` 被代码引用（`start.ps1` 错误提示、`app/cloud_server.py`、`cloud.ps1` 打包上云），移动前必须同步改引用。
