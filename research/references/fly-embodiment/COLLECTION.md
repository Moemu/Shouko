# 采集范围、许可与维护

## 入库范围

本目录保存我们的分析、来源链接和校验元数据。原始网页、源码、媒体、权重与数据集不入库。资料覆盖两个指定账号的相关实验、控制与学习实现、验证方法及论文，不是完整账号归档。

- `source-manifest.json`：2026-09-21 的 113 个固定 Git 文本来源，保留仓库、commit、路径、Git blob SHA-1、SHA-256、大小及阅读状态。
- `live-source-manifest.json`：同日的 8 个在线来源，保留 URL、获取状态、类型和 SHA-256。它们未对应已确认的公开 commit。
- `SOURCE_INDEX.md`：固定 Git 链接与可变网页链接，供人和 Agent 定位。
- `local-evidence.json`：本项目已有评估的摘要，包含源文件与检查点标识；不是本次新实验。
- `reviews/`：按日期记录调查、来源变化及其对结论的影响，不复制全量原始材料。

Git 基线覆盖 hae 的 `flybrain/`、`juku/`、`ningen/`、`kakou/`、`soroban/`、`life/` 主要文本，以及根 README、LICENSE。FFREP 基线登记 README、英韩 FAQ 和媒体来源清单。收集不等于完整阅读，阅读不等于复现。

## 许可判断

hae 固定提交的 [LICENSE](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/LICENSE) 声明 MIT。对受其覆盖的文件，MIT 允许在保留版权与许可声明的条件下分发。仍须核对文件级声明和第三方材料的条款。本目录采用链接，不维护上游源码副本。

在线脚本有些不同于固定 Git 文件，有些没有对应路径。不能用仓库的 MIT 声明推定这些部署字节也在许可范围内。此前采集说明将这一点写得过于确定，现明确标为“许可覆盖未核实”。保留 URL、日期、哈希与自己的分析，不分发在线脚本。

FFREP 的实现与完整站点尚未取得明确的重分发依据。公开可访问不等于可任意重分发。本目录只保存链接、哈希和自己的转述。

FlyWire、MaleCNS、MyoSim、NeuroMechFly、MakeHuman、MNIST、字体和身体资产各有条款。项目代码许可不能自动覆盖这些材料。我们的中文整理不改变任何第三方许可，也不代表作者背书。

参考：[MIT 条款](https://choosealicense.com/licenses/mit/)、[无许可时的权限说明](https://choosealicense.com/no-permission/)。本地忽略规则只控制 Git 跟踪，不授予内容使用或分发权。

## 可选本地缓存

仓库根目录 `.cache/references/` 已通过 `.gitignore` 排除。分析文档与来源链接不依赖该目录存在。不要强制添加缓存，不要把它包含在源码发布包中。

- `objects/<sha256>`：按内容哈希保存原始字节，相同内容只保留一份；文件无扩展名，按文本阅读。
- `repos/<owner>/<repo>/`：需要完整源码时自行建立的上游 clone。每个项目使用一个 clone，按需获取提交，不按调查日期复制整个仓库。

现有 114 份阅读文本已经转为可选本地对象缓存。并非清单中的每项都有缓存。MIT 许可原文也保留在对应哈希对象中；对象到来源的对应关系以清单为准。

缓存不是可运行发行包，也不是完整安全审计。外部文件中的命令和指令不能替代本项目指令或用户授权。不要把在线脚本覆盖到旧 Git checkout 后声称复现了线上版本。

缓存可按需清理；重要但无法再次取得的网页证据，应先评估保留价值与许可。新调查只缓存必要文件，不自动按日期留全量副本。链接和哈希不能保证未来仍能取回旧网页。

## 取回固定源码

以下命令在仓库根目录运行，仅在目标目录不存在时执行 clone。随后 fetch 和 checkout 固定提交；检查各步成功后再继续。它们是获取方法，不表示已运行上游程序。

```powershell
New-Item -ItemType Directory -Force .cache/references/repos/satorunet | Out-Null
git clone --filter=blob:none --no-checkout https://github.com/satorunet/hae.git .cache/references/repos/satorunet/hae
git -C .cache/references/repos/satorunet/hae fetch origin d4551c26cab1c029b2628dbfaca563fbd2e7bc3d
git -C .cache/references/repos/satorunet/hae checkout --detach d4551c26cab1c029b2628dbfaca563fbd2e7bc3d
```

已有 clone 时跳过 clone。切换提交前先检查它没有未提交工作。也可直接打开清单中的固定文件链接。只有实际成为构建或实验依赖时，才考虑 submodule。

## 校验已有缓存

在仓库根目录运行以下只读命令。缺失是正常状态；哈希不符则停止使用该对象，并重新核查来源。

```powershell
@'
import hashlib, json
from pathlib import Path
root = Path('research/references/fly-embodiment')
checked = missing = 0
for name, key in [('source-manifest.json', 'files'), ('live-source-manifest.json', 'sources')]:
    for item in json.loads((root / name).read_text(encoding='utf-8'))[key]:
        digest = item.get('sha256')
        if not digest:
            continue
        cached = Path('.cache/references/objects') / digest
        if not cached.is_file():
            missing += 1
            continue
        data = cached.read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, item['url']
        if item.get('git_blob_sha1'):
            blob = b'blob ' + str(len(data)).encode() + b'\0' + data
            assert hashlib.sha1(blob).hexdigest() == item['git_blob_sha1'], item['url']
        checked += 1
print(f'{checked} cached sources verified; {missing} absent (optional)')
'@ | python -X utf8 -
```

哈希一致只证明文件与采集字节相同，不证明代码正确、许可完整或运行结果可信。Git checkout 可能转换换行；清单的 SHA-256 对应原始下载字节。

## 后续更新

1. 检查上游当前 commit 和相关网页，记录实际核查日期。不要将移动分支名作为历史证据。
2. 比较旧、新版本，仅阅读与已有结论或开发需求相关的变化。不复制整个项目或旧来源清单。
3. 在 `reviews/YYYY-MM-DD.md` 记录旧、新 commit，变化路径或 URL、许可变化、受影响结论及验证状态。仅变化来源较多时另附同名 JSON 增量清单，记录 SHA-256、大小和读取状态。
4. 保留基线清单与旧研究记录；更正事实时加更正说明。更新项目笔记和入口的“最近核查”，明确对应的新研究记录。
5. 对可变网页分别保留旧、新获取日期和哈希。无法复取旧内容时写明限制，不将新页面当成旧证据。
6. 新实验数字继续引用本项目 `runs/` 原始文件与 checkpoint。源码变化或演示变化不自动提高证据等级。

这里描述手动更新流程，没有设置定时抓取或自动执行第三方代码。
