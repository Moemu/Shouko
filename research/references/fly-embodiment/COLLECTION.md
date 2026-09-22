# 来源、许可与取回

## 本目录保存什么

本目录保存来源链接、校验元数据与我们的分析。上游源码、网页、媒体、权重与数据集不随本仓库分发；任何人都可以按下文的方法从公开地址取回，并逐文件核对 SHA-256。

- [source-manifest.json](source-manifest.json)：2026-09-21 采集的 113 个固定 Git 文本文件；逐项记录仓库、提交、路径、Git blob SHA-1、SHA-256、字节数、raw 链接与阅读状态。
- [live-source-manifest.json](live-source-manifest.json)：同日采集的 8 个在线来源；记录 URL、获取状态、内容类型与 SHA-256。它们没有对应的公开提交。
- [SOURCE_INDEX.md](SOURCE_INDEX.md)：固定 Git 与在线来源的链接索引，按任务分组，并逐行标注阅读程度。
- [local-evidence.json](local-evidence.json)：本项目已有评估的摘要，含源文件、检查点与图哈希，数据来自 `runs/` 的实测记录。
- [reviews/](reviews/)：按日期记录调查、来源变化及其对结论的影响。

Git 基线覆盖 hae 的 `flybrain/`、`juku/`、`ningen/`、`kakou/`、`soroban/`、`life/` 主要文本，以及根 README 与 LICENSE；FFREP 基线登记 README、英 / 韩 FAQ 与媒体来源清单。登记不等于完整阅读，阅读也不等于复现；各文件的阅读程度见 [源码索引](SOURCE_INDEX.md)。

## 许可

hae 固定提交的 [LICENSE](https://github.com/satorunet/hae/blob/d4551c26cab1c029b2628dbfaca563fbd2e7bc3d/LICENSE) 声明 MIT：在其覆盖范围内的文件，保留版权与许可声明即可分发。文件级声明与第三方材料的条款仍需逐项核对。本目录采用链接与哈希，不维护上游源码副本。

在线部署（hae.satoru.net）的部分脚本与固定 Git 文件内容不同，有的没有对应路径。这些部署字节的许可覆盖未核实——此前记录曾依据仓库许可作过推定，现更正为未核实。本目录保留 URL、日期、哈希与自己的分析，不分发在线脚本。

FFREP 的实现与完整站点没有取得明确的重分发依据；公开可访问不等于可再分发。本目录只保存链接、哈希与转述。

FlyWire、MaleCNS、MyoSim、NeuroMechFly、MakeHuman、MNIST、字体与身体资产各有自己的条款，项目代码许可不自动覆盖这些材料。我们的中文整理不改变任何第三方许可，也不代表作者背书。

参考：[MIT 条款](https://choosealicense.com/licenses/mit/)、[无许可时的权限说明](https://choosealicense.com/no-permission/)。`.gitignore` 只控制 Git 是否跟踪文件，与内容的使用或分发权无关。

## 取回与校验（任何人可复现）

**按固定提交克隆**（在仓库根目录运行；克隆到任意目录均可）：

```powershell
git clone --filter=blob:none --no-checkout https://github.com/satorunet/hae.git .cache/references/repos/hae
git -C .cache/references/repos/hae fetch origin d4551c26cab1c029b2628dbfaca563fbd2e7bc3d
git -C .cache/references/repos/hae checkout --detach d4551c26cab1c029b2628dbfaca563fbd2e7bc3d
```

`.cache/references/` 已被本仓库 `.gitignore` 排除，适合存放这类可选副本，不会进入提交；放在仓库外同样可以。切换提交前先确认克隆目录内没有未提交的工作。

**逐文件下载并核对 SHA-256**（以 hae 的 LICENSE 为例，2026-09-22 复核通过）：

```python
import hashlib, json, urllib.request
from pathlib import Path

manifest = json.loads(Path('research/references/fly-embodiment/source-manifest.json').read_text(encoding='utf-8'))
item = next(f for f in manifest['files'] if f['path'] == 'LICENSE')
data = urllib.request.urlopen(item['raw_url'], timeout=30).read()
assert hashlib.sha256(data).hexdigest() == item['sha256'], item['raw_url']
print('OK', item['repository'], item['path'], len(data), 'bytes')
```

清单中的 `url` 是 GitHub 页面地址，`raw_url` 是原始字节地址；替换过滤条件即可逐项或批量校验。Git checkout 可能转换换行，而清单的 SHA-256 对应原始下载字节，因此批量校验以 `raw_url` 为准；`git_blob_sha1` 可与 Git 对象互查（`git hash-object`）。

**版本一致性的两层含义**

- **固定提交的文件**（[Git 清单](source-manifest.json)）：raw 地址指向不可变的历史内容，SHA-256 核对应始终一致；不一致属于异常，需要排查下载与中间环节。
- **在线部署与其它未固定来源**（[在线清单](live-source-manifest.json)）：它们是活跃上游的流动内容，上游更新后再次取回、字节与采集日不同，是正常情况。处理方式是记录新的获取日期与哈希、评估变化是否影响既有结论（见下文更新流程），并以「采集日期 + 版本」标识被引用的那一份。哈希不一致本身只说明上游有更新，上游仍然可以继续参考。

哈希一致说明文件与采集字节相同；它不说明代码正确、许可完整或运行结果可信。

## 后续更新

1. 检查上游当前提交与相关页面，记录实际核查日期；历史引用使用固定提交，移动的分支名不作为历史证据。
2. 只比较与已有结论或开发需求相关的变化，不复制整个项目或旧清单。
3. 在 [reviews/](reviews/) 的 `YYYY-MM-DD.md` 中记录旧 / 新提交、变化路径或 URL、许可变化、受影响的结论与验证状态；变化来源较多时另附同名增量清单（SHA-256、大小、读取状态）。
4. 基线清单与旧记录保留；更正事实时写明更正说明，同时更新入口页的「最近核查」日期与对应研究记录。
5. 在线来源保留各自的获取日期与哈希；旧内容无法复取时写明限制，不用新页面替代旧证据。
6. 新的实验数字继续引用本项目 `runs/` 的原始文件，并与检查点绑定。
