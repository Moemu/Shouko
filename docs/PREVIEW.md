# 浏览器端预览 · 静态托管

连接组推理（WebGPU）与物理（MuJoCo-WASM）全部在访客浏览器内运行。**服务器不参与计算**，只负责发送字节，因此可以是一台无 GPU 的普通主机。

页面：`web/preview.html` → 构建为 `dist/preview.html`。
权重包：`app/export_preview_weights.py` 生成，默认 `artifacts/preview/`（已 ignore，不入库）。

本文分三层，请按用途取用：

| 层 | 性质 | 位置 |
|---|---|---|
| **工件**：构建产物与权重包的布局 | 规范 | §1–§3 |
| **规格**：服务器必须满足什么 | 规范 | §4 |
| **参考实现**：本机实际怎么接的 | **仅举例**，站点特定 | 附录 |

社区复现只需要前两层：`app/serve_preview.py` 是 §4 的可执行版本，无需 Caddy、Docker 或任何 web 服务器就能把预览跑起来。**未实测的项已逐条标注**；平台侧限制来自官方文档的核对，不是本项目的实测。

## 1. 两个生成步骤

```bash
npm run build                                              # → web/dist/
.venv-gpu/Scripts/python.exe -m app.export_preview_weights  # → artifacts/preview/
```

导出脚本会在打包前用**服务端自己的** `matches_checkpoint()` 校验留出证据，证据与检查点不符即拒绝导出。本地先自检：

```bash
npm test                                        # 包契约、证据绑定、页面资源可解析
.venv/Scripts/python.exe -m app.test_serve_preview   # §4 规格的逐条回归
```

## 2. 首屏字节（实测）

| 项 | 大小 | 来源 |
|---|---:|---|
| `values.f32.bin` | 97.6 MiB | 导出实测 25,582,938 × 4 |
| `pre.i32.bin` | 97.6 MiB | 同上 |
| `mujoco.wasm` + `mujoco.js` | 9.7 + 0.3 MiB | `@mujoco/mujoco@3.13.0` |
| 其余张量（`encoder` / `ptr` / `bias` / …） | 4.7 MiB | 导出实测 |
| `yumi.vrm` | 24.6 MiB | `web/public/` |
| 三个 chunk + CSS + 报告 | 0.9 MiB | 构建输出 |
| **合计** | **235 MiB** | 首次访问；权重包本身 22 个文件共 210.0 MiB |

包内 18 个文件**内容寻址**（另外 4 个是固定名：meta.json / body_config.json / scene.xml / yumi.xml，它们换检查点后内容会变而名字不变，必须重新验证——见附录）（如 `values.350ad897.bin`），因此可以长缓存；`meta.json` 是固定名的清单，不缓存。回访者第二次进入只取 `meta.json` 与 HTML，其余全部命中缓存。

**这是首次访问的量。** 传输压缩后的实际大小未实测（见 §5）。

## 3. 目录布局

站点根目录的形态：

```
<root>/index.html        ← 由 dist/preview.html 发布而来（见下）
<root>/assets/  favicon.png  yumi.vrm
<root>/artifacts/preview/<18 个内容寻址文件 + 4 个固定名文件>
```

两点必须说明：

- **预览页要发布成站点的 `index.html`。** `dist/` 里同时有 `index.html`（本地工作台）与 `preview.html`，而工作台依赖 `/api/*`，静态托管下必然报错。所以发布映射是：`dist/preview.html` → `<root>/index.html`，`dist/index.html` **不发布**。这样 `/` 天然落到预览页，不需要任何 URL 重写；页面内部全用绝对路径（`/assets/...`），改名不影响它。
- **不要发布 `dist/avatar.vrm`**（10.3 MiB）：预览页只用 `yumi.vrm`，那是 G1 角色，只给工作台用。
- `/artifacts/preview/` 这个路径在 dev 中间件（`vite.config.js` 的 `serveExternal`）、参考静态服务器与生产边缘**三处保持一致**。因此 `<meta name="preview-package">` 只在把权重放到 CDN 时才需要改。

## 4. 服务器规格

这是规范性清单。任何静态服务器（Caddy / nginx / 对象存储 + CDN）满足它即可，**参考实现见附录，不作为要求**。

**必需的（缺则页面功能受损或校验失效）**

1. **HTTPS。** `crypto.subtle` 只在 secure context 存在。纯 HTTP 下浏览器根本不提供它，权重 SHA-256 校验会被**静默跳过**——页面只会弹一条 `preview.error.insecure` 提示，不会失败。纯 HTTP 部署等于没有完整性校验。
2. **按 §3 的路径提供文件。** 权重包必须在 `/artifacts/preview/`（或改 `<meta name="preview-package">` 指向别处）。
3. **响应体原样发送。** 不得改写、重新编码或截断内容——页面拿到的字节要能对上 `meta.json` 里的 sha256。
4. **`.wasm` 用 `application/wasm`。** 否则 `WebAssembly.instantiateStreaming` 失败并回退成整文件重下。
5. **不做目录列表。**

**强烈建议（不满足也能跑，但结果不对或体验差）**

6. **`.md` 用 `text/plain; charset=utf-8`。** 报告是中文；缺 charset 时浏览器可能按本地默认编码（GBK）解出乱码。
7. **支持 Range 请求。** 包里有两个 97.6 MiB 的文件，没有续传时一次断线就要从头再来。
8. **缓存分流：** 内容寻址的文件给长缓存（`public, max-age=31536000, immutable`）；`meta.json` 与 HTML 不缓存，这样换包与重新部署立即生效。`yumi.vrm` 没有内容寻址，只能给较短的 TTL。
9. **传输压缩**（见 §5）。这一步是纯优化，且无损。

`app/serve_preview.py` 就是这份清单的可执行版本，同时用于本地免服务器预览：

```bash
.venv/Scripts/python.exe -m app.serve_preview        # http://127.0.0.1:8741/
```

它只用标准库，不需要项目的 Python 环境。它的回归测试 `app/test_serve_preview.py` 逐条覆盖上面的必需项与建议项，包括「服务出的字节与清单一致」与 Range 切片。

## 5. 传输压缩（**实测可行，但当前未启用**）

**结论（2026-09-25 决定）：不使用预压缩。** 是否有效完全取决于路径：

- **橙云路径拿不到收益。** CF 对这个对象**存并回的是未压缩体**，与客户端的 `Accept-Encoding` 无关——`gzip` / `zstd` / `identity` 三种客户端都收到 `Content-Length: 102,331,752` 且无 `Content-Encoding`；**清掉整个站点缓存后依旧如此**，所以不是缓存被污染。
- **能拿到收益的只有灰云直连**，但那会暴露源站 IP、并让该路径失去 CF 的防护 —— 已明确不接受。

因此 sidecar 已从服务器与仓库包两侧删除，包内保持 22 个文件。下面的数据保留，是为了让后来者不必重做这次调查。

### 实测比

**注意"前缀实测"偏乐观**：`pre` 的整文件比例（0.382 / 0.594）明显差于用 20 MiB 前缀估出的（0.360 / 0.545），所以 `values` 只给前缀值并标明未做整文件。

| 文件 | 范围 | 原始 | gzip-9 | zstd-3 | **zstd-19** |
|---|---|---:|---:|---:|---:|
| `pre.i32.bin` | **整文件** | 97.6 MiB | 0.594（57.9 MiB） | — | **0.382（37.3 MiB）** |
| `values.f32.bin` | 前 20 MiB | 97.6 MiB | 0.820 | 0.898 | 0.802 |

耗时（一次性）：`pre` 整文件 zstd-19 约 40 s、gzip-9 约 9 s；两种 sidecar 合计 95.2 MiB 磁盘。

- **等级影响极大。** `pre` 是行内升序的 CSR 列索引，快速等级抓不到它的结构：前缀上 zstd-3 反而比 gzip 差（0.568 vs 0.545），zstd-19 才有量级提升。
- `values` 是训练权重，尾数近似随机，压不动（前缀 0.802）。真正能大幅缩小它的是 fp16 量化，但那会改变数值，见 §9。

### 生效验证（源站侧，2026-09-25）

部署 sidecar 后**绕过 CF 直连**，三种协商都正确，且**解码后字节与清单逐一相符**：

| 客户端 `Accept-Encoding` | `Content-Encoding` | `Content-Length` |
|---|---|---|
| `zstd` | `zstd` | 39,134,093 |
| `gzip` | `gzip` | 60,751,573 |
| `identity` | 无（原始兜底） | 102,331,752 |

`Vary: Accept-Encoding` 三种情况都在 ✓；`Range` 与压缩变体共存正常（三种编码下 `-r 0-1023` 都返回 `206` / 1024 B）—— 但**"下载中断后续传"在压缩变体上的字节正确性未验证**，需要真做一次中断恢复才能下结论。

**经 CF 时**：未命中缓存会把源站的压缩变体转发过来（实测 `Content-Encoding: gzip` + 60,751,573 ✓），但一旦进入 CF 缓存就是未压缩体。正确性无虞 —— 它给所有客户端的都是可直接使用的原始字节，identity 客户端解码后 sha256 与清单仍一致 ✓。

**关键：`meta.json` 里的 sha256 是对未压缩工件算的。** `fetch().arrayBuffer()` 拿到的是解码后的字节（HTTP `Content-Encoding` 由浏览器处理），所以校验仍然通过。反过来，如果自己把 `.zst` 当普通二进制传、再由 JS 手动解压，校验的就是压缩字节了——本页不走这条路。

**`Vary: Accept-Encoding` 不能省。** 若响应缺它，中间层（尤其 CDN）可能把压缩变体缓存下来再发给不支持该编码的客户端。§4 的参考静态服务器不压缩，所以不涉及；接边缘时必须确认。

### 若将来要启用

```bash
cd artifacts/preview
# 包内文件名是内容寻址的，所以按前缀匹配
for f in pre.*.bin; do zstd -19 -q -k -f "$f"; done    # 约 40 s
for f in pre.*.bin; do gzip -9 -k -f "$f"; done        # gzip 回退，约 9 s
```

**这两个命令必须在每次导出之后重跑**：`app/export_preview_weights.py` 的 `clear_previous()` 会把 `*.bin.zst` / `*.bin.gz` 一并清掉（派生物，换检查点后必须重建），顺序是「导出 → 预压缩 → 推送」。想自动化可把 `--precompress` 做进导出脚本（每次导出多约 50 秒）。

两点取舍：浏览器对 zstd 的支持不齐，只放 zstd 时老浏览器拿到的是**未压缩原文件**（不是失败）；`pre` 值得两种都放，`values` 不值得（gzip 80.0 与 zstd 78.3 几乎一样）。

### 一条未验证的替代途径

CF 会压缩 `text/plain`（`REPORT-*.md` 经 CF 实测带 `Content-Encoding: gzip` ✓），只是跳过 `application/octet-stream`。**若把包内 `.bin` 的 Content-Type 谎报为 `text/plain`，CF 或许会动态压缩** —— 既保住橙云又不暴露源站。**未实测**，且取决于 CF 对超大对象的压缩上限。页面用 `fetch().arrayBuffer()` 取这些文件，MIME 对功能无影响；但谎报类型本身需要谨慎评估，不宜默认采用。


## 6. Cloudflare

已核实（来自 Cloudflare 官方文档，非本项目实测）：`.bin` **已在**默认缓存扩展名列表内；可缓存对象上限 Free/Pro/Business 为 **512 MB**，本项目最大文件 97.6 MiB 远低于此；100 MB 是**上传**请求体限制，与下载无关。

- **HTTPS 与源站证书**：Cloudflare 终止访客 TLS，源站仍需持有该域名的有效证书。CF 侧的 SSL/TLS 模式需为 Full 或 Full (strict)；否则会看到 **525**（= CF 到源站的 TLS 握手失败，通常是源站还没有这个域名的证书）。
- **橙云开着时，ACME 的挑战类型有讲究**：
  - **`tls-alpn-01` 不可能成功** —— CF 自己终止 TLS，ALPN 协商到不了源站。Caddy 会先试它，失败后才换下一个。
  - **`http-01` 可以**（实测：同一台同构主机上 `note` / `nightcord` 的证书就是在本机全记录开着橙云的状态下签发的，签发日期 2026-09-07/08）。CF 会把 `/.well-known/acme-challenge/` 转发到源站。
  - 因此签发慢或一直失败时，先确认 DNS 真的解析了（`NXDOMAIN` 会让两种挑战都失败），再考虑给该站点加 `tls { issuer acme { disable_tlsalpn_challenge } }` 强制走 HTTP-01。
  - 想彻底不依赖 ACME，可用 Cloudflare 侧的 **Origin Certificate**（15 年、CF 信任），在 site block 里用 `tls <cert> <key>` 指定。
- **缓存**：源站已发 `public, max-age=31536000, immutable`，CF 默认会沿用。若要更可控，为 `/artifacts/preview/*` 与 `/assets/*` 建 Cache Rule 显式设长 Edge TTL（规则名称与选项随套餐/版本变化，**在后台按当前版本确认**）。
- **实测（2026-09-25，橙云开启）**：`/` 与 `meta.json` 返回 `cf-cache-status: DYNAMIC`（按预期**不缓存**）；`values.*.bin` 首次 `MISS`、第二次 **`HIT`** —— CF 确实把 97.6 MiB 的文件缓存在了边缘。同时 `.wasm` 的 `Content-Type`、`.md` 的 `charset`、以及**全部字节**都原样穿过 CF（sha256 与清单逐一相符，含完整的 97.6 MiB 文件）。
- **验证是否真的缓存了**：

```bash
curl -sI https://<域名>/artifacts/preview/values.350ad897.bin \
  | grep -iE 'cf-cache-status|cache-control|content-encoding|vary|content-length'
```

`cf-cache-status: HIT` 说明边缘命中；`MISS` 第二次应转为 `HIT`。

- **大陆可达性**：普通 Cloudflare 没有大陆 POP（大陆节点属 Cloudflare China Network，需企业版加订阅）。**Anycast 落到哪个 POP 取决于运营商、地区与当时的网络调度，不能预设为某一地**，所以 `CN → CF → 源站` 是否比 `CN → 源站` 直连更快，必须实测。建议留一个不开橙云的子域直连源站，按 §7 对比。

  §7 已测了其中一条线路（上海 · 中国移动），结果是**冷热路径相反**：CF 延迟更差、首次更慢、但**回访更快**。这与本节"CF 的确定性收益是缓存与隐藏源站、不是必然的加速"是一致的。

## 7. 橙云 vs 直连的 A/B 实测

对**橙云**与**直连源站**各测一遍。97.6 MiB 量级下 TTFB 的差异远不如持续吞吐重要。

### 已测：上海 · 中国移动（2026-09-25，单条线路）

对象是 `values.350ad897.bin`（97.6 MiB，整包里最大的两个文件之一）。直连臂用 `curl --resolve` 指向源站，不经 CF，**也不计 DNS 解析**（真用灰云记录时会多一次 DNS，约数十毫秒，相对十几秒可忽略）。

| 测量 | 橙云 | 直连源站 |
|---|---|---|
| 1 MB × 3 吞吐 | 521 / 650 / 604 KB/s | 893 / 913 / 943 KB/s |
| 10 MB × 3 吞吐 | 2.99 / 3.35 / 2.99 MB/s | 4.13 / 3.99 / 4.02 MB/s |
| 整文件（各 1 次） | **12.98 s**（7.88 MB/s） | 18.88 s（5.42 MB/s） |
| TTFB | 0.64–0.70 s | **0.36–0.38 s** |
| 该臂首次 `cf-cache-status` | MISS | — |

解读——**冷热路径表现相反，这是本节最要紧的一点**：

1. **延迟上直连始终更好**：TTFB 0.36 s 对 0.67 s，前 1 MB 到达约 1.1 s 对 1.6–2.0 s。也就是说 CF 给大陆访客多加了约 **0.3 s** 的往返。
2. **首次访问（CF miss）直连更好**：橙云 miss 时 10 MB 只有 2.3–3.3 MB/s，因为它要边回源边转发。按此**推算**整包约 30–40 s —— 这是**由 10 MB 样本外推的推断，未实测**。
3. **回访（CF 命中）橙云更好**：整文件 12.98 s 对 18.88 s，边缘到客户端的路径确实优于客户端直连香港。注意这一次是在同臂前几次 10 MB 探测把对象预热之后跑的，所以它代表的是**命中**情形，不是首次。

所以结论不是"CF 好或不好"，而是**按路径分流**：包是不可变长缓存、回访会占多数，但**每个新访客都要先付一笔冷启动**（§2 的 235 MiB）。值得认真考虑的是——页面与小资源走橙云（拿缓存与隐藏源站），195 MiB 的权重包走灰云直连。

下面是单次请求的指标形式；两臂的完整跑法见本节末尾。

```bash
# 单次：DNS、TCP、TLS、TTFB、总耗时、平均吞吐
curl -o /dev/null -s -w 'dns=%{time_namelookup} tcp=%{time_connect} tls=%{time_appconnect}
ttfb=%{time_starttransfer} total=%{time_total} size=%{size_download} speed=%{speed_download}\n' \
  https://<被测域>/artifacts/preview/values.350ad897.bin
```

### 仍需实测

- **只有一条线路**（上海 · 中国移动 AS9808）。**联通、电信未测**，而分线路正是本节的目的。
- **整文件每臂只测了 1 次**，不足以给出 P50 / P90 / P99；重复 10 次才有意义（代价约每臂 1 GiB 流量）。前 1 MB / 前 10 MB 用 `-r 0-1048575` / `-r 0-10485759` 单测即可。
- **中断率未测**，单一时段、未覆盖大陆不同地区。

补测推荐做法：加一条灰云子域（`shouko-direct.<域名>` → A → 源站 IP），然后在不同网络与时段各跑一遍两臂：

```bash
U=https://<域名>/artifacts/preview/values.350ad897.bin
DIRECT='--resolve <域名>:443:<源站 IP>'   # 等效于灰云记录，不计 DNS

# 橙云
curl -s -o /dev/null -w 'ttfb=%{time_starttransfer} total=%{time_total} speed=%{speed_download}\n' "$U"
# 直连（不经 CF）
curl -s -o /dev/null -w 'ttfb=%{time_starttransfer} total=%{time_total} speed=%{speed_download}\n' $DIRECT "$U"
# 首次访问的橙云表现：加查询串强制 miss
curl -s -o /dev/null -r 0-10485759 -w 'ttfb=%{time_starttransfer} total=%{time_total} speed=%{speed_download}\n' "$U?cb=$RANDOM"
```

结论写回本节，标注日期、城市与运营商。

## 8. 上线与核对

### 推送实现不在本文范围内（有意为之）

「把工件送上主机」这一步**不放入仓库**。理由是它站点特定（目标主机、路径、凭据）且**不构成社区复现路径**：社区没有这台主机，拿到脚本也用不上。把它排除并不损害可复现性——§1–§4 的工件与规格才是复现所需的全部。

但推送必须满足以下结果约束，否则上线会出问题：

- **构建与导出在本地完成。** 导出需要 `runs/yumi/best.pt`，主机上不必也不应装 torch / MuJoCo。
- **先传内容寻址文件，最后才替换 `meta.json`。** 在此之前旧修订照常服务，所以切换是原子的，不存在半新半旧的状态。
- **旧修订先不要删。** 正在下载旧修订的访客不受影响，而且天然留下回滚点；之后再按保留条数做 GC（每份约 210 MiB）。
- **不发布** `dist/index.html` 与 `avatar.vrm`（§3）。

### 核对清单

无论用什么方式推送，上线后逐条核对：

- [ ] `/` 返回 200 且是预览页：`curl -sI https://<域名>/ | head -1` 与 `| grep -i content-type`（应为 `text/html`）
- [ ] 工作台没有被发布：`curl -s -o /dev/null -w '%{http_code}\n' https://<域名>/index.html` 应为 `404` 或落到预览页（取决于采用了哪种映射）
- [ ] 浏览器打开页面，Network 面板里 `assets/` 下的三个 chunk/CSS 与 `REPORT-*.md` 全部 200（构建后 `npm test` 已本地断言这些路径可解析）
- [ ] 大文件可续传：`curl -s -o /dev/null -r 0-1023 -w '%{http_code}\n' https://<域名>/artifacts/preview/values.350ad897.bin` 应为 `206`
- [ ] 用 HTTPS 打开页面，控制台无 sha256 mismatch，也没有 `preview.error.insecure` 提示
- [ ] 页脚检查点哈希与 `meta.json` 的 `checkpoint_sha256` 前 8 位一致
- [ ] 评估页展开后有数据（留出证据与检查点绑定，不符时页面会拒绝显示）
- [ ] 服务出的字节与清单一致：下面这段就是页面自己做的事

```bash
python3 - <<'PY'
import hashlib, json, urllib.request
BASE = 'https://<域名>/artifacts/preview/'
# 必须带 browser-like UA：Cloudflare 的机器人防护会把 Python 默认的 UA 挡成 403
# （实测：同一 URL 用 curl 200，用 urllib 默认 UA 403）。
def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return urllib.request.urlopen(request, timeout=300).read()
meta = json.loads(get(BASE + 'meta.json'))
bad = [key for key, entry in meta['files'].items()
       if hashlib.sha256(get(BASE + entry['path'])).hexdigest() != entry['sha256']]
print('checkpoint', meta['checkpoint_sha256'][:8], '| mismatched:', bad or 'none')
PY
```

注意这会下载**整个包**（约 210 MiB）。只想快速抽查时，把 `meta['files'].items()` 换成 `[('ptr', meta['files']['ptr'])]` 之类的子集即可。

这段经过 HTTP 取回并校验，能同时排除压缩、缓存、MIME 与中间层改写内容的问题。本地那一份由 `app/test_serve_preview.py::test_served_bytes_match_the_manifest` 覆盖。

## 9. 已知边界

- **首次访问约 235 MiB**，跨境链路上这是体验的主要成本；回访由长缓存归零。
- **权重包按未压缩传输。** CF 对 `.bin` 不压缩（§5），而唯一能拿到压缩的灰云直连又会暴露源站，因此访客要下完整的约 195 MiB。这是**有意接受的取舍**，不是遗漏。
- **主机的磁盘余量是硬约束。** 实测部署机根分区 30G 用了 **88%，仅余约 4 GiB**。站点约 26 MiB + 权重包 210 MiB 之后余量本就不多，因此：**不要保留多份修订**（每份 210 MiB），预压缩那 95.2 MiB 已按 §5 的决定回收。
- **算力在访客机器上**：「服务端无 GPU」不等于「不需要 GPU」。WebGPU 推理在 RTX 4070 Laptop 上实测均值 7.54 ms / p99 14.7 ms（来源 `bench_webgpu/`，2026-09-23 提案引用）；核显设备会落到页面自带的「慢放 ×N」。手机上既有下载量也有显存问题，建议直接提示。
- **Safari 的 WebGPU 支持情况未核实**，上线前应查当前版本。
- **fp16 量化已推迟**：不是精度不够（实测往返 maxAbs 4.77e-4），而是同一个 `checkpoint_sha256` 不能同时标签两份不同字节的包。改走未来的模型/版本切换器，每个变体自带身份与等价性证据。
- **评估是快照**：静态页展示的是导出时的留出证据。换检查点必须重新导出——导出器会因绑定不符而拒绝，页面也会拒绝渲染，失败方向是「不显示」而不是「显示过期数字」。
- **页面数字不是评估数据**：精度以服务端独立评估为准，页面自己在 `preview.disclaimer` 里也这么写。
- **VRM 署名不可去除**：角色许可要求页面显示署名（`THIRD_PARTY.md`），静态页同样携带。

## 10. 已经核对到什么程度

- **已实测**：包内容与基准夹具逐字节一致；展开保真（与提交版 `index.html` 逐字节相同）；静态页不含任何服务器专属标记；包内每个文件的 sha256；证据与检查点绑定；页面资源全部可解析；评估页折叠联动。这些都在 `npm test` 里。
- **已实测（§4 规格）**：`app/test_serve_preview.py` 九项回归覆盖 Root 映射、MIME（含 `.wasm` 与 `.md` 的 charset）、缓存分流、Range 切片（206 与 416）、目录列表拒绝、路径穿越拒绝，以及**经 HTTP 取回后 sha256 与清单相符**。变异检查确认这些断言能失败（去掉 Range 支持即报 `200 != 206`）。
- **已核对**：按 §3 布局用参考静态服务器能完整拉起页面、通过校验、展开评估页看到 63 行。
- **已在真实主机上核对（2026-09-25，Caddy 2.10.2 / Docker）**：
  - `caddy validate` 通过，且配置已加载进**运行中**的 edge —— 用 admin API 查运行配置确认，而不是看 reload 的返回码（见附录：返回 0 也可能加载的是旧内容）。
  - §4 的必需项与建议项逐条实测通过：`/` 与 `meta.json` 为 `no-cache`；`values.*.bin` 为 `immutable` + `application/octet-stream`；`mujoco-wasm.*.wasm` 为 `application/wasm`；`REPORT-*.md` 为 `text/plain; charset=utf-8`；Range 返回 `206 Partial Content`；HTML 协商压缩返回 `Content-Encoding: gzip` 且带 `Vary: Accept-Encoding`。
  - 该站点在边缘**未发布任何宿主端口**，只在 `edge` 网络内可达（同时排除了与既有 80/443 的占用冲突）。
  - **同机另外三个既有站点的 TLS 未受影响**：改动前后均返回 `200/302` 且证书校验通过。
  - 证书签发走**生产 CA**，挑战类型为 **`http-01`**（橙云开启时 TLS-ALPN 到不了源站，但 Caddy 会自行选用 http-01，不需要额外配置）。此前长时间没有任何尝试，原因是 Caddy **内部的速率限制器排队**（日志 `done waiting on internal rate limiter`），与 CF 和网络无关。
  - DNS 配好、证书签发后，**经 Cloudflare 把规格又完整走了一遍**：`/` 200 + `no-cache` + `DYNAMIC`；`meta.json` 200 + `no-cache`；`mujoco-wasm.*.wasm` 为 `application/wasm`；`REPORT-*.md` 为 `text/plain; charset=utf-8`；Range 返回 `206` 且正好 1024 字节；`values.*.bin` 第二次请求 `cf-cache-status: HIT`；**四个文件的 sha256（含完整的 97.6 MiB）与清单逐一相符**——即 CF 不改写字节、不改 MIME、不改 charset。
  - **传输压缩（§5）**：源站侧三种 `Accept-Encoding` 协商全部正确、解码后字节与清单相符；但**经 CF 时收益为零** —— 清缓存前后都返回 `HIT` + 97.6 MiB 且不带 `Content-Encoding`，三种客户端一视同仁。因不接受暴露源站，**已决定不启用预压缩**，sidecar 已从两侧删除。
- **已在浏览器中验证通过（2026-09-25，人工验收）**：公网域名下的真实加载正常，包含 VRM 渲染、WebGPU 推理，以及「评估结果 / 研究方法」两个视图的展开。这一层本文件无法自动核对（需要真实的 WebGPU 与浏览器），故记为人工验收而非实测脚本。
- **已实测一条线路**：§7 记录了上海 · 中国移动的单次 A/B（冷热路径表现相反，含一处标注为推断的外推）。**联通、电信、重复取样与中断率仍未测**，因此"分流"这个方向目前只由一条线路支持。

## 附录：参考实现（本机为 Caddy，装在 Docker 里）

**这只是本机的一种接法，不是要求。** §4 的规格才是规范；用 nginx、独立 Caddy、对象存储 + CDN 都同样成立。

结构：Caddy 跑在 Docker Compose 里，占住宿主 80/443，通过一个 external network 与其它服务共用边缘。静态站点由 Caddy 自己的 `file_server` 提供，因此**不需要新容器**，只需要把一个宿主目录 bind mount 进容器，并加一个 site block。

```yaml
# compose.yaml 中 caddy 服务的挂载。注意挂的是**目录**而不是单个文件——
# 原因见下方"三个坑"第 2 条，本机就是踩过之后才改成这样的。
    volumes:
      - ./caddy:/etc/caddy:ro              # 目录里放 Caddyfile
      - ./preview:/srv/preview:ro          # 站点根，推送目标就是它
      - caddy_data:/data
      - caddy_config:/config
```

```caddyfile
# Caddyfile 中新增一个 site block（与既有的 reverse_proxy 站点并列）
<预览子域> {
    root * /srv/preview

    # 小文本实时压。大文件不做传输压缩（§5 的决定），所以不启用 precompressed。
    encode zstd gzip
    file_server

    # 内容寻址的文件改名即失效，可以长缓存；固定名的四个（meta.json /
    # body_config.json / scene.xml / yumi.xml）内容会变而名字不变，必须重新验证，
    # 否则换检查点后会出现「新清单 + 新张量 + 旧物理接口」。两条规则不重叠，不依赖顺序。
    @fixed path /artifacts/preview/meta.json /artifacts/preview/body_config.json /artifacts/preview/scene.xml /artifacts/preview/yumi.xml
    header @fixed Cache-Control "no-cache"
    @immutable {
        not path /artifacts/preview/meta.json /artifacts/preview/body_config.json /artifacts/preview/scene.xml /artifacts/preview/yumi.xml
        path /assets/* /artifacts/preview/*
    }
    header @immutable Cache-Control "public, max-age=31536000, immutable"

    # HTML 与站点入口不缓存，重新部署立即生效。
    # "/" 单独列一条：它由 index.html 内容响应，但路径本身不匹配 *.html。
    header / Cache-Control "no-cache"
    header /*.html Cache-Control "no-cache"

    # 角色文件没有内容寻址，只能给较短 TTL
    header /yumi.vrm Cache-Control "public, max-age=604800"

    # 中文报告：Caddy 的 mime 表未必认识 .md，显式指定
    @markdown path *.md
    header @markdown Content-Type "text/plain; charset=utf-8"
}
```

部署方式（一次性）：

```bash
# 1) 改完 compose.yaml 后重建容器（证书在 caddy_data 卷里，不受影响）
docker compose up -d caddy
# 2) 校验并热加载 Caddyfile（等价于 nginx -t + reload）
docker compose exec caddy caddy validate --address 127.0.0.1:2019 --config /etc/caddy/Caddyfile
docker compose exec caddy caddy reload   --address 127.0.0.1:2019 --config /etc/caddy/Caddyfile
# 3) 确认真的生效了——不要只看上一条的返回码
docker compose exec caddy wget -qO- http://127.0.0.1:2019/config/ | grep -o '<你的站点名>' | sort -u
```

之后换检查点、换权重包、改页面都只往 `/srv/preview` 对应的宿主目录写文件，**不需要重启或重载任何容器**：`file_server` 每次从磁盘读。

### 改这个文件时的三个坑（都真实踩到过）

1. **`--address 127.0.0.1:2019` 不能省。** `caddy reload` 默认连 `localhost:2019`，而容器内 `localhost` 会先解析到 `::1`，Caddy 的管理端点只监听 `127.0.0.1` → 连接被拒。此时 reload 仍会打印"adapted config to JSON"然后失败，**极易误判为成功**。
2. **不要挂单个文件，要挂目录。** 挂 `./Caddyfile:/etc/caddy/Caddyfile` 时 Docker 钉住的是**当时那个 inode**；若文件之后被"写临时文件再改名"的方式替换（某些编辑器和 `cp`/`mv` 的写法），容器里看到的仍是旧 inode 的旧内容，**表现为宿主文件明明改了、`reload` 也返回 0，但配置没生效**。本机实测遇到过宿主 inode `569801` 与容器内 `567954` 不一致、reload 返回 0 却加载旧内容的状况。
   - **本机已改为挂目录**（`./caddy:/etc/caddy:ro`，Caddyfile 放在该目录里），换文件不再受影响。验证方式是两边 inode 一致：
     ```bash
     stat -c %i /root/services/caddy/Caddyfile
     docker compose -f /root/services/compose.yaml exec caddy stat -c %i /etc/caddy/Caddyfile
     ```
   - 改挂载必须重建容器（本机实测**停机约 1 秒**，且证书在 `caddy_data` 卷里不受影响）。重建后**务必把旧路径上的文件改名或删除**，否则会留下一个"改了没反应"的诱饵。
   - 若暂时不能重建，应急做法（零停机）：`docker cp Caddyfile <容器>:/tmp/Caddyfile` 再 `caddy reload --config /tmp/Caddyfile`。
3. 校验用的 `--config` 也要确认指向**容器内**路径；`caddy validate` 只验语法，不保证你改的那份就是运行中的那份。

三点提醒：

1. **两条 `header` 不要写成互相重叠的路径。** 重叠时哪条生效取决于版本与求值顺序，是个隐蔽的坑；上面用 `@immutable` 里的 `not path ...` 把清单排除掉，规则互不重叠，就不依赖顺序了。
2. `file_server` 没有 nginx 的 `index` 覆盖选项，所以 §3 用「把 `preview.html` 发布成 `index.html`」来满足 `/`，而不是依赖 `try_files`（后者需要 Caddy ≥ 2.8）。
3. `not` 形式的 matcher 需要较新的 Caddy 2.x（若将来按 §5 启用压缩，还要 `precompressed`）；部署前用 `caddy version` 与 `caddy validate` 各自确认一次。注意 `caddy validate` **不接受** `--address`，而 `caddy reload` 需要它。
4. **加载后实测一次响应头**，确认压缩与缓存真的按预期生效：

```bash
curl -sI --compressed https://<预览子域>/assets/REPORT-<hash>.md | grep -iE 'content-encoding|vary|cache-control'
curl -sI --compressed https://<预览子域>/artifacts/preview/values.350ad897.bin | grep -iE 'content-encoding|vary|cache-control'
curl -sI https://<预览子域>/artifacts/preview/meta.json | grep -i cache-control
```

第一条应看到 `content-encoding: gzip`（或 `zstd`，取决于 Caddy 的 `encode`）与 `vary: Accept-Encoding`；**第二条是 `.bin`，按 §5 的决定应看到 `cache-control: immutable` 且没有 `content-encoding`**；第三条应是 `no-cache`。**若第一条没有 `vary`，先别接 CDN。**
再补一条现在最要紧的检查（换检查点后会直接咬人）：

```bash
for f in meta.json body_config.json scene.xml yumi.xml; do
  curl -sI "https://<预览子域>/artifacts/preview/$f" | grep -i 'cache-control'
done      # 四个固定名都必须是 no-cache
```

## 来源

- [Cloudflare China Network](https://developers.cloudflare.com/china-network/)
- [Cloudflare Cache](https://developers.cloudflare.com/cache/)
- [Caddy file_server](https://caddyserver.com/docs/caddyfile/directives/file_server)
- [MuJoCo WASM bindings（@mujoco/mujoco）](https://www.npmjs.com/package/@mujoco/mujoco)

模型与数据的溯源见 `research/REPORT.md`、`data/full_graph.json`；角色许可与署名见 `THIRD_PARTY.md`、`research/avatar/YUMI.md`。云实例的计费与保留规则见 `docs/CLOUD.md`。
