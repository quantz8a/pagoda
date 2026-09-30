# P1 验证手册：真实 tokenizer + 真实权重

目标：证明 pagoda 引擎的调度/缓存/采样链路在**真实构件**下端到端成立
（真实 BPE 分词 + 真实 safetensors 权重前向）。

> 本仓库主 crate 保持零依赖离线闭环；本手册的操作**需要联网**
> （拉取 cargo 依赖 + HuggingFace 模型文件）。

## 一键验证（联网机器）

```powershell
cd pagoda-hf
powershell -ExecutionPolicy Bypass -File scripts\verify-p1.ps1          # 直连
powershell -ExecutionPolicy Bypass -File scripts\verify-p1.ps1 -Mirror  # 国内镜像 hf-mirror.com
```

```bash
cd pagoda-hf
bash scripts/verify-p1.sh            # 直连
bash scripts/verify-p1.sh --mirror   # 国内镜像 hf-mirror.com
```

脚本全自动完成：拉取 cargo 依赖 → release 编译 → 下载 tiny 模型 → 跑全部断言，
日志落盘 `verify-p1.log`，末尾打印 `P1 VERIFICATION OK` 即通过。

使用模型：`hf-internal-testing/tiny-random-LlamaForCausalLM`
（公开免授权、体积小；权重是随机初始化的，**输出文本无意义是正常的**——
验证的是链路而不是智能）。

## 通过标准（example 内置断言，逐条对应）

| 步骤 | 验证点 | 断言 |
| --- | --- | --- |
| 1 | HF tokenizer 加载与往返 | encode 非空、decode 非空 |
| 2 | Candle 权重加载 + 单次前向 | logits 宽度 = vocab、全部有限值 |
| 3 | 冷缓存生成 | 非 Fault、非 Rejected、prefix_hit = 0 |
| 4 | 热缓存复用 | prefix_hit = 整个 prompt、forward 数下降、greedy 输出逐 token 一致 |

末尾打印 `P1 VERIFICATION OK` 即通过。

## 验证记录

**2026-09-30 · 联网 Linux 开发机（x86_64, CPU F32）· 通过 ✅**

实测输出（tiny-random 权重，输出文本无意义属预期）：

```
==> [1/4] download + load HF tokenizer from hf-internal-testing/tiny-random-LlamaForCausalLM
    vocab_size=32000 eos=2 bos=Some(1)
    encode -> [1, 450, 7483, 310, 3444, 338]
    decode -> "The capital of France is"
==> [2/4] download config + safetensors weights, load on CPU (F32)
    model loaded: candle-llama
    forward ok: vocab=32000 min=-0.323 max=0.322
==> [3/4] engine generation, cold cache
    11.018314ms Length prefix_hit=0/6 forward=22
    text: "wurdenWS befhrteitect^{ ft helpfulج後 BodAtIndexskyHowever officially rout"
==> [4/4] identical request, warm cache
    11.123924ms Length prefix_hit=6/6 forward=16
==> stats: saved=6 skip=0.50 faults=0 rejected=0 kv_util=0.00

P1 VERIFICATION OK — real tokenizer + real weights drive pagoda end to end
```

要点解读：

- `forward ok` 的 logits **min/max 非零**（-0.323 / 0.322）：真实权重确实参与前向，
  不是静默 fallback 的均匀分布。
- 热缓存 `prefix_hit=6/6`（整个 prompt 命中）、`forward` 数从 22 降到 16、
  `saved=6 / skip=0.50`：radix 前缀缓存在真实分词下按预期复用。
- `faults=0`：全链路零故障。

调试中修复的两个 candle-transformers 0.8.4 行为（供后来者参考）：

1. `Cache::new(use_kv_cache=true, ...)` 会在每次 forward 追加 K/V 而不重置，
   全量重放场景触发 mask broadcast 维度错误（`[6,6] -> [1,4,6,12]`）——
   pagoda-hf 现以 `use_kv_cache=false` 构建（正确性优先，O(n²) 见下）。
2. `Llama::forward` 返回 `[b_sz, vocab]`（末位置已归约），不是 `[1, seq, vocab]`——
   logits 提取已改为 rank 无关的 `flatten_all` + 取末 `vocab_size` 个元素。

## P2 验证记录（增量 KV 会话 + checkpoint 分叉）

**2026-09-30 · 联网 Linux 开发机 · 通过 ✅**（同一 e2e 程序的第 3-5 步）

```
==> [3/5] engine generation, cold cache (incremental KV session)
    5.993083ms Length prefix_hit=0/6 forward=22
    text: "wurdenWS befhrteitect^{ ft helpfulج後 BodAtIndexskyHowever officially rout"
    model tokens fed: 21 (full replay would be 216, 10.3x saved)
==> [4/5] identical request, warm cache
    6.708585ms Length prefix_hit=6/6 forward=16
    model tokens fed: 21 (session recomputes the hit prefix once; cross-sequence KV sharing is the P3 milestone)
==> [5/5] checkpoint fork: shared trunk, branched continuation
    Length prefix_hit=6/11 branch_fed=12 (continuation 5 + 7 decode)

P1 VERIFICATION OK — real tokenizer + real weights drive pagoda end to end
P2 VERIFICATION OK — incremental KV sessions + checkpoint fork verified
```

要点解读：

- `tokens fed: 21`：prompt 6 个 token 预填充一次 + 15 步解码各 1 个。
  对比全量重放的 216（6+7+…+21），**省 10.3 倍**；墙钟同步下降（11ms → 6ms）。
  `tokens_fed` 是 `CandleModel::tokens_fed_handle()` 的实测计数，不是账本估计。
- 输出文本与 P1 全量重放逐字节一致——增量会话不改变贪心结果。
- `[5/5]` 树干 6 token 只预填充一次（创建 checkpoint 时）；两条分支各自只喂
  续写 5 + 解码 7 = 12 token，`fork` 零重算树干，且两分支输出逐 token 一致。

## 在任意联网 Linux 机器上执行

```bash
git clone https://github.com/quantz8a/pagoda.git && cd pagoda
cd pagoda && cargo test                                   # 主 crate 全绿基线
cd ../pagoda-hf && bash scripts/verify-p1.sh --mirror     # 国内镜像；直连去掉 --mirror
```

## 换真实大模型

example 里把 `REPO` 换成任何 candle-transformers 支持的 Llama 系仓库即可，
例如 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`（免授权，约 2.2 GB，输出开始像话）。
meta-llama 官方仓库是 gated 的，需要：

```bash
$env:HF_TOKEN="hf_..."   # Windows PowerShell
export HF_TOKEN=hf_...   # Linux
```

多分片权重的仓库：`llama_from_hub` 只取单文件 `model.safetensors`；
分片仓库请改用 `llama_from_safetensors(config, &[所有分片路径], device)`。

## 已知限制（当前版本）

- ~~全量重放前向 O(n²)~~ → **P2 已解决**（2026-09-30）：引擎每序列持有一份
  增量 KV 会话（`ModelSession`），prompt 一次预填充、之后每步只喂 1 个 token。
  实测 16 步生成模型吃 21 token（全量重放需 216，省 10.3 倍）。遗留两条：
  - candle-transformers 0.8 的因果 mask 是 `[seq, seq]` 方阵，cache 非空时
    只能逐 token 喂入（`seq_len==1` 跳过 mask）——多 token 续写被拆成逐 token
    前向，生产实现应换 flash-attn 或自研 kernel。
  - 会话 KV 是**序列私有**的：新会话遇到 radix 命中的前缀仍要自己预填充一遍。
    跨序列张量级 KV 嫁接（真·RadixAttention）是 P3。
- **grammar 约束在真实 tokenizer 上会被拒绝**（`unsupported`）：字节级 mask 只对
  字节级 tokenizer 合法，这是显式护栏而非缺陷。

## 排障

| 症状 | 原因与对策 |
| --- | --- |
| `no matching package named 'anyhow'` | 没网 / 代理未配。配 `CARGO_HTTP_PROXY` 或换镜像源 |
| `api::sync` 不存在 | hf-hub 需要 `features=["ureq"]`（本仓库已修） |
| `failed to download ... 401/403` | gated 仓库，需要 HF_TOKEN 或换免授权仓库 |
| `missing field use_flash_attn` | Candle 版本差异；加载器已做 Value 级容错，若仍报错升级 candle-transformers |
| 输出是乱码 | tiny-random 权重是随机的，正常；换 TinyLlama 看像话输出 |
