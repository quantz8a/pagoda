# Good-First Issues · 首批认领清单

每条都能独立完成，难度分 L0–L3。认领前先跑通 `CONTRIBUTING.md` 的「十分钟跑通」。
标 ⭐ 的是最推荐的入门任务。

## A. Rust 多模态（MM）处理器移植线 ⭐

> 背景：`rs-mm/` 已并入本仓库。Qwen2-VL 叶子已闭环（`src/model/qwen2_vl.rs`），
> 接口在 `pipeline.rs` 的 `MmFamilyProcessor`。后续叶子照抄 Qwen2-VL，复用
> `common::resize` / `common::token_layout`。算法血缘见根目录 `NOTICE`
> （SGLang `rust/sglang-mm`，Apache-2.0）。

- ~~**[L1] 移植 Qwen2-VL 图像处理器** `model/qwen2_vl.rs`~~ → 已合入，见 `rs-mm/`

- **[L1] 移植 MiniCPM-V 图像处理器** `model/minicpmv.rs`
  - 从哪下手：参考 `minicpmv.py` 的 slice 切分；`layout` 用 `TokenPattern::Explicit` 拼
    slice 分隔符。
  - 自测：sliced 长图 1 vs 多 slice 的 token 数对得上。
  - 驱动轴：TTFT。

- **[L1] 移植 LLaVA-1.5 图像处理器** `model/llava.rs`
  - 从哪下手：最简单的一条——单图 resize 到固定 grid，`closest_ratio` 都不用。
  - 自测：`feature.shape == [1, 3, h, w]`，normalize 值域校验。
  - 驱动轴：Useful Life（最常见生态覆盖）。

- **[L2] `common/decode` 接真实 PNG/JPEG 解码**
  - 从哪下手：目前 `decode.rs` 只有 BMP 最小闭环；接入零依赖纯 Rust 解码，或先只加
    PNG（flate），JPEG 留占位错误。
  - 自测：写死一张 base64 小图，解码后像素与参考一致。
  - 驱动轴：Useful Life。

## B. 调度 / 缓存

- **[L1] 第 4 种调度策略**：`engine.rs` 的 `SchedulePolicy` 加 `ShortestRemaining`。
  - 自测：构造长短混合队列，断言出队顺序。
  - 驱动轴：TTFT。

- **[L1] Radix 命中率单测扩展**：`radix_cache.rs` 补分支/删除节点后的命中率断言。
  - 驱动轴：TTFT · Token/Watt（compute-skip 口径）。

- **[L2] `evict_lru` 非叶节点回收策略**：目前只回收叶子，给非叶加「父子合并」或
  「引用为 0 才可回收」的规则并补测试。
  - 驱动轴：Token/Watt · Useful Life。

## C. 兼容 / 生态

- **[L1] OpenAI 端点契约回归测试**：`tests/system_tests.rs` 锁死 `/v1/chat/completions`
  与 `/generate` 的 request/response 形状。
  - 驱动轴：Useful Life。

- **[L1] 补一页 guide 文档**：任选一个已实现机制，写「为什么 + 怎么自测 + 一段可跑
  demo」，跟 `docs/guide/03-radix-prefix-cache.md` 对齐。
  - 驱动轴：Useful Life。

- **[L2] `pagoda-hf` 接一个新后端**：照 `src/llama.rs` 的结构加一个 RMSNorm+GQA+RoPE
  的模型（Qwen/Mistral 优先），并加一条 `e2e_*.rs`。
  - 驱动轴：Token/Watt（真实后端面）、Useful Life。

## D. 指标 / 四象限

- **[L1] `/stats` 字段单测**：锁死 14 个字段 + 4 个派生 KPI 的存在与单调性。
  - 驱动轴：Revenue（AI Factory 可观测）。

- **[L2] Token/Watt 真实度代理增强**：把 `avg_forward_per_output_token` 迁成
  `decode_batch_factor`、`kv_pages_reused` 的加权口径，给每个字段补语义注释。
  - 驱动轴：Token/Watt。

## 发 Issue 的标题模板

认领一个就按下面格式开新 issue（或让维护者复制）：

```text
[good-first-issue] <标题>
难度：L1
驱动轴：TTFT / Token-Watt / MTBI / Useful Life
从哪下手：<文件路径 + 参考实现>
自测：cargo test --offline 里 <测试名>
```

> 维护者注意：MM 线已在仓库根目录 `rs-mm/`。后续叶子（MiniCPM-V / LLaVA / decode）
> 直接在该 crate 里加文件即可；CI 的 `rs-mm` job 会跑 `cargo test --offline`。