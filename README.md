<p align="center">
  <img src="assets/logo.png" alt="pagoda" width="560" />
</p>
<p align="center"><a href="https://github.com/quantz8a/pagoda/actions/workflows/ci.yml"><img src="https://github.com/quantz8a/pagoda/actions/workflows/ci.yml/badge.svg" alt="CI" /></a> <a href="https://crates.io/crates/pagoda"><img src="https://img.shields.io/crates/v/pagoda.svg" alt="crates.io" /></a> <a href="https://docs.rs/pagoda"><img src="https://docs.rs/pagoda/badge.svg" alt="docs.rs" /></a> <a href="https://github.com/quantz8a/pagoda/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22"><img src="https://img.shields.io/github/issues/quantz8a/pagoda/good%20first%20issue" alt="good first issues" /></a> <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="contributions welcome" /></a></p>

Apple Silicon 上的 SGLang 式服务：一个 Rust 二进制。OpenAI 兼容接口、radix 前缀缓存、连续批。不带 Torch。

当前保证一个量化模型：Qwen2.5-0.5B-Instruct Q4_K_M。贪心生成，重复的提示走前缀缓存。2026-10-16 在 M3 Ultra 上，32 token 热解码 **64 tok/s**，第二次请求只把 32 个 token 送进模型（冷跑是 112）。mlx-lm 和 SGLang 的 MLX 路径没有在这台机器上重测。

*SGLang-style serving as one Rust binary on Apple Silicon: OpenAI-shaped API, radix prefix cache, continuous batching, no Torch. The supported checkpoint today is Qwen2.5-0.5B-Instruct Q4_K_M.*

## 仓库结构

| 目录 | 说明 |
| --- | --- |
| [`pagoda/`](pagoda/) | 主 crate：**零依赖、离线可编译可测试**（78 项测试）。引擎、双前缀缓存（Radix/APC）、分页 KV、调度、约束解码、HTTP 服务、DSL |
| [`pagoda-hf/`](pagoda-hf/) | 真实权重。Apple Silicon 用 `--features metal`，`PAGODA_DEVICE=metal`。Qwen2 GGUF 见 `examples/qwen2_gguf.rs` |
| [`rs-mm/`](rs-mm/) | 多模态预处理 crate（零依赖）：Qwen2-VL 图像 `smart_resize` / patchify / token layout，离线可测 |

## 在 Mac 上跑这个模型

```bash
cd pagoda-hf
cargo build --release --features metal --example qwen2_gguf
PAGODA_DEVICE=metal cargo run --release --features metal --example qwen2_gguf -- \
  --gguf /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokenizer-file /path/to/tokenizer.json \
  --max-tokens 32
```

权重是 `Qwen/Qwen2.5-0.5B-Instruct-GGUF` 里的 `qwen2.5-0.5b-instruct-q4_k_m.gguf`，分词器是 `Qwen/Qwen2.5-0.5B-Instruct` 的 `tokenizer.json`。方法、冷热数字和 Metal 上 Q4 展开成 F16 的原因写在 [`pagoda/docs/BENCHMARK-METAL.md`](pagoda/docs/BENCHMARK-METAL.md)。

| | 墙钟 | 送进模型的 token | 产出吞吐 |
| --- | ---: | ---: | ---: |
| cold（81 token 提示 + 32 token 生成） | 514 ms | 112 | 62 tok/s |
| warm（同一提示再跑） | 497 ms | 32 | 64 tok/s |

两次贪心输出逐 token 相同。warm 少喂的 80 个 token 是前缀嫁接，不是少生成。

## 离线引擎（教学）

主 crate 不加载权重。下面这条命令跑的是 n-gram，用来看调度，不用来回答问题。

```bash
cd pagoda
cargo test                    # 离线测试全绿即环境 OK
cargo run --bin pagoda -- sample -p "你好" --repeat 2
cargo run --bin pagoda -- serve --port 8080
```

## 从这里开始读

- 小白文档体系（从"什么是 LLM 推理服务"讲起）：[`pagoda/docs/guide/README.md`](pagoda/docs/guide/README.md)
- 架构设计与路线图：[`pagoda/docs/DESIGN.md`](pagoda/docs/DESIGN.md)
- Apple Silicon 上 Qwen2.5-0.5B Q4 的实测：[`pagoda/docs/BENCHMARK-METAL.md`](pagoda/docs/BENCHMARK-METAL.md)
- 与 SGLang / HF transformers 的旧对比（Linux 共享机）：[`pagoda/docs/BENCHMARK.md`](pagoda/docs/BENCHMARK.md)
- 卖点与差异化：[`pagoda/docs/SELLING-POINTS.md`](pagoda/docs/SELLING-POINTS.md)

## 参与贡献

想在社区里露一手？看这里：

- 贡献指南（十分钟跑通 + PR 规范）：[`CONTRIBUTING.md`](CONTRIBUTING.md)
- 首批可认领任务（难度 L0–L3 + 从哪下手 + 怎么自测）：[`GOOD_FIRST_ISSUES.md`](GOOD_FIRST_ISSUES.md)
- 项目目标 → 四象限性能的公开路线图（可认领入口）：[`https://quantz8a.github.io/pagoda/roadmap.html`](https://quantz8a.github.io/pagoda/roadmap.html)

无需 GPU、无需联网：`cargo test --offline` 全绿就能开工。

## License

[Apache-2.0](LICENSE) — 可自由使用、修改与分发，包括商用和闭源。与上游
SGLang 同属宽松开源生态，署名与血缘由 `NOTICE` 保留。
