<p align="center">
  <img src="assets/logo.png" alt="pagoda" width="560" />
</p>
<p align="center"><a href="https://github.com/quantz8a/pagoda/actions/workflows/ci.yml"><img src="https://github.com/quantz8a/pagoda/actions/workflows/ci.yml/badge.svg" alt="CI" /></a> <a href="https://crates.io/crates/pagoda"><img src="https://img.shields.io/crates/v/pagoda.svg" alt="crates.io" /></a> <a href="https://docs.rs/pagoda"><img src="https://docs.rs/pagoda/badge.svg" alt="docs.rs" /></a> <a href="https://github.com/quantz8a/pagoda/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22"><img src="https://img.shields.io/github/issues/quantz8a/pagoda/good%20first%20issue" alt="good first issues" /></a> <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen" alt="contributions welcome" /></a></p>

Apple Silicon 上的 SGLang 式服务：一个 Rust 二进制。OpenAI 兼容接口、radix 前缀缓存、连续批。不带 Torch。

当前保证一个量化模型：Qwen2.5-0.5B-Instruct Q4_K_M。贪心生成，重复的提示走前缀缓存。2026-10-16 在 M3 Ultra 上，32 token 热解码 **64 tok/s**，第二次请求只把 32 个 token 送进模型（冷跑是 112）。mlx-lm 和 SGLang 的 MLX 路径没有在这台机器上重测。

*SGLang-style serving as one Rust binary on Apple Silicon: OpenAI-shaped API, radix prefix cache, continuous batching, no Torch. The supported checkpoint today is Qwen2.5-0.5B-Instruct Q4_K_M.*

## 为什么用 pagoda 接 Codex++

同一台 Apple M3 Ultra（96GB）、同一份 `mlx-community/Qwen3.8-27B-8bit`。目录名里的 8 是量化位宽，这颗是 270 亿那一档的文本塔，视觉塔不读。

| 你想引用的话 | 数字 | 它量的是什么 |
| --- | --- | --- |
| 单流略快于同机 mlx-serve | 23.5 对 23.3 tok/s | 2026-10-17。权重已在 GPU 上，短上下文贪心，比的是稳住之后的解码 |
| 一次验证四个已知 token | 50.9 对 22.4 tok/s，约 2.28 倍 | 2026-10-18。四个输入已知且全部接受。逐条 pagoda 是 22.3。四个 logit 绝对差是 0 |
| 八个 Agent 少做 98% 的前向 | 32,368 → 624 次 | 约 3968 token 的共享树干，各生成 32 token。toy 引擎的前向计数，不是 27B 的墙钟 |
| 这一批活的 GPU 时间少大约 68% | 约 69 秒 → 约 22 秒 | 用本机 prefill / 解码速率估算，见下面的公式 |

一个人短聊，热解码已经和 mlx-serve 打平，还略快 0.9%。Codex 这种长提示反复续写，省的是把同一段系统提示和工具说明重算很多遍。0.9% 撑不起换引擎。98% 的前向撑得起，前提写在下面。

热解码的 23.5 不是打开的第一秒。第一 token 曾经要 46 秒，里面含着把打包权重上传到 GPU，然后才落到大约 0.2 秒一个。mlx-serve 的 23.3 也是模型已经常驻之后的解码。冷启动那一段，mlx-serve 更合适。同一句 `Hi` 的 16 个贪心 token 与 mlx-serve 逐字相同，对过的三个 logit 是 12.848 / 13.408 / 18.540。

50.9 tok/s 是老师强迫：四个输入已知，而且都会被接受。真做投机解码时，接受几个算几个；经常只中一个，就回到逐条那一档。`qwen3_server` 用这条路径做 prefill（已知 token 四个一组喂进去）。新生成的 token 仍是一次一个。

八个 Agent 共用约 3968 token 树干时，各自重算是 32,368 次前向，树干只付一次是 624 次，`(32368 − 624) / 32368 = 98.1%`。这是 toy 引擎的前向次数。毫秒墙钟不能外推成 27B。它量的是树干被付了 8 次，还是被付了 1 次。

前向少 98%，不等于 GPU 时间少 98%。一个解码 token 大约是一个 prefill token 的二十多倍墙钟。`T` 是树干，`S` 是每条分支的短后缀，`G` 是产出，`N` 是分支数。取整参数 `T = 4000`，`S = 50`，`G = 32`，`N = 8`，prefill 按本机 mlx-serve 约 560 tok/s，解码按当时约 18 tok/s：

```text
不共享    8 × 4050 / 560 + 256 / 18 ≈ 69 秒
树干一次  4400 / 560 + 256 / 18     ≈ 22 秒
```

少大约 68%。10 月 17 日解码测到 23.5 tok/s 之后，同一公式靠近 19 秒，节省靠近 73%。

### 接到 Codex++

Codex 0.133 要求 `wire_api = "responses"`。`wire_api = "chat"` 会被拒绝。请求打到 `POST /v1/responses`。

```bash
cd pagoda-hf
PAGODA_DEVICE=metal cargo build --release --features metal --bin qwen3_server
PAGODA_DEVICE=metal ./target/release/qwen3_server \
  --model ~/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit \
  --port 8000
```

不设 `PAGODA_DEVICE=metal` 会落到 CPU。进程先加载、再 warmup，然后才打印 `qwen3_server listening`。同一份 8-bit 权重大约 29.5GB。另一份推理进程已经把它常驻时，先停掉那个进程再启动。

2026-10-18 冒烟：`/health` 返回 `{"status":"ok","model":"qwen3.8"}`，`input` 为 “Reply with the single word pong.” 时，`output_text` 是 `pong`（提示 19 token，生成 1 token）。

`~/.codex/config.toml`：

```toml
model = "mlx-community/Qwen3.8-27B-8bit"
model_provider = "pagoda"
model_catalog_json = "/绝对路径/mlx-serve-models.json"

[model_providers.pagoda]
name = "pagoda"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
requires_openai_auth = false
```

`model_catalog_json` 里要有这个 slug。改完退出 Codex++ 再开。服务把 `instructions` 和 `input` 收成 Qwen 聊天模板，工具调用按权重目录里的 `chat_template.jinja` 写成 `<tool_call>`，再翻成 Responses 的 `function_call`。生成提示里的思考块是合上的。响应里 `temperature` 记 0，因为这条路径是贪心。下一轮 token 若是上一轮的延长，只喂新增后缀；对不上就整段重喂，日志写 `prefix miss`。

| 项 | 这版 |
| --- | --- |
| 提示 | 超过 16384 token 返回 400 |
| 生成 | `max_output_tokens`，再被帽在 4096 |
| 并发 | 一个请求 |
| 图像 | 换成文本 `[image]` |
| `/v1/responses/compact` | 404 |
| 解码 | 热了之后约 23 tok/s。50.9 只属于已知 token 的验证 |

## 仓库结构

| 目录 | 说明 |
| --- | --- |
| [`pagoda/`](pagoda/) | 主 crate：**零依赖、离线可编译可测试**（78 项测试）。引擎、双前缀缓存（Radix/APC）、分页 KV、调度、约束解码、HTTP 服务、DSL |
| [`pagoda-hf/`](pagoda-hf/) | 真实权重。Apple Silicon 用 `--features metal`，`PAGODA_DEVICE=metal`。Qwen2 GGUF 见 `examples/qwen2_gguf.rs` |
| [`rs-mm/`](rs-mm/) | 多模态预处理 crate（零依赖）：Qwen2-VL 图像 `smart_resize` / patchify / token layout，离线可测 |

## 模块边界

<p align="center">
  <img src="assets/module-boundary.png" alt="模块边界：qwen2_gguf 与 e2e_tiny_llama 进入 Engine；qwen3_8 的前向停在 Qwen35，分词接到 Tokenizer；laya 是示例" width="760" />
</p>

`qwen2_gguf` 和 `e2e_tiny_llama` 走 `Engine::generate`。`qwen3_8` 调用 `Qwen35::forward_greedy`，分词接到 `Tokenizer`，设备选择用 `CandleModel::device_from_env`。`laya` 是示例。

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
