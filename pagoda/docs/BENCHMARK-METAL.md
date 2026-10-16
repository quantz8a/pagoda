# Apple Silicon：Qwen2.5-0.5B Q4_K_M

测于 2026-10-16，Apple M3 Ultra，96 GiB 统一内存，macOS 26.5.1。
一次进程内运行，release 构建，`PAGODA_DEVICE=metal`。数字是这一次运行的墙钟，不是实验室多轮均值。

## 保证的范围

一个量化模型：`Qwen/Qwen2.5-0.5B-Instruct-GGUF` 的 `qwen2.5-0.5b-instruct-q4_k_m.gguf`。
贪心（temperature 0），重复同一段提示时走张量前缀嫁接。
没有张量并行、MoE、投机解码。批量解码在这条 GGUF 路径上走逐条前向。

Metal 上 candle 0.8.4 的 Q4 矩阵乘会把 SwiGLU 乘到 F16 上限（65504）并变成 NaN。
加载时把 GGUF 权展开成 F16，之后用普通 Metal GEMM。CPU 仍走原生 Q4 内核。

## 方法

- 提示是 `"The quick brown fox jumps over the lazy dog. "` 重复 8 次，分词后 81 token。
- 生成 32 token，贪心。
- cold：新引擎的第一次 `generate`。
- warm：同一提示再跑一次。`warm_tokens_fed` 是第二次实际送进模型的 token 数。
- 脚本：`pagoda-hf/examples/qwen2_gguf.rs`。

## 结果

| | cold | warm |
| --- | ---: | ---: |
| 墙钟 | 514 ms | 497 ms |
| 送进模型的 token | 112 | 32 |
| 产出吞吐 | 62 tok/s | 64 tok/s |
| 输出 | 两次逐 token 相同 | |

112 = 81 个提示 token + 31 步解码（第 32 个输出 token 由最后一次前向的 logits 采样，不再喂回）。
warm 的 32 = 嫁接之后剩下的 1 个提示 token + 31 步解码。81 个提示 token 里有 80 个没有重算。

加载权重 1.28 s（含 Q4→F16 展开）。

同一次 CPU 对照（`PAGODA_DEVICE=cpu`，原生 Q4，max-tokens 8）输出同一句开头
` What is the next sentence in the sequence`，warm 送进模型的 token 数等于生成长度。

## 和本机 mlx-serve 的 Qwen3.8-27B

同一台 M3 Ultra，2026-10-16。mlx-serve 已经在 `127.0.0.1:11246` 加载
`mlx-community/Qwen3.8-27B-8bit`（约 29.5 GB 常驻，8-bit，`--prefix-cache-mem 4GB`）。
流式请求，temperature 0。

| 场景 | TTFT | 解码 | 前缀缓存 |
| --- | ---: | ---: | --- |
| 短提示，80 token 生成，第一次 | 0.59 s | 24.1 tok/s | 0 / 30 |
| 同一提示再跑 | 0.05 s | 24.0 tok/s | 29 / 30 |
| 827 token 提示，40 token 生成，第一次 | 1.47 s | 23.2 tok/s | 387 / 827 |
| 同一长提示再跑 | 0.29 s | 23.0 tok/s | 796 / 827 |

这个模型的 `config.json` 是 `qwen3_5`：64 层里每 4 层才有一层全注意力，其余是线性注意力（`linear_num_key_heads=16`，`linear_conv_kernel_dim=4`）。pagoda 的 GGUF 路径只实现了 Qwen2 全注意力，加载这份权重会算错，所以没有拿它冒充对比。

pagoda 能跑的更大模型是 Qwen2.5-7B-Instruct Q4_K_M（bartowski 单文件）。Metal 上展开成 F16 后，81 token 提示 + 32 token 贪心：

| | 墙钟 | 送进模型的 token | 产出 |
| --- | ---: | ---: | --- |
| cold | 122.7 s | 112 | 0.26 tok/s |
| warm | 153.3 s | 32 | 0.21 tok/s |

输出是连贯的（重复 “The quick brown fox…”）。前缀嫁接生效（112→32）。解码比 mlx-serve 上的 27B 慢大约两个数量级。这次 7B 运行时 27B 仍常驻内存，墙钟还含着争用；即便把争用去掉，未融合的 F16 GEMM 也到不了 mlx 的 8-bit 解码。

文本塔已经能加载这份 MLX 目录（`pagoda-hf/src/qwen3_5.rs`）。视觉塔不读。`PAGODA_DEVICE=metal` 时，每个矩阵第一次反量化成 F16 后留在 GPU 上，后面的 token 只做 GEMM。2026-10-16 同一句 `Hi`、贪心、4 个 token，CPU 和 Metal 都和 mlx-serve `/v1/completions` 一致：`, I'm a`。

| 设备 | 第 1 个 token | 之后 |
| --- | ---: | --- |
| CPU，每次反量化 | 22 s | 14 s/token |
| Metal，F16 常驻 | 86 s（含展开上传） | 7.6 s、0.9 s、0.3 s |
| Metal，8-bit GEMV | 46 s（含打包权重上传） | 4.2 s、0.5 s，然后 0.2 s/token |

8-bit 路径的输出仍是 `, I'm a 20-year`，logit 和本机 mlx-serve 的贪心续写一致（`12.848 / 13.408 / 18.540`）。2026-10-17 热解码墙钟，跳过加载后的 7 个 token 平均 **23.5 tok/s**（42–43 ms）。同刻 mlx-serve 32 token 三次中位 **23.3 tok/s**。文本塔权重大约 27.2 GB（含 bf16 scale/bias 和 lm_head）。

这一步主要不是 GEMV。原先全注意力是每个头一个线程，kv=32 时一层就要约 2 ms，16 层约 32 ms，所以生成越长越慢。改成每个头一个 simdgroup、在线 softmax 之后，同样长度约 **0.11 ms/层**。Gated delta 从约 106 µs 降到约 **31 µs**（48 层从 5.1 ms 降到 1.5 ms）。权重在 Private 缓冲里，scale/bias 是 bf16。

MLP 瓦片（17408×5120）稳定在大约 **590–600 GB/s**。累加顺序和这条 fp32 链不一样的写法（把长 K 拆成两路再相加）点积会偏大约 0.016，没有采用。`Hi` 贪心 16 token 与 mlx-serve 逐字相同。

### 单流还能再抠哪里

| 杠杆 | 现状 | 天花板感 |
| --- | --- | --- |
| Affine 8-bit GEMV | 主耗时，MLP 瓦片约 600 GB/s | 再抬到约 650 GB/s 就能过 mlx 的 23.3 |
| 全注意力 | simdgroup + 在线 softmax | 短上下文已经不是大头 |
| Prefill / 冷启动 | 权重打包上传、未融合 prefill | 改善 TTFT，不抬稳态 decode |
| 投机解码等 | 未做 | 另一条产品线 |

结论：短上下文热解码大约 **23.3–23.5 tok/s**，和同机 mlx-serve 的 **23.3** 持平或略高。`Hi` 的 16 个贪心 token 与 mlx-serve 一致。更快但会改 fp32 累加顺序的核没有进默认路径。

### 与 SGLang（Python）在 Mac 上

SGLang 在 Apple Silicon 有两条路径（见 [SGLang MLX PR](https://github.com/sgl-project/sglang/pull/20342) / [说明 gist](https://gist.github.com/yeahdongcn/161f0718d55c7022791261e6d6a0b57d)）：

1. **默认 PyTorch MPS**：调度还在，算子走 MPS，解码慢一个数量级。
2. **`SGLANG_USE_MLX=1`**：计算走 **mlx-lm / `mx.fast`**，调度仍是 SGLang。单流速度应接近同机 mlx-serve；**跨请求 radix / 前缀 KV 复用在 MLX 路径上尚未接通**（官方限制说明：每请求独立 KV）。

公开小模型对照（**M1 16GB**，Qwen3-0.6B，`bench_one_batch`，in=60 / out=10，非本机）：

| 后端 | Prefill | Decode（median） |
| --- | ---: | ---: |
| SGLang + PyTorch MPS | ~230 tok/s | **~7.9 tok/s** |
| SGLang + MLX native | ~903 tok/s | **~31 tok/s** |

本机 **M3 Ultra 96GB**、同一权重 `Qwen3.8-27B-8bit`（本机未装 SGLang 进程；MLX 路径用已测的 mlx-serve 作代理——同一套 mlx-lm 内核）：

| 栈 | 热解码 | 前缀 / agent 树干 |
| --- | ---: | --- |
| SGLang Python + MPS（量级） | 远慢于 MLX（公开 ~4×） | 有 radix，但 MPS 单流太慢 |
| SGLang Python + MLX | **≈ mlx-serve ~23.5 tok/s**（代理） | **暂无跨请求 KV 复用** → 多 agent 近似每人付满树干 |
| mlx-serve（本机实测） | **23–24 tok/s** | prefix cache 可碰巧命中 |
| **pagoda Metal 融合**（本机） | **~23.5 tok/s** | **checkpoint pin + fork 契约** |

读法：Mac 上单聊吞吐，SGLang-MLX / mlx 仍快一截；多 agent 共享长 system prompt 时，SGLang 的 Mac-MLX 路径目前**没有** CUDA 上那套 radix 省树干的能力，pagoda 的优势在契约式复用，不在 GEMV。

```bash
PAGODA_DEVICE=metal cargo run --release --features metal --example qwen3_8 -- \
  --model ~/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit \
  --prompt "Hi" --max-tokens 32
```

## 复现

```bash
cd pagoda-hf
cargo build --release --features metal --example qwen2_gguf
PAGODA_DEVICE=metal cargo run --release --features metal --example qwen2_gguf -- \
  --gguf /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokenizer-file /path/to/tokenizer.json \
  --max-tokens 32
```

权重文件来自 `Qwen/Qwen2.5-0.5B-Instruct-GGUF`，分词器 `tokenizer.json` 来自
`Qwen/Qwen2.5-0.5B-Instruct`。`--repo` / `--file` / `--tokenizer` 走 Hugging Face
客户端；镜像站若缺 `Content-Range`，客户端会失败，改用上面的本地路径。
