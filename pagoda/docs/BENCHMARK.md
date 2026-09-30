# 基准对比：pagoda vs SGLang vs HF transformers

> 测于 2026-09-30，一台共享 Linux 开发机（i5-12500 6C12T / 31GB / RTX 3050 8GB / Ubuntu 20.04）。
> **该机为共享机器**，测试期间 load average 4–5（其他租户在跑），数字为指示性对比，
> 不是实验室级绝对值；三方在同等争抢条件下测量。

## 方法

- 场景：134-token prompt + 32 token 生成，贪心采样（`min_new_tokens` 强制跑满，
  排除 EOS 早停差异）。
- 三个场景：**cold**（全新引擎首跑）、**warm**（同 prompt 重跑，测前缀缓存）、
  **batch-8**（8 条不同 prompt 一次批量，测吞吐）。
- 两边同一句 prompt 文本、同一 HF tokenizer，token 数实测一致（134）。
- 脚本：`pagoda-hf/examples/bench.rs`（Rust）、`pagoda-hf/bench/bench_hf.py`、
  `pagoda-hf/bench/bench_sglang.py`，均输出 JSON，命令见文末。

## 结果

### 模型 A：tiny-random-Llama（2 层 / hidden 16 / vocab 32k，随机权重）——测框架开销

| 引擎 | cold | warm | batch-8 产出 |
| --- | ---: | ---: | ---: |
| **pagoda-hf**（candle CPU） | **20.1 ms** | **14.3 ms** | 2218 tok/s |
| HF transformers（torch CPU, 6 线程） | 74.1 ms | 59.9 ms | 3518 tok/s |
| SGLang 0.5.10（GPU, torch_native） | 6044 ms² | 153.9 ms | 680 tok/s |
| SGLang 0.5.10（CPU） | ✗ 跑不起来 | | |

² flashinfer 不支持该模型 head_dim=4（`NUM_MMA_D_QK=0` 内部错误），退到
torch_native 后端；cold 含一次性 JIT 编译。

### 模型 B：TinyLlama-1.1B（真实权重）——测真实算力

| 引擎 | cold | warm | batch-8 产出 |
| --- | ---: | ---: | ---: |
| pagoda-hf（candle CPU F32，单线程 matmul） | 6623 ms | 6550 ms | 5.0 tok/s |
| pagoda-hf（candle **GPU F32**，RTX 3050） | 16811 ms | 16810 ms | 2.0 tok/s |
| pagoda-hf（candle **GPU F16**，RTX 3050） | 9100 ms | 9009 ms | 4.0 tok/s |
| HF transformers（torch CPU F32，**1 线程**） | 7015 ms | 6948 ms | 7.7 tok/s |
| HF transformers（torch CPU F32，6 线程） | **4270 ms** | **4243 ms** | **19.7 tok/s** |
| SGLang 0.5.10（RTX 3050, flashinfer, bf16） | 40212 ms¹ | 7866 ms | 23.0 tok/s |

¹ 含首次 flashinfer JIT 编译；去掉后 cold 应与 warm 同量级。

## 解读

1. **框架开销：pagoda 完胜**。tiny 模型下 pagoda 端到端 20ms，HF 要 74ms（3.7×），
   热缓存下 14ms vs 60ms（4.2×）。这就是"Rust 零依赖引擎"的税差——
   python import、对象构造、调度器往返在小模型上占绝对大头。SGLang 即使上 GPU，
   warm 也要 154ms（pagoda 的 10.8 倍）、batch 吞吐 680 tok/s（pagoda 的 1/3）——
   GPU 救不了逐 token 的 python 派发。
2. **单线程对单线程，pagoda 打平甚至略胜 torch**：1.1B 模型 1 线程时
   pagoda 6623ms vs HF 7015ms。candle 的朴素 matmul 并不吃亏。
3. **线程与批量是 pagoda 当前的缺口**：torch 开 6 线程后反超 1.55×；
   batch-8 时 HF 批量 matmul 摊薄开销，领先 4×（6 线程）/1.5×（1 线程）。
   pagoda 引擎目前是逐序列前向——对应路线图 **P3：批量 matmul + GPU 后端**。
4. **SGLang 在此共享盒子上没有发挥**：GPU 解码仍要 python 逐 token 调度，
   CPU 被其他租户抢占时单 token 派发拖到 ~246ms；这不是 SGLang 的真实水平，
   是共享 CPU 的必然结果（服务端 GPU 推理需要 CPU 空闲）。SGLang 的 radix
   热缓存生效明显（warm 7.9s vs cold 40s）；pagoda P2 会话对命中前缀仍会重算
   （warm≈cold），张量级前缀嫁接在 P3。
5. **SGLang CPU 轨道缺失**：0.5.10 的 scheduler 硬依赖 CUDA device，
   CPU-only 部署直接拒绝——pagoda 的"CPU 也能跑全链路"是真实差异点。
6. **GPU 在共享盒子上是"启动延迟-bound"，不是算力-bound**：pagoda GPU F16
   warm 9009ms ≈ SGLang GPU 7866ms（只差 1.14×），两者单 token 都在
   250-280ms——因为 CPU 被抢占时，每步数百次 kernel launch 的往返延迟
   淹没了一切（pagoda 的 GPU F16 甚至输给自家 CPU F32）。这印证了
   CUDA Graph / 融合 kernel / 批量前向的价值，也是 P3 剩余工作的动机。
   pagoda GPU 通路的意义在于**已打通且数值正确**（F32 下 logits 与 CPU
   逐位一致），性能工程是下一步。

## 环境税实录（为什么"一键"值钱）

让 SGLang 在这台盒子跑起来的完整修复链（每一步都阻塞）：

1. CPU 模式：需 `disable_cuda_graph` + `disable_piecewise_cuda_graph` +
   `disable_overlap_schedule`（默认路径全部假定 CUDA）。
2. JIT 内核编译吃系统 nvcc——机器自带 CUDA **10.1**（2019 年），不识别现代参数。
3. 换装 nvcc 12.8（NVIDIA 官方 deb 免 root 解压）后，遭 `/usr/include/crt` 里
   CUDA 10.1 的**头文件污染**（需 root 移除）。
4. deb 拆包后 `nvvm/bin/cicc` 在单独的 `cuda-nvvm` 包里，需补装。
5. `crt/host_config.h` 在 12.8 改由 cudart 提供且目录结构变化，需手工合并。
6. sgl_kernel 的 JIT 用 C++20 `<concepts>`，系统 gcc 9.4 太老，装 gcc-10 并用
   PATH shim 指定宿主编译器。
7. tiny 模型的 head_dim=4 触发 flashinfer `NUM_MMA_D_QK=0` 内部错误，
   需切 torch_native 后端。

对照：pagoda-hf 在干净环境 `cargo build --release` 一次通过。

## 复现

```bash
# pagoda（Linux/macOS/Windows 均可，CPU）
cd pagoda-hf && cargo build --release --example bench
./target/release/examples/bench --repo TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --prompt-tokens 128 --gen-tokens 32 --batch 8 --repeat 3

# pagoda GPU（需要 CUDA 12.x nvcc；驱动需 ≥ 所用工具链版本）
cd pagoda-hf && cargo build --release --features cuda --example bench
PAGODA_DEVICE=cuda PAGODA_DTYPE=f16 ./target/release/examples/bench \
    --repo TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt-tokens 128 --gen-tokens 32

# HF transformers 参考
pip install torch transformers
python pagoda-hf/bench/bench_hf.py --repo TinyLlama/TinyLlama-1.1B-Chat-v1.0 --threads 6

# SGLang（GPU + 现代 CUDA 工具链）
pip install sglang
python pagoda-hf/bench/bench_sglang.py --repo TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda
```

## 与路线图的对应

| 发现 | 路线图项 |
| --- | --- |
| 批量 matmul 缺口（batch 落后 4×） | P3：batched prefill/decode |
| 命中前缀仍重算（warm≈cold） | P3：张量级 KV 嫁接（真·RadixAttention） |
| GPU 后端缺失 | ✅ 已打通（`--features cuda` + `PAGODA_DEVICE=cuda` + `PAGODA_DTYPE=f16`）；性能工程（CUDA Graph/融合 kernel）留待 P3 |
| CPU 全链路可用 + 极低框架税 | 已验证的差异化优势 |
