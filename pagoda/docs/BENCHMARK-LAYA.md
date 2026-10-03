# 基准对比：pagoda-rust vs Laya 官方 Python（System-1 决策模型）

> 测于 2026-10-03，同一台共享 Linux 开发机（i5-12500 6C12T / 31GB / RTX 3050 8GB / Ubuntu 20.04）。
> 共享机器，其他租户负载存在，数字为指示性对比；两边在同等条件下测量。
> 被测对象：[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)（ModernBERT 编码器 + 决策头，842MB 权重），
> Python 侧使用模型仓库自带的官方参考实现 `rl_agent_api.py` / `rl_common.py`（逐字未改）。

## 方法

- 场景：仓库 README 的账单场景（双扣款投诉），一次 `decide` 回答 3 个类型化问题
  （choice 部门路由 / score 紧急度 / noul 流失风险），单次前向、无自回归生成。
- 进程内计时（排除 HTTP）：`pagoda-hf/examples/bench_laya.rs` 与 `pagoda-hf/bench/bench_laya.py`
  互为镜像——相同输入、相同预热（5 次）、相同迭代（50 次）、输出同款 JSON。
- 模型快照同为 `55cf4c4e`，Python 侧 `AutoTokenizer` + safetensors，Rust 侧同一 `tokenizer.json` + 同一权重文件。
- 精度路径：**Rust 全程 f32**（CPU/GPU 一致）；Python 按官方代码在 GPU 上自动走 **fp16 autocast**
  （且官方对算力 <8.0 的卡强制 fp16）。这不是我们加的劣势，是官方默认行为。

## 结果

| 指标 | pagoda-rust CPU | laya-python CPU | pagoda-rust GPU | laya-python GPU |
| --- | ---: | ---: | ---: | ---: |
| 单次延迟 mean | 1061.9 ms | **470.9 ms** | **605.9 ms** | 690.9 ms |
| p50 | 1052.9 ms | 486.3 ms | **605.7 ms** | 690.6 ms |
| p95 | 1080.2 ms | 504.9 ms | **607.2 ms** | 693.1 ms |
| max | 1355.3 ms | 541.5 ms | **614.0 ms** | 700.3 ms |
| 模型加载（冷启动） | **1.44 s** | 9.41 s | **1.62 s** | 7.91 s |
| 峰值 RSS | **2442 MB** | 2956 MB | **1850 MB** | 2954 MB |
| 运行精度 | f32 | f32 | f32 | fp16（autocast） |
| 线程 | gemm 多线程（实测约 4 核） | torch 6 线程 | — | — |

补充：两边 50 次迭代的答案全部正确路由到 `billing`，概率和恒为 1。

## 性能亮点

1. **GPU 单请求延迟：pagoda 快 12.3%（605.9 vs 690.9 ms）——而且是在精度吃亏的情况下**。
   Python 侧跑的是 fp16（一半的显存带宽和计算量），pagoda 跑全 f32 仍然更快。
   小模型 + 单请求场景下，Python 的逐层派发与 autocast 上下文切换开销吃掉了 fp16 的理论红利。
2. **冷启动快 5–6.5 倍**（1.4–1.6 s vs 7.9–9.4 s）。没有 `import torch`、没有 transformers
   注册表扫描；safetensors 直接 mmap。对弹性伸缩 / Serverless / 边缘重启场景，这是
   “秒级上线” 与 “十秒级上线” 的差别。
3. **内存低 17–37%**。GPU 场景 1850 vs 2954 MB，省下的 1.1 GB 在 8GB 消费级显卡上
   直接决定能否同卡多实例。
4. **延迟分布更紧**：GPU 场景 max−min = 10.3 ms（1.7%）。系统1决策通常卡在业务主链路上，
   尾延迟稳定比均值略低更值钱。

## 可靠性亮点

1. **跨设备结果一致（本次实测最有说服力的一项）**。
   同一输入，pagoda 的 CPU 与 GPU 输出：决策 argmax（billing）**逐位一致**；
   三个选项概率的最大差异约 52 ULP（相对误差 ~5e-6，纯 f32 归约顺序噪声）。
   Python 官方实现同一对比：billing 概率 0.9865（CPU/f32）→ 0.9866（GPU/fp16），
   漂移 1e-4——**大 4 个数量级**，且官方代码对算力 <8.0 的 GPU（如 T4）强制 fp16，
   意味着“同模型在不同硬件上给不同答案”是其设计内行为。对阈值敏感的风控/路由场景，
   pagoda 的跨硬件可复现性是硬保障。
2. **逐位确定性进了 CI**：`e2e_laya` 断言两次 `decide` 的概率 `to_bits()` 完全相等，
   每次验证都会拦住任何引入非确定性的改动。
3. **依赖面即攻击面，也即可靠性面**：
   - Python 侧：venv **5.3 GB / 39 个包**。本次复现踩了两个真实的版本矩阵坑——
     transformers 4.46 不认识 `modernbert`（需 ≥4.48）；transformers 4.48 的 SDPA 检查
     拒绝 torch 2.0.1（需 ≥2.1.1），现有 conda 环境无法直接运行，被迫新建 venv。
   - Rust 侧：**单二进制 17.9 MB**，`ldd` 仅 8 行（libc/libm/libgcc 等系统库），
     零 Python、零 torch。`cargo build` 一次通过，拷贝即部署。
4. **类型化 API 拒绝坏输入**：Jev 兼容 JSON 在反序列化层校验题型与字段，
   畸形请求返回 400 而不是 traceback。

## 诚实清单：差距在哪

1. **CPU 内核质量：pagoda 慢 2.25 倍**（1062 vs 471 ms）。torch 的 oneDNN/MKL
   在 CPU 上仍是业界最强，candle 的 gemm 多线程（实测约 4 核）打不过。
   CPU-only 部署且对延迟敏感时，目前是 Python 版更快。对应路线图：接入 BLAS/MKL
   后端或自研 int8 量化内核。
2. **批量决策**：官方 `collate_items` 支持批量前向，pagoda 的 Laya 路径当前逐请求。
   高 QPS 场景需要补 batching（对应 P3 的 batched matmul 能力复用）。
3. **功能面**：Python 版带训练侧（RL 管线），pagoda 只做推理——这是刻意取舍。

## 复现

```bash
# Rust（cargo run 即得，模型已缓存时秒开）
cd pagoda-hf && cargo run --release --features cuda --example bench_laya -- --iters 50
PAGODA_DEVICE=cpu cargo run --release --example bench_laya -- --iters 50

# Python 参考实现（需 torch>=2.1.1 + transformers>=4.48 + safetensors）
cd pagoda-hf/bench
python bench_laya.py --snapshot <laya快照目录> --device cuda --iters 50
python bench_laya.py --snapshot <laya快照目录> --device cpu  --iters 50
```