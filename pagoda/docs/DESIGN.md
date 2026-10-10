# Pagoda 设计与实现文档

本目录是一个从零实现、**零第三方依赖**的 SGLang（[sgl-project/sglang](https://github.com/sgl-project/sglang)）范式推理服务运行时。它不是把 SGLang 的 Python 代码移植成 Rust，而是提取其核心架构思想，用 Rust 重新实现一个可独立编译、可测试、可运行的参考版本，用以验证"Rust 承载 SGLang 运行时"这一方向的可行性与边界。

## 1. 目标与非目标

**目标（v1，已实现）**

- 以 `std` 实现完整推理生命周期：分词 → 前向 → 采样 → KV 页管理 → 批调度 → 前缀缓存。
- 复刻 SGLang 的三大标志性机制：RadixAttention 前缀缓存、连续批调度、分页 KV 缓存 + 写时复制。
- 提供 SGLang 风格前端语言（`gen` / `select` / `fork`）。
- 提供离线引擎 + 最小 HTTP 服务端 + CLI，全部可在无网环境构建运行。

**非目标（v1，明确不做）**

- 不做 GPU kernel / 高性能 Tensor 内核（主 crate 用确定性 n-gram 玩具模型打通链路）。
- 不做分布式并行、多模态预处理（可对接仓库已有 gap 分析）。受限解码（grammar）已在
  主 crate 用零依赖实现（见 §5.1），不再是非目标。

真实权重与真实分词器**已提供对接点**：姊妹 crate `pagoda-hf` 用 `tokenizers` + Candle
实现同一套 `ModelEngine` / `Tokenizer` trait（`HfTokenizer` + `CandleModel`）。该 crate
需要联网下载依赖与模型权重，未在本离线沙箱内编译验证。

## 2. 架构

```
                 ┌───────────────────────────────────────────┐
   CLI / HTTP ──▶│                     Engine                 │
                 │  ┌─────────────┐   ┌───────────────────┐  │
                 │  │  Scheduler  │──▶│   RadixCache      │  │  prefix cache
                 │  │ (continuous │   │  (RadixAttention) │  │
                 │  │  batching)  │   └───────────────────┘  │
                 │  └──────┬──────┘                          │
                 │         │ token ids                       │
                 │  ┌──────▼──────┐   ┌───────────────────┐  │
                 │  │  PagedKv    │──▶│    Sampler        │  │
                 │  │  Cache      │   │ temp/topk/topp/pen │  │
                 │  └──────▲──────┘   └──────┬────────────┘  │
                 │         │ pages           │ logits        │
                 │  ┌──────┴──────┐   ┌──────▼────────────┐  │
                 │  │ Tokenizer   │   │   ModelEngine     │  │
                 │  │  (trait)    │   │   (trait/ngram)   │  │
                 │  └─────────────┘   └───────────────────┘  │
                 └───────────────────────────────────────────┘
```

模块与文件：

| 模块 | 文件 | 对应 SGLang 概念 |
| --- | --- | --- |
| 数据模型 | `spec.rs` | `srt.sampling_params`, `GenerationOutput` |
| 分词 | `tokenizer.rs` | HuggingFace/GGUF tokenizer 的 trait 抽象 |
| 采样 | `sampler.rs` | `sampling_params` 的 temperature/top-k/top-p/penalty |
| 分页 KV | `kv_cache.rs` | `memory_pool` / paged attention，refcount + COW |
| 前缀缓存 | `radix_cache.rs` | `RadixAttention` 的 radix tree |
| 块级前缀缓存 | `apc.rs` | vLLM Automatic Prefix Caching（链式块哈希） |
| 后端模型 | `model.rs` | `model_runner` / tensor 前向 |
| 引擎/调度 | `engine.rs` | `scheduler` + `manager`（continuous batching） |
| 前端语言 | `dsl.rs` | `sglang.lang` 的 `SglExpr` 编译器前端 |
| 服务 | `server.rs` | `sglang.srt.server`（OpenAI 兼容 + `/generate`） |
| CLI | `src/bin/pagoda.rs` | 命令行演示 |

## 3. 核心机制

### 3.1 Radix 前缀缓存（radix_cache.rs）

每个「已完成的 token 序列」被插入一棵按 token id 键控的 Trie。新请求到达时，`match_prefix` 找到最长已缓存前缀，这部分 prompt 的 prefill 计算被**跳过**（`prefix_hit_tokens` 度量）。树的命中率通过 `num_queries` / `hit_queries` / `hit_tokens` 统计。

v1 的先验：radix 只记录 token 路径并承担「计算跳过」，物理 KV 块共享（跨并发请求零拷贝）作为一个后续里程碑；写时复制原语已在 `kv_cache::fork_block` 就绪并有单测。

### 3.2 分页 KV + 写时复制（kv_cache.rs）

KV 空间被切成固定大小 `block_size` 的物理页（real 实现每页存 `num_layers * 2 * head_dim` 浮点；本参考版存 token id）。块带引用计数，只有引用归零才回收到空闲池。`fork_block` 实现 COW：当写者要扩展一个被共享（`ref_count > 1`）的尾部页时，先克隆成私有页再写，避免污染共享前缀。

### 3.3 连续批调度（engine.rs）

`generate_batch` 一次性接纳所有请求（prefill 在接纳时完成），随后每一 `step` 让**每个活跃序列各产出一个 token**（decode），已结束的序列从本步的 batch 中退出。这即 SGLang「统一 prefill+decode」连续批调度的单步形式。`engine.stats()` 输出 forward 次数、KV 分配/释放次数、radix 命中率等，直接可见批调度效果。

### 3.4 前端语言（dsl.rs）

SGLang 的 `sgl.gen / sgl.select / sgl.fork` 在 Rust 中无动态求值语法糖，故用一个小型算子代数建模，`Engine::run_program` 解释执行：

- `System/User/Assistant`：追加带角色标签的上下文。
- `Gen`：以当前 transcript 为 prompt 生成，并把结果绑定到变量名。
- `Select`：对每个候选做「逐 token log-softmax 平均分」打分，选最高分。
- `Fork`：复制当前上下文/变量到每个分支，并行执行并汇合（SGLang 并发状态机）。

## 4. 数据流（一次 `generate`）

1. `tokenizer.encode(prompt)` → token ids。
2. `radix.record_match(prompt)` → 命中前缀长度 `L`。
3. 把 prompt 按页写入 `PagedKvCache`；估算 prefill 成本 = `prompt_len - L`。
4. 循环直到 `max_tokens`/停止条件：
   - `model.forward(tokens)` → logits；
   - `apply_penalties`（频次/存在惩罚）；
   - `sampler.sample`（temperature/top-k/top-p）→ 新 token；
   - 追加到页面（满页则分配新页）；
   - 检查 EOS / stop_token_ids / stop 字符串。
5. 结束：释放本请求页；把完整 token 序列插入 radix；返回 `GenerationOutput`。

## 5. 已实现 vs SGLang 官方（对照）

| 能力 | 状态 |
| --- | --- |
| 离线批量推理（continuous batching） | ✅ |
| Radix 前缀缓存（compute-skip + 命中统计） | ✅ |
| 分页 KV（分配/回收 + refcount） | ✅ |
| 写时复制（`fork_block` + 单测） | ✅ 原语就绪 |
| 采样（temperature/top-k/top-p/frequency/presence） | ✅ |
| SGLang DSL（gen/select/fork） | ✅ |
| OpenAI 兼容 + 原生 HTTP 端点 | ✅ 最小实现 |
| 真实 tokenizer / 权重后端 | ✅ `pagoda-hf` 适配器（tokenizers + Candle，需联网） |
| **物理 KV 块跨请求共享**（零拷贝复用） | ✅ 已接入 hot path（inc_ref → COW → 释放） |
| chunked prefill / 分页预填 | ✅ 每步 token 预算跨 step 物化 |
| 调度策略（FCFS / longest-prefix / shortest-prompt） | ✅ SchedulePolicy 可配置 |
| KV 淘汰（LRU/eviction） | ✅ evict_lru + evict_on_pressure |
| 请求级 fault isolation / 属性测试 | ✅ logits 校验 + `FinishReason::Fault` + 确定性 property tests |
| 准入控制 / SLO 护栏 | ✅ `max_waiting_requests` + `max_total_tokens` → `FinishReason::Rejected` |
| 约束解码（regex / JSON grammar） | ✅ `grammar.rs`：Thompson NFA byte-regex + JSON pushdown scanner → logit mask |
| APC 块级前缀缓存（vLLM 风格） | ✅ `apc.rs` + `CacheBackend` 可切换，LRU 淘汰，命中统计 |
| KV checkpoint / 分支原语 | ✅ pin 语义 + 保证全命中分支 + HTTP 端点 |
| 分布式并行 / 多模态 | ❌ 见仓库既有 gap 分析 |

### 5.1 约束解码（constrained decoding）

`grammar.rs` 在 `SamplingParams` 上新增可选字段 `grammar: Option<Grammar>`，零依赖实现两种约束：

- `Grammar::Regex(ByteRegex)`：byte 级正则（隐式 `^...$` 全匹配）。手写递归下降解析器把
  模式编译成 Thompson NFA，支持 `| () [] ^ $ - \ 转义 . ? * + {m}` 等；`allowed_bytes(partial)`
  对 NFA 当前状态集做逆向可达性分析，返回可继续走向终态的下一字节集合。
- `Grammar::Json`：pushdown 前缀扫描器（对象/数组/字符串/数字/true/false/null），
  `scan()` 给出 `Complete` / `Prefix(mask)` / `Dead` 三态，边扫描边给出合法的下一字节。

采样前，引擎解码当前已生成输出，用 `allowed_bytes` 得到合法续写，再调用
`mask_logits` 把所有 special token 与非法字节 logit 置为 `-∞`；已处于 complete 状态时
保留 EOS 作为合法终止；`allowed_bytes == None`（已死路）时直接以 `Stop` 收尾，避免采样
出越界 token。sampler 同步做到 `-∞` 感知（greedy argmax 跳过非有限值、温度路径先清空
非有限 logit），因此约束下采样不会选到被 mask 的 token。

grammar 按字节工作，因此**仅对字节级 tokenizer 合法**：`Tokenizer::is_byte_level()`
（默认 `false`，`ByteTokenizer` 返回 `true`）是显式能力标记，引擎在 admission 时
拒绝非字节 tokenizer 上的 grammar 请求（`RejectReason::Unsupported`），避免对 BPE
词表做错误的 logit mask。`server.rs` 的 `/generate` 端点已接受
`sampling_params.grammar = {"type":"json"}` 或 `{"type":"regex","pattern":"..."}`。

### 5.2 APC 块级前缀缓存（apc.rs）

`CacheBackend::Apc` 把前缀缓存从 radix 切换为 vLLM 风格的块级链式哈希表：
每个**完整块**以其全部 token 加前一块哈希做链式哈希（splitmix64 混合），查找是每块
一次哈希探测、淘汰以块为粒度；链式结构保证前缀语义——只有前置上下文完全一致时块
才可复用。与 radix 的取舍：查找 O(块数) 而非 O(token 数)，命中不含尾部不满一块的
部分（块粒度 vs token 粒度）。缓存为每个索引块持有一份基础引用，`evict_lru` 按
`last_used` 时钟淘汰并释放。

引擎通过 `match_prefix` / `publish_prefix` 两个私有方法在两种后端间切换，`/stats`
新增 `apc_blocks` / `apc_hit_tokens`，`compute_saved_tokens()` 对两种后端统一计账。

### 5.3 KV checkpoint（分支 / agent 树原语）

`create_checkpoint(text)` 把一段文本物化进 KV 池、发布进前缀缓存，并保留请求侧引用
作为 pin：这些块从此对缓存淘汰免疫，直到 `drop_checkpoint(id)`。
`generate_from_checkpoint(id, continuation, sampling)` 直接用 checkpoint 存储的物理
槽位作为前缀——不走 radix/APC 查找，保证 100% 前缀命中——把 continuation 接在后面
正常走 admission、prefill、decode 流程。分支追加共享尾块时照旧 COW（checkpoint 块
永不被改写，有 `engine::tests` 单测保证）。

这是 agent 集群 / 树搜索（ToT、MCTS、并行分支采样）场景的核心原语：共享 system
prompt / 共享树干只 prefill 一次，任意多分支零重算接入。HTTP 端点：
`POST /checkpoint`、`POST /checkpoint/generate`、`POST /checkpoint/delete`。

## 6. 关键取舍与差距

1. **玩具模型 vs 真实权重**：`NGramModel` 是对 corpus 训练的 byte n-gram，众数为确定性续写，`temperature>0` 为采样。它用于把整条链路跑通；换真实模型只需实现 `ModelEngine::forward`（可返回 CUDA/Candle/GGML 结果）。
2. **token-only KV vs tensor KV**：`PagedKvCache` 存的是 token id 而非 K/V 张量，因此无需 GPU。页槽位/引用计数/COW 语义与真实实现一致，替换槽内 `u32` 为 `[f32; head_dim]` 即可。
3. **前缀命中 = 计算跳过 + 物理共享**：`match_path` 直接返回 `(block_id, offset)` 槽位，请求在 `admit` 对共享块 `inc_ref`、扩展尾部时 `COW`、完成时 `dec_ref` 释放本请求引用，缓存持有一份基础引用——零拷贝复用已进 hot path。
4. **prefill 已分块**：长 prompt 按 `max_prefill_tokens_per_step` 预算跨 step 物化（`prefill_chunks` 度量），并在 ready/active 两段队列间按 `max_running_requests` 控制解码并发。仍保留的简化是「同一 batch 内的并发请求之间不互相复用前缀」（只复用已完成序列的缓存）。

## 7. Roadmap（下一里程碑，按收入贡献排序）

1. **四轴指标仪表盘（P0）**：把 `EngineStats` / `/stats` 输出重构为「省下的 token / 页 / forward」× 四轴的收入视角。
2. **车间化调度（P0）**：最长公共前缀优先级、KV LRU 淘汰、准入控制与 SLO 护栏。→ ✅ 已完成（`SchedulePolicy` + `evict_lru`/`evict_on_pressure` + `max_waiting_requests`/`max_total_tokens` 准入护栏）。
3. **APC 块级缓存 + KV checkpoint（P0，面向 agent 集群）**：vLLM 风格块级链式哈希缓存与可 pin 的分支原语。→ ✅ 已完成（`CacheBackend::Apc` + `apc.rs` 链式块哈希/LRU 淘汰/命中统计；`create_checkpoint` / `generate_from_checkpoint` / `drop_checkpoint` + `/checkpoint*` HTTP 端点；详见 §5.2 / §5.3）。
4. **真实后端（P1）**：`ModelEngine::forward` 换 Candle / GGML / CUDA 真实权重，度量真实时延与能效。→ ✅ 已完成并于 2026-09-30 在一台联网 Linux 开发机上验证通过：姊妹 crate `pagoda-hf`（`HfTokenizer` + `CandleModel`，candle-llama F32/CPU），e2e 程序 `examples/e2e_tiny_llama.rs` 逐条断言真实分词/权重/前缀缓存/确定性/零故障——实测热缓存 `prefix_hit=6/6`、`faults=0`、logits 为真实非零值（min=-0.323 max=0.322）；运行手册与实测输出见 `docs/P1-VERIFICATION.md`。修复要点：candle `Cache` 在 `use_kv_cache=true` 时会重复追加 K/V 导致 mask broadcast 报错，且 `Llama::forward` 返回 `[b_sz, vocab]` 而非 `[1, seq, vocab]`。
5. **稳定性硬化（P1）**：请求级 fault isolation、调度器/radix 的 property-based 测试，变 Rust 安全势能为可验证 MTBI。→ ✅ 已完成（logits 校验隔离故障请求 + `tests/fault_tests.rs`、`tests/property_tests.rs` 确定性属性测试）。
6. **约束解码与多模态（P2）**：JSON schema / regex grammar 与既有多模态管线对接。
   → 解码部分 ✅ 已完成（零依赖 `grammar.rs` + 引擎 logit mask + server 端点 + 单测/集成测试）；
   多模态对接留待下一里程碑。
7. **增量 KV 会话（P2，性能）**：消灭 O(n²) 全量重放——`ModelSession` trait（`context_len` / `forward(新后缀)` / `fork`）+ 引擎每序列私有会话 + checkpoint 树干 KV 分叉；pagoda-hf 侧每会话一份 Candle KV cache（首调全量预填充、之后逐 token，规避 candle 0.8 mask 广播限制）。→ ✅ 已完成并于 2026-09-30 在联网 Linux 开发机上验证：`tests/session_tests.rs` 6 项（投喂恰好一次/无状态等价/批量隔离/异常回退/checkpoint 分叉/空续写），e2e 实测 16 步生成模型仅吃 21 token（全量重放需 216，省 10.3 倍，墙钟 11ms→6ms），checkpoint 分叉零重算树干。已知边界：跨序列张量级 KV 共享（radix 命中前缀的 KV 嫁接）为 P3；candle mask 限制下续写逐 token 喂入。
8. **GPU 后端（P3）**：→ ✅ 通路已打通（2026-09-30，RTX 3050）：`pagoda-hf --features cuda` + `PAGODA_DEVICE=cuda` + `PAGODA_DTYPE=f16`；e2e 全绿且 F32 logits 与 CPU 逐位一致。实测 GPU F16 warm 9.0s ≈ SGLang GPU 7.9s（共享盒子 launch-latency-bound，详见 docs/BENCHMARK.md）。
9. **批量 matmul（P3）**：→ ✅ 已完成（2026-09-30 验证）：引擎解码阶段把「恰好缺一个 token」的会话聚成一批，一次 `[B,1]` 前向推进（`ModelEngine::session_forward_batch` + `ModelSession::as_any_mut` 下沉钩子；不支持的后端自动回退逐条，语义不变）。pagoda-hf 侧换用自研批量 Llama（`pagoda-hf/src/llama.rs`，约 400 行 candle-nn 原语，与 candle 官方 Llama **逐 bit 一致**：F32/F16 parity 均 0.0e0）——每会话私有 KV、补齐+遮罩拼批、按行 RoPE、原子提交（失败不留半状态，回退安全）。实测：物理调用 6.24× 收缩（256 步/41 次）；tiny 模型 batch-8 吞吐 2218 → 3103 tok/s（+40%）；1.1B 大模型墙钟暂持平（kernel 时间主导）。新指标：`total_decode_steps` / `total_decode_calls` / `decode_batch_factor`（/stats + CLI revenue 行）。测试：核心 `tests/batch_tests.rs` 3 项（调用数收敛、批量=逐条逐 token 相等、空投喂分支不挤占批次）+ e2e `e2e_batched_decode`（对拍/等价/收敛/热缓存复跑）。修复的坑：多 token 后缀喂入非空 cache 的因果遮罩必须按 `[seq, index_pos+seq]` 开窗（candle 逐 token 循环的限制随之解除）；`Tensor::stack` 会新增轴（需先挤掉每会话前导维）。CUDA Graph 已调研（2026-10-02）：candle-core 0.8.4 无 stream capture / 图回放 API，本路线内不可实现；解锁路径：上游 candle 支持 CUDA Graph，或以 cudarc 自研 paged-attention 融合 kernel（等效目标：把整步压成极少次 kernel 发射）。
10. **张量级前缀嫁接（P3，真·RadixAttention）**：→ ✅ 已完成（2026-09-30 验证）：逻辑前缀缓存（radix/APC）之外，把完赛会话的 KV **张量本体**收进跨请求仓库 `KvVault`（`pagoda-hf/src/vault.rs`：键 = token 路径，prompt 前缀 + 已喂入完整路径双键，Arc 快照零拷贝，LRU 上限默认 128 条、`PAGODA_KV_VAULT_ENTRIES` 可调，故障请求不入库）；新请求 admit 时最长前缀匹配并 `narrow` 切片嫁接（`ModelEngine::graft_session` / `offer_session_kv` + `ModelSession::as_any`，上限 prompt_len−1 保证首步照常产出 logits、自动落入批量通路）。因果注意力保证前缀切片与重算逐 bit 相等。实测（tiny 模型）：重复请求 21 → 16 token 投喂，checkpoint 创建 6 → 1 token。测试：核心 `tests/graft_tests.rs` 4 项（嫁接覆盖数、嫁接 vs 不嫁接逐 token 相等、故障 KV 不入库、APC 正交）+ e2e 断言 `model_graft_tokens`；`pagoda-hf` vault 单测 2 项（最长前缀+上限、LRU 淘汰）。稀疏键取舍：只存 prompt 边界与完整路径两处键，任意深度命中留作规模化路线。→ 2026-10-02 已升级：radix 键控 trie（任意深度命中，插入/命中/淘汰 O(路径长)，同路径重复入库替换旧快照不泄漏预算）+ 双封顶（条数 PAGODA_KV_VAULT_ENTRIES 默认 128 + 字节 PAGODA_KV_VAULT_BYTES 默认 2GiB，按 SessionKv 张量实测字节 LRU 淘汰）；e2e 新增 [6/6] 子前缀命中断言（grafted=4，稀疏键时代为 0）；vault 单测 5 项（任意深度/LRU/字节预算/重复入库/淘汰剪枝保共享前缀）。

11. **SGLang 共部署网关（P4）**：→ ✅ 已完成（2026-10-02）：`pagoda serve --upstream http://host:port` 进入代理模式——`/generate` 与 `/v1/chat/completions` 逐字节透传给上游 SGLang worker（零依赖阻塞式 HTTP/1.1 客户端 `http_client.rs`，整段缓冲，上游故障降级 502），控制面（`/health` `/stats` `/checkpoint*`）保持本地；`/stats` 增加 `upstream` 与 `proxied_requests` 可观测字段。一键脚本 `scripts/co-deploy.ps1` / `co-deploy.sh`（构建 → 可选 venv 装 SGLang → 起 worker 等健康 → 起网关等健康）。系统测试 `http_proxy_forwards_generation_and_keeps_control_plane`（mock 上游断言透传逐字节、控制面零转发、计数正确）。2026-10-03 续：SSE 流式透传 ✅——`http_client::request_open`/`UpstreamStream` 支持 chunked 分帧（缓冲路径透明解 chunked），请求体带 `"stream":true` 时代理逐 chunk 中继上游 SSE（`Proxy::forward_streaming`，Content-Type 透传 text/event-stream），上游非流式自动回退缓冲，流式期间不持引擎锁（控制面不被长连接阻塞）；`/stats` 增加 `streamed_requests`。系统测试 `http_proxy_streams_sse_verbatim_and_counts`（mock SSE 上游断言分帧原样、顺序、终止帧、缓冲路径解帧、计数）。2026-10-10 续：多上游前缀感知路由 ✅（P3 仓库内项清零）——`--upstream` 接受逗号分隔列表，`Proxy` 内嵌 `PrefixRouter`：按 prompt 最长前缀亲和选上游（同前缀会话粘住 KV 已热的 worker），`/stats` 增加 `upstream_pool` 与 `upstream_pool_routed`。顺带修掉路由器根节点候选偏置的真 bug：冷请求原先被全部吸到首个热 worker——现在 `deepest==0` 时退回全体轮询，且亲和挑选不再消耗轮询游标（冷路径独立 `rr_cold`）。系统测试 `http_proxy_prefix_pool_sticks_and_spreads`（三 mock 上游断言同前缀粘住、异前缀散开、计数正确）。

12. **Laya 决策模型支持（P4，System 1）**：→ ✅ 已完成（2026-10-02 验证）：`pagoda-hf/src/laya.rs` 完整移植 Laya 推理管线（`rl_agent_api.py` + `rl_common.py` 的 candle 版）——ModernBERT 编码器（candle-transformers 现成，权重名 `encoder.*` → `model.*` 重映射加载）+ 决策头（2 层 norm_first transformer + type_emb + scorer + act_head 手搓，`nn.TransformerEncoderLayer` 语义逐行对齐：in_proj 分体、key_padding_mask、relu FFN）+ `build_sequence` 逐 token 级移植（[CLS] 题型+指令 [SEP] [MASK] 选项… [SEP] state [SEP]）+ 按题型分桶温度校准。三题型 choice/score/noul 的 Jev 兼容答案（choice 标签+分布、score 期望值、noul 概率、confidence=1−归一化熵、act_probability）。e2e `examples/e2e_laya.rs`：README 账单场景断言 department=billing（confidence 0.927）、churn_risk=0.879>0.5、概率和为 1、两次运行逐 bit 一致。架构意义：System 1 分诊台嵌入 pagoda 网关（路由/护栏/审核），与 System 2 生成（SGLang/本地）分层。2026-10-03 续：`src/bin/laya_server.rs` 独立 HTTP 服务（GET /health + POST /decide，Jev 兼容 JSON，坏请求 400），输出与官方 Python API 逐字段对拍一致（billing 0.9865 / conf 0.9267 / churn 0.879 / act 1.0 全同）；一键脚本 `pagoda-hf/scripts/serve-laya.{ps1,sh}`；同机基准 `docs/BENCHMARK-LAYA.md`（GPU 快 12.3%、冷启动 5–6.5×、内存 −37%、17.9MB 单二进制 vs 5.3GB venv、跨设备决策逐位一致；CPU 纯算力落后 oneDNN 2.25× 为诚实差距）。2026-10-03 续：网关按 Laya 判定路由 ✅（见第 13 项之后的分诊网关条目）。待做：multilingual 子目录检查点、批量决策。

13. **一键蒸馏管线（P5，教师→学生）**：→ ✅ 已完成（2026-10-03 端到端实跑）：`distill/` 框架（gen_data.py 教师造数 + train_lora.py LoRA 蒸馏 + eval.py 逐字段评估 + run-distill.sh/ps1 一键编排）。业务场景：跨境电商客服工单结构化（7 字段 JSON + 回复草稿）。教师接口为 OpenAI 兼容 chat——付费 API / 私有部署 / 本地 SGLang 同一代码路径（`TEACHER_BASE_URL/TEACHER_API_KEY/TEACHER_MODEL` 三变量切换）。实跑：本地 Qwen2.5-1.5B 教师（SGLang）→ 91 条校验合格数据 → Qwen2.5-0.5B+LoRA（1.75% 参数，498s）→ department 15%→65%、urgency 5%→70%、整单全对 0%→20% → SGLang 部署 + 未见样本冒烟通过。排障实录五条（SGLang JIT 需 nvcc≥12.8+gcc≥10 等）见 docs/DISTILL-REPORT.md。概念澄清（guide 15）：无 P/D 的判断题不需要 SGLang（Laya 的活）；自回归小模型都有 P/D，SGLang 正为主场。2026-10-03 续：蒸馏学生接入 pagoda 网关路由 ✅（Laya 分诊→学生生成→危险转人工，见下条）。待做：低置信转发教师兜底、数据飞轮（线上低置信样本回流种子库）。

14. **Laya 分诊网关（P5，System 1 路由 System 2）**：→ ✅ 已完成（2026-10-03 三进程实跑验证）：
    `pagoda serve --upstream <SGLang> --laya-url <Laya>` 进入分诊模式——`pagoda/src/triage.rs`
    零依赖 Laya 客户端（复用 `http_client.rs`），每个请求先内置三问（部门 choice +
    churn_risk/needs_human 两个 noul），命中升级条件（needs_human>0.5 ｜ churn>阈值 ｜
    conf<min_confidence）则由网关直接返回 OpenAI 形状的"转人工"回复 + `pagoda_triage`
    结构化扩展（部门/概率/置信度/原因，可审计）；安全请求逐字节转发上游。`--laya-shadow`
    影子模式只记录不拦截；默认 fail-open（Laya 宕机照常转发，`/stats` 计
    `triage_unavailable`），`--laya-required` 可 fail-closed。实跑：威胁工单 2.9s 转人工
    （churn 0.9425 / needs_human 0.6177，学生 0 GPU），安全工单照常生成，kill Laya 后
    请求仍转发。测试：`triage.rs` 5 单测 + 系统测试
    `http_triage_gateway_routes_and_escalates`（mock Laya 按 state 判定 + mock 上游断言
    威胁拦截/安全透传/计数正确）。小白文档：guide/16。SSE 与分诊并存 ✅（分诊先行缓冲判定，放行后才进入流式中继；升级响应始终是缓冲 JSON）。2026-10-03 续：按部门路由 ✅——`pagoda serve --route dept=http://host:port`（可重复），Laya 判定的部门直接选择上游（`Proxy::with_routes`/`pick`），未配置部门的落到默认 `--upstream`；`/stats` 增加 `routed_requests` 与 `routes` 路由表；缓冲与 SSE 流式两条转发路径都按部门选路。系统测试 `http_triage_routes_by_department_to_dedicated_upstream`（双 mock 上游断言 billing 工单进专线上游、未匹配部门走默认、计数正确）。待做：批量分诊、路由维度扩展到置信度/负载。

15. **Mooncake 风格 PD 分离（P6）**：→ ✅ 已完成（2026-10-09 三进程实跑验证）：`pagoda/src/pd.rs`
    移植 Mooncake（Moonshot AI）的 P/D 解耦架构——prefill worker 算完 prompt KV 发布到
    KV 对象池，decode worker 拉取后不重算直接解码。映射：`KvStore` trait = Mooncake Store
    语义（内容哈希键 = prompt token 路径的 FNV-1a，幂等 PUT，字节预算内 LRU 淘汰，对齐
    Mooncake memory pool）；`LocalStore` 同进程实现 + `HttpStore`/`pagoda store` 守护进程
    （`PUT/GET/DELETE /kv/<key>` + `/store/stats`，零依赖 HTTP/1.1，生产环境在 trait 后面
    换 RDMA 传输引擎）；`PrefillBundle` = 传输载荷（`prompt_tokens` 即逻辑 KV 镜像——
    pagoda 参考实现的物理页内容就是 token 路径——外加 `ModelSession::export_kv` /
    `ModelEngine::import_session` 张量 KV 钩子，玩具模型为 null，candle 后端可直接挂上）。
    引擎两半：`prefill_only`（admit→分块物化→推 session→发布 bundle→释放本地页，计
    `pd_prefill_requests`）与 `decode_from_kv`（拉 bundle→本地页 adopt→`drain` 复用连续批
    解码环路，`total_prefill_tokens` 恒 0——prefill 算力全部留在对端）。HTTP 面：
    `POST /prefill`（prefill worker）→ `{kv_key,...}`；`POST /generate {"kv_key":...}`
    （decode worker，miss 404）；`--prefill-url` 让 decode worker 兼任 conductor——纯文本
    `/generate` 自动转发 prefill 再本地解码（Mooncake conductor 角色）；角色纪律：
    prefill 角色拒绝 `/generate`、decode 角色拒绝 `/prefill`、无 store 一律 400。
    CLI：`pagoda store --port 9100` + `pagoda serve --role prefill|decode --store ...`。
    测试：`pd.rs` 3 单测（bundle serde / 内容寻址键 / LRU 淘汰）+ `tests/pd_tests.rs`
    3 项（分离 vs 统一输出逐 token 相等且 decode 侧零 prefill 计费、幂等重发 store_hit、
    HTTP store 往返、三进程 conductor e2e 含角色纪律与 /stats 计数）。小白文档：guide/18。
    2026-10-09 续：三条待做全部清零 ✅——
    ① candle 张量 KV export/import：`SessionKv::export_bytes/import_bytes`（F32 上线，
    F16/BF16 往返逐位一致；载荷含 last-position logits，decode 侧首 token 零重算），
    `CandleCausalLM::cache_export/cache_import` 默认 None 保持其他后端兼容，
    `CandleModel::import_session` 校验长度后重建会话；零权重 Llama 引擎级对拍
    `candle_pd_split_matches_unified`（输出逐 token 相等，decode 侧 tokens_fed 只含
    生成 token）+ `llama::tests::session_kv_bytes_roundtrip`（形状/dtype/logits/坏包）。
    ② 多 decode worker 前缀亲和调度：`pd::PrefixRouter`（字符 trie + 每节点 worker 集合，
    冷路径 round-robin；根节点不打标避免全局磁吸）+ `pagoda route --prefill-url ...
    --decode-url ...`（可重复）——router 即 conductor：/generate 先转 prefill 再按亲和
    选 worker 转发 kv_key，响应逐字节中继（SSE 逐 chunk），`/route/stats` 暴露每 worker
    命中数。③ SSE 流式 PD：`drain` 调度环路加可选 per-token sink（`seq_id + 解码片段`），
    `Engine::decode_from_kv_streaming` + 服务端 `/generate {"kv_key"/conductor, stream:true}`
    逐 token 发 `data:` 帧、末帧带 finish_reason/usage、`[DONE] 收尾；头部延迟到首帧才写，
    kv_miss 等前置错误仍返回普通 JSON。新增测试 4 项（sink 顺序与拼接等价、亲和单测、
    SSE 线上 e2e、router e2e 含粘性命中）。
    2026-10-09 再续：工程化三件套 ✅——
    ④ store TTL：`pagoda store --max-age-secs N`（`StoreCore` 条目带插入时间戳，
    get/put 时惰性过期，字节立即归还预算，`/store/stats` 新增 `expired` 计数）——
    decode 侧宕机没人来取的 bundle 不再永久占用池容量（Mooncake 的 lease/TTL 语义）。
    ⑤ router 分诊门：`pagoda route --laya-url ...`（同 serve 的 shadow/required/
    churn-threshold/min-confidence 四参）——/generate 在**花任何 prefill 算力之前**先过
    Laya System-1，escalate 直接回人工接管 JSON（测试用死掉的 prefill 证明：敌意请求
    200 升级、干净请求 502 prefill_unreachable，门的位置无可辩驳）。
    ⑥ 真权重 serving 二进制：`pagoda-hf/src/bin/serve.rs`（HF tokenizer + Candle Llama
    接进 `run_full`，全套 --role/--store/--prefill-url 参数）——三进程真权重 PD 冒烟
    输出与统一服务**逐字节相等**；实测小模型上 PD 暖路径 57ms vs 统一 29ms（~500KB
    bundle 传输主导，符合 Mooncake 论断：prefill 算力远大于传输时分离才划算），
    数字与复现步骤见 docs/BENCHMARK-PD.md。
    2026-10-09 1.1B 交叉点实测（TinyLlama 真权重，长 prompt 扫频 + 并发隔离实验）：
    ⑦ bundle 线格式 v2：kv 张量字节从 JSON 字节数组（~4x 膨胀、110MB 张量变 ~440MB
    文本、解析数秒）改 base64 字符串（+33%），零依赖编解码器 + 单测——传输瓶颈就此
    消失，1.1B 下单请求 PD 从 128 token 起全程不亏（64_codec_roundtrip）。
    ⑧ conductor 锁范围修复：原实现把 prefill 轮询放在 decode 引擎锁内，排队请求的
    prefill 无法与在途 decode 重叠（738-token prefill 白等 ~37s）；把 prefill 提升到
    锁外（请求体 {text}->{kv_key} 重写后走正常带锁解码路径）后，钉核隔离下并发
    请求 B 延迟 48.4s→38.2s（-21%）；回归测试
    `conductor_prefill_does_not_block_decode_engine`。结论：交叉点不在 prompt
    长度而在并发与资源隔离——prefill 算力跑在 decode 不共享的资源上时 PD 才赢，
    单机 CPU 不分区则保持 unified。

16. **请求级并发调度（P7）**：→ ✅ 已完成（2026-10-09 真权重实测）：引擎装进调度 actor
    （`Engine::into_actor` + `pagoda/src/actor.rs` 通道协议：Generate（带事件回传通道）/
    Stats/Shutdown），HTTP 处理线程只投递请求，一个长跑 continuous-batching 循环统一
    调度——在途请求的分块 prefill 与 decode 在引擎内交错推进，取代原先「每请求一把
    全局引擎锁」的全串行。`drain` 拆出单步 `drain_step`；actor 空闲阻塞 recv，等待
    队列有上限（超限回 QueueFull）。统一角色用 `pagoda serve --concurrent`（PD 角色
    保持 run_full——PD 的并发来自进程分离，不来自进程内重叠）；/stats 报
    `mode: concurrent`；SSE 帧格式与 PD 流式路径一致。1.1B 同会话复测：串行 B 墙钟
    82.8s → 并发 76.5s（仅 -8%：CPU 算力饱和时交错不产出新算力，makespan 反增 9%，
    A 延迟 2.3x 换公平性），PD 钉核 67.5s 仍最优——进程内调度换不来资源隔离
    （数字见 docs/BENCHMARK-PD.md）。测试 `tests/concurrent_tests.rs` 2 项：actor
    并发 4 请求 == 串行逐 token 输出（流式片段拼接 == Done 全文）；HTTP e2e
    3 并发 buffered + 1 SSE 全部对拍串行基线，/stats 并发下可应答。
    2026-10-09 硬化续：① actor 循环改突发排空——每步把入队消息全部 drain 进来，
    突发请求同步入批（原先一步一条，N 并发白等 N-1 个调度步）；② 补齐
    EngineHandle::shutdown（原 Shutdown 消息变体无从发送，属死路径）；③
    concurrent 服务器补齐 /v1/chat/completions（缓冲式，与 run_full 行为一致，
    复用同一 build_chat_request/chat_response）；④ 补测三例：QueueFull 即拒
    （max_waiting_requests=0 确定性触发）、shutdown 后通道收束无 Done、
    chat e2e 对拍串行基线。

17. **服务化硬化包（P7 续，2026-10-10）**：① 请求取消——引擎新增 `abort_seq`
    （从等待队列摘除 + 归还 KV 块 + 按 fault 保守丢弃共享前缀），actor 循环在
    Token 事件发送失败（客户端断连）时即触发中止，不再空烧算力；新指标
    `aborted_requests`（EngineStats + base_stats_fields + CLI revenue 行）。
    ② 分诊 × 并发模式打通——`run_concurrent(engine, addr, triage)` 接受可选
    `Arc<Triage>`，CLI 放行 `--concurrent --laya-url`（原先两参数互斥），并发
    服务器同样先过分诊门再入批。③ checkpoint 入 actor——`generate_from_checkpoint`
    拆出 `admit_from_checkpoint`，ActorMsg 增加 CheckpointCreate / Drop / Generate
    三消息（checkpoint 不存在经 admitted 通道回 404 语义），concurrent 服务器补齐
    `/checkpoint` 三端点（原先仅 run_full 有）；`actor_stream` 改为接收
    `Receiver<StreamEvent>` 由调用方持有。测试 +4（`tests/concurrent_tests.rs`
    现 9 项）：断连中止计数、actor checkpoint 对拍直调引擎、HTTP checkpoint e2e、
    concurrent 分诊门（mock Laya 威胁拦截/安全放行）。

18. **性能三件套（P8，2026-10-10 真权重验证）**：① varlen 注意力——`batch_decode`
    去掉「把每条会话 KV 补齐到批内最长 + 注意力遮罩」，改为逐会话按各自精确长度
    attend（投影/FFN 仍批量）；`forward_batch` 删掉 mask/lmax 参数。1.1B 同会话
    复测：并发模式 B 墙钟 60.7s→48.1s（对串行基线从 -8% 改善到 -21%，补齐浪费
    约占三倍差距）；e2e `e2e_batched_decode` 批量==逐条对拍全过。② KV 线格式
    F16——`KV_WIRE_VERSION_V2=2` 自描述载荷（头部携带 dtype 线编码），
    `KV_WIRE_DTYPE: AtomicU32` + `set_kv_wire_f16`，手写 f32↔f16 位转换零依赖；
    `pagoda serve --kv-dtype f16` 开启；v1 载荷向后兼容（旧版本字节点按 F32 解析）。
    ③ savings 四轴指标——`/stats` 增加 `savings = {tokens, pages, forwards} ×
    {radix, apc, checkpoint, graft}` 交叉表（EngineStats 补 `kv_block_size` 用于
    页换算），CLI revenue 行加 `forwards_saved` / `aborted`。1.1B 同日同 binary
    复测（solo A 27.0s / solo B 43.2s）：串行 B 60.7s；并发+varlen B 48.1s
    （-21%）；PD 钉核 B 47.1s——PD 对 B 的延迟优势被并发+varlen 磨平，PD 剩余
    价值在 A 的延迟（43.8s vs 59.4s：A 的 decode 独占核）。完整表格与解读见
    docs/BENCHMARK-PD.md「2026-10-10 rerun」。

## 8. 验证（v2）

- `cargo test --offline`：120 项全绿（46 库内单测 + 67 集成测试 + 7 系统测试），
  零 rustc warning（仅预编译依赖的良性链接器提示）。
- 系统测试（`tests/system_tests.rs`）：以真实 HTTP server（loopback 端口 + 手写 HTTP/1.1
  客户端）端到端覆盖 `/health`、`/generate`（普通 + grammar 约束）、`/v1/chat/completions`、
  `/stats`、`/checkpoint` 生命周期（创建/分支/404/删除）、404 路由与 admission 拒绝路径。
- APC 集成测试（`tests/apc_tests.rs`）：跨请求全块复用、尾部不满块不复用、首块分叉
  全链失效、压力淘汰持续服务、radix/APC 输出一致性。
- checkpoint 集成测试（`tests/checkpoint_tests.rs`）：分支全前缀命中、pin 抗淘汰、
  drop 释放语义、SLO 护栏、分支间零污染（确定性输出对比）、APC 后端兼容；另有
  `engine::tests` 单测直接断言「分支追加永不改写 checkpoint 块」「分支结束后引用
  计数归零泄漏」。
- KV 会话集成测试（`tests/session_tests.rs`）：每个 token 恰好喂给模型一次
  （录制会话断言投喂序列 `[prompt, 1, 1, …]`）、增量解码与全量重放逐 token 等价、
  批量会话互相隔离、异常会话自动回退无状态、checkpoint 分叉零重算树干、
  空续写复用树干缓存 logits。
- 张量嫁接集成测试（`tests/graft_tests.rs`）：重复请求恰好嫁接 prompt_len−1 个
  prompt token、嫁接与不嫁接输出逐 token 相等、故障请求 KV 绝不入库、
  与 APC 逻辑缓存后端正交共存。
- CLI：`pagoda sample -p "…" --repeat 2` 输出 `revenue:` 行，可见 `compute_saved` / `prefill_skip` / `avg_forward/token` / `kv_util`。
- HTTP：`/stats` 已扩展为四轴收入指标（`compute_saved_tokens`、`prefill_skip_ratio`、`avg_forward_per_output_token`、`kv_utilization` 等）。
- 采样器回归：temperature 路径下被 mask 的 `-inf` logit 必须获得**零概率质量**
  （而非被重置为 0.0 logit 反超合法低分 token）——`temperature_path_gives_masked_logits_zero_mass`
  与 `grammar_constraints_hold_under_temperature_sampling` 两项测试锁死。

## 9. v2 设计优化（对应需求分析）

详细需求与验收标准见 `docs/REQUIREMENTS.md`。本版本新增三项设计能力：

1. **四轴指标（COMPUTE IS REVENUE）**：`EngineStats` 新增 `total_prompt_tokens / total_prefill_tokens / total_output_tokens`，并派生 `compute_saved_tokens()`、`prefill_skip_ratio()`、`avg_forward_per_output_token()`（Token/Watt 代理）、`kv_utilization()`；`/stats` 与 CLI 同步输出。
2. **调度策略（EXTREME CO-DESIGN / AI Factory）**：`EngineConfig.schedule_policy` 支持 `Fcfs` / `LongestPrefix` / `ShortestPrompt`，等待队列按策略出队，默认 `Fcfs` 保持向后兼容。
3. **KV LRU 淘汰（USEFUL LIFE）**：`RadixCache` 节点记录 `last_used` 访问时钟，`evict_lru()` 淘汰最久未用的叶子前缀并释放缓存持有的块；`EngineConfig.evict_on_pressure` 在 KV 池耗尽时自动触发，替代直接失败。
4. **约束解码（CONSTRAINED DECODING）**：`SamplingParams.grammar` 挂入 `Grammar`（byte-regex / JSON），采样前用 `allowed_bytes` + `mask_logits` 屏蔽非法 token；sampler 对 `-∞` 感知，constraint 下 greedy/采样均保证续写合法。

## 10. Laya 专科化微调管线（distill/train_laya.py）

- **训练/推理布局不变量**：训练端逐 token 复刻 Rust 端 `build_sequence`
  （[CLS]+题型头+[SEP]+[MASK]选项+[SEP]+state+[SEP]，选项标记位即打分行），
  保证"训练考的卷子"与"上线考的卷子"完全一致。
- **轻量微调**：LoRA(r=16) 只挂编码器 Wqkv/Wo/Wi（7.2M，1.8%）+ 决策头全量
  （26.2M）+ act_head 冻结；梯度检查点 + expandable_segments 使 8GB 共享卡可行。
- **温度重标定**：留出集上按 (题型, 选项数桶) 网格搜索 NLL 最优温度，写回
  `rl_agent_config.json` 的 `temperature_by_options`——置信度成为可用阈值。
- **即训即上线**：导出键布局与官方检查点逐字节兼容（合并 LoRA 后存 f16
  safetensors），`laya_server --model-dir` 直接加载，服务端零 Python。
- 实测（40 篇 T2DM 摘要，含 12 篇边界案例）：纳入判定 67.5%→97.5%、
  边界 50%→100%、设计分类 85%→100%；训练 27 分钟（RTX 3050 共享卡）。
  小白文档见 `docs/guide/17-laya-finetune-screening.md`。
