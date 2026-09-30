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

## 8. 验证（v2）

- `cargo test --offline`：78 项全绿（35 库内单测 + 41 集成测试 + 2 系统测试），
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

