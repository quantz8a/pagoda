# Pagoda 卖点（巧思在哪里）

> 一句话定位：**pagoda 是唯一一个"零依赖、可离线、可一页纸看懂"的 SGLang 范式
> Rust 推理运行时——并且为 agent 集群场景造了别人没有的原语。**

## 六大卖点

### 1. 零第三方依赖，离线可编译可测试

整个运行时只用 Rust 标准库：`cargo test --offline` 一次跑完 72 项测试。
没有 tokenizers、没有 candle、没有 CUDA——CI、内网、涉密环境直接可用。
这不是玩具妥协，而是**架构证明**：SGLang 的调度/缓存体系不依赖任何重型生态。

为什么值钱：企业内网和金融/政务环境拉不动依赖链；Rust 重写派（如 SGLang
官方 sgl-router 只做了路由层）没有一个做到"全链路零依赖"。

### 2. 双前缀缓存后端：Radix 与 APC 同框架对比

SGLang 选了 radix tree，vLLM 选了块级哈希（APC）——社区争论多年没有同条件对比。
pagoda 把两种实现做进同一引擎，`EngineConfig.cache_backend` 一行切换，
共享同一套 KV 池、同一套指标（`compute_saved_tokens` 统一计账），
可以直接 A/B：`tests/apc_tests.rs` 里甚至锁死了"两种后端输出逐字节一致"。

为什么值钱：这是**研究和选型的实验台**。别人要搭两套系统才能比，我们切一个枚举。

### 3. KV Checkpoint：agent 集群的第一类原语（独有）

`create_checkpoint` 把共享树干（system prompt / 对话历史 / 工具说明）物化并**钉住**，
`generate_from_checkpoint` 让任意多分支以 100% 前缀命中接入——免疫 LRU 淘汰、
追加写时复制、引用计数零泄漏有单测锁死。

为什么值钱：Manus 类自主 agent 集群、Tree-of-Thoughts、MCTS 并行探索的共同点
是"一棵共享树干 + 大量分支"。Radix/APC 是被动缓存（碰巧命中才省），checkpoint
是**主动契约**（保证命中、保证驻留）。vLLM 和 SGLang 目前都没有这个原语。

P2 起 checkpoint 更进一步：`ModelSession::fork()` 让分支直接**复印树干的
张量级 KV 快照**（candle 张量 Arc 共享、追加不可变，零拷贝分叉）——
树干只预填充一次，百条分支各自只为自己的续写付费（e2e 断言锁死）。

### 3.5 增量 KV 会话：每个 token 只算一次

`ModelSession` 把"无状态全量重放"升级为"每序列私有 KV 会话"：prompt 预填充
一次，之后每步只喂新 token，16 步生成实测省 10.3 倍模型计算（21 vs 216 token）。
契约极小（`context_len` / `forward(新后缀)` / `fork`），无状态后端零改动兼容；
会话异常自动回退重放，正确性永远优先。

### 3.6 批量解码：一次前向养活整个批次

引擎把「恰好缺一个 token」的会话拼成 `[B,1]` 一次前向（不支持的后端自动回退，
语义不变）；pagoda-hf 内置自研批量 Llama（candle-nn 原语，与官方实现对拍到
**逐 bit 一致**），私有 KV 补齐+遮罩拼批、原子提交（失败可安全回退）。
实测物理调用收缩 6.24×，框架开销场景吞吐 +40%；`decode_batch_factor` 进
/stats 与 CLI，收益可观测、可断言。CUDA Graph 与融合 kernel 是下一跳。

### 3.7 张量级前缀嫁接：真·RadixAttention

逻辑前缀缓存只记"算过"；pagoda 把完赛会话的 **KV 张量本体**收进跨请求仓库
（`KvVault`，Arc 快照零拷贝、LRU 封顶、故障请求永不入库），新请求最长前缀
匹配后直接切片嫁接——prompt 的 prefill **物理上消失**（重复请求 21→16 token，
checkpoint 创建 6→1 token；134-token prompt 的 warm prefill 归零）。
与逻辑缓存（radix/APC）正交、与批量解码自然复合（嫁接上限刻意留 1 个待喂
token，首步自动落入批量通路）。输出与冷跑逐 token 相等是硬断言。

为什么值钱：agent 集群的系统提示动辄几千 token，每个 worker 都重算一遍是
纯烧钱；嫁接让"共享系统提示"从逻辑省算力变成物理零 prefill。

### 4. 零依赖约束解码

regex（Thompson NFA）+ JSON（下推扫描器）两套 grammar 编译器，纯手写无依赖，
接入 logit mask；sampler 对 `-inf` 完全感知（包括 temperature 路径——有回归测试）。

为什么值钱：Outlines/xgrammar 都是重依赖。嵌入式、边缘、教学场景需要能
`cargo build` 就出来的约束解码。

### 5. 收入视角的指标体系

`/stats` 不只报延迟吞吐，直接报 **compute_saved_tokens / prefill_skip_ratio /
avg_forward_per_output_token / kv_utilization**——"省了多少算力"是一等公民。

为什么值钱：COMPUTE IS REVENUE。给老板汇报时，`compute_saved_tokens` 就是钱。

### 6. 小白文档体系

`docs/guide/` 七篇中文教程，从"什么是 LLM 推理服务"讲到调度指标，
每篇都有生活化类比、ASCII 图、可运行的动手环节、与 SGLang/vLLM 对照表。

为什么值钱：开源项目的采用率死于"看不懂"。文档即获客。

## 诚实边界（避免过度宣传）

- 主 crate 跑的是确定性 n-gram 玩具模型：链路全真，算力是模拟的。
  真实权重对接点在 `pagoda-hf`（HF tokenizer + Candle），需联网编译。
- KV 块里存的是 token id 而非张量：管理语义（分页/引用计数/COW）全真，
  数值计算不接 GPU。P2 起**会话内**是张量级 KV（candle），跨序列的张量共享
  （radix 命中前缀的 KV 嫁接）留待 P3。
- grammar 是字节级：只在字节级 tokenizer 上合法（引擎有显式护栏拒绝误用）。

## 一句话电梯稿

> pagoda = SGLang 的架构思想 × Rust 的安全与零依赖 × agent 集群的原生原语，
> 配一套让小白也能上手的文档。
