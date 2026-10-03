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
（`KvVault`：radix 键控 trie，任意前缀深度命中；Arc 快照零拷贝；条数 + 字节
双封顶 LRU；故障请求永不入库），新请求最长前缀匹配后直接切片嫁接——
prompt 的 prefill **物理上消失**（重复请求 21→16 token，checkpoint 创建
6→1 token，子前缀 prompt 嫁接 4 token；134-token prompt 的 warm prefill 归零）。
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

`docs/guide/` 十二篇中文教程，从"什么是 LLM 推理服务"讲到调度指标、
每篇都有生活化类比、ASCII 图、可运行的动手环节、与 SGLang/vLLM 对照表。

为什么值钱：开源项目的采用率死于"看不懂"。文档即获客。

### 7. 与 SGLang 一键共部署（网关模式）

`pagoda serve --upstream http://host:port` 一条命令变成 SGLang 的前厅网关：
/generate 与 /v1/chat/completions 逐字节透传给 SGLang worker，/health /stats
/checkpoint* 留在本地。一键脚本 `scripts/co-deploy.ps1|.sh` 连 SGLang 的
venv 安装、健康等待、双进程拉起全包。

为什么值钱：不与 SGLang 抢"大模型 GPU 吞吐"的主场，而是站在它前面补齐
它没有的东西——agent checkpoint 原语、嫁接指标、准入护栏、零依赖可嵌入的
控制面。竞合而非竞争：存量 SGLang 部署零改造接入。

### 8. System 1 + System 2：内置 Laya 决策模型

完整移植 Laya（ModernBERT 编码器 + 决策头，candle 实现）：路由/护栏/审核这类
"判断题"不用惊动生成模型——choice/score/noul 三种题型一次前向出校准概率，
e2e 断言 README 账单场景全对且逐 bit 确定。嵌入网关即是分诊台：安全的走
大模型，危险的就地拦截，亚秒级单次开销（实测 RTX 3050 约 606 ms / CPU 约 1.0 s）。

与官方 Python 版同机实测（见 docs/BENCHMARK-LAYA.md）：

- GPU 单请求快 12.3%（606 vs 691 ms），且 pagoda 跑 f32 对 Python 的 fp16；
- 冷启动快 5–6.5 倍（1.4–1.6 s vs 7.9–9.4 s），峰值内存省最多 37%（−1.1 GB）；
- 部署面 17.9 MB 单二进制（ldd 仅系统库）对 5.3 GB / 39 包 venv，约 300 倍；
- 跨设备决策逐位一致（CPU/GPU argmax bit-identical，概率漂移 ~5e-6），
  Python 官方路径 CPU↔GPU 漂移 1e-4 且对旧卡强制 fp16——同模型不同硬件不同答案；
- 诚实差距：CPU 纯算力落后 torch oneDNN 约 2.25 倍，批量决策待补。

为什么值钱：agent 时代的三件刚需（路由、护栏、审核）都是判断不是生成；
用 GPT 级模型干这个又贵又慢还得解析输出。System 1 + System 2 分层是
agent 基础设施的正确形状。

### 9. 一键蒸馏管线：付费教师 → 本地学生（SGLang 部署）

`distill/` 目录是一个端到端可跑的蒸馏框架，业务场景为跨境电商客服工单
结构化（部门/紧急度/情绪/订单号/退款金额/是否人工/回复草稿）：

- **教师零绑定**：OpenAI 兼容接口通吃付费 API（GPT/DeepSeek）、私有部署、
  本地 SGLang——换环境变量就换老师，代码零改动；
- **数据引擎**：24 条手工种子 → 教师改写扩增 ×3 + 标注 → schema 校验
  （不合格丢弃）→ 按种子分组切分防泄漏；
- **LoRA 蒸馏**：Qwen2.5-0.5B，1.75% 参数，completion-only loss，
  消费级 8GB 显卡 8 分钟训完；
- **诚实评估**：逐字段 exact-match 对照教师金标准（实测 department
  15%→65%、urgency 5%→70%、整单全对 0%→20%，瓶颈在数据量有明确修复路径）；
- **一键部署**：合并权重经 SGLang 以 OpenAI 兼容端点上线，可挂 pagoda 网关。

顺手把概念说清（guide 15）：不用 prefill/decode 的判断题根本不需要
SGLang（那是 Laya 的活）；所有自回归小模型都用 prefill/decode——
"小"只省算力，省不了调度，这正是 SGLang 的主场。

实测记录：docs/DISTILL-REPORT.md（含 SGLang JIT 内核需 nvcc≥12.8 等
五条排障实录）。

### 10. 分诊网关：System 1 给 System 2 当门卫（三进程实跑验证）

`pagoda serve --upstream <SGLang> --laya-url <Laya>` 把卖点 8 和 9 串成闭环：
请求进门先过 Laya 三问（部门/流失风险/是否转人工），安全的逐字节转发给
SGLang 上的蒸馏学生正常生成，危险的**门口转人工、0 GPU、响应附
`pagoda_triage` 结构化判定依据**（可审计、阈值可调）。

2026-10-03 三进程全链路实跑（Laya + Qwen2.5-0.5B 学生 + 网关）：

- 威胁工单（"不退款就投诉拒付"）2.9s 转人工：churn_risk 0.9425、
  needs_human 0.6177，student 零消耗；
- 安全工单照常生成，只多一次 Laya 前向的开销；
- fail-open 演练：kill 掉 Laya 后请求照常转发，`/stats` 里
  `triage_unavailable` 计数 +1——分诊台倒了业务不停；
- `--laya-shadow` 影子模式只记录不拦截，新分诊策略灰度上线零风险；
- `/stats` 四计数器（triaged / escalated / unavailable / proxied）开箱可观测。

为什么值钱：这是 agent 基础设施的正确形状——便宜的判断在门口做，
昂贵的生成只为值得的请求烧。关键词过滤做不到校准概率，
让 GPT 级模型自省又贵又慢还要解析输出。

## 诚实边界（避免过度宣传）

- 主 crate 跑的是确定性 n-gram 玩具模型：链路全真，算力是模拟的。
  真实权重对接点在 `pagoda-hf`（HF tokenizer + Candle），需联网编译。
- KV 块里存的是 token id 而非张量：管理语义（分页/引用计数/COW）全真，
  数值计算不接 GPU。P2 起**会话内**是张量级 KV（candle），跨序列的张量共享
  （radix 命中前缀的 KV 嫁接）留待 P3。
- grammar 是字节级：只在字节级 tokenizer 上合法（引擎有显式护栏拒绝误用）。

## 一句话电梯稿

> pagoda = SGLang 的架构思想 × Rust 的安全与零依赖 × agent 集群的原生原语，
> System 1 分诊 + System 2 生成的完整闭环，配一套让小白也能上手的文档。
