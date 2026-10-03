// Pagoda 站点 · 统一可搜索知识索引
// 每个对象: id / bucket(过滤桶) / title / en(英文) / minutes(阅读时长) /
// blurb(短摘要) / body(参与全文检索的长文本) / keywords[] /
// points[](要点) / cmd(可运行命令) / note(一句话说清有什么用)
window.DOC_INDEX = [
  // ============ 文档系列（docs/guide 00–13） ============
  { id:"guide-00", bucket:"docs", title:"00 · 一键上手", en:"Quickstart", minutes:"5 min",
    blurb:"一条脚本把构建、测试、demo 跑完。",
    body:"build、87 个测试、离线生成、DSL 示例、HTTP 服务都在 quickstart 里。建好即健康检查。",
    keywords:["quickstart","快速开始","离线","cargo test","HTTP","入门"],
    points:["一条脚本：build + 87 tests + demo","cargo test --offline 全绿就说明环境 OK","HTTP 端点：/health /generate /stats /checkpoint"],
    cmd:'powershell -ExecutionPolicy Bypass -File scripts\\quickstart.ps1',
    note:"先跑一遍，后面所有文档的“动手试试”都能复现。" },

  { id:"guide-01", bucket:"docs", title:"01 · LLM 推理服务是什么", en:"LLM serving basics", minutes:"10 min",
    blurb:"prefill / decode 两阶段，先搞懂为什么费钱。",
    body:"prefill 一次读入全部 prompt 算所有位置的 attention；decode 每步只算一个新 token。长上下文、高并发会把算力和显存放大，后面所有优化都是在压这两个数。",
    keywords:["prefill","decode","推理","attention","服务化","烧钱"],
    points:["prefill 算全部，decode 算一个","KV cache 避免重复计算","长文本 + 高并发 = 双重放大"],
    note:"不懂两阶段，后面每个优化都看不明白。" },

  { id:"guide-02", bucket:"docs", title:"02 · KV Cache 与分页内存", en:"Paged KV cache", minutes:"10 min",
    blurb:"PagedAttention、引用计数、写时复制。",
    body:"每个 token 的 key/value 要缓存下来复用，不然 decode 会重算历史。分页把 KV 切成固定 page，避免碎片；引用计数管共享块的生命周期；分叉时写时复制。",
    keywords:["kv cache","paged","pagedattention","引用计数","refcount","copy on write","cow","内存池"],
    points:["KV 是每个 token 的 key/value 投影","paged 分块消除显存碎片","refcount + COW 支撑共享和分叉"],
    note:"内存利用率直接决定一个 GPU 能接多少并发。" },

  { id:"guide-03", bucket:"docs", title:"03 · Radix 前缀缓存", en:"RadixAttention", minutes:"12 min",
    blurb:"记住前缀等于少算：trie + 最长前缀匹配。",
    body:"SGLang 的标志机制。跑完的 token 序列按 id 插进 trie，新请求找最长已缓存前缀，命中的部分跳过 prefill。radix_cache.rs 里实现，输出 hit_tokens / compute_saved_tokens。",
    keywords:["radix","radixattention","前缀缓存","trie","最长前缀","compute_saved_tokens","hit"],
    points:["trie 存 token 路径","最长前缀命中，跳过 prefill","命中率和省算力都能看"],
    cmd:'cargo run --offline --bin pagoda -- sample -p "SGLang is a serving framework" --max-tokens 40 --repeat 3',
    note:"共享 system prompt 是 agent 工作负载的常态，这块省的是实打实的计算。" },

  { id:"guide-04", bucket:"docs", title:"04 · APC 块级哈希缓存", en:"vLLM-style APC", minutes:"10 min",
    blurb:"vLLM 的块级哈希，和 radix 放进同一个 A/B。",
    body:"vLLM 用块级哈希做 Automatic Prefix Caching。Pagoda 把 radix 和 APC 做成两个后端，EngineConfig.cache_backend 一行切换，共用同一套 KV 池和指标，测试锁死两个后端输出一致。",
    keywords:["apc","automatic prefix caching","vllm","块哈希","链式哈希","cache_backend","ab测试"],
    points:["链式块哈希","同一引擎切 radix / apc","同条件 A/B"],
    note:"要对比两种缓存策略，不用搭两套系统。" },

  { id:"guide-05", bucket:"docs", title:"05 · Checkpoint 与分支", en:"KV checkpoint", minutes:"12 min",
    blurb:"钉住共享前缀，开分支 100% 命中。",
    body:"create_checkpoint 把 system prompt / 历史 / 工具说明物化并钉住；generate_from_checkpoint 开任意多分支、全前缀命中，不会被 LRU 挤掉。radix/APC 是被动缓存，这是主动契约。",
    keywords:["checkpoint","分支","branch","pin","树干","agent","树搜索","mcts","100%命中"],
    points:["pin 共享前缀，保证驻留","分支追加写时复制","引用计数零泄漏有单测"],
    note:"agent 集群 / ToT / MCTS 的共同形状是一棵树；SGLang 和 vLLM 都没这个原语。" },

  { id:"guide-06", bucket:"docs", title:"06 · 约束解码", en:"Constrained decoding", minutes:"10 min",
    blurb:"让模型只输出合法 JSON / 正则，零依赖。",
    body:"regex（Thompson NFA）+ JSON（下推扫描器）两套 grammar 手写，接 sampler 的 logit mask，-inf 处理正确。输出永远合法。",
    keywords:["grammar","constraint","regex","json","约束解码","nfa","logit mask","结构化输出"],
    points:["regex 用 Thompson NFA","JSON 用下推扫描器","logit mask + -inf 感知"],
    note:"不依赖 Outlines/xgrammar，越狱边缘场景也能做结构化输出。" },

  { id:"guide-07", bucket:"docs", title:"07 · 调度与指标", en:"Scheduling & metrics", minutes:"12 min",
    blurb:"continuous batching、chunked prefill、四轴指标。",
    body:"连续批调度把“来了就补位”做成队列；chunked prefill 限制单步体量；等待队列支持 FCFS / 最长前缀 / 最短 prompt；KV 满时 LRU 自动淘汰。指标直接给 compute_saved_tokens 等四项。",
    keywords:["continuous batching","chunked prefill","scheduler","fcfs","longestprefix","shortestprompt","lru","metrics","指标"],
    points:["continuous batching + chunked prefill","三种出队策略 + LRU 淘汰","四轴指标：省算力、prefill 跳过、forward/输出、kv 利用率"],
    note:"省了多少算力直接进 /stats，不用自己从日志反推。" },

  { id:"guide-08", bucket:"docs", title:"08 · 增量 KV 会话", en:"KV session", minutes:"10 min",
    blurb:"每个 token 只算一次：O(n²) 全量重放 → O(n) 会话。",
    body:"ModelSession 把“每步全量重放”升级成“每序列私有 KV 会话”：prompt 预填充一次，之后每步只喂新 token。16 步从 216 个降到 21 个。异常就回退到全量重放，正确性优先。",
    keywords:["session","kv 会话","增量解码","增量","每 token 只算一次","on squared"],
    points:["私有 KV 会话","16 步 216→21 token","异常自动回退"],
    note:"O(n²)→O(n)，把无谓的重复前向直接删掉。" },

  { id:"guide-09", bucket:"docs", title:"09 · 开源协议：Apache-2.0", en:"License", minutes:"5 min",
    blurb:"能商用、能改、能闭源，SGLang 署名保留。",
    body:"Apache-2.0：可商用、可闭源、可修改分发；NOTICE 永久记录 SGLang 的血缘与署名。和 SGLang 同协议，代码可双向流动。",
    keywords:["license","apache","apache-2.0","开源","协议","署名","notice","商用"],
    points:["Apache-2.0，可商用可闭源","NOTICE 保留 SGLang 署名","clean-room 重写，不是 fork"] },

  { id:"guide-10", bucket:"docs", title:"10 · 批量解码", en:"Batched decode", minutes:"10 min",
    blurb:"一次前向喂整个批次，权重只读一遍。",
    body:"把“正好缺一个 token”的会话拼成 [B,1] 一次前向，权重只读一遍；不支持的后端自动回退。实测调用次数降 6.24×，框架开销场景吞吐 +40%，decode_batch_factor 进 /stats。",
    keywords:["batch","batched decode","批量解码","[B,1]","decode_batch_factor","一次前向"],
    points:["[B,1] 拼批一次前向","6.24× 调用收缩，吞吐 +40%","decode_batch_factor 可观测"],
    note:"权重在批内是共享的，读一次喂一堆。" },

  { id:"guide-11", bucket:"docs", title:"11 · 张量级前缀嫁接", en:"Tensor KV grafting", minutes:"12 min",
    blurb:"真·RadixAttention：KV 本体跨请求复用，prompt 物理零重算。",
    body:"逻辑缓存只记“算过”；Pagoda 把跑完会话的 KV 张量本体收进 KvVault（radix 键控 trie、Arc 零拷贝、双封顶 LRU、故障不入库），新请求按最长前缀切片嫁接。134-token warm prefill 归零。",
    keywords:["graft","嫁接","kv vault","张量复用","零拷贝","零 prefill","true radixattention","arc"],
    points:["KV 张量本体跨请求复用","物理零 prefill（134-token 归零）","和逻辑缓存正交，和批量解码叠加"],
    note:"几千 token 的系统提示，不用每个 worker 都重算一遍。" },

  { id:"guide-12", bucket:"docs", title:"12 · 一键共部署网关", en:"Co-deploy with SGLang", minutes:"8 min",
    blurb:"pagoda 守门，引擎干活，一条脚本起两个进程。",
    body:"一条命令双进程：pagoda 网关保留 /health /stats /checkpoint*，/generate 和 /v1/chat/* 原样透传给 SGLang worker。上游挂了回 502，网关不崩。现有部署不用改。",
    keywords:["co-deploy","共部署","gateway","网关","upstream","代理","透传","sglang worker","proxy"],
    points:["一条脚本拉起双进程","控制面本地，生成透传","上游挂了回干净的 502"],
    note:"不抢 SGLang 的 GPU 主战场，只在前面补它没有的控制面。" },

  { id:"guide-13", bucket:"laya", title:"13 · Laya：只判断不生成", en:"Laya System-1", minutes:"12 min",
    blurb:"决策模型：choice / score / noul 三种题型，一次前向出校准概率。",
    body:"Laya 不吐 token。ModernBERT 编码器 + 决策头，一次非自回归前向，对“选一 / 打分 / 是否”给校准过的概率。账单场景：department=billing、confidence 0.927；churn_risk=0.879。",
    keywords:["laya","decision model","决策模型","system 1","choice","score","noul","modernbert","校准","概率","分诊"],
    points:["choice / score / noul 三题型","一次前向出校准概率","路由 / 护栏 / 审核 = 判断，不是生成"],
    cmd:'cargo run --release --example e2e_laya',
    note:"用生成模型做判断又贵又慢还得解析输出；Laya 几十毫秒 CPU 直接给概率。" },

  // ============ 核心机制（概念卡片） ============
  { id:"concept-radix", bucket:"concept", title:"RadixAttention", en:"Prefix cache", minutes:"",
    blurb:"前缀缓存：trie + 最长前缀命中，跳过 prefill。",
    body:"在 radix_cache.rs。跑完的序列插进 token-trie，新请求匹配最长前缀，命中部分直接跳过计算，输出 hit_tokens / compute_saved_tokens。",
    keywords:["radixattention","radix","前缀缓存","trie"] },

  { id:"concept-paged", bucket:"concept", title:"Paged KV Cache", en:"PagedAttention", minutes:"",
    blurb:"固定 page 池 + 引用计数 + COW，没碎片。",
    body:"在 kv_cache.rs（先存 token id 示范语义）。分块分配、引用计数、写时复制；P2 起会话内是张量级 KV。",
    keywords:["paged","kv cache","pagedattention","引用计数","cow","内存池"] },

  { id:"concept-checkpoint", bucket:"concept", title:"KV Checkpoint", en:"Agent tree primitive", minutes:"",
    blurb:"钉住共享树干，开分支 100% 命中。",
    body:"create_checkpoint 物化并钉住共享前缀，generate_from_checkpoint 开分支全命中。追加 COW、引用计数零泄漏。和被动缓存不同，这是主动契约。",
    keywords:["checkpoint","分支","树干","agent","pin","树搜索"] },

  { id:"concept-graft", bucket:"concept", title:"张量级前缀嫁接", en:"Tensor KV grafting", minutes:"",
    blurb:"KV 张量本体跨请求零拷贝复用。",
    body:"KvVault 收跑完会话的 KV 张量（radix 键控 trie、Arc 快照、双封顶 LRU、故障不入库），新请求最长前缀切片嫁接。和逻辑缓存正交。",
    keywords:["嫁接","graft","kv vault","零拷贝","零 prefill","张量"] },

  { id:"concept-batch", bucket:"concept", title:"批量解码", en:"Batched decode", minutes:"",
    blurb:"同形状会话拼 [B,1]，一次前向。",
    body:"把正好缺一个 token 的会话拼批，权重只读一遍；6.24× 调用收缩；decode_batch_factor 进 /stats。",
    keywords:["批量解码","[B,1]","batch","decode_batch_factor"] },

  { id:"concept-session", bucket:"concept", title:"增量 KV 会话", en:"KV session", minutes:"",
    blurb:"每序列私有 KV 会话，每个 token 只算一次。",
    body:"prompt 预填充一次，之后每步只喂新 token；16 步 216→21 token；异常自动回退全量重放。",
    keywords:["session","增量","kv 会话","每 token 只算一次"] },

  { id:"concept-grammar", bucket:"concept", title:"零依赖约束解码", en:"Grammar", minutes:"",
    blurb:"regex + JSON 编译器，logit mask，结构化输出。",
    body:"Thompson NFA（regex）+ 下推扫描器（JSON），接 sampler，-inf 正确处理。输出永远合法。",
    keywords:["grammar","regex","json","约束解码","logit mask","结构化"] },

  { id:"concept-scheduler", bucket:"concept", title:"连续批调度", en:"Continuous batching", minutes:"",
    blurb:"chunked prefill + 等待队列 + LRU。",
    body:"engine.rs 的 scheduler：FCFS / LongestPrefix / ShortestPrompt 三策略；KV 池压满自动 LRU。",
    keywords:["continuous batching","chunked prefill","scheduler","lru"] },

  // ============ 指标 ============
  { id:"metric-1", bucket:"metrics", title:"compute_saved_tokens", en:"Saved compute", minutes:"",
    blurb:"省下多少算力 token。",
    body:"前缀缓存 / 嫁接 / 批量解码一起省下的算力总量，进 /stats 和 CLI 的 revenue: 行。",
    keywords:["compute_saved_tokens","指标","省算力","metrics"] },

  { id:"metric-2", bucket:"metrics", title:"prefill_skip_ratio", en:"Prefill skip", minutes:"",
    blurb:"prefill 被跳掉的 prompt token 占比。",
    body:"衡量前缀缓存对 prefill 的削减程度。",
    keywords:["prefill_skip_ratio","指标","prefill","metrics"] },

  { id:"metric-3", bucket:"metrics", title:"avg_forward_per_output_token", en:"Token/Watt proxy", minutes:"",
    blurb:"每个输出 token 的平均前向次数，越低越好。",
    body:"把全量重放白算的那部分暴露出来。",
    keywords:["avg_forward_per_output_token","指标","token/watt","metrics"] },

  { id:"metric-4", bucket:"metrics", title:"kv_utilization", en:"KV utilization", minutes:"",
    blurb:"KV 池占用率。",
    body:"看显存有没有被用好，顺手指导要不要扩容、何时触 LRU。",
    keywords:["kv_utilization","指标","显存","利用率","metrics"] },

  // ============ SGLang / sglang-rust 关联 ============
  { id:"rel-sglang-py", bucket:"sglang", title:"SGLang（上游 · Python）", en:"The origin", minutes:"",
    blurb:"思路来源：radix、连续批调度、分页 KV 都从这来。",
    body:"sgl-project/sglang，高性能 LLM 服务框架。Pagoda 不是它的 fork，是照着核心思路用 Rust 干净重写。同 Apache-2.0，署名在 NOTICE。",
    keywords:["sglang","上游","python","radixattention","gpu","推理引擎"] },

  { id:"rel-compat", bucket:"sglang", title:"兼容矩阵", en:"Perfect fusion", minutes:"",
    blurb:"API 直通 + DSL 直通 + 架构映射。",
    body:"base_url 指到 pagoda 就能用；/generate、/v1/chat/completions 同形；gen/select/fork 同构；模块一一对应。",
    keywords:["兼容","api","dsl","openai","base_url","映射","fusion"] },

  { id:"rel-gateway", bucket:"sglang", title:"共部署网关模式", en:"Co-deploy gateway", minutes:"",
    blurb:"pagoda serve --upstream，把生成透传给 SGLang。",
    body:"/generate、/v1/chat/* 透传，本地留 /health /stats /checkpoint*。上游挂回 502，网关不崩。",
    keywords:["gateway","网关","upstream","co-deploy","透传","控制面"] },

  { id:"rel-sglang-rust", bucket:"sglang", title:"sglang-rust（上游 Rust 迁移）", en:"Upstream Rust line", minutes:"",
    blurb:"上游两条 Rust 线：rust/sglang-mm + rust/sglang-server。",
    body:"上游在推进 Rust：sglang-mm（多模态预处理 pyo3）和 sglang-server（服务端/路由）。Pagoda 是独立 clean-room 的另一条线，重在零依赖运行时 + agent 原语。",
    keywords:["sglang-rust","rust","pyo3","sglang-mm","sglang-server","迁移","路由"] },

  { id:"rel-sglang-mm", bucket:"sglang", title:"rust/sglang-mm", en:"Multimodal in Rust", minutes:"",
    blurb:"多模态预处理的 pyo3 扩展，按 family 移植。",
    body:"MmFamilyProcessor + pipeline 分发 + common/{resize,decode,token_layout}；InternVL 是首个 family。编译闭环还缺 registry / driver。",
    keywords:["sglang-mm","multimodal","pyo3","internvl","resize","decode","token_layout"] },

  { id:"rel-sglang-server", bucket:"sglang", title:"rust/sglang-server", en:"Rust server line", minutes:"",
    blurb:"Rust 服务端 / 路由（数据面直连 DP）。",
    body:"和官方 sglang-router 同思路：Rust 控制面 + Python worker。目标是 scheduler / api server / prefix tree 的 Rust 重写。",
    keywords:["sglang-server","router","gRPC","api server","prefix tree","数据面"] },

  { id:"rel-router", bucket:"sglang", title:"官方 sglang-router 对照", en:"Router comparison", minutes:"",
    blurb:"思路同源，差在控制面厚一层。",
    body:"官方 sglang-router 做路由；pagoda 网关不止路由——checkpoint 生命周期、嫁接指标、准入护栏、约束解码预检都在网关上，且零依赖可离线嵌入。",
    keywords:["router","sglang-router","控制面","网关对照"] },

  // ============ Laya（System 1）关联 ============
  { id:"rel-laya", bucket:"laya", title:"Laya：System 1 决策模型", en:"Laya decision model", minutes:"",
    blurb:"ModernBERT 编码器 + 决策头，一次前向做判断。",
    body:"移植自 convaiinnovations/laya（PyTorch → candle）。三题型 choice/score/noul，输出校准概率 + confidence + act_probability。",
    keywords:["laya","decision","system 1","modernbert","decision head","candle"] },

  { id:"rel-laya-q", bucket:"laya", title:"三种题型 choice / score / noul", en:"Typed questions", minutes:"",
    blurb:"选一个、打分、是否，三种判断。",
    body:"choice=N 选一带分布；score=有序打分输出期望档位；noul=是否输出成立概率。build_sequence 逐 token 拼 [CLS]/[MASK]/[SEP]。",
    keywords:["choice","score","noul","题型","概率","置信度"] },

  { id:"rel-laya-calib", bucket:"laya", title:"RLCD 温度校准", en:"Calibrated probabilities", minutes:"",
    blurb:"说 80% 的事真的 80% 发生。",
    body:"严格适当评分规则训练，只有报真实概率才拿满分，再加事后温度拟合和按题型分桶。0.879 可以当真用。",
    keywords:["rlcd","校准","proper scoring","温度","概率"] },

  { id:"rel-laya-sys12", bucket:"laya", title:"System 1 + System 2 分层", en:"Two-system architecture", minutes:"",
    blurb:"分诊台 + 主治医生。",
    body:"Laya（System 1）几十毫秒 CPU 做路由/护栏/审核；生成模型（System 2）只干重活。安全的走大模型，危险的就地拦。",
    keywords:["system 1","system 2","分诊","护栏","审核","路由"] },

  { id:"rel-laya-decide", bucket:"laya", title:"/decide 端点（路线图）", en:"Decide endpoint", minutes:"",
    blurb:"把 Laya 判定暴露成网关 HTTP 原语。",
    body:"现在 e2e_laya.rs 能端到端跑通；/decide HTTP 端点、网关按 Laya 判定路由、multilingual 检查点还在路线图上。",
    keywords:["decide","endpoint","路线图","multilingual","路由"] },

  // ============ 模型（agent 常用大模型） ============
  { id:"model-qwen", bucket:"model", title:"Qwen3（agent 常用大模型）", en:"Qwen3", minutes:"",
    blurb:"工具调用和多步任务最常用的大模型之一，跟 pagoda 有两条接法。",
    body:"Qwen 是 agent 场景用得最多的大模型之一：function calling / 工具调用、多语言、多步任务都稳。在 pagoda 里有两条接法：一是 SGLang co-deploy 透传（SGLang 里加载 Qwen3，pagoda 只做控制面）；二是把 Candle 后端从 Llama 扩到 Qwen（同为 RMSNorm + GQA + RoPE）。",
    keywords:["qwen","qwen3","通义千问","agent","function calling","tool use","工具调用"] },

  { id:"model-backend", bucket:"model", title:"模型后端：Llama → Qwen", en:"Model backend", minutes:"",
    blurb:"Candle 后端现在接 Llama-family，Qwen 是同构的下一批。",
    body:"pagoda-hf 的 CandleModel 目前加载 Llama-family 的 config.json + safetensors。Qwen 同属 decoder-only（RMSNorm、GQA、RoPE），把权重名和 RoPE 参数对齐后就能复用同一套 batch / 会话 / 嫁接机制。",
    keywords:["model backend","candle","llama","qwen","decoder-only","rmsnorm","gqa","rope"] },

  { id:"model-deepseek", bucket:"model", title:"DeepSeek（推理型 agent）", en:"DeepSeek", minutes:"",
    blurb:"长链推理 + 工具调用是另一类常见选择。",
    body:"agent 里另一类常见的是推理型模型（DeepSeek 这类）：先在内部 step-by-step 想，再调用工具或输出。接法一样：SGLang co-deploy 透传，或在本地小模型里用 Candle 后端。",
    keywords:["deepseek","reasoning","推理","tool use","agent"] },

  { id:"model-distill", bucket:"model", title:"Laya 蒸馏：零幻觉小模型", en:"Laya distill", minutes:"",
    blurb:"用 Laya 当老师，蒸一个只出概率、不生成文本的小模型。",
    body:"判断类任务别惊动大模型。以 Laya 当老师：choice/score/noul 三种题型输出校准概率，用它做软标签，蒸一个几十 MB、CPU 几十毫秒的小模型；学生同样走概率头，不生成自由文本，所以判断类任务上没有幻觉。",
    keywords:["distill","蒸馏","零幻觉","0 幻觉","laya","小模型","teacher","student","软标签","高性价比"] },

  { id:"model-case", bucket:"model", title:"实用场景：工单分流与流失拦截", en:"Triage use case", minutes:"",
    blurb:"一封工单进来，小模型一次判定：转哪个组、急不急、会不会退订。",
    body:"输入一封工单原文，蒸馏小模型一次前向给三个判断：department=billing(choice, confidence 0.927)、urgency=1.77/2(score, blocking)、churn_risk=0.879(noul)。网关按概率设阈值：转账单组、排队置顶、churn 高危转人工。CPU 几十毫秒，不生成自由文本，概率直接当阈值。",
    keywords:["场景","use case","工单","ticket","分流","routing","triage","流失","churn","拦截","department","urgency"] },

  { id:"distill-demo", bucket:"model", title:"一键蒸馏 demo（distill/）", en:"One-click distill demo", minutes:"",
    blurb:"付费大模型当老师 → 蒸判断小模型 → 本地 serve；生成小模型才用 SGLang。",
    body:"distill/ 是一键框架：teacher.py 调付费/私有大模型出软标签，train.py 蒸出判断头（非自回归，无 prefill/decode，serve.py 本地部署即可，不用 SGLang）；如果学生是自回归生成小模型，那就有 prefill/decode，用 serve_sglang.py 基于 SGLang 本地部署。",
    keywords:["distill","一键","teacher","serve","sglang","prefill","decode","部署","软标签","蒸馏"] },

  { id:"laya-pipeline", bucket:"laya", title:"大模型 → Laya → 小模型（避开 RLHF）", en:"Teacher → Laya distill", minutes:"",
    blurb:"大模型走生成层（radix + 连续批），中间用 Laya 的校准概率自动蒸小模型，不靠 RLHF。",
    body:"三层闭环：大模型层用 SGLang 的 radix 前缀缓存和连续批调度派发生成；中间 Laya 用严格适当评分规则给校准概率当软标签；最后蒸馏出只出概率、不生成自由文本的小模型。相比 RLHF 用人类偏好奖励，这套不会学成“说你想听的”谄媚模型，判断类任务不幻觉。",
    keywords:["sglang","radix","continuous batching","rlhf","谄媚","sycophancy","幻觉","蒸馏","laya","校准概率","proper scoring","大模型","小模型"] },
  // ============ 卖点 / why ============
  { id:"why-1", bucket:"why", title:"零第三方依赖", en:"Zero dependencies", minutes:"",
    blurb:"只用 std，离线编译、离线测试。",
    body:"cargo test --offline 一遍过 87 个测试；没有 tokenizers/candle/CUDA。内网、金融、政务、涉密环境直接 build。",
    keywords:["零依赖","离线","std","cargo","内网","涉密"] },

  { id:"why-2", bucket:"why", title:"两种前缀缓存后端", en:"Radix vs APC", minutes:"",
    blurb:"radix 和 APC 同框架 A/B。",
    body:"EngineConfig.cache_backend 一行切换；共用 KV 池和指标；测试锁死两种后端输出一致。",
    keywords:["radix","apc","ab测试","双后端","对比"] },

  { id:"why-3", bucket:"why", title:"agent 场景原语", en:"Agent primitives", minutes:"",
    blurb:"checkpoint / 嫁接 / 会话 / 批量解码，给 agent 用的。",
    body:"共享树干 + 大量分支是 agent 树搜索的共同形状；这几个原语就是为它做的。",
    keywords:["agent","checkpoint","嫁接","树搜索","原语"] },

  { id:"why-4", bucket:"why", title:"收入指标", en:"Saved compute", minutes:"",
    blurb:"/stats 直接给省了多少算力。",
    body:"compute_saved_tokens / prefill_skip_ratio / avg_forward_per_output_token / kv_utilization。",
    keywords:["指标","收入","compute is revenue","stats"] },

  { id:"why-5", bucket:"why", title:"十四篇中文文档", en:"Docs", minutes:"",
    blurb:"从零讲到张量嫁接，带可跑命令。",
    body:"从“推理服务是什么”讲到调度指标，每篇一句总结 + 类比 + 图 + 动手命令 + 和 SGLang/vLLM 对照。",
    keywords:["文档","教程","小白","中文"] },

  { id:"why-6", bucket:"why", title:"共部署网关", en:"Coopetition", minutes:"",
    blurb:"不抢主战场，只补控制面。",
    body:"现有 SGLang 部署不用改，接一层网关就有 checkpoint、指标和护栏。",
    keywords:["竞合","共部署","sglang","网关"] },

  { id:"why-7", bucket:"why", title:"内置 Laya 决策模型", en:"Bundled System-1", minutes:"",
    blurb:"路由/护栏/审核用 Laya，不用惊动大模型。",
    body:"完整 candle 移植，e2e 断言账单场景全对、两次运行逐 bit 一致。",
    keywords:["laya","system 1","决策","护栏","路由"] }
];

// 过滤桶的中文标签
window.BUCKET_LABELS = {
  all: { zh:"全部", dot:"#F3EEE4" },
  docs: { zh:"文档", dot:"#FF5A36" },
  concept: { zh:"核心机制", dot:"#39D08F" },
  sglang: { zh:"整合·兼容", dot:"#8F7FF2" },
  laya: { zh:"Laya", dot:"#E8BC5C" },
  model: { zh:"模型", dot:"#F472B6" },
  metrics: { zh:"指标", dot:"#54C1EE" },
  why: { zh:"卖点", dot:"#F2DCA6" }
};