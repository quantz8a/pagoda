# Pagoda 推理框架详细设计

本文描述 2026-10-19 仓库里实际存在的两套推理路径，以及它们怎样接在一起、哪些地方还没有接上。数字只写已经测过或从这份 `mlx-community/Qwen3.8-27B-8bit` 的 `config.json` 读出来的值。`docs/DESIGN.md` 讲零依赖控制面；本文把控制面和 Qwen3.8 的 Metal 权重面放在同一张图里。

## 1. 要解决的问题

一次推理服务要同时做三件事：

1. 把提示变成 token，再把 token 变成下一词。
2. 记住已经算过的上下文，避免每个新 token 都把整段提示重算一遍。
3. 在多条请求之间决定谁先算、前缀能不能共用、KV 放在哪。

Pagoda 把这三件事拆成两个 crate。

| Crate | 依赖 | 现在干什么 |
| --- | --- | --- |
| `pagoda` | 只有 Rust `std` | 调度、分页 KV、前缀缓存、采样、约束解码、DSL、HTTP 壳。离线可编译、可单测。模型是确定性 n-gram，用来把链路跑通。 |
| `pagoda-hf` | Candle、tokenizers、Metal | 读真实权重。Qwen3.8 文本塔是 `Qwen35`；`QwenEngine` 实现 `ModelEngine`，经 `QwenSession` 挂进 `Engine`。连续批仍未接通。 |

飞书 Hermes 和 Codex 现在打的是 `pagoda-hf` 的 `qwen3_server`，监听 `127.0.0.1:8000`。生成已经交给 `pagoda::Engine`：`QwenEngine` 实现 `ModelEngine`，每条序列一个 `QwenSession`。HTTP、聊天模板和交互窗裁剪（现与模型上限同为 262144）仍在这个进程里，不在 `pagoda serve` 那个二进制里。

## 2. 总图

```text
客户端
  Codex++ / Codex CLI     wire = Responses   POST /v1/responses
  Hermes 飞书             wire = Chat        POST /v1/chat/completions  stream
        │
        ▼
qwen3_server                         一个进程，一个 Metal 设备，一条请求
  渲染 Qwen 聊天模板
  编码
  若新提示里含有上一轮的 token 序列，只喂新增后缀
  否则整段重喂（上限 262144）
        │
        ▼
Qwen35::prefill / forward_greedy
  权重：affine 8-bit，组大小 64，反量化 bias + scale * code
  常驻 GPU。decode 一次 command buffer 走完 64 层
  prefill 四个已知 token 一组，64 个 token 一次提交，不算 lm_head
        │
        ▼
一份 Resident
  token id 序列 + Qwen35Cache
  线性层：因果卷积窗口 + Gated Delta 状态
  全注意力层：K/V，按 token 追加
```

控制面 `Engine` 在旁边，结构是另一套：

```text
HTTP /generate
  → 准入（队列长度、prompt+max_tokens）
  → Radix 或 APC 匹配前缀
  → 分页 KV 写入未命中的后缀（chunked prefill）
  → 每步让所有在跑的序列各出一个 token
  → 采样（温度 / top-k / top-p / 惩罚 / grammar mask）
  → 结束时把 token 路径插回前缀缓存，页引用归零则回收
```

Qwen3.8 的前向通过 `QwenSession` 接到 `Engine`。引擎仍用自己的分页表记账；真实的 KV 和 Gated Delta 状态在会话里。多条请求的连续批还没有：`session_forward_batch` 返回空，引擎退回逐条 `forward`。

## 3. 模型：这份 Qwen3.8 是什么

权重目录 `mlx-community/Qwen3.8-27B-8bit`。配置架构在 `text_config`。目录名里的 8 是量化位宽。

| 项 | 值 |
| --- | --- |
| 层数 | 64 |
| 层型 | 48 层 `linear_attention`，16 层 `full_attention`，每 4 层一轮，第 4 层是全注意力 |
| hidden | 5120 |
| MLP intermediate | 17408 |
| 词表 | 248320 |
| 全注意力 | 24 个 query 头，4 个 KV 头，`head_dim` 256 |
| 线性注意力 | 16 个 key 头、48 个 value 头，头维都是 128，因果卷积核长 4 |
| RoPE | `rope_theta` 10_000_000，`partial_rotary_factor` 0.25，所以旋转维是 64 |
| 最大位置 | 262144 |
| 视觉塔 | 不加载 |

配置里还有 `mrope_section: [11, 11, 10]`。文本服务没用多段 mRoPE，位置是一个整数，只旋转前 25% 的头维。

全注意力的 KV 随序列变长。线性注意力的状态是定长的：卷积窗口和 Gated Delta 的递推状态，不随 token 数线性涨。所以长上下文的显存压力主要在那 16 层全注意力上。

## 4. 权重怎么放

MLX 目录是 safetensors 分片。8-bit 仿射量化：一组 64 个数共用一个 scale 和一个 bias，编码是 uint8，打包在 little-endian 的 uint32 里。反量化是

```text
value = bias + scale * code
```

Metal 路径不把矩阵展开成 F16 再留在显存里。打包权重、BF16 的 scale 和 bias 上传一次，之后每个 token 做 GEMV / 小批量 GEMM。MLP 那条 17408×5120 的瓦片，2026-10-17 前后测到大约 590–600 GB/s。

文本塔权重大约 27.2 GB。mlx-serve 把同一份权重常驻时大约 29.5 GB，另加前缀缓存。两份同时常驻会把这台 96 GB 的 M3 Ultra 挤满，所以 `qwen3_server` 独占这份权重。

## 5. 一次解码

热路径是 `forward_greedy`，`seq == 1`。

1. 嵌入一行：按 token id 把 embed 矩阵的一行反量化进 ping 缓冲。
2. 64 层交替读写 ping / pong。线性层做输入 RMSNorm、QKV/Z/A/B 投影、因果卷积、Gated Delta、输出投影、残差、SwiGLU。全注意力层做 Q/K RMSNorm、部分 RoPE、把 K/V 写入该层的 KV，再做带输出门的注意力和 SwiGLU。
3. 最后一层的 hidden 做 final RMSNorm，lm_head GEMV，argmax。
4. 整个塔加词表头在一个 command buffer 里，一次 `wait`。

贪心。温度记成 0，因为这条路径没有采样。EOS 或 `<|im_end|>` / `<|endoftext|>` 结束。

2026-10-17，权重已经在 GPU 上、短上下文、跳过加载之后，7 个 token 平均 **23.5 tok/s**（约 42–43 ms）。同机 mlx-serve 三次中位 **23.3 tok/s**。同一句 `Hi` 的 16 个贪心 token 与 mlx-serve 逐字相同。这是热解码，不是冷启动。冷启动第一 token 曾经到 46 秒，里面包含打包权重上传。

解码慢在读权重，不在注意力的短上下文。27 GB 量级的权重按大约 600 GB/s 读一遍，就是几十毫秒，和测到的 43 ms 同一量级。

## 6. 一次预填

预填的 token 是已知的。服务要更新缓存，不需要每个位置的 logits。

`fused_prefill`：

- 四个 token 一组。GEMM 核把同一组权重读一遍，算出最多四个输出。核里累加器是 4 路，所以批量上限是 4。
- 线性层的卷积和 Delta 状态、全注意力的 KV 仍按 token 顺序写。后一组能看见前一组写完的状态，因为它们在同一个 command encoder 里按顺序 dispatch。
- 每 64 个 token 提交一次 command buffer，然后等待。这样编码器不会无限变大，又不用每 4 个 token 做一次 CPU 往返。
- 预填不算 lm_head，也不做 argmax。词表 248320，这一步只在真正要吐出下一个 token 时才做。
- scratch、ping、pong、整段 RoPE 分配一次，各组复用。

`forward_verify` 仍在，用来对四个已知位置做带 logits 的对照。2026-10-18，四个已知且全部接受的 token 一次验证 **50.9 tok/s**；逐条是 22.3 tok/s。四个位置的 token 和最大 logit 与逐条解码相同，绝对差是 0。预填走的是不算 logits 的那条，目标是把状态推过去，不是把 50.9 当成生成速度。

2026-10-19，交互服务上 255 个 token 的预填稳定在大约 **5.0 秒**，约 51 tok/s。这是整塔状态更新的墙钟，不是 mlx-serve 那次 827 token / 1.47 秒（约 560 tok/s）的数字。两套内核不一样，不能把 560 安到 pagoda 的预填上。

## 7. 会话：一份 Resident

`qwen3_server` 进程里只有一份 `Resident`：

```text
ids:   已经喂进缓存的 token，含提示和已生成
cache: 各层线性状态 + 各层全注意力 KV + 当前位置
```

下一次请求编码之后：

1. 在完整 token 序列里从右往左找 `ids`。找到了，而且后面的新 token 不少于 1、不多于交互窗上限，就只把这段后缀送进 `prefill`。RoPE 的位置从 `cache.len` 接着数，和上次停住的位置对齐。
2. 找不到，或新后缀超过上限，就丢掉更早的 token，只留末尾最多 262144 个，缓存作废，整段重喂。日志是 `prefix miss` 或 `trim`。

2026-10-19 接上的短后缀：21 个新 token 预填 0.46 秒，39 个新 token 0.75 秒。上下文变长以后，同样 21 个 token 会到大约 1.2 秒，因为 16 层全注意力要扫更长的 KV。

这份缓存对不上就会整段重来。Codex 把助手回复按聊天模板再渲染一遍时，token 边界常常和上次生成的 id 不一致，于是 miss。命中的条件是下一轮编码结果里原样含有上一轮的 id 序列。

进程同时只处理一条连接。预填或解码没结束，下一条在监听队列里等。

## 8. 交互窗口与模型上限对齐

模型位置上限是 262144。服务按同一上限拒绝「提示 + 本次最大输出」，交互裁剪也收到这个数，不再额外砍到 256。代价是预填大约 50 tok/s：两万 token 要几分钟，而且在出第一个字之前客户端看不到正文。

CoT 默认打开：未传 `reasoning.effort` 时助手前缀仍留未闭合的 `<think>`（Codex++ 有时会省掉该字段）。`low` / `medium` / `high` 同样打开；`none` / `off` / `minimal` 或 `enable_thinking: false` 合上。流式路径里可见正文仍等 `</think>`；思考进最终响应的 `reasoning` 项。

## 9. HTTP

`qwen3_server` 实现的面：

| 方法 | 路径 | 谁在用 |
| --- | --- | --- |
| GET | `/health` | 探活 |
| GET | `/v1/models` | 列出已加载的 `mlx-community/Qwen3.8-27B-8bit` |
| POST | `/v1/responses` | Codex。`stream` 时用 SSE |
| POST | `/v1/chat/completions` | Hermes。`stream: true` 时按 OpenAI chunk 吐 `delta.content` |
| POST | `/v1/completions` | 纯文本补全 |

聊天模板把 `instructions` 和 `input` 收成 Qwen 的 `<|im_start|>` 轮次。工具按权重目录里的 `chat_template.jinja` 写成 `<tool_call>`，再翻成 Responses 的 `function_call`。CoT 默认打开；`reasoning.effort` 为 `low` / `medium` / `high` 时同样打开。思考在 `</think>` 之前，正文在之后。`none` / `off` / `minimal` 在生成前把思考块合上。

`/v1/responses/compact` 没有实现。Codex 在它自己的窗口里要压缩会话时，这里回 404。

`pagoda` 主 crate 的 `server.rs` 是另一套 HTTP：`/generate`、OpenAI 兼容路由，以及把 `/v1/responses` 原样转给上游。它不加载 Qwen3.8。

## 10. 控制面引擎

`pagoda::Engine` 是给多请求、可单测的运行时准备的。Qwen3.8 已通过 `QwenEngine` / `QwenSession` 挂上生成路径；分页表仍是记账，真实 KV 和 Gated Delta 状态在会话里。下面这些不变式仍约束后续把连续批和页映射做完。

**分页 KV。** 空间切成固定 `block_size` 的页，默认 16 个 token 一页，池子默认 4096 页。页有引用计数，归零才回空闲链表。`fork_block` 在引用大于 1 时先复制再写。参考实现里页里存的是 token id；真实权重应把同一页号映射到各层 K/V。

**前缀缓存。** 默认 Radix：按 token id 走一棵树，命中长度是可以跳过的前缀。APC 是另一后端：完整块用「块内 token + 前一块哈希」做链式哈希，淘汰以块为单位。两者都只保证「前缀相同才命中」。

**调度。** 等待队列三种顺序：先来先服务、最长共享前缀优先、最短剩余提示优先。每步有预填 token 预算，长提示跨 step 写完。解码步里每个在跑的序列出一个 token，结束的退出本步。同时在跑的请求数、等待队列长度、`prompt + max_tokens` 都有上限，超出分别拒绝为队列满或过长。

**采样与约束。** 温度、top-k、top-p、频次和存在惩罚。grammar 是字节级正则或 JSON 下推扫描，采样前把非法 logit 打成负无穷。只允许字节级 tokenizer，BPE 在准入时拒绝，避免按字节去 mask 词表。

**检查点。** 把一条会话的 KV 钉住，分支从钉住的前缀接着算，而不是每个分支重填树干。HTTP 上有对应端点。这是多 agent 共用长系统提示时省前向的那个机制。玩具计次里，8 路、树干约 3968 token、每路生成 32，前向从 32368 次降到 624 次，少 98.1%。这是计数，不是 27B 的墙钟。

**PD 分离。** prefill 角色把 KV 打成包放进对象库，decode 角色按 key 取回再生成。统一角色也可以先把预填转给对端。输出要求与不分离的路径逐 token 相同。Qwen3.8 的单机服务没有走这条路：预填和解码在同一个进程、同一份缓存里。

## 11. 和「接上 Qwen」之间还差什么

要把 `qwen3_server` 收成 `Engine` 的一个 `ModelEngine`，至少要补这几块。现在都没做。

1. **页号到真实 KV 的映射。** 引擎的页是 token id。Qwen 的缓存是 16 层全注意力的 K/V，加上 48 层线性状态。线性状态不能按 token 页拆开再随便拼接，分叉时要整份复制递推状态。
2. **批量大于 4 的预填。** 现在的 GEMM 是 4 路。mlx 那条大约 560 tok/s 的预填，靠的是一次读权重覆盖更多 token。4 路把权重摊到 4 个 token 上，墙钟仍在大约 50 tok/s。
3. **多条请求的连续批。** 现在一条连接占满 GPU。引擎的解码步假设同一形状的会话可以拼成一批、权重读一遍。Qwen 的 fused decode 是单条 `seq == 1`。
4. **稳定的前缀身份。** 现在用「整段 id 是否原样出现」判断能不能延长。聊天模板一改写就 miss。引擎用的是 token 路径上的 Radix，命中粒度是前缀而不是「整段必须相等」。
5. **交互窗口。** 已与模型上限 262144 对齐。预填不上来时，长提示的第一条消息仍会沉默很久；这是墙钟，不是再砍上下文能单独修好的。

## 12. 一条请求的时间怎么花

以 Codex 一条新会话、约 256 个 token 的提示、生成几十个 token 为例（2026-10-19，本机；更长提示按约 50 tok/s 线性拉长 prefill）：

| 段 | 大约时间 | 原因 |
| --- | --- | --- |
| 进程已热、前缀未中 | 约 5 秒 | 255 个 token 把 64 层状态推一遍 |
| 前缀命中、只补 20 个 token | 0.5–1.2 秒 | 全注意力 KV 越长越慢 |
| 每个新 token | 约 43 ms | 读一遍 8-bit 权重 |
| 生成顶满 512 | 约 22 秒 | 没遇到停止符，解码按上限跑完 |

冷启动另计：加载约 27 GB、上传打包权重、warmup 一次前向，然后才监听。

## 13. 不变量

这些是改内核时要守住的。

- 预填四个 token 一组之后的缓存，必须和下一步贪心解码从同一状态出发的结果一致。`forward_verify` 已用来对过 logits。
- 延长命中时，新 token 的 RoPE 位置等于上一轮结束时的 `cache.len`，不能从 0 重新数。
- 线性层的卷积窗口和 Delta 状态只在本条序列上按时间前进。另一条请求不能接着写。
- 全注意力只追加 K/V，不改已经写入的前缀。
- 反量化公式保持 `bias + scale * code`，组大小 64。换累加顺序会让 logit 漂，默认路径不换。
- 视觉权重不进这个进程。
- 主 crate 继续零依赖、离线可测。真实权重只活在 `pagoda-hf`。
