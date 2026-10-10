# 18. PD 分离：prefill 和 decode 分开部署（Mooncake 风格）

## 一句话总结

把"读题"和"写字"拆成两个工种：prefill worker 专门读 prompt 算出 KV，
往公共仓库（KV store）一放；decode worker 从仓库领走 KV，不重读题就直接开始写字。

## 生活化类比

同声传译的接力。**Prefill** 像速记员：把演讲者已经说完的内容（prompt）
速记成一份笔记（KV）。**Decode** 像译员：拿到笔记就开始逐句翻译，不需要
让演讲者从头再讲一遍。中间的 **KV store**（Mooncake 里叫 Mooncake Store）
就是传递笔记的公文柜——笔记按内容哈希编号，同一篇演讲第二次来时，
速记员直接说"笔记已在柜子里"（`store_hit`），连速记都省了。

## 它是怎么工作的

```
                 +----------------------------+
                 |  pagoda store（KV 对象池）  |
                 |  PUT/GET/DELETE /kv/<key>  |   <- Mooncake Store 的对应物：
                 |  内容哈希键 + LRU 字节预算  |     生产环境用 RDMA transfer engine
                 +-------^------------+-------+
             PUT bundle  |            | GET bundle
                         |            v
  POST /prefill  +-------+------+   +-------------+   POST /generate
 --------------->| prefill       |   | decode       |<---------------
   {"text":...}  | worker :8001  |   | worker :8002 |   {"kv_key":...}
                 | 只读题不写字  |   | 只写字不读题 |
                 +---------------+   +-------------+
                        ^                     ^
                        |   --prefill-url     |  （conductor：decode 收到纯文本
                        +---------------------+    会自己去找 prefill 要 kv_key）
```

传输的"笔记"（`PrefillBundle`）包含：

1. `prompt_tokens` —— prompt 的 token 路径。pagoda 参考实现的物理 KV 页
   里存的就是 token id（见第 2 篇），所以 token 路径**就是**逻辑 KV 镜像；
2. `kv` —— 真实后端的张量 KV（`ModelSession::export_kv` 导出，
   decode 侧 `ModelEngine::import_session` 重建；玩具模型为空，退化为
   无状态重放，语义不变）。

decode worker 领到 bundle 后，把 prompt 页"誊写"进自己的分页 KV
（相当于 transfer engine 把远端 HBM 的 KV 拉进本地），然后进入**同一个**
连续批解码环路——`drain()` 一条代码路径，所以分离版和统一版的输出
逐 token 相等（测试锁死这一点）。账本也分得很清：prefill 的算力记
在 prefill worker（`total_prefill_tokens`），decode worker 这项恒为 0。

## 在 pagoda 里动手试试

三条命令拉起整套 PD 拓扑（三个终端）：

```powershell
cargo run --offline --bin pagoda -- store --port 9100
cargo run --offline --bin pagoda -- serve --port 8001 --role prefill --store http://127.0.0.1:9100
cargo run --offline --bin pagoda -- serve --port 8002 --role decode --store http://127.0.0.1:9100 --prefill-url http://127.0.0.1:8001
```

客户端两种玩法：

```powershell
# 玩法一：手动接力（先存笔记，再凭编号取）
Invoke-RestMethod -Method Post http://127.0.0.1:8001/prefill -Body '{"text":"SGLang uses a radix tree to cache"}' -ContentType 'application/json'
#   -> {"kv_key":"kv-ab12...","prompt_tokens":9,"prefill_tokens":9,...}
Invoke-RestMethod -Method Post http://127.0.0.1:8002/generate -Body '{"kv_key":"kv-ab12...","sampling_params":{"max_tokens":32}}' -ContentType 'application/json'

# 玩法二：conductor（对客户端完全透明，就像普通 /generate）
Invoke-RestMethod -Method Post http://127.0.0.1:8002/generate -Body '{"text":"SGLang uses a radix tree to cache","sampling_params":{"max_tokens":32}}' -ContentType 'application/json'
```

看账本：`GET /stats` 里 `pd_role` / `pd_prefill_requests` /
`pd_decode_requests` / `pd_kv_bytes`；store 侧 `GET /store/stats` 有
entries / bytes / hits / misses / evictions。

## 和 Mooncake / SGLang 的对照

| Mooncake（生产） | pagoda（参考实现） |
| --- | --- |
| Mooncake Store（分布式 KV 池） | `pagoda store`（HTTP 对象池，LRU 字节预算） |
| Transfer Engine（RDMA 点对点） | `HttpStore`（TCP/HTTP）；trait 不变，实现可换 |
| Conductor（调度器） | decode worker 的 `--prefill-url` 自动接力 |
| KV block 按哈希寻址 | `bundle_key` = prompt token 路径的 FNV-1a |
| prefill/decode GPU 分池 | `--role prefill` / `--role decode` 进程分离 |

什么时候值得拆：prompt 长、输出短、TTFT 和 TPOT 要分别兜底（见第 1 篇）。
小模型/短 prompt 不用拆——统一的 `pagoda serve` 就够了。

## 常见疑问

**Q：decode worker 本地没有 prefill，前缀缓存还有用吗？**
有。decode 完成后照常 `publish_prefix`，同 worker 上的后续请求照样命中
本地 radix/APC；跨 worker 的复用则靠 store 的内容哈希键。

**Q：bundle 会被挤掉吗？**
会，store 有字节预算，超了按 LRU 淘汰；decode 拉到一半没了就是
`kv_miss`（404），重新走 prefill 即可。

**Q：玩具模型没有张量 KV，这分离是不是"假"的？**
不假。分离的是**调度与计费语义**：prefill 的页物化和算力账目只发生在
prefill worker，decode 侧零 prefill。张量 KV 的序列化钩子
（`export_kv`/`import_session`）已经留在 trait 上，candle 后端补上即可。

## 进阶三件套（2026-10-09 全部落地）

**1. 张量 KV 真的上线（pagoda-hf）**
玩具模型的"笔记"是 token 路径；candle 后端的笔记是**真的 K/V 张量**：
`SessionKv::export_bytes` 把每层 K/V 搬到 CPU、拓宽成 F32 写进 bundle
（F16/BF16 的值在 F32 里都能精确表示，回来再 cast 回计算精度，逐位一致），
还顺带捎上 prefill 的最后一个 logits——所以 decode worker 的**第一个**
采样也不用重算。`pagoda-hf` 的引擎级对拍测试证明：同一份请求，统一引擎
和"prefill 导出→store→decode 导入"的输出逐 token 相等，且 decode 侧
`tokens_fed` 只包含生成的 token。

**2. 流式 PD（SSE）**
`/generate` 带 `"stream":true` 时，decode worker 每采一个 token 就推一帧
`data: {"id":N,"token":"..."}`，最后来一帧 finish_reason/usage，再 `[DONE]`
收尾。调度环路挂了一个可选的 per-token sink，统一路径零开销；kv_miss 这类
前置错误在第一帧之前发生，所以还是干净的 JSON 错误（404/502）。conductor
模式下 text + stream 也通：先 buffered 做完 prefill，再流式解码。

**3. 前缀亲和路由器（多 decode worker）**
`pagoda route --prefill-url ... --decode-url ...（可重复）` 是 Mooncake
conductor 的完整形态：收到 text 先转 prefill 拿 `kv_key`，再查一张
字符 trie——**哪个 decode worker 以前见过最长的相同前缀，就派给谁**
（它的本地 radix 缓存还能再省一道），打平/冷启动走 round-robin。
响应逐字节中继（SSE 逐 chunk），`GET /route/stats` 看每个 worker 的
命中次数。

```
client → router → prefill worker ──PUT bundle──▶ KV store
              │                                     │
              └──── kv_key → decode worker ◀──GET───┘
                      （亲和选择：trie 最长前缀）
```

## 再加固（2026-10-09）：TTL、分诊门、真权重

**store 的租约（TTL）**：公文柜里的包裹不能永远占着格子——
如果 decode worker 宕机、没人来取，bundle 会永久占用池容量。
`pagoda store --max-age-secs 300` 给每件包裹一个有效期：
到期后第一次被摸到（get）或上新货（put）时惰性清掉，字节立刻归还预算，
`/store/stats` 里的 `expired` 计数器记录清了多少件。
这正是 Mooncake 给 KV block 加 lease 的用意。

**router 的分诊门**：`pagoda route --laya-url http://laya:8081`
让分诊站在 prefill **之前**——敌意在门口就被升级成人工接管回复，
一文 prefill 算力都不花（对比一下：`pagoda serve --laya-url` 的门
在代理模式上游之前；router 模式的门在整个 PD 流水线之前）。
同样的四个旋钮：`--laya-shadow`（只看不拦）、`--laya-required`
（Laya 挂了宁可拒单）、`--churn-threshold`、`--min-confidence`。

**真权重上线**：`pagoda-hf` 新增 `serve` 二进制
（`cargo run --release --bin serve -- --repo <hf-repo>`），
HuggingFace tokenizer + Candle Llama 权重直接接进同一套 HTTP/PD 前端，
`--role/--store/--prefill-url` 参数与玩具版完全一致。
三进程真权重实测：PD 分离输出与统一服务逐字节相等；
性能数字与"什么时候该分"的分析见 `docs/BENCHMARK-PD.md`
（一句话：prefill 算力 ≫ KV 传输时，分离才赢）。

## 1.1B 实测：什么时候分，什么时候不分

拿 TinyLlama-1.1B 真权重扫了 prompt 长度、又做了并发隔离实验
（细节和数字在 `docs/BENCHMARK-PD.md`）：

* **一个请求时**：分离从不吃亏——KV 打包走 base64 之后，传输相对
  CPU prefill 的算力开销可以忽略，128 token 的小 prompt 也一样。
* **多个请求时**：分离的好处是"prefill 不占 decode 的资源"。前提是
  资源真的分开（另一台机器、另一张卡、至少钉核）；如果两个角色挤在
  同一颗 CPU 上抢核，反而比不分更慢。
* 一个教训藏在实现里：conductor 最初把 prefill 调用放在 decode 引擎的
  锁里面，隔离形同虚设——排队请求还是得等。把 prefill 挪到锁外，
  并发延迟立刻降了 21%。**架构对不对，要用并发实验来验证。**
* 进程内并发调度（`pagoda serve --concurrent`，P7 调度 actor）也不是
  隔离的替代品：CPU 打满时，交错调度只把「排队」变成「一起变慢」
  （B 仅快 ~8%，A 慢 2.3 倍）——换的是公平性和流式体验，不是吞吐。
  2026-10-10 复测修正这个结论的一半：把批量解码的「补齐+遮罩」换成
  varlen 逐会话精确长度注意力之后，进程内并发对 B 的优势从 -8% 扩大到
  **-21%**（48.1s vs 串行 60.7s），和 PD 钉核对 B 的效果（47.1s）打平。
  PD 剩下的独占价值是 A 的延迟：43.8s vs 并发模式 59.4s——A 的 decode
  始终独占核，不被 B 的长 prefill 插队。