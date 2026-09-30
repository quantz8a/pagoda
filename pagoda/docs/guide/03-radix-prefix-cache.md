# 03 · Radix 前缀缓存（RadixAttention）

## 一句话总结

把"所有历史请求的 token 序列"存成一棵前缀树（trie）；新请求开头和历史越长地重合，
就越多地跳过 prefill 计算——重合部分的 KV 块直接引用，一个字都不用重算。

## 生活化类比

字典的前缀检索：查 "apple" 时你已经翻到了 `a-p-p-l-e` 这条路径；再查 "application"
时不用从头翻——`a-p-p` 这三层是现成的，直接从第 4 个字母继续。

推理服务里：100 个客服机器人都以同一段 500 字的 system prompt 开头。没有前缀缓存，
这段开头被算 100 次；有了它，**算 1 次，引用 99 次**。

## 它是怎么工作的

```
root
 └─ "你"
     └─ "好"
         ├─ "吗"        ← 请求 1 的路径：你好吗
         └─ "，" 
             └─ "请"    ← 请求 2 的路径：你好，请...
```

新请求 "你好呀" 到达：

1. 从根往下走："你" ✅ "好" ✅ "呀" ❌ —— 命中长度 = 2。
2. 前 2 个 token 的 KV 块**直接引用**（引用计数 +1），prefill 从第 3 个 token 开始。
3. 请求结束后，完整路径 "你好呀" 也插进树里，供后来的请求复用。

pagoda 的实现（`src/radix_cache.rs`）比"纯 token 树"多走一步：每个节点还带
**物理槽位** `(block_id, offset)`，所以命中不只是"少算"，而是**零拷贝**——
连 KV 块都是共享的，只有追加写时才 COW（见第 2 篇）。

### 淘汰（LRU）

树不可能无限大。每个节点记着"上次被用的时间"，池子满了就淘汰最久没用的**叶子**
（叶子被淘汰不影响它的祖先，因为祖先还被别的路径用着）。pagoda 会循环淘汰直到
真的腾出块——pin 住的 checkpoint 块除外（第 5 篇）。

## 在 pagoda 里动手试试

```powershell
cargo run --offline --bin pagoda -- sample -p "同一段很长的开头" --repeat 3
```

输出里第二次请求起 `prefix_hit` = 整个 prompt 长度，`compute_saved` 同步增加。
也可以看 `GET /stats` 的 `radix_hit_tokens` 与 `prefill_skip_ratio`。

## 和 SGLang 的对照

| 机制 | SGLang RadixAttention | pagoda |
| --- | --- | --- |
| token 粒度的前缀树 | ✅ | ✅ |
| 节点携带 KV 槽位 | ✅（GPU 池槽位） | ✅（`(BlockId, offset)`） |
| LRU 淘汰 | ✅ | ✅ 叶子淘汰 + 循环到腾出块 |
| 命中统计 | ✅ | ✅ `hit_queries` / `hit_tokens` / `cache_hit_rate` |

## 常见疑问

**Q：radix 和 APC（下一篇）哪个好？**
radix 是 token 粒度：哪怕前缀只重合 3 个 token 也能复用，查找代价是逐 token 走树。
APC 是块粒度：查找是几次哈希探测、极快，但不满一块的尾部不能复用。
SGLang 选了 radix，vLLM 选了 APC，pagoda 两个都有，`EngineConfig.cache_backend` 一行切换。

**Q：前缀树会不会越长越乱？**
会，所以要有淘汰。真实部署里淘汰策略（LRU / LFU / 成本感知）是调优重点之一。