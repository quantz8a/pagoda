# 11 · 张量级前缀嫁接：真·RadixAttention（P3）

## 一句话总结

逻辑前缀缓存（第 3、4 篇）记住的是"这段前缀算过"；**张量级嫁接**更进一步——
把上一条请求算完的 **KV 张量本体**留下来，新请求的 prompt 只要和它撞前缀，
就直接把那一段 KV **切片拿来用**，一个 token 都不用重算。

## 生活化类比

烤蛋糕的两种"复用"：

- **记菜谱（逻辑前缀缓存）**：上次烤过"原味蛋糕胚"，本子记一笔"胚子烤过"。
  下次同样的订单，你不用重烤胚——但装饰、奶油还得从头做一遍。
  （引擎跳过了逻辑上的重复计算，但模型物理上还是要把 prompt 再喂一遍。）
- **连胚子一起留（张量级嫁接）**：上次烤好的胚子直接放架子上。
  新订单前缀相同？**切片端走**，从第 N+1 步接着做。
  模型连 prompt 都不用再读，首 token 延迟直接砍掉整段 prefill。

## 它是怎么工作的

```
一次请求完赛时（引擎 finalize）
──────────────────────────────────────────────
  会话的 KV 快照 → 存进 KvVault（键 = token 路径）
  存两把钥匙：
    ① prompt 前缀    → 下次"原样再问"直接命中
    ② 已喂入的完整路径 → "接着上次继续聊"整轮命中
  （张量是 Arc 共享的，存快照 ≈ 零拷贝，只多一把钥匙）
   fault 的请求？—— 不入库，坏 KV 绝不传染。

新请求进来时（引擎 admit）
──────────────────────────────────────────────
  1. 在 KvVault 里找"是 prompt 前缀的最长钥匙"
  2. 命中 → 把快照切片到覆盖长度（因果注意力保证：
     前缀的 KV 与后面的 token 无关，切片数学上严格相等）
  3. 上限 prompt_len - 1：至少留 1 个 token 要喂，
     这样第一步照常产出 logits，批量解码通路零改动
  4. 没命中？退回全新会话，逻辑前缀缓存照常工作——
     嫁接只是加速器，正确性从不依赖它
```

容量有上限（默认 128 条，`PAGODA_KV_VAULT_ENTRIES` 可调），LRU 淘汰——
淘汰只影响"下次还能不能命中"，永远不影响对错。

## 在 pagoda 里动手试试

```powershell
# 核心语义（离线，玩具模型）：4 项嫁接测试
cd pagoda; cargo test --offline --test graft_tests
#   warm_request_grafts_prompt_kv        重复请求嫁接 prompt_len-1 个 token
#   graft_preserves_outputs              嫁接 vs 不嫁接，输出逐 token 相等
#   faulted_sequence_kv_is_not_retained  故障请求的 KV 绝不入库
#   graft_works_with_apc_backend         与 APC 块缓存正交共存

# 真实权重端到端（联网机器）
cd pagoda-hf; cargo run --release --example e2e_tiny_llama
#   [4/5] model tokens fed: 16 (grafted 5 cached prompt tokens; cold was 21)
#   [5/5] checkpoint 嫁接已有树干 KV，只喂 1 个 token 就建好
#   stats: grafted=10 ← 物理上省掉的 prompt token 数

# 库容量：PAGODA_KV_VAULT_ENTRIES=1024 cargo run --release --example bench
```

## 实测效果（tiny 模型，CPU）

| 场景 | 冷请求 | 嫁接后 | 省掉 |
| --- | --- | --- | --- |
| 重复 6-token prompt + 16 token 生成 | 喂 21 token | **喂 16 token** | 24% |
| 创建 checkpoint（已有树干） | 喂 6 token | **喂 1 token** | 83% |

prompt 越长省得越多：agent 场景常见的"几千 token 系统提示 + 短问题"，
嫁接把整段系统提示的 prefill 直接归零——首 token 延迟和算力账单同步下降。

## 和 SGLang / vLLM 的对照

- SGLang 的 RadixAttention 用一棵 radix tree 同时索引"逻辑前缀"和"GPU 上的
  KV 张量"。pagoda 把两层拆开：引擎里的 radix/APC 管逻辑命中（快、确定性），
  KvVault 管张量本体（跨请求嫁接）——职责分离后，嫁接失败只是"没加速"，
  永远不是"算错了"。
- vLLM 的 APC 是块级复用；pagoda 的嫁接是 token 级切片（`narrow` 视图，
  零拷贝），粒度更细，且与逻辑缓存后端（radix 或 APC）正交。
- 稀疏键是刻意的取舍：只在"prompt 边界"和"完整路径"两处存钥匙，内存可控；
  任意深度命中（完整 radix 键控）是规模化路线，见 DESIGN.md。

## 常见疑问

**Q：切片别人的 KV 给新请求用，结果还一样吗？**
A：严格一样。因果注意力下第 i 个位置的 KV 只由前 i 个 token 决定，
和第 i+1 个及以后的 token 无关。测试里"嫁接 vs 重算输出逐 token 相等"
是硬断言，不是抽样观察。

**Q：存一堆 KV 快照，显存会不会爆？**
A：快照是 Arc 共享的视图，存的时候零拷贝；真正占显存的是张量本体，
由 LRU 上限（默认 128 条）封顶。淘汰一条只是下次少一次命中。

**Q：逻辑前缀缓存已经有命中率了，为什么还要嫁接？**
A：逻辑缓存省的是"引擎侧"的重复劳动；没有嫁接时，模型物理上仍要把
命中的 prompt 重喂一遍（warm 请求 21 token）。嫁接后只喂 16 个——
那 5 个 token 的 prefill 物理上消失了。agent 集群场景（共享长系统提示）
这个差值是数量级的。