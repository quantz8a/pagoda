# 08 · 增量 KV 会话：每个 token 只算一次（P2）

## 一句话总结

模型侧给每条序列开一个"会话"，里面装着它自己的 KV cache：
prompt 只喂一遍，之后每生成一个新 token 就只喂**这一个** token——
而不是像早期版本那样，每生成一个词都把整段上下文重算一遍。

## 生活化类比

开一场长会的两种记笔记方式：

- **全量重放（旧做法）**：每有人说一句新话，秘书就把**从开会第一分钟起**
  的全部对话重新抄一遍，再在最后添上这句新话。会越开越慢——说第 100 句话时
  要重抄前面 99 句。总工作量是平方级（O(n²)）。
- **增量会话（现在的做法）**：秘书手里有一本**持续更新的笔记**（KV cache）。
  新人只说新话，秘书只记新话。笔记越厚查阅略慢，但**写字的工作量**每场会
  只与"说了多少句"成正比（O(n)）。

`fork`（分叉）则是：把笔记**复印**一份交给另一个人接着开分会场——
原件共享、互不影响，不用回头重抄。这正是 agent 集群共享树干的开法
（配合 [05 · Checkpoint 与分支](05-checkpoint-branching.md)）。

## 它是怎么工作的

```
引擎侧（pagoda 主 crate）
────────────────────────────────────────────────────
trait ModelEngine {
    fn forward(&self, context)               // 无状态：全量重放（保底路径）
    fn begin_session(&self) -> Option<Box<dyn ModelSession>>   // 有 KV cache 的后端实现它
}

trait ModelSession {
    fn context_len(&self)                    // 我已经消化到第几个 token
    fn forward(&mut self, new_tokens)        // 只喂没见过的后缀
    fn fork(&self)                           // 复印 KV 快照，开分支
}
```

```
一次生成的投喂序列（max_tokens = 16，prompt = 6 个 token）：

  旧（全量重放）        新（增量会话）
  ─────────────        ─────────────
  forward(6)           session.forward(6)   ← prompt 一次性预填充
  forward(7)           session.forward(1)   ← 之后每步只喂新 token
  forward(8)           session.forward(1)
  ...                  ...
  forward(21)          session.forward(1)
  ─────────────        ─────────────
  共喂 216 个 token     共喂 21 个 token（省 10.3 倍）
```

引擎怎么知道喂多少？**它不问**。会话自己报告 `context_len()`，引擎永远只切
`tokens[context_len()..]` 这段新后缀。会话异常（报的长度比序列还长）时引擎
直接丢弃会话、退回全量重放——正确性永远优先于性能。

Checkpoint 分叉（agent 集群场景）：

```
create_checkpoint("共享树干")
    └── 树干一次性预填充进一个会话，pin 在 Checkpoint 里

generate_from_checkpoint(id, "分支 A 的问题")
    └── session.fork() ── 复印树干 KV（张量 Arc 共享，零拷贝）
        └── 只喂"分支 A 的问题"+ 后续每步 1 个 token，树干永不重算
```

## 在 pagoda 里动手试试

```bash
# 联网机器（真实模型，国内加镜像）
cd pagoda-hf && bash scripts/verify-p1.sh --mirror
```

输出里这两行就是增量会话的证据：

```
==> [3/5] engine generation, cold cache (incremental KV session)
    model tokens fed: 21 (full replay would be 216, 10.3x saved)
==> [5/5] checkpoint fork: shared trunk, branched continuation
    Length prefix_hit=6/11 branch_fed=12 (continuation 5 + 7 decode)
```

`tokens_fed` 是模型真实吃掉的 token 数（`CandleModel::tokens_fed_handle()`），
不是账本估计。主 crate 的 `tests/session_tests.rs` 用录制会话把
"每个 token 恰好喂一次"写成了断言，离线可跑。

## 和 SGLang / vLLM 的对照

| | 它们 | pagoda（当前） |
| --- | --- | --- |
| 单序列增量 | KV cache 常驻显存 | `ModelSession`，每序列一份 Candle KV cache |
| 跨序列共享前缀 KV | RadixAttention 张量级共享 | 引擎侧已有 token 级共享（[03](03-radix-prefix-cache.md)）；张量级共享是下一个里程碑（P3） |
| 分叉 | Radix 树天然支持 | `ModelSession::fork()` + checkpoint pin |
| 长续写的分块预填充 | flash-attn varlen | 受 candle 0.8 mask 限制，续写逐 token 喂入（见下） |

## 常见疑问

**Q：为什么续写（continuation）要逐 token 喂，不能一把喂？**
candle-transformers 0.8 的因果 mask 是 `[seq, seq]` 方阵，只在 KV cache 为空时
形状才匹配；cache 非空且一次喂多个 token 时，注意力分数是 `[seq, 已缓存+seq]`，
mask 广播直接报错。而 `seq=1` 时 candle 跳过 mask——正好就是 decode 的形状。
所以 pagoda-hf 的纪律是：首调全量预填充，之后逐 token。生产实现应换
flash-attn 或自研 kernel 解除这个限制。

**Q：会话和 03 篇讲的 radix 前缀缓存是什么关系？**
两层缓存，各管一段：

- **引擎层 radix/APC**（token 级）：记住"这段前缀我见过"，负责跳过记账、
  物理页共享、淘汰——对任何模型后端都成立。
- **模型层会话**（张量级）：记住"这些 token 的 K/V 我算过了"，负责省 GPU/CPU
  计算——当前是会话私有，新会话仍要自己预填充一遍命中的前缀。
  把两者缝起来（radix 命中的前缀直接嫁接张量 KV）就是 P3 的活。

**Q：fork 真的不重算吗？**
`LlamaCache` 里每层是 `(K, V)` 张量，candle 的 `Tensor` 内部 Arc 共享且不可变；
追加时 concat 出**新**张量。所以 clone 一份 cache 是元数据级开销，
两条分支各自追加互不干扰。`e2e_tiny_llama` 第 5 步断言了：
分叉只喂"续写 + 解码步"，树干 6 个 token 一次都没重算。

**Q：空续写（直接让树干开口说话）怎么办？**
会话缓存最近一次前向的 logits；引擎喂空切片时返回缓存值——
分支的第一个 token 就是从树干末尾的 logits 采出来的，
这在 `tests/session_tests.rs` 里有专门测试。