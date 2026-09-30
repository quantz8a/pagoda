# 10 · 批量解码：一次前向，养活整个批次（P3）

## 一句话总结

decode 阶段每生成一个 token，传统做法是**每条序列各跑一遍模型**（B 条序列 = 读 B 遍权重）。
批量解码把"恰好都只差一个 token"的序列拼成 `[B, 1]` **一次前向**跑完——
权重只从内存/显存里读一遍，B 条序列同时前进一格。

## 生活化类比

送快递的两种派单方式：

- **逐单派送（旧做法）**：8 个包裹在同一个小区，快递员往返 8 趟。
  每趟的油费（从内存读一遍模型权重）一分不少。
- **拼车派送（批量解码）**：8 个包裹装同一辆车，**一趟全送完**。
  油费还是一趟的油费，但 8 个客户同时收到货。

decode 是"读权重"密集型工作：模型 1.1B 参数，F16 就是 2.2GB，每吐一个 token 都要
完整读一遍。单条序列时这是逃不掉的成本；但 8 条序列各读一遍就是纯浪费——
权重明明可以同时服务所有人。

## 它是怎么工作的

```
引擎调度循环（pagoda 主 crate）
────────────────────────────────────────────────────────────
每个 decode 步：
  1. 长度闸：到 max_tokens 的序列先完赛，不占模型
  2. 分组：把"会话恰好缺 1 个 token"的序列挑出来（稳定批）
     - 首步（整条 prompt 要预填充）→ 每条单独走
     - 缺 0 个 token（刚分叉的 checkpoint 分支）→ 直接复用缓存 logits
     - 缺 1 个 token（绝大多数情况）→ 进批次
  3. 一次调用：model.session_forward_batch(会话们, 各自的 1 个 token)
     - 后端不支持批量？逐条 fallback，结果完全等价
  4. 每条的尾巴不变：罚分 → grammar 约束 → 采样 → 写回分页 KV
```

```
pagoda-hf 侧（自研 batched Llama，src/llama.rs）
────────────────────────────────────────────────────────────
难点：6 条序列的历史长度不一样，怎么拼成一个批次？
  K/V 补齐：每条会话的私有 KV 补齐到批内最长（pad 0），
            再用加法遮罩（pad 位置 -inf）让注意力无视补齐部分
  RoPE 按行取位：每条序列的位置不同 → 按行从 cos/sin 表里各取各的
  原子提交：先在克隆的工作副本上算，全部成功才写回——
            中途失败不留半拉子状态，回退逐条路径照样正确
```

## 在 pagoda 里动手试试

```powershell
# 核心语义（离线，玩具模型）：批量 vs 逐条输出逐 token 相等
cd pagoda; cargo test --offline --test batch_tests

# 真实权重端到端（联网机器）：与 candle 官方 Llama 逐 logit 对拍
cd pagoda-hf; cargo run --release --example e2e_batched_decode
# PASS parity: max |logit diff| vs candle Llama = 0.000e0   ← 逐 bit 相等

# 看批量倍率：物理调用数 vs 逻辑步数
cargo run --release --example bench
#   "decode_steps": 256, "decode_calls": 41, "decode_batch_factor": 6.24
#   含义：256 个逐序列解码步，只打了 41 次模型，权重少读了 6.24 倍
```

HTTP 服务的 `/stats` 和 CLI 的 `revenue:` 行也带了 `decode_batch_factor`。

## 实测效果（2026-09-30，共享开发机）

| 场景 | 批量前 | 批量后 | 说明 |
| --- | --- | --- | --- |
| tiny 模型 CPU，batch-8 吞吐 | 2218 tok/s | **3103 tok/s（+40%）** | 框架开销主导的场景直接受益 |
| 任意规模 | B 次物理前向/步 | **1 次/步** | 调用数 6.24× 收缩（可观测、可断言） |
| 1.1B 大模型（CPU/GPU） | 持平 | 持平（暂） | kernel 时间主导，见"边界" |

## 和 SGLang / vLLM 的对照

- SGLang/vLLM 的 continuous batching 是同一个思想：把序列拼进一个前向。
  它们靠手写 CUDA kernel（paged attention）把"不同长度的注意力"压进一次 kernel；
  pagoda 目前是"补齐 + 遮罩"的通用实现（任何 candle 支持的设备都能跑），
  融合 kernel 与 CUDA Graph 在路线图上（见 DESIGN.md §7）。
- 工程纪律一致：批量路径必须和逐条路径**逐 token 等价**（我们拿 candle 官方
  Llama 对拍到 0.0 差异，并在测试里锁死）。

## 常见疑问

**Q：为什么大模型下吞吐暂时没涨？**
A：批量解决的是"读权重次数"。当每步时间都耗在 kernel 本身的执行与发射上
（candle 目前是逐算子朴素 kernel），省掉的是调度/分发开销。下一步 CUDA Graph
把整个 decode 步录制成一次 kernel 发射，批量红利就会完全释放——
这也是为什么本步先把"一次调用养活一个批次"的数据通路修好了。

**Q：批量会让不同用户的输出互相影响吗？**
A：不会。数学上每行独立（注意力只在自己历史内），补齐部分被 -inf 遮罩。
e2e 用 4 条完全相同的 prompt 验证过：4 行输出逐 bit 一致。
唯一例外：低精度（F16）下不同批次形状的舍入噪声可能翻转"几乎同分"的
候选词——vLLM/SGLang 同样如此，属于硬件数值属性，不是逻辑 bug。

**Q：为什么自己实现一份 Llama 前向，不用 candle 官方的？**
A：candle 官方的 KV cache 是私有字段且一次只能跑一条序列，物理上无法批量。
我们用 candle-nn 的基础算子重写了一版（约 400 行），数学上与官方逐 bit 一致，
同时拥有每条会话私有的、可分叉、可拼批的 KV。
