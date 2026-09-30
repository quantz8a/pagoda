# 05 · Checkpoint 与分支（agent 集群原语）

## 一句话总结

把一段"共享树干"（比如很长的 system prompt 或一段对话历史）物化进 KV 池并**钉住**，
之后任意多个分支直接从树干的物理槽位接着长——不重算、不怕被淘汰、互不污染。

## 生活化类比

做实验报告的**模板复印**：

- 没有 checkpoint：每个人拿到空白纸，把 10 页模板手抄一遍再写自己的部分。
- 有 checkpoint：模板复印 100 份发下去，大家直接接着写。
- **pin（钉住）** = 这份模板被锁进保险柜，保洁（LRU 淘汰）永远收不走。
- **写时复制** = 任何人在自己那份上涂改，都不会弄脏保险柜里的原件。

这正是 Manus 这类自主 agent 集群的场景：几十个 agent 共享同一段系统提示词 +
工具说明 + 初始上下文，然后各自分叉探索；还有树搜索（Tree-of-Thoughts / MCTS）
里一个节点分出 N 个子节点。共享树干只算一次，是整个场景能跑得起的前提。

## 它是怎么工作的

```
create_checkpoint("共享树干……")
        │
        ▼
  物化进 KV 池 ──▶ 发布进前缀缓存 ──▶ 保留一份"pin 引用"
        │                                │
        │                    LRU 淘汰来了也收不走这些块
        ▼
generate_from_checkpoint(id, "分支 A 的问题")
generate_from_checkpoint(id, "分支 B 的问题")
        │
        ▼
  每个分支：直接引用树干的物理槽位（不走缓存查找，100% 命中）
           续写自己的部分（追加共享尾块时自动 COW）
           结束时释放自己的引用；树干的 pin 还在
```

三个 API（`src/engine.rs`）：

| API | 作用 |
| --- | --- |
| `create_checkpoint(text) -> CheckpointId` | 物化 + 发布 + pin |
| `generate_from_checkpoint(id, continuation, sampling)` | 从 checkpoint 分支生成 |
| `drop_checkpoint(id)` | 解除 pin（块仍可被普通缓存复用，直到被淘汰） |

HTTP 端点：`POST /checkpoint`、`POST /checkpoint/generate`、`POST /checkpoint/delete`。

## 在 pagoda 里动手试试

```powershell
# 启动服务后：
Invoke-RestMethod http://127.0.0.1:8080/checkpoint -Method Post -ContentType "application/json" `
  -Body '{"text":"你是客服机器人。规则：……（很长的共享 prompt）"}'
# → {"checkpoint_id":0}

Invoke-RestMethod http://127.0.0.1:8080/checkpoint/generate -Method Post -ContentType "application/json" `
  -Body '{"checkpoint_id":0,"text":"用户：我要退货","sampling_params":{"max_tokens":20}}'
# → 返回里 prefix_hit_tokens = 共享 prompt 的全部 token 数
```

行为规格见 `tests/checkpoint_tests.rs`：pin 抗淘汰、drop 语义、SLO 护栏、
分支零污染、与 APC 后端兼容。

## 和普通前缀缓存的区别

| | 前缀缓存（radix/APC） | checkpoint |
| --- | --- | --- |
| 复用方式 | 被动：请求碰巧撞上历史 | 主动：显式创建、显式引用 |
| 抗淘汰 | 不保证（LRU 可能收走） | 保证（pin 免疫淘汰） |
| 命中查找 | 走树/哈希表 | 零查找，直接用存的槽位 |
| 适用 | 通用流量 | agent 集群、树搜索、长共享 prompt |

## 常见疑问

**Q：checkpoint 会泄漏内存吗？**
pin 住的块对淘汰免疫，所以 checkpoint 本身是"用户管理的内存"——忘了 drop 就一直占着。
`/stats` 的 `active_checkpoints` 可以监控。引擎级单测
`completed_branch_leaves_no_dangling_references` 保证**分支**不会泄漏引用。

**Q：checkpoint 和 SGLang 的 session 有什么区别？**
SGLang 的 RadixAttention 靠"同一会话的请求天然共享前缀"被动复用；checkpoint 把这个
行为显式化、可编程化，并加了抗淘汰保证。vLLM 目前没有等价物。