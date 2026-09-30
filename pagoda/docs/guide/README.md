# Pagoda 小白文档体系

看不懂 LLM 推理服务？没关系。这一系列文档假设你**只懂最基本的编程概念**，
从零讲到你能看懂 pagoda 的每一行设计决策。

## 阅读顺序

| 顺序 | 文档 | 你会搞懂什么 |
| --- | --- | --- |
| 0 | [一键上手](00-quickstart.md) | 5 分钟跑通：一键脚本 + agent 集群演示 |
| 1 | [LLM 推理服务是什么](01-llm-serving-basics.md) | prefill / decode、为什么服务化很烧钱 |
| 2 | [KV Cache 与分页内存](02-kv-cache-paging.md) | KV cache、PagedAttention、引用计数、写时复制 |
| 3 | [Radix 前缀缓存](03-radix-prefix-cache.md) | RadixAttention、为什么"记住前缀"等于省钱 |
| 4 | [APC 块级哈希缓存](04-apc-block-cache.md) | vLLM 风格 APC、和 radix 的取舍 |
| 5 | [Checkpoint 与分支](05-checkpoint-branching.md) | agent 集群 / 树搜索怎么共享算力 |
| 6 | [约束解码](06-constrained-decoding.md) | 让模型只能输出合法 JSON / 正则匹配 |
| 7 | [调度与指标](07-scheduling-and-metrics.md) | continuous batching、chunked prefill、四轴指标 |
| 8 | [增量 KV 会话](08-kv-session.md) | 每个 token 只算一次：O(n²) 全量重放 → O(n) 会话 + KV 分叉 |
| 9 | [开源协议：为什么 AGPL](09-open-source-license.md) | 协议怎么防抄袭、我能怎么用它 |

## 怎么用这份文档

- **只想用 pagoda**：读 1 → 7 就够，其他按需。
- **想改代码**：按顺序读完，再去读 `../DESIGN.md`（设计文档，面向工程师）。
- **每篇的结构都一样**：
  1. 一句话总结
  2. 生活化类比
  3. 它是怎么工作的（有图）
  4. 在 pagoda 里动手试试（可运行的命令/代码）
  5. 和 SGLang / vLLM 的对照
  6. 常见疑问

## 跑起来的最小命令

```powershell
cd pagoda
powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1   # 一键：构建+测试+演示
cargo test --offline                                        # 78 项测试全绿即环境 OK
cargo run --offline --bin pagoda -- sample -p "你好" --repeat 2
cargo run --offline --bin pagoda -- serve --port 8080       # HTTP 服务
```
