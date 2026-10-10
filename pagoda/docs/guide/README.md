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
| 9 | [开源协议：为什么用 Apache-2.0](09-open-source-license.md) | 宽松开源、怎么自由使用和贡献 |
| 10 | [批量解码](10-batched-decode.md) | 一次前向养活整个批次：权重只读一遍，decode_batch_factor 可观测 |
| 11 | [张量级前缀嫁接](11-tensor-kv-grafting.md) | 真·RadixAttention：KV 张量本体跨请求复用，prompt 物理上零重算 |
| 12 | [与 SGLang 一键共部署](12-sglang-co-deploy.md) | pagoda 守大门（控制面），SGLang 干重活（GPU worker），一条脚本拉起 |
| 13 | [Laya：不生成文本的决策模型](13-laya-system1.md) | System 1 分诊台：choice/score/noul 三题型，一次前向出校准概率 |
| 14 | [一键部署 Laya](14-laya-one-click-deploy.md) | 17.9MB 单二进制替代 5.3GB Python 环境，一条脚本构建+启动+冒烟 |
| 15 | [一键蒸馏：付费大模型教出本地小模型](15-distill.md) | 教师造数→LoRA 蒸馏→逐字段考试→SGLang 部署；顺带说清什么时候根本不需要 SGLang |
| 16 | [分诊网关](16-triage-gateway.md) | System 1 给 System 2 当门卫：Laya 门口分诊，危险工单 0 GPU 转人工，fail-open |
| 17 | [Laya 专科化微调](17-laya-finetune-screening.md) | LoRA 1.8% 参数 + 温度重标定，27 分钟把通用决策模型训成领域筛选员；检查点即训即上线 |
| 18 | [PD 分离：prefill/decode 分开部署](18-pd-disaggregation.md) | Mooncake 风格：KV 对象池当公文柜，prefill 读题 decode 写字，输出与统一版逐 token 相等 |

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
cargo test --offline                                        # 120 项测试全绿即环境 OK
cargo run --offline --bin pagoda -- sample -p "你好" --repeat 2
cargo run --offline --bin pagoda -- serve --port 8080       # HTTP 服务
```
