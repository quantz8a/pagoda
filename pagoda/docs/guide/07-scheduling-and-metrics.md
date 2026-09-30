# 07 · 调度与指标（店长的大脑和账本）

## 一句话总结

调度器决定"先算谁、每步算多少"，指标告诉你"省了多少算力、机器用得多满"。
前者决定用户体验，后者决定老板赚不赚钱。

## 生活化类比

回到奶茶店（第 1 篇）：

- **continuous batching** = 柜台不等一单做完才接下一单，而是哪个步骤有空就插谁的单。
- **chunked prefill** = 超长订单（比如 100 杯）拆成小批做，不让它独占柜台 10 分钟。
- **准入控制** = 门口排队也有上限，排不下就礼貌拒绝，而不是让所有人等死。
- **四轴指标** = 每天的账本：省了多少原料、柜台利用率、平均每杯耗时。

## 它是怎么工作的

每个调度步（`src/engine.rs` 的 `drain` 循环）：

```
┌─ 等待队列（waiting）────────────────────────────┐
│  按策略出队：FCFS / 最长前缀优先 / 最短 prompt 优先 │
└──────┬─────────────────────────────────────────┘
       ▼  每步最多花 max_prefill_tokens_per_step 的预算
┌─ prefill（可分块，跨步继续）────────────────────┐
└──────┬─────────────────────────────────────────┘
       ▼  prefill 完成进入 ready
┌─ decode 批次（active，上限 max_running_requests）┐
│  每条序列本步生成 1 个 token                      │
└──────┬─────────────────────────────────────────┘
       ▼  命中停止条件则结束，释放块、发布前缀进缓存
     完成
```

### 准入护栏（宁可拒绝，不可拖死）

| 护栏 | 配置 | 拒绝原因 |
| --- | --- | --- |
| 排队上限 | `max_waiting_requests` | `queue_full` |
| 单请求总长 | `max_total_tokens` | `too_long` |
| 空 prompt | — | `empty_prompt` |
| 不支持的特性 | — | `unsupported`（如非字节 tokenizer 上挂 grammar） |

被拒绝的请求拿到 `FinishReason::Rejected` 和具体原因，绝不在队列里幽灵排队。

### 故障隔离

某条请求的模型输出出现 NaN / 维度不对 → 该请求单独以 `FinishReason::Fault` 终止，
同批其他请求不受影响（`tests/fault_tests.rs`）。

### 四轴指标（COMPUTE IS REVENUE）

`GET /stats` 的核心字段：

| 指标 | 含义 | 为什么值钱 |
| --- | --- | --- |
| `compute_saved_tokens` | 前缀缓存省掉的 token 数 | 省下的都是纯利润 |
| `prefill_skip_ratio` | prefill 被跳过的比例 | 越高说明缓存越有效 |
| `avg_forward_per_output_token` | 每个产出 token 的前向成本 | Token/Watt 的代理 |
| `kv_utilization` | KV 池驻留率 | 机器用得多满 |

## 在 pagoda 里动手试试

```powershell
cargo run --offline --bin pagoda -- sample -p "观察指标" --repeat 3
# 输出末尾的 revenue: 行有四轴指标

Invoke-RestMethod http://127.0.0.1:8080/stats   # HTTP 版
```

## 和 SGLang 的对照

| | SGLang | pagoda |
| --- | --- | --- |
| continuous batching | ✅ | ✅ 简化版 |
| chunked prefill | ✅ | ✅ 每步 token 预算 |
| 调度策略 | FCFS 为主 | FCFS / 最长前缀 / 最短 prompt |
| 准入与拒绝 | ✅ | ✅ 四种拒绝原因 |
| 故障隔离 | ✅ | ✅ 请求级 Fault |

## 常见疑问

**Q：三个调度策略怎么选？**
客服类"共享长前缀"流量：`LongestPrefix`（先算能蹭缓存的）。
追求低尾延迟：`ShortestPrompt`（短任务先走）。
不知道选啥：`Fcfs`，公平且可预期。

**Q：KV 池满了会拒绝请求吗？**
默认不——先淘汰缓存里最久没用的前缀块（`evict_on_pressure: true`，循环淘汰到
腾出块为止）。只有池子被在飞请求和 pin 住的 checkpoint 真正占满时才会报
"KV cache exhausted"。