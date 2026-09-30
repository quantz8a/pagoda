# pagoda 需求分析（v2 · 基于 "COMPUTE IS REVENUE" 图识别的问题）

> 输入：一张 AI 基础设施战略图（标题 **COMPUTE IS REVENUE**，四支柱
> **EXTREME CO-DESIGN / SCALE & EXPERIENCE / CUDA ECOSYSTEM / DEMAND**，
> 绑定指标 **TTFT / Token-Watt / MTBI / Useful Life**，中心落点 **AI FACTORY**）。
> 本文件把图中的问题翻译成可验收的工程需求，并标注实现状态。

## 1. 需求来源：图 → 问题 → 需求

| 图中主张 | 识别出的问题 | 对应需求簇 |
| --- | --- | --- |
| COMPUTE IS REVENUE | 重复 prefill / 重复物化 KV = 收入漏损，需可度量 | FR-1、FR-5 |
| EXTREME CO-DESIGN（TTFT, Token/Watt） | 时延与能效是两条独立硬轴，缺逐 token 成本计量 | FR-1..FR-6 |
| SCALE & EXPERIENCE（TTFT, MTBI） | 固定接纳、无策略调度、稳定性未产品化 | FR-3、FR-7..FR-9 |
| CUDA ECOSYSTEM（Useful Life） | 必须即插即用，不能另起炉灶 | FR-10..FR-12 |
| AI FACTORY | 缺准入/淘汰/利用率/SLO，调度器只是演示 | FR-3、FR-6、FR-13 |

## 2. 功能需求（FR）

| ID | 需求 | 优先级 | 验收标准 | 状态 |
| --- | --- | --- | --- | --- |
| FR-1 | 前缀缓存跳过重复 prefill（compute-skip） | P0 | 同一 prompt 第二次请求 `forward_count == 输出 token 数` | ✅ 已实现 |
| FR-2 | 前缀物理零拷贝复用（inc_ref → COW → 释放） | P0 | 复用长 prompt 的第二次请求触发更少 KV 分配 | ✅ 已实现 |
| FR-3 | 可配置调度策略：FCFS / LongestPrefix / ShortestPrompt | P0 | LongestPrefix 下命中缓存请求先于未命中请求完成出队 | ✅ 已实现（v2） |
| FR-4 | 分页 KV + 写时复制，控制显存密度 | P0 | 共享块分叉后私有页写不污染共享页（单测覆盖） | ✅ 已实现 |
| FR-5 | 四轴收入指标（compute_saved / prefill_skip / forward-per-token / 利用率） | P0 | `/stats` 输出四轴字段；重复请求 `compute_saved_tokens > 0` | ✅ 已实现（v2） |
| FR-6 | KV LRU 淘汰，池耗尽时自动回收最久未用前缀 | P1 | `radix.evict_lru` 回收缓存持有块；`evict_on_pressure` 兜底 | ✅ 已实现（v2，原语 + 兜底） |
| FR-7 | 确定性可复现（同 seed 同输出） | P0 | 集成测试 `generation_is_deterministic_for_a_seed` 通过 | ✅ 已实现 |
| FR-8 | 请求级 fault isolation（坏请求不污染共享页/批） | P1 | 单请求崩坏不影响缓存与其它 batch | ⏳ 待实现 |
| FR-9 | 调度器 / radix 的 property-based 测试与 fuzz | P1 | 随机操作序列满足引用计数不变量 | ⏳ 待实现 |
| FR-10 | OpenAI 兼容端点 + SGLang DSL 兼容 | P0 | `/v1/chat/completions`、`gen/select/fork` 端到端可用 | ✅ 已实现（最小） |
| FR-11 | 可插拔后端 trait（`ModelEngine::forward`） | P1 | 替换为真实后端不改调度逻辑 | ✅ trait 占位，⏳ 待接真实权重 |
| FR-12 | 兼容性回归测试（OpenAI/DSL 契约） | P1 | 变更不破坏既有端点行为 | ⏳ 待补充 |
| FR-13 | 准入控制 / 排队 / SLO 护栏 / 利用率全景 | P1 | batch 大小、队列长度、P95 时延可观测 | ⏳ 待实现（`max_running_requests` 已部分具备） |

## 3. 非功能需求（NFR）

| ID | 需求 | 验收标准 | 状态 |
| --- | --- | --- | --- |
| NFR-1 | 内存安全：无 `unsafe` | `src/` 全量无 `unsafe` 关键字 | ✅ |
| NFR-2 | 零第三方依赖、可离线构建 | `cargo build --offline` 成功 | ✅ |
| NFR-3 | 可观测性：stats 覆盖四条收入轴 | `/stats` 返回四轴字段 | ✅（v2） |
| NFR-4 | 性能：release LTO + codegen-units=1 | `Cargo.toml` 配置生效 | ✅ |
| NFR-5 | 可移植寿命：纯 std、可长期维护 | 不引入平台锁定依赖 | ✅ |

## 4. 四轴 → 可测信号 → 代码映射

| 收入轴 | pagoda 信号 | 代码位置 |
| --- | --- | --- |
| TTFT | `prefix_hit_tokens` / `prefill_skip_ratio` / `prefill_chunks` | `radix_cache.rs` · `engine.rs` |
| Token/Watt | `avg_forward_per_output_token` / `kv_utilization` / `kv_pages_reused` | `engine.rs` · `kv_cache.rs` |
| MTBI | 无 unsafe + 确定性重放 + 测试覆盖 | `lib.rs` · `engine_tests.rs` |
| Useful Life | OpenAI/DSL 兼容 + 后端 trait 数 | `server.rs` · `dsl.rs` · `model.rs` |

## 5. v2 增量（本轮已完成）

- `SchedulePolicy`（FCFS / LongestPrefix / ShortestPrompt）与 `EngineConfig.schedule_policy`。
- `EngineStats` 扩展 + 派生指标（`compute_saved_tokens / prefill_skip_ratio / avg_forward_per_output_token / kv_utilization`）。
- `RadixCache::evict_lru` 与 `EngineConfig.evict_on_pressure`。
- `/stats` 与 CLI 输出四轴收入口径。
- 新增测试：`evict_lru_releases_owned_blocks`、`stats_expose_revenue_metrics`、`longest_prefix_policy_schedules_hit_first`；合计 29 项全绿。

## 6. 开放问题与风险

- **真实后端缺失**：Token/Watt 只有「forward 次数」代理，尚无真实每瓦产出；FR-11 落地前能效无法实测定标。
- **evict_lru 为叶子级淘汰**：非叶前缀节点不会被回收（避免切断子孙路径），极端压力下回收率受缓存拓扑影响。
- **fault isolation 待设计**：需定义「坏请求」的边界与回滚语义，避免与零拷贝共享冲突。
