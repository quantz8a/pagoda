# SGLang 融合指南（Perfect Fusion）

pagoda 与 SGLang 的关系不是"替代"，而是**同构**：相同的 API 形状、相同的 DSL、
相同的架构术语。你可以把 pagoda 当作 SGLang 的离线沙盒、教学镜像和 agent 原语层。

## 三种融合方式

### 方式一：API 直通（客户端零改动）

SGLang 服务的客户端代码，把 `base_url` 指到 pagoda 即可工作：

```python
# 原来连 SGLang 的代码，一字不改
import openai
client = openai.OpenAI(base_url="http://127.0.0.1:8080", api_key="none")
client.chat.completions.create(
    model="pagoda-toy",
    messages=[{"role": "user", "content": "hello"}],
)
```

兼容矩阵：

| SGLang 端点/参数 | pagoda 状态 |
| --- | --- |
| `POST /generate`（`text` + `sampling_params`） | ✅ 原生实现 |
| `POST /v1/chat/completions`（`messages`） | ✅ OpenAI 兼容形状 |
| `max_tokens / temperature / top_p / top_k` | ✅ 同名同语义 |
| `stop / stop_token_ids` | ✅ |
| `frequency_penalty / presence_penalty` | ✅ |
| `sampling_params.grammar`（json / regex） | ✅ pagoda 扩展，零依赖 |
| `GET /health`、`GET /stats` | ✅ |
| streaming（SSE） | ❌ 待做 |
| `model` 路由、LoRA | ❌ 单模型运行时 |

### 方式二：DSL 直通（程序零改动）

SGLang 的前端程序范式在 pagoda 里有同构实现：

| SGLang `sglang.lang` | pagoda `Program` |
| --- | --- |
| `@function` + `s += sgl.system(...)` | `p.system(...)` |
| `sgl.gen("name", max_tokens=..)` | `p.gen("name", params)` |
| `sgl.select("name", choices=[..])` | `p.select("name", choices, params)` |
| `sgl.fork(n)` | `p.fork(branches)` |

`cargo run --offline --bin pagoda -- program` 跑一个完整示例。

### 方式三：架构映射（读代码零障碍）

改造 SGLang 的人读 pagoda 源码时，每个模块都有精确对应（见 `docs/DESIGN.md` §2
的映射表）：`scheduler` ↔ `engine.rs`、`RadixAttention` ↔ `radix_cache.rs`、
`memory_pool` ↔ `kv_cache.rs`、`sampling_params` ↔ `sampler.rs` + `spec.rs`。

## pagoda 超出 SGLang 的部分

| pagoda 扩展 | 说明 |
| --- | --- |
| `POST /checkpoint` 系列 | agent 集群原语：pin 共享树干、分支 100% 命中。SGLang 无对应物 |
| `CacheBackend::Apc` | 可切换到 vLLM 风格块级缓存，同框架 A/B 对比 |
| 零依赖 grammar | 不依赖 Outlines/xgrammar 的约束解码 |
| 四轴收入指标 | `compute_saved_tokens` 等直接进 `/stats` |

这些扩展全部**向后兼容**：SGLang 客户端不调用就不会碰到。

## 定位声明（避免社区误解）

- pagoda **不是** SGLang 的 fork：没有复制其代码（clean-room 重写，见 NOTICE）。
- pagoda **致敬并署名** SGLang：架构血缘永久记录在 NOTICE 与 DESIGN.md。
- pagoda 与 SGLang 同样采用 Apache-2.0：代码可以互鉴、双向流动，血缘与
  署名师承关系永久记录在 NOTICE。
