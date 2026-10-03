# 12 · 与 SGLang 一键共部署：pagoda 当网关，SGLang 当引擎（P4）

## 一句话总结

一条命令同时拉起两个进程：**pagoda 守大门**（控制面：健康检查、指标、
checkpoint、准入护栏），**SGLang 干重活**（GPU 大模型生成）。
客户端只认 pagoda 的端口，根本不知道后面站着谁。

## 生活化类比

餐厅的前厅与后厨：

- **前厅经理（pagoda）**：迎客、点单、记账、回答"还要等多久"。
  轻量、永远在线、单二进制启动毫秒级。
- **后厨大师傅（SGLang）**：火力全开的 GPU 灶台，菜（token）做得又快又好，
  但你不会让他去门口迎宾——python 技术栈干重活合算，守门太浪费。

前厅与后厨用一张小票（HTTP JSON）沟通。客人只跟前厅打交道。

## 它是怎么工作的

```
            ┌─────────────────────────┐
 客户端 ──► │  pagoda 网关 :30000      │
            │  本地: /health           │
            │       /stats            │← 指标里带 upstream 和
            │       /checkpoint*      │  proxied_requests（转发了多少单）
            │  转发: /generate ───────┼──► SGLang worker :30001
            │       /v1/chat/*  ──────┤    （GPU，大模型）
            └─────────────────────────┘
```

- 转发是**逐字节透传**：请求体原样发给上游，响应原样发回客户端，
  pagoda 不解析、不改写；请求带 `"stream":true` 时逐 chunk 中继 SSE。
- 上游挂了？客户端收到干净的 `502 upstream_unreachable`，网关自己不崩。
- pagoda 侧的引擎照常空转待命——网关模式下它不占算力，只为控制面服务。

## 动手试试

```powershell
# Windows（首次加 -InstallSglang 自动装 SGLang）
cd pagoda
powershell -ExecutionPolicy Bypass -File scripts\co-deploy.ps1 -InstallSglang

# Linux / macOS
INSTALL_SGLANG=1 bash scripts/co-deploy.sh
```

脚本做四件事：构建 pagoda →（可选）venv 里 pip install sglang →
起 SGLang worker 并等健康检查 → 起 pagoda 网关并等健康检查。
完成后冒烟测试：

```bash
curl -X POST http://127.0.0.1:30000/generate -H "Content-Type: application/json" \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16}}'
curl http://127.0.0.1:30000/stats   # 看 proxied_requests 计数
```

手动模式（已经各自在跑，只想接管流量）：

```bash
cargo run --release --bin pagoda -- serve --port 30000 --upstream http://127.0.0.1:30001
```

## 什么时候用哪种部署

| 场景 | 推荐 |
| --- | --- |
| 小模型 / CPU / 边缘 | pagoda 单飞（pagoda-hf 后端），不需要 SGLang |
| 大模型 GPU 吞吐优先 | SGLang 单飞，或本方案（多一层网关换控制面） |
| agent 集群（checkpoint/嫁接 + 重生成混部） | **本方案**：pagoda 管 agent 原语，SGLang 扛生成 |
| 想白嫖 pagoda 的指标与准入护栏 | 本方案，网关税约 0.1ms/请求（本机 loopback） |

## 和 SGLang 官方 router 的对照

SGLang 官方 `sglang-router` 也是 Rust 写的，思想一致（Rust 控制面 +
python worker）。pagoda 网关的差异：**不止路由**——checkpoint 生命周期、
张量嫁接指标、准入护栏、约束解码预检都在网关侧，且零依赖可离线编译，
嵌进任何环境不拖 python 运行时。

## 常见疑问

**Q：转发会不会成为瓶颈？**
A：loopback 上一次 JSON 转发约 0.1ms 级，相比 GPU 生成的数百毫秒/token
可以忽略。真正的瓶颈从来是模型前向，不是网关。

**Q：流式输出（SSE）呢？**
A：已支持。请求体带 `"stream":true` 即可：上游的 SSE 事件逐 chunk 中继，
打字机效果零缓冲延迟；上游不流式时自动回退整段缓冲，两种模式共用
一条代码路径。`/stats` 里 `streamed_requests` 可观测。

**Q：为什么不用 nginx / envoy 转发？**
A：可以，而且生产环境推荐。这个内置代理的价值在**一键与零依赖**：
开发机、边缘盒子、教学环境里没有 nginx，`cargo run` 就有网关。