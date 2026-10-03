# 14 · 一键部署 Laya：一个二进制就是全部

## 一句话总结

官方 Laya 是 Python 库：装它要拖上 PyTorch、transformers 和几十个依赖（实测
venv **5.3 GB / 39 个包**），还要赌 torch×transformers 的版本矩阵不打架。
pagoda 的 Rust 版 Laya 是**一个 17.9 MB 的二进制**：拷过去、跑起来、完。
一条脚本命令完成构建、启动、健康检查和冒烟验证。

## 生活化类比

搬家的两种方式：

- **Python 部署 = 整屋搬运**：torch 是钢琴，transformers 是组合柜，
  39 个箱子一整车。到了新家（新机器）还要摆对位置——版本对不上，
  钢琴进不了门（我们实测：transformers 4.46 不认识 modernbert；
  换成 4.48 又嫌 torch 2.0 太旧，两个坑都是真实踩出来的）。
- **Rust 单二进制 = 拎包入住**：一个 17.9 MB 的包，里面什么都有，
  放哪台机器哪台就能跑。没有"在我机器上是好的"。

## 它是怎么工作的

```
scripts/serve-laya.sh（或 .ps1）一条命令做四件事：
──────────────────────────────────────────────────
[1/3] cargo build --release --bin laya_server
        → 产出单个可执行文件（CPU 版连 CUDA 工具链都不需要）
[2/3] 启动服务，监听 127.0.0.1:8081
        → 首次运行自动从 Hugging Face 拉权重（约 842MB），之后秒开
[3/3] 轮询 GET /health 直到就绪
--smoke → 打一发 README 账单场景，断言路由到 billing，错了直接报错退出

之后任何系统（Jev 或别的）直接 HTTP 调用：
  POST /decide
  {"state": "<现场文本>",
   "questions": {"部门": {"type": "choice", "instructions": "...", "criteria": {...}},
                 "紧急度": {"type": "score", ...},
                 "流失风险": {"type": "noul", ...}}}
```

## 在 pagoda 里动手试试

```powershell
# Windows
cd pagoda-hf
powershell -ExecutionPolicy Bypass -File scripts\serve-laya.ps1 -Smoke
```

```bash
# Linux / macOS（有 N 卡加 --cuda，需 CUDA 12.x 工具链）
cd pagoda-hf
bash scripts/serve-laya.sh --smoke            # CPU
bash scripts/serve-laya.sh --smoke --cuda     # GPU
```

手工调一发（服务就绪后）：

```bash
curl -X POST http://127.0.0.1:8081/decide -H "Content-Type: application/json" -d '{
  "state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}
  }
}'
```

返回（与官方 Python 版逐位一致）：

```json
{"model": "rl-agent",
 "answers": {
   "department": {"type": "choice", "choice": "billing",
                  "probabilities": {"billing": 0.9865, "technical": 0.008, "other": 0.0055},
                  "confidence": 0.9267, "rl_agent": {"act_probability": 1.0}},
   "churn_risk": {"type": "noul", "noul": 0.879, "confidence": 0.4678,
                  "rl_agent": {"act_probability": 1.0}}},
 "usage": {"input_tokens": 115, "output_tokens": 0}}
```

## 和官方部署的对照（同机实测，2026-10-03）

| | 官方 Python 版 | pagoda Rust 版 |
| --- | --- | --- |
| 部署物 | venv **5.3 GB**（torch+transformers+39 包） | **单二进制 17.9 MB**（约 1/300） |
| 运行时依赖 | Python 3.10+、torch≥2.1.1、transformers≥4.48 | 无（ldd 仅 libc 等 8 行系统库） |
| 模型加载 | 7.9–9.4 s | **1.4–1.6 s**（5–6.5 倍） |
| 单请求延迟（RTX 3050） | 691 ms（fp16） | **606 ms（f32，精度更高仍更快）** |
| 单请求延迟（纯 CPU） | **471 ms**（torch oneDNN） | 1062 ms（诚实差距，见下） |
| 峰值内存（GPU） | 2954 MB | **1850 MB（省 37%）** |
| 跨设备一致性 | CPU↔GPU 概率漂移 1e-4，旧卡强制 fp16 | **决策逐位一致**（漂移 ~5e-6） |
| 输出兼容性 | —— 基准 —— | 与官方 API 响应**逐字段一致**（实测对拍） |

诚实说明：纯 CPU 且对单请求延迟敏感的场景，目前 torch 的 oneDNN 内核更快
（差距 2.25 倍），对应路线图"CPU BLAS/MKL 后端"。GPU、冷启动、内存、
部署与可靠性维度全面领先。完整方法与数据见 [../BENCHMARK-LAYA.md](../BENCHMARK-LAYA.md)。

## 常见疑问

**Q：没有 GPU 能跑吗？**
A：能，默认就是 CPU。构建时不加 `--features cuda`，编译出来的二进制
连 CUDA 工具链都不依赖，任意 x86_64 Linux / Windows 机器拷过去就跑。

**Q：和 Jev 怎么对接？**
A：`/decide` 的请求/响应形状与 Jev 的 rl-agent 接口一致（`state` +
带类型的 `questions`，返回 `answers` + `usage`）。把 Jev 里 rl-agent 的
URL 指向 `http://<本机>:8081/decide` 即可。

**Q：坏请求会把服务打挂吗？**
A：不会。JSON 解析失败、缺字段、未知题型都返回 400 + 可读的错误说明
（如 `unknown question type "bogus" (want choice|score|noul)`），
进程继续服务。类型系统在反序列化层就拦住了畸形输入。

**Q：多语言呢？**
A：模型仓库自带 `multilingual/` 检查点（100+ 语言、8K 上下文）。
pagoda 当前加载的是根目录英文检查点，子目录检查点的加载支持在路线图上。
