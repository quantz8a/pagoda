# 15 · 一键蒸馏：付费大模型教出本地小模型

## 一句话总结

把"贵但强"的大模型（GPT / DeepSeek / 私有部署）当老师，把"便宜能跑"的
0.5B 小模型当学生：老师自动造数据 → 学生 LoRA 学习 → 逐字段考试 →
SGLang 一键部署。一条脚本跑完全程，之后客服工单结构化这件事就**不再花
API 的钱、数据也不出域**。

## 先回答两个关键问题

### "不用 prefill 和 decode，还需要 SGLang 吗？"

**不需要。** 但要先搞清楚 prefill/decode 是什么——它们不是 SGLang 的功能，
而是**自回归生成范式**的两个阶段：

```
prefill: 一次性吃完整个 prompt，算出每个位置的 KV 缓存
decode:  逐 token 吐字，每个新 token 依赖之前所有 KV
```

SGLang 的全部价值（RadixAttention 前缀缓存、KV 分页、PD 分离、连续批处理）
都是"把 prefill/decode 跑到极致"。**如果你的任务根本不需要逐 token 生成，
这些优化就没有用武之地**：

| 任务形态 | 有没有 prefill/decode | 该用什么 |
| --- | --- | --- |
| 判断题（分类/打分/是否） | **没有**——单次前向出结果 | pagoda 的 Laya（guide 13/14），或本仓库归档的 TF-IDF 头（`distill/legacy-tfidf/`） |
| 生成题（回复草拟/JSON 抽取/写作） | **有**——天生逐 token | SGLang 部署蒸馏小模型（本篇） |

### "有使用 prefill 和 decode 的小模型吗？"

**所有自回归模型都用 prefill/decode，与大小无关。** Qwen2.5-0.5B、
Llama-3.2-1B 这些小模型，生成每个字照样要 prefill（吃 prompt）+
decode（逐字吐）。"小"只意味着算力需求低、能跑在消费级显卡上——
而"跑得省"恰恰更需要 SGLang：小模型的单请求延迟低，**并发与 KV 管理
效率**就是吞吐的全部来源。

## 生活化类比

老中医带徒弟：

- **老师（付费大模型）**：医术高明但挂号费贵、还要去医院（数据出域）。
  不能天天麻烦他，只请他做一件事——**写教案**（标注数据）。
- **学生（本地小模型）**：把教案背到滚瓜烂熟（LoRA 微调），之后在
  社区诊所独立坐诊（本地 SGLang 服务）。看不了的疑难重症
  （needs_human / 低置信）再转诊给老师或人工。

关键数字：老师只出场一次（造约百条教案，几分钱或零成本），
学生之后**每次调用都是免费的本地推理**。

## 它是怎么工作的

```
distill/ 一条命令 bash run-distill.sh（或 --local-teacher 零成本演示）
──────────────────────────────────────────────────────────
[1/6] 环境自检：openai / torch / peft / transformers
[2/6] 教师造数据 gen_data.py
        24 条手工种子工单 ──教师改写──→ ×3 变体 ──教师标注──→ ~96 条
        每条输出过 schema 校验，不合格丢弃；按种子分组切 train/eval（防泄漏）
[3/6] LoRA 蒸馏 train_lora.py
        Qwen2.5-0.5B + LoRA r=16（1.75% 参数），completion-only loss：
        prompt 打 -100，只学"怎么答"不学"怎么问"
[4/6] 合并权重 → out/student-merged（独立模型，不再依赖 LoRA 运行时）
[5/6] 逐字段考试 eval.py：base vs distilled，对照教师金标准
        department/urgency/sentiment/order_id/refund_amount/needs_human
        逐字段 exact-match + 整单 JSON 全对率
[6/6] SGLang 部署学生 + 冒烟：OpenAI 兼容端点直接可用
```

教师接口只有一种形状：**OpenAI 兼容 chat API**。所以付费 API
（`TEACHER_BASE_URL=https://api.openai.com/v1`）、私有 vLLM、本地 SGLang
教师，走的是同一份代码——换环境变量就换老师。

## 动手试试

```bash
cd distill

# A. 付费教师（生产）：三分钱级别的数据成本
export TEACHER_BASE_URL="https://api.openai.com/v1"
export TEACHER_API_KEY="sk-..."
export TEACHER_MODEL="gpt-4o-mini"
bash run-distill.sh

# B. 本地教师（零成本演示）：SGLang 起 Qwen2.5-1.5B 当老师
bash run-distill.sh --local-teacher
```

## 真实运行记录（2026-10-03，共享 GPU 开发机 RTX 3050 8GB）

完整报告见 [../DISTILL-REPORT.md](../DISTILL-REPORT.md)。头条：

- 教师 Qwen2.5-1.5B（本地 SGLang）→ 91 条校验合格样本（71 训练 / 20 考试）；
- LoRA 训练 498 秒；department 字段准确率 15% → 65%，urgency 5% → 70%，
  整单 JSON 全对率 0% → 20%（瓶颈在数据量，补种子 `--force` 重跑即可提升）；
- 学生经 SGLang 以 OpenAI 兼容端点上线，未见样本冒烟：结构合法、
  订单号精确抽取、人工介入信号正确。

踩坑实录（都进了排障文档）：sglang 0.5.10 的 JIT 内核要求 nvcc≥12.8 +
gcc≥10（旧 nvcc 报 `Unknown option '-generate-dependencies-with-compile'`，
旧 gcc 缺 `<concepts>`）；共享 CPU 上 SGLang 逐 token 派发只有 ~2.6 tok/s，
**造数据必须并发**（12 路并行后聚合 ~13.6 tok/s，5.2 倍）。

## 常见疑问

**Q：为什么不直接用 Laya？**
A：分工不同。部门路由/紧急度/是否人工这类**判断**用 Laya 更省（无 P/D、
单二进制、毫秒级）；但"写一段 80 字回复草稿"是**生成**，Laya 不会写字。
真实系统是两层：Laya 分诊（System 1）→ 蒸馏小模型起草（System 2 轻量版）
→ 拿不准的转付费大模型或人工。

**Q：学生能换成别的模型吗？**
A：`STUDENT_BASE` 环境变量换成任何 HF 因果语言模型即可
（Qwen2.5-1.5B、Llama-3.2-1B……）。显存够大就能换。

**Q：蒸馏合规吗？**
A：取决于教师 API 的服务条款（多数付费 API 允许用输出训练自有模型，
但禁止训练与其竞争的通用模型——领域专用小模型通常在允许范围内，
商用前请核对条款）。本地开源教师（如 Qwen）无此顾虑。

**Q：数据量这么小够吗？**
A：窄域结构化任务够演示和起步（格式合规 + 字段抽取是低自由度技能）。
要追平老师的泛化能力需要更多种子与多轮迭代——框架支持往
seeds/tickets.jsonl 里加条目后 `--force` 重跑。
