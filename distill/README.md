# pagoda-distill：一键蒸馏框架（真实业务场景演示）

**业务场景：跨境电商客服工单自动结构化。** 客户来一封消息，系统要给出
部门路由、紧急度、情绪、订单号、退款金额、是否需要人工、回复草稿——
原来由付费大模型逐条处理（贵、慢、数据出域），蒸馏后由本地 0.5B 小模型
以 SGLang 承载（便宜、快、数据不出域）。

## 流水线

```
seeds/tickets.jsonl          24 条手工种子工单（5 类业务）
      │
      ▼  gen_data.py   教师（付费 API 或本地大模型）改写扩增 + 结构化标注
dataset.jsonl          ~96 条带标准答案的训练数据（按种子分组切分，防泄漏）
      │
      ▼  train_lora.py LoRA SFT（completion-only loss，r=16，bf16）
out/student-merged     蒸馏后学生（合并权重，可直接部署）
      │
      ▼  eval.py       逐字段 exact-match：base vs distilled vs 教师
out/eval_report.json
      │
      ▼  SGLang launch_server
http://127.0.0.1:30002/v1   OpenAI 兼容服务（可挂 pagoda 网关）
```

## 一键运行

```bash
# A. 付费/私有教师（生产推荐；DeepSeek/通义/私有 vLLM 同理，换 base_url 即可）
export TEACHER_BASE_URL="https://api.openai.com/v1"
export TEACHER_API_KEY="sk-..."
export TEACHER_MODEL="gpt-4o-mini"
bash run-distill.sh

# B. 本地教师（零成本演示：SGLang 起 Qwen2.5-3B-Instruct-AWQ 当教师）
bash run-distill.sh --local-teacher
```

幂等：数据集/权重/报告存在即跳过对应阶段，`--force` 强制重跑。
Windows 用 `run-distill.ps1`（参数 `-LocalTeacher` / `-Force`）。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TEACHER_BASE_URL` | `http://127.0.0.1:30001/v1` | 任意 OpenAI 兼容端点 |
| `TEACHER_API_KEY` | `EMPTY` | 付费教师填真实 key |
| `TEACHER_MODEL` | （必填） | 如 `gpt-4o-mini`、`deepseek-chat` |
| `STUDENT_BASE` | `Qwen/Qwen2.5-0.5B-Instruct` | 学生基座 |
| `VARIATIONS` | `3` | 每条种子的教师改写数 |
| `EPOCHS` | `3` | LoRA 训练轮数 |
| `PYTHON` | `python3` | 解释器（指向含依赖的 venv 即可） |

## 为什么这样设计

- **教师接口只有 OpenAI 兼容 chat**：付费 API、私有部署、本地 SGLang 同一份代码，
  换环境变量即换教师，不绑死任何供应商。
- **每条教师输出过 schema 校验**：不合格直接丢弃计数，坏数据不进训练集。
- **按种子分组切分 train/eval**：同一工单的变体不会一边训练一边考试，评估数字可信。
- **completion-only loss**：prompt 部分打 -100，模型只学"怎么答"，不学"怎么问"。
- **学生部署用 SGLang 而不是裸 transformers**：学生是自回归模型，prefill/decode
  的调度、KV 分页、前缀缓存正是 SGLang 的主场（详见
  `../pagoda/docs/guide/15-distill.md` 对"什么时候需要 SGLang"的拆解）。
## 共部署（训练完之后）

权重已在 `out/student-merged` 时，用 `codeploy.py` 直接上线，不重训：

```bash
python codeploy.py                  # SGLang 起学生(:30002) + 3 张工单演示 + 保持运行
python codeploy.py --attach         # 学生已在跑，只接上去演示
python codeploy.py --exit-after-demo  # 演示完即退（CI 冒烟用）
```

演示链路：客户消息 -> 学生输出结构化 JSON -> 路由层分流
（`needs_human=true` 或 `urgency=2` 转人工；其余 reply 草稿进自动回复队列）。
学生是自回归小模型，生成走 prefill/decode，所以部署层用 SGLang 承载；
教师只在训练期出现，部署期不在链路里。Windows 用 `codeploy.ps1`，参数同。