# 蒸馏演示运行报告：付费/本地教师 → 本地 0.5B 学生（SGLang 部署）

> 测于 2026-10-03，共享 GPU 开发机（i5-12500 / 31GB / RTX 3050 8GB，与其他租户共享）。
> 框架代码：`distill/`；复现：`bash run-distill.sh --local-teacher`（零成本）
> 或配置 `TEACHER_BASE_URL/TEACHER_API_KEY/TEACHER_MODEL` 用付费大模型。

## 场景

跨境电商客服工单结构化：客户消息 → JSON（department / urgency / sentiment /
order_id / refund_amount / needs_human / reply 草稿）。

## 配置

- 教师：Qwen2.5-1.5B-Instruct（本地 SGLang，OpenAI 兼容端点）——生产环境换
  付费模型只需改三个环境变量，代码零改动。
- 学生：Qwen2.5-0.5B-Instruct + LoRA r=16（训练参数 8.8M = 全量的 1.75%）。
- 教师与学生部署均用 SGLang 0.5.10（torch_native 后端 + pytorch sampling）。

## 数据（gen_data.py）

- 24 条手工种子工单（billing/shipping/technical/product/other 五类）
- 教师改写扩增 ×3 + 教师标注：**91 条通过 schema 校验**（5 条不合格自动丢弃）
- 按种子分组切分：71 训练 / 20 考试（变体不跨组，无泄漏）
- 已知偏差：教师标注的 department 分布偏斜（technical 51 / billing 6 /
  other 0），学生在 department 上的系统性错误部分来源于此。修复路径明确：
  往 `seeds/tickets.jsonl` 补种子后 `--force` 重跑。

## 训练（train_lora.py）

- 3 epochs，有效 batch 16，bf16，completion-only loss（prompt 打 -100）
- 训练时长 **498 秒**（共享 3050 上 12 步 × ~41s），train_loss 2.0 → 0.47
- 产物：LoRA 适配器 + 合并后的独立模型 `out/student-merged/`（约 1GB）

## 评估（eval.py，20 条考试集，逐字段 exact-match）

| 字段 | base 0.5B | 蒸馏后 | 提升 |
| --- | ---: | ---: | ---: |
| department | 15% | **65%** | **+50pp** |
| urgency | 5% | **70%** | **+65pp** |
| sentiment | 50% | 60% | +10pp |
| order_id | 80% | 85% | +5pp |
| refund_amount | 95% | 85% | −10pp（n=20 噪声内，多为 null 易例） |
| needs_human | 25% | 30% | +5pp |
| 整单 JSON 全对 | **0%** | **20%** | **+20pp** |

解读：蒸馏在最难的语义字段（department/urgency）上提升 3–13 倍，
证明管线端到端有效；整单全对率 20% 是起点不是终点——瓶颈在数据量与
标注分布，不在框架（补种子重跑即可提升）。

## 部署冒烟（SGLang，OpenAI 兼容）

学生合并权重由 SGLang 直接加载。未见样本实测（投影仪故障+消协投诉威胁）：

```json
{"department":"technical","urgency":2,"sentiment":"neutral",
 "order_id":"ORD-33456","refund_amount":null,"needs_human":true,
 "reply":"尊敬的客户，感谢您的反馈。我们将尽快处理并联系您进行退款或投诉。"}
```

结构合法、订单号精确抽取、识别出人工介入信号；sentiment/refund 有漏判，
与评估一致。

## 排障实录（复现必读）

1. **SGLang 0.5.10 的 JIT 内核要 nvcc≥12.8 + gcc≥10**：默认 rope/kv-cache
   kernel 走 tvm-ffi JIT。旧 nvcc 报
   `Unknown option '-generate-dependencies-with-compile'`，旧 gcc 报
   `fatal error: concepts`。解决：`PATH` 前置 CUDA 12.8 工具链与 gcc-10。
2. **量化模型在旧驱动栈上额外踩雷**：AWQ 的 marlin repack 同样走 JIT；
   bf16 模型配合 `--attention-backend torch_native --sampling-backend pytorch`
   最稳。
3. **共享 CPU 上 SGLang 逐 token 派发仅 ~2.6 tok/s**：python 调度被其他租户
   抢占所致。造数据阶段必须并发请求（`GEN_WORKERS=12` 后聚合 ~13.6 tok/s，
   5.2 倍）。这也是 BENCHMARK.md 已记录的结论，不是新问题。
4. **显存要算总账**：8GB 卡有其他租户占 2.7GB 时，`--mem-fraction-static`
   按**总显存**比例计算，1.5B bf16 教师需要 0.68 才起得来。
5. **transformers 版本陷阱**：4.x 用 `torch_dtype=`，5.x 用 `dtype=`；
   框架代码用前者（两版兼容）。