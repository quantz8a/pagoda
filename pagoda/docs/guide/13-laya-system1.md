# 13 · Laya：不生成文本的"决策模型"（System 1）

## 一句话总结

不是所有模型都用来"写话"。**Laya 只做判断**：给它一段现场（工单、邮件、JSON）和
几个带类型的问题（选一个 / 打分 / 是否），**一次前向**就返回答案和校准过的概率——
不生成一个字，所以没有幻觉可捉。

## 生活化类比

急诊室的分诊台护士 vs 主治医生：

- **分诊台（Laya，System 1）**：病人进门，护士 30 秒内判断"挂哪个科、急不急"。
  快、便宜、一天几千次。判断错了代价小（医生会复核）。
- **主治医生（大模型，System 2）**：慢慢问诊、写病历。贵、慢、一次几分钟。

聪明的医院不会让医生去干分诊的活。聪明的 agent 系统也不该让 GPT 级模型
去判断"这封邮件该不该转给账单组"——那是 Laya 的活。

## 它是怎么工作的

```
输入: state（现场文本）+ questions（带类型的问题清单）
──────────────────────────────────────────────────
每个问题拼成一条序列:
  [CLS] "choice question: 该转给哪个部门?" [SEP]
        [MASK] billing: 发票付款退款   ← 每个选项一个 [MASK] 哨兵
        [MASK] technical: 故障报错
        [MASK] other: 其他
        [SEP] state 原文 [SEP]
ModernBERT 编码器（双向注意力，28 层）
  → 决策头（2 层 transformer 只看哨兵位置）
  → 每个选项一个分数 → softmax
  → 温度校准（RLCD 训练：报真实概率才有奖励，概率是"诚实"的）
输出: choice=billing  概率分布  confidence  act_probability
```

三种题型：**choice**（N 选一）、**score**（有序打分，输出期望档位）、
**noul**（是否题，输出"成立"的概率）。

## 在 pagoda 里动手试试

```powershell
# 真实权重端到端（联网，首次下载约 842MB）
cd pagoda-hf
cargo run --release --example e2e_laya
#   department: choice=billing confidence=0.927
#   urgency:    score=1.772（满分 2，" blocking" 档）
#   churn_risk: noul=0.879（识别出"取消订阅"威胁）
```

代码里只要：

```rust
let laya = Laya::from_hub("convaiinnovations/laya", Device::Cpu)?;
let d = laya.decide(state, &[("department".into(), Question::choice("...", &opts))])?;
```

## 为什么对 pagoda 重要（System 1 + System 2 架构）

pagoda 的网关位置（第 12 篇）天然适合放 Laya：

```
请求 → pagoda 网关
        ├─ Laya 分诊（CPU，几十毫秒）：安不安全？转哪个模型？急不急？
        ├─ 危险/超纲   → 直接拦截或降级
        ├─ 小活        → pagoda-hf 本地小模型
        └─ 重活        → 转发 SGLang / 大模型
```

路由、护栏、审核这三件 agent 时代的刚需，都是"判断"而不是"生成"——
用生成模型干判断，又贵又慢还得解析它的输出；Laya 一次前向直接给概率。

## 和参考实现的对照

- 参考实现是 PyTorch（`rl_agent_api.py`）；pagoda-hf 的移植用 candle，
  逐模块对齐：序列拼装（`build_sequence`）、决策头（norm_first transformer
  层 + scorer + act head）、按题型分桶的温度校准，全部逐一对应。
- 数值纪律：pagoda 全设备 F32 精确计算（参考实现 GPU 上默认 fp16 autocast，
  对算力 <8.0 的卡强制 fp16——我们反而更精确）；两次运行结果**逐 bit 一致**（e2e 硬断言）。

## 性能实测（vs 官方 Python 版）

同机基准详见 [../BENCHMARK-LAYA.md](../BENCHMARK-LAYA.md)，头条数字：

- GPU 单请求 pagoda（f32）606 ms，官方（fp16）691 ms——**精度吃亏仍快 12.3%**；
- 冷启动 1.4–1.6 s vs 7.9–9.4 s（快 5–6.5 倍），峰值内存最多省 37%；
- 部署面 17.9 MB 单二进制 vs 5.3 GB Python venv（约 300 倍）；
- 跨设备一致：pagoda 的 CPU/GPU 决策逐位一致，官方路径 CPU↔GPU 概率漂移 1e-4；
- 诚实差距：CPU 纯算力落后 torch oneDNN 约 2.25 倍。

## 常见疑问

**Q：概率"校准"是什么意思？**
A：说 80% 的事真的 80% 发生。Laya 训练时用严格适当评分规则（RLCD）——
只有报真实概率才能拿满分，再加事后温度拟合。所以它的 0.879 可以当真用。

**Q：能多语言吗？**
A：模型支持 100+ 语言（`laya-multilingual` 检查点，8K 上下文）。
当前 pagoda-hf 加载的是英文检查点，多语言版在路线图上。

**Q：为什么不用 candle 现成的 `ModernBertForSequenceClassification`？**
A：Laya 的头不是普通分类头——它是"每个 [MASK] 哨兵一个分数"的决策头
（外加 act head 和题型嵌入），按参考实现逐层手搓并对齐权重名。
