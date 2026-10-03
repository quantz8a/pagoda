# 17 · Laya 专科化：30 分钟训出一个领域筛选员

## 一句话总结

`distill/train_laya.py` 把通用决策模型 Laya 微调成了"你这个课题的专科筛选员"：
**LoRA 只动编码器 1.8% 的参数 + 决策头全量训练 + 留出集温度重标定**，
一张和别人共享的 RTX 3050 上 27 分钟训完，导出的检查点直接能被
Rust 单文件 `laya_server --model-dir` 加载上线——从训练到部署没有一行 Python 依赖留在服务端。

## 生活化类比

专科医生的规培。Laya 出厂时是"全科实习医生"：什么都懂一点（通用决策），
但遇到你科室的疑难病例（领域边界案例）只能抛硬币。规培的做法不是重新读医学院
（全量微调），而是**跟着主任出门诊**（LoRA 小剂量）+ **出科考**（留出集评估）
+ **校准自我感觉**（温度重标定：说"九成把握"时真的十次对九次）。
规培完发一张专科执照（新检查点），直接上岗。

## 它是怎么工作的

```
seeds/abstracts.jsonl            每篇种子摘要 = 一道 noul + 一道 choice
        │                        （纳入判定）    （研究设计五分类）
        ▼
build_sequence 逐 token 复刻 Rust 端布局
  [CLS] "noul question: <指令>" [SEP] [MASK] 选项0 [MASK] 选项1 [SEP] <摘要> [SEP]
        │                        ▲ 选项标记位（打分只取这些位置）
        ▼
ModernBERT-large 编码器 ← LoRA(r=16) 只挂 Wqkv/Wo/Wi（7.2M 参数）
  + 决策头全量训练（26.2M）      act_head 冻结（本任务不做动作决策）
        ▼
留出集温度重标定：按 (题型, 选项数) 桶网格搜索 NLL 最优温度
        ▼
导出 = 合并 LoRA → 原检查点布局的 model.safetensors
       + tokenizer/ + encoder/config.json + rl_agent_config.json（新温度表）
```

关键不变量：**导出的键布局与官方检查点逐字节兼容**
（`encoder.*` / `head.layers.*` / `scorer.{0,1,3}` / `type_emb.weight` / `act_head.{0,2}`），
所以 Rust 端零改动直接加载。

## 跑起来的最小命令

```bash
# ① 准备种子（一行 PICO 头 + 每篇一行 JSON：title/abstract/include/design/borderline）
$EDITOR distill/seeds/abstracts.jsonl

# ② 先看原版零样本什么水平（CPU 也能跑）
python distill/train_laya.py --eval-only

# ③ 训练（GPU；显存占用约 4GB，可与 2.7GB 的常驻租户共享 8GB 卡）
python distill/train_laya.py --out distill/out/laya-screening

# ④ Rust 端上线（CPU 即可，和 GPU 上的学生模型互不抢资源）
cargo run --release --bin laya_server -- --port 31181 --model-dir distill/out/laya-screening

# ⑤ A/B 对比 + 端到端演示（初筛 → 网关 → 学生抽取）
python distill/demo_screening.py
```

## 实测数字（RTX 3050 8GB，40 篇 T2DM 运动 RCT 种子摘要）

| 指标 | 原版零样本 | 微调 + 标定后 |
| --- | --- | --- |
| 纳入判定 | 67.5% | **97.5%** |
| 边界案例（12 篇） | 50.0% | **100%** |
| 研究设计分类 | 85.0% | **100%** |
| 校准误差 ECE（留出集） | 0.126 | **0.117** |

训练 27 分钟（14 epoch × 114 s）；筛选吞吐 CPU 端约 2.1 s/篇（两道题）。

> **诚实声明**：40 篇中 34 篇参与训练（in-sample 能力验证）；6 篇留出集
> 趋势一致（边界 67%→100%、设计 83%→100%）。要发论文请自行扩大留出集。

## 踩过的坑（都在脚本里修好了）

1. **transformers 4.48 的 ModernBERT 默认 `reference_compile=True`**：
   attention/MLP 会被 torch.compile 包住，首次 inductor 编译在 nohup
   管道里能卡 15 分钟以上。脚本里 `cfg.reference_compile = False` + 强制 sdpa。
2. **小显存共享卡**：LoRA + `gradient_checkpointing(use_reentrant=False)`
   + `enable_input_require_grads()` + `expandable_segments:True`，
   峰值从 OOM 压到 4 GB 出头。
3. **noul 答案是概率不是布尔**：`/decide` 返回的 `noul` 字段是
   P(陈述成立)，判定要 `> 0.5`，别直接 `bool()`（演示脚本第一版就踩了）。

## 常见疑问

**Q: 为什么训练要逐 token 复刻 Rust 的序列布局？**
推理时序列是 Rust 端 `build_sequence` 拼的；训练如果用另一套拼法，
[MASK] 标记位和截断策略对不上，等于让模型考一套没复习过的卷子。

**Q: 40 篇够吗？**
对"学会一个明确写出来的 PICO 标准"够了（LoRA 参数才 7.2M）。
换课题时先跑 `--eval-only` 看零样本基线，再决定要不要补种子。

**Q: act_head 为什么冻结？**
那是 RL 动作头（决定"再想想还是直接答"）。筛选任务只要答案分布，
训练它只会让 40 篇小数据过拟合得更快。
