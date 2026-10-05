# 实验方案：真实金标准验证（Protocol）

**文档编号** PAGODA-VAL-001 · **版本** v1.0 · **日期** 2026-10-04
**状态** 已按本方案执行完毕，结果见同目录《测试报告》

---

## 1. 背景与动机

pagoda 的文献筛选流水线（Laya 初筛 → 网关路由 → 学生模型抽取）此前仅在 40 篇
**程序生成的合成摘要**上评测（纳入判定准确率 97.5%）。合成测试集由生成数据的同一
分布采样而来，存在"自己考自己"的循环论证风险。本实验用**真实 PubMed 文献**和
**已发表系统综述的人工标注结果**作为外部金标准，检验流水线在真实分布上的可用性。

## 2. 研究问题

> 在真实检索产出上，Laya（基座零样本 / 合成数据微调版）做摘要级纳入排除判定，
> 能否达到系统综述初筛的可用标准？

**可用标准**（依据 TAR/CLEF 领域惯例）：初筛阶段漏掉一篇合格文献的代价远高于
多看一篇垃圾，因此**主要终点为灵敏度，目标 ≥ 0.95**；特异度决定省多少人力，
为次要终点。

## 3. 金标准

### 3.1 来源综述

Umpierre D, et al. *Physical Activity Advice Only or Structured Exercise
Training and Association With HbA1c Levels in Type 2 Diabetes: A Systematic
Review and Meta-analysis.* **JAMA. 2011;305(17):1790-1799.** PMID 21540423。

选择理由：其 PICO（2 型糖尿病成人 · 结构化运动 vs 对照 · HbA1c 结局 · 仅 RCT）
与 pagoda 演示流水线的纳入语句**逐条对应**；纳入 47 项 RCT，均为人工双人筛选
并经全文确认，是可获得的最强摘要级金标准之一。

### 3.2 金标准构造（三步，全部程序化可复查）

1. **参考文献清单**：经 OpenAlex API 取该综述的 76 条参考文献，解析出 67 个 PMID；
2. **检索池交集**：仅保留同时出现在 §4 重建检索池中的 38 篇（排除综述、方法学
   等背景引用——它们不会被该检索式命中）；
3. **出版类型过滤**：经 PubMed PublicationType 保留标注为
   Randomized/Controlled Clinical Trial 者，剔除 4 篇 Meta 分析/综述，
   得 **34 篇金标准阳性**。

## 4. 检索池重建

按该综述 Methods 的检索概念复建 PubMed 检索式（时间窗截至其检索月份）：

```
("diabetes mellitus, type 2"[MeSH Terms])
AND ("exercise"[MeSH Terms] OR "exercise therapy"[MeSH Terms]
     OR "resistance training"[MeSH Terms] OR exercise[Title/Abstract]
     OR aerobic[Title/Abstract] OR resistance[Title/Abstract]
     OR "physical activity"[Title/Abstract] OR training[Title/Abstract])
AND ("glycated hemoglobin"[MeSH Terms] OR hba1c[Title/Abstract]
     OR "glycated hemoglobin"[Title/Abstract] OR glycosylated[Title/Abstract]
     OR "hemoglobin a1c"[Title/Abstract])
AND ("1900/01/01"[Date - Publication] : "2011/03/31"[Date - Publication])
```

命中 **1975 篇**（与原综述同一数量级）。

## 5. 评估集

| 组成 | 数量 | 说明 |
|---|---|---|
| 阳性 | 34 | §3 金标准 |
| 阴性 | 250 | 检索池内随机抽样（随机种子 42），即"被检索命中但未最终纳入"，与 CLEF TAR 的摘要级阴性定义一致 |
| 剔除 | 5 | PubMed 无摘要（无法评估） |
| **合计** | **279** | 34 阳 + 245 阴 |

## 6. 被测系统与输入

| 配置 | 值 |
|---|---|
| 基座 | laya_server（Rust）:31180，Laya 原始权重，零样本 |
| 微调 | laya_server（Rust）:31181，LoRA r=16 + 决策头全量，40 篇合成种子训练 27 min，留出集温度重标定 |
| 任务 1 | noul 纳入判定（输出 P(include) 与置信度），判定阈值默认 0.5 |
| 任务 2 | choice 研究设计分类（rct/cohort/review/invitro/other） |
| 输入 | 标题 + 摘要原文，两模型逐字节一致 |

## 7. 指标与分析计划

- **主要终点**：灵敏度（金标准阳性被判定纳入的比例），目标 ≥ 0.95，Wilson 95% CI
- **次要终点**：特异度、PPV、准确率（均报 Wilson 95% CI）
- **阈值分析**：阈值 t ∈ [0.02, 0.98] 扫描灵敏度/特异度；ROC 与 AUC；
  达到灵敏度 ≥ 0.95 的最高特异度工作点
- **描述性分析**：分数分布（按金标准分组）、混淆矩阵、假阴性案例文本分析
- **判定规则**：任何模型若在默认阈值下主要终点未达标，即判"不可直接部署"，
  并转入根因分析

## 8. 可复现性

| 项 | 位置 |
|---|---|
| 引用解析 | /tmp/fetch_refs.py → /tmp/umpierre_refs.json |
| 检索池 | /tmp/build_pool.py → /tmp/umpierre_pool.json |
| 主验证 | /tmp/validate_real.py → /tmp/umpierre_validation.json |
| 图表 | /tmp/val_figures.py → /tmp/valfig/v1-v4 |
| 执行环境 | 共享 GPU 开发机（i5-12500 / RTX 3050 8GB），2026-10-04 |

外部依赖：NCBI E-utilities（公开）、OpenAlex API（公开）。所有数据为公开文献元数据。
