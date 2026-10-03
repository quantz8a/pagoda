# 16 · 分诊网关：System 1 给 System 2 当门卫

## 一句话总结

`pagoda serve --upstream ... --laya-url ...` 让每个请求在见到大模型之前，
先被 Laya 这个"不生成文本的决策模型"看一眼：**安全的放行**给后面的
SGLang 学生模型正常生成，**危险的就地拦截**直接转人工——危险流量
连 GPU 都不用碰，响应里还附带完整的判定依据。

## 生活化类比

医院急诊的分诊台。病人进门先找护士台（Laya），30 秒量血压、问症状：

- 普通感冒 → 去门诊排队（学生模型正常处理，烧 GPU 值得）；
- 胸痛疑似心梗 → 直接推抢救室（转人工，一秒都不耽误）。

分诊台本身不治病，但它站在门口，决定了**谁配得上最贵的资源**。
护士请假了怎么办？急诊不能关门——fail-open：病人照常进门诊，
登记本上记一笔"今日无分诊"（`triage_unavailable` 计数）。

## 它是怎么工作的

```
POST /v1/chat/completions
        │
        ▼
┌──────────────── pagoda serve（网关）────────────────┐
│ ① 提取文本：/generate 取 text；chat 取最后一条 user 消息 │
│ ② 调 Laya 三问（一次 HTTP）：                          │
│    · 部门 choice      → billing / technical / ...      │
│    · 流失风险 noul    → 0.94                           │
│    · 要转人工吗 noul  → 0.62                           │
│ ③ 判定升级条件（任一命中即转人工）：                     │
│    needs_human > 0.5 ｜ churn_risk > 阈值 ｜ 置信度过低   │
└──────┬──────────────────────────┬─────────────────────┘
    安全 │                          │ 危险
         ▼                          ▼
   逐字节转发给 upstream        网关直接回复
   （SGLang 上的学生模型）       "已为您优先转接人工客服"
   正常生成，原样返回            + pagoda_triage 扩展字段
                                 全程 0 GPU、0 生成 token
```

五个开关（都有默认值，不传也能跑）：

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| `--laya-url` | 无 | 接上 Laya 服务；不传就是纯代理 |
| `--churn-threshold` | 0.5 | 流失风险超过它 → 转人工 |
| `--min-confidence` | 0.0 | Laya 不自信 → 转人工（不自信也是一种危险） |
| `--laya-shadow` | 关 | 影子模式：判定照做、日志照记、**永不拦截**（灰度上线用） |
| `--laya-required` | 关 | 打开后 Laya 挂了返回 502（fail-closed），默认 fail-open |

## 在 pagoda 里动手试试

下面是 2026-10-03 在真实机器上三进程全链路实跑的记录
（Laya + Qwen2.5-0.5B 蒸馏学生 + pagoda 网关）：

```bash
# ① 起 Laya（17.9MB 单二进制，见第 14 篇）
pagoda-hf/target/release/laya_server --port 31180

# ② 起学生模型（蒸馏产物，见第 15 篇）
python -m sglang.launch_server --model-path out/student-merged \
  --port 31102 --served-model-name student \
  --attention-backend torch_native --sampling-backend pytorch \
  --mem-fraction-static 0.55

# ③ 起网关：前面是分诊台，后面是学生
pagoda serve --port 31100 \
  --upstream http://127.0.0.1:31102 \
  --laya-url http://127.0.0.1:31180
```

**安全工单**（查物流）→ 转发给学生，正常生成：

```json
{"model": "student", "choices": [{"message": {"content":
  "您好，感谢您的反馈。我已记录下您的订单信息，并将尽快为您查询物流情况……"}}]}
```

**威胁工单**（"不退款就投诉拒付"）→ 2.9 秒内转人工，0 GPU：

```json
{"model": "pagoda-triage", "choices": [{"message": {"content":
  "您的诉求已收到。为保障您的权益，已为您优先转接人工客服专员处理，请稍候。
   （pagoda 分诊：needs_human=0.62, churn_risk=0.94 > 0.5）"}}],
 "pagoda_triage": {"escalated": true, "department": "billing",
                   "churn_risk": 0.9425, "needs_human": 0.6177,
                   "reason": "needs_human=0.62, churn_risk=0.94 > 0.5"}}
```

**分诊台宕机演练**（kill 掉 Laya 再发请求）→ 请求照常转发，
`GET /stats` 里 `triage_unavailable` 计数 +1——这就是 fail-open。

```json
{"proxied_requests": 2, "triaged_requests": 3,
 "escalated_requests": 1, "triage_unavailable": 1}
```

## 和 SGLang / vLLM 的对照

| | 纯 SGLang / vLLM | pagoda 分诊网关 |
| --- | --- | --- |
| 威胁工单 | 照样进 GPU 队列烧 token | 门口拦截，0 GPU，2.9s 转人工 |
| 判定依据 | 无（或在 prompt 里求模型自省） | 校准概率，结构化字段，可审计 |
| 判定服务挂了 | —（没有这个概念） | fail-open 照常服务，计数可查 |
| 上线新分诊策略 | 改代码 | `--laya-shadow` 影子模式先看一周 |

## 常见疑问

**Q：为什么不用关键词过滤"投诉/退款"？**
关键词不懂"我要给五星好评——才怪"。Laya 是校准过的概率模型，
给出的是 0.9425 这样的数字，阈值可随业务调；关键词只有 0 和 1。

**Q：Laya 误判了怎么办？**
先 `--laya-shadow` 跑一阵：判定全记日志但永不拦截，人工抽查准确率
满意了再关影子模式。误判代价也可控——误升级只是人工客服多接一单，
误放行才有业务风险，所以默认阈值宁可偏严。

**Q：这和第 15 篇的蒸馏是什么关系？**
合起来是完整的业务闭环：**蒸馏造学生**（便宜的 System 2），
**分诊台当门卫**（Laya，System 1），**转人工兜底**（真人）。
学生答不了、答不好、不该答的，分诊台在门口就分流掉了。

**Q：性能开销多大？**
安全请求只多一次 Laya 前向：GPU 实测 606ms、CPU 约 1.0s（数字见
`docs/BENCHMARK-LAYA.md`）。危险请求反而省下了整个生成过程。
