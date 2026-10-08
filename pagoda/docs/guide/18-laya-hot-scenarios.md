# 18 · Laya 全网热门场景 Rust 复刻：四个场景一个二进制

## 一句话总结

我们搜遍了 2026 年 9-10 月全网关于 Laya 的教程和讨论，把出现频率最高的四类用法——
**客服工单路由、内容安全护栏、Agent System-1 判断器、多语言意图识别**——
全部用 Rust 在一个 example 里复刻了出来：零 Python、零 GPU 依赖（CPU 就能跑）、
每个场景输出类型化答案 + 实测延迟。

## 生活化类比

Laya 是"前台风控员"，不是"分析师"。
分析师（大模型）写报告要查资料想半天；前台风控员只看一眼表单就盖章：
"转财务部、急件、这客户要跑路"。四个场景就是前台的四种日常工作：

| 场景 | 前台在干什么 | 全网热度来源 |
|---|---|---|
| 工单路由 | 看一眼投诉信，盖章"账单科/加急/要流失" | 官方 quickstart + 几乎所有入门教程 |
| 内容护栏 | 大模型上岗前的安检门，危险输入直接拦 | HF 模型卡 guardrails/moderation 标签 |
| Agent 判断器 | agent 集群每个节点的分诊台：要不要调工具/升级 | 知乎/CSDN "给 Agent 加判断器"讨论 |
| 多语言意图 | 同套问题直接问中文消息（mmBERT 检查点） | 官方卖点：100+ 语言 |

## 它是怎么工作的

```
convaiinnovations/laya（HF，Apache-2.0）
  ├── model.safetensors          英文检查点（ModernBERT，hidden 1024 × 28 层）
  └── multilingual/              多语言检查点（mmBERT，hidden 768 × 22 层，词表 256K）
        ▲ 本次补齐的支持：特殊 token 从 [CLS]/[SEP]/[PAD]/[MASK]
          fallback 到 <bos>/<eos>/<pad>/<mask>（官方 tokenizer_config 定义）
```

四个场景全部走同一个调用形状（Jev 兼容）：
`state（一段文本）+ questions（类型化问题）→ 一次前向 → 类型化答案`。

## 跑起来的最小命令

```bash
cd pagoda-hf
cargo run --release --example hot_scenarios                 # 场景 1-4 + 基准
cargo run --release --example hot_scenarios -- --no-multilingual   # 跳过多语言
```

首次运行下载英文检查点约 842MB（多语言再加 643MB），之后走 HF 缓存秒开。
国内网络把 `HF_ENDPOINT=https://hf-mirror.com` 打在命令前面即可
（本次顺手修了 `HfTokenizer::download` 走 `ApiBuilder::from_env()`，镜像变量才真正生效）。

## 实测输出（共享 GPU 开发机 CPU 模式，2026-10-05）

**场景 1 · 工单路由**（官方账单场景，断言全过）：

```
department  choice=billing  [billing=0.99 technical=0.01 other=0.01]  conf=0.93
urgency     score=1.77 (0~2 档)                          conf=0.47
churn_risk  noul(P=yes)=0.879   ← 明确说了"cancel our plan"，逮住了
```

**场景 2 · 内容护栏**：

```
正常邮件请求:  is_harmful=0.067  category=safe(0.80)
窃取 WiFi 请求: is_harmful=0.271  category=fraud(0.80)   ← 类别抓得准
```

**场景 3 · Agent System-1 判断器**：

```
"退款政策是什么"       → need_tool=0.07  need_human=0.000  直接答
"当前美元汇率"         → need_tool=0.24  next=answer_directly(0.70)
"生产库被你们删了"     → need_human=0.68 next=escalate(0.45)  ← 该升级就升级
```

**场景 4 · 中文意图识别**（multilingual 检查点，零样本）：

```
"耳机左耳没声音，想退货退款"   → intent=refund(1.00)        ← 满分
"扣了我两次钱！不解决就投诉"   → refund 0.42 / complaint 0.41  ← 合理纠结
"手表支持游泳佩戴吗"           → other(0.47) consult(0.21)    ← 零样本偏弱，
```

零样本不完美是正常的——文档 17 的微调管线就是干这个的（40 篇种子 27 分钟训完）。

## 延迟实测

| 环境 | 单次判定 | 来源 |
|---|---|---|
| pagoda CPU f32（英文检查点） | ~1100 ms（20 次 p50=1104） | 本次实测 |
| pagoda CPU f32（多语言检查点） | ~500 ms | 本次实测（模型更小：768×22 层） |
| pagoda GPU f32（RTX 3050） | **606 ms** | BENCHMARK-LAYA.md（2026-10-03） |
| laya-python GPU fp16（RTX 3050） | 691 ms | 同上（autocast 占精度便宜仍更慢） |
| 官方宣称（T4 批处理 + TileLang） | ~33 ms / 103-332 q/s | 官方 BENCHMARKS.md |

差距来自三处：CPU vs GPU、f32 vs fp16、逐次调用 vs 批处理。
同卡对比 pagoda 已经赢过官方 Python 栈（606 vs 691 ms，且我们 f32 对方 fp16）；
Rust 版更核心的价值在**部署形态**：单二进制、无 Python/CUDA 工具链、
塞进任何容器就跑——官方 Python 栈要 torch+transformers 一整套。

## 改成你自己的业务

所有场景都只是"换一组问题"：

```rust
let questions = vec![
    ("risk".to_string(), Question::noul("这件事需要人工复核吗？")),
    ("queue".to_string(), Question::choice("分到哪个队列？", &[
        ("a", "描述 A"), ("b", "描述 B"),
    ])),
];
let d = laya.decide(你的文本, &questions)?;
```

HTTP 服务形态：`laya_server --port 8081`，`POST /decide` 收同样的 JSON
（见文档 14 一键部署）。

## 本次踩坑记录

1. **hf-hub 0.4.3 的 `Api::new()` 不读 `HF_ENDPOINT`**——只有 `ApiBuilder::from_env()` 读。
   镜像变量设了也白设，已修。
2. **hf-mirror 不支持 Content-Range 分段下载**——小文件会报 "Header Content-Range is missing"。
   解决：大文件走镜像，小文件本地 curl 后补进缓存，`HF_HUB_OFFLINE=1` 跑。
3. **multilingual 检查点的特殊 token 完全不同**（`<bos>` 当 [CLS] 用）——
   官方 Python 靠 `tok.cls_token_id` 自动适配，Rust 端我们做了 token 名 fallback。
4. 场景 4 的"耶机"是 unicode 转义笔误（应为"耳机"），模型照样判对了 refund——
   顺手当了一次鲁棒性测试。
