# Contributing to Pagoda

感谢你来帮忙。Pagoda 是一个**零第三方依赖、可离线构建**的 Rust LLM 推理运行时：
目标是 `cargo build --offline` 能编、`cargo test --offline` 全绿、每个能力都有
`/stats` 信号或一个测试把它锁死。你不需要 GPU、不需要联网，就能跑起来和改起来。

## 十分钟跑通（先验证环境）

```powershell
cd pagoda
# 一键：build + 测试 + demo
powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1   # bash: scripts/quickstart.sh

# 全量测试（离线）
cargo test --offline

# 跑一个能看见「前缀缓存省算力」的 demo
cargo run --offline --bin pagoda -- sample -p "SGLang is a serving framework" --max-tokens 40 --repeat 3
```

测试全绿=环境 OK，之后所有改动都以它为准。


## 认领任务

看到 `good first issue`，不需要等批准，直接在该 issue 下评论：

- `/claim`：机器人把你的名字标成 `claimed:@你`，并发一条开工清单。
- `/unclaim`：取消认领，把机会让给别人。

一个任务同一时间只给一个人；被认领后入口会显示绿色标签，其他人去 [已开放任务](https://github.com/quantz8a/pagoda/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) 找下一条。

## 找活干：三条路径

1. **修 Bug / 补测试**：看 `GOOD_FIRST_ISSUES.md`，或 GitHub 上 `good-first-issue`
   / `help-wanted` 标签。这是第一步上手最好的入口。
2. **做一个机制**：每个机制都对应一条收入轴（TTFT / Token-Watt / MTBI / Useful
   Life），做完要有 `/stats` 信号 + 测试。见 `docs/DESIGN.md` 和
   `docs/REQUIREMENTS.md`。
3. **文档 / 案例**：`docs/guide/` 里 00–18 篇，补缺、纠错、加 example 都欢迎。

## 贡献梯度（从易到难，顺着爬）

| 梯度 | 做什么 | 例子 |
| --- | --- | --- |
| L0 | 文档 / 示例 / 错误信息 | 补 guide、给 endpoint 加 example |
| L1 | 单测 / 属性测试 / 端点契约测试 | prop-test 引用计数不变量 |
| L2 | 移植一个机制 / 一个模型叶子 | MM 处理器、调度策略、真实后端 |
| L3 | 性能 / 真实负载验证 | benchmark、fault isolation、SLO |

新手从 L0/L1 进，L2 是主力贡献面，L3 需要先跑通 `pagoda-hf` 的真实权重链路。

## PR 流程

1. Fork + 开分支（命名 `feat/`、`fix/`、`docs/`）。
2. 改代码，保持主 crate 零第三方依赖、无 `unsafe`。
3. `cargo fmt`、`cargo test --offline` 全绿。
4. PR 描述写四件事：**做了什么 · 为什么 · 驱动哪条轴 · 怎么自测**。
5. 一条 PR 只做一件事，标题用祈使句（`Add qwen2-vl image processor`）。

## 代码约定

- 主 crate（`pagoda`）零第三方依赖；`pagoda-hf` 才允许拉 candle / tokenizer 等联网依赖。
- 无 `unsafe`（`src/` 全量保持）；确定性、可复现是硬约束。
- 新能力必须带一个「可验证信号」：要么进 `/stats`，要么一个测试锁死，不能只留演示。
- Rust edition 2021，风格跟着现有代码走，不加版权头。

## 行为准则

就三条：**对事不对人、新手犯傻耐心带、借了谁的思路就署名**（SGLang 的血缘写进
`NOTICE`）。这是 Apache-2.0，干净、可商用。

更多「认领后从哪里下手」的任务清单见 `GOOD_FIRST_ISSUES.md`。