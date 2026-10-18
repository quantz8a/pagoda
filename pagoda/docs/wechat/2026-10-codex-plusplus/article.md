---
title: "为什么用 pagoda"
author: quantz8a
digest: "同一台 M3 Ultra、同一份 Qwen3.8。单流 23.5 对 23.3。四个已知 token 50.9。八个 Agent 前向少 98%。"
---

![题图](images/fig0-hero.png)

同一台机器，同一份权重。

![单流热解码](images/fig1-decode.png)

已知的四个 token，一次算完。

![四个已知 token](images/fig2-verify.png)

长系统提示不用各算各的。

![八个 Agent](images/fig3-share.png)

Codex++ 接到本机。

![接到 Codex++](images/fig4-wire.png)

```bash
PAGODA_DEVICE=metal ./target/release/qwen3_server --port 8000
```

数字和接法写在 GitHub README。
