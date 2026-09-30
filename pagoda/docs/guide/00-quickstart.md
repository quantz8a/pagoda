# 00 · 一键上手（5 分钟跑通 pagoda）

## 一句话总结

不需要懂 Rust、不需要懂 LLM：克隆仓库，跑一个脚本，看到结果。

## 前提

只需要一个东西：Rust 工具链（[rustup.rs](https://rustup.rs) 一键安装）。
**不需要联网下载任何依赖**——pagoda 是零第三方依赖的。

## Windows（PowerShell）

```powershell
cd pagoda
powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1
```

这个脚本会自动做四件事：

```
[1/4] cargo build --offline        ← 编译（零依赖，离线可编译）
[2/4] cargo test --offline         ← 跑 72 项测试，全绿说明环境没问题
[3/4] sample --repeat 3            ← 同一个 prompt 跑 3 遍，看前缀缓存省算力
[4/4] program                      ← 跑一个 gen/select/fork 的 DSL 示例
```

## Linux / macOS（bash）

```bash
cd pagoda
bash scripts/quickstart.sh
```

内容完全一样。

## 接下来：agent 集群演示（卖点功能）

```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo-agent-cluster.ps1   # Windows
bash scripts/demo-agent-cluster.sh                                        # Linux/macOS
```

你会看到：

1. 一段"共享系统 prompt"被创建为 **checkpoint**（钉进 KV 池，见第 5 篇）。
2. 3 个模拟 agent 各自分叉提问——每个分支的 `prefix_hit` 都是 100% 命中树干。
3. 最后 `/stats` 打出账本：`compute_saved_tokens`（省下的算力）和
   `prefill_skip_ratio`（跳过的 prefill 比例）。agent 越多，省得越多。

## 手动探索

```powershell
# 启动 HTTP 服务
cargo run --offline --bin pagoda -- serve --port 8080

# 另一个终端里：
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/generate -Method Post `
  -ContentType "application/json" `
  -Body '{"text":"the quick brown fox","sampling_params":{"max_tokens":20}}'
Invoke-RestMethod http://127.0.0.1:8080/stats
```

## 常见疑问

**Q：`cargo build --offline` 报错？**
先确认在 `pagoda/` 目录里（有 Cargo.toml 的那层）。离线构建不下载任何东西，
报错最常见的原因是 Rust 版本太旧——`rustup update` 一下。

**Q：测试有几项失败？**
当前全部应为绿。如果红了，把输出发给维护者——72 项测试就是 pagoda 的行为规格。

**Q：脚本里的输出是乱码/看不懂的英文碎片？**
正常！默认模型是"玩具 n-gram 模型"，只负责把链路跑通，不负责聪明。
看 `prefix_hit` 和 `compute_saved` 这些数字就行——它们才是演示的主角。