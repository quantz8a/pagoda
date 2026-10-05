# 真实金标准验证 · 复现运行手册

**前置**：一台能访问 PubMed 的机器 + 跑起来的 laya_server（基座/微调各一实例）。
本文命令以一台共享 GPU 开发机为例（Linux）；换机器时把 `PAGODA_HOME` 指到你的 pagoda 部署目录即可（下文默认 `export PAGODA_HOME=~/pagoda`）。

## 0. 服务就绪检查

```bash
curl -s http://127.0.0.1:31180/health   # 基座 Laya
curl -s http://127.0.0.1:31181/health   # 微调 Laya（当前 v1；v2 起 31182）
```

## 1. 构建金标准（约 1 分钟）

```bash
python3 fetch_refs.py     # OpenAlex 取 Umpierre 2011 参考文献 → /tmp/umpierre_refs.json
python3 build_pool.py     # 重建 PubMed 检索池(1975篇) + 阳性交集 → /tmp/umpierre_pool.json
```

## 2. 跑主验证（约 60 分钟，279 篇 × 2 模型）

```bash
nohup python3 validate_real.py > /tmp/validate_real.log 2>&1 &
# 每 20 篇打一次进度；完成后输出 base/tuned 的 TP/FN/TN/FP 与灵敏度/特异度
```

产出：/tmp/umpierre_validation.json（含每篇的模型概率，供阈值扫描）

## 3. 出图（约 10 秒）

```bash
python3 val_figures.py    # → /tmp/valfig/v1-v4.png（阈值扫描/ROC/分布/混淆矩阵）
```

## 4. 训练 v2（真实分布 + FN 加权，约 40 分钟）

```bash
python3 build_seeds_v2.py   # 从检索池取银标数据（自动排除 279 篇评估集，防泄漏）
PY=$PAGODA_HOME/bench-ref/venv/bin/python   # 含 torch+peft 的环境
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup $PY train_laya.py --seeds seeds/abstracts-real-v2.jsonl \
  --out out/laya-screening-v2 --fn-weight 8 --epochs 6 > train-v2.log 2>&1 &
```

关键开关：
- --fn-weight 8：漏判一篇真实试验的代价设为误判的 8 倍（A3，灵敏度优先）
- 验证集由脚本按 (design, include) 分层自动切 1/5 —— 真实分布，温度标定在其上做（A4）

## 5. v2 复测（约 35 分钟，只测一个模型）

```bash
# v2 检查点起在 31182，复用同一评估集 /tmp/umpierre_validation.json 的摘要
./target/release/laya_server --port 31182 --model-dir out/laya-screening-v2 &
python3 revalidate_v2.py   # 对照 v1 结果输出 A/B/C 三方对比
```

## 常见问题

| 症状 | 处置 |
|---|---|
| CUDA OOM | 3050 只有 8GB；先停 student（kill 掉 sglang.launch_server 进程），训完再启 |
| torch 找不到 | 别用系统 python3，用 bench-ref/venv 里的 |
| NCBI 限流 | 脚本已带 0.4s 间隔 + 重试；频繁 429 就睡一觉再跑 |
| 中文图变方块 | 图脚本已注册 /usr/share/fonts/opentype/noto/NotoSansCJK-*.ttc |
| sglang 学生服务 JIT 编译失败（nvcc Unknown option） | 系统 nvcc 是 10.1；必须 export CUDA_HOME=$HOME/cuda12/root/usr/local/cuda-12.8 并前置其 bin 到 PATH（10/03 首次构建用的就是它） |
| ssh 会话 HOME 被本地 Windows 值污染 | 脚本里显式 export HOME=<真实 home 目录>，否则 tvm-ffi 找不到 JIT 缓存 |
| rl_agent_config.json 解析失败 | v2 之前 train_laya.py 会写出 NaN（非法 JSON）；已修：导出时 NaN→null |

全部脚本在仓库 docs/validation/ 下（本手册同目录）。
