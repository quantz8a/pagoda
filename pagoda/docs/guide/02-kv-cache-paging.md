# 02 · KV Cache 与分页内存

## 一句话总结

KV cache 是模型"记住前文"的中间结果；把它切成固定大小的页（block）来管理，
就像操作系统用分页管理内存一样——这就是 PagedAttention 的思想。

## 生活化类比

把 KV cache 想象成**停车场**：

- 每个 token 的中间结果 = 一辆车。
- 不切页的做法 = 每来一个车队（请求），就要找一块**恰好连续**的空地停整个车队。
  车队有大有小，空地很快被切成碎块，明明总车位够却停不下（内存碎片）。
- 切页的做法 = 停车场划成统一大小的车位区（block）。车队来了就分区停，
  车队的"停车单"（页表）记录每区停在哪。碎片问题消失。

## 它是怎么工作的

```
逻辑上：一个请求的 token 序列   [t0 t1 t2 t3 t4 t5 t6 t7 t8 t9]
                                  │        │        │
物理上：固定大小的块（每块 4 个槽位）▼        ▼        ▼
                          block 7: [t0 t1 t2 t3]
                          block 2: [t4 t5 t6 t7]
                          block 9: [t8 t9 __ __]   ← 尾部没停满
```

pagoda 的 `PagedKvCache`（`src/kv_cache.rs`）在此基础上加了两个关键机制：

### 引用计数（refcount）

每个块记着"现在有几个人在用我"。

- 请求结束时 `dec_ref`：用完一个块就还掉。
- 计数归零：块回到空闲池，可以被下一个人用。

### 写时复制（COW, copy-on-write）

两个请求共享同一段前缀时，它们**引用同一个物理块**，谁也不复制（零拷贝复用）。
直到其中一个请求要往共享的尾部块里**写新 token**——这时才把尾块复制一份私有的，
各写各的，互不干扰。

```
共享前缀块 [t0..t3] ←── 请求 A ─┐
                               ├─ 引用计数 = 2
              [t0..t3] ←── 请求 B ─┘

请求 B 要追加 t4：
  1. 复制一块私有的 [t0..t3]
  2. B 的停车单改指向私有块
  3. B 往私有块写 t4；A 的块原封不动
```

pagoda 有两条单测直接锁死这个行为：`kv_cache::tests::copy_on_write_isolates_writer`
（原语级）和 `engine::tests::branch_appends_never_touch_checkpoint_blocks`（引擎级）。

## 在 pagoda 里动手试试

```powershell
cargo run --offline --bin pagoda -- sample -p "同一段开头" --repeat 2
```

看输出里的 `kv_util`（KV 池利用率）和 `compute_saved`：第二次请求复用了第一次的块，
没有重新分配整段前缀。

## 和 SGLang / vLLM 的对照

| 机制 | vLLM / SGLang | pagoda |
| --- | --- | --- |
| 分页 KV 池 | `memory_pool`，GPU 显存 | `PagedKvCache`，内存块（存 token id 代替张量） |
| 页表 | attention kernel 间接触达 | 每请求 `locs: Vec<(BlockId, offset)>` |
| 引用计数 | ✅ | ✅ `inc_ref` / `dec_ref` |
| 写时复制 | ✅ fork 序列时用 | ✅ `fork_block` + 引擎 COW 路径 |

把块里存的 `u32`（token id）换成 `[f32; head_dim]` 张量，就是真实实现的样子——
管理逻辑一模一样。

## 常见疑问

**Q：块大小怎么选？**
真实系统一般 16 token/块。块越大，管理开销越小但尾部浪费越多；块越小越精细但元数据越多。
pagoda 里用 `EngineConfig.block_size` 调，测试里故意用 4 或 8 来放大各种边界情况。

**Q：池子满了怎么办？**
两个选择：拒绝新请求，或者**淘汰**（evict）缓存里最久没用的前缀块。
pagoda 默认后者（`evict_on_pressure: true`），见第 7 篇。