# rs-mm

零第三方依赖的多模态图像预处理 crate。算法血缘：SGLang
[`rust/sglang-mm`](https://github.com/sgl-project/sglang/tree/main/rust/sglang-mm)
（Apache-2.0），见仓库根目录 `NOTICE`。

## 五分钟

```bash
cd rs-mm
cargo test --offline
```

## 结构

| 路径 | 说明 |
| --- | --- |
| `src/pipeline.rs` | `MmFamilyProcessor`、`Geometry::Grid([t,h,w])`、`TokenLayout` |
| `src/common/resize.rs` | PIL / ATen uint8 bicubic，与上游 bit-exact |
| `src/common/token_layout.rs` | placeholder 展开 + offset |
| `src/model/qwen2_vl.rs` | Qwen2-VL / 2.5-VL / 3-VL 静图路径 |

新增叶子：实现 `MmFamilyProcessor`，放进 `src/model/`，并在 `src/model/mod.rs` 注册。
