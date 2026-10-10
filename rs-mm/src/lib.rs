//! Zero-dependency multimodal preprocessing for Pagoda.
//!
//! The image-processor math is a direct port of
//! `sgl-project/sglang` `rust/sglang-mm` (Qwen2-VL / 2.5-VL / 3-VL stills),
//! with the PyO3, serde, and thread-pool pieces left behind so
//! `cargo test --offline` needs no network and no third-party crates.

pub mod common;
pub mod model;
pub mod pipeline;
