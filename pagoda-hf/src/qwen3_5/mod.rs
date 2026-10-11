// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Qwen3.5 / Qwen3.8 hybrid text backbone for pagoda-hf.
//!
//! Local `mlx-community/Qwen3.8-27B-*` checkpoints advertise
//! `architectures: ["Qwen3_5ForConditionalGeneration"]` with a hybrid stack of
//! Gated DeltaNet (`linear_attention`) and full attention layers. Candle's
//! zoo only covers classic Qwen3 (dense attention), so this module ports the
//! transformers `qwen3_5` text path into Candle.
//!
//! Milestone status:
//! * ✅ config parse (nested `text_config`)
//! * ✅ recurrent gated-delta kernel (decode / short prefill) — fused S passes
//! * ✅ causal conv1d + GDN layer (CPU f32 reference forward; in-place conv ring)
//! * ⏳ full-attention block, hybrid decoder stack, MLX-8bit dequant load
//! * ⏳ chunked FLA prefill; Metal / Candle GPU kernels

pub mod causal_conv;
pub mod config;
pub mod gated_delta;
pub mod gdn;

pub use config::{LayerType, Qwen3_5Config, Qwen3_5TextConfig};
pub use gated_delta::{recurrent_gated_delta_rule, GatedDeltaState};
pub use gdn::{gdn_forward, GdnCache, GdnWeights};
