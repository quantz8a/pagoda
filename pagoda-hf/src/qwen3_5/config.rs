// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Config shapes for Qwen3.5 / Qwen3.8 hybrid checkpoints.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// One decoder block's token-mixing style.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

/// Text tower (`text_config` or flat text-only configs).
#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3_5TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
    #[serde(default = "default_output_gate")]
    pub output_gate_type: String,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    /// Nested rope block (optional; used for theta / partial rotary).
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

fn default_partial_rotary() -> f64 {
    0.25
}
fn default_output_gate() -> String {
    "swish".into()
}
fn default_rope_theta() -> f64 {
    10_000_000.0
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

/// Top-level HF `config.json` for multimodal Qwen3.5/3.8 packages.
#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3_5Config {
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Nested text tower (multimodal checkpoints).
    #[serde(default)]
    pub text_config: Option<Qwen3_5TextConfig>,
}

impl Qwen3_5Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let f = File::open(path.as_ref())
            .with_context(|| format!("open {}", path.as_ref().display()))?;
        let cfg: Self = serde_json::from_reader(BufReader::new(f))
            .context("parse Qwen3.5/3.8 config.json")?;
        Ok(cfg)
    }

    /// True when this config is the hybrid GDN + full-attention family.
    pub fn is_qwen3_5_family(&self) -> bool {
        if let Some(mt) = &self.model_type {
            if mt.starts_with("qwen3_5") || mt == "qwen3.5" {
                return true;
            }
        }
        self.architectures.iter().any(|a| {
            a.contains("Qwen3_5") || a.contains("Qwen3.5") || a.contains("Qwen3_8")
        }) || self.text_config.is_some()
            && self
                .text_config
                .as_ref()
                .map(|t| !t.layer_types.is_empty())
                .unwrap_or(false)
    }

    pub fn text(&self) -> Result<&Qwen3_5TextConfig> {
        self.text_config
            .as_ref()
            .context("config.json has no text_config (need Qwen3.5/3.8 multimodal or text tower)")
    }

    pub fn validate_text(text: &Qwen3_5TextConfig) -> Result<()> {
        if text.layer_types.len() != text.num_hidden_layers {
            bail!(
                "layer_types len {} != num_hidden_layers {}",
                text.layer_types.len(),
                text.num_hidden_layers
            );
        }
        if text.linear_num_value_heads % text.linear_num_key_heads != 0 {
            bail!(
                "linear_num_value_heads ({}) must be a multiple of linear_num_key_heads ({})",
                text.linear_num_value_heads,
                text.linear_num_key_heads
            );
        }
        Ok(())
    }

    pub fn rope_theta(&self) -> Result<f64> {
        let text = self.text()?;
        if let Some(rp) = &text.rope_parameters {
            if let Some(t) = rp.rope_theta {
                return Ok(t);
            }
        }
        Ok(text.rope_theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_local_qwen38_27b_if_present() {
        let p = std::path::Path::new(
            "/Users/zzz858aaa/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit/config.json",
        );
        if !p.exists() {
            return;
        }
        let cfg = Qwen3_5Config::from_file(p).unwrap();
        assert!(cfg.is_qwen3_5_family());
        let text = cfg.text().unwrap();
        Qwen3_5Config::validate_text(text).unwrap();
        assert_eq!(text.num_hidden_layers, 64);
        assert_eq!(text.linear_num_value_heads, 48);
    }

    #[test]
    fn parses_nested_qwen38_shape() {
        let raw = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "vocab_size": 248320,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "num_hidden_layers": 4,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 262144,
                "attn_output_gate": true,
                "partial_rotary_factor": 0.25,
                "output_gate_type": "swish",
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "rope_parameters": { "rope_theta": 10000000.0, "partial_rotary_factor": 0.25 }
            }
        }"#;
        let cfg: Qwen3_5Config = serde_json::from_str(raw).unwrap();
        assert!(cfg.is_qwen3_5_family());
        let text = cfg.text().unwrap();
        Qwen3_5Config::validate_text(text).unwrap();
        assert_eq!(text.layer_types[3], LayerType::FullAttention);
        assert_eq!(cfg.rope_theta().unwrap(), 10_000_000.0);
    }
}
