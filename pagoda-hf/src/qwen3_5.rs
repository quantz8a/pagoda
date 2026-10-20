// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Text-only Qwen3.5 / Qwen3.8.
//!
//! The checkpoint this targets is an MLX directory (config.json + safetensors
//! shards) quantized with affine 8-bit groups of 64: `weight` is packed
//! little-endian uint8 inside uint32, and `scales` / `biases` are BF16.
//! Dequant is `bias + scale * code`.
//!
//! 48 of the 64 blocks are gated-delta linear attention; 16 are full
//! attention with partial RoPE, Q/K RMSNorm and a sigmoid output gate.
//! The vision tower is not loaded.
//!
//! Metal decode (`seq == 1`) keeps affine weights packed and runs GEMV.
//! Linear-attention layers encode the full block (input RMSNorm →
//! projections → causal conv → Gated Delta → out → SwiGLU) in one
//! command buffer; full-attention layers still mix CPU softmax with a
//! fused residual MLP.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use memmap2::Mmap;
use serde_json::Value;

const GROUP: usize = 64;
const BITS: usize = 8;

pub struct Qwen35 {
    maps: Vec<Mmap>,
    embed: Affine,
    head: Affine,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
    eos: u32,
    hidden: usize,
    vocab: usize,
    rope_theta: f32,
    rotary_dim: usize,
    device: Device,
}

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer,
    gate: Affine,
    up: Affine,
    down: Affine,
    /// Warm decode: cloned Metal handles for a linear layer (built once).
    #[cfg(feature = "metal")]
    metal_linear: Mutex<Option<crate::qwen3_fuse::LinearLayerOwned>>,
    #[cfg(feature = "metal")]
    metal_full: Mutex<Option<crate::qwen3_fuse::FullLayerOwned>>,
}

enum Mixer {
    Linear(LinearAttn),
    Full(FullAttn),
}

struct LinearAttn {
    qkv: Affine,
    z: Affine,
    a: Affine,
    b: Affine,
    out: Affine,
    conv: Vec<f32>,
    conv_dim: usize,
    kernel: usize,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    k_heads: usize,
    v_heads: usize,
    k_dim: usize,
    v_dim: usize,
    #[cfg(feature = "metal")]
    metal_aux: Mutex<Option<crate::qwen3_fuse::LinearAttnAux>>,
}

struct FullAttn {
    q: Affine,
    k: Affine,
    v: Affine,
    o: Affine,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

struct Buf {
    shard: usize,
    off: usize,
}

struct Affine {
    packed: Buf,
    scales: Buf,
    biases: Buf,
    out: usize,
    inp: usize,
    /// Dequantized `[out, in]` F16, kept on the GPU after the first use.
    gpu: Mutex<Option<Tensor>>,
    /// Packed 8-bit matrix resident on the GPU. Decode uses this instead of `gpu`.
    #[cfg(feature = "metal")]
    quant: Mutex<Option<crate::qwen3_gemv::QuantMatrix>>,
}

struct CacheLayer {
    /// `true` for full attention, `false` for gated-delta linear attention.
    full: bool,
    /// Full-attention K/V, `[len, kv_heads, head_dim]`.
    k: Vec<f32>,
    v: Vec<f32>,
    kv_heads: usize,
    head_dim: usize,
    /// Last `kernel-1` pre-activation conv inputs, `[conv_dim, kernel-1]`.
    conv: Vec<f32>,
    /// Recurrent state `[v_heads, k_dim, v_dim]`.
    state: Vec<f32>,
    #[cfg(feature = "metal")]
    gpu: Option<crate::qwen3_fuse::GpuRecurrent>,
    #[cfg(feature = "metal")]
    gpu_kv: Option<crate::qwen3_fuse::GpuKv>,
}

pub struct Qwen35Cache {
    layers: Vec<CacheLayer>,
    len: usize,
}

impl Qwen35Cache {
    /// Independent copy of the recurrent state and full-attention KV.
    /// Linear-attention state is not a page; a branch has to clone the whole blob.
    pub fn snapshot(&self) -> Result<Self> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(CacheLayer {
                full: layer.full,
                k: layer.k.clone(),
                v: layer.v.clone(),
                kv_heads: layer.kv_heads,
                head_dim: layer.head_dim,
                conv: layer.conv.clone(),
                state: layer.state.clone(),
                #[cfg(feature = "metal")]
                gpu: match &layer.gpu {
                    Some(gpu) => Some(gpu.snapshot()?),
                    None => None,
                },
                #[cfg(feature = "metal")]
                gpu_kv: match &layer.gpu_kv {
                    Some(kv) => Some(kv.snapshot()?),
                    None => None,
                },
            });
        }
        Ok(Self {
            layers,
            len: self.len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Host bytes of the recurrent state and full-attention KV.
    /// GPU buffers are the source of truth after a Metal step.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            if layer.full {
                out.push(2);
                let (kv_len, kv_heads, head_dim, k, v) = layer.full_host();
                out.extend_from_slice(&(kv_len as u32).to_le_bytes());
                out.extend_from_slice(&(kv_heads as u32).to_le_bytes());
                out.extend_from_slice(&(head_dim as u32).to_le_bytes());
                push_f32(&mut out, &k);
                push_f32(&mut out, &v);
            } else {
                out.push(1);
                let (conv, state) = layer.linear_host();
                push_f32(&mut out, &conv);
                push_f32(&mut out, &state);
            }
        }
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut at = 0usize;
        let version = take_u32(bytes, &mut at)?;
        anyhow::ensure!(version == 1, "kv version {version}");
        let len = take_u64(bytes, &mut at)? as usize;
        let n = take_u32(bytes, &mut at)? as usize;
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let tag = take_u8(bytes, &mut at)?;
            if tag == 2 {
                let kv_len = take_u32(bytes, &mut at)? as usize;
                let kv_heads = take_u32(bytes, &mut at)? as usize;
                let head_dim = take_u32(bytes, &mut at)? as usize;
                let k = take_f32(bytes, &mut at)?;
                let v = take_f32(bytes, &mut at)?;
                anyhow::ensure!(
                    k.len() == kv_len * kv_heads * head_dim && k.len() == v.len(),
                    "kv shape"
                );
                layers.push(CacheLayer {
                    full: true,
                    k,
                    v,
                    kv_heads,
                    head_dim,
                    conv: Vec::new(),
                    state: Vec::new(),
                    #[cfg(feature = "metal")]
                    gpu: None,
                    #[cfg(feature = "metal")]
                    gpu_kv: None,
                });
            } else if tag == 1 {
                let conv = take_f32(bytes, &mut at)?;
                let state = take_f32(bytes, &mut at)?;
                layers.push(CacheLayer {
                    full: false,
                    k: Vec::new(),
                    v: Vec::new(),
                    kv_heads: 0,
                    head_dim: 0,
                    conv,
                    state,
                    #[cfg(feature = "metal")]
                    gpu: None,
                    #[cfg(feature = "metal")]
                    gpu_kv: None,
                });
            } else {
                anyhow::bail!("bad kv layer tag {tag}");
            }
        }
        anyhow::ensure!(at == bytes.len(), "kv trailing bytes");
        Ok(Self { layers, len })
    }
}

fn push_f32(out: &mut Vec<u8>, values: &[f32]) {
    out.extend_from_slice(&(values.len() as u32).to_le_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn take_u8(bytes: &[u8], at: &mut usize) -> Result<u8> {
    anyhow::ensure!(*at < bytes.len(), "kv truncated");
    let value = bytes[*at];
    *at += 1;
    Ok(value)
}

fn take_u32(bytes: &[u8], at: &mut usize) -> Result<u32> {
    anyhow::ensure!(*at + 4 <= bytes.len(), "kv truncated");
    let value = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap());
    *at += 4;
    Ok(value)
}

fn take_u64(bytes: &[u8], at: &mut usize) -> Result<u64> {
    anyhow::ensure!(*at + 8 <= bytes.len(), "kv truncated");
    let value = u64::from_le_bytes(bytes[*at..*at + 8].try_into().unwrap());
    *at += 8;
    Ok(value)
}

fn take_f32(bytes: &[u8], at: &mut usize) -> Result<Vec<f32>> {
    let n = take_u32(bytes, at)? as usize;
    anyhow::ensure!(*at + n * 4 <= bytes.len(), "kv truncated");
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        values.push(f32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap()));
        *at += 4;
    }
    Ok(values)
}

impl CacheLayer {
    fn linear_host(&self) -> (Vec<f32>, Vec<f32>) {
        #[cfg(feature = "metal")]
        if let Some(gpu) = &self.gpu {
            return gpu.read_host();
        }
        (self.conv.clone(), self.state.clone())
    }

    fn full_host(&self) -> (usize, usize, usize, Vec<f32>, Vec<f32>) {
        #[cfg(feature = "metal")]
        if let Some(kv) = &self.gpu_kv {
            return kv.read_host();
        }
        let width = self.kv_heads.saturating_mul(self.head_dim).max(1);
        let kv_len = if self.k.is_empty() { 0 } else { self.k.len() / width };
        (kv_len, self.kv_heads, self.head_dim, self.k.clone(), self.v.clone())
    }
}

struct Store {
    maps: Vec<Mmap>,
    tensors: HashMap<String, TensorLoc>,
}

struct TensorLoc {
    shard: usize,
    start: usize,
    shape: Vec<usize>,
    dtype: DTypeTag,
}

#[derive(Clone, Copy, PartialEq)]
enum DTypeTag {
    U32,
    Bf16,
    F32,
}

impl Qwen35 {
    /// Load `language_model.*` from an MLX model directory.
    pub fn load(dir: impl AsRef<Path>, device: Device) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg: Value = serde_json::from_reader(
            File::open(dir.join("config.json")).context("open config.json")?,
        )?;
        let text = cfg.get("text_config").unwrap_or(&cfg);
        let hidden = json_usize(text, "hidden_size")?;
        let layers_n = json_usize(text, "num_hidden_layers")?;
        let _intermediate = json_usize(text, "intermediate_size")?;
        let vocab = json_usize(text, "vocab_size")?;
        let heads = json_usize(text, "num_attention_heads")?;
        let kv_heads = json_usize(text, "num_key_value_heads")?;
        let head_dim = json_usize(text, "head_dim")?;
        let k_heads = json_usize(text, "linear_num_key_heads")?;
        let v_heads = json_usize(text, "linear_num_value_heads")?;
        let k_dim = json_usize(text, "linear_key_head_dim")?;
        let v_dim = json_usize(text, "linear_value_head_dim")?;
        let kernel = json_usize(text, "linear_conv_kernel_dim")?;
        let eps = json_f32(text, "rms_norm_eps")?;
        let rope = text
            .get("rope_parameters")
            .and_then(|v| v.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .context("rope_theta")? as f32;
        let partial = text
            .get("rope_parameters")
            .and_then(|v| v.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25) as f32;
        let rotary_dim = (head_dim as f32 * partial) as usize;
        let eos = json_usize(text, "eos_token_id")? as u32;
        let kinds = text
            .get("layer_types")
            .and_then(|v| v.as_array())
            .context("layer_types")?;
        anyhow::ensure!(kinds.len() == layers_n, "layer_types length");
        let q = cfg.get("quantization_config").context("quantization_config")?;
        anyhow::ensure!(json_usize(q, "bits")? == BITS, "only 8-bit affine");
        anyhow::ensure!(json_usize(q, "group_size")? == GROUP, "only group 64");
        anyhow::ensure!(
            q.get("mode").and_then(|v| v.as_str()) == Some("affine"),
            "only affine quantization"
        );
        let _ = eps;

        let store = Store::open(dir)?;
        let prefix = "language_model.model";
        let mut layers = Vec::with_capacity(layers_n);
        for i in 0..layers_n {
            let p = format!("{prefix}.layers.{i}");
            let kind = kinds[i].as_str().context("layer type")?;
            let mixer = if kind == "linear_attention" {
                let conv_dim = k_heads * k_dim * 2 + v_heads * v_dim;
                Mixer::Linear(LinearAttn {
                    qkv: store.affine(&format!("{p}.linear_attn.in_proj_qkv"))?,
                    z: store.affine(&format!("{p}.linear_attn.in_proj_z"))?,
                    a: store.affine(&format!("{p}.linear_attn.in_proj_a"))?,
                    b: store.affine(&format!("{p}.linear_attn.in_proj_b"))?,
                    out: store.affine(&format!("{p}.linear_attn.out_proj"))?,
                    conv: store.f32(&format!("{p}.linear_attn.conv1d.weight"))?,
                    conv_dim,
                    kernel,
                    a_log: store.f32(&format!("{p}.linear_attn.A_log"))?,
                    dt_bias: store.f32(&format!("{p}.linear_attn.dt_bias"))?,
                    norm: store.f32(&format!("{p}.linear_attn.norm.weight"))?,
                    k_heads,
                    v_heads,
                    k_dim,
                    v_dim,
                    #[cfg(feature = "metal")]
                    metal_aux: Mutex::new(None),
                })
            } else if kind == "full_attention" {
                Mixer::Full(FullAttn {
                    q: store.affine(&format!("{p}.self_attn.q_proj"))?,
                    k: store.affine(&format!("{p}.self_attn.k_proj"))?,
                    v: store.affine(&format!("{p}.self_attn.v_proj"))?,
                    o: store.affine(&format!("{p}.self_attn.o_proj"))?,
                    q_norm: store.f32(&format!("{p}.self_attn.q_norm.weight"))?,
                    k_norm: store.f32(&format!("{p}.self_attn.k_norm.weight"))?,
                    heads,
                    kv_heads,
                    head_dim,
                })
            } else {
                anyhow::bail!("unknown layer type {kind}");
            };
            layers.push(Layer {
                input_norm: store.f32(&format!("{p}.input_layernorm.weight"))?,
                post_norm: store.f32(&format!("{p}.post_attention_layernorm.weight"))?,
                mixer,
                gate: store.affine(&format!("{p}.mlp.gate_proj"))?,
                up: store.affine(&format!("{p}.mlp.up_proj"))?,
                down: store.affine(&format!("{p}.mlp.down_proj"))?,
                #[cfg(feature = "metal")]
                metal_linear: Mutex::new(None),
                #[cfg(feature = "metal")]
                metal_full: Mutex::new(None),
            });
        }
        eprintln!("==> qwen3.5 text tower: {layers_n} layers, vocab {vocab}");
        Ok(Self {
            embed: store.affine(&format!("{prefix}.embed_tokens"))?,
            head: store.affine("language_model.lm_head")?,
            final_norm: store.f32(&format!("{prefix}.norm.weight"))?,
            maps: store.maps,
            layers,
            eos,
            hidden,
            vocab,
            rope_theta: rope,
            rotary_dim,
            device,
        })
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }

    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn new_cache(&self) -> Qwen35Cache {
        Qwen35Cache {
            layers: self
                .layers
                .iter()
                .map(|layer| match &layer.mixer {
                    Mixer::Linear(attn) => CacheLayer {
                        full: false,
                        k: Vec::new(),
                        v: Vec::new(),
                        kv_heads: 0,
                        head_dim: 0,
                        conv: vec![0.0; attn.conv_dim * (attn.kernel - 1)],
                        state: vec![0.0; attn.v_heads * attn.k_dim * attn.v_dim],
                        #[cfg(feature = "metal")]
                        gpu: None,
                        #[cfg(feature = "metal")]
                        gpu_kv: None,
                    },
                    Mixer::Full(attn) => CacheLayer {
                        full: true,
                        k: Vec::new(),
                        v: Vec::new(),
                        kv_heads: attn.kv_heads,
                        head_dim: attn.head_dim,
                        conv: Vec::new(),
                        state: Vec::new(),
                        #[cfg(feature = "metal")]
                        gpu: None,
                        #[cfg(feature = "metal")]
                        gpu_kv: None,
                    },
                })
                .collect(),
            len: 0,
        }
    }

    /// Feed `tokens` at the cache's current position. Returns last-position logits.
    pub fn forward(&self, tokens: &[u32], cache: &mut Qwen35Cache) -> Result<Vec<f32>> {
        let last = self.forward_hidden(tokens, cache)?;
        self.logits(&last)
    }

    /// Feed `tokens` and greedily pick the next id (Metal: full tower+head one CB).
    pub fn forward_greedy(&self, tokens: &[u32], cache: &mut Qwen35Cache) -> Result<(u32, f32)> {
        #[cfg(feature = "metal")]
        if tokens.len() == 1 && !self.device.is_cpu() {
            let (token, _) = self.forward_greedy_metal(tokens, cache)?;
            return Ok(token);
        }
        let last = self.forward_hidden(tokens, cache)?;
        #[cfg(feature = "metal")]
        if !self.device.is_cpu() {
            let slot = self.head.ensure_quant(&self.maps)?;
            return crate::qwen3_fuse::gemv_argmax(slot.as_ref().unwrap(), &last);
        }
        let logits = self.logits(&last)?;
        Ok(logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i as u32, *v))
            .unwrap())
    }

    /// Feed `tokens`. Prefill all but the last, then return that position's logits.
    pub fn forward_logits(&self, tokens: &[u32], cache: &mut Qwen35Cache) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "empty feed");
        #[cfg(feature = "metal")]
        if !self.device.is_cpu() {
            if tokens.len() > 1 {
                self.prefill(&tokens[..tokens.len() - 1], cache)?;
            }
            let (_, logits) = self.forward_greedy_metal(&tokens[tokens.len() - 1..], cache)?;
            return Ok(logits);
        }
        self.forward(tokens, cache)
    }

    #[cfg(feature = "metal")]
    fn forward_greedy_metal(
        &self,
        tokens: &[u32],
        cache: &mut Qwen35Cache,
    ) -> Result<((u32, f32), Vec<f32>)> {
        anyhow::ensure!(tokens.len() == 1, "greedy metal decode");
        let id = tokens[0];
        anyhow::ensure!((id as usize) < self.vocab, "token {id} out of vocab");
        let prof = std::env::var("PAGODA_PROFILE").is_ok();
        let t0 = std::time::Instant::now();
        let embed = self.embed.quant_cloned(&self.maps)?;
        let t_dequant = t0.elapsed().as_secs_f64();
        let pos0 = cache.len;
        let limit = std::env::var("PAGODA_QWEN_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.layers.len())
            .min(self.layers.len());
        let t = std::time::Instant::now();
        self.ensure_metal_layers(limit, cache)?;
        for li in 0..limit {
            if matches!(self.layers[li].mixer, Mixer::Full(_)) {
                let kv = cache.layers[li].gpu_kv.as_mut().unwrap();
                kv.ensure_cap(kv.len + 1)?;
                kv.len += 1;
            }
        }
        let (cos, sin) = rope_cos_sin(pos0, self.rotary_dim, self.rope_theta);
        let lin_guards: Vec<_> = (0..limit)
            .filter(|&li| matches!(self.layers[li].mixer, Mixer::Linear(_)))
            .map(|li| {
                (
                    li,
                    self.layers[li]
                        .metal_linear
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()),
                )
            })
            .collect();
        let full_guards: Vec<_> = (0..limit)
            .filter(|&li| matches!(self.layers[li].mixer, Mixer::Full(_)))
            .map(|li| {
                (
                    li,
                    self.layers[li]
                        .metal_full
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()),
                )
            })
            .collect();
        let mut decode = Vec::with_capacity(limit);
        let mut lin_i = 0usize;
        let mut full_i = 0usize;
        for li in 0..limit {
            match &self.layers[li].mixer {
                Mixer::Linear(_) => {
                    decode.push(crate::qwen3_fuse::DecodeLayer::Linear(
                        lin_guards[lin_i].1.as_ref().unwrap(),
                        cache.layers[li].gpu.as_ref().unwrap(),
                    ));
                    lin_i += 1;
                }
                Mixer::Full(_) => {
                    decode.push(crate::qwen3_fuse::DecodeLayer::Full(
                        full_guards[full_i].1.as_ref().unwrap(),
                        cache.layers[li].gpu_kv.as_ref().unwrap(),
                    ));
                    full_i += 1;
                }
            }
        }
        let (lin_meta, full_meta, mid_n) = {
            let first_lin = lin_guards[0].1.as_ref().unwrap();
            let first_full = full_guards[0].1.as_ref().unwrap();
            (first_lin.meta, first_full.meta, first_lin.gate.n_out())
        };
        let t_setup = t0.elapsed().as_secs_f64();
        let final_norm = crate::qwen3_fuse::cached_final_norm(&self.final_norm)?;
        let head = self.head.quant_cloned(&self.maps)?;
        if prof {
            eprintln!(
                "  cpu dequant {:.4}s setup {:.4}s",
                t_dequant,
                t_setup - t_dequant
            );
        }
        let out = crate::qwen3_fuse::fused_decode_logits(
            &[],
            &decode,
            &cos,
            &sin,
            &lin_meta,
            &full_meta,
            mid_n,
            final_norm,
            &head,
            Some((&embed, id)),
        )?;
        cache.len += 1;
        if std::env::var("PAGODA_QWEN_DEBUG").is_ok() {
            eprintln!("  fused greedy {:.3}s", t.elapsed().as_secs_f64());
        }
        let token = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, v)| (i as u32, *v))
            .context("empty logits")?;
        Ok((token, out))
    }

    /// Teacher-forced next-token for each of `tokens` (1..=4) in one forward.
    /// Big projections use the batched GEMM; delta and attention stay in order.
    #[cfg(feature = "metal")]
    pub fn forward_verify(
        &self,
        tokens: &[u32],
        cache: &mut Qwen35Cache,
    ) -> Result<Vec<(u32, f32)>> {
        anyhow::ensure!((1..=4).contains(&tokens.len()), "verify batch");
        anyhow::ensure!(!self.device.is_cpu(), "verify is metal-only");
        for &id in tokens {
            anyhow::ensure!((id as usize) < self.vocab, "token {id} out of vocab");
        }
        let batch = tokens.len();
        let pos0 = cache.len;
        let limit = self.layers.len();
        self.ensure_metal_layers(limit, cache)?;
        let mut kv_bases = Vec::new();
        for li in 0..limit {
            if matches!(self.layers[li].mixer, Mixer::Full(_)) {
                let kv = cache.layers[li].gpu_kv.as_mut().unwrap();
                kv_bases.push(kv.len);
                kv.ensure_cap(kv.len + batch)?;
                kv.len += batch;
            }
        }
        let mut cos = Vec::with_capacity(batch * self.rotary_dim);
        let mut sin = Vec::with_capacity(batch * self.rotary_dim);
        for t in 0..batch {
            let (c, s) = rope_cos_sin(pos0 + t, self.rotary_dim, self.rope_theta);
            cos.extend(c);
            sin.extend(s);
        }
        let lin_guards: Vec<_> = (0..limit)
            .filter(|&li| matches!(self.layers[li].mixer, Mixer::Linear(_)))
            .map(|li| {
                self.layers[li]
                    .metal_linear
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
            })
            .collect();
        let full_guards: Vec<_> = (0..limit)
            .filter(|&li| matches!(self.layers[li].mixer, Mixer::Full(_)))
            .map(|li| {
                self.layers[li]
                    .metal_full
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
            })
            .collect();
        let mut decode = Vec::with_capacity(limit);
        let mut lin_i = 0usize;
        let mut full_i = 0usize;
        for li in 0..limit {
            match &self.layers[li].mixer {
                Mixer::Linear(_) => {
                    decode.push(crate::qwen3_fuse::DecodeLayer::Linear(
                        lin_guards[lin_i].as_ref().unwrap(),
                        cache.layers[li].gpu.as_ref().unwrap(),
                    ));
                    lin_i += 1;
                }
                Mixer::Full(_) => {
                    decode.push(crate::qwen3_fuse::DecodeLayer::Full(
                        full_guards[full_i].as_ref().unwrap(),
                        cache.layers[li].gpu_kv.as_ref().unwrap(),
                    ));
                    full_i += 1;
                }
            }
        }
        let (lin_meta, full_meta, mid_n) = {
            let first_lin = lin_guards[0].as_ref().unwrap();
            let first_full = full_guards[0].as_ref().unwrap();
            (first_lin.meta, first_full.meta, first_lin.gate.n_out())
        };
        let final_norm = crate::qwen3_fuse::cached_final_norm(&self.final_norm)?;
        let head = self.head.quant_cloned(&self.maps)?;
        let embed = self.embed.quant_cloned(&self.maps)?;
        let out = crate::qwen3_fuse::fused_verify(
            &embed,
            tokens,
            &decode,
            &kv_bases,
            &cos,
            &sin,
            &lin_meta,
            &full_meta,
            mid_n,
            final_norm,
            &head,
        )?;
        cache.len += batch;
        Ok(out)
    }

    /// Feed `tokens` into the cache. One setup, commits of 64 tokens, no lm_head.
    #[cfg(feature = "metal")]
    pub fn prefill(&self, tokens: &[u32], cache: &mut Qwen35Cache) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        anyhow::ensure!(!self.device.is_cpu(), "prefill is metal-only");
        for &id in tokens {
            anyhow::ensure!((id as usize) < self.vocab, "token {id} out of vocab");
        }
        let n = tokens.len();
        let pos0 = cache.len;
        let limit = self.layers.len();
        self.ensure_metal_layers(limit, cache)?;
        let mut kv_bases = Vec::new();
        for li in 0..limit {
            if matches!(self.layers[li].mixer, Mixer::Full(_)) {
                let kv = cache.layers[li].gpu_kv.as_mut().unwrap();
                kv_bases.push(kv.len);
                kv.ensure_cap(kv.len + n)?;
                kv.len += n;
            }
        }
        let mut cos = Vec::with_capacity(n * self.rotary_dim);
        let mut sin = Vec::with_capacity(n * self.rotary_dim);
        for t in 0..n {
            let (c, s) = rope_cos_sin(pos0 + t, self.rotary_dim, self.rope_theta);
            cos.extend(c);
            sin.extend(s);
        }
        let lin_guards: Vec<_> = (0..limit)
            .filter(|&li| matches!(self.layers[li].mixer, Mixer::Linear(_)))
            .map(|li| {
                self.layers[li]
                    .metal_linear
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
            })
            .collect();
        let full_guards: Vec<_> = (0..limit)
            .filter(|&li| matches!(self.layers[li].mixer, Mixer::Full(_)))
            .map(|li| {
                self.layers[li]
                    .metal_full
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
            })
            .collect();
        let mut decode = Vec::with_capacity(limit);
        let mut lin_i = 0usize;
        let mut full_i = 0usize;
        for li in 0..limit {
            match &self.layers[li].mixer {
                Mixer::Linear(_) => {
                    decode.push(crate::qwen3_fuse::DecodeLayer::Linear(
                        lin_guards[lin_i].as_ref().unwrap(),
                        cache.layers[li].gpu.as_ref().unwrap(),
                    ));
                    lin_i += 1;
                }
                Mixer::Full(_) => {
                    decode.push(crate::qwen3_fuse::DecodeLayer::Full(
                        full_guards[full_i].as_ref().unwrap(),
                        cache.layers[li].gpu_kv.as_ref().unwrap(),
                    ));
                    full_i += 1;
                }
            }
        }
        let (lin_meta, full_meta, mid_n) = {
            let first_lin = lin_guards[0].as_ref().unwrap();
            let first_full = full_guards[0].as_ref().unwrap();
            (first_lin.meta, first_full.meta, first_lin.gate.n_out())
        };
        let embed = self.embed.quant_cloned(&self.maps)?;
        crate::qwen3_fuse::fused_prefill(
            &embed,
            tokens,
            &decode,
            &kv_bases,
            &cos,
            &sin,
            &lin_meta,
            &full_meta,
            mid_n,
        )?;
        cache.len += n;
        Ok(())
    }

    fn forward_hidden(&self, tokens: &[u32], cache: &mut Qwen35Cache) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "empty feed");
        let seq = tokens.len();
        let mut x = Vec::with_capacity(seq * self.hidden);
        for &id in tokens {
            anyhow::ensure!((id as usize) < self.vocab, "token {id} out of vocab");
            let mut row = vec![0.0; self.hidden];
            self.embed.dequant_rows(&self.maps, id as usize, 1, &mut row)?;
            x.extend_from_slice(&row);
        }
        let pos0 = cache.len;
        let limit = std::env::var("PAGODA_QWEN_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(self.layers.len());
        #[cfg(feature = "metal")]
        if seq == 1 && !self.device.is_cpu() {
            x = self.forward_hidden_metal(&x, pos0, limit, cache)?;
        } else {
            for (li, layer) in self.layers.iter().enumerate().take(limit) {
                let t = std::time::Instant::now();
                x = layer.forward(self, &x, seq, pos0, &mut cache.layers[li])?;
                if std::env::var("PAGODA_QWEN_DEBUG").is_ok() {
                    eprintln!("  layer {li} {:.2}s", t.elapsed().as_secs_f64());
                }
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            for (li, layer) in self.layers.iter().enumerate().take(limit) {
                let t = std::time::Instant::now();
                x = layer.forward(self, &x, seq, pos0, &mut cache.layers[li])?;
                if std::env::var("PAGODA_QWEN_DEBUG").is_ok() {
                    eprintln!("  layer {li} {:.2}s", t.elapsed().as_secs_f64());
                }
            }
        }
        if limit < self.layers.len() {
            return Ok(vec![0.0; self.hidden]);
        }
        for row in x.chunks_mut(self.hidden) {
            rms_zero(row, &self.final_norm);
        }
        let last = x[(seq - 1) * self.hidden..].to_vec();
        cache.len += seq;
        Ok(last)
    }

    /// Decode (`seq==1`): entire tower in one Metal command buffer.
    #[cfg(feature = "metal")]
    fn forward_hidden_metal(
        &self,
        x0: &[f32],
        pos0: usize,
        limit: usize,
        cache: &mut Qwen35Cache,
    ) -> Result<Vec<f32>> {
        let t = std::time::Instant::now();
        self.ensure_metal_layers(limit, cache)?;
        // Advance KV lengths before encode so kernels see the new slot.
        for li in 0..limit {
            if matches!(self.layers[li].mixer, Mixer::Full(_)) {
                let kv = cache.layers[li].gpu_kv.as_mut().unwrap();
                kv.ensure_cap(kv.len + 1)?;
                kv.len += 1;
            }
        }
        let (cos, sin) = rope_cos_sin(pos0, self.rotary_dim, self.rope_theta);
        let lin_guards: Vec<_> = (0..limit)
            .filter(|&li| matches!(self.layers[li].mixer, Mixer::Linear(_)))
            .map(|li| {
                (
                    li,
                    self.layers[li]
                        .metal_linear
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()),
                )
            })
            .collect();
        let full_guards: Vec<_> = (0..limit)
            .filter(|&li| matches!(self.layers[li].mixer, Mixer::Full(_)))
            .map(|li| {
                (
                    li,
                    self.layers[li]
                        .metal_full
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()),
                )
            })
            .collect();
        let mut decode = Vec::with_capacity(limit);
        let mut lin_i = 0usize;
        let mut full_i = 0usize;
        for li in 0..limit {
            match &self.layers[li].mixer {
                Mixer::Linear(_) => {
                    let owned = lin_guards[lin_i].1.as_ref().unwrap();
                    let gpu = cache.layers[li].gpu.as_ref().unwrap();
                    decode.push(crate::qwen3_fuse::DecodeLayer::Linear(owned, gpu));
                    lin_i += 1;
                }
                Mixer::Full(_) => {
                    let owned = full_guards[full_i].1.as_ref().unwrap();
                    let kv = cache.layers[li].gpu_kv.as_ref().unwrap();
                    decode.push(crate::qwen3_fuse::DecodeLayer::Full(owned, kv));
                    full_i += 1;
                }
            }
        }
        let (lin_meta, full_meta, mid_n) = {
            let first_lin = lin_guards[0].1.as_ref().unwrap();
            let first_full = full_guards[0].1.as_ref().unwrap();
            (first_lin.meta, first_full.meta, first_lin.gate.n_out())
        };
        let out = crate::qwen3_fuse::fused_decode_all(
            x0, &decode, &cos, &sin, &lin_meta, &full_meta, mid_n,
        )?;
        if std::env::var("PAGODA_QWEN_DEBUG").is_ok() {
            eprintln!("  fused tower {:.3}s", t.elapsed().as_secs_f64());
        }
        Ok(out)
    }

    #[cfg(feature = "metal")]
    fn ensure_metal_layers(&self, limit: usize, cache: &mut Qwen35Cache) -> Result<()> {
        let seq_len = cache.len;
        for li in 0..limit {
            let layer = &self.layers[li];
            match &layer.mixer {
                Mixer::Linear(attn) => {
                    let cl = &mut cache.layers[li];
                    if cl.gpu.is_none() {
                        cl.gpu = Some(crate::qwen3_fuse::GpuRecurrent::new(&cl.conv, &cl.state)?);
                    }
                    let mut slot = layer.metal_linear.lock().unwrap_or_else(|e| e.into_inner());
                    if slot.is_none() {
                        let mut aux = attn.metal_aux.lock().unwrap_or_else(|e| e.into_inner());
                        if aux.is_none() {
                            *aux = Some(crate::qwen3_fuse::LinearAttnAux::upload(
                                &attn.conv,
                                &attn.a_log,
                                &attn.dt_bias,
                                &attn.norm,
                            )?);
                        }
                        let aux = aux.as_ref().unwrap().clone();
                        *slot = Some(crate::qwen3_fuse::LinearLayerOwned {
                            qkv: attn.qkv.quant_cloned(&self.maps)?,
                            z: attn.z.quant_cloned(&self.maps)?,
                            a: attn.a.quant_cloned(&self.maps)?,
                            b: attn.b.quant_cloned(&self.maps)?,
                            out: attn.out.quant_cloned(&self.maps)?,
                            gate: layer.gate.quant_cloned(&self.maps)?,
                            up: layer.up.quant_cloned(&self.maps)?,
                            down: layer.down.quant_cloned(&self.maps)?,
                            aux,
                            input_norm: crate::qwen3_fuse::buf_f32_pub(&layer.input_norm)?,
                            post_norm: crate::qwen3_fuse::buf_f32_pub(&layer.post_norm)?,
                            meta: crate::qwen3_fuse::LinearLayerMeta {
                                hidden: self.hidden,
                                conv_dim: attn.conv_dim,
                                kernel: attn.kernel,
                                k_heads: attn.k_heads,
                                v_heads: attn.v_heads,
                                k_dim: attn.k_dim,
                                v_dim: attn.v_dim,
                            },
                        });
                    }
                }
                Mixer::Full(attn) => {
                    let cl = &mut cache.layers[li];
                    if cl.gpu_kv.is_none() {
                        let k = cl.k.clone();
                        let v = cl.v.clone();
                        cl.gpu_kv = Some(if k.is_empty() {
                            crate::qwen3_fuse::GpuKv::new(attn.kv_heads, attn.head_dim, 64)?
                        } else {
                            crate::qwen3_fuse::GpuKv::from_host(
                                attn.kv_heads,
                                attn.head_dim,
                                seq_len,
                                &k,
                                &v,
                            )?
                        });
                    }
                    let mut slot = layer.metal_full.lock().unwrap_or_else(|e| e.into_inner());
                    if slot.is_none() {
                        *slot = Some(crate::qwen3_fuse::FullLayerOwned {
                            q: attn.q.quant_cloned(&self.maps)?,
                            k: attn.k.quant_cloned(&self.maps)?,
                            v: attn.v.quant_cloned(&self.maps)?,
                            o: attn.o.quant_cloned(&self.maps)?,
                            gate: layer.gate.quant_cloned(&self.maps)?,
                            up: layer.up.quant_cloned(&self.maps)?,
                            down: layer.down.quant_cloned(&self.maps)?,
                            input_norm: crate::qwen3_fuse::buf_f32_pub(&layer.input_norm)?,
                            post_norm: crate::qwen3_fuse::buf_f32_pub(&layer.post_norm)?,
                            q_norm: crate::qwen3_fuse::buf_f32_pub(&attn.q_norm)?,
                            k_norm: crate::qwen3_fuse::buf_f32_pub(&attn.k_norm)?,
                            meta: crate::qwen3_fuse::FullLayerMeta {
                                hidden: self.hidden,
                                heads: attn.heads,
                                kv_heads: attn.kv_heads,
                                head_dim: attn.head_dim,
                                rotary_dim: self.rotary_dim,
                            },
                        });
                    }
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    fn forward_linear_stack(
        &self,
        range: std::ops::Range<usize>,
        x: &[f32],
        cache: &mut Qwen35Cache,
    ) -> Result<Vec<f32>> {
        for li in range.clone() {
            let layer = &self.layers[li];
            let Mixer::Linear(attn) = &layer.mixer else {
                anyhow::bail!("expected linear");
            };
            let cl = &mut cache.layers[li];
            if cl.gpu.is_none() {
                cl.gpu = Some(crate::qwen3_fuse::GpuRecurrent::new(&cl.conv, &cl.state)?);
            }
            let mut slot = layer.metal_linear.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                let mut aux = attn.metal_aux.lock().unwrap_or_else(|e| e.into_inner());
                if aux.is_none() {
                    *aux = Some(crate::qwen3_fuse::LinearAttnAux::upload(
                        &attn.conv,
                        &attn.a_log,
                        &attn.dt_bias,
                        &attn.norm,
                    )?);
                }
                let aux = aux.as_ref().unwrap().clone();
                *slot = Some(crate::qwen3_fuse::LinearLayerOwned {
                    qkv: attn.qkv.quant_cloned(&self.maps)?,
                    z: attn.z.quant_cloned(&self.maps)?,
                    a: attn.a.quant_cloned(&self.maps)?,
                    b: attn.b.quant_cloned(&self.maps)?,
                    out: attn.out.quant_cloned(&self.maps)?,
                    gate: layer.gate.quant_cloned(&self.maps)?,
                    up: layer.up.quant_cloned(&self.maps)?,
                    down: layer.down.quant_cloned(&self.maps)?,
                    aux,
                    input_norm: crate::qwen3_fuse::buf_f32_pub(&layer.input_norm)?,
                    post_norm: crate::qwen3_fuse::buf_f32_pub(&layer.post_norm)?,
                    meta: crate::qwen3_fuse::LinearLayerMeta {
                        hidden: self.hidden,
                        conv_dim: attn.conv_dim,
                        kernel: attn.kernel,
                        k_heads: attn.k_heads,
                        v_heads: attn.v_heads,
                        k_dim: attn.k_dim,
                        v_dim: attn.v_dim,
                    },
                });
            }
        }
        let guards: Vec<_> = range
            .clone()
            .map(|li| {
                self.layers[li]
                    .metal_linear
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
            })
            .collect();
        let owned: Vec<&crate::qwen3_fuse::LinearLayerOwned> = guards
            .iter()
            .map(|g| g.as_ref().unwrap())
            .collect();
        let gpus: Vec<&crate::qwen3_fuse::GpuRecurrent> = range
            .clone()
            .map(|li| cache.layers[li].gpu.as_ref().unwrap())
            .collect();
        crate::qwen3_fuse::fused_linear_stack_ref(x, &owned, &gpus)
    }

    fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        #[cfg(feature = "metal")]
        if !self.device.is_cpu() {
            return self.head.metal_gemv(&self.maps, hidden);
        }
        let mut logits = vec![0.0f32; self.vocab];
        let chunk = 4096;
        let mut rows = vec![0.0f32; chunk * self.hidden];
        let mut start = 0;
        while start < self.vocab {
            let n = (self.vocab - start).min(chunk);
            self.head
                .dequant_rows(&self.maps, start, n, &mut rows[..n * self.hidden])?;
            for r in 0..n {
                let w = &rows[r * self.hidden..(r + 1) * self.hidden];
                let mut dot = 0.0f32;
                for (a, b) in hidden.iter().zip(w) {
                    dot += a * b;
                }
                logits[start + r] = dot;
            }
            start += n;
        }
        Ok(logits)
    }

    fn linear(&self, x: &[f32], seq: usize, w: &Affine) -> Result<Vec<f32>> {
        #[cfg(feature = "metal")]
        if seq == 1 && !self.device.is_cpu() {
            return w.metal_gemv(&self.maps, x);
        }
        let xt = Tensor::from_vec(x.to_vec(), (seq, w.inp), &Device::Cpu)?;
        let y = if self.device.is_cpu() {
            let weight = w.dequant_all(&self.maps)?;
            let wt = Tensor::from_vec(weight, (w.out, w.inp), &Device::Cpu)?;
            xt.matmul(&wt.t()?)?
        } else {
            let wt = w.on_device(&self.maps, &self.device)?;
            let xt = xt.to_dtype(DType::F16)?.to_device(&self.device)?;
            xt.matmul(&wt.t()?)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?
        };
        Ok(y.flatten_all()?.to_vec1::<f32>()?)
    }

    /// Same `x`, several matrices, one Metal round-trip when `seq == 1`.
    fn linear_shared(&self, x: &[f32], seq: usize, ws: &[&Affine]) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "metal")]
        if seq == 1 && !self.device.is_cpu() {
            return Affine::metal_gemv_shared(&self.maps, ws, x);
        }
        ws.iter()
            .map(|w| self.linear(x, seq, w))
            .collect()
    }
}

impl Layer {
    fn forward(
        &self,
        model: &Qwen35,
        x: &[f32],
        seq: usize,
        pos0: usize,
        cache: &mut CacheLayer,
    ) -> Result<Vec<f32>> {
        #[cfg(feature = "metal")]
        if seq == 1 && !model.device.is_cpu() {
            if let Mixer::Linear(attn) = &self.mixer {
                return attn.forward_fused(model, x, &self.input_norm, &self.post_norm, &self.gate, &self.up, &self.down, cache);
            }
        }
        let mut normed = x.to_vec();
        for row in normed.chunks_mut(model.hidden) {
            rms_zero(row, &self.input_norm);
        }
        let mixed = match &self.mixer {
            Mixer::Linear(attn) => attn.forward(model, &normed, seq, cache)?,
            Mixer::Full(attn) => attn.forward(model, &normed, seq, pos0, cache)?,
        };
        #[cfg(feature = "metal")]
        if seq == 1 && !model.device.is_cpu() {
            let gate = self.gate.ensure_quant(&model.maps)?;
            let up = self.up.ensure_quant(&model.maps)?;
            let down = self.down.ensure_quant(&model.maps)?;
            return crate::qwen3_fuse::fused_residual_mlp(
                x,
                &mixed,
                &self.post_norm,
                gate.as_ref().unwrap(),
                up.as_ref().unwrap(),
                down.as_ref().unwrap(),
            );
        }
        let mut h = x.to_vec();
        for (a, b) in h.iter_mut().zip(&mixed) {
            *a += b;
        }
        let mut normed = h.clone();
        for row in normed.chunks_mut(model.hidden) {
            rms_zero(row, &self.post_norm);
        }
        let mut gate_up = model.linear_shared(&normed, seq, &[&self.gate, &self.up])?;
        let up = gate_up.pop().unwrap();
        let gate = gate_up.pop().unwrap();
        let mut mid = Vec::with_capacity(gate.len());
        for (g, u) in gate.iter().zip(&up) {
            mid.push(silu(*g) * u);
        }
        let down = model.linear(&mid, seq, &self.down)?;
        for (a, b) in h.iter_mut().zip(&down) {
            *a += b;
        }
        Ok(h)
    }
}

impl LinearAttn {
    #[cfg(feature = "metal")]
    fn forward_fused(
        &self,
        model: &Qwen35,
        x: &[f32],
        input_norm: &[f32],
        post_norm: &[f32],
        gate: &Affine,
        up: &Affine,
        down: &Affine,
        cache: &mut CacheLayer,
    ) -> Result<Vec<f32>> {
        let qkv = self.qkv.ensure_quant(&model.maps)?;
        let z = self.z.ensure_quant(&model.maps)?;
        let a = self.a.ensure_quant(&model.maps)?;
        let b = self.b.ensure_quant(&model.maps)?;
        let out = self.out.ensure_quant(&model.maps)?;
        let gate_q = gate.ensure_quant(&model.maps)?;
        let up_q = up.ensure_quant(&model.maps)?;
        let down_q = down.ensure_quant(&model.maps)?;
        {
            let mut aux = self.metal_aux.lock().unwrap_or_else(|e| e.into_inner());
            if aux.is_none() {
                *aux = Some(crate::qwen3_fuse::LinearAttnAux::upload(
                    &self.conv,
                    &self.a_log,
                    &self.dt_bias,
                    &self.norm,
                )?);
            }
        }
        if cache.gpu.is_none() {
            cache.gpu = Some(crate::qwen3_fuse::GpuRecurrent::new(&cache.conv, &cache.state)?);
        }
        let aux_guard = self.metal_aux.lock().unwrap_or_else(|e| e.into_inner());
        let aux = aux_guard.as_ref().unwrap();
        let gpu = cache.gpu.as_ref().unwrap();
        let mats = crate::qwen3_fuse::LinearLayerMats {
            qkv: qkv.as_ref().unwrap(),
            z: z.as_ref().unwrap(),
            a: a.as_ref().unwrap(),
            b: b.as_ref().unwrap(),
            out: out.as_ref().unwrap(),
            gate: gate_q.as_ref().unwrap(),
            up: up_q.as_ref().unwrap(),
            down: down_q.as_ref().unwrap(),
        };
        let meta = crate::qwen3_fuse::LinearLayerMeta {
            hidden: model.hidden,
            conv_dim: self.conv_dim,
            kernel: self.kernel,
            k_heads: self.k_heads,
            v_heads: self.v_heads,
            k_dim: self.k_dim,
            v_dim: self.v_dim,
        };
        crate::qwen3_fuse::fused_linear_layer(x, input_norm, post_norm, &mats, aux, gpu, &meta)
    }

    fn forward(
        &self,
        model: &Qwen35,
        x: &[f32],
        seq: usize,
        cache: &mut CacheLayer,
    ) -> Result<Vec<f32>> {
        let mut proj = model.linear_shared(x, seq, &[&self.qkv, &self.z, &self.b, &self.a])?;
        let a = proj.pop().unwrap();
        let b = proj.pop().unwrap();
        let z = proj.pop().unwrap();
        let qkv = proj.pop().unwrap();
        let conv_out = causal_conv(&qkv, seq, self.conv_dim, self.kernel, &self.conv, &mut cache.conv);
        let key_dim = self.k_heads * self.k_dim;
        let value_dim = self.v_heads * self.v_dim;
        let repeat = self.v_heads / self.k_heads;
        let mut outputs = vec![0.0f32; seq * value_dim];
        for t in 0..seq {
            let base = t * self.conv_dim;
            let q = &conv_out[base..base + key_dim];
            let k = &conv_out[base + key_dim..base + 2 * key_dim];
            let v = &conv_out[base + 2 * key_dim..base + self.conv_dim];
            let mut qh = repeat_heads(q, self.k_heads, self.k_dim, repeat);
            let mut kh = repeat_heads(k, self.k_heads, self.k_dim, repeat);
            l2norm_heads(&mut qh, self.v_heads, self.k_dim);
            l2norm_heads(&mut kh, self.v_heads, self.k_dim);
            let scale = (self.k_dim as f32).sqrt();
            for q in &mut qh {
                *q /= scale;
            }
            for h in 0..self.v_heads {
                let g = -self.a_log[h].exp()
                    * softplus(a[t * self.v_heads + h] + self.dt_bias[h]);
                let decay = g.exp();
                let beta = sigmoid(b[t * self.v_heads + h]);
                let q = &qh[h * self.k_dim..(h + 1) * self.k_dim];
                let k = &kh[h * self.k_dim..(h + 1) * self.k_dim];
                let v = &v[h * self.v_dim..(h + 1) * self.v_dim];
                let state = &mut cache.state[h * self.k_dim * self.v_dim..(h + 1) * self.k_dim * self.v_dim];
                for s in state.iter_mut() {
                    *s *= decay;
                }
                let mut mem = vec![0.0f32; self.v_dim];
                for (ki, k) in k.iter().copied().enumerate() {
                    let row = &state[ki * self.v_dim..(ki + 1) * self.v_dim];
                    for (m, s) in mem.iter_mut().zip(row) {
                        *m += s * k;
                    }
                }
                let mut delta = vec![0.0f32; self.v_dim];
                for (d, (vv, m)) in delta.iter_mut().zip(v.iter().zip(&mem)) {
                    *d = (vv - m) * beta;
                }
                for (ki, k) in k.iter().copied().enumerate() {
                    let row = &mut state[ki * self.v_dim..(ki + 1) * self.v_dim];
                    for (s, d) in row.iter_mut().zip(&delta) {
                        *s += k * d;
                    }
                }
                let out = &mut outputs[t * value_dim + h * self.v_dim..t * value_dim + (h + 1) * self.v_dim];
                for (ki, q) in q.iter().copied().enumerate() {
                    let row = &state[ki * self.v_dim..(ki + 1) * self.v_dim];
                    for (o, s) in out.iter_mut().zip(row) {
                        *o += s * q;
                    }
                }
                // gated RMSNorm per head
                let gate = &z[t * value_dim + h * self.v_dim..t * value_dim + (h + 1) * self.v_dim];
                rms_gated(out, &self.norm, gate);
            }
        }
        model.linear(&outputs, seq, &self.out)
    }
}

impl FullAttn {
    fn forward(
        &self,
        model: &Qwen35,
        x: &[f32],
        seq: usize,
        pos0: usize,
        cache: &mut CacheLayer,
    ) -> Result<Vec<f32>> {
        let mut qkv = model.linear_shared(x, seq, &[&self.q, &self.k, &self.v])?;
        let v_all = qkv.pop().unwrap();
        let k_all = qkv.pop().unwrap();
        let q_all = qkv.pop().unwrap();
        let mut attn_out = vec![0.0f32; seq * self.heads * self.head_dim];
        for t in 0..seq {
            let mut q = vec![0.0f32; self.heads * self.head_dim];
            let mut gate = vec![0.0f32; self.heads * self.head_dim];
            let raw = &q_all[t * self.q.out..(t + 1) * self.q.out];
            for h in 0..self.heads {
                let span = self.head_dim * 2;
                let src = &raw[h * span..(h + 1) * span];
                q[h * self.head_dim..(h + 1) * self.head_dim].copy_from_slice(&src[..self.head_dim]);
                gate[h * self.head_dim..(h + 1) * self.head_dim]
                    .copy_from_slice(&src[self.head_dim..]);
                rms_zero(
                    &mut q[h * self.head_dim..(h + 1) * self.head_dim],
                    &self.q_norm,
                );
            }
            let mut k = k_all[t * self.kv_heads * self.head_dim..(t + 1) * self.kv_heads * self.head_dim]
                .to_vec();
            for h in 0..self.kv_heads {
                rms_zero(
                    &mut k[h * self.head_dim..(h + 1) * self.head_dim],
                    &self.k_norm,
                );
            }
            let v = v_all[t * self.kv_heads * self.head_dim..(t + 1) * self.kv_heads * self.head_dim]
                .to_vec();
            let (cos, sin) = rope_cos_sin(pos0 + t, model.rotary_dim, model.rope_theta);
            for h in 0..self.heads {
                rope_inplace(&mut q[h * self.head_dim..(h + 1) * self.head_dim], &cos, &sin);
            }
            for h in 0..self.kv_heads {
                rope_inplace(&mut k[h * self.head_dim..(h + 1) * self.head_dim], &cos, &sin);
            }
            cache.k.extend_from_slice(&k);
            cache.v.extend_from_slice(&v);
            let len = cache.k.len() / (self.kv_heads * self.head_dim);
            let groups = self.heads / self.kv_heads;
            let scale = (self.head_dim as f32).sqrt();
            for h in 0..self.heads {
                let kv_h = h / groups;
                let qh = &q[h * self.head_dim..(h + 1) * self.head_dim];
                let mut scores = vec![0.0f32; len];
                let mut max_s = f32::NEG_INFINITY;
                for p in 0..len {
                    let kh = &cache.k[(p * self.kv_heads + kv_h) * self.head_dim
                        ..(p * self.kv_heads + kv_h + 1) * self.head_dim];
                    let mut dot = 0.0f32;
                    for (a, b) in qh.iter().zip(kh) {
                        dot += a * b;
                    }
                    let s = dot / scale;
                    scores[p] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                let oh = &mut attn_out[t * self.heads * self.head_dim + h * self.head_dim
                    ..t * self.heads * self.head_dim + (h + 1) * self.head_dim];
                for p in 0..len {
                    let vh = &cache.v[(p * self.kv_heads + kv_h) * self.head_dim
                        ..(p * self.kv_heads + kv_h + 1) * self.head_dim];
                    let w = scores[p] / sum;
                    for (o, vv) in oh.iter_mut().zip(vh) {
                        *o += w * vv;
                    }
                }
                for (i, o) in oh.iter_mut().enumerate() {
                    *o *= sigmoid(gate[h * self.head_dim + i]);
                }
            }
        }
        model.linear(&attn_out, seq, &self.o)
    }
}

impl Affine {
    #[cfg(feature = "metal")]
    fn ensure_quant(
        &self,
        maps: &[Mmap],
    ) -> Result<std::sync::MutexGuard<'_, Option<crate::qwen3_gemv::QuantMatrix>>> {
        let mut slot = self.quant.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            let words = self.inp / 4;
            let groups = self.inp / GROUP;
            let packed = read_u32s(&maps[self.packed.shard], self.packed.off, self.out * words)?;
            let scales_b = read_u16s(&maps[self.scales.shard], self.scales.off, self.out * groups)?;
            let biases_b = read_u16s(&maps[self.biases.shard], self.biases.off, self.out * groups)?;
            let scales: Vec<f32> = scales_b.into_iter().map(bf16_to_f32).collect();
            let biases: Vec<f32> = biases_b.into_iter().map(bf16_to_f32).collect();
            *slot = Some(crate::qwen3_gemv::QuantMatrix::upload(
                &packed, &scales, &biases, self.out, self.inp,
            )?);
        }
        Ok(slot)
    }

    #[cfg(feature = "metal")]
    fn quant_cloned(&self, maps: &[Mmap]) -> Result<crate::qwen3_gemv::QuantMatrix> {
        let slot = self.ensure_quant(maps)?;
        Ok(slot.as_ref().unwrap().clone())
    }

    #[cfg(feature = "metal")]
    fn metal_gemv(&self, maps: &[Mmap], x: &[f32]) -> Result<Vec<f32>> {
        self.quant_cloned(maps)?.gemv(x)
    }

    #[cfg(feature = "metal")]
    fn metal_gemv_shared(maps: &[Mmap], ws: &[&Affine], x: &[f32]) -> Result<Vec<Vec<f32>>> {
        let mats: Vec<crate::qwen3_gemv::QuantMatrix> = ws
            .iter()
            .map(|w| w.quant_cloned(maps))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<&crate::qwen3_gemv::QuantMatrix> = mats.iter().collect();
        crate::qwen3_gemv::QuantMatrix::gemv_shared(&refs, x)
    }

    /// F16 copy of the dequantized matrix on `device`. The first call pays
    /// the dequant; later calls reuse it. Decode is a GEMM over weights that
    /// already sit in GPU memory.
    fn on_device(&self, maps: &[Mmap], device: &Device) -> Result<Tensor> {
        let mut slot = self.gpu.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(t) = slot.as_ref() {
            return Ok(t.clone());
        }
        let weight = self.dequant_all(maps)?;
        let t = Tensor::from_vec(weight, (self.out, self.inp), &Device::Cpu)?
            .to_dtype(DType::F16)?
            .to_device(device)?;
        *slot = Some(t.clone());
        Ok(t)
    }

    fn dequant_all(&self, maps: &[Mmap]) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.out * self.inp];
        self.dequant_rows(maps, 0, self.out, &mut out)?;
        Ok(out)
    }

    fn dequant_rows(&self, maps: &[Mmap], start: usize, rows: usize, dst: &mut [f32]) -> Result<()> {
        anyhow::ensure!(start + rows <= self.out, "row range");
        anyhow::ensure!(dst.len() >= rows * self.inp, "dst short");
        let words = self.inp / 4;
        let groups = self.inp / GROUP;
        // Offsets are relative to a data section that is not necessarily aligned.
        // Read only the requested rows: the embedding and lm_head are hundreds of MB.
        let packed = read_u32s(
            &maps[self.packed.shard],
            self.packed.off + start * words * 4,
            rows * words,
        )?;
        let scales = read_u16s(
            &maps[self.scales.shard],
            self.scales.off + start * groups * 2,
            rows * groups,
        )?;
        let biases = read_u16s(
            &maps[self.biases.shard],
            self.biases.off + start * groups * 2,
            rows * groups,
        )?;
        for r in 0..rows {
            let row = &packed[r * words..(r + 1) * words];
            let srow = &scales[r * groups..(r + 1) * groups];
            let brow = &biases[r * groups..(r + 1) * groups];
            let out = &mut dst[r * self.inp..(r + 1) * self.inp];
            let mut i = 0;
            for word in row {
                for k in 0..4 {
                    let code = ((word >> (8 * k)) & 255) as f32;
                    let g = i / GROUP;
                    out[i] = bf16_to_f32(brow[g]) + bf16_to_f32(srow[g]) * code;
                    i += 1;
                }
            }
        }
        Ok(())
    }
}

impl Store {
    fn open(dir: &Path) -> Result<Self> {
        let index: Value = serde_json::from_reader(
            File::open(dir.join("model.safetensors.index.json")).context("safetensors index")?,
        )?;
        let map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .context("weight_map")?;
        let mut files: Vec<String> = map.values().filter_map(|v| v.as_str().map(str::to_string)).collect();
        files.sort();
        files.dedup();
        let mut maps = Vec::new();
        let mut tensors = HashMap::new();
        for (shard, name) in files.iter().enumerate() {
            let file = File::open(dir.join(name)).with_context(|| format!("open {name}"))?;
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {name}"))?;
            let (header, data_base) = safetensors_header(&mmap)?;
            let obj = header.as_object().context("header object")?;
            for (key, meta) in obj {
                if key == "__metadata__" {
                    continue;
                }
                if map.get(key).and_then(|v| v.as_str()) != Some(name.as_str()) {
                    continue;
                }
                let dtype = meta.get("dtype").and_then(|v| v.as_str()).unwrap_or("");
                let shape = meta
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .context("shape")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect::<Vec<_>>();
                let offsets = meta
                    .get("data_offsets")
                    .and_then(|v| v.as_array())
                    .context("offsets")?;
                let start = data_base + offsets[0].as_u64().unwrap_or(0) as usize;
                let tag = match dtype {
                    "U32" => DTypeTag::U32,
                    "BF16" => DTypeTag::Bf16,
                    "F32" => DTypeTag::F32,
                    other => anyhow::bail!("unsupported dtype {other} for {key}"),
                };
                tensors.insert(
                    key.clone(),
                    TensorLoc {
                        shard,
                        start,
                        shape,
                        dtype: tag,
                    },
                );
            }
            maps.push(mmap);
        }
        Ok(Self { maps, tensors })
    }

    fn loc(&self, name: &str) -> Result<&TensorLoc> {
        self.tensors.get(name).with_context(|| format!("missing {name}"))
    }

    fn affine(&self, name: &str) -> Result<Affine> {
        let w = self.loc(&format!("{name}.weight"))?;
        let s = self.loc(&format!("{name}.scales"))?;
        let b = self.loc(&format!("{name}.biases"))?;
        anyhow::ensure!(w.dtype == DTypeTag::U32, "{name} weight is not packed u32");
        let out = w.shape[0];
        let inp = w.shape[1] * (32 / BITS);
        let groups = s.shape[1];
        anyhow::ensure!(inp / GROUP == groups, "{name} group mismatch");
        Ok(Affine {
            packed: Buf { shard: w.shard, off: w.start },
            scales: Buf { shard: s.shard, off: s.start },
            biases: Buf { shard: b.shard, off: b.start },
            out,
            inp,
            gpu: Mutex::new(None),
            #[cfg(feature = "metal")]
            quant: Mutex::new(None),
        })
    }

    fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let loc = self.loc(name)?;
        let n: usize = loc.shape.iter().product();
        let map = &self.maps[loc.shard];
        let mut out = Vec::with_capacity(n);
        match loc.dtype {
            DTypeTag::Bf16 => {
                let raw = read_u16s(map, loc.start, n)?;
                out.extend(raw.iter().copied().map(bf16_to_f32));
            }
            DTypeTag::F32 => {
                let raw = read_u32s(map, loc.start, n)?;
                out.extend(raw.iter().map(|u| f32::from_bits(*u)));
            }
            DTypeTag::U32 => anyhow::bail!("{name} is packed, not a plain tensor"),
        }
        Ok(out)
    }
}

fn safetensors_header(mmap: &Mmap) -> Result<(Value, usize)> {
    anyhow::ensure!(mmap.len() >= 8, "safetensors too small");
    let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
    let json = std::str::from_utf8(&mmap[8..8 + n]).context("header utf8")?;
    let value: Value = serde_json::from_str(json).context("header json")?;
    Ok((value, 8 + n))
}

fn read_u32s(map: &Mmap, off: usize, n: usize) -> Result<Vec<u32>> {
    let bytes = n * 4;
    anyhow::ensure!(off + bytes <= map.len(), "u32 slice oob");
    Ok(map[off..off + bytes]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_u16s(map: &Mmap, off: usize, n: usize) -> Result<Vec<u16>> {
    let bytes = n * 2;
    anyhow::ensure!(off + bytes <= map.len(), "bf16 slice oob");
    Ok(map[off..off + bytes]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn json_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(|x| x.as_u64().or_else(|| x.as_i64().map(|i| i as u64)))
        .with_context(|| format!("missing {key}"))
        .map(|n| n as usize)
}

fn json_f32(v: &Value, key: &str) -> Result<f32> {
    v.get(key)
        .and_then(|x| x.as_f64())
        .with_context(|| format!("missing {key}"))
        .map(|n| n as f32)
}

fn rms_zero(x: &mut [f32], weight: &[f32]) {
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += v * v;
    }
    let inv = (ss / x.len() as f32 + 1e-6).sqrt().recip();
    for (i, v) in x.iter_mut().enumerate() {
        *v = *v * inv * weight[i];
    }
}

fn rms_gated(x: &mut [f32], weight: &[f32], gate: &[f32]) {
    let mut ss = 0.0f32;
    for v in x.iter() {
        ss += v * v;
    }
    let inv = (ss / x.len() as f32 + 1e-6).sqrt().recip();
    for (i, v) in x.iter_mut().enumerate() {
        *v = *v * inv * weight[i] * silu(gate[i]);
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn repeat_heads(src: &[f32], heads: usize, dim: usize, repeat: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(heads * repeat * dim);
    for h in 0..heads {
        for _ in 0..repeat {
            out.extend_from_slice(&src[h * dim..(h + 1) * dim]);
        }
    }
    out
}

fn l2norm_heads(x: &mut [f32], heads: usize, dim: usize) {
    for h in 0..heads {
        let row = &mut x[h * dim..(h + 1) * dim];
        let mut ss = 0.0f32;
        for v in row.iter() {
            ss += v * v;
        }
        let inv = (ss + 1e-6).sqrt().recip();
        for v in row {
            *v *= inv;
        }
    }
}

fn causal_conv(
    x: &[f32],
    seq: usize,
    channels: usize,
    kernel: usize,
    weight: &[f32],
    state: &mut [f32],
) -> Vec<f32> {
    let hist = kernel - 1;
    let mut out = vec![0.0f32; seq * channels];
    for t in 0..seq {
        for c in 0..channels {
            let mut acc = 0.0f32;
            for k in 0..kernel {
                let src = t as isize - hist as isize + k as isize;
                let sample = if src < 0 {
                    let idx = (hist as isize + src) as usize;
                    if (hist as isize + src) < 0 {
                        0.0
                    } else {
                        state[c * hist + idx]
                    }
                } else {
                    x[src as usize * channels + c]
                };
                acc += sample * weight[c * kernel + k];
            }
            out[t * channels + c] = silu(acc);
        }
    }
    if seq >= hist {
        for c in 0..channels {
            for k in 0..hist {
                state[c * hist + k] = x[(seq - hist + k) * channels + c];
            }
        }
    } else {
        let keep = hist;
        let mut window = vec![0.0f32; channels * (keep + seq)];
        for c in 0..channels {
            for k in 0..keep {
                window[c * (keep + seq) + k] = state[c * keep + k];
            }
            for t in 0..seq {
                window[c * (keep + seq) + keep + t] = x[t * channels + c];
            }
            let start = seq;
            for k in 0..keep {
                state[c * keep + k] = window[c * (keep + seq) + start + k];
            }
        }
    }
    out
}

fn rope_cos_sin(pos: usize, rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0.0f32; rotary_dim];
    let mut sin = vec![0.0f32; rotary_dim];
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f32 / rotary_dim as f32);
        let ang = pos as f32 * freq;
        let c = ang.cos();
        let s = ang.sin();
        cos[i] = c;
        cos[half + i] = c;
        sin[i] = s;
        sin[half + i] = s;
    }
    (cos, sin)
}

fn rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let n = cos.len();
    let half = n / 2;
    let mut rot = vec![0.0f32; n];
    for i in 0..half {
        rot[i] = -x[half + i];
        rot[half + i] = x[i];
    }
    for i in 0..n {
        x[i] = x[i] * cos[i] + rot[i] * sin[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_u8_matches_mlx_arange() {
        // mx.quantize(arange(64).reshape(1, 64), group_size=64, bits=8)
        // first packed word and the reconstructed first four values.
        let packed = [4093115391u32];
        let scale = bf16_round_trip(-0.24705882370471954);
        let bias = bf16_round_trip(63.0);
        let mut dst = [0.0f32; 4];
        // inp 4 is not a full group of 64; check the formula directly.
        for (i, code) in [255.0f32, 251.0, 247.0, 243.0].into_iter().enumerate() {
            dst[i] = bias + scale * code;
        }
        let expect = [0.0f32, 0.98823525, 1.9764705, 2.9647057];
        for (g, e) in dst.iter().zip(expect) {
            assert!((g - e).abs() < 1e-3, "{g} vs {e}");
        }
        assert_eq!(packed[0] & 255, 255);
    }

    #[test]
    fn kv_host_bytes_roundtrip() {
        let cache = Qwen35Cache {
            len: 2,
            layers: vec![
                CacheLayer {
                    full: false,
                    k: Vec::new(),
                    v: Vec::new(),
                    kv_heads: 0,
                    head_dim: 0,
                    conv: vec![1.0, 2.0],
                    state: vec![3.0],
                    #[cfg(feature = "metal")]
                    gpu: None,
                    #[cfg(feature = "metal")]
                    gpu_kv: None,
                },
                CacheLayer {
                    full: true,
                    k: vec![4.0, 5.0, 6.0, 7.0],
                    v: vec![8.0, 9.0, 10.0, 11.0],
                    kv_heads: 1,
                    head_dim: 2,
                    conv: Vec::new(),
                    state: Vec::new(),
                    #[cfg(feature = "metal")]
                    gpu: None,
                    #[cfg(feature = "metal")]
                    gpu_kv: None,
                },
            ],
        };
        let back = Qwen35Cache::from_bytes(&cache.to_bytes().unwrap()).unwrap();
        assert_eq!(back.len, 2);
        assert_eq!(back.layers[0].conv, vec![1.0, 2.0]);
        assert_eq!(back.layers[0].state, vec![3.0]);
        assert_eq!(back.layers[1].k, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(back.layers[1].kv_heads, 1);
        assert_eq!(back.layers[1].head_dim, 2);
    }

    fn bf16_round_trip(x: f32) -> f32 {
        let bits = (x.to_bits() + 0x8000) >> 16;
        bf16_to_f32(bits as u16)
    }
}
