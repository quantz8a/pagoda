// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! A batch-capable Llama-family causal LM built directly on candle-nn
//! primitives.
//!
//! Why vendored: `candle-transformers`' own `Llama` keeps its KV cache
//! private and advances exactly one sequence per forward. A serving engine
//! needs the opposite shape — many sequences, each with its own KV history,
//! advanced in **one** batched call — so the forward pass is re-implemented
//! here over the same HF checkpoint layout (`model.layers.{i}.self_attn.*`)
//! with the same math as candle 0.8 (F32 attention core, NeoX-style RoPE,
//! additive causal mask). `examples/e2e_batched_decode.rs` asserts logits
//! parity against candle's `Llama` on real weights.
//!
//! Two entry points:
//!
//! * [`OwnedLlama::forward_tokens`] — single-sequence prefill / suffix feed
//!   with a square causal mask (any length, any position).
//! * [`OwnedLlama::batch_decode`] — the continuous-batching step: many
//!   sessions, one fresh token each, one forward pass. Per-session KV
//!   histories are padded to the longest in the batch and masked out, so
//!   sequences of different lengths share the projection matmuls.

use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use candle_transformers::models::llama::{Config, Llama3RopeConfig, Llama3RopeType};
use candle_transformers::utils::repeat_kv;
use std::f32::consts::PI;

/// Per-session KV state: one `(k, v)` pair per layer, each of shape
/// `[1, n_kv_heads, len, head_dim]`. Candle tensors are reference-counted, so
/// cloning a session (checkpoint forking) shares all storage — only the
/// tokens appended after the fork allocate new memory.
#[derive(Clone)]
pub struct SessionKv {
    layers: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
}

impl SessionKv {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
            len: 0,
        }
    }

    /// Tokens pushed through the model so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// A graftable view restricted to the first `len` tokens: per-layer
    /// narrows that share storage with the original. Exact by causality —
    /// the KV at position `i` depends only on tokens `0..=i`, so a sliced
    /// prefix is bit-identical to a recomputed one. Appends to the view
    /// concatenate into fresh tensors and never disturb the original.
    pub fn prefix(&self, len: usize) -> Option<SessionKv> {
        if len > self.len {
            return None;
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            match layer {
                Some((k, v)) => layers.push(Some((
                    k.narrow(2, 0, len).ok()?,
                    v.narrow(2, 0, len).ok()?,
                ))),
                None => layers.push(None),
            }
        }
        Some(SessionKv { layers, len })
    }
}

/// Inverse RoPE frequencies, including Llama-3 style rope scaling (mirrors
/// candle 0.8 `llama::Cache::new`).
fn inv_freq(config: &Config) -> Vec<f32> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    let default = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / config.rope_theta.powf(i as f32 / head_dim as f32))
        .collect::<Vec<_>>();
    match &config.rope_scaling {
        None
        | Some(Llama3RopeConfig {
            rope_type: Llama3RopeType::Default,
            ..
        }) => default,
        Some(scaling) => {
            let low = scaling.original_max_position_embeddings as f32 / scaling.low_freq_factor;
            let high = scaling.original_max_position_embeddings as f32 / scaling.high_freq_factor;
            default
                .into_iter()
                .map(|freq| {
                    let wavelen = 2. * PI / freq;
                    if wavelen < high {
                        freq
                    } else if wavelen > low {
                        freq / scaling.factor
                    } else {
                        let smooth = (scaling.original_max_position_embeddings as f32 / wavelen
                            - scaling.low_freq_factor)
                            / (scaling.high_freq_factor - scaling.low_freq_factor);
                        (1. - smooth) * freq / scaling.factor + smooth * freq
                    }
                })
                .collect()
        }
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

#[derive(Debug, Clone)]
struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

#[derive(Debug, Clone)]
struct Block {
    rms1: RmsNorm,
    attn: Attention,
    rms2: RmsNorm,
    mlp: Mlp,
}

impl Mlp {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("gate_proj"))?,
            up: linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("up_proj"))?,
            down: linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        self.down.forward(&x).map_err(anyhow::Error::from)
    }
}

impl Attention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let size_in = config.hidden_size;
        let head_dim = config.hidden_size / config.num_attention_heads;
        Ok(Self {
            q_proj: linear_no_bias(size_in, head_dim * config.num_attention_heads, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(size_in, head_dim * config.num_key_value_heads, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(size_in, head_dim * config.num_key_value_heads, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(head_dim * config.num_attention_heads, size_in, vb.pp("o_proj"))?,
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            head_dim,
        })
    }

    /// Project x into per-head q/k/v: `[B, seq, hidden]` to
    /// `[B, heads, seq, head_dim]`.
    fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, seq, _hidden) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, seq, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok((q, k, v))
    }

    /// Attention core shared by both paths: F32 scores, optional additive
    /// mask, softmax, weighted sum, back to the input dtype.
    fn attend(&self, q: &Tensor, k: Tensor, v: Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let rep = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(k, rep)?;
        let v = repeat_kv(v, rep)?;
        let in_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            Some(mask) => att.broadcast_add(mask)?,
            None => att,
        };
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        // matmul requires the second operand contiguous for strided inputs.
        att.matmul(&v.contiguous()?)?
            .to_dtype(in_dtype)
            .map_err(anyhow::Error::from)
    }

    /// Merge heads back to `[B, seq, hidden]` and apply o_proj.
    fn combine(&self, y: &Tensor, b: usize, seq: usize) -> Result<Tensor> {
        let hidden = self.n_heads * self.head_dim;
        let y = y.transpose(1, 2)?.reshape((b, seq, hidden))?;
        self.o_proj.forward(&y).map_err(anyhow::Error::from)
    }
}

/// NeoX-style RoPE with a shared position offset (prefill / single-sequence
/// suffix feeds): exactly candle's own kernel.
fn rope_at(x: &Tensor, cos: &Tensor, sin: &Tensor, index_pos: usize, seq: usize) -> Result<Tensor> {
    let cos = cos.narrow(0, index_pos, seq)?;
    let sin = sin.narrow(0, index_pos, seq)?;
    candle_nn::rotary_emb::rope(x, &cos, &sin).map_err(anyhow::Error::from)
}

/// NeoX-style RoPE with a per-row position (batched decode): gather each
/// sequence's cos/sin row and apply the rotate-half identity elementwise,
/// which is what candle's `rope` kernel computes for a shared offset.
fn rope_rows(x: &Tensor, cos_rows: &Tensor, sin_rows: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let b = x.dim(0)?;
    let cos = cos_rows.reshape((b, 1, 1, half))?;
    let sin = sin_rows.reshape((b, 1, 1, half))?;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    // rotate_half(x) = cat(-x2, x1); y = x*cos + rotate_half(x)*sin.
    let y1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let y2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&y1, &y2], D::Minus1).map_err(anyhow::Error::from)
}

/// A Llama-family causal LM owning its weights and RoPE tables.
pub struct OwnedLlama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    cos: Tensor, // [max_position_embeddings, head_dim/2]
    sin: Tensor,
    n_kv_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    device: Device,
}

impl OwnedLlama {
    /// Load from a VarBuilder over HF safetensors weights.
    pub fn load(vb: VarBuilder, config: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let wte = embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(wte.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        let ln_f = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let blocks = (0..config.num_hidden_layers)
            .map(|i| Block::load(vb.pp(format!("model.layers.{i}")), config))
            .collect::<Result<Vec<_>>>()?;
        let head_dim = config.hidden_size / config.num_attention_heads;
        let theta = Tensor::new(inv_freq(config), device)?;
        let idx = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?;
        let freqs = idx.matmul(&theta.reshape((1, head_dim / 2))?)?;
        // RoPE tables live in the compute dtype: candle's rope kernel refuses
        // mixed dtypes, and the batched path re-casts gathered rows anyway.
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            cos,
            sin,
            n_kv_heads: config.num_key_value_heads,
            head_dim,
            max_position_embeddings: config.max_position_embeddings,
            device: device.clone(),
        })
    }

    pub fn new_cache(&self) -> SessionKv {
        SessionKv::new(self.blocks.len())
    }

    /// Single-sequence forward over `tokens` (any length), appended to
    /// `cache` at its current position. Returns last-position logits.
    pub fn forward_tokens(&self, tokens: &[u32], cache: &mut SessionKv) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            anyhow::bail!("empty feed");
        }
        let index_pos = cache.len;
        let seq = tokens.len();
        anyhow::ensure!(
            index_pos + seq <= self.max_position_embeddings,
            "context {} + {seq} exceeds max_position_embeddings {}",
            cache.len,
            self.max_position_embeddings
        );
        let input = Tensor::new(tokens, &self.device)?.reshape((1, seq))?;
        let mut x = self.wte.forward(&input)?;
        let causal = if seq > 1 {
            Some(self.causal_mask(seq, index_pos + seq)?)
        } else {
            None
        };
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward_single(&x, index_pos, layer, cache, self, causal.as_ref())?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&x)?.to_dtype(DType::F32)?;
        cache.len += seq;
        logits
            .flatten_all()?
            .to_vec1::<f32>()
            .map_err(anyhow::Error::from)
    }

    /// The continuous-batching decode step: `kvs.len()` sessions, one fresh
    /// token each, advanced in a single forward pass. Returns one logits
    /// vector per session, in order.
    ///
    /// Per-session KV histories differ in length, so K/V are padded to the
    /// longest history and the padding is masked out of the attention
    /// scores. That gather is O(batch * history) per step; a paged-attention
    /// kernel that reads the engine's KV blocks directly is the roadmap
    /// successor.
    pub fn batch_decode(&self, tokens: &[u32], kvs: &mut [&mut SessionKv]) -> Result<Vec<Vec<f32>>> {
        let b = kvs.len();
        anyhow::ensure!(b >= 2, "batch_decode needs at least two sessions");
        anyhow::ensure!(tokens.len() == b, "one token per session");
        for kv in kvs.iter() {
            anyhow::ensure!(
                kv.len < self.max_position_embeddings,
                "context {} exceeds max_position_embeddings {}",
                kv.len,
                self.max_position_embeddings
            );
        }
        // Atomic commit: run the whole forward on cheap Arc-shared clones of
        // the session caches and only write back on success, so a mid-forward
        // failure leaves every session's KV untouched and the per-session
        // fallback path stays correct (no double-fed tokens).
        let mut work: Vec<SessionKv> = kvs.iter().map(|kv| (*kv).clone()).collect();
        let mut work_refs: Vec<&mut SessionKv> = work.iter_mut().collect();
        let input = Tensor::new(tokens, &self.device)?.reshape((b, 1))?;
        let mut x = self.wte.forward(&input)?;
        let positions = Tensor::new(
            work_refs.iter().map(|kv| kv.len as u32).collect::<Vec<_>>(),
            &self.device,
        )?;
        let cos_rows = self.cos.index_select(&positions, 0)?.to_dtype(x.dtype())?;
        let sin_rows = self.sin.index_select(&positions, 0)?.to_dtype(x.dtype())?;
        let lmax = work_refs.iter().map(|kv| kv.len + 1).max().unwrap_or(0);
        // Additive padding mask: 0 over the real history, -inf over padding.
        let mut mask = vec![0f32; b * lmax];
        for (row, kv) in work_refs.iter().enumerate() {
            for j in (kv.len + 1)..lmax {
                mask[row * lmax + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::new(mask, &self.device)?.reshape((b, 1, 1, lmax))?;
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward_batch(&x, layer, &mut work_refs, self, &cos_rows, &sin_rows, &mask, lmax)?;
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?.to_dtype(DType::F32)?; // [B, 1, vocab]
        for kv in work_refs.iter_mut() {
            kv.len += 1;
        }
        let out = logits
            .reshape((b, logits.dim(D::Minus1)?))?
            .to_vec2::<f32>()?;
        for (kv, w) in kvs.iter_mut().zip(work) {
            **kv = w;
        }
        Ok(out)
    }

    /// Additive causal mask `[seq, kv_len]` for a suffix fed at
    /// `kv_len - seq`: query row `i` (absolute position `kv_len - seq + i`)
    /// attends keys `j <= kv_len - seq + i`, so the forbidden zone is
    /// `j > i + (kv_len - seq)`. softmax(x + mask) equals
    /// softmax(masked_fill(x, -inf)).
    fn causal_mask(&self, seq: usize, kv_len: usize) -> Result<Tensor> {
        let offset = kv_len - seq;
        let mut mask = vec![0f32; seq * kv_len];
        for i in 0..seq {
            for j in (i + 1 + offset)..kv_len {
                mask[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::new(mask, &self.device)?
            .reshape((seq, kv_len))
            .map_err(anyhow::Error::from)
    }
}

impl Block {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            rms1: rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: Attention::load(vb.pp("self_attn"), config)?,
            rms2: rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(vb.pp("mlp"), config)?,
        })
    }

    /// Single-sequence layer: RoPE at a shared offset, append K/V to the
    /// session cache, causal-masked attention.
    fn forward_single(
        &self,
        x: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut SessionKv,
        model: &OwnedLlama,
        causal: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq, _hidden) = x.dims3()?;
        let residual = x;
        let x = self.rms1.forward(x)?;
        let (q, k, v) = self.attn.project_qkv(&x)?;
        let q = rope_at(&q, &model.cos, &model.sin, index_pos, seq)?;
        let k = rope_at(&k, &model.cos, &model.sin, index_pos, seq)?;
        let (k, v) = match &cache.layers[layer] {
            Some((ck, cv)) => (
                Tensor::cat(&[ck, &k], 2)?.contiguous()?,
                Tensor::cat(&[cv, &v], 2)?.contiguous()?,
            ),
            None => (k, v),
        };
        cache.layers[layer] = Some((k.clone(), v.clone()));
        let y = self.attn.attend(&q, k, v, causal)?;
        let x = (self.attn.combine(&y, b, seq)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    /// Batched decode layer: per-row RoPE positions, per-session K/V
    /// write-back, then a padded + masked batch attention.
    fn forward_batch(
        &self,
        x: &Tensor,
        layer: usize,
        kvs: &mut [&mut SessionKv],
        model: &OwnedLlama,
        cos_rows: &Tensor,
        sin_rows: &Tensor,
        mask: &Tensor,
        lmax: usize,
    ) -> Result<Tensor> {
        let (b, seq, _hidden) = x.dims3()?;
        debug_assert_eq!(seq, 1, "batched decode feeds exactly one token");
        let residual = x;
        let x = self.rms1.forward(x)?;
        let (q, k_new, v_new) = self.attn.project_qkv(&x)?;
        let q = rope_rows(&q, cos_rows, sin_rows)?;
        let k_new = rope_rows(&k_new, cos_rows, sin_rows)?;
        // Write each session's fresh K/V row back into its own cache.
        for (row, kv) in kvs.iter_mut().enumerate() {
            let k_row = k_new.narrow(0, row, 1)?;
            let v_row = v_new.narrow(0, row, 1)?;
            let (k, v) = match &kv.layers[layer] {
                Some((ck, cv)) => (Tensor::cat(&[ck, &k_row], 2)?, Tensor::cat(&[cv, &v_row], 2)?),
                None => (k_row, v_row),
            };
            kv.layers[layer] = Some((k, v));
        }
        // Pad every history to the longest one, drop the per-session leading
        // dim, and stack into the batch: [1, kv, lmax, hd] x B to
        // [B, kv, lmax, hd].
        let mut ks = Vec::with_capacity(b);
        let mut vs = Vec::with_capacity(b);
        for kv in kvs.iter() {
            let (k, v) = kv.layers[layer].clone().expect("just written");
            let cur = k.dim(2)?;
            let pad = lmax - cur;
            let (k, v) = if pad > 0 {
                let zk = Tensor::zeros((1, model.n_kv_heads, pad, model.head_dim), k.dtype(), k.device())?;
                let zv = Tensor::zeros((1, model.n_kv_heads, pad, model.head_dim), v.dtype(), v.device())?;
                (Tensor::cat(&[&k, &zk], 2)?, Tensor::cat(&[&v, &zv], 2)?)
            } else {
                (k, v)
            };
            ks.push(k.reshape((model.n_kv_heads, lmax, model.head_dim))?);
            vs.push(v.reshape((model.n_kv_heads, lmax, model.head_dim))?);
        }
        let k_batch = Tensor::stack(&ks, 0)?;
        let v_batch = Tensor::stack(&vs, 0)?;
        let y = self.attn.attend(&q, k_batch, v_batch, Some(mask))?;
        let x = (self.attn.combine(&y, b, seq)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}
