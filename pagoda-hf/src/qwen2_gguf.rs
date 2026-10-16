// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! One quantized model: Qwen2 GGUF (Qwen2.5 Q4_K_M is the checkpoint this
//! path is written for).
//!
//! On CPU the weights stay in the GGUF quant. On Metal they are expanded to
//! F16 at load: candle 0.8.4's Q4 kernel overflows the SwiGLU product.
//! Activations use the dtype passed to [`Qwen2Gguf::load`] — F16 on Metal.
//! Each engine sequence owns a [`SessionKv`], so greedy decode feeds every
//! token once and a repeated prompt grafts the cached prefix. There is no
//! batched-decode kernel on this path; the engine falls back to one forward
//! per sequence.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{ops, Embedding, RmsNorm};
use candle_transformers::utils::repeat_kv;

use crate::llama::SessionKv;
use crate::CandleCausalLM;

pub struct Qwen2Gguf {
    embed: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    lm_head: QMatMul,
    cos: Tensor,
    sin: Tensor,
    max_position_embeddings: usize,
    vocab_size: usize,
    device: Device,
    dtype: DType,
}

struct Block {
    attn_norm: RmsNorm,
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    q_bias: Tensor,
    k_bias: Tensor,
    v_bias: Tensor,
    o: QMatMul,
    ffn_norm: RmsNorm,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Qwen2Gguf {
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Load a Qwen2-architecture GGUF file onto `device`.
    pub fn load(path: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let ct = Content::read(&mut file).context("parse gguf header")?;
        let arch = meta_string(&ct.metadata, "general.architecture")?;
        anyhow::ensure!(
            arch == "qwen2",
            "only Qwen2 GGUF is supported (general.architecture = {arch})"
        );

        let n_heads = meta_usize(&ct.metadata, "qwen2.attention.head_count")?;
        let n_kv_heads = meta_usize(&ct.metadata, "qwen2.attention.head_count_kv")?;
        let hidden = meta_usize(&ct.metadata, "qwen2.embedding_length")?;
        let n_layers = meta_usize(&ct.metadata, "qwen2.block_count")?;
        let context = meta_usize(&ct.metadata, "qwen2.context_length")?;
        let eps = meta_f32(&ct.metadata, "qwen2.attention.layer_norm_rms_epsilon")? as f64;
        let rope_base = ct
            .metadata
            .get("qwen2.rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1_000_000.0);
        let head_dim = hidden / n_heads;
        anyhow::ensure!(hidden % n_heads == 0, "hidden {hidden} not divisible by heads {n_heads}");
        anyhow::ensure!(n_heads % n_kv_heads == 0, "GQA heads {n_heads} / {n_kv_heads}");

        let embed_q = ct.tensor(&mut file, "token_embd.weight", device)?;
        let vocab_size = embed_q.shape().dims()[0];
        let embed = Embedding::new(dequant(embed_q, device, dtype)?, hidden);
        let norm = RmsNorm::new(
            dequant(ct.tensor(&mut file, "output_norm.weight", device)?, device, dtype)?,
            eps,
        );
        let lm_head = match ct.tensor(&mut file, "output.weight", device) {
            Ok(w) => qweight(w, device)?,
            Err(_) => qweight(ct.tensor(&mut file, "token_embd.weight", device)?, device)?,
        };

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{i}");
            let mut qtensor = |name: &str| {
                ct.tensor(&mut file, &format!("{p}.{name}"), device)
                    .with_context(|| format!("missing {p}.{name}"))
            };
            blocks.push(Block {
                attn_norm: RmsNorm::new(dequant(qtensor("attn_norm.weight")?, device, dtype)?, eps),
                q: qweight(qtensor("attn_q.weight")?, device)?,
                k: qweight(qtensor("attn_k.weight")?, device)?,
                v: qweight(qtensor("attn_v.weight")?, device)?,
                q_bias: dequant(qtensor("attn_q.bias")?, device, dtype)?,
                k_bias: dequant(qtensor("attn_k.bias")?, device, dtype)?,
                v_bias: dequant(qtensor("attn_v.bias")?, device, dtype)?,
                o: qweight(qtensor("attn_output.weight")?, device)?,
                ffn_norm: RmsNorm::new(dequant(qtensor("ffn_norm.weight")?, device, dtype)?, eps),
                gate: qweight(qtensor("ffn_gate.weight")?, device)?,
                up: qweight(qtensor("ffn_up.weight")?, device)?,
                down: qweight(qtensor("ffn_down.weight")?, device)?,
                n_heads,
                n_kv_heads,
                head_dim,
            });
        }

        let (cos, sin) = rope_tables(head_dim, rope_base, context, device, dtype)?;
        Ok(Self {
            embed,
            blocks,
            norm,
            lm_head,
            cos,
            sin,
            max_position_embeddings: context,
            vocab_size,
            device: device.clone(),
            dtype,
        })
    }

    fn forward_tokens(&self, tokens: &[u32], cache: &mut SessionKv) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "empty feed");
        let index_pos = cache.len();
        let seq = tokens.len();
        anyhow::ensure!(
            index_pos + seq <= self.max_position_embeddings,
            "context {index_pos} + {seq} exceeds context {}",
            self.max_position_embeddings
        );
        let input = Tensor::new(tokens, &self.device)?.reshape((1, seq))?;
        let mut x = self.embed.forward(&input)?;
        let causal = if seq > 1 {
            Some(causal_mask(seq, index_pos + seq, &self.device, self.dtype)?)
        } else {
            None
        };
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, layer, cache, &self.cos, &self.sin, causal.as_ref())?;
        }
        let x = self.norm.forward(&x)?;
        let x = x.i((.., seq - 1, ..))?.contiguous()?;
        let logits = qlinear(&self.lm_head, &x)?.to_dtype(DType::F32)?;
        cache.advance(seq);
        Ok(logits.flatten_all()?.to_vec1::<f32>()?)
    }
}

impl CandleCausalLM for Qwen2Gguf {
    type Cache = SessionKv;

    fn engine_name(&self) -> &'static str {
        "candle-qwen2-gguf"
    }

    fn new_cache(&self, _device: &Device) -> Result<Self::Cache> {
        Ok(SessionKv::new(self.blocks.len()))
    }

    fn cache_len(cache: &Self::Cache) -> usize {
        cache.len()
    }

    fn cache_prefix(cache: &Self::Cache, len: usize) -> Option<Self::Cache> {
        cache.prefix(len)
    }

    fn cache_export(cache: &Self::Cache, last_logits: Option<&[f32]>) -> Option<Vec<u8>> {
        cache.export_bytes(last_logits)
    }

    fn cache_import(
        bytes: &[u8],
        device: &Device,
        dtype: DType,
    ) -> Option<(Self::Cache, Option<Vec<f32>>)> {
        SessionKv::import_bytes(bytes, device, dtype).ok()
    }

    fn forward_tokens(&self, tokens: &[u32], cache: &mut Self::Cache) -> Result<Vec<f32>> {
        self.forward_tokens(tokens, cache)
    }
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        layer: usize,
        cache: &mut SessionKv,
        cos: &Tensor,
        sin: &Tensor,
        causal: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        let residual = x;
        let x = self.attn_norm.forward(x)?;
        let (q, k, v) = self.project(&x)?;
        let q = rope_at(&q, cos, sin, index_pos, seq)?;
        let k = rope_at(&k, cos, sin, index_pos, seq)?;
        let (k, v) = cache.push_kv(layer, k, v)?;
        let y = self.attend(&q, k, v, causal)?;
        let y = y.transpose(1, 2)?.reshape((b, seq, self.n_heads * self.head_dim))?;
        let x = (qlinear(&self.o, &y)? + residual)?;
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let gated = (ops::silu(&qlinear(&self.gate, &x)?)? * qlinear(&self.up, &x)?)?;
        Ok((qlinear(&self.down, &gated)? + residual)?)
    }

    fn project(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, seq, _) = x.dims3()?;
        let q = qlinear(&self.q, x)?.broadcast_add(&self.q_bias)?;
        let k = qlinear(&self.k, x)?.broadcast_add(&self.k_bias)?;
        let v = qlinear(&self.v, x)?.broadcast_add(&self.v_bias)?;
        let q = q
            .reshape((b, seq, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, seq, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        Ok((q, k, v))
    }

    fn attend(&self, q: &Tensor, k: Tensor, v: Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Scores in F32. F16 QK on Metal overflows to inf, softmax becomes
        // NaN, and the engine faults the request before any token is sampled.
        let out_dtype = q.dtype();
        let k = repeat_kv(k, self.n_heads / self.n_kv_heads)?.to_dtype(DType::F32)?;
        let v = repeat_kv(v, self.n_heads / self.n_kv_heads)?.to_dtype(DType::F32)?;
        let q = q.to_dtype(DType::F32)?;
        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let att = match mask {
            Some(mask) => att.broadcast_add(&mask.to_dtype(DType::F32)?)?,
            None => att,
        };
        let att = ops::softmax_last_dim(&att)?;
        Ok(att.matmul(&v.contiguous()?)?.to_dtype(out_dtype)?)
    }
}

/// Quantized matmul on Metal returns F32. Cast back to the activation dtype
/// before the residual add, which stays in F16.
fn qlinear(w: &QMatMul, x: &Tensor) -> Result<Tensor> {
    let y = w.forward(x)?;
    if y.dtype() == x.dtype() {
        Ok(y)
    } else {
        Ok(y.to_dtype(x.dtype())?)
    }
}

/// On Metal, candle's quantized matmul overflows the SwiGLU product (F16
/// saturates at 65504). Expand the GGUF weights to F16 once and use the
/// ordinary Metal GEMM. CPU keeps the native Q4 kernel.
fn qweight(q: candle_core::quantized::QTensor, device: &Device) -> Result<QMatMul> {
    if device.is_metal() {
        Ok(QMatMul::TensorF16(q.dequantize_f16(device)?))
    } else {
        Ok(QMatMul::from_qtensor(q)?)
    }
}

fn dequant(q: candle_core::quantized::QTensor, device: &Device, dtype: DType) -> Result<Tensor> {
    let t = q.dequantize(device)?;
    if t.dtype() == dtype {
        Ok(t)
    } else {
        Ok(t.to_dtype(dtype)?)
    }
}

fn meta_string(md: &HashMap<String, Value>, key: &str) -> Result<String> {
    match md.get(key) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) => anyhow::bail!("gguf {key} is not a string: {other:?}"),
        None => anyhow::bail!("gguf missing {key}"),
    }
}

fn meta_usize(md: &HashMap<String, Value>, key: &str) -> Result<usize> {
    let v = md.get(key).with_context(|| format!("gguf missing {key}"))?;
    if let Ok(n) = v.to_u64() {
        return Ok(n as usize);
    }
    if let Ok(n) = v.to_i64() {
        return Ok(n as usize);
    }
    if let Ok(n) = v.to_i32() {
        return Ok(n as usize);
    }
    anyhow::bail!("gguf {key} is not an integer: {v:?}")
}

fn meta_f32(md: &HashMap<String, Value>, key: &str) -> Result<f32> {
    let v = md.get(key).with_context(|| format!("gguf missing {key}"))?;
    v.to_f32().or_else(|_| v.to_f64().map(|n| n as f32)).with_context(|| format!("gguf {key} is not a float: {v:?}"))
}

fn rope_tables(
    head_dim: usize,
    freq_base: f32,
    context: usize,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx = Tensor::arange(0, context as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context, 1))?;
    let freqs = idx.matmul(&theta.reshape((1, head_dim / 2))?)?;
    Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
}

fn rope_at(x: &Tensor, cos: &Tensor, sin: &Tensor, index_pos: usize, seq: usize) -> Result<Tensor> {
    let cos = cos.narrow(0, index_pos, seq)?;
    let sin = sin.narrow(0, index_pos, seq)?;
    Ok(candle_nn::rotary_emb::rope(x, &cos, &sin)?)
}

fn causal_mask(seq: usize, kv_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let offset = kv_len - seq;
    let mut mask = vec![0f32; seq * kv_len];
    for i in 0..seq {
        for j in (i + 1 + offset)..kv_len {
            mask[i * kv_len + j] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::new(mask, device)?.reshape((seq, kv_len))?.to_dtype(dtype)?)
}
