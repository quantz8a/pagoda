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
//! Qwen2-family checkpoints (Qwen2.5 students included) load through the same
//! path: q/k/v bias is probed per checkpoint, tied embeddings and GQA are
//! config-driven. `examples/e2e_qwen_student.rs` verifies a real distilled
//! Qwen2.5 student end to end.
//!
//! Two entry points:
//!
//! * [`OwnedLlama::forward_tokens`] — single-sequence prefill / suffix feed
//!   with a square causal mask (any length, any position).
//! * [`OwnedLlama::batch_decode`] — the continuous-batching step: many
//!   sessions, one fresh token each, one forward pass. Attention runs per
//!   session over exactly its own KV history (variable-length, no padding
//!   waste); the projection and FFN matmuls are shared across the batch.

use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Module, Tensor};
use std::sync::atomic::{AtomicU32, Ordering};
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

    /// Estimated resident bytes of the cached K/V tensors (for the vault's
    /// memory budget). Arc-shared storage is counted once per snapshot.
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|(k, v)| (k.elem_count() + v.elem_count()) * k.dtype().size_in_bytes())
            .sum()
    }
}

impl crate::vault::VaultEntry for SessionKv {
    fn bytes(&self) -> usize {
        self.bytes()
    }
}

/// Wire magic for [`SessionKv::export_bytes`] payloads ("PDKV").
const KV_WIRE_MAGIC: u32 = 0x5044_4B56;
/// v1: layer payload always F32, the dtype field informational only.
/// v2: the dtype field is the wire encoding (0 = f32-le, 1 = f16-le).
const KV_WIRE_VERSION: u32 = 1;
const KV_WIRE_VERSION_V2: u32 = 2;

/// Wire encoding for exported KV (0 = f32, 1 = f16), process-wide. F16
/// halves PD transfer volume at the cost of a lossy roundtrip; the payload
/// is self-describing, so only the prefill side sets this.
pub static KV_WIRE_DTYPE: AtomicU32 = AtomicU32::new(0);

/// Select the KV wire encoding for [`SessionKv::export_bytes`].
pub fn set_kv_wire_f16(on: bool) {
    KV_WIRE_DTYPE.store(if on { 1 } else { 0 }, Ordering::Relaxed);
}

/// f32 -> IEEE-754 half-precision bits (round-to-nearest-even).
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00; // overflow -> inf
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign; // underflow -> zero
        }
        let mant = mant | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = mant >> shift;
        let rem = mant & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        if rem > halfway || (rem == halfway && half_mant & 1 == 1) {
            half_mant += 1;
        }
        return sign | half_mant as u16;
    }
    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) {
        half = half.wrapping_add(1);
    }
    half
}

/// IEEE-754 half-precision bits -> f32 (exact).
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: normalize into the f32 exponent range.
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13) // inf / nan
    } else {
        sign | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}



fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32> {
    let end = *pos + 4;
    anyhow::ensure!(end <= buf.len(), "kv payload truncated");
    let v = u32::from_le_bytes(buf[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(v)
}

impl SessionKv {
    /// Serialize the cache plus the session's last-position logits into a
    /// flat little-endian payload for PD transfer (prefill worker side).
    ///
    /// Tensors are moved to the CPU and widened to F32 on the wire: every
    /// F16/BF16 value is exactly representable in F32, so the round trip is
    /// bit-identical when the decode side casts back to its compute dtype.
    /// Format: magic, version, token count, layer count, logits (len + f32s),
    /// then per layer a present flag followed by (kv_heads, head_dim,
    /// source-dtype id) and the K then V values, F32 LE.
    pub fn export_bytes(&self, last_logits: Option<&[f32]>) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        push_u32(&mut out, KV_WIRE_MAGIC);
        push_u32(&mut out, KV_WIRE_VERSION_V2);
        push_u32(&mut out, self.len as u32);
        push_u32(&mut out, self.layers.len() as u32);
        let logits = last_logits.unwrap_or(&[]);
        push_u32(&mut out, logits.len() as u32);
        for x in logits {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for layer in &self.layers {
            let Some((k, v)) = layer else {
                push_u32(&mut out, 0);
                continue;
            };
            push_u32(&mut out, 1);
            let dims = k.dims();
            if dims.len() != 4 || v.dims() != dims {
                return None;
            }
            push_u32(&mut out, dims[1] as u32); // kv heads
            push_u32(&mut out, dims[3] as u32); // head dim
            let wire_dtype = KV_WIRE_DTYPE.load(Ordering::Relaxed);
            push_u32(&mut out, wire_dtype);
            for t in [k, v] {
                let flat = t
                    .to_device(&Device::Cpu)
                    .ok()?
                    .flatten_all()
                    .ok()?
                    .to_dtype(DType::F32)
                    .ok()?
                    .to_vec1::<f32>()
                    .ok()?;
                if wire_dtype == 1 {
                    for x in flat {
                        out.extend_from_slice(&f32_to_f16_bits(x).to_le_bytes());
                    }
                } else {
                    for x in flat {
                        out.extend_from_slice(&x.to_le_bytes());
                    }
                }
            }
        }
        Some(out)
    }

    /// Rebuild a cache (and the stashed last-position logits) from
    /// [`SessionKv::export_bytes`] payload, on `device` in the decode worker's
    /// compute `dtype`.
    pub fn import_bytes(
        bytes: &[u8],
        device: &Device,
        dtype: DType,
    ) -> Result<(SessionKv, Option<Vec<f32>>)> {
        let mut pos = 0usize;
        anyhow::ensure!(
            read_u32(bytes, &mut pos)? == KV_WIRE_MAGIC,
            "bad kv payload magic"
        );
        let version = read_u32(bytes, &mut pos)?;
        anyhow::ensure!(
            version == KV_WIRE_VERSION || version == KV_WIRE_VERSION_V2,
            "unsupported kv payload version {version}"
        );
        let len = read_u32(bytes, &mut pos)? as usize;
        let num_layers = read_u32(bytes, &mut pos)? as usize;
        let logits_len = read_u32(bytes, &mut pos)? as usize;
        let mut logits = Vec::with_capacity(logits_len);
        for _ in 0..logits_len {
            anyhow::ensure!(pos + 4 <= bytes.len(), "kv payload truncated");
            logits.push(f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let present = read_u32(bytes, &mut pos)?;
            if present == 0 {
                layers.push(None);
                continue;
            }
            let kv_heads = read_u32(bytes, &mut pos)? as usize;
            let head_dim = read_u32(bytes, &mut pos)? as usize;
            let wire_dtype = read_u32(bytes, &mut pos)?;
            let count = kv_heads * len * head_dim;
            let f16_wire = version >= KV_WIRE_VERSION_V2 && wire_dtype == 1;
            let mut tensors = Vec::with_capacity(2);
            for _ in 0..2 {
                let byte_len = count * if f16_wire { 2 } else { 4 };
                anyhow::ensure!(pos + byte_len <= bytes.len(), "kv payload truncated");
                let mut vals = Vec::with_capacity(count);
                if f16_wire {
                    for chunk in bytes[pos..pos + byte_len].chunks_exact(2) {
                        vals.push(f16_bits_to_f32(u16::from_le_bytes(
                            chunk.try_into().unwrap(),
                        )));
                    }
                } else {
                    for chunk in bytes[pos..pos + byte_len].chunks_exact(4) {
                        vals.push(f32::from_le_bytes(chunk.try_into().unwrap()));
                    }
                }
                pos += byte_len;
                let t = Tensor::from_vec(vals, (1, kv_heads, len, head_dim), &Device::Cpu)?
                    .to_dtype(dtype)?
                    .to_device(device)?;
                tensors.push(t);
            }
            let mut it = tensors.into_iter();
            layers.push(Some((
                it.next().expect("k tensor"),
                it.next().expect("v tensor"),
            )));
        }
        let logits = if logits.is_empty() { None } else { Some(logits) };
        Ok((SessionKv { layers, len }, logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes tests that touch the process-wide KV wire dtype flag.
    static WIRE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn f16_bits_exact_and_tolerant() {
        for x in [0.0f32, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0] {
            assert_eq!(f16_bits_to_f32(f32_to_f16_bits(x)), x, "exact {x}");
        }
        for x in [0.1f32, 0.3333, 100.25, -1234.5] {
            let back = f16_bits_to_f32(f32_to_f16_bits(x));
            let tol = x.abs() * 0.001 + 1e-6;
            assert!((back - x).abs() <= tol, "{x} -> {back}");
        }
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(f32::INFINITY)), f32::INFINITY);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn session_kv_bytes_roundtrip_f16_wire() {
        let _guard = WIRE_LOCK.lock().unwrap();
        let dev = Device::Cpu;
        let k0 = Tensor::rand(0f32, 1f32, (1, 2, 5, 4), &dev).unwrap();
        let v0 = Tensor::rand(0f32, 1f32, (1, 2, 5, 4), &dev).unwrap();
        let kv = SessionKv {
            layers: vec![Some((k0.clone(), v0.clone()))],
            len: 5,
        };
        let logits = vec![0.25f32, -1.5, 3.75];

        set_kv_wire_f16(true);
        let bytes = kv.export_bytes(Some(&logits)).expect("export f16");
        set_kv_wire_f16(false);
        let bytes32 = kv.export_bytes(Some(&logits)).expect("export f32");
        // Tensor bytes halve on the F16 wire (2 heads x 5 tokens x 4 dim x K+V).
        assert_eq!(bytes.len(), bytes32.len() - 2 * 5 * 4 * 2 * 2);

        let (back, back_logits) = SessionKv::import_bytes(&bytes, &dev, DType::F32).unwrap();
        assert_eq!(back.len(), 5);
        // Logits stay F32-exact even on the F16 wire.
        assert_eq!(back_logits.as_deref(), Some(logits.as_slice()));
        let (k0b, _) = back.layers[0].as_ref().unwrap();
        let orig = k0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let got = k0b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in orig.iter().zip(&got) {
            assert!((a - b).abs() <= 0.001, "{a} vs {b}");
        }
    }

    #[test]
    fn session_kv_bytes_roundtrip() {
        let _guard = WIRE_LOCK.lock().unwrap();
        let dev = Device::Cpu;
        let k0 = Tensor::rand(0f32, 1f32, (1, 2, 5, 4), &dev).unwrap();
        let v0 = Tensor::rand(0f32, 1f32, (1, 2, 5, 4), &dev).unwrap();
        let k1 = Tensor::rand(0f32, 1f32, (1, 2, 5, 4), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v1 = Tensor::rand(0f32, 1f32, (1, 2, 5, 4), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let kv = SessionKv {
            layers: vec![Some((k0.clone(), v0.clone())), None, Some((k1.clone(), v1))],
            len: 5,
        };
        let logits = vec![0.25f32, -1.5, 3.75];
        let bytes = kv.export_bytes(Some(&logits)).expect("export");

        // F32 target: bit-identical values.
        let (back, back_logits) = SessionKv::import_bytes(&bytes, &dev, DType::F32).unwrap();
        assert_eq!(back.len(), 5);
        assert_eq!(back.layers.len(), 3);
        assert!(back.layers[1].is_none());
        assert_eq!(back_logits.as_deref(), Some(logits.as_slice()));
        let (k0b, v0b) = back.layers[0].as_ref().unwrap();
        assert_eq!(
            k0.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            k0b.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            v0.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            v0b.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // F16 source round-trips exactly through the F32 wire into F16.
        let (back16, _) = SessionKv::import_bytes(&bytes, &dev, DType::F16).unwrap();
        let (k1b, _) = back16.layers[2].as_ref().unwrap();
        assert_eq!(k1b.dtype(), DType::F16);
        assert_eq!(
            k1.flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            k1b.flatten_all()
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );

        // Corrupt payloads fail cleanly.
        assert!(SessionKv::import_bytes(&bytes[..12], &dev, DType::F32).is_err());
        assert!(SessionKv::import_bytes(b"garbage", &dev, DType::F32).is_err());
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

/// Linear with an optional bias, probed from the checkpoint: Qwen2-family
/// weights carry q/k/v bias, plain Llama does not — detect, don't configure.
fn linear_maybe_bias(in_size: usize, out_size: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_size, in_size), "weight")?;
    let bias = vb.get(out_size, "bias").ok();
    Ok(Linear::new(weight, bias))
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
            q_proj: linear_maybe_bias(size_in, head_dim * config.num_attention_heads, vb.pp("q_proj"))?,
            k_proj: linear_maybe_bias(size_in, head_dim * config.num_key_value_heads, vb.pp("k_proj"))?,
            v_proj: linear_maybe_bias(size_in, head_dim * config.num_key_value_heads, vb.pp("v_proj"))?,
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
    /// Per-session KV histories differ in length; attention runs per session
    /// over exactly its own history (variable-length, no padding, no mask —
    /// the query row is the newest position, so the whole history is
    /// visible) while the projection and FFN matmuls stay batched. A
    /// paged-attention kernel that reads the engine's KV blocks directly is
    /// the roadmap successor.
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
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward_batch(&x, layer, &mut work_refs, &cos_rows, &sin_rows)?;
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
    /// write-back, then variable-length per-session attention.
    fn forward_batch(
        &self,
        x: &Tensor,
        layer: usize,
        kvs: &mut [&mut SessionKv],
        cos_rows: &Tensor,
        sin_rows: &Tensor,
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
        // Variable-length attention: each session attends over exactly its
        // own history (no padding, no mask — the query row is the newest
        // position, so the whole history is visible). Same math as the
        // padded-then-masked batch this replaces, minus the pad waste;
        // projections stay batched. Rows concat back into [B, h, 1, hd].
        let mut rows = Vec::with_capacity(b);
        for (row, kv) in kvs.iter().enumerate() {
            let (k, v) = kv.layers[layer].clone().expect("just written");
            let q_row = q.narrow(0, row, 1)?;
            rows.push(self.attn.attend(&q_row, k, v, None)?);
        }
        let y = Tensor::cat(&rows, 0)?;
        let x = (self.attn.combine(&y, b, seq)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}
