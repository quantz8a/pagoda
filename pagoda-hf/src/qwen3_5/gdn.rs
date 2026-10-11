// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Gated DeltaNet layer (projections + conv + recurrent delta rule).
//!
//! Weight layout mirrors HF / MLX text tower after dequant:
//! `*.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,conv1d,A_log,dt_bias,norm}`.

use anyhow::{bail, Result};

use super::causal_conv::{causal_conv1d_fn_with_state, causal_conv1d_update, ConvState};
use super::config::Qwen3_5TextConfig;
use super::gated_delta::{recurrent_gated_delta_rule, GatedDeltaState};

/// Per-layer GDN cache (conv ring + recurrent S).
#[derive(Clone, Debug)]
pub struct GdnCache {
    pub conv: ConvState,
    pub recurrent: GatedDeltaState,
}

impl GdnCache {
    pub fn new(cfg: &Qwen3_5TextConfig) -> Self {
        let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
        let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        Self {
            conv: ConvState::zeros(conv_dim, cfg.linear_conv_kernel_dim),
            recurrent: GatedDeltaState::zeros(
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
            ),
        }
    }
}

/// Dense float weights for one GDN layer (post-dequant).
pub struct GdnWeights {
    pub in_proj_qkv: Dense, // [hidden, key*2+value]
    pub in_proj_z: Dense,
    pub in_proj_a: Dense, // [hidden, num_v_heads]
    pub in_proj_b: Dense,
    pub out_proj: Dense, // [value_dim, hidden]
    /// Depthwise conv weight `[conv_dim * kernel]`.
    pub conv1d: Vec<f32>,
    pub a_log: Vec<f32>,  // [num_v_heads]
    pub dt_bias: Vec<f32>, // [num_v_heads]
    pub norm_weight: Vec<f32>, // [head_v_dim]
    pub rms_eps: f32,
    pub output_gate_swish: bool,
}

pub struct Dense {
    pub rows: usize, // out
    pub cols: usize, // in
    pub weight: Vec<f32>, // row-major [out, in]
}

impl Dense {
    pub fn gemm(&self, x: &[f32], seq: usize) -> Result<Vec<f32>> {
        if x.len() != seq * self.cols {
            bail!("gemm: x len {} != seq*cols {}", x.len(), seq * self.cols);
        }
        let mut y = vec![0.0f32; seq * self.rows];
        let cols = self.cols;
        let rows = self.rows;
        // Decode (seq=1) and short prefill: keep the inner product tight;
        // weight is row-major [out, in].
        for t in 0..seq {
            let xbase = t * cols;
            let ybase = t * rows;
            let xin = &x[xbase..xbase + cols];
            for o in 0..rows {
                let wrow = &self.weight[o * cols..(o + 1) * cols];
                let mut acc = 0.0f32;
                // Manual unroll-friendly accumulate.
                let n = cols;
                let mut i = 0;
                while i + 4 <= n {
                    acc += wrow[i] * xin[i]
                        + wrow[i + 1] * xin[i + 1]
                        + wrow[i + 2] * xin[i + 2]
                        + wrow[i + 3] * xin[i + 3];
                    i += 4;
                }
                while i < n {
                    acc += wrow[i] * xin[i];
                    i += 1;
                }
                y[ybase + o] = acc;
            }
        }
        Ok(y)
    }
}

/// Run GDN over a sequence (CPU f32 reference path).
pub fn gdn_forward(
    hidden: &[f32],
    seq: usize,
    cfg: &Qwen3_5TextConfig,
    w: &GdnWeights,
    cache: &mut GdnCache,
) -> Result<Vec<f32>> {
    let hidden_size = cfg.hidden_size;
    if hidden.len() != seq * hidden_size {
        bail!("gdn: hidden len mismatch");
    }
    let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
    let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
    let dk = cfg.linear_key_head_dim;
    let dv = cfg.linear_value_head_dim;
    let nk = cfg.linear_num_key_heads;
    let nv = cfg.linear_num_value_heads;
    let rep = nv / nk;

    let mixed = w.in_proj_qkv.gemm(hidden, seq)?;
    let z = w.in_proj_z.gemm(hidden, seq)?;
    let a = w.in_proj_a.gemm(hidden, seq)?;
    let b = w.in_proj_b.gemm(hidden, seq)?;

    // conv expects time-major `[seq * conv_dim]` — same as mixed layout
    let conv_dim = key_dim * 2 + value_dim;
    let mixed_conv = if seq == 1 {
        causal_conv1d_update(&mixed, &mut cache.conv, &w.conv1d)?
    } else {
        // Prefill rebuilds conv state from the sequence and keeps the ring
        // for subsequent decode steps (avoids a second pass).
        let (y, st) =
            causal_conv1d_fn_with_state(&mixed, &w.conv1d, conv_dim, cfg.linear_conv_kernel_dim)?;
        cache.conv = st;
        y
    };

    // split q,k,v
    let mut q = vec![0.0f32; seq * nk * dk];
    let mut k = vec![0.0f32; seq * nk * dk];
    let mut v = vec![0.0f32; seq * nv * dv];
    for t in 0..seq {
        let base = t * conv_dim;
        q[t * nk * dk..(t + 1) * nk * dk]
            .copy_from_slice(&mixed_conv[base..base + key_dim]);
        k[t * nk * dk..(t + 1) * nk * dk]
            .copy_from_slice(&mixed_conv[base + key_dim..base + 2 * key_dim]);
        v[t * nv * dv..(t + 1) * nv * dv]
            .copy_from_slice(&mixed_conv[base + 2 * key_dim..base + conv_dim]);
    }

    // repeat k/q heads to nv
    let mut q_rep = vec![0.0f32; seq * nv * dk];
    let mut k_rep = vec![0.0f32; seq * nv * dk];
    for t in 0..seq {
        for hk in 0..nk {
            for r in 0..rep {
                let h = hk * rep + r;
                let src = t * nk * dk + hk * dk;
                let dst = t * nv * dk + h * dk;
                q_rep[dst..dst + dk].copy_from_slice(&q[src..src + dk]);
                k_rep[dst..dst + dk].copy_from_slice(&k[src..src + dk]);
            }
        }
    }

    let mut beta = vec![0.0f32; seq * nv];
    let mut g = vec![0.0f32; seq * nv];
    for t in 0..seq {
        for h in 0..nv {
            beta[t * nv + h] = sigmoid(b[t * nv + h]);
            let soft = softplus(a[t * nv + h] + w.dt_bias[h]);
            g[t * nv + h] = -w.a_log[h].exp() * soft;
        }
    }

    let core = recurrent_gated_delta_rule(
        &q_rep,
        &k_rep,
        &v,
        &g,
        &beta,
        &mut cache.recurrent,
        true,
    )?;

    // RMSNormGated: norm(core) * act(z)
    let mut gated = vec![0.0f32; seq * value_dim];
    for t in 0..seq {
        for h in 0..nv {
            let cbase = t * value_dim + h * dv;
            let mut buf = core[cbase..cbase + dv].to_vec();
            rms_norm_inplace(&mut buf, &w.norm_weight, w.rms_eps);
            for i in 0..dv {
                let gate = if w.output_gate_swish {
                    silu(z[cbase + i])
                } else {
                    sigmoid(z[cbase + i])
                };
                gated[cbase + i] = buf[i] * gate;
            }
        }
    }

    w.out_proj.gemm(&gated, seq)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn softplus(x: f32) -> f32 {
    // stable softplus
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let mut n2 = 0.0f32;
    for v in x.iter() {
        n2 += *v * *v;
    }
    let inv = (n2 / x.len() as f32 + eps).sqrt().recip();
    for (i, v) in x.iter_mut().enumerate() {
        *v *= inv * weight[i];
    }
}
