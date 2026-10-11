// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Gated DeltaNet recurrent kernel (transformers `torch_recurrent_gated_delta_rule`).
//!
//! Decode and short-prefill path: iterate tokens, keep a compact state
//! `S ∈ R^{B × H_v × D_k × D_v}`. Chunked prefill (FLA) lands later.
//!
//! Hot-path notes (decode / short prefill):
//! * state rows are touched via contiguous slices (no per-element get/set);
//! * decay + `kv_mem` fuse into one pass; `S += k⊗δ` fuses with the output
//!   matvec so each `S` row is written once per token;
//! * q/k/δ/`kv_mem` scratch is reused across heads and time steps.

use anyhow::{bail, Result};

/// Recurrent state for one GDN layer: `[num_v_heads][k_dim][v_dim]`.
#[derive(Clone, Debug)]
pub struct GatedDeltaState {
    pub num_v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    /// Flat row-major: head-major, then k, then v.
    pub data: Vec<f32>,
}

impl GatedDeltaState {
    pub fn zeros(num_v_heads: usize, k_dim: usize, v_dim: usize) -> Self {
        Self {
            num_v_heads,
            k_dim,
            v_dim,
            data: vec![0.0; num_v_heads * k_dim * v_dim],
        }
    }

    #[inline]
    fn head_stride(&self) -> usize {
        self.k_dim * self.v_dim
    }
}

/// Run the token-wise gated delta rule.
///
/// Shapes (row-major, matching transformers after transpose to head-major
/// inside the loop):
/// * `query` / `key`: `[seq, num_v_heads, k_dim]` (k-heads already repeated)
/// * `value`: `[seq, num_v_heads, v_dim]`
/// * `g`: `[seq, num_v_heads]` — log-space decay (≤ 0)
/// * `beta`: `[seq, num_v_heads]`
///
/// Returns `out` with the same layout as `value`, and writes the final state
/// into `state`.
pub fn recurrent_gated_delta_rule(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut GatedDeltaState,
    use_qk_l2norm: bool,
) -> Result<Vec<f32>> {
    let hv = state.num_v_heads;
    let dk = state.k_dim;
    let dv = state.v_dim;
    if hv == 0 || dk == 0 || dv == 0 {
        bail!("empty GatedDeltaState");
    }
    let q_stride = hv * dk;
    let v_stride = hv * dv;
    if query.len() % q_stride != 0
        || key.len() != query.len()
        || value.len() % v_stride != 0
        || g.len() % hv != 0
        || beta.len() != g.len()
    {
        bail!("gated-delta input length mismatch");
    }
    let seq = query.len() / q_stride;
    if value.len() / v_stride != seq || g.len() / hv != seq {
        bail!("gated-delta sequence length mismatch");
    }

    let scale = (dk as f32).sqrt().recip();
    let mut out = vec![0.0f32; seq * v_stride];
    let head_stride = state.head_stride();

    let mut q_scratch = vec![0.0f32; dk];
    let mut k_scratch = vec![0.0f32; dk];
    let mut kv_mem = vec![0.0f32; dv];
    let mut delta = vec![0.0f32; dv];

    for t in 0..seq {
        for h in 0..hv {
            let q_base = t * q_stride + h * dk;
            let v_base = t * v_stride + h * dv;
            q_scratch.copy_from_slice(&query[q_base..q_base + dk]);
            k_scratch.copy_from_slice(&key[q_base..q_base + dk]);
            if use_qk_l2norm {
                l2norm_inplace(&mut q_scratch, 1e-6);
                l2norm_inplace(&mut k_scratch, 1e-6);
            }
            for x in &mut q_scratch {
                *x *= scale;
            }

            let decay = g[t * hv + h].exp();
            let b = beta[t * hv + h];
            let s = &mut state.data[h * head_stride..(h + 1) * head_stride];

            // Fuse S *= decay with kv_mem[v] = Σ_k S[k,v] * k[k].
            kv_mem.fill(0.0);
            for ki in 0..dk {
                let kk = k_scratch[ki];
                let row = &mut s[ki * dv..(ki + 1) * dv];
                for vi in 0..dv {
                    let sv = row[vi] * decay;
                    row[vi] = sv;
                    kv_mem[vi] += sv * kk;
                }
            }

            for vi in 0..dv {
                delta[vi] = (value[v_base + vi] - kv_mem[vi]) * b;
            }

            // Fuse S += k ⊗ delta with out = Σ_k S[k,v] * q[k].
            let o_base = t * v_stride + h * dv;
            let o_row = &mut out[o_base..o_base + dv];
            o_row.fill(0.0);
            for ki in 0..dk {
                let kk = k_scratch[ki];
                let qq = q_scratch[ki];
                let row = &mut s[ki * dv..(ki + 1) * dv];
                for vi in 0..dv {
                    let sv = row[vi] + kk * delta[vi];
                    row[vi] = sv;
                    o_row[vi] += sv * qq;
                }
            }
        }
    }
    Ok(out)
}

fn l2norm_inplace(x: &mut [f32], eps: f32) {
    let mut n2 = 0.0f32;
    for v in x.iter() {
        n2 += *v * *v;
    }
    let inv = (n2 + eps).sqrt().recip();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero initial state, one step: out = (q·k) * (β v) after scaling / optional L2.
    #[test]
    fn zero_state_single_step() {
        let mut state = GatedDeltaState::zeros(1, 2, 2);
        // q=k=[1,0], v=[2,4], g=0 → decay=1, beta=1, no L2
        let query = [1.0f32, 0.0];
        let key = [1.0f32, 0.0];
        let value = [2.0f32, 4.0];
        let g = [0.0f32];
        let beta = [1.0f32];
        let out = recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, &mut state, false)
            .unwrap();
        let scale = (2.0f32).sqrt().recip(); // q scaled
        // S = k ⊗ (βv) = [[2,4],[0,0]]; out_v = S^T q_scaled = q0*row0 = scale*[2,4]
        assert!((out[0] - scale * 2.0).abs() < 1e-5);
        assert!((out[1] - scale * 4.0).abs() < 1e-5);
        // Second identical step should not explode and should keep finite state.
        let out2 = recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, &mut state, false)
            .unwrap();
        assert!(out2.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn decay_shrinks_state() {
        let mut state = GatedDeltaState::zeros(1, 1, 1);
        state.data[0] = 10.0;
        let query = [1.0f32];
        let key = [0.0f32]; // no write
        let value = [0.0f32];
        let g = [-(2.0f32).ln()]; // decay = 0.5
        let beta = [0.0f32];
        let _ = recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, &mut state, false)
            .unwrap();
        assert!((state.data[0] - 5.0).abs() < 1e-5);
    }

    /// Micro-throughput guard: Qwen3.8-sized head (dk=dv=128, hv=48), 32 decode
    /// steps must finish quickly on a laptop CPU (catches accidental O(n³) / alloc storms).
    #[test]
    fn decode_throughput_qwen38_shaped() {
        let hv = 48;
        let dk = 128;
        let dv = 128;
        let mut state = GatedDeltaState::zeros(hv, dk, dv);
        let q = vec![0.01f32; hv * dk];
        let k = vec![0.01f32; hv * dk];
        let v = vec![0.01f32; hv * dv];
        let g = vec![-0.1f32; hv];
        let beta = vec![0.5f32; hv];
        let t0 = std::time::Instant::now();
        for _ in 0..32 {
            let out = recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, &mut state, true).unwrap();
            assert_eq!(out.len(), hv * dv);
            assert!(out.iter().all(|x| x.is_finite()));
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        // Loose ceiling: even a slow debug build on CI should stay well under this.
        assert!(
            ms < 5_000.0,
            "32 decode steps of hv=48 dk=dv=128 took {ms:.1}ms (too slow)"
        );
    }
}
