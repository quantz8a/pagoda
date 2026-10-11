// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Depthwise causal Conv1d used by Gated DeltaNet (decode update + prefill).

use anyhow::{bail, Result};

/// Ring / left-pad state for one layer: `[conv_dim * (kernel - 1)]` layout
/// matching transformers' conv cache (channels-last flattened history).
#[derive(Clone, Debug)]
pub struct ConvState {
    pub conv_dim: usize,
    pub kernel: usize,
    /// Oldest → newest, length `conv_dim * (kernel - 1)`.
    pub data: Vec<f32>,
}

impl ConvState {
    pub fn zeros(conv_dim: usize, kernel: usize) -> Self {
        let hist = kernel.saturating_sub(1);
        Self {
            conv_dim,
            kernel,
            data: vec![0.0; conv_dim * hist],
        }
    }
}

/// Single-token depthwise causal conv update (decode).
///
/// `weight` is `[conv_dim * kernel]` with layout `channel-major, tap 0..K-1`
/// (tap 0 = oldest / left pad). `x` is `[conv_dim]` for the new token.
/// Applies SiLU on the convolution output.
///
/// State is rolled in place (`copy_within`) — no per-token heap alloc.
pub fn causal_conv1d_update(
    x: &[f32],
    state: &mut ConvState,
    weight: &[f32],
) -> Result<Vec<f32>> {
    let c = state.conv_dim;
    let k = state.kernel;
    if x.len() != c {
        bail!("conv update: x len {} != conv_dim {}", x.len(), c);
    }
    if weight.len() != c * k {
        bail!("conv update: weight len {} != {}", weight.len(), c * k);
    }
    let hist = k - 1;
    let mut out = vec![0.0f32; c];
    for ch in 0..c {
        let mut acc = 0.0f32;
        // history taps
        for t in 0..hist {
            let w = weight[ch * k + t];
            acc += state.data[t * c + ch] * w;
        }
        // newest tap
        acc += x[ch] * weight[ch * k + hist];
        out[ch] = silu(acc);
    }
    // roll state in place: drop oldest frame, append x
    if hist > 0 {
        if hist > 1 {
            state.data.copy_within(c..(hist * c), 0);
        }
        state.data[(hist - 1) * c..].copy_from_slice(x);
    }
    Ok(out)
}

/// Prefill: causal depthwise conv over a sequence `[seq * conv_dim]` (time-major).
///
/// Also leaves `state` (when provided via the update path) unused — this cold
/// path rebuilds from zeros so the returned sequence matches a fresh decode
/// chain. Prefer calling [`causal_conv1d_update`] in a loop when you need the
/// final ring state for subsequent decode.
pub fn causal_conv1d_fn(x: &[f32], weight: &[f32], conv_dim: usize, kernel: usize) -> Result<Vec<f32>> {
    if x.len() % conv_dim != 0 {
        bail!("conv fn: x not multiple of conv_dim");
    }
    if weight.len() != conv_dim * kernel {
        bail!("conv fn: bad weight len");
    }
    let seq = x.len() / conv_dim;
    let mut state = ConvState::zeros(conv_dim, kernel);
    let mut out = Vec::with_capacity(x.len());
    for t in 0..seq {
        let xt = &x[t * conv_dim..(t + 1) * conv_dim];
        let yt = causal_conv1d_update(xt, &mut state, weight)?;
        out.extend_from_slice(&yt);
    }
    Ok(out)
}

/// Prefill that also returns the final conv ring (for decode continuity).
pub fn causal_conv1d_fn_with_state(
    x: &[f32],
    weight: &[f32],
    conv_dim: usize,
    kernel: usize,
) -> Result<(Vec<f32>, ConvState)> {
    if x.len() % conv_dim != 0 {
        bail!("conv fn: x not multiple of conv_dim");
    }
    if weight.len() != conv_dim * kernel {
        bail!("conv fn: bad weight len");
    }
    let seq = x.len() / conv_dim;
    let mut state = ConvState::zeros(conv_dim, kernel);
    let mut out = Vec::with_capacity(x.len());
    for t in 0..seq {
        let xt = &x[t * conv_dim..(t + 1) * conv_dim];
        let yt = causal_conv1d_update(xt, &mut state, weight)?;
        out.extend_from_slice(&yt);
    }
    Ok((out, state))
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel1_is_pointwise_silu() {
        let mut st = ConvState::zeros(2, 1);
        let w = [1.0f32, 2.0]; // per-channel scale
        let y = causal_conv1d_update(&[1.0, 0.5], &mut st, &w).unwrap();
        assert!((y[0] - silu(1.0)).abs() < 1e-5);
        assert!((y[1] - silu(1.0)).abs() < 1e-5);
    }

    #[test]
    fn in_place_roll_matches_allocating_chain() {
        let c = 4;
        let k = 4;
        let w: Vec<f32> = (0..c * k).map(|i| (i as f32) * 0.01).collect();
        let mut st = ConvState::zeros(c, k);
        let xs = [
            [0.1f32, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
        ];
        let mut outs = Vec::new();
        for x in &xs {
            outs.push(causal_conv1d_update(x, &mut st, &w).unwrap());
        }
        // Prefill path over the same tokens must match token-wise updates.
        let flat: Vec<f32> = xs.iter().flat_map(|r| r.iter().copied()).collect();
        let y = causal_conv1d_fn(&flat, &w, c, k).unwrap();
        for (t, o) in outs.iter().enumerate() {
            for ch in 0..c {
                assert!(
                    (y[t * c + ch] - o[ch]).abs() < 1e-5,
                    "t={t} ch={ch}: {} vs {}",
                    y[t * c + ch],
                    o[ch]
                );
            }
        }
        assert_eq!(st.data.len(), c * (k - 1));
    }
}
