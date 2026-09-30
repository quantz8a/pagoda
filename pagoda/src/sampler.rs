// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Logit post-processing and categorical sampling.

use crate::rng::Rng;
use crate::spec::SamplingParams;

const EPS: f32 = 1e-9;

/// Stateful sampler (holds a reproducible RNG stream).
pub struct Sampler {
    rng: Rng,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Sampler { rng: Rng::new(seed) }
    }

    /// Reset the underlying RNG stream for reproducible per-request sampling.
    pub fn seed(&mut self, seed: u64) {
        self.rng = Rng::new(seed);
    }

    /// Sample a single token id from raw logits. A temperature below `1e-5`
    /// selects greedy decoding (argmax), otherwise top-k and top-p are applied
    /// before categorical sampling.
    pub fn sample(&mut self, logits: &[f32], params: &SamplingParams) -> u32 {
        if params.temperature < 1e-5 {
            return argmax(logits);
        }
        let mut probs = logits.to_vec();
        temperature_softmax(&mut probs, params.temperature);
        apply_top_k(&mut probs, params.top_k);
        apply_top_p(&mut probs, params.top_p);
        self.sample_categorical(&probs)
    }

    /// Sample from an already-normalized probability vector.
    pub fn sample_categorical(&mut self, probs: &[f32]) -> u32 {
        debug_assert!(!probs.is_empty());
        let total: f32 = probs.iter().sum();
        if total <= EPS {
            return argmax(probs);
        }
        let u = self.rng.next_f32() * total;
        let mut acc = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if u < acc {
                return i as u32;
            }
        }
        (probs.len() - 1) as u32
    }
}

/// Index of the largest element.
pub fn argmax(xs: &[f32]) -> u32 {
    debug_assert!(!xs.is_empty());
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x.is_finite() && x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

/// Numerically stable log-softmax, useful for scoring candidate continuations.
pub fn log_softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0.0f32; xs.len()];
    let mut sum = 0.0f32;
    for (i, &x) in xs.iter().enumerate() {
        let e = (x - max).exp();
        out[i] = e;
        sum += e;
    }
    let log_sum = sum.ln().max(f32::MIN_POSITIVE);
    for v in out.iter_mut() {
        *v = v.ln() - log_sum;
    }
    out
}

/// Softmax over finite logits only: masked (`-inf`) / NaN entries get exactly
/// zero probability mass instead of being reset to a 0.0 logit (which would
/// *raise* their weight above any negative-scored legal token).
fn temperature_softmax(xs: &mut [f32], temp: f32) {
    let t = temp.max(EPS);
    let max = xs
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in xs.iter_mut() {
        if v.is_finite() {
            let e = ((*v - max) / t).exp();
            *v = e;
            sum += e;
        } else {
            *v = 0.0;
        }
    }
    if sum > EPS {
        for v in xs.iter_mut() {
            *v /= sum;
        }
    }
}

fn apply_top_k(xs: &mut [f32], k: usize) {
    if k == 0 || k >= xs.len() {
        return;
    }
    let mut sorted: Vec<f32> = xs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k.saturating_sub(1)];
    for v in xs.iter_mut() {
        if *v < threshold {
            *v = 0.0;
        }
    }
}

fn apply_top_p(xs: &mut [f32], p: f32) {
    if p >= 1.0 - EPS {
        return;
    }
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| xs[b].partial_cmp(&xs[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32;
    let mut keep = vec![false; xs.len()];
    for &i in &idx {
        cum += xs[i];
        keep[i] = true;
        if cum >= p {
            break;
        }
    }
    for (i, v) in xs.iter_mut().enumerate() {
        if !keep[i] {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_is_argmax() {
        let logits = vec![0.1, 5.0, 2.0];
        let params = SamplingParams::default();
        let mut s = Sampler::new(0);
        assert_eq!(s.sample(&logits, &params), 1);
    }

    #[test]
    fn sampler_is_reproducible() {
        let logits = vec![1.0f32; 100];
        let params = SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        };
        let mut a = Sampler::new(99);
        let mut b = Sampler::new(99);
        let seq_a: Vec<u32> = (0..20).map(|_| a.sample(&logits, &params)).collect();
        let seq_b: Vec<u32> = (0..20).map(|_| b.sample(&logits, &params)).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn top_k_mass_is_some() {
        let logits = vec![0.0, 10.0, 9.0, 8.0, 1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 3,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(1);
        let tok = s.sample(&logits, &params);
        assert!(tok == 1 || tok == 2 || tok == 3, "got {}", tok);
    }

    #[test]
    fn top_p_keeps_only_the_nucleus() {
        // softmax(1.0, 0.0, -1.0) ~= (0.665, 0.245, 0.090); top_p = 0.75 keeps
        // tokens {0, 1} and must never sample token 2.
        let logits = vec![1.0f32, 0.0, -1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.75,
            top_k: usize::MAX,
            ..SamplingParams::default()
        };
        let mut s = Sampler::new(123);
        for _ in 0..200 {
            let t = s.sample(&logits, &params);
            assert!(t == 0 || t == 1, "sampled outside the nucleus: {t}");
        }
    }

    #[test]
    fn temperature_path_gives_masked_logits_zero_mass() {
        // Tokens 2 and 3 are masked to -inf (e.g. by a grammar). With
        // temperature sampling they must get exactly zero probability —
        // resetting a masked logit to 0.0 would *raise* its weight above the
        // legal token 1 scored at -5.
        let logits = vec![0.0f32, -5.0, f32::NEG_INFINITY, f32::NEG_INFINITY];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: usize::MAX,
            ..SamplingParams::default()
        };
        for seed in 0..16 {
            let mut s = Sampler::new(seed);
            for _ in 0..50 {
                let t = s.sample(&logits, &params);
                assert!(t == 0 || t == 1, "masked token sampled: {t}");
            }
        }
    }

    #[test]
    fn log_softmax_sums_to_one() {
        let lp = log_softmax(&[0.0, 1.0, 2.0]);
        let sum: f32 = lp.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
