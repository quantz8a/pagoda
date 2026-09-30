// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: AGPL-3.0-only
//! Deterministic, dependency-free property-style tests. Each test generates a
//! pseudorandom workload from the crate's own SplitMix64 RNG and checks a
//! structural invariant across many iterations, in the spirit of `proptest`
//! without an external dependency.

use pagoda::rng::Rng;
use pagoda::{EngineConfig, FinishReason, RadixCache, SamplingParams, ToyEngine, WriteRequest};

/// Token sequences over a small alphabet with heavy sharing of prefixes,
/// mirroring real prompt structure.
fn random_sequences(seed: u64, count: usize) -> Vec<Vec<u32>> {
    let mut rng = Rng::new(seed);
    let mut seqs = Vec::with_capacity(count);
    for _ in 0..count {
        let len = 1 + (rng.next_u64() % 12) as usize;
        let mut s = Vec::with_capacity(len);
        for _ in 0..len {
            s.push((rng.next_u64() % 4) as u32); // alphabet {0,1,2,3}
        }
        seqs.push(s);
    }
    seqs
}

#[test]
fn radix_trie_is_reachable_and_monotonic() {
    for seed in 0..20u64 {
        let seqs = random_sequences(seed, 40);
        let mut rc = RadixCache::new();
        for s in &seqs {
            rc.insert(s);
        }

        // Every inserted sequence must be fully reachable from the root.
        for s in &seqs {
            let (matched, _) = rc.match_prefix(s);
            assert_eq!(matched, s.len(), "seq {:?} not reachable under seed {}", s, seed);
        }

        // Re-inserting an existing sequence must not create nodes; insertion is
        // monotonic with respect to node count.
        let before = rc.num_nodes();
        for s in &seqs {
            assert_eq!(rc.insert(s), 0, "re-insert created nodes");
        }
        assert_eq!(rc.num_nodes(), before);
    }
}

#[test]
fn scheduler_never_exceeds_max_tokens_and_is_deterministic() {
    let words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "SGLang", "prefix", "cache",
    ];
    let wlen = words.len() as u64;

    for seed in 0..20u64 {
        let mut rng = Rng::new(seed);
        let n = 1 + (rng.next_u64() % 5) as usize;
        let mut reqs = Vec::with_capacity(n);
        let mut caps = Vec::with_capacity(n);

        for i in 0..n {
            let k = 1 + (rng.next_u64() % wlen) as usize;
            let mut prompt = String::new();
            for j in 0..k {
                if j > 0 { prompt.push(' '); }
                prompt.push_str(words[(rng.next_u64() % wlen) as usize]);
            }
            // A unique numeric tag keeps every prompt disjoint, so each output
            // can be matched back to its request regardless of completion order.
            prompt.push_str(&format!(" #{i}"));

            let max_tokens = 1 + (rng.next_u64() % 8) as usize;
            caps.push(max_tokens);
            reqs.push(WriteRequest::new(
                prompt,
                SamplingParams { max_tokens, ..SamplingParams::default() },
            ));
        }

        let cfg = || EngineConfig {
            num_kv_blocks: 1024,
            block_size: 8,
            max_running_requests: 32,
            max_prefill_tokens_per_step: 4,
            seed,
            ..EngineConfig::default()
        };
        let mut a = ToyEngine::toy(cfg());
        let mut b = ToyEngine::toy(cfg());

        let outs_a = a.generate_batch(&reqs);
        let outs_b = b.generate_batch(&reqs);

        assert_eq!(outs_a.len(), reqs.len());

        // Completion order may differ from submission order; match each output
        // back to its prompt through the decoded full text.
        for out in &outs_a {
            let cap = reqs
                .iter()
                .zip(&caps)
                .find_map(|(r, c)| {
                    out.full_text.as_ref().and_then(|f| f.starts_with(&r.text).then(|| *c))
                })
                .expect("each output must match a submitted prompt");
            assert_eq!(out.finish_reason, FinishReason::Length);
            assert_eq!(out.output_token_ids.len(), cap, "generated {} tokens but cap is {}", out.output_token_ids.len(), cap);
        }

        // Same seed + same requests => identical generation (order included).
        assert_eq!(outs_a.len(), outs_b.len());
        for (x, y) in outs_a.iter().zip(&outs_b) {
            assert_eq!(x.output_token_ids, y.output_token_ids);
            assert_eq!(x.finish_reason, y.finish_reason);
        }
    }
}

