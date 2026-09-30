// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Block-level chained-hash prefix cache, in the style of vLLM's Automatic
//! Prefix Caching (APC).
//!
//! Unlike the radix trie (which walks token by token), APC hashes each *full*
//! block of tokens chained on the previous block's hash, so lookup is one hash
//! probe per block and eviction works at block granularity. The chained hash
//! preserves prefix semantics: a cached block is only reusable when the whole
//! preceding context matches, which is exactly what KV values require.
//!
//! The cache holds one base reference per indexed block; request-side
//! references are managed by the engine, mirroring the radix path.

use std::collections::HashMap;

use crate::kv_cache::{BlockId, PagedKvCache};
use crate::spec::TokenId;

/// Hash chain seed (arbitrary fixed constant).
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// splitmix64-style avalanche for one token into the running hash.
fn mix(h: u64, x: u64) -> u64 {
    let mut z = h ^ x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Chained hash of one block given the parent chain hash.
fn block_hash(parent: u64, block: &[TokenId]) -> u64 {
    let mut h = parent;
    for &t in block {
        h = mix(h, t as u64);
    }
    h
}

struct ApcEntry {
    block: BlockId,
    last_used: u64,
}

/// A global table from chained block hash to the physical block holding it.
pub struct ApcCache {
    map: HashMap<u64, ApcEntry>,
    clock: u64,
    num_queries: usize,
    hit_queries: usize,
    hit_tokens: usize,
}

impl Default for ApcCache {
    fn default() -> Self {
        Self::new()
    }
}

impl ApcCache {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            clock: 0,
            num_queries: 0,
            hit_queries: 0,
            hit_tokens: 0,
        }
    }

    pub fn num_blocks(&self) -> usize {
        self.map.len()
    }
    pub fn hit_tokens(&self) -> usize {
        self.hit_tokens
    }
    pub fn cache_hit_rate(&self) -> f64 {
        if self.num_queries == 0 {
            0.0
        } else {
            self.hit_queries as f64 / self.num_queries as f64
        }
    }

    /// Walk full blocks of `tokens`, chaining hashes, and stop at the first
    /// miss. Returns `(matched_tokens, matched_blocks)`. Only complete blocks
    /// participate; the tail partial block is left for regular prefill.
    pub fn match_blocks(&mut self, tokens: &[TokenId], block_size: usize) -> (usize, Vec<BlockId>) {
        self.num_queries += 1;
        let mut h = SEED;
        let mut blocks = Vec::new();
        for chunk in tokens.chunks(block_size) {
            if chunk.len() < block_size {
                break;
            }
            h = block_hash(h, chunk);
            match self.map.get_mut(&h) {
                Some(entry) => {
                    self.clock += 1;
                    entry.last_used = self.clock;
                    blocks.push(entry.block);
                }
                None => break,
            }
        }
        let matched = blocks.len() * block_size;
        if matched > 0 {
            self.hit_queries += 1;
            self.hit_tokens += matched;
        }
        (matched, blocks)
    }

    /// Index every full block of a finished sequence, taking a cache-owned
    /// reference on each newly indexed block. Blocks already present (identical
    /// chained hash implies identical content) are left untouched.
    pub fn insert_blocks(
        &mut self,
        tokens: &[TokenId],
        slots: &[(BlockId, usize)],
        kv: &mut PagedKvCache,
    ) {
        debug_assert_eq!(tokens.len(), slots.len());
        let mut h = SEED;
        let mut i = 0;
        for chunk in tokens.chunks(kv.block_size()) {
            if chunk.len() < kv.block_size() {
                break;
            }
            h = block_hash(h, chunk);
            if !self.map.contains_key(&h) {
                let block = slots[i].0;
                debug_assert!(slots[i..i + chunk.len()].iter().all(|s| s.0 == block));
                kv.inc_ref(block);
                self.map.insert(
                    h,
                    ApcEntry {
                        block,
                        last_used: self.clock,
                    },
                );
            }
            i += chunk.len();
        }
    }

    /// Evict up to `target` least-recently-used blocks, releasing the cache's
    /// references. Returns the number evicted.
    pub fn evict_lru(&mut self, kv: &mut PagedKvCache, target: usize) -> usize {
        let mut evicted = 0;
        while evicted < target {
            let Some((&hash, _)) = self
                .map
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(h, e)| (h, e))
            else {
                break;
            };
            let entry = self.map.remove(&hash).expect("entry exists");
            kv.dec_ref(entry.block);
            evicted += 1;
        }
        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(v: &[u32]) -> Vec<TokenId> {
        v.to_vec()
    }

    /// Materialize token blocks into a real paged cache and return slots.
    fn materialize(kv: &mut PagedKvCache, tokens: &[TokenId]) -> Vec<(BlockId, usize)> {
        let mut slots = Vec::new();
        let mut cur: Option<BlockId> = None;
        for (i, &t) in tokens.iter().enumerate() {
            if i % kv.block_size() == 0 {
                cur = Some(kv.alloc().unwrap());
            }
            let block = cur.expect("block allocated");
            assert!(kv.append(block, t));
            slots.push((block, i % kv.block_size()));
        }
        slots
    }

    #[test]
    fn full_blocks_hit_on_identical_prefix() {
        let mut kv = PagedKvCache::new(8, 4);
        let mut apc = ApcCache::new();
        let prompt = toks(&[10, 11, 12, 13, 14, 15, 16, 17, 99]);
        let slots = materialize(&mut kv, &prompt);
        apc.insert_blocks(&prompt, &slots, &mut kv);

        // Only the two full blocks (8 tokens) are reusable; the tail token 99
        // forms a partial block and must be re-prefilled.
        let (hit, blocks) = apc.match_blocks(&prompt, 4);
        assert_eq!(hit, 8);
        assert_eq!(blocks.len(), 2);

        // A divergent first block breaks the chain immediately.
        let other = toks(&[10, 11, 12, 42, 14, 15, 16, 17]);
        let (hit2, _) = apc.match_blocks(&other, 4);
        assert_eq!(hit2, 0, "chained hash must not reuse across divergent prefixes");
    }

    #[test]
    fn eviction_releases_block_references() {
        let mut kv = PagedKvCache::new(4, 2);
        let mut apc = ApcCache::new();
        let prompt = toks(&[1, 2, 3, 4]);
        let slots = materialize(&mut kv, &prompt);
        apc.insert_blocks(&prompt, &slots, &mut kv);
        assert_eq!(apc.num_blocks(), 2);
        // Request finishes: it releases its own references, leaving only the
        // cache-owned ones.
        let mut seen: Vec<BlockId> = Vec::new();
        for (b, _) in &slots {
            if !seen.contains(b) {
                seen.push(*b);
                kv.dec_ref(*b);
            }
        }
        let free_before = kv.num_free_blocks();
        assert_eq!(apc.evict_lru(&mut kv, 2), 2);
        assert_eq!(apc.num_blocks(), 0);
        assert!(kv.num_free_blocks() > free_before);
    }
}
