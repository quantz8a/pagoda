// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: AGPL-3.0-only
//! Radix-tree prefix cache — the analogue of SGLang's `RadixAttention`.
//!
//! Every completed token sequence is inserted into a trie keyed by token id.
//! Each node records the physical KV slot (`BlockId`, `offset`) where that
//! token lives. Incoming prompts find their longest cached prefix; the matched
//! physical blocks are shared (references counted), the tail block is
//! copy-on-written when a sequence extends it, and the request's references are
//! released on completion. See `docs/DESIGN.md` for the accounting model.

use std::collections::HashMap;

use crate::kv_cache::{BlockId, PagedKvCache};
use crate::spec::TokenId;

/// Physical location of one token's KV slot.
pub type KvSlot = (BlockId, usize);

struct RadixNode {
    token: TokenId,
    parent: Option<usize>,
    children: HashMap<TokenId, usize>,
    /// Where this token's KV lives. `None` for token-only (non-materialized)
    /// nodes created by [`RadixCache::insert`].
    kv: Option<KvSlot>,
    /// Monotonic tick of the most recent prefix walk through this node;
    /// drives least-recently-used eviction.
    last_used: u64,
}

/// A trie over token sequences with hit-rate accounting and physical block
/// ownership.
pub struct RadixCache {
    nodes: Vec<RadixNode>,
    root: usize,
    /// Number of nodes pointing at each block (for eviction accounting).
    owned: HashMap<BlockId, usize>,
    /// Monotonic access clock for LRU bookkeeping.
    clock: u64,
    num_insertions: usize,
    num_queries: usize,
    hit_queries: usize,
    hit_tokens: usize,
}

impl Default for RadixCache {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixCache {
    pub fn new() -> Self {
        let root = RadixNode {
            token: TokenId::MAX,
            parent: None,
            children: HashMap::new(),
            kv: None,
            last_used: 0,
        };
        Self {
            nodes: vec![root],
            root: 0,
            owned: HashMap::new(),
            clock: 0,
            num_insertions: 0,
            num_queries: 0,
            hit_queries: 0,
            hit_tokens: 0,
        }
    }

    fn new_node(&mut self, token: TokenId, parent: usize) -> usize {
        let id = self.nodes.len();
        self.clock += 1;
        self.nodes.push(RadixNode {
            token,
            parent: Some(parent),
            children: HashMap::new(),
            kv: None,
            last_used: self.clock,
        });
        self.nodes[parent].children.insert(token, id);
        id
    }

    /// Walk the longest prefix of `tokens` that exists in the tree.
    /// Returns `(matched_len, last_matched_node_or_none)`.
    pub fn match_prefix(&self, tokens: &[TokenId]) -> (usize, Option<usize>) {
        let mut cur = self.root;
        let mut matched = 0;
        for &t in tokens {
            match self.nodes[cur].children.get(&t) {
                Some(&next) => {
                    cur = next;
                    matched += 1;
                }
                None => break,
            }
        }
        (matched, if matched > 0 { Some(cur) } else { None })
    }

    /// Walk the longest prefix whose nodes all carry physical KV slots, returning
    /// the slot of each reusable token in order.
    pub fn match_path(&mut self, tokens: &[TokenId]) -> (usize, Vec<KvSlot>) {
        let mut cur = self.root;
        let mut slots = Vec::new();
        for &t in tokens {
            match self.nodes[cur].children.get(&t) {
                Some(&next) => {
                    cur = next;
                    match self.nodes[cur].kv {
                        Some(slot) => {
                            self.clock += 1;
                            self.nodes[cur].last_used = self.clock;
                            slots.push(slot);
                        }
                        None => break,
                    }
                }
                None => break,
            }
        }
        (slots.len(), slots)
    }

    /// Insert tokens without physical slots (token-only mode, used by tests).
    /// Returns the number of newly created nodes.
    pub fn insert(&mut self, tokens: &[TokenId]) -> usize {
        self.num_insertions += 1;
        let mut cur = self.root;
        let mut i = 0;
        while i < tokens.len() {
            match self.nodes[cur].children.get(&tokens[i]) {
                Some(&next) => {
                    cur = next;
                    i += 1;
                }
                None => break,
            }
        }
        for j in i..tokens.len() {
            let id = self.new_node(tokens[j], cur);
            cur = id;
        }
        tokens.len() - i
    }

    /// Insert a token sequence together with the physical slot of each token.
    /// Newly referenced blocks are adopted with a cache base reference.
    pub fn insert_with_kv(
        &mut self,
        tokens: &[TokenId],
        slots: &[KvSlot],
        kv: &mut PagedKvCache,
    ) -> usize {
        debug_assert_eq!(tokens.len(), slots.len());
        self.num_insertions += 1;
        let mut cur = self.root;
        let mut i = 0;
        while i < tokens.len() {
            match self.nodes[cur].children.get(&tokens[i]) {
                Some(&next) => {
                    cur = next;
                    i += 1;
                }
                None => break,
            }
        }
        for j in i..tokens.len() {
            let id = self.new_node(tokens[j], cur);
            self.nodes[id].kv = Some(slots[j]);
            self.mark_owned(slots[j].0, kv);
            cur = id;
        }
        tokens.len() - i
    }

    /// Give the cache a base reference on `block` the first time it is indexed.
    fn mark_owned(&mut self, block: BlockId, kv: &mut PagedKvCache) {
        let count = self.owned.entry(block).or_insert(0);
        if *count == 0 {
            kv.inc_ref(block);
        }
        *count += 1;
    }

    /// Record a token-only prefix lookup (metrics only). Returns matched length.
    pub fn record_match(&mut self, tokens: &[TokenId]) -> usize {
        let (matched, _) = self.match_prefix(tokens);
        self.record_matched(matched);
        matched
    }

    /// Record a prefix lookup that already computed its matched length.
    pub fn record_matched(&mut self, matched: usize) {
        self.num_queries += 1;
        self.hit_tokens += matched;
        if matched > 0 {
            self.hit_queries += 1;
        }
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len() - 1
    }
    pub fn num_insertions(&self) -> usize {
        self.num_insertions
    }
    pub fn num_queries(&self) -> usize {
        self.num_queries
    }
    pub fn hit_queries(&self) -> usize {
        self.hit_queries
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

    /// Evict up to `target` least-recently-used leaf entries, releasing their
    /// cache-owned KV block references. Returns the number of entries evicted.
    pub fn evict_lru(&mut self, kv: &mut PagedKvCache, target: usize) -> usize {
        let mut evicted = 0;
        while evicted < target {
            let mut best: Option<usize> = None;
            for idx in 1..self.nodes.len() {
                let node = &self.nodes[idx];
                if node.kv.is_some() && node.children.is_empty() {
                    best = Some(match best {
                        None => idx,
                        Some(b) => {
                            if node.last_used < self.nodes[b].last_used {
                                idx
                            } else {
                                b
                            }
                        }
                    });
                }
            }
            let Some(idx) = best else { break };

            let (parent, token, slot) = {
                let node = &self.nodes[idx];
                (node.parent.expect("non-root node"), node.token, node.kv)
            };
            if let Some((block, _off)) = slot {
                let reached_zero = match self.owned.get_mut(&block) {
                    Some(owned) => {
                        *owned = owned.saturating_sub(1);
                        *owned == 0
                    }
                    None => false,
                };
                if reached_zero {
                    self.owned.remove(&block);
                    kv.dec_ref(block);
                }
            }
            self.nodes[parent].children.remove(&token);
            self.nodes[idx].kv = None;
            self.nodes[idx].parent = None;
            self.nodes[idx].last_used = 0;
            evicted += 1;
        }
        evicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<u32> {
        s.bytes().map(|b| b as u32 + 3).collect()
    }

    #[test]
    fn match_and_insert() {
        let mut rc = RadixCache::new();
        assert_eq!(rc.insert(&toks("hello world")), 11);
        let (matched, _) = rc.match_prefix(&toks("hello there"));
        assert_eq!(matched, 6); // "hello "
    }

    #[test]
    fn hit_rate_accounting() {
        let mut rc = RadixCache::new();
        rc.insert(&toks("the quick"));
        let m = rc.record_match(&toks("the quick brown"));
        assert_eq!(m, 9);
        rc.record_match(&toks("unrelated"));
        assert_eq!(rc.num_queries(), 2);
        assert_eq!(rc.hit_queries(), 1);
        assert!((rc.cache_hit_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn match_path_reports_slots() {
        let mut kv = PagedKvCache::new(8, 4);
        let mut rc = RadixCache::new();
        let tokens = toks("hello");
        let b0 = kv.alloc().unwrap();
        let b1 = kv.alloc().unwrap();
        let slots: Vec<KvSlot> = (0..tokens.len())
            .map(|i| {
                if i < 2 {
                    (b0, i)
                } else {
                    (b1, i - 2)
                }
            })
            .collect();
        rc.insert_with_kv(&tokens, &slots, &mut kv);
        assert_eq!(rc.match_path(&tokens).1, slots);
        // Both physical blocks carry a cache base reference now.
        assert_eq!(kv.ref_count(b0), 2);
        assert_eq!(kv.ref_count(b1), 2);
    }

    #[test]
    fn evict_lru_releases_owned_blocks() {
        let mut kv = PagedKvCache::new(8, 4);
        let mut rc = RadixCache::new();

        let tokens_a = toks("aaaa");
        let b0 = kv.alloc().unwrap();
        let b1 = kv.alloc().unwrap();
        let slots_a: Vec<KvSlot> = vec![(b0, 0), (b0, 1), (b0, 2), (b1, 0)];
        rc.insert_with_kv(&tokens_a, &slots_a, &mut kv);

        let tokens_b = toks("bbbb");
        let b2 = kv.alloc().unwrap();
        let b3 = kv.alloc().unwrap();
        let slots_b: Vec<KvSlot> = vec![(b2, 0), (b2, 1), (b2, 2), (b3, 0)];
        rc.insert_with_kv(&tokens_b, &slots_b, &mut kv);

        // Touch "aaaa" so that "bbbb" becomes the LRU victim.
        let _ = rc.match_path(&tokens_a);

        assert_eq!(kv.ref_count(b2), 2);
        assert_eq!(kv.ref_count(b3), 2);

        let evicted = rc.evict_lru(&mut kv, 1);
        assert_eq!(evicted, 1);

        // "aaaa" is still fully cached; the LRU leaf of "bbbb" is gone.
        assert_eq!(rc.match_path(&tokens_a).0, tokens_a.len());
        assert!(rc.match_path(&tokens_b).0 < tokens_b.len());
    }
}

