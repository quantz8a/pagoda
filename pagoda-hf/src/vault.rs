// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Cross-request KV vault: the tensor-level half of prefix caching.
//!
//! The pagoda core engine keeps a *logical* prefix cache (token ids in paged
//! blocks); this vault is the companion *physical* store: finished sessions'
//! KV snapshots keyed by token path in a **radix trie**, so a later request
//! whose prompt shares a prefix — at ANY depth — can graft the
//! already-computed tensors instead of recomputing them (true RadixAttention).
//!
//! Design notes:
//!
//! * One entry per offered token path; the trie indexes every depth of every
//!   path, so matching is a plain walk: descend while the prompt agrees,
//!   remember the deepest node covered by some entry, slice that entry's
//!   snapshot down (`cache_prefix`). Causal attention makes the sliced prefix
//!   bit-identical to a recompute, and entries covering the same prefix are
//!   interchangeable (same tokens + same weights ⇒ same KV values).
//! * Snapshots are cheap: candle tensors are Arc-shared, so an entry pins the
//!   underlying K/V storage but copies nothing.
//! * Bounded two ways: `max_entries` (count) and `max_bytes` (sum of
//!   [`VaultEntry::bytes`]), both LRU-evicted. Eviction only forfeits future
//!   graft opportunities — correctness never depends on the vault.
//! * Insert / match / evict are all O(path length); node fan-out is a
//!   per-token HashMap. Stale trie references (to evicted entries) self-heal:
//!   a match treats a missing entry as absent and keeps walking shallower.

use std::collections::HashMap;

/// Byte cost of a cached snapshot, for the vault's memory budget.
pub trait VaultEntry {
    fn bytes(&self) -> usize;
}

/// One offered snapshot: the full token path it covers plus the cache.
struct Entry<C> {
    path: Vec<u32>,
    cache: C,
    tick: u64,
    bytes: usize,
}

#[derive(Default)]
struct Node {
    /// token -> child index.
    children: HashMap<u32, usize>,
    /// An entry whose path passes through this depth (interchangeable with
    /// any other such entry). May dangle after eviction; checked lazily.
    entry: Option<u64>,
}

pub struct KvVault<C> {
    entries: HashMap<u64, Entry<C>>,
    nodes: Vec<Node>,
    next_id: u64,
    clock: u64,
    total_bytes: usize,
    max_entries: usize,
    max_bytes: usize,
}

impl<C: Clone + VaultEntry> KvVault<C> {
    /// `max_entries` caps snapshot count, `max_bytes` caps their total
    /// estimated tensor bytes; both are enforced with LRU eviction.
    pub fn with_budget(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            nodes: vec![Node::default()], // root
            next_id: 0,
            clock: 0,
            total_bytes: 0,
            max_entries: max_entries.max(1),
            max_bytes: max_bytes.max(1),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Offer a finished session's cache under its full fed token path.
    /// Every prefix depth of `path` becomes graftable. Re-offering the same
    /// path replaces the previous snapshot instead of doubling the budget.
    pub fn offer(&mut self, path: &[u32], cache: C) {
        if path.is_empty() {
            return;
        }
        self.clock += 1;
        let id = self.next_id;
        self.next_id += 1;
        let bytes = cache.bytes();
        // Replace an identical existing path so repeat traffic doesn't leak.
        if let Some(old) = self.find_exact(path) {
            self.evict(old);
        }
        self.entries.insert(
            id,
            Entry {
                path: path.to_vec(),
                cache,
                tick: self.clock,
                bytes,
            },
        );
        self.total_bytes += bytes;
        // Index every depth.
        let mut node = 0usize;
        for &tok in path {
            let next = match self.nodes[node].children.get(&tok) {
                Some(&c) => c,
                None => {
                    let c = self.nodes.len();
                    self.nodes.push(Node::default());
                    self.nodes[node].children.insert(tok, c);
                    c
                }
            };
            node = next;
            self.nodes[node].entry = Some(id);
        }
        self.enforce_budget();
    }

    /// Longest covered prefix of `prompt`, capped at `max_len`. Returns a
    /// covering entry's cache and the covered depth; the caller slices the
    /// cache when it overshoots. Entries are interchangeable at a given
    /// depth, so any covering entry is correct.
    pub fn longest_prefix(&mut self, prompt: &[u32], max_len: usize) -> Option<(C, usize)> {
        let cap = max_len.min(prompt.len());
        if cap == 0 {
            return None;
        }
        let mut node = 0usize;
        let mut best: Option<(u64, usize)> = None;
        for (depth, &tok) in prompt[..cap].iter().enumerate() {
            let Some(&child) = self.nodes[node].children.get(&tok) else {
                break;
            };
            node = child;
            if let Some(id) = self.nodes[node].entry {
                if self.entries.contains_key(&id) {
                    best = Some((id, depth + 1));
                }
            }
        }
        let (id, covered) = best?;
        self.clock += 1;
        let entry = self.entries.get_mut(&id)?;
        entry.tick = self.clock;
        Some((entry.cache.clone(), covered))
    }

    /// The entry whose path is exactly `path`, if any.
    fn find_exact(&self, path: &[u32]) -> Option<u64> {
        let mut node = 0usize;
        for &tok in path {
            node = *self.nodes[node].children.get(&tok)?;
        }
        let id = self.nodes[node].entry?;
        let entry = self.entries.get(&id)?;
        (entry.path == path).then_some(id)
    }

    fn enforce_budget(&mut self) {
        while self.entries.len() > self.max_entries || self.total_bytes > self.max_bytes {
            let Some(&victim) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.tick)
                .map(|(id, _)| id)
            else {
                break;
            };
            self.evict(victim);
        }
    }

    /// Remove an entry and lazily prune the now-childless tail of its path.
    fn evict(&mut self, id: u64) {
        let Some(entry) = self.entries.remove(&id) else {
            return;
        };
        self.total_bytes -= entry.bytes;
        // Clear this id from the nodes along its path (deeper nodes may keep
        // other entries' ids), then prune childless, entry-less tail nodes.
        let mut chain = vec![0usize];
        let mut node = 0usize;
        for &tok in &entry.path {
            let Some(&child) = self.nodes[node].children.get(&tok) else {
                break;
            };
            chain.push(child);
            node = child;
        }
        for &n in &chain[1..] {
            if self.nodes[n].entry == Some(id) {
                self.nodes[n].entry = None;
            }
        }
        // Prune from the deepest node up while a node carries nothing.
        let mut depth = chain.len() - 1;
        while depth > 0 {
            let n = chain[depth];
            if self.nodes[n].entry.is_some() || !self.nodes[n].children.is_empty() {
                break;
            }
            let tok = entry.path[depth - 1];
            self.nodes[chain[depth - 1]].children.remove(&tok);
            depth -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl VaultEntry for Vec<u32> {
        fn bytes(&self) -> usize {
            self.len() * 4
        }
    }

    impl VaultEntry for u32 {
        fn bytes(&self) -> usize {
            4
        }
    }

    #[test]
    fn any_depth_prefix_hit() {
        let mut vault: KvVault<Vec<u32>> = KvVault::with_budget(8, 1 << 20);
        vault.offer(&[1, 2, 3, 4, 5], vec![9]);
        // Sub-key depths now hit: the trie indexes every prefix.
        for (depth, expect) in [(1usize, 1usize), (2, 2), (3, 3), (5, 5)] {
            let (_, covered) = vault
                .longest_prefix(&[1, 2, 3, 4, 5], depth)
                .expect("hit");
            assert_eq!(covered, expect);
        }
        // A prompt sharing only the first two tokens grafts depth 2.
        let (_, covered) = vault.longest_prefix(&[1, 2, 9], 2).expect("hit");
        assert_eq!(covered, 2);
        // Continuation grafts the whole path.
        let (_, covered) = vault.longest_prefix(&[1, 2, 3, 4, 5, 6, 7], 6).expect("hit");
        assert_eq!(covered, 5);
        // No common prefix -> miss.
        assert!(vault.longest_prefix(&[7, 8], 2).is_none());
    }

    #[test]
    fn lru_eviction() {
        let mut vault: KvVault<u32> = KvVault::with_budget(2, 1 << 20);
        vault.offer(&[1], 1);
        vault.offer(&[2], 2);
        vault.offer(&[3], 3); // evicts [1]
        assert!(vault.longest_prefix(&[1], 1).is_none());
        assert!(vault.longest_prefix(&[3], 1).is_some());
        assert_eq!(vault.len(), 2);
    }

    #[test]
    fn byte_budget_evicts() {
        // Budget fits exactly one 4-token path (16 bytes each via Vec<u32>).
        let mut vault: KvVault<Vec<u32>> = KvVault::with_budget(16, 20);
        vault.offer(&[1, 2, 3, 4], vec![1, 2, 3, 4]);
        assert_eq!(vault.total_bytes(), 16);
        vault.offer(&[5, 6, 7, 8], vec![5, 6, 7, 8]); // 32 > 20 -> evict oldest
        assert_eq!(vault.len(), 1);
        assert!(vault.longest_prefix(&[1, 2, 3, 4], 4).is_none());
        assert!(vault.longest_prefix(&[5, 6, 7, 8], 4).is_some());
    }

    #[test]
    fn reoffer_replaces_not_leaks() {
        let mut vault: KvVault<Vec<u32>> = KvVault::with_budget(8, 1 << 20);
        vault.offer(&[1, 2, 3], vec![1, 1, 1]);
        vault.offer(&[1, 2, 3], vec![2, 2, 2]);
        assert_eq!(vault.len(), 1);
        assert_eq!(vault.total_bytes(), 12);
        let (cache, covered) = vault.longest_prefix(&[1, 2, 3], 3).expect("hit");
        assert_eq!(covered, 3);
        assert_eq!(cache, vec![2, 2, 2], "newest snapshot wins");
    }

    #[test]
    fn eviction_prunes_trie_but_keeps_shared_prefix() {
        let mut vault: KvVault<u32> = KvVault::with_budget(2, 1 << 20);
        vault.offer(&[1, 2, 3, 4], 10);
        vault.offer(&[1, 2, 8], 20);
        vault.offer(&[9], 30); // evicts [1,2,3,4] (LRU)
        // Shared prefix [1,2] survives via the surviving entry.
        let (_, covered) = vault.longest_prefix(&[1, 2, 3, 4], 4).expect("hit");
        assert_eq!(covered, 2);
        // The evicted deep tail [1,2,3,4] is gone beyond depth 2.
        assert!(vault.longest_prefix(&[1, 2, 3, 4], 3).is_some_and(|(_, c)| c == 2));
    }
}