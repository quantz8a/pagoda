// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Cross-request KV vault: the tensor-level half of prefix caching.
//!
//! The pagoda core engine keeps a *logical* prefix cache (token ids in paged
//! blocks); this vault is the companion *physical* store: finished sessions'
//! KV snapshots keyed by token path, so a later request whose prompt shares
//! a prefix can graft the already-computed tensors instead of recomputing
//! them (true RadixAttention).
//!
//! Design notes:
//!
//! * Keys are full token paths; a graft matches the longest key that is a
//!   prefix of the incoming prompt, then slices the cached tensors to the
//!   covered length (exact by causality). Both a finished sequence's prompt
//!   prefix and its full path are inserted, so repeat-prompt and
//!   continue-this-conversation patterns both hit.
//! * Snapshots are cheap: candle tensors are Arc-shared, so a stored entry
//!   pins the underlying K/V storage but copies nothing.
//! * Bounded by `max_entries` with LRU eviction. Eviction only forfeits
//!   future graft opportunities — correctness never depends on the vault.
//! * Linear scans on insert-evict and match: at the default 128 entries this
//!   is microseconds, nothing next to a model forward. A radix-keyed vault
//!   is the scale-up path if entry counts grow.

use std::collections::HashMap;

pub struct KvVault<C> {
    entries: HashMap<Vec<u32>, (C, u64)>,
    clock: u64,
    max_entries: usize,
}

impl<C: Clone> KvVault<C> {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
            max_entries: max_entries.max(1),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Offer a finished session's cache under its full token path and its
    /// prompt prefix (`prompt_len`).
    pub fn offer(&mut self, path: &[u32], prompt_len: usize, cache: C) {
        let prompt_len = prompt_len.min(path.len());
        if prompt_len > 0 {
            self.insert(path[..prompt_len].to_vec(), cache.clone());
        }
        if path.len() > prompt_len {
            self.insert(path.to_vec(), cache);
        }
    }

    /// Longest stored key that is a prefix of `prompt`, capped at `max_len`.
    /// Returns the snapshot and the covered length (`min(key_len, max_len)`);
    /// the caller slices the cache when the key overshoots the cap.
    pub fn longest_prefix(&mut self, prompt: &[u32], max_len: usize) -> Option<(C, usize)> {
        let cap = max_len.min(prompt.len());
        if cap == 0 {
            return None;
        }
        let mut best: Option<&Vec<u32>> = None;
        for key in self.entries.keys() {
            if key.len() > prompt.len() {
                continue;
            }
            if key.as_slice() != &prompt[..key.len()] {
                continue;
            }
            let covered = key.len().min(cap);
            if covered == 0 {
                continue;
            }
            let better = match best {
                Some(b) => key.len().min(cap) > b.len().min(cap),
                None => true,
            };
            if better {
                best = Some(key);
            }
        }
        let key = best?.clone();
        self.clock += 1;
        let (cache, _) = self.entries.get_mut(&key).map(|(c, tick)| {
            *tick = self.clock;
            (c.clone(), *tick)
        })?;
        let covered = key.len().min(cap);
        Some((cache, covered))
    }

    fn insert(&mut self, key: Vec<u32>, cache: C) {
        if !self.entries.contains_key(&key) && self.entries.len() >= self.max_entries {
            // Evict the least-recently-used entry.
            let victim = self
                .entries
                .iter()
                .min_by_key(|(_, (_, tick))| *tick)
                .map(|(k, _)| k.clone());
            if let Some(victim) = victim {
                self.entries.remove(&victim);
            }
        }
        self.clock += 1;
        self.entries.insert(key, (cache, self.clock));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_prefix_and_cap() {
        let mut vault: KvVault<Vec<u32>> = KvVault::new(8);
        vault.offer(&[1, 2, 3, 4, 5], 3, vec![9]); // keys [1,2,3] and [1,2,3,4,5]
        // Keys are sparse (prompt prefix + full path), so a prompt that
        // shares only a sub-key prefix ([1,2,9] vs key [1,2,3]) misses:
        // matching is exact at stored key depths, like a coarse radix tree.
        assert!(vault.longest_prefix(&[1, 2, 9], 2).is_none());
        // Repeat of the prompt prefix hits at full key depth.
        let (_, covered) = vault.longest_prefix(&[1, 2, 3, 8], 2).expect("hit");
        assert_eq!(covered, 2);
        // Full-path key overshoots the cap: covered is capped.
        let (_, covered) = vault.longest_prefix(&[1, 2, 3, 4, 5], 4).expect("hit");
        assert_eq!(covered, 4);
        // Continuation (stored path + new suffix) grafts the whole path.
        let (_, covered) = vault.longest_prefix(&[1, 2, 3, 4, 5, 6, 7], 6).expect("hit");
        assert_eq!(covered, 5);
        // No common prefix -> miss.
        assert!(vault.longest_prefix(&[7, 8], 2).is_none());
    }

    #[test]
    fn lru_eviction() {
        let mut vault: KvVault<u32> = KvVault::new(2);
        vault.offer(&[1], 1, 1);
        vault.offer(&[2], 1, 2);
        vault.offer(&[3], 1, 3); // evicts [1]
        assert!(vault.longest_prefix(&[1], 1).is_none());
        assert!(vault.longest_prefix(&[3], 1).is_some());
        assert_eq!(vault.len(), 2);
    }
}
