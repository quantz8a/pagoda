// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Paged KV cache with reference counting and copy-on-write.
//!
//! Real KV caches store `num_layers * 2 * head_dim` floats per token; this
//! reference version stores token ids instead so the whole engine runs without
//! a GPU or a tensor library. The block allocation / ref-count / COW machinery
//! is exactly what a tensor-backed implementation would swap in.

/// Physical page handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

struct Block {
    tokens: Vec<u32>,
    ref_count: usize,
}

/// Fixed-size paged cache. Blocks are returned to the free pool only when their
/// reference count drops to zero.
pub struct PagedKvCache {
    block_size: usize,
    blocks: Vec<Block>,
    free: Vec<usize>,
    num_allocations: usize,
    num_frees: usize,
}

impl PagedKvCache {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        assert!(num_blocks > 0 && block_size > 0);
        let blocks = (0..num_blocks)
            .map(|_| Block {
                tokens: Vec::with_capacity(block_size),
                ref_count: 0,
            })
            .collect();
        let free = (0..num_blocks).collect();
        Self {
            block_size,
            blocks,
            free,
            num_allocations: 0,
            num_frees: 0,
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
    pub fn num_free_blocks(&self) -> usize {
        self.free.len()
    }
    pub fn num_allocations(&self) -> usize {
        self.num_allocations
    }
    pub fn num_frees(&self) -> usize {
        self.num_frees
    }

    /// Allocate a fresh private block (reference count 1).
    pub fn alloc(&mut self) -> Option<BlockId> {
        let id = self.free.pop()?;
        let b = &mut self.blocks[id];
        b.tokens.clear();
        b.ref_count = 1;
        self.num_allocations += 1;
        Some(BlockId(id))
    }

    pub fn ref_count(&self, id: BlockId) -> usize {
        self.blocks[id.0].ref_count
    }
    pub fn len(&self, id: BlockId) -> usize {
        self.blocks[id.0].tokens.len()
    }
    pub fn is_empty(&self, id: BlockId) -> bool {
        self.blocks[id.0].tokens.is_empty()
    }
    pub fn is_full(&self, id: BlockId) -> bool {
        self.blocks[id.0].tokens.len() >= self.block_size
    }
    pub fn tokens(&self, id: BlockId) -> &[u32] {
        &self.blocks[id.0].tokens
    }

    pub fn inc_ref(&mut self, id: BlockId) {
        debug_assert!(self.blocks[id.0].ref_count > 0, "inc_ref on free block");
        self.blocks[id.0].ref_count += 1;
    }

    pub fn dec_ref(&mut self, id: BlockId) {
        let b = &mut self.blocks[id.0];
        debug_assert!(b.ref_count > 0, "dec_ref on free block");
        b.ref_count -= 1;
        if b.ref_count == 0 {
            b.tokens.clear();
            self.free.push(id.0);
            self.num_frees += 1;
        }
    }

    /// Append one token. Returns `false` when the block is already full.
    pub fn append(&mut self, id: BlockId, token: u32) -> bool {
        let b = &mut self.blocks[id.0];
        if b.tokens.len() >= self.block_size {
            return false;
        }
        b.tokens.push(token);
        true
    }

    /// Copy-on-write tail: if `id` is shared (`ref_count > 1`), clone it into a
    /// freshly allocated private block and release the caller's reference to the
    /// shared one. If it is already private, returns `id` unchanged.
    pub fn fork_block(&mut self, id: BlockId) -> Option<BlockId> {
        if self.ref_count(id) == 1 {
            return Some(id);
        }
        let copy = self.blocks[id.0].tokens.clone();
        let new_id = self.alloc()?;
        self.blocks[new_id.0].tokens = copy;
        self.dec_ref(id);
        Some(new_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free() {
        let mut kv = PagedKvCache::new(4, 4);
        assert_eq!(kv.num_free_blocks(), 4);
        let a = kv.alloc().unwrap();
        assert_eq!(kv.num_free_blocks(), 3);
        kv.dec_ref(a);
        assert_eq!(kv.num_free_blocks(), 4);
    }

    #[test]
    fn append_respects_block_size() {
        let mut kv = PagedKvCache::new(2, 2);
        let a = kv.alloc().unwrap();
        assert!(kv.append(a, 10));
        assert!(kv.append(a, 11));
        assert!(!kv.append(a, 12));
        assert_eq!(kv.tokens(a), &[10, 11]);
    }

    #[test]
    fn copy_on_write_isolates_writer() {
        let mut kv = PagedKvCache::new(4, 4);
        let shared = kv.alloc().unwrap();
        kv.append(shared, 1);
        kv.inc_ref(shared); // second owner
        assert_eq!(kv.ref_count(shared), 2);

        let private = kv.fork_block(shared).unwrap();
        assert_ne!(private, shared);
        assert_eq!(kv.ref_count(shared), 1);
        assert!(kv.append(private, 2));
        // The shared block must be untouched by writes to the fork.
        assert_eq!(kv.tokens(shared), &[1]);
        assert_eq!(kv.tokens(private), &[1, 2]);
    }
}
