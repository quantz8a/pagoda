//! Inline fan-out. Pagoda's `rs-mm` stays single-threaded and dependency-free;
//! the SGLang crate puts rayon behind this same seam.

/// Apply `f(chunk_index, chunk)` over disjoint `chunk_size`-element windows.
/// The final chunk is short when `chunk_size` does not divide the length.
pub fn for_chunks_mut<T>(buf: &mut [T], chunk_size: usize, f: impl Fn(usize, &mut [T])) {
    for (index, chunk) in buf.chunks_mut(chunk_size).enumerate() {
        f(index, chunk);
    }
}

/// Run `f` on the calling thread. Present so resize can keep the same shape
/// as the upstream two-pass entry point.
pub fn in_pool<R>(f: impl FnOnce() -> R) -> R {
    f()
}
