// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Tokenizer abstraction plus a deterministic byte-level reference tokenizer.

/// Special token ids used by [`ByteTokenizer`].
pub const PAD: u32 = 0;
pub const BOS: u32 = 1;
pub const EOS: u32 = 2;
/// Byte tokens start here and occupy ids `3..=258`.
pub const FIRST_BYTE_ID: u32 = 3;
pub const BYTE_VOCAB_SIZE: usize = 256;

/// Minimal tokenizer interface. Real deployments plug in HF BPE/Unigram
/// tokenizers (via the `tokenizers` crate) or GGUF `llama.cpp` vocabularies.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn eos_token_id(&self) -> u32;
    fn bos_token_id(&self) -> Option<u32>;
    fn vocab_size(&self) -> usize;
    fn name(&self) -> &'static str;
    /// Whether every token id maps to exactly one byte. Constrained decoding
    /// (grammar masking) is only sound on byte-level tokenizers; the engine
    /// rejects grammar requests on any tokenizer for which this is `false`.
    fn is_byte_level(&self) -> bool {
        false
    }
}

/// A fully reversible, dependency-free tokenizer: every byte maps to exactly one
/// token id (`byte + 3`), and decoding is lossless. It keeps the whole pipeline
/// deterministic and lets the toy model operate over plain UTF-8 bytes.
#[derive(Clone, Copy, Debug, Default)]
pub struct ByteTokenizer;

impl ByteTokenizer {
    pub fn new() -> Self {
        ByteTokenizer
    }
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.as_bytes()
            .iter()
            .map(|&b| u32::from(b) + FIRST_BYTE_ID)
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter(|&&t| t >= FIRST_BYTE_ID && t < FIRST_BYTE_ID + BYTE_VOCAB_SIZE as u32)
            .map(|&t| (t - FIRST_BYTE_ID) as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn eos_token_id(&self) -> u32 {
        EOS
    }

    fn bos_token_id(&self) -> Option<u32> {
        Some(BOS)
    }

    fn vocab_size(&self) -> usize {
        3 + BYTE_VOCAB_SIZE
    }

    fn name(&self) -> &'static str {
        "byte"
    }

    fn is_byte_level(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_tokenizer_roundtrip() {
        let tk = ByteTokenizer::new();
        for text in ["", "hello", "hello world", "中文 文本 ✓", "What is 1+1?"] {
            let tokens = tk.encode(text);
            assert_eq!(tk.decode(&tokens), text);
        }
    }

    #[test]
    fn byte_tokenizer_specials() {
        let tk = ByteTokenizer::new();
        assert_eq!(tk.eos_token_id(), EOS);
        assert_eq!(tk.bos_token_id(), Some(BOS));
        assert_eq!(tk.vocab_size(), 259);
        assert_eq!(tk.encode("A"), vec![68]); // b'A' == 65, +3
    }
}
