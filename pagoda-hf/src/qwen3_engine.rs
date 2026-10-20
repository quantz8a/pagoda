// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Qwen3.8 behind [`pagoda::ModelEngine`].
//!
//! The session owns the real cache: 16 full-attention KV tensors plus the
//! 48 Gated Delta recurrent states. The engine's paged token cache stays the
//! scheduler's bookkeeping. One stored snapshot is enough for the single
//! resident conversation this GPU serves.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use pagoda::{ModelEngine, ModelSession};

use crate::qwen3_5::{Qwen35, Qwen35Cache};

const L1_GPU: usize = 2;
const L2_HOST: usize = 8;
const DISK_FILES: usize = 64;
const WEEK_SECS: u64 = 7 * 24 * 60 * 60;

struct Slot {
    tokens: Vec<u32>,
    /// L1: GPU-resident copy. `None` after it is demoted.
    gpu: Option<Qwen35Cache>,
    /// L2: host bytes, kept after the GPU copy is dropped.
    host: Vec<u8>,
    used: u64,
}

struct DiskRef {
    path: PathBuf,
    tokens: Vec<u32>,
    created: u64,
}

struct Vault {
    slots: Vec<Slot>,
    disk: Vec<DiskRef>,
    dir: PathBuf,
    ttl_secs: u64,
    layers: usize,
}

pub struct QwenEngine {
    model: Arc<Qwen35>,
    vault: Mutex<Vault>,
}

pub struct QwenSession {
    model: Arc<Qwen35>,
    cache: Qwen35Cache,
    fed: usize,
    last_logits: Option<Vec<f32>>,
}

impl QwenEngine {
    pub fn new(model: Arc<Qwen35>) -> Self {
        let dir = kv_dir();
        let ttl_secs = std::env::var("PAGODA_KV_TTL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(WEEK_SECS);
        let layers = model.n_layers();
        let disk = load_disk(&dir, ttl_secs, layers);
        eprintln!(
            "==> kv cache L1 {L1_GPU} GPU, L2 {L2_HOST} host, L3 {} files ttl {ttl_secs}s in {}",
            disk.len(),
            dir.display()
        );
        Self {
            model,
            vault: Mutex::new(Vault {
                slots: Vec::new(),
                disk,
                dir,
                ttl_secs,
                layers,
            }),
        }
    }
}

impl ModelEngine for QwenEngine {
    fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    fn forward(&self, context: &[u32]) -> Vec<f32> {
        let mut cache = self.model.new_cache();
        match self.model.forward_logits(context, &mut cache) {
            Ok(logits) => logits,
            Err(e) => {
                eprintln!("[qwen3.8] forward fell back to uniform logits: {e:#}");
                vec![0.0; self.model.vocab_size()]
            }
        }
    }

    fn name(&self) -> &'static str {
        "qwen3.8"
    }

    fn begin_session(&self) -> Option<Box<dyn ModelSession>> {
        Some(Box::new(QwenSession {
            model: Arc::clone(&self.model),
            cache: self.model.new_cache(),
            fed: 0,
            last_logits: None,
        }))
    }

    fn graft_session(&self, prompt: &[u32], max_len: usize) -> Option<Box<dyn ModelSession>> {
        let mut vault = self.vault.lock().ok()?;
        let gpu = vault.take_gpu(prompt, max_len);
        let host = vault.take_host(prompt, max_len);
        let disk_len = vault
            .best_disk(prompt, max_len)
            .map(|item| item.tokens.len())
            .unwrap_or(0);
        let gpu_len = gpu.as_ref().map(|(n, _)| *n).unwrap_or(0);
        let host_len = host.as_ref().map(|(n, _)| *n).unwrap_or(0);
        if gpu_len > 0 && gpu_len >= host_len && gpu_len >= disk_len {
            let (fed, cache) = gpu?;
            eprintln!("==> kv L1 graft {fed} tokens");
            return Some(session(Arc::clone(&self.model), cache, fed));
        }
        if host_len > 0 && host_len >= disk_len {
            let (fed, bytes) = host?;
            let cache = Qwen35Cache::from_bytes(&bytes).ok()?;
            eprintln!("==> kv L2 graft {fed} tokens");
            return Some(session(Arc::clone(&self.model), cache, fed));
        }
        if disk_len == 0 {
            return None;
        }
        let (path, fed, tokens) = {
            let hit = vault.best_disk(prompt, max_len)?;
            (hit.path.clone(), hit.tokens.len(), hit.tokens.clone())
        };
        let bytes = fs::read(&path).ok()?;
        let body = strip_disk_header(&bytes)?.to_vec();
        let cache = Qwen35Cache::from_bytes(&body).ok()?;
        vault.remember(tokens, None, body);
        eprintln!("==> kv L3 graft {fed} tokens");
        Some(session(Arc::clone(&self.model), cache, fed))
    }

    fn offer_session_kv(
        &self,
        tokens: &[u32],
        _prompt_len: usize,
        session: Box<dyn ModelSession>,
    ) {
        let Some(qs) = session.as_any().and_then(|a| a.downcast_ref::<QwenSession>()) else {
            return;
        };
        let fed = qs.fed.min(tokens.len());
        if fed == 0 {
            return;
        }
        let Ok(host) = qs.cache.to_bytes() else {
            return;
        };
        let gpu = qs.cache.snapshot().ok();
        let tokens = tokens[..fed].to_vec();
        if let Ok(mut vault) = self.vault.lock() {
            if let Err(e) = vault.save_disk(&tokens, &host) {
                eprintln!("==> kv disk save failed: {e:#}");
            }
            vault.remember(tokens, gpu, host);
        }
    }
}

fn session(model: Arc<Qwen35>, cache: Qwen35Cache, fed: usize) -> Box<dyn ModelSession> {
    Box::new(QwenSession {
        model,
        cache,
        fed,
        last_logits: None,
    })
}

impl ModelSession for QwenSession {
    fn context_len(&self) -> usize {
        self.fed
    }

    fn forward(&mut self, new_tokens: &[u32]) -> Vec<f32> {
        if new_tokens.is_empty() {
            return self
                .last_logits
                .clone()
                .unwrap_or_else(|| vec![0.0; self.model.vocab_size()]);
        }
        match self.model.forward_logits(new_tokens, &mut self.cache) {
            Ok(logits) => {
                self.fed += new_tokens.len();
                self.last_logits = Some(logits.clone());
                logits
            }
            Err(e) => {
                eprintln!("[qwen3.8] session forward fell back to uniform logits: {e:#}");
                vec![0.0; self.model.vocab_size()]
            }
        }
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn fork(&self) -> Option<Box<dyn ModelSession>> {
        let cache = self.cache.snapshot().ok()?;
        Some(Box::new(Self {
            model: Arc::clone(&self.model),
            cache,
            fed: self.fed,
            last_logits: self.last_logits.clone(),
        }))
    }
}

impl Vault {
    fn remember(&mut self, tokens: Vec<u32>, gpu: Option<Qwen35Cache>, host: Vec<u8>) {
        let now = unix_secs();
        if let Some(slot) = self.slots.iter_mut().find(|slot| slot.tokens == tokens) {
            if gpu.is_some() {
                slot.gpu = gpu;
            }
            if !host.is_empty() {
                slot.host = host;
            }
            slot.used = now;
        } else {
            self.slots.push(Slot {
                tokens,
                gpu,
                host,
                used: now,
            });
        }
        self.evict();
    }

    fn evict(&mut self) {
        let mut gpu_idx: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.gpu.is_some())
            .map(|(i, _)| i)
            .collect();
        gpu_idx.sort_by_key(|i| self.slots[*i].used);
        while gpu_idx.len() > L1_GPU {
            let i = gpu_idx.remove(0);
            self.slots[i].gpu = None;
        }
        if self.slots.len() > L2_HOST {
            let mut order: Vec<usize> = (0..self.slots.len()).collect();
            order.sort_by_key(|i| self.slots[*i].used);
            let drop_n = self.slots.len() - L2_HOST;
            let drop: std::collections::HashSet<usize> = order.into_iter().take(drop_n).collect();
            let mut kept = Vec::new();
            for (i, slot) in self.slots.drain(..).enumerate() {
                if !drop.contains(&i) {
                    kept.push(slot);
                }
            }
            self.slots = kept;
        }
    }

    fn take_gpu(&mut self, prompt: &[u32], max_len: usize) -> Option<(usize, Qwen35Cache)> {
        let i = best_slot(&self.slots, prompt, max_len, true)?;
        self.slots[i].used = unix_secs();
        let cache = self.slots[i].gpu.as_ref()?.snapshot().ok()?;
        Some((self.slots[i].tokens.len(), cache))
    }

    fn take_host(&mut self, prompt: &[u32], max_len: usize) -> Option<(usize, Vec<u8>)> {
        let i = best_slot(&self.slots, prompt, max_len, false)?;
        if self.slots[i].gpu.is_some() {
            return None;
        }
        self.slots[i].used = unix_secs();
        Some((self.slots[i].tokens.len(), self.slots[i].host.clone()))
    }

    fn best_disk(&self, prompt: &[u32], max_len: usize) -> Option<&DiskRef> {
        self.disk
            .iter()
            .filter(|item| prefix_ok(&item.tokens, prompt, max_len))
            .max_by_key(|item| item.tokens.len())
    }

    fn save_disk(&mut self, tokens: &[u32], host: &[u8]) -> std::io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let created = unix_secs();
        let name = format!("{created}-{:x}.kv", token_hash(tokens));
        let path = self.dir.join(name);
        let mut bytes = Vec::with_capacity(16 + tokens.len() * 4 + host.len());
        bytes.extend_from_slice(b"PGK1");
        bytes.extend_from_slice(&created.to_le_bytes());
        bytes.extend_from_slice(&(self.layers as u32).to_le_bytes());
        bytes.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for token in tokens {
            bytes.extend_from_slice(&u32::to_le_bytes(*token));
        }
        bytes.extend_from_slice(host);
        fs::write(&path, &bytes)?;
        self.disk.push(DiskRef {
            path,
            tokens: tokens.to_vec(),
            created,
        });
        self.sweep_disk();
        Ok(())
    }

    fn sweep_disk(&mut self) {
        let now = unix_secs();
        self.disk.retain(|item| {
            if now.saturating_sub(item.created) <= self.ttl_secs {
                return true;
            }
            let _ = fs::remove_file(&item.path);
            false
        });
        if self.disk.len() > DISK_FILES {
            self.disk.sort_by_key(|item| item.created);
            let drop_n = self.disk.len() - DISK_FILES;
            for item in self.disk.drain(..drop_n) {
                let _ = fs::remove_file(&item.path);
            }
        }
    }
}

fn best_slot(slots: &[Slot], prompt: &[u32], max_len: usize, gpu: bool) -> Option<usize> {
    slots
        .iter()
        .enumerate()
        .filter(|(_, slot)| {
            prefix_ok(&slot.tokens, prompt, max_len)
                && if gpu {
                    slot.gpu.is_some()
                } else {
                    slot.gpu.is_none() && !slot.host.is_empty()
                }
        })
        .max_by_key(|(_, slot)| slot.tokens.len())
        .map(|(i, _)| i)
}

fn prefix_ok(tokens: &[u32], prompt: &[u32], max_len: usize) -> bool {
    !tokens.is_empty() && tokens.len() <= max_len && prompt.len() >= tokens.len() && prompt.starts_with(tokens)
}

fn kv_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("PAGODA_KV_DIR") {
        return PathBuf::from(dir);
    }
    let mut path = PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()));
    path.push(".pagoda");
    path.push("kv");
    path
}

fn load_disk(dir: &Path, ttl_secs: u64, layers: usize) -> Vec<DiskRef> {
    let now = unix_secs();
    let mut found = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return found,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("kv") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        match parse_disk_header(&bytes, layers) {
            Some((created, tokens)) if now.saturating_sub(created) <= ttl_secs => {
                found.push(DiskRef { path, tokens, created });
            }
            _ => {
                let _ = fs::remove_file(&path);
            }
        }
    }
    found
}

fn parse_disk_header(bytes: &[u8], layers: usize) -> Option<(u64, Vec<u32>)> {
    if bytes.len() < 20 || &bytes[..4] != b"PGK1" {
        return None;
    }
    let created = u64::from_le_bytes(bytes[4..12].try_into().ok()?);
    let file_layers = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    if file_layers != layers {
        return None;
    }
    let n = u32::from_le_bytes(bytes[16..20].try_into().ok()?) as usize;
    if bytes.len() < 20 + n * 4 {
        return None;
    }
    let mut tokens = Vec::with_capacity(n);
    for i in 0..n {
        let at = 20 + i * 4;
        tokens.push(u32::from_le_bytes(bytes[at..at + 4].try_into().ok()?));
    }
    Some((created, tokens))
}

fn strip_disk_header(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.len() < 20 || &bytes[..4] != b"PGK1" {
        return None;
    }
    let n = u32::from_le_bytes(bytes[16..20].try_into().ok()?) as usize;
    let at = 20 + n * 4;
    bytes.get(at..)
}

fn token_hash(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for token in tokens {
        hash ^= *token as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_header_roundtrip_keeps_tokens() {
        let tokens = vec![1, 2, 3, 9];
        let created = 1_700_000_000u64;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"PGK1");
        bytes.extend_from_slice(&created.to_le_bytes());
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for token in &tokens {
            bytes.extend_from_slice(&u32::to_le_bytes(*token));
        }
        bytes.extend_from_slice(&[9, 9, 9]);
        let (got_created, got_tokens) = parse_disk_header(&bytes, 64).unwrap();
        assert_eq!(got_created, created);
        assert_eq!(got_tokens, tokens);
        assert_eq!(strip_disk_header(&bytes), Some(&[9, 9, 9][..]));
        assert!(parse_disk_header(&bytes, 63).is_none());
    }
}
