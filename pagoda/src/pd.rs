// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Mooncake-style prefill/decode (PD) disaggregation substrate.
//!
//! Mooncake (Moonshot AI) splits serving into prefill workers that compute
//! prompt KV and decode workers that continue generation, connected by a
//! KV-cache-centric object store (Mooncake Store) plus a transfer engine that
//! moves KV blocks point-to-point. This module ports that architecture into
//! pagoda's zero-dependency world:
//!
//! * [`KvStore`] — the object-store abstraction (Mooncake Store equivalent).
//!   Keys are content hashes of the prompt token path (idempotent PUTs, same
//!   cache-key semantics as the radix cache); values are serialized
//!   [`PrefillBundle`]s. LRU eviction under a byte budget, like Mooncake's
//!   memory pool.
//! * [`LocalStore`] — in-process store for tests and single-node splits.
//! * [`HttpStore`] — TCP client for a standalone store daemon
//!   ([`run_store`], the `pagoda store` subcommand), the cross-process
//!   transfer path. HTTP/1.1 keeps the runtime dependency-free; a production
//!   deployment swaps in RDMA behind the same trait.
//! * The engine-side halves live in [`crate::engine`]: `prefill_only`
//!   materializes a prompt and pushes its KV bundle into the store;
//!   `decode_from_kv` pulls a bundle and decodes without re-prefill.

use std::collections::HashMap;
use std::io;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::http_client;
use crate::json::{self, Value};
use crate::server::{read_http_request, write_http_response, HttpResponse};
use crate::spec::{GenerationOutput, TokenId};

/// Serving role of a pagoda process under PD disaggregation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PdRole {
    /// Prefill and decode in the same engine (the default, pre-PD behavior).
    Unified,
    /// Computes prompt KV and publishes it to the store; does not decode.
    Prefill,
    /// Pulls prompt KV from the store and only runs decode.
    Decode,
}

impl PdRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            PdRole::Unified => "unified",
            PdRole::Prefill => "prefill",
            PdRole::Decode => "decode",
        }
    }
}

/// What a prefill worker hands back after materializing a prompt.
#[derive(Clone, Debug)]
pub struct PrefillReceipt {
    /// Store object key the KV bundle was published under.
    pub kv_key: String,
    /// Total prompt tokens.
    pub prompt_tokens: usize,
    /// Prompt tokens actually materialized here (after local prefix hits).
    pub prefill_tokens: usize,
    /// Serialized bundle size in bytes (transfer-engine payload volume).
    pub kv_bytes: usize,
    /// The bundle already existed in the store (cross-request KV reuse —
    /// the prefill was published earlier by another request).
    pub store_hit: bool,
}

/// Failure modes of the PD path.
#[derive(Debug)]
pub enum PdError {
    /// Store unreachable / malformed response.
    Store(String),
    /// No bundle under the requested key (evicted or never prefilled).
    KvMiss(String),
    /// The stored payload did not parse as a [`PrefillBundle`].
    BadBundle(String),
    /// Admission rejected the request before prefill.
    Rejected(Box<GenerationOutput>),
}

impl std::fmt::Display for PdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdError::Store(e) => write!(f, "kv store error: {e}"),
            PdError::KvMiss(k) => write!(f, "no kv bundle under key {k:?}"),
            PdError::BadBundle(k) => write!(f, "corrupt kv bundle under key {k:?}"),
            PdError::Rejected(out) => write!(f, "rejected: {}", out.finish_reason),
        }
    }
}

impl std::error::Error for PdError {}

/// The payload a prefill worker publishes: everything a decode worker needs
/// to continue without re-prefill.
///
/// In pagoda's reference KV cache the physical page content *is* the token
/// path (see `kv_cache.rs`), so `prompt_tokens` doubles as the logical KV
/// image. Backends with real tensor KV additionally fill `kv` via
/// `ModelSession::export_kv`; `None` keeps the stateless-replay contract.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillBundle {
    pub prompt_tokens: Vec<TokenId>,
    pub kv: Option<Vec<u8>>,
}

/// Standard base64 (RFC 4648, with padding) — tensor KV payloads are
/// megabytes; a JSON array-of-bytes inflates them ~4x and costs seconds to
/// parse, so bundles carry the bytes as one base64 string instead.
const B64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64_ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(B64_ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            B64_ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            B64_ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn b64_decode(input: &str) -> Option<Vec<u8>> {
    if input.len() % 4 != 0 {
        return None;
    }
    let table: &[u8; 128] = &{
        let mut t = [0xffu8; 128];
        let mut i = 0;
        while i < 64 {
            t[B64_ALPHABET[i] as usize] = i as u8;
            i += 1;
        }
        t
    };
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    for chunk in input.as_bytes().chunks(4) {
        let pad = chunk.iter().filter(|&&c| c == b'=').count();
        if pad > 2 {
            return None;
        }
        let mut n: u32 = 0;
        for (i, &c) in chunk.iter().enumerate() {
            if c == b'=' {
                if i < 4 - pad {
                    return None;
                }
                n <<= 6;
            } else {
                let v = *table.get(c as usize)?;
                if v == 0xff {
                    return None;
                }
                n = (n << 6) | v as u32;
            }
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Some(out)
}

impl PrefillBundle {
    pub fn to_json(&self) -> String {
        let tokens = self
            .prompt_tokens
            .iter()
            .map(|&t| Value::Number(t as f64))
            .collect();
        let kv = match &self.kv {
            Some(bytes) => Value::String(b64_encode(bytes)),
            None => Value::Null,
        };
        Value::Object(vec![
            ("version".to_string(), Value::Number(2.0)),
            ("prompt_tokens".to_string(), Value::Array(tokens)),
            ("kv".to_string(), kv),
        ])
        .to_json()
    }

    pub fn parse(input: &str) -> Option<Self> {
        let v = json::parse(input).ok()?;
        let tokens: Vec<TokenId> = match v.get("prompt_tokens") {
            Some(Value::Array(items)) => items
                .iter()
                .map(|i| i.as_usize().map(|n| n as TokenId))
                .collect::<Option<Vec<_>>>()?,
            _ => return None,
        };
        let kv = match v.get("kv") {
            Some(Value::String(s)) => Some(b64_decode(s)?),
            _ => None,
        };
        Some(Self {
            prompt_tokens: tokens,
            kv,
        })
    }
}

/// Content-addressed store key for a prompt's KV bundle (FNV-1a over the
/// token path). Same prompt → same key → idempotent PUT, mirroring how
/// Mooncake keys KV blocks by hash.
pub fn bundle_key(tokens: &[TokenId]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("kv-{hash:016x}")
}

/// The KV object store abstraction (Mooncake Store equivalent).
pub trait KvStore: Send + Sync {
    fn put(&self, key: &str, value: &str) -> io::Result<()>;
    fn get(&self, key: &str) -> io::Result<Option<String>>;
    fn delete(&self, key: &str) -> io::Result<bool>;
}

/// Store counters, surfaced by `GET /store/stats` and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct StoreStats {
    pub entries: usize,
    pub bytes: usize,
    pub puts: u64,
    pub gets: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Entries dropped because their TTL expired (lazy, on access/insert).
    pub expired: u64,
}

impl StoreStats {
    fn to_json(&self) -> String {
        Value::Object(vec![
            ("entries".to_string(), Value::Number(self.entries as f64)),
            ("bytes".to_string(), Value::Number(self.bytes as f64)),
            ("puts".to_string(), Value::Number(self.puts as f64)),
            ("gets".to_string(), Value::Number(self.gets as f64)),
            ("hits".to_string(), Value::Number(self.hits as f64)),
            ("misses".to_string(), Value::Number(self.misses as f64)),
            (
                "evictions".to_string(),
                Value::Number(self.evictions as f64),
            ),
            ("expired".to_string(), Value::Number(self.expired as f64)),
        ])
        .to_json()
    }
}

/// Capacity-budgeted LRU object table shared by [`LocalStore`] and the
/// standalone store daemon. Entries are `(value, lru tick, inserted-at)`;
/// `max_age` (when set) expires entries lazily on access and insert, like
/// Mooncake Store's TTL on KV blocks whose owner has gone away.
struct StoreCore {
    map: HashMap<String, (String, u64, Instant)>,
    bytes: usize,
    max_bytes: usize,
    max_age: Option<Duration>,
    tick: u64,
    stats: StoreStats,
}

impl StoreCore {
    fn new(max_bytes: usize, max_age: Option<Duration>) -> Self {
        Self {
            map: HashMap::new(),
            bytes: 0,
            max_bytes: max_bytes.max(1),
            max_age,
            tick: 0,
            stats: StoreStats::default(),
        }
    }

    /// Drop every entry older than `max_age`; returns how many expired.
    fn prune_expired(&mut self) -> usize {
        let Some(max_age) = self.max_age else {
            return 0;
        };
        let now = Instant::now();
        let victims: Vec<String> = self
            .map
            .iter()
            .filter(|(_, (_, _, at))| now.duration_since(*at) > max_age)
            .map(|(k, _)| k.clone())
            .collect();
        let n = victims.len();
        for key in victims {
            let (value, _, _) = self.map.remove(&key).expect("victim exists");
            self.bytes -= value.len();
        }
        if n > 0 {
            self.stats.expired += n as u64;
            self.stats.entries = self.map.len();
            self.stats.bytes = self.bytes;
        }
        n
    }

    /// True when `key` holds a live (unexpired) entry.
    fn is_live(&self, key: &str) -> bool {
        match (self.max_age, self.map.get(key)) {
            (Some(max_age), Some((_, _, at))) => at.elapsed() <= max_age,
            (_, Some(_)) => true,
            _ => false,
        }
    }

    fn put(&mut self, key: &str, value: &str) {
        self.prune_expired();
        self.tick += 1;
        self.stats.puts += 1;
        if let Some((old, _, _)) = self.map.remove(key) {
            self.bytes -= old.len();
        }
        self.bytes += value.len();
        self.map
            .insert(key.to_string(), (value.to_string(), self.tick, Instant::now()));
        while self.bytes > self.max_bytes {
            let victim = self
                .map
                .iter()
                .min_by_key(|(_, (_, tick, _))| *tick)
                .map(|(k, _)| k.clone());
            let Some(victim) = victim else {
                break;
            };
            let (value, _, _) = self.map.remove(&victim).expect("victim exists");
            self.bytes -= value.len();
            self.stats.evictions += 1;
        }
        self.stats.entries = self.map.len();
        self.stats.bytes = self.bytes;
    }

    fn get(&mut self, key: &str) -> Option<String> {
        self.stats.gets += 1;
        self.tick += 1;
        if !self.is_live(key) {
            // Expired (or absent): drop the stale entry so `bytes` stays
            // honest, then report a miss.
            if let Some((value, _, _)) = self.map.remove(key) {
                self.bytes -= value.len();
                self.stats.expired += 1;
                self.stats.entries = self.map.len();
                self.stats.bytes = self.bytes;
            }
            self.stats.misses += 1;
            return None;
        }
        match self.map.get_mut(key) {
            Some((value, tick, _)) => {
                *tick = self.tick;
                self.stats.hits += 1;
                Some(value.clone())
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    fn delete(&mut self, key: &str) -> bool {
        match self.map.remove(key) {
            Some((value, _, _)) => {
                self.bytes -= value.len();
                self.stats.entries = self.map.len();
                self.stats.bytes = self.bytes;
                true
            }
            None => false,
        }
    }
}

/// In-process store: same-node PD splits and tests.
pub struct LocalStore {
    core: Mutex<StoreCore>,
}

impl LocalStore {
    pub fn new(max_bytes: usize) -> Self {
        Self::with_max_age(max_bytes, None)
    }

    /// [`LocalStore`] whose entries expire `max_age` after insertion.
    pub fn with_max_age(max_bytes: usize, max_age: Option<Duration>) -> Self {
        Self {
            core: Mutex::new(StoreCore::new(max_bytes, max_age)),
        }
    }

    pub fn stats(&self) -> StoreStats {
        self.core.lock().unwrap().stats
    }
}

impl Default for LocalStore {
    fn default() -> Self {
        Self::new(64 << 20)
    }
}

impl KvStore for LocalStore {
    fn put(&self, key: &str, value: &str) -> io::Result<()> {
        self.core.lock().unwrap().put(key, value);
        Ok(())
    }

    fn get(&self, key: &str) -> io::Result<Option<String>> {
        Ok(self.core.lock().unwrap().get(key))
    }

    fn delete(&self, key: &str) -> io::Result<bool> {
        Ok(self.core.lock().unwrap().delete(key))
    }
}

/// TCP client for the standalone store daemon ([`run_store`]): the
/// cross-process transfer path between prefill and decode workers.
pub struct HttpStore {
    addr: String,
    timeout: Duration,
}

impl HttpStore {
    /// `url` is `http://host[:port]` of a `pagoda store` daemon.
    pub fn from_url(url: &str) -> Option<Self> {
        Some(Self {
            addr: http_client::parse_http_addr(url)?,
            timeout: http_client::DEFAULT_TIMEOUT,
        })
    }
}

impl KvStore for HttpStore {
    fn put(&self, key: &str, value: &str) -> io::Result<()> {
        let path = format!("/kv/{key}");
        let (status, _) = http_client::request(&self.addr, "PUT", &path, Some(value), self.timeout)?;
        if status == 200 {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("kv store PUT {key} -> {status}"),
            ))
        }
    }

    fn get(&self, key: &str) -> io::Result<Option<String>> {
        let path = format!("/kv/{key}");
        let (status, body) = http_client::request(&self.addr, "GET", &path, None, self.timeout)?;
        match status {
            200 => Ok(Some(body)),
            404 => Ok(None),
            other => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("kv store GET {key} -> {other}"),
            )),
        }
    }

    fn delete(&self, key: &str) -> io::Result<bool> {
        let path = format!("/kv/{key}");
        let (status, _) = http_client::request(&self.addr, "DELETE", &path, None, self.timeout)?;
        Ok(status == 200)
    }
}

/// Run the standalone KV store daemon (the `pagoda store` subcommand):
/// `PUT /kv/<key>` publish, `GET /kv/<key>` pull, `DELETE /kv/<key>`,
/// `GET /store/stats`. One thread per connection, like the main server.
/// `max_age` (when set) expires entries lazily, so a decode worker that
/// never picks up its bundle stops pinning store capacity forever.
pub fn run_store(addr: &str, max_bytes: usize, max_age: Option<Duration>) -> io::Result<()> {
    let core = Arc::new(Mutex::new(StoreCore::new(max_bytes, max_age)));
    let listener = TcpListener::bind(addr)?;
    eprintln!(
        "pagoda kv store on http://{} (budget {} bytes{})",
        listener.local_addr()?,
        max_bytes,
        match max_age {
            Some(ttl) => format!(", ttl {}s", ttl.as_secs_f64()),
            None => String::new(),
        }
    );
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let core = Arc::clone(&core);
                std::thread::spawn(move || {
                    let _ = stream.set_nodelay(true);
                    let parsed = read_http_request(&mut stream);
                    let resp = match parsed {
                        Ok((method, path, body)) => store_handle(&core, &method, &path, &body),
                        Err(_) => HttpResponse::json(400, r#"{"error":"bad request"}"#),
                    };
                    let _ = write_http_response(&mut stream, &resp);
                });
            }
            Err(e) => eprintln!("store accept error: {e}"),
        }
    }
    Ok(())
}

fn store_handle(core: &Mutex<StoreCore>, method: &str, path: &str, body: &str) -> HttpResponse {
    if method == "GET" && path == "/store/stats" {
        return HttpResponse::json(200, &core.lock().unwrap().stats.to_json());
    }
    if method == "GET" && path == "/health" {
        return HttpResponse::json(200, r#"{"status":"ok"}"#);
    }
    let Some(key) = path.strip_prefix("/kv/") else {
        return HttpResponse::json(404, r#"{"error":"not found"}"#);
    };
    if key.is_empty() || key.contains(['/', '?', '#']) {
        return HttpResponse::json(400, r#"{"error":"bad key"}"#);
    }
    match method {
        "PUT" => {
            core.lock().unwrap().put(key, body);
            HttpResponse::json(200, r#"{"ok":true}"#)
        }
        "GET" => match core.lock().unwrap().get(key) {
            Some(value) => HttpResponse::json(200, &value),
            None => HttpResponse::json(404, r#"{"error":"miss"}"#),
        },
        "DELETE" => {
            let deleted = core.lock().unwrap().delete(key);
            HttpResponse::json(
                200,
                &Value::Object(vec![("deleted".to_string(), Value::Bool(deleted))]).to_json(),
            )
        }
        _ => HttpResponse::json(400, r#"{"error":"bad method"}"#),
    }
}

/// Prefix-affinity router across several decode workers (Mooncake's
/// conductor role): each prompt goes to the decode worker that has served the
/// longest matching prompt prefix before, so the worker's local radix cache
/// pays off. Affinity is tracked in a character trie over prompt text with a
/// per-node worker set; ties and cold paths break round-robin.
pub struct PrefixRouter {
    addrs: Vec<String>,
    urls: Vec<String>,
    trie: Vec<TrieNode>,
    /// Round-robin cursor among prefix-affined candidates.
    rr: usize,
    /// Round-robin cursor for cold prompts (no prefix evidence), kept
    /// separate so affinity picks never disturb the spreading sequence.
    rr_cold: usize,
    routed: Vec<u64>,
}

struct TrieNode {
    children: HashMap<char, usize>,
    workers: Vec<bool>,
}

impl TrieNode {
    fn new(n_workers: usize) -> Self {
        Self {
            children: HashMap::new(),
            workers: vec![false; n_workers],
        }
    }
}

impl PrefixRouter {
    /// `decode_urls` are `http://host[:port]` URLs of decode workers.
    pub fn new(decode_urls: &[String]) -> Option<Self> {
        if decode_urls.is_empty() {
            return None;
        }
        let mut addrs = Vec::with_capacity(decode_urls.len());
        for url in decode_urls {
            addrs.push(http_client::parse_http_addr(url)?);
        }
        let n = decode_urls.len();
        Some(Self {
            addrs,
            urls: decode_urls.to_vec(),
            trie: vec![TrieNode::new(n)],
            rr: 0,
            rr_cold: 0,
            routed: vec![0; n],
        })
    }

    /// Worker dial address (`host:port`) by index.
    pub fn addr(&self, idx: usize) -> &str {
        &self.addrs[idx]
    }

    /// Worker URLs as configured.
    pub fn urls(&self) -> &[String] {
        &self.urls
    }

    /// Per-worker routed request counts.
    pub fn routed_counts(&self) -> &[u64] {
        &self.routed
    }

    /// Decode worker addr for `prompt`: longest-prefix affinity first,
    /// round-robin among candidates and on cold paths.
    pub fn pick(&mut self, prompt: &str) -> usize {
        let mut node = 0usize;
        let mut deepest = 0usize;
        for ch in prompt.chars() {
            match self.trie[node].children.get(&ch) {
                Some(&next) => {
                    node = next;
                    if self.trie[node].workers.iter().any(|&w| w) {
                        deepest = node;
                    }
                }
                None => break,
            }
        }
        // Prefix evidence only counts below the root: treating the root's
        // warm workers as candidates would pin every cold prompt onto the
        // first-used worker instead of spreading across the pool.
        let candidates: Vec<usize> = if deepest == 0 {
            Vec::new()
        } else {
            self.trie[deepest]
                .workers
                .iter()
                .enumerate()
                .filter_map(|(i, &w)| w.then_some(i))
                .collect()
        };
        let n = self.addrs.len();
        let idx = match candidates.len() {
            0 => {
                self.rr_cold = (self.rr_cold + 1) % n;
                self.rr_cold
            }
            k => {
                self.rr = (self.rr + 1) % k;
                candidates[self.rr]
            }
        };
        idx
    }

    /// Remember that `worker` served `prompt` (called after a 200).
    pub fn record(&mut self, prompt: &str, worker: usize) {
        let mut node = 0usize;
        for ch in prompt.chars() {
            let next = match self.trie[node].children.get(&ch) {
                Some(&next) => next,
                None => {
                    let next = self.trie.len();
                    self.trie.push(TrieNode::new(self.addrs.len()));
                    self.trie[node].children.insert(ch, next);
                    next
                }
            };
            node = next;
            self.trie[node].workers[worker] = true;
        }
        self.routed[worker] += 1;
    }

    pub fn stats_json(&self) -> String {
        let workers = self
            .urls
            .iter()
            .zip(&self.routed)
            .map(|(url, &n)| {
                Value::Object(vec![
                    ("url".to_string(), Value::String(url.clone())),
                    ("routed".to_string(), Value::Number(n as f64)),
                ])
            })
            .collect();
        Value::Object(vec![
            ("decode_workers".to_string(), Value::Array(workers)),
            (
                "affinity_nodes".to_string(),
                Value::Number(self.trie.len() as f64),
            ),
        ])
        .to_json()
    }
}

/// Run the PD conductor router: plain-text `/generate` is prefilled at
/// `prefill_url`, then decoded on the prefix-affine decode worker; the
/// decode response is relayed verbatim (SSE included). `/health` and
/// `/route/stats` stay local. `triage` (when set) gates `/generate`
/// through Laya before any prefill compute is spent.
pub fn run_router(
    addr: &str,
    prefill_url: &str,
    decode_urls: &[String],
    triage: Option<Arc<crate::triage::Triage>>,
) -> io::Result<()> {
    let Some(prefill_addr) = http_client::parse_http_addr(prefill_url) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("bad prefill url {prefill_url:?}"),
        ));
    };
    let Some(router) = PrefixRouter::new(decode_urls) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "need at least one valid --decode-url",
        ));
    };
    let router = Arc::new(Mutex::new(router));
    let listener = TcpListener::bind(addr)?;
    eprintln!(
        "pagoda pd router on http://{} (prefill: {}, decode workers: {}{})",
        listener.local_addr()?,
        prefill_url,
        decode_urls.len(),
        match &triage {
            Some(t) => format!(", triage: laya at {}", t.url()),
            None => String::new(),
        }
    );
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let router = Arc::clone(&router);
                let prefill_addr = prefill_addr.clone();
                let triage = triage.clone();
                std::thread::spawn(move || {
                    let _ = stream.set_nodelay(true);
                    let parsed = read_http_request(&mut stream);
                    let result = match parsed {
                        Ok((method, path, body)) => router_handle(
                            &router,
                            &prefill_addr,
                            triage.as_deref(),
                            &method,
                            &path,
                            &body,
                            &mut stream,
                        ),
                        Err(_) => write_http_response(
                            &mut stream,
                            &HttpResponse::json(400, r#"{"error":"bad request"}"#),
                        ),
                    };
                    let _ = result;
                });
            }
            Err(e) => eprintln!("router accept error: {e}"),
        }
    }
    Ok(())
}

fn router_handle(
    router: &Mutex<PrefixRouter>,
    prefill_addr: &str,
    triage: Option<&crate::triage::Triage>,
    method: &str,
    path: &str,
    body: &str,
    stream: &mut std::net::TcpStream,
) -> io::Result<()> {
    use std::io::Write;
    match (method, path) {
        ("GET", "/health") => {
            write_http_response(stream, &HttpResponse::json(200, r#"{"status":"ok"}"#))
        }
        ("GET", "/route/stats") => {
            let body = router.lock().unwrap().stats_json();
            write_http_response(stream, &HttpResponse::json(200, &body))
        }
        ("POST", "/generate") => {
            let v = match json::parse(body) {
                Ok(v) => v,
                Err(_) => {
                    return write_http_response(
                        stream,
                        &HttpResponse::json(400, r#"{"error":"bad json"}"#),
                    )
                }
            };
            let Some(text) = v.get("text").and_then(Value::as_str) else {
                return write_http_response(
                    stream,
                    &HttpResponse::json(400, r#"{"error":"expected {\"text\": ...}"}"#),
                );
            };
            let stream_wanted = v.get("stream").and_then(Value::as_bool).unwrap_or(false);

            // 0. Laya System-1 gate: escalate before spending any prefill
            //    compute (fail-open when Laya is unreachable).
            if let Some(t) = triage {
                let out = t.check(text);
                if out.escalate {
                    return write_http_response(
                        stream,
                        &HttpResponse::json(200, &t.escalation_body(&out)),
                    );
                }
            }

            // 1. Prefill on the prefill worker (verbatim body carries text +
            //    sampling_params).
            let prefill =
                http_client::request(prefill_addr, "POST", "/prefill", Some(body), http_client::DEFAULT_TIMEOUT);
            let (status, resp) = match prefill {
                Ok(pair) => pair,
                Err(e) => {
                    return write_http_response(
                        stream,
                        &HttpResponse::json(
                            502,
                            &format!(r#"{{"error":"prefill_unreachable","detail":"{e}"}}"#),
                        ),
                    )
                }
            };
            if status != 200 {
                return write_http_response(
                    stream,
                    &HttpResponse::json(
                        502,
                        &format!(r#"{{"error":"prefill_failed","status":{status},"detail":{resp:?}}}"#),
                    ),
                );
            }
            let key = json::parse(&resp)
                .ok()
                .and_then(|v| v.get("kv_key").and_then(Value::as_str).map(str::to_string));
            let Some(key) = key else {
                return write_http_response(
                    stream,
                    &HttpResponse::json(502, r#"{"error":"prefill_failed","detail":"no kv_key"}"#),
                );
            };

            // 2. Pick the prefix-affine decode worker and hand it the bundle.
            let (worker, addr) = {
                let mut r = router.lock().unwrap();
                let w = r.pick(text);
                (w, r.addrs[w].clone())
            };
            let mut decode_body = format!(r#"{{"kv_key":"{key}""#);
            if let Some(sp) = v.get("sampling_params") {
                decode_body.push_str(&format!(r#","sampling_params":{}"#, sp.to_json()));
            }
            if stream_wanted {
                decode_body.push_str(r#","stream":true"#);
            }
            decode_body.push('}');

            // 3. Relay the decode response verbatim (SSE chunk-by-chunk).
            let up = http_client::request_open(
                &addr,
                "POST",
                "/generate",
                Some(&decode_body),
                http_client::DEFAULT_TIMEOUT,
            );
            let mut up = match up {
                Ok(up) => up,
                Err(e) => {
                    return write_http_response(
                        stream,
                        &HttpResponse::json(
                            502,
                            &format!(r#"{{"error":"decode_unreachable","detail":"{e}"}}"#),
                        ),
                    )
                }
            };
            if up.status == 200 {
                router.lock().unwrap().record(text, worker);
            }
            if !up.chunked {
                let mut buf = Vec::new();
                while let Some(payload) = up.next_payload()? {
                    buf.extend_from_slice(&payload);
                }
                return write_http_response(
                    stream,
                    &HttpResponse {
                        status: up.status,
                        body: String::from_utf8_lossy(&buf).into_owned(),
                    },
                );
            }
            let content_type = up.content_type.as_deref().unwrap_or("text/event-stream");
            let head = format!(
                "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nConnection: close\r\n\r\n",
                up.status, content_type
            );
            stream.write_all(head.as_bytes())?;
            while let Some(payload) = up.next_payload()? {
                stream.write_all(&payload)?;
            }
            Ok(())
        }
        _ => write_http_response(stream, &HttpResponse::json(404, r#"{"error":"not found"}"#)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn b64_codec_roundtrip() {
        for n in [0usize, 1, 2, 3, 4, 255, 256, 1024] {
            let bytes: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
            assert_eq!(b64_decode(&b64_encode(&bytes)).as_deref(), Some(bytes.as_slice()));
        }
        assert_eq!(b64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(b64_decode("aGVsbG8="), Some(b"hello".to_vec()));
        assert!(b64_decode("aGVsbG8").is_none(), "bad length");
        assert!(b64_decode("aGVsbG8\x01").is_none(), "bad char");
        // Wire size is ~4/3 of raw — the whole point of the base64 switch.
        let big: Vec<u8> = (0..3000u32).map(|i| (i % 251) as u8).collect();
        assert_eq!(b64_encode(&big).len(), 4000);
    }

    #[test]
    fn bundle_roundtrip() {
        let bundle = PrefillBundle {
            prompt_tokens: vec![1, 2, 300, u32::MAX],
            kv: Some(vec![0, 1, 255]),
        };
        let parsed = PrefillBundle::parse(&bundle.to_json()).unwrap();
        assert_eq!(parsed, bundle);
        let no_kv = PrefillBundle {
            prompt_tokens: vec![7],
            kv: None,
        };
        assert_eq!(PrefillBundle::parse(&no_kv.to_json()), Some(no_kv));
    }

    #[test]
    fn bundle_key_is_content_addressed() {
        assert_eq!(bundle_key(&[1, 2, 3]), bundle_key(&[1, 2, 3]));
        assert_ne!(bundle_key(&[1, 2, 3]), bundle_key(&[1, 2, 4]));
        assert!(bundle_key(&[]).starts_with("kv-"));
    }

    #[test]
    fn local_store_lru_eviction() {
        let store = LocalStore::new(8);
        store.put("a", "1234").unwrap();
        store.put("b", "5678").unwrap();
        // Touch `a` so `b` is the LRU victim.
        assert_eq!(store.get("a").unwrap().as_deref(), Some("1234"));
        store.put("c", "9999").unwrap();
        assert_eq!(store.get("a").unwrap().as_deref(), Some("1234"));
        assert_eq!(store.get("b").unwrap(), None);
        assert_eq!(store.get("c").unwrap().as_deref(), Some("9999"));
        assert!(store.stats().evictions >= 1);
    }
}
