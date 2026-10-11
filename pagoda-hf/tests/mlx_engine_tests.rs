// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Mock-upstream tests for the MLX HTTP ModelSession adapter.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use pagoda::model::{ModelEngine, ModelSession};
use pagoda::spec::{SamplingParams, WriteRequest};
use pagoda::{Engine, EngineConfig};
use pagoda_hf::mlx_engine::{MlxApi, MlxHttpEngine, MlxTokenCodec};
use pagoda_hf::HfTokenizer;

struct TinyCodec;
impl MlxTokenCodec for TinyCodec {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }
    fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|&t| t as u8 as char).collect()
    }
    fn vocab_size(&self) -> usize {
        256
    }
}

/// Tiny byte tokenizer matching TinyCodec so Engine encode/decode agree.
struct TinyTok;
impl pagoda::tokenizer::Tokenizer for TinyTok {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(|b| b as u32).collect()
    }
    fn decode(&self, tokens: &[u32]) -> String {
        tokens.iter().map(|&t| t as u8 as char).collect()
    }
    fn eos_token_id(&self) -> u32 {
        // Keep clear of printable ASCII so a failed/uniform logits argmax
        // (index 0) does not look like an immediate EOS.
        255
    }
    fn bos_token_id(&self) -> Option<u32> {
        None
    }
    fn vocab_size(&self) -> usize {
        256
    }
    fn name(&self) -> &'static str {
        "tiny"
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Mock that returns the next char in a fixed script for each /v1/completions call.
fn mock_completions(port: u16, script: &'static [u8], hits: Arc<AtomicUsize>) {
    thread::spawn(move || {
        let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind mock");
        listener
            .set_nonblocking(false)
            .ok();
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut buf = [0u8; 8192];
            let _ = stream.read(&mut buf);
            let i = hits.fetch_add(1, Ordering::SeqCst);
            let ch = script.get(i).copied().unwrap_or(b'.');
            let body = format!(
                r#"{{"choices":[{{"text":"{}","logprobs":{{"token_logprobs":[-0.01]}}}}],"usage":{{"prompt_tokens":{},"prompt_tokens_details":{{"cached_tokens":{}}}}}}}"#,
                ch as char,
                10 + i,
                if i > 0 { 8 } else { 0 }
            );
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    thread::sleep(Duration::from_millis(50));
}

#[test]
fn session_steps_and_forks_with_mock_upstream() {
    let port = free_port();
    let hits = Arc::new(AtomicUsize::new(0));
    mock_completions(port, b"XY", Arc::clone(&hits));

    let engine = MlxHttpEngine::new(
        &format!("http://127.0.0.1:{port}"),
        Arc::new(TinyCodec),
        MlxApi::Completions,
    )
    .unwrap()
    .with_timeout(Duration::from_secs(5));

    let mut s1 = engine.begin_session().expect("session");
    let logits = s1.forward(&[b'h' as u32, b'i' as u32]);
    assert!((logits[b'X' as usize] + 0.01).abs() < 1e-5);
    assert!(logits.iter().all(|x| x.is_finite()));
    assert!(logits[b'X' as usize] > logits[0]);

    let mut s2 = s1.fork().expect("fork");
    assert_eq!(s2.context_len(), 2);
    let logits2 = s2.forward(&[b'X' as u32]);
    assert!((logits2[b'Y' as usize] + 0.01).abs() < 1e-5);
    assert_eq!(s2.context_len(), 3);
    // Parent unchanged.
    assert_eq!(s1.context_len(), 2);

    assert!(engine.remote_steps() >= 2);
    assert!(engine.prompt_tokens() > 0);
}

#[test]
fn engine_generate_uses_mlx_session() {
    let port = free_port();
    let hits = Arc::new(AtomicUsize::new(0));
    // Produce "OK" then eos-ish zeros won't stop; use max_tokens=2.
    mock_completions(port, b"OK........", Arc::clone(&hits));

    let model = MlxHttpEngine::new(
        &format!("http://127.0.0.1:{port}"),
        Arc::new(TinyCodec),
        MlxApi::Completions,
    )
    .unwrap()
    .with_timeout(Duration::from_secs(5));

    let mut eng = Engine::new(
        TinyTok,
        model,
        EngineConfig {
            num_kv_blocks: 256,
            block_size: 16,
            max_running_requests: 4,
            max_total_tokens: 256,
            ..EngineConfig::default()
        },
    );
    let out = eng.generate(&WriteRequest::new(
        "hi",
        SamplingParams {
            max_tokens: 2,
            temperature: 0.0,
            ..SamplingParams::default()
        },
    ));
    assert!(
        out.rejection.is_none(),
        "rejected: {:?}",
        out.rejection
    );
    assert!(
        !out.text.is_empty() || !out.output_token_ids.is_empty(),
        "empty generation: text={:?} ids={:?} reason={:?} hits={}",
        out.text,
        out.output_token_ids,
        out.finish_reason,
        hits.load(Ordering::SeqCst)
    );
    assert!(hits.load(Ordering::SeqCst) >= 1);
}

#[test]
fn checkpoint_branch_uses_session_fork_path() {
    let port = free_port();
    let hits = Arc::new(AtomicUsize::new(0));
    mock_completions(port, b"ABCDEFGH", Arc::clone(&hits));
    let model = MlxHttpEngine::new(
        &format!("http://127.0.0.1:{port}"),
        Arc::new(TinyCodec),
        MlxApi::Completions,
    )
    .unwrap()
    .with_timeout(Duration::from_secs(5));
    let mut eng = Engine::new(
        TinyTok,
        model,
        EngineConfig {
            num_kv_blocks: 256,
            block_size: 16,
            max_running_requests: 4,
            max_total_tokens: 256,
            ..EngineConfig::default()
        },
    );
    let ckpt = eng.create_checkpoint("sys ");
    let out = eng
        .generate_from_checkpoint(
            ckpt,
            "q",
            SamplingParams {
                max_tokens: 2,
                temperature: 0.0,
                ..SamplingParams::default()
            },
        )
        .expect("branch");
    assert!(out.rejection.is_none(), "{:?}", out.rejection);
    assert!(
        !out.output_token_ids.is_empty() || !out.text.is_empty(),
        "empty branch {:?}",
        out
    );
    assert!(hits.load(Ordering::SeqCst) >= 1);
}

#[test]
fn usage_handles_extra_stats_shape() {
    let port = free_port();
    let hits = Arc::new(AtomicUsize::new(0));
    mock_completions(port, b"Z", Arc::clone(&hits));
    let eng = MlxHttpEngine::new(
        &format!("http://127.0.0.1:{port}"),
        Arc::new(TinyCodec),
        MlxApi::Completions,
    )
    .unwrap()
    .with_timeout(Duration::from_secs(5));
    let handles = eng.usage_handles();
    let mut s = eng.begin_session().unwrap();
    let _ = ModelSession::forward(s.as_mut(), &[b'a' as u32]);
    let fields = handles.extra_stats();
    let keys: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
    assert!(keys.contains(&"upstream_cached_tokens"));
    assert!(keys.contains(&"mlx_remote_steps"));
    assert!(handles.remote_steps() >= 1);
}

/// Smoke: local Qwen3.8 tokenizer.json loads into HfTokenizer when present.
#[test]
fn local_qwen38_tokenizer_loads_if_present() {
    let p = "/Users/zzz858aaa/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit/tokenizer.json";
    if !std::path::Path::new(p).exists() {
        return;
    }
    let tok = HfTokenizer::from_file(p).expect("load qwen38 tokenizer");
    assert!(pagoda::tokenizer::Tokenizer::vocab_size(&tok) > 1000);
}
