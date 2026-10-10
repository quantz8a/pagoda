// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! End-to-end PD disaggregation: Mooncake-style KV store + split
//! prefill/decode workers must reproduce the unified engine bit-for-bit.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use pagoda::pd::{self, KvStore, LocalStore, PrefillBundle};
use pagoda::server::{self, Conductor};
use pagoda::{EngineConfig, PdRole, SamplingParams, ToyEngine, WriteRequest};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn http(
    port: u16,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> std::io::Result<(u16, String)> {
    pagoda::http_client::request(
        &format!("127.0.0.1:{port}"),
        method,
        path,
        body,
        std::time::Duration::from_secs(10),
    )
}

fn wait_until_ready(port: u16) {
    for _ in 0..100 {
        if let Ok((200, _)) = http(port, "GET", "/health", None) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("server on {port} did not become ready");
}

fn params() -> SamplingParams {
    SamplingParams {
        max_tokens: 32,
        ..SamplingParams::default()
    }
}

/// The split prefill→store→decode path must produce exactly the unified
/// engine's output, with prefill compute charged only on the prefill side.
#[test]
fn pd_split_matches_unified() {
    let prompt = "SGLang uses a radix tree to cache";
    let baseline = {
        let mut engine = ToyEngine::toy(EngineConfig::default());
        engine.generate(&WriteRequest::new(prompt, params()))
    };

    let store = Arc::new(LocalStore::default());
    let mut prefill = ToyEngine::toy(EngineConfig::default());
    prefill.enable_pd(PdRole::Prefill, store.clone());
    let mut decode = ToyEngine::toy(EngineConfig::default());
    decode.enable_pd(PdRole::Decode, store.clone());

    let receipt = prefill
        .prefill_only(&WriteRequest::new(prompt, params()))
        .expect("prefill");
    assert_eq!(receipt.prompt_tokens, receipt.prefill_tokens);
    assert!(!receipt.store_hit, "first publish must miss");
    assert_eq!(receipt.kv_key, pd::bundle_key(&baseline_tokens(prompt)));

    let out = decode
        .decode_from_kv(&receipt.kv_key, params())
        .expect("decode");
    assert_eq!(out.text, baseline.text);
    assert_eq!(out.output_token_ids, baseline.output_token_ids);
    assert_eq!(out.finish_reason, baseline.finish_reason);
    assert_eq!(out.prefix_hit_tokens, baseline.prompt_tokens);

    let p_stats = prefill.stats();
    let d_stats = decode.stats();
    assert_eq!(p_stats.pd_prefill_requests, 1);
    assert_eq!(d_stats.pd_decode_requests, 1);
    assert_eq!(d_stats.total_prefill_tokens, 0, "decode worker must not prefill");
    assert!(d_stats.pd_kv_bytes > 0);
    assert_eq!(
        p_stats.total_prefill_tokens, baseline.prompt_tokens as u64,
        "prefill compute lives on the prefill worker"
    );

    // Idempotent re-publish: same prompt hashes to the same key.
    let receipt2 = prefill
        .prefill_only(&WriteRequest::new(prompt, params()))
        .expect("re-prefill");
    assert_eq!(receipt2.kv_key, receipt.kv_key);
    assert!(receipt2.store_hit);

    // Unknown keys miss cleanly.
    assert!(decode.decode_from_kv("kv-does-not-exist", params()).is_err());
}

fn baseline_tokens(text: &str) -> Vec<u32> {
    use pagoda::Tokenizer;
    pagoda::ByteTokenizer::new().encode(text)
}

/// The standalone HTTP store daemon round-trips bundles between processes.
#[test]
fn http_store_roundtrip() {
    let port = free_port();
    std::thread::spawn(move || pd::run_store(&format!("127.0.0.1:{port}"), 1 << 20, None).unwrap());
    wait_until_ready(port);

    let store = pd::HttpStore::from_url(&format!("http://127.0.0.1:{port}")).unwrap();
    let bundle = PrefillBundle {
        prompt_tokens: vec![10, 20, 30],
        kv: None,
    };
    let key = pd::bundle_key(&bundle.prompt_tokens);

    assert!(store.get(&key).unwrap().is_none(), "miss before put");
    store.put(&key, &bundle.to_json()).unwrap();
    let got = store.get(&key).unwrap().expect("hit after put");
    assert_eq!(PrefillBundle::parse(&got), Some(bundle));
    assert!(store.delete(&key).unwrap());
    assert!(store.get(&key).unwrap().is_none(), "gone after delete");

    let (_, stats) = http(port, "GET", "/store/stats", None).expect("http");
    assert!(stats.contains("\"puts\":1"), "stats: {stats}");
    assert!(stats.contains("\"hits\":1"), "stats: {stats}");
}

/// Full three-process topology: store daemon + prefill worker + decode worker
/// with a conductor, driven over HTTP from the client side.
#[test]
fn pd_conductor_e2e() {
    let store_port = free_port();
    let store_url = format!("http://127.0.0.1:{store_port}");
    std::thread::spawn(move || pd::run_store(&format!("127.0.0.1:{store_port}"), 1 << 20, None).unwrap());
    wait_until_ready(store_port);

    let mut prefill_engine = ToyEngine::toy(EngineConfig::default());
    prefill_engine.enable_pd(
        PdRole::Prefill,
        Arc::new(pd::HttpStore::from_url(&store_url).unwrap()),
    );
    let prefill_port = free_port();
    let prefill_addr = format!("127.0.0.1:{prefill_port}");
    std::thread::spawn(move || {
        server::run_full(
            Arc::new(Mutex::new(prefill_engine)),
            &prefill_addr,
            None,
            None,
            None,
        )
        .unwrap();
    });
    wait_until_ready(prefill_port);

    let mut decode_engine = ToyEngine::toy(EngineConfig::default());
    decode_engine.enable_pd(
        PdRole::Decode,
        Arc::new(pd::HttpStore::from_url(&store_url).unwrap()),
    );
    let decode_port = free_port();
    let decode_addr = format!("127.0.0.1:{decode_port}");
    let conductor = Conductor::from_url(&format!("http://127.0.0.1:{prefill_port}")).unwrap();
    std::thread::spawn(move || {
        server::run_full(
            Arc::new(Mutex::new(decode_engine)),
            &decode_addr,
            None,
            None,
            Some(Arc::new(conductor)),
        )
        .unwrap();
    });
    wait_until_ready(decode_port);

    let prompt = "The scheduler runs continuous batching to merge";
    let baseline = ToyEngine::toy(EngineConfig::default())
        .generate(&WriteRequest::new(prompt, params()));

    // 1. Client-driven split: /prefill then /generate with kv_key.
    let (status, receipt) = http(
        prefill_port,
        "POST",
        "/prefill",
        Some(&format!(
            r#"{{"text":{:?},"sampling_params":{{"max_tokens":32}}}}"#,
            prompt
        )),
    )
    .expect("http");
    assert_eq!(status, 200, "prefill: {receipt}");
    let kv_key = extract(&receipt, "kv_key");
    let (status, body) = http(
        decode_port,
        "POST",
        "/generate",
        Some(&format!(
            r#"{{"kv_key":"{kv_key}","sampling_params":{{"max_tokens":32,"seed":42}}}}"#
        )),
    )
    .expect("http");
    assert_eq!(status, 200, "decode: {body}");
    assert!(
        body.contains(&json_string(&baseline.text)),
        "decode must equal unified baseline\nbaseline: {:?}\ngot: {body}",
        baseline.text
    );

    // 2. Conductor path: plain text into the decode worker.
    let (status, body) = http(
        decode_port,
        "POST",
        "/generate",
        Some(&format!(
            r#"{{"text":{:?},"sampling_params":{{"max_tokens":32,"seed":42}}}}"#,
            prompt
        )),
    )
    .expect("http");
    assert_eq!(status, 200, "conduct: {body}");
    assert!(
        body.contains(&json_string(&baseline.text)),
        "conducted output must equal unified baseline\ngot: {body}"
    );

    // 3. Role discipline and observability.
    let (status, _) = http(prefill_port, "POST", "/generate", Some(r#"{"text":"hi"}"#)).expect("http");
    assert_eq!(status, 400, "prefill worker must refuse decode");
    let (status, body) = http(
        decode_port,
        "POST",
        "/generate",
            Some(r#"{"kv_key":"kv-missing"}"#),
    )
    .expect("http");
    assert_eq!(status, 404, "unknown key must miss: {body}");
    let (_, stats) = http(decode_port, "GET", "/stats", None).expect("http");
    assert!(stats.contains("\"pd_role\":\"decode\""), "stats: {stats}");
    assert!(stats.contains("\"pd_decode_requests\":2"), "stats: {stats}");
    let (_, stats) = http(prefill_port, "GET", "/stats", None).expect("http");
    assert!(stats.contains("\"pd_prefill_requests\":2"), "stats: {stats}");
}

fn extract(json: &str, key: &str) -> String {
    pagoda::json::parse(json)
        .ok()
        .and_then(|v| v.get(key).and_then(|k| k.as_str().map(str::to_string)))
        .unwrap_or_else(|| panic!("no {key} in {json}"))
}

/// The JSON-escaped form of `text` as it appears inside a response body
/// (without the surrounding quotes).
fn json_string(text: &str) -> String {
    let quoted = format!("{text:?}");
    quoted[1..quoted.len() - 1].to_string()
}

/// Streaming decode fires one sink event per sampled token, in order, and the
/// concatenated pieces equal the buffered output.
#[test]
fn pd_stream_decode_emits_tokens_in_order() {
    let prompt = "The scheduler runs continuous batching to merge";
    let baseline = ToyEngine::toy(EngineConfig::default())
        .generate(&WriteRequest::new(prompt, params()));

    let store = Arc::new(LocalStore::default());
    let mut prefill = ToyEngine::toy(EngineConfig::default());
    prefill.enable_pd(PdRole::Prefill, store.clone());
    let mut decode = ToyEngine::toy(EngineConfig::default());
    decode.enable_pd(PdRole::Decode, store.clone());

    let receipt = prefill
        .prefill_only(&WriteRequest::new(prompt, params()))
        .expect("prefill");

    let mut pieces: Vec<String> = Vec::new();
    let out = decode
        .decode_from_kv_streaming(&receipt.kv_key, params(), &mut |_id, piece| {
            pieces.push(piece.to_string());
        })
        .expect("stream decode");
    assert_eq!(pieces.concat(), baseline.text);
    assert_eq!(pieces.len(), baseline.output_token_ids.len());
    assert_eq!(out.text, baseline.text);
}

/// Prefix-affinity routing: cold prompts spread round-robin; a served prompt
/// (and any extension of it) sticks to the worker that saw it.
#[test]
fn router_prefix_affinity() {
    let workers = vec![
        "http://127.0.0.1:1".to_string(),
        "http://127.0.0.1:2".to_string(),
    ];
    let mut router = pd::PrefixRouter::new(&workers).unwrap();
    let first = router.pick("hello world");
    router.record("hello world", first);
    // Same prompt and a prefix-extension stick to the recorded worker.
    assert_eq!(router.pick("hello world"), first);
    assert_eq!(router.pick("hello world, again"), first);
    // An unrelated prompt keeps spreading.
    let other = router.pick("zzz unrelated");
    router.record("zzz unrelated", other);
    assert_eq!(router.pick("zzz unrelated"), other);
    let stats = pagoda::json::parse(&router.stats_json()).unwrap();
    let Some(pagoda::json::Value::Array(workers)) = stats.get("decode_workers") else {
        panic!("stats shape: {stats:?}")
    };
    let total: f64 = workers
        .iter()
        .filter_map(|w| w.get("routed").and_then(|n| n.as_f64()))
        .sum();
    assert_eq!(total, 2.0, "two recorded routes: {stats:?}");
}

/// Raw HTTP/1.1 request returning the full wire response (for SSE reading).
fn raw_http(port: u16, path: &str, body: &str) -> String {
    use std::io::{Read, Write};
    let mut s = std::net::TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).unwrap();
    String::from_utf8_lossy(&buf).into_owned()
}

/// SSE over the wire: token frames arrive in order, followed by the final
/// usage frame and [DONE]; joined tokens equal the unified baseline.
#[test]
fn pd_stream_http_e2e() {
    let store_port = free_port();
    let store_url = format!("http://127.0.0.1:{store_port}");
    std::thread::spawn(move || {
        pd::run_store(&format!("127.0.0.1:{store_port}"), 1 << 20, None).unwrap()
    });
    wait_until_ready(store_port);

    let mut decode_engine = ToyEngine::toy(EngineConfig::default());
    decode_engine.enable_pd(
        PdRole::Decode,
        Arc::new(pd::HttpStore::from_url(&store_url).unwrap()),
    );
    let decode_port = free_port();
    let decode_addr = format!("127.0.0.1:{decode_port}");
    std::thread::spawn(move || {
        server::run_full(
            Arc::new(Mutex::new(decode_engine)),
            &decode_addr,
            None,
            None,
            None,
        )
        .unwrap();
    });
    wait_until_ready(decode_port);

    // Publish a bundle directly (no prefill worker needed for this test).
    let prompt = "SGLang uses a radix tree to cache";
    let baseline = ToyEngine::toy(EngineConfig::default())
        .generate(&WriteRequest::new(prompt, params()));
    let tokens = baseline_tokens(prompt);
    let key = pd::bundle_key(&tokens);
    let store = pd::HttpStore::from_url(&store_url).unwrap();
    store
        .put(
            &key,
            &PrefillBundle {
                prompt_tokens: tokens,
                kv: None,
            }
            .to_json(),
        )
        .unwrap();

    let wire = raw_http(
        decode_port,
        "/generate",
        &format!(r#"{{"kv_key":"{key}","stream":true,"sampling_params":{{"max_tokens":32,"seed":42}}}}"#),
    );
    let head_end = wire.find("\r\n\r\n").expect("http head");
    let head = &wire[..head_end];
    assert!(head.contains("200 OK"), "head: {head}");
    assert!(head.contains("text/event-stream"), "head: {head}");
    let body = &wire[head_end + 4..];
    let mut pieces = Vec::new();
    let mut saw_done = false;
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else { continue };
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        let v = pagoda::json::parse(data).unwrap();
        if let Some(piece) = v.get("token").and_then(|t| t.as_str()) {
            pieces.push(piece.to_string());
        }
    }
    assert!(saw_done, "stream must terminate with [DONE]: {body}");
    assert_eq!(pieces.concat(), baseline.text, "wire: {body}");
}

/// The conductor router: prefill once, decode on the prefix-affine worker;
/// repeated prompts stick to the same worker.
#[test]
fn pd_router_e2e() {
    let store_port = free_port();
    let store_url = format!("http://127.0.0.1:{store_port}");
    std::thread::spawn(move || {
        pd::run_store(&format!("127.0.0.1:{store_port}"), 1 << 20, None).unwrap()
    });
    wait_until_ready(store_port);

    let mut prefill_engine = ToyEngine::toy(EngineConfig::default());
    prefill_engine.enable_pd(
        PdRole::Prefill,
        Arc::new(pd::HttpStore::from_url(&store_url).unwrap()),
    );
    let prefill_port = free_port();
    let prefill_addr = format!("127.0.0.1:{prefill_port}");
    std::thread::spawn(move || {
        server::run_full(
            Arc::new(Mutex::new(prefill_engine)),
            &prefill_addr,
            None,
            None,
            None,
        )
        .unwrap();
    });
    wait_until_ready(prefill_port);

    let mut decode_ports = Vec::new();
    for _ in 0..2 {
        let mut engine = ToyEngine::toy(EngineConfig::default());
        engine.enable_pd(
            PdRole::Decode,
            Arc::new(pd::HttpStore::from_url(&store_url).unwrap()),
        );
        let port = free_port();
        let addr = format!("127.0.0.1:{port}");
        std::thread::spawn(move || {
            server::run_full(Arc::new(Mutex::new(engine)), &addr, None, None, None).unwrap();
        });
        wait_until_ready(port);
        decode_ports.push(port);
    }

    let router_port = free_port();
    let router_addr = format!("127.0.0.1:{router_port}");
    let decode_urls: Vec<String> = decode_ports
        .iter()
        .map(|p| format!("http://127.0.0.1:{p}"))
        .collect();
    let prefill_url = format!("http://127.0.0.1:{prefill_port}");
    std::thread::spawn(move || {
        pd::run_router(&router_addr, &prefill_url, &decode_urls, None).unwrap();
    });
    wait_until_ready(router_port);

    let prompt = "Reliability comes from metrics and structured logging";
    let baseline = ToyEngine::toy(EngineConfig::default())
        .generate(&WriteRequest::new(prompt, params()));
    let body = format!(
        r#"{{"text":{:?},"sampling_params":{{"max_tokens":32,"seed":42}}}}"#,
        prompt
    );
    let (status, resp) = http(router_port, "POST", "/generate", Some(&body)).expect("http");
    assert_eq!(status, 200, "router: {resp}");
    assert!(
        resp.contains(&json_string(&baseline.text)),
        "router output must equal unified baseline\ngot: {resp}"
    );
    // Second identical prompt sticks to the same decode worker.
    let (status, _) = http(router_port, "POST", "/generate", Some(&body)).expect("http");
    assert_eq!(status, 200);

    let (_, stats) = http(router_port, "GET", "/route/stats", None).expect("http");
    assert!(
        stats.contains("\"routed\":2"),
        "one worker must hold both requests: {stats}"
    );
    let mut total_decodes = 0f64;
    for port in &decode_ports {
        let (_, s) = http(*port, "GET", "/stats", None).expect("http");
        let v = pagoda::json::parse(&s).unwrap();
        total_decodes += v
            .get("pd_decode_requests")
            .and_then(|n| n.as_f64())
            .unwrap_or(0.0);
    }
    assert_eq!(total_decodes, 2.0, "both decodes happened on workers");
}
/// A canned Laya decision server (same shape as system_tests' mock): texts
/// containing 威胁 or cancel escalate; everything else passes clean.
fn mock_laya(port: u16) {
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind laya");
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { continue };
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            let text = String::from_utf8_lossy(&buf);
                            if let Some(head_end) = text.find("\r\n\r\n") {
                                let len: usize = text[..head_end]
                                    .to_ascii_lowercase()
                                    .split("\r\n")
                                    .find_map(|h| h.strip_prefix("content-length:"))
                                    .and_then(|v| v.trim().parse().ok())
                                    .unwrap_or(0);
                                if buf.len() >= head_end + 4 + len {
                                    break;
                                }
                            }
                        }
                    }
                }
                let text = String::from_utf8_lossy(&buf);
                // Judge only the request's state field — the triage questions
                // themselves contain words like cancel and would otherwise
                // look hostile to this mock.
                let state = text
                    .split(r#""state":""#)
                    .nth(1)
                    .and_then(|rest| rest.split('"').next())
                    .unwrap_or("");
                let hostile = state.contains("威胁") || state.contains("cancel");
                let (churn, human) = if hostile { (0.91, 0.85) } else { (0.03, 0.02) };
                let resp = format!(
                    r#"{{"model":"rl-agent","answers":{{"department":{{"type":"choice","choice":"billing","confidence":0.93}},"churn_risk":{{"type":"noul","noul":{churn}}},"needs_human":{{"type":"noul","noul":{human}}}}},"usage":{{"input_tokens":64,"output_tokens":0}}}}"#
                );
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    resp.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(resp.as_bytes());
            });
        }
    });
}

/// Store TTL: an entry outliving max_age must read as a miss, free its
/// bytes, and count under stats.expired.
#[test]
fn store_ttl_expires_entries() {
    let store = LocalStore::with_max_age(1 << 20, Some(std::time::Duration::from_millis(80)));
    store.put("k1", "bundle-bytes").unwrap();
    assert_eq!(store.get("k1").unwrap().as_deref(), Some("bundle-bytes"));

    std::thread::sleep(std::time::Duration::from_millis(150));
    assert_eq!(store.get("k1").unwrap(), None, "entry must expire");
    let stats = store.stats();
    assert_eq!(stats.expired, 1, "one lazy expiry: {stats:?}");
    assert_eq!(stats.entries, 0, "expired entry removed: {stats:?}");
    assert_eq!(stats.bytes, 0, "expired bytes freed: {stats:?}");

    // No TTL: nothing expires.
    let store = LocalStore::new(1 << 20);
    store.put("k", "v").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert_eq!(store.get("k").unwrap().as_deref(), Some("v"));
}

/// Router triage gate: hostile text is escalated before any prefill compute
/// (the prefill worker is deliberately dead, so only the gate can produce a
/// 200); clean text passes the gate and fails at the dead prefill with 502.
#[test]
fn pd_router_triage_gate_escalates_before_prefill() {
    let laya_port = free_port();
    mock_laya(laya_port);
    wait_until_ready(laya_port);

    let dead_prefill_port = free_port();
    let dead_decode_port = free_port();
    let router_port = free_port();
    let router_addr = format!("127.0.0.1:{router_port}");
    let prefill_url = format!("http://127.0.0.1:{dead_prefill_port}");
    let decode_urls = vec![format!("http://127.0.0.1:{dead_decode_port}")];
    let triage = Arc::new(
        pagoda::triage::Triage::from_url(&format!("http://127.0.0.1:{laya_port}")).unwrap(),
    );
    std::thread::spawn(move || {
        pd::run_router(&router_addr, &prefill_url, &decode_urls, Some(triage)).unwrap();
    });
    wait_until_ready(router_port);

    // Hostile: the gate answers directly; the dead prefill is never called.
    let (status, body) = http(
        router_port,
        "POST",
        "/generate",
        Some(r#"{"text":"我要威胁投诉你们","sampling_params":{"max_tokens":4}}"#),
    )
    .expect("http");
    assert_eq!(status, 200, "escalation should be a 200 handoff: {body}");
    assert!(
        body.contains(r#""escalated":true"#),
        "escalation body: {body}"
    );

    // Clean: passes the gate, then reaches the (dead) prefill stage.
    let (status, body) = http(
        router_port,
        "POST",
        "/generate",
        Some(r#"{"text":"hello world","sampling_params":{"max_tokens":4}}"#),
    )
    .expect("http");
    assert_eq!(status, 502, "clean request must reach prefill: {body}");
    assert!(body.contains("prefill_unreachable"), "{body}");
}

/// A prefill worker that answers after a fixed delay, to expose lock scope:
/// the decode engine must stay usable while a conductor prefill is in flight.
fn slow_prefill(port: u16, delay: std::time::Duration) {
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind slow prefill");
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { continue };
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            buf.extend_from_slice(&chunk[..n]);
                            let text = String::from_utf8_lossy(&buf);
                            if let Some(head_end) = text.find("\r\n\r\n") {
                                let len: usize = text[..head_end]
                                    .to_ascii_lowercase()
                                    .split("\r\n")
                                    .find_map(|h| h.strip_prefix("content-length:"))
                                    .and_then(|v| v.trim().parse().ok())
                                    .unwrap_or(0);
                                if buf.len() >= head_end + 4 + len {
                                    break;
                                }
                            }
                        }
                    }
                }
                std::thread::sleep(delay);
                let body = r#"{"kv_key":"kv-slow","prompt_tokens":2,"prefill_tokens":2,"kv_bytes":0}"#;
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(body.as_bytes());
            });
        }
    });
}

/// Conductor prefill must NOT hold the decode engine lock: while request A
/// waits on a slow prefill worker, request B (already holding a kv_key) must
/// decode immediately. Regression test for the prefill-hoist fix.
#[test]
fn conductor_prefill_does_not_block_decode_engine() {
    use pagoda::Tokenizer;

    let prefill_port = free_port();
    slow_prefill(prefill_port, std::time::Duration::from_secs(3));
    wait_until_ready(prefill_port);

    // Decode engine with a seeded bundle B can decode from directly.
    let store = Arc::new(LocalStore::default());
    let mut decode = ToyEngine::toy(EngineConfig::default());
    decode.enable_pd(PdRole::Decode, store.clone());
    let seed_tokens = pagoda::ByteTokenizer::new().encode("seeded prompt for b");
    let seed_key = pd::bundle_key(&seed_tokens);
    let bundle = PrefillBundle {
        prompt_tokens: seed_tokens,
        kv: None,
    };
    store.put(&seed_key, &bundle.to_json()).unwrap();

    let decode_port = free_port();
    let decode_addr = format!("127.0.0.1:{decode_port}");
    let conductor = Conductor::from_url(&format!("http://127.0.0.1:{prefill_port}")).unwrap();
    std::thread::spawn(move || {
        server::run_full(
            Arc::new(Mutex::new(decode)),
            &decode_addr,
            None,
            None,
            Some(Arc::new(conductor)),
        )
        .unwrap();
    });
    wait_until_ready(decode_port);

    // A: text request -> conductor -> slow prefill (3s). Runs in background.
    let a = std::thread::spawn(move || {
        http(
            decode_port,
            "POST",
            "/generate",
            Some(r#"{"text":"hello","sampling_params":{"max_tokens":2}}"#),
        )
        .expect("http A")
    });

    // B at +0.3s: kv_key decode; must not wait for A's 3s prefill.
    std::thread::sleep(std::time::Duration::from_millis(300));
    let t0 = std::time::Instant::now();
    let (status, body) = http(
        decode_port,
        "POST",
        "/generate",
        Some(&format!(
            r#"{{"kv_key":"{seed_key}","sampling_params":{{"max_tokens":4}}}}"#
        )),
    )
    .expect("http B");
    let b_wall = t0.elapsed();
    assert_eq!(status, 200, "B must decode from the seeded bundle: {body}");
    assert!(
        b_wall < std::time::Duration::from_secs(2),
        "B waited {:?} for the decode engine; conductor prefill must not hold the lock",
        b_wall
    );

    // A completes after the slow prefill, then misses the bogus kv key.
    let (a_status, _) = a.join().unwrap();
    assert_eq!(a_status, 404, "A decodes a key the slow prefill never stored");
}