// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: AGPL-3.0-only
//! System-level end-to-end test: boot the real HTTP server on a loopback port
//! and drive it with hand-written HTTP/1.1 requests, exercising every endpoint
//! including constrained decoding through the wire.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pagoda::server;
use pagoda::{EngineConfig, ToyEngine};

/// Ask the OS for an ephemeral port, then release it for the server to bind.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// One HTTP/1.1 round trip; returns (status, body).
fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).expect("write request");
    let mut resp = String::new();
    stream.read_to_string(&mut resp).expect("read response");
    let status: u16 = resp
        .split_whitespace()
        .nth(1)
        .expect("status code")
        .parse()
        .expect("numeric status");
    let body_start = resp.find("\r\n\r\n").map(|i| i + 4).unwrap_or(resp.len());
    (status, resp[body_start..].to_string())
}

fn wait_until_ready(port: u16) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("server did not start on port {port}");
}

#[test]
fn http_server_end_to_end() {
    let engine = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 256,
        block_size: 8,
        ..EngineConfig::default()
    });
    let shared = Arc::new(Mutex::new(engine));
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run(shared, &addr).expect("server run");
    });
    wait_until_ready(port);

    // 1. Health probe.
    let (status, body) = http(port, "GET", "/health", None);
    assert_eq!(status, 200);
    assert!(body.contains("\"status\":\"ok\""), "health body: {body}");

    // 2. Plain generation.
    let (status, body) = http(
        port,
        "POST",
        "/generate",
        Some(r#"{"text":"the quick brown fox","sampling_params":{"max_tokens":8}}"#),
    );
    assert_eq!(status, 200, "generate body: {body}");
    let v = pagoda::json::parse(&body).expect("generate json");
    let text = v.get("text").and_then(|t| t.as_str()).unwrap();
    assert_eq!(text.len(), 8, "byte-level output should be exactly max_tokens");
    assert_eq!(
        v.get("finish_reason").and_then(|t| t.as_str()),
        Some("length")
    );

    // 3. Constrained decoding over the wire: output must be a full regex match.
    let (status, body) = http(
        port,
        "POST",
        "/generate",
        Some(
            r#"{"text":"","sampling_params":{"max_tokens":8,"grammar":{"type":"regex","pattern":"\"[a-z]{3}\""}}}"#,
        ),
    );
    assert_eq!(status, 200, "grammar body: {body}");
    let v = pagoda::json::parse(&body).expect("grammar json");
    let text = v.get("text").and_then(|t| t.as_str()).unwrap();
    assert_eq!(text.len(), 5, "expected a quoted 3-letter string, got {text:?}");
    assert!(text.starts_with('"') && text.ends_with('"'), "got {text:?}");
    assert!(text[1..4].bytes().all(|b| b.is_ascii_lowercase()));
    assert_eq!(
        v.get("finish_reason").and_then(|t| t.as_str()),
        Some("stop")
    );

    // 4. Admission guard over the wire: empty prompt without grammar is rejected.
    let (status, body) = http(
        port,
        "POST",
        "/generate",
        Some(r#"{"text":"","sampling_params":{"max_tokens":4}}"#),
    );
    assert_eq!(status, 400);
    assert!(body.contains("empty_prompt"), "rejection body: {body}");

    // 5. OpenAI-compatible chat endpoint.
    let (status, body) = http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"hello"}]}"#),
    );
    assert_eq!(status, 200, "chat body: {body}");
    assert!(body.contains("content"), "chat body: {body}");

    // 6. Stats reflect the traffic above.
    let (status, body) = http(port, "GET", "/stats", None);
    assert_eq!(status, 200);
    let v = pagoda::json::parse(&body).expect("stats json");
    let total = v
        .get("total_requests")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0);
    assert!(total >= 3.0, "stats should count generations: {body}");

    // 7. Unknown route.
    let (status, _) = http(port, "GET", "/nope", None);
    assert_eq!(status, 404);
}

#[test]
fn http_checkpoint_lifecycle() {
    let engine = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 256,
        block_size: 8,
        ..EngineConfig::default()
    });
    let shared = Arc::new(Mutex::new(engine));
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run(shared, &addr).expect("server run");
    });
    wait_until_ready(port);

    // 1. Create a checkpoint (the shared agent trunk).
    let trunk = "shared system prompt for agents"; // 31 bytes
    let (status, body) = http(
        port,
        "POST",
        "/checkpoint",
        Some(r#"{"text":"shared system prompt for agents"}"#),
    );
    assert_eq!(status, 200, "checkpoint body: {body}");
    let v = pagoda::json::parse(&body).expect("checkpoint json");
    let id = v
        .get("checkpoint_id")
        .and_then(|t| t.as_f64())
        .expect("checkpoint id") as u64;

    // 2. Branch off it: the whole trunk must come back as a prefix hit.
    let (status, body) = http(
        port,
        "POST",
        "/checkpoint/generate",
        Some(&format!(
            r#"{{"checkpoint_id":{id},"text":" user turn","sampling_params":{{"max_tokens":6}}}}"#
        )),
    );
    assert_eq!(status, 200, "branch body: {body}");
    let v = pagoda::json::parse(&body).expect("branch json");
    assert_eq!(
        v.get("prefix_hit_tokens").and_then(|t| t.as_f64()),
        Some(trunk.len() as f64),
        "branch must hit the whole pinned trunk: {body}"
    );
    assert_eq!(
        v.get("checkpoint_id").and_then(|t| t.as_f64()),
        Some(id as f64)
    );

    // 3. Unknown checkpoint -> 404.
    let (status, _) = http(
        port,
        "POST",
        "/checkpoint/generate",
        Some(r#"{"checkpoint_id":9999,"text":"hi"}"#),
    );
    assert_eq!(status, 404);

    // 4. Delete the checkpoint.
    let (status, body) = http(
        port,
        "POST",
        "/checkpoint/delete",
        Some(&format!(r#"{{"checkpoint_id":{id}}}"#)),
    );
    assert_eq!(status, 200);
    assert!(body.contains("\"deleted\":true"), "delete body: {body}");

    // 5. Branching off a deleted checkpoint fails.
    let (status, _) = http(
        port,
        "POST",
        "/checkpoint/generate",
        Some(&format!(r#"{{"checkpoint_id":{id},"text":"hi"}}"#)),
    );
    assert_eq!(status, 404);
}
