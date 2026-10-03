// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
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

/// A canned upstream worker (stand-in for SGLang): replies with fixed JSON
/// and records the request lines it received.
fn mock_upstream(port: u16, received: Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind upstream");
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { continue };
            let received = Arc::clone(&received);
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                // Read until the full body arrived (headers + content-length).
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
                                    received.lock().unwrap().push(text.into_owned());
                                    break;
                                }
                            }
                        }
                    }
                }
                let resp = r#"{"text":"upstream says hi","finish_reason":"length","meta":{"upstream":true}}"#;
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

/// Proxy mode: generation endpoints forward verbatim to the upstream worker
/// (SGLang co-deployment), while the control plane (health/stats/checkpoints)
/// stays local and stats expose the upstream + proxied request count.
#[test]
fn http_proxy_forwards_generation_and_keeps_control_plane() {
    let upstream_port = free_port();
    let received = Arc::new(Mutex::new(Vec::new()));
    mock_upstream(upstream_port, Arc::clone(&received));
    wait_until_ready(upstream_port);

    let engine = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 64,
        block_size: 8,
        ..EngineConfig::default()
    });
    let proxy = pagoda::server::Proxy::from_url(&format!("http://127.0.0.1:{upstream_port}"))
        .expect("valid upstream url");
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run_with_proxy(Arc::new(Mutex::new(engine)), &addr, Some(Arc::new(proxy)))
            .expect("server run");
    });
    wait_until_ready(port);

    // 1. /generate is proxied verbatim: the upstream's canned body comes back.
    let (status, body) = http(
        port,
        "POST",
        "/generate",
        Some(r#"{"text":"hello upstream","sampling_params":{"max_tokens":4}}"#),
    );
    assert_eq!(status, 200, "proxied body: {body}");
    assert!(body.contains("upstream says hi"), "body: {body}");

    // 2. /v1/chat/completions is proxied too.
    let (status, body) = http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"hi"}]}"#),
    );
    assert_eq!(status, 200);
    assert!(body.contains("upstream says hi"), "body: {body}");

    // The upstream saw exactly these two requests, with bodies intact.
    {
        let got = received.lock().unwrap();
        assert_eq!(got.len(), 2, "upstream requests: {got:?}");
        assert!(got[0].starts_with("POST /generate "), "got: {}", got[0]);
        assert!(got[0].contains("hello upstream"), "got: {}", got[0]);
        assert!(
            got[1].starts_with("POST /v1/chat/completions "),
            "got: {}",
            got[1]
        );
    }

    // 3. Control plane stays local: health does not hit the upstream.
    let (status, body) = http(port, "GET", "/health", None);
    assert_eq!(status, 200);
    assert!(body.contains("\"status\":\"ok\""));
    assert_eq!(received.lock().unwrap().len(), 2, "health must stay local");

    // 4. /stats is local and exposes the upstream + proxied count.
    let (status, body) = http(port, "GET", "/stats", None);
    assert_eq!(status, 200);
    let v = pagoda::json::parse(&body).expect("stats json");
    assert_eq!(
        v.get("proxied_requests").and_then(|t| t.as_f64()),
        Some(2.0),
        "stats body: {body}"
    );
    assert!(
        v.get("upstream")
            .and_then(|t| t.as_str())
            .is_some_and(|u| u.contains(&upstream_port.to_string())),
        "stats body: {body}"
    );

    // 5. Checkpoints stay local too (upstream never sees them).
    let (status, _) = http(port, "POST", "/checkpoint", Some(r#"{"text":"trunk"}"#));
    assert_eq!(status, 200);
    assert_eq!(received.lock().unwrap().len(), 2);
}

/// A canned Laya decision server: answers the triage questions with fixed
/// probabilities tuned per request — texts containing "威胁" escalate,
/// everything else passes clean.
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
                // themselves contain words like "cancel" and would otherwise
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

/// Triage gateway: the Laya System-1 gate sits between the client and the
/// upstream worker. Safe requests forward; hostile ones get a human-handoff
/// reply and never reach the (expensive) generation worker.
#[test]
fn http_triage_gateway_routes_and_escalates() {
    let laya_port = free_port();
    mock_laya(laya_port);
    wait_until_ready(laya_port);
    let upstream_port = free_port();
    let received = Arc::new(Mutex::new(Vec::new()));
    mock_upstream(upstream_port, Arc::clone(&received));
    wait_until_ready(upstream_port);

    let engine = ToyEngine::toy(EngineConfig::default());
    let proxy = server::Proxy::from_url(&format!("http://127.0.0.1:{upstream_port}")).unwrap();
    let triage = pagoda::triage::Triage::from_url(&format!("http://127.0.0.1:{laya_port}")).unwrap();
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run_with_triage(
            Arc::new(Mutex::new(engine)),
            &addr,
            Some(Arc::new(proxy)),
            Some(Arc::new(triage)),
        )
        .expect("server run");
    });
    wait_until_ready(port);

    // 1. Safe request: triaged, then forwarded to the upstream worker.
    let (status, body) = http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"帮我改一下收货地址"}]}"#),
    );
    assert_eq!(status, 200);
    assert!(body.contains("upstream says hi"), "safe must forward: {body}");
    assert_eq!(received.lock().unwrap().len(), 1);

    // 2. Hostile request: escalated, upstream never sees it.
    let (status, body) = http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"再不处理我就威胁投诉到底 cancel"}]}"#),
    );
    assert_eq!(status, 200);
    assert!(body.contains("转接人工客服"), "escalation reply: {body}");
    assert!(body.contains("pagoda_triage"), "triage extension: {body}");
    assert_eq!(received.lock().unwrap().len(), 1, "upstream must not be hit");

    // 3. /generate is triaged too.
    let (status, body) = http(
        port,
        "POST",
        "/generate",
        Some(r#"{"text":"我要 cancel 订阅并威胁拒付"}"#),
    );
    assert_eq!(status, 200);
    assert!(body.contains("转接人工客服"), "generate escalation: {body}");
    assert_eq!(received.lock().unwrap().len(), 1);

    // 4. Stats expose the triage counters (3 triaged, 2 escalated).
    let (status, body) = http(port, "GET", "/stats", None);
    assert_eq!(status, 200);
    let v = pagoda::json::parse(&body).expect("stats json");
    assert_eq!(v.get("triaged_requests").and_then(|t| t.as_f64()), Some(3.0), "{body}");
    assert_eq!(v.get("escalated_requests").and_then(|t| t.as_f64()), Some(2.0), "{body}");
    assert_eq!(v.get("proxied_requests").and_then(|t| t.as_f64()), Some(1.0), "{body}");
}

/// Raw variant of http(): returns the whole response, framing included,
/// so streaming tests can assert on Transfer-Encoding and chunk bytes.
fn http_raw(port: u16, method: &str, path: &str, body: Option<&str>) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).expect("write request");
    let mut resp = String::new();
    stream.read_to_string(&mut resp).expect("read response");
    resp
}

/// A canned streaming upstream (stand-in for SGLang SSE): answers every
/// request with three chunked data events, 20ms apart.
fn mock_streaming_upstream(port: u16) {
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind streaming upstream");
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { continue };
            std::thread::spawn(move || {
                // Drain the request (headers + content-length body).
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
                let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(head.as_bytes());
                for ev in [
                    "data: {\"text\":\"你\"}\n\n",
                    "data: {\"text\":\"好\"}\n\n",
                    "data: [DONE]\n\n",
                ] {
                    let framed = format!("{:x}\r\n{}\r\n", ev.len(), ev);
                    let _ = stream.write_all(framed.as_bytes());
                    std::thread::sleep(Duration::from_millis(20));
                }
                let _ = stream.write_all(b"0\r\n\r\n");
            });
        }
    });
}

/// SSE passthrough: a stream:true request gets the upstream's events relayed
/// chunk by chunk (framing intact, order preserved), a plain request to the
/// same upstream is transparently de-chunked, and /stats counts both.
#[test]
fn http_proxy_streams_sse_verbatim_and_counts() {
    let upstream_port = free_port();
    mock_streaming_upstream(upstream_port);
    wait_until_ready(upstream_port);

    let engine = ToyEngine::toy(EngineConfig {
        num_kv_blocks: 64,
        block_size: 8,
        ..EngineConfig::default()
    });
    let proxy = pagoda::server::Proxy::from_url(&format!("http://127.0.0.1:{upstream_port}"))
        .expect("valid upstream url");
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run_with_proxy(Arc::new(Mutex::new(engine)), &addr, Some(Arc::new(proxy)))
            .expect("server run");
    });
    wait_until_ready(port);

    // 1. stream:true: chunked SSE relayed with framing and order intact.
    let raw = http_raw(
        port,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"model":"x","stream":true,"messages":[{"role":"user","content":"hi"}]}"#),
    );
    assert!(raw.starts_with("HTTP/1.1 200 OK\r\n"), "raw: {raw:?}");
    let lower = raw.to_ascii_lowercase();
    assert!(lower.contains("transfer-encoding: chunked"), "raw: {raw:?}");
    assert!(lower.contains("content-type: text/event-stream"), "raw: {raw:?}");
    let first = raw.find("你").expect("event 1");
    let second = raw.find("好").expect("event 2");
    let done = raw.find("data: [DONE]").expect("done event");
    assert!(first < second && second < done, "order broken: {raw:?}");
    assert!(raw.ends_with("0\r\n\r\n"), "terminal chunk missing: {raw:?}");

    // 2. stream absent: the buffered path transparently de-chunks.
    let (status, body) = http(port, "POST", "/generate", Some(r#"{"text":"hi"}"#));
    assert_eq!(status, 200);
    assert!(body.contains("你") && body.contains("[DONE]"), "dechunked body: {body}");

    // 3. Counters: two proxied, exactly one streamed.
    let (status, stats) = http(port, "GET", "/stats", None);
    assert_eq!(status, 200);
    let v = pagoda::json::parse(&stats).expect("stats json");
    assert_eq!(
        v.get("proxied_requests").and_then(|t| t.as_f64()),
        Some(2.0),
        "stats: {stats}"
    );
    assert_eq!(
        v.get("streamed_requests").and_then(|t| t.as_f64()),
        Some(1.0),
        "stats: {stats}"
    );
}

/// A routing-aware Laya mock: the department answer follows the state text
/// (发票/账单 → billing，报错/故障 → technical，其余 → other)，全程低风险不升级。
fn mock_laya_routing(port: u16) {
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind laya routing");
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
                let state = text
                    .split(r#""state":""#)
                    .nth(1)
                    .and_then(|rest| rest.split('"').next())
                    .unwrap_or("");
                let dept = if state.contains("发票") || state.contains("账单") {
                    "billing"
                } else if state.contains("报错") || state.contains("故障") {
                    "technical"
                } else {
                    "other"
                };
                let resp = format!(
                    r#"{{"model":"rl-agent","answers":{{"department":{{"type":"choice","choice":"{dept}","confidence":0.93}},"churn_risk":{{"type":"noul","noul":0.03}},"needs_human":{{"type":"noul","noul":0.02}}}},"usage":{{"input_tokens":64,"output_tokens":0}}}}"#
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

/// Department routing: Laya's department pick steers the request to the
/// dedicated --route upstream; unmatched departments fall to the default.
#[test]
fn http_triage_routes_by_department_to_dedicated_upstream() {
    let laya_port = free_port();
    mock_laya_routing(laya_port);
    wait_until_ready(laya_port);

    let default_port = free_port();
    let received_default = Arc::new(Mutex::new(Vec::new()));
    mock_upstream(default_port, Arc::clone(&received_default));
    wait_until_ready(default_port);

    let billing_port = free_port();
    let received_billing = Arc::new(Mutex::new(Vec::new()));
    mock_upstream(billing_port, Arc::clone(&received_billing));
    wait_until_ready(billing_port);

    let engine = ToyEngine::toy(EngineConfig::default());
    let proxy = server::Proxy::with_routes(
        &format!("http://127.0.0.1:{default_port}"),
        &[("billing", &format!("http://127.0.0.1:{billing_port}"))],
    )
    .expect("proxy with routes");
    let triage = pagoda::triage::Triage::from_url(&format!("http://127.0.0.1:{laya_port}")).unwrap();
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run_with_triage(
            Arc::new(Mutex::new(engine)),
            &addr,
            Some(Arc::new(proxy)),
            Some(Arc::new(triage)),
        )
        .expect("server run");
    });
    wait_until_ready(port);

    // 1. Billing-ish request → the billing upstream, not the default.
    let (status, _) = http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"我的发票金额算错了，请重新开具"}]}"#),
    );
    assert_eq!(status, 200);
    assert_eq!(received_billing.lock().unwrap().len(), 1, "billing must serve");
    assert_eq!(received_default.lock().unwrap().len(), 0, "default must stay idle");

    // 2. Unmatched department → default upstream.
    let (status, _) = http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"你好，随便聊聊"}]}"#),
    );
    assert_eq!(status, 200);
    assert_eq!(received_default.lock().unwrap().len(), 1);
    assert_eq!(received_billing.lock().unwrap().len(), 1);

    // 3. /stats: both proxied, exactly one routed to a department upstream,
    //    and the routing table is observable.
    let (status, stats) = http(port, "GET", "/stats", None);
    assert_eq!(status, 200);
    let v = pagoda::json::parse(&stats).expect("stats json");
    assert_eq!(
        v.get("proxied_requests").and_then(|t| t.as_f64()),
        Some(2.0),
        "stats: {stats}"
    );
    assert_eq!(
        v.get("routed_requests").and_then(|t| t.as_f64()),
        Some(1.0),
        "stats: {stats}"
    );
    assert!(
        stats.contains(&format!("billing")), "routes visible: {stats}"
    );
}
