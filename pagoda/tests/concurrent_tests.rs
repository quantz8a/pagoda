// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Request-level concurrency: the scheduler actor must interleave in-flight
//! requests in one continuous-batching loop while producing exactly the
//! outputs a sequential engine would.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use pagoda::actor::StreamEvent;
use pagoda::server;
use pagoda::{EngineConfig, SamplingParams, ToyEngine, WriteRequest};

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

const PROMPTS: [&str; 4] = [
    "SGLang uses a radix tree to cache",
    "Continuous batching interleaves prefill and decode",
    "A scheduler actor owns the engine",
    "Mooncake separates prefill from decode",
];

/// Four requests submitted at once through the actor must each deliver the
/// sequential baseline text, and the per-token events must join to the same
/// string the terminal Done carries.
#[test]
fn actor_matches_sequential_outputs() {
    let mut sequential = ToyEngine::toy(EngineConfig::default());
    let baselines: Vec<String> = PROMPTS
        .iter()
        .map(|p| sequential.generate(&WriteRequest::new(*p, params())).text)
        .collect();

    let handle = ToyEngine::toy(EngineConfig::default()).into_actor();
    let receivers: Vec<_> = PROMPTS
        .iter()
        .map(|p| handle.subscribe(WriteRequest::new(*p, params())))
        .collect();
    for (rx, expected) in receivers.into_iter().zip(baselines.iter()) {
        let mut streamed = String::new();
        let mut done = false;
        for event in rx {
            match event {
                StreamEvent::Token(piece) => streamed.push_str(&piece),
                StreamEvent::Done(out) => {
                    assert_eq!(&out.text, expected, "buffered text");
                    assert_eq!(out.finish_reason.to_string(), "length");
                    done = true;
                }
            }
        }
        assert!(done, "stream must end with Done");
        assert_eq!(&streamed, expected, "streamed pieces join to baseline");
    }
}

/// Raw HTTP/1.1 request returning the full wire response (for SSE reading).
fn raw_http(port: u16, path: &str, body: &str) -> String {
    let mut s = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).unwrap();
    String::from_utf8_lossy(&buf).into_owned()
}

fn body_for(prompt: &str) -> String {
    format!(r#"{{"text":"{prompt}","sampling_params":{{"max_tokens":32,"seed":42}}}}"#)
}

/// HTTP e2e over the concurrent server: three buffered requests in flight
/// together plus one SSE stream, all matching sequential baselines; /stats
/// stays responsive and reports the concurrent mode.
#[test]
fn concurrent_http_buffered_and_sse_match_sequential() {
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run_concurrent(ToyEngine::toy(EngineConfig::default()), &addr, None).unwrap();
    });
    wait_until_ready(port);

    let mut sequential = ToyEngine::toy(EngineConfig::default());
    let baselines: Vec<String> = PROMPTS
        .iter()
        .map(|p| sequential.generate(&WriteRequest::new(*p, params())).text)
        .collect();

    // Three buffered requests on their own threads.
    let mut joins = Vec::new();
    for (i, prompt) in PROMPTS.iter().take(3).enumerate() {
        let expected = baselines[i].clone();
        let body = body_for(prompt);
        joins.push(std::thread::spawn(move || {
            let (status, resp) = http(port, "POST", "/generate", Some(&body)).expect("http");
            assert_eq!(status, 200, "resp: {resp}");
            let v = pagoda::json::parse(&resp).unwrap();
            let text = v.get("text").and_then(|t| t.as_str()).unwrap();
            assert_eq!(text, expected, "buffered response text");
        }));
    }

    // One SSE stream on this thread.
    let wire = raw_http(
        port,
        "/generate",
        &body_for(PROMPTS[3]).replace(r#"{"text"#, r#"{"stream":true,"text"#),
    );
    let head_end = wire.find("\r\n\r\n").expect("http head");
    let head = &wire[..head_end];
    assert!(head.contains("200 OK"), "head: {head}");
    assert!(head.contains("text/event-stream"), "head: {head}");
    let mut pieces = Vec::new();
    let mut saw_done = false;
    for line in wire[head_end + 4..].lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            saw_done = true;
            continue;
        }
        let v = pagoda::json::parse(data).unwrap();
        if let Some(piece) = v.get("token").and_then(|t| t.as_str()) {
            pieces.push(piece.to_string());
        }
    }
    assert!(saw_done, "stream must terminate with [DONE]: {wire}");
    assert_eq!(pieces.concat(), baselines[3], "wire: {wire}");

    for join in joins {
        join.join().expect("buffered worker");
    }

    // Stats stays responsive after the burst and names the concurrent mode.
    let (status, resp) = http(port, "GET", "/stats", None).expect("stats");
    assert_eq!(status, 200, "resp: {resp}");
    let v = pagoda::json::parse(&resp).unwrap();
    assert_eq!(
        v.get("mode").and_then(|m| m.as_str()),
        Some("concurrent"),
        "stats: {resp}"
    );
}

/// A full waiting queue rejects immediately with QueueFull instead of
/// parking the request behind in-flight work.
#[test]
fn actor_rejects_when_waiting_queue_full() {
    let config = EngineConfig {
        max_waiting_requests: 0,
        ..EngineConfig::default()
    };
    let handle = ToyEngine::toy(config).into_actor();
    let rx = handle.subscribe(WriteRequest::new("hello world", params()));
    let mut done = None;
    for event in rx {
        if let StreamEvent::Done(out) = event {
            done = Some(out);
        }
    }
    let out = done.expect("Done arrives even for rejects");
    assert_eq!(out.finish_reason.to_string(), "rejected");
    assert_eq!(
        out.rejection.map(|r| r.to_string()),
        Some("queue_full".to_string())
    );
}

/// Shutdown stops the actor thread: later subscriptions find the channel
/// closed without a Done.
#[test]
fn actor_shutdown_stops_the_loop() {
    let handle = ToyEngine::toy(EngineConfig::default()).into_actor();
    let out = handle.generate(WriteRequest::new("hello world", params()));
    assert_eq!(out.finish_reason.to_string(), "length");

    handle.shutdown();
    for _ in 0..100 {
        let rx = handle.subscribe(WriteRequest::new("hello world", params()));
        let mut events = 0usize;
        for _ in rx {
            events += 1;
        }
        if events == 0 {
            return; // channel closed without Done: actor is gone
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("actor still serving after shutdown");
}

/// /v1/chat/completions on the concurrent server: buffered OpenAI-compat
/// response matching the sequential engine on the rendered prompt, and a
/// 400 for a malformed messages payload.
#[test]
fn concurrent_http_chat_completions() {
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run_concurrent(ToyEngine::toy(EngineConfig::default()), &addr, None).unwrap();
    });
    wait_until_ready(port);

    let body = r#"{"messages":[{"role":"user","content":"SGLang uses a radix tree to cache"}],"max_tokens":32,"seed":42}"#;
    let (status, resp) = http(port, "POST", "/v1/chat/completions", Some(body)).expect("http");
    assert_eq!(status, 200, "resp: {resp}");
    let v = pagoda::json::parse(&resp).unwrap();
    assert_eq!(
        v.get("object").and_then(|o| o.as_str()),
        Some("chat.completion")
    );
    let Some(pagoda::json::Value::Array(choices)) = v.get("choices") else {
        panic!("choices shape: {resp}")
    };
    let content = choices[0]
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap();

    let mut sampling = params();
    sampling.stop = vec!["\n".to_string()];
    let baseline = ToyEngine::toy(EngineConfig::default()).generate(&WriteRequest::new(
        "[USER] SGLang uses a radix tree to cache",
        sampling,
    ));
    assert_eq!(content, baseline.text);

    let (status, _) = http(port, "POST", "/v1/chat/completions", Some(r#"{"foo":1}"#))
        .expect("http");
    assert_eq!(status, 400, "malformed payload must be rejected");
}

/// A canned Laya decision server (same fixture as the system tests): texts
/// containing "威胁" escalate, everything else passes clean.
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

/// Cancel-on-disconnect: once the subscriber's receiver goes away, the next
/// token delivery fails and the actor aborts the sequence (KV refs released)
/// instead of running it to completion.
#[test]
fn actor_aborts_when_client_disconnects() {
    let config = EngineConfig {
        max_total_tokens: 1_000_000,
        ..EngineConfig::default()
    };
    let handle = ToyEngine::toy(config).into_actor();
    let long = SamplingParams {
        max_tokens: 100_000,
        ..SamplingParams::default()
    };
    let rx = handle.subscribe(WriteRequest::new("hello world", long));
    match rx.recv() {
        Ok(StreamEvent::Token(_)) => {}
        _ => panic!("expected a first token before disconnect"),
    }
    drop(rx);
    for _ in 0..200 {
        if handle.stats().aborted_requests == 1 {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("actor never aborted the disconnected request");
}

/// Checkpoint lifecycle through the actor protocol: create, branch-generate
/// (matching the lock-based engine bit-for-bit), delete; unknown ids surface
/// as None.
#[test]
fn actor_checkpoint_lifecycle() {
    let mut sequential = ToyEngine::toy(EngineConfig::default());
    let base_id = sequential.create_checkpoint("SGLang uses a radix tree to cache");
    let baseline = sequential
        .generate_from_checkpoint(base_id, " and shares prefixes", params())
        .expect("baseline checkpoint");

    let handle = ToyEngine::toy(EngineConfig::default()).into_actor();
    let id = handle
        .checkpoint_create("SGLang uses a radix tree to cache".to_string())
        .expect("create");
    let rx = handle
        .checkpoint_generate(id, " and shares prefixes".to_string(), params())
        .expect("known id");
    let mut done = None;
    for event in rx {
        if let StreamEvent::Done(out) = event {
            done = Some(out);
        }
    }
    let out = done.expect("Done");
    assert_eq!(out.text, baseline.text, "actor branch == sequential branch");
    assert_eq!(out.prefix_hit_tokens, baseline.prefix_hit_tokens);

    assert!(
        handle
            .checkpoint_generate(999_999, "x".to_string(), params())
            .is_none(),
        "unknown checkpoint id"
    );
    assert!(handle.checkpoint_drop(id));
    assert!(!handle.checkpoint_drop(id), "second drop finds nothing");
}

/// HTTP checkpoint lifecycle on the concurrent server: create, generate
/// (bit-identical to the sequential engine), delete, plus the 404/400 edges.
#[test]
fn concurrent_http_checkpoint_lifecycle() {
    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run_concurrent(ToyEngine::toy(EngineConfig::default()), &addr, None).unwrap();
    });
    wait_until_ready(port);

    let (status, resp) = http(
        port,
        "POST",
        "/checkpoint",
        Some(r#"{"text":"SGLang uses a radix tree to cache"}"#),
    )
    .expect("http");
    assert_eq!(status, 200, "resp: {resp}");
    let v = pagoda::json::parse(&resp).unwrap();
    let id = v
        .get("checkpoint_id")
        .and_then(|i| i.as_f64())
        .expect("checkpoint_id") as u64;

    let gen_body = format!(
        r#"{{"checkpoint_id":{id},"text":" and shares prefixes","sampling_params":{{"max_tokens":32,"seed":42}}}}"#
    );
    let (status, resp) = http(port, "POST", "/checkpoint/generate", Some(&gen_body)).expect("http");
    assert_eq!(status, 200, "resp: {resp}");
    let v = pagoda::json::parse(&resp).unwrap();
    let text = v.get("text").and_then(|t| t.as_str()).unwrap().to_string();
    assert_eq!(v.get("checkpoint_id").and_then(|i| i.as_f64()), Some(id as f64));

    let mut sequential = ToyEngine::toy(EngineConfig::default());
    let base_id = sequential.create_checkpoint("SGLang uses a radix tree to cache");
    let baseline = sequential
        .generate_from_checkpoint(base_id, " and shares prefixes", params())
        .expect("baseline");
    assert_eq!(text, baseline.text, "http branch == sequential branch");

    let (status, _) = http(
        port,
        "POST",
        "/checkpoint/generate",
        Some(r#"{"checkpoint_id":999999,"text":"x"}"#),
    )
    .expect("http");
    assert_eq!(status, 404, "unknown checkpoint must 404");

    let del_body = format!(r#"{{"checkpoint_id":{id}}}"#);
    let (status, resp) = http(port, "POST", "/checkpoint/delete", Some(&del_body)).expect("http");
    assert_eq!(status, 200, "resp: {resp}");
    let v = pagoda::json::parse(&resp).unwrap();
    assert_eq!(v.get("deleted").and_then(|d| d.as_bool()), Some(true));
    let (_, resp) = http(port, "POST", "/checkpoint/delete", Some(&del_body)).expect("http");
    let v = pagoda::json::parse(&resp).unwrap();
    assert_eq!(v.get("deleted").and_then(|d| d.as_bool()), Some(false));
}

/// The triage gate on the concurrent server escalates hostile text before
/// any compute: the engine never admits the request (total_requests counts
/// only the clean one).
#[test]
fn concurrent_http_triage_gate() {
    let laya_port = free_port();
    mock_laya(laya_port);
    wait_until_ready(laya_port);
    let triage = pagoda::triage::Triage::from_url(&format!("http://127.0.0.1:{laya_port}"))
        .expect("triage url");

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    std::thread::spawn(move || {
        server::run_concurrent(
            ToyEngine::toy(EngineConfig::default()),
            &addr,
            Some(Arc::new(triage)),
        )
        .unwrap();
    });
    wait_until_ready(port);

    let (status, body) = http(
        port,
        "POST",
        "/generate",
        Some(r#"{"text":"我要威胁你","sampling_params":{"max_tokens":8}}"#),
    )
    .expect("http");
    assert_eq!(status, 200, "body: {body}");
    assert!(body.contains("转接人工客服"), "escalation reply: {body}");

    let baseline = ToyEngine::toy(EngineConfig::default())
        .generate(&WriteRequest::new("hello world", params()));
    let (status, body) = http(
        port,
        "POST",
        "/generate",
        Some(r#"{"text":"hello world","sampling_params":{"max_tokens":32,"seed":42}}"#),
    )
    .expect("http");
    assert_eq!(status, 200, "body: {body}");
    let v = pagoda::json::parse(&body).unwrap();
    assert_eq!(
        v.get("text").and_then(|t| t.as_str()),
        Some(baseline.text.as_str()),
        "clean request generates normally"
    );

    let (_, stats) = http(port, "GET", "/stats", None).expect("stats");
    let s = pagoda::json::parse(&stats).unwrap();
    assert_eq!(
        s.get("total_requests").and_then(|r| r.as_f64()),
        Some(1.0),
        "escalated request never reached the engine: {stats}"
    );
}
