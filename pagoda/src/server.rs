// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! A minimal, dependency-free HTTP serving frontend.
//!
//! Implements a tiny subset of the HTTP/1.1 protocol on top of
//! `std::net::TcpListener`. Production deployments would replace this with a
//! full async server (axum/hyper), with the same request-handling logic.
//!
//! The frontend is backend-agnostic: it only requires the [`ServingEngine`]
//! capability (`generate` + `stats`), which the toy [`crate::engine::Engine`]
//! and real backends (e.g. the companion `pagoda-hf` crate) both implement.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use crate::engine::{CheckpointId, Engine, EngineStats};
use crate::grammar::Grammar;
use crate::json::{self, Value};
use crate::model::ModelEngine;
use crate::spec::{FinishReason, GenerationOutput, SamplingParams, WriteRequest};
use crate::tokenizer::Tokenizer;

/// A raw HTTP response ready to be serialized on the wire.
#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

impl HttpResponse {
    fn json(status: u16, body: &str) -> Self {
        Self {
            status,
            body: body.to_string(),
        }
    }
}

/// The minimal capability the HTTP frontend needs from a serving engine.
///
/// Implemented for every [`ModelEngine`] + [`Tokenizer`] pair via the blanket
/// impl below, so the server surface is independent of the concrete backend.
pub trait ServingEngine {
    fn generate(&mut self, req: &WriteRequest) -> GenerationOutput;
    fn stats(&self) -> EngineStats;

    /// Create a pinned KV checkpoint from `text`. Default: unsupported.
    fn create_checkpoint(&mut self, _text: &str) -> Option<u64> {
        None
    }
    /// Release a pinned checkpoint. Default: `false` (unsupported).
    fn drop_checkpoint(&mut self, _id: u64) -> bool {
        false
    }
    /// Branch generation off a pinned checkpoint. Default: unsupported.
    fn generate_from_checkpoint(
        &mut self,
        _id: u64,
        _continuation: &str,
        _sampling: SamplingParams,
    ) -> Option<GenerationOutput> {
        None
    }
}

impl<M: ModelEngine, T: Tokenizer> ServingEngine for Engine<M, T> {
    fn generate(&mut self, req: &WriteRequest) -> GenerationOutput {
        Engine::generate(self, req)
    }

    fn stats(&self) -> EngineStats {
        Engine::stats(self)
    }

    fn create_checkpoint(&mut self, text: &str) -> Option<u64> {
        Some(Engine::create_checkpoint(self, text).0)
    }

    fn drop_checkpoint(&mut self, id: u64) -> bool {
        Engine::drop_checkpoint(self, CheckpointId(id))
    }

    fn generate_from_checkpoint(
        &mut self,
        id: u64,
        continuation: &str,
        sampling: SamplingParams,
    ) -> Option<GenerationOutput> {
        Engine::generate_from_checkpoint(self, CheckpointId(id), continuation, sampling)
    }
}

/// Routing + request handling, factored out so it can be unit-tested without
/// a socket.
pub fn handle<E: ServingEngine>(
    engine: &mut E,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> HttpResponse {
    match (method, path) {
        ("GET", "/health") => HttpResponse::json(200, r#"{"status":"ok"}"#),
        ("GET", "/stats") => {
            let s = engine.stats();
            let body = Value::Object(vec![
                ("total_requests".to_string(), Value::Number(s.total_requests as f64)),
                ("total_prompt_tokens".to_string(), Value::Number(s.total_prompt_tokens as f64)),
                ("total_prefill_tokens".to_string(), Value::Number(s.total_prefill_tokens as f64)),
                ("total_output_tokens".to_string(), Value::Number(s.total_output_tokens as f64)),
                ("total_forward".to_string(), Value::Number(s.total_forward as f64)),
                (
                    "total_decode_steps".to_string(),
                    Value::Number(s.total_decode_steps as f64),
                ),
                (
                    "total_decode_calls".to_string(),
                    Value::Number(s.total_decode_calls as f64),
                ),
                (
                    "decode_batch_factor".to_string(),
                    Value::Number(s.decode_batch_factor()),
                ),
                (
                    "model_graft_tokens".to_string(),
                    Value::Number(s.model_graft_tokens as f64),
                ),
                ("radix_nodes".to_string(), Value::Number(s.radix_nodes as f64)),
                (
                    "radix_hit_tokens".to_string(),
                    Value::Number(s.radix_hit_tokens as f64),
                ),
                ("apc_blocks".to_string(), Value::Number(s.apc_blocks as f64)),
                (
                    "apc_hit_tokens".to_string(),
                    Value::Number(s.apc_hit_tokens as f64),
                ),
                (
                    "active_checkpoints".to_string(),
                    Value::Number(s.active_checkpoints as f64),
                ),
                (
                    "checkpoint_hit_tokens".to_string(),
                    Value::Number(s.checkpoint_hit_tokens as f64),
                ),
                (
                    "compute_saved_tokens".to_string(),
                    Value::Number(s.compute_saved_tokens() as f64),
                ),
                (
                    "prefill_skip_ratio".to_string(),
                    Value::Number(s.prefill_skip_ratio()),
                ),
                (
                    "avg_forward_per_output_token".to_string(),
                    Value::Number(s.avg_forward_per_output_token()),
                ),
                (
                    "kv_blocks".to_string(),
                    Value::Number(s.kv_blocks as f64),
                ),
                ("kv_free_blocks".to_string(), Value::Number(s.kv_free_blocks as f64)),
                ("kv_allocations".to_string(), Value::Number(s.kv_allocations as f64)),
                ("kv_frees".to_string(), Value::Number(s.kv_frees as f64)),
                ("kv_utilization".to_string(), Value::Number(s.kv_utilization())),
                ("faulted_requests".to_string(), Value::Number(s.faulted_requests as f64)),
                ("rejected_requests".to_string(), Value::Number(s.rejected_requests as f64)),
            ])
            .to_json();
            HttpResponse::json(200, &body)
        }
        ("POST", "/generate") => {
            let req = match body
                .and_then(|b| json::parse(b).ok())
                .and_then(|v| build_generate_request(v))
            {
                Some(req) => req,
                None => {
                    return HttpResponse::json(
                        400,
                        r#"{"error":"invalid request; expected {\"text\": ...}}"#,
                    )
                }
            };
            let out = engine.generate(&req);
            if out.finish_reason == FinishReason::Rejected {
                return rejection_response(&out);
            }
            HttpResponse::json(200, &generate_response(&out).to_json())
        }
        ("POST", "/v1/chat/completions") => {
            let req = match body
                .and_then(|b| json::parse(b).ok())
                .and_then(build_chat_request)
            {
                Some(req) => req,
                None => {
                    return HttpResponse::json(
                        400,
                        r#"{"error":"invalid messages payload"}"#,
                    )
                }
            };
            let out = engine.generate(&req);
            if out.finish_reason == FinishReason::Rejected {
                return rejection_response(&out);
            }
            HttpResponse::json(200, &chat_response(&out).to_json())
        }
        ("POST", "/checkpoint") => {
            let text = body
                .and_then(|b| json::parse(b).ok())
                .and_then(|v| v.get("text").and_then(Value::as_str).map(str::to_string));
            let Some(text) = text else {
                return HttpResponse::json(400, r#"{"error":"expected {\"text\": ...}"}"#);
            };
            match engine.create_checkpoint(&text) {
                Some(id) => HttpResponse::json(
                    200,
                    &Value::Object(vec![(
                        "checkpoint_id".to_string(),
                        Value::Number(id as f64),
                    )])
                    .to_json(),
                ),
                None => HttpResponse::json(
                    400,
                    r#"{"error":"checkpoints not supported by this backend"}"#,
                ),
            }
        }
        ("POST", "/checkpoint/generate") => {
            let parsed = body.and_then(|b| json::parse(b).ok()).and_then(|v| {
                let id = v.get("checkpoint_id")?.as_usize()? as u64;
                let text = v.get("text")?.as_str()?.to_string();
                let sampling = v
                    .get("sampling_params")
                    .map(parse_sampling)
                    .unwrap_or_default();
                Some((id, text, sampling))
            });
            let Some((id, text, sampling)) = parsed else {
                return HttpResponse::json(
                    400,
                    r#"{"error":"expected {\"checkpoint_id\": N, \"text\": ...}"}"#,
                );
            };
            match engine.generate_from_checkpoint(id, &text, sampling) {
                Some(out) => {
                    if out.finish_reason == FinishReason::Rejected {
                        return rejection_response(&out);
                    }
                    let mut resp = generate_response(&out);
                    if let Value::Object(ref mut entries) = resp {
                        entries.push(("checkpoint_id".to_string(), Value::Number(id as f64)));
                    }
                    HttpResponse::json(200, &resp.to_json())
                }
                None => HttpResponse::json(404, r#"{"error":"checkpoint not found"}"#),
            }
        }
        ("POST", "/checkpoint/delete") => {
            let id = body
                .and_then(|b| json::parse(b).ok())
                .and_then(|v| v.get("checkpoint_id").and_then(Value::as_usize))
                .map(|x| x as u64);
            let Some(id) = id else {
                return HttpResponse::json(400, r#"{"error":"expected {\"checkpoint_id\": N}"}"#);
            };
            let deleted = engine.drop_checkpoint(id);
            HttpResponse::json(
                200,
                &Value::Object(vec![("deleted".to_string(), Value::Bool(deleted))]).to_json(),
            )
        }
        _ => HttpResponse::json(404, r#"{"error":"not found"}"#),
    }
}

fn build_generate_request(v: Value) -> Option<WriteRequest> {
    let text = v.get("text")?.as_str()?.to_string();
    let sampling = v
        .get("sampling_params")
        .map(parse_sampling)
        .unwrap_or_default();
    Some(WriteRequest::new(text, sampling))
}

fn build_chat_request(v: Value) -> Option<WriteRequest> {
    let messages = v.get("messages")?;
    let Value::Array(items) = messages else { return None };
    let mut prompt = String::new();
    for item in items {
        let role = item.get("role")?.as_str()?;
        let content = item.get("content")?.as_str()?.to_string();
        match role {
            "system" => prompt.push_str(&format!("[SYS] {content}\n")),
            "assistant" => prompt.push_str(&format!("[ASSISTANT] {content}\n")),
            _ => prompt.push_str(&format!("[USER] {content}\n")),
        }
    }
    let mut sampling = parse_sampling(&v);
    if sampling.stop.is_empty() {
        sampling.stop = vec!["\n".to_string()]; // chat-completions convention
    }
    Some(WriteRequest::new(prompt.trim_end().to_string(), sampling))
}

fn parse_sampling(v: &Value) -> SamplingParams {
    let mut p = SamplingParams::default();
    if let Some(n) = v.get("max_tokens").and_then(Value::as_usize) {
        p.max_tokens = n;
    }
    if let Some(t) = v.get("temperature").and_then(Value::as_f64) {
        p.temperature = t as f32;
    }
    if let Some(t) = v.get("top_p").and_then(Value::as_f64) {
        p.top_p = t as f32;
    }
    if let Some(k) = v.get("top_k").and_then(Value::as_usize) {
        p.top_k = k;
    }
    if let Some(f) = v.get("frequency_penalty").and_then(Value::as_f64) {
        p.frequency_penalty = f as f32;
    }
    if let Some(f) = v.get("presence_penalty").and_then(Value::as_f64) {
        p.presence_penalty = f as f32;
    }
    if let Some(seed) = v.get("seed").and_then(Value::as_f64) {
        p.seed = seed as u64;
    }
    if let Some(Value::Array(stops)) = v.get("stop") {
        p.stop = stops
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
    }
    if let Some(Value::Array(ids)) = v.get("stop_token_ids") {
        p.stop_token_ids = ids.iter().filter_map(Value::as_usize).map(|x| x as u32).collect();
    }
    if let Some(Value::Object(entries)) = v.get("grammar") {
        let typ = entries
            .iter()
            .find(|(k, _)| k == "type")
            .and_then(|(_, val)| val.as_str());
        match typ {
            Some("json") => p.grammar = Some(Grammar::json()),
            Some("regex") => {
                if let Some(pattern) = entries
                    .iter()
                    .find(|(k, _)| k == "pattern")
                    .and_then(|(_, val)| val.as_str())
                {
                    p.grammar = Grammar::regex(pattern).ok();
                }
            }
            _ => {}
        }
    }
    p
}

fn rejection_response(out: &GenerationOutput) -> HttpResponse {
    let reason = out
        .rejection
        .map(|r| r.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    HttpResponse::json(
        400,
        &Value::Object(vec![(
            "error".to_string(),
            Value::String(format!("request rejected: {reason}")),
        )])
        .to_json(),
    )
}

fn generate_response(out: &GenerationOutput) -> Value {
    Value::Object(vec![
        ("text".to_string(), Value::String(out.text.clone())),
        (
            "finish_reason".to_string(),
            Value::String(out.finish_reason.to_string()),
        ),
        (
            "prompt_tokens".to_string(),
            Value::Number(out.prompt_tokens as f64),
        ),
        (
            "prefix_hit_tokens".to_string(),
            Value::Number(out.prefix_hit_tokens as f64),
        ),
        ("forward_count".to_string(), Value::Number(out.forward_count as f64)),
    ])
}

fn chat_response(out: &GenerationOutput) -> Value {
    let finish = match out.finish_reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::Fault => "error",
        FinishReason::Rejected => "error",
    };
    Value::Object(vec![
        ("object".to_string(), Value::String("chat.completion".to_string())),
        (
            "choices".to_string(),
            Value::Array(vec![Value::Object(vec![
                (
                    "message".to_string(),
                    Value::Object(vec![
                        ("role".to_string(), Value::String("assistant".to_string())),
                        ("content".to_string(), Value::String(out.text.clone())),
                    ]),
                ),
                ("finish_reason".to_string(), Value::String(finish.to_string())),
            ])]),
        ),
    ])
}

/// Read one HTTP/1.1 request from a connection.
fn read_http_request(stream: &mut TcpStream) -> std::io::Result<(String, String, String)> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        if header == "\r\n" || header == "\n" || header.is_empty() {
            break;
        }
        let lower = header.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok((method, path, String::from_utf8_lossy(&body).into_owned()))
}

fn write_http_response(stream: &mut TcpStream, resp: &HttpResponse) -> std::io::Result<()> {
    let reason = match resp.status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        resp.status,
        reason,
        resp.body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(resp.body.as_bytes())
}

/// Blocking accept loop. Spawns one thread per connection.
pub fn run<E>(engine: Arc<Mutex<E>>, addr: &str) -> std::io::Result<()>
where
    E: ServingEngine + Send + 'static,
{
    let listener = TcpListener::bind(addr)?;
    let local = listener.local_addr()?;
    eprintln!("pagoda serving on http://{local}");
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let engine = Arc::clone(&engine);
                std::thread::spawn(move || {
                    let _ = stream.set_nodelay(true);
                    let result = read_http_request(&mut stream);
                    let (method, path, body) = match result {
                        Ok(req) => req,
                        Err(_) => {
                            let _ = write_http_response(
                                &mut stream,
                                &HttpResponse::json(400, r#"{"error":"bad request"}"#),
                            );
                            return;
                        }
                    };
                    let response = {
                        let mut guard = engine.lock().unwrap();
                        handle(&mut *guard, &method, &path, Some(&body))
                    };
                    let _ = write_http_response(&mut stream, &response);
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineConfig, ToyEngine};

    #[test]
    fn health_and_generate() {
        let mut engine = ToyEngine::toy(EngineConfig {
            num_kv_blocks: 64,
            block_size: 8,
            ..EngineConfig::default()
        });
        let health = handle(&mut engine, "GET", "/health", None);
        assert_eq!(health.status, 200);

        let resp = handle(
            &mut engine,
            "POST",
            "/generate",
            Some(r#"{"text":"Once upon","sampling_params":{"max_tokens":12}}"#),
        );
        assert_eq!(resp.status, 200);
        let v = json::parse(&resp.body).unwrap();
        assert!(v.get("text").unwrap().as_str().unwrap().len() > 0);
    }

    #[test]
    fn parse_sampling_accepts_grammar() {
        let v = json::parse(
            r#"{"max_tokens":10,"grammar":{"type":"json"}}"#,
        )
        .unwrap();
        assert_eq!(parse_sampling(&v).grammar, Some(Grammar::json()));

        let v = json::parse(
            r#"{"grammar":{"type":"regex","pattern":"\"[a-z]{3}\""}}"#,
        )
        .unwrap();
        assert!(parse_sampling(&v).grammar.is_some());
    }
}

