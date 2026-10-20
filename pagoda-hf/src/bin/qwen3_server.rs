// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Qwen3.8 (Qwen3.5 text tower) behind the OpenAI Responses API Codex uses.
//!
//! ```text
//! PAGODA_DEVICE=metal cargo run --release --features metal --bin qwen3_server -- \
//!   --model ~/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit --port 8000
//! ```
//!
//! Greedy Metal decode. One resident KV: a later request that extends the
//! previous token prefix only feeds the suffix.

use std::io::{Write};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use pagoda::server::{read_http_request, write_http_response, HttpResponse};
use pagoda::{Engine, EngineConfig, SamplingParams, Tokenizer, WriteRequest};
use pagoda_hf::{CandleModel, HfTokenizer, Qwen35, QwenEngine};
use serde_json::{json, Value};

const TOOL_GUIDE: &str = "\
# Tools\n\nYou have access to the following functions:\n\n<tools>";

const TOOL_IMPORTANT: &str = "\
\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n\
value_1\n</parameter>\n<parameter=example_parameter_2>\n\
This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n\
</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
- Required parameters MUST be specified\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
</IMPORTANT>";

/// `text_config.max_position_embeddings` for this Qwen3.8 checkpoint.
const MAX_CONTEXT_TOKENS: usize = 262144;
const MAX_OUTPUT_TOKENS: usize = 32768;
/// Feed up to the model window. Prefill is still ~50 tok/s and streams nothing
/// until the prompt is consumed, so a cold multi-k prompt can sit silent for
/// minutes; that is the trade for keeping system / tool text in context.
const INTERACTIVE_PROMPT_TOKENS: usize = MAX_CONTEXT_TOKENS;

struct Args {
    model_dir: String,
    addr: String,
    port: u16,
    model_id: String,
}

struct Call {
    name: String,
    arguments: Value,
}

struct Msg {
    role: String,
    content: String,
    tool_calls: Vec<Call>,
}

struct Resident {
    ids: Vec<u32>,
}

fn parse_args() -> Args {
    let mut args = Args {
        model_dir: std::env::var("HOME").unwrap_or_default()
            + "/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit",
        addr: "127.0.0.1".to_string(),
        port: 8000,
        model_id: "mlx-community/Qwen3.8-27B-8bit".to_string(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else { break };
        match flag.as_str() {
            "--model" => args.model_dir = value,
            "--addr" => args.addr = value,
            "--port" => args.port = value.parse().expect("--port"),
            "--model-id" => args.model_id = value,
            other => eprintln!("ignoring unknown flag {other}"),
        }
    }
    args
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let mut out = String::new();
            for item in items {
                if let Some(s) = item.get("text").and_then(Value::as_str) {
                    out.push_str(s);
                } else if item.get("image").is_some()
                    || item.get("image_url").is_some()
                    || item.get("type").and_then(Value::as_str) == Some("image")
                {
                    out.push_str("[image]");
                }
            }
            out
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => String::new(),
        other => other.to_string(),
    }
}

fn arguments_object(v: &Value) -> Value {
    match v {
        Value::Object(_) => v.clone(),
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| json!({ "raw": s })),
        other => json!({ "raw": other }),
    }
}

fn push_call(messages: &mut Vec<Msg>, name: &str, arguments: Value) {
    if let Some(last) = messages.last_mut() {
        if last.role == "assistant" {
            last.tool_calls.push(Call {
                name: name.to_string(),
                arguments,
            });
            return;
        }
    }
    messages.push(Msg {
        role: "assistant".into(),
        content: String::new(),
        tool_calls: vec![Call {
            name: name.to_string(),
            arguments,
        }],
    });
}

fn conversation(body: &Value) -> Result<(Vec<Msg>, Vec<Value>)> {
    let mut messages = Vec::new();
    if let Some(instr) = body.get("instructions").and_then(Value::as_str) {
        if !instr.is_empty() {
            messages.push(Msg {
                role: "system".into(),
                content: instr.to_string(),
                tool_calls: Vec::new(),
            });
        }
    }
    match body.get("input") {
        Some(Value::String(s)) => messages.push(Msg {
            role: "user".into(),
            content: s.clone(),
            tool_calls: Vec::new(),
        }),
        Some(Value::Array(items)) => {
            for item in items {
                let kind = item.get("type").and_then(Value::as_str).unwrap_or("message");
                match kind {
                    "message" => {
                        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                        let content = item
                            .get("content")
                            .map(text_of)
                            .unwrap_or_default();
                        if role == "system" {
                            if let Some(first) = messages.first_mut() {
                                if first.role == "system" {
                                    if !first.content.is_empty() && !content.is_empty() {
                                        first.content.push_str("\n\n");
                                    }
                                    first.content.push_str(&content);
                                    continue;
                                }
                            }
                        }
                        messages.push(Msg {
                            role: role.to_string(),
                            content,
                            tool_calls: Vec::new(),
                        });
                    }
                    "function_call" => {
                        let name = item.get("name").and_then(Value::as_str).unwrap_or("tool");
                        let args = item.get("arguments").cloned().unwrap_or(Value::Null);
                        push_call(&mut messages, name, arguments_object(&args));
                    }
                    "function_call_output" => {
                        let content = if let Some(c) = item.get("output") {
                            text_of(c)
                        } else {
                            item.get("content").map(text_of).unwrap_or_default()
                        };
                        messages.push(Msg {
                            role: "tool".into(),
                            content,
                            tool_calls: Vec::new(),
                        });
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    if !messages.iter().any(|m| m.role == "user" || m.role == "tool") {
        bail!("no user message");
    }
    let tools = body
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok((messages, tools))
}

fn reasoning_enabled(req: &Value) -> bool {
    if let Some(flag) = req.get("enable_thinking").and_then(Value::as_bool) {
        return flag;
    }
    let effort = req
        .get("reasoning")
        .and_then(|v| v.get("effort"))
        .and_then(Value::as_str)
        .or_else(|| req.get("reasoning_effort").and_then(Value::as_str));
    match effort {
        // Explicit off. Codex++ sometimes sends `minimal` for the same intent.
        Some("none") | Some("off") | Some("minimal") => false,
        // Open `<think>` for low/medium/high (and any other effort). Streamed
        // answer text still waits for `</think>`; reasoning lands in the final
        // response `reasoning` item.
        Some(_) => true,
        // Qwen3.8 is a thinking model. Codex++ omits the field on some turns;
        // leave the think block open unless the client explicitly turns it off.
        None => true,
    }
}

fn render(messages: &[Msg], tools: &[Value], think: bool) -> String {
    let mut out = String::new();
    let system = messages.first().filter(|m| m.role == "system");
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(TOOL_GUIDE);
        for tool in tools {
            out.push('\n');
            out.push_str(&tool.to_string());
        }
        out.push_str("\n</tools>");
        out.push_str(TOOL_IMPORTANT);
        if let Some(sys) = system {
            let content = sys.content.trim();
            if !content.is_empty() {
                out.push_str("\n\n");
                out.push_str(content);
            }
        }
        out.push_str("<|im_end|>\n");
    } else if let Some(sys) = system {
        let content = sys.content.trim();
        if !content.is_empty() {
            out.push_str("<|im_start|>system\n");
            out.push_str(content);
            out.push_str("<|im_end|>\n");
        }
    }

    let mut i = 0;
    while i < messages.len() {
        let message = &messages[i];
        if message.role == "system" {
            i += 1;
            continue;
        }
        if message.role == "tool" {
            out.push_str("<|im_start|>user");
            while i < messages.len() && messages[i].role == "tool" {
                out.push_str("\n<tool_response>\n");
                out.push_str(messages[i].content.trim());
                out.push_str("\n</tool_response>");
                i += 1;
            }
            out.push_str("<|im_end|>\n");
            continue;
        }
        if message.role == "assistant" {
            out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
            out.push_str(message.content.trim());
            for (n, call) in message.tool_calls.iter().enumerate() {
                if n == 0 {
                    if !message.content.trim().is_empty() {
                        out.push_str("\n\n");
                    }
                } else {
                    out.push('\n');
                }
                out.push_str("<tool_call>\n<function=");
                out.push_str(&call.name);
                out.push_str(">\n");
                if let Value::Object(map) = &call.arguments {
                    for (key, value) in map {
                        out.push_str("<parameter=");
                        out.push_str(key);
                        out.push_str(">\n");
                        match value {
                            Value::String(s) => out.push_str(s),
                            other => out.push_str(&other.to_string()),
                        }
                        out.push_str("\n</parameter>\n");
                    }
                }
                out.push_str("</function>\n</tool_call>");
            }
            out.push_str("<|im_end|>\n");
            i += 1;
            continue;
        }
        out.push_str("<|im_start|>");
        out.push_str(&message.role);
        out.push('\n');
        out.push_str(message.content.trim());
        out.push_str("<|im_end|>\n");
        i += 1;
    }
    if think {
        out.push_str("<|im_start|>assistant\n<think>\n");
    } else {
        out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }
    out
}

struct Parsed {
    text: String,
    reasoning: String,
    calls: Vec<Call>,
}

fn param_value(raw: &str) -> Value {
    let t = raw.trim();
    let jsonish = t.starts_with('{')
        || t.starts_with('[')
        || t.starts_with('"')
        || t == "true"
        || t == "false"
        || t == "null"
        || t.parse::<f64>().is_ok();
    if jsonish {
        if let Ok(v) = serde_json::from_str::<Value>(t) {
            return v;
        }
    }
    Value::String(t.to_string())
}

fn parse_generation(raw: &str) -> Parsed {
    let mut text = raw.to_string();
    let mut reasoning = String::new();
    if let Some(end) = text.find("</think>") {
        reasoning = text[..end]
            .trim()
            .trim_start_matches("<think>")
            .trim()
            .to_string();
        text = text[end + "</think>".len()..]
            .trim_start_matches('\n')
            .to_string();
    } else if let Some(start) = text.find("<think>") {
        reasoning = text[start + "<think>".len()..].trim().to_string();
        text.clear();
    }
    let mut calls = Vec::new();
    let mut visible = String::new();
    let mut rest = text.as_str();
    while let Some(at) = rest.find("<tool_call>") {
        visible.push_str(&rest[..at]);
        rest = &rest[at + "<tool_call>".len()..];
        let block_end = rest.find("</tool_call>").unwrap_or(rest.len());
        let block = &rest[..block_end];
        if let Some(name_at) = block.find("<function=") {
            let after = &block[name_at + "<function=".len()..];
            let name_end = after.find('>').unwrap_or(after.len());
            let name = after[..name_end].trim().to_string();
            let mut args = serde_json::Map::new();
            let mut scan = &after[name_end.min(after.len())..];
            if scan.starts_with('>') {
                scan = &scan[1..];
            }
            while let Some(p) = scan.find("<parameter=") {
                scan = &scan[p + "<parameter=".len()..];
                let key_end = scan.find('>').unwrap_or(scan.len());
                let key = scan[..key_end].trim().to_string();
                scan = &scan[key_end.min(scan.len())..];
                if scan.starts_with('>') {
                    scan = &scan[1..];
                }
                let value_end = scan.find("</parameter>").unwrap_or(scan.len());
                let value = param_value(&scan[..value_end]);
                if !key.is_empty() {
                    args.insert(key, value);
                }
                scan = &scan[value_end.min(scan.len())..];
                if scan.starts_with("</parameter>") {
                    scan = &scan["</parameter>".len()..];
                }
            }
            if !name.is_empty() {
                calls.push(Call {
                    name,
                    arguments: Value::Object(args),
                });
            }
        }
        rest = if block_end + "</tool_call>".len() <= rest.len() && rest[block_end..].starts_with("</tool_call>")
        {
            &rest[block_end + "</tool_call>".len()..]
        } else {
            ""
        };
    }
    visible.push_str(rest);
    Parsed {
        text: visible.trim().to_string(),
        reasoning,
        calls,
    }
}

fn rfind_ids(hay: &[u32], needle: &[u32]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    let mut i = hay.len() - needle.len();
    loop {
        if &hay[i..i + needle.len()] == needle {
            return Some(i);
        }
        if i == 0 {
            return None;
        }
        i -= 1;
    }
}

fn take_tail(ids: Vec<u32>, budget: usize) -> (Vec<u32>, usize) {
    if budget == 0 || ids.len() <= budget {
        return (ids, 0);
    }
    let dropped = ids.len() - budget;
    let tail = ids[dropped..].to_vec();
    (tail, dropped)
}

fn fit_encoded(run: &ArgsRun<'_>, encoded: Vec<u32>) -> Vec<u32> {
    if let Some(prev) = run.resident.as_ref() {
        if let Some(at) = rfind_ids(&encoded, &prev.ids) {
            let suffix = encoded.len() - at - prev.ids.len();
            if suffix > 0 && suffix <= INTERACTIVE_PROMPT_TOKENS {
                eprintln!(
                    "==> extend cached {} plus {suffix} new tokens",
                    prev.ids.len()
                );
                return encoded[at..].to_vec();
            }
        }
    }
    let (ids, dropped) = take_tail(encoded, INTERACTIVE_PROMPT_TOKENS);
    if dropped > 0 {
        eprintln!(
            "==> trim {dropped} older prompt tokens; feeding last {}",
            ids.len()
        );
    }
    ids
}

fn context_limit(prompt_tokens: usize, max_output: usize) -> Result<()> {
    if prompt_tokens.saturating_add(max_output) > MAX_CONTEXT_TOKENS {
        bail!(
            "prompt is {prompt_tokens} tokens plus {max_output} output; pagoda qwen3.8 window is {MAX_CONTEXT_TOKENS}"
        );
    }
    Ok(())
}

fn generate(
    engine: &mut Engine<QwenEngine, HfTokenizer>,
    resident: &mut Option<Resident>,
    prompt_ids: &[u32],
    max_tokens: usize,
    mut on_text: impl FnMut(&str) -> bool,
) -> Result<Vec<u32>> {
    if prompt_ids.is_empty() {
        bail!("empty prompt");
    }
    let req = WriteRequest {
        text: String::new(),
        sampling: SamplingParams {
            max_tokens,
            temperature: 0.0,
            stop: vec!["<|im_end|>".into(), "<|endoftext|>".into()],
            stop_token_ids: vec![248046],
            ..SamplingParams::default()
        },
        prompt_tokens: Some(prompt_ids.to_vec()),
    };
    let mut acc = String::new();
    let mut stop = false;
    let out = engine.generate_streaming(&req, &mut |_id, piece| {
        if stop {
            return false;
        }
        acc.push_str(piece);
        if !on_text(&acc) {
            stop = true;
            return false;
        }
        true
    });
    if let Some(reason) = out.rejection {
        bail!("rejected: {reason:?}");
    }
    let produced = out.output_token_ids;
    let mut ids = prompt_ids.to_vec();
    ids.extend_from_slice(&produced);
    *resident = Some(Resident { ids });
    Ok(produced)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn response_json(
    id: &str,
    model: &str,
    previous: Option<&str>,
    output: Vec<Value>,
    prompt_tokens: usize,
    output_tokens: usize,
    reasoning_tokens: usize,
    max_output_tokens: usize,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now_secs(),
        "completed_at": now_secs(),
        "status": "completed",
        "incomplete_details": null,
        "model": model,
        "previous_response_id": previous,
        "instructions": null,
        "output": output,
        "error": null,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "disabled",
        "parallel_tool_calls": true,
        "text": {"format": {"type": "text"}},
        "temperature": 0,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens}
        },
        "max_output_tokens": max_output_tokens,
        "store": false
    })
}

fn output_items(reasoning: &str, text: &str, calls: &[Call]) -> Vec<Value> {
    let mut items = Vec::new();
    if !reasoning.is_empty() {
        items.push(json!({
            "type": "reasoning",
            "id": format!("rs_{}", now_secs()),
            "summary": [{"type": "summary_text", "text": reasoning}],
            "content": [{"type": "reasoning_text", "text": reasoning}]
        }));
    }
    if !text.is_empty() {
        items.push(json!({
            "type": "message",
            "id": format!("msg_{}", now_secs()),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]
        }));
    }
    for (i, call) in calls.iter().enumerate() {
        let args = call.arguments.to_string();
        items.push(json!({
            "type": "function_call",
            "id": format!("fc_{}_{i}", now_secs()),
            "call_id": format!("call_{}_{i}", now_secs()),
            "name": call.name,
            "arguments": args,
            "status": "completed"
        }));
    }
    if items.is_empty() {
        items.push(json!({
            "type": "message",
            "id": format!("msg_{}", now_secs()),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "", "annotations": []}]
        }));
    }
    items
}

fn write_chunk(stream: &mut TcpStream, payload: &[u8]) -> std::io::Result<()> {
    stream.write_all(format!("{:x}\r\n", payload.len()).as_bytes())?;
    stream.write_all(payload)?;
    stream.write_all(b"\r\n")
}

fn sse(stream: &mut TcpStream, event: &str, data: &Value) -> std::io::Result<()> {
    let payload = format!("event: {event}\ndata: {data}\n\n");
    write_chunk(stream, payload.as_bytes())?;
    stream.flush()
}

struct ArgsRun<'a> {
    engine: &'a mut Engine<QwenEngine, HfTokenizer>,
    tokenizer: &'a HfTokenizer,
    resident: &'a mut Option<Resident>,
    model_id: &'a str,
}

fn handle(stream: &mut TcpStream, run: &mut ArgsRun<'_>) -> Result<()> {
    let (method, path, body) = read_http_request(stream).context("read request")?;
    let path = path.split('?').next().unwrap_or(path.as_str());
    match (method.as_str(), path) {
        ("GET", "/health") => {
            write_http_response(stream, &HttpResponse::json(200, r#"{"status":"ok","model":"qwen3.8"}"#))?;
        }
        ("GET", "/v1/models") => {
            let body = json!({
                "object": "list",
                "data": [{
                    "id": run.model_id,
                    "object": "model",
                    "owned_by": "pagoda",
                    "loaded": true
                }]
            });
            write_http_response(stream, &HttpResponse::json(200, &body.to_string()))?;
        }
        ("POST", "/v1/responses") => respond(stream, run, &body)?,
        ("POST", "/v1/chat/completions") => chat_complete(stream, run, &body)?,
        ("POST", "/v1/completions") => text_complete(stream, run, &body)?,
        _ => {
            write_http_response(stream, &HttpResponse::json(404, r#"{"error":"not found"}"#))?;
        }
    }
    Ok(())
}

fn chat_messages(body: &Value) -> Result<Vec<Msg>> {
    let items = body
        .get("messages")
        .and_then(Value::as_array)
        .context("messages")?;
    let mut messages = Vec::new();
    for item in items {
        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = item.get("content").map(text_of).unwrap_or_default();
        messages.push(Msg {
            role: role.to_string(),
            content,
            tool_calls: Vec::new(),
        });
    }
    if messages.iter().all(|m| m.content.trim().is_empty()) {
        bail!("empty messages");
    }
    Ok(messages)
}

fn complete_text(
    run: &mut ArgsRun<'_>,
    prompt: &str,
    max_tokens: usize,
) -> Result<(Vec<u32>, Vec<u32>, String)> {
    let prompt_ids = fit_encoded(run, run.tokenizer.encode(prompt));
    if prompt_ids.is_empty() {
        bail!("empty prompt");
    }
    context_limit(prompt_ids.len(), max_tokens)?;
    let produced = generate(
        run.engine,
        run.resident,
        &prompt_ids,
        max_tokens,
        |_| true,
    )?;
    let text = run.tokenizer.decode(&produced);
    Ok((prompt_ids, produced, text))
}

fn chat_stream(
    stream: &mut TcpStream,
    run: &mut ArgsRun<'_>,
    prompt: &str,
    max_tokens: usize,
    model: &str,
) -> Result<()> {
    let prompt_ids = fit_encoded(run, run.tokenizer.encode(prompt));
    if prompt_ids.is_empty() {
        write_http_response(stream, &HttpResponse::json(400, r#"{"error":{"message":"empty prompt"}}"#))?;
        return Ok(());
    }
    if let Err(e) = context_limit(prompt_ids.len(), max_tokens) {
        let err = json!({"error": {"message": e.to_string()}}).to_string();
        write_http_response(stream, &HttpResponse::json(400, &err))?;
        return Ok(());
    }
    let id = format!("chatcmpl_{}", now_secs());
    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    stream.write_all(head.as_bytes())?;
    let role = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now_secs(),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    });
    write_chunk(stream, format!("data: {role}\n\n").as_bytes())?;
    stream.flush()?;
    let mut sent = String::new();
    let produced = generate(
        run.engine,
        run.resident,
        &prompt_ids,
        max_tokens,
        |decoded| {
            if decoded.len() > sent.len() && decoded.starts_with(&sent) {
                let delta = &decoded[sent.len()..];
                let chunk = json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": now_secs(),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": null}]
                });
                if write_chunk(stream, format!("data: {chunk}\n\n").as_bytes()).is_err() {
                    return false;
                }
                let _ = stream.flush();
                sent = decoded.to_string();
            }
            true
        },
    );
    let finish = match &produced {
        Ok(tokens) if tokens.len() >= max_tokens => "length",
        Ok(_) => "stop",
        Err(_) => "stop",
    };
    let done = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now_secs(),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]
    });
    let _ = write_chunk(stream, format!("data: {done}\n\n").as_bytes());
    let _ = write_chunk(stream, b"data: [DONE]\n\n");
    let _ = write_chunk(stream, b"");
    Ok(())
}

fn chat_complete(stream: &mut TcpStream, run: &mut ArgsRun<'_>, body: &str) -> Result<()> {
    let req: Value = serde_json::from_str(body).context("json")?;
    let messages = match chat_messages(&req) {
        Ok(v) => v,
        Err(e) => {
            let err = json!({"error": {"message": e.to_string()}}).to_string();
            write_http_response(stream, &HttpResponse::json(400, &err))?;
            return Ok(());
        }
    };
    let prompt = render(&messages, &[], reasoning_enabled(&req));
    let max_tokens = req
        .get("max_tokens")
        .or_else(|| req.get("max_completion_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(512)
        .min(MAX_OUTPUT_TOKENS as u64) as usize;
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(run.model_id);
    if req.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        return chat_stream(stream, run, &prompt, max_tokens, model);
    }
    let (prompt_ids, produced, text) = match complete_text(run, &prompt, max_tokens) {
        Ok(v) => v,
        Err(e) => {
            let err = json!({"error": {"message": e.to_string()}}).to_string();
            write_http_response(stream, &HttpResponse::json(400, &err))?;
            return Ok(());
        }
    };
    let finish = if produced.len() >= max_tokens { "length" } else { "stop" };
    let resp = json!({
        "id": format!("chatcmpl_{}", now_secs()),
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish
        }],
        "usage": {
            "prompt_tokens": prompt_ids.len(),
            "completion_tokens": produced.len(),
            "total_tokens": prompt_ids.len() + produced.len()
        }
    });
    write_http_response(stream, &HttpResponse::json(200, &resp.to_string()))?;
    Ok(())
}

fn text_complete(stream: &mut TcpStream, run: &mut ArgsRun<'_>, body: &str) -> Result<()> {
    let req: Value = serde_json::from_str(body).context("json")?;
    let Some(prompt) = req.get("prompt").and_then(Value::as_str) else {
        write_http_response(stream, &HttpResponse::json(400, r#"{"error":{"message":"prompt required"}}"#))?;
        return Ok(());
    };
    let max_tokens = req
        .get("max_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(512)
        .min(MAX_OUTPUT_TOKENS as u64) as usize;
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(run.model_id);
    let (prompt_ids, produced, text) = match complete_text(run, prompt, max_tokens) {
        Ok(v) => v,
        Err(e) => {
            let err = json!({"error": {"message": e.to_string()}}).to_string();
            write_http_response(stream, &HttpResponse::json(400, &err))?;
            return Ok(());
        }
    };
    let finish = if produced.len() >= max_tokens { "length" } else { "stop" };
    let resp = json!({
        "id": format!("cmpl_{}", now_secs()),
        "object": "text_completion",
        "created": now_secs(),
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": prompt_ids.len(),
            "completion_tokens": produced.len(),
            "total_tokens": prompt_ids.len() + produced.len()
        }
    });
    write_http_response(stream, &HttpResponse::json(200, &resp.to_string()))?;
    Ok(())
}

fn respond(stream: &mut TcpStream, run: &mut ArgsRun<'_>, body: &str) -> Result<()> {
    let req: Value = serde_json::from_str(body).context("json")?;
    let (messages, tools) = match conversation(&req) {
        Ok(v) => v,
        Err(e) => {
            let err = json!({"error": {"message": e.to_string()}}).to_string();
            write_http_response(stream, &HttpResponse::json(400, &err))?;
            return Ok(());
        }
    };
    let think = reasoning_enabled(&req);
    let prompt = render(&messages, &tools, think);
    let prompt_ids = fit_encoded(run, run.tokenizer.encode(&prompt));
    if prompt_ids.is_empty() {
        write_http_response(stream, &HttpResponse::json(400, r#"{"error":"empty prompt"}"#))?;
        return Ok(());
    }
    let max_tokens_preview = req
        .get("max_output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(512)
        .min(MAX_OUTPUT_TOKENS as u64) as usize;
    if let Err(e) = context_limit(prompt_ids.len(), max_tokens_preview) {
        let err = json!({"error": {"message": e.to_string()}}).to_string();
        write_http_response(stream, &HttpResponse::json(400, &err))?;
        return Ok(());
    }
    let max_tokens = req
        .get("max_output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(512)
        .min(MAX_OUTPUT_TOKENS as u64) as usize;
    let stream_wanted = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let requested_model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(run.model_id)
        .to_string();
    let previous = req
        .get("previous_response_id")
        .and_then(Value::as_str)
        .map(str::to_string);
    let id = format!("resp_{}", now_secs());
    eprintln!(
        "==> {id} prompt_tokens {} max_output {max_tokens} stream {stream_wanted}",
        prompt_ids.len()
    );

    if !stream_wanted {
        let produced = generate(
            run.engine,
            run.resident,
            &prompt_ids,
            max_tokens,
            |_| true,
        )?;
        let raw = run.tokenizer.decode(&produced);
        let parsed = parse_generation(&raw);
        let reasoning_tokens = if parsed.reasoning.is_empty() {
            0
        } else {
            run.tokenizer.encode(&parsed.reasoning).len()
        };
        let output = output_items(&parsed.reasoning, &parsed.text, &parsed.calls);
        let resp = response_json(
            &id,
            &requested_model,
            previous.as_deref(),
            output,
            prompt_ids.len(),
            produced.len(),
            reasoning_tokens,
            max_tokens,
        );
        write_http_response(stream, &HttpResponse::json(200, &resp.to_string()))?;
        eprintln!("==> {id} done {} tokens", produced.len());
        return Ok(());
    }

    let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";
    stream.write_all(head.as_bytes())?;
    let skeleton = response_json(
        &id,
        &requested_model,
        previous.as_deref(),
        Vec::new(),
        prompt_ids.len(),
        0,
        0,
        max_tokens,
    );
    sse(stream, "response.created", &json!({"type":"response.created","response": skeleton}))?;
    sse(stream, "response.in_progress", &json!({"type":"response.in_progress","response": skeleton}))?;

    let mut sent = String::new();
    let mut in_tool = false;
    let msg_id = format!("msg_{}", now_secs());
    let mut opened = false;
    let gen = generate(
        run.engine,
        run.resident,
        &prompt_ids,
        max_tokens,
        |decoded| {
            let (visible, tool) = split_visible(decoded);
            if tool {
                in_tool = true;
            }
            if !in_tool && visible.len() > sent.len() && visible.starts_with(&sent) {
                let delta = visible[sent.len()..].to_string();
                if !opened {
                    let _ = sse(stream, "response.output_item.added", &json!({
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {"type":"message","id": msg_id, "role":"assistant","status":"in_progress","content":[]}
                    }));
                    let _ = sse(stream, "response.content_part.added", &json!({
                        "type": "response.content_part.added",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type":"output_text","text":"","annotations":[]}
                    }));
                    opened = true;
                }
                let _ = sse(stream, "response.output_text.delta", &json!({
                    "type": "response.output_text.delta",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta
                }));
                sent = visible;
            }
            let _ = stream.flush();
            true
        },
    );
    let produced = match gen {
        Ok(v) => v,
        Err(e) => {
            let _ = sse(stream, "error", &json!({"type":"error","message": e.to_string()}));
            write_chunk(stream, b"data: [DONE]\n\n")?;
            write_chunk(stream, b"")?;
            return Ok(());
        }
    };
    let raw = run.tokenizer.decode(&produced);
    let parsed = parse_generation(&raw);
    if opened {
        let _ = sse(stream, "response.output_text.done", &json!({
            "type": "response.output_text.done",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "text": parsed.text
        }));
        let _ = sse(stream, "response.output_item.done", &json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type":"message","id": msg_id, "role":"assistant","status":"completed",
                "content":[{"type":"output_text","text": parsed.text, "annotations":[]}]
            }
        }));
    }
    let mut index = if opened { 1 } else { 0 };
    for call in &parsed.calls {
        let args = call.arguments.to_string();
        let fc_id = format!("fc_{index}");
        let call_id = format!("call_{}_{index}", now_secs());
        let _ = sse(stream, "response.output_item.added", &json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": {
                "type":"function_call","id": fc_id, "call_id": call_id,
                "name": call.name, "arguments": "", "status": "in_progress"
            }
        }));
        let _ = sse(stream, "response.function_call_arguments.delta", &json!({
            "type": "response.function_call_arguments.delta",
            "item_id": fc_id,
            "output_index": index,
            "delta": args
        }));
        let _ = sse(stream, "response.function_call_arguments.done", &json!({
            "type": "response.function_call_arguments.done",
            "item_id": fc_id,
            "output_index": index,
            "arguments": args
        }));
        let _ = sse(stream, "response.output_item.done", &json!({
            "type": "response.output_item.done",
            "output_index": index,
            "item": {
                "type":"function_call","id": fc_id, "call_id": call_id,
                "name": call.name, "arguments": args, "status": "completed"
            }
        }));
        index += 1;
    }
    let reasoning_tokens = if parsed.reasoning.is_empty() {
        0
    } else {
        run.tokenizer.encode(&parsed.reasoning).len()
    };
    let output = output_items(&parsed.reasoning, &parsed.text, &parsed.calls);
    let done = response_json(
        &id,
        &requested_model,
        previous.as_deref(),
        output,
        prompt_ids.len(),
        produced.len(),
        reasoning_tokens,
        max_tokens,
    );
    sse(stream, "response.completed", &json!({"type":"response.completed","response": done}))?;
    write_chunk(stream, b"data: [DONE]\n\n")?;
    write_chunk(stream, b"")?;
    eprintln!("==> {id} done {} tokens", produced.len());
    Ok(())
}

/// Text before a tool call, holding back a partial `<tool_call>` prefix.
fn split_visible(decoded: &str) -> (String, bool) {
    let parsed_src = if let Some(start) = decoded.find("<think>") {
        if let Some(end) = decoded[start..].find("</think>") {
            decoded[start + end + "</think>".len()..].trim_start_matches('\n')
        } else {
            return (String::new(), false);
        }
    } else {
        decoded
    };
    if let Some(at) = parsed_src.find("<tool_call>") {
        return (parsed_src[..at].to_string(), true);
    }
    const MARK: &str = "<tool_call>";
    for n in (1..MARK.len()).rev() {
        if parsed_src.ends_with(&MARK[..n]) && parsed_src.is_char_boundary(parsed_src.len() - n) {
            return (parsed_src[..parsed_src.len() - n].to_string(), false);
        }
    }
    (parsed_src.to_string(), false)
}

fn main() -> Result<()> {
    let args = parse_args();
    let device = CandleModel::device_from_env()?;
    let verify = cfg!(feature = "metal") && !device.is_cpu();
    eprintln!("==> device {device:?} verify_prefill {verify}");
    eprintln!("==> load {}", args.model_dir);
    let tokenizer = HfTokenizer::from_file(format!("{}/tokenizer.json", args.model_dir))?;
    let model = std::sync::Arc::new(Qwen35::load(&args.model_dir, device)?);
    eprintln!("==> warmup");
    {
        let mut cache = model.new_cache();
        let _ = model.forward_greedy(&[tokenizer.encode("Hi")[0]], &mut cache)?;
    }
    let engine_tokenizer = HfTokenizer::from_file(format!("{}/tokenizer.json", args.model_dir))?;
    let mut engine = Engine::new(
        engine_tokenizer,
        QwenEngine::new(model),
        EngineConfig {
            // Bookkeeping pages for token ids (real KV lives in QwenSession).
            // Need ceiling(MAX_CONTEXT / block_size) to admit a full window.
            num_kv_blocks: MAX_CONTEXT_TOKENS.div_ceil(16),
            block_size: 16,
            max_running_requests: 1,
            max_prefill_tokens_per_step: 256,
            max_total_tokens: MAX_CONTEXT_TOKENS,
            max_waiting_requests: 4,
            ..EngineConfig::default()
        },
    );
    let _ = verify;
    let mut resident = None;
    let listener = TcpListener::bind(format!("{}:{}", args.addr, args.port))?;
    eprintln!(
        "qwen3_server listening on http://{}:{} model {} (engine)",
        args.addr, args.port, args.model_id
    );
    for conn in listener.incoming() {
        let mut stream = conn?;
        let _ = stream.set_nodelay(true);
        let mut run = ArgsRun {
            engine: &mut engine,
            tokenizer: &tokenizer,
            resident: &mut resident,
            model_id: &args.model_id,
        };
        if let Err(e) = handle(&mut stream, &mut run) {
            eprintln!("request error: {e:#}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_user_closes_think_before_generation() {
        let messages = vec![Msg {
            role: "user".into(),
            content: "ping".into(),
            tool_calls: Vec::new(),
        }];
        let closed = render(&messages, &[], false);
        assert!(closed.contains("<|im_start|>user\nping<|im_end|>\n"));
        assert!(closed.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        let open = render(&messages, &[], true);
        assert!(open.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(!open.ends_with("</think>\n\n"));
    }

    #[test]
    fn reasoning_effort_opens_cot_except_none() {
        // Omitted effort stays open: Codex++ drops the field on some turns.
        assert!(reasoning_enabled(&json!({})));
        assert!(!reasoning_enabled(&json!({"reasoning": {"effort": "none"}})));
        assert!(!reasoning_enabled(&json!({"reasoning_effort": "off"})));
        assert!(!reasoning_enabled(&json!({"reasoning": {"effort": "minimal"}})));
        assert!(reasoning_enabled(&json!({"reasoning": {"effort": "low"}})));
        assert!(reasoning_enabled(&json!({"reasoning": {"effort": "medium"}})));
        assert!(reasoning_enabled(&json!({"reasoning": {"effort": "high"}})));
        assert!(reasoning_enabled(&json!({"enable_thinking": true})));
        assert!(!reasoning_enabled(&json!({"enable_thinking": false, "reasoning": {"effort": "high"}})));
    }

    #[test]
    fn fit_encoded_keeps_full_prompt_within_window() {
        let ids: Vec<u32> = (0..512).collect();
        let (tail, dropped) = take_tail(ids, INTERACTIVE_PROMPT_TOKENS);
        assert_eq!(dropped, 0);
        assert_eq!(tail.len(), 512);
        assert_eq!(INTERACTIVE_PROMPT_TOKENS, MAX_CONTEXT_TOKENS);
    }

    #[test]
    fn rfind_ids_picks_the_latest_copy() {
        let hay = [1, 2, 9, 9, 1, 2, 3];
        assert_eq!(rfind_ids(&hay, &[1, 2]), Some(4));
        assert_eq!(rfind_ids(&hay, &[9, 9, 1]), Some(2));
        assert_eq!(rfind_ids(&hay, &[7]), None);
    }

    #[test]
    fn take_tail_keeps_the_latest_tokens() {
        let ids: Vec<u32> = (0..10).collect();
        let (tail, dropped) = take_tail(ids.clone(), 4);
        assert_eq!(dropped, 6);
        assert_eq!(tail, vec![6, 7, 8, 9]);
        let (same, none) = take_tail(ids, 10);
        assert_eq!(none, 0);
        assert_eq!(same.len(), 10);
    }

    #[test]
    fn parse_splits_reasoning_from_answer() {
        let parsed = parse_generation("先想一步\n</think>\n\n答案");
        assert_eq!(parsed.reasoning, "先想一步");
        assert_eq!(parsed.text, "答案");
    }

    #[test]
    fn parse_tool_call_parameters() {
        let raw = "listing\n\n<tool_call>\n<function=shell>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
        let parsed = parse_generation(raw);
        assert_eq!(parsed.text, "listing");
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "shell");
        assert_eq!(parsed.calls[0].arguments["command"], json!("ls"));
    }

    #[test]
    fn function_output_becomes_tool_response() {
        let body = json!({
            "instructions": "be brief",
            "input": [
                {"type":"message","role":"user","content":"list"},
                {"type":"function_call","name":"shell","arguments":"{\"command\":\"ls\"}"},
                {"type":"function_call_output","call_id":"c1","output":"a.txt"}
            ],
            "tools": [{"type":"function","name":"shell"}]
        });
        let (messages, tools) = conversation(&body).unwrap();
        let prompt = render(&messages, &tools, false);
        assert!(prompt.contains("<tool_response>\na.txt\n</tool_response>"));
        assert!(prompt.contains("<function=shell>"));
        assert!(prompt.contains("\"name\":\"shell\"") || prompt.contains("\"name\": \"shell\""));
        assert_eq!(messages.len(), 4);
    }
}
