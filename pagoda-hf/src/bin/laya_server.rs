// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Standalone Laya decision server — the one-binary Rust deployment of
//! `convaiinnovations/laya` (no Python, no PyTorch, no CUDA toolkit needed
//! on CPU).
//!
//! ```text
//! cargo run --release --bin laya_server -- --port 8081
//! curl -X POST http://127.0.0.1:8081/decide -H "Content-Type: application/json" -d '{
//!   "state": "Hi, we were billed twice for March. Please refund the duplicate.",
//!   "questions": {
//!     "department": {"type": "choice", "instructions": "Which department?",
//!                    "criteria": {"billing": "invoices, refunds", "technical": "bugs, outages"}},
//!     "churn_risk": {"type": "noul", "instructions": "Does the user threaten to leave?"}
//!   }
//! }'
//! ```
//!
//! Endpoints:
//! * `GET  /health` — liveness probe
//! * `POST /decide` — Jev-compatible typed decisions (choice / score / noul)

use std::net::TcpListener;
use std::sync::Arc;

use anyhow::{Context, Result};
use pagoda::server::{read_http_request, write_http_response, HttpResponse};
use pagoda_hf::{Laya, QType, Question};
use serde_json::{json, Map, Value};

struct Args {
    addr: String,
    port: u16,
    repo: String,
}

fn parse_args() -> Args {
    let mut args = Args {
        addr: "127.0.0.1".to_string(),
        port: 8081,
        repo: "convaiinnovations/laya".to_string(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else { break };
        match flag.as_str() {
            "--addr" => args.addr = value,
            "--port" => args.port = value.parse().expect("--port"),
            "--repo" => args.repo = value,
            other => eprintln!("ignoring unknown flag {other}"),
        }
    }
    args
}

/// Jev request shape -> pagoda Question. Choice criteria arrive as a JSON
/// object (insertion order = label order), score criteria as an array.
fn parse_question(v: &Value) -> Result<Question> {
    let qtype = v
        .get("type")
        .and_then(Value::as_str)
        .context("question missing \"type\"")?;
    let ins = v
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    match qtype {
        "choice" => {
            let crit = v
                .get("criteria")
                .and_then(Value::as_object)
                .context("choice question needs object \"criteria\"")?;
            let pairs: Vec<(&str, &str)> = crit
                .iter()
                .map(|(k, d)| (k.as_str(), d.as_str().unwrap_or("")))
                .collect();
            if pairs.is_empty() {
                anyhow::bail!("choice question needs at least one criterion");
            }
            Ok(Question::choice(&ins, &pairs))
        }
        "score" => {
            let crit = v
                .get("criteria")
                .and_then(Value::as_array)
                .context("score question needs array \"criteria\"")?;
            let items: Vec<&str> = crit.iter().filter_map(Value::as_str).collect();
            if items.len() < 2 {
                anyhow::bail!("score question needs at least two levels");
            }
            Ok(Question::score(&ins, &items))
        }
        "noul" => Ok(Question::noul(&ins)),
        other => anyhow::bail!("unknown question type {other:?} (want choice|score|noul)"),
    }
}

fn answer_json(a: &pagoda_hf::Answer, q: &Question) -> Value {
    // Round in f64 like Python's round(x, 4): f32 rounding leaves artifacts
    // such as 0.9865000247955322 in the JSON output.
    let round4 = |x: f32| ((x as f64) * 10000.0).round() / 10000.0;
    let probs: Map<String, Value> = a
        .probabilities
        .iter()
        .map(|(k, p)| (k.clone(), json!(round4(*p))))
        .collect();
    let mut out = Map::new();
    out.insert("type".into(), json!(q.qtype.name()));
    match q.qtype {
        QType::Choice => {
            out.insert("choice".into(), json!(a.choice));
            out.insert("probabilities".into(), Value::Object(probs));
        }
        QType::Score => {
            out.insert("score".into(), json!(a.score.map(round4)));
            let legend: Map<String, Value> = q
                .options
                .iter()
                .enumerate()
                .map(|(i, (_, c))| (i.to_string(), json!(c)))
                .collect();
            out.insert("legend".into(), Value::Object(legend));
            out.insert("probabilities".into(), Value::Object(probs));
        }
        QType::Noul => {
            out.insert("noul".into(), json!(a.noul.map(round4)));
        }
    }
    out.insert("confidence".into(), json!(round4(a.confidence)));
    out.insert(
        "rl_agent".into(),
        json!({ "act_probability": round4(a.act_probability) }),
    );
    Value::Object(out)
}

fn decide(laya: &Laya, body: &str) -> HttpResponse {
    let result = (|| -> Result<Value> {
        let req: Value = serde_json::from_str(body).context("invalid JSON body")?;
        let state = req
            .get("state")
            .and_then(Value::as_str)
            .context("missing \"state\" string")?;
        let questions = req
            .get("questions")
            .and_then(Value::as_object)
            .context("missing \"questions\" object")?;
        let mut qs = Vec::with_capacity(questions.len());
        for (qid, qdef) in questions {
            qs.push((qid.clone(), parse_question(qdef)?));
        }
        let decision = laya.decide(state, &qs)?;
        let answers: Map<String, Value> = decision
            .answers
            .iter()
            .zip(qs.iter())
            .map(|((qid, a), (_, q))| (qid.clone(), answer_json(a, q)))
            .collect();
        Ok(json!({
            "model": "rl-agent",
            "answers": answers,
            "usage": { "input_tokens": decision.input_tokens, "output_tokens": 0 },
        }))
    })();
    match result {
        Ok(v) => HttpResponse::json(200, &v.to_string()),
        Err(e) => HttpResponse::json(
            400,
            &json!({ "error": "bad_decide_request", "detail": format!("{e:#}") }).to_string(),
        ),
    }
}

fn main() -> Result<()> {
    let args = parse_args();
    let device = pagoda_hf::CandleModel::device_from_env()?;
    eprintln!("==> loading Laya from {} on {device:?}", args.repo);
    let laya = Arc::new(Laya::from_hub(&args.repo, device)?);
    let listener = TcpListener::bind(format!("{}:{}", args.addr, args.port))?;
    eprintln!(
        "laya_server listening on http://{}:{} (POST /decide)",
        args.addr, args.port
    );
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                let laya = Arc::clone(&laya);
                std::thread::spawn(move || {
                    let _ = stream.set_nodelay(true);
                    let resp = match read_http_request(&mut stream) {
                        Ok((method, path, body)) => match (method.as_str(), path.as_str()) {
                            ("GET", "/health") => {
                                HttpResponse::json(200, r#"{"status":"ok","model":"rl-agent"}"#)
                            }
                            ("POST", "/decide") => decide(&laya, &body),
                            _ => HttpResponse::json(404, r#"{"error":"not found"}"#),
                        },
                        Err(_) => HttpResponse::json(400, r#"{"error":"bad request"}"#),
                    };
                    let _ = write_http_response(&mut stream, &resp);
                });
            }
            Err(e) => eprintln!("accept error: {e}"),
        }
    }
    Ok(())
}
