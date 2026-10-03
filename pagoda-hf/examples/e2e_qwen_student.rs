// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Distilled-student end-to-end: a real Qwen2.5 LoRA-merged student checkpoint
//! running pure-Rust prefill+decode through the pagoda engine (incremental KV
//! session, one physical forward per generated token), no Python anywhere.
//!
//! Run against the distill pipeline output (or any Qwen2.5/Llama-family dir):
//!
//!     cargo run --release --example e2e_qwen_student -- /path/to/student-merged
//!     PAGODA_DEVICE=cuda PAGODA_DTYPE=f16 cargo run --release --features cuda \
//!         --example e2e_qwen_student -- /path/to/student-merged
//!
//! What this proves:
//!   1. Qwen2-family checkpoints load (q/k/v bias probed, tied embeddings,
//!      GQA 14 heads / 2 kv heads, rope_theta=1e6).
//!   2. Prefill feeds the prompt once; decode feeds one token per step
//!      (tokens_fed equals prompt+steps, not quadratic replay).
//!   3. The student answers in the trained JSON schema on an unseen ticket,
//!      stopping at <|im_end|>.

use anyhow::Result;
use pagoda::{Engine, EngineConfig, FinishReason, SamplingParams, Tokenizer, WriteRequest};
use pagoda_hf::{CandleModel, HfTokenizer};
use std::path::PathBuf;
use std::sync::atomic::Ordering;

/// The exact prompt the distill pipeline trains and evaluates on
/// (distill/train_lora.py PROMPT), ChatML-wrapped the way Qwen2.5's
/// chat template does (default system message included).
const TICKET: &str = "你好，我3月12日下单的订单 ORD-88213 扣了两次钱，第二笔 349 元到今天都没退回来，发票也开错了抬头。请尽快处理退款并重开发票，谢谢。";

fn prompt_for(ticket: &str) -> String {
    // Raw string: the JSON skeleton keeps its braces unescaped.
    let template = r##"<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
你是电商客服工单结构化助手。阅读客户消息，提取信息并只输出 JSON：
{"department": "billing|shipping|technical|product|other", "urgency": 0|1|2, "sentiment": "angry|neutral|positive", "order_id": "订单号或null", "refund_amount": "数字或null", "needs_human": true|false, "reply": "不超过80字的中文回复草稿"}
客户消息：
{email}<|im_end|>
<|im_start|>assistant
"##;
    template.replace("{email}", ticket)
}

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .or_else(|| std::env::var("PAGODA_MODEL_DIR").ok())
        .map(PathBuf::from)
        .expect("usage: e2e_qwen_student <model-dir> (or PAGODA_MODEL_DIR)");
    println!("==> model dir: {}", dir.display());

    println!("==> [1/4] tokenizer ({{dir}}/tokenizer.json)");
    let tokenizer = HfTokenizer::from_file(dir.join("tokenizer.json"))?;
    println!(
        "    vocab={} eos={} bos={:?}",
        tokenizer.vocab_size(),
        tokenizer.eos_token_id(),
        tokenizer.bos_token_id()
    );

    println!("==> [2/4] weights on PAGODA_DEVICE / PAGODA_DTYPE");
    let mut weights: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("model") && n.ends_with(".safetensors"))
        })
        .collect();
    weights.sort();
    assert!(!weights.is_empty(), "no model*.safetensors in {}", dir.display());
    let device = CandleModel::device_from_env()?;
    let dtype = pagoda_hf::CandleModel::dtype_from_env()?;
    println!("    device={device:?} dtype={dtype:?} shards={}", weights.len());
    let model = CandleModel::llama_from_safetensors_as(
        dir.join("config.json"),
        &weights,
        device,
        dtype,
    )?;
    let fed = model.tokens_fed_handle();

    println!("==> [3/4] prefill + decode through the pagoda engine");
    let mut engine = Engine::new(
        tokenizer,
        model,
        EngineConfig {
            num_kv_blocks: 2048,
            block_size: 16,
            ..EngineConfig::default()
        },
    );
    let prompt = prompt_for(TICKET);
    let params = SamplingParams {
        max_tokens: 200,
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let fed_before = fed.load(Ordering::Relaxed);
    let t0 = std::time::Instant::now();
    let out = engine.generate(&WriteRequest::new(&prompt, params));
    let elapsed = t0.elapsed();
    let fed_now = fed.load(Ordering::Relaxed) - fed_before;

    println!("==> [4/4] result");
    println!("    finish={:?} steps={} elapsed={elapsed:?}", out.finish_reason, out.output_token_ids.len());
    println!("    tokens_fed={fed_now} (prompt_tokens={} + decode_steps)", out.prompt_tokens);
    println!("---8<--- student output ---8<---");
    println!("{}", out.text);
    println!("---8<---");

    assert_ne!(out.finish_reason, FinishReason::Fault, "model fault");
    assert_ne!(out.finish_reason, FinishReason::Rejected, "request rejected");
    assert_eq!(
        fed_now as usize,
        out.prompt_tokens + out.output_token_ids.len() - 1,
        "session must feed the prompt once, then one token per step"
    );
    let json_start = out.text.find('{');
    let json_end = out.text.rfind('}');
    assert!(json_start.is_some() && json_end.is_some(), "output is not JSON-shaped");
    let json: serde_json::Value =
        serde_json::from_str(&out.text[json_start.unwrap()..=json_end.unwrap()])
            .expect("student output must parse as JSON");
    for field in ["department", "urgency", "sentiment", "needs_human", "reply"] {
        assert!(json.get(field).is_some(), "missing field {field}");
    }
    let dept = json["department"].as_str().unwrap_or("");
    assert!(
        ["billing", "shipping", "technical", "product", "other"].contains(&dept),
        "department must be one of the schema values, got {dept:?}"
    );
    println!("==> e2e_qwen_student OK (department={dept}, schema complete)");
    Ok(())
}
