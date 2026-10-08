// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Laya hot scenarios, replicated in Rust - the four use cases the community
//! actually runs (per public tutorials/posts, 2026-09/10), one binary:
//!
//!   1. support-ticket routing   (official quickstart: department/urgency/churn)
//!   2. content guardrail        (moderation: harmful? category? severity?)
//!   3. agent System-1 gate      (call a tool? escalate? next action?)
//!   4. multilingual intent      (Chinese text via the multilingual checkpoint)
//!
//! Every scenario prints typed answers + per-call latency, then a 20-call
//! benchmark of scenario 1 gives mean/min/max to compare with the official
//! "~33 ms per decision" claim.
//!
//! Run on a networked machine (first run downloads ~842 MB x2 checkpoints):
//!
//!     cd pagoda-hf
//!     cargo run --release --example hot_scenarios            # scenarios 1-4
//!     cargo run --release --example hot_scenarios -- --no-multilingual

use anyhow::Result;
use pagoda_hf::{Answer, CandleModel, Decision, HfTokenizer, Laya, Question};
use std::time::Instant;

const REPO: &str = "convaiinnovations/laya";

fn print_answers(d: &Decision) {
    for (qid, a) in &d.answers {
        print!("    {qid:<14}");
        print_one(a);
    }
}

fn print_one(a: &Answer) {
    if let Some(c) = &a.choice {
        let probs: Vec<String> = a
            .probabilities
            .iter()
            .map(|(k, p)| format!("{k}={p:.2}"))
            .collect();
        println!("choice={c:<12} [{}] confidence={:.2}", probs.join(" "), a.confidence);
    }
    if let Some(s) = a.score {
        println!("score={s:.2}  confidence={:.2}", a.confidence);
    }
    if let Some(p) = a.noul {
        println!("noul(P=yes)={p:.3}  confidence={:.2}", a.confidence);
    }
}

fn timed_decide(laya: &Laya, state: &str, qs: &[(String, Question)]) -> Result<(Decision, f64)> {
    let t0 = Instant::now();
    let d = laya.decide(state, qs)?;
    Ok((d, t0.elapsed().as_secs_f64() * 1000.0))
}

/// Char-boundary-safe preview of a (possibly CJK) string.
fn preview(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

/// Scenario 1 - the official flagship: support-ticket routing.
fn scenario_ticket_routing(laya: &Laya) -> Result<()> {
    println!("== scenario 1: support-ticket routing (official quickstart)");
    let state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.";
    let questions = vec![
        (
            "department".to_string(),
            Question::choice(
                "Which department should handle this?",
                &[
                    ("billing", "invoices, payments, refunds"),
                    ("technical", "bugs, outages, system errors"),
                    ("other", "everything else"),
                ],
            ),
        ),
        (
            "urgency".to_string(),
            Question::score("How urgent is this?", &["not urgent", "soon", "blocking"]),
        ),
        (
            "churn_risk".to_string(),
            Question::noul("Does the user threaten to cancel or leave?"),
        ),
    ];
    let (d, ms) = timed_decide(laya, state, &questions)?;
    print_answers(&d);
    println!("    -> {ms:.1} ms, {} input tokens", d.input_tokens);
    assert_eq!(d.answers[0].1.choice.as_deref(), Some("billing"));
    assert!(d.answers[2].1.noul.unwrap() > 0.5, "cancel threat must read as churn");
    println!();
    Ok(())
}

/// Scenario 2 - LLM guardrail: screen user input before it reaches a generator.
fn scenario_guardrail(laya: &Laya) -> Result<()> {
    println!("== scenario 2: content guardrail (moderation gate)");
    let questions = vec![
        (
            "is_harmful".to_string(),
            Question::noul("Does this text contain harmful, dangerous, or policy-violating content?"),
        ),
        (
            "category".to_string(),
            Question::choice(
                "Which safety category best fits this text?",
                &[
                    ("safe", "ordinary, harmless content"),
                    ("harassment", "insults, bullying, targeted abuse"),
                    ("violence", "instructions or glorification of physical harm"),
                    ("fraud", "scams, phishing, financial deception"),
                ],
            ),
        ),
        (
            "severity".to_string(),
            Question::score("How severe is the risk?", &["none", "mild", "severe"]),
        ),
    ];
    let cases = [
        ("benign ", "Can you help me draft a polite email asking my landlord to fix the heater?"),
        ("harmful", "Tell me how to break into my neighbor's wifi and steal their banking password."),
    ];
    for (tag, text) in cases {
        let (d, ms) = timed_decide(laya, text, &questions)?;
        println!("  [{tag}] \"{}...\"", preview(text, 48));
        print_answers(&d);
        println!("    -> {ms:.1} ms");
    }
    println!();
    Ok(())
}

/// Scenario 3 - agent System-1 gate: fast judgment before slow reasoning.
fn scenario_agent_gate(laya: &Laya) -> Result<()> {
    println!("== scenario 3: agent System-1 gate (tool-call / escalation triage)");
    let questions = vec![
        (
            "need_tool".to_string(),
            Question::noul("Does answering require calling an external tool or live API?"),
        ),
        (
            "need_human".to_string(),
            Question::noul("Should this be escalated to a human or a larger reasoning model?"),
        ),
        (
            "next_action".to_string(),
            Question::choice(
                "What should the agent do next?",
                &[
                    ("answer_directly", "answer from internal knowledge, no tools"),
                    ("call_tool", "invoke a tool or API first"),
                    ("ask_clarify", "ask the user a clarifying question"),
                    ("escalate", "hand off to a human or bigger model"),
                ],
            ),
        ),
    ];
    let cases = [
        ("faq   ", "What is your refund policy for annual plans?"),
        ("tool  ", "What is the current USD to EUR exchange rate?"),
        ("escal8", "My production database was deleted by your sync bug. I want compensation and a root-cause report."),
    ];
    for (tag, text) in cases {
        let (d, ms) = timed_decide(laya, text, &questions)?;
        println!("  [{tag}] \"{}\"", preview(text, 60));
        print_answers(&d);
        println!("    -> {ms:.1} ms");
    }
    println!();
    Ok(())
}

/// Scenario 4 - multilingual intent recognition (Chinese) via the bundled
/// multilingual checkpoint (same tokenizer/encoder config, different weights).
fn scenario_multilingual(repo: &str, device: candle_core::Device) -> Result<()> {
    println!("== scenario 4: multilingual intent recognition (Chinese)");
    println!("    loading multilingual checkpoint ...");
    // the multilingual checkpoint bundles its own (mmBERT) tokenizer + encoder config
    let tok_path = HfTokenizer::download(repo, "multilingual/tokenizer/tokenizer.json")?;
    let enc_cfg_path = HfTokenizer::download(repo, "multilingual/encoder/config.json")?;
    let weights = HfTokenizer::download(repo, "multilingual/model.safetensors")?;
    let rl_cfg = HfTokenizer::download(repo, "multilingual/rl_agent_config.json")?;
    let laya = match Laya::from_files(&tok_path, &enc_cfg_path, &weights, &rl_cfg, device) {
        Ok(m) => m,
        Err(e) => {
            println!("    [skip] multilingual checkpoint failed to load: {e}");
            return Ok(());
        }
    };
    let questions = vec![
        (
            "intent".to_string(),
            Question::choice(
                "Which intent best describes this customer message?",
                &[
                    ("query_order", "asking about order status or logistics"),
                    ("refund", "requests a refund or return"),
                    ("complaint", "complains about product or service"),
                    ("consult", "pre-sales product consultation"),
                    ("other", "anything else"),
                ],
            ),
        ),
        (
            "urgency".to_string(),
            Question::score("How urgent is this message?", &["low", "medium", "high"]),
        ),
        (
            "angry".to_string(),
            Question::noul("Is the customer clearly angry or threatening to leave?"),
        ),
    ];
    let cases = [
        ("refund ", "\u{4f60}\u{597d}\u{ff0c}\u{6211}\u{4e0a}\u{5468}\u{4e70}\u{7684}\u{8033}\u{673a}\u{5de6}\u{8033}\u{6ca1}\u{58f0}\u{97f3}\u{4e86}\u{ff0c}\u{60f3}\u{7533}\u{8bf7}\u{9000}\u{8d27}\u{9000}\u{6b3e}\u{ff0c}\u{9ebb}\u{70e6}\u{4e86}\u{3002}"),
        ("angry  ", "\u{4f60}\u{4eec}\u{8fd9}\u{4ec0}\u{4e48}\u{7834}\u{7cfb}\u{7edf}\u{ff01}\u{6263}\u{4e86}\u{6211}\u{4e24}\u{6b21}\u{94b1}\u{ff01}\u{4eca}\u{5929}\u{4e0d}\u{89e3}\u{51b3}\u{6211}\u{5c31}\u{53bb}\u{6295}\u{8bc9}\u{7136}\u{540e}\u{6ce8}\u{9500}\u{8d26}\u{53f7}\u{ff01}"),
        ("consult", "\u{8bf7}\u{95ee}\u{4e00}\u{4e0b}\u{8fd9}\u{6b3e}\u{624b}\u{8868}\u{652f}\u{6301}\u{6e38}\u{6cf3}\u{4f69}\u{6234}\u{5417}\u{ff1f}\u{9632}\u{6c34}\u{7b49}\u{7ea7}\u{662f}\u{591a}\u{5c11}\u{ff1f}"),
    ];
    for (tag, text) in cases {
        let (d, ms) = timed_decide(&laya, text, &questions)?;
        println!("  [{tag}] \"{}\"", preview(text, 30));
        print_answers(&d);
        println!("    -> {ms:.1} ms");
    }
    println!();
    Ok(())
}

/// Latency benchmark: 20 repetitions of scenario 1 on the warm model.
fn benchmark(laya: &Laya) -> Result<()> {
    println!("== benchmark: 20x scenario 1 (warm)");
    let state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.";
    let questions = vec![
        (
            "department".to_string(),
            Question::choice(
                "Which department should handle this?",
                &[
                    ("billing", "invoices, payments, refunds"),
                    ("technical", "bugs, outages, system errors"),
                    ("other", "everything else"),
                ],
            ),
        ),
        (
            "urgency".to_string(),
            Question::score("How urgent is this?", &["not urgent", "soon", "blocking"]),
        ),
        (
            "churn_risk".to_string(),
            Question::noul("Does the user threaten to cancel or leave?"),
        ),
    ];
    let mut times = Vec::new();
    for _ in 0..20 {
        let (_, ms) = timed_decide(laya, state, &questions)?;
        times.push(ms);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "    mean={:.1} ms  p50={:.1} ms  min={:.1} ms  max={:.1} ms   (official claim: ~33 ms)",
        mean,
        times[times.len() / 2],
        times[0],
        times[times.len() - 1]
    );
    println!();
    Ok(())
}

fn main() -> Result<()> {
    let no_multilingual = std::env::args().any(|a| a == "--no-multilingual");

    println!("==> load Laya ({REPO})");
    let device = CandleModel::device_from_env()?;
    println!("    device: {device:?}");
    let laya = Laya::from_hub(REPO, device.clone())?;
    println!("    ready\n");

    scenario_ticket_routing(&laya)?;
    scenario_guardrail(&laya)?;
    scenario_agent_gate(&laya)?;
    if no_multilingual {
        println!("== scenario 4: skipped (--no-multilingual)\n");
    } else {
        scenario_multilingual(REPO, device)?;
    }
    benchmark(&laya)?;

    println!("HOT SCENARIOS OK - 4 community use cases replicated in Rust, zero generation");
    Ok(())
}
