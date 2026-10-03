// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! End-to-end verification for the Laya decision model (System-1):
//! real HF weights (ModernBERT encoder + decision head) answering typed
//! questions about a state in one non-autoregressive forward pass.
//!
//! Run on a networked machine (downloads ~842 MB on first use):
//!
//! ```text
//! cd pagoda-hf
//! cargo run --release --example e2e_laya
//! ```
//!
//! Uses `convaiinnovations/laya` (Apache-2.0). Asserts the README's billing
//! scenario: department routes to "billing", churn risk fires, probabilities
//! are calibrated (sum to 1), and results are deterministic across runs.

use anyhow::Result;
use pagoda_hf::{CandleModel, Laya, Question};

fn main() -> Result<()> {
    const REPO: &str = "convaiinnovations/laya";

    println!("==> [1/3] download + load Laya ({REPO})");
    let device = CandleModel::device_from_env()?;
    println!("    device: {device:?}");
    let laya = Laya::from_hub(REPO, device)?;
    println!("    model loaded (ModernBERT encoder + decision head)");

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

    println!("==> [2/3] decide: README billing scenario");
    let t0 = std::time::Instant::now();
    let d1 = laya.decide(state, &questions)?;
    println!("    {:?} input_tokens={}", t0.elapsed(), d1.input_tokens);
    for (qid, a) in &d1.answers {
        if let Some(c) = &a.choice {
            println!("    {qid}: choice={c} confidence={:.3}", a.confidence);
        }
        if let Some(s) = a.score {
            println!("    {qid}: score={s:.3} confidence={:.3}", a.confidence);
        }
        if let Some(p) = a.noul {
            println!("    {qid}: noul={p:.3} confidence={:.3}", a.confidence);
        }
        let sum: f32 = a.probabilities.iter().map(|(_, p)| p).sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "{qid}: probabilities must sum to 1, got {sum}"
        );
        assert!((0.0..=1.0).contains(&a.confidence), "{qid}: confidence range");
        assert!((0.0..=1.0).contains(&a.act_probability), "{qid}: act range");
    }

    let dept = &d1.answers[0].1;
    assert_eq!(
        dept.choice.as_deref(),
        Some("billing"),
        "double-billing ticket must route to billing"
    );
    let urgency = &d1.answers[1].1;
    let u = urgency.score.expect("score answer");
    assert!((0.0..=2.0).contains(&u), "score must stay within the 3-level scale");
    let churn = &d1.answers[2].1;
    let risk = churn.noul.expect("noul answer");
    assert!(
        risk > 0.5,
        "an explicit cancel threat must read as churn risk, got {risk}"
    );

    println!("==> [3/3] determinism: same input twice, bit-identical answers");
    let d2 = laya.decide(state, &questions)?;
    for (a, b) in d1.answers.iter().zip(&d2.answers) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1.choice, b.1.choice);
        for ((ka, pa), (kb, pb)) in a.1.probabilities.iter().zip(&b.1.probabilities) {
            assert_eq!(ka, kb);
            assert_eq!(pa.to_bits(), pb.to_bits(), "probabilities must be bit-identical");
        }
    }

    println!();
    println!("LAYA VERIFICATION OK — typed decisions with calibrated probabilities, zero generation");
    Ok(())
}