// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! In-process latency benchmark for the Laya decision model.
//! Mirrors `bench-ref/bench_laya.py` (the reference PyTorch harness) so the
//! two can be compared run-for-run on the same machine.
//!
//! ```text
//! cargo run --release --example bench_laya -- [--iters 50] [--warmup 5]
//! ```
//!
//! Prints one JSON line: load time, latency percentiles, peak RSS.

use anyhow::Result;
use pagoda_hf::{CandleModel, Laya, Question};
use std::time::Instant;

fn billing_questions() -> Vec<(String, Question)> {
    vec![
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
    ]
}

const STATE: &str =
    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.";

fn peak_rss_kb() -> u64 {
    // Linux: VmHWM is the high-water mark of resident memory.
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() -> Result<()> {
    let mut iters = 50usize;
    let mut warmup = 5usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                iters = args[i + 1].parse()?;
                i += 2;
            }
            "--warmup" => {
                warmup = args[i + 1].parse()?;
                i += 2;
            }
            other => anyhow::bail!("unknown arg {other}"),
        }
    }

    let device = CandleModel::device_from_env()?;
    eprintln!("device: {device:?}, iters={iters} warmup={warmup}");

    let t_load = Instant::now();
    let laya = Laya::from_hub("convaiinnovations/laya", device)?;
    let load_s = t_load.elapsed().as_secs_f64();

    let questions = billing_questions();
    for _ in 0..warmup {
        let _ = laya.decide(STATE, &questions)?;
    }

    let mut lat_ms: Vec<f64> = Vec::with_capacity(iters);
    let mut dept = String::new();
    let mut dept_probs: Vec<(String, u32)> = Vec::new();
    for _ in 0..iters {
        let t = Instant::now();
        let d = laya.decide(STATE, &questions)?;
        lat_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        dept = d.answers[0].1.choice.clone().unwrap_or_default();
        dept_probs = d.answers[0]
            .1
            .probabilities
            .iter()
            .map(|(k, p)| (k.clone(), p.to_bits()))
            .collect();
    }
    assert_eq!(dept, "billing", "sanity: department must route to billing");

    lat_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = lat_ms.iter().sum::<f64>() / lat_ms.len() as f64;
    let probs_json: Vec<String> = dept_probs
        .iter()
        .map(|(k, b)| format!("\"{k}\":{b}"))
        .collect();
    println!(
        "{{\"impl\":\"pagoda-rust\",\"dtype\":\"f32\",\"load_s\":{load_s:.3},\"mean_ms\":{mean:.2},\"p50_ms\":{:.2},\"p95_ms\":{:.2},\"min_ms\":{:.2},\"max_ms\":{:.2},\"peak_rss_mb\":{:.0},\"iters\":{iters},\"dept_prob_bits\":{{{}}}}}",
        percentile(&lat_ms, 0.50),
        percentile(&lat_ms, 0.95),
        lat_ms[0],
        lat_ms[lat_ms.len() - 1],
        peak_rss_kb() as f64 / 1024.0,
        probs_json.join(","),
    );
    Ok(())
}
