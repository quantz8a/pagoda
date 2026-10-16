// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Agent-cluster throughput: shared trunk (KV checkpoint) vs naive prefill.
//!
//! N agents share one system prompt. Pagoda pins it once with
//! `create_checkpoint`, then each branch only pays for its task suffix +
//! decode. The naive baseline re-feeds trunk+task for every agent.
//!
//! Default is offline (`ByteTokenizer` + `NGramModel`). Pass `--repo` /
//! `--gguf` for real weights when Hub or local files are available.
//!
//! ```text
//! cargo run --release --example bench_agent_cluster -- \
//!   --agents 8 --gen-tokens 32 --trunk-repeats 64
//! ```

use anyhow::{Context, Result};
use pagoda::{Engine, EngineConfig, ModelEngine, SamplingParams, Tokenizer, WriteRequest};
use std::time::Instant;

struct Args {
    agents: usize,
    gen_tokens: usize,
    trunk_repeats: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            agents: 8,
            gen_tokens: 32,
            trunk_repeats: 64,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else { break };
        match flag.as_str() {
            "--agents" => args.agents = value.parse().expect("--agents"),
            "--gen-tokens" => args.gen_tokens = value.parse().expect("--gen-tokens"),
            "--trunk-repeats" => args.trunk_repeats = value.parse().expect("--trunk-repeats"),
            other => eprintln!("ignoring {other} (this binary is offline toy; use docs for gguf)"),
        }
    }
    args
}

fn main() -> Result<()> {
    let args = parse_args();
    eprintln!("==> backend toy (ByteTokenizer + NGramModel, offline)");

    let sentence = "You are a careful research agent. Follow tools. Cite sources. ";
    let trunk = sentence.repeat(args.trunk_repeats);
    let trunk_tokens = pagoda::ByteTokenizer::new().encode(&trunk).len();
    eprintln!(
        "==> agents={} trunk_tokens={} gen_tokens={}",
        args.agents, trunk_tokens, args.gen_tokens
    );
    anyhow::ensure!(trunk_tokens > 32, "trunk too short: {trunk_tokens}");

    let tasks: Vec<String> = (0..args.agents)
        .map(|i| format!(" Task {i}: summarize finding #{i} in one sentence."))
        .collect();
    let greedy = SamplingParams {
        max_tokens: args.gen_tokens,
        temperature: 0.0,
        ..SamplingParams::default()
    };
    let cfg = EngineConfig {
        num_kv_blocks: 65536,
        block_size: 16,
        max_running_requests: 64,
        ..EngineConfig::default()
    };

    // Naive = no cross-agent sharing: each agent gets a fresh engine so radix
    // / vault cannot accidentally reuse the trunk (the common "N workers,
    // N full prefills" deployment).
    let naive = measure_naive_isolated(&cfg, &trunk, &tasks, &greedy)?;
    let (create_ms, branches_ms, ckpt) = {
        let mut engine = Engine::toy(cfg);
        measure_checkpoint(&mut engine, &trunk, &tasks, &greedy)?
    };

    let speedup_wall = naive.wall_ms / ckpt.wall_ms.max(1e-9);
    let speedup_fwd = naive.forwards as f64 / ckpt.forwards.max(1) as f64;

    let report = serde_json::json!({
        "scenario": "agent_cluster_shared_trunk",
        "device": "Cpu",
        "model": "toy/NGramModel",
        "agents": args.agents,
        "trunk_tokens": trunk_tokens,
        "gen_tokens_per_agent": args.gen_tokens,
        "naive": {
            "wall_ms": round2(naive.wall_ms),
            "model_forwards": naive.forwards,
            "output_tokens": naive.out,
            "output_tok_per_s": round1(naive.out as f64 / (naive.wall_ms / 1e3).max(1e-9)),
            "prefix_hit_tokens_sum": naive.prefix_hits,
        },
        "checkpoint": {
            "create_ms": round2(create_ms),
            "branches_ms": round2(branches_ms),
            "total_wall_ms": round2(ckpt.wall_ms),
            "model_forwards": ckpt.forwards,
            "output_tokens": ckpt.out,
            "output_tok_per_s": round1(ckpt.out as f64 / (ckpt.wall_ms / 1e3).max(1e-9)),
            "prefix_hit_tokens_sum": ckpt.prefix_hits,
            "prefix_hit_per_agent": ckpt.prefix_hits / args.agents.max(1),
        },
        "speedup": {
            "wall_clock": round2(speedup_wall),
            "model_forwards": round2(speedup_fwd),
            "forwards_saved": naive.forwards.saturating_sub(ckpt.forwards),
            "forwards_saved_ratio": round2(
                1.0 - ckpt.forwards as f64 / naive.forwards.max(1) as f64
            ),
        },
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!(
        "==> wall {speedup_wall:.2}×  forwards {speedup_fwd:.2}×  (naive {:.1}ms / ckpt {:.1}ms)",
        naive.wall_ms, ckpt.wall_ms
    );
    Ok(())
}

struct Arm {
    wall_ms: f64,
    forwards: u64,
    out: usize,
    prefix_hits: usize,
}

fn measure_naive_isolated(
    cfg: &EngineConfig,
    trunk: &str,
    tasks: &[String],
    greedy: &SamplingParams,
) -> Result<Arm> {
    let t = Instant::now();
    let mut out = 0usize;
    let mut prefix_hits = 0usize;
    let mut forwards = 0u64;
    for task in tasks {
        let mut engine = Engine::toy(cfg.clone());
        let s0 = engine.stats();
        let prompt = format!("{trunk}{task}");
        let o = engine.generate(&WriteRequest::new(&prompt, greedy.clone()));
        anyhow::ensure!(o.finish_reason != pagoda::FinishReason::Fault, "naive fault");
        let s1 = engine.stats();
        out += o.output_token_ids.len();
        prefix_hits += o.prefix_hit_tokens;
        forwards += s1.total_forward - s0.total_forward;
    }
    Ok(Arm {
        wall_ms: t.elapsed().as_secs_f64() * 1e3,
        forwards,
        out,
        prefix_hits,
    })
}

fn measure_checkpoint<M: ModelEngine, T: Tokenizer>(
    engine: &mut Engine<M, T>,
    trunk: &str,
    tasks: &[String],
    greedy: &SamplingParams,
) -> Result<(f64, f64, Arm)> {
    let s0 = engine.stats();
    let t = Instant::now();
    let ckpt = engine.create_checkpoint(trunk);
    let create_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut out = 0usize;
    let mut prefix_hits = 0usize;
    for task in tasks {
        let o = engine
            .generate_from_checkpoint(ckpt, task, greedy.clone())
            .context("checkpoint missing")?;
        anyhow::ensure!(o.finish_reason != pagoda::FinishReason::Fault, "branch fault");
        out += o.output_token_ids.len();
        prefix_hits += o.prefix_hit_tokens;
    }
    let branches_ms = t.elapsed().as_secs_f64() * 1e3;
    let s1 = engine.stats();
    Ok((
        create_ms,
        branches_ms,
        Arm {
            wall_ms: create_ms + branches_ms,
            forwards: s1.total_forward - s0.total_forward,
            out,
            prefix_hits,
        },
    ))
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}
fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
