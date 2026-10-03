// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! `pagoda` command line interface.
//!
//! Subcommands:
//! * `sample`  — offline generation (repeat a prompt to show prefix caching)
//! * `program` — run a canned SGLang-style DSL program
//! * `serve`   — start the minimal HTTP serving frontend

use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use pagoda::{CacheBackend, EngineConfig, Program, SamplingParams, SchedulePolicy, ToyEngine, WriteRequest};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = match args.split_first() {
        Some((c, r)) => (c.as_str(), r.to_vec()),
        None => {
            print_usage();
            return ExitCode::SUCCESS;
        }
    };

    match cmd {
        "sample" => sample(&rest),
        "program" => program(),
        "serve" => serve(&rest),
        "help" | "-h" | "--help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown command: {other}\n");
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn print_usage() {
    println!("pagoda — an SGLang-inspired LLM serving runtime");
    println!();
    println!("USAGE:");
    println!("  pagoda sample -p \"prompt\" [--max-tokens N] [--temperature F] [--repeat N]");
    println!("  pagoda program");
    println!("  pagoda serve   [--port 8080] [--addr 127.0.0.1] [--upstream http://host:port]");
    println!("                 [--route dept=http://host:port]   (repeatable; Laya picks the upstream)");
    println!("                 [--laya-url http://host:port] [--laya-shadow] [--laya-required]");
    println!("                 [--churn-threshold 0.5] [--min-confidence 0.0]");
}

fn cfg() -> EngineConfig {
    EngineConfig {
        num_kv_blocks: 65536,
        block_size: 16,
        max_running_requests: 64,
        max_prefill_tokens_per_step: 32,
        seed: 0,
        schedule_policy: SchedulePolicy::default(),
        evict_on_pressure: true,
        max_waiting_requests: 256,
        max_total_tokens: 8192,
        cache_backend: CacheBackend::default(),
    }
}

fn sample(args: &[String]) -> ExitCode {
    let mut prompt = String::from("The quick brown fox");
    let mut max_tokens = 48usize;
    let mut temperature = 0.0f32;
    let mut repeat = 1usize;
    let mut seed = 0u64;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-p" | "--prompt" => prompt = it.next().cloned().unwrap_or_default(),
            "--max-tokens" => {
                max_tokens = it.next().and_then(|v| v.parse().ok()).unwrap_or(max_tokens)
            }
            "--temperature" => {
                temperature = it.next().and_then(|v| v.parse().ok()).unwrap_or(temperature)
            }
            "--repeat" => repeat = it.next().and_then(|v| v.parse().ok()).unwrap_or(repeat),
            "--seed" => seed = it.next().and_then(|v| v.parse().ok()).unwrap_or(seed),
            other => {
                eprintln!("unknown flag: {other}");
                return ExitCode::FAILURE;
            }
        }
    }

    let mut engine = ToyEngine::toy(cfg());
    let sampling = SamplingParams {
        max_tokens,
        temperature,
        seed,
        ..SamplingParams::default()
    };
    let req = WriteRequest::new(prompt.clone(), sampling.clone());

    for i in 0..repeat {
        let out = engine.generate(&req);
        println!("[{i}] {} => {}", out.finish_reason, out.text.trim());
        println!(
            "     prefix_hit={}/{} forward={}",
            out.prefix_hit_tokens, out.prompt_tokens, out.forward_count
        );
    }

    let s = engine.stats();
    println!();
    println!(
        "stats: requests={} forward={} radix_nodes={} hit_tokens={} hit_rate={:.2} kv_blocks_in_use={} kv_allocated={} kv_freed={} prefill_chunks={}",
        s.total_requests,
        s.total_forward,
        s.radix_nodes,
        s.radix_hit_tokens,
        engine.radix_hit_rate(),
        s.kv_blocks - s.kv_free_blocks,
        s.kv_allocations,
        s.kv_frees,
        s.prefill_chunks
    );
    println!(
        "revenue: compute_saved={} graft_saved={} prefill_skip={:.2} avg_forward/token={:.2} kv_util={:.2} decode_batch={:.2}x output_tokens={}",
        s.compute_saved_tokens(),
        s.model_graft_tokens,
        s.prefill_skip_ratio(),
        s.avg_forward_per_output_token(),
        s.kv_utilization(),
        s.decode_batch_factor(),
        s.total_output_tokens
    );
    ExitCode::SUCCESS
}

fn program() -> ExitCode {
    let mut engine = ToyEngine::toy(cfg());

    let mut p = Program::new();
    p.system("You are a helpful assistant whose answer is a single short phrase.");
    p.user("Pick the best continuation for: the capital of France is");
    p.select(
        "city",
        vec![
            " Paris".to_string(),
            " London".to_string(),
            " Berlin".to_string(),
        ],
        SamplingParams::default(),
    );
    p.gen(
        "explain",
        SamplingParams {
            max_tokens: 48,
            temperature: 0.8,
            seed: 5,
            ..SamplingParams::default()
        },
    );

    let results = engine.run_program(&p);
    for (i, r) in results.iter().enumerate() {
        println!("--- stream {i} ---");
        println!("city    = {:?}", r.get("city"));
        println!("explain = {:?}", r.get("explain").map(|s| s.trim()));
    }
    println!();
    println!("transcript:");
    for r in &results {
        println!("{}", r.transcript.trim_end());
    }

    ExitCode::SUCCESS
}

fn serve(args: &[String]) -> ExitCode {
    let mut port = 8000u16;
    let mut addr = "127.0.0.1".to_string();
    let mut upstream: Option<String> = None;
    let mut laya: Option<String> = None;
    let mut laya_shadow = false;
    let mut laya_required = false;
    let mut churn_threshold = 0.5f64;
    let mut min_confidence = 0.0f64;
    let mut routes: Vec<(String, String)> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => port = it.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "--addr" => addr = it.next().cloned().unwrap_or(addr),
            "--upstream" => upstream = it.next().cloned(),
            "--laya-url" => laya = it.next().cloned(),
            "--route" => {
                let spec = it.next().cloned().unwrap_or_default();
                match spec.split_once('=') {
                    Some((dept, url)) if !dept.is_empty() && url.starts_with("http://") => {
                        routes.push((dept.to_string(), url.to_string()))
                    }
                    _ => {
                        eprintln!("bad --route {spec:?} (want dept=http://host[:port])");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--laya-shadow" => laya_shadow = true,
            "--laya-required" => laya_required = true,
            "--churn-threshold" => {
                churn_threshold = it.next().and_then(|v| v.parse().ok()).unwrap_or(churn_threshold)
            }
            "--min-confidence" => {
                min_confidence = it.next().and_then(|v| v.parse().ok()).unwrap_or(min_confidence)
            }
            other => {
                eprintln!("unknown flag: {other}");
                return ExitCode::FAILURE;
            }
        }
    }
    let proxy = match upstream.as_deref() {
        Some(url) => {
            let route_refs: Vec<(&str, &str)> = routes
                .iter()
                .map(|(d, u)| (d.as_str(), u.as_str()))
                .collect();
            match pagoda::server::Proxy::with_routes(url, &route_refs) {
                Some(p) => {
                    eprintln!("proxy mode: /generate + /v1/chat/completions -> {url}");
                    for (dept, route_url) in &routes {
                        eprintln!("route: {dept} -> {route_url}");
                    }
                    Some(Arc::new(p))
                }
                None => {
                    eprintln!("bad --upstream {url:?} (want http://host[:port])");
                    return ExitCode::FAILURE;
                }
            }
        }
        None => None,
    };
    let triage = match laya.as_deref() {
        Some(url) => match pagoda::triage::Triage::from_url(url) {
            Some(mut t) => {
                t.shadow = laya_shadow;
                t.required = laya_required;
                t.churn_threshold = churn_threshold;
                t.min_confidence = min_confidence;
                eprintln!(
                    "triage: Laya at {url} (churn>{churn_threshold}, conf<{min_confidence}{}{})",
                    if laya_shadow { ", shadow" } else { "" },
                    if laya_required { ", fail-closed" } else { "" },
                );
                Some(Arc::new(t))
            }
            None => {
                eprintln!("bad --laya-url {url:?} (want http://host[:port])");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let engine = Arc::new(Mutex::new(ToyEngine::toy(cfg())));
    match pagoda::server::run_with_triage(engine, &format!("{addr}:{port}"), proxy, triage) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("server error: {e}");
            ExitCode::FAILURE
        }
    }
}
