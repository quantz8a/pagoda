// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! `pagoda` command line interface.
//!
//! Subcommands:
//! * `sample`  — offline generation (repeat a prompt to show prefix caching)
//! * `program` — run a canned SGLang-style DSL program
//! * `serve`   — start the minimal HTTP serving frontend
//! * `store`   — run the standalone KV object store for PD disaggregation

use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use pagoda::{
    pd, CacheBackend, EngineConfig, PdRole, Program, SamplingParams, SchedulePolicy, ToyEngine,
    WriteRequest,
};

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
        "store" => store(&rest),
        "route" => route(&rest),
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
    println!("                                 (comma-separated list = prefix-affine worker pool)");
    println!("                 [--route dept=http://host:port]   (repeatable; Laya picks the upstream)");
    println!("                 [--laya-url http://host:port] [--laya-shadow] [--laya-required]");
    println!("                 [--churn-threshold 0.5] [--min-confidence 0.0]");
    println!("                 [--role unified|prefill|decode] [--store http://host:port]");
    println!("                 [--concurrent]  (unified role: scheduler actor, requests overlap;");
    println!("                                 combines with --laya-url triage)");
    println!("                 [--prefill-url http://host:port]  (decode role: conductor)");
    println!("  pagoda store   [--port 9100] [--addr 127.0.0.1] [--max-bytes 67108864]");
    println!("                 [--max-age-secs N]  (entry TTL; expired bundles count as misses)");
    println!("  pagoda route   [--port 8000] --prefill-url http://host:port");
    println!("                 --decode-url http://host:port   (repeatable; prefix-affinity)");
    println!("                 [--laya-url http://host:port] [--laya-shadow] [--laya-required]");
    println!("                 [--churn-threshold 0.5] [--min-confidence 0.0]");
    println!();
    println!("  PD disaggregation (Mooncake-style):");
    println!("    pagoda store --port 9100");
    println!("    pagoda serve --port 8001 --role prefill --store http://127.0.0.1:9100");
    println!("    pagoda serve --port 8002 --role decode  --store http://127.0.0.1:9100 \\");
    println!("                --prefill-url http://127.0.0.1:8001");
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
        "revenue: compute_saved={} graft_saved={} prefill_skip={:.2} avg_forward/token={:.2} kv_util={:.2} decode_batch={:.2}x forwards_saved={} aborted={} output_tokens={}",
        s.compute_saved_tokens(),
        s.model_graft_tokens,
        s.prefill_skip_ratio(),
        s.avg_forward_per_output_token(),
        s.kv_utilization(),
        s.decode_batch_factor(),
        s.compute_saved_tokens()
            + s.model_graft_tokens
            + s.total_decode_steps.saturating_sub(s.total_decode_calls),
        s.aborted_requests,
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
    let mut role = PdRole::Unified;
    let mut store_url: Option<String> = None;
    let mut prefill_url: Option<String> = None;
    let mut concurrent = false;
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
            "--role" => {
                let v = it.next().cloned().unwrap_or_default();
                role = match v.as_str() {
                    "unified" => PdRole::Unified,
                    "prefill" => PdRole::Prefill,
                    "decode" => PdRole::Decode,
                    _ => {
                        eprintln!("bad --role {v:?} (want unified|prefill|decode)");
                        return ExitCode::FAILURE;
                    }
                };
            }
            "--store" => store_url = it.next().cloned(),
            "--concurrent" => concurrent = true,
            "--prefill-url" => prefill_url = it.next().cloned(),
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
    let mut engine = ToyEngine::toy(cfg());
    if role != PdRole::Unified || store_url.is_some() {
        let Some(url) = store_url.as_deref() else {
            eprintln!("--role prefill|decode requires --store http://host:port");
            return ExitCode::FAILURE;
        };
        let Some(store) = pd::HttpStore::from_url(url) else {
            eprintln!("bad --store {url:?} (want http://host[:port])");
            return ExitCode::FAILURE;
        };
        engine.enable_pd(role, Arc::new(store));
        eprintln!("pd role: {} (kv store: {url})", role.as_str());
    }
    let conductor = match prefill_url.as_deref() {
        Some(url) => match pagoda::server::Conductor::from_url(url) {
            Some(c) => Some(Arc::new(c)),
            None => {
                eprintln!("bad --prefill-url {url:?} (want http://host[:port])");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    if concurrent {
        if role != PdRole::Unified
            || store_url.is_some()
            || prefill_url.is_some()
            || upstream.is_some()
        {
            eprintln!("--concurrent serves the unified role only (no --role/--store/--prefill-url/--upstream; --laya-url is allowed)");
            return ExitCode::FAILURE;
        }
        return match pagoda::server::run_concurrent(engine, &format!("{addr}:{port}"), triage) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("server error: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let engine = Arc::new(Mutex::new(engine));
    match pagoda::server::run_full(
        engine,
        &format!("{addr}:{port}"),
        proxy,
        triage,
        conductor,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("server error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn store(args: &[String]) -> ExitCode {
    let mut port = 9100u16;
    let mut addr = "127.0.0.1".to_string();
    let mut max_bytes = 64usize << 20;
    let mut max_age: Option<std::time::Duration> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => port = it.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "--addr" => addr = it.next().cloned().unwrap_or(addr),
            "--max-bytes" => {
                max_bytes = it.next().and_then(|v| v.parse().ok()).unwrap_or(max_bytes)
            }
            "--max-age-secs" => {
                let secs: f64 = it.next().and_then(|v| v.parse().ok()).unwrap_or(0.0);
                max_age = (secs > 0.0).then(|| std::time::Duration::from_secs_f64(secs));
            }
            other => {
                eprintln!("unknown flag: {other}");
                return ExitCode::FAILURE;
            }
        }
    }
    match pd::run_store(&format!("{addr}:{port}"), max_bytes, max_age) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("store error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn route(args: &[String]) -> ExitCode {
    let mut port = 8000u16;
    let mut addr = "127.0.0.1".to_string();
    let mut prefill_url: Option<String> = None;
    let mut decode_urls: Vec<String> = Vec::new();
    let mut laya: Option<String> = None;
    let mut laya_shadow = false;
    let mut laya_required = false;
    let mut churn_threshold = 0.5f64;
    let mut min_confidence = 0.0f64;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port" => port = it.next().and_then(|v| v.parse().ok()).unwrap_or(port),
            "--addr" => addr = it.next().cloned().unwrap_or(addr),
            "--prefill-url" => prefill_url = it.next().cloned(),
            "--decode-url" => {
                if let Some(u) = it.next() {
                    decode_urls.push(u.clone());
                }
            }
            "--laya-url" => laya = it.next().cloned(),
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
    let Some(prefill_url) = prefill_url else {
        eprintln!("route requires --prefill-url http://host:port");
        return ExitCode::FAILURE;
    };
    if decode_urls.is_empty() {
        eprintln!("route requires at least one --decode-url http://host:port");
        return ExitCode::FAILURE;
    }
    let triage = match laya.as_deref() {
        Some(url) => match pagoda::triage::Triage::from_url(url) {
            Some(mut t) => {
                t.shadow = laya_shadow;
                t.required = laya_required;
                t.churn_threshold = churn_threshold;
                t.min_confidence = min_confidence;
                Some(Arc::new(t))
            }
            None => {
                eprintln!("bad --laya-url {url:?} (want http://host[:port])");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    match pd::run_router(
        &format!("{addr}:{port}"),
        &prefill_url,
        &decode_urls,
        triage,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("router error: {e}");
            ExitCode::FAILURE
        }
    }
}
