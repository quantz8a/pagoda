// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Real-weights pagoda server: HF tokenizer + Candle weights behind the
//! pagoda HTTP frontend, including Mooncake-style PD disaggregation.
//!
//! Unified serving:
//!
//!     cargo run --release --bin serve -- --repo <hf-repo> --port 8000
//!
//! PD split (store daemon comes from the pagoda crate):
//!
//!     pagoda store --port 9100
//!     serve --repo R --port 8001 --role prefill --store http://127.0.0.1:9100
//!     serve --repo R --port 8002 --role decode  --store http://127.0.0.1:9100 \
//!         --prefill-url http://127.0.0.1:8001
//!
//! Endpoints (same wire protocol as pagoda serve): GET /health, GET /stats,
//! POST /generate (text, sampling_params, stream?), POST /prefill (prefill
//! role), POST /v1/chat/completions, POST /v1/completions.

use std::process::ExitCode;
use std::sync::{Arc, Mutex};

use pagoda::{pd, Engine, EngineConfig, PdRole};
use pagoda_hf::{CandleModel, HfTokenizer};

struct Args {
    repo: String,
    addr: String,
    port: u16,
    role: PdRole,
    store_url: Option<String>,
    prefill_url: Option<String>,
    concurrent: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        repo: "hf-internal-testing/tiny-random-LlamaForCausalLM".to_string(),
        addr: "127.0.0.1".to_string(),
        port: 8000,
        role: PdRole::Unified,
        store_url: None,
        prefill_url: None,
        concurrent: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--concurrent" {
            args.concurrent = true;
            continue;
        }
        let Some(value) = it.next() else { break };
        match flag.as_str() {
            "--repo" => args.repo = value,
            "--addr" => args.addr = value,
            "--port" => args.port = value.parse().map_err(|_| "bad --port".to_string())?,
            "--role" => {
                args.role = match value.as_str() {
                    "unified" => PdRole::Unified,
                    "prefill" => PdRole::Prefill,
                    "decode" => PdRole::Decode,
                    other => return Err(format!("bad --role {other:?} (want unified|prefill|decode)")),
                }
            }
            "--store" => args.store_url = Some(value),
            "--kv-dtype" => match value.as_str() {
                "f32" => pagoda_hf::llama::set_kv_wire_f16(false),
                "f16" => pagoda_hf::llama::set_kv_wire_f16(true),
                other => return Err(format!("bad --kv-dtype {other:?} (want f32|f16)")),
            },
            "--prefill-url" => args.prefill_url = Some(value),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(args)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            eprintln!(
                "usage: serve [--repo <hf-repo>] [--addr 127.0.0.1] [--port 8000] \
                 [--role unified|prefill|decode] [--store http://host:port] \
                 [--prefill-url http://host:port] [--kv-dtype f32|f16]"
            );
            return ExitCode::FAILURE;
        }
    };

    eprintln!("==> loading tokenizer + weights from {}", args.repo);
    let tokenizer = match HfTokenizer::from_hub(&args.repo) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer load failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let device = match CandleModel::device_from_env() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("device init failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("==> device: {device:?}");
    let model = match CandleModel::llama_from_hub_on(&args.repo, device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("weight load failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut engine = Engine::new(
        tokenizer,
        model,
        EngineConfig {
            num_kv_blocks: 8192,
            block_size: 16,
            max_running_requests: 64,
            max_total_tokens: 8192,
            ..EngineConfig::default()
        },
    );

    if args.concurrent {
        if args.role != PdRole::Unified || args.store_url.is_some() || args.prefill_url.is_some()
        {
            eprintln!("--concurrent is unified-role only (no --role/--store/--prefill-url)");
            return ExitCode::FAILURE;
        }
        return match pagoda::server::run_concurrent(
            engine,
            &format!("{}:{}", args.addr, args.port),
            None,
        ) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("server error: {e}");
                ExitCode::FAILURE
            }
        };
    }

    if args.role != PdRole::Unified || args.store_url.is_some() {
        let Some(url) = args.store_url.as_deref() else {
            eprintln!("--role prefill|decode requires --store http://host:port");
            return ExitCode::FAILURE;
        };
        let Some(store) = pd::HttpStore::from_url(url) else {
            eprintln!("bad --store {url:?} (want http://host[:port])");
            return ExitCode::FAILURE;
        };
        engine.enable_pd(args.role, Arc::new(store));
        eprintln!("pd role: {} (kv store: {url})", args.role.as_str());
    }
    let conductor = match args.prefill_url.as_deref() {
        Some(url) => match pagoda::server::Conductor::from_url(url) {
            Some(c) => Some(Arc::new(c)),
            None => {
                eprintln!("bad --prefill-url {url:?} (want http://host[:port])");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };

    let engine = Arc::new(Mutex::new(engine));
    match pagoda::server::run_full(
        engine,
        &format!("{}:{}", args.addr, args.port),
        None,
        None,
        conductor,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("server error: {e}");
            ExitCode::FAILURE
        }
    }
}