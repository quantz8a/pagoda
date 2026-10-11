// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Real-weights pagoda server: HF tokenizer + Candle weights, or an MLX
//! HTTP session backend behind the pagoda Engine frontend.
//!
//! Candle (Llama / Qwen2.5):
//!
//!     cargo run --release --bin serve -- --backend candle --repo <hf-repo>
//!
//! MLX (Qwen3.8 via gpu gate — ModelSession adapter):
//!
//!     cargo run --release --bin serve -- --backend mlx \
//!         --upstream http://127.0.0.1:11236 \
//!         --tokenizer /path/to/tokenizer.json
//!
//! Endpoints: GET /health, GET /stats, POST /generate, checkpoint + OpenAI
//! chat/completions (same wire protocol as `pagoda serve`).

use std::path::Path;
use std::process::ExitCode;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pagoda::model::ModelEngine as _;
use pagoda::server::ServingEngine;
use pagoda::spec::{GenerationOutput, SamplingParams, WriteRequest};
use pagoda::tokenizer::Tokenizer as _;
use pagoda::{pd, Engine, EngineConfig, EngineStats, PdRole};
use pagoda_hf::{
    CandleModel, HfTokenizer, MlxApi, MlxHttpEngine, MlxUsageHandles, Qwen3_5Config,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Candle,
    Mlx,
}

struct Args {
    backend: Backend,
    repo: String,
    tokenizer: Option<String>,
    upstream: String,
    mlx_api: MlxApi,
    addr: String,
    port: u16,
    role: PdRole,
    store_url: Option<String>,
    prefill_url: Option<String>,
    concurrent: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        backend: Backend::Candle,
        repo: "hf-internal-testing/tiny-random-LlamaForCausalLM".to_string(),
        tokenizer: None,
        upstream: "http://127.0.0.1:11236".to_string(),
        mlx_api: MlxApi::Completions,
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
        let Some(value) = it.next() else {
            return Err(format!("flag {flag} expects a value"));
        };
        match flag.as_str() {
            "--backend" => {
                args.backend = match value.as_str() {
                    "candle" => Backend::Candle,
                    "mlx" => Backend::Mlx,
                    other => {
                        return Err(format!("bad --backend {other:?} (want candle|mlx)"));
                    }
                }
            }
            "--repo" => args.repo = value,
            "--tokenizer" => args.tokenizer = Some(value),
            "--upstream" => args.upstream = value,
            "--mlx-api" => {
                args.mlx_api = match value.as_str() {
                    "completions" | "v1" => MlxApi::Completions,
                    "generate" => MlxApi::Generate,
                    other => {
                        return Err(format!(
                            "bad --mlx-api {other:?} (want completions|generate)"
                        ));
                    }
                }
            }
            "--addr" => args.addr = value,
            "--port" => args.port = value.parse().map_err(|_| "bad --port".to_string())?,
            "--role" => {
                args.role = match value.as_str() {
                    "unified" => PdRole::Unified,
                    "prefill" => PdRole::Prefill,
                    "decode" => PdRole::Decode,
                    other => {
                        return Err(format!(
                            "bad --role {other:?} (want unified|prefill|decode)"
                        ));
                    }
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

fn usage() {
    eprintln!(
        "usage: serve [--backend candle|mlx] [--repo <hf-repo>] [--tokenizer <path|hub-repo>] \
         [--upstream http://host:port] [--mlx-api completions|generate] \
         [--addr 127.0.0.1] [--port 8000] \
         [--role unified|prefill|decode] [--store http://host:port] \
         [--prefill-url http://host:port] [--kv-dtype f32|f16] [--concurrent]"
    );
}

fn load_tokenizer(args: &Args) -> Result<HfTokenizer, String> {
    if let Some(spec) = &args.tokenizer {
        let p = Path::new(spec);
        if p.is_file() {
            return HfTokenizer::from_file(p).map_err(|e| e.to_string());
        }
        // Treat as a hub repo id, or a directory containing tokenizer.json.
        if p.is_dir() {
            return HfTokenizer::from_file(p.join("tokenizer.json")).map_err(|e| e.to_string());
        }
        return HfTokenizer::from_hub(spec).map_err(|e| e.to_string());
    }
    HfTokenizer::from_hub(&args.repo).map_err(|e| e.to_string())
}

fn maybe_redirect_qwen38_to_mlx(args: &mut Args) {
    if args.backend != Backend::Candle {
        return;
    }
    let cfg_path = if let Some(spec) = &args.tokenizer {
        let p = Path::new(spec);
        if p.is_file() {
            p.parent().map(|d| d.join("config.json"))
        } else if p.is_dir() {
            Some(p.join("config.json"))
        } else {
            HfTokenizer::download(spec, "config.json").ok()
        }
    } else {
        HfTokenizer::download(&args.repo, "config.json").ok()
    };
    let Some(cfg_path) = cfg_path else {
        return;
    };
    if let Ok(cfg) = Qwen3_5Config::from_file(&cfg_path) {
        if cfg.is_qwen3_5_family() {
            eprintln!(
                "==> Qwen3.5/3.8 hybrid detected; switching --backend mlx \
                 (Candle GDN path is WIP). upstream={}",
                args.upstream
            );
            args.backend = Backend::Mlx;
        }
    }
}

fn main() -> ExitCode {
    let mut args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            usage();
            return ExitCode::FAILURE;
        }
    };

    maybe_redirect_qwen38_to_mlx(&mut args);

    match args.backend {
        Backend::Mlx => run_mlx(args),
        Backend::Candle => run_candle(args),
    }
}

/// Engine + mlx usage handles so `GET /stats` exposes remote cache counters.
struct MlxServing {
    engine: Engine<MlxHttpEngine, HfTokenizer>,
    usage: MlxUsageHandles,
}

impl ServingEngine for MlxServing {
    fn generate(&mut self, req: &WriteRequest) -> GenerationOutput {
        self.engine.generate(req)
    }
    fn stats(&self) -> EngineStats {
        self.engine.stats()
    }
    fn extra_stats(&self) -> Vec<(String, pagoda::json::Value)> {
        self.usage.extra_stats()
    }
    fn create_checkpoint(&mut self, text: &str) -> Option<u64> {
        Some(self.engine.create_checkpoint(text).0)
    }
    fn drop_checkpoint(&mut self, id: u64) -> bool {
        self.engine.drop_checkpoint(pagoda::CheckpointId(id))
    }
    fn generate_from_checkpoint(
        &mut self,
        id: u64,
        continuation: &str,
        sampling: SamplingParams,
    ) -> Option<GenerationOutput> {
        self.engine
            .generate_from_checkpoint(pagoda::CheckpointId(id), continuation, sampling)
    }
}

fn run_mlx(args: Args) -> ExitCode {
    eprintln!(
        "==> mlx session backend → {} ({:?})",
        args.upstream, args.mlx_api
    );
    let tokenizer = match load_tokenizer(&args) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer load failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let model = match MlxHttpEngine::from_tokenizer(&args.upstream, tokenizer.clone(), args.mlx_api)
    {
        Ok(m) => m.with_timeout(Duration::from_secs(600)),
        Err(e) => {
            eprintln!("mlx engine init failed: {e:#}");
            return ExitCode::FAILURE;
        }
    };
    let usage = model.usage_handles();
    eprintln!(
        "==> tokenizer vocab={} model={}",
        tokenizer.vocab_size(),
        model.name()
    );

    let engine = Engine::new(
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
    let serving = MlxServing { engine, usage };
    if args.concurrent {
        eprintln!("--concurrent with mlx: extras stats use lock server; falling back to run_full");
    }
    if args.role != PdRole::Unified || args.store_url.is_some() {
        eprintln!("mlx backend is unified-role only in Phase 1 (no PD split yet)");
        return ExitCode::FAILURE;
    }
    let engine = Arc::new(Mutex::new(serving));
    match pagoda::server::run_full(
        engine,
        &format!("{}:{}", args.addr, args.port),
        None,
        None,
        None,
    ) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("server error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_candle(args: Args) -> ExitCode {
    eprintln!("==> loading tokenizer + Candle weights from {}", args.repo);
    if let Ok(cfg_path) = HfTokenizer::download(&args.repo, "config.json") {
        if let Ok(cfg) = Qwen3_5Config::from_file(&cfg_path) {
            if cfg.is_qwen3_5_family() {
                eprintln!(
                    "Qwen3.5/3.8 hybrid checkpoint detected. Use:\n  \
                     serve --backend mlx --upstream http://127.0.0.1:11236 \
                     --tokenizer <tokenizer.json>\n\
                     (or pass --repo/--tokenizer that resolves config.json and auto-switch.)"
                );
                return ExitCode::FAILURE;
            }
        }
    }
    let tokenizer = match load_tokenizer(&args) {
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
        if args.role != PdRole::Unified || args.store_url.is_some() || args.prefill_url.is_some() {
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

