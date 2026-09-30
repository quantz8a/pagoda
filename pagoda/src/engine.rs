// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Offline batch engine plus the continuous-batching scheduler.

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};

use crate::apc::ApcCache;
use crate::dsl::{Op, Program, StreamResult};
use crate::grammar::mask_logits;
use crate::kv_cache::{BlockId, PagedKvCache};
use crate::model::{ModelEngine, ModelSession, NGramModel};
use crate::radix_cache::{KvSlot, RadixCache};
use crate::sampler::{log_softmax, Sampler};
use crate::spec::{
    FinishReason, GenerationOutput, RejectReason, SamplingParams, TokenId, WriteRequest,
};
use crate::tokenizer::{ByteTokenizer, Tokenizer};

/// Admission / prefill ordering for the waiting queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SchedulePolicy {
    /// First-come, first-served (the original behavior).
    #[default]
    Fcfs,
    /// Materialize the request with the longest shared prefix first, maximizing
    /// cross-request reuse (best for TTFT / compute-saved under chat traffic).
    LongestPrefix,
    /// Materialize the request with the fewest remaining prompt tokens first,
    /// maximizing admission throughput (best for low tail latency).
    ShortestPrompt,
}

/// Which prefix-cache structure the engine consults on admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CacheBackend {
    /// Radix trie (token-granular walk, partial-tail reuse).
    #[default]
    Radix,
    /// Block-level chained-hash table (vLLM APC-style, block-granular reuse
    /// and eviction).
    Apc,
}

/// Runtime configuration.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// Number of physical KV pages.
    pub num_kv_blocks: usize,
    /// Tokens per page.
    pub block_size: usize,
    /// Admitted-request cap for a decode batch.
    pub max_running_requests: usize,
    /// Per-step token budget for prefill (chunked prefill).
    pub max_prefill_tokens_per_step: usize,
    /// RNG seed for the engine sampler.
    pub seed: u64,
    /// Waiting-queue ordering policy.
    pub schedule_policy: SchedulePolicy,
    /// Evict the least-recently-used radix prefix entries when the KV pool is
    /// exhausted instead of failing the request.
    pub evict_on_pressure: bool,
    /// Maximum number of requests admitted into the waiting queue of one batch.
    /// Requests beyond this cap are rejected with [`RejectReason::QueueFull`].
    pub max_waiting_requests: usize,
    /// Per-request SLO guard: a request whose `prompt + max_tokens` exceeds
    /// this cap is rejected with [`RejectReason::TooLong`].
    pub max_total_tokens: usize,
    /// Prefix-cache structure consulted on admission.
    pub cache_backend: CacheBackend,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_kv_blocks: 4096,
            block_size: 16,
            max_running_requests: 32,
            max_prefill_tokens_per_step: 32,
            seed: 0,
            schedule_policy: SchedulePolicy::default(),
            evict_on_pressure: true,
            max_waiting_requests: 256,
            max_total_tokens: 4096,
            cache_backend: CacheBackend::default(),
        }
    }
}

/// A token production loop independent of the model/tokenizer backends.
pub struct Engine<M: ModelEngine, T: Tokenizer> {
    tokenizer: T,
    model: M,
    sampler: Sampler,
    kv: PagedKvCache,
    radix: RadixCache,
    apc: ApcCache,
    config: EngineConfig,
    next_request_id: usize,
    max_prefill_tokens_per_step: usize,
    checkpoints: HashMap<u64, Checkpoint>,
    next_checkpoint_id: u64,
    total_requests: u64,
    total_forward: u64,
    total_prefill_chunks: u64,
    total_prompt_tokens: u64,
    total_prefill_tokens: u64,
    total_output_tokens: u64,
    total_faults: u64,
    total_rejected: u64,
    total_checkpoint_hits: u64,
}

/// Handle to a pinned KV checkpoint: a materialized prompt prefix that later
/// requests can branch from with a guaranteed full prefix hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CheckpointId(pub u64);

/// A pinned prefix: logical tokens, their physical slots, and the deduped
/// block list whose references the checkpoint holds as its pin.
struct Checkpoint {
    tokens: Vec<TokenId>,
    locs: Vec<KvSlot>,
    blocks: Vec<BlockId>,
    /// Model-side KV snapshot of the trunk, when the backend supports
    /// sessions: branches fork it instead of recomputing the prefix.
    session: Option<Box<dyn ModelSession>>,
}

/// The battery out-of-the-box engine: byte tokenizer + n-gram toy model.
pub type ToyEngine = Engine<NGramModel, ByteTokenizer>;

struct Seq {
    id: usize,
    sampling: SamplingParams,
    /// Full prompt tokens (logical).
    prompt: Vec<TokenId>,
    /// Logical buffer materialized so far (prefix + prefilled suffix), then generated tokens.
    tokens: Vec<TokenId>,
    /// Generated tokens only.
    output: Vec<TokenId>,
    prompt_len: usize,
    prefix_hit: usize,
    prefill_cost: usize,
    /// How many prompt tokens have been materialized into KV pages so far.
    prefill_pos: usize,
    /// Physical pages holding `tokens`.
    blocks: Vec<BlockId>,
    /// Physical slot of each token in `tokens` (parallel array).
    locs: Vec<KvSlot>,
    /// Incremental decode session (models with a real KV cache). `None` keeps
    /// the stateless full-replay contract.
    session: Option<Box<dyn ModelSession>>,
}

enum Advance {
    Running(Seq),
    Done(GenerationOutput),
}

struct Job {
    vars: HashMap<String, String>,
    transcript: String,
    ops: Vec<Op>,
}

impl<M: ModelEngine, T: Tokenizer> Engine<M, T> {
    pub fn new(tokenizer: T, model: M, config: EngineConfig) -> Self {
        let sampler = Sampler::new(config.seed);
        let kv = PagedKvCache::new(config.num_kv_blocks, config.block_size);
        let max_prefill = config.max_prefill_tokens_per_step.max(1);
        Self {
            tokenizer,
            model,
            sampler,
            kv,
            radix: RadixCache::new(),
            apc: ApcCache::new(),
            config,
            next_request_id: 0,
            max_prefill_tokens_per_step: max_prefill,
            checkpoints: HashMap::new(),
            next_checkpoint_id: 0,
            total_requests: 0,
            total_forward: 0,
            total_prefill_chunks: 0,
            total_prompt_tokens: 0,
            total_prefill_tokens: 0,
            total_output_tokens: 0,
            total_faults: 0,
            total_rejected: 0,
            total_checkpoint_hits: 0,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }
    pub fn tokenizer(&self) -> &T {
        &self.tokenizer
    }
    pub fn model(&self) -> &M {
        &self.model
    }
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub fn stats(&self) -> EngineStats {
        EngineStats {
            total_requests: self.total_requests,
            total_forward: self.total_forward,
            radix_nodes: self.radix.num_nodes(),
            radix_queries: self.radix.num_queries() as u64,
            radix_hit_queries: self.radix.hit_queries() as u64,
            radix_hit_tokens: self.radix.hit_tokens() as u64,
            apc_blocks: self.apc.num_blocks(),
            apc_hit_tokens: self.apc.hit_tokens() as u64,
            kv_blocks: self.kv.num_blocks(),
            kv_free_blocks: self.kv.num_free_blocks(),
            kv_allocations: self.kv.num_allocations(),
            kv_frees: self.kv.num_frees(),
            prefill_chunks: self.total_prefill_chunks,
            total_prompt_tokens: self.total_prompt_tokens,
            total_prefill_tokens: self.total_prefill_tokens,
            total_output_tokens: self.total_output_tokens,
            faulted_requests: self.total_faults,
            rejected_requests: self.total_rejected,
            active_checkpoints: self.checkpoints.len(),
            checkpoint_hit_tokens: self.total_checkpoint_hits,
        }
    }

    /// Fraction of prefix-cache queries that hit a cached prefix.
    pub fn radix_hit_rate(&self) -> f64 {
        self.radix.cache_hit_rate()
    }

    /// Generate for a single request. Reseeds the sampler from the request so
    /// output is reproducible per request.
    pub fn generate(&mut self, req: &WriteRequest) -> GenerationOutput {
        self.sampler.seed(req.sampling.seed);
        self.generate_batch(std::slice::from_ref(req))
            .pop()
            .expect("generate_batch returned no output")
    }

    /// Continuous (dynamic) batching with chunked prefill: each scheduler step
    /// spends up to `max_prefill_tokens_per_step` materializing waiting prompts
    /// (long prompts are split across steps), then decodes one token for every
    /// ready sequence, until both the waiting queue and the decode batch drain.
    pub fn generate_batch(&mut self, reqs: &[WriteRequest]) -> Vec<GenerationOutput> {
        if let Some(first) = reqs.first() {
            self.sampler.seed(first.sampling.seed);
        }

        let mut waiting: VecDeque<Seq> = VecDeque::with_capacity(reqs.len());
        let mut outputs = Vec::with_capacity(reqs.len());
        for req in reqs {
            if waiting.len() >= self.config.max_waiting_requests {
                outputs.push(self.reject(RejectReason::QueueFull, 0));
                continue;
            }
            match self.admit(req) {
                Ok(seq) => waiting.push_back(seq),
                Err(rejected) => outputs.push(rejected),
            }
        }
        self.drain(waiting, outputs)
    }

    /// Continuous-batching loop: prefill waiting sequences under the per-step
    /// token budget, decode one token for every ready sequence, until both the
    /// waiting queue and the decode batch drain.
    fn drain(
        &mut self,
        mut waiting: VecDeque<Seq>,
        mut outputs: Vec<GenerationOutput>,
    ) -> Vec<GenerationOutput> {
        let mut ready: VecDeque<Seq> = VecDeque::new();
        let mut active: Vec<Seq> = Vec::new();

        loop {
            // Prefill phase: spend the per-step token budget.
            let mut budget = self.max_prefill_tokens_per_step;
            while budget > 0 {
                let mut seq = match pop_next(&mut waiting, self.config.schedule_policy) {
                    Some(s) => s,
                    None => break,
                };
                let remaining = seq.prompt.len().saturating_sub(seq.prefill_pos);
                let chunk = remaining.min(budget);
                if chunk == 0 {
                    // Entire prompt already served from the prefix cache.
                    ready.push_back(seq);
                    continue;
                }
                self.prefill_chunk(&mut seq, chunk);
                budget -= chunk;
                if seq.prefill_pos >= seq.prompt.len() {
                    ready.push_back(seq);
                } else {
                    waiting.push_back(seq);
                }
            }

            // Admit fully prefilled sequences while there is decode room.
            while active.len() < self.config.max_running_requests {
                match ready.pop_front() {
                    Some(seq) => active.push(seq),
                    None => break,
                }
            }

            if active.is_empty() && ready.is_empty() && waiting.is_empty() {
                break;
            }

            // Decode phase: one token for every ready sequence.
            let mut next = Vec::with_capacity(active.len());
            for seq in active.drain(..) {
                match self.advance(seq) {
                    Advance::Running(s) => next.push(s),
                    Advance::Done(out) => outputs.push(out),
                }
            }
            active = next;
        }
        outputs
    }

    /// Materialize `text` into the KV cache, publish it into the prefix cache,
    /// and pin its physical blocks so later branches always get a full prefix
    /// hit. This is the primitive for agent-tree / parallel-branch workloads:
    /// create the shared trunk once, then branch many continuations off it
    /// without re-prefill and without fear of eviction.
    pub fn create_checkpoint(&mut self, text: &str) -> CheckpointId {
        let tokens = self.tokenizer.encode(text);
        let (prefix_hit, prefix_slots) = self.match_prefix(&tokens);
        let mut seq = Seq {
            id: usize::MAX,
            sampling: SamplingParams::default(),
            prompt: tokens.clone(),
            tokens: tokens[..prefix_hit].to_vec(),
            output: Vec::new(),
            prompt_len: tokens.len(),
            prefix_hit,
            prefill_cost: tokens.len() - prefix_hit,
            prefill_pos: prefix_hit,
            blocks: Vec::new(),
            locs: Vec::with_capacity(tokens.len()),
            session: None,
        };
        for slot in &prefix_slots {
            let block = slot.0;
            if seq.blocks.last().copied() != Some(block) {
                self.kv.inc_ref(block);
                seq.blocks.push(block);
            }
        }
        seq.locs.extend_from_slice(&prefix_slots);
        while seq.prefill_pos < seq.prompt.len() {
            let chunk = (seq.prompt.len() - seq.prefill_pos).min(self.max_prefill_tokens_per_step);
            self.prefill_chunk(&mut seq, chunk);
        }
        self.publish_prefix(&seq.tokens, &seq.locs);
        // Model-side pinned trunk: push the whole checkpoint through one
        // session so branches can fork its KV state instead of recomputing
        // the prefix. Backends without sessions keep the stateless path.
        let session = self.model.begin_session().map(|mut s| {
            s.forward(&seq.tokens);
            s
        });
        // The request-side references are kept as the checkpoint pin: the
        // blocks stay resident (immune to prefix-cache eviction) until
        // `drop_checkpoint`.
        let id = CheckpointId(self.next_checkpoint_id);
        self.next_checkpoint_id += 1;
        self.checkpoints.insert(
            id.0,
            Checkpoint {
                tokens: seq.tokens,
                locs: seq.locs,
                blocks: seq.blocks,
                session,
            },
        );
        id
    }

    /// Number of currently pinned checkpoints.
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    /// Token count of a checkpoint's pinned prefix.
    pub fn checkpoint_tokens(&self, id: CheckpointId) -> Option<usize> {
        self.checkpoints.get(&id.0).map(|c| c.tokens.len())
    }

    /// Release a checkpoint pin. The blocks stay in the prefix cache and
    /// remain reusable until ordinary eviction reclaims them. Returns `false`
    /// when no such checkpoint exists.
    pub fn drop_checkpoint(&mut self, id: CheckpointId) -> bool {
        let Some(cp) = self.checkpoints.remove(&id.0) else {
            return false;
        };
        for b in cp.blocks {
            self.kv.dec_ref(b);
        }
        true
    }

    /// Branch generation off a pinned checkpoint: the checkpoint tokens act as
    /// the prompt prefix with a guaranteed full hit (no prefix-cache walk,
    /// immune to eviction), and `continuation` is appended on top. Returns
    /// `None` when the checkpoint does not exist.
    pub fn generate_from_checkpoint(
        &mut self,
        id: CheckpointId,
        continuation: &str,
        sampling: SamplingParams,
    ) -> Option<GenerationOutput> {
        let (tokens, locs, blocks, session) = {
            let cp = self.checkpoints.get(&id.0)?;
            (
                cp.tokens.clone(),
                cp.locs.clone(),
                cp.blocks.clone(),
                // Branch off the checkpoint's KV snapshot when the backend
                // supports forking; the branch then only pays for the
                // continuation, not the trunk.
                cp.session.as_ref().and_then(|s| s.fork()),
            )
        };
        self.sampler.seed(sampling.seed);
        let max_new = sampling.max_tokens;
        let mut prompt = tokens.clone();
        prompt.extend_from_slice(&self.tokenizer.encode(continuation));
        let prompt_len = prompt.len();
        let reason = if sampling.grammar.is_some() && !self.tokenizer.is_byte_level() {
            Some(RejectReason::Unsupported)
        } else if prompt_len.saturating_add(max_new) > self.config.max_total_tokens {
            Some(RejectReason::TooLong)
        } else {
            None
        };
        if let Some(reason) = reason {
            return Some(self.reject(reason, prompt_len));
        }

        // Take branch-side references on the pinned blocks.
        for &b in &blocks {
            self.kv.inc_ref(b);
        }
        let prefix_hit = tokens.len();
        let prefill_cost = prompt_len - prefix_hit;
        self.total_forward += prefill_cost as u64;
        self.total_prompt_tokens += prompt_len as u64;
        self.total_prefill_tokens += prefill_cost as u64;
        self.total_requests += 1;
        self.total_checkpoint_hits += prefix_hit as u64;

        let rid = self.next_request_id;
        self.next_request_id += 1;
        let seq = Seq {
            id: rid,
            sampling,
            prompt,
            tokens,
            output: Vec::with_capacity(max_new),
            prompt_len,
            prefix_hit,
            prefill_cost,
            prefill_pos: prefix_hit,
            blocks,
            locs,
            session: session.or_else(|| self.model.begin_session()),
        };
        let mut waiting = VecDeque::new();
        waiting.push_back(seq);
        let mut outputs = self.drain(waiting, Vec::new());
        Some(outputs.pop().expect("one sequence yields one output"))
    }

    /// Build a rejected-output marker and charge the rejection metric.
    fn reject(&mut self, reason: RejectReason, prompt_tokens: usize) -> GenerationOutput {
        self.total_rejected += 1;
        GenerationOutput {
            text: String::new(),
            full_text: None,
            output_token_ids: Vec::new(),
            finish_reason: FinishReason::Rejected,
            prompt_tokens,
            prefix_hit_tokens: 0,
            forward_count: 0,
            rejection: Some(reason),
        }
    }

    /// Encode, apply admission / SLO guards, match the prefix cache, and take
    /// references to the shared blocks. Returns a not-yet-prefilled sequence for
    /// accepted requests, or a `Rejected` [`GenerationOutput`] otherwise.
    fn admit(&mut self, req: &WriteRequest) -> Result<Seq, GenerationOutput> {
        let prompt = self.tokenizer.encode(&req.text);
        let prompt_len = prompt.len();
        let max_new = req.sampling.max_tokens;

        let reason = if req.sampling.grammar.is_some() && !self.tokenizer.is_byte_level() {
            Some(RejectReason::Unsupported)
        } else if prompt_len == 0 && req.sampling.grammar.is_none() {
            Some(RejectReason::EmptyPrompt)
        } else if prompt_len.saturating_add(max_new) > self.config.max_total_tokens {
            Some(RejectReason::TooLong)
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(self.reject(reason, prompt_len));
        }

        let id = self.next_request_id;
        self.next_request_id += 1;
        let (prefix_hit, prefix_slots) = self.match_prefix(&prompt);
        let prefill_cost = prompt_len.saturating_sub(prefix_hit);
        self.total_forward += prefill_cost as u64;
        self.total_prompt_tokens += prompt_len as u64;
        self.total_prefill_tokens += prefill_cost as u64;

        let mut seq = Seq {
            id,
            sampling: req.sampling.clone(),
            prompt: prompt.clone(),
            tokens: prompt[..prefix_hit].to_vec(),
            output: Vec::with_capacity(max_new),
            prompt_len,
            prefix_hit,
            prefill_cost,
            prefill_pos: prefix_hit,
            blocks: Vec::new(),
            locs: Vec::with_capacity(prompt_len + max_new),
            session: self.model.begin_session(),
        };

        for slot in &prefix_slots {
            let block = slot.0;
            if seq.blocks.last().copied() != Some(block) {
                self.kv.inc_ref(block);
                seq.blocks.push(block);
            }
        }
        seq.locs.extend_from_slice(&prefix_slots);

        self.total_requests += 1;
        Ok(seq)
    }

    /// Materialize the next `n` prompt tokens into copy-on-write paged memory.
    fn prefill_chunk(&mut self, seq: &mut Seq, n: usize) {
        if n == 0 {
            return;
        }
        let end = seq.prefill_pos + n;
        let chunk = seq.prompt[seq.prefill_pos..end].to_vec();
        for tok in &chunk {
            self.append_physical(seq, *tok);
        }
        seq.tokens.extend_from_slice(&chunk);
        seq.prefill_pos = end;
        self.total_prefill_chunks += 1;
    }
    /// Score candidate continuations for SGLang-style `select`.
    /// Returns `(choice, mean_log_prob)` pairs.
    pub fn score(&mut self, context_text: &str, choices: &[String]) -> Vec<(String, f32)> {
        let ctx = self.tokenizer.encode(context_text);
        choices
            .iter()
            .map(|choice| {
                let toks = self.tokenizer.encode(choice);
                let mut prefix = ctx.clone();
                let mut total = 0.0f32;
                for &t in &toks {
                    self.total_forward += 1;
                    let logits = self.model.forward(&prefix);
                    let lp = log_softmax(&logits);
                    let idx = t as usize;
                    total += if idx < lp.len() { lp[idx] } else { -10.0 };
                    prefix.push(t);
                }
                let mean = if toks.is_empty() {
                    f32::NEG_INFINITY
                } else {
                    total / toks.len() as f32
                };
                (choice.clone(), mean)
            })
            .collect()
    }

    /// Execute an SGLang-style frontend program. Returns one [`StreamResult`]
    /// per forked branch.
    pub fn run_program(&mut self, program: &Program) -> Vec<StreamResult> {
        let mut jobs = VecDeque::new();
        jobs.push_back(Job {
            vars: HashMap::new(),
            transcript: String::new(),
            ops: program.ops.clone(),
        });

        let mut results = Vec::new();
        while let Some(mut job) = jobs.pop_front() {
            if job.ops.is_empty() {
                results.push(StreamResult {
                    vars: job.vars,
                    transcript: job.transcript,
                });
                continue;
            }

            let op = job.ops.remove(0);
            match op {
                Op::System(text) => {
                    job.transcript.push_str(&format!("[SYS] {text}\n"));
                    jobs.push_back(job);
                }
                Op::User(text) => {
                    job.transcript.push_str(&format!("[USER] {text}\n"));
                    jobs.push_back(job);
                }
                Op::Assistant(text) => {
                    job.transcript.push_str(&format!("[ASSISTANT] {text}\n"));
                    jobs.push_back(job);
                }
                Op::Gen { name, params } => {
                    let out = self.generate(&WriteRequest::new(job.transcript.clone(), params));
                    job.vars.insert(name, out.text.clone());
                    job.transcript.push_str(&format!("[ASSISTANT] {}\n", out.text));
                    jobs.push_back(job);
                }
                Op::Select {
                    name,
                    choices,
                    params: _params,
                } => {
                    let scores = self.score(&job.transcript, &choices);
                    let best = scores
                        .into_iter()
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                        .unwrap_or((String::new(), f32::NEG_INFINITY));
                    job.vars.insert(name, best.0.clone());
                    job.transcript.push_str(&format!("[ASSISTANT] {}\n", best.0));
                    jobs.push_back(job);
                }
                Op::Fork { branches } => {
                    let remaining = job.ops.clone();
                    for branch in branches {
                        let mut ops = branch.ops.clone();
                        ops.extend(remaining.clone());
                        jobs.push_back(Job {
                            vars: job.vars.clone(),
                            transcript: job.transcript.clone(),
                            ops,
                        });
                    }
                }
            }
        }
        results
    }


    // --- internals -----------------------------------------------------------

    /// Consult the configured prefix cache for `prompt`, returning the number
    /// of reusable tokens and the physical slot of each, in order.
    fn match_prefix(&mut self, prompt: &[TokenId]) -> (usize, Vec<KvSlot>) {
        match self.config.cache_backend {
            CacheBackend::Radix => {
                let (hit, slots) = self.radix.match_path(prompt);
                self.radix.record_matched(hit);
                (hit, slots)
            }
            CacheBackend::Apc => {
                let bs = self.kv.block_size();
                let (hit, blocks) = self.apc.match_blocks(prompt, bs);
                let slots: Vec<KvSlot> = blocks
                    .iter()
                    .flat_map(|b| (0..bs).map(move |off| (*b, off)))
                    .collect();
                debug_assert_eq!(hit, slots.len());
                (hit, slots)
            }
        }
    }

    /// Publish a finished token path (with physical slots) into the configured
    /// prefix cache, adopting cache-owned references on newly indexed blocks.
    fn publish_prefix(&mut self, tokens: &[TokenId], locs: &[KvSlot]) {
        match self.config.cache_backend {
            CacheBackend::Radix => {
                let _ = self.radix.insert_with_kv(tokens, locs, &mut self.kv);
            }
            CacheBackend::Apc => self.apc.insert_blocks(tokens, locs, &mut self.kv),
        }
    }

    /// Allocate a KV block, evicting LRU prefix-cache entries on pressure when
    /// enabled. Eviction repeats until a block is actually freed: evicting one
    /// radix leaf does not necessarily release its block (sibling nodes may
    /// still reference it), and pinned checkpoint blocks are never freed.
    fn alloc_kv_block(&mut self) -> Option<BlockId> {
        if let Some(block) = self.kv.alloc() {
            return Some(block);
        }
        if self.config.evict_on_pressure {
            loop {
                let evicted = match self.config.cache_backend {
                    CacheBackend::Radix => self.radix.evict_lru(&mut self.kv, 1),
                    CacheBackend::Apc => self.apc.evict_lru(&mut self.kv, 1),
                };
                if let Some(block) = self.kv.alloc() {
                    return Some(block);
                }
                if evicted == 0 {
                    // Nothing left to evict: the pool is genuinely exhausted.
                    return None;
                }
            }
        }
        None
    }

    /// Append one token to a sequence's physical pages, copy-on-writing the
    /// tail block when it is shared. Only updates `blocks`/`locs`, not the
    /// logical token buffer.
    fn append_physical(&mut self, seq: &mut Seq, tok: TokenId) {
        match seq.blocks.last().copied() {
            Some(tail) if !self.kv.is_full(tail) => {
                if self.kv.ref_count(tail) > 1 {
                    // Copy-on-write: the tail block is shared with the prefix
                    // cache or a checkpoint pin. Allocate the private copy
                    // through the engine allocator so eviction policy applies,
                    // then move this sequence's tail slots onto the copy (the
                    // prefix content stays identical in the shared block).
                    // Note the shared block may hold *more* tokens than this
                    // sequence owns (the cached chain may continue past the
                    // matched prefix), so only copy the owned head.
                    let owned = seq.locs.iter().rev().take_while(|s| s.0 == tail).count();
                    debug_assert!(owned > 0, "tail block must hold >=1 token of this sequence");
                    let private = self
                        .alloc_kv_block()
                        .expect("KV cache exhausted: raise num_kv_blocks");
                    let copied: Vec<TokenId> = self.kv.tokens(tail)[..owned].to_vec();
                    for t in &copied {
                        let _ = self.kv.append(private, *t);
                    }
                    self.kv.dec_ref(tail);
                    *seq.blocks.last_mut().expect("tail block exists") = private;
                    for slot in seq.locs.iter_mut().rev().take(owned) {
                        debug_assert_eq!(slot.0, tail);
                        *slot = (private, slot.1);
                    }
                    let _ = self.kv.append(private, tok);
                    seq.locs.push((private, owned));
                } else {
                    let off = self.kv.len(tail);
                    let _ = self.kv.append(tail, tok);
                    seq.locs.push((tail, off));
                }
            }
            _ => {
                let nb = self
                    .alloc_kv_block()
                    .expect("KV cache exhausted: raise num_kv_blocks");
                seq.blocks.push(nb);
                let _ = self.kv.append(nb, tok);
                seq.locs.push((nb, 0));
            }
        }
    }

    /// A model backend returns sane logits when they are exactly vocab-sized
    /// and contain no NaN / infinity. Anything else is a request-level fault:
    /// the affected sequence is terminated in isolation rather than poisoning
    /// the rest of the batch.
    fn logits_sane(&self, logits: &[f32]) -> bool {
        logits.len() == self.model.vocab_size() && logits.iter().all(|x| x.is_finite())
    }

    fn advance(&mut self, mut seq: Seq) -> Advance {
        if seq.output.len() >= seq.sampling.max_tokens {
            return Advance::Done(self.finalize(seq, FinishReason::Length));
        }

        self.total_forward += 1;
        // Incremental path: a session-backed model only receives the tokens it
        // has not seen yet (the whole prompt on the first step, then one fresh
        // token per step). Stateless models keep the full-replay contract.
        let use_session = matches!(
            seq.session.as_ref(),
            Some(s) if s.context_len() <= seq.tokens.len()
        );
        let mut logits = if use_session {
            let session = seq.session.as_mut().expect("session checked above");
            let fed = session.context_len();
            session.forward(&seq.tokens[fed..])
        } else {
            // A misbehaving session (context ahead of the sequence) is dropped
            // rather than panicking the engine.
            seq.session = None;
            self.model.forward(&seq.tokens)
        };
        if !self.logits_sane(&logits) {
            self.total_faults += 1;
            return Advance::Done(self.finalize(seq, FinishReason::Fault));
        }
        apply_penalties(&mut logits, &seq.output, &seq.sampling);
        if let Some(grammar) = seq.sampling.grammar.as_ref() {
            let partial = self.tokenizer.decode(&seq.output);
            let Some(allowed) = grammar.allowed_bytes(&partial) else {
                // The partial output can no longer be completed: stop instead
                // of sampling a token that violates the constraint.
                return Advance::Done(self.finalize(seq, FinishReason::Stop));
            };
            mask_logits(&allowed, &mut logits);
            // Once a valid complete result is available, keep the end-of-stream
            // token as a legal continuation so the model can finish here.
            if grammar.is_complete(&partial) {
                let eos = self.tokenizer.eos_token_id() as usize;
                if eos < logits.len() && !logits[eos].is_finite() {
                    logits[eos] = 0.0;
                }
            }
        }
        let tok = self.sampler.sample(&logits, &seq.sampling);
        self.append_physical(&mut seq, tok);
        seq.tokens.push(tok);
        seq.output.push(tok);

        if let Some(reason) = self.finish_reason(&seq) {
            return Advance::Done(self.finalize(seq, reason));
        }
        Advance::Running(seq)
    }

    fn finish_reason(&self, seq: &Seq) -> Option<FinishReason> {
        if let Some(&last) = seq.output.last() {
            if last == self.tokenizer.eos_token_id()
                || seq.sampling.stop_token_ids.contains(&last)
            {
                return Some(FinishReason::Stop);
            }
        }
        if !seq.sampling.stop.is_empty() {
            let text = self.tokenizer.decode(&seq.output);
            if seq
                .sampling
                .stop
                .iter()
                .any(|s| !s.is_empty() && text.ends_with(s))
            {
                return Some(FinishReason::Stop);
            }
        }
        None
    }

    fn finalize(&mut self, seq: Seq, reason: FinishReason) -> GenerationOutput {
        // Index the full token path into the prefix cache, adopting new blocks
        // with a cache base reference, then release this request's reference
        // on every block it holds.
        self.publish_prefix(&seq.tokens, &seq.locs);
        for &b in &seq.blocks {
            self.kv.dec_ref(b);
        }

        let text = self.tokenizer.decode(&seq.output);
        let full = self.tokenizer.decode(&seq.tokens);
        let out_len = seq.output.len();
        self.total_output_tokens += out_len as u64;
        GenerationOutput {
            text,
            full_text: Some(full),
            output_token_ids: seq.output,
            finish_reason: reason,
            prompt_tokens: seq.prompt_len,
            prefix_hit_tokens: seq.prefix_hit,
            forward_count: seq.prefill_cost + out_len,
            rejection: None,
        }
    }
}
impl Engine<NGramModel, ByteTokenizer> {
    /// Build the battery-included engine (byte tokenizer + n-gram toy model).
    pub fn toy(config: EngineConfig) -> Self {
        Self::new(ByteTokenizer::new(), NGramModel::default(), config)
    }

    /// Build the toy engine with a custom n-gram order and training corpus.
    pub fn toy_with_corpus(config: EngineConfig, order: usize, corpus: &str) -> Self {
        Self::new(ByteTokenizer::new(), NGramModel::new(order, corpus), config)
    }
}

fn apply_penalties(logits: &mut [f32], output: &[TokenId], params: &SamplingParams) {
    if params.frequency_penalty.abs() < 1e-6 && params.presence_penalty.abs() < 1e-6 {
        return;
    }
    let mut counts: HashMap<TokenId, usize> = HashMap::new();
    for &t in output {
        *counts.entry(t).or_insert(0) += 1;
    }
    for (&t, &c) in counts.iter() {
        if (t as usize) < logits.len() {
            logits[t as usize] -= params.frequency_penalty * (c as f32);
        }
    }
    if params.presence_penalty.abs() > 1e-6 {
        for &t in counts.keys() {
            if (t as usize) < logits.len() {
                logits[t as usize] -= params.presence_penalty;
            }
        }
    }
}

/// Pop the next sequence to prefill according to the configured policy.
fn pop_next(waiting: &mut VecDeque<Seq>, policy: SchedulePolicy) -> Option<Seq> {
    if waiting.is_empty() {
        return None;
    }
    match policy {
        SchedulePolicy::Fcfs => waiting.pop_front(),
        SchedulePolicy::LongestPrefix => {
            let mut best = 0usize;
            let mut best_score = 0usize;
            for (i, s) in waiting.iter().enumerate() {
                if s.prefix_hit > best_score {
                    best_score = s.prefix_hit;
                    best = i;
                }
            }
            waiting.remove(best)
        }
        SchedulePolicy::ShortestPrompt => {
            let mut best = 0usize;
            let mut best_score = usize::MAX;
            for (i, s) in waiting.iter().enumerate() {
                let remaining = s.prompt.len().saturating_sub(s.prefill_pos);
                if remaining < best_score {
                    best_score = remaining;
                    best = i;
                }
            }
            waiting.remove(best)
        }
    }
}

/// Aggregated runtime metrics.
#[derive(Clone, Copy, Debug)]
pub struct EngineStats {
    pub total_requests: u64,
    pub total_forward: u64,
    pub radix_nodes: usize,
    pub radix_queries: u64,
    pub radix_hit_queries: u64,
    pub radix_hit_tokens: u64,
    /// Blocks currently indexed by the APC hash table (APC backend only).
    pub apc_blocks: usize,
    /// Tokens served from the APC cache (APC backend only).
    pub apc_hit_tokens: u64,
    pub kv_blocks: usize,
    pub kv_free_blocks: usize,
    pub kv_allocations: usize,
    pub kv_frees: usize,
    /// Number of distinct prefill chunk materializations.
    pub prefill_chunks: u64,
    /// Prompt tokens submitted (sum of request prompt lengths).
    pub total_prompt_tokens: u64,
    /// Prompt tokens actually materialized into KV (prefill work performed).
    pub total_prefill_tokens: u64,
    /// Newly generated tokens.
    pub total_output_tokens: u64,
    /// Requests aborted because the model returned invalid logits.
    pub faulted_requests: u64,
    /// Requests refused during admission (queue full / too long / empty prompt).
    pub rejected_requests: u64,
    /// Currently pinned KV checkpoints.
    pub active_checkpoints: usize,
    /// Prompt tokens served from pinned checkpoints (guaranteed-hit branches).
    pub checkpoint_hit_tokens: u64,
}

impl EngineStats {
    /// Compute-saved tokens: prompt tokens served from the prefix cache (radix
    /// or APC, whichever backend is active) or from pinned checkpoints, rather
    /// than re-materialized. The revenue head-line metric for "省算力".
    pub fn compute_saved_tokens(&self) -> u64 {
        self.radix_hit_tokens + self.apc_hit_tokens + self.checkpoint_hit_tokens
    }

    /// Fraction of prefill work that the prefix cache skipped.
    pub fn prefill_skip_ratio(&self) -> f64 {
        let saved = self.compute_saved_tokens();
        let denom = saved + self.total_prefill_tokens;
        if denom == 0 {
            0.0
        } else {
            saved as f64 / denom as f64
        }
    }

    /// Forward-pass cost per generated token (a token/watt proxy: lower is
    /// better).
    pub fn avg_forward_per_output_token(&self) -> f64 {
        if self.total_output_tokens == 0 {
            0.0
        } else {
            self.total_forward as f64 / self.total_output_tokens as f64
        }
    }

    /// Fraction of the KV pool currently resident.
    pub fn kv_utilization(&self) -> f64 {
        if self.kv_blocks == 0 {
            0.0
        } else {
            (self.kv_blocks - self.kv_free_blocks) as f64 / self.kv_blocks as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> ToyEngine {
        ToyEngine::toy(EngineConfig {
            num_kv_blocks: 64,
            block_size: 8,
            ..EngineConfig::default()
        })
    }

    fn params(max_tokens: usize) -> SamplingParams {
        SamplingParams {
            max_tokens,
            ..SamplingParams::default()
        }
    }

    #[test]
    fn branch_appends_never_touch_checkpoint_blocks() {
        let mut e = engine();
        // 31 bytes: 3 full blocks + a partial tail block of 7 tokens.
        let cp = e.create_checkpoint("checkpoint with a partial tail");
        let tail = e.checkpoints[&cp.0].blocks.last().copied().expect("tail");
        let tail_len = e.kv.len(tail);
        assert!(tail_len < e.kv.block_size());

        let out = e
            .generate_from_checkpoint(cp, " + branch A continuation", params(8))
            .expect("branch");
        assert!(matches!(
            out.finish_reason,
            FinishReason::Stop | FinishReason::Length
        ));
        assert_eq!(
            e.kv.len(tail),
            tail_len,
            "copy-on-write must keep the checkpoint's tail block immutable"
        );
    }

    #[test]
    fn completed_branch_leaves_no_dangling_references() {
        let mut e = engine();
        let cp = e.create_checkpoint("shared trunk for reference accounting");
        let blocks = e.checkpoints[&cp.0].blocks.clone();
        let before: Vec<usize> = blocks.iter().map(|&b| e.kv.ref_count(b)).collect();

        let out = e
            .generate_from_checkpoint(cp, "branch once", params(6))
            .expect("branch");
        assert!(matches!(
            out.finish_reason,
            FinishReason::Stop | FinishReason::Length
        ));

        let after: Vec<usize> = blocks.iter().map(|&b| e.kv.ref_count(b)).collect();
        assert_eq!(
            before, after,
            "a finished branch must release exactly the references it took"
        );
    }

    #[test]
    fn frequency_penalty_scales_with_occurrence_count() {
        let mut logits = vec![0.0f32; 8];
        logits[3] = 5.0;
        let output = vec![3u32, 7, 3]; // token 3 seen twice, 7 once
        let params = SamplingParams {
            frequency_penalty: 1.0,
            ..SamplingParams::default()
        };
        apply_penalties(&mut logits, &output, &params);
        assert_eq!(logits[3], 5.0 - 2.0, "penalty must scale with count");
        assert_eq!(logits[7], -1.0, "seen once: one penalty unit");
        assert_eq!(logits[4], 0.0, "unseen token untouched");
    }

    #[test]
    fn presence_penalty_applies_once_per_token() {
        let mut logits = vec![0.0f32; 8];
        logits[3] = 5.0;
        logits[7] = 4.0;
        let output = vec![3u32, 3, 7];
        let params = SamplingParams {
            presence_penalty: 1.0,
            ..SamplingParams::default()
        };
        apply_penalties(&mut logits, &output, &params);
        assert_eq!(logits[3], 4.0, "once per token, not per occurrence");
        assert_eq!(logits[7], 3.0);
        assert_eq!(logits[4], 0.0);
    }
}













