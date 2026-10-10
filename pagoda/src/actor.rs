// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Client handle for the scheduler actor ([crate::engine::Engine::into_actor]).
//!
//! The actor owns the engine on a background thread and runs one long-lived
//! continuous-batching loop, so concurrent HTTP requests interleave (one
//! request's chunked prefill shares scheduler steps with another's decode)
//! instead of serializing behind a process-wide mutex. This module carries
//! the channel protocol; the loop itself lives in engine.rs.

use std::sync::mpsc;

use crate::engine::EngineStats;
use crate::spec::{GenerationOutput, WriteRequest};

/// One event on a generation stream: a decoded piece per sampled token, then
/// exactly one terminal Done.
pub enum StreamEvent {
    Token(String),
    Done(GenerationOutput),
}

/// Messages the actor understands.
pub enum ActorMsg {
    /// Admit a request into the continuous-batching loop; events (tokens,
    /// then Done) arrive on events.
    Generate {
        req: WriteRequest,
        events: mpsc::Sender<StreamEvent>,
    },
    /// Snapshot of engine counters.
    Stats { reply: mpsc::Sender<EngineStats> },
    /// Create a pinned KV checkpoint from `text`; reply carries the id.
    CheckpointCreate {
        text: String,
        reply: mpsc::Sender<u64>,
    },
    /// Release a pinned checkpoint; reply carries whether it existed.
    CheckpointDrop {
        id: u64,
        reply: mpsc::Sender<bool>,
    },
    /// Branch generation off a pinned checkpoint into the batching loop.
    /// `admitted` receives false when the checkpoint id is unknown (the HTTP
    /// layer maps that to 404 before writing any response head); otherwise
    /// events stream on `events` exactly like Generate.
    CheckpointGenerate {
        id: u64,
        continuation: String,
        sampling: crate::spec::SamplingParams,
        events: mpsc::Sender<StreamEvent>,
        admitted: mpsc::Sender<bool>,
    },
    /// Stop the actor loop (in-flight work is abandoned).
    Shutdown,
}

/// Cloneable handle to the scheduler actor; one per engine.
#[derive(Clone)]
pub struct EngineHandle {
    tx: mpsc::Sender<ActorMsg>,
}

impl EngineHandle {
    pub(crate) fn new(tx: mpsc::Sender<ActorMsg>) -> Self {
        Self { tx }
    }

    /// Buffered generation: subscribe, drain token events, return the
    /// terminal output.
    pub fn generate(&self, req: WriteRequest) -> GenerationOutput {
        let rx = self.subscribe(req);
        let mut out = None;
        for event in rx {
            if let StreamEvent::Done(o) = event {
                out = Some(o);
            }
        }
        out.expect("actor delivers Done before closing")
    }

    /// Streaming generation: token events arrive live; the terminal
    /// StreamEvent::Done carries the full output. The channel closes after
    /// Done.
    pub fn subscribe(&self, req: WriteRequest) -> mpsc::Receiver<StreamEvent> {
        let (tx, rx) = mpsc::channel();
        // If the actor is gone the receiver just closes; callers treat a
        // closed channel without Done as a fault.
        let _ = self.tx.send(ActorMsg::Generate { req, events: tx });
        rx
    }

    /// Engine counters snapshot.
    pub fn stats(&self) -> EngineStats {
        let (tx, rx) = mpsc::channel();
        let _ = self.tx.send(ActorMsg::Stats { reply: tx });
        rx.recv().expect("actor alive")
    }

    /// Ask the actor to stop. In-flight requests are abandoned; later
    /// submissions find the channel closed and their streams end without a
    /// Done.
    pub fn shutdown(&self) {
        let _ = self.tx.send(ActorMsg::Shutdown);
    }

    /// Create a pinned KV checkpoint. None when the actor is gone.
    pub fn checkpoint_create(&self, text: String) -> Option<u64> {
        let (tx, rx) = mpsc::channel();
        self.tx.send(ActorMsg::CheckpointCreate { text, reply: tx }).ok()?;
        rx.recv().ok()
    }

    /// Release a pinned checkpoint; false when it did not exist.
    pub fn checkpoint_drop(&self, id: u64) -> bool {
        let (tx, rx) = mpsc::channel();
        let _ = self.tx.send(ActorMsg::CheckpointDrop { id, reply: tx });
        rx.recv().unwrap_or(false)
    }

    /// Branch generation off a pinned checkpoint, streaming events like
    /// [Self::subscribe]. None when the checkpoint id is unknown.
    pub fn checkpoint_generate(
        &self,
        id: u64,
        continuation: String,
        sampling: crate::spec::SamplingParams,
    ) -> Option<mpsc::Receiver<StreamEvent>> {
        let (events_tx, events_rx) = mpsc::channel();
        let (admitted_tx, admitted_rx) = mpsc::channel();
        self.tx
            .send(ActorMsg::CheckpointGenerate {
                id,
                continuation,
                sampling,
                events: events_tx,
                admitted: admitted_tx,
            })
            .ok()?;
        match admitted_rx.recv() {
            Ok(true) => Some(events_rx),
            _ => None,
        }
    }
}