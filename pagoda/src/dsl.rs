// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! An SGLang-style frontend language, expressed as an embeddable builder API.
//!
//! The real SGLang lets you write programs like `sgl.gen(...)`, `sgl.select(...)`
//! and `sgl.fork(...)` and then compiles them into a schedule. Rust has no
//! equivalent dynamic evaluation sugar, so this crate models the *program* with
//! a small operation algebra that [`crate::engine::Engine::run_program`]
//! interprets. It keeps the same semantics: sequential roles and generations,
//! scored choice selection, and forking into concurrent branches.

use std::collections::HashMap;

use crate::spec::SamplingParams;

/// One frontend operation.
#[derive(Clone, Debug)]
pub enum Op {
    /// Add a system-turn message.
    System(String),
    /// Add a user-turn message.
    User(String),
    /// Add an assistant-turn message (a static prefix).
    Assistant(String),
    /// Generate text and bind it to `name`.
    Gen {
        name: String,
        params: SamplingParams,
    },
    /// Score `choices` and bind the best-scoring one to `name`.
    Select {
        name: String,
        choices: Vec<String>,
        params: SamplingParams,
    },
    /// Run each branch concurrently, joining their results.
    Fork { branches: Vec<Program> },
}

/// A buildable frontend program.
#[derive(Clone, Debug, Default)]
pub struct Program {
    pub ops: Vec<Op>,
}

impl Program {
    pub fn new() -> Self {
        Program { ops: Vec::new() }
    }

    pub fn system(&mut self, text: impl Into<String>) -> &mut Self {
        self.ops.push(Op::System(text.into()));
        self
    }

    pub fn user(&mut self, text: impl Into<String>) -> &mut Self {
        self.ops.push(Op::User(text.into()));
        self
    }

    pub fn assistant(&mut self, text: impl Into<String>) -> &mut Self {
        self.ops.push(Op::Assistant(text.into()));
        self
    }

    pub fn gen(&mut self, name: impl Into<String>, params: SamplingParams) -> &mut Self {
        self.ops.push(Op::Gen {
            name: name.into(),
            params,
        });
        self
    }

    pub fn select(
        &mut self,
        name: impl Into<String>,
        choices: Vec<String>,
        params: SamplingParams,
    ) -> &mut Self {
        self.ops.push(Op::Select {
            name: name.into(),
            choices,
            params,
        });
        self
    }

    pub fn fork(&mut self, branches: Vec<Program>) -> &mut Self {
        self.ops.push(Op::Fork { branches });
        self
    }
}

/// The final state of one executed branch.
#[derive(Clone, Debug, Default)]
pub struct StreamResult {
    /// Variables bound by `gen` and `select`.
    pub vars: HashMap<String, String>,
    /// The full rendered transcript for this branch.
    pub transcript: String,
}

impl StreamResult {
    /// Looks up a bound variable by name.
    pub fn get(&self, name: &str) -> Option<&String> {
        self.vars.get(name)
    }
}
