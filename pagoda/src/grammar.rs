// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Dependency-free constrained decoding.
//!
//! The reference tokenizer is byte-level (one token equals one byte), so a
//! "guided generation" constraint can be enforced as a finite-state byte filter:
//! at each step the engine decodes the output produced so far, asks the grammar
//! which bytes may legally follow, and masks every disallowed logit to `-inf`.
//! This mirrors the guided-generation family of techniques (Outlines / guidance)
//! without any external dependency.
//!
//! Two constraints ship out of the box:
//!
//! * [`Grammar::Regex`] — a full-match (`^...$`) byte regular expression.
//! * [`Grammar::Json`]  — a complete, syntactically valid JSON value.

use crate::tokenizer::{BYTE_VOCAB_SIZE, FIRST_BYTE_ID};

// ---------------------------------------------------------------------------
// Public grammar type
// ---------------------------------------------------------------------------

/// A compiled output constraint.
#[derive(Clone, Debug, PartialEq)]
pub enum Grammar {
    /// Full-match byte regular expression (implicit `^...$` anchors).
    Regex(ByteRegex),
    /// A complete JSON value.
    Json,
}

impl Grammar {
    /// Compile a byte regular expression pattern.
    pub fn regex(pattern: &str) -> Result<Grammar, String> {
        ByteRegex::new(pattern).map(Grammar::Regex)
    }

    /// Constrain output to a syntactically valid JSON value.
    pub fn json() -> Grammar {
        Grammar::Json
    }

    pub fn name(&self) -> &'static str {
        match self {
            Grammar::Regex(_) => "regex",
            Grammar::Json => "json",
        }
    }

    /// Bytes that may legally follow `partial`, or `None` when the string is
    /// already dead (cannot be completed to a match).
    pub fn allowed_bytes(&self, partial: &str) -> Option<Vec<u8>> {
        match self {
            Grammar::Regex(re) => re.allowed_bytes(partial),
            Grammar::Json => json::allowed(partial),
        }
    }

    /// Whether `partial` is a complete, acceptable result.
    pub fn is_complete(&self, partial: &str) -> bool {
        match self {
            Grammar::Regex(re) => re.is_complete(partial),
            Grammar::Json => json::complete(partial),
        }
    }
}

/// Mask every disallowed logit to `-inf`: the special-token prefix and any byte
/// token not present in `allowed`. Logits of allowed byte tokens keep their
/// original model scores, so the model's own distribution still ranks the
/// legal continuations (greedy picks the model's favourite legal byte, not
/// the lowest byte value).
pub fn mask_logits(allowed: &[u8], logits: &mut [f32]) {
    let mut keep = [false; BYTE_VOCAB_SIZE];
    for &b in allowed {
        keep[b as usize] = true;
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        let byte_idx = i.wrapping_sub(FIRST_BYTE_ID as usize);
        let is_allowed_byte = byte_idx < BYTE_VOCAB_SIZE && keep[byte_idx];
        if !is_allowed_byte {
            *logit = f32::NEG_INFINITY;
        }
    }
}

// ---------------------------------------------------------------------------
// Byte regex (Thompson NFA + guidance queries)
// ---------------------------------------------------------------------------

/// A compiled full-match byte regex.
#[derive(Clone, Debug, PartialEq)]
pub struct ByteRegex {
    nfa: Nfa,
}

#[derive(Clone, Debug, PartialEq)]
struct Nfa {
    edges: Vec<Vec<Edge>>,
    accept: Vec<bool>,
    start: usize,
}

#[derive(Clone, Debug, PartialEq)]
enum Edge {
    Char(u8, usize),
    Eps(usize),
}

struct Builder {
    edges: Vec<Vec<Edge>>,
    accept: Vec<bool>,
}

impl Builder {
    fn new() -> Self {
        Builder { edges: Vec::new(), accept: Vec::new() }
    }

    fn fresh(&mut self) -> usize {
        self.edges.push(Vec::new());
        self.accept.push(false);
        self.edges.len() - 1
    }

    fn eps(&mut self, from: usize, to: usize) {
        self.edges[from].push(Edge::Eps(to));
    }

    fn chr(&mut self, from: usize, to: usize, c: u8) {
        self.edges[from].push(Edge::Char(c, to));
    }

    fn frag_char(&mut self, c: u8) -> (usize, usize) {
        let s = self.fresh();
        let e = self.fresh();
        self.chr(s, e, c);
        (s, e)
    }

    fn frag_set(&mut self, set: &[bool; 256]) -> (usize, usize) {
        let s = self.fresh();
        let e = self.fresh();
        for c in 0..=255u8 {
            if set[c as usize] {
                self.chr(s, e, c);
            }
        }
        (s, e)
    }

    fn frag_dot(&mut self) -> (usize, usize) {
        let s = self.fresh();
        let e = self.fresh();
        for c in 0..=255u8 {
            self.chr(s, e, c);
        }
        (s, e)
    }

    fn concat(&mut self, a: (usize, usize), b: (usize, usize)) -> (usize, usize) {
        self.eps(a.1, b.0);
        (a.0, b.1)
    }

    fn alt(&mut self, a: (usize, usize), b: (usize, usize)) -> (usize, usize) {
        let s = self.fresh();
        let e = self.fresh();
        self.eps(s, a.0);
        self.eps(s, b.0);
        self.eps(a.1, e);
        self.eps(b.1, e);
        (s, e)
    }

    fn star(&mut self, a: (usize, usize)) -> (usize, usize) {
        let s = self.fresh();
        let e = self.fresh();
        self.eps(s, a.0);
        self.eps(s, e);
        self.eps(a.1, a.0);
        self.eps(a.1, e);
        (s, e)
    }

    fn plus(&mut self, a: (usize, usize)) -> (usize, usize) {
        let s = self.fresh();
        let e = self.fresh();
        self.eps(s, a.0);
        self.eps(a.1, a.0);
        self.eps(a.1, e);
        (s, e)
    }

    fn opt(&mut self, a: (usize, usize)) -> (usize, usize) {
        let s = self.fresh();
        let e = self.fresh();
        self.eps(s, a.0);
        self.eps(s, e);
        self.eps(a.1, e);
        (s, e)
    }

    /// Deep-copy the sub-NFA between `range.0..=range.1` into fresh states and
    /// return the new (start, end) state ids.
    fn copy_fragment(&mut self, range: (usize, usize)) -> (usize, usize) {
        let (lo, hi) = range;
        let span = hi - lo + 1;
        let mut sources: Vec<Vec<Edge>> = Vec::with_capacity(span);
        let mut accepts = Vec::with_capacity(span);
        for node in lo..=hi {
            sources.push(self.edges[node].clone());
            accepts.push(self.accept[node]);
        }
        let offset = self.edges.len();
        for _ in 0..span {
            self.edges.push(Vec::new());
            self.accept.push(false);
        }
        for (node, edges) in (lo..=hi).zip(sources) {
            let remapped: Vec<Edge> = edges
                .into_iter()
                .filter(|e| matches!(e, Edge::Eps(t) | Edge::Char(_, t) if (lo..=hi).contains(t)))
                .map(|e| match e {
                    Edge::Eps(t) => Edge::Eps(t - lo + offset),
                    Edge::Char(c, t) => Edge::Char(c, t - lo + offset),
                })
                .collect();
            self.edges[offset + node - lo] = remapped;
            self.accept[offset + node - lo] = accepts[node - lo];
        }
        (offset, offset + span - 1)
    }
}

struct Parser {
    bytes: Vec<u8>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek1(&self) -> Option<u8> {
        self.bytes.get(self.pos + 1).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn parse_alt(&mut self, b: &mut Builder) -> Result<(usize, usize), String> {
        let mut left = self.parse_concat(b)?;
        while self.peek() == Some(b'|') {
            self.pos += 1;
            let right = self.parse_concat(b)?;
            left = b.alt(left, right);
        }
        Ok(left)
    }

    fn parse_concat(&mut self, b: &mut Builder) -> Result<(usize, usize), String> {
        let mut left = self.parse_repeat(b)?;
        loop {
            match self.peek() {
                None | Some(b')') | Some(b'|') => break,
                _ => {
                    let right = self.parse_repeat(b)?;
                    left = b.concat(left, right);
                }
            }
        }
        Ok(left)
    }

    fn parse_repeat(&mut self, b: &mut Builder) -> Result<(usize, usize), String> {
        let mut frag = self.parse_atom(b)?;
        loop {
            match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    frag = b.star(frag);
                }
                Some(b'+') => {
                    self.pos += 1;
                    frag = b.plus(frag);
                }
                Some(b'?') => {
                    self.pos += 1;
                    frag = b.opt(frag);
                }
                Some(b'{') => {
                    self.pos += 1;
                    let lo_str = self.take_while(|c| c.is_ascii_digit());
                    if lo_str.is_empty() {
                        return Err("missing repetition count".to_string());
                    }
                    let lo: usize = lo_str.parse().map_err(|_| "bad repetition count".to_string())?;
                    let hi = if self.peek() == Some(b',') {
                        self.pos += 1;
                        let hi_str = self.take_while(|c| c.is_ascii_digit());
                        if hi_str.is_empty() {
                            return Err("missing max repetition count".to_string());
                        }
                        hi_str.parse().map_err(|_| "bad max repetition count".to_string())?
                    } else {
                        lo
                    };
                    if self.peek() != Some(b'}') {
                        return Err("unclosed repetition".to_string());
                    }
                    self.pos += 1;
                    if hi < lo {
                        return Err("descending repetition bounds".to_string());
                    }
                    frag = Self::parse_exact(b, frag, lo)?;
                }
                _ => break,
            }
        }
        Ok(frag)
    }

    fn take_while<F: Fn(u8) -> bool>(&mut self, pred: F) -> String {
        let mut out = Vec::new();
        while let Some(c) = self.peek() {
            if pred(c) {
                out.push(c);
                self.pos += 1;
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    fn parse_exact(b: &mut Builder, frag: (usize, usize), n: usize) -> Result<(usize, usize), String> {
        let mut acc = (frag.0, frag.1);
        for _ in 1..n {
            let copy = b.copy_fragment(frag);
            acc = b.concat(acc, copy);
        }
        Ok(acc)
    }

    fn parse_atom(&mut self, b: &mut Builder) -> Result<(usize, usize), String> {
        match self.peek() {
            None => Err("unexpected end of pattern".to_string()),
            Some(b'(') => {
                self.pos += 1;
                let frag = self.parse_alt(b)?;
                if self.bump() == Some(b')') {
                    Ok(frag)
                } else {
                    Err("unclosed group".to_string())
                }
            }
            Some(b'[') => {
                self.pos += 1;
                self.parse_class(b)
            }
            Some(b'.') => {
                self.pos += 1;
                Ok(b.frag_dot())
            }
            Some(b'^') | Some(b'$') => {
                self.pos += 1;
                self.parse_atom(b)
            }
            Some(b'\\') => {
                self.pos += 1;
                match self.bump() {
                    None => Err("dangling escape".to_string()),
                    Some(c) => Ok(b.frag_char(unescape(c))),
                }
            }
            Some(c) => {
                self.pos += 1;
                Ok(b.frag_char(c))
            }
        }
    }

    fn parse_class(&mut self, b: &mut Builder) -> Result<(usize, usize), String> {
        let mut set = [false; 256];
        let mut negated = false;
        if self.peek() == Some(b'^') {
            negated = true;
            self.pos += 1;
        }
        let mut first = true;
        loop {
            match self.peek() {
                None => return Err("unclosed character class".to_string()),
                Some(b']') if !first => {
                    self.pos += 1;
                    break;
                }
                Some(c) => {
                    first = false;
                    self.pos += 1;
                    let lo = if c == b'\\' {
                        match self.bump() {
                            None => return Err("bad escape in class".to_string()),
                            Some(e) => unescape(e),
                        }
                    } else {
                        c
                    };
                    set[lo as usize] = true;

                    if self.peek() == Some(b'-') && self.peek1().map_or(false, |n| n != b']') {
                        self.pos += 1; // consume '-'
                        let hi_raw = self.bump().ok_or_else(|| "bad range".to_string())?;
                        let hi = if hi_raw == b'\\' {
                            match self.bump() {
                                None => return Err("bad escape in range".to_string()),
                                Some(e) => unescape(e),
                            }
                        } else {
                            hi_raw
                        };
                        if hi < lo {
                            return Err("character range is descending".to_string());
                        }
                        for v in lo..=hi {
                            set[v as usize] = true;
                        }
                    }
                }
            }
        }
        if negated {
            for v in set.iter_mut() {
                *v = !*v;
            }
        }
        Ok(b.frag_set(&set))
    }
}

fn unescape(c: u8) -> u8 {
    match c {
        b'n' => b'\n',
        b't' => b'\t',
        b'r' => b'\r',
        b'0' => 0,
        other => other,
    }
}

impl ByteRegex {
    pub fn new(pattern: &str) -> Result<ByteRegex, String> {
        let mut parser = Parser {
            bytes: pattern.as_bytes().to_vec(),
            pos: 0,
        };
        let mut builder = Builder::new();
        let (start, accept) = parser.parse_alt(&mut builder)?;
        if parser.pos != parser.bytes.len() {
            return Err(format!(
                "unexpected byte {:?} at position {}",
                parser.bytes[parser.pos] as char, parser.pos
            ));
        }
        builder.accept[accept] = true;
        Ok(ByteRegex {
            nfa: Nfa {
                edges: builder.edges,
                accept: builder.accept,
                start,
            },
        })
    }

    fn eps_closure(&self, seeds: &[usize]) -> Vec<usize> {
        let n = self.nfa.edges.len();
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        let mut stack: Vec<usize> = seeds.to_vec();
        while let Some(s) = stack.pop() {
            if seen[s] {
                continue;
            }
            seen[s] = true;
            out.push(s);
            for e in &self.nfa.edges[s] {
                if let Edge::Eps(t) = e {
                    stack.push(*t);
                }
            }
        }
        out
    }

    fn advance_states(&self, states: &[usize], byte: u8) -> Vec<usize> {
        let mut next = Vec::new();
        for &s in states {
            for e in &self.nfa.edges[s] {
                if let Edge::Char(c, t) = e {
                    if *c == byte {
                        next.push(*t);
                    }
                }
            }
        }
        next
    }

    fn run(&self, text: &str) -> Vec<usize> {
        let mut cur = self.eps_closure(&[self.nfa.start]);
        for &byte in text.as_bytes() {
            let next = self.advance_states(&cur, byte);
            cur = self.eps_closure(&next);
            if cur.is_empty() {
                return cur;
            }
        }
        cur
    }

    /// Reverse reachability from any accept state.
    fn can_reach_accept(&self) -> Vec<bool> {
        let n = self.nfa.edges.len();
        let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
        for s in 0..n {
            for e in &self.nfa.edges[s] {
                match e {
                    Edge::Char(_, t) => rev[*t].push(s),
                    Edge::Eps(t) => rev[*t].push(s),
                }
            }
        }
        let mut reach = vec![false; n];
        let mut stack: Vec<usize> = (0..n).filter(|&s| self.nfa.accept[s]).collect();
        while let Some(s) = stack.pop() {
            if reach[s] {
                continue;
            }
            reach[s] = true;
            for &p in &rev[s] {
                stack.push(p);
            }
        }
        reach
    }

    pub fn is_complete(&self, partial: &str) -> bool {
        self.run(partial).iter().any(|&s| self.nfa.accept[s])
    }

    pub fn allowed_bytes(&self, partial: &str) -> Option<Vec<u8>> {
        let cur = self.run(partial);
        if cur.is_empty() {
            return None;
        }
        let reach = self.can_reach_accept();
        if !cur.iter().any(|&s| reach[s]) {
            return None;
        }
        let mut allowed = Vec::new();
        for byte in 0..=255u8 {
            let next = self.eps_closure(&self.advance_states(&cur, byte));
            if next.iter().any(|&s| reach[s]) {
                allowed.push(byte);
            }
        }
        Some(allowed)
    }
}

// ---------------------------------------------------------------------------
// JSON value guidance (pushdown prefix scanner)
// ---------------------------------------------------------------------------

mod json {
    #[derive(Clone, Copy)]
    enum Frame {
        Value,
        Object,
        Colon,
        AfterMember,
        Array,
        ArrayComma,
    }

    #[derive(Clone, Copy)]
    enum Scan {
        Complete,
        Prefix([bool; 256]),
        Dead,
    }

    #[derive(Clone, Copy)]
    enum Step {
        Continue,
        Dead,
        Eof([bool; 256]),
    }

    struct Scanner<'a> {
        b: &'a [u8],
        i: usize,
        stack: Vec<Frame>,
    }

    pub(super) fn allowed(partial: &str) -> Option<Vec<u8>> {
        match scan(partial.as_bytes()) {
            Scan::Complete => Some(Vec::new()),
            Scan::Prefix(mask) => Some(mask_to_vec(mask)),
            Scan::Dead => None,
        }
    }

    pub(super) fn complete(partial: &str) -> bool {
        matches!(scan(partial.as_bytes()), Scan::Complete)
    }

    fn scan(b: &[u8]) -> Scan {
        let mut st = Scanner {
            b,
            i: 0,
            stack: vec![Frame::Value],
        };
        loop {
            let Some(top) = st.stack.last().copied() else {
                st.skip_ws();
                return if st.i == st.b.len() { Scan::Complete } else { Scan::Dead };
            };
            st.skip_ws();
            if st.i == st.b.len() {
                return Scan::Prefix(prefix_allowed(top));
            }
            let c = st.b[st.i];
            match top {
                Frame::Value => match c {
                    b'{' => {
                        st.i += 1;
                        st.stack.push(Frame::Object);
                    }
                    b'[' => {
                        st.i += 1;
                        st.stack.push(Frame::Array);
                    }
                    b'"' => {
                        st.i += 1;
                        match st.scan_string() {
                            Step::Continue => st.finish_value(),
                            other => return to_scan(other),
                        }
                    }
                    b'-' | b'0'..=b'9' => match st.scan_number() {
                        Step::Continue => st.finish_value(),
                        other => return to_scan(other),
                    },
                    b't' => match st.match_literal(b"true") {
                        Step::Continue => st.finish_value(),
                        other => return to_scan(other),
                    },
                    b'f' => match st.match_literal(b"false") {
                        Step::Continue => st.finish_value(),
                        other => return to_scan(other),
                    },
                    b'n' => match st.match_literal(b"null") {
                        Step::Continue => st.finish_value(),
                        other => return to_scan(other),
                    },
                    _ => return Scan::Dead,
                },
                Frame::Object => match c {
                    b'"' => {
                        st.i += 1;
                        match st.scan_string() {
                            Step::Continue => st.stack.push(Frame::Colon),
                            other => return to_scan(other),
                        }
                    }
                    b'}' => {
                        st.i += 1;
                        st.finish_container();
                    }
                    _ => return Scan::Dead,
                },
                Frame::Colon => {
                    if c == b':' {
                        st.i += 1;
                        st.stack.pop();
                        st.stack.push(Frame::Value);
                    } else {
                        return Scan::Dead;
                    }
                }
                Frame::AfterMember => match c {
                    b',' => {
                        st.i += 1;
                        st.stack.pop();
                        st.stack.push(Frame::Object);
                    }
                    b'}' => {
                        st.i += 1;
                        st.finish_container();
                    }
                    _ => return Scan::Dead,
                },
                Frame::Array => {
                    if c == b']' {
                        st.i += 1;
                        st.finish_container();
                    } else {
                        st.stack.push(Frame::Value);
                    }
                }
                Frame::ArrayComma => match c {
                    b',' => {
                        st.i += 1;
                        st.stack.pop();
                        st.stack.push(Frame::Array);
                    }
                    b']' => {
                        st.i += 1;
                        st.finish_container();
                    }
                    _ => return Scan::Dead,
                },
            }
        }
    }

    fn to_scan(step: Step) -> Scan {
        match step {
            Step::Continue => Scan::Prefix(empty_mask()),
            Step::Dead => Scan::Dead,
            Step::Eof(mask) => Scan::Prefix(mask),
        }
    }

    impl Scanner<'_> {
        fn skip_ws(&mut self) {
            while self.i < self.b.len() && is_ws(self.b[self.i]) {
                self.i += 1;
            }
        }

        fn finish_value(&mut self) {
            self.stack.pop();
            match self.stack.last_mut() {
                Some(frame @ Frame::Object) => *frame = Frame::AfterMember,
                Some(frame @ Frame::Array) => *frame = Frame::ArrayComma,
                _ => {}
            }
        }

        fn finish_container(&mut self) {
            self.stack.pop();
            self.finish_value();
        }

        fn scan_string(&mut self) -> Step {
            loop {
                if self.i == self.b.len() {
                    return Step::Eof(string_continuation());
                }
                let c = self.b[self.i];
                match c {
                    b'"' => {
                        self.i += 1;
                        return Step::Continue;
                    }
                    b'\\' => {
                        self.i += 1;
                        if self.i == self.b.len() {
                            return Step::Eof(escape_continuation());
                        }
                        let e = self.b[self.i];
                        self.i += 1;
                        if e == b'u' {
                            for _ in 0..4 {
                                if self.i == self.b.len() {
                                    return Step::Eof(hex_continuation());
                                }
                                if !self.b[self.i].is_ascii_hexdigit() {
                                    return Step::Dead;
                                }
                                self.i += 1;
                            }
                        } else if !matches!(e, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') {
                            return Step::Dead;
                        }
                    }
                    c if c < 0x20 => return Step::Dead,
                    _ => self.i += 1,
                }
            }
        }

        fn scan_number(&mut self) -> Step {
            if self.b[self.i] == b'-' {
                self.i += 1;
            }
            if self.i == self.b.len() {
                return Step::Eof(digits_mask());
            }
            let phase = match self.b[self.i] {
                b'0' => {
                    self.i += 1;
                    NumberPhase::Zero
                }
                b'1'..=b'9' => {
                    self.i += 1;
                    NumberPhase::Int
                }
                _ => return Step::Dead,
            };
            self.scan_number_tail(phase)
        }

        fn scan_number_tail(&mut self, mut phase: NumberPhase) -> Step {
            loop {
                if self.i == self.b.len() {
                    return match phase {
                        NumberPhase::Zero | NumberPhase::Int
                        | NumberPhase::FracDigits | NumberPhase::ExpDigits => Step::Continue,
                        _ => Step::Eof(number_continuation(phase)),
                    };
                }
                let c = self.b[self.i];
                match phase {
                    NumberPhase::Zero => match c {
                        b'.' => {
                            self.i += 1;
                            phase = NumberPhase::FracDot;
                        }
                        b'e' | b'E' => {
                            self.i += 1;
                            phase = NumberPhase::ExpE;
                        }
                        _ => return Step::Continue,
                    },
                    NumberPhase::Int => match c {
                        b'0'..=b'9' => self.i += 1,
                        b'.' => {
                            self.i += 1;
                            phase = NumberPhase::FracDot;
                        }
                        b'e' | b'E' => {
                            self.i += 1;
                            phase = NumberPhase::ExpE;
                        }
                        _ => return Step::Continue,
                    },
                    NumberPhase::FracDot => {
                        if c.is_ascii_digit() {
                            self.i += 1;
                            phase = NumberPhase::FracDigits;
                        } else {
                            return Step::Dead;
                        }
                    }
                    NumberPhase::FracDigits => match c {
                        b'0'..=b'9' => self.i += 1,
                        b'e' | b'E' => {
                            self.i += 1;
                            phase = NumberPhase::ExpE;
                        }
                        _ => return Step::Continue,
                    },
                    NumberPhase::ExpE => {
                        if c == b'+' || c == b'-' {
                            self.i += 1;
                            phase = NumberPhase::ExpSign;
                        } else if c.is_ascii_digit() {
                            self.i += 1;
                            phase = NumberPhase::ExpDigits;
                        } else {
                            return Step::Dead;
                        }
                    }
                    NumberPhase::ExpSign => {
                        if c.is_ascii_digit() {
                            self.i += 1;
                            phase = NumberPhase::ExpDigits;
                        } else {
                            return Step::Dead;
                        }
                    }
                    NumberPhase::ExpDigits => {
                        if c.is_ascii_digit() {
                            self.i += 1;
                        } else {
                            return Step::Continue;
                        }
                    }
                }
            }
        }

        fn match_literal(&mut self, lit: &[u8]) -> Step {
            for &want in lit {
                if self.i == self.b.len() {
                    return Step::Eof(single_mask(want));
                }
                if self.b[self.i] != want {
                    return Step::Dead;
                }
                self.i += 1;
            }
            Step::Continue
        }
    }

    #[derive(Clone, Copy)]
    enum NumberPhase {
        Zero,
        Int,
        FracDot,
        FracDigits,
        ExpE,
        ExpSign,
        ExpDigits,
    }

    fn is_ws(c: u8) -> bool {
        matches!(c, b' ' | b'\t' | b'\n' | b'\r')
    }

    fn empty_mask() -> [bool; 256] {
        [false; 256]
    }

    fn mask_to_vec(mask: [bool; 256]) -> Vec<u8> {
        (0..=255u8).filter(|&b| mask[b as usize]).collect()
    }

    fn single_mask(byte: u8) -> [bool; 256] {
        let mut m = [false; 256];
        m[byte as usize] = true;
        m
    }

    fn bytes_mask(bytes: &[u8]) -> [bool; 256] {
        let mut m = [false; 256];
        for &b in bytes {
            m[b as usize] = true;
        }
        m
    }

    fn with_ws(mask: &mut [bool; 256]) -> [bool; 256] {
        let mut m = *mask;
        for &b in b" \t\n\r" {
            m[b as usize] = true;
        }
        m
    }

    fn digits_mask() -> [bool; 256] {
        bytes_mask(b"0123456789")
    }

    fn value_starts() -> [bool; 256] {
        let mut m = bytes_mask(b"{\"tf n0123456789");
        m[b'[' as usize] = true;
        m[b'"' as usize] = true;
        m[b'-' as usize] = true;
        m
    }

    fn prefix_allowed(top: Frame) -> [bool; 256] {
        match top {
            Frame::Value => with_ws(&mut value_starts()),
            Frame::Object => with_ws(&mut bytes_mask(b"\"}")),
            Frame::Colon => with_ws(&mut bytes_mask(b":")),
            Frame::AfterMember => with_ws(&mut bytes_mask(b",}")),
            Frame::Array => {
                let mut m = value_starts();
                m[b']' as usize] = true;
                with_ws(&mut m)
            }
            Frame::ArrayComma => with_ws(&mut bytes_mask(b",]")),
        }
    }

    fn string_continuation() -> [bool; 256] {
        let mut m = [false; 256];
        for c in 0x20..=0xFFu8 {
            m[c as usize] = true;
        }
        m
    }

    fn escape_continuation() -> [bool; 256] {
        bytes_mask(b"\"\\/bfnrtu")
    }

    fn hex_continuation() -> [bool; 256] {
        bytes_mask(b"0123456789abcdefABCDEF")
    }

    fn number_continuation(phase: NumberPhase) -> [bool; 256] {
        match phase {
            NumberPhase::Zero => bytes_mask(b".eE"),
            NumberPhase::Int => bytes_mask(b"0123456789.eE"),
            NumberPhase::FracDot => digits_mask(),
            NumberPhase::FracDigits => bytes_mask(b"0123456789eE"),
            NumberPhase::ExpE => bytes_mask(b"0123456789+-"),
            NumberPhase::ExpSign => digits_mask(),
            NumberPhase::ExpDigits => digits_mask(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{mask_logits, ByteRegex, Grammar};
    use crate::tokenizer::{BYTE_VOCAB_SIZE, FIRST_BYTE_ID};

    #[test]
    fn regex_matches_across_full_string() {
        let g = ByteRegex::new("(ab)+c").unwrap();
        assert!(g.is_complete("abc"));
        assert!(g.is_complete("ababc"));
        assert!(!g.is_complete("ab"));
        assert!(!g.is_complete("abcd"));
    }

    #[test]
    fn regex_character_class_and_digits() {
        let g = ByteRegex::new("[a-z][0-9]{2}").unwrap();
        assert!(g.is_complete("x42"));
        assert!(!g.is_complete("x4"));
        assert!(!g.is_complete("X42"));
    }

    #[test]
    fn regex_allowed_bytes_are_prefix_legal() {
        let g = ByteRegex::new("[abc]+").unwrap();
        let allowed = g.allowed_bytes("cb").unwrap();
        assert_eq!(allowed, vec![b'a', b'b', b'c']);
        // A byte outside the class is already dead.
        assert!(g.allowed_bytes("z").is_none());
    }

    #[test]
    fn regex_compilation_rejects_dangling_and_mismatched() {
        assert!(ByteRegex::new("(").is_err());
        assert!(ByteRegex::new("[a").is_err());
        assert!(ByteRegex::new("a|").is_err());
    }

    #[test]
    fn json_complete_for_scalars_and_containers() {
        for text in [
            "null", "true", "false", "42", "-3.5e2", "\"hi\"", "{}", "[]",
            "{\"a\":1,\"b\":[true,null]}",
        ] {
            assert!(Grammar::Json.is_complete(text), "should be complete: {text}");
        }
        assert!(!Grammar::json().is_complete(""));
        assert!(!Grammar::json().is_complete("{"));
    }

    #[test]
    fn json_prefix_allows_continuation() {
        let allowed = Grammar::json().allowed_bytes("").unwrap();
        assert!(allowed.contains(&b'{'));
        assert!(allowed.contains(&b'['));
        assert!(allowed.contains(&b'"'));
        assert!(allowed.contains(&b't'));

        // Inside an object the next legal byte is a key quote or a close brace.
        let allowed = Grammar::json().allowed_bytes("{").unwrap();
        assert!(allowed.contains(&b'"'));
        assert!(allowed.contains(&b'}'));
    }

    #[test]
    fn json_dead_for_trailing_content() {
        assert!(Grammar::json().allowed_bytes("{\"a\":1}  ]").is_none());
    }

    #[test]
    fn mask_logits_keeps_allowed_and_zeros_specials() {
        let mut logits = vec![3.0f32; FIRST_BYTE_ID as usize + BYTE_VOCAB_SIZE];
        mask_logits(&[b'a'], &mut logits);
        assert!(!logits[0].is_finite());
        assert!(!logits[1].is_finite());
        assert!(!logits[2].is_finite());
        // b'a' == 97, so its token id is FIRST_BYTE_ID + 97; the mask must
        // preserve the model's original score for allowed bytes.
        assert_eq!(logits[FIRST_BYTE_ID as usize + b'a' as usize], 3.0);
        assert!(!logits[FIRST_BYTE_ID as usize + b'b' as usize].is_finite());
    }
}

