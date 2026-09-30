// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! End-to-end constrained decoding: the grammar must be enforced through the
//! sampler mask so greedy decoding emits only legal continuations.

use pagoda::{FinishReason, Grammar, SamplingParams, ToyEngine, WriteRequest};

#[test]
fn regex_grammar_produces_parseable_string_value() {
    // A grammar that pins the output to one valid JSON string. Because the
    // tokenizer is byte-level, the regex is written directly in bytes.
    let mut e = ToyEngine::toy(Default::default());
    let g = Grammar::regex("\"[a-z]{3}\"").unwrap();
    let out = e.generate(&WriteRequest::new(
        "",
        SamplingParams {
            max_tokens: 8,
            grammar: Some(g.clone()),
            ..SamplingParams::default()
        },
    ));
    assert!(g.is_complete(&out.text), "not a full match: {:?}", out.text);
    assert_eq!(out.finish_reason, FinishReason::Stop);
    assert!(
        pagoda::json::parse(out.text.as_str()).is_ok(),
        "grammar output must still be valid JSON: {:?}",
        out.text
    );
}

#[test]
fn regex_grammar_terminates_on_dead_end() {
    let mut e = ToyEngine::toy(Default::default());
    let g = Grammar::regex("[a-z]{3}").unwrap();
    let out = e.generate(&WriteRequest::new(
        "",
        SamplingParams {
            max_tokens: 8,
            grammar: Some(g.clone()),
            ..SamplingParams::default()
        },
    ));
    assert!(g.is_complete(&out.text), "not a full match: {:?}", out.text);
    assert_eq!(out.finish_reason, FinishReason::Stop);
    for &b in out.text.as_bytes() {
        assert!(b.is_ascii_lowercase());
    }
}

#[test]
fn grammar_constraints_hold_under_temperature_sampling() {
    // Regression test for the masked-logit handling on the temperature path:
    // across many seeds, every emitted byte must keep the output on a legal
    // path of the grammar (or be a complete match).
    let mut e = ToyEngine::toy(Default::default());
    let g = Grammar::regex("\"[a-z]{3}\"").unwrap();
    for seed in 0..8 {
        let out = e.generate(&WriteRequest::new(
            "",
            SamplingParams {
                max_tokens: 8,
                temperature: 1.0,
                top_p: 1.0,
                top_k: usize::MAX,
                seed,
                grammar: Some(g.clone()),
                ..SamplingParams::default()
            },
        ));
        assert!(
            g.is_complete(&out.text) || g.allowed_bytes(&out.text).is_some(),
            "seed {seed}: output left the grammar: {:?}",
            out.text
        );
    }
}

#[test]
fn json_grammar_e2e_output_is_valid_json_or_completable_prefix() {
    // The real JSON grammar (not a regex stand-in): every emitted byte must
    // keep the output on a path to valid JSON. When the grammar reports the
    // value complete, it must parse as real JSON.
    let mut e = ToyEngine::toy(Default::default());
    let g = Grammar::json();
    let out = e.generate(&WriteRequest::new(
        "",
        SamplingParams {
            max_tokens: 48,
            grammar: Some(g.clone()),
            ..SamplingParams::default()
        },
    ));
    assert!(
        !out.text.is_empty(),
        "JSON grammar should admit at least one byte"
    );
    if g.is_complete(&out.text) {
        assert!(
            pagoda::json::parse(out.text.as_str()).is_ok(),
            "grammar-complete output must parse as JSON: {:?}",
            out.text
        );
    } else {
        assert!(
            g.allowed_bytes(&out.text).is_some(),
            "output must remain a completable JSON prefix: {:?}",
            out.text
        );
    }
}
