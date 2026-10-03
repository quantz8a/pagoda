// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! Laya triage: a System-1 gate in front of generation (System 2).
//!
//! Before a chat/generate request reaches the upstream LLM worker, the
//! gateway can ask a Laya decision server (`pagoda-hf`'s `laya_server`,
//! Jev-compatible `/decide`) to judge the user's text. Routing policy:
//!
//! * escalate (block generation, return a human-handoff reply) when
//!   `needs_human` fires, churn risk passes `--churn-threshold`, or the
//!   department confidence is below `--min-confidence`;
//! * otherwise forward to the upstream worker untouched;
//! * if the Laya server is unreachable, **fail open** (forward anyway) and
//!   count it in `triage_unavailable` — availability beats enforcement
//!   unless you pass `--laya-required` to fail closed.
//!
//! Zero-dependency: uses the crate's own `http_client` and `json`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::http_client;
use crate::json::{self, Value};

/// Built-in triage questions. `state` is filled with the user text.
fn triage_body(state: &str) -> String {
    // Note: json string escaping for the state text.
    let escaped: String = state
        .chars()
        .flat_map(|c| match c {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            '\n' => "\\n".chars().collect(),
            '\r' => "\\r".chars().collect(),
            '\t' => "\\t".chars().collect(),
            c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32).chars().collect(),
            c => vec![c],
        })
        .collect();
    format!(
        r#"{{"state":"{escaped}","questions":{{"department":{{"type":"choice","instructions":"Which department should handle this?","criteria":{{"billing":"invoices, payments, refunds","shipping":"delivery, logistics, address","technical":"bugs, outages, login, API","product":"product quality, features, warranty","other":"everything else"}}}},"churn_risk":{{"type":"noul","instructions":"Does the user threaten to cancel, leave, dispute, or complain publicly?"}},"needs_human":{{"type":"noul","instructions":"Does this require a human agent (legal threats, complaints about staff, complex negotiation)?"}}}}}}"#
    )
}

/// What triage concluded about one request.
#[derive(Clone, Debug)]
pub struct TriageOutcome {
    /// True → the gateway should NOT forward to the upstream worker.
    pub escalate: bool,
    /// Human-readable reason (shown in the handoff reply and /stats logs).
    pub reason: String,
    pub department: String,
    pub churn_risk: f64,
    pub needs_human: f64,
    pub confidence: f64,
    /// False when the Laya server could not be reached (fail-open path).
    pub available: bool,
}

/// Triage configuration + counters (one per server, shared across threads).
pub struct Triage {
    addr: String,
    pub churn_threshold: f64,
    pub min_confidence: f64,
    /// Shadow mode: triage runs and counts, but never blocks the request.
    pub shadow: bool,
    /// Fail closed when the Laya server is unreachable (default: fail open).
    pub required: bool,
    timeout: Duration,
    triaged: AtomicU64,
    escalated: AtomicU64,
    unavailable: AtomicU64,
}

impl Triage {
    /// `url` like `http://127.0.0.1:8081`. Returns None for a bad URL.
    pub fn from_url(url: &str) -> Option<Self> {
        Some(Self {
            addr: http_client::parse_http_addr(url)?,
            churn_threshold: 0.5,
            min_confidence: 0.0,
            shadow: false,
            required: false,
            timeout: Duration::from_secs(5),
            triaged: AtomicU64::new(0),
            escalated: AtomicU64::new(0),
            unavailable: AtomicU64::new(0),
        })
    }

    pub fn url(&self) -> &str {
        &self.addr
    }
    pub fn triaged(&self) -> u64 {
        self.triaged.load(Ordering::Relaxed)
    }
    pub fn escalated(&self) -> u64 {
        self.escalated.load(Ordering::Relaxed)
    }
    pub fn unavailable(&self) -> u64 {
        self.unavailable.load(Ordering::Relaxed)
    }

    /// Ask Laya about `user_text`. Never panics; unreachable → fail-open.
    pub fn check(&self, user_text: &str) -> TriageOutcome {
        self.triaged.fetch_add(1, Ordering::Relaxed);
        match self.check_inner(user_text) {
            Ok(out) => {
                if out.escalate {
                    self.escalated.fetch_add(1, Ordering::Relaxed);
                }
                Ok(out)
            }
            Err(e) => Err(e),
        }
        .unwrap_or_else(|e| {
            self.unavailable.fetch_add(1, Ordering::Relaxed);
            TriageOutcome {
                escalate: self.required,
                reason: format!("triage unavailable ({e}); {}", if self.required {
                    "fail-closed: escalating"
                } else {
                    "fail-open: forwarding"
                }),
                department: String::new(),
                churn_risk: 0.0,
                needs_human: 0.0,
                confidence: 0.0,
                available: false,
            }
        })
    }

    fn check_inner(&self, user_text: &str) -> Result<TriageOutcome, String> {
        let (status, body) = http_client::request(
            &self.addr,
            "POST",
            "/decide",
            Some(&triage_body(user_text)),
            self.timeout,
        )
        .map_err(|e| format!("http: {e}"))?;
        if status != 200 {
            return Err(format!("laya /decide -> {status}"));
        }
        let v = json::parse(&body)?;
        let answers = v.get("answers").ok_or("missing answers")?;
        let get_f = |q: &str, field: &str| -> f64 {
            answers
                .get(q)
                .and_then(|a| a.get(field))
                .and_then(Value::as_f64)
                .unwrap_or(0.0)
        };
        let department = answers
            .get("department")
            .and_then(|a| a.get("choice"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let confidence = get_f("department", "confidence");
        let churn = get_f("churn_risk", "noul");
        let human = get_f("needs_human", "noul");

        let mut reasons = Vec::new();
        if human > 0.5 {
            reasons.push(format!("needs_human={human:.2}"));
        }
        if churn > self.churn_threshold {
            reasons.push(format!(
                "churn_risk={churn:.2} > {}",
                self.churn_threshold
            ));
        }
        if confidence < self.min_confidence {
            reasons.push(format!(
                "confidence={confidence:.2} < {}",
                self.min_confidence
            ));
        }
        let escalate = !reasons.is_empty();
        Ok(TriageOutcome {
            escalate,
            reason: if escalate {
                reasons.join(", ")
            } else {
                "ok".to_string()
            },
            department,
            churn_risk: churn,
            needs_human: human,
            confidence,
            available: true,
        })
    }

    /// OpenAI-shaped assistant reply used when a request is escalated.
    pub fn escalation_body(&self, out: &TriageOutcome) -> String {
        let content = format!(
            "您的诉求已收到。为保障您的权益，已为您优先转接人工客服专员处理，请稍候。（pagoda 分诊：{}）",
            out.reason
        );
        let escaped: String = content
            .chars()
            .flat_map(|c| match c {
                '"' => "\\\"".chars().collect::<Vec<_>>(),
                '\\' => "\\\\".chars().collect(),
                c => vec![c],
            })
            .collect();
        format!(
            r#"{{"id":"pagoda-triage","object":"chat.completion","model":"pagoda-triage","choices":[{{"index":0,"message":{{"role":"assistant","content":"{escaped}"}},"finish_reason":"stop"}}],"pagoda_triage":{{"escalated":true,"department":"{}","churn_risk":{:.4},"needs_human":{:.4},"confidence":{:.4},"reason":"{}"}}}}"#,
            out.department, out.churn_risk, out.needs_human, out.confidence, out.reason
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;

    fn mock_laya(reply_json: &'static str) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", listener.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = vec![0u8; 65536];
            let _ = s.read(&mut buf).unwrap();
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply_json.len(),
                reply_json
            );
            use std::io::Write;
            s.write_all(resp.as_bytes()).unwrap();
        });
        (addr, h)
    }

    #[test]
    fn parses_laya_answer_and_escalates_on_churn() {
        let (url, h) = mock_laya(
            r#"{"model":"rl-agent","answers":{"department":{"type":"choice","choice":"billing","confidence":0.93},"churn_risk":{"type":"noul","noul":0.88},"needs_human":{"type":"noul","noul":0.1}},"usage":{"input_tokens":10}}"#,
        );
        let t = Triage::from_url(&url).unwrap();
        let out = t.check("重复扣款，不退款我就注销");
        assert!(out.available);
        assert_eq!(out.department, "billing");
        assert!(out.escalate);
        assert!(out.reason.contains("churn_risk"));
        h.join().unwrap();
    }

    #[test]
    fn passes_clean_request() {
        let (url, h) = mock_laya(
            r#"{"answers":{"department":{"choice":"shipping","confidence":0.97},"churn_risk":{"noul":0.02},"needs_human":{"noul":0.01}}}"#,
        );
        let t = Triage::from_url(&url).unwrap();
        let out = t.check("帮我改一下收货地址谢谢");
        assert!(!out.escalate);
        assert_eq!(out.reason, "ok");
        h.join().unwrap();
    }

    #[test]
    fn fails_open_when_laya_down() {
        let t = Triage::from_url("http://127.0.0.1:1").unwrap();
        let out = t.check("hello");
        assert!(!out.available);
        assert!(!out.escalate, "default fail-open");
        assert_eq!(t.unavailable(), 1);
    }

    #[test]
    fn fails_closed_when_required() {
        let mut t = Triage::from_url("http://127.0.0.1:1").unwrap();
        t.required = true;
        let out = t.check("hello");
        assert!(out.escalate, "required => fail-closed");
    }

    #[test]
    fn escapes_control_chars_in_state() {
        let body = triage_body("a\"b\nc\\d");
        let v = json::parse(&body).expect("must stay valid JSON");
        assert_eq!(
            v.get("state").and_then(Value::as_str),
            Some("a\"b\nc\\d")
        );
    }
}