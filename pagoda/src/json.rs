// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! A minimal JSON value, parser, and serializer. Self-contained (no `serde`)
//! because the runtime must compile offline.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(entries) => entries
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().and_then(|f| {
            if f >= 0.0 && f.fract() == 0.0 {
                Some(f as usize)
            } else {
                None
            }
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Parse a JSON document. Returns a human-readable error on invalid input.
pub fn parse(input: &str) -> Result<Value, String> {
    let mut p = Parser {
        chars: input.chars().collect(),
        pos: 0,
    };
    let v = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.chars.len() {
        return Err(format!("trailing characters at index {}", p.pos));
    }
    Ok(v)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
}

impl Parser {
    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        match self.next() {
            Some(x) if x == c => Ok(()),
            other => Err(format!("expected {:?}, found {:?} at {}", c, other, self.pos)),
        }
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        self.skip_ws();
        match self.peek() {
            None => Err("unexpected end of input".to_string()),
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('"') => Ok(Value::String(self.parse_string()?)),
            Some('t') => self.parse_literal("true", Value::Bool(true)),
            Some('f') => self.parse_literal("false", Value::Bool(false)),
            Some('n') => self.parse_literal("null", Value::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(format!("unexpected character {:?} at {}", c, self.pos)),
        }
    }

    fn parse_literal(&mut self, lit: &str, value: Value) -> Result<Value, String> {
        for expected in lit.chars() {
            match self.next() {
                Some(c) if c == expected => {}
                other => {
                    return Err(format!(
                        "expected literal {:?}, found {:?} at {}",
                        lit, other, self.pos
                    ))
                }
            }
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        if self.peek() == Some('-') {
            self.pos += 1;
        }
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let slice: String = self.chars[start..self.pos].iter().collect();
        slice
            .parse::<f64>()
            .map(Value::Number)
            .map_err(|_| format!("invalid number {:?}", slice))
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect('"')?;
        let mut out = String::new();
        while let Some(c) = self.next() {
            match c {
                '"' => return Ok(out),
                '\\' => match self.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('/') => out.push('/'),
                    Some('b') => out.push('\u{0008}'),
                    Some('f') => out.push('\u{000c}'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('u') => {
                        let hex: String = (0..4)
                            .map(|_| self.next().unwrap_or('\0'))
                            .collect();
                        let code = u32::from_str_radix(&hex, 16)
                            .map_err(|_| format!("invalid unicode escape \\u{}", hex))?;
                        out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                    }
                    other => return Err(format!("invalid escape {:?}", other)),
                },
                other => out.push(other),
            }
        }
        Err("unterminated string".to_string())
    }

    fn parse_object(&mut self) -> Result<Value, String> {
        self.expect('{')?;
        let mut entries = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.pos += 1;
            return Ok(Value::Object(entries));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':')?;
            let value = self.parse_value()?;
            entries.push((key, value));
            self.skip_ws();
            match self.next() {
                Some(',') => continue,
                Some('}') => break,
                other => {
                    return Err(format!(
                        "expected ',' or '}}', found {:?} at {}",
                        other, self.pos
                    ))
                }
            }
        }
        Ok(Value::Object(entries))
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.next() {
                Some(',') => continue,
                Some(']') => break,
                other => {
                    return Err(format!(
                        "expected ',' or ']', found {:?} at {}",
                        other, self.pos
                    ))
                }
            }
        }
        Ok(Value::Array(items))
    }
}

impl Value {
    /// Serialize this value compactly.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Value::Null => out.push_str("null"),
            Value::Bool(true) => out.push_str("true"),
            Value::Bool(false) => out.push_str("false"),
            Value::Number(n) => {
                if n.fract() == 0.0 && n.is_finite() && n.abs() < 1e15 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{}", n));
                }
            }
            Value::String(s) => write_string(out, s),
            Value::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Value::Object(entries) => {
                out.push('{');
                for (i, (k, v)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_string(out, k);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_json())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_object() {
        let v = parse(r#"{"a":1,"b":[true,null,"x"]}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_usize(), Some(1));
        match v.get("b").unwrap() {
            Value::Array(items) => assert_eq!(items.len(), 3),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn roundtrip_escapes() {
        let v = Value::String("line\n\"quote\"\\".to_string());
        let ser = v.to_json();
        let back = parse(&ser).unwrap();
        assert_eq!(back, v);
    }

    #[test]
    fn parse_errors() {
        assert!(parse("{").is_err());
        assert!(parse("[1,]").is_err());
    }
}
