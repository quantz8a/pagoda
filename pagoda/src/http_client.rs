// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! A minimal, dependency-free blocking HTTP/1.1 client.
//!
//! Just enough to reverse-proxy JSON requests to an upstream inference
//! server (e.g. SGLang): one connection per request, `Content-Length`
//! framing, `Connection: close`. Streaming (SSE/chunked) is deliberately
//! out of scope — the proxy buffers the full response.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Default connect/read/write timeout for upstream calls.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// Parse `http://host[:port]` into `host:port` (default port 80).
/// Returns `None` for anything else (https is out of scope for a LAN
/// co-deployment; put a TLS terminator in front if needed).
pub fn parse_http_addr(url: &str) -> Option<String> {
    let rest = url.strip_prefix("http://")?;
    let authority = rest.split('/').next().unwrap_or("");
    if authority.is_empty() {
        return None;
    }
    if authority.contains(':') {
        Some(authority.to_string())
    } else {
        Some(format!("{authority}:80"))
    }
}

/// One blocking HTTP/1.1 round trip; returns (status, body).
///
/// The whole response body is read (Content-Length, or until EOF when the
/// server closes without one). Errors surface as `io::Error` so the caller
/// can degrade (502) instead of panicking.
pub fn request(
    addr: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
    timeout: Duration,
) -> std::io::Result<(u16, String)> {
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    stream.set_nodelay(true).ok();

    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes())?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(502);

    let mut content_length: Option<usize> = None;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        if header == "\r\n" || header == "\n" || header.is_empty() {
            break;
        }
        let lower = header.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().ok();
        }
    }

    let body = match content_length {
        Some(n) => {
            let mut buf = vec![0u8; n];
            reader.read_exact(&mut buf)?;
            String::from_utf8_lossy(&buf).into_owned()
        }
        None => {
            let mut buf = String::new();
            reader.read_to_string(&mut buf)?;
            buf
        }
    };
    Ok((status, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_addr() {
        assert_eq!(
            parse_http_addr("http://127.0.0.1:30001"),
            Some("127.0.0.1:30001".to_string())
        );
        assert_eq!(
            parse_http_addr("http://gpu-box/"),
            Some("gpu-box:80".to_string())
        );
        assert_eq!(parse_http_addr("https://x"), None);
        assert_eq!(parse_http_addr("http://"), None);
    }
}