// Copyright (C) 2026  quantz8a
// SPDX-License-Identifier: Apache-2.0
//! A minimal, dependency-free blocking HTTP/1.1 client.
//!
//! Just enough to reverse-proxy requests to an upstream inference
//! server (e.g. SGLang): one connection per request, `Content-Length` and
//! `Transfer-Encoding: chunked` framing, `Connection: close`. Buffered by
//! default; [`request_open`] exposes the body as a stream for SSE relay.

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
/// The whole response body is drained via [`request_open`] (chunked bodies
/// are transparently de-chunked). Errors surface as `io::Error` so the
/// caller can degrade (502) instead of panicking.
pub fn request(
    addr: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
    timeout: Duration,
) -> std::io::Result<(u16, String)> {
    let mut up = request_open(addr, method, path, body, timeout)?;
    let mut body = Vec::new();
    while let Some(payload) = up.next_payload()? {
        body.extend_from_slice(&payload);
    }
    Ok((up.status, String::from_utf8_lossy(&body).into_owned()))
}

/// An upstream response with the body still on the wire: status and headers
/// are parsed, payloads are pulled one at a time via [`next_payload`].
///
/// [`next_payload`]: UpstreamStream::next_payload
pub struct UpstreamStream {
    reader: BufReader<TcpStream>,
    /// HTTP status code from the status line (502 when unparseable).
    pub status: u16,
    /// Whether the body uses `Transfer-Encoding: chunked` (SSE streams do).
    pub chunked: bool,
    /// Upstream `Content-Type` (e.g. `text/event-stream`), if present.
    pub content_type: Option<String>,
    remaining: Option<usize>,
    done: bool,
}

impl UpstreamStream {
    /// Pull the next body payload: one chunk for chunked bodies, up to
    /// 64 KiB otherwise. `Ok(None)` marks the end of the body.
    pub fn next_payload(&mut self) -> std::io::Result<Option<Vec<u8>>> {
        if self.done {
            return Ok(None);
        }
        if self.chunked {
            let mut line = String::new();
            self.reader.read_line(&mut line)?;
            let size_field = line.trim().split(';').next().unwrap_or("");
            let size = usize::from_str_radix(size_field, 16).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad chunk size: {e}"))
            })?;
            if size == 0 {
                // Drain optional trailers up to the blank line.
                loop {
                    let mut trailer = String::new();
                    self.reader.read_line(&mut trailer)?;
                    if trailer == "\r\n" || trailer == "\n" || trailer.is_empty() {
                        break;
                    }
                }
                self.done = true;
                return Ok(None);
            }
            let mut buf = vec![0u8; size];
            self.reader.read_exact(&mut buf)?;
            let mut crlf = [0u8; 2];
            self.reader.read_exact(&mut crlf)?;
            return Ok(Some(buf));
        }
        match self.remaining {
            Some(0) => {
                self.done = true;
                Ok(None)
            }
            Some(left) => {
                let take = left.min(64 * 1024);
                let mut buf = vec![0u8; take];
                self.reader.read_exact(&mut buf)?;
                self.remaining = Some(left - take);
                Ok(Some(buf))
            }
            None => {
                // No framing: the body runs until the server closes.
                let mut buf = Vec::new();
                self.reader.read_to_end(&mut buf)?;
                self.done = true;
                if buf.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(buf))
                }
            }
        }
    }
}

/// Open one upstream request and parse the status line and headers, leaving
/// the body to be streamed through [`UpstreamStream`].
pub fn request_open(
    addr: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
    timeout: Duration,
) -> std::io::Result<UpstreamStream> {
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
    let mut chunked = false;
    let mut content_type: Option<String> = None;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header)?;
        if header == "\r\n" || header == "\n" || header.is_empty() {
            break;
        }
        let lower = header.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            content_length = rest.trim().parse().ok();
        } else if let Some(rest) = lower.strip_prefix("transfer-encoding:") {
            chunked = rest.contains("chunked");
        } else if let Some(rest) = lower.strip_prefix("content-type:") {
            content_type = Some(rest.trim().to_string());
        }
    }

    Ok(UpstreamStream {
        reader,
        status,
        chunked,
        content_type,
        remaining: content_length,
        done: false,
    })
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

    /// A loopback peer answering with chunked framing: two data chunks plus
    /// a trailer, then the terminal zero chunk.
    #[test]
    fn chunked_bodies_dechunk_in_order() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr").to_string();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut stream) = conn else { continue };
                std::thread::spawn(move || {
                    // Drain the request (head + fixed-length body).
                    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
                    let mut content_length = 0usize;
                    let mut line = String::new();
                    reader.read_line(&mut line).expect("request line");
                    loop {
                        let mut header = String::new();
                        reader.read_line(&mut header).expect("header");
                        if header == "\r\n" || header.is_empty() {
                            break;
                        }
                        if let Some(rest) =
                            header.to_ascii_lowercase().strip_prefix("content-length:")
                        {
                            content_length = rest.trim().parse().unwrap_or(0);
                        }
                    }
                    let mut body = vec![0u8; content_length];
                    reader.read_exact(&mut body).expect("body");
                    let resp = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n\
                                5\r\nhello\r\n6\r\n world\r\n0\r\nX-Trailer: done\r\n\r\n";
                    stream.write_all(resp.as_bytes()).expect("write");
                });
            }
        });

        // Buffered API: chunks are transparently concatenated.
        let (status, body) = request(&addr, "POST", "/generate", Some("{}"), Duration::from_secs(5))
            .expect("request");
        assert_eq!(status, 200);
        assert_eq!(body, "hello world");

        // Streaming API: payloads arrive chunk by chunk with metadata intact.
        let mut up = request_open(&addr, "POST", "/generate", Some("{}"), Duration::from_secs(5))
            .expect("request_open");
        assert_eq!(up.status, 200);
        assert!(up.chunked);
        assert_eq!(up.content_type.as_deref(), Some("text/event-stream"));
        assert_eq!(up.next_payload().expect("chunk1"), Some(b"hello".to_vec()));
        assert_eq!(up.next_payload().expect("chunk2"), Some(b" world".to_vec()));
        assert_eq!(up.next_payload().expect("end"), None);
    }
}