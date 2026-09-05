use std::error::Error;
use std::fmt;
use std::future::Future;
use std::pin::Pin;

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};

/// Headers supplied by the caller for one provider request.
///
/// Kodkod deliberately does not know how these credentials were acquired,
/// refreshed, or stored. A source is asked immediately before every request.
#[derive(Clone, Default)]
pub struct RequestCredentials {
    headers: HeaderMap,
}

impl RequestCredentials {
    pub fn bearer(token: impl AsRef<str>) -> Result<Self, CredentialError> {
        let mut value = HeaderValue::from_str(&format!("Bearer {}", token.as_ref()))
            .map_err(|error| CredentialError::new(error.to_string()))?;
        value.set_sensitive(true);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, value);
        Ok(Self { headers })
    }

    pub fn from_headers(mut headers: HeaderMap) -> Self {
        for value in headers.values_mut() {
            value.set_sensitive(true);
        }
        Self { headers }
    }

    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }
}

impl fmt::Debug for RequestCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestCredentials")
            .field("header_count", &self.headers.len())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CredentialError {
    message: String,
}

impl CredentialError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CredentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for CredentialError {}

pub type CredentialFuture<'a> =
    Pin<Box<dyn Future<Output = Result<RequestCredentials, CredentialError>> + Send + 'a>>;

/// Caller-owned, asynchronous source of per-request authentication headers.
pub trait CredentialSource: Send + Sync {
    fn credentials(&self) -> CredentialFuture<'_>;
}

#[derive(Clone)]
pub struct StaticCredentials(StaticCredentialKind);

#[derive(Clone)]
enum StaticCredentialKind {
    Headers(RequestCredentials),
    Bearer(String),
}

impl StaticCredentials {
    pub fn bearer(token: impl Into<String>) -> Self {
        Self(StaticCredentialKind::Bearer(token.into()))
    }

    pub fn new(credentials: RequestCredentials) -> Self {
        Self(StaticCredentialKind::Headers(credentials))
    }
}

impl CredentialSource for StaticCredentials {
    fn credentials(&self) -> CredentialFuture<'_> {
        Box::pin(async {
            match &self.0 {
                StaticCredentialKind::Headers(credentials) => Ok(credentials.clone()),
                StaticCredentialKind::Bearer(token) => RequestCredentials::bearer(token),
            }
        })
    }
}

impl fmt::Debug for StaticCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticCredentials").finish_non_exhaustive()
    }
}

const MAX_SSE_EVENT_BYTES: usize = 1024 * 1024;

/// Incremental decoder for UTF-8 Server-Sent Events carried across arbitrary byte chunks.
#[derive(Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct SseEvent {
    pub event_name: Option<String>,
    pub data: String,
}

impl SseDecoder {
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, String> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some((payload_end, delimiter_end)) = find_event_boundary(&self.buffer) {
            if payload_end > MAX_SSE_EVENT_BYTES {
                return Err("SSE event exceeded size limit".into());
            }
            let payload = self.buffer[..payload_end].to_vec();
            self.buffer.drain(..delimiter_end);
            if let Some(event) = parse_sse_block(&payload)? {
                events.push(event);
            }
        }
        if self.buffer.len() > MAX_SSE_EVENT_BYTES {
            return Err("SSE event exceeded size limit".into());
        }
        Ok(events)
    }

    pub fn finish(&self) -> Result<(), String> {
        if self.buffer.iter().any(|byte| !byte.is_ascii_whitespace()) {
            return Err("truncated SSE event".into());
        }
        Ok(())
    }
}

fn find_event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let mut line_start = 0;
    let mut index = 0;
    while index < buffer.len() {
        let delimiter_end = match buffer[index] {
            b'\n' => index + 1,
            b'\r' if index + 1 == buffer.len() => index + 1,
            b'\r' if buffer[index + 1] == b'\n' => index + 2,
            b'\r' => index + 1,
            _ => {
                index += 1;
                continue;
            }
        };
        if index == line_start {
            return Some((line_start, delimiter_end));
        }
        line_start = delimiter_end;
        index = delimiter_end;
    }
    None
}

fn parse_sse_block(payload: &[u8]) -> Result<Option<SseEvent>, String> {
    let block = std::str::from_utf8(payload)
        .map_err(|error| format!("SSE event was not UTF-8: {error}"))?;
    let mut event_name = None;
    let normalized = block.replace("\r\n", "\n").replace('\r', "\n");
    let mut data_lines = Vec::new();
    for line in normalized.split_terminator('\n') {
        if line.starts_with(':') {
            continue;
        }
        let (field, value) = line.split_once(':').map_or((line, ""), |(field, value)| {
            (field, strip_optional_space(value))
        });
        match field {
            "event" => event_name = Some(value.to_owned()),
            "data" => data_lines.push(value),
            _ => {}
        }
    }
    if data_lines.is_empty() {
        return Ok(None);
    }
    let data = data_lines.join("\n");
    Ok(Some(SseEvent { event_name, data }))
}

fn strip_optional_space(value: &str) -> &str {
    value.strip_prefix(' ').unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_builds_authorization_header() {
        let credentials = RequestCredentials::bearer("token").unwrap();
        assert_eq!(credentials.headers()[AUTHORIZATION], "Bearer token");
    }

    #[test]
    fn debug_never_reveals_credentials() {
        let credentials = RequestCredentials::bearer("top-secret").unwrap();
        assert!(!format!("{credentials:?}").contains("top-secret"));
        let source = StaticCredentials::bearer("top-secret");
        assert!(!format!("{source:?}").contains("top-secret"));
    }

    #[test]
    fn sse_decoder_preserves_data_whitespace_and_multiline_values() {
        let mut decoder = SseDecoder::default();
        let events = decoder
            .push(b"event: answer\ndata:  leading\ndata: second  \n\n")
            .unwrap();
        assert_eq!(
            events,
            [SseEvent {
                event_name: Some("answer".into()),
                data: " leading\nsecond  ".into(),
            }]
        );
    }

    #[test]
    fn sse_decoder_handles_empty_data_fields_and_ignores_comments() {
        let mut decoder = SseDecoder::default();
        let events = decoder.push(b": comment\ndata\ndata: x\n\n").unwrap();
        assert_eq!(events[0].data, "\nx");
    }

    #[test]
    fn sse_decoder_accepts_lf_crlf_cr_and_mixed_line_endings() {
        for bytes in [
            &b"data: one\n\n"[..],
            &b"data: one\r\n\r\n"[..],
            &b"data: one\r\r: next"[..],
            &b"event: x\r\ndata: one\r\n\r: next"[..],
        ] {
            let mut decoder = SseDecoder::default();
            let events = decoder.push(bytes).unwrap();
            assert_eq!(events[0].data, "one");
        }
    }

    #[test]
    fn sse_decoder_handles_fragmented_utf8_and_crlf_boundaries() {
        let bytes = "data: hällo\r\n\r\n".as_bytes();
        for split in 1..bytes.len() {
            let mut decoder = SseDecoder::default();
            let mut events = decoder.push(&bytes[..split]).unwrap();
            events.extend(decoder.push(&bytes[split..]).unwrap());
            assert_eq!(events.len(), 1, "split={split}");
            assert_eq!(events[0].data, "hällo");
        }
    }

    #[test]
    fn sse_decoder_dispatches_a_terminal_bare_cr_event_at_eof() {
        let mut decoder = SseDecoder::default();
        let events = decoder.push(b"data: [DONE]\r\r").unwrap();
        assert_eq!(events[0].data, "[DONE]");
        decoder.finish().unwrap();
    }

    #[test]
    fn sse_decoder_does_not_duplicate_data_across_a_split_crlf() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"data: first\r").unwrap().is_empty());
        let events = decoder.push(b"\ndata: second\r\n\r\n").unwrap();
        assert_eq!(events[0].data, "first\nsecond");
    }

    #[test]
    fn sse_decoder_rejects_oversized_and_truncated_events() {
        let mut oversized = SseDecoder::default();
        assert!(
            oversized
                .push(&vec![b'x'; MAX_SSE_EVENT_BYTES + 1])
                .is_err()
        );

        let mut truncated = SseDecoder::default();
        assert!(truncated.push(b"data: incomplete").unwrap().is_empty());
        assert!(truncated.finish().unwrap_err().contains("truncated"));
    }
}
