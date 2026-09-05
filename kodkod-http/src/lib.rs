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
}
