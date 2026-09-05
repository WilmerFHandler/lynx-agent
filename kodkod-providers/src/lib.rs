//! Fixed-endpoint providers for Codex subscriptions and OpenCode services.
//!
//! Applications remain responsible for login, token refresh, account selection,
//! and credential storage. These providers ask application-owned sources for
//! fresh credentials immediately before a request and own the service endpoint,
//! protocol routing, and required headers.

mod codex;
mod opencode;

use std::error::Error;
use std::fmt;

use reqwest::header::HeaderValue;
use serde::{Deserialize, Serialize};

pub use codex::{
    CODEX_ENDPOINT, CodexAccess, CodexAccessFuture, CodexCredentialSource, CodexProvider,
};
pub use opencode::{
    OPEN_CODE_GO_ENDPOINT, OPEN_CODE_ZEN_ENDPOINT, OpenCodeContinuation, OpenCodeModel,
    OpenCodeProvider, OpenCodeService, ProviderError, open_code_catalog, open_code_model,
};

/// An HTTP model protocol, independent of the service hosting the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    Responses,
    ChatCompletions,
    Messages,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderConfigError {
    message: String,
}

impl ProviderConfigError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ProviderConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for ProviderConfigError {}

fn header_value(name: &str, value: &str) -> Result<HeaderValue, ProviderConfigError> {
    HeaderValue::from_str(value)
        .map_err(|_| ProviderConfigError::new(format!("invalid {name} header value")))
}

fn fixed_endpoint_client() -> Result<reqwest::Client, ProviderConfigError> {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| ProviderConfigError::new(format!("failed to build HTTP client: {error}")))
}

#[cfg(test)]
mod tests;
