use std::error::Error;
use std::fmt;

use kodkod_core::Retryable;

#[derive(Debug)]
pub enum OpenAiError {
    Http(reqwest::Error),
    Api { status: u16, message: String },
    Json(serde_json::Error),
    EmptyResponse,
}

impl fmt::Display for OpenAiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(error) => write!(f, "http request failed: {error}"),
            Self::Api { status, message } => write!(f, "api error ({status}): {message}"),
            Self::Json(error) => write!(f, "failed to parse response: {error}"),
            Self::EmptyResponse => f.write_str("chat completion returned no choices"),
        }
    }
}

impl From<reqwest::Error> for OpenAiError {
    fn from(error: reqwest::Error) -> Self {
        Self::Http(error)
    }
}

impl From<serde_json::Error> for OpenAiError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl Error for OpenAiError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Http(error) => Some(error),
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl Retryable for OpenAiError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Http(error) => error.is_connect() || error.is_timeout() || error.is_request(),
            Self::Api { status, .. } => matches!(*status, 429 | 500 | 502 | 503 | 504),
            Self::Json(_) | Self::EmptyResponse => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_api_errors() {
        assert!(
            OpenAiError::Api {
                status: 429,
                message: "rate limited".into(),
            }
            .is_retryable()
        );
        assert!(
            OpenAiError::Api {
                status: 503,
                message: "unavailable".into(),
            }
            .is_retryable()
        );
        assert!(
            !OpenAiError::Api {
                status: 401,
                message: "unauthorized".into(),
            }
            .is_retryable()
        );
    }

    #[test]
    fn non_retryable_parse_errors() {
        assert!(!OpenAiError::EmptyResponse.is_retryable());
    }
}
