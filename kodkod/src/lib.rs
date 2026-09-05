//! Facade crate for [`kodkod-core`] and optional OpenAI-compatible providers.

pub use kodkod_core::*;

#[cfg(any(feature = "http", feature = "openai", feature = "anthropic"))]
pub use kodkod_http as http;

#[cfg(feature = "openai")]
pub use kodkod_openai as openai;

#[cfg(feature = "anthropic")]
pub use kodkod_anthropic as anthropic;

#[cfg(feature = "providers")]
pub use kodkod_providers as providers;
