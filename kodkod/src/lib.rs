//! Facade crate for [`kodkod-core`] and optional OpenAI-compatible providers.

pub use kodkod_core::*;

#[cfg(feature = "openai")]
pub use kodkod_openai as openai;
