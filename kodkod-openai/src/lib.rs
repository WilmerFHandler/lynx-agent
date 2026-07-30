//! OpenAI-compatible [`Provider`] implementation for [`kodkod_core`].

mod api;
mod completion;
mod convert;
mod error;
mod model;
mod provider;

pub use completion::{chat_completions_url, complete};
pub use error::OpenAiError;
pub use model::OpenAiModel;
pub use provider::OpenAiCompatibleProvider;
