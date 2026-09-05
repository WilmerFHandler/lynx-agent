//! OpenAI-compatible [`Provider`] implementation for [`kodkod_core`].

mod api;
mod completion;
mod convert;
mod error;
mod model;
mod provider;
mod responses;

pub use completion::{chat_completions_url, complete, complete_with_credentials};
pub use error::OpenAiError;
pub use kodkod_http::{
    CredentialError, CredentialFuture, CredentialSource, RequestCredentials, StaticCredentials,
};
pub use model::OpenAiModel;
pub use provider::OpenAiCompatibleProvider;
pub use responses::{OpenAiResponsesProvider, ResponsesContinuation, responses_url};
