use std::error::Error;
use std::future::Future;

use crate::compact::{CompactError, CompactOptions};
use crate::{AssistantMessage, Conversation, ToolSpec};

/// A model provider with an opaque, checkpointed continuation value.
pub trait Provider: Sync {
    type Model: Sync;
    type Error: Error + Send + Sync + 'static;
    /// Provider-private state which can be retained between agent turns.
    type Continuation: Send + Sync;

    fn supports_vision(&self, model: &Self::Model) -> bool;

    /// Whether the model is explicitly supported for computer-use tools.
    fn supports_computer_use(&self, _model: &Self::Model) -> bool {
        false
    }

    /// Create a fresh continuation.
    fn create_continuation(&self, model: &Self::Model) -> Self::Continuation;

    /// Conservatively estimate private tokens replayed by this continuation.
    ///
    /// The default is appropriate for providers whose continuation does not add
    /// request content. Implementations must not expose native contents while
    /// calculating or reporting this value.
    fn estimate_continuation_tokens(_continuation: &Self::Continuation) -> u64 {
        0
    }

    /// Produce one assistant response and its next continuation checkpoint.
    ///
    /// `continuation` is never modified in place. The caller accepts the returned
    /// checkpoint only when it also accepts the assistant message. Consequently,
    /// errors and dropped futures cannot advance retained provider state.
    fn complete(
        &self,
        continuation: &Self::Continuation,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<(AssistantMessage, Self::Continuation), Self::Error>> + Send;

    /// Perform an independent one-round operation with a fresh continuation.
    fn complete_once(
        &self,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, Self::Error>> + Send {
        async move {
            let continuation = self.create_continuation(model);
            self.complete(&continuation, model, conversation, tools)
                .await
                .map(|(message, _)| message)
        }
    }

    /// Summarize dropped prefix messages and return a smaller conversation.
    ///
    /// The default implementation is provider-agnostic: it plans a protocol-safe
    /// tail, asks `complete_once` for a summary, and splices that summary into a
    /// new conversation. The original conversation is not modified.
    fn compact(
        &self,
        model: &Self::Model,
        conversation: &Conversation,
        options: CompactOptions,
    ) -> impl Future<Output = Result<Conversation, CompactError<Self::Error>>> + Send {
        async move { crate::compact::run(self, model, conversation, options).await }
    }
}
