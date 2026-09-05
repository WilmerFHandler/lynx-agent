use std::error::Error;
use std::future::Future;
use std::pin::Pin;

use futures::Stream;

use crate::compact::{CompactError, CompactOptions};
use crate::{AssistantMessage, Conversation, ToolSpec};

/// Provisional progress and the authoritative result of one provider round.
pub enum ProviderEvent<C> {
    /// Text which may be displayed while the request is running. It is not a
    /// conversation checkpoint and may be followed by an error or cancellation.
    TextDelta(String),
    /// The only event whose message and continuation may be committed. This is
    /// terminal: a provider must emit no events after it.
    Completed(AssistantMessage, C),
}

/// A provider response stream. Successful streams end with exactly one
/// [`ProviderEvent::Completed`] event and emit nothing afterward. Agent
/// consumers intentionally stop at the first completion, so trailing errors or
/// events from a nonconforming producer are not observed.
pub type ProviderStream<'a, C, E> =
    Pin<Box<dyn Stream<Item = Result<ProviderEvent<C>, E>> + Send + 'a>>;

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

    /// Stream one assistant response and its next continuation checkpoint.
    ///
    /// The default keeps complete-only providers source-compatible. Streaming
    /// providers may emit text deltas, but must emit owned authoritative state
    /// only in the final `Completed` event and must end immediately afterward.
    fn complete_stream<'a>(
        &'a self,
        continuation: &'a Self::Continuation,
        model: &'a Self::Model,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, Self::Continuation, Self::Error> {
        let completion = self.complete(continuation, model, conversation, tools);
        Box::pin(futures::stream::once(async move {
            completion
                .await
                .map(|(message, continuation)| ProviderEvent::Completed(message, continuation))
        }))
    }

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
