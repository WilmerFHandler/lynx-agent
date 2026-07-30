use std::error::Error;
use std::future::Future;

use crate::{AssistantMessage, Conversation, ToolSpec};

/// A model provider whose opaque continuation state is scoped to one agent turn.
pub trait Provider: Sync {
    type Model: Sync;
    type Error: Error + Send + Sync + 'static;
    /// Provider-private state retained for exactly one execution turn.
    type TurnState: Send;

    fn supports_vision(&self, model: &Self::Model) -> bool;

    /// Whether the model is explicitly supported for computer-use tools.
    fn supports_computer_use(&self, _model: &Self::Model) -> bool {
        false
    }

    /// Create state for one execution turn. `Agent::run_turn` calls this once.
    fn create_turn_state(&self, model: &Self::Model) -> Self::TurnState;

    /// Produce one assistant response, advancing `state` only on success.
    ///
    /// The returned future may be dropped before completion when the agent turn
    /// is cancelled or its event stream is dropped. Implementations must leave
    /// committed state unchanged unless the future returns `Ok`.
    fn complete_round(
        &self,
        state: &mut Self::TurnState,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, Self::Error>> + Send;

    /// Perform an independent one-round operation with fresh state.
    fn complete_once(
        &self,
        model: &Self::Model,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, Self::Error>> + Send {
        async move {
            let mut state = self.create_turn_state(model);
            self.complete_round(&mut state, model, conversation, tools)
                .await
        }
    }
}
