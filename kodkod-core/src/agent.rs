pub mod control;
pub mod error;
pub mod event;

use std::borrow::Cow;
use std::future::Future;
use std::sync::Arc;
use std::task::Poll;

use async_stream::try_stream;

pub use control::{SteerError, TaskControl};
pub use error::AgentError;
pub use event::AgentEvent;

/// Streaming events from an [`Agent::run_turn`] call.
pub type Task<'a, E> = futures::stream::BoxStream<'a, Result<AgentEvent, AgentError<E>>>;

use crate::{Conversation, Message, Provider, Tool, ToolExecutor, UserMessage};

pub struct Agent<P> {
    provider: P,
    tools: ToolExecutor,
    max_tool_rounds: Option<usize>,
}

struct TurnGuard<'a> {
    control: &'a TaskControl,
}
impl Drop for TurnGuard<'_> {
    fn drop(&mut self) {
        self.control.close();
    }
}

impl<P> Agent<P>
where
    P: Provider,
{
    pub fn new(provider: P) -> Self {
        Self {
            provider,
            tools: ToolExecutor::new(),
            max_tool_rounds: None,
        }
    }
    pub fn provider(&self) -> &P {
        &self.provider
    }
    pub fn tools(&self) -> &ToolExecutor {
        &self.tools
    }
    pub fn max_tool_rounds(&self) -> Option<usize> {
        self.max_tool_rounds
    }
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.register_tool(tool);
        self
    }
    pub fn with_max_tool_rounds(mut self, max: usize) -> Self {
        self.max_tool_rounds = Some(max);
        self
    }
    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) {
        self.tools.register(tool);
    }

    /// Run one logical user turn with one provider-defined state value.
    pub fn run_turn<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a P::Model,
        user: UserMessage,
        control: &'a TaskControl,
    ) -> Task<'a, P::Error> {
        if control.open().is_err() {
            return Box::pin(futures::stream::once(async {
                Err(AgentError::ControlAlreadyUsed)
            }));
        }
        let guard = TurnGuard { control };
        conversation.push_message(Message::User(user.with_steered(false)));
        let state = self.provider.create_turn_state(model);
        self.run_opened(conversation, model, control, state, guard)
    }

    /// Compatibility entry point. Creates one state, but does not append a prompt.
    #[deprecated(note = "use run_turn; it explicitly starts a logical user turn")]
    pub fn run<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a P::Model,
        control: &'a TaskControl,
    ) -> Task<'a, P::Error> {
        if control.open().is_err() {
            return Box::pin(futures::stream::once(async {
                Err(AgentError::ControlAlreadyUsed)
            }));
        }
        let guard = TurnGuard { control };
        let state = self.provider.create_turn_state(model);
        self.run_opened(conversation, model, control, state, guard)
    }

    fn run_opened<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a P::Model,
        control: &'a TaskControl,
        state: P::TurnState,
        guard: TurnGuard<'a>,
    ) -> Task<'a, P::Error> {
        Box::pin(try_stream! {
            // Keep this option so terminal paths can drop state before yielding.
            let mut state = Some(state);
            let vision_enabled = self.provider.supports_vision(model);
            let computer_use_enabled = self.provider.supports_computer_use(model);
            let tool_specs = self.tools.specs_for_capabilities(vision_enabled, computer_use_enabled);
            let mut tool_rounds_executed = 0;

            loop {
                if control.is_cancelled() { Err(AgentError::Cancelled)?; }

                for user in control.drain_pending_steers() {
                    conversation.push_message(Message::User(user.clone()));
                    yield AgentEvent::Steered(user);
                }

                let provider_input: Cow<'_, Conversation> = if vision_enabled {
                    Cow::Borrowed(conversation)
                } else {
                    Cow::Owned(conversation.without_images())
                };

                let message = {
                    let completion = self.provider.complete_round(
                        state.as_mut().expect("turn state exists"), model, &provider_input, &tool_specs,
                    );
                    let cancellation = control.cancelled();
                    futures::pin_mut!(completion, cancellation);
                    let result = futures::future::poll_fn(|cx| {
                        if cancellation.as_mut().poll(cx).is_ready() {
                            Poll::Ready(None)
                        } else {
                            completion.as_mut().poll(cx).map(Some)
                        }
                    }).await;
                    match result {
                        Some(result) => {
                            if control.is_cancelled() { Err(AgentError::Cancelled)?; }
                            result.map_err(AgentError::Provider)?
                        }
                        None => Err(AgentError::Cancelled)?,
                    }
                };
                let tool_calls = message.tool_calls().to_vec();
                conversation.push_message(Message::Assistant(message.clone()));
                yield AgentEvent::AssistantReply(message.clone());

                if tool_calls.is_empty() {
                    if !control.close_if_empty() {
                        continue;
                    }
                    drop(state.take());
                    drop(guard);
                    yield AgentEvent::Completed(message);
                    return;
                }

                if let Some(max) = self.max_tool_rounds
                    && tool_rounds_executed >= max
                {
                    Err(AgentError::MaxToolRoundsExceeded { max })?;
                }

                for tool_call in &tool_calls {
                    yield AgentEvent::ToolStarted(tool_call.clone());
                }
                let results = futures::future::join_all(tool_calls.iter().map(|tool_call| {
                    self.tools.execute_for_capabilities(
                        tool_call, vision_enabled, computer_use_enabled,
                    )
                })).await;
                for result in results {
                    conversation.push_message(Message::ToolResult(result.clone()));
                    yield AgentEvent::ToolFinished(result);
                }
                tool_rounds_executed += 1;
            }
        })
    }
}
