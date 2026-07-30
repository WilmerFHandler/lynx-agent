pub mod control;
pub mod error;
pub mod event;

use std::borrow::Cow;
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

/// Opaque provider continuation retained by an application across agent turns.
///
/// Create one with [`Agent::new_context`] and pass it to [`Agent::run_turn_in`].
/// Its checkpoint advances only when the corresponding assistant message is
/// accepted into the conversation.
pub struct AgentContext<P: Provider> {
    continuation: P::Continuation,
}

impl<P: Provider> AgentContext<P> {
    /// Conservatively estimate provider-private tokens replayed on the next request.
    pub fn estimated_continuation_tokens(&self) -> u64 {
        P::estimate_continuation_tokens(&self.continuation)
    }
}

enum ContextSlot<'a, P: Provider> {
    Owned(AgentContext<P>),
    Borrowed(&'a mut AgentContext<P>),
}

impl<P: Provider> ContextSlot<'_, P> {
    fn continuation(&self) -> &P::Continuation {
        match self {
            Self::Owned(context) => &context.continuation,
            Self::Borrowed(context) => &context.continuation,
        }
    }

    fn checkpoint(&mut self, continuation: P::Continuation) {
        match self {
            Self::Owned(context) => context.continuation = continuation,
            Self::Borrowed(context) => context.continuation = continuation,
        }
    }
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

    /// Create a fresh context which can retain provider continuation across turns.
    pub fn new_context(&self, model: &P::Model) -> AgentContext<P> {
        AgentContext {
            continuation: self.provider.create_continuation(model),
        }
    }

    /// Run one logical user turn with a fresh provider continuation.
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
        conversation.push_message(Message::User(user.with_steered(false)));
        let context = ContextSlot::Owned(self.new_context(model));
        self.run_opened(conversation, model, control, context, TurnGuard { control })
    }

    /// Run one logical user turn in a retained provider context.
    pub fn run_turn_in<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a P::Model,
        user: UserMessage,
        context: &'a mut AgentContext<P>,
        control: &'a TaskControl,
    ) -> Task<'a, P::Error> {
        if control.open().is_err() {
            return Box::pin(futures::stream::once(async {
                Err(AgentError::ControlAlreadyUsed)
            }));
        }
        conversation.push_message(Message::User(user.with_steered(false)));
        self.run_opened(
            conversation,
            model,
            control,
            ContextSlot::Borrowed(context),
            TurnGuard { control },
        )
    }

    /// Compatibility entry point. Uses a fresh continuation without appending a prompt.
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
        let context = ContextSlot::Owned(self.new_context(model));
        self.run_opened(conversation, model, control, context, TurnGuard { control })
    }

    fn run_opened<'a>(
        &'a self,
        conversation: &'a mut Conversation,
        model: &'a P::Model,
        control: &'a TaskControl,
        context: ContextSlot<'a, P>,
        guard: TurnGuard<'a>,
    ) -> Task<'a, P::Error> {
        Box::pin(try_stream! {
            // Keep these options so terminal paths release retained state and
            // close steering before yielding Completed.
            let mut context = Some(context);
            let mut guard = Some(guard);
            let vision_enabled = self.provider.supports_vision(model);
            let computer_use_enabled = self.provider.supports_computer_use(model);
            let tool_specs = self.tools.specs_for_capabilities(vision_enabled, computer_use_enabled);
            let mut tool_rounds_executed = 0;

            loop {
                let steers = control.drain_pending_steers_unless_cancelled()
                    .ok_or(AgentError::Cancelled)?;
                for user in steers {
                    conversation.push_message(Message::User(user.clone()));
                    yield AgentEvent::Steered(user);
                }

                let provider_input: Cow<'_, Conversation> = if vision_enabled {
                    Cow::Borrowed(conversation)
                } else {
                    Cow::Owned(conversation.without_images())
                };

                let result = {
                    let completion = self.provider.complete(
                        context.as_ref().expect("agent context exists").continuation(),
                        model,
                        &provider_input,
                        &tool_specs,
                    );
                    let cancellation = control.cancelled();
                    futures::pin_mut!(completion, cancellation);
                    futures::future::poll_fn(|cx| {
                        // Cancellation wins when both become ready in one poll.
                        if cancellation.as_mut().poll(cx).is_ready() {
                            Poll::Ready(None)
                        } else {
                            completion.as_mut().poll(cx).map(Some)
                        }
                    }).await
                };

                let (message, next_continuation) = match result {
                    Some(Ok(completion)) if !control.is_cancelled() => completion,
                    Some(Ok(_)) | None => Err(AgentError::Cancelled)?,
                    Some(Err(error)) => Err(AgentError::Provider(error))?,
                };

                // The transcript and continuation are one logical checkpoint.
                // No cancellation check occurs between these operations: once
                // accepted, both survive cancellation racing immediately after.
                let tool_calls = message.tool_calls().to_vec();
                conversation.push_message(Message::Assistant(message.clone()));
                context.as_mut().expect("agent context exists").checkpoint(next_continuation);
                yield AgentEvent::AssistantReply(message.clone());

                if tool_calls.is_empty() {
                    if !control.close_if_empty() {
                        continue;
                    }
                    drop(context.take());
                    drop(guard.take());
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
