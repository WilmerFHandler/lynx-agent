//! A small provider-agnostic agent loop with tool calling support.
//!
//! `kodkod-core` provides the core types needed to run an agent against any
//! backend that implements [`Provider`]. The crate keeps provider integration,
//! conversation state, and tool execution separate so applications can bring
//! their own model backend and tool implementations.

pub mod agent;
pub mod conversation;
pub mod message;
pub mod provider;
pub mod retry;
pub mod tool;
pub mod turns;

mod compact;
mod estimate;

pub use agent::{Agent, AgentContext, AgentError, AgentEvent, SteerError, Task, TaskControl};
pub use compact::{CompactError, CompactOptions, DEFAULT_KEEP_TAIL_TOKENS, TurnCompaction};
pub use conversation::Conversation;
pub use message::{AssistantMessage, Image, Message, SystemMessage, UserMessage};
pub use provider::{Provider, ProviderEvent, ProviderStream};
pub use retry::{RetryPolicy, RetryProvider, Retryable};
pub use tool::{
    Tool, ToolCall, ToolError, ToolExecutor, ToolExecutorError, ToolFuture, ToolOutput, ToolResult,
    ToolResultOutcome, ToolSpec,
};
pub use turns::{Turn, TurnIter, Turns, turns};

#[cfg(test)]
mod tests;
