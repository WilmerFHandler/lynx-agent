use crate::{AssistantMessage, Conversation, ToolCall, ToolResult, UserMessage};

/// Incremental progress from a running agent turn.
///
/// Consumers matching variants explicitly must handle
/// [`AssistantTextDelta`](Self::AssistantTextDelta), which can occur before
/// each authoritative [`AssistantReply`](Self::AssistantReply).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentEvent {
    /// Provisional assistant text from the current provider round.
    AssistantTextDelta(String),
    /// The provider returned an assistant message for the current round.
    AssistantReply(AssistantMessage),
    /// A user message was steered into the turn at a round boundary.
    Steered(UserMessage),
    /// Compaction is about to run at this round boundary.
    CompactionStarted,
    /// The working conversation was replaced by a compacted snapshot.
    Compacted(Conversation),
    /// Compaction failed; the turn continues with the uncompacted conversation.
    CompactionFailed(String),
    /// A tool call is about to execute.
    ToolStarted(ToolCall),
    /// A tool call finished. Parallel calls emit this in completion order, not
    /// call order; [`ToolResult::tool_call_id`] associates each result.
    ToolFinished(ToolResult),
    /// The turn completed with a final assistant message (no pending tool calls).
    Completed(AssistantMessage),
}
