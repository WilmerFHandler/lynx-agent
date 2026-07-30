//! Structural user turns as views over a flat message transcript.
//!
//! These persisted transcript views are distinct from [`Provider::Continuation`](crate::Provider::Continuation),
//! which is opaque provider state retained through an [`AgentContext`](crate::AgentContext).

use crate::{AssistantMessage, Message, UserMessage};

/// One user-initiated turn: user message plus assistant replies and tool traffic until the next user message.
#[derive(Debug, Clone, Copy)]
pub struct Turn<'a> {
    messages: &'a [Message],
    index: usize,
    start: usize,
}

impl<'a> Turn<'a> {
    /// Zero-based turn index in the transcript.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Message index where this turn starts in the parent transcript.
    pub fn start_index(&self) -> usize {
        self.start
    }

    /// All messages belonging to this turn (inclusive user through following assistant/tool messages).
    pub fn messages(&self) -> &'a [Message] {
        self.messages
    }

    /// The user message that started this turn.
    pub fn user(&self) -> Option<&UserMessage> {
        match self.messages.first()? {
            Message::User(user) => Some(user),
            _ => None,
        }
    }

    /// Whether this turn contains tool calls or tool results.
    pub fn uses_tools(&self) -> bool {
        self.messages.iter().any(|msg| match msg {
            Message::Assistant(assistant) => !assistant.tool_calls().is_empty(),
            Message::ToolResult(_) => true,
            _ => false,
        })
    }

    /// Last assistant message in this turn (if any).
    pub fn final_assistant(&self) -> Option<&AssistantMessage> {
        self.messages.iter().rev().find_map(|msg| match msg {
            Message::Assistant(assistant) => Some(assistant),
            _ => None,
        })
    }

    /// Text content of the last non-empty assistant reply in this turn.
    pub fn final_assistant_text(&self) -> Option<&str> {
        self.messages.iter().rev().find_map(|msg| match msg {
            Message::Assistant(assistant) => {
                let content = assistant.content();
                (!content.is_empty()).then_some(content)
            }
            _ => None,
        })
    }
}

/// Iterator over [`Turn`] views of a message transcript.
#[derive(Debug, Clone)]
pub struct Turns<'a> {
    messages: &'a [Message],
    starts: Vec<usize>,
}

impl<'a> Turns<'a> {
    pub fn new(messages: &'a [Message]) -> Self {
        Self {
            messages,
            starts: turn_start_indices(messages),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }

    pub fn count(&self) -> usize {
        self.starts.len()
    }

    pub fn get(&self, index: usize) -> Option<Turn<'a>> {
        let start = *self.starts.get(index)?;
        let end = self
            .starts
            .get(index + 1)
            .copied()
            .unwrap_or(self.messages.len());
        Some(Turn {
            messages: &self.messages[start..end],
            index,
            start,
        })
    }

    pub fn last(&self) -> Option<Turn<'a>> {
        (!self.starts.is_empty())
            .then(|| self.get(self.starts.len() - 1))
            .flatten()
    }

    pub fn iter(&self) -> TurnIter<'a> {
        TurnIter {
            turns: self.clone(),
            next_index: 0,
        }
    }
}

impl<'a> IntoIterator for Turns<'a> {
    type Item = Turn<'a>;
    type IntoIter = TurnIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        TurnIter {
            turns: self,
            next_index: 0,
        }
    }
}

/// Borrowing iterator over turns in a transcript.
#[derive(Debug, Clone)]
pub struct TurnIter<'a> {
    turns: Turns<'a>,
    next_index: usize,
}

impl<'a> Iterator for TurnIter<'a> {
    type Item = Turn<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.turns.starts.len() {
            return None;
        }
        let turn = self.turns.get(self.next_index)?;
        self.next_index += 1;
        Some(turn)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.turns.starts.len().saturating_sub(self.next_index);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TurnIter<'a> {}

/// Build turn views over a message slice.
pub fn turns(messages: &[Message]) -> Turns<'_> {
    Turns::new(messages)
}

fn turn_start_indices(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| match message {
            // Steered messages are part of the current turn, not a new one.
            Message::User(user) if !user.steered() => Some(index),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssistantMessage, UserMessage};
    use serde_json::json;

    fn sample_with_tools() -> Vec<Message> {
        vec![
            Message::User(UserMessage::new("a")),
            Message::Assistant(
                AssistantMessage::new("").with_tool_calls(vec![crate::ToolCall::new(
                    "c1",
                    "echo",
                    json!({}),
                )]),
            ),
            Message::ToolResult(crate::ToolResult::new(
                "c1",
                crate::ToolResultOutcome::Success(json!("ok").into()),
            )),
            Message::Assistant(AssistantMessage::new("done")),
            Message::User(UserMessage::new("b")),
        ]
    }

    #[test]
    fn turns_iterates_user_boundaries() {
        let messages = vec![
            Message::User(UserMessage::new("a")),
            Message::Assistant(AssistantMessage::new("1")),
            Message::User(UserMessage::new("b")),
        ];
        let collected: Vec<_> = turns(&messages).into_iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].start_index(), 0);
        assert_eq!(collected[0].messages().len(), 2);
        assert_eq!(collected[1].start_index(), 2);
        assert_eq!(collected[1].messages().len(), 1);
    }

    #[test]
    fn turn_helpers_match_segment_semantics() {
        let messages = sample_with_tools();
        let first = turns(&messages).get(0).unwrap();
        assert!(first.uses_tools());
        assert_eq!(first.final_assistant_text(), Some("done"));
        assert_eq!(first.final_assistant().map(|a| a.content()), Some("done"));
    }

    #[test]
    fn steered_messages_do_not_start_new_turns() {
        // A steered user message injected mid-turn stays in the same turn.
        let messages = vec![
            Message::User(UserMessage::new("go")),
            Message::Assistant(AssistantMessage::new("")),
            Message::User(UserMessage::new("wait, also check tests").with_steered(true)),
            Message::Assistant(AssistantMessage::new("done")),
        ];

        let collected: Vec<_> = turns(&messages).into_iter().collect();
        assert_eq!(
            collected.len(),
            1,
            "steered message should not split the turn"
        );
        assert_eq!(collected[0].messages().len(), 4);
        assert_eq!(collected[0].start_index(), 0);
    }
}
