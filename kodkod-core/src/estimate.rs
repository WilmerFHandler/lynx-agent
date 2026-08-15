use crate::{Conversation, Message, ToolResultOutcome, ToolSpec};

const IMAGE_TOKENS: u64 = 2_000;
pub(crate) const BASE_OVERHEAD_TOKENS: u64 = 256;
const MESSAGE_OVERHEAD_TOKENS: u64 = 16;

fn text_tokens(bytes: usize) -> u64 {
    (bytes as u64).div_ceil(3)
}

fn json_tokens(value: &impl serde::Serialize) -> u64 {
    text_tokens(serde_json::to_string(value).unwrap_or_default().len())
}

pub(crate) fn estimate_message(message: &Message) -> u64 {
    let mut tokens = MESSAGE_OVERHEAD_TOKENS;
    match message {
        Message::System(message) => tokens += text_tokens(message.content().len()),
        Message::User(message) => {
            tokens += text_tokens(message.content().len());
            tokens += message.images().len() as u64 * IMAGE_TOKENS;
        }
        Message::Assistant(message) => {
            tokens += text_tokens(message.content().len());
            for call in message.tool_calls() {
                tokens += text_tokens(call.id().len() + call.name().len());
                tokens += json_tokens(call.arguments());
            }
        }
        Message::ToolResult(result) => {
            tokens += text_tokens(result.tool_call_id().len());
            match result.outcome() {
                ToolResultOutcome::Success(output) => {
                    tokens += json_tokens(output.value());
                    tokens += output.images().len() as u64 * IMAGE_TOKENS;
                }
                ToolResultOutcome::Error(error) => tokens += text_tokens(error.to_string().len()),
            }
        }
    }
    tokens
}

pub(crate) fn estimate_conversation(conversation: &Conversation, tools: &[ToolSpec]) -> u64 {
    let mut tokens = BASE_OVERHEAD_TOKENS;
    if let Some(system_prompt) = conversation.system_prompt() {
        tokens += MESSAGE_OVERHEAD_TOKENS + text_tokens(system_prompt.len());
    }
    for message in conversation.messages() {
        tokens += estimate_message(message);
    }
    for spec in tools {
        tokens += MESSAGE_OVERHEAD_TOKENS;
        tokens += text_tokens(spec.name().len() + spec.description().len());
        tokens += json_tokens(spec.input_schema());
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AssistantMessage, Image, SystemMessage, ToolCall, ToolError, ToolExecutorError, ToolResult,
        UserMessage,
    };
    use serde_json::json;

    #[test]
    fn empty_conversation_is_base_overhead() {
        assert_eq!(
            Conversation::new().estimate_tokens(&[]),
            BASE_OVERHEAD_TOKENS
        );
    }

    #[test]
    fn system_prompt_increases_estimate() {
        let without = Conversation::new();
        let with = Conversation::new().with_system_prompt("a conversation-level system prompt");
        assert!(with.estimate_tokens(&[]) > without.estimate_tokens(&[]));
    }

    #[test]
    fn user_text_increases_estimate() {
        let mut conversation = Conversation::new();
        let initial = conversation.estimate_tokens(&[]);
        conversation.push_message(Message::User(UserMessage::new("a fairly long request")));
        assert!(conversation.estimate_tokens(&[]) > initial);
    }

    #[test]
    fn images_increase_estimate() {
        let mut text_only = Conversation::new();
        text_only.push_message(Message::User(UserMessage::new("describe")));
        let mut with_image = Conversation::new();
        with_image.push_message(Message::User(
            UserMessage::new("describe")
                .with_images(vec![Image::new("image/png", vec![0x89, 0x50])]),
        ));
        assert!(with_image.estimate_tokens(&[]) > text_only.estimate_tokens(&[]));
    }

    #[test]
    fn assistant_tool_call_args_increase_estimate() {
        let mut without_args = Conversation::new();
        without_args.push_message(Message::Assistant(
            AssistantMessage::new("").with_tool_calls(vec![ToolCall::new("c1", "read", json!({}))]),
        ));
        let mut with_args = Conversation::new();
        with_args.push_message(Message::Assistant(
            AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                "c1",
                "read",
                json!({"path": "a/very/long/path/to/a/file.rs"}),
            )]),
        ));
        assert!(with_args.estimate_tokens(&[]) > without_args.estimate_tokens(&[]));
    }

    #[test]
    fn tool_specs_increase_estimate() {
        let conversation = Conversation::new();
        let spec = ToolSpec::new(
            "echo",
            "echoes input back",
            json!({"type": "object", "properties": {"text": {"type": "string"}}}),
        );
        assert!(conversation.estimate_tokens(&[spec]) > conversation.estimate_tokens(&[]));
    }

    #[test]
    fn tool_results_contribute() {
        let mut success = Conversation::new();
        success.push_message(Message::ToolResult(ToolResult::success(
            "c1",
            json!({"body": "a fairly long successful payload"}),
        )));
        let mut error = Conversation::new();
        error.push_message(Message::ToolResult(ToolResult::failure(
            "c1",
            ToolExecutorError::Tool(ToolError::new("a fairly long tool failure")),
        )));
        assert!(success.estimate_tokens(&[]) > BASE_OVERHEAD_TOKENS);
        assert!(error.estimate_tokens(&[]) > BASE_OVERHEAD_TOKENS);
    }

    #[test]
    fn system_messages_in_the_list_are_counted() {
        let mut conversation = Conversation::new();
        let initial = conversation.estimate_tokens(&[]);
        conversation.push_message(Message::System(SystemMessage::new("inline system")));
        assert!(conversation.estimate_tokens(&[]) > initial);
    }
}
