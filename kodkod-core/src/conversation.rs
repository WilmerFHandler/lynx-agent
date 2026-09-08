use serde::{Deserialize, Serialize};

use crate::{Message, ToolSpec, UserMessage};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conversation {
    system_prompt: Option<String>,
    messages: Vec<Message>,
}

impl Conversation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Tokenizer-free estimate of the tokens this conversation would send.
    ///
    /// Pass `&[]` when tools are not part of the request. Continuation tokens
    /// are not included; add `Provider::estimate_continuation_tokens` separately.
    pub fn estimate_tokens(&self, tools: &[ToolSpec]) -> u64 {
        crate::estimate::estimate_conversation(self, tools)
    }

    /// User turns as views over this conversation's messages.
    pub fn turns(&self) -> crate::turns::Turns<'_> {
        crate::turns::turns(self.messages())
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Append a user message verbatim (preserves `steered`, images, documents, etc.).
    pub fn push_user_message(&mut self, user: UserMessage) {
        self.messages.push(Message::User(user));
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn replace_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    /// Return a copy of this conversation with all image attachments removed.
    pub fn without_images(&self) -> Self {
        Self {
            system_prompt: self.system_prompt.clone(),
            messages: self
                .messages
                .iter()
                .map(|message| match message {
                    Message::User(user) => Message::User(
                        UserMessage::new(user.content())
                            .with_images(Vec::new())
                            .with_documents(user.documents().to_vec())
                            .with_steered(user.steered()),
                    ),
                    Message::ToolResult(result) => Message::ToolResult(result.without_images()),
                    _ => message.clone(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, UserMessage};

    #[test]
    fn steered_flag_survives_push_and_image_stripping() {
        let mut conv = Conversation::new();
        conv.push_user_message(UserMessage::new("go"));
        conv.push_message(Message::User(
            UserMessage::new("steer me").with_steered(true),
        ));
        assert_eq!(conv.turns().count(), 1);
        assert!(matches!(
            conv.messages().last(),
            Some(Message::User(user)) if user.steered()
        ));

        let stripped = conv.without_images();
        assert!(matches!(
            stripped.messages().last(),
            Some(Message::User(user)) if user.steered()
        ));
    }

    #[test]
    fn without_images_strips_attachments() {
        use crate::{Document, Image, ToolOutput, ToolResult};

        let mut conv = Conversation::new();
        conv.push_user_message(
            UserMessage::new("describe")
                .with_images(vec![Image::new("image/png", vec![0x89, 0x50])])
                .with_documents(vec![
                    Document::try_new("text/plain", "notes.txt", b"notes").unwrap(),
                ]),
        );
        conv.push_message(Message::ToolResult(ToolResult::success(
            "call_1",
            ToolOutput::new(serde_json::json!({"path": "image.png"}))
                .with_images(vec![Image::new("image/png", vec![0x89, 0x50])]),
        )));
        let stripped = conv.without_images();
        assert!(matches!(
            stripped.messages().first(),
            Some(Message::User(user)) if user.content() == "describe"
                && user.images().is_empty()
                && user.documents().len() == 1
        ));
        assert!(matches!(
            stripped.messages().get(1),
            Some(Message::ToolResult(result))
                if matches!(result.outcome(), crate::ToolResultOutcome::Success(output) if output.images().is_empty())
        ));
    }
}
