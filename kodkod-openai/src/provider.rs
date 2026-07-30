use std::marker::PhantomData;

use kodkod_core::{AssistantMessage, Conversation, Provider, ToolSpec};

use super::completion;
use super::error::OpenAiError;
use super::model::OpenAiModel;

/// Provider for OpenAI-compatible `/chat/completions` endpoints with a static bearer token.
///
/// `M` is a zero-sized type marker tying this provider to your [`OpenAiModel`]
/// implementation (e.g. `OpenAiCompatibleProvider::<MyModel>::new(url)`).
///
/// For per-request bearer tokens (e.g. OAuth), use [`completion::complete`] directly.
#[derive(Clone)]
pub struct OpenAiCompatibleProvider<M = ()> {
    chat_completions_url: String,
    api_key: Option<String>,
    client: reqwest::Client,
    _model: PhantomData<M>,
}

impl<M> OpenAiCompatibleProvider<M> {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            chat_completions_url: completion::chat_completions_url(&base_url.into()),
            api_key: None,
            client: reqwest::Client::new(),
            _model: PhantomData,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub fn chat_completions_url(&self) -> &str {
        &self.chat_completions_url
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }
}

impl<M> Provider for OpenAiCompatibleProvider<M>
where
    M: OpenAiModel,
{
    type Model = M;
    type Error = OpenAiError;
    type TurnState = ();

    fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

    fn supports_vision(&self, model: &M) -> bool {
        model.supports_vision()
    }

    async fn complete_round(
        &self,
        _state: &mut Self::TurnState,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<AssistantMessage, Self::Error> {
        completion::complete(
            &self.client,
            &self.chat_completions_url,
            self.api_key.as_deref(),
            model.id(),
            conversation,
            tools,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kodkod_core::UserMessage;
    use serde_json::json;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct TestModel {
        id: &'static str,
        vision: bool,
    }

    impl OpenAiModel for TestModel {
        fn id(&self) -> &str {
            self.id
        }

        fn supports_vision(&self) -> bool {
            self.vision
        }
    }

    #[tokio::test]
    async fn posts_chat_completion_and_parses_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(body_json(json!({
                "model": "llama3",
                "messages": [{ "role": "user", "content": "hello" }],
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {
                        "content": "world",
                        "tool_calls": []
                    }
                }]
            })))
            .mount(&server)
            .await;

        let provider = OpenAiCompatibleProvider::<TestModel>::new(format!("{}/v1", server.uri()));
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("hello"));

        let model = TestModel {
            id: "llama3",
            vision: false,
        };
        let message = provider
            .complete_once(&model, &conversation, &[])
            .await
            .expect("completion should succeed");

        assert_eq!(message.content(), "world");
        assert!(message.tool_calls().is_empty());
    }
}
