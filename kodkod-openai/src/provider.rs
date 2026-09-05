use std::marker::PhantomData;
use std::sync::Arc;

use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, Conversation, Provider, ProviderEvent, ProviderStream, ToolSpec,
};

use super::completion;
use super::error::OpenAiError;
use super::model::OpenAiModel;
use crate::{CredentialSource, StaticCredentials};

/// Provider for OpenAI-compatible `/chat/completions` endpoints.
///
/// `M` is a zero-sized type marker tying this provider to your [`OpenAiModel`]
/// implementation (e.g. `OpenAiCompatibleProvider::<MyModel>::new(url)`).
///
/// Use [`Self::with_credentials`] when the caller needs to supply fresh
/// authentication headers for every request.
#[derive(Clone)]
pub struct OpenAiCompatibleProvider<M = ()> {
    chat_completions_url: String,
    credentials: Option<Arc<dyn CredentialSource>>,
    client: reqwest::Client,
    _model: PhantomData<M>,
}

impl<M> OpenAiCompatibleProvider<M> {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            chat_completions_url: completion::chat_completions_url(&base_url.into()),
            credentials: None,
            client: reqwest::Client::new(),
            _model: PhantomData,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.credentials = Some(Arc::new(StaticCredentials::bearer(api_key.into())));
        self
    }

    pub fn with_credentials(mut self, credentials: Arc<dyn CredentialSource>) -> Self {
        self.credentials = Some(credentials);
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
    type Continuation = ();

    fn create_continuation(&self, _model: &Self::Model) -> Self::Continuation {}

    fn supports_vision(&self, model: &M) -> bool {
        model.supports_vision()
    }

    async fn complete(
        &self,
        _continuation: &Self::Continuation,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, Self::Continuation), Self::Error> {
        let credentials = match &self.credentials {
            Some(source) => Some(source.credentials().await?),
            None => None,
        };
        completion::complete_with_credentials(
            &self.client,
            &self.chat_completions_url,
            credentials.as_ref(),
            model.id(),
            conversation,
            tools,
        )
        .await
        .map(|message| (message, ()))
    }

    fn complete_stream<'a>(
        &'a self,
        _continuation: &'a Self::Continuation,
        model: &'a M,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, Self::Continuation, Self::Error> {
        Box::pin(async_stream::try_stream! {
            let credentials = match &self.credentials {
                Some(source) => Some(source.credentials().await?),
                None => None,
            };
            let mut stream = completion::stream_with_credentials(
                &self.client,
                &self.chat_completions_url,
                credentials.as_ref(),
                model.id(),
                conversation,
                tools,
            );
            while let Some(event) = stream.next().await {
                match event? {
                    ProviderEvent::TextDelta(delta) => yield ProviderEvent::TextDelta(delta),
                    ProviderEvent::Completed(message, ()) => {
                        yield ProviderEvent::Completed(message, ());
                        return;
                    }
                }
            }
            Err(OpenAiError::Protocol("chat completion stream ended without completion".into()))?;
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use kodkod_core::{ProviderEvent, UserMessage};
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

    #[tokio::test]
    async fn streams_chat_text_and_requires_done_before_completion() {
        let server = MockServer::start().await;
        let body = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        let provider = OpenAiCompatibleProvider::<TestModel>::new(format!("{}/v1", server.uri()));
        let model = TestModel {
            id: "llama3",
            vision: false,
        };
        let conversation = Conversation::new();
        let mut stream = provider.complete_stream(&(), &model, &conversation, &[]);
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::TextDelta(text) if text == "hel")
        );
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::TextDelta(text) if text == "lo")
        );
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::Completed(message, ()) if message.content() == "hello")
        );
    }
}
