use std::marker::PhantomData;
use std::sync::Arc;

use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, Conversation, Provider, ProviderEvent, ProviderStream, ToolSpec,
};

use super::completion;
use super::convert::validate_documents;
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
    supports_pdf_inputs: bool,
    _model: PhantomData<M>,
}

impl<M> OpenAiCompatibleProvider<M> {
    pub fn new(base_url: impl Into<String>) -> Self {
        let base_url = base_url.into();
        Self {
            chat_completions_url: completion::chat_completions_url(&base_url),
            credentials: None,
            client: reqwest::Client::new(),
            supports_pdf_inputs: is_official_openai_api(&base_url),
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

    /// Enable PDF inputs for a compatible endpoint that accepts Chat Completions file parts.
    pub fn with_pdf_inputs(mut self, supports_pdf_inputs: bool) -> Self {
        self.supports_pdf_inputs = supports_pdf_inputs;
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

    fn supports_document(&self, model: &M, mime: &str) -> bool {
        self.supports_pdf_inputs && model.supports_vision() && mime == "application/pdf"
    }

    async fn complete(
        &self,
        _continuation: &Self::Continuation,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, Self::Continuation), Self::Error> {
        let pdf_allowed = self.supports_document(model, "application/pdf");
        validate_documents(conversation, pdf_allowed)?;
        let credentials = match &self.credentials {
            Some(source) => Some(source.credentials().await?),
            None => None,
        };
        completion::complete_with_document_support(
            &self.client,
            &self.chat_completions_url,
            credentials.as_ref(),
            model.id(),
            conversation,
            tools,
            pdf_allowed,
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
            let pdf_allowed = self.supports_document(model, "application/pdf");
            validate_documents(conversation, pdf_allowed)?;
            let credentials = match &self.credentials {
                Some(source) => Some(source.credentials().await?),
                None => None,
            };
            let mut stream = completion::stream_with_document_support(
                &self.client,
                &self.chat_completions_url,
                credentials.as_ref(),
                model.id(),
                conversation,
                tools,
                pdf_allowed,
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

fn is_official_openai_api(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };

    url.scheme() == "https"
        && url.host_str() == Some("api.openai.com")
        && url.port_or_known_default() == Some(443)
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && matches!(url.path(), "/v1" | "/v1/")
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use kodkod_core::{Document, ProviderEvent, UserMessage};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct TestModel {
        id: &'static str,
        vision: bool,
    }

    struct CountingCredentials(AtomicUsize);

    impl crate::CredentialSource for CountingCredentials {
        fn credentials(&self) -> crate::CredentialFuture<'_> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(crate::RequestCredentials::bearer("secret").unwrap()) })
        }
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

    #[tokio::test]
    async fn rejects_documents_before_chat_completion_requests() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let provider = OpenAiCompatibleProvider::<TestModel>::new(format!("{}/v1", server.uri()));
        let model = TestModel {
            id: "llama3",
            vision: false,
        };
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("read").with_documents(vec![
            Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
        ]));

        let error = provider
            .complete_once(&model, &conversation, &[])
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            OpenAiError::UnsupportedDocument {
                provider: "OpenAI-compatible Chat Completions",
                mime
            } if mime == "application/pdf"
        ));

        let mut stream = provider.complete_stream(&(), &model, &conversation, &[]);
        assert!(matches!(
            stream.next().await.unwrap(),
            Err(OpenAiError::UnsupportedDocument { .. })
        ));
        server.verify().await;
    }

    #[tokio::test]
    async fn rejects_unsupported_documents_before_acquiring_credentials() {
        let credentials = Arc::new(CountingCredentials(AtomicUsize::new(0)));
        let provider = OpenAiCompatibleProvider::<TestModel>::new("http://[::1]:1/v1")
            .with_pdf_inputs(true)
            .with_credentials(credentials.clone());
        let nonvision = TestModel {
            id: "text-model",
            vision: false,
        };
        let vision = TestModel {
            id: "vision-model",
            vision: true,
        };
        let mut pdf = Conversation::new();
        pdf.push_user_message(UserMessage::new("read").with_documents(vec![
            Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
        ]));
        let mut text = Conversation::new();
        text.push_user_message(UserMessage::new("read").with_documents(vec![
            Document::try_new("text/plain", "notes.txt", b"notes").unwrap(),
        ]));

        assert!(matches!(
            provider.complete_once(&nonvision, &pdf, &[]).await,
            Err(OpenAiError::UnsupportedDocument { .. })
        ));
        let mut stream = provider.complete_stream(&(), &vision, &text, &[]);
        assert!(matches!(
            stream.next().await.unwrap(),
            Err(OpenAiError::UnsupportedDocument { .. })
        ));
        assert_eq!(credentials.0.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn posts_pdf_when_the_compatible_endpoint_is_explicitly_enabled() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{ "message": { "content": "done" } }]
            })))
            .mount(&server)
            .await;
        let provider = OpenAiCompatibleProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_pdf_inputs(true);
        let model = TestModel {
            id: "vision-model",
            vision: true,
        };
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("read").with_documents(vec![
            Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
        ]));

        provider
            .complete_once(&model, &conversation, &[])
            .await
            .expect("completion should succeed");
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["messages"][0]["content"][1]["type"], "file");
        assert_eq!(
            body["messages"][0]["content"][1]["file"]["file_data"],
            "data:application/pdf;base64,JVBERg=="
        );
    }

    #[tokio::test]
    async fn streams_pdf_when_the_compatible_endpoint_is_explicitly_enabled() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(concat!(
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                        "data: [DONE]\n\n"
                    )),
            )
            .mount(&server)
            .await;
        let provider = OpenAiCompatibleProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_pdf_inputs(true);
        let model = TestModel {
            id: "vision-model",
            vision: true,
        };
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("read").with_documents(vec![
            Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
        ]));

        let mut stream = provider.complete_stream(&(), &model, &conversation, &[]);
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            ProviderEvent::TextDelta(text) if text == "done"
        ));
        assert!(matches!(
            stream.next().await.unwrap().unwrap(),
            ProviderEvent::Completed(_, ())
        ));
        let requests = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["content"][1]["type"], "file");
    }

    #[test]
    fn enables_pdf_only_for_the_exact_official_openai_api_base_url() {
        for base_url in [
            "https://api.openai.com/v1",
            "https://api.openai.com/v1/",
            "https://api.openai.com:443/v1",
        ] {
            assert!(is_official_openai_api(base_url), "{base_url}");
        }
        for base_url in [
            "http://api.openai.com/v1",
            "https://api.openai.com/v1/other",
            "https://api.openai.com/v1?x=y",
            "https://user@api.openai.com/v1",
            "https://api.openai.com.example/v1",
        ] {
            assert!(!is_official_openai_api(base_url), "{base_url}");
        }
    }
}
