use std::error::Error;
use std::fmt;
use std::sync::{Arc, OnceLock};

use futures_util::StreamExt;
use kodkod_anthropic::{
    AnthropicContinuation, AnthropicError, AnthropicMessagesProvider, AnthropicModel,
};
use kodkod_core::{
    AssistantMessage, Conversation, Provider, ProviderEvent, ProviderStream, Retryable, ToolSpec,
};
use kodkod_http::{CredentialSource, RequestCredentials, StaticCredentials};
use kodkod_openai::{
    OpenAiCompatibleProvider, OpenAiError, OpenAiModel, OpenAiResponsesProvider,
    ResponsesContinuation,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, USER_AGENT};
use serde::{Deserialize, Serialize};

use crate::{Protocol, ProviderConfigError, fixed_endpoint_client, header_value};
pub const OPEN_CODE_GO_ENDPOINT: &str = "https://opencode.ai/zen/go/v1";
pub const OPEN_CODE_ZEN_ENDPOINT: &str = "https://opencode.ai/zen/v1";

/// An OpenCode service with a fixed base endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenCodeService {
    Go,
    Zen,
}

impl OpenCodeService {
    pub const fn endpoint(self) -> &'static str {
        match self {
            Self::Go => OPEN_CODE_GO_ENDPOINT,
            Self::Zen => OPEN_CODE_ZEN_ENDPOINT,
        }
    }
}

/// Reviewed metadata for one OpenCode model and its official endpoint protocol.
#[derive(Clone, Debug)]
pub struct OpenCodeModel {
    service: OpenCodeService,
    id: String,
    name: String,
    protocol: Protocol,
    vision: bool,
}

impl OpenCodeModel {
    pub const fn service(&self) -> OpenCodeService {
        self.service
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn protocol(&self) -> Protocol {
        self.protocol
    }

    pub const fn supports_vision(&self) -> bool {
        self.vision
    }
}

impl OpenAiModel for OpenCodeModel {
    fn id(&self) -> &str {
        self.id()
    }

    fn supports_vision(&self) -> bool {
        self.supports_vision()
    }
}

impl AnthropicModel for OpenCodeModel {
    fn id(&self) -> &str {
        self.id()
    }

    fn supports_vision(&self) -> bool {
        self.supports_vision()
    }
}

/// The reviewed OpenCode catalog bundled with this crate.
pub fn open_code_catalog() -> &'static [OpenCodeModel] {
    #[derive(Deserialize)]
    struct CatalogEntry {
        service: OpenCodeService,
        id: String,
        name: String,
        protocol: Protocol,
        vision: bool,
    }

    static MODELS: OnceLock<Vec<OpenCodeModel>> = OnceLock::new();
    MODELS.get_or_init(|| {
        serde_json::from_str::<Vec<CatalogEntry>>(include_str!("open-code-models.json"))
            .expect("bundled OpenCode model catalog must be valid")
            .into_iter()
            .map(|entry| OpenCodeModel {
                service: entry.service,
                id: entry.id,
                name: entry.name,
                protocol: entry.protocol,
                vision: entry.vision,
            })
            .collect()
    })
}

pub fn open_code_model(service: OpenCodeService, id: &str) -> Option<&'static OpenCodeModel> {
    open_code_catalog()
        .iter()
        .find(|model| model.service == service && model.id == id)
}

#[derive(Clone)]
enum OpenCodeAuth {
    ApiKey {
        bearer: RequestCredentials,
        messages: RequestCredentials,
    },
    Credentials(Arc<dyn CredentialSource>),
}

/// OpenCode provider that fixes the service endpoint and routes catalog models.
#[derive(Clone)]
pub struct OpenCodeProvider {
    service: OpenCodeService,
    auth: OpenCodeAuth,
    session: HeaderValue,
    user_agent: HeaderValue,
    client: reqwest::Client,
    endpoint: String,
}

impl OpenCodeProvider {
    pub fn with_api_key(
        service: OpenCodeService,
        api_key: impl Into<String>,
        session: impl fmt::Display,
        user_agent: impl AsRef<str>,
    ) -> Result<Self, ProviderConfigError> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(ProviderConfigError::new("OpenCode API key is empty"));
        }
        let mut message_headers = HeaderMap::new();
        message_headers.insert(
            HeaderName::from_static("x-api-key"),
            header_value("OpenCode API key", &api_key)?,
        );
        let bearer = RequestCredentials::bearer(&api_key)
            .map_err(|error| ProviderConfigError::new(error.to_string()))?;
        Self::new(
            service,
            OpenCodeAuth::ApiKey {
                bearer,
                messages: RequestCredentials::from_headers(message_headers),
            },
            session,
            user_agent,
        )
    }

    pub fn with_credentials(
        service: OpenCodeService,
        credentials: Arc<dyn CredentialSource>,
        session: impl fmt::Display,
        user_agent: impl AsRef<str>,
    ) -> Result<Self, ProviderConfigError> {
        Self::new(
            service,
            OpenCodeAuth::Credentials(credentials),
            session,
            user_agent,
        )
    }

    fn new(
        service: OpenCodeService,
        auth: OpenCodeAuth,
        session: impl fmt::Display,
        user_agent: impl AsRef<str>,
    ) -> Result<Self, ProviderConfigError> {
        let session = session.to_string();
        if session.trim().is_empty() {
            return Err(ProviderConfigError::new("OpenCode session is empty"));
        }
        if user_agent.as_ref().trim().is_empty() {
            return Err(ProviderConfigError::new("user agent is empty"));
        }
        Ok(Self {
            service,
            auth,
            session: header_value("OpenCode session", &session)?,
            user_agent: header_value("user agent", user_agent.as_ref())?,
            client: fixed_endpoint_client()?,
            endpoint: service.endpoint().to_owned(),
        })
    }

    /// Use a caller-configured HTTP client without changing the fixed endpoint.
    ///
    /// The caller remains responsible for client-wide proxy, redirect, timeout,
    /// and default-header policy. Provider-owned headers are applied per request.
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub const fn service(&self) -> OpenCodeService {
        self.service
    }

    pub const fn endpoint(&self) -> &'static str {
        self.service.endpoint()
    }

    fn credentials(&self, protocol: Protocol) -> Arc<dyn CredentialSource> {
        let auth: Arc<dyn CredentialSource> = match &self.auth {
            OpenCodeAuth::ApiKey { messages, .. } if protocol == Protocol::Messages => {
                Arc::new(StaticCredentials::new(messages.clone()))
            }
            OpenCodeAuth::ApiKey { bearer, .. } => Arc::new(StaticCredentials::new(bearer.clone())),
            OpenCodeAuth::Credentials(credentials) => credentials.clone(),
        };
        Arc::new(OpenCodeHttpCredentials {
            auth,
            session: self.session.clone(),
            user_agent: self.user_agent.clone(),
        })
    }

    fn validate_model(&self, model: &OpenCodeModel) -> Result<(), ProviderConfigError> {
        if model.service != self.service {
            return Err(ProviderConfigError::new(format!(
                "model {} belongs to OpenCode {:?}, not {:?}",
                model.id, model.service, self.service
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn with_test_endpoint(mut self, endpoint: String) -> Self {
        self.endpoint = endpoint;
        self
    }
}

struct OpenCodeHttpCredentials {
    auth: Arc<dyn CredentialSource>,
    session: HeaderValue,
    user_agent: HeaderValue,
}

impl CredentialSource for OpenCodeHttpCredentials {
    fn credentials(&self) -> kodkod_http::CredentialFuture<'_> {
        Box::pin(async {
            let credentials = self.auth.credentials().await?;
            let mut headers = credentials.headers().clone();
            headers.insert(
                HeaderName::from_static("x-opencode-session"),
                self.session.clone(),
            );
            headers.insert(USER_AGENT, self.user_agent.clone());
            Ok(RequestCredentials::from_headers(headers))
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OpenCodeContinuation {
    Responses(ResponsesContinuation),
    ChatCompletions,
    Messages(AnthropicContinuation),
}

#[derive(Debug)]
pub enum ProviderError {
    Configuration(ProviderConfigError),
    OpenAi(OpenAiError),
    Anthropic(AnthropicError),
}

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Configuration(error) => error.fmt(f),
            Self::OpenAi(error) => error.fmt(f),
            Self::Anthropic(error) => error.fmt(f),
        }
    }
}

impl Error for ProviderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Configuration(error) => Some(error),
            Self::OpenAi(error) => Some(error),
            Self::Anthropic(error) => Some(error),
        }
    }
}

impl Retryable for ProviderError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Configuration(_) => false,
            Self::OpenAi(error) => error.is_retryable(),
            Self::Anthropic(error) => error.is_retryable(),
        }
    }
}

impl From<ProviderConfigError> for ProviderError {
    fn from(error: ProviderConfigError) -> Self {
        Self::Configuration(error)
    }
}

impl From<OpenAiError> for ProviderError {
    fn from(error: OpenAiError) -> Self {
        Self::OpenAi(error)
    }
}

impl From<AnthropicError> for ProviderError {
    fn from(error: AnthropicError) -> Self {
        Self::Anthropic(error)
    }
}

impl Provider for OpenCodeProvider {
    type Model = OpenCodeModel;
    type Error = ProviderError;
    type Continuation = OpenCodeContinuation;

    fn supports_vision(&self, model: &OpenCodeModel) -> bool {
        model.supports_vision()
    }

    fn create_continuation(&self, model: &OpenCodeModel) -> Self::Continuation {
        match model.protocol {
            Protocol::Responses => OpenCodeContinuation::Responses(Default::default()),
            Protocol::ChatCompletions => OpenCodeContinuation::ChatCompletions,
            Protocol::Messages => OpenCodeContinuation::Messages(Default::default()),
        }
    }

    fn estimate_continuation_tokens(continuation: &Self::Continuation) -> u64 {
        match continuation {
            OpenCodeContinuation::Responses(continuation) => {
                <OpenAiResponsesProvider<OpenCodeModel> as Provider>::estimate_continuation_tokens(
                    continuation,
                )
            }
            OpenCodeContinuation::ChatCompletions => 0,
            OpenCodeContinuation::Messages(continuation) => {
                <AnthropicMessagesProvider<OpenCodeModel> as Provider>::estimate_continuation_tokens(
                    continuation,
                )
            }
        }
    }

    async fn complete(
        &self,
        continuation: &Self::Continuation,
        model: &OpenCodeModel,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, Self::Continuation), Self::Error> {
        self.validate_model(model)?;
        match (model.protocol, continuation) {
            (Protocol::Responses, OpenCodeContinuation::Responses(continuation)) => {
                let provider = OpenAiResponsesProvider::new(&self.endpoint)
                    .with_credentials(self.credentials(Protocol::Responses))
                    .with_client(self.client.clone());
                let (message, next) = provider
                    .complete(continuation, model, conversation, tools)
                    .await?;
                Ok((message, OpenCodeContinuation::Responses(next)))
            }
            (Protocol::ChatCompletions, OpenCodeContinuation::ChatCompletions) => {
                let provider = OpenAiCompatibleProvider::new(&self.endpoint)
                    .with_credentials(self.credentials(Protocol::ChatCompletions))
                    .with_client(self.client.clone());
                let (message, ()) = provider.complete(&(), model, conversation, tools).await?;
                Ok((message, OpenCodeContinuation::ChatCompletions))
            }
            (Protocol::Messages, OpenCodeContinuation::Messages(continuation)) => {
                let provider = AnthropicMessagesProvider::new(&self.endpoint)
                    .with_credentials(self.credentials(Protocol::Messages))
                    .with_client(self.client.clone());
                let (message, next) = provider
                    .complete(continuation, model, conversation, tools)
                    .await?;
                Ok((message, OpenCodeContinuation::Messages(next)))
            }
            _ => Err(ProviderConfigError::new(
                "OpenCode continuation protocol does not match the selected model",
            )
            .into()),
        }
    }

    fn complete_stream<'a>(
        &'a self,
        continuation: &'a Self::Continuation,
        model: &'a OpenCodeModel,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, Self::Continuation, Self::Error> {
        Box::pin(async_stream::try_stream! {
            self.validate_model(model)?;
            match (model.protocol, continuation) {
                (Protocol::Responses, OpenCodeContinuation::Responses(continuation)) => {
                    let provider = OpenAiResponsesProvider::new(&self.endpoint)
                        .with_credentials(self.credentials(Protocol::Responses))
                        .with_client(self.client.clone());
                    let mut stream = provider.complete_stream(continuation, model, conversation, tools);
                    while let Some(event) = stream.next().await {
                        match event? {
                            ProviderEvent::TextDelta(delta) => yield ProviderEvent::TextDelta(delta),
                            ProviderEvent::Completed(message, next) => {
                                yield ProviderEvent::Completed(message, OpenCodeContinuation::Responses(next));
                                return;
                            }
                        }
                    }
                }
                (Protocol::ChatCompletions, OpenCodeContinuation::ChatCompletions) => {
                    let provider = OpenAiCompatibleProvider::new(&self.endpoint)
                        .with_credentials(self.credentials(Protocol::ChatCompletions))
                        .with_client(self.client.clone());
                    let mut stream = provider.complete_stream(&(), model, conversation, tools);
                    while let Some(event) = stream.next().await {
                        match event? {
                            ProviderEvent::TextDelta(delta) => yield ProviderEvent::TextDelta(delta),
                            ProviderEvent::Completed(message, ()) => {
                                yield ProviderEvent::Completed(message, OpenCodeContinuation::ChatCompletions);
                                return;
                            }
                        }
                    }
                }
                (Protocol::Messages, OpenCodeContinuation::Messages(continuation)) => {
                    let provider = AnthropicMessagesProvider::new(&self.endpoint)
                        .with_credentials(self.credentials(Protocol::Messages))
                        .with_client(self.client.clone());
                    let mut stream = provider.complete_stream(continuation, model, conversation, tools);
                    while let Some(event) = stream.next().await {
                        match event? {
                            ProviderEvent::TextDelta(delta) => yield ProviderEvent::TextDelta(delta),
                            ProviderEvent::Completed(message, next) => {
                                yield ProviderEvent::Completed(message, OpenCodeContinuation::Messages(next));
                                return;
                            }
                        }
                    }
                }
                _ => Err(ProviderConfigError::new(
                    "OpenCode continuation protocol does not match the selected model",
                ))?,
            }
        })
    }
}
