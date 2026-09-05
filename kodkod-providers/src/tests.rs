use super::*;
use futures_util::StreamExt;
use kodkod_core::{Conversation, Provider, ProviderEvent, UserMessage};
use kodkod_http::{RequestCredentials, StaticCredentials};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::json;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

struct FreshCodexAccess(AtomicUsize);

struct TestCodexModel;

impl kodkod_openai::OpenAiModel for TestCodexModel {
    fn id(&self) -> &str {
        "gpt-test"
    }

    fn supports_vision(&self) -> bool {
        true
    }
}

impl CodexCredentialSource for FreshCodexAccess {
    fn access(&self) -> CodexAccessFuture<'_> {
        Box::pin(async {
            let request = self.0.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(CodexAccess {
                access_token: format!("token-{request}"),
                account_id: "account-1".into(),
            })
        })
    }
}

fn completed(text: &str) -> String {
    format!(
        "event: response.completed\ndata: {}\n\n",
        json!({
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}]
                }]
            }
        })
    )
}

#[tokio::test]
async fn codex_adds_fresh_account_credentials_and_originator() {
    let server = MockServer::start().await;
    for request in 1..=2 {
        Mock::given(method("POST"))
            .and(path("/responses"))
            .and(header("authorization", format!("Bearer token-{request}")))
            .and(header("chatgpt-account-id", "account-1"))
            .and(header("originator", "lynx"))
            .respond_with(ResponseTemplate::new(200).set_body_string(completed("done")))
            .expect(1)
            .mount(&server)
            .await;
    }
    let provider =
        CodexProvider::<TestCodexModel>::new(Arc::new(FreshCodexAccess(AtomicUsize::new(0))))
            .unwrap()
            .with_originator("lynx")
            .unwrap()
            .with_test_endpoint(server.uri());
    let conversation = Conversation::new();
    for _ in 0..2 {
        provider
            .complete_once(&TestCodexModel, &conversation, &[])
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn codex_forwards_responses_text_deltas() {
    let server = MockServer::start().await;
    let body = format!(
        "event: response.output_text.delta\ndata: {}\n\n{}",
        json!({"type":"response.output_text.delta","delta":"live"}),
        completed("final")
    );
    Mock::given(method("POST"))
        .and(path("/responses"))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .mount(&server)
        .await;
    let provider =
        CodexProvider::<TestCodexModel>::new(Arc::new(FreshCodexAccess(AtomicUsize::new(0))))
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = TestCodexModel;
    let conversation = Conversation::new();
    let continuation = provider.create_continuation(&model);
    let mut stream = provider.complete_stream(&continuation, &model, &conversation, &[]);
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::TextDelta(text) if text == "live")
    );
    assert!(
        matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::Completed(message, _) if message.content() == "final")
    );
}

#[test]
fn catalog_routes_models_by_service_and_protocol() {
    assert_eq!(
        open_code_model(OpenCodeService::Go, "gpt-5.6-luna")
            .unwrap()
            .protocol(),
        Protocol::Responses
    );
    assert_eq!(
        open_code_model(OpenCodeService::Go, "minimax-m2.7")
            .unwrap()
            .protocol(),
        Protocol::Messages
    );
    assert_eq!(
        open_code_model(OpenCodeService::Zen, "minimax-m2.7")
            .unwrap()
            .protocol(),
        Protocol::ChatCompletions
    );
    let mut seen = HashSet::new();
    for model in open_code_catalog() {
        assert!(seen.insert((model.service(), model.id())));
    }
}

#[tokio::test]
async fn open_code_routes_chat_and_owns_required_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer secret"))
        .and(header("x-opencode-session", "session-1"))
        .and(header("user-agent", "Lynx/1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"content": "world", "tool_calls": []}}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut supplied = HeaderMap::new();
    supplied.insert("authorization", HeaderValue::from_static("Bearer secret"));
    supplied.insert(
        "x-opencode-session",
        HeaderValue::from_static("wrong-session"),
    );
    supplied.insert("user-agent", HeaderValue::from_static("wrong-agent"));
    let provider = OpenCodeProvider::with_credentials(
        OpenCodeService::Zen,
        Arc::new(StaticCredentials::new(RequestCredentials::from_headers(
            supplied,
        ))),
        "session-1",
        "Lynx/1",
    )
    .unwrap()
    .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Zen, "minimax-m2.7").unwrap();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("hello"));
    let message = provider
        .complete_once(model, &conversation, &[])
        .await
        .unwrap();
    assert_eq!(message.content(), "world");
}

#[tokio::test]
async fn open_code_routes_responses_with_bearer_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer secret"))
        .and(header("x-opencode-session", "session-2"))
        .and(header("user-agent", "Lynx/1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(completed("response")))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session-2", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Go, "gpt-5.6-luna").unwrap();
    let message = provider
        .complete_once(model, &Conversation::new(), &[])
        .await
        .unwrap();
    assert_eq!(message.content(), "response");
}

#[tokio::test]
async fn open_code_routes_messages_with_native_api_key_header() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "secret"))
        .and(header("x-opencode-session", "session-3"))
        .and(header("user-agent", "Lynx/1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "message"}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session-3", "Lynx/1")
            .unwrap()
            .with_test_endpoint(server.uri());
    let model = open_code_model(OpenCodeService::Go, "minimax-m2.7").unwrap();
    let message = provider
        .complete_once(model, &Conversation::new(), &[])
        .await
        .unwrap();
    assert_eq!(message.content(), "message");
}

#[tokio::test]
async fn open_code_rejects_a_model_from_the_other_service_before_http() {
    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session-1", "Lynx/1")
            .unwrap();
    let model = open_code_model(OpenCodeService::Zen, "minimax-m2.7").unwrap();
    let error = provider
        .complete_once(model, &Conversation::new(), &[])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("belongs to OpenCode"));
}

#[tokio::test]
async fn named_provider_does_not_follow_redirects() {
    let source = MockServer::start().await;
    let destination = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(302).insert_header(
            "location",
            format!("{}/chat/completions", destination.uri()),
        ))
        .expect(1)
        .mount(&source)
        .await;
    Mock::given(method("GET"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200))
        .expect(0)
        .mount(&destination)
        .await;

    let provider =
        OpenCodeProvider::with_api_key(OpenCodeService::Zen, "secret", "session", "Lynx/1")
            .unwrap()
            .with_test_endpoint(source.uri());
    let model = open_code_model(OpenCodeService::Zen, "minimax-m2.7").unwrap();
    let error = provider
        .complete_once(model, &Conversation::new(), &[])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("302"));
}

#[test]
fn named_endpoints_and_protocol_serialization_are_stable() {
    assert_eq!(
        CodexProvider::<TestCodexModel>::new(Arc::new(FreshCodexAccess(AtomicUsize::new(0))))
            .unwrap()
            .endpoint(),
        CODEX_ENDPOINT
    );
    assert_eq!(OpenCodeService::Go.endpoint(), OPEN_CODE_GO_ENDPOINT);
    assert_eq!(OpenCodeService::Zen.endpoint(), OPEN_CODE_ZEN_ENDPOINT);
    assert_eq!(
        serde_json::to_string(&Protocol::ChatCompletions).unwrap(),
        "\"chat_completions\""
    );
}

#[test]
fn required_header_values_reject_empty_or_invalid_input() {
    assert!(OpenCodeProvider::with_api_key(OpenCodeService::Go, " ", "session", "Lynx/1").is_err());
    assert!(OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", " ", "Lynx/1").is_err());
    assert!(OpenCodeProvider::with_api_key(OpenCodeService::Go, "secret", "session", " ").is_err());
    assert!(
        OpenCodeProvider::with_api_key(
            OpenCodeService::Go,
            "secret\nsecond-header: value",
            "session",
            "Lynx/1",
        )
        .is_err()
    );
}
