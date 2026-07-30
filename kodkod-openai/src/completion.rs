use kodkod_core::{AssistantMessage, Conversation, ToolSpec};

use super::api::{ApiErrorResponse, ChatCompletionResponse};
use super::convert::{build_request, parse_assistant_message};
use super::error::OpenAiError;

pub fn chat_completions_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    format!("{base}/chat/completions")
}

/// Post a chat completion request to an OpenAI-compatible endpoint.
pub async fn complete(
    client: &reqwest::Client,
    chat_completions_url: &str,
    bearer: Option<&str>,
    model_id: &str,
    conversation: &Conversation,
    tools: &[ToolSpec],
) -> Result<AssistantMessage, OpenAiError> {
    let request = build_request(model_id, conversation, tools);
    let mut http_request = client.post(chat_completions_url).json(&request);

    if let Some(bearer) = bearer {
        http_request = http_request.bearer_auth(bearer);
    }

    let response = http_request.send().await?;
    let status = response.status();

    if !status.is_success() {
        let message = match response.json::<ApiErrorResponse>().await {
            Ok(body) => body.error.message,
            Err(_) => status.to_string(),
        };
        return Err(OpenAiError::Api {
            status: status.as_u16(),
            message,
        });
    }

    let body = response.json::<ChatCompletionResponse>().await?;
    parse_assistant_message(body)
}
