use std::collections::BTreeMap;

use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, Conversation, ProviderEvent, ProviderStream, ToolCall, ToolSpec,
};
use serde_json::{Value, json};

use super::api::{ApiErrorResponse, ChatCompletionResponse};
use super::convert::{build_request, parse_assistant_message};
use super::error::OpenAiError;
use crate::RequestCredentials;
use kodkod_http::SseDecoder;

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
    let credentials = match bearer {
        Some(token) => Some(RequestCredentials::bearer(token)?),
        None => None,
    };
    complete_with_credentials(
        client,
        chat_completions_url,
        credentials.as_ref(),
        model_id,
        conversation,
        tools,
    )
    .await
}

/// Post a chat completion with arbitrary caller-supplied authentication headers.
pub async fn complete_with_credentials(
    client: &reqwest::Client,
    chat_completions_url: &str,
    credentials: Option<&RequestCredentials>,
    model_id: &str,
    conversation: &Conversation,
    tools: &[ToolSpec],
) -> Result<AssistantMessage, OpenAiError> {
    complete_with_document_support(
        client,
        chat_completions_url,
        credentials,
        model_id,
        conversation,
        tools,
        false,
    )
    .await
}

pub(crate) async fn complete_with_document_support(
    client: &reqwest::Client,
    chat_completions_url: &str,
    credentials: Option<&RequestCredentials>,
    model_id: &str,
    conversation: &Conversation,
    tools: &[ToolSpec],
    pdf_allowed: bool,
) -> Result<AssistantMessage, OpenAiError> {
    let request = build_request(model_id, conversation, tools, pdf_allowed)?;
    let mut http_request = client.post(chat_completions_url).json(&request);

    if let Some(credentials) = credentials {
        http_request = http_request.headers(credentials.headers().clone());
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

pub(crate) fn stream_with_document_support<'a>(
    client: &'a reqwest::Client,
    chat_completions_url: &'a str,
    credentials: Option<&'a RequestCredentials>,
    model_id: &'a str,
    conversation: &'a Conversation,
    tools: &'a [ToolSpec],
    pdf_allowed: bool,
) -> ProviderStream<'a, (), OpenAiError> {
    Box::pin(async_stream::try_stream! {
        let mut request = serde_json::to_value(build_request(
            model_id,
            conversation,
            tools,
            pdf_allowed,
        )?)?;
        request["stream"] = Value::Bool(true);
        let mut http_request = client.post(chat_completions_url).json(&request);
        if let Some(credentials) = credentials {
            http_request = http_request.headers(credentials.headers().clone());
        }
        let response = http_request.send().await?;
        let status = response.status();
        if !status.is_success() {
            let message = match response.json::<ApiErrorResponse>().await {
                Ok(body) => body.error.message,
                Err(_) => status.to_string(),
            };
            Err(OpenAiError::Api { status: status.as_u16(), message })?;
            unreachable!("error propagation exits the stream");
        }

        if response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.split(';').next() == Some("application/json"))
        {
            let body = response.json::<ChatCompletionResponse>().await?;
            yield ProviderEvent::Completed(parse_assistant_message(body)?, ());
            return;
        }

        let mut decoder = SseDecoder::default();
        let mut accumulator = ChatStreamAccumulator::default();
        let mut bytes = response.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            for event in decoder.push(&chunk?).map_err(OpenAiError::Protocol)? {
                if event.data == "[DONE]" {
                    let message = accumulator.finish()?;
                    yield ProviderEvent::Completed(message, ());
                    return;
                }
                if let Some(delta) = accumulator.push(&event.data)? {
                    yield ProviderEvent::TextDelta(delta);
                }
            }
        }
        decoder.finish().map_err(OpenAiError::Protocol)?;
        Err(OpenAiError::Protocol("chat completion stream ended before [DONE]".into()))?;
    })
}

#[derive(Default)]
struct ChatStreamAccumulator {
    text: String,
    calls: BTreeMap<usize, PartialToolCall>,
    legacy_call: Option<PartialToolCall>,
    finish_reason: Option<String>,
}

#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ChatStreamAccumulator {
    fn push(&mut self, data: &str) -> Result<Option<String>, OpenAiError> {
        let payload: Value = serde_json::from_str(data)?;
        if payload.get("error").is_some() {
            let message = payload
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("unknown streaming error")
                .to_owned();
            return Err(OpenAiError::Protocol(message));
        }
        let choices = payload
            .get("choices")
            .and_then(Value::as_array)
            .ok_or_else(|| OpenAiError::Protocol("streaming chunk missing choices".into()))?;
        if choices.is_empty() {
            return Ok(None);
        }
        if choices.len() != 1 || choices[0].get("index").and_then(Value::as_u64) != Some(0) {
            return Err(OpenAiError::Protocol(
                "only chat completion choice index 0 is supported".into(),
            ));
        }
        let choice = &choices[0];
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            if self
                .finish_reason
                .as_deref()
                .is_some_and(|seen| seen != reason)
            {
                return Err(OpenAiError::Protocol(
                    "chat completion finish reason changed".into(),
                ));
            }
            self.finish_reason = Some(reason.to_owned());
        }
        let Some(delta) = choice.get("delta") else {
            return Ok(None);
        };
        let text = delta
            .get("content")
            .and_then(Value::as_str)
            .or_else(|| delta.get("refusal").and_then(Value::as_str))
            .filter(|text| !text.is_empty())
            .map(str::to_owned);
        if let Some(text) = &text {
            self.text.push_str(text);
        }
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                let index = call
                    .get("index")
                    .and_then(Value::as_u64)
                    .and_then(|index| usize::try_from(index).ok())
                    .ok_or_else(|| {
                        OpenAiError::Protocol("streamed tool call missing index".into())
                    })?;
                let partial = self.calls.entry(index).or_default();
                append_once(
                    &mut partial.id,
                    call.get("id").and_then(Value::as_str),
                    "tool call id",
                )?;
                if let Some(kind) = call.get("type").and_then(Value::as_str)
                    && kind != "function"
                {
                    return Err(OpenAiError::Protocol(format!(
                        "unsupported streamed tool call type: {kind}"
                    )));
                }
                if let Some(function) = call.get("function") {
                    append_once(
                        &mut partial.name,
                        function.get("name").and_then(Value::as_str),
                        "tool call name",
                    )?;
                    if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                        partial.arguments.push_str(arguments);
                    }
                }
            }
        }
        if let Some(function) = delta.get("function_call") {
            let partial = self.legacy_call.get_or_insert_with(|| PartialToolCall {
                id: "legacy_function_call".into(),
                ..Default::default()
            });
            append_once(
                &mut partial.name,
                function.get("name").and_then(Value::as_str),
                "function call name",
            )?;
            if let Some(arguments) = function.get("arguments").and_then(Value::as_str) {
                partial.arguments.push_str(arguments);
            }
        }
        Ok(text)
    }

    fn finish(self) -> Result<AssistantMessage, OpenAiError> {
        let reason = self.finish_reason.ok_or_else(|| {
            OpenAiError::Protocol("chat completion stream missing finish reason".into())
        })?;
        if matches!(reason.as_str(), "length" | "content_filter") {
            return Err(OpenAiError::Incomplete { reason });
        }
        if !matches!(reason.as_str(), "stop" | "tool_calls" | "function_call") {
            return Err(OpenAiError::Protocol(format!(
                "unsupported chat completion finish reason: {reason}"
            )));
        }
        let mut calls = self
            .calls
            .into_iter()
            .enumerate()
            .map(|(expected, (index, call))| {
                if expected != index {
                    return Err(OpenAiError::Protocol(
                        "streamed tool call indices are not contiguous".into(),
                    ));
                }
                finish_tool_call(call)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(call) = self.legacy_call {
            calls.push(finish_tool_call(call)?);
        }
        if matches!(reason.as_str(), "tool_calls" | "function_call") && calls.is_empty() {
            return Err(OpenAiError::Protocol(
                "tool-call finish reason contained no tool calls".into(),
            ));
        }
        if reason == "stop" && !calls.is_empty() {
            return Err(OpenAiError::Protocol(
                "stop finish reason unexpectedly contained tool calls".into(),
            ));
        }
        if self.text.is_empty() && calls.is_empty() {
            return Err(OpenAiError::EmptyResponse);
        }
        Ok(AssistantMessage::new(self.text).with_tool_calls(calls))
    }
}

fn append_once(target: &mut String, value: Option<&str>, field: &str) -> Result<(), OpenAiError> {
    let Some(value) = value else { return Ok(()) };
    if target.is_empty() || target == value {
        if target.is_empty() {
            target.push_str(value);
        }
        Ok(())
    } else {
        Err(OpenAiError::Protocol(format!("streamed {field} changed")))
    }
}

fn finish_tool_call(call: PartialToolCall) -> Result<ToolCall, OpenAiError> {
    if call.id.is_empty() || call.name.is_empty() {
        return Err(OpenAiError::Protocol(
            "streamed tool call missing id or name".into(),
        ));
    }
    let arguments = if call.arguments.is_empty() {
        json!({})
    } else {
        serde_json::from_str(&call.arguments)?
    };
    Ok(ToolCall::new(call.id, call.name, arguments))
}
