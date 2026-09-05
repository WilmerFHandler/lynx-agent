use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, Conversation, Message, Provider, ProviderEvent, ProviderStream, Retryable,
    ToolCall, ToolExecutorError, ToolResult, ToolResultOutcome, ToolSpec,
};
use kodkod_http::SseDecoder;
pub use kodkod_http::{
    CredentialError, CredentialFuture, CredentialSource, RequestCredentials, StaticCredentials,
};
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub trait AnthropicModel: Sync {
    fn id(&self) -> &str;
    fn supports_vision(&self) -> bool;
}

impl AnthropicModel for String {
    fn id(&self) -> &str {
        self
    }
    fn supports_vision(&self) -> bool {
        true
    }
}

impl AnthropicModel for str {
    fn id(&self) -> &str {
        self
    }
    fn supports_vision(&self) -> bool {
        true
    }
}

pub fn messages_url(base_url: &str) -> String {
    format!("{}/messages", base_url.trim_end_matches('/'))
}

#[derive(Clone)]
pub struct AnthropicMessagesProvider<M = ()> {
    messages_url: String,
    credentials: Option<Arc<dyn CredentialSource>>,
    client: reqwest::Client,
    max_tokens: u32,
    _model: PhantomData<M>,
}

/// Opaque assistant content blocks retained for native thinking replay.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct AnthropicContinuation {
    checkpoints: Vec<AnthropicCheckpoint>,
}

#[derive(Clone, Serialize, Deserialize)]
struct AnthropicCheckpoint {
    model: String,
    assistant_message_index: usize,
    assistant: AssistantMessage,
    content: Vec<Value>,
}

impl fmt::Debug for AnthropicContinuation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicContinuation")
            .field("checkpoint_count", &self.checkpoints.len())
            .finish_non_exhaustive()
    }
}

impl<M> AnthropicMessagesProvider<M> {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            messages_url: messages_url(&base_url.into()),
            credentials: None,
            client: reqwest::Client::new(),
            max_tokens: 8192,
            _model: PhantomData,
        }
    }

    /// Configure an Anthropic API key. Gateways can instead use
    /// `with_credentials` to supply their required headers per request.
    pub fn with_api_key(mut self, api_key: impl AsRef<str>) -> Self {
        self.credentials = Some(Arc::new(AnthropicApiKeyCredentials(
            api_key.as_ref().to_owned(),
        )));
        self
    }

    pub fn with_credentials(mut self, credentials: Arc<dyn CredentialSource>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub fn messages_url(&self) -> &str {
        &self.messages_url
    }
}

struct AnthropicApiKeyCredentials(String);

impl CredentialSource for AnthropicApiKeyCredentials {
    fn credentials(&self) -> CredentialFuture<'_> {
        Box::pin(async {
            let key = HeaderValue::from_str(&self.0)
                .map_err(|error| CredentialError::new(error.to_string()))?;
            let mut headers = HeaderMap::new();
            headers.insert("x-api-key", key);
            Ok(RequestCredentials::from_headers(headers))
        })
    }
}

impl<M: AnthropicModel> Provider for AnthropicMessagesProvider<M> {
    type Model = M;
    type Error = AnthropicError;
    type Continuation = AnthropicContinuation;

    fn supports_vision(&self, model: &M) -> bool {
        model.supports_vision()
    }
    fn create_continuation(&self, _model: &M) -> Self::Continuation {
        AnthropicContinuation::default()
    }

    fn estimate_continuation_tokens(continuation: &Self::Continuation) -> u64 {
        continuation
            .checkpoints
            .iter()
            .flat_map(|checkpoint| &checkpoint.content)
            .filter(|block| {
                !matches!(
                    block.get("type").and_then(Value::as_str),
                    Some("text" | "tool_use")
                )
            })
            .map(|block| {
                serde_json::to_string(block).map_or(0, |text| text.len().div_ceil(4) as u64)
            })
            .sum()
    }

    async fn complete(
        &self,
        continuation: &AnthropicContinuation,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, AnthropicContinuation), AnthropicError> {
        let credentials = match &self.credentials {
            Some(source) => Some(source.credentials().await?),
            None => None,
        };
        let request = build_request(
            model.id(),
            self.max_tokens,
            conversation,
            tools,
            continuation,
        )?;
        let mut headers = credentials
            .map(|credentials| credentials.headers().clone())
            .unwrap_or_default();
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let http_request = self
            .client
            .post(&self.messages_url)
            .headers(headers)
            .json(&request);
        let response = http_request.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = read_limited_body(response, MAX_ERROR_BODY_BYTES).await?;
            let message = serde_json::from_slice::<Value>(&body)
                .ok()
                .and_then(|value| value.pointer("/error/message")?.as_str().map(str::to_owned))
                .unwrap_or_else(|| {
                    if body.is_empty() {
                        status.to_string()
                    } else {
                        String::from_utf8_lossy(&body).into_owned()
                    }
                });
            return Err(AnthropicError::Api {
                status: status.as_u16(),
                message,
            });
        }
        let body: Value = response.json().await?;
        let stop_reason = body
            .get("stop_reason")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                AnthropicError::Protocol("message response missing stop_reason".into())
            })?;
        validate_stop_reason_before_content(stop_reason)?;
        let content = body
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| AnthropicError::Protocol("message response missing content".into()))?;
        let message = parse_content(&content)?;
        validate_stop_reason(stop_reason, &message)?;
        let mut next = continuation.clone();
        next.checkpoints.push(AnthropicCheckpoint {
            model: model.id().to_owned(),
            assistant_message_index: conversation.messages().len(),
            assistant: message.clone(),
            content,
        });
        Ok((message, next))
    }

    fn complete_stream<'a>(
        &'a self,
        continuation: &'a AnthropicContinuation,
        model: &'a M,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, AnthropicContinuation, AnthropicError> {
        Box::pin(async_stream::try_stream! {
            let credentials = match &self.credentials {
                Some(source) => Some(source.credentials().await?),
                None => None,
            };
            let mut request = build_request(model.id(), self.max_tokens, conversation, tools, continuation)?;
            request["stream"] = Value::Bool(true);
            let mut headers = credentials.map(|credentials| credentials.headers().clone()).unwrap_or_default();
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
            let response = self.client.post(&self.messages_url).headers(headers).json(&request).send().await?;
            let status = response.status();
            if !status.is_success() {
                let body = read_limited_body(response, MAX_ERROR_BODY_BYTES).await?;
                let message = serde_json::from_slice::<Value>(&body).ok()
                    .and_then(|value| value.pointer("/error/message")?.as_str().map(str::to_owned))
                    .unwrap_or_else(|| if body.is_empty() { status.to_string() } else { String::from_utf8_lossy(&body).into_owned() });
                Err(AnthropicError::Api { status: status.as_u16(), message })?;
                unreachable!("error propagation exits the stream");
            }
            if response.headers().get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| value.split(';').next() == Some("application/json"))
            {
                let body: Value = response.json().await?;
                let stop_reason = body.get("stop_reason").and_then(Value::as_str)
                    .ok_or_else(|| AnthropicError::Protocol("message response missing stop_reason".into()))?;
                validate_stop_reason_before_content(stop_reason)?;
                let content = body.get("content").and_then(Value::as_array).cloned()
                    .ok_or_else(|| AnthropicError::Protocol("message response missing content".into()))?;
                let message = parse_content(&content)?;
                validate_stop_reason(stop_reason, &message)?;
                let next = checkpoint_anthropic(continuation, model.id(), conversation, message.clone(), content);
                yield ProviderEvent::Completed(message, next);
                return;
            }
            let mut decoder = SseDecoder::default();
            let mut accumulator = AnthropicStreamAccumulator::default();
            let mut bytes = response.bytes_stream();
            while let Some(chunk) = bytes.next().await {
                for event in decoder.push(&chunk?).map_err(AnthropicError::Protocol)? {
                    if let Some(delta) = accumulator.push(&event.data)? {
                        yield ProviderEvent::TextDelta(delta);
                    }
                    if accumulator.finished {
                        let (message, content) = accumulator.finish()?;
                        let next = checkpoint_anthropic(continuation, model.id(), conversation, message.clone(), content);
                        yield ProviderEvent::Completed(message, next);
                        return;
                    }
                }
            }
            decoder.finish().map_err(AnthropicError::Protocol)?;
            Err(AnthropicError::Protocol("message stream ended before message_stop".into()))?;
        })
    }
}

fn checkpoint_anthropic(
    continuation: &AnthropicContinuation,
    model: &str,
    conversation: &Conversation,
    message: AssistantMessage,
    content: Vec<Value>,
) -> AnthropicContinuation {
    let mut next = continuation.clone();
    next.checkpoints.push(AnthropicCheckpoint {
        model: model.to_owned(),
        assistant_message_index: conversation.messages().len(),
        assistant: message,
        content,
    });
    next
}

#[derive(Default)]
struct AnthropicStreamAccumulator {
    blocks: Vec<StreamBlock>,
    stop_reason: Option<String>,
    finished: bool,
}

struct StreamBlock {
    value: Value,
    stopped: bool,
}

const MAX_STREAM_CONTENT_BLOCKS: usize = 512;

impl AnthropicStreamAccumulator {
    fn push(&mut self, data: &str) -> Result<Option<String>, AnthropicError> {
        if self.finished {
            return Err(AnthropicError::Protocol(
                "stream event followed message_stop".into(),
            ));
        }
        let payload: Value = serde_json::from_str(data)?;
        match payload
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "content_block_start" => {
                let index = stream_index(&payload)?;
                if index != self.blocks.len() {
                    return Err(AnthropicError::Protocol(
                        "content block indices must be contiguous".into(),
                    ));
                }
                if self.blocks.len() >= MAX_STREAM_CONTENT_BLOCKS {
                    return Err(AnthropicError::Protocol("too many content blocks".into()));
                }
                let value = payload.get("content_block").cloned().ok_or_else(|| {
                    AnthropicError::Protocol("content block start missing block".into())
                })?;
                let object = value.as_object().ok_or_else(|| {
                    AnthropicError::Protocol("content block must be an object".into())
                })?;
                match object.get("type").and_then(Value::as_str) {
                    Some("text") if object.get("text").is_some_and(Value::is_string) => {}
                    Some("tool_use")
                        if object.get("id").is_some_and(Value::is_string)
                            && object.get("name").is_some_and(Value::is_string) => {}
                    Some("thinking")
                        if object.get("thinking").is_some_and(Value::is_string)
                            && object.get("signature").is_some_and(Value::is_string) => {}
                    Some("redacted_thinking") => {}
                    Some(kind) => {
                        return Err(AnthropicError::Protocol(format!(
                            "invalid streamed content block: {kind}"
                        )));
                    }
                    None => {
                        return Err(AnthropicError::Protocol(
                            "content block missing type".into(),
                        ));
                    }
                }
                self.blocks.push(StreamBlock {
                    value,
                    stopped: false,
                });
                Ok(None)
            }
            "content_block_delta" => {
                let index = stream_index(&payload)?;
                let state = self.blocks.get_mut(index).ok_or_else(|| {
                    AnthropicError::Protocol("content block delta preceded start".into())
                })?;
                if state.stopped {
                    return Err(AnthropicError::Protocol(
                        "content block delta followed stop".into(),
                    ));
                }
                let block = &mut state.value;
                let delta = payload.get("delta").ok_or_else(|| {
                    AnthropicError::Protocol("content block delta missing delta".into())
                })?;
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        require_block_type(block, "text", "text_delta")?;
                        let text = delta.get("text").and_then(Value::as_str).ok_or_else(|| {
                            AnthropicError::Protocol("text delta missing text".into())
                        })?;
                        append_value_string(block, "text", Some(text)).map_err(|_| {
                            AnthropicError::Protocol("text delta targeted non-text block".into())
                        })?;
                        Ok((!text.is_empty()).then(|| text.to_owned()))
                    }
                    Some("input_json_delta") => {
                        require_block_type(block, "tool_use", "input_json_delta")?;
                        let fragment = delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                AnthropicError::Protocol("input delta missing partial_json".into())
                            })?;
                        let existing = block
                            .get("_partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        block["_partial_json"] = Value::String(format!("{existing}{fragment}"));
                        Ok(None)
                    }
                    Some("thinking_delta") => {
                        require_block_type(block, "thinking", "thinking_delta")?;
                        append_value_string(
                            block,
                            "thinking",
                            delta.get("thinking").and_then(Value::as_str),
                        )?;
                        Ok(None)
                    }
                    Some("signature_delta") => {
                        require_block_type(block, "thinking", "signature_delta")?;
                        append_value_string(
                            block,
                            "signature",
                            delta.get("signature").and_then(Value::as_str),
                        )?;
                        Ok(None)
                    }
                    Some(other) => Err(AnthropicError::Protocol(format!(
                        "unsupported content block delta: {other}"
                    ))),
                    None => Err(AnthropicError::Protocol(
                        "content block delta missing type".into(),
                    )),
                }
            }
            "content_block_stop" => {
                let index = stream_index(&payload)?;
                let state = self.blocks.get_mut(index).ok_or_else(|| {
                    AnthropicError::Protocol("content block stop preceded start".into())
                })?;
                if state.stopped {
                    return Err(AnthropicError::Protocol(
                        "duplicate content block stop".into(),
                    ));
                }
                if let Some(fragment) = state
                    .value
                    .as_object_mut()
                    .and_then(|object| object.remove("_partial_json"))
                {
                    state.value["input"] =
                        serde_json::from_str(fragment.as_str().unwrap_or_default())?;
                }
                state.stopped = true;
                Ok(None)
            }
            "message_delta" => {
                if let Some(reason) = payload
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                {
                    if self
                        .stop_reason
                        .as_deref()
                        .is_some_and(|seen| seen != reason)
                    {
                        return Err(AnthropicError::Protocol(
                            "message stop reason changed".into(),
                        ));
                    }
                    self.stop_reason = Some(reason.to_owned());
                }
                Ok(None)
            }
            "message_stop" => {
                if self.blocks.iter().any(|block| !block.stopped) {
                    return Err(AnthropicError::Protocol(
                        "message stopped with an open content block".into(),
                    ));
                }
                self.finished = true;
                Ok(None)
            }
            "error" => Err(AnthropicError::Protocol(
                payload
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown streaming error")
                    .to_owned(),
            )),
            "message_start" | "message_ping" | "ping" => Ok(None),
            other => Err(AnthropicError::Protocol(format!(
                "unknown message stream event: {other}"
            ))),
        }
    }

    fn finish(self) -> Result<(AssistantMessage, Vec<Value>), AnthropicError> {
        let reason = self
            .stop_reason
            .ok_or_else(|| AnthropicError::Protocol("message stream missing stop reason".into()))?;
        validate_stop_reason_before_content(&reason)?;
        let content = self
            .blocks
            .into_iter()
            .map(|block| block.value)
            .collect::<Vec<_>>();
        let message = parse_content(&content)?;
        validate_stop_reason(&reason, &message)?;
        Ok((message, content))
    }
}

fn require_block_type(block: &Value, expected: &str, delta: &str) -> Result<(), AnthropicError> {
    if block.get("type").and_then(Value::as_str) == Some(expected) {
        Ok(())
    } else {
        Err(AnthropicError::Protocol(format!(
            "{delta} targeted incompatible block"
        )))
    }
}

fn stream_index(payload: &Value) -> Result<usize, AnthropicError> {
    payload
        .get("index")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())
        .ok_or_else(|| AnthropicError::Protocol("stream event missing valid index".into()))
}

fn append_value_string(
    block: &mut Value,
    field: &str,
    value: Option<&str>,
) -> Result<(), AnthropicError> {
    let value = value.ok_or_else(|| AnthropicError::Protocol(format!("delta missing {field}")))?;
    match block.get_mut(field) {
        Some(Value::String(target)) => target.push_str(value),
        _ => {
            return Err(AnthropicError::Protocol(format!(
                "{field} delta targeted incompatible block"
            )));
        }
    }
    Ok(())
}

const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

async fn read_limited_body(
    response: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, AnthropicError> {
    let mut bytes = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let remaining = limit.saturating_sub(bytes.len());
        bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if bytes.len() == limit {
            break;
        }
    }
    Ok(bytes)
}

fn validate_stop_reason(
    stop_reason: &str,
    message: &AssistantMessage,
) -> Result<(), AnthropicError> {
    validate_stop_reason_before_content(stop_reason)?;
    match stop_reason {
        "end_turn" | "stop_sequence" if message.tool_calls().is_empty() => Ok(()),
        "tool_use" if !message.tool_calls().is_empty() => Ok(()),
        "end_turn" | "stop_sequence" => Err(AnthropicError::Protocol(format!(
            "{stop_reason} response unexpectedly contained tool calls"
        ))),
        "tool_use" => Err(AnthropicError::Protocol(
            "tool_use response contained no client tool calls".into(),
        )),
        other => Err(AnthropicError::Protocol(format!(
            "unexpected stop reason after validation: {other}"
        ))),
    }
}

fn validate_stop_reason_before_content(stop_reason: &str) -> Result<(), AnthropicError> {
    match stop_reason {
        "end_turn" | "stop_sequence" | "tool_use" => Ok(()),
        "max_tokens" | "model_context_window_exceeded" => Err(AnthropicError::Incomplete {
            reason: stop_reason.to_owned(),
        }),
        "pause_turn" | "refusal" => Err(AnthropicError::Protocol(format!(
            "unsupported nonterminal stop reason: {stop_reason}"
        ))),
        other => Err(AnthropicError::Protocol(format!(
            "unknown stop reason: {other}"
        ))),
    }
}

fn build_request(
    model: &str,
    max_tokens: u32,
    conversation: &Conversation,
    tools: &[ToolSpec],
    continuation: &AnthropicContinuation,
) -> Result<Value, AnthropicError> {
    validate_continuation(model, conversation, continuation)?;
    let mut messages = Vec::new();
    let mut system_messages = conversation
        .system_prompt()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut index = 0;
    while index < conversation.messages().len() {
        match &conversation.messages()[index] {
            Message::ToolResult(_) => {
                let mut content = Vec::new();
                while let Some(Message::ToolResult(result)) = conversation.messages().get(index) {
                    content.extend(convert_tool_result(result));
                    index += 1;
                }
                messages.push(json!({"role":"user", "content":content}));
            }
            Message::Assistant(_)
                if continuation
                    .checkpoints
                    .iter()
                    .any(|checkpoint| checkpoint.assistant_message_index == index) =>
            {
                let checkpoint = continuation
                    .checkpoints
                    .iter()
                    .find(|checkpoint| checkpoint.assistant_message_index == index)
                    .ok_or_else(|| {
                        AnthropicError::Protocol("continuation checkpoint disappeared".into())
                    })?;
                messages.push(json!({"role":"assistant", "content":checkpoint.content}));
                index += 1;
            }
            Message::System(system) => {
                system_messages.push(system.content().to_owned());
                index += 1;
            }
            message => {
                messages.push(convert_message(message));
                index += 1;
            }
        }
    }
    let mut request = json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "tools": tools.iter().map(|tool| json!({
            "name":tool.name(), "description":tool.description(), "input_schema":tool.input_schema()
        })).collect::<Vec<_>>()
    });
    if !system_messages.is_empty() {
        request["system"] = Value::String(system_messages.join("\n\n"));
    }
    Ok(request)
}

fn validate_continuation(
    model: &str,
    conversation: &Conversation,
    continuation: &AnthropicContinuation,
) -> Result<(), AnthropicError> {
    let mut previous_index = None;
    for checkpoint in &continuation.checkpoints {
        if checkpoint.model != model {
            return Err(AnthropicError::Protocol(
                "continuation belongs to a different model".into(),
            ));
        }
        if previous_index.is_some_and(|index| checkpoint.assistant_message_index <= index) {
            return Err(AnthropicError::Protocol(
                "continuation checkpoints are out of order".into(),
            ));
        }
        let expected = Message::Assistant(checkpoint.assistant.clone());
        if conversation
            .messages()
            .get(checkpoint.assistant_message_index)
            != Some(&expected)
        {
            return Err(AnthropicError::Protocol(
                "continuation does not match conversation history".into(),
            ));
        }
        previous_index = Some(checkpoint.assistant_message_index);
    }
    Ok(())
}

fn convert_message(message: &Message) -> Value {
    match message {
        Message::User(user) => {
            let mut content = Vec::new();
            for image in user.images() {
                content.push(image_block(image));
            }
            content.push(json!({"type":"text", "text":user.content()}));
            json!({"role":"user", "content":content})
        }
        Message::Assistant(assistant) => {
            let mut content = Vec::new();
            if !assistant.content().is_empty() {
                content.push(json!({"type":"text", "text":assistant.content()}));
            }
            content.extend(assistant.tool_calls().iter().map(|call| {
                json!({
                    "type":"tool_use", "id":call.id(), "name":call.name(), "input":call.arguments()
                })
            }));
            json!({"role":"assistant", "content":content})
        }
        Message::System(_) | Message::ToolResult(_) => unreachable!(),
    }
}

fn convert_tool_result(result: &ToolResult) -> Vec<Value> {
    let (content, is_error, images) = match result.outcome() {
        ToolResultOutcome::Success(output) => (
            serde_json::to_string(output.value()).unwrap_or_default(),
            false,
            output.images(),
        ),
        ToolResultOutcome::Error(ToolExecutorError::UnknownTool(name)) => {
            (format!("unknown tool: {name}"), true, &[][..])
        }
        ToolResultOutcome::Error(ToolExecutorError::Tool(error)) => {
            (error.message().to_owned(), true, &[][..])
        }
    };
    let mut blocks = vec![json!({
        "type":"tool_result", "tool_use_id":result.tool_call_id(),
        "content":content, "is_error":is_error
    })];
    blocks.extend(images.iter().map(image_block));
    blocks
}

fn image_block(image: &kodkod_core::Image) -> Value {
    let data_url = image.to_data_url();
    let data = data_url
        .split_once(',')
        .map(|(_, data)| data)
        .unwrap_or_default();
    json!({"type":"image", "source":{
        "type":"base64", "media_type":image.mime(), "data":data
    }})
}

fn parse_content(content: &[Value]) -> Result<AssistantMessage, AnthropicError> {
    let mut text = String::new();
    let mut calls = Vec::new();
    for block in content {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => text.push_str(&required(block, "text")?),
            Some("tool_use") => {
                let id = required(block, "id")?;
                let name = required(block, "name")?;
                let input = block
                    .get("input")
                    .cloned()
                    .ok_or_else(|| AnthropicError::Protocol("tool use missing input".into()))?;
                calls.push(ToolCall::new(id, name, input));
            }
            Some("thinking" | "redacted_thinking") => {}
            Some(kind) => {
                return Err(AnthropicError::Protocol(format!(
                    "unsupported message content type: {kind}"
                )));
            }
            None => {
                return Err(AnthropicError::Protocol(
                    "content block missing type".into(),
                ));
            }
        }
    }
    Ok(AssistantMessage::new(text).with_tool_calls(calls))
}

fn required(value: &Value, field: &str) -> Result<String, AnthropicError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| AnthropicError::Protocol(format!("tool use missing {field}")))
}

#[derive(Debug)]
pub enum AnthropicError {
    Http(reqwest::Error),
    Json(serde_json::Error),
    Credentials(CredentialError),
    Api { status: u16, message: String },
    Incomplete { reason: String },
    Protocol(String),
}

impl fmt::Display for AnthropicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(error) => write!(f, "http request failed: {error}"),
            Self::Json(error) => write!(f, "failed to parse response: {error}"),
            Self::Credentials(error) => write!(f, "credentials unavailable: {error}"),
            Self::Api { status, message } => write!(f, "api error ({status}): {message}"),
            Self::Incomplete { reason } => write!(f, "message incomplete: {reason}"),
            Self::Protocol(message) => write!(f, "messages protocol error: {message}"),
        }
    }
}
impl Error for AnthropicError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Http(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Credentials(error) => Some(error),
            _ => None,
        }
    }
}
impl From<reqwest::Error> for AnthropicError {
    fn from(value: reqwest::Error) -> Self {
        Self::Http(value)
    }
}
impl From<serde_json::Error> for AnthropicError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
impl From<CredentialError> for AnthropicError {
    fn from(value: CredentialError) -> Self {
        Self::Credentials(value)
    }
}
impl Retryable for AnthropicError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Http(error) => error.is_connect() || error.is_timeout() || error.is_request(),
            Self::Api { status, .. } => matches!(status, 429 | 500 | 502 | 503 | 504 | 529),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kodkod_core::{Image, UserMessage};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct TestModel;
    impl AnthropicModel for TestModel {
        fn id(&self) -> &str {
            "claude-test"
        }
        fn supports_vision(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn posts_native_messages_with_tools_images_and_headers() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "stop_reason":"tool_use",
                "content":[
                    {"type":"text","text":"checking"},
                    {"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}
                ]
            })))
            .mount(&server)
            .await;
        let provider = AnthropicMessagesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_api_key("secret")
            .with_max_tokens(4096);
        let mut conversation = Conversation::new().with_system_prompt("system");
        conversation.push_user_message(
            UserMessage::new("look").with_images(vec![Image::new("image/png", b"png")]),
        );
        let response = provider
            .complete_once(
                &TestModel,
                &conversation,
                &[ToolSpec::new("lookup", "Lookup", json!({"type":"object"}))],
            )
            .await
            .unwrap();
        assert_eq!(response.content(), "checking");
        assert_eq!(response.tool_calls()[0].id(), "toolu_1");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests[0].headers["x-api-key"], "secret");
        assert_eq!(requests[0].headers["anthropic-version"], "2023-06-01");
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["model"], "claude-test");
        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["system"], "system");
        assert_eq!(body["messages"][0]["content"][0]["type"], "image");
        assert_eq!(body["tools"][0]["name"], "lookup");
    }

    #[tokio::test]
    async fn streams_only_text_and_checkpoints_signed_thinking_on_message_stop() {
        let server = MockServer::start().await;
        let body = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"private\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"signed\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"draft\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(body),
            )
            .mount(&server)
            .await;
        let provider = AnthropicMessagesProvider::<TestModel>::new(format!("{}/v1", server.uri()));
        let conversation = Conversation::new();
        let continuation = provider.create_continuation(&TestModel);
        let mut stream = provider.complete_stream(&continuation, &TestModel, &conversation, &[]);
        assert!(
            matches!(stream.next().await.unwrap().unwrap(), ProviderEvent::TextDelta(text) if text == "draft")
        );
        let ProviderEvent::Completed(message, continuation) = stream.next().await.unwrap().unwrap()
        else {
            panic!("expected terminal completion");
        };
        assert_eq!(message.content(), "draft");
        drop(stream);

        let mut history = Conversation::new();
        history.push_message(Message::Assistant(message));
        let replay = build_request("claude-test", 1, &history, &[], &continuation).unwrap();
        assert_eq!(replay["messages"][0]["content"][0]["thinking"], "private");
        assert_eq!(replay["messages"][0]["content"][0]["signature"], "signed");
    }

    #[test]
    fn groups_parallel_tool_results_into_one_user_message() {
        let mut conversation = Conversation::new();
        conversation.push_message(Message::Assistant(
            AssistantMessage::new("").with_tool_calls(vec![
                ToolCall::new("a", "one", json!({})),
                ToolCall::new("b", "two", json!({})),
            ]),
        ));
        conversation.push_message(Message::ToolResult(ToolResult::success("a", json!(1))));
        conversation.push_message(Message::ToolResult(ToolResult::success("b", json!(2))));
        let request = build_request(
            "claude",
            10,
            &conversation,
            &[],
            &AnthropicContinuation::default(),
        )
        .unwrap();
        assert_eq!(request["messages"].as_array().unwrap().len(), 2);
        assert_eq!(
            request["messages"][1]["content"].as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn replays_opaque_thinking_and_preserves_all_system_messages() {
        let thinking = json!({"type":"thinking", "thinking":"private", "signature":"signed"});
        let continuation = AnthropicContinuation {
            checkpoints: vec![AnthropicCheckpoint {
                model: "claude".into(),
                assistant_message_index: 1,
                assistant: AssistantMessage::new("answer"),
                content: vec![thinking.clone(), json!({"type":"text", "text":"answer"})],
            }],
        };
        let mut conversation = Conversation::new().with_system_prompt("first");
        conversation.push_user_message(UserMessage::new("question"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("answer")));
        conversation.push_message(Message::System(kodkod_core::SystemMessage::new("second")));
        conversation.push_user_message(UserMessage::new("next"));
        let request = build_request("claude", 10, &conversation, &[], &continuation).unwrap();
        assert_eq!(request["messages"][1]["content"][0], thinking);
        assert_eq!(request["system"], "first\n\nsecond");
        assert!(!format!("{continuation:?}").contains("private"));
    }

    #[test]
    fn continuation_rejects_edited_history_and_model_changes() {
        let continuation = AnthropicContinuation {
            checkpoints: vec![AnthropicCheckpoint {
                model: "claude".into(),
                assistant_message_index: 1,
                assistant: AssistantMessage::new("original"),
                content: vec![json!({"type":"text", "text":"original"})],
            }],
        };
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("question"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("edited")));
        assert!(build_request("claude", 10, &conversation, &[], &continuation).is_err());
        conversation.replace_messages(vec![
            Message::User(UserMessage::new("question")),
            Message::Assistant(AssistantMessage::new("original")),
        ]);
        assert!(build_request("other", 10, &conversation, &[], &continuation).is_err());
    }

    #[test]
    fn malformed_or_unknown_content_is_rejected() {
        assert!(parse_content(&[json!({"type":"tool_use", "id":"x", "name":"f"})]).is_err());
        assert!(parse_content(&[json!({"type":"future_action", "id":"x"})]).is_err());
    }

    #[test]
    fn stop_reasons_reject_partial_and_inconsistent_responses() {
        let text = AssistantMessage::new("partial");
        assert!(matches!(
            validate_stop_reason("max_tokens", &text),
            Err(AnthropicError::Incomplete { reason }) if reason == "max_tokens"
        ));
        assert!(validate_stop_reason("pause_turn", &text).is_err());
        assert!(validate_stop_reason("refusal", &text).is_err());
        assert!(validate_stop_reason("tool_use", &text).is_err());

        let tool = AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
            "call",
            "lookup",
            json!({}),
        )]);
        assert!(validate_stop_reason("end_turn", &tool).is_err());
        assert!(validate_stop_reason("future_reason", &text).is_err());
    }

    #[test]
    fn stream_rejects_unbounded_invalid_or_incomplete_block_lifecycles() {
        let mut accumulator = AnthropicStreamAccumulator::default();
        assert!(accumulator
            .push(r#"{"type":"content_block_start","index":999999999,"content_block":{"type":"text","text":""}}"#)
            .is_err());
        assert!(
            accumulator
                .push(r#"{"type":"content_block_start","index":0,"content_block":"scalar"}"#)
                .is_err()
        );

        let mut accumulator = AnthropicStreamAccumulator::default();
        accumulator
            .push(r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"x","name":"f","input":{}}}"#)
            .unwrap();
        assert!(accumulator
            .push(r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"wrong"}}"#)
            .is_err());
        assert!(accumulator.push(r#"{"type":"message_stop"}"#).is_err());

        let mut accumulator = AnthropicStreamAccumulator::default();
        accumulator
            .push(r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"x","name":"f","input":{}}}"#)
            .unwrap();
        accumulator
            .push(r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}}"#)
            .unwrap();
        accumulator
            .push(r#"{"type":"content_block_stop","index":0}"#)
            .unwrap();
        assert!(
            accumulator
                .push(r#"{"type":"content_block_stop","index":0}"#)
                .is_err()
        );
        assert!(accumulator
            .push(r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}"#)
            .is_err());
        accumulator
            .push(r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#)
            .unwrap();
        accumulator.push(r#"{"type":"message_stop"}"#).unwrap();
        assert!(accumulator.push(r#"{"type":"message_stop"}"#).is_err());
        let (message, _) = accumulator.finish().unwrap();
        assert_eq!(message.tool_calls()[0].arguments(), &json!({"q":"x"}));
    }
}
