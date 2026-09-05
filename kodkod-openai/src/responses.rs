use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, Conversation, Message, Provider, ToolCall, ToolExecutorError, ToolResult,
    ToolResultOutcome, ToolSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{CredentialSource, OpenAiError, OpenAiModel};

pub fn responses_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    format!("{base}/responses")
}

/// Opaque Responses output retained for stateless replay.
///
/// Output items are intentionally not exposed. In particular, encrypted
/// reasoning and future item fields must be returned to the service unchanged.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ResponsesContinuation {
    checkpoints: Vec<ResponseCheckpoint>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ResponseCheckpoint {
    model: String,
    assistant_message_index: usize,
    assistant: AssistantMessage,
    output: Vec<Value>,
}

impl std::fmt::Debug for ResponsesContinuation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesContinuation")
            .field("checkpoint_count", &self.checkpoints.len())
            .finish_non_exhaustive()
    }
}

/// Provider for the HTTP/SSE Responses API.
///
/// It uses stateless replay (`store: false`) and does not implement the newer
/// WebSocket-only steering or asynchronous tool-result transport.
#[derive(Clone)]
pub struct OpenAiResponsesProvider<M = ()> {
    responses_url: String,
    credentials: Option<Arc<dyn CredentialSource>>,
    instructions: Option<String>,
    client: reqwest::Client,
    _model: PhantomData<M>,
}

impl<M> OpenAiResponsesProvider<M> {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            responses_url: responses_url(&base_url.into()),
            credentials: None,
            instructions: None,
            client: reqwest::Client::new(),
            _model: PhantomData,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.credentials = Some(Arc::new(crate::StaticCredentials::bearer(api_key.into())));
        self
    }

    pub fn with_credentials(mut self, credentials: Arc<dyn CredentialSource>) -> Self {
        self.credentials = Some(credentials);
        self
    }

    /// Set the optional Responses `instructions` field for every request.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    pub fn responses_url(&self) -> &str {
        &self.responses_url
    }
}

impl<M> Provider for OpenAiResponsesProvider<M>
where
    M: OpenAiModel,
{
    type Model = M;
    type Error = OpenAiError;
    type Continuation = ResponsesContinuation;

    fn supports_vision(&self, model: &M) -> bool {
        model.supports_vision()
    }

    fn create_continuation(&self, _model: &M) -> Self::Continuation {
        ResponsesContinuation::default()
    }

    fn estimate_continuation_tokens(continuation: &Self::Continuation) -> u64 {
        continuation
            .checkpoints
            .iter()
            .flat_map(|checkpoint| &checkpoint.output)
            .filter(|item| {
                !matches!(
                    item.get("type").and_then(Value::as_str),
                    Some("message" | "function_call")
                )
            })
            .map(|item| serde_json::to_string(item).map_or(0, |text| text.len().div_ceil(4) as u64))
            .sum()
    }

    async fn complete(
        &self,
        continuation: &Self::Continuation,
        model: &M,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> Result<(AssistantMessage, Self::Continuation), Self::Error> {
        let credentials = match &self.credentials {
            Some(source) => Some(source.credentials().await?),
            None => None,
        };
        let request = build_request(
            model.id(),
            self.instructions.as_deref(),
            conversation,
            tools,
            continuation,
        )?;
        let mut http_request = self.client.post(&self.responses_url).json(&request);
        if let Some(credentials) = credentials {
            http_request = http_request.headers(credentials.headers().clone());
        }

        let response = http_request.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = read_limited_body(response, MAX_ERROR_BODY_BYTES).await?;
            let message = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|value| value.pointer("/error/message")?.as_str().map(str::to_owned))
                .unwrap_or_else(|| {
                    if body.is_empty() {
                        status.to_string()
                    } else {
                        body
                    }
                });
            return Err(OpenAiError::Api {
                status: status.as_u16(),
                message,
            });
        }

        let terminal = read_terminal_response(response).await?;
        let output = terminal
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| OpenAiError::Protocol("completed response missing output".into()))?;
        let assistant = parse_output(&output)?;

        let mut next = continuation.clone();
        next.checkpoints.push(ResponseCheckpoint {
            model: model.id().to_owned(),
            assistant_message_index: conversation.messages().len(),
            assistant: assistant.clone(),
            output,
        });
        Ok((assistant, next))
    }
}

fn build_request(
    model: &str,
    configured_instructions: Option<&str>,
    conversation: &Conversation,
    tools: &[ToolSpec],
    continuation: &ResponsesContinuation,
) -> Result<Value, OpenAiError> {
    validate_continuation(model, conversation, continuation)?;
    let checkpoints: BTreeMap<usize, &ResponseCheckpoint> = continuation
        .checkpoints
        .iter()
        .map(|checkpoint| (checkpoint.assistant_message_index, checkpoint))
        .collect();
    let mut input = Vec::new();
    for (index, message) in conversation.messages().iter().enumerate() {
        if let Some(checkpoint) = checkpoints.get(&index) {
            input.extend(checkpoint.output.iter().cloned());
        } else {
            input.extend(convert_message(message));
        }
    }

    let instructions = configured_instructions.or(conversation.system_prompt());
    let mut request = json!({
        "model": model,
        "input": input,
        "tools": tools.iter().map(convert_tool).collect::<Vec<_>>(),
        "stream": true,
        "store": false,
        "include": ["reasoning.encrypted_content"]
    });
    if let Some(instructions) = instructions {
        request["instructions"] = Value::String(instructions.to_owned());
    }
    Ok(request)
}

fn validate_continuation(
    model: &str,
    conversation: &Conversation,
    continuation: &ResponsesContinuation,
) -> Result<(), OpenAiError> {
    let mut previous_index = None;
    for checkpoint in &continuation.checkpoints {
        if checkpoint.model != model {
            return Err(OpenAiError::Protocol(
                "continuation belongs to a different model".into(),
            ));
        }
        if previous_index.is_some_and(|index| checkpoint.assistant_message_index <= index) {
            return Err(OpenAiError::Protocol(
                "continuation checkpoints are out of order".into(),
            ));
        }
        let expected = Message::Assistant(checkpoint.assistant.clone());
        if conversation
            .messages()
            .get(checkpoint.assistant_message_index)
            != Some(&expected)
        {
            return Err(OpenAiError::Protocol(
                "continuation does not match conversation history".into(),
            ));
        }
        previous_index = Some(checkpoint.assistant_message_index);
    }
    Ok(())
}

fn convert_message(message: &Message) -> Vec<Value> {
    match message {
        Message::System(system) => vec![json!({
            "role": "system",
            "content": [{"type": "input_text", "text": system.content()}]
        })],
        Message::User(user) => {
            let mut content = vec![json!({"type": "input_text", "text": user.content()})];
            content.extend(user.images().iter().map(|image| {
                json!({"type": "input_image", "image_url": image.to_data_url(), "detail": "auto"})
            }));
            vec![json!({"role": "user", "content": content})]
        }
        Message::Assistant(assistant) => {
            let mut items = Vec::new();
            if !assistant.content().is_empty() {
                items.push(json!({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": assistant.content()}]
                }));
            }
            items.extend(assistant.tool_calls().iter().map(|call| json!({
                "type": "function_call",
                "call_id": call.id(),
                "name": call.name(),
                "arguments": serde_json::to_string(call.arguments()).unwrap_or_else(|_| "{}".into())
            })));
            items
        }
        Message::ToolResult(result) => convert_tool_result(result),
    }
}

fn convert_tool_result(result: &ToolResult) -> Vec<Value> {
    let output = match result.outcome() {
        ToolResultOutcome::Success(output) => {
            serde_json::to_string(output.value()).unwrap_or_default()
        }
        ToolResultOutcome::Error(ToolExecutorError::UnknownTool(name)) => {
            format!("unknown tool: {name}")
        }
        ToolResultOutcome::Error(ToolExecutorError::Tool(error)) => error.message().to_owned(),
    };
    let mut items = vec![json!({
        "type": "function_call_output",
        "call_id": result.tool_call_id(),
        "output": output
    })];
    if let ToolResultOutcome::Success(output) = result.outcome()
        && !output.images().is_empty()
    {
        let mut content = vec![json!({
            "type": "input_text",
            "text": format!("Images returned by tool call '{}'.", result.tool_call_id())
        })];
        content.extend(output.images().iter().map(|image| {
            json!({"type": "input_image", "image_url": image.to_data_url(), "detail": "auto"})
        }));
        items.push(json!({"role": "user", "content": content}));
    }
    items
}

fn convert_tool(spec: &ToolSpec) -> Value {
    json!({
        "type": "function",
        "name": spec.name(),
        "description": spec.description(),
        "parameters": spec.input_schema(),
        "strict": false
    })
}

const MAX_SSE_EVENT_BYTES: usize = 1024 * 1024;
const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;

async fn read_terminal_response(response: reqwest::Response) -> Result<Value, OpenAiError> {
    let mut stream = response.bytes_stream();
    let mut decoder = SseDecoder::default();
    while let Some(chunk) = stream.next().await {
        for event in decoder.push(&chunk?)? {
            if let Some(terminal) = handle_sse_event(event)? {
                return Ok(terminal);
            }
        }
    }
    decoder.finish()?;
    Err(OpenAiError::Protocol(
        "stream ended without a terminal event".into(),
    ))
}

async fn read_limited_body(
    response: reqwest::Response,
    limit: usize,
) -> Result<String, OpenAiError> {
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
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

struct SseEvent {
    event_name: Option<String>,
    data: String,
}

impl SseDecoder {
    fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, OpenAiError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some((payload_end, delimiter_end)) = find_event_boundary(&self.buffer) {
            if payload_end > MAX_SSE_EVENT_BYTES {
                return Err(OpenAiError::Protocol(
                    "SSE event exceeded size limit".into(),
                ));
            }
            let payload = self.buffer[..payload_end].to_vec();
            self.buffer.drain(..delimiter_end);
            if let Some(event) = parse_sse_block(&payload)? {
                events.push(event);
            }
        }
        if self.buffer.len() > MAX_SSE_EVENT_BYTES {
            return Err(OpenAiError::Protocol(
                "SSE event exceeded size limit".into(),
            ));
        }
        Ok(events)
    }

    fn finish(&self) -> Result<(), OpenAiError> {
        if self.buffer.iter().any(|byte| !byte.is_ascii_whitespace()) {
            return Err(OpenAiError::Protocol("truncated SSE event".into()));
        }
        Ok(())
    }
}

fn find_event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    for index in 0..buffer.len() {
        if buffer.get(index..index + 2) == Some(b"\n\n") {
            return Some((index, index + 2));
        }
        if buffer.get(index..index + 4) == Some(b"\r\n\r\n") {
            return Some((index, index + 4));
        }
    }
    None
}

fn parse_sse_block(payload: &[u8]) -> Result<Option<SseEvent>, OpenAiError> {
    let block = std::str::from_utf8(payload)
        .map_err(|error| OpenAiError::Protocol(format!("SSE event was not UTF-8: {error}")))?;
    let mut event_name = None;
    let mut data = String::new();
    for line in block.lines() {
        if let Some(value) = line.strip_prefix("event:") {
            event_name = Some(value.trim().to_owned());
        } else if let Some(value) = line.strip_prefix("data:") {
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(value.trim_start());
        }
    }
    if data.is_empty() || data == "[DONE]" {
        return Ok(None);
    }
    Ok(Some(SseEvent { event_name, data }))
}

fn handle_sse_event(event: SseEvent) -> Result<Option<Value>, OpenAiError> {
    let payload: Value = serde_json::from_str(&event.data)?;
    let kind = payload
        .get("type")
        .and_then(Value::as_str)
        .or(event.event_name.as_deref())
        .unwrap_or_default();
    match kind {
        "response.completed" => payload
            .get("response")
            .cloned()
            .map(Some)
            .ok_or_else(|| OpenAiError::Protocol("completed event missing response".into())),
        "response.failed" => Err(response_failure(&payload)),
        "response.incomplete" => {
            let reason = payload
                .pointer("/response/incomplete_details/reason")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_owned();
            Err(OpenAiError::Incomplete { reason })
        }
        "error" => {
            let message = payload
                .get("message")
                .or_else(|| payload.pointer("/error/message"))
                .and_then(Value::as_str)
                .unwrap_or("unknown streaming error")
                .to_owned();
            Err(OpenAiError::Protocol(message))
        }
        _ => Ok(None),
    }
}

fn response_failure(event: &Value) -> OpenAiError {
    let message = event
        .pointer("/response/error/message")
        .and_then(Value::as_str)
        .unwrap_or("response failed")
        .to_owned();
    OpenAiError::Protocol(message)
}

fn parse_output(output: &[Value]) -> Result<AssistantMessage, OpenAiError> {
    if output.is_empty() {
        return Err(OpenAiError::Protocol(
            "completed response returned no output items".into(),
        ));
    }
    let mut text = String::new();
    let mut calls = Vec::new();
    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let content = item
                    .get("content")
                    .and_then(Value::as_array)
                    .ok_or_else(|| OpenAiError::Protocol("message item missing content".into()))?;
                for part in content {
                    match part.get("type").and_then(Value::as_str) {
                        Some("output_text") => {
                            text.push_str(&required_string(part, "text")?);
                        }
                        Some("refusal") => {
                            text.push_str(&required_string(part, "refusal")?);
                        }
                        Some(kind) => {
                            return Err(OpenAiError::Protocol(format!(
                                "unsupported response content type: {kind}"
                            )));
                        }
                        None => {
                            return Err(OpenAiError::Protocol(
                                "response content missing type".into(),
                            ));
                        }
                    }
                }
            }
            Some("function_call") => {
                let call_id = required_string(item, "call_id")?;
                let name = required_string(item, "name")?;
                let arguments = required_string(item, "arguments")?;
                let arguments = serde_json::from_str(&arguments)?;
                calls.push(ToolCall::new(call_id, name, arguments));
            }
            Some("reasoning") => {}
            Some(kind) => {
                return Err(OpenAiError::Protocol(format!(
                    "unsupported response output type: {kind}"
                )));
            }
            None => return Err(OpenAiError::Protocol("response output missing type".into())),
        }
    }
    if text.is_empty() && calls.is_empty() {
        return Err(OpenAiError::Protocol(
            "completed response contained no assistant output".into(),
        ));
    }
    Ok(AssistantMessage::new(text).with_tool_calls(calls))
}

fn required_string(item: &Value, field: &str) -> Result<String, OpenAiError> {
    item.get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| OpenAiError::Protocol(format!("function call missing {field}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kodkod_core::{Image, UserMessage};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    struct TestModel;
    impl OpenAiModel for TestModel {
        fn id(&self) -> &str {
            "gpt-test"
        }
        fn supports_vision(&self) -> bool {
            true
        }
    }

    struct RotatingCredentials(AtomicUsize);

    impl CredentialSource for RotatingCredentials {
        fn credentials(&self) -> crate::CredentialFuture<'_> {
            Box::pin(async {
                let token = if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
                    "first"
                } else {
                    "second"
                };
                crate::RequestCredentials::bearer(token)
            })
        }
    }

    fn completed(output: Value) -> String {
        format!(
            "event: response.completed\ndata: {}\n\n",
            json!({
                "type": "response.completed",
                "response": {"status": "completed", "output": output}
            })
        )
    }

    fn parse_complete_stream(body: &[u8]) -> Result<Value, OpenAiError> {
        let mut decoder = SseDecoder::default();
        for event in decoder.push(body)? {
            if let Some(response) = handle_sse_event(event)? {
                return Ok(response);
            }
        }
        decoder.finish()?;
        Err(OpenAiError::Protocol(
            "stream ended without a terminal event".into(),
        ))
    }

    #[tokio::test]
    async fn streams_and_replays_opaque_output_with_fresh_credentials() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer first"))
            .respond_with(ResponseTemplate::new(200).set_body_string(completed(json!([
                {"id":"rs_1","type":"reasoning","encrypted_content":"opaque","phase":"analysis"},
                {"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}
            ]))))
            .expect(1)
            .mount(&server).await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .and(header("authorization", "Bearer second"))
            .respond_with(ResponseTemplate::new(200).set_body_string(completed(json!([
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}
            ]))))
            .expect(1)
            .mount(&server)
            .await;

        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()))
            .with_credentials(Arc::new(RotatingCredentials(AtomicUsize::new(0))))
            .with_instructions("developer");
        let mut conversation = Conversation::new();
        conversation.push_user_message(
            UserMessage::new("look").with_images(vec![Image::new("image/png", b"png")]),
        );
        let (assistant, continuation) = provider
            .complete(
                &ResponsesContinuation::default(),
                &TestModel,
                &conversation,
                &[],
            )
            .await
            .unwrap();
        assert_eq!(assistant.tool_calls()[0].name(), "lookup");

        let serialized = serde_json::to_value(&continuation).unwrap();
        assert_eq!(
            serialized["checkpoints"][0]["output"][0]["encrypted_content"],
            "opaque"
        );
        let (second, _) = provider
            .complete(
                &ResponsesContinuation::default(),
                &TestModel,
                &conversation,
                &[],
            )
            .await
            .unwrap();
        assert_eq!(second.content(), "done");
    }

    #[test]
    fn request_replaces_normalized_assistant_with_opaque_items() {
        let opaque =
            json!({"id":"rs_1","type":"reasoning","encrypted_content":"secret","phase":"analysis"});
        let continuation = ResponsesContinuation {
            checkpoints: vec![ResponseCheckpoint {
                model: "gpt-test".into(),
                assistant_message_index: 1,
                assistant: AssistantMessage::new("answer"),
                output: vec![
                    opaque.clone(),
                    json!({"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}),
                ],
            }],
        };
        let mut conversation = Conversation::new().with_system_prompt("system");
        conversation.push_user_message(UserMessage::new("question"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("answer")));
        conversation.push_user_message(UserMessage::new("next"));
        let request = build_request("gpt-test", None, &conversation, &[], &continuation).unwrap();
        assert_eq!(request["stream"], true);
        assert_eq!(request["store"], false);
        assert_eq!(request["include"][0], "reasoning.encrypted_content");
        assert_eq!(request["input"][1], opaque);
        assert_eq!(request["instructions"], "system");
    }

    #[test]
    fn reports_incomplete_terminal_event() {
        let error = parse_complete_stream(
            format!(
                "event: response.incomplete\ndata: {}\n\n",
                json!({
                    "type":"response.incomplete",
                    "response":{"incomplete_details":{"reason":"max_output_tokens"}}
                })
            )
            .as_bytes(),
        )
        .unwrap_err();
        assert!(
            matches!(error, OpenAiError::Incomplete { reason } if reason == "max_output_tokens")
        );
    }

    #[test]
    fn continuation_debug_does_not_reveal_encrypted_reasoning() {
        let continuation = ResponsesContinuation {
            checkpoints: vec![ResponseCheckpoint {
                model: "gpt-test".into(),
                assistant_message_index: 0,
                assistant: AssistantMessage::new(""),
                output: vec![json!({"type":"reasoning", "encrypted_content":"top-secret"})],
            }],
        };
        assert!(!format!("{continuation:?}").contains("top-secret"));
    }

    #[test]
    fn decoder_handles_utf8_chunk_boundaries_and_crlf() {
        let stream = completed(json!([{
            "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"räksmörgås"}]
        }]))
        .replace('\n', "\r\n");
        let bytes = stream.as_bytes();
        let split = stream.find('ä').unwrap() + 1;
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(&bytes[..split]).unwrap().is_empty());
        let events = decoder.push(&bytes[split..]).unwrap();
        let response = events
            .into_iter()
            .find_map(|event| handle_sse_event(event).unwrap())
            .unwrap();
        assert_eq!(
            parse_output(response["output"].as_array().unwrap())
                .unwrap()
                .content(),
            "räksmörgås"
        );
    }

    #[test]
    fn decoder_surfaces_truncated_stream_and_terminal_before_finish() {
        let mut decoder = SseDecoder::default();
        decoder.push(b"event: response.created\ndata: {").unwrap();
        assert!(
            matches!(decoder.finish(), Err(OpenAiError::Protocol(message)) if message.contains("truncated"))
        );

        let terminal = completed(json!([{
            "type":"message", "role":"assistant", "content":[{"type":"output_text", "text":"done"}]
        }]));
        let mut decoder = SseDecoder::default();
        let events = decoder.push(terminal.as_bytes()).unwrap();
        assert!(
            events
                .into_iter()
                .any(|event| handle_sse_event(event).unwrap().is_some())
        );
    }

    #[test]
    fn continuation_rejects_edited_history_and_model_changes() {
        let checkpoint = ResponseCheckpoint {
            model: "gpt-test".into(),
            assistant_message_index: 1,
            assistant: AssistantMessage::new("original"),
            output: vec![
                json!({"type":"message", "role":"assistant", "content":[{"type":"output_text", "text":"original"}]}),
            ],
        };
        let continuation = ResponsesContinuation {
            checkpoints: vec![checkpoint],
        };
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("question"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("edited")));
        assert!(build_request("gpt-test", None, &conversation, &[], &continuation).is_err());
        conversation.replace_messages(vec![
            Message::User(UserMessage::new("question")),
            Message::Assistant(AssistantMessage::new("original")),
        ]);
        assert!(build_request("different", None, &conversation, &[], &continuation).is_err());
    }

    #[test]
    fn malformed_or_unknown_output_is_rejected() {
        assert!(
            parse_output(&[json!({"type":"function_call", "call_id":"x", "name":"f"})]).is_err()
        );
        assert!(parse_output(&[json!({"type":"future_action", "id":"x"})]).is_err());
    }
}
