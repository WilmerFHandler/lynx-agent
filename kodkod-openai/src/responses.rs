use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use futures_util::StreamExt;
use kodkod_core::{
    AssistantMessage, Conversation, Message, Provider, ProviderEvent, ProviderStream, ToolCall,
    ToolExecutorError, ToolResult, ToolResultOutcome, ToolSpec,
};
use kodkod_http::{SseDecoder, SseEvent};
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

    fn supports_document(&self, model: &M, mime: &str) -> bool {
        supports_inline_document(mime) && (mime != "application/pdf" || model.supports_vision())
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
        let mut stream = self.complete_stream(continuation, model, conversation, tools);
        while let Some(event) = stream.next().await {
            if let ProviderEvent::Completed(message, continuation) = event? {
                return Ok((message, continuation));
            }
        }
        Err(OpenAiError::Protocol(
            "stream ended without a terminal event".into(),
        ))
    }

    fn complete_stream<'a>(
        &'a self,
        continuation: &'a Self::Continuation,
        model: &'a M,
        conversation: &'a Conversation,
        tools: &'a [ToolSpec],
    ) -> ProviderStream<'a, Self::Continuation, Self::Error> {
        Box::pin(async_stream::try_stream! {
        validate_documents(conversation, model.supports_vision())?;
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
            Err(OpenAiError::Api {
                status: status.as_u16(),
                message,
            })?;
            unreachable!("error propagation exits the stream");
        }

            let mut bytes = response.bytes_stream();
            let mut decoder = SseDecoder::default();
            let mut output = OutputAccumulator::default();
            while let Some(chunk) = bytes.next().await {
                for event in decoder.push(&chunk?).map_err(OpenAiError::Protocol)? {
                    if let Some(delta) = response_text_delta(&event)? {
                        yield ProviderEvent::TextDelta(delta);
                    }
                    if let Some(terminal) = handle_sse_event(event, &mut output)? {
                        let final_output = terminal
                            .get("output")
                            .and_then(Value::as_array)
                            .cloned()
                            .ok_or_else(|| OpenAiError::Protocol("completed response missing output".into()))?;
                        let assistant = parse_output(&final_output)?;
                        let mut next = continuation.clone();
                        next.checkpoints.push(ResponseCheckpoint {
                            model: model.id().to_owned(),
                            assistant_message_index: conversation.messages().len(),
                            assistant: assistant.clone(),
                            output: final_output,
                        });
                        yield ProviderEvent::Completed(assistant, next);
                        return;
                    }
                }
            }
            decoder.finish().map_err(OpenAiError::Protocol)?;
            Err(OpenAiError::Protocol("stream ended without a terminal event".into()))?;
        })
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
    validate_document_mime_types(conversation)?;
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
            content.extend(user.documents().iter().map(|document| {
                json!({
                    "type": "input_file",
                    "filename": document.filename(),
                    "file_data": document.to_data_url(),
                })
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

fn supports_inline_document(mime: &str) -> bool {
    matches!(
        mime,
        "application/pdf"
            | "text/plain"
            | "text/markdown"
            | "text/csv"
            | "text/tab-separated-values"
            | "text/html"
            | "text/xml"
            | "application/json"
            | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            | "application/msword"
            | "application/rtf"
            | "text/rtf"
            | "application/vnd.oasis.opendocument.text"
            | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            | "application/vnd.ms-powerpoint"
            | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            | "application/vnd.ms-excel"
            | "application/csv"
    )
}

const MAX_INLINE_DOCUMENT_BYTES: usize = 50 * 1024 * 1024;

fn validate_documents(
    conversation: &Conversation,
    supports_vision: bool,
) -> Result<(), OpenAiError> {
    validate_document_mime_types(conversation)?;
    let mut total_bytes = 0usize;
    for message in conversation.messages() {
        if let Message::User(user) = message {
            if !supports_vision
                && user
                    .documents()
                    .iter()
                    .any(|document| document.mime() == "application/pdf")
            {
                return Err(OpenAiError::UnsupportedDocument {
                    provider: "OpenAI Responses",
                    mime: "application/pdf".to_owned(),
                });
            }
            for document in user.documents() {
                total_bytes = total_bytes.saturating_add(document.data().len());
                if document.data().len() >= MAX_INLINE_DOCUMENT_BYTES
                    || total_bytes > MAX_INLINE_DOCUMENT_BYTES
                {
                    return Err(OpenAiError::DocumentLimitExceeded {
                        provider: "OpenAI Responses",
                        limit_bytes: MAX_INLINE_DOCUMENT_BYTES,
                    });
                }
            }
        }
    }
    Ok(())
}

fn validate_document_mime_types(conversation: &Conversation) -> Result<(), OpenAiError> {
    for message in conversation.messages() {
        if let Message::User(user) = message
            && let Some(document) = user
                .documents()
                .iter()
                .find(|document| !supports_inline_document(document.mime()))
        {
            return Err(OpenAiError::UnsupportedDocument {
                provider: "OpenAI Responses",
                mime: document.mime().to_owned(),
            });
        }
    }
    Ok(())
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

const MAX_ERROR_BODY_BYTES: usize = 64 * 1024;
const MAX_ASSEMBLED_OUTPUT_BYTES: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_ITEMS: usize = 512;

#[cfg(test)]
fn process_sse_chunk(
    decoder: &mut SseDecoder,
    output: &mut OutputAccumulator,
    chunk: &[u8],
) -> Result<Option<Value>, OpenAiError> {
    for event in decoder.push(chunk).map_err(OpenAiError::Protocol)? {
        if let Some(terminal) = handle_sse_event(event, output)? {
            return Ok(Some(terminal));
        }
    }
    Ok(None)
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
struct OutputAccumulator {
    items: BTreeMap<usize, Value>,
    serialized_bytes: usize,
}

impl OutputAccumulator {
    fn insert(&mut self, payload: &Value) -> Result<(), OpenAiError> {
        let index = payload
            .get("output_index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| {
                OpenAiError::Protocol("output_item.done missing valid output_index".into())
            })?;
        if index >= MAX_OUTPUT_ITEMS {
            return Err(OpenAiError::Protocol(format!(
                "output_index {index} exceeds item limit"
            )));
        }
        let item = payload
            .get("item")
            .cloned()
            .ok_or_else(|| OpenAiError::Protocol("output_item.done missing item".into()))?;
        if self.items.contains_key(&index) {
            return Err(OpenAiError::Protocol(format!(
                "duplicate output_item.done index {index}"
            )));
        }
        if let Some(id) = item.get("id").and_then(Value::as_str)
            && self
                .items
                .values()
                .any(|existing| existing.get("id").and_then(Value::as_str) == Some(id))
        {
            return Err(OpenAiError::Protocol(format!(
                "duplicate output item id at index {index}"
            )));
        }
        let bytes = serde_json::to_vec(&item)?.len();
        self.serialized_bytes = self
            .serialized_bytes
            .checked_add(bytes)
            .filter(|bytes| *bytes <= MAX_ASSEMBLED_OUTPUT_BYTES)
            .ok_or_else(|| {
                OpenAiError::Protocol("assembled response output exceeded size limit".into())
            })?;
        self.items.insert(index, item);
        Ok(())
    }

    fn validate_final(&self, final_output: &[Value]) -> Result<(), OpenAiError> {
        for (&index, streamed) in &self.items {
            let final_item = final_output.get(index).ok_or_else(|| {
                OpenAiError::Protocol(format!(
                    "completed output missing streamed item at index {index}"
                ))
            })?;
            validate_item_identity(index, streamed, final_item)?;
        }
        Ok(())
    }

    fn take_ordered(&mut self) -> Result<Vec<Value>, OpenAiError> {
        for (expected, actual) in self.items.keys().copied().enumerate() {
            if expected != actual {
                return Err(OpenAiError::Protocol(format!(
                    "response output indices are not contiguous: expected {expected}, got {actual}"
                )));
            }
        }
        self.serialized_bytes = 0;
        Ok(std::mem::take(&mut self.items).into_values().collect())
    }
}

fn validate_item_identity(
    index: usize,
    streamed: &Value,
    final_item: &Value,
) -> Result<(), OpenAiError> {
    for field in ["type", "id"] {
        if streamed.get(field) != final_item.get(field) {
            return Err(OpenAiError::Protocol(format!(
                "completed output {field} disagrees with streamed item at index {index}"
            )));
        }
    }
    if streamed.get("type").and_then(Value::as_str) == Some("function_call")
        && streamed.get("call_id") != final_item.get("call_id")
    {
        return Err(OpenAiError::Protocol(format!(
            "completed output call_id disagrees with streamed item at index {index}"
        )));
    }
    Ok(())
}

fn handle_sse_event(
    event: SseEvent,
    output: &mut OutputAccumulator,
) -> Result<Option<Value>, OpenAiError> {
    if event.data == "[DONE]" {
        return Ok(None);
    }
    let payload: Value = serde_json::from_str(&event.data)?;
    let kind = payload
        .get("type")
        .and_then(Value::as_str)
        .or(event.event_name.as_deref())
        .unwrap_or_default();
    match kind {
        "response.output_item.done" => {
            output.insert(&payload)?;
            Ok(None)
        }
        "response.completed" => {
            let mut response = payload
                .get("response")
                .cloned()
                .ok_or_else(|| OpenAiError::Protocol("completed event missing response".into()))?;
            let final_output = response
                .get("output")
                .and_then(Value::as_array)
                .ok_or_else(|| OpenAiError::Protocol("completed response missing output".into()))?;
            if final_output.is_empty() && !output.items.is_empty() {
                response["output"] = Value::Array(output.take_ordered()?);
            } else {
                output.validate_final(final_output)?;
            }
            Ok(Some(response))
        }
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

fn response_text_delta(event: &SseEvent) -> Result<Option<String>, OpenAiError> {
    if event.data == "[DONE]" {
        return Ok(None);
    }
    let payload: Value = serde_json::from_str(&event.data)?;
    let kind = payload
        .get("type")
        .and_then(Value::as_str)
        .or(event.event_name.as_deref())
        .unwrap_or_default();
    match kind {
        "response.output_text.delta" | "response.refusal.delta" => payload
            .get("delta")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .map(Some)
            .ok_or_else(|| OpenAiError::Protocol(format!("{kind} missing delta"))),
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
    use kodkod_core::{Document, Image, UserMessage};
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

    fn output_item_done(index: usize, item: Value) -> String {
        format!(
            "event: response.output_item.done\ndata: {}\n\n",
            json!({
                "type": "response.output_item.done",
                "output_index": index,
                "item": item
            })
        )
    }

    fn parse_complete_stream(body: &[u8]) -> Result<Value, OpenAiError> {
        let mut decoder = SseDecoder::default();
        let mut output = OutputAccumulator::default();
        if let Some(response) = process_sse_chunk(&mut decoder, &mut output, body)? {
            return Ok(response);
        }
        decoder.finish().map_err(OpenAiError::Protocol)?;
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
    fn assembles_done_items_when_codex_completed_output_is_empty() {
        let reasoning = json!({
            "id":"rs_1", "type":"reasoning", "encrypted_content":"opaque"
        });
        let call = json!({
            "id":"fc_1", "type":"function_call", "call_id":"call_1",
            "name":"lookup", "arguments":"{\"q\":\"x\"}"
        });
        let message = json!({
            "id":"msg_1", "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"working"}]
        });
        let body = format!(
            "event: response.output_item.added\ndata: {}\n\n{}{}{}{}",
            json!({
                "type":"response.output_item.added", "output_index":0,
                "item":{"id":"rs_1", "type":"reasoning"}
            }),
            output_item_done(0, reasoning),
            output_item_done(1, call),
            output_item_done(2, message),
            completed(json!([])),
        );
        let response = parse_complete_stream(body.as_bytes()).unwrap();
        let output = response["output"].as_array().unwrap().clone();
        let assistant = parse_output(&output).unwrap();
        let continuation = ResponsesContinuation {
            checkpoints: vec![ResponseCheckpoint {
                model: "gpt-test".into(),
                assistant_message_index: 1,
                assistant: assistant.clone(),
                output,
            }],
        };

        assert_eq!(assistant.content(), "working");
        assert_eq!(assistant.tool_calls()[0].name(), "lookup");
        let serialized = serde_json::to_value(continuation).unwrap();
        let output = serialized["checkpoints"][0]["output"].as_array().unwrap();
        assert_eq!(output[0]["encrypted_content"], "opaque");
        assert_eq!(output[1]["call_id"], "call_1");
        assert_eq!(output[2]["id"], "msg_1");
    }

    #[test]
    fn completed_output_is_authoritative_and_may_add_metadata() {
        let streamed = json!({
            "id":"msg_1", "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"done"}]
        });
        let final_item = json!({
            "id":"msg_1", "type":"message", "role":"assistant",
            "status":"completed", "content":[{"type":"output_text", "text":"final"}]
        });
        let response = parse_complete_stream(
            format!(
                "{}{}",
                output_item_done(0, streamed),
                completed(json!([final_item]))
            )
            .as_bytes(),
        )
        .unwrap();
        assert_eq!(response["output"][0]["id"], "msg_1");
        assert_eq!(response["output"][0]["status"], "completed");
        assert_eq!(response["output"][0]["content"][0]["text"], "final");
    }

    #[test]
    fn rejects_duplicate_gapped_or_inconsistent_done_items() {
        let first = json!({
            "id":"msg_1", "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"one"}]
        });
        let second = json!({
            "id":"msg_2", "type":"message", "role":"assistant",
            "content":[{"type":"output_text", "text":"two"}]
        });

        let duplicate = format!(
            "{}{}{}",
            output_item_done(0, first.clone()),
            output_item_done(0, second.clone()),
            completed(json!([]))
        );
        assert!(
            matches!(parse_complete_stream(duplicate.as_bytes()), Err(OpenAiError::Protocol(message)) if message.contains("duplicate"))
        );

        let gapped = format!(
            "{}{}",
            output_item_done(1, first.clone()),
            completed(json!([]))
        );
        assert!(
            matches!(parse_complete_stream(gapped.as_bytes()), Err(OpenAiError::Protocol(message)) if message.contains("contiguous"))
        );

        let inconsistent = format!(
            "{}{}",
            output_item_done(0, first),
            completed(json!([second]))
        );
        assert!(
            matches!(parse_complete_stream(inconsistent.as_bytes()), Err(OpenAiError::Protocol(message)) if message.contains("id disagrees"))
        );
    }

    #[test]
    fn rejects_truncation_after_a_done_item() {
        let stream = format!(
            "{}event: response.completed\ndata: {{",
            output_item_done(
                0,
                json!({
                    "id":"msg_1", "type":"message", "role":"assistant",
                    "content":[{"type":"output_text", "text":"partial"}]
                })
            )
        );
        assert!(
            matches!(parse_complete_stream(stream.as_bytes()), Err(OpenAiError::Protocol(message)) if message.contains("truncated"))
        );
    }

    #[test]
    fn bounds_accumulated_output_items_and_bytes() {
        let mut output = OutputAccumulator::default();
        let too_many = json!({
            "output_index": MAX_OUTPUT_ITEMS,
            "item": {"id":"msg_limit", "type":"message"}
        });
        assert!(
            matches!(output.insert(&too_many), Err(OpenAiError::Protocol(message)) if message.contains("item limit"))
        );

        let oversized = json!({
            "output_index": 0,
            "item": {
                "id":"msg_large", "type":"message",
                "content":"x".repeat(MAX_ASSEMBLED_OUTPUT_BYTES)
            }
        });
        assert!(
            matches!(output.insert(&oversized), Err(OpenAiError::Protocol(message)) if message.contains("size limit"))
        );
    }

    #[test]
    fn incomplete_event_wins_over_accumulated_done_items() {
        let stream = format!(
            "{}event: response.incomplete\ndata: {}\n\n",
            output_item_done(
                0,
                json!({
                    "id":"msg_1", "type":"message", "role":"assistant",
                    "content":[{"type":"output_text", "text":"partial"}]
                })
            ),
            json!({
                "type":"response.incomplete",
                "response":{"incomplete_details":{"reason":"max_output_tokens"}}
            })
        );
        assert!(matches!(
            parse_complete_stream(stream.as_bytes()),
            Err(OpenAiError::Incomplete { reason }) if reason == "max_output_tokens"
        ));
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
            .find_map(|event| handle_sse_event(event, &mut OutputAccumulator::default()).unwrap())
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
        assert!(matches!(decoder.finish(), Err(message) if message.contains("truncated")));

        let terminal = completed(json!([{
            "type":"message", "role":"assistant", "content":[{"type":"output_text", "text":"done"}]
        }]));
        let mut decoder = SseDecoder::default();
        let events = decoder.push(terminal.as_bytes()).unwrap();
        assert!(events.into_iter().any(|event| {
            handle_sse_event(event, &mut OutputAccumulator::default())
                .unwrap()
                .is_some()
        }));
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

    #[test]
    fn serializes_inline_document_as_responses_input_file() {
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("summarize").with_documents(vec![
            Document::try_new("application/pdf", "notes.pdf", b"%PDF").unwrap(),
        ]));

        let request = build_request(
            "gpt-test",
            None,
            &conversation,
            &[],
            &ResponsesContinuation::default(),
        )
        .unwrap();

        assert_eq!(
            request["input"][0]["content"][1],
            json!({
                "type": "input_file",
                "filename": "notes.pdf",
                "file_data": "data:application/pdf;base64,JVBERg==",
            })
        );
    }

    #[tokio::test]
    async fn rejects_unsupported_documents_before_a_responses_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        let provider = OpenAiResponsesProvider::<TestModel>::new(format!("{}/v1", server.uri()));
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("read").with_documents(vec![
            Document::try_new("application/zip", "notes.zip", b"PK").unwrap(),
        ]));

        let error = provider
            .complete_once(&TestModel, &conversation, &[])
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            OpenAiError::UnsupportedDocument { provider: "OpenAI Responses", mime }
                if mime == "application/zip"
        ));
        server.verify().await;
    }
}
