use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Serialize)]
pub(crate) struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<RequestMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub(crate) enum RequestMessage {
    System {
        content: String,
    },
    User {
        content: UserContent,
    },
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<WireToolCall>,
    },
    Tool {
        tool_call_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub(crate) enum UserContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
pub(crate) struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct WireToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ToolCallKind,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ToolCallKind {
    Function,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: ToolDefinitionKind,
    pub function: FunctionDefinition,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ToolDefinitionKind {
    Function,
}

#[derive(Debug, Serialize)]
pub(crate) struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ChatCompletionResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Choice {
    pub message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ResponseMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub refusal: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<WireToolCall>,
    #[serde(default)]
    pub function_call: Option<FunctionCall>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorResponse {
    pub error: ApiErrorBody,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ApiErrorBody {
    pub message: String,
}
