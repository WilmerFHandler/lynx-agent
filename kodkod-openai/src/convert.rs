use kodkod_core::{
    AssistantMessage, Conversation, Message, ToolCall, ToolExecutorError, ToolResult,
    ToolResultOutcome, ToolSpec,
};
use serde_json::Value;

use super::api::{
    ChatCompletionRequest, ChatCompletionResponse, ContentPart, FunctionDefinition, ImageUrl,
    RequestMessage, ToolCallKind, ToolDefinition, ToolDefinitionKind, UserContent, WireToolCall,
};
use super::error::OpenAiError;

pub(crate) fn build_request(
    model_id: &str,
    conversation: &Conversation,
    tools: &[ToolSpec],
) -> Result<ChatCompletionRequest, OpenAiError> {
    validate_documents(conversation)?;
    let mut messages = Vec::new();

    if let Some(system) = conversation.system_prompt() {
        messages.push(RequestMessage::System {
            content: system.to_owned(),
        });
    }

    let mut message_index = 0;
    while message_index < conversation.messages().len() {
        if matches!(
            conversation.messages()[message_index],
            Message::ToolResult(_)
        ) {
            let mut image_batches = Vec::new();
            while let Some(Message::ToolResult(result)) = conversation.messages().get(message_index)
            {
                messages.push(RequestMessage::Tool {
                    tool_call_id: result.tool_call_id().to_owned(),
                    content: tool_result_content(result),
                });
                if let ToolResultOutcome::Success(output) = result.outcome()
                    && !output.images().is_empty()
                {
                    let mut parts = vec![ContentPart::Text {
                        text: format!("Images returned by tool call '{}'.", result.tool_call_id()),
                    }];
                    parts.extend(output.images().iter().map(|image| ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: image.to_data_url(),
                        },
                    }));
                    image_batches.push(parts);
                }
                message_index += 1;
            }
            for parts in image_batches {
                messages.push(RequestMessage::User {
                    content: UserContent::Parts(parts),
                });
            }
            continue;
        }

        messages.push(convert_message(&conversation.messages()[message_index]));
        message_index += 1;
    }

    Ok(ChatCompletionRequest {
        model: model_id.to_owned(),
        messages,
        tools: tools.iter().map(convert_tool_spec).collect(),
    })
}

pub(crate) fn parse_assistant_message(
    response: ChatCompletionResponse,
) -> Result<AssistantMessage, OpenAiError> {
    let message = response
        .choices
        .into_iter()
        .next()
        .ok_or(OpenAiError::EmptyResponse)?
        .message;

    let content = message.content.or(message.refusal).unwrap_or_default();

    let mut tool_calls: Vec<ToolCall> = message
        .tool_calls
        .into_iter()
        .map(parse_tool_call)
        .collect::<Result<Vec<_>, _>>()?;

    if tool_calls.is_empty()
        && let Some(function_call) = message.function_call
    {
        tool_calls.push(parse_legacy_function_call(function_call)?);
    }

    Ok(AssistantMessage::new(content).with_tool_calls(tool_calls))
}

fn convert_message(message: &Message) -> RequestMessage {
    match message {
        Message::System(system) => RequestMessage::System {
            content: system.content().to_owned(),
        },
        Message::User(user) => {
            let images = user.images();
            let content = if images.is_empty() {
                UserContent::Text(user.content().to_owned())
            } else {
                let mut parts = vec![ContentPart::Text {
                    text: user.content().to_owned(),
                }];
                parts.extend(images.iter().map(|image| ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: image.to_data_url(),
                    },
                }));
                UserContent::Parts(parts)
            };

            RequestMessage::User { content }
        }
        Message::Assistant(assistant) => RequestMessage::Assistant {
            content: if assistant.content().is_empty() {
                None
            } else {
                Some(assistant.content().to_owned())
            },
            tool_calls: assistant
                .tool_calls()
                .iter()
                .map(convert_tool_call)
                .collect(),
        },
        Message::ToolResult(_) => unreachable!("tool result groups are converted by build_request"),
    }
}

fn validate_documents(conversation: &Conversation) -> Result<(), OpenAiError> {
    for message in conversation.messages() {
        if let Message::User(user) = message
            && let Some(document) = user.documents().first()
        {
            return Err(OpenAiError::UnsupportedDocument {
                provider: "OpenAI-compatible Chat Completions",
                mime: document.mime().to_owned(),
            });
        }
    }
    Ok(())
}

fn convert_tool_call(call: &ToolCall) -> WireToolCall {
    WireToolCall {
        id: call.id().to_owned(),
        kind: ToolCallKind::Function,
        function: super::api::FunctionCall {
            name: call.name().to_owned(),
            arguments: serde_json::to_string(call.arguments()).unwrap_or_else(|_| "{}".to_owned()),
        },
    }
}

fn parse_legacy_function_call(
    function_call: super::api::FunctionCall,
) -> Result<ToolCall, OpenAiError> {
    let arguments = parse_function_arguments(&function_call.arguments)?;
    Ok(ToolCall::new(
        "legacy_function_call",
        function_call.name,
        arguments,
    ))
}

fn parse_function_arguments(arguments: &str) -> Result<Value, OpenAiError> {
    if arguments.is_empty() {
        Ok(Value::Object(serde_json::Map::new()))
    } else {
        Ok(serde_json::from_str(arguments)?)
    }
}

fn parse_tool_call(call: WireToolCall) -> Result<ToolCall, OpenAiError> {
    let arguments = parse_function_arguments(&call.function.arguments)?;

    Ok(ToolCall::new(call.id, call.function.name, arguments))
}

fn convert_tool_spec(spec: &ToolSpec) -> ToolDefinition {
    ToolDefinition {
        kind: ToolDefinitionKind::Function,
        function: FunctionDefinition {
            name: spec.name().to_owned(),
            description: spec.description().to_owned(),
            parameters: spec.input_schema().clone(),
        },
    }
}

fn tool_result_content(result: &ToolResult) -> String {
    match result.outcome() {
        ToolResultOutcome::Success(output) => {
            serde_json::to_string(output.value()).unwrap_or_default()
        }
        ToolResultOutcome::Error(ToolExecutorError::UnknownTool(name)) => {
            format!("unknown tool: {name}")
        }
        ToolResultOutcome::Error(ToolExecutorError::Tool(error)) => error.message().to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kodkod_core::{Image, UserMessage};
    use serde_json::json;

    #[test]
    fn builds_messages_with_system_prompt_and_tool_result() {
        let mut conversation = Conversation::new().with_system_prompt("You are helpful.");
        conversation.push_user_message(UserMessage::new("hi"));
        conversation.push_message(Message::Assistant(
            AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                "call_1",
                "lookup",
                json!({"q": "weather"}),
            )]),
        ));
        conversation.push_message(Message::ToolResult(ToolResult::success(
            "call_1",
            json!({"temp": 72}),
        )));

        let request = build_request("gpt-4o", &conversation, &[]).unwrap();
        assert_eq!(request.model, "gpt-4o");
        assert_eq!(request.messages.len(), 4);

        let serialized = serde_json::to_value(&request).expect("request should serialize");
        let tool_message = &serialized["messages"][3];
        assert_eq!(tool_message["role"], "tool");
        assert_eq!(tool_message["tool_call_id"], "call_1");
        assert_eq!(tool_message["content"], r#"{"temp":72}"#);

        let assistant_message = &serialized["messages"][2];
        assert_eq!(
            assistant_message["tool_calls"][0]["function"]["arguments"],
            r#"{"q":"weather"}"#
        );
    }

    #[test]
    fn builds_user_message_with_vision_parts() {
        let mut conversation = Conversation::new();
        conversation.push_user_message(
            UserMessage::new("what is this?").with_images(vec![Image::new("image/png", b"abc")]),
        );

        let request = build_request("gpt-4o", &conversation, &[]).unwrap();
        let serialized = serde_json::to_value(&request).expect("request should serialize");
        let parts = &serialized["messages"][0]["content"];

        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert!(
            parts[1]["image_url"]["url"]
                .as_str()
                .expect("url")
                .starts_with("data:image/png;base64,")
        );
    }

    #[test]
    fn places_tool_images_in_a_following_vision_message() {
        use kodkod_core::ToolOutput;

        let mut conversation = Conversation::new();
        conversation.push_message(Message::Assistant(
            AssistantMessage::new("").with_tool_calls(vec![
                ToolCall::new("call_1", "view_image", json!({"path": "sample.png"})),
                ToolCall::new("call_2", "lookup", json!({"query": "sample"})),
            ]),
        ));
        conversation.push_message(Message::ToolResult(ToolResult::success(
            "call_1",
            ToolOutput::new(json!({"path": "sample.png"}))
                .with_images(vec![Image::new("image/png", b"abc")]),
        )));
        conversation.push_message(Message::ToolResult(ToolResult::success(
            "call_2",
            ToolOutput::new(json!({"path": "other.png"}))
                .with_images(vec![Image::new("image/png", b"def")]),
        )));

        let request = build_request("gpt-4o", &conversation, &[]).unwrap();
        let serialized = serde_json::to_value(request).unwrap();
        assert_eq!(serialized["messages"][0]["role"], "assistant");
        assert_eq!(serialized["messages"][1]["role"], "tool");
        assert_eq!(
            serialized["messages"][1]["content"],
            r#"{"path":"sample.png"}"#
        );
        assert_eq!(serialized["messages"][2]["role"], "tool");
        assert_eq!(serialized["messages"][3]["role"], "user");
        assert_eq!(serialized["messages"][4]["role"], "user");
        assert_eq!(
            serialized["messages"][3]["content"][0]["text"],
            "Images returned by tool call 'call_1'."
        );
        assert_eq!(serialized["messages"][3]["content"][1]["type"], "image_url");
        assert!(
            serialized["messages"][3]["content"][1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
        assert_eq!(
            serialized["messages"][4]["content"][0]["text"],
            "Images returned by tool call 'call_2'."
        );
    }

    #[test]
    fn parses_refusal_when_content_is_null() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "choices": [{
                "message": {
                    "content": null,
                    "refusal": "I can't help with that.",
                    "tool_calls": []
                }
            }]
        }))
        .expect("response should deserialize");

        let message = parse_assistant_message(response).expect("assistant message should parse");
        assert_eq!(message.content(), "I can't help with that.");
    }

    #[test]
    fn parses_legacy_function_call_response() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "choices": [{
                "message": {
                    "content": null,
                    "function_call": {
                        "name": "lookup",
                        "arguments": "{\"q\":\"weather\"}"
                    }
                }
            }]
        }))
        .expect("response should deserialize");

        let message = parse_assistant_message(response).expect("assistant message should parse");
        assert_eq!(message.tool_calls().len(), 1);
        assert_eq!(message.tool_calls()[0].name(), "lookup");
    }

    #[test]
    fn parses_assistant_message_with_tool_calls() {
        let response: ChatCompletionResponse = serde_json::from_value(json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"q\":\"weather\"}"
                        }
                    }]
                }
            }]
        }))
        .expect("response should deserialize");

        let message = parse_assistant_message(response).expect("assistant message should parse");
        assert_eq!(message.content(), "");
        assert_eq!(message.tool_calls().len(), 1);
        assert_eq!(message.tool_calls()[0].name(), "lookup");
        assert_eq!(message.tool_calls()[0].arguments()["q"], "weather");
    }
}
