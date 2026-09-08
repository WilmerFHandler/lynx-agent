# kodkod

`kodkod` is a small Rust library for running provider-agnostic agent loops
with tool calling support.

The crate defines conversation state, the [`Provider`] trait, tool traits,
tool execution, retry middleware, and structured message/result types.

## Installation

```toml
[dependencies]
kodkod = "0.1"
```

## Example

Providers bring their own model and error types. The agent only asks for vision
support and delegates the actual request to `complete`.

```rust
use std::error::Error;
use std::fmt;
use std::future::ready;

use futures::StreamExt;
use kodkod::{
    Agent, AgentEvent, AssistantMessage, Conversation, Provider, TaskControl, ToolSpec,
};

struct EchoModel;

#[derive(Debug)]
struct EchoError;

impl fmt::Display for EchoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("echo provider failed")
    }
}

impl Error for EchoError {}

struct EchoProvider;

impl Provider for EchoProvider {
    type Model = EchoModel;
    type Error = EchoError;
    type Continuation = ();

    fn create_continuation(&self, _model: &EchoModel) -> Self::Continuation {}

    fn supports_vision(&self, _model: &EchoModel) -> bool {
        false
    }

    fn complete(
        &self,
        _continuation: &Self::Continuation,
        _model: &EchoModel,
        conversation: &Conversation,
        _tools: &[ToolSpec],
    ) -> impl std::future::Future<
        Output = Result<(AssistantMessage, Self::Continuation), Self::Error>,
    > + Send {
        let content = conversation
            .messages()
            .iter()
            .rev()
            .find_map(|message| match message {
                kodkod::Message::User(user) => Some(user.content().to_owned()),
                _ => None,
            })
            .unwrap_or_else(|| "hello".to_owned());

        ready(Ok((AssistantMessage::new(content), ())))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new(EchoProvider);
    let mut conversation = Conversation::new();
    let model = EchoModel;
    let control = TaskControl::new();
    let mut stream = agent.run_turn(
        &mut conversation,
        &model,
        kodkod::UserMessage::new("hello"),
        &control,
    );

    while let Some(event) = stream.next().await {
        if let AgentEvent::Completed(message) = event? {
            assert_eq!(message.content(), "hello");
            break;
        }
    }

    Ok(())
}
```

With retry middleware:

```rust
use kodkod::RetryProvider;

let agent = Agent::new(RetryProvider::new(EchoProvider));
```

`Conversation::estimate_tokens` is a tokenizer-free request-size heuristic.
Inline documents use `Document::try_new(mime, filename, bytes)` and
`UserMessage::with_documents`. They stay in the application-owned conversation
as validated bytes; a provider must opt in to each MIME type with
`Provider::supports_document` before an agent submits them.
`Provider::compact` is a default method: it sends the dropped prefix (including
any earlier summary) to `complete_once` and returns a new conversation that keeps
the system prompt, the latest user plus later steers, and a recent protocol-safe
tail.

## License

MIT
