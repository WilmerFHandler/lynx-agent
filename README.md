# Kodkod providers

Kodkod keeps model protocols separate from product authentication. The caller
owns login, refresh policy, secure storage, and account selection; provider
crates ask a `kodkod_http::CredentialSource` for HTTP headers immediately before
each request.

```rust
use std::sync::Arc;
use kodkod::http::{CredentialFuture, CredentialSource, RequestCredentials};
use kodkod::openai::OpenAiResponsesProvider;

struct AppCredentials;

impl CredentialSource for AppCredentials {
    fn credentials(&self) -> CredentialFuture<'_> {
        Box::pin(async {
            // Obtain a currently valid token from the application-owned session.
            RequestCredentials::bearer("current-token")
        })
    }
}

let provider = OpenAiResponsesProvider::<MyModel>::new("https://api.openai.com/v1")
    .with_credentials(Arc::new(AppCredentials));
# struct MyModel;
```

`kodkod-openai` provides the existing OpenAI-compatible Chat Completions
adapter and an HTTP/SSE Responses adapter. Responses requests use stateless
replay with `stream: true`, `store: false`, and encrypted reasoning output;
opaque response items are checkpointed and replayed only when the model and
normalized conversation still match. It returns on the terminal SSE event and
does not implement the newer WebSocket-only steering or asynchronous tool-result
transport.

`kodkod-anthropic` provides the native Messages protocol used by compatible
gateways. It supports text, images, client tools, parallel tool results, and
opaque thinking-block replay. The provider validates `stop_reason` before
returning tool calls. Both stateful adapters reject continuations after the
caller edits or truncates their bound history; create a fresh continuation in
that case.

No provider opens a browser, stores credentials, or performs an OAuth flow.
