# Kodkod providers

Kodkod keeps model protocols separate from product authentication. The caller
owns login, refresh policy, secure storage, and account selection; provider
crates ask a `kodkod_http::CredentialSource` for HTTP headers immediately before
each request.

Enable the facade's `providers` feature for the reusable named services. A
`CodexProvider` fixes the ChatGPT Codex endpoint and adds the bearer, account,
and originator headers from a fresh `CodexCredentialSource`; the application
still owns login, refresh, account state, and storage. `OpenCodeProvider` fixes
the Go or Zen endpoint, adds its session and user-agent headers, and routes a
model through Responses, Chat Completions, or Messages according to the bundled
reviewed catalog.

See [`kodkod-providers/examples/provider_setup.rs`](kodkod-providers/examples/provider_setup.rs)
for a standalone, compile-checked example covering Codex, OpenCode Go, and a
custom endpoint. The custom protocol adapters remain available as
`kodkod::openai` and `kodkod::anthropic`, while an application with different
transport needs can implement the core `kodkod::Provider` trait directly.

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

Applications can render provider text before a round completes by consuming the
agent task as a stream. Deltas are provisional: replace the preview with the
`AssistantReply` message, which is the authoritative transcript checkpoint and
may differ from the concatenated deltas. Complete-only providers emit no deltas,
so rendering the final reply once also avoids duplicate text. Existing exhaustive
matches over AgentEvent must add the AssistantTextDelta case.

```rust,ignore
let mut preview = String::new();
let mut task = agent.run_turn(&mut conversation, &model, user_message, &control);
while let Some(event) = task.next().await {
    match event? {
        AgentEvent::AssistantTextDelta(delta) => {
            preview.push_str(&delta);
            render_preview(&preview);
        }
        AgentEvent::AssistantReply(message) => {
            preview.clear();
            render_committed(message.content());
        }
        _ => {}
    }
}
// Calling `control.cancel()` makes the task return `AgentError::Cancelled`;
// any displayed preview remains outside the conversation checkpoint.
```

Parallel tool calls produce ToolFinished in completion order; use each result's
call ID rather than relying on the original call order. Dropping or cancelling
a task drops its local in-flight futures, but it cannot reverse an HTTP request
or tool side effect that already happened.

Refresh the reviewed OpenCode snapshot with
`python3 scripts/update-open-code-models.py`, inspect the resulting diff, and
run the provider tests before publishing it. Protocol routing comes from the
official OpenCode endpoint tables and vision metadata comes from models.dev;
unknown models are not guessed.
