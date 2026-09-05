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

Refresh the reviewed OpenCode snapshot with
`python3 scripts/update-open-code-models.py`, inspect the resulting diff, and
run the provider tests before publishing it. Protocol routing comes from the
official OpenCode endpoint tables and vision metadata comes from models.dev;
unknown models are not guessed.
