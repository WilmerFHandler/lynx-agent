use std::sync::Arc;

use kodkod_core::{Conversation, Provider};
use kodkod_http::{CredentialFuture, CredentialSource, RequestCredentials};
use kodkod_openai::{OpenAiModel, OpenAiResponsesProvider};
use kodkod_providers::{
    CodexAccess, CodexAccessFuture, CodexCredentialSource, CodexProvider, OpenCodeProvider,
    OpenCodeService, open_code_model,
};

struct AppCodexAccount;

impl CodexCredentialSource for AppCodexAccount {
    fn access(&self) -> CodexAccessFuture<'_> {
        Box::pin(async {
            // Refresh and read these values from application-owned storage here.
            Ok(CodexAccess {
                access_token: "current-access-token".into(),
                account_id: "chatgpt-account-id".into(),
            })
        })
    }
}

struct AppHttpCredentials;

impl CredentialSource for AppHttpCredentials {
    fn credentials(&self) -> CredentialFuture<'_> {
        Box::pin(async { RequestCredentials::bearer("current-api-key") })
    }
}

struct AppModel {
    id: String,
    vision: bool,
}

impl OpenAiModel for AppModel {
    fn id(&self) -> &str {
        &self.id
    }

    fn supports_vision(&self) -> bool {
        self.vision
    }
}

fn accepts_any_provider<P: Provider>(_provider: P) {}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let codex =
        CodexProvider::<AppModel>::new(Arc::new(AppCodexAccount))?.with_originator("my-app")?;
    accepts_any_provider(codex);

    let text_codex: CodexProvider = CodexProvider::new(Arc::new(AppCodexAccount))?;
    let text_model = "gpt-text-model".to_owned();
    assert!(!text_codex.supports_vision(&text_model));
    let conversation = Conversation::new();
    drop(text_codex.complete_once(&text_model, &conversation, &[]));

    let go = OpenCodeProvider::with_api_key(
        OpenCodeService::Go,
        "open-code-key",
        "application-session-id",
        "MyApp/1.0",
    )?;
    let model = open_code_model(OpenCodeService::Go, "gpt-5.6-luna")
        .expect("choose a model from open_code_catalog()")
        .clone();
    assert!(go.supports_vision(&model));

    let custom = OpenAiResponsesProvider::<AppModel>::new("https://example.invalid/v1")
        .with_credentials(Arc::new(AppHttpCredentials));
    accepts_any_provider(custom);

    Ok(())
}
