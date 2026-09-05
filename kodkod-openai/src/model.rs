/// Metadata for a model served by an OpenAI-compatible API.
pub trait OpenAiModel: Sync {
    fn id(&self) -> &str;
    fn supports_vision(&self) -> bool;
}

impl OpenAiModel for String {
    fn id(&self) -> &str {
        self
    }

    fn supports_vision(&self) -> bool {
        false
    }
}

impl OpenAiModel for str {
    fn id(&self) -> &str {
        self
    }

    fn supports_vision(&self) -> bool {
        false
    }
}
