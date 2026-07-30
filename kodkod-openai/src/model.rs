/// Metadata for a model served by an OpenAI-compatible API.
pub trait OpenAiModel: Sync {
    fn id(&self) -> &str;
    fn supports_vision(&self) -> bool;
}
