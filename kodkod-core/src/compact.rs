use std::collections::HashSet;
use std::error::Error as StdError;
use std::fmt;

use crate::{AssistantMessage, Conversation, Message, Provider, ToolResultOutcome, UserMessage};

pub const DEFAULT_KEEP_TAIL_TOKENS: u64 = 20_000;

const COMPACTION_SYSTEM: &str = include_str!("compact/prompt.txt");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactOptions {
    pub keep_tail_tokens: u64,
}

impl Default for CompactOptions {
    fn default() -> Self {
        Self {
            keep_tail_tokens: DEFAULT_KEEP_TAIL_TOKENS,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CompactError<E> {
    NothingToCompact,
    EmptySummary,
    Provider(E),
}

impl<E: fmt::Display> fmt::Display for CompactError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NothingToCompact => f.write_str("conversation has nothing to compact"),
            Self::EmptySummary => f.write_str("compaction model returned an empty summary"),
            Self::Provider(error) => write!(f, "{error}"),
        }
    }
}

impl<E: StdError + 'static> StdError for CompactError<E> {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::NothingToCompact | Self::EmptySummary => None,
        }
    }
}

pub(crate) async fn run<P: Provider + ?Sized>(
    provider: &P,
    model: &P::Model,
    conversation: &Conversation,
    options: CompactOptions,
) -> Result<Conversation, CompactError<P::Error>> {
    let kept = plan(conversation.messages(), options.keep_tail_tokens)
        .ok_or(CompactError::NothingToCompact)?;
    let request = summarizer_conversation(conversation, &kept);
    let message = provider
        .complete_once(model, &request, &[])
        .await
        .map_err(CompactError::Provider)?;
    let summary = message.content().trim();
    if summary.is_empty() {
        return Err(CompactError::EmptySummary);
    }
    Ok(splice(conversation, &kept, AssistantMessage::new(summary)))
}

fn plan(messages: &[Message], keep_tail_tokens: u64) -> Option<Vec<usize>> {
    if messages.is_empty() {
        return None;
    }

    let mut tail_starts_at = messages.len();
    let mut tail_tokens = 0_u64;
    while tail_starts_at > 0 {
        let next = messages[tail_starts_at - 1].estimate_tokens();
        if next > keep_tail_tokens.saturating_sub(tail_tokens) {
            break;
        }
        tail_tokens += next;
        tail_starts_at -= 1;
    }
    tail_starts_at = protocol_safe_tail_start(messages, tail_starts_at);

    let mut kept = pinned_indices(messages);
    kept.extend(tail_starts_at..messages.len());
    kept.sort_unstable();
    kept.dedup();

    if kept.len() == messages.len() {
        None
    } else {
        Some(kept)
    }
}

fn pinned_indices(messages: &[Message]) -> Vec<usize> {
    let Some(latest_user) = messages
        .iter()
        .rposition(|message| matches!(message, Message::User(user) if !user.steered()))
    else {
        return Vec::new();
    };

    messages
        .iter()
        .enumerate()
        .skip(latest_user)
        .filter_map(|(index, message)| match message {
            Message::User(user) if index == latest_user || user.steered() => Some(index),
            _ => None,
        })
        .collect()
}

fn protocol_safe_tail_start(messages: &[Message], mut start: usize) -> usize {
    while let Some(index) = first_protocol_violation(messages, start) {
        start = index + 1;
    }
    start
}

fn first_protocol_violation(messages: &[Message], start: usize) -> Option<usize> {
    let mut index = start;
    while index < messages.len() {
        match &messages[index] {
            Message::ToolResult(_) => return Some(index),
            Message::Assistant(assistant) if !assistant.tool_calls().is_empty() => {
                let mut pending = HashSet::new();
                for call in assistant.tool_calls() {
                    if !pending.insert(call.id()) {
                        return Some(index);
                    }
                }

                let call_index = index;
                index += 1;
                while !pending.is_empty() {
                    let Some(Message::ToolResult(result)) = messages.get(index) else {
                        return Some(call_index);
                    };
                    if !pending.remove(result.tool_call_id()) {
                        return Some(index);
                    }
                    index += 1;
                }
            }
            _ => index += 1,
        }
    }
    None
}

fn splice(conversation: &Conversation, kept: &[usize], summary: AssistantMessage) -> Conversation {
    let mut compacted = match conversation.system_prompt() {
        Some(prompt) => Conversation::new().with_system_prompt(prompt),
        None => Conversation::new(),
    };
    compacted.replace_messages(splice_messages(conversation.messages(), kept, summary));
    compacted
}

fn splice_messages(
    messages: &[Message],
    kept: &[usize],
    summary: AssistantMessage,
) -> Vec<Message> {
    let first_dropped = (0..messages.len()).find(|index| kept.binary_search(index).is_err());
    let mut out = Vec::with_capacity(kept.len() + 1);
    for (index, message) in messages.iter().enumerate() {
        if Some(index) == first_dropped {
            out.push(Message::Assistant(summary.clone()));
        }
        if kept.binary_search(&index).is_ok() {
            out.push(message.clone());
        }
    }
    if first_dropped.is_none() {
        out.insert(0, Message::Assistant(summary));
    }
    out
}

fn summarizer_conversation(conversation: &Conversation, kept: &[usize]) -> Conversation {
    let messages = conversation.messages();
    let mut body = String::new();

    if let Some(system) = conversation.system_prompt() {
        body.push_str("The agent system prompt is:\n");
        body.push_str(system);
        body.push_str("\n\n");
    }

    body.push_str("Conversation history to summarize:\n");
    let mut dropped_any = false;
    for (index, message) in messages.iter().enumerate() {
        if kept.binary_search(&index).is_ok() {
            continue;
        }
        dropped_any = true;
        body.push_str(&serialize_message(message));
        body.push('\n');
    }
    if !dropped_any {
        body.push_str("(none)\n");
    }

    let pinned = pinned_indices(messages);
    if !pinned.is_empty() {
        body.push_str(
            "\nThe following user instructions remain verbatim outside the summary. \
             Use them only to interpret the summarized work; do not claim they were \
             completed unless the history shows that.\n",
        );
        for index in pinned {
            body.push_str(&serialize_message(&messages[index]));
            body.push('\n');
        }
    }

    let mut request = Conversation::new().with_system_prompt(COMPACTION_SYSTEM.trim());
    request.push_user_message(UserMessage::new(body));
    request
}

fn serialize_message(message: &Message) -> String {
    match message {
        Message::System(system) => format!("system: {}", system.content()),
        Message::User(user) if user.steered() => format!("steer: {}", user.content()),
        Message::User(user) => format!("user: {}", user.content()),
        Message::Assistant(assistant) => {
            let mut out = format!("assistant: {}", assistant.content());
            for call in assistant.tool_calls() {
                let args = serde_json::to_string(call.arguments()).unwrap_or_else(|_| "{}".into());
                out.push('\n');
                out.push_str(&format!(
                    "assistant tool_call id={} name={} args={args}",
                    call.id(),
                    call.name(),
                ));
            }
            out
        }
        Message::ToolResult(result) => match result.outcome() {
            ToolResultOutcome::Success(output) => {
                let value = serde_json::to_string(output.value()).unwrap_or_else(|_| "null".into());
                format!("tool_result id={} success={value}", result.tool_call_id())
            }
            ToolResultOutcome::Error(error) => {
                format!("tool_result id={} error={error}", result.tool_call_id())
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Provider, ToolCall, ToolResult, ToolSpec};
    use serde_json::json;
    use std::error::Error;
    use std::future::{Future, ready};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[test]
    fn empty_conversation_has_nothing_to_compact() {
        assert_eq!(plan(&[], 20_000), None);
    }

    #[test]
    fn nothing_dropped_when_budget_covers_a_safe_conversation() {
        let messages = vec![
            Message::User(UserMessage::new("hello")),
            Message::Assistant(AssistantMessage::new("hi")),
        ];
        assert_eq!(plan(&messages, u64::MAX), None);
    }

    #[test]
    fn keeps_a_recent_token_bounded_suffix_and_latest_user() {
        let messages = vec![
            Message::User(UserMessage::new("old")),
            Message::Assistant(AssistantMessage::new("x".repeat(300))),
            Message::User(UserMessage::new("latest instructions")),
            Message::Assistant(AssistantMessage::new("recent answer")),
        ];
        let budget = messages[3].estimate_tokens();
        assert_eq!(plan(&messages, budget), Some(vec![2, 3]));
    }

    #[test]
    fn preserves_latest_ordinary_user_and_steers_when_tail_is_empty() {
        let messages = vec![
            Message::User(UserMessage::new("do this exactly")),
            Message::Assistant(AssistantMessage::new("x".repeat(300))),
            Message::User(UserMessage::new("and this").with_steered(true)),
            Message::Assistant(AssistantMessage::new("y".repeat(300))),
        ];
        assert_eq!(plan(&messages, 1), Some(vec![0, 2]));
    }

    #[test]
    fn does_not_start_a_tail_with_an_orphaned_tool_result() {
        let messages = vec![
            Message::User(UserMessage::new("run it")),
            Message::Assistant(
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "call-1",
                    "read_file",
                    json!({"path": "large"}),
                )]),
            ),
            Message::ToolResult(ToolResult::success("call-1", json!("result"))),
            Message::Assistant(AssistantMessage::new("done")),
        ];
        let budget = messages[2].estimate_tokens() + messages[3].estimate_tokens();
        assert_eq!(plan(&messages, budget), Some(vec![0, 3]));
    }

    #[test]
    fn omits_incomplete_tool_exchanges_from_the_verbatim_tail() {
        let messages = vec![
            Message::User(UserMessage::new("first request")),
            Message::Assistant(
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "unfinished",
                    "read_file",
                    json!({}),
                )]),
            ),
            Message::User(UserMessage::new("latest request")),
            Message::Assistant(AssistantMessage::new("completed answer")),
        ];
        assert_eq!(plan(&messages, u64::MAX), Some(vec![2, 3]));
    }

    #[test]
    fn does_not_duplicate_last_user_already_in_the_tail() {
        let messages = vec![
            Message::User(UserMessage::new("old")),
            Message::Assistant(AssistantMessage::new("x".repeat(300))),
            Message::User(UserMessage::new("new")),
            Message::Assistant(AssistantMessage::new("recent")),
        ];
        let budget = messages[2].estimate_tokens() + messages[3].estimate_tokens();
        let kept = plan(&messages, budget).unwrap();
        assert_eq!(kept, vec![2, 3]);
        let spliced = splice_messages(&messages, &kept, AssistantMessage::new("SUM"));
        assert_eq!(
            spliced
                .iter()
                .filter(|message| matches!(message, Message::User(user) if user.content() == "new"))
                .count(),
            1
        );
    }

    #[test]
    fn inserts_summary_at_first_dropped_index() {
        let messages = vec![
            Message::User(UserMessage::new("old")),
            Message::Assistant(AssistantMessage::new("old answer")),
            Message::User(UserMessage::new("new")),
        ];
        let spliced = splice_messages(&messages, &[2], AssistantMessage::new("SUM"));
        assert!(matches!(&spliced[0], Message::Assistant(a) if a.content() == "SUM"));
        assert!(matches!(&spliced[1], Message::User(u) if u.content() == "new"));
        assert_eq!(spliced.len(), 2);
    }

    #[test]
    fn mid_exchange_steer_inserts_summary_between_user_and_steer() {
        let messages = vec![
            Message::User(UserMessage::new("latest")),
            Message::Assistant(
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "call-1",
                    "read",
                    json!({}),
                )]),
            ),
            Message::User(UserMessage::new("nudge").with_steered(true)),
            Message::ToolResult(ToolResult::success("call-1", json!("ok"))),
            Message::Assistant(AssistantMessage::new("done")),
        ];
        let kept = plan(&messages, u64::MAX).unwrap();
        assert_eq!(kept, vec![0, 2, 4]);
        let spliced = splice_messages(&messages, &kept, AssistantMessage::new("SUM"));
        assert!(matches!(&spliced[0], Message::User(u) if u.content() == "latest" && !u.steered()));
        assert!(matches!(&spliced[1], Message::Assistant(a) if a.content() == "SUM"));
        assert!(matches!(&spliced[2], Message::User(u) if u.steered()));
        assert!(matches!(&spliced[3], Message::Assistant(a) if a.content() == "done"));
    }

    #[test]
    fn interleaved_tool_exchange_is_not_a_valid_tail() {
        let messages = vec![
            Message::User(UserMessage::new("run both")),
            Message::Assistant(AssistantMessage::new("").with_tool_calls(vec![
                ToolCall::new("call-1", "one", json!({})),
                ToolCall::new("call-2", "two", json!({})),
            ])),
            Message::ToolResult(ToolResult::success("call-1", json!("one"))),
            Message::User(UserMessage::new("keep going").with_steered(true)),
            Message::ToolResult(ToolResult::success("call-2", json!("two"))),
            Message::Assistant(AssistantMessage::new("done")),
        ];
        assert_eq!(plan(&messages, u64::MAX), Some(vec![0, 3, 5]));
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestError(&'static str);

    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.0)
        }
    }

    impl Error for TestError {}

    #[derive(Clone)]
    struct FakeProvider {
        reply: Result<String, TestError>,
        created: Arc<AtomicUsize>,
        completes: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<(usize, Conversation)>>>,
    }

    impl FakeProvider {
        fn new(reply: Result<String, TestError>) -> Self {
            Self {
                reply,
                created: Arc::new(AtomicUsize::new(0)),
                completes: Arc::new(AtomicUsize::new(0)),
                seen: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    impl Provider for FakeProvider {
        type Model = ();
        type Error = TestError;
        type Continuation = ();

        fn supports_vision(&self, _model: &Self::Model) -> bool {
            false
        }

        fn create_continuation(&self, _model: &Self::Model) -> Self::Continuation {
            self.created.fetch_add(1, Ordering::SeqCst);
        }

        fn complete(
            &self,
            _continuation: &Self::Continuation,
            _model: &Self::Model,
            conversation: &Conversation,
            tools: &[ToolSpec],
        ) -> impl Future<Output = Result<(AssistantMessage, Self::Continuation), Self::Error>> + Send
        {
            self.completes.fetch_add(1, Ordering::SeqCst);
            self.seen
                .lock()
                .unwrap()
                .push((tools.len(), conversation.clone()));
            ready(
                self.reply
                    .clone()
                    .map(|text| (AssistantMessage::new(text), ())),
            )
        }
    }

    fn history_needing_compact() -> Conversation {
        let mut conversation = Conversation::new().with_system_prompt("be helpful");
        conversation.push_user_message(UserMessage::new("old task"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("x".repeat(300))));
        conversation.push_user_message(UserMessage::new("latest task"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("y".repeat(300))));
        conversation.push_user_message(UserMessage::new("nudge").with_steered(true));
        conversation.push_message(Message::Assistant(AssistantMessage::new("recent")));
        conversation
    }

    fn tiny_tail() -> CompactOptions {
        CompactOptions {
            keep_tail_tokens: 1,
        }
    }

    fn last_message_tail(conversation: &Conversation) -> CompactOptions {
        CompactOptions {
            keep_tail_tokens: conversation.messages().last().unwrap().estimate_tokens(),
        }
    }

    #[tokio::test]
    async fn compact_uses_a_fresh_one_shot_summarizer_request() {
        let provider = FakeProvider::new(Ok("SUM".into()));
        let conversation = history_needing_compact();
        let compacted = provider
            .compact(&(), &conversation, last_message_tail(&conversation))
            .await
            .unwrap();

        assert_eq!(provider.created.load(Ordering::SeqCst), 1);
        assert_eq!(provider.completes.load(Ordering::SeqCst), 1);
        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, 0);
        assert_ne!(seen[0].1, conversation);
        assert_eq!(seen[0].1.system_prompt(), Some(COMPACTION_SYSTEM.trim()));
        assert_eq!(seen[0].1.messages().len(), 1);
        assert!(matches!(
            seen[0].1.messages().first(),
            Some(Message::User(_))
        ));

        assert_eq!(compacted.system_prompt(), Some("be helpful"));
        assert!(matches!(
            compacted.messages().first(),
            Some(Message::Assistant(a)) if a.content() == "SUM"
        ));
        assert!(
            compacted
                .messages()
                .iter()
                .any(|message| matches!(message, Message::User(u) if u.content() == "latest task"))
        );
        assert!(
            compacted
                .messages()
                .iter()
                .any(|message| matches!(message, Message::User(u) if u.steered()))
        );
        assert!(
            compacted
                .messages()
                .iter()
                .any(|message| matches!(message, Message::Assistant(a) if a.content() == "recent"))
        );
        assert!(
            !compacted
                .messages()
                .iter()
                .any(|message| matches!(message, Message::User(u) if u.content() == "old task"))
        );
    }

    #[tokio::test]
    async fn compact_does_not_mutate_the_original_conversation() {
        let provider = FakeProvider::new(Ok("SUM".into()));
        let conversation = history_needing_compact();
        let before = conversation.clone();
        let _ = provider
            .compact(&(), &conversation, tiny_tail())
            .await
            .unwrap();
        assert_eq!(conversation, before);
    }

    #[tokio::test]
    async fn empty_summary_is_an_error() {
        let provider = FakeProvider::new(Ok("  \n".into()));
        let error = provider
            .compact(&(), &history_needing_compact(), tiny_tail())
            .await
            .unwrap_err();
        assert_eq!(error, CompactError::EmptySummary);
        assert_eq!(provider.completes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn provider_failure_is_wrapped() {
        let provider = FakeProvider::new(Err(TestError("nope")));
        let error = provider
            .compact(&(), &history_needing_compact(), tiny_tail())
            .await
            .unwrap_err();
        assert_eq!(error, CompactError::Provider(TestError("nope")));
    }

    #[tokio::test]
    async fn nothing_to_compact_does_not_call_the_provider() {
        let provider = FakeProvider::new(Ok("SUM".into()));
        let mut conversation = Conversation::new();
        conversation.push_user_message(UserMessage::new("hello"));
        conversation.push_message(Message::Assistant(AssistantMessage::new("hi")));
        let error = provider
            .compact(&(), &conversation, CompactOptions::default())
            .await
            .unwrap_err();
        assert_eq!(error, CompactError::NothingToCompact);
        assert_eq!(provider.created.load(Ordering::SeqCst), 0);
        assert_eq!(provider.completes.load(Ordering::SeqCst), 0);
    }
}
