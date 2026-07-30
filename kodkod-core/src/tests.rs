#![allow(deprecated, unused_must_use)]

use std::error::Error;
use std::fmt;
use std::future::{Future, ready};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use futures::StreamExt;

use serde_json::{Value, json};

use crate::{
    Agent, AgentError, AgentEvent, AssistantMessage, Conversation, Image, Message, Provider,
    TaskControl, Tool, ToolCall, ToolError, ToolExecutor, ToolExecutorError, ToolFuture,
    ToolResult, ToolResultOutcome, ToolSpec, UserMessage,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct TestError(&'static str);

impl fmt::Display for TestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl Error for TestError {}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestModel {
    vision: bool,
    computer_use: bool,
}

impl TestModel {
    const fn new() -> Self {
        Self {
            vision: false,
            computer_use: false,
        }
    }

    const fn with_vision() -> Self {
        Self {
            vision: true,
            computer_use: false,
        }
    }

    const fn with_computer_use() -> Self {
        Self {
            vision: true,
            computer_use: true,
        }
    }

    fn vision(&self) -> bool {
        self.vision
    }

    fn computer_use(&self) -> bool {
        self.computer_use
    }
}

fn conversation_has_images(conversation: &Conversation) -> bool {
    conversation
        .messages()
        .iter()
        .any(|message| matches!(message, Message::User(user) if !user.images().is_empty()))
}

#[derive(Default)]
struct RecordingProvider {
    seen_tool_names: Arc<Mutex<Vec<String>>>,
}

impl Provider for RecordingProvider {
    type Model = TestModel;
    type Error = TestError;
    type TurnState = ();

    fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

    fn supports_vision(&self, model: &TestModel) -> bool {
        model.vision()
    }

    fn supports_computer_use(&self, model: &TestModel) -> bool {
        model.computer_use()
    }

    fn complete_round(
        &self,
        _state: &mut Self::TurnState,
        _model: &TestModel,
        _conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
        self.seen_tool_names
            .lock()
            .unwrap()
            .extend(tools.iter().map(|tool| tool.name().to_owned()));

        ready(Ok(AssistantMessage::new("done")))
    }
}

#[derive(Default)]
struct ToolCallingProvider {
    calls: Arc<AtomicUsize>,
}

impl Provider for ToolCallingProvider {
    type Model = TestModel;
    type Error = TestError;
    type TurnState = ();

    fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

    fn supports_vision(&self, model: &TestModel) -> bool {
        model.vision()
    }

    fn complete_round(
        &self,
        _state: &mut Self::TurnState,
        _model: &TestModel,
        conversation: &Conversation,
        tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
        let call_count = self.calls.fetch_add(1, Ordering::SeqCst);
        assert!(tools.iter().any(|tool| tool.name() == "echo"));

        let has_tool_result = conversation
            .messages()
            .iter()
            .any(|message| matches!(message, Message::ToolResult(_)));

        ready(Ok(if call_count == 0 {
            AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                "call_1",
                "echo",
                json!({ "value": "hello" }),
            )])
        } else {
            assert!(has_tool_result);
            AssistantMessage::new("done")
        }))
    }
}

struct AlwaysToolCallingProvider;

impl Provider for AlwaysToolCallingProvider {
    type Model = TestModel;
    type Error = TestError;
    type TurnState = ();

    fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

    fn supports_vision(&self, model: &TestModel) -> bool {
        model.vision()
    }

    fn complete_round(
        &self,
        _state: &mut Self::TurnState,
        _model: &TestModel,
        _conversation: &Conversation,
        _tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
        ready(Ok(AssistantMessage::new("").with_tool_calls(vec![
            ToolCall::new("call_1", "missing", json!({})),
        ])))
    }
}

struct EchoTool;

impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "echo",
            "Returns the provided arguments unchanged.",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            }),
        )
    }

    fn execute<'a>(&'a self, arguments: &'a Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(arguments.clone().into()) })
    }
}

struct FailingTool;

impl Tool for FailingTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("fail", "Always fails.", json!({ "type": "object" }))
    }

    fn execute<'a>(&'a self, _arguments: &'a Value) -> ToolFuture<'a> {
        Box::pin(async { Err(ToolError::new("boom")) })
    }
}

struct VisionTool;

impl Tool for VisionTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new("vision", "Requires vision.", json!({ "type": "object" }))
    }

    fn requires_vision(&self) -> bool {
        true
    }

    fn execute<'a>(&'a self, arguments: &'a Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(arguments.clone().into()) })
    }
}

struct ComputerUseTool;

impl Tool for ComputerUseTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "computer_use",
            "Requires computer use.",
            json!({ "type": "object" }),
        )
    }

    fn requires_vision(&self) -> bool {
        true
    }

    fn requires_computer_use(&self) -> bool {
        true
    }

    fn execute<'a>(&'a self, arguments: &'a Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(arguments.clone().into()) })
    }
}

#[test]
fn agent_passes_registered_tool_specs_to_provider() {
    let provider = RecordingProvider::default();
    let seen_tool_names = provider.seen_tool_names.clone();
    let agent = Agent::new(provider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = TestModel::new();
    block_on(collect_run(&agent, &mut conversation, "hello", &model)).unwrap();

    assert_eq!(*seen_tool_names.lock().unwrap(), vec!["echo"]);
}

#[test]
fn agent_only_advertises_vision_tools_to_vision_models() {
    let non_vision_provider = RecordingProvider::default();
    let non_vision_names = non_vision_provider.seen_tool_names.clone();
    let non_vision_agent = Agent::new(non_vision_provider)
        .with_tool(Arc::new(EchoTool))
        .with_tool(Arc::new(VisionTool));
    block_on(collect_run(
        &non_vision_agent,
        &mut Conversation::new(),
        "hello",
        &TestModel::new(),
    ))
    .unwrap();
    assert_eq!(*non_vision_names.lock().unwrap(), vec!["echo"]);

    let vision_provider = RecordingProvider::default();
    let vision_names = vision_provider.seen_tool_names.clone();
    let vision_agent = Agent::new(vision_provider)
        .with_tool(Arc::new(EchoTool))
        .with_tool(Arc::new(VisionTool));
    block_on(collect_run(
        &vision_agent,
        &mut Conversation::new(),
        "hello",
        &TestModel::with_vision(),
    ))
    .unwrap();
    assert_eq!(*vision_names.lock().unwrap(), vec!["echo", "vision"]);
}

#[test]
fn agent_only_advertises_computer_use_tools_to_supported_models() {
    let unsupported_provider = RecordingProvider::default();
    let unsupported_names = unsupported_provider.seen_tool_names.clone();
    let unsupported_agent = Agent::new(unsupported_provider)
        .with_tool(Arc::new(EchoTool))
        .with_tool(Arc::new(ComputerUseTool));
    block_on(collect_run(
        &unsupported_agent,
        &mut Conversation::new(),
        "hello",
        &TestModel::with_vision(),
    ))
    .unwrap();
    assert_eq!(*unsupported_names.lock().unwrap(), vec!["echo"]);

    let supported_provider = RecordingProvider::default();
    let supported_names = supported_provider.seen_tool_names.clone();
    let supported_agent = Agent::new(supported_provider)
        .with_tool(Arc::new(EchoTool))
        .with_tool(Arc::new(ComputerUseTool));
    block_on(collect_run(
        &supported_agent,
        &mut Conversation::new(),
        "hello",
        &TestModel::with_computer_use(),
    ))
    .unwrap();
    assert_eq!(
        *supported_names.lock().unwrap(),
        vec!["computer_use", "echo"]
    );
}

#[test]
fn agent_executes_tool_calls_until_final_response() {
    let provider = ToolCallingProvider::default();
    let calls = provider.calls.clone();
    let agent = Agent::new(provider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = TestModel::new();
    let events = block_on(collect_events(&agent, &mut conversation, "hello", &model)).unwrap();
    let response = events
        .iter()
        .find_map(|event| match event {
            AgentEvent::Completed(message) => Some(message),
            _ => None,
        })
        .unwrap();

    assert_eq!(response.content(), "done");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(matches!(
        &events[0],
        AgentEvent::AssistantReply(message) if message.tool_calls().len() == 1
    ));
    assert!(matches!(&events[1], AgentEvent::ToolStarted(call) if call.name() == "echo"));
    assert!(matches!(
        &events[2],
        AgentEvent::ToolFinished(result) if result.tool_call_id() == "call_1"
    ));
    assert_eq!(conversation.messages().len(), 4);
    assert!(matches!(&conversation.messages()[0], Message::User(_)));
    assert!(matches!(
        &conversation.messages()[1],
        Message::Assistant(message) if message.tool_calls().len() == 1
    ));
    assert!(matches!(
        &conversation.messages()[2],
        Message::ToolResult(result) if result.value() == Some(&json!({ "value": "hello" }))
    ));
    assert!(matches!(
        &conversation.messages()[3],
        Message::Assistant(message) if message.content() == "done"
    ));
}

struct SlowEchoTool {
    max_in_flight: Arc<AtomicUsize>,
    in_flight: Arc<AtomicUsize>,
}

impl SlowEchoTool {
    fn new() -> (Self, Arc<AtomicUsize>) {
        let max_in_flight = Arc::new(AtomicUsize::new(0));
        (
            Self {
                max_in_flight: Arc::clone(&max_in_flight),
                in_flight: Arc::new(AtomicUsize::new(0)),
            },
            max_in_flight,
        )
    }
}

impl Tool for SlowEchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec::new(
            "echo",
            "Returns the provided arguments unchanged.",
            json!({
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                },
                "required": ["value"]
            }),
        )
    }

    fn execute<'a>(&'a self, arguments: &'a Value) -> ToolFuture<'a> {
        let max_in_flight = Arc::clone(&self.max_in_flight);
        let in_flight = Arc::clone(&self.in_flight);
        Box::pin(async move {
            let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut observed = max_in_flight.load(Ordering::SeqCst);
            while observed < now {
                match max_in_flight.compare_exchange_weak(
                    observed,
                    now,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(current) => observed = current,
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(arguments.clone().into())
        })
    }
}

struct TwoToolCallsProvider;

impl Provider for TwoToolCallsProvider {
    type Model = TestModel;
    type Error = TestError;
    type TurnState = ();

    fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

    fn supports_vision(&self, model: &TestModel) -> bool {
        model.vision()
    }

    fn complete_round(
        &self,
        _state: &mut Self::TurnState,
        _model: &TestModel,
        conversation: &Conversation,
        _tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
        let has_tool_results = conversation
            .messages()
            .iter()
            .filter(|message| matches!(message, Message::ToolResult(_)))
            .count();

        ready(Ok(if has_tool_results == 0 {
            AssistantMessage::new("").with_tool_calls(vec![
                ToolCall::new("call_1", "echo", json!({ "value": "a" })),
                ToolCall::new("call_2", "echo", json!({ "value": "b" })),
            ])
        } else {
            assert_eq!(has_tool_results, 2);
            AssistantMessage::new("done")
        }))
    }
}

#[tokio::test]
async fn agent_executes_multiple_tool_calls_in_parallel() {
    let (slow_tool, max_in_flight) = SlowEchoTool::new();
    let agent = Agent::new(TwoToolCallsProvider).with_tool(Arc::new(slow_tool));
    let mut conversation = Conversation::new();

    let model = TestModel::new();
    let response = collect_run(&agent, &mut conversation, "hello", &model)
        .await
        .unwrap();

    assert_eq!(response.content(), "done");
    assert!(
        max_in_flight.load(Ordering::SeqCst) >= 2,
        "expected at least two tool calls to overlap, got max_in_flight={}",
        max_in_flight.load(Ordering::SeqCst)
    );
}

#[test]
fn agent_stops_after_max_tool_rounds() {
    let agent = Agent::new(AlwaysToolCallingProvider).with_max_tool_rounds(0);
    let mut conversation = Conversation::new();

    let model = TestModel::new();
    let result = block_on(collect_run(&agent, &mut conversation, "hello", &model));

    assert!(matches!(
        result,
        Err(AgentError::MaxToolRoundsExceeded { max: 0 })
    ));
    assert_eq!(conversation.messages().len(), 2);
}

#[test]
fn tool_executor_routes_calls_and_errors() {
    let mut executor = ToolExecutor::new();
    executor.register(Arc::new(EchoTool));
    executor.register(Arc::new(FailingTool));

    let success = block_on(executor.execute(&ToolCall::new(
        "call_1",
        "echo",
        json!({ "value": "hello" }),
    )));
    assert_eq!(
        success.outcome(),
        &ToolResultOutcome::Success(json!({ "value": "hello" }).into())
    );

    let unknown = block_on(executor.execute(&ToolCall::new("call_2", "missing", json!({}))));
    assert_eq!(
        unknown.outcome(),
        &ToolResultOutcome::Error(ToolExecutorError::UnknownTool("missing".to_owned()))
    );

    let failure = block_on(executor.execute(&ToolCall::new("call_3", "fail", json!({}))));
    assert_eq!(
        failure.outcome(),
        &ToolResultOutcome::Error(ToolExecutorError::Tool(ToolError::new("boom")))
    );
}

#[test]
fn tool_executor_rejects_computer_use_for_unsupported_models() {
    let mut executor = ToolExecutor::new();
    executor.register(Arc::new(ComputerUseTool));
    let call = ToolCall::new("call_1", "computer_use", json!({}));

    let result = block_on(executor.execute_for_capabilities(&call, true, false));

    assert_eq!(
        result.outcome(),
        &ToolResultOutcome::Error(ToolExecutorError::Tool(ToolError::new(
            "tool 'computer_use' requires a computer-use-capable model"
        )))
    );
}

#[test]
fn conversation_round_trips_through_json() {
    let mut conversation = Conversation::new().with_system_prompt("Be concise.");
    conversation.push_user_message(
        UserMessage::new("describe").with_images(vec![Image::new("image/png", vec![0x89, 0x50])]),
    );
    conversation.push_message(Message::Assistant(
        AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
            "call_1",
            "echo",
            json!({ "value": "hello" }),
        )]),
    ));
    conversation.push_message(Message::ToolResult(ToolResult::success(
        "call_1",
        json!({ "value": "hello" }),
    )));
    conversation.push_message(Message::Assistant(AssistantMessage::new("done")));

    let encoded = serde_json::to_string(&conversation).unwrap();
    let decoded: Conversation = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, conversation);
}

struct CapturingProvider {
    saw_images: Arc<Mutex<Vec<bool>>>,
}

impl Provider for CapturingProvider {
    type Model = TestModel;
    type Error = TestError;
    type TurnState = ();

    fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

    fn supports_vision(&self, model: &TestModel) -> bool {
        model.vision()
    }

    fn complete_round(
        &self,
        _state: &mut Self::TurnState,
        _model: &TestModel,
        conversation: &Conversation,
        _tools: &[ToolSpec],
    ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
        self.saw_images
            .lock()
            .unwrap()
            .push(conversation_has_images(conversation));
        ready(Ok(AssistantMessage::new("ok")))
    }
}

#[test]
fn agent_strips_images_based_on_provider_vision_support() {
    let saw_images = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::new(CapturingProvider {
        saw_images: saw_images.clone(),
    });
    let image = vec![Image::new("image/png", vec![0x89, 0x50])];

    let mut non_vision = Conversation::new();
    non_vision.push_user_message(UserMessage::new("describe").with_images(image.clone()));
    block_on(collect_run(
        &agent,
        &mut non_vision,
        "go",
        &TestModel::new(),
    ))
    .unwrap();
    assert_eq!(saw_images.lock().unwrap().as_slice(), &[false]);

    let mut vision = Conversation::new();
    vision.push_user_message(UserMessage::new("describe").with_images(image));
    block_on(collect_run(
        &agent,
        &mut vision,
        "go",
        &TestModel::with_vision(),
    ))
    .unwrap();
    assert_eq!(saw_images.lock().unwrap().as_slice(), &[false, true]);
}

async fn collect_events<P>(
    agent: &Agent<P>,
    conversation: &mut Conversation,
    prompt: &str,
    model: &P::Model,
) -> Result<Vec<AgentEvent>, AgentError<TestError>>
where
    P: Provider<Error = TestError> + Sync,
{
    conversation.push_user_message(UserMessage::new(prompt));
    let control = TaskControl::new();
    let mut stream = agent.run(conversation, model, &control);
    let mut events = Vec::new();

    while let Some(item) = stream.next().await {
        events.push(item?);
    }

    Ok(events)
}

async fn collect_run<P>(
    agent: &Agent<P>,
    conversation: &mut Conversation,
    prompt: &str,
    model: &P::Model,
) -> Result<AssistantMessage, AgentError<TestError>>
where
    P: Provider<Error = TestError> + Sync,
{
    conversation.push_user_message(UserMessage::new(prompt));
    let control = TaskControl::new();
    let mut stream = agent.run(conversation, model, &control);

    while let Some(item) = stream.next().await {
        if let Ok(AgentEvent::Completed(message)) = item {
            return Ok(message);
        }
        item?;
    }

    Err(AgentError::Provider(TestError(
        "agent stream ended without completion",
    )))
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(NoopWaker));
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);

    loop {
        match Pin::new(&mut future).poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

struct NoopWaker;

impl Wake for NoopWaker {
    fn wake(self: Arc<Self>) {}
}

#[test]
fn steered_message_is_injected_between_rounds() {
    // On round 1 the provider returns a tool call, then the test steers a
    // message before round 2. Round 2 asserts the steer is present in the
    // conversation and replies with it.
    struct SteerAwareProvider {
        calls: Arc<AtomicUsize>,
        saw_steer: Arc<Mutex<bool>>,
    }

    impl Provider for SteerAwareProvider {
        type Model = TestModel;
        type Error = TestError;
        type TurnState = ();

        fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

        fn supports_vision(&self, model: &TestModel) -> bool {
            model.vision()
        }

        fn complete_round(
            &self,
            _state: &mut Self::TurnState,
            _model: &TestModel,
            conversation: &Conversation,
            tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
            let round = self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(tools.iter().any(|tool| tool.name() == "echo"));

            let reply = if round == 0 {
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "call_1",
                    "echo",
                    json!({ "value": "hello" }),
                )])
            } else {
                // Round 1: the steered message should now be in the conversation.
                let found = conversation.messages().iter().any(|message| {
                    matches!(message,
                        Message::User(user) if user.content() == "wait, stop")
                });
                *self.saw_steer.lock().unwrap() = found;
                AssistantMessage::new(if found { "ack" } else { "missed" })
            };

            ready(Ok(reply))
        }
    }

    let saw_steer = Arc::new(Mutex::new(false));
    let provider = SteerAwareProvider {
        calls: Arc::new(AtomicUsize::new(0)),
        saw_steer: saw_steer.clone(),
    };
    let agent = Agent::new(provider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = TestModel::new();
    conversation.push_user_message(UserMessage::new("go"));
    let control = TaskControl::new();
    let mut stream = agent.run(&mut conversation, &model, &control);

    // Round 1: assistant replies with a tool call.
    drain_until_assistant_reply(&mut stream);
    drain_until_tool_finished(&mut stream);

    // Now queue a steer before round 2 begins.
    control.steer(UserMessage::new("wait, stop"));

    // Round 2: the loop drains the steer, appends it, then calls the provider.
    let mut final_message = None;
    while let Some(item) = block_on(stream.next()) {
        if let AgentEvent::Completed(message) = item.unwrap() {
            final_message = Some(message);
            break;
        }
    }

    assert_eq!(final_message.unwrap().content(), "ack");
    assert!(*saw_steer.lock().unwrap(), "provider did not see the steer");
}

#[test]
fn steer_events_are_emitted_in_order() {
    // A provider that loops through a tool call each round up to a max,
    // allowing multiple steers to be queued and drained across boundaries.
    struct LoopingProvider {
        calls: Arc<AtomicUsize>,
        rounds: usize,
    }

    impl Provider for LoopingProvider {
        type Model = TestModel;
        type Error = TestError;
        type TurnState = ();

        fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

        fn supports_vision(&self, model: &TestModel) -> bool {
            model.vision()
        }

        fn complete_round(
            &self,
            _state: &mut Self::TurnState,
            _model: &TestModel,
            _conversation: &Conversation,
            tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
            let round = self.calls.fetch_add(1, Ordering::SeqCst);
            assert!(tools.iter().any(|tool| tool.name() == "echo"));
            ready(Ok(if round + 1 >= self.rounds {
                AssistantMessage::new("done")
            } else {
                AssistantMessage::new("").with_tool_calls(vec![ToolCall::new(
                    "call_1",
                    "echo",
                    json!({}),
                )])
            }))
        }
    }

    let agent = Agent::new(LoopingProvider {
        calls: Arc::new(AtomicUsize::new(0)),
        rounds: 3,
    })
    .with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = TestModel::new();
    conversation.push_user_message(UserMessage::new("start"));
    let control = TaskControl::new();

    // Collect every event from the run while queuing steers between rounds.
    // The stream borrows `conversation` mutably, so drive it to completion and
    // drop it before inspecting the conversation.
    let events = {
        let mut stream = agent.run(&mut conversation, &model, &control);
        // Round 1: tool call then tool result.
        block_on(stream.next()).unwrap().unwrap(); // AssistantReply
        block_on(stream.next()).unwrap().unwrap(); // ToolStarted
        block_on(stream.next()).unwrap().unwrap(); // ToolFinished
        // Queue two steers before round 2 begins.
        control.steer(UserMessage::new("one"));
        control.steer(UserMessage::new("two"));
        // Drain to completion: round 2 (with the steers) and round 3 (final).
        let mut collected = Vec::new();
        while let Some(item) = block_on(stream.next()) {
            collected.push(item.unwrap());
        }
        collected
    };

    // The two steers appear as Steered events, in queue order.
    let steered: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            AgentEvent::Steered(user) => Some(user.content()),
            _ => None,
        })
        .collect();
    assert_eq!(steered, vec!["one", "two"]);

    // The conversation holds the steered messages.
    let user_contents: Vec<&str> = conversation
        .messages()
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(user.content()),
            _ => None,
        })
        .collect();
    assert!(user_contents.contains(&"one"));
    assert!(user_contents.contains(&"two"));
}

#[test]
fn cancel_takes_precedence_over_steer() {
    // If a steer and a cancel are both pending, the cancel check runs first
    // and the turn ends without consuming the steer.
    struct IdleProvider;
    impl Provider for IdleProvider {
        type Model = TestModel;
        type Error = TestError;
        type TurnState = ();

        fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

        fn supports_vision(&self, model: &TestModel) -> bool {
            model.vision()
        }

        fn complete_round(
            &self,
            _state: &mut Self::TurnState,
            _model: &TestModel,
            _conversation: &Conversation,
            _tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
            ready(Ok(AssistantMessage::new("").with_tool_calls(vec![
                ToolCall::new("call_1", "echo", json!({})),
            ])))
        }
    }

    let agent = Agent::new(IdleProvider).with_tool(Arc::new(EchoTool));
    let mut conversation = Conversation::new();

    let model = TestModel::new();
    conversation.push_user_message(UserMessage::new("start"));
    let control = TaskControl::new();
    let mut stream = agent.run(&mut conversation, &model, &control);

    // Round 1 returns a tool call.
    drain_until_tool_finished(&mut stream);

    // Queue both a steer and a cancel.
    control.steer(UserMessage::new("ignored"));
    control.cancel();

    // The loop should cancel before processing the steer.
    let mut cancelled = false;
    while let Some(item) = block_on(stream.next()) {
        match item {
            Err(AgentError::Cancelled) => {
                cancelled = true;
                break;
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    assert!(cancelled);
}

#[test]
fn cancel_interrupts_and_drops_in_flight_provider_future() {
    struct PendingProvider {
        started: Arc<AtomicBool>,
        dropped: Arc<AtomicBool>,
    }

    struct PendingCompletion {
        started: Arc<AtomicBool>,
        dropped: Arc<AtomicBool>,
    }

    impl Future for PendingCompletion {
        type Output = Result<AssistantMessage, TestError>;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            self.started.store(true, Ordering::SeqCst);
            Poll::Pending
        }
    }

    impl Drop for PendingCompletion {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    impl Provider for PendingProvider {
        type Model = TestModel;
        type Error = TestError;
        type TurnState = ();

        fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

        fn supports_vision(&self, model: &TestModel) -> bool {
            model.vision()
        }

        fn complete_round(
            &self,
            _state: &mut Self::TurnState,
            _model: &TestModel,
            _conversation: &Conversation,
            _tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
            PendingCompletion {
                started: Arc::clone(&self.started),
                dropped: Arc::clone(&self.dropped),
            }
        }
    }

    let started = Arc::new(AtomicBool::new(false));
    let dropped = Arc::new(AtomicBool::new(false));
    let agent = Agent::new(PendingProvider {
        started: Arc::clone(&started),
        dropped: Arc::clone(&dropped),
    });
    let model = TestModel::new();
    let control = TaskControl::new();
    let cancellation = control.clone();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("stop this"));

    let cancel_thread = std::thread::spawn(move || {
        while !started.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        cancellation.cancel();
    });

    {
        let mut stream = agent.run(&mut conversation, &model, &control);
        assert!(matches!(
            block_on(stream.next()),
            Some(Err(AgentError::Cancelled))
        ));
        assert!(block_on(stream.next()).is_none());
    }

    cancel_thread.join().unwrap();
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(conversation.messages().len(), 1);
    assert!(matches!(conversation.messages()[0], Message::User(_)));
}

#[test]
fn cancellation_wins_when_provider_completes_in_the_same_poll() {
    struct CancellingProvider {
        control: TaskControl,
    }

    impl Provider for CancellingProvider {
        type Model = TestModel;
        type Error = TestError;
        type TurnState = ();

        fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

        fn supports_vision(&self, model: &TestModel) -> bool {
            model.vision()
        }

        fn complete_round(
            &self,
            _state: &mut Self::TurnState,
            _model: &TestModel,
            _conversation: &Conversation,
            _tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
            let control = self.control.clone();
            async move {
                control.cancel();
                Ok(AssistantMessage::new("too late"))
            }
        }
    }

    let control = TaskControl::new();
    let agent = Agent::new(CancellingProvider {
        control: control.clone(),
    });
    let model = TestModel::new();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("race"));

    {
        let mut stream = agent.run(&mut conversation, &model, &control);
        assert!(matches!(
            block_on(stream.next()),
            Some(Err(AgentError::Cancelled))
        ));
    }

    assert_eq!(conversation.messages().len(), 1);
    assert!(matches!(conversation.messages()[0], Message::User(_)));
}

#[test]
fn cancellation_during_completion_construction_prevents_provider_poll() {
    struct CancellingProvider {
        control: TaskControl,
        polled: Arc<AtomicBool>,
    }

    struct PollTrackingCompletion {
        polled: Arc<AtomicBool>,
    }

    impl Future for PollTrackingCompletion {
        type Output = Result<AssistantMessage, TestError>;

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            self.polled.store(true, Ordering::SeqCst);
            Poll::Pending
        }
    }

    impl Provider for CancellingProvider {
        type Model = TestModel;
        type Error = TestError;
        type TurnState = ();

        fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {}

        fn supports_vision(&self, model: &TestModel) -> bool {
            model.vision()
        }

        fn complete_round(
            &self,
            _state: &mut Self::TurnState,
            _model: &TestModel,
            _conversation: &Conversation,
            _tools: &[ToolSpec],
        ) -> impl Future<Output = Result<AssistantMessage, TestError>> + Send {
            self.control.cancel();
            PollTrackingCompletion {
                polled: Arc::clone(&self.polled),
            }
        }
    }

    let control = TaskControl::new();
    let polled = Arc::new(AtomicBool::new(false));
    let agent = Agent::new(CancellingProvider {
        control: control.clone(),
        polled: Arc::clone(&polled),
    });
    let model = TestModel::new();
    let mut conversation = Conversation::new();
    conversation.push_user_message(UserMessage::new("cancel before polling"));

    {
        let mut stream = agent.run(&mut conversation, &model, &control);
        assert!(matches!(
            block_on(stream.next()),
            Some(Err(AgentError::Cancelled))
        ));
    }

    assert!(!polled.load(Ordering::SeqCst));
    assert_eq!(conversation.messages().len(), 1);
}

/// Pull events from the stream until (and including) the first assistant reply.
fn drain_until_assistant_reply(stream: &mut crate::Task<'_, TestError>) {
    while let Some(item) = block_on(stream.next()) {
        if let AgentEvent::AssistantReply(_) = item.unwrap() {
            return;
        }
    }
    panic!("stream ended before an assistant reply");
}

/// Pull events until a ToolFinished has been emitted (completing a tool round).
fn drain_until_tool_finished(stream: &mut crate::Task<'_, TestError>) {
    while let Some(item) = block_on(stream.next()) {
        if let AgentEvent::ToolFinished(_) = item.unwrap() {
            return;
        }
    }
    panic!("stream ended before a tool finished");
}

#[tokio::test]
async fn run_turn_appends_prompt_once_creates_one_state_and_drops_before_completion() {
    struct State {
        drops: Arc<AtomicUsize>,
    }
    impl Drop for State {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct StatefulProvider {
        creates: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }
    impl Provider for StatefulProvider {
        type Model = TestModel;
        type Error = TestError;
        type TurnState = State;

        fn supports_vision(&self, _model: &Self::Model) -> bool {
            false
        }
        fn create_turn_state(&self, _model: &Self::Model) -> Self::TurnState {
            self.creates.fetch_add(1, Ordering::SeqCst);
            State {
                drops: Arc::clone(&self.drops),
            }
        }
        async fn complete_round(
            &self,
            _state: &mut Self::TurnState,
            _model: &Self::Model,
            conversation: &Conversation,
            _tools: &[ToolSpec],
        ) -> Result<AssistantMessage, Self::Error> {
            assert_eq!(conversation.messages().len(), 1);
            assert!(
                matches!(&conversation.messages()[0], Message::User(user) if user.content() == "hello" && !user.steered())
            );
            Ok(AssistantMessage::new("done"))
        }
    }

    let creates = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));
    let agent = Agent::new(StatefulProvider {
        creates: Arc::clone(&creates),
        drops: Arc::clone(&drops),
    });
    let mut conversation = Conversation::new();
    let control = TaskControl::new();
    let model = TestModel::new();
    let mut task = agent.run_turn(
        &mut conversation,
        &model,
        UserMessage::new("hello").with_steered(true),
        &control,
    );

    assert!(matches!(
        task.next().await.unwrap().unwrap(),
        AgentEvent::AssistantReply(_)
    ));
    assert_eq!(creates.load(Ordering::SeqCst), 1);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(matches!(
        task.next().await.unwrap().unwrap(),
        AgentEvent::Completed(_)
    ));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(control.steer(UserMessage::new("late")).is_err());
    drop(task);
    assert_eq!(conversation.messages().len(), 2);
}

#[tokio::test]
async fn abandoning_run_turn_drops_state_and_closes_steering() {
    struct PendingState(Arc<AtomicUsize>);
    impl Drop for PendingState {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct PendingStateProvider(Arc<AtomicUsize>);
    impl Provider for PendingStateProvider {
        type Model = TestModel;
        type Error = TestError;
        type TurnState = PendingState;
        fn supports_vision(&self, _: &Self::Model) -> bool {
            false
        }
        fn create_turn_state(&self, _: &Self::Model) -> Self::TurnState {
            PendingState(Arc::clone(&self.0))
        }
        async fn complete_round(
            &self,
            _: &mut Self::TurnState,
            _: &Self::Model,
            _: &Conversation,
            _: &[ToolSpec],
        ) -> Result<AssistantMessage, Self::Error> {
            futures::future::pending().await
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let agent = Agent::new(PendingStateProvider(Arc::clone(&drops)));
    let mut conversation = Conversation::new();
    let control = TaskControl::new();
    let model = TestModel::new();
    let mut task = agent.run_turn(
        &mut conversation,
        &model,
        UserMessage::new("hello"),
        &control,
    );
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(1), task.next())
            .await
            .is_err()
    );
    drop(task);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert!(control.steer(UserMessage::new("late")).is_err());
}
