//! Helpers shared by the integration tests: a collecting trace processor, the
//! span forest in the goldens' normalised form, and small builders.

#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use lattice_agents::testing::{ScriptedModel, ScriptedStep};
use lattice_agents::{
    Agent, ApprovalDecision, ApprovalPort, ApprovalRequest, FunctionTool, InputItem, NeedsApproval,
    RunConfig, RunError, RunHandle, RunResult, Session, SessionError, StreamEvent, ToolError,
    TraceProcessor, TraceRecord, run_streamed, run_streamed_items, strict_object_schema,
};
use lattice_protocol::SpanRecord;
use serde_json::{Value, json};

#[derive(Default)]
struct Inner {
    trace_starts: Vec<TraceRecord>,
    trace_ends: Vec<TraceRecord>,
    span_starts: Vec<SpanRecord>,
    span_ends: Vec<SpanRecord>,
    calls: Vec<String>,
}

/// Records everything a run tells its processors.
#[derive(Default)]
pub struct Collector {
    inner: Mutex<Inner>,
}

impl TraceProcessor for Collector {
    fn on_trace_start(&self, trace: &TraceRecord) {
        let mut inner = self.inner.lock().unwrap();
        inner.calls.push("trace_start".into());
        inner.trace_starts.push(trace.clone());
    }

    fn on_trace_end(&self, trace: &TraceRecord) {
        let mut inner = self.inner.lock().unwrap();
        inner.calls.push("trace_end".into());
        inner.trace_ends.push(trace.clone());
    }

    fn on_span_start(&self, span: &SpanRecord) {
        let mut inner = self.inner.lock().unwrap();
        inner.calls.push(format!("span_start:{}", span.order));
        inner.span_starts.push(span.clone());
    }

    fn on_span_end(&self, span: &SpanRecord) {
        let mut inner = self.inner.lock().unwrap();
        inner.calls.push(format!("span_end:{}", span.order));
        inner.span_ends.push(span.clone());
    }
}

impl Collector {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn span_starts(&self) -> Vec<SpanRecord> {
        self.inner.lock().unwrap().span_starts.clone()
    }

    /// The final record of every span, in the order the spans ended.
    pub fn span_ends(&self) -> Vec<SpanRecord> {
        self.inner.lock().unwrap().span_ends.clone()
    }

    pub fn trace_starts(&self) -> Vec<TraceRecord> {
        self.inner.lock().unwrap().trace_starts.clone()
    }

    pub fn trace_ends(&self) -> Vec<TraceRecord> {
        self.inner.lock().unwrap().trace_ends.clone()
    }

    /// `trace_start`, `span_start:<order>`, `span_end:<order>`, `trace_end`, in call order.
    pub fn calls(&self) -> Vec<String> {
        self.inner.lock().unwrap().calls.clone()
    }

    /// The span forest in the goldens' normalised form: nested by parent,
    /// children in start order, each node `{kind, span_data, error, children}`.
    pub fn forest(&self) -> Vec<Value> {
        let mut ends = self.span_ends();
        ends.sort_by_key(|span| span.order);
        nest(&ends, None)
    }
}

fn kind_of(span_data: &Value) -> String {
    let kind = span_data["type"].as_str().unwrap_or("unknown");
    if kind == "custom" {
        span_data["name"].as_str().unwrap_or("custom").to_owned()
    } else {
        kind.to_owned()
    }
}

fn nest(spans: &[SpanRecord], parent: Option<&str>) -> Vec<Value> {
    spans
        .iter()
        .filter(|span| span.parent_id.as_deref() == parent)
        .map(|span| {
            json!({
                "kind": kind_of(&span.span_data),
                "span_data": span.span_data,
                "error": span.error,
                "children": nest(spans, Some(&span.id)),
            })
        })
        .collect()
}

/// A finished run: its events (by SDK name, text deltas collapsed), outcome and spans.
pub struct Outcome {
    pub events: Vec<StreamEvent>,
    pub names: Vec<&'static str>,
    pub result: Result<RunResult, RunError>,
    pub collector: Arc<Collector>,
}

/// Run `agent` to completion against `model` and gather everything.
pub async fn run_to_end(
    agent: Arc<Agent>,
    input: &str,
    model: Arc<ScriptedModel>,
    tweak: impl FnOnce(&mut RunConfig),
) -> Outcome {
    let collector = Collector::new();
    let mut config = RunConfig::new(model);
    config.processors = vec![collector.clone()];
    tweak(&mut config);
    let handle = run_streamed(agent, input.to_owned(), config);
    gather(handle, collector).await
}

/// [`run_to_end`] for a run started from a list of input items.
pub async fn run_items_to_end(
    agent: Arc<Agent>,
    input: Vec<InputItem>,
    model: Arc<ScriptedModel>,
    tweak: impl FnOnce(&mut RunConfig),
) -> Outcome {
    run_into(agent, Start::Items(input), model, Collector::new(), tweak).await
}

/// What a run starts from: one message (`run_streamed`) or a list of input
/// items (`run_streamed_items`).
#[derive(Clone)]
pub enum Start {
    Text(String),
    Items(Vec<InputItem>),
}

/// A run to its end whose trace goes to `collector`, which may already hold an
/// earlier run's (several runs on one session make one forest).
pub async fn run_into(
    agent: Arc<Agent>,
    start: Start,
    model: Arc<ScriptedModel>,
    collector: Arc<Collector>,
    tweak: impl FnOnce(&mut RunConfig),
) -> Outcome {
    let mut config = RunConfig::new(model);
    config.processors = vec![collector.clone()];
    tweak(&mut config);
    let handle = match start {
        Start::Text(text) => run_streamed(agent, text, config),
        Start::Items(items) => run_streamed_items(agent, items, config),
    };
    gather(handle, collector).await
}

/// A session kept in memory, as the SDK's `SQLiteSession(":memory:")` keeps
/// one, that remembers what each call was given (a test fake: the crate itself
/// implements no session).
#[derive(Default)]
pub struct MemorySession {
    items: Mutex<Vec<InputItem>>,
    adds: Mutex<Vec<Vec<InputItem>>>,
    reads: Mutex<Vec<Option<usize>>>,
}

impl MemorySession {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// A session that already holds `items`.
    pub fn holding(items: Vec<InputItem>) -> Arc<Self> {
        let session = Self::default();
        *session.items.lock().unwrap() = items;
        Arc::new(session)
    }

    /// What each `add_items` call was given, in order.
    pub fn adds(&self) -> Vec<Vec<InputItem>> {
        self.adds.lock().unwrap().clone()
    }

    /// The `limit` of each `get_items` call, in order.
    pub fn reads(&self) -> Vec<Option<usize>> {
        self.reads.lock().unwrap().clone()
    }

    pub fn items(&self) -> Vec<InputItem> {
        self.items.lock().unwrap().clone()
    }
}

impl Session for MemorySession {
    fn get_items(
        &self,
        limit: Option<usize>,
    ) -> BoxFuture<'static, Result<Vec<InputItem>, SessionError>> {
        self.reads.lock().unwrap().push(limit);
        let items = self.items.lock().unwrap().clone();
        let items = match limit {
            Some(limit) => items[items.len().saturating_sub(limit)..].to_vec(),
            None => items,
        };
        async move { Ok(items) }.boxed()
    }

    fn add_items(&self, items: Vec<InputItem>) -> BoxFuture<'static, Result<(), SessionError>> {
        self.adds.lock().unwrap().push(items.clone());
        self.items.lock().unwrap().extend(items);
        async { Ok(()) }.boxed()
    }
}

async fn gather(mut handle: RunHandle, collector: Arc<Collector>) -> Outcome {
    let mut events = Vec::new();
    while let Some(event) = handle.events.next().await {
        events.push(event);
    }
    let result = handle.result.await.expect("the run task must not panic");
    let mut names: Vec<&'static str> = Vec::new();
    // The SDK's stream has no counterpart of the port's own events.
    for event in events.iter().filter(|event| !event.is_native_only()) {
        let name = event.sdk_name();
        if name == "raw_text_delta" && names.last() == Some(&"raw_text_delta") {
            continue;
        }
        names.push(name);
    }
    Outcome {
        events,
        names,
        result,
        collector,
    }
}

pub fn scripted(steps: Vec<ScriptedStep>) -> Arc<ScriptedModel> {
    Arc::new(ScriptedModel::new(steps))
}

// The scenarios' tools, the same behaviours `tools/lattice_sdk_goldens.py` registers.

pub fn echo() -> FunctionTool {
    FunctionTool::new(
        "echo",
        "Repeat the text back.",
        strict_object_schema(json!({"text": {"type": "string"}}), &["text"]),
        |_, arguments| async move { Ok(format!("echo:{}", arguments["text"].as_str().unwrap_or(""))) },
    )
}

pub fn weather() -> FunctionTool {
    FunctionTool::new(
        "weather",
        "The weather in a city.",
        strict_object_schema(json!({"city": {"type": "string"}}), &["city"]),
        |_, arguments| async move {
            Ok(format!(
                "sunny in {}",
                arguments["city"].as_str().unwrap_or("")
            ))
        },
    )
}

pub fn clock() -> FunctionTool {
    FunctionTool::new(
        "clock",
        "The time in a city.",
        strict_object_schema(json!({"city": {"type": "string"}}), &["city"]),
        |_, arguments| async move {
            Ok(format!(
                "noon in {}",
                arguments["city"].as_str().unwrap_or("")
            ))
        },
    )
}

pub fn lookup() -> FunctionTool {
    FunctionTool::new(
        "lookup",
        "Look something up.",
        strict_object_schema(json!({"query": {"type": "string"}}), &["query"]),
        |_, arguments| async move {
            Ok(format!(
                "found:{}",
                arguments["query"].as_str().unwrap_or("")
            ))
        },
    )
}

pub fn explode() -> FunctionTool {
    FunctionTool::new(
        "explode",
        "A tool that always fails.",
        strict_object_schema(json!({"reason": {"type": "string"}}), &["reason"]),
        |_, arguments| async move {
            Err(ToolError::new(format!(
                "boom: {}",
                arguments["reason"].as_str().unwrap_or("")
            )))
        },
    )
}

/// A tool every call of which needs approval (the generator's `delete_file`,
/// `needs_approval=True`).
pub fn delete_file() -> FunctionTool {
    FunctionTool::new(
        "delete_file",
        "Delete a file (it only says so).",
        strict_object_schema(json!({"path": {"type": "string"}}), &["path"]),
        |_, arguments| async move {
            Ok(format!(
                "deleted {}",
                arguments["path"].as_str().unwrap_or("")
            ))
        },
    )
    .with_needs_approval(NeedsApproval::Always)
}

/// A tool whose calls need approval only for `secret.txt` (the generator's
/// `remove`, whose `needs_approval` is a predicate).
pub fn remove() -> FunctionTool {
    FunctionTool::new(
        "remove",
        "Remove a file (it only says so); only secret.txt needs approval.",
        strict_object_schema(json!({"path": {"type": "string"}}), &["path"]),
        |_, arguments| async move {
            Ok(format!(
                "removed {}",
                arguments["path"].as_str().unwrap_or("")
            ))
        },
    )
    .with_needs_approval(NeedsApproval::Predicate(Arc::new(|_, arguments| {
        arguments["path"] == json!("secret.txt")
    })))
}

/// A port that answers each request with the next of its decisions, in order,
/// and remembers what it was asked.
pub struct ScriptedApprovals {
    decisions: Mutex<std::collections::VecDeque<ApprovalDecision>>,
    asked: Mutex<Vec<ApprovalRequest>>,
}

impl ScriptedApprovals {
    pub fn new(decisions: Vec<ApprovalDecision>) -> Arc<Self> {
        Arc::new(Self {
            decisions: Mutex::new(decisions.into()),
            asked: Mutex::new(Vec::new()),
        })
    }

    pub fn asked(&self) -> Vec<ApprovalRequest> {
        self.asked.lock().unwrap().clone()
    }
}

impl ApprovalPort for ScriptedApprovals {
    fn request(&self, request: ApprovalRequest) -> BoxFuture<'static, ApprovalDecision> {
        self.asked.lock().unwrap().push(request);
        // A request nobody scripted is refused, never approved.
        let decision = self
            .decisions
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(ApprovalDecision::Reject { note: None });
        async move { decision }.boxed()
    }
}

/// A model that answers from a queue of prepared responses, records what it is
/// asked, and lets a test choose how its generation spans are filled in and
/// whether it reports usage (the scripted model always reports some).
pub struct PlainModel {
    name: String,
    trace: lattice_agents::GenerationTrace,
    config: Value,
    responses: Mutex<
        std::collections::VecDeque<
            Result<lattice_agents::ModelResponse, lattice_agents::ModelError>,
        >,
    >,
    requests: Mutex<Vec<lattice_agents::ModelRequest>>,
}

impl PlainModel {
    pub fn new(
        trace: lattice_agents::GenerationTrace,
        responses: Vec<Result<lattice_agents::ModelResponse, lattice_agents::ModelError>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            name: "plain-model".into(),
            trace,
            config: json!({"base_url": "http://127.0.0.1:1/v1"}),
            responses: Mutex::new(responses.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    pub fn requests(&self) -> Vec<lattice_agents::ModelRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl lattice_agents::Model for PlainModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn config_for_trace(&self) -> Value {
        self.config.clone()
    }

    fn generation_trace(&self) -> lattice_agents::GenerationTrace {
        self.trace
    }

    fn stream(
        &self,
        request: lattice_agents::ModelRequest,
    ) -> futures::stream::BoxStream<
        'static,
        Result<lattice_agents::ModelEvent, lattice_agents::ModelError>,
    > {
        self.requests.lock().unwrap().push(request);
        let next = self.responses.lock().unwrap().pop_front();
        let events: Vec<Result<lattice_agents::ModelEvent, lattice_agents::ModelError>> = match next
        {
            None => vec![Err(lattice_agents::ModelError::Failed(
                "no prepared response".into(),
            ))],
            Some(Err(error)) => vec![Err(error)],
            Some(Ok(response)) => {
                let mut events = Vec::new();
                for item in &response.output {
                    if let lattice_agents::OutputItem::Message { text } = item {
                        events.push(Ok(lattice_agents::ModelEvent::TextDelta(text.clone())));
                    }
                }
                events.push(Ok(lattice_agents::ModelEvent::Done(response)));
                events
            }
        };
        futures::stream::iter(events).boxed()
    }
}

/// A response of one text message.
pub fn text_response(
    text: &str,
    usage: Option<lattice_protocol::Usage>,
) -> lattice_agents::ModelResponse {
    lattice_agents::ModelResponse {
        output: vec![lattice_agents::OutputItem::Message {
            text: text.to_owned(),
        }],
        usage,
    }
}

/// A named agent with tools.
pub fn agent_with(name: &str, instructions: &str, tools: Vec<FunctionTool>) -> Arc<Agent> {
    Agent::builder(name)
        .instructions(instructions)
        .tools(tools)
        .build()
}
