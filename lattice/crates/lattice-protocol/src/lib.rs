//! The data contract between Lattice's agent runtime and its interface.
//!
//! Everything the interface draws arrives as one of these types, and nothing
//! else crosses from the runtime (`lattice-core`, `lattice-agents`) into the
//! window (`lattice-app`). Keeping the contract in its own dependency-light
//! crate lets the interface be built and tested against a fake service, and
//! lets the runtime be tested without a window.
//!
//! Spans keep the OpenAI Agents SDK's export shape (`Span.export()` in
//! openai-agents-python 0.22.3: `object`, `id`, `trace_id`, `parent_id`,
//! `started_at`, `ended_at`, `span_data`, `error`). A trace recorded here is
//! therefore readable by anything that reads the SDK's traces, and the port's
//! conformance tests can compare its spans with the Python SDK's field by field.
//!
//! Invariants every producer keeps:
//! - Event `seq` numbers start at 1 and increase by one per run, in the order
//!   the events happened. A consumer that has seen `seq = n` asks for events
//!   after `n` and misses nothing.
//! - A span's `order` is assigned when it starts and never changes; parents
//!   start before their children, so sorting by `order` is a valid tree order
//!   even when two timestamps are equal (Windows clocks tick at ~15 ms).
//! - A `SpanEnd` for a span supersedes its `SpanStart`: the SDK fills span data
//!   in after the start, so only the end record is authoritative.
//! - Strings in events and spans are already bounded by the producer.
//! - No API key, key name or credential-bearing URL appears in any value here.
//!
//! [`chat`] holds Chat's types (threads, turns, choices, answer jobs) and the
//! `ChatService` trait. Its vocabulary is re-exported here; its request and
//! response envelopes (`chat::SendRequest`, `chat::RegenerateRequest`,
//! `chat::Accepted`) stay module-qualified, because the conversation service
//! has envelopes of its own by those names.
//!
//! [`conversation`] holds the agent chat's types (conversations, their
//! events, staged changes, approvals, checkpoints) and the `AgentChatService`
//! trait. They stay module-qualified.

use std::fmt;

pub mod chat;
pub mod conversation;

pub use chat::{
    ChatChoice, ChatEvent, ChatEventKind, ChatService, ChatTurn, Fact, JobId, LocalRuntime,
    OpenedThread, Role, RuntimeState, Shown, Stage, ThreadId, ThreadList, ThreadSummary, TurnId,
    is_job_id, is_thread_id,
};

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A run's identifier: 16 lowercase hexadecimal characters.
pub type RunId = String;

/// True when `id` has the shape of a [`RunId`]. Ids reach paths on disk, so
/// anything else is refused before it is used.
pub fn is_run_id(id: &str) -> bool {
    id.len() == 16
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Where a model runs. Computed from the endpoint's address (loopback or
/// not), never from a label a person typed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locality {
    /// The model runs on this machine; nothing leaves it.
    Local,
    /// The task, tool results and the conversation go to another machine.
    Remote,
}

/// What kind of server a model choice talks to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    /// Retired (ADR-0041): a saved Ollama row, shown but never ready.
    Ollama,
    OpenaiCompatible,
    Anthropic,
    /// The platform's own llama.cpp server (ADR-0041), which the platform
    /// starts on loopback with a per-launch token.
    Llamacpp,
    /// The scripted development model: not a language model.
    Development,
}

/// One entry of the model picker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelChoice {
    /// `endpoint:<id>` (the registry's rows, `endpoint:llamacpp-local` being
    /// the platform's managed llama.cpp server) or `dev:scripted`.
    pub id: String,
    pub label: String,
    pub kind: ModelKind,
    pub locality: Locality,
    pub ready: bool,
    /// Why the choice cannot be used, in one sentence, when it cannot.
    pub refusal: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Start,
    End,
    Agent,
    Tool,
    Mcp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Start,
    Tool,
    Mcp,
    Handoff,
    End,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub kind: NodeKind,
    pub label: String,
    /// The agent a run starts with.
    pub root: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub kind: EdgeKind,
}

/// An agent definition's static structure: the walk the SDK's
/// `agents.extensions.visualization` makes (agents, their tools, MCP servers
/// and handoffs), as data instead of Graphviz DOT.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentGraph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

/// One entry of the agent picker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub label: String,
    pub description: String,
    pub graph: AgentGraph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    /// A guardrail refused the task before any model saw it.
    Refused,
    Stopped,
    /// The run was working when Lattice last closed.
    Interrupted,
}

impl RunStatus {
    pub fn is_active(self) -> bool {
        matches!(self, RunStatus::Running)
    }
}

/// Token counts summed over a run's model calls. Absent (not zero) when the
/// model server did not report usage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub requests: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

/// A run as the list shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    pub id: RunId,
    pub task: String,
    pub agent: String,
    pub agent_label: String,
    pub model: String,
    pub model_label: String,
    pub locality: Locality,
    pub status: RunStatus,
    /// Seconds since the Unix epoch.
    pub created_at: f64,
    pub updated_at: f64,
    pub ended_at: Option<f64>,
    pub trace_id: String,
    pub usage: Option<Usage>,
    /// The final output, bounded for the list (the full text is in the events).
    pub output: Option<String>,
    pub error: Option<String>,
    /// How many spans have ended.
    pub spans: u32,
}

/// A trace's own record. The SDK's `Trace.export()` carries no times, so
/// these are stamped when the trace starts and ends.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceInfo {
    pub id: String,
    pub workflow_name: String,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
}

/// The error the SDK attaches to a span.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpanError {
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

/// One span, in the SDK's export shape plus the start `order`.
///
/// `span_data` is kept as JSON because the SDK's span types are an open set
/// (`agent`, `function`, `generation`, `response`, `handoff`, `custom`,
/// `guardrail`, `mcp_tools`, `transcription`, `speech`, `speech_group`); the
/// accessors below read the fields the interface needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpanRecord {
    pub order: u64,
    pub id: String,
    pub trace_id: String,
    pub parent_id: Option<String>,
    /// ISO 8601 in UTC with microseconds, as Python's `isoformat()` writes it.
    pub started_at: String,
    /// None while the span is open.
    pub ended_at: Option<String>,
    pub span_data: Value,
    pub error: Option<SpanError>,
}

impl SpanRecord {
    /// `span_data.type`, or `"unknown"`.
    pub fn data_type(&self) -> &str {
        self.span_data
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
    }

    /// A string field of `span_data`.
    pub fn data_str(&self, key: &str) -> Option<&str> {
        self.span_data.get(key).and_then(Value::as_str)
    }

    /// For the SDK's `custom` spans that stand for its task and turn spans:
    /// `span_data.data.sdk_span_type`.
    pub fn sdk_span_type(&self) -> Option<&str> {
        self.span_data.get("data")?.get("sdk_span_type")?.as_str()
    }
}

/// The status a run ended with, as its `End` event carries it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndStatus {
    Completed,
    Failed,
    Refused,
    Stopped,
}

impl From<EndStatus> for RunStatus {
    fn from(end: EndStatus) -> Self {
        match end {
            EndStatus::Completed => RunStatus::Completed,
            EndStatus::Failed => RunStatus::Failed,
            EndStatus::Refused => RunStatus::Refused,
            EndStatus::Stopped => RunStatus::Stopped,
        }
    }
}

/// What happened in a run. The names follow the SDK's stream events
/// (`agent_updated_stream_event`, `run_item_stream_event` with
/// `message_output_created`, `tool_called`, `tool_output`,
/// `handoff_requested`, `handoff_occured`, `reasoning_item_created`, and the
/// raw `response.output_text.delta`), plus the trace processor's callbacks.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunEventKind {
    /// Streamed text. Live only: never written to disk.
    Delta {
        agent: String,
        text: String,
    },
    /// The agent now answering (at the start, and after each handoff).
    Agent {
        name: String,
    },
    /// A complete message from an agent.
    Message {
        agent: String,
        text: String,
    },
    ToolCall {
        agent: String,
        name: String,
        call_id: String,
        arguments: String,
    },
    ToolOutput {
        agent: String,
        call_id: String,
        output: String,
    },
    HandoffRequested {
        agent: String,
    },
    Handoff {
        from: String,
        to: String,
    },
    Reasoning {
        agent: String,
        text: String,
    },
    /// A guardrail refused the task. `message` never repeats what matched.
    Guardrail {
        name: String,
        message: String,
    },
    Result {
        output: String,
        usage: Option<Usage>,
        turns: u32,
        last_agent: String,
    },
    Error {
        message: String,
    },
    /// Always the last event of a run.
    End {
        status: EndStatus,
    },
    // `RunEvent` flattens its kind, and `RunEvent` has an `at` of its own (seconds).
    // The trace's clock text is `at` in Rust and `started_at` / `ended_at` in JSON,
    // as in `TraceInfo`; two `at` keys in one object could not be read back.
    TraceStart {
        trace_id: String,
        workflow_name: String,
        #[serde(rename = "started_at")]
        at: String,
    },
    TraceEnd {
        trace_id: String,
        #[serde(rename = "ended_at")]
        at: String,
    },
    SpanStart {
        span: SpanRecord,
    },
    SpanEnd {
        span: SpanRecord,
    },
}

/// One event of one run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunEvent {
    pub seq: u64,
    /// Seconds since the Unix epoch.
    pub at: f64,
    #[serde(flatten)]
    pub kind: RunEventKind,
}

/// A run with everything recorded about it so far.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunDetail {
    pub run: RunSummary,
    pub trace: Option<TraceInfo>,
    /// Latest record per span (an end supersedes a start), sorted by `order`.
    pub spans: Vec<SpanRecord>,
    /// Every recorded event except `Delta`, in `seq` order.
    pub events: Vec<RunEvent>,
}

/// What the person asked for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartRun {
    pub task: String,
    pub agent: String,
    pub model: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefusalKind {
    /// The request itself was wrong (an empty task, an unknown agent, a model
    /// that is not ready).
    Invalid,
    /// Too many runs are already working, or the run is not running.
    Conflict,
    /// The runtime cannot run agents at all right now.
    Unavailable,
    NotFound,
}

/// A refusal the interface shows as it is: one sentence for a person.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Refusal {
    pub kind: RefusalKind,
    pub message: String,
}

impl Refusal {
    pub fn new(kind: RefusalKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Refusal {}

/// What the runtime says about itself.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceStatus {
    /// For example "lattice-agents 0.1.0 (a port of openai-agents 0.22.3)".
    pub runtime: String,
    /// Where traces go. Always "this machine": the runtime has no exporter.
    pub traces: String,
    /// Why runs cannot start, when they cannot.
    pub refusal: Option<String>,
}

/// The runtime as the interface sees it. `lattice-core` implements it over
/// real runs; `lattice-app` also has an in-memory fake for tests and demos.
///
/// Every method returns promptly: starting a run schedules it and returns its
/// summary, and progress arrives through [`RunService::follow`].
pub trait RunService: Send + Sync + 'static {
    fn status(&self) -> ServiceStatus;
    fn agents(&self) -> Vec<AgentInfo>;
    fn models(&self) -> Vec<ModelChoice>;
    /// Newest first.
    fn runs(&self) -> Vec<RunSummary>;
    fn run(&self, id: &str) -> Result<RunDetail, Refusal>;
    fn start(&self, request: StartRun) -> Result<RunSummary, Refusal>;
    fn stop(&self, id: &str) -> Result<(), Refusal>;
    /// The run's events with `seq > after`, in order, including live
    /// `Delta`s, arriving in batches (a batch is whatever accumulated since the
    /// consumer last polled, so a burst of streamed text costs one wake-up, not
    /// one per token). The stream ends after the batch holding `End`, or at
    /// once for a run that already ended and has nothing after `after`.
    fn follow(&self, id: &str, after: u64) -> Result<BoxStream<'static, Vec<RunEvent>>, Refusal>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_serialize_with_a_flat_type_tag() {
        let event = RunEvent {
            seq: 3,
            at: 1.5,
            kind: RunEventKind::Handoff {
                from: "Lattice assistant".into(),
                to: "Model advisor".into(),
            },
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "handoff");
        assert_eq!(json["seq"], 3);
        assert_eq!(json["from"], "Lattice assistant");
        let back: RunEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
    }

    #[test]
    fn every_kind_of_event_survives_a_json_round_trip_with_each_key_once() {
        let span = SpanRecord {
            order: 1,
            id: "span_1".into(),
            trace_id: "trace_1".into(),
            parent_id: None,
            started_at: "2026-09-30T05:21:05.123456+00:00".into(),
            ended_at: None,
            span_data: serde_json::json!({"type": "agent", "name": "A"}),
            error: None,
        };
        let usage = Usage {
            requests: 1,
            input_tokens: 2,
            output_tokens: 3,
            total_tokens: 5,
        };
        let kinds = vec![
            RunEventKind::Delta {
                agent: "A".into(),
                text: "t".into(),
            },
            RunEventKind::Agent { name: "A".into() },
            RunEventKind::Message {
                agent: "A".into(),
                text: "t".into(),
            },
            RunEventKind::ToolCall {
                agent: "A".into(),
                name: "n".into(),
                call_id: "c".into(),
                arguments: "{}".into(),
            },
            RunEventKind::ToolOutput {
                agent: "A".into(),
                call_id: "c".into(),
                output: "o".into(),
            },
            RunEventKind::HandoffRequested { agent: "A".into() },
            RunEventKind::Handoff {
                from: "A".into(),
                to: "B".into(),
            },
            RunEventKind::Reasoning {
                agent: "A".into(),
                text: "t".into(),
            },
            RunEventKind::Guardrail {
                name: "g".into(),
                message: "m".into(),
            },
            RunEventKind::Result {
                output: "o".into(),
                usage: Some(usage),
                turns: 1,
                last_agent: "A".into(),
            },
            RunEventKind::Error {
                message: "m".into(),
            },
            RunEventKind::End {
                status: EndStatus::Refused,
            },
            RunEventKind::TraceStart {
                trace_id: "trace_1".into(),
                workflow_name: "w".into(),
                at: "2026-09-30T05:21:05.000000+00:00".into(),
            },
            RunEventKind::TraceEnd {
                trace_id: "trace_1".into(),
                at: "2026-09-30T05:21:06.000000+00:00".into(),
            },
            RunEventKind::SpanStart { span: span.clone() },
            RunEventKind::SpanEnd { span },
        ];
        for (index, kind) in kinds.into_iter().enumerate() {
            let event = RunEvent {
                seq: index as u64 + 1,
                at: 1_790_000_000.5,
                kind,
            };
            let text = serde_json::to_string(&event).unwrap();
            // A key written twice would read back as an error.
            let back: RunEvent =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(back, event, "{text}");
            assert_eq!(
                text.matches("\"at\":").count(),
                1,
                "the event's own `at` is the only `at` key: {text}"
            );
        }
    }

    #[test]
    fn a_span_reads_its_sdk_fields() {
        let span: SpanRecord = serde_json::from_value(serde_json::json!({
            "order": 2, "id": "span_1", "trace_id": "trace_1", "parent_id": null,
            "started_at": "2026-09-30T05:21:05.123456+00:00", "ended_at": null,
            "span_data": {"type": "custom", "name": "turn", "data": {"sdk_span_type": "turn", "turn": 1}},
            "error": null
        }))
        .unwrap();
        assert_eq!(span.data_type(), "custom");
        assert_eq!(span.data_str("name"), Some("turn"));
        assert_eq!(span.sdk_span_type(), Some("turn"));
    }

    #[test]
    fn run_ids_are_sixteen_lowercase_hex_characters() {
        assert!(is_run_id("0123456789abcdef"));
        assert!(!is_run_id("0123456789ABCDEF"));
        assert!(!is_run_id("../../etc/passwd0"));
        assert!(!is_run_id("0123456789abcde"));
    }
}
