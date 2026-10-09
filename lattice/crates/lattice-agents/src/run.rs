//! The agent loop: streamed runs, turns, tools, handoffs, guardrails, tracing.
//!
//! Ports `AgentRunner.run_streamed` and the streaming loop behind it
//! (`run_internal/run_loop.py: start_streaming, run_single_turn_streamed`,
//! `turn_resolution.py: process_model_response, execute_tools_and_side_effects,
//! execute_handoffs`, `tool_execution.py`, `guardrails.py`, `streaming.py`;
//! 0.22.3), for output type `str` and `tool_use_behavior="run_llm_again"`.
//!
//! One run is one tokio task. It owns a trace and this span tree, exactly the
//! SDK's:
//!
//! ```text
//! task                      one per run (unless include_task_and_turn_spans is off)
//!   agent                   one per agent that runs; a handoff ends it and the
//!                           next agent's span is a sibling under the task
//!     guardrail             the first agent's input guardrails, before turn 1
//!     turn                  one per model call round trip
//!       generation          the model call
//!       function            one per tool call, started in call order
//!       handoff             from the agent that asked to the agent that answers
//!     guardrail             the final output's output guardrails, after the last turn
//! ```
//!
//! Input: a run starts from a list of input items ([`run_streamed_items`]), as
//! the SDK's `Runner.run_streamed(agent, input: str | list[...])` does; a string
//! ([`run_streamed`]) is the list of one user message. Every model call is given
//! those items first, in order and unchanged, then what the run has added since.
//! A run with no input items fails with `RunError::User("the run has no input")`
//! before anything starts: no trace, no span, no event.
//!
//! Turn semantics, in the SDK's order:
//! 1. The turn counter counts model round trips across the whole run, handoffs
//!    included. Starting turn `n > max_turns` is an error: the agent span gets
//!    `Max turns exceeded` and the run ends with `MaxTurnsExceeded`.
//! 2. Before the first turn only, the first agent's input guardrails run, one at
//!    a time. The SDK's default runs them alongside the first model call; its
//!    `run_in_parallel=False` runs them first. This port always runs them first,
//!    so a task a guardrail would refuse is never sent to a model. One visible
//!    difference from `run_in_parallel=False`: the SDK starts all of an agent's
//!    input guardrails together, so when one trips the others' spans exist too;
//!    here the guardrails after the tripping one never run and have no span.
//! 3. The model answers. Its output items become run items and events in the
//!    order it produced them (messages, tool calls, handoff calls, reasoning).
//! 4. Function calls run concurrently; their outputs are appended in call order
//!    (when some need approval, theirs come after the others'; see below).
//! 5. If the response called handoff tools, the FIRST is executed (handoff span,
//!    `HandoffCall`/`HandoffOutput` items, the next agent takes over); every
//!    other handoff call in the response is answered with "Multiple handoffs
//!    detected, ignoring this one." and the handoff span records the error
//!    `Multiple handoffs requested`. Function calls in the same response still
//!    run first, as in the SDK.
//! 6. Otherwise, a response with no function calls is the final output: the text
//!    of its last message, or the empty string when it had none. The agent's
//!    output guardrails run over it, alongside each other; a tripwire ends the
//!    run and the answer is not returned.
//! 7. Otherwise the loop runs again.
//!
//! Failure semantics: a tool that fails is answered to the model (see
//! [`crate::tool`]) and the run goes on. A model that calls a tool the agent does
//! not have is `ModelBehaviorError`: the turn span records `Tool not found` and
//! the run ends. So is a call with no id, and one id used for two different
//! calls in a response; an exact repeat of a call in the same response (same id,
//! tool and arguments) is skipped, as if made once. A response that declines to
//! answer (a refusal, or an answer the provider filtered away) and asks for no
//! tool or handoff ends the run with `ModelRefusalError`. A model failure ends the
//! run and the agent span records `Error in agent run`. A guardrail tripwire ends
//! the run and the agent span records `Guardrail tripwire triggered`.
//!
//! Tool approval (see [`crate::tool`] for the decision and its text): a call
//! whose tool needs approval waits IN PLACE, inside its own function span and
//! its own future, for [`RunConfig::approvals`]; the other calls of the same
//! response keep running. Outputs follow the SDK's order (spec 22.6 A3a,
//! pinned by the golden `approval_needed_before_free_calls`): once every call
//! that needs no approval has finished, their outputs are appended and
//! announced in call order, while the approvals may still wait; then, when
//! every decided call has finished, theirs follow in call order. With a
//! session, the turn's items up to the free calls' outputs are saved at that
//! point, as the SDK saves them at its interruption, so a stop during the
//! wait loses no output of a call that ran. The run sends
//! `ToolApprovalRequested` when it asks and `ToolApprovalResolved` when the
//! answer comes. Both are native-only events: the SDK streams no
//! approval event ("approvals represent interruptions, not streamed items").
//! An immediate cancel drops the waiting call, whose function span then ends
//! with the error `Cancelled while awaiting approval`.
//!
//! Steering (a native addition; the SDK has none): [`RunControl::steer`] hands
//! the run a message while it works. Steers are taken, in the order they were
//! sent, just before each model call (so after the previous turn's tool
//! results), appended to the history as user messages and announced as
//! `Steered`. A steer sent while the run's last model call is answering cannot
//! reach a model any more: when that call turns out to be the final one, the
//! run takes whatever is waiting and closes steering in one step, and returns
//! those messages in [`RunResult::unsent_steers`]. After that, and after any
//! other ending, `steer` gives the message back (`Steer::Closed`); a run that
//! ends otherwise keeps the steers it accepted but never sent for
//! [`RunControl::take_unsent_steers`], including one whose output guardrail
//! trips, or that is cancelled while its output guardrails or its final save
//! are awaited, after the final model call. No steer is lost.
//!
//! Cancellation: [`CancelMode::Immediate`] aborts the model stream and any
//! running tools at once (dropped futures close their spans) and the run ends
//! with `Cancelled`; [`CancelMode::AfterTurn`] lets the current turn finish and
//! stops before the next one. A run that had already produced its final output
//! completes normally.
//!
//! Invariants:
//! - Events reach the consumer in the order they happen and the event stream
//!   ends when the run task does (after its final span and the trace have ended).
//! - Every span that starts ends, on every path including errors and
//!   cancellation; the trace ends last.
//! - A consumer that stops reading events does not stop the run: events are
//!   buffered without bound, one per run item and one per text chunk.
//!
//! Deviations, and why:
//! - Errors are values, not exceptions: [`RunError`] replaces the SDK's
//!   exception classes, `MaxTurnsExceeded` and the tripwire exceptions keep their
//!   names.
//! - Stream events for the model's raw response frames (`response.created` and
//!   the rest) do not exist; only text deltas ([`StreamEvent::RawTextDelta`]).
//! - Tool calls run concurrently on the run's task, not on separate threads: a
//!   handler must not block (use `tokio::task::spawn_blocking` inside it).
//! - A call that needs approval is decided within the run. The SDK interrupts
//!   the run there and resumes it as a second run from its saved state, so it
//!   records two task spans, two agent spans and, for an approved call, two
//!   function spans (the interrupted one with `output: null`, then the resumed
//!   one); this port records one of each, with turns numbered across the
//!   wait, and a rejected call's span keeps `output: null` as the SDK's one
//!   span does. The goldens `approval_*` are the SDK's two runs stitched into
//!   one by the generator (`tools/lattice_sdk_goldens.py`), and the comparison
//!   leaves the native-only events out of the stream.
//! - Input guardrails are given the text of the LAST user item of the run's
//!   input (the empty string when it has none). The SDK gives them the whole
//!   input list; this port's guardrails take a string (see [`crate::guardrail`]).
//!   Pinned by the golden `list_input_guardrail_on_last_user`.

use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};
use futures::future::{BoxFuture, Either};
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt};
use lattice_protocol::Usage;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::agent::Agent;
use crate::chat_completions::chat_messages_for_trace;
use crate::guardrail::{GuardrailContext, GuardrailOutput};
use crate::handoff::Handoff;
use crate::model::{
    GenerationTrace, InputItem, Model, ModelError, ModelEvent, ModelRequest, ModelResponse,
    ModelSettings, OutputItem, ToolCallItem, ToolSpec,
};
use crate::session::{Session, SessionError};
use crate::tool::{
    ApprovalDecision, ApprovalPort, ApprovalRequest, FunctionTool, NO_ONE_ASKED, NeedsApproval,
    ToolContext, invoke, parse_arguments, rejection_output,
};
use crate::trace::{
    Span, TraceProcessor, Tracer, agent_data, function_data, generation_data, guardrail_data,
    handoff_data, span_error, task_data, turn_data,
};
use crate::usage;

/// The value a caller shares with its tools and guardrails (`RunConfig::context`).
pub type RunContext = Arc<dyn Any + Send + Sync>;

/// Text of the SDK's answer to every handoff call after the first in a response.
const MULTIPLE_HANDOFFS_OUTPUT: &str = "Multiple handoffs detected, ignoring this one.";
/// What a trace shows instead of an error's text when sensitive data is off.
const REDACTED_TRACE_ERROR: &str = "Error details are redacted.";
/// The id the SDK gives items of a Chat Completions response
/// (`FAKE_RESPONSES_ID`), used in the recorded generation output.
const FAKE_RESPONSES_ID: &str = "__fake_id__";
/// What a run given no input items fails with.
const NO_INPUT: &str = "the run has no input";
/// The error a function span ends with when its call is dropped while it
/// waits for approval.
const CANCELLED_WHILE_WAITING: &str = "Cancelled while awaiting approval";

#[derive(Clone)]
pub struct RunConfig {
    /// The model every agent uses. Required: the port has no default model
    /// provider, so nothing can fall back to a remote service.
    pub model: Arc<dyn Model>,
    /// Settings that override each agent's, field by field.
    pub model_settings: Option<ModelSettings>,
    /// The most model round trips a run may make (default 10).
    pub max_turns: u32,
    /// Names the trace and the task span (default "Agent workflow").
    pub workflow_name: String,
    pub trace_id: Option<String>,
    pub group_id: Option<String>,
    pub trace_metadata: Option<Value>,
    /// Record tool arguments and outputs and model input and output in spans
    /// (default true). Off blanks those fields and redacts error text.
    pub include_sensitive_data: bool,
    /// Create the task and turn spans (default true, as the SDK).
    pub include_task_and_turn_spans: bool,
    /// Where the trace goes. Nothing else receives it.
    pub processors: Vec<Arc<dyn TraceProcessor>>,
    /// Shared with tools and guardrails.
    pub context: RunContext,
    /// Where the conversation is kept between runs (default none): read at
    /// the start, written with the input and after each completed turn (see
    /// [`crate::session`]).
    pub session: Option<Arc<dyn Session>>,
    /// Where a call whose tool needs approval is decided (default none: such
    /// a call is rejected, "No one was asked.").
    pub approvals: Option<Arc<dyn ApprovalPort>>,
    /// A falsifier's mutant (test builds only).
    #[cfg(test)]
    pub(crate) mutant: Option<Mutant>,
}

impl RunConfig {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            model_settings: None,
            max_turns: 10,
            workflow_name: "Agent workflow".to_owned(),
            trace_id: None,
            group_id: None,
            trace_metadata: None,
            include_sensitive_data: true,
            include_task_and_turn_spans: true,
            processors: Vec::new(),
            context: Arc::new(()),
            session: None,
            approvals: None,
            #[cfg(test)]
            mutant: None,
        }
    }
}

impl fmt::Debug for RunConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunConfig")
            .field("model", &self.model.name())
            .field("max_turns", &self.max_turns)
            .field("workflow_name", &self.workflow_name)
            .field("include_sensitive_data", &self.include_sensitive_data)
            .field(
                "include_task_and_turn_spans",
                &self.include_task_and_turn_spans,
            )
            .field("processors", &self.processors.len())
            .field("session", &self.session.is_some())
            .field("approvals", &self.approvals.is_some())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelMode {
    /// Abort the model call and any tools at once.
    Immediate,
    /// Let the turn in progress finish, then stop.
    AfterTurn,
}

struct ControlInner {
    immediate: CancellationToken,
    after_turn: AtomicBool,
    processor_panics: Arc<AtomicU64>,
    steering: Mutex<Steering>,
}

/// Messages steered into a run: waiting for the next model call while the run
/// works; once it has ended, the ones it accepted and never sent.
enum Steering {
    Open(VecDeque<String>),
    Closed(Vec<String>),
}

/// Why a steer was not taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Steer {
    /// The run has ended, or its final model call has answered: the message
    /// is given back, to be sent as the next message.
    Closed(String),
}

impl fmt::Display for Steer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Steer::Closed(_) => f.write_str("the run takes no more messages"),
        }
    }
}

impl std::error::Error for Steer {}

/// A handle that stops a run. Cheap to clone; safe to use from any thread.
#[derive(Clone)]
pub struct RunControl {
    inner: Arc<ControlInner>,
}

impl RunControl {
    fn new(processor_panics: Arc<AtomicU64>) -> Self {
        Self {
            inner: Arc::new(ControlInner {
                immediate: CancellationToken::new(),
                after_turn: AtomicBool::new(false),
                processor_panics,
                steering: Mutex::new(Steering::Open(VecDeque::new())),
            }),
        }
    }

    /// Ask the run to stop. Idempotent; a stop asked for after the run finished
    /// does nothing.
    pub fn cancel(&self, mode: CancelMode) {
        match mode {
            CancelMode::Immediate => self.inner.immediate.cancel(),
            CancelMode::AfterTurn => self.inner.after_turn.store(true, Ordering::SeqCst),
        }
    }

    /// How many times a trace processor panicked (each call that panicked
    /// counts once). The run survives them; this is how they are noticed.
    pub fn processor_panics(&self) -> u64 {
        self.inner.processor_panics.load(Ordering::Relaxed)
    }

    /// Give the run a message to see before its next model call. `Err` gives
    /// the message back when the run takes no more (it has ended, or its final
    /// model call has answered).
    pub fn steer(&self, text: String) -> Result<(), Steer> {
        match &mut *self.steering() {
            Steering::Open(waiting) => {
                waiting.push_back(text);
                Ok(())
            }
            Steering::Closed(_) => Err(Steer::Closed(text)),
        }
    }

    /// After a run has ended without returning a result (an error, a
    /// cancellation): the steers it accepted but never gave a model, in the
    /// order they were sent. Taking them leaves none; while the run works there
    /// are none. A run that returned a result has them in
    /// [`RunResult::unsent_steers`] instead.
    pub fn take_unsent_steers(&self) -> Vec<String> {
        match &mut *self.steering() {
            Steering::Open(_) => Vec::new(),
            Steering::Closed(unsent) => std::mem::take(unsent),
        }
    }

    fn after_turn_requested(&self) -> bool {
        self.inner.after_turn.load(Ordering::SeqCst)
    }

    fn steering(&self) -> std::sync::MutexGuard<'_, Steering> {
        self.inner
            .steering
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The steers waiting now, in order (none once steering is closed).
    fn take_steers(&self) -> Vec<String> {
        match &mut *self.steering() {
            Steering::Open(waiting) => waiting.drain(..).collect(),
            Steering::Closed(_) => Vec::new(),
        }
    }

    /// Close steering, keeping what was waiting as unsent. One lock, so no
    /// steer can arrive between the last look and the close. Idempotent.
    fn close_steering(&self) {
        let mut steering = self.steering();
        if let Steering::Open(waiting) = &mut *steering {
            let unsent = waiting.drain(..).collect();
            *steering = Steering::Closed(unsent);
        }
    }
}

impl fmt::Debug for RunControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunControl").finish_non_exhaustive()
    }
}

/// The name of a run-item stream event, as the SDK spells it (including
/// `handoff_occured`, which the SDK cannot correct without breaking callers).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RunItemName {
    #[serde(rename = "message_output_created")]
    MessageOutputCreated,
    #[serde(rename = "tool_called")]
    ToolCalled,
    #[serde(rename = "tool_output")]
    ToolOutput,
    #[serde(rename = "handoff_requested")]
    HandoffRequested,
    #[serde(rename = "handoff_occured")]
    HandoffOccurred,
    #[serde(rename = "reasoning_item_created")]
    ReasoningItemCreated,
}

impl RunItemName {
    pub fn as_str(self) -> &'static str {
        match self {
            RunItemName::MessageOutputCreated => "message_output_created",
            RunItemName::ToolCalled => "tool_called",
            RunItemName::ToolOutput => "tool_output",
            RunItemName::HandoffRequested => "handoff_requested",
            RunItemName::HandoffOccurred => "handoff_occured",
            RunItemName::ReasoningItemCreated => "reasoning_item_created",
        }
    }
}

/// One thing that happened in a run (the SDK's `RunItem` subclasses).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RunItem {
    MessageOutput {
        agent: String,
        text: String,
    },
    /// A function tool call the model made.
    ToolCall {
        agent: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    ToolOutput {
        agent: String,
        call_id: String,
        output: String,
    },
    /// A handoff tool call the model made.
    HandoffCall {
        agent: String,
        call_id: String,
        tool_name: String,
    },
    HandoffOutput {
        agent: String,
        call_id: String,
        source_agent: String,
        target_agent: String,
        output: String,
    },
    Reasoning {
        agent: String,
        text: String,
    },
}

impl RunItem {
    /// The stream event this item is announced by (`stream_step_items_to_queue`).
    pub fn event_name(&self) -> RunItemName {
        match self {
            RunItem::MessageOutput { .. } => RunItemName::MessageOutputCreated,
            RunItem::ToolCall { .. } => RunItemName::ToolCalled,
            RunItem::ToolOutput { .. } => RunItemName::ToolOutput,
            RunItem::HandoffCall { .. } => RunItemName::HandoffRequested,
            RunItem::HandoffOutput { .. } => RunItemName::HandoffOccurred,
            RunItem::Reasoning { .. } => RunItemName::ReasoningItemCreated,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A piece of the answer text, live.
    RawTextDelta {
        agent: String,
        delta: String,
    },
    /// The agent now answering: at the start, and after each handoff.
    AgentUpdated {
        agent: String,
    },
    RunItem {
        name: RunItemName,
        item: RunItem,
    },
    /// The run's session refused a write (`add_items`). Sent once per run, at
    /// the first failure; the run goes on. Native only: the SDK has no such
    /// event (see [`crate::session`]).
    SessionWriteFailed {
        message: String,
    },
    /// A call is waiting for approval. Native only (see [`crate::tool`]).
    ToolApprovalRequested {
        agent: String,
        call_id: String,
        tool: String,
        /// The parsed arguments, as the port is given them.
        arguments: Value,
    },
    /// The call's approval was decided. Native only.
    ToolApprovalResolved {
        agent: String,
        call_id: String,
        approved: bool,
    },
    /// A steered message was added to the conversation, before the model call
    /// that follows. Native only.
    Steered {
        agent: String,
        text: String,
    },
}

impl StreamEvent {
    /// The event's name in the SDK's vocabulary: the run item's name,
    /// `agent_updated_stream_event`, or `raw_text_delta` for a text delta. An
    /// event the SDK does not have is named by its own snake-case type (see
    /// [`StreamEvent::is_native_only`]).
    pub fn sdk_name(&self) -> &'static str {
        match self {
            StreamEvent::RawTextDelta { .. } => "raw_text_delta",
            StreamEvent::AgentUpdated { .. } => "agent_updated_stream_event",
            StreamEvent::RunItem { name, .. } => name.as_str(),
            StreamEvent::SessionWriteFailed { .. } => "session_write_failed",
            StreamEvent::ToolApprovalRequested { .. } => "tool_approval_requested",
            StreamEvent::ToolApprovalResolved { .. } => "tool_approval_resolved",
            StreamEvent::Steered { .. } => "steered",
        }
    }

    /// Whether the event is this port's own, with no counterpart in the SDK's
    /// stream. A comparison with the SDK's recorded stream leaves these out.
    pub fn is_native_only(&self) -> bool {
        match self {
            StreamEvent::RawTextDelta { .. }
            | StreamEvent::AgentUpdated { .. }
            | StreamEvent::RunItem { .. } => false,
            StreamEvent::SessionWriteFailed { .. }
            | StreamEvent::ToolApprovalRequested { .. }
            | StreamEvent::ToolApprovalResolved { .. }
            | StreamEvent::Steered { .. } => true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RunResult {
    pub final_output: String,
    /// The name of the agent that produced the final output.
    pub last_agent: String,
    /// Summed over the model calls that reported usage; absent when none did.
    pub usage: Option<Usage>,
    /// Model round trips made.
    pub turns: u32,
    /// Every item the run produced, in order.
    pub new_items: Vec<RunItem>,
    /// Steered messages that arrived while the final model call answered, so
    /// no model saw them; in the order they were sent (see [`RunControl::steer`]).
    pub unsent_steers: Vec<String>,
}

#[derive(Debug, Error)]
pub enum RunError {
    #[error("Max turns ({max_turns}) exceeded")]
    MaxTurnsExceeded { max_turns: u32 },
    #[error("Guardrail {guardrail} triggered tripwire")]
    InputGuardrailTripwire {
        guardrail: String,
        output_info: Value,
    },
    #[error("Guardrail {guardrail} triggered tripwire")]
    OutputGuardrailTripwire {
        guardrail: String,
        output_info: Value,
    },
    #[error(transparent)]
    Model(#[from] ModelError),
    /// The model declined to answer, or the provider withheld the answer, and
    /// nothing else in the response asked for a tool or a handoff
    /// (`ModelRefusalError`, `turn_resolution.py:952-988`).
    #[error("Model refused to produce output: {refusal}")]
    ModelRefusal { refusal: String },
    #[error("the run was cancelled")]
    Cancelled,
    #[error("{0}")]
    User(String),
}

impl RunError {
    /// The SDK's exception class this error stands for.
    pub fn sdk_name(&self) -> &'static str {
        match self {
            RunError::MaxTurnsExceeded { .. } => "MaxTurnsExceeded",
            RunError::InputGuardrailTripwire { .. } => "InputGuardrailTripwireTriggered",
            RunError::OutputGuardrailTripwire { .. } => "OutputGuardrailTripwireTriggered",
            RunError::Model(ModelError::Behavior(_) | ModelError::Truncated) => {
                "ModelBehaviorError"
            }
            RunError::Model(_) => "ModelError",
            RunError::ModelRefusal { .. } => "ModelRefusalError",
            RunError::Cancelled => "CancelledError",
            RunError::User(_) => "UserError",
        }
    }
}

/// A run in flight.
pub struct RunHandle {
    /// Everything that happens, in order; ends when the run does.
    pub events: UnboundedReceiver<StreamEvent>,
    pub control: RunControl,
    /// The run's outcome.
    pub result: JoinHandle<Result<RunResult, RunError>>,
}

/// Start a run from one user message on the current tokio runtime and return
/// at once: [`run_streamed_items`] with `[InputItem::User(input)]`.
///
/// # Panics
/// Panics when called outside a tokio runtime (it spawns the run).
pub fn run_streamed(agent: Arc<Agent>, input: String, config: RunConfig) -> RunHandle {
    run_streamed_items(agent, vec![InputItem::User(input)], config)
}

/// Start a run from a list of input items on the current tokio runtime and
/// return at once. The items reach the first model call in order and
/// unchanged; an empty list is refused (`RunError::User`) before the run
/// starts anything.
///
/// # Panics
/// Panics when called outside a tokio runtime (it spawns the run).
pub fn run_streamed_items(
    agent: Arc<Agent>,
    input: Vec<InputItem>,
    config: RunConfig,
) -> RunHandle {
    let (sender, receiver) = unbounded();
    let panics = Arc::new(AtomicU64::new(0));
    let control = RunControl::new(panics.clone());
    if input.is_empty() {
        // Nothing to send: no trace, no span and no event, so the stream ends
        // at once and the outcome says why. Nothing will take a steer either.
        drop(sender);
        control.close_steering();
        return RunHandle {
            events: receiver,
            control,
            result: tokio::spawn(async { Err(RunError::User(NO_INPUT.to_owned())) }),
        };
    }
    let tracer = Tracer::new(
        config.trace_id.clone(),
        config.workflow_name.clone(),
        config.group_id.clone(),
        config.trace_metadata.clone(),
        config.processors.clone(),
        panics,
    );
    let run = Run {
        model: config.model.clone(),
        model_settings: config.model_settings.clone(),
        max_turns: config.max_turns,
        include_sensitive_data: config.include_sensitive_data,
        use_task_and_turn_spans: config.include_task_and_turn_spans,
        workflow_name: config.workflow_name.clone(),
        context: config.context.clone(),
        events: sender,
        control: control.clone(),
        tracer,
        input,
        session: config.session.clone(),
        session_items: Vec::new(),
        session_write_failed: false,
        saved_through: 0,
        approvals: config.approvals.clone(),
        #[cfg(test)]
        mutant: config.mutant,
        starting_agent: agent.clone(),
        current_agent: agent,
        turn: 0,
        usage: None,
        history: Vec::new(),
        new_items: Vec::new(),
        agents_that_used_tools: HashSet::new(),
        task_span: None,
        agent_span: None,
        turn_scope: None,
    };
    RunHandle {
        events: receiver,
        control,
        result: tokio::spawn(run_task(run)),
    }
}

struct TurnScope {
    span: Span,
    usage_at_start: Usage,
}

struct Run {
    model: Arc<dyn Model>,
    model_settings: Option<ModelSettings>,
    max_turns: u32,
    include_sensitive_data: bool,
    use_task_and_turn_spans: bool,
    workflow_name: String,
    context: RunContext,
    events: UnboundedSender<StreamEvent>,
    control: RunControl,
    tracer: Arc<Tracer>,
    /// The items the run was started from; never empty.
    input: Vec<InputItem>,
    session: Option<Arc<dyn Session>>,
    /// What the session held when the run started; every model call is given
    /// these first.
    session_items: Vec<InputItem>,
    /// Whether `SessionWriteFailed` has been sent.
    session_write_failed: bool,
    /// How much of `history` the session has been given (a turn whose calls
    /// wait for approval saves part of itself before the rest).
    saved_through: usize,
    approvals: Option<Arc<dyn ApprovalPort>>,
    #[cfg(test)]
    mutant: Option<Mutant>,
    starting_agent: Arc<Agent>,
    current_agent: Arc<Agent>,
    turn: u32,
    usage: Option<Usage>,
    /// The conversation after the input, in model input form.
    history: Vec<InputItem>,
    new_items: Vec<RunItem>,
    agents_that_used_tools: HashSet<usize>,
    task_span: Option<Span>,
    agent_span: Option<Span>,
    turn_scope: Option<TurnScope>,
}

impl Drop for Run {
    fn drop(&mut self) {
        // However the run ended, it takes no more steers; what it accepted and
        // never sent stays with the control.
        self.control.close_steering();
        self.close_spans_and_trace();
    }
}

enum Step {
    RunAgain,
    Handoff(Arc<Agent>),
    Final(String),
}

/// One response's calls and text, sorted out.
struct Processed {
    items: Vec<RunItem>,
    functions: Vec<(FunctionTool, ToolCallItem)>,
    handoffs: Vec<(Handoff, ToolCallItem)>,
    last_text: Option<String>,
    /// What the model declined to say, if it did.
    refusal: Option<String>,
    /// Positions in the response's `output` of calls left out because an exact
    /// copy of each came earlier in the same response.
    skipped_outputs: HashSet<usize>,
}

/// The SDK's refusal of a call with no id (`tool_planning.py:406-410`).
const EMPTY_CALL_ID: &str = "Tool invocations require a non-empty string call ID before execution.";
/// The SDK's refusal of one id for two different calls in one response
/// (`tool_planning.py:436-441`).
const REUSED_CALL_ID: &str = "Model reused a tool call ID for a different invocation in one response. Use a unique call ID for each tool invocation.";

/// A call's arguments as the SDK compares them (`_tool_invocation.py:
/// _normalize_arguments`): parsed JSON, so key order and spacing do not matter,
/// or the text itself when it is not JSON.
#[derive(PartialEq)]
enum Arguments {
    Json(Value),
    Text(String),
}

/// What makes two calls with one id "the same invocation" (the SDK's
/// fingerprint: its kind, the tool asked for, and the arguments).
#[derive(PartialEq)]
struct Invocation {
    handoff: bool,
    name: String,
    arguments: Arguments,
}

impl Invocation {
    fn of(call: &ToolCallItem, handoff: bool) -> Self {
        Self {
            handoff,
            name: call.name.clone(),
            arguments: serde_json::from_str(&call.arguments)
                .map_or_else(|_| Arguments::Text(call.arguments.clone()), Arguments::Json),
        }
    }
}

/// A function or handoff call of a response, with where it came from.
struct Planned {
    output_index: usize,
    item_index: usize,
    call: ToolCallItem,
}

async fn run_task(mut run: Run) -> Result<RunResult, RunError> {
    let control = run.control.clone();
    if run.session.is_some() {
        // Read the conversation and save the input before anything starts; a
        // session that cannot be read ends the run here, with no trace.
        let opened = tokio::select! {
            biased;
            () = control.inner.immediate.cancelled() => Err(RunError::Cancelled),
            opened = run.open_session() => opened,
        };
        opened?;
    }
    run.tracer.start_trace();
    if run.use_task_and_turn_spans {
        run.task_span = Some(Span::start(
            &run.tracer,
            None,
            task_data(&run.workflow_name),
        ));
    }
    let outcome = {
        let drive = drive(&mut run);
        tokio::select! {
            biased;
            () = control.inner.immediate.cancelled() => Err(RunError::Cancelled),
            result = drive => result,
        }
    };
    run.finish(&outcome);
    outcome
}

async fn drive(run: &mut Run) -> Result<RunResult, RunError> {
    run.emit(StreamEvent::AgentUpdated {
        agent: run.current_agent.name.clone(),
    });
    loop {
        if run.control.after_turn_requested() {
            return Err(RunError::Cancelled);
        }
        if run.agent_span.is_none() {
            let parent = run.task_span.as_ref().map(|span| span.id().to_owned());
            run.agent_span = Some(Span::start(
                &run.tracer,
                parent.as_deref(),
                agent_data(&run.current_agent.name),
            ));
        }
        run.turn += 1;
        if run.turn > run.max_turns {
            let max_turns = run.max_turns;
            run.set_agent_error(span_error(
                "Max turns exceeded",
                json!({"max_turns": max_turns}),
            ));
            return Err(RunError::MaxTurnsExceeded { max_turns });
        }
        if run.turn == 1 {
            run.run_input_guardrails().await?;
        }
        let turn_start = run.history.len();
        if !run.steers_after_the_model_call() {
            run.take_steers();
        }
        run.start_turn_span();
        let step = run.single_turn(turn_start).await;
        run.finish_turn_span();
        match step? {
            Step::RunAgain => run.save_turn(turn_start).await,
            Step::Handoff(next) => {
                run.save_turn(turn_start).await;
                if let Some(mut span) = run.agent_span.take() {
                    span.finish();
                }
                run.current_agent = next;
                run.emit(StreamEvent::AgentUpdated {
                    agent: run.current_agent.name.clone(),
                });
            }
            Step::Final(text) => {
                // No model call follows: steering closes now, and whatever was
                // steered during this call stays with the control until the
                // result is built. A tripped or panicking output guardrail, or
                // a cancel while the guardrails or the final save are awaited,
                // therefore leaves those steers for `take_unsent_steers`
                // (spec 22.6 A4a).
                let held = run.close_steering_at_final();
                run.run_output_guardrails(&text).await?;
                run.save_turn(turn_start).await;
                let unsent_steers = run.unsent_steers_at_final(held);
                return Ok(RunResult {
                    final_output: text,
                    last_agent: run.current_agent.name.clone(),
                    usage: run.usage,
                    turns: run.turn,
                    new_items: std::mem::take(&mut run.new_items),
                    unsent_steers,
                });
            }
        }
    }
}

impl Run {
    /// Add the steers waiting now to the history, in order, and announce each.
    fn take_steers(&mut self) {
        for text in self.control.take_steers() {
            self.history.push(InputItem::User(text.clone()));
            self.emit(StreamEvent::Steered {
                agent: self.current_agent.name.clone(),
                text,
            });
        }
    }

    /// Close steering at the final output. The steers that were waiting stay
    /// with the control (as unsent) until [`Self::unsent_steers_at_final`];
    /// the run itself holds none of them, so an ending that drops it loses
    /// none. Returns what the run holds instead: nothing, except under AF7's
    /// mutant `DrainBeforeOutputGuardrails` (the code before A4a).
    fn close_steering_at_final(&mut self) -> Vec<String> {
        #[cfg(test)]
        if self.mutant == Some(Mutant::NoDrainAtFinal) {
            return Vec::new();
        }
        self.control.close_steering();
        #[cfg(test)]
        if self.mutant == Some(Mutant::DrainBeforeOutputGuardrails) {
            return self.control.take_unsent_steers();
        }
        Vec::new()
    }

    /// The final output's unsent steers, taken from the control only once the
    /// result is certain (after the output guardrails and the final save).
    fn unsent_steers_at_final(&mut self, mut held: Vec<String>) -> Vec<String> {
        held.extend(self.control.take_unsent_steers());
        held
    }

    /// AF5's mutant only: steers are taken after the model call, not before.
    fn steers_after_the_model_call(&self) -> bool {
        #[cfg(test)]
        if self.mutant == Some(Mutant::SteerAfterModelCall) {
            return true;
        }
        false
    }

    /// Read the session's conversation, then save the run's input to it.
    async fn open_session(&mut self) -> Result<(), RunError> {
        let Some(session) = self.session.clone() else {
            return Ok(());
        };
        let read = match std::panic::catch_unwind(AssertUnwindSafe(|| session.get_items(None))) {
            Ok(future) => AssertUnwindSafe(future)
                .catch_unwind()
                .await
                .unwrap_or_else(|_| Err(session_panicked())),
            Err(_) => Err(session_panicked()),
        };
        match read {
            Ok(items) => self.session_items = items,
            Err(error) => {
                return Err(RunError::User(format!(
                    "the session could not be read: {error}"
                )));
            }
        }
        let input = self.input.clone();
        self.save_to_session(input).await;
        Ok(())
    }

    /// Save what the turn that began at `turn_start` added to the history.
    async fn save_turn(&mut self, turn_start: usize) {
        let from = turn_start.max(self.saved_through);
        if self.session.is_none() || self.history.len() <= from {
            return;
        }
        let items = self.history[from..].to_vec();
        self.saved_through = self.history.len();
        self.save_to_session(items).await;
    }

    /// `add_items`, where a failure is announced once and the run goes on.
    async fn save_to_session(&mut self, items: Vec<InputItem>) {
        let Some(session) = self.session.clone() else {
            return;
        };
        let saved = match std::panic::catch_unwind(AssertUnwindSafe(|| session.add_items(items))) {
            Ok(future) => AssertUnwindSafe(future)
                .catch_unwind()
                .await
                .unwrap_or_else(|_| Err(session_panicked())),
            Err(_) => Err(session_panicked()),
        };
        if let Err(error) = saved
            && !self.session_write_failed
        {
            self.session_write_failed = true;
            self.emit(StreamEvent::SessionWriteFailed {
                message: error.to_string(),
            });
        }
    }

    /// What an input guardrail checks: the text of the last user item of the
    /// run's input, or the empty string when there is none.
    fn last_user_text(&self) -> String {
        self.input
            .iter()
            .rev()
            .find_map(|item| match item {
                InputItem::User(text) => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    fn emit(&self, event: StreamEvent) {
        // A consumer that dropped the receiver has stopped listening; the run
        // is unaffected.
        let _ = self.events.unbounded_send(event);
    }

    fn emit_item(&mut self, item: RunItem) {
        self.emit(StreamEvent::RunItem {
            name: item.event_name(),
            item: item.clone(),
        });
        self.new_items.push(item);
    }

    /// The span children of the current step hang under: the turn span, or the
    /// agent span when turn spans are off (the SDK's "current span").
    fn current_span_id(&self) -> Option<String> {
        self.turn_scope
            .as_ref()
            .map(|turn| turn.span.id().to_owned())
            .or_else(|| self.agent_span.as_ref().map(|span| span.id().to_owned()))
    }

    fn set_agent_error(&mut self, error: lattice_protocol::SpanError) {
        if let Some(span) = self.agent_span.as_mut() {
            span.set_error(error);
        }
    }

    /// `attach_error_to_current_span`.
    fn set_current_span_error(&mut self, error: lattice_protocol::SpanError) {
        if let Some(turn) = self.turn_scope.as_mut() {
            turn.span.set_error(error);
        } else {
            self.set_agent_error(error);
        }
    }

    fn start_turn_span(&mut self) {
        if !self.use_task_and_turn_spans {
            return;
        }
        let parent = self.agent_span.as_ref().map(|span| span.id().to_owned());
        let span = Span::start(
            &self.tracer,
            parent.as_deref(),
            turn_data(self.turn, &self.current_agent.name),
        );
        self.turn_scope = Some(TurnScope {
            span,
            usage_at_start: self.usage.unwrap_or_default(),
        });
    }

    fn finish_turn_span(&mut self) {
        if let Some(mut turn) = self.turn_scope.take() {
            let added = usage::delta(turn.usage_at_start, self.usage.unwrap_or_default());
            if !usage::is_zero(added) {
                turn.span.data_mut()["data"]["usage"] = usage::turn_span_usage(added);
            }
            turn.span.finish();
        }
    }

    /// The end of a run, on every path: the generic agent error, then the
    /// spans innermost first, then the trace.
    fn finish(&mut self, outcome: &Result<RunResult, RunError>) {
        if let Err(error) = outcome {
            self.attach_generic_agent_error(error);
        }
        self.close_spans_and_trace();
    }

    /// Close whatever is open, innermost first, and end the trace. Idempotent:
    /// `Drop` calls it too, so a run task that unwinds (a bug, not a caller's
    /// mistake: tools, guardrails, models and processors are all shielded) still
    /// ends its spans and its trace.
    fn close_spans_and_trace(&mut self) {
        self.finish_turn_span();
        if let Some(mut span) = self.agent_span.take() {
            span.finish();
        }
        if let Some(mut span) = self.task_span.take() {
            let total = self.usage.unwrap_or_default();
            if !usage::is_zero(total) {
                span.data_mut()["data"]["usage"] = usage::task_span_usage(total);
            }
            span.finish();
        }
        self.tracer.end_trace();
    }

    /// `attach_generic_agent_error`: an unexpected failure marks the agent span,
    /// unless a more specific error is already there. Model misbehaviour,
    /// tripwires, turn limits and cancellation carry their own diagnosis.
    fn attach_generic_agent_error(&mut self, error: &RunError) {
        // The SDK excludes `ModelBehaviorError` and the tripwires
        // (`error_handlers.py: _is_generic_agent_error`); a refusal is generic.
        let generic = match error {
            RunError::User(_) | RunError::ModelRefusal { .. } => true,
            RunError::Model(ModelError::Behavior(_) | ModelError::Truncated) => false,
            RunError::Model(_) => true,
            _ => false,
        };
        if !generic {
            return;
        }
        let detail = if self.include_sensitive_data {
            error.to_string()
        } else {
            REDACTED_TRACE_ERROR.to_owned()
        };
        if let Some(span) = self.agent_span.as_mut()
            && !span.has_error()
        {
            span.set_error(span_error("Error in agent run", json!({"error": detail})));
        }
    }

    async fn run_input_guardrails(&mut self) -> Result<(), RunError> {
        let agent = self.starting_agent.clone();
        for guardrail in &agent.input_guardrails {
            let parent = self.agent_span.as_ref().map(|span| span.id().to_owned());
            let mut span = Span::start(
                &self.tracer,
                parent.as_deref(),
                guardrail_data(&guardrail.name),
            );
            let context = GuardrailContext {
                run_context: self.context.clone(),
                agent: agent.name.clone(),
            };
            let check = std::panic::catch_unwind(AssertUnwindSafe(|| {
                (guardrail.check)(context, self.last_user_text())
            }));
            let output = match check {
                Ok(future) => AssertUnwindSafe(future).catch_unwind().await.ok(),
                Err(_) => None,
            };
            let Some(output) = output else {
                span.finish();
                return Err(RunError::User(format!(
                    "input guardrail {} failed",
                    guardrail.name
                )));
            };
            span.data_mut()["triggered"] = json!(output.tripwire_triggered);
            span.finish();
            if output.tripwire_triggered {
                self.set_agent_error(span_error(
                    "Guardrail tripwire triggered",
                    json!({"guardrail": guardrail.name, "type": "input_guardrail"}),
                ));
                return Err(RunError::InputGuardrailTripwire {
                    guardrail: guardrail.name.clone(),
                    output_info: output.output_info,
                });
            }
        }
        Ok(())
    }

    async fn run_output_guardrails(&mut self, text: &str) -> Result<(), RunError> {
        let agent = self.current_agent.clone();
        if agent.output_guardrails.is_empty() {
            return Ok(());
        }
        let parent = self.agent_span.as_ref().map(|span| span.id().to_owned());
        let mut running = FuturesUnordered::new();
        for guardrail in &agent.output_guardrails {
            let tracer = self.tracer.clone();
            let parent = parent.clone();
            let name = guardrail.name.clone();
            let check = guardrail.check.clone();
            let context = GuardrailContext {
                run_context: self.context.clone(),
                agent: agent.name.clone(),
            };
            let text = text.to_owned();
            running.push(async move {
                let mut span = Span::start(&tracer, parent.as_deref(), guardrail_data(&name));
                let output =
                    match std::panic::catch_unwind(AssertUnwindSafe(|| check(context, text))) {
                        Ok(future) => AssertUnwindSafe(future).catch_unwind().await.ok(),
                        Err(_) => None,
                    };
                if let Some(output) = &output {
                    span.data_mut()["triggered"] = json!(output.tripwire_triggered);
                }
                span.finish();
                (name, output)
            });
        }
        while let Some((name, output)) = running.next().await {
            let Some(GuardrailOutput {
                tripwire_triggered,
                output_info,
            }) = output
            else {
                return Err(RunError::User(format!("output guardrail {name} failed")));
            };
            if tripwire_triggered {
                // The guardrails still running are dropped; their spans end.
                drop(running);
                self.set_agent_error(span_error(
                    "Guardrail tripwire triggered",
                    json!({"guardrail": name}),
                ));
                return Err(RunError::OutputGuardrailTripwire {
                    guardrail: name,
                    output_info,
                });
            }
        }
        Ok(())
    }

    /// One turn: the model call, then what the response asks for.
    async fn single_turn(&mut self, turn_start: usize) -> Result<Step, RunError> {
        let agent = self.current_agent.clone();
        let (tools, handoffs) = resolve_tool_name_collisions(&agent);

        if let Some(span) = self.agent_span.as_mut() {
            let data = span.data_mut();
            data["handoffs"] = json!(handoffs.iter().map(|h| h.agent_name()).collect::<Vec<_>>());
            data["tools"] = json!(tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>());
        }

        let mut settings =
            ModelSettings::resolve(&agent.model_settings, self.model_settings.as_ref());
        if self
            .agents_that_used_tools
            .contains(&Agent::identity(&agent))
        {
            // `maybe_reset_tool_choice`: a forced tool choice applies to the
            // agent's first turn only, or it would call the tool forever.
            settings.tool_choice = None;
        }
        let mut specs: Vec<ToolSpec> = tools
            .iter()
            .map(|tool| ToolSpec {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
                strict: tool.strict,
            })
            .collect();
        specs.extend(handoffs.iter().map(|handoff| ToolSpec {
            name: handoff.tool_name.clone(),
            description: handoff.tool_description.clone(),
            parameters: Handoff::input_json_schema(),
            strict: true,
        }));
        let mut input =
            Vec::with_capacity(self.session_items.len() + self.input.len() + self.history.len());
        input.extend(self.session_items.iter().cloned());
        input.extend(self.input.iter().cloned());
        input.extend(self.history.iter().cloned());
        let request = ModelRequest {
            system: agent.instructions.clone(),
            input,
            tools: specs,
            settings,
        };

        let response = self.call_model(&agent, request).await?;
        if self.steers_after_the_model_call() {
            self.take_steers();
        }
        if let Some(reported) = response.usage {
            self.usage = Some(usage::add(self.usage.unwrap_or_default(), reported));
        }

        let mut processed = self.process_model_response(&agent, &tools, &handoffs, &response)?;
        // A refusal with nothing else to do ends the run, after the turn's items
        // are announced, as the SDK ends it (`turn_resolution.py:952-988`).
        let refused = if processed.functions.is_empty() && processed.handoffs.is_empty() {
            processed.refusal.take()
        } else {
            None
        };
        if !processed.functions.is_empty() || !processed.handoffs.is_empty() {
            self.agents_that_used_tools.insert(Agent::identity(&agent));
        }
        for (position, item) in response.output.iter().enumerate() {
            match item {
                OutputItem::Message { text } => {
                    self.history.push(InputItem::Assistant {
                        text: Some(text.clone()),
                        tool_calls: Vec::new(),
                    });
                }
                OutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                } if !processed.skipped_outputs.contains(&position) => {
                    self.history.push(InputItem::Assistant {
                        text: None,
                        tool_calls: vec![ToolCallItem {
                            call_id: call_id.clone(),
                            name: name.clone(),
                            arguments: arguments.clone(),
                        }],
                    });
                }
                OutputItem::FunctionCall { .. }
                | OutputItem::Reasoning { .. }
                | OutputItem::Refusal { .. } => {}
            }
        }
        for item in processed.items.iter().cloned() {
            self.emit_item(item);
        }
        if let Some(refusal) = refused {
            return Err(RunError::ModelRefusal { refusal });
        }

        // Function calls run concurrently, outputs in call order (when some
        // need approval, theirs come after the others').
        let parent = self.current_span_id();
        let gate = Gate {
            port: self.approvals.clone(),
            events: self.events.clone(),
            #[cfg(test)]
            mutant: self.mutant,
            #[cfg(test)]
            approved_by_text: Arc::new(if self.mutant == Some(Mutant::DecisionFromText) {
                af_tests::approved_by_text(&self.history)
            } else {
                HashSet::new()
            }),
        };
        // Every call's span starts here, in call order. The calls that need
        // approval ask in place while the others run.
        let mut free = Vec::new();
        let mut gated = Vec::new();
        for (tool, call) in &processed.functions {
            let context = ToolContext {
                run_context: self.context.clone(),
                agent: agent.name.clone(),
                call_id: call.call_id.clone(),
            };
            let to_approve = arguments_to_approve(tool, &context, &call.arguments);
            let asks = to_approve.is_some();
            let run = run_function(
                self.tracer.clone(),
                parent.clone(),
                tool.clone(),
                call.clone(),
                context,
                self.include_sensitive_data,
                gate.clone(),
                to_approve,
            );
            if asks {
                gated.push((call.call_id.clone(), run));
            } else {
                free.push((call.call_id.clone(), run));
            }
        }
        #[cfg(test)]
        if self.mutant == Some(Mutant::OutputsInCallOrder) {
            self.outputs_in_call_order(&agent, &processed, free, gated)
                .await;
            return self.step_after_tools(&agent, processed);
        }
        let (free_ids, free_runs): (Vec<String>, Vec<_>) = free.into_iter().unzip();
        let (gated_ids, gated_runs): (Vec<String>, Vec<_>) = gated.into_iter().unzip();
        let free_all = futures::future::join_all(free_runs);
        let gated_all = futures::future::join_all(gated_runs);
        let (free_outputs, gated_left) = if gated_ids.is_empty() {
            (free_all.await, None)
        } else {
            match futures::future::select(free_all, gated_all).await {
                Either::Left((outputs, waiting)) => (outputs, Some(Either::Left(waiting))),
                Either::Right((decided, free_all)) => {
                    (free_all.await, Some(Either::Right(decided)))
                }
            }
        };
        self.append_outputs(&agent, free_ids, free_outputs);
        if let Some(gated_left) = gated_left {
            // The SDK's interruption: what the turn holds so far is saved
            // before the approvals are awaited to their end.
            self.save_turn(turn_start).await;
            let decided = match gated_left {
                Either::Left(waiting) => waiting.await,
                Either::Right(decided) => decided,
            };
            self.append_outputs(&agent, gated_ids, decided);
        }
        self.step_after_tools(&agent, processed)
    }

    /// Append each call's output to the history and announce it, in the order
    /// given.
    fn append_outputs(&mut self, agent: &Arc<Agent>, call_ids: Vec<String>, outputs: Vec<String>) {
        for (call_id, output) in call_ids.into_iter().zip(outputs) {
            self.history.push(InputItem::ToolResult {
                call_id: call_id.clone(),
                output: output.clone(),
            });
            self.emit_item(RunItem::ToolOutput {
                agent: agent.name.clone(),
                call_id,
                output,
            });
        }
    }

    /// A3a's mutant only: every call is awaited, then every output appended in
    /// call order, with no save before the approvals end (the code before A3a).
    #[cfg(test)]
    async fn outputs_in_call_order(
        &mut self,
        agent: &Arc<Agent>,
        processed: &Processed,
        free: Vec<(String, BoxFuture<'static, String>)>,
        gated: Vec<(String, BoxFuture<'static, String>)>,
    ) {
        let mut runs: HashMap<String, BoxFuture<'static, String>> =
            free.into_iter().chain(gated).collect();
        let order: Vec<String> = processed
            .functions
            .iter()
            .map(|(_, call)| call.call_id.clone())
            .collect();
        let ordered: Vec<_> = order.iter().filter_map(|id| runs.remove(id)).collect();
        let outputs = futures::future::join_all(ordered).await;
        self.append_outputs(agent, order, outputs);
    }

    /// What follows a turn's tool calls: a handoff, the final output, or
    /// another turn.
    fn step_after_tools(
        &mut self,
        agent: &Arc<Agent>,
        processed: Processed,
    ) -> Result<Step, RunError> {
        if let Some((first, first_call)) = processed.handoffs.first() {
            return Ok(Step::Handoff(self.execute_handoffs(
                agent,
                first,
                first_call,
                &processed.handoffs,
            )));
        }
        if processed.functions.is_empty() {
            return Ok(Step::Final(processed.last_text.unwrap_or_default()));
        }
        Ok(Step::RunAgain)
    }

    /// The model call, its generation span, and its live text.
    async fn call_model(
        &mut self,
        agent: &Arc<Agent>,
        request: ModelRequest,
    ) -> Result<ModelResponse, RunError> {
        let style = self.model.generation_trace();
        let parent = self.current_span_id();
        let mut generation = Span::start(&self.tracer, parent.as_deref(), generation_data());
        if style == GenerationTrace::Full {
            let mut config = request.settings.to_traceable_dict();
            if let (Some(target), Value::Object(extra)) =
                (config.as_object_mut(), self.model.config_for_trace())
            {
                target.extend(extra);
            }
            let data = generation.data_mut();
            data["model"] = json!(self.model.name());
            data["model_config"] = config;
            if self.include_sensitive_data {
                data["input"] =
                    Value::Array(chat_messages_for_trace(&request.system, &request.input));
            }
        }

        // A model that panics (while starting the call or while producing a
        // chunk) fails this call like any other model failure; it does not take
        // the run task, and with it the trace, down.
        let model = self.model.clone();
        let mut stream = match std::panic::catch_unwind(AssertUnwindSafe(|| model.stream(request)))
        {
            Ok(stream) => stream,
            Err(_) => futures::stream::once(async { Err(model_panicked()) }).boxed(),
        };
        let mut response = None;
        loop {
            let item = match AssertUnwindSafe(stream.next()).catch_unwind().await {
                Ok(item) => item,
                Err(_) => Some(Err(model_panicked())),
            };
            let Some(item) = item else { break };
            match item {
                Ok(ModelEvent::TextDelta(delta)) => {
                    self.emit(StreamEvent::RawTextDelta {
                        agent: agent.name.clone(),
                        delta,
                    });
                }
                Ok(ModelEvent::ReasoningDelta(_)) => {}
                Ok(ModelEvent::Done(done)) => {
                    response = Some(done);
                    break;
                }
                Err(error) => {
                    generation.set_error(self.generation_error(style, &error));
                    generation.finish();
                    return Err(RunError::Model(error));
                }
            }
        }
        drop(stream);

        let Some(response) = response else {
            generation.finish();
            return Err(RunError::Model(ModelError::Behavior(
                "Model did not produce a final response!".into(),
            )));
        };
        if style == GenerationTrace::Full {
            let data = generation.data_mut();
            if self.include_sensitive_data {
                data["output"] = json!([response_dump(self.model.name(), &response)]);
            }
            if let Some(reported) = response.usage {
                data["usage"] = usage::generation_span_usage(reported);
            }
        }
        generation.finish();
        Ok(response)
    }

    /// What a failed model call records on its generation span: the scripted
    /// model's shape (`Error` with the exception's name and text) or the real
    /// model's (`Error streaming response`).
    fn generation_error(
        &self,
        style: GenerationTrace,
        error: &ModelError,
    ) -> lattice_protocol::SpanError {
        let text = if self.include_sensitive_data {
            error.to_string()
        } else {
            REDACTED_TRACE_ERROR.to_owned()
        };
        match style {
            GenerationTrace::Full => span_error("Error streaming response", json!({"error": text})),
            GenerationTrace::Bare => {
                span_error("Error", json!({"name": error.kind(), "message": text}))
            }
        }
    }

    /// `process_model_response`: sort a response into items, function calls and
    /// handoff calls, in the order the model produced them.
    fn process_model_response(
        &mut self,
        agent: &Arc<Agent>,
        tools: &[&FunctionTool],
        handoffs: &[&Handoff],
        response: &ModelResponse,
    ) -> Result<Processed, RunError> {
        let mut processed = Processed {
            items: Vec::new(),
            functions: Vec::new(),
            handoffs: Vec::new(),
            last_text: None,
            refusal: None,
            skipped_outputs: HashSet::new(),
        };
        // Where each call came from: its place in `output` and in `items`, in
        // the order `functions` and `handoffs` hold them.
        let mut function_places: Vec<Planned> = Vec::new();
        let mut handoff_places: Vec<Planned> = Vec::new();
        for (output_index, output) in response.output.iter().enumerate() {
            match output {
                OutputItem::Message { text } => {
                    processed.last_text = Some(text.clone());
                    processed.items.push(RunItem::MessageOutput {
                        agent: agent.name.clone(),
                        text: text.clone(),
                    });
                }
                OutputItem::Reasoning { text } => {
                    processed.items.push(RunItem::Reasoning {
                        agent: agent.name.clone(),
                        text: text.clone(),
                    });
                }
                OutputItem::Refusal { text } => {
                    processed
                        .refusal
                        .get_or_insert_with(String::new)
                        .push_str(text);
                    // The SDK's message item carries the refusal as a part of its
                    // content, and announces the message (whose text is empty when
                    // the refusal is all it holds) whether the run then goes on or not.
                    if !processed
                        .items
                        .iter()
                        .any(|item| matches!(item, RunItem::MessageOutput { .. }))
                    {
                        processed.items.push(RunItem::MessageOutput {
                            agent: agent.name.clone(),
                            text: String::new(),
                        });
                    }
                }
                OutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                } => {
                    let call = ToolCallItem {
                        call_id: call_id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    };
                    if let Some(handoff) =
                        handoffs.iter().find(|handoff| handoff.tool_name == *name)
                    {
                        processed.items.push(RunItem::HandoffCall {
                            agent: agent.name.clone(),
                            call_id: call_id.clone(),
                            tool_name: name.clone(),
                        });
                        handoff_places.push(Planned {
                            output_index,
                            item_index: processed.items.len() - 1,
                            call: call.clone(),
                        });
                        processed.handoffs.push(((*handoff).clone(), call));
                    } else if let Some(tool) = tools.iter().rev().find(|tool| tool.name == *name) {
                        processed.items.push(RunItem::ToolCall {
                            agent: agent.name.clone(),
                            call_id: call_id.clone(),
                            name: name.clone(),
                            arguments: arguments.clone(),
                        });
                        function_places.push(Planned {
                            output_index,
                            item_index: processed.items.len() - 1,
                            call: call.clone(),
                        });
                        processed.functions.push(((*tool).clone(), call));
                    } else {
                        self.set_current_span_error(span_error(
                            "Tool not found",
                            json!({"tool_name": name}),
                        ));
                        return Err(RunError::Model(ModelError::Behavior(format!(
                            "Tool {name} not found in agent {}",
                            agent.name
                        ))));
                    }
                }
            }
        }
        Self::plan_invocations(&mut processed, &function_places, &handoff_places)?;
        Ok(processed)
    }

    /// `_dedupe_processed_response_invocations`, for the calls of one response,
    /// before any tool starts: every call needs an id; a call that repeats an
    /// earlier one exactly (same id, same tool, same arguments) is left out, as if
    /// it had been made once; one id for two different calls is the model's
    /// error. Functions are checked before handoffs, as the SDK checks them.
    ///
    /// Not ported: the same checks against calls completed in EARLIER turns
    /// (`tool_planning.py:380-411`). A model that reuses an id across turns is
    /// answered each time here, where the SDK would skip an identical repeat or
    /// refuse a different one.
    fn plan_invocations(
        processed: &mut Processed,
        function_places: &[Planned],
        handoff_places: &[Planned],
    ) -> Result<(), RunError> {
        let behavior = |message: &str| RunError::Model(ModelError::Behavior(message.to_owned()));
        let mut first: HashMap<String, Invocation> = HashMap::new();
        let mut skipped_items: HashSet<usize> = HashSet::new();
        let ordered = function_places
            .iter()
            .map(|place| (place, false))
            .chain(handoff_places.iter().map(|place| (place, true)));
        for (place, is_handoff) in ordered {
            if place.call.call_id.is_empty() {
                return Err(behavior(EMPTY_CALL_ID));
            }
            let invocation = Invocation::of(&place.call, is_handoff);
            match first.get(&place.call.call_id) {
                Some(earlier) if *earlier == invocation => {
                    skipped_items.insert(place.item_index);
                    processed.skipped_outputs.insert(place.output_index);
                }
                Some(_) => return Err(behavior(REUSED_CALL_ID)),
                None => {
                    first.insert(place.call.call_id.clone(), invocation);
                }
            }
        }
        if skipped_items.is_empty() {
            return Ok(());
        }
        let skipped_outputs = &processed.skipped_outputs;
        let mut keep_function = function_places
            .iter()
            .map(|place| !skipped_outputs.contains(&place.output_index));
        processed
            .functions
            .retain(|_| keep_function.next().unwrap_or(true));
        let mut keep_handoff = handoff_places
            .iter()
            .map(|place| !skipped_outputs.contains(&place.output_index));
        processed
            .handoffs
            .retain(|_| keep_handoff.next().unwrap_or(true));
        let mut position = 0;
        processed.items.retain(|_| {
            position += 1;
            !skipped_items.contains(&(position - 1))
        });
        Ok(())
    }

    /// `execute_handoffs`: run the first handoff, answer the rest.
    fn execute_handoffs(
        &mut self,
        agent: &Arc<Agent>,
        handoff: &Handoff,
        call: &ToolCallItem,
        all: &[(Handoff, ToolCallItem)],
    ) -> Arc<Agent> {
        let multiple = all.len() > 1;
        for (_, ignored) in all.iter().skip(1) {
            self.history.push(InputItem::ToolResult {
                call_id: ignored.call_id.clone(),
                output: MULTIPLE_HANDOFFS_OUTPUT.to_owned(),
            });
            self.emit_item(RunItem::ToolOutput {
                agent: agent.name.clone(),
                call_id: ignored.call_id.clone(),
                output: MULTIPLE_HANDOFFS_OUTPUT.to_owned(),
            });
        }

        let parent = self.current_span_id();
        let mut span = Span::start(&self.tracer, parent.as_deref(), handoff_data(&agent.name));
        let target = handoff.agent.clone();
        span.data_mut()["to_agent"] = json!(target.name);
        if multiple {
            let requested: Vec<&str> = all.iter().map(|(h, _)| h.agent_name()).collect();
            span.set_error(span_error(
                "Multiple handoffs requested",
                json!({"requested_agents": requested}),
            ));
        }
        let output = handoff.transfer_message(&target);
        self.history.push(InputItem::ToolResult {
            call_id: call.call_id.clone(),
            output: output.clone(),
        });
        self.emit_item(RunItem::HandoffOutput {
            agent: agent.name.clone(),
            call_id: call.call_id.clone(),
            source_agent: agent.name.clone(),
            target_agent: target.name.clone(),
            output,
        });
        span.finish();
        target
    }
}

/// What a panic inside a session is reported as. The panic's own message is
/// not repeated: it may hold conversation text.
fn session_panicked() -> SessionError {
    SessionError::new("the session panicked")
}

/// What a panic inside a model is reported as. The panic's own message is not
/// repeated: it may hold data the model was given.
fn model_panicked() -> ModelError {
    ModelError::Failed("the model panicked".into())
}

/// What a call that needs approval uses: where to ask, and where to say so.
#[derive(Clone)]
struct Gate {
    port: Option<Arc<dyn ApprovalPort>>,
    events: UnboundedSender<StreamEvent>,
    #[cfg(test)]
    mutant: Option<Mutant>,
    /// AF4's mutant only: calls "approved" by earlier tool output text.
    #[cfg(test)]
    approved_by_text: Arc<HashSet<String>>,
}

/// One function tool call: its span, its approval, its handler, its outcome.
/// The span starts when this is called (so a response's spans start in call
/// order); the rest runs when the future is polled. `to_approve` is
/// [`arguments_to_approve`]'s answer for this call.
#[allow(clippy::too_many_arguments)]
fn run_function(
    tracer: Arc<Tracer>,
    parent: Option<String>,
    tool: FunctionTool,
    call: ToolCallItem,
    context: ToolContext,
    include_sensitive_data: bool,
    gate: Gate,
    to_approve: Option<Value>,
) -> BoxFuture<'static, String> {
    let mut span = CallSpan::start(&tracer, parent.as_deref(), &tool.name, &gate);
    if include_sensitive_data {
        span.set("input", json!(call.arguments));
    }
    finish_function(
        span,
        tool,
        call,
        context,
        include_sensitive_data,
        gate,
        to_approve,
    )
    .boxed()
}

/// The part of [`run_function`] that waits: the approval, then the handler.
async fn finish_function(
    mut span: CallSpan,
    tool: FunctionTool,
    call: ToolCallItem,
    context: ToolContext,
    include_sensitive_data: bool,
    gate: Gate,
    to_approve: Option<Value>,
) -> String {
    if let Some(arguments) = to_approve {
        #[cfg(test)]
        if gate.mutant == Some(Mutant::InvokeBeforeApproval) {
            let _ = invoke(
                &tool,
                context.clone(),
                &call.arguments,
                include_sensitive_data,
            )
            .await;
        }
        // Should this future be dropped while it waits (an immediate cancel),
        // the span ends with this error.
        span.set_error(lattice_protocol::SpanError {
            message: CANCELLED_WHILE_WAITING.to_owned(),
            data: None,
        });
        let decided = ask(&gate, &tool, &call, &context, arguments).await;
        span.clear_error();
        if let Err(output) = decided {
            // The handler never runs; the span keeps `output: null`.
            span.finish();
            return output;
        }
    }
    // `invoke_mcp_tool` names the server on the span as the call runs.
    if let Some(server) = &tool.mcp_server {
        span.set("mcp_data", json!({"server": server}));
    }
    let invocation = invoke(&tool, context, &call.arguments, include_sensitive_data).await;
    if let Some(error) = invocation.span_error {
        span.set_error(error);
    }
    if include_sensitive_data {
        span.set("output", json!(invocation.output));
    }
    span.finish();
    invocation.output
}

/// The parsed arguments of a call that must be approved before it runs, or
/// `None` when it runs without asking: its tool never asks, its predicate says
/// no, or its arguments are not a JSON object (which `invoke` then answers
/// with the error, as for any call).
fn arguments_to_approve(
    tool: &FunctionTool,
    context: &ToolContext,
    arguments: &str,
) -> Option<Value> {
    if matches!(tool.needs_approval, NeedsApproval::Never) {
        return None;
    }
    let parsed = parse_arguments(&tool.name, arguments).ok()?;
    tool.needs_approval.asks(context, &parsed).then_some(parsed)
}

/// Ask the port and wait for its answer: `Ok` to run the handler, or the text
/// the model is told instead.
async fn ask(
    gate: &Gate,
    tool: &FunctionTool,
    call: &ToolCallItem,
    context: &ToolContext,
    arguments: Value,
) -> Result<(), String> {
    #[cfg(test)]
    if gate.mutant == Some(Mutant::DecisionFromText)
        && gate.approved_by_text.contains(&call.call_id)
    {
        return Ok(());
    }
    let Some(port) = gate.port.clone() else {
        #[cfg(test)]
        if gate.mutant == Some(Mutant::ApproveWithoutPort) {
            return Ok(());
        }
        return Err(NO_ONE_ASKED.to_owned());
    };
    let _ = gate
        .events
        .unbounded_send(StreamEvent::ToolApprovalRequested {
            agent: context.agent.clone(),
            call_id: call.call_id.clone(),
            tool: tool.name.clone(),
            arguments: arguments.clone(),
        });
    let request = ApprovalRequest {
        agent: context.agent.clone(),
        call_id: call.call_id.clone(),
        tool: tool.name.clone(),
        arguments,
    };
    // A port that panics has decided nothing: the call does not run.
    let decision = match std::panic::catch_unwind(AssertUnwindSafe(|| port.request(request))) {
        Ok(pending) => AssertUnwindSafe(pending)
            .catch_unwind()
            .await
            .unwrap_or(ApprovalDecision::Reject { note: None }),
        Err(_) => ApprovalDecision::Reject { note: None },
    };
    let _ = gate
        .events
        .unbounded_send(StreamEvent::ToolApprovalResolved {
            agent: context.agent.clone(),
            call_id: call.call_id.clone(),
            approved: decision == ApprovalDecision::Approve,
        });
    match decision {
        ApprovalDecision::Approve => Ok(()),
        ApprovalDecision::Reject { note } => Err(rejection_output(note.as_deref())),
    }
}

/// A call's function span. It ends when finished or dropped, like any span;
/// under AF3's mutant only, one dropped unfinished is left open.
struct CallSpan {
    span: Option<Span>,
    #[cfg(test)]
    leave_open: bool,
}

impl CallSpan {
    fn start(tracer: &Arc<Tracer>, parent: Option<&str>, name: &str, _gate: &Gate) -> Self {
        Self {
            span: Some(Span::start(tracer, parent, function_data(name))),
            #[cfg(test)]
            leave_open: _gate.mutant == Some(Mutant::LeaveSpanOpenOnCancel),
        }
    }

    fn set(&mut self, key: &str, value: Value) {
        if let Some(span) = self.span.as_mut() {
            span.data_mut()[key] = value;
        }
    }

    fn set_error(&mut self, error: lattice_protocol::SpanError) {
        if let Some(span) = self.span.as_mut() {
            span.set_error(error);
        }
    }

    fn clear_error(&mut self) {
        if let Some(span) = self.span.as_mut() {
            span.clear_error();
        }
    }

    fn finish(mut self) {
        if let Some(mut span) = self.span.take() {
            span.finish();
        }
    }
}

#[cfg(test)]
impl Drop for CallSpan {
    fn drop(&mut self) {
        if self.leave_open
            && let Some(span) = self.span.take()
        {
            std::mem::forget(span);
        }
    }
}

/// The falsifiers' mutants (spec section 16.4, suite AF): each switches one
/// safeguard off, so a test can show that it fails without it. Test builds
/// only; nothing shipped can select one.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mutant {
    /// AF1: the handler runs before the approval is awaited.
    InvokeBeforeApproval,
    /// AF2: with no port configured, a call that needs approval runs.
    ApproveWithoutPort,
    /// AF3: a call dropped while it waits leaves its span open.
    LeaveSpanOpenOnCancel,
    /// AF4: a call is approved by earlier tool output that names it.
    DecisionFromText,
    /// AF5: steers are taken after the model call instead of before it.
    SteerAfterModelCall,
    /// AF7: nothing is taken, and steering is not closed, at the final output.
    NoDrainAtFinal,
    /// AF7 (A4a): the final output's steers are taken out of the control
    /// before the output guardrails and the final save, so an ending there
    /// drops them with the run.
    DrainBeforeOutputGuardrails,
    /// A3a: every call of a response is awaited and the outputs appended in
    /// call order, with no save before the approvals end.
    OutputsInCallOrder,
}

#[cfg(test)]
mod af_tests;

/// `resolve_tool_name_collisions` under the SDK's default `warn` policy: when
/// several function tools and handoffs share a tool name, the model is offered
/// only the one dispatch would choose: the last handoff of that name if there is
/// one, otherwise the last entry.
fn resolve_tool_name_collisions(agent: &Agent) -> (Vec<&FunctionTool>, Vec<&Handoff>) {
    #[derive(Clone, Copy)]
    enum Entry {
        Tool(usize),
        Handoff(usize),
    }
    let mut by_name: HashMap<&str, Vec<Entry>> = HashMap::new();
    for (index, tool) in agent.tools.iter().enumerate() {
        by_name
            .entry(tool.name.as_str())
            .or_default()
            .push(Entry::Tool(index));
    }
    for (index, handoff) in agent.handoffs.iter().enumerate() {
        if !handoff.tool_name.is_empty() {
            by_name
                .entry(handoff.tool_name.as_str())
                .or_default()
                .push(Entry::Handoff(index));
        }
    }
    let mut dropped_tools = HashSet::new();
    let mut dropped_handoffs = HashSet::new();
    for entries in by_name.values().filter(|entries| entries.len() > 1) {
        let winner = entries
            .iter()
            .rposition(|entry| matches!(entry, Entry::Handoff(_)))
            .unwrap_or(entries.len() - 1);
        for (position, entry) in entries.iter().enumerate() {
            if position == winner {
                continue;
            }
            match entry {
                Entry::Tool(index) => dropped_tools.insert(*index),
                Entry::Handoff(index) => dropped_handoffs.insert(*index),
            };
        }
    }
    (
        agent
            .tools
            .iter()
            .enumerate()
            .filter(|(i, _)| !dropped_tools.contains(i))
            .map(|(_, t)| t)
            .collect(),
        agent
            .handoffs
            .iter()
            .enumerate()
            .filter(|(i, _)| !dropped_handoffs.contains(i))
            .map(|(_, h)| h)
            .collect(),
    )
}

/// A recorded model output in the Responses shape the SDK records for a Chat
/// Completions call (`[final_response.model_dump()]`), reduced to its content.
fn response_dump(model: &str, response: &ModelResponse) -> Value {
    let mut output: Vec<Value> = Vec::new();
    // The message whose content a refusal joins: the SDK's message item holds
    // the text part and the refusal part side by side.
    let mut open_message: Option<usize> = None;
    for item in &response.output {
        match item {
            OutputItem::Message { text } => {
                output.push(json!({
                    "id": FAKE_RESPONSES_ID,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
                }));
                open_message = Some(output.len() - 1);
            }
            OutputItem::Refusal { text } => {
                let part = json!({"type": "refusal", "refusal": text});
                match open_message.and_then(|at| output[at]["content"].as_array_mut()) {
                    Some(content) => content.push(part),
                    None => {
                        output.push(json!({
                            "id": FAKE_RESPONSES_ID,
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [part],
                        }));
                        open_message = Some(output.len() - 1);
                    }
                }
            }
            OutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => output.push(json!({
                "id": FAKE_RESPONSES_ID,
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": "completed",
            })),
            OutputItem::Reasoning { text } => output.push(json!({
                "id": FAKE_RESPONSES_ID,
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": text}],
            })),
        }
    }
    json!({
        "id": FAKE_RESPONSES_ID,
        "object": "response",
        "model": model,
        "status": "completed",
        "output": output,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::tool::strict_object_schema;

    fn tool(name: &str) -> FunctionTool {
        FunctionTool::new(
            name,
            "",
            strict_object_schema(json!({}), &[]),
            |_, _| async { Ok(String::new()) },
        )
    }

    #[test]
    fn run_item_names_keep_the_sdks_spelling_in_serialised_form() {
        assert_eq!(
            serde_json::to_value(RunItemName::HandoffOccurred).unwrap(),
            json!("handoff_occured")
        );
        assert_eq!(RunItemName::HandoffOccurred.as_str(), "handoff_occured");
        assert_eq!(
            serde_json::to_value(RunItemName::ReasoningItemCreated).unwrap(),
            json!("reasoning_item_created")
        );
        let item = RunItem::HandoffOutput {
            agent: "A".into(),
            call_id: "c".into(),
            source_agent: "A".into(),
            target_agent: "B".into(),
            output: "{}".into(),
        };
        assert_eq!(item.event_name(), RunItemName::HandoffOccurred);
    }

    #[test]
    fn stream_events_are_named_in_the_sdks_vocabulary() {
        assert_eq!(
            StreamEvent::AgentUpdated { agent: "A".into() }.sdk_name(),
            "agent_updated_stream_event"
        );
        assert_eq!(
            StreamEvent::RawTextDelta {
                agent: "A".into(),
                delta: "x".into()
            }
            .sdk_name(),
            "raw_text_delta"
        );
        let event = StreamEvent::RunItem {
            name: RunItemName::ToolCalled,
            item: RunItem::ToolCall {
                agent: "A".into(),
                call_id: "c".into(),
                name: "t".into(),
                arguments: "{}".into(),
            },
        };
        assert_eq!(event.sdk_name(), "tool_called");
        assert_eq!(serde_json::to_value(&event).unwrap()["name"], "tool_called");
    }

    #[test]
    fn a_handoff_wins_a_name_collision_with_a_tool_and_the_last_duplicate_tool_wins() {
        let target = Agent::builder("Helper").build();
        let agent = Agent::builder("Root")
            .tool(tool("transfer_to_helper"))
            .tool(tool("dup"))
            .tool(tool("dup"))
            .tool(tool("solo"))
            .handoff_to(target)
            .build();
        let (tools, handoffs) = resolve_tool_name_collisions(&agent);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["dup", "solo"]);
        assert!(std::ptr::eq(tools[0], &agent.tools[2]));
        assert_eq!(handoffs.len(), 1);
    }

    #[test]
    fn the_recorded_generation_output_is_a_responses_shaped_dump() {
        let response = ModelResponse {
            output: vec![
                OutputItem::Message { text: "hi".into() },
                OutputItem::FunctionCall {
                    call_id: "c1".into(),
                    name: "t".into(),
                    arguments: "{}".into(),
                },
            ],
            usage: None,
        };
        let dump = response_dump("m", &response);
        assert_eq!(dump["model"], "m");
        assert_eq!(dump["output"][0]["content"][0]["text"], "hi");
        assert_eq!(dump["output"][1]["call_id"], "c1");
    }

    #[test]
    fn run_errors_name_the_sdk_exceptions() {
        assert_eq!(
            RunError::MaxTurnsExceeded { max_turns: 2 }.sdk_name(),
            "MaxTurnsExceeded"
        );
        assert_eq!(
            RunError::MaxTurnsExceeded { max_turns: 2 }.to_string(),
            "Max turns (2) exceeded"
        );
        assert_eq!(
            RunError::InputGuardrailTripwire {
                guardrail: "g".into(),
                output_info: Value::Null
            }
            .sdk_name(),
            "InputGuardrailTripwireTriggered"
        );
        assert_eq!(
            RunError::Model(ModelError::Behavior("x".into())).sdk_name(),
            "ModelBehaviorError"
        );
    }
}
