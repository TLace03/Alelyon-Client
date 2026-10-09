//! The recorder: the trace processor that turns a run's spans and stream
//! events into [`RunEvent`](lattice_protocol::RunEvent)s, in the order they
//! happened (the chat core's spec §2.2, §5.7; row E10). Not a port.
//!
//! Extracted from `manager.rs` with no change in behaviour: the run manager
//! records its runs through it exactly as before (every `manager/tests.rs`
//! test is unchanged), and the agent chat records its turns through it into
//! `<native>/chat/runs/` ([`crate::state::StateRoot::chat_runs_dir`]), in the
//! run store's own format, so a Traces viewer reads them unchanged.
//!
//! - [`RunSink`]: where a run's events go. The run manager's sink is its run
//!   entry; the agent chat's is its turn.
//! - [`Intake`]: the hand-over between the run's stream events and its trace
//!   callbacks, so that the two sources are recorded in one order.
//! - [`Recorder`]: the trace processor of one run.
//! - [`map_stream_event`], [`bounded_span`], [`bounded_text`]: an SDK event
//!   as a run event, within the bounds of [`crate::bound`].
//! - [`ending`]: the events that end a run, and the status it ends with (one
//!   sentence, never the transport's text).
//! - [`redacted_event`]: an event with each secret-looking part of its text
//!   replaced (T15).
//!
//! Nothing here prints, logs or reads the environment.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::Poll;
use std::time::Duration;

use futures::StreamExt;
use futures::channel::mpsc::UnboundedReceiver;
use lattice_agents::{RunError, RunItem, RunResult, StreamEvent, TraceProcessor, TraceRecord};
use lattice_protocol::{EndStatus, RunEvent, RunEventKind, SpanRecord};
use serde_json::Value;
use tokio::task::JoinError;

use crate::bound::{MAX_TEXT_CHARS, MAX_VALUE_CHARS, bound_value, cap_text};
use crate::models;
use crate::secrets;

/// The longest a guardrail's reason may be.
pub const REASON_CHARS: usize = 500;
/// The longest the first trace callback waits for the run's events to be handed over.
pub const INSTALL_WAIT: Duration = Duration::from_secs(10);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Where one run's events are recorded.
pub trait RunSink: Send + Sync {
    /// Record one run event.
    fn record(&self, kind: RunEventKind);
    /// Record one of the run's stream events (mapped by the sink, usually
    /// through [`map_stream_event`]).
    fn record_stream(&self, event: StreamEvent);
}

/// `event`, with each secret-looking part of any text in it replaced; `None`
/// when it held none.
pub fn redacted_event(event: &RunEvent) -> Option<RunEvent> {
    let mut value = serde_json::to_value(event).ok()?;
    if !secrets::redact_strings(&mut value) {
        return None;
    }
    serde_json::from_value(value).ok()
}

/// Bound one span record for the log.
pub fn bounded_span(span: &SpanRecord) -> SpanRecord {
    let mut span = span.clone();
    let mut budget = MAX_VALUE_CHARS;
    bound_value(&mut span.span_data, &mut budget);
    if let Some(error) = span.error.as_mut() {
        error.message = cap_text(&error.message, MAX_TEXT_CHARS);
        if let Some(data) = error.data.as_mut() {
            let mut budget = MAX_VALUE_CHARS;
            bound_value(data, &mut budget);
        }
    }
    span
}

pub fn bounded_text(value: String) -> String {
    cap_text(&value, MAX_TEXT_CHARS)
}

/// The run event an SDK stream event stands for, or `None` for an event only
/// a native caller's configuration produces (a session, approvals, steering).
/// The run manager configures none of them, so its runs never send one and
/// their records are exactly what they were; the match stays exhaustive so a
/// new event is decided here, not silently dropped.
pub fn map_stream_event(event: StreamEvent) -> Option<RunEventKind> {
    Some(match event {
        StreamEvent::RawTextDelta { agent, delta } => RunEventKind::Delta {
            agent: bounded_text(agent),
            text: bounded_text(delta),
        },
        StreamEvent::AgentUpdated { agent } => RunEventKind::Agent {
            name: bounded_text(agent),
        },
        StreamEvent::SessionWriteFailed { .. }
        | StreamEvent::ToolApprovalRequested { .. }
        | StreamEvent::ToolApprovalResolved { .. }
        | StreamEvent::Steered { .. } => return None,
        StreamEvent::RunItem { item, .. } => match item {
            RunItem::MessageOutput {
                agent,
                text: message,
            } => RunEventKind::Message {
                agent: bounded_text(agent),
                text: bounded_text(message),
            },
            RunItem::ToolCall {
                agent,
                call_id,
                name,
                arguments,
            } => RunEventKind::ToolCall {
                agent: bounded_text(agent),
                name: bounded_text(name),
                call_id: bounded_text(call_id),
                arguments: bounded_text(arguments),
            },
            RunItem::ToolOutput {
                agent,
                call_id,
                output,
            } => RunEventKind::ToolOutput {
                agent: bounded_text(agent),
                call_id: bounded_text(call_id),
                output: bounded_text(output),
            },
            RunItem::HandoffCall { agent, .. } => RunEventKind::HandoffRequested {
                agent: bounded_text(agent),
            },
            RunItem::HandoffOutput {
                source_agent,
                target_agent,
                ..
            } => RunEventKind::Handoff {
                from: bounded_text(source_agent),
                to: bounded_text(target_agent),
            },
            RunItem::Reasoning {
                agent,
                text: reasoning,
            } => RunEventKind::Reasoning {
                agent: bounded_text(agent),
                text: bounded_text(reasoning),
            },
        },
    })
}

/// The hand-over between the run's stream events and its trace callbacks.
///
/// The run sends stream events into a channel and calls the trace processor
/// synchronously, in order. To keep one order, whoever records something first
/// drains the channel: the processor before it records a trace or span event
/// (so everything the run sent earlier is recorded first), and a consumer task
/// for the rest. Both hold `slot` while they record, so neither can slip an
/// event between the other's drain and its own.
pub struct Intake {
    slot: Mutex<Option<UnboundedReceiver<StreamEvent>>>,
    installed: Condvar,
}

impl Intake {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            slot: Mutex::new(None),
            installed: Condvar::new(),
        })
    }

    pub fn install(&self, receiver: UnboundedReceiver<StreamEvent>) {
        *lock(&self.slot) = Some(receiver);
        self.installed.notify_all();
    }

    /// Wait (briefly) until the channel is handed over: the run starts before
    /// `start` has it in hand, and its first callback must not go first.
    fn wait_installed(&self) {
        let guard = lock(&self.slot);
        let _ = self
            .installed
            .wait_timeout_while(guard, INSTALL_WAIT, |slot| slot.is_none())
            .unwrap_or_else(PoisonError::into_inner);
    }

    /// Record everything the run has already sent, then `kind`.
    pub fn record_after_pending<S: RunSink + ?Sized>(&self, sink: &S, kind: RunEventKind) {
        let mut slot = lock(&self.slot);
        if let Some(receiver) = slot.as_mut() {
            while let Ok(event) = receiver.try_recv() {
                sink.record_stream(event);
            }
        }
        sink.record(kind);
    }

    /// Record whatever is left (after the run has ended).
    pub fn drain<S: RunSink + ?Sized>(&self, sink: &S) {
        let mut slot = lock(&self.slot);
        if let Some(receiver) = slot.as_mut() {
            while let Ok(event) = receiver.try_recv() {
                sink.record_stream(event);
            }
        }
    }

    /// Record the run's stream events as they arrive, until the run has ended.
    pub async fn consume<S: RunSink + ?Sized>(self: Arc<Self>, sink: Arc<S>) {
        loop {
            let more = std::future::poll_fn(|cx| {
                let mut slot = lock(&self.slot);
                let Some(receiver) = slot.as_mut() else {
                    return Poll::Ready(false);
                };
                match receiver.poll_next_unpin(cx) {
                    Poll::Ready(Some(event)) => {
                        sink.record_stream(event);
                        Poll::Ready(true)
                    }
                    Poll::Ready(None) => Poll::Ready(false),
                    Poll::Pending => Poll::Pending,
                }
            })
            .await;
            if !more {
                break;
            }
        }
    }
}

/// The trace processor of one run: its callbacks become events, recorded
/// into `sink` after every stream event the run sent before them.
pub struct Recorder<S: RunSink + ?Sized> {
    pub sink: Arc<S>,
    pub intake: Arc<Intake>,
}

impl<S: RunSink + ?Sized + 'static> TraceProcessor for Recorder<S> {
    fn on_trace_start(&self, trace: &TraceRecord) {
        self.intake.wait_installed();
        self.intake.record_after_pending(
            self.sink.as_ref(),
            RunEventKind::TraceStart {
                trace_id: trace.id.clone(),
                workflow_name: bounded_text(trace.workflow_name.clone()),
                at: trace.started_at.clone(),
            },
        );
    }

    fn on_trace_end(&self, trace: &TraceRecord) {
        self.intake.record_after_pending(
            self.sink.as_ref(),
            RunEventKind::TraceEnd {
                trace_id: trace.id.clone(),
                at: trace.ended_at.clone().unwrap_or_default(),
            },
        );
    }

    fn on_span_start(&self, span: &SpanRecord) {
        self.intake.record_after_pending(
            self.sink.as_ref(),
            RunEventKind::SpanStart {
                span: bounded_span(span),
            },
        );
    }

    fn on_span_end(&self, span: &SpanRecord) {
        self.intake.record_after_pending(
            self.sink.as_ref(),
            RunEventKind::SpanEnd {
                span: bounded_span(span),
            },
        );
    }
}

/// The events that end a run, and the status it ends with.
pub fn ending(
    outcome: Result<Result<RunResult, RunError>, JoinError>,
    base_url: Option<&str>,
) -> (Vec<RunEventKind>, EndStatus) {
    let failed = |message: String| (vec![RunEventKind::Error { message }], EndStatus::Failed);
    match outcome {
        Ok(Ok(result)) => (
            vec![RunEventKind::Result {
                output: bounded_text(result.final_output),
                usage: result.usage,
                turns: result.turns,
                last_agent: bounded_text(result.last_agent),
            }],
            EndStatus::Completed,
        ),
        Ok(Err(RunError::Cancelled)) => (Vec::new(), EndStatus::Stopped),
        Ok(Err(
            RunError::InputGuardrailTripwire {
                guardrail,
                output_info,
            }
            | RunError::OutputGuardrailTripwire {
                guardrail,
                output_info,
            },
        )) => {
            let reason = output_info
                .get("reason")
                .and_then(Value::as_str)
                .map_or_else(
                    || "A guardrail refused the task.".to_owned(),
                    |reason| cap_text(reason, REASON_CHARS),
                );
            (
                vec![RunEventKind::Guardrail {
                    name: bounded_text(guardrail),
                    message: reason,
                }],
                EndStatus::Refused,
            )
        }
        Ok(Err(RunError::MaxTurnsExceeded { max_turns })) => failed(format!(
            "Stopped after {max_turns} turns without a final answer."
        )),
        Ok(Err(RunError::ModelRefusal { refusal })) => failed(models::refusal_sentence(&refusal)),
        // The address as a trace shows it: no credentials, query or fragment.
        Ok(Err(RunError::Model(error))) => failed(models::error_sentence(&error, base_url)),
        Ok(Err(RunError::User(_))) => {
            failed("The run stopped because one of its checks failed.".to_owned())
        }
        Err(_) => failed("The run stopped unexpectedly.".to_owned()),
    }
}
