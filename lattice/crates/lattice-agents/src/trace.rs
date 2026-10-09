//! Tracing: the SDK's trace and span records, delivered to local processors.
//!
//! Ports `agents.tracing` (0.22.3): `TraceImpl`, `SpanImpl`, the span data
//! classes' `export()` shapes, `SynchronousMultiTracingProcessor` and the id and
//! timestamp generators.
//!
//! THERE IS NO EXPORTER. This crate has no code path that sends a trace to any
//! machine: the SDK's `BatchTraceProcessor` and `BackendSpanExporter` (which post
//! to OpenAI) are deliberately not ported, and the only consumers are the
//! [`TraceProcessor`]s the caller puts in [`crate::RunConfig::processors`]. A
//! source-guard test keeps the SDK's endpoints out of this crate's sources.
//!
//! Invariants:
//! - Processors are called synchronously, in registration order, in the order
//!   things happen: trace start, span starts and ends, trace end. A processor
//!   that panics is skipped for that call and counted; it never stops the run
//!   or the processors after it.
//! - `SpanRecord::order` is assigned when the span starts (1, 2, 3 ... per
//!   trace), so a parent always sorts before its children.
//! - A span's start record is a snapshot: the SDK fills span data in after the
//!   start (tools, handoffs, the handoff target, function input and output,
//!   usage), so the record delivered at the end is the authoritative one.
//! - A span that is dropped without an explicit `finish` (a run cancelled in
//!   the middle of a model call, say) is finished by `Drop`, so every started
//!   span ends.
//! - Ids are `trace_` + 32 hex digits and `span_` + 24 hex digits (a v4 UUID's
//!   hex form), as the SDK generates them. Timestamps are UTC ISO 8601 with
//!   microseconds and `+00:00`.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lattice_protocol::{SpanError, SpanRecord, TraceInfo};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::pyjson::iso_now;

/// A trace: one run, as the SDK's `Trace.export()` describes it (`id`,
/// `workflow_name`, `group_id`, `metadata`) plus the times the SDK's export
/// lacks.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceRecord {
    pub id: String,
    pub workflow_name: String,
    pub group_id: Option<String>,
    pub metadata: Option<Value>,
    pub started_at: String,
    /// `None` until the trace ends.
    pub ended_at: Option<String>,
}

impl From<&TraceRecord> for TraceInfo {
    fn from(trace: &TraceRecord) -> Self {
        TraceInfo {
            id: trace.id.clone(),
            workflow_name: trace.workflow_name.clone(),
            started_at: Some(trace.started_at.clone()),
            ended_at: trace.ended_at.clone(),
        }
    }
}

/// Receives a run's trace and spans, on the run's task, in order. Implementations
/// must be quick and must not block: they run inline with the agent loop.
pub trait TraceProcessor: Send + Sync {
    fn on_trace_start(&self, trace: &TraceRecord);
    fn on_trace_end(&self, trace: &TraceRecord);
    /// The span has started. Its data is what the SDK knows at the start;
    /// `on_span_end` carries the final record.
    fn on_span_start(&self, span: &SpanRecord);
    fn on_span_end(&self, span: &SpanRecord);
}

/// `trace_` + 32 hex digits.
pub(crate) fn new_trace_id() -> String {
    format!("trace_{}", Uuid::new_v4().simple())
}

/// `span_` + the first 24 hex digits of a v4 UUID.
fn new_span_id() -> String {
    let hex = Uuid::new_v4().simple().to_string();
    format!("span_{}", &hex[..24])
}

/// One run's tracing: the processors, the span counter, the panic counter.
pub(crate) struct Tracer {
    trace: Mutex<TraceRecord>,
    processors: Vec<Arc<dyn TraceProcessor>>,
    order: AtomicU64,
    panics: Arc<AtomicU64>,
}

impl Tracer {
    pub(crate) fn new(
        trace_id: Option<String>,
        workflow_name: String,
        group_id: Option<String>,
        metadata: Option<Value>,
        processors: Vec<Arc<dyn TraceProcessor>>,
        panics: Arc<AtomicU64>,
    ) -> Arc<Self> {
        Arc::new(Self {
            trace: Mutex::new(TraceRecord {
                id: trace_id.unwrap_or_else(new_trace_id),
                workflow_name,
                group_id,
                metadata,
                started_at: String::new(),
                ended_at: None,
            }),
            processors,
            order: AtomicU64::new(0),
            panics,
        })
    }

    pub(crate) fn trace_id(&self) -> String {
        self.trace
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .id
            .clone()
    }

    pub(crate) fn start_trace(&self) {
        let record = {
            let mut trace = self
                .trace
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            trace.started_at = iso_now();
            trace.clone()
        };
        self.each(|processor| processor.on_trace_start(&record));
    }

    pub(crate) fn end_trace(&self) {
        let record = {
            let mut trace = self
                .trace
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // A trace that never started (a run that ended before it began,
            // such as one whose session could not be read) has nothing to end.
            if trace.ended_at.is_some() || trace.started_at.is_empty() {
                return;
            }
            trace.ended_at = Some(iso_now());
            trace.clone()
        };
        self.each(|processor| processor.on_trace_end(&record));
    }

    /// Call every processor, in order, surviving a panic in any of them.
    fn each(&self, call: impl Fn(&dyn TraceProcessor)) {
        for processor in &self.processors {
            if catch_unwind(AssertUnwindSafe(|| call(processor.as_ref()))).is_err() {
                self.panics.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// An open span. Finishing is idempotent and also happens on drop.
pub(crate) struct Span {
    tracer: Arc<Tracer>,
    record: SpanRecord,
    finished: bool,
}

impl Span {
    /// Start a span under `parent_id` (a span id, or `None` for a root span).
    pub(crate) fn start(tracer: &Arc<Tracer>, parent_id: Option<&str>, span_data: Value) -> Self {
        let record = SpanRecord {
            order: tracer.order.fetch_add(1, Ordering::Relaxed) + 1,
            id: new_span_id(),
            trace_id: tracer.trace_id(),
            parent_id: parent_id.map(str::to_owned),
            started_at: iso_now(),
            ended_at: None,
            span_data,
            error: None,
        };
        tracer.each(|processor| processor.on_span_start(&record));
        Self {
            tracer: tracer.clone(),
            record,
            finished: false,
        }
    }

    pub(crate) fn id(&self) -> &str {
        &self.record.id
    }

    /// The span's data, for filling in after the start.
    pub(crate) fn data_mut(&mut self) -> &mut Value {
        &mut self.record.span_data
    }

    pub(crate) fn set_error(&mut self, error: SpanError) {
        self.record.error = Some(error);
    }

    pub(crate) fn clear_error(&mut self) {
        self.record.error = None;
    }

    pub(crate) fn has_error(&self) -> bool {
        self.record.error.is_some()
    }

    pub(crate) fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.record.ended_at = Some(iso_now());
        let record = &self.record;
        self.tracer.each(|processor| processor.on_span_end(record));
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        self.finish();
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Span")
            .field("id", &self.record.id)
            .field("finished", &self.finished)
            .finish()
    }
}

/// `SpanError(message=..., data=...)`.
pub(crate) fn span_error(message: &str, data: Value) -> SpanError {
    SpanError {
        message: message.to_owned(),
        data: Some(data),
    }
}

// The span data shapes: `SpanData.export()` at the moment the SDK creates each
// span. Fields the SDK fills in later start as `null`, exactly as its
// constructors leave them.

pub(crate) fn agent_data(name: &str) -> Value {
    json!({"type": "agent", "name": name, "handoffs": [], "tools": [], "output_type": "str"})
}

pub(crate) fn task_data(name: &str) -> Value {
    json!({"type": "custom", "name": "task", "data": {"sdk_span_type": "task", "name": name}})
}

pub(crate) fn turn_data(turn: u32, agent_name: &str) -> Value {
    json!({"type": "custom", "name": "turn", "data": {"sdk_span_type": "turn", "turn": turn, "agent_name": agent_name}})
}

pub(crate) fn function_data(name: &str) -> Value {
    json!({"type": "function", "name": name, "input": null, "output": null, "mcp_data": null})
}

pub(crate) fn generation_data() -> Value {
    json!({"type": "generation", "input": null, "output": null, "model": null, "model_config": null, "usage": null})
}

pub(crate) fn handoff_data(from_agent: &str) -> Value {
    json!({"type": "handoff", "from_agent": from_agent, "to_agent": null})
}

pub(crate) fn guardrail_data(name: &str) -> Value {
    json!({"type": "guardrail", "name": name, "triggered": false})
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[derive(Default)]
    struct Log(Mutex<Vec<String>>);

    impl TraceProcessor for Log {
        fn on_trace_start(&self, trace: &TraceRecord) {
            self.0
                .lock()
                .unwrap()
                .push(format!("trace_start {}", trace.workflow_name));
        }
        fn on_trace_end(&self, trace: &TraceRecord) {
            self.0
                .lock()
                .unwrap()
                .push(format!("trace_end {}", trace.ended_at.is_some()));
        }
        fn on_span_start(&self, span: &SpanRecord) {
            self.0
                .lock()
                .unwrap()
                .push(format!("span_start {} {}", span.order, span.data_type()));
        }
        fn on_span_end(&self, span: &SpanRecord) {
            self.0.lock().unwrap().push(format!(
                "span_end {} {}",
                span.order,
                span.ended_at.is_some()
            ));
        }
    }

    struct Panicker;

    impl TraceProcessor for Panicker {
        fn on_trace_start(&self, _: &TraceRecord) {
            panic!("processor failure");
        }
        fn on_trace_end(&self, _: &TraceRecord) {}
        fn on_span_start(&self, _: &SpanRecord) {
            panic!("processor failure");
        }
        fn on_span_end(&self, _: &SpanRecord) {
            panic!("processor failure");
        }
    }

    fn tracer(processors: Vec<Arc<dyn TraceProcessor>>) -> (Arc<Tracer>, Arc<AtomicU64>) {
        let panics = Arc::new(AtomicU64::new(0));
        (
            Tracer::new(
                None,
                "Agent workflow".into(),
                None,
                None,
                processors,
                panics.clone(),
            ),
            panics,
        )
    }

    #[test]
    fn ids_have_the_sdk_shape() {
        let trace = new_trace_id();
        assert!(
            trace.starts_with("trace_") && trace.len() == 6 + 32,
            "{trace}"
        );
        let span = new_span_id();
        assert!(span.starts_with("span_") && span.len() == 5 + 24, "{span}");
        assert!(
            span[5..]
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        );
        assert_ne!(new_span_id(), new_span_id());
    }

    #[test]
    fn processors_hear_everything_in_order_and_orders_count_up_from_one() {
        let log = Arc::new(Log::default());
        let (tracer, _) = tracer(vec![log.clone()]);
        tracer.start_trace();
        let mut task = Span::start(&tracer, None, task_data("Agent workflow"));
        let mut child = Span::start(&tracer, Some(task.id()), turn_data(1, "A"));
        assert_eq!(child.record.parent_id.as_deref(), Some(task.id()));
        child.finish();
        child.finish();
        task.finish();
        tracer.end_trace();
        tracer.end_trace();
        assert_eq!(
            *log.0.lock().unwrap(),
            [
                "trace_start Agent workflow",
                "span_start 1 custom",
                "span_start 2 custom",
                "span_end 2 true",
                "span_end 1 true",
                "trace_end true",
            ]
        );
    }

    #[test]
    fn a_trace_that_never_started_is_never_ended() {
        let log = Arc::new(Log::default());
        let (tracer, _) = tracer(vec![log.clone()]);
        tracer.end_trace();
        assert!(log.0.lock().unwrap().is_empty());
        tracer.start_trace();
        tracer.end_trace();
        assert_eq!(
            *log.0.lock().unwrap(),
            ["trace_start Agent workflow", "trace_end true"]
        );
    }

    #[test]
    fn a_dropped_span_still_ends() {
        let log = Arc::new(Log::default());
        let (tracer, _) = tracer(vec![log.clone()]);
        drop(Span::start(&tracer, None, generation_data()));
        assert_eq!(
            *log.0.lock().unwrap(),
            ["span_start 1 generation", "span_end 1 true"]
        );
    }

    #[test]
    fn a_panicking_processor_is_counted_and_the_others_still_run() {
        let log = Arc::new(Log::default());
        let (tracer, panics) = tracer(vec![Arc::new(Panicker), log.clone()]);
        tracer.start_trace();
        let mut span = Span::start(&tracer, None, agent_data("A"));
        span.finish();
        tracer.end_trace();
        assert_eq!(panics.load(Ordering::Relaxed), 3);
        assert_eq!(log.0.lock().unwrap().len(), 4);
    }

    #[test]
    fn a_trace_record_converts_to_the_protocols_trace_info() {
        let (tracer, _) = tracer(vec![]);
        tracer.start_trace();
        let record = tracer.trace.lock().unwrap().clone();
        let info = TraceInfo::from(&record);
        assert_eq!(info.id, record.id);
        assert!(info.started_at.is_some() && info.ended_at.is_none());
    }

    #[test]
    fn span_data_starts_the_way_the_sdk_constructs_it() {
        assert_eq!(agent_data("A")["output_type"], "str");
        assert_eq!(generation_data()["model"], Value::Null);
        assert_eq!(handoff_data("A")["to_agent"], Value::Null);
        assert_eq!(turn_data(2, "A")["data"]["sdk_span_type"], "turn");
    }
}
