//! The run manager: [`CoreService`], the `RunService` over `lattice-agents`.
//!
//! What it owns:
//! - a small tokio runtime (two worker threads named `lattice-runs`) on which
//!   every run executes, so the window never waits on a model; or, when it is
//!   built with [`CoreService::with_handle`], no runtime at all: its runs then
//!   execute on the runtime the caller shares with it, and dropping the service
//!   aborts the tasks it spawned there, as dropping its own runtime would;
//! - the runs: newest first, at most the newest 200 loaded from the store when
//!   it starts, plus the runs of this session;
//! - the agents ([`crate::catalog`]), the model registry and the keys, read
//!   afresh whenever they are asked for, so an edit on disk shows up at once.
//!
//! One run, start to end:
//! 1. `start` checks the request (a task of 1 to 20,000 characters after
//!    trimming; a known agent; a model choice that exists and is ready) and
//!    that fewer than three runs are working, builds the model, writes the
//!    summary, and starts the run with a trace processor that turns the SDK's
//!    trace and span callbacks into events. It returns the summary at once.
//! 2. The run's events are the SDK's stream events (text deltas, agent changes,
//!    messages, tool calls and outputs, handoffs, reasoning) and the trace
//!    processor's (trace and span starts and ends), each given the next `seq`
//!    and the time `at` under the run's lock. The two sources are merged in the
//!    order things happened: before the processor records a span or trace event
//!    it first drains every stream event the run has already sent.
//! 3. It ends with `Result` and `End{completed}`; or `Error` and `End{failed}`
//!    (one sentence, never the transport's text: no address beyond the model's
//!    base URL with its credentials, query and fragment removed, no key, no
//!    response body; a refusal is quoted, bounded to 500 characters); or
//!    `Guardrail` and `End{refused}`, whose message is the guardrail's reason
//!    and never the text it matched, and whose stored task has each
//!    secret-looking part replaced by `[redacted: looks like a secret]`; or
//!    `End{stopped}` after `stop` (a stop that arrives before the run has its
//!    control is queued and applied the moment it does).
//!
//! Storage (see [`crate::store`]): every event except `Delta` is appended to the
//! run's file as it happens, and the file is let go of once `End` is written;
//! the summary is rewritten atomically at the start, at most once a second while
//! the run works, and at the end. A run that was still `running` when Lattice
//! last closed is `interrupted` when the next one starts, with the error "Ended
//! when Lattice last closed.".
//!
//! One writer per directory: the process that records runs holds
//! `<runs>/.lattice-runs.lock` (`File::try_lock`) for its whole life. It takes
//! the lock when the directory exists at construction, else at its first `start`
//! (or when `status` finds the directory free). A process that cannot take it is
//! a reader: it lists and follows the runs on disk (looking again at most once a
//! second, and every 300 ms while it follows a run still working), never
//! rewrites a record another process owns, `start` refuses with `Unavailable`,
//! and `status().refusal` says so. A reader that finds the lock free becomes the
//! writer, and ends the records the window that left had not.
//!
//! Bounds: 20,000 persisted events per run (past it, further events other than
//! the ones that end the run are dropped and counted); 50,000 live deltas and at
//! most 8 MiB of their text (beyond that, dropped and counted); every string
//! capped at 20,000 characters and a span's strings at 200,000 together
//! ([`crate::bound`]); events kept in memory for the sixteen most recently used
//! finished runs (the rest are read again from disk when asked for).
//!
//! `follow` gives a stream of batches: each time it is polled after new events
//! arrived, everything after the last `seq` it gave, as one `Vec`; it ends after
//! the batch that holds `End`.
//!
//! Invariants: a run's events have `seq` 1, 2, 3 ... with no repeat and no
//! reordering (deltas that are dropped take no `seq`); `End` is the last event
//! and is always recorded; a failure to write to disk never stops a run (it is
//! counted, see [`CoreService::persist_failures`]); no lock is held across an
//! `.await`; nothing here blocks on the network.

use std::collections::VecDeque;
use std::fs::File;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use futures::stream::BoxStream;
use futures::{StreamExt, stream};
use lattice_agents::{
    CancelMode, Model, RunConfig, RunControl, RunError, RunHandle, RunResult, StreamEvent,
    run_streamed,
};
// The run manager's tests name it through `use super::*`; its wording moved
// to `models`, and the tests are kept unchanged.
#[cfg(test)]
use lattice_agents::ModelError;
use lattice_protocol::{
    AgentInfo, EndStatus, Locality, ModelChoice, Refusal, RefusalKind, RunDetail, RunEvent,
    RunEventKind, RunId, RunService, RunStatus, RunSummary, ServiceStatus, SpanRecord, StartRun,
    TraceInfo, is_run_id,
};
use tokio::runtime::{Builder, Handle, Runtime};
use tokio::sync::watch;
use tokio::task::{AbortHandle, JoinHandle};
use uuid::Uuid;

use crate::bound::cap_text;
use crate::catalog::{AgentRunContext, Catalog};
use crate::choices;
use crate::clock::{Clock, system_clock};
use crate::devmodel::{DEFAULT_STEP_DELAY, DevModel};
use crate::env::{Env, ProcessEnv};
use crate::keys::KeyStore;
use crate::models;
use crate::recorder::{Intake, Recorder, RunSink, ending, map_stream_event, redacted_event};
// The run manager's tests name these through `use super::*`; they moved to
// `recorder` (row E10), and the tests are kept unchanged.
#[cfg(test)]
use crate::bound::MAX_TEXT_CHARS;
#[cfg(test)]
use crate::recorder::bounded_span;
#[cfg(test)]
use lattice_agents::RunItem;

use crate::secrets;
use crate::state::{self, StateRoot};
use crate::store::{MAX_PERSISTED_EVENTS, RunStore, WriterLock};

/// Runs that may work at once.
pub const MAX_ACTIVE_RUNS: usize = 3;
/// The longest task, in characters, after trimming.
pub const MAX_TASK_CHARS: usize = 20_000;
/// Live `Delta` events kept per run.
pub const MAX_LIVE_DELTAS: usize = 50_000;
/// The most characters of streamed text kept per run, in all its deltas.
pub const MAX_LIVE_DELTA_CHARS: usize = 8 * 1024 * 1024;
/// Runs loaded from the store at start.
pub const LOADED_RUNS: usize = 200;
/// Finished runs whose events stay in memory.
const CACHED_FINISHED_RUNS: usize = 16;
/// The shortest time between two non-final summary writes of one run.
const SUMMARY_WRITE_INTERVAL: f64 = 1.0;
/// The final output as the list carries it.
const SUMMARY_OUTPUT_CHARS: usize = 2_000;
/// How often a process that does not record runs looks at the disk again for
/// runs it has not seen.
const SYNC_INTERVAL: Duration = Duration::from_secs(1);
/// How often a follower of a run another process records looks at its files
/// (nothing in this process wakes it).
const FOREIGN_POLL: Duration = Duration::from_millis(300);
/// How long a run another process records may stay silent before a follower
/// stops waiting for it: a window that died leaves its run `running` for good.
const FOREIGN_SILENCE: Duration = Duration::from_secs(600);

const INTERRUPTED: &str = "Ended when Lattice last closed.";
/// What a window that may not record runs says, and what `start` refuses with.
const ANOTHER_WINDOW: &str =
    "Another Lattice window is recording runs. Start runs there, or close it.";

/// What a service is built from.
#[derive(Clone)]
pub struct CoreConfig {
    /// Where state lives (the run store is under `<globals>/lattice_native/runs`).
    pub state: StateRoot,
    /// Where environment variables come from.
    pub env: Arc<dyn Env>,
    /// Offer the scripted development model, `dev:scripted`.
    pub development: bool,
    pub clock: Clock,
    /// The development model's pause before each step.
    pub dev_step_delay: Duration,
    /// A hook to give a run a different client than the one a registry endpoint
    /// describes (tests use it to keep every run off the network and to see
    /// whether a model was called). It is offered each real endpoint's client
    /// configuration and returns `None` to use the real client.
    pub model_factory: Option<ModelFactory>,
}

/// See [`CoreConfig::model_factory`] (the type lives in [`crate::models`], which
/// chat shares).
pub use crate::models::ModelFactory;

impl CoreConfig {
    /// This process: its environment, its state root, the system clock.
    pub fn from_process(development: bool) -> Self {
        Self {
            state: state::resolve(),
            env: Arc::new(ProcessEnv),
            development,
            clock: system_clock(),
            dev_step_delay: DEFAULT_STEP_DELAY,
            model_factory: None,
        }
    }

    /// A configuration over `state` and `env`, with the system clock.
    pub fn new(state: StateRoot, env: Arc<dyn Env>) -> Self {
        Self {
            state,
            env,
            development: false,
            clock: system_clock(),
            dev_step_delay: DEFAULT_STEP_DELAY,
            model_factory: None,
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

fn refuse(kind: RefusalKind, message: &str) -> Refusal {
    Refusal::new(kind, message)
}

fn not_found() -> Refusal {
    refuse(RefusalKind::NotFound, "That run does not exist.")
}

/// Sixteen lowercase hexadecimal characters from the operating system's random
/// generator (a version-4 UUID's random bytes, without the six fixed bits).
pub(crate) fn new_run_id() -> RunId {
    let bytes = *Uuid::new_v4().as_bytes();
    bytes[..6]
        .iter()
        .chain(&bytes[9..11])
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// What the entries of every run share.
struct Shared {
    store: RunStore,
    clock: Clock,
    persist_failures: AtomicU64,
    /// This process records runs here (it holds the writer lock, or could not
    /// try one): no other process writes to the directory, so what it reads
    /// back is what it wrote. False in a process that may only read.
    writer: AtomicBool,
}

impl Shared {
    fn note_failure(&self, what: &str) {
        if self.persist_failures.fetch_add(1, Ordering::Relaxed) == 0 {
            // Once, and with no content: the run goes on in memory.
            eprintln!("lattice: could not save a run ({what}); it continues in memory only");
        }
    }
}

struct EntryState {
    summary: RunSummary,
    events: Vec<RunEvent>,
    /// The events are in memory (always true for a run started in this process).
    loaded: bool,
    next_seq: u64,
    persisted: usize,
    live_deltas: usize,
    live_delta_chars: usize,
    dropped_events: u64,
    dropped_deltas: u64,
    ended: bool,
    last_summary_write: f64,
    /// Counts the summary snapshots taken for writing, so a stale one is never
    /// written over a newer one.
    summary_version: u64,
    /// For a run another process records: the length of its events file when
    /// they were last read, and whether that was after the run had ended.
    events_len: Option<u64>,
    read_after_end: bool,
    /// For a run another process records: when it last changed, as far as this
    /// process saw.
    last_change: Instant,
}

/// A run's events file: opened when the first event is saved, closed when the
/// run's `End` has been saved, given up on when a write fails (the run goes on
/// in memory).
struct EventsFile {
    file: Option<File>,
    failed: bool,
}

/// What `stop` needs from a run: its control once it has one, and a stop that
/// came before that.
#[derive(Default)]
struct ControlSlot {
    control: Option<RunControl>,
    stop_requested: bool,
}

/// One run: its summary and events, and what wakes those following it.
struct RunEntry {
    id: RunId,
    shared: Arc<Shared>,
    /// Started by this process. A run read from disk belongs to another window
    /// or to an earlier session, and only a process that holds the writer lock
    /// ever rewrites it.
    owned: bool,
    state: Mutex<EntryState>,
    /// Held for the whole of `append`, taken before `state`: appends and the
    /// writes they make are one at a time, in `seq` order, and a slow disk holds
    /// up the next append but never a reader of the run (who takes `state` only).
    events_file: Mutex<EventsFile>,
    /// Set when saving this run's events failed: they exist only in memory, so
    /// they must stay there.
    unsaved: AtomicBool,
    /// The latest `seq`; a follower waits for it to change.
    wake: watch::Sender<u64>,
    control: Mutex<ControlSlot>,
    /// The version of the summary last written. Held while writing, so two
    /// writes of one run never overlap; the state lock is NOT held, so a slow
    /// disk does not stall a reader of the run.
    summary_io: Mutex<u64>,
}

fn is_status_event(kind: &RunEventKind) -> bool {
    matches!(
        kind,
        RunEventKind::Result { .. }
            | RunEventKind::Error { .. }
            | RunEventKind::Guardrail { .. }
            | RunEventKind::End { .. }
    )
}

impl RunEntry {
    fn new(
        shared: Arc<Shared>,
        summary: RunSummary,
        loaded: bool,
        owned: bool,
        now: f64,
    ) -> Arc<Self> {
        let ended = !summary.status.is_active();
        Arc::new(Self {
            id: summary.id.clone(),
            shared,
            owned,
            state: Mutex::new(EntryState {
                summary,
                events: Vec::new(),
                loaded,
                next_seq: 1,
                persisted: 0,
                live_deltas: 0,
                live_delta_chars: 0,
                dropped_events: 0,
                dropped_deltas: 0,
                ended,
                last_summary_write: now,
                summary_version: 0,
                events_len: None,
                read_after_end: false,
                last_change: Instant::now(),
            }),
            events_file: Mutex::new(EventsFile {
                file: None,
                failed: false,
            }),
            unsaved: AtomicBool::new(false),
            wake: watch::channel(0).0,
            control: Mutex::new(ControlSlot::default()),
            summary_io: Mutex::new(0),
        })
    }

    fn is_active(&self) -> bool {
        lock(&self.state).summary.status.is_active()
    }

    /// Another process records this run, and it is still working as far as this
    /// process can tell.
    fn is_foreign_active(&self) -> bool {
        !self.owned
            && !self.shared.writer.load(Ordering::Relaxed)
            && lock(&self.state).summary.status.is_active()
    }

    fn summary(&self) -> RunSummary {
        lock(&self.state).summary.clone()
    }

    /// Wake those following this run without a new event (its record changed).
    fn poke(&self) {
        self.wake
            .send_modify(|count| *count = count.wrapping_add(1));
    }

    /// Replace the events in memory with what the store holds.
    fn read_events_into(&self, state: &mut EntryState) {
        // The length first: a line appended while reading is then seen as growth
        // at the next look, not missed.
        let length = self.shared.store.events_len(&self.id);
        state.events = self.shared.store.read_events(&self.id);
        // Never `seq + 1` on what a file says: a hostile value must not overflow.
        state.next_seq = state
            .events
            .last()
            .map_or(1, |event| event.seq.saturating_add(1));
        state.persisted = state.events.len();
        state.loaded = true;
        state.ended = !state.summary.status.is_active();
        state.events_len = length;
        state.read_after_end = state.ended;
    }

    /// Read the events from the store if this run's are not in memory.
    fn ensure_loaded(&self) {
        let mut state = lock(&self.state);
        if state.loaded {
            return;
        }
        self.read_events_into(&mut state);
    }

    /// Take a summary another process wrote, for a run that process records.
    fn adopt_disk_summary(&self, summary: RunSummary) {
        if self.owned {
            return;
        }
        let mut state = lock(&self.state);
        if state.summary == summary {
            return;
        }
        let was_active = state.summary.status.is_active();
        state.summary = summary;
        state.last_change = Instant::now();
        if was_active && !state.summary.status.is_active() {
            state.ended = true;
        }
        drop(state);
        self.poke();
    }

    /// For a run another process records: look at its files again, and take what
    /// changed. Does nothing for a run this process records, or in a process
    /// that holds the writer lock (nobody else writes).
    fn refresh_foreign(&self) {
        if self.owned || self.shared.writer.load(Ordering::Relaxed) {
            return;
        }
        if let Some(summary) = self.shared.store.read_summary(&self.id) {
            self.adopt_disk_summary(summary);
        }
        let length = self.shared.store.events_len(&self.id);
        let mut state = lock(&self.state);
        if !state.loaded {
            return;
        }
        let ended = !state.summary.status.is_active();
        if length != state.events_len || (ended && !state.read_after_end) {
            self.read_events_into(&mut state);
            state.last_change = Instant::now();
            drop(state);
            self.poke();
        }
    }

    /// Let go of a finished run's events (they can be read again).
    fn evict(&self) {
        if self.unsaved.load(Ordering::Relaxed) {
            return;
        }
        let mut state = lock(&self.state);
        if state.summary.status.is_active() || !state.loaded {
            return;
        }
        state.events = Vec::new();
        state.loaded = false;
    }

    /// Record one event: give it its `seq` and time, keep it, save it, wake the
    /// followers. Events that are over a bound are dropped and counted instead.
    fn append(&self, kind: RunEventKind) {
        let now = (self.shared.clock)();
        let mut events_file = lock(&self.events_file);
        let mut state = lock(&self.state);
        if state.ended {
            return;
        }
        match &kind {
            RunEventKind::Delta { text, .. } => {
                let chars = text.chars().count();
                if state.live_deltas >= MAX_LIVE_DELTAS
                    || state.live_delta_chars + chars > MAX_LIVE_DELTA_CHARS
                {
                    state.dropped_deltas += 1;
                    return;
                }
                state.live_deltas += 1;
                state.live_delta_chars += chars;
            }
            other if !is_status_event(other) && state.persisted >= MAX_PERSISTED_EVENTS => {
                state.dropped_events += 1;
                return;
            }
            _ => {}
        }
        let seq = state.next_seq;
        state.next_seq = seq.saturating_add(1);
        let event = RunEvent { seq, at: now, kind };
        let is_end = matches!(event.kind, RunEventKind::End { .. });
        if is_end {
            // `End` is published only once it and the final summary are on disk.
            // Whoever sees a run end may read the store at once (the interface
            // lists runs from it); publishing first let a follower, woken by an
            // earlier event, find `End` in memory and read a summary on disk that
            // still said Running. `events_file` stays locked throughout, so no
            // other append can slip in; nothing is appended after `End` anyway.
            let mut summary = state.summary.clone();
            Self::update_summary(&mut summary, &event);
            state.persisted += 1;
            state.last_summary_write = now;
            state.summary_version += 1;
            let version = state.summary_version;
            drop(state);
            self.write_line(&mut events_file, RunStore::encode_event(&event));
            // Nothing is appended after `End`: let go of the file, or a window
            // that has run many runs holds a descriptor for each of them.
            events_file.file = None;
            self.write_summary(&summary, true, version);
            let mut state = lock(&self.state);
            Self::update_summary(&mut state.summary, &event);
            state.events.push(event);
            state.ended = true;
            drop(state);
            drop(events_file);
            self.wake.send_replace(seq);
            return;
        }
        Self::update_summary(&mut state.summary, &event);
        // Encoded here, written below once `state` is free.
        let line = if matches!(event.kind, RunEventKind::Delta { .. }) {
            None
        } else {
            state.persisted += 1;
            Some(RunStore::encode_event(&event))
        };
        state.events.push(event);
        if is_end {
            state.ended = true;
        }
        let mut snapshot = None;
        if is_end || now - state.last_summary_write >= SUMMARY_WRITE_INTERVAL {
            state.last_summary_write = now;
            state.summary_version += 1;
            snapshot = Some((state.summary.clone(), state.summary_version));
        }
        drop(state);
        self.wake.send_replace(seq);
        if let Some(line) = line {
            self.write_line(&mut events_file, line);
        }
        if is_end {
            // Nothing is appended after `End`: let go of the file, or a window
            // that has run many runs holds a descriptor for each of them.
            events_file.file = None;
        }
        drop(events_file);
        if let Some((summary, version)) = snapshot {
            self.write_summary(&summary, is_end, version);
        }
    }

    fn update_summary(summary: &mut RunSummary, event: &RunEvent) {
        summary.updated_at = event.at;
        match &event.kind {
            RunEventKind::SpanEnd { .. } => summary.spans = summary.spans.saturating_add(1),
            RunEventKind::Result { output, usage, .. } => {
                summary.output = Some(cap_text(output, SUMMARY_OUTPUT_CHARS));
                summary.usage = *usage;
            }
            RunEventKind::Error { message } => summary.error = Some(message.clone()),
            RunEventKind::End { status } => {
                summary.status = (*status).into();
                summary.ended_at = Some(event.at);
            }
            _ => {}
        }
    }

    /// Save one encoded event line. A failure is counted once and this run stops
    /// trying to save (it goes on in memory); it never reaches the run itself.
    fn write_line(&self, target: &mut EventsFile, line: std::io::Result<Vec<u8>>) {
        if target.failed {
            return;
        }
        let Ok(line) = line else {
            self.give_up(target, "event");
            return;
        };
        if target.file.is_none() {
            match self.shared.store.open_events(&self.id) {
                Ok(file) => target.file = Some(file),
                Err(_) => {
                    self.give_up(target, "events file");
                    return;
                }
            }
        }
        if let Some(file) = target.file.as_mut()
            && RunStore::append_line(file, &line).is_err()
        {
            self.give_up(target, "events");
        }
    }

    fn give_up(&self, target: &mut EventsFile, what: &str) {
        target.failed = true;
        self.unsaved.store(true, Ordering::Relaxed);
        self.shared.note_failure(what);
    }

    /// Write a summary snapshot, unless a newer one has been written already.
    fn write_summary(&self, summary: &RunSummary, durable: bool, version: u64) {
        let mut written = lock(&self.summary_io);
        if version <= *written {
            return;
        }
        match self.shared.store.write_summary(summary, durable) {
            Ok(()) => *written = version,
            Err(_) => self.shared.note_failure("summary"),
        }
    }

    /// Map an SDK stream event to a run event and record it.
    fn append_stream(&self, event: StreamEvent) {
        if let Some(kind) = map_stream_event(event) {
            self.append(kind);
        }
    }

    /// The run as the interface reads it: summary, trace, latest record per span,
    /// events without deltas.
    fn detail(&self) -> RunDetail {
        self.ensure_loaded();
        self.refresh_foreign();
        let state = lock(&self.state);
        let mut trace: Option<TraceInfo> = None;
        let mut spans: std::collections::HashMap<&str, &SpanRecord> =
            std::collections::HashMap::new();
        for event in &state.events {
            match &event.kind {
                RunEventKind::SpanStart { span } => {
                    spans.entry(span.id.as_str()).or_insert(span);
                }
                RunEventKind::SpanEnd { span } => {
                    spans.insert(span.id.as_str(), span);
                }
                RunEventKind::TraceStart {
                    trace_id,
                    workflow_name,
                    at,
                } => {
                    trace = Some(TraceInfo {
                        id: trace_id.clone(),
                        workflow_name: workflow_name.clone(),
                        started_at: Some(at.clone()),
                        ended_at: None,
                    });
                }
                RunEventKind::TraceEnd { at, .. } => {
                    if let Some(trace) = trace.as_mut() {
                        trace.ended_at = Some(at.clone());
                    }
                }
                _ => {}
            }
        }
        let mut spans: Vec<SpanRecord> = spans.into_values().cloned().collect();
        spans.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.id.cmp(&b.id)));
        RunDetail {
            run: state.summary.clone(),
            trace,
            spans,
            events: state
                .events
                .iter()
                .filter(|event| !matches!(event.kind, RunEventKind::Delta { .. }))
                .cloned()
                .collect(),
        }
    }

    /// The events after `last`, and whether there will be no more: the run has
    /// ended, or (for a run another process records) it has been silent so long
    /// that the window recording it must be gone.
    fn events_after(&self, last: u64) -> (Vec<RunEvent>, bool) {
        self.ensure_loaded();
        self.refresh_foreign();
        let state = lock(&self.state);
        let from = state.events.partition_point(|event| event.seq <= last);
        let silent = !self.owned
            && !self.shared.writer.load(Ordering::Relaxed)
            && state.summary.status.is_active()
            && state.last_change.elapsed() > FOREIGN_SILENCE;
        (state.events[from..].to_vec(), state.ended || silent)
    }

    /// A run a guardrail refused keeps no copy of what it was refused for: the
    /// task in its summary loses each secret-looking part. Done before the
    /// run's last events, so the summary written with `End` is already clean.
    fn scrub_task(&self) {
        let mut state = lock(&self.state);
        let clean = secrets::redact(&state.summary.task);
        if clean != state.summary.task {
            state.summary.task = clean;
        }
    }

    /// The same for any event of the run that carries one (there is none in a run
    /// the built-in guardrail refused, which never reached a model; this is for
    /// the day an agent records more). The events file is rewritten atomically
    /// when anything changed, and only then.
    fn scrub_events(&self) {
        let events_file = lock(&self.events_file);
        let mut state = lock(&self.state);
        let mut changed = false;
        for event in &mut state.events {
            if let Some(clean) = redacted_event(event) {
                *event = clean;
                changed = true;
            }
        }
        if !changed || self.unsaved.load(Ordering::Relaxed) {
            return;
        }
        let saved: Vec<RunEvent> = state
            .events
            .iter()
            .filter(|event| !matches!(event.kind, RunEventKind::Delta { .. }))
            .cloned()
            .collect();
        drop(state);
        if self.shared.store.rewrite_events(&self.id, &saved).is_err() {
            self.shared.note_failure("events");
        }
        drop(events_file);
    }
}

/// A run's events are recorded into its entry (the recorder's sink).
impl RunSink for RunEntry {
    fn record(&self, kind: RunEventKind) {
        self.append(kind);
    }

    fn record_stream(&self, event: StreamEvent) {
        self.append_stream(event);
    }
}

/// Whether this process may record runs in the directory (see the module docs).
enum Role {
    /// The directory did not exist when the service started, so nobody was
    /// recording; the lock is taken at the first `start`.
    Unclaimed,
    /// This process records runs. The lock is held for as long as the service
    /// lives, which is all it is for (`None` when it could not be tried at all,
    /// as on a file system that refuses locks: then it records anyway, as before
    /// there was a lock).
    Writer { _lock: Option<WriterLock> },
    /// Another process records runs here.
    Reader,
}

struct Inner {
    env: Arc<dyn Env>,
    state: StateRoot,
    keys: KeyStore,
    development: bool,
    dev_step_delay: Duration,
    model_factory: Option<ModelFactory>,
    shared: Arc<Shared>,
    catalog: Catalog,
    /// The runtime this service built and owns (`None` when it was given a
    /// handle to share, or could not build one).
    runtime: Option<Runtime>,
    /// Where runs execute: the owned runtime's handle, or the shared one.
    handle: Option<Handle>,
    /// Every task this service spawned for a run (the run's own task and its
    /// supervisor) that may still be working. Dropping an owned runtime drops
    /// them; a shared runtime is not this service's to shut down, so they are
    /// aborted instead.
    tasks: Mutex<Vec<AbortHandle>>,
    /// Newest first.
    runs: Mutex<Vec<Arc<RunEntry>>>,
    /// Finished runs whose events are in memory, least recently used first.
    cached: Mutex<VecDeque<Arc<RunEntry>>>,
    /// Never held while taking `runs` or an entry's locks.
    role: Mutex<Role>,
    /// When this process last looked at the disk for runs it had not seen.
    last_sync: Mutex<Option<Instant>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Without waiting: this may be dropped anywhere, including inside a
        // runtime, and a run still working is left as `running` on disk, which
        // the next start reads as interrupted.
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        } else {
            // A shared runtime outlives this service. Its runs end here all the
            // same, with nothing more written, exactly as an owned runtime's do.
            let tasks = self.tasks.get_mut().unwrap_or_else(PoisonError::into_inner);
            for task in tasks.drain(..) {
                task.abort();
            }
        }
    }
}

impl Inner {
    /// Remember tasks spawned for a run, forgetting those that have finished.
    fn track(&self, spawned: impl IntoIterator<Item = AbortHandle>) {
        let mut tasks = lock(&self.tasks);
        tasks.retain(|task| !task.is_finished());
        tasks.extend(spawned);
    }

    fn find(&self, id: &str) -> Result<Arc<RunEntry>, Refusal> {
        if !is_run_id(id) {
            return Err(not_found());
        }
        let found = || {
            lock(&self.runs)
                .iter()
                .find(|entry| entry.id == id)
                .cloned()
        };
        // A run another window started after this one last looked is on disk.
        found()
            .or_else(|| {
                self.sync_from_disk(false);
                found()
            })
            .ok_or_else(not_found)
    }

    /// May this process record runs? Takes the writer lock if nobody holds it (and
    /// the state says none is held yet); a process that just took it ends the
    /// records the window that left had not. False when another process records.
    fn claim_writer(&self) -> bool {
        let mut role = lock(&self.role);
        if matches!(*role, Role::Writer { .. }) {
            return true;
        }
        match self.shared.store.try_lock_writer() {
            Ok(Some(held)) => {
                *role = Role::Writer { _lock: Some(held) };
            }
            Ok(None) => {
                *role = Role::Reader;
                return false;
            }
            Err(_) => {
                // The lock could not even be tried; recording still works or
                // fails on its own terms, counted as it always was.
                self.shared.note_failure("writer lock");
                *role = Role::Writer { _lock: None };
            }
        }
        drop(role);
        // What is on disk that this process has not seen, then the records of
        // runs nobody is working on any more.
        self.sync_from_disk(true);
        self.shared.writer.store(true, Ordering::Relaxed);
        self.interrupt_orphans();
        true
    }

    /// A process that holds the writer lock ends every `running` record it did not
    /// start: whoever was working on it has gone.
    fn interrupt_orphans(&self) {
        let entries: Vec<Arc<RunEntry>> = lock(&self.runs).clone();
        for entry in entries {
            if entry.owned {
                continue;
            }
            let mut state = lock(&entry.state);
            if !state.summary.status.is_active() {
                continue;
            }
            state.summary.status = RunStatus::Interrupted;
            state.summary.error = Some(INTERRUPTED.to_owned());
            state.summary.ended_at = Some(state.summary.updated_at);
            state.ended = true;
            state.summary_version += 1;
            let (summary, version) = (state.summary.clone(), state.summary_version);
            drop(state);
            entry.write_summary(&summary, true, version);
            entry.poke();
        }
    }

    /// A process that does not record runs (another does, or none has yet) lists
    /// what is on disk: the runs it has not seen, and the records of the ones it
    /// has, looking at most once a second unless `force`. Never writes.
    fn sync_from_disk(&self, force: bool) {
        if self.shared.writer.load(Ordering::Relaxed) {
            return;
        }
        {
            let mut last = lock(&self.last_sync);
            if !force && last.is_some_and(|at| at.elapsed() < SYNC_INTERVAL) {
                return;
            }
            *last = Some(Instant::now());
        }
        let summaries = self.shared.store.load_summaries(LOADED_RUNS);
        if summaries.is_empty() {
            return;
        }
        let now = (self.shared.clock)();
        let mut runs = lock(&self.runs);
        let mut added = false;
        for summary in summaries {
            match runs.iter().find(|entry| entry.id == summary.id) {
                Some(entry) => entry.adopt_disk_summary(summary),
                None => {
                    runs.push(RunEntry::new(
                        self.shared.clone(),
                        summary,
                        false,
                        false,
                        now,
                    ));
                    added = true;
                }
            }
        }
        if added {
            runs.sort_by(|a, b| {
                let (a_at, b_at) = (a.summary().created_at, b.summary().created_at);
                b_at.total_cmp(&a_at).then_with(|| a.id.cmp(&b.id))
            });
            // The newest runs stay, as at the start; a run of this process always does.
            let mut kept = 0;
            runs.retain(|entry| {
                kept += 1;
                kept <= LOADED_RUNS || entry.owned
            });
        }
    }

    /// Why this process may not record runs, if it may not. A reader looks again
    /// at the lock each time it is asked: the other window may have closed.
    fn recording_refusal(&self) -> Option<String> {
        if !matches!(*lock(&self.role), Role::Reader) {
            return None;
        }
        (!self.claim_writer()).then(|| ANOTHER_WINDOW.to_owned())
    }

    /// Note that a finished run's events are in memory, and let go of the least
    /// recently used ones beyond the sixteenth.
    fn touch(&self, entry: &Arc<RunEntry>) {
        if entry.is_active() {
            return;
        }
        let mut cached = lock(&self.cached);
        cached.retain(|held| !Arc::ptr_eq(held, entry));
        cached.push_back(entry.clone());
        while cached.len() > CACHED_FINISHED_RUNS {
            if let Some(oldest) = cached.pop_front() {
                oldest.evict();
            }
        }
    }

    fn choices(&self) -> Vec<choices::Resolved> {
        choices::resolve_all(self.env.as_ref(), &self.state, &self.keys, self.development)
    }
}

/// The `RunService` over the agent runtime.
#[derive(Clone)]
pub struct CoreService {
    inner: Arc<Inner>,
}

impl CoreService {
    /// Build the service: start its runtime, and load the newest runs from the
    /// store (marking any that were working when Lattice last closed).
    pub fn new(config: CoreConfig) -> Self {
        let runtime = Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("lattice-runs")
            .enable_all()
            .build()
            .ok();
        let handle = runtime.as_ref().map(|runtime| runtime.handle().clone());
        Self::build(config, runtime, handle)
    }

    /// Build the service on a runtime the caller already has (a shell that
    /// runs one runtime for everything), and load the newest runs as [`new`]
    /// does. The service builds no runtime of its own: its runs, their
    /// supervisors and its followers' polls execute on `handle`, whose runtime
    /// must have its time driver enabled (`enable_all`) and must outlive the
    /// runs the caller wants to finish. Dropping the service aborts the tasks it
    /// spawned there, so a run still working is left `running` on disk, which
    /// the next start reads as interrupted, exactly as with [`new`]; the runtime
    /// itself keeps running.
    ///
    /// [`new`]: CoreService::new
    pub fn with_handle(config: CoreConfig, handle: Handle) -> Self {
        Self::build(config, None, Some(handle))
    }

    fn build(config: CoreConfig, runtime: Option<Runtime>, handle: Option<Handle>) -> Self {
        let shared = Arc::new(Shared {
            store: RunStore::new(config.state.runs_dir()),
            clock: config.clock.clone(),
            persist_failures: AtomicU64::new(0),
            writer: AtomicBool::new(false),
        });
        let now = (config.clock)();
        let entries: Vec<Arc<RunEntry>> = shared
            .store
            .load_summaries(LOADED_RUNS)
            .into_iter()
            .map(|summary| RunEntry::new(shared.clone(), summary, false, false, now))
            .collect();
        let keys = KeyStore::new(config.env.clone(), &config.state);
        let inner = Arc::new(Inner {
            keys,
            env: config.env,
            state: config.state,
            development: config.development,
            dev_step_delay: config.dev_step_delay,
            model_factory: config.model_factory,
            catalog: Catalog::new(config.clock.clone()),
            shared,
            runtime,
            handle,
            tasks: Mutex::new(Vec::new()),
            runs: Mutex::new(entries),
            cached: Mutex::new(VecDeque::new()),
            role: Mutex::new(Role::Unclaimed),
            last_sync: Mutex::new(None),
        });
        // A directory that exists may hold records to settle (and a window that
        // is recording in it). One that does not hold nothing, and is made, with
        // its lock, by the first run.
        if inner.state.runs_dir().is_dir() {
            inner.claim_writer();
        }
        Self { inner }
    }

    /// How many runs hold their events file open: none once a run has ended
    /// (for diagnostics and tests).
    pub fn open_event_files(&self) -> usize {
        lock(&self.inner.runs)
            .iter()
            .filter(|entry| lock(&entry.events_file).file.is_some())
            .count()
    }

    /// How many times writing to the store has failed (each is counted once;
    /// the runs it concerned continued in memory).
    pub fn persist_failures(&self) -> u64 {
        self.inner.shared.persist_failures.load(Ordering::Relaxed)
    }

    /// For a run: `(events dropped, deltas dropped)` because a bound was reached.
    pub fn dropped(&self, id: &str) -> Option<(u64, u64)> {
        let entry = self.inner.find(id).ok()?;
        let state = lock(&entry.state);
        Some((state.dropped_events, state.dropped_deltas))
    }

    /// The models a person may pick from, with what a run of each would need.
    fn resolved(&self) -> Vec<choices::Resolved> {
        self.inner.choices()
    }
}

/// The client for a run, and what its errors may say about where it points.
struct Built {
    model: Arc<dyn Model>,
    base_url: Option<String>,
}

impl CoreService {
    fn build_model(&self, chosen: &choices::Resolved) -> Result<Built, Refusal> {
        let inner = &self.inner;
        let Some(endpoint) = &chosen.endpoint else {
            return Ok(Built {
                model: Arc::new(DevModel::new(
                    &inner.catalog.handoff_tool_name(),
                    inner.dev_step_delay,
                )),
                base_url: None,
            });
        };
        let built = models::build(
            endpoint,
            inner.env.as_ref(),
            &inner.keys,
            inner.model_factory.as_ref(),
        )?;
        Ok(Built {
            model: built.model,
            base_url: Some(built.base_url),
        })
    }
}

impl RunService for CoreService {
    fn status(&self) -> ServiceStatus {
        ServiceStatus {
            runtime: format!(
                // The crates of this workspace share one version.
                "lattice-agents {} (a port of openai-agents 0.22.3)",
                env!("CARGO_PKG_VERSION")
            ),
            traces: "this machine".to_owned(),
            refusal: if self.inner.handle.is_none() {
                Some("Lattice could not start its background runtime.".to_owned())
            } else {
                self.inner.recording_refusal()
            },
        }
    }

    fn agents(&self) -> Vec<AgentInfo> {
        self.inner.catalog.agents()
    }

    fn models(&self) -> Vec<ModelChoice> {
        self.resolved()
            .into_iter()
            .map(|resolved| resolved.choice)
            .collect()
    }

    fn runs(&self) -> Vec<RunSummary> {
        // Runs another window recorded since this one last looked.
        self.inner.sync_from_disk(false);
        lock(&self.inner.runs)
            .iter()
            .map(|entry| entry.summary())
            .collect()
    }

    fn run(&self, id: &str) -> Result<RunDetail, Refusal> {
        let entry = self.inner.find(id)?;
        let detail = entry.detail();
        self.inner.touch(&entry);
        Ok(detail)
    }

    fn start(&self, request: StartRun) -> Result<RunSummary, Refusal> {
        let inner = &self.inner;
        let Some(handle) = inner.handle.clone() else {
            return Err(refuse(
                RefusalKind::Unavailable,
                "Lattice could not start its background runtime.",
            ));
        };
        let task = request.task.trim();
        if task.is_empty() {
            return Err(refuse(RefusalKind::Invalid, "Write the task first."));
        }
        if task.chars().count() > MAX_TASK_CHARS {
            return Err(refuse(
                RefusalKind::Invalid,
                "The task is longer than 20,000 characters.",
            ));
        }
        let Some(agent) = inner.catalog.agent(&request.agent) else {
            return Err(refuse(RefusalKind::Invalid, "That agent does not exist."));
        };
        let all = self.resolved();
        let Some(chosen) = all
            .iter()
            .find(|resolved| resolved.choice.id == request.model)
        else {
            return Err(refuse(RefusalKind::Invalid, "That model does not exist."));
        };
        if !chosen.choice.ready {
            let why = chosen
                .choice
                .refusal
                .clone()
                .unwrap_or_else(|| "That model is not ready.".to_owned());
            return Err(refuse(RefusalKind::Invalid, &why));
        }
        // Only now, with a request that can run: the first run makes the
        // directory and takes the right to record in it. A window that may only
        // read says so, whatever the request.
        if !inner.claim_writer() {
            return Err(refuse(RefusalKind::Unavailable, ANOTHER_WINDOW));
        }
        let too_many = || {
            refuse(
                RefusalKind::Conflict,
                &format!(
                    "{MAX_ACTIVE_RUNS} runs are already working. Wait for one to finish, or stop it."
                ),
            )
        };
        if lock(&inner.runs)
            .iter()
            .filter(|entry| entry.is_active())
            .count()
            >= MAX_ACTIVE_RUNS
        {
            return Err(too_many());
        }
        let built = self.build_model(chosen)?;

        let now = (inner.shared.clock)();
        let id = new_run_id();
        let trace_id = format!("trace_{}", Uuid::new_v4().simple());
        let agent_label = inner
            .catalog
            .agents()
            .into_iter()
            .find(|info| info.id == request.agent)
            .map_or_else(|| request.agent.clone(), |info| info.label);
        let summary = RunSummary {
            id: id.clone(),
            task: task.to_owned(),
            agent: request.agent.clone(),
            agent_label: agent_label.clone(),
            model: chosen.choice.id.clone(),
            model_label: chosen.choice.label.clone(),
            locality: chosen.choice.locality,
            status: RunStatus::Running,
            created_at: now,
            updated_at: now,
            ended_at: None,
            trace_id: trace_id.clone(),
            usage: None,
            output: None,
            error: None,
            spans: 0,
        };
        let entry = RunEntry::new(inner.shared.clone(), summary.clone(), true, true, now);
        {
            let mut runs = lock(&inner.runs);
            if runs.iter().filter(|held| held.is_active()).count() >= MAX_ACTIVE_RUNS {
                return Err(too_many());
            }
            runs.insert(0, entry.clone());
        }
        if inner.shared.store.write_summary(&summary, true).is_err() {
            inner.shared.note_failure("summary");
        }

        let intake = Intake::new();
        let mut config = RunConfig::new(built.model);
        config.workflow_name = agent_label;
        config.trace_id = Some(trace_id);
        config.processors = vec![Arc::new(Recorder {
            sink: entry.clone(),
            intake: intake.clone(),
        })];
        config.context = Arc::new(AgentRunContext {
            model_local: chosen.choice.locality == Locality::Local,
            models: all.iter().map(|resolved| resolved.choice.clone()).collect(),
        });
        let RunHandle {
            events,
            control,
            result,
        } = {
            // `run_streamed` spawns on the current runtime.
            let _inside = handle.enter();
            run_streamed(agent, task.to_owned(), config)
        };
        {
            // The run now has a control. A stop that arrived while it was being
            // set up (the run is listed from the moment it is inserted) is
            // applied here, not refused.
            let mut slot = lock(&entry.control);
            if slot.stop_requested {
                control.cancel(CancelMode::Immediate);
            }
            slot.control = Some(control);
        }
        intake.install(events);
        let weak = Arc::downgrade(inner);
        let run_task = result.abort_handle();
        let supervisor = handle.spawn(supervise(entry, intake, result, built.base_url, weak));
        inner.track([run_task, supervisor.abort_handle()]);
        Ok(summary)
    }

    fn stop(&self, id: &str) -> Result<(), Refusal> {
        let entry = self.inner.find(id)?;
        if !entry.is_active() {
            return Err(refuse(RefusalKind::Conflict, "That run is not running."));
        }
        if !entry.owned {
            // Another window's run (or one that died with it): stopping it is
            // that window's to do.
            return Err(refuse(
                RefusalKind::Conflict,
                "That run is not running here. It belongs to another Lattice window.",
            ));
        }
        let mut slot = lock(&entry.control);
        match &slot.control {
            Some(control) => control.cancel(CancelMode::Immediate),
            // Not started yet: remember, and `start` applies it.
            None => slot.stop_requested = true,
        }
        Ok(())
    }

    fn follow(&self, id: &str, after: u64) -> Result<BoxStream<'static, Vec<RunEvent>>, Refusal> {
        let entry = self.inner.find(id)?;
        entry.ensure_loaded();
        self.inner.touch(&entry);
        let receiver = entry.wake.subscribe();
        // For a run another window records: nothing here wakes the follower, so it
        // looks again on the manager's runtime (a sleep needs a timer, which the
        // consumer's executor may not have; awaiting the task that sleeps needs none).
        let poller = self.inner.handle.clone();
        let state = (entry, receiver, after, false, poller);
        Ok(stream::unfold(
            state,
            |(entry, mut receiver, mut last, done, poller)| async move {
                if done {
                    return None;
                }
                loop {
                    // Mark the wake-up seen before reading, so an event that arrives
                    // while reading is not missed.
                    receiver.borrow_and_update();
                    let (batch, ended) = entry.events_after(last);
                    if let Some(newest) = batch.last() {
                        last = newest.seq;
                        let finished = matches!(newest.kind, RunEventKind::End { .. });
                        return Some((batch, (entry, receiver, last, finished, poller)));
                    }
                    if ended {
                        return None;
                    }
                    if entry.is_foreign_active() {
                        let Some(handle) = &poller else { return None };
                        // The sleep is made inside the task: making it needs a timer.
                        let _ = handle
                            .spawn(async { tokio::time::sleep(FOREIGN_POLL).await })
                            .await;
                        continue;
                    }
                    if receiver.changed().await.is_err() {
                        return None;
                    }
                }
            },
        )
        .boxed())
    }
}

/// See a run through: record its stream, then how it ended.
async fn supervise(
    entry: Arc<RunEntry>,
    intake: Arc<Intake>,
    result: JoinHandle<Result<RunResult, RunError>>,
    base_url: Option<String>,
    inner: Weak<Inner>,
) {
    let (outcome, ()) = tokio::join!(result, intake.clone().consume(entry.clone()));
    intake.drain(&*entry);
    let (closing, status) = ending(outcome, base_url.as_deref());
    if status == EndStatus::Refused {
        // Before the run's last events are written, so none of them carries it.
        entry.scrub_task();
    }
    for kind in closing {
        entry.append(kind);
    }
    entry.append(RunEventKind::End { status });
    if status == EndStatus::Refused {
        entry.scrub_events();
    }
    if let Some(inner) = inner.upgrade() {
        inner.touch(&entry);
    }
}

#[cfg(test)]
mod tests;
