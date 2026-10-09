//! The state of one run as the interface holds it, and the reducer that folds
//! the runtime's events into it.
//!
//! Invariants (each one has a test):
//! - Events are applied in `seq` order and at most once: an event whose `seq`
//!   is not greater than the last applied one is ignored, so a duplicate batch
//!   or an out-of-order straggler cannot change anything. Gaps are normal,
//!   because live-only `Delta` events take sequence numbers too.
//! - A `SpanEnd` supersedes its `SpanStart` (the SDK fills a span in after it
//!   starts); a `SpanStart` never overwrites a span that has already ended.
//! - Streamed text is kept per agent and concatenated; the complete `Message`
//!   for that agent clears it, because the message replaces what was streamed.
//! - `End` is final: the run stops being followed, its status is the event's,
//!   and spans still open are drawn as ended when the run ended.
//! - Spans keep a parsed start and end so drawing never parses text.

use std::collections::{BTreeMap, HashMap};

use lattice_protocol::{
    RunDetail, RunEvent, RunEventKind, RunStatus, RunSummary, SpanRecord, TraceInfo, Usage,
};

use crate::clock::parse_iso;
use crate::spans::HasSpan;

/// A generous ceiling on streamed text kept per run, so a runaway model cannot
/// grow the interface without bound. The complete `Message` is not limited by it.
pub const LIVE_TEXT_LIMIT: usize = 1 << 20;

/// A span with its timestamps parsed once.
#[derive(Clone, Debug, PartialEq)]
pub struct SpanEntry {
    pub rec: SpanRecord,
    /// Seconds since the epoch.
    pub start: f64,
    pub end: Option<f64>,
}

impl SpanEntry {
    /// `fallback` stands in for a start time that does not parse (it is the time
    /// of the event that carried the span).
    pub fn new(rec: SpanRecord, fallback: f64) -> Self {
        let start = parse_iso(&rec.started_at).unwrap_or(fallback);
        let end = rec
            .ended_at
            .as_deref()
            .and_then(parse_iso)
            .map(|e| e.max(start));
        Self { rec, start, end }
    }
}

impl HasSpan for SpanEntry {
    fn span(&self) -> &SpanRecord {
        &self.rec
    }
}

/// What an applied event changed, so the caller clears only the caches that
/// hold what changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Changes {
    /// A span started or ended: the waterfall, its tree and the graph are stale.
    pub spans: bool,
    /// Streamed or final text changed: the output view is stale.
    pub text: bool,
    /// The status, usage, error or refusal changed: the header is stale.
    pub meta: bool,
    /// The run ended.
    pub ended: bool,
    /// Events that were applied.
    pub applied: u32,
}

impl Changes {
    pub fn any(self) -> bool {
        self.spans || self.text || self.meta || self.ended
    }

    pub fn merge(&mut self, other: Changes) {
        self.spans |= other.spans;
        self.text |= other.text;
        self.meta |= other.meta;
        self.ended |= other.ended;
        self.applied += other.applied;
    }
}

#[derive(Clone, Debug)]
pub struct RunState {
    pub summary: RunSummary,
    pub trace: Option<TraceInfo>,
    /// Sorted by `(order, id)`.
    pub spans: Vec<SpanEntry>,
    index: HashMap<String, usize>,
    pub last_seq: u64,
    /// Streamed text per agent, cleared by that agent's complete message.
    pub live: BTreeMap<String, String>,
    live_len: usize,
    pub messages: Vec<(String, String)>,
    pub current_agent: Option<String>,
    pub output: Option<String>,
    pub usage: Option<Usage>,
    pub turns: Option<u32>,
    pub last_agent: Option<String>,
    pub error: Option<String>,
    /// A guardrail refusal: its name and the message (which never repeats what matched).
    pub refusal: Option<(String, String)>,
    pub status: RunStatus,
    pub ended_at: Option<f64>,
    /// False once the run has ended or is not running.
    pub following: bool,
}

impl RunState {
    pub fn new(summary: RunSummary) -> Self {
        let status = summary.status;
        Self {
            trace: None,
            spans: Vec::new(),
            index: HashMap::new(),
            last_seq: 0,
            live: BTreeMap::new(),
            live_len: 0,
            messages: Vec::new(),
            current_agent: None,
            output: summary.output.clone(),
            usage: summary.usage,
            turns: None,
            last_agent: None,
            error: summary.error.clone(),
            refusal: None,
            status,
            ended_at: summary.ended_at,
            following: status.is_active(),
            summary,
        }
    }

    /// The state of a run as the service last recorded it: its spans, then its
    /// recorded events replayed through the reducer.
    pub fn from_detail(detail: RunDetail) -> Self {
        let mut state = Self::new(detail.run);
        state.trace = detail.trace;
        let at = state.summary.created_at;
        for span in detail.spans {
            state.put_span(SpanEntry::new(span, at), true);
        }
        let mut events = detail.events;
        events.sort_by_key(|e| e.seq);
        for event in &events {
            state.apply(event);
        }
        state
    }

    /// Insert or replace a span. `authoritative` (an end record, or a span from a
    /// detail) replaces what is there; a start record only fills a gap.
    fn put_span(&mut self, entry: SpanEntry, authoritative: bool) -> bool {
        if let Some(&i) = self.index.get(&entry.rec.id) {
            if authoritative && self.spans[i] != entry {
                self.spans[i] = entry;
                return true;
            }
            return false;
        }
        let key = (entry.rec.order, entry.rec.id.clone());
        let appended = self
            .spans
            .last()
            .is_none_or(|last| (last.rec.order, last.rec.id.as_str()) <= (key.0, key.1.as_str()));
        if appended {
            self.index.insert(entry.rec.id.clone(), self.spans.len());
            self.spans.push(entry);
        } else {
            let at = self
                .spans
                .partition_point(|s| (s.rec.order, s.rec.id.as_str()) <= (key.0, key.1.as_str()));
            self.spans.insert(at, entry);
            self.index.clear();
            for (i, s) in self.spans.iter().enumerate() {
                self.index.insert(s.rec.id.clone(), i);
            }
        }
        true
    }

    fn push_live(&mut self, agent: &str, text: &str) -> bool {
        if text.is_empty() || self.live_len >= LIVE_TEXT_LIMIT {
            return false;
        }
        let room = LIVE_TEXT_LIMIT - self.live_len;
        let mut take = text.len().min(room);
        while !text.is_char_boundary(take) {
            take -= 1;
        }
        if take == 0 {
            return false;
        }
        self.live
            .entry(agent.to_string())
            .or_default()
            .push_str(&text[..take]);
        self.live_len += take;
        true
    }

    /// Apply one event. Returns what changed; an ignored event changes nothing.
    pub fn apply(&mut self, event: &RunEvent) -> Changes {
        let mut c = Changes::default();
        if event.seq <= self.last_seq {
            return c;
        }
        self.last_seq = event.seq;
        c.applied = 1;
        match &event.kind {
            RunEventKind::Delta { agent, text } => {
                c.text = self.push_live(agent, text);
            }
            RunEventKind::Agent { name } => {
                self.current_agent = Some(name.clone());
                c.meta = true;
            }
            RunEventKind::Message { agent, text } => {
                if let Some(streamed) = self.live.remove(agent) {
                    self.live_len = self.live_len.saturating_sub(streamed.len());
                }
                self.messages.push((agent.clone(), text.clone()));
                c.text = true;
            }
            RunEventKind::ToolCall { .. }
            | RunEventKind::ToolOutput { .. }
            | RunEventKind::HandoffRequested { .. }
            | RunEventKind::Handoff { .. }
            | RunEventKind::Reasoning { .. } => {}
            RunEventKind::Guardrail { name, message } => {
                self.refusal = Some((name.clone(), message.clone()));
                c.meta = true;
                c.text = true;
            }
            RunEventKind::Result {
                output,
                usage,
                turns,
                last_agent,
            } => {
                self.output = Some(output.clone());
                self.usage = usage.or(self.usage);
                self.turns = Some(*turns);
                self.last_agent = Some(last_agent.clone());
                c.meta = true;
                c.text = true;
            }
            RunEventKind::Error { message } => {
                self.error = Some(message.clone());
                c.meta = true;
                c.text = true;
            }
            RunEventKind::End { status } => {
                self.status = (*status).into();
                self.summary.status = self.status;
                self.following = false;
                self.ended_at = Some(event.at);
                c.meta = true;
                c.ended = true;
                // The complete text has arrived; nothing is still streaming.
                if !self.live.is_empty() {
                    self.live.clear();
                    self.live_len = 0;
                    c.text = true;
                }
            }
            RunEventKind::TraceStart {
                trace_id,
                workflow_name,
                at,
            } => {
                self.trace = Some(TraceInfo {
                    id: trace_id.clone(),
                    workflow_name: workflow_name.clone(),
                    started_at: Some(at.clone()),
                    ended_at: None,
                });
                c.meta = true;
            }
            RunEventKind::TraceEnd { trace_id, at } => {
                match &mut self.trace {
                    Some(trace) if trace.id == *trace_id => trace.ended_at = Some(at.clone()),
                    _ => {
                        self.trace = Some(TraceInfo {
                            id: trace_id.clone(),
                            workflow_name: String::new(),
                            started_at: None,
                            ended_at: Some(at.clone()),
                        });
                    }
                }
                c.meta = true;
            }
            RunEventKind::SpanStart { span } => {
                c.spans = self.put_span(SpanEntry::new(span.clone(), event.at), false);
            }
            RunEventKind::SpanEnd { span } => {
                c.spans = self.put_span(SpanEntry::new(span.clone(), event.at), true);
            }
        }
        c
    }

    pub fn apply_batch(&mut self, events: &[RunEvent]) -> Changes {
        let mut c = Changes::default();
        for event in events {
            c.merge(self.apply(event));
        }
        c
    }

    /// The end of a span for drawing: its own, or, for a span still open in a
    /// run that is no longer running, the moment the run stopped. `None` means
    /// open and running: draw to "now".
    pub fn display_end(&self, entry: &SpanEntry) -> Option<f64> {
        entry.end.or_else(|| {
            if self.status.is_active() {
                None
            } else {
                Some(
                    self.ended_at
                        .unwrap_or(self.summary.updated_at)
                        .max(entry.start),
                )
            }
        })
    }

    /// True when the run is working and this span has not ended.
    pub fn is_open(&self, entry: &SpanEntry) -> bool {
        self.display_end(entry).is_none()
    }

    pub fn span_by_id(&self, id: &str) -> Option<&SpanEntry> {
        self.index.get(id).map(|&i| &self.spans[i])
    }

    pub fn span_index(&self, id: &str) -> Option<usize> {
        self.index.get(id).copied()
    }

    /// The text to show as the run's output: the final output when there is
    /// one, else what has been said so far (the last complete message plus any
    /// text still streaming). The flag says the text is still arriving.
    pub fn output_text(&self) -> Option<(String, bool)> {
        if let Some(output) = &self.output {
            return Some((output.clone(), false));
        }
        let streaming: String = self.live.values().cloned().collect::<Vec<_>>().join("\n\n");
        if !streaming.is_empty() {
            let mut text = self
                .messages
                .last()
                .map(|(_, t)| t.clone())
                .unwrap_or_default();
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            text.push_str(&streaming);
            return Some((text, self.status.is_active()));
        }
        self.messages
            .last()
            .map(|(_, t)| (t.clone(), self.status.is_active()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spans::test_span;
    use lattice_protocol::{EndStatus, Locality};
    use serde_json::json;

    pub(crate) fn summary(status: RunStatus) -> RunSummary {
        RunSummary {
            id: "0123456789abcdef".into(),
            task: "task".into(),
            agent: "lattice-assistant".into(),
            agent_label: "Lattice assistant".into(),
            model: "local".into(),
            model_label: "Local model".into(),
            locality: Locality::Local,
            status,
            created_at: 1000.0,
            updated_at: 1000.0,
            ended_at: None,
            trace_id: "trace_x".into(),
            usage: None,
            output: None,
            error: None,
            spans: 0,
        }
    }

    fn ev(seq: u64, kind: RunEventKind) -> RunEvent {
        RunEvent {
            seq,
            at: 1000.0 + seq as f64,
            kind,
        }
    }

    fn span(id: &str, order: u64, ended: bool) -> SpanRecord {
        let mut s = test_span(id, None, order, json!({"type":"agent","name":id}));
        if !ended {
            s.ended_at = None;
        }
        s
    }

    #[test]
    fn a_span_end_supersedes_its_start() {
        let mut state = RunState::new(summary(RunStatus::Running));
        let started = span("s1", 1, false);
        let mut ended = span("s1", 1, true);
        ended.span_data = json!({"type":"agent","name":"s1","tools":["a"]});
        let c = state.apply(&ev(1, RunEventKind::SpanStart { span: started }));
        assert!(c.spans);
        assert!(state.spans[0].end.is_none());
        let c = state.apply(&ev(
            2,
            RunEventKind::SpanEnd {
                span: ended.clone(),
            },
        ));
        assert!(c.spans);
        assert_eq!(state.spans.len(), 1);
        assert_eq!(state.spans[0].rec, ended);
        assert!(state.spans[0].end.is_some());
    }

    #[test]
    fn a_late_span_start_never_reopens_an_ended_span() {
        let mut state = RunState::new(summary(RunStatus::Running));
        state.apply(&ev(
            5,
            RunEventKind::SpanEnd {
                span: span("s1", 1, true),
            },
        ));
        // The start record arrives after the end (out of order across two batches).
        let c = state.apply(&ev(
            4,
            RunEventKind::SpanStart {
                span: span("s1", 1, false),
            },
        ));
        assert!(!c.any());
        assert!(state.spans[0].end.is_some());
        // Even with a fresh sequence number a start does not overwrite the end.
        let c = state.apply(&ev(
            6,
            RunEventKind::SpanStart {
                span: span("s1", 1, false),
            },
        ));
        assert!(!c.spans);
        assert!(state.spans[0].end.is_some());
    }

    #[test]
    fn deltas_concatenate_per_agent_and_the_complete_message_clears_them() {
        let mut state = RunState::new(summary(RunStatus::Running));
        for (seq, agent, text) in [
            (1, "A", "Hel"),
            (2, "A", "lo "),
            (3, "B", "Other"),
            (4, "A", "world"),
        ] {
            state.apply(&ev(
                seq,
                RunEventKind::Delta {
                    agent: agent.into(),
                    text: text.into(),
                },
            ));
        }
        assert_eq!(state.live["A"], "Hello world");
        assert_eq!(state.live["B"], "Other");
        state.apply(&ev(
            5,
            RunEventKind::Message {
                agent: "A".into(),
                text: "Hello world!".into(),
            },
        ));
        assert!(
            !state.live.contains_key("A"),
            "the message replaces what was streamed"
        );
        assert_eq!(
            state.live["B"], "Other",
            "another agent's stream is untouched"
        );
        assert_eq!(
            state.messages,
            [("A".to_string(), "Hello world!".to_string())]
        );
        // Streaming text is shown after the last complete message.
        let (text, streaming) = state.output_text().unwrap();
        assert_eq!(text, "Hello world!\n\nOther");
        assert!(streaming);
    }

    #[test]
    fn a_duplicate_or_out_of_order_event_changes_nothing() {
        let mut state = RunState::new(summary(RunStatus::Running));
        let batch = [
            ev(
                1,
                RunEventKind::Delta {
                    agent: "A".into(),
                    text: "one ".into(),
                },
            ),
            ev(
                2,
                RunEventKind::Delta {
                    agent: "A".into(),
                    text: "two".into(),
                },
            ),
        ];
        assert_eq!(state.apply_batch(&batch).applied, 2);
        // The same batch again, and an older straggler: ignored.
        assert_eq!(state.apply_batch(&batch).applied, 0);
        assert_eq!(
            state
                .apply(&ev(
                    1,
                    RunEventKind::Delta {
                        agent: "A".into(),
                        text: "STALE".into()
                    }
                ))
                .applied,
            0
        );
        assert_eq!(state.live["A"], "one two");
        assert_eq!(state.last_seq, 2);
        // A gap is fine: live-only deltas take sequence numbers too.
        assert_eq!(
            state
                .apply(&ev(
                    9,
                    RunEventKind::Delta {
                        agent: "A".into(),
                        text: "!".into()
                    }
                ))
                .applied,
            1
        );
        assert_eq!(state.live["A"], "one two!");
    }

    #[test]
    fn end_stops_following_and_fixes_the_status() {
        let mut state = RunState::new(summary(RunStatus::Running));
        assert!(state.following);
        state.apply(&ev(
            1,
            RunEventKind::Delta {
                agent: "A".into(),
                text: "partial".into(),
            },
        ));
        state.apply(&ev(
            2,
            RunEventKind::Result {
                output: "done".into(),
                usage: Some(Usage {
                    requests: 1,
                    input_tokens: 10,
                    output_tokens: 4,
                    total_tokens: 14,
                }),
                turns: 2,
                last_agent: "A".into(),
            },
        ));
        let c = state.apply(&ev(
            3,
            RunEventKind::End {
                status: EndStatus::Completed,
            },
        ));
        assert!(c.ended && c.meta);
        assert!(!state.following);
        assert_eq!(state.status, RunStatus::Completed);
        assert_eq!(state.summary.status, RunStatus::Completed);
        assert_eq!(state.ended_at, Some(1003.0));
        assert!(state.live.is_empty(), "nothing is streaming after the end");
        assert_eq!(state.output_text(), Some(("done".to_string(), false)));
        assert_eq!(state.usage.unwrap().total_tokens, 14);
        assert_eq!(state.turns, Some(2));
        // Events after the end are ignored only by sequence: a higher one would apply,
        // but the service never sends one (End is always last).
        assert_eq!(
            state
                .apply(&ev(
                    2,
                    RunEventKind::Error {
                        message: "late".into()
                    }
                ))
                .applied,
            0
        );
    }

    #[test]
    fn refused_and_failed_runs_keep_their_notices() {
        let mut state = RunState::new(summary(RunStatus::Running));
        state.apply(&ev(
            1,
            RunEventKind::Guardrail {
                name: "secrets_stay_local".into(),
                message: "The task looks like it contains a secret.".into(),
            },
        ));
        state.apply(&ev(
            2,
            RunEventKind::End {
                status: EndStatus::Refused,
            },
        ));
        assert_eq!(state.status, RunStatus::Refused);
        assert_eq!(state.refusal.as_ref().unwrap().0, "secrets_stay_local");

        let mut failed = RunState::new(summary(RunStatus::Running));
        failed.apply(&ev(
            1,
            RunEventKind::Error {
                message: "The model server did not answer.".into(),
            },
        ));
        failed.apply(&ev(
            2,
            RunEventKind::End {
                status: EndStatus::Failed,
            },
        ));
        assert_eq!(
            failed.error.as_deref(),
            Some("The model server did not answer.")
        );
        assert_eq!(failed.status, RunStatus::Failed);
    }

    #[test]
    fn open_spans_of_an_ended_run_are_drawn_as_ended_at_the_end() {
        let mut state = RunState::new(summary(RunStatus::Running));
        state.apply(&ev(
            1,
            RunEventKind::SpanStart {
                span: span("open", 1, false),
            },
        ));
        assert!(state.is_open(&state.spans[0]));
        assert_eq!(state.display_end(&state.spans[0]), None);
        let start = state.spans[0].start;
        state.apply(&RunEvent {
            seq: 2,
            at: start + 5.0,
            kind: RunEventKind::End {
                status: EndStatus::Stopped,
            },
        });
        assert!(!state.is_open(&state.spans[0]));
        assert_eq!(state.display_end(&state.spans[0]), Some(start + 5.0));

        // A clock that ran backwards never gives an end before the start.
        let mut skewed = RunState::new(summary(RunStatus::Running));
        skewed.apply(&ev(
            1,
            RunEventKind::SpanStart {
                span: span("open", 1, false),
            },
        ));
        skewed.apply(&ev(
            2,
            RunEventKind::End {
                status: EndStatus::Stopped,
            },
        )); // at = 1002, long before the span started
        assert_eq!(
            skewed.display_end(&skewed.spans[0]),
            Some(skewed.spans[0].start)
        );
    }

    #[test]
    fn spans_stay_sorted_by_order_even_when_they_arrive_out_of_order() {
        let mut state = RunState::new(summary(RunStatus::Running));
        state.apply(&ev(
            1,
            RunEventKind::SpanStart {
                span: span("c", 3, false),
            },
        ));
        state.apply(&ev(
            2,
            RunEventKind::SpanStart {
                span: span("a", 1, false),
            },
        ));
        state.apply(&ev(
            3,
            RunEventKind::SpanStart {
                span: span("b", 2, false),
            },
        ));
        let ids: Vec<&str> = state.spans.iter().map(|s| s.rec.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        assert_eq!(state.span_by_id("b").unwrap().rec.order, 2);
        assert_eq!(state.span_index("c"), Some(2));
        // The index survived the reinsertion.
        state.apply(&ev(
            4,
            RunEventKind::SpanEnd {
                span: span("a", 1, true),
            },
        ));
        assert!(state.spans[0].end.is_some());
    }

    #[test]
    fn a_detail_replays_into_the_same_state_a_live_follow_reaches() {
        let events = vec![
            ev(
                1,
                RunEventKind::TraceStart {
                    trace_id: "trace_x".into(),
                    workflow_name: "Agent workflow".into(),
                    at: "2026-09-30T05:21:05+00:00".into(),
                },
            ),
            ev(
                2,
                RunEventKind::SpanStart {
                    span: span("s1", 1, false),
                },
            ),
            ev(
                3,
                RunEventKind::Message {
                    agent: "A".into(),
                    text: "hi".into(),
                },
            ),
            ev(
                5,
                RunEventKind::SpanEnd {
                    span: span("s1", 1, true),
                },
            ),
            ev(
                6,
                RunEventKind::Result {
                    output: "hi".into(),
                    usage: None,
                    turns: 1,
                    last_agent: "A".into(),
                },
            ),
            ev(
                7,
                RunEventKind::End {
                    status: EndStatus::Completed,
                },
            ),
        ];
        let mut live = RunState::new(summary(RunStatus::Running));
        live.apply_batch(&events);
        let detail = RunDetail {
            run: summary(RunStatus::Completed),
            trace: None,
            spans: vec![span("s1", 1, true)],
            events: events.clone(),
        };
        let replayed = RunState::from_detail(detail);
        assert_eq!(replayed.spans, live.spans);
        assert_eq!(replayed.output, live.output);
        assert_eq!(replayed.status, live.status);
        assert_eq!(replayed.last_seq, 7);
        assert!(!replayed.following);
        assert_eq!(
            replayed.trace.as_ref().unwrap().workflow_name,
            "Agent workflow"
        );
    }

    #[test]
    fn a_running_detail_is_followed_from_its_last_recorded_event() {
        let events = vec![
            ev(1, RunEventKind::Agent { name: "A".into() }),
            ev(
                4,
                RunEventKind::Message {
                    agent: "A".into(),
                    text: "so far".into(),
                },
            ),
        ];
        let detail = RunDetail {
            run: summary(RunStatus::Running),
            trace: None,
            spans: vec![],
            events,
        };
        let state = RunState::from_detail(detail);
        assert!(state.following);
        assert_eq!(state.last_seq, 4);
        assert_eq!(state.current_agent.as_deref(), Some("A"));
        assert_eq!(state.output_text(), Some(("so far".to_string(), true)));
    }

    #[test]
    fn streamed_text_is_bounded_and_cut_on_a_character_boundary() {
        let mut state = RunState::new(summary(RunStatus::Running));
        let chunk = "é".repeat(LIVE_TEXT_LIMIT / 2 - 1); // 2 bytes each
        assert!(state.push_live("A", &chunk));
        assert!(state.push_live("A", &"é".repeat(10))); // fits partly
        assert!(state.live["A"].len() <= LIVE_TEXT_LIMIT);
        assert!(state.live["A"].is_char_boundary(state.live["A"].len()));
        assert!(!state.push_live("A", "more"), "over the limit: dropped");
        // The complete message frees the room.
        state.apply(&ev(
            1,
            RunEventKind::Message {
                agent: "A".into(),
                text: "x".into(),
            },
        ));
        assert!(state.push_live("A", "again"));
    }
}
