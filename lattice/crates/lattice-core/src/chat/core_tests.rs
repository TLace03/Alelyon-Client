//! The plain turn's pipeline (the native chat's spec §5.1 "jobs", "answer",
//! "stop", "follow" and F4; row C8 of the chat core's spec).
//!
//! Every test runs the real `ChatCore` on its own runtime, with the transcript
//! store in memory, a managed runtime that hands out a loopback address that
//! nothing listens on, and a model factory that records every client it is
//! asked for and answers with a scripted model. Nothing reaches the network,
//! a GPU or the real `globals/`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use lattice_agents::model::{
    GenerationTrace, Model, ModelEvent, ModelRequest, ModelResponse, OutputItem,
};
use lattice_agents::testing::{ScriptedModel, ScriptedStep, assistant_message};
use lattice_agents::{ChatCompletionsConfig, ModelError};
use lattice_protocol::chat::{
    Accepted, ChatEvent, ChatEventKind, ChatService, RegenerateRequest, SendRequest, Shown, Stage,
};
use lattice_protocol::{Locality, RefusalKind};
use serde_json::{Value, json};

use super::memory::MemoryTranscriptStore;
use super::store::{IndexState, uuid_ids};
use super::transcript::TranscriptStore;
use super::vocab::tests::Fixture;
use super::{ChatConfig, ChatCore, answer, jobs, refusals};
use crate::clock::Clock;
use crate::env::MapEnv;
use crate::llama::files::LocalModel;
use crate::llama::{Lease, LlamaError, ManagedEndpoint, ManagedRuntime, Opened};
use crate::models::ModelFactory;

/// The managed runtime, faked: a loopback address nothing listens on.
#[derive(Default)]
pub(crate) struct FakeRuntime {
    pub opens: AtomicUsize,
    pub fail: Mutex<Option<LlamaError>>,
    /// What its `/props` says (`None`: it cannot be read).
    pub props: Mutex<Option<crate::llama::server::Props>>,
}

impl FakeRuntime {
    /// What its `/props` says from now on (`None`: it cannot be read).
    pub(crate) fn set_props(&self, props: Option<crate::llama::server::Props>) {
        *self.props.lock().unwrap() = props;
    }
}

impl ManagedRuntime for FakeRuntime {
    fn props(
        &self,
        _endpoint: &crate::llama::ManagedEndpoint,
    ) -> BoxFuture<'static, Option<crate::llama::server::Props>> {
        let props = self.props.lock().unwrap().clone();
        Box::pin(async move { props })
    }

    fn running(&self) -> Option<String> {
        None
    }

    fn failed(&self) -> Option<String> {
        None
    }

    fn open(&self, model: LocalModel) -> BoxFuture<'static, Result<Opened, LlamaError>> {
        self.opens.fetch_add(1, Ordering::Relaxed);
        let fail = *self.fail.lock().unwrap();
        Box::pin(async move {
            if let Some(error) = fail {
                return Err(error);
            }
            Ok(Opened {
                endpoint: ManagedEndpoint {
                    base_url: "http://127.0.0.1:9".into(),
                    alias: model.name,
                    token: "fake-launch-token".into(),
                    binary_sha256: None,
                    generation: 1,
                },
                lease: Lease::detached(),
            })
        })
    }
}

/// A model that writes `text` one piece at a time, `gap` apart, then ends
/// (or stalls for good, or fails, or panics).
pub(crate) struct Drip {
    pub pieces: Vec<String>,
    pub gap: Duration,
    pub end: DripEnd,
    pub calls: Arc<Mutex<Vec<ModelRequest>>>,
    /// After this many pieces, the stream waits for the gate (the test's
    /// word, not time) before it goes on.
    pub hold: Option<(usize, Arc<tokio::sync::Notify>)>,
}

#[derive(Clone)]
pub(crate) enum DripEnd {
    Done,
    Stall,
    Fail(ModelError),
    Panic,
}

impl Drip {
    pub(crate) fn chars(text: &str, gap: Duration, end: DripEnd) -> Self {
        Self {
            pieces: text.chars().map(String::from).collect(),
            gap,
            end,
            calls: Arc::default(),
            hold: None,
        }
    }

    /// The stream holds after `pieces` pieces until `gate` is notified
    /// (`notify_one` before the wait counts).
    pub(crate) fn held_after(mut self, pieces: usize, gate: Arc<tokio::sync::Notify>) -> Self {
        self.hold = Some((pieces, gate));
        self
    }
}

/// A model that writes one character per `gap` of elapsed time until
/// `length` has passed, then ends; what it wrote is kept in `written`. When
/// the machine's timer is coarser than `gap` (about 15.6 ms on Windows), it
/// catches up with one delta per character due, so the rate holds.
pub(crate) struct Timed {
    pub gap: Duration,
    pub length: Duration,
    pub written: Arc<Mutex<String>>,
}

impl Model for Timed {
    fn name(&self) -> &str {
        "timed"
    }

    fn config_for_trace(&self) -> Value {
        json!({})
    }

    fn stream(&self, _request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        let (gap, length, written) = (self.gap, self.length, self.written.clone());
        let state = (None::<Instant>, 0usize, false);
        futures::stream::unfold(state, move |(start, sent, over)| {
            let written = written.clone();
            async move {
                if over {
                    return None;
                }
                let start = start.unwrap_or_else(Instant::now);
                let due = |start: Instant| {
                    let elapsed = start.elapsed().min(length);
                    (elapsed.as_micros() / gap.as_micros().max(1)) as usize
                };
                if sent >= due(start) {
                    if start.elapsed() >= length {
                        let done = ModelEvent::Done(ModelResponse::default());
                        return Some((Ok(done), (Some(start), sent, true)));
                    }
                    tokio::time::sleep(gap).await;
                }
                if sent >= due(start) {
                    // The timer woke early: nothing is due yet.
                    return Some((
                        Ok(ModelEvent::TextDelta(String::new())),
                        (Some(start), sent, false),
                    ));
                }
                let piece = char::from(b'a' + (sent % 26) as u8).to_string();
                written.lock().unwrap().push_str(&piece);
                Some((
                    Ok(ModelEvent::TextDelta(piece)),
                    (Some(start), sent + 1, false),
                ))
            }
        })
        .boxed()
    }
}

impl Model for Drip {
    fn name(&self) -> &str {
        "drip"
    }

    fn config_for_trace(&self) -> Value {
        json!({})
    }

    fn generation_trace(&self) -> GenerationTrace {
        GenerationTrace::Bare
    }

    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        self.calls.lock().unwrap().push(request);
        let total = self.pieces.len();
        let hold = self.hold.clone();
        let state = (
            VecDeque::from(self.pieces.clone()),
            self.gap,
            self.end.clone(),
            false,
        );
        futures::stream::unfold(state, move |(mut pieces, gap, end, over)| {
            let hold = hold.clone();
            async move {
                if over {
                    return None;
                }
                if let Some((after, gate)) = &hold
                    && total - pieces.len() == *after
                {
                    gate.notified().await;
                }
                if let Some(piece) = pieces.pop_front() {
                    tokio::time::sleep(gap).await;
                    return Some((Ok(ModelEvent::TextDelta(piece)), (pieces, gap, end, false)));
                }
                match end.clone() {
                    DripEnd::Done => Some((
                        Ok(ModelEvent::Done(ModelResponse::default())),
                        (pieces, gap, end, true),
                    )),
                    DripEnd::Stall => {
                        futures::future::pending::<()>().await;
                        None
                    }
                    DripEnd::Fail(error) => Some((Err(error), (pieces, gap, end, true))),
                    DripEnd::Panic => panic!("a model that panics"),
                }
            }
        })
        .boxed()
    }
}

/// A client asked for: its address, model name and locality.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Asked {
    pub base_url: String,
    pub model: String,
    pub local: bool,
}

pub(crate) struct Harness {
    pub fixture: Fixture,
    pub runtime: tokio::runtime::Runtime,
    pub store: Arc<MemoryTranscriptStore>,
    pub local: Arc<FakeRuntime>,
    pub asked: Arc<Mutex<Vec<Asked>>>,
    pub model: Arc<Mutex<Option<Arc<dyn Model>>>>,
    pub core: ChatCore,
}

pub(crate) fn shown(locality: Locality, label: &str) -> Shown {
    Shown {
        locality,
        label: label.into(),
    }
}

pub(crate) fn local_shown() -> Shown {
    shown(Locality::Local, "Local")
}

impl Harness {
    pub(crate) fn new(tag: &str, env: MapEnv) -> Self {
        Self::with(tag, env, |_, _, _| {})
    }

    pub(crate) fn with(
        tag: &str,
        env: MapEnv,
        change: impl FnOnce(&mut ChatConfig, &Fixture, &tokio::runtime::Handle),
    ) -> Self {
        let fixture = Fixture::new(tag, env);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let clock: Clock = crate::clock::system_clock();
        let store = Arc::new(MemoryTranscriptStore::new(uuid_ids(), clock.clone()));
        let local = Arc::new(FakeRuntime::default());
        let asked = Arc::new(Mutex::new(Vec::new()));
        let model: Arc<Mutex<Option<Arc<dyn Model>>>> = Arc::new(Mutex::new(None));
        let (seen, given) = (asked.clone(), model.clone());
        let factory: ModelFactory = Arc::new(move |config: &ChatCompletionsConfig| {
            seen.lock().unwrap().push(Asked {
                base_url: config.base_url.clone(),
                model: config.model.clone(),
                local: config.local,
            });
            given.lock().unwrap().clone()
        });
        let mut config = ChatConfig::new(fixture.state.clone(), fixture.env.clone(), local.clone());
        config.store = store.clone();
        config.store_dir = "memory".into();
        config.development = true;
        config.clock = clock;
        config.model_factory = Some(factory);
        config.echo_gap = Duration::from_millis(1);
        change(&mut config, &fixture, runtime.handle());
        let core = ChatCore::new(config, runtime.handle().clone());
        Self {
            fixture,
            runtime,
            store,
            local,
            asked,
            model,
            core,
        }
    }

    pub(crate) fn answer_with(&self, model: impl Model + 'static) {
        *self.model.lock().unwrap() = Some(Arc::new(model));
    }

    pub(crate) fn text(&self, text: &str) {
        self.answer_with(ScriptedModel::new([ScriptedStep::respond(vec![
            assistant_message(text),
        ])
        .with_tokens(11, 7)]));
    }

    pub(crate) fn send(
        &self,
        thread: Option<&str>,
        text: &str,
        choice: &str,
        shown: Shown,
    ) -> Result<Accepted, lattice_protocol::Refusal> {
        self.runtime.block_on(self.core.send(SendRequest {
            thread: thread.map(str::to_owned),
            text: text.into(),
            choice: choice.into(),
            edit_of: None,
            shown,
        }))
    }

    /// Follow a job from the start to its `Done`; every batch.
    pub(crate) fn batches(&self, job: &str) -> Vec<Vec<ChatEvent>> {
        let stream = self.core.follow(job, 0).unwrap();
        self.runtime.block_on(stream.collect::<Vec<_>>())
    }

    pub(crate) fn events(&self, job: &str) -> Vec<ChatEvent> {
        self.batches(job).into_iter().flatten().collect()
    }

    /// The saved turn's event (`Turn` or `Error`).
    pub(crate) fn outcome(&self, job: &str) -> ChatEventKind {
        self.events(job)
            .into_iter()
            .map(|event| event.kind)
            .find(|kind| {
                matches!(
                    kind,
                    ChatEventKind::Turn { .. } | ChatEventKind::Error { .. }
                )
            })
            .expect("a saved turn")
    }

    /// Everything the memory store holds, as text.
    pub(crate) fn snapshot(&self) -> String {
        let state = self.store.list();
        let mut text = format!("{state:?}");
        if let IndexState::Rows(rows) = &state {
            for row in rows {
                text.push_str(&format!("\n{}: {:?}", row.id, self.store.load(&row.id)));
            }
        }
        text
    }

    pub(crate) fn asked(&self) -> Vec<Asked> {
        self.asked.lock().unwrap().clone()
    }
}

fn error_of(kind: &ChatEventKind) -> (&str, bool) {
    match kind {
        ChatEventKind::Error { message, saved, .. } => (message.as_str(), *saved),
        other => panic!("not an error: {other:?}"),
    }
}

fn turn_of(kind: &ChatEventKind) -> &lattice_protocol::chat::ChatTurn {
    match kind {
        ChatEventKind::Turn { turn, .. } | ChatEventKind::Error { turn, .. } => turn,
        other => panic!("not a turn: {other:?}"),
    }
}

#[test]
fn a_local_turn_streams_saves_and_ends_with_done() {
    let h = Harness::new("core-turn", MapEnv::new());
    h.fixture.install("qwen3-8b");
    h.text("Hello there, it is 1987.");
    let accepted = h
        .send(None, "When was it?", "local", local_shown())
        .unwrap();
    assert_eq!(accepted.question.as_ref().unwrap().text, "When was it?");
    assert_eq!(accepted.thread.pinned, "local");
    let events = h.events(&accepted.job);
    let seqs: Vec<u64> = events.iter().map(|e| e.seq).collect();
    assert!(seqs.windows(2).all(|w| w[0] < w[1]), "{seqs:?}");
    let stages: Vec<Stage> = events
        .iter()
        .filter_map(|e| match &e.kind {
            ChatEventKind::Stage { stage, .. } => Some(*stage),
            _ => None,
        })
        .collect();
    assert_eq!(
        stages,
        [
            Stage::Thinking,
            Stage::Writing,
            Stage::Checking,
            Stage::Done
        ]
    );
    assert!(matches!(events.last().unwrap().kind, ChatEventKind::Done));
    let saved = match &events[events.len() - 2].kind {
        ChatEventKind::Turn { turn, saved: true } => turn.clone(),
        other => panic!("{other:?}"),
    };
    assert_eq!(saved.text, "Hello there, it is 1987.");
    assert_eq!(saved.provider, "llamacpp:qwen3-8b");
    assert_eq!(
        (saved.prompt_tokens, saved.completion_tokens),
        (Some(11), Some(7))
    );
    assert!(
        saved.unsupported.is_empty(),
        "1987 is a year: {:?}",
        saved.unsupported
    );
    assert_eq!(h.local.opens.load(Ordering::Relaxed), 1);
    assert_eq!(
        h.asked(),
        [Asked {
            base_url: "http://127.0.0.1:9/v1".into(),
            model: "qwen3-8b".into(),
            local: true
        }]
    );
    let stored = h.store.load(&accepted.thread.id);
    assert_eq!(stored.len(), 2);
    assert_eq!(stored[1].provider, "llamacpp:qwen3-8b");
}

#[test]
fn the_request_is_the_webs_plain_turn() {
    let h = Harness::new("core-request", MapEnv::new());
    h.fixture.install("m");
    let model = ScriptedModel::new([ScriptedStep::respond(vec![assistant_message("ok")])]);
    let calls = Arc::new(model);
    h.answer_with(SharedModel(calls.clone()));
    let accepted = h
        .send(None, "Question one?", "local", local_shown())
        .unwrap();
    h.events(&accepted.job);
    let request = calls.calls().remove(0);
    // The history Prepare read: the question just written, nothing else yet.
    let history = &h.store.load(&accepted.thread.id)[..1];
    let expected = super::prompt::messages("Question one?", history);
    assert_eq!(request.system, expected.system);
    assert_eq!(request.input, expected.input);
    assert!(request.tools.is_empty());
    assert_eq!(request.settings.temperature, Some(answer::TEMPERATURE));
}

/// A model shared with the test, so its calls can be read.
pub(crate) struct SharedModel(pub Arc<ScriptedModel>);

impl Model for SharedModel {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn config_for_trace(&self) -> Value {
        self.0.config_for_trace()
    }
    fn generation_trace(&self) -> GenerationTrace {
        self.0.generation_trace()
    }
    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        self.0.stream(request)
    }
}

#[test]
fn one_answer_per_thread_and_six_racing_sends_write_one_question() {
    let h = Harness::new("core-one", MapEnv::new());
    h.fixture.install("m");
    h.answer_with(Drip::chars(
        "slow answer",
        Duration::from_millis(40),
        DripEnd::Done,
    ));
    let first = h.send(None, "first", "local", local_shown()).unwrap();
    let id = first.thread.id.clone();
    let again = h
        .send(Some(&id), "second", "local", local_shown())
        .unwrap_err();
    assert_eq!(
        (again.kind, again.message.as_str()),
        (RefusalKind::Conflict, refusals::ANSWERING)
    );
    h.events(&first.job);
    // Six sends at once to the same, idle thread: one is accepted.
    let sends: Vec<_> = (0..6)
        .map(|n| {
            h.core.send(SendRequest {
                thread: Some(id.clone()),
                text: format!("racing {n}"),
                choice: "local".into(),
                edit_of: None,
                shown: local_shown(),
            })
        })
        .collect();
    let outcomes = h.runtime.block_on(futures::future::join_all(sends));
    let accepted: Vec<&Accepted> = outcomes.iter().filter_map(|o| o.as_ref().ok()).collect();
    assert_eq!(accepted.len(), 1, "{outcomes:?}");
    assert!(
        outcomes
            .iter()
            .filter_map(|o| o.as_ref().err())
            .all(|refusal| refusal.kind == RefusalKind::Conflict)
    );
    h.events(&accepted[0].job);
    let users = h
        .store
        .load(&id)
        .iter()
        .filter(|turn| turn.role == "user")
        .count();
    assert_eq!(users, 2, "the first question and exactly one racing one");
}

#[test]
fn another_windows_answer_lock_refuses_a_send_and_shows_on_open() {
    let h = Harness::new("core-lock", MapEnv::new());
    h.fixture.install("m");
    h.text("one");
    let first = h.send(None, "q", "local", local_shown()).unwrap();
    h.events(&first.job);
    let id = first.thread.id.clone();
    let dir = h.fixture.state.chat_locks_dir();
    let other = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join(format!("{id}.lock")))
        .unwrap();
    other.try_lock().unwrap();
    let refusal = h.send(Some(&id), "q2", "local", local_shown()).unwrap_err();
    assert_eq!(
        (refusal.kind, refusal.message.as_str()),
        (RefusalKind::Conflict, refusals::ELSEWHERE)
    );
    let opened = h.runtime.block_on(h.core.open(&id)).unwrap();
    assert!(opened.answering_elsewhere && opened.job.is_none());
    other.unlock().unwrap();
    let opened = h.runtime.block_on(h.core.open(&id)).unwrap();
    assert!(!opened.answering_elsewhere);
    assert_eq!(opened.turns.len(), 2);
}

#[test]
fn finished_jobs_are_pruned_lazily_and_nothing_waits_for_them() {
    let now = Arc::new(Mutex::new(1_000.0));
    let clock_now = now.clone();
    let h = Harness::with("core-prune", MapEnv::new(), move |config, _, _| {
        config.clock = Arc::new(move || *clock_now.lock().unwrap());
    });
    h.fixture.install("m");
    h.text("one");
    let first = h.send(None, "q", "local", local_shown()).unwrap();
    h.events(&first.job);
    assert_eq!(h.core.job_count(), 1);
    let deadline = Instant::now() + Duration::from_secs(5);
    while h.runtime.handle().metrics().num_alive_tasks() > 0 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        h.runtime.handle().metrics().num_alive_tasks(),
        0,
        "a finished job leaves no task behind"
    );
    *now.lock().unwrap() = 1_000.0 + jobs::FINISHED_RETENTION_S + 1.0;
    assert_eq!(h.core.job_count(), 1, "nothing prunes on its own");
    h.text("two");
    let second = h
        .send(Some(&first.thread.id), "q2", "local", local_shown())
        .unwrap();
    assert!(h.core.job(&first.job).is_none(), "pruned by the next send");
    h.events(&second.job);
}

#[test]
fn stop_saves_what_was_written_or_says_nothing_was() {
    let h = Harness::new("core-stop", MapEnv::new());
    h.fixture.install("m");
    h.answer_with(Drip::chars(
        "partial",
        Duration::from_millis(5),
        DripEnd::Stall,
    ));
    let accepted = h.send(None, "q", "local", local_shown()).unwrap();
    let job = h.core.job(&accepted.job).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let written = || {
        job.events_after(0)
            .0
            .iter()
            .filter(|e| matches!(e.kind, ChatEventKind::Delta { .. }))
            .count()
    };
    while written() < 7 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(h.core.stop(&accepted.job));
    let turn = turn_of(&h.outcome(&accepted.job)).clone();
    assert_eq!(
        (turn.text.as_str(), turn.cancelled, turn.error.as_str()),
        ("partial", true, "")
    );
    assert!(
        !h.core.stop(&accepted.job),
        "a finished job is not stopped again"
    );

    h.answer_with(Drip::chars("", Duration::from_millis(5), DripEnd::Stall));
    let accepted = h
        .send(Some(&accepted.thread.id), "q2", "local", local_shown())
        .unwrap();
    std::thread::sleep(Duration::from_millis(50));
    assert!(h.core.stop(&accepted.job));
    let outcome = h.outcome(&accepted.job);
    assert_eq!(error_of(&outcome), (answer::NOTHING_WRITTEN, true));
    assert!(turn_of(&outcome).cancelled);
}

#[test]
fn every_ending_is_saved_as_the_table_says() {
    let h = Harness::new("core-endings", MapEnv::new());
    h.fixture.install("m");
    let cases: Vec<(Box<dyn Model>, &str)> = vec![
        (
            Box::new(Drip::chars("", Duration::ZERO, DripEnd::Done)),
            "empty",
        ),
        (
            Box::new(Drip::chars(
                "",
                Duration::ZERO,
                DripEnd::Fail(ModelError::Status(503)),
            )),
            "error",
        ),
        (
            Box::new(Drip::chars(
                "half",
                Duration::ZERO,
                DripEnd::Fail(ModelError::Timeout),
            )),
            "truncated",
        ),
        (
            Box::new(ScriptedModel::new([ScriptedStep::respond(vec![
                OutputItem::Refusal { text: "no".into() },
            ])])),
            "refused",
        ),
        (
            Box::new(Drip::chars(
                "<think>secret plan</think>The answer",
                Duration::ZERO,
                DripEnd::Done,
            )),
            "scratchpad",
        ),
    ];
    let mut thread: Option<String> = None;
    for (model, case) in cases {
        *h.model.lock().unwrap() = Some(Arc::from(model));
        let accepted = h
            .send(thread.as_deref(), case, "local", local_shown())
            .unwrap();
        thread = Some(accepted.thread.id.clone());
        let events = h.events(&accepted.job);
        let outcome = events
            .iter()
            .map(|e| e.kind.clone())
            .find(|k| matches!(k, ChatEventKind::Turn { .. } | ChatEventKind::Error { .. }))
            .unwrap();
        let shown_text: String = events
            .iter()
            .filter_map(|e| match &e.kind {
                ChatEventKind::Delta { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        match case {
            "empty" => assert_eq!(error_of(&outcome), (answer::EMPTY_ANSWER, true)),
            "error" => assert_eq!(error_of(&outcome), ("The model server answered 503.", true)),
            "truncated" => {
                let turn = turn_of(&outcome);
                assert_eq!(
                    (turn.text.as_str(), turn.truncated, turn.error.as_str()),
                    ("half", true, "")
                );
            }
            "refused" => assert_eq!(error_of(&outcome), ("The model refused: no", true)),
            _ => {
                assert_eq!(turn_of(&outcome).text, "The answer");
                assert!(!shown_text.contains("secret plan"), "{shown_text:?}");
            }
        }
    }
}

#[test]
fn a_thread_archived_while_its_answer_is_written_is_not_brought_back() {
    let h = Harness::new("core-gone", MapEnv::new());
    h.fixture.install("m");
    h.answer_with(Drip::chars(
        "late answer",
        Duration::from_millis(30),
        DripEnd::Done,
    ));
    let accepted = h.send(None, "q", "local", local_shown()).unwrap();
    std::thread::sleep(Duration::from_millis(60));
    h.store.archive(&accepted.thread.id).unwrap();
    let outcome = h.outcome(&accepted.job);
    assert_eq!(error_of(&outcome), (answer::GONE, false));
    let listed = match h.store.list() {
        IndexState::Rows(rows) => rows.iter().any(|row| row.id == accepted.thread.id),
        _ => false,
    };
    assert!(!listed, "the thread does not come back (D8)");
}

#[test]
fn shutdown_stops_every_answer_and_waits_for_its_save() {
    let h = Harness::new("core-shutdown", MapEnv::new());
    h.fixture.install("m");
    h.answer_with(Drip::chars("abc", Duration::from_millis(5), DripEnd::Stall));
    let accepted = h.send(None, "q", "local", local_shown()).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let started = Instant::now();
    h.runtime.block_on(h.core.shutdown(Duration::from_secs(5)));
    assert!(started.elapsed() < Duration::from_secs(5));
    let job = h.core.job(&accepted.job).unwrap();
    assert!(job.finished());
    let stored = h.store.load(&accepted.thread.id);
    let answer = stored.last().unwrap();
    assert_eq!((answer.text.as_str(), answer.cancelled), ("abc", true));
}

/// F4: one character every 5 ms (as fast as this machine's timer gives) for
/// 2 s reaches a follower in at most 27 batches, at least 80 ms apart except
/// the one holding `Done`; what the follower read live is the start of the
/// model's text, and the saved turn is all of it.
#[test]
fn streaming_at_200_tokens_a_second_delivers_at_most_13_batches_a_second() {
    let h = Harness::new("core-f4", MapEnv::new());
    h.fixture.install("m");
    let written = Arc::new(Mutex::new(String::new()));
    h.answer_with(Timed {
        gap: Duration::from_millis(5),
        length: Duration::from_secs(2),
        written: written.clone(),
    });
    let accepted = h.send(None, "q", "local", local_shown()).unwrap();
    let stream = h.core.follow(&accepted.job, 0).unwrap();
    let timed: Vec<(Instant, Vec<ChatEvent>)> = h.runtime.block_on(
        stream
            .map(|batch| (Instant::now(), batch))
            .collect::<Vec<_>>(),
    );
    let joined: String = timed
        .iter()
        .flat_map(|(_, batch)| batch.iter())
        .filter_map(|e| match &e.kind {
            ChatEventKind::Delta { text } => Some(text.clone()),
            _ => None,
        })
        .collect();
    let text = written.lock().unwrap().clone();
    println!(
        "{} characters, {} batches over {:?}",
        text.len(),
        timed.len(),
        timed.last().unwrap().0 - timed[0].0
    );
    assert!(
        text.len() >= 390,
        "about 400 characters in 2 s: {}",
        text.len()
    );
    assert!(timed.len() <= 27, "{} batches", timed.len());
    let last = timed.len() - 1;
    for (index, pair) in timed.windows(2).enumerate() {
        let urgent = index + 1 == last;
        let gap = pair[1].0 - pair[0].0;
        assert!(
            urgent || gap >= Duration::from_millis(78),
            "batch {index}: {gap:?}"
        );
    }
    let saved = turn_of(&h.outcome(&accepted.job)).text.clone();
    assert_eq!(saved, text, "the saved turn is the whole text");
    assert!(
        text.starts_with(&joined),
        "what was read live is the text's start"
    );
    assert!(joined.len() * 2 > text.len(), "most of it was read live");
}

#[test]
fn a_follower_from_zero_replays_everything_and_a_later_one_only_what_is_new() {
    let h = Harness::new("core-follow", MapEnv::new());
    h.fixture.install("m");
    h.text("a short answer");
    let accepted = h.send(None, "q", "local", local_shown()).unwrap();
    let all = h.events(&accepted.job);
    let late = h.runtime.block_on(
        h.core
            .follow(&accepted.job, all[1].seq)
            .unwrap()
            .collect::<Vec<_>>(),
    );
    let late: Vec<u64> = late.into_iter().flatten().map(|e| e.seq).collect();
    assert!(late.iter().all(|seq| *seq > all[1].seq));
    assert_eq!(late.last(), all.last().map(|e| e.seq).as_ref());
    assert!(h.core.follow("not-a-job", 0).is_err());
}

#[test]
fn a_panic_in_the_answer_saves_that_it_stopped() {
    let h = Harness::new("core-panic", MapEnv::new());
    h.fixture.install("m");
    h.answer_with(Drip::chars("", Duration::ZERO, DripEnd::Panic));
    let accepted = h.send(None, "q", "local", local_shown()).unwrap();
    let outcome = h.outcome(&accepted.job);
    assert_eq!(error_of(&outcome), (answer::STOPPED_UNEXPECTEDLY, true));
    assert_eq!(turn_of(&outcome).provider, "llamacpp:m");
}

#[test]
fn the_echo_and_endpoints_are_recorded_with_pythons_provider_names() {
    let h = Harness::new(
        "core-names",
        MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
    );
    h.fixture.registry(
        r#"{"version": 1, "endpoints": [{"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1", "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true}]}"#,
    );
    let echo = h
        .send(
            None,
            "echo  this  question",
            "dev:echo",
            shown(Locality::Local, "Development echo"),
        )
        .unwrap();
    let events = h.events(&echo.job);
    assert!(events.iter().any(|e| matches!(&e.kind, ChatEventKind::Stage { stage: Stage::Writing, detail } if detail == "writing\u{2026}")));
    let turn = turn_of(&h.outcome(&echo.job)).clone();
    assert_eq!(turn.provider, "dev:echo");
    assert_eq!(turn.text, super::echo::echo_text("echo  this  question"));
    assert!(h.asked().is_empty(), "the echo builds no client");
    h.text("hosted answer");
    let hosted = h
        .send(
            None,
            "q",
            "endpoint:hosted",
            shown(Locality::Remote, "Hosted"),
        )
        .unwrap();
    assert_eq!(turn_of(&h.outcome(&hosted.job)).provider, "hosted:big");
    assert_eq!(h.asked()[0].base_url, "https://hosted.example.test/v1");
    assert!(!h.asked()[0].local);
}

#[test]
fn regenerate_answers_the_last_question_again_and_an_edit_replaces_one() {
    let h = Harness::new("core-regen", MapEnv::new());
    h.fixture.install("m");
    h.text("first answer");
    let first = h
        .send(None, "the question", "local", local_shown())
        .unwrap();
    h.events(&first.job);
    let id = first.thread.id.clone();
    let old_answer = h.store.load(&id)[1].id.clone();
    h.text("second answer");
    let again = h
        .runtime
        .block_on(h.core.regenerate(RegenerateRequest {
            thread: id.clone(),
            choice: "local".into(),
            shown: local_shown(),
        }))
        .unwrap();
    assert_eq!(again.superseded, [old_answer]);
    assert!(again.question.is_none());
    h.events(&again.job);
    let visible: Vec<String> = h.store.load(&id).iter().map(|t| t.text.clone()).collect();
    assert_eq!(visible, ["the question", "second answer"]);

    let question = h.store.load(&id)[0].id.clone();
    let answer_id = h.store.load(&id)[1].id.clone();
    h.text("third answer");
    let edited = h
        .runtime
        .block_on(h.core.send(SendRequest {
            thread: Some(id.clone()),
            text: "the better question".into(),
            choice: "local".into(),
            edit_of: Some(question.clone()),
            shown: local_shown(),
        }))
        .unwrap();
    assert_eq!(edited.superseded, [question, answer_id.clone()]);
    h.events(&edited.job);
    let visible: Vec<String> = h.store.load(&id).iter().map(|t| t.text.clone()).collect();
    assert_eq!(visible, ["the better question", "third answer"]);
    let refusal = h
        .runtime
        .block_on(h.core.send(SendRequest {
            thread: Some(id.clone()),
            text: "x".into(),
            choice: "local".into(),
            edit_of: Some(h.store.load(&id)[1].id.clone()),
            shown: local_shown(),
        }))
        .unwrap_err();
    assert_eq!(refusal.message, refusals::NOT_EDITABLE);
    let empty = Harness::new("core-regen-empty", MapEnv::new());
    let refusal = empty
        .runtime
        .block_on(empty.core.regenerate(RegenerateRequest {
            thread: "nope".into(),
            choice: "local".into(),
            shown: local_shown(),
        }))
        .unwrap_err();
    assert_eq!(refusal.kind, RefusalKind::NotFound);
}

/// A shared store whose gate is closed (the test constructor; a shipped
/// build's has been open since row G5) refuses a send with the gate's
/// sentence before anything is written or asked.
/// Mutant: a store built with its writes open.
#[test]
fn a_closed_gate_saves_nothing_and_asks_nothing() {
    let h = Harness::with("core-shared-closed", MapEnv::new(), |config, _, _| {
        let shared =
            super::store::SharedThreadStore::with_gate(config.state.chat_dir(), uuid_ids(), false);
        config.store = Arc::new(shared);
    });
    h.fixture.install("m");
    h.text("never asked");
    let refusal = h.send(None, "q", "local", local_shown()).unwrap_err();
    assert_eq!(refusal.kind, RefusalKind::Unavailable);
    assert_eq!(refusal.message, super::store_gate::REFUSAL);
    assert!(!h.fixture.state.chat_dir().exists(), "nothing was written");
    assert!(h.asked().is_empty());
}

/// Row G5: a shipped build's store (`SharedThreadStore::new`) saves a send
/// and its answer into the shared store, where `history.py` reads them.
/// Mutant: `SHARED_WRITES = false` (the send refuses at the gate).
#[test]
fn a_shipped_build_saves_to_the_shared_store() {
    let h = Harness::with("core-shared", MapEnv::new(), |config, _, _| {
        let shared = super::store::SharedThreadStore::new(config.state.chat_dir());
        config.store = Arc::new(shared);
    });
    h.fixture.install("m");
    h.text("an answer");
    let accepted = h.send(None, "q", "local", local_shown()).unwrap();
    assert!(matches!(
        h.outcome(&accepted.job),
        ChatEventKind::Turn { .. }
    ));
    let store = super::store::SharedThreadStore::new(h.fixture.state.chat_dir());
    let IndexState::Rows(rows) = store.list() else {
        panic!("the shared index lists the chat")
    };
    assert_eq!(rows.len(), 1);
    let texts: Vec<String> = store
        .load(&rows[0].id)
        .iter()
        .map(|turn| turn.text.clone())
        .collect();
    assert_eq!(texts, ["q", "an answer"]);
    assert_eq!(h.asked().len(), 1);
}

#[test]
fn messages_and_choices_are_checked_before_anything_happens() {
    let h = Harness::new("core-validate", MapEnv::new());
    h.fixture.install("m");
    let before = h.snapshot();
    for (text, choice, message) in [
        ("", "local", refusals::EMPTY_MESSAGE),
        ("   \n", "local", refusals::EMPTY_MESSAGE),
        (&"x".repeat(32_001) as &str, "local", refusals::LONG_MESSAGE),
        ("q", "dev:scripted", super::vocab::words::UNKNOWN_CHOICE),
        ("q", "cloud", super::vocab::words::CLOUD_REFUSAL),
    ] {
        let refusal = h.send(None, text, choice, local_shown()).unwrap_err();
        assert_eq!(refusal.message, message, "{choice}");
    }
    assert!(
        h.send(None, &"x".repeat(32_000), "local", local_shown())
            .is_ok()
    );
    let bad_thread = h
        .send(Some("a/b"), "q", "local", local_shown())
        .unwrap_err();
    assert_eq!(bad_thread.kind, RefusalKind::NotFound);
    assert_ne!(h.snapshot(), before, "the one valid send was written");
}

/// N4's backstop: the answer checks the request it built again, because
/// another process may have appended to the thread after Prepare checked it.
/// A secret that reached `recent` that way is not sent to an endpoint.
#[test]
fn the_answer_checks_the_request_again_before_it_leaves() {
    let h = Harness::new(
        "core-backstop",
        MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
    );
    h.fixture.registry(
        r#"{"version": 1, "endpoints": [{"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1", "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true}]}"#,
    );
    h.text("never asked");
    let accepted = h
        .send(
            None,
            "first",
            "endpoint:hosted",
            shown(Locality::Remote, "Hosted"),
        )
        .unwrap();
    h.events(&accepted.job);
    let asked = h.asked().len();
    let id = accepted.thread.id.clone();
    let secret = format!("sk-{}", "q7".repeat(12));
    // What another process appended after Prepare looked.
    h.store
        .append(
            &id,
            super::transcript::NewTurn::user(format!("the key is {secret}")),
        )
        .unwrap();
    let resolution = super::vocab::resolve(
        "endpoint:hosted",
        h.fixture.env.as_ref(),
        &h.fixture.state,
        &h.fixture.keys,
        true,
        &h.fixture.view(),
    );
    let job = super::jobs::Job::new(id.clone());
    let context = h.core.inner.context.clone();
    let turn = answer::Turn {
        thread: id.clone(),
        question: "next".into(),
        recent: h.store.recent(&id),
        resolution,
    };
    h.runtime
        .block_on(answer::answer(context, job.clone(), turn));
    let saved = job
        .events_after(0)
        .0
        .into_iter()
        .map(|event| event.kind)
        .find(|kind| matches!(kind, ChatEventKind::Error { .. }))
        .unwrap();
    assert_eq!(error_of(&saved).0, answer::secret_sentence("Hosted"));
    assert_eq!(h.asked().len(), asked, "nothing was sent");
}
