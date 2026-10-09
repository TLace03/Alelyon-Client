//! The chat core's ports of the web Lattice's chat (the chat core's spec
//! §2.2, from the native chat's spec §3.3).
//!
//! - [`pyjson`]: Python's JSON both ways: a value that keeps an object's key
//!   order and an integer's digits, `json.dumps` with and without `indent=1`,
//!   and the coercions (`str`, `float`, `int`, truthiness, iteration) that the
//!   shared chat store's reader applies to what it parses.
//! - [`store`]: the shared chat store (`<globals>/lattice_chat/`): its read
//!   path (the index, a thread's turns, the file an id names, the titles), and
//!   its writes, which refuse behind the gate.
//! - [`archive`]: the store's archive (`evicted/`), read and written.
//! - [`archived_by_reader`]: the reader's own Archive, recorded natively
//!   (S21), so the archived list can say `Reader` rather than `Cap`.
//! - [`transcript`]: the transcript-store seam the agent chat records through,
//!   [`transcript::TranscriptStore`].
//! - [`store_gate`]: gate G-WEB: the shared store is read-only here until the
//!   web Lattice's privacy fix is installed.
//! - `memory`: the transcript store in memory, for tests and the `dev-host`
//!   feature only; no shipped build contains it.
//! - [`prompt`], [`think`], [`grounding`], [`echo`]: a plain turn's request,
//!   the reasoning scratchpad removed from its answer (whole and streamed),
//!   the figures no tool backs, and the development echo's reply.
//! - [`vocab`]: the model choices (`auto`, `local`, `cloud`, `endpoint:<id>`,
//!   `dev:echo`) and what each resolves to; Local and Auto are the core's
//!   managed llama.cpp server and nothing else (spec §22, LR1).
//! - [`ChatCore`], with [`jobs`] and [`answer`]: the plain turn, the chat
//!   spec's pipeline (its C9; row C8 here). A send is validated, its thread
//!   claimed and its answer lock taken; Prepare resolves the choice once and
//!   holds it to what the reader was shown (N10), reads the context, runs the
//!   secret tripwire over everything a target off the machine would receive
//!   (N4), and only then writes the question; the answer streams from the
//!   target Prepare resolved and is saved with `append_answer`. Followers get
//!   batches at least 80 ms apart; finished jobs are pruned lazily, never by
//!   a timer; `shutdown` stops every answer and waits for its save.
//!
//! **The production store (row G4).** [`ChatConfig::new`], the shipped
//! composition, records through [`SharedThreadStore`] on `<globals>/lattice_chat`,
//! the web Lattice's own store: its chats are listed ([`ChatService::threads`]),
//! opened with their visible last 400 turns ([`ChatService::open`]) and their
//! archive listed a page at a time ([`ChatCore::archived`]). Its writes pass
//! gate G-WEB, open since row G5 (`store_gate::SHARED_WRITES = true`): a
//! send, regenerate, rename and pin are saved there, as `history.py` writes
//! them, under Python's own lock. With the gate closed (only this crate's
//! tests build such a store) each refuses with the gate's sentence and
//! writes nothing. The memory store is never part of a shipped build.
//!
//! Nothing here prints, logs or reads the environment.

pub mod archive;
pub mod archived_by_reader;
#[cfg(test)]
pub(crate) mod core_tests;
pub mod echo;
pub mod grounding;
#[cfg(test)]
pub(crate) mod interop_tests;
#[cfg(any(test, feature = "dev-host"))]
pub mod memory;
#[cfg(test)]
mod privacy_tests;
pub mod prompt;
pub mod pyjson;
#[cfg(test)]
mod shared_core_tests;
pub mod store;
pub mod store_gate;
pub mod think;
pub mod transcript;
pub mod vocab;
#[cfg(test)]
mod write_parity_tests;

pub mod answer;
pub mod jobs;

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use lattice_protocol::chat::{
    Accepted, ChatChoice, ChatEvent, ChatEventKind, ChatService, ChatTurn, LocalRuntime,
    OpenedThread, RegenerateRequest, Role, RuntimeState, SendRequest, Shown, ThreadList,
    ThreadSummary, is_job_id, is_thread_id,
};
use lattice_protocol::conversation::ArchivedSummary;
use lattice_protocol::{Locality, Refusal, RefusalKind};
use tokio::runtime::Handle;

use crate::clock::{Clock, system_clock};
use crate::env::Env;
use crate::keys::KeyStore;
use crate::llama::ManagedRuntime;
use crate::llama::files::LlamaPaths;
use crate::models::ModelFactory;
use crate::py;
use crate::secrets::looks_like_secret;
use crate::state::StateRoot;

use answer::{Context, Turn};
use archive::ArchiveState;
use jobs::{Job, Jobs};
use store::{IndexRow, IndexState, RECENT_TURNS, SharedThreadStore, StoredTurn, normalize_title};
use transcript::{NewTurn, StoreError, TranscriptStore};
use vocab::{LocalView, Resolution};

/// The one-sentence refusals of the plain chat (the native chat's spec §3.2,
/// Appendix A). Those Appendix A does not word are native and PROVISIONAL.
pub mod refusals {
    pub const RUNTIME: &str = "Lattice could not start its chat runtime.";
    /// PROVISIONAL.
    pub const EMPTY_MESSAGE: &str = "A message needs at least one character.";
    /// PROVISIONAL.
    pub const LONG_MESSAGE: &str = "A message can be at most 32,000 characters.";
    pub const ANSWERING: &str = "An answer is already being written in this chat.";
    pub const ELSEWHERE: &str = "An answer is being written in another Lattice window.";
    pub const MOVED: &str = "Where this model runs has changed. Check the model and send again.";
    pub const NO_QUESTION: &str = "There is no question to answer again.";
    pub const NOT_EDITABLE: &str = "That message can no longer be edited.";
    pub const EMPTY_TITLE: &str = "A title needs at least one character.";
    pub const GONE: &str = "That conversation no longer exists.";
    /// PROVISIONAL.
    pub const NO_JOB: &str = "That answer is not being written here.";
}

/// What the plain chat is built from.
#[derive(Clone)]
pub struct ChatConfig {
    pub state: StateRoot,
    pub env: Arc<dyn Env>,
    /// Offer the development echo, `dev:echo`.
    pub development: bool,
    pub clock: Clock,
    /// Where turns are recorded: the shared store in a shipped build (its
    /// writes open since row G5), the memory store in tests and the
    /// development host.
    pub store: Arc<dyn TranscriptStore>,
    /// The store's folder, shown in the thread list.
    pub store_dir: String,
    /// The managed llama.cpp server (Local and Auto).
    pub local: Arc<dyn ManagedRuntime>,
    /// Where the local runtime's files are (`~/.alelyon`, the models).
    pub paths: LlamaPaths,
    /// Tests give a client their own model (as `CoreConfig::model_factory`).
    pub model_factory: Option<ModelFactory>,
    /// The labs' agents' sessions (`crate::acp`); `None` offers them as not available.
    pub agents: Option<Arc<crate::acp::pool::Pool>>,
    /// The echo's pause between pieces (12 ms).
    pub echo_gap: Duration,
    /// A follower's shortest gap between batches (80 ms).
    pub follow_gap: Duration,
}

impl ChatConfig {
    /// The shipped composition: the shared store under `<globals>`, and
    /// `local`, the process's one managed server.
    pub fn new(state: StateRoot, env: Arc<dyn Env>, local: Arc<dyn ManagedRuntime>) -> Self {
        let dir = state.chat_dir();
        Self {
            store_dir: dir.display().to_string(),
            store: Arc::new(SharedThreadStore::new(dir)),
            paths: LlamaPaths::from_env(env.as_ref()),
            state,
            env,
            development: false,
            clock: system_clock(),
            local,
            model_factory: None,
            agents: None,
            echo_gap: Duration::from_millis(echo::PIECE_GAP_MS),
            follow_gap: jobs::MIN_GAP,
        }
    }
}

/// One page of the archived list (§5.6): newest `evicted_at` first,
/// [`archive::ARCHIVE_PAGE`] at a time, each with why it is there.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchivedPage {
    pub archived: Vec<ArchivedSummary>,
    /// Which page this is, from 0.
    pub page: usize,
    /// How many archived rows there are in all.
    pub total: usize,
    /// `evicted/index.json` cannot be read just now, or only Python can read
    /// it (S20, D7): nothing is listed and nothing will be changed.
    pub archive_unreadable: bool,
}

/// A thread claimed by one send or regenerate while it prepares.
struct Claim {
    claims: Arc<Mutex<HashSet<String>>>,
    thread: String,
}

impl Drop for Claim {
    fn drop(&mut self) {
        self.claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.thread);
    }
}

/// The per-thread answer lock, held while this process writes its answer
/// (the native chat's spec §3.3.5). It is outside the shared store; Python never
/// sees it. Lock files are never removed (the never-delete rule).
pub(crate) struct AnswerLock {
    file: File,
}

impl Drop for AnswerLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

pub(crate) enum Locked {
    Held(AnswerLock),
    /// Another process holds it.
    Elsewhere,
    /// The lock could not be tried (no folder, no locks here): not evidence
    /// of another writer; the answer goes ahead without it.
    Untried,
}

pub(crate) fn answer_lock(dir: &Path, thread: &str) -> Locked {
    if std::fs::create_dir_all(dir).is_err() {
        return Locked::Untried;
    }
    let opened = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join(format!("{thread}.lock")));
    let Ok(file) = opened else {
        return Locked::Untried;
    };
    match file.try_lock() {
        Ok(()) => Locked::Held(AnswerLock { file }),
        Err(std::fs::TryLockError::WouldBlock) => Locked::Elsewhere,
        Err(std::fs::TryLockError::Error(_)) => Locked::Untried,
    }
}

fn refuse(kind: RefusalKind, message: &str) -> Refusal {
    Refusal::new(kind, message)
}

/// N10: the choice moved off the machine, or is called something else now.
pub(crate) fn moved(shown: &Shown, resolved: &Shown) -> bool {
    (shown.locality == Locality::Local && resolved.locality == Locality::Remote)
        || shown.label != resolved.label
}

/// N4: anything a target off the machine would receive that looks like a key.
fn holds_a_secret(question: &str, recent: &[StoredTurn]) -> bool {
    looks_like_secret(question) || recent.iter().any(|turn| looks_like_secret(&turn.text))
}

struct Inner {
    config: ChatConfig,
    handle: Handle,
    keys: KeyStore,
    jobs: Jobs,
    claims: Arc<Mutex<HashSet<String>>>,
    context: Arc<Context>,
}

/// What Prepare hands the start.
struct Prepared {
    thread: ThreadSummary,
    question: Option<ChatTurn>,
    superseded: Vec<String>,
    turn: Turn,
    lock: Option<AnswerLock>,
}

impl Inner {
    fn now(&self) -> f64 {
        (self.config.clock)()
    }

    fn claim(&self, thread: &str) -> Result<Claim, Refusal> {
        let mut claims = self
            .claims
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.jobs.running_for(thread).is_some() || !claims.insert(thread.to_owned()) {
            return Err(refuse(RefusalKind::Conflict, refusals::ANSWERING));
        }
        Ok(Claim {
            claims: self.claims.clone(),
            thread: thread.to_owned(),
        })
    }

    fn lock_for(&self, thread: &str) -> Result<Option<AnswerLock>, Refusal> {
        match answer_lock(&self.config.state.chat_locks_dir(), thread) {
            Locked::Held(lock) => Ok(Some(lock)),
            Locked::Elsewhere => Err(refuse(RefusalKind::Conflict, refusals::ELSEWHERE)),
            Locked::Untried => Ok(None),
        }
    }

    fn view(&self) -> LocalView {
        let mut view = LocalView::read_at(self.config.paths.clone(), &self.config.state);
        view.running = self.config.local.running();
        view.failed = self.config.local.failed();
        view
    }

    /// Prepare (a): resolve the choice once and hold it to what was shown.
    fn resolve(&self, choice: &str, shown: &Shown) -> Result<Resolution, Refusal> {
        let resolution = vocab::resolve(
            choice,
            self.config.env.as_ref(),
            &self.config.state,
            &self.keys,
            self.config.development,
            &self.view(),
        );
        if moved(shown, &resolution.shown) {
            return Err(refuse(RefusalKind::Conflict, refusals::MOVED));
        }
        Ok(resolution)
    }

    /// The thread's row, or why there is none.
    fn row(&self, id: &str) -> Result<IndexRow, Refusal> {
        match self.config.store.list() {
            IndexState::Rows(rows) => rows
                .into_iter()
                .find(|row| row.id == id)
                .ok_or_else(|| refuse(RefusalKind::NotFound, refusals::GONE)),
            IndexState::Absent => Err(refuse(RefusalKind::NotFound, refusals::GONE)),
            IndexState::Unreadable => Err(StoreError::IndexUnreadable.refusal()),
        }
    }

    /// Prepare for a send, under the claim, on the blocking pool: resolve,
    /// read and check (steps (a) to (c), which write nothing), then write (d).
    fn prepare_send(&self, request: &SendRequest) -> Result<Prepared, Refusal> {
        let resolution = self.resolve(&request.choice, &request.shown)?;
        let store = &self.config.store;
        // (b) The context a target off the machine would receive.
        let (context, superseded) = match &request.thread {
            None => (Vec::new(), Vec::new()),
            Some(id) => {
                self.row(id)?;
                match &request.edit_of {
                    None => (store.recent(id), Vec::new()),
                    Some(edit_of) => {
                        let visible = store.load(id);
                        let at = visible
                            .iter()
                            .position(|turn| &turn.id == edit_of)
                            .filter(|at| visible[*at].role() == Role::User)
                            .ok_or_else(|| refuse(RefusalKind::Conflict, refusals::NOT_EDITABLE))?;
                        let kept = &visible[..at];
                        let window = kept[kept.len().saturating_sub(RECENT_TURNS)..].to_vec();
                        let gone = visible[at..].iter().map(|turn| turn.id.clone()).collect();
                        (window, gone)
                    }
                }
            }
        };
        // (c) The tripwire, before anything is written (N4).
        if !resolution.affirmatively_local() && holds_a_secret(&request.text, &context) {
            return Err(refuse(
                RefusalKind::Invalid,
                &answer::secret_sentence(&resolution.shown.label),
            ));
        }
        // (d) Write.
        let (thread, question, lock) = match &request.thread {
            None => {
                let (row, turn) = store
                    .first_message(&request.text, &request.choice)
                    .map_err(StoreError::refusal)?;
                let lock = match answer_lock(&self.config.state.chat_locks_dir(), &row.id) {
                    Locked::Held(lock) => Some(lock),
                    _ => None,
                };
                (row, turn, lock)
            }
            Some(id) => {
                if let Some(edit_of) = &request.edit_of {
                    store.supersede(id, edit_of).map_err(StoreError::refusal)?;
                }
                let turn = store
                    .append(id, NewTurn::user(request.text.clone()))
                    .map_err(StoreError::refusal)?;
                store
                    .pin(id, &request.choice)
                    .map_err(StoreError::refusal)?;
                (self.row(id)?, turn, None)
            }
        };
        let recent = store.recent(&thread.id);
        Ok(Prepared {
            thread: thread.summary(),
            question: Some(question.to_chat_turn()),
            superseded,
            turn: Turn {
                thread: thread.id.clone(),
                question: request.text.clone(),
                recent,
                resolution,
            },
            lock,
        })
    }

    /// Prepare for a regenerate: the last question, answered again.
    fn prepare_regenerate(&self, request: &RegenerateRequest) -> Result<Prepared, Refusal> {
        let resolution = self.resolve(&request.choice, &request.shown)?;
        let store = &self.config.store;
        let id = &request.thread;
        self.row(id)?;
        let visible = store.load(id);
        let at = visible
            .iter()
            .rposition(|turn| turn.role() == Role::User)
            .ok_or_else(|| refuse(RefusalKind::Conflict, refusals::NO_QUESTION))?;
        let question = visible[at].text.clone();
        let upto = &visible[..=at];
        let window = &upto[upto.len().saturating_sub(RECENT_TURNS)..];
        if !resolution.affirmatively_local() && holds_a_secret(&question, window) {
            return Err(refuse(
                RefusalKind::Invalid,
                &answer::secret_sentence(&resolution.shown.label),
            ));
        }
        let superseded: Vec<String> = visible[at + 1..]
            .iter()
            .map(|turn| turn.id.clone())
            .collect();
        if let Some(first) = superseded.first() {
            store.supersede(id, first).map_err(StoreError::refusal)?;
        }
        store
            .pin(id, &request.choice)
            .map_err(StoreError::refusal)?;
        let thread = self.row(id)?;
        Ok(Prepared {
            thread: thread.summary(),
            question: None,
            superseded,
            turn: Turn {
                thread: id.clone(),
                question,
                recent: store.recent(id),
                resolution,
            },
            lock: None,
        })
    }

    /// Register the job and start its supervised answer. A panic in the
    /// answer saves "The answer stopped unexpectedly."; the job always ends,
    /// and its answer lock is released with it.
    fn start(&self, prepared: Prepared, lock: Option<AnswerLock>) -> Accepted {
        let job = Job::new(prepared.turn.thread.clone());
        self.jobs.insert(job.clone());
        let context = self.context.clone();
        let provider = prepared.turn.resolution.provider.clone();
        let thread = prepared.turn.thread.clone();
        let task = self
            .handle
            .spawn(answer::answer(context.clone(), job.clone(), prepared.turn));
        let supervised = job.clone();
        let clock = self.config.clock.clone();
        self.handle.spawn(async move {
            if task.await.is_err() && !recorded(&supervised) {
                let turn = NewTurn {
                    error: answer::STOPPED_UNEXPECTEDLY.into(),
                    ..NewTurn::assistant("", provider)
                };
                answer::record(&context, &supervised, &thread, turn).await;
            }
            // The thread is free once the job is finished, so the lock goes
            // first; then `Done`.
            drop(lock);
            supervised.finish(clock());
        });
        Accepted {
            thread: prepared.thread,
            job: job.id.clone(),
            question: prepared.question,
            superseded: prepared.superseded,
        }
    }
}

/// Has the job recorded its answer (saved or not)?
fn recorded(job: &Job) -> bool {
    job.events_after(0).0.iter().any(|event| {
        matches!(
            event.kind,
            ChatEventKind::Turn { .. } | ChatEventKind::Error { .. }
        )
    })
}

/// Step 1: a send is checked before anything else happens.
fn validate(request: &SendRequest, development: bool) -> Result<(), Refusal> {
    let length = request.text.chars().count();
    if length == 0 || py::strip(&request.text).is_empty() {
        return Err(refuse(RefusalKind::Invalid, refusals::EMPTY_MESSAGE));
    }
    if length > vocab::MAX_MESSAGE_CHARS {
        return Err(refuse(RefusalKind::Invalid, refusals::LONG_MESSAGE));
    }
    validate_choice(&request.choice, development)?;
    if let Some(id) = &request.thread
        && !is_thread_id(id)
    {
        return Err(refuse(RefusalKind::NotFound, refusals::GONE));
    }
    Ok(())
}

fn validate_choice(choice: &str, development: bool) -> Result<(), Refusal> {
    if choice == vocab::CLOUD {
        return Err(refuse(RefusalKind::Invalid, vocab::words::CLOUD_REFUSAL));
    }
    if !vocab::is_valid_choice(choice, development) {
        return Err(refuse(RefusalKind::Invalid, vocab::words::UNKNOWN_CHOICE));
    }
    Ok(())
}

/// The plain chat (the native chat's spec §3.3; row C8): one model stream per
/// turn, no tools, recorded in the transcript store.
///
/// It builds no runtime of its own: everything runs on the handle it is
/// given, and store reads and writes on that runtime's blocking pool.
#[derive(Clone)]
pub struct ChatCore {
    inner: Arc<Inner>,
}

impl ChatCore {
    pub fn new(config: ChatConfig, handle: Handle) -> Self {
        let keys = KeyStore::new(config.env.clone(), &config.state);
        let context = Arc::new(Context {
            store: config.store.clone(),
            local: config.local.clone(),
            env: config.env.clone(),
            keys: keys.clone(),
            factory: config.model_factory.clone(),
            handle: handle.clone(),
            clock: config.clock.clone(),
            echo_gap: config.echo_gap,
            agents: config.agents.clone(),
        });
        Self {
            inner: Arc::new(Inner {
                config,
                handle,
                keys,
                jobs: Jobs::default(),
                claims: Arc::new(Mutex::new(HashSet::new())),
                context,
            }),
        }
    }

    /// Run `work` on the core's runtime's blocking pool.
    fn blocking<T: Send + 'static>(
        &self,
        work: impl FnOnce(&Inner) -> Result<T, Refusal> + Send + 'static,
    ) -> BoxFuture<'static, Result<T, Refusal>> {
        let inner = self.inner.clone();
        let handle = inner.handle.clone();
        async move {
            handle
                .spawn_blocking(move || work(&inner))
                .await
                .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))
        }
        .boxed()
    }

    /// One page of the archive (§5.6, S20): the store's `evicted/index.json`
    /// joined with the reader's own Archive record (S21) for each row's
    /// reason. An archive index Python would set aside (corrupt) lists
    /// nothing; one that cannot be read lists nothing and says so.
    pub fn archived(&self, page: usize) -> BoxFuture<'static, Result<ArchivedPage, Refusal>> {
        self.blocking(move |inner| {
            let empty = |archive_unreadable| ArchivedPage {
                archived: Vec::new(),
                page,
                total: 0,
                archive_unreadable,
            };
            Ok(match inner.config.store.archived() {
                ArchiveState::Absent | ArchiveState::Corrupt => empty(false),
                ArchiveState::Unreadable => empty(true),
                ArchiveState::Rows(rows) => {
                    let by_reader = archived_by_reader::read(&inner.config.state.native_chat_dir());
                    let all = archive::summaries(&rows, &by_reader);
                    ArchivedPage {
                        archived: archive::page(&all, page).to_vec(),
                        page,
                        total: all.len(),
                        archive_unreadable: false,
                    }
                }
            })
        })
    }

    /// Stop every running answer, then wait until each is saved or `wait`
    /// has passed (`LS/jobs.py` `shutdown`). The managed server is its
    /// runtime's to stop: dropping the runtime ends it.
    pub async fn shutdown(&self, wait: Duration) {
        let running = self.inner.jobs.running();
        for job in &running {
            job.stop();
        }
        let all = async move {
            for job in running {
                job.wait_finished().await;
            }
        };
        let _ = self
            .inner
            .handle
            .spawn(async move { tokio::time::timeout(wait, all).await })
            .await;
    }

    /// How many jobs this process keeps (tests).
    #[cfg(test)]
    pub(crate) fn job_count(&self) -> usize {
        self.inner.jobs.len()
    }

    /// A job by id (tests).
    #[cfg(test)]
    pub(crate) fn job(&self, id: &str) -> Option<Arc<Job>> {
        self.inner.jobs.get(id)
    }
}

impl ChatService for ChatCore {
    fn choices(&self) -> BoxFuture<'static, Vec<ChatChoice>> {
        let work = self.blocking(|inner| {
            let choices = vocab::chat_choices(
                inner.config.env.as_ref(),
                &inner.config.state,
                &inner.keys,
                inner.config.development,
                &inner.view(),
            );
            Ok(choices
                .entries
                .into_iter()
                .map(|entry| entry.choice)
                .collect())
        });
        async move { work.await.unwrap_or_default() }.boxed()
    }

    fn threads(&self) -> BoxFuture<'static, Result<ThreadList, Refusal>> {
        self.blocking(|inner| {
            let (threads, index_unreadable) = match inner.config.store.list() {
                IndexState::Absent => (Vec::new(), false),
                IndexState::Rows(rows) => (rows.iter().map(IndexRow::summary).collect(), false),
                IndexState::Unreadable => (Vec::new(), true),
            };
            Ok(ThreadList {
                threads,
                store_dir: inner.config.store_dir.clone(),
                installed: inner.config.state.installed,
                index_unreadable,
            })
        })
    }

    fn open(&self, thread: &str) -> BoxFuture<'static, Result<OpenedThread, Refusal>> {
        let id = thread.to_owned();
        self.inner.jobs.prune(self.inner.now());
        self.blocking(move |inner| {
            if !is_thread_id(&id) {
                return Err(refuse(RefusalKind::NotFound, refusals::GONE));
            }
            let row = inner.row(&id)?;
            let turns = inner
                .config
                .store
                .load(&id)
                .iter()
                .map(StoredTurn::to_chat_turn)
                .collect();
            let job = inner.jobs.running_for(&id).map(|job| job.id.clone());
            let answering_elsewhere = job.is_none()
                && matches!(
                    answer_lock(&inner.config.state.chat_locks_dir(), &id),
                    Locked::Elsewhere
                );
            Ok(OpenedThread {
                thread: row.summary(),
                turns,
                job,
                answering_elsewhere,
            })
        })
    }

    fn send(&self, request: SendRequest) -> BoxFuture<'static, Result<Accepted, Refusal>> {
        if let Err(refusal) = validate(&request, self.inner.config.development) {
            return async move { Err(refusal) }.boxed();
        }
        self.inner.jobs.prune(self.inner.now());
        self.blocking(move |inner| {
            let (claim, lock) = match &request.thread {
                Some(id) => (Some(inner.claim(id)?), inner.lock_for(id)?),
                None => (None, None),
            };
            let mut prepared = inner.prepare_send(&request)?;
            let lock = lock.or_else(|| prepared.lock.take());
            let accepted = inner.start(prepared, lock);
            // The running job holds the thread from here on.
            drop(claim);
            Ok(accepted)
        })
    }

    fn regenerate(
        &self,
        request: RegenerateRequest,
    ) -> BoxFuture<'static, Result<Accepted, Refusal>> {
        if !is_thread_id(&request.thread) {
            return async { Err(refuse(RefusalKind::NotFound, refusals::GONE)) }.boxed();
        }
        if let Err(refusal) = validate_choice(&request.choice, self.inner.config.development) {
            return async move { Err(refusal) }.boxed();
        }
        self.inner.jobs.prune(self.inner.now());
        self.blocking(move |inner| {
            let claim = inner.claim(&request.thread)?;
            let lock = inner.lock_for(&request.thread)?;
            let prepared = inner.prepare_regenerate(&request)?;
            let accepted = inner.start(prepared, lock);
            drop(claim);
            Ok(accepted)
        })
    }

    fn stop(&self, job: &str) -> bool {
        self.inner.jobs.get(job).is_some_and(|job| job.stop())
    }

    fn follow(&self, job: &str, after: u64) -> Result<BoxStream<'static, Vec<ChatEvent>>, Refusal> {
        self.inner.jobs.prune(self.inner.now());
        let found = is_job_id(job)
            .then(|| self.inner.jobs.get(job))
            .flatten()
            .ok_or_else(|| refuse(RefusalKind::NotFound, refusals::NO_JOB))?;
        Ok(jobs::follow(
            found,
            after,
            self.inner.config.follow_gap,
            self.inner.handle.clone(),
        ))
    }

    fn rename(
        &self,
        thread: &str,
        title: &str,
    ) -> BoxFuture<'static, Result<ThreadSummary, Refusal>> {
        let (id, title) = (thread.to_owned(), title.to_owned());
        self.blocking(move |inner| {
            if !is_thread_id(&id) {
                return Err(refuse(RefusalKind::NotFound, refusals::GONE));
            }
            if normalize_title(&title).is_empty() {
                return Err(refuse(RefusalKind::Invalid, refusals::EMPTY_TITLE));
            }
            match inner.config.store.rename(&id, &title) {
                Ok(true) => inner.row(&id).map(|row| row.summary()),
                Ok(false) => Err(refuse(RefusalKind::NotFound, refusals::GONE)),
                Err(error) => Err(error.refusal()),
            }
        })
    }

    fn pin(&self, thread: &str, choice: &str) -> BoxFuture<'static, Result<(), Refusal>> {
        let (id, choice) = (thread.to_owned(), choice.to_owned());
        let development = self.inner.config.development;
        self.blocking(move |inner| {
            if !is_thread_id(&id) {
                return Err(refuse(RefusalKind::NotFound, refusals::GONE));
            }
            if !vocab::is_valid_choice(&choice, development) {
                return Err(refuse(RefusalKind::Invalid, vocab::words::UNKNOWN_CHOICE));
            }
            match inner.config.store.pin(&id, &choice) {
                Ok(true) => Ok(()),
                Ok(false) => Err(refuse(RefusalKind::NotFound, refusals::GONE)),
                Err(error) => Err(error.refusal()),
            }
        })
    }

    fn local_runtime(&self) -> BoxFuture<'static, LocalRuntime> {
        let work = self.blocking(|inner| Ok(vocab::describe_local(&inner.view())));
        async move {
            work.await.unwrap_or_else(|refusal| LocalRuntime {
                state: RuntimeState::Error,
                model: String::new(),
                installed: Vec::new(),
                headline: refusal.message,
                detail: String::new(),
            })
        }
        .boxed()
    }
}
