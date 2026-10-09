//! The shared chat store: `<globals>/lattice_chat/`, as the web Lattice's
//! `history.py` reads and writes it (the chat core's spec §5.2 and S26;
//! the native chat's spec §2.8, rules S1–S14, S16 and deviations D1, D7, D8).
//! The read path is row C2's; the write path, behind gate G-WEB (§5.9), is
//! row G2's.
//!
//! What is read:
//! - `index.json`, the thread rows ([`SharedThreadStore::list`], rule S3),
//!   in three states: [`IndexState::Absent`] (no file: an empty list, and a
//!   first write may create it), [`IndexState::Rows`] (a value that is not a
//!   list reads as no rows, as in Python; each row coerced as
//!   `_read_index_locked` coerces it, a row Python would skip skipped) and
//!   [`IndexState::Unreadable`] (the file is there and cannot be read or
//!   parsed here). Unreadable includes what Python reads and this port cannot
//!   hold (D7: `NaN`, `Infinity`, a lone surrogate, a number past a double, a
//!   count past `i64`, a numeral in another script): every later write must
//!   refuse in that state rather than drop rows it did not see.
//! - `<id>.jsonl`, one turn per line ([`SharedThreadStore::load`], S6 and S7):
//!   UTF-8 (anything else reads as no turns), split with Python's
//!   `splitlines`, each line stripped, blank lines skipped, a line that does
//!   not parse or does not coerce skipped (D1: also one Python reads and this
//!   port cannot hold), superseded turns left out, the last 400 kept. A falsy
//!   id gets a new one on every read, from the injected [`IdSource`], in the
//!   order Python draws them. [`SharedThreadStore::recent`] is the last 8
//!   (S14).
//! - The file a thread id names ([`thread_file_name`], S16): the id with every
//!   character outside `[0-9A-Za-z_-]` dropped, plus `.jsonl`; none when
//!   nothing is left.
//! - The titles ([`auto_title`], [`normalize_title`]), pure functions the
//!   write path uses (S8, S10).
//!
//! S26: every shared file is read with one whole-file read, and the handle is
//! closed before the bytes are parsed; nothing here holds a handle across an
//! await (the API is synchronous and is called from `spawn_blocking`).
//!
//! S15′, the store's lock (row G3): every write, and `load` and `recent`,
//! hold the in-process mutex and then the operating-system lock on
//! `<store>/.store.lock`, which Python's `_store_lock` takes too (it locks
//! byte 0 with `msvcrt.locking`; `File::try_lock` locks the whole file, and
//! the two exclude each other both ways, as the interop tests I1 and I2
//! show). The lock is tried every 20 ms for at most 10 s; after that the call
//! goes on under the mutex alone, as Python's does, and is counted
//! ([`SharedThreadStore::lock_timeouts`]). It is released with an explicit
//! `unlock` before its handle closes. A write opens `.store.lock` with create
//! (Python's format holds the file); a read opens it only if it is there, so
//! reading creates nothing, and a store with no `.store.lock` has never had a
//! Python write that a read could overlap. `list` and `archived` take the
//! mutex only, as Python's `list_threads` does: Python replaces `index.json`
//! and `evicted/index.json` with a retry that a whole-file read never
//! outlasts, but it replaces `<id>.jsonl` (`supersede`) with none, so a read
//! of a thread must not overlap a Python write at all (I10).
//!
//! What is written, past the gate (row G2), each in one hold of the store's
//! lock and with Python's bytes (S2: lines end in the platform's line end;
//! S5: `ensure_ascii`, Python's key order and float repr):
//! - every write first reads the index, and an index that exists and cannot
//!   be read here refuses the write before anything is written (S3, D7);
//! - an append (S8) heals a torn tail with a bare LF, appends the line, and
//!   makes a missing row (titled from the text) or moves the thread's row;
//!   the first message (S9) is that append with its pin, in one index write;
//!   an answer (`append_answer`) is not written at all when the thread is no
//!   longer listed (D8);
//! - rename (S10), pin (S11) and touch (S25) change one listed row; supersede
//!   (S13) rewrites the thread through `<id>.jsonl.tmp` as Python does;
//! - the index (S4′) is sorted by `updated`, the rows past 60 are archived
//!   (S19, `archive`) and those that could not be are kept, `index.tmp` is
//!   written and replaced, and a failed index write takes the batch's archive
//!   moves back;
//! - the reader's Archive (S21) is the eviction's own procedure for one
//!   thread, and Unarchive (S22) follows S22's order: refuse a listed id,
//!   move the transcript back, drop the archive row, list the thread as the
//!   most recently used one, and put everything back if that index write
//!   fails.
//!
//! Nothing is deleted (ND1): a transcript is moved, and Python's temporaries
//! are replaced into place, never removed (ND3). There is no delete.
//!
//! Parity is pinned by `tests/parity/chat/store_parse.json`,
//! `store_index.json` and `titles.json` (reads), and `store_ops.json`,
//! `evicted_ops.json` and `archive_row.json` (writes, replayed by
//! `write_parity_tests`), recorded from `history.py` by
//! `tools/lattice_native_parity.py`.

use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use lattice_protocol::chat::{ChatTurn, Fact, Role, ThreadSummary};

use lattice_protocol::conversation::ArchiveKey;

use super::archive::{self, ArchiveState, Disk, LINE_END, text_mode};
use super::pyjson::{self, CoerceError, PyValue};
use super::store_gate;
use super::transcript::{NewTurn, StoreError, TranscriptStore};
use crate::clock::{Clock, system_clock};
use crate::py;

/// The most threads the list holds (`history.MAX_THREADS`).
pub const MAX_THREADS: usize = 60;
/// The most visible turns a load returns (`history.MAX_TURNS`).
pub const MAX_TURNS: usize = 400;
/// `recent_exchanges(pairs=4)`: the last 8 visible turns.
pub const RECENT_TURNS: usize = 8;
/// `auto_title`'s length, in code points, ellipsis included.
pub const TITLE_CHARS: usize = 48;
/// `rename`'s length, in code points.
pub const RENAME_CHARS: usize = 80;
/// The title of a thread with no question yet.
pub const NEW_THREAD_TITLE: &str = "New thread";
/// The index's file name.
pub const INDEX: &str = "index.json";
/// The file the store's operating-system lock is taken on (S15′).
pub const LOCK_FILE: &str = ".store.lock";
/// How long a call waits for another process's lock before it goes on under
/// the in-process mutex alone (`history.LOCK_WAIT_S`).
pub const LOCK_WAIT: Duration = Duration::from_secs(10);
/// The pause between tries of the lock (Python's `time.sleep(0.02)`).
pub const LOCK_STEP: Duration = Duration::from_millis(20);

/// Where new ids come from: uuid4 hex, first 12 characters (`history._new_id`).
pub type IdSource = Arc<dyn Fn() -> String + Send + Sync>;

/// Python's id source.
pub fn uuid_ids() -> IdSource {
    Arc::new(|| {
        let mut id = uuid::Uuid::new_v4().simple().to_string();
        id.truncate(12);
        id
    })
}

/// One row of `index.json`, as `_read_index_locked` coerces it.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexRow {
    pub id: String,
    pub title: String,
    pub created: f64,
    pub updated: f64,
    pub turns: i64,
    pub pinned_provider: String,
}

impl IndexRow {
    /// The row as the protocol shows it; a negative turn count reads as 0.
    pub fn summary(&self) -> ThreadSummary {
        ThreadSummary {
            id: self.id.clone(),
            title: self.title.clone(),
            created: self.created,
            updated: self.updated,
            turns: u64::try_from(self.turns).unwrap_or(0),
            pinned: self.pinned_provider.clone(),
        }
    }
}

/// What `index.json` holds (rule S3).
#[derive(Clone, Debug, PartialEq)]
pub enum IndexState {
    /// There is no index: an empty list.
    Absent,
    /// The rows, in the file's order (most recently used first, as written).
    Rows(Vec<IndexRow>),
    /// The index is there and cannot be read or held here: nothing may be
    /// written over it (D7).
    Unreadable,
}

/// One turn of a thread, as `_parse_turn` coerces it.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredTurn {
    pub id: String,
    pub ts: f64,
    /// As stored; any value but `"assistant"` is the reader's.
    pub role: String,
    pub text: String,
    pub tools: Vec<String>,
    /// The fact rows that are objects, as stored.
    pub facts: Vec<PyValue>,
    pub unsupported: Vec<String>,
    pub provider: String,
    pub error: String,
    pub constrained: bool,
    pub truncated: bool,
    pub cancelled: bool,
    /// `None`: not reported (UNMEASURED), never 0.
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub superseded: bool,
}

impl StoredTurn {
    /// Who wrote it: `"assistant"` is the model, anything else the reader.
    pub fn role(&self) -> Role {
        if self.role == "assistant" {
            Role::Assistant
        } else {
            Role::User
        }
    }

    /// The turn as the protocol shows it. A fact's `label`, `rendered`,
    /// `as_of` and `note` are `""` when missing or null and their JSON text
    /// when not a string (D2). A negative token count is not a count: `None`.
    pub fn to_chat_turn(&self) -> ChatTurn {
        let field = |fact: &PyValue, key: &str| match fact.get(key) {
            None | Some(PyValue::Null) => String::new(),
            Some(PyValue::Str(text)) => text.clone(),
            Some(other) => pyjson::dumps(other),
        };
        ChatTurn {
            id: self.id.clone(),
            ts: self.ts,
            role: self.role(),
            text: self.text.clone(),
            tools: self.tools.clone(),
            facts: self
                .facts
                .iter()
                .map(|fact| Fact {
                    label: field(fact, "label"),
                    rendered: field(fact, "rendered"),
                    as_of: field(fact, "as_of"),
                    note: field(fact, "note"),
                })
                .collect(),
            unsupported: self.unsupported.clone(),
            provider: self.provider.clone(),
            error: self.error.clone(),
            constrained: self.constrained,
            truncated: self.truncated,
            cancelled: self.cancelled,
            prompt_tokens: self.prompt_tokens.and_then(|n| u64::try_from(n).ok()),
            completion_tokens: self.completion_tokens.and_then(|n| u64::try_from(n).ok()),
        }
    }
}

/// The outcome of coercing one record's fields in Python's order: Python
/// raises at the first field that fails; a record with any field Python
/// rejects is one Python skips, whatever this port could or could not hold
/// elsewhere in it.
#[derive(Default)]
struct Fields {
    error: Option<CoerceError>,
}

impl Fields {
    fn take<T: Default>(&mut self, result: Result<T, CoerceError>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => {
                if self.error != Some(CoerceError::Python) {
                    self.error = Some(error);
                }
                T::default()
            }
        }
    }

    fn finish<T>(self, value: T) -> Result<T, CoerceError> {
        match self.error {
            Some(error) => Err(error),
            None => Ok(value),
        }
    }
}

/// `d.get(key, default)`.
fn get<'a>(record: &'a PyValue, key: &str, default: &'a PyValue) -> &'a PyValue {
    record.get(key).unwrap_or(default)
}

/// `_parse_turn(record)`: the turn, [`CoerceError::Python`] where Python
/// returns `None`, [`CoerceError::Native`] where Python keeps a turn this port
/// cannot hold (D1). A falsy id takes a new one from `ids` first, as Python
/// does, even when a later field fails.
pub fn parse_turn(record: &PyValue, ids: &IdSource) -> Result<StoredTurn, CoerceError> {
    if !record.is_object() {
        // `d.get` on anything but a dict raises.
        return Err(CoerceError::Python);
    }
    let null = PyValue::Null;
    let id = match record.get("id") {
        Some(value) if pyjson::py_truthy(value) => pyjson::py_str(value),
        _ => ids(),
    };
    let mut fields = Fields::default();
    let ts = fields.take(pyjson::py_float(get(record, "ts", &PyValue::Float(0.0))));
    let role = pyjson::py_str(get(record, "role", &PyValue::str("user")));
    let empty = PyValue::str("");
    let text = pyjson::py_str(get(record, "text", &empty));
    let tools = fields.take(pyjson::py_iter_strs(get(record, "tools", &null)));
    let facts = fields.take(pyjson::py_iter_objects(get(record, "facts", &null)));
    let unsupported = fields.take(pyjson::py_iter_strs(get(record, "unsupported", &null)));
    let provider = pyjson::py_str(get(record, "provider", &empty));
    let error = pyjson::py_str(get(record, "error", &empty));
    let flag = |key: &str| pyjson::py_truthy(get(record, key, &PyValue::Bool(false)));
    let constrained = flag("constrained");
    let truncated = flag("truncated");
    let cancelled = flag("cancelled");
    let prompt_tokens = fields.take(pyjson::py_count(get(record, "prompt_tokens", &null)));
    let completion_tokens = fields.take(pyjson::py_count(get(record, "completion_tokens", &null)));
    // Only a literal true supersedes.
    let superseded = record.get("superseded") == Some(&PyValue::Bool(true));
    fields.finish(StoredTurn {
        id,
        ts,
        role,
        text,
        tools,
        facts,
        unsupported,
        provider,
        error,
        constrained,
        truncated,
        cancelled,
        prompt_tokens,
        completion_tokens,
        superseded,
    })
}

/// `load_thread` over a thread file's bytes: the visible turns, the last
/// [`MAX_TURNS`].
pub fn parse_thread(bytes: &[u8], ids: &IdSource) -> Vec<StoredTurn> {
    // `read_text(encoding="utf-8")`: anything that is not UTF-8 is no turns.
    // A byte order mark stays, so the first line does not parse (as in Python).
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Vec::new();
    };
    let mut turns = Vec::new();
    for line in py::splitlines(text) {
        let line = py::strip(line);
        if line.is_empty() {
            continue;
        }
        let Ok(record) = pyjson::loads(line) else {
            continue; // a torn line must not hide the rest
        };
        if let Ok(turn) = parse_turn(&record, ids)
            && !turn.superseded
        {
            turns.push(turn);
        }
    }
    let skip = turns.len().saturating_sub(MAX_TURNS);
    turns.split_off(skip)
}

/// `_read_index_locked`'s coercion of one row.
pub fn parse_index_row(row: &PyValue) -> Result<IndexRow, CoerceError> {
    // `d["id"]` raises for a missing key and for anything but a dict.
    let Some(id) = row.get("id") else {
        return Err(CoerceError::Python);
    };
    let mut fields = Fields::default();
    let empty = PyValue::str("");
    let zero = PyValue::Float(0.0);
    let id = pyjson::py_str(id);
    let title = pyjson::py_str(get(row, "title", &empty));
    let created = fields.take(pyjson::py_float(get(row, "created", &zero)));
    let updated = fields.take(pyjson::py_float(get(row, "updated", &zero)));
    let turns = fields.take(pyjson::py_int(get(row, "turns", &PyValue::int(0))));
    let pinned_provider = pyjson::py_str(get(row, "pinned_provider", &empty));
    fields.finish(IndexRow {
        id,
        title,
        created,
        updated,
        turns,
        pinned_provider,
    })
}

/// `_read_index_locked` over the bytes of an index that exists.
pub fn parse_index(bytes: &[u8]) -> IndexState {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return IndexState::Unreadable;
    };
    let items = match pyjson::loads(text) {
        Ok(PyValue::List(items)) => items,
        // Read, and not a list: Python reads it as no rows.
        Ok(_) => return IndexState::Rows(Vec::new()),
        // Python's json.loads refuses it (Python then reads no rows and its
        // next write replaces the file), or only native cannot hold it (D7).
        // Either way nothing will be written over it here.
        Err(_) => return IndexState::Unreadable,
    };
    let mut rows = Vec::new();
    for item in &items {
        match parse_index_row(item) {
            Ok(row) => rows.push(row),
            Err(CoerceError::Python) => {}
            // Python keeps this row and this port cannot hold it (D7).
            Err(CoerceError::Native) => return IndexState::Unreadable,
        }
    }
    IndexState::Rows(rows)
}

/// The file name a thread id names: the id with every character outside
/// `[0-9A-Za-z_-]` dropped, plus `.jsonl` (`_thread_path`); `None` when
/// nothing is left.
pub fn thread_file_name(id: &str) -> Option<String> {
    let safe: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    (!safe.is_empty()).then(|| format!("{safe}.jsonl"))
}

/// `" ".join(text.split())`.
fn collapse_whitespace(text: &str) -> String {
    text.split(py::is_space)
        .filter(|word| !word.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// `auto_title(text)`: the question with its whitespace collapsed, cut to
/// [`TITLE_CHARS`] code points with an ellipsis; "New thread" when blank.
pub fn auto_title(text: &str) -> String {
    let collapsed = collapse_whitespace(text);
    if collapsed.is_empty() {
        return NEW_THREAD_TITLE.to_owned();
    }
    if collapsed.chars().count() <= TITLE_CHARS {
        return collapsed;
    }
    let head: String = collapsed.chars().take(TITLE_CHARS - 1).collect();
    format!("{}\u{2026}", head.trim_end_matches(py::is_space))
}

/// `rename`'s normalisation: whitespace collapsed, cut to [`RENAME_CHARS`]
/// code points. `""` means the rename is refused (S10).
pub fn normalize_title(title: &str) -> String {
    collapse_whitespace(title)
        .chars()
        .take(RENAME_CHARS)
        .collect()
}

/// The store's lock, held (S15′): the in-process mutex and, when it could be
/// had, the operating-system lock on [`LOCK_FILE`].
pub(crate) struct StoreLock<'a> {
    /// The lock file, locked; `None` when the call goes on under the mutex
    /// alone.
    file: Option<File>,
    /// Released after the file: fields drop after `drop` has run.
    _mutex: MutexGuard<'a, ()>,
}

impl StoreLock<'_> {
    /// Whether the operating-system lock is held.
    #[cfg(test)]
    pub(crate) fn os_locked(&self) -> bool {
        self.file.is_some()
    }
}

impl Drop for StoreLock<'_> {
    fn drop(&mut self) {
        // An explicit unlock, so the release does not wait on the handle's
        // close; then the handle closes, then the mutex is released.
        if let Some(file) = self.file.take() {
            let _ = file.unlock();
        }
    }
}

/// The shared store at `<globals>/lattice_chat/`.
///
/// Its writes go through gate G-WEB (§5.9): while `store_gate::SHARED_WRITES`
/// is `false` each refuses with [`StoreError::SharedWritesOff`] and writes
/// nothing. Row G5 opened the gate (2026-10-05); only this crate's tests
/// build a store with it closed.
pub struct SharedThreadStore {
    dir: PathBuf,
    ids: IdSource,
    /// `history._now`: a turn's default time, `touch`, `evicted_at`, and the
    /// time in a set-aside archive index's name.
    clock: Clock,
    mutex: Mutex<()>,
    /// Whether writes may pass: `store_gate::SHARED_WRITES`, except in this
    /// crate's tests.
    writes: bool,
    /// How long the operating-system lock is waited for ([`LOCK_WAIT`]; a
    /// unit test shortens it).
    lock_wait: Duration,
    /// Calls that went on without the operating-system lock (S15′).
    lock_timeouts: AtomicU64,
    /// Replaces a unit test makes fail, as the parity generator patches
    /// Python's `_replace`.
    #[cfg(test)]
    faults: Mutex<Vec<archive::Fault>>,
}

impl SharedThreadStore {
    /// The store in `dir`, with Python's id source.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self::with_ids(dir, uuid_ids())
    }

    /// The store in `dir`, drawing new ids from `ids`.
    pub fn with_ids(dir: impl Into<PathBuf>, ids: IdSource) -> Self {
        Self {
            dir: dir.into(),
            ids,
            clock: system_clock(),
            mutex: Mutex::new(()),
            writes: store_gate::SHARED_WRITES,
            lock_wait: LOCK_WAIT,
            lock_timeouts: AtomicU64::new(0),
            #[cfg(test)]
            faults: Mutex::new(Vec::new()),
        }
    }

    /// A store whose gate is given, so a unit test can reach either side of
    /// it. There is no production switch.
    #[cfg(test)]
    pub(crate) fn with_gate(dir: impl Into<PathBuf>, ids: IdSource, writes: bool) -> Self {
        Self {
            writes,
            ..Self::with_ids(dir, ids)
        }
    }

    /// The same store reading `clock` (tests: the parity replay feeds it the
    /// values Python was given). It does not touch the gate.
    #[cfg(test)]
    pub(crate) fn clocked(mut self, clock: Clock) -> Self {
        self.clock = clock;
        self
    }

    /// The same store waiting at most `wait` for the operating-system lock
    /// (tests: a lock held for good must not cost a test 10 s).
    #[cfg(test)]
    pub(crate) fn lock_waiting(mut self, wait: Duration) -> Self {
        self.lock_wait = wait;
        self
    }

    /// Hold the store's lock, as a write holds it, until the guard drops
    /// (tests: I2 holds it while Python tries).
    #[cfg(test)]
    pub(crate) fn hold_lock(&self) -> StoreLock<'_> {
        self.lock(true)
    }

    /// Make every later replace that a rule names fail (tests only).
    #[cfg(test)]
    pub(crate) fn set_faults(&self, faults: Vec<archive::Fault>) {
        *self
            .faults
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = faults;
    }

    /// Whether this store's writes pass the gate.
    pub fn writes_open(&self) -> bool {
        self.writes
    }

    /// The store's folder.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// How many calls went on without the operating-system lock, because
    /// another process held it for the whole wait (S15′).
    pub fn lock_timeouts(&self) -> u64 {
        self.lock_timeouts.load(Ordering::Relaxed)
    }

    fn guard(&self) -> MutexGuard<'_, ()> {
        // A panic while reading poisons nothing worth refusing for.
        self.mutex
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// S15′: the in-process mutex, then the operating-system lock on
    /// [`LOCK_FILE`], tried every [`LOCK_STEP`] for at most the wait. `write`
    /// makes the store's folder and the lock file, as Python's `_store_lock`
    /// does; a read uses the file only if it is there. A file that cannot be
    /// opened leaves the mutex alone holding, which is all any write had
    /// before the lock existed; a wait that runs out is counted.
    fn lock(&self, write: bool) -> StoreLock<'_> {
        let mutex = self.guard();
        let path = self.dir.join(LOCK_FILE);
        let opened = if write {
            fs::create_dir_all(&self.dir).and_then(|()| {
                OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(&path)
            })
        } else {
            OpenOptions::new().read(true).write(true).open(&path)
        };
        let Ok(file) = opened else {
            return StoreLock {
                file: None,
                _mutex: mutex,
            };
        };
        let deadline = Instant::now() + self.lock_wait;
        loop {
            // Python tries again after any OSError until its deadline; so
            // does this, whatever `try_lock` reports.
            if file.try_lock().is_ok() {
                return StoreLock {
                    file: Some(file),
                    _mutex: mutex,
                };
            }
            if Instant::now() >= deadline {
                self.lock_timeouts.fetch_add(1, Ordering::Relaxed);
                return StoreLock {
                    file: None,
                    _mutex: mutex,
                };
            }
            std::thread::sleep(LOCK_STEP);
        }
    }

    /// The thread rows (S3).
    pub fn list(&self) -> IndexState {
        let _guard = self.guard();
        self.read_index()
    }

    /// A thread's visible turns, the last [`MAX_TURNS`] (S7). An id that names
    /// no file, a missing file and a file that cannot be read are no turns.
    pub fn load(&self, id: &str) -> Vec<StoredTurn> {
        let Some(name) = thread_file_name(id) else {
            return Vec::new();
        };
        // S26: one whole-file read under the store's lock; the handle and the
        // lock are gone before the bytes are parsed.
        let bytes = {
            let _lock = self.lock(false);
            match fs::read(self.dir.join(name)) {
                Ok(bytes) => bytes,
                Err(_) => return Vec::new(),
            }
        };
        parse_thread(&bytes, &self.ids)
    }

    /// The last [`RECENT_TURNS`] visible turns (S14).
    pub fn recent(&self, id: &str) -> Vec<StoredTurn> {
        let mut turns = self.load(id);
        let skip = turns.len().saturating_sub(RECENT_TURNS);
        turns.split_off(skip)
    }

    /// The archive's rows (S20).
    pub fn archived(&self) -> ArchiveState {
        let _guard = self.guard();
        archive::read_evicted(&self.dir)
    }
}

impl TranscriptStore for SharedThreadStore {
    fn list(&self) -> IndexState {
        SharedThreadStore::list(self)
    }

    fn archived(&self) -> ArchiveState {
        SharedThreadStore::archived(self)
    }

    fn load(&self, id: &str) -> Vec<StoredTurn> {
        SharedThreadStore::load(self, id)
    }

    fn recent(&self, id: &str) -> Vec<StoredTurn> {
        SharedThreadStore::recent(self, id)
    }

    fn first_message(&self, text: &str, pin: &str) -> Result<(IndexRow, StoredTurn), StoreError> {
        self.gate()?;
        let id = (self.ids)();
        let (turn, row) = self.append_turn(&id, NewTurn::user(text), false, Some(pin))?;
        Ok((row.ok_or(StoreError::NotSaved)?, turn))
    }

    fn append(&self, id: &str, turn: NewTurn) -> Result<StoredTurn, StoreError> {
        self.append_turn(id, turn, false, None)
            .map(|(turn, _)| turn)
    }

    fn append_answer(&self, id: &str, turn: NewTurn) -> Result<StoredTurn, StoreError> {
        self.append_turn(id, turn, true, None).map(|(turn, _)| turn)
    }

    fn rename(&self, id: &str, title: &str) -> Result<bool, StoreError> {
        self.gate()?;
        let title = normalize_title(title);
        if title.is_empty() {
            return Ok(false);
        }
        self.change_row(id, |row| row.title = title)
    }

    fn pin(&self, id: &str, pin: &str) -> Result<bool, StoreError> {
        self.gate()?;
        let pin = py::strip(pin).to_owned();
        // `updated` is never touched (S11).
        self.change_row(id, |row| row.pinned_provider = pin)
    }

    fn supersede(&self, id: &str, from: &str) -> Result<usize, StoreError> {
        self.gate()?;
        let Some(name) = thread_file_name(id) else {
            return Ok(0);
        };
        if from.is_empty() {
            return Ok(0);
        }
        let _lock = self.lock(true);
        self.supersede_locked(id, &name, from)
    }

    fn touch(&self, id: &str) -> Result<bool, StoreError> {
        self.gate()?;
        self.change_row(id, |row| row.updated = (self.clock)())
    }

    fn archive(&self, id: &str) -> Result<(), StoreError> {
        self.gate()?;
        let _lock = self.lock(true);
        self.archive_locked(id)
    }

    fn unarchive(&self, key: &ArchiveKey) -> Result<IndexRow, StoreError> {
        self.gate()?;
        let _lock = self.lock(true);
        self.unarchive_locked(key)
    }
}

// ------------------------------------------------------------------ writing

/// A turn as `json.dumps(asdict(turn))` writes it, without the line end (S5).
pub fn turn_line(turn: &StoredTurn) -> String {
    let count = |value: Option<i64>| value.map_or(PyValue::Null, PyValue::int);
    let strs =
        |values: &[String]| PyValue::List(values.iter().cloned().map(PyValue::Str).collect());
    pyjson::dumps(&PyValue::Object(vec![
        ("id".into(), PyValue::str(turn.id.clone())),
        ("ts".into(), PyValue::Float(turn.ts)),
        ("role".into(), PyValue::str(turn.role.clone())),
        ("text".into(), PyValue::str(turn.text.clone())),
        ("tools".into(), strs(&turn.tools)),
        ("facts".into(), PyValue::List(turn.facts.clone())),
        ("unsupported".into(), strs(&turn.unsupported)),
        ("provider".into(), PyValue::str(turn.provider.clone())),
        ("error".into(), PyValue::str(turn.error.clone())),
        ("constrained".into(), PyValue::Bool(turn.constrained)),
        ("truncated".into(), PyValue::Bool(turn.truncated)),
        ("cancelled".into(), PyValue::Bool(turn.cancelled)),
        ("prompt_tokens".into(), count(turn.prompt_tokens)),
        ("completion_tokens".into(), count(turn.completion_tokens)),
        ("superseded".into(), PyValue::Bool(turn.superseded)),
    ]))
}

/// `_heal_torn_tail`: end a partial last line with a bare `\n` before an
/// append, so the damage stops at that line. Failures are ignored, as in
/// Python.
fn heal_torn_tail(path: &Path) {
    let heal = || -> std::io::Result<()> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        if file.metadata()?.len() == 0 {
            return Ok(());
        }
        file.seek(SeekFrom::End(-1))?;
        let mut last = [0u8; 1];
        file.read_exact(&mut last)?;
        if last[0] != b'\n' {
            file.seek(SeekFrom::End(0))?;
            file.write_all(b"\n")?;
        }
        Ok(())
    };
    let _ = heal();
}

/// `_write_index_locked` (S4′): sort by `updated`, newest first; archive the
/// rows past [`MAX_THREADS`] (S19); keep every row that could not be
/// archived; write `index.tmp` and replace `index.json`; and if that write
/// fails, take this batch's archive moves back.
fn write_index(disk: &Disk<'_>, mut rows: Vec<IndexRow>) -> std::io::Result<()> {
    fs::create_dir_all(disk.root())?;
    // Python's sort is stable and sees -0.0 and 0.0 as equal (deviation: a
    // NaN `updated` sorts first here, where Python's order is an artefact of
    // its comparisons).
    let key = |row: &IndexRow| if row.updated == 0.0 { 0.0 } else { row.updated };
    rows.sort_by(|a, b| key(b).total_cmp(&key(a)));
    let stale = rows.split_off(rows.len().min(MAX_THREADS));
    let (kept, batch) = archive::archive_evicted(disk, &stale);
    rows.extend(kept);
    let value = PyValue::List(rows.iter().map(archive::row_value).collect());
    let written = disk.write_index_file(disk.root(), &value);
    if written.is_err() {
        archive::undo(disk, batch);
    }
    written
}

/// Why an append was refused before anything was written.
fn saved<T>(result: std::io::Result<T>) -> Result<T, StoreError> {
    result.map_err(|_| StoreError::NotSaved)
}

impl SharedThreadStore {
    /// Gate G-WEB (§5.9): every write starts here.
    fn gate(&self) -> Result<(), StoreError> {
        if self.writes {
            Ok(())
        } else {
            Err(StoreError::SharedWritesOff)
        }
    }

    fn disk(&self) -> Disk<'_> {
        let disk = Disk::new(&self.dir, &self.clock);
        #[cfg(test)]
        let disk = disk.with_faults(
            self.faults
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone(),
        );
        disk
    }

    /// The index without the lock (S3).
    fn read_index(&self) -> IndexState {
        match fs::read(self.dir.join(INDEX)) {
            Ok(bytes) => parse_index(&bytes),
            Err(error) if error.kind() == ErrorKind::NotFound => IndexState::Absent,
            Err(_) => IndexState::Unreadable,
        }
    }

    /// The rows a write starts from; an index that exists and cannot be read
    /// here refuses every write before anything is written (S3, D7).
    fn rows_for_write(&self) -> Result<Vec<IndexRow>, StoreError> {
        match self.read_index() {
            IndexState::Absent => Ok(Vec::new()),
            IndexState::Rows(rows) => Ok(rows),
            IndexState::Unreadable => Err(StoreError::IndexUnreadable),
        }
    }

    /// `append` (S8), with S9's pin and D8's condition, in one hold of the
    /// lock and one index write. Returns the turn and the thread's row.
    fn append_turn(
        &self,
        id: &str,
        turn: NewTurn,
        answer: bool,
        pin: Option<&str>,
    ) -> Result<(StoredTurn, Option<IndexRow>), StoreError> {
        self.gate()?;
        let name = thread_file_name(id).ok_or(StoreError::NotSaved)?;
        let _lock = self.lock(true);
        let mut rows = self.rows_for_write()?;
        if answer && !rows.iter().any(|row| row.id == id) {
            // D8: the thread was archived (or deleted by the web) meanwhile.
            return Err(StoreError::ThreadGone);
        }
        // The defaults `append` gives, drawn once nothing can refuse the
        // write before it is made.
        let turn_id = if turn.id.is_empty() {
            (self.ids)()
        } else {
            turn.id.clone()
        };
        // `if not turn.ts`: zero is unset (NaN is not).
        let ts = if turn.ts == 0.0 {
            (self.clock)()
        } else {
            turn.ts
        };
        let stored = turn.stored(turn_id, ts);
        let disk = self.disk();
        saved(fs::create_dir_all(&self.dir))?;
        let path = self.dir.join(name);
        heal_torn_tail(&path);
        saved(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .and_then(|mut file| {
                    file.write_all(format!("{}{LINE_END}", turn_line(&stored)).as_bytes())
                }),
        )?;
        match rows.iter_mut().find(|row| row.id == id) {
            Some(row) => {
                row.updated = ts;
                row.turns += 1;
                // The first question names the thread.
                if stored.role == "user" && (row.title.is_empty() || row.title == NEW_THREAD_TITLE)
                {
                    row.title = auto_title(&stored.text);
                }
            }
            None => rows.push(IndexRow {
                id: id.to_owned(),
                title: auto_title(&stored.text),
                created: ts,
                updated: ts,
                turns: 1,
                pinned_provider: String::new(),
            }),
        }
        if let (Some(pin), Some(row)) = (pin, rows.iter_mut().find(|row| row.id == id)) {
            py::strip(pin).clone_into(&mut row.pinned_provider);
        }
        let row = rows.iter().find(|row| row.id == id).cloned();
        saved(write_index(&disk, rows))?;
        Ok((stored, row))
    }

    /// A change to one listed row and an index write; `false` for an unknown
    /// thread, which is not made.
    fn change_row(&self, id: &str, change: impl FnOnce(&mut IndexRow)) -> Result<bool, StoreError> {
        let _lock = self.lock(true);
        let mut rows = self.rows_for_write()?;
        let Some(row) = rows.iter_mut().find(|row| row.id == id) else {
            return Ok(false);
        };
        change(row);
        saved(write_index(&self.disk(), rows))?;
        Ok(true)
    }

    /// `supersede` (S13), under the lock.
    fn supersede_locked(&self, id: &str, name: &str, wanted: &str) -> Result<usize, StoreError> {
        self.rows_for_write()?;
        let path = self.dir.join(name);
        let Ok(bytes) = fs::read(&path) else {
            return Ok(0);
        };
        let Ok(text) = std::str::from_utf8(&bytes) else {
            return Ok(0);
        };
        let mut records: Vec<(String, Option<PyValue>)> = Vec::new();
        for line in py::splitlines(text) {
            let stripped = py::strip(line);
            if stripped.is_empty() {
                continue;
            }
            // D1: a line Python reads and this port cannot hold is written back
            // stripped, as any line that does not parse, and never marked.
            let record = pyjson::loads(stripped).ok().filter(PyValue::is_object);
            records.push((stripped.to_owned(), record));
        }
        let superseded = |record: &PyValue| record.get("superseded") == Some(&PyValue::Bool(true));
        let id_of =
            |record: &PyValue| pyjson::py_str(record.get("id").unwrap_or(&PyValue::str("")));
        let Some(start) = records.iter().position(|(_, record)| {
            record
                .as_ref()
                .is_some_and(|record| id_of(record) == wanted && !superseded(record))
        }) else {
            return Ok(0);
        };
        let mut marked = 0;
        let mut visible = 0_i64;
        let mut lines = Vec::with_capacity(records.len());
        for (index, (line, record)) in records.into_iter().enumerate() {
            let mut line = line;
            if let Some(mut record) = record {
                if index >= start && !superseded(&record) {
                    // `{**record, "superseded": True}`.
                    if let PyValue::Object(pairs) = &mut record {
                        match pairs.iter_mut().find(|(key, _)| key == "superseded") {
                            Some((_, value)) => *value = PyValue::Bool(true),
                            None => pairs.push(("superseded".into(), PyValue::Bool(true))),
                        }
                    }
                    line = pyjson::dumps(&record);
                    marked += 1;
                }
                if !superseded(&record) {
                    visible += 1;
                }
            }
            lines.push(line);
        }
        let disk = self.disk();
        let temporary = self.dir.join(format!("{name}.tmp"));
        saved(fs::write(&temporary, text_mode(&(lines.join("\n") + "\n"))))?;
        saved(disk.replace(&temporary, &path))?;
        let mut rows = self.rows_for_write()?;
        if let Some(row) = rows.iter_mut().find(|row| row.id == id) {
            row.turns = visible;
        }
        saved(write_index(&disk, rows))?;
        Ok(marked)
    }

    /// The reader's Archive (S21), under the lock: the thread through the
    /// eviction's own procedure, then the index without it; when that index
    /// write fails, the archive is taken back.
    fn archive_locked(&self, id: &str) -> Result<(), StoreError> {
        let rows = self.rows_for_write()?;
        let Some(at) = rows.iter().position(|row| row.id == id) else {
            return Err(StoreError::NotFound);
        };
        if matches!(
            archive::read_entries(&self.dir),
            archive::Entries::Unreadable
        ) {
            return Err(StoreError::ArchiveUnreadable);
        }
        let disk = self.disk();
        let row = rows[at].clone();
        let (kept, batch) = archive::archive_evicted(&disk, std::slice::from_ref(&row));
        if !kept.is_empty() {
            return Err(StoreError::NotSaved);
        }
        let rest: Vec<IndexRow> = rows
            .into_iter()
            .enumerate()
            .filter_map(|(index, row)| (index != at).then_some(row))
            .collect();
        if write_index(&disk, rest).is_err() {
            archive::undo(&disk, batch);
            return Err(StoreError::NotSaved);
        }
        Ok(())
    }

    /// Unarchive (S22), under the lock, in S22's order.
    fn unarchive_locked(&self, key: &ArchiveKey) -> Result<IndexRow, StoreError> {
        let mut rows = self.rows_for_write()?;
        let mut items = match archive::read_entries(&self.dir) {
            archive::Entries::Items(items) => items,
            archive::Entries::Unreadable | archive::Entries::Corrupt => {
                return Err(StoreError::ArchiveUnreadable);
            }
        };
        let evicted = self.dir.join(archive::EVICTED_DIR);
        let at = items
            .find(&key.id, key.created)
            .ok_or(StoreError::NotFound)?;
        let thread =
            archive::archived_thread(items.get(at), &evicted).ok_or(StoreError::NotFound)?;
        let name = thread_file_name(&thread.id).ok_or(StoreError::NotFound)?;
        let live = self.dir.join(name);
        // 1. A chat with this id is already in the list.
        if live.exists() || rows.iter().any(|row| row.id == thread.id) {
            return Err(StoreError::AlreadyListed);
        }
        let disk = self.disk();
        // 2. The transcript back to `<id>.jsonl` (a row with no transcript, or
        //    one whose transcript is missing, comes back as a row alone).
        let archived = if thread.transcript_missing || thread.file.is_empty() {
            None
        } else {
            archive::file_path(&evicted, &thread.file)
        };
        if let Some(archived) = &archived {
            saved(disk.move_new(archived, &live))?;
        }
        let put_back = |disk: &Disk<'_>| {
            if let Some(archived) = &archived {
                let _ = disk.move_new(&live, archived);
            }
        };
        // 3. The archive row out.
        let raw = items.remove(at);
        if archive::write_entries(&disk, &items).is_err() {
            put_back(&disk);
            return Err(StoreError::NotSaved);
        }
        // 4. Listed as the most recently used thread (Python's `touch` rule).
        let row = IndexRow {
            id: thread.id.clone(),
            title: thread.title.clone(),
            created: thread.created,
            updated: disk.now(),
            turns: thread.turns,
            pinned_provider: thread.pinned_provider.clone(),
        };
        rows.push(row.clone());
        if write_index(&disk, rows).is_err() {
            // 5. The archive row back, and the transcript back into `evicted/`.
            if let archive::Entries::Items(mut now) = archive::read_entries(&self.dir) {
                now.insert(at, raw);
                let _ = archive::write_entries(&disk, &now);
            }
            put_back(&disk);
            return Err(StoreError::NotSaved);
        }
        Ok(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    fn counter() -> IdSource {
        let next = Arc::new(std::sync::atomic::AtomicU64::new(1));
        Arc::new(move || {
            let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            format!("id{n:010}")
        })
    }

    fn turn(line: &str) -> Result<StoredTurn, CoerceError> {
        parse_turn(&pyjson::loads(line).unwrap(), &counter())
    }

    #[test]
    fn a_record_python_skips_is_skipped_whatever_else_it_holds() {
        // ts in another script (native cannot) and tools a number (Python raises).
        let both = format!(r#"{{"ts": "{}", "tools": 3}}"#, '\u{661}');
        assert_eq!(turn(&both), Err(CoerceError::Python));
        let native_only = format!(r#"{{"ts": "{}"}}"#, '\u{661}');
        assert_eq!(turn(&native_only), Err(CoerceError::Native));
        assert_eq!(turn("[1]"), Err(CoerceError::Python));
        assert_eq!(turn(r#"{"ts": null}"#), Err(CoerceError::Python));
    }

    #[test]
    fn a_falsy_id_draws_a_new_one_even_when_the_record_is_skipped() {
        let ids = counter();
        let bytes = b"{\"ts\": null}\n{\"text\": \"a\"}\n{\"id\": \"x\", \"text\": \"b\"}\n";
        let turns = parse_thread(bytes, &ids);
        assert_eq!(
            turns.iter().map(|t| t.id.as_str()).collect::<Vec<_>>(),
            ["id0000000002", "x"]
        );
    }

    #[test]
    fn the_protocol_view_maps_roles_facts_and_counts() {
        let stored = turn(
            r#"{"id": "t", "ts": 2.0, "role": "system", "text": "x", "facts": [{"label": "L", "rendered": 1.5, "as_of": null}], "prompt_tokens": -3, "completion_tokens": 4}"#,
        )
        .unwrap();
        let chat = stored.to_chat_turn();
        assert_eq!(chat.role, Role::User);
        assert_eq!(
            chat.facts,
            [Fact {
                label: "L".into(),
                rendered: "1.5".into(),
                as_of: String::new(),
                note: String::new()
            }]
        );
        assert_eq!(chat.prompt_tokens, None, "a negative count is no count");
        assert_eq!(chat.completion_tokens, Some(4));
        let assistant = turn(r#"{"role": "assistant"}"#).unwrap();
        assert_eq!(assistant.role(), Role::Assistant);
    }

    #[test]
    fn the_index_has_three_states() {
        let dir = TempDir::new("chat-index-states");
        let store = SharedThreadStore::new(dir.path());
        assert_eq!(store.list(), IndexState::Absent);
        fs::write(dir.path().join(INDEX), "{}").unwrap();
        assert_eq!(store.list(), IndexState::Rows(Vec::new()));
        fs::write(dir.path().join(INDEX), r#"[{"id": "a", "turns": "2"}]"#).unwrap();
        let IndexState::Rows(rows) = store.list() else {
            panic!("rows")
        };
        assert_eq!(rows[0].turns, 2);
        assert_eq!(rows[0].summary().turns, 2);
        for unreadable in [
            "[{\"id\": \"a\", \"updated\": NaN}]".to_owned(),
            "[{\"id\": \"a\", \"updated\": Infinity}]".to_owned(),
            "[{\"id\": \"a\", \"updated\": 1e400}]".to_owned(),
            format!("[{{\"id\": \"a\", \"title\": \"{}ud800\"}}]", '\\'),
            "[{\"id\": \"a\", \"turns\": 100000000000000000000}]".to_owned(),
            "[{\"id\": \"a\"".to_owned(),
        ] {
            fs::write(dir.path().join(INDEX), &unreadable).unwrap();
            assert_eq!(store.list(), IndexState::Unreadable, "{unreadable}");
        }
        let negative = IndexRow {
            id: "a".into(),
            title: String::new(),
            created: 0.0,
            updated: 0.0,
            turns: -4,
            pinned_provider: String::new(),
        };
        assert_eq!(negative.summary().turns, 0);
    }

    #[test]
    fn reading_leaves_every_file_as_it_was() {
        let dir = TempDir::new("chat-read-only");
        let index = br#"[{"id": "abc", "title": "T", "created": 1.0, "updated": 2.0, "turns": 1}]"#;
        let thread = b"{\"id\": \"t1\", \"ts\": 1.0, \"role\": \"user\", \"text\": \"q\"}\r\n{\"id\": \"t2\"";
        fs::write(dir.path().join(INDEX), index).unwrap();
        fs::write(dir.path().join("abc.jsonl"), thread).unwrap();
        let store = SharedThreadStore::new(dir.path());
        assert!(matches!(store.list(), IndexState::Rows(rows) if rows.len() == 1));
        assert_eq!(store.load("abc").len(), 1);
        assert_eq!(store.recent("abc").len(), 1);
        assert_eq!(
            store.load("a/b/c").len(),
            1,
            "a/b/c names abc.jsonl, as in Python"
        );
        assert!(
            store.load("../").is_empty(),
            "nothing is left of it: no file"
        );
        assert_eq!(fs::read(dir.path().join(INDEX)).unwrap(), index);
        assert_eq!(fs::read(dir.path().join("abc.jsonl")).unwrap(), thread);
        let mut names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["abc.jsonl", "index.json"], "nothing created");
    }

    #[test]
    fn recent_is_the_last_eight() {
        let dir = TempDir::new("chat-recent");
        let lines: String = (0..11)
            .map(|i| format!("{{\"id\": \"t{i}\", \"text\": \"q{i}\"}}\n"))
            .collect();
        fs::write(dir.path().join("th.jsonl"), lines).unwrap();
        let store = SharedThreadStore::new(dir.path());
        let recent: Vec<_> = store.recent("th").into_iter().map(|t| t.id).collect();
        assert_eq!(recent, ["t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]);
    }

    #[test]
    fn titles_count_code_points() {
        assert_eq!(auto_title(" \t "), NEW_THREAD_TITLE);
        let long: String = "\u{1f600}".repeat(49);
        let title = auto_title(&long);
        assert_eq!(title.chars().count(), TITLE_CHARS);
        assert!(title.ends_with('\u{2026}'));
        assert_eq!(
            normalize_title(&"y".repeat(81)).chars().count(),
            RENAME_CHARS
        );
        assert_eq!(normalize_title("  \u{3000} "), "");
    }

    // ------------------------------------------------------------- the gate

    /// Every write of the transcript store, by name.
    fn every_write(store: &dyn TranscriptStore) -> Vec<(&'static str, Option<StoreError>)> {
        let key = ArchiveKey {
            id: "abc".into(),
            created: 1.0,
        };
        vec![
            ("first_message", store.first_message("q", "auto").err()),
            ("append", store.append("abc", NewTurn::user("q")).err()),
            (
                "append_answer",
                store
                    .append_answer("abc", NewTurn::assistant("a", "lab:m"))
                    .err(),
            ),
            ("rename", store.rename("abc", "T").err()),
            ("pin", store.pin("abc", "auto").err()),
            ("supersede", store.supersede("abc", "t1").err()),
            ("touch", store.touch("abc").err()),
            ("archive", store.archive("abc").err()),
            ("unarchive", store.unarchive(&key).err()),
        ]
    }

    /// Every file under `dir`, with its bytes, but the lock file: a write
    /// that refuses has still taken the store's lock, as Python's would, and
    /// `.store.lock` is part of Python's format (S24) and holds no bytes.
    fn snapshot(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        let mut out = Vec::new();
        let mut pending = vec![dir.to_path_buf()];
        while let Some(folder) = pending.pop() {
            for entry in fs::read_dir(&folder).unwrap() {
                let path = entry.unwrap().path();
                if path.file_name().is_some_and(|name| name == LOCK_FILE) {
                    continue;
                }
                if path.is_dir() {
                    out.push((path.clone(), Vec::new()));
                    pending.push(path);
                } else {
                    let bytes = fs::read(&path).unwrap();
                    out.push((path, bytes));
                }
            }
        }
        out.sort();
        out
    }

    /// A store with an index, a thread and an archive, as Python leaves one.
    fn populated(tag: &str) -> TempDir {
        let dir = TempDir::new(tag);
        let index = [
            "[",
            " {",
            r#"  "id": "abc","#,
            r#"  "title": "T","#,
            r#"  "created": 1.0,"#,
            r#"  "updated": 2.0,"#,
            r#"  "turns": 1,"#,
            r#"  "pinned_provider": """#,
            " }",
            "]",
        ]
        .join("\r\n");
        fs::write(dir.path().join(INDEX), index).unwrap();
        fs::write(
            dir.path().join("abc.jsonl"),
            r#"{"id": "t1", "ts": 2.0, "role": "user", "text": "q"}"#.to_owned() + "\r\n",
        )
        .unwrap();
        fs::create_dir(dir.path().join(archive::EVICTED_DIR)).unwrap();
        fs::write(
            dir.path().join(archive::EVICTED_DIR).join(INDEX),
            "[{\"id\": \"abc\", \"created\": 0.5, \"evicted_at\": 1.5, \"file\": \"abc.jsonl\"}]",
        )
        .unwrap();
        fs::write(
            dir.path().join(archive::EVICTED_DIR).join("abc.jsonl"),
            "old",
        )
        .unwrap();
        dir
    }

    #[test]
    fn the_closed_gate_refuses_every_write_and_changes_no_byte() {
        let dir = populated("chat-gate-closed");
        let before = snapshot(dir.path());
        let store = SharedThreadStore::with_gate(dir.path(), uuid_ids(), false);
        assert!(!store.writes_open());
        let results = every_write(&store);
        println!("closed gate: {results:?}");
        assert_eq!(results.len(), 9);
        for (name, result) in &results {
            assert_eq!(*result, Some(StoreError::SharedWritesOff), "{name}");
        }
        let refusal = StoreError::SharedWritesOff.refusal();
        assert_eq!(refusal.kind, lattice_protocol::RefusalKind::Unavailable);
        assert_eq!(refusal.message, store_gate::REFUSAL);
        assert_eq!(snapshot(dir.path()), before, "nothing was written");
        assert!(
            !dir.path().join(LOCK_FILE).exists(),
            "a closed gate takes no lock and makes no lock file"
        );
        // Reading goes on.
        assert!(matches!(store.list(), IndexState::Rows(rows) if rows.len() == 1));
        assert_eq!(store.load("abc").len(), 1);
    }

    #[test]
    fn past_an_open_gate_a_write_is_not_refused_by_the_gate() {
        let dir = populated("chat-gate-open");
        let before = snapshot(dir.path());
        let store = SharedThreadStore::with_gate(dir.path(), uuid_ids(), true);
        let results = every_write(&store);
        println!("open gate: {results:?}");
        for (name, result) in &results {
            assert_ne!(*result, Some(StoreError::SharedWritesOff), "{name}");
        }
        assert_ne!(snapshot(dir.path()), before, "the writes wrote");
    }

    #[test]
    fn the_product_store_follows_the_gate_constant() {
        let dir = populated("chat-gate-constant");
        let before = snapshot(dir.path());
        let store = SharedThreadStore::new(dir.path());
        assert_eq!(store.writes_open(), store_gate::SHARED_WRITES);
        assert_eq!(store_gate::shared_writes(), store_gate::SHARED_WRITES);
        let results = every_write(&store);
        println!("SHARED_WRITES = {}: {results:?}", store_gate::SHARED_WRITES);
        for (name, result) in &results {
            if store_gate::SHARED_WRITES {
                // After G5: the gate lets writes through.
                assert_ne!(*result, Some(StoreError::SharedWritesOff), "{name}");
            } else {
                assert_eq!(*result, Some(StoreError::SharedWritesOff), "{name}");
            }
        }
        if !store_gate::SHARED_WRITES {
            assert_eq!(snapshot(dir.path()), before);
        }
    }

    #[test]
    fn both_stores_serve_the_seam() {
        use super::super::memory::MemoryTranscriptStore;
        let dir = populated("chat-seam");
        let shared: Box<dyn TranscriptStore> = Box::new(SharedThreadStore::new(dir.path()));
        let memory: Box<dyn TranscriptStore> = Box::new(MemoryTranscriptStore::new(
            counter(),
            crate::clock::system_clock(),
        ));
        assert!(matches!(shared.list(), IndexState::Rows(rows) if rows[0].id == "abc"));
        assert!(matches!(shared.archived(), ArchiveState::Rows(rows) if rows.len() == 1));
        assert_eq!(shared.recent("abc").len(), 1);
        assert_eq!(memory.list(), IndexState::Absent);
        let (row, _) = memory.first_message("q", "auto").unwrap();
        assert_eq!(memory.recent(&row.id).len(), 1);
    }

    // ------------------------------------------------------- the write path

    use super::super::write_parity_tests::{PLATFORM_LINE_END, assert_nothing_lost, raw_files};

    /// A clock that moves one second per reading, from `start`.
    fn ticking(start: f64) -> Clock {
        let next = Arc::new(Mutex::new(start));
        Arc::new(move || {
            let mut now = next.lock().unwrap();
            *now += 1.0;
            *now
        })
    }

    fn open_store(dir: &Path) -> SharedThreadStore {
        SharedThreadStore::with_gate(dir, counter(), true).clocked(ticking(2_000_000_000.0))
    }

    fn listed(store: &SharedThreadStore) -> Vec<IndexRow> {
        match store.list() {
            IndexState::Rows(rows) => rows,
            IndexState::Absent => Vec::new(),
            IndexState::Unreadable => panic!("unreadable"),
        }
    }

    /// Every archive row names a transcript that is there, and every
    /// transcript in `evicted/` is named by a row (S19, S22).
    fn assert_archive_whole(dir: &Path) {
        let evicted = dir.join(archive::EVICTED_DIR);
        let ArchiveState::Rows(rows) = archive::read_evicted(dir) else {
            panic!("the archive index cannot be read")
        };
        let mut named = Vec::new();
        for row in &rows {
            let archive::ArchiveRow::Thread(thread) = row else {
                continue;
            };
            assert!(
                !thread.transcript_missing,
                "{} names a missing file",
                thread.file
            );
            if !thread.file.is_empty() {
                named.push(thread.file.clone());
            }
        }
        for entry in fs::read_dir(&evicted).unwrap() {
            let name = entry.unwrap().file_name().into_string().unwrap();
            if name.ends_with(".jsonl") {
                assert!(
                    named.contains(&name),
                    "{name} sits in evicted/ and no row names it"
                );
            }
        }
    }

    #[test]
    fn an_index_native_cannot_hold_refuses_every_write_and_changes_no_byte() {
        // D7 (S3): Python reads this index and keeps its rows; a native write
        // over a list it read as empty would drop them.
        let dir = populated("chat-d7-writes");
        let unreadable = "[{\"id\": \"abc\", \"title\": \"T\", \"created\": 1.0, \"updated\": NaN, \"turns\": 1}]";
        fs::write(dir.path().join(INDEX), unreadable).unwrap();
        let before = snapshot(dir.path());
        let store = open_store(dir.path());
        let results = every_write(&store);
        println!("D7: {results:?}");
        for (name, result) in &results {
            assert_eq!(*result, Some(StoreError::IndexUnreadable), "{name}");
        }
        assert_eq!(snapshot(dir.path()), before, "nothing was written");
    }

    #[test]
    fn an_answer_to_a_thread_no_longer_listed_is_not_written() {
        // D8: Python's `append` would make a thread of the answer alone.
        let dir = populated("chat-d8");
        let before = snapshot(dir.path());
        let store = open_store(dir.path());
        let answer = NewTurn::assistant("late", "lab:m");
        assert_eq!(
            store.append_answer("gone1", answer.clone()),
            Err(StoreError::ThreadGone)
        );
        assert_eq!(snapshot(dir.path()), before);
        // The same answer to a listed thread is written.
        let saved = store.append_answer("abc", answer).unwrap();
        assert_eq!(
            store.load("abc").last().map(|t| t.id.clone()),
            Some(saved.id)
        );
    }

    #[test]
    fn lines_end_as_python_s_text_mode_ends_them() {
        // S2: CRLF on Windows, LF elsewhere; a torn tail is healed with a bare LF.
        let dir = TempDir::new("chat-line-ends");
        fs::write(dir.path().join("th.jsonl"), "{\"id\": \"t1\"").unwrap();
        let store = open_store(dir.path());
        store.append("th", NewTurn::user("q")).unwrap();
        let bytes = fs::read(dir.path().join("th.jsonl")).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.starts_with("{\"id\": \"t1\"\n{"), "{text:?}");
        assert!(text.ends_with(PLATFORM_LINE_END), "{text:?}");
        assert_eq!(text.matches('\n').count(), 2);
        let index = fs::read_to_string(dir.path().join(INDEX)).unwrap();
        assert_eq!(
            index.matches(PLATFORM_LINE_END).count(),
            index.matches('\n').count()
        );
    }

    #[test]
    fn a_line_python_reads_and_native_cannot_is_kept_and_never_marked() {
        // D1 in S13: written back stripped, as any line that does not parse.
        let dir = TempDir::new("chat-d1-supersede");
        fs::write(
            dir.path().join("th.jsonl"),
            "{\"id\": \"t1\", \"ts\": 1.0, \"role\": \"user\", \"text\": \"q\"}\n  {\"id\": \"t2\", \"ts\": NaN}  \n",
        )
        .unwrap();
        let store = open_store(dir.path());
        assert_eq!(store.supersede("th", "t1"), Ok(1));
        let text = fs::read_to_string(dir.path().join("th.jsonl")).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[1], "{\"id\": \"t2\", \"ts\": NaN}");
        assert!(lines[0].ends_with("\"superseded\": true}"));
    }

    #[test]
    fn the_reader_s_archive_moves_a_thread_and_refuses_what_it_cannot() {
        let dir = populated("chat-archive-reader");
        let store = open_store(dir.path());
        assert_eq!(store.archive("nope"), Err(StoreError::NotFound));
        let before = raw_files(dir.path());
        store.archive("abc").unwrap();
        let after = raw_files(dir.path());
        assert_nothing_lost(&before, &after, "archive");
        assert!(listed(&store).is_empty());
        assert!(!dir.path().join("abc.jsonl").exists());
        assert_archive_whole(dir.path());
        // An archive index only Python can read: nothing is archived (S20, D7).
        let other = populated("chat-archive-d7");
        fs::write(
            other.path().join(archive::EVICTED_DIR).join(INDEX),
            "[{\"id\": \"x\", \"created\": NaN}]",
        )
        .unwrap();
        let untouched = snapshot(other.path());
        let store = open_store(other.path());
        assert_eq!(store.archive("abc"), Err(StoreError::ArchiveUnreadable));
        assert_eq!(snapshot(other.path()), untouched);
    }

    /// A store at the cap: `count` chats, made in order, the first the oldest.
    fn chats(store: &SharedThreadStore, count: usize) -> Vec<IndexRow> {
        (0..count)
            .map(|n| store.first_message(&format!("question {n}"), "").unwrap().0)
            .collect()
    }

    #[test]
    fn unarchive_at_sixty_lists_the_thread_and_archives_an_older_one_whole() {
        // S22, I11's case: with 60 threads listed, the restored thread is the
        // most recently used one, so S4′ archives another, older thread, and
        // no archive row names a file that is not there, nor is any file in
        // `evicted/` unnamed.
        let dir = TempDir::new("chat-unarchive-60");
        let store = open_store(dir.path());
        let rows = chats(&store, 60);
        let chosen = &rows[10];
        store.archive(&chosen.id).unwrap();
        chats(&store, 1);
        assert_eq!(listed(&store).len(), 60);
        let key = ArchiveKey {
            id: chosen.id.clone(),
            created: chosen.created,
        };
        let before = raw_files(dir.path());
        let restored = store.unarchive(&key).unwrap();
        let after = raw_files(dir.path());
        assert_nothing_lost(&before, &after, "unarchive");
        let now = listed(&store);
        assert_eq!(now.len(), 60);
        assert_eq!(
            now[0].id, chosen.id,
            "listed as the most recently used thread"
        );
        assert!(restored.updated > chosen.updated);
        assert_eq!(restored.created, chosen.created);
        assert!(
            !now.iter().any(|row| row.id == rows[0].id),
            "the oldest was archived"
        );
        assert_eq!(store.load(&chosen.id).len(), 1, "its transcript is back");
        assert_archive_whole(dir.path());
        // Unarchiving it again: it is not in the archive any more.
        assert_eq!(store.unarchive(&key), Err(StoreError::NotFound));
    }

    #[test]
    fn unarchive_refuses_an_id_already_listed_and_changes_no_byte() {
        let dir = TempDir::new("chat-unarchive-conflict");
        let store = open_store(dir.path());
        let row = chats(&store, 1).remove(0);
        store.archive(&row.id).unwrap();
        // A late save brings the id back as a thread of its own.
        store.append(&row.id, NewTurn::user("again")).unwrap();
        let before = snapshot(dir.path());
        let key = ArchiveKey {
            id: row.id.clone(),
            created: row.created,
        };
        assert_eq!(store.unarchive(&key), Err(StoreError::AlreadyListed));
        assert_eq!(snapshot(dir.path()), before);
        assert_eq!(
            StoreError::AlreadyListed.refusal().kind,
            lattice_protocol::RefusalKind::Conflict
        );
    }

    #[test]
    fn an_unarchive_whose_index_write_fails_puts_everything_back() {
        // S22 step 5.
        let dir = TempDir::new("chat-unarchive-undo");
        let store = open_store(dir.path());
        let rows = chats(&store, 3);
        store.archive(&rows[1].id).unwrap();
        let without_temporaries = |dir: &Path| -> Vec<(PathBuf, Vec<u8>)> {
            snapshot(dir)
                .into_iter()
                .filter(|(path, _)| path.file_name().is_none_or(|name| name != "index.tmp"))
                .collect()
        };
        let before = without_temporaries(dir.path());
        store.set_faults(vec![archive::Fault {
            dst: Some(INDEX.to_owned()),
            ..archive::Fault::default()
        }]);
        let key = ArchiveKey {
            id: rows[1].id.clone(),
            created: rows[1].created,
        };
        assert_eq!(store.unarchive(&key), Err(StoreError::NotSaved));
        store.set_faults(Vec::new());
        assert_eq!(
            without_temporaries(dir.path()),
            before,
            "the row and the transcript are back"
        );
        assert_archive_whole(dir.path());
        store.unarchive(&key).unwrap();
        assert_eq!(listed(&store)[0].id, rows[1].id);
    }

    #[test]
    fn an_archive_index_only_python_can_read_stops_the_cap_and_loses_nothing() {
        // S20: no archiving, no set-aside, the bytes unchanged; the thread
        // past the cap stays listed.
        let dir = TempDir::new("chat-cap-d7");
        let store = open_store(dir.path());
        chats(&store, 60);
        let evicted = dir.path().join(archive::EVICTED_DIR);
        fs::create_dir_all(&evicted).unwrap();
        fs::write(evicted.join(INDEX), "[NaN]").unwrap();
        chats(&store, 1);
        assert_eq!(listed(&store).len(), 61);
        assert_eq!(fs::read(evicted.join(INDEX)).unwrap(), b"[NaN]");
        assert_eq!(fs::read_dir(&evicted).unwrap().count(), 1);
    }

    // ------------------------------------------------------- the lock (S15′)

    /// `.store.lock`, locked through a handle of its own, as another process
    /// (or another store) holds it.
    fn hold_raw(dir: &Path) -> File {
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(dir.join(LOCK_FILE))
            .unwrap();
        file.try_lock().unwrap();
        file
    }

    /// Release `held` after `after`; the instant just before the release.
    fn release_later(held: File, after: Duration) -> std::thread::JoinHandle<Instant> {
        std::thread::spawn(move || {
            std::thread::sleep(after);
            let at = Instant::now();
            held.unlock().unwrap();
            drop(held);
            at
        })
    }

    #[test]
    fn a_write_waits_for_the_lock_on_the_store_lock_file() {
        let dir = TempDir::new("chat-lock-write");
        let store = open_store(dir.path());
        store.append("th", NewTurn::user("first")).unwrap();
        let releaser = release_later(hold_raw(dir.path()), Duration::from_millis(400));
        let start = Instant::now();
        store.append("th", NewTurn::user("second")).unwrap();
        let done = Instant::now();
        let released = releaser.join().unwrap();
        println!(
            "the write took {:?}; the lock was released {:?} after it started",
            done - start,
            released.saturating_duration_since(start)
        );
        assert!(done >= released, "the write finished before the release");
        assert_eq!(store.lock_timeouts(), 0);
        assert_eq!(store.load("th").len(), 2);
    }

    #[test]
    fn a_read_of_a_thread_waits_for_the_lock_and_list_does_not() {
        // S26: `load` and `recent` take the lock; `list` and `archived` take
        // the mutex only, as Python's `list_threads`.
        let dir = TempDir::new("chat-lock-read");
        let store = open_store(dir.path());
        store.append("th", NewTurn::user("q")).unwrap();
        for read in ["load", "recent"] {
            let releaser = release_later(hold_raw(dir.path()), Duration::from_millis(300));
            let turns = match read {
                "load" => store.load("th"),
                _ => store.recent("th"),
            };
            let done = Instant::now();
            let released = releaser.join().unwrap();
            assert_eq!(turns.len(), 1);
            assert!(done >= released, "{read} read while the lock was held");
        }
        let held = hold_raw(dir.path());
        let start = Instant::now();
        assert!(matches!(store.list(), IndexState::Rows(rows) if rows.len() == 1));
        assert!(matches!(store.archived(), ArchiveState::Absent));
        assert!(start.elapsed() < Duration::from_secs(5), "list waited");
        held.unlock().unwrap();
        assert_eq!(store.lock_timeouts(), 0);
    }

    #[test]
    fn a_wait_that_runs_out_goes_on_under_the_mutex_and_is_counted() {
        let dir = TempDir::new("chat-lock-timeout");
        let store = open_store(dir.path()).lock_waiting(Duration::from_millis(100));
        let held = hold_raw(dir.path());
        store.append("th", NewTurn::user("q")).unwrap();
        assert_eq!(
            store.lock_timeouts(),
            1,
            "the write went on and was counted"
        );
        assert_eq!(store.load("th").len(), 1);
        assert_eq!(store.lock_timeouts(), 2, "so did the read");
        held.unlock().unwrap();
        drop(held);
        store.append("th", NewTurn::user("again")).unwrap();
        assert_eq!(store.lock_timeouts(), 2);
    }

    #[test]
    fn two_stores_on_one_folder_exclude_each_other_and_release_at_once() {
        let dir = TempDir::new("chat-lock-two");
        let first = open_store(dir.path());
        let second = open_store(dir.path()).lock_waiting(Duration::from_millis(100));
        {
            let lock = first.hold_lock();
            assert!(lock.os_locked());
            second.append("th", NewTurn::user("q")).unwrap();
            assert_eq!(second.lock_timeouts(), 1);
        }
        // Released when the guard dropped: the next write takes it at once.
        let lock = second.hold_lock();
        assert!(lock.os_locked());
        drop(lock);
        assert_eq!(second.lock_timeouts(), 1);
    }

    #[test]
    fn a_read_makes_no_lock_file_and_a_write_makes_one() {
        let dir = TempDir::new("chat-lock-file");
        fs::write(
            dir.path().join("th.jsonl"),
            "{\"id\": \"t1\"}
",
        )
        .unwrap();
        let store = open_store(dir.path());
        assert_eq!(store.load("th").len(), 1);
        assert!(!dir.path().join(LOCK_FILE).exists());
        assert_eq!(store.lock_timeouts(), 0);
        let missing = TempDir::new("chat-lock-none");
        let absent = open_store(&missing.path().join("never"));
        assert!(absent.load("th").is_empty());
        assert!(
            !missing.path().join("never").exists(),
            "a read makes no folder"
        );
        store.touch("th").unwrap();
        assert!(dir.path().join(LOCK_FILE).exists());
        assert_eq!(
            fs::read(dir.path().join(LOCK_FILE)).unwrap(),
            b"",
            "it holds no bytes"
        );
    }
}
