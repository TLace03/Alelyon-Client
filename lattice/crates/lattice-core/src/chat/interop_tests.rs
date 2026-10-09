//! Interop tests: the native store and the web Lattice's `history.py` on one
//! store, under one lock (the chat core's spec §16.3, I1–I11, I14, I15;
//! row G3).
//!
//! Each test points a [`SharedThreadStore`] (through the `#[cfg(test)]`
//! constructor that opens the gate, so these tests do not depend on
//! `SHARED_WRITES`) and a real
//! Python child at the same TEMPORARY folder. The child is
//! `tools/lattice_chat_interop.py`, run by the repository's interpreter: it
//! registers that folder as a named store, does what the test asks with the
//! branch's own `history.py`, and prints JSON lines (events, then one
//! result). It is the only process these tests start. Every child has a
//! deadline of [`DEADLINE`]; one that overruns it is killed (only that
//! child) and the test FAILS. `ALELYON_HOME` and the temporary-folder
//! variables point the child into the test's own temporary folder, and the
//! helper refuses any store outside the temporary folder.
//!
//! The interpreter is `LATTICE_PYTHON`, else the repository's virtual environment, else, in
//! a linked worktree, the main checkout's (found from `.git` and
//! `commondir`, without running git). When none exists, each test prints
//! that it was SKIPPED and why, and returns; when one exists, each prints
//! that it RAN.
//!
//! Not here: I12 and I13 (the writer lease, row E5), which are
//! `workspace/lease_interop_tests.rs` and start the same helper through
//! [`Py::start_with`]. I14 and I15 run here at the store's seam, with the
//! core's calls (§2.4) made by the test; their agent-level versions (I14's
//! sidecar half, I15 through a waiting approval) are
//! `convo/agent_interop_tests.rs` (row E11).
//!
//! I16 (spec Q6): the per-thread answer lock, both ways, through the web
//! service's own job registry: while the web answers a thread, the plain
//! chat core refuses a send there and shows it answering elsewhere; while
//! the native side holds the lock, the web refuses before it writes.
//!
//! W1 (row G4): web chats Python wrote, listed, opened (the visible last
//! 400) and their archive listed through the plain chat core composed as a
//! shipped build composes it (`ChatConfig::new`, the gate open since G5).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use lattice_protocol::chat::{ChatEventKind, ChatService};
use lattice_protocol::conversation::ArchiveKey;

use super::archive::{self, ArchiveRow, ArchiveState, text_mode};
use super::core_tests::{Drip, DripEnd, Harness, local_shown};
use super::pyjson::{self, PyValue};
use super::store::{IndexRow, IndexState, SharedThreadStore, StoredTurn, turn_line, uuid_ids};
use super::transcript::{NewTurn, StoreError, TranscriptStore};
use super::{ChatConfig, Locked, answer, answer_lock, refusals};
use crate::env::MapEnv;
use crate::fsx;
use crate::testkit::TempDir;

/// How long one Python child may run. Generous: an overrun is a failure,
/// never a skip, and is not a measurement of anything.
const DEADLINE: Duration = Duration::from_secs(300);

/// The repository: five folders above this crate.
pub(crate) fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(5)
        .expect("the crate sits five folders below the repository")
        .to_path_buf()
}

fn venv_python(checkout: &Path) -> PathBuf {
    if cfg!(windows) {
        checkout.join(".venv312").join("Scripts").join("python.exe")
    } else {
        checkout.join(".venv312").join("bin").join("python")
    }
}

/// A linked worktree's main checkout: `.git` is a file naming the worktree's
/// git folder, whose `commondir` names the main `.git`.
fn main_checkout(repo: &Path) -> Option<PathBuf> {
    let gitfile = fs::read_to_string(repo.join(".git")).ok()?;
    let gitdir = PathBuf::from(gitfile.trim().strip_prefix("gitdir:")?.trim());
    let common = fs::read_to_string(gitdir.join("commondir")).ok()?;
    Some(lexical(&gitdir.join(common.trim()).join("..")))
}

/// `path` with `.` and `..` resolved by its text alone.
fn lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for part in path.components() {
        match part {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other),
        }
    }
    out
}

/// The interpreter, or why there is none.
pub(crate) fn find_python() -> Result<PathBuf, String> {
    if let Some(given) = std::env::var_os("LATTICE_PYTHON") {
        let given = PathBuf::from(given);
        return if given.is_file() {
            Ok(given)
        } else {
            Err(format!(
                "LATTICE_PYTHON names {}, which is not a file",
                given.display()
            ))
        };
    }
    let repo = repo_root();
    let mut tried = vec![venv_python(&repo)];
    if let Some(main) = main_checkout(&repo) {
        tried.push(venv_python(&main));
    }
    tried
        .iter()
        .find(|path| path.is_file())
        .cloned()
        .ok_or_else(|| {
            format!(
                "no interpreter: LATTICE_PYTHON is unset and none of {} exists",
                tried
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// How many interop tests ran and were skipped in this process.
static RAN: AtomicUsize = AtomicUsize::new(0);
static SKIPPED: AtomicUsize = AtomicUsize::new(0);

/// The interpreter, or print that the test is skipped and return.
macro_rules! python_or_skip {
    ($name:expr) => {
        match Py::new($name) {
            Ok(py) => py,
            Err(why) => {
                let skipped = SKIPPED.fetch_add(1, Ordering::Relaxed) + 1;
                println!(
                    "interop {}: SKIPPED ({skipped} skipped so far): {why}",
                    $name
                );
                return;
            }
        }
    };
}

/// The repository's Python, with a state home of its own.
pub(crate) struct Py {
    name: &'static str,
    exe: PathBuf,
    script: PathBuf,
    home: TempDir,
}

impl Py {
    pub(crate) fn new(name: &'static str) -> Result<Self, String> {
        let exe = find_python()?;
        let script = repo_root().join("tools").join("lattice_chat_interop.py");
        if !script.is_file() {
            return Err(format!("{} is missing", script.display()));
        }
        let ran = RAN.fetch_add(1, Ordering::Relaxed) + 1;
        println!(
            "interop {name}: RAN ({ran} run so far) with {}",
            exe.display()
        );
        Ok(Self {
            name,
            exe,
            script,
            home: TempDir::new("interop-home"),
        })
    }

    /// Start `command` against `store`.
    pub(crate) fn start(&self, store: &Path, command: &str, args: &[&str]) -> Running {
        let mut all: Vec<std::ffi::OsString> = vec!["--store".into(), store.into()];
        all.extend(args.iter().map(|arg| (*arg).into()));
        self.start_with(
            command,
            &all,
            &[("ALELYON_HOME", self.home.path().as_os_str())],
            &[],
        )
    }

    /// Start `command` with `args`, the variables `set` and without the
    /// variables `remove` (the lease commands of row E5 run under Python's
    /// default state home, so the test chooses `ALELYON_HOME` and the home).
    pub(crate) fn start_with(
        &self,
        command: &str,
        args: &[std::ffi::OsString],
        set: &[(&str, &std::ffi::OsStr)],
        remove: &[&str],
    ) -> Running {
        let temp = std::env::temp_dir();
        let what = format!(
            "{}: {command} {}",
            self.name,
            args.iter()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(" ")
        );
        let mut command_line = Command::new(&self.exe);
        command_line
            .arg("-B")
            .arg(&self.script)
            .arg(command)
            .args(args)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("TMP", &temp)
            .env("TEMP", &temp)
            .env("TMPDIR", &temp)
            .current_dir(self.home.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for name in remove {
            command_line.env_remove(name);
        }
        for (name, value) in set {
            command_line.env(name, value);
        }
        let mut child = command_line
            .spawn()
            .unwrap_or_else(|error| panic!("{what}: could not start: {error}"));
        let stdout = child.stdout.take().expect("piped stdout");
        let (sender, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        let stderr = Arc::new(Mutex::new(String::new()));
        let mut pipe = child.stderr.take().expect("piped stderr");
        let sink = stderr.clone();
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text);
            sink.lock().unwrap().push_str(&text);
        });
        let stdin = child.stdin.take();
        Running {
            what,
            child: Some(child),
            stdin,
            lines,
            stderr,
            deadline: Instant::now() + DEADLINE,
        }
    }

    /// Run `command` to its end; its result.
    pub(crate) fn run(&self, store: &Path, command: &str, args: &[&str]) -> PyValue {
        self.start(store, command, args).finish()
    }
}

/// A Python child the test started, with its deadline.
pub(crate) struct Running {
    what: String,
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    lines: mpsc::Receiver<String>,
    stderr: Arc<Mutex<String>>,
    deadline: Instant,
}

impl Running {
    /// Kill the child (it is ours), and fail the test.
    fn overrun(&mut self) -> ! {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        panic!(
            "{} overran its deadline of {DEADLINE:?} and was killed; stderr: {}",
            self.what,
            self.stderr.lock().unwrap()
        );
    }

    /// The next line, or `None` at the end of its output.
    fn line(&mut self) -> Option<String> {
        let left = self.deadline.saturating_duration_since(Instant::now());
        match self.lines.recv_timeout(left) {
            Ok(line) => Some(line),
            Err(mpsc::RecvTimeoutError::Disconnected) => None,
            Err(mpsc::RecvTimeoutError::Timeout) => self.overrun(),
        }
    }

    /// The next line must be the event `name`; its fields.
    pub(crate) fn event(&mut self, name: &str) -> PyValue {
        let Some(line) = self.line() else {
            let status = self.wait();
            panic!(
                "{}: ended ({status}) before the event {name}; stderr: {}",
                self.what,
                self.stderr.lock().unwrap()
            );
        };
        let value = pyjson::loads(&line)
            .unwrap_or_else(|error| panic!("{}: {line:?} is not JSON: {error:?}", self.what));
        assert_eq!(
            value.get("event").and_then(PyValue::as_str),
            Some(name),
            "{}: {line}",
            self.what
        );
        value
    }

    /// Send a cue line.
    pub(crate) fn send(&mut self, cue: &str) {
        let stdin = self.stdin.as_mut().expect("stdin is open");
        writeln!(stdin, "{cue}")
            .and_then(|()| stdin.flush())
            .unwrap_or_else(|error| panic!("{}: cue {cue}: {error}", self.what));
    }

    /// Wait for the child's exit within the deadline.
    fn wait(&mut self) -> std::process::ExitStatus {
        loop {
            let child = self.child.as_mut().expect("the child");
            match child.try_wait() {
                Ok(Some(status)) => {
                    self.child = None;
                    return status;
                }
                Ok(None) if Instant::now() < self.deadline => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Ok(None) => self.overrun(),
                Err(error) => panic!("{}: wait: {error}", self.what),
            }
        }
    }

    /// Read to the end; the child must succeed, and its last line is the
    /// result.
    pub(crate) fn finish(mut self) -> PyValue {
        self.stdin = None;
        let mut last = None;
        while let Some(line) = self.line() {
            last = Some(line);
        }
        let status = self.wait();
        // The stderr reader ends with the child.
        let stderr = self.stderr.lock().unwrap().clone();
        assert!(
            status.success(),
            "{}: {status}; stderr: {stderr}",
            self.what
        );
        let line = last.unwrap_or_else(|| panic!("{}: no output; stderr: {stderr}", self.what));
        let value = pyjson::loads(&line)
            .unwrap_or_else(|error| panic!("{}: {line:?} is not JSON: {error:?}", self.what));
        value
            .get("result")
            .cloned()
            .unwrap_or_else(|| panic!("{}: no result in {line}", self.what))
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        // A test that failed half-way leaves no child behind: only this one,
        // which the test started, is killed.
        if let Some(child) = self.child.as_mut()
            && matches!(child.try_wait(), Ok(None))
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

// ------------------------------------------------------------------ values

fn field<'a>(value: &'a PyValue, key: &str) -> &'a PyValue {
    value
        .get(key)
        .unwrap_or_else(|| panic!("no {key} in {}", pyjson::dumps(value)))
}

fn float(value: &PyValue, key: &str) -> f64 {
    pyjson::py_float(field(value, key)).unwrap()
}

pub(crate) fn int(value: &PyValue, key: &str) -> i64 {
    pyjson::py_int(field(value, key)).unwrap()
}

pub(crate) fn flag(value: &PyValue, key: &str) -> bool {
    match field(value, key) {
        PyValue::Bool(value) => *value,
        other => panic!("{key} is {other:?}"),
    }
}

pub(crate) fn text(value: &PyValue, key: &str) -> String {
    field(value, key).as_str().expect("a string").to_owned()
}

fn list<'a>(value: &'a PyValue, key: &str) -> &'a [PyValue] {
    match field(value, key) {
        PyValue::List(items) => items,
        other => panic!("{key} is {other:?}"),
    }
}

fn strings(value: &PyValue, key: &str) -> Vec<String> {
    list(value, key)
        .iter()
        .map(|item| item.as_str().expect("a string").to_owned())
        .collect()
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("after 1970")
        .as_secs_f64()
}

/// A turn as Python's `asdict` gives it.
fn turn_value(turn: &StoredTurn) -> PyValue {
    pyjson::loads(&turn_line(turn)).unwrap()
}

/// A store whose writes pass (tests only), on the system clock.
fn shared(dir: &Path) -> SharedThreadStore {
    SharedThreadStore::with_gate(dir, uuid_ids(), true)
}

fn rows(store: &SharedThreadStore) -> Vec<IndexRow> {
    match store.list() {
        IndexState::Rows(rows) => rows,
        IndexState::Absent => Vec::new(),
        IndexState::Unreadable => panic!("the index cannot be read"),
    }
}

pub(crate) fn listed_ids(store: &SharedThreadStore) -> Vec<String> {
    rows(store).into_iter().map(|row| row.id).collect()
}

fn python_list(py: &Py, store: &Path) -> Vec<PyValue> {
    list(&py.run(store, "list", &[]), "threads").to_vec()
}

pub(crate) fn python_load(py: &Py, store: &Path, thread: &str, all: bool) -> Vec<PyValue> {
    let mut args = vec!["--thread", thread];
    if all {
        args.push("--all");
    }
    list(&py.run(store, "load", &args), "turns").to_vec()
}

/// Every file under `dir` (not `.store.lock`), with its bytes.
fn files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(folder) = pending.pop() {
        let Ok(entries) = fs::read_dir(&folder) else {
            continue;
        };
        for entry in entries {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            let name = path
                .strip_prefix(dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if name != ".store.lock" {
                out.insert(name, fs::read(&path).unwrap());
            }
        }
    }
    out
}

/// The archive is whole: no row names a missing file, and no transcript in
/// `evicted/` is named by no row (S19, S22).
pub(crate) fn assert_archive_whole(dir: &Path, context: &str) {
    let ArchiveState::Rows(rows) = archive::read_evicted(dir) else {
        panic!("{context}: the archive index cannot be read")
    };
    let mut named = BTreeSet::new();
    for row in &rows {
        let ArchiveRow::Thread(thread) = row else {
            continue;
        };
        assert!(
            !thread.transcript_missing,
            "{context}: a row names {}, which is missing",
            thread.file
        );
        if !thread.file.is_empty() {
            assert!(
                named.insert(thread.file.clone()),
                "{context}: two rows name {}",
                thread.file
            );
        }
    }
    for entry in fs::read_dir(dir.join(archive::EVICTED_DIR)).unwrap() {
        let name = entry.unwrap().file_name().into_string().unwrap();
        if Path::new(&name)
            .extension()
            .is_some_and(|ext| ext == "jsonl")
        {
            assert!(
                named.contains(&name),
                "{context}: {name} sits in evicted/ and no row names it"
            );
        }
    }
}

// ------------------------------------------------------------------ I1–I4

#[test]
fn i1_a_rust_write_started_while_python_holds_the_lock_ends_after_its_release() {
    let py = python_or_skip!("I1");
    let dir = TempDir::new("interop-i1");
    let root = dir.path().join("store");
    let store = shared(&root);
    let mut holder = py.start(&root, "hold-lock", &["--seconds", "2"]);
    let held = holder.event("held");
    assert!(
        flag(&held, "os_locked"),
        "Python holds the operating-system lock"
    );
    let started = unix_now();
    let saved = store.append("i1thread", NewTurn::user("written by Rust"));
    let ended = unix_now();
    let result = holder.finish();
    let releasing = float(&result, "releasing_at");
    println!(
        "I1: Python held at {:.3}, released at {releasing:.3}; Rust started {started:.3}, ended {ended:.3} ({:.3} s after the release)",
        float(&held, "at"),
        ended - releasing
    );
    assert!(saved.is_ok(), "{saved:?}");
    assert!(
        started < releasing,
        "the Rust write began while Python held the lock"
    );
    assert!(
        ended >= releasing,
        "the Rust write ended before Python released the lock"
    );
    assert_eq!(
        store.lock_timeouts(),
        0,
        "the write took the lock, it did not time out"
    );
    assert_eq!(python_load(&py, &root, "i1thread", false).len(), 1);
}

#[test]
fn i2_python_cannot_take_the_lock_while_rust_holds_it() {
    let py = python_or_skip!("I2");
    let dir = TempDir::new("interop-i2");
    let root = dir.path().join("store");
    let store = shared(&root);
    let lock = store.hold_lock();
    assert!(lock.os_locked());
    let refused = py.run(&root, "try-lock", &["--lock-wait", "0.3"]);
    println!("I2: while Rust held it: {}", pyjson::dumps(&refused));
    assert!(!flag(&refused, "locked"), "Python took the lock Rust holds");
    assert!(
        float(&refused, "waited") >= 0.29,
        "Python waited its LOCK_WAIT_S"
    );
    drop(lock);
    // The positive control: released, Python takes it at once.
    let taken = py.run(&root, "try-lock", &["--lock-wait", "0.3"]);
    println!("I2: after the release: {}", pyjson::dumps(&taken));
    assert!(flag(&taken, "locked"), "Python could not take a free lock");
}

#[test]
fn i3_python_and_rust_append_120_turns_each_to_one_thread_at_once() {
    let py = python_or_skip!("I3");
    let dir = TempDir::new("interop-i3");
    let root = dir.path().join("store");
    let store = shared(&root);
    let thread = "i3thread";
    let mut python = py.start(
        &root,
        "append",
        &[
            "--thread",
            thread,
            "--count",
            "120",
            "--prefix",
            "python",
            "--wait-go",
        ],
    );
    python.event("ready");
    python.send("go");
    let rust_first = unix_now();
    let mut failed = Vec::new();
    for n in 0..120 {
        if let Err(error) = store.append(thread, NewTurn::user(format!("rust {n}"))) {
            failed.push((n, error));
        }
    }
    let rust_last = unix_now();
    let result = python.finish();
    let (py_first, py_last) = (float(&result, "first_at"), float(&result, "last_at"));
    println!(
        "I3: Rust appended {rust_first:.3}..{rust_last:.3}, Python {py_first:.3}..{py_last:.3}; Python ok {} failed {}; Rust failed {failed:?}; lock timeouts {}",
        int(&result, "ok"),
        int(&result, "failed"),
        store.lock_timeouts()
    );
    assert!(
        py_first < rust_last && rust_first < py_last,
        "the two writers ran at the same time"
    );
    assert!(failed.is_empty(), "Rust appends failed: {failed:?}");
    assert_eq!(
        (int(&result, "ok"), int(&result, "failed")),
        (120, 0),
        "Python's appends"
    );
    let ours: Vec<PyValue> = store.load(thread).iter().map(turn_value).collect();
    let theirs = python_load(&py, &root, thread, false);
    assert_eq!(ours.len(), 240, "Rust reads 240 turns");
    assert_eq!(ours, theirs, "both loaders read the same list");
    let texts: BTreeSet<String> = ours.iter().map(|turn| text(turn, "text")).collect();
    assert_eq!(texts.len(), 240, "every turn once, none torn or lost");
    for side in ["python", "rust"] {
        let order: Vec<String> = ours
            .iter()
            .map(|turn| text(turn, "text"))
            .filter(|text| text.starts_with(side))
            .collect();
        let expected: Vec<String> = (0..120).map(|n| format!("{side} {n}")).collect();
        assert_eq!(order, expected, "{side}'s turns, in its order");
    }
    let row = rows(&store)
        .into_iter()
        .find(|row| row.id == thread)
        .unwrap();
    assert_eq!(row.turns, 240, "the index counts every turn");
    let python_rows = python_list(&py, &root);
    assert_eq!(int(&python_rows[0], "turns"), 240);
}

#[test]
fn i4_a_rust_index_write_retries_past_a_python_reader_and_never_loses_the_row() {
    let py = python_or_skip!("I4");
    let dir = TempDir::new("interop-i4");
    let root = dir.path().join("store");
    let store = Arc::new(shared(&root));
    let thread = "i4thread";
    store.append(thread, NewTurn::user("seed")).unwrap();
    let mut holder = py.start(&root, "hold-open", &["--file", "index.json"]);
    // A Python reader holds index.json open; the write that meets it
    // retries (fsx's 6 tries, 20 ms apart) and succeeds after the release.
    // The write waits at its first transient failure (a test hook in fsx's
    // retry) until the test has had Python release the handle, so the
    // order is the test's, not the clock's: no load can make the release
    // come before the write meets the handle, or after its retries run out.
    let mut tries = 0;
    let mut retried = 0;
    while tries < 20 && retried == 0 {
        tries += 1;
        holder.send("open");
        holder.event("opened");
        let writer = store.clone();
        let text = format!("during a read {tries}");
        let (met_tx, met_rx) = std::sync::mpsc::channel::<()>();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let write = std::thread::spawn(move || {
            let mut first = Some((met_tx, go_rx));
            fsx::on_transient(Some(Box::new(move || {
                if let Some((met, go)) = first.take() {
                    let _ = met.send(());
                    let _ = go.recv();
                }
            })));
            let before = fsx::transient_failures();
            let saved = writer.append(thread, NewTurn::user(text));
            fsx::on_transient(None);
            (saved, fsx::transient_failures() - before)
        });
        // Released once the write has met the handle (or has ended without
        // meeting it).
        let _ = met_rx.recv_timeout(DEADLINE);
        holder.send("release");
        holder.event("released");
        let _ = go_tx.send(());
        let (saved, failures) = write.join().unwrap();
        assert!(saved.is_ok(), "try {tries}: {saved:?}");
        retried = failures;
    }
    println!("I4: a write met the reader's handle and retried {retried} times, on try {tries}");
    assert!(
        retried > 0,
        "no write met the Python reader's handle in {tries} tries"
    );
    // Held for good: the write fails after its retries and the index is
    // left as it was, with every row.
    holder.send("open");
    holder.event("opened");
    let index = root.join("index.json");
    let before = fs::read(&index).unwrap();
    let held = store.append(thread, NewTurn::user("while held"));
    let after = fs::read(&index).unwrap();
    holder.send("release");
    holder.event("released");
    holder.send("end");
    holder.finish();
    println!("I4: held for good: {held:?}");
    assert_eq!(held, Err(StoreError::NotSaved));
    assert_eq!(before, after, "the index is unchanged, its row kept");
    store.append(thread, NewTurn::user("after")).unwrap();
    let listed = python_list(&py, &root);
    assert_eq!(listed.len(), 1);
    assert_eq!(text(&listed[0], "id"), thread);
    let turns = python_load(&py, &root, thread, false);
    assert_eq!(
        turns.last().map(|turn| text(turn, "text")),
        Some("after".to_owned())
    );
    // The line written before the failed index write is kept, as Python's
    // own `append` keeps it.
    assert!(turns.iter().any(|turn| text(turn, "text") == "while held"));
}

// ------------------------------------------------------------------ I5–I7

/// The turns both sides write in I5, with every field set.
fn i5_turns() -> Vec<NewTurn> {
    let fact = pyjson::loads(
        r#"{"label": "Rate", "rendered": "4.25%", "as_of": "2026-10-01", "note": ""}"#,
    )
    .unwrap();
    vec![
        NewTurn {
            id: "i5u1".into(),
            ts: 1_790_000_000.125,
            ..NewTurn::user(
                "What is the rate in Z\u{fc}rich \u{2013} \u{6771}\u{4eac}? \"quoted\"\n",
            )
        },
        NewTurn {
            id: "i5a1".into(),
            ts: 1_790_000_001.123_456_7,
            tools: vec!["rates".into()],
            facts: vec![fact],
            unsupported: vec!["9".into()],
            truncated: true,
            prompt_tokens: Some(11),
            completion_tokens: None,
            ..NewTurn::assistant("It is 4.25% \u{1f600}.", "llamacpp:qwen3-8b")
        },
        NewTurn {
            id: "i5u2".into(),
            ts: 1_790_000_002.5,
            ..NewTurn::user("and then?")
        },
        NewTurn {
            id: "i5a2".into(),
            ts: 1_790_000_003.0,
            error: "The answer stopped.".into(),
            cancelled: true,
            constrained: true,
            ..NewTurn::assistant("", "endpoint-x:model")
        },
    ]
}

#[test]
fn i5_threads_round_trip_both_ways_with_equal_bytes() {
    let py = python_or_skip!("I5");
    let dir = TempDir::new("interop-i5");
    let root = dir.path().join("store");
    let store = shared(&root);
    let turns = i5_turns();
    // Rust writes one thread; Python writes the same turns to another.
    for turn in &turns {
        if turn.role == "assistant" {
            store.append_answer("i5rust", turn.clone()).unwrap();
        } else {
            store.append("i5rust", turn.clone()).unwrap();
        }
    }
    let values: Vec<PyValue> = turns
        .iter()
        .map(|turn| turn_value(&turn.clone().stored(turn.id.clone(), turn.ts)))
        .collect();
    let file = dir.path().join("turns.json");
    fs::write(&file, pyjson::dumps(&PyValue::List(values.clone()))).unwrap();
    let written = py.run(
        &root,
        "append",
        &["--thread", "i5python", "--turns", file.to_str().unwrap()],
    );
    assert_eq!(int(&written, "ok"), 4);
    // Python reads Rust's thread, and Rust reads Python's, field for field.
    assert_eq!(python_load(&py, &root, "i5rust", false), values);
    let ours: Vec<PyValue> = store.load("i5python").iter().map(turn_value).collect();
    assert_eq!(ours, values);
    // The same turns make the same bytes (S2, S5).
    let rust_bytes = fs::read(root.join("i5rust.jsonl")).unwrap();
    let python_bytes = fs::read(root.join("i5python.jsonl")).unwrap();
    println!(
        "I5: {} bytes each; CRLF lines: {}",
        rust_bytes.len(),
        rust_bytes.windows(2).filter(|pair| pair == b"\r\n").count()
    );
    assert_eq!(rust_bytes, python_bytes, "equal bytes on this platform");
    // The two rows agree but for the id.
    let by_id = |id: &str| {
        let row = python_list(&py, &root)
            .into_iter()
            .find(|row| text(row, "id") == id)
            .unwrap();
        (
            text(&row, "title"),
            float(&row, "created"),
            float(&row, "updated"),
            int(&row, "turns"),
        )
    };
    assert_eq!(by_id("i5rust"), by_id("i5python"));
    // A Python supersede of Rust's thread: both read the same result.
    let marked = py.run(
        &root,
        "supersede",
        &["--thread", "i5rust", "--from-turn", "i5u2"],
    );
    assert_eq!(int(&marked, "marked"), 2);
    let ours: Vec<PyValue> = store.load("i5rust").iter().map(turn_value).collect();
    assert_eq!(ours, python_load(&py, &root, "i5rust", false));
    assert_eq!(ours.len(), 2);
}

#[test]
fn i6_an_index_python_accepts_and_rust_cannot_read_refuses_every_rust_write() {
    let py = python_or_skip!("I6");
    let dir = TempDir::new("interop-i6");
    let root = dir.path().join("store");
    let ids = py.run(&root, "write-d7-index", &[]);
    let ids = strings(&ids, "ids");
    let store = shared(&root);
    let before = files(&root);
    assert_eq!(store.list(), IndexState::Unreadable);
    let key = ArchiveKey {
        id: ids[0].clone(),
        created: 1.0,
    };
    let results = [
        ("first_message", store.first_message("q", "auto").err()),
        ("append", store.append(&ids[1], NewTurn::user("q")).err()),
        (
            "append_answer",
            store
                .append_answer(&ids[1], NewTurn::assistant("a", "lab:m"))
                .err(),
        ),
        ("rename", store.rename(&ids[1], "renamed").err()),
        ("pin", store.pin(&ids[1], "local").err()),
        ("supersede", store.supersede(&ids[1], "t1").err()),
        ("touch", store.touch(&ids[1]).err()),
        ("archive", store.archive(&ids[1]).err()),
        ("unarchive", store.unarchive(&key).err()),
    ];
    println!("I6: {results:?}");
    for (name, result) in results {
        assert_eq!(result, Some(StoreError::IndexUnreadable), "{name}");
    }
    assert_eq!(files(&root), before, "no byte changed and no file made");
    let listed = py.run(&root, "list", &["--ids"]);
    assert_eq!(strings(&listed, "ids"), ids, "Python still lists every row");
}

#[test]
fn i7_a_thread_python_deletes_while_a_rust_answer_streams_does_not_come_back() {
    let py = python_or_skip!("I7");
    let shared_store: Arc<Mutex<Option<Arc<SharedThreadStore>>>> = Arc::default();
    let slot = shared_store.clone();
    let h = Harness::with(
        "interop-i7",
        MapEnv::new(),
        move |config: &mut ChatConfig, _, _| {
            let store = Arc::new(shared(&config.state.chat_dir()));
            *slot.lock().unwrap() = Some(store.clone());
            config.store = store;
            config.store_dir = config.state.chat_dir().display().to_string();
        },
    );
    let store = shared_store.lock().unwrap().clone().unwrap();
    let root = store.dir().to_path_buf();
    h.fixture.install("m");
    // A scripted model whose stream is held open after its first characters
    // until the test lets it go, after Python's delete: by the test's word,
    // not by time, so a slow Python start-up cannot let the answer finish
    // first.
    let gate = Arc::new(tokio::sync::Notify::new());
    h.answer_with(
        Drip::chars(
            "an answer that is still being written when its chat is deleted",
            Duration::from_millis(35),
            DripEnd::Done,
        )
        .held_after(3, gate.clone()),
    );
    let accepted = h.send(None, "q", "local", local_shown()).unwrap();
    let thread = accepted.thread.id.clone();
    assert_eq!(
        python_load(&py, &root, &thread, false).len(),
        1,
        "the question is saved"
    );
    let deleted = py.run(&root, "delete", &["--thread", &thread]);
    assert!(flag(&deleted, "deleted"));
    gate.notify_one();
    let outcome = h.outcome(&accepted.job);
    println!("I7: {outcome:?}");
    match &outcome {
        ChatEventKind::Error { message, saved, .. } => {
            assert_eq!(message, answer::GONE);
            assert!(!saved, "the answer was not saved");
        }
        other => panic!("the answer was saved: {other:?}"),
    }
    assert!(
        !python_list(&py, &root)
            .iter()
            .any(|row| text(row, "id") == thread),
        "the thread does not come back (D8)"
    );
    let name = super::store::thread_file_name(&thread).unwrap();
    assert!(!root.join(name).exists(), "no transcript was made again");
    assert!(python_load(&py, &root, &thread, false).is_empty());
}

// ------------------------------------------------------------------ I8, I9

/// `value` with thread ids as `T<n>` (`order` gives n) and the times under
/// `created`, `updated` and `evicted_at` as their rank among `times`.
fn normalised(value: &PyValue, order: &[String], times: &[f64]) -> PyValue {
    let rename = |text: &str| -> String {
        let mut out = text.to_owned();
        for (n, id) in order.iter().enumerate() {
            out = out.replace(id.as_str(), &format!("T{n}"));
        }
        out
    };
    match value {
        PyValue::List(items) => PyValue::List(
            items
                .iter()
                .map(|item| normalised(item, order, times))
                .collect(),
        ),
        PyValue::Object(pairs) => PyValue::Object(
            pairs
                .iter()
                .map(|(key, item)| {
                    let item = match (key.as_str(), item) {
                        ("created" | "updated" | "evicted_at", PyValue::Float(time)) => {
                            let rank = times.iter().position(|t| t == time).unwrap();
                            PyValue::Int(rank.to_string())
                        }
                        ("id" | "file", PyValue::Str(text)) => PyValue::Str(rename(text)),
                        _ => normalised(item, order, times),
                    };
                    (key.clone(), item)
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

fn times_in(value: &PyValue, out: &mut Vec<f64>) {
    match value {
        PyValue::List(items) => items.iter().for_each(|item| times_in(item, out)),
        PyValue::Object(pairs) => {
            for (key, item) in pairs {
                match (key.as_str(), item) {
                    ("created" | "updated" | "evicted_at", PyValue::Float(time)) => {
                        out.push(*time);
                    }
                    _ => times_in(item, out),
                }
            }
        }
        _ => {}
    }
}

/// A store's `index.json` and `evicted/index.json`, each checked to be in
/// Python's format, normalised.
fn normalised_indexes(root: &Path, order: &[String]) -> (PyValue, PyValue) {
    let read = |path: PathBuf| {
        let bytes = fs::read(&path).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let value = pyjson::loads(&text).unwrap();
        assert_eq!(
            text,
            text_mode(&pyjson::dumps_indent1(&value)),
            "{} is json.dumps(indent=1) in text mode",
            path.display()
        );
        value
    };
    let index = read(root.join("index.json"));
    let evicted = read(root.join(archive::EVICTED_DIR).join("index.json"));
    let mut times = Vec::new();
    times_in(&index, &mut times);
    times_in(&evicted, &mut times);
    times.sort_by(f64::total_cmp);
    times.dedup();
    (
        normalised(&index, order, &times),
        normalised(&evicted, order, &times),
    )
}

#[test]
fn i8_eviction_both_ways() {
    let py = python_or_skip!("I8");
    let dir = TempDir::new("interop-i8");
    // Mixed: Python makes 60 threads and Rust the 61st.
    let mixed = dir.path().join("mixed");
    let made = py.run(
        &mixed,
        "fill-past-cap",
        &["--count", "60", "--prefix", "question"],
    );
    let mut mixed_order = strings(&made, "ids");
    let oldest = format!("{}.jsonl", mixed_order[0]);
    let oldest_bytes = fs::read(mixed.join(&oldest)).unwrap();
    let store = shared(&mixed);
    let (row, _) = store.first_message("question 60", "").unwrap();
    mixed_order.push(row.id.clone());
    // Python alone makes all 61.
    let python = dir.path().join("python");
    let made = py.run(
        &python,
        "fill-past-cap",
        &["--count", "61", "--prefix", "question"],
    );
    let python_order = strings(&made, "ids");
    let ours = normalised_indexes(&mixed, &mixed_order);
    let theirs = normalised_indexes(&python, &python_order);
    println!(
        "I8: normalised evicted/index.json: {}",
        pyjson::dumps(&ours.1)
    );
    assert_eq!(
        ours.0, theirs.0,
        "index.json as Python's eviction leaves it"
    );
    assert_eq!(
        ours.1, theirs.1,
        "evicted/index.json as Python's eviction leaves it"
    );
    assert_eq!(listed_ids(&store).len(), 60);
    assert!(!mixed.join(&oldest).exists());
    assert_eq!(
        fs::read(mixed.join(archive::EVICTED_DIR).join(&oldest)).unwrap(),
        oldest_bytes,
        "moved, not unlinked"
    );
    assert_archive_whole(&mixed, "I8 mixed");
    // Python's eviction, listed by Rust under Archived with equal rows.
    let state = py.run(&python, "archive-state", &[]);
    let python_rows = list(&state, "rows");
    let ArchiveState::Rows(rust_rows) = shared(&python).archived() else {
        panic!("Rust cannot read Python's archive")
    };
    assert_eq!(rust_rows.len(), 1);
    assert_eq!(python_rows.len(), 1);
    let ArchiveRow::Thread(thread) = &rust_rows[0] else {
        panic!("not a thread row")
    };
    let expected = &python_rows[0];
    assert_eq!(thread.raw, *expected, "the row as stored");
    assert_eq!(thread.id, text(expected, "id"));
    assert_eq!(
        thread.created.to_bits(),
        float(expected, "created").to_bits()
    );
    assert_eq!(thread.title, text(expected, "title"));
    assert_eq!(
        thread.updated.to_bits(),
        float(expected, "updated").to_bits()
    );
    assert_eq!(thread.turns, int(expected, "turns"));
    assert_eq!(thread.pinned_provider, text(expected, "pinned_provider"));
    assert_eq!(
        thread.evicted_at.to_bits(),
        float(expected, "evicted_at").to_bits()
    );
    assert_eq!(thread.file, text(expected, "file"));
    assert!(!thread.transcript_missing);
    assert_eq!(thread.id, python_order[0]);
}

/// A set-aside archive index's name: `index.corrupt-<yyyymmddTHHMMSS>[-n].json`.
fn is_set_aside_name(name: &str) -> bool {
    let Some(rest) = name
        .strip_prefix("index.corrupt-")
        .and_then(|rest| rest.strip_suffix(".json"))
    else {
        return false;
    };
    let (stamp, suffix) = rest.split_at(rest.len().min(15));
    let stamp_ok = stamp.len() == 15
        && stamp.char_indices().all(|(at, c)| {
            if at == 8 {
                c == 'T'
            } else {
                c.is_ascii_digit()
            }
        });
    let suffix_ok = suffix.is_empty()
        || suffix
            .strip_prefix('-')
            .is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
    stamp_ok && suffix_ok
}

fn set_aside(root: &Path) -> Vec<(String, Vec<u8>)> {
    fs::read_dir(root.join(archive::EVICTED_DIR))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("index.corrupt-"))
        })
        .map(|path| {
            (
                path.file_name().unwrap().to_string_lossy().into_owned(),
                fs::read(&path).unwrap(),
            )
        })
        .collect()
}

#[test]
fn i9_a_corrupt_archive_index_is_set_aside_by_either_side_and_read_by_the_other() {
    let py = python_or_skip!("I9");
    let dir = TempDir::new("interop-i9");
    let corrupt = "{not json";
    // Rust sets it aside; Python reads Rust's fresh index.
    let first = dir.path().join("rust-sets-aside");
    py.run(&first, "fill-past-cap", &["--count", "60"]);
    py.run(&first, "write-corrupt-archive-index", &["--text", corrupt]);
    let store = shared(&first);
    store.first_message("the 61st", "").unwrap();
    let aside = set_aside(&first);
    println!(
        "I9: Rust set aside {:?}",
        aside.iter().map(|(name, _)| name).collect::<Vec<_>>()
    );
    assert_eq!(aside.len(), 1);
    assert!(is_set_aside_name(&aside[0].0), "{}", aside[0].0);
    assert_eq!(aside[0].1, corrupt.as_bytes(), "byte for byte");
    let state = py.run(&first, "archive-state", &[]);
    let python_rows = list(&state, "rows");
    let ArchiveState::Rows(rust_rows) = store.archived() else {
        panic!("Rust's fresh index cannot be read")
    };
    assert_eq!(python_rows.len(), 1, "Python reads Rust's fresh index");
    assert_eq!(rust_rows[0].raw(), &python_rows[0]);
    assert_archive_whole(&first, "I9 Rust");
    // Python sets it aside; Rust reads Python's fresh index.
    let second = dir.path().join("python-sets-aside");
    let store = shared(&second);
    let made: Vec<String> = (0..60)
        .map(|n| store.first_message(&format!("rust {n}"), "").unwrap().0.id)
        .collect();
    py.run(&second, "write-corrupt-archive-index", &["--text", corrupt]);
    py.run(&second, "fill-past-cap", &["--count", "1"]);
    let aside = set_aside(&second);
    println!(
        "I9: Python set aside {:?}",
        aside.iter().map(|(name, _)| name).collect::<Vec<_>>()
    );
    assert_eq!(aside.len(), 1);
    assert!(is_set_aside_name(&aside[0].0), "{}", aside[0].0);
    assert_eq!(aside[0].1, corrupt.as_bytes());
    let ArchiveState::Rows(rust_rows) = store.archived() else {
        panic!("Rust cannot read Python's fresh index")
    };
    assert_eq!(rust_rows.len(), 1);
    let ArchiveRow::Thread(thread) = &rust_rows[0] else {
        panic!("not a thread row")
    };
    assert_eq!(
        thread.id, made[0],
        "Rust's oldest thread, archived by Python"
    );
    assert!(!thread.transcript_missing);
    assert_archive_whole(&second, "I9 Python");
}

// ------------------------------------------------------------------ I10

#[test]
fn i10_a_python_supersede_loop_never_fails_beside_a_rust_load_loop() {
    let py = python_or_skip!("I10");
    let dir = TempDir::new("interop-i10");
    let root = dir.path().join("store");
    let store = Arc::new(shared(&root));
    let thread = "i10thread";
    // A thread long enough that a read takes a while: 300 turns of 1 KB.
    let filler = "x".repeat(1000);
    for n in 0..300 {
        store
            .append(thread, NewTurn::user(format!("{n} {filler}")))
            .unwrap();
    }
    let mut python = py.start(
        &root,
        "supersede-loop",
        // A 30 ms pause between Python's iterations, outside its lock: a
        // writer in a tight loop retakes the lock before a reader polling
        // every 20 ms can, and the readers would starve (observed).
        &["--thread", thread, "--seconds", "10", "--pause", "0.03"],
    );
    python.event("ready");
    let stop = Arc::new(AtomicBool::new(false));
    // Four readers, so that a read without the lock would meet a replace.
    let readers: Vec<_> = (0..4)
        .map(|_| {
            let (reader, done) = (store.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut loads = 0_u64;
                let mut odd = Vec::new();
                while !done.load(Ordering::Relaxed) {
                    let count = reader.load(thread).len();
                    if !(300..=301).contains(&count) && odd.len() < 5 {
                        odd.push(count);
                    }
                    loads += 1;
                }
                (loads, odd)
            })
        })
        .collect();
    python.send("go");
    let result = python.finish();
    stop.store(true, Ordering::Relaxed);
    let (mut loads, mut odd) = (0, Vec::new());
    for reader in readers {
        let (count, seen) = reader.join().unwrap();
        loads += count;
        odd.extend(seen);
    }
    println!(
        "I10: Python {} supersedes, {} failures {}; Rust {loads} loads; odd counts {odd:?}; lock timeouts {}",
        int(&result, "iterations"),
        int(&result, "failures"),
        pyjson::dumps(field(&result, "errors")),
        store.lock_timeouts()
    );
    assert_eq!(int(&result, "failures"), 0, "a Python supersede failed");
    assert!(
        int(&result, "iterations") >= 20,
        "too few supersedes to tell"
    );
    assert!(loads >= 20, "too few loads to tell");
    assert!(odd.is_empty(), "Rust read a torn thread: {odd:?}");
}

// ------------------------------------------------------------------ I11

#[test]
fn i11_the_reader_s_archive_and_unarchive_at_sixty_listed_threads() {
    let py = python_or_skip!("I11");
    let dir = TempDir::new("interop-i11");
    let root = dir.path().join("store");
    let store = shared(&root);
    let first = strings(
        &py.run(&root, "fill-past-cap", &["--count", "60", "--prefix", "t"]),
        "ids",
    );
    let chosen = first[10].clone();
    let created = rows(&store)
        .into_iter()
        .find(|row| row.id == chosen)
        .unwrap()
        .created;
    // Rust archives one thread (S21).
    store.archive(&chosen).unwrap();
    let rust_file = format!("{chosen}.jsonl");
    let evicted = root.join(archive::EVICTED_DIR);
    let rust_bytes = fs::read(evicted.join(&rust_file)).unwrap();
    let rust_row = match archive::read_evicted(&root) {
        ArchiveState::Rows(rows) => rows[0].raw().clone(),
        other => panic!("{other:?}"),
    };
    // A late save brings the id back, and Python's cap then archives every
    // older thread, that one included.
    py.run(
        &root,
        "append",
        &["--thread", &chosen, "--count", "1", "--prefix", "late"],
    );
    let later = strings(
        &py.run(&root, "fill-past-cap", &["--count", "60", "--prefix", "u"]),
        "ids",
    );
    assert_eq!(listed_ids(&store).len(), 60);
    let ArchiveState::Rows(archived) = archive::read_evicted(&root) else {
        panic!("the archive cannot be read")
    };
    assert_eq!(archived.len(), 61, "Rust's row and Python's 60");
    assert!(
        archived.iter().any(|row| row.raw() == &rust_row),
        "Python kept Rust's row as it was"
    );
    assert_eq!(
        fs::read(evicted.join(&rust_file)).unwrap(),
        rust_bytes,
        "not overwritten"
    );
    assert!(
        evicted.join(format!("{chosen}-2.jsonl")).is_file(),
        "Python picked a name of its own for the id's second life"
    );
    assert_archive_whole(&root, "I11 after Python's evictions");
    // Rust unarchives with exactly 60 listed (S22).
    assert_eq!(listed_ids(&store).len(), 60);
    let restored = store
        .unarchive(&ArchiveKey {
            id: chosen.clone(),
            created,
        })
        .unwrap();
    let listed = listed_ids(&store);
    println!(
        "I11: restored {chosen} (updated {}); listed {}; first {}",
        restored.updated,
        listed.len(),
        listed[0]
    );
    assert_eq!(listed.len(), 60);
    assert_eq!(listed[0], chosen, "the most recently used thread");
    assert!(!listed.contains(&later[0]), "an older thread was archived");
    assert_eq!(
        fs::read(root.join(&rust_file)).unwrap(),
        rust_bytes,
        "its transcript is back"
    );
    assert_archive_whole(&root, "I11 after Rust's unarchive");
    let python_listed = python_list(&py, &root);
    assert_eq!(
        text(&python_listed[0], "id"),
        chosen,
        "Python lists it first"
    );
    // Python's next cap eviction archives correctly and overwrites nothing.
    let before = files(&evicted);
    py.run(&root, "fill-past-cap", &["--count", "1", "--prefix", "v"]);
    let after = files(&evicted);
    for (name, bytes) in &before {
        if name == "index.json" {
            continue;
        }
        assert_eq!(
            after.get(name),
            Some(bytes),
            "{name} was overwritten or lost"
        );
    }
    assert!(after.contains_key(&format!("{}.jsonl", later[1])));
    assert!(listed_ids(&store).contains(&chosen));
    assert_archive_whole(&root, "I11 after Python's next eviction");
}

// ------------------------------------------------------------------ I14, I15

#[test]
fn i14_supersede_across_the_two_lattices() {
    let py = python_or_skip!("I14");
    let dir = TempDir::new("interop-i14");
    let root = dir.path().join("store");
    let store = shared(&root);
    // A native turn: the question and the final answer, in the shared record.
    let (row, question) = store.first_message("native question", "auto").unwrap();
    let thread = row.id.clone();
    store
        .append_answer(&thread, NewTurn::assistant("native answer", "llamacpp:m"))
        .unwrap();
    // The web regenerates it: Python supersedes from the question.
    let marked = py.run(
        &root,
        "supersede",
        &["--thread", &thread, "--from-turn", &question.id],
    );
    assert_eq!(int(&marked, "marked"), 2);
    assert!(
        store.load(&thread).is_empty(),
        "Rust leaves out the superseded turn"
    );
    assert!(python_load(&py, &root, &thread, false).is_empty());
    let kept = python_load(&py, &root, &thread, true);
    assert_eq!(kept.len(), 2, "kept in the file");
    assert!(kept.iter().all(|turn| flag(turn, "superseded")));
    // A Rust regenerate is hidden by Python's load_thread.
    store.append(&thread, NewTurn::user("again")).unwrap();
    let first = store
        .append_answer(&thread, NewTurn::assistant("first answer", "llamacpp:m"))
        .unwrap();
    assert_eq!(store.supersede(&thread, &first.id), Ok(1));
    store
        .append_answer(&thread, NewTurn::assistant("regenerated", "llamacpp:m"))
        .unwrap();
    let visible: Vec<String> = python_load(&py, &root, &thread, false)
        .iter()
        .map(|turn| text(turn, "text"))
        .collect();
    println!("I14: Python reads {visible:?} after the Rust regenerate");
    assert_eq!(visible, ["again", "regenerated"]);
    assert_eq!(
        python_load(&py, &root, &thread, true).len(),
        5,
        "the superseded answer is kept"
    );
    let ours: Vec<PyValue> = store.load(&thread).iter().map(turn_value).collect();
    assert_eq!(ours, python_load(&py, &root, &thread, false));
    println!(
        "I14: the sidecar half (replay leaves out a superseded turn's sidecar items) is convo::agent_interop_tests (row E11)"
    );
}

/// The core's `touch` (spec §2.4): at the turn's start, after each approval
/// or question is resolved, and immediately before `append_answer`.
fn touched(store: &SharedThreadStore, thread: &str) {
    let _ = store.touch(thread);
}

/// An agent turn's calls into the shared store, in §2.4's order, with
/// `waiting` run while it waits on an approval and `continuing` while the
/// model goes on after it.
fn agent_turn(
    store: &SharedThreadStore,
    thread: &str,
    waiting: impl FnOnce(),
    continuing: impl FnOnce(),
) -> Result<StoredTurn, StoreError> {
    touched(store, thread);
    waiting();
    touched(store, thread);
    continuing();
    touched(store, thread);
    store.append_answer(
        thread,
        NewTurn::assistant("the agent's answer", "llamacpp:m"),
    )
}

#[test]
fn i15_a_long_turn_is_not_archived_while_python_creates_sixty_threads() {
    let py = python_or_skip!("I15");
    let dir = TempDir::new("interop-i15");
    let root = dir.path().join("store");
    let store = shared(&root);
    py.run(
        &root,
        "fill-past-cap",
        &["--count", "59", "--prefix", "older"],
    );
    let (row, _) = store.first_message("agent question", "auto").unwrap();
    let thread = row.id.clone();
    let saved = agent_turn(
        &store,
        &thread,
        || {
            // While the approval waits, Python makes 59 threads: the 59 older
            // ones are archived, and the thread is now the oldest listed.
            py.run(
                &root,
                "fill-past-cap",
                &["--count", "59", "--prefix", "during"],
            );
            assert!(listed_ids(&store).contains(&thread));
        },
        || {
            // After the approval, the 60th: the oldest listed thread goes.
            py.run(
                &root,
                "fill-past-cap",
                &["--count", "1", "--prefix", "after"],
            );
        },
    );
    println!("I15: {saved:?}");
    assert!(saved.is_ok(), "the answer was lost: {saved:?}");
    assert!(
        listed_ids(&store).contains(&thread),
        "the thread is still listed"
    );
    let turns: Vec<String> = python_load(&py, &root, &thread, false)
        .iter()
        .map(|turn| text(turn, "text"))
        .collect();
    assert_eq!(turns, ["agent question", "the agent's answer"]);
    assert_archive_whole(&root, "I15");
}

// ------------------------------------------------------------------ G4: web chats

/// A core composed as a shipped build composes it: `ChatConfig::new`'s own
/// store (the shared store, its gate open since G5), on the fixture's
/// `<globals>`.
fn shipped_core(tag: &str) -> Harness {
    Harness::with(tag, MapEnv::new(), |config: &mut ChatConfig, _, _| {
        let shipped = ChatConfig::new(
            config.state.clone(),
            config.env.clone(),
            config.local.clone(),
        );
        config.store = shipped.store;
        config.store_dir = shipped.store_dir;
    })
}

/// Row G4: web chats that Python's `history.py` wrote are listed, opened
/// with their visible last 400 turns, and their archive listed, through the
/// plain chat core on its production store, field for field as Python reads
/// them; and reading writes nothing.
#[test]
fn w1_web_chats_are_listed_and_opened_through_the_chat_core_as_python_reads_them() {
    let py = python_or_skip!("W1");
    let h = shipped_core("interop-w1");
    let root = h.fixture.state.chat_dir();
    // A long web chat: 405 turns of both kinds, with what an answer records,
    // then its last two superseded by a web regenerate.
    let turns: Vec<String> = (0..405)
        .map(|n| {
            if n % 2 == 0 {
                format!(r#"{{"id": "t{n:04}", "ts": {}.5, "role": "user", "text": "question {n} – café"}}"#, 1_790_000_000 + n)
            } else {
                format!(
                    r#"{{"id": "t{n:04}", "ts": {}.5, "role": "assistant", "text": "answer {n}", "provider": "llamacpp:m", "unsupported": ["{n}"], "error": "{}", "truncated": {}, "cancelled": {}, "prompt_tokens": {}, "completion_tokens": 7}}"#,
                    1_790_000_000 + n,
                    if n % 7 == 0 { "stopped" } else { "" },
                    if n % 5 == 0 { "true" } else { "false" },
                    if n % 11 == 0 { "true" } else { "false" },
                    if n % 3 == 0 { "null".to_owned() } else { n.to_string() },
                )
            }
        })
        .collect();
    let scratch = TempDir::new("interop-w1-turns");
    let turns_file = scratch.path().join("turns.json");
    fs::write(&turns_file, format!("[{}]", turns.join(","))).unwrap();
    let long = "webchat00001";
    let made = py.run(
        &root,
        "append",
        &["--thread", long, "--turns", turns_file.to_str().unwrap()],
    );
    assert_eq!(int(&made, "ok"), 405);
    py.run(
        &root,
        "supersede",
        &["--thread", long, "--from-turn", "t0403"],
    );
    let short = "webchat00002";
    py.run(&root, "append", &["--thread", short, "--count", "3"]);
    let before = files(&root);

    // Listed as Python lists them, in its order.
    let listing = h.runtime.block_on(h.core.threads()).unwrap();
    let python_rows = python_list(&py, &root);
    assert_eq!(listing.threads.len(), python_rows.len());
    for (ours, theirs) in listing.threads.iter().zip(&python_rows) {
        assert_eq!(ours.id, text(theirs, "id"));
        assert_eq!(ours.title, text(theirs, "title"));
        assert_eq!(ours.created.to_bits(), float(theirs, "created").to_bits());
        assert_eq!(ours.updated.to_bits(), float(theirs, "updated").to_bits());
        assert_eq!(ours.turns as i64, int(theirs, "turns"));
        assert_eq!(ours.pinned, text(theirs, "pinned_provider"));
    }
    assert_eq!(listing.store_dir, root.display().to_string());

    // Opened with the visible last 400, as `load_thread` returns them.
    let opened = h.runtime.block_on(h.core.open(long)).unwrap();
    let python_turns = python_load(&py, &root, long, false);
    println!(
        "W1: {} turns opened, Python loads {}; first {}",
        opened.turns.len(),
        python_turns.len(),
        opened.turns[0].id
    );
    assert_eq!(opened.turns.len(), 400);
    assert_eq!(python_turns.len(), 400);
    for (ours, theirs) in opened.turns.iter().zip(&python_turns) {
        let id = text(theirs, "id");
        assert_eq!(ours.id, id);
        assert_eq!(ours.ts.to_bits(), float(theirs, "ts").to_bits(), "{id}");
        let role = match ours.role {
            lattice_protocol::chat::Role::User => "user",
            lattice_protocol::chat::Role::Assistant => "assistant",
        };
        assert_eq!(role, text(theirs, "role"), "{id}");
        assert_eq!(ours.text, text(theirs, "text"), "{id}");
        assert_eq!(ours.unsupported, strings(theirs, "unsupported"), "{id}");
        assert_eq!(ours.provider, text(theirs, "provider"), "{id}");
        assert_eq!(ours.error, text(theirs, "error"), "{id}");
        assert_eq!(ours.truncated, flag(theirs, "truncated"), "{id}");
        assert_eq!(ours.cancelled, flag(theirs, "cancelled"), "{id}");
        let count = |key: &str| match field(theirs, key) {
            PyValue::Null => None,
            other => Some(pyjson::py_int(other).unwrap() as u64),
        };
        assert_eq!(ours.prompt_tokens, count("prompt_tokens"), "{id}");
        assert_eq!(ours.completion_tokens, count("completion_tokens"), "{id}");
    }
    assert_eq!(opened.turns.first().unwrap().id, "t0003");
    assert_eq!(opened.turns.last().unwrap().id, "t0402");
    let short_turns = h.runtime.block_on(h.core.open(short)).unwrap().turns;
    let python_short: Vec<String> = python_load(&py, &root, short, false)
        .iter()
        .map(|turn| text(turn, "text"))
        .collect();
    let ours_short: Vec<String> = short_turns.iter().map(|turn| turn.text.clone()).collect();
    assert_eq!(ours_short, python_short);
    assert_eq!(files(&root), before, "listing and opening wrote nothing");

    // Python's cap archives both web chats; the core lists them as Python
    // reads its archive, each `Cap`.
    py.run(
        &root,
        "fill-past-cap",
        &["--count", "60", "--prefix", "later"],
    );
    let state = py.run(&root, "archive-state", &[]);
    let python_archived = list(&state, "rows");
    assert_eq!(python_archived.len(), 2);
    let page = h.runtime.block_on(h.core.archived(0)).unwrap();
    println!("W1: archived {:?}", page.archived);
    assert_eq!(page.total, 2);
    assert!(!page.archive_unreadable);
    for row in &page.archived {
        let theirs = python_archived
            .iter()
            .find(|theirs| text(theirs, "id") == row.key.id)
            .unwrap_or_else(|| panic!("{} is not in Python's archive", row.key.id));
        assert_eq!(
            row.key.created.to_bits(),
            float(theirs, "created").to_bits()
        );
        assert_eq!(row.title, text(theirs, "title"));
        assert_eq!(row.turns as i64, int(theirs, "turns"));
        assert_eq!(
            row.evicted_at.to_bits(),
            float(theirs, "evicted_at").to_bits()
        );
        assert_eq!(row.file, text(theirs, "file"));
        assert_eq!(
            row.reason,
            lattice_protocol::conversation::ArchiveReason::Cap
        );
        assert!(!row.transcript_missing);
    }
    let order: Vec<f64> = page.archived.iter().map(|row| row.evicted_at).collect();
    assert!(
        order.windows(2).all(|pair| pair[0] >= pair[1]),
        "newest first"
    );
    let listed = h.runtime.block_on(h.core.threads()).unwrap().threads;
    assert_eq!(listed.len(), 60);
    assert!(!listed.iter().any(|row| row.id == long || row.id == short));
}

#[test]
fn i16_the_web_and_the_native_chat_honour_one_answer_lock_both_ways() {
    use lattice_protocol::RefusalKind;

    let py = python_or_skip!("I16");
    let h = Harness::new("interop-i16", MapEnv::new());
    h.fixture.install("m");
    h.text("one");
    let first = h.send(None, "q", "local", local_shown()).unwrap();
    h.events(&first.job);
    let id = first.thread.id.clone();
    let store = h.fixture.state.chat_dir();

    // The web answers: the native window refuses and shows it.
    let mut web = py.start(&store, "answer-hold", &["--thread", &id]);
    web.event("held");
    let refusal = h.send(Some(&id), "q2", "local", local_shown()).unwrap_err();
    assert_eq!(
        (refusal.kind, refusal.message.as_str()),
        (RefusalKind::Conflict, refusals::ELSEWHERE),
        "the native chat sent while the web was answering"
    );
    let opened = h.runtime.block_on(h.core.open(&id)).unwrap();
    assert!(opened.answering_elsewhere && opened.job.is_none());
    web.send("release");
    let held = web.finish();
    println!("I16: the web's answer: {}", pyjson::dumps(&held));
    assert!(flag(&held, "finished"));
    let opened = h.runtime.block_on(h.core.open(&id)).unwrap();
    assert!(!opened.answering_elsewhere, "the web let go after saving");

    // The native side answers: the web refuses before it writes.
    let lock = match answer_lock(&h.fixture.state.chat_locks_dir(), &id) {
        Locked::Held(lock) => lock,
        _ => panic!("the native side could not take a free answer lock"),
    };
    let refused = py.run(&store, "answer-try", &["--thread", &id]);
    println!("I16: while native held it: {}", pyjson::dumps(&refused));
    assert!(
        flag(&refused, "refused"),
        "the web answered under the native lock"
    );
    assert!(flag(&refused, "elsewhere"));
    assert!(
        !flag(&refused, "prepared"),
        "the web wrote its question first"
    );
    drop(lock);
    // The positive control: released, the web answers.
    let taken = py.run(&store, "answer-try", &["--thread", &id]);
    println!("I16: after the release: {}", pyjson::dumps(&taken));
    assert!(!flag(&taken, "refused") && flag(&taken, "prepared"));
}
