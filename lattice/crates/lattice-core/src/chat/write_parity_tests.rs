//! The shared chat store's write path against goldens recorded from the real
//! Python (the chat core's spec row G2, §16.2): `tests/parity/chat/`
//! `store_ops.json`, `evicted_ops.json` and `archive_row.json`, written by
//! `tools/lattice_native_parity.py` from `history.py`.
//!
//! Each scenario starts from the files it names (a base, then its own), and
//! each step runs one operation on a [`SharedThreadStore`] whose gate is
//! opened through the test-only constructor, fed exactly the clock values and
//! ids Python was given (in order: reading the clock or drawing an id more or
//! fewer times than Python fails the step), with the faults the generator
//! injected into `_replace`. After every step:
//! - the result is Python's;
//! - every file under the store, read with CRLF as LF (`.store.lock` aside,
//!   the OS lock being row G3's), equals Python's, byte for byte;
//! - the index files this port wrote end their lines with the platform's
//!   line end (S2);
//! - nothing was lost (NF1, ND5): a file that left its place is, byte for
//!   byte, somewhere else in the store, and no other file is gone.

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use super::archive::Fault;
use super::store::{IdSource, SharedThreadStore};
use super::transcript::{NewTurn, StoreError, TranscriptStore};
use crate::clock::Clock;
use crate::testkit::TempDir;

/// The line end Python's text mode writes here (`os.linesep`), stated in the
/// test, not taken from the code under test.
pub(super) const PLATFORM_LINE_END: &str = if cfg!(windows) { "\r\n" } else { "\n" };

fn golden(name: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("chat")
        .join(name);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A file's recorded contents: text (CRLF read as LF), hex, or a folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Entry {
    Text(String),
    Bytes(Vec<u8>),
    Folder,
}

fn entry_of(spec: &Value) -> Entry {
    if spec.get("directory").is_some() {
        return Entry::Folder;
    }
    if let Some(hex) = spec.get("hex").and_then(Value::as_str) {
        return Entry::Bytes(
            (0..hex.len())
                .step_by(2)
                .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).unwrap())
                .collect(),
        );
    }
    Entry::Text(spec["text"].as_str().unwrap().to_owned())
}

fn put(root: &Path, name: &str, entry: &Entry) {
    let path = root.join(name);
    match entry {
        Entry::Folder => fs::create_dir_all(&path).unwrap(),
        Entry::Text(text) => {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, text).unwrap();
        }
        Entry::Bytes(bytes) => {
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, bytes).unwrap();
        }
    }
}

/// Every file and folder under `root` with its raw bytes, by relative path.
pub(super) fn raw_files(root: &Path) -> BTreeMap<String, Option<Vec<u8>>> {
    let mut out = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(folder) = pending.pop() {
        for item in fs::read_dir(&folder).unwrap() {
            let path = item.unwrap().path();
            let name = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if name == ".store.lock" {
                continue;
            }
            if path.is_dir() {
                out.insert(name, None);
                pending.push(path);
            } else {
                out.insert(name, Some(fs::read(&path).unwrap()));
            }
        }
    }
    out
}

fn normalised(raw: &BTreeMap<String, Option<Vec<u8>>>) -> BTreeMap<String, Entry> {
    raw.iter()
        .map(|(name, bytes)| {
            let entry = match bytes {
                None => Entry::Folder,
                Some(bytes) => match std::str::from_utf8(bytes) {
                    Ok(text) => Entry::Text(text.replace("\r\n", "\n")),
                    Err(_) => Entry::Bytes(bytes.clone()),
                },
            };
            (name.clone(), entry)
        })
        .collect()
}

/// Python's own temporaries in the store (`index.tmp`, `<id>.jsonl.tmp`):
/// the only names that may leave their place without their bytes, because
/// they are replaced into place (ND3).
fn is_python_temporary(name: &str) -> bool {
    name.rsplit('/').next() == Some("index.tmp") || name.ends_with(".jsonl.tmp")
}

/// NF1 / ND5: what left its place in a step is, byte for byte, elsewhere;
/// only a named temporary may go without.
pub(super) fn assert_nothing_lost(
    before: &BTreeMap<String, Option<Vec<u8>>>,
    after: &BTreeMap<String, Option<Vec<u8>>>,
    context: &str,
) {
    for (name, bytes) in before {
        if after.contains_key(name) || is_python_temporary(name) {
            continue;
        }
        let bytes = bytes
            .as_ref()
            .unwrap_or_else(|| panic!("{context}: the folder {name} is gone"));
        assert!(
            after.values().any(|other| other.as_ref() == Some(bytes)),
            "{context}: {name} is gone and its bytes are nowhere in the store"
        );
    }
}

/// A queue a step fills with what Python was given, and the store drains.
#[derive(Clone, Default)]
struct Fed<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    step: Arc<Mutex<String>>,
}

impl<T: Clone + std::fmt::Debug> Fed<T> {
    fn fill(&self, values: Vec<T>, step: &str) {
        let mut queue = self.queue.lock().unwrap();
        assert!(queue.is_empty(), "left over from the last step: {queue:?}");
        queue.extend(values);
        step.clone_into(&mut self.step.lock().unwrap());
    }

    fn take(&self, what: &str) -> T {
        let next = self.queue.lock().unwrap().pop_front();
        next.unwrap_or_else(|| {
            panic!(
                "{}: the native writer used more {what} than Python did",
                self.step.lock().unwrap()
            )
        })
    }

    fn left(&self) -> Vec<T> {
        self.queue.lock().unwrap().drain(..).collect()
    }
}

fn strs(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect()
        })
        .unwrap_or_default()
}

fn new_turn(op: &Value) -> NewTurn {
    let text = |key: &str| op.get(key).and_then(Value::as_str).unwrap_or("").to_owned();
    let flag = |key: &str| op.get(key).and_then(Value::as_bool).unwrap_or(false);
    let count = |key: &str| op.get(key).and_then(Value::as_i64);
    NewTurn {
        id: text("turn_id"),
        ts: op.get("ts").and_then(Value::as_f64).unwrap_or(0.0),
        role: op
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
            .to_owned(),
        text: text("text"),
        provider: text("provider"),
        error: text("error"),
        truncated: flag("truncated"),
        cancelled: flag("cancelled"),
        prompt_tokens: count("prompt_tokens"),
        completion_tokens: count("completion_tokens"),
        ..NewTurn::default()
    }
}

/// Whether a native outcome is the one Python recorded.
fn same_result(native: &Result<Value, StoreError>, python: &Value) -> bool {
    match native {
        Ok(value) => value == python,
        // Python raised (`{"raised": ...}`), `append` caught it (`false`), the
        // composed Archive kept the thread (`"kept"`): nothing was saved.
        Err(StoreError::NotSaved) => {
            python.get("raised").is_some()
                || *python == Value::Bool(false)
                || python.as_str() == Some("kept")
                || python.get("saved") == Some(&Value::Bool(false))
        }
        Err(StoreError::ThreadGone) => python.as_str() == Some("thread_gone"),
        Err(StoreError::NotFound) => python.as_str() == Some("not_found"),
        Err(_) => false,
    }
}

fn run(store: &SharedThreadStore, root: &Path, op: &Value) -> Result<Value, StoreError> {
    let id = || op["id"].as_str().unwrap();
    let ids = |turns: Vec<super::store::StoredTurn>| {
        Value::from(turns.into_iter().map(|t| t.id).collect::<Vec<_>>())
    };
    Ok(match op["op"].as_str().unwrap() {
        "first_message" => {
            let (row, _) =
                store.first_message(op["text"].as_str().unwrap(), op["pin"].as_str().unwrap())?;
            serde_json::json!({"saved": true, "id": row.id})
        }
        "append" => {
            store.append(id(), new_turn(op))?;
            Value::Bool(true)
        }
        "append_answer" => {
            store.append_answer(id(), new_turn(op))?;
            Value::Bool(true)
        }
        "rename" => Value::Bool(store.rename(id(), op["title"].as_str().unwrap())?),
        "pin" => Value::Bool(store.pin(id(), op["pin"].as_str().unwrap())?),
        "touch" => Value::Bool(store.touch(id())?),
        "supersede" => Value::from(store.supersede(id(), op["from"].as_str().unwrap())?),
        "load" => ids(store.load(id())),
        "recent" => ids(store.recent(id())),
        "archive" => {
            store.archive(id())?;
            Value::from("archived")
        }
        "put" => {
            put(root, op["path"].as_str().unwrap(), &entry_of(&op["file"]));
            Value::Null
        }
        other => panic!("no such operation: {other}"),
    })
}

fn fault(rule: &Value) -> Fault {
    let text = |key: &str| rule.get(key).and_then(Value::as_str).map(str::to_owned);
    Fault {
        src_suffix: text("src_suffix"),
        dst: text("dst"),
        dst_prefix: text("dst_prefix"),
    }
}

/// Replay every scenario of one golden; returns how many steps ran.
fn replay(name: &str) -> usize {
    let document = golden(name);
    let bases = &document["bases"];
    let mut steps_run = 0;
    for scenario in document["scenarios"].as_array().unwrap() {
        let title = scenario["name"].as_str().unwrap();
        let dir = TempDir::new("chat-write-parity");
        let root = dir.path().join("store");
        fs::create_dir_all(&root).unwrap();
        let mut initial: BTreeMap<String, Entry> = BTreeMap::new();
        if let Some(base) = scenario.get("base").and_then(Value::as_str) {
            for (file, spec) in bases[base].as_object().unwrap() {
                initial.insert(file.clone(), entry_of(spec));
            }
        }
        for (file, spec) in scenario["files"].as_object().unwrap() {
            initial.insert(file.clone(), entry_of(spec));
        }
        for (file, entry) in &initial {
            put(&root, file, entry);
        }
        // The folders the files sit in, as Python's snapshot after the same
        // materialisation has them.
        let mut expected = normalised(&raw_files(&root));
        let clock_fed: Fed<f64> = Fed::default();
        let ids_fed: Fed<String> = Fed::default();
        let clock: Clock = {
            let fed = clock_fed.clone();
            Arc::new(move || fed.take("clock readings"))
        };
        let ids: IdSource = {
            let fed = ids_fed.clone();
            Arc::new(move || fed.take("ids"))
        };
        let store = SharedThreadStore::with_gate(&root, ids, true).clocked(clock);
        for (number, step) in scenario["steps"].as_array().unwrap().iter().enumerate() {
            let op = &step["op"];
            let context = format!("{name}: {title}: step {number} ({})", op["op"]);
            clock_fed.fill(
                strs(&step["clock"])
                    .iter()
                    .map(|value| value.parse::<f64>().unwrap())
                    .collect(),
                &context,
            );
            ids_fed.fill(strs(&step["ids"]), &context);
            store.set_faults(
                step.get("faults")
                    .and_then(Value::as_array)
                    .map(|rules| rules.iter().map(fault).collect())
                    .unwrap_or_default(),
            );
            let before = raw_files(&root);
            let result = run(&store, &root, op);
            store.set_faults(Vec::new());
            assert!(
                same_result(&result, &step["result"]),
                "{context}: native {result:?}, Python {}",
                step["result"]
            );
            assert_eq!(
                clock_fed.left(),
                Vec::<f64>::new(),
                "{context}: clock readings Python made and native did not"
            );
            assert_eq!(
                ids_fed.left(),
                Vec::<String>::new(),
                "{context}: ids Python drew and native did not"
            );
            for (file, change) in step["changes"].as_object().unwrap() {
                if change.is_null() {
                    expected.remove(file);
                } else {
                    expected.insert(file.clone(), entry_of(change));
                }
            }
            let after = raw_files(&root);
            let found = normalised(&after);
            for file in expected.keys().chain(found.keys()) {
                assert_eq!(
                    found.get(file),
                    expected.get(file),
                    "{context}: {file} differs from what Python left"
                );
            }
            for (file, bytes) in &after {
                let changed = step["changes"].get(file.as_str()).is_some();
                if changed
                    && op["op"] != "put"
                    && (file == "index.json" || file == "evicted/index.json")
                    && let Some(bytes) = bytes
                {
                    let text = String::from_utf8_lossy(bytes);
                    let line_ends = text.matches('\n').count();
                    assert_eq!(
                        text.matches(PLATFORM_LINE_END).count(),
                        line_ends,
                        "{context}: {file} has a line end that is not the platform's"
                    );
                }
            }
            assert_nothing_lost(&before, &after, &context);
            steps_run += 1;
        }
    }
    steps_run
}

#[test]
fn the_store_s_writes_leave_what_python_s_leave() {
    let steps = replay("store_ops.json");
    println!("store_ops: {steps} steps");
    assert!(steps >= 50);
}

#[test]
fn the_cap_archives_as_python_archives_failures_included() {
    let steps = replay("evicted_ops.json");
    println!("evicted_ops: {steps} steps");
    assert!(steps >= 25);
}

#[test]
fn the_reader_s_archive_writes_what_python_s_eviction_writes() {
    let steps = replay("archive_row.json");
    println!("archive_row: {steps} steps");
    assert!(steps >= 12);
}
