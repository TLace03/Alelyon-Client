//! The run store: what a run leaves on disk.
//!
//! Under `<globals>/lattice_native/runs/`, two files per run, named by the run's
//! id (16 lowercase hexadecimal characters, checked before any path is built):
//!
//! - `<id>.run.json`: the run's [`RunSummary`], one JSON object, replaced
//!   atomically (written to a temporary file in the same directory, named for
//!   this process and this write (`<id>.run.json.<pid>-<n>.tmp`), then renamed
//!   over the old file, with a few retries because Windows refuses a rename
//!   while an indexer or antivirus scanner holds the target open; a reader's
//!   open is retried the same way, because Windows also refuses, for a moment,
//!   an open of a name that a rename is replacing (error 5));
//! - `<id>.events.jsonl`: the run's [`RunEvent`]s, one JSON object per line,
//!   appended in `seq` order, with every `Delta` left out (streamed text lives
//!   in memory only; the complete text is in the `Message` and `Result` events);
//! - `.lattice-runs.lock`: held (an exclusive `File::try_lock`) by the one
//!   process that records runs here, for as long as it lives ([`WriterLock`]).
//!   A second process cannot take it and is a reader: the manager lists and
//!   follows the runs it finds and never rewrites them.
//!
//! Reading is as careful as writing is: a summary over 1 MiB is ignored, a line
//! over 4 MiB or one that is not an event (a torn last line after a crash, say)
//! is skipped, an event whose `seq` does not rise, or rises by more than any run
//! could have (a hostile file's jump to the end of the range), is skipped, and at
//! most 20,008 events are read from one file. Nothing a file holds can make a
//! read allocate without bound or overflow a counter, and nothing here deletes a
//! run.
//!
//! Invariants: an id that is not [`is_run_id`] never reaches the file system;
//! the directory is created when something is first written (or the writer lock
//! is first tried), never at construction; a failure is returned to the caller
//! (the manager keeps the run going in memory and counts the failure) and is
//! never silent inside this module.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use lattice_protocol::{RunEvent, RunSummary, is_run_id};

/// Events kept per run, apart from the handful that end it.
pub const MAX_PERSISTED_EVENTS: usize = 20_000;
/// What may be read from one events file: the cap, and room for the events that
/// end a run past it.
const MAX_EVENTS_READ: usize = MAX_PERSISTED_EVENTS + 8;
/// The most an event's `seq` may rise over the one before it. Live text deltas
/// take a `seq` and are not saved, so a saved run has gaps, but never more than
/// the deltas a run may keep (50,000) plus the handful of events around them. A
/// line that rises by more is not from a run this program recorded, and a `seq`
/// taken at face value near `u64::MAX` would overflow the counter that hands out
/// the next one.
const MAX_SEQ_GAP: u64 = 1_000_000;
const MAX_SUMMARY_BYTES: u64 = 1024 * 1024;
const MAX_LINE_BYTES: u64 = 4 * 1024 * 1024;
/// How many `*.run.json` files are examined when loading, newest first: a store
/// with more than this many has its oldest runs left unread.
const MAX_SCANNED: usize = 2_000;
/// How long a write keeps trying to rename its file over the target while
/// Windows refuses the rename (a reader or a scanner has the target open), and
/// how long a reader keeps trying an open that a rename or a scanner refuses.
/// Measured on 2026-10-09 under a release build's load beside lattice-core's
/// own tests: renames refused for up to 63 ms, and a summary being replaced
/// read as missing (error 2) for up to 263 ms; the 5 x 20 ms these had before
/// ran out under heavier load. A second is a bound on a refusal, not a speed:
/// an unrefused rename or open does not wait at all.
const REFUSAL_WAIT: Duration = Duration::from_secs(1);
const RENAME_PAUSE: Duration = Duration::from_millis(20);
const OPEN_PAUSE: Duration = Duration::from_millis(20);
/// The file the recording process holds locked.
const LOCK_FILE: &str = ".lattice-runs.lock";

/// Numbers the temporary files of this process, so that no two writes (of one
/// run, from two threads, or from two processes) ever share one.
static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// A directory of runs.
#[derive(Clone, Debug)]
pub struct RunStore {
    dir: PathBuf,
    /// How long a refused rename or open is tried again ([`REFUSAL_WAIT`]).
    refusal_wait: Duration,
}

/// The right to record runs in a directory: held, through this value, by exactly
/// one process at a time. Dropping it (or the process ending, however it ends)
/// gives the right up.
#[derive(Debug)]
pub struct WriterLock {
    _file: File,
}

fn invalid_id() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "not a run id")
}

/// A temporary file of this run store that a dead writer may have left behind:
/// `<id>.run.json.tmp` (the name before writes were numbered), `<id>.run.json.<n>.tmp`,
/// `<id>.events.jsonl.<n>.tmp`.
fn is_leftover_temporary(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".tmp") else {
        return false;
    };
    let Some((id, rest)) = stem.split_once('.') else {
        return false;
    };
    is_run_id(id)
        && (rest == "run.json"
            || rest.starts_with("run.json.")
            || rest.starts_with("events.jsonl."))
}

impl RunStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            refusal_wait: REFUSAL_WAIT,
        }
    }

    /// This store, trying a refused rename or open again for `wait` instead
    /// of [`REFUSAL_WAIT`]: for a test that must not depend on how long
    /// something outside the process holds a file.
    #[cfg(test)]
    pub(crate) fn with_refusal_wait(mut self, wait: Duration) -> Self {
        self.refusal_wait = wait;
        self
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path_of(&self, id: &str, suffix: &str) -> io::Result<PathBuf> {
        if !is_run_id(id) {
            return Err(invalid_id());
        }
        Ok(self.dir.join(format!("{id}{suffix}")))
    }

    pub fn summary_path(&self, id: &str) -> io::Result<PathBuf> {
        self.path_of(id, ".run.json")
    }

    pub fn events_path(&self, id: &str) -> io::Result<PathBuf> {
        self.path_of(id, ".events.jsonl")
    }

    /// Try to become the one process that records runs in this directory.
    ///
    /// `Ok(Some(_))`: this process now records them, for as long as it keeps the
    /// lock, and any temporary file a dead writer left behind is removed (no
    /// other writer exists to be using it). `Ok(None)`: another process holds
    /// it, and this one must only read. `Err(_)`: the lock could not be tried at
    /// all (the directory cannot be made, the file system refuses locks), which
    /// is not evidence of another writer.
    pub fn try_lock_writer(&self) -> io::Result<Option<WriterLock>> {
        fs::create_dir_all(&self.dir)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(self.dir.join(LOCK_FILE))?;
        match file.try_lock() {
            Ok(()) => {
                self.remove_leftover_temporaries();
                Ok(Some(WriterLock { _file: file }))
            }
            Err(fs::TryLockError::WouldBlock) => Ok(None),
            Err(fs::TryLockError::Error(error)) => Err(error),
        }
    }

    fn remove_leftover_temporaries(&self) {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return;
        };
        for entry in entries.filter_map(Result::ok) {
            if entry
                .file_name()
                .to_str()
                .is_some_and(is_leftover_temporary)
            {
                let _ = fs::remove_file(entry.path());
            }
        }
    }

    /// Write `bytes` to a temporary file of its own, in this directory, and
    /// rename it over `target`. `durable` flushes it to disk first.
    fn replace_file(
        &self,
        target: &Path,
        id: &str,
        what: &str,
        bytes: &[u8],
        durable: bool,
    ) -> io::Result<()> {
        fs::create_dir_all(&self.dir)?;
        let temporary = self.dir.join(format!(
            "{id}.{what}.{}-{}.tmp",
            std::process::id(),
            TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        {
            let mut file = File::create(&temporary)?;
            file.write_all(bytes)?;
            if durable {
                file.sync_all()?;
            }
        }
        let started = Instant::now();
        loop {
            match fs::rename(&temporary, target) {
                Ok(()) => return Ok(()),
                Err(_) if started.elapsed() < self.refusal_wait => std::thread::sleep(RENAME_PAUSE),
                Err(error) => {
                    let _ = fs::remove_file(&temporary);
                    return Err(error);
                }
            }
        }
    }

    /// Replace the run's summary atomically. `durable` also flushes the new file
    /// to disk before the rename (the first and last write of a run).
    pub fn write_summary(&self, summary: &RunSummary, durable: bool) -> io::Result<()> {
        let target = self.summary_path(&summary.id)?;
        let bytes = serde_json::to_vec(summary).map_err(io::Error::other)?;
        self.replace_file(&target, &summary.id, "run.json", &bytes, durable)
    }

    /// Replace the run's events file with exactly these events (one line each),
    /// atomically: used to take a secret out of a run that a guardrail refused.
    /// The file must not be open for appending.
    pub fn rewrite_events(&self, id: &str, events: &[RunEvent]) -> io::Result<()> {
        let target = self.events_path(id)?;
        let mut bytes = Vec::new();
        for event in events {
            bytes.extend_from_slice(&Self::encode_event(event)?);
        }
        self.replace_file(&target, id, "events.jsonl", &bytes, true)
    }

    /// The length in bytes of the run's events file, when there is one: what a
    /// reader compares to know the file has grown since it last read it.
    pub fn events_len(&self, id: &str) -> Option<u64> {
        let path = self.events_path(id).ok()?;
        fs::metadata(path).ok().map(|meta| meta.len())
    }

    /// The run's events file, opened for appending (created if new).
    pub fn open_events(&self, id: &str) -> io::Result<File> {
        let path = self.events_path(id)?;
        fs::create_dir_all(&self.dir)?;
        OpenOptions::new().create(true).append(true).open(path)
    }

    /// One event as one line (JSON and a newline): the bytes [`RunStore::append_line`] writes.
    pub fn encode_event(event: &RunEvent) -> io::Result<Vec<u8>> {
        let mut line = serde_json::to_vec(event).map_err(io::Error::other)?;
        line.push(b'\n');
        Ok(line)
    }

    /// Append an encoded line, whole, in one write.
    pub fn append_line(file: &mut File, line: &[u8]) -> io::Result<()> {
        file.write_all(line)
    }

    /// Append one event as one line.
    pub fn append_event(file: &mut File, event: &RunEvent) -> io::Result<()> {
        Self::append_line(file, &Self::encode_event(event)?)
    }

    /// One run's summary, if its file is a readable summary of that run.
    pub fn read_summary(&self, id: &str) -> Option<RunSummary> {
        let path = self.summary_path(id).ok()?;
        read_summary_file(&path, id, self.refusal_wait)
    }

    /// The `limit` newest runs by `created_at` (ties by id), unreadable and
    /// misnamed files skipped. A missing directory is an empty store.
    pub fn load_summaries(&self, limit: usize) -> Vec<RunSummary> {
        let Ok(entries) = fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut files: Vec<(std::time::SystemTime, String, PathBuf)> = entries
            .filter_map(Result::ok)
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                let id = name.strip_suffix(".run.json")?.to_owned();
                if !is_run_id(&id) {
                    return None;
                }
                let modified = entry.metadata().ok()?.modified().ok()?;
                Some((modified, id, entry.path()))
            })
            .collect();
        files.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        files.truncate(MAX_SCANNED);
        let mut summaries: Vec<RunSummary> = files
            .into_iter()
            .filter_map(|(_, id, path)| read_summary_file(&path, &id, self.refusal_wait))
            .collect();
        summaries.sort_by(|a, b| {
            b.created_at
                .total_cmp(&a.created_at)
                .then_with(|| a.id.cmp(&b.id))
        });
        summaries.truncate(limit);
        summaries
    }

    /// The run's recorded events, in `seq` order. A missing file is no events.
    pub fn read_events(&self, id: &str) -> Vec<RunEvent> {
        let Ok(path) = self.events_path(id) else {
            return Vec::new();
        };
        let Ok(file) = open_for_reading(&path, false, self.refusal_wait) else {
            return Vec::new();
        };
        let mut reader = BufReader::new(file);
        let mut events: Vec<RunEvent> = Vec::new();
        let mut line = Vec::new();
        while events.len() < MAX_EVENTS_READ {
            line.clear();
            let read = match (&mut reader)
                .take(MAX_LINE_BYTES + 1)
                .read_until(b'\n', &mut line)
            {
                Ok(0) | Err(_) => break,
                Ok(read) => read,
            };
            if read as u64 > MAX_LINE_BYTES && line.last() != Some(&b'\n') {
                // A line longer than any event: drop the rest of it.
                let mut discard = Vec::new();
                loop {
                    discard.clear();
                    match (&mut reader)
                        .take(MAX_LINE_BYTES)
                        .read_until(b'\n', &mut discard)
                    {
                        Ok(0) | Err(_) => break,
                        Ok(_) if discard.last() == Some(&b'\n') => break,
                        Ok(_) => {}
                    }
                }
                continue;
            }
            let Ok(event) = serde_json::from_slice::<RunEvent>(line.trim_ascii()) else {
                continue;
            };
            // Rising, and by a step a real run could have taken, so that no value
            // a file holds can overflow the counter that hands out the next `seq`.
            // (The subtraction follows the comparison: it cannot underflow.)
            let floor = events.last().map_or(0, |last| last.seq);
            if event.seq > floor && event.seq - floor <= MAX_SEQ_GAP {
                events.push(event);
            }
        }
        events
    }
}

/// Open `path` to read it, trying again for at most `wait` ([`REFUSAL_WAIT`]) while
/// Windows refuses the open for a moment: an access refusal (5), which an open
/// meets while a rename replaces the file, or a sharing violation (32), which a
/// scanner holding it causes. `replaced` also tries again on a missing file
/// (2): a file that is only ever replaced by a rename, never removed (a run's
/// summary), can read as missing while one replaces it. A file that may
/// simply not exist yet (a run's events) is not tried again when missing.
fn open_for_reading(path: &Path, replaced: bool, wait: Duration) -> io::Result<File> {
    let started = Instant::now();
    loop {
        match File::open(path) {
            Err(error)
                if matches!(error.raw_os_error(), Some(5 | 32))
                    || (replaced && error.kind() == io::ErrorKind::NotFound) =>
            {
                if started.elapsed() >= wait {
                    return Err(error);
                }
                std::thread::sleep(OPEN_PAUSE);
            }
            other => return other,
        }
    }
}

fn read_summary_file(path: &Path, id: &str, wait: Duration) -> Option<RunSummary> {
    let mut bytes = Vec::new();
    open_for_reading(path, true, wait)
        .ok()?
        .take(MAX_SUMMARY_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_SUMMARY_BYTES {
        return None;
    }
    let summary: RunSummary = serde_json::from_slice(&bytes).ok()?;
    (summary.id == id).then_some(summary)
}

#[cfg(test)]
mod tests {
    use lattice_protocol::{EndStatus, Locality, RunEventKind, RunStatus, SpanRecord, Usage};
    use serde_json::json;

    use super::*;
    use crate::testkit::TempDir;

    fn summary(id: &str, created_at: f64) -> RunSummary {
        RunSummary {
            id: id.to_owned(),
            task: "What is 6 * 7?".into(),
            agent: "lattice-assistant".into(),
            agent_label: "Lattice assistant".into(),
            model: "dev:scripted".into(),
            model_label: "Development model (scripted)".into(),
            locality: Locality::Local,
            status: RunStatus::Running,
            created_at,
            updated_at: created_at,
            ended_at: None,
            trace_id: "trace_0123456789abcdef0123456789abcdef".into(),
            usage: None,
            output: None,
            error: None,
            spans: 0,
        }
    }

    fn event(seq: u64, kind: RunEventKind) -> RunEvent {
        RunEvent {
            seq,
            at: 1_790_000_000.25 + seq as f64,
            kind,
        }
    }

    fn span(order: u64) -> SpanRecord {
        SpanRecord {
            order,
            id: format!("span_{order:024x}"),
            trace_id: "trace_0123456789abcdef0123456789abcdef".into(),
            parent_id: None,
            started_at: "2026-09-30T05:21:05.123456+00:00".into(),
            ended_at: Some("2026-09-30T05:21:06.000000+00:00".into()),
            span_data: json!({"type": "function", "name": "calculate", "input": "{\"expression\":\"6 * 7\"}", "output": "42"}),
            error: None,
        }
    }

    const ID_A: &str = "00000000000000aa";
    const ID_B: &str = "00000000000000bb";
    const ID_C: &str = "00000000000000cc";

    #[test]
    fn a_summary_round_trips_and_replaces_the_old_one() {
        let dir = TempDir::new("store-summary");
        let store = RunStore::new(dir.path().join("runs"));
        assert!(!store.dir().exists(), "constructing creates nothing");
        let mut one = summary(ID_A, 10.0);
        store.write_summary(&one, true).unwrap();
        assert_eq!(store.read_summary(ID_A), Some(one.clone()));
        one.status = RunStatus::Completed;
        one.usage = Some(Usage {
            requests: 2,
            input_tokens: 10,
            output_tokens: 5,
            total_tokens: 15,
        });
        one.output = Some("done".into());
        store.write_summary(&one, false).unwrap();
        assert_eq!(store.read_summary(ID_A), Some(one));
        let names: Vec<String> = fs::read_dir(store.dir())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(
            names,
            [format!("{ID_A}.run.json")],
            "no temporary file is left behind"
        );
    }

    #[test]
    fn ids_that_are_not_run_ids_never_reach_the_file_system() {
        let dir = TempDir::new("store-ids");
        let store = RunStore::new(dir.path().join("runs"));
        for bad in [
            "",
            "../../etc/passwd",
            "0123456789ABCDEF",
            "zz",
            "0123456789abcde",
            "0123456789abcdef0",
        ] {
            assert!(store.summary_path(bad).is_err(), "{bad}");
            assert!(store.events_path(bad).is_err(), "{bad}");
            assert!(store.open_events(bad).is_err(), "{bad}");
            assert!(store.read_summary(bad).is_none(), "{bad}");
            assert!(store.read_events(bad).is_empty(), "{bad}");
            assert!(
                store.write_summary(&summary(bad, 1.0), true).is_err(),
                "{bad}"
            );
        }
        assert!(
            !store.dir().exists(),
            "a refused id creates nothing, not even the directory"
        );
    }

    #[test]
    fn events_append_one_per_line_and_read_back_in_order() {
        let dir = TempDir::new("store-events");
        let store = RunStore::new(dir.path().join("runs"));
        let events = vec![
            event(
                1,
                RunEventKind::TraceStart {
                    trace_id: "trace_x".into(),
                    workflow_name: "Lattice assistant".into(),
                    at: "2026-09-30T05:21:05.000000+00:00".into(),
                },
            ),
            event(
                2,
                RunEventKind::SpanStart {
                    span: SpanRecord {
                        ended_at: None,
                        ..span(1)
                    },
                },
            ),
            event(
                4,
                RunEventKind::Agent {
                    name: "Lattice assistant".into(),
                },
            ),
            event(
                5,
                RunEventKind::Message {
                    agent: "Lattice assistant".into(),
                    text: "line one\nline two \u{1F600}".into(),
                },
            ),
            event(
                6,
                RunEventKind::ToolCall {
                    agent: "a".into(),
                    name: "calculate".into(),
                    call_id: "c1".into(),
                    arguments: "{}".into(),
                },
            ),
            event(
                7,
                RunEventKind::ToolOutput {
                    agent: "a".into(),
                    call_id: "c1".into(),
                    output: "42".into(),
                },
            ),
            event(8, RunEventKind::HandoffRequested { agent: "a".into() }),
            event(
                9,
                RunEventKind::Handoff {
                    from: "a".into(),
                    to: "b".into(),
                },
            ),
            event(
                10,
                RunEventKind::Reasoning {
                    agent: "a".into(),
                    text: "hm".into(),
                },
            ),
            event(
                11,
                RunEventKind::Guardrail {
                    name: "secrets_stay_local".into(),
                    message: "no".into(),
                },
            ),
            event(12, RunEventKind::SpanEnd { span: span(1) }),
            event(
                13,
                RunEventKind::Result {
                    output: "42".into(),
                    usage: Some(Usage {
                        requests: 1,
                        input_tokens: 1,
                        output_tokens: 2,
                        total_tokens: 3,
                    }),
                    turns: 1,
                    last_agent: "a".into(),
                },
            ),
            event(
                14,
                RunEventKind::Error {
                    message: "no".into(),
                },
            ),
            event(
                15,
                RunEventKind::TraceEnd {
                    trace_id: "trace_x".into(),
                    at: "2026-09-30T05:21:06.000000+00:00".into(),
                },
            ),
            event(
                16,
                RunEventKind::End {
                    status: EndStatus::Completed,
                },
            ),
        ];
        let mut file = store.open_events(ID_A).unwrap();
        for e in &events {
            RunStore::append_event(&mut file, e).unwrap();
        }
        drop(file);
        assert_eq!(
            store.read_events(ID_A),
            events,
            "every event kind survives a round trip, gaps in seq included"
        );
        let text = fs::read_to_string(store.events_path(ID_A).unwrap()).unwrap();
        assert_eq!(text.lines().count(), events.len());
        assert!(text.lines().all(|l| l.starts_with('{') && l.ends_with('}')));
        assert!(
            store.read_events(ID_B).is_empty(),
            "a missing file is no events"
        );
    }

    #[test]
    fn reading_skips_torn_out_of_order_and_oversize_lines() {
        let dir = TempDir::new("store-torn");
        let store = RunStore::new(dir.path().join("runs"));
        fs::create_dir_all(store.dir()).unwrap();
        let good = |seq: u64| {
            serde_json::to_string(&event(seq, RunEventKind::Agent { name: "a".into() })).unwrap()
        };
        let mut text = String::new();
        text.push_str(&good(1));
        text.push('\n');
        text.push_str("{\"seq\": 2, \"broken\n");
        text.push_str("not json at all\n");
        text.push('\n');
        text.push_str(&good(3));
        text.push('\n');
        text.push_str(&good(3));
        text.push('\n');
        text.push_str(&good(2));
        text.push('\n');
        text.push_str(&good(5));
        text.push('\n');
        text.push_str(&"x".repeat(MAX_LINE_BYTES as usize + 10));
        text.push('\n');
        text.push_str(&good(6));
        text.push('\n');
        text.push_str(&good(7)[..20]); // a torn last line, no newline
        fs::write(store.events_path(ID_A).unwrap(), text).unwrap();
        let seqs: Vec<u64> = store.read_events(ID_A).iter().map(|e| e.seq).collect();
        assert_eq!(seqs, [1, 3, 5, 6]);
    }

    #[test]
    fn at_most_the_cap_and_a_few_more_events_are_read() {
        let dir = TempDir::new("store-cap");
        let store = RunStore::new(dir.path().join("runs"));
        let mut file = store.open_events(ID_A).unwrap();
        for seq in 1..=(MAX_EVENTS_READ as u64 + 50) {
            RunStore::append_event(
                &mut file,
                &event(seq, RunEventKind::Agent { name: "a".into() }),
            )
            .unwrap();
        }
        drop(file);
        assert_eq!(store.read_events(ID_A).len(), MAX_EVENTS_READ);
    }

    #[test]
    fn the_newest_runs_are_loaded_and_junk_is_ignored() {
        let dir = TempDir::new("store-load");
        let store = RunStore::new(dir.path().join("runs"));
        assert!(
            store.load_summaries(10).is_empty(),
            "a missing directory is an empty store"
        );
        store.write_summary(&summary(ID_A, 10.0), true).unwrap();
        store.write_summary(&summary(ID_B, 30.0), true).unwrap();
        store.write_summary(&summary(ID_C, 20.0), true).unwrap();
        fs::write(store.dir().join("notes.txt"), "x").unwrap();
        fs::write(store.dir().join("0123456789abcdef.run.json"), "not json").unwrap();
        fs::write(
            store.dir().join("ffffffffffffffff.run.json"),
            serde_json::to_vec(&summary(ID_A, 99.0)).unwrap(),
        )
        .unwrap();
        fs::write(store.dir().join("SHOUTING.run.json"), "{}").unwrap();
        fs::write(store.dir().join(format!("{ID_A}.events.jsonl")), "").unwrap();
        fs::write(
            store.dir().join("eeeeeeeeeeeeeeee.run.json.tmp"),
            "half a wri",
        )
        .unwrap();
        let all: Vec<String> = store.load_summaries(10).into_iter().map(|s| s.id).collect();
        assert_eq!(
            all,
            [ID_B, ID_C, ID_A],
            "newest first; the wrong-id, corrupt, misnamed and temporary files are skipped"
        );
        let two: Vec<String> = store.load_summaries(2).into_iter().map(|s| s.id).collect();
        assert_eq!(two, [ID_B, ID_C]);
    }

    #[test]
    fn a_summary_write_never_touches_another_writers_temporary_file() {
        let dir = TempDir::new("store-tmp-names");
        let store = RunStore::new(dir.path().join("runs"));
        fs::create_dir_all(store.dir()).unwrap();
        // Another process is part-way through writing this run's summary.
        let theirs = store.dir().join(format!("{ID_A}.run.json.tmp"));
        fs::write(&theirs, "half a summary from another process").unwrap();
        store.write_summary(&summary(ID_A, 1.0), true).unwrap();
        assert_eq!(
            fs::read_to_string(&theirs).unwrap(),
            "half a summary from another process",
            "a temporary file with a shared name is truncated, then renamed away, by the other writer"
        );
        assert!(store.read_summary(ID_A).is_some());
    }

    /// Something outside the process (not identified; an on-access scanner
    /// fits) can refuse a rename over the summary (error 5) for longer than
    /// [`REFUSAL_WAIT`] under a verify's load: 2 of 200 loaded runs, once for
    /// 6+ s, on 2026-10-09. A store bound that this test does not control
    /// would make it measure the machine; its stores wait this long instead,
    /// and what it proves of the writers and the reader is unchanged.
    const OUTSIDE_HOLD_WAIT: Duration = Duration::from_secs(30);

    #[test]
    fn writers_of_one_summary_in_turn_never_fail_or_tear_the_file() {
        let dir = TempDir::new("store-concurrent");
        let runs = dir.path().join("runs");
        RunStore::new(&runs)
            .write_summary(&summary(ID_A, 0.0), false)
            .unwrap();
        let writers: Vec<_> = (0..2)
            .map(|writer| {
                let runs = runs.clone();
                std::thread::spawn(move || {
                    let store = RunStore::new(runs).with_refusal_wait(OUTSIDE_HOLD_WAIT);
                    let mut failures = 0;
                    for round in 0..150 {
                        let mut one = summary(ID_A, 1.0);
                        one.output = Some(format!("{writer}:{round}:{}", "x".repeat(round * 50)));
                        if store.write_summary(&one, false).is_err() {
                            failures += 1;
                        }
                    }
                    failures
                })
            })
            .collect();
        let reader = {
            let runs = runs.clone();
            std::thread::spawn(move || {
                let store = RunStore::new(runs).with_refusal_wait(OUTSIDE_HOLD_WAIT);
                let mut unreadable = 0;
                for _ in 0..2_000 {
                    if store.read_summary(ID_A).is_none() {
                        unreadable += 1;
                    }
                }
                unreadable
            })
        };
        let failures: u32 = writers.into_iter().map(|w| w.join().unwrap()).sum();
        let unreadable = reader.join().unwrap();
        assert_eq!(
            failures, 0,
            "a write failed because another writer took its temporary file"
        );
        assert_eq!(unreadable, 0, "a torn or missing summary was read");
        let names: Vec<String> = fs::read_dir(&runs)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(
            names,
            [format!("{ID_A}.run.json")],
            "no temporary file is left behind"
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_summary_refused_for_a_moment_is_read_once_it_is_released() {
        use std::os::windows::fs::OpenOptionsExt;
        let dir = TempDir::new("store-refused");
        let store = RunStore::new(dir.path().join("runs"));
        store.write_summary(&summary(ID_A, 1.0), false).unwrap();
        // A scanner holds the file with no sharing: every open is refused (32).
        let held = OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(store.summary_path(ID_A).unwrap())
            .unwrap();
        assert_eq!(
            File::open(store.summary_path(ID_A).unwrap())
                .unwrap_err()
                .raw_os_error(),
            Some(32)
        );
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            drop(held);
        });
        assert!(
            store.read_summary(ID_A).is_some(),
            "a summary refused for a moment was read as missing"
        );
        release.join().unwrap();
    }

    /// While a rename replaces a summary, an open can find no file (error 2)
    /// for a moment (263 ms measured under load, 2026-10-09): a summary missing
    /// for longer than the old 5 x 20 ms is still read once it is back.
    #[test]
    fn a_summary_missing_for_a_moment_is_read_once_it_is_back() {
        let dir = TempDir::new("store-missing");
        let store = RunStore::new(dir.path().join("runs"));
        store.write_summary(&summary(ID_A, 1.0), false).unwrap();
        let path = store.summary_path(ID_A).unwrap();
        let aside = path.with_extension("aside");
        fs::rename(&path, &aside).unwrap();
        let back = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            fs::rename(&aside, &path).unwrap();
        });
        assert!(
            store.read_summary(ID_A).is_some(),
            "a summary missing for a moment was read as missing"
        );
        back.join().unwrap();
    }

    /// A rename over a summary that a scanner holds open with no sharing is
    /// refused until it lets go: a write refused for longer than the old
    /// 5 x 20 ms still lands, and leaves no temporary file.
    #[cfg(windows)]
    #[test]
    fn a_write_refused_for_a_moment_lands_once_the_summary_is_released() {
        use std::os::windows::fs::OpenOptionsExt;
        let dir = TempDir::new("store-write-refused");
        let store = RunStore::new(dir.path().join("runs"));
        store.write_summary(&summary(ID_A, 1.0), false).unwrap();
        let held = OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(store.summary_path(ID_A).unwrap())
            .unwrap();
        let release = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            drop(held);
        });
        store
            .write_summary(&summary(ID_A, 2.0), false)
            .expect("a write refused for a moment lands once the file is released");
        release.join().unwrap();
        assert_eq!(store.read_summary(ID_A).unwrap().created_at, 2.0);
        let names: Vec<String> = fs::read_dir(store.dir())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(
            names,
            [format!("{ID_A}.run.json")],
            "no temporary file is left behind"
        );
    }

    /// The bound is a store's own: a hold past [`REFUSAL_WAIT`] fails a
    /// store's write after it (and removes its temporary file), while a store
    /// given a longer wait lands once the hold ends.
    #[cfg(windows)]
    #[test]
    fn a_hold_past_the_refusal_wait_fails_a_write_unless_the_store_waits_longer() {
        use std::os::windows::fs::OpenOptionsExt;
        let dir = TempDir::new("store-write-held-long");
        let store = RunStore::new(dir.path().join("runs"));
        store.write_summary(&summary(ID_A, 1.0), false).unwrap();
        let held = OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(store.summary_path(ID_A).unwrap())
            .unwrap();
        // Held for four times the bound, so that a loaded machine's late
        // wake-up cannot carry the failing write's last try past the hold.
        let release = std::thread::spawn(move || {
            std::thread::sleep(REFUSAL_WAIT * 4);
            drop(held);
        });
        let patient = store.clone().with_refusal_wait(OUTSIDE_HOLD_WAIT);
        let waits = std::thread::spawn(move || patient.write_summary(&summary(ID_A, 3.0), false));
        assert!(
            store.write_summary(&summary(ID_A, 2.0), false).is_err(),
            "a write refused for longer than REFUSAL_WAIT fails"
        );
        waits
            .join()
            .unwrap()
            .expect("a store that waits longer lands once the hold ends");
        release.join().unwrap();
        assert_eq!(store.read_summary(ID_A).unwrap().created_at, 3.0);
        let names: Vec<String> = fs::read_dir(store.dir())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(
            names,
            [format!("{ID_A}.run.json")],
            "no temporary file is left behind"
        );
    }

    #[test]
    fn a_line_whose_seq_jumps_to_the_end_of_the_range_is_skipped() {
        let dir = TempDir::new("store-seq-range");
        let store = RunStore::new(dir.path().join("runs"));
        fs::create_dir_all(store.dir()).unwrap();
        let line = |seq: u64| {
            serde_json::to_string(&event(seq, RunEventKind::Agent { name: "a".into() })).unwrap()
        };
        let write = |id: &str, seqs: &[u64]| {
            let text: String = seqs.iter().map(|seq| line(*seq) + "\n").collect();
            fs::write(store.events_path(id).unwrap(), text).unwrap();
            store
                .read_events(id)
                .iter()
                .map(|e| e.seq)
                .collect::<Vec<u64>>()
        };
        assert_eq!(write(ID_A, &[1, u64::MAX, 2, 3]), [1, 2, 3]);
        assert_eq!(write(ID_B, &[u64::MAX]), Vec::<u64>::new());
        assert_eq!(write(ID_C, &[u64::MAX - 1, 5, u64::MAX]), [5]);
        // Gaps that real runs make (live deltas take a seq and are not saved) stay.
        assert_eq!(
            write("00000000000000dd", &[1, 2, 40_000, 50_001]),
            [1, 2, 40_000, 50_001]
        );
    }

    #[test]
    fn one_process_at_a_time_holds_the_writer_lock() {
        let dir = TempDir::new("store-lock");
        let runs = dir.path().join("runs");
        let first = RunStore::new(&runs);
        let second = RunStore::new(&runs);
        assert!(!runs.exists(), "constructing creates nothing");

        let held = first.try_lock_writer().unwrap();
        assert!(held.is_some(), "the first takes it, making the directory");
        assert!(runs.join(LOCK_FILE).is_file());
        assert!(
            second.try_lock_writer().unwrap().is_none(),
            "the second finds it held"
        );
        assert!(
            second.try_lock_writer().unwrap().is_none(),
            "and again: asking does not take it"
        );
        drop(held);
        let taken = second
            .try_lock_writer()
            .unwrap()
            .expect("free once the first lets go");
        assert!(first.try_lock_writer().unwrap().is_none());
        drop(taken);
        // The lock file is not a run, and listing runs does not see it.
        assert!(first.load_summaries(10).is_empty());
    }

    #[test]
    fn taking_the_writer_lock_clears_what_a_dead_writer_left_and_only_then() {
        let dir = TempDir::new("store-sweep");
        let runs = dir.path().join("runs");
        fs::create_dir_all(&runs).unwrap();
        let leftovers = [
            format!("{ID_A}.run.json.tmp"),
            format!("{ID_A}.run.json.4242-7.tmp"),
            format!("{ID_B}.events.jsonl.4242-8.tmp"),
        ];
        let keep = [
            format!("{ID_A}.run.json"),
            format!("{ID_A}.events.jsonl"),
            "notes.tmp".to_owned(),
            ".tmp".to_owned(),
            "run.json.tmp".to_owned(),
            "0123456789ABCDEF.run.json.tmp".to_owned(),
        ];
        for name in leftovers.iter().chain(&keep) {
            fs::write(runs.join(name), "x").unwrap();
        }
        let store = RunStore::new(&runs);
        let other = RunStore::new(&runs);
        let held = store.try_lock_writer().unwrap().expect("the lock is free");
        for name in &leftovers {
            assert!(!runs.join(name).exists(), "{name} should be cleared");
        }
        for name in &keep {
            assert!(runs.join(name).exists(), "{name} is not a leftover");
        }
        // A window that does not get the lock leaves a live writer's temporary file alone.
        let live = runs.join(format!("{ID_C}.run.json.1-1.tmp"));
        fs::write(&live, "being written").unwrap();
        assert!(other.try_lock_writer().unwrap().is_none());
        assert!(live.exists());
        drop(held);
    }

    #[test]
    fn an_oversize_summary_is_ignored() {
        let dir = TempDir::new("store-big");
        let store = RunStore::new(dir.path().join("runs"));
        store.write_summary(&summary(ID_A, 1.0), true).unwrap();
        let mut big = summary(ID_A, 1.0);
        big.task = "x".repeat(MAX_SUMMARY_BYTES as usize);
        fs::write(
            store.summary_path(ID_A).unwrap(),
            serde_json::to_vec(&big).unwrap(),
        )
        .unwrap();
        assert!(store.read_summary(ID_A).is_none());
        assert!(store.load_summaries(10).is_empty());
    }
}
