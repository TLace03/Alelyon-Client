//! The shared chat store's archive: `<store>/evicted/`, which `history.py`
//! writes when a thread passes the 60-thread cap (the chat core's spec
//! §5.2 rules S19 and S20, §5.6). The reader is row C2's; the writer (S4′'s
//! archive and undo, S19), which the store's writes call, is row G2's.
//!
//! `evicted/index.json` holds rows `{**thread_row, "evicted_at": <float>,
//! "file": "<name>"}`. Reading it ([`read_evicted`]) gives one of four states,
//! as Python's `_read_evicted_index_locked` distinguishes them:
//! - [`ArchiveState::Absent`]: no file (Python: an empty archive);
//! - [`ArchiveState::Rows`]: any JSON list; every item is kept, an object
//!   with a thread id and a `created` time as an [`ArchivedThread`], anything
//!   else as it is ([`ArchiveRow::Other`]), since Python keeps and skips such
//!   rows;
//! - [`ArchiveState::Corrupt`]: read, and not UTF-8, not JSON Python reads,
//!   or not a list (Python sets such a file aside and starts a fresh one);
//! - [`ArchiveState::Unreadable`]: the file is there and cannot be read just
//!   now (a sharing violation, a permission error, a folder in its place), or
//!   it is JSON Python reads and this port cannot hold (D7: `NaN`, `Infinity`,
//!   a lone surrogate, a number past a double, nesting past 512). Then nothing
//!   is archived and nothing is set aside: it is never treated as corrupt,
//!   because Python reads it as a valid archive.
//!
//! A transcript is resolved only through its row's `file`, never from the id
//! (ids repeat across rows: `(id, created)` is a conversation's key). A row
//! whose file is missing is tolerated and shown as "transcript missing"; a row
//! with `file: ""` is a thread that never had a turn.
//!
//! Parity is pinned by `tests/parity/chat/evicted_read.json`, recorded from
//! this branch's `history.py` (the web privacy fix, merged at row G0), which
//! replaced C2's provisional fixtures at row G1.
//!
//! The writer moves a transcript and never copies or unlinks one, never
//! overwrites a file (its moves refuse an existing target), writes
//! `evicted/index.json` as `json.dumps(rows, indent=1)` in text mode through
//! `index.tmp` and a retried replace, and keeps every item of the archive
//! index as it was read, rows that are not threads included. Its bytes are
//! pinned by `chat/evicted_ops.json`, `chat/archive_row.json` and
//! `chat/store_ops.json` (`write_parity_tests`).

use std::collections::HashSet;
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use lattice_protocol::chat::is_thread_id;
use lattice_protocol::conversation::{ArchiveKey, ArchiveReason, ArchivedSummary};

use super::pyjson::{self, JsonError, PyValue};
use super::store::{IndexRow, thread_file_name};
use crate::clock::Clock;
use crate::fsx;

/// The folder, inside the store, that archived threads are moved into.
pub const EVICTED_DIR: &str = "evicted";
/// How many archived rows the list shows at a time (§5.6).
pub const ARCHIVE_PAGE: usize = 60;

/// One archived conversation.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchivedThread {
    pub id: String,
    pub created: f64,
    pub title: String,
    pub updated: f64,
    pub turns: i64,
    pub pinned_provider: String,
    pub evicted_at: f64,
    /// The transcript's name in `evicted/` as the row states it; `""` for a
    /// thread that never had a turn.
    pub file: String,
    /// The row names a transcript that is not in `evicted/`.
    pub transcript_missing: bool,
    /// The row as stored, for the writer to rewrite it unchanged.
    pub raw: PyValue,
}

/// One item of the archive index.
#[derive(Clone, Debug, PartialEq)]
pub enum ArchiveRow {
    Thread(ArchivedThread),
    /// Not an object with a thread id and a `created` time: kept as stored.
    Other(PyValue),
}

impl ArchiveRow {
    /// The row as stored.
    pub fn raw(&self) -> &PyValue {
        match self {
            Self::Thread(thread) => &thread.raw,
            Self::Other(raw) => raw,
        }
    }
}

/// What `evicted/index.json` holds (S20).
#[derive(Clone, Debug, PartialEq)]
pub enum ArchiveState {
    Absent,
    Rows(Vec<ArchiveRow>),
    /// It cannot be read just now, or only this port cannot hold it: leave
    /// everything alone.
    Unreadable,
    /// Python would set it aside.
    Corrupt,
}

/// `Path(name).name`: the last component of a name that may hold `/`, `\`
/// or a drive. `None` when that leaves nothing a file could be called.
fn base_name(name: &str) -> Option<&str> {
    let trimmed = name.trim_end_matches(['/', '\\']);
    let last = trimmed.rsplit(['/', '\\']).next().unwrap_or(trimmed);
    let last = match last.as_bytes() {
        [drive, b':', ..] if drive.is_ascii_alphabetic() => &last[2..],
        _ => last,
    };
    (!last.is_empty() && last != "." && last != "..").then_some(last)
}

/// One archive item as a conversation, when it is one.
pub(crate) fn archived_thread(row: &PyValue, evicted: &Path) -> Option<ArchivedThread> {
    let id = row.get("id")?.as_str()?;
    if !is_thread_id(id) {
        return None;
    }
    let created = pyjson::py_float(row.get("created")?).ok()?;
    let text = |key: &str| row.get(key).map(pyjson::py_str).unwrap_or_default();
    let number = |key: &str| {
        row.get(key)
            .and_then(|value| pyjson::py_float(value).ok())
            .unwrap_or(0.0)
    };
    // `str(entry.get("file") or "")`.
    let file = match row.get("file") {
        Some(value) if pyjson::py_truthy(value) => pyjson::py_str(value),
        _ => String::new(),
    };
    let transcript_missing =
        !file.is_empty() && !base_name(&file).is_some_and(|name| evicted.join(name).is_file());
    Some(ArchivedThread {
        id: id.to_owned(),
        created,
        title: text("title"),
        updated: number("updated"),
        turns: row
            .get("turns")
            .and_then(|value| pyjson::py_int(value).ok())
            .unwrap_or(0),
        pinned_provider: text("pinned_provider"),
        evicted_at: number("evicted_at"),
        file,
        transcript_missing,
        raw: row.clone(),
    })
}

/// Read `<store_dir>/evicted/index.json` with one whole-file read (S20, S26).
pub fn read_evicted(store_dir: &Path) -> ArchiveState {
    let evicted = store_dir.join(EVICTED_DIR);
    let bytes = match fs::read(evicted.join("index.json")) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return ArchiveState::Absent,
        Err(_) => return ArchiveState::Unreadable,
    };
    // Python: UnicodeDecodeError, and a ValueError from json.loads, are corrupt.
    let Ok(text) = std::str::from_utf8(&bytes) else {
        return ArchiveState::Corrupt;
    };
    match pyjson::loads(text) {
        Ok(PyValue::List(items)) => ArchiveState::Rows(
            items
                .into_iter()
                .map(|item| match archived_thread(&item, &evicted) {
                    Some(thread) => ArchiveRow::Thread(thread),
                    None => ArchiveRow::Other(item),
                })
                .collect(),
        ),
        Ok(_) | Err(JsonError::Invalid) => ArchiveState::Corrupt,
        Err(JsonError::BeyondNative(_)) => ArchiveState::Unreadable,
    }
}

/// The archived list as the sidebar shows it (§5.6): newest `evicted_at`
/// first; `reason` is `Reader` for a row whose `(id, created)` the reader
/// archived here (`archived_by_reader.json`, S21), `Cap` for every other
/// conversation and `Unknown` for an item that is not one.
pub fn summaries(rows: &[ArchiveRow], by_reader: &[ArchiveKey]) -> Vec<ArchivedSummary> {
    let mut out: Vec<ArchivedSummary> = rows
        .iter()
        .map(|row| match row {
            ArchiveRow::Thread(thread) => ArchivedSummary {
                key: ArchiveKey {
                    id: thread.id.clone(),
                    created: thread.created,
                },
                title: thread.title.clone(),
                updated: thread.updated,
                turns: u64::try_from(thread.turns).unwrap_or(0),
                evicted_at: thread.evicted_at,
                file: thread.file.clone(),
                reason: if by_reader
                    .iter()
                    .any(|key| key.id == thread.id && key.created == thread.created)
                {
                    ArchiveReason::Reader
                } else {
                    ArchiveReason::Cap
                },
                transcript_missing: thread.transcript_missing,
            },
            ArchiveRow::Other(raw) => ArchivedSummary {
                key: ArchiveKey {
                    id: raw.get("id").map(pyjson::py_str).unwrap_or_default(),
                    created: raw
                        .get("created")
                        .and_then(|value| pyjson::py_float(value).ok())
                        .unwrap_or(0.0),
                },
                title: raw.get("title").map(pyjson::py_str).unwrap_or_default(),
                updated: 0.0,
                turns: 0,
                evicted_at: raw
                    .get("evicted_at")
                    .and_then(|value| pyjson::py_float(value).ok())
                    .unwrap_or(0.0),
                file: String::new(),
                reason: ArchiveReason::Unknown,
                transcript_missing: false,
            },
        })
        .collect();
    // Stable, newest first; a NaN time sorts as total_cmp places it.
    out.sort_by(|a, b| b.evicted_at.total_cmp(&a.evicted_at));
    out
}

/// One page of [`ARCHIVE_PAGE`] summaries ("Show older" asks for the next).
pub fn page(summaries: &[ArchivedSummary], number: usize) -> &[ArchivedSummary] {
    let start = number.saturating_mul(ARCHIVE_PAGE).min(summaries.len());
    let end = start.saturating_add(ARCHIVE_PAGE).min(summaries.len());
    &summaries[start..end]
}

// ------------------------------------------------------------------ writing
//
// The archive's writer (S4′ steps 2, 3 and 5; S19; row G2), ported from
// `history.py`'s `_archive_evicted_locked`, `_archive_one_locked`,
// `_forget_rows_locked`, `_undo_archive_locked`, `_set_aside_evicted_index_locked`
// and `_free_archive_name`. Every caller holds the store's lock. A transcript is
// moved, never copied and never unlinked; an existing file is never overwritten
// (the move refuses one); the archive index is written through `index.tmp` and
// replaced, as Python writes it.

/// The platform's line end, which Python's text mode writes (S2).
pub(crate) const LINE_END: &str = if cfg!(windows) { "\r\n" } else { "\n" };

/// `text` as a text-mode write puts it on disk: each `\n` as [`LINE_END`].
/// Python's JSON here is `ensure_ascii`, so a line end inside a string is an
/// escape and every `\n` is structural.
pub(crate) fn text_mode(text: &str) -> String {
    if cfg!(windows) {
        text.replace('\n', LINE_END)
    } else {
        text.to_owned()
    }
}

/// A fault a unit test injects into the store's replaces, as the parity
/// generator patches Python's `_replace`: every replace of a store-relative
/// `src` onto `dst` that the rule names fails with a permission error.
#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Fault {
    pub src_suffix: Option<String>,
    pub dst: Option<String>,
    pub dst_prefix: Option<String>,
}

/// The store's folder, clock and (in tests) injected faults: what the writer
/// needs to touch the disk.
pub(crate) struct Disk<'a> {
    root: &'a Path,
    clock: &'a Clock,
    #[cfg(test)]
    faults: Vec<Fault>,
}

impl<'a> Disk<'a> {
    pub(crate) fn new(root: &'a Path, clock: &'a Clock) -> Self {
        Self {
            root,
            clock,
            #[cfg(test)]
            faults: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_faults(mut self, faults: Vec<Fault>) -> Self {
        self.faults = faults;
        self
    }

    pub(crate) fn root(&self) -> &Path {
        self.root
    }

    /// `_now()`.
    pub(crate) fn now(&self) -> f64 {
        (self.clock)()
    }

    fn evicted(&self) -> PathBuf {
        self.root.join(EVICTED_DIR)
    }

    #[cfg(test)]
    fn injected(&self, from: &Path, to: &Path) -> io::Result<()> {
        let relative = |path: &Path| {
            path.strip_prefix(self.root)
                .unwrap_or(path)
                .to_string_lossy()
                .replace('\\', "/")
        };
        let (source, target) = (relative(from), relative(to));
        let hit = self.faults.iter().any(|rule| {
            rule.src_suffix
                .as_deref()
                .is_none_or(|suffix| source.ends_with(suffix))
                && rule.dst.as_deref().is_none_or(|dst| target == dst)
                && rule
                    .dst_prefix
                    .as_deref()
                    .is_none_or(|prefix| target.starts_with(prefix))
        });
        if hit {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "an injected fault",
            ));
        }
        Ok(())
    }

    #[cfg(not(test))]
    #[allow(clippy::unused_self, clippy::unnecessary_wraps)]
    fn injected(&self, _from: &Path, _to: &Path) -> io::Result<()> {
        Ok(())
    }

    /// `_replace(from, to)`: a rename that replaces `to`, retried briefly on a
    /// sharing violation (6 tries, 20 ms apart).
    pub(crate) fn replace(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.injected(from, to)?;
        fsx::replace_shared(from, to)
    }

    /// A move that never replaces: where Python checks `exists()` and then
    /// replaces, the check and the move are one call here.
    pub(crate) fn move_new(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.injected(from, to)?;
        fsx::move_new(from, to)
    }

    /// `json.dumps(value, indent=1)` written to `<folder>/index.tmp` in text
    /// mode, then replaced onto `<folder>/index.json`.
    pub(crate) fn write_index_file(&self, folder: &Path, value: &PyValue) -> io::Result<()> {
        let temporary = folder.join(INDEX_TMP);
        fs::write(&temporary, text_mode(&pyjson::dumps_indent1(value)))?;
        self.replace(&temporary, &folder.join(INDEX))
    }
}

/// The index file's name, in the store and in `evicted/`.
pub(crate) const INDEX: &str = "index.json";
/// Python's temporary for it (`index.with_suffix(".tmp")`). Never removed by
/// native (ND3): Python writes the same name, and either side's next write
/// overwrites it.
pub(crate) const INDEX_TMP: &str = "index.tmp";

/// `Path(name).name` joined to `folder`: `None` when that names `folder` or
/// its parent (`""`, `.`, `..`), which always exists.
pub(crate) fn file_path(folder: &Path, name: &str) -> Option<PathBuf> {
    base_name(name).map(|base| folder.join(base))
}

/// Whether `folder / Path(name).name` exists.
fn exists_in(folder: &Path, name: &str) -> bool {
    file_path(folder, name).is_none_or(|path| path.exists())
}

/// Python's `value == number` for a JSON value and a float.
pub(crate) fn py_equals_float(value: Option<&PyValue>, number: f64) -> bool {
    match value {
        Some(PyValue::Bool(flag)) => f64::from(u8::from(*flag)) == number,
        Some(PyValue::Float(float)) => *float == number,
        Some(PyValue::Int(digits)) => {
            if !number.is_finite() || number.fract() != 0.0 {
                return false;
            }
            // The float's exact integer value, in decimal.
            let exact = format!("{number:.0}");
            let exact = if exact == "-0" { "0".to_owned() } else { exact };
            *digits == exact
        }
        _ => false,
    }
}

/// `str(value or "")`.
fn str_or_empty(value: Option<&PyValue>) -> String {
    match value {
        Some(value) if pyjson::py_truthy(value) => pyjson::py_str(value),
        _ => String::new(),
    }
}

/// `entry.update(key=value)`: in place when the key is there, else last.
fn set_key(entry: &mut PyValue, key: &str, value: PyValue) {
    if let PyValue::Object(pairs) = entry {
        match pairs.iter_mut().find(|(name, _)| name == key) {
            Some((_, slot)) => *slot = value,
            None => pairs.push((key.to_owned(), value)),
        }
    }
}

/// `asdict(row)`, in `Thread`'s field order.
pub(crate) fn row_value(row: &IndexRow) -> PyValue {
    PyValue::Object(vec![
        ("id".into(), PyValue::str(row.id.clone())),
        ("title".into(), PyValue::str(row.title.clone())),
        ("created".into(), PyValue::Float(row.created)),
        ("updated".into(), PyValue::Float(row.updated)),
        ("turns".into(), PyValue::int(row.turns)),
        (
            "pinned_provider".into(),
            PyValue::str(row.pinned_provider.clone()),
        ),
    ])
}

/// One item of the archive index, tagged so that a row can be taken back out
/// after others were (Python compares the dicts by identity).
#[derive(Clone, Debug)]
pub(crate) struct Item {
    tag: u64,
    value: PyValue,
}

/// The archive index's items as the writer holds them.
#[derive(Debug, Default)]
pub(crate) struct Items {
    items: Vec<Item>,
    next: u64,
}

impl Items {
    fn new(values: Vec<PyValue>) -> Self {
        let mut items = Self::default();
        for value in values {
            items.push(value);
        }
        items
    }

    fn push(&mut self, value: PyValue) -> u64 {
        let tag = self.next;
        self.next += 1;
        self.items.push(Item { tag, value });
        tag
    }

    fn value(&self) -> PyValue {
        PyValue::List(self.items.iter().map(|item| item.value.clone()).collect())
    }

    fn position(&self, tag: u64) -> Option<usize> {
        self.items.iter().position(|item| item.tag == tag)
    }

    /// The item at `index`, taken out.
    pub(crate) fn remove(&mut self, index: usize) -> PyValue {
        self.items.remove(index).value
    }

    /// Put `value` back at `index` (or last).
    pub(crate) fn insert(&mut self, index: usize, value: PyValue) {
        let tag = self.next;
        self.next += 1;
        let index = index.min(self.items.len());
        self.items.insert(index, Item { tag, value });
    }

    /// The first item that is the archive row of `(id, created)`.
    pub(crate) fn find(&self, id: &str, created: f64) -> Option<usize> {
        self.items.iter().position(|item| {
            item.value.is_object()
                && item.value.get("id") == Some(&PyValue::Str(id.to_owned()))
                && py_equals_float(item.value.get("created"), created)
        })
    }

    pub(crate) fn get(&self, index: usize) -> &PyValue {
        &self.items[index].value
    }
}

/// What the writer reads from `evicted/index.json` (S20).
pub(crate) enum Entries {
    Items(Items),
    /// It cannot be read just now, or only Python can: leave it alone.
    Unreadable,
    /// Python would set it aside.
    Corrupt,
}

/// `_read_evicted_index_locked`, for the writer.
pub(crate) fn read_entries(store_dir: &Path) -> Entries {
    match read_evicted(store_dir) {
        ArchiveState::Absent => Entries::Items(Items::default()),
        ArchiveState::Rows(rows) => Entries::Items(Items::new(
            rows.into_iter()
                .map(|row| match row {
                    ArchiveRow::Thread(thread) => thread.raw,
                    ArchiveRow::Other(raw) => raw,
                })
                .collect(),
        )),
        ArchiveState::Unreadable => Entries::Unreadable,
        ArchiveState::Corrupt => Entries::Corrupt,
    }
}

/// `_write_evicted_index_locked`.
pub(crate) fn write_entries(disk: &Disk<'_>, items: &Items) -> io::Result<()> {
    disk.write_index_file(&disk.evicted(), &items.value())
}

/// `time.strftime("%Y%m%dT%H%M%S", time.gmtime(seconds))`.
fn utc_stamp(seconds: f64) -> String {
    // gmtime floors a float; a clock before the epoch or past any calendar
    // this needs reads as the epoch.
    let whole = if seconds.is_finite() && (0.0..9.0e15).contains(&seconds) {
        seconds.floor() as i64
    } else {
        0
    };
    let (days, rest) = (whole.div_euclid(86_400), whole.rem_euclid(86_400));
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}{month:02}{day:02}T{:02}{:02}{:02}",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

/// `_set_aside_evicted_index_locked`: the damaged archive index moved, byte
/// for byte, to `index.corrupt-<UTC time>[-n].json`. True once it is out of
/// the way.
fn set_aside(disk: &Disk<'_>) -> bool {
    let folder = disk.evicted();
    let stamp = utc_stamp(disk.now());
    for n in 1..1000 {
        let suffix = if n == 1 {
            String::new()
        } else {
            format!("-{n}")
        };
        let target = folder.join(format!("index.corrupt-{stamp}{suffix}.json"));
        if target.exists() {
            continue;
        }
        return disk.move_new(&folder.join(INDEX), &target).is_ok();
    }
    false
}

/// `_free_archive_name`: `<stem>.jsonl`, then `<stem>-2.jsonl`, … for the
/// first name no archive row holds and no file in `evicted/` has.
fn free_archive_name(stem: &str, taken: &HashSet<String>, folder: &Path) -> Option<String> {
    (1..10_000)
        .map(|n| {
            if n == 1 {
                format!("{stem}.jsonl")
            } else {
                format!("{stem}-{n}.jsonl")
            }
        })
        .find(|name| !taken.contains(name) && !folder.join(name).exists())
}

/// One archive step, kept so that it can be taken back (`_Move`).
#[derive(Debug)]
pub(crate) struct Move {
    tag: u64,
    added: bool,
    source: Option<PathBuf>,
    target: Option<PathBuf>,
}

/// A batch of archive moves and the archive index they were recorded in: what
/// [`undo`] takes back when the thread index cannot be written.
#[derive(Debug, Default)]
pub(crate) struct Batch {
    moves: Vec<Move>,
    items: Items,
}

/// `_archive_one_locked`: the archive row first, then the move. `Err` when the
/// thread could not be archived, with its own row taken back out.
fn archive_one(
    disk: &Disk<'_>,
    row: &IndexRow,
    items: &mut Items,
    moves: &mut Vec<Move>,
) -> Result<(), ()> {
    let folder = disk.evicted();
    let source = thread_file_name(&row.id).map(|name| disk.root().join(name));
    let has_transcript = source.as_ref().is_some_and(|path| path.exists());
    let mut found = items.find(&row.id, row.created);
    if let Some(index) = found
        && has_transcript
    {
        let file = items.get(index).get("file");
        if !file.is_some_and(pyjson::py_truthy) || exists_in(&folder, &str_or_empty(file)) {
            // The earlier archive of this id is complete, and a transcript
            // sits at the id again: this one is archived separately.
            found = None;
        }
    }
    let added = found.is_none();
    let tag = match found {
        Some(index) => {
            let now = disk.now();
            let entry = &mut items.items[index].value;
            set_key(entry, "title", PyValue::str(row.title.clone()));
            set_key(entry, "updated", PyValue::Float(row.updated));
            set_key(entry, "turns", PyValue::int(row.turns));
            set_key(
                entry,
                "pinned_provider",
                PyValue::str(row.pinned_provider.clone()),
            );
            set_key(entry, "evicted_at", PyValue::Float(now));
            items.items[index].tag
        }
        None => {
            let taken: HashSet<String> = items
                .items
                .iter()
                .filter(|item| item.value.is_object())
                .map(|item| pyjson::py_str(item.value.get("file").unwrap_or(&PyValue::Null)))
                .collect();
            let name = if has_transcript {
                let stem = thread_file_name(&row.id)
                    .map(|name| name.trim_end_matches(".jsonl").to_owned())
                    .unwrap_or_default();
                free_archive_name(&stem, &taken, &folder).ok_or(())?
            } else {
                String::new()
            };
            let mut entry = row_value(row);
            set_key(&mut entry, "evicted_at", PyValue::Float(disk.now()));
            set_key(&mut entry, "file", PyValue::str(name));
            items.push(entry)
        }
    };
    if write_entries(disk, items).is_err() {
        if added && let Some(index) = items.position(tag) {
            items.items.remove(index);
        }
        return Err(());
    }
    let index = items.position(tag).ok_or(())?;
    let name = str_or_empty(items.get(index).get("file"));
    let target = if !name.is_empty() && has_transcript {
        Some(file_path(&folder, &name).unwrap_or_else(|| folder.clone()))
    } else {
        None
    };
    if let (Some(target), Some(source)) = (&target, &source)
        && disk.move_new(source, target).is_err()
    {
        if added {
            forget(disk, items, &[tag]);
        }
        return Err(());
    }
    moves.push(Move {
        tag,
        added,
        source,
        target,
    });
    Ok(())
}

/// `_forget_rows_locked`: take rows back out; best effort.
fn forget(disk: &Disk<'_>, items: &mut Items, tags: &[u64]) {
    items.items.retain(|item| !tags.contains(&item.tag));
    let _ = write_entries(disk, items);
}

/// `_undo_archive_locked`: transcripts back where they were, and the rows this
/// batch added out of the archive. A transcript that cannot be put back keeps
/// its row, which is then the only record of where it is.
pub(crate) fn undo(disk: &Disk<'_>, batch: Batch) {
    let Batch { moves, mut items } = batch;
    let mut gone = Vec::new();
    for step in moves.iter().rev() {
        let mut restored = true;
        if let (Some(target), Some(source)) = (&step.target, &step.source)
            && target.exists()
            && !source.exists()
            && disk.move_new(target, source).is_err()
        {
            restored = false;
        }
        if restored && step.added {
            gone.push(step.tag);
        }
    }
    if !gone.is_empty() {
        forget(disk, &mut items, &gone);
    }
}

/// `_archive_evicted_locked`: archive the threads the cap (or the reader)
/// moved out. Returns the threads that could NOT be archived, which stay
/// listed, and the batch, for [`undo`].
pub(crate) fn archive_evicted(disk: &Disk<'_>, stale: &[IndexRow]) -> (Vec<IndexRow>, Batch) {
    if stale.is_empty() {
        return (Vec::new(), Batch::default());
    }
    let mut items = match read_entries(disk.root()) {
        Entries::Items(items) => items,
        Entries::Corrupt => {
            if !set_aside(disk) {
                return (stale.to_vec(), Batch::default());
            }
            Items::default()
        }
        Entries::Unreadable => return (stale.to_vec(), Batch::default()),
    };
    if fs::create_dir_all(disk.evicted()).is_err() {
        return (stale.to_vec(), Batch::default());
    }
    let mut kept = Vec::new();
    let mut moves = Vec::new();
    for row in stale {
        if archive_one(disk, row, &mut items, &mut moves).is_err() {
            kept.push(row.clone());
        }
    }
    (kept, Batch { moves, items })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    fn write_index(store: &Path, text: &str) {
        let evicted = store.join(EVICTED_DIR);
        fs::create_dir_all(&evicted).unwrap();
        fs::write(evicted.join("index.json"), text).unwrap();
    }

    fn rows(store: &Path) -> Vec<ArchiveRow> {
        match read_evicted(store) {
            ArchiveState::Rows(rows) => rows,
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn transcripts_resolve_through_the_file_never_the_id() {
        let dir = TempDir::new("chat-archive-files");
        write_index(
            dir.path(),
            r#"[{"id": "a1", "created": 1.0, "file": "a1-2.jsonl", "evicted_at": 2.0},
                {"id": "a1", "created": 0.5, "file": "a1.jsonl", "evicted_at": 1.0},
                {"id": "b2", "created": 1.0, "file": "", "evicted_at": 3.0},
                {"id": "c3", "created": 1.0, "file": "../c3.jsonl", "evicted_at": 4.0},
                {"id": "d4", "created": 1.0, "file": "..", "evicted_at": 5.0}]"#,
        );
        // Only `a1.jsonl` and `c3.jsonl` are there; `a1-2.jsonl` is not, though
        // a file named after the id is.
        fs::write(dir.path().join(EVICTED_DIR).join("a1.jsonl"), "x").unwrap();
        fs::write(dir.path().join(EVICTED_DIR).join("c3.jsonl"), "x").unwrap();
        let found: Vec<(String, bool)> = rows(dir.path())
            .into_iter()
            .map(|row| match row {
                ArchiveRow::Thread(t) => (t.file, t.transcript_missing),
                ArchiveRow::Other(raw) => panic!("{raw:?}"),
            })
            .collect();
        assert_eq!(
            found,
            [
                ("a1-2.jsonl".to_owned(), true),
                ("a1.jsonl".to_owned(), false),
                (String::new(), false),
                ("../c3.jsonl".to_owned(), false),
                ("..".to_owned(), true),
            ]
        );
    }

    #[test]
    fn the_four_states() {
        let dir = TempDir::new("chat-archive-states");
        assert_eq!(read_evicted(dir.path()), ArchiveState::Absent);
        write_index(dir.path(), "{}");
        assert_eq!(read_evicted(dir.path()), ArchiveState::Corrupt);
        write_index(dir.path(), "[NaN, ");
        assert_eq!(read_evicted(dir.path()), ArchiveState::Corrupt);
        write_index(dir.path(), "[NaN]");
        assert_eq!(read_evicted(dir.path()), ArchiveState::Unreadable);
        fs::remove_file(dir.path().join(EVICTED_DIR).join("index.json")).unwrap();
        fs::create_dir(dir.path().join(EVICTED_DIR).join("index.json")).unwrap();
        assert_eq!(read_evicted(dir.path()), ArchiveState::Unreadable);
    }

    #[test]
    fn summaries_join_the_reader_s_archive_and_sort_newest_first() {
        let dir = TempDir::new("chat-archive-summaries");
        write_index(
            dir.path(),
            r#"[{"id": "old", "created": 1.0, "title": "Old", "turns": 3, "evicted_at": 10.0, "file": ""},
                {"id": "new", "created": 2.0, "title": "New", "turns": -1, "evicted_at": 20.0, "file": ""},
                5,
                {"title": "no id", "evicted_at": 15.0}]"#,
        );
        let listed = summaries(
            &rows(dir.path()),
            &[ArchiveKey {
                id: "new".into(),
                created: 2.0,
            }],
        );
        let shown: Vec<(&str, ArchiveReason, u64)> = listed
            .iter()
            .map(|s| (s.title.as_str(), s.reason, s.turns))
            .collect();
        assert_eq!(
            shown,
            [
                ("New", ArchiveReason::Reader, 0),
                ("no id", ArchiveReason::Unknown, 0),
                ("Old", ArchiveReason::Cap, 3),
                ("", ArchiveReason::Unknown, 0),
            ]
        );
        // The reader's key matches on both halves.
        let other_created = summaries(
            &rows(dir.path()),
            &[ArchiveKey {
                id: "new".into(),
                created: 2.5,
            }],
        );
        assert_eq!(other_created[0].reason, ArchiveReason::Cap);
    }

    #[test]
    fn pages_hold_sixty() {
        let one = |n: usize| ArchivedSummary {
            key: ArchiveKey {
                id: format!("t{n}"),
                created: 0.0,
            },
            title: String::new(),
            updated: 0.0,
            turns: 0,
            evicted_at: 0.0,
            file: String::new(),
            reason: ArchiveReason::Cap,
            transcript_missing: false,
        };
        let all: Vec<_> = (0..130).map(one).collect();
        assert_eq!(page(&all, 0).len(), 60);
        assert_eq!(page(&all, 2).len(), 10);
        assert!(page(&all, 3).is_empty());
        assert!(page(&all, usize::MAX).is_empty());
    }

    #[test]
    fn base_names_follow_path_name() {
        assert_eq!(base_name("a1.jsonl"), Some("a1.jsonl"));
        assert_eq!(base_name("../x.jsonl"), Some("x.jsonl"));
        assert_eq!(base_name("a\\b\\c.jsonl"), Some("c.jsonl"));
        assert_eq!(base_name("C:c.jsonl"), Some("c.jsonl"));
        assert_eq!(base_name("dir/"), Some("dir"));
        assert_eq!(base_name(".."), None);
        assert_eq!(base_name("/"), None);
    }
}
