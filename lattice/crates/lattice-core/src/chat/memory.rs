//! The transcript store in memory (the chat core's spec §5.10; row C3).
//!
//! The same contract as the shared store, S1–S25 at the level of rows and
//! turns, for the agent chat's tests and the development host while the shared
//! store is read-only here (gate G-WEB, §5.9). It is compiled only for this
//! crate's tests and the `dev-host` feature, so no shipped build holds a
//! store that forgets everything when it closes.
//!
//! What it follows, with its rule:
//! - S8: an append defaults its id and time, makes a missing row (titled from
//!   the text, `created = updated = ts`, one turn) or moves an existing one
//!   (`updated = ts`, one more turn, a user turn names an untitled thread);
//!   `append_answer` writes nothing for a row that is gone (D8);
//! - S9: a new thread, its first message and its pin in one index write;
//! - S10, S11: rename collapses and cuts; a pin is stripped and never moves
//!   `updated`;
//! - S13: supersede marks the first visible turn with the id and every later
//!   visible one, and sets the row's count to what is still visible;
//! - S25: touch sets `updated` to now, and never makes a row;
//! - S4′ and S19: every index write sorts by `updated` (newest first, ties
//!   kept in order) and archives every row past 60 with its transcript,
//!   under the first free name (`<id>.jsonl`, `<id>-2.jsonl`, …), the archive
//!   row first; a row whose archive cannot be made stays listed;
//! - S21: the reader's Archive moves a thread as an eviction does;
//! - S22: unarchive refuses an id that is listed, moves the transcript back,
//!   drops the archive row and lists the thread with `updated` set to now, in
//!   that order;
//! - S3, D7: while the index is marked unreadable ([`set_index_unreadable`]),
//!   every write refuses and nothing changes.
//!
//! There is no delete (ND1).
//!
//! [`set_index_unreadable`]: MemoryTranscriptStore::set_index_unreadable

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};

use lattice_protocol::conversation::ArchiveKey;

use super::archive::{ArchiveRow, ArchiveState, ArchivedThread};
use super::pyjson::PyValue;
use super::store::{
    IdSource, IndexRow, IndexState, MAX_THREADS, MAX_TURNS, NEW_THREAD_TITLE, RECENT_TURNS,
    StoredTurn, auto_title, normalize_title, thread_file_name,
};
use super::transcript::{NewTurn, StoreError, TranscriptStore};
use crate::clock::Clock;
use crate::py;

/// One row of the archive, and the transcript file it names.
#[derive(Clone, Debug)]
struct Entry {
    row: IndexRow,
    evicted_at: f64,
    file: String,
}

#[derive(Default)]
struct Memory {
    /// `None`: no index has been written yet.
    index: Option<Vec<IndexRow>>,
    index_unreadable: bool,
    /// Transcripts by file name, as `<id>.jsonl` would hold them.
    threads: HashMap<String, Vec<StoredTurn>>,
    /// `None`: no archive index has been written yet.
    archive: Option<Vec<Entry>>,
    /// `evicted/<file>`.
    archived_files: HashMap<String, Vec<StoredTurn>>,
}

impl Memory {
    fn rows(&mut self) -> &mut Vec<IndexRow> {
        self.index.get_or_insert_with(Vec::new)
    }

    /// A listed row; looking makes no index.
    fn row(&mut self, id: &str) -> Option<&mut IndexRow> {
        self.index.as_mut()?.iter_mut().find(|row| row.id == id)
    }

    fn writable(&self) -> Result<(), StoreError> {
        if self.index_unreadable {
            Err(StoreError::IndexUnreadable)
        } else {
            Ok(())
        }
    }

    /// The first name in `evicted/` that no archive row holds or promises.
    fn free_name(&self, stem: &str) -> String {
        let entries = self.archive.as_deref().unwrap_or_default();
        (1..)
            .map(|n| {
                if n == 1 {
                    format!("{stem}.jsonl")
                } else {
                    format!("{stem}-{n}.jsonl")
                }
            })
            .find(|name| {
                !entries.iter().any(|entry| &entry.file == name)
                    && !self.archived_files.contains_key(name)
            })
            .unwrap_or_default()
    }

    /// `_archive_one_locked`: the archive row first, then the transcript. True
    /// once it is archived.
    fn archive_one(&mut self, row: &IndexRow, now: f64) -> bool {
        let source = thread_file_name(&row.id);
        let has_transcript = source
            .as_ref()
            .is_some_and(|name| self.threads.contains_key(name));
        let entries = self.archive.get_or_insert_with(Vec::new);
        let mut found = entries
            .iter()
            .position(|entry| entry.row.id == row.id && entry.row.created == row.created);
        if let Some(index) = found
            && has_transcript
            && (entries[index].file.is_empty()
                || self.archived_files.contains_key(&entries[index].file))
        {
            // An earlier archive of this id is complete and a transcript sits
            // at the id again: this one is archived separately.
            found = None;
        }
        let (index, added) = match found {
            Some(index) => {
                let entry = &mut entries[index];
                entry.row.title.clone_from(&row.title);
                entry.row.updated = row.updated;
                entry.row.turns = row.turns;
                entry.row.pinned_provider.clone_from(&row.pinned_provider);
                entry.evicted_at = now;
                (index, false)
            }
            None => {
                let file = match (&source, has_transcript) {
                    (Some(name), true) => self.free_name(name.trim_end_matches(".jsonl")),
                    _ => String::new(),
                };
                let entries = self.archive.get_or_insert_with(Vec::new);
                entries.push(Entry {
                    row: row.clone(),
                    evicted_at: now,
                    file,
                });
                (entries.len() - 1, true)
            }
        };
        let entries = self.archive.get_or_insert_with(Vec::new);
        let file = entries[index].file.clone();
        if let (Some(source), true, false) = (source, has_transcript, file.is_empty()) {
            if self.archived_files.contains_key(&file) {
                // Never overwrite an archived transcript; take the row back.
                if added {
                    entries.remove(index);
                }
                return false;
            }
            if let Some(turns) = self.threads.remove(&source) {
                self.archived_files.insert(file, turns);
            }
        }
        true
    }

    /// `_write_index_locked` (S4′): sort, archive past the cap, keep what
    /// could not be archived.
    fn write_index(&mut self, now: f64) {
        let mut rows = self.index.take().unwrap_or_default();
        rows.sort_by(|a, b| b.updated.total_cmp(&a.updated));
        let stale = rows.split_off(rows.len().min(MAX_THREADS));
        let mut kept = Vec::new();
        for row in stale {
            if !self.archive_one(&row, now) {
                kept.push(row);
            }
        }
        rows.extend(kept);
        self.index = Some(rows);
    }

    fn visible(&self, id: &str) -> Vec<StoredTurn> {
        let Some(file) = thread_file_name(id) else {
            return Vec::new();
        };
        let turns: Vec<StoredTurn> = self
            .threads
            .get(&file)
            .map(|turns| {
                turns
                    .iter()
                    .filter(|turn| !turn.superseded)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        let skip = turns.len().saturating_sub(MAX_TURNS);
        turns[skip..].to_vec()
    }
}

/// The transcript store in memory.
pub struct MemoryTranscriptStore {
    ids: IdSource,
    clock: Clock,
    state: Mutex<Memory>,
}

impl MemoryTranscriptStore {
    /// An empty store: no index, no archive.
    pub fn new(ids: IdSource, clock: Clock) -> Self {
        Self {
            ids,
            clock,
            state: Mutex::new(Memory::default()),
        }
    }

    /// Act as a store whose index exists and cannot be read here (S3, D7).
    pub fn set_index_unreadable(&self, unreadable: bool) {
        self.lock().index_unreadable = unreadable;
    }

    fn lock(&self) -> MutexGuard<'_, Memory> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// S8, with S9's pin and D8's condition.
    fn append_turn(
        &self,
        memory: &mut Memory,
        id: &str,
        turn: NewTurn,
        answer: bool,
        pin: Option<&str>,
    ) -> Result<StoredTurn, StoreError> {
        let file = thread_file_name(id).ok_or(StoreError::NotSaved)?;
        memory.writable()?;
        if answer && memory.row(id).is_none() {
            return Err(StoreError::ThreadGone);
        }
        let now = (self.clock)();
        let turn_id = if turn.id.is_empty() {
            (self.ids)()
        } else {
            turn.id.clone()
        };
        // `if not turn.ts`: zero is unset (NaN is not).
        let ts = if turn.ts == 0.0 { now } else { turn.ts };
        let stored = turn.stored(turn_id, ts);
        memory.threads.entry(file).or_default().push(stored.clone());
        match memory.row(id) {
            Some(row) => {
                row.updated = ts;
                row.turns += 1;
                if stored.role == "user" && (row.title.is_empty() || row.title == NEW_THREAD_TITLE)
                {
                    row.title = auto_title(&stored.text);
                }
            }
            None => memory.rows().push(IndexRow {
                id: id.to_owned(),
                title: auto_title(&stored.text),
                created: ts,
                updated: ts,
                turns: 1,
                pinned_provider: String::new(),
            }),
        }
        if let (Some(pin), Some(row)) = (pin, memory.row(id)) {
            row.pinned_provider = py::strip(pin).to_owned();
        }
        memory.write_index(now);
        Ok(stored)
    }
}

impl TranscriptStore for MemoryTranscriptStore {
    fn list(&self) -> IndexState {
        let memory = self.lock();
        if memory.index_unreadable {
            return IndexState::Unreadable;
        }
        match &memory.index {
            None => IndexState::Absent,
            Some(rows) => IndexState::Rows(rows.clone()),
        }
    }

    fn archived(&self) -> ArchiveState {
        let memory = self.lock();
        let Some(entries) = &memory.archive else {
            return ArchiveState::Absent;
        };
        ArchiveState::Rows(
            entries
                .iter()
                .map(|entry| {
                    let row = &entry.row;
                    let raw = PyValue::Object(vec![
                        ("id".into(), PyValue::str(row.id.clone())),
                        ("title".into(), PyValue::str(row.title.clone())),
                        ("created".into(), PyValue::Float(row.created)),
                        ("updated".into(), PyValue::Float(row.updated)),
                        ("turns".into(), PyValue::int(row.turns)),
                        (
                            "pinned_provider".into(),
                            PyValue::str(row.pinned_provider.clone()),
                        ),
                        ("evicted_at".into(), PyValue::Float(entry.evicted_at)),
                        ("file".into(), PyValue::str(entry.file.clone())),
                    ]);
                    ArchiveRow::Thread(ArchivedThread {
                        id: row.id.clone(),
                        created: row.created,
                        title: row.title.clone(),
                        updated: row.updated,
                        turns: row.turns,
                        pinned_provider: row.pinned_provider.clone(),
                        evicted_at: entry.evicted_at,
                        file: entry.file.clone(),
                        transcript_missing: !entry.file.is_empty()
                            && !memory.archived_files.contains_key(&entry.file),
                        raw,
                    })
                })
                .collect(),
        )
    }

    fn load(&self, id: &str) -> Vec<StoredTurn> {
        self.lock().visible(id)
    }

    fn recent(&self, id: &str) -> Vec<StoredTurn> {
        let turns = self.load(id);
        let skip = turns.len().saturating_sub(RECENT_TURNS);
        turns[skip..].to_vec()
    }

    fn first_message(&self, text: &str, pin: &str) -> Result<(IndexRow, StoredTurn), StoreError> {
        let mut memory = self.lock();
        let id = (self.ids)();
        let turn = self.append_turn(&mut memory, &id, NewTurn::user(text), false, Some(pin))?;
        let row = memory.row(&id).cloned().ok_or(StoreError::NotSaved)?;
        Ok((row, turn))
    }

    fn append(&self, id: &str, turn: NewTurn) -> Result<StoredTurn, StoreError> {
        let mut memory = self.lock();
        self.append_turn(&mut memory, id, turn, false, None)
    }

    fn append_answer(&self, id: &str, turn: NewTurn) -> Result<StoredTurn, StoreError> {
        let mut memory = self.lock();
        self.append_turn(&mut memory, id, turn, true, None)
    }

    fn rename(&self, id: &str, title: &str) -> Result<bool, StoreError> {
        let title = normalize_title(title);
        if title.is_empty() {
            return Ok(false);
        }
        let mut memory = self.lock();
        memory.writable()?;
        let Some(row) = memory.row(id) else {
            return Ok(false);
        };
        row.title = title;
        memory.write_index((self.clock)());
        Ok(true)
    }

    fn pin(&self, id: &str, pin: &str) -> Result<bool, StoreError> {
        let mut memory = self.lock();
        memory.writable()?;
        let Some(row) = memory.row(id) else {
            return Ok(false);
        };
        row.pinned_provider = py::strip(pin).to_owned();
        memory.write_index((self.clock)());
        Ok(true)
    }

    fn supersede(&self, id: &str, from: &str) -> Result<usize, StoreError> {
        let Some(file) = thread_file_name(id) else {
            return Ok(0);
        };
        if from.is_empty() {
            return Ok(0);
        }
        let mut memory = self.lock();
        memory.writable()?;
        let Some(turns) = memory.threads.get_mut(&file) else {
            return Ok(0);
        };
        let Some(start) = turns
            .iter()
            .position(|turn| turn.id == from && !turn.superseded)
        else {
            return Ok(0);
        };
        let mut marked = 0;
        for turn in &mut turns[start..] {
            if !turn.superseded {
                turn.superseded = true;
                marked += 1;
            }
        }
        let visible = turns.iter().filter(|turn| !turn.superseded).count();
        if let Some(row) = memory.row(id) {
            row.turns = i64::try_from(visible).unwrap_or(i64::MAX);
        }
        memory.write_index((self.clock)());
        Ok(marked)
    }

    fn touch(&self, id: &str) -> Result<bool, StoreError> {
        let mut memory = self.lock();
        memory.writable()?;
        let now = (self.clock)();
        let Some(row) = memory.row(id) else {
            return Ok(false);
        };
        row.updated = now;
        memory.write_index(now);
        Ok(true)
    }

    fn archive(&self, id: &str) -> Result<(), StoreError> {
        let mut memory = self.lock();
        memory.writable()?;
        let rows = memory.rows();
        let Some(at) = rows.iter().position(|row| row.id == id) else {
            return Err(StoreError::NotFound);
        };
        let row = rows.remove(at);
        let now = (self.clock)();
        if !memory.archive_one(&row, now) {
            memory.rows().insert(at, row);
            return Err(StoreError::NotSaved);
        }
        memory.write_index(now);
        Ok(())
    }

    fn unarchive(&self, key: &ArchiveKey) -> Result<IndexRow, StoreError> {
        let file = thread_file_name(&key.id).ok_or(StoreError::NotFound)?;
        let mut memory = self.lock();
        memory.writable()?;
        let entries = memory.archive.get_or_insert_with(Vec::new);
        let Some(at) = entries
            .iter()
            .position(|entry| entry.row.id == key.id && entry.row.created == key.created)
        else {
            return Err(StoreError::NotFound);
        };
        // 1. A chat with this id is already in the list.
        if memory.threads.contains_key(&file) || memory.row(&key.id).is_some() {
            return Err(StoreError::AlreadyListed);
        }
        // 2. The transcript back to `<id>.jsonl`; 3. the archive row out.
        let entries = memory.archive.get_or_insert_with(Vec::new);
        let entry = entries.remove(at);
        if let Some(turns) = memory.archived_files.remove(&entry.file) {
            memory.threads.insert(file, turns);
        }
        // 4. Listed as the most recently used thread.
        let now = (self.clock)();
        let mut row = entry.row;
        row.updated = now;
        memory.rows().push(row.clone());
        memory.write_index(now);
        Ok(row)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// Ids `t1`, `t2`, … and a clock that moves one second per reading.
    fn store() -> MemoryTranscriptStore {
        let next = Arc::new(AtomicU64::new(1));
        let ticks = Arc::new(AtomicU64::new(0));
        MemoryTranscriptStore::new(
            Arc::new(move || format!("t{}", next.fetch_add(1, Ordering::Relaxed))),
            Arc::new(move || 1000.0 + ticks.fetch_add(1, Ordering::Relaxed) as f64),
        )
    }

    fn rows(store: &dyn TranscriptStore) -> Vec<IndexRow> {
        match store.list() {
            IndexState::Rows(rows) => rows,
            other => panic!("{other:?}"),
        }
    }

    fn archived(store: &dyn TranscriptStore) -> Vec<ArchivedThread> {
        match store.archived() {
            ArchiveState::Rows(rows) => rows
                .into_iter()
                .map(|row| match row {
                    ArchiveRow::Thread(thread) => thread,
                    ArchiveRow::Other(raw) => panic!("{raw:?}"),
                })
                .collect(),
            ArchiveState::Absent => Vec::new(),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_first_message_makes_the_row_the_turn_and_the_pin_at_once() {
        let store = store();
        assert_eq!(store.list(), IndexState::Absent);
        let (row, turn) = store.first_message("  What is   up?  ", " auto ").unwrap();
        assert_eq!(row.title, "What is up?");
        assert_eq!((row.created, row.updated, row.turns), (turn.ts, turn.ts, 1));
        assert_eq!(row.pinned_provider, "auto");
        assert_eq!(turn.role, "user");
        assert_eq!(store.load(&row.id), [turn]);
    }

    #[test]
    fn appends_follow_s8_and_an_answer_needs_its_row() {
        let store = store();
        // A missing row is made, titled from the text, as Python does.
        let first = store
            .append("x1", NewTurn::assistant("hello", "lab:m"))
            .unwrap();
        let row = rows(&store).remove(0);
        assert_eq!(
            (row.title.as_str(), row.turns, row.created),
            ("hello", 1, first.ts)
        );
        // A user turn names a thread titled "New thread"; it does not rename one
        // that has a title.
        store.rename("x1", "New thread").unwrap();
        store
            .append("x1", NewTurn::user("second question"))
            .unwrap();
        assert_eq!(rows(&store)[0].title, "second question");
        store.append("x1", NewTurn::user("third")).unwrap();
        let row = rows(&store).remove(0);
        assert_eq!((row.title.as_str(), row.turns), ("second question", 3));
        // D8: the answer to a thread whose row is gone writes nothing.
        let before = (store.list(), store.archived());
        assert_eq!(
            store.append_answer("gone", NewTurn::assistant("late", "lab:m")),
            Err(StoreError::ThreadGone)
        );
        assert_eq!((store.list(), store.archived()), before);
        assert!(store.load("gone").is_empty());
        // An id that names no file is not saved.
        assert_eq!(
            store.append("../", NewTurn::user("x")),
            Err(StoreError::NotSaved)
        );
        // Given ids and times are kept.
        let given = NewTurn {
            id: "fixed".into(),
            ts: 5.0,
            ..NewTurn::user("q")
        };
        assert_eq!(
            store.append("x1", given).map(|t| (t.id, t.ts)),
            Ok(("fixed".into(), 5.0))
        );
    }

    #[test]
    fn rename_pin_and_supersede_follow_python() {
        let store = store();
        let (row, question) = store.first_message("q1", "").unwrap();
        let id = row.id;
        let answer = store
            .append(&id, NewTurn::assistant("a1", "lab:m"))
            .unwrap();
        store.append(&id, NewTurn::user("q2")).unwrap();
        assert_eq!(store.rename(&id, " \t "), Ok(false));
        assert_eq!(store.rename("unknown", "x"), Ok(false));
        assert_eq!(store.rename(&id, &"y".repeat(90)), Ok(true));
        assert_eq!(rows(&store)[0].title, "y".repeat(80));
        let updated = rows(&store)[0].updated;
        assert_eq!(store.pin(&id, "  endpoint:x "), Ok(true));
        assert_eq!(rows(&store)[0].pinned_provider, "endpoint:x");
        assert_eq!(
            rows(&store)[0].updated,
            updated,
            "a pin never moves updated"
        );
        assert_eq!(store.pin("unknown", "auto"), Ok(false));
        // Supersede from the answer: it and the later question.
        assert_eq!(store.supersede(&id, &answer.id), Ok(2));
        assert_eq!(store.load(&id), std::slice::from_ref(&question));
        assert_eq!(rows(&store)[0].turns, 1);
        assert_eq!(
            store.supersede(&id, &answer.id),
            Ok(0),
            "already superseded"
        );
        assert_eq!(store.supersede(&id, ""), Ok(0));
        assert_eq!(store.supersede("unknown", &question.id), Ok(0));
    }

    #[test]
    fn the_sixty_first_thread_archives_the_least_recently_used_with_its_transcript() {
        let store = store();
        let ids: Vec<String> = (0..61)
            .map(|n| store.first_message(&format!("q{n}"), "").unwrap().0.id)
            .collect();
        let listed = rows(&store);
        assert_eq!(listed.len(), MAX_THREADS);
        assert!(!listed.iter().any(|row| row.id == ids[0]));
        let gone = archived(&store);
        assert_eq!(gone.len(), 1);
        assert_eq!(
            (gone[0].id.as_str(), gone[0].file.as_str()),
            (ids[0].as_str(), "t1.jsonl")
        );
        assert!(!gone[0].transcript_missing);
        assert!(store.load(&ids[0]).is_empty(), "the transcript moved");
        // A late append re-creates the id; its next eviction picks a free name.
        store
            .append(
                &ids[0],
                NewTurn {
                    ts: 1.0,
                    ..NewTurn::user("late")
                },
            )
            .unwrap();
        let gone = archived(&store);
        assert_eq!(
            gone.iter().map(|t| t.file.as_str()).collect::<Vec<_>>(),
            ["t1.jsonl", "t1-2.jsonl"]
        );
    }

    #[test]
    fn touch_keeps_a_long_turn_s_thread_listed() {
        let store = store();
        let ids: Vec<String> = (0..60)
            .map(|n| store.first_message(&format!("q{n}"), "").unwrap().0.id)
            .collect();
        // The oldest thread is mid-answer: touched, it is not the one archived.
        assert_eq!(store.touch(&ids[0]), Ok(true));
        store.first_message("one more", "").unwrap();
        assert!(rows(&store).iter().any(|row| row.id == ids[0]));
        assert_eq!(archived(&store)[0].id, ids[1]);
        // Its answer is saved.
        assert!(
            store
                .append_answer(&ids[0], NewTurn::assistant("a", "lab:m"))
                .is_ok()
        );
        assert_eq!(store.touch("unknown"), Ok(false));
        assert!(
            !rows(&store).iter().any(|row| row.id == "unknown"),
            "touch makes no row"
        );
    }

    #[test]
    fn archive_and_unarchive_at_exactly_sixty() {
        let store = store();
        let ids: Vec<String> = (0..60)
            .map(|n| store.first_message(&format!("q{n}"), "").unwrap().0.id)
            .collect();
        // The reader archives a thread: it leaves the list with its transcript.
        store.archive(&ids[30]).unwrap();
        assert_eq!(rows(&store).len(), 59);
        let entry = archived(&store).remove(0);
        assert_eq!(entry.file, format!("{}.jsonl", ids[30]));
        // Another thread fills the list to 60 again.
        store.first_message("q60", "").unwrap();
        assert_eq!(rows(&store).len(), 60);
        // Unarchive at 60: listed as the newest, and an older thread goes in
        // its place; no archive row names a missing file.
        let key = ArchiveKey {
            id: entry.id.clone(),
            created: entry.created,
        };
        let row = store.unarchive(&key).unwrap();
        let listed = rows(&store);
        assert_eq!(
            listed[0].id, row.id,
            "the restored thread is the most recently used"
        );
        assert_eq!(listed.len(), 60);
        assert_eq!(store.load(&row.id).len(), 1, "its transcript is back");
        let gone = archived(&store);
        assert_eq!(
            gone.iter().map(|t| t.id.clone()).collect::<Vec<_>>(),
            [ids[0].clone()]
        );
        assert!(gone.iter().all(|t| !t.transcript_missing));
        // Refusals: a key no archive row holds; an id that is listed again.
        assert_eq!(store.unarchive(&key), Err(StoreError::NotFound));
        store.archive(&row.id).unwrap();
        let again = archived(&store)
            .into_iter()
            .find(|t| t.id == row.id)
            .unwrap();
        store.append(&row.id, NewTurn::user("back")).unwrap();
        assert_eq!(
            store.unarchive(&ArchiveKey {
                id: again.id,
                created: again.created
            }),
            Err(StoreError::AlreadyListed)
        );
        assert_eq!(store.archive("unknown"), Err(StoreError::NotFound));
    }

    #[test]
    fn an_unreadable_index_refuses_every_write_and_changes_nothing() {
        let store = store();
        let (row, turn) = store.first_message("q", "auto").unwrap();
        store.set_index_unreadable(true);
        assert_eq!(store.list(), IndexState::Unreadable);
        let key = ArchiveKey {
            id: row.id.clone(),
            created: row.created,
        };
        let refusals = [
            store.first_message("x", "").err(),
            store.append(&row.id, NewTurn::user("x")).err(),
            store
                .append_answer(&row.id, NewTurn::assistant("x", "p"))
                .err(),
            store.rename(&row.id, "x").err(),
            store.pin(&row.id, "x").err(),
            store.supersede(&row.id, &turn.id).err(),
            store.touch(&row.id).err(),
            store.archive(&row.id).err(),
            store.unarchive(&key).err(),
        ];
        assert!(
            refusals
                .iter()
                .all(|r| *r == Some(StoreError::IndexUnreadable)),
            "{refusals:?}"
        );
        store.set_index_unreadable(false);
        assert_eq!(rows(&store), std::slice::from_ref(&row));
        assert_eq!(store.load(&row.id), [turn]);
    }

    /// The histories `chat/messages.json` built in Python with `append` and
    /// `supersede`, replayed here: `recent` gives the same roles and texts.
    #[test]
    fn python_s_scripted_histories_replay_to_the_same_recent_turns() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("parity")
            .join("chat")
            .join("messages.json");
        let golden: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let cases = golden["cases"].as_array().unwrap();
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let store = store();
            let mut ids = Vec::new();
            for op in case["ops"].as_array().unwrap() {
                if op["op"] == "append" {
                    let turn = NewTurn {
                        role: op["role"].as_str().unwrap().into(),
                        text: op["text"].as_str().unwrap().into(),
                        error: op["error"].as_str().unwrap().into(),
                        ..NewTurn::default()
                    };
                    ids.push(store.append("thread", turn).unwrap().id);
                } else {
                    let of = op["of"].as_u64().unwrap() as usize;
                    store.supersede("thread", &ids[of]).unwrap();
                }
            }
            let recent: Vec<(String, String)> = store
                .recent("thread")
                .into_iter()
                .map(|turn| (turn.role, turn.text))
                .collect();
            let expected: Vec<(String, String)> = case["recent"]
                .as_array()
                .unwrap()
                .iter()
                .map(|turn| {
                    (
                        turn["role"].as_str().unwrap().to_owned(),
                        turn["text"].as_str().unwrap().to_owned(),
                    )
                })
                .collect();
            assert_eq!(recent, expected, "{name}");
        }
        assert!(cases.len() >= 30);
    }
}
