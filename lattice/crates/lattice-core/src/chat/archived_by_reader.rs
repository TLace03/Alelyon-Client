//! Why a conversation is in the archive: the reader's own Archive, recorded
//! natively (the chat core's spec §5.2 rule S21, §5.6; row G4).
//!
//! Python's archive rows carry no reason, and native adds no key to them
//! (S24). So the reader's Archive is recorded here, outside the shared store,
//! in `<globals>/lattice_native/chat/archived_by_reader.json`: a JSON list of
//! `{"id", "created", "at"}`, one entry per archive the reader made, written
//! with [`fsx::atomic_write`]. The archived list joins it by `(id, created)`:
//! `Reader` when an entry matches, otherwise `Cap` (§5.6).
//!
//! - [`read`]: the recorded keys. An absent file is no entries; a file that
//!   cannot be read or parsed is no entries too, so the list shows `Cap`,
//!   which is what the row's own content says.
//! - [`record`]: one entry more. Entries are only ever added (ND1); an entry
//!   for a thread that was unarchived stays and simply matches again if the
//!   same `(id, created)` is archived by the reader again. A file that exists
//!   and cannot be parsed is never overwritten: the call refuses.
//! - [`archive_by_reader`]: the reader's Archive at the store's seam: the
//!   row's `(id, created)` is read, the store archives the thread (S21, through
//!   gate G-WEB: refused, and nothing written anywhere, while the gate is
//!   closed), and the reason is recorded only once the archive holds that
//!   `(id, created)`.
//!
//! The caller of [`archive_by_reader`] is the agent chat's `archive`
//! (`AgentChatService::archive`, `convo::agent`, row E11); the plain chat's
//! `ChatService` has no Archive.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use lattice_protocol::conversation::ArchiveKey;
use serde_json::{Map, Value};

use super::archive::{ArchiveRow, ArchiveState};
use super::store::IndexState;
use super::transcript::{StoreError, TranscriptStore};
use crate::fsx;

/// The record's file name, in `<globals>/lattice_native/chat/`.
pub const FILE: &str = "archived_by_reader.json";

/// The record's path under `native_chat_dir` (`StateRoot::native_chat_dir`).
pub fn path(native_chat_dir: &Path) -> PathBuf {
    native_chat_dir.join(FILE)
}

/// The entries as stored; `Ok(None)` when there is no file.
fn entries(native_chat_dir: &Path) -> io::Result<Option<Vec<Value>>> {
    let bytes = match fs::read(path(native_chat_dir)) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(Value::Array(items)) => Ok(Some(items)),
        _ => Err(io::Error::new(
            ErrorKind::InvalidData,
            "the reader's archive record is not a JSON list",
        )),
    }
}

/// One entry as a key, when it is one.
fn key_of(entry: &Value) -> Option<ArchiveKey> {
    let id = entry.get("id")?.as_str()?;
    let created = entry.get("created")?.as_f64()?;
    Some(ArchiveKey {
        id: id.to_owned(),
        created,
    })
}

/// The `(id, created)` of every conversation the reader archived here. A
/// file that is absent, unreadable or not a list gives none.
pub fn read(native_chat_dir: &Path) -> Vec<ArchiveKey> {
    entries(native_chat_dir)
        .ok()
        .flatten()
        .unwrap_or_default()
        .iter()
        .filter_map(key_of)
        .collect()
}

/// Add `(key, at)` to the record. A record that exists and cannot be read as
/// a list is left as it is, and the call fails.
pub fn record(native_chat_dir: &Path, key: &ArchiveKey, at: f64) -> io::Result<()> {
    let mut items = entries(native_chat_dir)?.unwrap_or_default();
    let mut entry = Map::new();
    entry.insert("id".into(), Value::String(key.id.clone()));
    entry.insert("created".into(), finite(key.created)?);
    entry.insert("at".into(), finite(at)?);
    items.push(Value::Object(entry));
    let mut bytes = serde_json::to_vec_pretty(&Value::Array(items)).map_err(io::Error::other)?;
    bytes.push(b'\n');
    fs::create_dir_all(native_chat_dir)?;
    fsx::atomic_write(&path(native_chat_dir), &bytes)
}

fn finite(value: f64) -> io::Result<Value> {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "not a finite time"))
}

/// What the reader's Archive did.
#[derive(Clone, Debug, PartialEq)]
pub struct Archived {
    /// The conversation, as the archive identifies it.
    pub key: ArchiveKey,
    /// Whether the reason was recorded. `false` leaves the conversation
    /// listed as `Cap`: archived all the same, only without the reader's mark.
    pub recorded: bool,
}

/// The reader's Archive (S21): archive the thread `id` in `store`, then record
/// the reason under `native_chat_dir` at time `at`. Nothing is recorded when
/// the store refuses (the gate, an unreadable index or archive, an unknown
/// thread), so a refused Archive leaves no mark anywhere.
pub fn archive_by_reader(
    store: &dyn TranscriptStore,
    native_chat_dir: &Path,
    id: &str,
    at: f64,
) -> Result<Archived, StoreError> {
    let created = match store.list() {
        IndexState::Rows(rows) => rows
            .into_iter()
            .find(|row| row.id == id)
            .map(|row| row.created)
            .ok_or(StoreError::NotFound)?,
        IndexState::Absent => return Err(StoreError::NotFound),
        IndexState::Unreadable => return Err(StoreError::IndexUnreadable),
    };
    store.archive(id)?;
    let key = ArchiveKey {
        id: id.to_owned(),
        created,
    };
    // Recorded only for the conversation the archive now holds: another
    // process may have replaced the thread between the read and the archive.
    let held = match store.archived() {
        ArchiveState::Rows(rows) => rows.iter().any(|row| {
            matches!(row, ArchiveRow::Thread(thread)
                if thread.id == key.id && thread.created.to_bits() == key.created.to_bits())
        }),
        _ => false,
    };
    let recorded = held && record(native_chat_dir, &key, at).is_ok();
    Ok(Archived { key, recorded })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    fn key(id: &str, created: f64) -> ArchiveKey {
        ArchiveKey {
            id: id.into(),
            created,
        }
    }

    #[test]
    fn entries_are_added_and_read_back_by_id_and_created() {
        let dir = TempDir::new("abr-roundtrip");
        let native = dir.path().join("native");
        assert!(read(&native).is_empty(), "no file, no entries");
        record(&native, &key("a1", 1.5), 10.0).unwrap();
        record(&native, &key("b2", 0.1 + 0.2), 11.0).unwrap();
        assert_eq!(read(&native), [key("a1", 1.5), key("b2", 0.1 + 0.2)]);
        // Exactly the f64 given, and the time of the archive.
        let stored: Value = serde_json::from_slice(&fs::read(path(&native)).unwrap()).unwrap();
        assert_eq!(
            stored[1]["created"].as_f64().unwrap().to_bits(),
            (0.1f64 + 0.2).to_bits()
        );
        assert_eq!(stored[1]["at"], 11.0);
    }

    #[test]
    fn a_damaged_record_is_never_overwritten_and_reads_as_no_entries() {
        let dir = TempDir::new("abr-damaged");
        let native = dir.path().join("native");
        fs::create_dir_all(&native).unwrap();
        fs::write(path(&native), b"{not a list").unwrap();
        assert!(read(&native).is_empty());
        assert!(record(&native, &key("a1", 1.0), 2.0).is_err());
        assert_eq!(fs::read(path(&native)).unwrap(), b"{not a list");
        // Entries that are not keys are kept, and skipped when read.
        fs::write(
            path(&native),
            br#"[1, {"id": "x"}, {"id": "ok", "created": 3.0}]"#,
        )
        .unwrap();
        assert_eq!(read(&native), [key("ok", 3.0)]);
        record(&native, &key("n", 4.0), 5.0).unwrap();
        let stored: Value = serde_json::from_slice(&fs::read(path(&native)).unwrap()).unwrap();
        assert_eq!(stored.as_array().unwrap().len(), 4, "nothing dropped");
    }

    #[test]
    fn a_time_that_is_not_finite_is_refused_and_nothing_is_written() {
        let dir = TempDir::new("abr-nan");
        let native = dir.path().join("native");
        assert!(record(&native, &key("a1", f64::NAN), 1.0).is_err());
        assert!(!path(&native).exists());
    }
}
