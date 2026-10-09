//! A conversation's sidecar, bound to its row in the shared chat store
//! (the chat core's spec §5.4, §5.5; row G4).
//!
//! The shared store (`chat::store::SharedThreadStore`, the production
//! `TranscriptStore`) holds the conversation's words; the native sidecar
//! (`convo::sidecar`, row B5) holds everything else, keyed by the thread id.
//! An id alone is not enough: a late save can make a thread again under an id
//! an earlier conversation had, with a new `created` [`history.py`'s
//! `append` makes a missing row with `created = ts`]. So a sidecar belongs to
//! the shared row with the same id **and** the same `created`:
//!
//! - [`view`] reads, and writes nothing: the row, and whether a sidecar is
//!   bound to it ([`SidecarView::Bound`]), there is none (a web chat,
//!   `Origin::Web`, §5.4), or the one there is an earlier conversation's
//!   ([`SidecarView::Stale`]: shown as no sidecar, never replayed);
//! - [`bind_for_writing`] opens the sidecar for writing against the row's
//!   own `created`: a stale one is moved aside to `<id>~<its created as 16
//!   hex digits>` and a fresh one starts (B5's `open_for_writing`). Nothing is
//!   removed (ND1).
//!
//! The `created` always comes from the shared store's row, never from the
//! sidecar. A thread that is not listed (absent, archived, or an index that
//! cannot be read) binds nothing: its sidecar stays where it is, and binds
//! again by `(id, created)` after an unarchive (§5.4).
//!
//! The callers that write items are the agent chat's open and send
//! (`convo::agent`, row E11); the plain chat writes no sidecar.

use lattice_protocol::conversation::Origin;

use super::sidecar::{Binding, Meta, NewMeta, Sidecar, SidecarError, SidecarStore, same_created};
use crate::chat::store::{IndexRow, IndexState};
use crate::chat::transcript::{StoreError, TranscriptStore};

/// Why no sidecar could be bound.
#[derive(Debug)]
pub enum BindError {
    /// The shared store has no such listed thread, or cannot be read.
    Store(StoreError),
    Sidecar(SidecarError),
}

impl std::fmt::Display for BindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(error) => error.fmt(f),
            Self::Sidecar(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for BindError {}

/// What the native record of a listed thread is.
#[derive(Clone, Debug, PartialEq)]
pub enum SidecarView {
    /// None: a web chat, or a native one that never needed a record.
    None,
    /// It belongs to this row. `None`: its `meta.json` is missing (a crash
    /// before it was written) and the item log's header names this row; the
    /// next open for writing writes it.
    Bound(Option<Meta>),
    /// It belongs to an earlier conversation with this id (`created` as the
    /// record states it, `None` when it states none): not this thread's.
    Stale { recorded: Option<f64> },
}

impl SidecarView {
    /// Where the conversation started, as the list shows it (§5.4): a thread
    /// with no record of its own here is the web's.
    pub fn origin(&self) -> Origin {
        match self {
            Self::Bound(Some(meta)) => meta.origin,
            // A record only the native Lattice makes.
            Self::Bound(None) => Origin::Native,
            Self::None | Self::Stale { .. } => Origin::Web,
        }
    }
}

/// The listed row of `id`.
pub fn listed_row(transcripts: &dyn TranscriptStore, id: &str) -> Result<IndexRow, StoreError> {
    match transcripts.list() {
        IndexState::Rows(rows) => rows
            .into_iter()
            .find(|row| row.id == id)
            .ok_or(StoreError::NotFound),
        IndexState::Absent => Err(StoreError::NotFound),
        IndexState::Unreadable => Err(StoreError::IndexUnreadable),
    }
}

/// The row of `id` and its sidecar, read only: nothing is moved or made.
pub fn view(
    transcripts: &dyn TranscriptStore,
    sidecars: &SidecarStore,
    id: &str,
) -> Result<(IndexRow, SidecarView), BindError> {
    let row = listed_row(transcripts, id).map_err(BindError::Store)?;
    let dir = sidecars.conversation_dir(id).map_err(BindError::Sidecar)?;
    if !dir.exists() {
        return Ok((row, SidecarView::None));
    }
    let meta = sidecars.read_meta(id).map_err(BindError::Sidecar)?;
    let view = match meta {
        Some(meta) if same_created(meta.created, row.created) => SidecarView::Bound(Some(meta)),
        Some(meta) => SidecarView::Stale {
            recorded: Some(meta.created),
        },
        None => match sidecars.read_items(id) {
            // A crash before `meta.json`: the log's header says whose it is.
            Ok(log) if same_created(log.header.created, row.created) => SidecarView::Bound(None),
            Ok(log) => SidecarView::Stale {
                recorded: Some(log.header.created),
            },
            // No item log either: nothing was recorded (B5 uses such a folder
            // as it is).
            Err(SidecarError::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {
                SidecarView::None
            }
            // Neither file says whose it is: B5 moves such a record aside
            // as `<id>~unbound` before it writes.
            Err(_) => SidecarView::Stale { recorded: None },
        },
    };
    Ok((row, view))
}

/// Open `id`'s sidecar for writing, bound to the shared row's `created`
/// (§5.5): an earlier conversation's record is moved aside first.
pub fn bind_for_writing(
    transcripts: &dyn TranscriptStore,
    sidecars: &SidecarStore,
    id: &str,
    new: NewMeta,
) -> Result<(IndexRow, Sidecar, Binding), BindError> {
    let row = listed_row(transcripts, id).map_err(BindError::Store)?;
    let (sidecar, binding) = sidecars
        .open_for_writing(id, row.created, new)
        .map_err(BindError::Sidecar)?;
    Ok((row, sidecar, binding))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use lattice_protocol::conversation::{Mode, Origin};

    use super::super::item::Item;
    use super::super::sidecar::{Binding, NewMeta, SidecarStore, created_tag};
    use super::*;
    use crate::chat::store::{IdSource, SharedThreadStore};
    use crate::chat::transcript::{NewTurn, TranscriptStore};
    use crate::clock::Clock;
    use crate::testkit::TempDir;

    /// A clock that reads 1000.25, 1001.25, … (exact in binary, so a test can
    /// say which `created` a row has).
    fn ticking() -> Clock {
        let next = Arc::new(AtomicU64::new(0));
        Arc::new(move || 1000.25 + next.fetch_add(1, Ordering::Relaxed) as f64)
    }

    fn ids(id: &'static str) -> IdSource {
        Arc::new(move || id.to_owned())
    }

    /// The shared store with its writes open (the test constructor, with a
    /// fixed id source), and the sidecars beside it, in one temporary root.
    fn stores(dir: &TempDir) -> (SharedThreadStore, SidecarStore) {
        let shared = SharedThreadStore::with_gate(
            dir.path().join("lattice_chat"),
            ids("abc123def456"),
            true,
        )
        .clocked(ticking());
        (shared, SidecarStore::new(dir.path().join("native")))
    }

    fn native() -> NewMeta {
        NewMeta {
            workspace: None,
            mode: Mode::Ask,
            origin: Origin::Native,
        }
    }

    /// Every file under `root` with its bytes (folders empty, `.lock` files
    /// as "lock": a held lock refuses reads).
    fn files(root: &Path) -> BTreeMap<String, Vec<u8>> {
        let mut out = BTreeMap::new();
        let mut pending = vec![root.to_path_buf()];
        while let Some(dir) = pending.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries {
                let path = entry.unwrap().path();
                let name = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                if path.is_dir() {
                    out.insert(name, Vec::new());
                    pending.push(path);
                } else if name.ends_with(".lock") {
                    out.insert(name, b"lock".to_vec());
                } else {
                    out.insert(name, fs::read(&path).unwrap());
                }
            }
        }
        out
    }

    fn notice(text: &str) -> Item {
        Item::Notice {
            text: text.into(),
            at: 1.0,
        }
    }

    #[test]
    fn a_web_chat_has_no_sidecar_and_reads_as_the_web_s() {
        let dir = TempDir::new("bind-web");
        let (shared, sidecars) = stores(&dir);
        let (row, _) = shared.first_message("from the web", "local").unwrap();
        let before = files(dir.path());
        let (listed, seen) = view(&shared, &sidecars, &row.id).unwrap();
        assert_eq!(listed, row);
        assert_eq!(seen, SidecarView::None);
        assert_eq!(seen.origin(), Origin::Web);
        assert_eq!(files(dir.path()), before, "reading made nothing");
    }

    #[test]
    fn a_sidecar_binds_to_the_shared_row_by_its_created() {
        let dir = TempDir::new("bind-same");
        let (shared, sidecars) = stores(&dir);
        let (row, _) = shared.first_message("q", "local").unwrap();
        let (_, sidecar, binding) =
            bind_for_writing(&shared, &sidecars, &row.id, native()).unwrap();
        assert_eq!(binding, Binding::Created);
        assert_eq!(sidecar.created().to_bits(), row.created.to_bits());
        sidecar.append(&notice("kept")).unwrap();
        drop(sidecar);
        let (_, seen) = view(&shared, &sidecars, &row.id).unwrap();
        assert!(
            matches!(&seen, SidecarView::Bound(Some(meta))
                if meta.created.to_bits() == row.created.to_bits()),
            "{seen:?}"
        );
        assert_eq!(seen.origin(), Origin::Native);
        // The row moves on (a later turn); `created` does not, so it binds again.
        shared.append(&row.id, NewTurn::user("again")).unwrap();
        let (_, sidecar, binding) =
            bind_for_writing(&shared, &sidecars, &row.id, native()).unwrap();
        assert_eq!(binding, Binding::Bound);
        drop(sidecar);
        assert_eq!(
            sidecars.read_items(&row.id).unwrap().items,
            [notice("kept")]
        );
    }

    /// §5.5: a thread made again under an earlier conversation's id (a late
    /// save: Python's `append` makes a missing row with `created = ts`) does
    /// not inherit that conversation's record. Reading shows it as not this
    /// thread's and moves nothing; opening for writing moves it aside under
    /// its own `created` and starts a fresh one; nothing is removed.
    /// Mutant: the binding takes `created` from the sidecar, not the row.
    #[test]
    fn a_thread_made_again_under_its_id_moves_the_earlier_record_aside() {
        let dir = TempDir::new("bind-again");
        let (shared, sidecars) = stores(&dir);
        let (first, _) = shared
            .first_message("the first conversation", "local")
            .unwrap();
        let (_, sidecar, _) = bind_for_writing(&shared, &sidecars, &first.id, native()).unwrap();
        sidecar
            .append(&notice("the first conversation's tool trail"))
            .unwrap();
        drop(sidecar);
        // The thread leaves the list, and a late save makes it again.
        shared.archive(&first.id).unwrap();
        assert!(
            matches!(
                bind_for_writing(&shared, &sidecars, &first.id, native()),
                Err(BindError::Store(StoreError::NotFound))
            ),
            "an archived thread binds nothing"
        );
        shared
            .append(&first.id, NewTurn::user("a late save"))
            .unwrap();
        let before = files(dir.path());
        let (again, seen) = view(&shared, &sidecars, &first.id).unwrap();
        assert_eq!(again.id, first.id);
        assert_ne!(
            again.created.to_bits(),
            first.created.to_bits(),
            "a new conversation"
        );
        assert_eq!(
            seen,
            SidecarView::Stale {
                recorded: Some(first.created)
            }
        );
        assert_eq!(seen.origin(), Origin::Web);
        assert_eq!(files(dir.path()), before, "reading moved nothing");
        let (_, sidecar, binding) =
            bind_for_writing(&shared, &sidecars, &first.id, native()).unwrap();
        let aside = format!("{}~{}", first.id, created_tag(first.created));
        println!("binding: {binding:?}");
        assert_eq!(binding, Binding::MovedAside(aside.clone()));
        assert_eq!(sidecar.created().to_bits(), again.created.to_bits());
        drop(sidecar);
        let fresh = sidecars.read_items(&first.id).unwrap();
        assert_eq!(fresh.header.created.to_bits(), again.created.to_bits());
        assert!(
            fresh.items.is_empty(),
            "the new thread starts with no record"
        );
        let moved = fs::read_to_string(
            sidecars
                .root()
                .join("conversations")
                .join(&aside)
                .join("items.jsonl"),
        )
        .unwrap();
        assert!(
            moved.contains("the first conversation's tool trail"),
            "kept, aside"
        );
        // Every file there before is still there, under its old or new name.
        let after = files(dir.path());
        let old_dir = format!("native/conversations/{}/", first.id);
        let new_dir = format!("native/conversations/{aside}/");
        for (name, bytes) in &before {
            if bytes.is_empty() || name.ends_with(".lock") {
                continue;
            }
            let renamed = name.replacen(&old_dir, &new_dir, 1);
            assert!(
                after.get(name) == Some(bytes) || after.get(&renamed) == Some(bytes),
                "{name} was lost"
            );
        }
    }

    /// The binding reads the shared store and writes only the native sidecar:
    /// through the production store, whose writes are open since G5, a web
    /// chat's sidecar is bound and the shared store's bytes do not change.
    #[test]
    fn binding_writes_nothing_into_the_shared_store() {
        let dir = TempDir::new("bind-gate");
        let (writer, sidecars) = stores(&dir);
        let (row, _) = writer.first_message("from the web", "local").unwrap();
        let production = SharedThreadStore::new(writer.dir());
        assert!(production.writes_open());
        let before = files(writer.dir());
        let (_, sidecar, binding) =
            bind_for_writing(&production, &sidecars, &row.id, native()).unwrap();
        assert_eq!(binding, Binding::Created);
        sidecar.append(&notice("native only")).unwrap();
        drop(sidecar);
        assert_eq!(files(writer.dir()), before);
    }
}
