//! A conversation's staged changes and its staged view
//! (the chat core's spec §7.4, OD2: diff review before any write). Not a
//! port.
//!
//! `edit_file`, `write_file` and `delete_file` (`tools::edit`) never touch the
//! folder. What they would do is recorded here as a [`StagedChange`]: its
//! file's state when first staged (the **base**), the bytes a Keep would
//! leave (the **new** state), and the `edit_file` calls that made it (the
//! **ops**, which a Keep applies again when the file moved on disk, row E3).
//! The reader's Keep is the write (row E3); until then:
//! - **ST1. No byte reaches a workspace file.** Staging writes only to the
//!   conversation's sidecar: the new bytes and the base bytes as
//!   content-addressed blobs (`BlobKind::Staged`, written as given: they are
//!   what a Keep writes), then one `Staged` item with the change in full.
//!   The record is written before the change is live, so the model is never
//!   told of a change the record lacks.
//! - **ST2. One live change per path.** A later staging of the same path
//!   updates the same change: its `new` state, its kind, and its ops. The
//!   base stays the disk state at first staging. Paths are compared without
//!   case on Windows, as its file system names them.
//! - **ST4.** The change carries the path's authority class, decided by the
//!   path rules on the derived long-name path (`workspace::paths`).
//! - **Ops are kept only while every staging of the change was an edit**
//!   (kind `Edit`): an overwrite or a delete cannot be applied again to
//!   another text, so such a change has none, and a file that moved on disk
//!   makes it a conflict rather than a guess.
//! - **The staged view (§7.4.2).** [`Staging`] is the read tools' overlay: a
//!   live change's file reads with its new bytes, a staged creation is listed,
//!   a staged deletion disappears. Other conversations and commands see the
//!   disk.
//! - Encoding: a staged file's bytes keep its base's byte-order mark and,
//!   when its line ends were uniform, those line ends (§7.4.1, §7.5 step 4),
//!   so the blob holds exactly what a whole Keep writes.
//!
//! The live states are `Pending`, `Rebased` and `PartlyKept`: their new
//! bytes are what the conversation sees. A `Conflict` leaves the staged view
//! (the agent is told to read the file again) but still holds the command
//! gate; `Kept` and `Undone` are history.
//!
//! [`review`] is the reader's side (row E3): the diff in hunks, Keep and Undo
//! per hunk, per file and for all, and the expected-hash write.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Read};
use std::sync::{Arc, Mutex, MutexGuard};

use lattice_protocol::TurnId;
use lattice_protocol::conversation::{
    CallId, ChangeId, ChangeKind, ChangeOrigin, ChangeState, is_change_id,
};
use sha2::{Digest, Sha256};

use crate::convo::item::{BaseState, EditOp, Eol, Item, NewState, StagedChange};
use crate::convo::sidecar::{BlobKind, Sidecar, SidecarError};
use crate::sha::sha256_hex;
use crate::tools::read::{Overlay, Staged};

/// The UTF-8 byte-order mark.
pub const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// The line ends of `bytes`: only `\r\n`, only a lone `\n`, both, or none.
pub fn eol_of(bytes: &[u8]) -> Eol {
    let (mut crlf, mut lf) = (false, false);
    for (at, byte) in bytes.iter().enumerate() {
        if *byte == b'\n' {
            if at > 0 && bytes[at - 1] == b'\r' {
                crlf = true;
            } else {
                lf = true;
            }
        }
    }
    match (crlf, lf) {
        (true, true) => Eol::Mixed,
        (true, false) => Eol::Crlf,
        (false, true) => Eol::Lf,
        (false, false) => Eol::None,
    }
}

/// `text` encoded as a file whose line ends are `eol` and whose byte-order
/// mark is `bom`: uniform line ends are applied (a lone `\n` becomes `\r\n`
/// for CRLF, a `\r\n` becomes `\n` for LF); mixed or no line ends leave the
/// text's own.
pub fn encode(text: &str, eol: Eol, bom: bool) -> Vec<u8> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let body = match eol {
        Eol::Crlf => text.replace("\r\n", "\n").replace('\n', "\r\n"),
        Eol::Lf => text.replace("\r\n", "\n"),
        Eol::Mixed | Eol::None => text.to_owned(),
    };
    let mut out = Vec::with_capacity(body.len() + 3);
    if bom {
        out.extend_from_slice(BOM);
    }
    out.extend_from_slice(body.as_bytes());
    out
}

/// Why an edit could not be applied to a text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpError {
    /// The file is not text `edit_file` can change, with the sentence.
    NotText(String),
    NotFound,
    /// It occurs this many times and `replace_all` is false.
    Many(usize),
}

impl OpError {
    /// The tool's one sentence (§7.4.1).
    pub fn sentence(&self) -> String {
        match self {
            Self::NotText(sentence) => sentence.clone(),
            Self::NotFound => "old_string was not found".to_owned(),
            Self::Many(n) => {
                format!("old_string occurs {n} times; give more context, or set replace_all")
            }
        }
    }
}

/// A file's bytes as text `edit_file` may change: UTF-8, its byte-order mark
/// (if any) taken off and reported.
pub fn text_of(bytes: &[u8]) -> Result<(&str, bool), OpError> {
    if bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF]) {
        return Err(OpError::NotText(
            "That file is UTF-16 text; edit_file changes UTF-8 files only.".to_owned(),
        ));
    }
    if bytes[..bytes.len().min(crate::tools::read::BINARY_PROBE)].contains(&0) {
        return Err(OpError::NotText(format!(
            "binary file, {} bytes; edit_file changes text files only",
            bytes.len()
        )));
    }
    let (body, bom) = match bytes.strip_prefix(BOM) {
        Some(rest) => (rest, true),
        None => (bytes, false),
    };
    std::str::from_utf8(body)
        .map(|text| (text, bom))
        .map_err(|_| {
            OpError::NotText(
                "That file is not UTF-8 text, so edit_file cannot change it.".to_owned(),
            )
        })
}

/// One `edit_file` call applied to a file's bytes, with the unique-match rule
/// (§7.4.1): an exact match of `old_string`, once unless `replace_all`. When
/// the file's line ends are CRLF throughout, matching runs on the LF form (of
/// the file and of both strings) and the result is written back as CRLF;
/// with mixed line ends it runs on the raw text. The byte-order mark is kept.
pub fn apply_op(current: &[u8], op: &EditOp) -> Result<Vec<u8>, OpError> {
    let (text, bom) = text_of(current)?;
    let eol = eol_of(text.as_bytes());
    let lf_form = eol == Eol::Crlf;
    let (text, old, new) = if lf_form {
        (
            text.replace("\r\n", "\n"),
            op.old_string.replace("\r\n", "\n"),
            op.new_string.replace("\r\n", "\n"),
        )
    } else {
        (
            text.to_owned(),
            op.old_string.clone(),
            op.new_string.clone(),
        )
    };
    if old.is_empty() {
        return Err(OpError::NotFound);
    }
    let found = text.matches(old.as_str()).count();
    if found == 0 {
        return Err(OpError::NotFound);
    }
    if found > 1 && !op.replace_all {
        return Err(OpError::Many(found));
    }
    let changed = if op.replace_all {
        text.replace(old.as_str(), &new)
    } else {
        text.replacen(old.as_str(), &new, 1)
    };
    let mut out = Vec::with_capacity(changed.len() + 3);
    if bom {
        out.extend_from_slice(BOM);
    }
    if lf_form {
        out.extend_from_slice(changed.replace('\n', "\r\n").as_bytes());
    } else {
        out.extend_from_slice(changed.as_bytes());
    }
    Ok(out)
}

/// Lines added and removed between two texts (the `+a −b` the model and the
/// Changes panel see), by `similar`'s line diff.
pub fn added_removed(old: &str, new: &str) -> (u32, u32) {
    let diff = similar::TextDiff::from_lines(old, new);
    let (mut added, mut removed) = (0u32, 0u32);
    for change in diff.iter_all_changes() {
        match change.tag() {
            similar::ChangeTag::Insert => added = added.saturating_add(1),
            similar::ChangeTag::Delete => removed = removed.saturating_add(1),
            similar::ChangeTag::Equal => {}
        }
    }
    (added, removed)
}

/// A file's state on disk when it is first staged, read through the handle
/// the path rules opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Base {
    pub state: BaseState,
    /// The bytes, when they were read whole (files of at most 2 MiB).
    pub bytes: Option<Vec<u8>>,
    /// Its line count, for the `−b` of a delete when the bytes are not kept.
    pub lines: u32,
}

impl Base {
    /// An absent file: a creation.
    pub fn absent() -> Self {
        Self {
            state: BaseState::Absent,
            bytes: None,
            lines: 0,
        }
    }

    /// A present file, from its bytes read whole.
    pub fn of_bytes(bytes: Vec<u8>) -> Self {
        let state = BaseState::Present {
            sha256: sha256_hex(&bytes),
            bytes: bytes.len() as u64,
            eol: eol_of(&bytes),
            bom: bytes.starts_with(BOM),
        };
        let lines = line_count(&bytes);
        Self {
            state,
            bytes: Some(bytes),
            lines,
        }
    }

    /// A present file too large to keep whole (a delete of a file over
    /// 2 MiB): its hash, length and line count, read in pieces. Its line ends
    /// are not recorded (`None`): nothing re-encodes a deleted file.
    pub fn of_reader(mut reader: impl Read) -> io::Result<Self> {
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024];
        let (mut total, mut newlines, mut first) = (0u64, 0u32, Vec::<u8>::new());
        let mut last = None;
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            let chunk = &buffer[..read];
            hasher.update(chunk);
            if first.len() < 3 {
                first.extend(chunk.iter().take(3 - first.len()));
            }
            newlines = newlines.saturating_add(
                u32::try_from(chunk.iter().filter(|b| **b == b'\n').count()).unwrap_or(u32::MAX),
            );
            last = chunk.last().copied();
            total += read as u64;
        }
        let digest = hasher.finalize();
        let sha256: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        let lines = newlines.saturating_add(u32::from(last.is_some_and(|byte| byte != b'\n')));
        Ok(Self {
            state: BaseState::Present {
                sha256,
                bytes: total,
                eol: Eol::None,
                bom: first == BOM,
            },
            bytes: None,
            lines,
        })
    }
}

/// Lines in `bytes`, counting a last line without a line end.
fn line_count(bytes: &[u8]) -> u32 {
    let newlines = bytes.iter().filter(|byte| **byte == b'\n').count();
    let partial = usize::from(bytes.last().is_some_and(|byte| *byte != b'\n'));
    u32::try_from(newlines + partial).unwrap_or(u32::MAX)
}

/// Which tool asks to stage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    /// `edit_file`: the op that made the new bytes.
    Edit(EditOp),
    /// `write_file`.
    Write,
    /// `delete_file`.
    Delete,
}

/// What a staging tool asks to record.
#[derive(Clone, Debug)]
pub struct Proposal {
    /// The derived path (§6.2 WP10).
    pub path: String,
    pub authority: bool,
    /// The disk state now; used only when the path has no live change.
    pub base: Base,
    /// The bytes a Keep would leave, or `None` for a delete.
    pub new: Option<Vec<u8>>,
    pub action: Action,
}

/// What [`Staging::record`] staged: the change, and its line counts against
/// its base.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recorded {
    pub change: StagedChange,
    pub added: u32,
    pub removed: u32,
}

/// Why nothing was staged.
#[derive(Debug)]
pub enum StageError {
    /// The sidecar could not take the record; nothing changed.
    Record(SidecarError),
    /// The change moved on since this snapshot was read (a later staging, or
    /// another review's update): nothing changed.
    Stale,
}

impl StageError {
    pub fn sentence(&self) -> &'static str {
        match self {
            Self::Record(_) => "Lattice could not record the staged change, so nothing was staged.",
            Self::Stale => "That change changed while it was being reviewed; look at it again.",
        }
    }
}

/// A change with its bytes, as [`Staging::snapshot`] gives it and
/// [`Staging::update`] takes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub change: StagedChange,
    /// The base's bytes, when they were read whole.
    pub base: Option<Vec<u8>>,
    /// The new bytes, when the change leaves bytes.
    pub new: Option<Vec<u8>>,
    /// The base's line count.
    pub base_lines: u32,
    /// The change's revision when this snapshot was read
    /// ([`Staging::snapshot`]); [`Staging::update`] refuses a snapshot whose
    /// change has moved on since. `0` for a change not yet staged.
    pub revision: u64,
}

/// A change and the bytes it holds in memory.
#[derive(Clone, Debug)]
struct Entry {
    change: StagedChange,
    /// The new bytes, for a change whose new state is bytes.
    new: Option<Vec<u8>>,
    /// The base bytes, when they were read whole.
    base: Option<Vec<u8>>,
    /// The base's line count.
    base_lines: u32,
    /// Bumped by every staging and update of the change.
    revision: u64,
}

impl Entry {
    fn is_live(&self) -> bool {
        is_live(&self.change.state)
    }
}

/// `Pending`, `Rebased` and `PartlyKept`: the change's new bytes are the
/// conversation's view of its file.
pub fn is_live(state: &ChangeState) -> bool {
    matches!(
        state,
        ChangeState::Pending | ChangeState::Rebased | ChangeState::PartlyKept
    )
}

/// `Pending`, `Rebased` and `Conflict`: commands wait for these (§7.5's
/// command gate).
pub fn is_waiting(state: &ChangeState) -> bool {
    matches!(
        state,
        ChangeState::Pending | ChangeState::Rebased | ChangeState::Conflict { .. }
    )
}

/// A path as the file system compares it: without case on Windows.
pub fn path_key(path: &str) -> String {
    if cfg!(windows) {
        path.to_lowercase()
    } else {
        path.to_owned()
    }
}

#[derive(Debug, Default)]
struct State {
    /// Every change, in the order first staged.
    entries: Vec<Entry>,
    /// Paths `read_file` returned in this session (by [`path_key`]).
    read: BTreeSet<String>,
    /// The last revision given to an entry.
    revision: u64,
}

impl State {
    fn next_revision(&mut self) -> u64 {
        self.revision += 1;
        self.revision
    }

    fn live_index(&self, key: &str) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.is_live() && path_key(&entry.change.path) == key)
    }
}

/// One conversation's staged changes, and its staged view.
pub struct Staging {
    sidecar: Arc<Sidecar>,
    state: Mutex<State>,
    /// One staging tool at a time: a response's calls run concurrently, and
    /// each reads the staged view, computes, and records under this.
    serial: Mutex<()>,
}

/// A fresh change id, `ch_<16 hex>`.
fn new_change_id(taken: &[Entry]) -> ChangeId {
    loop {
        let id = format!("ch_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]);
        if !taken.iter().any(|entry| entry.change.id == id) {
            return id;
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Staging {
    /// No staged changes yet.
    pub fn new(sidecar: Arc<Sidecar>) -> Self {
        Self {
            sidecar,
            state: Mutex::new(State::default()),
            serial: Mutex::new(()),
        }
    }

    /// The staged changes a reopened conversation's record holds: for each
    /// change id, its last `Staged` item, with its bytes read back from the
    /// blobs. A change whose blob cannot be read is left out of the staged
    /// view (it is not live) but kept in the list.
    pub fn from_items(sidecar: Arc<Sidecar>, items: &[Item]) -> Self {
        let mut entries: Vec<Entry> = Vec::new();
        for item in items {
            let Item::Staged { change, .. } = item else {
                continue;
            };
            if !is_change_id(&change.id) {
                continue;
            }
            let read = |sha: &str| sidecar.read_blob(sha).ok();
            let new = match &change.new {
                NewState::Bytes { blob } => read(blob),
                NewState::Deleted => None,
            };
            let base = match &change.base {
                BaseState::Present { sha256, .. } => read(sha256),
                BaseState::Absent => None,
            };
            let mut change = (**change).clone();
            if matches!(change.new, NewState::Bytes { .. }) && new.is_none() {
                change.state = ChangeState::Conflict {
                    reason: "Lattice could not read this change's staged bytes back.".to_owned(),
                };
            }
            let base_lines = base.as_deref().map_or(0, line_count);
            let entry = Entry {
                change,
                new,
                base,
                base_lines,
                revision: 0,
            };
            match entries.iter_mut().find(|e| e.change.id == entry.change.id) {
                Some(slot) => *slot = entry,
                None => entries.push(entry),
            }
        }
        Self {
            sidecar,
            state: Mutex::new(State {
                entries,
                read: BTreeSet::new(),
                revision: 0,
            }),
            serial: Mutex::new(()),
        }
    }

    /// The conversation's record.
    pub fn sidecar(&self) -> &Arc<Sidecar> {
        &self.sidecar
    }

    /// Hold while a staging tool reads the view, computes and records.
    pub fn serial(&self) -> MutexGuard<'_, ()> {
        lock(&self.serial)
    }

    /// Whether something holds [`Self::serial`] now (tests only).
    #[cfg(test)]
    pub(crate) fn serial_is_held(&self) -> bool {
        matches!(
            self.serial.try_lock(),
            Err(std::sync::TryLockError::WouldBlock)
        )
    }

    /// Every change, in the order first staged.
    pub fn changes(&self) -> Vec<StagedChange> {
        lock(&self.state)
            .entries
            .iter()
            .map(|entry| entry.change.clone())
            .collect()
    }

    /// The live change at `path`, if there is one.
    pub fn live(&self, path: &str) -> Option<StagedChange> {
        let state = lock(&self.state);
        state
            .live_index(&path_key(path))
            .map(|at| state.entries[at].change.clone())
    }

    /// Changes `Pending`, `Rebased` or in `Conflict` (the command gate).
    pub fn waiting(&self) -> u32 {
        let count = lock(&self.state)
            .entries
            .iter()
            .filter(|entry| is_waiting(&entry.change.state))
            .count();
        u32::try_from(count).unwrap_or(u32::MAX)
    }

    /// A change with its bytes: its base's (when read whole) and its new
    /// ones (when it leaves bytes).
    pub fn snapshot(&self, id: &str) -> Option<Snapshot> {
        lock(&self.state)
            .entries
            .iter()
            .find(|entry| entry.change.id == id)
            .map(|entry| Snapshot {
                change: entry.change.clone(),
                base: entry.base.clone(),
                new: entry.new.clone(),
                base_lines: entry.base_lines,
                revision: entry.revision,
            })
    }

    /// Record a change's next state (a review's: Rebased, Conflict, Kept,
    /// Undone, PartlyKept, a hunk Undo's new bytes), or a change staged by
    /// another path than the staging tools (a restore, row E4): its bytes as
    /// blobs, then a further `staged` item, then the change in memory. The
    /// item carries the origin's turn and call (empty when it has none).
    ///
    /// A snapshot of a change that moved on since it was read (its revision
    /// is not the change's) is refused with [`StageError::Stale`] and nothing
    /// is recorded: a review never overwrites a later staging.
    pub fn update(&self, snapshot: Snapshot) -> Result<(), StageError> {
        let mut state = lock(&self.state);
        if state.entries.iter().any(|entry| {
            entry.change.id == snapshot.change.id && entry.revision != snapshot.revision
        }) {
            return Err(StageError::Stale);
        }
        let blob = |bytes: &[u8]| {
            self.sidecar
                .put_blob(bytes, BlobKind::Staged)
                .map(|_| ())
                .map_err(StageError::Record)
        };
        if let Some(bytes) = &snapshot.base {
            blob(bytes)?;
        }
        if let Some(bytes) = &snapshot.new {
            blob(bytes)?;
        }
        let (turn, call) = match &snapshot.change.origin {
            ChangeOrigin::Agent { turn, call } => (turn.clone(), call.clone()),
            ChangeOrigin::CommandUndo { call } | ChangeOrigin::Command { call } => {
                (String::new(), call.clone())
            }
            ChangeOrigin::Restore { .. } => (String::new(), String::new()),
        };
        self.sidecar
            .append(&Item::Staged {
                turn,
                call_id: call,
                change: Box::new(snapshot.change.clone()),
            })
            .map_err(StageError::Record)?;
        let entry = Entry {
            change: snapshot.change,
            new: snapshot.new,
            base: snapshot.base,
            base_lines: snapshot.base_lines,
            revision: state.next_revision(),
        };
        match state
            .entries
            .iter_mut()
            .find(|other| other.change.id == entry.change.id)
        {
            Some(slot) => *slot = entry,
            None => state.entries.push(entry),
        }
        Ok(())
    }

    /// Whether `read_file` returned `path` in this session.
    pub fn has_read(&self, path: &str) -> bool {
        lock(&self.state).read.contains(&path_key(path))
    }

    /// Record `proposal` (ST1, ST2): blobs first, then the `Staged` item,
    /// then the change is live. On a record failure nothing changes.
    pub fn record(
        &self,
        proposal: Proposal,
        turn: &TurnId,
        call: &CallId,
    ) -> Result<Recorded, StageError> {
        let key = path_key(&proposal.path);
        let mut state = lock(&self.state);
        let live = state.live_index(&key);
        let previous = live.map(|at| state.entries[at].clone());
        let (id, base, base_bytes, base_lines, origin, old_kind, old_ops) = match &previous {
            Some(entry) => (
                entry.change.id.clone(),
                entry.change.base.clone(),
                entry.base.clone(),
                entry.base_lines,
                entry.change.origin.clone(),
                Some(entry.change.kind),
                entry.change.ops.clone(),
            ),
            None => (
                new_change_id(&state.entries),
                proposal.base.state.clone(),
                proposal.base.bytes.clone(),
                proposal.base.lines,
                ChangeOrigin::Agent {
                    turn: turn.clone(),
                    call: call.clone(),
                },
                None,
                Vec::new(),
            ),
        };
        let absent = base == BaseState::Absent;
        let (kind, ops) = match (&proposal.action, old_kind) {
            (Action::Edit(op), None) => (ChangeKind::Edit, vec![op.clone()]),
            (Action::Edit(op), Some(ChangeKind::Edit)) => {
                let mut ops = old_ops;
                ops.push(op.clone());
                (ChangeKind::Edit, ops)
            }
            (Action::Edit(_), Some(kind)) => (kind, Vec::new()),
            (Action::Write, _) if absent => (ChangeKind::Create, Vec::new()),
            (Action::Write, _) => (ChangeKind::Overwrite, Vec::new()),
            (Action::Delete, _) => (ChangeKind::Delete, Vec::new()),
        };
        let blob = |bytes: &[u8]| {
            self.sidecar
                .put_blob(bytes, BlobKind::Staged)
                .map(|blob| blob.sha256)
                .map_err(StageError::Record)
        };
        if let Some(bytes) = &base_bytes {
            blob(bytes)?;
        }
        let new = match &proposal.new {
            Some(bytes) => NewState::Bytes { blob: blob(bytes)? },
            None => NewState::Deleted,
        };
        let change = StagedChange {
            id,
            path: previous
                .as_ref()
                .map_or_else(|| proposal.path.clone(), |entry| entry.change.path.clone()),
            kind,
            base,
            new,
            ops,
            authority: proposal.authority
                || previous
                    .as_ref()
                    .is_some_and(|entry| entry.change.authority),
            origin,
            state: ChangeState::Pending,
        };
        self.sidecar
            .append(&Item::Staged {
                turn: turn.clone(),
                call_id: call.clone(),
                change: Box::new(change.clone()),
            })
            .map_err(StageError::Record)?;
        let entry = Entry {
            change: change.clone(),
            new: proposal.new.clone(),
            base: base_bytes,
            base_lines,
            revision: state.next_revision(),
        };
        match live {
            Some(at) => state.entries[at] = entry,
            None => state.entries.push(entry),
        }
        let (added, removed) = counts(state.entries.iter().find(|e| e.change.id == change.id));
        Ok(Recorded {
            change,
            added,
            removed,
        })
    }
}

/// `+a −b` of a change against its base.
fn counts(entry: Option<&Entry>) -> (u32, u32) {
    let Some(entry) = entry else {
        return (0, 0);
    };
    let text = |bytes: &Option<Vec<u8>>| {
        bytes
            .as_deref()
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_default()
    };
    match &entry.change.new {
        NewState::Deleted => (0, entry.base_lines),
        NewState::Bytes { .. } => added_removed(&text(&entry.base), &text(&entry.new)),
    }
}

impl Overlay for Staging {
    fn staged(&self, path: &str) -> Option<Staged> {
        let state = lock(&self.state);
        let at = state.live_index(&path_key(path))?;
        let entry = &state.entries[at];
        match (&entry.change.new, &entry.new) {
            (NewState::Deleted, _) => Some(Staged::Deleted),
            (NewState::Bytes { .. }, Some(bytes)) => Some(Staged::Bytes(bytes.clone())),
            (NewState::Bytes { .. }, None) => None,
        }
    }

    fn created(&self) -> Vec<String> {
        let state = lock(&self.state);
        let mut paths: BTreeMap<String, String> = BTreeMap::new();
        for entry in state.entries.iter().filter(|entry| entry.is_live()) {
            if entry.change.base == BaseState::Absent
                && matches!(entry.change.new, NewState::Bytes { .. })
            {
                paths.insert(path_key(&entry.change.path), entry.change.path.clone());
            }
        }
        paths.into_values().collect()
    }

    fn note_read(&self, path: &str) {
        lock(&self.state).read.insert(path_key(path));
    }
}

pub mod review;

#[cfg(test)]
mod review_tests;
#[cfg(test)]
mod tests;
