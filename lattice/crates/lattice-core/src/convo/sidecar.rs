//! The native sidecar: what only the native Lattice records about a
//! conversation (the chat core's spec §5.5, §5.7, §5.8, T15). Not a port;
//! its torn-tail healing follows Python's `history.py`.
//!
//! Under `<globals>/lattice_native/chat/conversations/<id>/`:
//! - `meta.json`: `{"v":1, "id", "created", "workspace": {"id","path"} or
//!   null, "mode", "origin"}`, replaced atomically;
//! - `items.jsonl`: append-only, LF line ends; its first line is
//!   `{"v":1,"id":"<id>","created":<f64>}` and each further line one
//!   [`Item`]. A torn last line (a crash mid-write) is healed with a bare LF
//!   before the next append, and skipped when read;
//! - `blobs/<sha256>`: content-addressed, write-once;
//! - `.lock`: held (an exclusive `File::try_lock`) while this process has the
//!   conversation open for writing, released explicitly when it closes.
//!
//! Invariants:
//! - **Binding.** A sidecar belongs to the shared thread with the same id
//!   *and* the same `created`. Opened for a row whose `created` differs, it is
//!   an earlier conversation's (a late save can re-create an id): its folder is
//!   moved aside to `<id>~<created's f64 bits as 16 hex>` and a fresh one
//!   starts. Nothing is removed (ND1).
//! - **Redaction (T15).** Every item but a staged change passes through
//!   `secrets::redact_strings` before it is written, and text blobs through
//!   `secrets::redact`; [`Appended::redacted`] says when something was
//!   replaced, so the caller can record a notice. A staged change is written as
//!   given: its edits and its blob are the bytes a Keep writes.
//! - **Bounds.** A line is at most 1 MiB (larger payloads go to blobs, a
//!   payload over 16 KiB always does); a reader skips a line over 4 MiB, as the
//!   run store does, and a line it cannot read.
//! - **Order.** Appends to one conversation are serialised by an in-process
//!   mutex and, across processes, by the `.lock`.
//! - Ids are checked before they become paths.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lattice_protocol::conversation::{Mode, Origin, WorkspaceId, is_conversation_id};
use serde::{Deserialize, Serialize};

use super::item::{INLINE_LIMIT, Item, Payload};
use crate::fsx::{atomic_write, move_dir_aside, write_once};
use crate::secrets;
use crate::sha::{is_sha256_hex, sha256_hex};

/// The most bytes one written line may have.
pub const MAX_LINE_BYTES: usize = 1024 * 1024;
/// Lines longer than this are skipped when read.
pub const MAX_READ_LINE_BYTES: u64 = 4 * 1024 * 1024;
/// How much of a payload kept in a blob is previewed inline.
pub const PREVIEW_BYTES: usize = 2 * 1024;

const META: &str = "meta.json";
const ITEMS: &str = "items.jsonl";
const BLOBS: &str = "blobs";
const LOCK: &str = ".lock";
const CONVERSATIONS: &str = "conversations";

/// What went wrong with a sidecar.
#[derive(Debug)]
pub enum SidecarError {
    /// An id or a name that cannot become a path.
    Invalid(&'static str),
    /// Another handle (another window, another process) has it open for writing.
    Busy,
    /// A line over [`MAX_LINE_BYTES`]; nothing was written.
    TooLong,
    /// `items.jsonl` does not start with this conversation's header.
    NotThisConversation,
    Io(io::Error),
}

impl std::fmt::Display for SidecarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(what) => write!(f, "not a valid {what}"),
            Self::Busy => f.write_str("another window is writing this conversation"),
            Self::TooLong => f.write_str("a record over 1 MiB"),
            Self::NotThisConversation => f.write_str("the record belongs to another conversation"),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for SidecarError {}

impl From<io::Error> for SidecarError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// The folder a conversation works in, as the record keeps it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceRef {
    pub id: WorkspaceId,
    pub path: String,
}

/// `meta.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub v: u32,
    pub id: String,
    pub created: f64,
    pub workspace: Option<WorkspaceRef>,
    pub mode: Mode,
    pub origin: Origin,
}

/// The first line of `items.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Header {
    pub v: u32,
    pub id: String,
    pub created: f64,
}

/// What a new sidecar starts with.
#[derive(Clone, Debug, PartialEq)]
pub struct NewMeta {
    pub workspace: Option<WorkspaceRef>,
    pub mode: Mode,
    pub origin: Origin,
}

/// How [`SidecarStore::open_for_writing`] found the conversation's folder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    /// There was none; a fresh one was made.
    Created,
    /// It belongs to this thread (same `created`).
    Bound,
    /// It belonged to an earlier conversation with this id, and was moved to
    /// this folder name; a fresh one was made.
    MovedAside(String),
}

/// What an append wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Appended {
    /// A secret-looking part of the record was replaced before writing.
    pub redacted: bool,
}

/// A conversation's records, as read.
#[derive(Clone, Debug, PartialEq)]
pub struct ItemLog {
    pub header: Header,
    pub items: Vec<Item>,
    /// Lines that were skipped: torn, too long, or of a kind this build does
    /// not know.
    pub skipped: usize,
}

/// Which bytes a blob holds, and so whether they are redacted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlobKind {
    /// Tool or command output: redacted before it is written (T15), read as
    /// UTF-8 and as UTF-16 in both byte orders at both alignments (T15a).
    Output,
    /// A staged file's new bytes: written as given (what a Keep writes).
    Staged,
}

/// A blob as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobRef {
    pub sha256: String,
    pub bytes: u64,
    pub redacted: bool,
}

/// `<globals>/lattice_native/chat/`: every conversation's sidecar.
#[derive(Clone, Debug)]
pub struct SidecarStore {
    root: PathBuf,
}

/// Whether two `created` times name the same conversation: the same bits
/// (§5.5, rows B5 and G4). Both sides read back exactly what was written:
/// `meta.json` and the item log's header are written by `serde_json` as the
/// shortest round-trip text and read by `serde_json` with its
/// `float_roundtrip` feature (row E11d; without it the default parser read
/// `0x41dab04a30eaee9f` back as `0x41dab04a30eaeea0`, and E11c needed a
/// tolerance), and the shared row's `created`, written by Python as
/// `repr(float)`, is read by `chat::pyjson` with Rust's correctly rounded
/// `str::parse`.
pub fn same_created(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

/// The `f64` as its 16 hex digits: a name that keeps the exact value.
pub fn created_tag(created: f64) -> String {
    format!("{:016x}", created.to_bits())
}

fn read_meta_file(dir: &Path) -> Option<Meta> {
    let bytes = fs::read(dir.join(META)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn read_header(dir: &Path) -> Option<Header> {
    let file = File::open(dir.join(ITEMS)).ok()?;
    let mut first = String::new();
    BufReader::new(file)
        .take(MAX_READ_LINE_BYTES)
        .read_line(&mut first)
        .ok()?;
    serde_json::from_str(first.trim_end()).ok()
}

/// Try the folder's `.lock`. `Ok(None)`: someone else holds it.
fn try_lock(dir: &Path) -> io::Result<Option<File>> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(dir.join(LOCK))?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(fs::TryLockError::WouldBlock) => Ok(None),
        Err(fs::TryLockError::Error(error)) => Err(error),
    }
}

fn release(lock: File) {
    // Explicitly, so the release does not wait for the handle to close.
    let _ = lock.unlock();
}

impl SidecarStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `conversations/<id>`, for a valid id only.
    pub fn conversation_dir(&self, id: &str) -> Result<PathBuf, SidecarError> {
        if !is_conversation_id(id) {
            return Err(SidecarError::Invalid("conversation id"));
        }
        Ok(self.root.join(CONVERSATIONS).join(id))
    }

    /// The sidecar's `meta.json`, when there is a readable one.
    pub fn read_meta(&self, id: &str) -> Result<Option<Meta>, SidecarError> {
        Ok(read_meta_file(&self.conversation_dir(id)?))
    }

    /// Open the conversation's sidecar for writing, bound to the shared row
    /// whose `created` is given (§5.5), and hold its `.lock` while the returned
    /// [`Sidecar`] lives. `Busy` when another handle holds it.
    pub fn open_for_writing(
        &self,
        id: &str,
        created: f64,
        new: NewMeta,
    ) -> Result<(Sidecar, Binding), SidecarError> {
        let dir = self.conversation_dir(id)?;
        let mut binding = Binding::Created;
        if dir.exists() {
            let recorded = read_meta_file(&dir)
                .map(|meta| meta.created)
                .or_else(|| read_header(&dir).map(|header| header.created));
            match recorded {
                Some(seen) if same_created(seen, created) => binding = Binding::Bound,
                _ if Self::holds_nothing(&dir)? => {}
                seen => {
                    let tag = seen.map_or_else(|| "unbound".to_owned(), created_tag);
                    binding = Binding::MovedAside(self.move_aside(id, &dir, &tag)?);
                }
            }
        }
        fs::create_dir_all(dir.join(BLOBS))?;
        let lock = try_lock(&dir)?.ok_or(SidecarError::Busy)?;
        let sidecar = Sidecar {
            dir,
            id: id.to_owned(),
            created,
            write: Mutex::new(()),
            lock: Some(lock),
        };
        if binding != Binding::Bound {
            sidecar.start(new)?;
        } else if read_meta_file(&sidecar.dir).is_none() {
            // Bound by the header alone (a crash before meta.json): write it.
            sidecar.write_meta(&Meta {
                v: 1,
                id: id.to_owned(),
                created,
                workspace: new.workspace,
                mode: new.mode,
                origin: new.origin,
            })?;
        }
        Ok((sidecar, binding))
    }

    /// True when the folder holds no record at all (only `.lock` or an empty
    /// `blobs/`), so it can be used as it is.
    fn holds_nothing(dir: &Path) -> io::Result<bool> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if name == LOCK {
                continue;
            }
            if name == BLOBS && fs::read_dir(entry.path())?.next().is_none() {
                continue;
            }
            return Ok(false);
        }
        Ok(true)
    }

    /// Move an earlier conversation's folder to `<id>~<tag>` (then `-2`, `-3`,
    /// …), first making sure no one is writing it.
    fn move_aside(&self, id: &str, dir: &Path, tag: &str) -> Result<String, SidecarError> {
        let lock = try_lock(dir)?.ok_or(SidecarError::Busy)?;
        release(lock);
        let parent = self.root.join(CONVERSATIONS);
        for attempt in 1..=100u32 {
            let name = if attempt == 1 {
                format!("{id}~{tag}")
            } else {
                format!("{id}~{tag}-{attempt}")
            };
            match move_dir_aside(dir, &parent.join(&name)) {
                Ok(()) => return Ok(name),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(SidecarError::Io(io::Error::other(
            "no free name for an earlier conversation's record",
        )))
    }

    /// Every record of a conversation, read in full (on open). Does not need
    /// the lock: lines are appended whole.
    pub fn read_items(&self, id: &str) -> Result<ItemLog, SidecarError> {
        let dir = self.conversation_dir(id)?;
        let file = File::open(dir.join(ITEMS))?;
        let mut reader = BufReader::new(file);
        let mut header = None;
        let mut items = Vec::new();
        let mut skipped = 0;
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = (&mut reader)
                .take(MAX_READ_LINE_BYTES + 1)
                .read_until(b'\n', &mut line)?;
            if read == 0 {
                break;
            }
            if read as u64 > MAX_READ_LINE_BYTES && line.last() != Some(&b'\n') {
                // Over the bound: skip to the end of this line.
                skipped += 1;
                let mut rest = Vec::new();
                reader.read_until(b'\n', &mut rest)?;
                continue;
            }
            let text = line.strip_suffix(b"\n").unwrap_or(&line);
            if header.is_none() {
                let parsed: Header =
                    serde_json::from_slice(text).map_err(|_| SidecarError::NotThisConversation)?;
                if parsed.id != id {
                    return Err(SidecarError::NotThisConversation);
                }
                header = Some(parsed);
                continue;
            }
            match serde_json::from_slice::<Item>(text) {
                Ok(item) => items.push(item),
                Err(_) => skipped += 1,
            }
        }
        let header = header.ok_or(SidecarError::NotThisConversation)?;
        Ok(ItemLog {
            header,
            items,
            skipped,
        })
    }

    /// A blob's bytes, by its hash.
    pub fn read_blob(&self, id: &str, sha256: &str) -> Result<Vec<u8>, SidecarError> {
        if !is_sha256_hex(sha256) {
            return Err(SidecarError::Invalid("blob name"));
        }
        Ok(fs::read(
            self.conversation_dir(id)?.join(BLOBS).join(sha256),
        )?)
    }
}

/// One conversation's sidecar, open for writing. Its `.lock` is held until it
/// drops.
#[derive(Debug)]
pub struct Sidecar {
    dir: PathBuf,
    id: String,
    created: f64,
    write: Mutex<()>,
    lock: Option<File>,
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        if let Some(lock) = self.lock.take() {
            release(lock);
        }
    }
}

impl Sidecar {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn created(&self) -> f64 {
        self.created
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn start(&self, new: NewMeta) -> Result<(), SidecarError> {
        self.write_meta(&Meta {
            v: 1,
            id: self.id.clone(),
            created: self.created,
            workspace: new.workspace,
            mode: new.mode,
            origin: new.origin,
        })?;
        let header = Header {
            v: 1,
            id: self.id.clone(),
            created: self.created,
        };
        let mut line = serde_json::to_vec(&header).map_err(io::Error::other)?;
        line.push(b'\n');
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.dir.join(ITEMS))?;
        file.write_all(&line)?;
        file.sync_all()?;
        Ok(())
    }

    /// Replace `meta.json` (a mode switch, a workspace attached).
    pub fn write_meta(&self, meta: &Meta) -> Result<(), SidecarError> {
        if meta.id != self.id || meta.created.to_bits() != self.created.to_bits() {
            return Err(SidecarError::NotThisConversation);
        }
        let bytes = serde_json::to_vec(meta).map_err(io::Error::other)?;
        atomic_write(&self.dir.join(META), &bytes)?;
        Ok(())
    }

    /// Append one record. Every string in it passes the secret redaction first,
    /// unless it is a staged change (see the module header).
    pub fn append(&self, item: &Item) -> Result<Appended, SidecarError> {
        let mut value = serde_json::to_value(item).map_err(io::Error::other)?;
        let redacted = !item.is_written_verbatim() && secrets::redact_strings(&mut value);
        let mut line = serde_json::to_vec(&value).map_err(io::Error::other)?;
        if line.len() + 1 > MAX_LINE_BYTES {
            return Err(SidecarError::TooLong);
        }
        line.push(b'\n');
        let _order = self
            .write
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(self.dir.join(ITEMS))?;
        heal_tail(&mut file)?;
        file.write_all(&line)?;
        Ok(Appended { redacted })
    }

    /// A blob's bytes, by its hash.
    pub fn read_blob(&self, sha256: &str) -> Result<Vec<u8>, SidecarError> {
        if !is_sha256_hex(sha256) {
            return Err(SidecarError::Invalid("blob name"));
        }
        Ok(fs::read(self.dir.join(BLOBS).join(sha256))?)
    }

    /// Write a blob (once: an existing one is the same bytes by its name).
    pub fn put_blob(&self, bytes: &[u8], kind: BlobKind) -> Result<BlobRef, SidecarError> {
        let (stored, redacted) = match kind {
            BlobKind::Staged => (bytes.to_vec(), false),
            BlobKind::Output => redact_output(bytes),
        };
        let sha256 = sha256_hex(&stored);
        write_once(&self.dir.join(BLOBS).join(&sha256), &stored)?;
        Ok(BlobRef {
            sha256,
            bytes: stored.len() as u64,
            redacted,
        })
    }

    /// `text` as a record's payload: redacted, then inline when it is at most
    /// 16 KiB, otherwise a blob with a preview of its first 2 KiB.
    pub fn payload(&self, text: &str) -> Result<(Payload, bool), SidecarError> {
        let clean = secrets::redact(text);
        let redacted = clean != text;
        if clean.len() <= INLINE_LIMIT {
            return Ok((Payload::Inline(clean), redacted));
        }
        // Already redacted above, so it is stored as it is.
        let blob = self.put_blob(clean.as_bytes(), BlobKind::Staged)?;
        let mut cut = PREVIEW_BYTES.min(clean.len());
        while !clean.is_char_boundary(cut) {
            cut -= 1;
        }
        Ok((
            Payload::Blob {
                sha256: blob.sha256,
                bytes: blob.bytes,
                preview: clean[..cut].to_owned(),
            },
            redacted,
        ))
    }
}

/// T15 over command or tool output, whatever its encoding (spec 22.6 T15a):
/// the stored bytes and whether redaction changed them. The intent: bytes
/// that redaction would have changed in any reading are never stored as they
/// came. So the bytes are read five ways, with no guess at which is right:
/// UTF-16LE and UTF-16BE each from the first and from the second byte (a
/// byte-order mark is just a character; an odd last byte is left out), and
/// UTF-8. Whatever a secret pattern matches in a reading is replaced in place,
/// in that reading's encoding, by [`secrets::REDACTED`]; the bytes around it
/// are kept exactly. The readings are taken again until none changes
/// anything (a replacement shifts the other readings); if that has not
/// happened after [`MAX_PASSES`], the whole output is withheld.
fn redact_output(bytes: &[u8]) -> (Vec<u8>, bool) {
    let mut current = bytes.to_vec();
    for _ in 0..MAX_PASSES {
        let mut changed = false;
        for reading in READINGS {
            if let Some(next) = redact_reading(&current, reading) {
                current = next;
                changed = true;
            }
        }
        if !changed {
            let redacted = current != bytes;
            return (current, redacted);
        }
    }
    (secrets::REDACTED.as_bytes().to_vec(), true)
}

/// How many times [`redact_output`] takes its readings before it withholds.
const MAX_PASSES: usize = 8;

/// One way to read output bytes as text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reading {
    Utf8,
    /// UTF-16 from byte `offset` (0 or 1), big-endian or little-endian.
    Utf16 {
        big: bool,
        offset: usize,
    },
}

/// Every reading [`redact_output`] takes.
const READINGS: [Reading; 5] = [
    Reading::Utf16 {
        big: false,
        offset: 0,
    },
    Reading::Utf16 {
        big: false,
        offset: 1,
    },
    Reading::Utf16 {
        big: true,
        offset: 0,
    },
    Reading::Utf16 {
        big: true,
        offset: 1,
    },
    Reading::Utf8,
];

impl Reading {
    /// The characters this reading sees, each with the byte range it came
    /// from. A lone surrogate or an invalid UTF-8 sequence reads as U+FFFD.
    fn chars(self, bytes: &[u8]) -> Vec<(char, usize, usize)> {
        let mut out = Vec::new();
        match self {
            Self::Utf8 => {
                let mut at = 0;
                for chunk in bytes.utf8_chunks() {
                    for (index, c) in chunk.valid().char_indices() {
                        out.push((c, at + index, at + index + c.len_utf8()));
                    }
                    at += chunk.valid().len();
                    if !chunk.invalid().is_empty() {
                        let length = chunk.invalid().len();
                        out.push((char::REPLACEMENT_CHARACTER, at, at + length));
                        at += length;
                    }
                }
            }
            Self::Utf16 { big, offset } => {
                let unit = |at: usize| -> Option<u16> {
                    let pair = [*bytes.get(at)?, *bytes.get(at + 1)?];
                    Some(if big {
                        u16::from_be_bytes(pair)
                    } else {
                        u16::from_le_bytes(pair)
                    })
                };
                let mut at = offset;
                while let Some(first) = unit(at) {
                    let pair = unit(at + 2).filter(|second| {
                        (0xD800..0xDC00).contains(&first) && (0xDC00..0xE000).contains(second)
                    });
                    match pair {
                        Some(second) => {
                            let c = char::decode_utf16([first, second])
                                .next()
                                .and_then(Result::ok)
                                .unwrap_or(char::REPLACEMENT_CHARACTER);
                            out.push((c, at, at + 4));
                            at += 4;
                        }
                        None => {
                            let c = char::from_u32(u32::from(first))
                                .unwrap_or(char::REPLACEMENT_CHARACTER);
                            out.push((c, at, at + 2));
                            at += 2;
                        }
                    }
                }
            }
        }
        out
    }

    /// `text` in this reading's encoding.
    fn encode(self, text: &str) -> Vec<u8> {
        match self {
            Self::Utf8 => text.as_bytes().to_vec(),
            Self::Utf16 { big, .. } => text
                .encode_utf16()
                .flat_map(|unit| {
                    if big {
                        unit.to_be_bytes()
                    } else {
                        unit.to_le_bytes()
                    }
                })
                .collect(),
        }
    }
}

/// `bytes` with every secret this reading sees replaced in place by
/// [`secrets::REDACTED`] in its encoding, or `None` when it sees none.
fn redact_reading(bytes: &[u8], reading: Reading) -> Option<Vec<u8>> {
    let decoded = reading.chars(bytes);
    let chars: Vec<char> = decoded.iter().map(|(c, _, _)| *c).collect();
    let spans = secrets::secret_spans(&chars);
    if spans.is_empty() {
        return None;
    }
    let marker = reading.encode(secrets::REDACTED);
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    for (start, end) in spans {
        let (from, to) = (decoded[start].1, decoded[end - 1].2);
        out.extend_from_slice(&bytes[at..from]);
        out.extend_from_slice(&marker);
        at = to;
    }
    out.extend_from_slice(&bytes[at..]);
    Some(out)
}

/// Before an append: when the file ends without a newline (a torn last line),
/// write a bare LF so the torn line stays its own (Python's heal).
fn heal_tail(file: &mut File) -> io::Result<()> {
    let length = file.metadata()?.len();
    if length == 0 {
        return Ok(());
    }
    file.seek(SeekFrom::Start(length - 1))?;
    let mut last = [0u8; 1];
    file.read_exact(&mut last)?;
    if last[0] != b'\n' {
        file.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use lattice_protocol::Locality;
    use lattice_protocol::conversation::*;
    use serde_json::Value;

    use super::super::item::{BaseState, EditOp, NewState, StagedChange};
    use super::*;
    use crate::fsx::MoveReason;
    use crate::testkit::TempDir;

    /// Whether the redaction would change any string of `value`.
    fn has_secret(value: &Value) -> bool {
        let mut copy = value.clone();
        secrets::redact_strings(&mut copy)
    }

    // Assembled from pieces so no source file holds a credential-shaped string.
    fn key() -> String {
        format!("{}{}", concat!("s", "k-"), "abcdefghijklmnop0123456789")
    }

    fn new_meta() -> NewMeta {
        NewMeta {
            workspace: Some(WorkspaceRef {
                id: "0f1e2d3c4b5a6978".into(),
                path: "C:\\work".into(),
            }),
            mode: Mode::Agent,
            origin: Origin::Native,
        }
    }

    fn shown() -> lattice_protocol::Shown {
        lattice_protocol::Shown {
            locality: Locality::Local,
            label: "Local".into(),
        }
    }

    /// Every file under `root` with the SHA-256 of its bytes (folders as "").
    fn snapshot(root: &Path) -> BTreeMap<String, String> {
        fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
            for entry in fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                let name = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                if path.is_dir() {
                    out.insert(name, String::new());
                    walk(root, &path, out);
                } else if path.file_name().is_some_and(|file| file == LOCK) {
                    // Held locks refuse reads (os error 33); the lock file holds
                    // nothing anyway.
                    out.insert(name, "lock".to_owned());
                } else {
                    out.insert(name, sha256_hex(&fs::read(&path).unwrap()));
                }
            }
        }
        let mut out = BTreeMap::new();
        if root.exists() {
            walk(root, root, &mut out);
        }
        out
    }

    /// NF1: every file's bytes before are still somewhere after; only named
    /// temporaries may disappear.
    fn nothing_lost(before: &BTreeMap<String, String>, after: &BTreeMap<String, String>) {
        let kept: std::collections::BTreeSet<&String> =
            after.values().filter(|hash| !hash.is_empty()).collect();
        for (name, hash) in before {
            if hash.is_empty() || name.ends_with(".tmp") {
                continue;
            }
            assert!(kept.contains(hash), "{name}'s bytes are gone");
        }
    }

    fn every_kind() -> Vec<Item> {
        let turn = || "a1b2c3d4e5f6".to_owned();
        let call = || "call_1".to_owned();
        vec![
            Item::TurnStart {
                turn: turn(),
                mode: Mode::Agent,
                kind: TurnKind::Agent,
                choice: "local".into(),
                shown: shown(),
                resolved: Locality::Local,
                label: "Local".into(),
                at: 1.0,
            },
            Item::ToolCall {
                turn: turn(),
                call_id: call(),
                tool: "read_file".into(),
                arguments: Payload::Inline("{\"path\":\"a.txt\"}".into()),
                summary: "Read a.txt".into(),
                at: 1.25,
            },
            Item::ToolResult {
                turn: turn(),
                call_id: call(),
                output: Payload::Blob {
                    sha256: sha256_hex(b"x"),
                    bytes: 1,
                    preview: "x".into(),
                },
                withheld: false,
                truncated: true,
                at: 1.5,
                output_blob: None,
            },
            Item::ApprovalRequested {
                turn: turn(),
                call_id: "call_2".into(),
                kind: ApprovalKind::Command,
                detail: ApprovalDetail::Command {
                    text: "cargo test".into(),
                    cwd: "".into(),
                    mode: CommandMode::PowerShell,
                    timeout_s: 600,
                    remote: None,
                    staged_waiting: 0,
                    background: false,
                },
                at: 2.0,
            },
            Item::ApprovalDecided {
                turn: turn(),
                call_id: "call_2".into(),
                decision: Decision::Reject {
                    note: Some("not now".into()),
                },
                by: DecidedBy::Reader,
                at: 2.5,
            },
            Item::Question {
                turn: turn(),
                call_id: "call_3".into(),
                text: "Which?".into(),
                at: 3.0,
            },
            Item::Answer {
                turn: turn(),
                call_id: "call_3".into(),
                text: "That one.".into(),
                at: 3.5,
            },
            Item::Staged {
                turn: turn(),
                call_id: "call_4".into(),
                change: Box::new(StagedChange {
                    id: "ch_00112233445566aa".into(),
                    path: "src/lib.rs".into(),
                    kind: ChangeKind::Edit,
                    base: BaseState::Present {
                        sha256: sha256_hex(b"base"),
                        bytes: 4,
                        eol: crate::convo::item::Eol::Crlf,
                        bom: false,
                    },
                    new: NewState::Bytes {
                        blob: sha256_hex(b"new"),
                    },
                    ops: vec![EditOp {
                        old_string: "a".into(),
                        new_string: "b".into(),
                        replace_all: false,
                    }],
                    authority: false,
                    origin: ChangeOrigin::Agent {
                        turn: turn(),
                        call: "call_4".into(),
                    },
                    state: ChangeState::Pending,
                }),
            },
            Item::Reviewed {
                change: "ch_00112233445566aa".into(),
                op: ReviewOp::KeepAll,
                result: ReviewResult::Kept {
                    changed_after: false,
                },
                at: 4.0,
                base_copy: Some("ab".repeat(32)),
            },
            Item::Checkpoint {
                id: 1,
                kind: CheckpointKind::Git,
                reason: CheckpointReason::BeforeKeep,
                exposed: false,
                omitted: 1,
                at: 4.25,
                commit: Some("ab".repeat(20)),
                tree: Some("cd".repeat(20)),
                omitted_list: Some("ef".repeat(32)),
                bytes: 0,
            },
            Item::CommandEffect {
                call_id: "call_2".into(),
                before: 1,
                after: 2,
                files: vec![ChangedPath {
                    path: "out.txt".into(),
                    change: PathChange::Added,
                }],
            },
            Item::MovedAside {
                path: "old.txt".into(),
                to: "1/old.txt".into(),
                sha256: sha256_hex(b"old"),
                why: MoveReason::KeptDelete,
                at: 4.5,
            },
            Item::ModeSwitch {
                mode: Mode::Ask,
                at: 5.0,
            },
            Item::ModelSwitch {
                choice: "endpoint:lab".into(),
                label: "Lab".into(),
                locality: Locality::Remote,
                at: 5.25,
            },
            Item::Steered {
                turn: turn(),
                text: "Only tests.".into(),
                at: 5.5,
            },
            Item::Queued {
                queued_id: "q_1".into(),
                text: "Then clippy.".into(),
                shown: shown(),
                at: 6.0,
            },
            Item::Dequeued {
                queued_id: "q_1".into(),
                outcome: DequeueOutcome::Cancelled,
                at: 6.25,
            },
            Item::Superseded {
                turns: vec![turn()],
                at: 6.5,
            },
            Item::TurnEnd {
                turn: turn(),
                answer: Some("f6e5d4c3b2a1".into()),
                status: TurnStatus::Completed,
                run_id: "0123456789abcdef".into(),
                at: 7.0,
            },
            Item::Notice {
                text: "This turn's tool history could not be saved.".into(),
                at: 7.5,
            },
            Item::Reasoning {
                turn: turn(),
                text: Payload::Inline("First read the file.".into()),
                at: 7.75,
            },
        ]
    }

    #[test]
    fn every_kind_of_item_round_trips_through_the_log() {
        let state = TempDir::new("sidecar-round");
        let store = SidecarStore::new(state.path());
        let (sidecar, binding) = store
            .open_for_writing("0123456789ab", 1_790_000_000.5, new_meta())
            .unwrap();
        assert_eq!(binding, Binding::Created);
        let items = every_kind();
        let kinds: std::collections::BTreeSet<String> = items
            .iter()
            .map(|item| {
                serde_json::to_value(item).unwrap()["type"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(kinds.len(), 21, "one sample of each of the 21 kinds");
        for item in &items {
            assert!(!sidecar.append(item).unwrap().redacted);
        }
        drop(sidecar);
        let log = store.read_items("0123456789ab").unwrap();
        assert_eq!(log.items, items);
        assert_eq!(log.skipped, 0);
        assert_eq!(
            log.header,
            Header {
                v: 1,
                id: "0123456789ab".into(),
                created: 1_790_000_000.5
            }
        );
        let dir = store.conversation_dir("0123456789ab").unwrap();
        let bytes = fs::read(dir.join(ITEMS)).unwrap();
        assert!(!bytes.contains(&b'\r'), "a native-only file: LF line ends");
        assert_eq!(bytes.iter().filter(|b| **b == b'\n').count(), 22);
        let meta: Value = serde_json::from_slice(&fs::read(dir.join(META)).unwrap()).unwrap();
        assert_eq!(
            meta,
            serde_json::json!({"v": 1, "id": "0123456789ab", "created": 1_790_000_000.5,
                "workspace": {"id": "0f1e2d3c4b5a6978", "path": "C:\\work"},
                "mode": "agent", "origin": "native"})
        );
        let text = String::from_utf8(fs::read(dir.join(META)).unwrap()).unwrap();
        let order: Vec<usize> = [
            "\"v\"",
            "\"id\"",
            "\"created\"",
            "\"workspace\"",
            "\"mode\"",
            "\"origin\"",
        ]
        .iter()
        .map(|key| text.find(key).unwrap())
        .collect();
        assert!(order.windows(2).all(|pair| pair[0] < pair[1]), "{text}");
    }

    #[test]
    fn a_torn_tail_is_skipped_on_read_and_healed_with_a_bare_lf() {
        let state = TempDir::new("sidecar-torn");
        let store = SidecarStore::new(state.path());
        let notice = |text: &str| Item::Notice {
            text: text.into(),
            at: 1.0,
        };
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        sidecar.append(&notice("before")).unwrap();
        let items = store.conversation_dir("abc").unwrap().join(ITEMS);
        let mut file = OpenOptions::new().append(true).open(&items).unwrap();
        file.write_all(b"{\"type\":\"notice\",\"te").unwrap();
        drop(file);
        let torn = store.read_items("abc").unwrap();
        assert_eq!((torn.items.len(), torn.skipped), (1, 1));
        sidecar.append(&notice("after")).unwrap();
        let bytes = fs::read(&items).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(
            text.contains("{\"type\":\"notice\",\"te\n{"),
            "the torn line ends with a bare LF and the next record follows: {text}"
        );
        let healed = store.read_items("abc").unwrap();
        assert_eq!(healed.items, vec![notice("before"), notice("after")]);
        assert_eq!(healed.skipped, 1);
    }

    #[test]
    fn a_second_writer_is_refused_until_the_first_closes() {
        let state = TempDir::new("sidecar-two");
        let store = SidecarStore::new(state.path());
        let (first, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        assert!(matches!(
            store.open_for_writing("abc", 1.0, new_meta()),
            Err(SidecarError::Busy)
        ));
        // Another conversation is not affected.
        let (_other, _) = store.open_for_writing("def", 1.0, new_meta()).unwrap();
        drop(first);
        let (_again, binding) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        assert_eq!(binding, Binding::Bound);
    }

    #[test]
    fn two_threads_appending_through_one_writer_lose_and_tear_nothing() {
        let state = TempDir::new("sidecar-threads");
        let store = SidecarStore::new(state.path());
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        let sidecar = Arc::new(sidecar);
        let workers: Vec<_> = (0..2)
            .map(|worker| {
                let sidecar = Arc::clone(&sidecar);
                std::thread::spawn(move || {
                    for n in 0..100 {
                        sidecar
                            .append(&Item::Notice {
                                text: format!("{worker}-{n}"),
                                at: f64::from(n),
                            })
                            .unwrap();
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        let log = store.read_items("abc").unwrap();
        assert_eq!((log.items.len(), log.skipped), (200, 0));
        for worker in 0..2 {
            let mine: Vec<String> = log
                .items
                .iter()
                .filter_map(|item| match item {
                    Item::Notice { text, .. } if text.starts_with(&format!("{worker}-")) => {
                        Some(text.clone())
                    }
                    _ => None,
                })
                .collect();
            let expected: Vec<String> = (0..100).map(|n| format!("{worker}-{n}")).collect();
            assert_eq!(mine, expected, "worker {worker}'s order");
        }
    }

    #[test]
    fn an_earlier_conversations_record_is_moved_aside_and_kept_whole() {
        let state = TempDir::new("sidecar-bind");
        let store = SidecarStore::new(state.path());
        let (old, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        old.append(&Item::Notice {
            text: "from the earlier conversation".into(),
            at: 1.0,
        })
        .unwrap();
        old.put_blob(b"an earlier blob", BlobKind::Staged).unwrap();
        drop(old);
        let before = snapshot(state.path());
        // A busy earlier record is not moved.
        let dir = store.conversation_dir("abc").unwrap();
        let held = try_lock(&dir).unwrap().unwrap();
        assert!(matches!(
            store.open_for_writing("abc", 2.0, new_meta()),
            Err(SidecarError::Busy)
        ));
        assert_eq!(snapshot(state.path()), before);
        release(held);

        let (fresh, binding) = store.open_for_writing("abc", 2.0, new_meta()).unwrap();
        let moved = format!("abc~{}", created_tag(1.0));
        assert_eq!(binding, Binding::MovedAside(moved.clone()));
        assert_eq!(created_tag(1.0), "3ff0000000000000");
        drop(fresh);
        let after = snapshot(state.path());
        nothing_lost(&before, &after);
        assert!(after.contains_key(&format!("conversations/{moved}/items.jsonl")));
        let fresh_log = store.read_items("abc").unwrap();
        assert_eq!((fresh_log.header.created, fresh_log.items.len()), (2.0, 0));
        // The same `created` again binds to the fresh record.
        let (_, binding) = store.open_for_writing("abc", 2.0, new_meta()).unwrap();
        assert_eq!(binding, Binding::Bound);
        // Back to the first `created`: the second record moves aside under its own.
        let (_, binding) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        assert_eq!(
            binding,
            Binding::MovedAside(format!("abc~{}", created_tag(2.0)))
        );
    }

    #[test]
    fn records_from_a_turn_are_redacted_and_a_staged_change_is_not() {
        let state = TempDir::new("sidecar-redact");
        let store = SidecarStore::new(state.path());
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        let secret = key();
        let (arguments, redacted) = sidecar
            .payload(&format!("{{\"content\":\"{secret}\"}}"))
            .unwrap();
        assert!(redacted);
        let call = Item::ToolCall {
            turn: "t1".into(),
            call_id: "c1".into(),
            tool: "write_file".into(),
            arguments,
            summary: format!("Write a file with {secret}"),
            at: 1.0,
        };
        assert!(
            sidecar.append(&call).unwrap().redacted,
            "the summary held it"
        );
        let output = Item::ToolResult {
            turn: "t1".into(),
            call_id: "c1".into(),
            output: Payload::Inline(format!("found {secret} in .env")),
            withheld: false,
            truncated: false,
            at: 1.5,
            output_blob: None,
        };
        assert!(sidecar.append(&output).unwrap().redacted);
        let blob = sidecar
            .put_blob(format!("PS> echo {secret}").as_bytes(), BlobKind::Output)
            .unwrap();
        assert!(blob.redacted);
        // The staged change is the exception: its edit is what a Keep writes.
        let staged = every_kind().remove(7);
        let Item::Staged {
            turn,
            call_id,
            mut change,
        } = staged
        else {
            unreachable!()
        };
        change.ops[0].new_string = secret.clone();
        let staged = Item::Staged {
            turn,
            call_id,
            change,
        };
        assert!(!sidecar.append(&staged).unwrap().redacted);
        let staged_blob = sidecar
            .put_blob(secret.as_bytes(), BlobKind::Staged)
            .unwrap();
        drop(sidecar);
        // On disk: the secret appears only in the staged change and its blob.
        let dir = store.conversation_dir("abc").unwrap();
        let log = store.read_items("abc").unwrap();
        for item in &log.items {
            let value = serde_json::to_value(item).unwrap();
            assert_eq!(has_secret(&value), item.is_written_verbatim(), "{value}");
        }
        for entry in fs::read_dir(dir.join(BLOBS)).unwrap() {
            let path = entry.unwrap().path();
            let text = String::from_utf8_lossy(&fs::read(&path).unwrap()).into_owned();
            let is_staged = path.file_name().unwrap().to_string_lossy() == staged_blob.sha256;
            assert_eq!(text.contains(&secret), is_staged, "{}", path.display());
        }
        assert!(
            fs::read_to_string(dir.join(ITEMS))
                .unwrap()
                .contains(secrets::REDACTED)
        );
    }

    fn utf16le(text: &str) -> Vec<u8> {
        text.encode_utf16().flat_map(u16::to_le_bytes).collect()
    }

    /// Whether `haystack` holds `needle` anywhere.
    fn holds(haystack: &[u8], needle: &[u8]) -> bool {
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    /// UTF-16 units of `bytes` from `offset`, as text (an odd last byte left
    /// out).
    fn decode16(bytes: &[u8], big: bool, offset: usize) -> String {
        let units: Vec<u16> = bytes[offset..]
            .chunks_exact(2)
            .map(|pair| {
                if big {
                    u16::from_be_bytes([pair[0], pair[1]])
                } else {
                    u16::from_le_bytes([pair[0], pair[1]])
                }
            })
            .collect();
        String::from_utf16_lossy(&units)
    }

    fn utf16be(text: &str) -> Vec<u8> {
        text.encode_utf16().flat_map(u16::to_be_bytes).collect()
    }

    /// The stored bytes of `bytes` put as output, and whether redacted.
    fn stored_output(sidecar: &Sidecar, store: &SidecarStore, bytes: &[u8]) -> (Vec<u8>, bool) {
        let blob = sidecar.put_blob(bytes, BlobKind::Output).unwrap();
        let stored = fs::read(
            store
                .conversation_dir("abc")
                .unwrap()
                .join(BLOBS)
                .join(&blob.sha256),
        )
        .unwrap();
        (stored, blob.redacted)
    }

    /// Nothing of the key in `stored`, in UTF-8 or in UTF-16 of either order.
    fn no_key_in(stored: &[u8], secret: &str) -> bool {
        [
            secret.as_bytes().to_vec(),
            utf16le(secret),
            utf16be(secret),
            utf16le("abcdefghijklmnop"),
            utf16be("abcdefghijklmnop"),
            b"abcdefghijklmnop".to_vec(),
        ]
        .iter()
        .all(|form| !holds(stored, form))
    }

    /// T15a (spec 22.6): command output holding a key is stored redacted in
    /// UTF-16LE with or without a byte-order mark, and in UTF-16BE with one;
    /// the blob keeps its encoding, and nothing of the key survives in it.
    #[test]
    fn t15a_utf16_output_holding_a_key_is_stored_redacted() {
        let state = TempDir::new("sidecar-utf16");
        let store = SidecarStore::new(state.path());
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        let secret = key();
        let text = format!("PS> Get-Content .env\r\nOPENAI_KEY={secret}\r\ndone\r\n");
        let with_mark = [&[0xFF, 0xFE][..], &utf16le(&text)].concat();
        let big = [&[0xFE, 0xFF][..], &utf16be(&text)].concat();
        let odd = [&with_mark[..], &[0x41][..]].concat();
        // (what, bytes, big-endian, where the text starts, the mark)
        type Case<'a> = (&'a str, Vec<u8>, bool, usize, &'a [u8]);
        let cases: [Case<'_>; 4] = [
            ("UTF-16LE without a mark", utf16le(&text), false, 0, &[]),
            ("UTF-16LE with a mark", with_mark, false, 2, &[0xFF, 0xFE]),
            ("UTF-16BE with a mark", big, true, 2, &[0xFE, 0xFF]),
            (
                "UTF-16LE with an odd last byte",
                odd,
                false,
                2,
                &[0xFF, 0xFE],
            ),
        ];
        for (what, bytes, big, offset, mark) in cases {
            let (stored, redacted) = stored_output(&sidecar, &store, &bytes);
            println!("{what}: redacted {redacted}, {} bytes", stored.len());
            assert!(redacted, "{what}");
            assert!(no_key_in(&stored, &secret), "{what}: the key survives");
            assert_eq!(
                decode16(&stored, big, offset),
                text.replace(&secret, secrets::REDACTED),
                "{what}: the blob keeps its encoding"
            );
            assert!(stored.starts_with(mark), "{what}");
            assert_eq!(
                stored.len() % 2,
                bytes.len() % 2,
                "{what}: an odd last byte is kept"
            );
        }
    }

    /// T15a's residual (the verifier's probe, phaseHA/logs/VERIFY-probe-t15a):
    /// the old heuristic missed BOM-less UTF-16LE with one trailing byte, an
    /// odd-length ASCII prefix before UTF-16LE, and BOM-less UTF-16LE that is
    /// mostly not ASCII (Cyrillic). Also BOM-less UTF-16BE at either
    /// alignment, and a blob holding the key in UTF-8 and in UTF-16 at once.
    /// Each is stored with nothing of the key in any encoding, the bytes
    /// around it kept exactly.
    /// A key right after one stray byte is seen only from the second byte
    /// (from the first, its first letter pairs with the stray byte).
    /// Mutants: only the old readings (UTF-16LE from the first byte, UTF-8);
    /// the second-byte alignments dropped.
    #[test]
    fn t15a_a_key_in_any_utf16_reading_is_stored_redacted() {
        let state = TempDir::new("sidecar-utf16-any");
        let store = SidecarStore::new(state.path());
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        let secret = key();
        let line = format!("OPENAI_KEY={secret}\r\n");
        let cyrillic =
            "Каталог файлов на диске С: Привет мир, это локализованный вывод программы\r\n"
                .repeat(3);
        let cases: Vec<(&str, Vec<u8>)> = vec![
            (
                "BOM-less UTF-16LE + one trailing byte",
                [utf16le(&line), vec![b'\n']].concat(),
            ),
            (
                "odd ASCII prefix then UTF-16LE",
                [b"start\r\n".to_vec(), utf16le(&line)].concat(),
            ),
            (
                "BOM-less UTF-16LE mostly Cyrillic",
                utf16le(&format!("{cyrillic}{line}")),
            ),
            ("BOM-less UTF-16BE", utf16be(&format!("{cyrillic}{line}"))),
            (
                "odd prefix then UTF-16BE",
                [b"x".to_vec(), utf16be(&line)].concat(),
            ),
            (
                "one stray byte, then the key in UTF-16LE",
                [b"x".to_vec(), utf16le(&secret)].concat(),
            ),
            (
                "one stray byte, then the key in UTF-16BE",
                [b"x".to_vec(), utf16be(&secret)].concat(),
            ),
            (
                "the key in UTF-8 and in UTF-16LE at once",
                [line.as_bytes().to_vec(), utf16le(&line)].concat(),
            ),
        ];
        for (what, bytes) in cases {
            let (stored, redacted) = stored_output(&sidecar, &store, &bytes);
            println!("{what}: redacted {redacted}, {} bytes", stored.len());
            assert!(redacted, "{what}");
            assert!(no_key_in(&stored, &secret), "{what}: the key survives");
        }
        // The bytes around a key are kept: the ASCII prefix, and the
        // Cyrillic text before it.
        let prefixed = [b"start\r\n".to_vec(), utf16le(&line)].concat();
        let (stored, _) = stored_output(&sidecar, &store, &prefixed);
        assert!(stored.starts_with(b"start\r\n"));
        assert_eq!(
            decode16(&stored, false, 7),
            line.replace(&secret, secrets::REDACTED)
        );
        let wide = utf16le(&format!("{cyrillic}{line}"));
        let (stored, _) = stored_output(&sidecar, &store, &wide);
        assert_eq!(
            decode16(&stored, false, 0),
            format!("{cyrillic}{line}").replace(&secret, secrets::REDACTED)
        );
    }

    /// T15a's controls: output with no key is stored byte for byte, whatever
    /// its encoding: UTF-16LE (ASCII or Cyrillic, with or without a mark, odd
    /// or even), UTF-16BE, UTF-8 (including invalid sequences), and bytes
    /// that are no text at all.
    #[test]
    fn t15a_output_without_a_key_is_stored_as_it_came() {
        let state = TempDir::new("sidecar-utf16-plain");
        let store = SidecarStore::new(state.path());
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        let cyrillic = "Каталог файлов на диске С: Привет мир\r\n".repeat(4);
        let wide = utf16le("Directory of C:\\work\r\n2 File(s)\r\n");
        let plain = "even-length ASCII text, read as UTF-8!".as_bytes().to_vec();
        let mut noise = Vec::new();
        let mut state_word: u32 = 0x1234_5678;
        for _ in 0..4096 {
            state_word = state_word
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            noise.push((state_word >> 24) as u8);
        }
        for bytes in [
            wide.clone(),
            [wide.clone(), vec![b'\n']].concat(),
            [b"odd\r\n".to_vec(), wide.clone()].concat(),
            utf16le(&cyrillic),
            [&[0xFF, 0xFE][..], &utf16le(&cyrillic)].concat(),
            utf16be(&cyrillic),
            plain,
            cyrillic.as_bytes().to_vec(),
            vec![0xFF, 0xFE, 0x00],
            vec![b'a', 0xC3, 0x28, b'b', 0xFF],
            noise,
        ] {
            let blob = sidecar.put_blob(&bytes, BlobKind::Output).unwrap();
            assert!(!blob.redacted, "{:?}", &bytes[..bytes.len().min(16)]);
            assert_eq!(blob.sha256, sha256_hex(&bytes));
        }
    }

    /// Every reading maps its characters back to the bytes they came from, so
    /// a replacement keeps the bytes around it.
    #[test]
    fn t15a_each_reading_knows_where_each_character_came_from() {
        let bytes = [0x41, 0x00, 0x3D, 0xD8, 0x00, 0xDE, 0x00, 0xD8, 0x42];
        let le = Reading::Utf16 {
            big: false,
            offset: 0,
        }
        .chars(&bytes);
        assert_eq!(
            le,
            [
                ('A', 0, 2),
                ('\u{1F600}', 2, 6),
                (char::REPLACEMENT_CHARACTER, 6, 8)
            ]
        );
        let utf8 = Reading::Utf8.chars(b"a\xC3\xA9\xFFb");
        assert_eq!(
            utf8,
            [
                ('a', 0, 1),
                ('\u{E9}', 1, 3),
                (char::REPLACEMENT_CHARACTER, 3, 4),
                ('b', 4, 5)
            ]
        );
    }

    #[test]
    fn a_long_payload_goes_to_a_blob_with_a_preview() {
        let state = TempDir::new("sidecar-blob");
        let store = SidecarStore::new(state.path());
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        let short = "x".repeat(INLINE_LIMIT);
        assert_eq!(
            sidecar.payload(&short).unwrap().0,
            Payload::Inline(short.clone())
        );
        let long = "é".repeat(INLINE_LIMIT);
        let (payload, _) = sidecar.payload(&long).unwrap();
        let Payload::Blob {
            sha256,
            bytes,
            preview,
        } = payload
        else {
            panic!("a blob")
        };
        assert_eq!(bytes, long.len() as u64);
        assert!(preview.len() <= PREVIEW_BYTES && long.starts_with(&preview));
        assert_eq!(store.read_blob("abc", &sha256).unwrap(), long.as_bytes());
        // Written once: the same bytes again add nothing.
        let before = snapshot(state.path());
        sidecar.payload(&long).unwrap();
        assert_eq!(snapshot(state.path()), before);
        assert!(matches!(
            store.read_blob("abc", "../meta.json"),
            Err(SidecarError::Invalid(_))
        ));
    }

    #[test]
    fn an_over_long_line_is_refused_and_writes_nothing() {
        let state = TempDir::new("sidecar-long");
        let store = SidecarStore::new(state.path());
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        let before = snapshot(state.path());
        let huge = Item::Notice {
            text: "y".repeat(MAX_LINE_BYTES),
            at: 1.0,
        };
        assert!(matches!(sidecar.append(&huge), Err(SidecarError::TooLong)));
        assert_eq!(snapshot(state.path()), before);
    }

    #[test]
    fn a_reader_skips_huge_and_unknown_lines() {
        let state = TempDir::new("sidecar-skip");
        let store = SidecarStore::new(state.path());
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        let notice = Item::Notice {
            text: "kept".into(),
            at: 1.0,
        };
        sidecar.append(&notice).unwrap();
        drop(sidecar);
        let items = store.conversation_dir("abc").unwrap().join(ITEMS);
        let mut file = OpenOptions::new().append(true).open(&items).unwrap();
        let mut huge = b"{\"type\":\"notice\",\"text\":\"".to_vec();
        huge.resize(huge.len() + 5 * 1024 * 1024, b'z');
        huge.extend_from_slice(b"\",\"at\":1.0}\n");
        file.write_all(&huge).unwrap();
        file.write_all(b"{\"type\":\"from_a_later_build\",\"at\":2.0}\n")
            .unwrap();
        drop(file);
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        sidecar.append(&notice).unwrap();
        let log = store.read_items("abc").unwrap();
        assert_eq!(log.items, vec![notice.clone(), notice]);
        assert_eq!(log.skipped, 2);
    }

    #[test]
    fn ids_are_checked_before_they_become_paths() {
        let state = TempDir::new("sidecar-ids");
        let store = SidecarStore::new(state.path());
        for bad in ["", "../x", "a/b", "a.b", "a~b"] {
            assert!(matches!(
                store.open_for_writing(bad, 1.0, new_meta()),
                Err(SidecarError::Invalid(_))
            ));
            assert!(matches!(
                store.read_items(bad),
                Err(SidecarError::Invalid(_))
            ));
        }
        assert!(snapshot(state.path()).is_empty());
        // A log whose header names another conversation is not read as this one.
        let (sidecar, _) = store.open_for_writing("abc", 1.0, new_meta()).unwrap();
        drop(sidecar);
        let dir = store.conversation_dir("abc").unwrap();
        let other = store.conversation_dir("abd").unwrap();
        fs::create_dir_all(&other).unwrap();
        fs::copy(dir.join(ITEMS), other.join(ITEMS)).unwrap();
        assert!(matches!(
            store.read_items("abd"),
            Err(SidecarError::NotThisConversation)
        ));
    }
}
