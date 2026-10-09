//! A conversation's checkpoints (the chat core's spec §8.1–§8.3, §8.6).
//! Not a port; `capture.py` (the web service's snapshot) and `safety.py` (the
//! Agent tab's undo net) are the references, and their caps are the same
//! numbers (`MAX_SNAPSHOT_BYTES`, `MAX_NEW_FILES`).
//!
//! **In a git workspace** (FT6's native read of `.git` is repeated first, and
//! a folder whose `.git` now names the network takes no checkpoint), every
//! git call goes through the one runner (hooks, fsmonitor and protocols
//! overridden, no lazy fetch), and:
//! 1. **List.** `git ls-files -z -s` (tracked, with modes) and `git ls-files
//!    -z --others --exclude-standard` (untracked and not ignored), relative
//!    to the attached folder. A tracked file missing on disk is left out of
//!    the tree, which records its deletion; an ignored file is never taken.
//!    **Omitted, each by name** in one list the `checkpoint` item names (a
//!    blob; `omitted` stays the count): untracked files past the first
//!    [`MAX_NEW_FILES`] in path order, files over [`MAX_SNAPSHOT_BYTES`],
//!    links, nested repositories, and files that could not be read.
//! 2. **Object ids from git**, never hashed here (no SHA-1 dependency): each
//!    file that misses the cache goes to `git hash-object -w --no-filters`,
//!    which writes a missing blob with no clean filter and prints its id in
//!    the repository's own format. **The cache** is per conversation, keyed by
//!    path: the file id, size, last-write time and change time, all read from
//!    one handle, with the object id and the time the entry was recorded. It
//!    is reused only when all four match and both times are older than the
//!    recording (git's racily-clean rule), so a same-size rewrite whose
//!    last-write time was set back still changes the change time and is
//!    hashed again.
//! 3. **Index and tree.** A private index at `<git common dir>/lattice-chat/
//!    index/<conversation>` is emptied (`read-tree --empty`) and filled with
//!    `update-index --add --cacheinfo <mode>,<id>,<path>` (no working file is
//!    read, so no filter runs), then `write-tree`. The reader's own index,
//!    `HEAD`, branches and working files are never touched.
//! 4. **Commit and ref.** `commit-tree --no-gpg-sign` (a fixed identity; git
//!    2.54's `commit-tree` signs only when asked, observed with a
//!    repository's `commit.gpgSign`, and the flag keeps it so in any version,
//!    since signing starts the repository's `gpg.program`) onto the previous
//!    checkpoint's commit, then `update-ref refs/lattice/
//!    chat/<conversation>/<n> <commit> <zero id>`, a compare-and-swap that
//!    refuses a ref that exists.
//! 5. **Unchanged.** A tree equal to the previous one records a `checkpoint`
//!    item that reuses the previous commit; no ref is made.
//!
//! **Without git** (§8.3, the undo net of `safety.py`): before Lattice writes
//! a file, its bytes are copied once per checkpoint into `checkpoints/<n>/
//! files/<path>` in the conversation's record, and `checkpoints/<n>/
//! manifest.json` lists `{path, state}`: present with its SHA-256 and size,
//! `absent` (a file that did not exist is recorded, not skipped), or
//! `skipped` (over [`MAX_SNAPSHOT_BYTES`], or unreadable). A command's
//! checkpoint there is **exposed**: Lattice cannot see what the command
//! changed, and a restore across it says so.
//!
//! **Failure** refuses the action that needed the checkpoint, with a sentence
//! (§8.6): nothing proceeds unprotected. Nothing is pruned (ND1).
//!
//! Deviations, each for a reason:
//! - The spec's one `hash-object --stdin-paths` call is a few calls with the
//!   paths as arguments (at most [`ARG_CHARS`] characters each): the one
//!   spawn primitive gives every child a `NUL` stdin (X5). The index is
//!   filled the same way, with `--cacheinfo` instead of `--index-info`.
//! - The recording time is taken before the files are read, not after, which
//!   only makes the racily-clean rule stricter.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use lattice_protocol::conversation::{
    CheckpointId, CheckpointKind, CheckpointReason, CheckpointView,
};
use lattice_sys::fs::{Access, FileIdentity, file_identity, file_times, is_storage_only_tag};
use serde::{Deserialize, Serialize};

use crate::clock::Clock;
use crate::convo::item::Item;
use crate::convo::sidecar::{BlobKind, Sidecar};
use crate::fsx::atomic_write;
use crate::git::dotgit::{self, LocalRepo, Repo};
use crate::git::runner::{Extra, GitError, GitRunner};
use crate::localfs::is_inside;
use crate::sha::sha256_hex;
use crate::workspace::paths::Want;
use crate::workspace::{Workspace, shown_path};

/// The largest file a checkpoint takes (`safety.MAX_SNAPSHOT_BYTES`, which
/// `capture.py` uses as its `MAX_CAPTURE_BYTES`).
pub const MAX_SNAPSHOT_BYTES: u64 = 16 * 1024 * 1024;
/// The most untracked files one checkpoint takes (`capture.MAX_NEW_FILES`).
pub const MAX_NEW_FILES: usize = 2000;
/// The most characters of paths one git call carries as arguments (the
/// command line's limit is 32,767).
pub const ARG_CHARS: usize = 16 * 1024;
/// The prefix of a checkpoint's ref, separate from the recorder's
/// `refs/lattice/snapshots/`.
pub const REF_PREFIX: &str = "refs/lattice/chat/";

/// Why a path is not in a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OmitWhy {
    /// Over [`MAX_SNAPSHOT_BYTES`].
    TooLarge,
    /// An untracked file past the first [`MAX_NEW_FILES`].
    UntrackedCap,
    /// A symbolic link or junction, or a file reached through one.
    Link,
    /// A nested repository (a submodule, or an untracked repository).
    NestedRepository,
    /// The file could not be opened or read.
    Unreadable,
}

/// One path a checkpoint left out, as its list records it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Omitted {
    pub path: String,
    pub why: OmitWhy,
    /// The size, when it was read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
}

/// A checkpoint as the record holds it.
#[derive(Clone, Debug, PartialEq)]
pub struct Taken {
    pub id: CheckpointId,
    pub kind: CheckpointKind,
    pub reason: CheckpointReason,
    pub exposed: bool,
    /// How many paths were left out.
    pub omitted: u32,
    /// Git: the commit and its tree.
    pub commit: Option<String>,
    pub tree: Option<String>,
    /// The blob listing every omitted path (JSON), when any was.
    pub omitted_list: Option<String>,
    /// Without git: bytes of the copies.
    pub bytes: u64,
    pub at: f64,
}

/// A copy's state in a non-git checkpoint's manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Copy {
    Present { sha256: String, bytes: u64 },
    Absent,
    Skipped { why: OmitWhy },
}

/// One line of a non-git checkpoint's manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub path: String,
    #[serde(flatten)]
    pub copy: Copy,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Manifest {
    v: u32,
    files: Vec<ManifestEntry>,
}

/// Why a checkpoint could not be taken. Nothing was recorded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointError {
    /// Git could not be run or did not answer.
    Git(GitError),
    /// FT6 read `.git` again and the folder is no longer one git may run in.
    NotLocal(String),
    /// The ref exists already: the compare-and-swap refused it.
    RefExists(String),
    /// The record (blobs, items, copies) could not be written.
    Record,
    /// A git answer that could not be read.
    BadAnswer,
}

impl CheckpointError {
    /// One sentence; never git's own output.
    pub fn sentence(&self) -> String {
        match self {
            Self::Git(error) => error.sentence(),
            Self::NotLocal(sentence) => sentence.clone(),
            Self::RefExists(name) => {
                format!("A checkpoint named {name} exists already in this repository.")
            }
            Self::Record => "Lattice could not record the checkpoint.".to_owned(),
            Self::BadAnswer => "git's answer could not be read.".to_owned(),
        }
    }
}

impl From<GitError> for CheckpointError {
    fn from(error: GitError) -> Self {
        Self::Git(error)
    }
}

/// What the cache compares: read from one handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stamp {
    pub(crate) identity: FileIdentity,
    pub(crate) size: u64,
    pub(crate) last_write: i64,
    pub(crate) change: i64,
}

/// A cached object id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheEntry {
    pub(crate) stamp: Stamp,
    pub(crate) object: String,
    /// When it was recorded, in `FILETIME` ticks.
    pub(crate) recorded: i64,
    /// The blob was written to the object store (a checkpoint's listing);
    /// a restore's comparison only hashes, and its entries are not reused by
    /// a checkpoint, whose tree must name blobs that exist.
    pub(crate) written: bool,
}

/// The racily-clean rule: an entry is reused only when the file's four
/// facts are unchanged and both its times are older than the recording.
pub(crate) fn reusable(entry: &CacheEntry, now: &Stamp) -> bool {
    entry.stamp == *now && now.last_write < entry.recorded && now.change < entry.recorded
}

/// Now, in `FILETIME` ticks (100 ns since 1601).
fn filetime_now() -> i64 {
    let since = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let ticks = since.as_nanos() / 100 + 116_444_736_000_000_000;
    i64::try_from(ticks).unwrap_or(i64::MAX)
}

#[derive(Default)]
struct State {
    taken: Vec<Taken>,
    cache: BTreeMap<String, CacheEntry>,
}

/// One conversation's checkpoints.
pub struct Checkpoints {
    sidecar: Arc<Sidecar>,
    clock: Clock,
    /// Held for a whole checkpoint: one at a time per conversation.
    state: Mutex<State>,
    /// A test's hold on the next checkpoints (a slow checkpoint).
    #[cfg(test)]
    hold: Mutex<Option<Arc<Hold>>>,
}

/// A test's hold on a conversation's checkpoints: each `take` says it began,
/// then waits for one release (or for the releaser to be dropped).
#[cfg(test)]
pub(crate) struct Hold {
    began: std::sync::mpsc::Sender<()>,
    release: Mutex<std::sync::mpsc::Receiver<()>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A candidate's facts, read from one handle opened without following a link.
enum Looked {
    File(Stamp),
    Gone,
    Folder,
    Omit(OmitWhy, Option<u64>),
}

fn look(root: &Path, path: &str) -> Looked {
    let full = path
        .split('/')
        .fold(root.to_path_buf(), |at, part| at.join(part));
    let opened = match lattice_sys::fs::open_no_follow(&full, Access::Attributes) {
        Ok(opened) => opened,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Looked::Gone,
        Err(_) => return Looked::Omit(OmitWhy::Unreadable, None),
    };
    if let Some(link) = &opened.link
        && !is_storage_only_tag(link.tag)
    {
        return Looked::Omit(OmitWhy::Link, None);
    }
    if opened.is_dir {
        return Looked::Folder;
    }
    // A file reached through a link in an earlier component (a junction).
    match lattice_sys::fs::final_path(&opened.file) {
        Ok(real) if is_inside(&real, root) => {}
        _ => return Looked::Omit(OmitWhy::Link, None),
    }
    let (Ok(identity), Ok(times), Ok(meta)) = (
        file_identity(&opened.file),
        file_times(&opened.file),
        opened.file.metadata(),
    ) else {
        return Looked::Omit(OmitWhy::Unreadable, None);
    };
    let size = meta.len();
    if size > MAX_SNAPSHOT_BYTES {
        return Looked::Omit(OmitWhy::TooLarge, Some(size));
    }
    Looked::File(Stamp {
        identity,
        size,
        last_write: times.last_write,
        change: times.change,
    })
}

/// A path's tracked entry: `<mode> <object> <stage>\t<path>`.
fn parse_staged(record: &[u8]) -> Option<(String, String)> {
    let text = std::str::from_utf8(record).ok()?;
    let (head, path) = text.split_once('\t')?;
    let mode = head.split(' ').next()?;
    if path.is_empty() {
        return None;
    }
    Some((mode.to_owned(), path.to_owned()))
}

/// The NUL-separated, UTF-8 records of a listing (others are left out: a
/// name that is not UTF-8 cannot be named in the record).
fn records(stdout: &[u8]) -> impl Iterator<Item = &[u8]> {
    stdout
        .split(|byte| *byte == 0)
        .filter(|raw| !raw.is_empty())
}

fn failed(output: crate::git::runner::Output) -> GitError {
    GitError::Failed {
        status: output.status,
        stderr: output.stderr,
    }
}

/// Run `args` followed by `items` in batches of at most [`ARG_CHARS`]
/// characters, and give each batch's output with its items.
fn batched<'i>(
    items: &'i [String],
    mut run: impl FnMut(&[&'i str]) -> Result<(), CheckpointError>,
) -> Result<(), CheckpointError> {
    let mut batch: Vec<&str> = Vec::new();
    let mut chars = 0usize;
    for item in items {
        if !batch.is_empty() && chars + item.len() + 3 > ARG_CHARS {
            run(&batch)?;
            batch.clear();
            chars = 0;
        }
        chars += item.len() + 3;
        batch.push(item);
    }
    if !batch.is_empty() {
        run(&batch)?;
    }
    Ok(())
}

/// The folder's current non-ignored files with their object ids, as a
/// checkpoint takes them, and what it left out.
pub(crate) struct Listed {
    /// Path to (mode, object id).
    pub(crate) entries: BTreeMap<String, (String, String)>,
    pub(crate) omitted: Vec<Omitted>,
}

impl Checkpoints {
    /// A conversation's checkpoints, from its record's `checkpoint` items.
    pub fn from_items(sidecar: Arc<Sidecar>, items: &[Item], clock: Clock) -> Self {
        let mut taken: Vec<Taken> = Vec::new();
        for item in items {
            if let Item::Checkpoint {
                id,
                kind,
                reason,
                exposed,
                omitted,
                at,
                commit,
                tree,
                omitted_list,
                bytes,
            } = item
            {
                taken.retain(|other| other.id != *id);
                taken.push(Taken {
                    id: *id,
                    kind: *kind,
                    reason: reason.clone(),
                    exposed: *exposed,
                    omitted: *omitted,
                    commit: commit.clone(),
                    tree: tree.clone(),
                    omitted_list: omitted_list.clone(),
                    bytes: *bytes,
                    at: *at,
                });
            }
        }
        Self {
            sidecar,
            clock,
            state: Mutex::new(State {
                taken,
                cache: BTreeMap::new(),
            }),
            #[cfg(test)]
            hold: Mutex::new(None),
        }
    }

    /// Hold every later `take` until the test releases it: the receiver
    /// hears each take begin; each send on the sender releases one.
    #[cfg(test)]
    pub(crate) fn hold_takes(
        &self,
    ) -> (std::sync::mpsc::Receiver<()>, std::sync::mpsc::Sender<()>) {
        let (began, heard) = std::sync::mpsc::channel();
        let (releaser, release) = std::sync::mpsc::channel();
        *lock(&self.hold) = Some(Arc::new(Hold {
            began,
            release: Mutex::new(release),
        }));
        (heard, releaser)
    }

    /// No checkpoints yet.
    pub fn new(sidecar: Arc<Sidecar>, clock: Clock) -> Self {
        Self::from_items(sidecar, &[], clock)
    }

    /// Every checkpoint, oldest first.
    pub fn all(&self) -> Vec<Taken> {
        lock(&self.state).taken.clone()
    }

    /// One checkpoint.
    pub fn get(&self, id: CheckpointId) -> Option<Taken> {
        lock(&self.state)
            .taken
            .iter()
            .find(|taken| taken.id == id)
            .cloned()
    }

    /// The restore menu's rows (§4.3 `checkpoints`).
    pub fn views(&self) -> Vec<CheckpointView> {
        lock(&self.state)
            .taken
            .iter()
            .map(|taken| CheckpointView {
                id: taken.id,
                kind: taken.kind,
                reason: taken.reason.clone(),
                at_turn: None,
                exposed: taken.exposed,
                omitted: taken.omitted,
                bytes: taken.bytes,
            })
            .collect()
    }

    /// The paths `id` left out, by name.
    pub fn omitted(&self, id: CheckpointId) -> Result<Vec<Omitted>, CheckpointError> {
        let Some(taken) = self.get(id) else {
            return Ok(Vec::new());
        };
        let Some(blob) = taken.omitted_list else {
            return Ok(Vec::new());
        };
        let bytes = self
            .sidecar
            .read_blob(&blob)
            .map_err(|_| CheckpointError::Record)?;
        serde_json::from_slice(&bytes).map_err(|_| CheckpointError::Record)
    }

    /// The manifest of a checkpoint without git.
    pub fn manifest(&self, id: CheckpointId) -> Result<Vec<ManifestEntry>, CheckpointError> {
        let bytes = std::fs::read(self.copies_dir(id).join("manifest.json"))
            .map_err(|_| CheckpointError::Record)?;
        let manifest: Manifest =
            serde_json::from_slice(&bytes).map_err(|_| CheckpointError::Record)?;
        Ok(manifest.files)
    }

    /// The bytes a checkpoint without git copied for `path`.
    pub fn copy_bytes(&self, id: CheckpointId, path: &str) -> Result<Vec<u8>, CheckpointError> {
        let file = path
            .split('/')
            .fold(self.copies_dir(id).join("files"), |at, part| at.join(part));
        std::fs::read(file).map_err(|_| CheckpointError::Record)
    }

    fn copies_dir(&self, id: CheckpointId) -> PathBuf {
        self.sidecar.dir().join("checkpoints").join(id.to_string())
    }

    /// Take a checkpoint of `workspace` before writing `paths` (the paths
    /// matter only without git) and record it.
    pub fn take(
        &self,
        workspace: &Workspace,
        runner: &GitRunner,
        reason: CheckpointReason,
        paths: &[String],
    ) -> Result<Taken, CheckpointError> {
        #[cfg(test)]
        {
            let hold = lock(&self.hold).clone();
            if let Some(hold) = hold {
                let _ = hold.began.send(());
                let _ = lock(&hold.release).recv();
            }
        }
        let mut state = lock(&self.state);
        let id = state
            .taken
            .iter()
            .map(|taken| taken.id)
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(CheckpointError::Record)?;
        let taken = match &workspace.repo {
            Repo::Git(_) => {
                // FT6: `.git` is read again before each checkpoint.
                let local = match dotgit::inspect(&workspace.root, runner.env()) {
                    Repo::Git(local) => local,
                    Repo::Without(without) => {
                        return Err(CheckpointError::NotLocal(without.sentence()));
                    }
                    Repo::None => {
                        return Err(CheckpointError::NotLocal(
                            "This folder is no longer in a git repository.".to_owned(),
                        ));
                    }
                };
                self.take_git(&mut state, &local, runner, id, reason)?
            }
            Repo::None | Repo::Without(_) => {
                self.take_copies(workspace, runner, id, reason, paths)?
            }
        };
        let item = Item::Checkpoint {
            id: taken.id,
            kind: taken.kind,
            reason: taken.reason.clone(),
            exposed: taken.exposed,
            omitted: taken.omitted,
            at: taken.at,
            commit: taken.commit.clone(),
            tree: taken.tree.clone(),
            omitted_list: taken.omitted_list.clone(),
            bytes: taken.bytes,
        };
        self.sidecar
            .append(&item)
            .map_err(|_| CheckpointError::Record)?;
        state.taken.push(taken.clone());
        Ok(taken)
    }

    /// The folder's current files and their object ids (step 1 and 2), with
    /// the cache; `write` writes missing blobs (a checkpoint) or not (a
    /// restore's comparison).
    pub(crate) fn list_current(
        &self,
        local: &LocalRepo,
        runner: &GitRunner,
        write: bool,
    ) -> Result<Listed, CheckpointError> {
        let mut state = lock(&self.state);
        list(&mut state.cache, local, runner, write)
    }

    fn take_git(
        &self,
        state: &mut State,
        local: &LocalRepo,
        runner: &GitRunner,
        id: CheckpointId,
        reason: CheckpointReason,
    ) -> Result<Taken, CheckpointError> {
        let listed = list(&mut state.cache, local, runner, true)?;
        let index = local
            .common_dir()
            .join("lattice-chat")
            .join("index")
            .join(self.sidecar.id());
        if let Some(parent) = index.parent() {
            std::fs::create_dir_all(parent).map_err(|_| CheckpointError::Record)?;
        }
        // git refuses a verbatim (`\?\`) index path.
        let private = Extra {
            index_file: Some(PathBuf::from(shown_path(&index))),
            identity: false,
        };
        let out = runner.run(
            local,
            &[OsStr::new("read-tree"), OsStr::new("--empty")],
            &private,
        )?;
        if out.status != 0 {
            return Err(failed(out).into());
        }
        let infos: Vec<String> = listed
            .entries
            .iter()
            .map(|(path, (mode, object))| format!("{mode},{object},{path}"))
            .collect();
        batched(&infos, |batch| {
            let mut args: Vec<OsString> = vec!["update-index".into(), "--add".into()];
            for info in batch {
                args.push("--cacheinfo".into());
                args.push((*info).into());
            }
            let refs: Vec<&OsStr> = args.iter().map(OsString::as_os_str).collect();
            let out = runner.run(local, &refs, &private)?;
            if out.status != 0 {
                return Err(failed(out).into());
            }
            Ok(())
        })?;
        let out = runner.run(local, &[OsStr::new("write-tree")], &private)?;
        if out.status != 0 {
            return Err(failed(out).into());
        }
        let tree = object_line(&out.stdout)?;
        let previous = state
            .taken
            .iter()
            .rev()
            .find(|taken| taken.commit.is_some());
        let at = (self.clock)();
        let omitted_list = self.omitted_blob(&listed.omitted)?;
        let omitted = u32::try_from(listed.omitted.len()).unwrap_or(u32::MAX);
        if let Some(previous) = previous
            && previous.tree.as_deref() == Some(tree.as_str())
        {
            return Ok(Taken {
                id,
                kind: CheckpointKind::Git,
                reason,
                exposed: false,
                omitted,
                commit: previous.commit.clone(),
                tree: Some(tree),
                omitted_list,
                bytes: 0,
                at,
            });
        }
        let message = format!("lattice checkpoint {id}");
        let mut args: Vec<&OsStr> = vec![
            OsStr::new("commit-tree"),
            OsStr::new("--no-gpg-sign"),
            OsStr::new(tree.as_str()),
        ];
        if let Some(parent) = previous.and_then(|previous| previous.commit.as_deref()) {
            args.push(OsStr::new("-p"));
            args.push(OsStr::new(parent));
        }
        args.push(OsStr::new("-m"));
        args.push(OsStr::new(message.as_str()));
        let out = runner.run(
            local,
            &args,
            &Extra {
                index_file: None,
                identity: true,
            },
        )?;
        if out.status != 0 {
            return Err(failed(out).into());
        }
        let commit = object_line(&out.stdout)?;
        let name = format!("{REF_PREFIX}{}/{id}", self.sidecar.id());
        let zero = "0".repeat(commit.len());
        let out = runner.run(
            local,
            &[
                OsStr::new("update-ref"),
                OsStr::new(name.as_str()),
                OsStr::new(commit.as_str()),
                OsStr::new(zero.as_str()),
            ],
            &Extra::default(),
        )?;
        if out.status != 0 {
            return Err(CheckpointError::RefExists(name));
        }
        Ok(Taken {
            id,
            kind: CheckpointKind::Git,
            reason,
            exposed: false,
            omitted,
            commit: Some(commit),
            tree: Some(tree),
            omitted_list,
            bytes: 0,
            at,
        })
    }

    fn omitted_blob(&self, omitted: &[Omitted]) -> Result<Option<String>, CheckpointError> {
        if omitted.is_empty() {
            return Ok(None);
        }
        let bytes = serde_json::to_vec(omitted).map_err(|_| CheckpointError::Record)?;
        self.sidecar
            .put_blob(&bytes, BlobKind::Staged)
            .map(|blob| Some(blob.sha256))
            .map_err(|_| CheckpointError::Record)
    }

    fn take_copies(
        &self,
        workspace: &Workspace,
        runner: &GitRunner,
        id: CheckpointId,
        reason: CheckpointReason,
        paths: &[String],
    ) -> Result<Taken, CheckpointError> {
        let exposed = matches!(
            reason,
            CheckpointReason::BeforeCommand { .. } | CheckpointReason::AfterCommand { .. }
        );
        let dir = self.copies_dir(id);
        let mut files = Vec::new();
        let mut omitted = Vec::new();
        let mut bytes_total = 0u64;
        let unique: BTreeSet<&String> = paths.iter().collect();
        for path in unique {
            let copy = self.copy_one(workspace, runner, &dir, path)?;
            match &copy {
                Copy::Present { bytes, .. } => bytes_total += bytes,
                Copy::Skipped { why } => omitted.push(Omitted {
                    path: path.clone(),
                    why: *why,
                    bytes: None,
                }),
                Copy::Absent => {}
            }
            files.push(ManifestEntry {
                path: path.clone(),
                copy,
            });
        }
        std::fs::create_dir_all(&dir).map_err(|_| CheckpointError::Record)?;
        let manifest =
            serde_json::to_vec(&Manifest { v: 1, files }).map_err(|_| CheckpointError::Record)?;
        atomic_write(&dir.join("manifest.json"), &manifest).map_err(|_| CheckpointError::Record)?;
        Ok(Taken {
            id,
            kind: CheckpointKind::Copies,
            reason,
            exposed,
            omitted: u32::try_from(omitted.len()).unwrap_or(u32::MAX),
            commit: None,
            tree: None,
            omitted_list: self.omitted_blob(&omitted)?,
            bytes: bytes_total,
            at: (self.clock)(),
        })
    }

    /// Copy `path`'s bytes now into `<dir>/files/<path>`, through the path
    /// rules (WP10).
    fn copy_one(
        &self,
        workspace: &Workspace,
        runner: &GitRunner,
        dir: &Path,
        path: &str,
    ) -> Result<Copy, CheckpointError> {
        let resolved = workspace.with_rules(runner, |rules| rules.resolve(path, Want::MayCreate));
        let mut resolved = match resolved {
            Ok(Ok(resolved)) => resolved,
            _ => {
                return Ok(Copy::Skipped {
                    why: OmitWhy::Unreadable,
                });
            }
        };
        if !resolved.exists {
            return Ok(Copy::Absent);
        }
        let Some(file) = resolved.file.take().filter(|_| !resolved.is_dir) else {
            return Ok(Copy::Skipped {
                why: OmitWhy::Unreadable,
            });
        };
        let size = file.metadata().map(|meta| meta.len()).unwrap_or(u64::MAX);
        if size > MAX_SNAPSHOT_BYTES {
            return Ok(Copy::Skipped {
                why: OmitWhy::TooLarge,
            });
        }
        let mut bytes = Vec::new();
        if file
            .take(MAX_SNAPSHOT_BYTES + 1)
            .read_to_end(&mut bytes)
            .is_err()
            || bytes.len() as u64 > MAX_SNAPSHOT_BYTES
        {
            return Ok(Copy::Skipped {
                why: OmitWhy::Unreadable,
            });
        }
        let target = path
            .split('/')
            .fold(dir.join("files"), |at, part| at.join(part));
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent).map_err(|_| CheckpointError::Record)?;
        }
        atomic_write(&target, &bytes).map_err(|_| CheckpointError::Record)?;
        Ok(Copy::Present {
            sha256: sha256_hex(&bytes),
            bytes: bytes.len() as u64,
        })
    }
}

/// The one object id a call printed.
fn object_line(stdout: &[u8]) -> Result<String, CheckpointError> {
    let text = std::str::from_utf8(stdout).map_err(|_| CheckpointError::BadAnswer)?;
    let line = text.trim_end_matches(['\r', '\n']);
    if line.is_empty() || !line.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(CheckpointError::BadAnswer);
    }
    Ok(line.to_owned())
}

/// Steps 1 and 2 (see the module header).
fn list(
    cache: &mut BTreeMap<String, CacheEntry>,
    local: &LocalRepo,
    runner: &GitRunner,
    write: bool,
) -> Result<Listed, CheckpointError> {
    let recorded = filetime_now();
    let root = local.folder();
    let tracked = runner.run(
        local,
        &[OsStr::new("ls-files"), OsStr::new("-z"), OsStr::new("-s")],
        &Extra::default(),
    )?;
    if tracked.status != 0 {
        return Err(failed(tracked).into());
    }
    let others = runner.run(
        local,
        &[
            OsStr::new("ls-files"),
            OsStr::new("-z"),
            OsStr::new("--others"),
            OsStr::new("--exclude-standard"),
        ],
        &Extra::default(),
    )?;
    if others.status != 0 {
        return Err(failed(others).into());
    }
    let mut omitted: Vec<Omitted> = Vec::new();
    // Path to mode, for every candidate.
    let mut candidates: BTreeMap<String, String> = BTreeMap::new();
    for record in records(&tracked.stdout) {
        let Some((mode, path)) = parse_staged(record) else {
            continue;
        };
        match mode.as_str() {
            "160000" => omitted.push(Omitted {
                path,
                why: OmitWhy::NestedRepository,
                bytes: None,
            }),
            "120000" => omitted.push(Omitted {
                path,
                why: OmitWhy::Link,
                bytes: None,
            }),
            _ => {
                candidates.entry(path).or_insert(mode);
            }
        }
    }
    let mut untracked: Vec<String> = Vec::new();
    for record in records(&others.stdout) {
        let Ok(path) = std::str::from_utf8(record) else {
            continue;
        };
        if let Some(folder) = path.strip_suffix('/') {
            omitted.push(Omitted {
                path: folder.to_owned(),
                why: OmitWhy::NestedRepository,
                bytes: None,
            });
        } else if !candidates.contains_key(path) {
            untracked.push(path.to_owned());
        }
    }
    untracked.sort();
    untracked.dedup();
    for (at, path) in untracked.into_iter().enumerate() {
        if at < MAX_NEW_FILES {
            candidates.insert(path, "100644".to_owned());
        } else {
            omitted.push(Omitted {
                path,
                why: OmitWhy::UntrackedCap,
                bytes: None,
            });
        }
    }
    let mut entries: BTreeMap<String, (String, String)> = BTreeMap::new();
    let mut misses: Vec<(String, String, Stamp)> = Vec::new();
    for (path, mode) in candidates {
        match look(root, &path) {
            Looked::File(stamp) => match cache.get(&path) {
                Some(entry) if reusable(entry, &stamp) && (entry.written || !write) => {
                    entries.insert(path, (mode, entry.object.clone()));
                }
                _ => misses.push((path, mode, stamp)),
            },
            // A tracked file missing on disk (or now a folder) is left out:
            // the tree records its deletion.
            Looked::Gone | Looked::Folder => {}
            Looked::Omit(why, bytes) => omitted.push(Omitted { path, why, bytes }),
        }
    }
    let names: Vec<String> = misses.iter().map(|(path, ..)| path.clone()).collect();
    let mut objects: BTreeMap<String, String> = BTreeMap::new();
    hash_objects(local, runner, &names, write, &mut objects, &mut omitted)?;
    for (path, mode, stamp) in misses {
        if let Some(object) = objects.remove(&path) {
            cache.insert(
                path.clone(),
                CacheEntry {
                    stamp,
                    object: object.clone(),
                    recorded,
                    written: write,
                },
            );
            entries.insert(path, (mode, object));
        }
    }
    omitted.sort_by(|a, b| a.path.cmp(&b.path));
    omitted.dedup_by(|a, b| a.path == b.path);
    Ok(Listed { entries, omitted })
}

/// `git hash-object [-w] --no-filters -- <paths>` in batches. A batch git
/// refuses (a file that vanished or cannot be read) is split until the one
/// path it cannot read is found, which is omitted, as `capture.py` does.
fn hash_objects(
    local: &LocalRepo,
    runner: &GitRunner,
    paths: &[String],
    write: bool,
    objects: &mut BTreeMap<String, String>,
    omitted: &mut Vec<Omitted>,
) -> Result<(), CheckpointError> {
    let mut pending: Vec<Vec<String>> = Vec::new();
    batched(paths, |batch| {
        pending.push(batch.iter().map(|path| (*path).to_owned()).collect());
        Ok(())
    })?;
    while let Some(batch) = pending.pop() {
        let mut args: Vec<&OsStr> = vec![OsStr::new("hash-object")];
        if write {
            args.push(OsStr::new("-w"));
        }
        args.push(OsStr::new("--no-filters"));
        args.push(OsStr::new("--"));
        args.extend(batch.iter().map(OsStr::new));
        let out = runner.run(local, &args, &Extra::default())?;
        if out.status == 0 {
            let text = std::str::from_utf8(&out.stdout).map_err(|_| CheckpointError::BadAnswer)?;
            let lines: Vec<&str> = text.lines().collect();
            if lines.len() != batch.len() {
                return Err(CheckpointError::BadAnswer);
            }
            for (path, line) in batch.into_iter().zip(lines) {
                let object = object_line(line.as_bytes())?;
                objects.insert(path, object);
            }
        } else if batch.len() == 1 {
            omitted.push(Omitted {
                path: batch[0].clone(),
                why: OmitWhy::Unreadable,
                bytes: None,
            });
        } else {
            let middle = batch.len() / 2;
            pending.push(batch[middle..].to_vec());
            pending.push(batch[..middle].to_vec());
        }
    }
    Ok(())
}
