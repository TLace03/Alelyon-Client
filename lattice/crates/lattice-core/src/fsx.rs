//! Writing files without losing one: atomic replacement of Lattice's own
//! files, write-once files, moving a file aside instead of removing it, and the
//! one function that removes a file (the chat core's spec §5.8, ND1–ND4).
//!
//! The rules this module keeps:
//! - **ND2. A file is moved aside, never removed.** [`RemovedArea::move_aside`]
//!   moves it into `removed/<n>/<relative path>` (a rename when both are on one
//!   volume) and appends a line to `removed/manifest.jsonl`. Across volumes it
//!   copies, verifies the copy's SHA-256, then renames the source to a sibling
//!   `.<name>.lattice-moved`: a source on another volume is never unlinked.
//! - **ND3. The only removal is of Lattice's own temporaries.**
//!   [`remove_own_temporary`] removes a regular file only when its name is
//!   exactly `.<name>.lattice-<pid>-<n>.tmp`, the shape [`temporary_for`]
//!   makes. Python's shared-store temporaries (`index.tmp`,
//!   `<id>.jsonl.tmp`) never have that shape, so they are never removed here.
//! - Nothing replaces an existing file except [`atomic_write`], which is for
//!   Lattice's own records (a sidecar's `meta.json`, trust, permissions), and
//!   [`replace_shared`], the shared chat store's index replace in Python's
//!   format (row G2); every other placement uses
//!   `lattice_sys::fs::move_no_replace`.
//! - A rename that meets a sharing violation is retried 6 times, 20 ms apart,
//!   as Python's `_replace` does (`history.py`), because an indexer or a
//!   scanner may hold the file for a moment.
//!
//! The never-delete guard (`tests/never_delete_guard.rs`) allows file and
//! directory removal in this file and nowhere else in the crate's code, apart
//! from the run store's two listed temporary removals and the test kit's own
//! temporary folders. Not a port of Python code, though ND2's move follows the
//! web fix's archive (`_archive_one_locked`), which moves and never unlinks.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::sha::{sha256_file, sha256_hex};

/// Numbers this process's temporaries, so no two writes share a name.
static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);
const RETRIES: u32 = 6;
const RETRY_PAUSE: Duration = Duration::from_millis(20);
/// Appended to a source that was copied to another volume and verified.
const MOVED_SUFFIX: &str = ".lattice-moved";
const MANIFEST: &str = "manifest.jsonl";

fn file_name(path: &Path) -> io::Result<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "a file name is needed"))
}

/// A new temporary name beside `target`: `.<name>.lattice-<pid>-<n>.tmp`.
pub fn temporary_for(target: &Path) -> io::Result<PathBuf> {
    let name = file_name(target)?;
    let n = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(target.with_file_name(format!(".{name}.lattice-{}-{n}.tmp", std::process::id())))
}

/// The number the next [`temporary_for`] name takes (tests only: a
/// falsifier pre-creates the names a predictable scheme would use next).
#[cfg(test)]
pub(crate) fn next_temporary_number() -> u64 {
    TEMPORARY_COUNTER.load(Ordering::Relaxed)
}

/// True when `name` has exactly the shape [`temporary_for`] makes.
pub fn is_own_temporary_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix('.') else {
        return false;
    };
    let Some(rest) = rest.strip_suffix(".tmp") else {
        return false;
    };
    let Some((stem, numbers)) = rest.rsplit_once(".lattice-") else {
        return false;
    };
    let Some((pid, n)) = numbers.split_once('-') else {
        return false;
    };
    let digits = |text: &str| !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit());
    !stem.is_empty() && digits(pid) && digits(n)
}

/// Remove `path` when, and only when, it is one of Lattice's own temporaries: a
/// regular file (not a link, not a folder) whose name [`is_own_temporary_name`]
/// accepts. `Ok(false)` when it is not one, or is already gone.
pub fn remove_own_temporary(path: &Path) -> io::Result<bool> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    if !is_own_temporary_name(name) {
        return Ok(false);
    }
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_file() => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    }
    fs::remove_file(path)?;
    Ok(true)
}

/// Make a fresh temporary folder beside `target`, named as
/// [`temporary_for`] names a temporary file, so that only
/// [`remove_own_temporary_dir`] can take it away again. Refused when the
/// name is taken.
pub fn create_own_temporary_dir(target: &Path) -> io::Result<PathBuf> {
    let dir = temporary_for(target)?;
    fs::create_dir(&dir)?;
    Ok(dir)
}

/// How many names [`create_private_temporary_dir`] tries.
const PRIVATE_DIR_TRIES: u32 = 16;

/// Make a fresh, private temporary folder in `root` (spec §22.6 LR6a): its
/// name is `.<prefix>-<random>.lattice-<pid>-0.tmp`, where `<random>` is a
/// version-4 UUID's 32 hex digits (122 bits from the operating system's
/// generator), so no one can predict it and pre-create it; a name that is
/// taken anyway (`AlreadyExists`) is retried with a new one. The folder is
/// made by `lattice_sys::fs::create_private_dir`: an explicit, protected DACL
/// of SYSTEM, Administrators and the user, inheriting nothing from `root`.
/// Only [`remove_own_temporary_dir`] can take it away again, once empty.
pub fn create_private_temporary_dir(root: &Path, prefix: &str) -> io::Result<PathBuf> {
    create_private_temporary_dir_named(root, prefix, || uuid::Uuid::new_v4().simple().to_string())
}

/// [`create_private_temporary_dir`] with the random part from `random`.
pub(crate) fn create_private_temporary_dir_named(
    root: &Path,
    prefix: &str,
    mut random: impl FnMut() -> String,
) -> io::Result<PathBuf> {
    let mut taken = None;
    for _ in 0..PRIVATE_DIR_TRIES {
        // The random part makes the name unique; the counter stays 0.
        let dir = root.join(format!(
            ".{prefix}-{}.lattice-{}-0.tmp",
            random(),
            std::process::id()
        ));
        match lattice_sys::fs::create_private_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => taken = Some(error),
            Err(error) => return Err(error),
        }
    }
    Err(taken.unwrap_or_else(|| io::Error::from(io::ErrorKind::AlreadyExists)))
}

/// Remove `path` when, and only when, it is an EMPTY folder (not a link)
/// whose name [`is_own_temporary_name`] accepts: the managed llama.cpp
/// server's token folder once its token file is gone (spec §22 LR6). A folder
/// that still holds anything is left alone, so this can never take a file
/// with it. `Ok(false)` when it is not one, or is already gone.
pub fn remove_own_temporary_dir(path: &Path) -> io::Result<bool> {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return Ok(false);
    };
    if !is_own_temporary_name(name) {
        return Ok(false);
    }
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_dir() => {}
        Ok(_) => return Ok(false),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    }
    if fs::read_dir(path)?.next().is_some() {
        return Ok(false);
    }
    fs::remove_dir(path)?;
    Ok(true)
}

/// Whether `error` is worth another try: a sharing violation (32) or an
/// access refusal (5), which an indexer or a scanner holding the file causes.
fn transient(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(5 | 32))
}

#[cfg(test)]
thread_local! {
    /// Transient failures `retrying` met on this thread (tests: the interop
    /// test I4 shows that a write met another process's handle and retried).
    static TRANSIENT_FAILURES: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// How many transient failures `retrying` has met on this thread.
#[cfg(test)]
pub(crate) fn transient_failures() -> u32 {
    TRANSIENT_FAILURES.with(std::cell::Cell::get)
}

#[cfg(test)]
thread_local! {
    /// A test's hook, run on this thread at each transient failure before
    /// the retry's pause (I4: the write that met Python's handle waits there
    /// until the test has had Python release it).
    static ON_TRANSIENT: std::cell::RefCell<Option<Box<dyn FnMut()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Set (or clear) this thread's transient-failure hook (tests only).
#[cfg(test)]
pub(crate) fn on_transient(hook: Option<Box<dyn FnMut()>>) {
    ON_TRANSIENT.with(|slot| *slot.borrow_mut() = hook);
}

/// Run `operation`, retrying a transient failure 6 times, 20 ms apart.
fn retrying<T>(mut operation: impl FnMut() -> io::Result<T>) -> io::Result<T> {
    let mut attempt = 1;
    loop {
        match operation() {
            Err(error) if transient(&error) && attempt < RETRIES => {
                #[cfg(test)]
                {
                    TRANSIENT_FAILURES.with(|seen| seen.set(seen.get() + 1));
                    let hook = ON_TRANSIENT.with(|slot| slot.borrow_mut().take());
                    if let Some(mut hook) = hook {
                        hook();
                        ON_TRANSIENT.with(|slot| *slot.borrow_mut() = Some(hook));
                    }
                }
                attempt += 1;
                std::thread::sleep(RETRY_PAUSE);
            }
            other => return other,
        }
    }
}

fn write_temporary(temporary: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary)?;
    file.write_all(bytes)?;
    file.sync_all()
}

/// Replace `target` (one of Lattice's own records) with `bytes`, atomically: a
/// reader sees the old bytes or the new ones, never a mixture. The folder must
/// exist. On failure the temporary is removed and `target` is unchanged.
pub fn atomic_write(target: &Path, bytes: &[u8]) -> io::Result<()> {
    let temporary = temporary_for(target)?;
    let result = write_temporary(&temporary, bytes)
        .and_then(|()| retrying(|| fs::rename(&temporary, target)));
    if result.is_err() {
        let _ = remove_own_temporary(&temporary);
    }
    result
}

/// Write `bytes` to `target` once: when `target` exists already, nothing is
/// written (a content-addressed file holds the same bytes by its name).
/// Returns whether this call wrote it.
pub fn write_once(target: &Path, bytes: &[u8]) -> io::Result<bool> {
    if fs::symlink_metadata(target).is_ok() {
        return Ok(false);
    }
    let temporary = temporary_for(target)?;
    let placed = write_temporary(&temporary, bytes)
        .and_then(|()| retrying(|| lattice_sys::fs::move_no_replace(&temporary, target)));
    match placed {
        Ok(()) => Ok(true),
        Err(error) => {
            let _ = remove_own_temporary(&temporary);
            if error.kind() == io::ErrorKind::AlreadyExists {
                Ok(false)
            } else {
                Err(error)
            }
        }
    }
}

/// Move one of Lattice's own folders (a sidecar bound to an earlier
/// conversation) to `to`, on the same volume, never replacing.
pub fn move_dir_aside(from: &Path, to: &Path) -> io::Result<()> {
    retrying(|| lattice_sys::fs::move_no_replace(from, to))
}

/// The shared chat store's replace (S4′, S19; row G2): `from` renamed onto
/// `to`, replacing it, retried as Python's `_replace` retries a sharing
/// violation. The store's format is Python's: its indexes are written to
/// Python's own temporary names (`index.tmp`, `<id>.jsonl.tmp`) and replaced
/// into place, so this is the one replace outside [`atomic_write`]. Nothing
/// is removed: the replaced bytes are the old version of the same file, and
/// a temporary left by a failed replace stays for the next write (ND3).
pub fn replace_shared(from: &Path, to: &Path) -> io::Result<()> {
    retrying(|| fs::rename(from, to))
}

/// Move a file to `to`, never replacing one, retried on a sharing violation:
/// the shared chat store's archive moves (S19, S22), which never overwrite.
pub fn move_new(from: &Path, to: &Path) -> io::Result<()> {
    retrying(|| lattice_sys::fs::move_no_replace(from, to))
}

/// Why a workspace file was moved aside.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoveReason {
    /// The reader kept a staged delete.
    KeptDelete,
    /// The reader kept a restore that removes a file made after the checkpoint.
    KeptRestore,
    /// The reader undid a command that had made the file.
    UndoneCommand,
    /// `ReplaceFileW`'s backup of the bytes a Keep replaced.
    ReplacedBackup,
}

/// One line of `removed/manifest.jsonl`: where a file went, and what it held.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovedAside {
    /// The workspace-relative path it had.
    pub path: String,
    /// Where it is now, relative to the removed area (`<n>/<path>`).
    pub to: String,
    pub sha256: String,
    pub bytes: u64,
    pub why: MoveReason,
    pub at: f64,
    /// Across volumes: the source, verified and copied, was renamed to this
    /// sibling (`.<name>.lattice-moved`) rather than removed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_kept_as: Option<String>,
}

/// A conversation's `removed/` folder.
#[derive(Clone, Debug)]
pub struct RemovedArea {
    root: PathBuf,
}

/// The components of a workspace-relative path, refused when one could leave
/// the removed area (`..`, `.`, empty, a drive or a backslash).
fn components(relative: &str) -> io::Result<Vec<&str>> {
    let parts: Vec<&str> = relative.split('/').collect();
    let bad = |part: &&str| {
        part.is_empty() || *part == "." || *part == ".." || part.contains(['\\', ':', '\0'])
    };
    if parts.iter().any(bad) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a workspace-relative path",
        ));
    }
    Ok(parts)
}

impl RemovedArea {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The next free slot number: one past the highest numbered folder (1 when
    /// there is none).
    pub fn next_slot(&self) -> io::Result<u32> {
        let entries = match fs::read_dir(&self.root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(1),
            Err(error) => return Err(error),
        };
        let mut highest = 0u32;
        for entry in entries {
            let entry = entry?;
            if let Some(n) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse::<u32>().ok())
            {
                highest = highest.max(n);
            }
        }
        highest
            .checked_add(1)
            .ok_or_else(|| io::Error::other("no slot number is left"))
    }

    /// Move the regular file `source` (the workspace file at `relative`) into
    /// `<slot>/<relative>` and record it. Never removes `source`: on one volume
    /// it is renamed; across volumes it is copied, the copy verified, and the
    /// source renamed to a sibling (ND2). An existing destination refuses.
    pub fn move_aside(
        &self,
        slot: u32,
        relative: &str,
        source: &Path,
        why: MoveReason,
        at: f64,
    ) -> io::Result<MovedAside> {
        self.move_aside_with(
            slot,
            relative,
            source,
            why,
            at,
            lattice_sys::fs::move_no_replace,
        )
    }

    /// [`Self::move_aside`] with the same-volume move given, so a test can
    /// make it report another volume.
    fn move_aside_with(
        &self,
        slot: u32,
        relative: &str,
        source: &Path,
        why: MoveReason,
        at: f64,
        rename: impl Fn(&Path, &Path) -> io::Result<()>,
    ) -> io::Result<MovedAside> {
        let parts = components(relative)?;
        let meta = fs::symlink_metadata(source)?;
        if !meta.file_type().is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "only a regular file is moved aside",
            ));
        }
        let (sha256, bytes) = sha256_file(source)?;
        let mut target = self.root.join(slot.to_string());
        for part in &parts {
            target.push(part);
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut source_kept_as = None;
        match retrying(|| rename(source, &target)) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::CrossesDevices => {
                source_kept_as = Some(copy_then_keep_source(source, &target, &sha256)?);
            }
            Err(error) => return Err(error),
        }
        let record = MovedAside {
            path: relative.to_owned(),
            to: format!("{slot}/{relative}"),
            sha256,
            bytes,
            why,
            at,
            source_kept_as,
        };
        self.append_manifest(&record)?;
        Ok(record)
    }

    fn append_manifest(&self, record: &MovedAside) -> io::Result<()> {
        fs::create_dir_all(&self.root)?;
        let mut line = serde_json::to_vec(record).map_err(io::Error::other)?;
        line.push(b'\n');
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join(MANIFEST))?;
        file.write_all(&line)?;
        file.sync_all()
    }

    /// Every record of the manifest, oldest first. A line that is not a record
    /// (a torn last line) is skipped.
    pub fn manifest(&self) -> io::Result<Vec<MovedAside>> {
        let text = match fs::read_to_string(self.root.join(MANIFEST)) {
            Ok(text) => text,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        Ok(text
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect())
    }
}

/// Across volumes: copy `source` to `target` (never replacing), check the
/// copy's hash, then rename `source` to a free sibling `.<name>.lattice-moved`
/// (`-2`, `-3`, …). Returns the sibling's name. On a hash mismatch nothing more
/// happens to the source.
fn copy_then_keep_source(source: &Path, target: &Path, sha256: &str) -> io::Result<String> {
    let bytes = fs::read(source)?;
    if sha256_hex(&bytes) != sha256 {
        return Err(io::Error::other(
            "the file changed while it was being moved aside",
        ));
    }
    if !write_once(target, &bytes)? {
        return Err(io::Error::from(io::ErrorKind::AlreadyExists));
    }
    if sha256_file(target)?.0 != sha256 {
        return Err(io::Error::other(
            "the copy did not verify; the source was left in place",
        ));
    }
    let name = file_name(source)?;
    for attempt in 1..=100u32 {
        let sibling = if attempt == 1 {
            format!(".{name}{MOVED_SUFFIX}")
        } else {
            format!(".{name}{MOVED_SUFFIX}-{attempt}")
        };
        match retrying(|| {
            lattice_sys::fs::move_no_replace(source, &source.with_file_name(&sibling))
        }) {
            Ok(()) => return Ok(sibling),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::other("no free name beside the source"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::testkit::TempDir;

    /// Every file and folder under `root`, relative, with `/`.
    fn tree(root: &Path) -> BTreeSet<String> {
        fn walk(root: &Path, dir: &Path, out: &mut BTreeSet<String>) {
            for entry in fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                out.insert(
                    path.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
                if path.is_dir() {
                    walk(root, &path, out);
                }
            }
        }
        let mut out = BTreeSet::new();
        walk(root, root, &mut out);
        out
    }

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|item| (*item).to_owned()).collect()
    }

    #[test]
    fn temporary_names_have_one_shape() {
        let temporary = temporary_for(Path::new("dir/meta.json")).unwrap();
        let name = temporary.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with(".meta.json.lattice-"), "{name}");
        assert!(is_own_temporary_name(name), "{name}");
        assert_eq!(temporary.parent(), Some(Path::new("dir")));
        for good in [".a.lattice-1-0.tmp", ".x.y.lattice-4242-17.tmp"] {
            assert!(is_own_temporary_name(good), "{good}");
        }
        for bad in [
            "index.tmp",
            "abc.jsonl.tmp",
            "index.json",
            ".lattice-1-0.tmp",
            ".a.lattice-1.tmp",
            ".a.lattice-x-0.tmp",
            ".a.lattice-1-0.tmp.bak",
            "a.lattice-1-0.tmp",
            ".a.lattice-1-0",
            "0123456789ab.run.json.1-2.tmp",
        ] {
            assert!(!is_own_temporary_name(bad), "{bad}");
        }
    }

    #[test]
    fn only_an_own_temporary_is_ever_removed() {
        let dir = TempDir::new("fsx-remove");
        let own = dir.path().join(".meta.json.lattice-7-3.tmp");
        let python = dir.path().join("index.tmp");
        let transcript = dir.path().join("0123456789ab.jsonl.tmp");
        let record = dir.path().join("meta.json");
        let folder = dir.path().join(".d.lattice-7-4.tmp");
        for path in [&own, &python, &transcript, &record] {
            fs::write(path, "x").unwrap();
        }
        fs::create_dir(&folder).unwrap();
        for kept in [&python, &transcript, &record, &folder] {
            assert!(!remove_own_temporary(kept).unwrap(), "{}", kept.display());
        }
        assert!(remove_own_temporary(&own).unwrap());
        assert!(!remove_own_temporary(&own).unwrap(), "already gone");
        assert_eq!(
            tree(dir.path()),
            set(&[
                "index.tmp",
                "0123456789ab.jsonl.tmp",
                "meta.json",
                ".d.lattice-7-4.tmp"
            ])
        );
    }

    #[test]
    fn only_an_empty_own_temporary_folder_is_ever_removed() {
        let dir = TempDir::new("fsx-remove-dir");
        let own = create_own_temporary_dir(&dir.path().join("alelyon-llama")).unwrap();
        let name = own.file_name().unwrap().to_str().unwrap().to_owned();
        assert!(name.starts_with(".alelyon-llama.lattice-"), "{name}");
        let full = create_own_temporary_dir(&dir.path().join("other")).unwrap();
        fs::write(full.join(".api-key.lattice-1-1.tmp"), "x").unwrap();
        let plain = dir.path().join("models");
        fs::create_dir(&plain).unwrap();
        let file = dir.path().join(".f.lattice-7-9.tmp");
        fs::write(&file, "x").unwrap();
        assert!(
            !remove_own_temporary_dir(&full).unwrap(),
            "a folder that holds a file stays"
        );
        assert!(
            !remove_own_temporary_dir(&plain).unwrap(),
            "not an own temporary"
        );
        assert!(
            !remove_own_temporary_dir(&file).unwrap(),
            "a file is not a folder"
        );
        assert!(remove_own_temporary_dir(&own).unwrap());
        assert!(!remove_own_temporary_dir(&own).unwrap(), "already gone");
        let full_name = full.file_name().unwrap().to_str().unwrap().to_owned();
        assert_eq!(
            tree(dir.path()),
            set(&[
                full_name.as_str(),
                &format!("{full_name}/.api-key.lattice-1-1.tmp"),
                "models",
                ".f.lattice-7-9.tmp"
            ])
        );
    }

    /// LR6a: a private temporary folder has a random name of the own
    /// temporaries' shape (so only `remove_own_temporary_dir` takes it away),
    /// and a name that is taken is retried with a new one.
    /// Mutant: no retry on `AlreadyExists`.
    #[test]
    fn a_private_temporary_folder_has_a_random_name_and_a_taken_one_is_retried() {
        let dir = TempDir::new("fsx-private");
        let a = create_private_temporary_dir(dir.path(), "alelyon-llama").unwrap();
        let b = create_private_temporary_dir(dir.path(), "alelyon-llama").unwrap();
        let random = |path: &Path| -> String {
            let name = path.file_name().unwrap().to_str().unwrap().to_owned();
            assert!(is_own_temporary_name(&name), "{name}");
            let rest = name.strip_prefix(".alelyon-llama-").unwrap();
            rest.split_once(".lattice-").unwrap().0.to_owned()
        };
        let (ra, rb) = (random(&a), random(&b));
        assert_eq!(ra.len(), 32);
        assert!(ra.bytes().all(|byte| byte.is_ascii_hexdigit()), "{ra}");
        assert_ne!(ra, rb, "two folders, two random names");
        assert!(a.is_dir() && b.is_dir());

        // Someone made the next name first: the folder is made under another.
        let squatted = dir.path().join(format!(
            ".alelyon-llama-squatted.lattice-{}-0.tmp",
            std::process::id()
        ));
        fs::create_dir(&squatted).unwrap();
        fs::write(squatted.join("planted"), "x").unwrap();
        let mut names = ["squatted", "fresh"].into_iter().map(String::from);
        let made = create_private_temporary_dir_named(dir.path(), "alelyon-llama", || {
            names.next().unwrap()
        })
        .unwrap();
        assert_ne!(made, squatted);
        assert_eq!(random(&made), "fresh");
        assert!(
            made.read_dir().unwrap().next().is_none(),
            "a new, empty folder"
        );
        assert_eq!(fs::read_to_string(squatted.join("planted")).unwrap(), "x");
        // Every name taken: an error, never someone else's folder.
        let error = create_private_temporary_dir_named(dir.path(), "alelyon-llama", || {
            "squatted".to_owned()
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
    }

    /// LR6a: the folder's DACL is the private one, inheriting nothing from the
    /// temporary root, whatever the root grants (read back with
    /// `GetNamedSecurityInfoW`).
    /// Mutant: the folder made by `fs::create_dir`.
    #[cfg(windows)]
    #[test]
    fn a_private_temporary_folder_inherits_nothing_from_its_root() {
        use lattice_sys::fs::{ADMINISTRATORS_SID, SYSTEM_SID, current_user_sid, read_dacl};
        let dir = TempDir::new("fsx-private-dacl");
        let made = create_private_temporary_dir(dir.path(), "alelyon-llama").unwrap();
        let dacl = read_dacl(&made).unwrap();
        assert!(dacl.protected, "{dacl:?}");
        assert!(!dacl.aces.iter().any(|ace| ace.inherited), "{dacl:?}");
        let user = current_user_sid().unwrap();
        let mut sids: Vec<&str> = dacl.aces.iter().map(|ace| ace.sid.as_str()).collect();
        sids.sort();
        let mut expected = vec![SYSTEM_SID, ADMINISTRATORS_SID, user.as_str()];
        expected.sort();
        assert_eq!(sids, expected);
    }

    #[test]
    fn an_atomic_write_replaces_and_leaves_no_temporary() {
        let dir = TempDir::new("fsx-atomic");
        let target = dir.path().join("meta.json");
        atomic_write(&target, b"{\"v\":1}").unwrap();
        atomic_write(&target, b"{\"v\":2}").unwrap();
        assert_eq!(fs::read(&target).unwrap(), b"{\"v\":2}");
        assert_eq!(tree(dir.path()), set(&["meta.json"]));
        // A folder in the way: the write fails, and the temporary is gone.
        let blocked = dir.path().join("blocked");
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("inside.txt"), "kept").unwrap();
        assert!(atomic_write(&blocked, b"x").is_err());
        assert_eq!(
            tree(dir.path()),
            set(&["meta.json", "blocked", "blocked/inside.txt"])
        );
    }

    #[test]
    fn write_once_never_overwrites() {
        let dir = TempDir::new("fsx-once");
        let blob = dir.path().join("ab12");
        assert!(write_once(&blob, b"first").unwrap());
        assert!(!write_once(&blob, b"second").unwrap());
        assert_eq!(fs::read(&blob).unwrap(), b"first");
        assert_eq!(tree(dir.path()), set(&["ab12"]));
    }

    #[test]
    fn a_file_is_moved_aside_with_a_manifest_line_and_nothing_is_lost() {
        let workspace = TempDir::new("fsx-ws");
        let state = TempDir::new("fsx-state");
        let source = workspace.path().join("src").join("old.rs");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, "fn old() {}\n").unwrap();
        let area = RemovedArea::new(state.path().join("removed"));
        assert_eq!(area.next_slot().unwrap(), 1);
        let record = area
            .move_aside(1, "src/old.rs", &source, MoveReason::KeptDelete, 1.5)
            .unwrap();
        assert!(!source.exists());
        let moved = state
            .path()
            .join("removed")
            .join("1")
            .join("src")
            .join("old.rs");
        assert_eq!(fs::read_to_string(&moved).unwrap(), "fn old() {}\n");
        assert_eq!(record.sha256, sha256_hex(b"fn old() {}\n"));
        assert_eq!((record.bytes, record.to.as_str()), (12, "1/src/old.rs"));
        assert_eq!(record.source_kept_as, None);
        assert_eq!(area.manifest().unwrap(), vec![record]);
        assert_eq!(area.next_slot().unwrap(), 2);
        assert_eq!(tree(workspace.path()), set(&["src"]));
        assert_eq!(
            tree(state.path()),
            set(&[
                "removed",
                "removed/1",
                "removed/1/src",
                "removed/1/src/old.rs",
                "removed/manifest.jsonl"
            ])
        );
    }

    #[test]
    fn across_volumes_the_source_is_copied_verified_and_renamed_never_unlinked() {
        let workspace = TempDir::new("fsx-xws");
        let state = TempDir::new("fsx-xstate");
        let source = workspace.path().join("data.bin");
        fs::write(&source, [7u8; 4096]).unwrap();
        // A sibling of the first choice exists already: the next name is taken.
        fs::write(workspace.path().join(".data.bin.lattice-moved"), "older").unwrap();
        let area = RemovedArea::new(state.path().join("removed"));
        let other_volume =
            |_: &Path, _: &Path| -> io::Result<()> { Err(io::ErrorKind::CrossesDevices.into()) };
        let record = area
            .move_aside_with(
                3,
                "data.bin",
                &source,
                MoveReason::UndoneCommand,
                2.0,
                other_volume,
            )
            .unwrap();
        assert_eq!(
            record.source_kept_as.as_deref(),
            Some(".data.bin.lattice-moved-2")
        );
        assert_eq!(
            fs::read(state.path().join("removed/3/data.bin")).unwrap(),
            vec![7u8; 4096]
        );
        assert_eq!(
            tree(workspace.path()),
            set(&[".data.bin.lattice-moved", ".data.bin.lattice-moved-2"])
        );
        assert_eq!(
            fs::read(workspace.path().join(".data.bin.lattice-moved-2")).unwrap(),
            vec![7u8; 4096]
        );
        assert_eq!(area.manifest().unwrap(), vec![record]);
    }

    #[test]
    fn a_taken_destination_or_a_bad_path_refuses_and_moves_nothing() {
        let workspace = TempDir::new("fsx-refuse");
        let state = TempDir::new("fsx-refuse-state");
        let source = workspace.path().join("a.txt");
        fs::write(&source, "a").unwrap();
        let area = RemovedArea::new(state.path().join("removed"));
        fs::create_dir_all(state.path().join("removed/1")).unwrap();
        fs::write(state.path().join("removed/1/a.txt"), "older").unwrap();
        let before = (tree(workspace.path()), tree(state.path()));
        let taken = area.move_aside(1, "a.txt", &source, MoveReason::KeptDelete, 0.0);
        assert_eq!(taken.unwrap_err().kind(), io::ErrorKind::AlreadyExists);
        for bad in [
            "../a.txt",
            "x/../../a.txt",
            "",
            "a//b",
            "C:/a",
            "a\\b",
            "./a",
        ] {
            let refused = area.move_aside(2, bad, &source, MoveReason::KeptDelete, 0.0);
            assert_eq!(
                refused.unwrap_err().kind(),
                io::ErrorKind::InvalidInput,
                "{bad}"
            );
        }
        let folder = workspace.path().join("dir");
        fs::create_dir(&folder).unwrap();
        let refused = area.move_aside(2, "dir", &folder, MoveReason::KeptDelete, 0.0);
        assert_eq!(refused.unwrap_err().kind(), io::ErrorKind::InvalidInput);
        fs::remove_dir(&folder).unwrap();
        assert_eq!((tree(workspace.path()), tree(state.path())), before);
        assert_eq!(fs::read_to_string(&source).unwrap(), "a");
        assert!(area.manifest().unwrap().is_empty());
    }

    #[test]
    fn a_folder_of_lattices_own_moves_aside_without_replacing() {
        let state = TempDir::new("fsx-dir");
        let from = state.path().join("abc");
        fs::create_dir(&from).unwrap();
        fs::write(from.join("items.jsonl"), "{}\n").unwrap();
        let to = state.path().join("abc~0000000000000000");
        move_dir_aside(&from, &to).unwrap();
        assert_eq!(fs::read_to_string(to.join("items.jsonl")).unwrap(), "{}\n");
        fs::create_dir(&from).unwrap();
        assert_eq!(
            move_dir_aside(&from, &to).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
    }
}
