//! The writer lease (ADR-0038; the chat core's spec §6.5): one lease per
//! checkout, shared with the Python hosts (the PyQt IDE, the web's agent
//! runs), so two agents never write one checkout at once.
//!
//! **The file** is Python's: `session.CHECKOUT_LEASE_FILE`
//! (`agent-session.lock`) under `RepositoryStatePaths.checkout_file(..)`,
//! that is `<state home>/fleet_repositories/<repository namespace>/
//! checkouts/<checkout namespace>/agent-session.lock`. Ported here:
//! - **The state home**: `worktree_cache.selected_repository_state_root()`,
//!   not `StateRoot::globals`: `ALELYON_HOME`'s state home when that is set
//!   (non-empty), else the per-user state home, which is `~/.alelyon/globals`
//!   in a source checkout and the platform folder's `globals/` when packaged
//!   ([`selected_repository_state_root`]). Python's W5 (a packaged run inside
//!   an AI agent's session moves to an agent state folder) is not ported: a
//!   shipped native build is not started by an agent.
//! - **The repository**: git's exact top level of the attached folder, never
//!   the folder itself (`repository_context.py` refuses anything else), and
//!   git's common folder (`rev-parse --path-format=absolute
//!   --git-common-dir`), both resolved to their final paths.
//! - **The arithmetic** ([`repository_namespace`], [`checkout_context_id`],
//!   [`checkout_namespace`], [`lease_path`]): SHA-256 over Python's exact
//!   strings, pinned by the `lease/derive.json` golden recorded from the
//!   Python functions.
//! - **The identity reading** ([`namespace_key`], [`read_marker`]): the
//!   common folder's `stat()` (device, 128-bit file id, birth time in ns),
//!   and the top level's and its `.git`'s `lstat()` (device, file id, mode,
//!   birth time, and for a `.git` file the SHA-256 of its bytes as an
//!   integer), as Python 3.12 reads them on Windows. Interop test I13
//!   compares the whole path with Python's over live repositories.
//!
//! **The lock** is Python's `process_lease` semantics: an OS lock only, owned
//! by the open handle; the file never holds an owner and is never deleted; a
//! lease path that is a link, a junction or any other reparse point, or not
//! a regular file, is refused, never followed. Python locks byte 0
//! (`msvcrt.locking(fd, LK_NBLCK, 1)`); `File::try_lock` locks the whole file,
//! which overlaps byte 0, so each side sees the other's lock (S15′; interop
//! test I12).
//!
//! **When it is held** ([`WriterLease`]): a conversation takes it at its
//! first Keep (and, in row E7, its first approved command), and holds it
//! while it has staged changes or a running command; a second writer is
//! refused with "Another Lattice agent is editing this folder." Ask mode
//! needs no lease.
//!
//! A folder without git has no checkout to share with Python; its lease is
//! native only, at `<native>/chat/leases/<workspace id>.lock` (a native
//! addition, so two native conversations still exclude each other).

use std::ffi::OsStr;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use lattice_sys::fs::{Access, LinkKind, file_attributes, file_identity, file_times};

use crate::env::{self, Env};
use crate::git::dotgit::{self, Repo};
use crate::git::runner::{Extra, GitError, GitRunner};
use crate::localfs::{LinkRule, open_walk};
use crate::policy::Lease;
use crate::sha::sha256_hex;
use crate::state::{Platform, StateRoot, forced_packaged, home_dir, tidy, user_state_dir};
use crate::workspace::{Workspace, shown_path};

/// `session.CHECKOUT_LEASE_FILE`: one name for every host.
pub const CHECKOUT_LEASE_FILE: &str = "agent-session.lock";
/// `repository_context._CHECKOUT_NAMESPACE_SCHEMA`.
pub const CHECKOUT_NAMESPACE_SCHEMA: &str = "alelyon-checkout-state-v1";
/// `worktree_cache._GIT_POINTER_IDENTITY_BYTES`.
pub const GIT_POINTER_IDENTITY_BYTES: u64 = 4096;

/// Why no lease path or no lock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaseError {
    /// Git could not say where the repository is.
    Git(GitError),
    /// The folder's `.git` names the network now (FT6).
    NotLocal,
    /// A path could not be opened or resolved.
    Unavailable(&'static str),
    /// The checkout's identity is not strong enough to tell incarnations
    /// apart (Python refuses the same).
    WeakIdentity,
    /// The lease path is a link, a reparse point or not a regular file
    /// (`UnsafeLeasePath`), or its identity changed while it was opened.
    Unsafe(&'static str),
    /// The lock call failed for a reason other than contention.
    Io(String),
}

impl LeaseError {
    pub fn sentence(&self) -> String {
        match self {
            Self::Git(error) => error.sentence(),
            Self::NotLocal => {
                "This folder's .git names a network path, so Lattice will not take its lease."
                    .to_owned()
            }
            Self::Unavailable(what) | Self::Unsafe(what) => {
                format!("Lattice could not take this folder's writer lease: {what}.")
            }
            Self::WeakIdentity => "This folder's file system cannot tell checkouts apart, so Lattice will not take its writer lease.".to_owned(),
            Self::Io(_) => "Lattice could not take this folder's writer lease.".to_owned(),
        }
    }
}

impl From<GitError> for LeaseError {
    fn from(error: GitError) -> Self {
        Self::Git(error)
    }
}

// ---------------------------------------------------------------- state home

/// `worktree_cache.selected_repository_state_root()`: `ALELYON_HOME` set (to
/// any non-empty text) means Python's `paths.GLOBALS_DIR`, which is `state`'s
/// state home resolved from the same environment; otherwise
/// `paths.user_state_home()`: the platform folder's `globals/` when packaged
/// (installed, or `ALELYON_FORCE_PACKAGED`), `~/.alelyon/globals` in a source
/// checkout.
pub fn selected_repository_state_root(
    env: &dyn Env,
    state: &StateRoot,
    platform: Platform,
) -> PathBuf {
    if env::text(env, "ALELYON_HOME").is_some_and(|home| !home.is_empty()) {
        return state.globals.clone();
    }
    if state.installed || forced_packaged(env) {
        return user_state_dir(env, platform).join("globals");
    }
    home_dir(env, platform).join(".alelyon").join("globals")
}

// ---------------------------------------------------------------- arithmetic

/// What `_repository_namespace` names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamespaceKind {
    /// `git_common=True`: the repository's common folder.
    GitCommonDir,
    /// `git_common=False`.
    SelectedRoot,
}

impl NamespaceKind {
    fn word(self) -> &'static str {
        match self {
            Self::GitCommonDir => "git-common-dir",
            Self::SelectedRoot => "selected-root",
        }
    }
}

/// A folder's identity as `_repository_namespace` reads it: `stat()`'s
/// device, file id and birth time, as Python's integers print them (any
/// width, `-1` for none).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Incarnation {
    pub device: String,
    pub inode: String,
    /// `st_birthtime_ns`.
    pub birth_ns: String,
}

/// `_repository_namespace`'s key: `"<device>:<inode>:<birth>"` (a birth of 0
/// is `-1`) when the file id is usable (neither -1 nor 0), else
/// `"<normcase'd path>\0metadata-unavailable"`, as when the identity could
/// not be read.
pub fn namespace_key(incarnation: Option<&Incarnation>, normalized: &str) -> String {
    match incarnation {
        Some(found) if found.inode != "0" && found.inode != "-1" => {
            let birth = if found.birth_ns == "0" {
                "-1"
            } else {
                found.birth_ns.as_str()
            };
            format!("{}:{}:{birth}", found.device, found.inode)
        }
        _ => format!("{normalized}\0metadata-unavailable"),
    }
}

/// `_repository_namespace`: SHA-256 of `"<kind>\0<key>"`, lower-case hex.
pub fn repository_namespace(kind: NamespaceKind, key: &str) -> String {
    sha256_hex(format!("{}\0{key}", kind.word()).as_bytes())
}

/// `repr()` of the marker tuple: `(a, b, …)`.
pub fn marker_repr(marker: &[String]) -> String {
    format!("({})", marker.join(", "))
}

/// `_repository_context_marker_is_strong`: the top level's file id, `.git`'s
/// file id, or `.git`'s pointer digest is known (not -1 or 0).
pub fn marker_is_strong(marker: &[String]) -> bool {
    [1usize, 6, 9]
        .iter()
        .filter_map(|at| marker.get(*at))
        .any(|value| value != "-1" && value != "0")
}

/// `_known_repository_context`: SHA-256 of the marker's `repr`, or, for a
/// weak marker, of `"<normcase'd top level>\0<repr>"`.
pub fn checkout_context_id(normalized_root: &str, marker: &[String]) -> String {
    let repr = marker_repr(marker);
    if marker_is_strong(marker) {
        sha256_hex(repr.as_bytes())
    } else {
        sha256_hex(format!("{normalized_root}\0{repr}").as_bytes())
    }
}

/// `repository_context._checkout_namespace`.
pub fn checkout_namespace(repository_namespace: &str, checkout_context_id: &str) -> String {
    sha256_hex(
        format!("{CHECKOUT_NAMESPACE_SCHEMA}\0{repository_namespace}\0{checkout_context_id}")
            .as_bytes(),
    )
}

/// The lease below the state home, with forward slashes:
/// `fleet_repositories/<repository>/checkouts/<checkout>/agent-session.lock`.
pub fn lease_relative(repository_namespace: &str, checkout_namespace: &str) -> String {
    format!(
        "fleet_repositories/{repository_namespace}/checkouts/{checkout_namespace}/{CHECKOUT_LEASE_FILE}"
    )
}

/// `RepositoryStatePaths.derive(..).checkout_file(CHECKOUT_LEASE_FILE)`: the
/// state home tidied lexically (`os.path.normpath`), then the lease.
pub fn lease_path(
    state_home: &Path,
    repository_namespace: &str,
    checkout_namespace: &str,
) -> PathBuf {
    lease_relative(repository_namespace, checkout_namespace)
        .split('/')
        .fold(tidy(state_home), |path, part| path.join(part))
}

// ---------------------------------------------------------- identity reading

/// `os.path.normcase(str(path))` on Windows for a resolved path: no verbatim
/// prefix, backslashes, lower case. (Python lower-cases with the invariant
/// locale's table; this uses Unicode's, which differ only outside ASCII.)
pub fn normcase(path: &Path) -> String {
    shown_path(path).replace('/', "\\").to_lowercase()
}

/// A FILETIME (100 ns since 1601) as nanoseconds since 1970, as Python's
/// `st_birthtime_ns`.
fn birth_ns(creation: i64) -> i128 {
    (i128::from(creation) - 116_444_736_000_000_000) * 100
}

/// Python's `st_mode` on Windows (`attributes_to_mode`, then `S_IFLNK` for
/// a symbolic link read without following it).
fn st_mode(attributes: u32, symlink: bool) -> u32 {
    const DIRECTORY: u32 = 0x10;
    const READONLY: u32 = 0x1;
    let mut mode = if attributes & DIRECTORY != 0 {
        0o040_000 | 0o111
    } else {
        0o100_000
    };
    mode |= if attributes & READONLY != 0 {
        0o444
    } else {
        0o666
    };
    if symlink {
        mode = (mode & !0o170_000) | 0o120_000;
    }
    mode
}

/// A 32-byte big-endian number in decimal (`int.from_bytes(digest, "big")`).
fn big_decimal(bytes: &[u8]) -> String {
    let mut number: Vec<u8> = bytes.to_vec();
    let mut digits: Vec<u8> = Vec::new();
    while number.iter().any(|byte| *byte != 0) {
        let mut remainder = 0u32;
        for byte in &mut number {
            let value = (remainder << 8) | u32::from(*byte);
            *byte = (value / 10) as u8;
            remainder = value % 10;
        }
        digits.push(b'0' + remainder as u8);
    }
    if digits.is_empty() {
        return "0".to_owned();
    }
    digits.reverse();
    String::from_utf8(digits).unwrap_or_default()
}

/// `_root_incarnation_marker` of a resolved top level: for the top level and
/// its `.git`, `lstat`'s device, file id, mode and birth time, and for a
/// `.git` that is a regular file of at most 4,096 bytes the SHA-256 of its
/// bytes as an integer (else -1); five `-1`s for an entry that cannot be
/// read.
pub fn read_marker(top: &Path) -> Vec<String> {
    let mut values = Vec::with_capacity(10);
    for (entry, is_git) in [(top.to_path_buf(), false), (top.join(".git"), true)] {
        let opened = lattice_sys::fs::open_no_follow(&entry, Access::Read)
            .or_else(|_| lattice_sys::fs::open_no_follow(&entry, Access::Attributes));
        let read = opened.and_then(|opened| {
            let identity = file_identity(&opened.file)?;
            let times = file_times(&opened.file)?;
            let attributes = file_attributes(&opened.file)?;
            Ok((opened, identity, times, attributes))
        });
        let Ok((opened, identity, times, attributes)) = read else {
            values.extend(std::iter::repeat_n("-1".to_owned(), 5));
            continue;
        };
        let symlink = opened
            .link
            .as_ref()
            .is_some_and(|link| link.kind == LinkKind::Symlink);
        let mode = st_mode(attributes, symlink);
        let birth = birth_ns(times.creation);
        values.push(identity.volume_serial.to_string());
        values.push(u128::from_le_bytes(identity.file_id).to_string());
        values.push(mode.to_string());
        values.push(if birth == 0 {
            "-1".to_owned()
        } else {
            birth.to_string()
        });
        let regular = mode & 0o170_000 == 0o100_000;
        let digest = if is_git && regular {
            pointer_digest(opened.file)
        } else {
            None
        };
        values.push(digest.unwrap_or_else(|| "-1".to_owned()));
    }
    values
}

/// `_bounded_identity_digest`: at most 4,096 bytes, else none.
fn pointer_digest(file: File) -> Option<String> {
    let size = file.metadata().ok()?.len();
    if size > GIT_POINTER_IDENTITY_BYTES {
        return None;
    }
    let mut payload = Vec::new();
    file.take(GIT_POINTER_IDENTITY_BYTES + 1)
        .read_to_end(&mut payload)
        .ok()?;
    if payload.len() as u64 > GIT_POINTER_IDENTITY_BYTES {
        return None;
    }
    let digest = sha256_hex(&payload);
    let bytes: Vec<u8> = (0..32)
        .map(|at| u8::from_str_radix(&digest[2 * at..2 * at + 2], 16).unwrap_or(0))
        .collect();
    Some(big_decimal(&bytes))
}

/// `Path.resolve()` of a folder: its final path, without the verbatim
/// prefix.
fn resolved(path: &Path) -> Result<PathBuf, LeaseError> {
    let walked = open_walk(path, Access::Attributes, LinkRule::AnyLocal)
        .map_err(|_| LeaseError::Unavailable("a folder could not be opened"))?;
    Ok(PathBuf::from(shown_path(&walked.final_path)))
}

/// The common folder's identity, read as `_namespace_metadata` (`stat()`,
/// following links).
fn read_incarnation(common: &Path) -> Option<Incarnation> {
    let walked = open_walk(common, Access::Attributes, LinkRule::AnyLocal).ok()?;
    let identity = file_identity(&walked.file).ok()?;
    let times = file_times(&walked.file).ok()?;
    Some(Incarnation {
        device: identity.volume_serial.to_string(),
        inode: u128::from_le_bytes(identity.file_id).to_string(),
        birth_ns: birth_ns(times.creation).to_string(),
    })
}

/// A checkout's lease, as Python derives it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckoutLease {
    /// Git's top level, resolved.
    pub top: PathBuf,
    /// Git's common folder, resolved.
    pub common: PathBuf,
    pub repository_namespace: String,
    pub checkout_namespace: String,
    pub path: PathBuf,
}

/// The lease of the checkout `workspace` is in, below `state_home` (§6.5):
/// from git's top level (never the attached folder) and common folder.
pub fn derive_checkout_lease(
    workspace: &Workspace,
    runner: &GitRunner,
    state_home: &Path,
) -> Result<CheckoutLease, LeaseError> {
    let Repo::Git(local) = &workspace.repo else {
        return Err(LeaseError::Unavailable(
            "the folder is not in a git repository",
        ));
    };
    let top = resolved(&runner.top_level(local)?)?;
    let top_repo = match dotgit::inspect(&top, runner.env()) {
        Repo::Git(top_repo) => top_repo,
        _ => return Err(LeaseError::NotLocal),
    };
    let out = runner.run(
        &top_repo,
        &[
            OsStr::new("rev-parse"),
            OsStr::new("--path-format=absolute"),
            OsStr::new("--git-common-dir"),
        ],
        &Extra::default(),
    )?;
    if out.status != 0 {
        return Err(GitError::Failed {
            status: out.status,
            stderr: out.stderr,
        }
        .into());
    }
    let text = String::from_utf8(out.stdout).map_err(|_| GitError::NotText)?;
    let common = resolved(Path::new(
        &text.trim_end_matches(['\r', '\n']).replace('/', "\\"),
    ))?;
    let marker = read_marker(&top);
    if !marker_is_strong(&marker) {
        return Err(LeaseError::WeakIdentity);
    }
    let key = namespace_key(read_incarnation(&common).as_ref(), &normcase(&common));
    let repository = repository_namespace(NamespaceKind::GitCommonDir, &key);
    let context = checkout_context_id(&normcase(&top), &marker);
    let checkout = checkout_namespace(&repository, &context);
    let path = lease_path(state_home, &repository, &checkout);
    Ok(CheckoutLease {
        top,
        common,
        repository_namespace: repository,
        checkout_namespace: checkout,
        path,
    })
}

/// The lease file for `workspace`: its checkout's (git), or the native one
/// (no git).
pub fn lease_file_for(
    workspace: &Workspace,
    runner: &GitRunner,
    state: &StateRoot,
    platform: Platform,
) -> Result<PathBuf, LeaseError> {
    match &workspace.repo {
        Repo::Git(_) => {
            let home = selected_repository_state_root(runner.env(), state, platform);
            Ok(derive_checkout_lease(workspace, runner, &home)?.path)
        }
        Repo::None | Repo::Without(_) => Ok(state
            .native_chat_dir()
            .join("leases")
            .join(format!("{}.lock", workspace.id))),
    }
}

// ------------------------------------------------------------------ the lock

/// A held lease: the lock lives on this handle, and goes with it.
#[derive(Debug)]
pub struct HeldLease {
    file: Option<File>,
    path: PathBuf,
}

impl HeldLease {
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Release: unlock, then close. The file stays (it is never deleted).
    pub fn release(mut self) {
        self.give_up();
    }

    fn give_up(&mut self) {
        if let Some(file) = self.file.take() {
            let _ = file.unlock();
        }
    }
}

impl Drop for HeldLease {
    fn drop(&mut self) {
        self.give_up();
    }
}

/// Open the lease file without following a link: create it new, or open the
/// existing one only when it is a regular file that is not a reparse point.
fn open_direct_regular(path: &Path) -> Result<File, LeaseError> {
    match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(file) => return Ok(file),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) if error.kind() == io::ErrorKind::PermissionDenied => {}
        Err(error) => return Err(LeaseError::Io(error.to_string())),
    }
    let opened = lattice_sys::fs::open_no_follow(&verbatim(path), Access::Read)
        .map_err(|_| LeaseError::Unsafe("the lease path cannot be opened without redirection"))?;
    if opened.link.is_some() {
        return Err(LeaseError::Unsafe(
            "the lease path is a link or a reparse point",
        ));
    }
    if opened.is_dir {
        return Err(LeaseError::Unsafe("the lease path must be a regular file"));
    }
    let identity = file_identity(&opened.file)
        .map_err(|_| LeaseError::Unsafe("the lease path has no stable identity"))?;
    if identity.file_id == [0; 16] {
        return Err(LeaseError::Unsafe("the lease path has no stable identity"));
    }
    Ok(opened.file)
}

/// `path` in verbatim form (`\\?\C:\…`) when it is a drive path: the lease
/// sits five folders below the state home, past `MAX_PATH` in a deep home,
/// and only a verbatim path opens there without a long-path manifest (std's
/// own opens add the prefix themselves).
fn verbatim(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    let bytes = text.as_bytes();
    if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' && bytes[2] == b'\\' {
        PathBuf::from(format!(r"\\?\{text}"))
    } else {
        path.to_path_buf()
    }
}

/// `process_lease.try_acquire`: try once. `Ok(None)` when another holder has
/// it; `Err` when the path is unsafe or the lock call failed.
pub fn try_acquire(path: &Path) -> Result<Option<HeldLease>, LeaseError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|_| LeaseError::Unsafe("the lease's folder is unavailable"))?;
    }
    let file = open_direct_regular(path)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(HeldLease {
            file: Some(file),
            path: path.to_path_buf(),
        })),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(error)) => Err(LeaseError::Io(error.to_string())),
    }
}

/// One conversation's hold on its folder's lease (see the module header).
#[derive(Debug)]
pub struct WriterLease {
    path: Result<PathBuf, LeaseError>,
    held: Mutex<Option<HeldLease>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl WriterLease {
    /// The lease of `workspace`'s checkout (or its native one), derived now
    /// and not taken.
    pub fn for_workspace(
        workspace: &Workspace,
        runner: &GitRunner,
        state: &StateRoot,
        platform: Platform,
    ) -> Self {
        Self::at(lease_file_for(workspace, runner, state, platform))
    }

    /// A lease at a path already derived (or the reason there is none).
    pub fn at(path: Result<PathBuf, LeaseError>) -> Self {
        Self {
            path,
            held: Mutex::new(None),
        }
    }

    /// The lease file, or why there is none.
    pub fn path(&self) -> Result<&Path, &LeaseError> {
        self.path.as_deref()
    }

    /// Whether this conversation holds it.
    pub fn is_held(&self) -> bool {
        lock(&self.held).is_some()
    }

    /// Take it if it is free: `Held`, or `Elsewhere` when another holder has
    /// it. A lease that cannot be derived or taken safely counts as held
    /// elsewhere: nothing is written without it (fail closed).
    pub fn take(&self) -> Lease {
        let mut held = lock(&self.held);
        if held.is_some() {
            return Lease::Held;
        }
        let Ok(path) = &self.path else {
            return Lease::Elsewhere;
        };
        match try_acquire(path) {
            Ok(Some(lease)) => {
                *held = Some(lease);
                Lease::Held
            }
            Ok(None) | Err(_) => Lease::Elsewhere,
        }
    }

    /// Give the lease back once nothing needs it: no change still waiting
    /// for review and no command running.
    pub fn release_when_idle(&self, waiting: u32, command_running: bool) -> bool {
        if waiting > 0 || command_running {
            return false;
        }
        lock(&self.held).take().is_some()
    }
}
