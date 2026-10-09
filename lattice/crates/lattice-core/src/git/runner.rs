//! The one place git runs (the chat core's spec §8.2, "The git runner"),
//! over B7's spawn and B8's resolver. Not a port: the web's `gitio.py` strips
//! inherited `GIT_*` names and keeps the rest of its environment; this runner
//! passes only X7's names and its own.
//!
//! Every call is fixed in three ways, because a folder's own `.git/config`
//! (hooks, an fsmonitor command, clean filters, protocol settings) is whatever
//! the folder says:
//! - **The program** is `git.exe` resolved by X2's rules (`exec::resolve`):
//!   never from inside the folder or Lattice's state, never a batch file.
//! - **The argv** starts with `-c core.hooksPath=<native>/chat/no-hooks` (an
//!   empty folder Lattice owns), `-c core.fsmonitor=false`,
//!   `-c core.untrackedCache=false` and `-c protocol.allow=never`, then
//!   `--no-pager` and `-c safe.bareRepository=explicit` (a native addition:
//!   git refuses to treat a folder as a bare repository by itself).
//! - **The environment** is X7's block (with its `PATH` filter and
//!   `NoDefaultCurrentDirectoryInExePath`) plus `GIT_TERMINAL_PROMPT=0`,
//!   `GIT_OPTIONAL_LOCKS=0`, `GIT_NO_LAZY_FETCH=1` and `GIT_ALLOW_PROTOCOL=none`:
//!   a repository's own `protocol.<scheme>.allow=always` overrides the `-c`
//!   flag but not the variable, and in a partial clone a read of a missing
//!   object starts a `git fetch` unless lazy fetching is off. `GIT_INDEX_FILE`
//!   and a fixed identity (`Lattice`, `lattice@localhost`) are added only when
//!   a call asks for them.
//!
//! No shell, no pager, no external diff and no textconv. A call runs in a
//! folder only with a [`LocalRepo`], which only FT6's native reader
//! (`dotgit::inspect`) makes, so no git process starts in a folder whose `.git`
//! names the network. A git older than [`MIN_VERSION`] silently ignores
//! `GIT_NO_LAZY_FETCH`, so it is refused (R10).
//!
//! There is no time limit (as in `gitio.py`: a slow git on a busy workstation
//! is working, and nothing here can wait on a person or a remote). Output is
//! bounded: a call that writes more than [`MAX_STDOUT`] bytes is ended.
//! Callers run it on the blocking pool (CR6).

use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use lattice_sys::fs::Access;
use lattice_sys::process::{JobLimits, SpawnRequest, spawn};

use super::dotgit::LocalRepo;
use crate::env::Env;
use crate::exec::resolve::{Ineligible, resolve_program};
use crate::exec::spawn::child_environment;
use crate::localfs::{LinkRule, WalkError, open_walk};
use crate::state::StateRoot;

/// The first release whose notes name lazy fetching as something a user can
/// switch off (`git --no-lazy-fetch`, the same as `GIT_NO_LAZY_FETCH=1`;
/// RelNotes 2.45.0). An older git ignores the variable.
pub const MIN_VERSION: GitVersion = GitVersion {
    major: 2,
    minor: 45,
    patch: 0,
};
/// The most paths one listing returns (the web service's `MAX_TREE_ENTRIES`).
pub const MAX_TREE_ENTRIES: usize = 100_000;
/// The most bytes read from a call's standard output.
pub const MAX_STDOUT: u64 = 64 * 1024 * 1024;
/// The most bytes of standard error kept (the rest is read and dropped).
pub const MAX_STDERR: usize = 64 * 1024;

/// The settings every call overrides, after `core.hooksPath` (§8.2).
pub const FIXED_OVERRIDES: [&str; 3] = [
    "core.fsmonitor=false",
    "core.untrackedCache=false",
    "protocol.allow=never",
];
/// The native addition, after `--no-pager`.
pub const BARE_OVERRIDE: &str = "safe.bareRepository=explicit";
/// The variables every call adds to X7's block (§8.2).
pub const GIT_VARIABLES: [(&str, &str); 4] = [
    ("GIT_TERMINAL_PROMPT", "0"),
    ("GIT_OPTIONAL_LOCKS", "0"),
    ("GIT_NO_LAZY_FETCH", "1"),
    ("GIT_ALLOW_PROTOCOL", "none"),
];
/// The fixed identity of a checkpoint's commit.
pub const IDENTITY: [(&str, &str); 4] = [
    ("GIT_AUTHOR_NAME", "Lattice"),
    ("GIT_AUTHOR_EMAIL", "lattice@localhost"),
    ("GIT_COMMITTER_NAME", "Lattice"),
    ("GIT_COMMITTER_EMAIL", "lattice@localhost"),
];

/// A git version, `major.minor.patch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GitVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl GitVersion {
    /// `git version 2.54.0.windows.1` (any suffix after the third number).
    pub fn parse(text: &str) -> Option<Self> {
        let rest = text.trim().strip_prefix("git version ")?;
        let mut numbers = rest.split(['.', ' ', '-']).map(str::parse::<u32>);
        Some(Self {
            major: numbers.next()?.ok()?,
            minor: numbers.next()?.ok()?,
            patch: numbers.next().and_then(Result::ok).unwrap_or(0),
        })
    }
}

impl std::fmt::Display for GitVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Why a git call did not give an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GitError {
    /// No usable `git.exe` (X2).
    Unavailable(Ineligible),
    /// The empty hooks folder could not be made, or is not empty.
    HooksFolder,
    /// The process could not be started or read.
    Spawn(String),
    /// `git version` said something else.
    BadVersion,
    /// Older than [`MIN_VERSION`].
    TooOld(GitVersion),
    /// A non-zero exit where only zero means an answer.
    Failed { status: u32, stderr: String },
    /// `check-ignore` exited with neither 0 nor 1.
    CouldNotSay { status: u32 },
    /// More than [`MAX_STDOUT`] bytes; the call was ended.
    TooMuchOutput,
    /// Output that is not UTF-8 where a path or a version was expected.
    NotText,
}

impl GitError {
    /// One sentence; never git's own output.
    pub fn sentence(&self) -> String {
        match self {
            Self::Unavailable(_) => {
                "git was not found outside this folder, so Lattice cannot ask it.".to_owned()
            }
            Self::HooksFolder => {
                "Lattice could not prepare its empty hooks folder for git.".to_owned()
            }
            Self::Spawn(_) => "git could not be started.".to_owned(),
            Self::BadVersion => "git did not say which version it is.".to_owned(),
            Self::TooOld(version) => format!(
                "git {version} is older than {MIN_VERSION}, which Lattice needs so that git never fetches on its own."
            ),
            Self::Failed { .. } => "git could not answer.".to_owned(),
            Self::CouldNotSay { .. } => {
                "git could not say whether that file is ignored, so it is not shown.".to_owned()
            }
            Self::TooMuchOutput => "git's answer was too long.".to_owned(),
            Self::NotText => "git's answer was not text.".to_owned(),
        }
    }
}

/// What a call asks for beyond the fixed environment.
#[derive(Clone, Debug, Default)]
pub struct Extra {
    /// `GIT_INDEX_FILE`: a private index.
    pub index_file: Option<PathBuf>,
    /// The fixed identity, for `commit-tree`.
    pub identity: bool,
}

/// A finished call.
#[derive(Clone, Debug)]
pub struct Output {
    pub status: u32,
    pub stdout: Vec<u8>,
    /// The first [`MAX_STDERR`] bytes, as text. Never shown to the reader
    /// or the model.
    pub stderr: String,
}

/// One record of `git ls-files -z -t -s`: `<tag> <mode> <object> <stage>\t<path>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct IndexEntry {
    /// The `S` tag: the entry is skip-worktree.
    pub(crate) skip_worktree: bool,
    pub(crate) mode: String,
    pub(crate) object: String,
    /// Relative to the top level, forward slashes.
    pub(crate) path: String,
}

impl IndexEntry {
    pub(crate) fn parse(record: &[u8]) -> Option<Self> {
        let text = std::str::from_utf8(record).ok()?;
        let (head, path) = text.split_once('\t')?;
        let mut fields = head.split(' ');
        let tag = fields.next()?;
        let mode = fields.next()?;
        let object = fields.next()?;
        fields.next()?;
        if path.is_empty() || object.is_empty() || !object.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        Some(Self {
            skip_worktree: tag == "S",
            mode: mode.to_owned(),
            object: object.to_owned(),
            path: path.to_owned(),
        })
    }
}

/// Folders relative to the top level, as folders relative to `prefix` (the
/// workspace folder below the top level, no trailing slash): one at or above
/// the prefix is the whole folder (`""`); one elsewhere does not concern it.
/// Sorted, each once, none below another.
pub(crate) fn below_prefix(prefix: &str, folders: Vec<String>) -> Vec<String> {
    let inside = |folder: &str, of: &str| {
        of.is_empty() || folder == of || folder.starts_with(&format!("{of}/"))
    };
    let mut out: Vec<String> = Vec::new();
    for folder in folders {
        let mapped = if inside(prefix, &folder) {
            String::new()
        } else if inside(&folder, prefix) {
            folder[prefix.len()..].trim_start_matches('/').to_owned()
        } else {
            continue;
        };
        out.push(mapped);
    }
    out.sort();
    out.dedup();
    let all = out.clone();
    out.retain(|folder| {
        !all.iter()
            .any(|other| other != folder && inside(folder, other))
    });
    out
}

/// The paths `git ls-files` listed, in its order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Listing {
    pub paths: Vec<String>,
    /// More than [`MAX_TREE_ENTRIES`] were listed.
    pub truncated: bool,
    /// Paths that were not UTF-8, left out.
    pub not_text: usize,
}

/// `ls-files -z` output as paths, at most `cap` of them.
pub(crate) fn parse_listing(stdout: &[u8], cap: usize) -> Listing {
    let mut listing = Listing::default();
    for raw in stdout.split(|b| *b == 0).filter(|raw| !raw.is_empty()) {
        if listing.paths.len() == cap {
            listing.truncated = true;
            break;
        }
        match std::str::from_utf8(raw) {
            Ok(path) => listing.paths.push(path.to_owned()),
            Err(_) => listing.not_text += 1,
        }
    }
    listing
}

/// Runs git as the module header says.
pub struct GitRunner {
    env: Arc<dyn Env>,
    globals: PathBuf,
    no_hooks: PathBuf,
    version: OnceLock<Result<GitVersion, GitError>>,
    /// `GIT_TRACE` for KF8's falsifier, in tests only.
    #[cfg(test)]
    pub(crate) trace: Option<PathBuf>,
}

/// A path as a process's working folder: a verbatim drive path loses its
/// `\\?\`, because some programs refuse a verbatim working folder.
fn plain_dir(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest)
            if rest.len() >= 3
                && rest.as_bytes()[0].is_ascii_alphabetic()
                && rest.as_bytes()[1] == b':' =>
        {
            PathBuf::from(rest)
        }
        _ => path.to_path_buf(),
    }
}

impl GitRunner {
    /// A runner reading Lattice's environment through `env`, with its empty
    /// hooks folder at `<globals>/lattice_native/chat/no-hooks`.
    pub fn new(env: Arc<dyn Env>, state: &StateRoot) -> Self {
        Self {
            env,
            globals: state.globals.clone(),
            no_hooks: state.no_hooks_dir(),
            version: OnceLock::new(),
            #[cfg(test)]
            trace: None,
        }
    }

    /// The environment this runner reads Lattice's own from.
    pub fn env(&self) -> &dyn Env {
        self.env.as_ref()
    }

    /// The empty folder `core.hooksPath` names.
    pub fn no_hooks_dir(&self) -> &Path {
        &self.no_hooks
    }

    /// The argv of a call: the program's name, the overrides, then `args`.
    pub fn argv(&self, program: &Path, args: &[OsString]) -> Vec<OsString> {
        let mut argv: Vec<OsString> = vec![program.as_os_str().to_owned()];
        let mut hooks = OsString::from("core.hooksPath=");
        hooks.push(plain_dir(&self.no_hooks).as_os_str());
        argv.push("-c".into());
        argv.push(hooks);
        for setting in FIXED_OVERRIDES {
            argv.push("-c".into());
            argv.push(setting.into());
        }
        argv.push("--no-pager".into());
        argv.push("-c".into());
        argv.push(BARE_OVERRIDE.into());
        argv.extend(args.iter().cloned());
        argv
    }

    /// The environment of a call (see the module header).
    pub fn environment(
        &self,
        workspace: Option<&Path>,
        extra: &Extra,
    ) -> Vec<(OsString, OsString)> {
        let mut block = child_environment(self.env.as_ref(), workspace, &self.globals);
        for (name, value) in GIT_VARIABLES {
            block.push((name.into(), value.into()));
        }
        if let Some(index) = &extra.index_file {
            block.push(("GIT_INDEX_FILE".into(), index.as_os_str().to_owned()));
        }
        if extra.identity {
            for (name, value) in IDENTITY {
                block.push((name.into(), value.into()));
            }
        }
        #[cfg(test)]
        if let Some(trace) = &self.trace {
            block.push(("GIT_TRACE".into(), trace.as_os_str().to_owned()));
        }
        block
    }

    fn hooks_ready(&self) -> Result<(), GitError> {
        std::fs::create_dir_all(&self.no_hooks).map_err(|_| GitError::HooksFolder)?;
        let mut entries = std::fs::read_dir(&self.no_hooks).map_err(|_| GitError::HooksFolder)?;
        if entries.next().is_some() {
            return Err(GitError::HooksFolder);
        }
        Ok(())
    }

    /// Start git in `cwd` and wait for it.
    fn raw(
        &self,
        cwd: &Path,
        workspace: Option<&Path>,
        args: &[OsString],
        extra: &Extra,
    ) -> Result<Output, GitError> {
        self.hooks_ready()?;
        let program = resolve_program("git", self.env.as_ref(), workspace, &self.globals)
            .map_err(GitError::Unavailable)?
            .path;
        let argv = self.argv(&program, args);
        let env = self.environment(workspace, extra);
        let cwd = plain_dir(cwd);
        let mut child = spawn(&SpawnRequest {
            program: &program,
            argv: &argv,
            cwd: &cwd,
            env: &env,
            limits: JobLimits::default(),
        })
        .map_err(|error| GitError::Spawn(error.kind().to_string()))?;
        let stderr_pipe = child.take_stderr();
        let stderr_reader = std::thread::spawn(move || {
            let mut kept = Vec::new();
            if let Some(mut pipe) = stderr_pipe {
                let mut buffer = [0u8; 8192];
                while let Ok(read) = pipe.read(&mut buffer) {
                    if read == 0 {
                        break;
                    }
                    let room = MAX_STDERR.saturating_sub(kept.len());
                    kept.extend_from_slice(&buffer[..read.min(room)]);
                }
            }
            kept
        });
        let mut stdout = Vec::new();
        let mut too_much = false;
        if let Some(pipe) = child.take_stdout() {
            let read = pipe.take(MAX_STDOUT + 1).read_to_end(&mut stdout);
            if read.is_err() {
                let _ = child.kill_tree();
                let _ = stderr_reader.join();
                return Err(GitError::Spawn("its output could not be read".into()));
            }
            if stdout.len() as u64 > MAX_STDOUT {
                too_much = true;
                let _ = child.kill_tree();
            }
        }
        let status = child
            .wait(None)
            .map_err(|error| GitError::Spawn(error.kind().to_string()))?
            .unwrap_or(u32::MAX);
        let stderr = stderr_reader.join().unwrap_or_default();
        if too_much {
            return Err(GitError::TooMuchOutput);
        }
        Ok(Output {
            status,
            stdout,
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        })
    }

    /// The installed git's version, asked once per runner (`git version`, in
    /// the empty hooks folder, which no repository reads).
    pub fn version(&self) -> Result<GitVersion, GitError> {
        self.version
            .get_or_init(|| {
                let output = self.raw(
                    &self.no_hooks.clone(),
                    None,
                    &[OsString::from("version")],
                    &Extra::default(),
                )?;
                if output.status != 0 {
                    return Err(GitError::BadVersion);
                }
                let text = String::from_utf8(output.stdout).map_err(|_| GitError::NotText)?;
                GitVersion::parse(&text).ok_or(GitError::BadVersion)
            })
            .clone()
    }

    /// Run `args` in `repo`'s folder, once the version is known to honour
    /// every variable.
    pub fn run(
        &self,
        repo: &LocalRepo,
        args: &[&OsStr],
        extra: &Extra,
    ) -> Result<Output, GitError> {
        let version = self.version()?;
        if version < MIN_VERSION {
            return Err(GitError::TooOld(version));
        }
        let args: Vec<OsString> = args.iter().map(|arg| (*arg).to_owned()).collect();
        self.raw(repo.folder(), Some(repo.folder()), &args, extra)
    }

    /// `git rev-parse --show-toplevel`: the repository's top level.
    pub fn top_level(&self, repo: &LocalRepo) -> Result<PathBuf, GitError> {
        let output = self.run(
            repo,
            &[OsStr::new("rev-parse"), OsStr::new("--show-toplevel")],
            &Extra::default(),
        )?;
        if output.status != 0 {
            return Err(GitError::Failed {
                status: output.status,
                stderr: output.stderr,
            });
        }
        let text = String::from_utf8(output.stdout).map_err(|_| GitError::NotText)?;
        Ok(PathBuf::from(
            text.trim_end_matches(['\r', '\n']).replace('/', "\\"),
        ))
    }

    /// The folder's non-ignored file set (§6.2 "Walks"): one
    /// `git ls-files -z --cached --others --exclude-standard`, relative to the
    /// folder, capped at [`MAX_TREE_ENTRIES`].
    pub fn ls_files(&self, repo: &LocalRepo) -> Result<Listing, GitError> {
        let output = self.run(
            repo,
            &[
                OsStr::new("ls-files"),
                OsStr::new("-z"),
                OsStr::new("--cached"),
                OsStr::new("--others"),
                OsStr::new("--exclude-standard"),
            ],
            &Extra::default(),
        )?;
        if output.status != 0 {
            return Err(GitError::Failed {
                status: output.status,
                stderr: output.stderr,
            });
        }
        Ok(parse_listing(&output.stdout, MAX_TREE_ENTRIES))
    }

    /// Take `version` as the installed git's, in tests only.
    #[cfg(test)]
    pub(crate) fn assume_version(&self, version: GitVersion) {
        let _ = self.version.set(Ok(version));
    }

    /// WP8: does git ignore `path` (relative to the folder, forward
    /// slashes)? `git check-ignore -q -- <path>`: exit 0 is yes, 1 is no, and
    /// anything else is [`GitError::CouldNotSay`], with `views.py`'s rule.
    pub fn check_ignore(&self, repo: &LocalRepo, path: &str) -> Result<bool, GitError> {
        let output = self.run(
            repo,
            &[
                OsStr::new("check-ignore"),
                OsStr::new("-q"),
                OsStr::new("--"),
                OsStr::new(path),
            ],
            &Extra::default(),
        )?;
        match output.status {
            0 => Ok(true),
            1 => Ok(false),
            status => Err(GitError::CouldNotSay { status }),
        }
    }

    /// The file `core.excludesFile` names, as git reads it (`git config
    /// --path --get`: `~` expanded; a relative path is the folder's), or
    /// `None` when it is not set or empty.
    pub fn excludes_file(&self, repo: &LocalRepo) -> Result<Option<PathBuf>, GitError> {
        let output = self.run(
            repo,
            &[
                OsStr::new("config"),
                OsStr::new("--path"),
                OsStr::new("--get"),
                OsStr::new("core.excludesFile"),
            ],
            &Extra::default(),
        )?;
        match output.status {
            0 => {}
            1 => return Ok(None),
            status => {
                return Err(GitError::Failed {
                    status,
                    stderr: output.stderr,
                });
            }
        }
        let text = String::from_utf8(output.stdout).map_err(|_| GitError::NotText)?;
        let text = text.trim_end_matches(['\r', '\n']);
        if text.is_empty() {
            return Ok(None);
        }
        let path = PathBuf::from(text.replace('/', "\\"));
        Ok(Some(if path.is_absolute() {
            path
        } else {
            repo.folder().join(path)
        }))
    }

    /// WP8b (spec §22.7): the folders whose ignore rules git cannot read
    /// here, relative to the folder (`""` is the whole folder), each once and
    /// none below another. In such a folder `check-ignore` (lazy fetching
    /// off) answers "not ignored" for what the missing rules would ignore.
    /// A folder is incomplete when:
    /// - `info/exclude` exists but cannot be read (the whole folder);
    /// - the file `core.excludesFile` names (`git config --path --get`)
    ///   exists but cannot be read (the whole folder);
    /// - a `.gitignore` git would consult (any in the index: `ls-files -t -s
    ///   --sparse`, from the top level) is not in the working tree, is
    ///   skip-worktree (so git reads its blob from the index), and its blob
    ///   is not in the object store (`cat-file -e`; nothing is fetched);
    /// - a `.gitignore` in the working tree cannot be read, or is a link
    ///   that leaves the repository or the machine;
    /// - a sparse directory entry stands for files not listed one by one.
    ///
    /// A folder above this one makes the whole folder incomplete.
    pub fn ignore_rules_incomplete(&self, repo: &LocalRepo) -> Result<Vec<String>, GitError> {
        let index = self.index_ignore_files(repo)?;
        self.incomplete_now(repo, &index)
    }

    /// The part of [`GitRunner::ignore_rules_incomplete`] that only the
    /// index and the configuration decide, which a caller may keep while
    /// [`IndexKey`] is unchanged: the folder's prefix below the top level,
    /// the top level, the index's `.gitignore` entries and sparse directory
    /// entries (`ls-files -t -s --sparse`), and the `core.excludesFile` path.
    pub fn index_ignore_files(&self, repo: &LocalRepo) -> Result<IndexIgnoreFiles, GitError> {
        let prefix_out = self.run(
            repo,
            &[OsStr::new("rev-parse"), OsStr::new("--show-prefix")],
            &Extra::default(),
        )?;
        if prefix_out.status != 0 {
            return Err(GitError::Failed {
                status: prefix_out.status,
                stderr: prefix_out.stderr,
            });
        }
        let prefix = String::from_utf8(prefix_out.stdout)
            .map_err(|_| GitError::NotText)?
            .trim_end_matches(['\r', '\n'])
            .trim_end_matches('/')
            .to_owned();
        let top = self.top_level(repo)?;
        let excludes_file = self.excludes_file(repo)?;
        let listing = self.run(
            repo,
            &[
                OsStr::new("ls-files"),
                OsStr::new("-z"),
                OsStr::new("-t"),
                OsStr::new("-s"),
                OsStr::new("--sparse"),
                OsStr::new("--full-name"),
                OsStr::new("--"),
                OsStr::new(":/"),
            ],
            &Extra::default(),
        )?;
        if listing.status != 0 {
            return Err(GitError::Failed {
                status: listing.status,
                stderr: listing.stderr,
            });
        }
        let mut sparse_dirs = Vec::new();
        let mut gitignores = Vec::new();
        for record in listing.stdout.split(|byte| *byte == 0) {
            let Some(entry) = IndexEntry::parse(record) else {
                continue;
            };
            if entry.mode == "040000" {
                sparse_dirs.push(entry.path.trim_end_matches('/').to_owned());
                continue;
            }
            let name = entry
                .path
                .rsplit_once('/')
                .map_or(entry.path.as_str(), |(_, name)| name);
            if name == ".gitignore" {
                gitignores.push(entry);
            }
        }
        Ok(IndexIgnoreFiles {
            prefix,
            top,
            excludes_file,
            sparse_dirs,
            gitignores,
            blobs: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        })
    }

    /// The rest of [`GitRunner::ignore_rules_incomplete`], read now: whether
    /// `info/exclude`, the `core.excludesFile` file and each index
    /// `.gitignore` in the working tree can be read. A skip-worktree
    /// `.gitignore` missing from the working tree is asked about with
    /// `cat-file -e` once per blob per `index` (an index whose blob is
    /// present keeps it: git does not prune what the index names; one found
    /// missing stays missing until the index changes, which fails closed).
    pub fn incomplete_now(
        &self,
        repo: &LocalRepo,
        index: &IndexIgnoreFiles,
    ) -> Result<Vec<String>, GitError> {
        // Folders, relative to the top level, whose rules cannot be read.
        let mut unreadable: Vec<String> = Vec::new();
        let exclude = repo.common_dir().join("info").join("exclude");
        match open_walk(&exclude, Access::Read, LinkRule::AnyLocal) {
            Ok(_) | Err(WalkError::NotFound) => {}
            Err(_) => unreadable.push(String::new()),
        }
        if let Some(excludes) = &index.excludes_file {
            match open_walk(excludes, Access::Read, LinkRule::AnyLocal) {
                Ok(_) | Err(WalkError::NotFound) => {}
                Err(_) => unreadable.push(String::new()),
            }
        }
        unreadable.extend(index.sparse_dirs.iter().cloned());
        for entry in &index.gitignores {
            let folder = entry.path.rsplit_once('/').map_or("", |(folder, _)| folder);
            let worktree = entry
                .path
                .split('/')
                .fold(index.top.clone(), |path, part| path.join(part));
            let readable = match open_walk(&worktree, Access::Read, LinkRule::Inside(&index.top)) {
                Ok(walked) => !walked.is_dir,
                Err(WalkError::NotFound) if entry.skip_worktree => {
                    self.blob_present(repo, index, &entry.object)?
                }
                // Removed from the working tree by the reader: git reads
                // nothing there, and neither does Lattice.
                Err(WalkError::NotFound) => true,
                Err(_) => false,
            };
            if !readable {
                unreadable.push(folder.to_owned());
            }
        }
        Ok(below_prefix(&index.prefix, unreadable))
    }

    /// `cat-file -e <object>` (nothing is fetched), remembered in `index`.
    fn blob_present(
        &self,
        repo: &LocalRepo,
        index: &IndexIgnoreFiles,
        object: &str,
    ) -> Result<bool, GitError> {
        let known = index
            .blobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(object)
            .copied();
        if let Some(present) = known {
            return Ok(present);
        }
        let present = self
            .run(
                repo,
                &[OsStr::new("cat-file"), OsStr::new("-e"), OsStr::new(object)],
                &Extra::default(),
            )?
            .status
            == 0;
        index
            .blobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(object.to_owned(), present);
        Ok(present)
    }
}

/// What [`GitRunner::index_ignore_files`] read from the index and the
/// configuration; [`GitRunner::incomplete_now`] reads the working tree
/// against it.
#[derive(Debug)]
pub struct IndexIgnoreFiles {
    /// The folder below the top level (no trailing slash; `""` at the top).
    prefix: String,
    top: PathBuf,
    excludes_file: Option<PathBuf>,
    /// Sparse directory entries, relative to the top level.
    sparse_dirs: Vec<String>,
    /// The index's `.gitignore` entries.
    gitignores: Vec<IndexEntry>,
    /// `cat-file -e` answers, by object id.
    blobs: std::sync::Mutex<std::collections::BTreeMap<String, bool>>,
}

/// What the index-derived part of WP8b depends on, read natively without
/// starting git: the index file's size, modification time and identity,
/// `HEAD`'s text and the reference it names (its file, or `packed-refs`),
/// and the repository's configuration files (`core.excludesFile` is read
/// from them). A change to any of them means the index part is read again.
/// A global or system configuration change is not seen until one of them
/// changes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexKey(Vec<(PathBuf, Option<FileStamp>)>, Vec<u8>);

/// A file's size, modification time and identity, or its absence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileStamp {
    len: u64,
    modified: Option<std::time::SystemTime>,
    identity: Option<lattice_sys::fs::FileIdentity>,
}

/// At most this much of `HEAD` is read.
const MAX_HEAD: u64 = 4096;

impl IndexKey {
    /// The key for `repo` now, or `None` when one of its files cannot be
    /// read (the caller then reads the index part again every time).
    pub fn read(repo: &LocalRepo) -> Option<Self> {
        let stamp = |path: &Path| -> Option<Option<FileStamp>> {
            match open_walk(path, Access::Attributes, LinkRule::AnyLocal) {
                Ok(walked) => {
                    let meta = walked.file.metadata().ok()?;
                    Some(Some(FileStamp {
                        len: meta.len(),
                        modified: meta.modified().ok(),
                        identity: lattice_sys::fs::file_identity(&walked.file).ok(),
                    }))
                }
                Err(WalkError::NotFound) => Some(None),
                Err(_) => None,
            }
        };
        let head_path = repo.git_dir().join("HEAD");
        let mut head = Vec::new();
        open_walk(&head_path, Access::Read, LinkRule::AnyLocal)
            .ok()?
            .file
            .take(MAX_HEAD)
            .read_to_end(&mut head)
            .ok()?;
        let mut files = vec![
            repo.git_dir().join("index"),
            head_path,
            repo.common_dir().join("packed-refs"),
            repo.common_dir().join("config"),
            repo.git_dir().join("config.worktree"),
        ];
        if let Some(reference) = std::str::from_utf8(&head)
            .ok()
            .and_then(|text| text.trim_end().strip_prefix("ref: "))
        {
            let parts: Vec<&str> = reference.split('/').collect();
            if parts
                .iter()
                .any(|part| part.is_empty() || *part == "." || *part == "..")
            {
                return None;
            }
            let at = |base: &Path| {
                parts
                    .iter()
                    .fold(base.to_path_buf(), |path, part| path.join(part))
            };
            files.push(at(repo.git_dir()));
            files.push(at(repo.common_dir()));
        }
        let stamps = files
            .into_iter()
            .map(|file| Some((file.clone(), stamp(&file)?)))
            .collect::<Option<Vec<_>>>()?;
        Some(Self(stamps, head))
    }
}
