//! The path rules a tool path must pass (the chat core's spec §6.2,
//! WP1–WP11), and its class (§9.1, ST4).
//!
//! WP1–WP4 are a port of the web's `workspaces/paths.py` `repo_path`, with its
//! refusal sentences, pinned by the parity golden `paths/repo_path.json`. The
//! rest is native:
//! - **WP7.** No component is a reserved device name (`CON`, `PRN`, `AUX`,
//!   `NUL`, `COM0`–`COM9`, `LPT0`–`LPT9`, the superscript-digit `COM¹`–`COM³`
//!   and `LPT¹`–`LPT³`, `CONIN$`, `CONOUT$`), with or without an extension.
//! - **WP11.** No component holds `~` followed by a digit: an 8.3 alias
//!   (`FAMENV~1.ENV`, `GITHUB~1`). "Use the file's full name."
//! - **WP10.** The path is opened from the root down, one component at a
//!   time, without following a link before its target is read
//!   (`localfs::open_walk` with the root as the bound): a link to anything
//!   outside the root, or to the network, is refused before it is followed.
//!   The path read back from the handle is the **derived path**: every
//!   component's long name, every link resolved. WP5 (inside the root, not the
//!   root), WP6 (not in `<root>/.git`) and WP9 (not Lattice's own state) are
//!   checked on it, and WP2–WP4, WP7, WP8 and the authority class (ST4) run on
//!   the derived path, never on the request. A file to be created has no
//!   handle yet: its parent folder goes through WP10, and the leaf, which
//!   passed WP2–WP4, WP7 and WP11 already, is appended.
//! - **WP8.** Not ignored, for reads as well as writes: in a git workspace
//!   `git check-ignore -q -- <derived path>` (0 refused, 1 allowed, anything
//!   else refused); without git, `.latticeignore` and the built-in defaults
//!   (`ignore.rs`).
//! - **WP8b.** In a git workspace, a folder whose ignore rules git cannot
//!   read (a skip-worktree `.gitignore` whose blob a partial clone lacks, an
//!   unreadable one, an unreadable `info/exclude`, a sparse directory entry;
//!   `GitRunner::ignore_rules_incomplete`) fails closed: every path at or
//!   below it is refused, for reads and for listings, with a sentence that
//!   names the folder. The blob is never fetched for the reader. Folders are
//!   compared without regard to case on Windows, as its file system names
//!   them (the index may say `sub/` where the disk says `SUB/`). On every
//!   read and every listed path, each folder from the root down to the
//!   target is also checked for a `.gitignore` that exists but cannot be
//!   opened for reading (an untracked one with a deny entry, say), which git
//!   skips with a warning: that folder fails closed the same way. Each folder
//!   is opened once per call (`IgnoreFileMemo`).
//!
//! Everything here happens before a byte of the file is read or written; a
//! read uses the handle the walk opened.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use lattice_sys::fs::Access;

use super::ignore::{IgnoreRules, Verdict};
use crate::git::dotgit::{LocalRepo, Repo};
use crate::git::runner::GitRunner;
use crate::localfs::{self, LinkRule, WalkError, is_inside, open_walk};
use crate::policy::{PathClass, PathRefusal};

/// `paths.MAX_PATH_CHARS`.
pub const MAX_PATH_CHARS: usize = 1024;

/// Why `repo_path` refuses a path (WP1–WP4), with Python's sentences.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RepoPathError {
    Required,
    TooLong,
    NotRelative,
    BadComponent,
    TrailingDotOrSpace,
    GitDir,
}

impl RepoPathError {
    /// `PathRefused`'s message, exactly.
    pub fn sentence(self) -> &'static str {
        match self {
            Self::Required => "a path is required",
            Self::TooLong => "the path is too long",
            Self::NotRelative => "only a relative path with forward slashes is accepted",
            Self::BadComponent => "the path has an empty, '.' or '..' component",
            Self::TrailingDotOrSpace => "a path component ends with a dot or a space",
            Self::GitDir => "git's own directory is not readable here",
        }
    }
}

/// `_REFUSED_CHARS`: control characters, a backslash, and the characters
/// Windows reserves in names.
fn is_refused_char(c: char) -> bool {
    (c as u32) < 0x20 || matches!(c, '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
}

/// WP1–WP4: `paths.repo_path` (see the module header).
pub fn repo_path(value: &str) -> Result<&str, RepoPathError> {
    if value.is_empty() {
        return Err(RepoPathError::Required);
    }
    if value.chars().count() > MAX_PATH_CHARS {
        return Err(RepoPathError::TooLong);
    }
    if value.starts_with('/') || value.chars().any(is_refused_char) {
        return Err(RepoPathError::NotRelative);
    }
    for part in value.split('/') {
        if matches!(part, "" | "." | "..") {
            return Err(RepoPathError::BadComponent);
        }
        if part != part.trim_end_matches(['.', ' ']) {
            return Err(RepoPathError::TrailingDotOrSpace);
        }
        if part.to_lowercase() == ".git" {
            return Err(RepoPathError::GitDir);
        }
    }
    Ok(value)
}

/// WP7: a reserved device name, with or without an extension (the part
/// before the first dot, trailing spaces removed, compared without case).
pub fn is_device_name(component: &str) -> bool {
    let base = component
        .split('.')
        .next()
        .unwrap_or("")
        .trim_end_matches(' ');
    let upper = base.to_uppercase();
    if matches!(
        upper.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) {
        return true;
    }
    let mut chars = upper.chars();
    let head: String = chars.by_ref().take(3).collect();
    let tail: Vec<char> = chars.collect();
    (head == "COM" || head == "LPT")
        && tail.len() == 1
        && (tail[0].is_ascii_digit() || matches!(tail[0], '\u{b9}' | '\u{b2}' | '\u{b3}'))
}

/// WP11: an 8.3 alias (`~` followed by a digit).
pub fn is_short_name_alias(component: &str) -> bool {
    let chars: Vec<char> = component.chars().collect();
    chars
        .windows(2)
        .any(|pair| pair[0] == '~' && pair[1].is_ascii_digit())
}

/// The folders whose files change what Lattice, git, a package manager or a
/// standing command does (ST4), at any depth.
const AUTHORITY_FOLDERS: [&str; 5] = [".lattice", ".cursor", ".github", ".vscode", ".cargo"];
/// Authority files by exact name, at any depth.
const AUTHORITY_NAMES: [&str; 19] = [
    "agents.md",
    "claude.md",
    // Claude Code's project MCP file, which Lattice reads in a trusted folder
    // (§12): its servers start programs, so a change to it is kept alone.
    ".mcp.json",
    ".gitignore",
    ".gitattributes",
    ".latticeignore",
    "cargo.toml",
    "cargo.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "pyproject.toml",
    "poetry.lock",
    "go.mod",
    "go.sum",
    ".npmrc",
    "nuget.config",
    "global.json",
];
/// Authority files by the start of their name.
const AUTHORITY_PREFIXES: [&str; 3] = ["rust-toolchain", ".yarnrc", "directory.build."];
/// Authority files by extension (scripts a person or a tool runs).
const AUTHORITY_EXTENSIONS: [&str; 4] = [".ps1", ".bat", ".cmd", ".sh"];

/// ST4: is the derived path an authority file? Compared without case, as
/// Windows names are.
pub fn is_authority(derived: &str) -> bool {
    let parts: Vec<String> = derived.split('/').map(str::to_lowercase).collect();
    let Some((name, folders)) = parts.split_last() else {
        return false;
    };
    if folders
        .iter()
        .chain(std::iter::once(name))
        .any(|part| AUTHORITY_FOLDERS.contains(&part.as_str()))
    {
        return true;
    }
    AUTHORITY_NAMES.contains(&name.as_str())
        || (name.starts_with("requirements") && name.ends_with(".txt"))
        || AUTHORITY_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
        || AUTHORITY_EXTENSIONS
            .iter()
            .any(|extension| name.ends_with(extension))
}

/// A refused path: its class for the policy, and the one sentence the tool
/// answers with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refused {
    pub reason: PathRefusal,
    pub sentence: String,
}

impl Refused {
    fn because(reason: PathRefusal) -> Self {
        Self {
            reason,
            sentence: reason.sentence().to_owned(),
        }
    }

    fn repo_path(value: &str, error: RepoPathError) -> Self {
        let reason = match error {
            RepoPathError::TooLong => PathRefusal::TooLong,
            RepoPathError::TrailingDotOrSpace => PathRefusal::TrailingDotOrSpace,
            RepoPathError::GitDir => PathRefusal::GitDir,
            RepoPathError::NotRelative if value.starts_with("//") || value.starts_with(r"\\") => {
                PathRefusal::Unc
            }
            RepoPathError::NotRelative if is_drive_absolute(value) => PathRefusal::Outside,
            RepoPathError::NotRelative if value.contains(':') => PathRefusal::Stream,
            RepoPathError::NotRelative if value.starts_with('/') => PathRefusal::Outside,
            RepoPathError::Required | RepoPathError::NotRelative | RepoPathError::BadComponent => {
                PathRefusal::Escape
            }
        };
        Self {
            reason,
            sentence: error.sentence().to_owned(),
        }
    }
}

fn is_drive_absolute(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// Why a path gives no file, without being refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathError {
    Refused(Refused),
    /// No such file or folder in the workspace.
    Missing,
    /// It exists, and the file system would not open it.
    Unreadable,
}

impl PathError {
    /// The tool's one sentence.
    pub fn sentence(&self) -> String {
        match self {
            Self::Refused(refused) => refused.sentence.clone(),
            Self::Missing => "There is no such file in this folder.".to_owned(),
            Self::Unreadable => "That file could not be opened.".to_owned(),
        }
    }
}

impl From<Refused> for PathError {
    fn from(refused: Refused) -> Self {
        Self::Refused(refused)
    }
}

/// What a tool wants of a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// A file or folder that exists; opened for reading (a file) or its
    /// attributes (a folder).
    Existing,
    /// A file that may be created: when it does not exist, its parent is
    /// resolved and the leaf appended.
    MayCreate,
}

/// A path that passed every rule.
#[derive(Debug)]
pub struct Resolved {
    /// The derived path: relative to the root, forward slashes, long names.
    pub derived: String,
    /// The final path (`\\?\C:\…`), or for a file to be created, its parent's
    /// final path joined with the leaf.
    pub final_path: PathBuf,
    pub exists: bool,
    pub is_dir: bool,
    /// [`PathClass::Normal`] or [`PathClass::Authority`].
    pub class: PathClass,
    /// The handle the walk opened (for reading when it is a file); `None`
    /// for a file to be created.
    pub file: Option<File>,
}

/// Which ignore rules decide WP8.
pub enum IgnoreSource<'a> {
    /// `git check-ignore` in this repository.
    Git(&'a GitRunner, &'a LocalRepo),
    /// `.latticeignore` and the built-in defaults.
    Lattice(&'a IgnoreRules),
}

/// The rules for one workspace.
pub struct PathRules<'a> {
    /// The workspace root's final path.
    pub root: &'a Path,
    /// Folders whose files are Lattice's own state (WP9).
    pub lattice_state: &'a [PathBuf],
    pub ignore: IgnoreSource<'a>,
    /// WP8b: folders (derived paths; `""` is the whole folder) whose ignore
    /// rules cannot be read: nothing at or below them is read or listed.
    pub incomplete: &'a [String],
    /// WP8b: each folder's own `.gitignore`, checked once per call.
    pub ignore_files: IgnoreFileMemo,
}

/// WP8b: whether each folder's own `.gitignore` can be read (or is absent),
/// by derived folder, for one set of path rules.
#[derive(Debug, Default)]
pub struct IgnoreFileMemo(Mutex<BTreeMap<String, bool>>);

/// A folder or path as Windows compares it: without regard to case.
fn folded(text: &str) -> String {
    if cfg!(windows) {
        text.to_lowercase()
    } else {
        text.to_owned()
    }
}

/// Is the derived path `derived` the folder `folder` or below it (`""` is
/// the whole folder)? Without regard to case on Windows.
fn at_or_below(derived: &str, folder: &str) -> bool {
    if folder.is_empty() {
        return true;
    }
    let (derived, folder) = (folded(derived), folded(folder));
    derived == folder || derived.starts_with(&format!("{folder}/"))
}

fn walk_refusal(error: WalkError) -> PathError {
    match error {
        WalkError::NotFound => PathError::Missing,
        WalkError::NotLocal(_) | WalkError::Outside(_) | WalkError::BadLink(_) => {
            Refused::because(PathRefusal::RemoteLink).into()
        }
        WalkError::TooManyLinks | WalkError::NotAbsolute => {
            Refused::because(PathRefusal::Escape).into()
        }
        WalkError::Io(_) => PathError::Unreadable,
    }
}

/// A path's components as text, without a verbatim prefix.
fn components(path: &Path) -> Vec<String> {
    let text = path.to_string_lossy();
    let plain = match text.strip_prefix(r"\\?\") {
        Some(rest) if rest.as_bytes().get(1) == Some(&b':') => rest.to_owned(),
        _ => text.into_owned(),
    };
    Path::new(&plain)
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect()
}

/// The derived path: `final_path` below `root`, with forward slashes and the
/// file system's own spelling; `None` for the root itself or outside it.
fn derived_of(final_path: &Path, root: &Path) -> Option<String> {
    let path = components(final_path);
    let root = components(root);
    if path.len() <= root.len()
        || path
            .iter()
            .zip(&root)
            .any(|(a, b)| a.to_lowercase() != b.to_lowercase())
    {
        return None;
    }
    Some(path[root.len()..].join("/"))
}

impl PathRules<'_> {
    /// Every rule of the module header, for `request`.
    pub fn resolve(&self, request: &str, want: Want) -> Result<Resolved, PathError> {
        let checked = repo_path(request).map_err(|error| Refused::repo_path(request, error))?;
        let parts: Vec<&str> = checked.split('/').collect();
        if parts.iter().any(|part| is_device_name(part)) {
            return Err(Refused::because(PathRefusal::Device).into());
        }
        if parts.iter().any(|part| is_short_name_alias(part)) {
            return Err(Refused::because(PathRefusal::ShortName).into());
        }
        let target = parts
            .iter()
            .fold(self.root.to_path_buf(), |path, part| path.join(part));
        match open_walk(&target, Access::Read, LinkRule::Inside(self.root)) {
            Ok(walked) => {
                let derived = self.check_final(&walked.final_path)?;
                let class = self.classify(&derived, walked.is_dir)?;
                Ok(Resolved {
                    derived,
                    final_path: walked.final_path,
                    exists: true,
                    is_dir: walked.is_dir,
                    class,
                    file: Some(walked.file),
                })
            }
            Err(WalkError::NotFound) if want == Want::MayCreate => {
                let (leaf, parents) = parts.split_last().ok_or(PathError::Missing)?;
                let parent = parents
                    .iter()
                    .fold(self.root.to_path_buf(), |path, part| path.join(part));
                let parent_walk =
                    open_walk(&parent, Access::Attributes, LinkRule::Inside(self.root))
                        .map_err(walk_refusal)?;
                if !parent_walk.is_dir {
                    return Err(PathError::Missing);
                }
                let parent_derived = if parents.is_empty() {
                    String::new()
                } else {
                    self.check_final(&parent_walk.final_path)?
                };
                let derived = if parent_derived.is_empty() {
                    (*leaf).to_owned()
                } else {
                    format!("{parent_derived}/{leaf}")
                };
                self.check_derived(&derived)?;
                let class = self.classify(&derived, false)?;
                Ok(Resolved {
                    derived,
                    final_path: parent_walk.final_path.join(leaf),
                    exists: false,
                    is_dir: false,
                    class,
                    file: None,
                })
            }
            Err(error) => Err(walk_refusal(error)),
        }
    }

    /// WP5, WP6 and WP9 on a final path, then the derived path's own rules.
    fn check_final(&self, final_path: &Path) -> Result<String, PathError> {
        if !is_inside(final_path, self.root) {
            return Err(Refused::because(PathRefusal::Outside).into());
        }
        let Some(derived) = derived_of(final_path, self.root) else {
            // The root itself.
            return Err(Refused::because(PathRefusal::Outside).into());
        };
        if is_inside(final_path, &self.root.join(".git")) {
            return Err(Refused::because(PathRefusal::GitDir).into());
        }
        if self
            .lattice_state
            .iter()
            .any(|state| is_inside(final_path, state))
        {
            return Err(Refused::because(PathRefusal::LatticeState).into());
        }
        self.check_derived(&derived)?;
        Ok(derived)
    }

    /// WP2–WP4 and WP7 on the derived path.
    fn check_derived(&self, derived: &str) -> Result<(), PathError> {
        repo_path(derived).map_err(|error| Refused::repo_path(derived, error))?;
        if derived.split('/').any(is_device_name) {
            return Err(Refused::because(PathRefusal::Device).into());
        }
        Ok(())
    }

    /// WP8b: the refusal for a derived path at or below a folder whose
    /// ignore rules cannot be read, naming that folder: one of
    /// [`PathRules::incomplete`], or, in a git workspace, a folder from the
    /// root down to the path (the path itself too when it is a folder) whose
    /// own `.gitignore` exists but cannot be read now.
    fn incomplete_refusal(&self, derived: &str, is_dir: bool) -> Result<(), PathError> {
        let refusal = |folder: &str| -> Result<(), PathError> {
            Err(Refused {
                reason: PathRefusal::Ignored,
                sentence: incomplete_sentence(folder),
            }
            .into())
        };
        if let Some(folder) = self
            .incomplete
            .iter()
            .find(|folder| at_or_below(derived, folder))
        {
            return refusal(folder);
        }
        if !matches!(self.ignore, IgnoreSource::Git(..)) {
            return Ok(());
        }
        let parts: Vec<&str> = derived.split('/').filter(|part| !part.is_empty()).collect();
        let deepest = if is_dir {
            parts.len()
        } else {
            parts.len().saturating_sub(1)
        };
        for depth in 0..=deepest {
            let folder = parts[..depth].join("/");
            if !self.ignore_file_readable(&folder) {
                return refusal(&folder);
            }
        }
        Ok(())
    }

    /// Can `folder`'s own `.gitignore` be read, or is there none? A link is
    /// followed only inside the root (as `GitRunner::ignore_rules_incomplete`
    /// follows one); one leaving it, or any file that exists and cannot be
    /// opened for reading, cannot be read. Opened once per folder per call.
    fn ignore_file_readable(&self, folder: &str) -> bool {
        let key = folded(folder);
        let mut memo = self
            .ignore_files
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(readable) = memo.get(&key) {
            return *readable;
        }
        let file = folder
            .split('/')
            .filter(|part| !part.is_empty())
            .fold(self.root.to_path_buf(), |path, part| path.join(part))
            .join(".gitignore");
        let readable = match lattice_sys::fs::open_no_follow(&file, Access::Read) {
            Ok(opened) if opened.link.is_none() => true,
            Ok(_) => matches!(
                open_walk(&file, Access::Read, LinkRule::Inside(self.root)),
                Ok(_) | Err(WalkError::NotFound)
            ),
            Err(error) => error.kind() == std::io::ErrorKind::NotFound,
        };
        memo.insert(key, readable);
        readable
    }

    /// WP8b: every folder a listing leaves out because its ignore rules
    /// cannot be read: [`PathRules::incomplete`], then the folders whose own
    /// `.gitignore` this call found unreadable.
    pub fn withheld(&self) -> Vec<String> {
        let mut out = self.incomplete.to_vec();
        let memo = self
            .ignore_files
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (folder, readable) in memo.iter() {
            if !readable && !out.iter().any(|known| at_or_below(folder, known)) {
                out.push(folder.clone());
            }
        }
        out
    }

    /// WP8b, then WP8 on the derived path, then its class (ST4).
    fn classify(&self, derived: &str, is_dir: bool) -> Result<PathClass, PathError> {
        self.incomplete_refusal(derived, is_dir)?;
        match &self.ignore {
            IgnoreSource::Git(runner, repo) => match runner.check_ignore(repo, derived) {
                Ok(false) => {}
                Ok(true) => return Err(Refused::because(PathRefusal::Ignored).into()),
                Err(error) => {
                    return Err(Refused {
                        reason: PathRefusal::Ignored,
                        sentence: match error {
                            crate::git::runner::GitError::CouldNotSay { .. } => error.sentence(),
                            _ => "git could not say whether that file is ignored, so it is not shown."
                                .to_owned(),
                        },
                    }
                    .into());
                }
            },
            IgnoreSource::Lattice(rules) => match rules.check(derived, is_dir) {
                Verdict::Allowed => {}
                Verdict::Default => return Err(Refused::because(PathRefusal::SecretDefault).into()),
                Verdict::Ignored => return Err(Refused::because(PathRefusal::Ignored).into()),
            },
        }
        Ok(if is_authority(derived) {
            PathClass::Authority
        } else {
            PathClass::Normal
        })
    }

    /// A file a walk listed, opened because its content is about to be
    /// returned (§6.2: "a file whose content is returned also passes WP10"):
    /// the lexical rules, the no-follow walk bounded by the root, WP5, WP6
    /// and WP9 on the final path. WP8 already held for the listed path; when
    /// the derived path differs (a link inside the folder), WP8 runs again on
    /// the derived path, and the class is the derived path's.
    pub fn open_listed(&self, listed: &str) -> Result<Resolved, PathError> {
        self.listed(listed)?;
        let target = listed
            .split('/')
            .fold(self.root.to_path_buf(), |path, part| path.join(part));
        let walked =
            open_walk(&target, Access::Read, LinkRule::Inside(self.root)).map_err(walk_refusal)?;
        let derived = self.check_final(&walked.final_path)?;
        let class = if derived == listed {
            if is_authority(&derived) {
                PathClass::Authority
            } else {
                PathClass::Normal
            }
        } else {
            self.classify(&derived, walked.is_dir)?
        };
        Ok(Resolved {
            derived,
            final_path: walked.final_path,
            exists: true,
            is_dir: walked.is_dir,
            class,
            file: Some(walked.file),
        })
    }

    /// The lexical rules (WP1–WP4, WP7, WP9, WP11) and the class (ST4) of a
    /// path a walk listed. The walk itself applied WP8 (`git ls-files
    /// --exclude-standard`, or the ignore rules); a file whose content is
    /// returned goes through [`PathRules::open_listed`] as well (WP10).
    pub fn listed(&self, path: &str) -> Result<PathClass, PathError> {
        let checked = repo_path(path).map_err(|error| Refused::repo_path(path, error))?;
        let parts: Vec<&str> = checked.split('/').collect();
        if parts.iter().any(|part| is_device_name(part)) {
            return Err(Refused::because(PathRefusal::Device).into());
        }
        if parts.iter().any(|part| is_short_name_alias(part)) {
            return Err(Refused::because(PathRefusal::ShortName).into());
        }
        let target = parts
            .iter()
            .fold(self.root.to_path_buf(), |path, part| path.join(part));
        if under_any(&target, self.lattice_state) {
            return Err(Refused::because(PathRefusal::LatticeState).into());
        }
        self.incomplete_refusal(checked, false)?;
        Ok(if is_authority(checked) {
            PathClass::Authority
        } else {
            PathClass::Normal
        })
    }
}

/// WP8b's answer, naming the folder whose ignore rules git cannot read.
pub fn incomplete_sentence(folder: &str) -> String {
    let place = if folder.is_empty() {
        "this folder".to_owned()
    } else {
        format!("{folder}/")
    };
    format!(
        "git cannot read the ignore rules for {place} (an ignore file there is missing from this partial or sparse checkout, or cannot be read), so nothing at or below it is read or listed."
    )
}

/// The repository's ignore source for a workspace's FT6 result.
pub fn ignore_source<'a>(
    repo: &'a Repo,
    runner: &'a GitRunner,
    rules: &'a IgnoreRules,
) -> IgnoreSource<'a> {
    match repo {
        Repo::Git(local) => IgnoreSource::Git(runner, local),
        Repo::None | Repo::Without(_) => IgnoreSource::Lattice(rules),
    }
}

/// Whether `path` (a final path) is under any of `roots`.
pub fn under_any(path: &Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|root| localfs::is_inside(path, root))
}
