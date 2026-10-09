//! FT6: a folder's `.git` read natively before any git process starts in it
//! (the chat core's spec §6.3 FT6, §6.1). Not a port.
//!
//! Git opens whatever path its repository files name, so a downloaded folder
//! could make a git call reach the network: a gitfile naming `\\host\share`,
//! an alternate object store on a share, a configuration include or an
//! excludes file there. [`inspect`] finds the repository a folder is in as
//! git does (from the folder up to its volume's root, `.git` first, then a
//! bare repository's own layout), and reads, without starting git:
//! - `.git`: a folder, or a gitfile (`gitdir: <path>`), and a link at `.git`
//!   only when its target is local;
//! - the git folder's `commondir`;
//! - `objects/info/alternates`, and each alternate's own, to git's depth (5);
//! - the repository's `config` (and the worktree's `config.worktree`), with
//!   every `include.path` and `includeIf.*.path` followed whatever its
//!   condition, to git's depth (10).
//!
//! If any of the gitfile target, `commondir`, an alternate, an include,
//! `core.excludesFile`, `core.attributesFile` or `core.worktree` names a path
//! that is **not local** (UNC, `\\?\UNC`, a device path, a `DRIVE_REMOTE`
//! drive), the folder counts as one **without git** ([`Repo::Without`]): no git
//! process starts there. So does anything that cannot be read or parsed (fail
//! closed), and a bare repository's layout (git would read its `config` too).
//! A gitfile with a local target, which every linked worktree is, is allowed.
//! `core.worktree` is checked beyond the spec's list: git changes into it
//! before it reads the working tree, so a share there connects as well.
//!
//! Only [`inspect`] makes a [`LocalRepo`], and the git runner (`runner.rs`)
//! takes one for every call that runs in a folder, so no git process starts in
//! a folder this has not read first. The user's global and system git
//! configuration is the user's own and is not read here (spec §9.4's excluded
//! actor). Every file is opened through `localfs::open_walk`, which never
//! follows a link to anything but a local path; a path whose text is not local
//! is refused here before it reaches an open at all.

use std::io::Read;
use std::path::{Path, PathBuf};

use lattice_sys::fs::{Access, PathKind, path_kind};

use super::config::{self, Entry};
use crate::env::Env;
use crate::localfs::{self, LinkRule, WalkError, is_local_text, open_walk};

/// How deep git follows alternates of alternates.
pub const MAX_ALTERNATE_DEPTH: usize = 5;
/// How deep git follows configuration includes (`MAX_INCLUDE_DEPTH`).
pub const MAX_INCLUDE_DEPTH: usize = 10;
/// The largest gitfile, `commondir` or `alternates` file read.
pub const MAX_SMALL_FILE: u64 = 64 * 1024;
/// The largest configuration file read.
pub const MAX_CONFIG_FILE: u64 = 1024 * 1024;

/// Which file or setting named a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Named {
    /// The folder itself.
    Folder,
    /// A link at `.git`.
    DotGit,
    /// A gitfile's `gitdir:` line.
    Gitfile,
    CommonDir,
    Alternate,
    /// `include.path` or `includeIf.*.path`.
    Include,
    ExcludesFile,
    AttributesFile,
    WorkTree,
    Config,
}

impl Named {
    pub fn what(self) -> &'static str {
        match self {
            Self::Folder => "the folder",
            Self::DotGit => "its .git link",
            Self::Gitfile => "its .git file",
            Self::CommonDir => "the repository's commondir",
            Self::Alternate => "an alternate object store",
            Self::Include => "a configuration include",
            Self::ExcludesFile => "core.excludesFile",
            Self::AttributesFile => "core.attributesFile",
            Self::WorkTree => "core.worktree",
            Self::Config => "the repository's configuration",
        }
    }
}

/// Why a folder counts as one without git.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WithoutGit {
    /// A file or setting names a path that is not on this machine.
    NotLocal { named: Named, path: String },
    /// A file could not be read.
    Unreadable { named: Named, path: String },
    /// A file git would read cannot be parsed, or uses a form this reader
    /// does not resolve (`~user/`, `%(prefix)/`).
    Unparsable {
        named: Named,
        path: String,
        line: usize,
    },
    /// Includes nested deeper than git allows.
    TooDeep,
    /// The folder is in a bare repository's layout.
    Bare { path: String },
}

impl WithoutGit {
    /// One sentence for the folder's view.
    pub fn sentence(&self) -> String {
        match self {
            Self::NotLocal { named, .. } => format!(
                "Lattice treats this folder as one without git: {} names a network or device path.",
                named.what()
            ),
            Self::Unreadable { named, .. } => format!(
                "Lattice treats this folder as one without git: {} could not be read.",
                named.what()
            ),
            Self::Unparsable { named, .. } => format!(
                "Lattice treats this folder as one without git: {} could not be read as git reads it.",
                named.what()
            ),
            Self::TooDeep => "Lattice treats this folder as one without git: its configuration includes are nested too deeply.".to_owned(),
            Self::Bare { .. } => "Lattice treats this folder as one without git: it is inside a bare repository.".to_owned(),
        }
    }
}

/// A repository whose files name only local paths: git may run in
/// [`LocalRepo::folder`]. Made only by [`inspect`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalRepo {
    folder: PathBuf,
    git_dir: PathBuf,
    common_dir: PathBuf,
    gitfile: bool,
}

impl LocalRepo {
    /// The folder inspected (its final path), where git runs.
    pub fn folder(&self) -> &Path {
        &self.folder
    }

    /// The git folder (`.git`, or a gitfile's target).
    pub fn git_dir(&self) -> &Path {
        &self.git_dir
    }

    /// Where the objects, refs and configuration live.
    pub fn common_dir(&self) -> &Path {
        &self.common_dir
    }

    /// True when `.git` is a gitfile (a linked worktree, or a submodule).
    pub fn is_gitfile(&self) -> bool {
        self.gitfile
    }
}

/// What a folder is, for git.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Repo {
    /// No repository from the folder up to its volume's root.
    None,
    /// Git may run here.
    Git(LocalRepo),
    /// Treated as a folder without git (FT6); no git process starts.
    Without(WithoutGit),
}

fn shown(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn walk_error(named: Named, path: &Path, error: WalkError) -> WithoutGit {
    match error {
        WalkError::NotLocal(target) => WithoutGit::NotLocal {
            named,
            path: shown(&target),
        },
        _ => WithoutGit::Unreadable {
            named,
            path: shown(path),
        },
    }
}

/// Read a whole small file through the no-follow walk: `Ok(None)` when it
/// does not exist.
fn read_file(named: Named, path: &Path, limit: u64) -> Result<Option<String>, WithoutGit> {
    let walked = match open_walk(path, Access::Read, LinkRule::AnyLocal) {
        Ok(walked) => walked,
        Err(WalkError::NotFound) => return Ok(None),
        Err(error) => return Err(walk_error(named, path, error)),
    };
    let unreadable = || WithoutGit::Unreadable {
        named,
        path: shown(path),
    };
    if walked.is_dir {
        return Err(unreadable());
    }
    let mut bytes = Vec::new();
    walked
        .file
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| unreadable())?;
    if bytes.len() as u64 > limit {
        return Err(WithoutGit::Unparsable {
            named,
            path: shown(path),
            line: 0,
        });
    }
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| WithoutGit::Unparsable {
            named,
            path: shown(path),
            line: 0,
        })
}

/// The home folder `~` means for git: `HOME`, else `USERPROFILE`.
fn home(env: &dyn Env) -> Option<PathBuf> {
    ["HOME", "USERPROFILE"]
        .into_iter()
        .find_map(|name| env.var(name).filter(|value| !value.is_empty()))
        .map(PathBuf::from)
}

/// `value` as a path git would use, judged before anything opens it:
/// `~/` expanded, a relative path joined to `base`, a rooted one put on
/// `base`'s drive. A path that is not local is refused; so is a form this
/// reader does not resolve.
fn resolve_named(
    named: Named,
    value: &str,
    base: &Path,
    env: &dyn Env,
    file: &Path,
    line: usize,
) -> Result<PathBuf, WithoutGit> {
    let unparsable = || WithoutGit::Unparsable {
        named,
        path: shown(file),
        line,
    };
    let expanded: PathBuf = if value == "~" {
        home(env).ok_or_else(unparsable)?
    } else if let Some(rest) = value
        .strip_prefix("~/")
        .or_else(|| value.strip_prefix("~\\"))
    {
        home(env).ok_or_else(unparsable)?.join(rest)
    } else if value.starts_with('~') || value.starts_with("%(prefix)") {
        return Err(unparsable());
    } else {
        PathBuf::from(value)
    };
    let full = match path_kind(&expanded) {
        PathKind::Relative => {
            // A drive-relative `C:x` names that drive's current folder, which
            // this reader cannot know: refused, failing closed.
            let text = expanded.to_string_lossy();
            let bytes = text.as_bytes();
            if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
                return Err(unparsable());
            }
            base.join(&expanded)
        }
        PathKind::Rooted => {
            let text = shown(base);
            let plain = text.strip_prefix(r"\\?\").unwrap_or(&text);
            let drive: String = plain.chars().take(2).collect();
            PathBuf::from(format!(
                r"{drive}\{}",
                expanded.to_string_lossy().trim_start_matches(['\\', '/'])
            ))
        }
        _ => expanded,
    };
    if !is_local_text(&full) {
        return Err(WithoutGit::NotLocal {
            named,
            path: shown(&full),
        });
    }
    Ok(full)
}

/// Open `path` as a folder through the walk.
fn folder_at(named: Named, path: &Path) -> Result<Option<PathBuf>, WithoutGit> {
    match open_walk(path, Access::Attributes, LinkRule::AnyLocal) {
        Ok(walked) if walked.is_dir => Ok(Some(walked.final_path)),
        Ok(_) => Err(WithoutGit::Unreadable {
            named,
            path: shown(path),
        }),
        Err(WalkError::NotFound) => Ok(None),
        Err(error) => Err(walk_error(named, path, error)),
    }
}

/// Does `dir` exist as a file or a folder, without following a link at its
/// last component? (`dir`'s own components are already a final path.)
fn exists(path: &Path) -> Option<bool> {
    localfs::probe(path).ok().map(|opened| opened.is_dir)
}

/// Git's test for a bare repository's layout at `dir`.
fn looks_bare(dir: &Path) -> bool {
    exists(&dir.join("HEAD")) == Some(false)
        && exists(&dir.join("objects")) == Some(true)
        && exists(&dir.join("refs")) == Some(true)
}

/// Find the repository `folder` is in, and read what it names (FT6).
pub fn inspect(folder: &Path, env: &dyn Env) -> Repo {
    match inspect_inner(folder, env) {
        Ok(Some(repo)) => Repo::Git(repo),
        Ok(None) => Repo::None,
        Err(without) => Repo::Without(without),
    }
}

fn inspect_inner(folder: &Path, env: &dyn Env) -> Result<Option<LocalRepo>, WithoutGit> {
    let Some(start) = folder_at(Named::Folder, folder)? else {
        return Err(WithoutGit::Unreadable {
            named: Named::Folder,
            path: shown(folder),
        });
    };
    // (the git folder, whether `.git` was a gitfile)
    let mut found: Option<(PathBuf, bool)> = None;
    for dir in start.ancestors() {
        let dot_git = dir.join(".git");
        let opened = match localfs::probe(&dot_git) {
            Ok(opened) => opened,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                if looks_bare(dir) {
                    return Err(WithoutGit::Bare { path: shown(dir) });
                }
                continue;
            }
            Err(_) => {
                return Err(WithoutGit::Unreadable {
                    named: Named::DotGit,
                    path: shown(&dot_git),
                });
            }
        };
        // A link at `.git` is followed only to a local path (the walk refuses
        // anything else before following it).
        let walked = open_walk(&dot_git, Access::Attributes, LinkRule::AnyLocal)
            .map_err(|error| walk_error(Named::DotGit, &dot_git, error))?;
        drop(opened);
        if walked.is_dir {
            found = Some((walked.final_path, false));
        } else {
            let text = read_file(Named::Gitfile, &dot_git, MAX_SMALL_FILE)?.unwrap_or_default();
            let unparsable = || WithoutGit::Unparsable {
                named: Named::Gitfile,
                path: shown(&dot_git),
                line: 1,
            };
            let first = text.lines().next().ok_or_else(unparsable)?;
            let target = first
                .strip_prefix("gitdir: ")
                .map(str::trim_end)
                .filter(|target| !target.is_empty())
                .ok_or_else(unparsable)?;
            let target = resolve_named(Named::Gitfile, target, dir, env, &dot_git, 1)?;
            let git_dir =
                folder_at(Named::Gitfile, &target)?.ok_or_else(|| WithoutGit::Unreadable {
                    named: Named::Gitfile,
                    path: shown(&target),
                })?;
            found = Some((git_dir, true));
        }
        break;
    }
    let Some((git_dir, gitfile)) = found else {
        return Ok(None);
    };

    let commondir_file = git_dir.join("commondir");
    let common_dir = match read_file(Named::CommonDir, &commondir_file, MAX_SMALL_FILE)? {
        None => git_dir.clone(),
        Some(text) => {
            let value = text.lines().next().unwrap_or("").trim_end();
            if value.is_empty() {
                git_dir.clone()
            } else {
                let path =
                    resolve_named(Named::CommonDir, value, &git_dir, env, &commondir_file, 1)?;
                folder_at(Named::CommonDir, &path)?.ok_or_else(|| WithoutGit::Unreadable {
                    named: Named::CommonDir,
                    path: shown(&path),
                })?
            }
        }
    };

    check_alternates(&common_dir.join("objects"), env, 0)?;

    // `config.worktree` counts only with `extensions.worktreeConfig`; it is
    // read whenever it exists, which is stricter.
    for file in [common_dir.join("config"), git_dir.join("config.worktree")] {
        check_config(&file, env, 0)?;
    }

    Ok(Some(LocalRepo {
        folder: start,
        git_dir,
        common_dir,
        gitfile,
    }))
}

/// Git's `unquote_c_style` for an alternates line in quotes.
fn unquote(line: &str) -> Option<String> {
    let inner = line.strip_prefix('"')?.strip_suffix('"')?;
    let mut out: Vec<u8> = Vec::new();
    let bytes = inner.as_bytes();
    let mut at = 0;
    while at < bytes.len() {
        let b = bytes[at];
        if b == b'"' {
            return None;
        }
        if b != b'\\' {
            out.push(b);
            at += 1;
            continue;
        }
        let next = *bytes.get(at + 1)?;
        let mapped = match next {
            b'a' => 7,
            b'b' => 8,
            b'f' => 12,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'v' => 11,
            b'\\' => b'\\',
            b'"' => b'"',
            b'0'..=b'3' => {
                let digits = bytes.get(at + 1..at + 4)?;
                if !digits.iter().all(|d| (b'0'..=b'7').contains(d)) {
                    return None;
                }
                let value = (u32::from(digits[0] - b'0') << 6)
                    | (u32::from(digits[1] - b'0') << 3)
                    | u32::from(digits[2] - b'0');
                out.push(u8::try_from(value).ok()?);
                at += 4;
                continue;
            }
            _ => return None,
        };
        out.push(mapped);
        at += 2;
    }
    String::from_utf8(out).ok()
}

fn check_alternates(objects: &Path, env: &dyn Env, depth: usize) -> Result<(), WithoutGit> {
    if depth >= MAX_ALTERNATE_DEPTH {
        return Ok(());
    }
    let file = objects.join("info").join("alternates");
    let Some(text) = read_file(Named::Alternate, &file, MAX_SMALL_FILE)? else {
        return Ok(());
    };
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let value = if line.starts_with('"') {
            unquote(line).ok_or_else(|| WithoutGit::Unparsable {
                named: Named::Alternate,
                path: shown(&file),
                line: index + 1,
            })?
        } else {
            line.to_owned()
        };
        let path = resolve_named(Named::Alternate, &value, objects, env, &file, index + 1)?;
        // Git skips an alternate that does not exist.
        if let Some(alternate) = folder_at(Named::Alternate, &path)? {
            check_alternates(&alternate, env, depth + 1)?;
        }
    }
    Ok(())
}

fn is_include(entry: &Entry) -> bool {
    entry.key == "include.path"
        || (entry.key.starts_with("includeif.") && entry.key.ends_with(".path"))
}

fn check_config(file: &Path, env: &dyn Env, depth: usize) -> Result<(), WithoutGit> {
    if depth > MAX_INCLUDE_DEPTH {
        return Err(WithoutGit::TooDeep);
    }
    let named = if depth == 0 {
        Named::Config
    } else {
        Named::Include
    };
    let Some(text) = read_file(named, file, MAX_CONFIG_FILE)? else {
        return Ok(());
    };
    let entries = config::parse(&text).map_err(|error| WithoutGit::Unparsable {
        named,
        path: shown(file),
        line: error.line,
    })?;
    let base = file.parent().unwrap_or(file).to_path_buf();
    for entry in &entries {
        let checked = if is_include(entry) {
            Some(Named::Include)
        } else {
            match entry.key.as_str() {
                "core.excludesfile" => Some(Named::ExcludesFile),
                "core.attributesfile" => Some(Named::AttributesFile),
                "core.worktree" => Some(Named::WorkTree),
                _ => None,
            }
        };
        let Some(which) = checked else {
            continue;
        };
        let Some(value) = entry.value.as_deref() else {
            // Git refuses a path setting with no value.
            return Err(WithoutGit::Unparsable {
                named: which,
                path: shown(file),
                line: entry.line,
            });
        };
        if value.is_empty() {
            continue;
        }
        let path = resolve_named(which, value, &base, env, file, entry.line)?;
        if which == Named::Include {
            check_config(&path, env, depth + 1)?;
        }
    }
    Ok(())
}
