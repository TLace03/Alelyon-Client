//! Attaching a folder to a conversation (the chat core's spec §6.1).
//!
//! The path comes from the shell's native folder picker or Tauri's Rust-side
//! drop event ([`AttachSource::Native`]), or from the page
//! ([`AttachSource::Page`]), which is accepted only after
//! `ConfirmPort(AttachFolder{path})` answers true; a page path that would be
//! refused anyway is refused without a dialog.
//!
//! Refused, with a sentence ([`AttachRefusal`]), before anything is
//! recorded:
//! - by its text, before any open: a UNC or `\\?\UNC` path, a device path, a
//!   folder on a `DRIVE_REMOTE` drive, a relative path, a drive root;
//! - after the no-follow walk (a link to the network is refused before it is
//!   followed): a path that does not exist, a file, and on the final path:
//!   a drive root, `%SystemRoot%` or anything in it (stricter than the spec,
//!   which names the folder only), `<globals>` or anything in it, the
//!   Python state home `~/.alelyon` or anything in it, `%USERPROFILE%`
//!   itself, `%APPDATA%` or `%LOCALAPPDATA%` or anything in them, and any
//!   other ancestor of `<globals>`, **except** a git checkout's top level
//!   whose own `globals/` folder is `<globals>` (the source-checkout layout),
//!   where WP9 refuses `<globals>` path by path.
//!
//! Then the folder's identity is read from its handle (`FILE_ID_INFO`), FT6
//! reads its `.git` natively, and only for a folder FT6 allows is git asked
//! for its top level, through the git runner (which also refuses a git older
//! than its minimum, R10).

use std::path::{Component, Path, PathBuf};

use lattice_sys::fs::{Access, DriveType, PathKind, drive_type, file_identity, path_kind};

use super::{Workspace, workspace_id};
use crate::env::Env;
use crate::git::dotgit::{self, Repo};
use crate::git::runner::GitRunner;
use crate::localfs::{self, LinkRule, WalkError, is_inside, open_walk};
use crate::ports::{ConfirmPort, ConfirmRequest};
use crate::state::StateRoot;

/// Where a path to attach came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AttachSource {
    /// The shell's native folder picker, or a Rust-side drop: OS paths.
    Native(PathBuf),
    /// A string from the page.
    Page(String),
}

/// Why a folder is not attached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttachRefusal {
    NotAbsolute,
    Network,
    Device,
    RemoteDrive,
    DriveRoot,
    Missing,
    NotAFolder,
    Unreadable,
    SystemFolder,
    LatticeState,
    Profile,
    AppData,
    AboveLatticeState,
    /// The reader did not confirm a path that came from the page.
    NotConfirmed,
}

impl AttachRefusal {
    pub fn sentence(self) -> &'static str {
        match self {
            Self::NotAbsolute => "Attach a folder by its full path.",
            Self::Network => "A folder on the network cannot be attached.",
            Self::Device => "A device path cannot be attached.",
            Self::RemoteDrive => "A folder on a network drive cannot be attached.",
            Self::DriveRoot => "A whole drive cannot be attached; choose a folder on it.",
            Self::Missing => "That folder does not exist.",
            Self::NotAFolder => "That is a file, not a folder.",
            Self::Unreadable => "That folder could not be opened.",
            Self::SystemFolder => "The Windows folder cannot be attached.",
            Self::LatticeState => "Lattice's own state cannot be attached.",
            Self::Profile => {
                "Your whole user folder cannot be attached; choose a project folder in it."
            }
            Self::AppData => {
                "Application data folders hold credentials and application state, so they cannot be attached."
            }
            Self::AboveLatticeState => {
                "That folder holds Lattice's own state, so it cannot be attached."
            }
            Self::NotConfirmed => "The folder was not attached.",
        }
    }
}

/// The refusals a path's text decides, before anything opens it.
fn by_text(path: &Path) -> Result<(), AttachRefusal> {
    match path_kind(path) {
        PathKind::Unc => return Err(AttachRefusal::Network),
        PathKind::Device => return Err(AttachRefusal::Device),
        PathKind::Rooted | PathKind::Relative => return Err(AttachRefusal::NotAbsolute),
        PathKind::Drive | PathKind::VolumeGuid => {}
    }
    if drive_type(path) == DriveType::Remote {
        return Err(AttachRefusal::RemoteDrive);
    }
    if is_drive_root(path) {
        return Err(AttachRefusal::DriveRoot);
    }
    Ok(())
}

/// A drive's or a volume's root: no folder below the prefix, once `.` and
/// `..` are taken as Windows takes them.
fn is_drive_root(path: &Path) -> bool {
    let text = path.to_string_lossy();
    let plain = match text.strip_prefix(r"\\?\") {
        Some(rest) if rest.as_bytes().get(1) == Some(&b':') => rest.replace('/', "\\"),
        _ => text.replace('/', "\\"),
    };
    let mut depth: usize = 0;
    for component in Path::new(&plain).components() {
        match component {
            Component::Normal(_) => depth += 1,
            Component::ParentDir => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    if path_kind(path) == PathKind::VolumeGuid {
        // `\\?\Volume{…}\` parses as one verbatim prefix and nothing else.
        return plain.trim_end_matches('\\').matches('\\').count() <= 3;
    }
    depth == 0
}

/// The path to attach: a native path as it is; a page string only when its
/// text could be attached at all, and after the reader's native
/// confirmation. Nothing is attached when the reader declines.
pub async fn confirmed_path(
    source: AttachSource,
    confirm: &dyn ConfirmPort,
) -> Result<PathBuf, AttachRefusal> {
    match source {
        AttachSource::Native(path) => Ok(path),
        AttachSource::Page(text) => {
            let path = PathBuf::from(&text);
            by_text(&path)?;
            if confirm
                .confirm(ConfirmRequest::AttachFolder { path: text })
                .await
            {
                Ok(path)
            } else {
                Err(AttachRefusal::NotConfirmed)
            }
        }
    }
}

/// `path`'s final path when it exists, else `path` as written.
fn canonical(path: &Path) -> PathBuf {
    match open_walk(path, Access::Attributes, LinkRule::AnyLocal) {
        Ok(walked) => walked.final_path,
        Err(_) => path.to_path_buf(),
    }
}

fn same(a: &Path, b: &Path) -> bool {
    is_inside(a, b) && is_inside(b, a)
}

fn env_path(env: &dyn Env, name: &str) -> Option<PathBuf> {
    env.var(name)
        .filter(|value| !value.is_empty())
        .map(|value| canonical(Path::new(&value)))
}

/// The folders WP9 refuses inside a workspace: `<globals>` and the Python
/// state home (`~/.alelyon`), as final paths where they exist.
pub fn lattice_state(env: &dyn Env, state: &StateRoot) -> Vec<PathBuf> {
    let mut roots = vec![canonical(&state.globals)];
    if let Some(home) = ["USERPROFILE", "HOME"]
        .into_iter()
        .find_map(|name| env.var(name).filter(|value| !value.is_empty()))
    {
        roots.push(canonical(&Path::new(&home).join(".alelyon")));
    }
    roots
}

/// Attach `path` (see the module header). Blocking: run it on the blocking
/// pool.
pub fn attach_path(
    path: &Path,
    env: &dyn Env,
    state: &StateRoot,
    runner: &GitRunner,
) -> Result<Workspace, AttachRefusal> {
    by_text(path)?;
    let walked =
        open_walk(path, Access::Attributes, LinkRule::AnyLocal).map_err(|error| match error {
            WalkError::NotFound => AttachRefusal::Missing,
            WalkError::NotLocal(_) => AttachRefusal::Network,
            _ => AttachRefusal::Unreadable,
        })?;
    if !walked.is_dir {
        return Err(AttachRefusal::NotAFolder);
    }
    let root = walked.final_path;
    if is_drive_root(&root) {
        return Err(AttachRefusal::DriveRoot);
    }
    if let Some(system) = env_path(env, "SystemRoot")
        && is_inside(&root, &system)
    {
        return Err(AttachRefusal::SystemFolder);
    }
    let state_roots = lattice_state(env, state);
    if state_roots.iter().any(|state| is_inside(&root, state)) {
        return Err(AttachRefusal::LatticeState);
    }
    if let Some(profile) = env_path(env, "USERPROFILE")
        && same(&root, &profile)
    {
        return Err(AttachRefusal::Profile);
    }
    for name in ["APPDATA", "LOCALAPPDATA"] {
        if let Some(folder) = env_path(env, name)
            && is_inside(&root, &folder)
        {
            return Err(AttachRefusal::AppData);
        }
    }
    let globals = &state_roots[0];
    if is_inside(globals, &root) && !is_checkout_top_holding(&root, globals) {
        return Err(AttachRefusal::AboveLatticeState);
    }
    let identity = file_identity(&walked.file).map_err(|_| AttachRefusal::Unreadable)?;
    let repo = dotgit::inspect(&root, env);
    let (top_level, git_note) = match &repo {
        Repo::Git(local) => match runner.top_level(local) {
            Ok(top) => (Some(top), None),
            Err(error) => (None, Some(error.sentence())),
        },
        Repo::Without(without) => (None, Some(without.sentence())),
        Repo::None => (None, None),
    };
    let name = root
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ignore_cache = super::IgnoreCache::default();
    let ignore_rules_incomplete = super::incomplete_rules_cached(&repo, runner, &ignore_cache);
    Ok(Workspace {
        id: workspace_id(&identity),
        root,
        name,
        identity,
        repo,
        top_level,
        git_note,
        ignore_rules_incomplete,
        ignore_cache,
        lattice_state: state_roots,
    })
}

/// The source-checkout exception: `root` is a git checkout's top level (it
/// holds `.git`) whose own `globals/` folder is `<globals>`.
fn is_checkout_top_holding(root: &Path, globals: &Path) -> bool {
    localfs::probe(&root.join(".git")).is_ok() && same(&canonical(&root.join("globals")), globals)
}
