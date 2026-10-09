//! A conversation's folder (the chat core's spec §6): attaching it, its
//! identity, and the rules every tool path must pass. Not a port, apart from
//! `paths::repo_path` (the web's `workspaces/paths.py`).
//!
//! - [`attach`]: §6.1's refusals (a drive root, the Windows folder, a network
//!   or device path, a network drive, `<globals>` and its ancestors, the
//!   user's profile itself, `%APPDATA%` and `%LOCALAPPDATA%`), a page's path
//!   only after the reader's native confirmation, the folder's identity, and
//!   FT6 before git's top level is asked for;
//! - [`paths`]: WP1–WP11 and the authority class (ST4), decided on the
//!   derived long-name path;
//! - [`ignore`]: WP8 for a folder without git;
//! - [`trust`]: folder trust (FT1–FT5), asked for only through the native
//!   dialog and kept outside the folder;
//! - [`rules`]: the reader's rules and, after trust, the folder's rules
//!   files, as one leading `User` item that grants nothing;
//! - [`lease`]: the writer lease (ADR-0038), the one file every host locks
//!   before it writes a checkout, derived as Python derives it.
//!
//! A [`Workspace`] is matched to a stored trust or permission record only when
//! both its id and its path match ([`WorkspaceKey`]), so a folder that is
//! replaced, renamed or moved is asked about again (§4.1).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lattice_protocol::conversation::WorkspaceBadge;
use lattice_sys::fs::FileIdentity;

use crate::exec::spawn::comparable;
use crate::git::dotgit::Repo;
use crate::git::runner::{GitRunner, IndexIgnoreFiles, IndexKey};
use crate::sha::sha256_hex;

pub mod attach;
pub mod ignore;
pub mod lease;
pub mod paths;
pub mod rules;
pub mod trust;

#[cfg(test)]
mod lease_interop_tests;
#[cfg(test)]
mod lease_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod trust_tests;

/// `WorkspaceId`: 16 hexadecimal characters of SHA-256 over the root's
/// volume serial and 128-bit file id (§4.1).
pub fn workspace_id(identity: &FileIdentity) -> String {
    sha256_hex(&identity.to_bytes())[..16].to_owned()
}

/// What a trust or permission record is matched by: the id and the
/// canonical path, both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkspaceKey {
    pub id: String,
    pub path: String,
}

impl WorkspaceKey {
    /// True only when both the id and the path (compared without case or a
    /// verbatim prefix) are the same.
    pub fn matches(&self, other: &WorkspaceKey) -> bool {
        self.id == other.id && comparable(&self.path) == comparable(&other.path)
    }
}

/// An attached folder.
#[derive(Clone, Debug)]
pub struct Workspace {
    /// [`workspace_id`] of the root.
    pub id: String,
    /// The root's final path (`\\?\C:\…`): long names, links resolved.
    pub root: PathBuf,
    /// The folder's own name.
    pub name: String,
    pub identity: FileIdentity,
    /// FT6's verdict at attach.
    pub repo: Repo,
    /// `git rev-parse --show-toplevel`, when git may run here and answered.
    pub top_level: Option<PathBuf>,
    /// Why git could not be asked, in one sentence (FT6, R10, no git).
    pub git_note: Option<String>,
    /// Folders whose files are Lattice's own state (WP9).
    pub lattice_state: Vec<PathBuf>,
    /// WP8b: folders, relative to the root (`""` is the whole folder), whose
    /// ignore rules git could not read at attach, so nothing at or below
    /// them is read or listed. The interface says so (the workspace view's
    /// `ignore_rules_incomplete`). Each tool call reads this again
    /// ([`Workspace::with_rules`]); this is the attach-time answer.
    pub ignore_rules_incomplete: Vec<String>,
    /// WP8b's index part, kept while the index, `HEAD` and the configuration
    /// are unchanged ([`IndexKey`]), so a tool call starts no git process for
    /// it. Shared by this workspace's clones.
    pub ignore_cache: IgnoreCache,
}

/// What [`incomplete_rules_cached`] keeps for one workspace: the index part
/// of WP8b with the [`IndexKey`] it was read under.
#[derive(Clone, Debug, Default)]
pub struct IgnoreCache(Arc<Mutex<Option<KeptIndex>>>);

/// The index part and the key it was read under.
type KeptIndex = (IndexKey, Arc<IndexIgnoreFiles>);

impl Workspace {
    pub fn key(&self) -> WorkspaceKey {
        WorkspaceKey {
            id: self.id.clone(),
            path: self.root.to_string_lossy().into_owned(),
        }
    }

    /// The sidebar's badge.
    pub fn badge(&self, trusted: bool) -> WorkspaceBadge {
        WorkspaceBadge {
            id: self.id.clone(),
            name: self.name.clone(),
            path: shown_path(&self.root),
            trusted,
        }
    }

    /// The path rules for this folder, with `runner` for a git workspace and
    /// `.latticeignore` read now for one without git.
    pub fn with_rules<T>(
        &self,
        runner: &GitRunner,
        then: impl FnOnce(&paths::PathRules<'_>) -> T,
    ) -> Result<T, ignore::RulesError> {
        let rules = match &self.repo {
            Repo::Git(_) => ignore::IgnoreRules::defaults_only(),
            Repo::None | Repo::Without(_) => ignore::IgnoreRules::load(&self.root)?,
        };
        let incomplete = incomplete_rules_cached(&self.repo, runner, &self.ignore_cache);
        let path_rules = paths::PathRules {
            root: &self.root,
            lattice_state: &self.lattice_state,
            ignore: paths::ignore_source(&self.repo, runner, &rules),
            incomplete: &incomplete,
            ignore_files: paths::IgnoreFileMemo::default(),
        };
        Ok(then(&path_rules))
    }
}

/// WP8b: the folders whose git ignore rules cannot be read now
/// ([`GitRunner::ignore_rules_incomplete`]). When git ran and could not say,
/// the whole folder (fail closed). When git cannot run at all, none here:
/// every read then fails closed on its own `check-ignore` and every listing
/// on its `ls-files`, each with its own sentence. Without git there are
/// none: `.latticeignore` is read by Lattice itself, and an unreadable one
/// already refuses everything.
pub fn incomplete_rules(repo: &Repo, runner: &GitRunner) -> Vec<String> {
    incomplete_rules_cached(repo, runner, &IgnoreCache::default())
}

/// [`incomplete_rules`], with the index part kept in `cache` while its
/// [`IndexKey`] is unchanged (read natively, no git process). The working
/// tree's part (`info/exclude`, `core.excludesFile`, each index
/// `.gitignore`) is read on every call. When the key cannot be read, the
/// index part is read again every time and not kept.
pub fn incomplete_rules_cached(
    repo: &Repo,
    runner: &GitRunner,
    cache: &IgnoreCache,
) -> Vec<String> {
    use crate::git::runner::GitError;
    let read = |local: &crate::git::dotgit::LocalRepo| -> Result<Vec<String>, GitError> {
        let key = IndexKey::read(local);
        let kept = {
            let guard = cache
                .0
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match (&key, &*guard) {
                (Some(key), Some((kept_key, index))) if key == kept_key => Some(Arc::clone(index)),
                _ => None,
            }
        };
        let index = match kept {
            Some(index) => index,
            None => {
                let index = Arc::new(runner.index_ignore_files(local)?);
                if let Some(key) = key {
                    *cache
                        .0
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                        Some((key, Arc::clone(&index)));
                }
                index
            }
        };
        runner.incomplete_now(local, &index)
    };
    match repo {
        Repo::Git(local) => match read(local) {
            Ok(folders) => folders,
            Err(
                GitError::Unavailable(_)
                | GitError::HooksFolder
                | GitError::Spawn(_)
                | GitError::BadVersion
                | GitError::TooOld(_),
            ) => Vec::new(),
            Err(_) => vec![String::new()],
        },
        Repo::None | Repo::Without(_) => Vec::new(),
    }
}

/// A final path as people read it: no verbatim prefix.
pub fn shown_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if rest.as_bytes().get(1) == Some(&b':') => rest.to_owned(),
        _ => text.into_owned(),
    }
}
