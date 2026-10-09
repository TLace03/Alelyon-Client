//! Standing approvals: one exact command, in one folder of one workspace
//! (the chat core's spec §7.6 X3, X4; §9.4). Not a port.
//!
//! - **Where.** Only `<native>/chat/permissions/<workspace-id>.json`:
//!   `{"v": 1, "records": [{"path": <the workspace's canonical path>,
//!   "entry": AllowEntry}]}`. Nothing in the folder is read to decide it.
//! - **How an entry is made.** Only through [`Permissions::allow_always`],
//!   which asks the reader in the native `AllowAlways` dialog
//!   (`ConfirmPort`, CP1–CP3) and writes nothing unless the reader confirms.
//!   The entry is `Exact`: its argv, the resolved program's absolute path,
//!   the workspace-relative working folder, `created_at`. `Prefix` does not
//!   exist in this increment (X3, X13).
//! - **Matching (X4)** ([`Permissions::matching`]): the command is eligible
//!   (X1), its `argv[0]` resolves (X2) to the entry's program (compared as
//!   canonical paths, without case), its arguments equal the entry's, and its
//!   working folder **equals** the entry's. The entry is not revoked, and its
//!   record matches the workspace by both id and path (§4.1), so a folder
//!   that was replaced, renamed or moved is asked about again. The gates
//!   (no staged change waiting, no other command running) are the policy
//!   engine's (§9.2), which the caller asks with the match.
//! - **Revoking** is narrowing: no dialog. The entry stays listed with its
//!   `revoked_at` (ND1) and no longer matches.
//! - A permissions file that cannot be read or parsed matches nothing and is
//!   never overwritten.

use std::path::PathBuf;
use std::sync::Mutex;

use lattice_protocol::conversation::{AllowEntry, MatchScope};
use serde::{Deserialize, Serialize};

use super::command::{eligible, may_be_entry};
use super::resolve::Resolved;
use super::spawn::comparable;
use crate::clock::Clock;
use crate::fsx;
use crate::ports::{ConfirmRequest, Confirmer, Initiated};
use crate::staging::path_key;
use crate::state::StateRoot;
use crate::workspace::{Workspace, WorkspaceKey, shown_path};

/// The permissions file's version.
pub const PERMISSIONS_VERSION: u32 = 1;
/// The largest permissions file read.
pub const MAX_PERMISSIONS_FILE: u64 = 4 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Record {
    path: String,
    entry: AllowEntry,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct PermissionsFile {
    v: u32,
    records: Vec<Record>,
}

/// Why no entry was made or changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PermissionError {
    /// The command can never be an entry (X1–X3), with the sentence.
    NotAllowed(String),
    /// The reader did not confirm; nothing was written.
    NotConfirmed,
    /// The permissions file cannot be read or parsed: nothing was changed.
    Unreadable,
    /// The file could not be written.
    Write,
    /// No such entry in this workspace.
    NoSuchEntry,
}

impl PermissionError {
    pub fn sentence(&self) -> String {
        match self {
            Self::NotAllowed(sentence) => sentence.clone(),
            Self::NotConfirmed => "The command was not allowed; it will ask each time.".to_owned(),
            Self::Unreadable => {
                "Lattice's record of allowed commands could not be read, so nothing was changed."
                    .to_owned()
            }
            Self::Write => "Lattice could not record the allowed command.".to_owned(),
            Self::NoSuchEntry => "There is no such allowed command in this folder.".to_owned(),
        }
    }
}

/// What a standing approval would be: the command's argv as X1 split it,
/// the program it resolved to, and its workspace-relative working folder.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub argv: Vec<String>,
    pub resolved: Resolved,
    /// Workspace-relative, forward slashes; `""` is the root.
    pub cwd: String,
}

/// The standing approvals of every workspace.
pub struct Permissions {
    dir: PathBuf,
    clock: Clock,
    write: Mutex<()>,
}

/// The program path an entry stores and X4 compares: the final path, as
/// people read it.
fn program_text(resolved: &Resolved) -> String {
    shown_path(&resolved.real)
}

fn same_program(entry: &str, resolved: &Resolved) -> bool {
    let entry = comparable(entry);
    entry == comparable(&resolved.real.to_string_lossy())
        || entry == comparable(&resolved.path.to_string_lossy())
}

fn same_cwd(a: &str, b: &str) -> bool {
    path_key(a.trim_matches('/')) == path_key(b.trim_matches('/'))
}

impl Permissions {
    pub fn new(state: &StateRoot, clock: Clock) -> Self {
        Self {
            dir: state.native_chat_dir().join("permissions"),
            clock,
            write: Mutex::default(),
        }
    }

    /// `<native>/chat/permissions/<workspace-id>.json`.
    pub fn file(&self, workspace: &Workspace) -> PathBuf {
        self.dir.join(format!("{}.json", workspace.id))
    }

    fn read(&self, workspace: &Workspace) -> Result<PermissionsFile, PermissionError> {
        let file = self.file(workspace);
        let bytes = match std::fs::read(&file) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(PermissionsFile {
                    v: PERMISSIONS_VERSION,
                    records: Vec::new(),
                });
            }
            Err(_) => return Err(PermissionError::Unreadable),
        };
        if bytes.len() as u64 > MAX_PERMISSIONS_FILE {
            return Err(PermissionError::Unreadable);
        }
        let parsed: PermissionsFile =
            serde_json::from_slice(&bytes).map_err(|_| PermissionError::Unreadable)?;
        if parsed.v != PERMISSIONS_VERSION {
            return Err(PermissionError::Unreadable);
        }
        Ok(parsed)
    }

    fn write_file(
        &self,
        workspace: &Workspace,
        file: &PermissionsFile,
    ) -> Result<(), PermissionError> {
        std::fs::create_dir_all(&self.dir).map_err(|_| PermissionError::Write)?;
        let bytes = serde_json::to_vec_pretty(file).map_err(|_| PermissionError::Write)?;
        fsx::atomic_write(&self.file(workspace), &bytes).map_err(|_| PermissionError::Write)
    }

    fn mine(workspace: &Workspace, record: &Record) -> bool {
        WorkspaceKey {
            id: record.entry.workspace.clone(),
            path: record.path.clone(),
        }
        .matches(&workspace.key())
    }

    /// This workspace's entries, revoked ones included (`permissions`, §4.3).
    pub fn entries(&self, workspace: &Workspace) -> Result<Vec<AllowEntry>, PermissionError> {
        Ok(self
            .read(workspace)?
            .records
            .into_iter()
            .filter(|record| Self::mine(workspace, record))
            .map(|record| record.entry)
            .collect())
    }

    /// X4: the entry `text` matches in `cwd`, when one does. `resolve` is
    /// X2's resolution of a bare name (the caller's, with its environment).
    pub fn matching(
        &self,
        workspace: &Workspace,
        text: &str,
        cwd: &str,
        resolve: impl FnOnce(&str) -> Option<Resolved>,
    ) -> Option<AllowEntry> {
        let argv = eligible(text).ok()?;
        let resolved = resolve(&argv[0])?;
        let entries = self.entries(workspace).ok()?;
        entries.into_iter().find(|entry| {
            entry.revoked_at.is_none()
                && entry.scope == MatchScope::Exact
                && entry.argv.len() == argv.len()
                && entry.argv[1..] == argv[1..]
                && same_program(&entry.program, &resolved)
                && same_cwd(&entry.cwd, cwd)
        })
    }

    /// Make a standing entry, only after the reader confirms it in the
    /// native `AllowAlways` dialog. An identical live entry is given back
    /// without a dialog. `key` names the request for CP3 (a fresh pending
    /// call is a fresh key).
    pub async fn allow_always(
        &self,
        confirmer: &Confirmer,
        workspace: &Workspace,
        candidate: &Candidate,
        key: &str,
    ) -> Result<AllowEntry, PermissionError> {
        if !may_be_entry(&candidate.argv, &candidate.resolved) {
            return Err(PermissionError::NotAllowed(
                "That program runs other programs or scripts, so it cannot be allowed always."
                    .to_owned(),
            ));
        }
        let program = program_text(&candidate.resolved);
        let existing = self.entries(workspace)?.into_iter().find(|entry| {
            entry.revoked_at.is_none()
                && entry.argv == candidate.argv
                && same_program(&entry.program, &candidate.resolved)
                && same_cwd(&entry.cwd, &candidate.cwd)
        });
        if let Some(entry) = existing {
            return Ok(entry);
        }
        let confirmed = confirmer
            .ask(
                key,
                ConfirmRequest::AllowAlways {
                    workspace: shown_path(&workspace.root),
                    argv: candidate.argv.clone(),
                    program: program.clone(),
                    cwd: candidate.cwd.clone(),
                },
                Initiated::Page,
            )
            .await;
        if !confirmed {
            return Err(PermissionError::NotConfirmed);
        }
        let _write = self
            .write
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut file = self.read(workspace)?;
        let entry = AllowEntry {
            id: format!("al_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]),
            workspace: workspace.id.clone(),
            argv: candidate.argv.clone(),
            program,
            cwd: candidate.cwd.clone(),
            created_at: (self.clock)(),
            scope: MatchScope::Exact,
            revoked_at: None,
        };
        file.records.push(Record {
            path: workspace.root.to_string_lossy().into_owned(),
            entry: entry.clone(),
        });
        self.write_file(workspace, &file)?;
        Ok(entry)
    }

    /// Revoke an entry: no dialog (narrowing). It stays listed.
    pub fn revoke(&self, workspace: &Workspace, id: &str) -> Result<(), PermissionError> {
        let _write = self
            .write
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut file = self.read(workspace)?;
        let now = (self.clock)();
        let mut found = false;
        for record in &mut file.records {
            if record.entry.id == id && Self::mine(workspace, record) {
                found = true;
                if record.entry.revoked_at.is_none() {
                    record.entry.revoked_at = Some(now);
                }
            }
        }
        if !found {
            return Err(PermissionError::NoSuchEntry);
        }
        self.write_file(workspace, &file)
    }
}
