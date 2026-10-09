//! Folder trust (the chat core's spec §6.3, FT1–FT5). Not a port.
//!
//! - **FT1.** Trusting a folder asks the reader through the native dialog
//!   (`ConfirmPort(TrustFolder)`, under CP1–CP3), naming the rules files found
//!   and saying that Agent mode will run git there and write snapshot objects
//!   and refs into its `.git`.
//! - **FT2.** Trust lives only in `<native>/chat/trust.json`:
//!   `{"v": 1, "folders": [{"id", "path", "trusted_at", "rules_off"}],
//!   "revoked": [{"id", "path", "revoked_at"}]}`. A record matches only when
//!   both the folder's id and its canonical path match (§4.1). It is never
//!   stored in the folder, and nothing in the folder is read to decide it.
//! - **FT3.** An untrusted folder allows Ask mode with the read tools; rules
//!   files, Agent mode and checkpoints wait for trust (the rules loader and
//!   the policy engine read [`TrustStore::state`]).
//! - **FT4.** Nothing in the folder grants anything: only [`TrustStore::trust`]
//!   writes a trust record, after the reader's click.
//! - **FT5.** Revoking appends to `revoked` (ND1: no record is removed); the
//!   newest of a folder's trust and revoke records decides, so a revoked
//!   folder is untrusted again until it is trusted again.
//!
//! On FAT and exFAT a file id is a directory entry's position and is reused,
//! so trust there is kept for this session only, in memory, and never written
//! (§4.1). A `trust.json` that cannot be read or parsed trusts nothing and is
//! never overwritten.
//!
//! The file operations are small and synchronous; the service runs them on
//! the blocking pool. Only the reader's answer is awaited.

use std::path::PathBuf;
use std::sync::Mutex;

use lattice_protocol::conversation::TrustState;
use lattice_sys::fs::{Access, file_system_name};
use serde::{Deserialize, Serialize};

use super::{Workspace, WorkspaceKey, shown_path};
use crate::clock::Clock;
use crate::fsx;
use crate::localfs::{LinkRule, open_walk};
use crate::ports::{ConfirmRequest, Confirmer, Initiated};
use crate::state::StateRoot;

/// The trust file's version.
pub const TRUST_VERSION: u32 = 1;
/// The largest trust file read.
pub const MAX_TRUST_FILE: u64 = 4 * 1024 * 1024;

/// One trust record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FolderRecord {
    pub id: String,
    pub path: String,
    pub trusted_at: f64,
    #[serde(default)]
    pub rules_off: Vec<String>,
}

/// One revocation record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RevokedRecord {
    pub id: String,
    pub path: String,
    pub revoked_at: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct TrustFile {
    v: u32,
    folders: Vec<FolderRecord>,
    revoked: Vec<RevokedRecord>,
}

/// Why trust was not changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrustError {
    /// `trust.json` cannot be read or parsed: nothing was changed.
    Unreadable,
    /// The reader did not confirm.
    NotConfirmed,
    /// The record could not be written.
    Write,
}

impl TrustError {
    pub fn sentence(&self) -> &'static str {
        match self {
            Self::Unreadable => "Lattice's trust record could not be read, so nothing was changed.",
            Self::NotConfirmed => "The folder was not trusted.",
            Self::Write => "Lattice could not record the folder's trust.",
        }
    }
}

/// Trust records, on disk and (for FAT and exFAT volumes) in memory.
pub struct TrustStore {
    file: PathBuf,
    clock: Clock,
    session: Mutex<Vec<FolderRecord>>,
    write: Mutex<()>,
    /// The file system name to assume, in tests only.
    #[cfg(test)]
    pub(crate) file_system: Option<String>,
}

fn key_of(id: &str, path: &str) -> WorkspaceKey {
    WorkspaceKey {
        id: id.to_owned(),
        path: path.to_owned(),
    }
}

impl TrustStore {
    pub fn new(state: &StateRoot, clock: Clock) -> Self {
        Self {
            file: state.native_chat_dir().join("trust.json"),
            clock,
            session: Mutex::default(),
            write: Mutex::default(),
            #[cfg(test)]
            file_system: None,
        }
    }

    /// `<native>/chat/trust.json`.
    pub fn file(&self) -> &std::path::Path {
        &self.file
    }

    fn read(&self) -> Result<TrustFile, TrustError> {
        let bytes = match std::fs::read(&self.file) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(TrustFile {
                    v: TRUST_VERSION,
                    ..TrustFile::default()
                });
            }
            Err(_) => return Err(TrustError::Unreadable),
        };
        if bytes.len() as u64 > MAX_TRUST_FILE {
            return Err(TrustError::Unreadable);
        }
        let file: TrustFile = serde_json::from_slice(&bytes).map_err(|_| TrustError::Unreadable)?;
        if file.v != TRUST_VERSION {
            return Err(TrustError::Unreadable);
        }
        Ok(file)
    }

    fn write_file(&self, file: &TrustFile) -> Result<(), TrustError> {
        let parent = self.file.parent().ok_or(TrustError::Write)?;
        std::fs::create_dir_all(parent).map_err(|_| TrustError::Write)?;
        let bytes = serde_json::to_vec_pretty(file).map_err(|_| TrustError::Write)?;
        fsx::atomic_write(&self.file, &bytes).map_err(|_| TrustError::Write)
    }

    /// The latest matching trust record, and whether a revoke came after it.
    fn decide(&self, workspace: &Workspace) -> (TrustState, Option<FolderRecord>) {
        let key = workspace.key();
        let session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .rev()
            .find(|record| key_of(&record.id, &record.path).matches(&key))
            .cloned();
        if let Some(record) = session {
            return (TrustState::Trusted, Some(record));
        }
        let Ok(file) = self.read() else {
            return (TrustState::Untrusted, None);
        };
        let trusted = file
            .folders
            .iter()
            .filter(|record| key_of(&record.id, &record.path).matches(&key))
            .cloned()
            .collect::<Vec<_>>();
        let latest_trust = trusted
            .iter()
            .map(|record| record.trusted_at)
            .fold(f64::NEG_INFINITY, f64::max);
        let latest_revoke = file
            .revoked
            .iter()
            .filter(|record| key_of(&record.id, &record.path).matches(&key))
            .map(|record| record.revoked_at)
            .fold(f64::NEG_INFINITY, f64::max);
        match (trusted.last(), latest_revoke.is_finite()) {
            (None, false) => (TrustState::Untrusted, None),
            (None, true) => (TrustState::Revoked, None),
            // A tie goes to the revocation (fail closed).
            (Some(_), true) if latest_revoke >= latest_trust => (TrustState::Revoked, None),
            (Some(record), _) => (TrustState::Trusted, Some(record.clone())),
        }
    }

    /// The folder's trust now.
    pub fn state(&self, workspace: &Workspace) -> TrustState {
        self.decide(workspace).0
    }

    /// The rules the reader switched off for this folder (from its latest
    /// trust record); empty when it is not trusted.
    pub fn rules_off(&self, workspace: &Workspace) -> Vec<String> {
        self.decide(workspace)
            .1
            .map(|record| record.rules_off)
            .unwrap_or_default()
    }

    fn on_fat(&self, workspace: &Workspace) -> bool {
        #[cfg(test)]
        if let Some(name) = &self.file_system {
            return is_fat(name);
        }
        open_walk(&workspace.root, Access::Attributes, LinkRule::AnyLocal)
            .ok()
            .and_then(|walked| file_system_name(&walked.file).ok())
            .is_some_and(|name| is_fat(&name))
    }

    /// FT1: ask the reader to trust `workspace`, naming `will_read` (the rules
    /// files found) and whether git will write into its `.git`. Nothing is
    /// written unless the reader confirms. An already trusted folder is not
    /// asked about again.
    pub async fn trust(
        &self,
        workspace: &Workspace,
        confirmer: &Confirmer,
        initiated: Initiated,
        will_read: Vec<String>,
        git_writes: bool,
    ) -> Result<TrustState, TrustError> {
        if self.state(workspace) == TrustState::Trusted {
            return Ok(TrustState::Trusted);
        }
        // A file that cannot be read is never overwritten, so do not ask.
        self.read()?;
        let path = shown_path(&workspace.root);
        let confirmed = confirmer
            .ask(
                &format!("trust:{}:{path}", workspace.id),
                ConfirmRequest::TrustFolder {
                    path,
                    will_read,
                    git_writes,
                },
                initiated,
            )
            .await;
        if !confirmed {
            return Err(TrustError::NotConfirmed);
        }
        self.record(workspace, Vec::new())?;
        Ok(TrustState::Trusted)
    }

    /// Append a trust record (after the reader's confirmation only).
    fn record(&self, workspace: &Workspace, rules_off: Vec<String>) -> Result<(), TrustError> {
        let record = FolderRecord {
            id: workspace.id.clone(),
            path: workspace.root.to_string_lossy().into_owned(),
            trusted_at: (self.clock)(),
            rules_off,
        };
        if self.on_fat(workspace) {
            self.session
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(record);
            return Ok(());
        }
        let _guard = self
            .write
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut file = self.read()?;
        file.folders.push(record);
        self.write_file(&file)
    }

    /// FT5: revoke trust. A record is appended; none is removed.
    pub fn revoke(&self, workspace: &Workspace) -> Result<(), TrustError> {
        let key = workspace.key();
        self.session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retain(|record| !key_of(&record.id, &record.path).matches(&key));
        let _guard = self
            .write
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut file = self.read()?;
        file.revoked.push(RevokedRecord {
            id: workspace.id.clone(),
            path: workspace.root.to_string_lossy().into_owned(),
            revoked_at: (self.clock)(),
        });
        self.write_file(&file)
    }

    /// Switch rules off (or back on) for a trusted folder: a new trust record
    /// with the same trust time and these `rules_off` (`"*"`: every rules file
    /// of the folder).
    pub fn set_rules_off(
        &self,
        workspace: &Workspace,
        rules_off: Vec<String>,
    ) -> Result<(), TrustError> {
        let (state, record) = self.decide(workspace);
        let Some(record) = record.filter(|_| state == TrustState::Trusted) else {
            return Err(TrustError::NotConfirmed);
        };
        let updated = FolderRecord {
            rules_off,
            ..record
        };
        if self.on_fat(workspace) {
            self.session
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(updated);
            return Ok(());
        }
        let _guard = self
            .write
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut file = self.read()?;
        file.folders.push(updated);
        self.write_file(&file)
    }
}

/// FAT, FAT12, FAT16, FAT32 or exFAT.
fn is_fat(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    upper.starts_with("FAT") || upper == "EXFAT"
}
