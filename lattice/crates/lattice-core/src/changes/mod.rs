//! Checkpoints and restore (the chat core's spec §8). Not a port: the web
//! service's `workspaces/capture.py` and the Agent tab's `safety.py` undo net
//! are the references for the shape and the caps.
//!
//! - [`checkpoint`]: a conversation's checkpoints. In a git workspace, a
//!   filter-free, hook-free, network-free snapshot of the folder's non-ignored
//!   files as a commit under `refs/lattice/chat/<conversation>/<n>`, its
//!   object ids from `git hash-object --no-filters` behind a racily-clean
//!   cache, every omitted path recorded by name. Without git, first-touch
//!   copies of the files about to be written, and a command marks its
//!   checkpoint `exposed`.
//! - [`restore`]: restore to a checkpoint **stages** the difference as
//!   `Restore` changes and writes nothing; a path the checkpoint omitted is
//!   left alone, never staged as absent; a file made after the checkpoint is
//!   staged as a deletion whose Keep moves it aside (ND2).
//!
//! - [`effects`]: the checkpoints before and after each approved command,
//!   and what the command changed in a git folder, listed as changes already
//!   on disk that the reader acknowledges or undoes (a `CommandUndo` change,
//!   reviewed like any other).
//!
//! [`FolderGuards`] is what a review's Keep needs from the rest of the core
//! (`staging::review::KeepGuards`): the before-Keep checkpoint taken here, the
//! conversation's writer lease (`workspace::lease`, row E5), and the command
//! gate: the process's one command slot per workspace (`exec::run`'s
//! `CommandSlots`, X14), so a Keep waits while a command runs in the folder.

use lattice_protocol::conversation::{CheckpointId, CheckpointReason};

use crate::exec::run::CommandSlots;
use crate::git::runner::GitRunner;
use crate::policy::Lease;
use crate::staging::review::KeepGuards;
use crate::workspace::Workspace;
use crate::workspace::lease::WriterLease;

pub mod checkpoint;
pub mod effects;
pub mod restore;

#[cfg(test)]
mod checkpoint_tests;
#[cfg(test)]
mod effects_tests;
#[cfg(test)]
mod restore_tests;

use checkpoint::Checkpoints;

/// A review's guards over one folder: the real checkpoint (this module), the
/// conversation's writer lease, and the real command gate (X14).
pub struct FolderGuards<'a> {
    pub checkpoints: &'a Checkpoints,
    pub workspace: &'a Workspace,
    pub runner: &'a GitRunner,
    /// The conversation's writer lease for this folder (§6.5): taken at the
    /// first Keep if it is free.
    pub lease: &'a WriterLease,
    /// The process's running commands, one per workspace (X14).
    pub commands: &'a CommandSlots,
}

impl KeepGuards for FolderGuards<'_> {
    fn command_running(&self) -> bool {
        self.commands.running(&self.workspace.id)
    }

    fn lease(&self) -> Lease {
        self.lease.take()
    }

    fn checkpoint(
        &self,
        paths: &[String],
        reason: CheckpointReason,
    ) -> Result<CheckpointId, String> {
        self.checkpoints
            .take(self.workspace, self.runner, reason, paths)
            .map(|taken| taken.id)
            .map_err(|error| error.sentence())
    }
}
