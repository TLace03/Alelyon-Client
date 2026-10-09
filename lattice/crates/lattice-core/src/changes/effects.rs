//! Checkpoints around commands, and what a command changed
//! (the chat core's spec §8.1, §8.4, §7.5's X14). Not a port.
//!
//! [`CommandGuards`] is what `exec::run` needs around a command
//! (`CommandHooks`):
//! - **before the spawn** (X11), the conversation's writer lease and
//!   checkpoint **B** (`BeforeCommand`); if either cannot be had, the command
//!   does not run;
//! - **after exit, stop or timeout**, checkpoint **A** (`AfterCommand`), then,
//!   in a git folder, `git diff-tree -r -z --no-renames --raw B A` (plumbing:
//!   no external diff, no `textconv`) lists the paths the command added,
//!   modified or deleted. They are recorded as one `command_effect` item and,
//!   for each path, a change with `origin: Command` in state `OnDisk`
//!   (already on disk), whose base is the file at B and whose new bytes are
//!   the file at A (`git cat-file blob`, no filter). The model reads the list,
//!   and "Files git ignores are not tracked here." A path either checkpoint
//!   omitted (too large, past the untracked cap, a link) is not listed as
//!   changed: it is named as not tracked, so an omitted file is never taken
//!   for a created or deleted one.
//! - **Without git** a command's checkpoints are exposed (§8.3): Lattice
//!   cannot see what it changed, there is no effect, and the model is told.
//!
//! These run while the command still holds the folder's one command slot
//! (X14), so no Keep (refused while a command runs) and no other command can
//! land between B and A: a command's effect never lists the reader's kept
//! edit.
//!
//! The reader's review of an effect (`staging::review`):
//! - **Keep** only acknowledges it (`Kept`) and writes nothing;
//! - **Undo** stages the inverse with [`stage_undo`]: a `CommandUndo` change
//!   from the file as it is now back to its bytes at B, or, for a file the
//!   command created, a deletion whose Keep moves it aside (ND2). That change
//!   is reviewed and kept like any other (a checkpoint first, KP1's base copy,
//!   the expected-hash write, refused while a command runs).

use std::collections::BTreeSet;
use std::ffi::OsStr;

use lattice_protocol::conversation::{
    CallId, ChangeKind, ChangeOrigin, ChangeState, ChangedPath, CheckpointId, CheckpointKind,
    CheckpointReason, PathChange,
};

use super::checkpoint::Checkpoints;
use super::restore::{Back, blob, change_id, stage_back};
use crate::convo::item::{Item, NewState, StagedChange};
use crate::exec::run::{AfterCommand, CommandHooks, EffectSummary};
use crate::git::dotgit::{self, Repo};
use crate::git::runner::{Extra, GitRunner};
use crate::policy::{Lease, PathClass};
use crate::staging::{Base, Snapshot, Staging};
use crate::workspace::Workspace;
use crate::workspace::lease::WriterLease;
use crate::workspace::paths::Want;

/// Told after every command whose effect Lattice saw.
pub const IGNORED_NOTE: &str = "Files git ignores are not tracked here.";
/// Told after a command in a folder without git.
pub const EXPOSED_NOTE: &str = "This folder has no git, so Lattice cannot see what the command changed; a restore across it is incomplete.";
/// Told when the after-command checkpoint or the comparison failed.
pub const AFTER_FAILED: &str = "Lattice could not record the folder's state after the command, so what it changed is not listed.";
/// The most changed paths named to the model.
pub const MAX_NAMED: usize = 50;

/// A command's guards over one folder.
pub struct CommandGuards<'a> {
    pub checkpoints: &'a Checkpoints,
    pub workspace: &'a Workspace,
    pub runner: &'a GitRunner,
    pub staging: &'a Staging,
    /// The conversation's writer lease (§6.5): taken before the spawn.
    pub lease: &'a WriterLease,
}

impl CommandHooks for CommandGuards<'_> {
    fn lease(&self) -> Lease {
        self.lease.take()
    }

    fn before(&self, call: &CallId) -> Result<Option<CheckpointId>, String> {
        self.checkpoints
            .take(
                self.workspace,
                self.runner,
                CheckpointReason::BeforeCommand {
                    call_id: call.clone(),
                },
                &[],
            )
            .map(|taken| Some(taken.id))
            .map_err(|error| error.sentence())
    }

    fn after(&self, call: &CallId, before: Option<CheckpointId>) -> AfterCommand {
        effect(self, call, before).unwrap_or_else(|_| AfterCommand {
            notes: vec![AFTER_FAILED.to_owned()],
            effect: None,
        })
    }
}

/// One path of a `diff-tree --raw` answer.
struct Raw {
    old: String,
    new: String,
    status: u8,
    path: String,
}

fn zero(object: &str) -> bool {
    object.bytes().all(|b| b == b'0')
}

/// Parse `diff-tree -r -z --raw`: `:<mode> <mode> <old> <new> <status>\0<path>\0`.
fn parse_raw(stdout: &[u8]) -> Option<Vec<Raw>> {
    let mut out = Vec::new();
    let mut fields = stdout.split(|byte| *byte == 0);
    while let Some(head) = fields.next() {
        if head.is_empty() {
            continue;
        }
        let head = std::str::from_utf8(head).ok()?;
        let path = std::str::from_utf8(fields.next()?).ok()?;
        let parts: Vec<&str> = head.trim_start_matches(':').split(' ').collect();
        if parts.len() != 5 {
            return None;
        }
        out.push(Raw {
            old: parts[2].to_owned(),
            new: parts[3].to_owned(),
            status: *parts[4].as_bytes().first()?,
            path: path.to_owned(),
        });
    }
    Some(out)
}

/// Checkpoint A, the comparison with B, and the record (see the header).
fn effect(
    guards: &CommandGuards<'_>,
    call: &CallId,
    before: Option<CheckpointId>,
) -> Result<AfterCommand, ()> {
    let before = before.and_then(|id| guards.checkpoints.get(id)).ok_or(())?;
    let after = guards
        .checkpoints
        .take(
            guards.workspace,
            guards.runner,
            CheckpointReason::AfterCommand {
                call_id: call.clone(),
            },
            &[],
        )
        .map_err(|_| ())?;
    if before.kind == CheckpointKind::Copies || after.kind == CheckpointKind::Copies {
        return Ok(AfterCommand {
            notes: vec![EXPOSED_NOTE.to_owned()],
            effect: None,
        });
    }
    let (Some(b), Some(a)) = (before.commit.clone(), after.commit.clone()) else {
        return Err(());
    };
    let Repo::Git(local) = dotgit::inspect(&guards.workspace.root, guards.runner.env()) else {
        return Err(());
    };
    let raw = if a == b {
        Vec::new()
    } else {
        let out = guards
            .runner
            .run(
                &local,
                &[
                    OsStr::new("diff-tree"),
                    OsStr::new("-r"),
                    OsStr::new("-z"),
                    OsStr::new("--no-renames"),
                    OsStr::new("--raw"),
                    OsStr::new(b.as_str()),
                    OsStr::new(a.as_str()),
                ],
                &Extra::default(),
            )
            .map_err(|_| ())?;
        if out.status != 0 {
            return Err(());
        }
        parse_raw(&out.stdout).ok_or(())?
    };
    let mut omitted: BTreeSet<String> = BTreeSet::new();
    for id in [before.id, after.id] {
        for path in guards.checkpoints.omitted(id).map_err(|_| ())? {
            omitted.insert(path.path);
        }
    }
    let mut files: Vec<ChangedPath> = Vec::new();
    let mut staged: Vec<StagedChange> = Vec::new();
    let mut not_tracked: Vec<String> = Vec::new();
    for entry in raw {
        if omitted.contains(&entry.path) {
            not_tracked.push(entry.path);
            continue;
        }
        let (change, kind) = match entry.status {
            b'A' => (PathChange::Added, ChangeKind::Create),
            b'D' => (PathChange::Deleted, ChangeKind::Delete),
            b'M' | b'T' => (PathChange::Modified, ChangeKind::Overwrite),
            _ => continue,
        };
        let old = if zero(&entry.old) {
            None
        } else {
            Some(blob(guards.runner, &local, &entry.old).map_err(|_| ())?)
        };
        let new = if zero(&entry.new) {
            None
        } else {
            Some(blob(guards.runner, &local, &entry.new).map_err(|_| ())?)
        };
        let base = old.map_or_else(Base::absent, Base::of_bytes);
        let authority = guards
            .workspace
            .with_rules(guards.runner, |rules| {
                rules.resolve(&entry.path, Want::MayCreate)
            })
            .ok()
            .and_then(Result::ok)
            .is_some_and(|resolved| resolved.class == PathClass::Authority);
        let record = StagedChange {
            id: change_id(guards.staging),
            path: entry.path.clone(),
            kind,
            base: base.state.clone(),
            new: match &new {
                Some(bytes) => NewState::Bytes {
                    blob: crate::sha::sha256_hex(bytes),
                },
                None => NewState::Deleted,
            },
            ops: Vec::new(),
            authority,
            origin: ChangeOrigin::Command { call: call.clone() },
            state: ChangeState::OnDisk,
        };
        guards
            .staging
            .update(Snapshot {
                change: record.clone(),
                base: base.bytes,
                new,
                base_lines: base.lines,
                revision: 0,
            })
            .map_err(|_| ())?;
        staged.push(record);
        files.push(ChangedPath {
            path: entry.path,
            change,
        });
    }
    guards
        .staging
        .sidecar()
        .append(&Item::CommandEffect {
            call_id: call.clone(),
            before: before.id,
            after: after.id,
            files: files.clone(),
        })
        .map_err(|_| ())?;
    let mut notes = Vec::new();
    if files.is_empty() {
        notes.push("The command changed no file Lattice tracks in this folder.".to_owned());
    } else {
        let named: Vec<String> = files
            .iter()
            .take(MAX_NAMED)
            .map(|file| {
                let how = match file.change {
                    PathChange::Added => "added",
                    PathChange::Modified => "modified",
                    PathChange::Deleted => "deleted",
                };
                format!("{} ({how})", file.path)
            })
            .collect();
        let more = files.len().saturating_sub(MAX_NAMED);
        let mut line = format!(
            "The command changed {} file{} on disk: {}",
            files.len(),
            if files.len() == 1 { "" } else { "s" },
            named.join(", ")
        );
        if more > 0 {
            line.push_str(&format!(", and {more} more"));
        }
        line.push_str(". The user can undo them.");
        notes.push(line);
    }
    if !not_tracked.is_empty() {
        notes.push(format!(
            "Not tracked (too large, too many new files, or links): {}.",
            not_tracked.join(", ")
        ));
    }
    notes.push(IGNORED_NOTE.to_owned());
    Ok(AfterCommand {
        notes,
        effect: Some(EffectSummary {
            before: before.id,
            after: after.id,
            files,
        }),
    })
}

/// Why a command effect's Undo staged nothing.
pub const UNDO_UNREADABLE: &str =
    "Lattice could not read the file's earlier bytes back, so nothing was staged.";

/// Undo of a command effect (§8.4): stage the inverse as a `CommandUndo`
/// change (the file now, back to its bytes before the command; a file the
/// command created, as a deletion), and mark the effect `Undone`. Writes
/// nothing to the folder. `Ok(None)` when the file already is as it was.
pub fn stage_undo(
    workspace: &Workspace,
    runner: &GitRunner,
    staging: &Staging,
    effect: &str,
) -> Result<Option<StagedChange>, String> {
    let Some(mut snapshot) = staging.snapshot(effect) else {
        return Err("There is no such change.".to_owned());
    };
    let ChangeOrigin::Command { call } = snapshot.change.origin.clone() else {
        return Err("That change is not a command's.".to_owned());
    };
    if snapshot.change.state != ChangeState::OnDisk {
        return Err("That change is no longer waiting for review.".to_owned());
    }
    let target = match &snapshot.change.base {
        crate::convo::item::BaseState::Absent => None,
        crate::convo::item::BaseState::Present { .. } => Some(
            snapshot
                .base
                .clone()
                .ok_or_else(|| UNDO_UNREADABLE.to_owned())?,
        ),
    };
    let staged = match stage_back(
        workspace,
        runner,
        staging,
        &snapshot.change.path,
        target,
        ChangeKind::CommandUndo,
        ChangeOrigin::CommandUndo { call },
    )
    .map_err(|error| error.sentence())?
    {
        Back::Staged(change) => Some(change),
        Back::Same => None,
        Back::LeftAlone(reason) => return Err(reason),
    };
    snapshot.change.state = ChangeState::Undone;
    staging
        .update(snapshot)
        .map_err(|error| error.sentence().to_owned())?;
    Ok(staged)
}
