//! Restore to a checkpoint: staged, reviewed, moved aside
//! (the chat core's spec §8.5). Not a port.
//!
//! `restore` **stages** the restore and writes nothing to the folder. For
//! each path that differs between the checkpoint and the folder now:
//! - in the checkpoint, different now: a `Restore` change whose new bytes are
//!   the checkpoint's (`git cat-file blob`, which applies no filter; or the
//!   copy, without git);
//! - in the checkpoint, absent now: a `Restore` creation;
//! - absent from the checkpoint, present now: a `Restore` deletion, whose
//!   Keep moves the file aside (ND2), never unlinks it.
//!
//! The changes are reviewed like any other (`staging::review`); their Keep
//! is preceded by a checkpoint, so a restore can itself be undone.
//!
//! What is left alone, and listed:
//! - **every path the checkpoint omitted** (§8.2 step 1): an omitted path is
//!   not an absent one, so a file too large to capture is never staged for a
//!   move aside;
//! - ignored files and paths outside the folder (they are never listed, and
//!   each staged path passes the path rules, WP1–WP11, again);
//! - a path that is now a link, a nested repository or unreadable;
//! - a path with a staged change still waiting for review (one live change
//!   per path, ST2).
//!
//! Without git the checkpoint's copies are used: for each path, the copy made
//! at the restored checkpoint or, failing that, at the first later one (its
//! state before Lattice first wrote it after that point). When a command ran
//! in that range (an `exposed` checkpoint) the result says the restore is
//! incomplete: "Commands ran that Lattice cannot see in a folder without git;
//! their changes are not restored."
//!
//! Messages are never removed: the record only grows.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::io::Read;

use lattice_protocol::conversation::{
    ChangeKind, ChangeOrigin, ChangeState, CheckpointId, CheckpointKind,
};

use super::checkpoint::{CheckpointError, Checkpoints, Copy, OmitWhy};
use crate::convo::item::{NewState, StagedChange};
use crate::git::dotgit::{self, Repo};
use crate::git::runner::{Extra, GitError, GitRunner};
use crate::policy::PathClass;
use crate::sha::sha256_hex;
use crate::staging::{Base, Snapshot, Staging, is_live, is_waiting, path_key};
use crate::tools::read::MAX_FILE_BYTES;
use crate::workspace::Workspace;
use crate::workspace::paths::Want;

/// The sentence of a restore across a command Lattice could not see.
pub const INCOMPLETE: &str =
    "Commands ran that Lattice cannot see in a folder without git; their changes are not restored.";
const WAITING: &str = "A staged change to this file is waiting for review; keep or undo it first.";
const THROUGH_LINK: &str = "This path is reached through a link, so it is left alone.";

/// What a restore works with.
pub struct RestoreContext<'a> {
    pub workspace: &'a Workspace,
    pub runner: &'a GitRunner,
    pub staging: &'a Staging,
    pub checkpoints: &'a Checkpoints,
}

/// A path the restore did not stage, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeftAlone {
    pub path: String,
    pub reason: String,
}

/// What a restore staged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Restored {
    pub to: CheckpointId,
    /// The `Restore` changes, in path order.
    pub staged: Vec<StagedChange>,
    /// Paths the checkpoint omitted, left alone.
    pub omitted: Vec<String>,
    pub left_alone: Vec<LeftAlone>,
    /// [`INCOMPLETE`] when a command ran in the restored range without git.
    pub incomplete: Option<String>,
}

/// Why nothing was staged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RestoreError {
    NoSuchCheckpoint,
    Checkpoint(CheckpointError),
    /// The record could not take a staged change; what was staged before it
    /// stays staged.
    Record,
}

impl RestoreError {
    pub fn sentence(&self) -> String {
        match self {
            Self::NoSuchCheckpoint => "There is no such checkpoint.".to_owned(),
            Self::Checkpoint(error) => error.sentence(),
            Self::Record => {
                "Lattice could not record the restore, so it is not complete.".to_owned()
            }
        }
    }
}

impl From<CheckpointError> for RestoreError {
    fn from(error: CheckpointError) -> Self {
        Self::Checkpoint(error)
    }
}

impl From<GitError> for RestoreError {
    fn from(error: GitError) -> Self {
        Self::Checkpoint(CheckpointError::Git(error))
    }
}

/// Stage the restore of the folder to checkpoint `to` (see the module
/// header). Writes nothing to the folder.
pub fn restore(ctx: &RestoreContext<'_>, to: CheckpointId) -> Result<Restored, RestoreError> {
    let taken = ctx
        .checkpoints
        .get(to)
        .ok_or(RestoreError::NoSuchCheckpoint)?;
    let mut out = Restored {
        to,
        staged: Vec::new(),
        omitted: Vec::new(),
        left_alone: Vec::new(),
        incomplete: None,
    };
    match taken.kind {
        CheckpointKind::Git => restore_git(ctx, to, &taken, &mut out)?,
        CheckpointKind::Copies => restore_copies(ctx, to, &mut out)?,
    }
    out.staged.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn restore_git(
    ctx: &RestoreContext<'_>,
    to: CheckpointId,
    taken: &super::checkpoint::Taken,
    out: &mut Restored,
) -> Result<(), RestoreError> {
    let commit = taken
        .commit
        .clone()
        .ok_or(RestoreError::Checkpoint(CheckpointError::Record))?;
    let local = match dotgit::inspect(&ctx.workspace.root, ctx.runner.env()) {
        Repo::Git(local) => local,
        Repo::Without(without) => {
            return Err(CheckpointError::NotLocal(without.sentence()).into());
        }
        Repo::None => {
            return Err(CheckpointError::NotLocal(
                "This folder is no longer in a git repository.".to_owned(),
            )
            .into());
        }
    };
    let omitted_then: BTreeSet<String> = ctx
        .checkpoints
        .omitted(to)?
        .into_iter()
        .map(|omitted| omitted.path)
        .collect();
    let listing = ctx.runner.run(
        &local,
        &[
            OsStr::new("ls-tree"),
            OsStr::new("-r"),
            OsStr::new("-z"),
            OsStr::new("--full-tree"),
            OsStr::new(commit.as_str()),
        ],
        &Extra::default(),
    )?;
    if listing.status != 0 {
        return Err(GitError::Failed {
            status: listing.status,
            stderr: listing.stderr,
        }
        .into());
    }
    // Path to object id, for the checkpoint's blobs.
    let mut then: BTreeMap<String, String> = BTreeMap::new();
    for record in listing.stdout.split(|byte| *byte == 0) {
        let Ok(text) = std::str::from_utf8(record) else {
            continue;
        };
        let Some((head, path)) = text.split_once('\t') else {
            continue;
        };
        let fields: Vec<&str> = head.split(' ').collect();
        if fields.len() == 3 && fields[1] == "blob" {
            then.insert(path.to_owned(), fields[2].to_owned());
        }
    }
    let now = ctx.checkpoints.list_current(&local, ctx.runner, false)?;
    // Present now but not hashed (too large, past the cap): differ from any
    // object. Links, nested repositories and unreadable files: left alone.
    let mut present_unhashed: BTreeSet<String> = BTreeSet::new();
    let mut untouchable: BTreeMap<String, OmitWhy> = BTreeMap::new();
    for omitted in &now.omitted {
        match omitted.why {
            OmitWhy::TooLarge | OmitWhy::UntrackedCap => {
                present_unhashed.insert(omitted.path.clone());
            }
            why => {
                untouchable.insert(omitted.path.clone(), why);
            }
        }
    }
    let mut paths: BTreeSet<String> = then.keys().cloned().collect();
    paths.extend(now.entries.keys().cloned());
    paths.extend(present_unhashed.iter().cloned());
    paths.extend(untouchable.keys().cloned());
    for path in paths {
        if omitted_then.contains(&path) {
            out.omitted.push(path);
            continue;
        }
        if untouchable.contains_key(&path) {
            out.left_alone.push(LeftAlone {
                path,
                reason: "This path is a link, a nested repository or unreadable now, so it is left alone.".to_owned(),
            });
            continue;
        }
        let want = then.get(&path);
        let have = now.entries.get(&path).map(|(_, object)| object);
        let present = have.is_some() || present_unhashed.contains(&path);
        match (want, have) {
            (Some(want), Some(have)) if want == have => continue,
            (None, _) if !present => continue,
            _ => {}
        }
        let bytes = match want {
            Some(object) => Some(blob(ctx.runner, &local, object)?),
            None => None,
        };
        stage(ctx, to, &path, bytes, out)?;
    }
    Ok(())
}

/// `git cat-file blob <object>`: the bytes as stored, with no filter.
pub(crate) fn blob(
    runner: &GitRunner,
    local: &crate::git::dotgit::LocalRepo,
    object: &str,
) -> Result<Vec<u8>, RestoreError> {
    let out = runner.run(
        local,
        &[
            OsStr::new("cat-file"),
            OsStr::new("blob"),
            OsStr::new(object),
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
    Ok(out.stdout)
}

fn restore_copies(
    ctx: &RestoreContext<'_>,
    to: CheckpointId,
    out: &mut Restored,
) -> Result<(), RestoreError> {
    // Each path's state at `to`: its copy at `to` or at the first later
    // checkpoint that copied it.
    let mut first: BTreeMap<String, (CheckpointId, Copy)> = BTreeMap::new();
    for taken in ctx.checkpoints.all() {
        if taken.id < to || taken.kind != CheckpointKind::Copies {
            continue;
        }
        if taken.exposed {
            out.incomplete = Some(INCOMPLETE.to_owned());
        }
        for entry in ctx.checkpoints.manifest(taken.id)? {
            first
                .entry(entry.path.clone())
                .or_insert((taken.id, entry.copy));
        }
    }
    for (path, (at, copy)) in first {
        match copy {
            Copy::Skipped { .. } => out.omitted.push(path),
            Copy::Absent => stage(ctx, to, &path, None, out)?,
            Copy::Present { .. } => {
                let bytes = ctx.checkpoints.copy_bytes(at, &path)?;
                stage(ctx, to, &path, Some(bytes), out)?;
            }
        }
    }
    Ok(())
}

/// A fresh change id, `ch_<16 hex>`, unused in this conversation.
pub(crate) fn change_id(staging: &Staging) -> String {
    let taken: BTreeSet<String> = staging
        .changes()
        .into_iter()
        .map(|change| change.id)
        .collect();
    loop {
        let id = format!("ch_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]);
        if !taken.contains(&id) {
            return id;
        }
    }
}

/// Stage `path` back to `target` (bytes, or absent), through the path rules,
/// unless it already is so or must be left alone.
fn stage(
    ctx: &RestoreContext<'_>,
    to: CheckpointId,
    path: &str,
    target: Option<Vec<u8>>,
    out: &mut Restored,
) -> Result<(), RestoreError> {
    match stage_back(
        ctx.workspace,
        ctx.runner,
        ctx.staging,
        path,
        target,
        ChangeKind::Restore,
        ChangeOrigin::Restore { to },
    )? {
        Back::Staged(change) => out.staged.push(change),
        Back::Same => {}
        Back::LeftAlone(reason) => out.left_alone.push(LeftAlone {
            path: path.to_owned(),
            reason,
        }),
    }
    Ok(())
}

/// What staging a path back to earlier bytes came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Back {
    /// A change of `kind` was staged.
    Staged(StagedChange),
    /// The file already is so: nothing to stage.
    Same,
    /// Left alone, for this reason.
    LeftAlone(String),
}

/// Stage `path` back to `target` (bytes, or absent) as a change of `kind`
/// from `origin`, through the path rules: a restore's (`Restore`) or a
/// command effect's undo (`CommandUndo`, row E8). Writes nothing to the
/// folder.
pub(crate) fn stage_back(
    workspace: &Workspace,
    runner: &GitRunner,
    staging: &Staging,
    path: &str,
    target: Option<Vec<u8>>,
    kind: ChangeKind,
    origin: ChangeOrigin,
) -> Result<Back, RestoreError> {
    // ST2: held from the check below through the update, as the staging
    // tools hold it, so an agent's staging of this path cannot fall between
    // them and leave two live changes on one path.
    let _one = staging.serial();
    let key = path_key(path);
    if staging.changes().iter().any(|change| {
        path_key(&change.path) == key && (is_live(&change.state) || is_waiting(&change.state))
    }) {
        return Ok(Back::LeftAlone(WAITING.to_owned()));
    }
    let resolved = workspace.with_rules(runner, |rules| rules.resolve(path, Want::MayCreate));
    let mut resolved = match resolved {
        Ok(Ok(resolved)) => resolved,
        Ok(Err(error)) => return Ok(Back::LeftAlone(error.sentence())),
        Err(_) => {
            return Ok(Back::LeftAlone(
                "Lattice could not read this folder's .latticeignore.".to_owned(),
            ));
        }
    };
    if resolved.derived != path || resolved.is_dir {
        return Ok(Back::LeftAlone(THROUGH_LINK.to_owned()));
    }
    let base = match resolved.file.take().filter(|_| resolved.exists) {
        None => Base::absent(),
        Some(file) => {
            let size = file.metadata().map(|meta| meta.len()).unwrap_or(u64::MAX);
            let read = if size <= MAX_FILE_BYTES {
                let mut bytes = Vec::new();
                file.take(MAX_FILE_BYTES + 1)
                    .read_to_end(&mut bytes)
                    .map(|_| Base::of_bytes(bytes))
            } else {
                Base::of_reader(file)
            };
            match read {
                Ok(base) => base,
                Err(_) => {
                    return Ok(Back::LeftAlone(
                        "That file could not be read, so it is left alone.".to_owned(),
                    ));
                }
            }
        }
    };
    let new = match &target {
        Some(bytes) => {
            let sha256 = sha256_hex(bytes);
            if let crate::convo::item::BaseState::Present { sha256: now, .. } = &base.state
                && *now == sha256
            {
                return Ok(Back::Same);
            }
            NewState::Bytes { blob: sha256 }
        }
        None if base.state == crate::convo::item::BaseState::Absent => return Ok(Back::Same),
        None => NewState::Deleted,
    };
    let change = StagedChange {
        id: change_id(staging),
        path: path.to_owned(),
        kind,
        base: base.state,
        new,
        ops: Vec::new(),
        authority: resolved.class == PathClass::Authority,
        origin,
        state: ChangeState::Pending,
    };
    staging
        .update(Snapshot {
            change: change.clone(),
            base: base.bytes,
            new: target,
            base_lines: base.lines,
            revision: 0,
        })
        .map_err(|_| RestoreError::Record)?;
    Ok(Back::Staged(change))
}
