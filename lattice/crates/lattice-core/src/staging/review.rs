//! The reader's review of staged changes: the diff in hunks, Keep and Undo per
//! hunk, per file or for all, and the expected-hash write
//! (the chat core's spec §7.5). Not a port.
//!
//! **Keep** runs, for each change, in this order:
//! 1. **Preconditions,** by the policy engine (§9.2): Agent mode, a trusted
//!    folder, the writer lease held or taken, no command running in the
//!    folder (X14: "A command is still running in this folder; Keep when it
//!    has finished."). An authority file (ST4) is kept only one at a time and
//!    only after the reader confirms it in the native `KeepAuthority` dialog;
//!    Keep All leaves it out. After the dialog the change is read again
//!    under the staging lock, which is held from there through the change's
//!    update: a change the agent staged again, or another review kept, while
//!    the dialog was open is skipped ("changed while you were confirming")
//!    and nothing is written; [`Staging::update`] itself refuses a stale
//!    snapshot.
//!    The path is resolved again at the write (step 3): when it now leads
//!    elsewhere (a folder swapped for a link since staging) or its class is
//!    now authority although the policy checked an ordinary file, nothing is
//!    written and the result is a conflict. Every Keep that writes (an
//!    agent's change, a restore's, a command effect's inverse) passes here.
//! 2. **A checkpoint first** (§8), once for the whole review: if it fails,
//!    nothing is written ("Lattice could not record the folder's state
//!    first, so nothing was written."). The checkpoint is row E4's
//!    (`changes::checkpoint`, through `changes::FolderGuards`), taken for
//!    the paths about to be written, and named `before_restore_keep` when the
//!    review keeps a restore.
//! 3. **The expected hash.** The target is opened through the path rules
//!    (WP10) and read. **KP1:** its bytes are first copied into the
//!    conversation's blobs (content-addressed, write-once) as a base copy,
//!    named by the `reviewed` item, because a checkpoint may omit the file; a
//!    target over 64 MiB is refused with a conflict. When the bytes already
//!    are what the Keep would leave (an earlier Keep of the change wrote them
//!    and stopped before its update), nothing is written and the Keep
//!    finishes. When the bytes are not
//!    the change's base, nothing is written: a change made only of edits is
//!    applied again to the file as it is now (the unique-match rule) and
//!    becomes `Rebased`, to be reviewed again; any other becomes a
//!    `Conflict`.
//! 4. **The bytes:** a whole Keep takes the staged bytes (already in the
//!    base's line ends and byte-order mark, row E2); a hunk Keep takes the
//!    base with only the chosen hunks applied, every line with its own line
//!    end.
//! 5. **The write:** a temporary `.<name>.lattice-<pid>-<n>.tmp` beside the
//!    target, synced, then `ReplaceFileW` **with a backup name**
//!    (`.<name>.lattice-bak-<pid>-<n>`), whose replaced bytes are then moved
//!    into `removed/<n>/` (ND2), never unlinked. Its two partial failures:
//!    1176 leaves the target as it was (the temporary, whose bytes are the
//!    staged blob, is removed as Lattice's own); 1177 leaves the target
//!    under the backup name, so the temporary is moved into place without
//!    replacing, or else the backup is moved back. The target path is never
//!    left empty while a move can fill it, and neither file is removed while
//!    it holds the only copy. A new file is placed by a move that never
//!    replaces: a file that appeared meanwhile makes a conflict. Sharing
//!    violations are retried 6 times, 20 ms apart.
//! 6. **Verify:** the target is hashed again; a difference is reported as
//!    "written, then changed by another program" (`changed_after`), not as
//!    an error.
//! 7. **A delete** is moved aside into `removed/<n>/` (ND2) after the hash
//!    check, with a manifest line and a `moved_aside` item.
//! 8. **The change:** a whole Keep makes it `Kept`; a hunk Keep leaves the
//!    other hunks staged against the bytes just written (`PartlyKept`), so
//!    the remaining diff is exactly the unkept hunks.
//! 9. **The record:** the change's next state as a further `staged` item,
//!    then a `reviewed` item with the operation, the result and the base
//!    copy.
//!
//! **Undo** writes nothing to the folder: a whole Undo marks the change
//! `Undone` (its blobs stay, ND1); a hunk Undo stages the new bytes without
//! the chosen hunks. A note reaches the record, cut at 2,000 characters.
//! **Keep All** keeps every live change that is not an authority file, in
//! path order, under one checkpoint; **Undo All** undoes every change still
//! waiting.
//!
//! **A command's effect** (`OnDisk`, `changes::effects`, row E8) is already
//! on disk: its Keep only acknowledges it (`Kept`, nothing written), and its
//! Undo stages the inverse as a `CommandUndo` change, which is then kept like
//! any other (and, like any Keep, refused while a command runs).
//!
//! The work is synchronous apart from the native dialog. [`review_with`]
//! runs every step that touches the folder, git or the record through an
//! [`Offload`] (the agent chat's runs them on the blocking pool), and
//! awaits only the dialog on the caller's runtime.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::BoxFuture;

use lattice_protocol::conversation::{
    Against, ChangeId, ChangeKind, ChangeResult, ChangeState, CheckpointId, CheckpointReason,
    FileDiff, HunkId, HunkState, Mode, ReviewOp, ReviewOutcome, ReviewResult,
};
use lattice_sys::fs::ReplaceError;

use super::{Base, Snapshot, Staging, apply_op, is_live, is_waiting, text_of};
use crate::clock::Clock;
use crate::convo::item::{BaseState, Item, NewState};
use crate::convo::sidecar::BlobKind;
use crate::fsx::{MoveReason, RemovedArea, remove_own_temporary, temporary_for};
use crate::git::runner::GitRunner;
use crate::policy::{
    Gates, KeepKind, Lease, PathClass, Reason, Standing, Target, ToolClass, Verdict, decide,
};
use crate::ports::{ConfirmRequest, Confirmer, Initiated};
use crate::sha::sha256_hex;
use crate::text::diff::{self, Plan};
use crate::workspace::Workspace;
use crate::workspace::paths::Want;

/// KP1: the largest target a Keep copies first, and so the largest it writes.
pub const MAX_BASE_COPY: u64 = 64 * 1024 * 1024;
/// The longest note that reaches the record.
pub const MAX_NOTE_CHARS: usize = 2000;
const RETRIES: u32 = 6;
const RETRY_PAUSE: Duration = Duration::from_millis(20);

/// Numbers the backup names of this process's replacements.
static BACKUP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// What a Keep needs from the rest of the core: the command gate (X14, rows
/// E7 and E8), the writer lease (§6.5, row E5) and the before-Keep
/// checkpoint (§8, row E4).
pub trait KeepGuards: Send + Sync {
    /// A command is running in this folder.
    fn command_running(&self) -> bool;
    /// The writer lease for this folder: `Held` (taken now if it was free),
    /// or `Elsewhere`.
    fn lease(&self) -> Lease;
    /// Take the before-Keep checkpoint over the folder; the paths about to
    /// be written are named, and the reason is `BeforeKeep` or, when the
    /// review keeps a restore, `BeforeRestoreKeep` (§8.1). `Err` is why it
    /// could not be taken.
    fn checkpoint(
        &self,
        paths: &[String],
        reason: CheckpointReason,
    ) -> Result<CheckpointId, String>;
}

/// What a review works with.
pub struct ReviewContext<'a> {
    pub workspace: &'a Workspace,
    pub runner: &'a GitRunner,
    pub staging: &'a Staging,
    pub mode: Mode,
    pub trusted: bool,
    pub guards: &'a dyn KeepGuards,
    /// The native dialogs, with CP3's memory of refusals.
    pub confirmer: &'a Confirmer,
    pub clock: &'a Clock,
}

const CHECKPOINT_FAILED: &str =
    "Lattice could not record the folder's state first, so nothing was written.";
const TOO_LARGE: &str =
    "That file is too large for Lattice to keep a copy of, so nothing was written.";
const NOT_WRITTEN: &str = "That file could not be written; it is unchanged.";
const WHOLE_ONLY: &str = "This change can only be kept or undone whole.";
const NO_SUCH_HUNK: &str = "That part of the change is no longer there; look at the diff again.";
const NOT_PENDING: &str = "That change is no longer waiting for review.";
const IN_CONFLICT: &str = "That change conflicts with the file on disk; undo it, or ask the agent to read the file again.";
const NOT_CONFIRMED: &str = "You did not confirm keeping this file, so nothing was written.";
const CHANGED_WHILE_CONFIRMING: &str =
    "That change changed while you were confirming, so nothing was written; look at it again.";
const THROUGH_LINK: &str =
    "This path is now reached through a link to another place, so nothing was written.";
const NOW_AUTHORITY: &str = "This path now names a file that changes how tools or Lattice behave; keep it on its own, after confirming it.";
const CHANGED_MEANWHILE: &str =
    "That change changed while it was being kept, so nothing was written; look at it again.";

/// The conflict sentence the agent hears (§7.5 step 3).
pub fn conflict_sentence(path: &str) -> String {
    format!(
        "Your change to `{path}` conflicts with an edit made outside Lattice; read the file again."
    )
}

/// The base and new texts of a change, without byte-order marks, when both
/// sides are text the diff can cut: `(base, new, bom)`.
fn texts(snapshot: &Snapshot) -> Option<(String, String, bool)> {
    let base = match (&snapshot.change.base, &snapshot.base) {
        (BaseState::Absent, _) => (String::new(), false),
        (BaseState::Present { .. }, Some(bytes)) => {
            let (text, bom) = text_of(bytes).ok()?;
            (text.to_owned(), bom)
        }
        (BaseState::Present { .. }, None) => return None,
    };
    let new = match (&snapshot.change.new, &snapshot.new) {
        (NewState::Deleted, _) => return None,
        (NewState::Bytes { .. }, Some(bytes)) => text_of(bytes).ok()?.0.to_owned(),
        (NewState::Bytes { .. }, None) => return None,
    };
    Some((base.0, new, base.1))
}

fn base_sha(base: &BaseState) -> String {
    match base {
        BaseState::Present { sha256, .. } => sha256.clone(),
        BaseState::Absent => String::new(),
    }
}

/// The change's diff, cut into hunks, when it can be: `(plan, base, new,
/// bom)`.
fn plan_of(snapshot: &Snapshot) -> Option<(Plan, String, String, bool)> {
    let (base, new, bom) = texts(snapshot)?;
    let plan = diff::plan(
        &snapshot.change.id,
        &base_sha(&snapshot.change.base),
        &base,
        &new,
    );
    Some((plan, base, new, bom))
}

/// The change's diff for the interface (`diff`, §4.3): the shipping app's
/// `FileDiff`, with each hunk's id and state. A change that is not text on
/// both sides (a delete, a binary file) is `binary`, with no hunks.
pub fn file_diff(staging: &Staging, change: &str) -> Option<FileDiff> {
    let snapshot = staging.snapshot(change)?;
    let mut out = FileDiff {
        change: snapshot.change.id.clone(),
        path: snapshot.change.path.clone(),
        against: Against::Previous,
        snapshot: 0,
        binary: true,
        hunks: Vec::new(),
        truncated: false,
    };
    if let Some((plan, ..)) = plan_of(&snapshot) {
        out.binary = false;
        out.truncated = plan.truncated;
        out.hunks = plan.hunks();
        let state = match snapshot.change.state {
            ChangeState::Kept => HunkState::Kept,
            ChangeState::Undone => HunkState::Undone,
            _ => HunkState::Pending,
        };
        for hunk in &mut out.hunks {
            hunk.state = state;
        }
    }
    Some(out)
}

/// The bytes of `text` with the base's byte-order mark.
fn with_bom(text: &str, bom: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() + 3);
    if bom {
        out.extend_from_slice(super::BOM);
    }
    out.extend_from_slice(text.as_bytes());
    out
}

/// The hunks chosen, checked against the plan: `Err` with the reason.
fn chosen(
    snapshot: &Snapshot,
    hunks: &[HunkId],
) -> Result<(Plan, String, String, bool, BTreeSet<HunkId>), String> {
    let Some((plan, base, new, bom)) = plan_of(snapshot) else {
        return Err(WHOLE_ONLY.to_owned());
    };
    if plan.truncated {
        return Err(WHOLE_ONLY.to_owned());
    }
    let ids: BTreeSet<HunkId> = plan.ids().into_iter().collect();
    let take: BTreeSet<HunkId> = hunks.iter().cloned().collect();
    if take.is_empty() || !take.is_subset(&ids) {
        return Err(NO_SUCH_HUNK.to_owned());
    }
    Ok((plan, base, new, bom, take))
}

/// A note as the record keeps it: at most [`MAX_NOTE_CHARS`] characters.
fn cut_note(note: &Option<String>) -> Option<String> {
    note.as_ref()
        .map(|text| text.chars().take(MAX_NOTE_CHARS).collect())
}

/// One review's shared state: the checkpoint, taken at most once, and the
/// lease, read at most once.
struct Batch {
    paths: Vec<String>,
    /// The review keeps a restore.
    restore: bool,
    checkpoint: Option<Result<CheckpointId, String>>,
    lease: Option<Lease>,
}

impl Batch {
    fn checkpoint(&mut self, guards: &dyn KeepGuards) -> Result<CheckpointId, String> {
        let reason = if self.restore {
            CheckpointReason::BeforeRestoreKeep
        } else {
            CheckpointReason::BeforeKeep
        };
        self.checkpoint
            .get_or_insert_with(|| guards.checkpoint(&self.paths, reason))
            .clone()
    }

    fn lease(&mut self, guards: &dyn KeepGuards) -> Lease {
        *self.lease.get_or_insert_with(|| guards.lease())
    }
}

/// What one change's Keep or Undo came to.
struct Done {
    result: ReviewResult,
    base_copy: Option<String>,
}

impl Done {
    fn of(result: ReviewResult) -> Self {
        Self {
            result,
            base_copy: None,
        }
    }

    fn failed(reason: impl Into<String>) -> Self {
        Self::of(ReviewResult::Failed {
            reason: reason.into(),
        })
    }
}

/// A piece of a review's work that touches the folder, git or the record:
/// it runs with a [`ReviewContext`] over the review's own parts.
pub type Job = Box<dyn FnOnce(&ReviewContext<'_>) + Send + 'static>;

/// Where a review runs its [`Job`]s. The agent chat's service runs them on
/// the runtime's blocking pool (`spawn_blocking`), so a checkpoint's git
/// processes, the `ReplaceFileW` write and its retries never hold an async
/// worker; only the native dialog is awaited on the runtime. [`review`]
/// runs them in place.
pub trait Offload: Sync {
    /// Run `job` to its end; a job that panicked has simply not finished.
    fn run(&self, job: Job) -> BoxFuture<'_, ()>;
}

/// Jobs run in place, on the caller's thread, with the caller's context.
struct InPlace<'c, 'a>(&'c ReviewContext<'a>);

impl Offload for InPlace<'_, '_> {
    fn run(&self, job: Job) -> BoxFuture<'_, ()> {
        job(self.0);
        Box::pin(std::future::ready(()))
    }
}

/// Run `work` through `offload` and give back what it returned (`None`
/// when it did not finish).
async fn off<R: Send + 'static>(
    offload: &dyn Offload,
    work: impl FnOnce(&ReviewContext<'_>) -> R + Send + 'static,
) -> Option<R> {
    let slot = Arc::new(Mutex::new(None));
    let out = Arc::clone(&slot);
    offload
        .run(Box::new(move |ctx| {
            let value = work(ctx);
            *out.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(value);
        }))
        .await;
    slot.lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
}

/// Why a step that did not finish wrote nothing more.
const UNFINISHED: &str =
    "Lattice could not finish this step of the review; look at the change again.";

/// A batch that lost its state to a step that did not finish: no further
/// Keep in the review writes (its checkpoint is refused).
fn unfinished_batch() -> Batch {
    Batch {
        paths: Vec::new(),
        restore: false,
        checkpoint: Some(Err(UNFINISHED.to_owned())),
        lease: None,
    }
}

/// The result of a step that did not finish.
fn unfinished(change: &ChangeId) -> ChangeResult {
    ChangeResult {
        change: change.clone(),
        path: String::new(),
        result: ReviewResult::Failed {
            reason: UNFINISHED.to_owned(),
        },
    }
}

/// Run the review operations, in order, and say what each did, per change in
/// path order. The disk work runs in place (tests, and callers already off
/// the async runtime); see [`review_with`].
pub async fn review(ctx: &ReviewContext<'_>, ops: Vec<ReviewOp>) -> ReviewOutcome {
    review_with(ctx, &InPlace(ctx), ops).await
}

/// [`review`], with every step that touches the folder, git or the record
/// run through `offload`; `ctx` itself serves only the native dialog and
/// the staged changes in memory.
pub async fn review_with(
    ctx: &ReviewContext<'_>,
    offload: &dyn Offload,
    ops: Vec<ReviewOp>,
) -> ReviewOutcome {
    let (paths, restore) = keep_paths(ctx.staging, &ops);
    let mut batch = Batch {
        paths,
        restore,
        checkpoint: None,
        lease: None,
    };
    let mut results: Vec<ChangeResult> = Vec::new();
    for op in ops {
        match &op {
            ReviewOp::Keep { change, hunks } => {
                let (result, next) = keep(
                    ctx,
                    offload,
                    batch,
                    change.clone(),
                    hunks.clone(),
                    false,
                    op.clone(),
                )
                .await;
                batch = next;
                results.push(result);
            }
            ReviewOp::KeepAll => {
                let mut targets: Vec<_> = ctx
                    .staging
                    .changes()
                    .into_iter()
                    .filter(|change| is_live(&change.state))
                    .collect();
                targets.sort_by(|a, b| a.path.cmp(&b.path));
                for change in targets {
                    let (result, next) =
                        keep(ctx, offload, batch, change.id, None, true, op.clone()).await;
                    batch = next;
                    results.push(result);
                }
            }
            ReviewOp::Undo {
                change,
                hunks,
                note,
            } => {
                let op = ReviewOp::Undo {
                    change: change.clone(),
                    hunks: hunks.clone(),
                    note: cut_note(note),
                };
                let (id, hunks) = (change.clone(), hunks.clone());
                let result = off(offload, move |ctx| {
                    let done = undo(ctx, &id, hunks.as_deref());
                    record(ctx, &id, &op, done)
                })
                .await;
                results.push(result.unwrap_or_else(|| unfinished(change)));
            }
            ReviewOp::UndoAll { note } => {
                let op = ReviewOp::UndoAll {
                    note: cut_note(note),
                };
                let done = off(offload, move |ctx| {
                    let mut out = Vec::new();
                    for change in ctx.staging.changes() {
                        if is_waiting(&change.state) || is_live(&change.state) {
                            let done = undo(ctx, &change.id, None);
                            out.push(record(ctx, &change.id, &op, done));
                        }
                    }
                    out
                })
                .await;
                results.extend(done.unwrap_or_default());
            }
        }
    }
    results.sort_by(|a, b| a.path.cmp(&b.path));
    ReviewOutcome { results }
}

/// Every path a Keep in `ops` may write, for the checkpoint, and whether
/// one of them is a restore's.
fn keep_paths(staging: &Staging, ops: &[ReviewOp]) -> (Vec<String>, bool) {
    let mut paths = BTreeSet::new();
    let mut restore = false;
    for op in ops {
        match op {
            ReviewOp::Keep { change, .. } => {
                if let Some(snapshot) = staging.snapshot(change) {
                    restore |= snapshot.change.kind == ChangeKind::Restore;
                    paths.insert(snapshot.change.path);
                }
            }
            ReviewOp::KeepAll => {
                for change in staging.changes() {
                    if is_live(&change.state) && !change.authority {
                        restore |= change.kind == ChangeKind::Restore;
                        paths.insert(change.path);
                    }
                }
            }
            ReviewOp::Undo { .. } | ReviewOp::UndoAll { .. } => {}
        }
    }
    (paths.into_iter().collect(), restore)
}

/// Append the `reviewed` item, and give the change's result.
fn record(ctx: &ReviewContext<'_>, change: &ChangeId, op: &ReviewOp, done: Done) -> ChangeResult {
    let path = ctx
        .staging
        .snapshot(change)
        .map(|snapshot| snapshot.change.path)
        .unwrap_or_default();
    let _ = ctx.staging.sidecar().append(&Item::Reviewed {
        change: change.clone(),
        op: op.clone(),
        result: done.result.clone(),
        at: (ctx.clock)(),
        base_copy: done.base_copy,
    });
    ChangeResult {
        change: change.clone(),
        path,
        result: done.result,
    }
}

fn keep_kind(kind: ChangeKind) -> KeepKind {
    match kind {
        ChangeKind::Restore => KeepKind::Restore,
        ChangeKind::CommandUndo => KeepKind::CommandUndo,
        _ => KeepKind::Change,
    }
}

/// What a Keep's preconditions came to.
enum Prepared {
    /// Nothing to write: the result.
    Done(Done),
    /// The reader must confirm in the native dialog first.
    Ask {
        snapshot: Snapshot,
        class: PathClass,
        key: String,
        request: ConfirmRequest,
    },
    /// Go on to the disk.
    Go {
        snapshot: Snapshot,
        class: PathClass,
    },
}

/// One change's Keep: the preconditions (through `offload`), the native
/// dialog for an authority file (awaited here, on the caller's runtime),
/// then the disk and the record (through `offload`). The batch goes in and
/// comes back.
async fn keep(
    ctx: &ReviewContext<'_>,
    offload: &dyn Offload,
    batch: Batch,
    id: ChangeId,
    hunks: Option<Vec<HunkId>>,
    all: bool,
    op: ReviewOp,
) -> (ChangeResult, Batch) {
    let prepare_id = id.clone();
    let prepared = off(offload, move |ctx| {
        let mut batch = batch;
        let prepared = prepare(ctx, &mut batch, &prepare_id, hunks.as_deref(), all);
        (prepared, batch, hunks)
    })
    .await;
    let Some((prepared, batch, hunks)) = prepared else {
        return (unfinished(&id), unfinished_batch());
    };
    let (snapshot, class, asked) = match prepared {
        Prepared::Done(done) => {
            let record_id = id.clone();
            let finished = off(offload, move |ctx| {
                (record(ctx, &record_id, &op, done), batch)
            })
            .await;
            return finished.unwrap_or_else(|| (unfinished(&id), unfinished_batch()));
        }
        Prepared::Go { snapshot, class } => (snapshot, class, false),
        Prepared::Ask {
            snapshot,
            class,
            key,
            request,
        } => {
            let confirmed = ctx.confirmer.ask(&key, request, Initiated::Page).await;
            if !confirmed {
                let record_id = id.clone();
                let finished = off(offload, move |ctx| {
                    let done = Done::of(ReviewResult::Skipped {
                        reason: NOT_CONFIRMED.to_owned(),
                    });
                    (record(ctx, &record_id, &op, done), batch)
                })
                .await;
                return finished.unwrap_or_else(|| (unfinished(&id), unfinished_batch()));
            }
            (snapshot, class, true)
        }
    };
    let finish_id = id.clone();
    let finished = off(offload, move |ctx| {
        let mut batch = batch;
        let done = finish(
            ctx,
            &mut batch,
            &finish_id,
            snapshot,
            hunks.as_deref(),
            class,
            asked,
        );
        (record(ctx, &finish_id, &op, done), batch)
    })
    .await;
    finished.unwrap_or_else(|| (unfinished(&id), unfinished_batch()))
}

/// A Keep's preconditions: the change's state, then the policy (§9.2) with
/// its gates (which may take the writer lease).
fn prepare(
    ctx: &ReviewContext<'_>,
    batch: &mut Batch,
    id: &ChangeId,
    hunks: Option<&[HunkId]>,
    all: bool,
) -> Prepared {
    let Some(snapshot) = ctx.staging.snapshot(id) else {
        return Prepared::Done(Done::of(ReviewResult::Skipped {
            reason: "There is no such change.".to_owned(),
        }));
    };
    match &snapshot.change.state {
        // A command's effect is already on disk: Keep only acknowledges it
        // and writes nothing (§8.4).
        ChangeState::OnDisk => return Prepared::Done(acknowledge(ctx, snapshot, hunks)),
        ChangeState::Conflict { .. } => {
            return Prepared::Done(Done::of(ReviewResult::Skipped {
                reason: IN_CONFLICT.to_owned(),
            }));
        }
        state if !is_live(state) => {
            return Prepared::Done(Done::of(ReviewResult::Skipped {
                reason: NOT_PENDING.to_owned(),
            }));
        }
        _ => {}
    }
    let class = if snapshot.change.authority {
        PathClass::Authority
    } else {
        PathClass::Normal
    };
    let tool = if all {
        ToolClass::KeepAll
    } else {
        ToolClass::Keep(keep_kind(snapshot.change.kind))
    };
    let gates = Gates {
        staged_waiting: ctx.staging.waiting(),
        command_running: ctx.guards.command_running(),
        lease: batch.lease(ctx.guards),
    };
    match decide(
        ctx.mode,
        ctx.trusted,
        tool,
        &Target::Path(class),
        &gates,
        &Standing::default(),
    ) {
        Verdict::Refuse(Reason::AuthorityKeptAlone) => {
            Prepared::Done(Done::of(ReviewResult::Skipped {
                reason: Reason::AuthorityKeptAlone.sentence().to_owned(),
            }))
        }
        Verdict::Refuse(reason) if reason.is_conflict() => {
            Prepared::Done(Done::of(ReviewResult::Conflict {
                reason: reason.sentence().to_owned(),
            }))
        }
        Verdict::Refuse(reason) => Prepared::Done(Done::failed(reason.sentence())),
        Verdict::Ask(_) => {
            let (added, removed) = plan_of(&snapshot)
                .map(|(plan, ..)| plan.counts())
                .unwrap_or((0, snapshot.base_lines));
            let key = format!(
                "keep-authority:{}:{}",
                snapshot.change.id,
                match &snapshot.change.new {
                    NewState::Bytes { blob } => blob.as_str(),
                    NewState::Deleted => "deleted",
                }
            );
            Prepared::Ask {
                request: ConfirmRequest::KeepAuthority {
                    path: snapshot.change.path.clone(),
                    added,
                    removed,
                },
                snapshot,
                class,
                key,
            }
        }
        Verdict::Allow(_) => Prepared::Go { snapshot, class },
    }
}

/// After the preconditions and any dialog: the re-check under the staging
/// lock, then the disk.
fn finish(
    ctx: &ReviewContext<'_>,
    batch: &mut Batch,
    id: &ChangeId,
    snapshot: Snapshot,
    hunks: Option<&[HunkId]>,
    class: PathClass,
    asked: bool,
) -> Done {
    // The dialog may have waited while the agent staged this path again, or
    // while another review kept this change: re-read it under the staging
    // lock, held from here through the change's update, and write nothing
    // for a change that moved since the snapshot the reader confirmed.
    let _one = ctx.staging.serial();
    let Some(fresh) = ctx.staging.snapshot(id) else {
        return Done::of(ReviewResult::Skipped {
            reason: "There is no such change.".to_owned(),
        });
    };
    if fresh.change.state == ChangeState::Kept {
        // A duplicate Keep (a double click, or two pages) whose twin kept
        // it while this one waited: it is kept; nothing more is written.
        return Done::of(ReviewResult::Kept {
            changed_after: false,
        });
    }
    if !is_live(&fresh.change.state) {
        return Done::of(ReviewResult::Skipped {
            reason: NOT_PENDING.to_owned(),
        });
    }
    if fresh.revision != snapshot.revision {
        return Done::of(ReviewResult::Skipped {
            reason: if asked {
                CHANGED_WHILE_CONFIRMING
            } else {
                CHANGED_MEANWHILE
            }
            .to_owned(),
        });
    }
    keep_on_disk(ctx, batch, snapshot, hunks, class)
}

/// Keep of a command's effect (§8.4): acknowledged, nothing written.
fn acknowledge(ctx: &ReviewContext<'_>, mut snapshot: Snapshot, hunks: Option<&[HunkId]>) -> Done {
    if hunks.is_some() {
        return Done::failed(WHOLE_ONLY);
    }
    snapshot.change.state = ChangeState::Kept;
    match ctx.staging.update(snapshot) {
        Ok(()) => Done::of(ReviewResult::Kept {
            changed_after: false,
        }),
        Err(error) => Done::failed(error.sentence()),
    }
}

/// What the Keep will leave at the path: bytes, or nothing (a delete); and
/// for a hunk Keep, the plan and the hunks taken.
struct Composed {
    bytes: Option<Vec<u8>>,
    partial: Option<(Plan, String, String, bool, BTreeSet<HunkId>)>,
}

fn compose(snapshot: &Snapshot, hunks: Option<&[HunkId]>) -> Result<Composed, String> {
    if let Some(hunks) = hunks {
        let (plan, base, new, bom, take) = chosen(snapshot, hunks)?;
        if take.len() < plan.hunks.len() {
            let text = diff::apply(&plan, &base, &new, &take);
            return Ok(Composed {
                bytes: Some(with_bom(&text, bom)),
                partial: Some((plan, base, new, bom, take)),
            });
        }
    }
    match (&snapshot.change.new, &snapshot.new) {
        (NewState::Deleted, _) => Ok(Composed {
            bytes: None,
            partial: None,
        }),
        (NewState::Bytes { .. }, Some(bytes)) => Ok(Composed {
            bytes: Some(bytes.clone()),
            partial: None,
        }),
        (NewState::Bytes { .. }, None) => Err(
            "Lattice could not read this change's staged bytes back, so nothing was written."
                .to_owned(),
        ),
    }
}

/// The target's bytes now, through the handle the path rules opened: `None`
/// when there is no file. `Err(true)` when it is over [`MAX_BASE_COPY`].
fn current(file: Option<File>) -> Result<Option<Vec<u8>>, bool> {
    let Some(file) = file else {
        return Ok(None);
    };
    let size = file.metadata().map_err(|_| false)?.len();
    if size > MAX_BASE_COPY {
        return Err(true);
    }
    let mut bytes = Vec::new();
    file.take(MAX_BASE_COPY + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| false)?;
    if bytes.len() as u64 > MAX_BASE_COPY {
        return Err(true);
    }
    Ok(Some(bytes))
}

/// Steps 2–9 for one change whose preconditions held.
fn keep_on_disk(
    ctx: &ReviewContext<'_>,
    batch: &mut Batch,
    mut snapshot: Snapshot,
    hunks: Option<&[HunkId]>,
    class: PathClass,
) -> Done {
    let composed = match compose(&snapshot, hunks) {
        Ok(composed) => composed,
        Err(reason) => return Done::failed(reason),
    };
    if batch.checkpoint(ctx.guards).is_err() {
        return Done::failed(CHECKPOINT_FAILED);
    }
    let path = snapshot.change.path.clone();
    let resolved = ctx
        .workspace
        .with_rules(ctx.runner, |rules| rules.resolve(&path, Want::MayCreate));
    let mut resolved = match resolved {
        Ok(Ok(resolved)) => resolved,
        Ok(Err(error)) => return Done::failed(error.sentence()),
        Err(_) => {
            return Done::failed(
                "Lattice could not read this folder's .latticeignore, so nothing was written.",
            );
        }
    };
    if resolved.is_dir {
        return Done::failed("That path is now a folder, so nothing was written.");
    }
    // ST4 at the write, not only at staging: a folder swapped for a link
    // (a junction to `.github`, say) since the change was staged must not
    // carry an ordinary change into another file. The path must still lead
    // to itself (compared as the file system names it), and its class now
    // must be the one the policy checked.
    if super::path_key(&resolved.derived) != super::path_key(&path) {
        return Done::of(ReviewResult::Conflict {
            reason: THROUGH_LINK.to_owned(),
        });
    }
    if resolved.class == PathClass::Authority && class != PathClass::Authority {
        return Done::of(ReviewResult::Conflict {
            reason: NOW_AUTHORITY.to_owned(),
        });
    }
    let target = resolved.final_path.clone();
    let now = match current(resolved.file.take()) {
        Ok(now) => now,
        Err(true) => {
            return Done::of(ReviewResult::Conflict {
                reason: TOO_LARGE.to_owned(),
            });
        }
        Err(false) => return Done::failed(NOT_WRITTEN),
    };
    // KP1: the bytes about to be replaced or moved are kept first.
    let base_copy = match &now {
        Some(bytes) => match ctx.staging.sidecar().put_blob(bytes, BlobKind::Staged) {
            Ok(blob) => Some(blob.sha256),
            Err(_) => {
                return Done::failed(
                    "Lattice could not keep a copy of the file first, so nothing was written.",
                );
            }
        },
        None => None,
    };
    let expected = match (&snapshot.change.base, &now) {
        (BaseState::Absent, None) => true,
        (BaseState::Present { sha256, .. }, Some(bytes)) => sha256_hex(bytes) == *sha256,
        _ => false,
    };
    // The file already holds what this Keep would leave: an earlier Keep of
    // the same change wrote it and stopped before its update (a crash), or a
    // duplicate Keep raced it. Applying the edits again would double them
    // ("ab" -> "abb"); write nothing and finish the Keep.
    let already = match (&composed.bytes, &now) {
        (Some(bytes), Some(now)) => bytes == now,
        (None, None) => true,
        _ => false,
    };
    if !expected && !already {
        let mut done = rebase_or_conflict(ctx, snapshot, now);
        done.base_copy = base_copy;
        return done;
    }
    let at = (ctx.clock)();
    let written = match &composed.bytes {
        _ if already => Written::Done,
        None => match now {
            None => Written::Done,
            Some(_) => move_deleted_aside(ctx, &snapshot, &target, at),
        },
        Some(bytes) if now.is_some() => replace(ctx, &path, &target, bytes, at),
        Some(bytes) => create(&target, bytes),
    };
    match written {
        Written::Done => {}
        Written::Failed(reason) => {
            return Done {
                result: ReviewResult::Failed { reason },
                base_copy,
            };
        }
        Written::Conflict(reason) => {
            snapshot.change.state = ChangeState::Conflict {
                reason: reason.clone(),
            };
            let _ = ctx.staging.update(snapshot);
            return Done {
                result: ReviewResult::Conflict { reason },
                base_copy,
            };
        }
    }
    let changed_after = match &composed.bytes {
        Some(bytes) => std::fs::read(&target).map_or(true, |after| after != *bytes),
        None => std::fs::symlink_metadata(&target).is_ok(),
    };
    let result = match composed.partial {
        None => {
            snapshot.change.state = ChangeState::Kept;
            ReviewResult::Kept { changed_after }
        }
        Some((plan, _, _, _, take)) => {
            let written = composed.bytes.clone().unwrap_or_default();
            let remaining = plan.hunks.len() - take.len();
            let base = Base::of_bytes(written);
            snapshot.change.base = base.state;
            snapshot.base = base.bytes;
            snapshot.base_lines = base.lines;
            // The ops no longer lead from this base to the new bytes.
            snapshot.change.ops.clear();
            snapshot.change.state = ChangeState::PartlyKept;
            ReviewResult::PartlyKept {
                remaining_hunks: u32::try_from(remaining).unwrap_or(u32::MAX),
            }
        }
    };
    let _ = ctx.staging.update(snapshot);
    Done { result, base_copy }
}

/// Step 3's mismatch: apply a change of edits again to the file as it is
/// (`Rebased`, nothing written), or mark it a conflict.
fn rebase_or_conflict(
    ctx: &ReviewContext<'_>,
    mut snapshot: Snapshot,
    now: Option<Vec<u8>>,
) -> Done {
    if snapshot.change.kind == ChangeKind::Edit
        && !snapshot.change.ops.is_empty()
        && let Some(current) = now
    {
        let mut text = current.clone();
        let applied = snapshot
            .change
            .ops
            .iter()
            .try_for_each(|op| apply_op(&text, op).map(|next| text = next));
        if applied.is_ok() {
            let base = Base::of_bytes(current);
            snapshot.change.base = base.state;
            snapshot.base = base.bytes;
            snapshot.base_lines = base.lines;
            snapshot.change.new = NewState::Bytes {
                blob: sha256_hex(&text),
            };
            snapshot.new = Some(text);
            snapshot.change.state = ChangeState::Rebased;
            return match ctx.staging.update(snapshot) {
                Ok(()) => Done::of(ReviewResult::Rebased),
                Err(error) => Done::failed(error.sentence()),
            };
        }
    }
    let reason = conflict_sentence(&snapshot.change.path);
    snapshot.change.state = ChangeState::Conflict {
        reason: reason.clone(),
    };
    let _ = ctx.staging.update(snapshot);
    Done::of(ReviewResult::Conflict { reason })
}

/// How a write went.
enum Written {
    Done,
    Failed(String),
    Conflict(String),
}

/// Retry a sharing violation 6 times, 20 ms apart, as Python's `_replace`.
fn retrying<T, E>(
    mut operation: impl FnMut() -> Result<T, E>,
    transient: impl Fn(&E) -> bool,
) -> Result<T, E> {
    let mut attempt = 1;
    loop {
        match operation() {
            Err(error) if transient(&error) && attempt < RETRIES => {
                attempt += 1;
                std::thread::sleep(RETRY_PAUSE);
            }
            other => return other,
        }
    }
}

fn sharing(error: &io::Error) -> bool {
    matches!(error.raw_os_error(), Some(5 | 32))
}

/// Write `bytes` to a fresh temporary beside `target`, synced.
fn temporary_with(target: &Path, bytes: &[u8]) -> io::Result<PathBuf> {
    let temporary = temporary_for(target)?;
    let written = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .and_then(|mut file| {
            file.write_all(bytes)?;
            file.sync_all()
        });
    if let Err(error) = written {
        let _ = remove_own_temporary(&temporary);
        return Err(error);
    }
    Ok(temporary)
}

/// `.<name>.lattice-bak-<pid>-<n>` beside `target`.
fn backup_for(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let n = BACKUP_COUNTER.fetch_add(1, Ordering::Relaxed);
    target.with_file_name(format!(".{name}.lattice-bak-{}-{n}", std::process::id()))
}

/// The removed area of this conversation (`removed/`, ND2).
fn removed_area(ctx: &ReviewContext<'_>) -> RemovedArea {
    RemovedArea::new(ctx.staging.sidecar().dir().join("removed"))
}

/// Move `source` into `removed/<n>/<path>` and record it; on failure, say
/// where the bytes stayed.
fn set_aside(
    ctx: &ReviewContext<'_>,
    path: &str,
    source: &Path,
    why: MoveReason,
    at: f64,
) -> io::Result<()> {
    let area = removed_area(ctx);
    let moved = area
        .next_slot()
        .and_then(|slot| area.move_aside(slot, path, source, why, at))?;
    let _ = ctx.staging.sidecar().append(&Item::MovedAside {
        path: moved.path,
        to: moved.to,
        sha256: moved.sha256,
        why,
        at,
    });
    Ok(())
}

/// Step 5 for a file that exists: `ReplaceFileW` with a backup name.
fn replace(ctx: &ReviewContext<'_>, path: &str, target: &Path, bytes: &[u8], at: f64) -> Written {
    let temporary = match temporary_with(target, bytes) {
        Ok(temporary) => temporary,
        Err(_) => return Written::Failed(NOT_WRITTEN.to_owned()),
    };
    let backup = backup_for(target);
    let replaced = retrying(
        || lattice_sys::fs::replace_file(target, &temporary, &backup),
        |error| matches!(error, ReplaceError::Io(error) if sharing(error)),
    );
    let keep_backup = |ctx: &ReviewContext<'_>| {
        if set_aside(ctx, path, &backup, MoveReason::ReplacedBackup, at).is_err() {
            let name = backup
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            let _ = ctx.staging.sidecar().append(&Item::Notice {
                text: format!("The bytes a Keep replaced in `{path}` stay beside it as {name}."),
                at,
            });
        }
    };
    match replaced {
        Ok(()) => {
            keep_backup(ctx);
            Written::Done
        }
        // 1176: both files keep their names; the target is as it was.
        Err(ReplaceError::UnableToMoveReplacement) | Err(ReplaceError::Io(_)) => {
            let _ = remove_own_temporary(&temporary);
            Written::Failed(NOT_WRITTEN.to_owned())
        }
        // 1177: the target is under the backup name; fill its path again.
        Err(ReplaceError::UnableToMoveReplacement2) => {
            let placed = retrying(
                || lattice_sys::fs::move_no_replace(&temporary, target),
                sharing,
            );
            if placed.is_ok() {
                keep_backup(ctx);
                return Written::Done;
            }
            match retrying(
                || lattice_sys::fs::move_no_replace(&backup, target),
                sharing,
            ) {
                Ok(()) => {
                    let _ = remove_own_temporary(&temporary);
                    Written::Failed(NOT_WRITTEN.to_owned())
                }
                Err(_) => {
                    let name = |file: &Path| {
                        file.file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default()
                    };
                    Written::Failed(format!(
                        "That file could not be written; its old bytes are beside it as {} and the new ones as {}.",
                        name(&backup),
                        name(&temporary)
                    ))
                }
            }
        }
    }
}

/// Step 5 for a new file: a move that never replaces.
fn create(target: &Path, bytes: &[u8]) -> Written {
    let temporary = match temporary_with(target, bytes) {
        Ok(temporary) => temporary,
        Err(_) => return Written::Failed(NOT_WRITTEN.to_owned()),
    };
    match retrying(
        || lattice_sys::fs::move_no_replace(&temporary, target),
        sharing,
    ) {
        Ok(()) => Written::Done,
        Err(error) => {
            let _ = remove_own_temporary(&temporary);
            if error.kind() == io::ErrorKind::AlreadyExists {
                Written::Conflict(
                    "A file appeared at that path while it was kept, so nothing was written."
                        .to_owned(),
                )
            } else {
                Written::Failed(NOT_WRITTEN.to_owned())
            }
        }
    }
}

/// Step 7: a delete is moved aside, never unlinked.
fn move_deleted_aside(
    ctx: &ReviewContext<'_>,
    snapshot: &Snapshot,
    target: &Path,
    at: f64,
) -> Written {
    let why = match snapshot.change.kind {
        ChangeKind::Restore => MoveReason::KeptRestore,
        ChangeKind::CommandUndo => MoveReason::UndoneCommand,
        _ => MoveReason::KeptDelete,
    };
    match set_aside(ctx, &snapshot.change.path, target, why, at) {
        Ok(()) => Written::Done,
        Err(_) => {
            Written::Failed("That file could not be moved aside; it is unchanged.".to_owned())
        }
    }
}

/// One change's Undo: whole, or the chosen hunks taken out of its new bytes.
fn undo(ctx: &ReviewContext<'_>, id: &ChangeId, hunks: Option<&[HunkId]>) -> Done {
    let staging = ctx.staging;
    let Some(mut snapshot) = staging.snapshot(id) else {
        return Done::of(ReviewResult::Skipped {
            reason: "There is no such change.".to_owned(),
        });
    };
    if snapshot.change.state == ChangeState::OnDisk {
        // A command's effect: stage its inverse (§8.4), reviewed like any
        // change; nothing is written here.
        if hunks.is_some() {
            return Done::failed(WHOLE_ONLY);
        }
        return match crate::changes::effects::stage_undo(ctx.workspace, ctx.runner, staging, id) {
            Ok(_) => Done::of(ReviewResult::Undone),
            Err(reason) => Done::failed(reason),
        };
    }
    if !(is_live(&snapshot.change.state) || is_waiting(&snapshot.change.state)) {
        return Done::of(ReviewResult::Skipped {
            reason: NOT_PENDING.to_owned(),
        });
    }
    if let Some(hunks) = hunks {
        let (plan, base, new, bom, undone) = match chosen(&snapshot, hunks) {
            Ok(chosen) => chosen,
            Err(reason) => return Done::failed(reason),
        };
        let keep: BTreeSet<HunkId> = plan
            .ids()
            .into_iter()
            .filter(|id| !undone.contains(id))
            .collect();
        if !keep.is_empty() {
            let text = diff::apply(&plan, &base, &new, &keep);
            let bytes = with_bom(&text, bom);
            snapshot.change.new = NewState::Bytes {
                blob: sha256_hex(&bytes),
            };
            snapshot.new = Some(bytes);
            // The ops no longer lead from the base to the new bytes.
            snapshot.change.ops.clear();
            snapshot.change.state = ChangeState::Pending;
            return match staging.update(snapshot) {
                Ok(()) => Done::of(ReviewResult::Undone),
                Err(error) => Done::failed(error.sentence()),
            };
        }
    }
    snapshot.change.state = ChangeState::Undone;
    match staging.update(snapshot) {
        Ok(()) => Done::of(ReviewResult::Undone),
        Err(error) => Done::failed(error.sentence()),
    }
}
