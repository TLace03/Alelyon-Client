//! The review against folders and scratch repositories this file makes
//! (the chat core's spec §7.5; §16.4 SF2–SF7, SF11, SF13–SF15). Every
//! folder is temporary; git runs only in scratch repositories. The checkpoint
//! is the real one (row E4, `changes::FolderGuards`), and so is the writer
//! lease (row E5, under the scratch area's own home), and so is the command
//! gate (rows E7 and E8: the process's command slots, which a test holds as
//! a running command would); the native dialog is a recording fake. Tests compare the
//! scratch area's file set before and after: no file's bytes are lost, and
//! no temporary or backup is left behind (NF1).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::executor::block_on;
use lattice_protocol::conversation::{
    ChangeKind, ChangeOrigin, ChangeState, CheckpointKind, CheckpointReason, DiffLineKind, Hunk,
    Mode, Origin, ReviewOp, ReviewOutcome, ReviewResult,
};
use lattice_sys::fs::seam::{ReplaceFault, inject_replace_fault};

use super::review::{MAX_NOTE_CHARS, ReviewContext, conflict_sentence, file_diff, review};
use super::{Action, BOM, Base, Proposal, Staging};
use crate::changes::FolderGuards;
use crate::changes::checkpoint::{Checkpoints, OmitWhy};
use crate::changes::restore::{RestoreContext, Restored, restore};
use crate::clock::Clock;
use crate::convo::item::{Item, NewState, StagedChange};
use crate::convo::sidecar::{NewMeta, SidecarStore};
use crate::exec::run::CommandSlots;
use crate::fsx::{MoveReason, RemovedArea};
use crate::git::runner::GitRunner;
use crate::git::tests::Scratch;
use crate::policy::Lease;
use crate::ports::fake::RecordingConfirm;
use crate::ports::{ConfirmRequest, Confirmer};
use crate::sha::sha256_hex;
use crate::state::{Platform, StateRoot};
use crate::text::diff;
use crate::tools::edit::{
    DeleteFileArgs, EditFileArgs, StageContext, WriteFileArgs, delete_file, edit_file, write_file,
};
use crate::tools::read::{ReadContext, ReadFileArgs, read_file};
use crate::workspace::Workspace;
use crate::workspace::attach::attach_path;
use crate::workspace::lease::WriterLease;

const ID: &str = "c0ffee000001";

/// The command gate (the real slots, X14); and whether git is gone, for the
/// checkpoint only or for the whole review (a runner whose `PATH` holds no
/// git).
struct Guards {
    commands: CommandSlots,
    git_gone: AtomicBool,
    git_gone_for_the_checkpoint: AtomicBool,
}

impl Default for Guards {
    fn default() -> Self {
        Self {
            commands: CommandSlots::default(),
            git_gone: AtomicBool::new(false),
            git_gone_for_the_checkpoint: AtomicBool::new(false),
        }
    }
}

/// One conversation over one folder: its staging, its review's parts and
/// the store its record is in.
struct Convo {
    folder: PathBuf,
    workspace: Workspace,
    runner: GitRunner,
    /// A runner whose `PATH` is an empty folder: git is not available.
    no_git: GitRunner,
    store: SidecarStore,
    staging: Staging,
    checkpoints: Checkpoints,
    /// The conversation's writer lease (row E5).
    lease: WriterLease,
    guards: Guards,
    confirm: RecordingConfirm,
    confirmer: Confirmer,
    clock: Clock,
    mode: Mode,
    turn: String,
    call: String,
    scratch: Scratch,
}

fn plain(scratch: &Scratch) -> PathBuf {
    let folder = scratch.path().join("plain");
    std::fs::create_dir_all(folder.join("sub")).unwrap();
    std::fs::write(folder.join("a.txt"), "old\nkeep\n").unwrap();
    std::fs::write(folder.join("gone.txt"), "here\n").unwrap();
    folder
}

fn repo(scratch: &Scratch) -> PathBuf {
    let repo = scratch.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    scratch.git(&repo, &["init", "-q"]);
    std::fs::write(repo.join("a.txt"), "old\nkeep\n").unwrap();
    std::fs::write(repo.join("gone.txt"), "here\n").unwrap();
    scratch.git(&repo, &["add", "."]);
    scratch.git(&repo, &["commit", "-q", "-m", "one"]);
    repo
}

impl Convo {
    fn new(tag: &str, make: fn(&Scratch) -> PathBuf) -> Self {
        let scratch = Scratch::new(tag);
        let folder = make(&scratch);
        let workspace = attach_path(
            &folder,
            &scratch.env(),
            &StateRoot::at(scratch.path().join("state")),
            &scratch.runner(),
        )
        .unwrap();
        let store = SidecarStore::new(scratch.path().join("state").join("chat"));
        let (sidecar, _) = store
            .open_for_writing(
                ID,
                1.25,
                NewMeta {
                    workspace: None,
                    mode: Mode::Agent,
                    origin: Origin::Native,
                },
            )
            .unwrap();
        let confirm = RecordingConfirm::answering(true);
        let empty = scratch.path().join("no-git-here");
        std::fs::create_dir_all(&empty).unwrap();
        let mut env = scratch.env();
        env.set("PATH", empty.as_os_str());
        let no_git = GitRunner::new(Arc::new(env), &StateRoot::at(scratch.path().join("state")));
        let sidecar = Arc::new(sidecar);
        let clock: Clock = Arc::new(|| 1000.0);
        let lease = WriterLease::for_workspace(
            &workspace,
            &scratch.runner(),
            &StateRoot::at(scratch.path().join("state")),
            Platform::host(),
        );
        Self {
            lease,
            runner: scratch.runner(),
            no_git,
            workspace,
            folder,
            store,
            checkpoints: Checkpoints::new(Arc::clone(&sidecar), Arc::clone(&clock)),
            staging: Staging::new(sidecar),
            guards: Guards::default(),
            confirmer: Confirmer::new(Arc::new(confirm.clone())),
            confirm,
            clock,
            mode: Mode::Agent,
            turn: "a1b2c3d4e5f6".into(),
            call: "call_1".into(),
            scratch,
        }
    }

    fn stage_ctx(&self) -> StageContext<'_> {
        StageContext {
            workspace: &self.workspace,
            runner: &self.runner,
            staging: &self.staging,
            mode: Mode::Agent,
            trusted: true,
            turn: &self.turn,
            call: &self.call,
        }
    }

    fn edit(&self, path: &str, old: &str, new: &str) {
        edit_file(
            &self.stage_ctx(),
            &EditFileArgs {
                path: path.into(),
                old_string: old.into(),
                new_string: new.into(),
                replace_all: false,
            },
        )
        .unwrap();
    }

    /// `read_file` (so a replace is allowed), then `write_file`.
    fn write(&self, path: &str, content: &str) {
        let _ = read_file(
            &ReadContext {
                workspace: &self.workspace,
                runner: &self.runner,
                overlay: &self.staging,
            },
            &ReadFileArgs {
                path: path.into(),
                ..ReadFileArgs::default()
            },
        );
        write_file(
            &self.stage_ctx(),
            &WriteFileArgs {
                path: path.into(),
                content: content.into(),
            },
        )
        .unwrap();
    }

    fn delete(&self, path: &str) {
        delete_file(&self.stage_ctx(), &DeleteFileArgs { path: path.into() }).unwrap();
    }

    fn change(&self, path: &str) -> StagedChange {
        self.staging
            .changes()
            .into_iter()
            .rev()
            .find(|change| change.path == path)
            .unwrap_or_else(|| panic!("no change at {path}"))
    }

    /// The runner the review uses: one without git when git is gone.
    fn review_runner(&self) -> &GitRunner {
        if self.guards.git_gone.load(Ordering::SeqCst) {
            &self.no_git
        } else {
            &self.runner
        }
    }

    fn review(&self, ops: Vec<ReviewOp>) -> ReviewOutcome {
        let checkpoint_runner = if self
            .guards
            .git_gone_for_the_checkpoint
            .load(Ordering::SeqCst)
        {
            &self.no_git
        } else {
            self.review_runner()
        };
        let guards = FolderGuards {
            checkpoints: &self.checkpoints,
            workspace: &self.workspace,
            runner: checkpoint_runner,
            lease: &self.lease,
            commands: &self.guards.commands,
        };
        let ctx = ReviewContext {
            workspace: &self.workspace,
            runner: self.review_runner(),
            staging: &self.staging,
            mode: self.mode,
            trusted: true,
            guards: &guards,
            confirmer: &self.confirmer,
            clock: &self.clock,
        };
        block_on(review(&ctx, ops))
    }

    /// Stage the restore to checkpoint `to` (row E4).
    fn restore(&self, to: u32) -> Restored {
        restore(
            &RestoreContext {
                workspace: &self.workspace,
                runner: &self.runner,
                staging: &self.staging,
                checkpoints: &self.checkpoints,
            },
            to,
        )
        .unwrap()
    }

    fn keep(&self, path: &str) -> ReviewResult {
        let id = self.change(path).id;
        only(self.review(vec![ReviewOp::Keep {
            change: id,
            hunks: None,
        }]))
    }

    fn keep_hunks(&self, path: &str, hunks: Vec<String>) -> ReviewResult {
        let id = self.change(path).id;
        only(self.review(vec![ReviewOp::Keep {
            change: id,
            hunks: Some(hunks),
        }]))
    }

    fn disk(&self, path: &str) -> Option<Vec<u8>> {
        std::fs::read(self.folder.join(path)).ok()
    }

    fn items(&self) -> Vec<Item> {
        self.store.read_items(ID).unwrap().items
    }

    fn removed(&self) -> RemovedArea {
        RemovedArea::new(self.staging.sidecar().dir().join("removed"))
    }
}

fn only(outcome: ReviewOutcome) -> ReviewResult {
    assert_eq!(outcome.results.len(), 1, "{outcome:?}");
    outcome.results[0].result.clone()
}

/// Every file under `root` and its bytes' hash.
fn files(root: &Path) -> BTreeMap<String, String> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            if meta.is_dir() {
                walk(root, &path, out);
            } else if meta.is_file() {
                let name = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(name, sha256_hex(&std::fs::read(&path).unwrap_or_default()));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// NF1: every file's bytes before are still somewhere after (a kept file's
/// old bytes in `removed/` or a base copy), and no temporary or backup of
/// Lattice's is left outside `removed/`.
fn nothing_lost(before: &BTreeMap<String, String>, after: &BTreeMap<String, String>) {
    for (name, hash) in before {
        assert!(
            after.contains_key(name) || after.values().any(|other| other == hash),
            "{name} is gone and its bytes are nowhere"
        );
    }
    for name in after.keys() {
        assert!(
            !name.contains(".lattice-") || name.contains("/removed/"),
            "a temporary or a backup was left behind: {name}"
        );
    }
}

fn kept() -> ReviewResult {
    ReviewResult::Kept {
        changed_after: false,
    }
}

/// SF2: when the file changed on disk since it was staged, nothing is
/// written: a change made of edits is applied again to the new text and
/// becomes `Rebased` (kept by the next Keep), any other becomes a conflict;
/// the other writer's bytes survive either way.
/// Mutant: no expected-hash check.
#[test]
fn sf2_a_changed_disk_means_no_write() {
    let c = Convo::new("review-sf2", repo);
    let before = files(c.scratch.path());
    c.edit("a.txt", "old", "new");
    std::fs::write(c.folder.join("a.txt"), "old\nKEEP\n").unwrap();
    assert_eq!(c.keep("a.txt"), ReviewResult::Rebased);
    assert_eq!(c.disk("a.txt").unwrap(), b"old\nKEEP\n", "nothing written");
    assert_eq!(c.change("a.txt").state, ChangeState::Rebased);
    assert_eq!(c.keep("a.txt"), kept());
    assert_eq!(c.disk("a.txt").unwrap(), b"new\nKEEP\n", "both edits");

    c.write("gone.txt", "mine\n");
    std::fs::write(c.folder.join("gone.txt"), "theirs\n").unwrap();
    let reason = conflict_sentence("gone.txt");
    assert_eq!(
        c.keep("gone.txt"),
        ReviewResult::Conflict {
            reason: reason.clone()
        }
    );
    assert_eq!(c.disk("gone.txt").unwrap(), b"theirs\n");
    assert_eq!(c.change("gone.txt").state, ChangeState::Conflict { reason });
    assert!(matches!(c.keep("gone.txt"), ReviewResult::Skipped { .. }));
    nothing_lost(&before, &files(c.scratch.path()));
}

/// SF3: Keep All skips authority files, even with a reader who would
/// confirm, and keeps the rest, in path order, under one checkpoint.
/// Mutant: Keep All treating an authority file as a single Keep.
#[test]
fn sf3_keep_all_skips_authority_files() {
    let c = Convo::new("review-sf3", plain);
    std::fs::create_dir_all(c.folder.join(".github")).unwrap();
    c.edit("a.txt", "old", "new");
    c.write("Cargo.toml", "[package]\n");
    c.write(".github/ci.yml", "on: push\n");
    c.write("sub/b.txt", "b\n");
    let outcome = c.review(vec![ReviewOp::KeepAll]);
    let results: Vec<(String, ReviewResult)> = outcome
        .results
        .into_iter()
        .map(|result| (result.path, result.result))
        .collect();
    let skipped = ReviewResult::Skipped {
        reason: "This file changes how tools or Lattice behave; keep it on its own.".into(),
    };
    assert_eq!(
        results,
        vec![
            (".github/ci.yml".into(), skipped.clone()),
            ("Cargo.toml".into(), skipped),
            ("a.txt".into(), kept()),
            ("sub/b.txt".into(), kept()),
        ]
    );
    assert!(c.disk("Cargo.toml").is_none());
    assert!(c.disk(".github/ci.yml").is_none());
    assert_eq!(c.disk("a.txt").unwrap(), b"new\nkeep\n");
    assert!(c.confirm.asked().is_empty(), "no dialog for Keep All");
    let taken = c.checkpoints.all();
    assert_eq!(taken.len(), 1, "one checkpoint");
    assert_eq!(taken[0].kind, CheckpointKind::Copies);
    let copied: Vec<String> = c
        .checkpoints
        .manifest(taken[0].id)
        .unwrap()
        .into_iter()
        .map(|entry| entry.path)
        .collect();
    assert_eq!(
        copied,
        ["a.txt", "sub/b.txt"],
        "it copied what is written, and only that"
    );
}

/// `n` numbered CRLF lines.
fn crlf_lines(n: usize) -> String {
    (1..=n).map(|i| format!("line {i}\r\n")).collect()
}

/// SF4: a CRLF file with a byte-order mark keeps both through a hunk Keep
/// and a whole Keep.
/// Mutant: a hunk Keep composed with LF line ends.
#[test]
fn sf4_crlf_and_the_bom_are_kept() {
    let c = Convo::new("review-sf4", plain);
    let original = crlf_lines(30);
    std::fs::write(c.folder.join("w.txt"), [BOM, original.as_bytes()].concat()).unwrap();
    c.edit("w.txt", "line 2\n", "line two\n");
    c.edit("w.txt", "line 28\n", "line twenty-eight\n");
    let diff = file_diff(&c.staging, &c.change("w.txt").id).unwrap();
    assert_eq!(diff.hunks.len(), 2);
    assert_eq!(
        c.keep_hunks("w.txt", vec![diff.hunks[0].id.clone()]),
        ReviewResult::PartlyKept { remaining_hunks: 1 }
    );
    let first = original.replace("line 2\r\n", "line two\r\n");
    assert_eq!(c.disk("w.txt").unwrap(), [BOM, first.as_bytes()].concat());
    assert_eq!(c.keep("w.txt"), kept());
    let both = first.replace("line 28\r\n", "line twenty-eight\r\n");
    assert_eq!(c.disk("w.txt").unwrap(), [BOM, both.as_bytes()].concat());
}

/// A small deterministic generator (no new dependency).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self, below: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % below.max(1)
    }
}

/// The changed lines of some hunks, in order.
fn changed(hunks: &[Hunk]) -> Vec<(DiffLineKind, String)> {
    hunks
        .iter()
        .flat_map(|hunk| hunk.lines.iter())
        .filter(|line| line.kind != DiffLineKind::Context)
        .map(|line| (line.kind, line.text.clone()))
        .collect()
}

/// SF5: a partial Keep puts exactly the kept hunks on disk, and the diff
/// that remains is exactly the unkept hunks: a property test over random
/// edits and random choices of hunks (a fixed seed, 40 cases).
/// Mutant: the remaining diff taken against the original base.
#[test]
fn sf5_a_partial_keep_leaves_exactly_the_unkept_hunks() {
    let c = Convo::new("review-sf5", plain);
    let mut rng = Lcg(0x5f5);
    let mut cases = 0;
    for case in 0..40 {
        let base: Vec<String> = (0..40).map(|i| format!("{case} {i}\n")).collect();
        let mut new = base.clone();
        for _ in 0..(2 + rng.next(5)) {
            let at = rng.next(new.len());
            match rng.next(3) {
                0 => new[at] = format!("changed {}\n", rng.next(1000)),
                1 => new.insert(at, format!("added {}\n", rng.next(1000))),
                _ => {
                    new.remove(at);
                }
            }
        }
        let (base, new) = (base.concat(), new.concat());
        let path = format!("sub/p{case}.txt");
        std::fs::write(c.folder.join(&path), &base).unwrap();
        c.write(&path, &new);
        let id = c.change(&path).id;
        let diff = file_diff(&c.staging, &id).unwrap();
        if diff.hunks.len() < 2 {
            continue;
        }
        cases += 1;
        let mut take: Vec<String> = diff
            .hunks
            .iter()
            .filter(|_| rng.next(2) == 0)
            .map(|hunk| hunk.id.clone())
            .collect();
        if take.is_empty() || take.len() == diff.hunks.len() {
            take = vec![diff.hunks[0].id.clone()];
        }
        let unkept: Vec<Hunk> = diff
            .hunks
            .iter()
            .filter(|hunk| !take.contains(&hunk.id))
            .cloned()
            .collect();
        let result = c.keep_hunks(&path, take.clone());
        assert!(
            matches!(result, ReviewResult::PartlyKept { .. }),
            "{case}: {result:?}"
        );
        let plan = diff::plan(&id, &sha256_hex(base.as_bytes()), &base, &new);
        let chosen: BTreeSet<String> = take.into_iter().collect();
        assert_eq!(
            String::from_utf8(c.disk(&path).unwrap()).unwrap(),
            diff::apply(&plan, &base, &new, &chosen),
            "{case}: the kept hunks are on disk"
        );
        let remaining = file_diff(&c.staging, &id).unwrap();
        assert_eq!(
            changed(&remaining.hunks),
            changed(&unkept),
            "{case}: the remaining diff is the unkept hunks"
        );
        assert_eq!(c.keep(&path), kept());
        assert_eq!(c.disk(&path).unwrap(), new.as_bytes());
    }
    assert!(cases >= 20, "only {cases} cases had two hunks or more");
}

/// SF6: the Keep of a delete moves the file into `removed/` with a manifest
/// line and a `moved_aside` item; its bytes are there.
/// Mutant: unlinking.
#[test]
fn sf6_keep_of_a_delete_moves_the_file_aside() {
    let c = Convo::new("review-sf6", repo);
    let before = files(c.scratch.path());
    c.delete("gone.txt");
    assert_eq!(c.keep("gone.txt"), kept());
    assert!(c.disk("gone.txt").is_none());
    let manifest = c.removed().manifest().unwrap();
    assert_eq!(manifest.len(), 1);
    assert_eq!(manifest[0].path, "gone.txt");
    assert_eq!(manifest[0].why, MoveReason::KeptDelete);
    assert_eq!(
        std::fs::read(c.removed().root().join(&manifest[0].to)).unwrap(),
        b"here\n"
    );
    assert!(c.items().iter().any(|item| matches!(
        item,
        Item::MovedAside { path, why: MoveReason::KeptDelete, .. } if path == "gone.txt"
    )));
    nothing_lost(&before, &files(c.scratch.path()));
}

/// SF7: a restore (row E4, staged by `changes::restore`) of a file made
/// after its checkpoint is a deletion whose Keep moves the file aside, never
/// unlinks it; the Keep takes a `before_restore_keep` checkpoint first, and
/// the restored edit is back.
/// Mutant: unlinking.
#[test]
fn sf7_a_restore_moves_created_files_aside() {
    let c = Convo::new("review-sf7", repo);
    c.edit("a.txt", "old", "new");
    assert_eq!(c.keep("a.txt"), kept());
    std::fs::write(c.folder.join("made.txt"), "made later\n").unwrap();
    let before = files(c.scratch.path());
    let restored = c.restore(1);
    let staged: Vec<(String, ChangeKind, ChangeOrigin, bool)> = restored
        .staged
        .iter()
        .map(|change| {
            (
                change.path.clone(),
                change.kind,
                change.origin.clone(),
                change.new == NewState::Deleted,
            )
        })
        .collect();
    assert_eq!(
        staged,
        vec![
            (
                "a.txt".to_owned(),
                ChangeKind::Restore,
                ChangeOrigin::Restore { to: 1 },
                false
            ),
            (
                "made.txt".to_owned(),
                ChangeKind::Restore,
                ChangeOrigin::Restore { to: 1 },
                true
            ),
        ]
    );
    assert!(c.disk("made.txt").is_some(), "staging writes nothing");
    assert_eq!(c.keep("made.txt"), kept());
    assert_eq!(
        c.checkpoints.all().last().unwrap().reason,
        CheckpointReason::BeforeRestoreKeep
    );
    assert_eq!(c.keep("a.txt"), kept());
    assert_eq!(
        c.disk("a.txt").unwrap(),
        b"old\nkeep\n",
        "the restored bytes"
    );
    assert!(c.disk("made.txt").is_none());
    let manifest = c.removed().manifest().unwrap();
    let moved = manifest
        .iter()
        .find(|line| line.path == "made.txt")
        .expect("made.txt was moved aside");
    assert_eq!(moved.why, MoveReason::KeptRestore);
    assert_eq!(
        std::fs::read(c.removed().root().join(&moved.to)).unwrap(),
        b"made later\n"
    );
    nothing_lost(&before, &files(c.scratch.path()));
}

/// SF11: when git is unavailable the real checkpoint fails, and nothing is
/// written, for a Keep and for Keep All: first with git gone for the
/// checkpoint alone (the rest of the review could write), then with git
/// gone for the whole review.
/// Mutant: proceeding without the checkpoint.
#[test]
fn sf11_no_checkpoint_means_nothing_written() {
    let c = Convo::new("review-sf11", repo);
    let before = files(&c.folder);
    c.edit("a.txt", "old", "new");
    c.delete("gone.txt");
    let failed = ReviewResult::Failed {
        reason: "Lattice could not record the folder's state first, so nothing was written.".into(),
    };
    for gone in [&c.guards.git_gone_for_the_checkpoint, &c.guards.git_gone] {
        gone.store(true, Ordering::SeqCst);
        assert_eq!(c.keep("a.txt"), failed);
        assert_eq!(files(&c.folder), before, "nothing was written");
        let all = c.review(vec![ReviewOp::KeepAll]);
        assert_eq!(all.results.len(), 2);
        assert!(
            all.results.iter().all(|result| result.result == failed),
            "{all:?}"
        );
        assert_eq!(files(&c.folder), before, "nothing was written");
        assert_eq!(c.change("a.txt").state, ChangeState::Pending);
        assert!(c.checkpoints.all().is_empty(), "no checkpoint was recorded");
    }
}

/// SF13: `ReplaceFileW` failing with 1176, then with 1177 (injected through
/// the lattice-sys seam): the target path is never left empty, no temporary
/// or backup stays beside it, and the old and new bytes are both
/// recoverable.
/// Mutant: 1177 handled as 1176 (the target left under the backup name).
#[test]
fn sf13_replace_file_partial_failures_lose_nothing() {
    let c = Convo::new("review-sf13", plain);
    let before = files(c.scratch.path());
    c.edit("a.txt", "old", "new");
    let target = c.workspace.root.join("a.txt");
    inject_replace_fault(&target, ReplaceFault::UnableToMoveReplacement);
    assert_eq!(
        c.keep("a.txt"),
        ReviewResult::Failed {
            reason: "That file could not be written; it is unchanged.".into()
        }
    );
    assert_eq!(c.disk("a.txt").unwrap(), b"old\nkeep\n");
    assert_eq!(c.change("a.txt").state, ChangeState::Pending);
    nothing_lost(&before, &files(c.scratch.path()));

    inject_replace_fault(&target, ReplaceFault::UnableToMoveReplacement2);
    assert_eq!(c.keep("a.txt"), kept());
    assert_eq!(
        c.disk("a.txt").unwrap(),
        b"new\nkeep\n",
        "the path is filled"
    );
    let manifest = c.removed().manifest().unwrap();
    assert_eq!(manifest.len(), 1);
    assert_eq!(manifest[0].why, MoveReason::ReplacedBackup);
    assert_eq!(
        std::fs::read(c.removed().root().join(&manifest[0].to)).unwrap(),
        b"old\nkeep\n",
        "the old bytes"
    );
    nothing_lost(&before, &files(c.scratch.path()));
}

/// Stage an overwrite of `path` (on disk, possibly over 2 MiB, so its base
/// is read in pieces) with `new`, as `write_file` would after a read.
fn stage_overwrite(c: &Convo, path: &str, new: &[u8]) {
    let file = std::fs::File::open(c.folder.join(path)).unwrap();
    c.staging
        .record(
            Proposal {
                path: path.into(),
                authority: false,
                base: Base::of_reader(file).unwrap(),
                new: Some(new.to_vec()),
                action: Action::Write,
            },
            &c.turn,
            &c.call,
        )
        .unwrap();
}

/// The last checkpoint's omitted paths and why.
fn last_omitted(c: &Convo) -> Vec<(String, OmitWhy)> {
    let last = c.checkpoints.all().last().unwrap().id;
    c.checkpoints
        .omitted(last)
        .unwrap()
        .into_iter()
        .map(|omitted| (omitted.path, omitted.why))
        .collect()
}

/// SF14: a Keep over a 17 MiB file, which the real checkpoint omits for
/// size, and over an untracked file past the 2,000-file cap, which it
/// records as not taken: the pre-Keep bytes are recovered from the base copy
/// the `reviewed` item names. A 65 MiB target is refused and left
/// unchanged.
/// Mutant: a Keep that relies on the checkpoint alone (no base copy).
#[test]
fn sf14_the_overwritten_bytes_are_kept_first() {
    let c = Convo::new("review-sf14", repo);
    let big: Vec<u8> = b"0123456789abcdef".repeat(17 * 1024 * 1024 / 16);
    std::fs::write(c.folder.join("big.bin"), &big).unwrap();
    stage_overwrite(&c, "big.bin", b"small now\n");
    assert_eq!(c.keep("big.bin"), kept());
    assert_eq!(c.disk("big.bin").unwrap(), b"small now\n");
    let copy = c
        .items()
        .into_iter()
        .find_map(|item| match item {
            Item::Reviewed { base_copy, .. } => base_copy,
            _ => None,
        })
        .expect("the reviewed item names a base copy");
    assert_eq!(copy, sha256_hex(&big));
    assert!(
        c.staging.sidecar().read_blob(&copy).unwrap() == big,
        "the pre-Keep bytes are recovered from the base copy"
    );
    assert_eq!(
        last_omitted(&c),
        [("big.bin".to_owned(), OmitWhy::TooLarge)],
        "the checkpoint omitted it"
    );

    std::fs::create_dir_all(c.folder.join("many")).unwrap();
    for n in 0..2000 {
        std::fs::write(c.folder.join("many").join(format!("{n:04}.txt")), "x").unwrap();
    }
    std::fs::write(c.folder.join("zz.txt"), "past the cap\n").unwrap();
    stage_overwrite(&c, "zz.txt", b"kept\n");
    assert_eq!(c.keep("zz.txt"), kept());
    assert_eq!(
        last_omitted(&c),
        [
            ("many/1999.txt".to_owned(), OmitWhy::UntrackedCap),
            ("zz.txt".to_owned(), OmitWhy::UntrackedCap),
        ],
        "big.bin and the first 1,999 are taken; the rest are named"
    );
    let copy = c
        .items()
        .into_iter()
        .rev()
        .find_map(|item| match item {
            Item::Reviewed { base_copy, .. } => base_copy,
            _ => None,
        })
        .unwrap();
    assert_eq!(
        c.staging.sidecar().read_blob(&copy).unwrap(),
        b"past the cap\n"
    );

    let huge = vec![b'x'; 65 * 1024 * 1024];
    std::fs::write(c.folder.join("huge.bin"), &huge).unwrap();
    let hash = sha256_hex(&huge);
    drop(huge);
    stage_overwrite(&c, "huge.bin", b"tiny\n");
    assert_eq!(
        c.keep("huge.bin"),
        ReviewResult::Conflict {
            reason: "That file is too large for Lattice to keep a copy of, so nothing was written."
                .into()
        }
    );
    assert_eq!(sha256_hex(&c.disk("huge.bin").unwrap()), hash, "unchanged");
}

/// SF15: the Keep of an authority file asks the reader in the native
/// `KeepAuthority` dialog; refused, the file's bytes are unchanged and the
/// page cannot ask again for the same change (CP3); a fresh change asks
/// again, and confirmed, it is written.
/// Mutant: the dialog's answer ignored.
#[test]
fn sf15_an_authority_keep_needs_the_readers_confirmation() {
    let c = Convo::new("review-sf15", plain);
    std::fs::write(c.folder.join("Cargo.toml"), "[package]\nname = \"a\"\n").unwrap();
    c.edit("Cargo.toml", "\"a\"", "\"b\"");
    *c.confirm.answer.lock().unwrap() = false;
    let refused = ReviewResult::Skipped {
        reason: "You did not confirm keeping this file, so nothing was written.".into(),
    };
    assert_eq!(c.keep("Cargo.toml"), refused);
    assert_eq!(c.disk("Cargo.toml").unwrap(), b"[package]\nname = \"a\"\n");
    assert_eq!(
        c.confirm.asked(),
        vec![ConfirmRequest::KeepAuthority {
            path: "Cargo.toml".into(),
            added: 1,
            removed: 1,
        }]
    );
    *c.confirm.answer.lock().unwrap() = true;
    assert_eq!(c.keep("Cargo.toml"), refused, "CP3: not asked again");
    assert_eq!(c.confirm.asked().len(), 1);
    c.edit("Cargo.toml", "\"b\"", "\"c\"");
    assert_eq!(c.keep("Cargo.toml"), kept(), "a fresh change asks again");
    assert_eq!(c.disk("Cargo.toml").unwrap(), b"[package]\nname = \"c\"\n");
}

/// Undo writes nothing: per hunk (the hunk leaves the staged bytes), whole,
/// and for all; a note is kept, cut at 2,000 characters.
#[test]
fn undo_writes_nothing() {
    let c = Convo::new("review-undo", plain);
    std::fs::write(c.folder.join("w.txt"), crlf_lines(30)).unwrap();
    let before = files(&c.folder);
    c.edit("w.txt", "line 2\n", "line two\n");
    c.edit("w.txt", "line 28\n", "line twenty-eight\n");
    c.edit("a.txt", "old", "new");
    c.delete("gone.txt");
    let id = c.change("w.txt").id;
    let diff = file_diff(&c.staging, &id).unwrap();
    let outcome = c.review(vec![ReviewOp::Undo {
        change: id.clone(),
        hunks: Some(vec![diff.hunks[0].id.clone()]),
        note: Some("n".repeat(MAX_NOTE_CHARS + 10)),
    }]);
    assert_eq!(only(outcome), ReviewResult::Undone);
    assert_eq!(c.change("w.txt").state, ChangeState::Pending);
    let left = file_diff(&c.staging, &id).unwrap();
    assert_eq!(left.hunks.len(), 1);
    assert!(
        left.hunks[0]
            .lines
            .iter()
            .any(|line| line.text == "line twenty-eight")
    );
    let note = c.items().into_iter().find_map(|item| match item {
        Item::Reviewed {
            op: ReviewOp::Undo { note, .. },
            ..
        } => note,
        _ => None,
    });
    assert_eq!(note.unwrap().chars().count(), MAX_NOTE_CHARS);
    let gone = c.change("gone.txt").id;
    assert_eq!(
        only(c.review(vec![ReviewOp::Undo {
            change: gone,
            hunks: None,
            note: None,
        }])),
        ReviewResult::Undone
    );
    let all = c.review(vec![ReviewOp::UndoAll { note: None }]);
    assert_eq!(all.results.len(), 2, "{all:?}");
    assert!(
        c.staging
            .changes()
            .iter()
            .all(|change| change.state == ChangeState::Undone)
    );
    assert_eq!(c.staging.waiting(), 0);
    assert_eq!(files(&c.folder), before, "Undo wrote nothing");
}

/// The preconditions (Ask mode, a command running, the lease held
/// elsewhere), a creation whose path filled meanwhile, and the Keep's state
/// read back by a reopened conversation.
#[test]
fn keep_preconditions_and_the_record() {
    let mut c = Convo::new("review-pre", plain);
    let before = files(&c.folder);
    c.edit("a.txt", "old", "new");
    c.mode = Mode::Ask;
    assert_eq!(
        c.keep("a.txt"),
        ReviewResult::Failed {
            reason: "That is not available in Ask mode.".into()
        }
    );
    c.mode = Mode::Agent;
    // The folder's command slot, held as a running command holds it (a real
    // command running across a Keep is changes::effects_tests' CF13).
    let running = c
        .guards
        .commands
        .claim(&c.workspace.id, "call_running")
        .unwrap();
    assert_eq!(
        c.keep("a.txt"),
        ReviewResult::Conflict {
            reason: "A command is still running in this folder; Keep when it has finished.".into()
        }
    );
    drop(running);
    // The Keep refused above had taken this conversation's lease; it gives
    // it back, and another holder (a second conversation) takes it.
    assert!(c.lease.is_held());
    assert!(c.lease.release_when_idle(0, false));
    let other = WriterLease::at(Ok(c.lease.path().unwrap().to_path_buf()));
    assert_eq!(other.take(), Lease::Held);
    assert_eq!(
        c.keep("a.txt"),
        ReviewResult::Conflict {
            reason: "Another Lattice agent is editing this folder.".into()
        }
    );
    assert_eq!(files(&c.folder), before);
    assert!(other.release_when_idle(0, false));
    c.write("sub/new.txt", "fresh\n");
    assert_eq!(c.keep("a.txt"), kept());
    assert_eq!(c.keep("sub/new.txt"), kept());
    assert_eq!(c.disk("sub/new.txt").unwrap(), b"fresh\n");
    let reopened = Staging::from_items(Arc::clone(c.staging.sidecar()), &c.items());
    assert!(
        reopened
            .changes()
            .iter()
            .all(|change| change.state == ChangeState::Kept)
    );
    assert_eq!(reopened.waiting(), 0);
    c.write("sub/late.txt", "mine\n");
    std::fs::write(c.folder.join("sub").join("late.txt"), "theirs\n").unwrap();
    assert!(matches!(
        c.keep("sub/late.txt"),
        ReviewResult::Conflict { .. }
    ));
    assert_eq!(c.disk("sub/late.txt").unwrap(), b"theirs\n");
}

/// A native dialog that says it was asked, then waits for the test's word
/// before it confirms: the reader's KeepAuthority dialog left open while
/// something else happens.
struct HeldDialog {
    asked: std::sync::Mutex<Option<std::sync::mpsc::Sender<()>>>,
    go: std::sync::Mutex<Option<std::sync::mpsc::Receiver<()>>>,
}

impl HeldDialog {
    /// The dialog, the receiver that hears it was asked, and the sender that
    /// lets it confirm.
    fn new() -> (
        Arc<Self>,
        std::sync::mpsc::Receiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        let (asked_tx, asked_rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel();
        let dialog = Arc::new(Self {
            asked: std::sync::Mutex::new(Some(asked_tx)),
            go: std::sync::Mutex::new(Some(go_rx)),
        });
        (dialog, asked_rx, go_tx)
    }
}

impl crate::ports::ConfirmPort for HeldDialog {
    fn confirm(&self, _request: ConfirmRequest) -> futures::future::BoxFuture<'static, bool> {
        let asked = self.asked.lock().unwrap().take();
        let go = self.go.lock().unwrap().take();
        Box::pin(async move {
            if let Some(asked) = asked {
                let _ = asked.send(());
            }
            if let Some(go) = go {
                let _ = go.recv();
            }
            true
        })
    }
}

impl Convo {
    /// A review whose native dialogs are `confirmer`'s.
    fn review_with(&self, confirmer: &Confirmer, ops: Vec<ReviewOp>) -> ReviewOutcome {
        let guards = FolderGuards {
            checkpoints: &self.checkpoints,
            workspace: &self.workspace,
            runner: &self.runner,
            lease: &self.lease,
            commands: &self.guards.commands,
        };
        let ctx = ReviewContext {
            workspace: &self.workspace,
            runner: &self.runner,
            staging: &self.staging,
            mode: Mode::Agent,
            trusted: true,
            guards: &guards,
            confirmer,
            clock: &self.clock,
        };
        block_on(review(&ctx, ops))
    }

    /// Keep the change `id` in a review on another thread whose
    /// KeepAuthority dialog stays open while `meanwhile` runs here.
    fn keep_while_the_dialog_is_open<T>(
        &self,
        id: &str,
        meanwhile: impl FnOnce() -> T,
    ) -> (ReviewOutcome, T) {
        let (dialog, asked, go) = HeldDialog::new();
        let confirmer = Confirmer::new(dialog);
        std::thread::scope(|s| {
            let worker = s.spawn(|| {
                self.review_with(
                    &confirmer,
                    vec![ReviewOp::Keep {
                        change: id.to_owned(),
                        hunks: None,
                    }],
                )
            });
            asked.recv().unwrap();
            let done = meanwhile();
            go.send(()).unwrap();
            (worker.join().unwrap(), done)
        })
    }
}

/// F1 (verifier, CONFIRMED medium): the agent stages a further edit of an
/// authority file while the reader's KeepAuthority dialog is open. The
/// confirmed Keep writes nothing and says why; the later edit stays staged
/// (in memory and in the record) and is kept by the next Keep.
/// Falsifier: fails on the code before this fix (v1 written, the change Kept, v2 lost).
#[test]
fn f1_an_edit_staged_while_the_dialog_is_open_is_never_lost() {
    let c = Convo::new("review-f1", plain);
    let before = files(c.scratch.path());
    std::fs::write(c.folder.join("AGENTS.md"), "v0\n").unwrap();
    c.edit("AGENTS.md", "v0", "v1");
    let id = c.change("AGENTS.md").id;
    let (outcome, ()) = c.keep_while_the_dialog_is_open(&id, || c.edit("AGENTS.md", "v1", "v2"));
    assert_eq!(
        only(outcome),
        ReviewResult::Skipped {
            reason: "That change changed while you were confirming, so nothing was written; look at it again.".into()
        }
    );
    assert_eq!(c.disk("AGENTS.md").unwrap(), b"v0\n", "nothing written");
    let live = c
        .staging
        .live("AGENTS.md")
        .expect("the later edit is staged");
    assert_eq!(live.id, id);
    assert_eq!(live.state, ChangeState::Pending);
    assert_eq!(
        c.staging.snapshot(&id).unwrap().new.unwrap(),
        b"v2\n",
        "the later edit's bytes"
    );
    let reopened = Staging::from_items(Arc::clone(c.staging.sidecar()), &c.items());
    assert_eq!(
        reopened.snapshot(&id).unwrap().change.state,
        ChangeState::Pending,
        "the record agrees"
    );
    assert_eq!(c.keep("AGENTS.md"), kept(), "the next Keep keeps it");
    assert_eq!(c.disk("AGENTS.md").unwrap(), b"v2\n");
    nothing_lost(&before, &files(c.scratch.path()));
}

/// F1: `Staging::update` refuses a snapshot of a change that moved on since
/// it was read, and records nothing.
/// Falsifier: fails on the code before this fix (the stale snapshot overwrote the change).
#[test]
fn f1_update_refuses_a_stale_snapshot() {
    let c = Convo::new("review-f1-stale", plain);
    c.edit("a.txt", "old", "new");
    let id = c.change("a.txt").id;
    let mut stale = c.staging.snapshot(&id).unwrap();
    c.edit("a.txt", "keep", "kept");
    let items = c.items().len();
    stale.change.state = ChangeState::Kept;
    assert!(
        c.staging.update(stale).is_err(),
        "a stale snapshot is refused"
    );
    assert_eq!(c.items().len(), items, "nothing recorded");
    let now = c.staging.snapshot(&id).unwrap();
    assert_eq!(now.change.state, ChangeState::Pending);
    assert_eq!(now.new.unwrap(), b"new\nkept\n");
    let mut fresh = c.staging.snapshot(&id).unwrap();
    fresh.change.state = ChangeState::Undone;
    assert!(c.staging.update(fresh).is_ok(), "a fresh one is taken");
}

/// The real guards, noting whether the staging lock is held when the Keep
/// takes its checkpoint (inside the write's critical section).
struct LockWatch<'a> {
    inner: FolderGuards<'a>,
    staging: &'a Staging,
    held_at_checkpoint: std::sync::Mutex<Vec<bool>>,
}

impl super::review::KeepGuards for LockWatch<'_> {
    fn command_running(&self) -> bool {
        self.inner.command_running()
    }

    fn lease(&self) -> Lease {
        self.inner.lease()
    }

    fn checkpoint(
        &self,
        paths: &[String],
        reason: CheckpointReason,
    ) -> Result<lattice_protocol::conversation::CheckpointId, String> {
        self.held_at_checkpoint
            .lock()
            .unwrap()
            .push(self.staging.serial_is_held());
        self.inner.checkpoint(paths, reason)
    }
}

/// F1: the staging lock is held from the re-check through the write and the
/// change's update, so no staging tool runs between them.
/// Falsifier: the lock taken and dropped at once (mutant EFIX-1-m3).
#[test]
fn f1_the_staging_lock_is_held_through_the_write() {
    let c = Convo::new("review-f1-lock", plain);
    c.edit("a.txt", "old", "new");
    let id = c.change("a.txt").id;
    let watch = LockWatch {
        inner: FolderGuards {
            checkpoints: &c.checkpoints,
            workspace: &c.workspace,
            runner: &c.runner,
            lease: &c.lease,
            commands: &c.guards.commands,
        },
        staging: &c.staging,
        held_at_checkpoint: std::sync::Mutex::new(Vec::new()),
    };
    let ctx = ReviewContext {
        workspace: &c.workspace,
        runner: &c.runner,
        staging: &c.staging,
        mode: Mode::Agent,
        trusted: true,
        guards: &watch,
        confirmer: &c.confirmer,
        clock: &c.clock,
    };
    let outcome = block_on(review(
        &ctx,
        vec![ReviewOp::Keep {
            change: id,
            hunks: None,
        }],
    ));
    assert_eq!(only(outcome), kept());
    assert_eq!(*watch.held_at_checkpoint.lock().unwrap(), vec![true]);
    assert!(!c.staging.serial_is_held(), "and given back");
}

/// F3 (verifier, CONFIRMED medium): `docs/ci.yml` staged as an ordinary
/// file, then the empty `docs` folder swapped for an in-root junction to
/// `.github`. Keep All writes nothing into `.github` and asks no dialog; the
/// path's change is a conflict that stays staged.
/// Falsifier: fails on the code before this fix (`.github/ci.yml` written, no dialog).
#[test]
fn f3_keep_all_never_writes_an_authority_file_through_a_junction() {
    let c = Convo::new("review-f3", plain);
    let before = files(c.scratch.path());
    std::fs::create_dir_all(c.folder.join(".github")).unwrap();
    std::fs::create_dir_all(c.folder.join("docs")).unwrap();
    c.write("docs/ci.yml", "on: push\n");
    assert!(!c.change("docs/ci.yml").authority);
    lattice_sys::fs::seam::create_junction(&c.folder.join("docs"), &c.folder.join(".github"))
        .unwrap();
    let outcome = c.review(vec![ReviewOp::KeepAll]);
    assert_eq!(
        only(outcome),
        ReviewResult::Conflict {
            reason:
                "This path is now reached through a link to another place, so nothing was written."
                    .into()
        }
    );
    assert!(c.confirm.asked().is_empty());
    assert!(!c.folder.join(".github").join("ci.yml").exists());
    assert_eq!(c.change("docs/ci.yml").state, ChangeState::Pending);
    nothing_lost(&before, &files(c.scratch.path()));
}

/// F3: the same check on every Keep that writes: a single Keep (which would
/// have asked no dialog for an ordinary file), a restore's change and a
/// command effect's inverse, each staged under `docs/` before `docs` became
/// a junction to `.github`.
/// Falsifier: fails on the code before this fix (each written into `.github`).
#[test]
fn f3_every_keep_that_writes_checks_the_path_again() {
    use crate::changes::restore::{Back, stage_back};
    let c = Convo::new("review-f3-all", plain);
    std::fs::create_dir_all(c.folder.join(".github")).unwrap();
    std::fs::create_dir_all(c.folder.join("docs")).unwrap();
    c.write("docs/agent.yml", "a: 1\n");
    let staged = |path: &str, kind, origin| match stage_back(
        &c.workspace,
        &c.runner,
        &c.staging,
        path,
        Some(b"r: 1\n".to_vec()),
        kind,
        origin,
    )
    .unwrap()
    {
        Back::Staged(change) => change,
        other => panic!("{other:?}"),
    };
    let restore = staged(
        "docs/restore.yml",
        ChangeKind::Restore,
        ChangeOrigin::Restore { to: 1 },
    );
    let inverse = staged(
        "docs/inverse.yml",
        ChangeKind::CommandUndo,
        ChangeOrigin::CommandUndo {
            call: "call_cmd".into(),
        },
    );
    assert!(!restore.authority && !inverse.authority);
    lattice_sys::fs::seam::create_junction(&c.folder.join("docs"), &c.folder.join(".github"))
        .unwrap();
    let through = ReviewResult::Conflict {
        reason: "This path is now reached through a link to another place, so nothing was written."
            .into(),
    };
    for path in ["docs/agent.yml", "docs/restore.yml", "docs/inverse.yml"] {
        assert_eq!(c.keep(path), through, "{path}");
    }
    assert!(c.confirm.asked().is_empty());
    let written: Vec<_> = std::fs::read_dir(c.folder.join(".github"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert!(written.is_empty(), "nothing in .github: {written:?}");
}

/// F3: the class is taken again at the write. A change whose record says
/// ordinary (an older record, or a classifier that changed) at a path that
/// is an authority file now is not written without the dialog.
/// Falsifier: the class check dropped (mutant EFIX-2-m2).
#[test]
fn f3_a_path_that_is_authority_now_is_not_kept_as_ordinary() {
    use crate::convo::item::BaseState;
    let c = Convo::new("review-f3-class", plain);
    let bytes = b"[package]\n".to_vec();
    let change = StagedChange {
        id: "ch_0123456789abcdef".into(),
        path: "Cargo.toml".into(),
        kind: ChangeKind::Create,
        base: BaseState::Absent,
        new: NewState::Bytes {
            blob: sha256_hex(&bytes),
        },
        ops: Vec::new(),
        authority: false,
        origin: ChangeOrigin::Restore { to: 1 },
        state: ChangeState::Pending,
    };
    c.staging
        .update(super::Snapshot {
            change,
            base: None,
            new: Some(bytes),
            base_lines: 0,
            revision: 0,
        })
        .unwrap();
    assert_eq!(
        c.keep("Cargo.toml"),
        ReviewResult::Conflict {
            reason: "This path now names a file that changes how tools or Lattice behave; keep it on its own, after confirming it.".into()
        }
    );
    assert!(c.confirm.asked().is_empty());
    assert!(c.disk("Cargo.toml").is_none());
}

/// F2 (verifier, CONFIRMED low): the page sends Keep for one change twice;
/// the first waits in the KeepAuthority dialog while the second is
/// confirmed and written. The first then reports the change kept and writes
/// nothing: it stays Kept, its edit applied once.
/// Falsifier: fails on the code before this fix (the edit applied again, the change
/// Rebased, "ab" staged as "abb").
#[test]
fn f2_a_duplicate_keep_of_one_change_applies_it_once() {
    let c = Convo::new("review-f2", plain);
    std::fs::write(c.folder.join("AGENTS.md"), "a\n").unwrap();
    c.edit("AGENTS.md", "a", "ab");
    let id = c.change("AGENTS.md").id;
    let (first, second) = c.keep_while_the_dialog_is_open(&id, || c.keep("AGENTS.md"));
    assert_eq!(second, kept());
    assert_eq!(only(first), kept());
    assert_eq!(c.disk("AGENTS.md").unwrap(), b"ab\n");
    assert_eq!(c.change("AGENTS.md").state, ChangeState::Kept);
    assert_eq!(c.staging.waiting(), 0);
    assert!(c.staging.live("AGENTS.md").is_none());
}

/// F2, the crash variant: a Keep wrote the file and stopped before it
/// marked the change Kept (the record still says Pending). The next Keep
/// finds the bytes already there: Kept, nothing written, the edit not
/// applied a second time.
/// Falsifier: fails on the code before this fix (Rebased, "ab" staged as "abb").
#[test]
fn f2_a_keep_after_a_crash_between_the_write_and_the_update_writes_nothing() {
    let c = Convo::new("review-f2-crash", plain);
    c.edit("a.txt", "old", "olds");
    // What the interrupted Keep left: the staged bytes in place.
    std::fs::write(c.folder.join("a.txt"), "olds\nkeep\n").unwrap();
    assert_eq!(c.keep("a.txt"), kept());
    assert_eq!(c.disk("a.txt").unwrap(), b"olds\nkeep\n");
    assert_eq!(c.change("a.txt").state, ChangeState::Kept);
    assert!(
        c.removed().manifest().unwrap_or_default().is_empty(),
        "nothing replaced"
    );
    // And for a delete whose file is already gone.
    c.delete("gone.txt");
    let moved = c.scratch.path().join("gone.txt.elsewhere");
    std::fs::rename(c.folder.join("gone.txt"), &moved).unwrap();
    assert_eq!(c.keep("gone.txt"), kept());
    assert_eq!(
        std::fs::read(&moved).unwrap(),
        b"here
"
    );
}

/// SF13, 1177's fallback (verifier's surviving mutant m2): the target is
/// under the backup name and the staged bytes cannot be moved into place,
/// so the backup is moved back: the path holds its old bytes again, nothing
/// is written, the change stays staged, and nothing is lost.
/// Falsifier: the backup never moved back (VERIFY-m2, re-run as EFIX-4-m1).
#[test]
fn sf13_a_1177_whose_placement_fails_moves_the_backup_back() {
    let c = Convo::new("review-sf13-back", plain);
    let before = files(c.scratch.path());
    c.edit("a.txt", "old", "new");
    let target = c.workspace.root.join("a.txt");
    inject_replace_fault(&target, ReplaceFault::UnableToMoveReplacement2);
    lattice_sys::fs::seam::inject_move_fault(&target);
    assert_eq!(
        c.keep("a.txt"),
        ReviewResult::Failed {
            reason: "That file could not be written; it is unchanged.".into()
        }
    );
    assert_eq!(
        c.disk("a.txt").unwrap(),
        b"old\nkeep\n",
        "the path is filled again"
    );
    assert_eq!(c.change("a.txt").state, ChangeState::Pending);
    nothing_lost(&before, &files(c.scratch.path()));
    assert_eq!(c.keep("a.txt"), kept(), "a later Keep writes");
    assert_eq!(c.disk("a.txt").unwrap(), b"new\nkeep\n");
}

/// SF13, 1177 when neither file can be moved back into the path: the
/// reader is told where both are, and both keep their bytes.
#[test]
fn sf13_a_1177_whose_both_moves_fail_names_where_the_bytes_are() {
    let c = Convo::new("review-sf13-both", plain);
    c.edit("a.txt", "old", "new");
    let target = c.workspace.root.join("a.txt");
    inject_replace_fault(&target, ReplaceFault::UnableToMoveReplacement2);
    lattice_sys::fs::seam::inject_move_fault(&target);
    lattice_sys::fs::seam::inject_move_fault(&target);
    let ReviewResult::Failed { reason } = c.keep("a.txt") else {
        panic!("not a failure");
    };
    assert!(
        reason.starts_with(
            "That file could not be written; its old bytes are beside it as .a.txt.lattice-bak-"
        ),
        "{reason}"
    );
    let beside: BTreeMap<String, Vec<u8>> = std::fs::read_dir(&c.folder)
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_name().to_string_lossy().contains(".lattice-"))
        .map(|entry| {
            (
                entry.file_name().to_string_lossy().into_owned(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect();
    let bytes: BTreeSet<Vec<u8>> = beside.values().cloned().collect();
    assert_eq!(beside.len(), 2, "{beside:?}");
    assert!(bytes.contains(b"old\nkeep\n".as_slice()) && bytes.contains(b"new\nkeep\n".as_slice()));
    for name in beside.keys() {
        assert!(reason.contains(name.as_str()), "{name} named in {reason}");
    }
    assert!(c.disk("a.txt").is_none());
    assert_eq!(c.change("a.txt").state, ChangeState::Pending);
}
