//! Checkpoints around commands and their effects, against scratch
//! repositories and folders this file makes (the chat core's spec §8.4,
//! §7.5's X14; §16.4 KF4, CF13, and KF1 and KF8 for a command's effect).
//!
//! Every folder is temporary; git runs only in scratch repositories, through
//! the runner (the setup's own git and the positive controls aside). The
//! commands are the ones these tests write, run by Windows PowerShell 5.1
//! inside the temporary folder, each with a timeout, ended through its Job
//! Object.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::executor::block_on;
use lattice_protocol::conversation::{
    ChangeKind, ChangeOrigin, ChangeState, ChangedPath, CheckpointKind, CheckpointReason,
    ExitReason, Mode, Origin, PathChange, ReviewOp, ReviewResult,
};

use super::FolderGuards;
use super::checkpoint::Checkpoints;
use super::checkpoint_tests::{fetch_lines, plain, plain_git};
use super::effects::{CommandGuards, EXPOSED_NOTE, IGNORED_NOTE};
use crate::clock::Clock;
use crate::convo::item::{BaseState, Item, NewState, StagedChange};
use crate::convo::sidecar::{NewMeta, SidecarStore};
use crate::env::MapEnv;
use crate::exec::allowlist::Permissions;
use crate::exec::run::{
    Approval, CommandContext, CommandOutcome, CommandSlots, Prepared, RunCommandArgs,
    SpawnLauncher, StopHandle, approve, prepare, run,
};
use crate::fsx::RemovedArea;
use crate::git::runner::GitRunner;
use crate::git::tests::{Scratch, hostile, markers};
use crate::ports::Confirmer;
use crate::ports::fake::RecordingConfirm;
use crate::sha::sha256_hex;
use crate::staging::Staging;
use crate::staging::review::{ReviewContext, review};
use crate::state::{Platform, StateRoot};
use crate::tools::edit::{EditFileArgs, StageContext, edit_file};
use crate::workspace::Workspace;
use crate::workspace::attach::attach_path;
use crate::workspace::lease::WriterLease;

const ID: &str = "c0ffee000008";

/// One conversation over one folder, with its checkpoints, staging, lease,
/// the process's command slots and everything a command needs.
struct Rig {
    scratch: Scratch,
    folder: PathBuf,
    workspace: Workspace,
    runner: GitRunner,
    store: SidecarStore,
    staging: Staging,
    checkpoints: Checkpoints,
    lease: WriterLease,
    slots: CommandSlots,
    permissions: Permissions,
    confirmer: Confirmer,
    launcher: SpawnLauncher,
    env: MapEnv,
    globals: PathBuf,
    clock: Clock,
}

impl Rig {
    fn new(tag: &str, make: impl FnOnce(&Scratch) -> PathBuf) -> Self {
        Self::with_runner(tag, make, |scratch| scratch.runner())
    }

    fn with_runner(
        tag: &str,
        make: impl FnOnce(&Scratch) -> PathBuf,
        runner: impl FnOnce(&Scratch) -> GitRunner,
    ) -> Self {
        let scratch = Scratch::new(tag);
        let folder = make(&scratch);
        let runner = runner(&scratch);
        let state = StateRoot::at(scratch.path().join("state"));
        let workspace = attach_path(&folder, &scratch.env(), &state, &runner).unwrap();
        let store = SidecarStore::new(scratch.path().join("state").join("chat"));
        let (sidecar, _) = store
            .open_for_writing(
                ID,
                2.5,
                NewMeta {
                    workspace: None,
                    mode: Mode::Agent,
                    origin: Origin::Native,
                },
            )
            .unwrap();
        let sidecar = Arc::new(sidecar);
        let clock: Clock = Arc::new(|| 4000.0);
        let lease = WriterLease::for_workspace(&workspace, &runner, &state, Platform::host());
        let env = scratch.env();
        Self {
            folder,
            runner,
            checkpoints: Checkpoints::new(Arc::clone(&sidecar), Arc::clone(&clock)),
            staging: Staging::new(sidecar),
            store,
            lease,
            slots: CommandSlots::default(),
            permissions: Permissions::new(&state, Arc::clone(&clock)),
            confirmer: Confirmer::new(Arc::new(RecordingConfirm::answering(true))),
            launcher: SpawnLauncher {
                env: Arc::new(env.clone()),
                globals: state.globals.clone(),
            },
            env,
            globals: state.globals.clone(),
            workspace,
            clock,
            scratch,
        }
    }

    fn guards(&self) -> CommandGuards<'_> {
        CommandGuards {
            checkpoints: &self.checkpoints,
            workspace: &self.workspace,
            runner: &self.runner,
            staging: &self.staging,
            lease: &self.lease,
        }
    }

    fn ctx<'a>(&'a self, hooks: &'a CommandGuards<'a>) -> CommandContext<'a> {
        CommandContext {
            workspace: &self.workspace,
            runner: &self.runner,
            staging: &self.staging,
            mode: Mode::Agent,
            trusted: true,
            slots: &self.slots,
            permissions: &self.permissions,
            confirmer: &self.confirmer,
            launcher: &self.launcher,
            hooks,
            env: &self.env,
            globals: &self.globals,
            remote: None,
        }
    }

    fn prepare(&self, text: &str) -> Result<Prepared, String> {
        let hooks = self.guards();
        prepare(
            &self.ctx(&hooks),
            &RunCommandArgs {
                command: text.into(),
                cwd: None,
                timeout_s: Some(60),
                background: false,
            },
        )
        .map_err(|error| error.0)
    }

    /// The reader approves in the native dialog; `Err` is the refusal.
    fn approve(&self, prepared: &Prepared, call: &str) -> Result<Approval, String> {
        let hooks = self.guards();
        block_on(approve(&self.ctx(&hooks), prepared, &call.to_owned()))
            .map_err(|refused| refused.sentence)
    }

    fn run_approved(&self, prepared: &Prepared, approval: &Approval, call: &str) -> CommandOutcome {
        let hooks = self.guards();
        run(
            &self.ctx(&hooks),
            prepared,
            approval,
            &call.to_owned(),
            &StopHandle::default(),
        )
        .unwrap()
    }

    /// Prepare, approve and run a command.
    fn command(&self, text: &str, call: &str) -> CommandOutcome {
        let prepared = self.prepare(text).unwrap();
        let approval = self.approve(&prepared, call).unwrap();
        let outcome = self.run_approved(&prepared, &approval, call);
        assert_eq!(outcome.reason, ExitReason::Exited, "{}", outcome.model_text);
        assert_eq!(outcome.code, Some(0), "{}", outcome.model_text);
        outcome
    }

    fn review(&self, ops: Vec<ReviewOp>) -> Vec<(String, ReviewResult)> {
        let guards = FolderGuards {
            checkpoints: &self.checkpoints,
            workspace: &self.workspace,
            runner: &self.runner,
            lease: &self.lease,
            commands: &self.slots,
        };
        let ctx = ReviewContext {
            workspace: &self.workspace,
            runner: &self.runner,
            staging: &self.staging,
            mode: Mode::Agent,
            trusted: true,
            guards: &guards,
            confirmer: &self.confirmer,
            clock: &self.clock,
        };
        block_on(review(&ctx, ops))
            .results
            .into_iter()
            .map(|result| (result.path, result.result))
            .collect()
    }

    fn edit(&self, path: &str, old: &str, new: &str) {
        edit_file(
            &StageContext {
                workspace: &self.workspace,
                runner: &self.runner,
                staging: &self.staging,
                mode: Mode::Agent,
                trusted: true,
                turn: &"a1b2c3d4e5f6".to_owned(),
                call: &"call_edit".to_owned(),
            },
            &EditFileArgs {
                path: path.into(),
                old_string: old.into(),
                new_string: new.into(),
                replace_all: false,
            },
        )
        .unwrap();
    }

    /// The latest change at `path`.
    fn change_at(&self, path: &str) -> StagedChange {
        self.staging
            .changes()
            .into_iter()
            .rev()
            .find(|change| change.path == path)
            .unwrap_or_else(|| panic!("no change at {path}"))
    }

    /// The changes a command's effect recorded, by path.
    fn effect_changes(&self, call: &str) -> BTreeMap<String, StagedChange> {
        self.staging
            .changes()
            .into_iter()
            .filter(|change| {
                change.origin
                    == ChangeOrigin::Command {
                        call: call.to_owned(),
                    }
            })
            .map(|change| (change.path.clone(), change))
            .collect()
    }

    fn items(&self) -> Vec<Item> {
        self.store.read_items(ID).unwrap().items
    }

    fn disk(&self, path: &str) -> Option<Vec<u8>> {
        std::fs::read(self.folder.join(path)).ok()
    }

    fn removed(&self) -> RemovedArea {
        RemovedArea::new(self.staging.sidecar().dir().join("removed"))
    }
}

/// Every file under `root` (outside `.git`) with its SHA-256.
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
                if path.file_name().is_some_and(|name| name == ".git") {
                    continue;
                }
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

fn effect_files(outcome: &CommandOutcome) -> Vec<ChangedPath> {
    outcome
        .after
        .effect
        .as_ref()
        .map(|effect| effect.files.clone())
        .unwrap_or_default()
}

fn path(path: &str, change: PathChange) -> ChangedPath {
    ChangedPath {
        path: path.into(),
        change,
    }
}

/// KF4: a command's creation, change and deletion are listed (and nothing
/// git ignores); each is a change already on disk; Undo stages the inverse
/// and writes nothing; Keep of those inverses restores the folder, the
/// created file moved aside with its bytes, never removed.
#[test]
fn kf4_a_commands_effect_is_listed_and_undone() {
    let r = Rig::new("fx-kf4", |scratch| {
        let repo = super::checkpoint_tests::repo(scratch);
        std::fs::write(
            repo.join(".gitignore"),
            "*.log
",
        )
        .unwrap();
        repo
    });
    let outcome = r.command(
        "Set-Content -LiteralPath new.txt -Value n -NoNewline; \
         Set-Content -LiteralPath a.txt -Value changed -NoNewline; \
         Remove-Item -LiteralPath gone.txt; \
         Set-Content -LiteralPath x.log -Value ignored",
        "call_k4",
    );
    assert_eq!(
        effect_files(&outcome),
        [
            path("a.txt", PathChange::Modified),
            path("gone.txt", PathChange::Deleted),
            path("new.txt", PathChange::Added),
        ]
    );
    let effect = outcome.after.effect.clone().unwrap();
    let before = r.checkpoints.get(effect.before).unwrap();
    let after = r.checkpoints.get(effect.after).unwrap();
    assert_eq!(
        (before.reason, after.reason),
        (
            CheckpointReason::BeforeCommand {
                call_id: "call_k4".into()
            },
            CheckpointReason::AfterCommand {
                call_id: "call_k4".into()
            }
        )
    );
    assert!(r.items().iter().any(|item| matches!(
        item,
        Item::CommandEffect { call_id, files, .. } if call_id == "call_k4" && files.len() == 3
    )));
    assert!(
        outcome.model_text.contains(
            "The command changed 3 files on disk: a.txt (modified), gone.txt (deleted), new.txt (added)."
        ),
        "{}",
        outcome.model_text
    );
    assert!(outcome.model_text.ends_with(IGNORED_NOTE));
    let changes = r.effect_changes("call_k4");
    assert_eq!(changes.len(), 3);
    assert!(changes.values().all(|c| c.state == ChangeState::OnDisk));
    assert_eq!(changes["a.txt"].kind, ChangeKind::Overwrite);
    assert_eq!(changes["gone.txt"].new, NewState::Deleted);
    assert_eq!(changes["new.txt"].base, BaseState::Absent);
    assert_eq!(
        r.staging.waiting(),
        0,
        "an effect does not shut the command gate"
    );

    // Undo stages each inverse and writes nothing.
    let disk_before = files(&r.folder);
    let undone = r.review(
        changes
            .values()
            .map(|change| ReviewOp::Undo {
                change: change.id.clone(),
                hunks: None,
                note: None,
            })
            .collect(),
    );
    assert!(
        undone
            .iter()
            .all(|(_, result)| *result == ReviewResult::Undone),
        "{undone:?}"
    );
    assert_eq!(files(&r.folder), disk_before, "Undo wrote nothing");
    let undo: BTreeMap<String, StagedChange> = r
        .staging
        .changes()
        .into_iter()
        .filter(|c| c.kind == ChangeKind::CommandUndo)
        .map(|c| (c.path.clone(), c))
        .collect();
    assert_eq!(undo.len(), 3);
    assert!(undo.values().all(|c| c.state == ChangeState::Pending
        && c.origin
            == ChangeOrigin::CommandUndo {
                call: "call_k4".into()
            }));
    assert_eq!(undo["new.txt"].new, NewState::Deleted);
    assert_eq!(undo["gone.txt"].base, BaseState::Absent);
    assert_eq!(r.staging.waiting(), 3, "the inverses wait like any change");
    assert!(
        r.effect_changes("call_k4")
            .values()
            .all(|c| c.state == ChangeState::Undone)
    );

    // Keep restores the folder.
    let kept = r.review(vec![ReviewOp::KeepAll]);
    assert!(
        kept.iter()
            .all(|(_, result)| matches!(result, ReviewResult::Kept { .. })),
        "{kept:?}"
    );
    assert_eq!(r.disk("a.txt").unwrap(), b"old\nkeep\n");
    assert_eq!(r.disk("gone.txt").unwrap(), b"here\n");
    assert_eq!(r.disk("new.txt"), None);
    let manifest = r.removed().manifest().unwrap();
    let moved = manifest.iter().find(|m| m.path == "new.txt").unwrap();
    assert_eq!(
        std::fs::read(r.removed().root().join(&moved.to)).unwrap(),
        b"n"
    );
    assert_eq!(
        r.disk("x.log").unwrap(),
        b"ignored\r\n",
        "an ignored file is left alone"
    );
}

/// KF4 and §8.4: Keep of a command's effect only acknowledges it: nothing
/// is written, no checkpoint is taken; it is no longer waiting.
#[test]
fn keep_of_an_effect_acknowledges_it_and_writes_nothing() {
    let r = Rig::new("fx-acked", super::checkpoint_tests::repo);
    r.command(
        "Set-Content -LiteralPath a.txt -Value mine -NoNewline",
        "call_ack",
    );
    let change = r.effect_changes("call_ack")["a.txt"].clone();
    let disk = files(&r.folder);
    let taken = r.checkpoints.all().len();
    let kept = r.review(vec![ReviewOp::Keep {
        change: change.id.clone(),
        hunks: None,
    }]);
    assert_eq!(
        kept,
        [(
            "a.txt".to_owned(),
            ReviewResult::Kept {
                changed_after: false
            }
        )]
    );
    assert_eq!(files(&r.folder), disk);
    assert_eq!(
        r.checkpoints.all().len(),
        taken,
        "no checkpoint for an acknowledgement"
    );
    assert_eq!(
        r.effect_changes("call_ack")["a.txt"].state,
        ChangeState::Kept
    );
    let again = r.review(vec![ReviewOp::Undo {
        change: change.id,
        hunks: None,
        note: None,
    }]);
    assert!(
        matches!(again[0].1, ReviewResult::Skipped { .. }),
        "{again:?}"
    );
}

/// A path a checkpoint omitted (here a 17 MiB file the command shrank) is
/// never listed as created or deleted: it is named as not tracked, so its
/// Undo can never move it aside.
#[test]
fn an_omitted_path_is_never_taken_for_a_created_one() {
    let r = Rig::new("fx-omit", super::checkpoint_tests::repo);
    std::fs::write(r.folder.join("big.bin"), vec![7u8; 17 * 1024 * 1024]).unwrap();
    let outcome = r.command(
        "Set-Content -LiteralPath big.bin -Value small -NoNewline; Set-Content -LiteralPath a.txt -Value x -NoNewline",
        "call_om",
    );
    assert_eq!(
        effect_files(&outcome),
        [path("a.txt", PathChange::Modified)]
    );
    assert!(
        outcome
            .model_text
            .contains("Not tracked (too large, too many new files, or links): big.bin."),
        "{}",
        outcome.model_text
    );
    assert!(!r.effect_changes("call_om").contains_key("big.bin"));
}

/// §8.3: without git a command's checkpoints are exposed and there is no
/// effect; the model is told.
#[test]
fn without_git_a_command_is_exposed_and_has_no_effect() {
    let r = Rig::new("fx-plain", plain);
    let outcome = r.command("Set-Content -LiteralPath a.txt -Value x", "call_pl");
    assert_eq!(outcome.after.effect, None);
    assert!(outcome.model_text.ends_with(EXPOSED_NOTE));
    let taken = r.checkpoints.all();
    assert_eq!(taken.len(), 2);
    assert!(
        taken
            .iter()
            .all(|t| t.exposed && t.kind == CheckpointKind::Copies)
    );
    assert!(r.effect_changes("call_pl").is_empty());
}

/// KF1, the command effect's part: a hostile repository (hooks, an
/// fsmonitor command, clean, smudge and process filters) runs none of its
/// programs during the checkpoints around a command, the comparison, the
/// Undo and the inverse's Keep.
#[test]
fn kf1_a_command_effect_runs_none_of_the_repositorys_programs() {
    let r = Rig::new("fx-kf1", |scratch| hostile(scratch, "hostile").0);
    let repo_markers = r.scratch.path().join("hostile-markers");
    let outcome = r.command(
        "Set-Content -LiteralPath b.txt -Value changed -NoNewline; Set-Content -LiteralPath c.txt -Value c -NoNewline",
        "call_h",
    );
    assert_eq!(
        markers(&repo_markers),
        Vec::<String>::new(),
        "the checkpoints and the comparison ran none of the repository's programs"
    );
    assert_eq!(
        effect_files(&outcome),
        [
            path("b.txt", PathChange::Modified),
            path("c.txt", PathChange::Added)
        ]
    );
    let ops = r
        .effect_changes("call_h")
        .values()
        .map(|change| ReviewOp::Undo {
            change: change.id.clone(),
            hunks: None,
            note: None,
        })
        .collect();
    r.review(ops);
    let kept = r.review(vec![ReviewOp::KeepAll]);
    assert!(
        kept.iter()
            .all(|(_, result)| matches!(result, ReviewResult::Kept { .. })),
        "{kept:?}"
    );
    assert_eq!(r.disk("b.txt").unwrap(), b"b\n");
    assert_eq!(markers(&repo_markers), Vec::<String>::new());
}

/// KF8, the command effect's part: in a `file://` partial clone that allows
/// the `file` protocol and lacks a `.gitignore` blob, the checkpoints around
/// a command, the comparison, the Undo and the inverse's Keep start no fetch
/// (`GIT_TRACE`). The positive control: plain git's untracked listing in
/// another clone fetches the blob.
#[test]
fn kf8_a_command_effect_starts_no_fetch() {
    let scratch = Scratch::new("fx-kf8-control");
    let control = crate::git::tests::unreadable_gitignore_clone(&scratch, "secret\n");
    let control_trace = scratch.path().join("control.trace");
    plain_git(
        &scratch,
        &control,
        &["ls-files", "-z", "--others", "--exclude-standard"],
        &[("GIT_TRACE", &control_trace)],
    );
    assert!(
        !fetch_lines(&control_trace).is_empty(),
        "the positive control: plain git fetches the missing blob"
    );

    let trace_at = Arc::new(std::sync::Mutex::new(None::<PathBuf>));
    let kept = Arc::clone(&trace_at);
    let r = Rig::with_runner(
        "fx-kf8",
        |scratch| crate::git::tests::unreadable_gitignore_clone(scratch, "secret\n"),
        move |scratch| {
            let mut runner = scratch.runner();
            let trace = scratch.path().join("runner.trace");
            runner.trace = Some(trace.clone());
            *kept.lock().unwrap() = Some(trace);
            runner
        },
    );
    let trace = trace_at.lock().unwrap().clone().unwrap();
    let outcome = r.command(
        "Set-Content -LiteralPath later.txt -Value later -NoNewline",
        "call_k8",
    );
    assert_eq!(
        effect_files(&outcome),
        [path("later.txt", PathChange::Added)]
    );
    let change = r.effect_changes("call_k8")["later.txt"].clone();
    r.review(vec![ReviewOp::Undo {
        change: change.id,
        hunks: None,
        note: None,
    }]);
    let kept = r.review(vec![ReviewOp::KeepAll]);
    assert!(matches!(kept[0].1, ReviewResult::Kept { .. }), "{kept:?}");
    assert_eq!(r.disk("later.txt"), None, "moved aside");
    assert!(
        trace.exists(),
        "the trace was written, so it would show a fetch"
    );
    assert_eq!(fetch_lines(&trace), Vec::<String>::new());
}

/// CF13: `[run_command, edit_file]` in one response, both orders. Whatever
/// the order, the command's effect never lists the reader's kept edit; a
/// Keep while the command runs is refused (the review asks the real command
/// slots); a second command in the folder is refused.
#[test]
fn cf13_a_command_and_an_edit_in_one_response() {
    // The command was asked for first; the edit staged before the reader
    // approved: Approve waits for the review, the edit is kept before the
    // command's first checkpoint, and the effect is the command's alone.
    let r = Rig::new("fx-cf13a", super::checkpoint_tests::repo);
    let prepared = r
        .prepare("Set-Content -LiteralPath c.txt -Value c -NoNewline")
        .unwrap();
    r.edit("a.txt", "old", "new");
    assert_eq!(
        r.approve(&prepared, "call_a").unwrap_err(),
        "Review 1 staged change first."
    );
    let id = r.change_at("a.txt").id;
    let kept = r.review(vec![ReviewOp::Keep {
        change: id,
        hunks: None,
    }]);
    assert!(matches!(kept[0].1, ReviewResult::Kept { .. }), "{kept:?}");
    let approval = r.approve(&prepared, "call_a").unwrap();
    let outcome = r.run_approved(&prepared, &approval, "call_a");
    assert_eq!(effect_files(&outcome), [path("c.txt", PathChange::Added)]);

    // The command runs first; the edit is staged while it runs: its Keep and
    // a second command are refused until the command ends, and the effect
    // does not list the edit.
    let r = Rig::new("fx-cf13b", super::checkpoint_tests::repo);
    let prepared = r
        .prepare(
            "Start-Sleep -Milliseconds 3000; Set-Content -LiteralPath c.txt -Value c -NoNewline",
        )
        .unwrap();
    let approval = r.approve(&prepared, "call_b").unwrap();
    std::thread::scope(|scope| {
        let running = scope.spawn(|| r.run_approved(&prepared, &approval, "call_b"));
        let deadline = Instant::now() + Duration::from_secs(20);
        while !r.slots.running(&r.workspace.id) {
            assert!(Instant::now() < deadline, "the command started");
            std::thread::sleep(Duration::from_millis(10));
        }
        r.edit("a.txt", "old", "new");
        let id = r.change_at("a.txt").id;
        let refused = r.review(vec![ReviewOp::Keep {
            change: id,
            hunks: None,
        }]);
        assert_eq!(
            refused[0].1,
            ReviewResult::Conflict {
                reason: "A command is still running in this folder; Keep when it has finished."
                    .into()
            }
        );
        assert_eq!(r.disk("a.txt").unwrap(), b"old\nkeep\n", "nothing written");
        assert_eq!(
            r.prepare("Write-Output second").unwrap_err(),
            "Another command is still running in this folder."
        );
        let outcome = running.join().unwrap();
        assert_eq!(effect_files(&outcome), [path("c.txt", PathChange::Added)]);
    });
    let id = r.change_at("a.txt").id;
    let kept = r.review(vec![ReviewOp::Keep {
        change: id,
        hunks: None,
    }]);
    assert!(matches!(kept[0].1, ReviewResult::Kept { .. }), "{kept:?}");
    assert_eq!(r.disk("a.txt").unwrap(), b"new\nkeep\n");
    assert!(r.effect_changes("call_b").keys().all(|p| p == "c.txt"));
}
