//! Checkpoints against folders and scratch repositories this file makes
//! (the chat core's spec §8.2, §8.3; §16.4 KF1–KF3, KF7–KF9). Every
//! folder is temporary; git runs only in scratch repositories, through the
//! runner, except the setup's own git and the positive controls, which run
//! plain git in a repository made for them. The setup's git reads no global
//! or system configuration.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use futures::executor::block_on;
use lattice_protocol::conversation::{
    CheckpointKind, CheckpointReason, Mode, Origin, ReviewOp, ReviewResult,
};

use super::FolderGuards;
use super::checkpoint::{
    CacheEntry, CheckpointError, Checkpoints, Copy, MAX_NEW_FILES, OmitWhy, Stamp, Taken, reusable,
};
use super::restore::{RestoreContext, Restored, restore};
use crate::clock::Clock;
use crate::convo::item::Item;
use crate::convo::sidecar::{NewMeta, SidecarStore};
use crate::exec::run::CommandSlots;
use crate::git::runner::{Extra, GitRunner};
use crate::git::tests::{Scratch, hostile, markers};
use crate::ports::Confirmer;
use crate::ports::fake::RecordingConfirm;
use crate::staging::Staging;
use crate::staging::review::{ReviewContext, review};
use crate::state::{Platform, StateRoot};
use crate::tools::edit::{EditFileArgs, StageContext, edit_file};
use crate::workspace::attach::attach_path;
use crate::workspace::lease::WriterLease;
use crate::workspace::{IgnoreCache, Workspace};

pub(super) const ID: &str = "c0ffee000002";

/// One conversation over one folder, with its checkpoints and staging.
pub(super) struct Fixture {
    pub(super) scratch: Scratch,
    pub(super) folder: PathBuf,
    pub(super) workspace: Workspace,
    pub(super) runner: GitRunner,
    pub(super) store: SidecarStore,
    pub(super) staging: Staging,
    pub(super) checkpoints: Checkpoints,
    /// The conversation's writer lease (row E5), free until a Keep takes it.
    pub(super) lease: WriterLease,
    clock: Clock,
}

impl Fixture {
    /// `make` builds the folder in the scratch area; `runner` (when given)
    /// replaces the scratch's own for everything after the setup.
    pub(super) fn new(tag: &str, make: impl FnOnce(&Scratch) -> PathBuf) -> Self {
        Self::with_runner(tag, make, |scratch| scratch.runner())
    }

    pub(super) fn with_runner(
        tag: &str,
        make: impl FnOnce(&Scratch) -> PathBuf,
        runner: impl FnOnce(&Scratch) -> GitRunner,
    ) -> Self {
        let scratch = Scratch::new(tag);
        let folder = make(&scratch);
        let runner = runner(&scratch);
        let workspace = attach_path(
            &folder,
            &scratch.env(),
            &StateRoot::at(scratch.path().join("state")),
            &runner,
        )
        .unwrap();
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
        let clock: Clock = Arc::new(|| 2000.0);
        let lease = WriterLease::for_workspace(
            &workspace,
            &runner,
            &StateRoot::at(scratch.path().join("state")),
            Platform::host(),
        );
        Self {
            lease,
            folder,
            workspace,
            runner,
            store,
            checkpoints: Checkpoints::new(Arc::clone(&sidecar), Arc::clone(&clock)),
            staging: Staging::new(sidecar),
            clock,
            scratch,
        }
    }

    pub(super) fn take(&self) -> Result<Taken, CheckpointError> {
        self.take_for(CheckpointReason::BeforeKeep, &[])
    }

    pub(super) fn take_for(
        &self,
        reason: CheckpointReason,
        paths: &[&str],
    ) -> Result<Taken, CheckpointError> {
        let paths: Vec<String> = paths.iter().map(|path| (*path).to_owned()).collect();
        self.checkpoints
            .take(&self.workspace, &self.runner, reason, &paths)
    }

    pub(super) fn restore(&self, to: u32) -> Restored {
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

    /// Stage an `edit_file`.
    pub(super) fn edit(&self, path: &str, old: &str, new: &str) {
        edit_file(
            &StageContext {
                workspace: &self.workspace,
                runner: &self.runner,
                staging: &self.staging,
                mode: Mode::Agent,
                trusted: true,
                turn: &"a1b2c3d4e5f6".to_owned(),
                call: &"call_1".to_owned(),
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

    /// Keep every live change (Keep All) through the review, with the real
    /// checkpoint.
    pub(super) fn keep_all(&self) -> Vec<(String, ReviewResult)> {
        let commands = CommandSlots::default();
        let guards = FolderGuards {
            checkpoints: &self.checkpoints,
            workspace: &self.workspace,
            runner: &self.runner,
            lease: &self.lease,
            commands: &commands,
        };
        let confirmer = Confirmer::new(Arc::new(RecordingConfirm::answering(true)));
        let ctx = ReviewContext {
            workspace: &self.workspace,
            runner: &self.runner,
            staging: &self.staging,
            mode: Mode::Agent,
            trusted: true,
            guards: &guards,
            confirmer: &confirmer,
            clock: &self.clock,
        };
        block_on(review(&ctx, vec![ReviewOp::KeepAll]))
            .results
            .into_iter()
            .map(|result| (result.path, result.result))
            .collect()
    }

    pub(super) fn items(&self) -> Vec<Item> {
        self.store.read_items(ID).unwrap().items
    }

    pub(super) fn disk(&self, path: &str) -> Option<Vec<u8>> {
        std::fs::read(self.folder.join(path)).ok()
    }

    /// `git cat-file blob <rev>:<path>` through the runner.
    pub(super) fn blob_at(&self, rev: &str, path: &str) -> Vec<u8> {
        let crate::git::dotgit::Repo::Git(local) = &self.workspace.repo else {
            panic!("not a git workspace");
        };
        let spec = format!("{rev}:{path}");
        let out = self
            .runner
            .run(
                local,
                &[
                    OsStr::new("cat-file"),
                    OsStr::new("blob"),
                    OsStr::new(&spec),
                ],
                &Extra::default(),
            )
            .unwrap();
        assert_eq!(out.status, 0, "{}", out.stderr);
        out.stdout
    }

    /// The paths of `commit`'s tree.
    pub(super) fn tree_paths(&self, commit: &str) -> Vec<String> {
        let crate::git::dotgit::Repo::Git(local) = &self.workspace.repo else {
            panic!("not a git workspace");
        };
        let out = self
            .runner
            .run(
                local,
                &[
                    OsStr::new("ls-tree"),
                    OsStr::new("-r"),
                    OsStr::new("--name-only"),
                    OsStr::new("-z"),
                    OsStr::new("--full-tree"),
                    OsStr::new(commit),
                ],
                &Extra::default(),
            )
            .unwrap();
        out.stdout
            .split(|byte| *byte == 0)
            .filter(|raw| !raw.is_empty())
            .map(|raw| String::from_utf8(raw.to_vec()).unwrap())
            .collect()
    }
}

/// A repository with `a.txt` and `gone.txt` committed.
pub(super) fn repo(scratch: &Scratch) -> PathBuf {
    let repo = scratch.path().join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    scratch.git(&repo, &["init", "-q"]);
    std::fs::write(repo.join("a.txt"), "old\nkeep\n").unwrap();
    std::fs::write(repo.join("gone.txt"), "here\n").unwrap();
    scratch.git(&repo, &["add", "."]);
    scratch.git(&repo, &["commit", "-q", "-m", "one"]);
    repo
}

/// A folder in no repository, with `a.txt`.
pub(super) fn plain(scratch: &Scratch) -> PathBuf {
    let folder = scratch.path().join("plain");
    std::fs::create_dir_all(folder.join("sub")).unwrap();
    std::fs::write(folder.join("a.txt"), "old\nkeep\n").unwrap();
    folder
}

/// The plain git of a positive control: no global or system configuration,
/// none of the runner's overrides.
pub(super) fn plain_git(
    scratch: &Scratch,
    cwd: &Path,
    args: &[&str],
    envs: &[(&str, &Path)],
) -> bool {
    let mut command = Command::new("git");
    command
        .args(args)
        .current_dir(cwd)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", scratch.home().join(".gitconfig"))
        .env("HOME", scratch.home())
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .env_remove("GIT_DIR")
        .env_remove("GIT_INDEX_FILE");
    for (name, value) in envs {
        command.env(name, value);
    }
    command.output().unwrap().status.success()
}

/// KF1: a hostile repository (hooks, an fsmonitor command, clean, smudge and
/// process filters for every file) runs none of its programs during
/// checkpoints, a restore and the restore's Keep. The positive control runs
/// plain git's `update-index --add` (which reads the working file, as
/// `capture.py`'s method does) there, and must leave markers. (The command
/// effect's part is row E8's.)
/// Mutant: `update-index --add` of the working files instead of
/// `--cacheinfo`.
#[test]
fn kf1_a_hostile_repository_runs_none_of_its_programs() {
    let scratch = Scratch::new("ck-kf1-control");
    let (control, control_markers) = hostile(&scratch, "control");
    let index = scratch.path().join("control-index");
    plain_git(
        &scratch,
        &control,
        &["update-index", "--add", "b.txt"],
        &[("GIT_INDEX_FILE", &index)],
    );
    let seen = markers(&control_markers);
    println!("plain git ran: {seen:?}");
    assert!(
        seen.iter().any(|m| m == "clean" || m == "process"),
        "the positive control: plain git runs the repository's filter: {seen:?}"
    );

    let f = Fixture::new("ck-kf1", |scratch| hostile(scratch, "hostile").0);
    let repo_markers = f.scratch.path().join("hostile-markers");
    let first = f.take();
    assert_eq!(
        markers(&repo_markers),
        Vec::<String>::new(),
        "the first checkpoint ran none of the repository's programs"
    );
    let first = first.unwrap();
    assert_eq!(first.kind, CheckpointKind::Git);
    let paths = f.tree_paths(first.commit.as_deref().unwrap());
    for path in [
        ".gitattributes",
        "a.txt",
        "b.txt",
        "own-hooks/post-checkout",
    ] {
        assert!(paths.contains(&path.to_owned()), "{path} in {paths:?}");
    }
    std::fs::write(f.folder.join("b.txt"), "b changed\n").unwrap();
    std::fs::write(f.folder.join("c.txt"), "c\n").unwrap();
    let second = f.take().unwrap();
    assert_ne!(second.commit, first.commit);
    assert_eq!(
        f.blob_at(second.commit.as_deref().unwrap(), "b.txt"),
        b"b changed\n"
    );
    let restored = f.restore(first.id);
    assert_eq!(restored.staged.len(), 2, "{restored:?}");
    let kept = f.keep_all();
    assert!(
        kept.iter()
            .all(|(_, result)| matches!(result, ReviewResult::Kept { .. })),
        "{kept:?}"
    );
    assert_eq!(f.disk("b.txt").unwrap(), b"b\n");
    assert!(f.disk("c.txt").is_none());
    assert_eq!(
        markers(&repo_markers),
        Vec::<String>::new(),
        "through the runner, no program of the repository ran"
    );
}

/// Every file under `dir` (recursively) and its bytes.
fn snapshot_of(dir: &Path) -> Vec<(String, Vec<u8>)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, Vec<u8>)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out);
            } else {
                let name = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                out.push((name, std::fs::read(&path).unwrap_or_default()));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

/// KF2: a checkpoint and a restore leave the reader's index bytes, `HEAD`,
/// branches, working files and `git status` output exactly as they were.
/// Mutant: the default `GIT_INDEX_FILE` (the reader's own index).
#[test]
fn kf2_the_readers_index_head_branches_and_files_are_untouched() {
    let f = Fixture::new("ck-kf2", repo);
    std::fs::write(f.folder.join("a.txt"), "changed\nkeep\n").unwrap();
    std::fs::write(f.folder.join("new.txt"), "untracked\n").unwrap();
    f.scratch.git(&f.folder, &["branch", "side"]);
    let git = f.folder.join(".git");
    let status = || {
        String::from_utf8(
            f.scratch
                .git(&f.folder, &["status", "--porcelain=v2", "--branch"])
                .stdout,
        )
        .unwrap()
    };
    // `git status` may refresh the index, so it runs first and last.
    let before = status();
    let index = std::fs::read(git.join("index")).unwrap();
    let head = std::fs::read(git.join("HEAD")).unwrap();
    let refs = snapshot_of(&git.join("refs").join("heads"));
    let working: Vec<_> = snapshot_of(&f.folder)
        .into_iter()
        .filter(|(name, _)| !name.starts_with(".git"))
        .collect();

    let first = f.take().unwrap();
    std::fs::write(f.folder.join("later.txt"), "later\n").unwrap();
    f.take().unwrap();
    std::fs::remove_file(f.folder.join("later.txt")).unwrap();
    let restored = f.restore(first.id);
    assert!(restored.staged.is_empty(), "{restored:?}");

    assert_eq!(
        std::fs::read(git.join("index")).unwrap(),
        index,
        "the index"
    );
    assert_eq!(std::fs::read(git.join("HEAD")).unwrap(), head, "HEAD");
    assert_eq!(
        snapshot_of(&git.join("refs").join("heads")),
        refs,
        "branches"
    );
    let working_after: Vec<_> = snapshot_of(&f.folder)
        .into_iter()
        .filter(|(name, _)| !name.starts_with(".git"))
        .collect();
    assert_eq!(working_after, working, "working files");
    assert_eq!(status(), before, "git status");
    assert!(
        git.join("lattice-chat").join("index").join(ID).is_file(),
        "the private index is where the checkpoint wrote"
    );
}

/// KF3: the ref is a compare-and-swap. A ref that exists already (here made
/// beforehand) is refused, kept as it was, and no checkpoint is recorded.
/// Mutant: `update-ref` without the old value.
#[test]
fn kf3_the_compare_and_swap_refuses_an_existing_ref() {
    let f = Fixture::new("ck-kf3", repo);
    let name = format!("refs/lattice/chat/{ID}/1");
    f.scratch.git(&f.folder, &["update-ref", &name, "HEAD"]);
    let head = String::from_utf8(f.scratch.git(&f.folder, &["rev-parse", "HEAD"]).stdout).unwrap();
    let refused = f.take();
    assert_eq!(refused, Err(CheckpointError::RefExists(name.clone())));
    assert_eq!(
        refused.unwrap_err().sentence(),
        format!("A checkpoint named {name} exists already in this repository.")
    );
    let now = String::from_utf8(f.scratch.git(&f.folder, &["rev-parse", &name]).stdout).unwrap();
    assert_eq!(now, head, "the existing ref is unchanged");
    assert!(f.checkpoints.all().is_empty());
    assert!(
        !f.items()
            .iter()
            .any(|item| matches!(item, Item::Checkpoint { .. }))
    );
}

/// §8.2 steps 5 and 6: checkpoints are numbered, each new tree is a commit
/// on the previous one under its own ref, and an unchanged tree reuses the
/// previous commit with no new ref. A reopened record carries on.
#[test]
fn checkpoints_chain_and_an_unchanged_tree_makes_no_ref() {
    let f = Fixture::new("ck-chain", repo);
    let one = f.take().unwrap();
    let two = f.take().unwrap();
    assert_eq!((one.id, two.id), (1, 2));
    assert_eq!(two.commit, one.commit, "unchanged: the previous commit");
    let refs = String::from_utf8(
        f.scratch
            .git(
                &f.folder,
                &["for-each-ref", "--format=%(refname)", "refs/lattice/"],
            )
            .stdout,
    )
    .unwrap();
    assert_eq!(refs, format!("refs/lattice/chat/{ID}/1\n"));
    std::fs::write(f.folder.join("a.txt"), "three\n").unwrap();
    let three = f.take().unwrap();
    let parent = String::from_utf8(
        f.scratch
            .git(
                &f.folder,
                &["rev-parse", &format!("refs/lattice/chat/{ID}/3^")],
            )
            .stdout,
    )
    .unwrap();
    assert_eq!(parent.trim(), one.commit.as_deref().unwrap());
    assert_eq!(
        f.blob_at(three.commit.as_deref().unwrap(), "a.txt"),
        b"three\n"
    );
    let message = String::from_utf8(
        f.scratch
            .git(
                &f.folder,
                &[
                    "log",
                    "-1",
                    "--format=%an <%ae>|%s",
                    three.commit.as_deref().unwrap(),
                ],
            )
            .stdout,
    )
    .unwrap();
    assert_eq!(
        message.trim(),
        "Lattice <lattice@localhost>|lattice checkpoint 3"
    );
    let reopened = Checkpoints::from_items(
        Arc::clone(f.staging.sidecar()),
        &f.items(),
        Arc::new(|| 0.0),
    );
    assert_eq!(reopened.all(), f.checkpoints.all());
    std::fs::write(f.folder.join("a.txt"), "four\n").unwrap();
    let four = reopened
        .take(&f.workspace, &f.runner, CheckpointReason::BeforeKeep, &[])
        .unwrap();
    assert_eq!(four.id, 4);
}

/// A folder below the top level: the checkpoint's tree is the folder's own
/// files, named relative to it.
#[test]
fn a_folder_below_the_top_level_checkpoints_its_own_files() {
    let f = Fixture::new("ck-below", |scratch| {
        let top = repo(scratch);
        std::fs::create_dir_all(top.join("w").join("x")).unwrap();
        std::fs::write(top.join("w").join("x").join("in.txt"), "in\n").unwrap();
        scratch.git(&top, &["add", "."]);
        scratch.git(&top, &["commit", "-q", "-m", "two"]);
        std::fs::write(top.join("w").join("new.txt"), "new\n").unwrap();
        top.join("w")
    });
    let one = f.take().unwrap();
    assert_eq!(
        f.tree_paths(one.commit.as_deref().unwrap()),
        ["new.txt", "x/in.txt"]
    );
    std::fs::remove_file(f.folder.join("new.txt")).unwrap();
    let restored = f.restore(one.id);
    assert_eq!(
        restored
            .staged
            .iter()
            .map(|change| change.path.as_str())
            .collect::<Vec<_>>(),
        ["new.txt"]
    );
}

/// KF7 (the checkpoint's part): a file over 16 MiB is omitted and named;
/// untracked files past the first 2,000, in path order, are recorded as not
/// taken, each by name; the count is `omitted`. A tracked file over the cap
/// is omitted too, and kept out of the tree.
/// Mutant: no caps.
#[test]
fn kf7_large_and_uncapped_files_are_omitted_by_name() {
    let f = Fixture::new("ck-kf7", repo);
    let big = vec![b'b'; 16 * 1024 * 1024 + 1];
    std::fs::write(f.folder.join("big.bin"), &big).unwrap();
    std::fs::create_dir_all(f.folder.join("many")).unwrap();
    for n in 0..MAX_NEW_FILES + 1 {
        std::fs::write(f.folder.join("many").join(format!("{n:04}.txt")), "x").unwrap();
    }
    let one = f.take().unwrap();
    let omitted: Vec<(String, OmitWhy)> = f
        .checkpoints
        .omitted(one.id)
        .unwrap()
        .into_iter()
        .map(|omitted| (omitted.path, omitted.why))
        .collect();
    // Untracked in path order: big.bin, many/0000 ... many/2000. The first
    // 2,000 are candidates (big.bin among them, then omitted for size).
    assert_eq!(
        omitted,
        [
            ("big.bin".to_owned(), OmitWhy::TooLarge),
            ("many/1999.txt".to_owned(), OmitWhy::UntrackedCap),
            ("many/2000.txt".to_owned(), OmitWhy::UntrackedCap),
        ]
    );
    assert_eq!(one.omitted, 3);
    let paths = f.tree_paths(one.commit.as_deref().unwrap());
    assert!(!paths.contains(&"big.bin".to_owned()));
    assert!(paths.contains(&"many/1998.txt".to_owned()));
    assert!(!paths.contains(&"many/1999.txt".to_owned()));
    assert_eq!(
        paths.len(),
        2 + MAX_NEW_FILES - 1,
        "a.txt, gone.txt and 1,999"
    );
    assert!(
        f.items().iter().any(|item| matches!(
            item,
            Item::Checkpoint {
                omitted: 3,
                omitted_list: Some(_),
                ..
            }
        )),
        "the item names the list"
    );
}

/// The lines of a `GIT_TRACE` file that start a fetch.
pub(super) fn fetch_lines(trace: &Path) -> Vec<String> {
    std::fs::read_to_string(trace)
        .unwrap_or_default()
        .lines()
        .filter(|line| line.contains(" fetch "))
        .map(str::to_owned)
        .collect()
}

/// KF8: in a `file://` partial clone that allows the `file` protocol itself
/// and whose `sub/.gitignore` blob is missing, a checkpoint, a restore and
/// the restore's Keep start no fetch (`GIT_TRACE`). The positive control
/// runs plain git's untracked listing in another clone of the same source,
/// which reads that `.gitignore` and must fetch it.
/// Mutant: `GIT_NO_LAZY_FETCH` and `GIT_ALLOW_PROTOCOL` dropped.
#[test]
fn kf8_a_partial_clone_starts_no_fetch() {
    let scratch = Scratch::new("ck-kf8-control");
    let control = crate::git::tests::unreadable_gitignore_clone(&scratch, "secret\n");
    let control_trace = scratch.path().join("control.trace");
    plain_git(
        &scratch,
        &control,
        &["ls-files", "-z", "--others", "--exclude-standard"],
        &[("GIT_TRACE", &control_trace)],
    );
    let fetched = fetch_lines(&control_trace);
    println!("plain git: {} fetch line(s)", fetched.len());
    assert!(
        !fetched.is_empty(),
        "the positive control: plain git fetches the missing blob"
    );

    let trace_dir = Arc::new(std::sync::Mutex::new(None::<PathBuf>));
    let kept = Arc::clone(&trace_dir);
    let mut f = Fixture::with_runner(
        "ck-kf8",
        |scratch| crate::git::tests::unreadable_gitignore_clone(scratch, "secret\n"),
        move |scratch| {
            let mut runner = scratch.runner();
            let trace = scratch.path().join("runner.trace");
            runner.trace = Some(trace.clone());
            *kept.lock().unwrap() = Some(trace);
            runner
        },
    );
    let trace = trace_dir.lock().unwrap().clone().unwrap();
    let one = f.take().unwrap();
    std::fs::write(f.folder.join("later.txt"), "later\n").unwrap();
    // A fresh ignore cache, so the restore asks git about the rules again.
    f.workspace.ignore_cache = IgnoreCache::default();
    let restored = f.restore(one.id);
    assert_eq!(
        restored
            .staged
            .iter()
            .map(|change| change.path.as_str())
            .collect::<Vec<_>>(),
        ["later.txt"]
    );
    let kept = f.keep_all();
    assert!(matches!(kept[0].1, ReviewResult::Kept { .. }), "{kept:?}");
    assert!(
        trace.exists(),
        "the trace was written, so it would show a fetch"
    );
    assert_eq!(fetch_lines(&trace), Vec::<String>::new());
}

/// KF9: a same-size rewrite whose last-write time is set back to what it
/// was: the next checkpoint captures the new bytes, because the change time
/// moved.
/// Mutant: a cache keyed on size and last-write time only.
#[test]
fn kf9_a_rewrite_with_its_time_restored_is_captured() {
    let f = Fixture::new("ck-kf9", repo);
    let path = f.folder.join("a.txt");
    let one = f.take().unwrap();
    let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
    std::fs::write(&path, "OLD\nKEEP\n").unwrap();
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(modified)
        .unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().modified().unwrap(),
        modified
    );
    let two = f.take().unwrap();
    assert_ne!(two.commit, one.commit, "the tree changed");
    assert_eq!(
        f.blob_at(two.commit.as_deref().unwrap(), "a.txt"),
        b"OLD\nKEEP\n"
    );
}

/// The racily-clean rule, by itself: an entry is reused only when every
/// fact matches and both times are older than the recording.
#[test]
fn the_cache_reuses_only_entries_older_than_their_recording() {
    let stamp = Stamp {
        identity: lattice_sys::fs::FileIdentity {
            volume_serial: 7,
            file_id: [1; 16],
        },
        size: 10,
        last_write: 100,
        change: 100,
    };
    let entry = CacheEntry {
        stamp,
        object: "ab".repeat(20),
        recorded: 200,
        written: true,
    };
    assert!(reusable(&entry, &stamp));
    for other in [
        Stamp { size: 11, ..stamp },
        Stamp {
            last_write: 101,
            ..stamp
        },
        Stamp {
            change: 101,
            ..stamp
        },
        Stamp {
            identity: lattice_sys::fs::FileIdentity {
                volume_serial: 7,
                file_id: [2; 16],
            },
            ..stamp
        },
    ] {
        assert!(!reusable(&entry, &other), "{other:?}");
    }
    let racy = CacheEntry {
        recorded: 100,
        ..entry.clone()
    };
    assert!(!reusable(&racy, &stamp), "written in the recording's tick");
}

/// §8.3: without git, a checkpoint copies the paths about to be written:
/// present (with their bytes), absent, or skipped when over the cap; a
/// command's checkpoint is exposed.
#[test]
fn without_git_the_checkpoint_copies_what_will_be_written() {
    let f = Fixture::new("ck-copies", plain);
    std::fs::write(f.folder.join("big.bin"), vec![b'x'; 16 * 1024 * 1024 + 1]).unwrap();
    let one = f
        .take_for(
            CheckpointReason::BeforeKeep,
            &["a.txt", "sub/new.txt", "big.bin"],
        )
        .unwrap();
    assert_eq!(one.kind, CheckpointKind::Copies);
    assert!(!one.exposed);
    assert_eq!(one.bytes, 9);
    assert_eq!(one.omitted, 1);
    let manifest: Vec<(String, Copy)> = f
        .checkpoints
        .manifest(one.id)
        .unwrap()
        .into_iter()
        .map(|entry| (entry.path, entry.copy))
        .collect();
    assert_eq!(
        manifest,
        [
            (
                "a.txt".to_owned(),
                Copy::Present {
                    sha256: crate::sha::sha256_hex(b"old\nkeep\n"),
                    bytes: 9
                }
            ),
            (
                "big.bin".to_owned(),
                Copy::Skipped {
                    why: OmitWhy::TooLarge
                }
            ),
            ("sub/new.txt".to_owned(), Copy::Absent),
        ]
    );
    assert_eq!(
        f.checkpoints.copy_bytes(one.id, "a.txt").unwrap(),
        b"old\nkeep\n"
    );
    let command = f
        .take_for(
            CheckpointReason::BeforeCommand {
                call_id: "call_9".into(),
            },
            &[],
        )
        .unwrap();
    assert!(command.exposed);
    assert_eq!(
        f.checkpoints
            .views()
            .iter()
            .map(|view| (view.id, view.exposed, view.bytes))
            .collect::<Vec<_>>(),
        [(1, false, 9), (2, true, 0)]
    );
}

/// SF11's checkpoint part: with no git on `PATH`, a git folder's checkpoint
/// fails with a sentence and records nothing.
#[test]
fn without_git_on_path_a_git_folder_takes_no_checkpoint() {
    let f = Fixture::new("ck-nogit", repo);
    let empty = f.scratch.path().join("empty-path");
    std::fs::create_dir_all(&empty).unwrap();
    let mut env = f.scratch.env();
    env.set("PATH", empty.as_os_str());
    let runner = GitRunner::new(
        Arc::new(env),
        &StateRoot::at(f.scratch.path().join("state")),
    );
    let failed = f
        .checkpoints
        .take(&f.workspace, &runner, CheckpointReason::BeforeKeep, &[])
        .unwrap_err();
    assert_eq!(
        failed.sentence(),
        "git was not found outside this folder, so Lattice cannot ask it."
    );
    assert!(f.checkpoints.all().is_empty());
}
