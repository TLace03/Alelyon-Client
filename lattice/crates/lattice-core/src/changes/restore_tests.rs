//! Restore against folders and scratch repositories this file makes
//! (the chat core's spec §8.5, §8.3; §16.4 KF5–KF7). Every folder is
//! temporary. A restore stages and writes nothing; its Keep (the review's)
//! moves files aside.

use lattice_protocol::conversation::{ChangeKind, ChangeState, CheckpointReason, ReviewResult};

use super::checkpoint_tests::{Fixture, plain, repo};
use super::restore::INCOMPLETE;
use crate::convo::item::NewState;
use crate::fsx::{MoveReason, RemovedArea};

/// Every file under `dir` with its bytes and last-write time.
fn state_of(dir: &std::path::Path) -> Vec<(String, Vec<u8>, std::time::SystemTime)> {
    fn walk(
        root: &std::path::Path,
        dir: &std::path::Path,
        out: &mut Vec<(String, Vec<u8>, std::time::SystemTime)>,
    ) {
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
                let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
                out.push((name, std::fs::read(&path).unwrap(), modified));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn kept(results: &[(String, ReviewResult)]) -> bool {
    results
        .iter()
        .all(|(_, result)| matches!(result, ReviewResult::Kept { .. }))
}

/// KF5: without git, a command's checkpoint is exposed, and a restore across
/// it says it is incomplete; one before any command does not.
/// Mutant: claiming a complete restore.
#[test]
fn kf5_without_git_a_restore_across_a_command_says_it_is_incomplete() {
    let f = Fixture::new("rs-kf5", plain);
    f.edit("a.txt", "old", "new");
    assert!(kept(&f.keep_all()));
    assert_eq!(f.disk("a.txt").unwrap(), b"new\nkeep\n");
    let clean = f.restore(1);
    assert_eq!(clean.incomplete, None);
    assert_eq!(clean.staged.len(), 1);
    // Undo it, so the next restore can stage the path again.
    let mut undone = f.staging.snapshot(&clean.staged[0].id).unwrap();
    undone.change.state = ChangeState::Undone;
    f.staging.update(undone).unwrap();

    let command = f
        .take_for(
            CheckpointReason::BeforeCommand {
                call_id: "call_7".into(),
            },
            &[],
        )
        .unwrap();
    assert!(command.exposed, "the command's checkpoint is exposed");
    let across = f.restore(1);
    assert_eq!(across.incomplete.as_deref(), Some(INCOMPLETE));
    assert_eq!(across.staged.len(), 1, "what Lattice saw is still restored");
}

/// KF6: a restore stages and removes nothing: the folder's files, bytes and
/// times are unchanged, and the record only grows (no message or item is
/// removed). Its Keep moves a created file aside.
/// Mutant: the restore unlinking a file made after the checkpoint.
#[test]
fn kf6_a_restore_removes_nothing_and_keeps_every_message() {
    let f = Fixture::new("rs-kf6", repo);
    f.take().unwrap();
    std::fs::write(f.folder.join("made.txt"), "made\n").unwrap();
    std::fs::write(f.folder.join("a.txt"), "changed\n").unwrap();
    std::fs::remove_file(f.folder.join("gone.txt")).unwrap();
    let items_before = f.items();
    let before: Vec<_> = state_of(&f.folder)
        .into_iter()
        .filter(|(name, ..)| !name.starts_with(".git"))
        .collect();
    let restored = f.restore(1);
    let staged: Vec<(&str, ChangeKind, bool)> = restored
        .staged
        .iter()
        .map(|change| {
            (
                change.path.as_str(),
                change.kind,
                change.new == NewState::Deleted,
            )
        })
        .collect();
    assert_eq!(
        staged,
        [
            ("a.txt", ChangeKind::Restore, false),
            ("gone.txt", ChangeKind::Restore, false),
            ("made.txt", ChangeKind::Restore, true),
        ]
    );
    let after: Vec<_> = state_of(&f.folder)
        .into_iter()
        .filter(|(name, ..)| !name.starts_with(".git"))
        .collect();
    assert_eq!(after, before, "the restore wrote and removed nothing");
    let items_after = f.items();
    assert_eq!(
        &items_after[..items_before.len()],
        &items_before[..],
        "the record only grows"
    );
    assert!(kept(&f.keep_all()));
    assert_eq!(f.disk("a.txt").unwrap(), b"old\nkeep\n");
    assert_eq!(f.disk("gone.txt").unwrap(), b"here\n");
    assert!(f.disk("made.txt").is_none());
    let area = RemovedArea::new(f.staging.sidecar().dir().join("removed"));
    let moved = area
        .manifest()
        .unwrap()
        .into_iter()
        .find(|line| line.path == "made.txt")
        .expect("made.txt was moved aside");
    assert_eq!(moved.why, MoveReason::KeptRestore);
    assert_eq!(
        std::fs::read(area.root().join(&moved.to)).unwrap(),
        b"made\n"
    );
}

/// KF7 (the restore's part): every path the checkpoint omitted is left
/// alone, never staged as absent or restored: a 17 MiB file, and untracked
/// files past the cap, each there at the checkpoint and changed since.
/// Mutant: omitted paths restored as absent.
#[test]
fn kf7_a_restore_leaves_every_omitted_path_alone() {
    let f = Fixture::new("rs-kf7", repo);
    std::fs::write(f.folder.join("big.bin"), vec![b'b'; 17 * 1024 * 1024]).unwrap();
    std::fs::create_dir_all(f.folder.join("many")).unwrap();
    for n in 0..2000 {
        std::fs::write(f.folder.join("many").join(format!("{n:04}.txt")), "x").unwrap();
    }
    std::fs::write(f.folder.join("zz.txt"), "past the cap\n").unwrap();
    let one = f.take().unwrap();
    assert_eq!(
        one.omitted, 3,
        "big.bin for size; many/1999 and zz.txt past the cap"
    );
    std::fs::write(f.folder.join("big.bin"), b"small now\n").unwrap();
    std::fs::write(f.folder.join("zz.txt"), b"changed\n").unwrap();
    std::fs::write(f.folder.join("a.txt"), b"changed\n").unwrap();
    let restored = f.restore(one.id);
    let mut omitted = restored.omitted.clone();
    omitted.sort();
    assert_eq!(omitted, ["big.bin", "many/1999.txt", "zz.txt"]);
    assert_eq!(
        restored
            .staged
            .iter()
            .map(|change| change.path.as_str())
            .collect::<Vec<_>>(),
        ["a.txt"],
        "only a.txt, which the checkpoint holds"
    );
}

/// A path with a staged change still waiting for review is left alone (ST2:
/// one live change per path), and listed with why.
#[test]
fn a_waiting_change_is_left_alone() {
    let f = Fixture::new("rs-waiting", repo);
    f.take().unwrap();
    std::fs::write(f.folder.join("a.txt"), "old\nother\n").unwrap();
    f.edit("a.txt", "other", "mine");
    let restored = f.restore(1);
    assert!(restored.staged.is_empty(), "{restored:?}");
    assert_eq!(restored.left_alone.len(), 1);
    assert_eq!(restored.left_alone[0].path, "a.txt");
    assert_eq!(f.staging.waiting(), 1, "the agent's change is untouched");
}

/// Without git: the copies restore a file's bytes and remove (by moving
/// aside) a file the agent created; a file first written after the restored
/// checkpoint is restored to its copy at that later checkpoint, and a file
/// Lattice never wrote is not touched.
#[test]
fn without_git_the_copies_are_restored() {
    let f = Fixture::new("rs-copies", plain);
    std::fs::write(f.folder.join("b.txt"), "bee\n").unwrap();
    f.edit("a.txt", "old", "new");
    assert!(kept(&f.keep_all()));
    assert!(matches!(f.workspace.repo, crate::git::dotgit::Repo::None));
    std::fs::write(f.folder.join("sub").join("made.txt"), "x").unwrap();
    // The agent creates a file and changes b.txt, after checkpoint 1.
    let stage = |path: &str, bytes: &[u8]| {
        let base = match f.disk(path) {
            Some(old) => crate::staging::Base::of_bytes(old),
            None => crate::staging::Base::absent(),
        };
        f.staging
            .record(
                crate::staging::Proposal {
                    path: path.into(),
                    authority: false,
                    base,
                    new: Some(bytes.to_vec()),
                    action: crate::staging::Action::Write,
                },
                &"a1b2c3d4e5f6".to_owned(),
                &"call_2".to_owned(),
            )
            .unwrap();
    };
    stage("new.txt", b"created\n");
    stage("b.txt", b"BEE\n");
    assert!(kept(&f.keep_all()));
    assert_eq!(f.checkpoints.all().len(), 2);
    let restored = f.restore(1);
    assert_eq!(restored.incomplete, None);
    let staged: Vec<(&str, bool)> = restored
        .staged
        .iter()
        .map(|change| (change.path.as_str(), change.new == NewState::Deleted))
        .collect();
    assert_eq!(
        staged,
        [("a.txt", false), ("b.txt", false), ("new.txt", true)],
        "sub/made.txt was never written by Lattice, so it is not touched"
    );
    assert!(kept(&f.keep_all()));
    assert_eq!(f.disk("a.txt").unwrap(), b"old\nkeep\n");
    assert_eq!(f.disk("b.txt").unwrap(), b"bee\n");
    assert!(f.disk("new.txt").is_none());
    assert_eq!(f.disk("sub/made.txt").unwrap(), b"x");
}

/// ST2 for a restore's (or an effect Undo's) staging: `stage_back` holds the
/// staging lock from its "no change waiting here" check through its update,
/// as the staging tools do, so an agent's staging of the same path cannot
/// fall between them and leave two live changes on one path. Here the test
/// holds the lock (as a staging tool would): `stage_back` waits; the agent's
/// change is recorded meanwhile; once the lock is free, `stage_back` sees
/// it and leaves the path alone.
/// Falsifier: fails on the code before this fix (stage_back runs at once, without the
/// lock, and stages a second live change at a.txt).
#[test]
fn stage_back_holds_the_staging_lock_from_its_check_through_its_update() {
    use crate::staging::{Action, Base, Proposal};
    use lattice_protocol::conversation::ChangeOrigin;
    let f = Fixture::new("rs-stage-back-lock", plain);
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let held = f.staging.serial();
    std::thread::scope(|s| {
        s.spawn(|| {
            let back = super::restore::stage_back(
                &f.workspace,
                &f.runner,
                &f.staging,
                "a.txt",
                Some(b"restored\n".to_vec()),
                ChangeKind::Restore,
                ChangeOrigin::Restore { to: 1 },
            );
            let _ = done_tx.send(back);
        });
        let early = done_rx.recv_timeout(std::time::Duration::from_secs(2));
        // The agent's staging, as a staging tool does it under the lock.
        f.staging
            .record(
                Proposal {
                    path: "a.txt".into(),
                    authority: false,
                    base: Base::of_bytes(b"old\nkeep\n".to_vec()),
                    new: Some(b"agent\n".to_vec()),
                    action: Action::Write,
                },
                &"a1b2c3d4e5f6".to_owned(),
                &"call_1".to_owned(),
            )
            .unwrap();
        drop(held);
        assert!(
            early.is_err(),
            "stage_back finished while the staging lock was held: {early:?}"
        );
        let back = done_rx.recv().unwrap().unwrap();
        assert!(
            matches!(back, super::restore::Back::LeftAlone(_)),
            "{back:?}"
        );
    });
    let live: Vec<_> = f
        .staging
        .changes()
        .into_iter()
        .filter(|change| change.path == "a.txt" && crate::staging::is_live(&change.state))
        .collect();
    assert_eq!(live.len(), 1, "one live change per path: {live:?}");
    assert_eq!(
        f.staging.snapshot(&live[0].id).unwrap().new.unwrap(),
        b"agent\n"
    );
}
