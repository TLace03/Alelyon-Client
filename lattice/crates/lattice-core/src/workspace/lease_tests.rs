//! The writer lease against folders and scratch repositories this file makes
//! (the chat core's spec §6.5; §16.4 SF10, WF4's lease part). Every
//! folder is temporary, the state home included (the scratch area's own
//! home). Interop with Python is `lease_interop_tests.rs`; the derivation's
//! arithmetic is held to `lease/derive.json` by `tests/lease_parity.rs`.

use std::sync::Arc;

use futures::executor::block_on;
use lattice_protocol::conversation::{Mode, Origin, ReviewOp, ReviewResult};

use super::attach::attach_path;
use super::lease::{
    CHECKOUT_LEASE_FILE, LeaseError, WriterLease, derive_checkout_lease, lease_file_for,
    selected_repository_state_root, try_acquire,
};
use crate::changes::FolderGuards;
use crate::changes::checkpoint::Checkpoints;
use crate::clock::Clock;
use crate::convo::sidecar::{NewMeta, SidecarStore};
use crate::env::MapEnv;
use crate::exec::run::CommandSlots;
use crate::git::tests::Scratch;
use crate::policy::Lease;
use crate::ports::Confirmer;
use crate::ports::fake::RecordingConfirm;
use crate::staging::Staging;
use crate::staging::review::{ReviewContext, review};
use crate::state::{Platform, StateRoot};
use crate::tools::edit::{EditFileArgs, StageContext, edit_file};
use crate::workspace::Workspace;

/// `selected_repository_state_root`: `ALELYON_HOME` (any non-empty text)
/// means the state root's own home; otherwise the per-user home: the
/// platform folder's `globals/` when installed or forced packaged,
/// `~/.alelyon/globals` in a source checkout.
#[test]
fn the_state_home_is_pythons_selected_repository_state_root() {
    let checkout = StateRoot::at(r"C:\checkout");
    let installed = StateRoot {
        installed: true,
        ..StateRoot::at(r"C:\installed")
    };
    let env = MapEnv::new()
        .with("USERPROFILE", r"D:\Profiles\someone")
        .with("LOCALAPPDATA", r"D:\Profiles\someone\AppData\Local");
    let win = Platform::Windows;
    assert_eq!(
        selected_repository_state_root(&env, &checkout, win),
        std::path::PathBuf::from(r"D:\Profiles\someone\.alelyon\globals")
    );
    assert_eq!(
        selected_repository_state_root(&env, &installed, win),
        std::path::PathBuf::from(r"D:\Profiles\someone\AppData\Local\Alelyon\globals")
    );
    let forced = env.clone().with("ALELYON_FORCE_PACKAGED", " Yes ");
    assert_eq!(
        selected_repository_state_root(&forced, &checkout, win),
        std::path::PathBuf::from(r"D:\Profiles\someone\AppData\Local\Alelyon\globals")
    );
    let homed = env.clone().with("ALELYON_HOME", r"C:\elsewhere");
    let state = crate::state::resolve_with(&homed, None, win);
    assert_eq!(
        selected_repository_state_root(&homed, &state, win),
        std::path::PathBuf::from(r"C:\elsewhere\globals")
    );
    // Python's `os.environ.get("ALELYON_HOME")` is true for blank text too:
    // then it is `paths.GLOBALS_DIR`, the state root's own home.
    let blank = env.with("ALELYON_HOME", "  ");
    assert_eq!(
        selected_repository_state_root(&blank, &checkout, win),
        std::path::PathBuf::from(r"C:\checkout\globals")
    );
}

/// The lock: one holder at a time, even two handles of one process; a
/// release lets the next one in; the file is never deleted.
#[test]
fn one_holder_at_a_time_and_the_file_stays() {
    let scratch = Scratch::new("lease-lock");
    let path = scratch.path().join("a").join("b").join(CHECKOUT_LEASE_FILE);
    let first = try_acquire(&path).unwrap().expect("free: taken");
    assert!(path.is_file(), "made with its folders");
    assert!(try_acquire(&path).unwrap().is_none(), "held: refused");
    first.release();
    assert!(path.is_file(), "released, not deleted");
    let again = try_acquire(&path).unwrap().expect("free again");
    drop(again);
    assert!(path.is_file());
}

/// WF4's lease part, and Python's `UnsafeLeasePath`: a junction or a folder
/// at the lease path is refused, never followed or locked.
/// Mutant: following the link (a lexical check only).
#[test]
fn a_reparse_point_or_a_folder_at_the_lease_path_is_refused() {
    let scratch = Scratch::new("lease-unsafe");
    let target = scratch.path().join("elsewhere");
    std::fs::create_dir_all(&target).unwrap();
    let dir = scratch.path().join("leases");
    std::fs::create_dir_all(&dir).unwrap();
    let junction = dir.join("junction.lock");
    std::fs::create_dir_all(&junction).unwrap();
    lattice_sys::fs::seam::create_junction(&junction, &target).unwrap();
    assert!(matches!(try_acquire(&junction), Err(LeaseError::Unsafe(_))));
    let folder = dir.join("folder.lock");
    std::fs::create_dir_all(&folder).unwrap();
    assert!(matches!(try_acquire(&folder), Err(LeaseError::Unsafe(_))));
    let lease = WriterLease::at(Ok(junction));
    assert_eq!(lease.take(), Lease::Elsewhere, "fails closed");
    assert!(
        std::fs::read_dir(&target).unwrap().next().is_none(),
        "nothing was made through the junction"
    );
}

/// The derivation: below the state home, Python's layout; a folder below
/// the top level has the top level's lease; a linked worktree shares the
/// repository namespace and has its own checkout namespace; a folder without
/// git gets a native lease.
#[test]
fn the_lease_is_the_checkouts_and_a_folder_without_git_has_its_own() {
    let scratch = Scratch::new("lease-derive");
    let repo = scratch.repo("repo");
    std::fs::create_dir_all(repo.join("sub")).unwrap();
    let wt = scratch.path().join("wt");
    scratch.git(
        &repo,
        &["worktree", "add", "-q", &wt.to_string_lossy(), "-b", "side"],
    );
    let runner = scratch.runner();
    let state = StateRoot::at(scratch.path().join("state"));
    let attach =
        |path: &std::path::Path| attach_path(path, &scratch.env(), &state, &runner).unwrap();
    let home = scratch.path().join("home").join(".alelyon").join("globals");
    let top = derive_checkout_lease(&attach(&repo), &runner, &home).unwrap();
    let relative = top.path.strip_prefix(&home).unwrap();
    let parts: Vec<String> = relative
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect();
    assert_eq!(parts.len(), 5);
    assert_eq!(parts[0], "fleet_repositories");
    assert_eq!(parts[1], top.repository_namespace);
    assert_eq!(parts[2], "checkouts");
    assert_eq!(parts[3], top.checkout_namespace);
    assert_eq!(parts[4], CHECKOUT_LEASE_FILE);
    let below = derive_checkout_lease(&attach(&repo.join("sub")), &runner, &home).unwrap();
    assert_eq!(below, top, "the top level's, not the attached folder's");
    let linked = derive_checkout_lease(&attach(&wt), &runner, &home).unwrap();
    assert_eq!(linked.repository_namespace, top.repository_namespace);
    assert_ne!(linked.checkout_namespace, top.checkout_namespace);
    assert_eq!(
        lease_file_for(&attach(&repo), &runner, &state, Platform::host()).unwrap(),
        top.path,
        "the production derivation: Python's default state home"
    );
    let plain = scratch.path().join("plain");
    std::fs::create_dir_all(&plain).unwrap();
    let workspace = attach(&plain);
    assert_eq!(
        lease_file_for(&workspace, &runner, &state, Platform::host()).unwrap(),
        state
            .native_chat_dir()
            .join("leases")
            .join(format!("{}.lock", workspace.id))
    );
}

/// One conversation over the shared folder: its record, staging, checkpoints
/// and lease.
struct Conversation {
    staging: Staging,
    checkpoints: Checkpoints,
    lease: WriterLease,
    clock: Clock,
}

impl Conversation {
    fn new(id: &str, store: &SidecarStore, workspace: &Workspace, scratch: &Scratch) -> Self {
        let (sidecar, _) = store
            .open_for_writing(
                id,
                3.5,
                NewMeta {
                    workspace: None,
                    mode: Mode::Agent,
                    origin: Origin::Native,
                },
            )
            .unwrap();
        let sidecar = Arc::new(sidecar);
        let clock: Clock = Arc::new(|| 3000.0);
        Self {
            staging: Staging::new(Arc::clone(&sidecar)),
            checkpoints: Checkpoints::new(sidecar, Arc::clone(&clock)),
            lease: WriterLease::for_workspace(
                workspace,
                &scratch.runner(),
                &StateRoot::at(scratch.path().join("state")),
                Platform::host(),
            ),
            clock,
        }
    }

    fn edit(&self, workspace: &Workspace, scratch: &Scratch, path: &str, old: &str, new: &str) {
        edit_file(
            &StageContext {
                workspace,
                runner: &scratch.runner(),
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

    fn review(
        &self,
        workspace: &Workspace,
        scratch: &Scratch,
        ops: Vec<ReviewOp>,
    ) -> Vec<ReviewResult> {
        let runner = scratch.runner();
        let commands = CommandSlots::default();
        let guards = FolderGuards {
            checkpoints: &self.checkpoints,
            workspace,
            runner: &runner,
            lease: &self.lease,
            commands: &commands,
        };
        let confirmer = Confirmer::new(Arc::new(RecordingConfirm::answering(true)));
        block_on(review(
            &ReviewContext {
                workspace,
                runner: &runner,
                staging: &self.staging,
                mode: Mode::Agent,
                trusted: true,
                guards: &guards,
                confirmer: &confirmer,
                clock: &self.clock,
            },
            ops,
        ))
        .results
        .into_iter()
        .map(|result| result.result)
        .collect()
    }

    fn keep(&self, workspace: &Workspace, scratch: &Scratch, path: &str) -> ReviewResult {
        let id = self
            .staging
            .changes()
            .into_iter()
            .rev()
            .find(|change| change.path == path)
            .unwrap()
            .id;
        self.review(
            workspace,
            scratch,
            vec![ReviewOp::Keep {
                change: id,
                hunks: None,
            }],
        )
        .remove(0)
    }
}

/// SF10: while one conversation holds the folder's lease (it kept a change
/// and has another waiting), a second conversation's Keep is refused and
/// writes nothing; once the first has nothing waiting and gives the lease
/// back, the second keeps.
/// Mutant: no lease (`take` always answers `Held`).
#[test]
fn sf10_a_second_conversations_keep_waits_for_the_lease() {
    let scratch = Scratch::new("lease-sf10");
    let repo = scratch.repo("repo");
    std::fs::write(repo.join("b.txt"), "b\n").unwrap();
    let runner = scratch.runner();
    let state = StateRoot::at(scratch.path().join("state"));
    let workspace = attach_path(&repo, &scratch.env(), &state, &runner).unwrap();
    let store = SidecarStore::new(state.native_chat_dir());
    let first = Conversation::new("c0ffee00000a", &store, &workspace, &scratch);
    let second = Conversation::new("c0ffee00000b", &store, &workspace, &scratch);
    assert_eq!(
        first.lease.path(),
        second.lease.path(),
        "one lease per checkout"
    );

    first.edit(&workspace, &scratch, "a.txt", "a", "first");
    first.edit(&workspace, &scratch, ".gitignore", "*.log", "*.tmp");
    assert!(matches!(
        first.keep(&workspace, &scratch, "a.txt"),
        ReviewResult::Kept { .. }
    ));
    assert!(
        !first
            .lease
            .release_when_idle(first.staging.waiting(), false),
        "a change still waits"
    );

    second.edit(&workspace, &scratch, "b.txt", "b", "second");
    assert_eq!(
        second.keep(&workspace, &scratch, "b.txt"),
        ReviewResult::Conflict {
            reason: "Another Lattice agent is editing this folder.".into()
        }
    );
    assert_eq!(
        std::fs::read(repo.join("b.txt")).unwrap(),
        b"b\n",
        "nothing written"
    );
    assert!(first.lease.is_held(), "the first conversation holds it");

    first.review(&workspace, &scratch, vec![ReviewOp::UndoAll { note: None }]);
    assert_eq!(first.staging.waiting(), 0);
    assert!(
        first
            .lease
            .release_when_idle(first.staging.waiting(), false)
    );
    let kept = second.keep(&workspace, &scratch, "b.txt");
    assert!(matches!(kept, ReviewResult::Kept { .. }), "{kept:?}");
    assert_eq!(std::fs::read(repo.join("b.txt")).unwrap(), b"second\n");
    assert!(second.lease.is_held());
}
