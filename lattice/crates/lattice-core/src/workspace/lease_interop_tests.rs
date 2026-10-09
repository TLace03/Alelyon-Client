//! Interop tests: the native writer lease and Python's on one file
//! (the chat core's spec §6.5, §16.3 I12 and I13; row E5).
//!
//! Each test makes scratch repositories in a temporary folder and starts the
//! repository's Python (`tools/lattice_chat_interop.py`, through
//! `chat::interop_tests::Py`, with that module's deadline) as a child that
//! derives, takes or tries the lease with the web's own code:
//! `RepositoryStatePaths.derive(resolve_repository_context_blocking(root),
//! state_home=selected_repository_state_root()).checkout_file(
//! CHECKOUT_LEASE_FILE)` and `process_lease.try_acquire`. Both sides use
//! the **production default** state home, with no path injected: the
//! child's `USERPROFILE` and `HOME` are the scratch area's own home and its
//! `ALELYON_HOME` is unset, and the native side reads the same variables
//! through its `Env`, so `~/.alelyon/globals` is a temporary folder on both
//! sides. One case sets `ALELYON_HOME` (to a temporary folder) on both. The
//! helper refuses a state home or a checkout outside the temporary folder.
//!
//! When no interpreter is found each test prints that it was SKIPPED and why,
//! as the store's interop tests do; when one is, each prints that it RAN.

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};

use super::attach::attach_path;
use super::lease::{WriterLease, derive_checkout_lease, normcase, selected_repository_state_root};
use crate::chat::interop_tests::{Py, Running};
use crate::chat::pyjson::PyValue;
use crate::env::MapEnv;
use crate::git::runner::GitRunner;
use crate::git::tests::Scratch;
use crate::policy::Lease;
use crate::state::{Platform, StateRoot, resolve_with};

fn python_or_skip(name: &'static str) -> Option<Py> {
    match Py::new(name) {
        Ok(py) => Some(py),
        Err(why) => {
            println!("interop {name}: SKIPPED: {why}");
            None
        }
    }
}

fn text(value: &PyValue, key: &str) -> String {
    value
        .get(key)
        .and_then(PyValue::as_str)
        .unwrap_or_else(|| panic!("no {key}"))
        .to_owned()
}

fn flag(value: &PyValue, key: &str) -> bool {
    match value.get(key) {
        Some(PyValue::Bool(value)) => *value,
        other => panic!("{key} is {other:?}"),
    }
}

/// Start the helper's lease `command` for the checkout at `root`, with the
/// scratch home as `USERPROFILE` and `HOME`, and `ALELYON_HOME` set only
/// when `alelyon_home` is given.
fn start(
    py: &Py,
    scratch: &Scratch,
    alelyon_home: Option<&Path>,
    command: &str,
    root: &Path,
) -> Running {
    let home = scratch.home();
    let mut set: Vec<(&str, &OsStr)> = vec![
        ("USERPROFILE", home.as_os_str()),
        ("HOME", home.as_os_str()),
    ];
    let mut remove = vec!["ALELYON_FORCE_PACKAGED", "CLAUDECODE", "AI_AGENT"];
    match alelyon_home {
        Some(alelyon) => set.push(("ALELYON_HOME", alelyon.as_os_str())),
        None => remove.push("ALELYON_HOME"),
    }
    let args: [OsString; 2] = ["--root".into(), root.as_os_str().to_owned()];
    py.start_with(command, &args, &set, &remove)
}

/// The native side's runner and state root: the scratch's environment, with
/// `ALELYON_HOME` when given (and the state root resolved from it), else a
/// source checkout's.
fn native(scratch: &Scratch, alelyon_home: Option<&Path>) -> (GitRunner, StateRoot) {
    let mut env: MapEnv = scratch.env();
    let state = match alelyon_home {
        Some(home) => {
            env.set("ALELYON_HOME", home.as_os_str());
            resolve_with(&env, None, Platform::host())
        }
        None => StateRoot::at(scratch.path().join("state")),
    };
    (GitRunner::new(std::sync::Arc::new(env), &state), state)
}

/// I12 with one state home: Python holds the lease and the native side is
/// refused; then the native side holds it and Python is refused; released,
/// Python takes it. The file stays (never deleted), and both sides name the
/// same file.
fn i12_round(name: &'static str, alelyon_home: Option<&Path>, scratch: &Scratch, repo: &Path) {
    let Some(py) = python_or_skip(name) else {
        return;
    };
    let (runner, state) = native(scratch, alelyon_home);
    let workspace = attach_path(repo, runner.env(), &state, &runner).unwrap();
    let lease = WriterLease::for_workspace(&workspace, &runner, &state, Platform::host());
    let path = lease.path().unwrap().to_path_buf();
    let home = selected_repository_state_root(runner.env(), &state, Platform::host());
    assert!(
        path.starts_with(&home),
        "{} under {}",
        path.display(),
        home.display()
    );

    let mut holding = start(&py, scratch, alelyon_home, "lease-hold", repo);
    let held = holding.event("held");
    assert!(flag(&held, "got"), "Python took the free lease");
    assert_eq!(
        normcase(Path::new(&text(&held, "path"))),
        normcase(&path),
        "one file on both sides"
    );
    assert_eq!(lease.take(), Lease::Elsewhere, "Python holds it: refused");
    holding.send("release");
    let done = holding.finish();
    assert!(
        flag(&done, "exists_after"),
        "Python's release deletes nothing"
    );

    assert_eq!(lease.take(), Lease::Held, "free again: taken");
    let tried = start(&py, scratch, alelyon_home, "lease-try", repo).finish();
    assert!(
        !flag(&tried, "got"),
        "the native side holds it: Python refused"
    );
    assert!(lease.release_when_idle(0, false));
    assert!(path.is_file(), "the native release deletes nothing");
    let tried = start(&py, scratch, alelyon_home, "lease-try", repo).finish();
    assert!(flag(&tried, "got"), "released: Python takes it");
}

/// I12: the lease both ways, under the production default state home
/// (`~/.alelyon/globals`, the home a temporary folder), and under
/// `ALELYON_HOME`.
/// Mutant: the state home taken from `StateRoot::globals` (the checkout's
/// `globals/`), which locks another file and excludes nothing.
#[test]
fn i12_python_and_the_native_lease_exclude_each_other() {
    let scratch = Scratch::new("lease-i12");
    let repo = scratch.repo("repo");
    i12_round("I12 default home", None, &scratch, &repo);
    let alelyon = scratch.path().join("alelyon-home");
    i12_round("I12 ALELYON_HOME", Some(&alelyon), &scratch, &repo);
}

/// I13: over live repositories made here (a plain repository, two linked
/// worktrees sharing its common folder, and a folder attached below the top
/// level), the native lease path equals Python's for the same checkout, with
/// Python's default state home. The worktrees share the repository
/// namespace and differ in the checkout namespace; the folder below the top
/// level has its top level's lease.
/// Mutants: the lease derived from the attached folder rather than git's top
/// level; the linked worktree's `.git` pointer left out of the marker.
#[test]
fn i13_the_native_lease_path_is_pythons() {
    let Some(py) = python_or_skip("I13") else {
        return;
    };
    let scratch = Scratch::new("lease-i13");
    let repo = scratch.repo("repo");
    std::fs::create_dir_all(repo.join("sub").join("deeper")).unwrap();
    std::fs::write(repo.join("sub").join("deeper").join("f.txt"), "f\n").unwrap();
    scratch.git(&repo, &["add", "."]);
    scratch.git(&repo, &["commit", "-q", "-m", "two"]);
    let wt = scratch.path().join("wt");
    let wt2 = scratch.path().join("wt2");
    scratch.git(
        &repo,
        &["worktree", "add", "-q", &wt.to_string_lossy(), "-b", "side"],
    );
    scratch.git(
        &repo,
        &[
            "worktree",
            "add",
            "-q",
            &wt2.to_string_lossy(),
            "-b",
            "other",
        ],
    );
    let (runner, state) = native(&scratch, None);
    let home = selected_repository_state_root(runner.env(), &state, Platform::host());
    assert_eq!(home, scratch.home().join(".alelyon").join("globals"));

    // (the attached folder, the checkout's top level Python is given)
    let cases: [(PathBuf, &Path); 4] = [
        (repo.clone(), &repo),
        (wt.clone(), &wt),
        (wt2.clone(), &wt2),
        (repo.join("sub").join("deeper"), &repo),
    ];
    let mut seen = Vec::new();
    for (attached, top) in &cases {
        let workspace = attach_path(attached, runner.env(), &state, &runner).unwrap();
        let native = derive_checkout_lease(&workspace, &runner, &home).unwrap();
        let python = start(&py, &scratch, None, "lease-path", top).finish();
        println!("{}: {}", attached.display(), native.path.display());
        assert_eq!(
            normcase(&native.path),
            normcase(Path::new(&text(&python, "path"))),
            "{}",
            attached.display()
        );
        assert_eq!(
            native.repository_namespace,
            text(&python, "repository_namespace")
        );
        assert_eq!(
            native.checkout_namespace,
            text(&python, "checkout_namespace")
        );
        seen.push(native);
    }
    let repository = &seen[0].repository_namespace;
    assert!(
        seen.iter()
            .all(|lease| &lease.repository_namespace == repository),
        "one repository: one namespace"
    );
    assert_ne!(seen[0].checkout_namespace, seen[1].checkout_namespace);
    assert_ne!(seen[1].checkout_namespace, seen[2].checkout_namespace);
    assert_eq!(
        seen[3].path, seen[0].path,
        "below the top level: the top level's lease"
    );
}
