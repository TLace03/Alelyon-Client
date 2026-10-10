//! Git from the chat (`super::git_tools`), over the agent harness's scratch
//! repository: the status read; a commit after the reader's yes, with the
//! repository's hook run and the files shown; a no commits nothing; Lattice's
//! staged changes waiting refuse a commit; a branch made, committed and pushed
//! to a bare remote on this PC (and a pull request asked of `gh` when it is
//! found); the base branch refused.

use std::path::Path;
use std::process::Command;

use serde_json::json;

use super::agent_tests::{H, call, say};
use super::git_tools::{Status, commit_files, parse_status, status_text};
use crate::ports::ConfirmRequest;

fn git(folder: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args(args)
        .current_dir(folder)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

fn result_of(calls: &[lattice_agents::model::ModelRequest], call_id: &str) -> String {
    calls
        .iter()
        .flat_map(|request| request.input.iter())
        .find_map(|item| match item {
            lattice_agents::model::InputItem::ToolResult {
                call_id: id,
                output,
            } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no result for {call_id}"))
}

/// The reader's own identity, as their git config would give it.
fn identity(folder: &Path) {
    git(folder, &["config", "user.name", "Reader"]);
    git(folder, &["config", "user.email", "reader@example.com"]);
}

#[test]
fn a_status_is_read_from_gits_porcelain() {
    let out = b"## feature...origin/feature [ahead 2, behind 1]\0 M a.txt\0?? new.txt\0R  b.txt\0old.txt\0";
    let status = parse_status(out);
    assert_eq!(status.branch.as_deref(), Some("feature"));
    assert_eq!(status.upstream.as_deref(), Some("origin/feature"));
    assert_eq!((status.ahead, status.behind), (2, 1));
    assert_eq!(
        status.files,
        [
            (" M".into(), "a.txt".into()),
            ("??".into(), "new.txt".into()),
            ("R ".into(), "b.txt".into())
        ]
    );
    let text = status_text(&status);
    assert!(
        text.starts_with("On branch feature, tracking origin/feature (2 ahead, 1 behind)."),
        "{text}"
    );
    assert_eq!(parse_status(b"## HEAD (no branch)\0").branch, None);
    assert_eq!(
        parse_status(b"## No commits yet on main\0")
            .branch
            .as_deref(),
        Some("main")
    );
    // A commit's files: those named (each changed), else every changed file.
    assert_eq!(
        commit_files(&status, &[]).unwrap(),
        ["a.txt", "new.txt", "b.txt"]
    );
    assert_eq!(
        commit_files(&status, &["a.txt".into(), "a.txt".into()]).unwrap(),
        ["a.txt"]
    );
    assert!(
        commit_files(&status, &["c.txt".into()])
            .unwrap_err()
            .contains("no change")
    );
    assert!(
        commit_files(&Status::default(), &[])
            .unwrap_err()
            .contains("nothing to commit")
    );
}

#[test]
fn a_commit_waits_for_the_readers_yes_and_runs_the_repositorys_hook() {
    let h = H::new("git-commit");
    let ws = h.workspace();
    identity(&h.folder);
    // The repository's own hook: it leaves a mark when it runs.
    let hooks = h.folder.join(".git").join("hooks");
    std::fs::write(
        hooks.join("pre-commit"),
        "#!/bin/sh\necho ran > hook-ran.mark\n",
    )
    .unwrap();
    std::fs::write(h.folder.join("a.txt"), "changed\n").unwrap();
    let before = git(&h.folder, &["rev-parse", "HEAD"]);
    let model = h.script(vec![
        call("git_status", json!({}), "g0"),
        call(
            "git_commit",
            json!({"message": "Change a", "paths": ["a.txt"]}),
            "g1",
        ),
        say("committed"),
        call("git_commit", json!({"message": "Again"}), "g2"),
        say("not committed"),
    ]);
    let id = h.agent(None, "commit it", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(
        result_of(&calls, "g0").contains(" M a.txt"),
        "{}",
        result_of(&calls, "g0")
    );
    let done = result_of(&calls, "g1");
    assert!(done.starts_with("Committed on "), "{done}");
    assert_eq!(git(&h.folder, &["log", "-1", "--format=%s"]), "Change a");
    assert_ne!(git(&h.folder, &["rev-parse", "HEAD"]), before);
    assert!(
        h.folder.join("hook-ran.mark").exists(),
        "the repository's hook ran"
    );
    let asked = h.confirm.asked();
    assert!(asked.iter().any(|request| matches!(
        request,
        ConfirmRequest::GitCommit { message, files, .. } if message == "Change a" && files == &["a.txt".to_string()]
    )), "{asked:?}");

    // Declined: nothing is committed.
    std::fs::write(h.folder.join("a.txt"), "again\n").unwrap();
    *h.confirm.answer.lock().unwrap() = false;
    let head = git(&h.folder, &["rev-parse", "HEAD"]);
    h.agent(Some(&id), "commit again", &ws);
    h.turns_end(&id, 2);
    assert!(result_of(&model.calls(), "g2").contains("did not allow the commit"));
    assert_eq!(git(&h.folder, &["rev-parse", "HEAD"]), head);
}

#[test]
fn a_commit_waits_for_lattices_staged_changes_to_be_reviewed() {
    let h = H::new("git-commit-staged");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "edit_file",
            json!({"path": "a.txt", "old_string": "a", "new_string": "b"}),
            "e1",
        ),
        call("git_commit", json!({"message": "Too soon"}), "g1"),
        say("waiting"),
    ]);
    let id = h.agent(None, "edit and commit", &ws);
    h.turns_end(&id, 1);
    let refused = result_of(&model.calls(), "g1");
    assert!(
        refused.contains("Review 1 staged change first"),
        "{refused}"
    );
    assert!(
        !h.confirm
            .asked()
            .iter()
            .any(|r| matches!(r, ConfirmRequest::GitCommit { .. }))
    );
}

#[test]
fn a_branch_is_made_committed_and_pushed_after_the_readers_yes() {
    let h = H::new("git-push");
    let ws = h.workspace();
    identity(&h.folder);
    // A remote on this PC.
    let remote = h.folder.parent().unwrap().join("remote.git");
    std::fs::create_dir_all(&remote).unwrap();
    git(&remote, &["init", "-q", "--bare"]);
    git(
        &h.folder,
        &["remote", "add", "origin", &remote.to_string_lossy()],
    );
    let base = git(&h.folder, &["rev-parse", "--abbrev-ref", "HEAD"]);
    std::fs::write(h.folder.join("a.txt"), "feature\n").unwrap();
    let model = h.script(vec![
        call(
            "git_push_pr",
            json!({"title": "From base", "base": base}),
            "p0",
        ),
        call("git_branch", json!({"name": "feature-x"}), "b1"),
        call("git_commit", json!({"message": "Feature"}), "c1"),
        call(
            "git_push_pr",
            json!({"title": "Add the feature", "body": "Why", "base": base}),
            "p1",
        ),
        say("pushed"),
    ]);
    let id = h.agent(None, "push it", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(
        result_of(&calls, "p0").contains("itself: make a branch"),
        "{}",
        result_of(&calls, "p0")
    );
    assert!(
        result_of(&calls, "b1").starts_with("Made the branch"),
        "{}",
        result_of(&calls, "b1")
    );
    assert_eq!(
        git(&h.folder, &["rev-parse", "--abbrev-ref", "HEAD"]),
        "feature-x"
    );
    let pushed = result_of(&calls, "p1");
    assert!(pushed.starts_with("Pushed"), "{pushed}");
    assert!(
        !git(&remote, &["branch", "--list", "feature-x"]).is_empty(),
        "the branch is on the remote"
    );
    assert!(h.confirm.asked().iter().any(|request| matches!(
        request,
        ConfirmRequest::GitPush { branch, remote, title, .. }
            if branch == "feature-x" && remote == "origin" && title == "Add the feature"
    )));
}
