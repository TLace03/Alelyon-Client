//! Commit, push and open a pull request from the chat, as Claude Code's and
//! Codex's agents do.
//!
//! - `git_status {}` reads the folder's branch, its upstream and how far it
//!   is ahead or behind, and the files changed, through Lattice's own git
//!   (`git::runner`: no hook, no network), so reading starts no code of the
//!   repository's. Both modes.
//! - `git_branch {name}` makes a branch and switches to it (local only).
//! - `git_commit {message, paths}` commits after the reader's yes in the core's
//!   own dialog, which shows the branch, the message and the files. It runs
//!   the reader's own git, as a commit by hand would: the repository's hooks
//!   run (the dialog says so), with the reader's identity and signing. It
//!   takes what is on disk, so it waits while Lattice's staged changes wait
//!   for review. With no `paths`, every changed file is committed.
//! - `git_push_pr {title, body, base, draft}` pushes the branch to its remote
//!   and opens a pull request with GitHub's CLI (`gh`) when it is installed,
//!   after the reader's yes in the core's own dialog (branch, remote, base,
//!   title); without `gh` it pushes and says where to open one.
//!
//! The last three are Agent mode tools, and hold the folder's command slot
//! while they run, so no Keep or command lands inside them. A commit and a
//! push run as the reader, with their environment's credentials; the agent
//! never sees a credential, only git's own words (the last of them, redacted).

use std::ffi::{OsStr, OsString};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;
use futures::future::BoxFuture;
use lattice_sys::process::{JobLimits, SpawnRequest};
use serde_json::Value;

use super::turn::TurnTools;
use crate::git::dotgit::Repo;
use crate::git::runner::Extra;
use crate::ports::{ConfirmRequest, Initiated};
use crate::tools::read::ToolError;

/// How long a commit (its hooks included) may take.
pub const COMMIT_WAIT: Duration = Duration::from_secs(600);
/// How long a push or a pull request may take.
pub const PUSH_WAIT: Duration = Duration::from_secs(300);
/// The most files a commit lists in its dialog.
pub const DIALOG_FILES: usize = 40;
/// The most of git's own words the agent reads after a step.
const TAIL: usize = 4000;

/// Names a commit or a push needs beyond the safe block: proxies, signing,
/// and GitHub's CLI's own token when the reader set one.
const MORE_ENV: [&str; 8] = [
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "GNUPGHOME",
    "SSH_AUTH_SOCK",
    "GIT_SSH_COMMAND",
    "GH_TOKEN",
    "GH_HOST",
];

/// What one run of the reader's git (or `gh`) gave.
#[derive(Debug)]
pub struct Ran {
    pub ok: bool,
    /// Its stdout and stderr, the last [`TAIL`] characters, redacted.
    pub words: String,
}

fn tail(text: &str) -> String {
    let text = crate::secrets::redact(text.trim());
    let n = text.chars().count();
    if n <= TAIL {
        return text;
    }
    let cut: String = text.chars().skip(n - TAIL).collect();
    format!("[...]{cut}")
}

/// The environment a commit or a push runs with: the safe block, then
/// [`MORE_ENV`] as this process has them.
fn environment(
    env: &dyn crate::env::Env,
    folder: &Path,
    globals: &Path,
) -> Vec<(OsString, OsString)> {
    let mut block = crate::exec::spawn::child_environment(env, Some(folder), globals);
    for name in MORE_ENV {
        if block
            .iter()
            .any(|(n, _)| n.eq_ignore_ascii_case(OsStr::new(name)))
        {
            continue;
        }
        if let Some(value) = env.var(name) {
            block.push((name.into(), value));
        }
    }
    block
}

/// Run the reader's own `program` (resolved as a command's is: never from
/// the folder or Lattice's state) with `args` in `folder`, waiting at most
/// `wait`; its tree ends then.
pub fn run_as_reader(
    env: &dyn crate::env::Env,
    globals: &Path,
    folder: &Path,
    program: &str,
    args: &[&str],
    wait: Duration,
) -> Result<Ran, String> {
    let resolved = crate::exec::resolve::resolve_program(program, env, Some(folder), globals)
        .map_err(|_| format!("{program} was not found on this PC."))?;
    let mut argv: Vec<OsString> = vec![program.into()];
    argv.extend(args.iter().map(OsString::from));
    let block = environment(env, folder, globals);
    let mut child = lattice_sys::process::spawn(&SpawnRequest {
        program: &resolved.path,
        argv: &argv,
        cwd: folder,
        env: &block,
        limits: JobLimits::default(),
    })
    .map_err(|_| format!("{program} could not start."))?;
    let read = |pipe: Option<std::fs::File>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.by_ref().take(1024 * 1024).read_to_end(&mut bytes);
            }
            bytes
        })
    };
    let (out, err) = (read(child.take_stdout()), read(child.take_stderr()));
    let waited = child.wait(Some(wait));
    let _ = child.kill_tree();
    let mut text = String::from_utf8_lossy(&out.join().unwrap_or_default()).into_owned();
    text.push_str(&String::from_utf8_lossy(&err.join().unwrap_or_default()));
    match waited {
        Ok(Some(code)) => Ok(Ran {
            ok: code == 0,
            words: tail(&text),
        }),
        Ok(None) => Err(format!(
            "{program} did not finish within {} minutes, so it was stopped.",
            wait.as_secs() / 60
        )),
        Err(_) => Err(format!("{program} could not be waited on.")),
    }
}

/// The folder's state as `git status --porcelain=v1 --branch -z` gives it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Status {
    pub branch: Option<String>,
    pub upstream: Option<String>,
    pub ahead: u32,
    pub behind: u32,
    /// `(code, path)`: git's two-letter code and the path.
    pub files: Vec<(String, String)>,
}

/// Parse `git status --porcelain=v1 --branch -z`.
pub fn parse_status(out: &[u8]) -> Status {
    let text = String::from_utf8_lossy(out);
    let mut status = Status::default();
    let entries = text.split('\0').filter(|e| !e.is_empty());
    let mut renamed_from = false;
    for entry in entries {
        if renamed_from {
            // The old name of a rename: skipped.
            renamed_from = false;
            continue;
        }
        if let Some(head) = entry.strip_prefix("## ") {
            let (names, counts) = match head.split_once(" [") {
                Some((names, counts)) => (names, Some(counts.trim_end_matches(']'))),
                None => (head, None),
            };
            let (branch, upstream) = match names.split_once("...") {
                Some((branch, upstream)) => (branch, Some(upstream)),
                None => (names, None),
            };
            let branch = branch.trim();
            status.branch = (!branch.starts_with("HEAD (no branch)") && !branch.is_empty())
                .then(|| branch.trim_start_matches("No commits yet on ").to_owned());
            status.upstream = upstream.map(str::to_owned);
            for count in counts.into_iter().flat_map(|c| c.split(", ")) {
                if let Some(n) = count.strip_prefix("ahead ") {
                    status.ahead = n.parse().unwrap_or(0);
                } else if let Some(n) = count.strip_prefix("behind ") {
                    status.behind = n.parse().unwrap_or(0);
                }
            }
            continue;
        }
        if entry.len() > 3 {
            let (code, path) = entry.split_at(2);
            if code.starts_with('R') || code.starts_with('C') {
                renamed_from = true;
            }
            status.files.push((code.to_owned(), path[1..].to_owned()));
        }
    }
    status
}

/// The folder's status, through Lattice's own git (no hook, no network).
pub(crate) fn read_status(tools: &TurnTools) -> Result<Status, ToolError> {
    let workspace = tools.folder()?;
    let Repo::Git(repo) = &workspace.repo else {
        return Err(ToolError::new(
            "This folder has no git, so there is nothing to commit.",
        ));
    };
    let out = tools
        .inner
        .runner
        .run(
            repo,
            &[
                OsStr::new("status"),
                OsStr::new("--porcelain=v1"),
                OsStr::new("--branch"),
                OsStr::new("-z"),
                OsStr::new("--untracked-files=all"),
            ],
            &Extra::default(),
        )
        .map_err(|_| ToolError::new("git could not read this folder's status."))?;
    Ok(parse_status(&out.stdout))
}

/// `git_status`'s answer.
pub fn status_text(status: &Status) -> String {
    let mut out = match (&status.branch, &status.upstream) {
        (Some(branch), Some(up)) => format!(
            "On branch {branch}, tracking {up} ({} ahead, {} behind).",
            status.ahead, status.behind
        ),
        (Some(branch), None) => format!("On branch {branch}, with no upstream yet."),
        (None, _) => "Not on a branch (a detached HEAD).".to_owned(),
    };
    if status.files.is_empty() {
        out.push_str("\nNo file has changed.");
    } else {
        out.push_str(&format!("\n{} changed:", status.files.len()));
        for (code, path) in status.files.iter().take(200) {
            out.push_str(&format!("\n{code} {path}"));
        }
        if status.files.len() > 200 {
            out.push_str(&format!("\n... and {} more", status.files.len() - 200));
        }
    }
    out
}

fn text(args: &Value, key: &str) -> String {
    args.get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_owned()
}

/// The folder's root, its id, and the paths an Agent-mode git step needs.
fn folder(tools: &TurnTools) -> Result<(PathBuf, String), ToolError> {
    let workspace = tools.folder()?;
    if !matches!(workspace.repo, Repo::Git(_)) {
        return Err(ToolError::new("This folder has no git."));
    }
    Ok((workspace.root.clone(), workspace.id.clone()))
}

/// Run `work` holding the folder's command slot, on the blocking pool.
async fn in_slot<T: Send + 'static>(
    tools: &Arc<TurnTools>,
    call: &str,
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, ToolError> {
    let (_, id) = folder(tools)?;
    let Some(slot) = tools.inner.slots.claim(&id, call) else {
        return Err(ToolError::new(
            "A command is running in this folder; try again when it ends.",
        ));
    };
    let done = tools
        .inner
        .handle
        .spawn_blocking(move || {
            let done = work();
            drop(slot);
            done
        })
        .await
        .map_err(|_| ToolError::new("The git step stopped unexpectedly."))?;
    done.map_err(ToolError)
}

/// The four git tools.
pub(crate) fn tool(
    tools: Arc<TurnTools>,
    call: String,
    name: &'static str,
    args: Value,
) -> BoxFuture<'static, Result<String, ToolError>> {
    async move {
        match name {
            "git_status" => {
                let reading = tools.clone();
                let status = tools
                    .inner
                    .handle
                    .spawn_blocking(move || read_status(&reading))
                    .await
                    .map_err(|_| ToolError::new("The git step stopped unexpectedly."))??;
                Ok(status_text(&status))
            }
            "git_branch" => branch(tools, call, &text(&args, "name")).await,
            "git_commit" => {
                let paths: Vec<String> = args
                    .get("paths")
                    .and_then(Value::as_array)
                    .map(|all| {
                        all.iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect()
                    })
                    .unwrap_or_default();
                commit(tools, call, &text(&args, "message"), paths).await
            }
            _ => {
                let draft = args.get("draft").and_then(Value::as_bool).unwrap_or(false);
                push_pr(
                    tools,
                    call,
                    &text(&args, "title"),
                    &text(&args, "body"),
                    &text(&args, "base"),
                    draft,
                )
                .await
            }
        }
    }
    .boxed()
}

async fn branch(tools: Arc<TurnTools>, call: String, name: &str) -> Result<String, ToolError> {
    if name.is_empty()
        || name.len() > 200
        || name.chars().any(|c| c.is_whitespace() || c.is_control())
    {
        return Err(ToolError::new("Give the branch a name with no spaces."));
    }
    let (root, _) = folder(&tools)?;
    let (env, globals) = (
        tools.inner.config.env.clone(),
        tools.inner.config.state.globals.clone(),
    );
    let name = name.to_owned();
    let ran = in_slot(&tools, &call, move || {
        let checked = run_as_reader(
            env.as_ref(),
            &globals,
            &root,
            "git",
            &["check-ref-format", "--branch", &name],
            COMMIT_WAIT,
        )?;
        if !checked.ok {
            return Err(format!("{name} is not a name git takes for a branch."));
        }
        run_as_reader(
            env.as_ref(),
            &globals,
            &root,
            "git",
            &["switch", "-c", &name],
            COMMIT_WAIT,
        )
    })
    .await?;
    if ran.ok {
        Ok(format!(
            "Made the branch and switched to it.\n{}",
            ran.words
        ))
    } else {
        Err(ToolError(format!(
            "git did not make the branch:\n{}",
            ran.words
        )))
    }
}

/// The files a commit takes: the paths named, else every changed file.
pub fn commit_files(status: &Status, paths: &[String]) -> Result<Vec<String>, String> {
    let changed: Vec<&str> = status.files.iter().map(|(_, p)| p.as_str()).collect();
    if changed.is_empty() {
        return Err("No file has changed, so there is nothing to commit.".to_owned());
    }
    if paths.is_empty() {
        return Ok(changed.iter().map(|p| (*p).to_owned()).collect());
    }
    let mut files = Vec::new();
    for path in paths {
        let path = path.trim().replace('\\', "/");
        if !changed.contains(&path.as_str()) {
            return Err(format!("{path} has no change to commit."));
        }
        if !files.contains(&path) {
            files.push(path);
        }
    }
    Ok(files)
}

async fn commit(
    tools: Arc<TurnTools>,
    call: String,
    message: &str,
    paths: Vec<String>,
) -> Result<String, ToolError> {
    if message.is_empty() {
        return Err(ToolError::new("Write the commit message."));
    }
    if message.chars().count() > 5000 {
        return Err(ToolError::new(
            "A commit message is at most 5,000 characters.",
        ));
    }
    let waiting = tools.staging.waiting();
    if waiting > 0 {
        return Err(ToolError(format!(
            "{} first: a commit takes what is on disk, and Lattice's staged changes are not there yet.",
            crate::exec::run::staged_first(waiting).trim_end_matches('.')
        )));
    }
    let reading = tools.clone();
    let status = tools
        .inner
        .handle
        .spawn_blocking(move || read_status(&reading))
        .await
        .map_err(|_| ToolError::new("The git step stopped unexpectedly."))??;
    let files = commit_files(&status, &paths).map_err(ToolError)?;
    let branch = status
        .branch
        .clone()
        .unwrap_or_else(|| "a detached HEAD".to_owned());
    let mut shown: Vec<String> = files.iter().take(DIALOG_FILES).cloned().collect();
    if files.len() > DIALOG_FILES {
        shown.push(format!("... and {} more", files.len() - DIALOG_FILES));
    }
    let yes = tools
        .inner
        .confirmer
        .ask(
            &format!("git-commit:{}:{call}", tools.convo.id),
            ConfirmRequest::GitCommit {
                branch: branch.clone(),
                message: message.to_owned(),
                files: shown,
            },
            Initiated::Page,
        )
        .await;
    if !yes {
        return Err(ToolError::new(
            "The user did not allow the commit, so nothing was committed.",
        ));
    }
    let (root, _) = folder(&tools)?;
    let (env, globals) = (
        tools.inner.config.env.clone(),
        tools.inner.config.state.globals.clone(),
    );
    let message = message.to_owned();
    let ran = in_slot(&tools, &call, move || {
        let mut add = vec!["add", "-A", "--"];
        add.extend(files.iter().map(String::as_str));
        let added = run_as_reader(env.as_ref(), &globals, &root, "git", &add, COMMIT_WAIT)?;
        if !added.ok {
            return Ok(added);
        }
        let mut commit = vec!["commit", "-m", message.as_str(), "--"];
        commit.extend(files.iter().map(String::as_str));
        run_as_reader(env.as_ref(), &globals, &root, "git", &commit, COMMIT_WAIT)
    })
    .await?;
    if ran.ok {
        Ok(format!("Committed on {branch}.\n{}", ran.words))
    } else {
        Err(ToolError(format!(
            "git did not commit (a hook may have refused it):\n{}",
            ran.words
        )))
    }
}

async fn push_pr(
    tools: Arc<TurnTools>,
    call: String,
    title: &str,
    body: &str,
    base: &str,
    draft: bool,
) -> Result<String, ToolError> {
    if title.is_empty() || title.chars().count() > 300 {
        return Err(ToolError::new(
            "Give the pull request a title of at most 300 characters.",
        ));
    }
    let reading = tools.clone();
    let status = tools
        .inner
        .handle
        .spawn_blocking(move || read_status(&reading))
        .await
        .map_err(|_| ToolError::new("The git step stopped unexpectedly."))??;
    let Some(branch) = status.branch.clone() else {
        return Err(ToolError::new(
            "Not on a branch: make one with git_branch and commit there first.",
        ));
    };
    let base = if base.is_empty() {
        "main".to_owned()
    } else {
        base.to_owned()
    };
    if branch == base {
        return Err(ToolError(format!(
            "This is {base} itself: make a branch with git_branch, commit there, then push it."
        )));
    }
    let remote = status
        .upstream
        .as_deref()
        .and_then(|up| up.split_once('/'))
        .map(|(remote, _)| remote.to_owned())
        .unwrap_or_else(|| "origin".to_owned());
    let (root, _) = folder(&tools)?;
    let env = tools.inner.config.env.clone();
    let globals = tools.inner.config.state.globals.clone();
    let has_gh =
        crate::exec::resolve::resolve_program("gh", env.as_ref(), Some(&root), &globals).is_ok();
    let yes = tools
        .inner
        .confirmer
        .ask(
            &format!("git-push:{}:{call}", tools.convo.id),
            ConfirmRequest::GitPush {
                branch: branch.clone(),
                remote: remote.clone(),
                base: base.clone(),
                title: title.to_owned(),
                pull_request: has_gh,
            },
            Initiated::Page,
        )
        .await;
    if !yes {
        return Err(ToolError::new(
            "The user did not allow the push, so nothing left this PC.",
        ));
    }
    let (title, body) = (title.to_owned(), body.to_owned());
    let (pushed, opened) = in_slot(&tools, &call, move || {
        let pushed = run_as_reader(
            env.as_ref(),
            &globals,
            &root,
            "git",
            &["push", "-u", &remote, &branch],
            PUSH_WAIT,
        )?;
        if !pushed.ok || !has_gh {
            return Ok((pushed, None));
        }
        let mut args = vec![
            "pr",
            "create",
            "--title",
            title.as_str(),
            "--body",
            body.as_str(),
            "--base",
            base.as_str(),
            "--head",
            branch.as_str(),
        ];
        if draft {
            args.push("--draft");
        }
        let opened = run_as_reader(env.as_ref(), &globals, &root, "gh", &args, PUSH_WAIT)?;
        Ok((pushed, Some(opened)))
    })
    .await?;
    if !pushed.ok {
        return Err(ToolError(format!("git did not push:\n{}", pushed.words)));
    }
    Ok(match opened {
        None => format!(
            "Pushed. GitHub's CLI (gh) is not installed, so no pull request was opened: the user can open one on the remote's site.\n{}",
            pushed.words
        ),
        Some(opened) if opened.ok => {
            format!("Pushed, and opened the pull request:\n{}", opened.words)
        }
        Some(opened) => format!(
            "Pushed, but gh did not open the pull request:\n{}",
            opened.words
        ),
    })
}
