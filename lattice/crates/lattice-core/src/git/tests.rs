//! The git runner and FT6's native reader against scratch repositories this
//! file makes (the chat core's spec §8.2, KF1 and KF8 for `ls-files` and
//! `check-ignore`; §6.3 FT6).
//!
//! Every repository is made in a temporary folder with `git init`, or as a
//! `file://` clone of one; no test runs git against a real checkout. The
//! setup's own git calls read no global or system configuration
//! (`GIT_CONFIG_GLOBAL` is an empty file, `GIT_CONFIG_NOSYSTEM=1`). The runner
//! reads its environment from a `MapEnv` whose home is the test's own folder.
//! Network paths are the documentation address `\\198.51.100.7\x`; nothing
//! here opens one (the falsifiers read `localfs::record`, which sees every
//! open the reader makes, and `lattice_sys` refuses such a path again).

use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use super::dotgit::{self, LocalRepo, Named, Repo, WithoutGit};
use super::runner::{
    BARE_OVERRIDE, Extra, FIXED_OVERRIDES, GIT_VARIABLES, GitError, GitRunner, GitVersion,
    IDENTITY, MIN_VERSION,
};
use crate::env::MapEnv;
use crate::exec::spawn::{NATIVE_ADDITIONS, SAFE_AGENT_ENV_NAMES};
use crate::localfs::{self, is_local_text};
use crate::state::StateRoot;
use crate::testkit::TempDir;

/// A scratch area: a home with an empty global configuration, and a state root.
pub(crate) struct Scratch {
    dir: TempDir,
}

impl Scratch {
    pub(crate) fn new(tag: &str) -> Self {
        let dir = TempDir::new(tag);
        std::fs::create_dir_all(dir.path().join("home")).unwrap();
        std::fs::write(dir.path().join("home").join(".gitconfig"), b"").unwrap();
        Self { dir }
    }

    pub(crate) fn path(&self) -> &Path {
        self.dir.path()
    }

    pub(crate) fn home(&self) -> PathBuf {
        self.path().join("home")
    }

    /// The setup's own git: no global or system configuration, a fixed
    /// identity, `main`, no line-end conversion.
    pub(crate) fn git(&self, cwd: &Path, args: &[&str]) -> std::process::Output {
        let output = Command::new("git")
            .arg("-c")
            .arg("user.name=t")
            .arg("-c")
            .arg("user.email=t@t")
            .arg("-c")
            .arg("init.defaultBranch=main")
            .arg("-c")
            .arg("core.autocrlf=false")
            .args(args)
            .current_dir(cwd)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_CONFIG_GLOBAL", self.home().join(".gitconfig"))
            .env("HOME", self.home())
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .output()
            .expect("git runs");
        assert!(
            output.status.success(),
            "setup git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        output
    }

    /// A repository with one commit: `a.txt`, `.gitignore` (`*.log`).
    pub(crate) fn repo(&self, name: &str) -> PathBuf {
        let repo = self.path().join(name);
        std::fs::create_dir_all(&repo).unwrap();
        self.git(&repo, &["init", "-q"]);
        std::fs::write(repo.join("a.txt"), b"a\n").unwrap();
        std::fs::write(repo.join(".gitignore"), b"*.log\n").unwrap();
        self.git(&repo, &["add", "a.txt", ".gitignore"]);
        self.git(&repo, &["commit", "-q", "-m", "one"]);
        repo
    }

    /// The runner's environment: this process's `PATH` and Windows' own
    /// names, a home of the test's own, and a sentinel key.
    pub(crate) fn env(&self) -> MapEnv {
        let mut env = MapEnv::new()
            .with("USERPROFILE", self.home().as_os_str())
            .with("HOME", self.home().as_os_str())
            .with("TEMP", self.path().as_os_str())
            .with("TMP", self.path().as_os_str())
            .with("ANTHROPIC_API_KEY", "lattice-git-runner-sentinel")
            .with("GIT_DIR", r"C:\elsewhere\.git")
            .with("GIT_CONFIG_PARAMETERS", "'core.fsmonitor'='calc'");
        for name in ["PATH", "SystemRoot", "windir", "PATHEXT", "COMSPEC"] {
            if let Some(value) = std::env::var_os(name) {
                env.set(name, value);
            }
        }
        env
    }

    pub(crate) fn runner(&self) -> GitRunner {
        GitRunner::new(
            Arc::new(self.env()),
            &StateRoot::at(self.path().join("state")),
        )
    }
}

fn local(repo: &Path, env: &MapEnv) -> LocalRepo {
    match dotgit::inspect(repo, env) {
        Repo::Git(local) => local,
        other => panic!("{}: {other:?}", repo.display()),
    }
}

fn names(block: &[(OsString, OsString)]) -> Vec<String> {
    block
        .iter()
        .map(|(name, _)| name.to_string_lossy().into_owned())
        .collect()
}

// ------------------------------------------------------------------ the runner

/// §8.2: every argv starts with the four overrides, then `--no-pager` and
/// the bare-repository override, then the call's own arguments.
/// Mutant: an override dropped (KF1's mutant below shows what that runs).
#[test]
fn every_argv_starts_with_the_overrides() {
    let scratch = Scratch::new("git-argv");
    let runner = scratch.runner();
    let argv = runner.argv(Path::new(r"C:\git\git.exe"), &[OsString::from("status")]);
    let text: Vec<String> = argv
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    assert_eq!(text[0], r"C:\git\git.exe");
    assert_eq!(text[1], "-c");
    assert!(text[2].starts_with("core.hooksPath="), "{}", text[2]);
    assert!(
        text[2].ends_with(r"lattice_native\chat\no-hooks"),
        "{}",
        text[2]
    );
    for (at, setting) in FIXED_OVERRIDES.iter().enumerate() {
        assert_eq!(text[3 + 2 * at], "-c");
        assert_eq!(text[4 + 2 * at], *setting);
    }
    assert_eq!(
        text[9..],
        ["--no-pager", "-c", BARE_OVERRIDE, "status"].map(String::from)
    );
}

/// §8.2 and X7: the environment is X7's names, the native addition and the
/// four git variables, and nothing else: no key, no `GIT_DIR`, no
/// `GIT_CONFIG_PARAMETERS` from Lattice's own environment. The index and the
/// identity are added only when asked for.
/// Mutant: `GIT_NO_LAZY_FETCH` and `GIT_ALLOW_PROTOCOL` dropped (KF8 shows
/// the fetch that follows).
#[test]
fn the_environment_is_x7_and_the_git_variables_only() {
    let scratch = Scratch::new("git-env");
    let runner = scratch.runner();
    let block = runner.environment(None, &Extra::default());
    let got = names(&block);
    let mut expected: Vec<String> = SAFE_AGENT_ENV_NAMES
        .iter()
        .filter(|name| scratch.env().var_present(name))
        .map(|name| (*name).to_owned())
        .collect();
    expected.extend(NATIVE_ADDITIONS.iter().map(|(name, _)| (*name).to_owned()));
    expected.extend(GIT_VARIABLES.iter().map(|(name, _)| (*name).to_owned()));
    assert_eq!(got, expected);
    for (name, value) in GIT_VARIABLES {
        assert!(
            block
                .iter()
                .any(|(n, v)| n == OsStr::new(name) && v == OsStr::new(value)),
            "{name}={value}"
        );
    }
    assert!(!format!("{block:?}").contains("sentinel"));
    let with = runner.environment(
        None,
        &Extra {
            index_file: Some(PathBuf::from(r"C:\idx")),
            identity: true,
        },
    );
    let got = names(&with);
    assert!(got.contains(&"GIT_INDEX_FILE".to_owned()));
    for (name, _) in IDENTITY {
        assert!(got.contains(&name.to_owned()), "{name}");
    }
}

trait Present {
    fn var_present(&self, name: &str) -> bool;
}

impl Present for MapEnv {
    fn var_present(&self, name: &str) -> bool {
        crate::env::Env::var(self, name).is_some()
    }
}

#[test]
fn versions_are_read_and_the_installed_git_is_new_enough() {
    assert_eq!(
        GitVersion::parse("git version 2.54.0.windows.1\n"),
        Some(GitVersion {
            major: 2,
            minor: 54,
            patch: 0
        })
    );
    assert_eq!(
        GitVersion::parse("git version 2.45"),
        Some(GitVersion {
            major: 2,
            minor: 45,
            patch: 0
        })
    );
    assert_eq!(GitVersion::parse("hello"), None);
    assert!(
        GitVersion {
            major: 2,
            minor: 44,
            patch: 9
        } < MIN_VERSION
    );
    let scratch = Scratch::new("git-version");
    let version = scratch.runner().version().unwrap();
    println!("the installed git is {version}");
    assert!(version >= MIN_VERSION, "{version}");
}

/// R10: a git older than the minimum is refused for every call in a folder.
/// Mutant: the version check skipped.
#[test]
fn a_git_older_than_the_minimum_is_refused() {
    let scratch = Scratch::new("git-old");
    let repo = scratch.repo("r");
    let env = scratch.env();
    let local = local(&repo, &env);
    let runner = scratch.runner();
    let old = GitVersion {
        major: 2,
        minor: 44,
        patch: 0,
    };
    runner.assume_version(old);
    assert_eq!(runner.ls_files(&local), Err(GitError::TooOld(old)));
    assert_eq!(
        runner.check_ignore(&local, "x.log"),
        Err(GitError::TooOld(old))
    );
}

#[test]
fn ls_files_lists_what_git_sees_and_check_ignore_answers_three_ways() {
    let scratch = Scratch::new("git-list");
    let repo = scratch.repo("r");
    std::fs::create_dir_all(repo.join("sub")).unwrap();
    std::fs::write(repo.join("sub").join("new.txt"), b"n").unwrap();
    std::fs::write(repo.join("sub").join("skip.log"), b"x").unwrap();
    let env = scratch.env();
    let runner = scratch.runner();
    let top = local(&repo, &env);
    let listing = runner.ls_files(&top).unwrap();
    // As git lists them: the untracked files it does not ignore, then the
    // tracked ones.
    assert_eq!(listing.paths, ["sub/new.txt", ".gitignore", "a.txt"]);
    assert!(!listing.truncated);
    // Below the top level: the folder's own files, relative to it.
    let below = local(&repo.join("sub"), &env);
    assert_eq!(runner.ls_files(&below).unwrap().paths, ["new.txt"]);
    assert_eq!(runner.check_ignore(&below, "skip.log"), Ok(true));
    assert_eq!(runner.check_ignore(&below, "new.txt"), Ok(false));
    assert!(matches!(
        runner.check_ignore(&below, "../../outside.txt"),
        Err(GitError::CouldNotSay { .. })
    ));
    let top_level = runner.top_level(&below).unwrap();
    assert!(
        localfs::is_inside(&repo, &top_level) && localfs::is_inside(&top_level, &repo),
        "{} {}",
        top_level.display(),
        repo.display()
    );
}

/// The listing's cap (`MAX_TREE_ENTRIES`), with a smaller cap.
#[test]
fn a_listing_past_its_cap_says_so() {
    let listing = super::runner::parse_listing(b"a\0b\0c\0\xff\0d\0", 3);
    assert_eq!(listing.paths, ["a", "b", "c"]);
    assert!(listing.truncated);
    let listing = super::runner::parse_listing(b"a\0\xff\0", 3);
    assert_eq!(listing.paths, ["a"]);
    assert_eq!(listing.not_text, 1);
    assert!(!listing.truncated);
}

/// A repository whose own configuration would run a program for anything
/// that reads its index or working tree: an fsmonitor command, hooks (in
/// `.git/hooks` and in a `core.hooksPath` of its own), and clean, smudge and
/// process filters for every file. Each writes a marker into `markers`.
pub(crate) fn hostile(scratch: &Scratch, name: &str) -> (PathBuf, PathBuf) {
    let repo = scratch.repo(name);
    let markers = scratch.path().join(format!("{name}-markers"));
    std::fs::create_dir_all(&markers).unwrap();
    let m = markers.to_string_lossy().replace('\\', "/");
    let hook_dirs = [repo.join(".git").join("hooks"), repo.join("own-hooks")];
    for dir in &hook_dirs {
        std::fs::create_dir_all(dir).unwrap();
        for hook in [
            "reference-transaction",
            "post-checkout",
            "post-index-change",
            "pre-commit",
            "post-commit",
            "pre-auto-gc",
        ] {
            std::fs::write(
                dir.join(hook),
                format!("#!/bin/sh\necho {hook} >> '{m}/hook-{hook}'\n"),
            )
            .unwrap();
        }
    }
    std::fs::write(repo.join(".gitattributes"), b"* filter=x\n").unwrap();
    std::fs::write(repo.join("b.txt"), b"b\n").unwrap();
    for (key, value) in [
        ("core.hooksPath", "own-hooks".to_owned()),
        (
            "core.fsmonitor",
            format!("echo fsmonitor >> '{m}/fsmonitor'; false"),
        ),
        (
            "filter.x.clean",
            format!("sh -c \"echo clean >> '{m}/clean'; cat\""),
        ),
        (
            "filter.x.smudge",
            format!("sh -c \"echo smudge >> '{m}/smudge'; cat\""),
        ),
        (
            "filter.x.process",
            format!("sh -c \"echo process >> '{m}/process'; exit 1\""),
        ),
    ] {
        scratch.git(&repo, &["config", key, &value]);
    }
    (repo, markers)
}

pub(crate) fn markers(dir: &Path) -> Vec<String> {
    let mut found: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    found
}

/// KF1, the parts for `ls-files`, `check-ignore` and `rev-parse`: a hostile
/// repository runs none of its programs through the runner. The positive
/// control runs the same listing with plain git and must leave a marker, or
/// the repository proves nothing.
/// Mutant: the `-c` overrides dropped from the argv.
#[test]
fn kf1_a_hostile_repository_runs_none_of_its_programs() {
    let scratch = Scratch::new("git-kf1");
    let (control, control_markers) = hostile(&scratch, "control");
    Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(&control)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", scratch.home().join(".gitconfig"))
        .output()
        .unwrap();
    let seen = markers(&control_markers);
    println!("plain git ran: {seen:?}");
    assert!(
        !seen.is_empty(),
        "the positive control: plain git runs the repository's fsmonitor"
    );

    let (repo, repo_markers) = hostile(&scratch, "hostile");
    let env = scratch.env();
    let local = local(&repo, &env);
    let runner = scratch.runner();
    let listing = runner.ls_files(&local).unwrap();
    assert!(listing.paths.contains(&"b.txt".to_owned()));
    assert_eq!(runner.check_ignore(&local, "x.log"), Ok(true));
    assert_eq!(runner.check_ignore(&local, "b.txt"), Ok(false));
    runner.top_level(&local).unwrap();
    assert_eq!(
        markers(&repo_markers),
        Vec::<String>::new(),
        "through the runner, no program of the repository ran"
    );
}

/// A `file://` partial clone (`--filter=blob:none`) of a scratch repository,
/// with `protocol.file.allow=always` in its own configuration and its
/// `.gitignore` files left out of the working tree (sparse), so reading them
/// needs blobs the clone does not have.
fn partial_clone(scratch: &Scratch, source: &Path, name: &str) -> PathBuf {
    partial_clone_sparse(scratch, source, name, &["/a.txt"])
}

/// [`partial_clone`] with the sparse checkout's own patterns (non-cone):
/// what they match is checked out, with its blobs; the rest is
/// skip-worktree, its blobs missing.
pub(crate) fn partial_clone_sparse(
    scratch: &Scratch,
    source: &Path,
    name: &str,
    patterns: &[&str],
) -> PathBuf {
    let clone = scratch.path().join(name);
    let url = format!("file:///{}", source.to_string_lossy().replace('\\', "/"));
    scratch.git(
        scratch.path(),
        &[
            "clone",
            "-q",
            "--filter=blob:none",
            "--no-checkout",
            &url,
            &clone.to_string_lossy(),
        ],
    );
    scratch.git(&clone, &["config", "protocol.file.allow", "always"]);
    let mut set = vec!["sparse-checkout", "set", "--no-cone"];
    set.extend_from_slice(patterns);
    scratch.git(&clone, &set);
    scratch.git(&clone, &["checkout", "-q", "main"]);
    clone
}

/// WP8b's fixture: a partial, sparse clone in which `sub/.gitignore`
/// (`secret.txt`) is skip-worktree and its blob is missing, while the top
/// level's and `other/`'s are checked out. `sub/` then holds an untracked
/// `secret.txt` and `visible.txt`, made after the checkout; `other/` holds
/// `x.tmp`, which its readable `.gitignore` ignores.
pub(crate) fn unreadable_gitignore_clone(scratch: &Scratch, secret: &str) -> PathBuf {
    let source = scratch.path().join("wp8b-source");
    std::fs::create_dir_all(source.join("sub")).unwrap();
    std::fs::create_dir_all(source.join("other")).unwrap();
    scratch.git(&source, &["init", "-q"]);
    std::fs::write(source.join("a.txt"), b"a\n").unwrap();
    std::fs::write(source.join(".gitignore"), b"*.log\n").unwrap();
    std::fs::write(source.join("sub").join(".gitignore"), b"secret.txt\n").unwrap();
    std::fs::write(source.join("sub").join("b.txt"), b"b\n").unwrap();
    std::fs::write(source.join("other").join(".gitignore"), b"*.tmp\n").unwrap();
    std::fs::write(source.join("other").join("o.txt"), b"other line\n").unwrap();
    scratch.git(&source, &["add", "."]);
    scratch.git(&source, &["commit", "-q", "-m", "one"]);
    scratch.git(&source, &["config", "uploadpack.allowFilter", "true"]);
    let clone = partial_clone_sparse(
        scratch,
        &source,
        "wp8b-clone",
        &["/a.txt", "/.gitignore", "/other/"],
    );
    std::fs::create_dir_all(clone.join("sub")).unwrap();
    std::fs::write(clone.join("sub").join("secret.txt"), secret).unwrap();
    std::fs::write(clone.join("sub").join("visible.txt"), b"visible line\n").unwrap();
    std::fs::write(clone.join("other").join("x.tmp"), secret).unwrap();
    clone
}

/// WP8b: the runner names the folder whose `.gitignore` it cannot read, as
/// the folder's own derived path, the whole folder when it is attached
/// below it, and nothing for a sibling or a plain repository. Plain git's
/// own answer is the gap: `check-ignore` says the secret is not ignored.
/// Nothing is fetched (`GIT_TRACE`).
/// Mutant: a skip-worktree `.gitignore` counts as readable without its blob.
#[test]
fn wp8b_the_runner_names_the_folders_whose_rules_it_cannot_read() {
    let scratch = Scratch::new("git-wp8b");
    let clone = unreadable_gitignore_clone(&scratch, "x");
    let env = scratch.env();
    let mut runner = scratch.runner();
    let trace = scratch.path().join("wp8b.trace");
    runner.trace = Some(trace.clone());
    let top = local(&clone, &env);
    assert_eq!(
        runner.check_ignore(&top, "sub/secret.txt"),
        Ok(false),
        "the gap: with no lazy fetch, git says the secret is not ignored"
    );
    assert_eq!(runner.check_ignore(&top, "other/x.tmp"), Ok(true));
    assert_eq!(runner.ignore_rules_incomplete(&top).unwrap(), ["sub"]);
    let below = local(&clone.join("sub"), &env);
    assert_eq!(runner.ignore_rules_incomplete(&below).unwrap(), [""]);
    let sibling = local(&clone.join("other"), &env);
    assert_eq!(
        runner.ignore_rules_incomplete(&sibling).unwrap(),
        Vec::<String>::new()
    );
    assert!(
        trace.exists(),
        "the trace was written, so it would show a fetch"
    );
    assert_eq!(
        fetch_lines(&trace),
        Vec::<String>::new(),
        "the missing blob is never fetched"
    );
    let plain = scratch.repo("plain");
    assert_eq!(
        runner
            .ignore_rules_incomplete(&local(&plain, &env))
            .unwrap(),
        Vec::<String>::new()
    );
}

#[test]
fn wp8b_index_records_and_folders_are_read_exactly() {
    use super::runner::{IndexEntry, below_prefix};
    let entry =
        IndexEntry::parse(b"S 100644 2d9d0f47938bca4cb66a74a99e92311f4e37a2db 0\tsub/.gitignore")
            .unwrap();
    assert!(entry.skip_worktree);
    assert_eq!(entry.mode, "100644");
    assert_eq!(entry.path, "sub/.gitignore");
    let tracked =
        IndexEntry::parse(b"H 100644 397b4a7624e35fa60563a9c03b1213d93f7b6546 0\t.gitignore")
            .unwrap();
    assert!(!tracked.skip_worktree);
    assert_eq!(tracked.path, ".gitignore");
    assert!(IndexEntry::parse(b"").is_none());
    assert!(IndexEntry::parse(b"S 100644 zz 0\tx").is_none());
    let folders = |list: &[&str]| list.iter().map(|f| (*f).to_owned()).collect::<Vec<_>>();
    assert_eq!(below_prefix("", folders(&["b", "a/c", "a"])), ["a", "b"]);
    assert_eq!(below_prefix("", folders(&["", "a"])), [""]);
    assert_eq!(below_prefix("w", folders(&["w/x", "y", "w2"])), ["x"]);
    assert_eq!(below_prefix("w/v", folders(&["w"])), [""]);
    assert_eq!(below_prefix("w", folders(&[""])), [""]);
    assert_eq!(below_prefix("w", folders(&["ww"])), Vec::<String>::new());
}

fn fetch_lines(trace: &Path) -> Vec<String> {
    std::fs::read_to_string(trace)
        .unwrap_or_default()
        .lines()
        .filter(|line| line.contains(" fetch "))
        .map(str::to_owned)
        .collect()
}

/// KF8, the parts for `check-ignore` and `ls-files`: in a partial clone that
/// allows the `file` protocol itself, `check-ignore` of a path whose
/// `.gitignore` blob is missing starts a lazy `git fetch` with plain git (the
/// positive control) and none through the runner (`GIT_TRACE`).
/// Mutant: `GIT_NO_LAZY_FETCH` and `GIT_ALLOW_PROTOCOL` dropped.
#[test]
fn kf8_a_partial_clone_starts_no_fetch() {
    let scratch = Scratch::new("git-kf8");
    let source = scratch.repo("source");
    std::fs::create_dir_all(source.join("sub")).unwrap();
    std::fs::write(source.join("sub").join(".gitignore"), b"*.tmp\n").unwrap();
    std::fs::write(source.join("sub").join("b.txt"), b"b\n").unwrap();
    scratch.git(&source, &["add", "."]);
    scratch.git(&source, &["commit", "-q", "-m", "two"]);
    scratch.git(&source, &["config", "uploadpack.allowFilter", "true"]);

    let control = partial_clone(&scratch, &source, "control");
    let control_trace = scratch.path().join("control.trace");
    Command::new("git")
        .args([
            "-c",
            "core.fsmonitor=false",
            "check-ignore",
            "-q",
            "--",
            "sub/x.tmp",
        ])
        .current_dir(&control)
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", scratch.home().join(".gitconfig"))
        .env("GIT_TRACE", &control_trace)
        .output()
        .unwrap();
    let control_fetches = fetch_lines(&control_trace);
    println!("plain git: {} fetch line(s)", control_fetches.len());
    assert!(
        !control_fetches.is_empty(),
        "the positive control: plain git fetches the missing blob"
    );

    let clone = partial_clone(&scratch, &source, "clone");
    let env = scratch.env();
    let local = local(&clone, &env);
    let mut runner = scratch.runner();
    let trace = scratch.path().join("runner.trace");
    runner.trace = Some(trace.clone());
    let ignored = runner.check_ignore(&local, "sub/x.tmp");
    let listing = runner.ls_files(&local).unwrap();
    println!(
        "through the runner: check-ignore {ignored:?}, {} listed",
        listing.paths.len()
    );
    assert!(
        trace.exists(),
        "the trace was written, so it would show a fetch"
    );
    assert_eq!(fetch_lines(&trace), Vec::<String>::new());
}

// ------------------------------------------------------------------- FT6

/// Run `inspect` and return what it decided and every path it opened.
fn inspect(folder: &Path, env: &MapEnv) -> (Repo, Vec<PathBuf>) {
    let _ = localfs::record::take();
    let repo = dotgit::inspect(folder, env);
    (repo, localfs::record::take())
}

fn assert_never_opened_a_network_path(opened: &[PathBuf]) {
    for path in opened {
        assert!(is_local_text(path), "opened {}", path.display());
    }
}

fn assert_not_local(repo: &Repo, named: Named) {
    match repo {
        Repo::Without(WithoutGit::NotLocal { named: got, path }) => {
            assert_eq!(*got, named, "{path}");
            assert!(!is_local_text(Path::new(path)), "{path}");
        }
        other => panic!("{named:?}: {other:?}"),
    }
}

const UNC: &str = r"\\198.51.100.7\x";

/// `path` as a git configuration value writes it (each backslash escaped).
fn escaped(path: &str) -> String {
    path.replace('\\', "\\\\")
}

#[test]
fn ft6_a_plain_repository_and_a_linked_worktree_are_local() {
    let scratch = Scratch::new("ft6-plain");
    let repo = scratch.repo("r");
    let env = scratch.env();
    let (found, opened) = inspect(&repo, &env);
    assert_never_opened_a_network_path(&opened);
    let Repo::Git(top) = found else {
        panic!("{found:?}")
    };
    assert!(!top.is_gitfile());
    assert!(top.git_dir().ends_with(".git"));
    assert_eq!(top.git_dir(), top.common_dir());
    std::fs::create_dir_all(repo.join("deep").join("er")).unwrap();
    let below = local(&repo.join("deep").join("er"), &env);
    assert_eq!(below.git_dir(), top.git_dir());
    assert!(below.folder().ends_with(r"deep\er"));

    let worktree = scratch.path().join("wt");
    scratch.git(
        &repo,
        &["worktree", "add", "-q", &worktree.to_string_lossy()],
    );
    let linked = local(&worktree, &env);
    assert!(linked.is_gitfile());
    assert_eq!(linked.common_dir(), top.git_dir());
    assert_ne!(linked.git_dir(), linked.common_dir());
}

#[test]
fn ft6_a_folder_in_no_repository_is_none() {
    let scratch = Scratch::new("ft6-none");
    let folder = scratch.path().join("plain");
    std::fs::create_dir_all(&folder).unwrap();
    let (found, _) = inspect(&folder, &scratch.env());
    assert_eq!(found, Repo::None);
}

/// A folder whose `.git` is a gitfile with `text` in it.
fn gitfile(scratch: &Scratch, name: &str, text: &str) -> PathBuf {
    let folder = scratch.path().join(name);
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join(".git"), text).unwrap();
    folder
}

/// FT6: a gitfile naming a network or device path makes a folder without
/// git, and nothing opens that path.
/// Mutant: the gitfile is not read (its folder taken as the git folder).
#[test]
fn ft6_a_gitfile_to_the_network_is_without_git() {
    let scratch = Scratch::new("ft6-gitfile");
    let env = scratch.env();
    for (at, target) in [
        UNC.to_owned(),
        "//198.51.100.7/x".to_owned(),
        r"\\?\UNC\198.51.100.7\x".to_owned(),
        r"\\.\pipe\lattice".to_owned(),
    ]
    .iter()
    .enumerate()
    {
        let folder = gitfile(&scratch, &format!("g{at}"), &format!("gitdir: {target}\n"));
        let (found, opened) = inspect(&folder, &env);
        assert_not_local(&found, Named::Gitfile);
        assert_never_opened_a_network_path(&opened);
    }
    // HOME on a share and a gitfile under it.
    let env_unc = scratch.env().with("HOME", UNC);
    let folder = gitfile(&scratch, "home", "gitdir: ~/repo/.git\n");
    let (found, opened) = inspect(&folder, &env_unc);
    assert_not_local(&found, Named::Gitfile);
    assert_never_opened_a_network_path(&opened);
    // A gitfile git would not read.
    let folder = gitfile(&scratch, "bad", "not a gitfile\n");
    assert!(matches!(
        dotgit::inspect(&folder, &env),
        Repo::Without(WithoutGit::Unparsable { .. })
    ));
}

/// FT6: `commondir` naming the network.
/// Mutant: `commondir` not read.
#[test]
fn ft6_a_commondir_on_the_network_is_without_git() {
    let scratch = Scratch::new("ft6-commondir");
    let repo = scratch.repo("r");
    std::fs::write(repo.join(".git").join("commondir"), format!("{UNC}\n")).unwrap();
    let (found, opened) = inspect(&repo, &scratch.env());
    assert_not_local(&found, Named::CommonDir);
    assert_never_opened_a_network_path(&opened);
}

/// FT6: an alternate object store on the network, directly, quoted, or as
/// an alternate of a local alternate.
/// Mutant: alternates not read.
#[test]
fn ft6_an_alternate_on_the_network_is_without_git() {
    let scratch = Scratch::new("ft6-alternates");
    let env = scratch.env();
    let info = |repo: &Path| repo.join(".git").join("objects").join("info");
    let direct = scratch.repo("direct");
    std::fs::create_dir_all(info(&direct)).unwrap();
    std::fs::write(
        info(&direct).join("alternates"),
        format!("# c\n\n{UNC}\\objects\n"),
    )
    .unwrap();
    let (found, opened) = inspect(&direct, &env);
    assert_not_local(&found, Named::Alternate);
    assert_never_opened_a_network_path(&opened);

    let quoted = scratch.repo("quoted");
    std::fs::create_dir_all(info(&quoted)).unwrap();
    std::fs::write(
        info(&quoted).join("alternates"),
        "\"\\\\\\\\198.51.100.7\\\\x\\\\objects\"\n",
    )
    .unwrap();
    let (found, opened) = inspect(&quoted, &env);
    assert_not_local(&found, Named::Alternate);
    assert_never_opened_a_network_path(&opened);

    let nested = scratch.repo("nested");
    let middle = scratch.repo("middle");
    std::fs::create_dir_all(info(&nested)).unwrap();
    std::fs::write(
        info(&nested).join("alternates"),
        format!("{}\n", middle.join(".git").join("objects").display()),
    )
    .unwrap();
    std::fs::create_dir_all(info(&middle)).unwrap();
    std::fs::write(
        info(&middle).join("alternates"),
        format!("{UNC}\\objects\n"),
    )
    .unwrap();
    let (found, opened) = inspect(&nested, &env);
    assert_not_local(&found, Named::Alternate);
    assert_never_opened_a_network_path(&opened);

    // A local alternate, and one that does not exist, are fine.
    let clean = scratch.repo("clean");
    let fine = scratch.repo("fine");
    std::fs::create_dir_all(info(&fine)).unwrap();
    std::fs::write(
        info(&fine).join("alternates"),
        format!(
            "{}\n{}\n",
            clean.join(".git").join("objects").display(),
            scratch.path().join("missing").display()
        ),
    )
    .unwrap();
    assert!(matches!(dotgit::inspect(&fine, &env), Repo::Git(_)));
}

/// FT6: configuration includes on the network (`include.path`,
/// `includeIf.*.path` whatever its condition, `~/` with a home on a share,
/// and a local include that includes the network).
/// Mutant: includes not followed.
#[test]
fn ft6_an_include_on_the_network_is_without_git() {
    let scratch = Scratch::new("ft6-include");
    let env = scratch.env();
    let cases = [
        format!("[include]\n\tpath = {}\\\\inc\n", escaped(UNC)),
        format!(
            "[includeIf \"onbranch:never\"]\n\tpath = {}\\\\inc\n",
            escaped(UNC)
        ),
        "[include]\n\tpath = //198.51.100.7/x/inc\n".to_owned(),
    ];
    for (at, text) in cases.iter().enumerate() {
        let repo = scratch.repo(&format!("i{at}"));
        let config = repo.join(".git").join("config");
        let mut body = std::fs::read_to_string(&config).unwrap();
        body.push_str(text);
        std::fs::write(&config, body).unwrap();
        let (found, opened) = inspect(&repo, &env);
        assert_not_local(&found, Named::Include);
        assert_never_opened_a_network_path(&opened);
    }
    let repo = scratch.repo("chain");
    let config = repo.join(".git").join("config");
    let mut body = std::fs::read_to_string(&config).unwrap();
    body.push_str("[include]\n\tpath = local.inc\n");
    std::fs::write(&config, body).unwrap();
    std::fs::write(
        repo.join(".git").join("local.inc"),
        format!("[include]\n\tpath = {}\\\\deeper\n", escaped(UNC)),
    )
    .unwrap();
    let (found, opened) = inspect(&repo, &env);
    assert_not_local(&found, Named::Include);
    assert_never_opened_a_network_path(&opened);

    let repo = scratch.repo("home");
    let config = repo.join(".git").join("config");
    let mut body = std::fs::read_to_string(&config).unwrap();
    body.push_str("[include]\n\tpath = ~/inc\n");
    std::fs::write(&config, body).unwrap();
    let (found, opened) = inspect(&repo, &scratch.env().with("HOME", UNC));
    assert_not_local(&found, Named::Include);
    assert_never_opened_a_network_path(&opened);
    // The same include with a local home, and a missing one, are fine.
    assert!(matches!(dotgit::inspect(&repo, &env), Repo::Git(_)));
}

/// FT6: `core.excludesFile`, `core.attributesFile` and `core.worktree` on
/// the network.
/// Mutant: those settings not checked.
#[test]
fn ft6_an_excludes_attributes_or_worktree_setting_on_the_network_is_without_git() {
    let scratch = Scratch::new("ft6-settings");
    let env = scratch.env();
    for (at, (key, named)) in [
        ("excludesFile", Named::ExcludesFile),
        ("attributesFile", Named::AttributesFile),
        ("worktree", Named::WorkTree),
    ]
    .into_iter()
    .enumerate()
    {
        let repo = scratch.repo(&format!("s{at}"));
        let config = repo.join(".git").join("config");
        let mut body = std::fs::read_to_string(&config).unwrap();
        body.push_str(&format!("[core]\n\t{key} = {}\\\\file\n", escaped(UNC)));
        std::fs::write(&config, body).unwrap();
        let (found, opened) = inspect(&repo, &env);
        assert_not_local(&found, named);
        assert_never_opened_a_network_path(&opened);
        // In config.worktree as well.
        let repo = scratch.repo(&format!("w{at}"));
        std::fs::write(
            repo.join(".git").join("config.worktree"),
            format!("[core]\n\t{key} = //198.51.100.7/x/file\n"),
        )
        .unwrap();
        let (found, _) = inspect(&repo, &env);
        assert_not_local(&found, named);
    }
    let repo = scratch.repo("fine");
    let config = repo.join(".git").join("config");
    let mut body = std::fs::read_to_string(&config).unwrap();
    body.push_str("[core]\n\texcludesFile = ~/.ignore\n\tattributesFile = ../attrs\n");
    std::fs::write(&config, body).unwrap();
    assert!(matches!(dotgit::inspect(&repo, &env), Repo::Git(_)));
}

/// FT6: what cannot be read as git reads it fails closed, and so does a
/// bare repository's layout.
/// Mutant: a parse error taken as an empty configuration.
#[test]
fn ft6_what_cannot_be_read_and_a_bare_layout_are_without_git() {
    let scratch = Scratch::new("ft6-closed");
    let env = scratch.env();
    let repo = scratch.repo("bad");
    let config = repo.join(".git").join("config");
    let mut body = std::fs::read_to_string(&config).unwrap();
    body.push_str("[core\n");
    std::fs::write(&config, body).unwrap();
    assert!(matches!(
        dotgit::inspect(&repo, &env),
        Repo::Without(WithoutGit::Unparsable {
            named: Named::Config,
            ..
        })
    ));
    for value in ["%(prefix)/etc/x", "~other/x"] {
        let repo = scratch.repo(&format!("form{}", value.len()));
        let config = repo.join(".git").join("config");
        let mut body = std::fs::read_to_string(&config).unwrap();
        body.push_str(&format!("[include]\n\tpath = {value}\n"));
        std::fs::write(&config, body).unwrap();
        assert!(
            matches!(
                dotgit::inspect(&repo, &env),
                Repo::Without(WithoutGit::Unparsable { .. })
            ),
            "{value}"
        );
    }
    let bare = scratch.path().join("bare.git");
    std::fs::create_dir_all(&bare).unwrap();
    scratch.git(&bare, &["init", "-q", "--bare"]);
    std::fs::create_dir_all(bare.join("hooks")).unwrap();
    assert!(matches!(
        dotgit::inspect(&bare.join("hooks"), &env),
        Repo::Without(WithoutGit::Bare { .. })
    ));
}

/// FT6 with a link at `.git`: to a local folder it is followed; to the
/// network it is refused before it is followed. Making a symbolic link may
/// need Developer Mode; when it cannot be made, that case says so and is
/// UNMEASURED.
#[test]
fn ft6_a_link_at_dot_git() {
    let scratch = Scratch::new("ft6-link");
    let env = scratch.env();
    let repo = scratch.repo("real");
    let linked = scratch.path().join("linked");
    std::fs::create_dir_all(linked.join(".git")).unwrap();
    lattice_sys::fs::seam::create_junction(&linked.join(".git"), &repo.join(".git")).unwrap();
    let found = local(&linked, &env);
    assert!(found.git_dir().ends_with(r"real\.git"));

    let network = scratch.path().join("network");
    std::fs::create_dir_all(&network).unwrap();
    match std::os::windows::fs::symlink_dir(UNC, network.join(".git")) {
        Ok(()) => {
            let (found, opened) = inspect(&network, &env);
            assert_not_local(&found, Named::DotGit);
            assert_never_opened_a_network_path(&opened);
        }
        Err(error) => println!(
            "UNMEASURED: a symbolic link to {UNC} could not be made here ({error}); the junction case ran"
        ),
    }
}
