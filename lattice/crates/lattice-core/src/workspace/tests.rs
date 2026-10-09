//! Attach, identity and the path rules against folders and scratch
//! repositories this file makes (the chat core's spec §6.1, §6.2, §4.1;
//! §16.4 WF4–WF6, SF9; §10.3 TF15's attach part).
//!
//! Everything is made in temporary folders; git runs only in scratch
//! repositories (`git::tests::Scratch`). Network paths are the documentation
//! address `\\198.51.100.7\x`; the falsifiers read `localfs::record`, which
//! sees every open the walk makes, to show none reached it. Symbolic links
//! need Developer Mode; a case that cannot make one says UNMEASURED and the
//! rest of its test still runs.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::attach::{AttachRefusal, AttachSource, attach_path, confirmed_path};
use super::paths::{
    PathError, RepoPathError, Want, is_authority, is_device_name, is_short_name_alias, repo_path,
};
use super::{Workspace, WorkspaceKey};
use crate::env::MapEnv;
use crate::git::dotgit::Repo;
use crate::git::runner::GitRunner;
use crate::git::tests::Scratch;
use crate::localfs::{self, is_local_text};
use crate::policy::{
    Gates, Lease, PathClass, PathRefusal, Reason, Standing, Target, ToolClass, Verdict, decide,
};
use crate::ports::ConfirmRequest;
use crate::ports::fake::RecordingConfirm;
use crate::state::StateRoot;

const UNC: &str = r"\\198.51.100.7\x";

fn state(scratch: &Scratch) -> StateRoot {
    StateRoot::at(scratch.path().join("state"))
}

fn attach(scratch: &Scratch, folder: &Path) -> Workspace {
    attach_path(folder, &scratch.env(), &state(scratch), &scratch.runner())
        .unwrap_or_else(|refusal| panic!("{}: {refusal:?}", folder.display()))
}

fn refused(scratch: &Scratch, env: &MapEnv, folder: &Path) -> AttachRefusal {
    match attach_path(folder, env, &state(scratch), &scratch.runner()) {
        Ok(workspace) => panic!("{} attached as {:?}", folder.display(), workspace.root),
        Err(refusal) => refusal,
    }
}

/// The derived path and class, or the error.
fn resolve(
    workspace: &Workspace,
    runner: &GitRunner,
    request: &str,
    want: Want,
) -> Result<(String, PathClass), PathError> {
    workspace
        .with_rules(runner, |rules| {
            rules
                .resolve(request, want)
                .map(|resolved| (resolved.derived, resolved.class))
        })
        .unwrap()
}

fn reason(result: Result<(String, PathClass), PathError>) -> PathRefusal {
    match result {
        Err(PathError::Refused(refused)) => refused.reason,
        other => panic!("not refused: {other:?}"),
    }
}

fn never_opened_the_network(opened: &[PathBuf]) {
    for path in opened {
        assert!(is_local_text(path), "opened {}", path.display());
    }
}

fn symlink_file(target: &Path, link: &Path) -> bool {
    match std::os::windows::fs::symlink_file(target, link) {
        Ok(()) => true,
        Err(error) => {
            println!(
                "UNMEASURED: a symbolic link {} -> {} could not be made ({error})",
                link.display(),
                target.display()
            );
            false
        }
    }
}

fn symlink_dir(target: &Path, link: &Path) -> bool {
    match std::os::windows::fs::symlink_dir(target, link) {
        Ok(()) => true,
        Err(error) => {
            println!(
                "UNMEASURED: a symbolic link {} -> {} could not be made ({error})",
                link.display(),
                target.display()
            );
            false
        }
    }
}

fn junction(link: &Path, target: &Path) {
    std::fs::create_dir(link).unwrap();
    lattice_sys::fs::seam::create_junction(link, target).unwrap();
}

// ------------------------------------------------------------- the lexical rules

#[test]
fn repo_path_follows_the_webs_rules() {
    assert_eq!(repo_path("src/lib.rs"), Ok("src/lib.rs"));
    assert_eq!(repo_path(""), Err(RepoPathError::Required));
    assert_eq!(repo_path(&"a".repeat(1025)), Err(RepoPathError::TooLong));
    assert!(repo_path(&"\u{e9}".repeat(1024)).is_ok());
    for value in ["/a", "a\\b", "a:b", "a*b", "a\u{1}b"] {
        assert_eq!(
            repo_path(value),
            Err(RepoPathError::NotRelative),
            "{value:?}"
        );
    }
    for value in ["a//b", "./a", "a/..", "."] {
        assert_eq!(
            repo_path(value),
            Err(RepoPathError::BadComponent),
            "{value:?}"
        );
    }
    for value in ["a.", "a ", "a. /b"] {
        assert_eq!(
            repo_path(value),
            Err(RepoPathError::TrailingDotOrSpace),
            "{value:?}"
        );
    }
    for value in [".git", ".GIT", "a/.Git/b"] {
        assert_eq!(repo_path(value), Err(RepoPathError::GitDir), "{value:?}");
    }
}

/// WP7.
/// Mutant: the superscript digits left out.
#[test]
fn wp7_device_names_are_refused_with_or_without_an_extension() {
    for name in [
        "CON",
        "con",
        "Con.txt",
        "PRN",
        "AUX.c",
        "NUL",
        "nul.tar.gz",
        "NUL .txt",
        "COM0",
        "COM1",
        "com9.log",
        "LPT1",
        "lpt9",
        "COM\u{b9}",
        "COM\u{b2}.x",
        "LPT\u{b3}",
        "CONIN$",
        "conout$.x",
    ] {
        assert!(is_device_name(name), "{name}");
    }
    for name in [
        "CONSOLE",
        "COM",
        "COM10",
        "LPT",
        "nulx",
        "xnul",
        "AUXILIARY",
        "con-fig",
        "COM\u{b4}",
    ] {
        assert!(!is_device_name(name), "{name}");
    }
}

/// WP11.
/// Mutant: WP11 dropped (the resolve test below then answers Ignored, not
/// ShortName).
#[test]
fn wp11_an_8_3_alias_is_refused() {
    for name in ["FAMENV~1.ENV", "ENV~1", "GITHUB~1", "a~0b", "x~9"] {
        assert!(is_short_name_alias(name), "{name}");
    }
    for name in ["~", "a~b", "x~", "~a1", "tilde~.txt"] {
        assert!(!is_short_name_alias(name), "{name}");
    }
}

/// ST4: the reviewed authority list, at any depth and without regard to
/// case, and what it leaves out.
/// Mutant: an extension compared with case.
#[test]
fn st4_the_authority_list() {
    for path in [
        "AGENTS.md",
        "sub/claude.md",
        ".lattice/rules/x.md",
        ".cursor/rules/a.mdc",
        ".github/workflows/ci.yml",
        ".VSCode/settings.json",
        "crate/.cargo/config.toml",
        ".gitignore",
        "sub/.gitattributes",
        ".latticeignore",
        "Cargo.toml",
        "crates/x/Cargo.lock",
        "package.json",
        "web/package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "poetry.lock",
        "go.mod",
        "go.sum",
        "rust-toolchain",
        "rust-toolchain.toml",
        ".npmrc",
        ".yarnrc.yml",
        "Directory.Build.props",
        "NuGet.Config",
        "global.json",
        "tools/run.PS1",
        "x.bat",
        "y.cmd",
        "z.sh",
        // The MCP files a trusted folder declares servers in (§12).
        ".mcp.json",
        "sub/.MCP.json",
        ".lattice/mcp.json",
        ".cursor/mcp.json",
    ] {
        assert!(is_authority(path), "{path}");
    }
    for path in [
        "src/main.rs",
        "README.md",
        "docs/agents.txt",
        "github/x.yml",
        "cargo.tom",
        "requirements.md",
        "package.json5",
        "x.shx",
        "batch/notes.txt",
        "mcp.json",
    ] {
        assert!(!is_authority(path), "{path}");
    }
}

// ------------------------------------------------------------------- attach

/// TF15, the attach part: the user's profile, `%APPDATA%` and
/// `%LOCALAPPDATA%` (and anything inside the last two) are refused.
/// Mutant: the application-data refusal dropped.
#[test]
fn tf15_the_profile_and_application_data_are_refused() {
    let scratch = Scratch::new("ws-tf15");
    let profile = scratch.home();
    let roaming = profile.join("AppData").join("Roaming");
    let local = profile.join("AppData").join("Local");
    std::fs::create_dir_all(roaming.join("Code")).unwrap();
    std::fs::create_dir_all(local.join("Programs")).unwrap();
    std::fs::create_dir_all(profile.join("projects").join("app")).unwrap();
    let env = scratch
        .env()
        .with("APPDATA", roaming.as_os_str())
        .with("LOCALAPPDATA", local.as_os_str());
    assert_eq!(refused(&scratch, &env, &profile), AttachRefusal::Profile);
    for folder in [
        &roaming,
        &roaming.join("Code"),
        &local,
        &local.join("Programs"),
    ] {
        assert_eq!(
            refused(&scratch, &env, folder),
            AttachRefusal::AppData,
            "{}",
            folder.display()
        );
    }
    let project = attach_path(
        &profile.join("projects").join("app"),
        &env,
        &state(&scratch),
        &scratch.runner(),
    )
    .unwrap();
    assert_eq!(project.name, "app");
    assert_eq!(project.repo, Repo::None);
}

/// TF15, the attach part: a path from the page attaches only after the
/// reader's native confirmation; declined, nothing is attached. A page path
/// that would be refused anyway is refused without a dialog.
/// Mutant: a page path attached without asking.
#[test]
fn tf15_a_page_path_needs_the_readers_native_confirmation() {
    let scratch = Scratch::new("ws-page");
    let folder = scratch.path().join("proj");
    std::fs::create_dir_all(&folder).unwrap();
    let text = folder.to_string_lossy().into_owned();

    let declined = RecordingConfirm::answering(false);
    let outcome =
        futures::executor::block_on(confirmed_path(AttachSource::Page(text.clone()), &declined));
    assert_eq!(outcome, Err(AttachRefusal::NotConfirmed));
    assert_eq!(
        declined.asked(),
        [ConfirmRequest::AttachFolder { path: text.clone() }]
    );

    let accepted = RecordingConfirm::answering(true);
    let path =
        futures::executor::block_on(confirmed_path(AttachSource::Page(text.clone()), &accepted))
            .unwrap();
    assert_eq!(attach(&scratch, &path).name, "proj");

    let network = RecordingConfirm::answering(true);
    let outcome = futures::executor::block_on(confirmed_path(
        AttachSource::Page(format!(r"{UNC}\proj")),
        &network,
    ));
    assert_eq!(outcome, Err(AttachRefusal::Network));
    assert!(network.asked().is_empty(), "refused without a dialog");

    let native = RecordingConfirm::answering(false);
    let path = futures::executor::block_on(confirmed_path(
        AttachSource::Native(folder.clone()),
        &native,
    ))
    .unwrap();
    assert_eq!(path, folder);
    assert!(native.asked().is_empty(), "a native path needs no dialog");
}

/// §6.1: a drive root, the Windows folder, network and device paths, a
/// relative path, a missing path and a file are refused; the network and
/// device paths before anything opens them. A network drive
/// (`DRIVE_REMOTE`) cannot be made here: UNMEASURED.
/// Mutant: the Windows-folder refusal dropped. (The drive-root and network
/// refusals each have a second layer, the check after the walk and the walk's
/// own text check, so dropping one of them alone changes no outcome.)
#[test]
fn attach_refuses_what_section_6_1_names() {
    let scratch = Scratch::new("ws-refusals");
    let windows = scratch.path().join("win");
    std::fs::create_dir_all(windows.join("System32")).unwrap();
    std::fs::write(scratch.path().join("file.txt"), b"x").unwrap();
    let env = scratch.env().with("SystemRoot", windows.as_os_str());
    for root in [r"C:\", r"\\?\C:\", r"C:\.", r"C:\x\.."] {
        assert_eq!(
            refused(&scratch, &env, Path::new(root)),
            AttachRefusal::DriveRoot,
            "{root}"
        );
    }
    assert_eq!(
        refused(&scratch, &env, &windows),
        AttachRefusal::SystemFolder
    );
    assert_eq!(
        refused(&scratch, &env, &windows.join("System32")),
        AttachRefusal::SystemFolder
    );
    let _ = localfs::record::take();
    for (path, expected) in [
        (UNC.to_owned(), AttachRefusal::Network),
        ("//198.51.100.7/x".to_owned(), AttachRefusal::Network),
        (r"\\?\UNC\198.51.100.7\x".to_owned(), AttachRefusal::Network),
        (r"\\.\pipe\lattice".to_owned(), AttachRefusal::Device),
        ("relative\\folder".to_owned(), AttachRefusal::NotAbsolute),
        (r"\rooted".to_owned(), AttachRefusal::NotAbsolute),
    ] {
        assert_eq!(
            refused(&scratch, &env, Path::new(&path)),
            expected,
            "{path}"
        );
    }
    assert_eq!(
        localfs::record::take(),
        Vec::<PathBuf>::new(),
        "nothing was opened"
    );
    assert_eq!(
        refused(&scratch, &env, &scratch.path().join("missing")),
        AttachRefusal::Missing
    );
    assert_eq!(
        refused(&scratch, &env, &scratch.path().join("file.txt")),
        AttachRefusal::NotAFolder
    );
    // A folder that is a link to the network: refused before it is followed.
    let link = scratch.path().join("share");
    if symlink_dir(Path::new(UNC), &link) {
        let _ = localfs::record::take();
        assert_eq!(refused(&scratch, &env, &link), AttachRefusal::Network);
        never_opened_the_network(&localfs::record::take());
    }
}

/// §6.1: `<globals>`, anything in it, the Python state home `~/.alelyon`,
/// and any other ancestor of `<globals>` are refused, except a git checkout's
/// top level whose own `globals/` is `<globals>`.
/// Mutant: the ancestor refusal dropped.
#[test]
fn attach_refuses_lattices_state_and_its_ancestors_but_not_the_checkout() {
    let scratch = Scratch::new("ws-state");
    let env = scratch.env();
    let st = state(&scratch);
    std::fs::create_dir_all(st.globals.join("lattice_native")).unwrap();
    std::fs::create_dir_all(scratch.home().join(".alelyon").join("llama")).unwrap();
    assert_eq!(
        refused(&scratch, &env, &st.globals),
        AttachRefusal::LatticeState
    );
    assert_eq!(
        refused(&scratch, &env, &st.globals.join("lattice_native")),
        AttachRefusal::LatticeState
    );
    assert_eq!(
        refused(
            &scratch,
            &env,
            &scratch.home().join(".alelyon").join("llama")
        ),
        AttachRefusal::LatticeState
    );
    // `state/` holds `globals/` and is no checkout.
    assert_eq!(
        refused(&scratch, &env, &st.root),
        AttachRefusal::AboveLatticeState
    );
    assert_eq!(
        refused(&scratch, &env, scratch.path()),
        AttachRefusal::AboveLatticeState
    );
    // The source-checkout layout: the same folder as a git checkout's top level.
    scratch.git(&st.root, &["init", "-q"]);
    let checkout = attach(&scratch, &st.root);
    assert!(matches!(checkout.repo, Repo::Git(_)));
    assert!(checkout.top_level.is_some());
    // WP9 then refuses `<globals>` path by path.
    let runner = scratch.runner();
    std::fs::write(st.globals.join("x.json"), b"{}").unwrap();
    assert_eq!(
        reason(resolve(
            &checkout,
            &runner,
            "globals/x.json",
            Want::Existing
        )),
        PathRefusal::LatticeState
    );
    assert_eq!(
        reason(resolve(
            &checkout,
            &runner,
            "globals/new.json",
            Want::MayCreate
        )),
        PathRefusal::LatticeState
    );
}

/// §4.1 and WF5's identity half: the id is the folder's file id, so it
/// survives a rename, and a record matches only when the id and the path
/// both do. A renamed folder, and a new folder given the old path, are
/// both asked about again.
/// Mutant: matching by the id only.
#[test]
fn wf5_identity_and_path_both_must_match() {
    let scratch = Scratch::new("ws-identity");
    let old = scratch.path().join("old");
    std::fs::create_dir_all(&old).unwrap();
    let first = attach(&scratch, &old);
    assert_eq!(first.id.len(), 16);
    assert!(lattice_protocol::conversation::is_workspace_id(&first.id));
    assert!(first.key().matches(&attach(&scratch, &old).key()));
    let new = scratch.path().join("new");
    std::fs::rename(&old, &new).unwrap();
    let moved = attach(&scratch, &new);
    assert_eq!(
        moved.id, first.id,
        "a move within the volume keeps the file id"
    );
    assert!(!first.key().matches(&moved.key()), "moved: asked again");
    std::fs::create_dir_all(&old).unwrap();
    let replaced = attach(&scratch, &old);
    assert_ne!(replaced.id, first.id);
    assert!(
        !first.key().matches(&replaced.key()),
        "replaced: asked again"
    );
    let same = WorkspaceKey {
        id: first.id.clone(),
        path: first.root.to_string_lossy().to_uppercase(),
    };
    assert!(first.key().matches(&same), "paths compare without case");
}

// ---------------------------------------------------------------- the rules

/// A repository that ignores `FAMEnvironment.env`, `.env`, `*.log` and
/// `/private/`, with a workflow under `.github`.
fn ignoring_repo(scratch: &Scratch) -> PathBuf {
    let repo = scratch.path().join("repo");
    std::fs::create_dir_all(repo.join(".github").join("workflows")).unwrap();
    std::fs::create_dir_all(repo.join("private")).unwrap();
    std::fs::create_dir_all(repo.join("sub")).unwrap();
    scratch.git(&repo, &["init", "-q"]);
    std::fs::write(
        repo.join(".gitignore"),
        b"FAMEnvironment.env\n.env\n*.log\n/private/\n",
    )
    .unwrap();
    std::fs::write(repo.join("FAMEnvironment.env"), b"KEY=1\n").unwrap();
    std::fs::write(repo.join(".env"), b"KEY=2\n").unwrap();
    std::fs::write(repo.join("private").join("data.txt"), b"p\n").unwrap();
    std::fs::write(
        repo.join(".github").join("workflows").join("ci.yml"),
        b"on: push\n",
    )
    .unwrap();
    std::fs::write(repo.join("sub").join("File.txt"), b"f\n").unwrap();
    std::fs::write(repo.join("readme.txt"), b"r\n").unwrap();
    repo
}

/// WF6: in a repository that ignores `FAMEnvironment.env` and `.env`, the
/// 8.3 aliases, an in-root link to `.env` and an in-root junction to an
/// ignored folder are refused: the ignore check runs on the derived path.
/// `GITHUB~1/…` is refused by WP11; the same file through an in-root
/// junction to `.github` is Authority, and Keep All leaves it out. A link to
/// the network is refused with no network open.
/// Mutant: the ignore check (and the class) on the request string.
#[test]
fn wf6_the_derived_path_decides_ignore_and_authority() {
    let scratch = Scratch::new("ws-wf6");
    let repo = ignoring_repo(&scratch);
    junction(&repo.join("pub"), &repo.join("private"));
    junction(&repo.join("gh"), &repo.join(".github"));
    let made_env_link = symlink_file(&repo.join(".env"), &repo.join("envlink"));
    let made_share = symlink_file(&PathBuf::from(format!(r"{UNC}\f")), &repo.join("share"));
    let workspace = attach(&scratch, &repo);
    assert!(matches!(workspace.repo, Repo::Git(_)));
    let runner = scratch.runner();
    let read = |request: &str| resolve(&workspace, &runner, request, Want::Existing);

    assert_eq!(
        read("readme.txt").unwrap(),
        ("readme.txt".into(), PathClass::Normal)
    );
    assert_eq!(reason(read("FAMEnvironment.env")), PathRefusal::Ignored);
    assert_eq!(reason(read("FAMENV~1.ENV")), PathRefusal::ShortName);
    assert_eq!(reason(read("ENV~1")), PathRefusal::ShortName);
    assert_eq!(
        reason(read("GITHUB~1/workflows/ci.yml")),
        PathRefusal::ShortName
    );
    assert_eq!(
        reason(read("pub/data.txt")),
        PathRefusal::Ignored,
        "the junction's target is ignored"
    );
    if made_env_link {
        assert_eq!(
            reason(read("envlink")),
            PathRefusal::Ignored,
            "the link's target is .env"
        );
    }
    let (derived, class) = read("gh/workflows/ci.yml").unwrap();
    assert_eq!(derived, ".github/workflows/ci.yml");
    assert_eq!(class, PathClass::Authority);
    let keep_all = decide(
        lattice_protocol::conversation::Mode::Agent,
        true,
        ToolClass::KeepAll,
        &Target::Path(class),
        &Gates {
            staged_waiting: 0,
            command_running: false,
            lease: Lease::Held,
        },
        &Standing::default(),
    );
    assert_eq!(keep_all, Verdict::Refuse(Reason::AuthorityKeptAlone));
    if made_share {
        let _ = localfs::record::take();
        assert_eq!(reason(read("share")), PathRefusal::RemoteLink);
        never_opened_the_network(&localfs::record::take());
    }
    // The real spelling, whatever the request's case.
    assert_eq!(read("SUB/FILE.TXT").unwrap().0, "sub/File.txt");
}

/// WF6 without git: the built-in defaults on the derived path.
#[test]
fn wf6_without_git_the_defaults_hold_on_the_derived_path() {
    let scratch = Scratch::new("ws-wf6-plain");
    let folder = scratch.path().join("plain");
    std::fs::create_dir_all(folder.join("keys")).unwrap();
    std::fs::write(folder.join("FAMEnvironment.env"), b"KEY=1\n").unwrap();
    std::fs::write(folder.join(".env"), b"KEY=2\n").unwrap();
    std::fs::write(folder.join("keys").join("id_rsa"), b"k\n").unwrap();
    std::fs::write(folder.join(".latticeignore"), b"build/\n").unwrap();
    std::fs::create_dir_all(folder.join("build")).unwrap();
    std::fs::write(folder.join("build").join("out.txt"), b"o\n").unwrap();
    junction(&folder.join("vault"), &folder.join("keys"));
    let made_link = symlink_file(&folder.join(".env"), &folder.join("envlink"));
    let workspace = attach(&scratch, &folder);
    assert_eq!(workspace.repo, Repo::None);
    let runner = scratch.runner();
    let read = |request: &str| resolve(&workspace, &runner, request, Want::Existing);
    assert_eq!(
        reason(read("FAMEnvironment.env")),
        PathRefusal::SecretDefault
    );
    assert_eq!(reason(read("FAMENV~1.ENV")), PathRefusal::ShortName);
    assert_eq!(reason(read("vault/id_rsa")), PathRefusal::SecretDefault);
    assert_eq!(reason(read("build/out.txt")), PathRefusal::Ignored);
    if made_link {
        assert_eq!(reason(read("envlink")), PathRefusal::SecretDefault);
    }
    assert_eq!(read(".latticeignore").unwrap().1, PathClass::Authority);
}

/// WF4: escapes are refused: a junction outside, an alternate data stream, a
/// verbatim path, device names, a link to the folder itself and to `.git`.
/// (A reparse point at the lease path belongs to the lease, row E5.)
/// Mutant: links followed anywhere local (the walk not bounded by the root).
#[test]
fn wf4_escapes_are_refused() {
    let scratch = Scratch::new("ws-wf4");
    let repo = ignoring_repo(&scratch);
    let outside = scratch.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("secret.txt"), b"s\n").unwrap();
    junction(&repo.join("out"), &outside);
    junction(&repo.join("self"), &repo);
    junction(&repo.join("gitlink"), &repo.join(".git"));
    std::fs::write(repo.join("a.txt"), b"a\n").unwrap();
    let workspace = attach(&scratch, &repo);
    let runner = scratch.runner();
    let read = |request: &str| resolve(&workspace, &runner, request, Want::Existing);
    let _ = localfs::record::take();
    assert_eq!(reason(read("out/secret.txt")), PathRefusal::RemoteLink);
    let opened = localfs::record::take();
    assert!(
        !opened.iter().any(|path| path.ends_with("secret.txt")),
        "refused before it was followed: {opened:?}"
    );
    assert_eq!(reason(read("a.txt:secret")), PathRefusal::Stream);
    assert_eq!(reason(read(r"\\?\C:\Windows\win.ini")), PathRefusal::Unc);
    assert_eq!(reason(read("C:/Windows/win.ini")), PathRefusal::Outside);
    for device in ["CON", "nul.txt", "sub/COM1", "LPT\u{b9}.log", "aux"] {
        assert_eq!(reason(read(device)), PathRefusal::Device, "{device}");
    }
    assert_eq!(
        reason(read("self")),
        PathRefusal::Outside,
        "the root itself"
    );
    assert_eq!(reason(read("gitlink/config")), PathRefusal::GitDir);
    assert_eq!(reason(read("../outside/secret.txt")), PathRefusal::Escape);
}

/// SF9's path half: what staging would refuse (`..`, a junction out, `.git`,
/// an ignored file, `<globals>`), and what it may create.
#[test]
fn sf9_paths_staging_would_refuse_and_create() {
    let scratch = Scratch::new("ws-sf9");
    let repo = ignoring_repo(&scratch);
    let outside = scratch.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    junction(&repo.join("out"), &outside);
    let workspace = attach(&scratch, &repo);
    let runner = scratch.runner();
    let stage = |request: &str| resolve(&workspace, &runner, request, Want::MayCreate);
    assert_eq!(reason(stage("../x.txt")), PathRefusal::Escape);
    assert_eq!(reason(stage("out/new.txt")), PathRefusal::RemoteLink);
    assert_eq!(reason(stage(".git/config")), PathRefusal::GitDir);
    assert_eq!(
        reason(stage("debug.log")),
        PathRefusal::Ignored,
        "a new ignored file"
    );
    assert_eq!(reason(stage("private/new.txt")), PathRefusal::Ignored);
    assert_eq!(reason(stage("CON.txt")), PathRefusal::Device);
    assert_eq!(
        stage("new.txt").unwrap(),
        ("new.txt".into(), PathClass::Normal)
    );
    assert_eq!(
        stage("SUB/new.txt").unwrap(),
        ("sub/new.txt".into(), PathClass::Normal),
        "the parent's real spelling"
    );
    assert_eq!(
        stage("tools/new.ps1"),
        Err(PathError::Missing),
        "a missing folder is not made here"
    );
    assert_eq!(stage("run.ps1").unwrap().1, PathClass::Authority);
    assert_eq!(
        resolve(&workspace, &runner, "nothing.txt", Want::Existing),
        Err(PathError::Missing)
    );
}

/// WP8 fails closed: when git cannot answer in a git workspace, the path is
/// refused with "git could not say".
#[test]
fn wp8_a_git_that_cannot_answer_refuses() {
    let scratch = Scratch::new("ws-nogit");
    let repo = ignoring_repo(&scratch);
    let workspace = attach(&scratch, &repo);
    let no_git = GitRunner::new(
        Arc::new(
            MapEnv::new()
                .with("PATH", scratch.path().join("empty").as_os_str())
                .with("SystemRoot", std::env::var_os("SystemRoot").unwrap()),
        ),
        &state(&scratch),
    );
    match resolve(&workspace, &no_git, "readme.txt", Want::Existing) {
        Err(PathError::Refused(refused)) => {
            assert_eq!(refused.reason, PathRefusal::Ignored);
            assert_eq!(
                refused.sentence,
                "git could not say whether that file is ignored, so it is not shown."
            );
        }
        other => panic!("{other:?}"),
    }
}

/// FT6 at attach: a folder whose gitfile names the network is attached as
/// one without git, with a note, and nothing opens the network path.
#[test]
fn ft6_at_attach_a_network_gitfile_means_without_git() {
    let scratch = Scratch::new("ws-ft6");
    let folder = scratch.path().join("nested");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::write(folder.join(".git"), format!("gitdir: {UNC}\n")).unwrap();
    let _ = localfs::record::take();
    let workspace = attach(&scratch, &folder);
    never_opened_the_network(&localfs::record::take());
    assert!(matches!(workspace.repo, Repo::Without(_)));
    assert!(workspace.top_level.is_none());
    assert!(workspace.git_note.unwrap().contains("network"));
}
