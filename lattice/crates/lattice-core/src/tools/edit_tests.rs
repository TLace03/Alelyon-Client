//! The staging tools against folders and scratch repositories this file makes
//! (the chat core's spec §7.4: SF1, SF8, SF9, SF12, the ST4 table on the
//! derived path, ST2, ST5). Every folder is temporary; git runs only in
//! scratch repositories. Each test that stages compares the workspace's files
//! (bytes and last-write times) before and after: staging writes nothing
//! there (ST1, SF1), and nothing under the temporary roots disappears (NF1).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use lattice_protocol::conversation::{ChangeKind, Mode, Origin};

use super::edit::{
    DeleteFileArgs, EditFileArgs, StageContext, WriteFileArgs, delete_file, edit_file, write_file,
};
use super::read::{
    GlobArgs, GrepArgs, ListDirArgs, NoOverlay, Overlay, ReadContext, ReadFileArgs, ToolError,
    glob, grep, list_dir, read_file,
};
use crate::convo::item::{BaseState, Eol};
use crate::convo::sidecar::{NewMeta, SidecarStore};
use crate::git::runner::GitRunner;
use crate::git::tests::Scratch;
use crate::sha::sha256_hex;
use crate::staging::{BOM, Staging};
use crate::state::StateRoot;
use crate::workspace::Workspace;
use crate::workspace::attach::attach_path;

const TURN: &str = "a1b2c3d4e5f6";
const CALL: &str = "call_1";

/// One conversation over one folder: its workspace, its staging (whose
/// record lives under the scratch area's `state/`), and the context the
/// tools take.
struct Convo {
    workspace: Workspace,
    runner: GitRunner,
    staging: Staging,
    mode: Mode,
    trusted: bool,
    turn: String,
    call: String,
}

fn attach(scratch: &Scratch, folder: &Path) -> Workspace {
    attach_path(
        folder,
        &scratch.env(),
        &StateRoot::at(scratch.path().join("state")),
        &scratch.runner(),
    )
    .unwrap()
}

impl Convo {
    fn new(scratch: &Scratch, folder: &Path, id: &str) -> Self {
        let store = SidecarStore::new(scratch.path().join("state").join("chat"));
        let (sidecar, _) = store
            .open_for_writing(
                id,
                1.25,
                NewMeta {
                    workspace: None,
                    mode: Mode::Agent,
                    origin: Origin::Native,
                },
            )
            .unwrap();
        Self {
            workspace: attach(scratch, folder),
            runner: scratch.runner(),
            staging: Staging::new(Arc::new(sidecar)),
            mode: Mode::Agent,
            trusted: true,
            turn: TURN.into(),
            call: CALL.into(),
        }
    }

    fn ctx(&self) -> StageContext<'_> {
        StageContext {
            workspace: &self.workspace,
            runner: &self.runner,
            staging: &self.staging,
            mode: self.mode,
            trusted: self.trusted,
            turn: &self.turn,
            call: &self.call,
        }
    }

    fn reads<'a>(&'a self, overlay: &'a dyn Overlay) -> ReadContext<'a> {
        ReadContext {
            workspace: &self.workspace,
            runner: &self.runner,
            overlay,
        }
    }

    fn edit(&self, path: &str, old: &str, new: &str) -> Result<String, ToolError> {
        edit_file(
            &self.ctx(),
            &EditFileArgs {
                path: path.into(),
                old_string: old.into(),
                new_string: new.into(),
                replace_all: false,
            },
        )
    }

    fn write(&self, path: &str, content: &str) -> Result<String, ToolError> {
        write_file(
            &self.ctx(),
            &WriteFileArgs {
                path: path.into(),
                content: content.into(),
            },
        )
    }

    fn delete(&self, path: &str) -> Result<String, ToolError> {
        delete_file(&self.ctx(), &DeleteFileArgs { path: path.into() })
    }

    /// `read_file` as this conversation (through its staged view).
    fn read(&self, path: &str) -> Result<String, ToolError> {
        read_file(
            &self.reads(&self.staging),
            &ReadFileArgs {
                path: path.into(),
                ..ReadFileArgs::default()
            },
        )
    }

    /// `read_file` as another conversation, or a command: the disk.
    fn read_disk(&self, path: &str) -> Result<String, ToolError> {
        read_file(
            &self.reads(&NoOverlay),
            &ReadFileArgs {
                path: path.into(),
                ..ReadFileArgs::default()
            },
        )
    }
}

/// Every file under `root`: its SHA-256 and last-write time.
fn files(root: &Path) -> BTreeMap<String, (String, Option<SystemTime>)> {
    fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<String, (String, Option<SystemTime>)>) {
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
                let bytes = std::fs::read(&path).unwrap_or_default();
                let name = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                out.insert(name, (sha256_hex(&bytes), meta.modified().ok()));
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// NF1: every file's bytes before are still somewhere after.
fn nothing_lost(
    before: &BTreeMap<String, (String, Option<SystemTime>)>,
    after: &BTreeMap<String, (String, Option<SystemTime>)>,
) {
    for (name, (hash, _)) in before {
        assert!(
            after.contains_key(name) || after.values().any(|(other, _)| other == hash),
            "{name} is gone and its bytes are nowhere"
        );
    }
}

fn plain_folder(scratch: &Scratch) -> PathBuf {
    let folder = scratch.path().join("plain");
    std::fs::create_dir_all(folder.join("sub")).unwrap();
    std::fs::write(folder.join("a.txt"), "old\nkeep\n").unwrap();
    std::fs::write(folder.join("gone.txt"), "here\n").unwrap();
    std::fs::write(folder.join("sub").join("b.txt"), "bee\n").unwrap();
    folder
}

fn git_folder(scratch: &Scratch) -> PathBuf {
    let repo = scratch.path().join("repo");
    std::fs::create_dir_all(repo.join("sub")).unwrap();
    std::fs::create_dir_all(repo.join("private")).unwrap();
    scratch.git(&repo, &["init", "-q"]);
    std::fs::write(repo.join(".gitignore"), "*.log\n/private/\n").unwrap();
    std::fs::write(repo.join("a.txt"), "old\nkeep\n").unwrap();
    std::fs::write(repo.join("gone.txt"), "here\n").unwrap();
    std::fs::write(repo.join("sub").join("b.txt"), "bee\n").unwrap();
    std::fs::write(repo.join("private").join("p.txt"), "p\n").unwrap();
    scratch.git(&repo, &["add", "."]);
    scratch.git(&repo, &["commit", "-q", "-m", "one"]);
    repo
}

fn junction(link: &Path, target: &Path) {
    std::fs::create_dir(link).unwrap();
    lattice_sys::fs::seam::create_junction(link, target).unwrap();
}

/// ST5's answer.
fn staged(path: &str, added: u32, removed: u32) -> String {
    format!(
        "Staged `{path}` (+{added} \u{2212}{removed}). It is not on disk until the user keeps it."
    )
}

/// SF8: read-your-writes. Within the conversation, `read_file`, `list_dir`,
/// `glob` and `grep` see the staged view: an edited file reads with its new
/// bytes, a staged creation is listed and matched, a staged deletion
/// disappears. Another conversation (and a command) sees the disk. Both in
/// a git repository and in a folder without git. SF1 holds throughout.
/// Mutant: the staging overlay answering nothing (the tools read the disk).
#[test]
fn sf8_read_your_writes_in_every_read_tool() {
    let scratch = Scratch::new("edit-sf8");
    for (n, folder) in [git_folder(&scratch), plain_folder(&scratch)]
        .into_iter()
        .enumerate()
    {
        let convo = Convo::new(&scratch, &folder, &format!("c0ffee00000{n}"));
        let before = files(&folder);
        assert_eq!(
            convo.edit("a.txt", "old", "new").unwrap(),
            staged("a.txt", 1, 1)
        );
        assert_eq!(
            convo.write("sub/made.txt", "fresh\n").unwrap(),
            staged("sub/made.txt", 1, 0)
        );
        assert_eq!(convo.delete("gone.txt").unwrap(), staged("gone.txt", 0, 1));

        assert_eq!(
            convo.read("a.txt").unwrap(),
            "a.txt (2 lines; lines 1-2 shown; staged version)\n1\tnew\n2\tkeep"
        );
        assert_eq!(
            convo.read("sub/made.txt").unwrap(),
            "sub/made.txt (1 lines; lines 1-1 shown; staged version)\n1\tfresh"
        );
        assert_eq!(
            convo.read("gone.txt"),
            Err(ToolError("There is no such file in this folder.".into()))
        );
        let ctx = convo.reads(&convo.staging);
        let listing = list_dir(
            &ctx,
            &ListDirArgs {
                path: None,
                depth: Some(2),
            },
        )
        .unwrap();
        assert!(listing.contains("sub/made.txt (staged)"), "{listing}");
        assert!(!listing.contains("gone.txt"), "{listing}");
        let globbed = glob(
            &ctx,
            &GlobArgs {
                pattern: "**/*.txt".into(),
            },
        )
        .unwrap();
        assert!(globbed.contains("sub/made.txt (staged)"), "{globbed}");
        assert!(!globbed.contains("gone.txt"), "{globbed}");
        let grepped = grep(
            &ctx,
            &GrepArgs {
                pattern: "new|fresh|here|old".into(),
                ..GrepArgs::default()
            },
        )
        .unwrap();
        assert_eq!(
            grepped, "2 matches in 2 files\na.txt:1:new\nsub/made.txt:1:fresh",
            "{folder:?}"
        );

        // Another conversation, or a command: the disk.
        assert_eq!(
            convo.read_disk("a.txt").unwrap(),
            "a.txt (2 lines; lines 1-2 shown)\n1\told\n2\tkeep"
        );
        assert!(convo.read_disk("gone.txt").is_ok());
        assert!(convo.read_disk("sub/made.txt").is_err());
        let disk = convo.reads(&NoOverlay);
        let disk_grep = grep(
            &disk,
            &GrepArgs {
                pattern: "new|fresh|here|old".into(),
                ..GrepArgs::default()
            },
        )
        .unwrap();
        assert_eq!(
            disk_grep,
            "2 matches in 2 files\na.txt:1:old\ngone.txt:1:here"
        );
        assert_eq!(files(&folder), before, "SF1: staging wrote nothing");
    }
}

/// SF1: no staging sequence changes a workspace byte or last-write time:
/// edits, an overwrite after a read, a creation and its edit, a delete, a
/// write over the staged deletion, a delete of a staged creation. The
/// record grows; nothing under the scratch area disappears (NF1).
/// Mutant: write-through staging (`edit_file` also writes the file).
#[test]
fn sf1_no_staging_sequence_touches_the_folder() {
    let scratch = Scratch::new("edit-sf1");
    let folder = git_folder(&scratch);
    let convo = Convo::new(&scratch, &folder, "c0ffee000001");
    let before = files(&folder);
    let everything = files(scratch.path());
    convo.edit("a.txt", "old", "new").unwrap();
    convo.edit("a.txt", "keep", "kept").unwrap();
    convo.read("sub/b.txt").unwrap();
    convo.write("sub/b.txt", "wasp\n").unwrap();
    convo.write("c.txt", "one\n").unwrap();
    convo.edit("c.txt", "one", "two").unwrap();
    convo.delete("gone.txt").unwrap();
    convo.write("gone.txt", "back\n").unwrap();
    convo.delete("c.txt").unwrap();
    assert_eq!(files(&folder), before, "a workspace byte or mtime changed");
    let after = files(scratch.path());
    nothing_lost(&everything, &after);
    assert!(after.len() > everything.len(), "the record grew");
    let kinds: Vec<(String, ChangeKind)> = convo
        .staging
        .changes()
        .into_iter()
        .map(|change| (change.path, change.kind))
        .collect();
    assert_eq!(
        kinds,
        vec![
            ("a.txt".into(), ChangeKind::Edit),
            ("sub/b.txt".into(), ChangeKind::Overwrite),
            ("c.txt".into(), ChangeKind::Delete),
            ("gone.txt".into(), ChangeKind::Overwrite),
        ]
    );
}

/// SF9: refused at staging, by every staging tool: `..`, a junction out of
/// the folder, `.git`, ignored files (an existing and a new one, and one in
/// an ignored folder), Lattice's own state (`<globals>`), a device name and
/// an 8.3 alias. Nothing is staged and nothing changes.
/// Mutant: a lexical check only (the junction and the ignored files pass).
#[test]
fn sf9_refused_at_staging() {
    let scratch = Scratch::new("edit-sf9");
    let folder = git_folder(&scratch);
    std::fs::write(folder.join("debug.log"), "log\n").unwrap();
    let outside = scratch.path().join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("x.txt"), "outside\n").unwrap();
    junction(&folder.join("out"), &outside);
    std::fs::create_dir_all(folder.join("globals")).unwrap();
    std::fs::write(folder.join("globals").join("state.json"), "{}\n").unwrap();
    let mut convo = Convo::new(&scratch, &folder, "c0ffee000001");
    let globals = convo.workspace.root.join("globals");
    convo.workspace.lattice_state.push(globals);
    let before = files(scratch.path());
    let cases = [
        (
            "../outside/x.txt",
            "the path has an empty, '.' or '..' component",
        ),
        (
            "out/x.txt",
            "That link points outside the folder or to the network.",
        ),
        (
            "out/new.txt",
            "That link points outside the folder or to the network.",
        ),
        (".git/config", "git's own directory is not readable here"),
        ("debug.log", "git ignores that file, so it is not shown."),
        ("new.log", "git ignores that file, so it is not shown."),
        (
            "private/p.txt",
            "git ignores that file, so it is not shown.",
        ),
        (
            "globals/state.json",
            "That is Lattice's own state, so it is not used.",
        ),
        ("CON.txt", "That name is a device, not a file."),
        ("A~1.TXT", "Use the file's full name."),
    ];
    for (path, sentence) in cases {
        let expected = Err(ToolError(sentence.to_owned()));
        assert_eq!(convo.edit(path, "x", "y"), expected, "edit {path}");
        assert_eq!(convo.write(path, "y\n"), expected, "write {path}");
        assert_eq!(convo.delete(path), expected, "delete {path}");
    }
    assert!(convo.staging.changes().is_empty());
    assert_eq!(files(scratch.path()), before);
}

/// SF12: `write_file` over an existing file this conversation has neither
/// read nor staged is refused; after `read_file` it stages; a file the
/// conversation staged needs no read; a new file needs none. Another
/// conversation's read does not count.
/// Mutant: no read check.
#[test]
fn sf12_write_over_an_unread_file_is_refused() {
    let scratch = Scratch::new("edit-sf12");
    let folder = plain_folder(&scratch);
    let convo = Convo::new(&scratch, &folder, "c0ffee000001");
    let before = files(&folder);
    let refused = Err(ToolError("Read the file before replacing it.".into()));
    assert_eq!(convo.write("a.txt", "x\n"), refused);
    convo.read_disk("a.txt").unwrap();
    assert_eq!(
        convo.write("A.TXT", "x\n"),
        refused,
        "another conversation's read does not count"
    );
    convo.read("a.txt").unwrap();
    assert_eq!(convo.write("A.TXT", "x\n").unwrap(), staged("a.txt", 1, 2));
    convo.edit("sub/b.txt", "bee", "bea").unwrap();
    assert!(
        convo.write("sub/b.txt", "wasp\n").is_ok(),
        "staged: no read needed"
    );
    assert!(convo.write("new.txt", "n\n").is_ok(), "new: no read needed");
    assert_eq!(files(&folder), before);
}

/// The ST4 table through the staging tools: every authority file (the
/// program-configuration files included) is staged marked `authority`, and
/// ordinary files are not, decided on the derived path: a file reached
/// through an in-root junction to `.github` is authority under its real
/// path, and one named in another case keeps its real spelling.
/// Mutant: the class taken from the request string.
#[test]
fn st4_staging_marks_authority_on_the_derived_path() {
    let scratch = Scratch::new("edit-st4");
    let folder = scratch.path().join("plain");
    for dir in [
        ".lattice/rules",
        ".cursor/rules",
        ".github/workflows",
        ".vscode",
        ".cargo",
        "crate/.cargo",
        "web",
        "tools",
        "src",
        "docs",
    ] {
        std::fs::create_dir_all(folder.join(dir)).unwrap();
    }
    std::fs::write(folder.join(".github/workflows/ci.yml"), "on: push\n").unwrap();
    junction(&folder.join("gh"), &folder.join(".github"));
    let convo = Convo::new(&scratch, &folder, "c0ffee000001");
    let before = files(&folder);
    let authority = [
        "AGENTS.md",
        "CLAUDE.md",
        ".lattice/rules/x.md",
        ".cursor/rules/a.mdc",
        ".vscode/settings.json",
        ".gitignore",
        ".gitattributes",
        ".latticeignore",
        "Cargo.toml",
        "Cargo.lock",
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
        // The program-configuration files (ST4's fifth group).
        ".cargo/config.toml",
        "crate/.cargo/config",
        "rust-toolchain",
        "rust-toolchain.toml",
        ".npmrc",
        ".yarnrc",
        ".yarnrc.yml",
        "Directory.Build.props",
        "Directory.Build.targets",
        "nuget.config",
        "global.json",
        "tools/run.ps1",
        "x.bat",
        "y.cmd",
        "z.sh",
    ];
    let ordinary = [
        "src/main.rs",
        "README.md",
        "docs/agents.txt",
        "cargo.tom",
        "requirements.md",
        "package.json5",
        "x.shx",
    ];
    for path in authority.iter().chain(ordinary.iter()) {
        convo.write(path, "x\n").unwrap();
    }
    convo
        .edit("gh/workflows/ci.yml", "push", "pull_request")
        .unwrap();
    let marked: BTreeMap<String, bool> = convo
        .staging
        .changes()
        .into_iter()
        .map(|change| (change.path, change.authority))
        .collect();
    for path in authority {
        assert_eq!(marked.get(path), Some(&true), "{path}");
    }
    for path in ordinary {
        assert_eq!(marked.get(path), Some(&false), "{path}");
    }
    assert_eq!(
        marked.get(".github/workflows/ci.yml"),
        Some(&true),
        "through the junction, under its real path: {marked:?}"
    );
    assert!(!marked.contains_key("gh/workflows/ci.yml"));
    assert_eq!(files(&folder), before);
}

/// The tools' sentences, ST5, Ask mode and an untrusted folder, folders,
/// missing files, a staged deletion, and the encoding a replaced file keeps.
#[test]
fn the_staging_tools_say_what_they_did_and_why_not() {
    let scratch = Scratch::new("edit-words");
    let folder = plain_folder(&scratch);
    std::fs::write(folder.join("crlf.txt"), [BOM, b"one\r\ntwo\r\n"].concat()).unwrap();
    std::fs::write(folder.join("many.txt"), "a a\n").unwrap();
    std::fs::write(folder.join("bin.dat"), b"a\0b").unwrap();
    let mut convo = Convo::new(&scratch, &folder, "c0ffee000001");
    let before = files(&folder);
    let err = |text: &str| Err(ToolError(text.to_owned()));
    assert_eq!(
        convo.edit("a.txt", "zzz", "y"),
        err("old_string was not found")
    );
    assert_eq!(
        convo.edit("many.txt", "a", "b"),
        err("old_string occurs 2 times; give more context, or set replace_all")
    );
    assert_eq!(
        convo.edit("a.txt", "old", "old"),
        err("old_string and new_string are the same, so nothing would change.")
    );
    assert_eq!(
        convo.edit("a.txt", "", "x"),
        err("old_string is empty; use write_file to create or replace a file.")
    );
    assert_eq!(
        convo.edit("bin.dat", "a", "b"),
        err("binary file, 3 bytes; edit_file changes text files only")
    );
    assert_eq!(
        convo.edit("nothing.txt", "a", "b"),
        err("There is no such file in this folder.")
    );
    assert_eq!(
        convo.edit("sub", "a", "b"),
        err("That is a folder; edit_file changes files.")
    );
    assert_eq!(
        convo.delete("sub"),
        err("delete_file removes files only; that is a folder.")
    );
    assert_eq!(
        convo.delete("nothing.txt"),
        err("There is no such file in this folder.")
    );
    assert_eq!(
        convo.write("nowhere/new.txt", "x"),
        err("That folder does not exist; write_file does not make folders.")
    );
    assert_eq!(
        convo.write("big.txt", &"x".repeat(2 * 1024 * 1024 + 1)),
        err("write_file writes at most 2 MiB (2,097,152 bytes).")
    );

    // A CRLF file with a BOM: an LF-form edit and an LF write both keep
    // CRLF and the BOM.
    convo.edit("crlf.txt", "one\ntwo", "uno\ndos").unwrap();
    let change = convo.staging.live("crlf.txt").unwrap();
    assert!(matches!(
        change.base,
        BaseState::Present {
            eol: Eol::Crlf,
            bom: true,
            ..
        }
    ));
    assert_eq!(
        convo.staging.staged("crlf.txt"),
        Some(super::read::Staged::Bytes(
            [BOM, b"uno\r\ndos\r\n"].concat()
        ))
    );
    convo.write("crlf.txt", "three\nfour\n").unwrap();
    assert_eq!(
        convo.staging.staged("crlf.txt"),
        Some(super::read::Staged::Bytes(
            [BOM, b"three\r\nfour\r\n"].concat()
        ))
    );

    // A staged deletion: the file is gone in this view; writing it again
    // overwrites the original.
    convo.delete("gone.txt").unwrap();
    assert_eq!(
        convo.edit("gone.txt", "here", "x"),
        err("There is no such file in this folder.")
    );
    assert_eq!(
        convo.delete("gone.txt"),
        err("There is no such file in this folder.")
    );
    assert_eq!(
        convo.write("gone.txt", "again\n").unwrap(),
        staged("gone.txt", 1, 1)
    );
    assert_eq!(
        convo.staging.live("gone.txt").unwrap().kind,
        ChangeKind::Overwrite
    );

    // Ask mode and an untrusted folder refuse before anything is read.
    let count = convo.staging.changes().len();
    convo.mode = Mode::Ask;
    assert_eq!(
        convo.edit("a.txt", "old", "x"),
        err("That is not available in Ask mode.")
    );
    assert_eq!(
        convo.write("n.txt", "x"),
        err("That is not available in Ask mode.")
    );
    assert_eq!(
        convo.delete("a.txt"),
        err("That is not available in Ask mode.")
    );
    convo.mode = Mode::Agent;
    convo.trusted = false;
    assert_eq!(
        convo.edit("a.txt", "old", "x"),
        err("Trust this folder to use Agent mode.")
    );
    assert_eq!(convo.staging.changes().len(), count);
    assert_eq!(files(&folder), before);
}
