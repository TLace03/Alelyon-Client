//! `run_command` against folders this file makes (the chat core's spec
//! §7.6, §9.3; §16.4 CF1, CF3–CF6, CF10–CF13 (the X14 part), CF15, CF17).
//!
//! Every folder is temporary. The commands that run are the ones these tests
//! choose: Windows PowerShell 5.1 on text that writes or sleeps inside the
//! temporary folder, and `where.exe` (a system program that only prints a
//! path) for a standing entry. Each has a timeout; a running command is
//! ended through its Job Object. Programs planted on a test `PATH` are empty
//! files that are resolved and never run.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::executor::block_on;
use lattice_protocol::conversation::{ApprovalDetail, CommandMode, ExitReason, Mode, Origin};

use super::*;
use crate::clock::Clock;
use crate::convo::sidecar::{NewMeta, SidecarStore};
use crate::env::MapEnv;
use crate::exec::allowlist::{Candidate, PermissionError, Permissions};
use crate::git::tests::Scratch;
use crate::ports::fake::RecordingConfirm;
use crate::ports::{ConfirmRequest, Confirmer};
use crate::state::StateRoot;
use crate::tools::edit::{EditFileArgs, StageContext, edit_file};
use crate::workspace::Workspace;
use crate::workspace::attach::attach_path;

const ID: &str = "c0ffee000007";

/// Records every child the run asks for, then starts it for real.
pub(crate) struct Recorder {
    inner: SpawnLauncher,
    pub(crate) launched: Mutex<Vec<ChildSpec>>,
}

impl Launcher for Recorder {
    fn launch(&self, spec: &ChildSpec, workspace: &Path) -> io::Result<Child> {
        self.launched.lock().unwrap().push(spec.clone());
        self.inner.launch(spec, workspace)
    }
}

/// The lease and the checkpoints as a test sets them.
#[derive(Default)]
pub(crate) struct FakeHooks {
    pub(crate) elsewhere: AtomicBool,
    pub(crate) before_fails: AtomicBool,
    pub(crate) calls: Mutex<Vec<String>>,
}

impl CommandHooks for FakeHooks {
    fn lease(&self) -> Lease {
        self.calls.lock().unwrap().push("lease".into());
        if self.elsewhere.load(Ordering::SeqCst) {
            Lease::Elsewhere
        } else {
            Lease::Held
        }
    }

    fn before(&self, call: &CallId) -> Result<Option<CheckpointId>, String> {
        self.calls.lock().unwrap().push(format!("before {call}"));
        if self.before_fails.load(Ordering::SeqCst) {
            Err("no checkpoint".into())
        } else {
            Ok(Some(7))
        }
    }

    fn after(&self, call: &CallId, before: Option<CheckpointId>) -> AfterCommand {
        self.calls
            .lock()
            .unwrap()
            .push(format!("after {call} {before:?}"));
        AfterCommand::default()
    }
}

/// One conversation over one plain folder, with everything a command needs.
pub(crate) struct Bench {
    pub(crate) scratch: Scratch,
    pub(crate) folder: PathBuf,
    pub(crate) workspace: Workspace,
    pub(crate) runner: GitRunner,
    pub(crate) staging: Staging,
    pub(crate) permissions: Permissions,
    pub(crate) confirm: RecordingConfirm,
    pub(crate) confirmer: Confirmer,
    pub(crate) slots: CommandSlots,
    pub(crate) launcher: Recorder,
    pub(crate) hooks: FakeHooks,
    /// Lattice's own environment, as the context reads it.
    pub(crate) env: MapEnv,
    pub(crate) globals: PathBuf,
    pub(crate) mode: Mode,
}

impl Bench {
    pub(crate) fn new(tag: &str) -> Self {
        let scratch = Scratch::new(tag);
        let folder = scratch.path().join("work");
        std::fs::create_dir_all(folder.join("sub")).unwrap();
        std::fs::write(folder.join("a.txt"), "old\n").unwrap();
        let state = StateRoot::at(scratch.path().join("state"));
        let runner = scratch.runner();
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
        let clock: Clock = Arc::new(|| 3000.0);
        let confirm = RecordingConfirm::answering(true);
        let env = scratch.env();
        Self {
            folder,
            permissions: Permissions::new(&state, clock),
            confirmer: Confirmer::new(Arc::new(confirm.clone())),
            confirm,
            slots: CommandSlots::default(),
            launcher: Recorder {
                inner: SpawnLauncher {
                    env: Arc::new(env.clone()),
                    globals: state.globals.clone(),
                },
                launched: Mutex::default(),
            },
            hooks: FakeHooks::default(),
            globals: state.globals.clone(),
            env,
            staging: Staging::new(Arc::new(sidecar)),
            workspace,
            runner,
            mode: Mode::Agent,
            scratch,
        }
    }

    pub(crate) fn ctx(&self) -> CommandContext<'_> {
        CommandContext {
            workspace: &self.workspace,
            runner: &self.runner,
            staging: &self.staging,
            mode: self.mode,
            trusted: true,
            slots: &self.slots,
            permissions: &self.permissions,
            confirmer: &self.confirmer,
            launcher: &self.launcher,
            hooks: &self.hooks,
            env: &self.env,
            globals: &self.globals,
            remote: None,
        }
    }

    pub(crate) fn answer(&self, yes: bool) {
        *self.confirm.answer.lock().unwrap() = yes;
    }

    pub(crate) fn launched(&self) -> Vec<ChildSpec> {
        self.launcher.launched.lock().unwrap().clone()
    }

    pub(crate) fn args(command: &str) -> RunCommandArgs {
        RunCommandArgs {
            command: command.into(),
            cwd: None,
            timeout_s: Some(60),
            background: false,
        }
    }

    pub(crate) fn prepare(&self, command: &str) -> Result<Prepared, ToolError> {
        prepare(&self.ctx(), &Self::args(command))
    }

    /// Prepare, approve and run, as a conversation would.
    pub(crate) fn run_text(&self, command: &str, call: &str) -> Result<CommandOutcome, Refused> {
        let prepared = self.prepare(command).map_err(|e| Refused::invalid(e.0))?;
        let call = call.to_owned();
        let approval = block_on(approve(&self.ctx(), &prepared, &call))?;
        run(
            &self.ctx(),
            &prepared,
            &approval,
            &call,
            &StopHandle::default(),
        )
    }

    /// Stage an `edit_file` of `a.txt`: one change waits.
    pub(crate) fn stage_edit(&self) {
        edit_file(
            &StageContext {
                workspace: &self.workspace,
                runner: &self.runner,
                staging: &self.staging,
                mode: Mode::Agent,
                trusted: true,
                turn: &"a1b2c3d4e5f6".to_owned(),
                call: &"call_e".to_owned(),
            },
            &EditFileArgs {
                path: "a.txt".into(),
                old_string: "old".into(),
                new_string: "new".into(),
                replace_all: false,
            },
        )
        .unwrap();
    }

    /// A folder on a test `PATH`, with empty files of these names (never run).
    pub(crate) fn plant(&self, dir: &str, files: &[&str]) -> PathBuf {
        let dir = self.scratch.path().join(dir);
        std::fs::create_dir_all(&dir).unwrap();
        for file in files {
            std::fs::write(dir.join(file), b"").unwrap();
        }
        dir
    }

    /// The context's environment with `PATH` set to `dirs`.
    pub(crate) fn with_path(&mut self, dirs: &[&Path]) {
        let joined: Vec<String> = dirs
            .iter()
            .map(|d| d.to_string_lossy().into_owned())
            .collect();
        self.env.set("PATH", joined.join(";"));
    }
}

fn file_list(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in std::fs::read_dir(&at).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(
                    path.strip_prefix(dir)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    out.sort();
    out
}

// ------------------------------------------------------------ in memory

#[test]
fn base64_and_the_encoded_command() {
    for (input, output) in [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ] {
        assert_eq!(
            base64(input.as_bytes()),
            output,
            "RFC 4648 vector {input:?}"
        );
    }
    // UTF-16LE: d\0i\0r\0.
    assert_eq!(encoded_command("dir"), "ZABpAHIA");
    assert_eq!(encoded_command("\u{2013}"), "EyA=");
}

#[test]
fn x9_the_buffer_keeps_its_head_and_its_ring() {
    let mut buffer = OutputBuffer::default();
    buffer.push(&vec![b'h'; HEAD_BYTES - 2]);
    buffer.push(b"ab\ncd");
    assert_eq!(buffer.kept().len(), HEAD_BYTES + 3);
    assert_eq!(&buffer.kept()[HEAD_BYTES - 2..], b"ab\ncd");
    assert_eq!(buffer.lines(), 1);
    buffer.push(&vec![b'r'; RING_BYTES]);
    buffer.push(b"END");
    let kept = buffer.kept();
    assert_eq!(kept.len(), HEAD_BYTES + RING_BYTES, "at most 8 MiB");
    assert!(kept.ends_with(b"rEND"));
    assert_eq!(
        buffer.dropped(),
        6,
        "cd and the ring's first bytes are counted"
    );
    assert_eq!(buffer.total(), (HEAD_BYTES + 3 + RING_BYTES + 3) as u64);
    let mut big = OutputBuffer::default();
    big.push(&vec![b'x'; HEAD_BYTES + RING_BYTES + 10]);
    assert_eq!(big.kept().len(), HEAD_BYTES + RING_BYTES);
    assert_eq!(big.dropped(), 10);
}

#[test]
fn x9_the_model_reads_8_kib_then_24_kib_of_stripped_text() {
    let short = model_text(b"\x1b[32mok\x1b[0m\n", 0, "Exit code 0 after 0.1 s");
    assert_eq!(short, "ok\nExit code 0 after 0.1 s");
    assert_eq!(
        model_text(b"", 0, "Stopped by the user"),
        "(no output)\nStopped by the user"
    );
    let mut long = String::new();
    for n in 0..5000 {
        long.push_str(&format!("\x1b[31mline {n:05}\x1b[0m\n"));
    }
    let text = model_text(long.as_bytes(), 0, "Exit code 3 after 2.0 s");
    assert!(text.starts_with("line 00000\n"), "{}", &text[..40]);
    assert!(text.ends_with("line 04999\nExit code 3 after 2.0 s"));
    assert!(!text.contains('\u{1b}'));
    let stripped = 5000 * 11;
    let omitted = stripped - MODEL_HEAD - MODEL_TAIL;
    let marker = format!("[\u{2026} {omitted} bytes omitted \u{2026}]");
    assert!(text.contains(&marker), "{marker}");
    assert!(text.len() <= MODEL_HEAD + MODEL_TAIL + 100);
    // Bytes the buffer dropped are counted too.
    let dropped = model_text(b"a\nb\n", 12, "x");
    assert!(
        dropped.contains("[\u{2026} 12 bytes omitted \u{2026}]"),
        "{dropped}"
    );
    // A cut never splits a character.
    let wide = "\u{2013}".repeat(20_000);
    let cut = model_text(wide.as_bytes(), 0, "x");
    assert!(cut.contains("bytes omitted"));
    // UTF-16LE output reads as text.
    let utf16: Vec<u8> = "hello\r\n"
        .encode_utf16()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(model_text(&utf16, 0, "x"), "hello\r\nx");
}

#[test]
fn x14_one_slot_per_workspace() {
    let slots = CommandSlots::default();
    let first = slots.claim("w1", "call_1").unwrap();
    assert!(slots.running("w1"));
    assert!(slots.claim("w1", "call_2").is_none());
    let other = slots.claim("w2", "call_3").unwrap();
    assert_eq!(slots.holder("w1").as_deref(), Some("call_1"));
    drop(first);
    assert!(!slots.running("w1"));
    assert!(slots.running("w2"));
    drop(other);
}

// ------------------------------------------------------------- prepare

/// The call's checks: length, bidi controls (CF17), the timeout's bounds,
/// the working folder (X10), and the policy's refusals.
#[test]
fn prepare_checks_the_call() {
    let mut b = Bench::new("run-prepare");
    let refused = |b: &Bench, args: RunCommandArgs| prepare(&b.ctx(), &args).unwrap_err().0;
    assert_eq!(
        refused(&b, Bench::args(&"x".repeat(8001))),
        "The command is longer than 8,000 characters."
    );
    assert!(b.prepare(&"x".repeat(8000)).is_ok());
    assert_eq!(
        refused(&b, Bench::args("echo \u{202e}txt.exe")),
        "The command contains characters that change how text is displayed."
    );
    for seconds in [0, 3601] {
        let mut args = Bench::args("echo hi");
        args.timeout_s = Some(seconds);
        assert_eq!(
            refused(&b, args),
            "timeout_s must be between 1 and 3,600 seconds."
        );
    }
    let mut args = Bench::args("echo hi");
    args.timeout_s = None;
    assert_eq!(prepare(&b.ctx(), &args).unwrap().timeout_s, 600);
    let mut args = Bench::args("echo hi");
    args.cwd = Some("a.txt".into());
    assert_eq!(refused(&b, args), "cwd must be a folder in the workspace.");
    let mut args = Bench::args("echo hi");
    args.cwd = Some("..".into());
    assert!(prepare(&b.ctx(), &args).is_err());
    let mut args = Bench::args("echo hi");
    args.cwd = Some("sub/".into());
    let prepared = prepare(&b.ctx(), &args).unwrap();
    assert_eq!(prepared.cwd, "sub");
    assert!(prepared.cwd_path.ends_with("sub"));
    assert!(!prepared.cwd_path.to_string_lossy().starts_with(r"\\?\"));
    b.mode = Mode::Ask;
    assert_eq!(
        refused(&b, Bench::args("echo hi")),
        "That is not available in Ask mode."
    );
    assert!(b.launched().is_empty());
}

/// CF17: U+202E is refused (above); a Unicode dash is shown escaped on the
/// card and in the native dialog, and the command always asks.
#[test]
fn cf17_a_unicode_dash_is_shown_escaped() {
    let b = Bench::new("run-cf17");
    b.answer(false);
    let prepared = b.prepare("Write-Output a\u{2013}b").unwrap();
    let ApprovalDetail::Command { text, mode, .. } = &prepared.detail else {
        panic!("a command card");
    };
    assert_eq!(text, "Write-Output a\\u{2013}b");
    assert_eq!(*mode, CommandMode::PowerShell);
    assert!(!prepared.allow_always_offer);
    let refused = block_on(approve(&b.ctx(), &prepared, &"call_1".to_owned())).unwrap_err();
    assert!(refused.rejected);
    let asked = b.confirm.asked();
    assert_eq!(asked.len(), 1);
    let dialog = asked[0].dialog();
    assert!(
        dialog
            .lines
            .iter()
            .any(|line| line == "Command: Write-Output a\\u{2013}b"),
        "{dialog:?}"
    );
    assert!(dialog.lines.iter().all(|line| line.is_ascii()));
}

/// CF15: the page's Approve only asks: with the native `RunCommand` dialog
/// answering no, nothing is started, and asking again for the same call
/// opens no second dialog (CP3).
#[test]
fn cf15_a_refused_dialog_starts_nothing() {
    let b = Bench::new("run-cf15");
    b.answer(false);
    let before = file_list(&b.folder);
    let prepared = b
        .prepare("Set-Content -LiteralPath made.txt -Value x")
        .unwrap();
    assert!(matches!(prepared.verdict, Verdict::Ask(AskKind::Command)));
    let call = "call_15".to_owned();
    let refused = block_on(approve(&b.ctx(), &prepared, &call)).unwrap_err();
    assert_eq!(
        refused,
        Refused {
            sentence: "You did not confirm running this command, so it did not run.".into(),
            conflict: false,
            rejected: true,
        }
    );
    assert_eq!(
        b.confirm.asked(),
        vec![ConfirmRequest::RunCommand {
            text: "Set-Content -LiteralPath made.txt -Value x".into(),
            cwd: String::new(),
            mode: CommandMode::PowerShell,
            background: false,
        }]
    );
    b.answer(true);
    assert!(block_on(approve(&b.ctx(), &prepared, &call)).is_err());
    assert_eq!(
        b.confirm.asked().len(),
        1,
        "CP3: not asked again for this call"
    );
    assert_eq!(b.launched().len(), 0, "zero spawns");
    assert!(
        b.hooks.calls.lock().unwrap().is_empty(),
        "no lease, no checkpoint"
    );
    assert_eq!(file_list(&b.folder), before);
}

/// CF10: the approved-once text arrives in PowerShell exactly as shown:
/// quotes, `$`, `%`, `^`, backslashes before quotes, a backtick, a Unicode
/// dash, a trailing backslash, two lines. It runs as `powershell.exe
/// -NoProfile -NonInteractive -EncodedCommand`, by absolute path.
#[test]
fn cf10_the_approved_text_arrives_in_powershell_exactly() {
    let b = Bench::new("run-cf10");
    let payload = "say \"hi\" $HOME %PATH% ^& \\\"q\\\" `t \u{2013} two  spaces \"a  b\" end\\";
    let literal = payload.replace('\'', "''");
    let text = format!(
        "$p = '{literal}'\r\n[IO.File]::WriteAllText((Join-Path (Get-Location) 'seen.txt'), $p)"
    );
    let outcome = b.run_text(&text, "call_10").unwrap();
    assert_eq!(outcome.reason, ExitReason::Exited, "{}", outcome.model_text);
    assert_eq!(outcome.code, Some(0), "{}", outcome.model_text);
    let seen = std::fs::read(b.folder.join("seen.txt")).unwrap();
    let seen = String::from_utf8(seen).unwrap();
    assert_eq!(seen.trim_start_matches('\u{feff}'), payload);
    let launched = b.launched();
    assert_eq!(launched.len(), 1);
    let spec = &launched[0];
    let root = std::env::var_os("SystemRoot").unwrap();
    assert_eq!(
        spec.program,
        PathBuf::from(root).join(r"System32\WindowsPowerShell\v1.0\powershell.exe")
    );
    let argv: Vec<String> = spec
        .argv
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        argv,
        [
            "powershell.exe".to_owned(),
            "-NoProfile".to_owned(),
            "-NonInteractive".to_owned(),
            "-EncodedCommand".to_owned(),
            encoded_command(&text)
        ]
    );
    assert_eq!(outcome.mode, CommandMode::PowerShell);
    assert!(outcome.model_text.contains("Exit code 0 after"));
    assert_eq!(
        *b.hooks.calls.lock().unwrap(),
        ["lease", "before call_10", "after call_10 Some(7)"],
        "X11: the lease and the checkpoint come before the spawn"
    );
}

/// CF11: "Allow always" with the native dialog answering no creates no
/// entry and writes no file; the command still asks.
#[test]
fn cf11_a_refused_allow_always_creates_no_entry() {
    let b = Bench::new("run-cf11");
    let prepared = b.prepare("where.exe cmd.exe").unwrap();
    assert!(prepared.allow_always_offer, "X1-X3 hold");
    let candidate = prepared.candidate.clone().unwrap();
    b.answer(false);
    let refused = block_on(b.permissions.allow_always(
        &b.confirmer,
        &b.workspace,
        &candidate,
        "allow:call_11",
    ));
    assert_eq!(refused, Err(PermissionError::NotConfirmed));
    assert!(
        !b.permissions.file(&b.workspace).exists(),
        "no file written"
    );
    assert_eq!(b.permissions.entries(&b.workspace).unwrap(), vec![]);
    assert!(matches!(
        b.confirm.asked()[0],
        ConfirmRequest::AllowAlways { .. }
    ));
    let again = b.prepare("where.exe cmd.exe").unwrap();
    assert!(matches!(again.verdict, Verdict::Ask(AskKind::Command)));
    assert_eq!(b.launched().len(), 0);
}

/// CF3: a standing entry runs its resolved program directly, with the argv
/// that matched, and no dialog. X4: the same text in another folder, or with
/// other arguments, asks; a revoked entry no longer matches.
#[test]
fn cf3_a_standing_match_runs_the_matched_argv_directly() {
    let b = Bench::new("run-cf3");
    let prepared = b.prepare("where.exe cmd.exe \"where.exe\"").unwrap();
    let candidate = prepared.candidate.clone().unwrap();
    let entry = block_on(b.permissions.allow_always(
        &b.confirmer,
        &b.workspace,
        &candidate,
        "allow:call_3",
    ))
    .unwrap();
    assert_eq!(entry.argv, ["where.exe", "cmd.exe", "where.exe"]);
    assert_eq!(entry.cwd, "");
    assert!(
        entry
            .program
            .to_lowercase()
            .ends_with(r"\system32\where.exe"),
        "{}",
        entry.program
    );
    let asked_before = b.confirm.asked().len();
    let outcome = b
        .run_text("where.exe   cmd.exe where.exe", "call_3")
        .unwrap();
    assert_eq!(
        b.confirm.asked().len(),
        asked_before,
        "no RunCommand dialog"
    );
    assert_eq!(outcome.mode, CommandMode::Direct);
    assert_eq!(outcome.code, Some(0), "{}", outcome.model_text);
    let shown = outcome.model_text.to_lowercase();
    assert!(
        shown.contains(r"\cmd.exe") && shown.contains(r"\where.exe"),
        "{shown}"
    );
    let launched = b.launched();
    assert_eq!(launched.len(), 1);
    let argv: Vec<String> = launched[0]
        .argv
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        argv,
        ["where.exe", "cmd.exe", "where.exe"],
        "the argv that matched"
    );
    assert_eq!(launched[0].program, candidate.resolved.path);
    // X4: another folder, other arguments: they ask.
    let mut args = Bench::args("where.exe cmd.exe where.exe");
    args.cwd = Some("sub".into());
    let elsewhere = prepare(&b.ctx(), &args).unwrap();
    assert!(matches!(elsewhere.verdict, Verdict::Ask(AskKind::Command)));
    let other = b.prepare("where.exe cmd.exe").unwrap();
    assert!(matches!(other.verdict, Verdict::Ask(AskKind::Command)));
    b.permissions.revoke(&b.workspace, &entry.id).unwrap();
    let revoked = b.prepare("where.exe cmd.exe \"where.exe\"").unwrap();
    assert!(matches!(revoked.verdict, Verdict::Ask(AskKind::Command)));
    let listed = b.permissions.entries(&b.workspace).unwrap();
    assert_eq!(listed.len(), 1, "a revoked entry stays listed");
    assert_eq!(listed[0].revoked_at, Some(3000.0));
}

/// CF1 and X4 in memory, against a test `PATH` of planted programs:
/// `cargo test; calc` never matches a `cargo test` entry; neither does a
/// longer argv, another program of the same name, another folder, or the
/// entry seen from a folder that only shares the workspace's id.
#[test]
fn cf1_x4_matching_is_exact() {
    let mut b = Bench::new("run-cf1");
    let tools = b.plant("tools", &["cargo.exe"]);
    let other = b.plant("other", &["cargo.exe"]);
    b.with_path(&[&tools]);
    let resolve = |b: &Bench, name: &str| {
        resolve_program(name, &b.env, Some(&b.workspace.root), &b.globals).ok()
    };
    let candidate = Candidate {
        argv: vec!["cargo".into(), "test".into()],
        resolved: resolve(&b, "cargo").unwrap(),
        cwd: String::new(),
    };
    let entry =
        block_on(
            b.permissions
                .allow_always(&b.confirmer, &b.workspace, &candidate, "allow:x"),
        )
        .unwrap();
    let matches = |b: &Bench, text: &str, cwd: &str| {
        b.permissions
            .matching(&b.workspace, text, cwd, |name| resolve(b, name))
            .map(|found| found.id)
    };
    assert_eq!(matches(&b, "cargo test", ""), Some(entry.id.clone()));
    assert_eq!(
        matches(&b, "Cargo  test", ""),
        Some(entry.id.clone()),
        "the program by path"
    );
    for text in [
        "cargo test; calc",
        "cargo test;calc",
        "cargo test & calc",
        "cargo test --release",
        "cargo test calc",
        "cargo",
        "cargo tests",
        "cargo TEST",
    ] {
        assert_eq!(matches(&b, text, ""), None, "{text:?}");
    }
    assert_eq!(
        matches(&b, "cargo test", "sub"),
        None,
        "X4: the folder must be equal"
    );
    let prepared = b.prepare("cargo test; calc").unwrap();
    assert!(matches!(prepared.verdict, Verdict::Ask(AskKind::Command)));
    assert!(!prepared.allow_always_offer);
    // Another cargo.exe first on PATH: not the entry's program.
    b.with_path(&[&other, &tools]);
    assert_eq!(matches(&b, "cargo test", ""), None);
    b.with_path(&[&tools]);
    // A folder with the same id at another path (moved or replaced).
    let mut moved = b.workspace.clone();
    moved.root = b.scratch.path().join("elsewhere");
    assert_eq!(
        b.permissions
            .matching(&moved, "cargo test", "", |name| resolve(&b, name)),
        None
    );
    assert_eq!(b.permissions.entries(&moved).unwrap(), vec![]);
}

/// CF4, CF5 and CF12 (the entry side), against planted programs:
/// interpreters (by the resolved file's stem, and through a link to one)
/// and batch-file shims are never offered "Allow always" and are refused as
/// entries without a dialog; a planted `cargo.exe` in the workspace is not
/// the program found, even with the workspace first on `PATH`.
#[test]
fn cf4_cf5_cf12_interpreters_and_shims_are_never_entries() {
    let mut b = Bench::new("run-cf4");
    let tools = b.plant(
        "tools",
        &[
            "python3.12.exe",
            "pythonw.exe",
            "dotnet.exe",
            "schtasks.exe",
            "ssh.exe",
            "npm.cmd",
            "npm.exe",
            "build.bat",
            "lint.ps1",
            "lint.exe",
            "cargo.exe",
            "PowerShell.EXE",
        ],
    );
    let py = b.plant("py", &["python.exe"]);
    std::os::windows::fs::symlink_file(py.join("python.exe"), tools.join("helper.exe")).unwrap();
    std::fs::write(b.folder.join("cargo.exe"), b"").unwrap();
    let folder = b.folder.clone();
    b.with_path(&[&folder, &tools]);
    for text in [
        "python3.12 -m pip list",
        "pythonw -c x",
        "dotnet build",
        "schtasks /query",
        "ssh host",
        "PowerShell.EXE -NoProfile",
        "helper --version",
        "npm test",
        "build x",
        "lint src",
    ] {
        let prepared = b.prepare(text).unwrap();
        assert!(!prepared.allow_always_offer, "{text:?}");
        assert!(prepared.candidate.is_none(), "{text:?}");
    }
    let resolved = resolve_program("helper", &b.env, Some(&b.workspace.root), &b.globals).unwrap();
    assert!(
        resolved
            .real
            .to_string_lossy()
            .to_lowercase()
            .ends_with("python.exe")
    );
    let refused = block_on(b.permissions.allow_always(
        &b.confirmer,
        &b.workspace,
        &Candidate {
            argv: vec!["helper".into(), "--version".into()],
            resolved,
            cwd: String::new(),
        },
        "allow:helper",
    ));
    assert!(matches!(refused, Err(PermissionError::NotAllowed(_))));
    assert!(
        b.confirm.asked().is_empty(),
        "no dialog for a refused entry"
    );
    let cargo = b.prepare("cargo build").unwrap();
    let found = cargo.candidate.unwrap().resolved.path;
    assert!(found.starts_with(&tools), "{}", found.display());
    assert!(b.launched().is_empty());
}

/// CF6: while a staged change waits, a standing match becomes a question
/// (`GateStaged`), the card says how many wait, and `Approve` is refused
/// with a conflict and no dialog; once the change is reviewed, the entry
/// runs.
#[test]
fn cf6_staged_changes_shut_the_command_gate() {
    let b = Bench::new("run-cf6");
    let prepared = b.prepare("where.exe cmd.exe").unwrap();
    let entry = block_on(b.permissions.allow_always(
        &b.confirmer,
        &b.workspace,
        &prepared.candidate.clone().unwrap(),
        "allow:call_6",
    ))
    .unwrap();
    let asked = b.confirm.asked().len();
    b.stage_edit();
    let gated = b.prepare("where.exe cmd.exe").unwrap();
    assert_eq!(
        gated.verdict,
        Verdict::Ask(AskKind::GateStaged { waiting: 1 })
    );
    let ApprovalDetail::Command {
        staged_waiting,
        mode,
        ..
    } = gated.detail.clone()
    else {
        panic!("a command card");
    };
    assert_eq!((staged_waiting, mode), (1, CommandMode::Direct));
    let refused = block_on(approve(&b.ctx(), &gated, &"call_6".to_owned())).unwrap_err();
    assert_eq!(refused, Refused::conflict("Review 1 staged change first."));
    // A one-time command too.
    let once = b.prepare("Write-Output x").unwrap();
    assert!(block_on(approve(&b.ctx(), &once, &"call_6b".to_owned())).is_err());
    assert_eq!(
        b.confirm.asked().len(),
        asked,
        "no dialog while the gate is shut"
    );
    assert!(b.launched().is_empty());
    // The reader undoes the change: the entry runs, directly.
    let mut snapshot = b.staging.snapshot(&b.staging.changes()[0].id).unwrap();
    snapshot.change.state = lattice_protocol::conversation::ChangeState::Undone;
    b.staging.update(snapshot).unwrap();
    let approval = block_on(approve(&b.ctx(), &gated, &"call_6".to_owned())).unwrap();
    assert_eq!(approval.entry(), Some(entry.id.as_str()));
}

/// X11: the lease held elsewhere, or a checkpoint that cannot be taken, and
/// the command does not start.
#[test]
fn x11_no_lease_or_no_checkpoint_no_command() {
    let b = Bench::new("run-x11");
    let prepared = b.prepare("Write-Output x").unwrap();
    let approval = block_on(approve(&b.ctx(), &prepared, &"c1".to_owned())).unwrap();
    b.hooks.elsewhere.store(true, Ordering::SeqCst);
    let refused = run(
        &b.ctx(),
        &prepared,
        &approval,
        &"c1".to_owned(),
        &StopHandle::default(),
    )
    .unwrap_err();
    assert_eq!(
        refused.sentence,
        "Another Lattice agent is editing this folder."
    );
    b.hooks.elsewhere.store(false, Ordering::SeqCst);
    b.hooks.before_fails.store(true, Ordering::SeqCst);
    let refused = run(
        &b.ctx(),
        &prepared,
        &approval,
        &"c1".to_owned(),
        &StopHandle::default(),
    )
    .unwrap_err();
    assert_eq!(
        refused.sentence,
        "Lattice could not record the folder's state first, so the command did not run."
    );
    assert!(b.launched().is_empty());
    assert!(!b.slots.running(&b.workspace.id), "the slot was given back");
}

/// X14 (CF13's command part): while one command runs in the folder, a
/// second is refused at the card, at Approve and at the run; a Keep would
/// be refused (the review asks these slots). Stop ends the first at once.
#[test]
fn x14_a_second_command_in_the_folder_is_refused() {
    let b = Bench::new("run-x14");
    let prepared = b.prepare("Start-Sleep -Seconds 30").unwrap();
    let second = b.prepare("Write-Output second").unwrap();
    let call = "call_long".to_owned();
    let approval = block_on(approve(&b.ctx(), &prepared, &call)).unwrap();
    let second_approval = block_on(approve(&b.ctx(), &second, &"call_2".to_owned())).unwrap();
    let stop = StopHandle::default();
    std::thread::scope(|scope| {
        let first = scope.spawn(|| run(&b.ctx(), &prepared, &approval, &call, &stop));
        let deadline = Instant::now() + Duration::from_secs(20);
        while !(b.slots.running(&b.workspace.id) && !b.launched().is_empty()) {
            assert!(Instant::now() < deadline, "the first command started");
            std::thread::sleep(Duration::from_millis(20));
        }
        let sentence = "Another command is still running in this folder.";
        assert_eq!(b.prepare("Write-Output third").unwrap_err().0, sentence);
        let refused = block_on(approve(&b.ctx(), &second, &"call_2".to_owned())).unwrap_err();
        assert_eq!(refused, Refused::conflict(sentence));
        let refused = run(
            &b.ctx(),
            &second,
            &second_approval,
            &"call_2".to_owned(),
            &StopHandle::default(),
        )
        .unwrap_err();
        assert_eq!(refused, Refused::conflict(sentence));
        let stopped_at = Instant::now();
        stop.stop();
        let outcome = first.join().unwrap().unwrap();
        assert_eq!(outcome.reason, ExitReason::Stopped);
        assert!(outcome.model_text.ends_with("Stopped by the user"));
        assert!(stopped_at.elapsed() < Duration::from_secs(10));
    });
    assert_eq!(b.launched().len(), 1, "the second never started");
    assert!(!b.slots.running(&b.workspace.id));
}

/// X9 and X8 against a real command: more than 32 KiB of coloured output
/// reaches the model as its first 8 KiB and last 24 KiB, stripped, with
/// the exit code; the whole output is one redacted blob in the record; a
/// timeout ends the command and says so.
#[test]
fn x9_output_and_the_timeout() {
    let b = Bench::new("run-x9");
    let outcome = b
        .run_text(
            "1..4000 | ForEach-Object { \"$([char]27)[31mline $_$([char]27)[0m\" }; exit 3",
            "call_9",
        )
        .unwrap();
    assert_eq!(outcome.code, Some(3));
    assert!(
        outcome.model_text.starts_with("line 1\r\n") || outcome.model_text.starts_with("line 1\n"),
        "{}",
        &outcome.model_text[..30]
    );
    assert!(outcome.model_text.contains("bytes omitted"));
    assert!(outcome.model_text.contains("line 4000"));
    assert!(outcome.model_text.contains("Exit code 3 after"));
    assert!(!outcome.model_text.contains('\u{1b}'));
    assert!(outcome.lines >= 4000);
    let blob = outcome.output_blob.clone().unwrap();
    let kept = b.staging.sidecar().read_blob(&blob).unwrap();
    assert_eq!(kept.len() as u64, outcome.bytes);
    let started = Instant::now();
    let mut args = Bench::args("Start-Sleep -Seconds 30");
    args.timeout_s = Some(1);
    let prepared = prepare(&b.ctx(), &args).unwrap();
    let approval = block_on(approve(&b.ctx(), &prepared, &"call_t".to_owned())).unwrap();
    let outcome = run(
        &b.ctx(),
        &prepared,
        &approval,
        &"call_t".to_owned(),
        &StopHandle::default(),
    )
    .unwrap();
    assert_eq!(outcome.reason, ExitReason::TimedOut);
    assert!(
        outcome.model_text.contains("Timed out after 1."),
        "{}",
        outcome.model_text
    );
    assert!(started.elapsed() < Duration::from_secs(15));
}
