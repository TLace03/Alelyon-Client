//! TF5 and CF16: what a child of the chat core can and cannot see (spec §7.6
//! X7, §10.3 TF5, §16.4 CF16).
//!
//! - **TF5.** A middle process plays Lattice: this test binary, started with
//!   `ANTHROPIC_API_KEY` and `ALELYON_SECRET` set to sentinels in its own
//!   environment. It spawns `cmd.exe /d /c set` through the chat core's spawn,
//!   with the X7 block built from its real environment, and records what the
//!   child printed. Neither sentinel may appear; the middle also records its own
//!   view of the sentinels, which shows the test would see a leak.
//! - **CF16.** A workspace holds a planted `tool.exe` (a link to this test
//!   binary, which writes a marker when it runs under that name). `cmd.exe /d /c
//!   tool …` runs in the workspace, with the workspace also on the `PATH` the
//!   block is built from. With the X7 block the planted program does not run.
//!   Two positive controls show each half of X7 matters: the unfiltered `PATH`
//!   runs it, and so does a block without `NoDefaultCurrentDirectoryInExePath`.
//!
//! The roles below are ignored in a normal run and act only when their folder
//! (or their own file name) says they were started by these tests. Every child
//! here is `cmd.exe` or this test binary; none reaches the network.

#![cfg(windows)]

use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use lattice_core::env::{Env, MapEnv, ProcessEnv};
use lattice_core::exec::spawn::{ChildSpec, child_environment, spawn};
use lattice_sys::process::{Exit, JobLimits, SpawnRequest};

const SENTINEL_KEY: &str = "lattice-tf5-sentinel-anthropic";
const SENTINEL_SECRET: &str = "lattice-tf5-sentinel-alelyon";
const MIDDLE_MARKER: &str = "lattice-tf5-middle.txt";
const WORKSPACE_MARKER: &str = "lattice-cf16-workspace.txt";

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "lattice-core-spawn-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        // Not canonicalised: cmd.exe refuses a verbatim (`\\?\`) working folder
        // and falls back to the Windows folder.
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Only the test's own temporary folder.
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn system32() -> PathBuf {
    let root = std::env::var_os("SystemRoot").expect("SystemRoot is set on Windows");
    PathBuf::from(root).join("System32")
}

fn cmd() -> PathBuf {
    system32().join("cmd.exe")
}

/// Run `cmd.exe /d /c <line>` in `cwd` with `block`; its stdout and exit code.
fn run_cmd(line: &[&str], cwd: &Path, block: &[(OsString, OsString)]) -> (String, u32) {
    let program = cmd();
    let mut argv: Vec<OsString> = vec![program.clone().into(), "/d".into(), "/c".into()];
    argv.extend(line.iter().map(OsString::from));
    let mut child = lattice_sys::process::spawn(&SpawnRequest {
        program: &program,
        argv: &argv,
        cwd,
        env: block,
        limits: JobLimits::default(),
    })
    .unwrap();
    let mut stderr = child.take_stderr().unwrap();
    let errors = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    let mut out = String::new();
    child
        .take_stdout()
        .unwrap()
        .read_to_string(&mut out)
        .unwrap();
    out.push_str(&errors.join().unwrap());
    match child.wait_or_kill(Duration::from_secs(60)).unwrap() {
        Exit::Exited(code) => (out, code),
        Exit::TimedOut => panic!("cmd.exe did not finish"),
    }
}

// ----------------------------------------------------------------------- TF5

#[test]
#[ignore = "runs only as the middle process of children_inherit_no_secrets"]
fn role_tf5_middle() {
    let cwd = std::env::current_dir().unwrap();
    if !cwd.join(MIDDLE_MARKER).is_file() {
        return;
    }
    let env = ProcessEnv;
    let seen = |name: &str| {
        env.var(name)
            .map(|value| value.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    std::fs::write(
        cwd.join("middle_saw.txt"),
        format!(
            "{}\n{}\n",
            seen("ANTHROPIC_API_KEY"),
            seen("ALELYON_SECRET")
        ),
    )
    .unwrap();
    let program = cmd();
    let spec = ChildSpec {
        program: program.clone(),
        argv: vec![program.into(), "/d".into(), "/c".into(), "set".into()],
        cwd: cwd.clone(),
        limits: JobLimits::default(),
    };
    let globals = cwd.join("globals");
    let mut child = spawn(&spec, &env, None, &globals).unwrap();
    let mut out = String::new();
    child
        .take_stdout()
        .unwrap()
        .read_to_string(&mut out)
        .unwrap();
    assert_eq!(
        child.wait_or_kill(Duration::from_secs(60)).unwrap(),
        Exit::Exited(0)
    );
    std::fs::write(cwd.join("child_env.txt"), out).unwrap();
}

#[test]
fn children_inherit_no_secrets() {
    let scratch = Scratch::new("tf5");
    std::fs::write(scratch.path().join(MIDDLE_MARKER), "middle").unwrap();
    // Its output is captured, so its own test summary stays out of this run's.
    let middle = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--ignored", "--exact", "role_tf5_middle"])
        .current_dir(scratch.path())
        .env("ANTHROPIC_API_KEY", SENTINEL_KEY)
        .env("ALELYON_SECRET", SENTINEL_SECRET)
        .output()
        .unwrap();
    assert!(
        middle.status.success(),
        "the middle process failed: {}",
        String::from_utf8_lossy(&middle.stdout)
    );
    // The positive control: Lattice's own environment holds both sentinels.
    let middle = std::fs::read_to_string(scratch.path().join("middle_saw.txt")).unwrap();
    assert_eq!(middle, format!("{SENTINEL_KEY}\n{SENTINEL_SECRET}\n"));
    let child = std::fs::read_to_string(scratch.path().join("child_env.txt")).unwrap();
    println!("the child's environment names: {:?}", names(&child));
    assert!(!child.contains(SENTINEL_KEY), "the key reached the child");
    assert!(
        !child.contains(SENTINEL_SECRET),
        "the secret reached the child"
    );
    for name in ["ANTHROPIC_API_KEY", "ALELYON_SECRET"] {
        assert!(!names(&child).contains(&name.to_uppercase()), "{name}");
    }
    // What does reach it: the allowlist and the native addition.
    let names = names(&child);
    for name in ["PATH", "SYSTEMROOT", "NODEFAULTCURRENTDIRECTORYINEXEPATH"] {
        assert!(
            names.contains(&name.to_owned()),
            "{name} missing: {names:?}"
        );
    }
    let allowed: Vec<String> = lattice_core::exec::spawn::SAFE_AGENT_ENV_NAMES
        .iter()
        .chain(["NoDefaultCurrentDirectoryInExePath"].iter())
        .map(|name| name.to_uppercase())
        .collect();
    // cmd.exe adds its own few (PROMPT, and `=X:` drive folders it keeps).
    let cmd_own = ["PROMPT"];
    for name in &names {
        assert!(
            allowed.contains(name) || cmd_own.contains(&name.as_str()),
            "{name} reached the child"
        );
    }
}

/// The upper-cased names in `set` output (`NAME=value` lines; cmd's hidden
/// `=X:` entries are not printed by `set`).
fn names(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(name, _)| name.to_uppercase())
        .collect()
}

// ---------------------------------------------------------------------- CF16

#[test]
#[ignore = "runs only as the planted tool.exe of a_planted_program_in_the_workspace_never_runs"]
fn role_planted_tool() {
    let exe = std::env::current_exe().unwrap();
    if !exe
        .file_name()
        .is_some_and(|name| name.eq_ignore_ascii_case("tool.exe"))
    {
        return;
    }
    let cwd = std::env::current_dir().unwrap();
    if cwd.join(WORKSPACE_MARKER).is_file() {
        std::fs::write(cwd.join("tool-ran.txt"), "the planted program ran").unwrap();
    }
}

/// A workspace with the planted `tool.exe`.
fn planted_workspace(tag: &str) -> Scratch {
    let scratch = Scratch::new(tag);
    std::fs::write(
        scratch.path().join(WORKSPACE_MARKER),
        "a planted program may write here",
    )
    .unwrap();
    let planted = scratch.path().join("tool.exe");
    let me = std::env::current_exe().unwrap();
    if std::fs::hard_link(&me, &planted).is_err() {
        std::fs::copy(&me, &planted).unwrap();
    }
    scratch
}

/// Whether `cmd /d /c tool …` in `workspace` ran the planted program.
fn planted_ran(workspace: &Scratch, block: &[(OsString, OsString)]) -> bool {
    let (output, code) = run_cmd(
        &["tool", "--ignored", "--exact", "role_planted_tool"],
        workspace.path(),
        block,
    );
    let ran = workspace.path().join("tool-ran.txt").exists();
    println!("cmd exit {code}, ran {ran}: {}", output.trim());
    ran
}

fn environment_with_workspace_on_path(workspace: &Path) -> MapEnv {
    let process = ProcessEnv;
    let mut env = MapEnv::new()
        .with(
            "PATH",
            format!("{};{}", workspace.display(), system32().display()),
        )
        .with("PATHEXT", ".COM;.EXE;.BAT;.CMD");
    for name in ["SystemRoot", "windir", "COMSPEC", "TEMP", "TMP"] {
        if let Some(value) = process.var(name) {
            env.set(name, value);
        }
    }
    env
}

#[test]
fn a_planted_program_in_the_workspace_never_runs() {
    let globals = Scratch::new("cf16-globals");
    // The real block: PATH filtered, the current folder not searched.
    let workspace = planted_workspace("cf16-real");
    let env = environment_with_workspace_on_path(workspace.path());
    let block = child_environment(&env, Some(workspace.path()), globals.path());
    let path = block
        .iter()
        .find(|(name, _)| name == "PATH")
        .unwrap()
        .1
        .to_string_lossy()
        .into_owned();
    assert!(
        !path
            .to_lowercase()
            .contains(&workspace.path().to_string_lossy().to_lowercase())
    );
    let real = planted_ran(&workspace, &block);

    // Positive control 1: the parent's PATH passed unchanged.
    let workspace_a = planted_workspace("cf16-path");
    let env_a = environment_with_workspace_on_path(workspace_a.path());
    let mut unfiltered = child_environment(&env_a, Some(workspace_a.path()), globals.path());
    for (name, value) in &mut unfiltered {
        if name == "PATH" {
            *value = env_a.var("PATH").unwrap();
        }
    }
    let through_path = planted_ran(&workspace_a, &unfiltered);

    // Positive control 2: PATH filtered, but the current folder searched.
    let workspace_b = planted_workspace("cf16-cwd");
    let env_b = environment_with_workspace_on_path(workspace_b.path());
    let searched: Vec<(OsString, OsString)> =
        child_environment(&env_b, Some(workspace_b.path()), globals.path())
            .into_iter()
            .filter(|(name, _)| name != "NoDefaultCurrentDirectoryInExePath")
            .collect();
    let through_cwd = planted_ran(&workspace_b, &searched);

    println!(
        "planted tool.exe ran: X7 block {real}, unfiltered PATH {through_path}, \
         current folder searched {through_cwd}"
    );
    assert!(through_path, "control: the workspace on PATH must run it");
    assert!(
        through_cwd,
        "control: searching the current folder must run it"
    );
    assert!(!real, "the X7 block ran the planted program");
}
