//! Starting the agent's browser: Microsoft Edge (or Google Chrome) with a
//! profile of its own, in a Job Object, its DevTools on two pipes it inherits
//! (no TCP socket listens). Not a port.
//!
//! - **The program** is found where its installer puts it (Edge first, as
//!   every Windows PC has it), never on a `PATH` and never in a folder the
//!   agent works in.
//! - **The profile** is `<globals>/lattice_native/browser/profile`: the
//!   sign-ins the reader makes there stay there, apart from their own
//!   browser's, and survive Lattice's restarts.
//! - **The arguments**: that profile, DevTools over pipes
//!   (`--remote-debugging-pipe`: the browser reads requests on its C runtime
//!   descriptor 3 and writes on 4, which
//!   `lattice_sys::process::spawn_with_fd_pipes` gives it), no first-run
//!   pages, and a window of a set size (or none, for the tests:
//!   `--headless=new`). No `--remote-debugging-port`: no socket listens, and
//!   no `DevToolsActivePort` file is written.
//! - **The environment** is X7's block plus the names a browser needs to find
//!   the system's folders ([`BROWSER_ENV_NAMES`]); none of them holds a key.
//! - **The Job** ([`LIMITS`]) holds the browser and every process it starts:
//!   closing Lattice, or Stop, ends them all.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use lattice_sys::process::{Child, FdPipes, JobLimits, SpawnRequest, spawn_with_fd_pipes};

use crate::env::Env;
use crate::exec::spawn::child_environment;
use crate::state::StateRoot;

/// A browser starts many processes (one per site, its GPU and network
/// services, utilities): more, and more memory, than a command's Job allows.
pub const LIMITS: JobLimits = JobLimits {
    active_processes: 512,
    job_memory: 16 * 1024 * 1024 * 1024,
};

/// The names beside X7's that a browser reads to find the system's folders.
pub const BROWSER_ENV_NAMES: [&str; 14] = [
    "SystemDrive",
    "ProgramData",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramW6432",
    "CommonProgramFiles",
    "CommonProgramFiles(x86)",
    "ALLUSERSPROFILE",
    "PUBLIC",
    "HOMEDRIVE",
    "HOMEPATH",
    "USERNAME",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
];

/// How long the browser has to answer its first DevTools request.
pub const START_WAIT: Duration = Duration::from_secs(30);

/// `<globals>/lattice_native/browser/profile`.
pub fn profile_dir(state: &StateRoot) -> PathBuf {
    state
        .globals
        .join("lattice_native")
        .join("browser")
        .join("profile")
}

/// The installed browser, Edge first.
pub fn find_browser(env: &dyn Env) -> Option<PathBuf> {
    installed_browsers(env).into_iter().next()
}

/// Every installed browser [`find_browser`] would take, in its order.
pub fn installed_browsers(env: &dyn Env) -> Vec<PathBuf> {
    let var = |name: &str| env.var(name).map(PathBuf::from);
    let mut candidates = Vec::new();
    for base in [
        var("ProgramFiles(x86)"),
        var("ProgramFiles"),
        var("LOCALAPPDATA"),
    ]
    .into_iter()
    .flatten()
    {
        candidates.push(base.join(r"Microsoft\Edge\Application\msedge.exe"));
    }
    for base in [
        var("ProgramFiles"),
        var("ProgramFiles(x86)"),
        var("LOCALAPPDATA"),
    ]
    .into_iter()
    .flatten()
    {
        candidates.push(base.join(r"Google\Chrome\Application\chrome.exe"));
    }
    candidates.retain(|path| path.is_file());
    candidates
}

/// The browser's arguments, its own name first.
pub fn arguments(
    program: &Path,
    profile: &Path,
    headless: bool,
    size: (u32, u32),
) -> Vec<OsString> {
    let mut argv: Vec<OsString> = vec![program.as_os_str().to_owned()];
    let mut add = |arg: String| argv.push(OsString::from(arg));
    add(format!("--user-data-dir={}", profile.display()));
    add("--remote-debugging-pipe".to_owned());
    add("--no-first-run".to_owned());
    add("--no-default-browser-check".to_owned());
    add("--disable-sync".to_owned());
    add(format!("--window-size={},{}", size.0, size.1 + 120));
    if headless {
        add("--headless=new".to_owned());
    }
    add("about:blank".to_owned());
    argv
}

/// X7's block and [`BROWSER_ENV_NAMES`], from Lattice's own environment.
pub fn environment(env: &dyn Env, globals: &Path) -> Vec<(OsString, OsString)> {
    let mut block = child_environment(env, None, globals);
    for name in BROWSER_ENV_NAMES {
        if block
            .iter()
            .any(|(have, _)| have.to_string_lossy().eq_ignore_ascii_case(name))
        {
            continue;
        }
        if let Some(value) = env.var(name) {
            block.push((OsString::from(name), value));
        }
    }
    block
}

/// A started browser: its process tree and its DevTools.
pub struct Started {
    pub child: Child,
    /// This process's ends of the browser's DevTools pipes: it reads
    /// requests from `to_child` and writes answers and events to
    /// `from_child`.
    pub pipes: FdPipes,
    /// The last lines it wrote to stdout and stderr.
    pub log: Arc<Mutex<Vec<String>>>,
}

fn drain(name: &'static str, stream: impl Read + Send + 'static, log: Arc<Mutex<Vec<String>>>) {
    let _ = std::thread::Builder::new()
        .name(format!("lattice-browser-{name}"))
        .spawn(move || {
            for line in BufReader::new(stream).lines() {
                let Ok(line) = line else { break };
                let mut log = log.lock().unwrap_or_else(|p| p.into_inner());
                if log.len() >= 50 {
                    log.remove(0);
                }
                log.push(line.chars().take(400).collect());
            }
        });
}

/// Start `program` on `profile` with its DevTools on two pipes. It returns
/// once the browser is started: whether it answers on them is the first
/// request's to find out ([`start_failure`] words a browser that does not).
/// Blocking.
pub fn start(
    program: &Path,
    profile: &Path,
    headless: bool,
    size: (u32, u32),
    env: &dyn Env,
    globals: &Path,
) -> Result<Started, String> {
    std::fs::create_dir_all(profile)
        .map_err(|_| "The browser's profile folder could not be made.".to_owned())?;
    start_with(
        program,
        &arguments(program, profile, headless, size),
        env,
        globals,
    )
}

/// Start `program` with `argv` (its own name first, a `--user-data-dir`
/// among them) as [`start`] does: the preview browser's own arguments
/// (`super::preview`). Blocking.
pub fn start_with(
    program: &Path,
    argv: &[OsString],
    env: &dyn Env,
    globals: &Path,
) -> Result<Started, String> {
    let block = environment(env, globals);
    let cwd = program
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut child = spawn_with_fd_pipes(&SpawnRequest {
        program,
        argv,
        cwd: &cwd,
        env: &block,
        limits: LIMITS,
    })
    .map_err(|error| format!("The browser could not start: {error}."))?;
    let pipes = child
        .take_fd_pipes()
        .ok_or_else(|| "The browser's DevTools pipes were not made.".to_owned())?;
    let log = Arc::new(Mutex::new(Vec::new()));
    if let Some(out) = child.take_stdout() {
        drain("out", out, log.clone());
    }
    if let Some(err) = child.take_stderr() {
        drain("err", err, log.clone());
    }
    Ok(Started { child, pipes, log })
}

/// Why a started browser did not answer its first DevTools request within
/// `waited` (`why` the request's own words, `open` whether its connection is
/// still open), after waiting a moment for it to end; its tree is ended if it
/// still runs. Blocking.
pub fn start_failure(child: Child, why: &str, open: bool, waited: Duration) -> String {
    let ended = child.wait(Some(Duration::from_secs(5))).ok().flatten();
    drop(child);
    match ended {
        Some(code) => format!(
            "The browser ended as it started (exit code {code}): another copy may be using its profile."
        ),
        None if open => format!(
            "The browser did not answer on its DevTools pipe within {} s.",
            waited.as_secs()
        ),
        None => why.to_owned(),
    }
}
