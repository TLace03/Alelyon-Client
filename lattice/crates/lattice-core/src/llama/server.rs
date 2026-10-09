//! The managed llama.cpp server: the one this process runs (the chat core's spec
//! §22 LR2, LR6, LR7, LR10; ADR-0041 decision 3). A port of
//! `llama_server.ManagedServer` and `Manager`, started through
//! the core's own spawn (B7) and stopped by a single idle timer instead of
//! Python's polling watcher.
//!
//! **Launch (LR6).** The command line is Python's `ManagedServer.command` plus
//! an explicit `--jinja` ([`command`]): `<binary> --host 127.0.0.1 --port <p>
//! -m <gguf> --alias <stem> -c <ctx> -ngl <layers> -np <parallel>
//! --api-key-file <file> --no-webui --offline --jinja`. llama.cpp's flag for
//! its experimental built-in tools is never passed: those tools would run
//! inside the server, outside the approval path, and a source guard
//! (`tests/llama_guard.rs`) holds every source to its absence. The
//! environment is X7's block ([`crate::exec::spawn::child_environment`]) with
//! every `LLAMA_ARG_*` and `OLLAMA_*` name removed ([`server_environment`]):
//! llama.cpp reads `LLAMA_ARG_*` as defaults for its flags. To it are added
//! the Vulkan loader's device-selection names ([`VULKAN_DEVICE_NAMES`]),
//! each only when it is set in the core's own environment (LR6′, amendment
//! 1b): X7 drops every `VK_*` name, and these are how a person picks the GPU
//! the server uses (the house rule, `VK_LOADER_DEVICE_ID_FILTER`, keeps it
//! on one card). No other `VK_*` name passes. The binary is
//! started by its absolute path, never searched for, and a path holding
//! `ollama` is refused before anything starts (LR3). The process runs in a
//! Job Object that ends it with the core (B7's spawn): dropping the runtime,
//! or the core's process ending, ends the server.
//!
//! **The token (LR2, LR6, LR6a).** A random token (two version-4 UUIDs: 244
//! bits from the operating system's generator) is made for each launch.
//! It is written to a file in a fresh, private folder of the core's own
//! temporaries (`fsx::create_private_temporary_dir`): its name is random, so
//! it cannot be pre-created, and the folder and the file each carry an
//! explicit, protected DACL of SYSTEM, Administrators and the user, inheriting
//! nothing from the temporary root (CPython's `mkdtemp` does the same). The
//! root is the first usable one of `TMPDIR`, `TEMP`, `TMP`,
//! `%LOCALAPPDATA%\Temp` and the user's profile ([`temp_roots`]): absolute,
//! on a local drive, an existing folder reached through local links only, on
//! a volume that keeps access-control lists (`FILE_PERSISTENT_ACLS`: not FAT
//! or exFAT, where the private DACL would not be kept), and one where the
//! folder can be made. The file is passed by its path, so the
//! token is never on a command line. llama.cpp reads it only at start-up, so
//! it is removed, with its folder (`fsx::remove_own_temporary`,
//! `remove_own_temporary_dir`), as soon as `/health` first answers; and in
//! any case whenever the server stops: a stop, an idle stop, a crash, a
//! failed start, a model switch, a shutdown, or the runtime being dropped.
//! The token reaches only request headers.
//!
//! **Lifecycle (LR7, LR10).** One server per process, serving one model.
//! [`LlamaRuntime::open`] starts it on the first Local turn (or restarts it
//! for another model, or after it died) and returns a [`Lease`]: while any
//! lease is alive the server is in use. Readiness polls `/health` (2 s a
//! request, 250 ms apart) for up to 180 s, and fails at once if the process
//! exits. It sends nothing to the port until the system's listener table
//! (`GetExtendedTcpTable`, `lattice_sys::net`) shows `127.0.0.1:<port>`
//! held by the child's own process id, and it reads the owning process again after
//! `/health` answers (LR7a): the port is chosen free and released before the
//! server binds it, so another process could take it and answer. A listener
//! of any other process stops the child, and the start is tried once more on
//! a new port. When the last lease is dropped, one timer is armed for the settings'
//! `idle_seconds`; a new lease cancels it, and when it fires with no lease
//! taken the server stops, giving its VRAM back (Angel shares the RX); that
//! stop holds the start mutex until the old process has ended and its token
//! is gone, so a Local turn arriving meanwhile waits rather than starting a
//! second server beside one still closing (LR7b). There
//! is no polling loop: an idle core with a server running schedules exactly
//! one sleeping timer (CB1). The log is `~/.alelyon/llama/logs/<stem>.log`,
//! appended by two reader threads that end with the process; the end of its
//! output is how a crash is noticed, without a timer.
//!
//! **Requests (LR2).** `/health` and `/props` go through [`HttpGet`]
//! (`net.rs`): 127.0.0.1 only, no proxy, no redirect. The model client of a
//! turn is built from the [`ManagedEndpoint`] by `choices` and `models`.

use std::ffi::{OsStr, OsString};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::time::Duration;

use futures::FutureExt;
use futures::future::BoxFuture;
use lattice_agents::SecretString;
use lattice_sys::process::{Child, JobLimits, SpawnRequest};
use serde_json::Value;
use tokio::runtime::Handle;
use tokio::task::AbortHandle;

use super::files::{LlamaPaths, LocalModel, find_binary};
use super::settings::{self, Settings};
use super::{Lease, LlamaError, ManagedEndpoint, ManagedRuntime, Opened};
use crate::env::{self, Env};
use crate::exec::spawn::child_environment;
use crate::fsx;
use crate::localfs::{self, LinkRule};
use crate::net::{self, HttpGet, HttpRequest, LOOPBACK, LoopbackHttp, NetError};
use crate::state::{self, Platform, StateRoot};
use lattice_sys::fs::Access;

/// How long a start may take: a 20 GB model takes a while to map and upload.
pub const START_TIMEOUT: Duration = Duration::from_secs(180);
/// One `/health` request.
pub const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);
/// Between two `/health` requests while the server starts.
pub const POLL_GAP: Duration = Duration::from_millis(250);
/// How long a stop waits for the process to end.
pub const STOP_TIMEOUT: Duration = Duration::from_secs(10);
/// One `/props` request.
pub const PROPS_TIMEOUT: Duration = Duration::from_secs(5);
/// Names removed from the server's environment (LR6).
pub const STRIPPED_ENV_PREFIXES: [&str; 2] = ["LLAMA_ARG_", "OLLAMA_"];
/// The Vulkan loader's (and ggml's) device-selection names: passed to the
/// server, beyond X7's block, each only when it is set in the core's own
/// environment (LR6′). Exactly these; no other `VK_*` name.
pub const VULKAN_DEVICE_NAMES: [&str; 6] = [
    "VK_LOADER_DEVICE_ID_FILTER",
    "VK_LOADER_DRIVERS_SELECT",
    "VK_LOADER_DRIVERS_DISABLE",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "GGML_VK_VISIBLE_DEVICES",
];
/// The flag that turns on llama.cpp's Jinja chat templates, which tool
/// calling needs (LR8). The installed build has it on by default; it is
/// passed anyway, so the command line says what the server does.
pub const JINJA: &str = "--jinja";
/// The server's Job: a few processes (the server and its console host) and
/// room for a large model's host buffers.
pub const SERVER_LIMITS: JobLimits = JobLimits {
    active_processes: 8,
    job_memory: 256 * 1024 * 1024 * 1024,
};
/// What the status line says after the server ended on its own.
const STOPPED_ON_ITS_OWN: &str = "llama.cpp's server stopped on its own.";

/// What the runtime is built from.
#[derive(Clone)]
pub struct RuntimeConfig {
    /// The environment the child's X7 block is built from.
    pub env: Arc<dyn Env>,
    /// `<globals>`: X7 drops it from the child's `PATH`.
    pub globals: PathBuf,
    pub paths: LlamaPaths,
    /// Where the token's folder may be made, in order ([`temp_roots`]); the
    /// first usable one is taken.
    pub temp_roots: Vec<PathBuf>,
    pub http: Arc<dyn HttpGet>,
    /// Picks each launch's loopback port ([`net::free_loopback_port`]; a
    /// test hands one that is taken).
    pub ports: Arc<dyn Fn() -> std::io::Result<u16> + Send + Sync>,
    pub start_timeout: Duration,
    pub health_timeout: Duration,
    pub poll_gap: Duration,
    pub stop_timeout: Duration,
    pub limits: JobLimits,
    /// Called when a server's close has finished: its process ended and its
    /// token removed. `None` in a shipped build; a test holds a close open
    /// with it to show what may happen meanwhile (LR7b).
    pub closed_hook: Option<ClosedHook>,
}

/// See [`RuntimeConfig::closed_hook`].
pub type ClosedHook = Arc<dyn Fn() + Send + Sync>;

impl RuntimeConfig {
    /// This environment's paths, the loopback client, Python's timings.
    pub fn new(env: Arc<dyn Env>, state: &StateRoot) -> Result<Self, NetError> {
        Ok(Self {
            paths: LlamaPaths::from_env(env.as_ref()),
            temp_roots: temp_roots(env.as_ref()),
            globals: state.globals.clone(),
            http: Arc::new(LoopbackHttp::new()?),
            ports: Arc::new(net::free_loopback_port),
            env,
            start_timeout: START_TIMEOUT,
            health_timeout: HEALTH_TIMEOUT,
            poll_gap: POLL_GAP,
            stop_timeout: STOP_TIMEOUT,
            limits: SERVER_LIMITS,
            closed_hook: None,
        })
    }
}

/// Where the token's folder may be made, in order (LR6a): `TMPDIR`, `TEMP`
/// and `TMP` when set and not empty, then `%LOCALAPPDATA%\Temp`, then the
/// user's profile, as `tempfile.gettempdir()` falls back. Each is checked
/// when a folder is made ([`usable_temp_root`]), so one that is relative,
/// missing or remote is passed over, not used.
pub fn temp_roots(env: &dyn Env) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = ["TMPDIR", "TEMP", "TMP"]
        .into_iter()
        .filter_map(|name| env::text(env, name).filter(|value| !value.is_empty()))
        .map(PathBuf::from)
        .collect();
    if let Some(local) = env::text(env, "LOCALAPPDATA").filter(|value| !value.is_empty()) {
        roots.push(PathBuf::from(local).join("Temp"));
    }
    roots.push(state::home_dir(env, Platform::host()));
    roots
}

/// Is `root` a folder the token's folder may be made in? Absolute, on a
/// drive of this machine by its text, an existing folder reached through
/// local links only (a link to a share is refused before it is followed),
/// and on a volume that keeps access-control lists (LR6a: on FAT or exFAT
/// the private DACL is not kept, so the next root is used).
pub fn usable_temp_root(root: &Path) -> bool {
    root.is_absolute()
        && localfs::is_local_text(root)
        && matches!(
            localfs::open_walk(root, Access::Attributes, LinkRule::AnyLocal),
            Ok(walked) if walked.is_dir
                && volume_flags(root, &walked.file)
                    .is_ok_and(|flags| flags & lattice_sys::fs::FILE_PERSISTENT_ACLS != 0)
        )
}

/// The volume flags of the folder `root` opened as `file`
/// (`lattice_sys::fs::volume_flags`). In tests, a root named by
/// [`acl_seam::pretend`] answers the flags given there instead, so a FAT or
/// exFAT volume is stood in for without one.
fn volume_flags(root: &Path, file: &std::fs::File) -> std::io::Result<u32> {
    #[cfg(test)]
    if let Some(flags) = acl_seam::flags_of(root) {
        return Ok(flags);
    }
    let _ = root;
    lattice_sys::fs::volume_flags(file)
}

/// The test seam for [`volume_flags`]: per thread, roots whose volume flags
/// are pretended.
#[cfg(test)]
pub(crate) mod acl_seam {
    use std::cell::RefCell;
    use std::path::{Path, PathBuf};

    thread_local! {
        static PRETENDED: RefCell<Vec<(PathBuf, u32)>> = const { RefCell::new(Vec::new()) };
    }

    /// `root`'s volume answers `flags` on this thread from now on.
    pub(crate) fn pretend(root: &Path, flags: u32) {
        PRETENDED.with(|pretended| pretended.borrow_mut().push((root.to_path_buf(), flags)));
    }

    pub(super) fn flags_of(root: &Path) -> Option<u32> {
        PRETENDED.with(|pretended| {
            pretended
                .borrow()
                .iter()
                .find(|(path, _)| path == root)
                .map(|(_, flags)| *flags)
        })
    }
}

/// Make the token's private folder in the first usable root where it can be
/// made, and its file inside it, holding `token` and a line end (LR6,
/// LR6a). Both are Lattice's own temporaries, removed by [`forget_key`].
pub(crate) fn write_key(
    roots: &[PathBuf],
    token: &SecretString,
) -> Result<(PathBuf, PathBuf), LlamaError> {
    let key_dir = roots
        .iter()
        .filter(|root| usable_temp_root(root))
        .find_map(|root| fsx::create_private_temporary_dir(root, "alelyon-llama").ok())
        .ok_or(LlamaError::Token)?;
    let key_file = match fsx::temporary_for(&key_dir.join("api-key")) {
        Ok(file) => file,
        Err(_) => {
            forget_key(&key_dir.join("none"), &key_dir);
            return Err(LlamaError::Token);
        }
    };
    let written = lattice_sys::fs::create_private_file(&key_file)
        .and_then(|mut file| file.write_all(format!("{}\n", token.expose()).as_bytes()));
    if written.is_err() {
        forget_key(&key_file, &key_dir);
        return Err(LlamaError::Token);
    }
    Ok((key_dir, key_file))
}

/// LR6's command line for one launch.
pub fn command(
    binary: &Path,
    port: u16,
    model: &LocalModel,
    settings: &Settings,
    key_file: &Path,
) -> Vec<OsString> {
    let mut argv: Vec<OsString> = vec![binary.as_os_str().to_owned()];
    let mut flag = |name: &str, value: &OsStr| {
        argv.push(name.into());
        argv.push(value.to_owned());
    };
    flag("--host", OsStr::new(LOOPBACK));
    flag("--port", OsStr::new(&port.to_string()));
    flag("-m", model.path.as_os_str());
    flag("--alias", OsStr::new(&model.name));
    flag("-c", OsStr::new(&settings.ctx_size.to_string()));
    flag("-ngl", OsStr::new(&settings.gpu_layers.to_string()));
    flag("-np", OsStr::new(&settings.parallel.to_string()));
    flag("--api-key-file", key_file.as_os_str());
    argv.push("--no-webui".into());
    argv.push("--offline".into());
    argv.push(JINJA.into());
    argv
}

/// Is `name` one of the names removed from the server's environment?
pub fn is_stripped_name(name: &OsStr) -> bool {
    let upper = name.to_string_lossy().to_uppercase();
    STRIPPED_ENV_PREFIXES
        .iter()
        .any(|prefix| upper.starts_with(prefix))
}

/// X7's block, without `LLAMA_ARG_*` or `OLLAMA_*` (LR6), plus each of
/// [`VULKAN_DEVICE_NAMES`] that is set in `env`, with its value (LR6′).
pub fn server_environment(env: &dyn Env, globals: &Path) -> Vec<(OsString, OsString)> {
    let mut block: Vec<(OsString, OsString)> = child_environment(env, None, globals)
        .into_iter()
        .filter(|(name, _)| !is_stripped_name(name))
        .collect();
    for name in VULKAN_DEVICE_NAMES {
        if let Some(value) = env.var(name) {
            block.push((OsString::from(name), value));
        }
    }
    block
}

/// A fresh launch token: two version-4 UUIDs in hex, 244 random bits from
/// the operating system's generator.
pub(crate) fn new_token() -> SecretString {
    let mut token = uuid::Uuid::new_v4().simple().to_string();
    token.push_str(&uuid::Uuid::new_v4().simple().to_string());
    SecretString::new(token)
}

/// The token file and its folder, removed (only Lattice's own temporaries).
pub(crate) fn forget_key(file: &Path, dir: &Path) {
    let _ = fsx::remove_own_temporary(file);
    let _ = fsx::remove_own_temporary_dir(dir);
}

/// What `/props` says that the core uses (LR8).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Props {
    /// The chat template, when the server reports a non-empty one.
    pub chat_template: Option<String>,
    /// `default_generation_settings.n_ctx`.
    pub n_ctx: Option<u64>,
    /// `modalities.vision`.
    pub vision: Option<bool>,
}

impl Props {
    pub fn parse(body: &[u8]) -> Option<Self> {
        let value: Value = serde_json::from_slice(body).ok()?;
        Some(Self {
            chat_template: value["chat_template"]
                .as_str()
                .filter(|text| !text.is_empty())
                .map(str::to_owned),
            n_ctx: value["default_generation_settings"]["n_ctx"].as_u64(),
            vision: value["modalities"]["vision"].as_bool(),
        })
    }
}

/// One launch: the process, its token, and how to reach it.
struct Server {
    generation: u64,
    model: LocalModel,
    endpoint: ManagedEndpoint,
    child: Arc<Child>,
    key_dir: PathBuf,
    key_file: PathBuf,
    /// The loopback port it was told to listen on.
    port: u16,
    closed_hook: Option<ClosedHook>,
    idle: Duration,
    stop_timeout: Duration,
    closed: bool,
}

impl Server {
    fn alive(&self) -> bool {
        matches!(self.child.wait(Some(Duration::ZERO)), Ok(None))
    }

    /// End the tree, wait for it, then remove the token and its folder.
    fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        let _ = self.child.kill_tree();
        let _ = self.child.wait(Some(self.stop_timeout));
        forget_key(&self.key_file, &self.key_dir);
        if let Some(closed) = &self.closed_hook {
            closed();
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.close();
    }
}

#[derive(Default)]
struct Slot {
    server: Option<Server>,
    /// The last launch's number.
    generation: u64,
    /// Leases alive on the current server.
    inflight: usize,
    /// The one idle timer: its number and its task.
    timer: Option<(u64, AbortHandle)>,
    timers: u64,
    failed: Option<String>,
    closed: bool,
}

struct Inner {
    config: RuntimeConfig,
    handle: Handle,
    /// Starts, switches and shutdowns, one at a time.
    start: tokio::sync::Mutex<()>,
    slot: Mutex<Slot>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let slot = self
            .slot
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, timer)) = slot.timer.take() {
            timer.abort();
        }
        slot.server.take();
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, Slot> {
        self.slot
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A lease on the current server, which also cancels the idle timer.
    fn lease(self: &Arc<Self>, slot: &mut Slot) -> Option<Opened> {
        let server = slot.server.as_ref()?;
        let (endpoint, generation) = (server.endpoint.clone(), server.generation);
        slot.inflight += 1;
        if let Some((_, timer)) = slot.timer.take() {
            timer.abort();
        }
        let weak = Arc::downgrade(self);
        Some(Opened {
            endpoint,
            lease: Lease::new(move || {
                if let Some(inner) = weak.upgrade() {
                    inner.release(generation);
                }
            }),
        })
    }

    /// A lease ended: when it was the last, arm the idle timer.
    fn release(self: &Arc<Self>, generation: u64) {
        let mut slot = self.lock();
        let Some(idle) = slot
            .server
            .as_ref()
            .filter(|server| server.generation == generation)
            .map(|server| server.idle)
        else {
            return;
        };
        slot.inflight = slot.inflight.saturating_sub(1);
        if slot.inflight == 0 {
            self.arm(&mut slot, generation, idle);
        }
    }

    /// The single idle timer: one sleeping task, replaced, never repeated.
    fn arm(self: &Arc<Self>, slot: &mut Slot, generation: u64, idle: Duration) {
        if let Some((_, old)) = slot.timer.take() {
            old.abort();
        }
        slot.timers += 1;
        let number = slot.timers;
        let weak = Arc::downgrade(self);
        let task = self.handle.spawn(async move {
            tokio::time::sleep(idle).await;
            if let Some(inner) = weak.upgrade() {
                inner.idle_fire(generation, number).await;
            }
        });
        slot.timer = Some((number, task.abort_handle()));
    }

    /// The timer fired: stop the server if it is still the one the timer was
    /// armed for and no lease was taken since. The stop holds the `start`
    /// mutex until the old process has exited and its token is gone (LR7b),
    /// so an `open()` that arrives meanwhile waits instead of starting a
    /// second server beside one still closing (Python's `Manager` holds its
    /// lock across `stop()` the same way). A lease taken while this waits for
    /// the mutex aborts it (the timer is its task); once the server is taken
    /// the timer is cleared, so nothing aborts the close half-way.
    async fn idle_fire(self: &Arc<Self>, generation: u64, number: u64) {
        let _one_at_a_time = self.start.lock().await;
        let server = {
            let mut slot = self.lock();
            let armed = matches!(slot.timer, Some((n, _)) if n == number);
            let same = slot
                .server
                .as_ref()
                .is_some_and(|server| server.generation == generation);
            if !(armed && same && slot.inflight == 0) {
                return;
            }
            slot.timer = None;
            slot.server.take()
        };
        if let Some(server) = server {
            self.retire(server).await;
        }
    }

    /// The server's output ended. When its process has ended too, it stopped
    /// on its own: forget it, and its token.
    fn on_exit(self: &Arc<Self>, generation: u64, child: &Child) {
        let _ = child.wait(Some(Duration::from_secs(2)));
        if matches!(child.wait(Some(Duration::ZERO)), Ok(None)) {
            return;
        }
        let server = {
            let mut slot = self.lock();
            if !slot
                .server
                .as_ref()
                .is_some_and(|server| server.generation == generation)
            {
                return;
            }
            if let Some((_, timer)) = slot.timer.take() {
                timer.abort();
            }
            slot.failed = Some(STOPPED_ON_ITS_OWN.to_owned());
            slot.server.take()
        };
        // A reader thread, not a runtime worker: closing may block.
        drop(server);
    }

    /// Stop a server on the blocking pool and wait for it.
    async fn retire(&self, server: Server) {
        let _ = self.handle.spawn_blocking(move || drop(server)).await;
    }
}

/// Copy a pipe into the log until it ends, then call `ended`.
pub(crate) fn drain(mut from: File, log: Arc<Mutex<File>>, ended: Option<Box<dyn FnOnce() + Send>>) {
    let _ = std::thread::Builder::new()
        .name("llama-server-log".into())
        .spawn(move || {
            let mut buffer = [0u8; 8192];
            loop {
                match from.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        let mut log = log.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        let _ = log.write_all(&buffer[..read]);
                    }
                }
            }
            if let Some(ended) = ended {
                ended();
            }
        });
}

/// Start the process for `model` (blocking: files and `CreateProcess`).
fn launch(
    config: &RuntimeConfig,
    model: LocalModel,
    generation: u64,
    owner: Weak<Inner>,
) -> Result<Server, LlamaError> {
    let binary = find_binary(&config.paths).map_err(LlamaError::Binary)?;
    let settings = settings::load(&config.paths);
    let binary_sha256 = settings::manifest_sha256(&config.paths, &binary);
    let port = (config.ports)().map_err(|_| LlamaError::Port)?;
    let token = new_token();
    let (key_dir, key_file) = write_key(&config.temp_roots, &token)?;
    let opened_log = std::fs::create_dir_all(config.paths.logs_dir()).and_then(|()| {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(config.paths.logs_dir().join(format!("{}.log", model.name)))
    });
    let Ok(log) = opened_log else {
        forget_key(&key_file, &key_dir);
        return Err(LlamaError::Token);
    };
    let argv = command(&binary, port, &model, &settings, &key_file);
    let environment = server_environment(config.env.as_ref(), &config.globals);
    let cwd = binary.parent().unwrap_or(&binary).to_path_buf();
    let mut child = match lattice_sys::process::spawn(&SpawnRequest {
        program: &binary,
        argv: &argv,
        cwd: &cwd,
        env: &environment,
        limits: config.limits,
    }) {
        Ok(child) => child,
        Err(_) => {
            forget_key(&key_file, &key_dir);
            return Err(LlamaError::Spawn);
        }
    };
    let (stdout, stderr) = (child.take_stdout(), child.take_stderr());
    let child = Arc::new(child);
    let log = Arc::new(Mutex::new(log));
    if let Some(stdout) = stdout {
        let watched = child.clone();
        drain(
            stdout,
            log.clone(),
            Some(Box::new(move || {
                if let Some(inner) = owner.upgrade() {
                    inner.on_exit(generation, &watched);
                }
            })),
        );
    }
    if let Some(stderr) = stderr {
        drain(stderr, log, None);
    }
    Ok(Server {
        generation,
        endpoint: ManagedEndpoint {
            base_url: format!("http://{LOOPBACK}:{port}"),
            alias: model.name.clone(),
            token,
            binary_sha256,
            generation,
        },
        model,
        child,
        key_dir,
        key_file,
        port,
        idle: settings.idle(),
        stop_timeout: config.stop_timeout,
        closed_hook: config.closed_hook.clone(),
        closed: false,
    })
}

pub(crate) async fn healthy(http: &dyn HttpGet, base_url: &str, timeout: Duration) -> bool {
    let request = HttpRequest {
        url: format!("{base_url}/health"),
        bearer: None,
        timeout,
    };
    matches!(http.get(request).await, Ok(response) if response.status == 200)
}

/// Who holds the loopback listener on a launch's port (LR7a).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ListenerOwner {
    /// No listener yet (the server has not bound), or the table could not be
    /// read: nothing is sent.
    Nobody,
    /// `127.0.0.1:<port>` is the child's, and no other process listens there
    /// or on `0.0.0.0:<port>`.
    Child,
    /// Another process listens on `127.0.0.1:<port>` or `0.0.0.0:<port>`.
    Other,
}

/// The owning process of `port`'s loopback listener, per the system's table
/// (`lattice_sys::net`, `GetExtendedTcpTable` with
/// `TCP_TABLE_OWNER_PID_LISTENER`), against the child's process id.
#[cfg(windows)]
pub(crate) fn listener_owner(port: u16, child: u32) -> ListenerOwner {
    const LOOPBACK_V4: [u8; 4] = [127, 0, 0, 1];
    const ANY_V4: [u8; 4] = [0, 0, 0, 0];
    let Ok(rows) = lattice_sys::net::tcp_listeners_v4() else {
        return ListenerOwner::Nobody;
    };
    let on_port: Vec<_> = rows
        .into_iter()
        .filter(|row| row.port == port && (row.address == LOOPBACK_V4 || row.address == ANY_V4))
        .collect();
    if on_port.iter().any(|row| row.pid != child) {
        ListenerOwner::Other
    } else if on_port.iter().any(|row| row.address == LOOPBACK_V4) {
        ListenerOwner::Child
    } else {
        ListenerOwner::Nobody
    }
}

/// Elsewhere there is no owner table: the listener is taken as the child's
/// (the chat core ships on Windows; this keeps other targets' tests going).
#[cfg(not(windows))]
pub(crate) fn listener_owner(_port: u16, _child: u32) -> ListenerOwner {
    ListenerOwner::Child
}

/// Launch, then wait for `/health`; the process's end, the time running out
/// or a shutdown ends the wait. Nothing is sent to the port until its
/// loopback listener is the child's own (LR7a): a listener of any other
/// process there ends the start at once ([`LlamaError::PortTaken`]), and the
/// owner is read again after `/health` answers, before the server counts as
/// ready.
async fn start(inner: &Arc<Inner>, model: LocalModel) -> Result<Server, LlamaError> {
    let generation = {
        let mut slot = inner.lock();
        slot.generation += 1;
        slot.generation
    };
    let config = inner.config.clone();
    let owner = Arc::downgrade(inner);
    let server = inner
        .handle
        .spawn_blocking(move || launch(&config, model, generation, owner))
        .await
        .map_err(|_| LlamaError::Spawn)??;
    let deadline = tokio::time::Instant::now() + inner.config.start_timeout;
    let pid = server.child.pid();
    loop {
        let listener = listener_owner(server.port, pid);
        if listener == ListenerOwner::Other {
            inner.retire(server).await;
            return Err(LlamaError::PortTaken);
        }
        if !server.alive() {
            inner.retire(server).await;
            return Err(LlamaError::ExitedWhileLoading);
        }
        if inner.lock().closed {
            inner.retire(server).await;
            return Err(LlamaError::Closed);
        }
        if listener == ListenerOwner::Child
            && healthy(
                inner.config.http.as_ref(),
                &server.endpoint.base_url,
                inner.config.health_timeout,
            )
            .await
            && listener_owner(server.port, pid) == ListenerOwner::Child
        {
            // llama.cpp has read the key file by now: it goes at once, not at
            // the stop (LR6a), so it is on disk only while the server starts.
            forget_key(&server.key_file, &server.key_dir);
            return Ok(server);
        }
        if tokio::time::Instant::now() >= deadline {
            inner.retire(server).await;
            return Err(LlamaError::NotReady);
        }
        tokio::time::sleep(inner.config.poll_gap).await;
    }
}

/// [`start`], and once more on a new port when another process held the
/// first one (LR7a).
async fn start_retrying(inner: &Arc<Inner>, model: LocalModel) -> Result<Server, LlamaError> {
    match start(inner, model.clone()).await {
        Err(LlamaError::PortTaken) => start(inner, model).await,
        other => other,
    }
}

async fn open(inner: Arc<Inner>, model: LocalModel) -> Result<Opened, LlamaError> {
    let _one_at_a_time = inner.start.lock().await;
    let previous = {
        let mut slot = inner.lock();
        if slot.closed {
            return Err(LlamaError::Closed);
        }
        let reusable = slot
            .server
            .as_ref()
            .is_some_and(|server| server.model.path == model.path && server.alive());
        if reusable {
            return inner.lease(&mut slot).ok_or(LlamaError::Closed);
        }
        // Another model (a switch), or a server that died: it stops first.
        if let Some((_, timer)) = slot.timer.take() {
            timer.abort();
        }
        slot.inflight = 0;
        slot.server.take()
    };
    if let Some(previous) = previous {
        inner.retire(previous).await;
    }
    match start_retrying(&inner, model).await {
        Ok(server) => {
            let refused = {
                let mut slot = inner.lock();
                if slot.closed {
                    Some(server)
                } else {
                    slot.failed = None;
                    slot.server = Some(server);
                    return inner.lease(&mut slot).ok_or(LlamaError::Closed);
                }
            };
            if let Some(server) = refused {
                inner.retire(server).await;
            }
            Err(LlamaError::Closed)
        }
        Err(error) => {
            inner.lock().failed = Some(error.sentence().to_owned());
            Err(error)
        }
    }
}

/// The one managed server this process runs.
#[derive(Clone)]
pub struct LlamaRuntime {
    inner: Arc<Inner>,
}

impl LlamaRuntime {
    /// A runtime whose tasks and timers run on `handle`. Nothing starts until
    /// the first [`open`](Self::open).
    pub fn new(config: RuntimeConfig, handle: Handle) -> Self {
        Self {
            inner: Arc::new(Inner {
                config,
                handle,
                start: tokio::sync::Mutex::new(()),
                slot: Mutex::new(Slot::default()),
            }),
        }
    }

    /// A server running `model`, and a lease that keeps it in use.
    pub fn open(&self, model: LocalModel) -> BoxFuture<'static, Result<Opened, LlamaError>> {
        let inner = self.inner.clone();
        let handle = inner.handle.clone();
        // On the runtime's own handle, whose timers its sleeps need.
        async move {
            handle
                .spawn(open(inner, model))
                .await
                .unwrap_or(Err(LlamaError::Closed))
        }
        .boxed()
    }

    /// The model running now, if a server runs.
    pub fn running(&self) -> Option<String> {
        let slot = self.inner.lock();
        slot.server
            .as_ref()
            .filter(|server| server.alive())
            .map(|server| server.model.name.clone())
    }

    /// The last start's failure, or how the last server ended on its own.
    pub fn failed(&self) -> Option<String> {
        self.inner.lock().failed.clone()
    }

    /// The running server's process id (for tests and diagnostics).
    pub fn pid(&self) -> Option<u32> {
        self.inner
            .lock()
            .server
            .as_ref()
            .map(|server| server.child.pid())
    }

    /// `GET /props` with the token (LR8).
    pub async fn props(&self, endpoint: &ManagedEndpoint) -> Result<Props, NetError> {
        let response = self
            .inner
            .config
            .http
            .get(HttpRequest {
                url: format!("{}/props", endpoint.base_url),
                bearer: Some(endpoint.token.clone()),
                timeout: PROPS_TIMEOUT,
            })
            .await?;
        if response.status != 200 {
            return Err(NetError::Connect);
        }
        Props::parse(&response.body).ok_or(NetError::Connect)
    }

    /// Stop the server now, if one runs, and start none again. A start in
    /// progress notices and stops what it started.
    pub async fn shutdown(&self) {
        let server = {
            let mut slot = self.inner.lock();
            slot.closed = true;
            if let Some((_, timer)) = slot.timer.take() {
                timer.abort();
            }
            slot.server.take()
        };
        if let Some(server) = server {
            self.inner.retire(server).await;
        }
    }
}

impl ManagedRuntime for LlamaRuntime {
    fn running(&self) -> Option<String> {
        LlamaRuntime::running(self)
    }

    fn failed(&self) -> Option<String> {
        LlamaRuntime::failed(self)
    }

    fn open(&self, model: LocalModel) -> BoxFuture<'static, Result<Opened, LlamaError>> {
        LlamaRuntime::open(self, model)
    }

    fn props(&self, endpoint: &ManagedEndpoint) -> BoxFuture<'static, Option<Props>> {
        let (runtime, endpoint) = (self.clone(), endpoint.clone());
        Box::pin(async move { LlamaRuntime::props(&runtime, &endpoint).await.ok() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::MapEnv;

    fn model() -> LocalModel {
        LocalModel {
            name: "tiny".into(),
            path: PathBuf::from(r"C:\m\tiny.gguf"),
            size: 1,
        }
    }

    #[test]
    fn the_command_line_is_pythons_plus_jinja_and_never_the_tools_flag() {
        let argv: Vec<String> = command(
            Path::new(r"C:\llama\llama-server.exe"),
            41000,
            &model(),
            &Settings::default(),
            Path::new(r"C:\t\key"),
        )
        .into_iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
        assert_eq!(
            argv,
            [
                r"C:\llama\llama-server.exe",
                "--host",
                "127.0.0.1",
                "--port",
                "41000",
                "-m",
                r"C:\m\tiny.gguf",
                "--alias",
                "tiny",
                "-c",
                "8192",
                "-ngl",
                "999",
                "-np",
                "1",
                "--api-key-file",
                r"C:\t\key",
                "--no-webui",
                "--offline",
                "--jinja",
            ]
        );
        let tools = format!("--{}", "tools");
        assert!(!argv.iter().any(|arg| arg.starts_with(&tools)));
    }

    #[test]
    fn the_servers_environment_is_x7_without_llama_or_ollama_names() {
        let env = MapEnv::new()
            .with("PATH", r"C:\Windows\System32")
            .with("SystemRoot", r"C:\Windows")
            .with("LLAMA_ARG_CTX_SIZE", "1")
            .with("OLLAMA_HOST", "0.0.0.0")
            .with("ANTHROPIC_API_KEY", "sentinel");
        let block = server_environment(&env, Path::new(r"C:\nowhere\globals"));
        let names: Vec<String> = block
            .iter()
            .map(|(name, _)| name.to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"PATH".to_owned()) && names.contains(&"SystemRoot".to_owned()));
        assert!(!names.iter().any(|name| name.starts_with("LLAMA_ARG_")
            || name.starts_with("OLLAMA_")
            || name.contains("API_KEY")));
        for name in ["llama_arg_x", "Ollama_Models", "LLAMA_ARG_", "OLLAMA_"] {
            assert!(is_stripped_name(OsStr::new(name)), "{name}");
        }
        for name in ["LLAMA", "OLLAMA", "PATH", "XLLAMA_ARG_"] {
            assert!(!is_stripped_name(OsStr::new(name)), "{name}");
        }
    }

    /// The Vulkan-looking names of a block, with their values, in order.
    fn vk(block: &[(OsString, OsString)]) -> Vec<(String, String)> {
        block
            .iter()
            .map(|(name, value)| {
                (
                    name.to_string_lossy().into_owned(),
                    value.to_string_lossy().into_owned(),
                )
            })
            .filter(|(name, _)| {
                let upper = name.to_uppercase();
                upper.starts_with("VK_") || upper.starts_with("GGML_VK_")
            })
            .collect()
    }

    /// Every one of [`VULKAN_DEVICE_NAMES`] set, each with its own value.
    fn every_device_name() -> Vec<(String, String)> {
        VULKAN_DEVICE_NAMES
            .iter()
            .enumerate()
            .map(|(at, name)| ((*name).to_owned(), format!("value-{at}")))
            .collect()
    }

    fn env_with(names: &[(String, String)]) -> MapEnv {
        let mut env = MapEnv::new().with("SystemRoot", r"C:\Windows");
        for (name, value) in names {
            env.set(name, value.as_str());
        }
        env
    }

    const GLOBALS: &str = r"C:\nowhere\globals";

    /// LR6′, case 1: every device-selection name that is set reaches the
    /// server, with its value. (LF3's three cases at the unit level; each one
    /// expects at least one name, so a block without the pass-through fails
    /// all three.)
    /// Mutant: no pass-through.
    #[test]
    fn lr6_vulkan_device_names_that_are_set_reach_the_server() {
        let all = every_device_name();
        let env = env_with(&all);
        assert_eq!(vk(&server_environment(&env, Path::new(GLOBALS))), all);
    }

    /// LR6′, case 2: no other `VK_*` name passes, even beside the six.
    /// Mutants: no pass-through; every `VK_*` name passed.
    #[test]
    fn lr6_no_other_vulkan_name_reaches_the_server() {
        let all = every_device_name();
        let mut env = env_with(&all);
        for (name, value) in [
            ("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation"),
            ("VK_LAYER_PATH", r"C:\layers"),
            ("VK_ADD_DRIVER_FILES", r"C:\drivers\extra.json"),
            ("VK_LOADER_DEBUG", "all"),
            ("VK_LOADER_LAYERS_ENABLE", "*"),
            ("GGML_VK_DISABLE_F16", "1"),
        ] {
            env.set(name, value);
        }
        assert_eq!(vk(&server_environment(&env, Path::new(GLOBALS))), all);
    }

    /// LR6′, case 3: a device-selection name that is not set is not added.
    /// Mutants: no pass-through; a name passed (empty) when it is not set.
    #[test]
    fn lr6_a_vulkan_device_name_that_is_not_set_is_not_added() {
        let only = vec![("VK_LOADER_DEVICE_ID_FILTER".to_owned(), "0x7550".to_owned())];
        let env = env_with(&only);
        assert_eq!(vk(&server_environment(&env, Path::new(GLOBALS))), only);
    }

    #[test]
    fn props_read_the_template_the_context_and_vision() {
        let props = Props::parse(
            br#"{"chat_template": "{{x}}", "default_generation_settings": {"n_ctx": 4096}, "modalities": {"vision": true}}"#,
        )
        .unwrap();
        assert_eq!(props.chat_template.as_deref(), Some("{{x}}"));
        assert_eq!(props.n_ctx, Some(4096));
        assert_eq!(props.vision, Some(true));
        let bare = Props::parse(br#"{"chat_template": ""}"#).unwrap();
        assert_eq!(bare, Props::default());
        assert_eq!(Props::parse(b"not json"), None);
    }

    /// LR6a: the temporary roots are `TMPDIR`, `TEMP`, `TMP`, then
    /// `%LOCALAPPDATA%\Temp`, then the profile, as `gettempdir` falls back.
    #[test]
    fn the_temporary_roots_fall_back_as_gettempdir_does() {
        let env = MapEnv::new()
            .with("TMPDIR", "relative_tmp")
            .with("TEMP", r"C:\t1")
            .with("TMP", "")
            .with("LOCALAPPDATA", r"D:\Profiles\x\AppData\Local")
            .with("USERPROFILE", r"D:\Profiles\x");
        let mut expected = vec![PathBuf::from("relative_tmp"), PathBuf::from(r"C:\t1")];
        expected.push(PathBuf::from(r"D:\Profiles\x\AppData\Local").join("Temp"));
        expected.push(state::home_dir(&env, Platform::host()));
        assert_eq!(temp_roots(&env), expected);
    }

    /// LR6a: a root that is relative, missing, a file, or on a share is not
    /// used (the share is refused by its text: nothing connects to it).
    /// Mutants: the whole check replaced by `is_dir()`; the folder check dropped.
    #[test]
    fn a_relative_missing_or_remote_temporary_root_is_not_used() {
        let dir = crate::testkit::TempDir::new("llama-temp-roots");
        let file = dir.path().join("a-file");
        std::fs::write(&file, "x").unwrap();
        for bad in [
            PathBuf::from("relative_tmp"),
            PathBuf::from("src"),
            dir.path().join("missing"),
            file,
            PathBuf::from(r"\\198.51.100.7\x\tmp"),
        ] {
            assert!(!usable_temp_root(&bad), "{}", bad.display());
        }
        assert!(usable_temp_root(dir.path()));
    }

    /// The SIDs of a DACL, sorted, and whether any ACE is inherited.
    #[cfg(windows)]
    fn trustees(path: &Path) -> (Vec<String>, bool) {
        let dacl = lattice_sys::fs::read_dacl(path).unwrap();
        assert!(dacl.protected, "{}: {dacl:?}", path.display());
        let mut sids: Vec<String> = dacl.aces.iter().map(|ace| ace.sid.clone()).collect();
        sids.sort();
        (sids, dacl.aces.iter().any(|ace| ace.inherited))
    }

    /// LR6a: the token's folder and file are made in the first usable root,
    /// each with the private DACL (no inherited ACE; SYSTEM, Administrators
    /// and the user only), read back with `GetNamedSecurityInfoW`; and with
    /// no usable root, no key is made.
    /// Mutants: the folder made the old way (`create_own_temporary_dir`); the
    /// file made by `OpenOptions`.
    #[cfg(windows)]
    #[test]
    fn the_key_is_private_and_made_in_the_first_usable_root() {
        use lattice_sys::fs::{ADMINISTRATORS_SID, SYSTEM_SID, current_user_sid};
        let dir = crate::testkit::TempDir::new("llama-key");
        let good = dir.path().join("good");
        std::fs::create_dir(&good).unwrap();
        let token = new_token();
        let roots = vec![dir.path().join("missing"), good.clone()];
        let (key_dir, key_file) = write_key(&roots, &token).unwrap();
        assert_eq!(key_dir.parent(), Some(good.as_path()));
        assert_eq!(key_file.parent(), Some(key_dir.as_path()));
        assert!(key_file.is_absolute());
        assert_eq!(
            std::fs::read_to_string(&key_file).unwrap(),
            format!("{}\n", token.expose())
        );
        let mut expected = vec![
            SYSTEM_SID.to_owned(),
            ADMINISTRATORS_SID.to_owned(),
            current_user_sid().unwrap(),
        ];
        expected.sort();
        assert_eq!(trustees(&key_dir), (expected.clone(), false), "the folder");
        assert_eq!(trustees(&key_file), (expected, false), "the file");
        forget_key(&key_file, &key_dir);
        assert!(!key_dir.exists());

        assert_eq!(
            write_key(&[dir.path().join("missing")], &token),
            Err(LlamaError::Token)
        );
    }

    /// LR6a's residual: a temporary root on a volume that keeps no
    /// access-control lists (FAT, exFAT: `FILE_PERSISTENT_ACLS` clear, stood
    /// in for through the volume-flags seam) is passed over, and the key is
    /// made in the next root. This machine's own temporary folder reports the
    /// flag (the real query, the positive control).
    /// Mutant: `usable_temp_root` ignores the volume flags.
    #[test]
    fn a_temporary_root_on_a_volume_without_acls_is_passed_over() {
        let dir = crate::testkit::TempDir::new("llama-key-fat");
        let fat = dir.path().join("fat");
        let ntfs = dir.path().join("ntfs");
        std::fs::create_dir(&fat).unwrap();
        std::fs::create_dir(&ntfs).unwrap();
        assert!(usable_temp_root(&fat), "the real volume keeps ACLs");
        // FILE_CASE_PRESERVED_NAMES | FILE_UNICODE_ON_DISK, as exFAT reports.
        acl_seam::pretend(&fat, 0x0000_0006);
        assert!(!usable_temp_root(&fat));
        assert!(usable_temp_root(&ntfs));
        let token = new_token();
        let (key_dir, key_file) = write_key(&[fat.clone(), ntfs.clone()], &token).unwrap();
        assert_eq!(key_dir.parent(), Some(ntfs.as_path()), "the next root");
        assert_eq!(
            std::fs::read_dir(&fat).unwrap().count(),
            0,
            "nothing was made on the FAT stand-in"
        );
        forget_key(&key_file, &key_dir);
        assert_eq!(write_key(&[fat], &token), Err(LlamaError::Token));
    }

    /// LR6a: folders pre-made under the names the old, predictable scheme
    /// would take next (`.alelyon-llama.lattice-<pid>-<n>.tmp`) do not stop
    /// the key from being made.
    /// Mutant: the folder made the old way (`create_own_temporary_dir`).
    #[test]
    fn folders_squatting_the_old_predictable_names_do_not_stop_a_key() {
        let dir = crate::testkit::TempDir::new("llama-key-squat");
        let next = fsx::next_temporary_number();
        for n in next..next + 64 {
            std::fs::create_dir(dir.path().join(format!(
                ".alelyon-llama.lattice-{}-{n}.tmp",
                std::process::id()
            )))
            .unwrap();
        }
        let token = new_token();
        let made = write_key(&[dir.path().to_path_buf()], &token);
        let (key_dir, key_file) = made.unwrap();
        assert!(key_file.is_file());
        forget_key(&key_file, &key_dir);
    }

    #[test]
    fn a_launch_token_is_fresh_hex_and_never_printed() {
        let a = new_token();
        let b = new_token();
        assert_eq!(a.expose().len(), 64);
        assert!(a.expose().bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(a.expose(), b.expose());
        assert_eq!(format!("{a:?}"), "***");
    }
}
