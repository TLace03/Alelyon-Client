//! [`McpHub`]: the MCP servers of this process (the chat core's spec §12,
//! "Lifecycle"; CR2, B18). Not a port.
//!
//! - **Declared** servers are read when asked ([`McpHub::declared`]): the
//!   reader's file always, a folder's only when the caller says it is trusted.
//!   Reading a folder's file goes through its path rules, which may run git,
//!   so the callers do it on the blocking pool.
//! - **Enabled** means the reader confirmed the entry's exact hash
//!   ([`McpHub::enable`], `ConfirmPort(EnableMcpServer)`); a server that is
//!   not enabled never starts.
//! - **Started lazily**: at an Agent-mode turn that offers its tools
//!   ([`McpHub::turn_tools`]), or when the reader asks ([`McpHub::start`]).
//!   One start at a time per server; the handshake and the first tool list
//!   are the only timers at start (CR2). A start that fails is a sentence,
//!   and the turn goes on without that server.
//! - **Running** until the reader stops it, its entry changes (the next use
//!   starts the new one), its output ends, or Lattice closes: dropping the
//!   hub closes every server's Job, which ends its whole tree. No keep-alive
//!   and no polling: an idle server costs Lattice no wake-up. Spec §12 also
//!   stops a server when no open conversation uses it; the window never says
//!   when a conversation closes, so this hub keeps a started server until one
//!   of the above (named in the native README).
//! - **Its log**: the last [`LOG_LINES`] lines of its stderr and its log
//!   messages, shown in the Tools view, never sent to a model.
//!
//! Every change (a start, a stop, an end, a decision) bumps
//! [`McpHub::changed`], so the window can read the servers again without
//! polling.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak, mpsc};
use std::time::Duration;

use lattice_sys::process::Child;
use serde_json::Value;
use tokio::runtime::Handle;
use tokio::sync::watch;

use super::approvals::{Approvals, WHOLE_SERVER};
use super::client::{self, Connection, Event, ServerInfo, Tool, cut};
use super::config::{self, Declared, Problem, Scope, ServerEntry, ServerKey};
use super::launch;
use super::names;
use super::result::{self, CallText};
use crate::clock::Clock;
use crate::env::Env;
use crate::git::runner::GitRunner;
use crate::ports::{ConfirmRequest, Confirmer, Initiated};
use crate::state::StateRoot;
use crate::workspace::Workspace;

/// The lines of a server's log kept.
pub const LOG_LINES: usize = 200;
/// The most MCP tools one turn offers.
pub const MAX_TURN_TOOLS: usize = 100;
/// The longest description a model is given for one tool.
pub const MAX_MODEL_DESCRIPTION: usize = 4 * 1024;

/// How long the hub waits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timeouts {
    /// For `initialize` (a first `npx` run fetches its package first).
    pub start: Duration,
    /// For each page of `tools/list`.
    pub list: Duration,
    /// For one `tools/call`.
    pub call: Duration,
    /// For a stopped server to end by itself before its Job ends it.
    pub stop: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            start: Duration::from_secs(30),
            list: Duration::from_secs(30),
            call: Duration::from_secs(300),
            stop: Duration::from_secs(2),
        }
    }
}

/// A server's state in this process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerStatus {
    Stopped,
    Starting,
    Running,
    /// It did not start, or it ended by itself: the sentence says why.
    Failed(String),
}

/// One tool, as the Tools view shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolView {
    pub name: String,
    /// What the model is offered it as.
    pub model_name: String,
    pub title: Option<String>,
    pub description: String,
    /// Switched on (offered to the model).
    pub on: bool,
    /// Allowed always: its calls do not ask.
    pub allowed: bool,
    /// The server's own hints (untrusted).
    pub read_only: Option<bool>,
    pub destructive: Option<bool>,
    pub open_world: Option<bool>,
}

/// One server, as the Tools view shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct ServerView {
    pub key: ServerKey,
    /// Where it is declared.
    pub file: String,
    pub command_line: String,
    pub env_names: Vec<String>,
    pub cwd: Option<String>,
    /// `"disabled": true` in its file, or switched off here.
    pub off: bool,
    /// The reader enabled exactly this entry.
    pub enabled: bool,
    pub status: ServerStatus,
    /// "name version, MCP <version>", once it has started.
    pub server: Option<String>,
    /// Its own instructions (untrusted), once it has started.
    pub instructions: Option<String>,
    /// Its tools, once it has started.
    pub tools: Vec<ToolView>,
    /// What its tool list left out.
    pub problems: Vec<String>,
    /// The last lines of its log.
    pub log: Vec<String>,
}

/// Every server the reader can see now, and the declarations not used.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Overview {
    pub servers: Vec<ServerView>,
    pub problems: Vec<Problem>,
    /// The reader's file, as a path.
    pub user_file: String,
}

/// One tool a turn offers.
#[derive(Clone, Debug, PartialEq)]
pub struct TurnTool {
    pub key: ServerKey,
    /// The entry's hash the server started with: a call goes only to that
    /// server.
    pub sha256: String,
    /// The file that declares the server.
    pub file: String,
    /// The server's own name for it.
    pub tool: String,
    pub model_name: String,
    pub description: String,
    pub parameters: Value,
    /// Allowed always when the turn began (each call reads it again).
    pub allowed: bool,
}

/// What a turn offers, and what it should be told.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TurnTools {
    pub tools: Vec<TurnTool>,
    /// A server that did not start, a list that was cut: one sentence each.
    pub notices: Vec<String>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

type Log = Arc<Mutex<VecDeque<String>>>;

fn push_log(log: &Log, line: String) {
    let mut log = lock(log);
    if log.len() >= LOG_LINES {
        log.pop_front();
    }
    log.push_back(line);
}

/// A running server.
struct Live {
    child: Mutex<Option<Child>>,
    conn: Arc<Connection>,
    info: ServerInfo,
    tools: Mutex<Vec<Tool>>,
    problems: Mutex<Vec<String>>,
    /// `notifications/tools/list_changed` came: list again before the next
    /// turn offers them.
    stale: Arc<AtomicBool>,
    sha256: String,
}

impl Live {
    fn end(&self) -> Option<Child> {
        self.conn.close();
        lock(&self.child).take()
    }
}

/// One declared server's place in the hub.
struct Slot {
    start: tokio::sync::Mutex<()>,
    live: Mutex<Option<Arc<Live>>>,
    status: Mutex<ServerStatus>,
    log: Log,
    /// Ends (disconnects) when the running server's stderr does.
    stderr_end: Mutex<Option<mpsc::Receiver<()>>>,
}

impl Slot {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            start: tokio::sync::Mutex::new(()),
            live: Mutex::new(None),
            status: Mutex::new(ServerStatus::Stopped),
            log: Arc::default(),
            stderr_end: Mutex::new(None),
        })
    }

    fn live(&self) -> Option<Arc<Live>> {
        lock(&self.live).clone()
    }

    /// `why`, with the last line of the log once a dead server's stderr has
    /// ended (at most [`DRAIN`] later: its pipe ends with it).
    fn ended(&self, why: &str, stderr_end: Option<mpsc::Receiver<()>>) -> String {
        if let Some(end) = stderr_end {
            let _ = end.recv_timeout(DRAIN);
        }
        match lock(&self.log).back() {
            Some(last) if !last.starts_with("(Lattice)") => {
                format!("{why} Its last line: {}", cut(last, 300))
            }
            _ => why.to_owned(),
        }
    }
}

/// How long a dead server's last stderr lines are waited for.
const DRAIN: Duration = Duration::from_millis(500);

/// The MCP servers of this process.
pub struct McpHub {
    state: StateRoot,
    env: Arc<dyn Env>,
    approvals: Approvals,
    handle: Handle,
    slots: Mutex<BTreeMap<ServerKey, Arc<Slot>>>,
    changed: Arc<watch::Sender<u64>>,
    timeouts: Timeouts,
}

/// End a server's tree: give it `wait` to end by itself, then close its Job.
fn end_child(child: Option<Child>, wait: Duration) {
    if let Some(child) = child {
        let _ = child.wait(Some(wait));
        drop(child);
    }
}

/// A folder's root as a process's working folder: without `\\?\` when the
/// rest is a drive path (`cmd.exe` refuses a verbatim current folder).
fn plain(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    match text.strip_prefix(r"\\?\") {
        Some(rest) if rest.as_bytes().get(1) == Some(&b':') => PathBuf::from(rest),
        _ => path.to_path_buf(),
    }
}

impl McpHub {
    pub fn new(state: StateRoot, env: Arc<dyn Env>, clock: Clock, handle: Handle) -> Self {
        let (changed, _) = watch::channel(0);
        Self {
            approvals: Approvals::new(&state, clock),
            state,
            env,
            handle,
            slots: Mutex::default(),
            changed: Arc::new(changed),
            timeouts: Timeouts::default(),
        }
    }

    /// The same hub with other waits (tests).
    pub fn with_timeouts(mut self, timeouts: Timeouts) -> Self {
        self.timeouts = timeouts;
        self
    }

    /// The reader's decisions.
    pub fn approvals(&self) -> &Approvals {
        &self.approvals
    }

    /// Bumped at every change; read the servers again when it moves.
    pub fn changed(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    fn bump(&self) {
        self.changed.send_modify(|n| *n = n.wrapping_add(1));
    }

    fn slot(&self, key: &ServerKey) -> Arc<Slot> {
        lock(&self.slots)
            .entry(key.clone())
            .or_insert_with(Slot::new)
            .clone()
    }

    fn existing(&self, key: &ServerKey) -> Option<Arc<Slot>> {
        lock(&self.slots).get(key).cloned()
    }

    /// The scope of a folder's servers.
    pub fn folder_scope(workspace: &Workspace) -> Scope {
        let key = workspace.key();
        Scope::Folder {
            id: key.id,
            path: key.path,
        }
    }

    /// What is declared now: the reader's file, and `folder`'s files when the
    /// caller passes it (only for a trusted folder, FT3). Blocking.
    pub fn declared(&self, folder: Option<(&Workspace, &GitRunner)>) -> Declared {
        let mut declared = config::load_user(&self.state);
        if let Some((workspace, runner)) = folder {
            let scope = Self::folder_scope(workspace);
            match workspace.with_rules(runner, |rules| config::load_folder(rules, &scope)) {
                Ok(found) => {
                    declared.servers.extend(found.servers);
                    declared.problems.extend(found.problems);
                }
                Err(_) => declared.problems.push(Problem {
                    name: String::new(),
                    file: workspace.name.clone(),
                    sentence: "The folder's ignore rules could not be read, so its MCP files were not read.".to_owned(),
                }),
            }
        }
        declared
    }

    fn server_on(&self, entry: &ServerEntry) -> bool {
        !entry.disabled && self.approvals.is_on(&entry.key, WHOLE_SERVER)
    }

    /// The Tools view's servers: `declared` as they are in this process.
    pub fn overview(&self, declared: &Declared) -> Overview {
        let mut taken = HashSet::new();
        let servers = declared
            .servers
            .iter()
            .map(|entry| {
                let slot = self.existing(&entry.key);
                let live = slot
                    .as_ref()
                    .and_then(|slot| slot.live())
                    .filter(|live| live.sha256 == entry.sha256);
                let status = slot
                    .as_ref()
                    .map(|slot| lock(&slot.status).clone())
                    .unwrap_or(ServerStatus::Stopped);
                let status = match (&status, &live) {
                    // A server started from an entry since changed.
                    (ServerStatus::Running, None) => ServerStatus::Stopped,
                    _ => status,
                };
                let tools = live
                    .as_ref()
                    .map(|live| {
                        lock(&live.tools)
                            .iter()
                            .map(|tool| ToolView {
                                name: tool.name.clone(),
                                model_name: names::unique_model_name(
                                    &entry.key.name,
                                    &tool.name,
                                    &mut taken,
                                ),
                                title: tool.title.clone(),
                                description: tool.description.clone(),
                                on: self.approvals.is_on(&entry.key, &tool.name),
                                allowed: self.approvals.allowed(entry, &tool.name),
                                read_only: tool.read_only,
                                destructive: tool.destructive,
                                open_world: tool.open_world,
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                ServerView {
                    key: entry.key.clone(),
                    file: entry.file.clone(),
                    command_line: entry.command_line(),
                    env_names: entry.env_names(),
                    cwd: entry.cwd.clone(),
                    off: !self.server_on(entry),
                    enabled: self.approvals.approved(entry),
                    status,
                    server: live.as_ref().map(|live| {
                        let info = &live.info;
                        let name = format!("{} {}", info.name, info.version);
                        format!("{}, MCP {}", name.trim(), info.protocol)
                    }),
                    instructions: live
                        .as_ref()
                        .and_then(|live| live.info.instructions.clone()),
                    tools,
                    problems: live
                        .as_ref()
                        .map(|live| lock(&live.problems).clone())
                        .unwrap_or_default(),
                    log: slot
                        .as_ref()
                        .map(|slot| lock(&slot.log).iter().cloned().collect())
                        .unwrap_or_default(),
                }
            })
            .collect();
        Overview {
            servers,
            problems: declared.problems.clone(),
            user_file: config::user_file(&self.state).display().to_string(),
        }
    }

    /// Ask the reader to enable `entry` (`ConfirmPort(EnableMcpServer)`),
    /// showing the program it would start; record their yes. `folder` is the
    /// root of the folder that declared it, `workspace` the attached one.
    pub async fn enable(
        &self,
        entry: &ServerEntry,
        folder: Option<&Path>,
        workspace: Option<&Path>,
        confirmer: &Confirmer,
        initiated: Initiated,
    ) -> Result<bool, String> {
        let program = {
            let (entry, env, globals) =
                (entry.clone(), self.env.clone(), self.state.globals.clone());
            let (folder, workspace) = (folder.map(plain), workspace.map(Path::to_path_buf));
            self.handle
                .spawn_blocking(move || {
                    launch::plan(
                        &entry,
                        folder.as_deref(),
                        env.as_ref(),
                        workspace.as_deref(),
                        &globals,
                    )
                    .map(|plan| (plan.program(), launch::shown(&plan.cwd)))
                })
                .await
                .map_err(|_| "The server's program could not be looked up.".to_owned())??
        };
        let request = ConfirmRequest::EnableMcpServer {
            name: entry.key.name.clone(),
            from: entry.file.clone(),
            command_line: entry.command_line(),
            program: program.0,
            cwd: program.1,
            env_names: entry.env_names(),
        };
        let key = format!("enable-mcp:{}:{}", entry.key.id(), entry.sha256);
        if !confirmer.ask(&key, request, initiated).await {
            return Ok(false);
        }
        self.approvals.approve(entry)?;
        self.bump();
        Ok(true)
    }

    /// End `key`'s approval (it asks again before it next starts) and stop it.
    pub async fn disable(&self, key: &ServerKey) -> Result<(), String> {
        self.approvals.revoke(key)?;
        self.stop(key).await;
        Ok(())
    }

    /// Stop `key`'s server, if it runs.
    pub async fn stop(&self, key: &ServerKey) {
        let Some(slot) = self.existing(key) else {
            return;
        };
        let _one = slot.start.lock().await;
        let live = lock(&slot.live).take();
        *lock(&slot.status) = ServerStatus::Stopped;
        if let Some(live) = live {
            let child = live.end();
            let wait = self.timeouts.stop;
            let _ = self
                .handle
                .spawn_blocking(move || end_child(child, wait))
                .await;
        }
        self.bump();
    }

    /// Stop every server (Lattice is closing); blocking for at most the stop
    /// wait per server.
    pub fn stop_all(&self) {
        let slots: Vec<Arc<Slot>> = lock(&self.slots).values().cloned().collect();
        for slot in slots {
            if let Some(live) = lock(&slot.live).take() {
                end_child(live.end(), Duration::ZERO);
            }
            *lock(&slot.status) = ServerStatus::Stopped;
        }
    }

    /// Start `entry`'s server now (the reader asked), if it is enabled.
    pub async fn start(
        &self,
        entry: &ServerEntry,
        folder: Option<&Path>,
        workspace: Option<&Path>,
    ) -> Result<(), String> {
        self.running(entry, folder, workspace).await.map(|_| ())
    }

    /// The running server for `entry`: the one already running from exactly
    /// this entry, or a new start.
    async fn running(
        &self,
        entry: &ServerEntry,
        folder: Option<&Path>,
        workspace: Option<&Path>,
    ) -> Result<Arc<Live>, String> {
        if !self.approvals.approved(entry) {
            return Err("It is not enabled; enable it in Tools.".to_owned());
        }
        let slot = self.slot(&entry.key);
        let _one = slot.start.lock().await;
        if let Some(live) = slot.live() {
            if live.sha256 == entry.sha256 && live.conn.closed().is_none() {
                if live.stale.swap(false, Ordering::SeqCst) {
                    self.relist(&live).await;
                }
                return Ok(live);
            }
            // Its entry changed, or it ended: the old one goes.
            let old = lock(&slot.live).take();
            if let Some(old) = old {
                let child = old.end();
                let wait = self.timeouts.stop;
                let _ = self
                    .handle
                    .spawn_blocking(move || end_child(child, wait))
                    .await;
            }
        }
        *lock(&slot.status) = ServerStatus::Starting;
        self.bump();
        let started = self.start_live(entry, folder, workspace, &slot).await;
        match started {
            Ok(live) => {
                *lock(&slot.live) = Some(live.clone());
                *lock(&slot.status) = ServerStatus::Running;
                self.bump();
                Ok(live)
            }
            Err(why) => {
                *lock(&slot.status) = ServerStatus::Failed(why.clone());
                push_log(&slot.log, format!("(Lattice) {why}"));
                self.bump();
                Err(why)
            }
        }
    }

    async fn relist(&self, live: &Live) {
        match client::list_tools(&live.conn, self.timeouts.list).await {
            Ok((tools, problems)) => {
                *lock(&live.tools) = tools;
                *lock(&live.problems) = problems;
            }
            Err(why) => {
                lock(&live.problems).push(format!("The tool list could not be read again: {why}"))
            }
        }
        self.bump();
    }

    async fn start_live(
        &self,
        entry: &ServerEntry,
        folder: Option<&Path>,
        workspace: Option<&Path>,
        slot: &Arc<Slot>,
    ) -> Result<Arc<Live>, String> {
        let (planned, env, globals) = (entry.clone(), self.env.clone(), self.state.globals.clone());
        let (folder, workspace) = (folder.map(plain), workspace.map(Path::to_path_buf));
        let mut child = self
            .handle
            .spawn_blocking(move || {
                let plan = launch::plan(
                    &planned,
                    folder.as_deref(),
                    env.as_ref(),
                    workspace.as_deref(),
                    &globals,
                )?;
                plan.start()
                    .map_err(|error| format!("The server could not start: {error}."))
            })
            .await
            .map_err(|_| "The server could not start.".to_owned())??;
        let (Some(stdin), Some(stdout)) = (child.take_stdin(), child.take_stdout()) else {
            return Err("The server's pipes could not be opened.".to_owned());
        };
        let stderr_end = child
            .take_stderr()
            .map(|stderr| read_stderr(&entry.key.name, stderr, slot.log.clone()));
        let stale = Arc::new(AtomicBool::new(false));
        let events = {
            let (slot, stale, changed) =
                (Arc::downgrade(slot), stale.clone(), self.changed.clone());
            Arc::new(move |event: Event| on_event(&slot, &stale, &changed, event)) as client::Events
        };
        let conn = Connection::start(&entry.key.name, Box::new(stdout), Box::new(stdin), events)
            .map_err(|_| "The server's connection could not start.".to_owned())?;
        let ready = async {
            let info = client::handshake(&conn, self.timeouts.start).await?;
            let (tools, problems) = client::list_tools(&conn, self.timeouts.list).await?;
            Ok::<_, String>((info, tools, problems))
        }
        .await;
        let (info, tools, problems) = match ready {
            Ok(ready) => ready,
            Err(why) => {
                conn.close();
                let wait = self.timeouts.stop;
                let slot = slot.clone();
                return Err(self
                    .handle
                    .spawn_blocking(move || {
                        end_child(Some(child), wait);
                        slot.ended(&why, stderr_end)
                    })
                    .await
                    .unwrap_or_else(|_| "The server could not start.".to_owned()));
            }
        };
        *lock(&slot.stderr_end) = stderr_end;
        Ok(Arc::new(Live {
            child: Mutex::new(Some(child)),
            conn,
            info,
            tools: Mutex::new(tools),
            problems: Mutex::new(problems),
            stale,
            sha256: entry.sha256.clone(),
        }))
    }

    /// The tools an Agent-mode turn offers from `servers`: each enabled,
    /// switched-on server, started now if it is not running (in parallel), and
    /// each of its switched-on tools. `folder` is the trusted folder's root
    /// (for its servers), `workspace` the attached folder.
    pub async fn turn_tools(
        &self,
        servers: &[ServerEntry],
        folder: Option<&Path>,
        workspace: Option<&Path>,
    ) -> TurnTools {
        let wanted: Vec<&ServerEntry> = servers
            .iter()
            .filter(|entry| self.server_on(entry) && self.approvals.approved(entry))
            .collect();
        let started = futures::future::join_all(wanted.iter().map(|entry| {
            let from = match entry.key.scope {
                Scope::Folder { .. } => folder,
                Scope::User => None,
            };
            self.running(entry, from, workspace)
        }))
        .await;
        let mut out = TurnTools::default();
        let mut taken = HashSet::new();
        for (entry, started) in wanted.into_iter().zip(started) {
            let live = match started {
                Ok(live) => live,
                Err(why) => {
                    out.notices.push(format!(
                        "The MCP server {} did not start, so its tools are not offered: {why}",
                        entry.key.name
                    ));
                    continue;
                }
            };
            let tools = lock(&live.tools).clone();
            for tool in tools {
                if !self.approvals.is_on(&entry.key, &tool.name) {
                    continue;
                }
                if out.tools.len() >= MAX_TURN_TOOLS {
                    out.notices.push(format!(
                        "Only {MAX_TURN_TOOLS} MCP tools are offered at once; switch some off in Tools."
                    ));
                    return out;
                }
                let description = if tool.description.is_empty() {
                    format!("A tool of the MCP server {}.", entry.key.name)
                } else {
                    format!(
                        "(MCP server {}) {}",
                        entry.key.name,
                        cut(&tool.description, MAX_MODEL_DESCRIPTION)
                    )
                };
                out.tools.push(TurnTool {
                    key: entry.key.clone(),
                    sha256: entry.sha256.clone(),
                    file: entry.file.clone(),
                    model_name: names::unique_model_name(&entry.key.name, &tool.name, &mut taken),
                    tool: tool.name.clone(),
                    description,
                    parameters: tool.input_schema.clone(),
                    allowed: self.approvals.allowed(entry, &tool.name),
                });
            }
        }
        out
    }

    /// Call `tool` on the server running from the entry with `sha256`.
    pub async fn call(
        &self,
        key: &ServerKey,
        sha256: &str,
        tool: &str,
        arguments: Value,
    ) -> Result<CallText, String> {
        let live = self
            .existing(key)
            .and_then(|slot| slot.live())
            .filter(|live| live.sha256 == sha256 && live.conn.closed().is_none())
            .ok_or_else(|| "That MCP server is not running any more.".to_owned())?;
        let answer = client::call_tool(&live.conn, tool, arguments, self.timeouts.call)
            .await
            .map_err(|error| error.sentence())?;
        Ok(result::call_text(&answer))
    }

    /// Ask the reader to allow `tool` of `entry` always
    /// (`ConfirmPort(AllowMcpTool)`); record their yes.
    pub async fn allow_always(
        &self,
        entry: &ServerEntry,
        tool: &str,
        confirmer: &Confirmer,
        key: &str,
        initiated: Initiated,
    ) -> Result<bool, String> {
        self.allow_always_for(
            &entry.key,
            &entry.sha256,
            &entry.file,
            tool,
            confirmer,
            key,
            initiated,
        )
        .await
    }

    /// [`McpHub::allow_always`] by the server's key, its entry's hash and the
    /// file that declares it (a turn's pending call knows only these).
    #[allow(clippy::too_many_arguments)]
    pub async fn allow_always_for(
        &self,
        server: &ServerKey,
        sha256: &str,
        from: &str,
        tool: &str,
        confirmer: &Confirmer,
        key: &str,
        initiated: Initiated,
    ) -> Result<bool, String> {
        let request = ConfirmRequest::AllowMcpTool {
            server: server.name.clone(),
            tool: tool.to_owned(),
            from: from.to_owned(),
        };
        if !confirmer.ask(key, request, initiated).await {
            return Ok(false);
        }
        self.approvals.allow_for(server, sha256, tool)?;
        self.bump();
        Ok(true)
    }

    /// End "allow always" for `tool`.
    pub fn disallow(&self, key: &ServerKey, tool: &str) -> Result<(), String> {
        self.approvals.disallow(key, tool)?;
        self.bump();
        Ok(())
    }

    /// Switch one tool (or [`WHOLE_SERVER`]) on or off. A server of the
    /// reader's own file is switched in that file (`"disabled"`).
    pub async fn switch(&self, key: &ServerKey, tool: &str, on: bool) -> Result<(), String> {
        if tool == WHOLE_SERVER && key.scope == Scope::User {
            config::set_user_disabled(&self.state, &key.name, !on)?;
        } else {
            self.approvals.switch(key, tool, on)?;
        }
        if tool == WHOLE_SERVER && !on {
            self.stop(key).await;
        }
        self.bump();
        Ok(())
    }

    /// Add or replace a server in the reader's file. A running server of that
    /// name keeps running until its next use, which starts the new entry
    /// (after the reader enables it).
    pub fn put_user_server(&self, name: &str, entry: &Value) -> Result<(), String> {
        config::put_user_server(&self.state, name, entry)?;
        self.bump();
        Ok(())
    }

    /// Take a server out of the reader's file, and stop it.
    pub async fn remove_user_server(&self, name: &str) -> Result<(), String> {
        config::remove_user_server(&self.state, name)?;
        self.stop(&ServerKey::user(name)).await;
        Ok(())
    }
}

impl Drop for McpHub {
    fn drop(&mut self) {
        // Every Job closes as its child drops; nothing waits here.
        let slots: Vec<Arc<Slot>> = lock(&self.slots).values().cloned().collect();
        for slot in slots {
            if let Some(live) = lock(&slot.live).take() {
                drop(live.end());
            }
        }
    }
}

/// What a connection's events do to its slot.
fn on_event(slot: &Weak<Slot>, stale: &AtomicBool, changed: &watch::Sender<u64>, event: Event) {
    let Some(slot) = slot.upgrade() else {
        return;
    };
    match event {
        Event::ToolsChanged => stale.store(true, Ordering::SeqCst),
        Event::Log(line) => push_log(&slot.log, line),
        Event::Closed(why) => {
            // Only a server that ended by itself, while it was the slot's
            // running one, is a failure; a stop already said Stopped.
            let ended_by_itself = matches!(*lock(&slot.status), ServerStatus::Running)
                && slot.live().is_some_and(|live| live.conn.closed().is_some());
            if ended_by_itself {
                // On the dead connection's own reader thread: waiting a
                // moment for the stderr's last lines blocks nothing else.
                let stderr_end = lock(&slot.stderr_end).take();
                let sentence = slot.ended(&why, stderr_end);
                *lock(&slot.status) = ServerStatus::Failed(sentence);
                if let Some(live) = lock(&slot.live).take() {
                    drop(live.end());
                }
            }
        }
    }
    changed.send_modify(|n| *n = n.wrapping_add(1));
}

/// A thread that keeps the server's stderr, line by line, in its log. The
/// receiver disconnects when the stderr ends.
fn read_stderr(name: &str, stderr: impl Read + Send + 'static, log: Log) -> mpsc::Receiver<()> {
    let (done, end) = mpsc::channel::<()>();
    let short: String = name.chars().take(24).collect();
    let _ = std::thread::Builder::new()
        .name(format!("lattice-mcp-err {short}"))
        .spawn(move || {
            let _done = done;
            let mut reader = BufReader::new(stderr);
            let mut line = Vec::new();
            loop {
                line.clear();
                match reader.by_ref().take(16 * 1024).read_until(b'\n', &mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(&line);
                        let text = text.trim_end();
                        if !text.is_empty() {
                            push_log(&log, cut(text, 2000));
                        }
                    }
                }
            }
        });
    end
}
