//! `AgentChat`: the agent chat's service, [`AgentChatService`]
//! (the chat core's spec §2.4, §4, §5, §6, §7, §9, §10, §11; row E11).
//! Not a port.
//!
//! It composes the parts the earlier rows built, over one transcript store
//! (the shared store in a shipped build, whose writes are open since row G5;
//! `MemoryTranscriptStore` in tests and the development host):
//!
//! - **A send** (§2.4) is validated, then the conversation is claimed: a
//!   send while it runs a turn is **queued** (§4.4), recorded as `Queued`.
//!   **Prepare**, before any write: (a) the choice is resolved once and held
//!   to what the reader was shown (N10); (b) the context is read; (c) the
//!   secret tripwire runs over what a target off the machine would receive
//!   (N4); then **T1**: the first remote send of a conversation, and the
//!   first after its remote label or its workspace changed, needs the native
//!   `FirstRemoteSend` dialog, which states how many earlier tool results
//!   read on this machine would be sent; refused, nothing is written and
//!   no model is called. Only then (d) the user turn is written, the sidecar
//!   bound to the shared row by its `created` (`convo::binding`), and
//!   `TurnStart` recorded.
//! - **The turn kind** (§2.4 step 3): with no workspace, or a model that
//!   cannot take tools (a managed model whose tool-call probe failed, the
//!   echo, an endpoint this process saw refuse tools), the plain chat's
//!   turn, unchanged (`chat::ChatCore`), its events forwarded into the
//!   conversation; otherwise an **agent turn** (`convo::turn`). Agent mode
//!   needs a trusted folder (FT3).
//! - **Stop, steer, continue, interrupted** (§4.4, §4.6): Stop cancels the
//!   run at once, every command it started and every wait; a steer passes
//!   N4 first, and a steer the run no longer takes, or any it accepted but
//!   never sent, is queued; Continue replays up to the last tool result with
//!   no new user item; a conversation reopened with a `TurnStart` and no
//!   `TurnEnd` gets `TurnEnd{Interrupted}`, and its queued messages come
//!   back as drafts (`Dequeued{Restored}`).
//! - **T2**: `pin`, `set_mode`, `attach_workspace` and `archive` are
//!   refused while a turn runs.
//! - **Follow** (§11.3): `convo::follow`'s coalescer over the conversation's
//!   log.
//! - The workspace, trust, files, changes, review, checkpoints, restore and
//!   permissions calls go to the modules that own them; a review's outcome
//!   (a conflict, what was kept and undone, the reader's notes) reaches the
//!   agent at its next model call.
//! - **Archive** is the reader's archive (`chat::archived_by_reader`): no
//!   delete exists (ND1).
//!
//! Nothing here prints, logs or reads the environment.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use lattice_agents::RunControl;
use lattice_protocol::chat::{ChatChoice, ChatEventKind, ChatService, LocalRuntime, Shown};
use lattice_protocol::conversation::{
    Accepted, AgentChatService, AllowEntry, ArchiveKey, ArtifactKind, ArtifactView, CallId,
    ChangeSet, ChatList, CheckpointId, CheckpointView, ConversationEvent, ConversationEventKind,
    ConversationSummary, CoreStatus, DecidedBy, Decision, DequeueOutcome, FileDiff, Lines,
    MatchScope, Mode, Origin, QueuedMessage, RankedPath, RegenerateRequest, ReviewOp,
    ReviewOutcome, SendRequest, Snapshot, TrustState, TurnKind, TurnStatus, ViewRef, WorkspaceView,
    is_conversation_id,
};
use lattice_protocol::{Refusal, RefusalKind};
use tokio::runtime::Handle;
use tokio::sync::oneshot;

use super::caps::{ModelCaps, Tri};
use super::follow::{self, Changed, Gap};
use super::item::Item;
use super::log::ConversationLog;
use super::replay;
use super::sidecar::{NewMeta, Sidecar, SidecarStore, WorkspaceRef};
use super::tripwire::{Tripwire, Withheld};
use super::{binding, turn, views};
use crate::changes::checkpoint::Checkpoints;
use crate::chat::archived_by_reader;
use crate::chat::store::{IndexRow, IndexState, SharedThreadStore, StoredTurn, normalize_title};
use crate::chat::transcript::{StoreError, TranscriptStore};
use crate::chat::vocab::{self, LocalView, Resolution, Target};
use crate::chat::{self, ChatConfig, ChatCore, Locked, answer, answer_lock, refusals};
use crate::clock::{Clock, system_clock};
use crate::env::Env;
use crate::exec::allowlist::Permissions;
use crate::exec::run::{CommandSlots, Launcher, SpawnLauncher, StopHandle};
use crate::git::runner::GitRunner;
use crate::keys::KeyStore;
use crate::llama::ManagedRuntime;
use crate::llama::files::LlamaPaths;
use crate::mcp::McpHub;
use crate::models::ModelFactory;
use crate::ports::{AttentionPort, ConfirmPort, ConfirmRequest, Confirmer, Initiated};
use crate::secrets::looks_like_secret;
use crate::staging::Staging;
use crate::state::{Platform, StateRoot};
use crate::tools::ask::Questions;
use crate::workspace::Workspace;
use crate::workspace::lease::WriterLease;
use crate::workspace::trust::TrustStore;

/// The agent chat's one-sentence refusals. Those the specification does not
/// word are PROVISIONAL.
pub mod words {
    /// PROVISIONAL.
    pub const RUNNING: &str = "A turn is running in this conversation; wait for it, or stop it.";
    /// PROVISIONAL.
    pub const TOO_MANY: &str = "Three agent turns are running; wait for one to finish.";
    /// PROVISIONAL.
    pub const NOT_CONFIRMED_REMOTE: &str =
        "You did not confirm sending this conversation off this machine, so nothing was sent.";
    /// §6.3 FT3 (the policy engine's sentence).
    pub const TRUST_FIRST: &str = "Trust this folder to use Agent mode.";
    /// PROVISIONAL.
    pub const NO_WORKSPACE: &str = "That folder is not attached here.";
    /// PROVISIONAL: a folder's tool in a turn without a folder.
    pub const NO_FOLDER_TOOL: &str =
        "No folder is attached to this conversation, so there are no files or commands here.";
    /// PROVISIONAL.
    pub const NOT_OPEN: &str = "Open the conversation first.";
    /// PROVISIONAL.
    pub const NOT_PENDING: &str = "That call is not waiting for a decision.";
    /// PROVISIONAL.
    pub const NOTHING_TO_CONTINUE: &str = "There is no turn to continue.";
    /// PROVISIONAL.
    pub const NOT_WITHHELD: &str = "Nothing of that call is withheld.";
    /// PROVISIONAL.
    pub const RELEASE_REFUSED: &str = "You did not confirm sending it, so it stays withheld.";
    /// §4.5.
    pub const NO_TOOLS: &str =
        "This model did not accept tools, so Agent mode is off for it in this session.";
    /// LR8′ (PROVISIONAL wording): the managed server's tool-call probe failed.
    pub const LOCAL_NO_TOOLS: &str = "This model did not pass Lattice's check that it can call tools, so its Agent-mode turns are plain answers from now on. Send again for an answer.";
    /// LR8′ (PROVISIONAL wording): the probe passed.
    pub const PROBE_PASSED: &str =
        "Lattice checked that this model calls tools well: it does, so it gets the agent's tools.";
    /// PROVISIONAL.
    pub const NO_VISION: &str = "This model did not accept images, so attached images and the agent's browser are off for it in this session.";
    /// §5.7.2: a sidecar write failed.
    pub const NOT_SAVED: &str = "This turn's tool history could not be saved.";
    /// T15.
    pub const ANSWER_REDACTED: &str =
        "Parts of the answer that looked like secrets were replaced before it was saved.";
    /// PROVISIONAL: a plain turn in a project with instructions or files.
    pub const PROJECT_PLAIN: &str = "This turn is a plain answer, which does not carry this project's instructions or reference files; a model that can call tools here gets them.";
    /// §4.4 N10 for a queued message.
    pub const QUEUED_MOVED: &str =
        "A queued message was not sent: where its model runs has changed. Send it again.";
}

/// The most agent turns running at once, across conversations (§2.4).
pub const MAX_AGENT_TURNS: u32 = 3;
/// `RunConfig::max_turns` for an agent turn (§1.2).
pub const MAX_TURNS: u32 = 40;

/// What a model can do, as a test or the Models increment says (§4.5).
pub type CapsFn = Arc<dyn Fn(&Resolution) -> ModelCaps + Send + Sync>;

/// What the agent chat is built from.
#[derive(Clone)]
pub struct AgentConfig {
    pub state: StateRoot,
    pub env: Arc<dyn Env>,
    pub development: bool,
    pub clock: Clock,
    pub store: Arc<dyn TranscriptStore>,
    pub store_dir: String,
    pub local: Arc<dyn ManagedRuntime>,
    pub paths: LlamaPaths,
    pub model_factory: Option<ModelFactory>,
    pub confirm: Arc<dyn ConfirmPort>,
    pub attention: Arc<dyn AttentionPort>,
    pub gap: Gap,
    /// What a resolved choice can do. `None`: the managed server's tools are
    /// its tool-call probe's record (LR8′: `Unknown` until it has run), the
    /// echo has none, an endpoint is `Unknown` until the Models increment.
    pub caps: Option<CapsFn>,
    /// How a command starts (tests record it).
    pub launcher: Option<Arc<dyn Launcher>>,
    pub max_turns: u32,
    /// The git runner (tests give a scratch home).
    pub runner: Option<Arc<GitRunner>>,
    /// How the agent's browser starts (tests: headless, on a page they serve).
    pub browser: crate::browser::BrowserConfig,
    /// The desktop auto mode drives: `None`, the system's (tests give a fake).
    pub desktop: Option<Arc<dyn crate::desktop::Driver>>,
    /// The stop hotkey's modifiers and virtual key (`None`: Ctrl+Alt+End;
    /// tests give keys nobody presses).
    pub stop_keys: Option<(u32, u32)>,
}

impl AgentConfig {
    /// The shipped composition: the shared store under `<globals>`, its
    /// writes open since row G5.
    pub fn new(
        state: StateRoot,
        env: Arc<dyn Env>,
        local: Arc<dyn ManagedRuntime>,
        confirm: Arc<dyn ConfirmPort>,
        attention: Arc<dyn AttentionPort>,
    ) -> Self {
        let dir = state.chat_dir();
        Self {
            store_dir: dir.display().to_string(),
            store: Arc::new(SharedThreadStore::new(dir)),
            paths: LlamaPaths::from_env(env.as_ref()),
            state,
            env,
            development: false,
            clock: system_clock(),
            local,
            model_factory: None,
            confirm,
            attention,
            gap: Gap::default(),
            caps: None,
            launcher: None,
            max_turns: MAX_TURNS,
            runner: None,
            browser: crate::browser::BrowserConfig::default(),
            desktop: None,
            stop_keys: None,
        }
    }
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// S25: `touch` the thread, so a long wait cannot make it the least
/// recently used one that the 60-thread cap archives. Called at
/// `TurnStart`, after each approval or question is resolved, and
/// immediately before `append_answer` (§2.4).
pub(crate) fn touch(store: &dyn TranscriptStore, id: &str) {
    let _ = store.touch(id);
}

pub(crate) fn refuse(kind: RefusalKind, message: &str) -> Refusal {
    Refusal::new(kind, message)
}

/// A call waiting for the reader's decision.
pub(crate) struct PendingApproval {
    pub turn: String,
    pub sender: Option<oneshot::Sender<lattice_agents::ApprovalDecision>>,
    pub pending: Pending,
}

/// What a pending call is.
#[derive(Clone)]
pub(crate) enum Pending {
    /// `run_command`: what its Approve runs, once the native dialog says yes.
    Command(crate::exec::run::Prepared),
    /// An MCP tool's call (§12): approved through `ConfirmPort(McpCall)`.
    Mcp(super::turn::McpCall),
    /// A browser action whose effect asks: approved through
    /// `ConfirmPort(BrowserAct)`.
    Browser(super::turn::BrowserCall),
    /// A desktop action in auto mode that moves money or changes an account:
    /// approved through `ConfirmPort(DesktopAct)`.
    Desktop(super::turn::DesktopCall),
}

/// A call, once its approval is decided.
pub(crate) enum Ready {
    Approved(Box<(crate::exec::run::Prepared, crate::exec::run::Approval)>),
    /// An MCP tool's call the reader, or its allowlist, approved.
    Mcp,
    /// A browser action whose effect asked, which the reader approved.
    Browser,
    /// A desktop action that asked, which the reader approved.
    Desktop,
    /// It does not run, for this reason (the tool's error).
    Refused(String),
}

/// The turn a conversation runs.
pub(crate) struct Running {
    pub turn: String,
    pub control: Option<RunControl>,
    /// A plain turn's job in the plain chat.
    pub job: Option<String>,
    pub affirmatively_local: bool,
    pub label: String,
    /// What the agent must hear at its next model call.
    pub notes: Arc<Mutex<Vec<String>>>,
    pub stop_requested: bool,
}

/// One conversation's state in this process.
pub(crate) struct State {
    /// Messages Stop hooks sent in a row (`super::hooked`).
    pub hook_follow_ups: u32,
    pub sidecar: Option<Arc<Sidecar>>,
    pub staging: Option<Arc<Staging>>,
    pub checkpoints: Option<Arc<Checkpoints>>,
    pub workspace: Option<Workspace>,
    pub lease: Option<Arc<WriterLease>>,
    pub mode: Mode,
    pub origin: Origin,
    pub running: Option<Running>,
    pub queue: VecDeque<QueuedMessage>,
    pub approvals: HashMap<CallId, PendingApproval>,
    /// A command call's state after its approval was decided: what the
    /// tool's handler runs, or why it does not.
    pub ready: HashMap<CallId, Ready>,
    pub stops: HashMap<CallId, StopHandle>,
    /// (label, workspace id, earlier local results) the reader confirmed
    /// sending to (T1): a later local result asks again.
    pub remote_ok: Option<(String, Option<String>, u32)>,
    /// What the agent must hear at its next turn.
    pub notes: Vec<String>,
    /// The last turn's user record and how it ended (Continue).
    pub last: Option<(String, TurnStatus)>,
    /// The choice of the last turn (`ModelSwitch`).
    pub last_choice: Option<String>,
    /// A sidecar write failed this turn (the notice is shown once).
    pub save_failed: bool,
}

/// One conversation.
pub(crate) struct Convo {
    pub id: String,
    pub log: Arc<ConversationLog>,
    pub tripwire: Arc<Tripwire>,
    pub state: Mutex<State>,
}

struct LogWithheld(Arc<ConversationLog>);

impl Withheld for LogWithheld {
    fn withheld(&self, call_id: &str) {
        self.0.push(ConversationEventKind::Withheld {
            call_id: call_id.to_owned(),
        });
    }
}

impl Convo {
    fn new(id: &str, clock: Clock) -> Arc<Self> {
        let log = Arc::new(ConversationLog::new(clock));
        Arc::new(Self {
            id: id.to_owned(),
            tripwire: Arc::new(Tripwire::new(Arc::new(LogWithheld(log.clone())))),
            log,
            state: Mutex::new(State {
                sidecar: None,
                staging: None,
                checkpoints: None,
                workspace: None,
                lease: None,
                mode: Mode::Ask,
                origin: Origin::Web,
                running: None,
                queue: VecDeque::new(),
                approvals: HashMap::new(),
                ready: HashMap::new(),
                stops: HashMap::new(),
                remote_ok: None,
                notes: Vec::new(),
                last: None,
                last_choice: None,
                save_failed: false,
                hook_follow_ups: 0,
            }),
        })
    }

    pub(crate) fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Append `item` to the sidecar, if there is one; a failure is told once.
    pub(crate) fn record(&self, item: &Item) -> bool {
        let sidecar = self.state().sidecar.clone();
        let Some(sidecar) = sidecar else {
            return false;
        };
        match sidecar.append(item) {
            Ok(appended) => appended.redacted,
            Err(_) => {
                let first = {
                    let mut state = self.state();
                    !std::mem::replace(&mut state.save_failed, true)
                };
                if first {
                    self.log.push(ConversationEventKind::Notice {
                        text: words::NOT_SAVED.to_owned(),
                    });
                }
                false
            }
        }
    }
}

/// What the agent chat holds.
pub(crate) struct Inner {
    pub config: AgentConfig,
    pub handle: Handle,
    pub keys: KeyStore,
    pub chat: ChatCore,
    pub sidecars: SidecarStore,
    pub runner: Arc<GitRunner>,
    pub trust: TrustStore,
    pub permissions: Permissions,
    pub confirmer: Confirmer,
    pub slots: CommandSlots,
    pub questions: Questions,
    pub launcher: Arc<dyn Launcher>,
    pub workspaces: Mutex<HashMap<String, Workspace>>,
    pub convos: Mutex<HashMap<String, Arc<Convo>>>,
    pub agent_turns: AtomicU32,
    /// Endpoint choices that refused tools in this process (§4.5).
    pub no_tools: Mutex<HashSet<String>>,
    /// Choices that refused a screenshot in this process: no browser for them.
    pub no_vision: Mutex<HashSet<String>>,
    /// The tool-call probe's answers in this process, by key (LR8′): what the
    /// record says too, and an answer the record could not keep.
    pub tool_probes: Mutex<HashMap<String, bool>>,
    /// The managed binary's SHA-256, by its path, size and last write.
    pub binary_sha: Mutex<Option<(PathBuf, u64, Option<std::time::SystemTime>, String)>>,
    pub changed: Changed,
    pub claims: Mutex<HashSet<String>>,
    /// The agent turns' traces (`<native>/chat/runs`).
    pub runs: crate::store::RunStore,
    /// The MCP servers of this process (§12).
    pub mcp: Arc<McpHub>,
    /// The agent's browser (started at its first action).
    pub browser: Arc<crate::browser::AgentBrowser>,
    /// The preview browser, with no network (started at the first preview of
    /// an artifact).
    pub preview: Arc<crate::browser::preview::PreviewBrowser>,
    /// The whole desktop, for auto mode.
    pub desktop: Arc<crate::desktop::Desktop>,
    /// The reader's projects.
    pub projects: Arc<crate::projects::Projects>,
    /// The notes beside folders' files, an agent's among them.
    pub notes: Arc<crate::notes::Notes>,
    /// Held while a task the agent suggested is suggested or settled, and
    /// naming the tasks being started (`<conversation>/<task>`): a task
    /// settles once (`super::tasks`).
    pub tasks: Mutex<HashSet<String>>,
    /// Held while a plan the agent proposed is proposed or settled, and
    /// naming the plans being approved (`<conversation>/<plan>`): a plan
    /// settles once (`super::plans`).
    pub plans: Mutex<HashSet<String>>,
    /// The reader's plugins.
    pub plugins: Arc<crate::plugins::Plugins>,
    /// Held while an artifact's version is numbered and written
    /// (`super::artifacts`).
    pub artifacts: Mutex<()>,
    /// The background commands (`super::background`).
    pub background: super::background::Background,
    /// The labs' agents' sessions (`crate::acp::pool`), also the chat core's.
    pub agent_pool: Arc<crate::acp::pool::Pool>,
}

impl Inner {
    pub(crate) fn now(&self) -> f64 {
        (self.config.clock)()
    }

    fn view(&self) -> LocalView {
        let mut view = LocalView::read_at(self.config.paths.clone(), &self.config.state);
        view.running = self.config.local.running();
        view.failed = self.config.local.failed();
        view
    }

    /// Prepare (a): the choice resolved once, held to what was shown (N10).
    pub(crate) fn resolve(&self, choice: &str, shown: &Shown) -> Result<Resolution, Refusal> {
        let resolution = vocab::resolve(
            choice,
            self.config.env.as_ref(),
            &self.config.state,
            &self.keys,
            self.config.development,
            &self.view(),
        );
        if chat::moved(shown, &resolution.shown) {
            return Err(refuse(RefusalKind::Conflict, refusals::MOVED));
        }
        Ok(resolution)
    }

    /// §4.5: what the resolved choice can do.
    pub(crate) fn caps(&self, resolution: &Resolution) -> ModelCaps {
        let mut caps = if let Some(caps) = &self.config.caps {
            let mut caps = caps(resolution);
            if lock(&self.no_tools).contains(&resolution.id) {
                caps.tools = Tri::No;
            }
            caps
        } else {
            match &resolution.target {
                Target::Endpoint(_) if !lock(&self.no_tools).contains(&resolution.id) => {
                    ModelCaps::UNKNOWN
                }
                // The managed server takes tools once its tool-call probe has
                // passed for this binary and model (LR8′); before the probe,
                // `Unknown`: the first Agent-mode turn runs it.
                Target::Managed(model) => ModelCaps {
                    tools: match self
                        .tool_probe_key(model)
                        .and_then(|key| self.tool_probe(&key))
                    {
                        Some(true) => Tri::Yes,
                        Some(false) => Tri::No,
                        None => Tri::Unknown,
                    },
                    vision: Tri::Unknown,
                    context_tokens: None,
                },
                // The echo and a refusal take no tools.
                _ => ModelCaps {
                    tools: Tri::No,
                    vision: Tri::Unknown,
                    context_tokens: None,
                },
            }
        };
        // A choice that refused a screenshot is not shown one again.
        if lock(&self.no_vision).contains(&resolution.id) {
            caps.vision = Tri::No;
        }
        caps
    }

    /// The tool-call probe's key for `model` on the managed binary (Python's
    /// `_probe_key`); `None` when the binary or the model cannot be read. The
    /// binary's SHA-256 is read once per path, size and last write.
    pub(crate) fn tool_probe_key(&self, model: &crate::llama::files::LocalModel) -> Option<String> {
        let binary = &self.config.paths.binary;
        let meta = std::fs::metadata(binary).ok()?;
        let stamp = (binary.clone(), meta.len(), meta.modified().ok());
        let sha = {
            let mut cached = lock(&self.binary_sha);
            match cached.as_ref() {
                Some((path, size, modified, sha))
                    if (path, size, modified) == (&stamp.0, &stamp.1, &stamp.2) =>
                {
                    sha.clone()
                }
                _ => {
                    let sha = crate::llama::probes::binary_sha256(binary)?;
                    *cached = Some((stamp.0, stamp.1, stamp.2, sha.clone()));
                    sha
                }
            }
        };
        crate::llama::probes::probe_key(&sha, &model.path)
    }

    /// The tool-call probe's answer for `key`: this process's, else the record's.
    pub(crate) fn tool_probe(&self, key: &str) -> Option<bool> {
        if let Some(passed) = lock(&self.tool_probes).get(key) {
            return Some(*passed);
        }
        crate::llama::probes::tool_probe(&self.config.paths, key)
    }

    pub(crate) fn convo(&self, id: &str) -> Arc<Convo> {
        let mut convos = lock(&self.convos);
        convos
            .entry(id.to_owned())
            .or_insert_with(|| Convo::new(id, self.config.clock.clone()))
            .clone()
    }

    pub(crate) fn loaded(&self, id: &str) -> Option<Arc<Convo>> {
        lock(&self.convos).get(id).cloned()
    }

    fn row(&self, id: &str) -> Result<IndexRow, Refusal> {
        binding::listed_row(self.config.store.as_ref(), id).map_err(|error| match error {
            StoreError::NotFound => refuse(RefusalKind::NotFound, refusals::GONE),
            other => other.refusal(),
        })
    }

    /// Open the sidecar of a listed thread for writing, bound by `created`
    /// (§5.5), and load what it holds into `convo` once.
    pub(crate) fn bind(&self, convo: &Convo, new: NewMeta) -> Result<Arc<Sidecar>, Refusal> {
        if let Some(sidecar) = convo.state().sidecar.clone() {
            return Ok(sidecar);
        }
        let (_, sidecar, _) =
            binding::bind_for_writing(self.config.store.as_ref(), &self.sidecars, &convo.id, new)
                .map_err(|error| match error {
                binding::BindError::Store(error) => error.refusal(),
                binding::BindError::Sidecar(_) => refuse(
                    RefusalKind::Unavailable,
                    "Lattice could not open this conversation's record here.",
                ),
            })?;
        let sidecar = Arc::new(sidecar);
        let items = self
            .sidecars
            .read_items(&convo.id)
            .map(|log| log.items)
            .unwrap_or_default();
        let meta = self.sidecars.read_meta(&convo.id).ok().flatten();
        let mut state = convo.state();
        state.staging = Some(Arc::new(Staging::from_items(sidecar.clone(), &items)));
        state.checkpoints = Some(Arc::new(Checkpoints::from_items(
            sidecar.clone(),
            &items,
            self.config.clock.clone(),
        )));
        if let Some(meta) = meta {
            state.mode = meta.mode;
            state.origin = meta.origin;
            if state.workspace.is_none()
                && let Some(reference) = meta.workspace
                && let Some(workspace) = self.reattach(&reference)
            {
                state.lease = Some(Arc::new(self.lease_for(&workspace)));
                state.workspace = Some(workspace);
            }
        }
        state.sidecar = Some(sidecar.clone());
        Ok(sidecar)
    }

    /// The folder a record names, attached again only when it is still the
    /// same folder (id and path).
    fn reattach(&self, reference: &WorkspaceRef) -> Option<Workspace> {
        if let Some(found) = lock(&self.workspaces).get(&reference.id) {
            return Some(found.clone());
        }
        let workspace = crate::workspace::attach::attach_path(
            &PathBuf::from(&reference.path),
            self.config.env.as_ref(),
            &self.config.state,
            &self.runner,
        )
        .ok()?;
        (workspace.id == reference.id).then(|| {
            lock(&self.workspaces).insert(workspace.id.clone(), workspace.clone());
            workspace
        })
    }

    pub(crate) fn lease_for(&self, workspace: &Workspace) -> WriterLease {
        WriterLease::for_workspace(
            workspace,
            &self.runner,
            &self.config.state,
            Platform::host(),
        )
    }

    pub(crate) fn write_meta(&self, convo: &Convo) {
        let state = convo.state();
        let Some(sidecar) = state.sidecar.clone() else {
            return;
        };
        let meta = super::sidecar::Meta {
            v: 1,
            id: convo.id.clone(),
            created: sidecar.created(),
            workspace: state.workspace.as_ref().map(|workspace| WorkspaceRef {
                id: workspace.id.clone(),
                path: workspace.root.to_string_lossy().into_owned(),
            }),
            mode: state.mode,
            origin: state.origin,
        };
        drop(state);
        let _ = sidecar.write_meta(&meta);
    }

    /// The conversation as the list shows it.
    pub(crate) fn summary(&self, row: &IndexRow) -> ConversationSummary {
        let convo = self.loaded(&row.id);
        let (workspace, mode, origin, running, needs_you) = match &convo {
            Some(convo) => {
                let state = convo.state();
                let needs_you =
                    !state.approvals.is_empty() || !self.questions.waiting(&convo.id).is_empty();
                let badge = state.workspace.as_ref().map(|workspace| {
                    workspace.badge(self.trust.state(workspace) == TrustState::Trusted)
                });
                (
                    badge,
                    state.mode,
                    if state.sidecar.is_some() {
                        state.origin
                    } else {
                        Origin::Web
                    },
                    state.running.is_some(),
                    needs_you,
                )
            }
            None => {
                let view = binding::view(self.config.store.as_ref(), &self.sidecars, &row.id)
                    .map(|(_, view)| view)
                    .ok();
                let meta = match &view {
                    Some(binding::SidecarView::Bound(Some(meta))) => Some(meta.clone()),
                    _ => None,
                };
                let badge = meta.as_ref().and_then(|meta| {
                    meta.workspace.as_ref().map(|reference| {
                        crate::workspace::shown_path(std::path::Path::new(&reference.path))
                    })
                });
                (
                    badge.map(|path| lattice_protocol::conversation::WorkspaceBadge {
                        id: meta
                            .as_ref()
                            .and_then(|meta| meta.workspace.as_ref())
                            .map(|reference| reference.id.clone())
                            .unwrap_or_default(),
                        name: std::path::Path::new(&path)
                            .file_name()
                            .map(|name| name.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                        path,
                        trusted: false,
                    }),
                    meta.as_ref().map_or(Mode::Ask, |meta| meta.mode),
                    view.map_or(Origin::Web, |view| view.origin()),
                    false,
                    false,
                )
            }
        };
        ConversationSummary {
            id: row.id.clone(),
            title: row.title.clone(),
            created: row.created,
            updated: row.updated,
            turns: u64::try_from(row.turns).unwrap_or(0),
            pinned_provider: row.pinned_provider.clone(),
            workspace,
            mode,
            origin,
            running,
            needs_you,
        }
    }

    /// Load a listed conversation once: its record, its events, and §4.6's
    /// interrupted turn.
    pub(crate) fn load(&self, id: &str) -> Result<Arc<Convo>, Refusal> {
        let row = self.row(id)?;
        if let Some(convo) = self.loaded(id) {
            return Ok(convo);
        }
        let convo = self.convo(id);
        let (_, view) = binding::view(self.config.store.as_ref(), &self.sidecars, id)
            .map_err(|_| refuse(RefusalKind::Unavailable, refusals::GONE))?;
        if let binding::SidecarView::Bound(meta) = view {
            let origin = meta.as_ref().map_or(Origin::Native, |meta| meta.origin);
            self.bind(
                &convo,
                NewMeta {
                    workspace: None,
                    mode: Mode::Ask,
                    origin,
                },
            )?;
            let items = self
                .sidecars
                .read_items(id)
                .map(|log| log.items)
                .unwrap_or_default();
            views::seed(&convo, &items);
            // §4.6: a turn that started and never ended was interrupted.
            let open = views::open_turn(&items);
            if let Some(turn) = open {
                let end = Item::TurnEnd {
                    turn: turn.clone(),
                    answer: None,
                    status: TurnStatus::Interrupted,
                    run_id: String::new(),
                    at: self.now(),
                };
                convo.record(&end);
                convo.log.push(ConversationEventKind::TurnEnded {
                    turn: turn.clone(),
                    status: TurnStatus::Interrupted,
                });
                convo.state().last = Some((turn, TurnStatus::Interrupted));
            } else {
                convo.state().last = views::last_end(&items);
            }
            // Queued messages come back as drafts, never sent by themselves.
            for (queued_id, _) in views::still_queued(&items) {
                let item = Item::Dequeued {
                    queued_id: queued_id.clone(),
                    outcome: DequeueOutcome::Restored,
                    at: self.now(),
                };
                convo.record(&item);
                convo.log.push(ConversationEventKind::Dequeued {
                    queued_id,
                    outcome: DequeueOutcome::Restored,
                });
            }
            convo.state().last_choice = Some(row.pinned_provider.clone());
        }
        Ok(convo)
    }

    pub(crate) fn note_changed(&self, id: &str) {
        self.changed.note(id);
    }

    /// The running agent turns, across conversations.
    pub(crate) fn agent_turns(&self) -> u32 {
        self.agent_turns.load(Ordering::SeqCst)
    }
}

/// A thread claimed by one send while it prepares.
struct Claim {
    inner: Arc<Inner>,
    id: String,
}

impl Drop for Claim {
    fn drop(&mut self) {
        lock(&self.inner.claims).remove(&self.id);
    }
}

fn claim(inner: &Arc<Inner>, id: &str) -> Option<Claim> {
    lock(&inner.claims).insert(id.to_owned()).then(|| Claim {
        inner: inner.clone(),
        id: id.to_owned(),
    })
}

/// What Prepare found, before anything is written.
pub(crate) struct Checked {
    pub resolution: Resolution,
    pub kind: TurnKind,
    pub workspace: Option<Workspace>,
    /// T1: the dialog to show first, when one is due.
    pub confirm: Option<(String, u32)>,
    /// T1: the earlier tool results read on this machine now.
    pub local_n: u32,
    pub plain_notice: bool,
    /// A plain turn of a project with instructions or files: the plain
    /// pipeline does not carry them, and the turn says so.
    pub project_plain: bool,
    /// Images go with this send to a model off this PC: asked every time
    /// (`super::images`), with their count.
    pub images_remote: Option<(String, u32)>,
}

/// What a send asks to run.
#[derive(Clone)]
pub(crate) enum Ask {
    Send(SendRequest),
    Regenerate(RegenerateRequest),
    Continue { id: String, shown: Shown },
}

impl Ask {
    pub(crate) fn id(&self) -> Option<&str> {
        match self {
            Ask::Send(request) => request.conversation.as_deref(),
            Ask::Regenerate(request) => Some(&request.conversation),
            Ask::Continue { id, .. } => Some(id),
        }
    }

    pub(crate) fn choice(&self, last: Option<&str>) -> String {
        match self {
            Ask::Send(request) => request.choice.clone(),
            Ask::Regenerate(request) => request.choice.clone(),
            Ask::Continue { .. } => last.unwrap_or_default().to_owned(),
        }
    }

    fn shown(&self) -> &Shown {
        match self {
            Ask::Send(request) => &request.shown,
            Ask::Regenerate(request) => &request.shown,
            Ask::Continue { shown, .. } => shown,
        }
    }

    pub(crate) fn mode(&self, current: Mode) -> Mode {
        match self {
            Ask::Send(request) => request.mode,
            Ask::Regenerate(request) => request.mode,
            Ask::Continue { .. } => current,
        }
    }
}

pub(crate) fn validate_send(request: &SendRequest, development: bool) -> Result<(), Refusal> {
    let length = request.text.chars().count();
    if length == 0 || request.text.trim().is_empty() {
        return Err(refuse(RefusalKind::Invalid, refusals::EMPTY_MESSAGE));
    }
    if length > vocab::MAX_MESSAGE_CHARS {
        return Err(refuse(RefusalKind::Invalid, refusals::LONG_MESSAGE));
    }
    validate_choice(&request.choice, development)?;
    if let Some(id) = &request.conversation
        && !is_conversation_id(id)
    {
        return Err(refuse(RefusalKind::NotFound, refusals::GONE));
    }
    super::images::check(&request.images)?;
    Ok(())
}

fn validate_choice(choice: &str, development: bool) -> Result<(), Refusal> {
    if choice == vocab::CLOUD {
        return Err(refuse(RefusalKind::Invalid, vocab::words::CLOUD_REFUSAL));
    }
    if !vocab::is_valid_choice(choice, development) {
        return Err(refuse(RefusalKind::Invalid, vocab::words::UNKNOWN_CHOICE));
    }
    Ok(())
}

/// The agent chat (see the module header).
#[derive(Clone)]
pub struct AgentChat {
    pub(crate) inner: Arc<Inner>,
}

impl AgentChat {
    /// The agent's notes about the folder `workspace` (its id), oldest first
    /// (`super::memory`): what CENTCOM's view of them shows.
    pub fn memory(&self, workspace: &str) -> Vec<super::memory::Note> {
        super::memory::list(&super::memory::file(&self.inner.config.state, workspace))
    }

    /// The reader's Forget on one of them; `Ok(false)` when none has that id.
    pub fn forget_memory(&self, workspace: &str, id: &str) -> Result<bool, String> {
        super::memory::remove(&super::memory::file(&self.inner.config.state, workspace), id)
    }

    /// The reader's hooks as a turn in the folder `workspace` (its id) would
    /// read them, and each one's allowance (`crate::hooks`): what CENTCOM's view
    /// of them shows. Blocking (it reads files).
    pub fn hooks(&self, workspace: Option<&str>) -> (crate::hooks::Found, Vec<bool>) {
        let inner = &self.inner;
        let folder = workspace.and_then(|id| lock(&inner.workspaces).get(id).cloned());
        let trusted = folder.as_ref().is_none_or(|w| {
            inner.trust.state(w) == lattice_protocol::conversation::TrustState::Trusted
        });
        let path = folder.as_ref().map(super::hooked::folder_of);
        let found = crate::hooks::sources::discover(inner.config.env.as_ref(), &inner.config.state, path.as_deref(), trusted);
        let file = crate::hooks::approvals::file(&inner.config.state);
        let allowed = found.hooks.iter().map(|hook| crate::hooks::approvals::allowed(&file, hook)).collect();
        (found, allowed)
    }

    /// Take back the reader's allowance of a hook (by its digest): it asks
    /// again before it next runs. `Ok(false)` when none had that digest.
    pub fn revoke_hook(&self, digest: &str) -> Result<bool, String> {
        crate::hooks::approvals::revoke(&crate::hooks::approvals::file(&self.inner.config.state), digest)
    }

    /// Build the service on the runtime it is given; it builds none.
    pub fn new(config: AgentConfig, handle: Handle) -> Self {
        let keys = KeyStore::new(config.env.clone(), &config.state);
        // The labs' agents (`crate::acp`): their permission questions go to the
        // core's own dialog, and each works in its conversation's folder. Both
        // need the service, made below; until then the slot is empty.
        let made: Arc<std::sync::OnceLock<std::sync::Weak<Inner>>> = Arc::default();
        let agents = {
            let (asking, folders) = (made.clone(), made.clone());
            Arc::new(crate::acp::pool::Pool::new(
                config.state.clone(),
                config.env.clone(),
                handle.clone(),
                crate::acp::find_node(config.env.as_ref()),
                Arc::new(move |agent: crate::acp::Agent, permission: crate::acp::session::Permission| {
                    let inner = asking.get().and_then(std::sync::Weak::upgrade);
                    Box::pin(async move {
                        let Some(inner) = inner else { return false };
                        let key = format!("agent-permission:{}", uuid::Uuid::new_v4().simple());
                        inner
                            .confirmer
                            .ask(
                                &key,
                                ConfirmRequest::AgentPermission {
                                    agent: agent.label().to_owned(),
                                    action: permission.title,
                                    diff: crate::acp::session::diff_lines(&permission.diffs),
                                    detail: permission.detail,
                                },
                                Initiated::Page,
                            )
                            .await
                    }) as futures::future::BoxFuture<'static, bool>
                }),
                Arc::new(move |thread: &str| {
                    let inner = folders.get().and_then(std::sync::Weak::upgrade)?;
                    let convo = inner.loaded(thread)?;
                    let root = convo.state().workspace.as_ref().map(|w| w.root.clone());
                    root
                }),
            ))
        };
        let chat_config = ChatConfig {
            state: config.state.clone(),
            env: config.env.clone(),
            development: config.development,
            clock: config.clock.clone(),
            store: config.store.clone(),
            store_dir: config.store_dir.clone(),
            local: config.local.clone(),
            paths: config.paths.clone(),
            model_factory: config.model_factory.clone(),
            agents: Some(agents.clone()),
            echo_gap: Duration::from_millis(chat::echo::PIECE_GAP_MS),
            follow_gap: config.gap.min,
        };
        let runner = config
            .runner
            .clone()
            .unwrap_or_else(|| Arc::new(GitRunner::new(config.env.clone(), &config.state)));
        let launcher = config.launcher.clone().unwrap_or_else(|| {
            Arc::new(SpawnLauncher {
                env: config.env.clone(),
                globals: config.state.globals.clone(),
            })
        });
        let chat = Self {
            inner: Arc::new(Inner {
                chat: ChatCore::new(chat_config, handle.clone()),
                sidecars: SidecarStore::new(config.state.native_chat_dir()),
                trust: TrustStore::new(&config.state, config.clock.clone()),
                permissions: Permissions::new(&config.state, config.clock.clone()),
                confirmer: Confirmer::new(config.confirm.clone()),
                questions: Questions::new(config.attention.clone()),
                runs: crate::store::RunStore::new(config.state.chat_runs_dir()),
                runner,
                launcher,
                slots: CommandSlots::default(),
                workspaces: Mutex::default(),
                convos: Mutex::default(),
                agent_turns: AtomicU32::new(0),
                no_tools: Mutex::default(),
                no_vision: Mutex::default(),
                tool_probes: Mutex::default(),
                binary_sha: Mutex::default(),
                changed: Changed::default(),
                claims: Mutex::default(),
                mcp: Arc::new(McpHub::new(
                    config.state.clone(),
                    config.env.clone(),
                    config.clock.clone(),
                    handle.clone(),
                )),
                browser: Arc::new(
                    crate::browser::AgentBrowser::new(
                        config.state.clone(),
                        config.env.clone(),
                        handle.clone(),
                    )
                    .with_config(config.browser.clone()),
                ),
                preview: Arc::new(
                    crate::browser::preview::PreviewBrowser::new(
                        config.state.clone(),
                        config.env.clone(),
                        handle.clone(),
                    )
                    .with_test_config(
                        config.browser.headless,
                        config.browser.program.clone(),
                        config
                            .browser
                            .profile
                            .as_ref()
                            .map(|profile| profile.with_extension("preview")),
                    ),
                ),
                projects: Arc::new(crate::projects::Projects::new(
                    &config.state,
                    config.clock.clone(),
                )),
                notes: Arc::new(crate::notes::Notes::new(&config.state, config.clock.clone())),
                tasks: Mutex::default(),
                plans: Mutex::default(),
                artifacts: Mutex::default(),
                background: Default::default(),
                agent_pool: agents,
                plugins: Arc::new(crate::plugins::Plugins::new(
                    &config.state,
                    config.clock.clone(),
                )),
                desktop: Arc::new({
                    let desktop = match &config.desktop {
                        Some(driver) => {
                            crate::desktop::Desktop::new(driver.clone(), handle.clone())
                        }
                        None => crate::desktop::Desktop::system(handle.clone()),
                    };
                    match config.stop_keys {
                        Some((modifiers, key)) => desktop.with_stop_keys(modifiers, key),
                        None => desktop,
                    }
                }),
                keys,
                handle,
                config,
            }),
        };
        let _ = made.set(Arc::downgrade(&chat.inner));
        // Auto mode left on: its stop hotkey is armed again.
        if crate::desktop::prefs::on(&chat.inner.config.state) {
            let _ = arm_stop_hotkey(&chat.inner);
        }
        chat
    }

    /// Attach a folder the shell's native picker or a drop gave (§6.1): no
    /// dialog, because the reader chose it in Rust-owned UI.
    pub fn attach_native(
        &self,
        id: Option<&str>,
        path: PathBuf,
    ) -> BoxFuture<'static, Result<WorkspaceView, Refusal>> {
        self.attach(
            id.map(str::to_owned),
            crate::workspace::attach::AttachSource::Native(path),
        )
    }

    fn attach(
        &self,
        id: Option<String>,
        source: crate::workspace::attach::AttachSource,
    ) -> BoxFuture<'static, Result<WorkspaceView, Refusal>> {
        let inner = self.inner.clone();
        async move {
            if let Some(id) = &id {
                if !is_conversation_id(id) {
                    return Err(refuse(RefusalKind::NotFound, refusals::GONE));
                }
                if inner
                    .loaded(id)
                    .is_some_and(|convo| convo.state().running.is_some())
                {
                    return Err(refuse(RefusalKind::Conflict, words::RUNNING));
                }
            }
            let path =
                crate::workspace::attach::confirmed_path(source, inner.config.confirm.as_ref())
                    .await
                    .map_err(|why| refuse(RefusalKind::Invalid, why.sentence()))?;
            let work = inner.clone();
            inner
                .handle
                .clone()
                .spawn_blocking(move || {
                    let workspace = crate::workspace::attach::attach_path(
                        &path,
                        work.config.env.as_ref(),
                        &work.config.state,
                        &work.runner,
                    )
                    .map_err(|why| refuse(RefusalKind::Invalid, why.sentence()))?;
                    lock(&work.workspaces).insert(workspace.id.clone(), workspace.clone());
                    if let Some(id) = &id {
                        let convo = work.load(id)?;
                        let origin = convo.state().origin;
                        work.bind(
                            &convo,
                            NewMeta {
                                workspace: None,
                                mode: Mode::Ask,
                                origin,
                            },
                        )?;
                        {
                            let mut state = convo.state();
                            // A new folder is a new destination (T1).
                            state.remote_ok = None;
                            state.lease = Some(Arc::new(work.lease_for(&workspace)));
                            state.workspace = Some(workspace.clone());
                        }
                        work.write_meta(&convo);
                        work.note_changed(id);
                    }
                    Ok(views::workspace_view(&work, &workspace))
                })
                .await
                .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))
        }
        .boxed()
    }

    /// The sidebar's change notices (FG8): ids only.
    pub fn chats_changed(&self) -> BoxStream<'static, Vec<String>> {
        self.inner
            .changed
            .stream(self.inner.config.gap.min, self.inner.handle.clone())
    }

    /// Run `work` on the runtime's blocking pool.
    fn blocking<T: Send + 'static>(
        &self,
        work: impl FnOnce(&Arc<Inner>) -> Result<T, Refusal> + Send + 'static,
    ) -> BoxFuture<'static, Result<T, Refusal>> {
        let inner = self.inner.clone();
        let handle = inner.handle.clone();
        async move {
            handle
                .spawn_blocking(move || work(&inner))
                .await
                .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))
        }
        .boxed()
    }

    /// Stop every running turn and wait for each to end (at most `wait`).
    pub async fn shutdown(&self, wait: Duration) {
        // Background commands end with Lattice.
        self.inner.background.stop_all();
        let ids: Vec<String> = lock(&self.inner.convos).keys().cloned().collect();
        for id in &ids {
            self.stop(id);
        }
        self.inner.chat.shutdown(wait).await;
        let inner = self.inner.clone();
        let all = async move {
            loop {
                let busy = lock(&inner.convos)
                    .values()
                    .any(|convo| convo.state().running.is_some());
                if !busy {
                    return;
                }
                let any = lock(&inner.convos).values().next().cloned();
                if let Some(convo) = any {
                    let mut receiver = convo.log.subscribe();
                    let _ = receiver.changed().await;
                }
            }
        };
        let _ = self
            .inner
            .handle
            .spawn(async move { tokio::time::timeout(wait, all).await })
            .await;
        // Every MCP server's Job closes: its whole tree ends.
        self.inner.mcp.stop_all();
        self.inner.browser.stop().await;
        self.inner.preview.stop().await;
        self.inner.desktop.disarm_hotkey();
    }

    // ------------------------------------------------------------ projects
    //
    // Projects ("Both"): named projects, each of which may be bound to a
    // folder. Each call reads or writes `projects.json` on the blocking pool.

    fn on_projects<T: Send + 'static>(
        &self,
        work: impl FnOnce(&crate::projects::Projects) -> Result<T, String> + Send + 'static,
    ) -> BoxFuture<'static, Result<T, Refusal>> {
        let inner = self.inner.clone();
        async move {
            let projects = inner.projects.clone();
            inner
                .handle
                .spawn_blocking(move || work(&projects))
                .await
                .unwrap_or_else(|_| Err(refusals::RUNTIME.to_owned()))
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))
        }
        .boxed()
    }

    /// Every project, archived ones included.
    pub fn projects(&self) -> BoxFuture<'static, Result<Vec<crate::projects::Project>, Refusal>> {
        self.on_projects(|projects| projects.list())
    }

    /// A new project.
    pub fn project_create(
        &self,
        name: String,
    ) -> BoxFuture<'static, Result<crate::projects::Project, Refusal>> {
        self.on_projects(move |projects| projects.create(&name))
    }

    /// Change a project.
    pub fn project_change(
        &self,
        id: String,
        change: crate::projects::Change,
    ) -> BoxFuture<'static, Result<crate::projects::Project, Refusal>> {
        self.on_projects(move |projects| projects.change(&id, change))
    }

    /// Put a chat in a project, or in none.
    pub fn project_assign(
        &self,
        conversation: String,
        project: Option<String>,
    ) -> BoxFuture<'static, Result<(), Refusal>> {
        self.on_projects(move |projects| projects.assign(&conversation, project.as_deref()))
    }

    // ------------------------------------------------------------- plugins
    //
    // "Plugins", their second part: folders in Claude Code's
    // plugin layout (`crate::plugins`), each call on the blocking pool.

    fn on_plugins<T: Send + 'static>(
        &self,
        work: impl FnOnce(&crate::plugins::Plugins) -> Result<T, String> + Send + 'static,
    ) -> BoxFuture<'static, Result<T, Refusal>> {
        let inner = self.inner.clone();
        async move {
            let plugins = inner.plugins.clone();
            inner
                .handle
                .spawn_blocking(move || work(&plugins))
                .await
                .unwrap_or_else(|_| Err(refusals::RUNTIME.to_owned()))
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))
        }
        .boxed()
    }

    /// Every plugin on the list, with what it brings.
    pub fn plugins(&self) -> BoxFuture<'static, Result<Vec<crate::plugins::PluginView>, Refusal>> {
        self.on_plugins(|plugins| plugins.list())
    }

    /// Add the plugin in `folder`, a folder the reader chose in Windows'
    /// picker (Rust-owned UI): so no dialog asks, as none asks for a folder
    /// attached the same way. It is on once added.
    pub fn plugin_add_native(
        &self,
        folder: PathBuf,
    ) -> BoxFuture<'static, Result<crate::plugins::PluginView, Refusal>> {
        self.on_plugins(move |plugins| plugins.add(&folder))
    }

    /// Switch a plugin on or off.
    pub fn plugin_set_on(&self, name: String, on: bool) -> BoxFuture<'static, Result<(), Refusal>> {
        self.on_plugins(move |plugins| plugins.set_on(&name, on))
    }

    /// Take a plugin off the list (its folder is left as it is).
    pub fn plugin_remove(&self, name: String) -> BoxFuture<'static, Result<(), Refusal>> {
        self.on_plugins(move |plugins| plugins.remove(&name))
    }

    /// A plugin's MCP servers, checked and with its folder filled in, for the
    /// window to copy into the reader's own settings (each then asks before
    /// it first starts, as any server does).
    pub fn plugin_mcp_entries(
        &self,
        name: String,
    ) -> BoxFuture<'static, Result<Vec<(String, serde_json::Value)>, Refusal>> {
        self.on_plugins(move |plugins| plugins.mcp_entries(&name))
    }

    /// Copy a plugin's MCP servers into the reader's own settings: a name the
    /// settings hold already is left as it is, and each copied server is
    /// enabled only if the reader says yes in the core's own
    /// `EnableMcpServer` dialog, as any server is.
    pub fn plugin_mcp_copy(
        &self,
        name: String,
        workspace: Option<String>,
    ) -> BoxFuture<'static, Result<crate::plugins::Copied, Refusal>> {
        let inner = self.inner.clone();
        let attached = self.mcp_folder(workspace.as_deref()).map(|(w, _)| w.root);
        async move {
            let plugins = inner.plugins.clone();
            let entries = inner
                .handle
                .spawn_blocking(move || plugins.mcp_entries(&name))
                .await
                .unwrap_or_else(|_| Err(refusals::RUNTIME.to_owned()))
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))?;
            let mut copied = crate::plugins::Copied::default();
            for (server, entry) in entries {
                let (state, hub, asked) = (
                    inner.config.state.clone(),
                    inner.mcp.clone(),
                    server.clone(),
                );
                let declared = inner
                    .handle
                    .spawn_blocking(move || {
                        if crate::mcp::config::user_entry_masked(&state, &asked).is_some() {
                            return Ok(None);
                        }
                        hub.put_user_server(&asked, &entry)?;
                        let key = crate::mcp::config::ServerKey::user(&asked);
                        Ok(hub
                            .declared(None)
                            .servers
                            .into_iter()
                            .find(|declared| declared.key == key))
                    })
                    .await
                    .unwrap_or_else(|_| Err(refusals::RUNTIME.to_owned()))
                    .map_err(|sentence: String| {
                        refuse(RefusalKind::Invalid, &format!("{server}: {sentence}"))
                    })?;
                let Some(declared) = declared else {
                    copied.kept.push(server);
                    continue;
                };
                match inner
                    .mcp
                    .enable(
                        &declared,
                        None,
                        attached.as_deref(),
                        &inner.confirmer,
                        Initiated::Native,
                    )
                    .await
                {
                    Ok(true) => copied.enabled.push(server),
                    Ok(false) => copied.not_enabled.push(server),
                    Err(sentence) => {
                        return Err(refuse(
                            RefusalKind::Invalid,
                            &format!("{server} was copied and not enabled: {sentence}"),
                        ));
                    }
                }
            }
            Ok(copied)
        }
        .boxed()
    }

    // ------------------------------------------------- commands and skills
    //
    // "Plugins": the reader's own and a trusted folder's prompt
    // templates (`crate::commands`) and skills (`crate::skills`), each read
    // on the blocking pool. An untrusted folder's files are not opened.

    /// The commands the reader can run now: their own, and the folder's
    /// (`workspace`, a workspace id) when it is trusted.
    pub fn commands(
        &self,
        workspace: Option<String>,
    ) -> BoxFuture<'static, crate::commands::Commands> {
        let inner = self.inner.clone();
        async move {
            let work = inner.clone();
            inner
                .handle
                .spawn_blocking(move || {
                    let state = &work.config.state;
                    match trusted_folder(&work, workspace) {
                        Some(folder) => folder
                            .with_rules(&work.runner, |rules| {
                                crate::commands::load(state, Some((rules, TrustState::Trusted)))
                            })
                            .unwrap_or_else(|_| crate::commands::load(state, None)),
                        None => crate::commands::load(state, None),
                    }
                })
                .await
                .unwrap_or_default()
        }
        .boxed()
    }

    /// The skills an agent turn would list now: the reader's own, and the
    /// folder's (`workspace`) when it is trusted.
    pub fn skills(&self, workspace: Option<String>) -> BoxFuture<'static, crate::skills::Skills> {
        let inner = self.inner.clone();
        async move {
            let work = inner.clone();
            inner
                .handle
                .spawn_blocking(move || {
                    let state = &work.config.state;
                    match trusted_folder(&work, workspace) {
                        Some(folder) => folder
                            .with_rules(&work.runner, |rules| {
                                crate::skills::load(state, Some((rules, TrustState::Trusted)))
                            })
                            .unwrap_or_else(|_| crate::skills::load(state, None)),
                        None => crate::skills::load(state, None),
                    }
                })
                .await
                .unwrap_or_default()
        }
        .boxed()
    }

    // --------------------------------------------------------------- tasks
    //
    // The tasks the agent suggests ("task chips", 2026-10-08;
    // `super::tasks`): nothing runs until the reader starts one.

    /// Start the task `task` the agent suggested in conversation `id`: a new
    /// conversation in the same folder and project, its first message the
    /// task's recorded prompt, with `choice` in `mode`. What a send would
    /// accept; the task is then settled as started, once.
    pub fn start_task(
        &self,
        id: &str,
        task: &str,
        choice: String,
        shown: Shown,
        mode: Mode,
    ) -> BoxFuture<'static, Result<Accepted, Refusal>> {
        super::tasks::start(
            self.inner.clone(),
            id.to_owned(),
            task.to_owned(),
            choice,
            shown,
            mode,
        )
        .boxed()
    }

    /// Approve the plan `plan` the agent proposed in conversation `id`: the
    /// conversation goes on in Agent mode, with `choice`, from a message that
    /// names the plan. What a send would accept; the plan is then settled as
    /// approved, once.
    pub fn approve_plan(
        &self,
        id: &str,
        plan: &str,
        choice: String,
        shown: Shown,
    ) -> BoxFuture<'static, Result<Accepted, Refusal>> {
        super::plans::approve(
            self.inner.clone(),
            id.to_owned(),
            plan.to_owned(),
            choice,
            shown,
        )
        .boxed()
    }

    /// Set aside the plan `plan` of conversation `id` for more planning.
    pub fn keep_planning(&self, id: &str, plan: &str) -> BoxFuture<'static, Result<(), Refusal>> {
        let (id, plan) = (id.to_owned(), plan.to_owned());
        self.blocking(move |inner| super::plans::keep_planning(inner, &id, &plan))
    }

    /// Set aside the task `task` the agent suggested in conversation `id`.
    pub fn dismiss_task(&self, id: &str, task: &str) -> BoxFuture<'static, Result<(), Refusal>> {
        let (id, task) = (id.to_owned(), task.to_owned());
        self.blocking(move |inner| super::tasks::dismiss(inner, &id, &task))
    }

    // ----------------------------------------------------------- artifacts
    //
    // Documents the agent saves beside the chat ("artifacts",
    // 2026-10-08; `super::artifacts`).

    /// One version of the artifact `name` of conversation `id` (the latest
    /// when `version` is `None`), its text read from the record.
    pub fn artifact(
        &self,
        id: &str,
        name: &str,
        version: Option<u32>,
    ) -> BoxFuture<'static, Result<ArtifactView, Refusal>> {
        let (id, name) = (id.to_owned(), name.to_owned());
        self.blocking(move |inner| {
            if !is_conversation_id(&id) {
                return Err(refuse(RefusalKind::NotFound, refusals::GONE));
            }
            super::artifacts::read(inner, &id, &name, version)
        })
    }

    /// Preview a page or a picture the agent saved (an `html` or `svg`
    /// artifact) in the preview browser, which has no network and none of
    /// the reader's sign-ins (`crate::browser::preview`), in a tab of its own.
    pub fn preview_artifact(
        &self,
        id: &str,
        name: &str,
        version: Option<u32>,
    ) -> BoxFuture<'static, Result<(), Refusal>> {
        let read = self.artifact(id, name, version);
        let preview = self.inner.preview.clone();
        async move {
            let view = read.await?;
            if !matches!(view.kind, ArtifactKind::Html | ArtifactKind::Svg) {
                return Err(refuse(
                    RefusalKind::Invalid,
                    "Only a page or a picture is previewed; this artifact is shown in the editor.",
                ));
            }
            let html = crate::browser::preview::page(view.kind, &view.title, &view.text);
            preview
                .show(&html)
                .await
                .map_err(|why| refuse(RefusalKind::Unavailable, &why))
        }
        .boxed()
    }

    // ------------------------------------------------------ the labs' agents
    //
    // Claude Code and Codex on the reader's own subscriptions (`crate::acp`):
    // offered as choices once installed.

    /// Each of the labs' agents: whether its adapter is installed, and whether
    /// Node (which runs Claude Code's) was found.
    pub fn agents(&self) -> Vec<(crate::acp::Agent, bool, bool)> {
        let dir = crate::acp::agents_dir(&self.inner.config.state);
        let node = crate::acp::find_node(self.inner.config.env.as_ref()).is_file();
        crate::acp::Agent::ALL.into_iter().map(|agent| (agent, agent.installed(&dir), node)).collect()
    }

    /// What `agent` last offered to answer with, and the reader's pick
    /// (`crate::acp::models`).
    pub fn agent_models(
        &self,
        agent: crate::acp::Agent,
    ) -> (crate::acp::models::Offered, Option<String>) {
        crate::acp::models::known(&crate::acp::agents_dir(&self.inner.config.state), agent)
    }

    /// Check models: start `agent` only to read the models it offers.
    pub fn agent_check_models(
        &self,
        agent: crate::acp::Agent,
    ) -> BoxFuture<'static, Result<crate::acp::models::Offered, Refusal>> {
        let pool = self.inner.agent_pool.clone();
        async move {
            pool.check_models(agent)
                .await
                .map_err(|why| refuse(RefusalKind::Unavailable, &why))
        }
        .boxed()
    }

    /// The reader's pick of `agent`'s model for its new chats; `None` for its default.
    pub fn agent_pick_model(
        &self,
        agent: crate::acp::Agent,
        model: Option<&str>,
    ) -> Result<(), String> {
        crate::acp::models::pick(&crate::acp::agents_dir(&self.inner.config.state), agent, model)
    }

    /// Install `agent`'s adapter with npm (the reader's action: a download that
    /// runs only when the person asks for it).
    pub fn agent_install(&self, agent: crate::acp::Agent) -> BoxFuture<'static, Result<(), Refusal>> {
        self.blocking(move |inner| {
            crate::acp::install(agent, &inner.config.state, inner.config.env.as_ref())
                .map_err(|why| refuse(RefusalKind::Unavailable, &why))
        })
    }

    // ----------------------------------------------------------- auto mode
    //
    // The whole desktop (chosen 2026-10-08): switched on only
    // in the core's own dialog; Ctrl+Alt+End stops every running agent turn.

    /// Is auto mode on?
    pub fn auto_mode_on(&self) -> bool {
        crate::desktop::prefs::on(&self.inner.config.state)
    }

    /// Whether the stop hotkey is armed (auto mode on and the keys free).
    pub fn auto_mode_armed(&self) -> bool {
        self.inner.desktop.hotkey_armed()
    }

    /// What the stop hotkey does when pressed (tests: no key is pressed).
    #[cfg(test)]
    pub(crate) fn press_stop_keys(&self) {
        stop_all(&self.inner);
    }

    /// Switch auto mode on (through the core's `AutoMode` dialog, and only
    /// with its stop hotkey armed) or off; `true` when it is on after.
    pub fn auto_mode_set(&self, on: bool) -> BoxFuture<'static, Result<bool, Refusal>> {
        let inner = self.inner.clone();
        async move {
            if on {
                let confirmed = inner
                    .confirmer
                    .ask(
                        "auto-mode",
                        ConfirmRequest::AutoMode {
                            stop_keys: crate::desktop::STOP_KEYS.to_owned(),
                        },
                        Initiated::Native,
                    )
                    .await;
                if !confirmed {
                    return Ok(false);
                }
                arm_stop_hotkey(&inner).map_err(|sentence| {
                    refuse(
                        RefusalKind::Unavailable,
                        &format!("{sentence} Auto mode stays off."),
                    )
                })?;
            } else {
                inner.desktop.disarm_hotkey();
            }
            let state = inner.config.state.clone();
            inner
                .handle
                .spawn_blocking(move || crate::desktop::prefs::set(&state, on))
                .await
                .unwrap_or_else(|_| Err(refusals::RUNTIME.to_owned()))
                .map_err(|sentence| {
                    inner.desktop.disarm_hotkey();
                    refuse(RefusalKind::Invalid, &sentence)
                })?;
            Ok(on)
        }
        .boxed()
    }

    // ------------------------------------------------------------- browser
    //
    // The agent's own browser: the reader switches it on (then Agent-mode
    // turns with a model that can see are offered its tools), shows it (to
    // sign in to a site themselves), and stops it.

    /// The agent's browser itself.
    pub fn browser(&self) -> Arc<crate::browser::AgentBrowser> {
        self.inner.browser.clone()
    }

    /// Is the agent's browser switched on?
    pub fn browser_on(&self) -> bool {
        crate::browser::prefs::on(&self.inner.config.state)
    }

    /// Switch the agent's browser on or off; off stops it.
    pub fn browser_set_on(&self, on: bool) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        async move {
            let state = inner.config.state.clone();
            inner
                .handle
                .spawn_blocking(move || crate::browser::prefs::set(&state, on))
                .await
                .unwrap_or_else(|_| Err(refusals::RUNTIME.to_owned()))
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))?;
            if !on {
                inner.browser.stop().await;
            }
            Ok(())
        }
        .boxed()
    }

    /// Show the browser (started now if it is not), at `url` when given: the
    /// reader's own action, to sign in to a site or to watch.
    pub fn browser_show(&self, url: Option<String>) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        async move {
            inner
                .browser
                .show(url.as_deref())
                .await
                .map(|_| ())
                .map_err(|sentence| refuse(RefusalKind::Unavailable, &sentence))
        }
        .boxed()
    }

    /// Open `url` in a tab of its own in the browser (started now if it is
    /// not): the reader's own action, a link of Alelyon's account panel. It
    /// does not switch the agent's browser on, and the agent keeps its page.
    pub fn browser_open_tab(&self, url: String) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        async move {
            inner
                .browser
                .open_tab(&url)
                .await
                .map_err(|sentence| refuse(RefusalKind::Unavailable, &sentence))
        }
        .boxed()
    }

    /// Stop the browser (its Job ends every process it started).
    pub fn browser_stop(&self) -> BoxFuture<'static, ()> {
        let inner = self.inner.clone();
        async move { inner.browser.stop().await }.boxed()
    }

    // ----------------------------------------------------------------- MCP
    //
    // The Tools view's calls (spec §12). Each decision that widens authority
    // asks the reader in the core's own dialog: enabling a server
    // (`EnableMcpServer`), allowing a tool always (`AllowMcpTool`), each call
    // (`McpCall`, through `decide`). The view's buttons are Rust-owned
    // controls, so they ask as a native action does (CP3: a refusal can be
    // asked again by the reader's own click).

    /// The MCP servers of this process.
    pub fn mcp(&self) -> Arc<McpHub> {
        self.inner.mcp.clone()
    }

    /// Bumped at every change to a server; read the overview again then.
    pub fn mcp_changed(&self) -> tokio::sync::watch::Receiver<u64> {
        self.inner.mcp.changed()
    }

    /// The attached folder `workspace` names, and whether it is trusted.
    fn mcp_folder(&self, workspace: Option<&str>) -> Option<(Workspace, bool)> {
        let found = lock(&self.inner.workspaces).get(workspace?).cloned()?;
        let trusted = self.inner.trust.state(&found) == TrustState::Trusted;
        Some((found, trusted))
    }

    /// The Tools view's servers: the reader's own, and the folder's when it
    /// is trusted (FT3). Off the caller's thread.
    pub fn mcp_overview(
        &self,
        workspace: Option<String>,
    ) -> BoxFuture<'static, crate::mcp::hub::Overview> {
        let inner = self.inner.clone();
        let folder = self.mcp_folder(workspace.as_deref());
        async move {
            inner
                .handle
                .clone()
                .spawn_blocking(move || {
                    let declared = match &folder {
                        Some((workspace, true)) => {
                            inner.mcp.declared(Some((workspace, &inner.runner)))
                        }
                        _ => inner.mcp.declared(None),
                    };
                    inner.mcp.overview(&declared)
                })
                .await
                .unwrap_or_default()
        }
        .boxed()
    }

    /// `key`'s entry as declared now, and the root of the folder that
    /// declares it, for an action on it. Off the caller's thread.
    fn mcp_entry(
        &self,
        key: crate::mcp::config::ServerKey,
        workspace: Option<String>,
    ) -> BoxFuture<'static, Result<(crate::mcp::config::ServerEntry, Option<PathBuf>), Refusal>>
    {
        let inner = self.inner.clone();
        let folder = self.mcp_folder(workspace.as_deref());
        async move {
            inner
                .handle
                .clone()
                .spawn_blocking(move || {
                    let (declared, root) = match (&key.scope, &folder) {
                        (crate::mcp::config::Scope::Folder { .. }, Some((workspace, true))) => (
                            inner.mcp.declared(Some((workspace, &inner.runner))),
                            Some(workspace.root.clone()),
                        ),
                        (crate::mcp::config::Scope::Folder { .. }, _) => {
                            return Err(refuse(
                                RefusalKind::Conflict,
                                "Trust the folder that declares that server first.",
                            ));
                        }
                        (crate::mcp::config::Scope::User, _) => (inner.mcp.declared(None), None),
                    };
                    declared
                        .servers
                        .into_iter()
                        .find(|entry| entry.key == key)
                        .map(|entry| (entry, root))
                        .ok_or_else(|| {
                            refuse(
                                RefusalKind::NotFound,
                                "That MCP server is not declared now.",
                            )
                        })
                })
                .await
                .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))
        }
        .boxed()
    }

    /// Ask the reader to enable `key` (`EnableMcpServer`); `Ok(false)` when
    /// they said no.
    pub fn mcp_enable(
        &self,
        key: crate::mcp::config::ServerKey,
        workspace: Option<String>,
    ) -> BoxFuture<'static, Result<bool, Refusal>> {
        let inner = self.inner.clone();
        let attached = self.mcp_folder(workspace.as_deref()).map(|(w, _)| w.root);
        let entry = self.mcp_entry(key, workspace);
        async move {
            let (entry, root) = entry.await?;
            inner
                .mcp
                .enable(
                    &entry,
                    root.as_deref(),
                    attached.as_deref(),
                    &inner.confirmer,
                    Initiated::Native,
                )
                .await
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))
        }
        .boxed()
    }

    /// Start `key`'s server now, if it is enabled.
    pub fn mcp_start(
        &self,
        key: crate::mcp::config::ServerKey,
        workspace: Option<String>,
    ) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        let attached = self.mcp_folder(workspace.as_deref()).map(|(w, _)| w.root);
        let entry = self.mcp_entry(key, workspace);
        async move {
            let (entry, root) = entry.await?;
            inner
                .mcp
                .start(&entry, root.as_deref(), attached.as_deref())
                .await
                .map_err(|sentence| refuse(RefusalKind::Unavailable, &sentence))
        }
        .boxed()
    }

    /// Stop `key`'s server.
    pub fn mcp_stop(&self, key: crate::mcp::config::ServerKey) -> BoxFuture<'static, ()> {
        let inner = self.inner.clone();
        async move { inner.mcp.stop(&key).await }.boxed()
    }

    /// End `key`'s approval and stop it: it asks again before it next starts.
    pub fn mcp_disable(
        &self,
        key: crate::mcp::config::ServerKey,
    ) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        async move {
            inner
                .mcp
                .disable(&key)
                .await
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))
        }
        .boxed()
    }

    /// Switch a tool (or `"*"`, the whole server) on or off.
    pub fn mcp_switch(
        &self,
        key: crate::mcp::config::ServerKey,
        tool: String,
        on: bool,
    ) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        async move {
            inner
                .mcp
                .switch(&key, &tool, on)
                .await
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))
        }
        .boxed()
    }

    /// Ask the reader to allow one tool of `key` always (`AllowMcpTool`), from
    /// the Tools view; `Ok(false)` when they said no.
    pub fn mcp_allow(
        &self,
        key: crate::mcp::config::ServerKey,
        tool: String,
        workspace: Option<String>,
    ) -> BoxFuture<'static, Result<bool, Refusal>> {
        let inner = self.inner.clone();
        let entry = self.mcp_entry(key, workspace);
        async move {
            let (entry, _) = entry.await?;
            let ask = format!("allow-mcp:{}:{}:{tool}", entry.key.id(), entry.sha256);
            inner
                .mcp
                .allow_always(&entry, &tool, &inner.confirmer, &ask, Initiated::Native)
                .await
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))
        }
        .boxed()
    }

    /// The reader's entry for `name`, its values masked, for the Tools view
    /// to edit.
    pub fn mcp_masked(&self, name: String) -> BoxFuture<'static, Option<serde_json::Value>> {
        let inner = self.inner.clone();
        async move {
            let state = inner.config.state.clone();
            inner
                .handle
                .spawn_blocking(move || crate::mcp::config::user_entry_masked(&state, &name))
                .await
                .ok()
                .flatten()
        }
        .boxed()
    }

    /// End "allow always" for one tool: its calls ask again.
    pub fn mcp_disallow(
        &self,
        key: crate::mcp::config::ServerKey,
        tool: String,
    ) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        async move {
            let hub = inner.mcp.clone();
            inner
                .handle
                .spawn_blocking(move || hub.disallow(&key, &tool))
                .await
                .unwrap_or_else(|_| Err(refusals::RUNTIME.to_owned()))
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))
        }
        .boxed()
    }

    /// Add or replace a server in the reader's own file (it is enabled only
    /// by [`AgentChat::mcp_enable`]).
    pub fn mcp_put(
        &self,
        name: String,
        entry: serde_json::Value,
    ) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        async move {
            let hub = inner.mcp.clone();
            inner
                .handle
                .spawn_blocking(move || hub.put_user_server(&name, &entry))
                .await
                .unwrap_or_else(|_| Err(refusals::RUNTIME.to_owned()))
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))
        }
        .boxed()
    }

    /// Take a server out of the reader's own file, and stop it.
    pub fn mcp_remove(&self, name: String) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        async move {
            inner
                .mcp
                .remove_user_server(&name)
                .await
                .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))
        }
        .boxed()
    }

    /// "Allow always" for a pending MCP call: the native `AllowMcpTool`
    /// dialog; the reader's yes allows the tool and runs this call.
    pub fn allow_mcp_always(
        &self,
        id: &str,
        call: &str,
    ) -> BoxFuture<'static, Result<(), Refusal>> {
        turn::allow_mcp_always(self.inner.clone(), id.to_owned(), call.to_owned())
    }
}

/// Start what `ask` asks, after its Prepare (see the module header).
pub(crate) async fn start(inner: Arc<Inner>, ask: Ask) -> Result<Accepted, Refusal> {
    // A send while the conversation runs a turn is queued (§4.4).
    if let Ask::Send(request) = &ask
        && let Some(id) = &request.conversation
        && let Some(convo) = inner.loaded(id)
    {
        let mut state = convo.state();
        if state.running.is_some() {
            if !request.images.is_empty() {
                return Err(refuse(RefusalKind::Conflict, super::images::words::RUNNING));
            }
            let queued_id = format!("q_{}", &uuid::Uuid::new_v4().simple().to_string()[..12]);
            let message = QueuedMessage {
                queued_id: queued_id.clone(),
                text: request.text.clone(),
                shown: request.shown.clone(),
            };
            state.queue.push_back(message);
            let position = state.queue.len() as u32;
            drop(state);
            convo.record(&Item::Queued {
                queued_id: queued_id.clone(),
                text: request.text.clone(),
                shown: request.shown.clone(),
                at: inner.now(),
            });
            convo.log.push(ConversationEventKind::Queued {
                queued_id: queued_id.clone(),
                text: request.text.clone(),
                position,
            });
            return Ok(Accepted::Queued {
                conversation: id.clone(),
                queued_id,
                position,
            });
        }
    }
    let claimed = match ask.id() {
        Some(id) => Some(
            claim(&inner, id).ok_or_else(|| refuse(RefusalKind::Conflict, refusals::ANSWERING))?,
        ),
        None => None,
    };
    // (a)-(c) and T1's count, on the blocking pool; nothing is written.
    let work = inner.clone();
    let probe = ask.clone();
    let checked = inner
        .handle
        .spawn_blocking(move || check(&work, &probe))
        .await
        .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))?;
    // T1: the native dialog, before anything is written. Its count of earlier
    // local tool results is information (Amendment 4 RP2): they go in full
    // once it is confirmed; refused, nothing is written or sent.
    if let Some((label, earlier)) = &checked.confirm {
        let workspace_key = checked
            .workspace
            .as_ref()
            .map_or_else(String::new, |workspace| workspace.id.clone());
        // CP3: refused, the same request (label, folder, count) is not
        // asked again from the page this session.
        let key = format!(
            "first-remote:{}:{label}:{workspace_key}:{earlier}",
            ask.id().unwrap_or("new")
        );
        let confirmed = inner
            .confirmer
            .ask(
                &key,
                ConfirmRequest::FirstRemoteSend {
                    conversation: ask.id().unwrap_or_default().to_owned(),
                    label: label.clone(),
                    earlier_local_results: *earlier,
                },
                Initiated::Page,
            )
            .await;
        if !confirmed {
            return Err(refuse(RefusalKind::Invalid, words::NOT_CONFIRMED_REMOTE));
        }
    }
    // The secret checks read words, not pictures: images off this PC ask
    // every time. Refused, nothing is written or sent.
    if let Some((label, images)) = &checked.images_remote {
        let key = format!(
            "send-images:{}:{}",
            ask.id().unwrap_or("new"),
            uuid::Uuid::new_v4().simple()
        );
        let confirmed = inner
            .confirmer
            .ask(
                &key,
                ConfirmRequest::SendImages {
                    conversation: ask.id().unwrap_or_default().to_owned(),
                    label: label.clone(),
                    images: *images,
                },
                Initiated::Page,
            )
            .await;
        if !confirmed {
            return Err(refuse(
                RefusalKind::Invalid,
                super::images::words::NOT_CONFIRMED,
            ));
        }
    }
    let accepted = match checked.kind {
        TurnKind::Plain => plain(&inner, &ask, &checked).await?,
        TurnKind::Agent => {
            let work = inner.clone();
            let ask = ask.clone();
            inner
                .handle
                .spawn_blocking(move || turn::begin(&work, &ask, checked))
                .await
                .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))?
        }
    };
    drop(claimed);
    Ok(accepted)
}

/// Prepare (a)-(c): resolve, read, check; T1's count. Writes nothing.
/// Arm the stop hotkey: each press stops every running agent turn.
fn arm_stop_hotkey(inner: &Arc<Inner>) -> Result<(), String> {
    let weak = Arc::downgrade(inner);
    inner.desktop.arm_hotkey(Box::new(move || {
        if let Some(inner) = weak.upgrade() {
            stop_all(&inner);
        }
    }))
}

/// Stop every running agent turn (the stop hotkey).
fn stop_all(inner: &Inner) {
    let convos: Vec<Arc<Convo>> = lock(&inner.convos).values().cloned().collect();
    for convo in convos {
        stop_turn(inner, &convo);
    }
}

/// Stop a conversation's running turn; `false` when none runs.
fn stop_turn(inner: &Inner, convo: &Convo) -> bool {
    let (control, job, stops) = {
        let mut state = convo.state();
        let Some(running) = state.running.as_mut() else {
            return false;
        };
        running.stop_requested = true;
        let control = running.control.clone();
        let job = running.job.clone();
        let stops: Vec<StopHandle> = state.stops.values().cloned().collect();
        (control, job, stops)
    };
    for stop in stops {
        stop.stop();
    }
    if let Some(control) = control {
        control.cancel(lattice_agents::CancelMode::Immediate);
    }
    if let Some(job) = job {
        inner.chat.stop(&job);
    }
    true
}

/// The attached folder `workspace` names, when it is trusted.
fn trusted_folder(inner: &Inner, workspace: Option<String>) -> Option<Workspace> {
    let folder = lock(&inner.workspaces).get(&workspace?).cloned()?;
    (inner.trust.state(&folder) == TrustState::Trusted).then_some(folder)
}

/// Whether an agent turn without a folder has anything to use: the agent's
/// browser or the whole desktop, switched on, or an enabled MCP server of the
/// reader's own.
fn folderless_tools(inner: &Inner) -> bool {
    if crate::browser::prefs::on(&inner.config.state)
        || crate::desktop::prefs::on(&inner.config.state)
    {
        return true;
    }
    let declared = inner.mcp.declared(None);
    inner
        .mcp
        .overview(&declared)
        .servers
        .iter()
        .any(|server| server.enabled && !server.off)
}

fn check(inner: &Arc<Inner>, ask: &Ask) -> Result<Checked, Refusal> {
    let convo = match ask.id() {
        Some(id) => Some(inner.load(id)?),
        None => None,
    };
    if let Some(convo) = &convo
        && convo.state().running.is_some()
    {
        return Err(refuse(RefusalKind::Conflict, words::RUNNING));
    }
    let last_choice = convo
        .as_ref()
        .and_then(|convo| convo.state().last_choice.clone());
    let choice = ask.choice(last_choice.as_deref());
    validate_choice(&choice, inner.config.development)?;
    let resolution = inner.resolve(&choice, ask.shown())?;
    let current_mode = convo.as_ref().map_or(Mode::Ask, |convo| convo.state().mode);
    let mode = ask.mode(current_mode);
    // The folder: the request's, else the conversation's.
    let requested = match ask {
        Ask::Send(request) => request.workspace.clone(),
        _ => None,
    };
    let workspace = match requested {
        Some(id) => Some(
            lock(&inner.workspaces)
                .get(&id)
                .cloned()
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_WORKSPACE))?,
        ),
        None => convo
            .as_ref()
            .and_then(|convo| convo.state().workspace.clone()),
    };
    let caps = inner.caps(&resolution);
    let mut plain_notice = false;
    // The chat's project: a new chat's as the send names it, else its own.
    let project = match ask {
        Ask::Send(request) if request.conversation.is_none() => request
            .project
            .as_deref()
            .and_then(|id| inner.projects.find(id)),
        _ => ask.id().and_then(|id| inner.projects.of_chat(id)),
    };
    let project_says = project
        .as_ref()
        .is_some_and(crate::projects::Project::leads);
    let kind = match &workspace {
        // A lab agent works with its own tools, in its own session: always a
        // plain turn of the chat, whatever the capabilities say.
        _ if matches!(resolution.target, crate::chat::vocab::Target::Agent(_)) => TurnKind::Plain,
        // No folder: Agent mode is an agent turn of the agent's browser and
        // the reader's own MCP servers, when there is one of them to use; a
        // chat of a project with instructions or files is one in either mode,
        // so they always reach the model.
        None if caps.tools != Tri::No
            && ((mode == Mode::Agent && folderless_tools(inner)) || project_says) =>
        {
            TurnKind::Agent
        }
        None => TurnKind::Plain,
        Some(_) if caps.tools == Tri::No => {
            plain_notice = true;
            TurnKind::Plain
        }
        Some(workspace) => {
            if mode == Mode::Agent && inner.trust.state(workspace) != TrustState::Trusted {
                return Err(refuse(RefusalKind::Conflict, words::TRUST_FIRST));
            }
            TurnKind::Agent
        }
    };
    if matches!(ask, Ask::Continue { .. }) && kind != TurnKind::Agent {
        return Err(refuse(RefusalKind::Conflict, words::NOTHING_TO_CONTINUE));
    }
    let project_plain = project_says && kind == TurnKind::Plain;
    // Images go to the agent, to a model that may see them.
    let images = match ask {
        Ask::Send(request) => request.images.len() as u32,
        _ => 0,
    };
    if images > 0 && kind == TurnKind::Plain {
        return Err(refuse(RefusalKind::Conflict, super::images::words::PLAIN));
    }
    if images > 0 && caps.vision == Tri::No {
        return Err(refuse(
            RefusalKind::Conflict,
            super::images::words::CANNOT_SEE,
        ));
    }
    // (b) and (c): what a target off the machine would receive (N4). Tool
    // items pass the tripwire per item (T3); words are refused here.
    let (turns, items) = match &convo {
        Some(convo) => (
            inner.config.store.load(&convo.id),
            inner
                .sidecars
                .read_items(&convo.id)
                .map(|log| log.items)
                .unwrap_or_default(),
        ),
        None => (Vec::new(), Vec::new()),
    };
    let remote = !resolution.affirmatively_local();
    if remote {
        let text = match ask {
            Ask::Send(request) => request.text.clone(),
            _ => String::new(),
        };
        if looks_like_secret(&text) || turns.iter().any(|turn| looks_like_secret(&turn.text)) {
            return Err(refuse(
                RefusalKind::Invalid,
                &answer::secret_sentence(&resolution.shown.label),
            ));
        }
    }
    // T1: the first remote send, or the first after the label or the folder
    // changed, asks; the dialog states the earlier local results.
    let local_n = replay::local_results(&turns, &items, None);
    let confirm = if remote && !matches!(resolution.target, Target::Refused(_)) {
        let workspace_id = workspace.as_ref().map(|workspace| workspace.id.clone());
        let ok = convo
            .as_ref()
            .and_then(|convo| convo.state().remote_ok.clone());
        match ok {
            Some((label, id, confirmed))
                if label == resolution.shown.label
                    && id == workspace_id
                    && local_n <= confirmed =>
            {
                None
            }
            _ => Some((resolution.shown.label.clone(), local_n)),
        }
    } else {
        None
    };
    let images_remote = (remote && images > 0 && !matches!(resolution.target, Target::Refused(_)))
        .then(|| (resolution.shown.label.clone(), images));
    Ok(Checked {
        resolution,
        kind,
        workspace,
        confirm,
        local_n,
        plain_notice,
        project_plain,
        images_remote,
    })
}

/// Bind conversation `id` to `workspace` (its record, staging and
/// checkpoints, and the folder in its state) in `mode`, as an agent turn's
/// send does: the mode decides whether its changes may be kept. Blocking.
fn bind_folder(inner: &Inner, id: &str, workspace: &Workspace, mode: Mode) -> Result<(), Refusal> {
    let convo = inner.convo(id);
    inner.bind(
        &convo,
        super::sidecar::NewMeta {
            workspace: Some(super::sidecar::WorkspaceRef {
                id: workspace.id.clone(),
                path: workspace.root.to_string_lossy().into_owned(),
            }),
            mode,
            origin: lattice_protocol::conversation::Origin::Native,
        },
    )?;
    {
        let mut state = convo.state();
        if state.workspace.as_ref().is_none_or(|current| current.id != workspace.id) {
            state.lease = Some(Arc::new(inner.lease_for(workspace)));
            state.workspace = Some(workspace.clone());
        }
    }
    if convo.state().mode != mode {
        convo.state().mode = mode;
        convo.record(&Item::ModeSwitch { mode, at: inner.now() });
    }
    Ok(())
}

/// A plain turn (§2.4 step 3): the plain chat's pipeline, unchanged; its
/// events are forwarded into the conversation.
async fn plain(inner: &Arc<Inner>, ask: &Ask, checked: &Checked) -> Result<Accepted, Refusal> {
    // A lab agent's chat with a folder works in that folder: an existing
    // conversation is bound to it before the send; a new one's first session
    // is told the folder (`acp::pool`), and the conversation is bound to it as
    // soon as the send has made it.
    let agent_folder = match (&checked.resolution.target, &checked.workspace) {
        (crate::chat::vocab::Target::Agent(agent), Some(workspace)) => Some((*agent, workspace.clone())),
        _ => None,
    };
    if let Some((agent, workspace)) = &agent_folder {
        let existing = match ask {
            Ask::Send(request) => request.conversation.clone(),
            Ask::Regenerate(request) => Some(request.conversation.clone()),
            Ask::Continue { .. } => None,
        };
        match existing {
            Some(id) => {
                let (bind_inner, bind_workspace) = (inner.clone(), workspace.clone());
                let mode = ask.mode(inner.convo(&id).state().mode);
                inner
                    .handle
                    .spawn_blocking(move || bind_folder(&bind_inner, &id, &bind_workspace, mode))
                    .await
                    .map_err(|_| refuse(RefusalKind::Unavailable, words::NOT_OPEN))??;
            }
            None => inner
                .agent_pool
                .next_folder(*agent, Some(workspace.root.clone())),
        }
    }
    let accepted = match ask {
        Ask::Send(request) => {
            inner
                .chat
                .send(lattice_protocol::chat::SendRequest {
                    thread: request.conversation.clone(),
                    text: request.text.clone(),
                    choice: request.choice.clone(),
                    edit_of: request.edit_of.clone(),
                    shown: request.shown.clone(),
                })
                .await?
        }
        Ask::Regenerate(request) => {
            inner
                .chat
                .regenerate(lattice_protocol::chat::RegenerateRequest {
                    thread: request.conversation.clone(),
                    choice: request.choice.clone(),
                    shown: request.shown.clone(),
                })
                .await?
        }
        Ask::Continue { .. } => {
            return Err(refuse(RefusalKind::Conflict, words::NOTHING_TO_CONTINUE));
        }
    };
    let id = accepted.thread.id.clone();
    if let Some((agent, workspace)) = &agent_folder {
        let (bind_inner, bind_id, bind_workspace) = (inner.clone(), id.clone(), workspace.clone());
        let mode = ask.mode(inner.convo(&id).state().mode);
        let bound = inner
            .handle
            .spawn_blocking(move || bind_folder(&bind_inner, &bind_id, &bind_workspace, mode))
            .await;
        inner.agent_pool.next_folder(*agent, None);
        if !matches!(bound, Ok(Ok(()))) {
            inner.convo(&id).log.push(ConversationEventKind::Notice {
                text: "Lattice could not record this chat's folder, so what the agent changes is not listed.".to_owned(),
            });
        }
    }
    // A new chat joins the project its send names, as an agent turn's does.
    if let Ask::Send(request) = ask
        && request.conversation.is_none()
        && let Some(project) = request.project.clone()
    {
        let projects = inner.projects.clone();
        let chat = id.clone();
        let _ = inner
            .handle
            .spawn_blocking(move || projects.assign(&chat, Some(&project)))
            .await;
    }
    let convo = inner.convo(&id);
    let turn_id = accepted
        .question
        .as_ref()
        .map(|turn| turn.id.clone())
        .unwrap_or_default();
    {
        let mut state = convo.state();
        state.running = Some(Running {
            turn: turn_id.clone(),
            control: None,
            job: Some(accepted.job.clone()),
            affirmatively_local: checked.resolution.affirmatively_local(),
            label: checked.resolution.shown.label.clone(),
            notes: Arc::default(),
            stop_requested: false,
        });
        if !checked.resolution.affirmatively_local() {
            state.remote_ok = Some((
                checked.resolution.shown.label.clone(),
                state
                    .workspace
                    .as_ref()
                    .map(|workspace| workspace.id.clone()),
                checked.local_n,
            ));
        }
        state.last_choice = Some(match ask {
            Ask::Send(request) => request.choice.clone(),
            Ask::Regenerate(request) => request.choice.clone(),
            Ask::Continue { .. } => String::new(),
        });
    }
    convo.log.push(ConversationEventKind::TurnStarted {
        turn: turn_id.clone(),
        kind: TurnKind::Plain,
        mode: convo.state().mode,
        label: checked.resolution.shown.label.clone(),
        locality: checked.resolution.shown.locality,
    });
    if checked.plain_notice {
        convo.log.push(ConversationEventKind::Notice {
            text: super::caps::PLAIN_FALLBACK.to_owned(),
        });
    }
    if checked.project_plain {
        convo.log.push(ConversationEventKind::Notice {
            text: words::PROJECT_PLAIN.to_owned(),
        });
    }
    // A lab agent's turn in a folder is watched as a command is: what it
    // changed is listed, and undoable (`super::lab_turns`).
    let watch = match &checked.resolution.target {
        crate::chat::vocab::Target::Agent(agent) => {
            let (watch_inner, watch_convo) = (inner.clone(), convo.clone());
            let (turn, label) = (turn_id.clone(), agent.label().to_owned());
            match inner
                .handle
                .spawn_blocking(move || super::lab_turns::begin(&watch_inner, &watch_convo, &turn, &label))
                .await
            {
                Ok(Some(Ok(watch))) => Some(watch),
                Ok(Some(Err(text))) => {
                    convo.log.push(ConversationEventKind::Notice { text });
                    None
                }
                Ok(None) | Err(_) => None,
            }
        }
        _ => None,
    };
    let stream = inner.chat.follow(&accepted.job, 0)?;
    let pump_inner = inner.clone();
    let pump_convo = convo.clone();
    inner.handle.spawn(pump_plain(
        pump_inner,
        pump_convo,
        stream,
        turn_id.clone(),
        watch,
    ));
    inner.note_changed(&id);
    Ok(Accepted::Started {
        conversation: Box::new(
            inner
                .row(&id)
                .map(|row| inner.summary(&row))
                .unwrap_or_else(|_| views::bare_summary(&id)),
        ),
        user_turn: Box::new(accepted.question.unwrap_or_else(views::no_turn)),
        turn_kind: TurnKind::Plain,
    })
}

/// Forward a plain turn's events into the conversation.
async fn pump_plain(
    inner: Arc<Inner>,
    convo: Arc<Convo>,
    mut stream: BoxStream<'static, Vec<lattice_protocol::chat::ChatEvent>>,
    turn: String,
    watch: Option<super::lab_turns::Watch>,
) {
    let mut status = TurnStatus::Completed;
    while let Some(batch) = stream.next().await {
        for event in batch {
            match event.kind {
                ChatEventKind::Stage { stage, detail } => {
                    convo
                        .log
                        .push(ConversationEventKind::Stage { stage, detail });
                }
                ChatEventKind::Delta { text } => {
                    convo.log.push(ConversationEventKind::Delta { text });
                }
                ChatEventKind::Turn { turn, saved } => {
                    if turn.cancelled {
                        status = TurnStatus::Stopped;
                    }
                    convo
                        .log
                        .push(ConversationEventKind::TurnSaved { turn, saved });
                }
                ChatEventKind::Error {
                    message,
                    turn,
                    saved,
                } => {
                    status = if turn.cancelled {
                        TurnStatus::Stopped
                    } else {
                        TurnStatus::Failed
                    };
                    convo.log.push(ConversationEventKind::Error { message });
                    convo
                        .log
                        .push(ConversationEventKind::TurnSaved { turn, saved });
                }
                ChatEventKind::Done => {}
            }
        }
    }
    // What a lab agent's turn changed, before the turn is said to end.
    if let Some(watch) = watch {
        let (end_inner, end_convo) = (inner.clone(), convo.clone());
        let _ = inner
            .handle
            .spawn_blocking(move || super::lab_turns::end(&end_inner, &end_convo, watch))
            .await;
    }
    convo.log.finish_turn();
    convo.state().running = None;
    convo.state().last = Some((turn.clone(), status));
    convo
        .log
        .push(ConversationEventKind::TurnEnded { turn, status });
    inner.note_changed(&convo.id);
    next_queued(inner, convo).await;
}

/// After a turn: send the first queued message, through Prepare (§4.4).
pub(crate) fn next_queued(inner: Arc<Inner>, convo: Arc<Convo>) -> BoxFuture<'static, ()> {
    async move {
        let next = {
            let mut state = convo.state();
            if state.running.is_some() {
                return;
            }
            state.queue.pop_front()
        };
        let Some(message) = next else {
            return;
        };
        let (mode, workspace) = {
            let state = convo.state();
            (
                state.mode,
                state
                    .workspace
                    .as_ref()
                    .map(|workspace| workspace.id.clone()),
            )
        };
        let choice = convo.state().last_choice.clone().unwrap_or_default();
        let request = SendRequest {
            conversation: Some(convo.id.clone()),
            text: message.text.clone(),
            choice,
            shown: message.shown.clone(),
            mode,
            workspace,
            edit_of: None,
            project: None,
            images: Vec::new(),
        };
        match start(inner.clone(), Ask::Send(request)).await {
            Ok(Accepted::Started { user_turn, .. }) => {
                let item = Item::Dequeued {
                    queued_id: message.queued_id.clone(),
                    outcome: DequeueOutcome::Sent {
                        turn: user_turn.id.clone(),
                    },
                    at: inner.now(),
                };
                convo.record(&item);
                convo.log.push(ConversationEventKind::Dequeued {
                    queued_id: message.queued_id,
                    outcome: DequeueOutcome::Sent { turn: user_turn.id },
                });
            }
            Ok(Accepted::Queued { .. }) => {}
            Err(refusal) => {
                // N10 (or anything else): it stays queued, with a notice.
                convo.state().queue.push_front(message);
                let text = if refusal.message == refusals::MOVED {
                    words::QUEUED_MOVED.to_owned()
                } else {
                    refusal.message
                };
                convo.log.push(ConversationEventKind::Notice { text });
            }
        }
    }
    .boxed()
}

impl AgentChatService for AgentChat {
    fn status(&self) -> CoreStatus {
        CoreStatus {
            runtime: "lattice-core".to_owned(),
            agent_turns: self.inner.agent_turns(),
            shared_writes: crate::chat::store_gate::SHARED_WRITES,
            refusal: None,
        }
    }

    fn list(&self) -> BoxFuture<'static, Result<ChatList, Refusal>> {
        let archived = self.inner.chat.archived(0);
        let rows = self.blocking(|inner| {
            Ok(match inner.config.store.list() {
                IndexState::Absent => (Vec::new(), false),
                IndexState::Rows(rows) => {
                    (rows.iter().map(|row| inner.summary(row)).collect(), false)
                }
                IndexState::Unreadable => (Vec::new(), true),
            })
        });
        let inner = self.inner.clone();
        async move {
            let (conversations, index_unreadable) = rows.await?;
            let page = archived.await?;
            Ok(ChatList {
                conversations,
                archived: page.archived,
                store_dir: inner.config.store_dir.clone(),
                installed: inner.config.state.installed,
                index_unreadable,
                archive_unreadable: page.archive_unreadable,
                shared_writes: crate::chat::store_gate::SHARED_WRITES,
                limit: 60,
            })
        }
        .boxed()
    }

    fn open(&self, id: &str) -> BoxFuture<'static, Result<Snapshot, Refusal>> {
        let id = id.to_owned();
        self.blocking(move |inner| {
            if !is_conversation_id(&id) {
                return Err(refuse(RefusalKind::NotFound, refusals::GONE));
            }
            let convo = inner.load(&id)?;
            let row = inner.row(&id)?;
            let turns = inner
                .config
                .store
                .load(&id)
                .iter()
                .map(StoredTurn::to_chat_turn)
                .collect();
            let (events, last_seq) = convo.log.recorded();
            let running = convo.state().running.is_some();
            let answering_elsewhere = !running
                && matches!(
                    answer_lock(&inner.config.state.chat_locks_dir(), &id),
                    Locked::Elsewhere
                );
            Ok(Snapshot {
                conversation: inner.summary(&row),
                turns,
                events,
                last_seq,
                queued: convo.state().queue.iter().cloned().collect(),
                answering_elsewhere,
            })
        })
    }

    fn choices(&self) -> BoxFuture<'static, Vec<ChatChoice>> {
        self.inner.chat.choices()
    }

    fn local_runtime(&self) -> BoxFuture<'static, LocalRuntime> {
        self.inner.chat.local_runtime()
    }

    fn send(&self, r: SendRequest) -> BoxFuture<'static, Result<Accepted, Refusal>> {
        if let Err(refusal) = validate_send(&r, self.inner.config.development) {
            return async move { Err(refusal) }.boxed();
        }
        start(self.inner.clone(), Ask::Send(r)).boxed()
    }

    fn regenerate(&self, r: RegenerateRequest) -> BoxFuture<'static, Result<Accepted, Refusal>> {
        if !is_conversation_id(&r.conversation) {
            return async { Err(refuse(RefusalKind::NotFound, refusals::GONE)) }.boxed();
        }
        if let Err(refusal) = validate_choice(&r.choice, self.inner.config.development) {
            return async move { Err(refusal) }.boxed();
        }
        start(self.inner.clone(), Ask::Regenerate(r)).boxed()
    }

    fn continue_turn(
        &self,
        id: &str,
        shown: Shown,
    ) -> BoxFuture<'static, Result<Accepted, Refusal>> {
        if !is_conversation_id(id) {
            return async { Err(refuse(RefusalKind::NotFound, refusals::GONE)) }.boxed();
        }
        start(
            self.inner.clone(),
            Ask::Continue {
                id: id.to_owned(),
                shown,
            },
        )
        .boxed()
    }

    fn steer(&self, id: &str, text: String) -> BoxFuture<'static, Result<(), Refusal>> {
        let inner = self.inner.clone();
        let id = id.to_owned();
        async move {
            let convo = inner
                .loaded(&id)
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_OPEN))?;
            if text.trim().is_empty() {
                return Err(refuse(RefusalKind::Invalid, refusals::EMPTY_MESSAGE));
            }
            let (control, local, label) = {
                let state = convo.state();
                match &state.running {
                    Some(running) => (
                        running.control.clone(),
                        running.affirmatively_local,
                        running.label.clone(),
                    ),
                    None => return Err(refuse(RefusalKind::Conflict, words::NOTHING_TO_CONTINUE)),
                }
            };
            // N4 first, when the target is not affirmatively local.
            if !local && looks_like_secret(&text) {
                return Err(refuse(
                    RefusalKind::Invalid,
                    &answer::secret_sentence(&label),
                ));
            }
            let given_back = match control {
                Some(control) => control
                    .steer(text)
                    .err()
                    .map(|lattice_agents::Steer::Closed(text)| text),
                None => Some(text),
            };
            if let Some(text) = given_back {
                turn::queue_message(&inner, &convo, text, None);
            }
            Ok(())
        }
        .boxed()
    }

    fn stop(&self, id: &str) -> bool {
        let Some(convo) = self.inner.loaded(id) else {
            return false;
        };
        stop_turn(&self.inner, &convo)
    }

    fn stop_command(&self, id: &str, call: &str) -> bool {
        self.inner.background.stop(id, call)
    }

    fn cancel_queued(&self, id: &str, queued_id: &str) -> Result<(), Refusal> {
        let convo = self
            .inner
            .loaded(id)
            .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_OPEN))?;
        let removed = {
            let mut state = convo.state();
            let before = state.queue.len();
            state.queue.retain(|message| message.queued_id != queued_id);
            before != state.queue.len()
        };
        if !removed {
            return Err(refuse(RefusalKind::NotFound, "That message is not queued."));
        }
        convo.record(&Item::Dequeued {
            queued_id: queued_id.to_owned(),
            outcome: DequeueOutcome::Cancelled,
            at: self.inner.now(),
        });
        convo.log.push(ConversationEventKind::Dequeued {
            queued_id: queued_id.to_owned(),
            outcome: DequeueOutcome::Cancelled,
        });
        Ok(())
    }

    fn decide(&self, id: &str, call: &str, d: Decision) -> BoxFuture<'static, Result<(), Refusal>> {
        turn::decide(self.inner.clone(), id.to_owned(), call.to_owned(), d)
    }

    fn allow_always(
        &self,
        id: &str,
        call: &str,
        scope: MatchScope,
    ) -> BoxFuture<'static, Result<AllowEntry, Refusal>> {
        let MatchScope::Exact = scope;
        turn::allow_always(self.inner.clone(), id.to_owned(), call.to_owned())
    }

    fn answer(&self, id: &str, call: &str, text: String) -> Result<(), Refusal> {
        let convo = self
            .inner
            .loaded(id)
            .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_OPEN))?;
        let turn = convo
            .state()
            .running
            .as_ref()
            .map(|running| running.turn.clone())
            .unwrap_or_default();
        self.inner
            .questions
            .answer(id, call, text.clone())
            .map_err(|error| refuse(RefusalKind::NotFound, error.sentence()))?;
        convo.record(&Item::Answer {
            turn,
            call_id: call.to_owned(),
            text,
            at: self.inner.now(),
        });
        convo.log.push(ConversationEventKind::QuestionAnswered {
            call_id: call.to_owned(),
        });
        // S25: after each question is resolved.
        let (store, id) = (self.inner.config.store.clone(), id.to_owned());
        self.inner
            .handle
            .spawn_blocking(move || touch(store.as_ref(), &id));
        Ok(())
    }

    fn follow(
        &self,
        id: &str,
        after: u64,
    ) -> Result<BoxStream<'static, Vec<ConversationEvent>>, Refusal> {
        let convo = is_conversation_id(id)
            .then(|| self.inner.loaded(id))
            .flatten()
            .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_OPEN))?;
        Ok(follow::follow(
            convo.log.clone(),
            after,
            self.inner.config.gap,
            self.inner.handle.clone(),
        ))
    }

    fn set_mode(&self, id: &str, mode: Mode) -> BoxFuture<'static, Result<(), Refusal>> {
        let id = id.to_owned();
        self.blocking(move |inner| {
            let convo = inner.load(&id)?;
            if convo.state().running.is_some() {
                return Err(refuse(RefusalKind::Conflict, words::RUNNING));
            }
            if convo.state().mode == mode {
                return Ok(());
            }
            convo.state().mode = mode;
            convo.record(&Item::ModeSwitch {
                mode,
                at: inner.now(),
            });
            inner.write_meta(&convo);
            inner.note_changed(&id);
            Ok(())
        })
    }

    fn attach_workspace(
        &self,
        id: Option<&str>,
        path: String,
    ) -> BoxFuture<'static, Result<WorkspaceView, Refusal>> {
        // A path string from the page attaches only after the native dialog.
        self.attach(
            id.map(str::to_owned),
            crate::workspace::attach::AttachSource::Page(path),
        )
    }

    fn trust(&self, workspace: &str) -> BoxFuture<'static, Result<TrustState, Refusal>> {
        let inner = self.inner.clone();
        let id = workspace.to_owned();
        async move {
            let workspace = lock(&inner.workspaces)
                .get(&id)
                .cloned()
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_WORKSPACE))?;
            let work = inner.clone();
            let probe = workspace.clone();
            let will_read = inner
                .handle
                .spawn_blocking(move || {
                    probe
                        .with_rules(&work.runner, crate::workspace::rules::found)
                        .unwrap_or_default()
                })
                .await
                .unwrap_or_default();
            let git = matches!(workspace.repo, crate::git::dotgit::Repo::Git(_));
            inner
                .trust
                .trust(
                    &workspace,
                    &inner.confirmer,
                    Initiated::Page,
                    will_read,
                    git,
                )
                .await
                .map_err(|error| refuse(RefusalKind::Invalid, error.sentence()))
        }
        .boxed()
    }

    fn files(
        &self,
        workspace: &str,
        query: String,
        limit: u32,
    ) -> BoxFuture<'static, Result<Vec<RankedPath>, Refusal>> {
        let id = workspace.to_owned();
        self.blocking(move |inner| {
            let workspace = lock(&inner.workspaces)
                .get(&id)
                .cloned()
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_WORKSPACE))?;
            let ctx = crate::tools::read::ReadContext {
                workspace: &workspace,
                runner: &inner.runner,
                overlay: &crate::tools::read::NoOverlay,
            };
            crate::tools::read::files(&ctx, &query, limit)
                .map_err(|error| refuse(RefusalKind::Invalid, &error.0))
        })
    }

    fn read_lines(
        &self,
        view: ViewRef,
        from: u32,
        count: u32,
    ) -> BoxFuture<'static, Result<Lines, Refusal>> {
        self.blocking(move |inner| views::read_lines(inner, &view, from, count))
    }

    fn changes(&self, id: &str) -> BoxFuture<'static, Result<ChangeSet, Refusal>> {
        let id = id.to_owned();
        self.blocking(move |inner| {
            let convo = inner.load(&id)?;
            Ok(views::change_set(inner, &convo))
        })
    }

    fn diff(&self, id: &str, change: &str) -> BoxFuture<'static, Result<FileDiff, Refusal>> {
        let (id, change) = (id.to_owned(), change.to_owned());
        self.blocking(move |inner| {
            let convo = inner.load(&id)?;
            let staging = convo
                .state()
                .staging
                .clone()
                .ok_or_else(|| refuse(RefusalKind::NotFound, "That change does not exist."))?;
            crate::staging::review::file_diff(&staging, &change)
                .ok_or_else(|| refuse(RefusalKind::NotFound, "That change does not exist."))
        })
    }

    fn review(
        &self,
        id: &str,
        ops: Vec<ReviewOp>,
    ) -> BoxFuture<'static, Result<ReviewOutcome, Refusal>> {
        turn::review(self.inner.clone(), id.to_owned(), ops)
    }

    fn checkpoints(&self, id: &str) -> BoxFuture<'static, Result<Vec<CheckpointView>, Refusal>> {
        let id = id.to_owned();
        self.blocking(move |inner| {
            let convo = inner.load(&id)?;
            Ok(convo
                .state()
                .checkpoints
                .as_ref()
                .map(|checkpoints| checkpoints.views())
                .unwrap_or_default())
        })
    }

    fn restore(
        &self,
        id: &str,
        to: CheckpointId,
    ) -> BoxFuture<'static, Result<ChangeSet, Refusal>> {
        let id = id.to_owned();
        self.blocking(move |inner| {
            let convo = inner.load(&id)?;
            let (workspace, staging, checkpoints) = {
                let state = convo.state();
                (
                    state.workspace.clone(),
                    state.staging.clone(),
                    state.checkpoints.clone(),
                )
            };
            let (Some(workspace), Some(staging), Some(checkpoints)) =
                (workspace, staging, checkpoints)
            else {
                return Err(refuse(
                    RefusalKind::NotFound,
                    "There is no checkpoint to restore.",
                ));
            };
            crate::changes::restore::restore(
                &crate::changes::restore::RestoreContext {
                    workspace: &workspace,
                    runner: &inner.runner,
                    staging: &staging,
                    checkpoints: &checkpoints,
                },
                to,
            )
            .map_err(|error| refuse(RefusalKind::Invalid, &error.sentence()))?;
            Ok(views::change_set(inner, &convo))
        })
    }

    fn rename(
        &self,
        id: &str,
        title: &str,
    ) -> BoxFuture<'static, Result<ConversationSummary, Refusal>> {
        let (id, title) = (id.to_owned(), title.to_owned());
        self.blocking(move |inner| {
            if !is_conversation_id(&id) {
                return Err(refuse(RefusalKind::NotFound, refusals::GONE));
            }
            if normalize_title(&title).is_empty() {
                return Err(refuse(RefusalKind::Invalid, refusals::EMPTY_TITLE));
            }
            match inner.config.store.rename(&id, &title) {
                Ok(true) => {
                    inner.note_changed(&id);
                    inner.row(&id).map(|row| inner.summary(&row))
                }
                Ok(false) => Err(refuse(RefusalKind::NotFound, refusals::GONE)),
                Err(error) => Err(error.refusal()),
            }
        })
    }

    fn pin(&self, id: &str, choice: &str) -> BoxFuture<'static, Result<(), Refusal>> {
        let (id, choice) = (id.to_owned(), choice.to_owned());
        self.blocking(move |inner| {
            if !is_conversation_id(&id) {
                return Err(refuse(RefusalKind::NotFound, refusals::GONE));
            }
            if !vocab::is_valid_choice(&choice, inner.config.development) {
                return Err(refuse(RefusalKind::Invalid, vocab::words::UNKNOWN_CHOICE));
            }
            // T2: locality is fixed per turn.
            if inner
                .loaded(&id)
                .is_some_and(|convo| convo.state().running.is_some())
            {
                return Err(refuse(RefusalKind::Conflict, words::RUNNING));
            }
            match inner.config.store.pin(&id, &choice) {
                Ok(true) => {
                    if let Some(convo) = inner.loaded(&id) {
                        convo.state().last_choice = Some(choice);
                    }
                    Ok(())
                }
                Ok(false) => Err(refuse(RefusalKind::NotFound, refusals::GONE)),
                Err(error) => Err(error.refusal()),
            }
        })
    }

    fn archive(&self, id: &str) -> BoxFuture<'static, Result<(), Refusal>> {
        let id = id.to_owned();
        self.blocking(move |inner| {
            if !is_conversation_id(&id) {
                return Err(refuse(RefusalKind::NotFound, refusals::GONE));
            }
            if inner
                .loaded(&id)
                .is_some_and(|convo| convo.state().running.is_some())
            {
                return Err(refuse(RefusalKind::Conflict, words::RUNNING));
            }
            archived_by_reader::archive_by_reader(
                inner.config.store.as_ref(),
                &inner.config.state.native_chat_dir(),
                &id,
                inner.now(),
            )
            .map_err(StoreError::refusal)?;
            // The sidecar stays where it is; it binds again on unarchive.
            lock(&inner.convos).remove(&id);
            // Its background commands end with it.
            inner.background.stop_conversation(&id);
            inner.note_changed(&id);
            Ok(())
        })
    }

    fn unarchive(
        &self,
        key: ArchiveKey,
    ) -> BoxFuture<'static, Result<ConversationSummary, Refusal>> {
        self.blocking(move |inner| {
            if !is_conversation_id(&key.id) {
                return Err(refuse(RefusalKind::NotFound, refusals::GONE));
            }
            let row = inner
                .config
                .store
                .unarchive(&key)
                .map_err(StoreError::refusal)?;
            inner.note_changed(&row.id);
            Ok(inner.summary(&row))
        })
    }

    fn permissions(&self, workspace: &str) -> BoxFuture<'static, Result<Vec<AllowEntry>, Refusal>> {
        let id = workspace.to_owned();
        self.blocking(move |inner| {
            let workspace = lock(&inner.workspaces)
                .get(&id)
                .cloned()
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_WORKSPACE))?;
            inner
                .permissions
                .entries(&workspace)
                .map_err(|error| refuse(RefusalKind::Unavailable, &error.sentence()))
        })
    }

    fn revoke(&self, workspace: &str, entry: &str) -> BoxFuture<'static, Result<(), Refusal>> {
        let (id, entry) = (workspace.to_owned(), entry.to_owned());
        self.blocking(move |inner| {
            let workspace = lock(&inner.workspaces)
                .get(&id)
                .cloned()
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_WORKSPACE))?;
            inner
                .permissions
                .revoke(&workspace, &entry)
                .map_err(|error| refuse(RefusalKind::NotFound, &error.sentence()))
        })
    }
}

/// `DecidedBy` for a standing entry (re-exported for the turn).
pub(crate) fn standing(entry: &str) -> DecidedBy {
    DecidedBy::Standing {
        entry: entry.to_owned(),
    }
}
