//! One agent turn (the chat core's spec §2.4 steps 2(d) to 5, §3, §4.4,
//! §4.5, §5.7, §7, §9.3, §10.2; row E11). Not a port.
//!
//! - **Begin** (on the blocking pool, after Prepare's checks and T1's
//!   dialog): the user turn is written (or, for Continue, nothing is), the
//!   sidecar bound by `created`, `TurnStart` recorded, `touch` called (S25),
//!   and the replay built: [`replay::replay`] stops before this turn's user
//!   record (UT1), every target gets the same items (local tool items go to
//!   a remote target in full, Amendment 4 RP2; T3 screens them), and the
//!   folder's rules lead as one user item
//!   (§6.4). The run's input is `[User(text)]`; Continue's is the replay up
//!   to its last tool result, with no session.
//! - **The model**: built once for the turn (T2) through `models`, which
//!   sets `.local(..)` (N6); wrapped in the tripwire unless the target is
//!   affirmatively local (T3); and in [`TurnModel`], which adds what the
//!   agent must hear at its next model call (a conflict, a review, a note)
//!   and counts the calls and their usage (D10).
//! - **The tools** by mode (§7.1), each a handler over the module that owns
//!   it; `run_command` needs approval, decided by [`TurnApprovals`]: a
//!   standing match runs with no wait, anything else registers a pending
//!   call, records `ApprovalRequested`, sends the urgent event and asks for
//!   attention, and waits with no timer for [`decide`].
//! - **The stream is the only writer of tool items** ([`TurnSink`], the
//!   recorder's sink): `ToolCall` when the call is announced (before any
//!   approval wait), `ToolResult` on its output (redacted, T15; a command's
//!   whole output named by its blob), `Steered` as it is taken, and
//!   `Reasoning` (§22.8 RP3) for each `<think>` span of a model message and
//!   each provider-separate reasoning item, redacted (T15) and never shown.
//!   The same
//!   sink writes the turn's trace into `<native>/chat/runs/` in the run
//!   store's format, every event redacted.
//! - **End**: the final text, think-stripped (a stopped turn's unfinished
//!   message has its reasoning recorded first) and redacted (T15, with a
//!   `Notice` when it changed), is appended to the transcript store after a
//!   `touch`; `TurnEnd` records the status; steers the run did not send are
//!   queued, in order (A4a); the next queued message goes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use lattice_agents::model::{
    GenerationTrace, InputItem, Model, ModelError, ModelEvent, ModelRequest, ModelSettings,
};
use lattice_agents::{
    Agent, ApprovalDecision, ApprovalPort, ApprovalRequest, FunctionTool, NeedsApproval, RunConfig,
    RunError, RunItem, RunResult, StreamEvent, ToolContext, ToolError as AgentToolError,
    TraceProcessor, run_streamed_items,
};
use lattice_protocol::conversation::{
    Accepted, AllowEntry, ApprovalKind, ConversationEventKind, DecidedBy, Decision, Mode, ReviewOp,
    ReviewOutcome, ReviewResult, TurnKind, TurnStatus, is_call_id,
};
use lattice_protocol::{
    EndStatus, Locality, Refusal, RefusalKind, RunEventKind, RunStatus, RunSummary,
};
use serde_json::Value;
use tokio::sync::oneshot;

use super::agent::{Ask, Checked, Convo, Inner, Ready, Running, lock, next_queued, refuse, words};
use super::item::Item;
use super::prompt_agent;
use super::replay::{self, Options, SidecarSession};
use super::sidecar::NewMeta;
use super::tripwire::TripwireModel;
use super::views;
use crate::browser::policy::{self as browser_policy, Effect};
use crate::browser::{BrowserStatus, Shot};
use crate::changes::FolderGuards;
use crate::changes::checkpoint::Checkpoints;
use crate::changes::effects::CommandGuards;
use crate::chat::store::StoredTurn;
use crate::chat::think::{ThinkFilter, strip_think, think_text};
use crate::chat::transcript::NewTurn;
use crate::chat::vocab::Target;
use crate::chat::{Locked, answer, answer_lock, refusals};
use crate::desktop::Look;
use crate::desktop::policy as desktop_policy;
use crate::exec::run::{
    CommandContext, Progress, ProgressSink, RunCommandArgs, StopHandle, approve, prepare,
    run_with_progress,
};
use crate::llama::Lease;
use crate::mcp::config::ServerEntry;
use crate::mcp::hub::{TurnTool as McpTurnTool, TurnTools as McpTurnTools};
use crate::models;
use crate::policy::{Because, Verdict};
use crate::ports::{ConfirmRequest, Initiated};
use crate::recorder::{Intake, Recorder, RunSink, map_stream_event, redacted_event};
use crate::secrets;
use crate::staging::Staging;
use crate::tools::read::{NoOverlay, Overlay, ReadContext, ToolError, parse_args};
use crate::workspace::Workspace;
use crate::workspace::lease::WriterLease;

/// The temperature of every chat request (Python's).
const TEMPERATURE: f64 = 0.2;
/// The most bytes of a tool output an event previews (CR5).
const PREVIEW: usize = 2 * 1024;
/// §4.4: what a turn stopped before any text saves.
const NOTHING_WRITTEN: &str = answer::NOTHING_WRITTEN;

fn preview(text: &str) -> String {
    let clean = secrets::redact(text);
    let mut cut = PREVIEW.min(clean.len());
    while !clean.is_char_boundary(cut) {
        cut -= 1;
    }
    clean[..cut].to_owned()
}

/// What a turn's tools, approvals and sink work with.
pub(crate) struct TurnTools {
    pub inner: Arc<Inner>,
    pub convo: Arc<Convo>,
    pub turn: String,
    pub mode: Mode,
    /// The folder is trusted; a turn without a folder counts as trusted (no
    /// folder supplies anything to it).
    pub trusted: bool,
    /// The conversation's folder; `None` in an agent turn without one, which
    /// offers only `ask_question`, the agent's browser and the reader's own
    /// MCP servers.
    pub workspace: Option<Workspace>,
    pub staging: Arc<Staging>,
    pub checkpoints: Arc<Checkpoints>,
    /// The folder's writer lease (`None` without a folder).
    pub lease: Option<Arc<WriterLease>>,
    /// The remote label the output goes to, when the target is remote.
    pub remote: Option<String>,
    /// The model's call ids, as the core records them (§4.1).
    ids: Mutex<HashMap<String, String>>,
    /// A command's output blob, until its `ToolResult` is written.
    outputs: Mutex<HashMap<String, String>>,
    /// The MCP tools this turn offers, by the name the model sees (§12).
    mcp: Mutex<HashMap<String, McpTurnTool>>,
    /// The pictures of the browser's page or the screen, waiting for the
    /// next model call.
    shots: Arc<Mutex<Vec<Picture>>>,
    /// The skills this turn lists, as read when it began.
    pub skills: crate::skills::Skills,
    /// The turn's model, behind its tripwire, for its helpers (`super::helpers`).
    helper_model: std::sync::OnceLock<Arc<dyn Model>>,
    /// Helpers running now.
    helpers: std::sync::atomic::AtomicUsize,
}

/// A picture for the next model call: the browser's page or the screen.
pub(crate) struct Picture {
    /// What the model is told it is.
    caption: String,
    /// What it is called once it is no longer shown.
    earlier: String,
    image: lattice_agents::model::Image,
}

impl Picture {
    fn of_page(shot: &Shot) -> Self {
        Self {
            caption: format!(
                "The agent's browser now: {} ({}).",
                if shot.title.is_empty() {
                    "untitled"
                } else {
                    &shot.title
                },
                shot.url
            ),
            earlier: format!(
                "[An earlier screenshot of the agent's browser ({}) is not shown again.]",
                shot.url
            ),
            image: shot.image.clone(),
        }
    }

    fn of_screen(look: &Look) -> Self {
        Self {
            caption: format!(
                "Your screen now: {} in front.",
                if look.front.is_empty() {
                    "no window"
                } else {
                    &look.front
                }
            ),
            earlier: "[An earlier picture of your screen is not shown again.]".to_owned(),
            image: look.image.clone(),
        }
    }
}

/// A desktop action waiting for the reader in auto mode: it moves money or
/// changes an account. What its card and its `DesktopAct` dialog show.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DesktopCall {
    pub app: String,
    pub action: String,
    pub what: String,
    /// The effect, in words.
    pub effect: String,
}

/// A browser action waiting for the reader: its effect asks (share, buy,
/// delete, account). What its card and its `BrowserAct` dialog show.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BrowserCall {
    pub site: String,
    pub action: String,
    pub what: String,
    /// The effect, in words.
    pub effect: String,
}

/// An MCP tool's call waiting for the reader (§12): what its card and its
/// `McpCall` dialog show, and where an approved call goes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct McpCall {
    pub key: crate::mcp::config::ServerKey,
    pub sha256: String,
    /// The file that declares the server (the "allow always" dialog says).
    pub file: String,
    pub tool: String,
    /// At most 2 KiB of the arguments, redacted.
    pub arguments_preview: String,
}

impl TurnTools {
    /// The turn's model for a helper, once the turn has built it.
    pub(crate) fn helper_model(&self) -> Option<Arc<dyn Model>> {
        self.helper_model.get().cloned()
    }

    /// The count of helpers running now.
    pub(crate) fn helpers(&self) -> &std::sync::atomic::AtomicUsize {
        &self.helpers
    }

    /// The core's id for the model's call id: the same when it is 1–128
    /// printable ASCII characters, else `lc_<16 hex>`.
    pub(crate) fn core_id(&self, model: &str) -> String {
        let mut ids = lock(&self.ids);
        if let Some(id) = ids.get(model) {
            return id.clone();
        }
        let id = if is_call_id(model) {
            model.to_owned()
        } else {
            let sha = crate::sha::sha256_hex(format!("{}\0{model}", self.turn).as_bytes());
            format!("lc_{}", &sha[..16])
        };
        ids.insert(model.to_owned(), id.clone());
        id
    }

    /// The folder, for a tool that reads or stages in it.
    pub(crate) fn folder(&self) -> Result<&Workspace, ToolError> {
        self.workspace
            .as_ref()
            .ok_or_else(|| ToolError(words::NO_FOLDER_TOOL.to_owned()))
    }

    pub(crate) fn command_context<'a>(&'a self, hooks: &'a CommandGuards<'a>) -> CommandContext<'a> {
        CommandContext {
            workspace: hooks.workspace,
            runner: &self.inner.runner,
            staging: &self.staging,
            mode: self.mode,
            trusted: self.trusted,
            slots: &self.inner.slots,
            permissions: &self.inner.permissions,
            confirmer: &self.inner.confirmer,
            launcher: self.inner.launcher.as_ref(),
            hooks,
            env: self.inner.config.env.as_ref(),
            globals: &self.inner.config.state.globals,
            remote: self.remote.clone(),
        }
    }

    /// What a command needs around it, when the turn has a folder (a turn
    /// without one runs no command).
    pub(crate) fn guards(&self) -> Option<CommandGuards<'_>> {
        Some(CommandGuards {
            checkpoints: &self.checkpoints,
            workspace: self.workspace.as_ref()?,
            runner: &self.inner.runner,
            staging: &self.staging,
            lease: self.lease.as_deref()?,
        })
    }
}

// ----------------------------------------------------------------- begin

/// Write the turn's start and run it (see the module header). On the
/// blocking pool.
pub(crate) fn begin(inner: &Arc<Inner>, ask: &Ask, checked: Checked) -> Result<Accepted, Refusal> {
    if inner.agent_turns() >= super::agent::MAX_AGENT_TURNS {
        return Err(refuse(RefusalKind::Unavailable, words::TOO_MANY));
    }
    let store = &inner.config.store;
    let resolution = checked.resolution.clone();
    let local_n = checked.local_n;
    // No folder: an agent turn of the browser and the reader's own MCP
    // servers only (`check` allows it in Agent mode).
    let workspace = checked.workspace.clone();
    // (d) Write: the user turn (not for Continue), with the answer lock.
    let mut held = None;
    let mut take_lock = |id: &str| -> Result<(), Refusal> {
        match answer_lock(&inner.config.state.chat_locks_dir(), id) {
            Locked::Held(lock) => {
                held = Some(lock);
                Ok(())
            }
            Locked::Elsewhere => Err(refuse(RefusalKind::Conflict, refusals::ELSEWHERE)),
            Locked::Untried => Ok(()),
        }
    };
    let (id, user_turn, text, supersede_items, new) = match ask {
        Ask::Send(request) => match &request.conversation {
            None => {
                let (row, turn) = store
                    .first_message(&request.text, &request.choice)
                    .map_err(|error| error.refusal())?;
                take_lock(&row.id)?;
                // A new chat joins the project its send names.
                if let Some(project) = &request.project {
                    let _ = inner.projects.assign(&row.id, Some(project));
                }
                (row.id, turn, request.text.clone(), false, true)
            }
            Some(id) => {
                take_lock(id)?;
                if let Some(edit_of) = &request.edit_of {
                    let visible = store.load(id);
                    let ok = visible
                        .iter()
                        .any(|turn| &turn.id == edit_of && turn.role == "user");
                    if !ok {
                        return Err(refuse(RefusalKind::Conflict, refusals::NOT_EDITABLE));
                    }
                    store
                        .supersede(id, edit_of)
                        .map_err(|error| error.refusal())?;
                }
                let turn = store
                    .append(id, NewTurn::user(request.text.clone()))
                    .map_err(|error| error.refusal())?;
                store
                    .pin(id, &request.choice)
                    .map_err(|error| error.refusal())?;
                (id.clone(), turn, request.text.clone(), false, false)
            }
        },
        Ask::Regenerate(request) => {
            let id = &request.conversation;
            take_lock(id)?;
            let visible = store.load(id);
            let at = visible
                .iter()
                .rposition(|turn| turn.role == "user")
                .ok_or_else(|| refuse(RefusalKind::Conflict, refusals::NO_QUESTION))?;
            if let Some(first) = visible.get(at + 1) {
                store
                    .supersede(id, &first.id)
                    .map_err(|error| error.refusal())?;
            }
            store
                .pin(id, &request.choice)
                .map_err(|error| error.refusal())?;
            (
                id.clone(),
                visible[at].clone(),
                visible[at].text.clone(),
                true,
                false,
            )
        }
        Ask::Continue { id, .. } => {
            take_lock(id)?;
            let convo = inner.load(id)?;
            let (turn, status) = convo
                .state()
                .last
                .clone()
                .ok_or_else(|| refuse(RefusalKind::Conflict, words::NOTHING_TO_CONTINUE))?;
            if !matches!(status, TurnStatus::MaxTurns | TurnStatus::Interrupted) {
                return Err(refuse(RefusalKind::Conflict, words::NOTHING_TO_CONTINUE));
            }
            let user = store
                .load(id)
                .into_iter()
                .find(|stored| stored.id == turn)
                .ok_or_else(|| refuse(RefusalKind::Conflict, words::NOTHING_TO_CONTINUE))?;
            (id.clone(), user, String::new(), false, false)
        }
    };
    let convo = inner.convo(&id);
    let mode = ask.mode(convo.state().mode);
    let origin = if new {
        lattice_protocol::conversation::Origin::Native
    } else {
        convo.state().origin
    };
    inner.bind(
        &convo,
        NewMeta {
            workspace: workspace
                .as_ref()
                .map(|workspace| super::sidecar::WorkspaceRef {
                    id: workspace.id.clone(),
                    path: workspace.root.to_string_lossy().into_owned(),
                }),
            mode,
            origin,
        },
    )?;
    let now = inner.now();
    let choice = ask.choice(convo.state().last_choice.as_deref());
    {
        let mut state = convo.state();
        if new {
            state.origin = lattice_protocol::conversation::Origin::Native;
        }
        if let Some(workspace) = &workspace {
            let changed = state
                .workspace
                .as_ref()
                .is_none_or(|current| current.id != workspace.id);
            if changed {
                state.lease = Some(Arc::new(inner.lease_for(workspace)));
                state.workspace = Some(workspace.clone());
            }
        }
    }
    if convo.state().mode != mode {
        convo.state().mode = mode;
        convo.record(&Item::ModeSwitch { mode, at: now });
    }
    let previous = convo.state().last_choice.clone();
    if previous
        .as_deref()
        .is_some_and(|previous| previous != choice)
    {
        convo.record(&Item::ModelSwitch {
            choice: choice.clone(),
            label: resolution.shown.label.clone(),
            locality: resolution.shown.locality,
            at: now,
        });
    }
    inner.write_meta(&convo);
    if supersede_items {
        // A native regenerate: the turn's earlier tool trail is hidden from
        // the replay (§5.4); nothing is removed.
        convo.record(&Item::Superseded {
            turns: vec![user_turn.id.clone()],
            at: now,
        });
    }
    let local = resolution.affirmatively_local();
    let resolved = if local {
        Locality::Local
    } else {
        Locality::Remote
    };
    convo.record(&Item::TurnStart {
        turn: user_turn.id.clone(),
        mode,
        kind: TurnKind::Agent,
        choice: choice.clone(),
        shown: resolution.shown.clone(),
        resolved,
        label: resolution.shown.label.clone(),
        at: now,
    });
    // S25: at TurnStart.
    super::agent::touch(store.as_ref(), &id);
    // The images attached to this send, with its user turn (`super::images`).
    let sent_images = match ask {
        Ask::Send(request) if !request.images.is_empty() => {
            let decoded = super::images::check(&request.images)?;
            super::images::store(&convo, &user_turn.id, &decoded, now)?
        }
        _ => Vec::new(),
    };
    // The replay (§5.7.2).
    let (staging, checkpoints, lease, notes) = {
        let mut state = convo.state();
        (
            state.staging.clone(),
            state.checkpoints.clone(),
            state.lease.clone(),
            std::mem::take(&mut state.notes),
        )
    };
    let (Some(staging), Some(checkpoints)) = (staging, checkpoints) else {
        return Err(refuse(
            RefusalKind::Unavailable,
            "Lattice could not open this conversation's record here.",
        ));
    };
    let lease = if workspace.is_some() {
        let Some(lease) = lease else {
            return Err(refuse(
                RefusalKind::Unavailable,
                "Lattice could not open this conversation's record here.",
            ));
        };
        Some(lease)
    } else {
        None
    };
    let turns: Vec<StoredTurn> = store.load(&id);
    let items = inner
        .sidecars
        .read_items(&id)
        .map(|log| log.items)
        .unwrap_or_default();
    let caps = inner.caps(&resolution);
    let continuing = matches!(ask, Ask::Continue { .. });
    let sidecar = convo.state().sidecar.clone();
    let read_blob = |sha: &str| {
        sidecar
            .as_ref()
            .and_then(|sidecar| sidecar.read_blob(sha).ok())
    };
    let replayed = replay::replay(
        &turns,
        &items,
        &Options {
            stop_before: (!continuing).then(|| user_turn.id.clone()),
            context_tokens: caps.context_tokens,
        },
        &read_blob,
    );
    let trusted = workspace.as_ref().is_none_or(|workspace| {
        inner.trust.state(workspace) == lattice_protocol::conversation::TrustState::Trusted
    });
    let rules = match &workspace {
        // No folder: the reader's own rules.
        None => crate::workspace::rules::load(&inner.config.state, None, &[]).for_turn(&[], &[]),
        Some(workspace) => trusted
            .then(|| {
                workspace
                    .with_rules(&inner.runner, |rules| {
                        crate::workspace::rules::load(
                            &inner.config.state,
                            Some((rules, lattice_protocol::conversation::TrustState::Trusted)),
                            &inner.trust.rules_off(workspace),
                        )
                        .for_turn(&[], &[])
                    })
                    .ok()
                    .flatten()
            })
            .flatten(),
    };
    // The skills: the reader's own, and a trusted folder's.
    let skills = match &workspace {
        Some(workspace) if trusted => workspace
            .with_rules(&inner.runner, |rules| {
                crate::skills::load(
                    &inner.config.state,
                    Some((rules, lattice_protocol::conversation::TrustState::Trusted)),
                )
            })
            .unwrap_or_else(|_| crate::skills::load(&inner.config.state, None)),
        _ => crate::skills::load(&inner.config.state, None),
    };
    // The chat's project leads, before the folder's rules.
    let project = inner
        .projects
        .of_chat(&id)
        .and_then(|project| inner.projects.lead_text(&project));
    let rules = match (project, rules) {
        (Some(project), Some(rules)) => Some(format!("{project}\n\n{rules}")),
        (project, rules) => project.or(rules),
    };
    // The skills' list follows them.
    let rules = match (rules, skills.lead_text()) {
        (Some(rules), Some(list)) => Some(format!("{rules}\n\n{list}")),
        (rules, list) => rules.or(list),
    };
    // The agent's own notes about the folder come last (`super::memory`).
    let notes_kept = workspace
        .as_ref()
        .map(|workspace| super::memory::list(&super::memory::file(&inner.config.state, &workspace.id)))
        .unwrap_or_default();
    let rules = match (rules, super::memory::lead_text(&notes_kept)) {
        (Some(rules), Some(kept)) => Some(format!("{rules}\n\n{kept}")),
        (rules, kept) => rules.or(kept),
    };
    let lead = replay::lead_item(rules.as_deref(), replayed.left_out, &notes);
    let mut session_items: Vec<InputItem> = lead.into_iter().map(InputItem::User).collect();
    let (session, input) = if continuing {
        session_items.extend(replay::through_last_result(replayed.items));
        if !session_items
            .iter()
            .any(|item| matches!(item, InputItem::ToolResult { .. }))
        {
            return Err(refuse(RefusalKind::Conflict, words::NOTHING_TO_CONTINUE));
        }
        (None, session_items)
    } else {
        session_items.extend(replayed.items);
        // A regenerate sends the message's images again.
        let images = if supersede_items {
            super::images::of_turn(&items, &user_turn.id, &read_blob)
        } else {
            sent_images
        };
        let message = if images.is_empty() {
            InputItem::User(text.clone())
        } else {
            InputItem::UserImages {
                text: text.clone(),
                images,
            }
        };
        // The files the message mentions (`@path`), read as `read_file`
        // reads, go just before it (`super::mentions`).
        let mut input = Vec::new();
        if let Some(workspace) = &workspace {
            let overlay: &dyn Overlay = staging.as_ref();
            let ctx = ReadContext {
                workspace,
                runner: &inner.runner,
                overlay,
            };
            let read = super::mentions::read(&ctx, &text);
            if let Some(item) = read.item {
                input.push(InputItem::User(item));
            }
            if let Some(notice) = read.notice {
                convo.record(&Item::Notice {
                    text: notice.clone(),
                    at: now,
                });
                convo
                    .log
                    .push(ConversationEventKind::Notice { text: notice });
            }
        }
        input.push(message);
        (Some(Arc::new(SidecarSession::new(session_items))), input)
    };
    let notes = Arc::new(Mutex::new(Vec::new()));
    {
        let mut state = convo.state();
        if !local {
            state.remote_ok = Some((
                resolution.shown.label.clone(),
                workspace.as_ref().map(|workspace| workspace.id.clone()),
                local_n,
            ));
        }
        state.last_choice = Some(choice.clone());
        state.save_failed = false;
        state.running = Some(Running {
            turn: user_turn.id.clone(),
            control: None,
            job: None,
            affirmatively_local: local,
            label: resolution.shown.label.clone(),
            notes: notes.clone(),
            stop_requested: false,
        });
    }
    inner.agent_turns.fetch_add(1, Ordering::SeqCst);
    convo.log.push(ConversationEventKind::TurnStarted {
        turn: user_turn.id.clone(),
        kind: TurnKind::Agent,
        mode,
        label: resolution.shown.label.clone(),
        locality: resolved,
    });
    // §12: MCP tools are Agent mode's, in a trusted folder; the folder's own
    // MCP files are read only then (FT3). Read here, on the blocking pool.
    // LR8′: the managed server's first Agent-mode turn probes its tool calls.
    let probe = match &resolution.target {
        crate::chat::vocab::Target::Managed(model)
            if mode == Mode::Agent && caps.tools == super::caps::Tri::Unknown =>
        {
            inner.tool_probe_key(model)
        }
        _ => None,
    };
    let browser = mode == Mode::Agent
        && trusted
        && caps.vision != super::caps::Tri::No
        && crate::browser::prefs::on(&inner.config.state);
    // Auto mode (the whole-desktop route): switched on in the core's
    // own dialog only.
    let desktop = mode == Mode::Agent
        && trusted
        && caps.vision != super::caps::Tri::No
        && crate::desktop::prefs::on(&inner.config.state);
    let mcp_servers = if mode == Mode::Agent && trusted {
        inner
            .mcp
            .declared(
                workspace
                    .as_ref()
                    .map(|workspace| (workspace, inner.runner.as_ref())),
            )
            .servers
    } else {
        Vec::new()
    };
    let tools = Arc::new(TurnTools {
        inner: inner.clone(),
        convo: convo.clone(),
        turn: user_turn.id.clone(),
        mode,
        trusted,
        workspace,
        staging,
        checkpoints,
        lease,
        remote: (!local).then(|| resolution.shown.label.clone()),
        ids: Mutex::default(),
        outputs: Mutex::default(),
        mcp: Mutex::default(),
        shots: Arc::default(),
        skills,
        helper_model: std::sync::OnceLock::new(),
        helpers: std::sync::atomic::AtomicUsize::new(0),
    });
    let plan = Plan {
        tools,
        resolution,
        session,
        input,
        notes,
        task: text,
        held,
        mcp_servers,
        browser,
        desktop,
        probe,
    };
    inner.handle.spawn(run(plan));
    inner.note_changed(&id);
    let row = crate::convo::binding::listed_row(store.as_ref(), &id).ok();
    Ok(Accepted::Started {
        conversation: Box::new(
            row.map(|row| inner.summary(&row))
                .unwrap_or_else(|| views::bare_summary(&id)),
        ),
        user_turn: Box::new(user_turn.to_chat_turn()),
        turn_kind: TurnKind::Agent,
    })
}

struct Plan {
    tools: Arc<TurnTools>,
    resolution: crate::chat::vocab::Resolution,
    session: Option<Arc<SidecarSession>>,
    input: Vec<InputItem>,
    notes: Arc<Mutex<Vec<String>>>,
    task: String,
    held: Option<crate::chat::AnswerLock>,
    /// The MCP servers declared for this turn (Agent mode in a trusted
    /// folder only); their tools are listed before the first model call.
    mcp_servers: Vec<ServerEntry>,
    /// The agent's browser is offered.
    browser: bool,
    /// The whole desktop is offered (auto mode).
    desktop: bool,
    /// The managed server's tool-call probe runs first, under this key (LR8′).
    probe: Option<String>,
}

// ------------------------------------------------------------ the model

/// Counts of one turn's model calls (D10).
#[derive(Default)]
struct Calls {
    calls: AtomicU32,
    reported: AtomicU32,
    input_tokens: Mutex<u64>,
    output_tokens: Mutex<u64>,
    first_status: Mutex<Option<u16>>,
    /// The status a call that carried a screenshot failed with.
    image_status: Mutex<Option<u16>>,
}

/// The turn's model: what the agent must hear at its next model call goes
/// in after the request's last item (and stays at its place for the rest of
/// the run), then the tripwire (when wrapped) and the client; every call and
/// its usage is counted.
struct TurnModel {
    inner: Arc<dyn Model>,
    pending: Arc<Mutex<Vec<String>>>,
    /// The pictures since the last call.
    shots: Arc<Mutex<Vec<Picture>>>,
    told: Mutex<Vec<(usize, Told)>>,
    calls: Arc<Calls>,
}

/// What the agent was told after a request's last item, kept at its place.
enum Told {
    Note(String),
    /// A picture; only the last [`SHOTS_SHOWN`] keep their image.
    Shot(Box<Picture>),
}

/// The screenshots a model call still sees (older ones are named, not shown).
const SHOTS_SHOWN: usize = 3;

impl Model for TurnModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn config_for_trace(&self) -> Value {
        self.inner.config_for_trace()
    }

    fn generation_trace(&self) -> GenerationTrace {
        self.inner.generation_trace()
    }

    fn stream(
        &self,
        mut request: ModelRequest,
    ) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        {
            let mut told = lock(&self.told);
            let fresh: Vec<String> = std::mem::take(&mut *lock(&self.pending));
            if !fresh.is_empty() {
                told.push((request.input.len(), Told::Note(fresh.join("\n\n"))));
            }
            for picture in std::mem::take(&mut *lock(&self.shots)) {
                told.push((request.input.len(), Told::Shot(Box::new(picture))));
            }
            // An older screenshot is named, not shown again (and not kept).
            let mut shown = 0;
            for (_, item) in told.iter_mut().rev() {
                if let Told::Shot(picture) = item {
                    shown += 1;
                    if shown > SHOTS_SHOWN {
                        *item = Told::Note(picture.earlier.clone());
                    }
                }
            }
            for (offset, (at, item)) in told.iter().enumerate() {
                let at = (*at + offset).min(request.input.len());
                let input = match item {
                    Told::Note(text) => InputItem::User(text.clone()),
                    Told::Shot(picture) => InputItem::UserImages {
                        text: picture.caption.clone(),
                        images: vec![picture.image.clone()],
                    },
                };
                request.input.insert(at, input);
            }
        }
        let carried = request
            .input
            .iter()
            .any(|item| matches!(item, InputItem::UserImages { .. }));
        let first = self.calls.calls.fetch_add(1, Ordering::SeqCst) == 0;
        let calls = self.calls.clone();
        self.inner
            .stream(request)
            .map(move |event| {
                match &event {
                    Ok(ModelEvent::Done(response)) => {
                        if let Some(usage) = response.usage {
                            calls.reported.fetch_add(1, Ordering::SeqCst);
                            *lock(&calls.input_tokens) += usage.input_tokens;
                            *lock(&calls.output_tokens) += usage.output_tokens;
                        }
                    }
                    // A call that carried images (the reader's or a
                    // screenshot) is the images' refusal first, even the
                    // first call, which carries the reader's.
                    Err(ModelError::Status(status)) if carried => {
                        *lock(&calls.image_status) = Some(*status);
                    }
                    Err(ModelError::Status(status)) if first => {
                        *lock(&calls.first_status) = Some(*status);
                    }
                    _ => {}
                }
                event
            })
            .boxed()
    }
}

// ------------------------------------------------------------ the tools

pub(crate) fn agent_error(error: ToolError) -> AgentToolError {
    AgentToolError::new(error.0)
}

fn blocking_tool(
    tools: Arc<TurnTools>,
    work: impl FnOnce(&TurnTools) -> Result<String, ToolError> + Send + 'static,
) -> BoxFuture<'static, Result<String, ToolError>> {
    let handle = tools.inner.handle.clone();
    async move {
        handle
            .spawn_blocking(move || work(&tools))
            .await
            .unwrap_or_else(|_| Err(ToolError("The tool stopped unexpectedly.".to_owned())))
    }
    .boxed()
}

pub(crate) fn read_tool(
    tools: Arc<TurnTools>,
    name: &'static str,
    args: Value,
) -> BoxFuture<'static, Result<String, ToolError>> {
    blocking_tool(tools, move |tools| {
        let overlay: &dyn Overlay = tools.staging.as_ref();
        let ctx = ReadContext {
            workspace: tools.folder()?,
            runner: &tools.inner.runner,
            overlay,
        };
        let _ = NoOverlay;
        use crate::tools::read;
        match name {
            "list_dir" => read::list_dir(&ctx, &parse_args(&args)?),
            "glob" => read::glob(&ctx, &parse_args(&args)?),
            "read_file" => read::read_file(&ctx, &parse_args(&args)?),
            _ => read::grep(&ctx, &parse_args(&args)?),
        }
    })
}

/// `use_skill` and `read_skill_file`: read tools, in both modes, over the
/// skills the turn lists (`crate::skills`). A folder's skill file is read as
/// `read_file` reads, through the path rules; the reader's own only inside
/// its folder.
fn skill_tool(
    tools: Arc<TurnTools>,
    name: &'static str,
    args: Value,
) -> BoxFuture<'static, Result<String, ToolError>> {
    blocking_tool(tools, move |tools| {
        let wanted = text_arg(&args, "name")?;
        let Some(skill) = tools.skills.find(&wanted) else {
            return Err(ToolError(format!(
                "There is no skill named {wanted}; the skills are listed in the first message."
            )));
        };
        if name == "use_skill" {
            return Ok(crate::skills::use_text(skill));
        }
        let path = text_arg(&args, "path")?;
        match &skill.place {
            crate::skills::Place::Local(dir) => {
                crate::skills::read_user_file(dir, &path).map_err(ToolError)
            }
            crate::skills::Place::Folder(dir) => {
                let rel = crate::skills::relative(&path).map_err(ToolError)?;
                let overlay: &dyn Overlay = tools.staging.as_ref();
                let ctx = ReadContext {
                    workspace: tools.folder()?,
                    runner: &tools.inner.runner,
                    overlay,
                };
                crate::tools::read::read_file(
                    &ctx,
                    &crate::tools::read::ReadFileArgs {
                        path: format!("{dir}/{rel}"),
                        offset: None,
                        limit: None,
                    },
                )
            }
        }
    })
}

/// The skills' function tools.
fn skill_function_tools(tools: &Arc<TurnTools>) -> Vec<FunctionTool> {
    prompt_agent::skill_tools()
        .into_iter()
        .map(|def| {
            let name: &'static str = def.name;
            let owner = tools.clone();
            FunctionTool::new(
                name,
                def.description,
                def.parameters,
                move |_context: ToolContext, args: Value| {
                    let work = skill_tool(owner.clone(), name, args);
                    async move { work.await.map_err(agent_error) }
                },
            )
            .with_strict(false)
        })
        .collect()
}

pub(crate) fn stage_tool(
    tools: Arc<TurnTools>,
    name: &'static str,
    call: String,
    args: Value,
) -> BoxFuture<'static, Result<String, ToolError>> {
    blocking_tool(tools, move |tools| {
        use crate::tools::edit;
        let before = tools.staging.changes();
        let ctx = edit::StageContext {
            workspace: tools.folder()?,
            runner: &tools.inner.runner,
            staging: &tools.staging,
            mode: tools.mode,
            trusted: tools.trusted,
            turn: &tools.turn,
            call: &call,
        };
        let result = match name {
            "edit_file" => edit::edit_file(&ctx, &parse_args(&args)?),
            "write_file" => edit::write_file(&ctx, &parse_args(&args)?),
            _ => edit::delete_file(&ctx, &parse_args(&args)?),
        }?;
        for change in tools.staging.changes() {
            if before.contains(&change) {
                continue;
            }
            let (added, removed) = views::added_removed(&tools.staging, &change.id);
            tools.convo.log.push(ConversationEventKind::Staged {
                change: change.id.clone(),
                path: change.path.clone(),
                added,
                removed,
                authority: change.authority,
            });
        }
        Ok(result)
    })
}

fn ask_tool(
    tools: Arc<TurnTools>,
    call: String,
    args: Value,
) -> BoxFuture<'static, Result<String, ToolError>> {
    async move {
        let args: crate::tools::ask::AskQuestionArgs = parse_args(&args)?;
        let convo = tools.convo.clone();
        let turn = tools.turn.clone();
        let now = tools.inner.now();
        let waiting = tools
            .inner
            .questions
            .ask(&convo.id.clone(), &call, args, |asked| {
                convo.record(&Item::Question {
                    turn,
                    call_id: asked.call_id.clone(),
                    text: asked.question.clone(),
                    at: now,
                });
                convo.log.push(ConversationEventKind::Question {
                    call_id: asked.call_id.clone(),
                    text: asked.question.clone(),
                    options: asked.options.clone(),
                });
            })?;
        tools.inner.note_changed(&tools.convo.id);
        waiting.await
    }
    .boxed()
}

fn command_tool(
    tools: Arc<TurnTools>,
    call: String,
    _args: Value,
) -> BoxFuture<'static, Result<String, ToolError>> {
    async move {
        let ready = tools.convo.state().ready.remove(&call);
        let (prepared, approval) = match ready {
            Some(Ready::Approved(approved)) => *approved,
            Some(Ready::Refused(sentence)) => return Err(ToolError(sentence)),
            // An MCP call's, a browser action's or a desktop action's
            // approval never readies a command.
            Some(Ready::Mcp) | Some(Ready::Browser) | Some(Ready::Desktop) | None => {
                return Err(ToolError("Tool execution was not approved.".to_owned()));
            }
        };
        if prepared.background {
            return super::background::start(tools, call, prepared, approval);
        }
        let stop = StopHandle::default();
        tools.convo.state().stops.insert(call.clone(), stop.clone());
        if tools
            .convo
            .state()
            .running
            .as_ref()
            .is_some_and(|running| running.stop_requested)
        {
            stop.stop();
        }
        tools.convo.log.push(ConversationEventKind::CommandStarted {
            call_id: call.clone(),
            mode: approval.mode(),
        });
        let log = tools.convo.log.clone();
        let progress_call = call.clone();
        let progress: ProgressSink = Arc::new(move |report: Progress| {
            log.push(ConversationEventKind::CommandProgress {
                call_id: progress_call.clone(),
                bytes: report.bytes,
                lines: report.lines,
                tail_preview: report.tail,
            });
        });
        let handle = tools.inner.handle.clone();
        let work = tools.clone();
        let run_call = call.clone();
        let outcome = handle
            .spawn_blocking(move || {
                let Some(guards) = work.guards() else {
                    return Err(crate::exec::run::Refused {
                        sentence: words::NO_FOLDER_TOOL.to_owned(),
                        conflict: false,
                        rejected: false,
                    });
                };
                let ctx = work.command_context(&guards);
                run_with_progress(&ctx, &prepared, &approval, &run_call, &stop, Some(progress))
            })
            .await;
        tools.convo.state().stops.remove(&call);
        let outcome = match outcome {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(refused)) => return Err(ToolError(refused.sentence)),
            Err(_) => return Err(ToolError("The command stopped unexpectedly.".to_owned())),
        };
        tools.convo.log.push(ConversationEventKind::CommandExited {
            call_id: call.clone(),
            code: outcome.code,
            duration_ms: outcome.duration_ms,
            reason: outcome.reason,
        });
        if let Some(effect) = &outcome.after.effect {
            tools.convo.log.push(ConversationEventKind::CommandEffect {
                call_id: call.clone(),
                before: effect.before,
                after: effect.after,
                files: effect.files.clone(),
            });
        }
        if let Some(blob) = &outcome.output_blob {
            lock(&tools.outputs).insert(call.clone(), blob.clone());
        }
        Ok(outcome.model_text)
    }
    .boxed()
}

/// An MCP tool's call, once approved (§12): it goes to the server the turn
/// listed it from, and its text comes back as the tool's output.
fn mcp_tool(
    tools: Arc<TurnTools>,
    call: String,
    model_name: String,
    args: Value,
) -> BoxFuture<'static, Result<String, ToolError>> {
    async move {
        match tools.convo.state().ready.remove(&call) {
            Some(Ready::Mcp) => {}
            Some(Ready::Refused(sentence)) => return Err(ToolError(sentence)),
            _ => return Err(ToolError("Tool execution was not approved.".to_owned())),
        }
        let bound = lock(&tools.mcp)
            .get(&model_name)
            .cloned()
            .ok_or_else(|| ToolError("That MCP tool is not offered in this turn.".to_owned()))?;
        let answer = tools
            .inner
            .mcp
            .call(&bound.key, &bound.sha256, &bound.tool, args)
            .await
            .map_err(ToolError)?;
        if answer.is_error {
            let text = if answer.text.trim().is_empty() {
                "The server said the call failed.".to_owned()
            } else {
                answer.text
            };
            return Err(ToolError(text));
        }
        Ok(answer.text)
    }
    .boxed()
}

fn text_arg(args: &Value, key: &str) -> Result<String, ToolError> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| ToolError(format!("Give {key}.")))
}

fn number_arg(args: &Value, key: &str) -> Result<f64, ToolError> {
    args.get(key)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite())
        .ok_or_else(|| ToolError(format!("Give {key} as a number.")))
}

/// The effect a browser action states, if it is one.
fn effect_of(args: &Value) -> Option<Effect> {
    args.get("effect")
        .and_then(Value::as_str)
        .and_then(Effect::parse)
}

/// One action in the agent's browser; its screenshot waits for the next
/// model call. An effect that asks needs the reader's approval first.
fn browser_tool(
    tools: Arc<TurnTools>,
    call: String,
    name: &'static str,
    args: Value,
) -> BoxFuture<'static, Result<String, ToolError>> {
    async move {
        let browser = tools.inner.browser.clone();
        // Words, not a screenshot: the page's text, or a search's results.
        match name {
            "browser_read" => {
                let from = args.get("from").and_then(Value::as_u64).unwrap_or(0);
                let page = browser.page_text().await.map_err(ToolError)?;
                return Ok(page_text_result(&page, usize::try_from(from).unwrap_or(usize::MAX)));
            }
            "web_search" => {
                let query = text_arg(&args, "query")?;
                let hits = browser.search(&query).await.map_err(ToolError)?;
                return Ok(search_result(&query, &hits));
            }
            _ => {}
        }
        let shot = match name {
            "browser_look" => browser.look().await,
            "browser_open" => browser.open(&text_arg(&args, "url")?).await,
            "browser_click" | "browser_key" => {
                let effect = effect_of(&args).ok_or_else(|| {
                    ToolError(
                        "Say its effect: view, edit, share, buy, delete or account.".to_owned(),
                    )
                })?;
                if effect.asks() {
                    match tools.convo.state().ready.remove(&call) {
                        Some(Ready::Browser) => {}
                        Some(Ready::Refused(sentence)) => return Err(ToolError(sentence)),
                        _ => return Err(ToolError("Tool execution was not approved.".to_owned())),
                    }
                }
                if name == "browser_click" {
                    let double = args.get("double").and_then(Value::as_bool).unwrap_or(false);
                    let (x, y) = (number_arg(&args, "x")?, number_arg(&args, "y")?);
                    browser.click(x, y, double, effect).await
                } else {
                    let press =
                        browser_policy::parse_keys(&text_arg(&args, "key")?).map_err(ToolError)?;
                    browser.press(&press, effect).await
                }
            }
            "browser_type" => browser.type_text(&text_arg(&args, "text")?).await,
            "browser_scroll" => {
                let pixels = args
                    .get("pixels")
                    .and_then(Value::as_f64)
                    .filter(|n| n.is_finite())
                    .unwrap_or(600.0)
                    .clamp(1.0, 4000.0);
                let (dx, dy) = match args.get("direction").and_then(Value::as_str) {
                    Some("up") => (0.0, -pixels),
                    Some("left") => (-pixels, 0.0),
                    Some("right") => (pixels, 0.0),
                    _ => (0.0, pixels),
                };
                browser.scroll(640.0, 400.0, dx, dy).await
            }
            _ => browser.back().await,
        }
        .map_err(ToolError)?;
        let text = shot.summary();
        lock(&tools.shots).push(Picture::of_page(&shot));
        Ok(text)
    }
    .boxed()
}

/// The most of a page's text one `browser_read` gives.
pub(crate) const PAGE_TEXT_CHARS: usize = 20_000;

/// `browser_read`'s answer: the page's address and title, then its text from
/// character `from`, at most [`PAGE_TEXT_CHARS`], and where the rest starts.
pub(crate) fn page_text_result(page: &crate::browser::session::PageText, from: usize) -> String {
    let total = page.text.chars().count();
    let start = from.min(total);
    let part: String = page.text.chars().skip(start).take(PAGE_TEXT_CHARS).collect();
    let end = start + part.chars().count();
    let mut out = format!("Address: {}\nTitle: {}\n", page.url, page.title);
    if total == 0 {
        out.push_str("The page shows no text.");
        return out;
    }
    out.push_str(&format!("Characters {start} to {end} of {total}:\n\n{part}"));
    if end < total {
        out.push_str(&format!("\n\n[More: browser_read with from {end}.]"));
    }
    out
}

/// `web_search`'s answer: the results as a numbered list.
pub(crate) fn search_result(query: &str, hits: &[crate::browser::session::Hit]) -> String {
    if hits.is_empty() {
        return format!(
            "No results came back for \"{query}\". Try other words, or browser_look to see the page."
        );
    }
    let mut out = format!("Results for \"{query}\" (Bing):");
    for (n, hit) in hits.iter().enumerate() {
        out.push_str(&format!("\n{}. {}\n   {}", n + 1, hit.title, hit.url));
        if !hit.snippet.is_empty() {
            out.push_str(&format!("\n   {}", hit.snippet));
        }
    }
    out
}

/// The browser's function tools: a click or a key press whose effect asks
/// waits for the reader (the approval port asks the policy).
fn browser_function_tools(tools: &Arc<TurnTools>) -> Vec<FunctionTool> {
    prompt_agent::browser_tools()
        .into_iter()
        .map(|def| {
            let name: &'static str = def.name;
            let owner = tools.clone();
            let tool = FunctionTool::new(
                name,
                def.description,
                def.parameters,
                move |context: ToolContext, args: Value| {
                    let tools = owner.clone();
                    let call = tools.core_id(&context.call_id);
                    let work = browser_tool(tools, call, name, args);
                    async move { work.await.map_err(agent_error) }
                },
            )
            .with_strict(false);
            if matches!(name, "browser_click" | "browser_key") {
                tool.with_needs_approval(NeedsApproval::Predicate(Arc::new(|_, args| {
                    effect_of(args).is_some_and(Effect::asks)
                })))
            } else {
                tool
            }
        })
        .collect()
}

/// One action on the whole desktop in auto mode; its picture waits for the
/// next model call. A purchase or an account change needs the reader's
/// approval first (`desktop::policy::asks`).
fn desktop_tool(
    tools: Arc<TurnTools>,
    call: String,
    name: &'static str,
    args: Value,
) -> BoxFuture<'static, Result<String, ToolError>> {
    async move {
        let desktop = tools.inner.desktop.clone();
        let look = match name {
            "desktop_look" => desktop.look().await,
            "desktop_click" | "desktop_key" => {
                let effect = effect_of(&args).ok_or_else(|| {
                    ToolError(
                        "Say its effect: view, edit, share, buy, delete or account.".to_owned(),
                    )
                })?;
                if desktop_policy::asks(effect) {
                    match tools.convo.state().ready.remove(&call) {
                        Some(Ready::Desktop) => {}
                        Some(Ready::Refused(sentence)) => return Err(ToolError(sentence)),
                        _ => return Err(ToolError("Tool execution was not approved.".to_owned())),
                    }
                }
                if name == "desktop_click" {
                    let double = args.get("double").and_then(Value::as_bool).unwrap_or(false);
                    let (x, y) = (number_arg(&args, "x")?, number_arg(&args, "y")?);
                    desktop.click(x, y, double).await
                } else {
                    let press =
                        browser_policy::parse_keys(&text_arg(&args, "key")?).map_err(ToolError)?;
                    desktop.press(&press).await
                }
            }
            "desktop_type" => desktop.type_text(&text_arg(&args, "text")?).await,
            _ => {
                let notches = args
                    .get("notches")
                    .and_then(Value::as_i64)
                    .unwrap_or(3)
                    .clamp(1, 30) as i32;
                let direction = args
                    .get("direction")
                    .and_then(Value::as_str)
                    .unwrap_or("down");
                desktop.scroll(direction, notches).await
            }
        }
        .map_err(ToolError)?;
        let text = look.summary();
        lock(&tools.shots).push(Picture::of_screen(&look));
        Ok(text)
    }
    .boxed()
}

/// The desktop's function tools: a click or a key press whose effect is a
/// purchase or an account change waits for the reader.
fn desktop_function_tools(tools: &Arc<TurnTools>) -> Vec<FunctionTool> {
    prompt_agent::desktop_tools()
        .into_iter()
        .map(|def| {
            let name: &'static str = def.name;
            let owner = tools.clone();
            let tool = FunctionTool::new(
                name,
                def.description,
                def.parameters,
                move |context: ToolContext, args: Value| {
                    let tools = owner.clone();
                    let call = tools.core_id(&context.call_id);
                    let work = desktop_tool(tools, call, name, args);
                    async move { work.await.map_err(agent_error) }
                },
            )
            .with_strict(false);
            if matches!(name, "desktop_click" | "desktop_key") {
                tool.with_needs_approval(NeedsApproval::Predicate(Arc::new(|_, args| {
                    effect_of(args).is_some_and(desktop_policy::asks)
                })))
            } else {
                tool
            }
        })
        .collect()
}

/// The turn's tools, by mode (§7.1), then the MCP tools it offers (§12),
/// each of which always asks first (the allowlist answers for the reader),
/// then the browser's and the desktop's, when they are offered.
fn function_tools(
    tools: &Arc<TurnTools>,
    mcp: &[McpTurnTool],
    browser: bool,
    desktop: bool,
) -> Vec<FunctionTool> {
    let mut all = builtin_tools(tools);
    if !tools.skills.skills.is_empty() {
        all.extend(skill_function_tools(tools));
    }
    if browser {
        all.extend(browser_function_tools(tools));
    }
    if desktop {
        all.extend(desktop_function_tools(tools));
    }
    for offered in mcp {
        let owner = tools.clone();
        let model_name = offered.model_name.clone();
        let tool = FunctionTool::new(
            offered.model_name.clone(),
            offered.description.clone(),
            offered.parameters.clone(),
            move |context: ToolContext, args: Value| {
                let tools = owner.clone();
                let call = tools.core_id(&context.call_id);
                let work = mcp_tool(tools, call, model_name.clone(), args);
                async move { work.await.map_err(agent_error) }
            },
        )
        .with_strict(false)
        .with_needs_approval(NeedsApproval::Always)
        .with_mcp_server(offered.key.name.clone());
        all.push(tool);
    }
    all
}

fn builtin_tools(tools: &Arc<TurnTools>) -> Vec<FunctionTool> {
    let defs = if tools.workspace.is_some() {
        prompt_agent::tools(tools.mode)
    } else {
        prompt_agent::tools_without_folder()
    };
    defs.into_iter()
        .map(|def| {
            let name: &'static str = def.name;
            let owner = tools.clone();
            let tool = FunctionTool::new(
                name,
                def.description,
                def.parameters,
                move |context: ToolContext, args: Value| {
                    let tools = owner.clone();
                    let call = tools.core_id(&context.call_id);
                    let work: BoxFuture<'static, Result<String, ToolError>> = match name {
                        "list_dir" | "glob" | "read_file" | "grep" => read_tool(tools, name, args),
                        "edit_file" | "write_file" | "delete_file" => {
                            stage_tool(tools, name, call, args)
                        }
                        "ask_question" => ask_tool(tools, call, args),
                        "spawn_agent" => super::helpers::spawn_agent(tools, call, args),
                        "remember" | "forget" => blocking_tool(tools, move |tools| {
                            super::memory::tool(tools, name, &args)
                        }),
                        "suggest_task" | "withdraw_task" => blocking_tool(tools, move |tools| {
                            super::tasks::tool(tools, name, &call, &args)
                        }),
                        "write_artifact" | "read_artifact" => blocking_tool(tools, move |tools| {
                            super::artifacts::tool(tools, name, &call, &args)
                        }),
                        "command_output" | "stop_command" => blocking_tool(tools, move |tools| {
                            super::background::tool(tools, name, &args)
                        }),
                        "update_todos" => blocking_tool(tools, move |tools| {
                            super::todos::tool(tools, &call, &args)
                        }),
                        "propose_plan" => blocking_tool(tools, move |tools| {
                            super::plans::tool(tools, &call, &args)
                        }),
                        _ => command_tool(tools, call, args),
                    };
                    async move { work.await.map_err(agent_error) }
                },
            )
            .with_strict(false);
            if name == "run_command" {
                tool.with_needs_approval(NeedsApproval::Always)
            } else {
                tool
            }
        })
        .collect()
}

// -------------------------------------------------------- the approvals

/// The turn's approval port (§9.3).
struct TurnApprovals {
    tools: Arc<TurnTools>,
}

/// An MCP tool's approval (§12, §9.2): the policy's `Mcp` row decides. In
/// Ask mode or an untrusted folder it is refused; on the allowlist it is
/// allowed (recorded with the rule's name); otherwise it waits for the reader,
/// whose Approve then asks `ConfirmPort(McpCall)` ([`decide`]).
async fn mcp_request(tools: Arc<TurnTools>, request: ApprovalRequest) -> ApprovalDecision {
    use crate::policy::{Gates, Lease as PolicyLease, Standing, Target as PolicyTarget, ToolClass};
    let call = tools.core_id(&request.call_id);
    let Some(bound) = lock(&tools.mcp).get(&request.tool).cloned() else {
        tools.convo.state().ready.insert(
            call,
            Ready::Refused("That MCP tool is not offered in this turn.".into()),
        );
        return ApprovalDecision::Approve;
    };
    let allowed = {
        let (hub, key, sha, tool) = (
            tools.inner.mcp.clone(),
            bound.key.clone(),
            bound.sha256.clone(),
            bound.tool.clone(),
        );
        tools
            .inner
            .handle
            .spawn_blocking(move || hub.approvals().allowed_for(&key, &sha, &tool))
            .await
            .unwrap_or(false)
    };
    let verdict = crate::policy::decide(
        tools.mode,
        tools.trusted,
        ToolClass::Mcp,
        &PolicyTarget::None,
        &Gates {
            staged_waiting: 0,
            command_running: false,
            lease: PolicyLease::Free,
        },
        &Standing {
            command_entry: None,
            mcp_allowlisted: allowed,
        },
    );
    match verdict {
        Verdict::Refuse(reason) => {
            tools
                .convo
                .state()
                .ready
                .insert(call, Ready::Refused(reason.sentence().to_owned()));
            ApprovalDecision::Approve
        }
        Verdict::Allow(_) => {
            let by = DecidedBy::Policy {
                rule: verdict.rule().to_owned(),
            };
            tools.convo.record(&Item::ApprovalDecided {
                turn: tools.turn.clone(),
                call_id: call.clone(),
                decision: Decision::Approve,
                by: by.clone(),
                at: tools.inner.now(),
            });
            tools
                .convo
                .log
                .push(ConversationEventKind::ApprovalResolved {
                    call_id: call.clone(),
                    approved: true,
                    by,
                });
            tools.convo.state().ready.insert(call, Ready::Mcp);
            ApprovalDecision::Approve
        }
        Verdict::Ask(_) => {
            let pending = McpCall {
                key: bound.key.clone(),
                sha256: bound.sha256.clone(),
                file: bound.file.clone(),
                tool: bound.tool.clone(),
                arguments_preview: preview(&request.arguments.to_string()),
            };
            let detail = lattice_protocol::conversation::ApprovalDetail::Mcp {
                server: bound.key.name.clone(),
                tool: bound.tool.clone(),
                arguments_preview: pending.arguments_preview.clone(),
            };
            let (sender, receiver) = oneshot::channel();
            tools.convo.state().approvals.insert(
                call.clone(),
                super::agent::PendingApproval {
                    turn: tools.turn.clone(),
                    sender: Some(sender),
                    pending: super::agent::Pending::Mcp(pending),
                },
            );
            tools.convo.record(&Item::ApprovalRequested {
                turn: tools.turn.clone(),
                call_id: call.clone(),
                kind: ApprovalKind::Mcp,
                detail: detail.clone(),
                at: tools.inner.now(),
            });
            tools
                .convo
                .log
                .push(ConversationEventKind::ApprovalRequested {
                    call_id: call.clone(),
                    kind: ApprovalKind::Mcp,
                    detail,
                    allow_always_offer: true,
                });
            tools.inner.config.attention.attention(&tools.convo.id);
            tools.inner.note_changed(&tools.convo.id);
            receiver
                .await
                .unwrap_or(ApprovalDecision::Reject { note: None })
        }
    }
}

/// A browser action whose effect asks (§9.2's Browser row): the card, then
/// the native `BrowserAct` dialog ([`decide`]); refused in Ask mode or an
/// untrusted folder.
async fn browser_request(tools: Arc<TurnTools>, request: ApprovalRequest) -> ApprovalDecision {
    use crate::policy::{Gates, Lease as PolicyLease, Standing, Target as PolicyTarget, ToolClass};
    let call = tools.core_id(&request.call_id);
    let effect = effect_of(&request.arguments).unwrap_or(Effect::Share);
    let verdict = crate::policy::decide(
        tools.mode,
        tools.trusted,
        ToolClass::Browser {
            sensitive: effect.asks(),
        },
        &PolicyTarget::None,
        &Gates {
            staged_waiting: 0,
            command_running: false,
            lease: PolicyLease::Free,
        },
        &Standing::default(),
    );
    match verdict {
        Verdict::Refuse(reason) => {
            tools
                .convo
                .state()
                .ready
                .insert(call, Ready::Refused(reason.sentence().to_owned()));
            return ApprovalDecision::Approve;
        }
        Verdict::Allow(_) => {
            let by = DecidedBy::Policy {
                rule: verdict.rule().to_owned(),
            };
            tools.convo.record(&Item::ApprovalDecided {
                turn: tools.turn.clone(),
                call_id: call.clone(),
                decision: Decision::Approve,
                by: by.clone(),
                at: tools.inner.now(),
            });
            tools
                .convo
                .log
                .push(ConversationEventKind::ApprovalResolved {
                    call_id: call.clone(),
                    approved: true,
                    by,
                });
            tools.convo.state().ready.insert(call, Ready::Browser);
            return ApprovalDecision::Approve;
        }
        Verdict::Ask(_) => {}
    }
    let site = match tools.inner.browser.status() {
        BrowserStatus::Running { url, .. } => browser_policy::site_of(&url).unwrap_or(url),
        _ => "the agent's browser".to_owned(),
    };
    let args = &request.arguments;
    let action = if request.tool == "browser_click" {
        let x = args.get("x").and_then(Value::as_f64).unwrap_or(0.0);
        let y = args.get("y").and_then(Value::as_f64).unwrap_or(0.0);
        let twice = args.get("double").and_then(Value::as_bool) == Some(true);
        format!(
            "{}click at ({x:.0}, {y:.0})",
            if twice { "double-" } else { "" }
        )
    } else {
        let key: String = args
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or("")
            .chars()
            .take(40)
            .collect();
        format!("press {key}")
    };
    let what = preview(args.get("what").and_then(Value::as_str).unwrap_or(""));
    let pending = BrowserCall {
        site: site.clone(),
        action: action.clone(),
        what: what.clone(),
        effect: effect.words().to_owned(),
    };
    let detail = lattice_protocol::conversation::ApprovalDetail::Browser {
        site,
        action,
        what,
        effect: effect.name().to_owned(),
    };
    let (sender, receiver) = oneshot::channel();
    tools.convo.state().approvals.insert(
        call.clone(),
        super::agent::PendingApproval {
            turn: tools.turn.clone(),
            sender: Some(sender),
            pending: super::agent::Pending::Browser(pending),
        },
    );
    tools.convo.record(&Item::ApprovalRequested {
        turn: tools.turn.clone(),
        call_id: call.clone(),
        kind: ApprovalKind::Browser,
        detail: detail.clone(),
        at: tools.inner.now(),
    });
    tools
        .convo
        .log
        .push(ConversationEventKind::ApprovalRequested {
            call_id: call.clone(),
            kind: ApprovalKind::Browser,
            detail,
            allow_always_offer: false,
        });
    tools.inner.config.attention.attention(&tools.convo.id);
    tools.inner.note_changed(&tools.convo.id);
    receiver
        .await
        .unwrap_or(ApprovalDecision::Reject { note: None })
}

/// A desktop action that moves money or changes an account (the policy's
/// `Desktop` row): the card, then the native `DesktopAct` dialog.
async fn desktop_request(tools: Arc<TurnTools>, request: ApprovalRequest) -> ApprovalDecision {
    use crate::policy::{Gates, Lease as PolicyLease, Standing, Target as PolicyTarget, ToolClass};
    let call = tools.core_id(&request.call_id);
    let effect = effect_of(&request.arguments).unwrap_or(Effect::Buy);
    let verdict = crate::policy::decide(
        tools.mode,
        tools.trusted,
        ToolClass::Desktop {
            asks: desktop_policy::asks(effect),
        },
        &PolicyTarget::None,
        &Gates {
            staged_waiting: 0,
            command_running: false,
            lease: PolicyLease::Free,
        },
        &Standing::default(),
    );
    match verdict {
        Verdict::Refuse(reason) => {
            tools
                .convo
                .state()
                .ready
                .insert(call, Ready::Refused(reason.sentence().to_owned()));
            return ApprovalDecision::Approve;
        }
        Verdict::Allow(_) => {
            tools.convo.state().ready.insert(call, Ready::Desktop);
            return ApprovalDecision::Approve;
        }
        Verdict::Ask(_) => {}
    }
    let app = tools.inner.desktop.front().await;
    let args = &request.arguments;
    let action = if request.tool == "desktop_click" {
        let x = args.get("x").and_then(Value::as_f64).unwrap_or(0.0);
        let y = args.get("y").and_then(Value::as_f64).unwrap_or(0.0);
        let twice = args.get("double").and_then(Value::as_bool) == Some(true);
        format!(
            "{}click at ({x:.0}, {y:.0}) of the screen's picture",
            if twice { "double-" } else { "" }
        )
    } else {
        let key: String = args
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or("")
            .chars()
            .take(40)
            .collect();
        format!("press {key}")
    };
    let what = preview(args.get("what").and_then(Value::as_str).unwrap_or(""));
    let app = if app.is_empty() {
        "your desktop".to_owned()
    } else {
        preview(&app)
    };
    let pending = DesktopCall {
        app: app.clone(),
        action: action.clone(),
        what: what.clone(),
        effect: effect.words().to_owned(),
    };
    let detail = lattice_protocol::conversation::ApprovalDetail::Desktop {
        app,
        action,
        what,
        effect: effect.name().to_owned(),
    };
    let (sender, receiver) = oneshot::channel();
    tools.convo.state().approvals.insert(
        call.clone(),
        super::agent::PendingApproval {
            turn: tools.turn.clone(),
            sender: Some(sender),
            pending: super::agent::Pending::Desktop(pending),
        },
    );
    tools.convo.record(&Item::ApprovalRequested {
        turn: tools.turn.clone(),
        call_id: call.clone(),
        kind: ApprovalKind::Desktop,
        detail: detail.clone(),
        at: tools.inner.now(),
    });
    tools
        .convo
        .log
        .push(ConversationEventKind::ApprovalRequested {
            call_id: call.clone(),
            kind: ApprovalKind::Desktop,
            detail,
            allow_always_offer: false,
        });
    tools.inner.config.attention.attention(&tools.convo.id);
    tools.inner.note_changed(&tools.convo.id);
    receiver
        .await
        .unwrap_or(ApprovalDecision::Reject { note: None })
}

impl ApprovalPort for TurnApprovals {
    fn request(&self, request: ApprovalRequest) -> BoxFuture<'static, ApprovalDecision> {
        let tools = self.tools.clone();
        async move {
            if crate::mcp::names::is_model_name(&request.tool) {
                return mcp_request(tools, request).await;
            }
            if request.tool.starts_with("browser_") {
                return browser_request(tools, request).await;
            }
            if request.tool.starts_with("desktop_") {
                return desktop_request(tools, request).await;
            }
            let call = tools.core_id(&request.call_id);
            let args: Result<RunCommandArgs, _> = serde_json::from_value(request.arguments);
            let Ok(args) = args else {
                tools.convo.state().ready.insert(
                    call,
                    Ready::Refused("The arguments do not fit this tool.".into()),
                );
                return ApprovalDecision::Approve;
            };
            let work = tools.clone();
            let prepared = tools
                .inner
                .handle
                .spawn_blocking(move || {
                    let guards = work
                        .guards()
                        .ok_or_else(|| ToolError(words::NO_FOLDER_TOOL.to_owned()))?;
                    prepare(&work.command_context(&guards), &args)
                })
                .await;
            let prepared = match prepared {
                Ok(Ok(prepared)) => prepared,
                Ok(Err(error)) => {
                    tools
                        .convo
                        .state()
                        .ready
                        .insert(call, Ready::Refused(error.0));
                    return ApprovalDecision::Approve;
                }
                Err(_) => {
                    tools.convo.state().ready.insert(
                        call,
                        Ready::Refused("The command could not be checked.".into()),
                    );
                    return ApprovalDecision::Approve;
                }
            };
            if let Verdict::Allow(Because::Standing { entry }) = &prepared.verdict {
                // A standing match: no dialog, no wait.
                let entry = entry.clone();
                let approved = match tools.guards() {
                    Some(guards) => {
                        approve(&tools.command_context(&guards), &prepared, &call).await
                    }
                    None => Err(crate::exec::run::Refused {
                        sentence: words::NO_FOLDER_TOOL.to_owned(),
                        conflict: false,
                        rejected: false,
                    }),
                };
                let now = tools.inner.now();
                match approved {
                    Ok(approval) => {
                        tools.convo.record(&Item::ApprovalDecided {
                            turn: tools.turn.clone(),
                            call_id: call.clone(),
                            decision: Decision::Approve,
                            by: super::agent::standing(&entry),
                            at: now,
                        });
                        tools
                            .convo
                            .log
                            .push(ConversationEventKind::ApprovalResolved {
                                call_id: call.clone(),
                                approved: true,
                                by: super::agent::standing(&entry),
                            });
                        tools
                            .convo
                            .state()
                            .ready
                            .insert(call, Ready::Approved(Box::new((prepared, approval))));
                    }
                    Err(refused) => {
                        tools
                            .convo
                            .state()
                            .ready
                            .insert(call, Ready::Refused(refused.sentence));
                    }
                }
                return ApprovalDecision::Approve;
            }
            // Ask the reader: pending, recorded, urgent, attention; then wait
            // with no timer.
            let (sender, receiver) = oneshot::channel();
            let detail = prepared.detail.clone();
            let offer = prepared.allow_always_offer;
            tools.convo.state().approvals.insert(
                call.clone(),
                super::agent::PendingApproval {
                    turn: tools.turn.clone(),
                    sender: Some(sender),
                    pending: super::agent::Pending::Command(prepared),
                },
            );
            tools.convo.record(&Item::ApprovalRequested {
                turn: tools.turn.clone(),
                call_id: call.clone(),
                kind: ApprovalKind::Command,
                detail: detail.clone(),
                at: tools.inner.now(),
            });
            tools
                .convo
                .log
                .push(ConversationEventKind::ApprovalRequested {
                    call_id: call.clone(),
                    kind: ApprovalKind::Command,
                    detail,
                    allow_always_offer: offer,
                });
            tools.inner.config.attention.attention(&tools.convo.id);
            tools.inner.note_changed(&tools.convo.id);
            receiver
                .await
                .unwrap_or(ApprovalDecision::Reject { note: None })
        }
        .boxed()
    }
}

/// The reader's decision on a pending call (§9.3).
pub(crate) fn decide(
    inner: Arc<Inner>,
    id: String,
    call: String,
    decision: Decision,
) -> BoxFuture<'static, Result<(), Refusal>> {
    async move {
        let convo = inner
            .loaded(&id)
            .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_OPEN))?;
        if let Decision::ReleaseWithheld = decision {
            if !convo.tripwire.withheld_calls().contains(&call) {
                return Err(refuse(RefusalKind::NotFound, words::NOT_WITHHELD));
            }
            let label = {
                let state = convo.state();
                state
                    .running
                    .as_ref()
                    .map(|running| running.label.clone())
                    .or_else(|| state.remote_ok.as_ref().map(|(label, _, _)| label.clone()))
                    .unwrap_or_default()
            };
            let confirmed = inner
                .confirmer
                .ask(
                    &format!("release:{id}:{call}"),
                    ConfirmRequest::ReleaseWithheld {
                        conversation: id.clone(),
                        label,
                        call: call.clone(),
                    },
                    Initiated::Page,
                )
                .await;
            if !confirmed {
                return Err(refuse(RefusalKind::Invalid, words::RELEASE_REFUSED));
            }
            convo.tripwire.release(&call);
            return Ok(());
        }
        let (turn, pending) = {
            let state = convo.state();
            let pending = state
                .approvals
                .get(&call)
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_PENDING))?;
            (pending.turn.clone(), pending.pending.clone())
        };
        // What an Approve's dialog gave, put in place only if this decision
        // is the first (below).
        let mut ready = None;
        let (approved, decided) = match (&decision, pending) {
            (Decision::Approve, super::agent::Pending::Command(prepared)) => {
                let tools = turn_tools_for(&inner, &convo, &turn)?;
                let result = match tools.guards() {
                    Some(guards) => {
                        approve(&tools.command_context(&guards), &prepared, &call).await
                    }
                    None => Err(crate::exec::run::Refused {
                        sentence: words::NO_FOLDER_TOOL.to_owned(),
                        conflict: false,
                        rejected: false,
                    }),
                };
                match result {
                    Ok(approval) => {
                        ready = Some(Ready::Approved(Box::new((prepared, approval))));
                        (true, ApprovalDecision::Approve)
                    }
                    // Refused in the native dialog: a Reject with no note
                    // (CP3), so the page cannot ask again for this call.
                    Err(refused) if refused.rejected => {
                        (false, ApprovalDecision::Reject { note: None })
                    }
                    Err(refused) => {
                        let kind = if refused.conflict {
                            RefusalKind::Conflict
                        } else {
                            RefusalKind::Invalid
                        };
                        return Err(refuse(kind, &refused.sentence));
                    }
                }
            }
            // §12: an MCP call's Approve only asks; the native McpCall dialog
            // decides, and its no is a Reject with no note (CP3).
            (Decision::Approve, super::agent::Pending::Mcp(pending)) => {
                let confirmed = inner
                    .confirmer
                    .ask(
                        &format!("mcp-call:{id}:{call}"),
                        ConfirmRequest::McpCall {
                            server: pending.key.name.clone(),
                            tool: pending.tool.clone(),
                            arguments: pending.arguments_preview.clone(),
                        },
                        Initiated::Page,
                    )
                    .await;
                if confirmed {
                    ready = Some(Ready::Mcp);
                    (true, ApprovalDecision::Approve)
                } else {
                    (false, ApprovalDecision::Reject { note: None })
                }
            }
            // A browser action whose effect asks: the native BrowserAct
            // dialog decides, and its no is a Reject with no note (CP3).
            (Decision::Approve, super::agent::Pending::Browser(pending)) => {
                let confirmed = inner
                    .confirmer
                    .ask(
                        &format!("browser:{id}:{call}"),
                        ConfirmRequest::BrowserAct {
                            site: pending.site.clone(),
                            action: pending.action.clone(),
                            what: pending.what.clone(),
                            effect: pending.effect.clone(),
                        },
                        Initiated::Page,
                    )
                    .await;
                if confirmed {
                    ready = Some(Ready::Browser);
                    (true, ApprovalDecision::Approve)
                } else {
                    (false, ApprovalDecision::Reject { note: None })
                }
            }
            // A desktop action that moves money or changes an account: the
            // native DesktopAct dialog decides, and its no is a Reject with no
            // note (CP3).
            (Decision::Approve, super::agent::Pending::Desktop(pending)) => {
                let confirmed = inner
                    .confirmer
                    .ask(
                        &format!("desktop:{id}:{call}"),
                        ConfirmRequest::DesktopAct {
                            app: pending.app.clone(),
                            action: pending.action.clone(),
                            what: pending.what.clone(),
                            effect: pending.effect.clone(),
                        },
                        Initiated::Page,
                    )
                    .await;
                if confirmed {
                    ready = Some(Ready::Desktop);
                    (true, ApprovalDecision::Approve)
                } else {
                    (false, ApprovalDecision::Reject { note: None })
                }
            }
            (Decision::Reject { note }, _) => {
                (false, ApprovalDecision::Reject { note: note.clone() })
            }
            (Decision::ReleaseWithheld, _) => unreachable!("handled above"),
        };
        settle(&inner, &convo, &id, &call, turn, approved, decided, ready).await
    }
    .boxed()
}

/// The end of a decision (§9.3). The first decision wins: an Approve whose
/// dialog was open while a Reject (or another Approve) decided the call
/// finds it no longer pending, and records, sends and makes ready nothing.
#[allow(clippy::too_many_arguments)]
async fn settle(
    inner: &Arc<Inner>,
    convo: &Arc<Convo>,
    id: &str,
    call: &str,
    turn: String,
    approved: bool,
    decided: ApprovalDecision,
    ready: Option<Ready>,
) -> Result<(), Refusal> {
    let sender = {
        let mut state = convo.state();
        let Some(mut pending) = state.approvals.remove(call) else {
            return Err(refuse(RefusalKind::NotFound, words::NOT_PENDING));
        };
        if let Some(ready) = ready {
            state.ready.insert(call.to_owned(), ready);
        }
        pending.sender.take()
    };
    let recorded = match &decided {
        ApprovalDecision::Approve => Decision::Approve,
        ApprovalDecision::Reject { note } => Decision::Reject { note: note.clone() },
    };
    convo.record(&Item::ApprovalDecided {
        turn,
        call_id: call.to_owned(),
        decision: recorded,
        by: DecidedBy::Reader,
        at: inner.now(),
    });
    convo.log.push(ConversationEventKind::ApprovalResolved {
        call_id: call.to_owned(),
        approved,
        by: DecidedBy::Reader,
    });
    if let Some(sender) = sender {
        let _ = sender.send(decided);
    }
    // S25: after each approval is resolved.
    let (store, thread) = (inner.config.store.clone(), id.to_owned());
    let _ = inner
        .handle
        .spawn_blocking(move || super::agent::touch(store.as_ref(), &thread))
        .await;
    inner.note_changed(id);
    Ok(())
}

/// "Allow always" for a pending MCP call (§12): only through the native
/// `AllowMcpTool` dialog; the reader's yes allows the tool and runs this call.
pub(crate) fn allow_mcp_always(
    inner: Arc<Inner>,
    id: String,
    call: String,
) -> BoxFuture<'static, Result<(), Refusal>> {
    async move {
        let convo = inner
            .loaded(&id)
            .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_OPEN))?;
        let (turn, pending) = {
            let state = convo.state();
            let pending = state
                .approvals
                .get(&call)
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_PENDING))?;
            let super::agent::Pending::Mcp(mcp) = &pending.pending else {
                return Err(refuse(
                    RefusalKind::Conflict,
                    "That call is not an MCP tool's.",
                ));
            };
            (pending.turn.clone(), mcp.clone())
        };
        let allowed = inner
            .mcp
            .allow_always_for(
                &pending.key,
                &pending.sha256,
                &pending.file,
                &pending.tool,
                &inner.confirmer,
                &format!("allow-mcp:{id}:{call}"),
                Initiated::Page,
            )
            .await
            .map_err(|sentence| refuse(RefusalKind::Invalid, &sentence))?;
        if !allowed {
            return Ok(());
        }
        settle(
            &inner,
            &convo,
            &id,
            &call,
            turn,
            true,
            ApprovalDecision::Approve,
            Some(Ready::Mcp),
        )
        .await
    }
    .boxed()
}

/// The pending call's tools context (decide runs outside the turn's tools).
fn turn_tools_for(
    inner: &Arc<Inner>,
    convo: &Arc<Convo>,
    turn: &str,
) -> Result<TurnTools, Refusal> {
    let state = convo.state();
    let (Some(workspace), Some(staging), Some(checkpoints), Some(lease)) = (
        state.workspace.clone(),
        state.staging.clone(),
        state.checkpoints.clone(),
        state.lease.clone(),
    ) else {
        return Err(refuse(RefusalKind::NotFound, words::NOT_PENDING));
    };
    let remote = state
        .running
        .as_ref()
        .filter(|running| !running.affirmatively_local)
        .map(|running| running.label.clone());
    let mode = state.mode;
    drop(state);
    let trusted =
        inner.trust.state(&workspace) == lattice_protocol::conversation::TrustState::Trusted;
    Ok(TurnTools {
        inner: inner.clone(),
        convo: convo.clone(),
        turn: turn.to_owned(),
        mode,
        trusted,
        workspace: Some(workspace),
        staging,
        checkpoints,
        lease: Some(lease),
        remote,
        ids: Mutex::default(),
        outputs: Mutex::default(),
        mcp: Mutex::default(),
        shots: Arc::default(),
        // A pending call's decision reads no skill.
        skills: crate::skills::Skills::default(),
        helper_model: std::sync::OnceLock::new(),
        helpers: std::sync::atomic::AtomicUsize::new(0),
    })
}

/// "Allow always" for a pending command (X3): only through the native
/// `AllowAlways` dialog.
pub(crate) fn allow_always(
    inner: Arc<Inner>,
    id: String,
    call: String,
) -> BoxFuture<'static, Result<AllowEntry, Refusal>> {
    async move {
        let convo = inner
            .loaded(&id)
            .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_OPEN))?;
        let (candidate, workspace) = {
            let state = convo.state();
            let pending = state
                .approvals
                .get(&call)
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NOT_PENDING))?;
            let candidate = match &pending.pending {
                super::agent::Pending::Command(prepared) => prepared.candidate.clone(),
                super::agent::Pending::Mcp(_)
                | super::agent::Pending::Browser(_)
                | super::agent::Pending::Desktop(_) => None,
            };
            (candidate, state.workspace.clone())
        };
        let (Some(candidate), Some(workspace)) = (candidate, workspace) else {
            return Err(refuse(
                RefusalKind::Conflict,
                "That command cannot be allowed always.",
            ));
        };
        inner
            .permissions
            .allow_always(
                &inner.confirmer,
                &workspace,
                &candidate,
                &format!("allow-always:{id}:{call}"),
            )
            .await
            .map_err(|error| refuse(RefusalKind::Invalid, &error.sentence()))
    }
    .boxed()
}

// -------------------------------------------------------------- review

/// A review's disk work on the runtime's blocking pool (as the service's
/// other methods run theirs): each step builds its context over the
/// conversation's own parts there, so a checkpoint's git processes, the
/// write and its retries never hold an async worker.
struct OnBlockingPool {
    inner: Arc<Inner>,
    workspace: Workspace,
    staging: Arc<Staging>,
    checkpoints: Arc<Checkpoints>,
    lease: Arc<WriterLease>,
    mode: Mode,
    trusted: bool,
}

impl crate::staging::review::Offload for OnBlockingPool {
    fn run(&self, job: crate::staging::review::Job) -> BoxFuture<'_, ()> {
        let inner = self.inner.clone();
        let workspace = self.workspace.clone();
        let staging = self.staging.clone();
        let checkpoints = self.checkpoints.clone();
        let lease = self.lease.clone();
        let (mode, trusted) = (self.mode, self.trusted);
        let handle = inner.handle.clone();
        async move {
            let _ = handle
                .spawn_blocking(move || {
                    let guards = FolderGuards {
                        checkpoints: &checkpoints,
                        workspace: &workspace,
                        runner: &inner.runner,
                        lease: &lease,
                        commands: &inner.slots,
                    };
                    let ctx = crate::staging::review::ReviewContext {
                        workspace: &workspace,
                        runner: &inner.runner,
                        staging: &staging,
                        mode,
                        trusted,
                        guards: &guards,
                        confirmer: &inner.confirmer,
                        clock: &inner.config.clock,
                    };
                    job(&ctx);
                })
                .await;
        }
        .boxed()
    }
}

/// A review, and what the agent hears of it at its next model call (§7.5
/// step 9: what was kept and undone, the notes; step 3: a conflict).
pub(crate) fn review(
    inner: Arc<Inner>,
    id: String,
    ops: Vec<ReviewOp>,
) -> BoxFuture<'static, Result<ReviewOutcome, Refusal>> {
    async move {
        let work = inner.clone();
        let load_id = id.clone();
        let convo = inner
            .handle
            .spawn_blocking(move || work.load(&load_id))
            .await
            .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))?;
        let (workspace, staging, checkpoints, lease, mode) = {
            let state = convo.state();
            (
                state.workspace.clone(),
                state.staging.clone(),
                state.checkpoints.clone(),
                state.lease.clone(),
                state.mode,
            )
        };
        let (Some(workspace), Some(staging), Some(checkpoints), Some(lease)) =
            (workspace, staging, checkpoints, lease)
        else {
            return Err(refuse(RefusalKind::NotFound, "That change does not exist."));
        };
        let trusted =
            inner.trust.state(&workspace) == lattice_protocol::conversation::TrustState::Trusted;
        let notes: Vec<Option<String>> = ops
            .iter()
            .map(|op| match op {
                ReviewOp::Undo { note, .. } | ReviewOp::UndoAll { note } => note.clone(),
                _ => None,
            })
            .collect();
        let outcome = {
            let guards = FolderGuards {
                checkpoints: &checkpoints,
                workspace: &workspace,
                runner: &inner.runner,
                lease: &lease,
                commands: &inner.slots,
            };
            let ctx = crate::staging::review::ReviewContext {
                workspace: &workspace,
                runner: &inner.runner,
                staging: &staging,
                mode,
                trusted,
                guards: &guards,
                confirmer: &inner.confirmer,
                clock: &inner.config.clock,
            };
            // Only the native dialog is awaited here; every step that
            // touches the folder, git or the record runs on the blocking
            // pool.
            let offload = OnBlockingPool {
                inner: inner.clone(),
                workspace: workspace.clone(),
                staging: staging.clone(),
                checkpoints: checkpoints.clone(),
                lease: lease.clone(),
                mode,
                trusted,
            };
            crate::staging::review::review_with(&ctx, &offload, ops).await
        };
        let mut heard = Vec::new();
        for result in &outcome.results {
            convo.log.push(ConversationEventKind::Reviewed {
                change: result.change.clone(),
                result: result.result.clone(),
            });
            match &result.result {
                ReviewResult::Kept { .. } => {
                    heard.push(format!("The user kept `{}`.", result.path))
                }
                ReviewResult::PartlyKept { .. } => {
                    heard.push(format!("The user kept part of `{}`.", result.path));
                }
                ReviewResult::Undone => heard.push(format!("The user undid `{}`.", result.path)),
                ReviewResult::Conflict { reason } => {
                    convo.log.push(ConversationEventKind::Conflict {
                        change: result.change.clone(),
                        path: result.path.clone(),
                        reason: reason.clone(),
                    });
                    heard.push(crate::staging::review::conflict_sentence(&result.path));
                }
                _ => {}
            }
        }
        for note in notes.into_iter().flatten() {
            let note: String = note
                .chars()
                .take(crate::staging::review::MAX_NOTE_CHARS)
                .collect();
            heard.push(format!("The user's note: {note}"));
        }
        if !heard.is_empty() {
            let line = heard.join(" ");
            let mut state = convo.state();
            match &state.running {
                Some(running) => lock(&running.notes).push(line),
                None => state.notes.push(line),
            }
        }
        let waiting = staging.waiting();
        let running = inner.slots.busy(&workspace.id);
        let _ = inner
            .handle
            .spawn_blocking(move || lease.release_when_idle(waiting, running))
            .await;
        inner.note_changed(&id);
        Ok(outcome)
    }
    .boxed()
}

// ---------------------------------------------------------------- queue

/// Queue `text` as the conversation's next message (a steer the run gave
/// back, or one it never sent; §4.4).
pub(crate) fn queue_message(
    inner: &Arc<Inner>,
    convo: &Arc<Convo>,
    text: String,
    shown: Option<lattice_protocol::Shown>,
) {
    let queued_id = format!("q_{}", &uuid::Uuid::new_v4().simple().to_string()[..12]);
    let shown = shown.unwrap_or_else(|| {
        let state = convo.state();
        let (local, label) = state
            .running
            .as_ref()
            .map(|running| (running.affirmatively_local, running.label.clone()))
            .unwrap_or((true, String::new()));
        lattice_protocol::Shown {
            locality: if local {
                Locality::Local
            } else {
                Locality::Remote
            },
            label,
        }
    });
    let position = {
        let mut state = convo.state();
        state
            .queue
            .push_back(lattice_protocol::conversation::QueuedMessage {
                queued_id: queued_id.clone(),
                text: text.clone(),
                shown: shown.clone(),
            });
        state.queue.len() as u32
    };
    convo.record(&Item::Queued {
        queued_id: queued_id.clone(),
        text: text.clone(),
        shown,
        at: inner.now(),
    });
    convo.log.push(ConversationEventKind::Queued {
        queued_id,
        text,
        position,
    });
}

// ------------------------------------------------------------- the sink

/// The turn's trace, in the run store's format (`<native>/chat/runs/`).
struct RunRecord {
    store: crate::store::RunStore,
    summary: Mutex<RunSummary>,
    file: Mutex<Option<std::fs::File>>,
    next_seq: Mutex<u64>,
    clock: crate::clock::Clock,
}

impl RunRecord {
    fn record(&self, kind: RunEventKind) {
        if matches!(kind, RunEventKind::Delta { .. }) {
            return;
        }
        let seq = {
            let mut next = lock(&self.next_seq);
            *next += 1;
            *next
        };
        let event = lattice_protocol::RunEvent {
            seq,
            at: (self.clock)(),
            kind,
        };
        // T15: every event redacted before it is written.
        let event = redacted_event(&event).unwrap_or(event);
        if let RunEventKind::SpanEnd { .. } = event.kind {
            lock(&self.summary).spans += 1;
        }
        let mut file = lock(&self.file);
        if file.is_none() {
            *file = self.store.open_events(&lock(&self.summary).id).ok();
        }
        if let (Some(file), Ok(line)) =
            (file.as_mut(), crate::store::RunStore::encode_event(&event))
        {
            let _ = crate::store::RunStore::append_line(file, &line);
        }
    }

    fn end(&self, status: RunStatus, output: Option<String>, error: Option<String>) {
        let mut summary = lock(&self.summary);
        summary.status = status;
        let now = (self.clock)();
        summary.updated_at = now;
        summary.ended_at = Some(now);
        summary.output = output.map(|text| secrets::redact(&text));
        summary.error = error;
        let _ = self.store.write_summary(&summary, true);
        *lock(&self.file) = None;
    }
}

/// The recorder's sink for one turn: the trace, and the conversation's
/// items and events from the stream.
struct TurnSink {
    tools: Arc<TurnTools>,
    record: RunRecord,
    think: Mutex<ThinkFilter>,
    /// The model's text since its last tool call (what a stop saves).
    text: Mutex<String>,
    /// Text has streamed since the last whole message arrived: a stop
    /// finds its reasoning only in `text` (RP3).
    open_text: AtomicBool,
    any_tool: AtomicBool,
}

impl TurnSink {
    /// Record one piece of reasoning (§22.8 RP3): redacted, inline or a
    /// blob, like a tool's output (T15); never an event.
    fn record_reasoning(&self, text: &str) {
        let text = crate::py::strip(text);
        if text.is_empty() {
            return;
        }
        let tools = &self.tools;
        let convo = &tools.convo;
        let payload = convo
            .state()
            .sidecar
            .as_ref()
            .and_then(|sidecar| sidecar.payload(text).ok())
            .map(|(payload, _)| payload)
            .unwrap_or(super::item::Payload::Inline(String::new()));
        convo.record(&Item::Reasoning {
            turn: tools.turn.clone(),
            text: payload,
            at: tools.inner.now(),
        });
    }
}

impl RunSink for TurnSink {
    fn record(&self, kind: RunEventKind) {
        self.record.record(kind);
    }

    fn record_stream(&self, event: StreamEvent) {
        if let Some(kind) = map_stream_event(event.clone()) {
            self.record.record(kind);
        }
        let tools = &self.tools;
        let convo = &tools.convo;
        let now = tools.inner.now();
        match event {
            StreamEvent::RawTextDelta { delta, .. } => {
                self.open_text.store(true, Ordering::SeqCst);
                lock(&self.text).push_str(&delta);
                let visible = lock(&self.think).feed(&delta);
                if !visible.is_empty() {
                    convo
                        .log
                        .push(ConversationEventKind::Delta { text: visible });
                }
            }
            StreamEvent::RunItem {
                item:
                    RunItem::ToolCall {
                        call_id,
                        name,
                        arguments,
                        ..
                    },
                ..
            } => {
                self.any_tool.store(true, Ordering::SeqCst);
                lock(&self.text).clear();
                let call = tools.core_id(&call_id);
                let (summary, target) = views::call_summary(&name, &arguments);
                let payload = convo
                    .state()
                    .sidecar
                    .as_ref()
                    .and_then(|sidecar| sidecar.payload(&arguments).ok())
                    .map(|(payload, _)| payload)
                    .unwrap_or(super::item::Payload::Inline(String::new()));
                convo.record(&Item::ToolCall {
                    turn: tools.turn.clone(),
                    call_id: call.clone(),
                    tool: name.clone(),
                    arguments: payload,
                    summary: secrets::redact(&summary),
                    at: now,
                });
                convo.log.push(ConversationEventKind::ToolCall {
                    call_id: call,
                    tool: name,
                    summary: secrets::redact(&summary),
                    target: target.map(|target| secrets::redact(&target)),
                });
            }
            StreamEvent::RunItem {
                item:
                    RunItem::ToolOutput {
                        call_id, output, ..
                    },
                ..
            } => {
                let call = tools.core_id(&call_id);
                let payload = convo
                    .state()
                    .sidecar
                    .as_ref()
                    .and_then(|sidecar| sidecar.payload(&output).ok());
                let (payload, truncated) = match payload {
                    Some((payload, _)) => {
                        let truncated = matches!(payload, super::item::Payload::Blob { .. });
                        (payload, truncated)
                    }
                    None => (super::item::Payload::Inline(String::new()), false),
                };
                let output_blob = lock(&tools.outputs).remove(&call);
                convo.record(&Item::ToolResult {
                    turn: tools.turn.clone(),
                    call_id: call.clone(),
                    output: payload,
                    withheld: false,
                    truncated,
                    at: now,
                    output_blob,
                });
                convo.log.push(ConversationEventKind::ToolOutput {
                    call_id: call,
                    preview: preview(&output),
                    withheld: false,
                    truncated: output.len() > PREVIEW,
                });
            }
            // RP3: a provider's separate reasoning, and each `<think>`
            // span of a whole message, in the order they arrive.
            StreamEvent::RunItem {
                item: RunItem::Reasoning { text, .. },
                ..
            } => self.record_reasoning(&text),
            StreamEvent::RunItem {
                item: RunItem::MessageOutput { text, .. },
                ..
            } => {
                self.open_text.store(false, Ordering::SeqCst);
                for part in think_text(&text) {
                    self.record_reasoning(&part);
                }
            }
            StreamEvent::Steered { text, .. } => {
                convo.record(&Item::Steered {
                    turn: tools.turn.clone(),
                    text: text.clone(),
                    at: now,
                });
                convo.log.push(ConversationEventKind::Steered { text });
            }
            _ => {}
        }
    }
}

// ------------------------------------------------------------------ run

pub(crate) fn settings() -> ModelSettings {
    ModelSettings {
        temperature: Some(TEMPERATURE),
        ..ModelSettings::default()
    }
}

/// Build the turn's one client (T2; N6 through `models`).
/// The turn's model client, its address, the lease that keeps a managed
/// server running, and that server's endpoint.
type Built = (
    Arc<dyn Model>,
    Option<String>,
    Lease,
    Option<crate::llama::ManagedEndpoint>,
);

async fn build_model(inner: &Inner, target: &Target) -> Result<Built, String> {
    match target {
        Target::Managed(model) => {
            let opened = inner
                .config
                .local
                .open(model.clone())
                .await
                .map_err(|error| error.sentence())?;
            let built =
                models::build_managed(&opened.endpoint, inner.config.model_factory.as_ref())
                    .map_err(|refusal| refusal.message)?;
            Ok((
                built.model,
                Some(built.base_url),
                opened.lease,
                Some(opened.endpoint),
            ))
        }
        Target::Endpoint(endpoint) => {
            let built = models::build(
                endpoint,
                inner.config.env.as_ref(),
                &inner.keys,
                inner.config.model_factory.as_ref(),
            )
            .map_err(|refusal| refusal.message)?;
            Ok((built.model, Some(built.base_url), Lease::detached(), None))
        }
        Target::Echo => Err("The development echo takes no tools.".to_owned()),
        // The labs' agents bring their own tools: their turns are plain (`crate::acp`).
        Target::Agent(agent) => Err(format!("{} works with its own tools, not Lattice's.", agent.label())),
        Target::Refused(reason) => Err(answer::not_ready_sentence(reason)),
    }
}

/// LR8′: send the tool-call probe through the turn's model, record the answer
/// (in this process, in `tool-probes.json`, and in the turn's sidecar), and
/// say so; `true` when it passed. A server whose `/props` reports no chat
/// template fails it unasked (LR8); one whose `/props` cannot be read is
/// judged by the probe alone.
async fn tool_probe(
    tools: &Arc<TurnTools>,
    model: &dyn Model,
    key: &str,
    template: Option<bool>,
) -> bool {
    let inner = &tools.inner;
    let convo = &tools.convo;
    convo.log.push(ConversationEventKind::Stage {
        stage: lattice_protocol::chat::Stage::Checking,
        detail: "Checking that this model calls tools".to_owned(),
    });
    let outcome = if template == Some(false) {
        super::probe::Outcome {
            passed: false,
            reply: serde_json::json!({"error": "the server's /props reports no chat template"})
                .to_string(),
        }
    } else {
        super::probe::run(model).await
    };
    lock(&inner.tool_probes).insert(key.to_owned(), outcome.passed);
    let (paths, record_key, passed) = (inner.config.paths.clone(), key.to_owned(), outcome.passed);
    let kept = inner
        .handle
        .spawn_blocking(move || {
            crate::llama::probes::record_tool_probe(&paths, &record_key, passed)
        })
        .await
        .unwrap_or_else(|_| Err("The record could not be written.".to_owned()));
    let payload = |text: &str| {
        convo
            .state()
            .sidecar
            .as_ref()
            .and_then(|sidecar| sidecar.payload(text).ok())
            .map(|(payload, _)| payload)
            .unwrap_or_else(|| super::item::Payload::Inline(String::new()))
    };
    convo.record(&Item::ToolProbe {
        turn: tools.turn.clone(),
        key: key.to_owned(),
        request: payload(&super::probe::request_record()),
        reply: payload(&outcome.reply),
        passed: outcome.passed,
        at: inner.now(),
    });
    // A pass is said; a failure is the turn's error. A record not kept is
    // said either way.
    let mut notice = outcome.passed.then(|| words::PROBE_PASSED.to_owned());
    if let Err(why) = kept {
        let text = notice.get_or_insert_with(String::new);
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&format!(
            "{why} It is checked again after Lattice restarts."
        ));
    }
    if let Some(text) = notice {
        convo.record(&Item::Notice {
            text: text.clone(),
            at: inner.now(),
        });
        convo.log.push(ConversationEventKind::Notice { text });
    }
    outcome.passed
}

/// The MCP tools this turn offers (§12): its declared servers, each started
/// if it is not running; a server that does not start is a notice, and the
/// turn goes on without it. They are bound by the name the model sees.
async fn mcp_tools(tools: &Arc<TurnTools>, servers: &[ServerEntry]) -> McpTurnTools {
    if servers.is_empty() {
        return McpTurnTools::default();
    }
    let root = tools
        .workspace
        .as_ref()
        .map(|workspace| workspace.root.clone());
    let offered = tools
        .inner
        .mcp
        .turn_tools(servers, root.as_deref(), root.as_deref())
        .await;
    for notice in &offered.notices {
        tools.convo.log.push(ConversationEventKind::Notice {
            text: notice.clone(),
        });
    }
    let mut bound = lock(&tools.mcp);
    for tool in &offered.tools {
        bound.insert(tool.model_name.clone(), tool.clone());
    }
    drop(bound);
    offered
}

async fn run(plan: Plan) {
    let Plan {
        tools,
        resolution,
        session,
        input,
        notes,
        task,
        held,
        mcp_servers,
        browser,
        desktop,
        probe,
    } = plan;
    let inner = tools.inner.clone();
    let convo = tools.convo.clone();
    let mcp = mcp_tools(&tools, &mcp_servers).await;
    let now = inner.now();
    let run_id = crate::manager::new_run_id();
    let summary = RunSummary {
        id: run_id.clone(),
        task: secrets::redact(&task),
        agent: "lattice-chat".to_owned(),
        agent_label: "Lattice chat".to_owned(),
        model: resolution.provider.clone(),
        model_label: resolution.shown.label.clone(),
        locality: resolution.shown.locality,
        status: RunStatus::Running,
        created_at: now,
        updated_at: now,
        ended_at: None,
        trace_id: format!("trace_{}", uuid::Uuid::new_v4().simple()),
        usage: None,
        output: None,
        error: None,
        spans: 0,
    };
    let _ = inner.runs.write_summary(&summary, true);
    let trace_id = summary.trace_id.clone();
    let sink = Arc::new(TurnSink {
        tools: tools.clone(),
        record: RunRecord {
            store: crate::store::RunStore::new(inner.config.state.chat_runs_dir()),
            summary: Mutex::new(summary),
            file: Mutex::new(None),
            next_seq: Mutex::new(0),
            clock: inner.config.clock.clone(),
        },
        think: Mutex::new(ThinkFilter::new()),
        text: Mutex::new(String::new()),
        open_text: AtomicBool::new(false),
        any_tool: AtomicBool::new(false),
    });
    let calls = Arc::new(Calls::default());
    let outcome: Result<Result<RunResult, RunError>, String> =
        match build_model(&inner, &resolution.target).await {
            Err(sentence) => Err(sentence),
            Ok((model, _base_url, lease, endpoint)) => 'turn: {
                // The managed server's /props (LR8), for the probe's chat
                // template and for whether the model sees images.
                let props = match &endpoint {
                    Some(endpoint) if probe.is_some() || browser || desktop => {
                        inner.config.local.props(endpoint).await
                    }
                    _ => None,
                };
                let sees = props.as_ref().and_then(|props| props.vision) != Some(false);
                let (browser, desktop) = (browser && sees, desktop && sees);
                // LR8′: one probe request before the agent's first call; a
                // failed probe ends the turn, and later sends are plain.
                if let Some(key) = &probe {
                    let template = props.as_ref().map(|props| props.chat_template.is_some());
                    if !tool_probe(&tools, model.as_ref(), key, template).await {
                        drop(lease);
                        break 'turn Err(words::LOCAL_NO_TOOLS.to_owned());
                    }
                }
                let wrapped: Arc<dyn Model> = if resolution.affirmatively_local() {
                    model
                } else {
                    // T3: every item behind the tripwire.
                    Arc::new(TripwireModel::new(model, convo.tripwire.clone()))
                };
                // A helper answers with this model, behind the same tripwire.
                let _ = tools.helper_model.set(wrapped.clone());
                let turn_model = Arc::new(TurnModel {
                    inner: wrapped,
                    pending: notes,
                    shots: tools.shots.clone(),
                    told: Mutex::default(),
                    calls: calls.clone(),
                });
                let agent = Agent::builder("Lattice")
                    .instructions({
                        let mut text = if tools.workspace.is_some() {
                            prompt_agent::system(!mcp.tools.is_empty(), browser, desktop)
                        } else {
                            prompt_agent::system_without_folder(
                                !mcp.tools.is_empty(),
                                browser,
                                desktop,
                            )
                        };
                        if !tools.skills.skills.is_empty() {
                            text.push_str("\n\n");
                            text.push_str(prompt_agent::SKILLS_NOTE);
                        }
                        text
                    })
                    .tools(function_tools(&tools, &mcp.tools, browser, desktop))
                    .model_settings(settings())
                    .build();
                let intake = Intake::new();
                let mut config = RunConfig::new(turn_model);
                config.max_turns = inner.config.max_turns;
                config.workflow_name = "Lattice chat".to_owned();
                config.trace_id = Some(trace_id);
                config.session = session.map(|session| session as Arc<dyn lattice_agents::Session>);
                config.approvals = Some(Arc::new(TurnApprovals {
                    tools: tools.clone(),
                }));
                let recorder: Arc<dyn TraceProcessor> = Arc::new(Recorder {
                    sink: sink.clone(),
                    intake: intake.clone(),
                });
                config.processors = vec![recorder];
                let handle = run_streamed_items(agent, input, config);
                let stop_now = {
                    let mut state = convo.state();
                    match state.running.as_mut() {
                        Some(running) => {
                            running.control = Some(handle.control.clone());
                            running.stop_requested
                        }
                        None => false,
                    }
                };
                if stop_now {
                    handle.control.cancel(lattice_agents::CancelMode::Immediate);
                }
                let control = handle.control.clone();
                intake.install(handle.events);
                let (result, ()) =
                    tokio::join!(handle.result, intake.clone().consume(sink.clone()));
                intake.drain(&*sink);
                drop(lease);
                let result =
                    result.unwrap_or_else(|_| Err(RunError::User("the run stopped".into())));
                // A4a: steers the run accepted and never sent are queued.
                let unsent = match &result {
                    Ok(done) => done.unsent_steers.clone(),
                    Err(_) => control.take_unsent_steers(),
                };
                for text in unsent {
                    queue_message(&inner, &convo, text, None);
                }
                Ok(result)
            }
        };
    finish(&inner, &tools, &resolution, &sink, &calls, outcome, run_id).await;
    drop(held);
    next_queued(inner, convo).await;
}

/// End the turn: save, record, tell (§2.4 step 5).
async fn finish(
    inner: &Arc<Inner>,
    tools: &Arc<TurnTools>,
    resolution: &crate::chat::vocab::Resolution,
    sink: &TurnSink,
    calls: &Calls,
    outcome: Result<Result<RunResult, RunError>, String>,
    run_id: String,
) {
    let convo = &tools.convo;
    let mut text = String::new();
    let mut error = String::new();
    let mut cancelled = false;
    let status = match outcome {
        Err(sentence) => {
            error = sentence;
            TurnStatus::Failed
        }
        Ok(Ok(result)) => {
            text = strip_think(crate::py::strip(&result.final_output));
            TurnStatus::Completed
        }
        Ok(Err(RunError::Cancelled)) => {
            cancelled = true;
            let raw = lock(&sink.text).clone();
            // RP3: the unfinished message's reasoning is recorded too.
            if sink.open_text.load(Ordering::SeqCst) {
                for part in think_text(&raw) {
                    sink.record_reasoning(&part);
                }
            }
            let partial = strip_think(crate::py::strip(&raw));
            if partial.is_empty() {
                error = NOTHING_WRITTEN.to_owned();
            } else {
                text = partial;
            }
            TurnStatus::Stopped
        }
        Ok(Err(RunError::MaxTurnsExceeded { max_turns })) => {
            error = format!("Stopped after {max_turns} turns without a final answer.");
            TurnStatus::MaxTurns
        }
        Ok(Err(
            RunError::InputGuardrailTripwire { .. } | RunError::OutputGuardrailTripwire { .. },
        )) => {
            error = "A guardrail refused the turn.".to_owned();
            TurnStatus::Refused
        }
        Ok(Err(RunError::ModelRefusal { refusal })) => {
            error = models::refusal_sentence(&refusal);
            TurnStatus::Refused
        }
        Ok(Err(RunError::Model(failure))) => {
            let first = *lock(&calls.first_status);
            if matches!(*lock(&calls.image_status), Some(400 | 415 | 422)) {
                // A call with images refused (a screenshot, or the reader's):
                // neither images nor the browser are offered to this choice
                // again in this process; nothing is retried.
                lock(&inner.no_vision).insert(resolution.id.clone());
                error = words::NO_VISION.to_owned();
            } else if calls.calls.load(Ordering::SeqCst) == 1
                && matches!(first, Some(400 | 422))
                && matches!(resolution.target, Target::Endpoint(_))
            {
                // §4.5: cached for the process; nothing is retried.
                lock(&inner.no_tools).insert(resolution.id.clone());
                error = words::NO_TOOLS.to_owned();
            } else {
                error = models::error_sentence(&failure, None);
            }
            TurnStatus::Failed
        }
        Ok(Err(RunError::User(_))) => {
            error = "The turn stopped because one of its checks failed.".to_owned();
            TurnStatus::Failed
        }
    };
    // T15: the final answer is redacted before it is written.
    let clean = secrets::redact(&text);
    if clean != text {
        convo.record(&Item::Notice {
            text: words::ANSWER_REDACTED.to_owned(),
            at: inner.now(),
        });
        convo.log.push(ConversationEventKind::Notice {
            text: words::ANSWER_REDACTED.to_owned(),
        });
    }
    // D10: the sums only when every call reported usage.
    let made = calls.calls.load(Ordering::SeqCst);
    let (prompt_tokens, completion_tokens) =
        if made > 0 && calls.reported.load(Ordering::SeqCst) == made {
            (
                i64::try_from(*lock(&calls.input_tokens)).ok(),
                i64::try_from(*lock(&calls.output_tokens)).ok(),
            )
        } else {
            (None, None)
        };
    let new_turn = NewTurn {
        error: error.clone(),
        cancelled,
        prompt_tokens,
        completion_tokens,
        ..NewTurn::assistant(clean.clone(), resolution.provider.clone())
    };
    let store = inner.config.store.clone();
    let id = convo.id.clone();
    let pending = new_turn.clone();
    let saved = inner
        .handle
        .spawn_blocking(move || {
            // Immediately before append_answer (S25).
            super::agent::touch(store.as_ref(), &id);
            store.append_answer(&id, pending)
        })
        .await
        .ok()
        .and_then(Result::ok);
    let answer_id = saved.as_ref().map(|turn| turn.id.clone());
    let turn_view = match &saved {
        Some(turn) => turn.to_chat_turn(),
        None => new_turn.stored(String::new(), inner.now()).to_chat_turn(),
    };
    if !error.is_empty() {
        convo.log.push(ConversationEventKind::Error {
            message: error.clone(),
        });
    }
    convo.log.push(ConversationEventKind::TurnSaved {
        turn: Box::new(turn_view),
        saved: saved.is_some(),
    });
    convo.record(&Item::TurnEnd {
        turn: tools.turn.clone(),
        answer: answer_id,
        status,
        run_id: run_id.clone(),
        at: inner.now(),
    });
    let end = match status {
        TurnStatus::Completed => EndStatus::Completed,
        TurnStatus::Stopped => EndStatus::Stopped,
        TurnStatus::Refused => EndStatus::Refused,
        _ => EndStatus::Failed,
    };
    sink.record.record(RunEventKind::End { status: end });
    sink.record.end(
        end.into(),
        (!clean.is_empty()).then(|| clean.clone()),
        (!error.is_empty()).then(|| error.clone()),
    );
    convo.log.finish_turn();
    {
        let mut state = convo.state();
        state.running = None;
        state.approvals.clear();
        state.ready.clear();
        state.stops.clear();
        state.last = Some((tools.turn.clone(), status));
    }
    inner.questions.withdraw(&convo.id);
    if let (Some(lease), Some(workspace)) = (&tools.lease, &tools.workspace) {
        lease.release_when_idle(tools.staging.waiting(), inner.slots.busy(&workspace.id));
    }
    inner.agent_turns.fetch_sub(1, Ordering::SeqCst);
    convo.log.push(ConversationEventKind::TurnEnded {
        turn: tools.turn.clone(),
        status,
    });
    inner.note_changed(&convo.id);
}
