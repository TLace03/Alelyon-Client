//! The agent chat as its interface sees it: conversations, their events, staged
//! changes, approvals and checkpoints, and the `AgentChatService` trait the
//! interface calls (`the chat core's spec` §4, §13.4 and §13.5).
//!
//! A conversation is a thread of the shared chat store (its id is the thread
//! id), plus what only the native Lattice records about it: tool calls and
//! results, approvals, questions, staged changes, reviews and checkpoints.
//! These types are the contract; `lattice-core` implements the service.
//!
//! Shapes the specification names are kept as named. Where it uses a type
//! without defining it (`CoreStatus`, `Snapshot`, `WorkspaceView`, `TrustState`,
//! `RankedPath`, `ViewRef`, `Lines`, `ChangeSet`, `ReviewOutcome`,
//! `CheckpointView`, `AllowEntry`, the event fields of §13.5), the shape here is
//! the smallest one its sections describe, and the payloads the shipping web
//! app already renders (`Thread`, `FileDiff`, `Hunk`, `DiffLine`, its fuzzy
//! `Ranked` path) keep that app's field names, as §13.4 asks.
//!
//! Serialisation: a unit-only enum is a snake_case string; an enum with data
//! is an object with a snake_case `type` tag; an event is flat, with `seq`,
//! `at` and `type` beside its own fields. Every payload has a golden under
//! `tests/goldens/conversation/`, which the future TypeScript guards will read
//! as well (§13.6).
//!
//! Invariants every producer keeps:
//! - **There is no delete.** The trait removes no conversation; Archive moves a
//!   thread into the shared store's `evicted/` folder (never-delete rule ND1).
//! - Ids are checked before they reach a path or a lookup ([`is_conversation_id`],
//!   [`is_change_id`], [`is_hunk_id`], [`is_workspace_id`], [`is_call_id`],
//!   [`is_task_id`]).
//! - Events carry previews of at most 2 KiB, counters and ids; files, diffs and
//!   command output are fetched by window ([`ViewRef`]) (rule CR5).
//! - An approval or an answer reaches the core only through `decide`, `answer`
//!   and `allow_always`; nothing here is ever read out of model or tool text.
//! - No key value or key name appears in any value here.

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::chat::{ChatChoice, ChatTurn, LocalRuntime, Shown, Stage, TurnId};
use crate::{Locality, Refusal};

// ------------------------------------------------------------------ identifiers

/// A conversation's id: the shared thread id, `^[0-9A-Za-z_-]{1,64}$`. Native
/// makes uuid4 hex, first 12 characters.
pub type ConversationId = String;
/// The model's tool-call id: 1 to 128 printable ASCII characters. The core
/// replaces any other with one of its own (`lc_<16 hex>`).
pub type CallId = String;
/// `ch_<16 hex>`, unique within a conversation.
pub type ChangeId = String;
/// 16 hexadecimal characters of SHA-256 over the hunk's change, base and lines.
pub type HunkId = String;
/// 1, 2, 3 … per conversation.
pub type CheckpointId = u32;
/// 16 hexadecimal characters of SHA-256 over the root folder's volume serial and
/// 128-bit file id. A record also keeps the path, and matches only when both do.
pub type WorkspaceId = String;
/// `task_<16 hex>`: a task the agent suggested, unique within a conversation.
pub type TaskId = String;
/// `plan_<16 hex>`: a plan the agent proposed, unique within a conversation.
pub type PlanId = String;

/// True when `id` has the shape of a [`ConversationId`].
pub fn is_conversation_id(id: &str) -> bool {
    crate::chat::is_thread_id(id)
}

fn is_lower_hex(text: &str) -> bool {
    text.bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// True when `id` has the shape of a [`ChangeId`].
pub fn is_change_id(id: &str) -> bool {
    id.strip_prefix("ch_")
        .is_some_and(|hex| hex.len() == 16 && is_lower_hex(hex))
}

/// True when `id` has the shape of a [`HunkId`].
pub fn is_hunk_id(id: &str) -> bool {
    id.len() == 16 && is_lower_hex(id)
}

/// True when `id` has the shape of a [`WorkspaceId`].
pub fn is_workspace_id(id: &str) -> bool {
    id.len() == 16 && is_lower_hex(id)
}

/// True when `id` can be a [`CallId`] as the model sent it.
pub fn is_call_id(id: &str) -> bool {
    (1..=128).contains(&id.len()) && id.bytes().all(|b| (0x21..=0x7e).contains(&b))
}

/// True when `id` has the shape of a [`TaskId`].
pub fn is_task_id(id: &str) -> bool {
    id.strip_prefix("task_")
        .is_some_and(|hex| hex.len() == 16 && is_lower_hex(hex))
}

/// True when `id` has the shape of a [`PlanId`].
pub fn is_plan_id(id: &str) -> bool {
    id.strip_prefix("plan_")
        .is_some_and(|hex| hex.len() == 16 && is_lower_hex(hex))
}

/// How the Python archive identifies a conversation: its id and when it was made.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArchiveKey {
    pub id: ConversationId,
    pub created: f64,
}

// ----------------------------------------------------------------- conversations

/// Ask: read tools only. Agent: it may stage edits and ask to run commands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Ask,
    Agent,
}

/// A plain turn (no workspace, or a model without tools: one model call) or an
/// agent turn (tools, approvals, staged changes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnKind {
    Plain,
    Agent,
}

/// Where a conversation started. `Web`: no native record exists for it yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    Native,
    Web,
}

/// The folder a conversation works in, as the sidebar shows it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceBadge {
    pub id: WorkspaceId,
    /// The folder's own name.
    pub name: String,
    /// The canonical path.
    pub path: String,
    pub trusted: bool,
}

/// A conversation as the list shows it: the shipping app's `Thread` fields,
/// plus what only the native Lattice knows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub id: ConversationId,
    pub title: String,
    pub created: f64,
    pub updated: f64,
    pub turns: u64,
    pub pinned_provider: String,
    pub workspace: Option<WorkspaceBadge>,
    pub mode: Mode,
    pub origin: Origin,
    pub running: bool,
    /// An approval or a question is waiting for the reader.
    pub needs_you: bool,
}

/// Why a conversation is in the archive.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchiveReason {
    /// The 60-thread cap moved it there (Python's or native's index write).
    Cap,
    /// The reader archived it here.
    Reader,
    /// A row that is not an object with an id.
    Unknown,
}

/// An archived conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArchivedSummary {
    pub key: ArchiveKey,
    pub title: String,
    pub updated: f64,
    pub turns: u64,
    pub evicted_at: f64,
    /// The transcript's file in `evicted/`; `""` for a thread that never had a turn.
    pub file: String,
    pub reason: ArchiveReason,
    /// The row names a file that is not there (a crash leftover, tolerated).
    pub transcript_missing: bool,
}

/// The sidebar: conversations, the archive and where they are kept.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatList {
    /// Newest first.
    pub conversations: Vec<ConversationSummary>,
    /// Newest `evicted_at` first, a page at a time.
    pub archived: Vec<ArchivedSummary>,
    pub store_dir: String,
    pub installed: bool,
    /// The shared index exists but cannot be read here: nothing will be changed.
    pub index_unreadable: bool,
    pub archive_unreadable: bool,
    /// Whether the shared store's writes are open (gate G-WEB; true in a
    /// shipped build since row G5).
    pub shared_writes: bool,
    /// The most conversations the list holds before the oldest is archived (60).
    pub limit: u32,
}

/// A message to send. `conversation: None` makes a new one, lazily.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SendRequest {
    pub conversation: Option<ConversationId>,
    pub text: String,
    pub choice: String,
    pub shown: Shown,
    pub mode: Mode,
    pub workspace: Option<WorkspaceId>,
    /// Replace this user turn (and hide everything after it) with `text`.
    pub edit_of: Option<TurnId>,
    /// The project a new conversation joins (an existing one keeps its own).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    /// Images the reader attached (at most 4 PNG or JPEG pictures of at most
    /// 5 MiB each): the agent's model sees them with the words.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<UserImage>,
}

/// An image the reader attached to a message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserImage {
    /// `image/png` or `image/jpeg`.
    pub media_type: String,
    /// The picture's bytes, in standard base64.
    pub base64: String,
}

/// Answer the conversation's last question again.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegenerateRequest {
    pub conversation: ConversationId,
    pub choice: String,
    pub shown: Shown,
    pub mode: Mode,
}

/// What a send, regenerate or continue did.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Accepted {
    /// A turn started; its events arrive through `follow`. (Boxed so `Queued`
    /// does not take the space of this variant; the JSON is the same.)
    Started {
        conversation: Box<ConversationSummary>,
        user_turn: Box<ChatTurn>,
        turn_kind: TurnKind,
    },
    /// The conversation is running a turn; the message waits its turn.
    Queued {
        conversation: ConversationId,
        queued_id: String,
        position: u32,
    },
}

/// A message waiting for the running turn to end.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueuedMessage {
    pub queued_id: String,
    pub text: String,
    pub shown: Shown,
}

/// A conversation opened: its visible turns and its native record so far.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Snapshot {
    pub conversation: ConversationSummary,
    /// The shared transcript's visible turns (at most the last 400).
    pub turns: Vec<ChatTurn>,
    /// The conversation's recorded events (tool cards, approvals, changes, …),
    /// without deltas; `follow(id, last_seq)` continues from here.
    pub events: Vec<ConversationEvent>,
    pub last_seq: u64,
    pub queued: Vec<QueuedMessage>,
    /// Another native window holds this conversation's answer lock.
    pub answering_elsewhere: bool,
}

/// What the core says about itself. Memory only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreStatus {
    pub runtime: String,
    /// Agent turns running now, across every conversation (at most 3).
    pub agent_turns: u32,
    pub shared_writes: bool,
    /// Why turns cannot start, when they cannot.
    pub refusal: Option<String>,
}

/// A turn's end, as `turn_ended` and the native record give it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Completed,
    Stopped,
    /// A model or tool-protocol error.
    Failed,
    /// A guardrail or the model refused.
    Refused,
    /// The turn limit was reached; Continue is offered.
    MaxTurns,
    /// Lattice closed while the turn ran or waited; Continue is offered.
    Interrupted,
}

// --------------------------------------------------------------------- approvals

/// The reader's answer to a pending call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Decision {
    /// For a command this only asks: the core then opens the native dialog.
    Approve,
    Reject {
        note: Option<String>,
    },
    /// Send a withheld tool result to the remote model once (native dialog).
    ReleaseWithheld,
}

/// What a standing entry matches. Only `Exact` exists in this increment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchScope {
    Exact,
}

/// How an approved command runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandMode {
    /// A standing match: the resolved program, run directly, with no shell.
    Direct,
    /// Approved once: the text runs in Windows PowerShell 5.1.
    #[serde(rename = "powershell")]
    PowerShell,
}

/// What kind of call waits for the reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalKind {
    Command,
    Mcp,
    /// An action in the agent's browser whose effect asks first.
    Browser,
    /// An action on the whole desktop, in auto mode, that moves money or
    /// changes an account.
    Desktop,
}

/// The facts an approval card and its native dialog show.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApprovalDetail {
    Command {
        /// The exact text, with every non-ASCII and control character escaped.
        text: String,
        /// Workspace-relative; `""` is the root.
        cwd: String,
        mode: CommandMode,
        timeout_s: u32,
        /// The remote label its output goes to, when the turn's target is remote.
        remote: Option<String>,
        /// Staged changes waiting for review: Run is disabled while this is not 0.
        staged_waiting: u32,
        /// It keeps running after the agent's call returns, until stopped.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        background: bool,
    },
    Mcp {
        server: String,
        tool: String,
        /// At most 2 KiB of the arguments.
        arguments_preview: String,
    },
    /// An action in the agent's browser that another person will see, that
    /// moves money, deletes something or changes an account.
    Browser {
        /// The site it acts on (`x.com`).
        site: String,
        /// What it does, as Lattice describes it (`click at (612, 304)`).
        action: String,
        /// What the agent says it is (its own words, untrusted).
        what: String,
        /// `share`, `buy`, `delete` or `account`.
        effect: String,
    },
    /// An action on the whole desktop, in auto mode, that moves money or
    /// changes an account.
    Desktop {
        /// The window it acts on, as its title names it.
        app: String,
        /// What it does, as Lattice describes it (`click at (612, 304)`).
        action: String,
        /// What the agent says it is (its own words, untrusted).
        what: String,
        /// `buy` or `account`.
        effect: String,
    },
}

/// Who decided a call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecidedBy {
    Reader,
    Standing { entry: String },
    Policy { rule: String },
}

/// A standing approval: one exact command, in one folder of one workspace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AllowEntry {
    pub id: String,
    pub workspace: WorkspaceId,
    pub argv: Vec<String>,
    /// The resolved program's absolute path.
    pub program: String,
    /// Workspace-relative; the command must run in exactly this folder.
    pub cwd: String,
    pub created_at: f64,
    pub scope: MatchScope,
    /// Revoked entries stay listed and no longer match.
    pub revoked_at: Option<f64>,
}

// --------------------------------------------------------------------- workspace

/// Whether a folder is trusted (rules files, Agent mode and checkpoints).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustState {
    Untrusted,
    Trusted,
    /// Trusted once, then revoked: untrusted again.
    Revoked,
}

/// A folder attached to a conversation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceView {
    pub workspace: WorkspaceBadge,
    pub trust: TrustState,
    /// The rules files found (`AGENTS.md`, `.lattice/rules/*.md`, …); read only
    /// after trust.
    pub rules_found: Vec<String>,
    /// The repository's top level when the folder is in one Lattice may use.
    pub git_top_level: Option<String>,
}

/// One path the fuzzy file search found: the shipping app's `Ranked`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankedPath {
    pub path: String,
    pub score: u32,
    /// Character positions in `path` that matched, for highlighting.
    pub positions: Vec<u32>,
}

/// Something whose lines are read by window.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ViewRef {
    /// A workspace file as it is on disk.
    File {
        workspace: WorkspaceId,
        path: String,
    },
    /// A staged change's new text.
    Staged {
        conversation: ConversationId,
        change: ChangeId,
    },
    /// A command's output.
    Output {
        conversation: ConversationId,
        call_id: CallId,
    },
}

/// A window of lines.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lines {
    /// 1-based number of the first line returned.
    pub from: u32,
    pub total: u32,
    pub lines: Vec<String>,
    /// A line was cut at its length cap.
    pub truncated: bool,
}

// ----------------------------------------------------------------------- changes

/// What a change does to its file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Edit,
    Create,
    Overwrite,
    Delete,
    Restore,
    CommandUndo,
}

/// Where a change came from. `Command` marks a command's effect, which is
/// already on disk and is listed so it can be acknowledged or undone.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChangeOrigin {
    Agent { turn: TurnId, call: CallId },
    Restore { to: CheckpointId },
    CommandUndo { call: CallId },
    Command { call: CallId },
}

/// Where a change is in its review.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChangeState {
    Pending,
    /// The file changed outside Lattice and the edits applied again: review anew.
    Rebased,
    Conflict {
        reason: String,
    },
    Kept,
    Undone,
    PartlyKept,
    /// A command's effect: already on disk.
    OnDisk,
}

/// One change as the Changes panel lists it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeView {
    pub id: ChangeId,
    pub path: String,
    pub kind: ChangeKind,
    /// An authority file: kept only one at a time, through a native dialog.
    pub authority: bool,
    pub state: ChangeState,
    pub origin: ChangeOrigin,
    pub added: u32,
    pub removed: u32,
    pub binary: bool,
}

/// A conversation's changes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeSet {
    pub changes: Vec<ChangeView>,
    /// Changes Pending, Rebased or in Conflict: commands wait while this is not 0.
    pub waiting: u32,
    /// A command is running in this workspace: Keep waits for it.
    pub command_running: bool,
}

/// What a diff is against: the shipping app's `Against`. The chat core uses
/// `Previous` (the change's base).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Against {
    Head,
    Main,
    Previous,
}

/// One line of a hunk: the shipping app's `DiffLine`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffLineKind {
    Context,
    Add,
    Remove,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffLine {
    pub kind: DiffLineKind,
    pub text: String,
}

/// Where a hunk is in its review.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HunkState {
    Pending,
    Kept,
    Undone,
}

/// The shipping app's `Hunk`, plus its id and state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hunk {
    pub id: HunkId,
    pub state: HunkState,
    pub old_start: u32,
    pub old_lines: u32,
    pub new_start: u32,
    pub new_lines: u32,
    pub section: String,
    pub lines: Vec<DiffLine>,
}

/// The shipping app's `FileDiff`, plus the change it belongs to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDiff {
    pub change: ChangeId,
    pub path: String,
    pub against: Against,
    /// The checkpoint the change's base was taken at; 0 when there is none.
    pub snapshot: CheckpointId,
    pub binary: bool,
    pub hunks: Vec<Hunk>,
    /// Over 6,000 lines: shown cut, kept or undone only whole.
    pub truncated: bool,
}

/// One review operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReviewOp {
    /// `hunks: None` keeps the whole file.
    Keep {
        change: ChangeId,
        hunks: Option<Vec<HunkId>>,
    },
    Undo {
        change: ChangeId,
        hunks: Option<Vec<HunkId>>,
        note: Option<String>,
    },
    /// Every pending change that is not an authority file.
    KeepAll,
    UndoAll {
        note: Option<String>,
    },
}

/// What a review did to one change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReviewResult {
    /// Written. `changed_after`: the file's hash differed right after the write
    /// ("written, then changed by another program").
    Kept {
        changed_after: bool,
    },
    PartlyKept {
        remaining_hunks: u32,
    },
    Undone,
    /// The file changed on disk and the edits applied again: nothing was written.
    Rebased,
    Conflict {
        reason: String,
    },
    /// Left out (an authority file in KeepAll, a change no longer pending).
    Skipped {
        reason: String,
    },
    /// Nothing was written, for this reason.
    Failed {
        reason: String,
    },
}

/// One change's result in a review.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangeResult {
    pub change: ChangeId,
    pub path: String,
    pub result: ReviewResult,
}

/// What `review` did, per change, in path order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReviewOutcome {
    pub results: Vec<ChangeResult>,
}

// ------------------------------------------------------------------- checkpoints

/// How a checkpoint was taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointKind {
    /// A tree in the repository, under `refs/lattice/chat/`.
    Git,
    /// First-touch copies in the native record (a folder without git).
    Copies,
}

/// Why a checkpoint was taken.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CheckpointReason {
    BeforeKeep,
    BeforeCommand { call_id: CallId },
    AfterCommand { call_id: CallId },
    BeforeRestoreKeep,
}

/// A checkpoint as the restore menu lists it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointView {
    pub id: CheckpointId,
    pub kind: CheckpointKind,
    pub reason: CheckpointReason,
    pub at_turn: Option<TurnId>,
    /// A command ran that Lattice cannot see (a folder without git): a restore
    /// across it is incomplete.
    pub exposed: bool,
    /// Paths left out (too large, past the untracked cap, links, nested
    /// repositories); listed by name in the record.
    pub omitted: u32,
    /// Bytes of the copies this checkpoint holds in the native record.
    pub bytes: u64,
}

/// How a path changed between two checkpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathChange {
    Added,
    Modified,
    Deleted,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangedPath {
    pub path: String,
    pub change: PathChange,
}

/// Why a command stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitReason {
    Exited,
    Stopped,
    TimedOut,
    /// It could not be started.
    Failed,
}

/// What became of a queued message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DequeueOutcome {
    Sent {
        turn: TurnId,
    },
    Cancelled,
    /// After a restart: given back as a draft, not sent.
    Restored,
}

/// What became of a task the agent suggested. A task settles once.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskOutcome {
    /// The reader started it: a new conversation began with its prompt.
    Started { conversation: ConversationId },
    /// The reader set it aside.
    Dismissed,
    /// The agent took it back, and said why.
    Withdrawn { reason: String },
}

/// Where one step of the agent's to-do list stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Not begun.
    Pending,
    /// Being worked on now; at most one step at a time.
    InProgress,
    /// Finished.
    Done,
}

/// One step of the agent's to-do list.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    /// One line of at most 100 characters.
    pub content: String,
    pub status: TodoStatus,
}

/// What became of a plan the agent proposed. A plan settles once.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PlanOutcome {
    /// The reader approved it: the conversation went on in Agent mode to
    /// carry it out.
    Approved,
    /// The reader kept planning: the plan was set aside for their changes.
    KeptPlanning,
    /// The agent proposed another plan in its place.
    Replaced,
}

/// What an artifact holds, and so how the window shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Shown formatted.
    Markdown,
    /// Shown as source, and previewed in a browser that has no network.
    Html,
    /// Shown as source, and previewed as HTML is.
    Svg,
    /// Shown as it is.
    Text,
}

/// One version of an artifact (a document the agent saved beside the chat),
/// as the window reads it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactView {
    pub name: String,
    pub title: String,
    pub kind: ArtifactKind,
    pub version: u32,
    /// How many versions the artifact has.
    pub versions: u32,
    pub text: String,
}

// ------------------------------------------------------------------------ events

/// What happened in a conversation (§13.5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationEventKind {
    TurnStarted {
        turn: TurnId,
        kind: TurnKind,
        mode: Mode,
        label: String,
        locality: Locality,
    },
    Stage {
        stage: Stage,
        detail: String,
    },
    /// Streamed text since the last batch, joined. Live only.
    Delta {
        text: String,
    },
    /// Deltas past the per-turn bound were dropped (counted, not lost silently).
    DeltasDropped {
        n: u64,
    },
    ToolCall {
        call_id: CallId,
        tool: String,
        summary: String,
        /// The path or command it is about, when there is one.
        target: Option<String>,
    },
    ToolOutput {
        call_id: CallId,
        /// At most 2 KiB.
        preview: String,
        withheld: bool,
        truncated: bool,
    },
    /// A tool result was held back from a remote model: it looks like a secret.
    Withheld {
        call_id: CallId,
    },
    ApprovalRequested {
        call_id: CallId,
        kind: ApprovalKind,
        detail: ApprovalDetail,
        allow_always_offer: bool,
    },
    ApprovalResolved {
        call_id: CallId,
        approved: bool,
        by: DecidedBy,
    },
    Question {
        call_id: CallId,
        text: String,
        options: Vec<String>,
    },
    QuestionAnswered {
        call_id: CallId,
    },
    /// The agent suggested a task for a conversation of its own (a chip): its
    /// title and summary as the chip shows them, and the prompt that
    /// conversation would begin with (at most 2,000 characters). Nothing runs
    /// until the reader starts it.
    TaskSuggested {
        task: TaskId,
        title: String,
        summary: String,
        prompt: String,
    },
    /// A suggested task was started, dismissed or withdrawn.
    TaskSettled {
        task: TaskId,
        outcome: TaskOutcome,
    },
    /// The agent saved a version of an artifact (a document beside the
    /// chat): its name, title and kind, and which version. Its text is read
    /// through the service, not carried here.
    ArtifactSaved {
        name: String,
        title: String,
        kind: ArtifactKind,
        version: u32,
        bytes: u64,
    },
    /// The agent wrote its to-do list for the work in hand: the whole list
    /// as it now stands (at most 12 steps), which replaces the one before.
    /// An empty list clears it.
    TodosUpdated {
        items: Vec<TodoItem>,
    },
    /// The reader attached images to the user turn `turn`: how many, and
    /// their size in all. The pictures stay in the conversation's record.
    ImagesAttached {
        turn: TurnId,
        count: u32,
        bytes: u64,
    },
    /// The agent, in Ask mode, proposed a plan for the reader to approve
    /// (Markdown, at most 2,000 characters). Approving it goes on in Agent
    /// mode to carry it out; nothing changes until then.
    PlanProposed {
        plan: PlanId,
        title: String,
        text: String,
    },
    /// A proposed plan was approved, kept for more planning, or replaced.
    PlanSettled {
        plan: PlanId,
        outcome: PlanOutcome,
    },
    Staged {
        change: ChangeId,
        path: String,
        added: u32,
        removed: u32,
        authority: bool,
    },
    Reviewed {
        change: ChangeId,
        result: ReviewResult,
    },
    Conflict {
        change: ChangeId,
        path: String,
        reason: String,
    },
    Checkpoint {
        id: CheckpointId,
        kind: CheckpointKind,
        reason: CheckpointReason,
    },
    CommandStarted {
        call_id: CallId,
        mode: CommandMode,
    },
    /// Counters for everything a command wrote since the last batch.
    CommandProgress {
        call_id: CallId,
        bytes: u64,
        lines: u64,
        /// At most 512 bytes.
        tail_preview: String,
    },
    CommandExited {
        call_id: CallId,
        code: Option<i32>,
        duration_ms: u64,
        reason: ExitReason,
    },
    CommandEffect {
        call_id: CallId,
        before: CheckpointId,
        after: CheckpointId,
        files: Vec<ChangedPath>,
    },
    Steered {
        text: String,
    },
    Queued {
        queued_id: String,
        text: String,
        position: u32,
    },
    Dequeued {
        queued_id: String,
        outcome: DequeueOutcome,
    },
    Notice {
        text: String,
    },
    /// The assistant turn as saved; `saved: false` when it could not be.
    TurnSaved {
        turn: Box<ChatTurn>,
        saved: bool,
    },
    /// One sentence.
    Error {
        message: String,
    },
    TurnEnded {
        turn: TurnId,
        status: TurnStatus,
    },
}

impl ConversationEventKind {
    /// Delivered at once, without waiting for the frame gap (FG3).
    pub fn is_urgent(&self) -> bool {
        matches!(
            self,
            Self::ApprovalRequested { .. }
                | Self::Question { .. }
                | Self::Withheld { .. }
                | Self::Error { .. }
                | Self::TurnSaved { .. }
                | Self::TurnEnded { .. }
        )
    }
}

/// One event of one conversation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationEvent {
    /// Starts at 1 and rises by one per conversation.
    pub seq: u64,
    /// Seconds since the Unix epoch.
    pub at: f64,
    #[serde(flatten)]
    pub kind: ConversationEventKind,
}

/// What the shell hands the page per batch (§13.5): contiguous `seq`s.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FollowBatch {
    pub conversation: ConversationId,
    pub first_seq: u64,
    pub last_seq: u64,
    pub urgent: bool,
    pub events: Vec<ConversationEvent>,
}

// ----------------------------------------------------------------------- service

/// What the interface calls. Anything that may touch the disk, a child process
/// or the network returns a future; `status`, `stop`, `cancel_queued`, `answer`
/// and `follow` touch memory only and return at once. Every call validates its
/// input and refuses with one sentence ([`Refusal`]), never with transport text.
/// There is no `delete`.
pub trait AgentChatService: Send + Sync + 'static {
    fn status(&self) -> CoreStatus;
    fn list(&self) -> BoxFuture<'static, Result<ChatList, Refusal>>;
    fn open(&self, id: &str) -> BoxFuture<'static, Result<Snapshot, Refusal>>;
    fn choices(&self) -> BoxFuture<'static, Vec<ChatChoice>>;
    /// On demand only.
    fn local_runtime(&self) -> BoxFuture<'static, LocalRuntime>;
    fn send(&self, r: SendRequest) -> BoxFuture<'static, Result<Accepted, Refusal>>;
    fn regenerate(&self, r: RegenerateRequest) -> BoxFuture<'static, Result<Accepted, Refusal>>;
    fn continue_turn(
        &self,
        id: &str,
        shown: Shown,
    ) -> BoxFuture<'static, Result<Accepted, Refusal>>;
    fn steer(&self, id: &str, text: String) -> BoxFuture<'static, Result<(), Refusal>>;
    fn stop(&self, id: &str) -> bool;
    /// End one background command of the conversation (the reader's Stop on
    /// its card); `false` when none runs by that call.
    fn stop_command(&self, id: &str, call: &str) -> bool {
        let _ = (id, call);
        false
    }
    fn cancel_queued(&self, id: &str, queued_id: &str) -> Result<(), Refusal>;
    fn decide(&self, id: &str, call: &str, d: Decision) -> BoxFuture<'static, Result<(), Refusal>>;
    fn allow_always(
        &self,
        id: &str,
        call: &str,
        scope: MatchScope,
    ) -> BoxFuture<'static, Result<AllowEntry, Refusal>>;
    fn answer(&self, id: &str, call: &str, text: String) -> Result<(), Refusal>;
    /// Events with `seq > after`, in batches per the frame-gap contract (§11.3).
    fn follow(
        &self,
        id: &str,
        after: u64,
    ) -> Result<BoxStream<'static, Vec<ConversationEvent>>, Refusal>;
    fn set_mode(&self, id: &str, mode: Mode) -> BoxFuture<'static, Result<(), Refusal>>;
    fn attach_workspace(
        &self,
        id: Option<&str>,
        path: String,
    ) -> BoxFuture<'static, Result<WorkspaceView, Refusal>>;
    fn trust(&self, workspace: &str) -> BoxFuture<'static, Result<TrustState, Refusal>>;
    fn files(
        &self,
        workspace: &str,
        query: String,
        limit: u32,
    ) -> BoxFuture<'static, Result<Vec<RankedPath>, Refusal>>;
    fn read_lines(
        &self,
        view: ViewRef,
        from: u32,
        count: u32,
    ) -> BoxFuture<'static, Result<Lines, Refusal>>;
    fn changes(&self, id: &str) -> BoxFuture<'static, Result<ChangeSet, Refusal>>;
    fn diff(&self, id: &str, change: &str) -> BoxFuture<'static, Result<FileDiff, Refusal>>;
    fn review(
        &self,
        id: &str,
        ops: Vec<ReviewOp>,
    ) -> BoxFuture<'static, Result<ReviewOutcome, Refusal>>;
    fn checkpoints(&self, id: &str) -> BoxFuture<'static, Result<Vec<CheckpointView>, Refusal>>;
    /// Stages the restore; writes nothing.
    fn restore(&self, id: &str, to: CheckpointId)
    -> BoxFuture<'static, Result<ChangeSet, Refusal>>;
    fn rename(
        &self,
        id: &str,
        title: &str,
    ) -> BoxFuture<'static, Result<ConversationSummary, Refusal>>;
    fn pin(&self, id: &str, choice: &str) -> BoxFuture<'static, Result<(), Refusal>>;
    fn archive(&self, id: &str) -> BoxFuture<'static, Result<(), Refusal>>;
    fn unarchive(
        &self,
        key: ArchiveKey,
    ) -> BoxFuture<'static, Result<ConversationSummary, Refusal>>;
    fn permissions(&self, workspace: &str) -> BoxFuture<'static, Result<Vec<AllowEntry>, Refusal>>;
    /// Narrowing: no dialog. The entry stays listed as revoked.
    fn revoke(&self, workspace: &str, entry: &str) -> BoxFuture<'static, Result<(), Refusal>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_have_their_shapes() {
        assert!(is_conversation_id("0123456789ab"));
        assert!(!is_conversation_id("a/b"));
        assert!(is_change_id("ch_0123456789abcdef"));
        for bad in [
            "ch_0123456789ABCDEF",
            "ch_0123456789abcde",
            "0123456789abcdef",
            "ch_",
        ] {
            assert!(!is_change_id(bad), "{bad}");
        }
        assert!(is_hunk_id("0123456789abcdef") && !is_hunk_id("0123456789abcdeg"));
        assert!(is_workspace_id("fedcba9876543210") && !is_workspace_id("fedcba987654321"));
        assert!(is_call_id("call_1") && is_call_id(&"x".repeat(128)));
        for bad in ["", " ", "call 1", "café", "a\nb", &"x".repeat(129)] {
            assert!(!is_call_id(bad), "{bad:?}");
        }
        assert!(is_task_id("task_0123456789abcdef"));
        for bad in [
            "task_0123456789ABCDEF",
            "task_0123456789abcde",
            "task_../../etc/passwd",
            "ch_0123456789abcdef",
            "task_",
        ] {
            assert!(!is_task_id(bad), "{bad}");
        }
        assert!(is_plan_id("plan_0123456789abcdef"));
        for bad in ["plan_0123456789ABCDEF", "task_0123456789abcdef", "plan_"] {
            assert!(!is_plan_id(bad), "{bad}");
        }
    }

    #[test]
    fn the_urgent_events_are_the_ones_fg3_names() {
        let urgent = ConversationEventKind::Error {
            message: "m".into(),
        };
        let not = ConversationEventKind::Delta { text: "t".into() };
        assert!(urgent.is_urgent());
        assert!(!not.is_urgent());
    }
}
