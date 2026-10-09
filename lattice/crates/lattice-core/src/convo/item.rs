//! What a sidecar records, one [`Item`] per line of `items.jsonl`
//! (the chat core's spec §5.7.1). Not a port: a native-only format.
//!
//! Each item is a JSON object with a snake_case `type` and the fields the
//! specification lists for it. The log is append-only: a correction is a new
//! item (`superseded`, `reviewed`), never a rewritten line.
//!
//! The types the interface also sees (modes, decisions, review operations and
//! results, checkpoints, changed paths) are the protocol's; the ones only this
//! record needs (a payload kept inline or as a blob, a staged change's full
//! state with the edits a Keep re-applies) are here.

use lattice_protocol::conversation::{
    ApprovalDetail, ApprovalKind, ArtifactKind, CallId, ChangeId, ChangeKind, ChangeOrigin,
    ChangeState, ChangedPath, CheckpointId, CheckpointKind, CheckpointReason, DecidedBy, Decision,
    DequeueOutcome, Mode, PlanId, PlanOutcome, ReviewOp, ReviewResult, TaskId, TaskOutcome,
    TodoItem, TurnKind, TurnStatus,
};
use lattice_protocol::{Locality, Shown, TurnId};
use serde::{Deserialize, Serialize};

use crate::fsx::MoveReason;

/// The most bytes a payload keeps inline; a longer one goes to a blob.
pub const INLINE_LIMIT: usize = 16 * 1024;

/// A tool's arguments or output: inline, or in `blobs/<sha256>` with a preview.
/// Serialised as a plain string or as an object.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Payload {
    Inline(String),
    Blob {
        sha256: String,
        bytes: u64,
        preview: String,
    },
}

/// Line ends of a file's base text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Eol {
    Lf,
    Crlf,
    Mixed,
    /// No line end at all (an empty or one-line file).
    None,
}

/// A staged change's file as it was when first staged.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BaseState {
    Absent,
    Present {
        sha256: String,
        bytes: u64,
        eol: Eol,
        bom: bool,
    },
}

/// What a Keep would leave at the path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NewState {
    /// The bytes in `blobs/<blob>`.
    Bytes {
        blob: String,
    },
    Deleted,
}

/// One `edit_file` call, kept so a change can be applied again to a file that
/// moved on disk.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EditOp {
    pub old_string: String,
    pub new_string: String,
    pub replace_all: bool,
}

/// A staged change in full (spec §7.4.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagedChange {
    pub id: ChangeId,
    /// Workspace-relative, forward slashes.
    pub path: String,
    pub kind: ChangeKind,
    pub base: BaseState,
    pub new: NewState,
    pub ops: Vec<EditOp>,
    pub authority: bool,
    pub origin: ChangeOrigin,
    pub state: ChangeState,
}

/// One record of a conversation's native history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Item {
    TurnStart {
        turn: TurnId,
        mode: Mode,
        kind: TurnKind,
        choice: String,
        shown: Shown,
        resolved: Locality,
        label: String,
        at: f64,
    },
    ToolCall {
        turn: TurnId,
        call_id: CallId,
        tool: String,
        arguments: Payload,
        summary: String,
        at: f64,
    },
    ToolResult {
        turn: TurnId,
        call_id: CallId,
        output: Payload,
        withheld: bool,
        truncated: bool,
        at: f64,
        /// A command's whole kept output (`blobs/<sha256>`, redacted), read
        /// by window; `output` is what the model read (row E11).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_blob: Option<String>,
    },
    /// The reasoning a model emitted in an agent turn (spec §22.8 RP3): a
    /// `<think>` span of its text, or a provider's separate reasoning
    /// content, in the order it arrived among the turn's tool calls.
    /// Stripped from the visible answer and never shown; redacted like
    /// every record (T15); replayed with the turn's assistant output. A new
    /// kind: earlier records read as before, and an earlier reader skips
    /// the line as one it cannot parse.
    Reasoning {
        turn: TurnId,
        text: Payload,
        at: f64,
    },
    /// The managed server's tool-call probe (spec §22 LR8′), sent at this
    /// turn's start: its key (binary and model), its request and reply, and
    /// whether it passed. A new kind: an earlier reader skips the line.
    ToolProbe {
        turn: TurnId,
        key: String,
        request: Payload,
        reply: Payload,
        passed: bool,
        at: f64,
    },
    ApprovalRequested {
        turn: TurnId,
        call_id: CallId,
        kind: ApprovalKind,
        detail: ApprovalDetail,
        at: f64,
    },
    ApprovalDecided {
        turn: TurnId,
        call_id: CallId,
        decision: Decision,
        by: DecidedBy,
        at: f64,
    },
    Question {
        turn: TurnId,
        call_id: CallId,
        text: String,
        at: f64,
    },
    Answer {
        turn: TurnId,
        call_id: CallId,
        text: String,
        at: f64,
    },
    /// A task the agent suggested for a conversation of its own (a chip,
    /// [`super::tasks`]); nothing runs until the reader starts it. A new
    /// kind: an earlier reader skips the line.
    TaskSuggested {
        turn: TurnId,
        call_id: CallId,
        task: TaskId,
        title: String,
        summary: String,
        prompt: String,
        at: f64,
    },
    /// What became of a suggested task: written once per task.
    TaskSettled {
        task: TaskId,
        outcome: TaskOutcome,
        at: f64,
    },
    /// A plan the agent proposed in Ask mode ([`super::plans`]). A new kind:
    /// an earlier reader skips the line.
    PlanProposed {
        turn: TurnId,
        call_id: CallId,
        plan: PlanId,
        title: String,
        text: String,
        at: f64,
    },
    /// The images the reader attached to the user turn `turn`
    /// ([`super::images`]): each a blob written as given. A new kind: an
    /// earlier reader skips the line.
    ImagesAttached {
        turn: TurnId,
        images: Vec<super::images::StoredImage>,
        at: f64,
    },
    /// What became of a proposed plan: written once per plan.
    PlanSettled {
        plan: PlanId,
        outcome: PlanOutcome,
        at: f64,
    },
    /// A version of an artifact the agent saved (a document beside the
    /// chat, [`super::artifacts`]), its text kept as any payload is. A new
    /// kind: an earlier reader skips the line.
    ArtifactSaved {
        turn: TurnId,
        call_id: CallId,
        name: String,
        title: String,
        kind: ArtifactKind,
        version: u32,
        text: Payload,
        bytes: u64,
        at: f64,
    },
    /// The agent's to-do list as it now stands ([`super::todos`]); the latest
    /// one is the list. A new kind: an earlier reader skips the line.
    TodosUpdated {
        turn: TurnId,
        call_id: CallId,
        items: Vec<TodoItem>,
        at: f64,
    },
    Staged {
        turn: TurnId,
        call_id: CallId,
        change: Box<StagedChange>,
    },
    Reviewed {
        change: ChangeId,
        op: ReviewOp,
        result: ReviewResult,
        at: f64,
        /// KP1: the blob holding the file's bytes as the Keep found them,
        /// copied before anything was written (§7.5 step 3).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base_copy: Option<String>,
    },
    Checkpoint {
        id: CheckpointId,
        kind: CheckpointKind,
        reason: CheckpointReason,
        exposed: bool,
        omitted: u32,
        at: f64,
        /// Git: the checkpoint's commit (`refs/lattice/chat/<id>/<n>`, or the
        /// previous checkpoint's when the tree was unchanged) and its tree.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        commit: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tree: Option<String>,
        /// The blob naming every omitted path (§8.2 step 1), when any was.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        omitted_list: Option<String>,
        /// Without git: bytes of the copies this checkpoint holds.
        #[serde(default)]
        bytes: u64,
    },
    CommandEffect {
        call_id: CallId,
        before: CheckpointId,
        after: CheckpointId,
        files: Vec<ChangedPath>,
    },
    MovedAside {
        path: String,
        to: String,
        sha256: String,
        why: MoveReason,
        at: f64,
    },
    ModeSwitch {
        mode: Mode,
        at: f64,
    },
    ModelSwitch {
        choice: String,
        label: String,
        locality: Locality,
        at: f64,
    },
    Steered {
        turn: TurnId,
        text: String,
        at: f64,
    },
    Queued {
        queued_id: String,
        text: String,
        shown: Shown,
        at: f64,
    },
    Dequeued {
        queued_id: String,
        outcome: DequeueOutcome,
        at: f64,
    },
    Superseded {
        turns: Vec<TurnId>,
        at: f64,
    },
    TurnEnd {
        turn: TurnId,
        answer: Option<TurnId>,
        status: TurnStatus,
        run_id: String,
        at: f64,
    },
    Notice {
        text: String,
        at: f64,
    },
}

impl Item {
    /// The turn an item belongs to, when it belongs to one.
    pub fn turn(&self) -> Option<&TurnId> {
        match self {
            Item::TurnStart { turn, .. }
            | Item::ToolCall { turn, .. }
            | Item::ToolResult { turn, .. }
            | Item::Reasoning { turn, .. }
            | Item::ToolProbe { turn, .. }
            | Item::ApprovalRequested { turn, .. }
            | Item::ApprovalDecided { turn, .. }
            | Item::Question { turn, .. }
            | Item::Answer { turn, .. }
            | Item::TaskSuggested { turn, .. }
            | Item::ArtifactSaved { turn, .. }
            | Item::TodosUpdated { turn, .. }
            | Item::PlanProposed { turn, .. }
            | Item::ImagesAttached { turn, .. }
            | Item::Staged { turn, .. }
            | Item::Steered { turn, .. }
            | Item::TurnEnd { turn, .. } => Some(turn),
            Item::Reviewed { .. }
            | Item::TaskSettled { .. }
            | Item::PlanSettled { .. }
            | Item::Checkpoint { .. }
            | Item::CommandEffect { .. }
            | Item::MovedAside { .. }
            | Item::ModeSwitch { .. }
            | Item::ModelSwitch { .. }
            | Item::Queued { .. }
            | Item::Dequeued { .. }
            | Item::Superseded { .. }
            | Item::Notice { .. } => None,
        }
    }

    /// Whether the record is written as given, without redaction. Only a staged
    /// change: its edits are what a Keep applies to the reader's file, and a
    /// redacted edit would write the redaction marker into it (spec §5.7.1:
    /// staged changes are the exception to T15).
    pub fn is_written_verbatim(&self) -> bool {
        matches!(self, Item::Staged { .. })
    }
}
