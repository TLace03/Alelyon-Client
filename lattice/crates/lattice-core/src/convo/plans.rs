//! Plan mode: the agent, in Ask mode, proposes a plan for the work the reader
//! asked for, and the reader approves it before anything changes (the
//! tool-parity direction of 2026-10-08, as Claude Code and Cursor plan).
//!
//! - `propose_plan` (`title`, `plan`) is offered in Ask mode with a folder
//!   only: Agent mode acts on its own, and with no folder there is nothing to
//!   carry a plan out in. It writes only the conversation's own record, with
//!   no approval, and the agent stops there.
//! - **Approve** (`AgentChat::approve_plan`) goes on in the same conversation
//!   in Agent mode: a send as the composer makes one, its words naming the
//!   plan, so every check a send meets applies (an untrusted folder, a model
//!   off this PC). **Keep planning** (`AgentChat::keep_planning`) sets the
//!   plan aside for the reader's changes; a new proposal replaces the one that
//!   waits.
//!
//! A plan is two records, [`Item::PlanProposed`] and, once,
//! [`Item::PlanSettled`] (approved, kept planning, or replaced), with the
//! events of the same names, so its card outlives a restart. Settling happens
//! under one lock of this process (`Inner::plans`), which also names the plans
//! being approved. **Bounds:** a title is one line of at most 80 characters, a
//! plan at most 2,000 characters (an event's text stays within 2 KiB; a longer
//! one belongs in an artifact). The text is redacted before it is written.
//!
//! Not a port: the web Lattice has no plan mode.

use std::sync::Arc;

use lattice_protocol::conversation::{
    Accepted, ConversationEventKind, Mode, PlanOutcome, SendRequest, is_conversation_id, is_plan_id,
};
use lattice_protocol::{Refusal, RefusalKind, Shown};
use serde::Deserialize;
use serde_json::Value;

use super::agent::{Ask, Convo, Inner, lock, refuse, validate_send};
use super::item::Item;
use super::turn::TurnTools;
use crate::chat::refusals;
use crate::secrets;
use crate::tools::read::{ToolError, parse_args};

/// The longest title, in characters.
pub const MAX_TITLE_CHARS: usize = 80;
/// The longest plan, in characters.
pub const MAX_PLAN_CHARS: usize = 2_000;

/// The reader's sentences.
pub mod words {
    pub const NO_PLAN: &str = "That plan is not in this conversation.";
    pub const APPROVING: &str = "That plan is being approved.";
    pub const APPROVED: &str = "That plan was approved already.";
    pub const KEPT: &str = "That plan was set aside for more planning.";
    pub const REPLACED: &str = "The agent proposed another plan in its place.";
    pub const UNREADABLE: &str = "Lattice could not read this conversation's record.";
    pub const NOT_SAVED: &str = "Lattice could not save that, so the plan still waits.";
    pub const APPROVE_NOT_SAVED: &str =
        "The plan went ahead, but Lattice could not record that here; its card may offer it again.";
}

/// `propose_plan`'s arguments.
#[derive(Clone, Debug, Deserialize)]
pub struct ProposeArgs {
    pub title: String,
    pub plan: String,
}

/// One plan as a conversation's records state it.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub id: String,
    pub title: String,
    pub text: String,
    /// What became of it; `None` while it waits.
    pub outcome: Option<PlanOutcome>,
}

/// Every plan the records name, in the order proposed, each with the first
/// settlement recorded for it.
pub fn plans(items: &[Item]) -> Vec<Plan> {
    let mut out: Vec<Plan> = Vec::new();
    for item in items {
        match item {
            Item::PlanProposed {
                plan, title, text, ..
            } if !out.iter().any(|known| &known.id == plan) => out.push(Plan {
                id: plan.clone(),
                title: title.clone(),
                text: text.clone(),
                outcome: None,
            }),
            Item::PlanSettled { plan, outcome, .. } => {
                if let Some(known) = out.iter_mut().find(|known| &known.id == plan)
                    && known.outcome.is_none()
                {
                    known.outcome = Some(outcome.clone());
                }
            }
            _ => {}
        }
    }
    out
}

/// The words of the message that approves `title`.
pub fn approval(title: &str) -> String {
    format!("Go ahead with the plan: {title}.")
}

fn settled(outcome: &PlanOutcome) -> &'static str {
    match outcome {
        PlanOutcome::Approved => words::APPROVED,
        PlanOutcome::KeptPlanning => words::KEPT,
        PlanOutcome::Replaced => words::REPLACED,
    }
}

fn key(conversation: &str, plan: &str) -> String {
    format!("{conversation}/{plan}")
}

fn read(inner: &Inner, conversation: &str) -> Option<Vec<Item>> {
    inner
        .sidecars
        .read_items(conversation)
        .ok()
        .map(|log| log.items)
}

fn write(convo: &Convo, item: &Item) -> bool {
    let sidecar = convo.state().sidecar.clone();
    sidecar.is_some_and(|sidecar| sidecar.append(item).is_ok())
}

/// Record `outcome` for `plan` and tell the window; whether it was written.
fn settle(inner: &Inner, convo: &Convo, plan: &str, outcome: PlanOutcome) -> bool {
    let written = write(
        convo,
        &Item::PlanSettled {
            plan: plan.to_owned(),
            outcome: outcome.clone(),
            at: inner.now(),
        },
    );
    if written {
        convo.log.push(ConversationEventKind::PlanSettled {
            plan: plan.to_owned(),
            outcome,
        });
    }
    written
}

/// `propose_plan`, on the blocking pool.
pub(crate) fn tool(tools: &TurnTools, call: &str, args: &Value) -> Result<String, ToolError> {
    let args: ProposeArgs = parse_args(args)?;
    let (title, text) = (args.title.trim(), args.plan.trim());
    if title.is_empty()
        || title.chars().count() > MAX_TITLE_CHARS
        || title.chars().any(char::is_control)
    {
        return Err(ToolError::new(
            "A plan's title is one line of 1 to 80 characters.",
        ));
    }
    if text.is_empty()
        || text.chars().count() > MAX_PLAN_CHARS
        || text
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\t')
    {
        return Err(ToolError::new(
            "A plan is 1 to 2,000 characters of Markdown; put a longer one in an artifact and name it here.",
        ));
    }
    let (clean_title, clean_text) = (secrets::redact(title), secrets::redact(text));
    let redacted = clean_title != title || clean_text != text;
    let inner = &tools.inner;
    let convo = &tools.convo;
    let settling = lock(&inner.plans);
    let Some(items) = read(inner, &convo.id) else {
        return Err(ToolError::new(
            "Lattice could not read this conversation's record, so no plan was proposed.",
        ));
    };
    let known = plans(&items);
    if known
        .iter()
        .any(|plan| plan.outcome.is_none() && settling.contains(&key(&convo.id, &plan.id)))
    {
        return Err(ToolError::new(
            "The user is approving your plan now; it stays as it is.",
        ));
    }
    let id = loop {
        let id = format!("plan_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]);
        if !known.iter().any(|plan| plan.id == id) {
            break id;
        }
    };
    let item = Item::PlanProposed {
        turn: tools.turn.clone(),
        call_id: call.to_owned(),
        plan: id.clone(),
        title: clean_title.clone(),
        text: clean_text.clone(),
        at: inner.now(),
    };
    if !write(convo, &item) {
        return Err(ToolError::new(
            "Lattice could not save the plan, so it was not proposed.",
        ));
    }
    // The plan that waited gives way to this one.
    for waiting in known.iter().filter(|plan| plan.outcome.is_none()) {
        settle(inner, convo, &waiting.id, PlanOutcome::Replaced);
    }
    convo.log.push(ConversationEventKind::PlanProposed {
        plan: id.clone(),
        title: clean_title,
        text: clean_text,
    });
    let mut said = format!(
        "Proposed as {id}. Stop here and wait: when the user approves it, this conversation goes on in Agent mode and you carry it out; or they ask for changes."
    );
    if redacted {
        said.push_str(&format!(
            " Text that looked like a secret was replaced with {}.",
            secrets::REDACTED
        ));
    }
    Ok(said)
}

fn check_ids(conversation: &str, plan: &str) -> Result<(), Refusal> {
    if !is_conversation_id(conversation) {
        return Err(refuse(RefusalKind::NotFound, refusals::GONE));
    }
    if !is_plan_id(plan) {
        return Err(refuse(RefusalKind::NotFound, words::NO_PLAN));
    }
    Ok(())
}

/// The plan as it waits, with the lock held.
fn waiting(
    inner: &Inner,
    settling: &std::collections::HashSet<String>,
    conversation: &str,
    plan: &str,
) -> Result<Plan, Refusal> {
    if settling.contains(&key(conversation, plan)) {
        return Err(refuse(RefusalKind::Conflict, words::APPROVING));
    }
    let items = read(inner, conversation)
        .ok_or_else(|| refuse(RefusalKind::Unavailable, words::UNREADABLE))?;
    let found = plans(&items)
        .into_iter()
        .find(|known| known.id == plan)
        .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_PLAN))?;
    match &found.outcome {
        Some(outcome) => Err(refuse(RefusalKind::Conflict, settled(outcome))),
        None => Ok(found),
    }
}

/// The plan's claim, given back when its approval ends however it ends.
struct Claim {
    inner: Arc<Inner>,
    key: String,
}

impl Drop for Claim {
    fn drop(&mut self) {
        lock(&self.inner.plans).remove(&self.key);
    }
}

struct Claimed {
    convo: Arc<Convo>,
    title: String,
    workspace: Option<String>,
    project: Option<String>,
}

fn claim(inner: &Arc<Inner>, conversation: &str, plan: &str) -> Result<(Claimed, Claim), Refusal> {
    let convo = inner.load(conversation)?;
    let found = {
        let mut settling = lock(&inner.plans);
        let found = waiting(inner, &settling, conversation, plan)?;
        settling.insert(key(conversation, plan));
        found
    };
    let held = Claim {
        inner: inner.clone(),
        key: key(conversation, plan),
    };
    let workspace = convo.state().workspace.as_ref().map(|w| w.id.clone());
    let project = inner.projects.of_chat(conversation).map(|p| p.id);
    Ok((
        Claimed {
            convo,
            title: found.title,
            workspace,
            project,
        },
        held,
    ))
}

/// Approve `plan` of `conversation`: a send in the same conversation, in Agent
/// mode, with `choice`, of the words that name it; the plan is then settled as
/// approved.
pub(crate) async fn approve(
    inner: Arc<Inner>,
    conversation: String,
    plan: String,
    choice: String,
    shown: Shown,
) -> Result<Accepted, Refusal> {
    check_ids(&conversation, &plan)?;
    let (claimed, held) = {
        let inner = inner.clone();
        let (conversation, plan) = (conversation.clone(), plan.clone());
        inner
            .handle
            .clone()
            .spawn_blocking(move || claim(&inner, &conversation, &plan))
            .await
            .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))?
    };
    let request = SendRequest {
        conversation: Some(conversation),
        text: approval(&claimed.title),
        choice,
        shown,
        mode: Mode::Agent,
        workspace: claimed.workspace,
        edit_of: None,
        project: claimed.project,
        images: Vec::new(),
    };
    validate_send(&request, inner.config.development)?;
    let accepted = super::agent::start(inner.clone(), Ask::Send(request)).await?;
    let convo = claimed.convo;
    let written = {
        let _settling = lock(&inner.plans);
        settle(&inner, &convo, &plan, PlanOutcome::Approved)
    };
    if !written {
        convo.log.push(ConversationEventKind::PlanSettled {
            plan,
            outcome: PlanOutcome::Approved,
        });
        convo.log.push(ConversationEventKind::Notice {
            text: words::APPROVE_NOT_SAVED.to_owned(),
        });
    }
    drop(held);
    Ok(accepted)
}

/// Set `plan` of `conversation` aside for more planning.
pub(crate) fn keep_planning(
    inner: &Arc<Inner>,
    conversation: &str,
    plan: &str,
) -> Result<(), Refusal> {
    check_ids(conversation, plan)?;
    let convo = inner.load(conversation)?;
    let settling = lock(&inner.plans);
    waiting(inner, &settling, conversation, plan)?;
    if !settle(inner, &convo, plan, PlanOutcome::KeptPlanning) {
        return Err(refuse(RefusalKind::Unavailable, words::NOT_SAVED));
    }
    Ok(())
}
