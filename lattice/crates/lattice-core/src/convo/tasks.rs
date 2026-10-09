//! Tasks the agent suggests: the "task chips" (2026-10-08, as Claude
//! Code has them). While it works, the agent may offer the reader a separate
//! task, something worth doing outside what was asked (a bug it noticed, stale
//! documentation, a missing test), as a chip in the conversation. Nothing runs
//! until the reader acts on it:
//!
//! - **Start** begins a new conversation in the same folder and project, its
//!   first message the task's prompt, with the model and mode the reader picks
//!   ([`start`], `AgentChat::start_task`): a send as the composer makes one, so
//!   every check a send meets applies. The prompt is the one recorded, never
//!   one the window hands back, so the prompt the chip shows is the prompt that
//!   is sent.
//! - **Dismiss** sets it aside ([`dismiss`], `AgentChat::dismiss_task`).
//! - The agent may **withdraw** one it no longer needs (`withdraw_task`),
//!   saying why.
//!
//! A task is two records in the conversation's sidecar, [`Item::TaskSuggested`]
//! and, once, [`Item::TaskSettled`] (started as which conversation, dismissed,
//! or withdrawn and why), with the events of the same names; the chip is built
//! from them, so it outlives a restart. A task settles once: a second Start, a
//! Dismiss after a Start or a withdrawal after either is refused with a
//! sentence. Settling, and counting what waits, happen under one lock of this
//! process (`Inner::tasks`), which also names the tasks being started, so a
//! Start that waits on a dialog holds nothing else up. Suggesting needs no
//! approval: it writes only the conversation's own record, and a suggestion
//! that could not be written is refused, not shown.
//!
//! **Bounds.** A title is one line of at most 80 characters; a summary one
//! paragraph of at most 300; a prompt at most 2,000 (so an event stays within
//! the protocol's 2 KiB of text) and not blank; a reason one line of at most
//! 200. At most 8 tasks wait in a conversation at once: the agent withdraws a
//! stale one before it suggests a ninth. The text is redacted before it is
//! written (T15), so the chip and the record agree, and a prompt that looked
//! like it held a secret is sent redacted.
//!
//! Not a port: the web Lattice has no tasks.

use std::sync::Arc;

use lattice_protocol::conversation::{
    Accepted, ConversationEventKind, Mode, SendRequest, TaskOutcome, is_conversation_id, is_task_id,
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
/// The longest summary, in characters.
pub const MAX_SUMMARY_CHARS: usize = 300;
/// The longest prompt, in characters.
pub const MAX_PROMPT_CHARS: usize = 2_000;
/// The longest reason for a withdrawal, in characters.
pub const MAX_REASON_CHARS: usize = 200;
/// The most tasks that wait in one conversation at once.
pub const MAX_WAITING: usize = 8;

/// The reader's sentences.
pub mod words {
    pub const NO_TASK: &str = "That task is not in this conversation.";
    pub const STARTING: &str = "That task is being started.";
    pub const STARTED: &str = "That task was started already.";
    pub const DISMISSED: &str = "That task was dismissed.";
    pub const WITHDRAWN: &str = "The agent withdrew that task.";
    pub const UNREADABLE: &str = "Lattice could not read this conversation's record.";
    pub const NOT_SAVED: &str = "Lattice could not save that, so the task still waits.";
    pub const START_NOT_SAVED: &str =
        "The task started, but Lattice could not record that here; its chip may offer it again.";
}

/// `suggest_task`'s arguments.
#[derive(Clone, Debug, Deserialize)]
pub struct SuggestArgs {
    pub title: String,
    pub summary: String,
    pub prompt: String,
}

/// `withdraw_task`'s arguments.
#[derive(Clone, Debug, Deserialize)]
pub struct WithdrawArgs {
    pub task_id: String,
    #[serde(default)]
    pub reason: Option<String>,
}

/// One task as a conversation's records state it.
#[derive(Clone, Debug, PartialEq)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub prompt: String,
    /// What became of it; `None` while it waits.
    pub outcome: Option<TaskOutcome>,
}

/// Every task the records name, in the order suggested, each with the first
/// settlement recorded for it (a later one is ignored, as is a settlement of
/// a task never suggested).
pub fn tasks(items: &[Item]) -> Vec<Task> {
    let mut out: Vec<Task> = Vec::new();
    for item in items {
        match item {
            Item::TaskSuggested {
                task,
                title,
                prompt,
                ..
            } if !out.iter().any(|known| &known.id == task) => out.push(Task {
                id: task.clone(),
                title: title.clone(),
                prompt: prompt.clone(),
                outcome: None,
            }),
            Item::TaskSettled { task, outcome, .. } => {
                if let Some(known) = out.iter_mut().find(|known| &known.id == task)
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

/// The reader's sentence for a task that has settled.
fn settled(outcome: &TaskOutcome) -> &'static str {
    match outcome {
        TaskOutcome::Started { .. } => words::STARTED,
        TaskOutcome::Dismissed => words::DISMISSED,
        TaskOutcome::Withdrawn { .. } => words::WITHDRAWN,
    }
}

/// The agent's sentence for a task that has settled.
fn settled_for_agent(task: &str, outcome: &TaskOutcome) -> String {
    match outcome {
        TaskOutcome::Started { .. } => {
            format!("The user already started {task}; it stays as it is.")
        }
        TaskOutcome::Dismissed => format!("The user dismissed {task}; it stays as it is."),
        TaskOutcome::Withdrawn { .. } => format!("{task} was withdrawn already."),
    }
}

/// The lock's name for a task being started.
fn key(conversation: &str, task: &str) -> String {
    format!("{conversation}/{task}")
}

fn read(inner: &Inner, conversation: &str) -> Option<Vec<Item>> {
    inner
        .sidecars
        .read_items(conversation)
        .ok()
        .map(|log| log.items)
}

/// Append `item` to the conversation's record; whether it was written. (Not
/// `Convo::record`, whose answer says whether it was redacted.)
fn write(convo: &Convo, item: &Item) -> bool {
    let sidecar = convo.state().sidecar.clone();
    sidecar.is_some_and(|sidecar| sidecar.append(item).is_ok())
}

/// Record `outcome` for `task` and tell the window; whether it was written.
fn settle(inner: &Inner, convo: &Convo, task: &str, outcome: TaskOutcome) -> bool {
    let written = write(
        convo,
        &Item::TaskSettled {
            task: task.to_owned(),
            outcome: outcome.clone(),
            at: inner.now(),
        },
    );
    if written {
        convo.log.push(ConversationEventKind::TaskSettled {
            task: task.to_owned(),
            outcome,
        });
    }
    written
}

fn has_control(text: &str, newlines: bool) -> bool {
    text.chars()
        .any(|c| c.is_control() && !(newlines && (c == '\n' || c == '\t')))
}

/// `suggest_task` and `withdraw_task`, on the blocking pool.
pub(crate) fn tool(
    tools: &TurnTools,
    name: &str,
    call: &str,
    args: &Value,
) -> Result<String, ToolError> {
    if name == "withdraw_task" {
        withdraw(tools, &parse_args(args)?)
    } else {
        suggest(tools, call, &parse_args(args)?)
    }
}

fn suggest(tools: &TurnTools, call: &str, args: &SuggestArgs) -> Result<String, ToolError> {
    let (title, summary, prompt) = (args.title.trim(), args.summary.trim(), args.prompt.trim());
    if title.is_empty() || title.chars().count() > MAX_TITLE_CHARS || has_control(title, false) {
        return Err(ToolError::new(
            "A task's title is one line of 1 to 80 characters.",
        ));
    }
    if summary.is_empty()
        || summary.chars().count() > MAX_SUMMARY_CHARS
        || has_control(summary, false)
    {
        return Err(ToolError::new(
            "A task's summary is one paragraph of 1 to 300 characters.",
        ));
    }
    if prompt.is_empty() || prompt.chars().count() > MAX_PROMPT_CHARS || has_control(prompt, true) {
        return Err(ToolError::new(
            "A task's prompt is 1 to 2,000 characters of text that stands on its own.",
        ));
    }
    let clean = [title, summary, prompt].map(secrets::redact);
    let redacted = clean
        .iter()
        .zip([title, summary, prompt])
        .any(|(a, b)| a != b);
    let [title, summary, prompt] = clean;
    let inner = &tools.inner;
    let convo = &tools.convo;
    let _settling = lock(&inner.tasks);
    let Some(items) = read(inner, &convo.id) else {
        return Err(ToolError::new(
            "Lattice could not read this conversation's record, so no task was suggested.",
        ));
    };
    let known = tasks(&items);
    if known.iter().filter(|task| task.outcome.is_none()).count() >= MAX_WAITING {
        return Err(ToolError::new(format!(
            "{MAX_WAITING} suggested tasks already wait for the user; withdraw one that is no longer needed first."
        )));
    }
    let task = loop {
        let id = format!("task_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]);
        if !known.iter().any(|task| task.id == id) {
            break id;
        }
    };
    let item = Item::TaskSuggested {
        turn: tools.turn.clone(),
        call_id: call.to_owned(),
        task: task.clone(),
        title: title.clone(),
        summary: summary.clone(),
        prompt: prompt.clone(),
        at: inner.now(),
    };
    if !write(convo, &item) {
        return Err(ToolError::new(
            "Lattice could not save the task, so it was not suggested.",
        ));
    }
    convo.log.push(ConversationEventKind::TaskSuggested {
        task: task.clone(),
        title,
        summary,
        prompt,
    });
    let mut said = format!(
        "Suggested as {task}. Nothing runs until the user starts it; withdraw it with withdraw_task if it is no longer needed."
    );
    if redacted {
        said.push_str(&format!(
            " Text that looked like a secret was replaced with {}.",
            secrets::REDACTED
        ));
    }
    Ok(said)
}

fn withdraw(tools: &TurnTools, args: &WithdrawArgs) -> Result<String, ToolError> {
    let task = args.task_id.trim();
    if !is_task_id(task) {
        return Err(ToolError::new(
            "That is not a task's id: use the id suggest_task returned (task_ and 16 letters and digits).",
        ));
    }
    let reason = args.reason.as_deref().unwrap_or("").trim();
    if reason.chars().count() > MAX_REASON_CHARS || has_control(reason, false) {
        return Err(ToolError::new(
            "A reason is one line of at most 200 characters.",
        ));
    }
    let reason = secrets::redact(reason);
    let inner = &tools.inner;
    let convo = &tools.convo;
    let settling = lock(&inner.tasks);
    if settling.contains(&key(&convo.id, task)) {
        return Err(ToolError::new(format!(
            "The user is starting {task} now; it stays as it is."
        )));
    }
    let Some(items) = read(inner, &convo.id) else {
        return Err(ToolError::new(
            "Lattice could not read this conversation's record, so nothing was withdrawn.",
        ));
    };
    let Some(found) = tasks(&items).into_iter().find(|known| known.id == task) else {
        return Err(ToolError::new(format!(
            "There is no task {task} in this conversation."
        )));
    };
    if let Some(outcome) = &found.outcome {
        return Err(ToolError::new(settled_for_agent(task, outcome)));
    }
    if !settle(inner, convo, task, TaskOutcome::Withdrawn { reason }) {
        return Err(ToolError::new(
            "Lattice could not save the withdrawal, so the task still waits.",
        ));
    }
    Ok(format!("Withdrew {task}."))
}

/// A task claimed for its Start: what the send needs.
struct Claimed {
    convo: Arc<Convo>,
    prompt: String,
    workspace: Option<String>,
    project: Option<String>,
}

/// The task's claim, given back when its Start ends however it ends.
struct Claim {
    inner: Arc<Inner>,
    key: String,
}

impl Drop for Claim {
    fn drop(&mut self) {
        lock(&self.inner.tasks).remove(&self.key);
    }
}

fn check_ids(conversation: &str, task: &str) -> Result<(), Refusal> {
    if !is_conversation_id(conversation) {
        return Err(refuse(RefusalKind::NotFound, refusals::GONE));
    }
    if !is_task_id(task) {
        return Err(refuse(RefusalKind::NotFound, words::NO_TASK));
    }
    Ok(())
}

/// The task as it waits, with the lock held: refused when it is being started,
/// is not there, or has settled.
fn waiting(
    inner: &Inner,
    settling: &std::collections::HashSet<String>,
    conversation: &str,
    task: &str,
) -> Result<Task, Refusal> {
    if settling.contains(&key(conversation, task)) {
        return Err(refuse(RefusalKind::Conflict, words::STARTING));
    }
    let items = read(inner, conversation)
        .ok_or_else(|| refuse(RefusalKind::Unavailable, words::UNREADABLE))?;
    let found = tasks(&items)
        .into_iter()
        .find(|known| known.id == task)
        .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_TASK))?;
    match &found.outcome {
        Some(outcome) => Err(refuse(RefusalKind::Conflict, settled(outcome))),
        None => Ok(found),
    }
}

fn claim(inner: &Arc<Inner>, conversation: &str, task: &str) -> Result<(Claimed, Claim), Refusal> {
    let convo = inner.load(conversation)?;
    let found = {
        let mut settling = lock(&inner.tasks);
        let found = waiting(inner, &settling, conversation, task)?;
        settling.insert(key(conversation, task));
        found
    };
    let held = Claim {
        inner: inner.clone(),
        key: key(conversation, task),
    };
    let workspace = convo.state().workspace.as_ref().map(|w| w.id.clone());
    let project = inner.projects.of_chat(conversation).map(|p| p.id);
    Ok((
        Claimed {
            convo,
            prompt: found.prompt,
            workspace,
            project,
        },
        held,
    ))
}

/// Start `task` of `conversation`: a send of its recorded prompt to a new
/// conversation in the same folder and project, with `choice` in `mode`; the
/// task is then settled as started.
pub(crate) async fn start(
    inner: Arc<Inner>,
    conversation: String,
    task: String,
    choice: String,
    shown: Shown,
    mode: Mode,
) -> Result<Accepted, Refusal> {
    check_ids(&conversation, &task)?;
    let (claimed, held) = {
        let inner = inner.clone();
        let (conversation, task) = (conversation.clone(), task.clone());
        inner
            .handle
            .clone()
            .spawn_blocking(move || claim(&inner, &conversation, &task))
            .await
            .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))?
    };
    let request = SendRequest {
        conversation: None,
        text: claimed.prompt,
        choice,
        shown,
        mode,
        workspace: claimed.workspace,
        edit_of: None,
        project: claimed.project,
        images: Vec::new(),
    };
    validate_send(&request, inner.config.development)?;
    let accepted = super::agent::start(inner.clone(), Ask::Send(request)).await?;
    if let Accepted::Started {
        conversation: started,
        ..
    } = &accepted
    {
        let outcome = TaskOutcome::Started {
            conversation: started.id.clone(),
        };
        let convo = claimed.convo;
        let written = {
            let _settling = lock(&inner.tasks);
            settle(&inner, &convo, &task, outcome.clone())
        };
        if !written {
            // Said where the chip is, so the reader knows why it still offers a Start.
            convo
                .log
                .push(ConversationEventKind::TaskSettled { task, outcome });
            convo.log.push(ConversationEventKind::Notice {
                text: words::START_NOT_SAVED.to_owned(),
            });
        }
    }
    drop(held);
    Ok(accepted)
}

/// Set `task` of `conversation` aside.
pub(crate) fn dismiss(inner: &Arc<Inner>, conversation: &str, task: &str) -> Result<(), Refusal> {
    check_ids(conversation, task)?;
    let convo = inner.load(conversation)?;
    let settling = lock(&inner.tasks);
    waiting(inner, &settling, conversation, task)?;
    if !settle(inner, &convo, task, TaskOutcome::Dismissed) {
        return Err(refuse(RefusalKind::Unavailable, words::NOT_SAVED));
    }
    Ok(())
}
