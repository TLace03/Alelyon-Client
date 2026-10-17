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
    pub const NO_GROUP: &str = "That group is no longer listed, so nothing was sent.";
    pub const NOT_SAVED: &str = "Lattice could not save that, so the task still waits.";
    pub const START_NOT_SAVED: &str =
        "The task started, but Lattice could not record that here; its chip may offer it again.";
}

/// `suggest_task`'s arguments, and `offer_to_group`'s with its group.
#[derive(Clone, Debug, Deserialize)]
pub struct SuggestArgs {
    pub title: String,
    pub summary: String,
    pub prompt: String,
    /// `offer_to_group`: a group's id, or `new`.
    #[serde(default)]
    pub group: Option<String>,
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
    /// An idea for a group (`offer_to_group`): the group its Start sends to.
    pub group: Option<String>,
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
                group,
                ..
            } if !out.iter().any(|known| &known.id == task) => out.push(Task {
                id: task.clone(),
                title: title.clone(),
                prompt: prompt.clone(),
                group: group.clone(),
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
    match name {
        "withdraw_task" => withdraw(tools, &parse_args(args)?),
        "offer_to_group" => {
            let args: SuggestArgs = parse_args(args)?;
            let group = args.group.as_deref().map(str::trim).unwrap_or_default();
            if group != super::groups::NEW {
                let project = tools.inner.projects.of_chat(&tools.convo.id).map(|p| p.id);
                let known = tools.inner.groups.list().into_iter().any(|g| g.id == group && !g.archived && g.project == project);
                if !known {
                    return Err(ToolError::new(
                        "That is not one of this project's groups: use a group's id as listed for you, or new.",
                    ));
                }
            }
            suggest(tools, call, &SuggestArgs { group: Some(group.to_owned()), ..args })
        }
        _ => suggest(tools, call, &SuggestArgs { group: None, ..parse_args(args)? }),
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
        group: args.group.clone(),
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
        group: args.group.clone(),
    });
    let mut said = if args.group.is_some() {
        format!("Offered as {task}. Nothing is sent until the user presses it; withdraw it with withdraw_task if it is no longer needed.")
    } else {
        format!("Suggested as {task}. Nothing runs until the user starts it; withdraw it with withdraw_task if it is no longer needed.")
    };
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
    title: String,
    /// An idea for a group: where its Start sends it.
    group: Option<String>,
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
    // The other topics working now; with the tasks being started, what the limit counts.
    let running = lock(&inner.convos)
        .iter()
        .filter(|(id, other)| id.as_str() != conversation && other.state().running.is_some())
        .count();
    let limit = inner.task_limit();
    let found = {
        let mut settling = lock(&inner.tasks);
        let found = waiting(inner, &settling, conversation, task)?;
        if !limit.allows(running + settling.len()) {
            return Err(refuse(RefusalKind::Conflict, &super::task_runs::full_of(inner)));
        }
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
            title: found.title,
            group: found.group,
            workspace,
            project,
        },
        held,
    ))
}

/// A Start past the reader's limit, waiting for a slot.
#[derive(Clone, Debug)]
pub struct Waiting {
    conversation: String,
    task: String,
    choice: String,
    shown: Shown,
    mode: Mode,
}

/// Where a waiting task stands in line: its group's place in the reader's
/// order (an idea's group; a task's, the group of the topic it came from).
fn rank_of(inner: &Inner, waiting: &Waiting, ranks: &std::collections::HashMap<String, usize>) -> usize {
    let idea = read(inner, &waiting.conversation)
        .map(|items| tasks(&items))
        .and_then(|known| known.into_iter().find(|t| t.id == waiting.task))
        .and_then(|t| t.group);
    let group = idea.or_else(|| inner.groups.of(&waiting.conversation).map(|g| g.id));
    group.and_then(|g| ranks.get(&g).copied()).unwrap_or(usize::MAX)
}

/// A slot freed: start the waiting tasks while the limit allows, the
/// highest-priority group's first (in the order they waited within one
/// group). One that cannot start any more is dropped, and its chip offers its
/// Start again.
pub(crate) fn start_next(inner: Arc<Inner>) -> futures::future::BoxFuture<'static, ()> {
    use futures::FutureExt;
    async move {
        let waiting = lock(&inner.task_queue).len();
        for _ in 0..waiting {
            let next = {
                let ranks = inner.groups.ranks();
                let mut queue = lock(&inner.task_queue);
                let Some(at) = (0..queue.len()).min_by_key(|i| (rank_of(&inner, &queue[*i], &ranks), *i)) else { return };
                queue.remove(at)
            };
            let full = next.clone();
            match start(inner.clone(), next.conversation, next.task, next.choice, next.shown, next.mode).await {
                // Still no slot: it waits again, where it was in line.
                Err(refusal) if refusal.message == super::task_runs::full_of(&inner) => {
                    let mut queue = lock(&inner.task_queue);
                    queue.retain(|w| !(w.conversation == full.conversation && w.task == full.task));
                    queue.insert(0, full);
                    return;
                }
                _ => {}
            }
        }
    }
    .boxed()
}

/// Start `task` of `conversation`: a send of its recorded prompt to a new
/// conversation in the same folder and project, with `choice` in `mode`; the
/// task is then settled as started. Past the reader's limit it waits for a
/// slot ([`start_next`]) and the Start is answered with the sentence that
/// says so.
pub(crate) async fn start(
    inner: Arc<Inner>,
    conversation: String,
    task: String,
    choice: String,
    shown: Shown,
    mode: Mode,
) -> Result<Accepted, Refusal> {
    check_ids(&conversation, &task)?;
    let claimed = {
        let inner = inner.clone();
        let (conversation, task) = (conversation.clone(), task.clone());
        inner
            .handle
            .clone()
            .spawn_blocking(move || claim(&inner, &conversation, &task))
            .await
            .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))
    };
    let (claimed, held) = match claimed {
        Err(refusal) if refusal.message == super::task_runs::full_of(&inner) => {
            let mut queue = lock(&inner.task_queue);
            if !queue.iter().any(|w| w.conversation == conversation && w.task == task) {
                queue.push(Waiting { conversation, task, choice, shown, mode });
            }
            return Err(refusal);
        }
        other => other?,
    };
    // An idea for a group goes to that group's session: its newest topic, or a new topic in it.
    let target = match claimed.group.as_deref() {
        None => None,
        Some(group) => Some(group_target(&inner, group, claimed.project.as_deref())?),
    };
    let request = SendRequest {
        conversation: target.as_ref().and_then(|t| t.topic.clone()),
        text: claimed.prompt,
        choice,
        shown,
        mode,
        workspace: target.as_ref().and_then(|t| t.workspace.clone()).or(claimed.workspace),
        edit_of: None,
        project: if target.as_ref().is_some_and(|t| t.topic.is_some()) { None } else { claimed.project.clone() },
        images: Vec::new(),
    };
    validate_send(&request, inner.config.development)?;
    let accepted = super::agent::start(inner.clone(), Ask::Send(request)).await?;
    // Sent to a group: it is in that group, and it is no task. A busy session takes it in turn.
    if let Some(target) = &target {
        let sent = match &accepted {
            Accepted::Started { conversation, .. } => Some(conversation.id.clone()),
            Accepted::Queued { conversation, .. } => Some(conversation.clone()),
        };
        if let Some(sent) = sent {
            let _ = inner.groups.add(&sent, &target.placement, claimed.project.as_deref(), inner.now());
            let outcome = TaskOutcome::Started { conversation: sent };
            let convo = claimed.convo;
            let written = {
                let _settling = lock(&inner.tasks);
                settle(&inner, &convo, &task, outcome.clone())
            };
            if !written {
                convo.log.push(ConversationEventKind::TaskSettled { task, outcome });
                convo.log.push(ConversationEventKind::Notice { text: words::START_NOT_SAVED.to_owned() });
            }
        }
        drop(held);
        return Ok(accepted);
    }
    if let Accepted::Started {
        conversation: started,
        ..
    } = &accepted
    {
        let outcome = TaskOutcome::Started {
            conversation: started.id.clone(),
        };
        // Where it came from, for the activity chart; a failure costs only the chart's line.
        let _ = inner.task_runs.started(
            &started.id,
            super::task_runs::Origin { from: conversation.clone(), title: claimed.title.clone(), at: inner.now(), ..Default::default() },
        );
        // It joins the group of the topic it was started from.
        let _ = inner.groups.follow(&started.id, &conversation, inner.now());
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

/// Where an idea for a group is sent.
struct GroupTarget {
    /// The group's newest topic (`None`: a new topic).
    topic: Option<String>,
    /// That topic's folder.
    workspace: Option<String>,
    placement: super::groups::Placement,
}

fn group_target(inner: &Arc<Inner>, group: &str, project: Option<&str>) -> Result<GroupTarget, Refusal> {
    use super::groups::Placement;
    if group == super::groups::NEW {
        return Ok(GroupTarget { topic: None, workspace: None, placement: Placement::New });
    }
    let found = inner
        .groups
        .list()
        .into_iter()
        .find(|g| g.id == group && !g.archived && g.project.as_deref() == project)
        .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_GROUP))?;
    let topic = super::groups::newest_topic(inner, &found);
    let workspace = topic
        .as_deref()
        .and_then(|t| inner.load(t).ok())
        .and_then(|convo| convo.state().workspace.as_ref().map(|w| w.id.clone()));
    Ok(GroupTarget { topic, workspace, placement: Placement::Group(found.id) })
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
