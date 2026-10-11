//! Steps for the reader: when the reader has something to do (sign in
//! somewhere, restart a program, set a value), the agent gives it as a
//! checklist rather than as directions that scroll away, and both sides see
//! where it stands.
//!
//! - `give_steps` (`title`, `steps`, `notify_when_done`) shows the reader a
//!   checklist, in the chat and pinned above the composer while it is open.
//!   `read_steps` reads every list of the conversation with what is checked.
//!   Both are offered in both modes, with or without a folder, with no
//!   approval: they write only the conversation's own record.
//! - **The reader** checks and unchecks steps (`AgentChat::check_step`).
//!   Each check is told to a running turn as a note at its next step, and is
//!   there for `read_steps` at any time; it starts nothing.
//! - **Done:** when the last step is checked, the agent is told, as a message
//!   sent in the conversation the way the composer sends one ("All steps of …
//!   are done"), which starts a turn if none runs (or is queued behind the one
//!   that does). A list's switch (`AgentChat::set_steps_notify`) turns that
//!   off; the reader then sends it with Done (`AgentChat::steps_done`), at any
//!   point. A list tells it done once.
//! - **The record:** [`Item::StepsGiven`], [`Item::StepChecked`],
//!   [`Item::StepsNotify`] and [`Item::StepsDone`], with the events of the same
//!   names, so a reopened conversation shows each list as it stands.
//! - **Bounds:** at most [`MAX_STEPS`] steps of one line of at most
//!   [`MAX_STEP_CHARS`] characters, a title of at most [`MAX_TITLE_CHARS`], and
//!   [`MAX_LISTS`] lists a conversation. The text is redacted before it is
//!   written.

use std::sync::{Arc, Mutex};

use lattice_protocol::conversation::{
    Accepted, ConversationEventKind, SendRequest, is_conversation_id, is_steps_id,
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

/// The most steps a list has.
pub const MAX_STEPS: usize = 20;
/// The longest step, in characters.
pub const MAX_STEP_CHARS: usize = 200;
/// The longest title, in characters.
pub const MAX_TITLE_CHARS: usize = 80;
/// The most lists a conversation keeps.
pub const MAX_LISTS: usize = 20;

/// Held while a list is checked or told done.
static SETTLING: Mutex<()> = Mutex::new(());

/// The reader's sentences.
pub mod words {
    pub const NO_LIST: &str = "That checklist is not in this conversation.";
    pub const NO_STEP: &str = "That step is not on the checklist.";
    pub const UNREADABLE: &str = "Lattice could not read this conversation's record.";
    pub const NOT_SAVED: &str = "Lattice could not save that, so the checklist is as it was.";
    pub const TOLD: &str = "The agent was told this checklist is done already.";
}

/// What the agent is told when a list is done.
pub fn done_message(title: &str, checked: usize, total: usize) -> String {
    if checked == total {
        format!("All steps of the checklist \u{201c}{title}\u{201d} are done.")
    } else {
        format!("I am done with the checklist \u{201c}{title}\u{201d} ({checked} of {total} steps checked).")
    }
}

/// One list as the records leave it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepList {
    pub id: String,
    pub title: String,
    pub steps: Vec<(String, bool)>,
    pub notify: bool,
    pub told: bool,
}

impl StepList {
    pub fn checked(&self) -> usize {
        self.steps.iter().filter(|(_, done)| *done).count()
    }

    pub fn complete(&self) -> bool {
        !self.steps.is_empty() && self.checked() == self.steps.len()
    }
}

/// Every list the records hold, oldest first.
pub fn lists(items: &[Item]) -> Vec<StepList> {
    let mut out: Vec<StepList> = Vec::new();
    for item in items {
        match item {
            Item::StepsGiven { list, title, steps, notify, .. } => out.push(StepList {
                id: list.clone(),
                title: title.clone(),
                steps: steps.iter().map(|s| (s.clone(), false)).collect(),
                notify: *notify,
                told: false,
            }),
            Item::StepChecked { list, step, done, .. } => {
                if let Some(found) = out.iter_mut().find(|l| &l.id == list)
                    && let Some(entry) = found.steps.get_mut(*step as usize)
                {
                    entry.1 = *done;
                }
            }
            Item::StepsNotify { list, notify, .. } => {
                if let Some(found) = out.iter_mut().find(|l| &l.id == list) {
                    found.notify = *notify;
                }
            }
            Item::StepsDone { list, .. } => {
                if let Some(found) = out.iter_mut().find(|l| &l.id == list) {
                    found.told = true;
                }
            }
            _ => {}
        }
    }
    out
}

/// A list as the agent reads it.
pub fn shown(list: &StepList) -> String {
    let mut out = format!(
        "{} ({}): {} of {} done{}",
        list.title,
        list.id,
        list.checked(),
        list.steps.len(),
        if list.told { ", and the user said it is done" } else { "" }
    );
    for (n, (text, done)) in list.steps.iter().enumerate() {
        out.push_str(&format!("\n  {}. [{}] {text}", n + 1, if *done { "x" } else { " " }));
    }
    out
}

/// `give_steps`'s arguments.
#[derive(Clone, Debug, Deserialize)]
pub struct GiveArgs {
    pub title: String,
    pub steps: Vec<String>,
    #[serde(default = "yes")]
    pub notify_when_done: bool,
}

fn yes() -> bool {
    true
}

fn one_line(text: &str, max: usize) -> Option<String> {
    let text = text.trim();
    (!text.is_empty() && text.chars().count() <= max && !text.chars().any(char::is_control))
        .then(|| secrets::redact(text))
}

fn read(inner: &Inner, conversation: &str) -> Result<Vec<Item>, Refusal> {
    inner
        .sidecars
        .read_items(conversation)
        .map(|log| log.items)
        .map_err(|_| refuse(RefusalKind::Unavailable, words::UNREADABLE))
}

fn write(convo: &Convo, item: &Item) -> bool {
    let sidecar = convo.state().sidecar.clone();
    sidecar.is_some_and(|sidecar| sidecar.append(item).is_ok())
}

/// `give_steps` and `read_steps`, on the blocking pool.
pub(crate) fn tool(tools: &TurnTools, name: &str, call: &str, args: &Value) -> Result<String, ToolError> {
    let convo = &tools.convo;
    let sidecar = convo
        .state()
        .sidecar
        .clone()
        .ok_or_else(|| ToolError::new("This conversation has no record yet."))?;
    let items = tools
        .inner
        .sidecars
        .read_items(&convo.id)
        .map(|log| log.items)
        .map_err(|_| ToolError::new("Lattice could not read this conversation's record."))?;
    let known = lists(&items);
    if name == "read_steps" {
        if known.is_empty() {
            return Ok("You have given the user no checklist in this conversation.".to_owned());
        }
        return Ok(known.iter().map(shown).collect::<Vec<_>>().join("\n\n"));
    }
    let args: GiveArgs = parse_args(args)?;
    let title = one_line(&args.title, MAX_TITLE_CHARS)
        .ok_or_else(|| ToolError::new(format!("The title is one line of 1 to {MAX_TITLE_CHARS} characters.")))?;
    if args.steps.is_empty() || args.steps.len() > MAX_STEPS {
        return Err(ToolError::new(format!("A checklist has 1 to {MAX_STEPS} steps.")));
    }
    let mut steps = Vec::with_capacity(args.steps.len());
    for step in &args.steps {
        steps.push(
            one_line(step, MAX_STEP_CHARS)
                .ok_or_else(|| ToolError::new(format!("Each step is one line of 1 to {MAX_STEP_CHARS} characters.")))?,
        );
    }
    if known.len() >= MAX_LISTS {
        return Err(ToolError::new(format!("This conversation has {MAX_LISTS} checklists already.")));
    }
    let list = format!("steps_{}", &uuid::Uuid::new_v4().simple().to_string()[..16]);
    let item = Item::StepsGiven {
        turn: tools.turn.clone(),
        call_id: call.to_owned(),
        list: list.clone(),
        title: title.clone(),
        steps: steps.clone(),
        notify: args.notify_when_done,
        at: tools.inner.now(),
    };
    if sidecar.append(&item).is_err() {
        return Err(ToolError::new("Lattice could not save the checklist, so the user does not see it."));
    }
    let n = steps.len();
    convo.log.push(ConversationEventKind::StepsGiven { list: list.clone(), title, steps, notify: args.notify_when_done });
    Ok(format!(
        "The user sees the checklist ({list}, {n} step{}) in the chat and above the composer, and checks each step as it is done. read_steps shows where it stands; you are told as you work, and {}",
        if n == 1 { "" } else { "s" },
        if args.notify_when_done {
            "a message tells you when every step is done."
        } else {
            "the user tells you when it is done."
        }
    ))
}

fn check_ids(conversation: &str, list: &str) -> Result<(), Refusal> {
    if !is_conversation_id(conversation) {
        return Err(refuse(RefusalKind::NotFound, refusals::GONE));
    }
    if !is_steps_id(list) {
        return Err(refuse(RefusalKind::NotFound, words::NO_LIST));
    }
    Ok(())
}

/// The list `list` of `conversation`, with the lock held.
fn found(inner: &Arc<Inner>, conversation: &str, list: &str) -> Result<(Arc<Convo>, StepList), Refusal> {
    check_ids(conversation, list)?;
    let convo = inner.load(conversation)?;
    let all = lists(&read(inner, conversation)?);
    let found = all.into_iter().find(|l| l.id == list).ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_LIST))?;
    Ok((convo, found))
}

/// The reader checked (or unchecked) step `step` (from 0) of `list`. Returns
/// the list's done message when it is now done and tells the agent by itself.
pub(crate) fn check(inner: &Arc<Inner>, conversation: &str, list: &str, step: u32, done: bool) -> Result<Option<String>, Refusal> {
    let _held = lock(&SETTLING);
    let (convo, mut found) = found(inner, conversation, list)?;
    let Some(text) = found.steps.get(step as usize).map(|(text, _)| text.clone()) else {
        return Err(refuse(RefusalKind::NotFound, words::NO_STEP));
    };
    if !write(&convo, &Item::StepChecked { list: list.to_owned(), step, done, at: inner.now() }) {
        return Err(refuse(RefusalKind::Unavailable, words::NOT_SAVED));
    }
    convo.log.push(ConversationEventKind::StepChecked { list: list.to_owned(), step, done });
    found.steps[step as usize].1 = done;
    // A running turn hears it at its next step.
    let note = format!(
        "The user {} step {} of the checklist \u{201c}{}\u{201d} ({}): {} of {} done.",
        if done { "checked" } else { "unchecked" },
        step + 1,
        found.title,
        text,
        found.checked(),
        found.steps.len()
    );
    {
        let mut state = convo.state();
        if let Some(running) = &state.running {
            lock(&running.notes).push(note);
        } else {
            state.notes.push(note);
        }
    }
    Ok((found.complete() && found.notify && !found.told).then(|| done_message(&found.title, found.checked(), found.steps.len())))
}

/// Whether finishing `list` tells the agent by itself.
pub(crate) fn set_notify(inner: &Arc<Inner>, conversation: &str, list: &str, notify: bool) -> Result<(), Refusal> {
    let _held = lock(&SETTLING);
    let (convo, _) = found(inner, conversation, list)?;
    if !write(&convo, &Item::StepsNotify { list: list.to_owned(), notify, at: inner.now() }) {
        return Err(refuse(RefusalKind::Unavailable, words::NOT_SAVED));
    }
    convo.log.push(ConversationEventKind::StepsNotify { list: list.to_owned(), notify });
    Ok(())
}

/// The reader's Done (or a list just finished): mark `list` told and send its
/// done message in the conversation, as the composer sends one.
pub(crate) async fn tell_done(
    inner: Arc<Inner>,
    conversation: String,
    list: String,
    choice: String,
    shown: Shown,
) -> Result<Accepted, Refusal> {
    let (convo, text, mode, workspace) = {
        let inner = inner.clone();
        let (conversation, list) = (conversation.clone(), list.clone());
        inner
            .handle
            .clone()
            .spawn_blocking(move || -> Result<_, Refusal> {
                let _held = lock(&SETTLING);
                let (convo, found) = found(&inner, &conversation, &list)?;
                if found.told {
                    return Err(refuse(RefusalKind::Conflict, words::TOLD));
                }
                if !write(&convo, &Item::StepsDone { list: list.clone(), at: inner.now() }) {
                    return Err(refuse(RefusalKind::Unavailable, words::NOT_SAVED));
                }
                convo.log.push(ConversationEventKind::StepsDone { list: list.clone() });
                let text = done_message(&found.title, found.checked(), found.steps.len());
                let (mode, workspace) = {
                    let state = convo.state();
                    (state.mode, state.workspace.as_ref().map(|w| w.id.clone()))
                };
                Ok((convo, text, mode, workspace))
            })
            .await
            .unwrap_or_else(|_| Err(refuse(RefusalKind::Unavailable, refusals::RUNTIME)))?
    };
    drop(convo);
    let request = SendRequest {
        conversation: Some(conversation.clone()),
        text,
        choice,
        shown,
        mode,
        workspace,
        edit_of: None,
        project: inner.projects.of_chat(&conversation).map(|p| p.id),
        images: Vec::new(),
    };
    validate_send(&request, inner.config.development)?;
    super::agent::start(inner, Ask::Send(request)).await
}
