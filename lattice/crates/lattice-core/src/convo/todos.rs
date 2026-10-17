//! The agent's to-do list: the steps of the work in hand, as Claude Code and
//! Cursor show theirs (the tool-parity direction, 2026-10-08). The
//! reader sees where the agent is while it works.
//!
//! - `update_todos` (`items`: each a `content` and a `status`, `pending`,
//!   `in_progress` or `done`) writes the whole list as it now stands; it
//!   replaces the one before, and an empty list clears it. It is offered in
//!   both modes and in a turn with no folder, with no approval: it writes only
//!   the conversation's own record.
//! - **The record:** each list is an [`Item::TodosUpdated`], and the event of
//!   the same name carries the list, so the window shows the latest one and
//!   a reopened conversation shows it again.
//! - **Bounds:** at most [`MAX_ITEMS`] steps, each one line of at most
//!   [`MAX_CONTENT_CHARS`] characters (so an event stays within the protocol's
//!   2 KiB), and at most one step in progress. The text is redacted before it
//!   is written, so what the reader sees and the record agree.
//!
//! - **Stack decisions** (`super::choices`): a step the agent marks with
//!   `decision` (an area of the advisor's catalog), or whose words match an
//!   area's triggers, needs a decision until one is made; a step in an area
//!   the project already uses, or one decided in the conversation, names the
//!   option instead. The agent is told which steps need `recommend_stack`.
//!
//! Not a port: the web Lattice has no to-do list.

use lattice_protocol::conversation::{ConversationEventKind, TodoItem, TodoStatus};
use serde::Deserialize;
use serde_json::Value;

use super::item::Item;
use super::turn::TurnTools;
use crate::secrets;
use crate::tools::read::{ToolError, parse_args};

/// The most steps in the list.
pub const MAX_ITEMS: usize = 12;
/// The longest step, in characters.
pub const MAX_CONTENT_CHARS: usize = 100;

/// `update_todos`'s arguments.
#[derive(Clone, Debug, Deserialize)]
pub struct UpdateArgs {
    pub items: Vec<TodoItem>,
}

/// The latest list the records hold (empty when there is none).
pub fn latest(items: &[Item]) -> Vec<TodoItem> {
    items
        .iter()
        .rev()
        .find_map(|item| match item {
            Item::TodosUpdated { items, .. } => Some(items.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// `update_todos`, on the blocking pool.
pub(crate) fn tool(tools: &TurnTools, call: &str, args: &Value) -> Result<String, ToolError> {
    let args: UpdateArgs = parse_args(args)?;
    if args.items.len() > MAX_ITEMS {
        return Err(ToolError::new(format!(
            "A to-do list has at most {MAX_ITEMS} steps; merge some, or drop the finished ones."
        )));
    }
    let mut items: Vec<TodoItem> = Vec::with_capacity(args.items.len());
    let mut redacted = false;
    for step in &args.items {
        let content = step.content.trim();
        if content.is_empty()
            || content.chars().count() > MAX_CONTENT_CHARS
            || content.chars().any(char::is_control)
        {
            return Err(ToolError::new(format!(
                "Each step is one line of 1 to {MAX_CONTENT_CHARS} characters."
            )));
        }
        let clean = secrets::redact(content);
        redacted |= clean != content;
        items.push(TodoItem {
            content: clean,
            status: step.status,
            decision: step.decision.clone(),
            chosen: None,
        });
    }
    let working = items
        .iter()
        .filter(|step| step.status == TodoStatus::InProgress)
        .count();
    if working > 1 {
        return Err(ToolError::new(
            "At most one step is in progress at a time; mark the others pending or done.",
        ));
    }
    // Stack decisions: a step marked with an area, or whose words match one, needs a decision until the
    // conversation made one or the project already uses an option of that area.
    let record = tools.inner.sidecars.read_items(&tools.convo.id).map(|log| log.items).unwrap_or_default();
    let mut project: Option<crate::advice::Project> = None;
    let mut needed: Vec<(usize, &'static crate::advice::Area)> = Vec::new();
    for (n, step) in items.iter_mut().enumerate() {
        let area = match step.decision.as_deref() {
            Some(id) => Some(crate::advice::area(id.trim()).ok_or_else(|| {
                ToolError::new(format!("{id} is not an area of the stack catalog; recommend_stack lists them."))
            })?),
            None if step.status != TodoStatus::Done => crate::advice::area_for_step(&step.content),
            None => None,
        };
        let Some(area) = area else { continue };
        let project = project.get_or_insert_with(|| super::choices::project_of(tools));
        if let Some(option) = super::choices::decided(&record, &area.id) {
            step.decision = None;
            step.chosen = Some(area.choice(&option).map_or(option, |c| c.name.clone()));
        } else if let Some(used) = area.options.iter().find(|c| project.uses(c)) {
            step.decision = None;
            step.chosen = Some(used.name.clone());
        } else if step.decision.is_some() || step.status != TodoStatus::Done {
            step.decision = Some(area.id.clone());
            needed.push((n + 1, area));
        }
    }
    let convo = &tools.convo;
    let sidecar = convo.state().sidecar.clone().ok_or_else(|| {
        ToolError::new("This conversation has no record yet, so the list was not saved.")
    })?;
    let item = Item::TodosUpdated {
        turn: tools.turn.clone(),
        call_id: call.to_owned(),
        items: items.clone(),
        at: tools.inner.now(),
    };
    if sidecar.append(&item).is_err() {
        return Err(ToolError::new(
            "Lattice could not save the to-do list, so it was not updated.",
        ));
    }
    let done = items
        .iter()
        .filter(|step| step.status == TodoStatus::Done)
        .count();
    let total = items.len();
    convo
        .log
        .push(ConversationEventKind::TodosUpdated { items });
    let mut said = if total == 0 {
        "The to-do list is cleared.".to_owned()
    } else {
        format!(
            "The to-do list has {total} step{}, {done} done; the user sees it above the composer.",
            if total == 1 { "" } else { "s" }
        )
    };
    for (n, area) in &needed {
        said.push_str(&format!(
            " Step {n} needs a decision ({}): call recommend_stack with area \"{}\" before doing it; the user picks.",
            area.title, area.id
        ));
    }
    if redacted {
        said.push_str(&format!(
            " Text that looked like a secret was replaced with {}.",
            secrets::REDACTED
        ));
    }
    Ok(said)
}
