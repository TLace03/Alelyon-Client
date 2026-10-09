//! What the agent chat shows: a record's events, its open turn and its queue,
//! a workspace's view, a conversation's changes, and lines read by window
//! (the chat core's spec §4.2, §4.6, §11.2 CR5, §13.4; row E11). Not a
//! port.

use lattice_protocol::chat::{ChatTurn, Role};
use lattice_protocol::conversation::{
    ChangeSet, ChangeView, ConversationEventKind, ConversationSummary, Decision, DiffLineKind,
    Lines, Mode, Origin, TrustState, TurnStatus, ViewRef, WorkspaceView,
};
use lattice_protocol::{Refusal, RefusalKind};
use serde_json::Value;

use super::agent::{Convo, Inner, lock, refuse, words};
use super::item::{Item, Payload};
use crate::staging::{Staging, is_waiting};
use crate::workspace::Workspace;
use crate::workspace::paths::Want;

/// The longest line `read_lines` returns, in characters.
pub const MAX_LINE_CHARS: usize = 2_000;
/// The most lines one `read_lines` returns.
pub const MAX_WINDOW: u32 = 2_000;
const PREVIEW: usize = 2 * 1024;

fn payload_preview(payload: &Payload) -> String {
    match payload {
        Payload::Inline(text) => {
            let mut cut = PREVIEW.min(text.len());
            while !text.is_char_boundary(cut) {
                cut -= 1;
            }
            text[..cut].to_owned()
        }
        Payload::Blob { preview, .. } => preview.clone(),
    }
}

/// The recorded events of a reopened conversation, from its items.
pub(crate) fn seed(convo: &Convo, items: &[Item]) {
    for item in items {
        let kind = match item {
            Item::TurnStart {
                turn,
                mode,
                kind,
                label,
                resolved,
                ..
            } => ConversationEventKind::TurnStarted {
                turn: turn.clone(),
                kind: *kind,
                mode: *mode,
                label: label.clone(),
                locality: *resolved,
            },
            Item::ToolCall {
                call_id,
                tool,
                summary,
                ..
            } => ConversationEventKind::ToolCall {
                call_id: call_id.clone(),
                tool: tool.clone(),
                summary: summary.clone(),
                target: None,
            },
            Item::ToolResult {
                call_id,
                output,
                withheld,
                truncated,
                ..
            } => ConversationEventKind::ToolOutput {
                call_id: call_id.clone(),
                preview: payload_preview(output),
                withheld: *withheld,
                truncated: *truncated,
            },
            Item::ApprovalRequested {
                call_id,
                kind,
                detail,
                ..
            } => ConversationEventKind::ApprovalRequested {
                call_id: call_id.clone(),
                kind: *kind,
                detail: detail.clone(),
                allow_always_offer: false,
            },
            Item::ApprovalDecided {
                call_id,
                decision,
                by,
                ..
            } => ConversationEventKind::ApprovalResolved {
                call_id: call_id.clone(),
                approved: *decision == Decision::Approve,
                by: by.clone(),
            },
            Item::Question { call_id, text, .. } => ConversationEventKind::Question {
                call_id: call_id.clone(),
                text: text.clone(),
                options: Vec::new(),
            },
            Item::Answer { call_id, .. } => ConversationEventKind::QuestionAnswered {
                call_id: call_id.clone(),
            },
            Item::TaskSuggested {
                task,
                title,
                summary,
                prompt,
                ..
            } => ConversationEventKind::TaskSuggested {
                task: task.clone(),
                title: title.clone(),
                summary: summary.clone(),
                prompt: prompt.clone(),
            },
            Item::ArtifactSaved {
                name,
                title,
                kind,
                version,
                bytes,
                ..
            } => ConversationEventKind::ArtifactSaved {
                name: name.clone(),
                title: title.clone(),
                kind: *kind,
                version: *version,
                bytes: *bytes,
            },
            Item::PlanProposed {
                plan, title, text, ..
            } => ConversationEventKind::PlanProposed {
                plan: plan.clone(),
                title: title.clone(),
                text: text.clone(),
            },
            Item::ImagesAttached { turn, images, .. } => ConversationEventKind::ImagesAttached {
                turn: turn.clone(),
                count: images.len() as u32,
                bytes: images.iter().map(|image| image.bytes).sum(),
            },
            Item::PlanSettled { plan, outcome, .. } => ConversationEventKind::PlanSettled {
                plan: plan.clone(),
                outcome: outcome.clone(),
            },
            Item::TodosUpdated { items, .. } => ConversationEventKind::TodosUpdated {
                items: items.clone(),
            },
            Item::TaskSettled { task, outcome, .. } => ConversationEventKind::TaskSettled {
                task: task.clone(),
                outcome: outcome.clone(),
            },
            Item::Staged { change, .. } => ConversationEventKind::Staged {
                change: change.id.clone(),
                path: change.path.clone(),
                added: 0,
                removed: 0,
                authority: change.authority,
            },
            Item::Reviewed { change, result, .. } => ConversationEventKind::Reviewed {
                change: change.clone(),
                result: result.clone(),
            },
            Item::Checkpoint {
                id, kind, reason, ..
            } => ConversationEventKind::Checkpoint {
                id: *id,
                kind: *kind,
                reason: reason.clone(),
            },
            Item::CommandEffect {
                call_id,
                before,
                after,
                files,
            } => ConversationEventKind::CommandEffect {
                call_id: call_id.clone(),
                before: *before,
                after: *after,
                files: files.clone(),
            },
            Item::Steered { text, .. } => ConversationEventKind::Steered { text: text.clone() },
            Item::Queued {
                queued_id, text, ..
            } => ConversationEventKind::Queued {
                queued_id: queued_id.clone(),
                text: text.clone(),
                position: 0,
            },
            Item::Dequeued {
                queued_id, outcome, ..
            } => ConversationEventKind::Dequeued {
                queued_id: queued_id.clone(),
                outcome: outcome.clone(),
            },
            Item::Notice { text, .. } => ConversationEventKind::Notice { text: text.clone() },
            Item::TurnEnd { turn, status, .. } => ConversationEventKind::TurnEnded {
                turn: turn.clone(),
                status: *status,
            },
            // Reasoning is never shown (§22.8 RP3); a probe is said by its
            // notice or its turn's error.
            Item::Reasoning { .. }
            | Item::ToolProbe { .. }
            | Item::MovedAside { .. }
            | Item::ModeSwitch { .. }
            | Item::ModelSwitch { .. }
            | Item::Superseded { .. } => continue,
        };
        convo.log.push(kind);
    }
}

/// The turn that started and never ended (§4.6), if the last one did not.
pub(crate) fn open_turn(items: &[Item]) -> Option<String> {
    let mut open = None;
    for item in items {
        match item {
            Item::TurnStart { turn, .. } => open = Some(turn.clone()),
            Item::TurnEnd { turn, .. } if open.as_ref() == Some(turn) => open = None,
            _ => {}
        }
    }
    open
}

/// The last turn's end.
pub(crate) fn last_end(items: &[Item]) -> Option<(String, TurnStatus)> {
    items.iter().rev().find_map(|item| match item {
        Item::TurnEnd { turn, status, .. } => Some((turn.clone(), *status)),
        _ => None,
    })
}

/// Messages queued and never sent, cancelled or restored.
pub(crate) fn still_queued(items: &[Item]) -> Vec<(String, String)> {
    let mut queued: Vec<(String, String)> = Vec::new();
    for item in items {
        match item {
            Item::Queued {
                queued_id, text, ..
            } => queued.push((queued_id.clone(), text.clone())),
            Item::Dequeued { queued_id, .. } => queued.retain(|(id, _)| id != queued_id),
            _ => {}
        }
    }
    queued
}

/// A workspace as the composer shows it.
pub(crate) fn workspace_view(inner: &Inner, workspace: &Workspace) -> WorkspaceView {
    let trust = inner.trust.state(workspace);
    let rules_found = workspace
        .with_rules(&inner.runner, crate::workspace::rules::found)
        .unwrap_or_default();
    WorkspaceView {
        workspace: workspace.badge(trust == TrustState::Trusted),
        trust,
        rules_found,
        git_top_level: workspace
            .top_level
            .as_deref()
            .map(crate::workspace::shown_path),
    }
}

/// A summary when the row cannot be read just now.
pub(crate) fn bare_summary(id: &str) -> ConversationSummary {
    ConversationSummary {
        id: id.to_owned(),
        title: String::new(),
        created: 0.0,
        updated: 0.0,
        turns: 0,
        pinned_provider: String::new(),
        workspace: None,
        mode: Mode::Ask,
        origin: Origin::Native,
        running: true,
        needs_you: false,
    }
}

/// A turn that was not written (no question for a regenerate).
pub(crate) fn no_turn() -> ChatTurn {
    ChatTurn {
        id: String::new(),
        ts: 0.0,
        role: Role::User,
        text: String::new(),
        tools: vec![],
        facts: vec![],
        unsupported: vec![],
        provider: String::new(),
        error: String::new(),
        constrained: false,
        truncated: false,
        cancelled: false,
        prompt_tokens: None,
        completion_tokens: None,
    }
}

/// Lines added and removed by a change.
pub(crate) fn added_removed(staging: &Staging, change: &str) -> (u32, u32) {
    let Some(diff) = crate::staging::review::file_diff(staging, change) else {
        return (0, 0);
    };
    let mut added = 0;
    let mut removed = 0;
    for hunk in &diff.hunks {
        for line in &hunk.lines {
            match line.kind {
                DiffLineKind::Add => added += 1,
                DiffLineKind::Remove => removed += 1,
                DiffLineKind::Context => {}
            }
        }
    }
    (added, removed)
}

/// A conversation's changes (the Changes panel).
pub(crate) fn change_set(inner: &Inner, convo: &Convo) -> ChangeSet {
    let (staging, workspace) = {
        let state = convo.state();
        (state.staging.clone(), state.workspace.clone())
    };
    let Some(staging) = staging else {
        return ChangeSet {
            changes: Vec::new(),
            waiting: 0,
            command_running: false,
        };
    };
    let changes = staging
        .changes()
        .into_iter()
        .map(|change| {
            let diff = crate::staging::review::file_diff(&staging, &change.id);
            let (added, removed) = added_removed(&staging, &change.id);
            ChangeView {
                binary: diff.as_ref().is_some_and(|diff| diff.binary),
                id: change.id,
                path: change.path,
                kind: change.kind,
                authority: change.authority,
                state: change.state,
                origin: change.origin,
                added,
                removed,
            }
        })
        .collect::<Vec<_>>();
    let waiting = changes
        .iter()
        .filter(|change| is_waiting(&change.state))
        .count() as u32;
    ChangeSet {
        changes,
        waiting,
        command_running: workspace
            .as_ref()
            .is_some_and(|workspace| inner.slots.running(&workspace.id)),
    }
}

fn window(text: &str, from: u32, count: u32) -> Lines {
    let all: Vec<&str> = text.lines().collect();
    let total = all.len() as u32;
    let from = from.max(1);
    let count = count.clamp(1, MAX_WINDOW);
    let mut truncated = false;
    let lines = all
        .iter()
        .skip((from - 1) as usize)
        .take(count as usize)
        .map(|line| {
            if line.chars().count() > MAX_LINE_CHARS {
                truncated = true;
                line.chars().take(MAX_LINE_CHARS).collect()
            } else {
                (*line).to_owned()
            }
        })
        .collect();
    Lines {
        from,
        total,
        lines,
        truncated,
    }
}

/// `read_lines` (§4.3, CR5): a workspace file through the path rules, a
/// staged change's new text, or a command's whole output, by window.
pub(crate) fn read_lines(
    inner: &Inner,
    view: &ViewRef,
    from: u32,
    count: u32,
) -> Result<Lines, Refusal> {
    let gone = || refuse(RefusalKind::NotFound, "That is not there to read.");
    let bytes: Vec<u8> = match view {
        ViewRef::File { workspace, path } => {
            let workspace = lock(&inner.workspaces)
                .get(workspace)
                .cloned()
                .ok_or_else(|| refuse(RefusalKind::NotFound, words::NO_WORKSPACE))?;
            workspace
                .with_rules(&inner.runner, |rules| {
                    let resolved = rules
                        .resolve(path, Want::Existing)
                        .map_err(|error| refuse(RefusalKind::Invalid, &error.sentence()))?;
                    let file = resolved.file.ok_or_else(gone)?;
                    crate::tools::read::read_bounded(file)
                        .map_err(|_| refuse(RefusalKind::Invalid, "That file cannot be read here."))
                })
                .map_err(|_| {
                    refuse(RefusalKind::Invalid, "That folder's rules cannot be read.")
                })??
        }
        ViewRef::Staged {
            conversation,
            change,
        } => {
            let convo = inner.load(conversation)?;
            let staging = convo.state().staging.clone().ok_or_else(gone)?;
            staging
                .snapshot(change)
                .and_then(|snapshot| snapshot.new)
                .ok_or_else(gone)?
        }
        ViewRef::Output {
            conversation,
            call_id,
        } => {
            let convo = inner.load(conversation)?;
            let sidecar = convo.state().sidecar.clone().ok_or_else(gone)?;
            let items = inner
                .sidecars
                .read_items(conversation)
                .map(|log| log.items)
                .unwrap_or_default();
            let blob = items
                .iter()
                .rev()
                .find_map(|item| match item {
                    Item::ToolResult {
                        call_id: id,
                        output_blob: Some(blob),
                        ..
                    } if id == call_id => Some(blob.clone()),
                    _ => None,
                })
                .ok_or_else(gone)?;
            sidecar.read_blob(&blob).map_err(|_| gone())?
        }
    };
    let text = crate::exec::run::output_text(&bytes);
    Ok(window(&text, from, count))
}

/// A tool call as its card names it: one line, and the path or command.
pub(crate) fn call_summary(tool: &str, arguments: &str) -> (String, Option<String>) {
    let value: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
    // §12: `mcp__<server>__<tool>`, named by its tool and server, with its
    // arguments as what it is about.
    if let Some(rest) = tool.strip_prefix(crate::mcp::names::PREFIX) {
        let (server, name) = rest.split_once("__").unwrap_or((rest, ""));
        let target = match &value {
            Value::Object(map) if !map.is_empty() => Some(value.to_string()),
            _ => None,
        };
        let short = target
            .as_deref()
            .map(|text| text.chars().take(200).collect::<String>())
            .unwrap_or_default();
        return (
            format!("{name} ({server}) {short}").trim_end().to_owned(),
            target,
        );
    }
    let field = |name: &str| value.get(name).and_then(Value::as_str).map(str::to_owned);
    // The agent's browser: what it opens, clicks (and its effect), types or presses.
    if tool.starts_with("browser_") || tool.starts_with("desktop_") || tool == "web_search" {
        let target = match tool {
            "browser_open" => field("url"),
            "web_search" => field("query"),
            "browser_click" | "desktop_click" => field("what").map(|what| match field("effect") {
                Some(effect) => format!("{what} ({effect})"),
                None => what,
            }),
            "browser_type" | "desktop_type" => field("text"),
            "browser_key" | "desktop_key" => field("key").map(|key| match field("effect") {
                Some(effect) => format!("{key} ({effect})"),
                None => key,
            }),
            "browser_scroll" | "desktop_scroll" => field("direction"),
            _ => None,
        };
        let short = target
            .as_deref()
            .map(|text| text.chars().take(200).collect::<String>())
            .unwrap_or_default();
        return (format!("{tool} {short}").trim_end().to_owned(), target);
    }
    let target = match tool {
        "run_command" => field("command"),
        "glob" | "grep" => field("pattern"),
        "ask_question" => field("question"),
        // A task, by its title; a withdrawal, by the task's id.
        "suggest_task" => field("title"),
        "withdraw_task" => field("task_id"),
        // An artifact, by its title when written and by its name when read.
        "write_artifact" => field("title"),
        "read_artifact" => field("name"),
        // A plan, by its title.
        "propose_plan" => field("title"),
        // The to-do list, by the step now in progress.
        "update_todos" => value
            .get("items")
            .and_then(Value::as_array)
            .and_then(|items| {
                items
                    .iter()
                    .find(|step| step.get("status").and_then(Value::as_str) == Some("in_progress"))
            })
            .and_then(|step| step.get("content").and_then(Value::as_str))
            .map(str::to_owned),
        // A skill, by its name; one of its files, by the name and the path.
        "use_skill" => field("name"),
        "read_skill_file" => field("name").map(|name| match field("path") {
            Some(path) => format!("{name}: {path}"),
            None => name,
        }),
        _ => field("path"),
    };
    let short = target
        .as_deref()
        .map(|text| text.chars().take(200).collect::<String>())
        .unwrap_or_default();
    (format!("{tool} {short}").trim_end().to_owned(), target)
}
