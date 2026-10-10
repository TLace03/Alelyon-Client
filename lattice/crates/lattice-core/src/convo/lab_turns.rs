//! What a lab agent's turn changed in its folder (Claude Code, Codex): each
//! writes with its own tools, its edits and its shell commands alike, so the
//! turn is watched the way a command is (`crate::changes::effects`).
//!
//! - **Before the turn** ([`begin`]): the folder's one command slot (so no
//!   Keep and no command lands inside the turn), the conversation's writer
//!   lease, and checkpoint B.
//! - **After it** ([`end`]): checkpoint A and, in a git folder, the paths the
//!   turn added, changed or deleted, recorded as one effect and listed in the
//!   Changes panel as changes already on disk. Keep acknowledges one; Undo
//!   stages putting it back, reviewed like any other change. A note in the
//!   conversation names the files. Without git the turn is not listed (the
//!   checkpoint says so).
//! - A turn in a conversation with no folder (the agent's own working folder)
//!   is not watched.
//!
//! Each edit the agent asks to make already waited for the reader's yes, with
//! its change shown (`crate::acp::pool`); this lists the whole turn's effect,
//! shell commands included, and makes it undoable.

use std::sync::Arc;

use lattice_protocol::conversation::{CallId, CheckpointId, ConversationEventKind};

use super::agent::{Convo, Inner};
use crate::changes::checkpoint::Checkpoints;
use crate::changes::effects::{CommandGuards, MAX_NAMED};
use crate::exec::run::{CommandHooks, SlotGuard};
use crate::policy::Lease;
use crate::staging::Staging;
use crate::workspace::Workspace;
use crate::workspace::lease::WriterLease;

/// A watched turn, from [`begin`] to [`end`].
pub(crate) struct Watch {
    call: CallId,
    before: Option<CheckpointId>,
    _slot: SlotGuard,
    workspace: Workspace,
    staging: Arc<Staging>,
    checkpoints: Arc<Checkpoints>,
    lease: Arc<WriterLease>,
    /// The agent's name, as the note says it.
    label: String,
}

/// The id the turn's effect is recorded under.
pub(crate) fn call_id(turn: &str) -> CallId {
    format!("agent_turn_{turn}")
}

/// Watch the turn `turn` of `convo`: `None` when the conversation has no
/// folder; `Some(Err)` when the folder cannot be watched now, in words.
/// Blocking (a checkpoint runs git).
pub(crate) fn begin(
    inner: &Inner,
    convo: &Convo,
    turn: &str,
    label: &str,
) -> Option<Result<Watch, String>> {
    let (workspace, staging, checkpoints, lease) = {
        let state = convo.state();
        (
            state.workspace.clone()?,
            state.staging.clone()?,
            state.checkpoints.clone()?,
            state.lease.clone()?,
        )
    };
    let call = call_id(turn);
    let Some(slot) = inner.slots.claim(&workspace.id, &call) else {
        return Some(Err(format!(
            "A command is running in this folder, so what {label} changes in this turn is not listed."
        )));
    };
    if lease.take() == Lease::Elsewhere {
        return Some(Err(format!(
            "Another conversation is changing this folder, so what {label} changes in this turn is not listed."
        )));
    }
    let guards = CommandGuards {
        checkpoints: &checkpoints,
        workspace: &workspace,
        runner: &inner.runner,
        staging: &staging,
        lease: &lease,
    };
    let before = match guards.before(&call) {
        Ok(before) => before,
        Err(why) => {
            return Some(Err(format!(
                "Lattice could not record the folder before {label}'s turn ({why}), so what it changes is not listed."
            )));
        }
    };
    Some(Ok(Watch {
        call,
        before,
        _slot: slot,
        workspace,
        staging,
        checkpoints,
        lease,
        label: label.to_owned(),
    }))
}

/// What the note says the turn changed.
pub(crate) fn note(label: &str, paths: &[String]) -> String {
    let n = paths.len();
    if n == 0 {
        return format!("{label} changed no file in this folder.");
    }
    let named = paths
        .iter()
        .take(MAX_NAMED)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let more = n.saturating_sub(MAX_NAMED);
    let tail = if more > 0 {
        format!(" and {more} more")
    } else {
        String::new()
    };
    let count = if n == 1 {
        "1 file".to_owned()
    } else {
        format!("{n} files")
    };
    format!(
        "{label} changed {count}: {named}{tail}. They are in the Changes panel: Keep acknowledges a change, Undo stages putting it back."
    )
}

/// Checkpoint A, the effect, the note, and the folder given back. Blocking.
pub(crate) fn end(inner: &Inner, convo: &Convo, watch: Watch) {
    let guards = CommandGuards {
        checkpoints: &watch.checkpoints,
        workspace: &watch.workspace,
        runner: &inner.runner,
        staging: &watch.staging,
        lease: &watch.lease,
    };
    let after = guards.after(&watch.call, watch.before);
    match &after.effect {
        Some(effect) => {
            convo.log.push(ConversationEventKind::CommandEffect {
                call_id: watch.call.clone(),
                before: effect.before,
                after: effect.after,
                files: effect.files.clone(),
            });
            let paths: Vec<String> = effect.files.iter().map(|f| f.path.clone()).collect();
            convo.log.push(ConversationEventKind::Notice {
                text: note(&watch.label, &paths),
            });
        }
        None => {
            for text in after.notes {
                convo.log.push(ConversationEventKind::Notice { text });
            }
        }
    }
    let busy = || inner.slots.busy(&watch.workspace.id);
    drop(watch._slot);
    if convo.state().running.is_none() {
        watch
            .lease
            .release_when_idle(watch.staging.waiting(), busy());
    }
    inner.note_changed(&convo.id);
}
