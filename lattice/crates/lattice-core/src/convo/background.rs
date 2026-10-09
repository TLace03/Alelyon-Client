//! Background commands (tool parity with Claude Code's and Codex's): `run_command {background: true}` starts a command that
//! keeps running after the call returns, a server or a watcher.
//!
//! - It is approved exactly as any command (`exec::run`: the same card, the
//!   native dialog saying it keeps running, standing entries), and runs in one
//!   of the folder's background slots (`exec::run::MAX_BACKGROUND`), not its
//!   one command slot, with no timeout.
//! - The call returns at once with the command's id. `command_output {id}`
//!   reads what it wrote since the last read (at most 32 KiB, the newest
//!   kept), and whether it still runs; `stop_command {id}` ends its tree. The
//!   reader's Stop on its card is [`Background::stop`].
//! - Its events (`CommandProgress`, then `CommandExited` and `CommandEffect`)
//!   go to its conversation's log after the turn has ended; the
//!   after-command hook records its effect as any command's.
//! - The writer lease is kept while it runs (`CommandSlots::busy`) and given
//!   back when it ends, if no turn of its conversation runs then.
//! - The turn's Stop does not end it. Closing its conversation, or Lattice,
//!   does ([`Background::stop_conversation`], [`Background::stop_all`]); the
//!   Job Object ends the tree if Lattice itself ends.
//!
//! An ended command is remembered (its last output and how it ended) until
//! [`KEPT_ENDED`] newer ones have ended. Nothing here waits on a timer.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use lattice_protocol::conversation::{CallId, ConversationEventKind, ExitReason};
use serde::Deserialize;

use super::turn::TurnTools;
use crate::exec::run::{
    Approval, OutputBuffer, Prepared, Progress, ProgressSink, StopHandle, output_text, run_into,
    status_line,
};
use crate::text::ansi_strip;
use crate::tools::read::ToolError;

/// Ended background commands remembered for `command_output`.
pub const KEPT_ENDED: usize = 20;
/// The most of its output one `command_output` returns.
pub const MAX_READ: usize = 32 * 1024;

const UNKNOWN: &str = "No background command of this conversation has that id.";

/// One background command.
struct Run {
    convo: String,
    stop: StopHandle,
    buffer: Arc<Mutex<OutputBuffer>>,
    /// Bytes of its output already returned by `command_output`.
    read: u64,
    /// How it ended, once it has.
    ended: Option<String>,
}

/// Every background command of the process, by call.
#[derive(Default)]
pub struct Background {
    runs: Mutex<BTreeMap<CallId, Run>>,
    /// The ended ones, oldest first.
    ended: Mutex<VecDeque<CallId>>,
}

impl std::fmt::Debug for Background {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Background")
            .field("runs", &lock(&self.runs).len())
            .finish()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Deserialize)]
struct IdArgs {
    id: String,
}

impl Background {
    fn add(&self, call: &CallId, convo: &str, stop: StopHandle, buffer: Arc<Mutex<OutputBuffer>>) {
        lock(&self.runs).insert(
            call.clone(),
            Run {
                convo: convo.to_owned(),
                stop,
                buffer,
                read: 0,
                ended: None,
            },
        );
    }

    /// It ended so: remembered until [`KEPT_ENDED`] newer ones have ended.
    fn end(&self, call: &CallId, how: String) {
        if let Some(run) = lock(&self.runs).get_mut(call) {
            run.ended = Some(how);
        }
        let mut ended = lock(&self.ended);
        ended.push_back(call.clone());
        while ended.len() > KEPT_ENDED {
            if let Some(old) = ended.pop_front() {
                lock(&self.runs).remove(&old);
            }
        }
    }

    /// The calls of `convo`'s background commands that still run.
    pub fn running(&self, convo: &str) -> Vec<CallId> {
        lock(&self.runs)
            .iter()
            .filter(|(_, run)| run.convo == convo && run.ended.is_none())
            .map(|(call, _)| call.clone())
            .collect()
    }

    /// End one of `convo`'s background commands; `false` when none runs by that id.
    pub fn stop(&self, convo: &str, call: &str) -> bool {
        let runs = lock(&self.runs);
        match runs.get(call) {
            Some(run) if run.convo == convo && run.ended.is_none() => {
                run.stop.stop();
                true
            }
            _ => false,
        }
    }

    /// End every background command of `convo` (it closes).
    pub fn stop_conversation(&self, convo: &str) {
        for run in lock(&self.runs).values().filter(|run| run.convo == convo) {
            run.stop.stop();
        }
    }

    /// End every background command (Lattice closes).
    pub fn stop_all(&self) {
        for run in lock(&self.runs).values() {
            run.stop.stop();
        }
    }

    /// `command_output`: what it wrote since the last read, and whether it runs.
    pub fn output(&self, convo: &str, call: &str) -> Result<String, ToolError> {
        let mut runs = lock(&self.runs);
        let run = runs
            .get_mut(call)
            .filter(|run| run.convo == convo)
            .ok_or_else(|| ToolError::new(UNKNOWN))?;
        let (total, new) = {
            let buffer = lock(&run.buffer);
            let total = buffer.total();
            let unread = usize::try_from(total.saturating_sub(run.read)).unwrap_or(usize::MAX);
            (total, (unread, buffer.tail(unread.min(MAX_READ))))
        };
        let (unread, bytes) = new;
        run.read = total;
        let mut text = match &run.ended {
            None => "It is still running.".to_owned(),
            Some(how) => format!("It has ended. {how}"),
        };
        if unread == 0 {
            text.push_str("\nNo new output since the last read.");
        } else {
            if unread > bytes.len() {
                text.push_str(&format!(
                    "\n[… {} earlier bytes not shown …]",
                    unread - bytes.len()
                ));
            }
            text.push('\n');
            text.push_str(&crate::secrets::redact(&ansi_strip::strip(&output_text(
                &bytes,
            ))));
        }
        Ok(text)
    }
}

/// `command_output` and `stop_command`.
pub(crate) fn tool(
    tools: &TurnTools,
    name: &str,
    args: &serde_json::Value,
) -> Result<String, ToolError> {
    let IdArgs { id } = serde_json::from_value(args.clone())
        .map_err(|_| ToolError::new("Give the background command's id."))?;
    let background = &tools.inner.background;
    match name {
        "stop_command" => {
            if background.stop(&tools.convo.id, &id) {
                Ok("Stopped: its whole process tree ends now.".to_owned())
            } else {
                Err(ToolError::new(
                    "No background command of this conversation runs by that id.",
                ))
            }
        }
        _ => background.output(&tools.convo.id, &id),
    }
}

/// Start an approved background command and return at once.
pub(crate) fn start(
    tools: Arc<TurnTools>,
    call: CallId,
    prepared: Prepared,
    approval: Approval,
) -> Result<String, ToolError> {
    let stop = StopHandle::default();
    let buffer = Arc::new(Mutex::new(OutputBuffer::default()));
    tools
        .inner
        .background
        .add(&call, &tools.convo.id, stop.clone(), buffer.clone());
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
    let work = tools.clone();
    let run_call = call.clone();
    // Detached: the run reports through the log and the registry, not to this call.
    drop(tools.inner.handle.spawn_blocking(move || {
        let result = match work.guards() {
            Some(guards) => {
                let ctx = work.command_context(&guards);
                run_into(
                    &ctx,
                    &prepared,
                    &approval,
                    &run_call,
                    &stop,
                    Some(progress),
                    buffer,
                )
            }
            None => Err(crate::exec::run::Refused {
                sentence: super::agent::words::NO_FOLDER_TOOL.to_owned(),
                conflict: false,
                rejected: false,
            }),
        };
        let how = match result {
            Ok(outcome) => {
                work.convo.log.push(ConversationEventKind::CommandExited {
                    call_id: run_call.clone(),
                    code: outcome.code,
                    duration_ms: outcome.duration_ms,
                    reason: outcome.reason,
                });
                if let Some(effect) = &outcome.after.effect {
                    work.convo.log.push(ConversationEventKind::CommandEffect {
                        call_id: run_call.clone(),
                        before: effect.before,
                        after: effect.after,
                        files: effect.files.clone(),
                    });
                }
                let mut how = status_line(
                    outcome.reason,
                    outcome.code,
                    outcome.duration_ms as f64 / 1000.0,
                );
                for note in &outcome.after.notes {
                    how.push(' ');
                    how.push_str(note);
                }
                how
            }
            Err(refused) => {
                work.convo.log.push(ConversationEventKind::CommandExited {
                    call_id: run_call.clone(),
                    code: None,
                    duration_ms: 0,
                    reason: ExitReason::Failed,
                });
                format!("It did not run: {}", refused.sentence)
            }
        };
        work.inner.background.end(&run_call, how);
        // The lease goes back once nothing of this folder runs, when no turn
        // of the conversation runs now.
        if let (Some(lease), Some(workspace)) = (&work.lease, &work.workspace)
            && work.convo.state().running.is_none()
        {
            lease.release_when_idle(work.staging.waiting(), work.inner.slots.busy(&workspace.id));
        }
        work.inner.note_changed(&work.convo.id);
    }));
    Ok(format!(
        "Started in the background with id {call}. It keeps running after this turn. Read what it writes with command_output and end it with stop_command, giving that id. The user can stop it too."
    ))
}
