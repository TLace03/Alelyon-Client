//! A lab agent's client reads and writes in its conversation's folder
//! (`fs/read_text_file`, `fs/write_text_file`; `crate::acp::pool::ChatFiles`).
//!
//! - **A read** sees the folder as the conversation does: the path rules of
//!   Lattice's own read tools (inside the folder, not ignored, not `.git`,
//!   no link out), and the conversation's staged changes over the disk
//!   (`tools::read::read_text`).
//! - **A write is never a write.** It is staged exactly as the built-in
//!   agent's `write_file` stages (`tools::edit::agent_write`): the same
//!   policy (Ask mode and an untrusted folder refuse), the same path rules,
//!   size bound and encoding, into the same review, recorded under the turn
//!   running in the conversation and the agent's call. It is listed in the
//!   turn as a staged change, and reaches the disk only when the reader keeps
//!   it in the Changes panel.
//! - A conversation not open now, or one with no folder, refuses both.

use std::sync::{Arc, OnceLock, Weak};

use lattice_protocol::conversation::{ConversationEventKind, TrustState};

use super::agent::{Convo, Inner};
use crate::acp::pool::ChatFiles;
use crate::tools::edit::{self, StageContext};
use crate::tools::read::{self, NoOverlay, Overlay, ReadContext};

/// Refused: the conversation is not open, or it has no folder (or no review).
pub const NOT_OPEN: &str =
    "This chat's folder is not open in Lattice now, so nothing was read or staged.";

/// The conversations' folders, through the chat service once it is made.
pub(crate) struct ConvoFiles {
    pub(crate) inner: Arc<OnceLock<Weak<Inner>>>,
}

impl ConvoFiles {
    fn open(&self, thread: &str) -> Result<(Arc<Inner>, Arc<Convo>), String> {
        let inner = self
            .inner
            .get()
            .and_then(Weak::upgrade)
            .ok_or_else(|| NOT_OPEN.to_owned())?;
        let convo = inner.loaded(thread).ok_or_else(|| NOT_OPEN.to_owned())?;
        Ok((inner, convo))
    }
}

impl ChatFiles for ConvoFiles {
    fn read(&self, thread: &str, path: &str) -> Result<String, String> {
        let (inner, convo) = self.open(thread)?;
        let (workspace, staging) = {
            let state = convo.state();
            (state.workspace.clone(), state.staging.clone())
        };
        let workspace = workspace.ok_or_else(|| NOT_OPEN.to_owned())?;
        let overlay: &dyn Overlay = match &staging {
            Some(staging) => staging.as_ref(),
            None => &NoOverlay,
        };
        let ctx = ReadContext {
            workspace: &workspace,
            runner: &inner.runner,
            overlay,
        };
        read::read_text(&ctx, path).map_err(|error| error.0)
    }

    fn write(&self, thread: &str, path: &str, content: &str, call: &str) -> Result<String, String> {
        let (inner, convo) = self.open(thread)?;
        let (workspace, staging, mode, turn) = {
            let state = convo.state();
            (
                state.workspace.clone(),
                state.staging.clone(),
                state.mode,
                state
                    .running
                    .as_ref()
                    .map(|running| running.turn.clone())
                    .unwrap_or_default(),
            )
        };
        let (Some(workspace), Some(staging)) = (workspace, staging) else {
            return Err(NOT_OPEN.to_owned());
        };
        let trusted = inner.trust.state(&workspace) == TrustState::Trusted;
        let call = call.to_owned();
        let before = staging.changes();
        let ctx = StageContext {
            workspace: &workspace,
            runner: &inner.runner,
            staging: &staging,
            mode,
            trusted,
            turn: &turn,
            call: &call,
        };
        let told = edit::agent_write(&ctx, path, content).map_err(|error| error.0)?;
        // Listed in the running turn, as the built-in agent's staging is.
        for change in staging.changes() {
            if before.contains(&change) {
                continue;
            }
            let (added, removed) = super::views::added_removed(&staging, &change.id);
            convo.log.push(ConversationEventKind::Staged {
                change: change.id.clone(),
                path: change.path.clone(),
                added,
                removed,
                authority: change.authority,
            });
        }
        Ok(told)
    }
}
