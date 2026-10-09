//! Sessions: where a run's conversation is kept from one run to the next.
//!
//! Ports the shape of `agents.memory.session.Session` (0.22.3) and the calls
//! the runner makes to one (`run_internal/session_persistence.py`:
//! `prepare_input_with_session`, `save_result_to_session`), for a session the
//! caller implements. This crate implements none: where a conversation is kept
//! is the caller's business, so nothing here reads or writes a file.
//!
//! What a run given [`RunConfig::session`](crate::RunConfig::session) does, in
//! the SDK's order (pinned by the goldens `session_*`, recorded with the SDK's
//! `SQLiteSession(":memory:")`):
//! 1. Before anything else, `get_items(None)`: the conversation so far, oldest
//!    first. Every model call of the run is given those items, then the run's
//!    own input items, then what the run has added since (the SDK prepares
//!    "history, then the new input").
//! 2. Then `add_items(input)`: the run's input items, saved before any
//!    guardrail runs (the SDK saves them even when an input guardrail trips).
//! 3. After each completed turn, `add_items` with that turn's new items, in
//!    the order the run's history holds them: the assistant's messages and
//!    tool calls as the model produced them, then the tool outputs. The final
//!    turn's items are saved after the output guardrails pass; when one trips,
//!    they are not. A turn that fails or is cancelled saves nothing, and a turn
//!    with no new items makes no call. So a crash loses at most one turn.
//!    One exception, the SDK's interruption: when a turn has calls that need
//!    approval, its items up to the outputs of the calls that need none are
//!    saved before the approvals are awaited to their end, and the decided
//!    calls' outputs in a later call (pinned by the golden
//!    `approval_needed_before_free_calls`).
//!
//! Failure:
//! - A `get_items` that fails (or panics) ends the run before it starts
//!   anything (no trace, no span, no event) with `RunError::User("the session
//!   could not be read: ...")`.
//! - An `add_items` that fails (or panics) does not stop the run. The run
//!   emits [`StreamEvent::SessionWriteFailed`](crate::StreamEvent) once, at the
//!   first failure, and goes on.
//!
//! Deviations, and why:
//! - There is no `pop_item` and no `clear_session`. Both remove records of a
//!   conversation, and Lattice never deletes them.
//! - The runner always asks for every item (`limit: None`); the SDK passes its
//!   `SessionSettings.limit`, which is not ported. A session that keeps the
//!   conversation within a budget returns fewer items itself.
//! - Reasoning items are not saved: a run's history here holds no reasoning
//!   (see [`crate::run`]), where the SDK saves and replays reasoning items.
//! - `session_input_callback` is not ported: the input items are appended to
//!   the history as given.

use std::fmt;

use futures::future::BoxFuture;

use crate::model::InputItem;

/// Why a session could not be read or written. The message is the session's
/// own; it reaches the caller in an error or an event, never a span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionError {
    message: String,
}

impl SessionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SessionError {}

/// A conversation kept between runs (`agents.memory.session.Session`).
pub trait Session: Send + Sync {
    /// The conversation so far, oldest first, as model input. `limit` counts
    /// items from the newest, as in the SDK; `None` is every item.
    fn get_items(
        &self,
        limit: Option<usize>,
    ) -> BoxFuture<'static, Result<Vec<InputItem>, SessionError>>;

    /// Called once at the start of a run with the run's input items, then once
    /// per completed model turn with that turn's new items.
    fn add_items(&self, items: Vec<InputItem>) -> BoxFuture<'static, Result<(), SessionError>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_error_says_only_what_it_was_given() {
        let error = SessionError::new("the disk is full");
        assert_eq!(error.to_string(), "the disk is full");
        assert_eq!(error.message(), "the disk is full");
    }
}
