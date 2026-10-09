//! The transcript-store seam (the chat core's spec §5.10; row C3).
//!
//! The agent chat records what was said (the reader's messages and the final
//! answers) through [`TranscriptStore`], so phases A–F can be built and
//! tested before the native Lattice may write the shared store (gate G-WEB,
//! §5.9). Two implementations:
//! - `SharedThreadStore` (`chat::store`), the shared `<globals>/lattice_chat/`
//!   in Python's format: its reads are real; its writes (row G2) refuse
//!   while `store_gate::SHARED_WRITES` is `false`, which it was until row G5
//!   opened it;
//! - `MemoryTranscriptStore` (`chat::memory`), the same contract in memory,
//!   compiled only for tests and the `dev-host` feature, so no shipped build
//!   holds a store that forgets.
//!
//! Every method is synchronous and blocking, and is called only from
//! `spawn_blocking`. Reads follow S26. There is no delete (ND1): a thread
//! leaves the list only by being archived.

use lattice_protocol::conversation::ArchiveKey;
use lattice_protocol::{Refusal, RefusalKind};

use super::archive::ArchiveState;
use super::pyjson::PyValue;
use super::store::{IndexRow, IndexState, StoredTurn};
use super::store_gate;

/// A turn to append. An empty `id` gets a new one and a zero `ts` the clock's
/// time, as `history.append` defaults them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NewTurn {
    pub id: String,
    pub ts: f64,
    /// `"user"` or `"assistant"`.
    pub role: String,
    pub text: String,
    pub tools: Vec<String>,
    pub facts: Vec<PyValue>,
    pub unsupported: Vec<String>,
    pub provider: String,
    pub error: String,
    pub constrained: bool,
    pub truncated: bool,
    pub cancelled: bool,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
}

impl NewTurn {
    /// The reader's message.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_owned(),
            text: text.into(),
            ..Self::default()
        }
    }

    /// An answer, recorded with the provider's name in Python's naming.
    pub fn assistant(text: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_owned(),
            text: text.into(),
            provider: provider.into(),
            ..Self::default()
        }
    }

    /// The stored turn, with its id and time decided.
    pub fn stored(self, id: String, ts: f64) -> StoredTurn {
        StoredTurn {
            id,
            ts,
            role: self.role,
            text: self.text,
            tools: self.tools,
            facts: self.facts,
            unsupported: self.unsupported,
            provider: self.provider,
            error: self.error,
            constrained: self.constrained,
            truncated: self.truncated,
            cancelled: self.cancelled,
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            superseded: false,
        }
    }
}

/// Why a store write wrote nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreError {
    /// Gate G-WEB: the shared store is read-only here (§5.9).
    SharedWritesOff,
    /// The index exists and cannot be read here (S3, D7).
    IndexUnreadable,
    /// The archive index cannot be read just now, or only Python can (S20).
    ArchiveUnreadable,
    /// The thread was deleted or archived while the answer was written (D8).
    ThreadGone,
    /// Unarchive: a chat with this id is already listed (S22 step 1).
    AlreadyListed,
    /// No such thread, or no such archived conversation.
    NotFound,
    /// The change was not saved (an id that names no file, a move that could
    /// not be made, a file that could not be written).
    NotSaved,
}

impl StoreError {
    /// The one sentence a refusal carries.
    pub fn refusal(self) -> Refusal {
        let (kind, message) = match self {
            Self::SharedWritesOff => (RefusalKind::Unavailable, store_gate::REFUSAL),
            Self::IndexUnreadable => (
                RefusalKind::Unavailable,
                "The chat list could not be read here; nothing will be changed.",
            ),
            Self::ArchiveUnreadable => (
                RefusalKind::Unavailable,
                "The archived chats could not be read here; nothing will be changed.",
            ),
            Self::ThreadGone => (
                RefusalKind::NotFound,
                "This chat was deleted while the answer was being written; the answer was not saved.",
            ),
            Self::AlreadyListed => (
                RefusalKind::Conflict,
                "A chat with this id is already in the list.",
            ),
            Self::NotFound => (RefusalKind::NotFound, "That conversation no longer exists."),
            Self::NotSaved => (RefusalKind::Invalid, "The message could not be saved."),
        };
        Refusal::new(kind, message)
    }
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.refusal().message)
    }
}

impl std::error::Error for StoreError {}

/// What the agent chat records through (§5.10).
pub trait TranscriptStore: Send + Sync {
    /// The thread rows (S3).
    fn list(&self) -> IndexState;
    /// The archive's rows (S20).
    fn archived(&self) -> ArchiveState;
    /// A thread's visible turns, the last 400 (S7).
    fn load(&self, id: &str) -> Vec<StoredTurn>;
    /// The last 8 visible turns (S14).
    fn recent(&self, id: &str) -> Vec<StoredTurn>;
    /// A new thread with its first message and its pin, in one write (S9).
    fn first_message(&self, text: &str, pin: &str) -> Result<(IndexRow, StoredTurn), StoreError>;
    /// Append a turn; a missing row is made, as Python makes it (S8).
    fn append(&self, id: &str, turn: NewTurn) -> Result<StoredTurn, StoreError>;
    /// Append an answer only while the thread's row exists (S8, D8).
    fn append_answer(&self, id: &str, turn: NewTurn) -> Result<StoredTurn, StoreError>;
    /// Rename; `false` for a blank title or an unknown thread (S10).
    fn rename(&self, id: &str, title: &str) -> Result<bool, StoreError>;
    /// Pin a choice; `updated` is not touched (S11).
    fn pin(&self, id: &str, pin: &str) -> Result<bool, StoreError>;
    /// Mark a turn and every later visible turn superseded; how many (S13).
    fn supersede(&self, id: &str, from: &str) -> Result<usize, StoreError>;
    /// Count the thread as used now; `false` for an unknown thread, which is
    /// not made (S25).
    fn touch(&self, id: &str) -> Result<bool, StoreError>;
    /// The reader's Archive: into `evicted/` as an eviction is (S21).
    fn archive(&self, id: &str) -> Result<(), StoreError>;
    /// Back into the list, as the most recently used thread (S22).
    fn unarchive(&self, key: &ArchiveKey) -> Result<IndexRow, StoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_refusal_is_one_sentence_without_transport_text() {
        let all = [
            StoreError::SharedWritesOff,
            StoreError::IndexUnreadable,
            StoreError::ArchiveUnreadable,
            StoreError::ThreadGone,
            StoreError::AlreadyListed,
            StoreError::NotFound,
            StoreError::NotSaved,
        ];
        for error in all {
            let refusal = error.refusal();
            assert!(refusal.message.ends_with('.'), "{refusal:?}");
            assert_eq!(refusal.message.matches(". ").count(), 0, "{refusal:?}");
            assert_eq!(error.to_string(), refusal.message);
        }
        assert_eq!(
            StoreError::SharedWritesOff.refusal(),
            Refusal::new(
                RefusalKind::Unavailable,
                "Chats are saved here once the web Lattice's privacy update is installed."
            )
        );
    }

    #[test]
    fn a_new_turn_keeps_what_it_was_given() {
        let turn = NewTurn::assistant("answer", "lab:model").stored("t1".into(), 2.5);
        assert_eq!(turn.role, "assistant");
        assert_eq!(turn.provider, "lab:model");
        assert!(!turn.superseded);
        assert_eq!(NewTurn::user("q").role, "user");
    }
}
