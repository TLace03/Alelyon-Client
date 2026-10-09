//! Chat as the window sees it: threads, turns, model choices, answer jobs and
//! their events. Field meanings follow the Python store
//! (the Python runtime's `history.py`, `Turn` and the index rows) so
//! a thread written by either Lattice reads the same in both.
//!
//! These are the native chat specification's protocol types (its §3.2, commit
//! C1), which the chat core's conversation types (`conversation.rs`) build on.
//! One change from that specification: [`ChatService`] has **no `delete`**. The
//! native Lattice removes no conversation (the chat core's never-delete rule,
//! ND1, "archive, never delete"); a thread leaves the list only
//! by being archived, which the conversation service offers.
//!
//! Invariants every producer keeps:
//! - A [`ThreadId`] matches `^[0-9A-Za-z_-]{1,64}$` ([`is_thread_id`]) before it
//!   reaches a path; a [`JobId`] is 32 lowercase hexadecimal characters.
//! - Token counts the model server did not report are `None` (UNMEASURED),
//!   never 0.
//! - Superseded turns never cross the protocol.
//! - Event `seq` numbers start at 1 and rise by one per job.
//! - No key value or key name appears in any value here.

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

use crate::{Locality, Refusal};

/// A thread's id: `^[0-9A-Za-z_-]{1,64}$`, the web's accepted id
/// (`lattice_service/app.py`). New ids are uuid4 hex, first 12 characters.
pub type ThreadId = String;
/// A turn's id (uuid4 hex, first 12 characters, for turns this side makes).
pub type TurnId = String;
/// An answer job's id: 32 lowercase hexadecimal characters, process-local.
pub type JobId = String;

/// True when `id` has the shape of a [`ThreadId`]. Ids reach file names, so
/// anything else is refused before it is used.
pub fn is_thread_id(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// True when `id` has the shape of a [`JobId`].
pub fn is_job_id(id: &str) -> bool {
    id.len() == 32
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Who wrote a turn. Any stored role other than `"assistant"` reads as
/// [`Role::User`] (the web's engine treats it so).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

/// One row of a turn's facts table. A missing or null value is `""`, and a
/// non-string value is its JSON text (the store's deviation D2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fact {
    pub label: String,
    pub rendered: String,
    pub as_of: String,
    pub note: String,
}

/// One turn of a thread, as the store holds it (less `superseded`, which never
/// crosses the protocol).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatTurn {
    pub id: TurnId,
    /// Seconds since the Unix epoch.
    pub ts: f64,
    pub role: Role,
    pub text: String,
    pub tools: Vec<String>,
    pub facts: Vec<Fact>,
    pub unsupported: Vec<String>,
    /// The provider in Python's naming: `ollama:<model>`,
    /// `<endpoint_id>:<model>` or `dev:echo`.
    pub provider: String,
    pub error: String,
    pub constrained: bool,
    pub truncated: bool,
    pub cancelled: bool,
    /// `None`: the server did not report it (UNMEASURED), never 0.
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
}

/// A thread as the list shows it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadSummary {
    pub id: ThreadId,
    pub title: String,
    pub created: f64,
    pub updated: f64,
    /// A negative stored value reads as 0 (the store keeps its own copy).
    pub turns: u64,
    /// `pinned_provider`, verbatim.
    pub pinned: String,
}

/// One entry of the chat model picker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatChoice {
    /// `auto`, `local`, `cloud`, `endpoint:<id>` or `dev:echo`.
    pub id: String,
    pub label: String,
    pub detail: String,
    pub locality: Locality,
    pub ready: bool,
    /// Why it cannot be used, in one sentence; never a key name.
    pub refusal: Option<String>,
}

/// The thread list and where it came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThreadList {
    pub threads: Vec<ThreadSummary>,
    /// Shown as the sidebar foot's tooltip.
    pub store_dir: String,
    /// `StateRoot::installed`: an installed build keeps its chats apart from
    /// the web's.
    pub installed: bool,
    /// The index exists but cannot be read here, so nothing will be changed.
    pub index_unreadable: bool,
}

/// A thread with its visible turns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpenedThread {
    pub thread: ThreadSummary,
    pub turns: Vec<ChatTurn>,
    pub job: Option<JobId>,
    /// Another native window holds this thread's answer lock.
    pub answering_elsewhere: bool,
}

/// What the picker showed for the choice when the reader pressed Send. The
/// core refuses a send whose choice now resolves elsewhere (rule N10).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shown {
    pub locality: Locality,
    pub label: String,
}

/// A message to send: to a new thread (`thread: None`) or an existing one,
/// optionally replacing (`edit_of`) one of its user turns.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SendRequest {
    pub thread: Option<ThreadId>,
    pub text: String,
    pub choice: String,
    pub edit_of: Option<TurnId>,
    pub shown: Shown,
}

/// Answer the thread's last question again.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegenerateRequest {
    pub thread: ThreadId,
    pub choice: String,
    pub shown: Shown,
}

/// A send or regenerate that started an answer job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Accepted {
    pub thread: ThreadSummary,
    pub job: JobId,
    /// The saved user turn; `None` for a regenerate.
    pub question: Option<ChatTurn>,
    pub superseded: Vec<TurnId>,
}

/// Where an answer is: the web's stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Thinking,
    Writing,
    Checking,
    Done,
}

/// What happened in an answer job: the web's event types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEventKind {
    Stage {
        stage: Stage,
        detail: String,
    },
    Delta {
        text: String,
    },
    // Boxed so a Delta does not take the space of the largest variant.
    Turn {
        turn: Box<ChatTurn>,
        saved: bool,
    },
    Error {
        message: String,
        turn: Box<ChatTurn>,
        saved: bool,
    },
    /// Always the last event of a job.
    Done,
}

/// One event of one answer job.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChatEvent {
    pub seq: u64,
    #[serde(flatten)]
    pub kind: ChatEventKind,
}

/// The local model runtime's state, as the probe found it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeState {
    Offline,
    NoModel,
    Ready,
    Error,
}

/// What the local-runtime probe found.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalRuntime {
    pub state: RuntimeState,
    pub model: String,
    pub installed: Vec<String>,
    pub headline: String,
    pub detail: String,
}

/// Chat as the window uses it. Unlike `RunService`, any call that may touch the
/// disk or the network returns a future, because a store write can wait up to
/// 10 s for another process's lock and a probe up to about 4.4 s. `stop` and
/// `follow` touch only memory and return at once. There is no `delete` (see the
/// module header).
pub trait ChatService: Send + Sync + 'static {
    fn choices(&self) -> BoxFuture<'static, Vec<ChatChoice>>;
    fn threads(&self) -> BoxFuture<'static, Result<ThreadList, Refusal>>;
    fn open(&self, thread: &str) -> BoxFuture<'static, Result<OpenedThread, Refusal>>;
    fn send(&self, request: SendRequest) -> BoxFuture<'static, Result<Accepted, Refusal>>;
    fn regenerate(
        &self,
        request: RegenerateRequest,
    ) -> BoxFuture<'static, Result<Accepted, Refusal>>;
    fn stop(&self, job: &str) -> bool;
    /// Events with `seq > after`, in batches no closer than 80 ms apart (a
    /// batch holding `Done` is delivered at once). Ends after the batch
    /// holding `Done`.
    fn follow(&self, job: &str, after: u64) -> Result<BoxStream<'static, Vec<ChatEvent>>, Refusal>;
    fn rename(
        &self,
        thread: &str,
        title: &str,
    ) -> BoxFuture<'static, Result<ThreadSummary, Refusal>>;
    fn pin(&self, thread: &str, choice: &str) -> BoxFuture<'static, Result<(), Refusal>>;
    fn local_runtime(&self) -> BoxFuture<'static, LocalRuntime>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn() -> ChatTurn {
        ChatTurn {
            id: "0123456789ab".into(),
            ts: 1_790_000_000.25,
            role: Role::Assistant,
            text: "An answer.".into(),
            tools: vec![],
            facts: vec![Fact {
                label: "l".into(),
                rendered: "r".into(),
                as_of: "".into(),
                note: "".into(),
            }],
            unsupported: vec!["12".into()],
            provider: "ollama:qwen3".into(),
            error: "".into(),
            constrained: false,
            truncated: false,
            cancelled: true,
            prompt_tokens: None,
            completion_tokens: Some(7),
        }
    }

    #[test]
    fn chat_events_serialize_with_a_flat_type_tag_and_seq() {
        let event = ChatEvent {
            seq: 4,
            kind: ChatEventKind::Stage {
                stage: Stage::Writing,
                detail: "writing the answer".into(),
            },
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"seq": 4, "type": "stage", "stage": "writing", "detail": "writing the answer"})
        );
        let back: ChatEvent = serde_json::from_value(json).unwrap();
        assert_eq!(back, event);
    }

    #[test]
    fn every_kind_of_chat_event_round_trips_with_each_key_once() {
        let kinds = vec![
            ChatEventKind::Stage {
                stage: Stage::Thinking,
                detail: "".into(),
            },
            ChatEventKind::Delta { text: "t".into() },
            ChatEventKind::Turn {
                turn: Box::new(turn()),
                saved: true,
            },
            ChatEventKind::Error {
                message: "That model is not ready.".into(),
                turn: Box::new(turn()),
                saved: false,
            },
            ChatEventKind::Done,
        ];
        let names = ["stage", "delta", "turn", "error", "done"];
        for (index, (kind, name)) in kinds.into_iter().zip(names).enumerate() {
            let event = ChatEvent {
                seq: index as u64 + 1,
                kind,
            };
            let text = serde_json::to_string(&event).unwrap();
            let back: ChatEvent =
                serde_json::from_str(&text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(back, event, "{text}");
            assert_eq!(text.matches("\"seq\":").count(), 1, "{text}");
            assert_eq!(text.matches("\"type\":").count(), 1, "{text}");
            assert!(text.contains(&format!("\"type\":\"{name}\"")), "{text}");
        }
    }

    #[test]
    fn an_unreported_token_count_is_null_never_zero() {
        let json = serde_json::to_value(turn()).unwrap();
        assert_eq!(json["prompt_tokens"], serde_json::Value::Null);
        assert_eq!(json["completion_tokens"], 7);
        assert_eq!(json["role"], "assistant");
        let back: ChatTurn = serde_json::from_value(json).unwrap();
        assert_eq!(back.prompt_tokens, None);
    }

    #[test]
    fn thread_ids_are_the_webs_accepted_ids() {
        for good in ["a", "0123456789ab", "A_b-9", &"x".repeat(64)] {
            assert!(is_thread_id(good), "{good}");
        }
        for bad in [
            "",
            &"x".repeat(65),
            "a/b",
            "a\\b",
            "a.b",
            "..",
            "a b",
            "é",
            "a\0",
            "../index",
        ] {
            assert!(!is_thread_id(bad), "{bad:?}");
        }
    }

    #[test]
    fn job_ids_are_thirty_two_lowercase_hex_characters() {
        assert!(is_job_id("0123456789abcdef0123456789abcdef"));
        assert!(!is_job_id("0123456789ABCDEF0123456789abcdef"));
        assert!(!is_job_id("0123456789abcdef0123456789abcde"));
        assert!(!is_job_id("0123456789abcdef0123456789abcdef0"));
        assert!(!is_job_id("0123456789abcdef0123456789abcdeg"));
    }

    #[test]
    fn locality_and_runtime_state_use_the_webs_words() {
        let choice = ChatChoice {
            id: "local".into(),
            label: "Local".into(),
            detail: "This machine's Ollama runtime (qwen3).".into(),
            locality: Locality::Local,
            ready: true,
            refusal: None,
        };
        let json = serde_json::to_value(&choice).unwrap();
        assert_eq!(json["locality"], "local");
        assert_eq!(
            serde_json::to_value(RuntimeState::NoModel).unwrap(),
            "no_model"
        );
    }
}
