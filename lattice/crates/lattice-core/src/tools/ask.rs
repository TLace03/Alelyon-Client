//! `ask_question`: the agent asks the reader, and waits (the chat core's spec
//! §7.1, §7.7, §9.2, §9.5; CR1, CR2). Not a port.
//!
//! - **Arguments.** `question`: 1 to 2,000 characters, not blank. `options`: at
//!   most 6 strings of 1 to 200 characters each. Anything else is a tool error
//!   with one sentence, and nothing is asked.
//! - **Asking.** [`Questions::ask`] registers the call as pending in its
//!   conversation, lets the caller announce it (the agent chat appends the
//!   `Question` item and sends the urgent `Question` event, §11.3), and calls
//!   [`AttentionPort::attention`] with the conversation's id only (no text).
//!   The policy engine allows the call in every mode (§9.2): a question widens
//!   nothing.
//! - **Waiting.** The returned future waits for [`Questions::answer`] **with no
//!   timer** (CR2): it awaits a one-shot channel, so an idle question schedules
//!   nothing and wakes nothing (CB1). There is no timeout.
//! - **Stop.** Stop cancels the run, which drops the waiting future; dropping
//!   it takes the question out of the pending set, so a later answer is
//!   refused ("That question is no longer waiting.") and reaches no one.
//!   Closing the conversation does the same through [`Questions::withdraw`].
//! - **The result** is "The user answered: <text>", at most 8,000 characters.
//! - **An answer is the reader's text, nothing more** (§9.5): it is never
//!   parsed as a decision about any call, and it resolves only the one call it
//!   names, in the one conversation it names.
//!
//! Nothing here prints, logs or keeps the answer beyond handing it to the
//! waiting call.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Mutex, MutexGuard};

use serde::Deserialize;
use tokio::sync::oneshot;

use super::read::ToolError;
use crate::ports::AttentionPort;

/// The longest question, in characters.
pub const MAX_QUESTION_CHARS: usize = 2_000;
/// The most options a question may offer.
pub const MAX_OPTIONS: usize = 6;
/// The longest option, in characters.
pub const MAX_OPTION_CHARS: usize = 200;
/// The longest result the model reads, in characters.
pub const MAX_RESULT_CHARS: usize = 8_000;
/// What the model reads before the answer.
pub const ANSWERED: &str = "The user answered: ";
/// An answer to a call that is not waiting (never asked, answered, or
/// cancelled).
pub const NOT_WAITING: &str = "That question is no longer waiting.";
/// What the model reads when the question was withdrawn without an answer
/// (the conversation closed).
pub const WITHDRAWN: &str = "The question was withdrawn before the user answered.";

/// `ask_question`'s arguments.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
pub struct AskQuestionArgs {
    pub question: String,
    #[serde(default)]
    pub options: Vec<String>,
}

/// Check the arguments' bounds (§7.7).
pub fn check(args: &AskQuestionArgs) -> Result<(), ToolError> {
    if args.question.trim().is_empty() {
        return Err(ToolError::new("A question needs at least one character."));
    }
    if args.question.chars().count() > MAX_QUESTION_CHARS {
        return Err(ToolError::new(
            "A question can be at most 2,000 characters.",
        ));
    }
    if args.options.len() > MAX_OPTIONS {
        return Err(ToolError::new("A question can offer at most 6 options."));
    }
    for option in &args.options {
        if option.trim().is_empty() {
            return Err(ToolError::new("An option needs at least one character."));
        }
        if option.chars().count() > MAX_OPTION_CHARS {
            return Err(ToolError::new("An option can be at most 200 characters."));
        }
    }
    Ok(())
}

/// The result the model reads for an answer: "The user answered: <text>",
/// cut to [`MAX_RESULT_CHARS`] characters.
pub fn result_text(answer: &str) -> String {
    ANSWERED
        .chars()
        .chain(answer.chars())
        .take(MAX_RESULT_CHARS)
        .collect()
}

/// A question as it is announced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Asked {
    pub conversation: String,
    pub call_id: String,
    pub question: String,
    pub options: Vec<String>,
}

/// Why an answer was not taken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnswerError {
    /// No such call waits in that conversation.
    NotWaiting,
}

impl AnswerError {
    pub fn sentence(&self) -> &'static str {
        match self {
            AnswerError::NotWaiting => NOT_WAITING,
        }
    }
}

type Key = (String, String);
type Pending = BTreeMap<Key, oneshot::Sender<String>>;

/// The questions waiting in this process, by (conversation, call).
#[derive(Clone)]
pub struct Questions {
    pending: Arc<Mutex<Pending>>,
    attention: Arc<dyn AttentionPort>,
}

fn lock(pending: &Mutex<Pending>) -> MutexGuard<'_, Pending> {
    pending
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Takes the question out of the pending set when the waiting call ends,
/// however it ends (answered, cancelled, dropped).
struct Waiting {
    pending: Arc<Mutex<Pending>>,
    key: Key,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        lock(&self.pending).remove(&self.key);
    }
}

impl Questions {
    pub fn new(attention: Arc<dyn AttentionPort>) -> Self {
        Self {
            pending: Arc::default(),
            attention,
        }
    }

    /// Ask: check the arguments, register the call as pending, `announce` it
    /// (the caller records and sends it), then ask for the reader's
    /// attention. The future resolves with the model's result when the reader
    /// answers; it has no timer. A call id already waiting in the
    /// conversation is refused.
    pub fn ask(
        &self,
        conversation: &str,
        call_id: &str,
        args: AskQuestionArgs,
        announce: impl FnOnce(&Asked),
    ) -> Result<impl Future<Output = Result<String, ToolError>> + Send + 'static, ToolError> {
        check(&args)?;
        let key = (conversation.to_owned(), call_id.to_owned());
        let (sender, receiver) = oneshot::channel();
        {
            let mut pending = lock(&self.pending);
            if pending.contains_key(&key) {
                return Err(ToolError::new("That question is already waiting."));
            }
            pending.insert(key.clone(), sender);
        }
        let waiting = Waiting {
            pending: Arc::clone(&self.pending),
            key,
        };
        announce(&Asked {
            conversation: conversation.to_owned(),
            call_id: call_id.to_owned(),
            question: args.question,
            options: args.options,
        });
        self.attention.attention(conversation);
        Ok(async move {
            let _waiting = waiting;
            match receiver.await {
                Ok(answer) => Ok(result_text(&answer)),
                Err(_) => Err(ToolError::new(WITHDRAWN)),
            }
        })
    }

    /// The reader's answer to `call` in `conversation`. Memory only; returns
    /// at once.
    pub fn answer(&self, conversation: &str, call: &str, text: String) -> Result<(), AnswerError> {
        let sender = lock(&self.pending)
            .remove(&(conversation.to_owned(), call.to_owned()))
            .ok_or(AnswerError::NotWaiting)?;
        sender.send(text).map_err(|_| AnswerError::NotWaiting)
    }

    /// The calls waiting for an answer in `conversation`.
    pub fn waiting(&self, conversation: &str) -> Vec<String> {
        lock(&self.pending)
            .keys()
            .filter(|(id, _)| id == conversation)
            .map(|(_, call)| call.clone())
            .collect()
    }

    /// Withdraw every question of `conversation` (it closed): each waiting
    /// call ends with [`WITHDRAWN`].
    pub fn withdraw(&self, conversation: &str) {
        lock(&self.pending).retain(|(id, _), _| id != conversation);
    }
}
