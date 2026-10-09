//! T3: the secret tripwire over every item an agent turn's model is sent
//! (the chat core's spec §10.2 T3, §9.5; the `secrets_stay_local`
//! pattern). Not a port.
//!
//! Every agent turn's model is wrapped in [`TripwireModel`] unless its target
//! is **affirmatively** local (fail closed; the caller decides). Before each
//! `stream(request)` the wrapper checks **every string in every item** of the
//! request with `secrets::looks_like_secret`:
//! - `User` text (rules, steers, replayed messages);
//! - `Assistant` text and the arguments of each of its tool calls (they hold
//!   file bytes: `edit_file`'s `old_string`/`new_string`, `write_file`'s
//!   `content`);
//! - `ToolResult` output.
//!
//! A match is withheld **per item**: the string is replaced by
//! [`WITHHELD`] (a tool call's arguments by a JSON object holding it), and
//! for a tool result or a tool call the call's id is reported once through
//! the [`Withheld`] port, never with what matched. Results are cached by the
//! SHA-256 of the string, so a long conversation is not scanned again at
//! every model call. Because the wrapper sits on the model, it covers
//! replayed items, Continue and steers alike.
//!
//! **Send this once** ([`Tripwire::release`]) lets the outputs withheld so
//! far for one call through, after the core's native `ReleaseWithheld`
//! confirmation; a new match (other content) is withheld again.
//!
//! Nothing here prints, logs or keeps raw text beyond the running turn.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

use futures::stream::BoxStream;
use lattice_agents::model::{
    GenerationTrace, InputItem, Model, ModelError, ModelEvent, ModelRequest,
};
use serde_json::Value;

use crate::secrets::looks_like_secret;
use crate::sha::sha256_hex;

/// What the model reads in place of a withheld string.
pub const WITHHELD: &str = "Withheld: this output looks like it contains a secret.";

/// Where a withheld call is reported (the conversation's `Withheld` event).
pub trait Withheld: Send + Sync {
    fn withheld(&self, call_id: &str);
}

#[derive(Default)]
struct State {
    /// SHA-256 of a string -> whether it looks like a secret.
    verdicts: HashMap<String, bool>,
    /// (call, SHA-256) withheld and reported.
    reported: HashSet<(String, String)>,
    /// (call, SHA-256) the reader released.
    released: HashSet<(String, String)>,
}

/// One conversation's tripwire: its cache, what it withheld and what the
/// reader released.
pub struct Tripwire {
    state: Mutex<State>,
    report: Arc<dyn Withheld>,
}

fn arguments_withheld() -> String {
    serde_json::json!({ "withheld": WITHHELD }).to_string()
}

impl Tripwire {
    pub fn new(report: Arc<dyn Withheld>) -> Self {
        Self {
            state: Mutex::default(),
            report,
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn looks(&self, text: &str) -> (bool, String) {
        let sha = sha256_hex(text.as_bytes());
        if let Some(verdict) = self.state().verdicts.get(&sha) {
            return (*verdict, sha);
        }
        let verdict = looks_like_secret(text);
        self.state().verdicts.insert(sha.clone(), verdict);
        (verdict, sha)
    }

    /// Whether `text` of `call` must be withheld; reports it once.
    fn withhold(&self, call: Option<&str>, text: &str) -> bool {
        let (looks, sha) = self.looks(text);
        if !looks {
            return false;
        }
        let Some(call) = call else {
            return true;
        };
        let key = (call.to_owned(), sha);
        let report = {
            let mut state = self.state();
            if state.released.contains(&key) {
                return false;
            }
            state.reported.insert(key)
        };
        if report {
            self.report.withheld(call);
        }
        true
    }

    /// The request as the model may see it.
    pub fn screen(&self, mut request: ModelRequest) -> ModelRequest {
        for item in &mut request.input {
            match item {
                // A screenshot's words are screened as any user item's; the
                // image itself is what the agent's browser showed, which the
                // reader agreed to send with the conversation (T1).
                InputItem::User(text) | InputItem::UserImages { text, .. } => {
                    if self.withhold(None, text) {
                        *text = WITHHELD.to_owned();
                    }
                }
                InputItem::Assistant { text, tool_calls } => {
                    if let Some(words) = text
                        && self.withhold(None, words)
                    {
                        *words = WITHHELD.to_owned();
                    }
                    for call in tool_calls {
                        if self.withhold(Some(&call.call_id), &call.arguments) {
                            call.arguments = arguments_withheld();
                        }
                    }
                }
                InputItem::ToolResult { call_id, output } => {
                    if self.withhold(Some(call_id), output) {
                        *output = WITHHELD.to_owned();
                    }
                }
            }
        }
        request
    }

    /// The calls whose output is withheld now (and not released).
    pub fn withheld_calls(&self) -> Vec<String> {
        let state = self.state();
        let mut calls: Vec<String> = state
            .reported
            .iter()
            .filter(|key| !state.released.contains(*key))
            .map(|(call, _)| call.clone())
            .collect();
        calls.sort();
        calls.dedup();
        calls
    }

    /// "Send this once": the outputs of `call` withheld so far go through
    /// from now on. Call only after the reader's native confirmation.
    /// `false` when nothing of that call is withheld.
    pub fn release(&self, call: &str) -> bool {
        let mut state = self.state();
        let keys: Vec<(String, String)> = state
            .reported
            .iter()
            .filter(|(id, _)| id == call)
            .cloned()
            .collect();
        let any = !keys.is_empty();
        state.released.extend(keys);
        any
    }
}

/// An agent turn's model, behind the tripwire.
pub struct TripwireModel {
    inner: Arc<dyn Model>,
    wire: Arc<Tripwire>,
}

impl TripwireModel {
    pub fn new(inner: Arc<dyn Model>, wire: Arc<Tripwire>) -> Self {
        Self { inner, wire }
    }
}

impl Model for TripwireModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn config_for_trace(&self) -> Value {
        self.inner.config_for_trace()
    }

    fn generation_trace(&self) -> GenerationTrace {
        self.inner.generation_trace()
    }

    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        self.inner.stream(self.wire.screen(request))
    }
}
