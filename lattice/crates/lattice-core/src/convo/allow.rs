//! Allow all: a project's switch that lets the agent finish work without the
//! reader. While a conversation's project
//! has it on:
//!
//! - commands run, MCP tools are called, and the agent's browser and desktop
//!   act, without asking (`crate::policy::Standing::allow_all`);
//! - the agent's file changes are kept as they are staged, authority files
//!   included (with the folder's checkpoint, as every Keep has);
//! - a stack decision takes the recommended option, and a question is answered
//!   by the agent's own judgement, which it states;
//! - still asking: whatever moves money (a purchase in the browser or on the
//!   desktop). The agent never types a password or a card number either way.
//!   The refusals stand too: Ask mode, an untrusted folder, a refused path.
//!
//! Everything allowed this way is logged ([`AutoLog`], `<native>/chat/
//! auto_decisions.jsonl`, the newest [`MAX_KEPT`]), so the reader can see what
//! was done without them and change what they would not have chosen.

use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use super::agent::Inner;
use crate::state::StateRoot;

/// The most entries kept; the oldest go first.
pub const MAX_KEPT: usize = 2_000;
/// The longest description of what was done, in characters.
pub const MAX_WHAT_CHARS: usize = 300;

/// Whether conversation `id`'s project allows all actions.
pub(crate) fn allow_all(inner: &Inner, id: &str) -> bool {
    inner.projects.of_chat(id).is_some_and(|p| p.allow_all && !p.archived)
}

/// What a question's call answers under allow all.
pub const QUESTION_UNASKED: &str = "Allow all is on for this project, so the user is not asked: choose the most reasonable answer yourself, say in your reply what you chose and why, and go on.";

/// What was done without the reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Command,
    Keep,
    Mcp,
    Browser,
    Desktop,
    Choice,
    Question,
}

impl Kind {
    pub fn words(self) -> &'static str {
        match self {
            Kind::Command => "Ran a command",
            Kind::Keep => "Kept a change",
            Kind::Mcp => "Used a tool",
            Kind::Browser => "Acted in the browser",
            Kind::Desktop => "Acted on the desktop",
            Kind::Choice => "Chose for a decision",
            Kind::Question => "Answered its own question",
        }
    }
}

/// One thing done without the reader.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub at: f64,
    pub project: String,
    pub conversation: String,
    pub kind: Kind,
    /// What, in a line (a command's text, a file's path, the option chosen).
    pub what: String,
}

/// The log.
pub struct AutoLog {
    path: PathBuf,
    lock: Mutex<()>,
}

impl AutoLog {
    pub fn new(state: &StateRoot) -> Self {
        Self { path: state.native_chat_dir().join("auto_decisions.jsonl"), lock: Mutex::new(()) }
    }

    /// Every entry, oldest first.
    pub fn all(&self) -> Vec<Entry> {
        let Ok(text) = std::fs::read_to_string(&self.path) else { return Vec::new() };
        text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
    }

    /// The entries of `project` (all projects when `None`), newest first.
    pub fn of(&self, project: Option<&str>) -> Vec<Entry> {
        let mut out: Vec<Entry> = self.all().into_iter().filter(|e| project.is_none_or(|p| e.project == p)).collect();
        out.reverse();
        out
    }

    /// Add an entry; past [`MAX_KEPT`] the oldest go. A failure costs only the entry.
    pub fn add(&self, entry: &Entry) {
        let _held = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let mut all = self.all();
        all.push(entry.clone());
        let start = all.len().saturating_sub(MAX_KEPT);
        let text: String = all[start..].iter().filter_map(|e| serde_json::to_string(e).ok()).map(|l| l + "\n").collect();
        let _ = crate::fsx::atomic_write(&self.path, text.as_bytes());
    }
}

/// Log that `kind` was done in conversation `id` without the reader.
pub(crate) fn log(inner: &Inner, id: &str, kind: Kind, what: &str) {
    let project = inner.projects.of_chat(id).map(|p| p.id).unwrap_or_default();
    let what: String = crate::secrets::redact(&what.split_whitespace().collect::<Vec<_>>().join(" ")).chars().take(MAX_WHAT_CHARS).collect();
    inner.auto_log.add(&Entry { at: inner.now(), project, conversation: id.to_owned(), kind, what });
}
