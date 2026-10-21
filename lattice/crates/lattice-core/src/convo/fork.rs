//! `staged_diff` and forking: two ways to look at a conversation's work before it lands.
//!
//! - **`staged_diff {path?}`** (a read tool, both modes, with a folder): the changes staged in this conversation and
//!   still waiting for the reader's review (pending, rebased or in conflict), as unified diffs against what each file
//!   held when it was first staged. Staged changes are not on disk, so `git_diff` never shows them; `/review` reads
//!   these first. At most [`MAX_DIFF_CHARS`] characters.
//! - **The context meter** ([`AgentChat::context_use`]): how much of the replay budget the conversation's last turn
//!   used, and how many earlier exchanges were left out; and **compacting** ([`AgentChat::compact`], `/compact`): its
//!   summary written now (with what to keep above all), the turns it covers left out of later turns, which it leads.
//! - **Forking** ([`AgentChat::fork`]): a new conversation holding a conversation's messages up to a turn, in the
//!   same folder, mode and group, titled as its fork, so the reader can try another way and keep the original. Its
//!   messages are copied; its tool steps, staged changes and checkpoints are not (they belong to the original).

use lattice_protocol::conversation::Mode;
use serde_json::Value;

use super::agent::AgentChat;
use super::turn::TurnTools;
use crate::tools::read::ToolError;

/// A conversation's context use at its last turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ContextUse {
    /// Characters replayed, and the budget they had.
    pub used: usize,
    pub limit: usize,
    /// Earlier exchanges left out (by the budget, or compacted).
    pub left_out: usize,
}

impl ContextUse {
    /// The share of the budget used, 0 to 100.
    pub fn percent(&self) -> u8 {
        (self.used.saturating_mul(100)).checked_div(self.limit).map_or(0, |p| p.min(100) as u8)
    }
}

/// The longest `staged_diff` answer, in characters.
pub const MAX_DIFF_CHARS: usize = 60_000;

/// One staged change as a unified diff.
fn unified(path: &str, base: Option<&[u8]>, new: Option<&[u8]>) -> String {
    let text = |bytes: Option<&[u8]>| bytes.map(|b| String::from_utf8_lossy(b).into_owned());
    match (text(base), text(new)) {
        (Some(old), Some(now)) => similar::TextDiff::from_lines(&old, &now)
            .unified_diff()
            .context_radius(3)
            .header(&format!("a/{path}"), &format!("b/{path}"))
            .to_string(),
        (None, Some(now)) => {
            let mut out = format!("--- /dev/null\n+++ b/{path}\n");
            for line in now.lines() {
                out.push_str(&format!("+{line}\n"));
            }
            out
        }
        (Some(_), None) => format!("--- a/{path}\n+++ /dev/null\n(the file is deleted)\n"),
        (None, None) => format!("{path}: changed (its bytes are not text read whole)\n"),
    }
}

/// `staged_diff`, on the blocking pool.
pub(crate) fn staged_diff(tools: &TurnTools, args: &Value) -> Result<String, ToolError> {
    let only = args.get("path").and_then(Value::as_str).map(|p| p.trim().trim_start_matches("./").replace('\\', "/"));
    let mut out = String::new();
    let mut shown = 0;
    for change in tools.staging.changes() {
        if !crate::staging::is_waiting(&change.state) {
            continue;
        }
        if only.as_deref().is_some_and(|p| !p.is_empty() && change.path != p) {
            continue;
        }
        let Some(snapshot) = tools.staging.snapshot(&change.id) else { continue };
        out.push_str(&unified(&change.path, snapshot.base.as_deref(), snapshot.new.as_deref()));
        out.push('\n');
        shown += 1;
    }
    if shown == 0 {
        return Ok(match only {
            Some(p) if !p.is_empty() => format!("Nothing staged in this conversation is waiting at {p}."),
            _ => "Nothing staged in this conversation is waiting for review.".to_owned(),
        });
    }
    if out.chars().count() > MAX_DIFF_CHARS {
        out = out.chars().take(MAX_DIFF_CHARS).collect();
        out.push_str("\n[Cut at 60,000 characters: ask for one path at a time.]");
    }
    Ok(format!("{shown} staged change{} waiting for review:\n\n{out}", if shown == 1 { "" } else { "s" }))
}

impl AgentChat {
    /// The conversation `id`'s context use at its last turn in this process (`None`: no turn yet).
    pub fn context_use(&self, id: &str) -> Option<ContextUse> {
        self.inner.context.lock().unwrap_or_else(|p| p.into_inner()).get(id).copied()
    }

    /// Compact the conversation `id` (`/compact`): its summary written now, from every turn so far, keeping `focus`
    /// above all; later turns leave those turns out and start from the summary. The summary's text.
    pub async fn compact(&self, id: &str, focus: Option<String>) -> Result<String, String> {
        let store = self.inner.config.store.clone();
        let turns = store.load(id);
        if turns.is_empty() {
            return Err("There is nothing to compact yet.".to_owned());
        }
        let row = super::binding::listed_row(store.as_ref(), id).map_err(|_| "That conversation is no longer listed.".to_owned())?;
        let Some((model, label, lease)) = super::turn::summary_model(&self.inner, id, &row.pinned_provider).await else {
            return Err("No model could write the summary: pick one for summaries, or start Lattice's own.".to_owned());
        };
        let text = super::summaries::write_focused(model.as_ref(), &row.title, &turns, focus.as_deref()).await;
        drop(lease);
        let text = text.ok_or("The model wrote no summary; nothing was compacted.")?;
        let summary = super::summaries::Summary {
            text: text.clone(),
            turns: turns.len(),
            at: self.inner.now(),
            by: label,
            compacted_turns: turns.len(),
        };
        self.inner.summaries.keep(id, summary).map_err(|e| e.to_string())?;
        Ok(text)
    }

    /// A new conversation holding the conversation `id`'s first `turns` messages (its user's and its answers, those
    /// not set aside by an edit), in the same folder, mode and group, titled as a fork. Its id.
    pub fn fork(&self, id: &str, turns: usize) -> Result<String, String> {
        use crate::chat::transcript::NewTurn;
        let store = self.inner.config.store.as_ref();
        let kept: Vec<_> = store.load(id).into_iter().filter(|t| !t.superseded).take(turns).collect();
        let Some(first) = kept.first().filter(|t| t.role == "user") else {
            return Err("There is no message to fork from there.".to_owned());
        };
        let row = super::binding::listed_row(store, id).map_err(|_| "That conversation is no longer listed.".to_owned())?;
        let (forked, _) = store.first_message(&first.text, &row.pinned_provider).map_err(|e| e.to_string())?;
        for turn in &kept[1..] {
            let new = if turn.role == "user" {
                NewTurn::user(turn.text.clone())
            } else {
                NewTurn::assistant(turn.text.clone(), turn.provider.clone())
            };
            store.append(&forked.id, new).map_err(|e| e.to_string())?;
        }
        let title = if row.title.is_empty() { "Fork".to_owned() } else { format!("{} (fork)", row.title) };
        let _ = store.rename(&forked.id, &title);
        // Its folder and mode, as the original's.
        let meta = self.inner.sidecars.read_meta(id).ok().flatten();
        let new_meta = super::sidecar::NewMeta {
            workspace: meta.as_ref().and_then(|m| m.workspace.clone()),
            mode: meta.as_ref().map_or(Mode::Agent, |m| m.mode),
            origin: lattice_protocol::conversation::Origin::Native,
        };
        super::binding::bind_for_writing(store, &self.inner.sidecars, &forked.id, new_meta)
            .map_err(|_| "The fork was made but its folder could not be set.".to_owned())?;
        // In the original's group.
        if let Some(group) = self.inner.groups.of(id) {
            let _ = self.group_add(&forked.id, &super::groups::Placement::Group(group.id), group.project.as_deref());
        }
        Ok(forked.id)
    }
}
