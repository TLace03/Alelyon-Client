//! The agent's browser: computer use on the web, as a person uses it (the
//! direction of 2026-10-08: connections are services the agent
//! operates "like a normal user would", not through their APIs). Not a port.
//!
//! - [`launch`]: Microsoft Edge (or Google Chrome) with a profile of its own
//!   under Lattice's state, in a Job Object, its DevTools on two pipes it
//!   inherits (`--remote-debugging-pipe`), so no port listens.
//! - [`pipe`] and [`cdp`]: the one connection to it (the pipes' NUL-ended
//!   frames, and the DevTools protocol over them), with no dependency added;
//!   [`ws`], a WebSocket client, carries the same protocol to a browser whose
//!   DevTools listen on a loopback port instead.
//! - [`session`]: [`AgentBrowser`], its page and the agent's actions: look
//!   (a screenshot), open an address, click at a point, type, press a key,
//!   scroll, go back. Each ends with a screenshot the model sees next.
//! - [`policy`]: what it may open (the public web, never this PC or its
//!   network), what an action's effect is, and which effects ask first.
//! - [`preview`]: a browser of its own with no network, where the reader
//!   previews a page or a picture the agent saved (an artifact).
//!
//! What it never does: type into a password, payment card or another site's
//! field (the reader signs in and pays), download a file, run the agent's
//! script, or open a local or private address. What it asks before: an action
//! whose effect another person will see, that moves money, deletes something,
//! or changes an account (the charter's rules), in the core's own dialog.
//!
//! The browser is switched on by the reader ([`prefs`]); until then no turn
//! offers its tools, and nothing starts it.

pub mod cdp;
pub mod launch;
pub mod pipe;
pub mod policy;
pub mod preview;
pub mod session;
#[cfg(test)]
pub(crate) mod tests;
pub mod ws;

pub use session::{AgentBrowser, BrowserConfig, BrowserStatus, Shot};

/// The reader's switch for the agent's browser: `<native>/chat/browser.json`,
/// `{"v": 1, "on": true}`. Off when the file is missing or unreadable.
pub mod prefs {
    use serde_json::{Value, json};

    use crate::fsx;
    use crate::state::StateRoot;

    pub fn file(state: &StateRoot) -> std::path::PathBuf {
        state.native_chat_dir().join("browser.json")
    }

    /// Is the agent's browser switched on?
    pub fn on(state: &StateRoot) -> bool {
        std::fs::read(file(state))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
            .is_some_and(|value| value.get("on").and_then(Value::as_bool) == Some(true))
    }

    /// Switch it on or off (the reader's own action).
    pub fn set(state: &StateRoot, on: bool) -> Result<(), String> {
        let path = file(state);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|_| "Lattice could not make its settings folder.".to_owned())?;
        }
        let bytes = serde_json::to_vec_pretty(&json!({"v": 1, "on": on})).unwrap_or_default();
        fsx::atomic_write(&path, &bytes)
            .map_err(|_| "The browser's setting could not be saved.".to_owned())
    }
}
