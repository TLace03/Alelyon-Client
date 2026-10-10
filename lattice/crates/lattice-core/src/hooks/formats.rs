//! Each tool's hook file, read into [`Hook`]s.
//!
//! - **Claude Code** (Lattice's own file and plugins' `hooks/hooks.json` too):
//!   `{"hooks": {"<Event>": [{"matcher": "...", "hooks": [{"type":
//!   "command", "command": "...", "timeout": <seconds>}]}]}}`.
//! - **Cursor:** `{"version": 1, "hooks": {"<event>": [{"command": "...",
//!   "matcher": "...", "timeout": <seconds>}]}}`; `beforeShellExecution`,
//!   `beforeMCPExecution` and `beforeReadFile` are tool hooks for the shell, MCP
//!   tools and reading a file, `afterFileEdit` for editing one.
//! - **Gemini CLI:** Claude Code's shape under `hooks` in its settings, with its
//!   own event names (`BeforeTool`, `AfterTool`, `BeforeAgent`, `AfterAgent`,
//!   `SessionStart`) and timeouts in milliseconds.
//! - **Antigravity:** the same, or hooks named by the reader, each holding its
//!   events (`PreToolUse`, `PostToolUse`, `PreInvocation`) and an `enabled`
//!   switch. Its documentation does not give every event; the names above are
//!   read, and others are listed as not run.
//!
//! A hook of `"type": "prompt"` (a model judges, not a command) is listed as
//! not run. An event Lattice has no equal of is listed as not run.

use std::path::Path;
use std::time::Duration;

use serde_json::Value;

use super::{Event, Family, Hook, Matcher, Problem, Source};

/// The default timeout, as Claude Code's.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
/// The longest a hook may be given.
pub const MAX_TIMEOUT: Duration = Duration::from_secs(600);

/// What a file's event name means here, and the matcher it implies.
fn event_of(name: &str) -> Option<(Event, Option<Matcher>)> {
    Some(match name {
        "PreToolUse" | "BeforeTool" | "preToolUse" => (Event::PreToolUse, None),
        "PostToolUse" | "AfterTool" | "postToolUse" => (Event::PostToolUse, None),
        "UserPromptSubmit" | "BeforeAgent" | "beforeSubmitPrompt" | "PreInvocation" => (Event::UserPromptSubmit, None),
        "Stop" | "AfterAgent" | "stop" => (Event::Stop, None),
        "SessionStart" | "sessionStart" => (Event::SessionStart, None),
        "beforeShellExecution" => (Event::PreToolUse, Some(Matcher::Shell)),
        "afterShellExecution" => (Event::PostToolUse, Some(Matcher::Shell)),
        "beforeMCPExecution" => (Event::PreToolUse, Some(Matcher::Mcp)),
        "afterMCPExecution" => (Event::PostToolUse, Some(Matcher::Mcp)),
        "beforeReadFile" => (Event::PreToolUse, Some(Matcher::ReadFile)),
        "afterFileEdit" => (Event::PostToolUse, Some(Matcher::FileEdit)),
        _ => return None,
    })
}

/// A timeout as written: seconds for Claude Code and Cursor, milliseconds for
/// Gemini CLI and Antigravity.
fn timeout(value: Option<&Value>, family: Family) -> Duration {
    let Some(n) = value.and_then(Value::as_f64).filter(|n| n.is_finite() && *n > 0.0) else {
        return DEFAULT_TIMEOUT;
    };
    let seconds = if family == Family::Gemini { n / 1000.0 } else { n };
    Duration::from_secs_f64(seconds).min(MAX_TIMEOUT)
}

struct Reader<'a> {
    source: &'a Source,
    family: Family,
    file: &'a Path,
    plugin_root: Option<&'a Path>,
    out: Vec<Hook>,
    problems: Vec<Problem>,
}

impl Reader<'_> {
    fn problem(&mut self, sentence: String) {
        self.problems.push(Problem { file: self.file.to_path_buf(), sentence });
    }

    /// One command entry: `{"type", "command", "timeout"}`.
    fn entry(&mut self, native: &str, matcher: Matcher, event: Event, entry: &Value) {
        let kind = entry.get("type").and_then(Value::as_str).unwrap_or("command");
        if kind != "command" {
            self.problem(format!("A {native} hook of type \"{kind}\" is not run: Lattice runs command hooks."));
            return;
        }
        let Some(command) = entry.get("command").and_then(Value::as_str).map(str::trim).filter(|c| !c.is_empty()) else {
            self.problem(format!("A {native} hook has no command."));
            return;
        };
        if entry.get("enabled").and_then(Value::as_bool) == Some(false) {
            return;
        }
        self.out.push(Hook {
            source: self.source.clone(),
            family: self.family,
            event,
            native: native.to_owned(),
            matcher,
            command: command.to_owned(),
            timeout: timeout(entry.get("timeout"), self.family),
            file: self.file.to_path_buf(),
            plugin_root: self.plugin_root.map(Path::to_path_buf),
        });
    }

    /// One event's list, in Claude Code's grouped shape (`matcher` and
    /// `hooks`) or Cursor's flat one (the command entries themselves).
    fn event(&mut self, native: &str, list: &Value) {
        let Some((event, implied)) = event_of(native) else {
            self.problem(format!("{native} hooks are not run: Lattice has no such event."));
            return;
        };
        let Some(groups) = list.as_array() else {
            self.problem(format!("The {native} hooks are not a list."));
            return;
        };
        for group in groups {
            let pattern = group.get("matcher").and_then(Value::as_str).unwrap_or("");
            let matcher = match (&implied, Matcher::pattern(pattern)) {
                (Some(implied), _) => implied.clone(),
                (None, Some(m)) => m,
                (None, None) => {
                    self.problem(format!("A {native} matcher, {pattern:?}, is not a pattern Lattice can read."));
                    continue;
                }
            };
            match group.get("hooks").and_then(Value::as_array) {
                Some(entries) => {
                    for entry in entries {
                        self.entry(native, matcher.clone(), event, entry);
                    }
                }
                None => self.entry(native, matcher, event, group),
            }
        }
    }

    /// `{"<event>": [...]}`.
    fn events(&mut self, map: &serde_json::Map<String, Value>) {
        for (native, list) in map {
            if list.is_array() {
                self.event(native, list);
            }
        }
    }
}

/// The hooks of one file's JSON, in `family`'s format.
pub fn read(value: &Value, source: &Source, family: Family, file: &Path, plugin_root: Option<&Path>) -> (Vec<Hook>, Vec<Problem>) {
    let mut reader = Reader { source, family, file, plugin_root, out: Vec::new(), problems: Vec::new() };
    match value.get("hooks").and_then(Value::as_object) {
        Some(map) => {
            reader.events(map);
            // Antigravity's named hooks: `{"<name>": {"<Event>": [...], "enabled": bool}}`.
            for named in map.values().filter_map(Value::as_object) {
                if named.get("enabled").and_then(Value::as_bool) == Some(false) {
                    continue;
                }
                reader.events(named);
            }
        }
        None if value.get("hooks").is_some() => reader.problem("Its \"hooks\" is not an object.".to_owned()),
        None => {}
    }
    (reader.out, reader.problems)
}
