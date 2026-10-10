//! Hooks (tool parity with Claude Code's hooks, and Cursor's, Gemini CLI's and
//! Antigravity's): commands the reader set to run when Lattice's agent does
//! something, which can stop it, change what it does, or tell it more.
//!
//! - **Events:** before a tool runs ([`Event::PreToolUse`]: block it, or change
//!   its arguments), after it ran ([`Event::PostToolUse`]: add to what the
//!   agent reads), when the reader sends a message ([`Event::UserPromptSubmit`]:
//!   block it, or add context), when the agent is about to stop
//!   ([`Event::Stop`]: have it go on, with a reason), and when a chat starts
//!   ([`Event::SessionStart`]: add context). Each runs in Lattice's own agent
//!   turns (Ask and Agent mode), not in plain answers or the labs' agents' turns.
//! - **Where they are read** ([`sources`]): Lattice's own file
//!   (`<native>/hooks.json`), the plugins switched on, Claude Code's settings
//!   (`~/.claude/settings.json`), Cursor's (`~/.cursor/hooks.json`), Gemini
//!   CLI's (`~/.gemini/settings.json`) and Antigravity's
//!   (`~/.gemini/config/hooks.json`); and in a trusted folder its
//!   `.lattice/hooks.json`, `.claude/settings.json` and
//!   `.claude/settings.local.json`, `.cursor/hooks.json`,
//!   `.gemini/settings.json` and `.agents/hooks.json`. Each file is read in its
//!   own tool's format ([`formats`]) and its events are mapped onto these five;
//!   an event Lattice has no equal of is listed as not run.
//! - **Asked once:** a hook runs only after the reader allowed it, in the
//!   core's own dialog, which shows where it is from, its event and its exact
//!   command ([`approvals`]). The yes is kept for that exact text: a changed
//!   command, matcher or event asks again. A no is remembered for the session.
//! - **How it runs** ([`run`]): as the tool it is from runs it. Its event is
//!   given on its standard input as that tool's JSON, the command is run by
//!   Git for Windows' bash when it is installed (as Claude Code runs hooks on
//!   Windows), else by `cmd.exe`, in the folder, with the safe environment and
//!   the tool's project variables, and ended at its timeout. Exit code 2, or a
//!   decision in its JSON answer, blocks; anything else that fails is noted and
//!   the agent goes on.

pub mod approvals;
pub mod formats;
pub mod run;
pub mod sources;
#[cfg(test)]
mod tests;

use std::path::PathBuf;
use std::time::Duration;

use regex::Regex;

/// The events Lattice runs hooks at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Event {
    PreToolUse,
    PostToolUse,
    UserPromptSubmit,
    Stop,
    SessionStart,
}

impl Event {
    pub fn label(self) -> &'static str {
        match self {
            Event::PreToolUse => "before a tool runs",
            Event::PostToolUse => "after a tool ran",
            Event::UserPromptSubmit => "when you send a message",
            Event::Stop => "when the agent stops",
            Event::SessionStart => "when a chat starts",
        }
    }
}

/// Whose format a hook is written in, which decides what it is given and how
/// its answer is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// Claude Code's (and Lattice's own, and plugins').
    Claude,
    Cursor,
    /// Gemini CLI's and Antigravity's.
    Gemini,
}

/// Where a hook was read.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Lattice,
    Plugin(String),
    ClaudeCode,
    Cursor,
    GeminiCli,
    Antigravity,
    /// A trusted folder's file, by its path in the folder.
    Folder(String),
}

impl Source {
    /// Its name, for the reader.
    pub fn label(&self) -> String {
        match self {
            Source::Lattice => "Your Lattice hooks".to_owned(),
            Source::Plugin(name) => format!("Plugin {name}"),
            Source::ClaudeCode => "Claude Code's settings".to_owned(),
            Source::Cursor => "Cursor's hooks".to_owned(),
            Source::GeminiCli => "Gemini CLI's settings".to_owned(),
            Source::Antigravity => "Antigravity's hooks".to_owned(),
            Source::Folder(file) => format!("This folder's {file}"),
        }
    }

    /// A stable key, for its approvals.
    pub fn key(&self) -> String {
        match self {
            Source::Lattice => "lattice".to_owned(),
            Source::Plugin(name) => format!("plugin:{name}"),
            Source::ClaudeCode => "claude".to_owned(),
            Source::Cursor => "cursor".to_owned(),
            Source::GeminiCli => "gemini".to_owned(),
            Source::Antigravity => "antigravity".to_owned(),
            Source::Folder(file) => format!("folder:{file}"),
        }
    }
}

/// Which tools a tool hook is for.
#[derive(Clone, Debug)]
pub enum Matcher {
    All,
    /// A pattern over the tool's name, in the hook's own tool's names.
    Pattern(String, Regex),
    /// Only the shell (Cursor's `beforeShellExecution`).
    Shell,
    /// Only MCP tools (Cursor's `beforeMCPExecution`).
    Mcp,
    /// Only reading a file (Cursor's `beforeReadFile`).
    ReadFile,
    /// Only editing or writing a file (Cursor's `afterFileEdit`).
    FileEdit,
}

impl Matcher {
    /// A tool-name pattern as hook files write it: empty or `*` for every
    /// tool, else a regular expression the whole name matches.
    pub fn pattern(text: &str) -> Option<Matcher> {
        let text = text.trim();
        if text.is_empty() || text == "*" {
            return Some(Matcher::All);
        }
        Regex::new(&format!("^(?:{text})$")).ok().map(|re| Matcher::Pattern(text.to_owned(), re))
    }

    /// What it says, for the reader and the approval's text.
    pub fn text(&self) -> String {
        match self {
            Matcher::All => "*".to_owned(),
            Matcher::Pattern(text, _) => text.clone(),
            Matcher::Shell => "shell".to_owned(),
            Matcher::Mcp => "MCP".to_owned(),
            Matcher::ReadFile => "read".to_owned(),
            Matcher::FileEdit => "edit".to_owned(),
        }
    }
}

impl PartialEq for Matcher {
    fn eq(&self, other: &Matcher) -> bool {
        self.text() == other.text()
    }
}

/// One hook, as read.
#[derive(Clone, Debug, PartialEq)]
pub struct Hook {
    pub source: Source,
    pub family: Family,
    pub event: Event,
    /// The event's name in its own file (`beforeShellExecution`, `BeforeTool`).
    pub native: String,
    pub matcher: Matcher,
    /// The command line, as written.
    pub command: String,
    pub timeout: Duration,
    /// The file it was read from.
    pub file: PathBuf,
    /// A plugin's folder, for `${CLAUDE_PLUGIN_ROOT}`.
    pub plugin_root: Option<PathBuf>,
}

impl Hook {
    /// What its approval is kept for: its source, event, matcher and exact command.
    pub fn digest(&self) -> String {
        let text = format!(
            "{}\n{}\n{}\n{}\n{}",
            self.source.key(),
            self.native,
            self.matcher.text(),
            self.command,
            self.file.display()
        );
        crate::sha::sha256_hex(text.as_bytes())
    }

    /// Whether it is for the tool `tool` (a Lattice tool name).
    pub fn applies(&self, tool: &str) -> bool {
        match &self.matcher {
            Matcher::All => true,
            Matcher::Pattern(_, re) => {
                re.is_match(tool) || re.is_match(&run::tool_name(self.family, tool))
            }
            Matcher::Shell => tool == "run_command",
            Matcher::Mcp => crate::mcp::names::is_model_name(tool),
            Matcher::ReadFile => tool == "read_file",
            Matcher::FileEdit => matches!(tool, "edit_file" | "write_file"),
        }
    }
}

/// A hook file that could not be read whole, or an event that is not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    pub file: PathBuf,
    pub sentence: String,
}

/// Every hook found, and what could not be used.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Found {
    pub hooks: Vec<Hook>,
    pub problems: Vec<Problem>,
}

impl Found {
    /// The hooks of `event` (for a tool event, those for `tool`).
    pub fn for_event(&self, event: Event, tool: Option<&str>) -> Vec<&Hook> {
        self.hooks
            .iter()
            .filter(|h| h.event == event && tool.is_none_or(|t| h.applies(t)))
            .collect()
    }
}
