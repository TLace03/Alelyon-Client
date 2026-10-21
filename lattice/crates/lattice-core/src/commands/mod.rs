//! Commands ("plugins" of 2026-10-08, their first part): prompt
//! templates the reader runs from the composer as `/name`, in the shape
//! Claude Code and Cursor give them. Not a port.
//!
//! - **Sources, in order:** the reader's own `<native>/commands/*.md`; each
//!   plugin's switched on (`crate::plugins`), its `commands/*.md` as
//!   `/<plugin>:<name>`, read only inside its folder; then, only for a
//!   trusted folder (FT3), `.lattice/commands/*.md`,
//!   `.claude/commands/*.md` and `.cursor/commands/*.md`, read through the
//!   path rules (WP1-WP11) as rules files are. A file one folder down is in
//!   that folder's namespace: `frontend/test.md` is `/frontend:test`.
//! - **Front matter:** `description` and `argument-hint`. Any other key
//!   (`allowed-tools`, `model` and the rest) is read and not honoured: a
//!   command is text for the composer, so it grants nothing and picks no
//!   model, and the menu names the keys set aside.
//! - **Expansion** ([`expand`]): `$ARGUMENTS` is what follows the name, and
//!   `$1` to `$9` are its words (double quotes group words). A body that uses
//!   neither gets the arguments after a blank line. Nothing else in it is
//!   acted on: a line starting `!` runs nothing and an `@path` reads nothing.
//!   The window puts the result in the composer, where the reader sees and
//!   can change it before sending.
//! - **Built in** ([`BUILT_IN`]): `/review`, the review Claude Code and Codex
//!   offer: a careful review of the changes staged in the conversation
//!   (`staged_diff`, first when there are any), the uncommitted changes, the
//!   branch against a base, or a commit (`git_diff`), findings by severity,
//!   nothing changed. It is text like any command; one of the reader's own (or a
//!   plugin's or the folder's) of the same name is listed in its place.
//! - **Bounds:** at most 64 KiB a command and [`MAX_COMMANDS`] commands.

use std::io::Read;
use std::path::{Path, PathBuf};

use lattice_protocol::conversation::TrustState;
use lattice_sys::fs::Access;

use crate::localfs::{LinkRule, WalkError, open_walk};
use crate::state::StateRoot;
use crate::workspace::paths::{PathRules, Want};
use crate::workspace::rules::{folder_dir, names_in};

/// At most this much of one command file.
pub const MAX_COMMAND: u64 = 64 * 1024;
/// At most this many commands are listed.
pub const MAX_COMMANDS: usize = 200;
/// The front matter's keys a command may use.
const KEYS: [&str; 2] = ["description", "argument-hint"];
/// A folder's command folders, in order.
pub const FOLDER_DIRS: [&str; 3] = [".lattice/commands", ".claude/commands", ".cursor/commands"];

/// Where a command came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// `<native>/commands`: the reader's own.
    User,
    /// A file of the folder (after trust).
    Folder,
    /// A plugin's, switched on.
    Plugin,
    /// Lattice's own ([`BUILT_IN`]).
    BuiltIn,
}

/// `/review`'s text.
pub const REVIEW: &str = "Review these code changes as a careful reviewer would, and change nothing: this is a review, not a fix.

What to review: $ARGUMENTS
- Nothing named: the changes staged in this conversation and waiting for the user's review (staged_diff), when there are any; else the uncommitted changes (git_diff with no argument).
- \"staged\": only the changes staged in this conversation (staged_diff).
- \"branch\", or \"against <branch>\": everything since this branch left that branch (git_diff with base; main when none is named).
- A commit id: that commit (git_diff with commit).
- Anything else says what to look at most closely, in the uncommitted changes.

How:
1. Read the change with staged_diff or git_diff, and the code around it with read_file and grep (they see the staged versions), as far as you need to judge it.
2. Look for what would go wrong: a bug and the input that triggers it, a missed edge case, a security problem (a secret, injection, unchecked input), lost data, a race, changed behaviour no test covers, code that does not do what its name or comment says. Leave style alone unless it hides a bug.
3. Report each finding, most severe first, as: **Critical**, **High**, **Medium** or **Low**, then `path:line`, then what is wrong in one sentence and the input or sequence that makes it fail.
4. End with a one-line verdict. When you find nothing worth reporting, say so plainly; do not invent findings.

Do not edit, stage, run or commit anything for this review.";

/// Lattice's own commands: (name, description, argument hint, text).
pub const BUILT_IN: [(&str, &str, &str, &str); 1] = [(
    "review",
    "Review the changes: the ones staged here, the uncommitted ones, the branch against a base, or a commit. Changes nothing.",
    "[staged | branch | against <branch> | <commit> | what to look at]",
    REVIEW,
)];

/// `found`, then each built-in command no command of the same name hides.
fn with_built_in(mut found: Commands) -> Commands {
    for (name, description, hint, body) in BUILT_IN {
        if found.commands.iter().any(|command| command.name == name) {
            continue;
        }
        found.commands.push(Command {
            id: format!("builtin:{name}"),
            name: name.to_owned(),
            source: Source::BuiltIn,
            path: "(built in)".to_owned(),
            description: description.to_owned(),
            hint: hint.to_owned(),
            set_aside: Vec::new(),
            body: body.to_owned(),
        });
    }
    found
}

/// One command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Command {
    /// How the window names it: `user:<file>`, or the folder's derived path.
    pub id: String,
    /// What follows `/`: the file's stem, after its namespace and a colon.
    pub name: String,
    pub source: Source,
    /// Where it is, as shown: the folder's derived path, or `(yours) <file>`.
    pub path: String,
    pub description: String,
    /// `argument-hint`: what to type after the name.
    pub hint: String,
    /// The front matter's keys read and not honoured.
    pub set_aside: Vec<String>,
    /// Its text after the front matter.
    pub body: String,
}

/// The commands found, and what was left out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Commands {
    pub commands: Vec<Command>,
    pub notices: Vec<String>,
}

impl Commands {
    pub fn find(&self, id: &str) -> Option<&Command> {
        self.commands.iter().find(|command| command.id == id)
    }
}

/// `<native>/commands`: the reader's own commands.
pub fn user_commands_dir(state: &StateRoot) -> PathBuf {
    state.globals.join("lattice_native").join("commands")
}

/// Folder names in `dir`, sorted, none starting with a dot.
pub(crate) fn dirs_in(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| !name.starts_with('.'))
        .collect();
    names.sort();
    names
}

/// Read at most `cap + 1` bytes of an open file.
pub(crate) fn read_capped(file: std::fs::File, cap: u64) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    file.take(cap + 1).read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

/// A command's name from its file: `test.md` is `test`, and in the folder
/// `frontend`, `frontend:test`.
fn name_of(namespace: Option<&str>, file: &str) -> String {
    let stem = &file[..file.len() - ".md".len()];
    match namespace {
        Some(space) => format!("{space}:{stem}"),
        None => stem.to_owned(),
    }
}

struct Loader {
    found: Commands,
}

impl Loader {
    fn add(
        &mut self,
        id: String,
        name: String,
        source: Source,
        path: String,
        bytes: Option<Vec<u8>>,
    ) {
        if self.found.commands.len() >= MAX_COMMANDS {
            self.found.notices.push(format!(
                "{path} was left out: at most {MAX_COMMANDS} commands are listed."
            ));
            return;
        }
        let Some(bytes) = bytes else {
            self.found
                .notices
                .push(format!("{path} could not be read, so it was left out."));
            return;
        };
        if bytes.len() as u64 > MAX_COMMAND {
            self.found
                .notices
                .push(format!("{path} is larger than 64 KiB, so it was left out."));
            return;
        }
        let Ok(text) = String::from_utf8(bytes) else {
            self.found
                .notices
                .push(format!("{path} is not UTF-8 text, so it was left out."));
            return;
        };
        let (pairs, body) = crate::front::split(&text);
        let set_aside = pairs
            .iter()
            .map(|(key, _)| key.clone())
            .filter(|key| !KEYS.contains(&key.as_str()))
            .collect();
        self.found.commands.push(Command {
            id,
            name,
            source,
            path,
            description: crate::front::value(&pairs, "description")
                .unwrap_or_default()
                .to_owned(),
            hint: crate::front::value(&pairs, "argument-hint")
                .unwrap_or_default()
                .to_owned(),
            set_aside,
            body: body.to_owned(),
        });
    }

    /// The reader's own file at `rel` (with forward slashes) in `dir`.
    fn user(&mut self, dir: &Path, rel: &str, name: String) {
        let bytes = match open_walk(&dir.join(rel), Access::Read, LinkRule::AnyLocal) {
            Ok(walked) if !walked.is_dir => read_capped(walked.file, MAX_COMMAND),
            Err(WalkError::NotFound) => return,
            _ => None,
        };
        self.add(
            format!("user:{rel}"),
            name,
            Source::User,
            format!("(yours) {rel}"),
            bytes,
        );
    }

    /// A plugin's file at `rel` in its `commands` folder, read only inside
    /// the plugin's folder (`root`).
    fn plugin(&mut self, plugin: &str, root: &Path, commands: &Path, rel: &str, name: String) {
        let bytes = match open_walk(&commands.join(rel), Access::Read, LinkRule::Inside(root)) {
            Ok(walked) if !walked.is_dir => read_capped(walked.file, MAX_COMMAND),
            Err(WalkError::NotFound) => return,
            _ => None,
        };
        self.add(
            format!("plugin:{plugin}/{rel}"),
            name,
            Source::Plugin,
            format!("(plugin {plugin}) commands/{rel}"),
            bytes,
        );
    }

    /// The folder's file at `request`, through the path rules.
    fn folder(&mut self, rules: &PathRules<'_>, request: String, name: String) {
        match rules.resolve(&request, Want::Existing) {
            Ok(resolved) if !resolved.is_dir => {
                let bytes = resolved
                    .file
                    .and_then(|file| read_capped(file, MAX_COMMAND));
                self.add(
                    resolved.derived.clone(),
                    name,
                    Source::Folder,
                    resolved.derived,
                    bytes,
                );
            }
            Ok(_) => {}
            Err(error) => self
                .found
                .notices
                .push(format!("{request} was not read: {}", error.sentence())),
        }
    }
}

/// The commands the reader can run now (see the module header): their own,
/// then a trusted folder's.
pub fn load(state: &StateRoot, workspace: Option<(&PathRules<'_>, TrustState)>) -> Commands {
    let mut loader = Loader {
        found: Commands::default(),
    };
    let user = user_commands_dir(state);
    for file in names_in(&user, ".md") {
        loader.user(&user, &file, name_of(None, &file));
    }
    for space in dirs_in(&user) {
        for file in names_in(&user.join(&space), ".md") {
            loader.user(
                &user,
                &format!("{space}/{file}"),
                name_of(Some(&space), &file),
            );
        }
    }
    for (plugin, dir) in crate::plugins::enabled(state) {
        let Some(root) = crate::plugins::root_of(&dir) else {
            continue;
        };
        let commands = dir.join("commands");
        for file in names_in(&commands, ".md") {
            loader.plugin(
                &plugin,
                &root,
                &commands,
                &file,
                name_of(Some(&plugin), &file),
            );
        }
        for space in dirs_in(&commands) {
            for file in names_in(&commands.join(&space), ".md") {
                let name = name_of(Some(&format!("{plugin}:{space}")), &file);
                loader.plugin(&plugin, &root, &commands, &format!("{space}/{file}"), name);
            }
        }
    }
    let Some((rules, TrustState::Trusted)) = workspace else {
        return with_built_in(loader.found);
    };
    for base in FOLDER_DIRS {
        let Some((derived, final_path)) = folder_dir(rules, base) else {
            continue;
        };
        for file in names_in(&final_path, ".md") {
            loader.folder(rules, format!("{derived}/{file}"), name_of(None, &file));
        }
        for space in dirs_in(&final_path) {
            let Some((sub, sub_path)) = folder_dir(rules, &format!("{derived}/{space}")) else {
                continue;
            };
            for file in names_in(&sub_path, ".md") {
                loader.folder(rules, format!("{sub}/{file}"), name_of(Some(&space), &file));
            }
        }
    }
    with_built_in(loader.found)
}

/// Its arguments' words: split at white space, double quotes keeping words
/// together (and taken off).
fn words(arguments: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut quoted = false;
    let mut started = false;
    for c in arguments.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                started = true;
            }
            c if c.is_whitespace() && !quoted => {
                if started {
                    words.push(std::mem::take(&mut word));
                    started = false;
                }
            }
            c => {
                word.push(c);
                started = true;
            }
        }
    }
    if started {
        words.push(word);
    }
    words
}

/// The text a command makes with `arguments` (see the module header).
pub fn expand(command: &Command, arguments: &str) -> String {
    let arguments = arguments.trim();
    let words = words(arguments);
    let mut out = String::with_capacity(command.body.len() + arguments.len());
    let mut used = false;
    let mut rest = command.body.as_str();
    while let Some(at) = rest.find('$') {
        out.push_str(&rest[..at]);
        let after = &rest[at + 1..];
        if let Some(tail) = after.strip_prefix("ARGUMENTS") {
            out.push_str(arguments);
            used = true;
            rest = tail;
        } else if let Some(digit) = after.chars().next().filter(|c| ('1'..='9').contains(c)) {
            let n = digit as usize - '1' as usize;
            out.push_str(words.get(n).map_or("", String::as_str));
            used = true;
            rest = &after[1..];
        } else {
            out.push('$');
            rest = after;
        }
    }
    out.push_str(rest);
    let out = out.trim().to_owned();
    if used || arguments.is_empty() {
        out
    } else {
        format!("{out}\n\n{arguments}")
    }
}

#[cfg(test)]
mod tests;
