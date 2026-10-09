//! Where MCP servers are declared (the chat core's spec §12,
//! "Configuration"; T7, FT3, FT4). Not a port; the shape is Cursor's (and
//! Claude Code's and Claude Desktop's): `{"mcpServers": {"<name>": {"command":
//! "...", "args": ["..."], "env": {"NAME": "value"}, "cwd": "..."}}}`.
//!
//! - **The reader's own file**, `<native>/chat/mcp.json`. Its environment
//!   values are used as written and live only there. Lattice writes it only
//!   for the reader's own edits in the Tools view ([`put_user_server`],
//!   [`remove_user_server`], [`set_user_disabled`]), and never over a file it
//!   cannot read or parse.
//! - **A folder's files**, read only when the folder is trusted (FT3), through
//!   the folder's path rules: `.lattice/mcp.json`, then `.mcp.json` (Claude
//!   Code's), then `.cursor/mcp.json`. A name the first file declares wins.
//!   Their environment *names* are used; their values are never read, so a
//!   folder can ask for a variable of Lattice's own environment by name, and
//!   the reader sees that name before enabling the server.
//! - **Refused:** a server with a `url`, or with a `type` other than
//!   `stdio` (T7: [`REMOTE_REFUSED`]); a name outside [`valid_name`]; an entry
//!   without a command; arguments or variables that are not text.
//! - **The pin.** Each entry's SHA-256 is taken over its canonical JSON (keys
//!   sorted at every level, no whitespace, values included), so any change to
//!   the entry, even of one value, asks again ([`super::approvals`]).
//!
//! Nothing here starts anything or reads a decision: declaring a server
//! grants nothing (FT4).

use std::io::Read;
use std::path::{Path, PathBuf};

use lattice_sys::fs::Access;
use serde_json::{Map, Value};

use crate::fsx;
use crate::localfs::{LinkRule, WalkError, open_walk};
use crate::sha::sha256_hex;
use crate::state::StateRoot;
use crate::workspace::paths::{PathError, PathRules, Want};

/// The reader's file, in `<native>/chat`.
pub const USER_FILE: &str = "mcp.json";
/// How the reader's file is named to the reader.
pub const USER_FILE_SHOWN: &str = "your MCP settings";
/// A trusted folder's files, in the order a name is looked for.
pub const FOLDER_FILES: [&str; 3] = [".lattice/mcp.json", ".mcp.json", ".cursor/mcp.json"];
/// The largest configuration file read.
pub const MAX_CONFIG_FILE: u64 = 1024 * 1024;
/// The most servers one file declares that are used.
pub const MAX_SERVERS: usize = 32;
/// The longest server name.
pub const MAX_NAME: usize = 64;
/// The most arguments, and variables, one entry may have.
pub const MAX_ITEMS: usize = 256;
/// The longest command, argument, variable value or folder.
pub const MAX_TEXT: usize = 32 * 1024;
/// T7.
pub const REMOTE_REFUSED: &str =
    "Remote MCP servers send data off this machine; not supported yet.";

/// Where a server is declared.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Scope {
    /// The reader's own `<native>/chat/mcp.json`.
    User,
    /// A trusted folder's file: the folder's id and canonical path, both
    /// matched, as trust records are (§4.1).
    Folder { id: String, path: String },
}

/// One server: where it is declared, and its name there.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ServerKey {
    pub scope: Scope,
    pub name: String,
}

impl ServerKey {
    pub fn user(name: &str) -> Self {
        Self {
            scope: Scope::User,
            name: name.to_owned(),
        }
    }

    /// A short, stable text for this key (a dialog's CP3 key, a log).
    pub fn id(&self) -> String {
        match &self.scope {
            Scope::User => format!("user/{}", self.name),
            Scope::Folder { id, .. } => format!("folder/{id}/{}", self.name),
        }
    }
}

/// One server as declared.
#[derive(Clone, Debug, PartialEq)]
pub struct ServerEntry {
    pub key: ServerKey,
    /// The file it came from, as the reader is told: [`USER_FILE_SHOWN`], or
    /// the folder file's derived path.
    pub file: String,
    pub command: String,
    pub args: Vec<String>,
    /// Each variable the entry names, with its value from the reader's own
    /// file; a folder file's values are never read (`None`).
    pub env: Vec<(String, Option<String>)>,
    /// The folder it runs in, as written (absolute, or relative to the
    /// folder a folder's file is in).
    pub cwd: Option<String>,
    /// `"disabled": true` in the file.
    pub disabled: bool,
    /// SHA-256 of the entry's canonical JSON.
    pub sha256: String,
}

impl ServerEntry {
    /// The command and its arguments as one line, an argument with a space
    /// or nothing in it quoted: what the enable dialog shows.
    pub fn command_line(&self) -> String {
        let mut parts = vec![quoted(&self.command)];
        parts.extend(self.args.iter().map(|arg| quoted(arg)));
        parts.join(" ")
    }

    /// The names of the variables it sets, in order.
    pub fn env_names(&self) -> Vec<String> {
        self.env.iter().map(|(name, _)| name.clone()).collect()
    }
}

fn quoted(text: &str) -> String {
    if text.is_empty() || text.contains([' ', '\t']) {
        format!("\"{text}\"")
    } else {
        text.to_owned()
    }
}

/// A declaration that is not used, and why, in one sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Problem {
    /// The server's name, or `""` when the whole file is at fault.
    pub name: String,
    pub file: String,
    pub sentence: String,
}

/// What one or more files declare.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Declared {
    /// In name order within each file, the files in their order.
    pub servers: Vec<ServerEntry>,
    pub problems: Vec<Problem>,
}

impl Declared {
    fn problem(&mut self, name: &str, file: &str, sentence: impl Into<String>) {
        self.problems.push(Problem {
            name: name.to_owned(),
            file: file.to_owned(),
            sentence: sentence.into(),
        });
    }

    /// Add `other`'s servers whose names are not declared yet; a name
    /// declared again is a problem, and the first declaration is used.
    fn merge(&mut self, other: Declared) {
        for server in other.servers {
            if let Some(first) = self
                .servers
                .iter()
                .find(|kept| kept.key.name == server.key.name)
            {
                let sentence = format!(
                    "It is declared in {} too; the one there is used.",
                    first.file
                );
                self.problem(&server.key.name, &server.file, sentence);
            } else {
                self.servers.push(server);
            }
        }
        self.problems.extend(other.problems);
    }
}

/// Is `name` a server name Lattice uses: 1 to 64 letters, digits, spaces,
/// `-`, `_` or `.`, not starting or ending with a space.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME
        && name.trim() == name
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.'))
}

/// Is `name` an environment variable name Lattice passes on: a letter or
/// `_`, then letters, digits or `_`, at most 128.
pub fn valid_env_name(name: &str) -> bool {
    let mut chars = name.chars();
    name.len() <= 128
        && chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The canonical JSON of `value`: keys sorted (by their UTF-8 bytes) at every
/// level, no whitespace, strings and numbers as `serde_json` writes them.
/// Sorted here, not by the map type, so the pin does not depend on a feature
/// of `serde_json` another crate turns on.
pub fn canonical(value: &Value) -> String {
    let mut out = String::new();
    write_canonical(value, &mut out);
    out
}

fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::to_string(key).unwrap_or_default());
                out.push(':');
                write_canonical(&map[key], out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push(',');
                }
                write_canonical(item, out);
            }
            out.push(']');
        }
        other => out.push_str(&serde_json::to_string(other).unwrap_or_default()),
    }
}

fn text_field(object: &Map<String, Value>, key: &str) -> Result<Option<String>, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text.len() <= MAX_TEXT => Ok(Some(text.clone())),
        Some(Value::String(_)) => Err(format!("Its {key} is longer than 32 KiB.")),
        Some(_) => Err(format!("Its {key} is not text.")),
    }
}

/// One entry, or why it is not used.
pub fn entry_of(
    name: &str,
    entry: &Value,
    scope: &Scope,
    file: &str,
) -> Result<ServerEntry, String> {
    if !valid_name(name) {
        return Err("A server's name may use letters, digits, spaces, '-', '_' and '.', at most 64 characters.".to_owned());
    }
    let Some(object) = entry.as_object() else {
        return Err("Its entry is not an object.".to_owned());
    };
    let kind = object.get("type").and_then(Value::as_str);
    if object.contains_key("url") || kind.is_some_and(|kind| kind != "stdio") {
        return Err(REMOTE_REFUSED.to_owned());
    }
    let command = text_field(object, "command")?
        .map(|command| command.trim().to_owned())
        .filter(|command| !command.is_empty())
        .ok_or_else(|| "It names no command.".to_owned())?;
    if command.chars().any(char::is_control) {
        return Err("Its command holds a control character.".to_owned());
    }
    let args = match object.get("args") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) if items.len() <= MAX_ITEMS => items
            .iter()
            .map(|item| match item {
                Value::String(text) if text.len() <= MAX_TEXT => Ok(text.clone()),
                _ => Err("Its args must be a list of texts.".to_owned()),
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => return Err("Its args must be a list of at most 256 texts.".to_owned()),
    };
    let env = match object.get("env") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Object(map)) if map.len() <= MAX_ITEMS => {
            let mut names: Vec<&String> = map.keys().collect();
            names.sort();
            let mut env = Vec::with_capacity(names.len());
            for name in names {
                if !valid_env_name(name) {
                    return Err(format!(
                        "{name} is not a variable name Lattice passes on (letters, digits and '_')."
                    ));
                }
                let value = match (&map[name], scope) {
                    (Value::String(value), Scope::User) if value.len() <= MAX_TEXT => {
                        Some(value.clone())
                    }
                    (Value::String(_), Scope::User) => {
                        return Err(format!("The value of {name} is longer than 32 KiB."));
                    }
                    // A folder's values are never read: only the name counts.
                    (_, Scope::Folder { .. }) => None,
                    _ => return Err(format!("The value of {name} is not text.")),
                };
                env.push((name.clone(), value));
            }
            env
        }
        Some(_) => return Err("Its env must map at most 256 names to texts.".to_owned()),
    };
    let cwd = text_field(object, "cwd")?.filter(|cwd| !cwd.trim().is_empty());
    let disabled = object
        .get("disabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(ServerEntry {
        key: ServerKey {
            scope: scope.clone(),
            name: name.to_owned(),
        },
        file: file.to_owned(),
        command,
        args,
        env,
        cwd,
        disabled,
        sha256: sha256_hex(canonical(entry).as_bytes()),
    })
}

/// What one file's text declares.
pub fn parse(text: &str, scope: &Scope, file: &str) -> Declared {
    let mut declared = Declared::default();
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let value: Value = match serde_json::from_str(text) {
        Ok(value) => value,
        Err(_) => {
            declared.problem("", file, "It is not JSON, so no server in it is used.");
            return declared;
        }
    };
    let servers = match value.get("mcpServers") {
        None => return declared,
        Some(Value::Object(servers)) => servers,
        Some(_) => {
            declared.problem(
                "",
                file,
                "Its mcpServers is not an object, so no server in it is used.",
            );
            return declared;
        }
    };
    let mut names: Vec<&String> = servers.keys().collect();
    names.sort();
    for name in names {
        if declared.servers.len() >= MAX_SERVERS {
            declared.problem(name, file, "Only the first 32 servers of a file are used.");
            continue;
        }
        match entry_of(name, &servers[name], scope, file) {
            Ok(entry) => declared.servers.push(entry),
            Err(sentence) => declared.problem(name, file, sentence),
        }
    }
    declared
}

/// Read at most `MAX_CONFIG_FILE + 1` bytes.
fn read_capped(file: std::fs::File) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    file.take(MAX_CONFIG_FILE + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    Some(bytes)
}

/// The text of a file read capped, or the sentence why there is none.
fn text_of(bytes: Option<Vec<u8>>) -> Result<String, &'static str> {
    let bytes = bytes.ok_or("It could not be read, so no server in it is used.")?;
    if bytes.len() as u64 > MAX_CONFIG_FILE {
        return Err("It is larger than 1 MiB, so no server in it is used.");
    }
    String::from_utf8(bytes).map_err(|_| "It is not UTF-8 text, so no server in it is used.")
}

/// `<native>/chat/mcp.json`.
pub fn user_file(state: &StateRoot) -> PathBuf {
    state.native_chat_dir().join(USER_FILE)
}

/// The reader's own servers. No file declares none.
pub fn load_user(state: &StateRoot) -> Declared {
    let path = user_file(state);
    let bytes = match open_walk(&path, Access::Read, LinkRule::AnyLocal) {
        Ok(walked) if !walked.is_dir => read_capped(walked.file),
        Err(WalkError::NotFound) => return Declared::default(),
        _ => None,
    };
    match text_of(bytes) {
        Ok(text) => parse(&text, &Scope::User, USER_FILE_SHOWN),
        Err(sentence) => {
            let mut declared = Declared::default();
            declared.problem("", USER_FILE_SHOWN, sentence);
            declared
        }
    }
}

/// The servers a folder declares, through its path rules. The caller reads
/// them only for a trusted folder (FT3); an ignored or refused file is a
/// problem, a missing one is nothing.
pub fn load_folder(rules: &PathRules<'_>, scope: &Scope) -> Declared {
    let mut declared = Declared::default();
    for file in FOLDER_FILES {
        match rules.resolve(file, Want::Existing) {
            Ok(resolved) if !resolved.is_dir => {
                let derived = resolved.derived.clone();
                match text_of(resolved.file.and_then(read_capped)) {
                    Ok(text) => declared.merge(parse(&text, scope, &derived)),
                    Err(sentence) => declared.problem("", &derived, sentence),
                }
            }
            Ok(_) | Err(PathError::Missing) => {}
            Err(error) => declared.problem("", file, error.sentence()),
        }
    }
    declared
}

/// The folder files a trusted folder has, by derived path, without reading
/// them: for the trust dialog and the Tools view.
pub fn folder_files(rules: &PathRules<'_>) -> Vec<String> {
    FOLDER_FILES
        .iter()
        .filter_map(|file| match rules.resolve(file, Want::Existing) {
            Ok(resolved) if !resolved.is_dir => Some(resolved.derived),
            _ => None,
        })
        .collect()
}

// ------------------------------------------------------- the reader's edits

/// Why an edit of the reader's file was not made.
pub const NOT_CHANGED: &str =
    "Your MCP settings could not be read as JSON, so they were not changed.";

/// The reader's file as JSON: `{}` when there is none; `Err` when it cannot be
/// read or parsed, so it is never overwritten.
fn user_value(path: &Path) -> Result<Value, String> {
    let bytes = match open_walk(path, Access::Read, LinkRule::AnyLocal) {
        Ok(walked) if !walked.is_dir => read_capped(walked.file),
        Err(WalkError::NotFound) => return Ok(Value::Object(Map::new())),
        _ => None,
    };
    let text = text_of(bytes).map_err(|_| NOT_CHANGED.to_owned())?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    match serde_json::from_str::<Value>(text) {
        Ok(value @ Value::Object(_)) => Ok(value),
        _ => Err(NOT_CHANGED.to_owned()),
    }
}

fn write_user(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|_| "Lattice could not make its settings folder.".to_owned())?;
    }
    let mut text = serde_json::to_string_pretty(value).unwrap_or_default();
    text.push('\n');
    fsx::atomic_write(path, text.as_bytes())
        .map_err(|_| "Your MCP settings could not be saved.".to_owned())
}

/// Change the `mcpServers` object of the reader's file with `change`, then
/// write it, or say why not.
fn edit_user(
    state: &StateRoot,
    change: impl FnOnce(&mut Map<String, Value>) -> Result<(), String>,
) -> Result<(), String> {
    let path = user_file(state);
    let mut value = user_value(&path)?;
    let Some(root) = value.as_object_mut() else {
        return Err(NOT_CHANGED.to_owned());
    };
    let servers = root
        .entry("mcpServers")
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(servers) = servers.as_object_mut() else {
        return Err(NOT_CHANGED.to_owned());
    };
    change(servers)?;
    write_user(&path, &value)
}

/// What an environment value is shown as when an entry is edited
/// ([`user_entry_masked`]): saved back unchanged, it keeps the value the file
/// has, so a key is never shown to be edited.
pub const KEPT_VALUE: &str = "<kept>";

/// The reader's entry for `name` as JSON, each environment value replaced
/// by [`KEPT_VALUE`]: what the Tools view edits. `None` when there is none.
pub fn user_entry_masked(state: &StateRoot, name: &str) -> Option<Value> {
    let value = user_value(&user_file(state)).ok()?;
    let mut entry = value.get("mcpServers")?.get(name)?.clone();
    if let Some(env) = entry.get_mut("env").and_then(Value::as_object_mut) {
        for value in env.values_mut() {
            *value = Value::String(KEPT_VALUE.to_owned());
        }
    }
    Some(entry)
}

/// Add the server `name`, or replace its entry, in the reader's file. The
/// entry must be one Lattice would use ([`entry_of`]). A variable whose
/// value is [`KEPT_VALUE`] keeps the value the file's entry for `name` has.
pub fn put_user_server(state: &StateRoot, name: &str, entry: &Value) -> Result<(), String> {
    let mut entry = entry.clone();
    let kept: Vec<String> = entry
        .get("env")
        .and_then(Value::as_object)
        .map(|env| {
            env.iter()
                .filter(|(_, value)| value.as_str() == Some(KEPT_VALUE))
                .map(|(name, _)| name.clone())
                .collect()
        })
        .unwrap_or_default();
    if !kept.is_empty() {
        let old = user_value(&user_file(state))?;
        let old_env = old
            .get("mcpServers")
            .and_then(|servers| servers.get(name))
            .and_then(|entry| entry.get("env"))
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let env = entry
            .get_mut("env")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| NOT_CHANGED.to_owned())?;
        for variable in kept {
            let value = old_env
                .get(&variable)
                .filter(|value| value.is_string())
                .cloned()
                .ok_or_else(|| format!("{variable} has no value to keep; type its value."))?;
            env.insert(variable, value);
        }
    }
    entry_of(name, &entry, &Scope::User, USER_FILE_SHOWN)?;
    edit_user(state, |servers| {
        if !servers.contains_key(name) && servers.len() >= MAX_SERVERS {
            return Err("Your MCP settings already declare 32 servers.".to_owned());
        }
        servers.insert(name.to_owned(), entry);
        Ok(())
    })
}

/// Take the server `name` out of the reader's file.
pub fn remove_user_server(state: &StateRoot, name: &str) -> Result<(), String> {
    edit_user(state, |servers| {
        servers
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| "That server is not in your MCP settings.".to_owned())
    })
}

/// Set or clear `"disabled": true` on the server `name` of the reader's file.
pub fn set_user_disabled(state: &StateRoot, name: &str, disabled: bool) -> Result<(), String> {
    edit_user(state, |servers| {
        let Some(Value::Object(entry)) = servers.get_mut(name) else {
            return Err("That server is not in your MCP settings.".to_owned());
        };
        if disabled {
            entry.insert("disabled".to_owned(), Value::Bool(true));
        } else {
            entry.remove("disabled");
        }
        Ok(())
    })
}

/// The entries in text a person pasted: a whole file (`{"mcpServers":
/// {...}}`), an object of named entries, or one entry with a command (named
/// `fallback`). Each must be one Lattice would use.
pub fn pasted(text: &str, fallback: &str) -> Result<Vec<(String, Value)>, String> {
    let value: Value = serde_json::from_str(text.trim()).map_err(|_| {
        "That is not JSON. Paste the server's entry from its instructions.".to_owned()
    })?;
    let Some(object) = value.as_object() else {
        return Err("That is not a JSON object.".to_owned());
    };
    let named: Vec<(String, Value)> = if let Some(servers) = object.get("mcpServers") {
        servers
            .as_object()
            .ok_or_else(|| "Its mcpServers is not an object.".to_owned())?
            .iter()
            .map(|(name, entry)| (name.clone(), entry.clone()))
            .collect()
    } else if object.contains_key("command") || object.contains_key("url") {
        let name = fallback.trim();
        if name.is_empty() {
            return Err("Give the server a name.".to_owned());
        }
        vec![(name.to_owned(), value.clone())]
    } else {
        object
            .iter()
            .map(|(name, entry)| (name.clone(), entry.clone()))
            .collect()
    };
    if named.is_empty() {
        return Err("No server is declared there.".to_owned());
    }
    for (name, entry) in &named {
        entry_of(name, entry, &Scope::User, USER_FILE_SHOWN)
            .map_err(|sentence| format!("{name}: {sentence}"))?;
    }
    Ok(named)
}
