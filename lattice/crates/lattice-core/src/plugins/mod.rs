//! Plugins ("plugins" of 2026-10-08, their second part): folders
//! in Claude Code's plugin layout that bundle commands, skills, agents, hooks
//! and MCP servers. Not a port.
//!
//! - **A plugin** is a folder holding `.claude-plugin/plugin.json` (`name`,
//!   and `version`, `description` and `author` when given), with its parts
//!   where Claude Code looks for them: `commands/*.md`,
//!   `skills/<name>/SKILL.md`, `agents/*.md`, `hooks/hooks.json` and
//!   `.mcp.json`. Component paths a manifest names elsewhere are not read.
//! - **Added** only from a folder the reader chose in Windows' picker
//!   (Rust-owned UI, as an attached folder is): absolute, on this PC, not in
//!   Lattice's own state, holding a manifest Lattice can read, under a name not
//!   taken. Kept in `<native>/chat/plugins.json`, which is never written over
//!   when it cannot be read. A plugin is switched off or taken off the list;
//!   its files are never deleted.
//! - **What it brings while it is on:** its commands as `/<plugin>:<name>`
//!   ([`crate::commands`]) and its skills as `<plugin>:<name>`
//!   ([`crate::skills`]), in every chat. Its MCP servers are only listed: the
//!   window may copy them into the reader's own MCP settings
//!   ([`Plugins::mcp_entries`], `${CLAUDE_PLUGIN_ROOT}` filled in), where each
//!   still asks before it first starts. Its agents are listed and not used.
//!   Its hooks (`hooks/hooks.json`) run as the reader's own do
//!   ([`crate::hooks`]): each asks once, in the core's own dialog.
//! - **Its files** are read inside its folder only, every link bounded by it,
//!   with the commands' and skills' own bounds.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use lattice_sys::fs::Access;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::clock::Clock;
use crate::commands::{dirs_in, read_capped};
use crate::localfs::{LinkRule, open_walk};
use crate::state::StateRoot;
use crate::workspace::rules::names_in;

/// At most this much of a manifest or a plugin's `.mcp.json`.
pub const MAX_MANIFEST: u64 = 64 * 1024;
/// At most this many plugins are kept.
pub const MAX_PLUGINS: usize = 50;
/// What a plugin's `.mcp.json` names its own folder by.
pub const ROOT_VARIABLE: &str = "${CLAUDE_PLUGIN_ROOT}";

/// One plugin as the store keeps it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Entry {
    name: String,
    path: String,
    #[serde(default)]
    off: bool,
    added: f64,
}

#[derive(Default, Serialize, Deserialize)]
struct Stored {
    #[serde(default)]
    v: u32,
    #[serde(default)]
    plugins: Vec<Entry>,
}

/// A plugin as the window shows it: its manifest, and what it brings.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PluginView {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub path: String,
    pub on: bool,
    /// Its commands, by the name typed after `/`.
    pub commands: Vec<String>,
    /// Its skills, by their names.
    pub skills: Vec<String>,
    /// Its agents (listed; not used).
    pub agents: Vec<String>,
    /// It declares hooks (run as the reader's own, each asked for once).
    pub hooks: bool,
    /// Its MCP servers, by name.
    pub mcp_servers: Vec<String>,
    /// Why it cannot be read now, when it cannot.
    pub problem: Option<String>,
}

/// What copying a plugin's MCP servers did
/// (`AgentChat::plugin_mcp_copy`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Copied {
    /// Copied into the reader's settings, and enabled in the core's dialog.
    pub enabled: Vec<String>,
    /// Copied, and not enabled: the reader said no.
    pub not_enabled: Vec<String>,
    /// Not copied: the reader's settings hold a server of that name already,
    /// which is left as it is.
    pub kept: Vec<String>,
}

/// The reader's plugins.
pub struct Plugins {
    path: PathBuf,
    /// Lattice's own state: no plugin is added from it.
    globals: PathBuf,
    clock: Clock,
    lock: Mutex<()>,
}

fn guard(mutex: &Mutex<()>) -> MutexGuard<'_, ()> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// `<native>/chat/plugins.json`.
fn store_path(state: &StateRoot) -> PathBuf {
    state.native_chat_dir().join("plugins.json")
}

fn read_store(path: &Path) -> Result<Stored, String> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|_| {
            "plugins.json is not a record Lattice can read, so it was left as it is.".to_owned()
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Stored::default()),
        Err(_) => Err("Lattice could not read plugins.json.".to_owned()),
    }
}

/// A plugin's or a part's name: 1 to 64 letters, digits, `-`, `_` or `.`.
pub(crate) fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// A plugin folder's own final path: what bounds every link read in it.
pub(crate) fn root_of(dir: &Path) -> Option<PathBuf> {
    open_walk(dir, Access::Attributes, LinkRule::AnyLocal)
        .ok()
        .filter(|walked| walked.is_dir)
        .map(|walked| walked.final_path)
}

/// Whether `file` is there inside the plugin's folder (`root`), no link
/// followed out of it.
fn there(root: &Path, file: &Path) -> bool {
    open_walk(file, Access::Attributes, LinkRule::Inside(root)).is_ok_and(|walked| !walked.is_dir)
}

/// A file of a plugin, read only inside its folder (`root`), at most `cap`
/// bytes; `None` when it is not there or could not be read whole.
pub(crate) fn read_inside(root: &Path, file: &Path, cap: u64) -> Option<Vec<u8>> {
    let walked = open_walk(file, Access::Read, LinkRule::Inside(root)).ok()?;
    if walked.is_dir {
        return None;
    }
    read_capped(walked.file, cap)
}

/// A plugin's manifest: `(name, version, description, author)`.
fn manifest(dir: &Path, root: &Path) -> Result<(String, String, String, String), String> {
    let bytes = read_inside(
        root,
        &dir.join(".claude-plugin").join("plugin.json"),
        MAX_MANIFEST,
    )
    .ok_or_else(|| "It has no .claude-plugin/plugin.json Lattice can read.".to_owned())?;
    if bytes.len() as u64 > MAX_MANIFEST {
        return Err("Its plugin.json is larger than 64 KiB.".to_owned());
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|_| "Its plugin.json is not JSON Lattice can read.".to_owned())?;
    let text = |key: &str| {
        value
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned()
    };
    let name = text("name");
    if !valid_name(&name) {
        return Err(
            "Its plugin.json needs a name of 1 to 64 letters, digits, -, _ or .".to_owned(),
        );
    }
    let author = match value.get("author") {
        Some(Value::String(author)) => author.trim().to_owned(),
        Some(Value::Object(author)) => author
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim()
            .to_owned(),
        _ => String::new(),
    };
    Ok((name, text("version"), text("description"), author))
}

/// A plugin's commands' names, as typed after `/`.
fn command_names(plugin: &str, dir: &Path) -> Vec<String> {
    let commands = dir.join("commands");
    let mut names: Vec<String> = names_in(&commands, ".md")
        .into_iter()
        .map(|file| format!("{plugin}:{}", &file[..file.len() - 3]))
        .collect();
    for space in dirs_in(&commands) {
        names.extend(
            names_in(&commands.join(&space), ".md")
                .into_iter()
                .map(|file| format!("{plugin}:{space}:{}", &file[..file.len() - 3])),
        );
    }
    names
}

/// A plugin's MCP servers: its `.mcp.json`'s entries, by name, with
/// [`ROOT_VARIABLE`] filled in with its folder.
fn servers(dir: &Path, root: &Path) -> Result<Vec<(String, Value)>, String> {
    let Some(bytes) = read_inside(root, &dir.join(".mcp.json"), MAX_MANIFEST) else {
        return Ok(Vec::new());
    };
    if bytes.len() as u64 > MAX_MANIFEST {
        return Err("Its .mcp.json is larger than 64 KiB.".to_owned());
    }
    let mut value: Value = serde_json::from_slice(&bytes)
        .map_err(|_| "Its .mcp.json is not JSON Lattice can read.".to_owned())?;
    fill(&mut value, &dir.display().to_string());
    let Some(object) = value.as_object() else {
        return Err("Its .mcp.json is not a JSON object.".to_owned());
    };
    let servers = object.get("mcpServers").unwrap_or(&value);
    Ok(servers
        .as_object()
        .map(|servers| {
            servers
                .iter()
                .map(|(name, entry)| (name.clone(), entry.clone()))
                .collect()
        })
        .unwrap_or_default())
}

/// Every string in `value` with [`ROOT_VARIABLE`] replaced by `root`.
fn fill(value: &mut Value, root: &str) {
    match value {
        Value::String(text) if text.contains(ROOT_VARIABLE) => {
            *text = text.replace(ROOT_VARIABLE, root);
        }
        Value::Array(items) => items.iter_mut().for_each(|item| fill(item, root)),
        Value::Object(map) => map.values_mut().for_each(|item| fill(item, root)),
        _ => {}
    }
}

/// What a plugin's folder holds, as the window shows it.
fn view(entry: &Entry) -> PluginView {
    let dir = PathBuf::from(&entry.path);
    let mut out = PluginView {
        name: entry.name.clone(),
        path: entry.path.clone(),
        on: !entry.off,
        ..PluginView::default()
    };
    let Some(root) = root_of(&dir) else {
        out.problem = Some("Its folder is not here now.".to_owned());
        return out;
    };
    match manifest(&dir, &root) {
        Ok((name, version, description, author)) => {
            if name != entry.name {
                out.problem = Some(format!(
                    "Its plugin.json now names it {name}: take it off the list and add it again."
                ));
            }
            out.version = version;
            out.description = description;
            out.author = author;
        }
        Err(why) => {
            out.problem = Some(why);
            return out;
        }
    }
    out.commands = command_names(&entry.name, &dir);
    out.skills = dirs_in(&dir.join("skills"))
        .into_iter()
        .filter(|folder| there(&root, &dir.join("skills").join(folder).join("SKILL.md")))
        .map(|folder| format!("{}:{folder}", entry.name))
        .collect();
    out.agents = names_in(&dir.join("agents"), ".md")
        .into_iter()
        .map(|file| file[..file.len() - 3].to_owned())
        .collect();
    out.hooks = there(&root, &dir.join("hooks").join("hooks.json"));
    match servers(&dir, &root) {
        Ok(found) => out.mcp_servers = found.into_iter().map(|(name, _)| name).collect(),
        Err(why) => out.problem = Some(why),
    }
    out
}

/// The plugins switched on, by name and folder, for the commands and skills
/// to read; none when the store cannot be read.
pub fn enabled(state: &StateRoot) -> Vec<(String, PathBuf)> {
    read_store(&store_path(state))
        .map(|stored| {
            stored
                .plugins
                .into_iter()
                .filter(|entry| !entry.off)
                .map(|entry| (entry.name, PathBuf::from(entry.path)))
                .collect()
        })
        .unwrap_or_default()
}

impl Plugins {
    pub fn new(state: &StateRoot, clock: Clock) -> Self {
        Self {
            path: store_path(state),
            globals: state.globals.clone(),
            clock,
            lock: Mutex::new(()),
        }
    }

    fn write(&self, stored: &Stored) -> Result<(), String> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)
                .map_err(|_| "Lattice could not make its settings folder.".to_owned())?;
        }
        let bytes = serde_json::to_vec_pretty(stored).unwrap_or_default();
        crate::fsx::atomic_write(&self.path, &bytes)
            .map_err(|_| "Lattice could not save plugins.json.".to_owned())
    }

    /// Every plugin on the list, with what it brings.
    pub fn list(&self) -> Result<Vec<PluginView>, String> {
        let _held = guard(&self.lock);
        Ok(read_store(&self.path)?.plugins.iter().map(view).collect())
    }

    /// Add the plugin in `folder`, a folder the reader chose in Windows'
    /// picker, switched on.
    pub fn add(&self, folder: &Path) -> Result<PluginView, String> {
        if !folder.is_absolute() || !crate::localfs::is_local_text(folder) {
            return Err("A plugin is a folder on this PC.".to_owned());
        }
        let root = root_of(folder).ok_or_else(|| "That folder is not here.".to_owned())?;
        if crate::localfs::is_inside(folder, &self.globals)
            || crate::localfs::is_inside(&root, &self.globals)
        {
            return Err("Lattice's own files cannot be a plugin.".to_owned());
        }
        let (name, ..) = manifest(folder, &root)?;
        let _held = guard(&self.lock);
        let mut stored = read_store(&self.path)?;
        let path = folder.display().to_string();
        if stored.plugins.iter().any(|entry| {
            crate::localfs::is_inside(Path::new(&entry.path), folder)
                && crate::localfs::is_inside(folder, Path::new(&entry.path))
        }) {
            return Err("That plugin is on the list already.".to_owned());
        }
        if stored
            .plugins
            .iter()
            .any(|entry| entry.name.eq_ignore_ascii_case(&name))
        {
            return Err(format!("A plugin named {name} is on the list already."));
        }
        if stored.plugins.len() >= MAX_PLUGINS {
            return Err(format!("At most {MAX_PLUGINS} plugins are kept."));
        }
        let entry = Entry {
            name,
            path,
            off: false,
            added: (self.clock)(),
        };
        stored.v = 1;
        stored.plugins.push(entry.clone());
        self.write(&stored)?;
        Ok(view(&entry))
    }

    fn change(&self, name: &str, work: impl FnOnce(&mut Stored, usize)) -> Result<(), String> {
        let _held = guard(&self.lock);
        let mut stored = read_store(&self.path)?;
        let at = stored
            .plugins
            .iter()
            .position(|entry| entry.name == name)
            .ok_or_else(|| "That plugin is not on the list.".to_owned())?;
        work(&mut stored, at);
        self.write(&stored)
    }

    /// Switch a plugin on or off.
    pub fn set_on(&self, name: &str, on: bool) -> Result<(), String> {
        self.change(name, |stored, at| stored.plugins[at].off = !on)
    }

    /// Take a plugin off the list (its folder is left as it is).
    pub fn remove(&self, name: &str) -> Result<(), String> {
        self.change(name, |stored, at| {
            stored.plugins.remove(at);
        })
    }

    /// A plugin's MCP servers, each checked as a pasted entry is, with its
    /// folder filled in: for the window to copy into the reader's settings.
    pub fn mcp_entries(&self, name: &str) -> Result<Vec<(String, Value)>, String> {
        let entry = {
            let _held = guard(&self.lock);
            read_store(&self.path)?
                .plugins
                .into_iter()
                .find(|entry| entry.name == name)
                .ok_or_else(|| "That plugin is not on the list.".to_owned())?
        };
        let dir = PathBuf::from(&entry.path);
        let root = root_of(&dir).ok_or_else(|| "Its folder is not here now.".to_owned())?;
        let found = servers(&dir, &root)?;
        if found.is_empty() {
            return Err("It declares no MCP server.".to_owned());
        }
        let block = serde_json::json!({ "mcpServers": found.into_iter().collect::<serde_json::Map<String, Value>>() });
        crate::mcp::config::pasted(&block.to_string(), "")
    }
}

#[cfg(test)]
mod tests;
