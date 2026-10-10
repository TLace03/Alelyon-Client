//! Where hooks are read: the reader's own files on this PC, the plugins
//! switched on, and a trusted folder's files.
//!
//! Only each file's `hooks` is used: a settings file's other entries (keys,
//! models, permissions) are not read into anything, logged or shown. A file in
//! the folder is read only when the folder is trusted, and only inside it (a
//! link out of the folder is not followed); a plugin's, only inside the plugin.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;

use super::formats;
use super::{Family, Found, Problem, Source};
use crate::env::Env;
use crate::state::{self, Platform, StateRoot};

/// The most of a hook or settings file that is read.
pub const MAX_FILE: u64 = 1024 * 1024;

/// Names the folder the reader's own tools' files are looked for in, in place
/// of the home folder: tests point it at their own scratch, so no test reads
/// (or runs) the reader's own hooks.
pub const HOME_ENV: &str = "ALELYON_HOOKS_HOME";

/// Lattice's own hooks file.
pub fn lattice_file(state: &StateRoot) -> PathBuf {
    state.globals.join("lattice_native").join("hooks.json")
}

/// The reader's own files: (source, format, path under the home folder).
const USER_FILES: &[(&str, Family, &[&str])] = &[
    ("claude", Family::Claude, &[".claude", "settings.json"]),
    ("cursor", Family::Cursor, &[".cursor", "hooks.json"]),
    ("gemini", Family::Gemini, &[".gemini", "settings.json"]),
    ("antigravity", Family::Gemini, &[".gemini", "config", "hooks.json"]),
];

/// A trusted folder's files: (format, path in the folder).
pub const FOLDER_FILES: &[(Family, &str)] = &[
    (Family::Claude, ".lattice/hooks.json"),
    (Family::Claude, ".claude/settings.json"),
    (Family::Claude, ".claude/settings.local.json"),
    (Family::Cursor, ".cursor/hooks.json"),
    (Family::Gemini, ".gemini/settings.json"),
    (Family::Gemini, ".agents/hooks.json"),
];

fn user_source(key: &str) -> Source {
    match key {
        "claude" => Source::ClaudeCode,
        "cursor" => Source::Cursor,
        "gemini" => Source::GeminiCli,
        _ => Source::Antigravity,
    }
}

fn read_capped(path: &Path) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::new();
    file.take(MAX_FILE + 1).read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

/// Read `bytes` (from `file`) into `found`.
fn take(found: &mut Found, bytes: Option<Vec<u8>>, source: Source, family: Family, file: &Path, plugin_root: Option<&Path>) {
    let Some(bytes) = bytes else { return };
    if bytes.len() as u64 > MAX_FILE {
        found.problems.push(Problem { file: file.to_path_buf(), sentence: "It is larger than 1 MiB, so it was not read.".to_owned() });
        return;
    }
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        found.problems.push(Problem { file: file.to_path_buf(), sentence: "It is not JSON Lattice can read.".to_owned() });
        return;
    };
    let (hooks, problems) = formats::read(&value, &source, family, file, plugin_root);
    found.hooks.extend(hooks);
    found.problems.extend(problems);
}

/// Every hook for a turn in `folder` (`trusted` when the reader trusts it), or
/// with no folder.
pub fn discover(env: &dyn Env, state: &StateRoot, folder: Option<&Path>, trusted: bool) -> Found {
    let mut found = Found::default();
    let own = lattice_file(state);
    take(&mut found, read_capped(&own), Source::Lattice, Family::Claude, &own, None);
    for (name, dir) in crate::plugins::enabled(state) {
        let Some(root) = crate::plugins::root_of(&dir) else { continue };
        let file = root.join("hooks").join("hooks.json");
        let bytes = crate::plugins::read_inside(&root, &file, MAX_FILE + 1);
        take(&mut found, bytes, Source::Plugin(name), Family::Claude, &file, Some(&root));
    }
    let home = crate::env::text(env, HOME_ENV)
        .filter(|home| !home.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| state::home_dir(env, Platform::host()));
    for (key, family, parts) in USER_FILES {
        let file = parts.iter().fold(home.clone(), |path, part| path.join(part));
        take(&mut found, read_capped(&file), user_source(key), *family, &file, None);
    }
    if let (Some(folder), true) = (folder, trusted)
        && let Some(root) = crate::plugins::root_of(folder)
    {
        for (family, rel) in FOLDER_FILES {
            let file = rel.split('/').fold(root.clone(), |path, part| path.join(part));
            let bytes = crate::plugins::read_inside(&root, &file, MAX_FILE + 1);
            take(&mut found, bytes, Source::Folder((*rel).to_owned()), *family, &file, None);
        }
    }
    found
}
