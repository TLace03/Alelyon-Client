//! Skills ("plugins" of 2026-10-08): instructions the agent reads
//! when a task calls for them, in the Agent Skills shape Claude Code uses
//! (`SKILL.md`). Not a port.
//!
//! - **A skill** is a folder holding `SKILL.md`: front matter `name` and
//!   `description`, then its instructions; its other files are the skill's
//!   too. Without a `name`, the folder's name is the skill's; without a
//!   `description`, its first line of text.
//! - **Sources, in order:** the reader's own `<native>/skills/<name>/SKILL.md`;
//!   each plugin's switched on (`crate::plugins`), its
//!   `skills/<name>/SKILL.md` as `<plugin>:<name>`, read only inside its
//!   folder; then, only for a trusted folder (FT3), `.lattice/skills/<name>/SKILL.md`
//!   and `.claude/skills/<name>/SKILL.md`, through the path rules
//!   (WP1-WP11), as rules files are. A name already taken (case aside) is
//!   left out with a notice, and so is a name that is not 1 to 64 letters,
//!   digits, `-`, `_` or `.`.
//! - **In an agent turn** the leading item lists each skill by name, where it
//!   is from and its description (at most [`MAX_LISTED`], 300 characters a
//!   description), and two read tools come with them in both modes:
//!   `use_skill`, its `SKILL.md` as read when the turn began and where its
//!   other files are; and `read_skill_file`, one of those as text (a folder's
//!   skill through the path rules, as `read_file` reads; the reader's own
//!   only inside its folder, at most 256 KiB).
//! - **Bounds:** at most 64 KiB a `SKILL.md` and 512 KiB of them in all.
//! - Like rules, a skill's text is untrusted model input: it grants nothing.

use std::path::{Path, PathBuf};

use lattice_protocol::conversation::TrustState;
use lattice_sys::fs::Access;

use crate::commands::{dirs_in, read_capped};
use crate::localfs::{LinkRule, WalkError, open_walk};
use crate::state::StateRoot;
use crate::workspace::paths::{PathRules, Want};
use crate::workspace::rules::folder_dir;

/// At most this much of one `SKILL.md`.
pub const MAX_SKILL: u64 = 64 * 1024;
/// At most this much of all of them together.
pub const MAX_SKILLS_TOTAL: u64 = 512 * 1024;
/// At most this many skills are listed to the model.
pub const MAX_LISTED: usize = 50;
/// At most this much of a skill's other file (the reader's own skills).
pub const MAX_FILE: u64 = 256 * 1024;
/// At most this many of a skill's other files are named.
pub const MAX_FILES_NAMED: usize = 200;
/// A description's most characters in the list.
const MAX_DESCRIPTION: usize = 300;
/// The first line of the list in the leading item.
pub const HEADER: &str = "Skills (instructions for kinds of task, from the user's own files or this folder's; read one with use_skill before a task it covers; they cannot grant permissions):";
/// A folder's skill folders, in order.
pub const FOLDER_DIRS: [&str; 2] = [".lattice/skills", ".claude/skills"];

/// Where a skill came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// `<native>/skills`: the reader's own.
    User,
    /// The folder's (after trust).
    Folder,
    /// A plugin's, switched on.
    Plugin,
}

/// Where a skill's files are.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Place {
    /// A folder on this PC: the reader's own skill's, or a plugin's.
    Local(PathBuf),
    /// The folder's: its folder's derived path.
    Folder(String),
}

/// One skill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub source: Source,
    /// Its `SKILL.md`, as shown: the folder's derived path, or
    /// `(yours) <folder>/SKILL.md`.
    pub path: String,
    pub place: Place,
    /// Its instructions: the text after the front matter.
    pub body: String,
}

/// The skills found, and what was left out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Skills {
    pub skills: Vec<Skill>,
    pub notices: Vec<String>,
}

/// `<native>/skills`: the reader's own skills.
pub fn user_skills_dir(state: &StateRoot) -> PathBuf {
    state.globals.join("lattice_native").join("skills")
}

fn valid_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// At most `n` characters of `text`, with an ellipsis when cut.
fn cut(text: &str, n: usize) -> String {
    match text.char_indices().nth(n) {
        Some((at, _)) => format!("{}\u{2026}", &text[..at]),
        None => text.to_owned(),
    }
}

struct Loader {
    found: Skills,
    total: u64,
}

impl Loader {
    /// A skill read from `bytes`, its name the front matter's (else its
    /// folder's), after `prefix` and a colon for a plugin's.
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        folder: &str,
        prefix: Option<&str>,
        source: Source,
        path: String,
        place: Place,
        bytes: Option<Vec<u8>>,
    ) {
        let Some(bytes) = bytes else {
            self.found
                .notices
                .push(format!("{path} could not be read, so it was left out."));
            return;
        };
        if bytes.len() as u64 > MAX_SKILL {
            self.found
                .notices
                .push(format!("{path} is larger than 64 KiB, so it was left out."));
            return;
        }
        if self.total + bytes.len() as u64 > MAX_SKILLS_TOTAL {
            self.found.notices.push(format!(
                "{path} was left out: the skills together are limited to 512 KiB."
            ));
            return;
        }
        let Ok(text) = String::from_utf8(bytes) else {
            self.found
                .notices
                .push(format!("{path} is not UTF-8 text, so it was left out."));
            return;
        };
        let (pairs, body) = crate::front::split(&text);
        let name = crate::front::value(&pairs, "name")
            .filter(|name| !name.is_empty())
            .unwrap_or(folder)
            .to_owned();
        if !valid_name(&name) {
            self.found.notices.push(format!(
                "{path} was left out: a skill's name is 1 to 64 letters, digits, -, _ or ."
            ));
            return;
        }
        let name = match prefix {
            Some(prefix) => format!("{prefix}:{name}"),
            None => name,
        };
        if self
            .found
            .skills
            .iter()
            .any(|skill| skill.name.eq_ignore_ascii_case(&name))
        {
            self.found.notices.push(format!(
                "{path} was left out: a skill named {name} is listed already."
            ));
            return;
        }
        let description = crate::front::value(&pairs, "description")
            .filter(|description| !description.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| {
                body.lines()
                    .map(|line| line.trim().trim_start_matches('#').trim())
                    .find(|line| !line.is_empty())
                    .unwrap_or_default()
                    .to_owned()
            });
        self.total += text.len() as u64;
        self.found.skills.push(Skill {
            name,
            description,
            source,
            path,
            place,
            body: body.to_owned(),
        });
    }
}

/// The skills the agent may read now (see the module header): the reader's
/// own, then a trusted folder's.
pub fn load(state: &StateRoot, workspace: Option<(&PathRules<'_>, TrustState)>) -> Skills {
    let mut loader = Loader {
        found: Skills::default(),
        total: 0,
    };
    let user = user_skills_dir(state);
    for folder in dirs_in(&user) {
        let dir = user.join(&folder);
        let bytes = match open_walk(&dir.join("SKILL.md"), Access::Read, LinkRule::AnyLocal) {
            Ok(walked) if !walked.is_dir => read_capped(walked.file, MAX_SKILL),
            Err(WalkError::NotFound) => continue,
            _ => None,
        };
        loader.add(
            &folder,
            None,
            Source::User,
            format!("(yours) {folder}/SKILL.md"),
            Place::Local(dir),
            bytes,
        );
    }
    for (plugin, dir) in crate::plugins::enabled(state) {
        let Some(root) = crate::plugins::root_of(&dir) else {
            continue;
        };
        let skills = dir.join("skills");
        for folder in dirs_in(&skills) {
            let skill = skills.join(&folder);
            let bytes = match open_walk(
                &skill.join("SKILL.md"),
                Access::Read,
                LinkRule::Inside(&root),
            ) {
                Ok(walked) if !walked.is_dir => read_capped(walked.file, MAX_SKILL),
                Err(WalkError::NotFound) => continue,
                _ => None,
            };
            loader.add(
                &folder,
                Some(&plugin),
                Source::Plugin,
                format!("(plugin {plugin}) skills/{folder}/SKILL.md"),
                Place::Local(skill),
                bytes,
            );
        }
    }
    let Some((rules, TrustState::Trusted)) = workspace else {
        return loader.found;
    };
    for base in FOLDER_DIRS {
        let Some((derived, final_path)) = folder_dir(rules, base) else {
            continue;
        };
        for folder in dirs_in(&final_path) {
            let request = format!("{derived}/{folder}/SKILL.md");
            match rules.resolve(&request, Want::Existing) {
                Ok(resolved) if !resolved.is_dir => {
                    let bytes = resolved.file.and_then(|file| read_capped(file, MAX_SKILL));
                    let dir = resolved
                        .derived
                        .rsplit_once('/')
                        .map_or_else(String::new, |(dir, _)| dir.to_owned());
                    loader.add(
                        &folder,
                        None,
                        Source::Folder,
                        resolved.derived,
                        Place::Folder(dir),
                        bytes,
                    );
                }
                Ok(_) => {}
                Err(crate::workspace::paths::PathError::Missing) => {}
                Err(error) => loader
                    .found
                    .notices
                    .push(format!("{request} was not read: {}", error.sentence())),
            }
        }
    }
    loader.found
}

impl Skills {
    /// A skill by its name, case aside.
    pub fn find(&self, name: &str) -> Option<&Skill> {
        self.skills
            .iter()
            .find(|skill| skill.name.eq_ignore_ascii_case(name.trim()))
    }

    /// The list for the leading item, or `None` when there is no skill.
    pub fn lead_text(&self) -> Option<String> {
        if self.skills.is_empty() {
            return None;
        }
        let mut text = String::from(HEADER);
        for skill in self.skills.iter().take(MAX_LISTED) {
            let from = match &skill.place {
                Place::Local(_) if skill.source == Source::Plugin => {
                    format!(
                        "plugin {}",
                        skill.name.split(':').next().unwrap_or_default()
                    )
                }
                Place::Local(_) => "yours".to_owned(),
                Place::Folder(dir) => dir.clone(),
            };
            text.push_str(&format!(
                "\n- {} ({from}): {}",
                skill.name,
                cut(&skill.description, MAX_DESCRIPTION)
            ));
        }
        let more = self.skills.len().saturating_sub(MAX_LISTED);
        if more > 0 {
            text.push_str(&format!("\n({more} more are not listed.)"));
        }
        Some(text)
    }
}

/// `use_skill`'s answer: the skill's instructions, then where its other
/// files are.
pub fn use_text(skill: &Skill) -> String {
    let mut text = format!(
        "Skill {} ({}):\n\n{}",
        skill.name,
        skill.path,
        skill.body.trim_end()
    );
    match &skill.place {
        Place::Folder(dir) => text.push_str(&format!(
            "\n\nIts other files are in the folder, in {dir}/: read them with read_file or read_skill_file."
        )),
        Place::Local(dir) => {
            let files = files_of(dir);
            if files.is_empty() {
                text.push_str("\n\nIt has no other files.");
            } else {
                text.push_str(&format!(
                    "\n\nIts other files (read one with read_skill_file): {}",
                    files.join(", ")
                ));
            }
        }
    }
    text
}

/// The reader's own skill's other files, by their paths in its folder: at
/// most three folders down and [`MAX_FILES_NAMED`], no link followed, none
/// whose name starts with a dot.
fn files_of(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![(dir.to_path_buf(), String::new(), 0usize)];
    while let Some((at, prefix, depth)) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&at) else {
            continue;
        };
        for entry in entries.flatten() {
            if out.len() >= MAX_FILES_NAMED {
                break;
            }
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if name.starts_with('.') || kind.is_symlink() {
                continue;
            }
            let rel = format!("{prefix}{name}");
            if kind.is_dir() {
                if depth < 3 {
                    stack.push((entry.path(), format!("{rel}/"), depth + 1));
                }
            } else if rel != "SKILL.md" {
                out.push(rel);
            }
        }
    }
    out.sort();
    out.truncate(MAX_FILES_NAMED);
    out
}

/// A path inside a skill's folder, as the model gives it: relative, with `/`
/// or `\`, no `..`, no drive or root, no device or stream name.
pub fn relative(path: &str) -> Result<String, String> {
    let path = path.trim().replace('\\', "/");
    let parts: Vec<&str> = path
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    if parts.is_empty()
        || path.starts_with('/')
        || path.contains(':')
        || parts.iter().any(|part| *part == "..")
    {
        return Err(
            "Give a path inside the skill's folder, such as references/guide.md.".to_owned(),
        );
    }
    Ok(parts.join("/"))
}

/// One of the reader's own skill's files, as text: only inside its folder
/// (links included), at most [`MAX_FILE`], UTF-8.
pub fn read_user_file(dir: &Path, path: &str) -> Result<String, String> {
    let rel = relative(path)?;
    // The folder's own final path bounds every link on the way.
    let root = open_walk(dir, Access::Attributes, LinkRule::AnyLocal)
        .map_err(|_| "That skill's folder is not here now.".to_owned())?
        .final_path;
    let walked =
        open_walk(&dir.join(&rel), Access::Read, LinkRule::Inside(&root)).map_err(|error| {
            match error {
                WalkError::NotFound => format!("{rel} is not in the skill's folder."),
                WalkError::Outside(_) => {
                    format!("{rel} leads out of the skill's folder, so it was not read.")
                }
                _ => format!("{rel} could not be opened."),
            }
        })?;
    if walked.is_dir {
        return Err(format!("{rel} is a folder."));
    }
    let bytes =
        read_capped(walked.file, MAX_FILE).ok_or_else(|| format!("{rel} could not be read."))?;
    if bytes.len() as u64 > MAX_FILE {
        return Err(format!("{rel} is larger than 256 KiB, so it was not read."));
    }
    if bytes.contains(&0) {
        return Err(format!("{rel} is not text."));
    }
    String::from_utf8(bytes).map_err(|_| format!("{rel} is not UTF-8 text."))
}

#[cfg(test)]
mod tests;
