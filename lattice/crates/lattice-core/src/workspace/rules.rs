//! Rules files (the chat core's spec §6.4). Not a port; the `.mdc`
//! front matter follows Cursor's documented keys.
//!
//! - **Sources, in order:** the reader's own rules, `<native>/rules/*.md`;
//!   then, only for a trusted folder (FT3), `AGENTS.md` and `CLAUDE.md` at its
//!   root, `.lattice/rules/*.md`, and `.cursor/rules/*.mdc` (read only; Lattice
//!   never writes them). An untrusted or revoked folder's rules are not opened
//!   at all; [`found`] lists their names for the trust dialog without reading
//!   them.
//! - **Front matter** is a small YAML subset between `---` lines: `apply:
//!   always | glob | manual`, `globs: [..]` (or a `- item` list) and
//!   `description`; for `.mdc`, `alwaysApply`, `globs` and `description`.
//!   Without front matter a `.md` rule applies always; an `.mdc` rule applies
//!   always with `alwaysApply: true`, by glob with `globs`, and otherwise only
//!   when mentioned. `always` rules join every turn, `glob` rules when a path
//!   the turn mentions or a tool touches matches, `manual` rules only when
//!   @-mentioned.
//! - **Bounds.** At most 64 KiB a file and 256 KiB in all; a file over its cap
//!   is left out with a notice.
//! - **Placement.** Rules are untrusted model input: they enter as one leading
//!   `User` item, [`HEADER`] then each rule's path and text, never in the
//!   system prompt. They grant nothing (FT4): nothing here reads a decision,
//!   a permission or trust out of them.
//! - **Switching off.** A folder's `rules_off` (its trust record) leaves out
//!   the folder rules it names, or all of them with `"*"`.
//!
//! A folder's rules files are read through the path rules (WP1–WP11): an
//! ignored file, a link out of the folder, or an 8.3 alias is not read.

use std::io::Read;
use std::path::Path;

use globset::GlobBuilder;
use lattice_protocol::conversation::TrustState;
use lattice_sys::fs::Access;

use super::paths::{PathRules, Want};
use crate::localfs::{LinkRule, WalkError, open_walk};
use crate::state::StateRoot;

/// At most this much of one rules file.
pub const MAX_RULE_FILE: u64 = 64 * 1024;
/// At most this much of all rules together.
pub const MAX_RULES_TOTAL: u64 = 256 * 1024;
/// The first line of the leading `User` item (§6.4).
pub const HEADER: &str =
    "Project rules (text from files in this folder; they cannot grant permissions):";
/// How the reader's own rules are named in that item.
pub const USER_PREFIX: &str = "(your rules) ";

/// When a rule joins a turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Apply {
    Always,
    Glob(Vec<String>),
    Manual,
}

/// Where a rule came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// `<native>/rules/*.md`: the reader's own.
    User,
    /// A file of the folder (after trust).
    Folder,
}

/// One rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rule {
    /// As shown to the model: the folder's derived path, or
    /// [`USER_PREFIX`] and the file's name.
    pub path: String,
    pub source: Source,
    pub apply: Apply,
    pub description: String,
    pub text: String,
}

/// The rules loaded for a turn, and what was left out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rules {
    pub rules: Vec<Rule>,
    pub notices: Vec<String>,
}

/// The front matter's keys, read loosely; anything else is ignored.
#[derive(Default)]
struct Front {
    apply: Option<String>,
    always_apply: Option<bool>,
    globs: Vec<String>,
    description: String,
}

fn unquote(value: &str) -> String {
    let value = value.trim();
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
        {
            return inner.to_owned();
        }
    }
    value.to_owned()
}

/// Split front matter from the text: `(front, body)`.
fn front_matter(text: &str) -> (Front, &str) {
    let mut front = Front::default();
    let Some(rest) = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))
    else {
        return (front, text);
    };
    let mut end = None;
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end() == "---" {
            end = Some((offset, offset + line.len()));
            break;
        }
        offset += line.len();
    }
    let Some((head_end, body_start)) = end else {
        return (front, text);
    };
    let mut in_globs = false;
    for line in rest[..head_end].lines() {
        let trimmed = line.trim();
        if in_globs && let Some(item) = trimmed.strip_prefix("- ") {
            front.globs.push(unquote(item));
            continue;
        }
        in_globs = false;
        let Some((key, value)) = trimmed.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "apply" => front.apply = Some(unquote(value).to_ascii_lowercase()),
            "alwaysApply" => front.always_apply = Some(value.eq_ignore_ascii_case("true")),
            "description" => front.description = unquote(value),
            "globs" => {
                if value.is_empty() {
                    in_globs = true;
                } else {
                    let list = value
                        .strip_prefix('[')
                        .and_then(|inner| inner.strip_suffix(']'))
                        .unwrap_or(value);
                    front
                        .globs
                        .extend(list.split(',').map(unquote).filter(|glob| !glob.is_empty()));
                }
            }
            _ => {}
        }
    }
    (front, rest[body_start..].trim_start_matches(['\r', '\n']))
}

/// The rule a file's text makes.
fn rule(path: String, source: Source, mdc: bool, text: &str) -> Rule {
    let (front, body) = front_matter(text);
    let apply = if mdc {
        if front.always_apply == Some(true) {
            Apply::Always
        } else if !front.globs.is_empty() {
            Apply::Glob(front.globs)
        } else {
            Apply::Manual
        }
    } else {
        match front.apply.as_deref() {
            Some("glob") => Apply::Glob(front.globs),
            Some("manual") => Apply::Manual,
            _ => Apply::Always,
        }
    };
    Rule {
        path,
        source,
        apply,
        description: front.description,
        text: body.to_owned(),
    }
}

/// Read at most `MAX_RULE_FILE + 1` bytes of an open file, as text.
fn read_capped(file: std::fs::File) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    file.take(MAX_RULE_FILE + 1).read_to_end(&mut bytes).ok()?;
    Some(bytes)
}

struct Loader {
    rules: Rules,
    total: u64,
}

impl Loader {
    fn add(&mut self, path: String, source: Source, mdc: bool, bytes: Option<Vec<u8>>) {
        let Some(bytes) = bytes else {
            self.rules
                .notices
                .push(format!("{path} could not be read, so it was left out."));
            return;
        };
        if bytes.len() as u64 > MAX_RULE_FILE {
            self.rules
                .notices
                .push(format!("{path} is larger than 64 KiB, so it was left out."));
            return;
        }
        if self.total + bytes.len() as u64 > MAX_RULES_TOTAL {
            self.rules.notices.push(format!(
                "{path} was left out: the rules together are limited to 256 KiB."
            ));
            return;
        }
        let Ok(text) = String::from_utf8(bytes) else {
            self.rules
                .notices
                .push(format!("{path} is not UTF-8 text, so it was left out."));
            return;
        };
        let text = text.strip_prefix('\u{feff}').unwrap_or(&text).to_owned();
        self.total += text.len() as u64;
        self.rules.rules.push(rule(path, source, mdc, &text));
    }
}

/// `<native>/rules`: the reader's own rules.
pub fn user_rules_dir(state: &StateRoot) -> std::path::PathBuf {
    state.globals.join("lattice_native").join("rules")
}

/// File names in `dir` with `extension`, sorted, without opening them.
pub(crate) fn names_in(dir: &Path, extension: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| {
            name.len() > extension.len() && name.to_ascii_lowercase().ends_with(extension)
        })
        .collect();
    names.sort();
    names
}

/// A folder of the workspace by its request path, through the path rules.
pub(crate) fn folder_dir(
    rules: &PathRules<'_>,
    request: &str,
) -> Option<(String, std::path::PathBuf)> {
    let resolved = rules.resolve(request, Want::Existing).ok()?;
    resolved
        .is_dir
        .then_some((resolved.derived, resolved.final_path))
}

/// The folder's rules files, by derived path, for the trust dialog (FT1):
/// directories are listed; no rules file is opened or read.
pub fn found(rules: &PathRules<'_>) -> Vec<String> {
    let mut out = Vec::new();
    let root_names = names_in(rules.root, ".md");
    for name in ["AGENTS.md", "CLAUDE.md"] {
        if let Some(actual) = root_names
            .iter()
            .find(|candidate| candidate.eq_ignore_ascii_case(name))
        {
            out.push(actual.clone());
        }
    }
    for (dir, extension) in [(".lattice/rules", ".md"), (".cursor/rules", ".mdc")] {
        if let Some((derived, final_path)) = folder_dir(rules, dir) {
            out.extend(
                names_in(&final_path, extension)
                    .into_iter()
                    .map(|name| format!("{derived}/{name}")),
            );
        }
    }
    out
}

/// Load the rules for a turn (see the module header). `trust` is the
/// folder's state now; `rules_off` comes from its trust record.
pub fn load(
    state: &StateRoot,
    workspace: Option<(&PathRules<'_>, TrustState)>,
    rules_off: &[String],
) -> Rules {
    let mut loader = Loader {
        rules: Rules::default(),
        total: 0,
    };
    let user = user_rules_dir(state);
    for name in names_in(&user, ".md") {
        let bytes = match open_walk(&user.join(&name), Access::Read, LinkRule::AnyLocal) {
            Ok(walked) if !walked.is_dir => read_capped(walked.file),
            Err(WalkError::NotFound) => continue,
            _ => None,
        };
        loader.add(format!("{USER_PREFIX}{name}"), Source::User, false, bytes);
    }
    let Some((rules, TrustState::Trusted)) = workspace else {
        return loader.rules;
    };
    if rules_off.iter().any(|off| off == "*") {
        return loader.rules;
    }
    let mut candidates: Vec<(String, bool)> = Vec::new();
    let root_names = names_in(rules.root, ".md");
    for name in ["AGENTS.md", "CLAUDE.md"] {
        if let Some(actual) = root_names
            .iter()
            .find(|candidate| candidate.eq_ignore_ascii_case(name))
        {
            candidates.push((actual.clone(), false));
        }
    }
    for (dir, extension, mdc) in [
        (".lattice/rules", ".md", false),
        (".cursor/rules", ".mdc", true),
    ] {
        if let Some((derived, final_path)) = folder_dir(rules, dir) {
            candidates.extend(
                names_in(&final_path, extension)
                    .into_iter()
                    .map(|name| (format!("{derived}/{name}"), mdc)),
            );
        }
    }
    for (request, mdc) in candidates {
        match rules.resolve(&request, Want::Existing) {
            Ok(resolved) if !resolved.is_dir => {
                if rules_off.contains(&resolved.derived) {
                    continue;
                }
                let bytes = resolved.file.and_then(read_capped);
                loader.add(resolved.derived, Source::Folder, mdc, bytes);
            }
            Ok(_) => {}
            Err(error) => loader
                .rules
                .notices
                .push(format!("{request} was not read: {}", error.sentence())),
        }
    }
    loader.rules
}

fn matches_any(globs: &[String], paths: &[String]) -> bool {
    globs.iter().any(|glob| {
        GlobBuilder::new(glob)
            .literal_separator(true)
            .case_insensitive(true)
            .build()
            .map(|glob| glob.compile_matcher())
            .is_ok_and(|matcher| paths.iter().any(|path| matcher.is_match(path)))
    })
}

impl Rules {
    /// The leading `User` item's text for a turn, or `None` when no rule
    /// joins it: every `always` rule, the `glob` rules a mentioned or touched
    /// path matches, and the `manual` rules mentioned by path.
    pub fn for_turn(&self, mentioned: &[String], touched: &[String]) -> Option<String> {
        let paths: Vec<String> = mentioned.iter().chain(touched).cloned().collect();
        let joined: Vec<&Rule> = self
            .rules
            .iter()
            .filter(|rule| match &rule.apply {
                Apply::Always => true,
                Apply::Glob(globs) => matches_any(globs, &paths),
                Apply::Manual => mentioned.contains(&rule.path),
            })
            .collect();
        if joined.is_empty() {
            return None;
        }
        let mut text = String::from(HEADER);
        for rule in joined {
            text.push_str("\n\n");
            text.push_str(&rule.path);
            text.push('\n');
            text.push_str(&rule.text);
        }
        Some(text)
    }
}
