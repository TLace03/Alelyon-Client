//! The stack advisor: what to use for each part of a piece of software, as a
//! curated catalog the agent and the reader read the same way.
//!
//! - **The catalog** (`catalog.json`, embedded): decision areas in four groups
//!   (languages and runtimes, databases and storage, hosted services,
//!   frameworks and architecture). Each area names the needs that tell its
//!   options apart; each option says what it is best for and poor for, its
//!   pros and cons, licence and cost, the signals that show a project already
//!   uses it, a setup guide of steps for the reader, and checks that suggest
//!   an improvement where it is already in use. Its date (`as_of`) is when it
//!   was written: a DECLARED summary, not a measurement of any product.
//! - **Ranking** ([`rank`]) is symbolic: each need an option is best for adds,
//!   each it is poor for subtracts, and an option the project already uses
//!   ranks higher, so advice fits what is there.
//! - **The project** ([`scan`]): the names of the files in a folder and one
//!   level below, and the text of its manifests (package and build files,
//!   container and deploy files), each at most [`MAX_FILE_BYTES`]. Nothing
//!   else is read; a `.env` file never is.
//! - **Review** ([`review`]): for each option the project uses, the catalog's
//!   checks whose conditions hold: a suggestion with why. A check reads the
//!   manifests only, so it can be wrong; each says what it looked for.
//! - **Steps that need a decision** ([`area_for_step`]): a to-do step whose
//!   words match an area's triggers.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// The most bytes read of one manifest.
pub const MAX_FILE_BYTES: u64 = 256 * 1024;
/// The most options a decision offers.
pub const MAX_OFFERED: usize = 4;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Catalog {
    pub as_of: String,
    pub areas: Vec<Area>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Area {
    pub id: String,
    pub group: String,
    pub title: String,
    pub question: String,
    pub triggers: Vec<String>,
    /// Need id -> what it means, in the reader's words.
    pub needs: BTreeMap<String, String>,
    pub options: Vec<Choice>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Choice {
    pub id: String,
    pub name: String,
    pub summary: String,
    pub best_for: Vec<String>,
    pub poor_for: Vec<String>,
    pub pros: Vec<String>,
    pub cons: Vec<String>,
    pub licence: String,
    pub cost: String,
    pub signals: Vec<String>,
    pub guide: Vec<String>,
    pub checks: Vec<Check>,
}

/// A suggestion that applies where an option is used and its conditions hold:
/// every `present` and at least one `present_any` found, no `absent` found,
/// and none of `files_absent` among the project's files.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Check {
    pub id: String,
    pub suggest: String,
    pub why: String,
    #[serde(default)]
    pub present: Vec<String>,
    #[serde(default)]
    pub present_any: Vec<String>,
    #[serde(default)]
    pub absent: Vec<String>,
    #[serde(default)]
    pub files_absent: Vec<String>,
}

/// The catalog, read once.
pub fn catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| serde_json::from_str(include_str!("catalog.json")).expect("the embedded catalog is valid"))
}

pub fn area(id: &str) -> Option<&'static Area> {
    catalog().areas.iter().find(|a| a.id == id)
}

impl Area {
    pub fn choice(&self, id: &str) -> Option<&Choice> {
        self.options.iter().find(|o| o.id == id)
    }
}

/// What a folder shows of its stack.
#[derive(Clone, Debug, Default)]
pub struct Project {
    /// File names (lower case) in the folder and one level below, each as its name alone and as `dir/name`.
    pub files: BTreeSet<String>,
    /// The manifests' text, lower case.
    pub text: String,
}

/// The files whose text is read: by exact name, or (for a pattern ending in
/// `*`) by prefix, or (starting with `*`) by suffix.
const MANIFESTS: &[&str] = &[
    "cargo.toml", "package.json", "tsconfig.json", "pyproject.toml", "requirements*", "setup.py", "setup.cfg", "pipfile",
    "go.mod", "pom.xml", "build.gradle", "build.gradle.kts", "*.csproj", "gemfile", "composer.json", "dockerfile",
    "docker-compose*", "compose.yaml", "compose.yml", "fly.toml", "vercel.json", "netlify.toml", "wrangler.toml",
    "cloudbuild.yaml", "serverless.yml", "kustomization.yaml", "chart.yaml", "settings.py", "astro.config*",
    "next.config*", "svelte.config*", "nuxt.config*", "app.yaml", "procfile",
];

fn is_manifest(name: &str) -> bool {
    MANIFESTS.iter().any(|m| match (m.strip_suffix('*'), m.strip_prefix('*')) {
        (Some(prefix), _) => name.starts_with(prefix),
        (_, Some(suffix)) => name.ends_with(suffix),
        _ => name == *m,
    })
}

/// Folders below the root that are never looked in.
const SKIPPED: &[&str] = &["node_modules", "target", ".git", ".venv", "venv", "dist", "build", "__pycache__", ".next"];

/// What `root` shows of its stack: its file names and its manifests' text.
pub fn scan(root: &Path) -> Project {
    let mut project = Project::default();
    let read = |dir: &Path, prefix: &str, project: &mut Project, deeper: bool| {
        let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
        let mut subdirs = Vec::new();
        for entry in entries.flatten().take(500) {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            let Ok(kind) = entry.file_type() else { continue };
            if kind.is_dir() {
                if deeper && !SKIPPED.contains(&name.as_str()) && !name.starts_with('.') {
                    subdirs.push((entry.path(), name));
                }
                continue;
            }
            if !kind.is_file() || name.starts_with(".env") {
                continue;
            }
            project.files.insert(name.clone());
            if !prefix.is_empty() {
                project.files.insert(format!("{prefix}/{name}"));
            }
            if is_manifest(&name)
                && entry.metadata().is_ok_and(|m| m.len() <= MAX_FILE_BYTES)
                && let Ok(text) = std::fs::read_to_string(entry.path())
            {
                project.text.push_str(&text.to_lowercase());
                project.text.push('\n');
            }
        }
        subdirs
    };
    for (dir, name) in read(root, "", &mut project, true).into_iter().take(40) {
        read(&dir, &name, &mut project, false);
    }
    project
}

/// Whether `needle` is in `hay`: as a whole word when it is one plain word
/// (so `ses` is not found in `uses`), else as written.
fn found(hay: &str, needle: &str) -> bool {
    let needle = needle.to_lowercase();
    let plain = !needle.is_empty() && needle.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !plain {
        return hay.contains(&needle);
    }
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
    hay.match_indices(&needle).any(|(at, _)| {
        let before = hay[..at].chars().next_back().is_none_or(|c| !word(c));
        let after = hay[at + needle.len()..].chars().next().is_none_or(|c| !word(c));
        before && after
    })
}

impl Project {
    fn has(&self, needle: &str) -> bool {
        found(&self.text, needle) || self.files.iter().any(|f| found(f, needle))
    }

    /// Whether the project already uses `choice`.
    pub fn uses(&self, choice: &Choice) -> bool {
        choice.signals.iter().any(|s| self.has(s))
    }
}

/// Every option the project already uses, by area, in the catalog's order.
pub fn detected(project: &Project) -> Vec<(&'static Area, &'static Choice)> {
    let mut out = Vec::new();
    for area in &catalog().areas {
        for choice in &area.options {
            if project.uses(choice) {
                out.push((area, choice));
            }
        }
    }
    out
}

/// An option ranked for a decision.
#[derive(Clone, Debug)]
pub struct Ranked {
    pub choice: &'static Choice,
    pub score: i32,
    /// The project already uses it.
    pub used: bool,
}

/// The area's options for `needs`, best first: +2 for each need it is best
/// for, -2 for each it is poor for, +3 when the project already uses it; ties
/// keep the catalog's order.
pub fn rank(area: &'static Area, needs: &[String], project: &Project) -> Vec<Ranked> {
    let mut ranked: Vec<Ranked> = area
        .options
        .iter()
        .map(|choice| {
            let best = needs.iter().filter(|n| choice.best_for.contains(n)).count() as i32;
            let poor = needs.iter().filter(|n| choice.poor_for.contains(n)).count() as i32;
            let used = project.uses(choice);
            Ranked { choice, score: 2 * best - 2 * poor + if used { 3 } else { 0 }, used }
        })
        .collect();
    ranked.sort_by_key(|r| std::cmp::Reverse(r.score));
    ranked
}

/// A suggestion for what the project already uses.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Suggestion {
    pub area: String,
    pub choice: String,
    /// The option's name, for the reader.
    pub name: String,
    pub check: String,
    pub suggest: String,
    pub why: String,
}

fn applies(check: &Check, project: &Project) -> bool {
    check.present.iter().all(|p| project.has(p))
        && (check.present_any.is_empty() || check.present_any.iter().any(|p| project.has(p)))
        && !check.absent.iter().any(|a| project.has(a))
        && (check.files_absent.is_empty() || !check.files_absent.iter().any(|f| project.files.contains(&f.to_lowercase())))
}

/// The improvements the catalog suggests for what the project uses.
pub fn review(project: &Project) -> Vec<Suggestion> {
    let mut out = Vec::new();
    for (area, choice) in detected(project) {
        for check in choice.checks.iter().filter(|c| applies(c, project)) {
            out.push(Suggestion {
                area: area.id.clone(),
                choice: choice.id.clone(),
                name: choice.name.clone(),
                check: check.id.clone(),
                suggest: check.suggest.clone(),
                why: check.why.clone(),
            });
        }
    }
    out
}

/// Words that make a step about bringing something in, not about work on what is there.
const SETUP: &[&str] = &[
    "set up", "setup", "add", "choose", "pick", "select", "create", "build", "implement", "deploy", "integrate",
    "introduce", "scaffold", "design", "start", "adopt", "switch to", "move to", "migrate to", "host", "store", "send",
];

/// The area a to-do step's words call for a decision in: the one with the
/// longest trigger the step contains, when the step is about bringing
/// something in (a setup word such as "add" or "set up"); none otherwise.
pub fn area_for_step(step: &str) -> Option<&'static Area> {
    let text = step.to_lowercase();
    if !SETUP.iter().any(|w| found(&text, w)) {
        return None;
    }
    catalog()
        .areas
        .iter()
        .filter_map(|a| a.triggers.iter().filter(|t| found(&text, t)).map(|t| t.len()).max().map(|n| (a, n)))
        .max_by_key(|(_, n)| *n)
        .map(|(a, _)| a)
}

#[cfg(test)]
mod tests;
