//! The read tools: `list_dir`, `glob`, `read_file` and `grep`
//! (the chat core's spec §7.1–§7.3). Not a port; the bounds are the web
//! service's (`workspaces/views.py`), pinned by the parity golden
//! `agent/limits.json`.
//!
//! **What they see.** The non-ignored file set (§6.2 "Walks"): in a git
//! workspace one `git ls-files -z --cached --others --exclude-standard` per
//! call; otherwise the `ignore` crate's walker over `.latticeignore` and the
//! built-in defaults, links not followed and `.git` left out. Each listed path
//! passes WP1–WP9 and WP11; a file whose content is returned also passes WP10
//! (`PathRules::open_listed`, `PathRules::resolve`). Within its conversation
//! every tool sees the staged view ([`Overlay`], §7.4.2): a staged file reads
//! with its new bytes, a staged creation is listed and matched, a staged
//! deletion disappears. The conversation's overlay is `staging::Staging`
//! (row E2); [`NoOverlay`] is the disk alone.
//!
//! **What they answer.** Plain text whose first line names what was read and
//! its bounds; every cap is stated, never applied silently (§7.2). A refusal
//! is one sentence ([`ToolError`]). The result is untrusted model input.
//!
//! [`files`] gives the interface's file picker the same file set, ranked
//! against its query (`text::rank`, row D6), and [`search`] a person's search
//! across the folder the same files `grep` searches, as data
//! (`text::find`).

use std::collections::BTreeSet;
use std::io::Read;

use globset::{GlobBuilder, GlobMatcher};
use ignore::WalkBuilder;
use regex::RegexBuilder;
use serde::Deserialize;

use crate::git::dotgit::Repo;
use crate::git::runner::{GitRunner, MAX_TREE_ENTRIES};
use crate::workspace::Workspace;
use crate::workspace::ignore::Verdict;
use crate::workspace::paths::{IgnoreSource, PathError, PathRules, Want};

/// `views.MAX_FILE_BYTES`: the largest file read or searched.
pub const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
/// `views._BINARY_PROBE`: how far a NUL byte is looked for (git's own test).
pub const BINARY_PROBE: usize = 8000;
/// `read_file`'s largest `limit`, and the longest line shown whole.
pub const MAX_READ_LINES: u64 = 2000;
pub const MAX_LINE_CHARS: usize = 2000;
/// `list_dir`'s most entries, and its deepest `depth`.
pub const MAX_LIST_ENTRIES: usize = 2000;
pub const MAX_LIST_DEPTH: u8 = 3;
/// `glob`'s most paths, and its longest pattern.
pub const MAX_GLOB_PATHS: usize = 1000;
pub const MAX_GLOB_CHARS: usize = 256;
/// `grep`'s most matches, its most output, its widest context and its
/// regular expression's compiled size.
pub const MAX_GREP_MATCHES: usize = 200;
pub const MAX_GREP_OUTPUT: usize = 64 * 1024;
pub const MAX_GREP_CONTEXT: u8 = 3;
pub const MAX_REGEX_BYTES: usize = 1024 * 1024;

/// A refused or failed call: one sentence for the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolError(pub String);

impl ToolError {
    pub(crate) fn new(sentence: impl Into<String>) -> Self {
        Self(sentence.into())
    }
}

impl From<PathError> for ToolError {
    fn from(error: PathError) -> Self {
        Self(error.sentence())
    }
}

/// A path's staged state in this conversation (§7.4.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Staged {
    /// The bytes a Keep would write.
    Bytes(Vec<u8>),
    /// A staged deletion: the file is not there in this view.
    Deleted,
}

/// The conversation's staged view, by derived path.
pub trait Overlay: Send + Sync {
    /// The staged state of `path`, when this conversation staged it.
    fn staged(&self, path: &str) -> Option<Staged>;
    /// Staged creations: paths that are new in this view.
    fn created(&self) -> Vec<String>;
    /// `read_file` returned `path` (a derived path): `write_file` may now
    /// replace it (§7.4.1).
    fn note_read(&self, _path: &str) {}
}

/// No staged changes: the disk alone (what another conversation, or a
/// command, sees).
pub struct NoOverlay;

impl Overlay for NoOverlay {
    fn staged(&self, _path: &str) -> Option<Staged> {
        None
    }

    fn created(&self) -> Vec<String> {
        Vec::new()
    }
}

/// What every read tool works with.
pub struct ReadContext<'a> {
    pub workspace: &'a Workspace,
    pub runner: &'a GitRunner,
    pub overlay: &'a dyn Overlay,
}

/// `list_dir`'s arguments.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ListDirArgs {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub depth: Option<u8>,
}

/// `glob`'s arguments.
#[derive(Clone, Debug, Deserialize)]
pub struct GlobArgs {
    pub pattern: String,
}

/// `read_file`'s arguments.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct ReadFileArgs {
    pub path: String,
    #[serde(default)]
    pub offset: Option<u64>,
    #[serde(default)]
    pub limit: Option<u64>,
}

/// `grep`'s arguments.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct GrepArgs {
    pub pattern: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub glob: Option<String>,
    #[serde(default)]
    pub case_insensitive: Option<bool>,
    #[serde(default)]
    pub context: Option<u8>,
}

/// A tool's arguments from the model's JSON.
pub fn parse_args<T: serde::de::DeserializeOwned>(
    value: &serde_json::Value,
) -> Result<T, ToolError> {
    serde_json::from_value(value.clone())
        .map_err(|_| ToolError::new("The arguments do not fit this tool."))
}

pub(crate) fn rules_error(_: crate::workspace::ignore::RulesError) -> ToolError {
    ToolError::new("Lattice could not read this folder's .latticeignore, so nothing is shown.")
}

/// The non-ignored file set, as derived paths, with the staged view applied.
struct FileSet {
    paths: Vec<String>,
    created: BTreeSet<String>,
    truncated: bool,
    /// WP8b: folders left out because git cannot read their ignore rules.
    withheld: Vec<String>,
}

fn file_set(ctx: &ReadContext<'_>, rules: &PathRules<'_>) -> Result<FileSet, ToolError> {
    let (listed, truncated) = match (&ctx.workspace.repo, &rules.ignore) {
        (Repo::Git(local), _) => {
            let listing = ctx.runner.ls_files(local).map_err(|error| {
                ToolError::new(format!(
                    "The folder's files could not be listed: {}",
                    error.sentence()
                ))
            })?;
            (listing.paths, listing.truncated)
        }
        (_, IgnoreSource::Lattice(ignore_rules)) => {
            let root = crate::workspace::shown_path(rules.root);
            let ignore_rules = (*ignore_rules).clone();
            let filter_root = std::path::PathBuf::from(&root);
            let mut builder = WalkBuilder::new(&root);
            builder
                .standard_filters(false)
                .hidden(false)
                .follow_links(false)
                .filter_entry(move |entry| {
                    let Ok(relative) = entry.path().strip_prefix(&filter_root) else {
                        return true;
                    };
                    if relative.as_os_str().is_empty() {
                        return true;
                    }
                    if entry.file_name().eq_ignore_ascii_case(".git") {
                        return false;
                    }
                    let relative = relative.to_string_lossy().replace('\\', "/");
                    let is_dir = entry.file_type().is_some_and(|kind| kind.is_dir());
                    ignore_rules.check(&relative, is_dir) == Verdict::Allowed
                });
            let mut paths = Vec::new();
            let mut truncated = false;
            for entry in builder.build().flatten() {
                let Some(kind) = entry.file_type() else {
                    continue;
                };
                #[cfg(windows)]
                let link_to_folder = {
                    use std::os::windows::fs::FileTypeExt;
                    kind.is_symlink_dir()
                };
                #[cfg(not(windows))]
                let link_to_folder = false;
                if !(kind.is_file() || kind.is_symlink()) || link_to_folder {
                    continue;
                }
                let Ok(relative) = entry.path().strip_prefix(&root) else {
                    continue;
                };
                if paths.len() == MAX_TREE_ENTRIES {
                    truncated = true;
                    break;
                }
                paths.push(relative.to_string_lossy().replace('\\', "/"));
            }
            (paths, truncated)
        }
        (_, IgnoreSource::Git(..)) => (Vec::new(), false),
    };
    let mut seen = BTreeSet::new();
    let mut paths = Vec::new();
    for path in listed {
        if rules.listed(&path).is_err() || ctx.overlay.staged(&path) == Some(Staged::Deleted) {
            continue;
        }
        if seen.insert(path.clone()) {
            paths.push(path);
        }
    }
    let mut created = BTreeSet::new();
    for path in ctx.overlay.created() {
        if rules.listed(&path).is_ok() && seen.insert(path.clone()) {
            created.insert(path.clone());
            paths.push(path);
        }
    }
    paths.sort();
    Ok(FileSet {
        paths,
        created,
        truncated,
        withheld: rules.withheld(),
    })
}

/// WP8b's part of a listing's first line: which folders were left out, and
/// why.
fn withheld_note(set: &FileSet) -> String {
    if set.withheld.is_empty() {
        return String::new();
    }
    let places: Vec<String> = set
        .withheld
        .iter()
        .map(|folder| {
            if folder.is_empty() {
                "the whole folder".to_owned()
            } else {
                format!("{folder}/")
            }
        })
        .collect();
    format!(
        "; not shown, because git cannot read their ignore rules: {}",
        places.join(", ")
    )
}

fn truncated_note(set: &FileSet) -> &'static str {
    if set.truncated {
        "; the folder lists more than 100,000 files, and only the first 100,000 were looked at"
    } else {
        ""
    }
}

/// Decoded text of a file's bytes: UTF-8 (a byte-order mark removed), or
/// UTF-16 with a byte-order mark; `Err` with the size for a binary file.
fn decode(bytes: &[u8]) -> Result<(String, &'static str), u64> {
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        let units: Vec<u16> = rest
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        return Ok((String::from_utf16_lossy(&units), "; UTF-16"));
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        let units: Vec<u16> = rest
            .chunks_exact(2)
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect();
        return Ok((String::from_utf16_lossy(&units), "; UTF-16"));
    }
    if bytes[..bytes.len().min(BINARY_PROBE)].contains(&0) {
        return Err(bytes.len() as u64);
    }
    let body = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    match std::str::from_utf8(body) {
        Ok(text) => Ok((text.to_owned(), "")),
        Err(_) => Ok((
            String::from_utf8_lossy(body).into_owned(),
            "; not valid UTF-8, shown with replacement characters",
        )),
    }
}

/// A file's lines, without their line ends.
fn lines_of(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let body = text.strip_suffix('\n').unwrap_or(text);
    body.split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect()
}

fn cut(line: &str) -> String {
    if line.chars().count() <= MAX_LINE_CHARS {
        return line.to_owned();
    }
    let mut out: String = line.chars().take(MAX_LINE_CHARS).collect();
    out.push_str(" [line cut at 2,000 characters]");
    out
}

/// Why an open file's bytes were not read.
pub(crate) enum Unread {
    TooLarge(u64),
    Unreadable,
}

/// At most [`MAX_FILE_BYTES`] of an open file.
pub(crate) fn read_bounded(file: std::fs::File) -> Result<Vec<u8>, Unread> {
    let size = file.metadata().map_err(|_| Unread::Unreadable)?.len();
    if size > MAX_FILE_BYTES {
        return Err(Unread::TooLarge(size));
    }
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| Unread::Unreadable)?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err(Unread::TooLarge(bytes.len() as u64));
    }
    Ok(bytes)
}

/// The bytes of a resolved file, or why not: staged bytes first.
fn bytes_of(
    ctx: &ReadContext<'_>,
    derived: &str,
    file: Option<std::fs::File>,
) -> Result<(Vec<u8>, bool), ToolError> {
    match ctx.overlay.staged(derived) {
        Some(Staged::Bytes(bytes)) => return Ok((bytes, true)),
        Some(Staged::Deleted) => return Err(PathError::Missing.into()),
        None => {}
    }
    let file = file.ok_or_else(|| ToolError::from(PathError::Missing))?;
    match read_bounded(file) {
        Ok(bytes) => Ok((bytes, false)),
        Err(Unread::TooLarge(size)) => Err(ToolError::new(format!(
            "That file is {size} bytes; Lattice reads files of at most 2 MiB (2,097,152 bytes)."
        ))),
        Err(Unread::Unreadable) => Err(PathError::Unreadable.into()),
    }
}

/// `read_file` (§7.3).
pub fn read_file(ctx: &ReadContext<'_>, args: &ReadFileArgs) -> Result<String, ToolError> {
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let resolved = rules.resolve(&args.path, Want::MayCreate)?;
            if resolved.is_dir {
                return Err(ToolError::new("That is a folder; use list_dir."));
            }
            let (bytes, staged) = bytes_of(ctx, &resolved.derived, resolved.file)?;
            let (text, encoding) = decode(&bytes).map_err(|size| {
                ToolError::new(format!("binary file, {size} bytes"))
            })?;
            ctx.overlay.note_read(&resolved.derived);
            let lines = lines_of(&text);
            let total = lines.len() as u64;
            let offset = args.offset.unwrap_or(1).max(1);
            let asked = args.limit.unwrap_or(MAX_READ_LINES).max(1);
            let limit = asked.min(MAX_READ_LINES);
            let staged_note = if staged { "; staged version" } else { "" };
            let limit_note = if asked > MAX_READ_LINES {
                "; at most 2,000 lines are shown at a time"
            } else {
                ""
            };
            let derived = &resolved.derived;
            if total == 0 {
                return Ok(format!("{derived} (empty file{staged_note}{encoding})"));
            }
            if offset > total {
                return Ok(format!(
                    "{derived} ({total} lines; none from line {offset}{staged_note}{encoding})"
                ));
            }
            let last = (offset + limit - 1).min(total);
            let mut out = format!(
                "{derived} ({total} lines; lines {offset}-{last} shown{limit_note}{staged_note}{encoding})"
            );
            for number in offset..=last {
                out.push('\n');
                out.push_str(&format!("{number}\t{}", cut(lines[(number - 1) as usize])));
            }
            Ok(out)
        })
        .map_err(rules_error)?
}

/// `list_dir` (§7.3): directories first, then files, sorted.
pub fn list_dir(ctx: &ReadContext<'_>, args: &ListDirArgs) -> Result<String, ToolError> {
    let depth = args.depth.unwrap_or(1);
    if !(1..=MAX_LIST_DEPTH).contains(&depth) {
        return Err(ToolError::new("depth is 1, 2 or 3."));
    }
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let (prefix, shown) = match args
                .path
                .as_deref()
                .filter(|path| !path.is_empty() && *path != ".")
            {
                None => (String::new(), ".".to_owned()),
                Some(path) => {
                    let resolved = rules.resolve(path, Want::Existing)?;
                    if !resolved.is_dir {
                        return Err(ToolError::new("That is a file; use read_file."));
                    }
                    (format!("{}/", resolved.derived), resolved.derived)
                }
            };
            let set = file_set(ctx, rules)?;
            let mut dirs = BTreeSet::new();
            let mut files = Vec::new();
            for path in &set.paths {
                let Some(rest) = path.strip_prefix(&prefix) else {
                    continue;
                };
                let parts: Vec<&str> = rest.split('/').collect();
                for level in 1..=usize::from(depth) {
                    if parts.len() > level {
                        dirs.insert(format!("{}/", parts[..level].join("/")));
                    }
                }
                if parts.len() <= usize::from(depth) {
                    let mark = if set.created.contains(path) {
                        " (staged)"
                    } else {
                        ""
                    };
                    files.push(format!("{rest}{mark}"));
                }
            }
            files.sort();
            let entries: Vec<String> = dirs.into_iter().chain(files).collect();
            let total = entries.len();
            let mut out = format!(
                "{shown} ({total} entries; depth {depth}{}{})",
                truncated_note(&set),
                withheld_note(&set)
            );
            for entry in entries.iter().take(MAX_LIST_ENTRIES) {
                out.push('\n');
                out.push_str(entry);
            }
            if total > MAX_LIST_ENTRIES {
                out.push_str(&format!("\n{} more not shown", total - MAX_LIST_ENTRIES));
            }
            Ok(out)
        })
        .map_err(rules_error)?
}

fn matcher(pattern: &str) -> Result<GlobMatcher, ToolError> {
    if pattern.chars().count() > MAX_GLOB_CHARS {
        return Err(ToolError::new("A glob pattern is at most 256 characters."));
    }
    GlobBuilder::new(pattern)
        .literal_separator(true)
        .case_insensitive(true)
        .build()
        .map(|glob| glob.compile_matcher())
        .map_err(|_| ToolError::new("That glob pattern is not valid."))
}

/// `glob` (§7.3): `/` is a literal separator; at most 1,000 paths, sorted.
pub fn glob(ctx: &ReadContext<'_>, args: &GlobArgs) -> Result<String, ToolError> {
    let matcher = matcher(&args.pattern)?;
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let set = file_set(ctx, rules)?;
            let matched: Vec<&String> = set
                .paths
                .iter()
                .filter(|path| matcher.is_match(path.as_str()))
                .collect();
            let total = matched.len();
            let mut out = format!(
                "{total} paths match {}{}{}",
                args.pattern,
                truncated_note(&set),
                withheld_note(&set)
            );
            for path in matched.iter().take(MAX_GLOB_PATHS) {
                out.push('\n');
                out.push_str(path);
                if set.created.contains(*path) {
                    out.push_str(" (staged)");
                }
            }
            if total > MAX_GLOB_PATHS {
                out.push_str(&format!("\n{} more not shown", total - MAX_GLOB_PATHS));
            }
            Ok(out)
        })
        .map_err(rules_error)?
}

/// `grep` (§7.3): the `regex` crate, linear time, a 1 MiB compiled size;
/// files of at most 2 MiB, binaries skipped; at most 200 matches and 64 KiB.
pub fn grep(ctx: &ReadContext<'_>, args: &GrepArgs) -> Result<String, ToolError> {
    let context = args.context.unwrap_or(0);
    if context > MAX_GREP_CONTEXT {
        return Err(ToolError::new("context is at most 3 lines."));
    }
    let regex = RegexBuilder::new(&args.pattern)
        .case_insensitive(args.case_insensitive.unwrap_or(false))
        .size_limit(MAX_REGEX_BYTES)
        .build()
        .map_err(|error| match error {
            regex::Error::CompiledTooBig(_) => {
                ToolError::new("That pattern is too large; Lattice compiles patterns of at most 1 MiB.")
            }
            _ => ToolError::new("That pattern is not a valid regular expression (no back-references or look-around)."),
        })?;
    let filter = args.glob.as_deref().map(matcher).transpose()?;
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let (scope_prefix, scope_file) = match args
                .path
                .as_deref()
                .filter(|path| !path.is_empty() && *path != ".")
            {
                None => (String::new(), None),
                Some(path) => {
                    let resolved = rules.resolve(path, Want::MayCreate)?;
                    if resolved.is_dir {
                        (format!("{}/", resolved.derived), None)
                    } else {
                        (String::new(), Some(resolved.derived))
                    }
                }
            };
            let set = file_set(ctx, rules)?;
            let mut out = String::new();
            let mut shown = 0usize;
            let mut found = 0usize;
            let mut files_matched = 0usize;
            let (mut too_large, mut binary, mut unreadable, mut cut_output) =
                (0usize, 0usize, 0usize, false);
            for path in &set.paths {
                let in_scope = match &scope_file {
                    Some(file) => path == file,
                    None => path.starts_with(&scope_prefix),
                };
                if !in_scope
                    || filter
                        .as_ref()
                        .is_some_and(|glob| !glob.is_match(path.as_str()))
                {
                    continue;
                }
                let bytes = match ctx.overlay.staged(path) {
                    Some(Staged::Bytes(bytes)) => bytes,
                    Some(Staged::Deleted) => continue,
                    None => {
                        let opened = match rules.open_listed(path) {
                            Ok(opened) if !opened.is_dir => opened,
                            Ok(_) => continue,
                            Err(_) => {
                                unreadable += 1;
                                continue;
                            }
                        };
                        let Some(file) = opened.file else {
                            unreadable += 1;
                            continue;
                        };
                        match read_bounded(file) {
                            Ok(bytes) => bytes,
                            Err(Unread::TooLarge(_)) => {
                                too_large += 1;
                                continue;
                            }
                            Err(Unread::Unreadable) => {
                                unreadable += 1;
                                continue;
                            }
                        }
                    }
                };
                let Ok((text, _)) = decode(&bytes) else {
                    binary += 1;
                    continue;
                };
                let lines = lines_of(&text);
                let hits: Vec<usize> = lines
                    .iter()
                    .enumerate()
                    .filter(|(_, line)| regex.is_match(line))
                    .map(|(at, _)| at)
                    .collect();
                if hits.is_empty() {
                    continue;
                }
                files_matched += 1;
                found += hits.len();
                let mut last_printed: Option<usize> = None;
                for &hit in &hits {
                    if shown == MAX_GREP_MATCHES || cut_output {
                        break;
                    }
                    let from = hit.saturating_sub(usize::from(context));
                    let to = (hit + usize::from(context)).min(lines.len() - 1);
                    let mut block = String::new();
                    // With context, separate blocks that do not touch, as grep does.
                    let separate =
                        context > 0 && last_printed.map_or(!out.is_empty(), |last| from > last + 1);
                    if separate {
                        block.push_str("--\n");
                    }
                    let start = last_printed.map_or(from, |last| from.max(last + 1));
                    for (at, line) in lines.iter().enumerate().take(to + 1).skip(start) {
                        let mark = if at == hit || hits.binary_search(&at).is_ok() {
                            ':'
                        } else {
                            '-'
                        };
                        block.push_str(&format!("{path}{mark}{}{mark}{}\n", at + 1, cut(line)));
                    }
                    if out.len() + block.len() > MAX_GREP_OUTPUT {
                        cut_output = true;
                        break;
                    }
                    out.push_str(&block);
                    last_printed = Some(to);
                    shown += 1;
                }
            }
            let mut head = format!("{found} matches in {files_matched} files");
            if found > shown {
                head.push_str(&format!("; {shown} shown, {} more matches", found - shown));
            }
            if cut_output {
                head.push_str("; the output stopped at 64 KiB");
            }
            if too_large > 0 {
                head.push_str(&format!("; {too_large} files over 2 MiB were not searched"));
            }
            if binary > 0 {
                head.push_str(&format!("; {binary} binary files were not searched"));
            }
            if unreadable > 0 {
                head.push_str(&format!("; {unreadable} files could not be read"));
            }
            head.push_str(truncated_note(&set));
            head.push_str(&withheld_note(&set));
            if out.is_empty() {
                Ok(head)
            } else {
                Ok(format!("{head}\n{}", out.trim_end_matches('\n')))
            }
        })
        .map_err(rules_error)?
}

/// The most paths `files` returns.
pub const MAX_FILES_LIMIT: u32 = 1000;

/// `files` (§4.3): the file picker's list, the folder's non-ignored files and
/// the conversation's staged creations, ranked against `query` by the
/// shipping app's quick-open rule (`text::rank`). `limit` 0 means the
/// shipping app's 50; more than [`MAX_FILES_LIMIT`] is cut to it.
pub fn files(
    ctx: &ReadContext<'_>,
    query: &str,
    limit: u32,
) -> Result<Vec<lattice_protocol::conversation::RankedPath>, ToolError> {
    let limit = match limit {
        0 => crate::text::rank::DEFAULT_LIMIT,
        n => n.min(MAX_FILES_LIMIT) as usize,
    };
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let set = file_set(ctx, rules)?;
            Ok(crate::text::rank::rank_paths(
                query,
                set.paths.iter().map(String::as_str),
                limit,
            ))
        })
        .map_err(rules_error)?
}

/// The folder's whole listing, the set [`files`] ranks: the non-ignored files and the staged creations, as derived
/// paths, sorted. For an interface's file tree (CENTCOM's Explorer), which shows every file the read tools see and no
/// other. `truncated`: a folder without git held more than the walk's cap, or git's listing was cut; `withheld`: the
/// folders WP8b left out (`""` is the whole folder).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileListing {
    pub paths: Vec<String>,
    pub truncated: bool,
    pub withheld: Vec<String>,
}

/// [`FileListing`] for `ctx`'s folder.
pub fn listing(ctx: &ReadContext<'_>) -> Result<FileListing, ToolError> {
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let set = file_set(ctx, rules)?;
            Ok(FileListing {
                paths: set.paths,
                truncated: set.truncated,
                withheld: set.withheld,
            })
        })
        .map_err(rules_error)?
}

/// [`search`]'s arguments: a person's search across the folder (CENTCOM's
/// Search view).
#[derive(Clone, Debug, Default)]
pub struct SearchArgs {
    pub query: crate::text::find::Query,
    /// Only files whose path fits this glob (as `grep`'s `glob`); empty is
    /// every file.
    pub glob: Option<String>,
    /// The most matches returned; the count goes on past it.
    pub max_matches: usize,
}

/// A matching line: its number (from 1), its text (at most
/// [`MAX_LINE_CHARS`] characters) and where in that text it matches, as byte
/// ranges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineMatch {
    pub line: u32,
    pub text: String,
    pub ranges: Vec<std::ops::Range<usize>>,
}

/// A file's matching lines, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileMatches {
    pub path: String,
    pub lines: Vec<LineMatch>,
}

/// What a [`search`] found, and what it could not look at: every cap and
/// every file left unsearched is counted, never dropped silently.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SearchReport {
    pub files: Vec<FileMatches>,
    /// Every match found, those past `max_matches` included.
    pub matches: usize,
    /// Files with at least one match, those past `max_matches` included.
    pub files_matched: usize,
    /// The matches in `files`.
    pub shown: usize,
    /// Files over [`MAX_FILE_BYTES`], not searched.
    pub too_large: usize,
    pub binary: usize,
    pub unreadable: usize,
    /// The folder lists more files than are looked at (as [`FileListing`]).
    pub truncated: bool,
    /// Folders left out because git cannot read their ignore rules.
    pub withheld: Vec<String>,
}

/// A person's search across the folder: exactly the files `grep` searches
/// (the same set, path rules, staged view, size bound and binary rule), each
/// line matched by `args.query` (`text::find`), as data for a view rather
/// than text for a model. A line longer than [`MAX_LINE_CHARS`] characters is
/// returned cut there, with the ranges that fall in what is kept.
pub fn search(ctx: &ReadContext<'_>, args: &SearchArgs) -> Result<SearchReport, ToolError> {
    let found = crate::text::find::Matcher::new(&args.query).map_err(ToolError::new)?;
    let filter = args
        .glob
        .as_deref()
        .map(str::trim)
        .filter(|glob| !glob.is_empty())
        .map(matcher)
        .transpose()?;
    ctx.workspace
        .with_rules(ctx.runner, |rules| {
            let set = file_set(ctx, rules)?;
            let mut report = SearchReport {
                truncated: set.truncated,
                withheld: set.withheld.clone(),
                ..SearchReport::default()
            };
            for path in &set.paths {
                if filter
                    .as_ref()
                    .is_some_and(|glob| !glob.is_match(path.as_str()))
                {
                    continue;
                }
                let bytes = match ctx.overlay.staged(path) {
                    Some(Staged::Bytes(bytes)) => bytes,
                    Some(Staged::Deleted) => continue,
                    None => {
                        let opened = match rules.open_listed(path) {
                            Ok(opened) if !opened.is_dir => opened,
                            Ok(_) => continue,
                            Err(_) => {
                                report.unreadable += 1;
                                continue;
                            }
                        };
                        let Some(file) = opened.file else {
                            report.unreadable += 1;
                            continue;
                        };
                        match read_bounded(file) {
                            Ok(bytes) => bytes,
                            Err(Unread::TooLarge(_)) => {
                                report.too_large += 1;
                                continue;
                            }
                            Err(Unread::Unreadable) => {
                                report.unreadable += 1;
                                continue;
                            }
                        }
                    }
                };
                let Ok((text, _)) = decode(&bytes) else {
                    report.binary += 1;
                    continue;
                };
                let mut lines = Vec::new();
                let mut matched = false;
                for (at, line) in lines_of(&text).into_iter().enumerate() {
                    let ranges = found.find_in(line);
                    if ranges.is_empty() {
                        continue;
                    }
                    matched = true;
                    report.matches += ranges.len();
                    if report.shown >= args.max_matches {
                        continue;
                    }
                    let kept: String = line.chars().take(MAX_LINE_CHARS).collect();
                    let room = args.max_matches - report.shown;
                    let ranges: Vec<std::ops::Range<usize>> = ranges
                        .into_iter()
                        .filter(|range| range.start < kept.len())
                        .map(|range| range.start..range.end.min(kept.len()))
                        .take(room)
                        .collect();
                    if ranges.is_empty() {
                        continue;
                    }
                    report.shown += ranges.len();
                    lines.push(LineMatch {
                        line: at as u32 + 1,
                        text: kept,
                        ranges,
                    });
                }
                if matched {
                    report.files_matched += 1;
                }
                if !lines.is_empty() {
                    report.files.push(FileMatches {
                        path: path.clone(),
                        lines,
                    });
                }
            }
            Ok(report)
        })
        .map_err(rules_error)?
}
