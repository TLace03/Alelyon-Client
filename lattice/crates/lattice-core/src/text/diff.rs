//! The review's line diff: hunks with 3 lines of context, their ids, the
//! application of a chosen set of hunks, and intraline marks
//! (the chat core's spec §7.5). Not a port; the payload shapes are the
//! shipping app's `Hunk` and `DiffLine` (`lattice_protocol::conversation`).
//!
//! - A text is cut into lines **with** their line ends, so composing a text
//!   from line slices gives back the exact bytes of each side: a CRLF file
//!   stays CRLF, a last line without a line end stays so.
//! - [`plan`] diffs a change's base text against its new text (`similar`'s
//!   Myers diff over lines) and groups the operations into hunks with
//!   [`CONTEXT`] lines of context. Each hunk's id is 16 hex characters of
//!   SHA-256 over the change id, the base's hash, the hunk's ranges and its
//!   lines, so it is stable while the change is unchanged (§4.1).
//! - [`apply`] composes a text that takes the new side of the chosen hunks
//!   and the base side of every other: a partial Keep writes it, and a hunk
//!   Undo stages it (with the choice reversed).
//! - A diff of more than [`MAX_DIFF_LINES`] lines (`views.MAX_DIFF_LINES`) is
//!   `truncated`: shown cut, and kept or undone only whole.
//! - [`intraline`] marks the changed words of a removed and an added line,
//!   for line pairs under [`MAX_INTRALINE_CHARS`] characters. The protocol's
//!   `DiffLine` (the shipping app's) has no field for them yet, so the
//!   interface payload does not carry them in this row.

use std::collections::BTreeSet;
use std::ops::Range;

use lattice_protocol::conversation::{DiffLine, DiffLineKind, Hunk, HunkId, HunkState};
use similar::{Algorithm, DiffOp, DiffTag};

use crate::sha::sha256_hex;

/// Lines of context around each change.
pub const CONTEXT: usize = 3;
/// `views.MAX_DIFF_LINES`: past this, a diff is cut and kept only whole.
pub const MAX_DIFF_LINES: usize = 6000;
/// Line pairs at least this long get no intraline marks.
pub const MAX_INTRALINE_CHARS: usize = 500;

/// `text` cut into lines, each with its own line end (the last may have
/// none).
pub fn lines(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

/// A line as the interface shows it: without its line end.
fn shown(line: &str) -> String {
    line.strip_suffix('\n')
        .map(|rest| rest.strip_suffix('\r').unwrap_or(rest))
        .unwrap_or(line)
        .to_owned()
}

/// One planned hunk: what the interface sees, and which of the diff's
/// operations it covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedHunk {
    pub hunk: Hunk,
    /// The diff's change operations (not context) this hunk holds.
    ops: Vec<DiffOp>,
}

/// A change's diff, ready to show and to apply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub hunks: Vec<PlannedHunk>,
    /// Every operation of the line diff, in order.
    ops: Vec<DiffOp>,
    /// More than [`MAX_DIFF_LINES`] lines: kept or undone only whole.
    pub truncated: bool,
}

impl Plan {
    /// The interface's hunks, each with the state given.
    pub fn hunks(&self) -> Vec<Hunk> {
        self.hunks
            .iter()
            .map(|planned| planned.hunk.clone())
            .collect()
    }

    /// Every hunk id, in order.
    pub fn ids(&self) -> Vec<HunkId> {
        self.hunks
            .iter()
            .map(|planned| planned.hunk.id.clone())
            .collect()
    }

    /// The lines it adds and removes.
    pub fn counts(&self) -> (u32, u32) {
        let (mut added, mut removed) = (0u32, 0u32);
        for op in &self.ops {
            let (tag, old, new) = op.as_tag_tuple();
            match tag {
                DiffTag::Equal => {}
                DiffTag::Delete => removed = removed.saturating_add(len32(&old)),
                DiffTag::Insert => added = added.saturating_add(len32(&new)),
                DiffTag::Replace => {
                    removed = removed.saturating_add(len32(&old));
                    added = added.saturating_add(len32(&new));
                }
            }
        }
        (added, removed)
    }
}

fn len32(range: &Range<usize>) -> u32 {
    u32::try_from(range.len()).unwrap_or(u32::MAX)
}

/// A unified-diff start: 1-based, or the line before an empty range.
fn start(range: &Range<usize>) -> u32 {
    let base = if range.is_empty() {
        range.start
    } else {
        range.start + 1
    };
    u32::try_from(base).unwrap_or(u32::MAX)
}

/// The hunk id (§4.1): SHA-256 over the change id, the base's hash, the
/// ranges and the lines.
fn hunk_id(change: &str, base_sha: &str, hunk: &Hunk) -> HunkId {
    let lines: Vec<(&str, &str)> = hunk
        .lines
        .iter()
        .map(|line| {
            let kind = match line.kind {
                DiffLineKind::Context => " ",
                DiffLineKind::Add => "+",
                DiffLineKind::Remove => "-",
            };
            (kind, line.text.as_str())
        })
        .collect();
    let key = serde_json::json!([
        change,
        base_sha,
        hunk.old_start,
        hunk.old_lines,
        hunk.new_start,
        hunk.new_lines,
        lines
    ]);
    sha256_hex(key.to_string().as_bytes())[..16].to_owned()
}

/// The diff of `old` against `new` for the change `change` whose base hash
/// is `base_sha`, in hunks with [`CONTEXT`] lines of context.
pub fn plan(change: &str, base_sha: &str, old: &str, new: &str) -> Plan {
    let old_lines = lines(old);
    let new_lines = lines(new);
    let ops = similar::capture_diff_slices(Algorithm::Myers, &old_lines, &new_lines);
    let groups = similar::group_diff_ops(ops.clone(), CONTEXT);
    let mut hunks = Vec::new();
    let mut shown_lines = 0usize;
    let mut truncated = false;
    for group in groups {
        let (Some(first), Some(last)) = (group.first(), group.last()) else {
            continue;
        };
        let old_range = first.old_range().start..last.old_range().end;
        let new_range = first.new_range().start..last.new_range().end;
        let mut diff_lines = Vec::new();
        let mut change_ops = Vec::new();
        for op in &group {
            let (tag, old_at, new_at) = op.as_tag_tuple();
            match tag {
                DiffTag::Equal => {
                    for line in &old_lines[old_at] {
                        diff_lines.push(DiffLine {
                            kind: DiffLineKind::Context,
                            text: shown(line),
                        });
                    }
                }
                _ => {
                    change_ops.push(*op);
                    for line in &old_lines[old_at] {
                        diff_lines.push(DiffLine {
                            kind: DiffLineKind::Remove,
                            text: shown(line),
                        });
                    }
                    for line in &new_lines[new_at] {
                        diff_lines.push(DiffLine {
                            kind: DiffLineKind::Add,
                            text: shown(line),
                        });
                    }
                }
            }
        }
        if shown_lines + diff_lines.len() > MAX_DIFF_LINES {
            truncated = true;
            break;
        }
        shown_lines += diff_lines.len();
        let mut hunk = Hunk {
            id: String::new(),
            state: HunkState::Pending,
            old_start: start(&old_range),
            old_lines: len32(&old_range),
            new_start: start(&new_range),
            new_lines: len32(&new_range),
            section: String::new(),
            lines: diff_lines,
        };
        hunk.id = hunk_id(change, base_sha, &hunk);
        hunks.push(PlannedHunk {
            hunk,
            ops: change_ops,
        });
    }
    Plan {
        hunks,
        ops,
        truncated,
    }
}

/// The text that takes the new side of the hunks in `take` and the base side
/// of every other change: line slices of each side, joined, so every line
/// keeps its own line end.
pub fn apply(plan: &Plan, old: &str, new: &str, take: &BTreeSet<HunkId>) -> String {
    let old_lines = lines(old);
    let new_lines = lines(new);
    let taken: Vec<&DiffOp> = plan
        .hunks
        .iter()
        .filter(|planned| take.contains(&planned.hunk.id))
        .flat_map(|planned| planned.ops.iter())
        .collect();
    let mut out = String::with_capacity(old.len().max(new.len()));
    for op in &plan.ops {
        let (tag, old_at, new_at) = op.as_tag_tuple();
        let side: &[&str] = if tag != DiffTag::Equal && taken.contains(&op) {
            &new_lines[new_at]
        } else {
            &old_lines[old_at]
        };
        for line in side {
            out.push_str(line);
        }
    }
    out
}

/// Intraline marks: the changed byte ranges of the removed line, then of
/// the added line.
pub type Marks = (Vec<Range<usize>>, Vec<Range<usize>>);

/// The changed spans of a removed line and of the added line paired with it
/// (byte ranges of each, by `similar`'s word diff), or `None` when either is
/// [`MAX_INTRALINE_CHARS`] characters or longer.
pub fn intraline(old: &str, new: &str) -> Option<Marks> {
    if old.chars().count() >= MAX_INTRALINE_CHARS || new.chars().count() >= MAX_INTRALINE_CHARS {
        return None;
    }
    let diff = similar::TextDiff::from_words(old, new);
    let (mut old_at, mut new_at) = (0usize, 0usize);
    let (mut old_marks, mut new_marks): (Vec<Range<usize>>, Vec<Range<usize>>) =
        (Vec::new(), Vec::new());
    for change in diff.iter_all_changes() {
        let len = change.value().len();
        match change.tag() {
            similar::ChangeTag::Equal => {
                old_at += len;
                new_at += len;
            }
            similar::ChangeTag::Delete => {
                push_mark(&mut old_marks, old_at..old_at + len);
                old_at += len;
            }
            similar::ChangeTag::Insert => {
                push_mark(&mut new_marks, new_at..new_at + len);
                new_at += len;
            }
        }
    }
    Some((old_marks, new_marks))
}

/// Add `range`, joining it to the last mark when they touch.
fn push_mark(marks: &mut Vec<Range<usize>>, range: Range<usize>) {
    match marks.last_mut() {
        Some(last) if last.end == range.start => last.end = range.end,
        _ => marks.push(range),
    }
}

#[cfg(test)]
mod tests;
