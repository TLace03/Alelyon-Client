//! The review's diff in memory (the chat core's spec §7.5): hunks and
//! their ids, applying a chosen set, line ends kept, the 6,000-line cut and
//! intraline marks.

use std::collections::BTreeSet;

use lattice_protocol::conversation::{DiffLineKind, is_hunk_id};

use super::{MAX_DIFF_LINES, apply, intraline, lines, plan};

fn numbered(n: usize) -> String {
    (1..=n).map(|i| format!("line {i}\n")).collect()
}

#[test]
fn lines_keep_their_ends() {
    assert_eq!(lines("a\r\nb\nc"), vec!["a\r\n", "b\n", "c"]);
    assert!(lines("").is_empty());
}

#[test]
fn hunks_have_three_lines_of_context_and_stable_ids() {
    let old = numbered(20);
    let new = old
        .replace("line 2\n", "line two\n")
        .replace("line 18\n", "line eighteen\n");
    let planned = plan("ch_0123456789abcdef", "base", &old, &new);
    let hunks = planned.hunks();
    assert_eq!(hunks.len(), 2, "two changes 16 lines apart are two hunks");
    let first = &hunks[0];
    assert_eq!(
        (
            first.old_start,
            first.old_lines,
            first.new_start,
            first.new_lines
        ),
        (1, 5, 1, 5)
    );
    let kinds: Vec<DiffLineKind> = first.lines.iter().map(|line| line.kind).collect();
    assert_eq!(
        kinds,
        vec![
            DiffLineKind::Context,
            DiffLineKind::Remove,
            DiffLineKind::Add,
            DiffLineKind::Context,
            DiffLineKind::Context,
            DiffLineKind::Context,
        ]
    );
    assert_eq!(first.lines[2].text, "line two");
    assert!(hunks.iter().all(|hunk| is_hunk_id(&hunk.id)));
    assert_ne!(hunks[0].id, hunks[1].id);
    assert_eq!(
        plan("ch_0123456789abcdef", "base", &old, &new).ids(),
        planned.ids(),
        "the same change gives the same ids"
    );
    assert_ne!(
        plan("ch_0123456789abcdee", "base", &old, &new).ids(),
        planned.ids(),
        "another change gives other ids"
    );
    assert_eq!(planned.counts(), (2, 2));
}

#[test]
fn applying_a_set_of_hunks_takes_their_new_side_only() {
    let old = numbered(20);
    let new = old
        .replace("line 2\n", "line two\n")
        .replace("line 18\n", "line eighteen\n");
    let planned = plan("ch_0123456789abcdef", "base", &old, &new);
    let ids = planned.ids();
    let all: BTreeSet<String> = ids.iter().cloned().collect();
    assert_eq!(apply(&planned, &old, &new, &all), new);
    assert_eq!(apply(&planned, &old, &new, &BTreeSet::new()), old);
    let first: BTreeSet<String> = [ids[0].clone()].into();
    assert_eq!(
        apply(&planned, &old, &new, &first),
        old.replace("line 2\n", "line two\n")
    );
}

/// Each line keeps its own line end: a CRLF text stays CRLF, and a last line
/// without one stays so.
#[test]
fn applying_keeps_every_line_end() {
    let old = "a\r\nb\r\nc";
    let new = "a\r\nB\r\nc!";
    let planned = plan("ch_0123456789abcdef", "base", old, new);
    let all: BTreeSet<String> = planned.ids().into_iter().collect();
    assert_eq!(apply(&planned, old, new, &all), new);
    assert_eq!(
        planned.hunks()[0].lines[0].text,
        "a",
        "shown without its end"
    );
}

#[test]
fn a_diff_past_6000_lines_is_truncated() {
    let old = numbered(MAX_DIFF_LINES);
    let new = old.replace("line", "LINE");
    let planned = plan("ch_0123456789abcdef", "base", &old, &new);
    assert!(planned.truncated);
    let small = plan("ch_0123456789abcdef", "base", "a\n", "b\n");
    assert!(!small.truncated);
}

#[test]
fn intraline_marks_the_changed_words_of_short_lines() {
    let (old, new) = intraline("let x = 1;", "let y = 1;").unwrap();
    assert_eq!(old, vec![4..5]);
    assert_eq!(new, vec![4..5]);
    assert!(intraline(&"a".repeat(500), "b").is_none());
    assert!(intraline("a", &"b".repeat(499)).is_some());
}
