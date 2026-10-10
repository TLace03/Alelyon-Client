//! Notes: where a note's lines are found after the file changes, the lines an
//! edit wrote, and the store (kept per folder, removed, never written over
//! when it cannot be read).

use std::sync::Arc;

use super::*;
use crate::testkit::TempDir;

fn store(dir: &TempDir) -> (Notes, StateRoot) {
    let state = StateRoot::at(dir.path().join("state"));
    std::fs::create_dir_all(&state.globals).unwrap();
    let clock: Clock = Arc::new(|| 1_000.0);
    (Notes::new(&state, clock), state)
}

fn new(path: &str, start: u32, end: u32, text: &str) -> NewNote {
    NewNote {
        path: path.into(),
        start,
        end,
        text: text.into(),
        author: Author::Reader,
        chat: None,
        change: None,
    }
}

const FILE: &str = "fn a() {}\nfn b() {\n    one();\n}\nfn c() {}\n";

#[test]
fn a_note_is_at_its_lines_moves_with_them_and_is_stale_once_they_change() {
    let dir = TempDir::new("notes-place");
    let (notes, _) = store(&dir);
    let note = notes
        .add("w1", FILE, new("src/a.rs", 2, 4, " why b exists "))
        .unwrap();
    assert_eq!(note.text, "why b exists");
    assert_eq!(note.quote, "fn b() {\n    one();\n}");
    assert_eq!(note.sha256, crate::sha::sha256_hex(note.quote.as_bytes()));
    assert_eq!(place(&note, FILE), Place::At { start: 2, end: 4 });
    // CRLF line ends read as the same lines.
    assert_eq!(
        place(&note, &FILE.replace('\n', "\r\n")),
        Place::At { start: 2, end: 4 }
    );
    // Two lines added above: the same lines, two further down.
    let moved = format!("// one\n// two\n{FILE}");
    assert_eq!(place(&note, &moved), Place::Moved { start: 4, end: 6 });
    // The nearest copy wins when the lines occur twice.
    let twice = format!("{FILE}fn b() {{\n    one();\n}}\n");
    assert_eq!(place(&note, &twice), Place::At { start: 2, end: 4 });
    let shifted = format!("x\n{twice}");
    assert_eq!(place(&note, &shifted), Place::Moved { start: 3, end: 5 });
    // One of its lines changed: stale.
    assert_eq!(place(&note, &FILE.replace("one()", "two()")), Place::Stale);
    assert_eq!(place(&note, ""), Place::Stale);
}

#[test]
fn the_lines_an_edit_wrote_are_found_by_its_text_or_by_the_common_ends() {
    assert_eq!(lines_holding(FILE, "    one();\n"), Some((3, 3)));
    assert_eq!(lines_holding(FILE, "fn b() {\n    one();"), Some((2, 3)));
    assert_eq!(lines_holding(FILE, "missing"), None);
    assert_eq!(lines_holding(FILE, "  \n"), None);
    let old = "a\nb\nc\nd\n";
    assert_eq!(changed_lines(old, "a\nB\nC\nd\n"), Some((2, 3)));
    assert_eq!(changed_lines(old, "a\nb\nnew\nc\nd\n"), Some((3, 3)));
    // Lines only removed: the line after them.
    assert_eq!(changed_lines(old, "a\nd\n"), Some((2, 2)));
    assert_eq!(changed_lines("", "x\ny\n"), Some((1, 2)));
    assert_eq!(changed_lines(old, old), None);
    assert_eq!(changed_lines(old, ""), None);
}

#[test]
fn notes_are_kept_per_folder_and_file_and_removed_by_id() {
    let dir = TempDir::new("notes-store");
    let (notes, _) = store(&dir);
    assert!(notes.of_folder("w1").unwrap().is_empty());
    assert!(
        notes.add("w1", FILE, new("src/a.rs", 1, 1, "  ")).is_err(),
        "a note has words"
    );
    assert!(
        notes.add("w1", FILE, new("src/a.rs", 9, 9, "x")).is_err(),
        "its lines are in the file"
    );
    assert!(notes.add("w1", FILE, new("src/a.rs", 3, 2, "x")).is_err());
    assert!(
        notes.add("w1", FILE, new("/etc/a", 1, 1, "x")).is_err(),
        "inside the folder"
    );
    assert!(notes.add("w1", FILE, new("src\\a.rs", 1, 1, "x")).is_err());
    assert!(
        notes
            .add("w1", FILE, new("src/a.rs", 1, 1, &"x".repeat(MAX_TEXT + 1)))
            .is_err()
    );
    let one = notes
        .add("w1", FILE, new("src/a.rs", 1, 1, "first"))
        .unwrap();
    let mut agent = new("src/b.rs", 5, 5, "second");
    agent.author = Author::Agent;
    agent.chat = Some("c_1".into());
    agent.change = Some("ch_1".into());
    let two = notes.add("w1", FILE, agent).unwrap();
    notes
        .add("w2", FILE, new("src/a.rs", 1, 1, "other folder"))
        .unwrap();
    assert!(one.id.starts_with("n_") && one.id.len() == 14, "{}", one.id);
    assert_eq!(one.created, 1_000.0);
    assert_eq!(notes.of_file("w1", "src/a.rs").unwrap(), vec![one.clone()]);
    assert_eq!(
        notes.of_file("w1", "src/b.rs").unwrap()[0]
            .change
            .as_deref(),
        Some("ch_1")
    );
    assert_eq!(notes.of_folder("w1").unwrap().len(), 2);
    notes.remove("w1", &one.id).unwrap();
    assert!(notes.remove("w1", &one.id).is_err(), "removed once");
    assert_eq!(notes.of_folder("w1").unwrap(), vec![two]);
    assert_eq!(notes.of_folder("w2").unwrap().len(), 1);
}

#[test]
fn a_long_span_is_cut_to_its_first_lines() {
    let dir = TempDir::new("notes-long");
    let (notes, _) = store(&dir);
    let text: String = (1..=500).map(|n| format!("line {n}\n")).collect();
    let note = notes
        .add("w1", &text, new("a.txt", 10, 400, "long"))
        .unwrap();
    assert_eq!((note.start, note.end), (10, 10 + MAX_LINES - 1));
    assert_eq!(
        place(&note, &text),
        Place::At {
            start: 10,
            end: 209
        }
    );
}

#[test]
fn a_store_that_cannot_be_read_is_never_written_over() {
    let dir = TempDir::new("notes-unreadable");
    let (notes, state) = store(&dir);
    let path = state.native_chat_dir().join("notes.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, b"{ not json").unwrap();
    assert!(notes.of_folder("w1").is_err());
    assert!(notes.add("w1", FILE, new("a.rs", 1, 1, "x")).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"{ not json");
}
