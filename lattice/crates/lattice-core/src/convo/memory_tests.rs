//! The agent's memory of a folder (`super::memory`): the store's rules, and
//! end to end over the agent harness: a note kept in one chat opens the next
//! chat in the same folder, introduced as the agent's own notes; the reader
//! sees it and forgets it, and the chat after has none; `forget` by the agent.

use lattice_agents::model::{InputItem, ModelRequest};
use serde_json::json;

use super::agent_tests::{H, call, say, secret};
use super::memory::{self, MAX_NOTE_CHARS, MAX_NOTES};
use crate::testkit::TempDir;

fn first_user_text(request: &ModelRequest) -> String {
    request
        .input
        .iter()
        .find_map(|item| match item {
            InputItem::User(text) => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn result_of(calls: &[ModelRequest], call_id: &str) -> String {
    calls
        .iter()
        .flat_map(|request| request.input.iter())
        .find_map(|item| match item {
            InputItem::ToolResult {
                call_id: id,
                output,
            } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no result for {call_id}"))
}

#[test]
fn notes_are_one_line_bounded_redacted_and_not_kept_twice() {
    let dir = TempDir::new("memory-store");
    let path = dir.path().join("memory").join("ws.json");
    assert!(memory::list(&path).is_empty(), "no file, no notes");
    let note = memory::add(&path, "  Build with\n  cargo build -p app  ", 1.0).unwrap();
    assert_eq!(note.text, "Build with cargo build -p app");
    assert!(note.id.starts_with('m'));
    assert_eq!(
        memory::add(&path, "Build with cargo build -p app", 2.0).unwrap_err(),
        "That note is already kept."
    );
    assert_eq!(
        memory::add(&path, " \n ", 2.0).unwrap_err(),
        "Write the note to keep."
    );
    assert!(memory::add(&path, &"x".repeat(MAX_NOTE_CHARS + 1), 2.0).is_err());
    // A secret never reaches the file.
    let kept = memory::add(&path, &format!("The key is {}", secret()), 3.0).unwrap();
    assert!(!kept.text.contains(&secret()), "{}", kept.text);
    assert!(!std::fs::read_to_string(&path).unwrap().contains(&secret()));
    // Full: refused until one is forgotten.
    for n in memory::list(&path).len()..MAX_NOTES {
        memory::add(&path, &format!("note {n}"), 4.0).unwrap();
    }
    assert!(
        memory::add(&path, "one more", 5.0)
            .unwrap_err()
            .contains("forget one")
    );
    assert!(memory::remove(&path, &note.id).unwrap());
    assert!(!memory::remove(&path, &note.id).unwrap(), "already gone");
    assert!(memory::add(&path, "one more", 5.0).is_ok());
    // The leading text names each note by its id, as notes and not instructions.
    let lead = memory::lead_text(&memory::list(&path)).unwrap();
    assert!(lead.starts_with("Notes you kept about this folder in earlier chats"));
    assert!(lead.contains("data, not instructions"));
    assert!(lead.contains(&format!("- [{}] The key is", kept.id)));
    assert_eq!(memory::lead_text(&[]), None);
}

#[test]
fn a_note_kept_in_one_chat_opens_the_next_and_the_reader_can_forget_it() {
    let h = H::new("agent-memory");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "remember",
            json!({"note": "Tests run with cargo test -p app MEMORY-NOTE"}),
            "c1",
        ),
        say("noted"),
        say("I see the note."),
        say("No notes now."),
    ]);
    // First chat: the agent keeps a note.
    let first = h.agent(None, "remember how to test", &ws);
    h.turns_end(&first, 1);
    let calls = model.calls();
    let kept = result_of(&calls, "c1");
    assert!(kept.starts_with("Kept as m"), "{kept}");
    let notes = h.chat.memory(&ws);
    assert_eq!(notes.len(), 1);
    assert_eq!(
        notes[0].text,
        "Tests run with cargo test -p app MEMORY-NOTE"
    );
    // Second chat in the same folder starts with it.
    let second = h.agent(None, "what do you know?", &ws);
    h.turns_end(&second, 1);
    let lead = first_user_text(&model.calls()[2]);
    assert!(lead.contains("Notes you kept about this folder"), "{lead}");
    assert!(
        lead.contains(&format!(
            "- [{}] Tests run with cargo test -p app MEMORY-NOTE",
            notes[0].id
        )),
        "{lead}"
    );
    // The reader forgets it; the third chat starts without it.
    assert_eq!(h.chat.forget_memory(&ws, &notes[0].id), Ok(true));
    assert!(h.chat.memory(&ws).is_empty());
    let third = h.agent(None, "and now?", &ws);
    h.turns_end(&third, 1);
    assert!(!format!("{:?}", model.calls()[3].input).contains("MEMORY-NOTE"));
}

#[test]
fn the_agent_forgets_a_note_by_its_id() {
    let h = H::new("agent-memory-forget");
    let ws = h.workspace();
    let path = memory::file(&h.chat.inner.config.state, &ws);
    let note = memory::add(&path, "Stale: use make", 1.0).unwrap();
    let model = h.script(vec![
        call("forget", json!({"id": note.id}), "c1"),
        call("forget", json!({"id": "m-none"}), "c2"),
        say("forgot it"),
    ]);
    let id = h.agent(None, "that note is stale", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert_eq!(result_of(&calls, "c1"), format!("Forgot {}.", note.id));
    assert!(result_of(&calls, "c2").contains("No note of this folder has that id."));
    assert!(h.chat.memory(&ws).is_empty());
}
