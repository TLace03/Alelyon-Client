//! A lab agent's turn watched as a command is (`super::lab_turns`), over the
//! agent harness's git folder: the files a turn changed on disk are one
//! effect, listed in the Changes panel and named in a note; Undo stages
//! putting one back and Keep of that writes it; a conversation with no folder
//! is not watched; a folder whose command slot is taken says why.

use lattice_protocol::conversation::{
    AgentChatService, ChangeKind, ConversationEvent, ConversationEventKind, ReviewOp,
};

use super::agent_tests::{H, say};
use super::lab_turns::{self, call_id, note};

fn effect_paths(events: &[ConversationEvent], call: &str) -> Option<Vec<String>> {
    events.iter().find_map(|event| match &event.kind {
        ConversationEventKind::CommandEffect { call_id, files, .. } if call_id == call => {
            Some(files.iter().map(|f| f.path.clone()).collect())
        }
        _ => None,
    })
}

fn notices(events: &[ConversationEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::Notice { text } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn what_an_agents_turn_changed_is_listed_and_undoable() {
    let h = H::new("lab-turn");
    let ws = h.workspace();
    h.script(vec![say("ok")]);
    let id = h.agent(None, "hello", &ws);
    h.turns_end(&id, 1);
    let convo = h.chat.inner.convo(&id);

    let watch = lab_turns::begin(&h.chat.inner, &convo, "t9", "Claude Code")
        .expect("a folder")
        .expect("watched");
    // While the turn runs, no Keep or command lands in the folder.
    assert!(h.chat.inner.slots.running(&ws));
    // The agent changes a file and makes one, with its own tools.
    std::fs::write(h.folder.join("a.txt"), "agent wrote\n").unwrap();
    std::fs::write(h.folder.join("new.txt"), "made\n").unwrap();
    lab_turns::end(&h.chat.inner, &convo, watch);
    assert!(!h.chat.inner.slots.running(&ws), "the folder given back");

    let events = h.events_until(&id, |events| effect_paths(events, &call_id("t9")).is_some());
    assert_eq!(
        effect_paths(&events, &call_id("t9")).unwrap(),
        ["a.txt", "new.txt"]
    );
    assert!(
        notices(&events).contains(&note("Claude Code", &["a.txt".into(), "new.txt".into()])),
        "{:?}",
        notices(&events)
    );
    // In the Changes panel, as changes already on disk.
    let changes = h.runtime.block_on(h.chat.changes(&id)).unwrap().changes;
    let a = changes
        .iter()
        .find(|c| c.path == "a.txt")
        .expect("a.txt listed")
        .id
        .clone();
    assert!(changes.iter().any(|c| c.path == "new.txt"));
    // Undo stages putting it back; Keep of that writes it.
    h.runtime
        .block_on(h.chat.review(
            &id,
            vec![ReviewOp::Undo {
                change: a,
                hunks: None,
                note: None,
            }],
        ))
        .unwrap();
    let back = h
        .runtime
        .block_on(h.chat.changes(&id))
        .unwrap()
        .changes
        .into_iter()
        .find(|c| c.path == "a.txt" && c.kind == ChangeKind::CommandUndo)
        .expect("the undo staged")
        .id;
    assert_eq!(
        std::fs::read_to_string(h.folder.join("a.txt")).unwrap(),
        "agent wrote\n"
    );
    h.runtime
        .block_on(h.chat.review(
            &id,
            vec![ReviewOp::Keep {
                change: back,
                hunks: None,
            }],
        ))
        .unwrap();
    assert_eq!(std::fs::read(h.folder.join("a.txt")).unwrap(), b"a\n");
}

#[test]
fn no_folder_is_not_watched_and_a_busy_folder_says_why() {
    let h = H::new("lab-turn-none");
    h.script(vec![say("ok"), say("ok")]);
    let id = match h
        .send(
            None,
            "hello",
            "local",
            super::agent_tests::local(),
            lattice_protocol::conversation::Mode::Ask,
            None,
        )
        .unwrap()
    {
        lattice_protocol::conversation::Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    };
    h.turns_end(&id, 1);
    let convo = h.chat.inner.convo(&id);
    assert!(lab_turns::begin(&h.chat.inner, &convo, "t1", "Codex").is_none());

    let ws = h.workspace();
    let id = h.agent(None, "hello", &ws);
    h.turns_end(&id, 1);
    let convo = h.chat.inner.convo(&id);
    let _command = h.chat.inner.slots.claim(&ws, "c1").unwrap();
    let refused = lab_turns::begin(&h.chat.inner, &convo, "t2", "Codex")
        .expect("a folder")
        .err()
        .expect("not watched");
    assert!(
        refused.contains("A command is running in this folder"),
        "{refused}"
    );
    assert_eq!(note("Codex", &[]), "Codex changed no file in this folder.");
}
