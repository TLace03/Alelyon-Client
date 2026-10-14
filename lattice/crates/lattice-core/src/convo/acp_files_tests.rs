//! A lab agent's client reads and writes through its conversation
//! (`super::acp_files`), over the agent harness's git folder: a write is a
//! staged change in the conversation's review, recorded under the running
//! turn and the agent's call, listed in the turn, and on disk only once kept;
//! a read sees the disk, then the staged text; a path outside the folder (or
//! in `.git`) is refused and nothing is written; a conversation not open, or
//! in Ask mode, refuses.

use std::sync::{Arc, OnceLock};

use lattice_protocol::conversation::{
    AgentChatService, ChangeKind, ConversationEventKind, Mode, ReviewOp,
};

use super::acp_files::{ConvoFiles, NOT_OPEN};
use super::agent::Running;
use super::agent_tests::{H, say};
use super::item::Item;
use crate::acp::pool::ChatFiles;

fn files(h: &H) -> ConvoFiles {
    let made = Arc::new(OnceLock::new());
    let _ = made.set(Arc::downgrade(&h.chat.inner));
    ConvoFiles { inner: made }
}

/// A lab agent's turn `turn` runs in `id` (as `convo::agent`'s plain turn
/// marks it); `None` ends it.
fn running(h: &H, id: &str, turn: Option<&str>) {
    h.chat.inner.convo(id).state().running = turn.map(|turn| Running {
        turn: turn.to_owned(),
        control: None,
        job: None,
        affirmatively_local: true,
        label: "agent".to_owned(),
        notes: Arc::default(),
        stop_requested: false,
    });
}

fn staged_events(h: &H, id: &str, n: usize) -> Vec<String> {
    h.events_until(id, |events| {
        events
            .iter()
            .filter(|e| matches!(e.kind, ConversationEventKind::Staged { .. }))
            .count()
            >= n
    })
    .into_iter()
    .filter_map(|e| match e.kind {
        ConversationEventKind::Staged { path, .. } => Some(path),
        _ => None,
    })
    .collect()
}

/// Mutant: `agent_write` writing the file instead of staging it.
#[test]
fn a_client_write_is_a_staged_change_until_the_reader_keeps_it() {
    let h = H::new("acp-files-write");
    let ws = h.workspace();
    h.script(vec![say("ok")]);
    let id = h.agent(None, "hello", &ws);
    h.turns_end(&id, 1);
    let files = files(&h);

    // A read sees the file on disk.
    assert_eq!(files.read(&id, "a.txt").unwrap(), "a\n");

    running(&h, &id, Some("t7"));
    let told = files
        .write(&id, "a.txt", "agent wrote\n", "agent_toolu_1")
        .unwrap();
    assert!(
        told.contains("not on disk until the user keeps it"),
        "{told}"
    );
    files
        .write(&id, "new.txt", "made\n", "agent_write_2")
        .unwrap();
    // Nothing reached the disk.
    assert_eq!(std::fs::read(h.folder.join("a.txt")).unwrap(), b"a\n");
    assert!(!h.folder.join("new.txt").exists());
    // A read now sees the staged text.
    assert_eq!(files.read(&id, "a.txt").unwrap(), "agent wrote\n");
    assert_eq!(files.read(&id, "new.txt").unwrap(), "made\n");

    // Recorded under the running turn and the agent's call, and listed.
    let items = h.sidecar_items(&id);
    let recorded = |path: &str, call: &str| {
        items.iter().any(|item| {
            matches!(item, Item::Staged { turn, call_id, change }
                if turn == "t7" && call_id == call && change.path == path)
        })
    };
    assert!(recorded("a.txt", "agent_toolu_1"));
    assert!(recorded("new.txt", "agent_write_2"));
    assert_eq!(staged_events(&h, &id, 2), ["a.txt", "new.txt"]);

    // In the Changes panel; the reader's Keep is the write.
    running(&h, &id, None);
    let changes = h.runtime.block_on(h.chat.changes(&id)).unwrap().changes;
    let a = changes
        .iter()
        .find(|c| c.path == "a.txt")
        .expect("a.txt staged");
    assert_eq!(a.kind, ChangeKind::Overwrite);
    let made = changes
        .iter()
        .find(|c| c.path == "new.txt")
        .expect("new.txt staged");
    assert_eq!(made.kind, ChangeKind::Create);
    h.runtime
        .block_on(h.chat.review(
            &id,
            vec![ReviewOp::Keep {
                change: a.id.clone(),
                hunks: None,
            }],
        ))
        .unwrap();
    assert_eq!(
        std::fs::read(h.folder.join("a.txt")).unwrap(),
        b"agent wrote\n"
    );
    assert!(!h.folder.join("new.txt").exists(), "only what was kept");
}

/// Mutant: the path rules skipped for a client write.
#[test]
fn a_path_outside_the_folder_is_refused_and_nothing_is_written() {
    let h = H::new("acp-files-outside");
    let ws = h.workspace();
    h.script(vec![say("ok")]);
    let id = h.agent(None, "hello", &ws);
    h.turns_end(&id, 1);
    let files = files(&h);
    running(&h, &id, Some("t1"));

    let beside = h.folder.parent().unwrap().join("outside.txt");
    for path in ["../outside.txt", ".git/config", "a/.././../outside.txt"] {
        assert!(files.write(&id, path, "x\n", "c1").is_err(), "{path}");
        assert!(files.read(&id, path).is_err(), "{path}");
    }
    assert!(!beside.exists());
    let changes = h.runtime.block_on(h.chat.changes(&id)).unwrap().changes;
    assert!(changes.is_empty(), "{changes:?}");

    // A conversation not open here refuses.
    assert_eq!(
        files.write("no-such-chat", "a.txt", "x", "c1").unwrap_err(),
        NOT_OPEN
    );
    assert_eq!(files.read("no-such-chat", "a.txt").unwrap_err(), NOT_OPEN);
}

/// The trust rules stand: in Ask mode a client write is refused, as the
/// built-in agent's staging is.
#[test]
fn ask_mode_refuses_a_client_write() {
    let h = H::new("acp-files-ask");
    let ws = h.workspace();
    h.script(vec![say("ok")]);
    let id = h.agent(None, "hello", &ws);
    h.turns_end(&id, 1);
    h.chat.inner.convo(&id).state().mode = Mode::Ask;
    let files = files(&h);
    assert!(files.write(&id, "a.txt", "x\n", "c1").is_err());
    assert_eq!(std::fs::read(h.folder.join("a.txt")).unwrap(), b"a\n");
    assert!(
        h.runtime
            .block_on(h.chat.changes(&id))
            .unwrap()
            .changes
            .is_empty()
    );
}
