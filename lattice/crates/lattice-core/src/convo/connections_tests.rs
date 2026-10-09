//! Agent turns without a folder (Connections, 2026-10-08): Agent
//! mode with no folder is an agent turn when the agent's browser is on or an
//! MCP server of the reader's own is enabled. It offers `ask_question`, the
//! browser and those servers' tools only, never a file or command tool,
//! under its own instructions; otherwise it is the plain turn it always was.

#![cfg(windows)]

use lattice_agents::model::ModelRequest;
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ConversationEventKind, Decision, Mode, TurnKind, TurnStatus,
};
use serde_json::json;

use super::agent_tests::{H, call, local, say, statuses};
use super::browser_tests::{browser_harness, opened_the_page};
use super::prompt_agent::{BROWSER_NOTE, MCP_NOTE, NO_FOLDER_SYSTEM};

fn names(request: &ModelRequest) -> Vec<String> {
    request.tools.iter().map(|tool| tool.name.clone()).collect()
}

/// An Agent-mode send with no folder; its conversation and turn kind.
fn send(h: &H, conversation: Option<&str>) -> (String, TurnKind) {
    match h
        .send(conversation, "Do it.", "local", local(), Mode::Agent, None)
        .unwrap()
    {
        Accepted::Started {
            conversation,
            turn_kind,
            ..
        } => (conversation.id, turn_kind),
        other => panic!("{other:?}"),
    }
}

#[test]
fn agent_mode_without_a_folder_is_a_plain_turn_until_the_browser_is_on() {
    let h = H::new("nofolder-plain");
    h.script(vec![say("plain")]);
    let (id, kind) = send(&h, None);
    assert_eq!(kind, TurnKind::Plain, "nothing to use without a folder");
    h.turns_end(&id, 1);
    crate::browser::prefs::set(&h.state, true).unwrap();
    let model = h.script(vec![say("ready")]);
    let (id, kind) = send(&h, Some(&id));
    assert_eq!(kind, TurnKind::Agent);
    let events = h.turns_end(&id, 2);
    assert_eq!(statuses(&events).last(), Some(&TurnStatus::Completed));
    let first = &model.calls()[0];
    let offered = names(first);
    assert!(offered.contains(&"ask_question".to_owned()));
    assert!(offered.contains(&"browser_open".to_owned()));
    for folder_tool in [
        "list_dir",
        "glob",
        "read_file",
        "grep",
        "edit_file",
        "write_file",
        "delete_file",
        "run_command",
    ] {
        assert!(!offered.contains(&folder_tool.to_owned()), "{folder_tool}");
    }
    assert!(first.system.starts_with(NO_FOLDER_SYSTEM));
    assert!(first.system.ends_with(BROWSER_NOTE));
    // Ask mode without a folder stays plain.
    h.script(vec![say("plain again")]);
    match h
        .send(Some(&id), "Look.", "local", local(), Mode::Ask, None)
        .unwrap()
    {
        Accepted::Started { turn_kind, .. } => assert_eq!(turn_kind, TurnKind::Plain),
        other => panic!("{other:?}"),
    }
}

/// An enabled MCP server of the reader's own makes a turn without a folder an
/// agent turn, with that server's tools (each call still asks).
#[test]
fn an_enabled_server_of_the_readers_own_gives_a_turn_without_a_folder_its_tools() {
    let h = H::new("nofolder-mcp");
    let stub = crate::mcp::hub_tests::stub();
    crate::mcp::config::put_user_server(
        &h.state,
        "stub",
        &json!({"command": stub.display().to_string(), "args": []}),
    )
    .unwrap();
    assert!(
        h.runtime
            .block_on(
                h.chat
                    .mcp_enable(crate::mcp::config::ServerKey::user("stub"), None)
            )
            .unwrap()
    );
    let model = h.script(vec![say("I can use echo.")]);
    let (id, kind) = send(&h, None);
    assert_eq!(kind, TurnKind::Agent);
    h.turns_end(&id, 1);
    let first = &model.calls()[0];
    assert!(names(first).contains(&"mcp__stub__echo".to_owned()));
    assert!(!names(first).contains(&"read_file".to_owned()));
    assert!(first.system.starts_with(NO_FOLDER_SYSTEM));
    assert!(first.system.ends_with(MCP_NOTE));
}

/// The agent's browser in a turn without a folder, on a real headless browser:
/// it opens a page, and a click stated as share asks on the card and in the
/// dialog, as in a folder.
#[test]
fn the_browser_works_without_a_folder_and_a_share_still_asks() {
    let _one = crate::testkit::one_real_browser();
    let Some(h) = browser_harness("nofolder-browser") else {
        return;
    };
    crate::browser::prefs::set(&h.state, true).unwrap();
    let url = format!("http://127.0.0.1:{}/", crate::browser::tests::serve());
    h.script(vec![
        call("browser_open", json!({"url": url}), "b1"),
        call(
            "browser_click",
            json!({"x": 450, "y": 125, "what": "the Post button", "effect": "share"}),
            "b2",
        ),
        say("Posted."),
    ]);
    let (id, kind) = send(&h, None);
    assert_eq!(kind, TurnKind::Agent);
    let events = h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
    });
    opened_the_page(&events);
    h.runtime
        .block_on(h.chat.decide(&id, "b2", Decision::Approve))
        .unwrap();
    let events = h.turns_end(&id, 1);
    assert_eq!(statuses(&events), [TurnStatus::Completed]);
    let outputs: Vec<String> = events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::ToolOutput { preview, .. } => Some(preview.clone()),
            _ => None,
        })
        .collect();
    assert!(outputs[1].starts_with("The page is posted"), "{outputs:?}");
}
