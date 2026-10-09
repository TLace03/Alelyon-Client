//! Background commands (`super::background`), end to end over the agent
//! harness with a real PowerShell command: approved as any command, it keeps
//! running after its turn ends; `command_output` reads what it wrote and that
//! it runs; `stop_command` ends it and its exit reaches the log; the reader's
//! Stop ends one; ids of another conversation are not reachable; and the
//! folder's slots keep foreground and background apart.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::{
    AgentChatService, ConversationEvent, ConversationEventKind, Decision, ExitReason,
};
use serde_json::json;

use super::agent_tests::{H, call, say};
use crate::exec::run::{CommandSlots, MAX_BACKGROUND};

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

fn exited(events: &[ConversationEvent], call: &str) -> Option<ExitReason> {
    events.iter().find_map(|event| match &event.kind {
        ConversationEventKind::CommandExited {
            call_id, reason, ..
        } if call_id == call => Some(*reason),
        _ => None,
    })
}

/// A command that writes a marker, then runs until it is ended.
const SERVER: &str = "Write-Output BG-READY; while ($true) { Start-Sleep -Milliseconds 200 }";

fn wrote_ready(events: &[ConversationEvent], call: &str) -> bool {
    events.iter().any(|event| {
        matches!(&event.kind, ConversationEventKind::CommandProgress { call_id, tail_preview, .. }
            if call_id == call && tail_preview.contains("BG-READY"))
    })
}

#[test]
fn a_background_command_outlives_its_turn_and_is_read_and_stopped_by_the_agent() {
    let h = H::new("agent-background");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "run_command",
            json!({"command": SERVER, "background": true}),
            "c1",
        ),
        say("started it"),
        call("command_output", json!({"id": "c1"}), "c2"),
        call("command_output", json!({"id": "c1"}), "c3"),
        call("stop_command", json!({"id": "c1"}), "c4"),
        say("stopped it"),
    ]);
    let id = h.agent(None, "start the server", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
    });
    h.runtime
        .block_on(h.chat.decide(&id, "c1", Decision::Approve))
        .unwrap();
    // The dialog said it keeps running.
    assert!(h.confirm.asked().iter().any(|request| matches!(
        request,
        crate::ports::ConfirmRequest::RunCommand {
            background: true,
            ..
        }
    )));
    // The turn ends while the command runs.
    let events = h.turns_end(&id, 1);
    assert!(
        result_of(&model.calls(), "c1").starts_with("Started in the background with id c1."),
        "{}",
        result_of(&model.calls(), "c1")
    );
    assert_eq!(exited(&events, "c1"), None, "it still runs after its turn");
    assert_eq!(h.chat.inner.background.running(&id), ["c1".to_string()]);
    h.events_until(&id, |events| wrote_ready(events, "c1"));

    // The next turn reads it, reads again (nothing new), and stops it.
    h.agent(Some(&id), "how is the server?", &ws);
    let events = h.turns_end(&id, 2);
    let calls = model.calls();
    let first = result_of(&calls, "c2");
    assert!(first.starts_with("It is still running."), "{first}");
    assert!(first.contains("BG-READY"), "{first}");
    let second = result_of(&calls, "c3");
    assert!(!second.contains("BG-READY"), "only what is new: {second}");
    assert_eq!(
        result_of(&calls, "c4"),
        "Stopped: its whole process tree ends now."
    );
    let events = if exited(&events, "c1").is_some() {
        events
    } else {
        h.events_until(&id, |events| exited(events, "c1").is_some())
    };
    assert_eq!(exited(&events, "c1"), Some(ExitReason::Stopped));
    assert!(h.chat.inner.background.running(&id).is_empty());
    // Once ended, it says how.
    let after = h.chat.inner.background.output(&id, "c1").unwrap();
    assert!(
        after.starts_with("It has ended. Stopped by the user"),
        "{after}"
    );
}

#[test]
fn the_readers_stop_ends_a_background_command_and_other_conversations_cannot_reach_it() {
    let h = H::new("agent-background-stop");
    let ws = h.workspace();
    h.script(vec![
        call(
            "run_command",
            json!({"command": SERVER, "background": true}),
            "c1",
        ),
        say("started it"),
    ]);
    let id = h.agent(None, "start the server", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
    });
    h.runtime
        .block_on(h.chat.decide(&id, "c1", Decision::Approve))
        .unwrap();
    h.turns_end(&id, 1);
    // Another conversation's id reaches nothing.
    assert!(!h.chat.stop_command("another-conversation", "c1"));
    assert!(
        h.chat
            .inner
            .background
            .output("another-conversation", "c1")
            .is_err()
    );
    // The reader's Stop on its card.
    assert!(h.chat.stop_command(&id, "c1"));
    let events = h.events_until(&id, |events| exited(events, "c1").is_some());
    assert_eq!(exited(&events, "c1"), Some(ExitReason::Stopped));
    assert!(!h.chat.stop_command(&id, "c1"), "it has ended");
}

#[test]
fn the_folders_slots_keep_foreground_and_background_apart() {
    let slots = CommandSlots::default();
    let fg = slots.claim("ws", "f1").expect("the foreground slot");
    let mut held = Vec::new();
    for n in 0..MAX_BACKGROUND {
        held.push(
            slots
                .claim_background("ws", &format!("b{n}"))
                .expect("a background slot"),
        );
    }
    // Full: a fourth background command waits for one to end; another folder is apart.
    assert!(slots.claim_background("ws", "b9").is_none());
    assert!(slots.claim_background("other", "b9").is_some());
    assert_eq!(slots.background("ws"), MAX_BACKGROUND);
    // Background commands do not hold the folder's one command (nor Keep).
    drop(fg);
    assert!(!slots.running("ws"));
    assert!(slots.busy("ws"));
    assert!(slots.claim("ws", "f2").is_some());
    held.clear();
    assert_eq!(slots.background("ws"), 0);
    assert!(!slots.busy("ws"));
}
