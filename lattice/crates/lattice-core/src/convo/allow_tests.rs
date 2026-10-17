//! Allow all (`super::allow`), end to end over the agent harness: in a
//! project with the switch on, a command runs with no approval and no dialog,
//! a written file is kept on disk as it is staged, a question is left to the
//! agent, a stack decision takes the recommended option; every one is in the
//! log of what was done without the reader. With the switch off, the same
//! command waits for the reader.

use lattice_agents::model::InputItem;
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ConversationEventKind, Mode, SendRequest,
};
use serde_json::json;

use super::agent_tests::{H, call, local, say};
use super::allow::{Kind, QUESTION_UNASKED};
use crate::projects::Change;

fn in_project(h: &H, ws: &str, project: &str, text: &str) -> String {
    match h
        .runtime
        .block_on(h.chat.send(SendRequest {
            conversation: None,
            text: text.into(),
            choice: "local".into(),
            shown: local(),
            mode: Mode::Agent,
            workspace: Some(ws.to_owned()),
            edit_of: None,
            project: Some(project.to_owned()),
            images: Vec::new(),
        }))
        .unwrap()
    {
        Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    }
}

fn project(h: &H, allow_all: bool) -> String {
    let made = h.runtime.block_on(h.chat.project_create("Shop".into())).unwrap();
    h.runtime
        .block_on(h.chat.project_change(made.id.clone(), Change { allow_all: Some(allow_all), ..Change::default() }))
        .unwrap();
    made.id
}

/// Mutants: a command under allow all still asking; a staged change left
/// waiting; a question still put to the reader; nothing logged.
#[test]
fn allow_all_runs_keeps_decides_and_logs_everything_it_did_without_the_reader() {
    let h = H::new("allow-all");
    let ws = h.workspace();
    let shop = project(&h, true);
    let model = h.script(vec![
        call("run_command", json!({"command": "Write-Output ALLOWED", "timeout_s": 300}), "c1"),
        call("write_file", json!({"path": "notes.md", "content": "kept\n"}), "w1"),
        call("ask_question", json!({"question": "Tabs or spaces?", "options": ["tabs", "spaces"]}), "q1"),
        call("recommend_stack", json!({"area": "database", "needs": ["embedded", "single-user"]}), "r1"),
        say("Done without asking."),
    ]);
    let id = in_project(&h, &ws, &shop, "Set the shop up.");
    let events = h.turns_end(&id, 1);
    // No approval was asked and no dialog opened; the command ran.
    assert!(!events.iter().any(|e| matches!(e.kind, ConversationEventKind::ApprovalRequested { .. })));
    assert!(!h.confirm.asked().iter().any(|r| matches!(r, crate::ports::ConfirmRequest::RunCommand { .. })));
    assert!(events.iter().any(|e| matches!(e.kind, ConversationEventKind::CommandExited { code: Some(0), .. })));
    // The file is on disk, kept as it was staged.
    assert_eq!(std::fs::read_to_string(h.folder.join("notes.md")).unwrap().replace("\r\n", "\n"), "kept\n");
    let results: Vec<(String, String)> = model
        .calls()
        .iter()
        .flat_map(|r| r.input.clone())
        .filter_map(|i| match i {
            InputItem::ToolResult { call_id, output } => Some((call_id, output)),
            _ => None,
        })
        .collect();
    let said = |id: &str| results.iter().find(|(c, _)| c == id).map(|(_, o)| o.clone()).unwrap_or_default();
    assert!(said("c1").contains("ALLOWED"), "{}", said("c1"));
    assert!(said("w1").contains("it is kept"), "{}", said("w1"));
    assert_eq!(said("q1"), QUESTION_UNASKED);
    assert!(said("r1").contains("recommended option was taken") && said("r1").contains("SQLite"), "{}", said("r1"));
    assert!(events.iter().any(|e| matches!(&e.kind, ConversationEventKind::ChoiceMade { option, by_agent: true, .. } if option == "sqlite")));
    // Everything is in the log, newest first, for the project.
    let log = h.chat.auto_decisions(Some(&shop));
    let kinds: Vec<Kind> = log.iter().rev().map(|e| e.kind).collect();
    assert_eq!(kinds, [Kind::Command, Kind::Keep, Kind::Question, Kind::Choice]);
    assert!(log.iter().all(|e| e.conversation == id && e.project == shop));
    assert!(log.iter().any(|e| e.what.contains("Write-Output ALLOWED")) && log.iter().any(|e| e.what == "notes.md"));
    assert!(h.chat.auto_decisions(Some("another")).is_empty());
}

#[test]
fn with_the_switch_off_a_command_waits_for_the_reader() {
    let h = H::new("allow-all-off");
    let ws = h.workspace();
    let shop = project(&h, false);
    h.script(vec![call("run_command", json!({"command": "Write-Output ASKED", "timeout_s": 300}), "c1"), say("ok")]);
    let id = in_project(&h, &ws, &shop, "Run it.");
    h.events_until(&id, |events| events.iter().any(|e| matches!(e.kind, ConversationEventKind::ApprovalRequested { .. })));
    assert!(h.chat.auto_decisions(None).is_empty());
    h.chat.stop(&id);
}

/// Spending money still asks: under allow all, a command the list of billed
/// commands names waits for the reader, and its card says why it may cost.
#[test]
fn a_command_that_may_cost_money_still_asks_under_allow_all() {
    let h = H::new("allow-all-cost");
    let ws = h.workspace();
    let shop = project(&h, true);
    h.script(vec![call("run_command", json!({"command": "terraform apply -auto-approve", "timeout_s": 300}), "c1"), say("ok")]);
    let id = in_project(&h, &ws, &shop, "Create the servers.");
    let events = h.events_until(&id, |events| events.iter().any(|e| matches!(e.kind, ConversationEventKind::ApprovalRequested { .. })));
    let cost = events.iter().find_map(|e| match &e.kind {
        ConversationEventKind::ApprovalRequested { detail: lattice_protocol::conversation::ApprovalDetail::Command { cost, .. }, .. } => cost.clone(),
        _ => None,
    });
    assert!(cost.as_deref().is_some_and(|c| c.starts_with("May cost money:") && c.contains("Terraform")), "{cost:?}");
    assert!(h.chat.auto_decisions(Some(&shop)).is_empty(), "nothing was done without the reader");
    h.chat.stop(&id);
}
