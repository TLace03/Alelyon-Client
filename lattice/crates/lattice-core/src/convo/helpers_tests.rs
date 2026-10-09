//! Subagents (`super::helpers`), end to end over the agent harness with one
//! scripted model answering the agent and its helper in turn: the helper gets
//! the task as its whole input, its own instructions and only the read tools,
//! reads the folder, and only its answer comes back as the call's result; the
//! agent's conversation holds none of the helper's steps; an empty task is
//! refused; and Ask mode offers it too (it changes nothing).

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::{AgentChatService, Mode};
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::helpers::{EDITING_HELPER_SYSTEM, HELPER_SYSTEM, staged_line};
use super::prompt_agent;

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

fn texts(request: &ModelRequest) -> Vec<String> {
    request
        .input
        .iter()
        .filter_map(|item| match item {
            InputItem::User(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_helper_reads_the_folder_on_its_own_and_only_its_answer_comes_back() {
    let h = H::new("agent-helper");
    let ws = h.workspace();
    let model = h.script(vec![
        // The agent sends a helper.
        call(
            "spawn_agent",
            json!({"task": "What letter does a.txt hold? HELPER-TASK"}),
            "c1",
        ),
        // The helper reads, then answers.
        call("read_file", json!({"path": "a.txt"}), "h1"),
        say("a.txt holds the letter a."),
        // The agent answers with what came back.
        say("The helper says a."),
    ]);
    let id = h.agent(None, "ask a helper", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert_eq!(calls.len(), 4, "agent, helper, helper, agent");
    // The helper's own run: its instructions, its task as its whole input,
    // the read tools only.
    let helper = &calls[1];
    assert_eq!(helper.system, HELPER_SYSTEM);
    assert_eq!(texts(helper), ["What letter does a.txt hold? HELPER-TASK"]);
    let mut tools: Vec<&str> = helper.tools.iter().map(|t| t.name.as_str()).collect();
    tools.sort_unstable();
    assert_eq!(tools, ["glob", "grep", "list_dir", "read_file"]);
    // It read the folder.
    assert!(
        result_of(&calls[2..3], "h1").contains("1\ta"),
        "{:?}",
        calls[2].input
    );
    // Only its answer comes back to the agent.
    assert_eq!(result_of(&calls[3..], "c1"), "a.txt holds the letter a.");
    assert!(
        !format!("{:?}", calls[3].input).contains("h1"),
        "the helper's steps stay out of the agent's run"
    );
    // The agent's record holds the helper's answer, not its steps.
    let record = format!("{:?}", h.sidecar_items(&id));
    assert!(record.contains("a.txt holds the letter a."));
    assert!(!record.contains("\"h1\""), "{record}");
}

#[test]
fn an_empty_task_is_refused_and_ask_mode_offers_a_helper() {
    let h = H::new("agent-helper-empty");
    let ws = h.workspace();
    let model = h.script(vec![
        call("spawn_agent", json!({"task": "   "}), "c1"),
        say("I need a task."),
    ]);
    let id = h.agent(None, "ask a helper", &ws);
    h.turns_end(&id, 1);
    assert!(
        result_of(&model.calls(), "c1").contains("Give the helper its task."),
        "{}",
        result_of(&model.calls(), "c1")
    );
    for mode in [Mode::Ask, Mode::Agent] {
        assert!(
            prompt_agent::tools(mode)
                .iter()
                .any(|t| t.name == "spawn_agent"),
            "{mode:?}"
        );
    }
    assert!(
        !prompt_agent::tools_without_folder()
            .iter()
            .any(|t| t.name == "spawn_agent"),
        "no folder, no helper"
    );
}

/// An editing helper (`edits: true`, Agent mode): its own instructions, the
/// read and staging tools, its change staged in the conversation's Changes
/// panel (not on disk), its step under a call id of its own, and its answer
/// ending with the file it staged.
#[test]
fn an_editing_helper_stages_its_change_for_review_and_names_it() {
    let h = H::new("agent-helper-edits");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "spawn_agent",
            json!({"task": "Change the letter in a.txt to b.", "edits": true}),
            "c1",
        ),
        call("read_file", json!({"path": "a.txt"}), "call_1"),
        call(
            "edit_file",
            json!({"path": "a.txt", "old_string": "a", "new_string": "b"}),
            "call_2",
        ),
        say("Changed the letter in a.txt from a to b."),
        say("The helper staged the change."),
    ]);
    let id = h.agent(None, "have a helper change a.txt", &ws);
    let events = h.turns_end(&id, 1);
    let calls = model.calls();
    let helper = &calls[1];
    assert_eq!(helper.system, EDITING_HELPER_SYSTEM);
    let mut tools: Vec<&str> = helper.tools.iter().map(|t| t.name.as_str()).collect();
    tools.sort_unstable();
    assert_eq!(
        tools,
        [
            "delete_file",
            "edit_file",
            "glob",
            "grep",
            "list_dir",
            "read_file",
            "write_file"
        ]
    );
    // Its answer, and the file it staged as its own staging call recorded it.
    let answer = result_of(&calls[4..], "c1");
    assert_eq!(
        answer,
        format!(
            "Changed the letter in a.txt from a to b.{}",
            staged_line(&["a.txt".to_string()])
        )
    );
    // The change waits in the Changes panel.
    let changes = h.runtime.block_on(h.chat.changes(&id)).unwrap();
    assert!(
        changes.changes.iter().any(|change| change.path == "a.txt"),
        "{changes:?}"
    );
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        lattice_protocol::conversation::ConversationEventKind::Staged { path, .. } if path == "a.txt"
    )));
    // The helper's step had a call id of its own, not its model's "call_2"
    // (which the agent's own calls may use too).
    let record = format!("{:?}", h.sidecar_items(&id));
    assert!(!record.contains("\"call_2\""), "{record}");
    assert!(
        record.contains("call_id: \"c1/call_2\""),
        "its id, under the helper's call, is recorded: {record}"
    );
}

/// A helper that only reads names no staged file; Ask mode refuses edits.
#[test]
fn a_reading_helper_stages_nothing_and_ask_mode_refuses_edits() {
    assert_eq!(staged_line(&[]), "\n\n[The helper staged no change.]");
    let h = H::new("agent-helper-ask-edits");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "spawn_agent",
            json!({"task": "Change a.txt.", "edits": true}),
            "c1",
        ),
        say("I can only read here."),
    ]);
    let id = match h
        .send(
            None,
            "change a.txt",
            "local",
            super::agent_tests::local(),
            Mode::Ask,
            Some(&ws),
        )
        .unwrap()
    {
        lattice_protocol::conversation::Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    };
    h.turns_end(&id, 1);
    assert!(
        result_of(&model.calls(), "c1").contains("only in Agent mode"),
        "{}",
        result_of(&model.calls(), "c1")
    );
    assert_eq!(model.calls().len(), 2, "no helper ran");
}
