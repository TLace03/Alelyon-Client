//! The agent's to-do list (`super::todos`), end to end over the agent
//! harness: each list is recorded and shown as an event that replaces the one
//! before; an empty list clears it; too many steps, a bad step and two in
//! progress are refused with a sentence and save nothing; a secret is
//! redacted; and a reopened conversation shows the latest list.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::{
    ConversationEvent, ConversationEventKind, TodoItem, TodoStatus,
};
use serde_json::{Value, json};

use super::agent_tests::{H, call, say, secret};
use super::item::Item;
use super::todos;

fn update(items: Value, id: &str) -> lattice_agents::testing::ScriptedStep {
    call("update_todos", json!({ "items": items }), id)
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

fn lists(events: &[ConversationEvent]) -> Vec<Vec<TodoItem>> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::TodosUpdated { items } => Some(items.clone()),
            _ => None,
        })
        .collect()
}

fn step(content: &str, status: TodoStatus) -> TodoItem {
    TodoItem {
        content: content.to_owned(),
        status,
    }
}

/// A list written, then written again as the work moves, then cleared: each
/// is an event and a record, the latest is the list, and the agent hears how
/// many steps are done.
/// Mutants: a list appended to the one before instead of replacing it; an
/// empty list refused.
#[test]
fn the_list_is_written_whole_each_time_and_can_be_cleared() {
    let h = H::new("todos-written");
    let ws = h.workspace();
    let model = h.script(vec![
        update(
            json!([
                {"content": "Read the sort", "status": "in_progress"},
                {"content": "Fix it", "status": "pending"},
            ]),
            "u1",
        ),
        update(
            json!([
                {"content": "Read the sort", "status": "done"},
                {"content": "Fix it", "status": "in_progress"},
            ]),
            "u2",
        ),
        update(json!([]), "u3"),
        say("Done."),
    ]);
    let id = h.agent(None, "Fix the sort.", &ws);
    let events = h.turns_end(&id, 1);
    assert_eq!(
        lists(&events),
        [
            vec![
                step("Read the sort", TodoStatus::InProgress),
                step("Fix it", TodoStatus::Pending),
            ],
            vec![
                step("Read the sort", TodoStatus::Done),
                step("Fix it", TodoStatus::InProgress),
            ],
            vec![],
        ]
    );
    let calls = model.calls();
    assert!(
        result_of(&calls, "u2").starts_with("The to-do list has 2 steps, 1 done;"),
        "{}",
        result_of(&calls, "u2")
    );
    assert_eq!(result_of(&calls, "u3"), "The to-do list is cleared.");
    let items = h.sidecar_items(&id);
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, Item::TodosUpdated { .. }))
            .count(),
        3
    );
    assert!(
        todos::latest(&items).is_empty(),
        "the latest list is the cleared one"
    );
}

/// Too many steps, an empty or two-line or too long step and two steps in
/// progress are refused with their sentences and save nothing; a secret in a
/// step is redacted before it is kept or shown.
/// Mutants: no limit on steps; two steps in progress allowed; the text not
/// redacted.
#[test]
fn bad_lists_are_refused_and_a_secret_is_redacted() {
    let h = H::new("todos-bounds");
    let ws = h.workspace();
    let key = secret();
    let many: Vec<Value> = (0..=todos::MAX_ITEMS)
        .map(|n| json!({"content": format!("Step {n}"), "status": "pending"}))
        .collect();
    let model = h.script(vec![
        update(Value::Array(many), "m1"),
        update(json!([{"content": "  ", "status": "pending"}]), "c1"),
        update(
            json!([{"content": "two\nlines", "status": "pending"}]),
            "c2",
        ),
        update(
            json!([{"content": "x".repeat(todos::MAX_CONTENT_CHARS + 1), "status": "pending"}]),
            "c3",
        ),
        update(
            json!([
                {"content": "One", "status": "in_progress"},
                {"content": "Two", "status": "in_progress"},
            ]),
            "p1",
        ),
        update(json!([{"content": "One", "status": "started"}]), "k1"),
        update(
            json!([{"content": format!("Rotate {key}"), "status": "pending"}]),
            "s1",
        ),
        say("Done."),
    ]);
    let id = h.agent(None, "Plan it.", &ws);
    let events = h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(result_of(&calls, "m1").contains("at most 12 steps"));
    for c in ["c1", "c2", "c3"] {
        assert!(
            result_of(&calls, c).contains("one line of 1 to 100 characters"),
            "{c}: {}",
            result_of(&calls, c)
        );
    }
    assert!(result_of(&calls, "p1").contains("At most one step is in progress"));
    assert!(
        result_of(&calls, "k1").contains("do not fit"),
        "{}",
        result_of(&calls, "k1")
    );
    assert!(result_of(&calls, "s1").contains(crate::secrets::REDACTED));
    let shown = lists(&events);
    assert_eq!(shown.len(), 1, "only the redacted list was saved");
    assert!(
        !shown[0][0].content.contains(&key)
            && shown[0][0].content.contains(crate::secrets::REDACTED),
        "{:?}",
        shown[0]
    );
    assert!(
        !h.sidecar_items(&id)
            .iter()
            .any(|item| matches!(item, Item::TodosUpdated { .. })
                && serde_json::to_string(item).unwrap().contains(&key))
    );
}

/// A conversation read again from its record shows its lists, the latest
/// last.
/// Mutant: the records not read back as events.
#[test]
fn a_reopened_conversation_shows_its_list() {
    let h = H::new("todos-seed");
    let convo = h.chat.inner.convo("0123456789ab");
    let first = vec![step("Read", TodoStatus::InProgress)];
    let second = vec![step("Read", TodoStatus::Done)];
    let records = [first.clone(), second.clone()].map(|items| Item::TodosUpdated {
        turn: "a1b2c3d4e5f6".into(),
        call_id: "u1".into(),
        items,
        at: 1.0,
    });
    super::views::seed(&convo, &records);
    let (events, _) = convo.log.recorded();
    assert_eq!(lists(&events), [first, second.clone()]);
    assert_eq!(todos::latest(&records), second);
}
