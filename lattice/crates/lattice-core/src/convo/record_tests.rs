//! A group's thread record (`super::record`): built from its topics (where each stands, the reader's answers as
//! decisions, what is open, the pointers), with the agents' recorded decisions and the reader's notes; what a turn in
//! the group starts with; and what the reader takes off it.

use lattice_agents::model::InputItem;
use lattice_protocol::conversation::{AgentChatService, ConversationEventKind};
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::groups::Placement;
use super::record::{self, By, LEAD_CHARS};

/// Mutants: an answered question not kept as a decision; an agent's decision not kept or kept twice; a step not done
/// not listed as open; the reader's notes not in the record; a decision taken off still shown; the record not given
/// to the group's next turn; the record given to a topic of no group.
#[test]
fn a_groups_record_keeps_answers_and_decisions_and_its_next_turn_starts_with_it() {
    let h = H::new("record");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "update_todos",
            json!({"items": [
                {"content": "Write the parser", "status": "done"},
                {"content": "Write the docs", "status": "in_progress"}
            ]}),
            "u1",
        ),
        call("ask_question", json!({"question": "Which format first?", "options": ["CSV", "JSON"]}), "q1"),
        say("CSV it is."),
    ]);
    let id = h.agent(None, "Build the parser.", &ws);
    h.events_until(&id, |events| events.iter().any(|e| matches!(e.kind, ConversationEventKind::Question { .. })));
    let group = h.chat.group_add(&id, &Placement::New, None).unwrap();
    h.chat.answer(&id, "q1", "CSV".into()).unwrap();
    h.turns_end(&id, 1);

    // The agent records a decision in the group's topic (the tool is offered there).
    model.enqueue(call("record_decision", json!({"decision": "Dates are ISO 8601 in every output."}), "r1"));
    model.enqueue(call("record_decision", json!({"decision": "Dates are ISO 8601 in every output."}), "r2"));
    model.enqueue(say("Recorded."));
    h.agent(Some(&id), "Settle the date format.", &ws);
    h.turns_end(&id, 2);
    let offered: Vec<String> = model.calls().last().unwrap().tools.iter().map(|t| t.name.clone()).collect();
    assert!(offered.iter().any(|t| t == "record_decision"), "{offered:?}");

    h.chat.record_notes(&group.id, "Ship before Friday.").unwrap();
    let rec = h.chat.thread_record(&group.id).unwrap();
    assert_eq!(rec.notes, "Ship before Friday.");
    let texts: Vec<(&str, By)> = rec.decisions.iter().map(|d| (d.text.as_str(), d.by)).collect();
    assert_eq!(
        texts,
        [("Which format first? \u{2192} CSV", By::Reader), ("Dates are ISO 8601 in every output.", By::Agent)],
        "the answer and the agent's decision, once each, oldest first"
    );
    assert!(rec.open.iter().any(|o| o.ends_with("Write the docs")), "{:?}", rec.open);
    assert!(!rec.open.iter().any(|o| o.ends_with("Write the parser")));
    assert_eq!(rec.stands.len(), 1);

    // A new topic of the group starts with the record (its own place left out).
    model.enqueue(say("On it."));
    let other = h.agent(None, "Write the CSV exporter.", &ws);
    h.turns_end(&other, 1);
    h.chat.group_add(&other, &Placement::Group(group.id.clone()), None).unwrap();
    model.enqueue(say("Exporting."));
    h.agent(Some(&other), "Go on.", &ws);
    h.turns_end(&other, 2);
    let lead = model.calls().last().unwrap().input.iter().find_map(|i| match i {
        InputItem::User(text) if text.contains("The record of this group's work") => Some(text.clone()),
        _ => None,
    });
    let lead = lead.expect("the group's turn starts with its record");
    assert!(lead.contains("Ship before Friday.") && lead.contains("Which format first? \u{2192} CSV"), "{lead}");
    assert!(lead.contains("Write the docs"));

    // Taken off by the reader: an answer's decision and an agent's.
    for d in rec.decisions.clone() {
        h.chat.record_take_off(&group.id, &d.id).unwrap();
    }
    assert!(h.chat.thread_record(&group.id).unwrap().decisions.is_empty());
    assert!(h.chat.thread_record("no-such-group").is_none());
    // A decision made outside Lattice, by the reader's own agent.
    h.chat.record_decision(&group.id, "Releases go out on Thursdays.", "your agent").unwrap();
    let outside = h.chat.thread_record(&group.id).unwrap().decisions;
    assert_eq!(outside.iter().map(|d| (d.text.as_str(), d.topic.as_str())).collect::<Vec<_>>(), [("Releases go out on Thursdays.", "your agent")]);
    assert!(h.chat.record_decision("no-such-group", "x", "your agent").is_err());
    assert!(h.chat.record_notes("no-such-group", "x").is_err());
}

/// The lead is cut to its bound, and an empty record adds nothing.
#[test]
fn the_records_lead_is_bounded_and_an_empty_one_adds_nothing() {
    let mut rec = record::Record { name: "Parser".into(), ..record::Record::default() };
    assert_eq!(record::lead_text(&rec, "t"), None);
    rec.open = (0..400).map(|i| format!("Step {i} of a very long list of things still to do")).collect();
    rec.notes = "n".repeat(5_000);
    let lead = record::lead_text(&rec, "t").unwrap();
    assert!(lead.chars().count() <= LEAD_CHARS);
}
