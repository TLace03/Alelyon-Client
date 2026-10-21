//! The fleet as another agent sees it (`super::fleet`): a group's topics with the step under way and their to-dos,
//! and every question waiting for the reader named by its group and topic, with its options, until it is answered.

use lattice_agents::model::InputItem;
use lattice_protocol::conversation::{AgentChatService, ConversationEventKind};
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::groups::{Placement, Status};

/// Mutants: a project's groups left out; the step under way not read from the latest to-do list; a waiting question not counted on its topic, or
/// not listed with its group, its options and its call; an answered question still listed.
#[test]
fn a_groups_topics_show_their_step_and_a_waiting_question_is_listed_until_answered() {
    let h = H::new("fleet-question");
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "update_todos",
            json!({"items": [
                {"content": "Write the parser", "status": "done"},
                {"content": "Test the parser", "status": "in_progress"},
                {"content": "Write the docs", "status": "pending"}
            ]}),
            "u1",
        ),
        call("ask_question", json!({"question": "Which format first?", "options": ["CSV", "JSON"]}), "q1"),
        say("Starting with CSV."),
    ]);
    let id = h.agent(None, "Build the parser.", &ws);
    h.events_until(&id, |events| events.iter().any(|e| matches!(e.kind, ConversationEventKind::Question { .. })));
    let group = h.chat.group_add(&id, &Placement::New, None).unwrap();

    let elsewhere = h.chat.group_create("Docs site", Some("site")).unwrap();
    let cards = h.chat.fleet_groups();
    let card_of = |id: &str| cards.iter().find(|(_, c)| c.group == id).cloned().unwrap();
    let (project, card) = card_of(&group.id);
    assert_eq!((project, card.status, card.waits.clone()), (None, Status::Waiting, vec!["a question"]));
    assert_eq!(card_of(&elsewhere.id).0.as_deref(), Some("site"), "every project's groups are listed");

    let topics = h.chat.fleet_topics(&group.id).unwrap();
    assert_eq!(topics.len(), 1);
    let topic = &topics[0];
    assert_eq!(topic.id, id);
    assert_eq!(topic.doing.as_deref(), Some("Test the parser"));
    assert_eq!((topic.todos_done, topic.todos_total), (1, 3));
    assert_eq!(topic.todos.iter().map(|t| t.content.as_str()).collect::<Vec<_>>(), ["Write the parser", "Test the parser", "Write the docs"]);
    assert!(!topic.recent.is_empty() && topic.recent.len() <= super::fleet::RECENT, "{:?}", topic.recent);
    assert_eq!(topic.waiting, 1);
    assert!(topic.running, "it waits inside its turn");
    assert!(h.chat.fleet_topics("no-such-group").is_none());

    let waiting = h.chat.fleet_waiting();
    assert_eq!(waiting.len(), 1, "{waiting:?}");
    let wait = &waiting[0];
    assert_eq!((wait.kind, wait.call.as_str(), wait.text.as_str()), ("question", "q1", "Which format first?"));
    assert_eq!(wait.options, ["CSV", "JSON"]);
    assert_eq!(wait.topic, id);
    assert_eq!(wait.group.as_deref(), Some(group.id.as_str()));
    assert!(wait.group_name.as_deref().is_some_and(|n| !n.is_empty()));

    let waiting_state = h.chat.fleet_state();
    assert_eq!(waiting_state, h.chat.fleet_state(), "nothing changed: the same digest");
    assert_eq!(waiting_state.waiting.len(), 1);

    h.chat.answer(&id, "q1", "CSV".into()).unwrap();
    h.turns_end(&id, 1);
    assert!(h.chat.fleet_waiting().is_empty(), "an answered question no longer waits");
    assert_ne!(h.chat.fleet_state().digest, waiting_state.digest, "an answer changes the fleet's digest");
    assert_eq!(h.chat.fleet_topics(&group.id).unwrap()[0].waiting, 0);
    let calls = model.calls();
    assert!(
        matches!(calls.last().unwrap().input.last(), Some(InputItem::ToolResult { output, .. }) if output == "The user answered: CSV")
    );
}

/// Sessions are found by what was said in them, one hit a session, best first; nothing for words never said.
#[test]
fn sessions_are_found_by_what_was_said_in_them() {
    let h = H::new("fleet-search");
    let ws = h.workspace();
    h.script(vec![say("The CSV parser reads quoted fields now."), say("The parser handles CSV headers too."), say("Lunch is at noon.")]);
    let parser = h.agent(None, "Fix the CSV parser for quoted fields.", &ws);
    h.turns_end(&parser, 1);
    h.agent(Some(&parser), "And CSV headers.", &ws);
    h.turns_end(&parser, 2);
    let lunch = h.agent(None, "When is lunch?", &ws);
    h.turns_end(&lunch, 1);
    let hits = h.chat.search_sessions("CSV parser", 10);
    assert_eq!(hits.iter().map(|h| h.conversation.as_str()).collect::<Vec<_>>(), [parser.as_str()], "one hit a session");
    assert!(h.chat.search_sessions("zebra", 10).is_empty());
    assert_eq!(h.chat.search_sessions("lunch noon", 10)[0].conversation, lunch);
}

/// A question in a topic of no group is still listed, with no group named.
#[test]
fn a_question_outside_any_group_is_listed_without_one() {
    let h = H::new("fleet-no-group");
    let ws = h.workspace();
    h.script(vec![call("ask_question", json!({"question": "Go ahead?"}), "q1"), say("ok")]);
    let id = h.agent(None, "Start.", &ws);
    h.events_until(&id, |events| events.iter().any(|e| matches!(e.kind, ConversationEventKind::Question { .. })));
    let waiting = h.chat.fleet_waiting();
    assert_eq!(waiting.len(), 1);
    assert_eq!((waiting[0].group.as_deref(), waiting[0].options.len()), (None, 0));
    h.chat.answer(&id, "q1", "yes".into()).unwrap();
    h.turns_end(&id, 1);
}
