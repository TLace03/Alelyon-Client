//! Stack decisions on the to-do list (`super::choices`), end to end over the
//! agent harness: a step that needs a database is marked; recommend_stack
//! offers the catalog's options ranked for the needs, with how each fits;
//! the reader's pick is recorded, kept as a note for the folder, and its
//! setup steps pinned as a checklist; the step then names the pick; a step in
//! an area the project already uses names that option; review_stack lists
//! the stack and its suggestions.

use lattice_agents::model::InputItem;
use lattice_protocol::conversation::{AgentChatService, ConversationEvent, ConversationEventKind, Mode, TodoItem};
use serde_json::json;

use super::agent_tests::{H, call, local, say};

fn todos(events: &[ConversationEvent]) -> Vec<TodoItem> {
    events
        .iter()
        .rev()
        .find_map(|e| match &e.kind {
            ConversationEventKind::TodosUpdated { items } => Some(items.clone()),
            _ => None,
        })
        .unwrap()
}

#[test]
fn a_database_step_is_decided_by_the_reader_from_the_catalog_and_set_up_from_its_guide() {
    let h = H::new("stack-choice");
    std::fs::write(h.folder.join("package.json"), r#"{"dependencies": {"express": "^4"}}"#).unwrap();
    let ws = h.workspace();
    let model = h.script(vec![
        call(
            "update_todos",
            json!({"items": [
                {"content": "Set up the database for orders", "status": "in_progress"},
                {"content": "Add an API route for orders", "status": "pending"},
                {"content": "Fix the typo in the README", "status": "pending"}
            ]}),
            "u1",
        ),
        call("recommend_stack", json!({"area": "database", "needs": ["teleport"]}), "r0"),
        call(
            "recommend_stack",
            json!({"area": "database", "needs": ["relational", "transactions", "managed"], "step": 1,
                   "why": "Orders relate to customers and must not be half-written.",
                   "fit": {"postgres": "Orders and customers join cleanly; managed hosting everywhere."}}),
            "r1",
        ),
        say("Waiting on the pick."),
    ]);
    let id = h.agent(None, "Build an order system.", &ws);
    let offered = h.events_until(&id, |events| events.iter().any(|e| matches!(e.kind, ConversationEventKind::ChoiceOffered { .. })));
    // The step needing a database is marked; the API step names what the project uses; the typo step is nothing.
    let steps = todos(&offered);
    assert_eq!(steps[0].decision.as_deref(), Some("database"));
    assert_eq!((steps[1].decision.as_deref(), steps[1].chosen.as_deref()), (None, Some("Fastify or Express (Node.js)")));
    assert_eq!((steps[2].decision.clone(), steps[2].chosen.clone()), (None, None));
    let card = offered
        .iter()
        .find_map(|e| match &e.kind {
            ConversationEventKind::ChoiceOffered { area, options, recommended, why, step, .. } => {
                Some((area.clone(), options.clone(), recommended.clone(), why.clone(), *step))
            }
            _ => None,
        })
        .unwrap();
    assert_eq!((card.0.as_str(), card.2.as_str(), card.4), ("database", "postgres", Some(1)));
    assert!(card.1.len() <= 4 && card.1[0].fit.starts_with("Orders and customers"), "{:?}", card.1);
    assert!(card.3.contains("half-written"));

    h.chat.answer(&id, "r1", "postgres".into()).unwrap();
    let events = h.turns_end(&id, 1);
    assert!(events.iter().any(|e| matches!(&e.kind, ConversationEventKind::ChoiceMade { option, by_agent: false, .. } if option == "postgres")));
    assert!(events.iter().any(|e| matches!(&e.kind, ConversationEventKind::StepsGiven { title, steps, .. } if title == "Set up PostgreSQL" && steps.len() >= 3)));
    let results: Vec<String> = model
        .calls()
        .iter()
        .flat_map(|r| r.input.clone())
        .filter_map(|i| match i {
            InputItem::ToolResult { call_id, output } => Some(format!("{call_id}: {output}")),
            _ => None,
        })
        .collect();
    let said = |id: &str| results.iter().find(|r| r.starts_with(&format!("{id}: "))).cloned().unwrap_or_default();
    assert!(said("u1").contains("Step 1 needs a decision (Primary database): call recommend_stack"), "{}", said("u1"));
    assert!(said("r0").contains("teleport is not a need of database"), "{}", said("r0"));
    assert!(said("r1").contains("The user chose PostgreSQL for Primary database") && said("r1").contains("Set up PostgreSQL"), "{}", said("r1"));
    let notes = super::memory::list(&super::memory::file(&h.state, &ws));
    assert!(notes.iter().any(|n| n.text.contains("Primary database = PostgreSQL")), "{notes:?}");

    // The same step again names the pick, and asking again is told the decision.
    model.enqueue(call("update_todos", json!({"items": [{"content": "Set up the database for orders", "status": "in_progress"}]}), "u2"));
    model.enqueue(call("recommend_stack", json!({"area": "database", "needs": ["embedded"]}), "r2"));
    model.enqueue(call("review_stack", json!({}), "v1"));
    model.enqueue(say("Done."));
    let sent = h.send(Some(&id), "Go on.", "local", local(), Mode::Agent, Some(&ws));
    assert!(sent.is_ok(), "{sent:?}");
    let events = h.turns_end(&id, 2);
    let steps = todos(&events);
    assert_eq!((steps[0].decision.clone(), steps[0].chosen.as_deref()), (None, Some("PostgreSQL")));
    let results: Vec<String> = model
        .calls()
        .iter()
        .flat_map(|r| r.input.clone())
        .filter_map(|i| match i {
            InputItem::ToolResult { call_id, output } => Some(format!("{call_id}: {output}")),
            _ => None,
        })
        .collect();
    let said = |id: &str| results.iter().find(|r| r.starts_with(&format!("{id}: "))).cloned().unwrap_or_default();
    assert!(said("r2").contains("already chose PostgreSQL"), "{}", said("r2"));
    let review = said("v1");
    assert!(review.contains("TypeScript (Node.js)") && review.contains("Commit a lock file") && review.contains("Add security headers"), "{review}");
}
