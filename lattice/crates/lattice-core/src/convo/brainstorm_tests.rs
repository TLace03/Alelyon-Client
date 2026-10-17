//! The brainstorm chat (`super::groups`), over the agent harness: begun for a
//! project, its first turn already asks only, offers read-only tools and
//! `offer_to_group`, and is told every group; an idea it offers waits as a
//! card, and pressing it sends the prompt to that group's session (or a new
//! group), not to a task.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ConversationEvent, ConversationEventKind, Mode, SendRequest, TaskOutcome,
};
use serde_json::json;

use super::agent_tests::{H, call, local, say};
use super::groups::{HEADER, Placement};

fn in_project(h: &H, ws: &str, project: &str, conversation: Option<&str>, text: &str) -> String {
    match h
        .runtime
        .block_on(h.chat.send(SendRequest {
            conversation: conversation.map(str::to_owned),
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

fn offered(events: &[ConversationEvent]) -> Vec<(String, Option<String>)> {
    events
        .iter()
        .filter_map(|e| match &e.kind {
            ConversationEventKind::TaskSuggested { task, group, .. } => Some((task.clone(), group.clone())),
            _ => None,
        })
        .collect()
}

fn result_of(calls: &[ModelRequest], call_id: &str) -> String {
    calls
        .iter()
        .flat_map(|r| r.input.iter())
        .find_map(|i| match i {
            InputItem::ToolResult { call_id: id, output } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn offer(group: &str, id: &str) -> lattice_agents::testing::ScriptedStep {
    call(
        "offer_to_group",
        json!({"group": group, "title": "Add webhook retries", "summary": "Failed webhooks are lost.", "prompt": "Retry failed webhooks three times."}),
        id,
    )
}

/// Mutants: a brainstorm turn with write tools or Agent mode; no groups in
/// its lead; an idea started as a task, or sent to a new chat instead of the
/// group's session.
#[test]
fn the_brainstorm_chat_asks_only_knows_the_groups_and_sends_an_idea_to_a_group_only_when_pressed() {
    let h = H::new("brainstorm");
    let ws = h.workspace();
    let shop = h.runtime.block_on(h.chat.project_create("Shop".into())).unwrap().id;
    let model = h.script(vec![say("The checkout is set up.")]);
    let checkout = in_project(&h, &ws, &shop, None, "Set up the checkout.");
    h.turns_end(&checkout, 1);
    let group = h.chat.group_add(&checkout, &Placement::New, Some(&shop)).unwrap();
    h.chat.group_rename(&group.id, "Payments").unwrap();

    // The brainstorm chat, begun for the project: sent in Agent mode, it asks only.
    model.enqueue(offer("g_unknown", "bad"));
    model.enqueue(offer(&group.id, "o1"));
    model.enqueue(offer("new", "o2"));
    model.enqueue(say("Two ideas offered."));
    let text = "What should we do about lost webhooks?";
    h.chat.begin_brainstorm(text, Some(&shop));
    let id = in_project(&h, &ws, &shop, None, text);
    let events = h.turns_end(&id, 1);
    assert_eq!(h.chat.brainstorms().iter().map(|b| (b.project.clone(), b.conversation.clone())).collect::<Vec<_>>(), [(Some(shop.clone()), id.clone())]);
    let calls = model.calls();
    let first = calls.iter().find(|r| r.input.iter().any(|i| matches!(i, InputItem::User(t) if t.contains(text)))).unwrap();
    let tools: Vec<&str> = first.tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tools.contains(&"offer_to_group") && tools.contains(&"read_file") && tools.contains(&"recall"), "{tools:?}");
    for writes in ["write_file", "edit_file", "run_command", "suggest_task", "message_task", "stop_task", "switch_project", "propose_plan"] {
        assert!(!tools.contains(&writes), "{writes} offered in {tools:?}");
    }
    let lead = first.input.iter().find_map(|i| match i { InputItem::User(t) if t.contains(HEADER) => Some(t.clone()), _ => None }).expect("the groups are told");
    assert!(lead.contains("\u{201c}Payments\u{201d}") && lead.contains(&group.id) && lead.contains("Set up the checkout."), "{lead}");
    assert!(result_of(&calls, "bad").contains("not one of this project's groups"));
    let cards = offered(&events);
    assert_eq!(cards.iter().map(|(_, g)| g.clone()).collect::<Vec<_>>(), [Some(group.id.clone()), Some("new".to_owned())]);
    assert!(h.chat.inner.groups.of(&id).is_none(), "the brainstorm chat is in no group");

    // Pressed: the idea goes to the group's session, its newest topic, and is no task.
    model.enqueue(say("Retries added."));
    let accepted = h.runtime.block_on(h.chat.start_task(&id, &cards[0].0, "local".into(), local(), Mode::Agent)).unwrap();
    let Accepted::Started { conversation, .. } = accepted else { panic!("{accepted:?}") };
    assert_eq!(conversation.id, checkout, "sent to the group's own topic");
    h.turns_end(&checkout, 2);
    assert!(!h.chat.task_origins().contains_key(&checkout), "not a task");
    let after = h.events_until(&id, |e| e.iter().any(|e| matches!(&e.kind, ConversationEventKind::TaskSettled { .. })));
    assert!(after.iter().any(|e| matches!(&e.kind, ConversationEventKind::TaskSettled { outcome: TaskOutcome::Started { conversation }, .. } if *conversation == checkout)));

    // The other card starts a group of its own in the project.
    model.enqueue(say("A new line of work."));
    let Accepted::Started { conversation: fresh, .. } =
        h.runtime.block_on(h.chat.start_task(&id, &cards[1].0, "local".into(), local(), Mode::Agent)).unwrap()
    else {
        panic!()
    };
    assert_ne!(fresh.id, checkout);
    let made = h.chat.inner.groups.of(&fresh.id).unwrap();
    assert_ne!(made.id, group.id);
    assert_eq!(made.project.as_deref(), Some(shop.as_str()));
    h.turns_end(&fresh.id, 1);

    // A second message in the brainstorm chat stays Ask-only though sent in Agent mode.
    model.enqueue(say("Noted."));
    in_project(&h, &ws, &shop, Some(&id), "And refunds?");
    h.turns_end(&id, 2);
    let snapshot = h.runtime.block_on(h.chat.open(&id)).unwrap();
    assert_eq!(snapshot.conversation.mode, Mode::Ask);
}
