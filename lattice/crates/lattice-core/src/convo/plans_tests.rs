//! Plan mode (`super::plans`), end to end over the agent harness: a plan
//! proposed in Ask mode is recorded and shown; Approve goes on in the same
//! conversation in Agent mode from the words that name it, once; Keep planning
//! sets it aside; a new proposal replaces the one that waits; bad plans are
//! refused and a secret redacted; and a reopened conversation shows its plans.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::RefusalKind;
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ConversationEvent, ConversationEventKind, Mode, PlanOutcome,
    is_plan_id,
};
use serde_json::json;

use super::agent_tests::{H, call, local, say, secret};
use super::item::Item;
use super::plans::{self, words};

const PLAN: &str =
    "1. Compare digit runs by value in `tree::order`.\n2. Add a test: file2 before file10.";

fn propose(title: &str, plan: &str, id: &str) -> lattice_agents::testing::ScriptedStep {
    call("propose_plan", json!({"title": title, "plan": plan}), id)
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

/// Every plan the events proposed, in order: its id and title.
fn proposed(events: &[ConversationEvent]) -> Vec<(String, String)> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::PlanProposed { plan, title, .. } => {
                Some((plan.clone(), title.clone()))
            }
            _ => None,
        })
        .collect()
}

fn settled(events: &[ConversationEvent], plan: &str) -> Vec<PlanOutcome> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::PlanSettled { plan: p, outcome } if p == plan => {
                Some(outcome.clone())
            }
            _ => None,
        })
        .collect()
}

/// An Ask-mode send in the trusted scratch folder; the conversation's id.
fn ask(h: &H, text: &str, workspace: &str) -> String {
    match h
        .send(None, text, "local", local(), Mode::Ask, Some(workspace))
        .unwrap()
    {
        Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    }
}

fn mode_of(h: &H, id: &str) -> Mode {
    h.runtime
        .block_on(h.chat.list())
        .unwrap()
        .conversations
        .into_iter()
        .find(|c| c.id == id)
        .unwrap()
        .mode
}

/// A plan proposed in Ask mode waits; Approve goes on in the same
/// conversation, in Agent mode, from the words that name it, with the Agent
/// tools offered; the plan is approved once and stays so.
/// Mutants: Approve sending in Ask mode; Approve not reading the record (a
/// settled plan approved again).
#[test]
fn an_approved_plan_goes_on_in_agent_mode_in_the_same_chat_once() {
    let h = H::new("plans-approve");
    let ws = h.workspace();
    let model = h.script(vec![
        propose("Natural sort in the explorer", PLAN, "p1"),
        say("That is my plan."),
    ]);
    let id = ask(&h, "Make the explorer sort file2 before file10.", &ws);
    let events = h.turns_end(&id, 1);
    let shown = proposed(&events);
    assert_eq!(shown.len(), 1);
    let (plan, title) = shown[0].clone();
    assert!(is_plan_id(&plan), "{plan}");
    assert_eq!(title, "Natural sort in the explorer");
    let heard = result_of(&model.calls(), "p1");
    assert!(
        heard.contains(&plan) && heard.contains("Stop here"),
        "{heard}"
    );
    assert!(h.sidecar_items(&id).iter().any(|item| matches!(
        item,
        Item::PlanProposed { plan: p, text, .. } if *p == plan && text == PLAN
    )));
    assert!(settled(&events, &plan).is_empty(), "it waits");
    assert_eq!(mode_of(&h, &id), Mode::Ask, "nothing changes until Approve");

    model.enqueue(say("Done: digit runs compare by value."));
    let accepted = h
        .runtime
        .block_on(h.chat.approve_plan(&id, &plan, "local".into(), local()))
        .unwrap();
    let Accepted::Started {
        conversation,
        user_turn,
        ..
    } = accepted
    else {
        panic!("{accepted:?}")
    };
    assert_eq!(conversation.id, id, "the same conversation");
    assert_eq!(
        user_turn.text,
        plans::approval("Natural sort in the explorer")
    );
    let events = h.events_until(&id, |events| {
        !settled(events, &plan).is_empty() && super::agent_tests::ended(events) >= 2
    });
    assert_eq!(settled(&events, &plan), [PlanOutcome::Approved]);
    assert_eq!(mode_of(&h, &id), Mode::Agent);
    let offered: Vec<String> = model
        .calls()
        .last()
        .unwrap()
        .tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect();
    assert!(
        offered.iter().any(|name| name == "edit_file")
            && !offered.iter().any(|name| name == "propose_plan"),
        "Agent mode's tools: {offered:?}"
    );
    let again = h
        .runtime
        .block_on(h.chat.approve_plan(&id, &plan, "local".into(), local()))
        .unwrap_err();
    assert_eq!(
        (again.kind, again.message.as_str()),
        (RefusalKind::Conflict, words::APPROVED)
    );
    let kept = h
        .runtime
        .block_on(h.chat.keep_planning(&id, &plan))
        .unwrap_err();
    assert_eq!(kept.message, words::APPROVED);
}

/// Keep planning sets a plan aside; a plan proposed while another waits
/// replaces it; neither can then be approved.
/// Mutants: a new proposal leaving the one that waited waiting; Keep planning
/// not recorded.
#[test]
fn keep_planning_sets_a_plan_aside_and_a_new_one_replaces_the_one_that_waits() {
    let h = H::new("plans-keep");
    let ws = h.workspace();
    let model = h.script(vec![
        propose("First plan", "1. One thing.", "p1"),
        say("Plan one."),
    ]);
    let id = ask(&h, "Plan the change.", &ws);
    let events = h.turns_end(&id, 1);
    let first = proposed(&events)[0].0.clone();
    h.runtime
        .block_on(h.chat.keep_planning(&id, &first))
        .unwrap();

    model.enqueue(propose("Second plan", "1. Another thing.", "p2"));
    model.enqueue(propose("Third plan", "1. A better thing.", "p3"));
    model.enqueue(say("Plans two and three."));
    h.send(
        Some(&id),
        "Do it differently.",
        "local",
        local(),
        Mode::Ask,
        Some(&ws),
    )
    .unwrap();
    let events = h.turns_end(&id, 2);
    let all = proposed(&events);
    assert_eq!(all.len(), 3);
    let (second, third) = (all[1].0.clone(), all[2].0.clone());
    assert_eq!(settled(&events, &first), [PlanOutcome::KeptPlanning]);
    assert_eq!(settled(&events, &second), [PlanOutcome::Replaced]);
    assert!(settled(&events, &third).is_empty(), "the latest waits");
    for (plan, why) in [(&first, words::KEPT), (&second, words::REPLACED)] {
        let refused = h
            .runtime
            .block_on(h.chat.approve_plan(&id, plan, "local".into(), local()))
            .unwrap_err();
        assert_eq!(
            (refused.kind, refused.message.as_str()),
            (RefusalKind::Conflict, why)
        );
    }
    let unknown = h
        .runtime
        .block_on(h.chat.keep_planning(&id, "plan_0000000000000000"))
        .unwrap_err();
    assert_eq!(
        (unknown.kind, unknown.message.as_str()),
        (RefusalKind::NotFound, words::NO_PLAN)
    );
}

/// A bad title or plan is refused with its sentence and saves nothing; a
/// secret is redacted before it is kept or shown.
/// Mutant: the plan not redacted.
#[test]
fn bad_plans_are_refused_and_a_secret_is_redacted() {
    let h = H::new("plans-bounds");
    let ws = h.workspace();
    let key = secret();
    let model = h.script(vec![
        propose("two\nlines", PLAN, "t1"),
        propose("", PLAN, "t2"),
        propose("Plan", "  ", "c1"),
        propose("Plan", &"x".repeat(plans::MAX_PLAN_CHARS + 1), "c2"),
        propose("Keys", &format!("1. Rotate {key}."), "s1"),
        say("Done."),
    ]);
    let id = ask(&h, "Plan it.", &ws);
    let events = h.turns_end(&id, 1);
    let calls = model.calls();
    for t in ["t1", "t2"] {
        assert!(
            result_of(&calls, t).contains("one line of 1 to 80 characters"),
            "{t}"
        );
    }
    for c in ["c1", "c2"] {
        assert!(
            result_of(&calls, c).contains("1 to 2,000 characters"),
            "{c}"
        );
    }
    assert!(result_of(&calls, "s1").contains(crate::secrets::REDACTED));
    assert_eq!(proposed(&events).len(), 1, "only the redacted one");
    let text = events
        .iter()
        .find_map(|event| match &event.kind {
            ConversationEventKind::PlanProposed { text, .. } => Some(text.clone()),
            _ => None,
        })
        .unwrap();
    assert!(
        !text.contains(&key) && text.contains(crate::secrets::REDACTED),
        "{text}"
    );
}

/// A conversation read again from its record shows its plans as they were
/// left.
/// Mutant: the records not read back as events.
#[test]
fn a_reopened_conversation_shows_its_plans_as_they_were_left() {
    let h = H::new("plans-seed");
    let convo = h.chat.inner.convo("0123456789ab");
    let plan = "plan_0011223344556677".to_owned();
    super::views::seed(
        &convo,
        &[
            Item::PlanProposed {
                turn: "a1b2c3d4e5f6".into(),
                call_id: "p1".into(),
                plan: plan.clone(),
                title: "The plan".into(),
                text: PLAN.into(),
                at: 1.0,
            },
            Item::PlanSettled {
                plan: plan.clone(),
                outcome: PlanOutcome::KeptPlanning,
                at: 2.0,
            },
        ],
    );
    let (events, _) = convo.log.recorded();
    assert_eq!(proposed(&events), [(plan.clone(), "The plan".to_owned())]);
    assert_eq!(settled(&events, &plan), [PlanOutcome::KeptPlanning]);
}
