//! The board (`super::board`): the column a group's work gives, the
//! reader's move kept until the work changes, the reader's order, and the
//! queue of tasks past the limit starting the highest-priority group's first.

use std::collections::HashMap;

use lattice_protocol::conversation::{AgentChatService, ConversationEventKind, Mode, TaskOutcome};
use serde_json::json;

use super::agent_tests::{H, call, local, say};
use super::board::{TopicState, cards, derived, work};
use super::groups::{Group, Move, Placement, Status};
use super::task_runs::Limit;

fn topic(turns: u64) -> TopicState {
    TopicState { turns, ..TopicState::default() }
}

/// Mutants: a waiting topic not putting its group in Waiting on you; done
/// steps not making it Done; a group with no work not Planned.
#[test]
fn the_column_follows_the_work() {
    assert_eq!(derived(&[]), Status::Planned, "nothing yet");
    assert_eq!(derived(&[topic(0)]), Status::Planned);
    assert_eq!(derived(&[TopicState { todos_total: 3, ..topic(2) }]), Status::Planned, "steps written, none done");
    assert_eq!(derived(&[TopicState { todos_total: 3, todos_done: 1, ..topic(4) }]), Status::InProgress);
    assert_eq!(derived(&[topic(4)]), Status::InProgress, "work done, no steps to tell it is finished");
    assert_eq!(derived(&[TopicState { todos_total: 3, todos_done: 3, ..topic(6) }]), Status::Done);
    let working = TopicState { running: true, todos_total: 3, todos_done: 3, ..topic(6) };
    assert_eq!(derived(std::slice::from_ref(&working)), Status::InProgress, "a working topic is not done");
    for waits in [
        TopicState { approval: true, ..working.clone() },
        TopicState { question: true, ..topic(1) },
        TopicState { decision: true, ..topic(1) },
    ] {
        assert_eq!(derived(&[topic(3), waits]), Status::Waiting);
    }
    assert_eq!(work(&[topic(2), topic(5)]), 7);
}

/// Mutants: a move that never lapses, or never holds; the order ignored;
/// another project's or an archived group shown.
#[test]
fn a_move_holds_until_the_work_changes_and_cards_come_in_the_readers_order() {
    let group = |id: &str, project: Option<&str>, topics: &[&str], archived: bool| Group {
        id: id.into(),
        name: String::new(),
        project: project.map(str::to_owned),
        topics: topics.iter().map(|t| t.to_string()).collect(),
        created: 1.0,
        updated: 1.0,
        archived,
    };
    let groups = vec![
        (group("a", None, &["t1", "t2"], false), "A".to_owned()),
        (group("b", None, &["t3"], false), "B".to_owned()),
        (group("c", Some("p"), &["t4"], false), "C".to_owned()),
        (group("d", None, &["t5"], true), "D".to_owned()),
    ];
    let states: HashMap<&str, TopicState> = [
        ("t1", TopicState { todos_total: 2, todos_done: 1, ..topic(3) }),
        ("t2", TopicState { task: true, running: true, ..topic(1) }),
        ("t3", TopicState { question: true, ..topic(2) }),
    ]
    .into_iter()
    .collect();
    let ranks: HashMap<String, usize> = [("b".to_owned(), 0), ("a".to_owned(), 1)].into_iter().collect();
    let state = |t: &str| states.get(t).cloned();
    let none = HashMap::new();
    let shown = cards(&groups, &ranks, &HashMap::new(), &none, "2026-10-15", None, state);
    assert_eq!(shown.iter().map(|c| c.group.as_str()).collect::<Vec<_>>(), ["b", "a"], "the reader's order, this project's, not archived");
    let a = &shown[1];
    assert_eq!((a.status, a.moved, a.topics, a.tasks, a.tasks_running, a.todos_done, a.todos_total), (Status::InProgress, false, 1, 1, 1, 1, 2));
    assert_eq!((shown[0].status, shown[0].waits.clone()), (Status::Waiting, vec!["a question"]));

    // Moved to Done at its work now (4 turns): it stays there.
    let moves: HashMap<String, Move> = [("a".to_owned(), Move { status: Status::Done, work: 4 })].into_iter().collect();
    let held = cards(&groups, &ranks, &moves, &none, "2026-10-15", None, state);
    assert_eq!((held[1].status, held[1].moved), (Status::Done, true));
    // A turn later, it follows the work again.
    let later: HashMap<String, Move> = [("a".to_owned(), Move { status: Status::Done, work: 3 })].into_iter().collect();
    let lapsed = cards(&groups, &ranks, &later, &none, "2026-10-15", None, state);
    assert_eq!((lapsed[1].status, lapsed[1].moved), (Status::InProgress, false));
}

fn suggest(id: &str, title: &str) -> lattice_agents::testing::ScriptedStep {
    call("suggest_task", json!({"title": title, "summary": "s", "prompt": format!("Do: {title}.")}), id)
}

/// Over the harness: the reader orders the groups and moves a card; with a
/// limit of one, tasks started while the slot is taken wait, and when it
/// frees the higher-priority group's task starts first, though it waited
/// last.
/// Mutants: the queue started in the order it was filled; a raise that does
/// not move; a move not kept.
#[test]
fn waiting_tasks_start_by_the_readers_priority_when_a_slot_frees() {
    let h = H::new("board-queue");
    let ws = h.workspace();
    let model = h.script(vec![suggest("a1", "Low task"), say("Suggested.")]);
    let low = h.agent(None, "Work on the docs.", &ws);
    let low_events = h.turns_end(&low, 1);
    model.enqueue(suggest("b1", "High task"));
    model.enqueue(say("Suggested."));
    let high = h.agent(None, "Work on payments.", &ws);
    let high_events = h.turns_end(&high, 1);
    let task_of = |events: &[lattice_protocol::conversation::ConversationEvent]| {
        events.iter().find_map(|e| match &e.kind {
            ConversationEventKind::TaskSuggested { task, .. } => Some(task.clone()),
            _ => None,
        })
    };
    let (low_task, high_task) = (task_of(&low_events).unwrap(), task_of(&high_events).unwrap());
    let g_low = h.chat.group_add(&low, &Placement::New, None).unwrap();
    let g_high = h.chat.group_add(&high, &Placement::New, None).unwrap();
    // Older first by default; the reader raises the payments group.
    assert!(h.chat.inner.groups.ranks()[&g_low.id] < h.chat.inner.groups.ranks()[&g_high.id]);
    h.chat.group_raise(&g_high.id, true).unwrap();
    assert!(h.chat.inner.groups.ranks()[&g_high.id] < h.chat.inner.groups.ranks()[&g_low.id]);
    let board = h.chat.board(None);
    assert_eq!(board.iter().map(|c| c.group.clone()).collect::<Vec<_>>(), [g_high.id.clone(), g_low.id.clone()]);

    // The reader moves the docs group to Done: kept while its work stands.
    h.chat.group_move(&g_low.id, Some(Status::Done)).unwrap();
    let card = |id: &str| h.chat.board(None).into_iter().find(|c| c.group == id).unwrap();
    assert_eq!((card(&g_low.id).status, card(&g_low.id).moved), (Status::Done, true));

    // One slot, taken by a slow topic of its own.
    h.chat.set_task_limit(Limit::At(1)).unwrap();
    let slow = std::sync::Arc::new(
        lattice_agents::testing::ScriptedModel::new([say("Busy for a while.")]).with_delay(std::time::Duration::from_secs(3)),
    );
    h.answer_with(slow.clone());
    let busy = h.agent(None, "Something slow.", &ws);
    let start = |from: &str, task: &str| h.runtime.block_on(h.chat.start_task(from, task, "local".into(), local(), Mode::Agent));
    // The low-priority task waits first, then the high-priority one.
    let first = start(&low, &low_task).unwrap_err();
    assert!(first.message.contains("queued"), "{}", first.message);
    start(&high, &high_task).unwrap_err();
    slow.enqueue(say("High done."));
    slow.enqueue(say("Low done."));
    h.turns_end(&busy, 1);
    // The slot freed: the high-priority group's task starts first.
    let started_from = |from: &str, task: &str| {
        h.events_until(from, |events| {
            events.iter().any(|e| matches!(&e.kind, ConversationEventKind::TaskSettled { task: t, .. } if t == task))
        })
        .iter()
        .find_map(|e| match &e.kind {
            ConversationEventKind::TaskSettled { task: t, outcome: TaskOutcome::Started { conversation } } if t == task => {
                Some(conversation.clone())
            }
            _ => None,
        })
        .unwrap()
    };
    let high_topic = started_from(&high, &high_task);
    let low_topic = started_from(&low, &low_task);
    let ran = |id: &str| match h.store.list() {
        crate::chat::store::IndexState::Rows(rows) => rows.into_iter().find(|r| r.id == id).map(|r| r.created).unwrap(),
        _ => panic!("no index"),
    };
    assert!(ran(&high_topic) < ran(&low_topic), "the higher priority started first");
    h.turns_end(&low_topic, 1);
    // The docs group's work changed (its task's turns): the move lapsed.
    assert!(!card(&g_low.id).moved);
}

/// Mutants: an answer without reported tokens counted as zero tokens; the
/// time not measured from the message; a provider priced wrongly.
#[test]
fn what_answers_spent_is_measured_and_an_unreported_answer_is_counted_apart() {
    use super::board::{Priced, priced, spent_by};
    assert_eq!(priced("llamacpp:qwen.gguf"), Priced::Local);
    assert_eq!(priced("agent:claude-code"), Priced::Subscription);
    assert_eq!(priced("openrouter:some/model"), Priced::Unknown);
    let spent = spent_by(&[
        (10.0, true, "", None, None),
        (14.0, false, "llamacpp:m", Some(100), Some(20)),
        (20.0, true, "", None, None),
        (30.0, false, "agent:codex", None, None),
        (31.0, false, "endpoint:x", Some(5), Some(1)),
    ]);
    assert_eq!((spent.tokens_in, spent.tokens_out, spent.unreported), (105, 21, 1));
    assert_eq!(spent.seconds, 14.0, "4 s, then 10 s; an answer with no message before it adds none");
    assert_eq!((spent.local, spent.subscription, spent.unpriced), (1, 1, 1));
}

/// Over the harness: a group's card adds up its topics' answers and the
/// commands that may cost money the agent asked to run.
/// Mutants: the costly command not counted; a waiting one counted as approved.
#[test]
fn a_cards_cost_and_time_add_up_its_topics() {
    let h = H::new("board-spent");
    let ws = h.workspace();
    h.script(vec![
        call("run_command", json!({"command": "terraform apply -auto-approve", "timeout_s": 300}), "c1"),
        say("ok"),
    ]);
    let topic = h.agent(None, "Create the servers.", &ws);
    h.events_until(&topic, |events| events.iter().any(|e| matches!(e.kind, ConversationEventKind::ApprovalRequested { .. })));
    // The reader says no: it was asked, and not approved.
    h.runtime
        .block_on(h.chat.decide(&topic, "c1", lattice_protocol::conversation::Decision::Reject { note: None }))
        .unwrap();
    h.turns_end(&topic, 1);
    let g = h.chat.group_add(&topic, &Placement::New, None).unwrap();
    let card = h.chat.board(None).into_iter().find(|c| c.group == g.id).unwrap();
    assert_eq!((card.spent.costly_asked, card.spent.costly_approved), (1, 0));
    let answers = card.spent.local + card.spent.subscription + card.spent.unpriced;
    assert_eq!(answers, 1, "{:?}", card.spent);
    assert!(card.spent.seconds >= 0.0);
}
