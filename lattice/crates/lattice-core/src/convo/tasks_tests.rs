//! The tasks the agent suggests (`super::tasks`), end to end over the agent
//! harness: a chip is recorded and shown, starts a new conversation in the
//! same folder with its recorded prompt, settles once, and survives being
//! read again; a dismissed or withdrawn task stays so; bad suggestions and
//! more than eight waiting are refused with a sentence the model reads.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::RefusalKind;
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ConversationEvent, ConversationEventKind, Mode, TaskOutcome,
    is_task_id,
};
use serde_json::json;

use super::agent_tests::{H, call, local, say};
use super::item::Item;
use super::tasks::{self, words};

const PROMPT: &str = "In README.md, the build badge points at .github/workflows/ci.yml, which is now build.yml. Point the badge at build.yml.";

fn suggest(id: &str) -> lattice_agents::testing::ScriptedStep {
    call(
        "suggest_task",
        json!({
            "title": "Fix the stale README badge",
            "summary": "The README's build badge still names the old workflow.",
            "prompt": PROMPT,
        }),
        id,
    )
}

/// The task the events suggested last: its id.
fn suggested(events: &[ConversationEvent]) -> String {
    events
        .iter()
        .rev()
        .find_map(|event| match &event.kind {
            ConversationEventKind::TaskSuggested { task, .. } => Some(task.clone()),
            _ => None,
        })
        .expect("a task was suggested")
}

fn settled(events: &[ConversationEvent], task: &str) -> Vec<TaskOutcome> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::TaskSettled { task: t, outcome } if t == task => {
                Some(outcome.clone())
            }
            _ => None,
        })
        .collect()
}

/// What the model read for its call `call_id`.
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

/// A chip is recorded and shown with its text; the model hears its id;
/// Start opens a new conversation in the same folder with the recorded
/// prompt as its first message, and settles the task once, so a second
/// Start or a Dismiss is refused.
/// Mutants: Start sending no workspace; the task left waiting after Start.
#[test]
fn a_suggested_task_starts_a_new_chat_in_the_same_folder_once() {
    let h = H::new("tasks-start");
    let ws = h.workspace();
    let model = h.script(vec![
        suggest("t1"),
        say("I suggested a task for the badge."),
    ]);
    let id = h.agent(None, "Fix the build.", &ws);
    let events = h.turns_end(&id, 1);
    let task = suggested(&events);
    assert!(is_task_id(&task), "{task}");
    let shown = events
        .iter()
        .find_map(|event| match &event.kind {
            ConversationEventKind::TaskSuggested {
                title,
                summary,
                prompt,
                ..
            } => Some((title.clone(), summary.clone(), prompt.clone())),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        shown,
        (
            "Fix the stale README badge".to_owned(),
            "The README's build badge still names the old workflow.".to_owned(),
            PROMPT.to_owned()
        )
    );
    let heard = result_of(&model.calls(), "t1");
    assert!(
        heard.contains(&task) && heard.contains("Nothing runs"),
        "{heard}"
    );
    assert!(h.sidecar_items(&id).iter().any(|item| matches!(
        item,
        Item::TaskSuggested { task: t, prompt, call_id, .. } if *t == task && prompt == PROMPT && call_id == "t1"
    )));
    assert!(settled(&events, &task).is_empty(), "it waits");

    // Start: a new chat, in the same folder, that begins with the prompt.
    model.enqueue(say("The badge now points at build.yml."));
    let accepted = h
        .runtime
        .block_on(
            h.chat
                .start_task(&id, &task, "local".into(), local(), Mode::Agent),
        )
        .unwrap();
    let Accepted::Started {
        conversation,
        user_turn,
        ..
    } = accepted
    else {
        panic!("{accepted:?}")
    };
    assert_ne!(conversation.id, id, "a conversation of its own");
    assert_eq!(user_turn.text, PROMPT);
    assert_eq!(
        conversation.workspace.as_ref().map(|w| w.id.as_str()),
        Some(ws.as_str()),
        "the same folder"
    );
    h.turns_end(&conversation.id, 1);
    assert_eq!(h.texts(&conversation.id)[0], PROMPT);
    let new_mode = h
        .runtime
        .block_on(h.chat.list())
        .unwrap()
        .conversations
        .into_iter()
        .find(|c| c.id == conversation.id)
        .unwrap()
        .mode;
    assert_eq!(new_mode, Mode::Agent);

    // Settled once, as an event and a record.
    let started = TaskOutcome::Started {
        conversation: conversation.id.clone(),
    };
    let events = h.events_until(&id, |events| !settled(events, &task).is_empty());
    assert_eq!(settled(&events, &task), [started.clone()]);
    assert!(h.sidecar_items(&id).iter().any(|item| matches!(
        item,
        Item::TaskSettled { task: t, outcome, .. } if *t == task && *outcome == started
    )));
    let again = h
        .runtime
        .block_on(
            h.chat
                .start_task(&id, &task, "local".into(), local(), Mode::Agent),
        )
        .unwrap_err();
    assert_eq!(
        (again.kind, again.message.as_str()),
        (RefusalKind::Conflict, words::STARTED)
    );
    let dismissed = h
        .runtime
        .block_on(h.chat.dismiss_task(&id, &task))
        .unwrap_err();
    assert_eq!(dismissed.message, words::STARTED);
    let listed = h
        .runtime
        .block_on(h.chat.list())
        .unwrap()
        .conversations
        .len();
    assert_eq!(listed, 2, "no third conversation");
}

/// Dismiss settles the task: it cannot be started, and the agent that tries
/// to withdraw it hears that the user dismissed it.
/// Mutant: Start not checking the record (a dismissed task starts).
#[test]
fn a_dismissed_task_stays_dismissed_and_the_agent_hears_so() {
    let h = H::new("tasks-dismiss");
    let ws = h.workspace();
    let model = h.script(vec![suggest("t1"), say("Suggested.")]);
    let id = h.agent(None, "Fix the build.", &ws);
    let task = suggested(&h.turns_end(&id, 1));
    h.runtime.block_on(h.chat.dismiss_task(&id, &task)).unwrap();
    let refused = h
        .runtime
        .block_on(
            h.chat
                .start_task(&id, &task, "local".into(), local(), Mode::Agent),
        )
        .unwrap_err();
    assert_eq!(refused.message, words::DISMISSED);
    model.enqueue(call("withdraw_task", json!({"task_id": task}), "w1"));
    model.enqueue(say("It was dismissed."));
    h.agent(Some(&id), "Withdraw the badge task.", &ws);
    let events = h.turns_end(&id, 2);
    assert_eq!(settled(&events, &task), [TaskOutcome::Dismissed]);
    let heard = result_of(&model.calls(), "w1");
    assert!(heard.contains("The user dismissed"), "{heard}");
    // Unknown and malformed ids.
    for (bad, kind) in [
        ("task_0000000000000000", RefusalKind::NotFound),
        ("../../etc", RefusalKind::NotFound),
    ] {
        let refused = h
            .runtime
            .block_on(h.chat.dismiss_task(&id, bad))
            .unwrap_err();
        assert_eq!(
            (refused.kind, refused.message.as_str()),
            (kind, words::NO_TASK)
        );
    }
}

/// The agent withdraws a task it no longer needs, saying why; a withdrawn
/// task neither starts nor withdraws again; a malformed id is refused.
#[test]
fn the_agent_withdraws_a_task_with_its_reason() {
    let h = H::new("tasks-withdraw");
    let ws = h.workspace();
    let model = h.script(vec![suggest("t1"), say("Suggested.")]);
    let id = h.agent(None, "Fix the build.", &ws);
    let task = suggested(&h.turns_end(&id, 1));
    model.enqueue(call(
        "withdraw_task",
        json!({"task_id": task, "reason": "Fixed here after all."}),
        "w1",
    ));
    model.enqueue(call("withdraw_task", json!({"task_id": task}), "w2"));
    model.enqueue(call("withdraw_task", json!({"task_id": "badge"}), "w3"));
    model.enqueue(say("Withdrawn."));
    h.agent(Some(&id), "Fix the badge here instead.", &ws);
    let events = h.turns_end(&id, 2);
    let withdrawn = TaskOutcome::Withdrawn {
        reason: "Fixed here after all.".into(),
    };
    assert_eq!(settled(&events, &task), [withdrawn]);
    let calls = model.calls();
    assert_eq!(result_of(&calls, "w1"), format!("Withdrew {task}."));
    assert!(result_of(&calls, "w2").contains("withdrawn already"));
    assert!(result_of(&calls, "w3").contains("not a task's id"));
    let refused = h
        .runtime
        .block_on(
            h.chat
                .start_task(&id, &task, "local".into(), local(), Mode::Agent),
        )
        .unwrap_err();
    assert_eq!(refused.message, words::WITHDRAWN);
}

/// A title of two lines, a blank prompt, a prompt past 2,000 characters and
/// a ninth waiting task are refused, each with its sentence, and none is
/// shown; a secret in a prompt is redacted before the chip shows it.
/// Mutant: no limit on the tasks that wait.
#[test]
fn bad_suggestions_and_a_ninth_waiting_task_are_refused() {
    let h = H::new("tasks-bounds");
    let ws = h.workspace();
    let mut steps = vec![
        call(
            "suggest_task",
            json!({"title": "Two\nlines", "summary": "s", "prompt": "p"}),
            "b1",
        ),
        call(
            "suggest_task",
            json!({"title": "Blank", "summary": "s", "prompt": "  \n "}),
            "b2",
        ),
        call(
            "suggest_task",
            json!({"title": "Long", "summary": "s", "prompt": "x".repeat(2_001)}),
            "b3",
        ),
        call(
            "suggest_task",
            json!({"title": "Key", "summary": "s", "prompt": format!("Use {} to call it.", super::agent_tests::secret())}),
            "k1",
        ),
    ];
    for n in 2..=8 {
        steps.push(call(
            "suggest_task",
            json!({"title": format!("Task {n}"), "summary": "s", "prompt": "p"}),
            &format!("s{n}"),
        ));
    }
    steps.push(call(
        "suggest_task",
        json!({"title": "Ninth", "summary": "s", "prompt": "p"}),
        "s9",
    ));
    steps.push(say("Done."));
    let model = h.script(steps);
    let id = h.agent(None, "Suggest things.", &ws);
    let events = h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(result_of(&calls, "b1").contains("one line of 1 to 80 characters"));
    assert!(result_of(&calls, "b2").contains("1 to 2,000 characters"));
    assert!(result_of(&calls, "b3").contains("1 to 2,000 characters"));
    assert!(result_of(&calls, "k1").contains(crate::secrets::REDACTED));
    assert!(result_of(&calls, "s9").contains("8 suggested tasks already wait"));
    let shown: Vec<(String, String)> = events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::TaskSuggested { title, prompt, .. } => {
                Some((title.clone(), prompt.clone()))
            }
            _ => None,
        })
        .collect();
    let titles: Vec<&str> = shown.iter().map(|(title, _)| title.as_str()).collect();
    assert_eq!(
        titles,
        [
            "Key", "Task 2", "Task 3", "Task 4", "Task 5", "Task 6", "Task 7", "Task 8"
        ]
    );
    let secret = super::agent_tests::secret();
    assert!(!shown[0].1.contains(&secret), "{}", shown[0].1);
    assert!(
        !h.sidecar_items(&id)
            .iter()
            .any(|item| serde_json::to_string(item).unwrap().contains(&secret))
    );
    // One settles, and a ninth may wait.
    let first = suggested(&events[..events
        .iter()
        .position(|e| matches!(&e.kind, ConversationEventKind::TaskSuggested { title, .. } if title == "Task 2"))
        .unwrap()]);
    h.runtime
        .block_on(h.chat.dismiss_task(&id, &first))
        .unwrap();
    model.enqueue(call(
        "suggest_task",
        json!({"title": "Ninth", "summary": "s", "prompt": "p"}),
        "s10",
    ));
    model.enqueue(say("Done."));
    h.agent(Some(&id), "One more.", &ws);
    h.turns_end(&id, 2);
    assert!(result_of(&model.calls(), "s10").starts_with("Suggested as task_"));
}

/// A conversation read again from its record shows its chips as they were
/// left: the records become the same events.
/// Mutant: the records not read back as events.
#[test]
fn a_reopened_conversation_shows_its_tasks_as_they_were_left() {
    let h = H::new("tasks-seed");
    let convo = h.chat.inner.convo("0123456789ab");
    let items = [
        Item::TaskSuggested {
            turn: "a1b2c3d4e5f6".into(),
            call_id: "t1".into(),
            task: "task_0011223344556677".into(),
            title: "Fix the stale README badge".into(),
            summary: "s".into(),
            prompt: PROMPT.into(),
            at: 1.0,
        },
        Item::TaskSettled {
            task: "task_0011223344556677".into(),
            outcome: TaskOutcome::Dismissed,
            at: 2.0,
        },
    ];
    super::views::seed(&convo, &items);
    let (events, _) = convo.log.recorded();
    let kinds: Vec<ConversationEventKind> = events.into_iter().map(|event| event.kind).collect();
    assert_eq!(
        kinds,
        [
            ConversationEventKind::TaskSuggested {
                task: "task_0011223344556677".into(),
                title: "Fix the stale README badge".into(),
                summary: "s".into(),
                prompt: PROMPT.into(),
            },
            ConversationEventKind::TaskSettled {
                task: "task_0011223344556677".into(),
                outcome: TaskOutcome::Dismissed,
            },
        ]
    );
    // The first settlement counts; one for a task never suggested is ignored.
    let later = Item::TaskSettled {
        task: "task_0011223344556677".into(),
        outcome: TaskOutcome::Started {
            conversation: "0a0b0c0d0e0f".into(),
        },
        at: 3.0,
    };
    let stray = Item::TaskSettled {
        task: "task_ffffffffffffffff".into(),
        outcome: TaskOutcome::Dismissed,
        at: 4.0,
    };
    let all = [items[0].clone(), items[1].clone(), later, stray];
    let known = tasks::tasks(&all);
    assert_eq!(known.len(), 1);
    assert_eq!(known[0].outcome, Some(TaskOutcome::Dismissed));
}
