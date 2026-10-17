//! Schedules and watches (`super::schedules`, `Background::watch`): a
//! schedule is asked for first and kept only on a yes; at its time it runs as
//! a new topic (a task of the conversation) or in its own conversation; one
//! due while Lattice was closed is reported, not run; a watch sees a line,
//! an end, or its time out.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::{AgentChatService, ConversationEventKind, Decision};
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::schedules::{Schedule, tick, when_words};

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

#[test]
fn when_a_schedule_runs_is_said_in_words() {
    assert_eq!(when_words(30, None), "once, in 30 minutes");
    assert_eq!(when_words(60, Some(60)), "in 1 hour, then every 1 hour");
    assert_eq!(when_words(1440, Some(2880)), "in 1 day, then every 2 days");
}

/// Mutants: a schedule kept on a no; a due schedule not run; a wake-up sent
/// to a new topic; a missed run run late.
#[test]
fn a_schedule_is_asked_for_first_runs_at_its_time_and_a_missed_one_is_reported() {
    let h = H::new("schedules");
    let ws = h.workspace();
    let ask = |id: &str, args: serde_json::Value| call("schedule_prompt", args, id);
    *h.confirm.answer.lock().unwrap() = false;
    let model = h.script(vec![
        ask("s1", json!({"title": "Check CI", "prompt": "Look at the build.", "every_minutes": 60})),
        say("Not allowed."),
    ]);
    let id = h.agent(None, "Watch CI for me.", &ws);
    h.turns_end(&id, 1);
    assert!(result_of(&model.calls(), "s1").contains("did not allow"), "{}", result_of(&model.calls(), "s1"));
    assert!(h.chat.schedules().is_empty());
    assert!(h.confirm.asked().iter().any(|r| matches!(r, crate::ports::ConfirmRequest::Schedule { .. })));

    *h.confirm.answer.lock().unwrap() = true;
    model.enqueue(ask("s2", json!({"title": "Check CI", "prompt": "Look at the build.", "every_minutes": 60})));
    model.enqueue(ask("s3", json!({"title": "Wake me", "prompt": "Is it done yet?", "in_minutes": 5, "same_topic": true})));
    model.enqueue(call("list_schedules", json!({}), "l1"));
    model.enqueue(say("Scheduled."));
    h.agent(Some(&id), "Again, please.", &ws);
    h.turns_end(&id, 2);
    let calls = model.calls();
    assert!(result_of(&calls, "s2").starts_with("Scheduled as sch_"), "{}", result_of(&calls, "s2"));
    let listed = result_of(&calls, "l1");
    assert!(listed.contains("\u{201c}Check CI\u{201d} (this conversation's)") && listed.contains("as a new topic") && listed.contains(", here"), "{listed}");
    let all = h.chat.schedules();
    assert_eq!(all.len(), 2);

    // Both come due now: the repeating one starts a task; the wake-up continues this conversation.
    let now = h.chat.inner.now();
    let due = |s: &Schedule| Schedule { next: now - 1.0, ..s.clone() };
    for s in &all {
        h.chat.inner.schedules.cancel(&s.id).unwrap();
        h.chat.inner.schedules.add(due(s)).unwrap();
    }
    model.enqueue(say("The build is green."));
    model.enqueue(say("Not yet."));
    h.runtime.block_on(tick(&h.chat.inner));
    let origins = h.chat.task_origins();
    let task = origins.iter().find(|(_, o)| o.title == "Scheduled: Check CI").map(|(t, _)| t.clone()).expect("a task started");
    h.turns_end(&task, 1);
    h.turns_end(&id, 3);
    let left = h.chat.schedules();
    assert_eq!(left.len(), 1, "the wake-up ran once and went; the repeat stays");
    assert!(left[0].next > now && left[0].runs == 1, "{left:?}");

    // Due while Lattice was closed: reported in its conversation, not run, and moved on.
    let missed = Schedule { next: now - 3600.0, ..left[0].clone() };
    h.chat.inner.schedules.cancel(&missed.id).unwrap();
    h.chat.inner.schedules.add(missed).unwrap();
    *crate::convo::agent::lock(&h.chat.inner.schedules.opened) = Some(now);
    let before = h.chat.task_origins().len();
    h.runtime.block_on(tick(&h.chat.inner));
    assert_eq!(h.chat.task_origins().len(), before, "a missed run does not run late");
    let events = h.events_until(&id, |events| {
        events.iter().any(|e| matches!(&e.kind, ConversationEventKind::Notice { text } if text.contains("was due while Alelyon was closed")))
    });
    assert!(!events.is_empty());
    assert!(h.chat.schedules()[0].next > now);
    assert!(h.chat.cancel_schedule(&h.chat.schedules()[0].id).unwrap());
    assert!(h.chat.schedules().is_empty());
}

/// Mutants: a watch that never sees the line; one that waits past an end.
#[test]
fn a_watch_sees_a_line_an_end_or_its_time_out() {
    let h = H::new("watch");
    let ws = h.workspace();
    let server = "Start-Sleep -Milliseconds 800; Write-Output 'Server READY on 3000'; while ($true) { Start-Sleep -Milliseconds 200 }";
    let model = h.script(vec![
        call("run_command", json!({"command": server, "background": true}), "c1"),
        call("watch_command", json!({"id": "c1", "until": "ready", "timeout_s": 60}), "w1"),
        call("watch_command", json!({"id": "c1", "until": "never printed", "timeout_s": 2}), "w2"),
        call("stop_command", json!({"id": "c1"}), "x1"),
        call("watch_command", json!({"id": "c1", "until": "never printed", "timeout_s": 30}), "w3"),
        say("Watched."),
    ]);
    let id = h.agent(None, "Start it and wait.", &ws);
    h.events_until(&id, |events| events.iter().any(|e| matches!(e.kind, ConversationEventKind::ApprovalRequested { .. })));
    h.runtime.block_on(h.chat.decide(&id, "c1", Decision::Approve)).unwrap();
    h.turns_end(&id, 1);
    let calls = model.calls();
    let seen = result_of(&calls, "w1");
    assert!(seen.starts_with("Seen after") && seen.contains("Server READY on 3000"), "{seen}");
    assert!(result_of(&calls, "w2").starts_with("Not seen in 2 s"), "{}", result_of(&calls, "w2"));
    assert!(result_of(&calls, "w3").starts_with("It ended before writing that."), "{}", result_of(&calls, "w3"));
}
