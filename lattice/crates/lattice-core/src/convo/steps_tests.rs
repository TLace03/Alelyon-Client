//! Steps for the reader (`super::steps`), over the agent harness with a
//! scripted model: a checklist given and read back; the reader's checks
//! recorded, told to the agent and kept across a reopen; the last check
//! telling the agent by itself (a new turn), once; a list whose switch is off
//! waiting for the reader's Done; bad ids and steps refused.

use lattice_agents::model::ModelRequest;
use lattice_protocol::conversation::{ConversationEvent, ConversationEventKind, Mode};
use serde_json::json;

use super::agent_tests::{H, call, hosted, local, say, user_texts};
use super::steps;

fn result_of(calls: &[ModelRequest], call_id: &str) -> String {
    calls
        .iter()
        .flat_map(|request| request.input.iter())
        .find_map(|item| match item {
            lattice_agents::model::InputItem::ToolResult { call_id: id, output } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no result for {call_id}"))
}

fn given(events: &[ConversationEvent]) -> Option<(String, String, Vec<String>, bool)> {
    events.iter().find_map(|e| match &e.kind {
        ConversationEventKind::StepsGiven { list, title, steps, notify } => {
            Some((list.clone(), title.clone(), steps.clone(), *notify))
        }
        _ => None,
    })
}

fn steps_json(notify: bool) -> serde_json::Value {
    json!({
        "title": "Connect OpenRouter",
        "steps": ["Open the Models page.", "Press Sign in with OpenRouter and approve the key."],
        "notify_when_done": notify,
    })
}

#[test]
fn a_checklist_is_given_checked_and_tells_the_agent_when_done_once() {
    let h = H::new("steps-done");
    let ws = h.workspace();
    let model = h.script(vec![
        call("give_steps", steps_json(true), "g1"),
        say("Follow the checklist."),
        // The turn the last check starts.
        call("read_steps", json!({}), "r1"),
        say("Thanks: connected."),
    ]);
    let id = h.agent(None, "connect openrouter", &ws);
    let events = h.turns_end(&id, 1);
    let (list, title, steps, notify) = given(&events).expect("the checklist is shown");
    assert_eq!((title.as_str(), steps.len(), notify), ("Connect OpenRouter", 2, true));
    assert!(result_of(&model.calls(), "g1").contains("a message tells you when every step is done"));

    // The first check starts nothing.
    h.runtime.block_on(h.chat.check_step(&id, &list, 0, true, "local".into(), local())).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert_eq!(model.calls().len(), 2, "a check alone starts no turn");
    // The last one tells the agent: a new turn, with the done message.
    h.runtime.block_on(h.chat.check_step(&id, &list, 1, true, "local".into(), local())).unwrap();
    let events = h.turns_end(&id, 2);
    let calls = model.calls();
    let texts = user_texts(&calls[2]);
    assert!(texts.iter().any(|t| t == "All steps of the checklist \u{201c}Connect OpenRouter\u{201d} are done."), "{texts:?}");
    assert!(texts.iter().any(|t| t.contains("The user checked step 1")), "the checks reach the next turn as notes: {texts:?}");
    let read = result_of(&calls, "r1");
    assert!(read.contains("2 of 2 done") && read.contains("[x] Open the Models page."), "{read}");
    assert!(events.iter().any(|e| matches!(&e.kind, ConversationEventKind::StepsDone { list: l } if *l == list)));

    // Told once: unchecking and checking again sends nothing more.
    h.runtime.block_on(h.chat.check_step(&id, &list, 1, false, "local".into(), local())).unwrap();
    h.runtime.block_on(h.chat.check_step(&id, &list, 1, true, "local".into(), local())).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert_eq!(model.calls().len(), 4);
    let again = h.runtime.block_on(h.chat.steps_done(&id, &list, "local".into(), local())).unwrap_err();
    assert_eq!(again.message, steps::words::TOLD);

    // The record keeps every check: a reopen shows the list as it stands.
    let items = h.sidecar_items(&id);
    let kept = steps::lists(&items);
    assert_eq!(kept.len(), 1);
    assert!(kept[0].complete() && kept[0].told);
}

#[test]
fn with_its_switch_off_a_list_waits_for_the_readers_done() {
    let h = H::new("steps-manual");
    let ws = h.workspace();
    let model = h.script(vec![call("give_steps", steps_json(true), "g1"), say("ok"), say("noted")]);
    let id = h.agent(None, "connect", &ws);
    let events = h.turns_end(&id, 1);
    let (list, ..) = given(&events).unwrap();
    h.runtime.block_on(h.chat.set_steps_notify(&id, &list, false)).unwrap();
    for step in 0..2 {
        h.runtime.block_on(h.chat.check_step(&id, &list, step, true, "local".into(), local())).unwrap();
    }
    std::thread::sleep(std::time::Duration::from_millis(500));
    assert_eq!(model.calls().len(), 2, "switched off: no message by itself");
    h.runtime.block_on(h.chat.steps_done(&id, &list, "local".into(), local())).unwrap();
    h.turns_end(&id, 2);
    let texts = user_texts(&model.calls()[2]);
    assert!(texts.iter().any(|t| t.contains("Connect OpenRouter") && t.contains("are done")), "{texts:?}");
}

#[test]
fn bad_checklists_and_bad_checks_are_refused() {
    let h = H::new("steps-refused");
    let ws = h.workspace();
    let long = "x".repeat(steps::MAX_STEP_CHARS + 1);
    let model = h.script(vec![
        call("give_steps", json!({"title": "T", "steps": []}), "g1"),
        call("give_steps", json!({"title": "T", "steps": [long]}), "g2"),
        call("give_steps", json!({"title": "T", "steps": ["two\nlines"]}), "g3"),
        call("read_steps", json!({}), "r1"),
        call("give_steps", json!({"title": "Restart", "steps": ["Close Alelyon."]}), "g4"),
        say("done"),
    ]);
    let id = h
        .send(None, "help", "local", local(), Mode::Ask, Some(&ws))
        .map(|accepted| match accepted {
            lattice_protocol::conversation::Accepted::Started { conversation, .. } => conversation.id,
            other => panic!("{other:?}"),
        })
        .unwrap();
    let events = h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(result_of(&calls, "g1").contains("1 to 20 steps"));
    assert!(result_of(&calls, "g2").contains("one line of 1 to 200"));
    assert!(result_of(&calls, "g3").contains("one line of 1 to 200"));
    assert!(result_of(&calls, "r1").contains("no checklist"));
    let (list, ..) = given(&events).expect("Ask mode gives steps too");
    let no_step = h.runtime.block_on(h.chat.check_step(&id, &list, 5, true, "local".into(), local())).unwrap_err();
    assert_eq!(no_step.message, steps::words::NO_STEP);
    let no_list = h
        .runtime
        .block_on(h.chat.check_step(&id, "steps_ffffffffffffffff", 0, true, "local".into(), local()))
        .unwrap_err();
    assert_eq!(no_list.message, steps::words::NO_LIST);
    assert!(h.runtime.block_on(h.chat.check_step(&id, "../x", 0, true, "local".into(), local())).is_err());
}

/// The done message goes to whatever agent the chat uses now: the window's
/// choice, not the model that gave the list (here the list came from the local
/// model and the reader had since chosen a hosted one).
#[test]
fn the_done_message_goes_to_the_agent_the_chat_uses_now() {
    let h = H::new("steps-any-agent");
    let ws = h.workspace();
    let model = h.script(vec![call("give_steps", steps_json(true), "g1"), say("ok"), say("thanks")]);
    let id = h.agent(None, "connect", &ws);
    let events = h.turns_end(&id, 1);
    let (list, ..) = given(&events).unwrap();
    for step in 0..2 {
        h.runtime.block_on(h.chat.check_step(&id, &list, step, true, "endpoint:hosted".into(), hosted())).unwrap();
    }
    h.turns_end(&id, 2);
    let turns = h.store.load(&id);
    let providers: Vec<&str> = turns.iter().filter(|t| t.role == "assistant").map(|t| t.provider.as_str()).collect();
    assert_eq!(providers.len(), 2, "{providers:?}");
    assert!(providers[1].contains("hosted"), "the done message was answered by the chosen agent: {providers:?}");
    assert_ne!(providers[0], providers[1], "not the model that gave the list: {providers:?}");
    assert_eq!(model.calls().len(), 3);
}
