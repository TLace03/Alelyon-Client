//! Auto mode in agent turns (as chosen on 2026-10-08): switched on
//! only through the core's own dialog, with its stop keys armed; then Agent
//! mode offers the desktop's tools, a view, an edit, a share or a deletion
//! goes ahead, and a purchase or an account change asks (the card, then the
//! native `DesktopAct` dialog); the stop keys stop every running turn; Alelyon
//! itself and a password box are refused. The desktop is a fake one: no test
//! here touches the real screen, mouse or keyboard, and the stop keys are
//! ones nobody presses.

#![cfg(windows)]

use std::sync::Arc;

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ApprovalDetail, ApprovalKind, ConversationEvent,
    ConversationEventKind, Decision, Mode, TurnStatus,
};
use serde_json::json;

use super::agent_tests::{H, call, local, say, statuses};
use super::prompt_agent::DESKTOP_NOTE;
use crate::desktop::tests::{FakeDesktop, window};
use crate::ports::ConfirmRequest;

/// Ctrl+Alt+Shift with F13 to F24: stop keys no keyboard here presses, one
/// per test (tests run at once, and a hotkey is held by one at a time).
const KEYS: u32 = lattice_sys::desktop::MOD_CONTROL
    | lattice_sys::desktop::MOD_ALT
    | lattice_sys::desktop::MOD_SHIFT;

fn harness(tag: &str, function_key: u32) -> (H, Arc<FakeDesktop>) {
    let fake = FakeDesktop::new(2560, 1440);
    let driver = fake.clone();
    let h = H::with(tag, &[], move |config| {
        config.desktop = Some(driver);
        config.stop_keys = Some((KEYS, 0x7B + function_key));
    });
    (h, fake)
}

fn switch_on(h: &H) {
    assert!(h.runtime.block_on(h.chat.auto_mode_set(true)).unwrap());
}

fn names(request: &ModelRequest) -> Vec<String> {
    request.tools.iter().map(|tool| tool.name.clone()).collect()
}

fn approval(events: &[ConversationEvent]) -> Option<(String, ApprovalDetail, bool)> {
    events.iter().find_map(|event| match &event.kind {
        ConversationEventKind::ApprovalRequested {
            call_id,
            kind: ApprovalKind::Desktop,
            detail,
            allow_always_offer,
        } => Some((call_id.clone(), detail.clone(), *allow_always_offer)),
        _ => None,
    })
}

fn outputs(events: &[ConversationEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::ToolOutput { preview, .. } => Some(preview.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn auto_mode_is_switched_on_only_in_the_dialog_with_its_stop_keys_armed() {
    let _keys = crate::testkit::stop_keys();
    let (h, _fake) = harness("auto-switch", 9);
    assert!(!h.chat.auto_mode_on());
    *h.confirm.answer.lock().unwrap() = false;
    assert!(
        !h.runtime.block_on(h.chat.auto_mode_set(true)).unwrap(),
        "a no keeps it off"
    );
    assert!(!h.chat.auto_mode_on() && !h.chat.auto_mode_armed());
    *h.confirm.answer.lock().unwrap() = true;
    switch_on(&h);
    assert!(h.chat.auto_mode_on() && h.chat.auto_mode_armed());
    assert!(h.confirm.asked().iter().any(|request| matches!(
        request,
        ConfirmRequest::AutoMode { stop_keys } if stop_keys == "Ctrl+Alt+End"
    )));
    assert!(!h.runtime.block_on(h.chat.auto_mode_set(false)).unwrap());
    assert!(!h.chat.auto_mode_on() && !h.chat.auto_mode_armed());
    // Its stop keys held by another program: it stays off.
    let held = lattice_sys::desktop::register_hotkey(KEYS, 0x7B + 9, Box::new(|| {})).unwrap();
    let refused = h.runtime.block_on(h.chat.auto_mode_set(true)).unwrap_err();
    assert!(refused.message.contains("stays off"), "{refused:?}");
    assert!(!h.chat.auto_mode_on() && !h.chat.auto_mode_armed());
    drop(held);
}

#[test]
fn in_auto_mode_the_agent_uses_the_desktop_and_money_asks_first() {
    let _keys = crate::testkit::stop_keys();
    let (h, fake) = harness("auto-turn", 10);
    let ws = h.workspace();
    // Off: no desktop tool.
    let model = h.script(vec![say("nothing")]);
    let id = h.agent(None, "Look.", &ws);
    h.turns_end(&id, 1);
    assert!(
        !names(&model.calls()[0])
            .iter()
            .any(|name| name.starts_with("desktop_"))
    );
    switch_on(&h);
    let click = |x: u32, what: &str, effect: &str, call_id: &str| {
        call(
            "desktop_click",
            json!({"x": x, "y": 360, "what": what, "effect": effect}),
            call_id,
        )
    };
    let model = h.script(vec![
        call("desktop_look", json!({}), "d1"),
        click(640, "the Send button", "share", "d2"),
        click(900, "the Place your order button", "buy", "d3"),
        say("Ordered."),
    ]);
    h.agent(Some(&id), "Send it and order the book.", &ws);
    let events = h.events_until(&id, |events| approval(events).is_some());
    let (call_id, detail, offer) = approval(&events).unwrap();
    assert_eq!(call_id, "d3", "the share went ahead without asking");
    assert!(!offer, "never allowed always");
    assert_eq!(
        detail,
        ApprovalDetail::Desktop {
            app: "notes.txt - Notepad".into(),
            action: "click at (900, 360) of the screen's picture".into(),
            what: "the Place your order button".into(),
            effect: "buy".into(),
        }
    );
    assert_eq!(
        fake.sent(),
        ["click 1280,720"],
        "only the share was sent so far"
    );
    h.runtime
        .block_on(h.chat.decide(&id, "d3", Decision::Approve))
        .unwrap();
    let events = h.turns_end(&id, 2);
    assert_eq!(statuses(&events).last(), Some(&TurnStatus::Completed));
    assert_eq!(fake.sent(), ["click 1280,720", "click 1800,720"]);
    assert!(h.confirm.asked().iter().any(|request| matches!(
        request,
        ConfirmRequest::DesktopAct { effect, .. } if effect == "a purchase or an order"
    )));
    let calls = model.calls();
    assert!(names(&calls[0]).contains(&"desktop_click".to_owned()));
    assert!(calls[0].system.ends_with(DESKTOP_NOTE));
    let pictures: Vec<&str> = calls[1]
        .input
        .iter()
        .filter_map(|item| match item {
            InputItem::UserImages { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(pictures, ["Your screen now: notes.txt - Notepad in front."]);
    let outputs = outputs(&events);
    assert!(
        outputs
            .iter()
            .any(|text| text.starts_with("notes.txt - Notepad is in front")),
        "{outputs:?}"
    );
}

#[test]
fn a_no_runs_nothing_and_the_stop_keys_stop_a_waiting_turn() {
    let _keys = crate::testkit::stop_keys();
    let (h, fake) = harness("auto-stop", 11);
    let ws = h.workspace();
    switch_on(&h);
    h.script(vec![
        call(
            "desktop_key",
            json!({"key": "Enter", "effect": "account"}),
            "k1",
        ),
        say("done"),
    ]);
    let id = h.agent(None, "Change the password.", &ws);
    h.events_until(&id, |events| approval(events).is_some());
    h.chat.press_stop_keys();
    let events = h.turns_end(&id, 1);
    assert_eq!(statuses(&events), [TurnStatus::Stopped]);
    assert!(
        fake.sent().is_empty(),
        "nothing was pressed: {:?}",
        fake.sent()
    );
}

#[test]
fn alelyon_and_a_password_box_are_refused_in_a_turn_and_ask_mode_has_no_desktop() {
    let _keys = crate::testkit::stop_keys();
    let (h, fake) = harness("auto-guard", 12);
    let ws = h.workspace();
    switch_on(&h);
    *fake.front.lock().unwrap() = Some(window(
        r"D:\src\centcom\target\release\centcom.exe",
        "Alelyon",
        11,
    ));
    h.script(vec![
        call("desktop_type", json!({"text": "hello"}), "t1"),
        say("I could not."),
    ]);
    let id = h.agent(None, "Type hello.", &ws);
    let events = h.turns_end(&id, 1);
    assert!(
        outputs(&events)[0].contains("Alelyon itself"),
        "{:?}",
        outputs(&events)
    );
    *fake.front.lock().unwrap() = Some(window(r"C:\Windows\System32\notepad.exe", "Sign in", 7));
    *fake.password.lock().unwrap() = true;
    h.script(vec![
        call("desktop_type", json!({"text": "hunter2"}), "t2"),
        say("I will not."),
    ]);
    h.agent(Some(&id), "Type my password.", &ws);
    let events = h.turns_end(&id, 2);
    assert!(
        outputs(&events)
            .iter()
            .any(|text| text.contains("never types a password"))
    );
    assert!(fake.sent().is_empty(), "{:?}", fake.sent());
    // Ask mode offers no desktop tool, though auto mode is on.
    let model = h.script(vec![say("ok")]);
    let accepted = h
        .send(None, "Look.", "local", local(), Mode::Ask, Some(&ws))
        .unwrap();
    let Accepted::Started { conversation, .. } = accepted else {
        panic!("{accepted:?}")
    };
    h.turns_end(&conversation.id, 1);
    assert!(
        !names(&model.calls()[0])
            .iter()
            .any(|name| name.starts_with("desktop_"))
    );
}
