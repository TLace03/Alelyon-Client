//! An agent turn with the agent's browser: it is offered only once the reader
//! has switched it on, in Agent mode; a click stated as `share` asks (the
//! card, then the native `BrowserAct` dialog) and a no runs nothing; a click
//! stated as `view` on a Post button is refused with the effect it looks
//! like; every action's screenshot reaches the next model call right after
//! its result, the last three as images; and a model that refuses a
//! screenshot is not offered the browser again in this process. The browser
//! is a real headless Edge or Chrome on a page the test serves on loopback.

#![cfg(windows)]

use lattice_agents::model::{InputItem, ModelError, ModelRequest};
use lattice_agents::testing::ScriptedStep;
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ApprovalDetail, ApprovalKind, ConversationEvent,
    ConversationEventKind, Decision, Mode, TurnStatus,
};
use serde_json::json;

use super::agent::words;
use super::agent_tests::{H, call, local, say, statuses};
use super::prompt_agent::BROWSER_NOTE;
use crate::browser::{BrowserConfig, BrowserStatus, launch};
use crate::env::ProcessEnv;
use crate::ports::ConfirmRequest;

/// A chat whose browser starts headless on a scratch profile and may open
/// the loopback page the test serves; `None` (the test skipped) where no
/// Edge or Chrome is installed.
///
/// The browser gets the system's folders from this process (not the
/// application-data folders: the scratch folder is inside one, and a folder
/// there is never attached). Its home is the test's own (`USERPROFILE`),
/// given the `AppData\Local` and `AppData\Roaming` folders every user's home
/// has: Windows finds a user's application-data folders there. Without them
/// Edge, now and then, cannot find its default profile folder, takes the
/// test's profile for it, and never answers on its DevTools pipe ("DevTools
/// remote debugging requires a non-default data directory"). Measured on
/// 2026-10-08, four starts at a time under a release build's load: 9 of 120
/// starts hung without the folders, none of 80 with them.
pub(super) fn browser_harness(tag: &str) -> Option<H> {
    let Some(program) = launch::find_browser(&ProcessEnv) else {
        eprintln!("no Edge or Chrome installed: skipped");
        return None;
    };
    let real: Vec<(String, String)> = launch::BROWSER_ENV_NAMES
        .iter()
        .filter_map(|name| {
            std::env::var(name)
                .ok()
                .map(|value| ((*name).to_owned(), value))
        })
        .collect();
    let extra: Vec<(&str, &str)> = real
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()))
        .collect();
    Some(H::with(tag, &extra, |config| {
        let home = config
            .env
            .var("USERPROFILE")
            .map(std::path::PathBuf::from)
            .expect("the test's home");
        for folder in ["Local", "Roaming"] {
            std::fs::create_dir_all(home.join("AppData").join(folder)).unwrap();
        }
        config.browser = BrowserConfig {
            headless: true,
            allow_local: true,
            profile: Some(config.state.globals.join("browser-test-profile")),
            program: Some(program),
            start_wait: crate::testkit::BROWSER_START_WAIT,
            load_wait: crate::testkit::BROWSER_LOAD_WAIT,
        };
    }))
}

/// The turn's first browser action, its `browser_open`, reached the test's
/// page: checked before anything that depends on the page being open.
pub(super) fn opened_the_page(events: &[ConversationEvent]) {
    let outputs = outputs(events);
    assert!(
        outputs
            .first()
            .is_some_and(|output| output.starts_with("The page is Lattice browser test")),
        "the browser did not open the test's page: {outputs:?}"
    );
}

/// Every browser tool's output in `events` saw a page: an action whose
/// browser did not start or whose look failed fails here, by its own words,
/// rather than as a screenshot missing later.
fn every_look_saw_the_page(events: &[ConversationEvent]) {
    let outputs = outputs(events);
    assert!(!outputs.is_empty(), "no browser action ran");
    for output in &outputs {
        assert!(
            output.starts_with("The page is "),
            "a browser action did not see the page: {output:?} (all: {outputs:?})"
        );
    }
}

fn page() -> String {
    format!("http://127.0.0.1:{}/", crate::browser::tests::serve())
}

fn approval(events: &[ConversationEvent]) -> Option<(String, ApprovalDetail, bool)> {
    events.iter().find_map(|event| match &event.kind {
        ConversationEventKind::ApprovalRequested {
            call_id,
            kind: ApprovalKind::Browser,
            detail,
            allow_always_offer,
        } => Some((call_id.clone(), detail.clone(), *allow_always_offer)),
        _ => None,
    })
}

fn approvals(events: &[ConversationEvent]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event.kind, ConversationEventKind::ApprovalRequested { .. }))
        .count()
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

fn offers_browser(request: &ModelRequest) -> bool {
    request
        .tools
        .iter()
        .any(|tool| tool.name.starts_with("browser_"))
}

fn images(input: &[InputItem]) -> usize {
    input
        .iter()
        .filter(|item| matches!(item, InputItem::UserImages { .. }))
        .count()
}

#[test]
fn the_browser_is_offered_once_switched_on_and_a_share_asks_on_the_card_then_in_the_dialog() {
    let _one = crate::testkit::one_real_browser();
    let Some(h) = browser_harness("browser-turn") else {
        return;
    };
    let workspace = h.workspace();
    // Off: no browser tool and no browser note.
    let model = h.script(vec![say("Nothing to do.")]);
    let id = h.agent(None, "Look around.", &workspace);
    h.turns_end(&id, 1);
    assert!(!h.chat.browser_on());
    assert!(!offers_browser(&model.calls()[0]));
    assert!(!model.calls()[0].system.contains(BROWSER_NOTE));
    assert_eq!(
        h.chat.browser().status(),
        BrowserStatus::Stopped,
        "nothing started it"
    );
    // On: open, a misstated click, then the click stated as share.
    h.runtime.block_on(h.chat.browser_set_on(true)).unwrap();
    assert!(h.chat.browser_on());
    let click = |effect: &str, call_id: &str| {
        call(
            "browser_click",
            json!({"x": 450, "y": 125, "what": "the Post button", "effect": effect}),
            call_id,
        )
    };
    let url = page();
    let model = h.script(vec![
        call("browser_open", json!({"url": url}), "b1"),
        click("view", "b2"),
        click("share", "b3"),
        say("Posted."),
    ]);
    h.agent(Some(&id), "Post it.", &workspace);
    let events = h.events_until(&id, |events| approval(events).is_some());
    opened_the_page(&events);
    let (call_id, detail, offer) = approval(&events).unwrap();
    assert_eq!(call_id, "b3", "the misstated click asked nothing");
    assert!(!offer, "a browser action is never allowed always");
    assert_eq!(
        detail,
        ApprovalDetail::Browser {
            site: "127.0.0.1".into(),
            action: "click at (450, 125)".into(),
            what: "the Post button".into(),
            effect: "share".into(),
        }
    );
    h.runtime
        .block_on(h.chat.decide(&id, "b3", Decision::Approve))
        .unwrap();
    let events = h.turns_end(&id, 2);
    assert_eq!(approvals(&events), 1);
    // Each step's card names what it opened or clicked, and the effect stated.
    let cards: Vec<(String, Option<String>)> = events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::ToolCall {
                summary, target, ..
            } => Some((summary.clone(), target.clone())),
            _ => None,
        })
        .collect();
    let post = |effect: &str| {
        (
            format!("browser_click the Post button ({effect})"),
            Some(format!("the Post button ({effect})")),
        )
    };
    assert_eq!(
        cards,
        [
            (format!("browser_open {url}"), Some(url.clone())),
            post("view"),
            post("share")
        ]
    );
    let outputs = outputs(&events);
    assert!(
        outputs[0].starts_with("The page is Lattice browser test"),
        "{outputs:?}"
    );
    assert!(
        outputs[1].contains("(\"Post\")") && outputs[1].contains("as share"),
        "{outputs:?}"
    );
    assert!(outputs[2].starts_with("The page is posted"), "{outputs:?}");
    let asked = h.confirm.asked();
    assert!(
        asked.iter().any(|request| matches!(
            request,
            ConfirmRequest::BrowserAct { site, what, effect, .. }
                if site == "127.0.0.1"
                    && what == "the Post button"
                    && effect == "something another person will see"
        )),
        "{asked:?}"
    );
    let calls = model.calls();
    assert_eq!(calls.len(), 4);
    assert!(offers_browser(&calls[0]));
    assert!(calls[0].system.ends_with(BROWSER_NOTE));
    // Each screenshot comes right after the result of the action that took
    // it; a refused action takes none.
    let counts: Vec<usize> = calls.iter().map(|request| images(&request.input)).collect();
    assert_eq!(counts, [0, 1, 1, 2]);
    let at = calls[1]
        .input
        .iter()
        .position(|item| matches!(item, InputItem::UserImages { .. }))
        .unwrap();
    assert!(
        matches!(&calls[1].input[at - 1], InputItem::ToolResult { call_id, .. } if call_id == "b1"),
        "{:?}",
        calls[1].input[at - 1]
    );
    let InputItem::UserImages { text, images } = &calls[1].input[at] else {
        unreachable!()
    };
    assert!(text.contains("Lattice browser test"), "{text}");
    assert!(images[0].base64.starts_with("iVBORw0KGgo"), "a PNG");
    // No screenshot is kept: not in the sidecar, not in the run's trace.
    let kept = super::agent_tests::files(&h.state.native_chat_dir());
    assert!(
        kept.keys().any(|path| path.contains("runs")),
        "the run was recorded: {:?}",
        kept.keys()
    );
    for (path, bytes) in kept {
        let text = String::from_utf8_lossy(&bytes);
        assert!(!text.contains("iVBORw0KGgo"), "{path} keeps a screenshot");
    }
}

#[test]
fn a_no_on_the_card_or_in_the_dialog_runs_nothing_and_ask_mode_offers_no_browser() {
    let _one = crate::testkit::one_real_browser();
    let Some(h) = browser_harness("browser-no") else {
        return;
    };
    let workspace = h.workspace();
    h.runtime.block_on(h.chat.browser_set_on(true)).unwrap();
    let click = |call_id: &str| {
        call(
            "browser_click",
            json!({"x": 450, "y": 125, "what": "the Post button", "effect": "share"}),
            call_id,
        )
    };
    h.script(vec![
        call("browser_open", json!({"url": page()}), "b1"),
        click("b2"),
        click("b3"),
        say("I did not post it."),
    ]);
    let id = h.agent(None, "Post it.", &workspace);
    // The card's Reject.
    let events = h.events_until(&id, |events| approval(events).is_some());
    opened_the_page(&events);
    h.runtime
        .block_on(h.chat.decide(&id, "b2", Decision::Reject { note: None }))
        .unwrap();
    // The card's Approve, then the dialog's no.
    *h.confirm.answer.lock().unwrap() = false;
    h.events_until(&id, |events| approvals(events) == 2);
    h.runtime
        .block_on(h.chat.decide(&id, "b3", Decision::Approve))
        .unwrap();
    let events = h.turns_end(&id, 1);
    let outputs = outputs(&events);
    assert_eq!(outputs.len(), 3, "{outputs:?}");
    assert!(
        outputs[1].starts_with("Tool execution was not approved."),
        "{outputs:?}"
    );
    assert!(
        outputs[2].starts_with("Tool execution was not approved."),
        "{outputs:?}"
    );
    match h.chat.browser().status() {
        BrowserStatus::Running { title, .. } => assert_eq!(title, "Lattice browser test"),
        other => panic!("{other:?}"),
    }
    // Ask mode: no browser, though it is on.
    let model = h.script(vec![say("Nothing to do.")]);
    let accepted = h
        .send(
            None,
            "Look around.",
            "local",
            local(),
            Mode::Ask,
            Some(&workspace),
        )
        .unwrap();
    let Accepted::Started { conversation, .. } = accepted else {
        panic!("{accepted:?}")
    };
    h.turns_end(&conversation.id, 1);
    assert!(!offers_browser(&model.calls()[0]));
    // Off stops it.
    h.runtime.block_on(h.chat.browser_set_on(false)).unwrap();
    assert_eq!(h.chat.browser().status(), BrowserStatus::Stopped);
}

#[test]
fn only_the_last_three_screenshots_are_shown_and_a_model_that_refuses_one_loses_the_browser() {
    let _one = crate::testkit::one_real_browser();
    let Some(h) = browser_harness("browser-shots") else {
        return;
    };
    let workspace = h.workspace();
    h.runtime.block_on(h.chat.browser_set_on(true)).unwrap();
    let model = h.script(vec![
        call("browser_open", json!({"url": page()}), "b1"),
        call("browser_look", json!({}), "b2"),
        call("browser_look", json!({}), "b3"),
        call("browser_look", json!({}), "b4"),
        say("Seen."),
    ]);
    let id = h.agent(None, "Look four times.", &workspace);
    let events = h.turns_end(&id, 1);
    every_look_saw_the_page(&events);
    assert_eq!(outputs(&events).len(), 4, "an open and three looks");
    let last = model.calls().pop().unwrap();
    // What each browser call answered: a missing screenshot is an answer
    // without a picture, and its words say why.
    let answered: Vec<&str> = last
        .input
        .iter()
        .filter_map(|item| match item {
            InputItem::ToolResult { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        images(&last.input),
        3,
        "the browser calls answered: {answered:#?}"
    );
    let named = last
        .input
        .iter()
        .filter(|item| matches!(item, InputItem::User(text) if text.contains("is not shown again")))
        .count();
    assert_eq!(named, 1, "the oldest is named, not shown");
    // A model that refuses the screenshot.
    h.script(vec![
        call("browser_look", json!({}), "c1"),
        ScriptedStep::error(ModelError::Status(400)),
    ]);
    h.agent(Some(&id), "Look again.", &workspace);
    let events = h.turns_end(&id, 2);
    every_look_saw_the_page(&events);
    assert_eq!(statuses(&events).last(), Some(&TurnStatus::Failed));
    assert!(
        events.iter().any(|event| matches!(
            &event.kind,
            ConversationEventKind::Error { message } if message == words::NO_VISION
        )),
        "{events:#?}"
    );
    let model = h.script(vec![say("Fine.")]);
    h.agent(Some(&id), "And now?", &workspace);
    h.turns_end(&id, 3);
    assert!(
        !offers_browser(&model.calls()[0]),
        "not offered to it again"
    );
}

/// `browser_read` gives the page 20,000 characters at a time and says where
/// the rest starts; `web_search` numbers its results.
#[test]
fn page_text_comes_in_parts_and_results_as_a_list() {
    use crate::browser::session::{Hit, PageText};
    use crate::convo::turn::{PAGE_TEXT_CHARS, page_text_result, search_result};

    let page = PageText {
        url: "https://example.com/".into(),
        title: "Example".into(),
        text: "é".repeat(PAGE_TEXT_CHARS + 5),
    };
    let first = page_text_result(&page, 0);
    assert!(first.starts_with("Address: https://example.com/\nTitle: Example\nCharacters 0 to 20000 of 20005:"));
    assert!(first.ends_with("[More: browser_read with from 20000.]"));
    let rest = page_text_result(&page, 20_000);
    assert!(rest.contains("Characters 20000 to 20005 of 20005:\n\néééé"));
    assert!(!rest.contains("[More"));
    assert!(page_text_result(&page, 99_999).contains("Characters 20005 to 20005"));
    let empty = PageText { text: String::new(), ..page };
    assert!(page_text_result(&empty, 0).ends_with("The page shows no text."));

    let hits = vec![
        Hit { title: "Learn Rust".into(), url: "https://www.rust-lang.org/learn".into(), snippet: "The book.".into() },
        Hit { title: "std".into(), url: "https://doc.rust-lang.org/std/".into(), snippet: String::new() },
    ];
    assert_eq!(
        search_result("rust", &hits),
        "Results for \"rust\" (Bing):\n1. Learn Rust\n   https://www.rust-lang.org/learn\n   The book.\n2. std\n   https://doc.rust-lang.org/std/"
    );
    assert!(search_result("zzz", &[]).starts_with("No results came back for \"zzz\"."));
}
