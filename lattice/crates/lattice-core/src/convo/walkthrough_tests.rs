//! Walkthroughs (`super::walkthrough`), end to end over the agent harness in
//! auto mode with a fake desktop: the turn's pictures, in the order they were
//! shown, become an HTML artifact with their captions and the agent's notes;
//! its words are redacted and its pictures embedded as they were shown; a
//! turn with no picture, or a picture it does not have, is refused.

use std::sync::Arc;

use lattice_agents::model::{Image, InputItem, ModelRequest};
use lattice_protocol::conversation::ArtifactKind;
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::walkthrough::{Shown, page};
use crate::desktop::tests::FakeDesktop;

fn harness(tag: &str, function_key: u32) -> (H, Arc<FakeDesktop>) {
    let fake = FakeDesktop::new(1280, 800);
    let driver = fake.clone();
    let h = H::with(tag, &[], move |config| {
        config.desktop = Some(driver);
        config.stop_keys = Some((
            lattice_sys::desktop::MOD_CONTROL | lattice_sys::desktop::MOD_ALT | lattice_sys::desktop::MOD_SHIFT,
            0x7B + function_key,
        ));
    });
    (h, fake)
}

fn result_of(calls: &[ModelRequest], call_id: &str) -> String {
    calls
        .iter()
        .flat_map(|request| request.input.iter())
        .find_map(|item| match item {
            InputItem::ToolResult { call_id: id, output } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no result for {call_id}"))
}

#[test]
fn the_turns_pictures_become_a_walkthrough_with_captions_and_notes() {
    let _keys = crate::testkit::stop_keys();
    let (h, _fake) = harness("walkthrough", 12);
    let ws = h.workspace();
    assert!(h.runtime.block_on(h.chat.auto_mode_set(true)).unwrap());
    // Built here, so no key-shaped text sits in the source.
    let key = ["sk", "ant", "api03", &"A".repeat(36)].join("-");
    let model = h.script(vec![
        call("write_walkthrough", json!({"name": "walk", "title": "Nothing yet", "summary": "None."}), "w0"),
        call("desktop_look", json!({}), "d1"),
        call("desktop_click", json!({"x": 10, "y": 10, "what": "the menu", "effect": "view"}), "d2"),
        call("write_walkthrough", json!({"name": "walk", "title": "T", "summary": "S", "pictures": [3]}), "w1"),
        call(
            "write_walkthrough",
            json!({
                "name": "walk",
                "title": "Opening the menu",
                "summary": format!("WALK-SUMMARY with {key}"),
                "notes": [{"picture": 2, "note": "The menu <opened>."}],
            }),
            "w2",
        ),
        say("Saved the walkthrough."),
    ]);
    let id = h.agent(None, "Open the menu and show me how.", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(result_of(&calls, "w0").contains("no pictures"), "{}", result_of(&calls, "w0"));
    assert!(result_of(&calls, "w1").contains("There is no picture 3"), "{}", result_of(&calls, "w1"));
    assert!(result_of(&calls, "w2").contains("It shows 2 pictures of this turn."), "{}", result_of(&calls, "w2"));
    let view = h.runtime.block_on(h.chat.artifact(&id, "walk", None)).unwrap();
    assert_eq!(view.kind, ArtifactKind::Html);
    assert_eq!(view.title, "Opening the menu");
    let html = view.text;
    assert!(html.contains("WALK-SUMMARY") && !html.contains(&key), "its words are redacted");
    assert!(html.contains("The menu &lt;opened&gt;."), "and escaped");
    assert_eq!(html.matches("<figure>").count(), 2);
    assert!(html.find("<figcaption>1. ").unwrap() < html.find("<figcaption>2. ").unwrap(), "in the order shown");
    assert!(html.contains("<figcaption>1. Your screen now: notes.txt - Notepad in front."), "{html}");
    // Each picture as it was shown to the model.
    let shown: Vec<String> = calls
        .iter()
        .flat_map(|request| request.input.iter())
        .filter_map(|item| match item {
            InputItem::UserImages { images, .. } => Some(images[0].base64.clone()),
            _ => None,
        })
        .collect();
    assert!(!shown.is_empty());
    assert!(html.contains(&format!("data:image/png;base64,{}", shown[0])));
}

#[test]
fn a_page_shows_only_images_a_page_can_show() {
    let png = Shown {
        caption: "a <b>".into(),
        image: Image { media_type: "image/png".into(), base64: "QUJD".into() },
    };
    let other = Shown {
        caption: "odd".into(),
        image: Image { media_type: "text/html".into(), base64: "PHNjcmlwdD4=".into() },
    };
    let html = page("T & T", "One.\n\nTwo.", &[(1, &png, Some("n")), (2, &other, None)]);
    assert!(html.contains("<h1>T &amp; T</h1>"));
    assert!(html.contains("<p>One.</p>\n<p>Two.</p>"));
    assert!(html.contains("src=\"data:image/png;base64,QUJD\""));
    assert!(html.contains("1. a &lt;b&gt;<span class=\"note\">n</span>"));
    assert!(!html.contains("PHNjcmlwdD4="), "a picture of another type is left out");
    // The page redacts its own words, whatever reaches it.
    let key = ["sk", "ant", "api03", &"A".repeat(36)].join("-");
    let secret = Shown { caption: format!("caption {key}"), image: png.image.clone() };
    let html = page(&format!("title {key}"), &format!("summary {key}"), &[(1, &secret, Some(&format!("note {key}")))]);
    assert!(!html.contains(&key), "{html}");
    assert!(html.contains("summary ") && html.contains("caption ") && html.contains("note "));
}
