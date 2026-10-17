//! `view_image` and `web_fetch` in a turn: an image of the folder reaches
//! the next model call, as a screenshot does; a file that is not the image
//! its name says, or not an image, is refused; a model that cannot see is
//! not offered it; `web_fetch` is offered in every turn and refuses this PC.

use std::sync::Arc;

use lattice_agents::model::{InputItem, ModelRequest};
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::caps::{ModelCaps, Tri};

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

const PNG: &[u8] = &[0x89, b'P', b'N', b'G', 13, 10, 26, 10, 0, 0, 0, 13];

/// Mutants: the image never reaching the model; a fake image shown; the
/// tool offered to a model that cannot see; web_fetch reaching this PC.
#[test]
fn an_image_of_the_folder_reaches_the_model_and_fakes_are_refused() {
    let h = H::new("sight");
    let ws = h.workspace();
    std::fs::write(h.folder.join("shot.png"), PNG).unwrap();
    std::fs::write(h.folder.join("fake.png"), b"not a picture").unwrap();
    std::fs::write(h.folder.join("notes.txt"), b"words").unwrap();
    let model = h.script(vec![
        call("view_image", json!({"path": "shot.png"}), "v1"),
        call("view_image", json!({"path": "fake.png"}), "v2"),
        call("view_image", json!({"path": "notes.txt"}), "v3"),
        call("web_fetch", json!({"url": "http://127.0.0.1:9/"}), "f1"),
        say("Looked."),
    ]);
    let id = h.agent(None, "Look at the screenshot.", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(calls[0].tools.iter().any(|t| t.name == "view_image") && calls[0].tools.iter().any(|t| t.name == "web_fetch"));
    assert!(result_of(&calls, "v1").starts_with("shot.png (12 bytes) comes with the next message."), "{}", result_of(&calls, "v1"));
    let shown = calls[1].input.iter().any(|i| match i {
        InputItem::UserImages { text, images } => text.contains("The image shot.png") && images.iter().any(|im| im.media_type == "image/png"),
        _ => false,
    });
    assert!(shown, "the image came with the next call");
    assert!(result_of(&calls, "v2").contains("not the image its name says"), "{}", result_of(&calls, "v2"));
    assert!(result_of(&calls, "v3").contains("PNG, JPEG, GIF and WebP"), "{}", result_of(&calls, "v3"));
    assert!(result_of(&calls, "f1").contains("this PC"), "{}", result_of(&calls, "f1"));

    // A PDF's page, rendered by Windows, comes with the next call; a page past the last says how many there are.
    std::fs::write(h.folder.join("plan.pdf"), crate::readpdf::tests::sample(2)).unwrap();
    model.enqueue(call("read_pdf", json!({"path": "plan.pdf", "page": 2}), "p1"));
    model.enqueue(call("read_pdf", json!({"path": "plan.pdf", "page": 9}), "p2"));
    model.enqueue(call("read_pdf", json!({"path": "shot.png"}), "p3"));
    model.enqueue(say("Read."));
    h.agent(Some(&id), "Read the plan.", &ws);
    h.turns_end(&id, 2);
    let calls = model.calls();
    if crate::exec::run::powershell_path(&crate::env::ProcessEnv).is_some() {
        assert!(result_of(&calls, "p1").starts_with("Page 2 of 2 of plan.pdf comes with the next message"), "{}", result_of(&calls, "p1"));
        let shown = calls.iter().any(|c| c.input.iter().any(|i| matches!(i, InputItem::UserImages { text, .. } if text.contains("Page 2 of 2 of plan.pdf"))));
        assert!(shown, "the page came with a call");
        assert!(result_of(&calls, "p2").contains("plan.pdf has 2 pages"), "{}", result_of(&calls, "p2"));
    }
    assert!(result_of(&calls, "p3").contains("reads PDF files"), "{}", result_of(&calls, "p3"));

    // A model that cannot see is not offered it.
    let blind = H::with("sight-blind", &[], |config| {
        config.caps = Some(Arc::new(|_| ModelCaps { tools: Tri::Yes, vision: Tri::No, context_tokens: None }));
    });
    let ws = blind.workspace();
    let model = blind.script(vec![say("ok")]);
    let id = blind.agent(None, "Hello.", &ws);
    blind.turns_end(&id, 1);
    let calls = model.calls();
    assert!(!calls[0].tools.iter().any(|t| t.name == "view_image"));
    assert!(calls[0].tools.iter().any(|t| t.name == "web_fetch"));
}
