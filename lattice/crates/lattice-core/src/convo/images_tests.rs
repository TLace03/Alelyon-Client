//! Images in the composer (`super::images`), end to end over the agent
//! harness: an attached image goes to the model with the words and is
//! recorded with the user turn; a regenerate sends it again and a later turn
//! names it; a plain turn, a model that cannot see and bad images are refused;
//! a model that refuses a message with images is learned; and images to a
//! model off this PC ask every time.

use lattice_agents::model::{InputItem, ModelError};
use lattice_agents::testing::ScriptedStep;
use lattice_protocol::conversation::{
    Accepted, AgentChatService, ConversationEventKind, Mode, RegenerateRequest, SendRequest,
    UserImage,
};
use lattice_protocol::{RefusalKind, Shown};

use super::agent::words as agent_words;
use super::agent_tests::{H, hosted, local, say};
use super::images::{self, words};
use super::item::Item;
use crate::ports::ConfirmRequest;

/// A 1x1 PNG.
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg==";

fn png() -> UserImage {
    UserImage {
        media_type: "image/png".into(),
        base64: PNG.into(),
    }
}

fn send(
    h: &H,
    conversation: Option<&str>,
    text: &str,
    (choice, shown): (&str, Shown),
    mode: Mode,
    workspace: Option<&str>,
    images: Vec<UserImage>,
) -> Result<Accepted, lattice_protocol::Refusal> {
    h.runtime.block_on(h.chat.send(SendRequest {
        conversation: conversation.map(str::to_owned),
        text: text.into(),
        choice: choice.into(),
        shown,
        mode,
        workspace: workspace.map(str::to_owned),
        edit_of: None,
        project: None,
        images,
    }))
}

fn started(accepted: Accepted) -> String {
    match accepted {
        Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    }
}

/// The images of the last message a model call carried, and its words.
fn carried(input: &[InputItem]) -> Option<(String, Vec<String>)> {
    input.iter().rev().find_map(|item| match item {
        InputItem::UserImages { text, images } => Some((
            text.clone(),
            images.iter().map(|image| image.base64.clone()).collect(),
        )),
        _ => None,
    })
}

/// An image goes to the model with the words, and is recorded with the user
/// turn (a blob, the record, the event); a regenerate sends it again; a later
/// turn names it in the replay and does not send it again.
/// Mutants: the images left out of the message; a regenerate without them;
/// the replay sending no note.
#[test]
fn an_image_goes_with_the_words_is_recorded_and_named_later() {
    let h = H::new("images-send");
    let ws = h.workspace();
    let model = h.script(vec![say("The button is disabled.")]);
    let id = started(
        send(
            &h,
            None,
            "Why is this button grey?",
            ("local", local()),
            Mode::Agent,
            Some(&ws),
            vec![png()],
        )
        .unwrap(),
    );
    let events = h.turns_end(&id, 1);
    let first = model.calls()[0].input.clone();
    assert_eq!(
        carried(&first),
        Some(("Why is this button grey?".to_owned(), vec![PNG.to_owned()]))
    );
    let attached: Vec<(u32, u64)> = events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::ImagesAttached { count, bytes, .. } => Some((*count, *bytes)),
            _ => None,
        })
        .collect();
    let png_bytes = images::decode_base64(PNG).unwrap();
    assert_eq!(attached, [(1, png_bytes.len() as u64)]);
    let items = h.sidecar_items(&id);
    let stored = items
        .iter()
        .find_map(|item| match item {
            Item::ImagesAttached { images, .. } => Some(images.clone()),
            _ => None,
        })
        .expect("recorded");
    let store = super::sidecar::SidecarStore::new(h.state.native_chat_dir());
    assert_eq!(
        store.read_blob(&id, &stored[0].sha256).unwrap(),
        png_bytes,
        "kept as given"
    );

    // A regenerate sends the image again.
    model.enqueue(say("It is disabled until the form is valid."));
    h.runtime
        .block_on(h.chat.regenerate(RegenerateRequest {
            conversation: id.clone(),
            choice: "local".into(),
            shown: local(),
            mode: Mode::Agent,
        }))
        .unwrap();
    h.turns_end(&id, 2);
    assert_eq!(
        carried(&model.calls()[1].input).map(|(_, images)| images),
        Some(vec![PNG.to_owned()])
    );

    // A later message: the image is named, not sent again.
    model.enqueue(say("Yes."));
    send(
        &h,
        Some(&id),
        "And now?",
        ("local", local()),
        Mode::Agent,
        Some(&ws),
        Vec::new(),
    )
    .unwrap();
    h.turns_end(&id, 3);
    let later = model.calls()[2].input.clone();
    assert!(carried(&later).is_none(), "not sent again: {later:?}");
    let named = later.iter().any(|item| {
        matches!(item, InputItem::User(text)
            if text.starts_with("Why is this button grey?") && text.contains(&images::note(1)))
    });
    assert!(named, "{later:?}");
}

/// A plain turn, a model this session learned cannot see, and a bad image
/// are refused with their sentences, and nothing is sent.
/// Mutants: images taken on a plain turn; the vision check skipped.
#[test]
fn images_are_refused_where_they_cannot_go() {
    let h = H::new("images-refused");
    let ws = h.workspace();
    let model = h.script(vec![]);
    // No folder, Ask mode: a plain turn.
    let plain = send(
        &h,
        None,
        "Look.",
        ("local", local()),
        Mode::Ask,
        None,
        vec![png()],
    )
    .unwrap_err();
    assert_eq!(
        (plain.kind, plain.message.as_str()),
        (RefusalKind::Conflict, words::PLAIN)
    );
    let bad = send(
        &h,
        None,
        "Look.",
        ("local", local()),
        Mode::Agent,
        Some(&ws),
        vec![UserImage {
            media_type: "image/png".into(),
            base64: crate::browser::ws::base64(b"GIF89a....."),
        }],
    )
    .unwrap_err();
    assert_eq!(bad.message, words::NOT_AN_IMAGE);
    let resolved = h.chat.inner.resolve("local", &local()).unwrap();
    super::agent::lock(&h.chat.inner.no_vision).insert(resolved.id.clone());
    let blind = send(
        &h,
        None,
        "Look.",
        ("local", local()),
        Mode::Agent,
        Some(&ws),
        vec![png()],
    )
    .unwrap_err();
    assert_eq!(blind.message, words::CANNOT_SEE);
    assert!(model.calls().is_empty(), "nothing was sent");
}

/// A model that refuses a message with images (a 400) is learned as one
/// that cannot see, not as one without tools; the next send with images is
/// refused before anything is sent.
/// Mutant: the first call's status read before the images'.
#[test]
fn a_model_that_refuses_images_is_learned_as_one_that_cannot_see() {
    let h = H::new("images-learned");
    let ws = h.workspace();
    let model = h.script(vec![ScriptedStep::error(ModelError::Status(400))]);
    let id = started(
        send(
            &h,
            None,
            "Look.",
            ("local", local()),
            Mode::Agent,
            Some(&ws),
            vec![png()],
        )
        .unwrap(),
    );
    let events = h.turns_end(&id, 1);
    let said = format!("{events:?}");
    assert!(said.contains(agent_words::NO_VISION), "{said}");
    assert!(!said.contains(agent_words::NO_TOOLS), "{said}");
    let again = send(
        &h,
        Some(&id),
        "Look again.",
        ("local", local()),
        Mode::Agent,
        Some(&ws),
        vec![png()],
    )
    .unwrap_err();
    assert_eq!(again.message, words::CANNOT_SEE);
    assert_eq!(model.calls().len(), 1);
}

/// Images to a model off this PC ask every time, after the first remote
/// send's question; refused, nothing is written or sent.
/// Mutants: the question skipped once the conversation was confirmed; a
/// refusal still sending.
#[test]
fn images_to_a_model_off_this_pc_ask_every_time() {
    let h = H::new("images-remote");
    let ws = h.workspace();
    let model = h.script(vec![say("Seen."), say("Seen again.")]);
    let id = started(
        send(
            &h,
            None,
            "Look.",
            ("endpoint:hosted", hosted()),
            Mode::Agent,
            Some(&ws),
            vec![png(), png()],
        )
        .unwrap(),
    );
    h.turns_end(&id, 1);
    send(
        &h,
        Some(&id),
        "And this one.",
        ("endpoint:hosted", hosted()),
        Mode::Agent,
        Some(&ws),
        vec![png()],
    )
    .unwrap();
    h.turns_end(&id, 2);
    let asked: Vec<u32> = h
        .confirm
        .asked()
        .iter()
        .filter_map(|request| match request {
            ConfirmRequest::SendImages { images, .. } => Some(*images),
            _ => None,
        })
        .collect();
    assert_eq!(asked, [2, 1], "asked for each send with images");
    assert_eq!(model.calls().len(), 2);

    *h.confirm.answer.lock().unwrap() = false;
    let refused = send(
        &h,
        Some(&id),
        "One more.",
        ("endpoint:hosted", hosted()),
        Mode::Agent,
        Some(&ws),
        vec![png()],
    )
    .unwrap_err();
    assert_eq!(refused.message, words::NOT_CONFIRMED);
    assert_eq!(model.calls().len(), 2, "nothing more was sent");
    assert_eq!(
        h.sidecar_items(&id)
            .iter()
            .filter(|item| matches!(item, Item::ImagesAttached { .. }))
            .count(),
        2,
        "nothing more was written"
    );
}
