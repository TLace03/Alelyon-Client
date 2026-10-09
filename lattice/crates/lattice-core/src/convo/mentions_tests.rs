//! @-mentions (`super::mentions`), end to end over the agent harness: the
//! files a message names are given to the model just before it, as
//! `read_file` shows them; one that cannot be read (missing, or ignored) is
//! said so to the model and the reader; later turns are not given them again;
//! and for a model off this PC a mentioned file holding a secret is withheld
//! while the message still goes.

use lattice_agents::model::InputItem;
use lattice_protocol::conversation::{ConversationEventKind, Mode};

use super::agent_tests::{H, hosted, local, say, secret};
use super::tripwire::WITHHELD;

fn started(accepted: lattice_protocol::conversation::Accepted) -> String {
    match accepted {
        lattice_protocol::conversation::Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    }
}

/// The user items of a model call, in order.
fn users(input: &[InputItem]) -> Vec<String> {
    input
        .iter()
        .filter_map(|item| match item {
            InputItem::User(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

/// Two files read and given just before the message, a missing one and an
/// ignored one said so; the reader's notice names both; a later message is
/// not given them again.
/// Mutants: the mentions not read; an ignored file read anyway (the read
/// tools' rules bypassed); the item put after the message.
#[test]
fn mentioned_files_go_just_before_the_message_as_read_file_shows_them() {
    let h = H::new("mentions-read");
    std::fs::write(h.folder.join("a.txt"), "alpha one\nalpha two\n").unwrap();
    std::fs::write(h.folder.join("b.txt"), "beta\n").unwrap();
    std::fs::write(h.folder.join(".gitignore"), "hidden.txt\n").unwrap();
    std::fs::write(h.folder.join("hidden.txt"), "do not read\n").unwrap();
    let ws = h.workspace();
    let model = h.script(vec![say("Read."), say("Again.")]);
    let text = "Compare @a.txt and @b.txt, then @missing.txt and @hidden.txt.";
    let id = started(
        h.send(None, text, "local", local(), Mode::Agent, Some(&ws))
            .unwrap(),
    );
    let events = h.turns_end(&id, 1);
    let first = users(&model.calls()[0].input);
    let n = first.len();
    assert!(n >= 2, "{first:?}");
    assert_eq!(first[n - 1], text, "the message is last");
    let given = &first[n - 2];
    assert!(
        given.contains("alpha one") && given.contains("alpha two") && given.contains("beta"),
        "{given}"
    );
    assert!(
        given.contains("@missing.txt was not read") && given.contains("@hidden.txt was not read"),
        "{given}"
    );
    assert!(
        !given.contains("do not read"),
        "an ignored file stays unread"
    );
    let notice = events
        .iter()
        .find_map(|event| match &event.kind {
            ConversationEventKind::Notice { text } if text.contains("@a.txt") => Some(text.clone()),
            _ => None,
        })
        .expect("a notice");
    assert!(
        notice.contains("given @a.txt, @b.txt") && notice.contains("Not read: @missing.txt"),
        "{notice}"
    );

    h.send(
        Some(&id),
        "And now?",
        "local",
        local(),
        Mode::Agent,
        Some(&ws),
    )
    .unwrap();
    h.turns_end(&id, 2);
    let later = users(&model.calls()[1].input);
    assert!(
        !later.iter().any(|text| text.contains("alpha two")),
        "not given again: {later:?}"
    );
}

/// For a model off this PC, a mentioned file holding a key-shaped string is
/// withheld by the tripwire, and the message itself still goes.
/// Mutant: the mentions joined into the message's own item (the message
/// withheld with them).
#[test]
fn a_mentioned_secret_is_withheld_from_a_model_off_this_pc_and_the_message_goes() {
    let h = H::new("mentions-secret");
    let key = secret();
    std::fs::write(h.folder.join("env.txt"), format!("TOKEN={key}\n")).unwrap();
    let ws = h.workspace();
    let model = h.script(vec![say("Seen.")]);
    let text = "What does @env.txt set?";
    let id = started(
        h.send(
            None,
            text,
            "endpoint:hosted",
            hosted(),
            Mode::Agent,
            Some(&ws),
        )
        .unwrap(),
    );
    h.turns_end(&id, 1);
    let sent = format!("{:?}", model.calls());
    assert!(!sent.contains(&key), "the key reached the remote model");
    let first = users(&model.calls()[0].input);
    assert!(first.iter().any(|item| item == WITHHELD), "{first:?}");
    assert_eq!(first.last().map(String::as_str), Some(text));
}
