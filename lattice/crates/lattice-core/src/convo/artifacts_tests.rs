//! The artifacts the agent saves (`super::artifacts`), end to end over the
//! agent harness: a version is recorded, shown as an event and read back by
//! the window and by the agent; the same name makes the next version; a long
//! text is kept as a blob; bad names, titles and sizes, and too many versions
//! or artifacts, are refused with a sentence; a secret is redacted; and a
//! reopened conversation shows what was saved.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::RefusalKind;
use lattice_protocol::conversation::{ArtifactKind, ConversationEvent, ConversationEventKind};
use serde_json::json;

use super::agent_tests::{H, call, say, secret};
use super::artifacts::{self, words};
use super::item::{Item, Payload};

fn write(
    name: &str,
    title: &str,
    kind: &str,
    content: &str,
    id: &str,
) -> lattice_agents::testing::ScriptedStep {
    call(
        "write_artifact",
        json!({"name": name, "title": title, "kind": kind, "content": content}),
        id,
    )
}

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

fn saved(events: &[ConversationEvent]) -> Vec<(String, u32, ArtifactKind, u64)> {
    events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::ArtifactSaved {
                name,
                version,
                kind,
                bytes,
                ..
            } => Some((name.clone(), *version, *kind, *bytes)),
            _ => None,
        })
        .collect()
}

/// Two versions of a plan and a long page: each is an event and a record;
/// the window reads the latest or a given version; the agent reads one back;
/// a page past 16 KiB is kept as a blob and read back whole.
/// Mutants: the same name starting over at version 1; the window reading
/// the first version for the latest.
#[test]
fn an_artifact_is_saved_versioned_and_read_back() {
    let h = H::new("artifacts-versions");
    let ws = h.workspace();
    let page = format!(
        "<!doctype html><title>Report</title><p>{}</p>",
        "x".repeat(20_000)
    );
    let model = h.script(vec![
        write(
            "plan",
            "The plan",
            "markdown",
            "## Steps\n\n1. Read.\n",
            "w1",
        ),
        write(
            "plan",
            "The plan, revised",
            "markdown",
            "## Steps\n\n1. Read.\n2. Fix.\n",
            "w2",
        ),
        write("report", "Report", "html", &page, "w3"),
        call("read_artifact", json!({"name": "plan", "version": 1}), "r1"),
        say("Saved."),
    ]);
    let id = h.agent(None, "Plan it.", &ws);
    let events = h.turns_end(&id, 1);
    assert_eq!(
        saved(&events),
        [
            ("plan".to_owned(), 1, ArtifactKind::Markdown, 19),
            ("plan".to_owned(), 2, ArtifactKind::Markdown, 27),
            (
                "report".to_owned(),
                1,
                ArtifactKind::Html,
                page.len() as u64
            ),
        ]
    );
    let calls = model.calls();
    assert!(result_of(&calls, "w2").starts_with("Saved plan, version 2."));
    let read = result_of(&calls, "r1");
    assert!(
        read.starts_with("The plan (markdown), version 1 of 2:"),
        "{read}"
    );
    assert!(
        read.ends_with("1. Read.\n") && !read.contains("2. Fix."),
        "{read}"
    );

    let latest = h
        .runtime
        .block_on(h.chat.artifact(&id, "plan", None))
        .unwrap();
    assert_eq!(
        (
            latest.title.as_str(),
            latest.version,
            latest.versions,
            latest.text.as_str()
        ),
        ("The plan, revised", 2, 2, "## Steps\n\n1. Read.\n2. Fix.\n")
    );
    let first = h
        .runtime
        .block_on(h.chat.artifact(&id, "plan", Some(1)))
        .unwrap();
    assert_eq!(
        (first.version, first.text.as_str()),
        (1, "## Steps\n\n1. Read.\n")
    );
    let shown = h
        .runtime
        .block_on(h.chat.preview_artifact(&id, "plan", None))
        .unwrap_err();
    assert_eq!(
        shown.kind,
        RefusalKind::Invalid,
        "Markdown is shown in the editor, not previewed"
    );
    let report = h
        .runtime
        .block_on(h.chat.artifact(&id, "report", None))
        .unwrap();
    assert_eq!(report.text, page, "a long page read back whole");
    assert!(h.sidecar_items(&id).iter().any(|item| matches!(
        item,
        Item::ArtifactSaved { name, text: Payload::Blob { .. }, .. } if name == "report"
    )));
    for (name, version, why) in [
        ("plan", Some(3), words::NO_VERSION),
        ("plan", Some(0), words::NO_VERSION),
        ("nothing", None, words::NONE),
        ("../plan", None, words::NONE),
    ] {
        let refused = h
            .runtime
            .block_on(h.chat.artifact(&id, name, version))
            .unwrap_err();
        assert_eq!(
            (refused.kind, refused.message.as_str()),
            (RefusalKind::NotFound, why)
        );
    }
}

/// Bad names, titles and contents are refused with their sentences and save
/// nothing; a secret is redacted before it is kept or shown.
/// Mutant: the content not redacted.
#[test]
fn bad_artifacts_are_refused_and_a_secret_is_redacted() {
    let h = H::new("artifacts-bounds");
    let ws = h.workspace();
    let key = secret();
    let model = h.script(vec![
        write("Plan", "t", "markdown", "x", "n1"),
        write("a--b", "t", "markdown", "x", "n2"),
        write(&"a".repeat(49), "t", "markdown", "x", "n3"),
        write("plan", "two\nlines", "markdown", "x", "t1"),
        write("plan", "t", "markdown", "  \n", "c1"),
        write(
            "plan",
            "t",
            "markdown",
            &"x".repeat(artifacts::MAX_BYTES + 1),
            "c2",
        ),
        write("plan", "t", "pdf", "x", "k1"),
        write("keys", "Keys", "text", &format!("Use {key} here."), "s1"),
        say("Done."),
    ]);
    let id = h.agent(None, "Save things.", &ws);
    let events = h.turns_end(&id, 1);
    let calls = model.calls();
    for n in ["n1", "n2", "n3"] {
        assert!(
            result_of(&calls, n).contains("lowercase letters, digits and single hyphens"),
            "{n}"
        );
    }
    assert!(result_of(&calls, "t1").contains("one line of 1 to 80 characters"));
    assert!(result_of(&calls, "c1").contains("1 byte to 256 KiB"));
    assert!(result_of(&calls, "c2").contains("1 byte to 256 KiB"));
    assert!(
        result_of(&calls, "k1").contains("do not fit"),
        "{}",
        result_of(&calls, "k1")
    );
    assert!(result_of(&calls, "s1").contains(crate::secrets::REDACTED));
    assert_eq!(saved(&events).len(), 1, "only the redacted one was saved");
    let kept = h
        .runtime
        .block_on(h.chat.artifact(&id, "keys", None))
        .unwrap();
    assert!(
        !kept.text.contains(&key) && kept.text.contains(crate::secrets::REDACTED),
        "{}",
        kept.text
    );
    assert!(
        !h.sidecar_items(&id)
            .iter()
            .any(|item| matches!(item, Item::ArtifactSaved { .. })
                && serde_json::to_string(item).unwrap().contains(&key))
    );
}

/// An artifact keeps 20 versions and a conversation 32 artifacts; past
/// either, a save is refused and says what to do instead.
/// Mutant: no limit on versions.
#[test]
fn versions_and_artifacts_have_limits() {
    let h = H::new("artifacts-limits");
    let ws = h.workspace();
    let mut steps = Vec::new();
    for n in 1..=artifacts::MAX_VERSIONS + 1 {
        steps.push(write(
            "log",
            "Log",
            "text",
            &format!("v{n}"),
            &format!("v{n}"),
        ));
    }
    steps.push(say("Done."));
    let model = h.script(steps);
    let id = h.agent(None, "Save many versions.", &ws);
    h.turns_end(&id, 1);
    // Another turn: a turn takes a bounded number of steps.
    for n in 2..=artifacts::MAX_ARTIFACTS + 1 {
        model.enqueue(write(&format!("a{n}"), "A", "text", "x", &format!("a{n}")));
    }
    model.enqueue(say("Done."));
    h.agent(Some(&id), "Save many artifacts.", &ws);
    let events = h.turns_end(&id, 2);
    let calls = model.calls();
    let past = format!("v{}", artifacts::MAX_VERSIONS + 1);
    assert!(
        result_of(&calls, &past).contains("versions already"),
        "{}",
        result_of(&calls, &past)
    );
    let last = format!("a{}", artifacts::MAX_ARTIFACTS + 1);
    assert!(
        result_of(&calls, &last).contains("artifacts already"),
        "{}",
        result_of(&calls, &last)
    );
    let all = saved(&events);
    assert_eq!(
        all.len(),
        artifacts::MAX_VERSIONS + artifacts::MAX_ARTIFACTS - 1
    );
    let names: std::collections::BTreeSet<&str> =
        all.iter().map(|(name, ..)| name.as_str()).collect();
    assert_eq!(names.len(), artifacts::MAX_ARTIFACTS);
}

/// A conversation read again from its record shows its artifacts as they
/// were saved.
/// Mutant: the records not read back as events.
#[test]
fn a_reopened_conversation_shows_its_artifacts() {
    let h = H::new("artifacts-seed");
    let convo = h.chat.inner.convo("0123456789ab");
    let item = Item::ArtifactSaved {
        turn: "a1b2c3d4e5f6".into(),
        call_id: "w1".into(),
        name: "plan".into(),
        title: "The plan".into(),
        kind: ArtifactKind::Markdown,
        version: 1,
        text: Payload::Inline("## Steps".into()),
        bytes: 8,
        at: 1.0,
    };
    super::views::seed(&convo, &[item]);
    let (events, _) = convo.log.recorded();
    assert_eq!(
        saved(&events),
        [("plan".to_owned(), 1, ArtifactKind::Markdown, 8)]
    );
    assert!(
        artifacts::is_name("api-report-2")
            && !artifacts::is_name("-a")
            && !artifacts::is_name("a-")
    );
}
