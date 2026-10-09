//! The managed server's tool-call probe in agent turns (spec §22 LR8′, row
//! E11's tests): a good tool call passes, is recorded once, and the turn goes
//! on as an agent turn; prose, a malformed call or two calls fail it, the turn
//! ends saying so, and later sends are plain; a record that exists sends no
//! probe; a record the core cannot read is left as it is, and the answer holds
//! for the process. The model is the scripted one, behind the fake runtime.

use lattice_agents::model::{InputItem, ModelRequest};
use lattice_agents::testing::ScriptedStep;
use lattice_protocol::conversation::{
    Accepted, ConversationEvent, ConversationEventKind, Mode, TurnKind, TurnStatus,
};
use serde_json::json;

use super::agent::words;
use lattice_agents::testing::function_call_json;

use super::agent_tests::{H, call, local, say, statuses};
use super::item::Item;
use super::probe::{PROMPT, TOOL};
use crate::llama::files::LlamaPaths;
use crate::llama::probes::{self, TOOL_PROBES};

/// A chat whose capabilities are the core's own (no test override).
fn harness(tag: &str) -> H {
    H::with(tag, &[], |config| config.caps = None)
}

fn paths(h: &H) -> LlamaPaths {
    LlamaPaths::from_env(&h.env)
}

/// The probe's key for the harness's managed model.
fn key(h: &H) -> String {
    let paths = paths(h);
    let sha = probes::binary_sha256(&paths.binary).unwrap();
    probes::probe_key(&sha, &paths.models_dir.join("qwen3-8b.gguf")).unwrap()
}

/// An Agent-mode send on Local; its conversation and turn kind.
fn send(h: &H, conversation: Option<&str>, workspace: &str) -> (String, TurnKind) {
    match h
        .send(
            conversation,
            "Look around.",
            "local",
            local(),
            Mode::Agent,
            Some(workspace),
        )
        .unwrap()
    {
        Accepted::Started {
            conversation,
            turn_kind,
            ..
        } => (conversation.id, turn_kind),
        other => panic!("{other:?}"),
    }
}

fn is_probe(request: &ModelRequest) -> bool {
    request.tools.len() == 1
        && request.tools[0].name == TOOL
        && request.input == vec![InputItem::User(PROMPT.to_owned())]
}

fn says(events: &[ConversationEvent], words: &str) -> bool {
    events.iter().any(|event| match &event.kind {
        ConversationEventKind::Notice { text } => text.starts_with(words),
        ConversationEventKind::Error { message } => message == words,
        _ => false,
    })
}

fn probes_recorded(h: &H, id: &str) -> Vec<bool> {
    h.sidecar_items(id)
        .into_iter()
        .filter_map(|item| match item {
            Item::ToolProbe { passed, .. } => Some(passed),
            _ => None,
        })
        .collect()
}

#[test]
fn a_passing_probe_is_recorded_once_and_the_turn_goes_on_as_an_agent_turn() {
    let h = harness("probe-pass");
    let ws = h.workspace();
    let model = h.script(vec![call(TOOL, json!({"text": "ok"}), "p1"), say("done")]);
    let (id, kind) = send(&h, None, &ws);
    assert_eq!(
        kind,
        TurnKind::Agent,
        "before a record, Agent mode is an agent turn"
    );
    let events = h.turns_end(&id, 1);
    assert_eq!(statuses(&events), [TurnStatus::Completed]);
    assert!(says(&events, words::PROBE_PASSED));
    let calls = model.calls();
    assert_eq!(calls.len(), 2);
    assert!(is_probe(&calls[0]), "{:?}", calls[0]);
    assert_eq!(calls[0].settings.tool_choice.as_deref(), Some("required"));
    assert!(!is_probe(&calls[1]));
    assert!(calls[1].tools.iter().any(|tool| tool.name == "read_file"));
    assert_eq!(probes::tool_probe(&paths(&h), &key(&h)), Some(true));
    assert_eq!(probes_recorded(&h, &id), [true]);
    // The next turn sends no probe.
    let model = h.script(vec![say("again")]);
    let (_, kind) = send(&h, Some(&id), &ws);
    assert_eq!(kind, TurnKind::Agent);
    h.turns_end(&id, 2);
    let calls = model.calls();
    assert_eq!(calls.len(), 1);
    assert!(!is_probe(&calls[0]));
    assert_eq!(probes_recorded(&h, &id), [true], "probed once");
}

#[test]
fn prose_a_malformed_call_or_two_calls_fail_the_probe_and_later_sends_are_plain() {
    let two = ScriptedStep::respond(vec![
        function_call_json(TOOL, &json!({"text": "ok"}), "p1"),
        function_call_json(TOOL, &json!({"text": "ok"}), "p2"),
    ]);
    let malformed = ScriptedStep::respond(vec![lattice_agents::model::OutputItem::FunctionCall {
        call_id: "p1".into(),
        name: TOOL.into(),
        arguments: "{\"text\": ".into(),
    }]);
    for (tag, reply) in [
        ("probe-prose", say("I would call it.")),
        ("probe-malformed", malformed),
        ("probe-two", two),
    ] {
        let h = harness(tag);
        let ws = h.workspace();
        let model = h.script(vec![reply]);
        let (id, kind) = send(&h, None, &ws);
        assert_eq!(kind, TurnKind::Agent, "{tag}");
        let events = h.turns_end(&id, 1);
        assert_eq!(statuses(&events), [TurnStatus::Failed], "{tag}");
        assert!(says(&events, words::LOCAL_NO_TOOLS), "{tag}");
        assert_eq!(model.calls().len(), 1, "{tag}: only the probe was sent");
        assert_eq!(
            probes::tool_probe(&paths(&h), &key(&h)),
            Some(false),
            "{tag}"
        );
        assert_eq!(probes_recorded(&h, &id), [false], "{tag}");
        // Later sends are plain, and say so.
        h.script(vec![say("plain answer")]);
        let (_, kind) = send(&h, Some(&id), &ws);
        assert_eq!(kind, TurnKind::Plain, "{tag}");
        let events = h.turns_end(&id, 2);
        assert!(says(&events, super::caps::PLAIN_FALLBACK), "{tag}");
    }
}

#[test]
fn a_record_that_exists_sends_no_probe() {
    for passed in [true, false] {
        let h = harness(if passed {
            "probe-known-yes"
        } else {
            "probe-known-no"
        });
        let ws = h.workspace();
        probes::record_tool_probe(&paths(&h), &key(&h), passed).unwrap();
        let model = h.script(vec![say("answer")]);
        let (id, kind) = send(&h, None, &ws);
        h.turns_end(&id, 1);
        if passed {
            assert_eq!(kind, TurnKind::Agent);
            assert!(!is_probe(&model.calls()[0]));
        } else {
            assert_eq!(kind, TurnKind::Plain);
        }
        assert!(probes_recorded(&h, &id).is_empty());
    }
}

#[test]
fn a_record_lattice_cannot_read_is_left_as_it_is_and_the_answer_holds_for_the_process() {
    let h = harness("probe-unreadable");
    let ws = h.workspace();
    let file = paths(&h).llama_dir.join(TOOL_PROBES);
    std::fs::write(&file, "[not a record").unwrap();
    h.script(vec![call(TOOL, json!({"text": "ok"}), "p1"), say("done")]);
    let (id, _) = send(&h, None, &ws);
    let events = h.turns_end(&id, 1);
    assert_eq!(statuses(&events), [TurnStatus::Completed]);
    assert!(
        events.iter().any(|event| matches!(
            &event.kind,
            ConversationEventKind::Notice { text }
                if text.starts_with(words::PROBE_PASSED) && text.contains("checked again after Lattice restarts")
        )),
        "{events:#?}"
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "[not a record");
    // This process remembers the answer: no second probe.
    let model = h.script(vec![say("again")]);
    send(&h, Some(&id), &ws);
    h.turns_end(&id, 2);
    assert!(!is_probe(&model.calls()[0]));
}

fn props(template: bool, vision: Option<bool>) -> crate::llama::server::Props {
    crate::llama::server::Props {
        chat_template: template.then(|| "{{ messages }}".to_owned()),
        n_ctx: Some(4096),
        vision,
    }
}

/// LR8: a server whose `/props` reports no chat template fails the probe
/// without being asked.
#[test]
fn a_server_with_no_chat_template_fails_the_probe_unasked() {
    let h = harness("probe-no-template");
    let ws = h.workspace();
    h.local.set_props(Some(props(false, None)));
    let model = h.script(vec![call(TOOL, json!({"text": "ok"}), "p1")]);
    let (id, _) = send(&h, None, &ws);
    let events = h.turns_end(&id, 1);
    assert!(says(&events, words::LOCAL_NO_TOOLS));
    assert!(model.calls().is_empty(), "nothing was sent");
    assert_eq!(probes::tool_probe(&paths(&h), &key(&h)), Some(false));
    let reply = h
        .sidecar_items(&id)
        .into_iter()
        .find_map(|item| match item {
            Item::ToolProbe { reply, .. } => Some(format!("{reply:?}")),
            _ => None,
        });
    assert!(reply.unwrap().contains("no chat template"));
}

/// The agent's browser is not offered to a managed model whose `/props` says
/// it cannot see, and is to one that can.
#[test]
fn the_browser_is_offered_to_a_local_model_only_when_its_props_do_not_say_it_is_blind() {
    for (vision, offered) in [(Some(false), false), (Some(true), true), (None, true)] {
        let h = harness("probe-vision");
        let ws = h.workspace();
        probes::record_tool_probe(&paths(&h), &key(&h), true).unwrap();
        crate::browser::prefs::set(&h.state, true).unwrap();
        h.local.set_props(Some(props(true, vision)));
        let model = h.script(vec![say("answer")]);
        let (id, _) = send(&h, None, &ws);
        h.turns_end(&id, 1);
        let tools = &model.calls()[0].tools;
        assert_eq!(
            tools.iter().any(|tool| tool.name == "browser_look"),
            offered,
            "{vision:?}"
        );
    }
}
