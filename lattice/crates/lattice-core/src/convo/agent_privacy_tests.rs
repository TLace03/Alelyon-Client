//! The agent chat's privacy falsifiers, end to end (spec §10.3 TF1–TF3,
//! TF6–TF8, TF12, TF15, and §22.8 RP4 (a) and (b), which replace TF14; the
//! chat spec's PF1, PF2, PF13, PF14 and PF15
//! re-run on agent turns, with tool calls in the scripted model). Each test
//! names the mutant it must catch; the mutant runs are logged by the row
//! (phaseE/logs/E11c-m*.log).

use lattice_protocol::conversation::{
    Accepted, AgentChatService, ConversationEventKind, Decision, Mode, TurnStatus,
};
use lattice_protocol::{Locality, RefusalKind, Shown};
use serde_json::json;

use super::agent::words;
use super::agent_tests::{H, call, ended, files, hosted, say, secret, statuses};
use crate::ports::ConfirmRequest;

fn conversation(accepted: Accepted) -> String {
    match accepted {
        Accepted::Started { conversation, .. } => conversation.id,
        other => panic!("{other:?}"),
    }
}

fn remote_agent(h: &H, id: Option<&str>, text: &str, ws: &str) -> String {
    conversation(
        h.send(id, text, "endpoint:hosted", hosted(), Mode::Agent, Some(ws))
            .unwrap(),
    )
}

fn first_remote_asks(h: &H) -> Vec<u32> {
    h.confirm
        .asked()
        .iter()
        .filter_map(|request| match request {
            ConfirmRequest::FirstRemoteSend {
                earlier_local_results,
                ..
            } => Some(*earlier_local_results),
            _ => None,
        })
        .collect()
}

/// TF1: a remote target: a `read_file` result holding a key-shaped string,
/// and the model's own earlier `edit_file` arguments holding one (replayed
/// as Assistant tool-call arguments), never reach the request, and
/// `Withheld` is emitted for each call.
/// Mutants: no tripwire for the remote target; the tripwire skipping
/// tool-call arguments (E11b's m2).
#[test]
fn tf1_a_secret_read_or_written_never_reaches_a_remote_model() {
    let h = H::new("tf1");
    let key = secret();
    std::fs::write(h.folder.join("key.txt"), format!("TOKEN={key}\n")).unwrap();
    let ws = h.workspace();
    let model = h.script(vec![
        call("read_file", json!({"path": "key.txt"}), "c_read"),
        call(
            "edit_file",
            json!({"path": "a.txt", "old_string": "a", "new_string": key}),
            "c_edit",
        ),
        say("done"),
    ]);
    let id = remote_agent(&h, None, "look at key.txt", &ws);
    let events = h.turns_end(&id, 1);
    let sent = format!("{:?}", model.calls());
    assert!(!sent.contains(&key), "the remote request held the key");
    let withheld: Vec<String> = events
        .iter()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::Withheld { call_id } => Some(call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(withheld, ["c_read", "c_edit"]);
    assert_eq!(first_remote_asks(&h), [0]);
    assert!(h.asked().iter().all(|asked| !asked.local));
}

/// TF2, TF1's positive control: the same on Local: the key reaches the
/// local model's request (nothing to withhold on this machine).
#[test]
fn tf2_on_local_the_same_request_holds_it() {
    let h = H::new("tf2");
    let key = secret();
    std::fs::write(h.folder.join("key.txt"), format!("TOKEN={key}\n")).unwrap();
    let ws = h.workspace();
    let model = h.script(vec![
        call("read_file", json!({"path": "key.txt"}), "c_read"),
        say("done"),
    ]);
    let id = h.agent(None, "look at key.txt", &ws);
    h.turns_end(&id, 1);
    assert!(format!("{:?}", model.calls()[1]).contains(&key));
    assert!(first_remote_asks(&h).is_empty());
}

/// TF3: `ReleaseWithheld` with the native dialog refusing leaves it
/// withheld; confirmed (for another call: CP3 does not ask again for a
/// refused one), the next model call carries that call's output, and the
/// refused one stays withheld.
/// Mutant: `decide(ReleaseWithheld)` skips the confirmation.
#[test]
fn tf3_a_withheld_result_is_sent_only_after_the_native_confirmation() {
    let h = H::new("tf3");
    let key = secret();
    std::fs::write(
        h.folder.join("key.txt"),
        format!(
            "TOKEN={key}
"
        ),
    )
    .unwrap();
    std::fs::write(
        h.folder.join("key2.txt"),
        format!(
            "OTHER={key}
"
        ),
    )
    .unwrap();
    let ws = h.workspace();
    let model = h.script(vec![
        call("read_file", json!({"path": "key.txt"}), "c_read"),
        call("read_file", json!({"path": "key2.txt"}), "c_read2"),
        call("ask_question", json!({"question": "Send it?"}), "q1"),
        say("done"),
    ]);
    let id = remote_agent(&h, None, "look", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::Question { .. }))
    });
    *h.confirm.answer.lock().unwrap() = false;
    let refused = h
        .runtime
        .block_on(h.chat.decide(&id, "c_read", Decision::ReleaseWithheld))
        .unwrap_err();
    assert_eq!(refused.message, words::RELEASE_REFUSED);
    *h.confirm.answer.lock().unwrap() = true;
    h.runtime
        .block_on(h.chat.decide(&id, "c_read2", Decision::ReleaseWithheld))
        .unwrap();
    h.chat.answer(&id, "q1", "yes".into()).unwrap();
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(!format!("{:?}", calls[2]).contains(&key), "withheld before");
    let last = format!("{:?}", calls[3]);
    assert!(last.contains(&format!("OTHER={key}")), "released: sent");
    assert!(
        !last.contains(&format!("TOKEN={key}")),
        "refused: still withheld"
    );
    let releases = h
        .confirm
        .asked()
        .iter()
        .filter(|request| matches!(request, ConfirmRequest::ReleaseWithheld { .. }))
        .count();
    assert_eq!(releases, 2);
}

/// TF8: the first remote send with the dialog refusing: zero model calls,
/// the store unchanged, the draft kept (the send is refused, not saved).
/// Mutant: the T1 dialog skipped.
#[test]
fn tf8_a_refused_first_remote_send_sends_and_writes_nothing() {
    let h = H::new("tf8");
    let ws = h.workspace();
    h.script(vec![say("never")]);
    *h.confirm.answer.lock().unwrap() = false;
    let before = h.snapshot();
    let refusal = h
        .send(
            None,
            "q",
            "endpoint:hosted",
            hosted(),
            Mode::Agent,
            Some(&ws),
        )
        .unwrap_err();
    assert_eq!(refusal.message, words::NOT_CONFIRMED_REMOTE);
    assert_eq!(h.snapshot(), before);
    assert!(h.asked().is_empty(), "zero model calls");
    assert_eq!(first_remote_asks(&h), [0]);
}

/// TF12: a Local agent reads a key-shaped string and quotes it in its
/// answer: no byte of it lands anywhere under the state root (sidecar,
/// blobs, runs) or in the transcript store, and a Notice says so.
/// Mutant: the final answer written before `redact`.
#[test]
fn tf12_no_key_shaped_string_is_written_anywhere() {
    let h = H::new("tf12");
    let key = secret();
    std::fs::write(h.folder.join("key.txt"), format!("TOKEN={key}\n")).unwrap();
    let ws = h.workspace();
    h.script(vec![
        call("read_file", json!({"path": "key.txt"}), "c_read"),
        say(&format!("The key is {key}.")),
    ]);
    let id = h.agent(None, "what is the key?", &ws);
    let events = h.turns_end(&id, 1);
    for (path, bytes) in files(&h.state.root) {
        assert!(
            !String::from_utf8_lossy(&bytes).contains(&key),
            "{path} holds the key"
        );
    }
    assert!(
        !h.snapshot().contains(&key),
        "the transcript store holds it"
    );
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        ConversationEventKind::Notice { text } if text == words::ANSWER_REDACTED
    )));
}

/// Append a record to a conversation's sidecar as it is, with no
/// redaction: a record written before T15, or one whose secret no
/// redaction pattern saw. Only the tripwire (T3) stands between it and a
/// remote model.
fn plant_raw(h: &H, id: &str, value: serde_json::Value) {
    use std::io::Write;
    let dir = super::sidecar::SidecarStore::new(h.state.native_chat_dir())
        .conversation_dir(id)
        .unwrap();
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(dir.join("items.jsonl"))
        .unwrap();
    writeln!(file, "{value}").unwrap();
}

/// RP4 (a), replacing TF14 (Amendment 4, RP2): tool results read under
/// Local, then a remote target, confirmed: the dialog states their number
/// as information, and they are sent in full (the call's arguments and its
/// output). The tripwire still screens every replayed item: a key planted
/// raw in the sidecar under the local turn is withheld, and `Withheld`
/// names its call.
/// Mutants: local tool items elided again for a remote target (m1); the
/// remote model not wrapped in the tripwire (m2).
#[test]
fn rp4a_local_tool_results_go_to_a_confirmed_remote_target_in_full() {
    let h = H::new("rp4a");
    let key = secret();
    std::fs::write(
        h.folder.join("notes.txt"),
        "LOCAL-CONTENT-77
",
    )
    .unwrap();
    let ws = h.workspace();
    h.script(vec![
        call("read_file", json!({"path": "notes.txt"}), "c_local"),
        say("read"),
    ]);
    let id = h.agent(None, "read notes", &ws);
    h.turns_end(&id, 1);
    let user = h.store.load(&id)[0].id.clone();
    plant_raw(
        &h,
        &id,
        json!({"type": "tool_call", "turn": user, "call_id": "c_planted",
               "tool": "read_file", "arguments": "{\"path\": \"old.txt\"}",
               "summary": "Read old.txt", "at": 2.0}),
    );
    plant_raw(
        &h,
        &id,
        json!({"type": "tool_result", "turn": user, "call_id": "c_planted",
               "output": format!("TOKEN={key}"), "withheld": false,
               "truncated": false, "at": 2.0}),
    );
    let model = h.script(vec![say("seen")]);
    remote_agent(&h, Some(&id), "now remote", &ws);
    let events = h.turns_end(&id, 2);
    assert_eq!(first_remote_asks(&h), [2], "the dialog states both");
    let sent = format!("{:?}", model.calls()[0]);
    assert!(
        sent.contains("LOCAL-CONTENT-77"),
        "the output in full: {sent}"
    );
    assert!(sent.contains("notes.txt"), "the call's arguments in full");
    assert!(!sent.contains(&key), "T3 withheld the planted key");
    assert!(sent.contains(super::tripwire::WITHHELD));
    assert!(events.iter().any(|event| matches!(
        &event.kind,
        ConversationEventKind::Withheld { call_id } if call_id == "c_planted"
    )));
}

/// RP4 (b): a refused `FirstRemoteSend` sends nothing and writes nothing,
/// whether it is the conversation's first remote send (with tool results
/// read under Local, which it states) or the first after a confirmed one
/// whose count has grown; CP3: the same request is not asked again from
/// the page.
/// Mutant: the refusal ignored (the turn goes ahead).
#[test]
fn rp4b_a_refused_first_remote_send_sends_nothing() {
    let h = H::new("rp4b");
    std::fs::write(
        h.folder.join("notes.txt"),
        "LOCAL-CONTENT-77
",
    )
    .unwrap();
    let ws = h.workspace();
    // A remote send first: confirmed with N = 0.
    h.script(vec![say("hello from far")]);
    let id = remote_agent(&h, None, "hi", &ws);
    h.turns_end(&id, 1);
    // Local reads.
    h.script(vec![
        call("read_file", json!({"path": "notes.txt"}), "c_local"),
        say("read"),
    ]);
    h.agent(Some(&id), "read notes", &ws);
    h.turns_end(&id, 2);
    let texts = h.texts(&id);
    let items = h.sidecar_items(&id).len();
    // Remote again, the same label and folder: asked again (the count grew),
    // refused: nothing sent, nothing written.
    let model = h.script(vec![say("never")]);
    *h.confirm.answer.lock().unwrap() = false;
    let refused = h
        .send(
            Some(&id),
            "now remote",
            "endpoint:hosted",
            hosted(),
            Mode::Agent,
            Some(&ws),
        )
        .unwrap_err();
    assert_eq!(refused.message, words::NOT_CONFIRMED_REMOTE);
    assert!(model.calls().is_empty(), "nothing sent");
    assert_eq!(first_remote_asks(&h), [0, 1]);
    assert_eq!(h.texts(&id), texts, "no turn written");
    assert_eq!(h.sidecar_items(&id).len(), items, "no item written");
    // CP3: the same request is not asked again from the page.
    *h.confirm.answer.lock().unwrap() = true;
    assert!(
        h.send(
            Some(&id),
            "now remote",
            "endpoint:hosted",
            hosted(),
            Mode::Agent,
            Some(&ws)
        )
        .is_err()
    );
    assert_eq!(first_remote_asks(&h), [0, 1]);
    assert!(model.calls().is_empty());
    // A new conversation's first remote send, refused: no conversation.
    *h.confirm.answer.lock().unwrap() = false;
    let before = h.snapshot();
    assert!(
        h.send(
            None,
            "fresh",
            "endpoint:hosted",
            hosted(),
            Mode::Agent,
            Some(&ws)
        )
        .is_err()
    );
    assert!(model.calls().is_empty(), "nothing sent");
    assert_eq!(h.snapshot(), before, "nothing written");
}

/// TF15: `%USERPROFILE%` and `%APPDATA%` are refused; a page's path with the
/// dialog refusing attaches nothing; a new folder on a conversation with a
/// remote label asks `FirstRemoteSend` again.
/// Mutant: a page's path attached without the dialog.
#[test]
fn tf15_attach_refusals_and_a_new_folder_asks_again() {
    let h = H::new("tf15");
    let home = h.scratch.home();
    let refused = h
        .runtime
        .block_on(h.chat.attach_native(None, home.clone()))
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::Invalid, "{}", refused.message);
    // A page path, dialog refusing: nothing attached.
    *h.confirm.answer.lock().unwrap() = false;
    let refused = h
        .runtime
        .block_on(
            h.chat
                .attach_workspace(None, h.folder.to_string_lossy().into_owned()),
        )
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::Invalid, "{}", refused.message);
    assert!(
        h.confirm
            .asked()
            .iter()
            .any(|request| matches!(request, ConfirmRequest::AttachFolder { .. }))
    );
    *h.confirm.answer.lock().unwrap() = true;
    // A remote conversation, then a second folder: asked again.
    let ws = h.workspace();
    h.script(vec![say("one")]);
    let id = remote_agent(&h, None, "hi", &ws);
    h.turns_end(&id, 1);
    let other = h.scratch.repo("other");
    let view = h
        .runtime
        .block_on(h.chat.attach_native(Some(&id), other))
        .unwrap();
    h.runtime
        .block_on(h.chat.trust(&view.workspace.id))
        .unwrap();
    h.script(vec![say("two")]);
    remote_agent(&h, Some(&id), "again", &view.workspace.id);
    h.turns_end(&id, 2);
    assert_eq!(first_remote_asks(&h).len(), 2, "the new folder asked again");
}

/// TF15's APPDATA half: a folder inside `%APPDATA%` is refused.
#[test]
fn tf15_appdata_is_refused() {
    let roaming = crate::testkit::TempDir::new("tf15-roaming");
    let inside = roaming.path().join("app");
    std::fs::create_dir_all(&inside).unwrap();
    let h = H::with(
        "tf15-appdata",
        &[("APPDATA", roaming.path().to_str().unwrap())],
        |_| {},
    );
    let refused = h
        .runtime
        .block_on(h.chat.attach_native(None, inside))
        .unwrap_err();
    assert_eq!(refused.kind, RefusalKind::Invalid, "{}", refused.message);
}

/// TF6, TF7 and PF13 on agent turns: Local and Auto with a workspace, in
/// PF15's three states, only ever build local clients (and one per turn);
/// the hosted endpoint's client is not local.
/// Mutant: the managed server's client built without `.local(true)`.
#[test]
fn tf6_pf13_pf15_local_and_auto_agent_turns_build_only_local_clients() {
    let states: [(&str, &[(&str, &str)]); 3] = [
        (
            "remote OLLAMA_BASE_URL",
            &[("OLLAMA_BASE_URL", "http://198.51.100.7:11434")],
        ),
        (
            "a cloud model on a loopback daemon",
            &[
                ("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                ("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
            ],
        ),
        ("nothing more", &[]),
    ];
    for (what, extra) in states {
        let h = H::with("tf6", extra, |_| {});
        let ws = h.workspace();
        for (choice, label) in [("local", "Local"), ("auto", "Auto")] {
            h.script(vec![
                call("read_file", json!({"path": "a.txt"}), "c1"),
                call("read_file", json!({"path": "a.txt"}), "c2"),
                say("ok"),
            ]);
            let before = h.asked().len();
            let id = conversation(
                h.send(
                    None,
                    "q",
                    choice,
                    Shown {
                        locality: Locality::Local,
                        label: label.into(),
                    },
                    Mode::Agent,
                    Some(&ws),
                )
                .unwrap(),
            );
            h.turns_end(&id, 1);
            let asked = h.asked();
            assert_eq!(asked.len(), before + 1, "{what}: one client per turn (TF7)");
            let last = asked.last().unwrap();
            assert!(last.local, "{what}: {choice}");
            assert!(
                last.base_url.starts_with("http://127.0.0.1:"),
                "{what}: {}",
                last.base_url
            );
        }
        h.script(vec![say("far")]);
        let id = remote_agent(&h, None, "q", &ws);
        h.turns_end(&id, 1);
        assert!(
            !h.asked().last().unwrap().local,
            "PF13: the endpoint is not local"
        );
    }
}

/// PF1 and PF2 on agent turns: with no local server installed, Local and
/// Auto call no model at all (a hosted endpoint is ready), and say why.
#[test]
fn pf1_pf2_agent_turns_with_no_local_server_call_nothing() {
    let h = H::with(
        "pf1-agent",
        &[("OLLAMA_BASE_URL", "http://198.51.100.7:11434")],
        |_| {},
    );
    let paths = crate::llama::files::LlamaPaths::from_env(&h.env);
    std::fs::remove_file(&paths.binary).unwrap();
    let ws = h.workspace();
    h.script(vec![
        call("read_file", json!({"path": "a.txt"}), "c1"),
        say("never"),
    ]);
    for (choice, label, sentence) in [
        ("local", "Local", "not ready"),
        ("auto", "Auto", "No model on this machine is ready."),
    ] {
        let id = conversation(
            h.send(
                None,
                "q",
                choice,
                Shown {
                    locality: Locality::Local,
                    label: label.into(),
                },
                Mode::Agent,
                Some(&ws),
            )
            .unwrap(),
        );
        let events = h.turns_end(&id, 1);
        assert_eq!(statuses(&events), [TurnStatus::Failed]);
        let message = events
            .iter()
            .find_map(|event| match &event.kind {
                ConversationEventKind::Error { message } => Some(message.clone()),
                _ => None,
            })
            .unwrap();
        assert!(message.contains(sentence), "{choice}: {message}");
    }
    assert!(h.asked().is_empty(), "zero model calls");
}

/// PF14 on an agent turn: a choice shown as Local that resolves elsewhere is
/// refused with `Conflict`, before anything is written or asked.
#[test]
fn pf14_a_choice_that_moved_is_refused_before_anything() {
    let h = H::new("pf14-agent");
    let ws = h.workspace();
    h.script(vec![say("never")]);
    let before = h.snapshot();
    let refusal = h
        .send(
            None,
            "q",
            "endpoint:hosted",
            Shown {
                locality: Locality::Local,
                label: "Hosted".into(),
            },
            Mode::Agent,
            Some(&ws),
        )
        .unwrap_err();
    assert_eq!(refusal.kind, RefusalKind::Conflict);
    assert_eq!(h.snapshot(), before);
    assert!(h.asked().is_empty());
    assert!(first_remote_asks(&h).is_empty(), "no dialog either");
}

/// N4 for an agent turn and a steer: a secret in the question bound for a
/// remote target is refused before anything; a steer holding one is
/// refused with the N4 sentence.
#[test]
fn n4_a_secret_in_the_words_never_goes_remote() {
    let h = H::new("n4-agent");
    let ws = h.workspace();
    h.script(vec![say("never")]);
    let key = secret();
    let refusal = h
        .send(
            None,
            &format!("my key {key}"),
            "endpoint:hosted",
            hosted(),
            Mode::Agent,
            Some(&ws),
        )
        .unwrap_err();
    assert!(refusal.message.starts_with("Not sent:"));
    assert!(h.asked().is_empty());
    h.script(vec![
        call("ask_question", json!({"question": "Wait"}), "q1"),
        say("done"),
    ]);
    let id = remote_agent(&h, None, "start", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::Question { .. }))
    });
    let refused = h
        .runtime
        .block_on(h.chat.steer(&id, format!("use {key}")))
        .unwrap_err();
    assert!(refused.message.starts_with("Not sent:"));
    h.chat.answer(&id, "q1", "go".into()).unwrap();
    let events = h.turns_end(&id, 1);
    assert_eq!(ended(&events), 1);
}

/// T1 (§10.3): the first send after a conversation's remote label changes
/// asks FirstRemoteSend again, naming the new label. A result released
/// ("Send this once") for one remote endpoint (Hosted) does not reach
/// another (Other) the conversation moves to. The verifier's
/// probe_released_secret_follows_a_label_change, as a test.
/// Mutant: the label left out of the "already confirmed" check
/// (convo/agent.rs; the verifier's m9, which no test caught).
#[test]
fn t1_a_changed_remote_label_asks_first_remote_send_again() {
    const TWO: &str = r#"{"version": 1, "endpoints": [
    {"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1",
     "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true},
    {"id": "other", "label": "Other", "base_url": "https://other.example.test/v1",
     "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true}
]}"#;
    let h = H::new("t1-label");
    let key = secret();
    std::fs::write(h.folder.join("key.txt"), format!("TOKEN={key}\n")).unwrap();
    let ws = h.workspace();
    let first = h.script(vec![
        call("read_file", json!({"path": "key.txt"}), "c_read"),
        call("ask_question", json!({"question": "Send it?"}), "q1"),
        say("done"),
    ]);
    let id = remote_agent(&h, None, "look", &ws);
    h.events_until(&id, |events| {
        events
            .iter()
            .any(|event| matches!(event.kind, ConversationEventKind::Question { .. }))
    });
    h.runtime
        .block_on(h.chat.decide(&id, "c_read", Decision::ReleaseWithheld))
        .unwrap();
    h.chat.answer(&id, "q1", "yes".into()).unwrap();
    h.turns_end(&id, 1);
    assert!(
        format!("{:?}", first.calls().last()).contains(&key),
        "positive control: released to Hosted"
    );
    std::fs::write(h.state.globals.join("model_endpoints.json"), TWO).unwrap();
    let second = h.script(vec![say("seen")]);
    let other = Shown {
        locality: Locality::Remote,
        label: "Other".into(),
    };
    h.send(
        Some(&id),
        "continue",
        "endpoint:other",
        other,
        Mode::Agent,
        Some(&ws),
    )
    .unwrap();
    h.turns_end(&id, 2);
    let labels: Vec<String> = h
        .confirm
        .asked()
        .into_iter()
        .filter_map(|request| match request {
            ConfirmRequest::FirstRemoteSend { label, .. } => Some(label),
            _ => None,
        })
        .collect();
    assert_eq!(labels, ["Hosted", "Other"], "asked again for the new label");
    assert!(
        !format!("{:?}", second.calls()).contains(&key),
        "the result released for Hosted did not reach Other"
    );
}

fn assistant_texts(request: &lattice_agents::model::ModelRequest) -> Vec<(String, bool)> {
    request
        .input
        .iter()
        .filter_map(|item| match item {
            lattice_agents::model::InputItem::Assistant { text, tool_calls } => text
                .as_ref()
                .map(|text| (text.clone(), !tool_calls.is_empty())),
            _ => None,
        })
        .collect()
}

/// RP4 (c) (Amendment 4, RP3): an agent turn's reasoning, a provider's
/// separate reasoning item before a tool call and a `<think>` span in the
/// final message, is recorded in the sidecar, redacted (T15), and appears in
/// the next turn's model input, on Local and on a remote target, as
/// `<think>` text at the start of the assistant message it preceded; the
/// visible answer, the events and the shared store never hold it.
/// Mutants: provider reasoning not recorded (m4); `<think>` spans not
/// recorded (m5); recorded reasoning not replayed (m6); the visible answer
/// not think-stripped (m7); the reasoning record written verbatim (m8).
#[test]
fn rp4c_reasoning_is_recorded_and_replayed_but_never_shown() {
    use lattice_agents::model::OutputItem;
    use lattice_agents::testing::{ScriptedStep, assistant_message, function_call_json};

    let h = H::new("rp4c");
    let key = secret();
    std::fs::write(h.folder.join("notes.txt"), "NOTES\n").unwrap();
    let ws = h.workspace();
    h.script(vec![
        ScriptedStep::respond(vec![
            OutputItem::Reasoning {
                text: format!("PLAN-R1 the key is {key}"),
            },
            function_call_json("read_file", &json!({"path": "notes.txt"}), "c_read"),
        ]),
        ScriptedStep::respond(vec![assistant_message(
            "<think>INLINE-R2</think>Answer one.",
        )]),
    ]);
    let id = h.agent(None, "read notes", &ws);
    h.turns_end(&id, 1);
    let free = |text: &str| !text.contains("PLAN-R1") && !text.contains("INLINE-R2");
    assert_eq!(
        h.texts(&id),
        ["read notes", "Answer one."],
        "the visible answer"
    );
    // Recorded, redacted, in order.
    let recorded: Vec<String> = h
        .sidecar_items(&id)
        .into_iter()
        .filter_map(|item| match item {
            super::item::Item::Reasoning {
                text: super::item::Payload::Inline(text),
                ..
            } => Some(text),
            _ => None,
        })
        .collect();
    assert_eq!(recorded.len(), 2, "{recorded:?}");
    assert!(recorded[0].starts_with("PLAN-R1") && recorded[0].contains(crate::secrets::REDACTED));
    assert_eq!(recorded[1], "INLINE-R2");
    for (path, bytes) in files(&h.state.root) {
        assert!(
            !String::from_utf8_lossy(&bytes).contains(&key),
            "{path} holds the key"
        );
    }
    // The next turn, on Local: replayed with the output it preceded.
    let local = h.script(vec![say("two")]);
    h.agent(Some(&id), "again", &ws);
    h.turns_end(&id, 2);
    // Then on a remote target.
    let remote = h.script(vec![say("three")]);
    remote_agent(&h, Some(&id), "remotely", &ws);
    let events = h.turns_end(&id, 3);
    for (target, model) in [("local", &local), ("remote", &remote)] {
        let request = &model.calls()[0];
        let texts = assistant_texts(request);
        assert!(
            texts.iter().any(|(text, calls)| *calls
                && text.starts_with("<think>")
                && text.contains("PLAN-R1")
                && text.contains(crate::secrets::REDACTED)),
            "{target}: the reasoning before the call: {texts:?}"
        );
        assert!(
            texts.iter().any(|(text, calls)| !*calls
                && text.starts_with("<think>")
                && text.contains("INLINE-R2")
                && text.ends_with("Answer one.")),
            "{target}: the reasoning with the answer: {texts:?}"
        );
        assert!(!format!("{request:?}").contains(&key), "{target}: redacted");
    }
    // Never shown, never in the shared store.
    assert!(free(&format!("{events:?}")), "an event held reasoning");
    assert!(free(&h.snapshot()), "the shared store held reasoning");
    assert_eq!(
        h.texts(&id),
        [
            "read notes",
            "Answer one.",
            "again",
            "two",
            "remotely",
            "three"
        ]
    );
}

/// A model that streams its pieces and then never finishes.
struct Unfinished(Vec<&'static str>);

impl lattice_agents::Model for Unfinished {
    fn name(&self) -> &str {
        "unfinished"
    }

    fn config_for_trace(&self) -> serde_json::Value {
        json!({})
    }

    fn stream(
        &self,
        _request: lattice_agents::model::ModelRequest,
    ) -> futures::stream::BoxStream<
        'static,
        Result<lattice_agents::model::ModelEvent, lattice_agents::model::ModelError>,
    > {
        use futures::StreamExt;
        let pieces: Vec<_> = self
            .0
            .iter()
            .map(|piece| {
                Ok(lattice_agents::model::ModelEvent::TextDelta(
                    (*piece).into(),
                ))
            })
            .collect();
        futures::stream::iter(pieces)
            .chain(futures::stream::pending())
            .boxed()
    }
}

/// RP3, a stopped turn: the reasoning of a message that never finished
/// (no whole message arrives, only its streamed text) is recorded when the
/// turn stops, and the saved partial answer is free of it.
/// Mutant: the stop path records nothing (m9).
#[test]
fn rp3_a_stopped_turns_unfinished_reasoning_is_recorded() {
    let h = H::new("rp3-stop");
    let ws = h.workspace();
    h.answer_with(std::sync::Arc::new(Unfinished(vec![
        "<think>PARTIAL-",
        "R3</think>",
        "visible",
    ])));
    let id = h.agent(None, "start", &ws);
    h.events_until(&id, |events| {
        events.iter().any(
            |event| matches!(&event.kind, ConversationEventKind::Delta { text } if text.contains("visible")),
        )
    });
    assert!(h.chat.stop(&id));
    let events = h.turns_end(&id, 1);
    assert_eq!(statuses(&events), [TurnStatus::Stopped]);
    assert_eq!(h.texts(&id), ["start", "visible"]);
    let recorded: Vec<super::item::Item> = h
        .sidecar_items(&id)
        .into_iter()
        .filter(|item| matches!(item, super::item::Item::Reasoning { .. }))
        .collect();
    assert!(
        matches!(
            recorded.as_slice(),
            [super::item::Item::Reasoning { text: super::item::Payload::Inline(text), .. }]
                if text == "PARTIAL-R3"
        ),
        "{recorded:?}"
    );
    assert!(!format!("{events:?}").contains("PARTIAL-R3"));
}

/// RP3's replay half: each reasoning item goes, as one `<think>` block, at
/// the start of the assistant message that followed it (a tool-call message
/// or the visible answer); reasoning after a turn's last call with no
/// visible answer is left out; the budget leaves older reasoning out with
/// older outputs.
/// Mutant: replayed reasoning dropped (m6, with rp4c).
#[test]
fn reasoning_replays_with_the_assistant_output_it_preceded() {
    use super::item::{Item, Payload};
    use super::replay::{Options, replay, think_block};
    use crate::chat::store::StoredTurn;
    let turn = |id: &str, role: &str, text: &str| StoredTurn {
        id: id.into(),
        ts: 1.0,
        role: role.into(),
        text: text.into(),
        tools: vec![],
        facts: vec![],
        unsupported: vec![],
        provider: String::new(),
        error: String::new(),
        constrained: false,
        truncated: false,
        cancelled: false,
        prompt_tokens: None,
        completion_tokens: None,
        superseded: false,
    };
    let reason = |turn: &str, text: &str| Item::Reasoning {
        turn: turn.into(),
        text: Payload::Inline(text.into()),
        at: 1.0,
    };
    let call = |turn: &str, id: &str| Item::ToolCall {
        turn: turn.into(),
        call_id: id.into(),
        tool: "read_file".into(),
        arguments: Payload::Inline("{}".into()),
        summary: "Read".into(),
        at: 1.0,
    };
    let result = |turn: &str, id: &str| Item::ToolResult {
        turn: turn.into(),
        call_id: id.into(),
        output: Payload::Inline("OUT".into()),
        withheld: false,
        truncated: false,
        at: 1.0,
        output_blob: None,
    };
    let turns = [
        turn("u1", "user", "one"),
        turn("a1", "assistant", "Answer one."),
        turn("u2", "user", "two"),
        turn("u3", "user", "three"),
    ];
    let items = [
        reason("u1", "R-BEFORE-CALL"),
        call("u1", "c1"),
        result("u1", "c1"),
        reason("u1", "R-AFTER-A"),
        reason("u1", "R-AFTER-B"),
        // u2 has no visible answer: its closing reasoning goes nowhere.
        call("u2", "c2"),
        result("u2", "c2"),
        reason("u2", "R-ORPHAN"),
    ];
    let options = Options {
        stop_before: Some("u3".into()),
        ..Options::default()
    };
    let replayed = replay(&turns, &items, &options, &|_| None);
    let texts: Vec<(Option<String>, usize)> = replayed
        .items
        .iter()
        .filter_map(|item| match item {
            lattice_agents::model::InputItem::Assistant { text, tool_calls } => {
                Some((text.clone(), tool_calls.len()))
            }
            _ => None,
        })
        .collect();
    let block = |parts: &[&str]| {
        think_block(
            &parts
                .iter()
                .map(|part| (*part).to_owned())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    };
    assert_eq!(
        texts,
        [
            (Some(block(&["R-BEFORE-CALL"])), 1),
            (Some(block(&["R-AFTER-A", "R-AFTER-B"]) + "Answer one."), 0),
            (None, 1),
        ]
    );
    assert!(!format!("{:?}", replayed.items).contains("R-ORPHAN"));
    assert_eq!(block(&["x"]), "<think>\nx\n</think>\n\n");
    // The budget: older reasoning is left out with older outputs; the last
    // two agent turns keep theirs.
    let mut many = vec![turn("u0", "user", "zero"), turn("a0", "assistant", "fine")];
    many.extend(turns.iter().cloned());
    let mut long = vec![
        reason("u0", &"R-OLD ".repeat(4_000)),
        call("u0", "c0"),
        result("u0", "c0"),
    ];
    long.extend(items.iter().cloned());
    let tight = replay(
        &many,
        &long,
        &Options {
            stop_before: Some("u3".into()),
            context_tokens: Some(4_000),
        },
        &|_| None,
    );
    let text = format!("{:?}", tight.items);
    assert!(!text.contains("R-OLD"), "older reasoning left out");
    assert!(text.contains("R-BEFORE-CALL") && text.contains("R-AFTER-B"));
}
