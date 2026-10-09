//! Replay (UT1, §4.6, §5.4, §5.7.2, T1), the tripwire (T3) and the agent
//! prompt's golden.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use lattice_agents::Model;
use lattice_agents::model::{InputItem, ModelRequest, ModelSettings, ToolCallItem};
use lattice_agents::session::Session;
use lattice_agents::testing::{ScriptedModel, assistant_message};
use lattice_protocol::conversation::{Mode, TurnKind};
use lattice_protocol::{Locality, Shown};

use super::item::{Item, Payload};
use super::prompt_agent;
use super::replay::*;
use super::tripwire::{Tripwire, TripwireModel, WITHHELD, Withheld};
use crate::chat::store::StoredTurn;

fn turn(id: &str, role: &str, text: &str) -> StoredTurn {
    StoredTurn {
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
    }
}

fn start(turn: &str, resolved: Locality) -> Item {
    Item::TurnStart {
        turn: turn.into(),
        mode: Mode::Agent,
        kind: TurnKind::Agent,
        choice: "local".into(),
        shown: Shown {
            locality: resolved,
            label: "Local".into(),
        },
        resolved,
        label: "Local".into(),
        at: 1.0,
    }
}

fn call(turn: &str, id: &str, tool: &str, arguments: &str) -> Item {
    Item::ToolCall {
        turn: turn.into(),
        call_id: id.into(),
        tool: tool.into(),
        arguments: Payload::Inline(arguments.into()),
        summary: format!("{tool} summary"),
        at: 1.0,
    }
}

fn result(turn: &str, id: &str, output: &str) -> Item {
    Item::ToolResult {
        turn: turn.into(),
        call_id: id.into(),
        output: Payload::Inline(output.into()),
        withheld: false,
        truncated: false,
        at: 1.0,
        output_blob: None,
    }
}

fn no_blob(_: &str) -> Option<Vec<u8>> {
    None
}

fn user_texts(items: &[InputItem]) -> Vec<&str> {
    items
        .iter()
        .filter_map(|item| match item {
            InputItem::User(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// UT1 and §5.7.2 step 2: replay stops before the current turn's user
/// record; each turn's tool calls are one assistant item, then their
/// results, then its answer.
/// Mutant: the replay includes the current user record.
#[test]
fn ut1_replay_stops_before_the_current_user_record() {
    let turns = [
        turn("u1", "user", "first question"),
        turn("a1", "assistant", "first answer"),
        turn("u2", "user", "the question now"),
    ];
    let items = [
        start("u1", Locality::Local),
        call("u1", "c1", "read_file", r#"{"path":"a.txt"}"#),
        call("u1", "c2", "grep", r#"{"pattern":"x"}"#),
        result("u1", "c1", "a.txt contents"),
        result("u1", "c2", "no matches"),
        start("u2", Locality::Local),
    ];
    let replay = replay(
        &turns,
        &items,
        &Options {
            stop_before: Some("u2".into()),
            ..Options::default()
        },
        &no_blob,
    );
    assert_eq!(user_texts(&replay.items), ["first question"]);
    assert_eq!(
        replay.items,
        [
            InputItem::User("first question".into()),
            InputItem::Assistant {
                text: None,
                tool_calls: vec![
                    ToolCallItem {
                        call_id: "c1".into(),
                        name: "read_file".into(),
                        arguments: r#"{"path":"a.txt"}"#.into()
                    },
                    ToolCallItem {
                        call_id: "c2".into(),
                        name: "grep".into(),
                        arguments: r#"{"pattern":"x"}"#.into()
                    },
                ],
            },
            InputItem::ToolResult {
                call_id: "c1".into(),
                output: "a.txt contents".into()
            },
            InputItem::ToolResult {
                call_id: "c2".into(),
                output: "no matches".into()
            },
            InputItem::Assistant {
                text: Some("first answer".into()),
                tool_calls: vec![]
            },
        ]
    );
}

/// §5.4: a superseded turn (not visible) hides its sidecar items; nothing is
/// removed from the record.
#[test]
fn a_superseded_turn_hides_its_tool_trail() {
    let turns = [
        turn("u2", "user", "asked again"),
        turn("a2", "assistant", "ok"),
    ];
    let items = [
        start("u1", Locality::Local),
        call("u1", "c1", "read_file", "{}"),
        result("u1", "c1", "old trail"),
        start("u2", Locality::Local),
    ];
    let replay = replay(&turns, &items, &Options::default(), &no_blob);
    assert!(!format!("{:?}", replay.items).contains("old trail"));
    assert_eq!(user_texts(&replay.items), ["asked again"]);
}

/// §4.6 and AF9's replay half: a call with no result is answered once with
/// "closed before deciding"; Continue's input ends at the last result.
#[test]
fn a_call_left_waiting_is_answered_once_and_continue_ends_at_it() {
    let turns = [turn("u1", "user", "run it")];
    let items = [
        start("u1", Locality::Local),
        call("u1", "c1", "run_command", r#"{"command":"cargo test"}"#),
    ];
    let replay = replay(&turns, &items, &Options::default(), &no_blob);
    let closed: Vec<&InputItem> = replay
        .items
        .iter()
        .filter(
            |item| matches!(item, InputItem::ToolResult { output, .. } if output == CLOSED_BEFORE),
        )
        .collect();
    assert_eq!(closed.len(), 1);
    let resumed = through_last_result(replay.items.clone());
    assert!(
        matches!(resumed.last(), Some(InputItem::ToolResult { call_id, .. }) if call_id == "c1")
    );
    assert_eq!(user_texts(&resumed), ["run it"], "no new user item");
}

/// Amendment 4 RP2 (replacing TF14's replay half): tool items read under a
/// local model replay in full to every target; the count FirstRemoteSend
/// states is theirs. (Locality is no longer an input to the replay: the
/// same items go to a local and a remote target.)
#[test]
fn local_tool_items_replay_in_full_and_are_counted() {
    let turns = [
        turn("u1", "user", "look"),
        turn("a1", "assistant", "seen"),
        turn("u2", "user", "and now remotely"),
    ];
    let items = [
        start("u1", Locality::Local),
        call("u1", "c1", "edit_file", r#"{"new_string":"LOCAL-BYTES"}"#),
        result("u1", "c1", "LOCAL-CONTENT"),
    ];
    assert_eq!(local_results(&turns, &items, Some("u2")), 1);
    let sent = replay(
        &turns,
        &items,
        &Options {
            stop_before: Some("u2".into()),
            ..Options::default()
        },
        &no_blob,
    );
    let text = format!("{:?}", sent.items);
    assert!(
        text.contains("LOCAL-CONTENT") && text.contains("LOCAL-BYTES"),
        "{text}"
    );
}

/// §5.7.2 step 3: over the budget, older results are elided first (the last
/// two agent turns kept whole), then older tool items, then the oldest
/// exchanges, which the lead item counts.
#[test]
fn the_budget_elides_older_output_first_and_keeps_the_last_two_agent_turns() {
    assert_eq!(budget(None), 96_000);
    assert_eq!(budget(Some(8_192)), 24_576);
    assert_eq!(budget(Some(1_000_000)), 96_000);
    let mut turns = Vec::new();
    let mut items = Vec::new();
    for n in 0..5 {
        let id = format!("u{n}");
        turns.push(turn(&id, "user", &format!("question {n}")));
        turns.push(turn(&format!("a{n}"), "assistant", "answer"));
        items.push(start(&id, Locality::Local));
        items.push(call(&id, &format!("c{n}"), "read_file", "{}"));
        items.push(result(&id, &format!("c{n}"), &"x".repeat(3_000)));
    }
    let small = replay(
        &turns,
        &items,
        &Options {
            context_tokens: Some(2_500),
            ..Options::default()
        },
        &no_blob,
    );
    let outputs: Vec<&str> = small
        .items
        .iter()
        .filter_map(|item| match item {
            InputItem::ToolResult { output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect();
    println!("{outputs:?}");
    assert_eq!(outputs.len(), 5);
    assert!(
        outputs[..3]
            .iter()
            .all(|o| o.starts_with("[output elided: read_file read_file summary, 3000 bytes]"))
    );
    assert!(
        outputs[3..].iter().all(|o| o.len() == 3_000),
        "the last two kept whole"
    );
    let tiny = replay(
        &turns,
        &items,
        &Options {
            context_tokens: Some(1_500),
            ..Options::default()
        },
        &no_blob,
    );
    assert!(tiny.left_out > 0, "{tiny:?}");
    assert_eq!(
        lead_item(None, tiny.left_out, &[]).unwrap(),
        format!(
            "{} earlier exchanges of this conversation were left out to fit the model's context.",
            tiny.left_out
        )
    );
}

#[test]
fn the_lead_item_holds_rules_then_omissions_then_notes() {
    assert_eq!(lead_item(None, 0, &[]), None);
    let lead = lead_item(
        Some("Project rules (x):\n\nAGENTS.md\nbe kind"),
        2,
        &["Kept a.txt.".into()],
    )
    .unwrap();
    assert!(lead.starts_with("Project rules"));
    assert!(lead.ends_with("Kept a.txt."));
}

/// The session hands the replay to the run and persists nothing.
#[test]
fn the_sidecar_session_reads_the_replay_and_writes_nothing() {
    let session = SidecarSession::new(vec![
        InputItem::User("a".into()),
        InputItem::User("b".into()),
    ]);
    let all = futures::executor::block_on(session.get_items(None)).unwrap();
    assert_eq!(all.len(), 2);
    let last = futures::executor::block_on(session.get_items(Some(1))).unwrap();
    assert_eq!(last, [InputItem::User("b".into())]);
    futures::executor::block_on(session.add_items(vec![InputItem::User("c".into())])).unwrap();
    assert_eq!(
        futures::executor::block_on(session.get_items(None))
            .unwrap()
            .len(),
        2
    );
}

// ----------------------------------------------------------------- T3

#[derive(Default)]
struct Reported(Mutex<Vec<String>>);

impl Withheld for Reported {
    fn withheld(&self, call_id: &str) {
        self.0.lock().unwrap().push(call_id.to_owned());
    }
}

fn secret() -> String {
    format!("sk-{}", "q7".repeat(12))
}

fn request(key: &str) -> ModelRequest {
    ModelRequest {
        system: String::new(),
        input: vec![
            InputItem::User(format!("user says {key}")),
            InputItem::Assistant {
                text: Some(format!("I saw {key}")),
                tool_calls: vec![ToolCallItem {
                    call_id: "c_edit".into(),
                    name: "edit_file".into(),
                    arguments: format!(r#"{{"new_string":"{key}"}}"#),
                }],
            },
            InputItem::ToolResult {
                call_id: "c_read".into(),
                output: format!("KEY={key}"),
            },
            InputItem::ToolResult {
                call_id: "c_plain".into(),
                output: "nothing here".into(),
            },
        ],
        tools: vec![],
        settings: ModelSettings::default(),
    }
}

/// TF1's unit half: every string of every item is checked; each match is
/// withheld on its own, reported once per call, and never repeated.
/// Mutants: the wrapper checks User items only; it skips tool-call arguments.
#[test]
fn tf1_the_tripwire_withholds_every_item_that_looks_like_a_secret() {
    let key = secret();
    let reported = Arc::new(Reported::default());
    let wire = Arc::new(Tripwire::new(reported.clone()));
    let inner = Arc::new(ScriptedModel::new([
        vec![assistant_message("ok")],
        vec![assistant_message("ok")],
    ]));
    let model = TripwireModel::new(inner.clone(), wire.clone());
    for _ in 0..2 {
        let events: Vec<_> =
            futures::executor::block_on(model.stream(request(&key)).collect::<Vec<_>>());
        assert!(events.iter().all(Result::is_ok));
    }
    let sent = format!("{:?}", inner.calls());
    assert!(!sent.contains(&key), "{sent}");
    assert!(sent.contains("nothing here"), "a clean result is untouched");
    let calls = inner.calls();
    let input = &calls[0].input;
    assert_eq!(input[0], InputItem::User(WITHHELD.into()));
    match &input[1] {
        InputItem::Assistant { text, tool_calls } => {
            assert_eq!(text.as_deref(), Some(WITHHELD));
            assert!(tool_calls[0].arguments.contains(WITHHELD));
            assert_eq!(tool_calls[0].call_id, "c_edit");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        *reported.0.lock().unwrap(),
        ["c_edit", "c_read"],
        "once per call, never the text"
    );
    assert_eq!(wire.withheld_calls(), ["c_edit", "c_read"]);
}

/// TF3's unit half: a release lets that call's withheld output through; a
/// new match of the same call is withheld again.
#[test]
fn tf3_a_release_sends_that_output_and_a_new_match_is_withheld_again() {
    let key = secret();
    let wire = Tripwire::new(Arc::new(Reported::default()));
    let first = wire.screen(request(&key));
    assert!(!format!("{first:?}").contains(&key));
    assert!(
        !wire.release("c_nothing"),
        "nothing of that call is withheld"
    );
    assert!(wire.release("c_read"));
    let second = wire.screen(request(&key));
    let text = format!("{second:?}");
    assert!(text.contains(&format!("KEY={key}")), "released: sent");
    let other = format!("sk-{}", "z9".repeat(12));
    let mut third = request(&key);
    third.input[2] = InputItem::ToolResult {
        call_id: "c_read".into(),
        output: format!("KEY={other}"),
    };
    let third = wire.screen(third);
    assert!(
        !format!("{third:?}").contains(&other),
        "a new match is withheld again"
    );
    assert_eq!(wire.withheld_calls(), ["c_edit", "c_read"]);
}

/// §7.1: the prompt and the tools' descriptions, byte for byte with the
/// native golden (line ends normalised); Ask mode offers no staging tool and
/// no run_command.
#[test]
fn the_agent_prompt_matches_its_golden() {
    let golden = include_str!("../../tests/goldens/agent_prompt.txt").replace("\r\n", "\n");
    assert_eq!(prompt_agent::render(), golden);
    let ask: Vec<&str> = prompt_agent::tools(Mode::Ask)
        .iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(
        ask,
        [
            "list_dir",
            "glob",
            "read_file",
            "grep",
            "ask_question",
            "remember",
            "forget",
            "spawn_agent",
            "suggest_task",
            "withdraw_task",
            "write_artifact",
            "read_artifact",
            "update_todos",
            "propose_plan"
        ]
    );
    let agent: Vec<&str> = prompt_agent::tools(Mode::Agent)
        .iter()
        .map(|t| t.name)
        .collect();
    assert_eq!(agent.len(), 19);
    assert!(prompt_agent::SYSTEM.contains("PowerShell"));
    assert!(!prompt_agent::SYSTEM.contains("Project rules"));
}
