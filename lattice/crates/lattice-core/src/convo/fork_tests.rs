//! `staged_diff` and forking (`super::fork`): the changes staged in a conversation read as diffs, and a fork holding
//! a conversation's messages up to a turn, in its folder and group.

use lattice_agents::model::{InputItem, ModelRequest};
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::groups::Placement;

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

/// Mutants: a staged change not shown; a file that is not staged shown; a new file's lines not marked added.
#[test]
fn staged_changes_read_as_diffs_that_git_diff_cannot_see() {
    let h = H::new("staged-diff");
    let ws = h.workspace();
    let model = h.script(vec![
        call("read_file", json!({"path": "a.txt"}), "r1"),
        call("write_file", json!({"path": "a.txt", "content": "b\n"}), "w1"),
        call("write_file", json!({"path": "new.txt", "content": "fresh\n"}), "w2"),
        call("staged_diff", json!({}), "d1"),
        call("staged_diff", json!({"path": "new.txt"}), "d2"),
        call("staged_diff", json!({"path": "other.txt"}), "d3"),
        say("Reviewed."),
    ]);
    let id = h.agent(None, "change a.txt and review it", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    let all = result_of(&calls, "d1");
    assert!(all.starts_with("2 staged changes waiting for review"), "{all}");
    assert!(all.contains("--- a/a.txt") && all.contains("+++ b/a.txt") && all.contains("+b"), "{all}");
    assert!(all.contains("+++ b/new.txt") && all.contains("+fresh"), "{all}");
    let one = result_of(&calls, "d2");
    assert!(one.contains("new.txt") && !one.contains("a.txt"), "{one}");
    assert_eq!(result_of(&calls, "d3"), "Nothing staged in this conversation is waiting at other.txt.");
}

/// Mutants: the fork keeping turns past the one chosen, or set-aside ones; not in the original's group; the original
/// changed; a fork from no message accepted.
#[test]
fn a_fork_holds_the_messages_up_to_a_turn_in_the_same_group() {
    let h = H::new("fork");
    let ws = h.workspace();
    h.script(vec![say("First answer."), say("Second answer.")]);
    let id = h.agent(None, "First question.", &ws);
    h.turns_end(&id, 1);
    h.agent(Some(&id), "Second question.", &ws);
    h.turns_end(&id, 2);
    let group = h.chat.group_add(&id, &Placement::New, None).unwrap();

    let forked = h.chat.fork(&id, 2).unwrap();
    let texts: Vec<String> = h.store.load(&forked).into_iter().map(|t| t.text).collect();
    assert_eq!(texts, ["First question.", "First answer."]);
    assert_eq!(h.store.load(&id).len(), 4, "the original is as it was");
    let (groups, _) = h.chat.groups();
    let shown = groups.iter().find(|(g, _)| g.id == group.id).unwrap();
    assert!(shown.0.topics.contains(&forked), "the fork is in the original's group");
    assert!(h.chat.fork(&id, 0).is_err(), "no message to fork from");
}

fn user_texts(request: &ModelRequest) -> Vec<String> {
    request
        .input
        .iter()
        .filter_map(|item| match item {
            InputItem::User(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

/// `/compact`: the summary is written now with what to keep above all; the next turn leaves the compacted turns out
/// and starts from the summary; the context meter says so.
/// Mutants: the focus not asked for; compacted turns still replayed; the summary not leading; no meter after a turn.
#[test]
fn compacting_replaces_the_earlier_turns_with_their_summary() {
    let h = H::new("compact");
    let ws = h.workspace();
    let model = h.script(vec![
        say("First answer."),
        say("Second answer."),
        say("COMPACTED: the parser reads CSV with quoted fields."),
        say("Third answer."),
    ]);
    let id = h.agent(None, "FIRST-QUESTION about the parser.", &ws);
    h.turns_end(&id, 1);
    h.agent(Some(&id), "SECOND-QUESTION about quotes.", &ws);
    h.turns_end(&id, 2);
    let before = h.chat.context_use(&id).expect("a meter after a turn");
    assert!(before.limit > 0 && before.used > 0 && before.percent() <= 100, "{before:?}");

    let text = h.runtime.block_on(h.chat.compact(&id, Some("CSV quoting".into()))).unwrap();
    assert_eq!(text, "COMPACTED: the parser reads CSV with quoted fields.");
    let asked = user_texts(&model.calls()[2]).join("\n");
    assert!(asked.contains("Keep above all: CSV quoting") && asked.contains("FIRST-QUESTION"), "{asked}");

    h.agent(Some(&id), "THIRD-QUESTION.", &ws);
    h.turns_end(&id, 3);
    let calls = model.calls();
    let replayed = user_texts(calls.last().unwrap()).join("\n");
    assert!(!replayed.contains("FIRST-QUESTION") && !replayed.contains("SECOND-QUESTION"), "{replayed}");
    assert!(replayed.contains("COMPACTED: the parser reads CSV"), "the summary leads: {replayed}");
    assert_eq!(h.chat.context_use(&id).unwrap().left_out, 2, "the two compacted exchanges");
    assert!(h.runtime.block_on(h.chat.compact("no-such-chat", None)).is_err());
}
