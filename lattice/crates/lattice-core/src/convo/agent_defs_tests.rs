//! Custom agents (`super::agent_defs`): a definition in the folder, the plugins or the reader's own, listed at the
//! start of a turn, and a helper sent as one with its instructions and only the tools it names.

use lattice_agents::model::{InputItem, ModelRequest};
use serde_json::json;

use super::agent_defs::{self, HEADER};
use super::agent_tests::{H, call, say};
use super::helpers::HELPER_SYSTEM;

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

fn write(path: &std::path::Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// Mutants: the definition not listed at the start of the turn; its instructions not given to the helper; its tools
/// not narrowed; an agent not listed accepted.
#[test]
fn a_helper_sent_as_a_folders_agent_has_its_instructions_and_only_its_tools() {
    let h = H::new("agent-defs");
    write(
        &h.folder.join(".claude").join("agents").join("reviewer.md"),
        "---\nname: reviewer\ndescription: Reviews code for bugs.\ntools: Read, Grep, Bash\n---\nREVIEWER-BODY: look for off-by-one errors.",
    );
    let ws = h.workspace();
    let model = h.script(vec![
        call("spawn_agent", json!({"task": "Review a.txt.", "agent": "Reviewer"}), "c1"),
        say("No bugs in a.txt."),
        call("spawn_agent", json!({"task": "Review it again.", "agent": "nope"}), "c2"),
        say("Done."),
    ]);
    let id = h.agent(None, "review the folder", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    let lead: String = calls[0]
        .input
        .iter()
        .filter_map(|i| match i {
            InputItem::User(text) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert!(lead.contains(HEADER) && lead.contains("- reviewer (.claude/agents/reviewer.md): Reviews code for bugs."), "{lead}");
    let helper = &calls[1];
    assert!(helper.system.starts_with(HELPER_SYSTEM) && helper.system.contains("REVIEWER-BODY"), "{}", helper.system);
    let mut tools: Vec<&str> = helper.tools.iter().map(|t| t.name.as_str()).collect();
    tools.sort_unstable();
    assert_eq!(tools, ["grep", "read_file"], "Bash names no helper tool, and a definition never widens");
    let refused = result_of(&calls, "c2");
    assert!(refused.ends_with("There is no agent named nope. The agents there are: reviewer."), "{refused}");
}

/// The reader's own and a definition's parts: a name from the file when none is given, Claude Code's tool names
/// mapped, no `tools` keeping every helper tool, and a second of one name left out with a notice.
#[test]
fn definitions_are_read_with_their_names_and_tools() {
    let h = H::new("agent-defs-own");
    let mine = agent_defs::user_dir(&h.state);
    write(&mine.join("tester.md"), "---\ndescription: Writes tests.\ntools: Read, Write, Edit\n---\nWrite the tests.");
    write(&mine.join("Tester2.md"), "Plain instructions only.");
    write(&mine.join("zz-dup.md"), "---\nname: tester\n---\nAgain.");
    let found = agent_defs::load(&h.state, None);
    let tester = found.find("tester").unwrap();
    assert_eq!(tester.tools.as_deref(), Some(&["read_file".to_string(), "write_file".to_string(), "edit_file".to_string()][..]));
    assert_eq!((tester.description.as_str(), tester.from.as_str()), ("Writes tests.", "yours"));
    let plain = found.find("Tester2").unwrap();
    assert_eq!((plain.tools.clone(), plain.description.as_str()), (None, "Plain instructions only."));
    assert!(found.notices.iter().any(|n| n.contains("an agent named tester is listed already")), "{:?}", found.notices);
    assert_eq!(found.agents.len(), 2);
}

/// A helper on another model: one off this computer is asked of the user first (the chat's own first-send dialog),
/// and refused, the helper does not start; a model Lattice cannot use is refused; `model` in a definition is read,
/// `inherit` meaning the turn's own.
/// Mutants: a remote helper model sent to without asking; a refusal ignored; `inherit` taken as a model's name.
#[test]
fn a_helper_on_a_model_off_this_computer_is_asked_of_the_user_first() {
    let h = H::new("helper-models");
    let ws = h.workspace();
    let model = h.script(vec![
        call("spawn_agent", json!({"task": "Count the files.", "model": "endpoint:hosted"}), "c1"),
        say("Three files."),
        call("spawn_agent", json!({"task": "Again.", "model": "no-such-model"}), "c2"),
        say("Done."),
    ]);
    let id = h.agent(None, "count with a cheaper helper", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(result_of(&calls, "c1").starts_with("Three files."), "{}", result_of(&calls, "c1"));
    let asked: Vec<_> = h
        .confirm
        .asked()
        .into_iter()
        .filter(|r| matches!(r, crate::ports::ConfirmRequest::FirstRemoteSend { conversation, .. } if *conversation == id))
        .collect();
    assert_eq!(asked.len(), 1, "the remote helper model was asked of the user: {asked:?}");
    assert!(result_of(&calls, "c2").contains("There is no model no-such-model a helper can use"));

    *h.confirm.answer.lock().unwrap() = false;
    model.enqueue(call("spawn_agent", json!({"task": "Once more.", "model": "endpoint:hosted"}), "c3"));
    model.enqueue(say("Refused."));
    h.agent(Some(&id), "again", &ws);
    h.turns_end(&id, 2);
    assert!(result_of(&model.calls(), "c3").contains("did not allow helpers to send to"), "{}", result_of(&model.calls(), "c3"));

    let mine = agent_defs::user_dir(&h.state);
    write(&mine.join("cheap.md"), "---\nmodel: local\n---\nCheap work.");
    write(&mine.join("same.md"), "---\nmodel: inherit\n---\nSame model.");
    let found = agent_defs::load(&h.state, None);
    assert_eq!(found.find("cheap").unwrap().model.as_deref(), Some("local"));
    assert_eq!(found.find("same").unwrap().model, None);
}
