//! The reader's hooks in agent turns (`super::hooked`), over the agent
//! harness with a scripted model and real hook commands run by the shell hooks
//! run in: a tool blocked before it runs, words added after it, the dialog asked
//! once per exact hook (a no runs nothing), a message blocked or given context,
//! and a Stop hook that has the agent go on once.

use std::path::Path;

use serde_json::json;

use super::agent_tests::{H, call, say, user_texts};
use crate::hooks::run;
use crate::ports::ConfirmRequest;

fn result_of(calls: &[lattice_agents::model::ModelRequest], call_id: &str) -> String {
    calls
        .iter()
        .flat_map(|request| request.input.iter())
        .find_map(|item| match item {
            lattice_agents::model::InputItem::ToolResult { call_id: id, output } if id == call_id => Some(output.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no result for {call_id}"))
}

/// The reader's own hooks file, in Claude Code's shape; `None` when no bash
/// runs hooks on this machine (the commands below are bash).
fn hooks(h: &H, events: serde_json::Value) -> Option<()> {
    let (shell, _) = run::shell(&h.env, &h.state.globals)?;
    if !shell.file_name().is_some_and(|n| n.eq_ignore_ascii_case("bash.exe")) {
        eprintln!("UNMEASURED: no bash to run hooks in");
        return None;
    }
    let file = crate::hooks::sources::lattice_file(&h.state);
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(file, json!({ "hooks": events }).to_string()).unwrap();
    Some(())
}

fn hook(command: &str) -> serde_json::Value {
    json!([{"hooks": [{"type": "command", "command": command, "timeout": 30}]}])
}

fn asked_hooks(h: &H) -> usize {
    h.confirm.asked().iter().filter(|r| matches!(r, ConfirmRequest::RunHook { .. })).count()
}

#[test]
fn a_tool_hook_blocks_before_and_adds_after_and_is_asked_for_once() {
    let h = H::new("hooks-tools");
    let ws = h.workspace();
    let Some(()) = hooks(
        &h,
        json!({
            "PreToolUse": [{"matcher": "Read", "hooks": [{"type": "command",
                "command": "grep -q secret && { echo 'not that file' >&2; exit 2; } || exit 0"}]}],
            "PostToolUse": [{"matcher": "LS", "hooks": [{"type": "command",
                "command": "echo '{\"hookSpecificOutput\": {\"additionalContext\": \"the folder is tidy\"}}'"}]}]
        }),
    ) else {
        return;
    };
    let model = h.script(vec![
        call("read_file", json!({"path": "secret.txt"}), "r1"),
        call("read_file", json!({"path": "a.txt"}), "r2"),
        call("list_dir", json!({"path": "."}), "l1"),
        say("done"),
    ]);
    let id = h.agent(None, "look", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    let blocked = result_of(&calls, "r1");
    assert!(blocked.contains("A hook blocked this read_file call: not that file"), "{blocked}");
    assert!(!result_of(&calls, "r2").contains("blocked"), "another file is read");
    let listed = result_of(&calls, "l1");
    assert!(listed.contains("[A hook adds] the folder is tidy"), "{listed}");
    assert_eq!(asked_hooks(&h), 2, "each hook asked once, though the Read hook ran twice");

    // A second turn runs them without asking again.
    let model = h.script(vec![call("list_dir", json!({"path": "."}), "l2"), say("again")]);
    h.agent(Some(&id), "again", &ws);
    h.turns_end(&id, 2);
    assert!(result_of(&model.calls(), "l2").contains("the folder is tidy"));
    assert_eq!(asked_hooks(&h), 2);
}

#[test]
fn a_hook_the_reader_refused_does_not_run() {
    let h = H::new("hooks-refused");
    let ws = h.workspace();
    let mark = h.folder.join("hook.mark");
    let Some(()) = hooks(&h, json!({"PreToolUse": hook("touch hook.mark")})) else { return };
    *h.confirm.answer.lock().unwrap() = false;
    let model = h.script(vec![call("list_dir", json!({"path": "."}), "l1"), say("done")]);
    let id = h.agent(None, "look", &ws);
    h.turns_end(&id, 1);
    assert!(!result_of(&model.calls(), "l1").contains("blocked"));
    assert!(!Path::new(&mark).exists(), "the refused hook did not run");
    assert_eq!(asked_hooks(&h), 1);
}

#[test]
fn a_message_hook_blocks_a_message_or_adds_to_it() {
    let h = H::new("hooks-prompt");
    let ws = h.workspace();
    let Some(()) = hooks(
        &h,
        json!({"UserPromptSubmit": hook(
            "grep -q password && { echo 'no passwords in chat' >&2; exit 2; } || echo 'Our tests run with cargo nextest.'"
        )}),
    ) else {
        return;
    };
    let refused = h
        .send(None, "my password is hunter2", "local", super::agent_tests::local(), lattice_protocol::conversation::Mode::Agent, Some(&ws))
        .unwrap_err();
    assert!(refused.message.contains("A hook stopped this message, so it was not sent: no passwords in chat"), "{refused:?}");

    let model = h.script(vec![say("ok")]);
    let id = h.agent(None, "how do I test?", &ws);
    h.turns_end(&id, 1);
    let texts = user_texts(&model.calls()[0]);
    let at = texts.iter().position(|t| t.contains("[A hook adds, for this message] Our tests run with cargo nextest.")).expect("the hook's words");
    assert_eq!(texts[at + 1], "how do I test?", "the words come just before the message");
}

#[test]
fn a_stop_hook_has_the_agent_go_on_once() {
    let h = H::new("hooks-stop");
    let ws = h.workspace();
    let Some(()) = hooks(
        &h,
        json!({"Stop": hook(
            "grep -q '\"stop_hook_active\":false' && echo '{\"decision\": \"block\", \"reason\": \"run the tests\"}' || true"
        )}),
    ) else {
        return;
    };
    let model = h.script(vec![say("done"), say("tests ran")]);
    let id = h.agent(None, "fix it", &ws);
    h.turns_end(&id, 2);
    let texts = h.texts(&id);
    assert!(
        texts.iter().any(|t| t == &format!("{}run the tests", super::hooked::FOLLOW_UP)),
        "{texts:?}"
    );
    assert_eq!(model.calls().len(), 2, "one more turn, then the hook let it stop");
}
