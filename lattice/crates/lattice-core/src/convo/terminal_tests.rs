//! The reader's terminals (`super::terminal`), end to end over the agent
//! harness with a fake window: a read asks the reader first, every time, and
//! what it gives back is redacted; a no reads nothing; a run asks with the
//! command and the folder, then runs in a new terminal; a command of more than
//! one line is refused before anyone is asked; and without the window's
//! terminals the tools say so.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use futures::future::BoxFuture;
use lattice_agents::model::{InputItem, ModelRequest};
use lattice_protocol::conversation::Mode;
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::prompt_agent;
use super::terminal::NO_TERMINALS;
use crate::ports::{ConfirmRequest, TerminalPort, TerminalTab};

#[derive(Default)]
struct FakeTerms {
    tabs: Vec<TerminalTab>,
    text: String,
    reads: Mutex<Vec<(u64, usize)>>,
    runs: Mutex<Vec<(PathBuf, String)>>,
}

impl TerminalPort for FakeTerms {
    fn list(&self) -> BoxFuture<'static, Vec<TerminalTab>> {
        let tabs = self.tabs.clone();
        async move { tabs }.boxed()
    }

    fn read(&self, id: u64, lines: usize) -> BoxFuture<'static, Result<String, String>> {
        self.reads.lock().unwrap().push((id, lines));
        let text = self.text.clone();
        async move { Ok(text) }.boxed()
    }

    fn run(&self, folder: PathBuf, command: String) -> BoxFuture<'static, Result<u64, String>> {
        self.runs.lock().unwrap().push((folder, command));
        async move { Ok(7) }.boxed()
    }
}

fn tab(id: u64, active: bool) -> TerminalTab {
    TerminalTab { id, label: "PowerShell 7".into(), folder: "C:\\work".into(), running: true, active }
}

fn harness(tag: &str, fake: Option<Arc<FakeTerms>>) -> H {
    H::with(tag, &[], move |config| {
        config.terminal = fake.map(|fake| fake as Arc<dyn TerminalPort>);
    })
}

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

#[test]
fn a_terminal_is_read_after_the_users_yes_and_what_it_shows_is_redacted() {
    // Built here, so no key-shaped text sits in the source.
    let key = ["sk", "ant", "api03", &"A".repeat(36)].join("-");
    let fake = Arc::new(FakeTerms {
        tabs: vec![tab(1, false), tab(2, true)],
        text: format!("PS C:\\work> npm test\nTESTS PASSED\n$env:KEY = '{key}'\n"),
        ..Default::default()
    });
    let h = harness("terminal-read", Some(fake.clone()));
    let ws = h.workspace();
    let model = h.script(vec![
        call("read_terminal", json!({"lines": 50}), "c1"),
        call("read_terminal", json!({"terminal": 9}), "c2"),
        say("The tests passed."),
    ]);
    let id = h.agent(None, "what did my terminal print?", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    let read = result_of(&calls, "c1");
    assert!(read.contains("TESTS PASSED"), "{read}");
    assert!(!read.contains(&key), "redacted: {read}");
    assert!(read.contains("terminal 2 (PowerShell 7, in C:\\work)"), "the active one: {read}");
    assert!(read.contains("Also open: terminal 1"), "{read}");
    assert_eq!(*fake.reads.lock().unwrap(), [(2, 50)]);
    assert!(h.confirm.asked().iter().any(|request| matches!(
        request,
        ConfirmRequest::TerminalRead { lines: 50, terminal } if terminal == "PowerShell 7 (C:\\work)"
    )));
    assert!(result_of(&calls, "c2").contains("There is no terminal 9"));
}

#[test]
fn a_no_reads_nothing_and_runs_nothing() {
    let fake = Arc::new(FakeTerms { tabs: vec![tab(1, true)], text: "hello".into(), ..Default::default() });
    let h = harness("terminal-no", Some(fake.clone()));
    let ws = h.workspace();
    *h.confirm.answer.lock().unwrap() = false;
    let model = h.script(vec![
        call("read_terminal", json!({}), "c1"),
        call("run_in_terminal", json!({"command": "npm run dev"}), "c2"),
        say("Understood."),
    ]);
    let id = h.agent(None, "look at my terminal", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(result_of(&calls, "c1").contains("did not allow reading"));
    assert!(result_of(&calls, "c2").contains("did not allow running"));
    assert!(fake.reads.lock().unwrap().is_empty());
    assert!(fake.runs.lock().unwrap().is_empty());
}

#[test]
fn a_run_asks_with_its_command_and_folder_then_runs_in_a_new_terminal() {
    let fake = Arc::new(FakeTerms::default());
    let h = harness("terminal-run", Some(fake.clone()));
    let ws = h.workspace();
    let model = h.script(vec![
        call("run_in_terminal", json!({"command": "npm run dev"}), "c1"),
        call("run_in_terminal", json!({"command": "echo a\nRemove-Item b"}), "c2"),
        say("It runs."),
    ]);
    let id = h.agent(None, "start the dev server where I can see it", &ws);
    h.turns_end(&id, 1);
    let calls = model.calls();
    assert!(result_of(&calls, "c1").contains("runs in a new terminal (7)"), "{}", result_of(&calls, "c1"));
    assert!(result_of(&calls, "c2").contains("Give one line"));
    let runs = fake.runs.lock().unwrap().clone();
    assert_eq!(runs.len(), 1, "the two-line command never ran");
    assert_eq!(runs[0].1, "npm run dev");
    let asked: Vec<ConfirmRequest> = h
        .confirm
        .asked()
        .into_iter()
        .filter(|request| matches!(request, ConfirmRequest::TerminalRun { .. }))
        .collect();
    assert_eq!(asked.len(), 1, "the refused command was never asked about");
    let ConfirmRequest::TerminalRun { command, folder } = &asked[0] else { unreachable!() };
    assert_eq!(command, "npm run dev");
    assert_eq!(PathBuf::from(folder), runs[0].0);
}

#[test]
fn without_the_windows_terminals_the_tools_say_so_and_they_are_agent_mode_tools() {
    let h = harness("terminal-none", None);
    let ws = h.workspace();
    let model = h.script(vec![call("read_terminal", json!({}), "c1"), say("None.")]);
    let id = h.agent(None, "read my terminal", &ws);
    h.turns_end(&id, 1);
    assert!(result_of(&model.calls(), "c1").contains(NO_TERMINALS));
    let agent = prompt_agent::tools(Mode::Agent);
    let ask = prompt_agent::tools(Mode::Ask);
    for name in ["read_terminal", "run_in_terminal"] {
        assert!(agent.iter().any(|t| t.name == name), "{name}");
        assert!(!ask.iter().any(|t| t.name == name), "{name} not in Ask mode");
    }
}
