//! The ACP client against a stub agent on in-process pipes that does what
//! the two adapters did when measured (2026-10-09): it answers the handshake,
//! signs in, opens a session, takes a model, and in a prompt streams text and
//! a tool call, asks permission, reads and writes a file, then ends its turn.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};

use super::connection::{Connection, Event};
use super::session::{self, AgentSession, Client, Model, Permission, Update};
use super::{Agent, environment};
use crate::env::MapEnv;

fn pipes() -> (std::fs::File, std::fs::File) {
    use std::os::windows::io::OwnedHandle;
    let (reader, writer) = std::io::pipe().unwrap();
    (
        std::fs::File::from(OwnedHandle::from(reader)),
        std::fs::File::from(OwnedHandle::from(writer)),
    )
}

/// The stub agent: reads Lattice's lines from `input`, writes to `output`;
/// records every method it was sent.
fn stub(input: std::fs::File, mut output: std::fs::File, seen: Arc<Mutex<Vec<Value>>>) {
    std::thread::spawn(move || {
        let mut lines = BufReader::new(input).lines();
        let mut send = |v: Value| {
            let _ = writeln!(output, "{v}");
            let _ = output.flush();
        };
        let mut next_line = || -> Option<Value> {
            lines
                .next()?
                .ok()
                .and_then(|l| serde_json::from_str(&l).ok())
        };
        while let Some(msg) = next_line() {
            seen.lock().unwrap().push(msg.clone());
            let id = msg.get("id").cloned().unwrap_or(Value::Null);
            match msg.get("method").and_then(Value::as_str) {
                Some("initialize") => send(
                    json!({"jsonrpc": "2.0", "id": id, "result": {"protocolVersion": 1, "authMethods": []}}),
                ),
                Some("authenticate")
                | Some("session/set_model")
                | Some("session/set_config_option") => {
                    send(json!({"jsonrpc": "2.0", "id": id, "result": {}}))
                }
                Some("session/new") => {
                    send(json!({"jsonrpc": "2.0", "id": id, "result": {"sessionId": "s1"}}))
                }
                Some("session/prompt") => {
                    let update = |u: Value| json!({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "s1", "update": u}});
                    send(update(
                        json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Hel"}}),
                    ));
                    send(update(
                        json!({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "lo"}}),
                    ));
                    send(update(
                        json!({"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Edit a.txt", "kind": "edit", "status": "pending"}),
                    ));
                    send(
                        json!({"jsonrpc": "2.0", "id": 100, "method": "session/request_permission", "params": {"sessionId": "s1",
                        "toolCall": {"toolCallId": "t1", "title": "Edit a.txt", "rawInput": {"path": "a.txt"}},
                        "options": [{"optionId": "allow", "name": "Allow", "kind": "allow_once"}, {"optionId": "no", "name": "Reject", "kind": "reject_once"}]}}),
                    );
                    send(
                        json!({"jsonrpc": "2.0", "id": 101, "method": "fs/read_text_file", "params": {"sessionId": "s1", "path": "a.txt", "line": 1, "limit": 5}}),
                    );
                    send(
                        json!({"jsonrpc": "2.0", "id": 102, "method": "fs/write_text_file", "params": {"sessionId": "s1", "path": "a.txt", "content": "new"}}),
                    );
                    send(
                        json!({"jsonrpc": "2.0", "id": 103, "method": "terminal/create", "params": {"sessionId": "s1", "command": "rm -rf /"}}),
                    );
                    // The answers come back as lines; wait for all four before ending the turn.
                    let mut answered = 0;
                    while answered < 4 {
                        let Some(answer) = next_line() else { return };
                        seen.lock().unwrap().push(answer.clone());
                        if answer.get("method").is_none() {
                            answered += 1;
                        }
                    }
                    send(update(
                        json!({"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed",
                        "content": [{"type": "content", "content": {"type": "text", "text": "done"}}]}),
                    ));
                    send(json!({"jsonrpc": "2.0", "id": id, "result": {"stopReason": "end_turn"}}));
                }
                _ => {}
            }
        }
    });
}

#[derive(Default)]
struct Recorder {
    updates: Mutex<Vec<Update>>,
    asked: Mutex<Vec<Permission>>,
    writes: Mutex<Vec<(String, String)>>,
}

impl Client for Recorder {
    fn update(&self, update: Update) {
        self.updates.lock().unwrap().push(update);
    }
    fn permission(&self, permission: Permission) -> BoxFuture<'static, Option<String>> {
        let choice = permission
            .options
            .iter()
            .find(|(_, _, kind)| kind == "reject_once")
            .map(|(id, ..)| id.clone());
        self.asked.lock().unwrap().push(permission);
        async move { choice }.boxed()
    }
    fn read(&self, path: &str, _line: Option<u64>, _limit: Option<u64>) -> Result<String, String> {
        Ok(format!("text of {path}"))
    }
    fn write(&self, path: &str, content: &str) -> Result<(), String> {
        self.writes
            .lock()
            .unwrap()
            .push((path.to_owned(), content.to_owned()));
        Err("Staged for your review, not written.".to_owned())
    }
}

fn answer_to(seen: &[Value], id: u64) -> Value {
    seen.iter()
        .find(|m| m.get("id").and_then(Value::as_u64) == Some(id) && m.get("method").is_none())
        .cloned()
        .unwrap_or(Value::Null)
}

/// A whole prompt: the handshake with sign-in and model, the streamed text and
/// tool call, the permission answered with the reader's choice, the read
/// answered, the write refused with the caller's sentence, an unoffered
/// request refused, and the turn's end.
/// Mutant: a permission answered with the first option instead of the choice.
#[test]
fn a_prompt_streams_asks_reads_and_writes_through_the_client() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (agent_in, to_agent) = pipes();
    let (from_agent, agent_out) = pipes();
    let seen = Arc::new(Mutex::new(Vec::new()));
    stub(agent_in, agent_out, seen.clone());
    let client = Arc::new(Recorder::default());
    let logged = Arc::new(Mutex::new(Vec::new()));
    let log = logged.clone();
    let conn = Connection::start(
        "stub",
        Box::new(from_agent),
        Box::new(to_agent),
        session::events(
            client.clone(),
            Arc::new(move |line| log.lock().unwrap().push(line)),
        ),
        session::asks(client.clone(), runtime.handle().clone()),
    )
    .unwrap();
    let session = runtime
        .block_on(AgentSession::start(
            conn.clone(),
            Path::new("C:/work"),
            Some("chatgpt"),
            &Model::ConfigOption("gpt-5.5".into()),
        ))
        .unwrap();
    assert_eq!(session.id, "s1");
    let stop = runtime.block_on(session.prompt("Edit a.txt")).unwrap();
    assert_eq!(stop, "end_turn");
    let updates = client.updates.lock().unwrap().clone();
    assert_eq!(
        updates[0..2],
        [Update::Text("Hel".into()), Update::Text("lo".into())]
    );
    assert!(updates.contains(&Update::ToolCall {
        id: "t1".into(),
        title: "Edit a.txt".into(),
        kind: "edit".into(),
        status: "pending".into()
    }));
    assert!(updates.contains(&Update::ToolUpdate {
        id: "t1".into(),
        status: Some("completed".into()),
        text: Some("done".into())
    }));
    let asked = client.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 1);
    assert_eq!(asked[0].title, "Edit a.txt");
    assert!(asked[0].detail.contains("a.txt"));
    let seen = seen.lock().unwrap().clone();
    let methods: Vec<&str> = seen
        .iter()
        .filter_map(|m| m.get("method").and_then(Value::as_str))
        .collect();
    assert_eq!(
        methods,
        [
            "initialize",
            "authenticate",
            "session/new",
            "session/set_config_option",
            "session/prompt"
        ]
    );
    assert_eq!(
        answer_to(&seen, 100)["result"],
        json!({"outcome": {"outcome": "selected", "optionId": "no"}})
    );
    assert_eq!(
        answer_to(&seen, 101)["result"],
        json!({"content": "text of a.txt"})
    );
    assert_eq!(
        answer_to(&seen, 102)["error"]["message"],
        "Staged for your review, not written."
    );
    assert_eq!(
        answer_to(&seen, 103)["error"]["message"],
        "Lattice does not offer that."
    );
    assert_eq!(
        client.writes.lock().unwrap().clone(),
        [("a.txt".to_owned(), "new".to_owned())]
    );
    // Cancel goes out as a notification; closing ends the connection.
    session.cancel();
    session.close();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while conn.closed().is_none() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(conn.closed().is_some());
}

/// An agent that stops answering: a waiting request ends with the reason the
/// output ended, and the holder hears the connection closed.
#[test]
fn an_agent_that_ends_ends_every_waiting_request() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (_agent_in, to_agent) = pipes();
    let (from_agent, agent_out) = pipes();
    let closed = Arc::new(Mutex::new(Vec::new()));
    let saw = closed.clone();
    let conn = Connection::start(
        "gone",
        Box::new(from_agent),
        Box::new(to_agent),
        Arc::new(move |event| {
            if let Event::Closed(why) = event {
                saw.lock().unwrap().push(why);
            }
        }),
        Arc::new(|_| {}),
    )
    .unwrap();
    drop(agent_out);
    let error = runtime
        .block_on(conn.request("initialize", json!({}), std::time::Duration::from_secs(10)))
        .unwrap_err();
    assert_eq!(error.sentence(), "The agent stopped.");
    // The reader ends the waiting request first and then says the connection
    // closed: wait for that word.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while closed.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert_eq!(closed.lock().unwrap().clone(), ["The agent stopped."]);
}

/// The agents as Lattice runs them: each choice names its agent, its sign-in
/// and model are the ones measured, and its environment never carries
/// Claude Code's own nesting marker.
#[test]
fn the_agents_start_with_their_measured_settings_and_no_nesting_marker() {
    for agent in Agent::ALL {
        assert_eq!(Agent::from_choice(agent.choice()), Some(agent));
    }
    assert_eq!(Agent::ClaudeCode.model(), Model::Agents);
    assert_eq!(Agent::Codex.model(), Model::SetModel("gpt-6-luna[high]".into()));
    assert_eq!(Agent::Codex.auth(), Some("chat-gpt"));
    let env = MapEnv::new()
        .with("PATH", r"C:\Windows")
        .with("USERPROFILE", r"D:\Profiles\me")
        .with("APPDATA", r"D:\Profiles\me\AppData\Roaming")
        .with("CLAUDECODE", "1")
        .with("CLAUDE_CODE_ENTRYPOINT", "cli")
        .with("OPENAI_API_KEY", "not for an agent");
    let block = environment(&env, None, Path::new(r"C:\globals"));
    let names: Vec<String> = block
        .iter()
        .map(|(n, _)| n.to_string_lossy().to_uppercase())
        .collect();
    assert!(names.contains(&"APPDATA".to_owned()) && names.contains(&"USERPROFILE".to_owned()));
    let mut unique = names.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), names.len(), "each name once: {names:?}");
    for absent in ["CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "OPENAI_API_KEY"] {
        assert!(!names.contains(&absent.to_owned()), "{absent}");
    }
    let (program, argv) =
        Agent::ClaudeCode.command(Path::new(r"C:\agents"), Path::new(r"C:\node\node.exe"));
    assert_eq!(program, Path::new(r"C:\node\node.exe"));
    assert!(
        argv[1]
            .to_string_lossy()
            .ends_with(r"@agentclientprotocol\claude-agent-acp\dist\index.js")
    );
    let (_, argv) =
        Agent::Codex.command(Path::new(r"C:\agents"), Path::new(r"C:\node\node.exe"));
    assert!(
        argv[1]
            .to_string_lossy()
            .ends_with(r"@agentclientprotocol\codex-acp\dist\index.js")
    );
}

/// By hand only (it spends a little of the reader's subscriptions): each real
/// agent, started as Lattice starts it from `LATTICE_ACP_AGENTS` (a folder
/// with the two adapters installed), answers a one-word prompt.
#[test]
#[ignore]
fn each_real_agent_answers_on_the_readers_subscription() {
    let Some(dir) = std::env::var_os("LATTICE_ACP_AGENTS").map(std::path::PathBuf::from) else {
        panic!("set LATTICE_ACP_AGENTS")
    };
    let node = std::path::PathBuf::from(r"C:\Program Files\nodejs\node.exe");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let work = crate::testkit::TempDir::new("acp-real");
    for agent in Agent::ALL {
        assert!(
            agent.installed(&dir),
            "{agent:?} not installed in {}",
            dir.display()
        );
        let mut child = super::start(
            agent,
            &dir,
            &node,
            work.path(),
            &crate::env::ProcessEnv,
            &work.path().join("globals"),
        )
        .unwrap();
        super::drain(child.take_stderr().unwrap(), Arc::new(|_| {}));
        let client = Arc::new(Recorder::default());
        let conn = Connection::start(
            agent.label(),
            Box::new(child.take_stdout().unwrap()),
            Box::new(child.take_stdin().unwrap()),
            session::events(client.clone(), Arc::new(|line| eprintln!("log: {line}"))),
            session::asks(client.clone(), runtime.handle().clone()),
        )
        .unwrap();
        let session = runtime
            .block_on(AgentSession::start(
                conn,
                work.path(),
                agent.auth(),
                &agent.model(),
            ))
            .unwrap();
        let stop = runtime
            .block_on(session.prompt("Reply with the single word: ready. Do not use any tools."))
            .unwrap();
        let text: String = client
            .updates
            .lock()
            .unwrap()
            .iter()
            .filter_map(|u| match u {
                Update::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        eprintln!("{}: {stop}: {text:?}", agent.label());
        assert_eq!(stop, "end_turn");
        assert!(text.to_lowercase().contains("ready"), "{text}");
        session.close();
    }
}
