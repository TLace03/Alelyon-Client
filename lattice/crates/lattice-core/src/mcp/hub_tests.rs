//! The hub with real server processes (the chat core's spec §12,
//! "Tests": conformance, a crashing server, kill on close, asking again after
//! a hash change, T5's environment). The server is `lattice-mcp-stub`
//! (`tests/stub/mcp_stub.rs`), started exactly as a reader's server is:
//! planned by `launch`, spawned by `lattice-sys` in a Job Object with pipes.

#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use super::config::{self, ServerEntry, ServerKey, entry_of};
use super::hub::{McpHub, ServerStatus, Timeouts};
use crate::env::{Env, MapEnv};
use crate::ports::fake::RecordingConfirm;
use crate::ports::{ConfirmRequest, Confirmer, Initiated};
use crate::state::StateRoot;
use crate::testkit::TempDir;

/// The stub this crate's tests build, beside the test binary's folder.
pub(crate) fn stub() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let dir = exe.parent().and_then(|deps| deps.parent()).unwrap();
    let stub = dir.join(format!("lattice-mcp-stub{}", std::env::consts::EXE_SUFFIX));
    assert!(
        stub.is_file(),
        "{} is missing: `cargo test` builds it with this crate's test targets",
        stub.display()
    );
    stub
}

struct Rig {
    dir: TempDir,
    state: StateRoot,
    hub: McpHub,
    confirm: RecordingConfirm,
    confirmer: Confirmer,
    runtime: tokio::runtime::Runtime,
}

fn environment(home: &Path) -> MapEnv {
    let root = std::env::var_os("SystemRoot").unwrap();
    let system32 = PathBuf::from(&root).join("System32");
    // A PATH entry inside `<globals>`: X7 leaves it out of every child's PATH.
    let planted = home.join("globals").join("planted-bin");
    let path = format!("{};{}", system32.display(), planted.display());
    MapEnv::new()
        .with("SystemRoot", root)
        .with("PATH", path)
        .with("USERPROFILE", home.as_os_str())
        .with("TEMP", home.as_os_str())
        // Never passed on (X7), unless an entry names it.
        .with("OPENAI_API_KEY", "sentinel-not-for-servers")
        .with("API_KEY", "from-lattice")
}

fn rig_with(timeouts: Timeouts) -> Rig {
    let dir = TempDir::new("mcp-hub");
    let state = StateRoot::at(dir.path());
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let env: Arc<dyn Env> = Arc::new(environment(dir.path()));
    let next = Arc::new(std::sync::Mutex::new(100.0));
    let clock: crate::clock::Clock = Arc::new(move || {
        let mut next = next.lock().unwrap();
        *next += 1.0;
        *next
    });
    let hub =
        McpHub::new(state.clone(), env, clock, runtime.handle().clone()).with_timeouts(timeouts);
    let confirm = RecordingConfirm::answering(true);
    let confirmer = Confirmer::new(Arc::new(confirm.clone()));
    Rig {
        dir,
        state,
        hub,
        confirm,
        confirmer,
        runtime,
    }
}

fn rig() -> Rig {
    rig_with(Timeouts {
        start: Duration::from_secs(20),
        list: Duration::from_secs(20),
        call: Duration::from_secs(20),
        stop: Duration::from_millis(500),
    })
}

impl Rig {
    /// Declare the stub as `name` in the reader's file, with `args`.
    fn declare(&self, name: &str, args: &[&str], env: Value) -> ServerEntry {
        let entry = json!({"command": stub().display().to_string(), "args": args, "env": env});
        config::put_user_server(&self.state, name, &entry).unwrap();
        self.entry(name)
    }

    fn entry(&self, name: &str) -> ServerEntry {
        config::load_user(&self.state)
            .servers
            .into_iter()
            .find(|entry| entry.key.name == name)
            .unwrap()
    }

    fn enable(&self, entry: &ServerEntry) -> bool {
        self.runtime
            .block_on(
                self.hub
                    .enable(entry, None, None, &self.confirmer, Initiated::Native),
            )
            .unwrap()
    }

    fn record(&self, name: &str) -> PathBuf {
        self.dir.path().join(format!("{name}.record"))
    }

    fn status(&self, entry: &ServerEntry) -> ServerStatus {
        let declared = config::load_user(&self.state);
        self.hub
            .overview(&declared)
            .servers
            .into_iter()
            .find(|view| view.key == entry.key)
            .map(|view| view.status)
            .unwrap()
    }
}

/// The enable dialog names the file, the line, the program it starts, the
/// folder and the variable names; a no enables nothing, a yes the exact hash.
#[test]
fn enabling_asks_with_the_program_and_the_names_and_a_no_enables_nothing() {
    let rig = rig();
    let entry = rig.declare("stub", &["--pages"], json!({"TOKEN": "t-1"}));
    *rig.confirm.answer.lock().unwrap() = false;
    assert!(!rig.enable(&entry));
    assert!(!rig.hub.approvals().approved(&entry));
    let asked = rig.confirm.asked();
    let ConfirmRequest::EnableMcpServer {
        name,
        from,
        command_line,
        program,
        cwd,
        env_names,
    } = &asked[0]
    else {
        panic!("{asked:?}")
    };
    assert_eq!(name, "stub");
    assert_eq!(from, "your MCP settings");
    assert!(command_line.ends_with("--pages"), "{command_line}");
    assert!(
        program
            .to_ascii_lowercase()
            .ends_with("lattice-mcp-stub.exe"),
        "{program}"
    );
    assert_eq!(Path::new(cwd), rig.dir.path(), "the reader's home");
    assert_eq!(env_names, &["TOKEN"]);
    let dialog = asked[0].dialog();
    assert!(
        !dialog.lines.join(" ").contains("t-1"),
        "a value is never shown"
    );
    *rig.confirm.answer.lock().unwrap() = true;
    assert!(rig.enable(&entry));
    assert!(rig.hub.approvals().approved(&entry));
}

#[test]
fn a_server_that_is_not_enabled_never_starts() {
    let rig = rig();
    let record = rig.record("off");
    let entry = rig.declare(
        "off",
        &["--record", &record.display().to_string()],
        json!({}),
    );
    let tools = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[entry.clone()], None, None));
    assert!(
        tools.tools.is_empty() && tools.notices.is_empty(),
        "{tools:?}"
    );
    let started = rig.runtime.block_on(rig.hub.start(&entry, None, None));
    assert!(started.unwrap_err().contains("not enabled"));
    assert!(
        !record.with_extension("record.pid").exists(),
        "nothing started"
    );
    assert_eq!(rig.status(&entry), ServerStatus::Stopped);
}

/// Conformance against the stub: the handshake, a paged list, each kind of
/// result, and the server's own ping answered mid-call.
#[test]
fn an_enabled_server_starts_lists_its_tools_and_answers_calls() {
    let rig = rig();
    let entry = rig.declare("stub", &["--pages"], json!({}));
    assert!(rig.enable(&entry));
    let turn = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[entry.clone()], None, None));
    assert!(turn.notices.is_empty(), "{:?}", turn.notices);
    let names: Vec<&str> = turn.tools.iter().map(|t| t.model_name.as_str()).collect();
    assert_eq!(
        names,
        [
            "mcp__stub__echo",
            "mcp__stub__add",
            "mcp__stub__fail",
            "mcp__stub__big",
            "mcp__stub__image",
            "mcp__stub__slow",
            "mcp__stub__env",
            "mcp__stub__ping_client"
        ]
    );
    assert!(
        turn.tools[0]
            .description
            .starts_with("(MCP server stub) Say the text back.")
    );
    assert_eq!(turn.tools[0].parameters["required"], json!(["text"]));
    let call = |tool: &str, arguments: Value| {
        rig.runtime
            .block_on(rig.hub.call(&entry.key, &entry.sha256, tool, arguments))
    };
    assert_eq!(call("echo", json!({"text": "hi"})).unwrap().text, "hi");
    assert_eq!(call("add", json!({"a": 2, "b": 3})).unwrap().text, "5");
    let failed = call("fail", json!({})).unwrap();
    assert!(failed.is_error);
    let big = call("big", json!({})).unwrap();
    assert!(big.cut && big.text.ends_with("more bytes are not shown.]"));
    assert!(
        call("image", json!({}))
            .unwrap()
            .text
            .starts_with("[An image (image/png")
    );
    assert_eq!(call("ping_client", json!({})).unwrap().text, "pong ok");
    assert!(
        call("nope", json!({}))
            .unwrap_err()
            .contains("no tool nope")
    );
    let declared = config::load_user(&rig.state);
    let view = rig.hub.overview(&declared).servers.remove(0);
    assert_eq!(view.status, ServerStatus::Running);
    assert_eq!(
        view.server.as_deref(),
        Some("lattice-mcp-stub 1.0, MCP 2025-11-25")
    );
    assert_eq!(view.instructions.as_deref(), Some("A stub for tests."));
    assert_eq!(view.tools.len(), 8);
    assert!(view.tools.iter().all(|t| t.on && !t.allowed));
    assert!(view.enabled && !view.off);
}

/// The pin: an edited entry is not enabled, and the server started from the
/// old one is not used for it.
#[test]
fn a_changed_entry_asks_again_and_its_old_server_is_not_used() {
    let rig = rig();
    let entry = rig.declare("stub", &[], json!({}));
    assert!(rig.enable(&entry));
    rig.runtime
        .block_on(rig.hub.start(&entry, None, None))
        .unwrap();
    let changed = rig.declare("stub", &["--pages"], json!({}));
    assert_ne!(changed.sha256, entry.sha256);
    let turn = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[changed.clone()], None, None));
    assert!(turn.tools.is_empty());
    let declared = config::load_user(&rig.state);
    let view = rig.hub.overview(&declared).servers.remove(0);
    assert!(!view.enabled);
    assert_eq!(
        view.status,
        ServerStatus::Stopped,
        "the running one is the old entry's"
    );
    assert!(view.tools.is_empty());
    assert!(
        rig.runtime
            .block_on(
                rig.hub
                    .call(&changed.key, &changed.sha256, "echo", json!({"text": "x"}))
            )
            .is_err()
    );
    // Enabled again (a new yes), the next use starts the new entry.
    assert!(rig.enable(&changed));
    let turn = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[changed.clone()], None, None));
    assert_eq!(turn.tools.len(), 8);
    assert!(turn.tools.iter().all(|t| t.sha256 == changed.sha256));
}

#[test]
fn a_server_that_exits_at_start_is_a_sentence_with_its_last_line() {
    let rig = rig();
    let entry = rig.declare("dies", &["--exit", "7"], json!({}));
    assert!(rig.enable(&entry));
    let turn = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[entry.clone()], None, None));
    assert!(turn.tools.is_empty());
    assert_eq!(turn.notices.len(), 1);
    assert!(
        turn.notices[0].contains("did not start"),
        "{:?}",
        turn.notices
    );
    let ServerStatus::Failed(why) = rig.status(&entry) else {
        panic!("{:?}", rig.status(&entry))
    };
    assert!(why.contains("stopped"), "{why}");
    let declared = config::load_user(&rig.state);
    let log = rig.hub.overview(&declared).servers.remove(0).log;
    assert!(
        log.iter().any(|line| line.contains("stub: exiting with 7")),
        "{log:?}"
    );
}

#[test]
fn a_server_that_crashes_during_a_call_fails_the_call_and_says_so() {
    let rig = rig();
    let entry = rig.declare("crash", &["--crash-on", "echo"], json!({}));
    assert!(rig.enable(&entry));
    rig.runtime
        .block_on(rig.hub.start(&entry, None, None))
        .unwrap();
    let error = rig
        .runtime
        .block_on(
            rig.hub
                .call(&entry.key, &entry.sha256, "echo", json!({"text": "x"})),
        )
        .unwrap_err();
    assert!(error.contains("stopped"), "{error}");
    // The end reaches the status from the connection's reader thread.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let why = loop {
        if let ServerStatus::Failed(why) = rig.status(&entry) {
            break why;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{:?}",
            rig.status(&entry)
        );
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(why.contains("stub: crashing on echo"), "{why}");
    let turn = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[entry.clone()], None, None));
    assert_eq!(turn.tools.len(), 8, "the next use starts it again");
}

#[test]
fn a_server_speaking_another_version_or_never_answering_does_not_start() {
    // A start limit that a loaded machine still answers within: the version
    // is what refuses this one, not the clock.
    let patient = rig_with(Timeouts {
        start: Duration::from_secs(20),
        list: Duration::from_secs(5),
        call: Duration::from_secs(5),
        stop: Duration::from_millis(200),
    });
    let other = patient.declare("other", &["--version", "2026-07-28"], json!({}));
    assert!(patient.enable(&other));
    let error = patient
        .runtime
        .block_on(patient.hub.start(&other, None, None))
        .unwrap_err();
    assert!(error.contains("speaks MCP 2026-07-28"), "{error}");
    let rig = rig_with(Timeouts {
        start: Duration::from_millis(400),
        list: Duration::from_secs(5),
        call: Duration::from_secs(5),
        stop: Duration::from_millis(200),
    });
    let hangs = rig.declare("hangs", &["--hang-init"], json!({}));
    assert!(rig.enable(&hangs));
    let error = rig
        .runtime
        .block_on(rig.hub.start(&hangs, None, None))
        .unwrap_err();
    assert_eq!(error, "The server did not finish starting within 400 ms.");
}

/// T5 and X7: the server's environment is X7's block and the variables its
/// entry names, nothing else of Lattice's.
#[test]
fn a_servers_environment_is_x7_and_the_names_its_entry_gives() {
    let rig = rig();
    let record = rig.record("env");
    let entry = rig.declare(
        "env",
        &["--record", &record.display().to_string()],
        json!({"FROM_FILE": "v1"}),
    );
    assert!(rig.enable(&entry));
    rig.runtime
        .block_on(rig.hub.start(&entry, None, None))
        .unwrap();
    let names = std::fs::read_to_string(format!("{}.env", record.display())).unwrap();
    let names: Vec<&str> = names.lines().collect();
    for wanted in [
        "FROM_FILE",
        "NoDefaultCurrentDirectoryInExePath",
        "PATH",
        "SystemRoot",
    ] {
        assert!(
            names.iter().any(|n| n.eq_ignore_ascii_case(wanted)),
            "{wanted}: {names:?}"
        );
    }
    assert!(
        !names
            .iter()
            .any(|n| n.eq_ignore_ascii_case("OPENAI_API_KEY")),
        "{names:?}"
    );
    assert!(
        !names.iter().any(|n| n.eq_ignore_ascii_case("API_KEY")),
        "{names:?}"
    );
    let value = |name: &str| {
        rig.runtime
            .block_on(
                rig.hub
                    .call(&entry.key, &entry.sha256, "env", json!({"name": name})),
            )
            .unwrap()
            .text
    };
    assert_eq!(value("FROM_FILE"), "v1");
    assert_eq!(value("OPENAI_API_KEY"), "(unset)");
}

/// A folder's server: the variable it names comes from Lattice's own
/// environment (never the folder's value), it may not replace PATH, and it
/// runs in the folder.
#[test]
fn a_folders_server_gets_lattices_value_for_a_name_and_runs_in_the_folder() {
    let rig = rig();
    let folder = rig.dir.path().join("repo");
    std::fs::create_dir_all(&folder).unwrap();
    let scope = config::Scope::Folder {
        id: "ef".repeat(16),
        path: folder.display().to_string(),
    };
    let record = rig.record("folder");
    let entry = entry_of(
        "repo-tools",
        &json!({
            "command": stub().display().to_string(),
            "args": ["--record", record.display().to_string()],
            "env": {"API_KEY": "planted-in-the-folder", "PATH": "C:\\planted"}
        }),
        &scope,
        ".lattice/mcp.json",
    )
    .unwrap();
    let enabled = rig
        .runtime
        .block_on(rig.hub.enable(
            &entry,
            Some(&folder),
            None,
            &rig.confirmer,
            Initiated::Native,
        ))
        .unwrap();
    assert!(enabled);
    let turn = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[entry.clone()], Some(&folder), None));
    assert_eq!(turn.tools.len(), 8, "{:?}", turn.notices);
    let value = |name: &str| {
        rig.runtime
            .block_on(
                rig.hub
                    .call(&entry.key, &entry.sha256, "env", json!({"name": name})),
            )
            .unwrap()
            .text
    };
    assert_eq!(value("API_KEY"), "from-lattice");
    // Neither the folder's value nor Lattice's own unfiltered PATH (with its
    // entry inside `<globals>`): X7's filtered one.
    let path = value("PATH");
    assert!(!path.contains("planted"), "{path}");
    assert!(path.to_ascii_lowercase().contains("system32"), "{path}");
    let ConfirmRequest::EnableMcpServer { cwd, from, .. } = &rig.confirm.asked()[0] else {
        panic!()
    };
    assert_eq!(Path::new(cwd), folder);
    assert_eq!(from, ".lattice/mcp.json");
}

fn alive(pid: &str) -> bool {
    let out = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).contains(&format!("\"{pid}\""))
}

/// Stop ends the server's process, and so does dropping the hub (Lattice
/// closing): every Job closes with it.
#[test]
fn stopping_or_dropping_the_hub_ends_the_servers_process() {
    let rig = rig();
    let record = rig.record("stop");
    let entry = rig.declare(
        "stop",
        &["--record", &record.display().to_string()],
        json!({}),
    );
    assert!(rig.enable(&entry));
    rig.runtime
        .block_on(rig.hub.start(&entry, None, None))
        .unwrap();
    let pid = std::fs::read_to_string(format!("{}.pid", record.display())).unwrap();
    assert!(alive(&pid));
    rig.runtime.block_on(rig.hub.stop(&entry.key));
    // Stop closes the server's Job and Windows ends the tree as it closes,
    // which a loaded machine has finished after the call returned: the exit
    // is waited for (60 s at most, a bound on a hang), not assumed.
    let stopped = pid.clone();
    crate::testkit::within("stop to end the server", 60, move || {
        while alive(&stopped) {
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    assert_eq!(rig.status(&entry), ServerStatus::Stopped);
    assert!(
        rig.runtime
            .block_on(
                rig.hub
                    .call(&entry.key, &entry.sha256, "echo", json!({"text": "x"}))
            )
            .unwrap_err()
            .contains("not running")
    );
    // Started again, then the hub drops.
    rig.runtime
        .block_on(rig.hub.start(&entry, None, None))
        .unwrap();
    let pid = std::fs::read_to_string(format!("{}.pid", record.display())).unwrap();
    assert!(alive(&pid));
    let Rig { hub, runtime, .. } = rig;
    drop(hub);
    crate::testkit::within("the server to end", 60, move || {
        while alive(&pid) {
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    drop(runtime);
}

/// A switched-off tool is not offered, an allowed one is marked, and a
/// switched-off server is stopped and offers nothing.
#[test]
fn switches_and_allow_always_shape_what_a_turn_offers() {
    let rig = rig();
    let entry = rig.declare("stub", &[], json!({}));
    assert!(rig.enable(&entry));
    rig.runtime
        .block_on(rig.hub.switch(&entry.key, "fail", false))
        .unwrap();
    let allowed = rig
        .runtime
        .block_on(rig.hub.allow_always(
            &entry,
            "echo",
            &rig.confirmer,
            "allow:test",
            Initiated::Native,
        ))
        .unwrap();
    assert!(allowed);
    assert!(matches!(
        rig.confirm.asked().last(),
        Some(ConfirmRequest::AllowMcpTool { tool, .. }) if tool == "echo"
    ));
    let turn = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[entry.clone()], None, None));
    assert!(!turn.tools.iter().any(|t| t.tool == "fail"));
    let echo = turn.tools.iter().find(|t| t.tool == "echo").unwrap();
    assert!(echo.allowed);
    assert!(!turn.tools.iter().find(|t| t.tool == "add").unwrap().allowed);
    rig.runtime
        .block_on(
            rig.hub
                .switch(&entry.key, super::approvals::WHOLE_SERVER, false),
        )
        .unwrap();
    let off = rig.entry("stub");
    assert!(off.disabled, "the reader's own file says disabled");
    assert_eq!(rig.status(&off), ServerStatus::Stopped);
    assert!(
        rig.runtime
            .block_on(rig.hub.turn_tools(&[off], None, None))
            .tools
            .is_empty()
    );
}

/// `notifications/tools/list_changed`: the next turn lists again.
#[test]
fn a_changed_tool_list_is_read_again_before_the_next_turn() {
    let rig = rig();
    let entry = rig.declare("grows", &["--list-changed"], json!({}));
    assert!(rig.enable(&entry));
    let first = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[entry.clone()], None, None));
    assert_eq!(first.tools.len(), 8);
    rig.runtime
        .block_on(
            rig.hub
                .call(&entry.key, &entry.sha256, "echo", json!({"text": "x"})),
        )
        .unwrap();
    std::thread::sleep(Duration::from_millis(200));
    let second = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[entry.clone()], None, None));
    assert_eq!(second.tools.len(), 9);
    assert!(second.tools.iter().any(|t| t.tool == "late"));
}

#[test]
fn a_reader_server_key_is_found_again_after_a_put() {
    let rig = rig();
    let entry = rig.declare("k", &[], json!({}));
    assert_eq!(entry.key, ServerKey::user("k"));
}

/// By hand: Lattice's client against a server it did not write, one built with
/// the official MCP Python SDK. `LATTICE_MCP_PYTHON` names a Python that has
/// the SDK (mcp 2.x), and `LATTICE_MCP_SDK_SERVER` a script that serves
/// `shout(text)` and `add(a, b)` with `MCPServer` on stdio.
#[test]
#[ignore = "by hand: needs a Python with the MCP SDK (see the comment)"]
fn by_hand_a_server_built_with_the_mcp_sdk_speaks_with_lattices_client() {
    let python = std::env::var("LATTICE_MCP_PYTHON").expect("LATTICE_MCP_PYTHON");
    let script = std::env::var("LATTICE_MCP_SDK_SERVER").expect("LATTICE_MCP_SDK_SERVER");
    let rig = rig();
    config::put_user_server(
        &rig.state,
        "sdk",
        &json!({"command": python, "args": [script]}),
    )
    .unwrap();
    let entry = rig.entry("sdk");
    assert!(rig.enable(&entry));
    let turn = rig
        .runtime
        .block_on(rig.hub.turn_tools(&[entry.clone()], None, None));
    assert!(turn.notices.is_empty(), "{:?}", turn.notices);
    let mut names: Vec<&str> = turn.tools.iter().map(|t| t.model_name.as_str()).collect();
    names.sort();
    assert_eq!(names, ["mcp__sdk__add", "mcp__sdk__shout"]);
    let call = |tool: &str, arguments: Value| {
        rig.runtime
            .block_on(rig.hub.call(&entry.key, &entry.sha256, tool, arguments))
            .unwrap()
    };
    assert_eq!(call("shout", json!({"text": "hello"})).text, "HELLO");
    assert_eq!(call("add", json!({"a": 2, "b": 3})).text, "5");
    let declared = config::load_user(&rig.state);
    let view = rig.hub.overview(&declared).servers.remove(0);
    println!("server: {:?}", view.server);
    assert!(view.server.is_some_and(|s| s.contains("MCP 2025-11-25")));
}
