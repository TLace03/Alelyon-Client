//! The MCP client's pure parts and its connection over in-process pipes
//! (the chat core's spec §12, "Tests": conformance, the pin, T7). The
//! real child process (spawn, crash, kill on close) is in `hub_tests`.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

use super::approvals::{Approvals, WHOLE_SERVER};
use super::client::{self, CallError, Connection, Event};
use super::config::{self, REMOTE_REFUSED, Scope, ServerKey, canonical, entry_of, parse, pasted};
use super::jsonrpc::{self, Incoming};
use super::names::{self, MAX_MODEL_NAME};
use super::result::{MAX_RESULT, call_text};
use crate::clock::Clock;
use crate::state::StateRoot;
use crate::testkit::TempDir;

fn folder() -> Scope {
    Scope::Folder {
        id: "ab".repeat(16),
        path: r"\\?\C:\Work\Repo".to_owned(),
    }
}

// ------------------------------------------------------------ configuration

#[test]
fn a_file_in_cursors_shape_declares_its_servers_in_name_order() {
    let text = r#"{"mcpServers": {
        "files": {"command": "npx", "args": ["-y", "@scope/server-files", "C:/Work"], "env": {"TOKEN": "t-1", "A_B": "2"}},
        "docs": {"command": "C:\\Tools\\docs.exe", "cwd": "C:\\Tools", "disabled": true}
    }}"#;
    let declared = parse(text, &Scope::User, "your MCP settings");
    assert!(declared.problems.is_empty(), "{:?}", declared.problems);
    let names: Vec<&str> = declared
        .servers
        .iter()
        .map(|s| s.key.name.as_str())
        .collect();
    assert_eq!(names, ["docs", "files"]);
    let files = &declared.servers[1];
    assert_eq!(files.command, "npx");
    assert_eq!(files.args, ["-y", "@scope/server-files", "C:/Work"]);
    assert_eq!(
        files.env,
        [
            ("A_B".to_owned(), Some("2".to_owned())),
            ("TOKEN".to_owned(), Some("t-1".to_owned()))
        ]
    );
    assert_eq!(files.command_line(), "npx -y @scope/server-files C:/Work");
    assert!(!files.disabled);
    let docs = &declared.servers[0];
    assert!(docs.disabled);
    assert_eq!(docs.cwd.as_deref(), Some(r"C:\Tools"));
    assert_eq!(docs.command_line(), r"C:\Tools\docs.exe");
}

#[test]
fn a_folders_file_gives_variable_names_and_never_their_values() {
    let text = r#"{"mcpServers": {"x": {"command": "node", "args": ["server.js"], "env": {"API_KEY": "planted-value"}}}}"#;
    let declared = parse(text, &folder(), ".lattice/mcp.json");
    let entry = &declared.servers[0];
    assert_eq!(entry.env, [("API_KEY".to_owned(), None)]);
    assert_eq!(entry.env_names(), ["API_KEY"]);
    assert_eq!(entry.file, ".lattice/mcp.json");
}

/// T7, and the entries Lattice will not use, each with its sentence; the
/// other servers of the file are still used.
#[test]
fn remote_servers_and_malformed_entries_are_refused_one_by_one() {
    let text = r#"{"mcpServers": {
        "web": {"url": "https://example.invalid/mcp"},
        "sse": {"type": "sse", "command": "x"},
        "http": {"type": "http", "command": "x"},
        "nothing": {"args": ["a"]},
        "bad args": {"command": "x", "args": [1]},
        "bad env": {"command": "x", "env": {"K": 3}},
        "bad=name": {"command": "x"},
        "bad var": {"command": "x", "env": {"1X": "v"}},
        "ok": {"command": "x", "type": "stdio"}
    }}"#;
    let declared = parse(text, &Scope::User, "f");
    let names: Vec<&str> = declared
        .servers
        .iter()
        .map(|s| s.key.name.as_str())
        .collect();
    assert_eq!(names, ["ok"]);
    let refused = |name: &str| {
        declared
            .problems
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.sentence.clone())
            .unwrap_or_default()
    };
    for remote in ["web", "sse", "http"] {
        assert_eq!(refused(remote), REMOTE_REFUSED, "{remote}");
    }
    assert!(refused("nothing").contains("no command"));
    assert!(refused("bad args").contains("args"));
    assert!(refused("bad env").contains("not text"));
    assert!(refused("bad=name").contains("name"));
    assert!(refused("bad var").contains("variable name"));
    assert!(
        parse("{nope", &Scope::User, "f").problems[0]
            .sentence
            .contains("not JSON")
    );
    assert!(parse("{}", &Scope::User, "f").servers.is_empty());
    assert!(
        parse(r#"{"mcpServers": []}"#, &Scope::User, "f").problems[0]
            .sentence
            .contains("not an object")
    );
}

/// The pin: the canonical JSON sorts keys at every level and has no
/// whitespace, so the hash moves with any value and with nothing else.
#[test]
fn the_pin_is_the_hash_of_the_canonical_json_and_moves_with_any_value() {
    let a = json!({"command": "x", "env": {"B": "2", "A": "1"}, "args": ["p", "q"]});
    assert_eq!(
        canonical(&a),
        r#"{"args":["p","q"],"command":"x","env":{"A":"1","B":"2"}}"#
    );
    let reordered: Value = serde_json::from_str(
        r#"{ "env" : { "A" : "1", "B" : "2" }, "command" : "x", "args" : [ "p" , "q" ] }"#,
    )
    .unwrap();
    let pin = |value: &Value| entry_of("s", value, &Scope::User, "f").unwrap().sha256;
    assert_eq!(pin(&a), pin(&reordered));
    let one_value = json!({"command": "x", "env": {"B": "2", "A": "one"}, "args": ["p", "q"]});
    assert_ne!(pin(&a), pin(&one_value), "a changed value asks again");
    let one_arg = json!({"command": "x", "env": {"B": "2", "A": "1"}, "args": ["q", "p"]});
    assert_ne!(pin(&a), pin(&one_arg), "argument order matters");
}

#[test]
fn pasted_text_may_be_a_whole_file_named_entries_or_one_entry() {
    let file = pasted(
        r#"{"mcpServers": {"a": {"command": "x"}, "b": {"command": "y"}}}"#,
        "",
    )
    .unwrap();
    assert_eq!(
        file.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        ["a", "b"]
    );
    let named = pasted(r#"{"c": {"command": "z"}}"#, "").unwrap();
    assert_eq!(named[0].0, "c");
    let one = pasted(r#"{"command": "z", "args": ["1"]}"#, "mine").unwrap();
    assert_eq!(
        one,
        vec![("mine".to_owned(), json!({"command": "z", "args": ["1"]}))]
    );
    assert!(
        pasted(r#"{"command": "z"}"#, " ")
            .unwrap_err()
            .contains("name")
    );
    assert!(
        pasted(r#"{"web": {"url": "https://example.invalid"}}"#, "")
            .unwrap_err()
            .contains("Remote MCP servers")
    );
    assert!(pasted("not json", "x").is_err());
}

#[test]
fn the_readers_file_is_edited_in_place_and_never_over_a_file_it_cannot_read() {
    let dir = TempDir::new("mcp-config");
    let state = StateRoot::at(dir.path());
    config::put_user_server(&state, "files", &json!({"command": "x", "args": ["1"]})).unwrap();
    config::put_user_server(&state, "docs", &json!({"command": "y"})).unwrap();
    config::set_user_disabled(&state, "docs", true).unwrap();
    let declared = config::load_user(&state);
    assert_eq!(declared.servers.len(), 2);
    assert!(declared.servers[0].disabled, "docs");
    config::set_user_disabled(&state, "docs", false).unwrap();
    config::remove_user_server(&state, "files").unwrap();
    let declared = config::load_user(&state);
    assert_eq!(declared.servers.len(), 1);
    assert!(!declared.servers[0].disabled);
    assert!(config::remove_user_server(&state, "files").is_err());
    // An entry Lattice would not use is not written.
    assert!(
        config::put_user_server(&state, "web", &json!({"url": "https://example.invalid"})).is_err()
    );
    // A file it cannot parse is reported and never overwritten.
    let file = config::user_file(&state);
    std::fs::write(&file, "{ broken").unwrap();
    assert!(
        config::load_user(&state).problems[0]
            .sentence
            .contains("not JSON")
    );
    assert_eq!(
        config::put_user_server(&state, "z", &json!({"command": "z"})).unwrap_err(),
        config::NOT_CHANGED
    );
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "{ broken");
}

/// An entry is edited with its values masked; saved back, a masked value
/// keeps what the file has, and a new variable must be given a value.
#[test]
fn an_edit_shows_no_value_and_a_kept_value_stays() {
    let dir = TempDir::new("mcp-kept");
    let state = StateRoot::at(dir.path());
    config::put_user_server(
        &state,
        "gh",
        &json!({"command": "x", "env": {"TOKEN": "t-1"}}),
    )
    .unwrap();
    let masked = config::user_entry_masked(&state, "gh").unwrap();
    assert_eq!(masked["env"]["TOKEN"], config::KEPT_VALUE);
    assert!(!masked.to_string().contains("t-1"));
    let mut edited = masked.clone();
    edited["args"] = json!(["--verbose"]);
    config::put_user_server(&state, "gh", &edited).unwrap();
    let entry = config::load_user(&state).servers.remove(0);
    assert_eq!(entry.env, [("TOKEN".to_owned(), Some("t-1".to_owned()))]);
    assert_eq!(entry.args, ["--verbose"]);
    let mut added = masked;
    added["env"]["OTHER"] = json!(config::KEPT_VALUE);
    assert!(
        config::put_user_server(&state, "gh", &added)
            .unwrap_err()
            .contains("OTHER")
    );
    assert!(config::user_entry_masked(&state, "nobody").is_none());
}

// ------------------------------------------------------------- approvals

fn ticking() -> Clock {
    let next = Arc::new(Mutex::new(1000.0));
    Arc::new(move || {
        let mut next = next.lock().unwrap();
        *next += 1.0;
        *next
    })
}

fn entry(scope: &Scope, command: &str) -> config::ServerEntry {
    entry_of("files", &json!({"command": command}), scope, "f").unwrap()
}

#[test]
fn an_approval_holds_for_exactly_its_entry_until_it_is_revoked() {
    let dir = TempDir::new("mcp-approvals");
    let approvals = Approvals::new(&StateRoot::at(dir.path()), ticking());
    let first = entry(&Scope::User, "one");
    assert!(!approvals.approved(&first), "nothing is enabled at first");
    approvals.approve(&first).unwrap();
    assert!(approvals.approved(&first));
    let changed = entry(&Scope::User, "two");
    assert!(!approvals.approved(&changed), "any change asks again");
    approvals.revoke(&first.key).unwrap();
    assert!(!approvals.approved(&first));
    approvals.approve(&first).unwrap();
    assert!(approvals.approved(&first), "enabled again after a new yes");
    // A folder's server is its own, matched by the folder's id and path.
    let in_folder = entry(&folder(), "one");
    assert!(!approvals.approved(&in_folder));
    approvals.approve(&in_folder).unwrap();
    let same_folder_other_case = entry(
        &Scope::Folder {
            id: "ab".repeat(16),
            path: r"c:\work\repo".to_owned(),
        },
        "one",
    );
    assert!(approvals.approved(&same_folder_other_case));
    let other_id = entry(
        &Scope::Folder {
            id: "cd".repeat(16),
            path: r"\\?\C:\Work\Repo".to_owned(),
        },
        "one",
    );
    assert!(!approvals.approved(&other_id));
}

#[test]
fn a_revocation_at_the_same_moment_wins() {
    let dir = TempDir::new("mcp-tie");
    let still: Clock = Arc::new(|| 5.0);
    let approvals = Approvals::new(&StateRoot::at(dir.path()), still);
    let e = entry(&Scope::User, "one");
    approvals.approve(&e).unwrap();
    approvals.revoke(&e.key).unwrap();
    assert!(!approvals.approved(&e), "a tie fails closed");
}

#[test]
fn allow_always_holds_for_one_tool_of_one_entry_and_switches_default_on() {
    let dir = TempDir::new("mcp-allow");
    let approvals = Approvals::new(&StateRoot::at(dir.path()), ticking());
    let e = entry(&Scope::User, "one");
    assert!(!approvals.allowed(&e, "read"));
    approvals.allow(&e, "read").unwrap();
    assert!(approvals.allowed(&e, "read"));
    assert!(!approvals.allowed(&e, "write"));
    assert!(
        !approvals.allowed(&entry(&Scope::User, "two"), "read"),
        "a changed entry"
    );
    approvals.disallow(&e.key, "read").unwrap();
    assert!(!approvals.allowed(&e, "read"));
    assert!(approvals.allow(&e, WHOLE_SERVER).is_err());
    assert!(approvals.is_on(&e.key, "write"));
    approvals.switch(&e.key, "write", false).unwrap();
    assert!(!approvals.is_on(&e.key, "write"));
    assert!(approvals.is_on(&e.key, "read"));
    approvals.switch(&e.key, "write", true).unwrap();
    assert!(approvals.is_on(&e.key, "write"));
}

#[test]
fn an_unreadable_approvals_file_approves_nothing_and_is_never_overwritten() {
    let dir = TempDir::new("mcp-unreadable");
    let state = StateRoot::at(dir.path());
    let approvals = Approvals::new(&state, ticking());
    let e = entry(&Scope::User, "one");
    approvals.approve(&e).unwrap();
    approvals.allow(&e, "read").unwrap();
    std::fs::write(approvals.file(), b"{ not json").unwrap();
    assert!(!approvals.approved(&e));
    assert!(!approvals.allowed(&e, "read"));
    assert!(approvals.approve(&e).is_err());
    assert_eq!(std::fs::read(approvals.file()).unwrap(), b"{ not json");
}

// ------------------------------------------------------------------ names

#[test]
fn a_model_name_is_prefixed_sanitised_bounded_and_unique() {
    assert_eq!(names::model_name("files", "read"), "mcp__files__read");
    assert_eq!(
        names::model_name("my docs", "get-page.v2"),
        "mcp__my_docs__get_page_v2"
    );
    let long = names::model_name(&"s".repeat(40), &"t".repeat(40));
    assert_eq!(long.len(), MAX_MODEL_NAME);
    assert!(long.starts_with("mcp__ssss"));
    let mut taken = HashSet::new();
    let first = names::unique_model_name("a__b", "c", &mut taken);
    let second = names::unique_model_name("a", "b__c", &mut taken);
    assert_eq!(first, "mcp__a__b__c");
    assert_ne!(first, second, "two tools never share a name");
    assert!(second.len() <= MAX_MODEL_NAME);
    for name in [&first, &second, &long] {
        assert!(
            name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "{name}"
        );
        assert!(names::is_model_name(name));
    }
    assert!(!names::is_model_name("read_file"));
}

// ----------------------------------------------------------------- results

#[test]
fn a_result_gives_its_text_names_what_it_cannot_pass_on_and_is_cut_at_32_kib() {
    let mixed = json!({"content": [
        {"type": "text", "text": "line one"},
        {"type": "image", "data": "AAAAAAAA", "mimeType": "image/png"},
        {"type": "resource", "resource": {"uri": "file:///a.txt", "text": "inside"}},
        {"type": "resource_link", "uri": "file:///b.txt", "name": "b"},
        {"type": "video"}
    ]});
    let text = call_text(&mixed);
    assert!(!text.is_error && !text.cut);
    let lines: Vec<&str> = text.text.lines().collect();
    assert_eq!(lines[0], "line one");
    assert!(
        lines[1].starts_with("[An image (image/png, 6 bytes)"),
        "{}",
        lines[1]
    );
    assert_eq!(lines[2], "Resource file:///a.txt:");
    assert_eq!(lines[3], "inside");
    assert_eq!(lines[4], "[Resource: b <file:///b.txt>]");
    assert!(lines[5].contains("type video"));
    let structured = call_text(&json!({"content": [], "structuredContent": {"sum": 3}}));
    assert_eq!(structured.text, r#"{"sum":3}"#);
    let failed = call_text(&json!({"content": [{"type": "text", "text": "no"}], "isError": true}));
    assert!(failed.is_error);
    assert_eq!(failed.text, "no");
    let wide = "\u{e9}".repeat(MAX_RESULT);
    let cut = call_text(&json!({"content": [{"type": "text", "text": wide}]}));
    assert!(cut.cut);
    assert!(cut.text.len() <= MAX_RESULT + 100);
    assert!(cut.text.ends_with("more bytes are not shown.]"));
}

// ----------------------------------------------------------------- framing

#[test]
fn each_message_is_one_line_and_each_kind_is_read_back() {
    let line = jsonrpc::request(7, "tools/call", json!({"text": "a\nb"}));
    assert_eq!(line.matches('\n').count(), 1, "{line}");
    assert!(line.ends_with('\n'));
    assert_eq!(
        jsonrpc::parse(&line),
        vec![Incoming::Request {
            id: json!(7),
            method: "tools/call".to_owned(),
            params: json!({"text": "a\nb"}),
        }]
    );
    let answered = jsonrpc::parse(r#"{"jsonrpc":"2.0","id":3,"result":{"ok":true}}"#);
    assert_eq!(
        answered,
        vec![Incoming::Response {
            id: json!(3),
            outcome: Ok(json!({"ok": true}))
        }]
    );
    let refused =
        jsonrpc::parse(r#"{"jsonrpc":"2.0","id":4,"error":{"code":-32601,"message":"no"}}"#);
    assert!(matches!(&refused[0], Incoming::Response { outcome: Err(e), .. } if e.code == -32601));
    let told = jsonrpc::parse(r#"{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}"#);
    assert!(
        matches!(&told[0], Incoming::Notification { method, .. } if method == "notifications/tools/list_changed")
    );
    let batch =
        jsonrpc::parse(r#"[{"jsonrpc":"2.0","id":1,"result":1},{"jsonrpc":"2.0","method":"x"}]"#);
    assert_eq!(batch.len(), 2);
    for junk in [
        "not json",
        "[]",
        r#"{"id":1,"result":1}"#,
        r#"{"jsonrpc":"2.0"}"#,
        "3",
    ] {
        assert!(
            matches!(jsonrpc::parse(junk).as_slice(), [Incoming::Invalid(_)]),
            "{junk}"
        );
    }
}

// ------------------------------------------------------- the connection

/// An in-process server: it reads the client's lines and answers with what
/// `answer` returns for each (`None`: no answer). It records each line.
struct Fake {
    conn: Arc<Connection>,
    seen: Arc<Mutex<Vec<Value>>>,
    events: Arc<Mutex<Vec<Event>>>,
    runtime: tokio::runtime::Runtime,
}

fn fake(answer: impl Fn(&Value) -> Vec<Value> + Send + 'static) -> Fake {
    let (client_reads, server_writes) = std::io::pipe().unwrap();
    let (server_reads, client_writes) = std::io::pipe().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_by_server = seen.clone();
    std::thread::spawn(move || {
        let mut out = server_writes;
        for line in BufReader::new(server_reads).lines() {
            let Ok(line) = line else { break };
            let Ok(message) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            seen_by_server.lock().unwrap().push(message.clone());
            for reply in answer(&message) {
                let mut text = serde_json::to_string(&reply).unwrap();
                text.push('\n');
                if out.write_all(text.as_bytes()).is_err() {
                    return;
                }
            }
        }
    });
    let events = Arc::new(Mutex::new(Vec::new()));
    let recorded = events.clone();
    let conn = Connection::start(
        "fake",
        Box::new(client_reads),
        Box::new(client_writes),
        Arc::new(move |event| recorded.lock().unwrap().push(event)),
    )
    .unwrap();
    Fake {
        conn,
        seen,
        events,
        runtime: tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap(),
    }
}

fn reply(message: &Value, result: Value) -> Vec<Value> {
    vec![json!({"jsonrpc": "2.0", "id": message["id"], "result": result})]
}

fn method(message: &Value) -> &str {
    message.get("method").and_then(Value::as_str).unwrap_or("")
}

const WAIT: Duration = Duration::from_secs(10);

#[test]
fn the_handshake_asks_for_the_newest_version_accepts_an_older_one_and_says_initialized() {
    let f = fake(|message| match method(message) {
        "initialize" => reply(
            message,
            json!({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "2"}, "instructions": "Be careful."}),
        ),
        _ => Vec::new(),
    });
    let info = f
        .runtime
        .block_on(client::handshake(&f.conn, WAIT))
        .unwrap();
    assert_eq!(info.protocol, "2025-06-18");
    assert_eq!((info.name.as_str(), info.version.as_str()), ("fake", "2"));
    assert_eq!(info.instructions.as_deref(), Some("Be careful."));
    crate::testkit::within("the initialized notification", 10, {
        let seen = f.seen.clone();
        move || loop {
            if seen.lock().unwrap().len() >= 2 {
                break;
            }
            std::thread::yield_now();
        }
    });
    let seen = f.seen.lock().unwrap();
    assert_eq!(
        seen[0]["params"]["protocolVersion"],
        client::REQUESTED_VERSION
    );
    assert_eq!(seen[0]["params"]["clientInfo"]["name"], "lattice");
    assert_eq!(method(&seen[1]), "notifications/initialized");
    assert!(seen[1].get("id").is_none());
}

#[test]
fn a_server_speaking_another_version_is_closed() {
    let f = fake(|message| match method(message) {
        "initialize" => reply(
            message,
            json!({"protocolVersion": "2026-07-28", "capabilities": {}}),
        ),
        _ => Vec::new(),
    });
    let refused = f
        .runtime
        .block_on(client::handshake(&f.conn, WAIT))
        .unwrap_err();
    assert!(refused.contains("2026-07-28"), "{refused}");
    assert!(f.conn.closed().is_some());
}

#[test]
fn the_tool_list_follows_its_cursor_and_bounds_what_it_keeps() {
    let f = fake(|message| match method(message) {
        "tools/list" if message["params"].get("cursor").is_none() => reply(
            message,
            json!({"tools": [
                {"name": "read", "description": "Reads.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
                {"name": "no_schema"},
                {"name": "", "inputSchema": {"type": "object"}}
            ], "nextCursor": "next"}),
        ),
        "tools/list" => reply(
            message,
            json!({"tools": [
                {"name": "read", "inputSchema": {"type": "object"}},
                {"name": "array_args", "inputSchema": {"type": "array"}},
                {"name": "write", "annotations": {"destructiveHint": true, "title": "Write a file"}}
            ]}),
        ),
        _ => Vec::new(),
    });
    let (tools, problems) = f
        .runtime
        .block_on(client::list_tools(&f.conn, WAIT))
        .unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["read", "no_schema", "write"]);
    assert_eq!(
        tools[1].input_schema,
        json!({"type": "object", "properties": {}})
    );
    assert_eq!(tools[2].destructive, Some(true));
    assert_eq!(tools[2].title.as_deref(), Some("Write a file"));
    assert_eq!(problems.len(), 3, "{problems:?}");
    assert!(problems.iter().any(|p| p.contains("listed twice")));
    assert!(problems.iter().any(|p| p.contains("not an object")));
}

#[test]
fn a_call_returns_its_result_or_the_servers_error() {
    let f = fake(|message| match method(message) {
        "tools/call" if message["params"]["name"] == "ok" => reply(
            message,
            json!({"content": [{"type": "text", "text": "fine"}]}),
        ),
        "tools/call" => vec![
            json!({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32602, "message": "unknown tool"}}),
        ],
        _ => Vec::new(),
    });
    let ok = f
        .runtime
        .block_on(client::call_tool(&f.conn, "ok", json!({"a": 1}), WAIT))
        .unwrap();
    assert_eq!(ok["content"][0]["text"], "fine");
    let error = f
        .runtime
        .block_on(client::call_tool(&f.conn, "other", json!(null), WAIT))
        .unwrap_err();
    assert!(matches!(&error, CallError::Rpc(e) if e.code == -32602));
    assert!(error.sentence().contains("unknown tool"));
    let seen = f.seen.lock().unwrap();
    assert_eq!(
        seen[1]["params"]["arguments"],
        json!({}),
        "arguments are always an object"
    );
}

/// A request given up is withdrawn with `notifications/cancelled`, naming
/// its id; `initialize` never is.
#[test]
fn a_timed_out_call_is_withdrawn_with_cancelled() {
    let f = fake(|_| Vec::new());
    let error = f
        .runtime
        .block_on(client::call_tool(
            &f.conn,
            "slow",
            json!({}),
            Duration::from_millis(50),
        ))
        .unwrap_err();
    assert_eq!(error, CallError::TimedOut);
    crate::testkit::within("the cancellation", 10, {
        let seen = f.seen.clone();
        move || loop {
            if seen
                .lock()
                .unwrap()
                .iter()
                .any(|m| method(m) == "notifications/cancelled")
            {
                break;
            }
            std::thread::yield_now();
        }
    });
    {
        let seen = f.seen.lock().unwrap();
        let cancelled = seen
            .iter()
            .find(|m| method(m) == "notifications/cancelled")
            .unwrap();
        assert_eq!(cancelled["params"]["requestId"], seen[0]["id"]);
    }
    let refused = f
        .runtime
        .block_on(client::handshake(&f.conn, Duration::from_millis(50)))
        .unwrap_err();
    assert!(refused.contains("did not finish starting"), "{refused}");
    std::thread::sleep(Duration::from_millis(100));
    let seen = f.seen.lock().unwrap();
    let initialize = seen.iter().find(|m| method(m) == "initialize").unwrap()["id"].clone();
    assert!(
        !seen
            .iter()
            .any(|m| method(m) == "notifications/cancelled"
                && m["params"]["requestId"] == initialize),
        "initialize is never cancelled"
    );
}

/// The server's own requests: a ping is answered, anything else refused;
/// its notifications reach the holder; what is not a message is logged.
#[test]
fn the_servers_ping_is_answered_and_its_other_requests_refused() {
    let f = fake(|message| match method(message) {
        "tools/call" => vec![
            json!({"jsonrpc": "2.0", "id": "p1", "method": "ping"}),
            json!({"jsonrpc": "2.0", "id": "s1", "method": "sampling/createMessage", "params": {}}),
            json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}),
            json!({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "warning", "data": "low disk"}}),
            json!({"not": "a message"}),
            json!({"jsonrpc": "2.0", "id": message["id"], "result": {"content": []}}),
        ],
        _ => Vec::new(),
    });
    f.runtime
        .block_on(client::call_tool(&f.conn, "x", json!({}), WAIT))
        .unwrap();
    crate::testkit::within("the replies", 10, {
        let seen = f.seen.clone();
        move || loop {
            if seen.lock().unwrap().len() >= 3 {
                break;
            }
            std::thread::yield_now();
        }
    });
    let seen = f.seen.lock().unwrap();
    let ping = seen.iter().find(|m| m["id"] == "p1").unwrap();
    assert_eq!(ping["result"], json!({}));
    let sampling = seen.iter().find(|m| m["id"] == "s1").unwrap();
    assert_eq!(sampling["error"]["code"], jsonrpc::METHOD_NOT_FOUND);
    let events = f.events.lock().unwrap();
    assert!(events.contains(&Event::ToolsChanged));
    assert!(events.contains(&Event::Log("[warning] low disk".to_owned())));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, Event::Log(line) if line.starts_with("(not an MCP message)")))
    );
}

#[test]
fn when_the_server_stops_a_waiting_call_ends_with_the_reason() {
    let f = fake(|message| match method(message) {
        // The server's output ends as it reads this: its thread returns.
        "tools/call" => vec![json!({"jsonrpc": "2.0", "id": "end", "method": "bye"})],
        _ => Vec::new(),
    });
    // Close the server side by dropping the connection's peer: here the
    // fake keeps writing, so end it by closing the connection instead.
    let conn = f.conn.clone();
    let waiting = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(client::call_tool(&conn, "x", json!({}), WAIT))
    });
    std::thread::sleep(Duration::from_millis(100));
    f.conn.close();
    let error = waiting.join().unwrap().unwrap_err();
    assert!(matches!(error, CallError::Closed(_)), "{error:?}");
    assert!(
        f.runtime
            .block_on(client::call_tool(&f.conn, "x", json!({}), WAIT))
            .is_err()
    );
}

#[test]
fn a_server_that_writes_garbage_first_still_connects() {
    let f = fake(|message| match method(message) {
        "initialize" => {
            let mut out = vec![json!("plain words on stdout")];
            out.extend(reply(
                message,
                json!({"protocolVersion": "2024-11-05", "capabilities": {}}),
            ));
            out
        }
        _ => Vec::new(),
    });
    let info = f
        .runtime
        .block_on(client::handshake(&f.conn, WAIT))
        .unwrap();
    assert_eq!(info.protocol, "2024-11-05");
}

#[test]
fn a_server_key_names_its_scope() {
    assert_eq!(ServerKey::user("files").id(), "user/files");
    let key = ServerKey {
        scope: folder(),
        name: "x".to_owned(),
    };
    assert!(key.id().starts_with("folder/abab"));
}
