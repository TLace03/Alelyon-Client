//! A scripted MCP server on stdio, for the MCP client's tests
//! (the chat core's spec §12, "Tests"). Test tooling only: it is a binary
//! of this package so the tests can find it beside their own binary; nothing
//! ships it, and it reaches no network and no GPU.
//!
//! It speaks newline-delimited JSON-RPC on stdin and stdout, as an MCP server
//! does, and what it does is read from its arguments:
//! - `--version <v>`: answer `initialize` with this version (by default the
//!   version asked for);
//! - `--exit <code>`: write one line to stderr and exit at once;
//! - `--stderr <text>`: write `text` to stderr at start;
//! - `--record <file>`: append each method received to `<file>`, and write
//!   the names of its environment to `<file>.env` and its process id to
//!   `<file>.pid` at start;
//! - `--garbage`: write a line that is not JSON before anything else;
//! - `--crash-on <tool>`: exit with code 3 when that tool is called;
//! - `--pages`: list the tools in two pages;
//! - `--hang-init`: never answer `initialize`;
//! - `--list-changed`: after its first call, add the tool `late` and send
//!   `notifications/tools/list_changed`.
//!
//! Its tools: `echo {text}`, `add {a, b}` (text and structured content),
//! `fail` (`isError`), `big` (40 KiB of text), `image` (an image part),
//! `slow {ms}` (answers after `ms`), `env {name}` (that variable, or
//! `(unset)`), and `ping_client` (pings the client and answers once the
//! client has answered). Every other method is "method not found".

use std::io::{BufRead, Write};

use serde_json::{Value, json};

struct Args {
    version: Option<String>,
    exit: Option<i32>,
    stderr: Option<String>,
    record: Option<String>,
    garbage: bool,
    crash_on: Option<String>,
    pages: bool,
    hang_init: bool,
    list_changed: bool,
}

fn args() -> Args {
    let mut args = Args {
        version: None,
        exit: None,
        stderr: None,
        record: None,
        garbage: false,
        crash_on: None,
        pages: false,
        hang_init: false,
        list_changed: false,
    };
    let mut given = std::env::args().skip(1);
    while let Some(arg) = given.next() {
        match arg.as_str() {
            "--version" => args.version = given.next(),
            "--exit" => args.exit = given.next().and_then(|code| code.parse().ok()),
            "--stderr" => args.stderr = given.next(),
            "--record" => args.record = given.next(),
            "--garbage" => args.garbage = true,
            "--crash-on" => args.crash_on = given.next(),
            "--pages" => args.pages = true,
            "--hang-init" => args.hang_init = true,
            "--list-changed" => args.list_changed = true,
            _ => {}
        }
    }
    args
}

fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": properties, "required": required},
    })
}

fn tools(late: bool) -> Vec<Value> {
    let mut tools = vec![
        tool(
            "echo",
            "Say the text back.",
            json!({"text": {"type": "string"}}),
            &["text"],
        ),
        tool(
            "add",
            "Add two numbers.",
            json!({"a": {"type": "number"}, "b": {"type": "number"}}),
            &["a", "b"],
        ),
        tool("fail", "Always fails.", json!({}), &[]),
        tool("big", "A long answer.", json!({}), &[]),
        tool("image", "An image.", json!({}), &[]),
        tool(
            "slow",
            "Answers late.",
            json!({"ms": {"type": "integer"}}),
            &["ms"],
        ),
        tool(
            "env",
            "A variable of its environment.",
            json!({"name": {"type": "string"}}),
            &["name"],
        ),
        tool("ping_client", "Pings the client first.", json!({}), &[]),
    ];
    if late {
        tools.push(tool("late", "Listed after a change.", json!({}), &[]));
    }
    tools
}

fn send(out: &mut impl Write, value: &Value) {
    let mut line = serde_json::to_string(value).unwrap();
    line.push('\n');
    let _ = out.write_all(line.as_bytes());
    let _ = out.flush();
}

fn text_result(text: &str) -> Value {
    json!({"content": [{"type": "text", "text": text}]})
}

fn main() {
    let args = args();
    if let Some(text) = &args.stderr {
        eprintln!("{text}");
    }
    if let Some(code) = args.exit {
        eprintln!("stub: exiting with {code}");
        std::process::exit(code);
    }
    if let Some(record) = &args.record {
        let mut names: Vec<String> = std::env::vars_os()
            .map(|(name, _)| name.to_string_lossy().into_owned())
            .collect();
        names.sort();
        let _ = std::fs::write(format!("{record}.env"), names.join("\n"));
        let _ = std::fs::write(format!("{record}.pid"), std::process::id().to_string());
    }
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    if args.garbage {
        let _ = out.write_all(b"this is not json\n");
        let _ = out.flush();
    }
    let mut late = false;
    let mut calls = 0u32;
    let mut lines = stdin.lock().lines();
    while let Some(Ok(line)) = lines.next() {
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        if let Some(record) = &args.record {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(record)
            {
                let shown = if method.is_empty() {
                    "(response)"
                } else {
                    method.as_str()
                };
                let _ = writeln!(file, "{shown}");
            }
        }
        let Some(id) = message.get("id").cloned() else {
            continue;
        };
        if method.is_empty() {
            // A response to the stub's own ping: nothing to do here.
            continue;
        }
        let params = message.get("params").cloned().unwrap_or(Value::Null);
        let result = match method.as_str() {
            "initialize" => {
                if args.hang_init {
                    continue;
                }
                let asked = params
                    .get("protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or("2025-06-18");
                let version = args.version.clone().unwrap_or_else(|| asked.to_owned());
                Ok(json!({
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": true}},
                    "serverInfo": {"name": "lattice-mcp-stub", "version": "1.0"},
                    "instructions": "A stub for tests.",
                }))
            }
            "ping" => Ok(json!({})),
            "tools/list" => {
                let all = tools(late);
                if args.pages {
                    let half = all.len() / 2;
                    match params.get("cursor").and_then(Value::as_str) {
                        None => Ok(json!({"tools": all[..half], "nextCursor": "page-2"})),
                        Some(_) => Ok(json!({"tools": all[half..]})),
                    }
                } else {
                    Ok(json!({"tools": all}))
                }
            }
            "tools/call" => {
                calls += 1;
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
                if args.crash_on.as_deref() == Some(name) {
                    eprintln!("stub: crashing on {name}");
                    std::process::exit(3);
                }
                let answer = match name {
                    "echo" => Ok(text_result(
                        arguments.get("text").and_then(Value::as_str).unwrap_or(""),
                    )),
                    "add" => {
                        let a = arguments.get("a").and_then(Value::as_f64).unwrap_or(0.0);
                        let b = arguments.get("b").and_then(Value::as_f64).unwrap_or(0.0);
                        Ok(json!({
                            "content": [{"type": "text", "text": format!("{}", a + b)}],
                            "structuredContent": {"sum": a + b},
                        }))
                    }
                    "fail" => Ok(
                        json!({"content": [{"type": "text", "text": "it failed"}], "isError": true}),
                    ),
                    "big" => Ok(text_result(&"x".repeat(40 * 1024))),
                    "image" => Ok(
                        json!({"content": [{"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"}]}),
                    ),
                    "slow" => {
                        let ms = arguments.get("ms").and_then(Value::as_u64).unwrap_or(0);
                        std::thread::sleep(std::time::Duration::from_millis(ms));
                        Ok(text_result("done"))
                    }
                    "env" => {
                        let wanted = arguments.get("name").and_then(Value::as_str).unwrap_or("");
                        let value = std::env::var(wanted).unwrap_or_else(|_| "(unset)".to_owned());
                        Ok(text_result(&value))
                    }
                    "ping_client" => {
                        send(
                            &mut out,
                            &json!({"jsonrpc": "2.0", "id": "stub-ping", "method": "ping"}),
                        );
                        let mut answered = false;
                        while let Some(Ok(reply)) = lines.next() {
                            let Ok(reply) = serde_json::from_str::<Value>(&reply) else {
                                continue;
                            };
                            if reply.get("id") == Some(&json!("stub-ping")) {
                                answered = reply.get("result").is_some();
                                break;
                            }
                        }
                        Ok(text_result(if answered { "pong ok" } else { "no pong" }))
                    }
                    _ => Err((-32602, format!("no tool {name}"))),
                };
                if args.list_changed && calls == 1 && !late {
                    late = true;
                    send(
                        &mut out,
                        &json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}),
                    );
                }
                answer
            }
            _ => Err((-32601, "method not found".to_owned())),
        };
        match result {
            Ok(result) => send(
                &mut out,
                &json!({"jsonrpc": "2.0", "id": id, "result": result}),
            ),
            Err((code, message)) => send(
                &mut out,
                &json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}),
            ),
        }
    }
}
