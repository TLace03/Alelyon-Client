//! An MCP server over stdio: newline-delimited JSON-RPC 2.0 on standard input and output, as the Model Context
//! Protocol's stdio transport describes it. Logs go to standard error; standard output carries only messages.
//!
//! It offers the actions in [`crate::tools`] as tools. Each call goes to the identity service with the person's
//! developer token, which decides what is allowed; a refusal comes back as a tool error whose text says why, so the
//! agent can tell the person which scope to add.

use std::io::{BufRead, Write};

use serde_json::{Value, json};

use crate::api::Pages;
use crate::tools::{APP, APP_TOOLS, TOOLS, Tool, check_arguments, find};

/// The protocol versions this server speaks, newest first.
pub const VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// The largest message read; a longer line is answered with a parse error and skipped.
const MAX_LINE: usize = 1 << 20;

fn reply(id: &Value, result: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "result": result})
}

fn error(id: &Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// What the server can reach: Pages with a developer token, and the app on this computer through its bridge.
pub struct Reach<'a> {
    pub pages: Option<&'a dyn Pages>,
    pub app: &'a dyn Pages,
}

fn tool_list(reach: &Reach) -> Value {
    let pages: &[Tool] = if reach.pages.is_some() { TOOLS } else { &[] };
    let tools: Vec<Value> = pages
        .iter()
        .chain(APP_TOOLS)
        .map(|t| {
            let needs = if t.scope == APP {
                "Needs the Alelyon app open with Settings, Agents on.".to_string()
            } else {
                format!("Needs the {} scope.", t.scope)
            };
            json!({
                "name": t.name,
                "description": format!("{} {needs}", t.description),
                "inputSchema": t.input_schema(),
                "annotations": {
                    "readOnlyHint": t.read_only,
                    "destructiveHint": t.destructive,
                    "openWorldHint": true,
                },
            })
        })
        .collect();
    json!({"tools": tools})
}

fn call_tool(reach: &Reach, params: &Value) -> Value {
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    let Some(tool) = find(name) else {
        return json!({"content": [{"type": "text", "text": format!("There is no tool named {name}.")}], "isError": true});
    };
    let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
    if let Err(why) = check_arguments(tool, &arguments) {
        return json!({"content": [{"type": "text", "text": why}], "isError": true});
    }
    let body = if arguments.is_null() { json!({}) } else { arguments };
    let target: &dyn Pages = match (tool.scope == APP, reach.pages) {
        (true, _) => reach.app,
        (false, Some(pages)) => pages,
        (false, None) => {
            let words = "No developer token: make one at https://www.alelyon.com/developers/ and run `alelyon-cli auth login`.";
            return json!({"content": [{"type": "text", "text": words}], "isError": true});
        }
    };
    match target.call(tool.route, &body) {
        Ok(answer) => json!({
            "content": [{"type": "text", "text": serde_json::to_string(&answer).unwrap_or_default()}],
            "structuredContent": answer,
            "isError": false,
        }),
        Err(failure) => {
            let mut words = failure.to_string();
            if failure.code == "scope-missing" {
                words.push_str(" Add that scope to the token in the developer console, or make a new token with it.");
            }
            json!({"content": [{"type": "text", "text": words}], "isError": true})
        }
    }
}

/// The answer to one message, or None for a notification (which is never answered).
pub fn handle(reach: &Reach, message: &Value) -> Option<Value> {
    let id = message.get("id").cloned();
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    let Some(id) = id else {
        // A notification (notifications/initialized, notifications/cancelled, ...): nothing to say back.
        return None;
    };
    if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Some(error(&id, -32600, "not a JSON-RPC 2.0 request"));
    }
    let params = message.get("params").cloned().unwrap_or(json!({}));
    Some(match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or("");
            let version = if VERSIONS.contains(&asked) { asked } else { VERSIONS[0] };
            reply(
                &id,
                json!({
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "alelyon", "title": "Alelyon Pages", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": "Tools for Alelyon Pages, acting as the person who made this developer token. \
                                     Read before you act. Posts you make are labeled with the token's name. Ask the \
                                     person before you post, delete or follow on their behalf unless they told you to.",
                }),
            )
        }
        "ping" => reply(&id, json!({})),
        "tools/list" => reply(&id, tool_list(reach)),
        "tools/call" => reply(&id, call_tool(reach, &params)),
        "resources/list" => reply(&id, json!({"resources": []})),
        "prompts/list" => reply(&id, json!({"prompts": []})),
        _ => error(&id, -32601, "method not found"),
    })
}

/// Serve until standard input closes.
pub fn serve(reach: &Reach, input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
    for line in input.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let answer = if line.len() > MAX_LINE {
            Some(error(&Value::Null, -32700, "message too large"))
        } else {
            match serde_json::from_str::<Value>(line) {
                Ok(message) if message.is_object() => handle(reach, &message),
                Ok(_) => Some(error(&Value::Null, -32600, "batches are not supported")),
                Err(_) => Some(error(&Value::Null, -32700, "parse error")),
            }
        };
        if let Some(answer) = answer {
            writeln!(output, "{}", serde_json::to_string(&answer).unwrap_or_default())?;
            output.flush()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Failure;
    use std::cell::RefCell;

    struct Fake {
        calls: RefCell<Vec<(String, Value)>>,
        refuse: Option<Failure>,
    }

    impl Pages for Fake {
        fn call(&self, route: &str, body: &Value) -> Result<Value, Failure> {
            self.calls.borrow_mut().push((route.to_string(), body.clone()));
            match &self.refuse {
                Some(f) => Err(f.clone()),
                None => Ok(json!({"post": {"id": "pst_1", "via": "My agent"}})),
            }
        }
    }

    fn fake() -> Fake {
        Fake { calls: RefCell::new(Vec::new()), refuse: None }
    }

    fn run(pages: &Fake, lines: &[Value]) -> Vec<Value> {
        let input: String = lines.iter().map(|l| format!("{l}\n")).collect();
        let mut out = Vec::new();
        serve(&Reach { pages: Some(pages), app: pages }, input.as_bytes(), &mut out).unwrap();
        String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    #[test]
    fn a_session_initializes_lists_and_calls() {
        let pages = fake();
        let out = run(
            &pages,
            &[
                json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}),
                json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "pages_post", "arguments": {"text": "Hello"}}}),
            ],
        );
        assert_eq!(out.len(), 3, "the notification is not answered");
        assert_eq!(out[0]["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(out[1]["result"]["tools"].as_array().unwrap().len(), TOOLS.len() + APP_TOOLS.len());
        assert_eq!(out[2]["result"]["isError"], false);
        assert_eq!(out[2]["result"]["structuredContent"]["post"]["via"], "My agent");
        assert_eq!(pages.calls.borrow()[0], ("post".to_string(), json!({"text": "Hello"})));
    }

    #[test]
    fn without_a_token_only_the_app_tools_are_offered_and_they_reach_the_app() {
        let app = fake();
        let lines = [
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                   "params": {"name": "app_chat_send", "arguments": {"text": "hi"}}}),
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                   "params": {"name": "pages_feed", "arguments": {}}}),
            json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                   "params": {"name": "fleet_answer", "arguments": {"topic": "t1", "call": "q1", "text": "CSV"}}}),
        ];
        let input: String = lines.iter().map(|l| format!("{l}\n")).collect();
        let mut out = Vec::new();
        serve(&Reach { pages: None, app: &app }, input.as_bytes(), &mut out).unwrap();
        let got: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let tools = got[0]["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), APP_TOOLS.len());
        assert!(tools.iter().all(|t| !t["name"].as_str().unwrap().starts_with("pages_")));
        assert_eq!(app.calls.borrow()[0], ("chat_send".to_string(), json!({"text": "hi"})));
        assert_eq!(got[2]["result"]["isError"], true, "a Pages tool without a token says how to make one");
        assert_eq!(app.calls.borrow()[1], ("fleet_answer".to_string(), json!({"topic": "t1", "call": "q1", "text": "CSV"})));
        assert_eq!(app.calls.borrow().len(), 2);
    }

    #[test]
    fn an_unknown_version_gets_the_newest() {
        let out = run(&fake(), &[json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                        "params": {"protocolVersion": "1999-01-01"}})]);
        assert_eq!(out[0]["result"]["protocolVersion"], VERSIONS[0]);
    }

    #[test]
    fn bad_arguments_never_reach_the_service() {
        let pages = fake();
        let out = run(&pages, &[json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                       "params": {"name": "pages_post", "arguments": {"text": "x", "refresh_token": "t"}}})]);
        assert_eq!(out[0]["result"]["isError"], true);
        assert!(pages.calls.borrow().is_empty());
    }

    #[test]
    fn a_refusal_is_a_tool_error_that_says_what_to_do() {
        let pages = Fake {
            calls: RefCell::new(Vec::new()),
            refuse: Some(Failure { status: 403, code: "scope-missing".into(),
                                   message: "This token was not given the pages:post scope.".into() }),
        };
        let out = run(&pages, &[json!({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                                       "params": {"name": "pages_post", "arguments": {"text": "x"}}})]);
        let text = out[0]["result"]["content"][0]["text"].as_str().unwrap();
        assert_eq!(out[0]["result"]["isError"], true);
        assert!(text.contains("pages:post") && text.contains("developer console"), "{text}");
    }

    #[test]
    fn garbage_and_unknown_methods_are_answered_not_fatal() {
        let input = "not json\n{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"nope\"}\n";
        let mut out = Vec::new();
        let f = fake();
        serve(&Reach { pages: Some(&f), app: &f }, input.as_bytes(), &mut out).unwrap();
        let lines: Vec<Value> = String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines[0]["error"]["code"], -32700);
        assert_eq!(lines[1]["error"]["code"], -32601);
    }
}
