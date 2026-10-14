//! The checklist store ([`crate::user_steps`]) as a Model Context Protocol
//! server, so an agent that runs inside Lattice over ACP gives the reader a
//! checklist and reads the reader's checks the way any other agent does.
//!
//! - **Transport:** newline-delimited JSON-RPC 2.0 on stdin and stdout (MCP's
//!   stdio transport), protocol versions [`PROTOCOLS`]. Requests are answered
//!   in the order they come, one at a time.
//! - **Tools:** `give_steps` (a title, 1 to 20 one-line steps, whether the list
//!   finishes by itself when every step is checked, and who asks) writes the
//!   list with the store's own writer ([`crate::user_steps::give`]), in the
//!   format `tools/user_steps.py` writes, and returns its id; `read_steps`
//!   reads a list with what the reader checked; `wait_for_steps` waits, at most
//!   [`MAX_WAIT_SECS`] seconds, until the list is done.
//! - **Served** by the desktop program started with [`FLAG`]; every ACP
//!   session Lattice opens is offered it ([`acp_entry`]), pointed at the store
//!   Lattice itself watches.
//! - The server writes only the list files it is given; it never writes a
//!   list's state, which is the reader's.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::user_steps::{self, ExternalList};

/// The protocol versions served, newest first.
pub const PROTOCOLS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];
/// The flag that starts the server instead of the window.
pub const FLAG: &str = "--steps-mcp";
/// The server's name, as an agent lists it.
pub const NAME: &str = "alelyon-checklists";
/// The longest `wait_for_steps` waits.
pub const MAX_WAIT_SECS: u64 = 300;
/// What `wait_for_steps` waits when not told.
pub const DEFAULT_WAIT_SECS: u64 = 60;

/// The `mcpServers` entry an ACP session is offered: `program` started with
/// [`FLAG`], on the store in `dir`.
pub fn acp_entry(program: &Path, dir: &Path) -> Value {
    json!({
        "name": NAME,
        "command": program.to_string_lossy(),
        "args": [FLAG],
        "env": [{"name": "ALELYON_STEPS_DIR", "value": dir.to_string_lossy()}],
    })
}

/// Serve the store in `dir` until the client closes stdin.
pub fn serve(dir: &Path, input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
    let folder = std::env::current_dir().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
    let server = Server::new(dir.to_path_buf(), folder);
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Some(reply) = server.handle_line(&line) {
            writeln!(output, "{reply}")?;
            output.flush()?;
        }
    }
    Ok(())
}

/// The server over one store.
pub struct Server {
    /// The store's folder.
    pub dir: PathBuf,
    /// The folder a list is given from, as the list records it.
    pub folder: String,
    /// How often `wait_for_steps` reads the list.
    pub poll: Duration,
}

impl Server {
    pub fn new(dir: PathBuf, folder: String) -> Self {
        Self { dir, folder, poll: Duration::from_millis(500) }
    }

    /// One line in, at most one line out (a notification gets none).
    pub fn handle_line(&self, line: &str) -> Option<Value> {
        let msg: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => return Some(json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "parse error"}})),
        };
        let id = msg.get("id").cloned()?;
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let params = msg.get("params").cloned().unwrap_or(json!({}));
        let result = match method {
            "initialize" => Ok(initialize(&params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tools()})),
            "tools/call" => Ok(self.call(&params)),
            _ => Err(format!("method not found: {method}")),
        };
        Some(match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err(message) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": message}}),
        })
    }

    fn call(&self, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        let outcome = match name {
            "give_steps" => self.give(&args),
            "read_steps" => self.list(&args).map(|list| shown(&list, false)),
            "wait_for_steps" => self.wait(&args),
            _ => Err(format!("unknown tool: {name}")),
        };
        match outcome {
            Ok((text, value)) => json!({
                "content": [{"type": "text", "text": text}],
                "structuredContent": value,
                "isError": false
            }),
            Err(why) => json!({"content": [{"type": "text", "text": why}], "isError": true}),
        }
    }

    fn give(&self, args: &Value) -> Result<(String, Value), String> {
        let title = args.get("title").and_then(Value::as_str).ok_or("give_steps needs a title.")?;
        let steps: Vec<String> = args
            .get("steps")
            .and_then(Value::as_array)
            .ok_or("give_steps needs steps: a list of one-line texts.")?
            .iter()
            .map(|s| s.as_str().map(str::to_owned).ok_or("Each step is one line of text."))
            .collect::<Result<_, _>>()?;
        let notify = args.get("notify_when_done").and_then(Value::as_bool).unwrap_or(true);
        let agent = args.get("agent").and_then(Value::as_str).unwrap_or("");
        let id = user_steps::give(&self.dir, title, &steps, agent, &self.folder, notify, now())?;
        let n = steps.len();
        let text = format!(
            "{id}\nThe reader sees the checklist ({n} step{}) pinned in Alelyon's window and checks each step as it is done. \
             read_steps {id} shows where it stands; wait_for_steps {id} returns once it is done ({}).",
            if n == 1 { "" } else { "s" },
            if notify { "when every step is checked" } else { "when the reader presses Done" }
        );
        Ok((text, json!({"id": id})))
    }

    fn list(&self, args: &Value) -> Result<ExternalList, String> {
        let id = args.get("id").and_then(Value::as_str).unwrap_or("").trim();
        if !user_steps::is_id(id) {
            return Err(format!("\u{201c}{id}\u{201d} is not a checklist's id (steps_ and 16 hex digits)."));
        }
        user_steps::read(&self.dir, id).ok_or_else(|| format!("There is no checklist {id}."))
    }

    fn wait(&self, args: &Value) -> Result<(String, Value), String> {
        let seconds = args.get("seconds").and_then(Value::as_u64).unwrap_or(DEFAULT_WAIT_SECS).clamp(1, MAX_WAIT_SECS);
        let until = Instant::now() + Duration::from_secs(seconds);
        loop {
            let list = self.list(args)?;
            let left = until.saturating_duration_since(Instant::now());
            if list.done || left.is_zero() {
                return Ok(shown(&list, !list.done));
            }
            std::thread::sleep(self.poll.min(left));
        }
    }
}

fn now() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// A list as text and as data; `waited` when a wait ended before it was done.
fn shown(list: &ExternalList, waited: bool) -> (String, Value) {
    let checked = list.checked.iter().filter(|c| **c).count();
    let mut text = format!(
        "{} ({}): {checked} of {} checked{}",
        list.title,
        list.id,
        list.steps.len(),
        if list.done { ", done" } else if waited { ", not done yet" } else { "" }
    );
    for (n, (step, done)) in list.steps.iter().zip(&list.checked).enumerate() {
        text.push_str(&format!("\n  {}. [{}] {step}", n + 1, if *done { "x" } else { " " }));
    }
    let steps: Vec<Value> = list.steps.iter().zip(&list.checked).map(|(s, c)| json!({"text": s, "checked": c})).collect();
    let mut value = json!({"id": list.id, "title": list.title, "steps": steps, "checked": checked, "total": list.steps.len(), "done": list.done});
    if waited {
        value["timed_out"] = json!(true);
    }
    (text, value)
}

fn initialize(params: &Value) -> Value {
    let asked = params.get("protocolVersion").and_then(Value::as_str).unwrap_or(PROTOCOLS[0]);
    let version = if PROTOCOLS.contains(&asked) { asked } else { PROTOCOLS[0] };
    json!({
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": false}},
        "serverInfo": {"name": NAME, "version": env!("CARGO_PKG_VERSION")},
        "instructions": "Checklists for the reader. When the reader has to do something themselves (sign in somewhere, \
            restart a program, approve a download, set a value), give it with give_steps instead of directions in your \
            answer: the reader sees it pinned in Alelyon's window and checks each step. read_steps shows what is checked; \
            wait_for_steps returns once the list is done (call it again when it returns not done yet)."
    })
}

fn tools() -> Value {
    let id = json!({"type": "string", "description": "The checklist's id, as give_steps returned it (steps_ and 16 hex digits)."});
    json!([
        {
            "name": "give_steps",
            "description": "Give the reader a checklist of things they do themselves (sign in somewhere, restart a program, approve a download). It is pinned in Alelyon's window until it is done, and the reader checks each step. Returns the list's id.",
            "inputSchema": {"type": "object", "properties": {
                "title": {"type": "string", "description": "What the list is for, in one line.", "maxLength": user_steps::MAX_TITLE_CHARS},
                "steps": {"type": "array", "minItems": 1, "maxItems": user_steps::MAX_STEPS,
                    "items": {"type": "string", "maxLength": user_steps::MAX_STEP_CHARS},
                    "description": "One line each: what the reader does, in order."},
                "notify_when_done": {"type": "boolean", "default": true,
                    "description": "True: the list is done when every step is checked. False: when the reader presses Done."},
                "agent": {"type": "string", "maxLength": user_steps::MAX_AGENT_CHARS,
                    "description": "Who asks, as the reader will see it (for example the session's name)."}
            }, "required": ["title", "steps"]},
            "annotations": {"readOnlyHint": false, "destructiveHint": false, "idempotentHint": false, "openWorldHint": false}
        },
        {
            "name": "read_steps",
            "description": "Read a checklist you gave: its title, each step checked or not, and whether it is done.",
            "inputSchema": {"type": "object", "properties": {"id": id}, "required": ["id"]},
            "annotations": {"readOnlyHint": true, "openWorldHint": false}
        },
        {
            "name": "wait_for_steps",
            "description": "Wait until the reader is done with a checklist, at most the given seconds; returns the list either way (timed_out when it is not done yet).",
            "inputSchema": {"type": "object", "properties": {
                "id": id,
                "seconds": {"type": "integer", "minimum": 1, "maximum": MAX_WAIT_SECS, "default": DEFAULT_WAIT_SECS}
            }, "required": ["id"]},
            "annotations": {"readOnlyHint": true, "openWorldHint": false}
        }
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    fn server(dir: &Path) -> Server {
        Server { dir: dir.to_path_buf(), folder: "C:/work".to_owned(), poll: Duration::from_millis(20) }
    }

    fn call(server: &Server, name: &str, args: Value) -> Value {
        let line = json!({"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": name, "arguments": args}});
        server.handle_line(&line.to_string()).unwrap()["result"].clone()
    }

    #[test]
    fn the_handshake_and_the_tool_list() {
        let dir = TempDir::new("steps-mcp-hello");
        let server = server(dir.path());
        let init = server
            .handle_line(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}"#)
            .unwrap();
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(init["result"]["serverInfo"]["name"], NAME);
        let other = server.handle_line(r#"{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"1999-01-01"}}"#).unwrap();
        assert_eq!(other["result"]["protocolVersion"], PROTOCOLS[0]);
        assert!(server.handle_line(r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#).is_none());
        let list = server.handle_line(r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#).unwrap();
        let names: Vec<&str> = list["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["give_steps", "read_steps", "wait_for_steps"]);
        let unknown = server.handle_line(r#"{"jsonrpc":"2.0","id":4,"method":"resources/list"}"#).unwrap();
        assert_eq!(unknown["error"]["code"], -32601);
        assert_eq!(server.handle_line("not json").unwrap()["error"]["code"], -32700);
        // Nothing was written by any of it.
        assert!(!dir.path().join("steps").exists() && std::fs::read_dir(dir.path()).unwrap().count() == 0);
    }

    #[test]
    fn a_given_list_is_read_back_with_the_readers_checks() {
        let dir = TempDir::new("steps-mcp-give");
        let store = dir.path().join("steps");
        let server = server(&store);
        let given = call(&server, "give_steps", json!({"title": "Connect a provider", "steps": ["Open the Models page.", "Sign in."], "agent": "Review session"}));
        assert_eq!(given["isError"], false, "{given}");
        let id = given["structuredContent"]["id"].as_str().unwrap().to_owned();
        // The list is in the store as the Python tool writes it, and the window reads it.
        let open = user_steps::open(&store);
        assert_eq!((open.len(), open[0].agent.as_str(), open[0].folder.as_str(), open[0].notify), (1, "Review session", "C:/work", true));
        user_steps::check(&store, &id, 0, true, 2.0).unwrap();
        let read = call(&server, "read_steps", json!({"id": id}));
        assert_eq!(read["structuredContent"]["checked"], 1);
        assert_eq!(read["structuredContent"]["done"], false);
        assert_eq!(read["structuredContent"]["steps"][0], json!({"text": "Open the Models page.", "checked": true}));
        assert!(read["content"][0]["text"].as_str().unwrap().contains("1. [x] Open the Models page."));
    }

    #[test]
    fn a_bad_list_or_id_is_refused_and_nothing_is_written() {
        let dir = TempDir::new("steps-mcp-refuse");
        let store = dir.path().join("steps");
        let server = server(&store);
        for args in [
            json!({"title": "x", "steps": []}),
            json!({"title": "x", "steps": ["two\nlines"]}),
            json!({"title": "x", "steps": ["s".repeat(201)]}),
            json!({"title": "x", "steps": (0..21).map(|n| format!("Step {n}")).collect::<Vec<_>>()}),
            json!({"title": "x", "steps": [3]}),
            json!({"steps": ["a"]}),
        ] {
            let refused = call(&server, "give_steps", args.clone());
            assert_eq!(refused["isError"], true, "{args}");
        }
        assert!(std::fs::read_dir(&store).map(|d| d.count()).unwrap_or(0) == 0);
        assert_eq!(call(&server, "read_steps", json!({"id": "../secret"}))["isError"], true);
        assert_eq!(call(&server, "read_steps", json!({"id": "steps_0011223344556677"}))["isError"], true);
        assert_eq!(call(&server, "wait_for_steps", json!({"id": "steps_0011223344556677"}))["isError"], true);
        assert_eq!(call(&server, "remove_steps", json!({}))["isError"], true);
    }

    #[test]
    fn a_wait_returns_once_the_list_is_done_or_its_time_is_up() {
        let dir = TempDir::new("steps-mcp-wait");
        let store = dir.path().join("steps");
        let server = server(&store);
        let given = call(&server, "give_steps", json!({"title": "Restart", "steps": ["Restart the program."]}));
        let id = given["structuredContent"]["id"].as_str().unwrap().to_owned();
        // Not done: the wait ends at its time, and says so.
        let started = Instant::now();
        let waited = call(&server, "wait_for_steps", json!({"id": id, "seconds": 1}));
        assert!(started.elapsed() >= Duration::from_secs(1));
        assert_eq!((waited["structuredContent"]["done"].clone(), waited["structuredContent"]["timed_out"].clone()), (json!(false), json!(true)));
        // The reader checks the step a moment into a long wait: it returns then.
        let (checker_store, checker_id) = (store.clone(), id.clone());
        let checker = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            user_steps::check(&checker_store, &checker_id, 0, true, 3.0).unwrap();
        });
        let started = Instant::now();
        let done = call(&server, "wait_for_steps", json!({"id": id, "seconds": 120}));
        checker.join().unwrap();
        assert!(started.elapsed() < Duration::from_secs(30), "{:?}", started.elapsed());
        assert_eq!(done["structuredContent"]["done"], true);
        assert!(done["structuredContent"].get("timed_out").is_none());
    }

    #[test]
    fn the_acp_entry_starts_this_program_with_the_flag_on_the_store() {
        let entry = acp_entry(Path::new("C:/Alelyon/centcom.exe"), Path::new("D:/steps"));
        assert_eq!(entry["name"], NAME);
        assert_eq!(entry["command"], "C:/Alelyon/centcom.exe");
        assert_eq!(entry["args"], json!([FLAG]));
        assert_eq!(entry["env"], json!([{"name": "ALELYON_STEPS_DIR", "value": "D:/steps"}]));
    }
}
