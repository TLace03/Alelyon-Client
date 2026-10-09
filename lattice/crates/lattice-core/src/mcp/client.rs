//! One MCP server's connection (the chat core's spec §12, "Lifecycle",
//! "Calls"; CR2). Not a port; the protocol is MCP's, and the client is
//! Lattice's own (H1's decision: no SDK crate, see the native README).
//!
//! [`Connection::start`] takes the server's stdout and stdin and runs two
//! threads, both blocked on their pipe or channel when nothing happens, so an
//! idle server costs Lattice no wake-ups (CR2):
//! - the **reader** reads one line at a time (at most [`jsonrpc::MAX_LINE`]):
//!   a response goes to the request waiting for it; a `ping` from the server
//!   is answered, and any other request it makes is refused (Lattice offers
//!   no sampling, roots or elicitation); `notifications/tools/list_changed`
//!   and log messages go to the holder's [`Events`]; output that is not a
//!   message is logged, not obeyed. When the output ends, every waiting
//!   request ends with the reason;
//! - the **writer** writes each line to the server's stdin, so a server that
//!   reads slowly never blocks an async worker. Closing the connection ends
//!   it, which closes the server's stdin: an MCP server ends when its input
//!   does.
//!
//! A request has a timeout (the handshake's is the only timer at start); a
//! request given up (its timeout, or the turn stopped) is withdrawn with
//! `notifications/cancelled`, except `initialize`, which MCP says is never
//! cancelled.
//!
//! [`handshake`] asks for [`REQUESTED_VERSION`] and accepts any of
//! [`HANDSHAKE_VERSIONS`]; a server that answers with another version is
//! closed. [`list_tools`] follows `nextCursor` (at most [`MAX_PAGES`] pages,
//! [`MAX_TOOLS`] tools). Everything a server sends is untrusted: names,
//! descriptions and schemas are bounded here and only shown or offered.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::sync::oneshot;

use super::jsonrpc::{self, Incoming, MAX_LINE, METHOD_NOT_FOUND, RpcError};

/// The MCP version Lattice asks for.
pub const REQUESTED_VERSION: &str = "2025-11-25";
/// The versions with the `initialize` handshake that Lattice speaks.
pub const HANDSHAKE_VERSIONS: [&str; 4] = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"];
/// The most tools one server's list gives.
pub const MAX_TOOLS: usize = 256;
/// The most pages of a tool list read.
pub const MAX_PAGES: usize = 32;
/// The longest description kept, in characters.
pub const MAX_DESCRIPTION: usize = 8 * 1024;
/// The largest input schema kept, as JSON.
pub const MAX_SCHEMA: usize = 64 * 1024;
/// The longest tool name kept.
pub const MAX_TOOL_NAME: usize = 128;
/// The longest server instructions kept, in characters.
pub const MAX_INSTRUCTIONS: usize = 8 * 1024;

/// What a connection tells its owner, from its reader thread.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// `notifications/tools/list_changed`.
    ToolsChanged,
    /// A log line: a `notifications/message`, or output that is not a message.
    Log(String),
    /// The output ended or broke; no message comes after this.
    Closed(String),
}

/// Where a connection's events go.
pub type Events = Arc<dyn Fn(Event) + Send + Sync>;

/// Why a request has no result.
#[derive(Clone, Debug, PartialEq)]
pub enum CallError {
    Rpc(RpcError),
    TimedOut,
    Closed(String),
}

impl CallError {
    /// One sentence for the reader or the model.
    pub fn sentence(&self) -> String {
        match self {
            Self::Rpc(error) if error.message.trim().is_empty() => {
                format!("The server answered with error {}.", error.code)
            }
            Self::Rpc(error) => format!(
                "The server answered with an error: {}",
                cut(error.message.trim(), 500)
            ),
            Self::TimedOut => "The server did not answer in time.".to_owned(),
            Self::Closed(why) => why.clone(),
        }
    }
}

/// `text`, at most `chars` characters, with "…" when cut.
pub fn cut(text: &str, chars: usize) -> String {
    let mut out: String = text.chars().take(chars).collect();
    if text.chars().nth(chars).is_some() {
        out.push('\u{2026}');
    }
    out
}

/// A wait as a sentence says it: `500 ms`, `30 s`.
pub fn duration_text(wait: Duration) -> String {
    if wait < Duration::from_secs(1) {
        format!("{} ms", wait.as_millis())
    } else {
        format!("{} s", wait.as_secs())
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

enum Out {
    Line(String),
    Close,
}

type Answer = oneshot::Sender<Result<Value, RpcError>>;

struct Shared {
    pending: Mutex<HashMap<u64, Answer>>,
    closed: Mutex<Option<String>>,
    out: mpsc::Sender<Out>,
}

impl Shared {
    /// Mark the connection closed (the first reason stays), end every waiting
    /// request, and stop the writer, which closes the server's stdin.
    fn close(&self, why: &str) {
        {
            let mut closed = lock(&self.closed);
            if closed.is_none() {
                *closed = Some(why.to_owned());
            }
        }
        // Dropping the senders ends each waiting request with `Closed`.
        let waiting: Vec<Answer> = lock(&self.pending)
            .drain()
            .map(|(_, sender)| sender)
            .collect();
        drop(waiting);
        let _ = self.out.send(Out::Close);
    }
}

/// One server's connection.
pub struct Connection {
    shared: Arc<Shared>,
    next: AtomicU64,
}

/// What a read gave.
enum Line {
    Text,
    End,
    TooLong,
}

/// Read up to the next `\n` (not kept) into `line`, at most `MAX_LINE` bytes.
fn read_line(reader: &mut impl BufRead, line: &mut Vec<u8>) -> std::io::Result<Line> {
    loop {
        let (done, used) = {
            let available = match reader.fill_buf() {
                Ok(available) => available,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            };
            if available.is_empty() {
                return Ok(if line.is_empty() {
                    Line::End
                } else {
                    Line::Text
                });
            }
            match available.iter().position(|byte| *byte == b'\n') {
                Some(at) => {
                    line.extend_from_slice(&available[..at]);
                    (true, at + 1)
                }
                None => {
                    line.extend_from_slice(available);
                    (false, available.len())
                }
            }
        };
        reader.consume(used);
        if line.len() > MAX_LINE {
            return Ok(Line::TooLong);
        }
        if done {
            return Ok(Line::Text);
        }
    }
}

/// A `notifications/message` as one log line.
fn log_line(params: &Value) -> String {
    let level = params
        .get("level")
        .and_then(Value::as_str)
        .unwrap_or("info");
    let data = match params.get("data") {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    };
    format!("[{}] {}", cut(level, 16), cut(&data, 2000))
}

fn read_loop(reader: Box<dyn Read + Send>, shared: Arc<Shared>, events: Events) {
    let mut reader = BufReader::with_capacity(64 * 1024, reader);
    let mut line = Vec::new();
    let why = loop {
        line.clear();
        match read_line(&mut reader, &mut line) {
            Ok(Line::Text) => {}
            Ok(Line::End) => break "The server stopped.".to_owned(),
            Ok(Line::TooLong) => {
                break "The server sent a message larger than 8 MiB, so Lattice stopped reading it."
                    .to_owned();
            }
            Err(_) => break "The server's output could not be read.".to_owned(),
        }
        let text = String::from_utf8_lossy(&line);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        for incoming in jsonrpc::parse(text) {
            match incoming {
                Incoming::Response { id, outcome } => {
                    let sender = id.as_u64().and_then(|id| lock(&shared.pending).remove(&id));
                    if let Some(sender) = sender {
                        let _ = sender.send(outcome);
                    }
                }
                Incoming::Request { id, method, .. } => {
                    let reply = if method == "ping" {
                        jsonrpc::result(&id, json!({}))
                    } else {
                        jsonrpc::error(&id, METHOD_NOT_FOUND, "Lattice does not offer that.")
                    };
                    let _ = shared.out.send(Out::Line(reply));
                }
                Incoming::Notification { method, params } => match method.as_str() {
                    "notifications/tools/list_changed" => events(Event::ToolsChanged),
                    "notifications/message" => events(Event::Log(log_line(&params))),
                    _ => {}
                },
                Incoming::Invalid(_) => {
                    events(Event::Log(format!(
                        "(not an MCP message) {}",
                        cut(text, 300)
                    )));
                }
            }
        }
    };
    shared.close(&why);
    events(Event::Closed(why));
}

fn write_loop(mut writer: Box<dyn Write + Send>, lines: mpsc::Receiver<Out>) {
    for out in lines {
        match out {
            Out::Line(line) => {
                if writer
                    .write_all(line.as_bytes())
                    .and_then(|()| writer.flush())
                    .is_err()
                {
                    break;
                }
            }
            Out::Close => break,
        }
    }
    // `writer` drops here: the server's stdin ends.
}

/// A request waiting for its answer. Given up (dropped before its answer),
/// it leaves the table and is withdrawn with `notifications/cancelled`.
struct Waiting<'a> {
    shared: &'a Shared,
    id: u64,
    cancel: bool,
    done: bool,
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let was_waiting = lock(&self.shared.pending).remove(&self.id).is_some();
        if self.cancel && was_waiting && lock(&self.shared.closed).is_none() {
            let _ = self.shared.out.send(Out::Line(jsonrpc::notification(
                "notifications/cancelled",
                Some(json!({"requestId": self.id, "reason": "Lattice stopped waiting for it."})),
            )));
        }
    }
}

impl Connection {
    /// Start the reader and the writer over a server's stdout (`reader`) and
    /// stdin (`writer`); `name` names their threads.
    pub fn start(
        name: &str,
        reader: Box<dyn Read + Send>,
        writer: Box<dyn Write + Send>,
        events: Events,
    ) -> std::io::Result<Arc<Self>> {
        let (out, lines) = mpsc::channel();
        let shared = Arc::new(Shared {
            pending: Mutex::default(),
            closed: Mutex::default(),
            out,
        });
        let short: String = name.chars().take(24).collect();
        std::thread::Builder::new()
            .name(format!("lattice-mcp-in {short}"))
            .spawn(move || write_loop(writer, lines))?;
        let reading = shared.clone();
        std::thread::Builder::new()
            .name(format!("lattice-mcp-out {short}"))
            .spawn(move || read_loop(reader, reading, events))?;
        Ok(Arc::new(Self {
            shared,
            next: AtomicU64::new(1),
        }))
    }

    /// Why the connection is closed, once it is.
    pub fn closed(&self) -> Option<String> {
        lock(&self.shared.closed).clone()
    }

    /// Close it: waiting requests end, and the server's stdin ends.
    pub fn close(&self) {
        self.shared.close("Lattice closed the connection.");
    }

    /// Send a notification (nothing comes back).
    pub fn notify(&self, method: &str, params: Option<Value>) {
        let _ = self
            .shared
            .out
            .send(Out::Line(jsonrpc::notification(method, params)));
    }

    /// Send a request and wait at most `timeout` for its answer.
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, CallError> {
        let gone = || {
            self.closed()
                .unwrap_or_else(|| "The server stopped.".to_owned())
        };
        if self.closed().is_some() {
            return Err(CallError::Closed(gone()));
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (sender, answer) = oneshot::channel();
        lock(&self.shared.pending).insert(id, sender);
        let mut waiting = Waiting {
            shared: &self.shared,
            id,
            cancel: method != "initialize",
            done: false,
        };
        if self
            .shared
            .out
            .send(Out::Line(jsonrpc::request(id, method, params)))
            .is_err()
        {
            return Err(CallError::Closed(gone()));
        }
        match tokio::time::timeout(timeout, answer).await {
            Ok(Ok(outcome)) => {
                waiting.done = true;
                outcome.map_err(CallError::Rpc)
            }
            Ok(Err(_)) => {
                waiting.done = true;
                Err(CallError::Closed(gone()))
            }
            // `waiting` withdraws the request as it drops.
            Err(_) => Err(CallError::TimedOut),
        }
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        let _ = self.shared.out.send(Out::Close);
    }
}

// --------------------------------------------------------------- the calls

/// What a server said about itself in its `initialize` answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerInfo {
    /// The MCP version it agreed to.
    pub protocol: String,
    /// `serverInfo.name` and `version`, bounded.
    pub name: String,
    pub version: String,
    /// Its `instructions` (untrusted text), bounded; shown, never sent.
    pub instructions: Option<String>,
}

fn text_of(value: Option<&Value>, chars: usize) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(|text| cut(text.trim(), chars))
        .filter(|text| !text.is_empty())
}

/// The handshake: `initialize`, then `notifications/initialized`.
pub async fn handshake(conn: &Connection, timeout: Duration) -> Result<ServerInfo, String> {
    let params = json!({
        "protocolVersion": REQUESTED_VERSION,
        "capabilities": {},
        "clientInfo": {
            "name": "lattice",
            "title": "Alelyon Lattice",
            "version": env!("CARGO_PKG_VERSION"),
        },
    });
    let answer =
        conn.request("initialize", params, timeout)
            .await
            .map_err(|error| match error {
                CallError::TimedOut => format!(
                    "The server did not finish starting within {}.",
                    duration_text(timeout)
                ),
                other => other.sentence(),
            })?;
    let Some(protocol) = answer.get("protocolVersion").and_then(Value::as_str) else {
        conn.close();
        return Err("The server did not say which MCP version it speaks.".to_owned());
    };
    if !HANDSHAKE_VERSIONS.contains(&protocol) {
        conn.close();
        return Err(format!(
            "The server speaks MCP {}, which Lattice does not speak.",
            cut(protocol, 40)
        ));
    }
    let info = answer.get("serverInfo");
    let server = ServerInfo {
        protocol: protocol.to_owned(),
        name: text_of(info.and_then(|info| info.get("name")), 80).unwrap_or_default(),
        version: text_of(info.and_then(|info| info.get("version")), 40).unwrap_or_default(),
        instructions: text_of(answer.get("instructions"), MAX_INSTRUCTIONS),
    };
    conn.notify("notifications/initialized", None);
    Ok(server)
}

/// One tool a server lists, bounded.
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub name: String,
    pub title: Option<String>,
    pub description: String,
    /// A JSON Schema object for its arguments.
    pub input_schema: Value,
    /// The server's hints (untrusted): it says it only reads, may destroy,
    /// or reaches outside this machine.
    pub read_only: Option<bool>,
    pub destructive: Option<bool>,
    pub open_world: Option<bool>,
}

/// A listed tool, or why it is left out.
fn tool_of(value: &Value) -> Result<Tool, String> {
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| {
            !name.is_empty() && name.len() <= MAX_TOOL_NAME && !name.chars().any(char::is_control)
        })
        .ok_or_else(|| "A tool without a usable name was left out.".to_owned())?;
    let mut schema = value
        .get("inputSchema")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object"}));
    let Some(object) = schema.as_object_mut() else {
        return Err(format!(
            "{name}: its input schema is not an object, so it was left out."
        ));
    };
    match object.get("type").and_then(Value::as_str) {
        None => {
            object.insert("type".to_owned(), json!("object"));
        }
        Some("object") => {}
        Some(_) => {
            return Err(format!(
                "{name}: its arguments are not an object, so it was left out."
            ));
        }
    }
    object.entry("properties").or_insert_with(|| json!({}));
    if serde_json::to_string(&schema).map_or(usize::MAX, |text| text.len()) > MAX_SCHEMA {
        return Err(format!(
            "{name}: its input schema is larger than 64 KiB, so it was left out."
        ));
    }
    let annotations = value.get("annotations");
    let hint = |key: &str| {
        annotations
            .and_then(|annotations| annotations.get(key))
            .and_then(Value::as_bool)
    };
    Ok(Tool {
        name: name.to_owned(),
        title: text_of(value.get("title"), 120)
            .or_else(|| text_of(annotations.and_then(|a| a.get("title")), 120)),
        description: text_of(value.get("description"), MAX_DESCRIPTION).unwrap_or_default(),
        input_schema: schema,
        read_only: hint("readOnlyHint"),
        destructive: hint("destructiveHint"),
        open_world: hint("openWorldHint"),
    })
}

/// `tools/list`, every page: the tools, and what was left out.
pub async fn list_tools(
    conn: &Connection,
    timeout: Duration,
) -> Result<(Vec<Tool>, Vec<String>), String> {
    let mut tools: Vec<Tool> = Vec::new();
    let mut problems = Vec::new();
    let mut cursor: Option<String> = None;
    for page in 0..MAX_PAGES {
        let params = match &cursor {
            Some(cursor) => json!({"cursor": cursor}),
            None => json!({}),
        };
        let answer = match conn.request("tools/list", params, timeout).await {
            Ok(answer) => answer,
            // A server without tools may not know the method.
            Err(CallError::Rpc(error)) if error.code == METHOD_NOT_FOUND && page == 0 => {
                return Ok((tools, problems));
            }
            Err(error) => return Err(error.sentence()),
        };
        for value in answer
            .get("tools")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            if tools.len() >= MAX_TOOLS {
                problems.push(format!("Only the first {MAX_TOOLS} tools are used."));
                return Ok((tools, problems));
            }
            match tool_of(value) {
                Ok(tool) if tools.iter().any(|kept| kept.name == tool.name) => {
                    problems.push(format!("{}: listed twice; the first is used.", tool.name));
                }
                Ok(tool) => tools.push(tool),
                Err(problem) => problems.push(problem),
            }
        }
        cursor = answer
            .get("nextCursor")
            .and_then(Value::as_str)
            .filter(|next| !next.is_empty())
            .map(str::to_owned);
        if cursor.is_none() {
            return Ok((tools, problems));
        }
    }
    problems.push(format!(
        "Only the first {MAX_PAGES} pages of tools were read."
    ));
    Ok((tools, problems))
}

/// `tools/call`: the server's result object, or why there is none.
pub async fn call_tool(
    conn: &Connection,
    name: &str,
    arguments: Value,
    timeout: Duration,
) -> Result<Value, CallError> {
    let arguments = if arguments.is_object() {
        arguments
    } else {
        json!({})
    };
    conn.request(
        "tools/call",
        json!({"name": name, "arguments": arguments}),
        timeout,
    )
    .await
}
