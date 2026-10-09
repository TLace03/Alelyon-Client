//! One working session with an agent over ACP: the handshake, the sign-in
//! method, a session in a folder, its model, prompts and cancelling; what the
//! agent streams back ([`Update`]) and what it asks (permission, a file to
//! read or write) go to the caller's [`Client`].
//!
//! Measured against the two adapters on 2026-10-09 (real subscriptions, a
//! scratch probe): Claude Code's session starts on a model this route may
//! not offer, so [`Model`] is set before the first prompt; Codex signs in with
//! `authenticate {methodId: "chatgpt"}` and takes its model as a config option.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures::future::BoxFuture;
use serde_json::{Value, json};
use tokio::runtime::Handle;

use super::connection::{Ask, Asks, CallError, Connection, Event, Events, cut};

/// How long the handshake, a session's start and a model change may take.
pub const START_WAIT: Duration = Duration::from_secs(120);
/// How long one prompt may run: an agent's turn can be long; Stop cancels it.
pub const PROMPT_WAIT: Duration = Duration::from_secs(6 * 60 * 60);
/// The longest text kept from one update.
pub const MAX_TEXT: usize = 64 * 1024;

/// How a session's model is chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Model {
    /// `session/set_model {modelId}` (Codex's adapter offers its models so).
    SetModel(String),
    /// `session/set_config_option {configId: "model", value}` (Claude
    /// Code's adapter offers its models so).
    ConfigOption(String),
    /// The agent's own choice.
    Agents,
}

/// What the agent streams back while a prompt runs.
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    /// A piece of its answer.
    Text(String),
    /// A piece of its reasoning (shown as reasoning, never as the answer).
    Thought(String),
    /// A tool it called: its id, title, kind and status.
    ToolCall {
        id: String,
        title: String,
        kind: String,
        status: String,
        /// The changes it proposes, when it carries them (Codex announces its
        /// edit's diff here, then asks).
        diffs: Vec<Diff>,
    },
    /// A tool call's progress or end, with any text it gave.
    ToolUpdate {
        id: String,
        status: Option<String>,
        text: Option<String>,
    },
    /// Its plan, as (step, status).
    Plan(Vec<(String, String)>),
    /// Anything else, by its kind (shown in the log only).
    Other(String),
}

/// A permission the agent asks for: what for, and its options (id, name, kind).
#[derive(Clone, Debug, PartialEq)]
pub struct Permission {
    pub title: String,
    pub detail: String,
    pub options: Vec<(String, String, String)>,
    /// The tool call it asks about.
    pub call_id: String,
    /// The changes it would make, when it says (Claude Code puts its edit's
    /// diff in the request itself).
    pub diffs: Vec<Diff>,
}

/// A change to one file an agent proposes: the whole old text (`None` for a
/// new file) or the part it replaces, and the new.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diff {
    pub path: String,
    pub old: Option<String>,
    pub new: String,
}

/// The `{type: "diff"}` items of a tool call's content.
pub fn diffs(content: Option<&Value>) -> Vec<Diff> {
    content
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("diff"))
                .map(|item| Diff {
                    path: item.get("path").and_then(Value::as_str).unwrap_or("").to_owned(),
                    old: item.get("oldText").and_then(Value::as_str).map(str::to_owned),
                    new: item.get("newText").and_then(Value::as_str).unwrap_or("").to_owned(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The most lines of a change the approval dialog shows.
pub const DIFF_LINES: usize = 60;

/// A change as the approval dialog shows it: each file, then its removed
/// (`- `) and added (`+ `) lines with a line of context around them, at most
/// [`DIFF_LINES`] in all.
pub fn diff_lines(diffs: &[Diff]) -> Vec<String> {
    let mut out = Vec::new();
    for diff in diffs {
        out.push(match &diff.old {
            None => format!("{} (new file)", diff.path),
            Some(_) => format!("{}:", diff.path),
        });
        let old = diff.old.clone().unwrap_or_default();
        let text = similar::TextDiff::from_lines(old.as_str(), diff.new.as_str());
        for group in text.grouped_ops(1) {
            for op in group {
                for change in text.iter_changes(&op) {
                    let mark = match change.tag() {
                        similar::ChangeTag::Delete => "- ",
                        similar::ChangeTag::Insert => "+ ",
                        similar::ChangeTag::Equal => "  ",
                    };
                    out.push(format!("{mark}{}", change.value().trim_end_matches(['\r', '\n'])));
                }
            }
        }
    }
    if out.len() > DIFF_LINES {
        let more = out.len() - DIFF_LINES;
        out.truncate(DIFF_LINES);
        out.push(format!("... and {more} more lines"));
    }
    out
}

/// What the caller does with what the agent sends and asks.
pub trait Client: Send + Sync + 'static {
    fn update(&self, update: Update);
    /// The reader's choice of one of the options' ids, or `None` to cancel.
    fn permission(&self, permission: Permission) -> BoxFuture<'static, Option<String>>;
    /// A text file's content, from `line` (1-based) for `limit` lines when given.
    fn read(&self, path: &str, line: Option<u64>, limit: Option<u64>) -> Result<String, String>;
    /// Write a text file (the caller may stage it for review instead).
    fn write(&self, path: &str, content: &str) -> Result<(), String>;
}

fn text_of(content: &Value) -> Option<String> {
    match content.get("type").and_then(Value::as_str) {
        Some("text") => content
            .get("text")
            .and_then(Value::as_str)
            .map(|t| cut(t, MAX_TEXT)),
        _ => None,
    }
}

/// One `session/update`'s `update` as an [`Update`].
pub fn parse_update(update: &Value) -> Update {
    let field = |name: &str| {
        update
            .get(name)
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned()
    };
    match update
        .get("sessionUpdate")
        .and_then(Value::as_str)
        .unwrap_or("")
    {
        "agent_message_chunk" => {
            Update::Text(update.get("content").and_then(text_of).unwrap_or_default())
        }
        "agent_thought_chunk" => {
            Update::Thought(update.get("content").and_then(text_of).unwrap_or_default())
        }
        "tool_call" => Update::ToolCall {
            id: field("toolCallId"),
            title: cut(&field("title"), 400),
            kind: field("kind"),
            status: field("status"),
            diffs: diffs(update.get("content")),
        },
        "tool_call_update" => {
            let text = update
                .get("content")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.get("content").and_then(text_of))
                        .collect::<Vec<_>>()
                        .join("\n")
                });
            Update::ToolUpdate {
                id: field("toolCallId"),
                status: update
                    .get("status")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                text: text.filter(|t| !t.is_empty()),
            }
        }
        "plan" => Update::Plan(
            update
                .get("entries")
                .and_then(Value::as_array)
                .map(|entries| {
                    entries
                        .iter()
                        .map(|e| {
                            (
                                cut(e.get("content").and_then(Value::as_str).unwrap_or(""), 400),
                                e.get("status")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_owned(),
                            )
                        })
                        .collect()
                })
                .unwrap_or_default(),
        ),
        other => Update::Other(other.to_owned()),
    }
}

/// A `session/request_permission`'s parameters as a [`Permission`].
pub fn parse_permission(params: &Value) -> Permission {
    let call = params.get("toolCall").cloned().unwrap_or(Value::Null);
    let title = cut(
        call.get("title")
            .and_then(Value::as_str)
            .unwrap_or("The agent asks to act."),
        400,
    );
    let detail = call
        .get("rawInput")
        .map(|raw| cut(&raw.to_string(), 2000))
        .unwrap_or_default();
    let options = params
        .get("options")
        .and_then(Value::as_array)
        .map(|options| {
            options
                .iter()
                .map(|o| {
                    let get = |k: &str| o.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
                    (get("optionId"), cut(&get("name"), 120), get("kind"))
                })
                .collect()
        })
        .unwrap_or_default();
    Permission {
        title,
        detail,
        options,
        call_id: call
            .get("toolCallId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
        diffs: diffs(call.get("content")),
    }
}

/// The agent's requests, answered through `client`.
pub fn asks(client: Arc<dyn Client>, handle: Handle) -> Asks {
    Arc::new(move |ask: Ask| {
        let client = client.clone();
        match ask.method.as_str() {
            "session/request_permission" => {
                let permission = parse_permission(&ask.params);
                handle.spawn(async move {
                    match client.permission(permission).await {
                        Some(option) => ask.answer(
                            json!({"outcome": {"outcome": "selected", "optionId": option}}),
                        ),
                        None => ask.answer(json!({"outcome": {"outcome": "cancelled"}})),
                    }
                });
            }
            "fs/read_text_file" => {
                let path = ask
                    .params
                    .get("path")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let line = ask.params.get("line").and_then(Value::as_u64);
                let limit = ask.params.get("limit").and_then(Value::as_u64);
                handle.spawn_blocking(move || match client.read(&path, line, limit) {
                    Ok(content) => ask.answer(json!({"content": content})),
                    Err(why) => ask.refuse(&why),
                });
            }
            "fs/write_text_file" => {
                let path = ask
                    .params
                    .get("path")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let content = ask
                    .params
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                handle.spawn_blocking(move || match client.write(&path, &content) {
                    Ok(()) => ask.answer(json!({})),
                    Err(why) => ask.refuse(&why),
                });
            }
            _ => ask.refuse("Lattice does not offer that."),
        }
    })
}

/// The agent's notifications, its updates going to `client`; `log` takes the rest.
pub fn events(client: Arc<dyn Client>, log: Arc<dyn Fn(String) + Send + Sync>) -> Events {
    Arc::new(move |event| match event {
        Event::Notification { method, params } if method == "session/update" => {
            if let Some(update) = params.get("update") {
                client.update(parse_update(update));
            }
        }
        Event::Notification { method, .. } => {
            log(format!("(a notification Lattice does not use) {method}"))
        }
        Event::Log(line) => log(line),
        Event::Closed(why) => log(why),
    })
}

/// One session with an agent.
pub struct AgentSession {
    conn: Arc<Connection>,
    pub id: String,
    /// The models the agent offered when the session opened.
    pub offered: super::models::Offered,
}

fn failed(step: &str, error: CallError) -> String {
    format!("{step}: {}", error.sentence())
}

impl AgentSession {
    /// Shake hands, sign in with `auth` when given, open a session in `cwd`,
    /// and choose `model`.
    pub async fn start(
        conn: Arc<Connection>,
        cwd: &Path,
        auth: Option<&str>,
        model: &Model,
        mode: Option<&str>,
    ) -> Result<Self, String> {
        let init = json!({
            "protocolVersion": 1,
            "clientCapabilities": {"fs": {"readTextFile": true, "writeTextFile": false}, "terminal": false},
        });
        conn.request("initialize", init, START_WAIT)
            .await
            .map_err(|e| failed("The agent did not start", e))?;
        if let Some(method) = auth {
            conn.request("authenticate", json!({"methodId": method}), START_WAIT)
                .await
                .map_err(|e| failed("The agent could not sign in", e))?;
        }
        let new = conn
            .request(
                "session/new",
                json!({"cwd": cwd.to_string_lossy(), "mcpServers": []}),
                START_WAIT,
            )
            .await
            .map_err(|e| failed("The agent could not open a session", e))?;
        let id = new
            .get("sessionId")
            .and_then(Value::as_str)
            .ok_or_else(|| "The agent opened no session.".to_owned())?
            .to_owned();
        let offered = super::models::offered(&new);
        let chosen = match model {
            Model::SetModel(m) => {
                Some(("session/set_model", json!({"sessionId": id, "modelId": m})))
            }
            Model::ConfigOption(m) => Some((
                "session/set_config_option",
                json!({"sessionId": id, "configId": "model", "value": m}),
            )),
            Model::Agents => None,
        };
        if let Some((method, params)) = chosen {
            conn.request(method, params, START_WAIT)
                .await
                .map_err(|e| failed("The agent's model could not be chosen", e))?;
        }
        if let Some(mode) = mode {
            conn.request(
                "session/set_config_option",
                json!({"sessionId": id, "configId": "mode", "value": mode}),
                START_WAIT,
            )
            .await
            .map_err(|e| failed("The agent's mode could not be set", e))?;
        }
        Ok(Self { conn, id, offered })
    }

    /// Send one prompt; its updates stream to the client. Its stop reason.
    pub async fn prompt(&self, text: &str) -> Result<String, String> {
        let params = json!({"sessionId": self.id, "prompt": [{"type": "text", "text": text}]});
        let answer = self
            .conn
            .request("session/prompt", params, PROMPT_WAIT)
            .await
            .map_err(|e| failed("The agent's turn failed", e))?;
        Ok(answer
            .get("stopReason")
            .and_then(Value::as_str)
            .unwrap_or("end_turn")
            .to_owned())
    }

    /// Ask the agent to stop its prompt (it answers that prompt `cancelled`).
    pub fn cancel(&self) {
        self.conn
            .notify("session/cancel", json!({"sessionId": self.id}));
    }

    /// End the session's connection (the agent's stdin closes).
    pub fn close(&self) {
        self.conn.close();
    }
}
