//! JSON-RPC 2.0 as MCP's stdio transport frames it: one message per line,
//! UTF-8, no newline inside a message (the chat core's spec §12). Not a
//! port. Pure: building lines and reading one line.

use serde_json::{Value, json};

/// The longest line read from a server; a longer one ends the connection.
pub const MAX_LINE: usize = 8 * 1024 * 1024;
/// JSON-RPC's "method not found".
pub const METHOD_NOT_FOUND: i64 = -32601;
/// JSON-RPC's "invalid request".
pub const INVALID_REQUEST: i64 = -32600;

/// A JSON-RPC error a server answered with.
#[derive(Clone, Debug, PartialEq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
}

/// One message from a server.
#[derive(Clone, Debug, PartialEq)]
pub enum Incoming {
    /// The answer to one of Lattice's requests.
    Response {
        id: Value,
        outcome: Result<Value, RpcError>,
    },
    /// A request the server makes of Lattice.
    Request {
        id: Value,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
    /// Not a JSON-RPC message (a server's stray output), and why.
    Invalid(String),
}

fn line(value: &Value) -> String {
    // `to_string` is compact: a newline in a string is escaped, so the
    // message is one line.
    let mut text = serde_json::to_string(value).unwrap_or_default();
    text.push('\n');
    text
}

/// A request, as a line.
pub fn request(id: u64, method: &str, params: Value) -> String {
    line(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))
}

/// A notification, as a line.
pub fn notification(method: &str, params: Option<Value>) -> String {
    match params {
        Some(params) => line(&json!({"jsonrpc": "2.0", "method": method, "params": params})),
        None => line(&json!({"jsonrpc": "2.0", "method": method})),
    }
}

/// A result for the server's request `id`, as a line.
pub fn result(id: &Value, result: Value) -> String {
    line(&json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// An error for the server's request `id`, as a line.
pub fn error(id: &Value, code: i64, message: &str) -> String {
    line(&json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}))
}

fn one(value: Value) -> Incoming {
    let Value::Object(mut object) = value else {
        return Incoming::Invalid("not a JSON object".to_owned());
    };
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Incoming::Invalid("no \"jsonrpc\": \"2.0\"".to_owned());
    }
    let method = object
        .get("method")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let id = object.remove("id").filter(|id| !id.is_null());
    match (method, id) {
        (Some(method), Some(id)) => Incoming::Request {
            id,
            method,
            params: object.remove("params").unwrap_or(Value::Null),
        },
        (Some(method), None) => Incoming::Notification {
            method,
            params: object.remove("params").unwrap_or(Value::Null),
        },
        (None, Some(id)) => {
            if let Some(error) = object.remove("error") {
                let code = error.get("code").and_then(Value::as_i64).unwrap_or(0);
                let message = error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                Incoming::Response {
                    id,
                    outcome: Err(RpcError { code, message }),
                }
            } else if let Some(result) = object.remove("result") {
                Incoming::Response {
                    id,
                    outcome: Ok(result),
                }
            } else {
                Incoming::Invalid("a response with neither a result nor an error".to_owned())
            }
        }
        (None, None) => Incoming::Invalid("neither a method nor an id".to_owned()),
    }
}

/// The messages in one line (a batch, an array, gives each of its own).
pub fn parse(line: &str) -> Vec<Incoming> {
    match serde_json::from_str::<Value>(line.trim()) {
        Ok(Value::Array(items)) if !items.is_empty() => items.into_iter().map(one).collect(),
        Ok(value) => vec![one(value)],
        Err(_) => vec![Incoming::Invalid("not JSON".to_owned())],
    }
}
