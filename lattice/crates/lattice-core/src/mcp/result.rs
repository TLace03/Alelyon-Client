//! What an MCP call gives the model (the chat core's spec §12, "Calls";
//! T3). Not a port: the SDK hands the model each content part's JSON; Lattice
//! hands it the text, which is what a model reads, and names what it cannot
//! pass on.
//!
//! - A `text` part is its text; an embedded `resource` with text is its URI
//!   and text; a `resource_link` is its name and URI.
//! - An `image` or `audio` part, or a resource's `blob`, is named with its
//!   type and size and not sent: no model client here takes content parts
//!   yet ("images go only to vision models").
//! - With no content at all, `structuredContent` is given as JSON.
//! - At most [`MAX_RESULT`] bytes; a longer text is cut, and says so.
//! - `isError: true` makes the call a tool error, its text the message.
//!
//! The text is untrusted data: the turn's sink redacts what it records, and
//! a remote target receives it only behind the secret tripwire (T3).

use serde_json::Value;

/// The most bytes of a result the model is given (§12).
pub const MAX_RESULT: usize = 32 * 1024;

/// A call's result as the model is given it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallText {
    pub text: String,
    /// The server said the call failed (`isError`).
    pub is_error: bool,
    /// The text was longer than [`MAX_RESULT`].
    pub cut: bool,
}

fn kib(base64: &str) -> String {
    let bytes = base64.len() / 4 * 3;
    if bytes < 1024 {
        format!("{bytes} bytes")
    } else {
        format!("{} KiB", bytes.div_ceil(1024))
    }
}

fn field<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn part(item: &Value) -> String {
    match field(item, "type") {
        "text" => field(item, "text").to_owned(),
        "image" | "audio" => {
            let kind = if field(item, "type") == "image" {
                "An image"
            } else {
                "Audio"
            };
            format!(
                "[{kind} ({}, {}) came back; it is not shown to this model.]",
                field(item, "mimeType"),
                kib(field(item, "data"))
            )
        }
        "resource_link" => {
            let name = field(item, "name");
            let uri = field(item, "uri");
            if name.is_empty() {
                format!("[Resource: {uri}]")
            } else {
                format!("[Resource: {name} <{uri}>]")
            }
        }
        "resource" => {
            let resource = item.get("resource").unwrap_or(&Value::Null);
            let uri = field(resource, "uri");
            if let Some(text) = resource.get("text").and_then(Value::as_str) {
                format!("Resource {uri}:\n{text}")
            } else {
                format!(
                    "[Resource {uri} ({}, {}) is binary; it is not shown to this model.]",
                    field(resource, "mimeType"),
                    kib(field(resource, "blob"))
                )
            }
        }
        "" => "[A part with no type came back; it is not shown.]".to_owned(),
        other => {
            let shown: String = other.chars().take(40).collect();
            format!("[A part of type {shown} came back; it is not shown.]")
        }
    }
}

/// The result object of `tools/call`, as the model is given it.
pub fn call_text(result: &Value) -> CallText {
    let is_error = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let parts: Vec<String> = result
        .get("content")
        .and_then(Value::as_array)
        .map(|items| items.iter().map(part).collect())
        .unwrap_or_default();
    let mut text = if parts.is_empty() {
        match result.get("structuredContent") {
            Some(structured) if !structured.is_null() => structured.to_string(),
            _ => String::new(),
        }
    } else {
        parts.join("\n")
    };
    let cut = text.len() > MAX_RESULT;
    if cut {
        let more = text.len() - MAX_RESULT;
        let mut end = MAX_RESULT;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        text.truncate(end);
        text.push_str(&format!(
            "\n[The result was cut here: {more} more bytes are not shown.]"
        ));
    }
    CallText {
        text,
        is_error,
        cut,
    }
}
