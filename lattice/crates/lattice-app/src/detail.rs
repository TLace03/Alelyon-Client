//! What the detail column says about one span, worded as plain text.
//!
//! Invariants:
//! - Everything here is plain text: there is no HTML in a native interface, so
//!   nothing a model or a tool wrote can be interpreted as markup. (The run's
//!   final output is the one place rendered as Markdown, and it is drawn by
//!   iced's markdown widget, which has no script or raw-HTML path.)
//! - The model is built when a span is selected or changes, not in `view()`.
//! - Long text is bounded for display ([`MAX_TEXT`]); the Copy button copies the
//!   text as recorded, up to that same bound, and says how much was left out.
//! - Missing data reads as "not recorded", never as an empty success: a guardrail
//!   with no `triggered` value is not reported as passed.
//! - Token usage reads "1,200 → 34 tokens", with `?` for a half the model server
//!   did not report.

use lattice_protocol::SpanRecord;
use serde_json::Value;

use crate::clock::{clock_time, format_count, format_duration, format_usage};
use crate::runstate::SpanEntry;
use crate::spans::{SpanKind, kind_of, title_of};

/// The most characters of one text shown or copied.
pub const MAX_TEXT: usize = 48 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Normal,
    Danger,
    Positive,
}

/// A labelled value in the overview.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub label: String,
    pub value: String,
    pub tone: Tone,
    pub mono: bool,
}

/// A block of text with a caption, for the Input and Output sections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub label: String,
    pub text: String,
    pub mono: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErrorInfo {
    pub message: String,
    /// The error's `data`, pretty-printed.
    pub data: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DetailModel {
    pub span_id: String,
    pub kind: SpanKind,
    pub title: String,
    /// `05:21:05.123 UTC`.
    pub started: String,
    /// `1.24 s`, or `1.24 s so far` while the span is open.
    pub duration: String,
    pub overview: Vec<Field>,
    pub input: Vec<Block>,
    pub output: Vec<Block>,
    pub error: Option<ErrorInfo>,
}

/// `text`, cut at [`MAX_TEXT`] characters with a note saying how much is missing.
pub fn bounded(text: &str) -> String {
    let count = text.chars().count();
    if count <= MAX_TEXT {
        return text.to_string();
    }
    let cut: String = text.chars().take(MAX_TEXT).collect();
    format!(
        "{cut}\n… ({} more characters not shown)",
        format_count((count - MAX_TEXT) as u64)
    )
}

/// JSON as indented text.
pub fn pretty(value: &Value) -> String {
    bounded(&serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()))
}

/// Text that may be JSON: pretty-printed when it parses as an object or array,
/// else as it is.
fn maybe_json(text: &str) -> (String, bool) {
    match serde_json::from_str::<Value>(text.trim()) {
        Ok(value @ (Value::Object(_) | Value::Array(_))) => (pretty(&value), true),
        _ => (bounded(text), false),
    }
}

fn field(label: &str, value: impl Into<String>) -> Field {
    Field {
        label: label.to_string(),
        value: value.into(),
        tone: Tone::Normal,
        mono: false,
    }
}

fn mono_field(label: &str, value: impl Into<String>) -> Field {
    Field {
        mono: true,
        ..field(label, value)
    }
}

fn toned(label: &str, value: impl Into<String>, tone: Tone) -> Field {
    Field {
        tone,
        ..field(label, value)
    }
}

fn names(value: Option<&Value>) -> Option<String> {
    let list = value?.as_array()?;
    if list.is_empty() {
        return Some("None".to_string());
    }
    Some(
        list.iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(", "),
    )
}

fn tokens(usage: Option<&Value>) -> (Option<u64>, Option<u64>) {
    let get = |keys: &[&str]| keys.iter().find_map(|k| usage?.get(*k)?.as_u64());
    (
        get(&["input_tokens", "prompt_tokens"]),
        get(&["output_tokens", "completion_tokens"]),
    )
}

/// The readable text of a chat message's `content`: a string, or the text parts
/// of a list of content parts.
fn content_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).or_else(|| p.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn tool_call_lines(message: &Value) -> Vec<String> {
    message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    let function = call.get("function").unwrap_or(call);
                    let name = function.get("name").and_then(Value::as_str).unwrap_or("?");
                    let args = function
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    format!("→ {name}({args})")
                })
                .collect()
        })
        .unwrap_or_default()
}

fn role_label(message: &Value) -> String {
    let role = message
        .get("role")
        .and_then(Value::as_str)
        .or_else(|| message.get("type").and_then(Value::as_str))
        .unwrap_or("item");
    match role {
        "system" => "System".to_string(),
        "user" => "User".to_string(),
        "assistant" => "Assistant".to_string(),
        "tool" => match message.get("tool_call_id").and_then(Value::as_str) {
            Some(id) => format!("Tool result ({id})"),
            None => "Tool result".to_string(),
        },
        other => {
            let mut chars = other.chars();
            chars.next().map_or_else(String::new, |c| {
                c.to_uppercase().collect::<String>() + chars.as_str()
            })
        }
    }
}

/// The messages of a generation span's `input` or `output`, one block each.
fn message_blocks(value: Option<&Value>) -> Vec<Block> {
    match value {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                let mut parts = Vec::new();
                let body = content_text(item);
                if !body.is_empty() {
                    parts.push(body);
                }
                parts.extend(tool_call_lines(item));
                if parts.is_empty() {
                    Block {
                        label: role_label(item),
                        text: pretty(item),
                        mono: true,
                    }
                } else {
                    Block {
                        label: role_label(item),
                        text: bounded(&parts.join("\n")),
                        mono: false,
                    }
                }
            })
            .collect(),
        Some(Value::String(text)) => vec![Block {
            label: "Text".to_string(),
            text: bounded(text),
            mono: false,
        }],
        Some(other) => vec![Block {
            label: "Data".to_string(),
            text: pretty(other),
            mono: true,
        }],
    }
}

fn usage_summary(usage: &Value) -> String {
    let (input, output) = tokens(Some(usage));
    let mut line = format_usage(input, output);
    if let Some(requests) = usage.get("requests").and_then(Value::as_u64) {
        line.push_str(&format!(
            " · {} request{}",
            format_count(requests),
            if requests == 1 { "" } else { "s" }
        ));
    }
    line
}

/// The detail of `entry`. `end` is the span's display end (`None` while it is
/// open in a working run) and `now` the time to measure an open span against.
pub fn build(entry: &SpanEntry, end: Option<f64>, now: f64) -> DetailModel {
    let span: &SpanRecord = &entry.rec;
    let kind = kind_of(span);
    let data = &span.span_data;
    let mut overview = Vec::new();
    let mut input = Vec::new();
    let mut output = Vec::new();

    match kind {
        SpanKind::Model if span.data_type() == "generation" => {
            overview.push(field(
                "Model",
                span.data_str("model").unwrap_or("not recorded"),
            ));
            match data.get("usage").filter(|u| !u.is_null()) {
                Some(usage) => overview.push(field("Usage", usage_summary(usage))),
                None => overview.push(field("Usage", format_usage(None, None))),
            }
            if let Some(config) = data.get("model_config").and_then(Value::as_object) {
                let set: serde_json::Map<String, Value> = config
                    .iter()
                    .filter(|(_, v)| !v.is_null())
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                if !set.is_empty() {
                    overview.push(mono_field(
                        "Settings",
                        serde_json::to_string(&Value::Object(set)).unwrap_or_default(),
                    ));
                }
            }
            input = message_blocks(data.get("input"));
            output = message_blocks(data.get("output"));
        }
        SpanKind::Model => {
            overview.push(field(
                "Response",
                span.data_str("response_id").unwrap_or("not recorded"),
            ));
            overview.push(mono_field("Data", pretty(data)));
        }
        SpanKind::Tool => {
            overview.push(field(
                "Tool",
                span.data_str("name").unwrap_or("not recorded"),
            ));
            if let Some(mcp) = data.get("mcp_data").filter(|m| !m.is_null()) {
                overview.push(mono_field("MCP", pretty(mcp)));
            }
            if let Some(text) = data.get("input").and_then(Value::as_str) {
                let (text, json) = maybe_json(text);
                input.push(Block {
                    label: "Arguments".to_string(),
                    text,
                    mono: json,
                });
            }
            if let Some(text) = data.get("output").and_then(Value::as_str) {
                let (text, json) = maybe_json(text);
                output.push(Block {
                    label: "Result".to_string(),
                    text,
                    mono: json,
                });
            }
        }
        SpanKind::Handoff => {
            let from = span.data_str("from_agent").unwrap_or("?");
            let to = span.data_str("to_agent").unwrap_or("?");
            overview.push(field("Handoff", format!("{from} → {to}")));
        }
        SpanKind::Guardrail => {
            overview.push(field(
                "Guardrail",
                span.data_str("name").unwrap_or("not recorded"),
            ));
            overview.push(match data.get("triggered").and_then(Value::as_bool) {
                Some(true) => toned("Result", "Triggered: the run was refused", Tone::Danger),
                Some(false) => toned("Result", "Passed", Tone::Positive),
                None if end.is_none() => field("Result", "Checking"),
                None => field("Result", "not recorded"),
            });
        }
        SpanKind::Agent => {
            overview.push(field(
                "Agent",
                span.data_str("name").unwrap_or("not recorded"),
            ));
            overview.push(field(
                "Tools",
                names(data.get("tools")).unwrap_or_else(|| "not recorded yet".to_string()),
            ));
            overview.push(field(
                "Handoffs",
                names(data.get("handoffs")).unwrap_or_else(|| "not recorded yet".to_string()),
            ));
            overview.push(field(
                "Output type",
                data.get("output_type")
                    .and_then(Value::as_str)
                    .unwrap_or("Text"),
            ));
        }
        SpanKind::Task => {
            let inner = data.get("data");
            overview.push(field(
                "Workflow",
                inner
                    .and_then(|d| d.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or("not recorded"),
            ));
            if let Some(usage) = inner.and_then(|d| d.get("usage")) {
                overview.push(field("Usage", usage_summary(usage)));
            }
        }
        SpanKind::Turn => {
            let inner = data.get("data");
            if let Some(n) = inner.and_then(|d| d.get("turn")).and_then(Value::as_u64) {
                overview.push(field("Turn", n.to_string()));
            }
            overview.push(field(
                "Agent",
                inner
                    .and_then(|d| d.get("agent_name"))
                    .and_then(Value::as_str)
                    .unwrap_or("not recorded"),
            ));
            if let Some(usage) = inner.and_then(|d| d.get("usage")) {
                overview.push(field("Usage", usage_summary(usage)));
            }
        }
        SpanKind::Mcp => {
            overview.push(field(
                "Server",
                span.data_str("server").unwrap_or("not recorded"),
            ));
            overview.push(field(
                "Tools",
                names(data.get("result")).unwrap_or_else(|| "not recorded".to_string()),
            ));
        }
        SpanKind::Custom | SpanKind::Voice | SpanKind::Unknown => {
            overview.push(mono_field("Data", pretty(data)));
        }
    }
    overview.push(mono_field("Span", span.id.clone()));

    let error = span.error.as_ref().map(|e| ErrorInfo {
        message: bounded(&e.message),
        data: e.data.as_ref().filter(|d| !d.is_null()).map(pretty),
    });

    let elapsed = end.unwrap_or(now.max(entry.start)) - entry.start;
    DetailModel {
        span_id: span.id.clone(),
        kind,
        title: title_of(span),
        started: clock_time(entry.start),
        duration: if end.is_some() {
            format_duration(elapsed)
        } else {
            format!("{} so far", format_duration(elapsed))
        },
        overview,
        input,
        output,
        error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spans::test_span;
    use lattice_protocol::SpanError;
    use serde_json::json;

    fn entry(data: Value) -> SpanEntry {
        SpanEntry::new(test_span("span_1", None, 1, data), 0.0)
    }

    fn overview<'a>(model: &'a DetailModel, label: &str) -> Option<&'a Field> {
        model.overview.iter().find(|f| f.label == label)
    }

    fn built(data: Value) -> DetailModel {
        let e = entry(data);
        let end = e.end;
        build(&e, end, e.start + 100.0)
    }

    #[test]
    fn a_generation_shows_model_usage_messages_and_settings() {
        let model = built(json!({
            "type": "generation",
            "model": "llama3.2:3b",
            "model_config": {"temperature": 0.2, "top_p": null, "stream": true},
            "usage": {"input_tokens": 1200, "output_tokens": 34},
            "input": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "What time is it?"}]},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "current_time", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "noon"}
            ],
            "output": [{"role": "assistant", "content": "It is noon."}]
        }));
        assert_eq!(model.kind, SpanKind::Model);
        assert_eq!(overview(&model, "Model").unwrap().value, "llama3.2:3b");
        assert_eq!(
            overview(&model, "Usage").unwrap().value,
            "1,200 → 34 tokens"
        );
        let settings = &overview(&model, "Settings").unwrap().value;
        assert!(
            settings.contains("\"temperature\":0.2") && !settings.contains("top_p"),
            "{settings}"
        );
        let labels: Vec<&str> = model.input.iter().map(|b| b.label.as_str()).collect();
        assert_eq!(labels, ["System", "User", "Assistant", "Tool result (c1)"]);
        assert_eq!(
            model.input[1].text, "What time is it?",
            "content parts are joined into text"
        );
        assert_eq!(model.input[2].text, "→ current_time({})");
        assert_eq!(model.output.len(), 1);
        assert_eq!(model.output[0].text, "It is noon.");
    }

    #[test]
    fn a_missing_half_of_the_usage_reads_as_a_question_mark() {
        let only_out =
            built(json!({"type": "generation", "model": "m", "usage": {"output_tokens": 34}}));
        assert_eq!(overview(&only_out, "Usage").unwrap().value, "? → 34 tokens");
        let only_in =
            built(json!({"type": "generation", "model": "m", "usage": {"prompt_tokens": 5}}));
        assert_eq!(overview(&only_in, "Usage").unwrap().value, "5 → ? tokens");
        let none = built(json!({"type": "generation", "model": "m", "usage": null}));
        assert_eq!(overview(&none, "Usage").unwrap().value, "? → ? tokens");
    }

    #[test]
    fn a_function_shows_arguments_as_pretty_json_when_they_parse() {
        let model = built(
            json!({"type": "function", "name": "calculate", "input": "{\"expression\":\"2+2\"}", "output": "4"}),
        );
        assert_eq!(model.input[0].label, "Arguments");
        assert!(
            model.input[0].text.contains("\n  \"expression\": \"2+2\""),
            "{}",
            model.input[0].text
        );
        assert!(model.input[0].mono);
        assert_eq!(model.output[0].text, "4");
        assert!(!model.output[0].mono);
        // Arguments that are not JSON are shown as they are.
        let raw =
            built(json!({"type": "function", "name": "t", "input": "not json {", "output": null}));
        assert_eq!(raw.input[0].text, "not json {");
        assert!(raw.output.is_empty());
    }

    #[test]
    fn handoff_guardrail_and_agent_overviews() {
        let handoff = built(json!({"type": "handoff", "from_agent": "A", "to_agent": "B"}));
        assert_eq!(overview(&handoff, "Handoff").unwrap().value, "A → B");
        let pending = built(json!({"type": "handoff", "from_agent": "A", "to_agent": null}));
        assert_eq!(
            overview(&pending, "Handoff").unwrap().value,
            "A → ?",
            "the target is filled in after the span starts"
        );

        let tripped =
            built(json!({"type": "guardrail", "name": "secrets_stay_local", "triggered": true}));
        assert_eq!(overview(&tripped, "Result").unwrap().tone, Tone::Danger);
        let passed = built(json!({"type": "guardrail", "name": "g", "triggered": false}));
        assert_eq!(overview(&passed, "Result").unwrap().value, "Passed");
        assert_eq!(overview(&passed, "Result").unwrap().tone, Tone::Positive);
        let unknown = built(json!({"type": "guardrail", "name": "g"}));
        assert_eq!(
            overview(&unknown, "Result").unwrap().value,
            "not recorded",
            "a missing flag is not a pass"
        );

        let agent = built(
            json!({"type": "agent", "name": "Lattice assistant", "tools": ["current_time", "calculate"], "handoffs": ["Model advisor"], "output_type": null}),
        );
        assert_eq!(
            overview(&agent, "Tools").unwrap().value,
            "current_time, calculate"
        );
        assert_eq!(overview(&agent, "Handoffs").unwrap().value, "Model advisor");
        assert_eq!(overview(&agent, "Output type").unwrap().value, "Text");
        let open = built(json!({"type": "agent", "name": "A", "tools": null, "handoffs": null}));
        assert_eq!(overview(&open, "Tools").unwrap().value, "not recorded yet");
        let bare = built(json!({"type": "agent", "name": "A", "tools": [], "handoffs": []}));
        assert_eq!(overview(&bare, "Tools").unwrap().value, "None");
    }

    #[test]
    fn task_and_turn_overviews_read_the_sdks_custom_data() {
        let task = built(
            json!({"type": "custom", "name": "task", "data": {"sdk_span_type": "task", "name": "Agent workflow", "usage": {"requests": 4, "input_tokens": 1992, "output_tokens": 238}}}),
        );
        assert_eq!(overview(&task, "Workflow").unwrap().value, "Agent workflow");
        assert_eq!(
            overview(&task, "Usage").unwrap().value,
            "1,992 → 238 tokens · 4 requests"
        );
        let turn = built(
            json!({"type": "custom", "name": "turn", "data": {"sdk_span_type": "turn", "turn": 2, "agent_name": "Model advisor"}}),
        );
        assert_eq!(overview(&turn, "Turn").unwrap().value, "2");
        assert_eq!(overview(&turn, "Agent").unwrap().value, "Model advisor");
        assert!(overview(&turn, "Usage").is_none());
    }

    #[test]
    fn an_error_shows_its_message_and_its_data_as_json() {
        let mut e =
            entry(json!({"type": "function", "name": "calculate", "input": "{}", "output": null}));
        e.rec.error = Some(SpanError {
            message: "Error running tool (non-fatal)".into(),
            data: Some(json!({"tool_name": "calculate", "error": "division by zero"})),
        });
        let model = build(&e, e.end, e.start + 1.0);
        let error = model.error.unwrap();
        assert_eq!(error.message, "Error running tool (non-fatal)");
        assert!(
            error
                .data
                .unwrap()
                .contains("\"error\": \"division by zero\"")
        );
        let mut bare = entry(json!({"type": "agent", "name": "A"}));
        bare.rec.error = Some(SpanError {
            message: "Max turns exceeded".into(),
            data: None,
        });
        assert_eq!(build(&bare, bare.end, 0.0).error.unwrap().data, None);
    }

    #[test]
    fn header_times_are_utc_and_an_open_span_counts_up() {
        let mut e = entry(json!({"type": "agent", "name": "A"}));
        e.rec.ended_at = None;
        e.end = None;
        let open = build(&e, None, e.start + 2.5);
        assert_eq!(open.started, "05:21:05.000 UTC");
        assert_eq!(open.duration, "2.50 s so far");
        let done = built(json!({"type": "agent", "name": "A"}));
        assert_eq!(done.duration, "1.00 s");
        // "Now" before the start (clock skew) is a zero duration, not a negative one.
        assert_eq!(build(&e, None, e.start - 50.0).duration, "0 ms so far");
    }

    #[test]
    fn long_text_is_bounded_and_says_so() {
        let long = "x".repeat(MAX_TEXT + 1234);
        let out = bounded(&long);
        assert!(
            out.starts_with("xxx") && out.ends_with("… (1,234 more characters not shown)"),
            "{}",
            &out[out.len() - 60..]
        );
        assert_eq!(bounded("short"), "short");
        let exact = "y".repeat(MAX_TEXT);
        assert_eq!(bounded(&exact), exact);
        // Counting is by character, not by byte: multi-byte text is not cut mid-character.
        let wide = "é".repeat(MAX_TEXT + 5);
        assert!(bounded(&wide).ends_with("… (5 more characters not shown)"));
    }

    #[test]
    fn markup_in_text_stays_text() {
        let model = built(
            json!({"type": "function", "name": "t", "input": "{}", "output": "<script>alert(1)</script> <b>bold</b>"}),
        );
        assert_eq!(
            model.output[0].text,
            "<script>alert(1)</script> <b>bold</b>"
        );
    }

    #[test]
    fn other_span_types_fall_back_to_their_data() {
        let model = built(json!({"type": "transcription", "input": {"format": "pcm"}}));
        assert_eq!(model.kind, SpanKind::Voice);
        assert!(
            overview(&model, "Data")
                .unwrap()
                .value
                .contains("\"format\": \"pcm\"")
        );
        assert!(overview(&model, "Span").is_some());
        let mcp =
            built(json!({"type": "mcp_tools", "server": "files", "result": ["read", "write"]}));
        assert_eq!(overview(&mcp, "Tools").unwrap().value, "read, write");
    }
}
