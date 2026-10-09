//! The managed server's tool-call probe (the chat core's spec §22 LR8′),
//! which LR8 names and Local's tool calling waits on.
//!
//! On the first Agent-mode turn that would use the managed server, when no
//! record exists for this binary and model (`llama::probes::tool_probe`), the
//! turn sends one request first, through its own model client: one function
//! tool, [`TOOL`] (`text: string`), `tool_choice: "required"`, and the user
//! message [`PROMPT`]. It passes only if the reply holds exactly one call, of
//! that function, with arguments that parse as the JSON object
//! `{"text": "ok"}` ([`judge`]). The answer is recorded once per binary and
//! model (`llama::probes::record_tool_probe`) and in the turn's sidecar
//! (`Item::ToolProbe`). A failed probe ends that turn with
//! `words::LOCAL_NO_TOOLS`, and the model's later Agent-mode sends are plain
//! turns. The probe never runs against a named endpoint.

use futures::StreamExt;
use lattice_agents::model::{
    InputItem, Model, ModelEvent, ModelRequest, ModelSettings, OutputItem, ToolSpec,
};
use serde_json::{Value, json};

/// The probe's one tool.
pub const TOOL: &str = "lattice_probe_echo";
/// The probe's user message.
pub const PROMPT: &str = "Call lattice_probe_echo with text 'ok'.";
/// The most tokens the reply may take: room for a short thought first.
const MAX_TOKENS: u32 = 512;

/// The probe's request.
pub fn request() -> ModelRequest {
    ModelRequest {
        system: String::new(),
        input: vec![InputItem::User(PROMPT.to_owned())],
        tools: vec![ToolSpec {
            name: TOOL.to_owned(),
            description: "Echo the given text.".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": false,
            }),
            strict: false,
        }],
        settings: ModelSettings {
            temperature: Some(0.0),
            max_tokens: Some(MAX_TOKENS),
            tool_choice: Some("required".to_owned()),
            ..ModelSettings::default()
        },
    }
}

/// The request as the sidecar records it.
pub fn request_record() -> String {
    json!({
        "tools": [TOOL],
        "tool_choice": "required",
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
    })
    .to_string()
}

/// Does the reply pass: exactly one function call, of [`TOOL`], whose
/// arguments are the JSON object `{"text": "ok"}`?
pub fn judge(output: &[OutputItem]) -> bool {
    let calls: Vec<(&str, &str)> = output
        .iter()
        .filter_map(|item| match item {
            OutputItem::FunctionCall {
                name, arguments, ..
            } => Some((name.as_str(), arguments.as_str())),
            _ => None,
        })
        .collect();
    match calls.as_slice() {
        [(name, arguments)] => {
            *name == TOOL
                && serde_json::from_str::<Value>(arguments).ok() == Some(json!({"text": "ok"}))
        }
        _ => false,
    }
}

/// What the probe saw.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub passed: bool,
    /// The reply as the sidecar records it: its items, as JSON (a reasoning
    /// item by its length only).
    pub reply: String,
}

/// Send the probe through `model` and judge the reply. An error, or a stream
/// that ends without a reply, fails it.
pub async fn run(model: &dyn Model) -> Outcome {
    let mut stream = model.stream(request());
    let mut done = None;
    let mut failure = None;
    while let Some(event) = stream.next().await {
        match event {
            Ok(ModelEvent::Done(response)) => done = Some(response),
            Ok(_) => {}
            Err(error) => {
                failure = Some(error.to_string());
                break;
            }
        }
    }
    match done {
        Some(response) => Outcome {
            passed: judge(&response.output),
            reply: reply_record(&response.output),
        },
        None => Outcome {
            passed: false,
            reply: json!({"error": failure.unwrap_or_else(|| "no reply".to_owned())}).to_string(),
        },
    }
}

fn reply_record(output: &[OutputItem]) -> String {
    let items: Vec<Value> = output
        .iter()
        .map(|item| match item {
            OutputItem::Message { text } => json!({"type": "message", "text": text}),
            OutputItem::FunctionCall {
                name, arguments, ..
            } => json!({"type": "function_call", "name": name, "arguments": arguments}),
            OutputItem::Reasoning { text } => {
                json!({"type": "reasoning", "chars": text.chars().count()})
            }
            OutputItem::Refusal { text } => json!({"type": "refusal", "text": text}),
        })
        .collect();
    Value::Array(items).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: &str) -> OutputItem {
        OutputItem::FunctionCall {
            call_id: "c".into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// LR8′: one well-formed call of the probe's tool with `{"text": "ok"}`
    /// passes; prose, a malformed call, another tool, other arguments and two
    /// calls fail.
    #[test]
    fn only_one_call_of_the_echo_with_ok_passes() {
        assert!(judge(&[call(TOOL, r#"{"text": "ok"}"#)]));
        assert!(judge(&[
            OutputItem::Reasoning {
                text: "I should call it.".into()
            },
            call(TOOL, r#"{"text":"ok"}"#),
        ]));
        for (why, output) in [
            ("prose", vec![OutputItem::Message { text: "ok".into() }]),
            ("nothing", vec![]),
            ("malformed", vec![call(TOOL, r#"{"text": "ok""#)]),
            ("other text", vec![call(TOOL, r#"{"text": "OK"}"#)]),
            ("more", vec![call(TOOL, r#"{"text": "ok", "x": 1}"#)]),
            ("not an object", vec![call(TOOL, r#""ok""#)]),
            ("another tool", vec![call("echo", r#"{"text": "ok"}"#)]),
            (
                "two calls",
                vec![
                    call(TOOL, r#"{"text": "ok"}"#),
                    call(TOOL, r#"{"text": "ok"}"#),
                ],
            ),
        ] {
            assert!(!judge(&output), "{why}");
        }
    }

    #[test]
    fn the_request_offers_one_tool_and_requires_a_call() {
        let request = request();
        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.tools[0].name, TOOL);
        assert_eq!(request.settings.tool_choice.as_deref(), Some("required"));
        assert_eq!(request.input, vec![InputItem::User(PROMPT.to_owned())]);
        assert!(request.system.is_empty());
        let record: Value = serde_json::from_str(&request_record()).unwrap();
        assert_eq!(record["tools"], json!([TOOL]));
    }
}
