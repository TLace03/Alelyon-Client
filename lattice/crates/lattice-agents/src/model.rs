//! The model boundary: what a model is asked, what it answers, and how it fails.
//!
//! Ports the shape of `agents.models.interface.Model` (0.22.3) for the one
//! kind of model this crate runs: a streaming chat model reached through a
//! function-calling API. The SDK's interface also carries prompts, previous
//! response ids, output schemas and hosted tools; none of those exist here, and
//! [`ModelSettings`] keeps only the fields the port sends.
//!
//! Invariants:
//! - A [`Model`] is asked for one response at a time and answers with a stream
//!   of [`ModelEvent`]s that ends with exactly one [`ModelEvent::Done`]. A model
//!   that ends the stream without it has not produced a response.
//! - The stream is lazy and owns everything it needs: dropping it abandons the
//!   request (a real model closes its connection), which is how the runner
//!   cancels a model call.
//! - There is no default model anywhere in the crate, so a run can never fall
//!   back to a remote service the caller did not choose.

use futures::stream::BoxStream;
use lattice_protocol::Usage;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

/// Why a model call failed. A message never carries a response body, a URL or
/// a credential.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ModelError {
    /// The server answered with a status outside 200-299.
    #[error("the model server answered HTTP {0}")]
    Status(u16),
    /// The server could not be reached, or the connection broke. The text is a
    /// fixed phrase, never the transport's own message (which names the URL).
    #[error("could not talk to the model server: {0}")]
    Connection(String),
    /// The server stopped answering for longer than the read timeout.
    #[error("the model server did not answer in time")]
    Timeout,
    /// The server answered with something this client cannot read.
    #[error("the model server sent something unreadable: {0}")]
    Protocol(String),
    /// The model did something a well-behaved model does not (asked for a tool
    /// the agent does not have, or produced no final response). The SDK's
    /// `ModelBehaviorError`.
    #[error("{0}")]
    Behavior(String),
    /// The model's whole token budget went before it produced any answer: the
    /// stream ended with `finish_reason` `length` and no text, tool call or
    /// refusal (a reasoning model that spent it all thinking). The SDK's
    /// `ModelBehaviorError` for the same case
    /// (`chatcmpl_stream_handler.py:1190-1205`), kept a variant of its own so a
    /// caller can say what happened in words better than "asked for something it
    /// could not follow".
    #[error(
        "Chat Completions stream terminated with finish_reason='length' but produced no assistant text, tool call, or refusal."
    )]
    Truncated,
    /// Any other failure; the scripted model uses it for a scripted error.
    #[error("{0}")]
    Failed(String),
}

impl ModelError {
    /// The variant's name, which a scripted model's error span records the way
    /// the SDK's records the exception's class name.
    pub fn kind(&self) -> &'static str {
        match self {
            ModelError::Status(_) => "Status",
            ModelError::Connection(_) => "Connection",
            ModelError::Timeout => "Timeout",
            ModelError::Protocol(_) => "Protocol",
            ModelError::Behavior(_) => "Behavior",
            ModelError::Truncated => "Truncated",
            ModelError::Failed(_) => "Failed",
        }
    }
}

/// Sampling and tool settings. All optional: `None` means "not set", so the
/// model server's own default applies.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelSettings {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    /// `auto`, `required`, `none`, or the name of one tool to force.
    pub tool_choice: Option<String>,
    pub parallel_tool_calls: Option<bool>,
    /// Ask the server to report token usage in the stream.
    pub include_usage: Option<bool>,
}

impl ModelSettings {
    /// `agent.model_settings.resolve(run_config.model_settings)`: every field
    /// the run sets overrides the agent's, field by field.
    pub fn resolve(agent: &ModelSettings, run: Option<&ModelSettings>) -> ModelSettings {
        let Some(run) = run else { return agent.clone() };
        ModelSettings {
            temperature: run.temperature.or(agent.temperature),
            top_p: run.top_p.or(agent.top_p),
            max_tokens: run.max_tokens.or(agent.max_tokens),
            tool_choice: run
                .tool_choice
                .clone()
                .or_else(|| agent.tool_choice.clone()),
            parallel_tool_calls: run.parallel_tool_calls.or(agent.parallel_tool_calls),
            include_usage: run.include_usage.or(agent.include_usage),
        }
    }

    /// `ModelSettings.to_traceable_dict()`: every field this port has, unset
    /// ones as `null`, in the SDK's order.
    pub fn to_traceable_dict(&self) -> Value {
        json!({
            "temperature": self.temperature,
            "top_p": self.top_p,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "max_tokens": self.max_tokens,
            "include_usage": self.include_usage,
        })
    }
}

/// A tool the model may call, as the model is told about it.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments object.
    pub parameters: Value,
    pub strict: bool,
}

/// One tool call the assistant made.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallItem {
    pub call_id: String,
    pub name: String,
    /// The arguments exactly as the model wrote them (JSON text, or empty).
    pub arguments: String,
}

/// An image a model is shown, as its media type and its bytes in standard
/// base64.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Image {
    /// `image/png` or `image/jpeg`.
    pub media_type: String,
    pub base64: String,
}

/// One item of the conversation sent to the model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputItem {
    User(String),
    /// A user message with images, for a model that can see (the SDK's
    /// `input_image` content, which its Chat Completions converter sends as
    /// `image_url` parts): words, then each image.
    UserImages {
        text: String,
        images: Vec<Image>,
    },
    /// Assistant text and/or tool calls. Consecutive assistant items belong to
    /// one assistant message when they are converted for the wire.
    Assistant {
        text: Option<String>,
        tool_calls: Vec<ToolCallItem>,
    },
    ToolResult {
        call_id: String,
        output: String,
    },
}

/// One item of a model's answer, in the order the model produced it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputItem {
    Message {
        text: String,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    Reasoning {
        text: String,
    },
    /// The model declined to answer (`delta.refusal`), or the provider withheld
    /// the answer (`finish_reason` `content_filter`, reported as "Response
    /// withheld by the provider's content filter."). A refusal with no tool or
    /// handoff call in the same response ends the run as a refusal, as in the
    /// SDK (`turn_resolution.py:952-988`).
    Refusal {
        text: String,
    },
}

/// The complete answer to one request.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ModelResponse {
    pub output: Vec<OutputItem>,
    /// Absent, not zero, when the server did not report token usage.
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelEvent {
    /// A piece of the answer text, in order.
    TextDelta(String),
    /// A piece of the model's reasoning text.
    ReasoningDelta(String),
    /// The complete response. Always the last event.
    Done(ModelResponse),
}

/// Everything one model call needs.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelRequest {
    /// The agent's instructions; empty means no system message.
    pub system: String,
    pub input: Vec<InputItem>,
    pub tools: Vec<ToolSpec>,
    pub settings: ModelSettings,
}

/// How the runner fills a model's generation span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationTrace {
    /// The runner records the model name, its configuration, the input
    /// messages, the output and the usage, as the SDK's chat completions model
    /// does (`generation_span(model=..., model_config=...)`).
    Full,
    /// The runner records an empty generation span, as the SDK's scripted
    /// testing model does (`generation_span(disabled=False)` with no content).
    Bare,
}

pub trait Model: Send + Sync {
    /// The model's name, recorded on its generation spans.
    fn name(&self) -> &str;

    /// Static configuration worth recording next to the settings (a sanitised
    /// base URL, for example). Never a credential.
    fn config_for_trace(&self) -> Value;

    fn generation_trace(&self) -> GenerationTrace {
        GenerationTrace::Full
    }

    /// Ask for one response. The returned stream is lazy; dropping it cancels
    /// the request.
    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_settings_override_agent_settings_field_by_field() {
        let agent = ModelSettings {
            temperature: Some(0.2),
            max_tokens: Some(100),
            tool_choice: Some("required".into()),
            ..ModelSettings::default()
        };
        let run = ModelSettings {
            temperature: Some(0.9),
            top_p: Some(0.5),
            ..ModelSettings::default()
        };
        let resolved = ModelSettings::resolve(&agent, Some(&run));
        assert_eq!(resolved.temperature, Some(0.9));
        assert_eq!(resolved.top_p, Some(0.5));
        assert_eq!(resolved.max_tokens, Some(100));
        assert_eq!(resolved.tool_choice.as_deref(), Some("required"));
        assert_eq!(ModelSettings::resolve(&agent, None), agent);
    }

    #[test]
    fn the_traceable_dict_lists_every_field_with_nulls() {
        let dict = ModelSettings {
            temperature: Some(0.5),
            ..ModelSettings::default()
        }
        .to_traceable_dict();
        assert_eq!(dict["temperature"], json!(0.5));
        assert_eq!(dict["top_p"], Value::Null);
        assert_eq!(dict.as_object().unwrap().len(), 6);
    }

    #[test]
    fn model_errors_never_print_more_than_their_reason() {
        assert_eq!(
            ModelError::Status(500).to_string(),
            "the model server answered HTTP 500"
        );
        assert_eq!(ModelError::Timeout.kind(), "Timeout");
    }
}
