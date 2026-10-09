//! Handoffs: one agent passing the conversation to another.
//!
//! Ports `agents.handoffs.Handoff` and `handoffs.handoff()` (defaults only) and
//! `agents.util._transforms.transform_string_function_style` (0.22.3).
//!
//! A handoff is offered to the model as a tool named `transfer_to_<agent>` that
//! takes no arguments. When the model calls it the run continues with the
//! target agent, which sees the whole conversation so far
//! (`RunConfig.nest_handoff_history` is off by default in the SDK and has no
//! counterpart here) and the handoff tool's output is the SDK's transfer
//! message, `{"assistant": "<agent name>"}`.
//!
//! Not ported: `input_type`/`on_handoff` callbacks, input filters, and
//! `is_enabled`. Handoffs are always enabled, always argument-free, and
//! always keep the full history.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::agent::Agent;
use crate::pyjson::transfer_message;

/// `transform_string_function_style`: every character that is not an ASCII
/// letter, digit or underscore (whitespace included) becomes `_`, then the
/// result is lowercased. One `_` per character, as Python's per-code-point
/// `re.sub` does.
pub fn transform_string_function_style(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct Handoff {
    pub agent: Arc<Agent>,
    /// The tool the model calls; `transfer_to_<agent name>` by default.
    pub tool_name: String,
    pub tool_description: String,
}

impl Handoff {
    /// A handoff to `agent` with the SDK's default tool name and description
    /// (`Handoff.default_tool_name` / `default_tool_description`). The
    /// description ends with a space when the agent has no
    /// `handoff_description`, exactly as the SDK's f-string leaves it.
    pub fn to(agent: Arc<Agent>) -> Self {
        let tool_name = transform_string_function_style(&format!("transfer_to_{}", agent.name));
        let tool_description = format!(
            "Handoff to the {} agent to handle the request. {}",
            agent.name,
            agent.handoff_description.as_deref().unwrap_or("")
        );
        Self {
            agent,
            tool_name,
            tool_description,
        }
    }

    pub fn with_tool_name(mut self, tool_name: impl Into<String>) -> Self {
        self.tool_name = tool_name.into();
        self
    }

    pub fn with_tool_description(mut self, tool_description: impl Into<String>) -> Self {
        self.tool_description = tool_description.into();
        self
    }

    /// The target agent's name (`Handoff.agent_name`).
    pub fn agent_name(&self) -> &str {
        &self.agent.name
    }

    /// The handoff tool's arguments schema: an empty, strict object
    /// (`ensure_strict_json_schema({})`).
    pub fn input_json_schema() -> Value {
        json!({"additionalProperties": false, "type": "object", "properties": {}, "required": []})
    }

    /// The handoff tool's output (`Handoff.get_transfer_message`), written the
    /// way Python's `json.dumps` writes it: `{"assistant": "Model advisor"}`.
    pub fn transfer_message(&self, target: &Agent) -> String {
        transfer_message(&target.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names_follow_the_sdk_transform() {
        assert_eq!(
            transform_string_function_style("transfer_to_Model advisor"),
            "transfer_to_model_advisor"
        );
        assert_eq!(
            transform_string_function_style("transfer_to_Caf\u{e9} Bot-2!"),
            "transfer_to_caf__bot_2_"
        );
        assert_eq!(transform_string_function_style("A\tB\nC"), "a_b_c");
        assert_eq!(transform_string_function_style("\u{1F600}"), "_");
    }

    #[test]
    fn the_default_handoff_matches_the_sdk() {
        let target = Agent::builder("Model advisor")
            .instructions("Advise.")
            .build();
        let handoff = Handoff::to(target.clone());
        assert_eq!(handoff.tool_name, "transfer_to_model_advisor");
        // Trailing space: `f"...request. {agent.handoff_description or ''}"`.
        assert_eq!(
            handoff.tool_description,
            "Handoff to the Model advisor agent to handle the request. "
        );
        assert_eq!(handoff.agent_name(), "Model advisor");
        assert_eq!(
            handoff.transfer_message(&target),
            r#"{"assistant": "Model advisor"}"#
        );

        let described = Agent::builder("Billing")
            .handoff_description("Handles invoices.")
            .build();
        assert_eq!(
            Handoff::to(described).tool_description,
            "Handoff to the Billing agent to handle the request. Handles invoices."
        );
    }

    #[test]
    fn the_schema_is_an_empty_strict_object() {
        let schema = Handoff::input_json_schema();
        assert_eq!(schema["additionalProperties"], json!(false));
        assert_eq!(schema["properties"], json!({}));
        assert_eq!(schema["required"], json!([]));
    }
}
