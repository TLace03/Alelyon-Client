//! Agents: a name, instructions, tools, handoffs and guardrails.
//!
//! Ports `agents.agent.Agent` (0.22.3) for what this crate runs: static
//! instructions, function tools, handoffs, input and output guardrails and
//! model settings. Output type is always plain text (`str`) and
//! `tool_use_behavior` is always `run_llm_again`, so neither is a field.
//!
//! Agents are immutable once built and shared as `Arc<Agent>`: a handoff holds
//! the target as an `Arc`, and the agent graph and the runner identify an agent
//! by that allocation, not by its name (two agents may share a name). An agent
//! that hands off to an agent that hands back to it cannot be built, because an
//! `Arc` cycle would need interior mutability the SDK's mutable
//! `agent.handoffs` list has and this port does not.

use std::sync::Arc;

use crate::guardrail::{InputGuardrail, OutputGuardrail};
use crate::handoff::Handoff;
use crate::model::ModelSettings;
use crate::tool::FunctionTool;

#[derive(Clone, Debug)]
pub struct Agent {
    pub name: String,
    /// The system prompt; empty means none.
    pub instructions: String,
    /// What other agents are told when they may hand off to this one.
    pub handoff_description: Option<String>,
    pub tools: Vec<FunctionTool>,
    pub handoffs: Vec<Handoff>,
    pub input_guardrails: Vec<InputGuardrail>,
    pub output_guardrails: Vec<OutputGuardrail>,
    pub model_settings: ModelSettings,
}

impl Agent {
    pub fn builder(name: impl Into<String>) -> AgentBuilder {
        AgentBuilder {
            agent: Agent {
                name: name.into(),
                instructions: String::new(),
                handoff_description: None,
                tools: Vec::new(),
                handoffs: Vec::new(),
                input_guardrails: Vec::new(),
                output_guardrails: Vec::new(),
                model_settings: ModelSettings::default(),
            },
        }
    }

    /// The agent's identity for the length of a run: its allocation.
    pub(crate) fn identity(agent: &Arc<Agent>) -> usize {
        Arc::as_ptr(agent) as usize
    }
}

#[derive(Debug)]
pub struct AgentBuilder {
    agent: Agent,
}

impl AgentBuilder {
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.agent.instructions = instructions.into();
        self
    }

    pub fn handoff_description(mut self, description: impl Into<String>) -> Self {
        self.agent.handoff_description = Some(description.into());
        self
    }

    pub fn tool(mut self, tool: FunctionTool) -> Self {
        self.agent.tools.push(tool);
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = FunctionTool>) -> Self {
        self.agent.tools.extend(tools);
        self
    }

    pub fn handoff(mut self, handoff: Handoff) -> Self {
        self.agent.handoffs.push(handoff);
        self
    }

    /// A handoff to `target` with the SDK's defaults.
    pub fn handoff_to(self, target: Arc<Agent>) -> Self {
        self.handoff(Handoff::to(target))
    }

    pub fn input_guardrail(mut self, guardrail: InputGuardrail) -> Self {
        self.agent.input_guardrails.push(guardrail);
        self
    }

    pub fn output_guardrail(mut self, guardrail: OutputGuardrail) -> Self {
        self.agent.output_guardrails.push(guardrail);
        self
    }

    pub fn model_settings(mut self, settings: ModelSettings) -> Self {
        self.agent.model_settings = settings;
        self
    }

    pub fn build(self) -> Arc<Agent> {
        Arc::new(self.agent)
    }
}
