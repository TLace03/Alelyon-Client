//! An agent runtime ported from the OpenAI Agents SDK.
//!
//! `lattice-agents` is a Rust translation of the subset of openai-agents-python
//! 0.22.3 (MIT, Copyright (c) 2025 OpenAI; see `NOTICE`) that Lattice runs:
//!
//! - agents with function tools, handoffs and input and output guardrails
//!   ([`Agent`], [`FunctionTool`], [`Handoff`], [`InputGuardrail`],
//!   [`OutputGuardrail`]);
//! - streamed runs with turn limits and cancellation, started from one message
//!   or from a list of input items ([`run_streamed`], [`run_streamed_items`]);
//! - sessions that keep a conversation between runs, implemented by the caller
//!   ([`Session`]);
//! - tool calls that wait in place for the caller's approval ([`NeedsApproval`],
//!   [`ApprovalPort`]);
//! - steering: messages handed to a working run, seen before its next model
//!   call ([`RunControl::steer`]), a native addition the SDK does not have;
//! - an OpenAI-compatible Chat Completions model ([`ChatCompletionsModel`]) and a
//!   scripted one for tests ([`testing::ScriptedModel`]);
//! - the SDK's trace and span tree, delivered to local [`TraceProcessor`]s;
//! - the SDK's agent-graph walk ([`graph::agent_graph`]).
//!
//! What it deliberately is not: it has no exporter (traces never leave the
//! process unless a caller's own processor sends them), no default model (a run
//! must be given one, so nothing silently falls back to a remote service), no
//! Responses API, no hosted tools, no session store of its own (a caller's
//! [`Session`] keeps the conversation, and nothing here removes a record of
//! one), no MCP client (a caller brings an MCP server's tools as function
//! tools; one names its server, and its span records it as the SDK's
//! `invoke_mcp_tool` does) and no Python. An approval comes only from the caller's
//! [`ApprovalPort`], never from text. `tests/source_guard.rs` holds the crate to
//! all of this.
//!
//! Faithfulness is checked, not asserted: `tests/conformance.rs` replays scenarios
//! through the port and compares stream events and span trees with goldens
//! recorded from the real SDK by `tools/lattice_sdk_goldens.py`. Each module
//! header names the SDK module it ports and lists where it deliberately differs.
//!
//! The wire types shared with the interface (spans, usage) are
//! [`lattice_protocol`]'s.

pub mod agent;
pub mod chat_completions;
pub mod graph;
pub mod guardrail;
pub mod handoff;
pub mod model;
mod pyjson;
pub mod run;
pub mod secret;
pub mod session;
pub mod testing;
pub mod tool;
pub mod trace;
mod usage;

pub use agent::{Agent, AgentBuilder};
pub use chat_completions::{ChatCompletionsConfig, ChatCompletionsModel, sanitize_url_for_trace};
pub use graph::agent_graph;
pub use guardrail::{GuardrailContext, GuardrailOutput, InputGuardrail, OutputGuardrail};
pub use handoff::{Handoff, transform_string_function_style};
pub use model::{
    GenerationTrace, InputItem, Model, ModelError, ModelEvent, ModelRequest, ModelResponse,
    ModelSettings, OutputItem, ToolCallItem, ToolSpec,
};
pub use run::{
    CancelMode, RunConfig, RunContext, RunControl, RunError, RunHandle, RunItem, RunItemName,
    RunResult, Steer, StreamEvent, run_streamed, run_streamed_items,
};
pub use secret::SecretString;
pub use session::{Session, SessionError};
pub use tool::{
    ApprovalDecision, ApprovalPort, ApprovalPredicate, ApprovalRequest, FunctionTool,
    MAX_NOTE_CHARS, NeedsApproval, ToolContext, ToolError, ToolErrorPolicy, strict_object_schema,
};
pub use trace::{TraceProcessor, TraceRecord};
