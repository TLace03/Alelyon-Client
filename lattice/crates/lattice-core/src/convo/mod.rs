//! The agent chat's conversations, as only the native Lattice records them
//! (the chat core's spec §5.7).
//!
//! A conversation's words (the reader's messages and the final answers) live
//! in the shared chat store, which the web Lattice reads too. Everything else
//! (tool calls and results, approvals, questions, staged changes, reviews,
//! checkpoints, files moved aside) lives in the native **sidecar**, keyed by the
//! shared thread id:
//! - [`item`]: the records a sidecar holds, one JSON object per line;
//! - [`sidecar`]: the store, `<globals>/lattice_native/chat/conversations/<id>/`,
//!   with its `meta.json`, its append-only `items.jsonl`, its content-addressed
//!   `blobs/`, its writer lock, and its binding to the shared row's `created`.
//!
//! - [`binding`]: a sidecar bound to its row in the shared store (the
//!   production `TranscriptStore`) by the row's `created`; a record left by an
//!   earlier conversation with the same id is moved aside, never used.
//! - [`caps`]: what a model can do (tools, vision, context), for the
//!   managed llama.cpp server from its header, `/props` and probe record.
//!
//! - [`log`] and [`follow`]: a conversation's events in memory, with their
//!   bounds (CR7), and the follow coalescer that hands them to the
//!   interface by the frame-gap contract (FG1–FG8).
//!
//! - [`replay`], [`tripwire`] and [`prompt_agent`]: an agent turn's model
//!   input: the transcript and the sidecar replayed (UT1, the same items for
//!   every target, Amendment 4; a budget), every item behind the secret
//!   tripwire (T3), and the prompt and tools, pinned by a golden.
//!
//! - [`agent`]: `AgentChat`, the `AgentChatService` the interface calls, with
//!   one agent turn in `turn` and what it shows in `views` (row E11).
//! - [`tasks`]: the tasks the agent suggests (chips), which run only when the
//!   reader starts one, as a new conversation.
//! - [`artifacts`]: documents the agent saves beside the chat, with their
//!   versions.
//! - [`todos`]: the agent's to-do list for the work in hand.
//! - [`plans`]: the plan the agent proposes in Ask mode, which the reader
//!   approves before anything changes.
//! - [`images`]: the images the reader attaches to a message.
//! - [`mentions`]: the files a message names with `@path`, given to the
//!   model with it.
//!
//! Not a port: the sidecar is native-only; its tail healing follows Python's
//! `history.py`.

pub mod agent;
#[cfg(test)]
mod agent_interop_tests;
#[cfg(test)]
mod agent_privacy_tests;
#[cfg(test)]
mod agents_acp_tests;
#[cfg(test)]
pub(crate) mod agent_tests;
pub mod artifacts;
pub mod background;
#[cfg(test)]
mod background_tests;
#[cfg(test)]
mod artifacts_tests;
#[cfg(test)]
mod auto_mode_tests;
pub mod binding;
#[cfg(test)]
mod browser_tests;
pub mod caps;
#[cfg(test)]
mod compute_tests;
#[cfg(test)]
mod connections_tests;
pub mod follow;
pub mod git_tools;
#[cfg(test)]
mod git_tools_tests;
pub mod helpers;
pub(crate) mod hooked;
#[cfg(test)]
mod hooks_tests;
#[cfg(test)]
mod helpers_tests;
#[cfg(test)]
mod follow_tests;
pub mod images;
#[cfg(test)]
mod images_tests;
pub mod item;
pub(crate) mod lab_turns;
#[cfg(test)]
mod lab_turns_tests;
pub mod log;
#[cfg(test)]
mod mcp_tests;
pub mod memory;
#[cfg(test)]
mod memory_tests;
pub mod mentions;
#[cfg(test)]
mod mentions_tests;
pub mod plans;
#[cfg(test)]
mod plans_tests;
#[cfg(test)]
mod plugins_tests;
pub mod probe;
#[cfg(test)]
mod probe_tests;
#[cfg(test)]
mod projects_tests;
pub mod prompt_agent;
pub mod replay;
#[cfg(test)]
mod replay_tests;
pub mod sidecar;
#[cfg(test)]
mod skills_tests;
pub mod tasks;
#[cfg(test)]
mod tasks_tests;
pub mod todos;
#[cfg(test)]
mod todos_tests;
pub mod tripwire;
pub(crate) mod turn;
pub(crate) mod views;
