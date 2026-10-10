//! Lattice's client services, in Rust.
//!
//! `lattice-core` is the layer between the window (`lattice-app`, which only
//! knows the `lattice-protocol` contract) and the agent runtime
//! (`lattice-agents`). It replaces what the Python Lattice service does for
//! agent runs, and it reads the SAME files the Python Lattice reads, so the web
//! Lattice (still shipping) and the native one agree during the migration:
//!
//! - [`state`]: where state lives (the Python runtime's `paths`);
//! - [`keys`]: API keys by name, from the environment and the env files
//!   (the Python runtime's `keys`);
//! - [`registry`] and [`local_model`]: the model registry and the local Ollama
//!   settings (the Python runtime's `model_config` and `local_model`);
//! - [`choices`]: the models a person can pick for a run, and how one becomes a
//!   client;
//! - [`catalog`]: the built-in agents, their tools and the `secrets_stay_local`
//!   guardrail, [`calc`], the arithmetic tool's evaluator, and [`devmodel`], the
//!   scripted model that stands in for a language model in development;
//! - [`store`] and [`manager`]: the run store and [`CoreService`], the
//!   `RunService` implementation over `lattice-agents`, and [`recorder`], the
//!   trace processor that turns a run's spans and stream events into run
//!   events (shared with the agent chat's turns);
//! - [`models`]: a model client built through [`choices`], and the one-sentence
//!   wording of its failures, shared by runs and chat;
//! - for the chat core: [`fsx`], the never-delete file primitives (atomic
//!   writes of Lattice's own records, write-once files, moving a file aside, and
//!   the one removal of Lattice's own temporaries), [`sha`], SHA-256, and
//!   [`convo`], a conversation's native record (the sidecar) and the agent
//!   chat itself ([`convo::agent::AgentChat`], the `AgentChatService`), and [`policy`],
//!   the pure engine that allows, asks or refuses each call, and [`exec`], the
//!   environment a child gets, the one way the core starts a child, and
//!   `run_command` (the standing entries, the native approval, PowerShell
//!   with `-EncodedCommand`, one command per workspace, the model's output).
//! - [`git`]: a folder's `.git` read natively before any git process (FT6),
//!   and the one git runner (fixed argv, X7's environment, no lazy fetch, no
//!   protocol), over [`localfs`], which opens a path one component at a time
//!   and never follows a link to the network.
//! - [`workspace`]: a conversation's folder: attaching it (§6.1's refusals, a
//!   page's path only after a native confirmation), its identity, and the
//!   path rules every tool path passes (WP1–WP11, decided on the derived
//!   long-name path, and the authority class); [`ports`], the confirmation
//!   and attention ports the shell provides; [`tools`], the read tools over
//!   the folder's non-ignored files and the conversation's staged view, and
//!   the file picker's list ([`tools::read::files`]), ranked by [`text`];
//!   the staging tools ([`tools::edit`]) and [`staging`], a conversation's
//!   staged changes and its staged view: nothing reaches the folder before
//!   the reader keeps it; [`changes`], the checkpoints a Keep takes first
//!   (git without filters, hooks or network; first-touch copies without
//!   git), restore, staged as changes, and the checkpoints around each
//!   command with what it changed (acknowledged, or undone as a staged
//!   inverse).
//! - [`chat`]: the ports of the web Lattice's chat that the agent chat reuses,
//!   starting with [`chat::pyjson`], Python's JSON both ways.
//! - [`llama`]: the local model runtime, llama.cpp's server that the core
//!   starts and owns (ADR-0041): its files, its binary, its models and their
//!   GGUF headers, its settings, and the server itself; [`net`], the one HTTP
//!   client the core builds for it, loopback only.
//! - [`browser`]: the agent's own browser (Edge or Chrome on a profile of its
//!   own, its DevTools on two pipes it inherits and no port), which the agent
//!   operates as a person does: screenshots, clicks, typing, keys; the public
//!   web only, never a password or card field, and what others see or what
//!   moves money asks.
//! - [`mcp`]: MCP over stdio (spec §12): the servers the reader declares (and a
//!   trusted folder's), each enabled only by the reader and pinned to its
//!   entry's hash, started in a Job Object of its own, and their tools offered
//!   to Agent-mode turns, every call asking unless the reader allowed it.
//! - [`projects`]: the reader's projects, named groups of chats with their
//!   instructions and reference files, which lead the chats' agent turns and
//!   grant nothing.
//! - [`commands`] and [`skills`]: the reader's own and a trusted folder's
//!   prompt templates (`/name` in the composer) and skills (`SKILL.md`, which
//!   the agent reads with `use_skill`), both text that grants nothing.
//! - [`plugins`]: folders in Claude Code's plugin layout the reader added,
//!   whose commands and skills join those while they are on; their MCP
//!   servers are only listed, and their hooks never run.
//!
//! Parity is pinned, not asserted: `tests/parity/registry/*.json` are recorded
//! from the real Python code by `tools/lattice_native_parity.py`, the Rust tests
//! hold this crate to them, and `tests/frontend/test_lattice_native_parity.py`
//! fails when the Python side moves without them.
//!
//! What crosses out of this crate is only what the protocol allows: no API key,
//! key name or credential-bearing URL appears in a run event, a span or a
//! refusal. Keys are [`lattice_agents::SecretString`] from the moment they are
//! read.

#![deny(unsafe_code)]

pub mod acp;
pub mod bound;
pub mod browser;
pub mod calc;
pub mod catalog;
pub mod changes;
pub mod chat;
pub mod choices;
pub mod clock;
pub mod commands;
pub mod complete;
pub mod convo;
pub mod desktop;
pub mod devmodel;
pub mod env;
pub mod exec;
mod front;
pub mod fsx;
pub mod git;
pub mod hooks;
pub mod hosted;
pub mod keys;
pub mod llama;
pub mod local_model;
pub mod localfs;
pub mod manager;
pub mod mcp;
pub mod models;
pub mod net;
pub mod notes;
pub mod plugins;
pub mod policy;
pub mod ports;
pub mod projects;
mod py;
mod pyurl;
pub mod recorder;
pub mod registry;
pub mod secrets;
pub mod sha;
pub mod skills;
pub mod staging;
pub mod state;
pub mod store;
#[cfg(test)]
mod testkit;
pub mod text;
pub mod tools;
pub mod workspace;

pub use catalog::Catalog;
pub use env::{Env, MapEnv, ProcessEnv};
pub use keys::KeyStore;
pub use manager::{CoreConfig, CoreService};
pub use state::StateRoot;
