//! MCP over stdio: the agent chat's MCP client (the chat core's spec §12,
//! phase H; T5, T7, CR2). Not a port; the protocol is the Model Context
//! Protocol's JSON-RPC over a child's stdin and stdout.
//!
//! - [`config`]: where servers are declared. The reader's own
//!   `<native>/chat/mcp.json`, in Cursor's shape (`{"mcpServers": {"<name>":
//!   {"command", "args", "env", "cwd"}}}`), whose environment values live
//!   only there; and, **only in a trusted folder**, the folder's
//!   `.lattice/mcp.json`, `.mcp.json` (Claude Code's) and `.cursor/mcp.json`,
//!   which may name environment variables but whose values are never read.
//!   A server with a `url` or a remote `type` is refused (T7).
//! - [`approvals`]: `<native>/chat/mcp_approvals.json`. A server runs only
//!   after the reader enabled it through `ConfirmPort(EnableMcpServer)`, and
//!   the approval is pinned to the SHA-256 of the entry's canonical JSON:
//!   any change to the entry asks again. The tools the reader allowed always
//!   (through `ConfirmPort(AllowMcpTool)`) and the tools the reader switched
//!   off are kept there too. Nothing in a folder grants any of this (FT4).
//! - [`launch`]: how a server starts: its program resolved by X2's rules
//!   (a bare name searched on Lattice's own `PATH`, never in the folder), a
//!   `.cmd` shim such as `npx` through `cmd.exe /d /s /c` only when nothing
//!   in its line has a meaning to `cmd.exe`, the X7 environment plus the
//!   variables the entry names, and stdin and stdout as pipes, in a Job
//!   Object of its own (`lattice-sys`).
//! - [`jsonrpc`] and [`client`]: one server's connection: newline-delimited
//!   JSON-RPC 2.0, `initialize` (the only timer at start, CR2), `tools/list`
//!   (paged, once per server process, again after `list_changed`) and
//!   `tools/call`; a `ping` from the server is answered and every other
//!   request it makes is refused.
//! - [`names`]: the name the model sees, `mcp__<server>__<tool>`, sanitised
//!   to `[A-Za-z0-9_]` and at most 64 characters.
//! - [`result`]: what a call returns to the model: its text, at most
//!   32 KiB; an image or audio part is named, not sent (no model here takes
//!   content parts yet).
//! - [`hub`]: [`McpHub`], the servers of this process: started lazily (at an
//!   Agent-mode turn that offers their tools, or when the reader asks in the
//!   Tools view), stopped when the reader stops them or Lattice closes
//!   (dropping the hub closes every Job). No keep-alive and no polling at
//!   idle (CR2): a server's reader threads block on its pipes.
//!
//! What the agent turn does with a server's tools (`convo::turn`): they are
//! offered in Agent mode, in a trusted folder (the policy engine refuses an
//! MCP call in Ask mode or an untrusted folder); **every call asks** unless
//! the reader allowed that tool always, and the approval itself is a native
//! dialog, as a command's is (`ConfirmPort(McpCall)`). What a server returns
//! is untrusted data, behind the secret tripwire like any tool result (T3).
//!
//! A server is a program on this machine: it runs as the reader and may use
//! the network or change files. Lattice says so when it is enabled; it does
//! not claim otherwise until a sandbox exists.

pub mod approvals;
pub mod client;
pub mod config;
pub mod hub;
#[cfg(test)]
pub(crate) mod hub_tests;
pub mod jsonrpc;
pub mod launch;
pub mod names;
pub mod result;
#[cfg(test)]
mod tests;

pub use hub::{McpHub, ServerStatus, ServerView, ToolView, TurnTool, TurnTools};
