//! Running programs for the agent chat (the chat core's spec §7.6).
//! Not a port; the environment allowlist is Python's agent host's
//! (the Python agent's `session.py`, `SAFE_AGENT_ENV_NAMES`).
//!
//! - [`spawn`]: the environment every child gets (X7) and the one way the core
//!   starts a child, through `lattice_sys::process` (X5, X8);
//! - [`resolve`]: which program a bare name runs (X2): never one in the
//!   workspace or Lattice's state, never a batch file or a script;
//! - [`command`]: which command text may ever match a standing entry (X1,
//!   with Windows PowerShell 5.1's own names refused), which program may
//!   ever be one (X3, by file stem), and the bidi refusal (X6);
//! - [`allowlist`]: the standing entries, `Exact` only, made only through the
//!   native `AllowAlways` dialog, matched by X4 (program, arguments and
//!   working folder);
//! - [`run`]: `run_command` itself: the card, the reader's Approve (which only
//!   asks: the native `RunCommand` dialog decides), the run in PowerShell
//!   with `-EncodedCommand` or directly for a standing match, one command per
//!   workspace (X14), and the output the model reads (X9).
//!
//! Nothing here starts a shell on its own: the program is an absolute path and
//! the arguments are an argv. A command approved once runs in Windows
//! PowerShell, by absolute path, with the text the reader saw.

pub mod allowlist;
pub mod command;
pub mod resolve;
pub mod run;
pub mod spawn;
