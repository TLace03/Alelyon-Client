//! The agent chat's tools (the chat core's spec §7). Not a port.
//!
//! - [`read`]: `list_dir`, `glob`, `read_file` and `grep`, over the folder's
//!   non-ignored file set and the conversation's staged view, every path
//!   through the workspace's path rules, every cap stated.
//! - [`edit`]: `edit_file`, `write_file` and `delete_file`, which stage a
//!   change in the conversation's record and write nothing to the folder
//!   (OD2); the reader's Keep is the write.
//!
//! - [`ask`]: `ask_question`, which waits for the reader's answer with no
//!   timer; Stop cancels it.
//!
//! `run_command` is `exec::run` (row E7); the tools' definitions for the
//! model (names, descriptions, JSON schemas) are `convo::prompt_agent`,
//! pinned by its golden, and their handlers `convo::turn` (row E11).

pub mod ask;
pub mod edit;
pub mod read;

#[cfg(test)]
mod ask_tests;
#[cfg(test)]
mod edit_tests;
#[cfg(test)]
mod read_tests;
