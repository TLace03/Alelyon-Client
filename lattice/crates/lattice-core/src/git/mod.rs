//! Git for the agent chat (the chat core's spec §6.3 FT6, §8.2): the
//! native reader of a folder's `.git`, which runs before any git process, and
//! the one runner every git call goes through. Not a port; the web's
//! `workspaces/gitio.py` and `capture.py` are the reference for its shape.
//!
//! - [`dotgit`]: FT6. A folder whose `.git` (a gitfile, `commondir`, an
//!   alternate, a configuration include, `core.excludesFile`,
//!   `core.attributesFile` or `core.worktree`) names a network or device path
//!   counts as one without git, and no git process starts there; only a
//!   folder it has read yields the [`dotgit::LocalRepo`] the runner needs.
//! - [`runner`]: `git.exe` by X2's rules, a fixed argv (hooks, fsmonitor,
//!   untracked cache and protocols overridden, no pager), X7's environment
//!   plus `GIT_NO_LAZY_FETCH=1` and `GIT_ALLOW_PROTOCOL=none`, and the
//!   listing and ignore checks the read tools use.
//! - [`config`]: git's configuration syntax, read for FT6.

pub mod config;
pub mod dotgit;
pub mod runner;

#[cfg(test)]
pub(crate) mod tests;
