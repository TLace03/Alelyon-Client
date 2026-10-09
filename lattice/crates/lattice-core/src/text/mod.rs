//! Text work the core does for the interface and the model
//! (the chat core's spec §2.2 `text/`). Not a port, apart from
//! [`rank`], the shipping app's quick-open ranking.
//!
//! - [`rank`]: fuzzy path ranking for the file picker (`files`).
//! - [`diff`]: the review's line diff, its hunks and their ids, applying a
//!   chosen set of hunks, and intraline marks (row E3).
//! - [`ansi_strip`]: terminal escapes taken out of command output before the
//!   model reads it (row E7).
//! - [`find`]: a person's find and replace (literal or regular expression,
//!   case, whole words), for an editor's find bar and the folder's search.

pub mod ansi_strip;
pub mod diff;
pub mod find;
pub mod rank;
