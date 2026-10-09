//! Gate G-WEB in code: whether the native Lattice writes the shared chat
//! store (the chat core's spec §5.9, rule S23; row C3).
//!
//! While [`SHARED_WRITES`] is `false`, every write of the shared store
//! (`first_message`, `append`, `append_answer`, `rename`, `pin`, `supersede`,
//! `touch`, `archive`, `unarchive`) refuses with `Unavailable` and
//! [`REFUSAL`], and writes nothing. There is no production switch: unit
//! tests reach both paths through a `#[cfg(test)]` constructor that takes
//! the gate as a parameter, and a source guard (`tests/store_gate_guard.rs`)
//! holds the constant to this file.
//!
//! Row G5 opened it on 2026-10-05, once approved ("yes to G5, flip
//! shared writes"), after G-WEB's other three conditions: the web privacy
//! fix is on `main` and merged here, the write-path goldens are
//! generated from it, and the interop tests I1–I15 pass.

/// Whether the native Lattice may write the shared chat store.
pub(crate) const SHARED_WRITES: bool = true;

/// What a refused shared-store write says.
pub const REFUSAL: &str =
    "Chats are saved here once the web Lattice's privacy update is installed.";

/// Whether this build writes the shared store (for the list's `shared_writes`).
pub fn shared_writes() -> bool {
    SHARED_WRITES
}
