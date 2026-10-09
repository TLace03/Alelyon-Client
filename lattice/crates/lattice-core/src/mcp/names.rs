//! The name the model sees for an MCP tool (the chat core's spec §12,
//! "Calls"). Not a port: the SDK offers a server's tool under its own name;
//! Lattice offers `mcp__<server>__<tool>`, so a tool can never take the name
//! of a built-in one (`read_file`, `run_command`) or of another server's.
//!
//! Every character outside `[A-Za-z0-9_]` becomes `_`, and the name is at
//! most [`MAX_MODEL_NAME`] characters (the limit the Chat Completions API
//! puts on a function's name). A name that would be longer, or that another
//! tool of the same turn already has, ends in `_` and 8 hex digits of the
//! SHA-256 of the server's and the tool's own names instead.

use std::collections::HashSet;

use crate::sha::sha256_hex;

/// The longest name a model is given.
pub const MAX_MODEL_NAME: usize = 64;
/// What every MCP tool's model name starts with.
pub const PREFIX: &str = "mcp__";

fn clean(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn hashed(server: &str, tool: &str, base: &str) -> String {
    let sha = sha256_hex(format!("{server}\0{tool}").as_bytes());
    let keep = MAX_MODEL_NAME - 9;
    let head: String = base.chars().take(keep).collect();
    format!("{head}_{}", &sha[..8])
}

/// `mcp__<server>__<tool>`, sanitised, without regard to other names.
pub fn model_name(server: &str, tool: &str) -> String {
    let base = format!("{PREFIX}{}__{}", clean(server), clean(tool));
    if base.len() <= MAX_MODEL_NAME {
        base
    } else {
        hashed(server, tool, &base)
    }
}

/// [`model_name`], made unique among `taken` (which it joins).
pub fn unique_model_name(server: &str, tool: &str, taken: &mut HashSet<String>) -> String {
    let mut name = model_name(server, tool);
    if taken.contains(&name) {
        name = hashed(server, tool, &name);
    }
    // Two pairs whose hashes share 8 hex digits and a head: count up.
    let mut n = 2u32;
    let base = name.clone();
    while taken.contains(&name) {
        let suffix = format!("_{n}");
        let head: String = base.chars().take(MAX_MODEL_NAME - suffix.len()).collect();
        name = format!("{head}{suffix}");
        n += 1;
    }
    taken.insert(name.clone());
    name
}

/// Is `name` one this module made: an MCP tool's model name?
pub fn is_model_name(name: &str) -> bool {
    name.starts_with(PREFIX)
}
