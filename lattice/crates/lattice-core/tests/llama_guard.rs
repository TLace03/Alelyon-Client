//! The managed llama.cpp server's launch, held by its source (spec
//! §22 LR6; ADR-0041 decision 1).
//!
//! - **The tools flag never appears.** llama.cpp's experimental built-in
//!   tools run inside the server, outside the approval path, so the flag that
//!   turns them on (two dashes and `tools`) may not appear in any crate's
//!   source outside a comment: not in code, not in a string, not in a test.
//! - **No Ollama in the local runtime.** The local runtime's sources
//!   (`llama/`, `net.rs`) and the chat's choices (`chat/vocab.rs`) name no
//!   Ollama endpoint (`/api/…`), no Ollama port (11434), and no `OLLAMA_`
//!   variable (a string that starts with it), apart from the one line that
//!   lists the names stripped from the server's environment.
//! - **`--jinja` is passed** (LR6, LR8).
//!
//! Unlike the other guards, this one reads string literals too, because a
//! flag is a string; only comment lines are left out. The tools rule reads
//! test modules as well (tests assemble the flag instead of writing it); the
//! Ollama rule skips them, since tests set `OLLAMA_*` on purpose to show it
//! changes nothing. A line counts as a test module's when the shared scanner
//! blanked it whole; a line of a multi-line string literal outside tests is
//! blanked the same way, which is this rule's known blind spot. The guard is
//! shown to catch a planted fixture before it is trusted.

mod scanner;

use scanner::{Source, sources};

/// The flag, assembled so this file does not hold it.
fn tools_flag() -> String {
    format!("--{}", "tools")
}

/// The lines of `source` that are not comment lines: (number, text).
fn code_lines(source: &Source) -> Vec<(usize, &str)> {
    source
        .original
        .lines()
        .enumerate()
        .map(|(at, line)| (at + 1, line))
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .collect()
}

fn tools_violations(sources: &[Source]) -> Vec<String> {
    let flag = tools_flag();
    let mut found = Vec::new();
    for source in sources {
        for (line, text) in code_lines(source) {
            if text.contains(&flag) {
                found.push(format!("{}:{line}: {}", source.relative, text.trim()));
            }
        }
    }
    found
}

/// The one line allowed to name `OLLAMA_`: the stripped prefixes.
const STRIPPED_LINE: &str =
    "pub const STRIPPED_ENV_PREFIXES: [&str; 2] = [\"LLAMA_ARG_\", \"OLLAMA_\"];";

/// Is line `number` of `source` inside a test module (blanked whole by the
/// scanner while the written line holds something)?
fn in_test_module(source: &Source, number: usize) -> bool {
    let written = source.original.lines().nth(number - 1).unwrap_or("");
    let scanned = source.code.lines().nth(number - 1).unwrap_or("");
    scanned.trim().is_empty() && !written.trim().is_empty()
}

fn ollama_violations(sources: &[Source]) -> Vec<String> {
    let mut found = Vec::new();
    for source in sources {
        let local_runtime = source.relative.starts_with("lattice-core/src/llama/")
            || source.relative == "lattice-core/src/net.rs"
            || source.relative == "lattice-core/src/chat/vocab.rs";
        if !local_runtime {
            continue;
        }
        for (line, text) in code_lines(source) {
            if in_test_module(source, line) {
                continue;
            }
            let named = text.contains("11434")
                || text.contains("/api/")
                || (text.contains("\"OLLAMA_") && text.trim() != STRIPPED_LINE);
            if named {
                found.push(format!("{}:{line}: {}", source.relative, text.trim()));
            }
        }
    }
    found
}

fn all_sources() -> Vec<Source> {
    sources(&[
        "lattice-protocol",
        "lattice-agents",
        "lattice-core",
        "lattice-app",
        "lattice-sys",
    ])
}

fn fixture(relative: &str, text: &str) -> Source {
    Source {
        relative: relative.to_owned(),
        original: text.to_owned(),
        code: scanner::without_test_modules(&scanner::code_only(text)),
    }
}

#[test]
fn the_guard_catches_a_planted_tools_flag_and_an_ollama_address() {
    let flag = tools_flag();
    let planted = fixture(
        "lattice-core/src/llama/planted.rs",
        &format!(
            "// A comment may say {flag}.\nfn argv() -> Vec<&'static str> {{\n    vec![\"{flag}\", \"web\"]\n}}\nconst A: &str = \"http://127.0.0.1:11434/api/tags\";\nfn v() {{ let _ = \"OLLAMA_HOST\"; }}\n"
        ),
    );
    let tools = tools_violations(std::slice::from_ref(&planted));
    let ollama = ollama_violations(std::slice::from_ref(&planted));
    println!("planted: {tools:?} {ollama:?}");
    assert_eq!(tools.len(), 1, "{tools:?}");
    assert!(tools[0].starts_with("lattice-core/src/llama/planted.rs:3:"));
    assert_eq!(ollama.len(), 2, "{ollama:?}");
    let elsewhere = fixture(
        "lattice-core/src/registry.rs",
        "const A: &str = \"OLLAMA_BASE_URL\";\n",
    );
    assert!(
        ollama_violations(&[elsewhere]).is_empty(),
        "the rule is the local runtime's"
    );
    let in_a_test = fixture(
        "lattice-core/src/llama/x.rs",
        "pub fn f() {}\n\n#[cfg(test)]\nmod tests {\n    fn t() { let _ = \"OLLAMA_HOST\"; }\n}\n",
    );
    assert!(
        ollama_violations(&[in_a_test]).is_empty(),
        "a test may set OLLAMA_*"
    );
}

#[test]
fn no_source_passes_the_tools_flag() {
    let sources = all_sources();
    assert!(
        sources
            .iter()
            .any(|s| s.relative == "lattice-core/src/llama/server.rs")
    );
    let found = tools_violations(&sources);
    assert!(found.is_empty(), "{found:#?}");
}

#[test]
fn the_local_runtime_names_no_ollama_endpoint_port_or_variable() {
    let sources = all_sources();
    let found = ollama_violations(&sources);
    assert!(found.is_empty(), "{found:#?}");
    let server = sources
        .iter()
        .find(|s| s.relative == "lattice-core/src/llama/server.rs")
        .unwrap();
    assert!(
        code_lines(server)
            .iter()
            .any(|(_, text)| text.trim() == STRIPPED_LINE),
        "the stripped names are listed where the guard expects them"
    );
    assert!(
        code_lines(server)
            .iter()
            .any(|(_, text)| text.contains("\"--jinja\"")),
        "--jinja is passed"
    );
}
