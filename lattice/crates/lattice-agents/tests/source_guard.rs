//! Guards on the crate's own sources and notice.
//!
//! The port must have no path by which a trace or a task reaches a machine the
//! caller did not choose. Reading the sources for the SDK's endpoints is a blunt
//! instrument, but it catches the mistake that matters (porting the SDK's
//! exporter or its default model along with everything else) at review time
//! rather than in production.
//!
//! The chat core's ports add three more: the crate implements no `Session`
//! (the core keeps conversations), nothing in it removes a conversation record
//! (`pop_item`, `clear_session`), and the loop and the tools hold no string
//! that could read an approval out of text. Each of those checks is first
//! shown to catch a mutant fixture.

use std::fs;
use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("the source directory is readable") {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn sources() -> Vec<(PathBuf, String)> {
    let mut paths = Vec::new();
    rust_sources(&manifest_dir().join("src"), &mut paths);
    assert!(paths.len() >= 10, "found only {} source files", paths.len());
    paths
        .into_iter()
        .map(|path| (path.clone(), fs::read_to_string(&path).unwrap()))
        .collect()
}

/// The source without its comments: module headers may say what was left out
/// (and by what name the SDK calls it); code may not contain it.
fn code_only(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn no_source_names_the_sdks_endpoints() {
    for (path, text) in sources() {
        for forbidden in ["api.openai.com", "traces/ingest", "openai.com/v1"] {
            assert!(
                !text.contains(forbidden),
                "{} mentions {forbidden}",
                path.display()
            );
        }
    }
}

#[test]
fn no_source_reads_credentials_from_the_environment_or_uses_unsafe() {
    for (path, text) in sources() {
        let code = code_only(&text);
        for forbidden in ["env::var", "OPENAI_API_KEY", "set_var", "unsafe "] {
            assert!(
                !code.contains(forbidden),
                "{} contains {forbidden}",
                path.display()
            );
        }
    }
}

#[test]
fn there_is_no_exporter_or_default_model_in_the_public_surface() {
    // A default model would let a run fall back to a service nobody chose, and
    // an exporter would send traces somewhere. Neither exists; a name like these
    // appearing in a source file means someone added one.
    for (path, text) in sources() {
        let code = code_only(&text);
        for forbidden in [
            "BatchTraceProcessor",
            "BackendSpanExporter",
            "default_model",
            "DefaultModel",
            "default_exporter",
        ] {
            assert!(
                !code.contains(forbidden),
                "{} contains {forbidden}",
                path.display()
            );
        }
    }
}

#[test]
fn every_module_explains_itself_and_names_what_it_ports() {
    for (path, text) in sources() {
        assert!(
            text.starts_with("//!"),
            "{} has no module header",
            path.display()
        );
        let header: String = text
            .lines()
            .take_while(|line| line.starts_with("//!"))
            .collect::<Vec<_>>()
            .join("\n")
            .to_lowercase();
        assert!(
            header.contains("port"),
            "{}'s header does not say what it ports (or that it is not a port)",
            path.display()
        );
    }
}

#[test]
fn the_notice_carries_the_attribution_line_and_the_full_mit_text() {
    let notice = fs::read_to_string(manifest_dir().join("NOTICE")).expect("the NOTICE file exists");
    let first = notice.lines().next().unwrap();
    assert_eq!(
        first,
        "Portions of lattice-agents are a Rust translation of openai-agents-python 0.22.3 \
         (https://github.com/openai/openai-agents-python), used under the MIT License below."
    );
    for required in [
        "MIT License",
        "Copyright (c) 2025 OpenAI",
        "Permission is hereby granted, free of charge",
        "The above copyright notice and this permission notice shall be included in all",
        "THE SOFTWARE IS PROVIDED \"AS IS\"",
    ] {
        assert!(notice.contains(required), "NOTICE lacks: {required}");
    }
}

// ---- the chat core's ports (the chat core's spec sections 3.2 rule 7 and 3.6)
//
// Each check below is a function over a file's code, so it can be shown to
// catch a mutant fixture before it is trusted to find nothing in the sources.

/// Code lines that implement `Session` for some type (`impl Session for X`,
/// `impl crate::session::Session for X`, `impl<T> Session for T`, ...).
fn session_impls(code: &str) -> Vec<String> {
    code.lines()
        .filter(|line| {
            let line = line.trim_start();
            line.starts_with("impl") && line.contains("Session for ")
        })
        .map(str::to_owned)
        .collect()
}

fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Every use of `name` as a whole identifier in `code`.
fn identifier_uses(code: &str, name: &str) -> usize {
    code.match_indices(name)
        .filter(|(at, _)| {
            let before = code[..*at].chars().next_back();
            let after = code[at + name.len()..].chars().next();
            !before.is_some_and(is_identifier_char) && !after.is_some_and(is_identifier_char)
        })
        .count()
}

/// The identifiers that remove conversation records, wherever code uses them.
fn record_removals(code: &str) -> Vec<&'static str> {
    ["pop_item", "clear_session"]
        .into_iter()
        .filter(|name| identifier_uses(code, name) > 0)
        .collect()
}

/// The string literals in `code` (a rough reader: double-quoted text with
/// backslash escapes; enough for this crate's sources).
fn string_literals(code: &str) -> Vec<String> {
    let mut literals = Vec::new();
    let mut current: Option<String> = None;
    let mut escaped = false;
    for c in code.chars() {
        match current.as_mut() {
            None if c == '"' => current = Some(String::new()),
            None => {}
            Some(text) if escaped => {
                text.push(c);
                escaped = false;
            }
            Some(_) if c == '\\' => escaped = true,
            Some(_) if c == '"' => literals.extend(current.take()),
            Some(text) => text.push(c),
        }
    }
    literals
}

/// String literals holding the word "approve" (any case): what code that read
/// an approval out of model or tool text would need.
fn approve_literals(code: &str) -> Vec<String> {
    string_literals(code)
        .into_iter()
        .filter(|literal| {
            literal
                .to_lowercase()
                .split(|c: char| !c.is_alphanumeric())
                .any(|word| word == "approve")
        })
        .collect()
}

#[test]
fn the_guards_catch_their_mutant_fixtures() {
    let session_fixture = "pub struct Disk;\nimpl Session for Disk {\n}\n";
    let pathed_fixture = "impl crate::session::Session for Disk {";
    let removal_fixture = "fn pop_item(&self) {}\nsession.clear_session().await;\n";
    let approve_fixture = "if output.contains(\"approve\") {\nlet said = \"Approve call_1\";\n";
    let caught = [
        ("Session impl", session_impls(session_fixture).len()),
        ("pathed Session impl", session_impls(pathed_fixture).len()),
        (
            "pop_item and clear_session",
            record_removals(removal_fixture).len(),
        ),
        (
            "\"approve\" literals",
            approve_literals(approve_fixture).len(),
        ),
    ];
    for (what, found) in caught {
        println!("mutant fixture ({what}): {found} found");
        assert!(found > 0, "the guard misses a {what}");
    }
    assert_eq!(record_removals(removal_fixture).len(), 2);
    assert_eq!(approve_literals(approve_fixture).len(), 2);
    // ... and does not trip on what the crate legitimately says.
    let legitimate = "const T: &str = \"Tool execution was not approved.\";\n\
                      Approve => Ok(()),\nlet popped = pop_items_total;\n";
    assert!(approve_literals(legitimate).is_empty());
    assert!(record_removals(legitimate).is_empty());
    assert!(session_impls("pub trait Session: Send + Sync {").is_empty());
}

#[test]
fn the_crate_implements_no_session_and_never_removes_a_conversation_record() {
    // Where a conversation is kept is the caller's business (the chat core's),
    // and Lattice never deletes one. A test-only module file (`*tests.rs`) may
    // hold a fake session; nothing else in `src` may implement one.
    for (path, text) in sources() {
        let code = code_only(&text);
        let test_only = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with("tests.rs"));
        if !test_only {
            let impls = session_impls(&code);
            assert!(
                impls.is_empty(),
                "{} implements Session: {impls:?}",
                path.display()
            );
        }
        let removals = record_removals(&code);
        assert!(removals.is_empty(), "{} uses {removals:?}", path.display());
    }
}

#[test]
fn the_loop_and_the_tools_read_no_approval_out_of_text() {
    // A decision comes only from the `ApprovalPort` (section 3.2 rule 7).
    let mut checked = 0;
    for (path, text) in sources() {
        let name = path.file_name().and_then(|name| name.to_str());
        if !matches!(name, Some("run.rs" | "tool.rs")) {
            continue;
        }
        checked += 1;
        let literals = approve_literals(&code_only(&text));
        assert!(
            literals.is_empty(),
            "{} has approval words in string literals: {literals:?}",
            path.display()
        );
    }
    assert_eq!(checked, 2, "run.rs and tool.rs are where they were");
}
