//! `unsafe` stays in lattice-sys (the chat core's spec §2.1, §10.4).
//!
//! Every other crate of the workspace is read for the word `unsafe` in code
//! (comments and string literals do not count). One site predates the chat
//! core and is listed by its exact text: `lattice-app`'s `prefer_opengl`, which
//! sets an environment variable before the window starts. Anything else fails.
//! In lattice-sys itself, every `unsafe` block must say why it is sound in a
//! `SAFETY:` comment just above it. Each check is first shown to catch a fixture.

use std::fs;
use std::path::{Path, PathBuf};

fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crates folder")
        .to_path_buf()
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// `text` with comments and the contents of string and character literals
/// blanked out, line structure kept. A small reader, enough for this
/// workspace's sources: line and block comments, plain, byte and raw strings
/// (`r"…"`, `r#"…"#`), character literals, and lifetimes left as code.
fn code_only(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c == '/' && next == Some('/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && next == Some('*') {
            let mut depth = 0;
            while i < chars.len() {
                if chars[i] == '/' && chars.get(i + 1) == Some(&'*') {
                    depth += 1;
                    i += 2;
                } else if chars[i] == '*' && chars.get(i + 1) == Some(&'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    if chars[i] == '\n' {
                        out.push('\n');
                    }
                    i += 1;
                }
            }
            continue;
        }
        let raw_start = c == 'r'
            && (next == Some('"') || next == Some('#'))
            && (i == 0 || !(chars[i - 1].is_alphanumeric() || chars[i - 1] == '_'));
        if raw_start {
            let mut j = i + 1;
            let mut hashes = 0;
            while chars.get(j) == Some(&'#') {
                hashes += 1;
                j += 1;
            }
            if chars.get(j) == Some(&'"') {
                j += 1;
                out.push_str("r\"");
                loop {
                    match chars.get(j) {
                        None => break,
                        Some('"') if (0..hashes).all(|k| chars.get(j + 1 + k) == Some(&'#')) => {
                            j += 1 + hashes;
                            break;
                        }
                        Some('\n') => out.push('\n'),
                        Some(_) => {}
                    }
                    j += 1;
                }
                out.push('"');
                i = j;
                continue;
            }
        }
        if c == '"' {
            out.push('"');
            i += 1;
            while i < chars.len() {
                match chars[i] {
                    '\\' => i += 2,
                    '"' => {
                        i += 1;
                        break;
                    }
                    '\n' => {
                        out.push('\n');
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
            out.push('"');
            continue;
        }
        if c == '\'' {
            // A character literal ('x', '\n', '\u{..}') or a lifetime ('a).
            let close = if next == Some('\\') {
                (i + 2..chars.len().min(i + 12)).find(|&k| chars[k] == '\'')
            } else if chars.get(i + 2) == Some(&'\'') {
                Some(i + 2)
            } else {
                None
            };
            if let Some(end) = close {
                out.push_str("' '");
                i = end + 1;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The lines (1-based, trimmed) where `unsafe` is used as a word in code.
fn unsafe_uses(text: &str) -> Vec<(usize, String)> {
    let code = code_only(text);
    let original: Vec<&str> = text.lines().collect();
    code.lines()
        .enumerate()
        .filter(|(_, line)| {
            line.match_indices("unsafe").any(|(at, _)| {
                let before = line[..at].chars().next_back();
                let after = line[at + "unsafe".len()..].chars().next();
                !before.is_some_and(is_identifier_char) && !after.is_some_and(is_identifier_char)
            })
        })
        .map(|(index, _)| {
            (
                index + 1,
                original.get(index).unwrap_or(&"").trim().to_owned(),
            )
        })
        .collect()
}

/// `unsafe {` blocks in `text` without a `SAFETY:` comment in the six lines
/// above them (a comment may run over several lines).
fn undocumented_blocks(text: &str) -> Vec<usize> {
    let lines: Vec<&str> = text.lines().collect();
    unsafe_uses(text)
        .into_iter()
        .filter(|(_, line)| line.contains("unsafe {"))
        .filter(|(number, _)| {
            let start = number.saturating_sub(7);
            !lines[start..number - 1]
                .iter()
                .any(|line| line.contains("SAFETY:"))
        })
        .map(|(number, _)| number)
        .collect()
}

/// The one site that predates the chat core, by its exact text.
const LISTED: [(&str, &str); 2] = [
    ("lattice-app/src/lib.rs", "#[allow(unsafe_code)]"),
    (
        "lattice-app/src/lib.rs",
        "unsafe { std::env::set_var(\"WGPU_BACKEND\", backend) };",
    ),
];

#[test]
fn the_reader_catches_unsafe_in_code_and_ignores_it_in_text() {
    let fixture = "fn f() {\n    // unsafe in a comment\n    let s = \"unsafe in a string\";\n    \
                   let r = r#\"unsafe \"raw\" too\"#;\n    let c = '\\'';\n    \
                   /* unsafe\n block */\n    unsafe { g() };\n}\n\
                   unsafe fn h() {}\nfn not_unsafe_x() {}\n";
    let found = unsafe_uses(fixture);
    println!("mutant fixture: {found:?}");
    assert_eq!(
        found.iter().map(|(line, _)| *line).collect::<Vec<_>>(),
        [8, 10]
    );
    assert_eq!(undocumented_blocks(fixture), [8]);
    let documented = "fn f() {\n    // SAFETY: g has no preconditions.\n    unsafe { g() };\n}\n";
    assert!(undocumented_blocks(documented).is_empty());
}

#[test]
fn no_crate_but_lattice_sys_uses_unsafe() {
    let crates = crates_dir();
    let mut checked = 0;
    let mut found = Vec::new();
    for name in [
        "lattice-protocol",
        "lattice-agents",
        "lattice-core",
        "lattice-app",
    ] {
        let mut paths = Vec::new();
        rust_sources(&crates.join(name).join("src"), &mut paths);
        assert!(!paths.is_empty(), "{name} has sources");
        for path in paths {
            checked += 1;
            let text = fs::read_to_string(&path).unwrap();
            let relative = path
                .strip_prefix(&crates)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            for (line, code) in unsafe_uses(&text) {
                if !LISTED.contains(&(relative.as_str(), code.as_str())) {
                    found.push(format!("{relative}:{line}: {code}"));
                }
            }
        }
    }
    assert!(checked > 40, "read only {checked} files");
    assert!(
        found.is_empty(),
        "`unsafe` outside lattice-sys:\n{}",
        found.join("\n")
    );
    let core = fs::read_to_string(crates.join("lattice-core/src/lib.rs")).unwrap();
    assert!(core.contains("#![deny(unsafe_code)]"));
}

#[test]
fn every_unsafe_block_in_lattice_sys_says_why_it_is_sound() {
    let mut paths = Vec::new();
    rust_sources(&crates_dir().join("lattice-sys").join("src"), &mut paths);
    let mut blocks = 0;
    for path in paths {
        let text = fs::read_to_string(&path).unwrap();
        blocks += unsafe_uses(&text)
            .iter()
            .filter(|(_, line)| line.contains("unsafe {"))
            .count();
        let missing = undocumented_blocks(&text);
        assert!(
            missing.is_empty(),
            "{}: unsafe blocks without SAFETY: at lines {missing:?}",
            path.display()
        );
    }
    assert!(blocks > 5, "found only {blocks} unsafe blocks");
}
