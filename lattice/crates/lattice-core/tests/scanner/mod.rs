//! The source scanner the chat core's guards share (the chat core's spec
//! §5.8 ND4, §10.4 and CR9).
//!
//! A guard asks which words a crate's code uses. "Code" leaves out:
//! - comments, and the contents of string and character literals (a module
//!   header may name what it forbids; a message may say "unlinked");
//! - test modules: a `#[cfg(test)] mod … { … }` block (also `cfg(all(test,
//!   …))`, which is compiled only for tests too), and whole files named
//!   `tests.rs` or `*_tests.rs`. A `cfg(any(test, …))` item is not a test
//!   module: it can be compiled into a shipped build.
//!
//! Lines keep their numbers, so a finding names `file:line`. Each rule is
//! pinned by a fixture with the forbidden word inside and outside a test module
//! (`never_delete_guard.rs`).

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

/// One source file, read for a guard.
pub struct Source {
    /// Relative to the crates folder, with `/` (`lattice-core/src/fsx.rs`).
    pub relative: String,
    /// The file as written.
    pub original: String,
    /// Comments, literal contents and test modules blanked; lines kept.
    pub code: String,
}

impl Source {
    /// The written text of 1-based line `number`, trimmed.
    pub fn line(&self, number: usize) -> &str {
        self.original.lines().nth(number - 1).unwrap_or("").trim()
    }
}

/// The workspace's crates folder.
pub fn crates_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crates folder")
        .to_path_buf()
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// `tests.rs` and `*_tests.rs` hold only tests.
pub fn is_test_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name == "tests.rs" || name.ends_with("_tests.rs"))
}

/// Every non-test source under `<crate>/src` for each crate named, read.
pub fn sources(crates: &[&str]) -> Vec<Source> {
    let root = crates_dir();
    let mut out = Vec::new();
    for name in crates {
        let mut paths = Vec::new();
        rust_files(&root.join(name).join("src"), &mut paths);
        assert!(!paths.is_empty(), "{name} has no sources");
        paths.sort();
        for path in paths {
            if is_test_file(&path) {
                continue;
            }
            let original = fs::read_to_string(&path).unwrap();
            let code = without_test_modules(&code_only(&original));
            let relative = path
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            out.push(Source {
                relative,
                original,
                code,
            });
        }
    }
    out
}

/// `text` with comments and the contents of string and character literals
/// blanked out, line structure kept: line and nested block comments; plain,
/// byte and raw strings (`r"…"`, `r#"…"#`, `br"…"`, `cr#"…"#`); character
/// literals, escaped ones included (`'\''`); a lifetime stays code.
pub fn code_only(text: &str) -> String {
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
        // `r"…"` / `r#"…"#`, also as a raw byte or raw C string (`br"…"`,
        // `cr#"…"#`): the `r` starts the token, or follows a `b` or `c` that
        // does (spec 22.6 ND4a; a backslash before the closing quote of a raw
        // string escapes nothing).
        let boundary = |at: usize| at == 0 || !is_identifier_char(chars[at - 1]);
        let raw_start = c == 'r'
            && (next == Some('"') || next == Some('#'))
            && (boundary(i) || (matches!(chars[i - 1], 'b' | 'c') && boundary(i - 1)));
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
                    '\\' => {
                        // A `\` at the end of a line continues the string: the
                        // newline it escapes still ends a line (with LF endings
                        // nothing else would keep it; with CRLF the `\r` was
                        // the escaped character and the `\n` stayed).
                        if chars.get(i + 1) == Some(&'\n') {
                            out.push('\n');
                        }
                        i += 2;
                    }
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
            // An escaped character literal closes after the escaped character
            // (`'\''`, `'\\'`, `'\u{1F600}'`), never on it (spec 22.6 ND4a).
            let close = if next == Some('\\') {
                (i + 3..chars.len().min(i + 12)).find(|&k| chars[k] == '\'')
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

/// Whether a `cfg` predicate is compiled only for tests: `test`, or `all(…)`
/// with `test` among its arguments. `any(test, …)` is not.
fn test_only_predicate(predicate: &str) -> bool {
    let compact: String = predicate.chars().filter(|c| !c.is_whitespace()).collect();
    if compact == "test" {
        return true;
    }
    let Some(inner) = compact
        .strip_prefix("all(")
        .and_then(|rest| rest.strip_suffix(')'))
    else {
        return false;
    };
    let mut depth = 0;
    let mut start = 0;
    let mut arguments = Vec::new();
    for (at, c) in inner.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                arguments.push(&inner[start..at]);
                start = at + 1;
            }
            _ => {}
        }
    }
    arguments.push(&inner[start..]);
    arguments.contains(&"test")
}

/// `code` (already through [`code_only`]) with every test-only `mod … { … }`
/// block blanked, lines kept.
pub fn without_test_modules(code: &str) -> String {
    let mut chars: Vec<char> = code.chars().collect();
    let mut at = 0;
    while let Some(found) = find(&chars, at, "#[cfg(") {
        let predicate_start = found + "#[cfg(".len();
        let Some(predicate_end) = matching(&chars, predicate_start - 1, '(', ')') else {
            break;
        };
        let predicate: String = chars[predicate_start..predicate_end].iter().collect();
        at = predicate_end;
        if !test_only_predicate(&predicate) {
            continue;
        }
        // After `)]`: other attributes, then an optional visibility and `mod`.
        let mut j = predicate_end + 1;
        if chars.get(j) != Some(&']') {
            continue;
        }
        j += 1;
        loop {
            while chars.get(j).is_some_and(|c| c.is_whitespace()) {
                j += 1;
            }
            if chars.get(j) == Some(&'#') && chars.get(j + 1) == Some(&'[') {
                match matching(&chars, j + 1, '[', ']') {
                    Some(end) => j = end + 1,
                    None => break,
                }
            } else {
                break;
            }
        }
        let rest: String = chars[j..chars.len().min(j + 200)].iter().collect();
        let after_visibility = rest
            .strip_prefix("pub(crate) ")
            .or_else(|| rest.strip_prefix("pub(super) "))
            .or_else(|| rest.strip_prefix("pub "))
            .unwrap_or(&rest);
        let Some(after_mod) = after_visibility.strip_prefix("mod ") else {
            continue;
        };
        let name_length = after_mod
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .count();
        let after_name = after_mod[name_length..].trim_start();
        if !after_name.starts_with('{') {
            continue; // `mod name;`: a declaration, its file is scanned on its own.
        }
        let Some(open) = (j..chars.len()).find(|&k| chars[k] == '{') else {
            break;
        };
        let Some(close) = matching(&chars, open, '{', '}') else {
            break;
        };
        for c in &mut chars[found..=close] {
            if *c != '\n' {
                *c = ' ';
            }
        }
        at = close + 1;
    }
    chars.into_iter().collect()
}

fn find(chars: &[char], from: usize, needle: &str) -> Option<usize> {
    let needle: Vec<char> = needle.chars().collect();
    (from..chars.len().saturating_sub(needle.len() - 1)).find(|&i| chars[i..].starts_with(&needle))
}

/// The index of the bracket that closes the one at `open`.
fn matching(chars: &[char], open: usize, opening: char, closing: char) -> Option<usize> {
    let mut depth = 0usize;
    for (k, c) in chars.iter().enumerate().skip(open) {
        if *c == opening {
            depth += 1;
        } else if *c == closing {
            depth -= 1;
            if depth == 0 {
                return Some(k);
            }
        }
    }
    None
}

fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The 1-based lines of `code` where `word` is used as a whole identifier.
pub fn word_lines(code: &str, word: &str) -> Vec<usize> {
    code.lines()
        .enumerate()
        .filter(|(_, line)| {
            line.match_indices(word).any(|(at, _)| {
                let before = line[..at].chars().next_back();
                let after = line[at + word.len()..].chars().next();
                !before.is_some_and(is_identifier_char) && !after.is_some_and(is_identifier_char)
            })
        })
        .map(|(index, _)| index + 1)
        .collect()
}
