//! The managed server's probe records (the chat core's spec §22 LR8, LR8′,
//! LR9; `llama_server._probe_key`, `recorded_grammar`).
//!
//! Python measures, on a running server, whether it restricts sampling to a
//! JSON Schema (`probe_grammar`), and records the answer in
//! `~/.alelyon/llama/grammar-probes.json` under a key naming the binary's
//! SHA-256 and the model file: `"<sha256>|<model path>|<size>|<mtime>"`
//! (`_probe_key`). A record counts only when it is literally `true`.
//!
//! LR8′ defines the tool-call probe that tool calling waits on
//! (`convo::probe`), and its record lives beside Python's, in the same shape:
//! [`TOOL_PROBES`], one key per binary and model, `true` when the probe
//! passed and `false` when it failed ([`tool_probe`], [`record_tool_probe`]).
//! Until a record says `true`, tool calling is not assumed.
//!
//! The core never writes `grammar-probes.json`. It writes `tool-probes.json`
//! only through `fsx`'s atomic write, keeping every other key, and never over
//! a file it cannot read as an object; it never deletes either.

use std::path::Path;
use std::time::UNIX_EPOCH;

use super::files::LlamaPaths;
use crate::chat::pyjson::{self, PyValue};
use crate::sha::sha256_file;

/// Python's grammar probe records.
pub const GRAMMAR_PROBES: &str = "grammar-probes.json";
/// The tool-call probe's records (LR8′).
pub const TOOL_PROBES: &str = "tool-probes.json";

/// The binary's SHA-256, read from the file now (the probe key's first part).
pub fn binary_sha256(binary: &Path) -> Option<String> {
    sha256_file(binary).ok().map(|(sha, _)| sha)
}

/// `_probe_key`: `"<binary sha256>|<str(model path)>|<size>|<int(mtime)>"`.
/// On Windows the path is written as `str(WindowsPath)` writes it
/// ([`windows_path_text`]), so a models folder named with `.` parts or
/// doubled separators gives Python's key (spec 22.6 P2).
pub fn probe_key(binary_sha256: &str, model: &Path) -> Option<String> {
    let meta = std::fs::metadata(model).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs();
    let mut path = model.to_string_lossy().into_owned();
    if cfg!(windows) {
        path = windows_path_text(&path);
    }
    Some(format!("{binary_sha256}|{path}|{}|{mtime}", meta.len()))
}

/// `str(PureWindowsPath(text))` as CPython 3.12 writes it: `/` becomes `\`,
/// the drive and root are split off as `ntpath.splitroot` splits them, empty
/// and `.` parts are dropped (`..` stays), and the parts are joined with `\`;
/// a path with nothing left is `.`. Pinned by the `probe_key_paths` cases of
/// `llama/llama_server.json`.
pub fn windows_path_text(text: &str) -> String {
    let path = text.replace('/', "\\");
    let (drive, mut root, rest) = split_root(&path);
    if root.is_empty() && drive.starts_with('\\') && !drive.ends_with('\\') {
        // pathlib's `_parse_path`: a whole UNC drive (`\\server\share`, or
        // `\\?\UNC\server\share`) has a root even when none was written.
        let parts: Vec<&str> = drive.split('\\').collect();
        if (parts.len() == 4 && !"?.".contains(parts[2])) || parts.len() == 6 {
            root = "\\";
        }
    }
    let mut tail: Vec<&str> = rest
        .split('\\')
        .filter(|part| !part.is_empty() && *part != ".")
        .collect();
    if !drive.is_empty() || !root.is_empty() {
        return format!("{drive}{root}{}", tail.join("\\"));
    }
    // A first part that would read as a drive keeps a `.` before it.
    if tail
        .first()
        .is_some_and(|first| first.chars().nth(1) == Some(':'))
    {
        tail.insert(0, ".");
    }
    if tail.is_empty() {
        ".".to_owned()
    } else {
        tail.join("\\")
    }
}

/// `ntpath.splitroot` (3.12) over a path whose separators are all `\`:
/// drive, root and the rest.
fn split_root(path: &str) -> (&str, &str, &str) {
    if let Some(after) = path.strip_prefix('\\') {
        if !after.starts_with('\\') {
            return ("", "\\", after);
        }
        // A UNC or device drive: `\\server\share`, `\\?\UNC\server\share`.
        let start = if path
            .get(..8)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("\\\\?\\UNC\\"))
        {
            8
        } else {
            2
        };
        let Some(first) = path[start..].find('\\').map(|at| at + start) else {
            return (path, "", "");
        };
        let Some(second) = path[first + 1..].find('\\').map(|at| at + first + 1) else {
            return (path, "", "");
        };
        return (
            &path[..second],
            &path[second..second + 1],
            &path[second + 1..],
        );
    }
    let mut chars = path.char_indices();
    if let (Some(_), Some((at, ':'))) = (chars.next(), chars.next()) {
        let end = at + 1;
        if path[end..].starts_with('\\') {
            return (&path[..end], &path[end..end + 1], &path[end + 1..]);
        }
        return (&path[..end], "", &path[end..]);
    }
    ("", "", path)
}

/// Is `key` recorded as literally `true` in `<llama dir>/<file>`? A file that
/// is missing, unreadable, not JSON or not an object records nothing.
pub fn recorded(paths: &LlamaPaths, file: &str, key: &str) -> bool {
    matches!(read_records(paths, file), Records::Object(records) if records.get(key) == Some(&PyValue::Bool(true)))
}

/// What the tool-call probe's record says of `key`: `Some(true)` it passed,
/// `Some(false)` it failed, `None` no probe is recorded for it (or the file
/// cannot be read, or the value is not a boolean).
pub fn tool_probe(paths: &LlamaPaths, key: &str) -> Option<bool> {
    match read_records(paths, TOOL_PROBES) {
        Records::Object(records) => match records.get(key) {
            Some(PyValue::Bool(passed)) => Some(*passed),
            _ => None,
        },
        Records::Missing | Records::Unreadable => None,
    }
}

/// Record the tool-call probe's answer for `key` (LR8′): the file's other keys
/// kept in their order, written through `fsx`'s atomic write. A file that
/// exists but cannot be read as an object is never written over: the answer
/// is then refused, with the sentence why.
pub fn record_tool_probe(paths: &LlamaPaths, key: &str, passed: bool) -> Result<(), String> {
    let mut pairs = match read_records(paths, TOOL_PROBES) {
        Records::Missing => Vec::new(),
        Records::Object(PyValue::Object(pairs)) => pairs,
        Records::Object(_) | Records::Unreadable => {
            return Err(format!(
                "{} is not a record Lattice can read, so it was left as it is.",
                TOOL_PROBES
            ));
        }
    };
    match pairs.iter_mut().find(|(have, _)| have == key) {
        Some((_, value)) => *value = PyValue::Bool(passed),
        None => pairs.push((key.to_owned(), PyValue::Bool(passed))),
    }
    std::fs::create_dir_all(&paths.llama_dir)
        .map_err(|_| "Lattice could not make the llama folder.".to_owned())?;
    let text = pyjson::dumps_indent1(&PyValue::Object(pairs)) + "\n";
    crate::fsx::atomic_write(&paths.llama_dir.join(TOOL_PROBES), text.as_bytes())
        .map_err(|_| format!("Lattice could not write {TOOL_PROBES}."))
}

enum Records {
    Missing,
    /// It exists and is not a JSON object Lattice can read.
    Unreadable,
    Object(PyValue),
}

fn read_records(paths: &LlamaPaths, file: &str) -> Records {
    let bytes = match std::fs::read(paths.llama_dir.join(file)) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Records::Missing,
        Err(_) => return Records::Unreadable,
    };
    let Ok(text) = String::from_utf8(bytes) else {
        return Records::Unreadable;
    };
    match pyjson::loads(&text) {
        Ok(records) if records.is_object() => Records::Object(records),
        _ => Records::Unreadable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::TempDir;

    #[test]
    fn a_record_counts_only_when_it_is_literally_true_for_this_key() {
        let dir = TempDir::new("llama-probes");
        let paths = LlamaPaths {
            home: dir.path().to_path_buf(),
            llama_dir: dir.path().join("llama"),
            models_dir: dir.path().join("models"),
            binary: dir.path().join("llama").join("llama-server.exe"),
            binary_from_env: false,
        };
        std::fs::create_dir_all(&paths.llama_dir).unwrap();
        std::fs::create_dir_all(&paths.models_dir).unwrap();
        let model = paths.models_dir.join("m.gguf");
        std::fs::write(&model, b"GGUF....").unwrap();
        let key = probe_key("ab", &model).unwrap();
        assert!(key.starts_with("ab|") && key.contains("|8|"), "{key}");
        assert!(!recorded(&paths, GRAMMAR_PROBES, &key), "no file");
        let write =
            |text: String| std::fs::write(paths.llama_dir.join(GRAMMAR_PROBES), text).unwrap();
        let json_key = pyjson::json_string(&key);
        write(format!("{{{json_key}: true}}"));
        assert!(recorded(&paths, GRAMMAR_PROBES, &key));
        assert!(
            !recorded(&paths, TOOL_PROBES, &key),
            "the other record says nothing"
        );
        for value in ["false", "1", "\"true\"", "null"] {
            write(format!("{{{json_key}: {value}}}"));
            assert!(!recorded(&paths, GRAMMAR_PROBES, &key), "{value}");
        }
        write("[true]".into());
        assert!(!recorded(&paths, GRAMMAR_PROBES, &key));
    }

    /// LR8′: the tool-call probe's record says passed, failed or nothing; a
    /// write keeps every other key and replaces only its own; a file that
    /// cannot be read is never written over.
    #[test]
    fn a_tool_probe_is_recorded_beside_the_others_and_never_over_an_unreadable_file() {
        let dir = TempDir::new("llama-tool-probes");
        let paths = LlamaPaths {
            home: dir.path().to_path_buf(),
            llama_dir: dir.path().join("llama"),
            models_dir: dir.path().join("models"),
            binary: dir.path().join("llama").join("llama-server.exe"),
            binary_from_env: false,
        };
        assert_eq!(tool_probe(&paths, "k1"), None, "no file, no folder");
        record_tool_probe(&paths, "k1", false).unwrap();
        assert_eq!(tool_probe(&paths, "k1"), Some(false));
        assert!(!recorded(&paths, TOOL_PROBES, "k1"), "false is not a pass");
        record_tool_probe(&paths, "k2", true).unwrap();
        record_tool_probe(&paths, "k1", true).unwrap();
        assert_eq!(tool_probe(&paths, "k1"), Some(true));
        assert!(recorded(&paths, TOOL_PROBES, "k2"));
        let file = paths.llama_dir.join(TOOL_PROBES);
        let text = std::fs::read_to_string(&file).unwrap();
        assert!(
            text.find("\"k1\"").unwrap() < text.find("\"k2\"").unwrap(),
            "{text}"
        );
        std::fs::write(&file, "{\"other\": 7, \"k3\": \"yes\"}").unwrap();
        assert_eq!(tool_probe(&paths, "k3"), None, "not a boolean");
        record_tool_probe(&paths, "k3", true).unwrap();
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "{\n \"other\": 7,\n \"k3\": true\n}\n"
        );
        for broken in ["[true]", "{not json", ""] {
            std::fs::write(&file, broken).unwrap();
            assert_eq!(tool_probe(&paths, "k3"), None);
            assert!(record_tool_probe(&paths, "k3", true).is_err(), "{broken:?}");
            assert_eq!(
                std::fs::read_to_string(&file).unwrap(),
                broken,
                "left as it was"
            );
        }
    }
}
