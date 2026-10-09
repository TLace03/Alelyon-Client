//! The local model: which model the managed llama.cpp row runs, and where a
//! saved Ollama row pointed.
//!
//! Parity with the two functions of the Python runtime's `local_model`
//! the registry reads:
//!
//! - [`selected_model`]: the `model` field of `<globals>/analyst_model.json`
//!   (the file is read at most 8,192 bytes; a larger one is ignored), a GGUF
//!   file name in `~/.alelyon/models`, else no model at all. A name is text,
//!   trimmed, at most 4,096 characters and non-empty, or it does not count.
//!   There is no default and no environment override: `OLLAMA_MODEL` named an
//!   Ollama tag, and a default named a download (ADR-0041).
//! - [`base_url`]: where a saved Ollama row with no address of its own pointed,
//!   `OLLAMA_BASE_URL` (trailing `/` removed) or `http://localhost:11434`. As
//!   in Python, the variable is not trimmed: a value of spaces is an address of
//!   spaces (which no request can use), not a request for the default. Ollama
//!   is retired and nothing calls this address; it decides only what such a
//!   row is said to be (`registry::ollama_is_local`).
//!
//! - [`set_selected_model`] / [`record_selected_model`]: choosing one, as the
//!   Python model bar does: a name that is no GGUF file in the models folder is
//!   refused, and `{"model": name}` is written as `json.dumps(...,
//!   ensure_ascii=False)` writes it, to a `.tmp` beside the file and then put in
//!   its place, so a reader never sees half of it.
//!
//! Invariant: never fails; anything unreadable, oversize or malformed means
//! "no preference", and then no model is selected.

use std::io::Read;
use std::path::Path;

use serde_json::Value;

use crate::env::{self, Env};
use crate::py;
use crate::state::StateRoot;

/// No model is selected until one is chosen: a model is a file, never a default.
pub const DEFAULT_MODEL: &str = "";
pub const DEFAULT_BASE: &str = "http://localhost:11434";
pub const MAX_MODEL_NAME_CHARS: usize = 4_096;
pub const MAX_MODEL_PREF_BYTES: usize = 8_192;

/// `base_url()`.
pub fn base_url(env: &dyn Env) -> String {
    let value = env::text(env, "OLLAMA_BASE_URL")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_BASE.to_owned());
    value.trim_end_matches('/').to_owned()
}

/// `_normalise_model_name` for a value that is already text: trimmed, non-empty
/// and bounded, or the empty string.
fn normalise_model_name(value: &str) -> String {
    let value = py::strip(value);
    if value.is_empty() || value.chars().count() > MAX_MODEL_NAME_CHARS {
        return String::new();
    }
    value.to_owned()
}

/// The model the managed llama.cpp row runs: the stored preference, or none.
/// `env` is not read (no variable names the model any more); it is kept so
/// every registry rule takes the same arguments.
pub fn selected_model(_env: &dyn Env, state: &StateRoot) -> String {
    stored_model(state).unwrap_or_else(|| DEFAULT_MODEL.to_owned())
}

/// The model in `analyst_model.json`, when there is a usable one. The
/// managed llama.cpp server's selection reads the same file with no
/// environment override and no default (`llama::files::selected_model`).
pub(crate) fn stored_model(state: &StateRoot) -> Option<String> {
    let mut bytes = Vec::new();
    std::fs::File::open(state.globals.join("analyst_model.json"))
        .ok()?
        .take(MAX_MODEL_PREF_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > MAX_MODEL_PREF_BYTES {
        return None;
    }
    let raw = py::loads_bytes(&bytes)?;
    let name = match raw.get("model") {
        Some(Value::String(name)) => normalise_model_name(name),
        _ => String::new(),
    };
    (!name.is_empty()).then_some(name)
}

/// Why a model was not chosen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChooseError {
    /// The name is blank, or longer than [`MAX_MODEL_NAME_CHARS`].
    NoName,
    /// No GGUF file in the models folder has that name.
    NotFound(String),
    /// The preference could not be written.
    Write(String),
}

impl std::fmt::Display for ChooseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChooseError::NoName => f.write_str("no model was named"),
            ChooseError::NotFound(name) => write!(f, "no GGUF model named {name} is in the models folder"),
            ChooseError::Write(why) => write!(f, "the choice was not saved: {why}"),
        }
    }
}

/// `set_selected_model`: choose the model Local uses, refusing a name that is
/// no GGUF file in `models_dir` (`resolve_model`'s rule: a stem, exactly, or
/// the same with `.gguf`). Returns the name recorded, the file's stem.
pub fn set_selected_model(state: &StateRoot, models_dir: &Path, name: &str) -> Result<String, ChooseError> {
    let name = normalise_model_name(name);
    if name.is_empty() {
        return Err(ChooseError::NoName);
    }
    let found = crate::llama::files::resolve_model(&name, models_dir).map_err(|_| ChooseError::NotFound(name))?;
    record_selected_model(state, &found.name)?;
    Ok(found.name)
}

/// `record_selected_model`: persist `name` without checking for its file.
/// What is written is what [`stored_model`] reads back: a payload over
/// [`MAX_MODEL_PREF_BYTES`] is refused rather than written unreadable.
pub fn record_selected_model(state: &StateRoot, name: &str) -> Result<(), ChooseError> {
    let name = normalise_model_name(name);
    if name.is_empty() {
        return Err(ChooseError::NoName);
    }
    let payload = preference_bytes(&name);
    if payload.len() > MAX_MODEL_PREF_BYTES {
        return Err(ChooseError::Write(format!("its file would be over {MAX_MODEL_PREF_BYTES} bytes")));
    }
    let pref = state.globals.join("analyst_model.json");
    let tmp = pref.with_extension("tmp");
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all(&state.globals)?;
        std::fs::write(&tmp, &payload)?;
        // Python's own temporary name, replaced into place as `Path.replace` does: the shared-format replace
        // (`fsx::replace_shared`, as the registry's writer uses), which removes nothing but the old version.
        crate::fsx::replace_shared(&tmp, &pref)
    };
    write().map_err(|e| ChooseError::Write(format!("{}: {e}", pref.display())))
}

/// `json.dumps({"model": name}, ensure_ascii=False)` in UTF-8: the quote, the
/// backslash and the control characters escaped as Python escapes them, every
/// other character as it is.
fn preference_bytes(name: &str) -> Vec<u8> {
    let mut out = String::from("{\"model\": \"");
    for c in name.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push_str("\"}");
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::MapEnv;
    use crate::testkit::TempDir;

    /// `json.dumps({"model": name}, ensure_ascii=False).encode("utf-8")` as CPython 3.12 wrote it, in hex
    /// (scratch script, 2026-10-07).
    const PYTHON_PREFERENCES: [(&str, &str); 7] = [
        ("qwen3-8b", "7b226d6f64656c223a20227177656e332d3862227d"),
        ("a\"b\\c", "7b226d6f64656c223a2022615c22625c5c63227d"),
        ("mod\u{e9}le-\u{6a21}\u{578b}", "7b226d6f64656c223a20226d6f64c3a96c652de6a8a1e59e8b227d"),
        ("tab\there", "7b226d6f64656c223a20227461625c7468657265227d"),
        ("ctl\u{1}x\u{1f}", "7b226d6f64656c223a202263746c5c7530303031785c7530303166227d"),
        ("line\u{2028}sep", "7b226d6f64656c223a20226c696e65e280a8736570227d"),
        ("nl\nx\r\u{8}\u{c}", "7b226d6f64656c223a20226e6c5c6e785c725c625c66227d"),
    ];

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn a_choice_is_written_byte_for_byte_as_python_writes_it_and_reads_back() {
        for (name, python) in PYTHON_PREFERENCES {
            assert_eq!(hex(&preference_bytes(name)), python, "{name:?}");
            // Recorded, the name is trimmed first, as Python's `_normalise_model_name` trims it (`str.strip`
            // takes U+001F off the end too).
            let (_dir, state) = state_with(None);
            record_selected_model(&state, name).unwrap();
            let trimmed = normalise_model_name(name);
            let written = std::fs::read(state.globals.join("analyst_model.json")).unwrap();
            assert_eq!(written, preference_bytes(&trimmed), "{name:?}");
            assert_eq!(stored_model(&state), Some(trimmed), "what is written is what is read");
            assert!(!state.globals.join("analyst_model.tmp").exists(), "the temporary file is put in place");
        }
    }

    #[test]
    fn a_choice_must_be_a_gguf_file_in_the_models_folder() {
        let (dir, state) = state_with(Some(br#"{"model": "old"}"#));
        let models = dir.path().join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("qwen3-8b.gguf"), b"GGUF\x03\x00\x00\x00").unwrap();
        std::fs::write(models.join("not-a-model.gguf"), b"nope").unwrap();
        assert_eq!(set_selected_model(&state, &models, "  qwen3-8b.GGUF ").as_deref(), Ok("qwen3-8b"));
        assert_eq!(stored_model(&state).as_deref(), Some("qwen3-8b"), "a stem, as Python records it");
        for (name, why) in [
            ("missing", ChooseError::NotFound("missing".into())),
            ("not-a-model", ChooseError::NotFound("not-a-model".into())),
            ("   ", ChooseError::NoName),
        ] {
            assert_eq!(set_selected_model(&state, &models, name), Err(why), "{name:?}");
        }
        assert_eq!(stored_model(&state).as_deref(), Some("qwen3-8b"), "a refusal leaves the choice as it was");
        let long = "x".repeat(MAX_MODEL_NAME_CHARS + 1);
        assert_eq!(record_selected_model(&state, &long), Err(ChooseError::NoName));
    }

    fn state_with(pref: Option<&[u8]>) -> (TempDir, StateRoot) {
        let dir = TempDir::new("local-model");
        let state = StateRoot::at(dir.path());
        if let Some(pref) = pref {
            std::fs::create_dir_all(&state.globals).unwrap();
            std::fs::write(state.globals.join("analyst_model.json"), pref).unwrap();
        }
        (dir, state)
    }

    #[test]
    fn the_base_url_defaults_to_the_loopback_server_and_loses_trailing_slashes() {
        assert_eq!(base_url(&MapEnv::new()), "http://localhost:11434");
        assert_eq!(
            base_url(&MapEnv::new().with("OLLAMA_BASE_URL", "")),
            "http://localhost:11434"
        );
        assert_eq!(
            base_url(&MapEnv::new().with("OLLAMA_BASE_URL", "http://box:1234//")),
            "http://box:1234"
        );
        assert_eq!(
            base_url(&MapEnv::new().with("OLLAMA_BASE_URL", "  ")),
            "  ",
            "as in Python, the variable is not trimmed"
        );
    }

    #[test]
    fn the_environment_names_no_model_any_more() {
        // `OLLAMA_MODEL` once won over the file (ADR-0041 retired it).
        let (_dir, state) = state_with(Some(br#"{"model": "from-file"}"#));
        for value in ["  from-env  ", "   ", "m"] {
            let env = MapEnv::new().with("OLLAMA_MODEL", value);
            assert_eq!(selected_model(&env, &state), "from-file", "{value:?}");
        }
        let (_none, empty) = state_with(None);
        let env = MapEnv::new().with("OLLAMA_MODEL", "from-env");
        assert_eq!(selected_model(&env, &empty), DEFAULT_MODEL);
        assert_eq!(
            DEFAULT_MODEL, "",
            "no model is selected until one is chosen"
        );
    }

    #[test]
    fn the_stored_preference_is_read_and_anything_else_is_the_default() {
        let none = MapEnv::new();
        let (_a, state) = state_with(Some(br#"{"model": "  stored:7b "}"#));
        assert_eq!(selected_model(&none, &state), "stored:7b");
        for (label, pref) in [
            ("absent", None),
            ("not json", Some(&b"nope"[..])),
            ("a list", Some(&b"[1]"[..])),
            ("a number", Some(&b"{\"model\": 3}"[..])),
            ("a blank name", Some(&b"{\"model\": \"  \"}"[..])),
            ("no field", Some(&b"{}"[..])),
            ("invalid utf-8", Some(&b"\xff\xfe"[..])),
        ] {
            let (_dir, state) = state_with(pref);
            assert_eq!(selected_model(&none, &state), DEFAULT_MODEL, "{label}");
        }
        let mut big = br#"{"model": "big"}"#.to_vec();
        big.resize(MAX_MODEL_PREF_BYTES + 1, b' ');
        let (_b, oversize) = state_with(Some(&big));
        assert_eq!(
            selected_model(&none, &oversize),
            DEFAULT_MODEL,
            "over 8,192 bytes is ignored"
        );
        big.truncate(MAX_MODEL_PREF_BYTES);
        let (_c, exact) = state_with(Some(&big));
        assert_eq!(
            selected_model(&none, &exact),
            "big",
            "exactly the bound is read"
        );
    }

    #[test]
    fn a_stored_preference_with_a_byte_order_mark_is_read() {
        let mut pref = vec![0xef, 0xbb, 0xbf];
        pref.extend_from_slice(br#"{"model": "bom-model"}"#);
        let (_dir, state) = state_with(Some(&pref));
        assert_eq!(selected_model(&MapEnv::new(), &state), "bom-model");
    }
}
