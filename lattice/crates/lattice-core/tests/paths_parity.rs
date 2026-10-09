//! Row D3: the tool-path check against the web's own (the chat core's spec
//! §6.2 WP1–WP4). `paths/repo_path.json` is recorded by
//! `tools/lattice_native_parity.py` from `workspaces/paths.repo_path`: each
//! input with its result or its refusal sentence. `workspace::paths::repo_path`
//! is held to every case, sentence for sentence. The native additions (WP7,
//! WP9, WP10, WP11) are falsifier tests in `workspace/tests.rs`, not goldens.

use std::collections::BTreeSet;
use std::path::Path;

use lattice_core::workspace::paths::{MAX_PATH_CHARS, repo_path};
use serde_json::Value;

fn golden() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("paths")
        .join("repo_path.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

#[test]
fn repo_path_decides_as_the_web_does() {
    let golden = golden();
    assert_eq!(
        golden["max_path_chars"].as_u64(),
        Some(MAX_PATH_CHARS as u64)
    );
    let cases = golden["cases"].as_array().unwrap();
    assert!(cases.len() >= 250, "{} cases", cases.len());
    let mut sentences = BTreeSet::new();
    let mut accepted = 0;
    for case in cases {
        let path = case["path"].as_str().unwrap();
        match (repo_path(path), case.get("ok"), case.get("refused")) {
            (Ok(value), Some(ok), None) => {
                assert_eq!(Some(value), ok.as_str(), "{path:?}");
                accepted += 1;
            }
            (Err(error), None, Some(refused)) => {
                assert_eq!(Some(error.sentence()), refused.as_str(), "{path:?}");
                sentences.insert(error.sentence());
            }
            (native, ok, refused) => {
                panic!("{path:?}: native {native:?}, Python ok {ok:?} refused {refused:?}")
            }
        }
    }
    assert!(accepted >= 50, "{accepted} accepted");
    assert_eq!(
        sentences.len(),
        6,
        "every refusal is exercised: {sentences:?}"
    );
    for length in [1024, 1025] {
        assert!(
            cases
                .iter()
                .any(|case| case["path"].as_str().unwrap().chars().count() == length),
            "a path of {length} characters"
        );
    }
}
