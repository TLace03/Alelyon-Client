//! Row D5: the read tools' bounds against the web's own
//! (the chat core's spec §16.2, `agent/limits.json`, recorded by
//! `tools/lattice_native_parity.py` from `workspaces/views.py`,
//! `agent/safety.py` and `workspaces/capture.py`). The read size, the binary
//! probe, the tree cap, and the checkpoint's snapshot size and new-file cap
//! (row E4) are held here; the diff cap is recorded for row E3.

use std::path::Path;

use lattice_core::changes::checkpoint::{MAX_NEW_FILES, MAX_SNAPSHOT_BYTES};
use lattice_core::git::runner::MAX_TREE_ENTRIES;
use lattice_core::tools::read::{BINARY_PROBE, MAX_FILE_BYTES};
use serde_json::Value;

#[test]
fn the_read_tools_bounds_are_the_webs() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("agent")
        .join("limits.json");
    let golden: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(golden["MAX_FILE_BYTES"].as_u64(), Some(MAX_FILE_BYTES));
    assert_eq!(golden["BINARY_PROBE"].as_u64(), Some(BINARY_PROBE as u64));
    assert_eq!(
        golden["MAX_TREE_ENTRIES"].as_u64(),
        Some(MAX_TREE_ENTRIES as u64)
    );
    assert_eq!(
        golden["MAX_SNAPSHOT_BYTES"].as_u64(),
        Some(MAX_SNAPSHOT_BYTES)
    );
    assert_eq!(golden["MAX_NEW_FILES"].as_u64(), Some(MAX_NEW_FILES as u64));
    assert!(
        golden["MAX_DIFF_LINES"].as_u64().is_some(),
        "MAX_DIFF_LINES is recorded"
    );
}
