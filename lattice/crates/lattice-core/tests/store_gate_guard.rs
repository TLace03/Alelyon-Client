//! The gate on shared writes cannot be moved or switched elsewhere
//! (the chat core's spec §5.9 and §5.10; row C3).
//!
//! - `SHARED_WRITES` is defined in exactly one place,
//!   `lattice-core/src/chat/store_gate.rs`, as a `bool` constant; no other
//!   source in the workspace defines an item of that name, so no second switch
//!   can shadow it.
//! - The transcript store in memory is declared only under
//!   `cfg(any(test, feature = "dev-host"))`, and `dev-host` is not a default
//!   feature, so no shipped build contains it.
//! - No shipped source outside `chat/memory.rs` names `MemoryTranscriptStore`
//!   (row G4): the production chat path is the shared store's.
//! - The development host `lattice-chat-host` (row F1) is lattice-core's one
//!   example that needs `dev-host`, and no manifest in the workspace turns
//!   `dev-host` on, so a default or release build holds neither it nor the
//!   memory store.
//!
//! Test modules are not shipped and are skipped (`scanner`). The guard is shown
//! to catch a planted fixture before it is trusted.

mod scanner;

use scanner::{Source, code_only, sources, without_test_modules, word_lines};

const GATE_FILE: &str = "lattice-core/src/chat/store_gate.rs";
const DEFINITION: &str = "pub(crate) const SHARED_WRITES: bool =";

/// Lines of `source` that define an item named `SHARED_WRITES`.
fn definitions(source: &Source) -> Vec<usize> {
    word_lines(&source.code, "SHARED_WRITES")
        .into_iter()
        .filter(|line| {
            let text = source.line(*line);
            [
                "const ", "static ", "fn ", "let ", "mod ", "struct ", "enum ", "type ",
            ]
            .iter()
            .any(|keyword| text.contains(&format!("{keyword}SHARED_WRITES")))
        })
        .collect()
}

fn violations(sources: &[Source]) -> Vec<String> {
    let mut found = Vec::new();
    let mut in_gate = 0;
    for source in sources {
        for line in definitions(source) {
            let text = source.line(line);
            if source.relative == GATE_FILE && text.starts_with(DEFINITION) {
                in_gate += 1;
            } else {
                found.push(format!("{}:{line}: {text}", source.relative));
            }
        }
    }
    if in_gate != 1 {
        found.push(format!("{GATE_FILE} defines SHARED_WRITES {in_gate} times"));
    }
    found
}

fn fixture(relative: &str, text: &str) -> Source {
    Source {
        relative: relative.to_owned(),
        original: text.to_owned(),
        code: without_test_modules(&code_only(text)),
    }
}

#[test]
fn the_guard_catches_a_second_switch() {
    let gate = || fixture(GATE_FILE, "pub(crate) const SHARED_WRITES: bool = false;\n");
    assert!(violations(&[gate()]).is_empty());
    // A planted switch elsewhere, the mutant this guard exists for.
    let planted = fixture(
        "lattice-core/src/chat/store.rs",
        "// SHARED_WRITES is the gate's\nconst SHARED_WRITES: bool = true;\n",
    );
    let found = violations(&[gate(), planted]);
    println!("planted fixture: {found:?}");
    assert_eq!(found.len(), 1, "{found:?}");
    // A gate that is not a bool constant in its file, or is defined twice.
    let switched = fixture(
        GATE_FILE,
        "pub static SHARED_WRITES: AtomicBool = AtomicBool::new(false);\n",
    );
    assert_eq!(violations(&[switched]).len(), 2);
    let twice = fixture(
        GATE_FILE,
        "pub(crate) const SHARED_WRITES: bool = false;\npub(crate) const SHARED_WRITES: bool = true;\n",
    );
    assert_eq!(violations(&[twice]).len(), 1);
    // A use is not a definition, and a test module's shadow is not shipped.
    let used = fixture(
        "lattice-core/src/chat/x.rs",
        "fn f() -> bool { store_gate::SHARED_WRITES }\n#[cfg(test)]\nmod tests {\n    const SHARED_WRITES: bool = true;\n}\n",
    );
    assert!(definitions(&used).is_empty());
}

#[test]
fn shared_writes_is_defined_only_in_the_gate_file() {
    let all = sources(&[
        "lattice-protocol",
        "lattice-agents",
        "lattice-core",
        "lattice-app",
        "lattice-sys",
    ]);
    assert!(all.iter().any(|source| source.relative == GATE_FILE));
    let found = violations(&all);
    assert!(
        found.is_empty(),
        "SHARED_WRITES outside its gate:\n{}",
        found.join("\n")
    );
}

#[test]
fn the_memory_store_is_only_in_tests_and_the_dev_host() {
    let chat = sources(&["lattice-core"])
        .into_iter()
        .find(|source| source.relative == "lattice-core/src/chat/mod.rs")
        .expect("chat/mod.rs");
    let declarations: Vec<usize> = word_lines(&chat.code, "memory")
        .into_iter()
        .filter(|line| chat.line(*line).contains("mod memory"))
        .collect();
    assert_eq!(declarations.len(), 1, "{declarations:?}");
    let line = declarations[0];
    assert_eq!(
        chat.line(line - 1),
        "#[cfg(any(test, feature = \"dev-host\"))]"
    );
    let manifest = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
    )
    .unwrap();
    assert!(
        manifest.contains("\ndev-host = []"),
        "dev-host adds nothing"
    );
    assert!(
        !manifest.contains("default ="),
        "no default features: dev-host stays off"
    );
}

const MEMORY_FILE: &str = "lattice-core/src/chat/memory.rs";

/// Shipped code that names the memory store, outside its own (gated) file.
fn memory_store_uses(sources: &[Source]) -> Vec<String> {
    sources
        .iter()
        .filter(|source| source.relative != MEMORY_FILE)
        .flat_map(|source| {
            word_lines(&source.code, "MemoryTranscriptStore")
                .into_iter()
                .map(move |line| format!("{}:{line}: {}", source.relative, source.line(line)))
        })
        .collect()
}

/// Row G4: the production chat path records through the shared store. No
/// shipped source names `MemoryTranscriptStore` outside `chat/memory.rs`
/// (which only tests and `dev-host` compile, above): not in `ChatConfig::new`,
/// not anywhere a shipped build could construct it. Tests and test modules
/// may, and are skipped.
#[test]
fn the_guard_catches_a_shipped_memory_store() {
    let planted = fixture(
        "lattice-core/src/chat/mod.rs",
        "// MemoryTranscriptStore is for tests\nfn new() { let store = Arc::new(MemoryTranscriptStore::new(ids, clock)); }\n#[cfg(test)]\nmod tests {\n    use super::memory::MemoryTranscriptStore;\n}\n",
    );
    let found = memory_store_uses(&[planted]);
    println!("planted fixture: {found:?}");
    assert_eq!(found.len(), 1, "{found:?}");
    let own = fixture(MEMORY_FILE, "pub struct MemoryTranscriptStore {}\n");
    assert!(memory_store_uses(&[own]).is_empty());
}

#[test]
fn no_shipped_source_constructs_the_memory_store() {
    let all = sources(&[
        "lattice-protocol",
        "lattice-agents",
        "lattice-core",
        "lattice-app",
        "lattice-sys",
    ]);
    assert!(all.iter().any(|source| source.relative == MEMORY_FILE));
    let found = memory_store_uses(&all);
    assert!(
        found.is_empty(),
        "the memory store in shipped code:\n{}",
        found.join("\n")
    );
}

/// Row F1: the development host is never part of a shipped build. Its one
/// target is an example of this crate that needs `dev-host`
/// (`required-features`), so `cargo build`, `cargo test` and a release build
/// leave it out; and no manifest in the workspace turns `dev-host` on (Cargo
/// unifies features across a workspace, so one `features = ["dev-host"]` on a
/// `lattice-core` dependency would put the memory store in every build).
/// `manifests` are `(crate, Cargo.toml text)`; lattice-core's is the one that
/// defines the feature and the host.
fn dev_host_violations(manifests: &[(&str, String)]) -> Vec<String> {
    let mut found = Vec::new();
    let mut hosts = 0;
    for (name, text) in manifests {
        // The manifest's sections, each with its header.
        let mut sections: Vec<(String, Vec<&str>)> = vec![(String::new(), Vec::new())];
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                sections.push((trimmed.to_owned(), Vec::new()));
            } else if !trimmed.starts_with('#') && !trimmed.is_empty() {
                sections.last_mut().unwrap().1.push(trimmed);
            }
        }
        for (header, lines) in &sections {
            let names_host = lines
                .iter()
                .any(|line| line.replace(' ', "") == "name=\"lattice-chat-host\"");
            if names_host {
                hosts += 1;
                let gated = lines
                    .iter()
                    .any(|line| line.replace(' ', "") == "required-features=[\"dev-host\"]");
                if *name != "lattice-core" || header != "[[example]]" || !gated {
                    found.push(format!(
                        "{name}: lattice-chat-host in {header} without required-features = [\"dev-host\"]"
                    ));
                }
                continue;
            }
            for line in lines {
                if !line.contains("dev-host") {
                    continue;
                }
                let own_definition =
                    *name == "lattice-core" && header == "[features]" && *line == "dev-host = []";
                if !own_definition {
                    found.push(format!("{name}: {header} {line}"));
                }
            }
        }
    }
    if hosts != 1 {
        found.push(format!("lattice-chat-host is declared {hosts} times"));
    }
    found
}

fn workspace_manifests() -> Vec<(&'static str, String)> {
    let core = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates = core.parent().unwrap();
    let mut out = vec![(
        "workspace",
        std::fs::read_to_string(crates.parent().unwrap().join("Cargo.toml")).unwrap(),
    )];
    for name in [
        "lattice-protocol",
        "lattice-agents",
        "lattice-core",
        "lattice-app",
        "lattice-sys",
    ] {
        out.push((
            name,
            std::fs::read_to_string(crates.join(name).join("Cargo.toml")).unwrap(),
        ));
    }
    out
}

#[test]
fn the_guard_catches_a_dev_host_in_a_default_build() {
    let core = |example: &str| {
        (
            "lattice-core",
            format!("[features]\ndev-host = []\n\n{example}\n[dependencies]\nserde = \"1\"\n"),
        )
    };
    let gated = "[[example]]\nname = \"lattice-chat-host\"\npath = \"examples/h.rs\"\nrequired-features = [\"dev-host\"]\n";
    assert!(dev_host_violations(&[core(gated)]).is_empty());
    // The host without its feature gate, or as a bin of another crate.
    let ungated = "[[example]]\nname = \"lattice-chat-host\"\npath = \"examples/h.rs\"\n";
    let found = dev_host_violations(&[core(ungated)]);
    println!("planted fixture: {found:?}");
    assert_eq!(found.len(), 1, "{found:?}");
    let elsewhere = (
        "lattice-app",
        "[[bin]]\nname = \"lattice-chat-host\"\nrequired-features = [\"dev-host\"]\n".to_owned(),
    );
    assert_eq!(
        dev_host_violations(&[core(""), elsewhere]).len(),
        1,
        "a host outside lattice-core's examples"
    );
    // A crate turning the feature on, or a default that holds it.
    let enables = (
        "lattice-app",
        "[dependencies]\nlattice-core = { workspace = true, features = [\"dev-host\"] }\n"
            .to_owned(),
    );
    assert_eq!(dev_host_violations(&[core(gated), enables]).len(), 1);
    let default = (
        "lattice-core",
        format!("[features]\ndefault = [\"dev-host\"]\ndev-host = []\n\n{gated}"),
    );
    assert_eq!(dev_host_violations(&[default]).len(), 1);
}

#[test]
fn the_dev_host_is_never_in_a_default_build() {
    let found = dev_host_violations(&workspace_manifests());
    assert!(
        found.is_empty(),
        "the development host could reach a shipped build:\n{}",
        found.join("\n")
    );
}
