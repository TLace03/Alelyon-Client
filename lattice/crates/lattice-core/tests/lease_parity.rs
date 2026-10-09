//! Row E5: the writer lease's derivation against Python's
//! (the chat core's spec §6.5, §16.2 `lease/derive.json`, recorded by
//! `tools/lattice_native_parity.py` from `worktree_cache.py`,
//! `repository_context.py` and `agent/session.py`). The pure arithmetic is
//! held here: the repository namespace over recorded identities (strong and
//! degraded), the checkout context id over recorded markers, the checkout
//! namespace, the lease path below a state home (tidied as `normpath`
//! tidies it), and the state home `selected_repository_state_root()` picks
//! for an environment. Paths in the golden are under `<root>`, which this
//! test replaces with a folder of its own choosing; nothing here reads the
//! disk. The reading of live identities is interop test I13.

use std::path::{Path, PathBuf};

use lattice_core::env::MapEnv;
use lattice_core::state::{Platform, StateRoot, resolve_with};
use lattice_core::workspace::lease::{
    CHECKOUT_LEASE_FILE, CHECKOUT_NAMESPACE_SCHEMA, GIT_POINTER_IDENTITY_BYTES, Incarnation,
    NamespaceKind, checkout_context_id, checkout_namespace, lease_path, marker_is_strong,
    namespace_key, repository_namespace, selected_repository_state_root,
};
use serde_json::Value;

fn golden() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join("lease")
        .join("derive.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn text(value: &Value) -> String {
    value.as_str().expect("a string").to_owned()
}

/// The folder `<root>` stands for: a drive path that is never opened.
fn root() -> PathBuf {
    PathBuf::from(r"C:\lease-golden")
}

/// `<root>/a/b` as a path under [`root`].
fn under_root(text: &str) -> PathBuf {
    let rest = text.strip_prefix("<root>").expect("a placeholder path");
    rest.split('/')
        .filter(|part| !part.is_empty())
        .fold(root(), |path, part| path.join(part))
}

/// A path under [`root`] as the golden writes it.
fn as_golden(path: &Path) -> String {
    let rest = path.strip_prefix(root()).expect("under the root");
    format!("<root>/{}", rest.to_string_lossy().replace('\\', "/"))
}

#[test]
fn the_constants_are_pythons() {
    let golden = golden();
    assert_eq!(text(&golden["lease_file"]), CHECKOUT_LEASE_FILE);
    assert_eq!(text(&golden["checkout_schema"]), CHECKOUT_NAMESPACE_SCHEMA);
    assert_eq!(
        golden["pointer_identity_bytes"].as_u64(),
        Some(GIT_POINTER_IDENTITY_BYTES)
    );
}

#[test]
fn repository_namespaces_are_pythons() {
    let golden = golden();
    let cases = golden["repository_namespaces"].as_array().unwrap();
    assert!(cases.len() >= 10);
    let mut degraded = 0;
    for case in cases {
        let kind = match case["kind"].as_str().unwrap() {
            "git-common-dir" => NamespaceKind::GitCommonDir,
            "selected-root" => NamespaceKind::SelectedRoot,
            other => panic!("{other}"),
        };
        let identity = case["identity"].as_object().map(|identity| Incarnation {
            device: text(&identity["device"]),
            inode: text(&identity["inode"]),
            birth_ns: text(&identity["birth"]),
        });
        let key = namespace_key(identity.as_ref(), case["normalized"].as_str().unwrap());
        degraded += usize::from(key.ends_with("\0metadata-unavailable"));
        assert_eq!(
            repository_namespace(kind, &key),
            text(&case["namespace"]),
            "{case}"
        );
    }
    assert!(degraded >= 4, "the degraded identities are covered");
}

#[test]
fn checkout_context_ids_and_namespaces_are_pythons() {
    let golden = golden();
    let contexts = golden["context_ids"].as_array().unwrap();
    assert!(contexts.len() >= 6);
    for case in contexts {
        let marker: Vec<String> = case["marker"]
            .as_array()
            .unwrap()
            .iter()
            .map(text)
            .collect();
        assert_eq!(
            marker_is_strong(&marker),
            case["strong"].as_bool().unwrap(),
            "{case}"
        );
        assert_eq!(
            checkout_context_id(case["anchor"].as_str().unwrap(), &marker),
            text(&case["context_id"]),
            "{case}"
        );
    }
    assert!(contexts.iter().any(|case| case["strong"] == true));
    assert!(contexts.iter().any(|case| case["strong"] == false));
    for case in golden["checkout_namespaces"].as_array().unwrap() {
        assert_eq!(
            checkout_namespace(
                case["repository_namespace"].as_str().unwrap(),
                case["context_id"].as_str().unwrap()
            ),
            text(&case["checkout_namespace"]),
            "{case}"
        );
    }
}

#[test]
fn lease_paths_are_pythons() {
    let golden = golden();
    for case in golden["lease_paths"].as_array().unwrap() {
        let home = under_root(case["state_home"].as_str().unwrap());
        let path = lease_path(
            &home,
            case["repository_namespace"].as_str().unwrap(),
            case["checkout_namespace"].as_str().unwrap(),
        );
        assert_eq!(as_golden(&path), text(&case["path"]), "{case}");
    }
}

/// `selected_repository_state_root()` over environments, on Windows rules,
/// for a source checkout (Python runs from one when it records them).
#[test]
fn state_homes_are_pythons() {
    let golden = golden();
    let cases = golden["state_homes"].as_array().unwrap();
    assert!(cases.len() >= 5);
    for case in cases {
        let mut env = MapEnv::new();
        for (name, value) in case["env"].as_object().unwrap() {
            let value = value.as_str().unwrap();
            if value.starts_with("<root>") {
                env.set(name, under_root(value).as_os_str());
            } else {
                env.set(name, value);
            }
        }
        let state = if case["env"].get("ALELYON_HOME").is_some() {
            resolve_with(&env, None, Platform::Windows)
        } else {
            StateRoot::at(root().join("checkout"))
        };
        let selected = selected_repository_state_root(&env, &state, Platform::Windows);
        assert_eq!(as_golden(&selected), text(&case["selected"]), "{case}");
    }
}
