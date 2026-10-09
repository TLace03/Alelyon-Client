//! The Rust port against goldens recorded from the real Python code.
//!
//! `tools/lattice_native_parity.py` runs `model_config`, `keys`, `local_model`
//! and `paths` from the Python Lattice, and an arithmetic reference built on
//! Python's own numbers, and writes what they decide to `tests/parity/`. These
//! tests read those files and require this crate to decide the same, exactly:
//! the merged endpoint list of every registry file, each endpoint's locality,
//! readiness and status, `is_local_url` over a corpus of addresses, the parsing
//! of env files and the order in which keys are found, the local model's address
//! and name, the state directory rules, and what the calculator answers.
//!
//! The Python side is the reference. If a test here fails after the goldens were
//! regenerated, the Python rule changed and this crate has to follow it; if a
//! golden is edited by hand, `tests/frontend/test_lattice_native_parity.py`
//! fails, because it regenerates them from Python and compares.
//!
//! Every file this test writes goes in a temporary directory, and the environment
//! is a `MapEnv`: nothing here reads the real environment, `globals/` or a key.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lattice_core::calc::{self, CalcError};
use lattice_core::env::MapEnv;
use lattice_core::keys::{self, KeyStore};
use lattice_core::local_model;
use lattice_core::registry::{self, LoadReport, ModelEndpoint};
use lattice_core::state::{self, Platform, StateRoot};
use serde_json::Value;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!("lattice-core-parity-{tag}-{}", unique()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn unique() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}-{:?}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

fn parity_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
}

fn read(relative: &str) -> Value {
    let path = parity_dir().join(relative);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn str_of<'a>(value: &'a Value, key: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} in {value}"))
}

/// A registry file's bytes as Python wrote them for a case.
fn case_bytes(input: &Value) -> Option<Vec<u8>> {
    if input.get("missing").is_some() {
        return None;
    }
    if let Some(kind) = input.get("generate").and_then(Value::as_str) {
        return match kind {
            "oversize" | "exactly_max" => {
                let mut bytes = br#"{"version": 1, "endpoints": []}"#.to_vec();
                let size = registry::MAX_CONFIG_BYTES + usize::from(kind == "oversize");
                bytes.resize(size, b' ');
                Some(bytes)
            }
            "many_rows" => {
                let rows = vec![r#"{"id": "openai", "note": "x"}"#; 10_005].join(", ");
                Some(format!(r#"{{"version": 1, "endpoints": [{rows}]}}"#).into_bytes())
            }
            _ => None,
        };
    }
    Some(encode(
        str_of(input, "text"),
        input
            .get("encoding")
            .and_then(Value::as_str)
            .unwrap_or("utf-8"),
    ))
}

/// Python's `str.encode` for the three encodings the goldens use.
fn encode(text: &str, encoding: &str) -> Vec<u8> {
    match encoding {
        "utf-8" => text.as_bytes().to_vec(),
        "utf-8-sig" => [&[0xef, 0xbb, 0xbf][..], text.as_bytes()].concat(),
        "utf-16" => {
            let mut bytes = vec![0xff, 0xfe];
            bytes.extend(text.encode_utf16().flat_map(u16::to_le_bytes));
            bytes
        }
        other => panic!("unknown encoding {other}"),
    }
}

fn keys_for(env_file: &str, scratch: &Scratch) -> KeyStore {
    let file = scratch.path().join("fixture.env");
    std::fs::write(&file, env_file).unwrap();
    let env = MapEnv::new().with("FAM_ENV_PATH", file.as_os_str());
    KeyStore::new(
        Arc::new(env),
        &StateRoot::at(scratch.path().join("empty-root")),
    )
}

fn kind_name(endpoint: &ModelEndpoint) -> &'static str {
    endpoint.kind.as_str()
}

/// Assert that `endpoint` is what the golden `row` says, field by field.
fn assert_row(endpoint: &ModelEndpoint, row: &Value, keys: &KeyStore, context: &str) {
    let check = |field: &str, got: Value| {
        assert_eq!(got, row[field], "{context}: {} / {field}", endpoint.id)
    };
    check("id", endpoint.id.clone().into());
    check("label", endpoint.label.clone().into());
    check("kind", kind_name(endpoint).into());
    check("base_url", endpoint.base_url.clone().into());
    check("model", endpoint.model.clone().into());
    check("api_key_name", endpoint.api_key_name.clone().into());
    check("enabled", endpoint.enabled.into());
    check("builtin", endpoint.builtin.into());
    check("note", endpoint.note.clone().into());
    // The goldens were recorded with no `OLLAMA_*` variable set.
    check("local", endpoint.local(&MapEnv::new()).into());
    check("needs_key", endpoint.needs_key().into());
    check("ready", endpoint.ready(keys).into());
    check("status", endpoint.status(keys).into());
}

fn catalogue() -> (Value, Vec<Value>) {
    let golden = read("registry/builtins.json");
    let rows = golden["endpoints"].as_array().unwrap().clone();
    (golden, rows)
}

#[test]
fn the_builtin_catalogue_matches_pythons() {
    let (golden, rows) = catalogue();
    let scratch = Scratch::new("builtins");
    let keys = keys_for(str_of(&golden, "env_file"), &scratch);
    let ours = registry::builtins();
    assert_eq!(ours.len(), rows.len());
    for (endpoint, row) in ours.iter().zip(&rows) {
        assert_row(endpoint, row, &keys, "builtins");
    }
}

fn assert_report(report: &LoadReport, golden: &Value, catalogue: &[Value], keys: &KeyStore) {
    let name = str_of(golden, "name");
    assert_eq!(
        report.complete(),
        golden["complete"].as_bool().unwrap(),
        "{name}: complete"
    );
    let issues: Vec<&str> = report.issues.iter().map(|issue| issue.as_str()).collect();
    let expected: Vec<&str> = golden["issues"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(issues, expected, "{name}: issue classes");

    let order: Vec<&str> = golden["order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let ids: Vec<&str> = report.endpoints.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, order, "{name}: the merged list's order");
    let changed: BTreeMap<&str, &Value> = golden["endpoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| (str_of(row, "id"), row))
        .collect();
    for endpoint in &report.endpoints {
        let row = changed
            .get(endpoint.id.as_str())
            .copied()
            .or_else(|| {
                catalogue
                    .iter()
                    .find(|row| str_of(row, "id") == endpoint.id)
            })
            .unwrap_or_else(|| panic!("{name}: no golden row for {}", endpoint.id));
        assert_row(endpoint, row, keys, name);
    }
}

#[test]
fn every_registry_case_merges_and_judges_as_python_does() {
    let (builtins, catalogue) = catalogue();
    let scratch = Scratch::new("registry-cases");
    let keys = keys_for(str_of(&builtins, "env_file"), &scratch);
    let mut names: Vec<String> = std::fs::read_dir(parity_dir().join("registry"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".json"))
        .collect();
    names.sort();
    let mut cases = 0;
    for name in names {
        let golden = read(&format!("registry/{name}"));
        if golden["kind"] != "case" {
            continue;
        }
        cases += 1;
        let input = &golden["input"];
        let target = scratch.path().join(format!("case-{cases}"));
        std::fs::create_dir_all(&target).unwrap();
        let path = if input.get("missing").is_some() {
            target
                .join("no_such_directory")
                .join("model_endpoints.json")
        } else if input.get("generate").and_then(Value::as_str) == Some("directory") {
            target.clone()
        } else {
            let path = target.join("model_endpoints.json");
            std::fs::write(&path, case_bytes(input).expect("bytes for the case")).unwrap();
            path
        };
        let from_path = registry::load_with_report(&path);
        assert_report(&from_path, &golden, &catalogue, &keys);
        if let Some(bytes) = case_bytes(input)
            && input.get("generate").and_then(Value::as_str) != Some("directory")
        {
            assert_report(
                &registry::load_bytes_with_report(&bytes),
                &golden,
                &catalogue,
                &keys,
            );
        }
    }
    assert!(cases >= 60, "the case files were found ({cases})");
}

#[test]
fn is_local_url_agrees_over_the_whole_corpus() {
    let golden = read("registry/is_local_url.json");
    let urls = golden["urls"].as_array().unwrap();
    assert!(urls.len() >= 60);
    let mut trues = 0;
    for entry in urls {
        let url = str_of(entry, "url");
        let expected = entry["local"].as_bool().unwrap();
        assert_eq!(registry::is_local_url(url), expected, "{url:?}");
        trues += usize::from(expected);
    }
    assert!(
        trues > 10 && trues < urls.len() - 10,
        "the corpus has both answers ({trues} of {})",
        urls.len()
    );
}

/// An Ollama endpoint is local when the address its client will call is, and the
/// model it asks for is not one the daemon serves remotely: Python's
/// `ModelEndpoint.local` and the `Provider.local` built for it, held to over
/// addresses in `OLLAMA_BASE_URL`, `*-cloud` models, and endpoints with and
/// without an address of their own. The client's `local` flag (which also turns
/// proxies off) is this crate's `Provider.local`.
#[test]
fn an_ollama_endpoint_is_local_exactly_when_python_says_its_provider_is() {
    let golden = read("registry/ollama_locality.json");
    let cases = golden["cases"].as_array().unwrap();
    assert!(
        cases.len() >= 40,
        "the case file was found ({})",
        cases.len()
    );
    let scratch = Scratch::new("ollama-locality");
    let state = StateRoot::at(scratch.path().join("empty-root"));
    let keys = KeyStore::new(Arc::new(MapEnv::new()), &state);
    let (mut locals, mut remotes) = (0, 0);
    for case in cases {
        let name = str_of(case, "name");
        let mut env = MapEnv::new();
        for (variable, key) in [
            ("OLLAMA_BASE_URL", "env_base_url"),
            ("OLLAMA_MODEL", "env_model"),
        ] {
            if let Some(value) = case[key].as_str() {
                env.set(variable, value);
            }
        }
        // Every case is a saved Ollama row: there is no built-in one any more
        // (ADR-0041), and `OLLAMA_MODEL` projects onto nothing.
        let endpoint = match case["endpoint"].as_object() {
            None => panic!("{name}: a case without a row"),
            Some(fields) => {
                let row = serde_json::json!({
                    "version": 1,
                    "endpoints": [{
                        "id": "o", "label": "O", "kind": "ollama",
                        "base_url": fields["base_url"], "model": fields["model"],
                    }],
                });
                let report = registry::load_bytes_with_report(row.to_string().as_bytes());
                assert!(report.complete(), "{name}: {:?}", report.issues);
                report.endpoints.into_iter().find(|e| e.id == "o").unwrap()
            }
        };
        assert_eq!(
            endpoint.model,
            str_of(case, "model"),
            "{name}: the model the client asks for"
        );
        let expected = case["local"].as_bool().unwrap();
        assert_eq!(endpoint.local(&env), expected, "{name}: local");
        assert_eq!(
            case["provider_local"].as_bool().unwrap(),
            expected,
            "{name}: Python's endpoint and its provider agree"
        );
        let config = lattice_core::choices::client_config(&endpoint, &env, &keys);
        assert_eq!(config.local, expected, "{name}: the client's flag");
        if expected {
            locals += 1;
        } else {
            remotes += 1;
        }
    }
    assert!(
        locals > 10 && remotes > 10,
        "both answers are well covered ({locals} local, {remotes} remote)"
    );
}

fn text_input(spec: &Value) -> Option<Vec<u8>> {
    if spec.is_null() {
        return None;
    }
    if let Some(kind) = spec.get("generate").and_then(Value::as_str) {
        let mut bytes = br#"{"model": "big"}"#.to_vec();
        let size = local_model::MAX_MODEL_PREF_BYTES + usize::from(kind == "oversize");
        bytes.resize(size, b' ');
        return Some(bytes);
    }
    Some(encode(
        str_of(spec, "text"),
        spec.get("encoding")
            .and_then(Value::as_str)
            .unwrap_or("utf-8"),
    ))
}

#[test]
fn env_files_parse_and_keys_are_found_as_python_does() {
    let golden = read("keys.json");
    for case in golden["parse_env_file"].as_array().unwrap() {
        let name = str_of(case, "name");
        let expected: BTreeMap<String, String> = case["parsed"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
            .collect();
        let text = match case.get("bytes_hex").and_then(Value::as_str) {
            Some(hex) => {
                let bytes: Vec<u8> = (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                    .collect();
                match String::from_utf8(bytes) {
                    Ok(text) => text,
                    Err(_) => {
                        assert!(
                            expected.is_empty(),
                            "{name}: a file that is not UTF-8 reads as nothing"
                        );
                        continue;
                    }
                }
            }
            None => str_of(case, "text").to_owned(),
        };
        let mut got = BTreeMap::new();
        for (key, value) in keys::parse_env_text(&text) {
            got.insert(key, value);
        }
        assert_eq!(got, expected, "{name}");
    }

    for case in golden["get_key"].as_array().unwrap() {
        let name = str_of(case, "name");
        let scratch = Scratch::new("get-key");
        let files = case["files"].as_array().unwrap();
        let explicit = scratch.path().join("explicit.env");
        let root_file = scratch.path().join("FAMEnvironment.env");
        let engines_file = scratch.path().join("engines").join("FAMEnvironment.env");
        std::fs::create_dir_all(engines_file.parent().unwrap()).unwrap();
        for (file, path) in files.iter().zip([&explicit, &root_file, &engines_file]) {
            if let Some(text) = file.as_str() {
                std::fs::write(path, text).unwrap();
            }
        }
        let mut env = MapEnv::new().with("FAM_ENV_PATH", explicit.as_os_str());
        for (variable, value) in case["env"].as_object().unwrap() {
            env.set(variable, value.as_str().unwrap());
        }
        let store = KeyStore::new(Arc::new(env), &StateRoot::at(scratch.path()));
        for (key, expected) in case["values"].as_object().unwrap() {
            let got = store.get(key).map(|secret| secret.expose().to_owned());
            assert_eq!(got.as_deref(), expected.as_str(), "{name}: {key}");
        }
    }
}

#[test]
fn the_local_models_address_and_name_are_read_as_python_reads_them() {
    let golden = read("local_model.json");
    for case in golden["base_url"].as_array().unwrap() {
        let mut env = MapEnv::new();
        if let Some(value) = case["env"].as_str() {
            env.set("OLLAMA_BASE_URL", value);
        }
        assert_eq!(
            local_model::base_url(&env),
            str_of(case, "base_url"),
            "OLLAMA_BASE_URL = {}",
            case["env"]
        );
    }
    for case in golden["selected_model"].as_array().unwrap() {
        let name = str_of(case, "name");
        let scratch = Scratch::new("selected-model");
        let state = StateRoot::at(scratch.path());
        if let Some(bytes) = text_input(&case["file"]) {
            std::fs::create_dir_all(&state.globals).unwrap();
            std::fs::write(state.globals.join("analyst_model.json"), bytes).unwrap();
        }
        let mut env = MapEnv::new();
        if let Some(value) = case["env"].as_str() {
            env.set("OLLAMA_MODEL", value);
        }
        assert_eq!(
            local_model::selected_model(&env, &state),
            str_of(case, "selected"),
            "{name}"
        );
    }
}

#[test]
fn the_state_directory_rules_agree() {
    let golden = read("state.json");
    let scratch = Scratch::new("state");
    let tmp = scratch.path().to_string_lossy().into_owned();
    for case in golden["resolve"].as_array().unwrap() {
        let name = str_of(case, "name");
        let mut env = MapEnv::new();
        for (variable, value) in case["env"].as_object().unwrap() {
            env.set(variable, value.as_str().unwrap().replace("<TMP>", &tmp));
        }
        let got = state::resolve_with(&env, None, Platform::host());
        assert_eq!(
            got.installed,
            case["installed"].as_bool().unwrap(),
            "{name}"
        );
        assert_eq!(got.globals, got.root.join("globals"), "{name}");
        if let Some(root) = case["root"].as_str() {
            let expected = PathBuf::from(root.replace("<TMP>", &tmp));
            assert_eq!(got.root, expected, "{name}");
            assert_eq!(
                got.globals,
                PathBuf::from(str_of(case, "globals").replace("<TMP>", &tmp)),
                "{name}"
            );
        }
    }
    for case in golden["force_packaged"].as_array().unwrap() {
        let value = str_of(case, "value");
        let env = MapEnv::new().with("ALELYON_FORCE_PACKAGED", value);
        assert_eq!(
            state::forced_packaged(&env),
            case["forced"].as_bool().unwrap(),
            "{value:?}"
        );
    }
}

fn fault_kind(error: CalcError) -> &'static str {
    match error {
        CalcError::Disallowed | CalcError::Incomplete => "syntax",
        CalcError::DivisionByZero => "division_by_zero",
        CalcError::NotReal => "not_real",
        CalcError::TooLong
        | CalcError::TooDeep
        | CalcError::NumberTooLarge
        | CalcError::ResultTooLarge
        | CalcError::ExponentRange => "limit",
    }
}

#[test]
fn the_calculator_answers_as_pythons_own_arithmetic_does() {
    let golden = read("calc.json");
    let cases = golden["cases"].as_array().unwrap();
    assert!(cases.len() > 600);
    let (mut answers, mut faults) = (0, 0);
    let mut kinds: BTreeMap<&str, usize> = BTreeMap::new();
    for case in cases {
        let expression = str_of(case, "expression");
        match (
            calc::calculate(expression),
            case.get("ok").and_then(Value::as_str),
        ) {
            (Ok(got), Some(expected)) => {
                assert_eq!(got, expected, "{expression:?}");
                answers += 1;
            }
            (Err(error), None) => {
                let expected = str_of(case, "error");
                assert_eq!(fault_kind(error), expected, "{expression:?} ({error})");
                *kinds.entry(fault_kind(error)).or_default() += 1;
                faults += 1;
            }
            (got, expected) => panic!(
                "{expression:?}: got {got:?}, Python says {expected:?} / {}",
                case["error"]
            ),
        }
    }
    assert!(
        answers > 300 && faults > 80,
        "answers {answers}, refusals {faults}"
    );
    for kind in ["syntax", "division_by_zero", "not_real", "limit"] {
        assert!(kinds.contains_key(kind), "no case of {kind}");
    }
    let limits = &golden["limits"];
    assert_eq!(limits["characters"], calc::MAX_EXPRESSION_CHARS);
    assert_eq!(limits["exponent"].as_f64().unwrap(), calc::MAX_EXPONENT);
    assert_eq!(limits["base"].as_f64().unwrap(), calc::MAX_BASE);
    assert_eq!(limits["magnitude"].as_f64().unwrap(), calc::MAX_MAGNITUDE);
}

/// X7: a child's environment allowlist is the Python agent host's
/// (`session.SAFE_AGENT_ENV_NAMES`), name for name and in order; the native
/// additions are stated beside it, not hidden in it.
#[test]
fn the_child_environment_allowlist_is_the_python_hosts() {
    use lattice_core::exec::spawn::{NATIVE_ADDITIONS, SAFE_AGENT_ENV_NAMES};
    let golden = read("agent/env_allowlist.json");
    let names: Vec<&str> = golden["SAFE_AGENT_ENV_NAMES"]
        .as_array()
        .expect("a list of names")
        .iter()
        .map(|name| name.as_str().expect("a name"))
        .collect();
    assert_eq!(names, SAFE_AGENT_ENV_NAMES);
    for (name, _) in NATIVE_ADDITIONS {
        assert!(!names.contains(&name), "{name} is a native addition");
    }
}

/// An endpoint from a golden's row (the dataclass's fields; the judged ones are ignored).
fn endpoint_of(row: &Value) -> ModelEndpoint {
    let kind = registry::EndpointKind::parse(str_of(row, "kind")).expect("a known kind");
    ModelEndpoint {
        id: str_of(row, "id").to_owned(),
        label: str_of(row, "label").to_owned(),
        kind,
        base_url: str_of(row, "base_url").to_owned(),
        model: str_of(row, "model").to_owned(),
        api_key_name: str_of(row, "api_key_name").to_owned(),
        enabled: row["enabled"].as_bool().unwrap(),
        builtin: row["builtin"].as_bool().unwrap(),
        note: str_of(row, "note").to_owned(),
    }
}

#[test]
fn registry_writes_leave_the_bytes_python_leaves() {
    let golden = read("registry/writes.json");
    let scratch = Scratch::new("writes");
    let path = scratch.path().join("model_endpoints.json");
    for (i, step) in golden["steps"].as_array().unwrap().iter().enumerate() {
        let result = match str_of(step, "op") {
            "upsert" => registry::upsert(&path, endpoint_of(&step["argument"])),
            "remove" => registry::remove(&path, step["argument"].as_str().unwrap()),
            other => panic!("unknown op {other}"),
        };
        result.unwrap_or_else(|e| panic!("step {i}: {e}"));
        let written = std::fs::read(&path).unwrap();
        let expected = str_of(step, "text").replace('\n', registry::LINE_END);
        assert_eq!(
            String::from_utf8(written).unwrap(),
            expected,
            "step {i} ({}) wrote other bytes than Python's save",
            str_of(step, "op")
        );
        // What was written reads back whole, as Python reads it.
        assert!(registry::load_with_report(&path).complete(), "step {i}");
    }
}

#[test]
fn a_registry_write_begins_only_from_a_complete_reading_and_never_saves_an_invalid_row() {
    let scratch = Scratch::new("writes-refused");
    let path = scratch.path().join("model_endpoints.json");
    std::fs::write(&path, b"{\"version\": 1, \"endpoints\": [{\"id\": \"x\"}]}").unwrap();
    let before = std::fs::read(&path).unwrap();
    let row = registry::builtins().into_iter().next().unwrap();
    assert!(matches!(
        registry::upsert(&path, row),
        Err(registry::WriteError::IncompleteRegistry)
    ));
    assert_eq!(
        std::fs::read(&path).unwrap(),
        before,
        "a refused write changed the file"
    );
    let fresh = scratch.path().join("fresh.json");
    let bad = ModelEndpoint {
        id: "bad".into(),
        label: "Bad".into(),
        kind: registry::EndpointKind::OpenaiCompatible,
        base_url: "http://user:pw@example.invalid/v1".into(),
        model: String::new(),
        api_key_name: String::new(),
        enabled: true,
        builtin: false,
        note: String::new(),
    };
    assert!(
        matches!(registry::upsert(&fresh, bad), Err(registry::WriteError::Invalid(id)) if id == "bad")
    );
    assert!(!fresh.exists(), "an invalid row was written");
}
