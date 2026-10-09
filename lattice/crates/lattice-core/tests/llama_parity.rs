//! The local runtime's ports against goldens recorded from ADR-0041 phase 1's
//! Python (the chat core's spec §22: LF6, LF7; LR3–LR6), as recorded:
//! - `llama/gguf_header.json`: `gguf_header.read_header`, `describe` and
//!   `is_gguf` over fixture files (valid, truncated, over a bound, not GGUF);
//! - `llama/llama_server.json`: `load_settings` over settings files,
//!   `list_models` and `resolve_model` over model folders, `_is_ollama_path`,
//!   and the constants the launch shares. `ManagedServer.command` is held by
//!   `llama_runtime.rs` (LF3).
//!
//! A case Python reads and the native port gives the default for instead
//! carries `deviation`; this test holds the port to the default there.
//! Files are written only in temporary folders this test makes.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use lattice_core::calc::float_repr;
use lattice_core::chat::pyjson;
use lattice_core::llama::files::{self, ModelProblem};
use lattice_core::llama::gguf::{self, GgufError};
use lattice_core::llama::probes;
use lattice_core::llama::server;
use lattice_core::llama::settings::{self, Settings};
use lattice_core::net::LOOPBACK;
use serde_json::Value;

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "lattice-core-llama-parity-{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn read(relative: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("parity")
        .join(relative);
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).unwrap())
        .collect()
}

/// LF7: the GGUF port reads and describes every fixture as Python does.
#[test]
fn gguf_headers_read_as_python_reads_them() {
    let golden = read("llama/gguf_header.json");
    let bounds = &golden["bounds"];
    assert_eq!(bounds["_MAX_STRING"].as_u64(), Some(gguf::MAX_STRING));
    assert_eq!(bounds["_MAX_COUNT"].as_u64(), Some(gguf::MAX_COUNT));
    assert_eq!(
        bounds["_MAX_DIMS"].as_u64(),
        Some(u64::from(gguf::MAX_DIMS))
    );
    assert_eq!(
        bounds["_KEEP_ARRAY_MAX"].as_u64(),
        Some(gguf::KEEP_ARRAY_MAX)
    );
    let table = golden["file_types"].as_object().unwrap();
    for number in 0..64 {
        let python = table.get(&number.to_string()).and_then(Value::as_str);
        assert_eq!(
            gguf::file_type_name(&number.to_string()),
            python,
            "file_type {number}"
        );
    }

    let scratch = Scratch::new("gguf");
    let cases = golden["cases"].as_array().unwrap();
    let mut errors = std::collections::BTreeSet::new();
    for (number, case) in cases.iter().enumerate() {
        let name = case["name"].as_str().unwrap();
        let path = scratch.0.join(format!("case{number}.gguf"));
        std::fs::write(&path, hex(case["hex"].as_str().unwrap())).unwrap();
        assert_eq!(
            files::is_gguf(&path),
            case["is_gguf"].as_bool().unwrap(),
            "{name}: is_gguf"
        );
        let outcome = gguf::read_header(&path);
        if let Some(kind) = case["error"].as_str() {
            errors.insert(kind.to_owned());
            assert_eq!(outcome.err().map(GgufError::kind), Some(kind), "{name}");
            continue;
        }
        let header = outcome.unwrap_or_else(|error| panic!("{name}: {error:?}"));
        assert_eq!(
            Some(u64::from(header.version)),
            case["version"].as_u64(),
            "{name}"
        );
        let keys: Vec<&str> = header
            .metadata
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        let python_keys: Vec<&str> = case["metadata_keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|key| key.as_str().unwrap())
            .collect();
        assert_eq!(keys, python_keys, "{name}: keys in Python's order");
        for (key, value) in &header.metadata {
            assert_eq!(
                pyjson::dumps(value),
                case["metadata"][key].as_str().unwrap(),
                "{name}: {key}"
            );
        }
        let lengths: Vec<(String, u64)> = case["array_lengths"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, value)| (key.clone(), value.as_u64().unwrap()))
            .collect();
        let mut native_lengths = header.array_lengths.clone();
        native_lengths.sort();
        assert_eq!(native_lengths, lengths, "{name}: counted arrays");
        let tensors: Vec<(String, Vec<u64>, u64)> = case["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                (
                    t["name"].as_str().unwrap().to_owned(),
                    t["shape"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|d| d.as_u64().unwrap())
                        .collect(),
                    t["type"].as_u64().unwrap(),
                )
            })
            .collect();
        let native_tensors: Vec<(String, Vec<u64>, u64)> = header
            .tensors
            .iter()
            .map(|t| (t.name.clone(), t.shape.clone(), u64::from(t.ggml_type)))
            .collect();
        assert_eq!(native_tensors, tensors, "{name}: tensors");
        match (header.describe(), case["describe_error"].as_str()) {
            (Err(error), Some(kind)) => assert_eq!(error.kind(), kind, "{name}"),
            (Ok(description), None) => {
                for (field, text) in description.dumped() {
                    assert_eq!(
                        text,
                        case["describe"][field].as_str().unwrap(),
                        "{name}: {field}"
                    );
                }
            }
            (native, python) => panic!("{name}: native {native:?}, Python {python:?}"),
        }
    }
    println!("{} cases; errors seen: {errors:?}", cases.len());
    for kind in [
        "ends_early",
        "not_gguf",
        "version",
        "implausible_count",
        "string_too_long",
        "array_too_long",
        "unknown_type",
        "too_many_dims",
    ] {
        assert!(errors.contains(kind), "no fixture fails with {kind}");
    }
}

/// LR5: settings follow Python's validation and defaults; the cases Python
/// reads and native gives the default for are named.
#[test]
fn settings_read_as_python_reads_them() {
    let golden = read("llama/llama_server.json");
    let constants = &golden["constants"];
    assert_eq!(constants["BINARY_ENV"], files::BINARY_ENV);
    assert_eq!(constants["MODELS_ENV"], files::MODELS_ENV);
    assert_eq!(constants["LOOPBACK"], LOOPBACK);
    assert_eq!(
        constants["DEFAULT_CTX"].as_u64(),
        Some(settings::DEFAULT_CTX)
    );
    assert_eq!(
        constants["DEFAULT_GPU_LAYERS"].as_u64(),
        Some(settings::DEFAULT_GPU_LAYERS)
    );
    assert_eq!(
        constants["DEFAULT_PARALLEL"].as_u64(),
        Some(settings::DEFAULT_PARALLEL)
    );
    assert_eq!(
        constants["DEFAULT_IDLE_SECONDS"].as_f64(),
        Some(settings::DEFAULT_IDLE_SECONDS)
    );
    assert_eq!(
        constants["START_TIMEOUT_S"].as_f64(),
        Some(server::START_TIMEOUT.as_secs_f64())
    );
    assert_eq!(
        constants["HEALTH_TIMEOUT_S"].as_f64(),
        Some(server::HEALTH_TIMEOUT.as_secs_f64())
    );
    assert_eq!(
        constants["STOP_TIMEOUT_S"].as_f64(),
        Some(server::STOP_TIMEOUT.as_secs_f64())
    );
    let prefixes: Vec<&str> = constants["STRIPPED_ENV_PREFIXES"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().unwrap())
        .collect();
    assert_eq!(prefixes, server::STRIPPED_ENV_PREFIXES);

    let mut deviations = 0;
    for case in golden["settings"].as_array().unwrap() {
        let native = match case["text"].as_str() {
            Some(text) => settings::parse(text.as_bytes()),
            None => Settings::default(),
        };
        let shown = [
            native.ctx_size.to_string(),
            native.gpu_layers.to_string(),
            native.parallel.to_string(),
            float_repr(native.idle_seconds),
        ];
        let python = [
            case["ctx_size"].as_str().unwrap(),
            case["gpu_layers"].as_str().unwrap(),
            case["parallel"].as_str().unwrap(),
            case["idle_seconds"].as_str().unwrap(),
        ];
        if case.get("deviation").is_some() {
            deviations += 1;
            assert_eq!(native, Settings::default(), "{case}");
            assert_ne!(
                shown.map(|s| s.to_owned()),
                python.map(str::to_owned),
                "{case}: a real deviation"
            );
        } else {
            assert_eq!(shown, python, "{case}");
        }
    }
    assert!(deviations >= 3);
}

/// LR3, LR4, LF6: the folder's models, the names that mean them, and the
/// paths refused as Ollama installs, as Python decides them.
#[test]
fn models_and_names_resolve_as_python_resolves_them() {
    let golden = read("llama/llama_server.json");
    for case in golden["is_ollama_path"].as_array().unwrap() {
        let path = case["path"].as_str().unwrap();
        assert_eq!(
            files::is_ollama_path(Path::new(path)),
            case["ollama"].as_bool().unwrap(),
            "{path}"
        );
    }
    let gguf = {
        let mut bytes = files::GGUF_MAGIC.to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes
    };
    let scratch = Scratch::new("models");
    for (number, folder) in golden["models"].as_array().unwrap().iter().enumerate() {
        let name = folder["name"].as_str().unwrap();
        let root = scratch.0.join(number.to_string());
        std::fs::create_dir_all(&root).unwrap();
        for file in folder["files"].as_array().unwrap() {
            let path = root.join(file["path"].as_str().unwrap());
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let bytes: &[u8] = match file["kind"].as_str().unwrap() {
                "gguf" => &gguf,
                "text" => b"not a model",
                _ => b"GG",
            };
            std::fs::write(&path, bytes).unwrap();
        }
        let listed: Vec<(String, String, u64)> = files::list_models(&root)
            .into_iter()
            .map(|m| {
                let relative = m
                    .path
                    .strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/");
                (m.name, relative, m.size)
            })
            .collect();
        let python: Vec<(String, String, u64)> = folder["listed"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                (
                    m["name"].as_str().unwrap().to_owned(),
                    m["path"].as_str().unwrap().to_owned(),
                    m["size"].as_u64().unwrap(),
                )
            })
            .collect();
        assert_eq!(listed, python, "{name}: listed");
        for query in folder["resolve"].as_array().unwrap() {
            let wanted = query["query"].as_str().unwrap();
            let native = files::resolve_model(wanted, &root);
            match query["model"].as_str() {
                Some(model) => assert_eq!(
                    native.map(|m| m.name).as_deref(),
                    Ok(model),
                    "{name}: {wanted:?}"
                ),
                None => {
                    let none_chosen = query["none_chosen"].as_bool().unwrap();
                    let problem = if none_chosen {
                        ModelProblem::NoneChosen
                    } else {
                        ModelProblem::NotFound
                    };
                    assert_eq!(native, Err(problem), "{name}: {wanted:?}");
                }
            }
        }
    }
}

/// The golden's `<drive X>` written back as `X` (`probe_key_drives`).
fn redrive(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("<drive ") {
        out.push_str(&rest[..at]);
        let after = &rest[at + "<drive ".len()..];
        let letter = after.chars().next().unwrap();
        assert_eq!(&after[1..2], ">", "{text}");
        out.push(letter);
        rest = &after[2..];
    }
    out.push_str(rest);
    out
}

/// Spec 22.6 P2: a probe record's key names the model by the path text
/// Python's `str(WindowsPath)` gives, so a models folder written with `.`
/// parts, doubled or forward separators finds Python's record.
#[test]
fn probe_keys_name_the_model_as_pathlib_writes_it() {
    let golden = read("llama/llama_server.json");
    let texts = golden["probe_key_paths"].as_array().unwrap();
    for case in texts {
        let text = redrive(case["text"].as_str().unwrap());
        assert_eq!(
            probes::windows_path_text(&text),
            redrive(case["python"].as_str().unwrap()),
            "{text:?}"
        );
    }
    let joined = golden["probe_key_joined"].as_array().unwrap();
    for case in joined {
        let folder = redrive(case["folder"].as_str().unwrap());
        let name = case["name"].as_str().unwrap();
        let path = Path::new(&folder).join(name);
        assert_eq!(
            probes::windows_path_text(&path.to_string_lossy()),
            redrive(case["python"].as_str().unwrap()),
            "{folder:?} + {name:?}"
        );
    }
    println!(
        "probe key paths: {} texts, {} joined",
        texts.len(),
        joined.len()
    );
}
