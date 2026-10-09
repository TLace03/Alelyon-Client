//! The models a person can pick for a run, and how a pick becomes a client.
//!
//! A choice is one of the registry's endpoints, `endpoint:<id>` (the registry
//! is the single list: `endpoint:llamacpp-local` IS the platform's managed
//! llama.cpp server, running the model the local-model preference selects), or
//! `dev:scripted`, the scripted development model, when the service was built
//! with development mode on.
//!
//! What the interface is told about a choice:
//! - `ready`: an OpenAI-compatible endpoint is ready when the registry says it
//!   is (enabled, a model name, an address, its key present); an Anthropic
//!   endpoint never is, because this runtime speaks only Chat Completions; a
//!   saved Ollama row never is, because Ollama is retired (ADR-0041); and the
//!   managed llama.cpp row is not ready HERE, because this runtime does not yet
//!   start that server (the platform's Python side does). A llama-server a
//!   person runs themselves is the OpenAI-compatible `llamacpp` row;
//! - `refusal`, when not ready: one sentence, derived from the registry's
//!   status. It never names the API key (the protocol keeps key names out of
//!   every value the interface receives), so "needs OPENAI_API_KEY" becomes "It
//!   needs an API key, and none is set.";
//! - `locality`: from the ADDRESS the client will actually call (and, for
//!   Ollama, the model it will ask for), never from the endpoint's label or kind.
//!   An Ollama endpoint with no address of its own calls `OLLAMA_BASE_URL` (or
//!   the default loopback server), so it is local exactly when that address is;
//!   and a model the local daemon serves remotely (`*-cloud`) is never local
//!   ([`registry::ollama_is_local`], the one rule, which the client's `local`
//!   flag, and so its proxy rule, and the `secrets_stay_local` guardrail share).
//!
//! A saved Ollama row is still shown with the locality of the address its
//! client would have called; its refusal never repeats that address.
//!
//! Order: the development model, then the models on this machine, then the rest,
//! each group by label (case-insensitive), as the Python Lattice orders them
//! (`ready_endpoints`: local first, then by label).
//!
//! Building a client from an endpoint (`client_config`): an Ollama endpoint's
//! server (its address, else the local model's, trimmed) gets `/v1`, unless it
//! already ends in it; an OpenAI-compatible endpoint's address loses a trailing
//! `/chat/completions` and trailing `/`; the key, when the endpoint names one,
//! is read at this moment and kept as a `SecretString`.
//!
//! Invariant: nothing here returns, formats or logs a key value or name.

use lattice_agents::ChatCompletionsConfig;
use lattice_protocol::{Locality, ModelChoice, ModelKind};

use crate::env::Env;
use crate::keys::KeyStore;
use crate::llama::ManagedEndpoint;
use crate::local_model;
use crate::py;
use crate::registry::{self, EndpointKind, ModelEndpoint};
use crate::state::StateRoot;

/// The scripted development model's id.
pub const DEV_MODEL_ID: &str = "dev:scripted";
/// Every registry endpoint's choice id starts with this.
pub const ENDPOINT_PREFIX: &str = "endpoint:";

const DEV_LABEL: &str = "Development model (scripted)";
const ANTHROPIC_REFUSAL: &str = "The Agents runtime needs an OpenAI-compatible Chat Completions endpoint; this one speaks Anthropic's Messages API.";
const LLAMACPP_REFUSAL: &str = "The native Lattice cannot start the managed llama.cpp server yet; turn on the \"llama.cpp server you run yourself\" entry and run llama-server, or use this model from the Python Lattice.";
const OLLAMA_RETIRED_REFUSAL: &str = "Ollama is retired (ADR-0041): use a llama.cpp model instead.";

/// A choice and the endpoint behind it (none for the development model).
#[derive(Clone, Debug)]
pub struct Resolved {
    pub choice: ModelChoice,
    pub endpoint: Option<ModelEndpoint>,
}

/// The choice id of a registry endpoint.
pub fn endpoint_choice_id(endpoint_id: &str) -> String {
    format!("{ENDPOINT_PREFIX}{endpoint_id}")
}

/// One sentence for why an endpoint cannot be used, from the registry's status
/// (which for a missing key would name it).
fn refusal_for(endpoint: &ModelEndpoint, keys: &KeyStore) -> String {
    match endpoint.kind {
        EndpointKind::Anthropic => return ANTHROPIC_REFUSAL.to_owned(),
        EndpointKind::Llamacpp => return LLAMACPP_REFUSAL.to_owned(),
        EndpointKind::Ollama => return OLLAMA_RETIRED_REFUSAL.to_owned(),
        EndpointKind::OpenaiCompatible => {}
    }
    let status = endpoint.status(keys);
    match status.as_str() {
        "disabled" => "It is turned off in the model registry.".to_owned(),
        "no server URL set" => "No server address is set for it.".to_owned(),
        "no model name set" => "No model name is set for it.".to_owned(),
        _ if status.starts_with("needs ") => "It needs an API key, and none is set.".to_owned(),
        _ => "It is not ready.".to_owned(),
    }
}

/// The server an Ollama endpoint with no address of its own calls, as text:
/// `OLLAMA_BASE_URL` or the default loopback server, trimmed and without a
/// trailing `/` (Python's client strips it the same way before it calls).
fn ollama_env_server(env: &dyn Env) -> String {
    py::strip(&local_model::base_url(env))
        .trim_end_matches('/')
        .to_owned()
}

/// Every choice, in order, each with its endpoint. Reads the registry and the
/// key sources now, so a change on disk shows up on the next call.
pub fn resolve_all(
    env: &dyn Env,
    state: &StateRoot,
    keys: &KeyStore,
    development: bool,
) -> Vec<Resolved> {
    let report = registry::load_with_report(&registry::config_path(env, state));
    let mut endpoints: Vec<ModelEndpoint> = report
        .endpoints
        .into_iter()
        .map(|endpoint| registry::runtime_endpoint(endpoint, env, state))
        .collect();
    endpoints.sort_by_cached_key(|endpoint| (!endpoint.local(env), endpoint.label.to_lowercase()));

    let mut all = Vec::with_capacity(endpoints.len() + 1);
    if development {
        all.push(Resolved {
            choice: ModelChoice {
                id: DEV_MODEL_ID.to_owned(),
                label: DEV_LABEL.to_owned(),
                kind: ModelKind::Development,
                locality: Locality::Local,
                ready: true,
                refusal: None,
            },
            endpoint: None,
        });
    }
    for endpoint in endpoints {
        // This runtime speaks Chat Completions to an address it is given; it
        // does not start the managed llama.cpp server.
        let supported = !matches!(
            endpoint.kind,
            EndpointKind::Anthropic | EndpointKind::Llamacpp
        );
        let ready = supported && endpoint.ready(keys);
        let refusal = (!ready).then(|| refusal_for(&endpoint, keys));
        all.push(Resolved {
            choice: ModelChoice {
                id: endpoint_choice_id(&endpoint.id),
                label: endpoint.label.clone(),
                kind: match endpoint.kind {
                    EndpointKind::Ollama => ModelKind::Ollama,
                    EndpointKind::OpenaiCompatible => ModelKind::OpenaiCompatible,
                    EndpointKind::Anthropic => ModelKind::Anthropic,
                    EndpointKind::Llamacpp => ModelKind::Llamacpp,
                },
                locality: if endpoint.local(env) {
                    Locality::Local
                } else {
                    Locality::Remote
                },
                ready,
                refusal,
            },
            endpoint: Some(endpoint),
        });
    }
    all
}

/// The API root a client talks to: `<address>/v1` for Ollama, the address
/// without `/chat/completions` for an OpenAI-compatible endpoint, and nothing
/// for the managed llama.cpp server, whose address exists only once the
/// platform has started it (this runtime never offers that row as ready).
pub fn api_base(endpoint: &ModelEndpoint, env: &dyn Env) -> String {
    match endpoint.kind {
        EndpointKind::Llamacpp => String::new(),
        EndpointKind::Ollama => {
            let address = if endpoint.base_url.is_empty() {
                ollama_env_server(env)
            } else {
                endpoint.base_url.clone()
            };
            let address = address.trim_end_matches('/');
            if address.ends_with("/v1") {
                address.to_owned()
            } else {
                format!("{address}/v1")
            }
        }
        _ => {
            let address = endpoint.base_url.trim().trim_end_matches('/');
            let address = address.strip_suffix("/chat/completions").unwrap_or(address);
            address.trim_end_matches('/').to_owned()
        }
    }
}

/// What a client for `endpoint` needs, with the key read now.
pub fn client_config(
    endpoint: &ModelEndpoint,
    env: &dyn Env,
    keys: &KeyStore,
) -> ChatCompletionsConfig {
    let mut config = ChatCompletionsConfig::new(api_base(endpoint, env), endpoint.model.clone())
        .local(endpoint.local(env));
    if endpoint.needs_key()
        && let Some(key) = keys.get(&endpoint.api_key_name)
    {
        config = config.with_api_key(key);
    }
    config
}

/// What a client for the core's managed llama.cpp server needs (spec §22
/// LR2): its loopback `/v1` root, the served alias, this launch's token as
/// the key, and `local` set, so no proxy ever sees the request. The server is
/// local by construction: the core started it on 127.0.0.1.
pub fn managed_client_config(endpoint: &ManagedEndpoint) -> ChatCompletionsConfig {
    ChatCompletionsConfig::new(endpoint.api_base(), endpoint.alias.clone())
        .local(true)
        .with_api_key(endpoint.token.clone())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use super::*;
    use crate::env::MapEnv;
    use crate::testkit::TempDir;

    fn setup(env: MapEnv, root: &Path) -> (Arc<MapEnv>, StateRoot, KeyStore) {
        let env = Arc::new(env);
        let state = StateRoot::at(root);
        let keys = KeyStore::new(env.clone(), &state);
        (env, state, keys)
    }

    fn write_registry(state: &StateRoot, json: &str) {
        std::fs::create_dir_all(&state.globals).unwrap();
        std::fs::write(state.globals.join("model_endpoints.json"), json).unwrap();
    }

    /// A saved Ollama row with no address of its own (retired, ADR-0041).
    const SAVED_OLLAMA: &str = r#"{"version": 1, "endpoints": [{"id": "mine", "label": "Mine", "kind": "ollama", "model": "m"}]}"#;

    fn saved_ollama_with_model(model: &str) -> String {
        serde_json::json!({"version": 1, "endpoints": [
            {"id": "mine", "label": "Mine", "kind": "ollama", "model": model}
        ]})
        .to_string()
    }

    fn choice<'a>(all: &'a [Resolved], id: &str) -> &'a ModelChoice {
        &all.iter()
            .find(|r| r.choice.id == id)
            .unwrap_or_else(|| panic!("no {id}"))
            .choice
    }

    #[test]
    fn a_fresh_machine_offers_the_registry_and_says_why_the_managed_row_waits() {
        let dir = TempDir::new("choices-fresh");
        let (env, state, keys) = setup(MapEnv::new(), dir.path());
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        assert_eq!(all.len(), 16, "every endpoint, and no development model");
        assert!(all.iter().all(|r| r.choice.id.starts_with("endpoint:")));
        assert!(
            all.iter().all(|r| !r.choice.ready),
            "the local model is the managed llama.cpp server, which this runtime does not start yet"
        );
        let local = choice(&all, "endpoint:llamacpp-local");
        assert_eq!(local.locality, Locality::Local);
        assert_eq!(local.kind, ModelKind::Llamacpp);
        assert_eq!(local.refusal.as_deref(), Some(LLAMACPP_REFUSAL));
    }

    #[test]
    fn the_order_is_development_then_local_then_the_rest_each_by_label() {
        let dir = TempDir::new("choices-order");
        let (env, state, keys) = setup(MapEnv::new(), dir.path());
        write_registry(
            &state,
            r#"{"version": 1, "endpoints": [
                {"id": "zed", "label": "zed box", "base_url": "http://127.0.0.1:9/v1", "model": "m"},
                {"id": "apple", "label": "Apple hosted", "base_url": "https://apple.example.test/v1", "model": "m"}
            ]}"#,
        );
        let all = resolve_all(env.as_ref(), &state, &keys, true);
        assert_eq!(all[0].choice.id, DEV_MODEL_ID);
        assert_eq!(all[0].choice.kind, ModelKind::Development);
        assert!(all[0].choice.ready && all[0].endpoint.is_none());
        let local: Vec<&str> = all
            .iter()
            .filter(|r| r.choice.locality == Locality::Local)
            .map(|r| r.choice.label.as_str())
            .collect();
        let first_remote = all
            .iter()
            .position(|r| r.choice.locality == Locality::Remote)
            .unwrap();
        assert!(
            all[..first_remote]
                .iter()
                .all(|r| r.choice.locality == Locality::Local)
        );
        assert!(
            all[first_remote..]
                .iter()
                .all(|r| r.choice.locality == Locality::Remote)
        );
        let mut sorted = local.clone();
        sorted[1..].sort_by_key(|label| label.to_lowercase());
        assert_eq!(local, sorted, "the local group is by label, ignoring case");
        let remote: Vec<String> = all[first_remote..]
            .iter()
            .map(|r| r.choice.label.to_lowercase())
            .collect();
        let mut remote_sorted = remote.clone();
        remote_sorted.sort();
        assert_eq!(remote, remote_sorted);
        assert!(
            remote.iter().position(|l| l == "apple hosted").unwrap()
                < remote.iter().position(|l| l == "openai").unwrap()
        );
    }

    #[test]
    fn locality_comes_from_the_address_not_the_label() {
        let dir = TempDir::new("choices-locality");
        let (env, state, keys) = setup(MapEnv::new(), dir.path());
        write_registry(
            &state,
            r#"{"version": 1, "endpoints": [
                {"id": "sneaky", "label": "Local Qwen (private, on this machine)", "base_url": "https://api.example.test/v1", "model": "m"},
                {"id": "honest", "label": "Some Server", "base_url": "http://localhost:8000/v1", "model": "m"}
            ]}"#,
        );
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        assert_eq!(choice(&all, "endpoint:sneaky").locality, Locality::Remote);
        assert_eq!(choice(&all, "endpoint:honest").locality, Locality::Local);
    }

    #[test]
    fn a_refusal_says_why_and_never_names_a_key() {
        let dir = TempDir::new("choices-refusal");
        let (env, state, keys) = setup(MapEnv::new(), dir.path());
        write_registry(
            &state,
            r#"{"version": 1, "endpoints": [
                {"id": "openai", "enabled": true},
                {"id": "groq", "enabled": true, "model": ""},
                {"id": "mine", "label": "Mine", "base_url": "http://localhost:1/v1"},
                {"id": "anthropic", "enabled": true},
                {"id": "old", "label": "Old Ollama", "kind": "ollama", "base_url": "http://localhost:11434", "model": "m"}
            ]}"#,
        );
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        let refusal = |id: &str| choice(&all, id).refusal.clone().unwrap();
        assert_eq!(refusal("endpoint:llamacpp-local"), LLAMACPP_REFUSAL);
        assert_eq!(refusal("endpoint:old"), OLLAMA_RETIRED_REFUSAL);
        assert_eq!(
            refusal("endpoint:openai"),
            "It needs an API key, and none is set."
        );
        assert_eq!(refusal("endpoint:groq"), "No model name is set for it.");
        assert_eq!(refusal("endpoint:mine"), "No model name is set for it.");
        assert_eq!(
            refusal("endpoint:xai"),
            "It is turned off in the model registry."
        );
        assert_eq!(
            refusal("endpoint:anthropic"),
            "The Agents runtime needs an OpenAI-compatible Chat Completions endpoint; this one speaks Anthropic's Messages API."
        );
        assert!(
            !choice(&all, "endpoint:anthropic").ready,
            "an Anthropic endpoint is never ready, key or no key"
        );
        let everything =
            serde_json::to_string(&all.iter().map(|r| &r.choice).collect::<Vec<_>>()).unwrap();
        for name in [
            "OPENAI_API_KEY",
            "GROQ_API_KEY",
            "HF_TOKEN",
            "ANTHROPIC_API_KEY",
            "_API_KEY",
        ] {
            assert!(
                !everything.contains(name),
                "{name} appears in what the interface receives"
            );
        }
    }

    #[test]
    fn a_key_in_the_environment_makes_a_remote_endpoint_ready() {
        let dir = TempDir::new("choices-key");
        let env = MapEnv::new().with("OPENAI_API_KEY", "fixture-value");
        let (env, state, keys) = setup(env, dir.path());
        write_registry(
            &state,
            r#"{"version": 1, "endpoints": [{"id": "openai", "enabled": true}]}"#,
        );
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        let openai = choice(&all, "endpoint:openai");
        assert!(openai.ready && openai.refusal.is_none());
        assert_eq!(openai.locality, Locality::Remote);
        assert_eq!(openai.kind, ModelKind::OpenaiCompatible);
    }

    #[test]
    fn the_managed_row_runs_the_selected_model() {
        let dir = TempDir::new("choices-selected");
        let (env, state, keys) = setup(MapEnv::new().with("OLLAMA_MODEL", "llama3:8b"), dir.path());
        std::fs::create_dir_all(&state.globals).unwrap();
        std::fs::write(
            state.globals.join("analyst_model.json"),
            br#"{"model": "qwen3-4b"}"#,
        )
        .unwrap();
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        let endpoint = all
            .iter()
            .find(|r| r.choice.id == "endpoint:llamacpp-local")
            .unwrap()
            .endpoint
            .clone()
            .unwrap();
        assert_eq!(
            endpoint.model, "qwen3-4b",
            "the stored preference, never the Ollama variable"
        );
    }

    #[test]
    fn the_api_base_is_worked_out_per_kind() {
        let env = MapEnv::new();
        assert_eq!(
            api_base(&registry::builtins().remove(0), &env),
            "",
            "the managed server has no address until the platform starts it"
        );
        let report = registry::load_bytes_with_report(SAVED_OLLAMA.as_bytes());
        let mut ollama = report
            .endpoints
            .into_iter()
            .find(|e| e.id == "mine")
            .unwrap();
        assert_eq!(api_base(&ollama, &env), "http://localhost:11434/v1");
        assert_eq!(
            api_base(
                &ollama,
                &MapEnv::new().with("OLLAMA_BASE_URL", "http://box:1234/")
            ),
            "http://box:1234/v1"
        );
        ollama.base_url = "http://box:11434/v1/".into();
        assert_eq!(
            api_base(&ollama, &env),
            "http://box:11434/v1",
            "an address that already ends in /v1 keeps it"
        );
        ollama.base_url = "http://box:11434".into();
        assert_eq!(api_base(&ollama, &env), "http://box:11434/v1");

        let mut openai = registry::builtins()
            .into_iter()
            .find(|e| e.id == "openai")
            .unwrap();
        assert_eq!(api_base(&openai, &env), "https://api.openai.com/v1");
        openai.base_url = "https://api.openai.com/v1/chat/completions".into();
        assert_eq!(api_base(&openai, &env), "https://api.openai.com/v1");
        openai.base_url = "https://host.example.test/v1/chat/completions/".into();
        assert_eq!(api_base(&openai, &env), "https://host.example.test/v1");
        openai.base_url = "  https://host.example.test/openai/  ".into();
        assert_eq!(api_base(&openai, &env), "https://host.example.test/openai");
    }

    fn the_ollama_row(all: &[Resolved]) -> &Resolved {
        all.iter()
            .find(|r| r.choice.id == "endpoint:mine")
            .expect("the saved Ollama row")
    }

    #[test]
    fn the_ollama_rows_locality_follows_the_address_the_client_will_call() {
        for (address, expected) in [
            ("http://192.168.1.50:11434", Locality::Remote),
            ("https://ollama.example.test", Locality::Remote),
            ("http://gpu-box:11434/", Locality::Remote),
            ("http://localhost:11434", Locality::Local),
            ("http://127.0.0.1:9999/", Locality::Local),
            ("http://[::1]:11434", Locality::Local),
        ] {
            let dir = TempDir::new("choices-ollama-address");
            let (env, state, keys) =
                setup(MapEnv::new().with("OLLAMA_BASE_URL", address), dir.path());
            write_registry(&state, SAVED_OLLAMA);
            let all = resolve_all(env.as_ref(), &state, &keys, false);
            let row = the_ollama_row(&all);
            assert_eq!(row.choice.locality, expected, "{address}: the label");
            assert!(!row.choice.ready, "{address}: retired (ADR-0041)");
            assert_eq!(row.choice.refusal.as_deref(), Some(OLLAMA_RETIRED_REFUSAL));
            let config = client_config(row.endpoint.as_ref().unwrap(), env.as_ref(), &keys);
            assert_eq!(
                config.local,
                expected == Locality::Local,
                "{address}: the client's flag, which also decides whether a proxy is used"
            );
        }
        // Unset: the default loopback server.
        let dir = TempDir::new("choices-ollama-default");
        let (env, state, keys) = setup(MapEnv::new(), dir.path());
        write_registry(&state, SAVED_OLLAMA);
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        assert_eq!(the_ollama_row(&all).choice.locality, Locality::Local);
    }

    #[test]
    fn a_custom_ollama_row_without_an_address_follows_the_environment_too() {
        let dir = TempDir::new("choices-ollama-custom");
        let (env, state, keys) = setup(
            MapEnv::new().with("OLLAMA_BASE_URL", "http://192.168.1.50:11434"),
            dir.path(),
        );
        write_registry(
            &state,
            r#"{"version": 1, "endpoints": [
                {"id": "mine", "label": "Mine", "kind": "ollama", "model": "m"},
                {"id": "near", "label": "Near", "kind": "ollama", "base_url": "http://localhost:11434", "model": "m"},
                {"id": "far", "label": "Far", "kind": "ollama", "base_url": "http://gpu-box:11434", "model": "m"}
            ]}"#,
        );
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        assert_eq!(choice(&all, "endpoint:mine").locality, Locality::Remote);
        assert_eq!(choice(&all, "endpoint:near").locality, Locality::Local);
        assert_eq!(choice(&all, "endpoint:far").locality, Locality::Remote);
        let mine = all.iter().find(|r| r.choice.id == "endpoint:mine").unwrap();
        assert!(!client_config(mine.endpoint.as_ref().unwrap(), env.as_ref(), &keys).local);
    }

    #[test]
    fn an_ollama_cloud_model_is_remote_whatever_the_address() {
        for (model, expected) in [
            ("gpt-oss:120b-cloud", Locality::Remote),
            ("qwen3-coder:480b-cloud", Locality::Remote),
            ("foo:cloud", Locality::Remote),
            ("FOO:CLOUD", Locality::Remote),
            ("Foo:120B-Cloud", Locality::Remote),
            ("registry.example.test/ns/foo:cloud", Locality::Remote),
            ("qwen3-coder:30b", Locality::Local),
            ("cloud", Locality::Local),
            ("my-cloud", Locality::Local),
            ("foo:cloudy", Locality::Local),
            ("cloud:7b", Locality::Local),
            ("foo:120b-cloud-x", Locality::Local),
        ] {
            let dir = TempDir::new("choices-ollama-cloud");
            let (env, state, keys) = setup(MapEnv::new(), dir.path());
            write_registry(&state, &saved_ollama_with_model(model));
            let all = resolve_all(env.as_ref(), &state, &keys, false);
            let row = the_ollama_row(&all);
            assert_eq!(row.choice.locality, expected, "{model}");
            let config = client_config(row.endpoint.as_ref().unwrap(), env.as_ref(), &keys);
            assert_eq!(config.local, expected == Locality::Local, "{model}");
        }
        // Behind an explicit loopback address as well.
        let dir = TempDir::new("choices-ollama-cloud-explicit");
        let (env, state, keys) = setup(MapEnv::new(), dir.path());
        write_registry(
            &state,
            r#"{"version": 1, "endpoints": [
                {"id": "cl", "label": "Cloud via loopback", "kind": "ollama", "base_url": "http://127.0.0.1:11434", "model": "gpt-oss:120b-cloud"}
            ]}"#,
        );
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        assert_eq!(choice(&all, "endpoint:cl").locality, Locality::Remote);
    }

    #[test]
    fn an_ollama_row_is_never_ready_and_an_invalid_address_is_never_echoed() {
        for bad in [
            "ftp://localhost:11434",
            "http://user:pw@localhost:11434",
            "http://localhost:11434/?token=abc",
            "http://localhost:11434/#frag",
            "not a url",
            "localhost:11434",
            "http://localhost:99999",
            "http:///v1",
            "   ",
        ] {
            let dir = TempDir::new("choices-ollama-invalid");
            let (env, state, keys) = setup(MapEnv::new().with("OLLAMA_BASE_URL", bad), dir.path());
            write_registry(
                &state,
                r#"{"version": 1, "endpoints": [{"id": "mine", "label": "Mine", "kind": "ollama", "model": "m"}]}"#,
            );
            let all = resolve_all(env.as_ref(), &state, &keys, false);
            let row = choice(&all, "endpoint:mine");
            assert!(!row.ready, "{bad}");
            assert_eq!(
                row.refusal.as_deref(),
                Some(OLLAMA_RETIRED_REFUSAL),
                "{bad}"
            );
            let shown =
                serde_json::to_string(&all.iter().map(|r| &r.choice).collect::<Vec<_>>()).unwrap();
            assert!(
                !shown.contains("token=abc") && !shown.contains("pw@"),
                "{bad}: {shown}"
            );
        }
        // A row with an address of its own is retired just the same.
        let dir = TempDir::new("choices-ollama-invalid-but-unused");
        let (env, state, keys) = setup(
            MapEnv::new().with("OLLAMA_BASE_URL", "ftp://nowhere"),
            dir.path(),
        );
        write_registry(
            &state,
            r#"{"version": 1, "endpoints": [{"id": "own", "label": "Own", "kind": "ollama", "base_url": "http://localhost:11434", "model": "m"}]}"#,
        );
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        assert!(!choice(&all, "endpoint:own").ready);
        assert_eq!(choice(&all, "endpoint:own").locality, Locality::Local);
    }

    #[test]
    fn an_address_with_spaces_around_it_is_used_trimmed_as_python_uses_it() {
        let dir = TempDir::new("choices-ollama-spaces");
        let (env, state, keys) = setup(
            MapEnv::new().with("OLLAMA_BASE_URL", "  http://localhost:1234/  "),
            dir.path(),
        );
        write_registry(&state, SAVED_OLLAMA);
        let all = resolve_all(env.as_ref(), &state, &keys, false);
        let row = the_ollama_row(&all);
        assert!(row.choice.locality == Locality::Local);
        assert_eq!(
            api_base(row.endpoint.as_ref().unwrap(), env.as_ref()),
            "http://localhost:1234/v1"
        );
    }

    #[test]
    fn the_managed_servers_client_is_local_with_its_launch_token() {
        let endpoint = ManagedEndpoint {
            base_url: "http://127.0.0.1:41000".into(),
            alias: "qwen3-8b".into(),
            token: "launch-token".into(),
            binary_sha256: None,
            generation: 1,
        };
        let config = managed_client_config(&endpoint);
        assert!(config.local, "no proxy for the managed server");
        assert_eq!(config.base_url, "http://127.0.0.1:41000/v1");
        assert_eq!(config.model, "qwen3-8b");
        assert_eq!(
            config.api_key.as_ref().map(|k| k.expose()),
            Some("launch-token")
        );
        assert!(!format!("{config:?}").contains("launch-token"));
    }

    #[test]
    fn a_client_config_carries_locality_and_a_key_only_when_one_is_named() {
        let dir = TempDir::new("choices-config");
        let env = MapEnv::new().with("OPENAI_API_KEY", "fixture-value");
        let (env, state, keys) = setup(env, dir.path());
        let mut local = registry::builtins()
            .into_iter()
            .find(|e| e.id == "lmstudio")
            .unwrap();
        local.model = "loopback-model".into();
        let config = client_config(&local, env.as_ref(), &keys);
        assert!(config.local && config.api_key.is_none());
        assert_eq!(config.model, "loopback-model");
        let openai = registry::builtins()
            .into_iter()
            .find(|e| e.id == "openai")
            .unwrap();
        let config = client_config(&openai, env.as_ref(), &keys);
        assert!(!config.local);
        assert_eq!(
            config
                .api_key
                .as_ref()
                .map(|k| k.expose().to_owned())
                .as_deref(),
            Some("fixture-value")
        );
        let printed = format!("{config:?}");
        assert!(
            !printed.contains("fixture-value"),
            "the config's Debug hides the key: {printed}"
        );
        let _ = state;
    }
}
