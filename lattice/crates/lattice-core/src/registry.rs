//! The model registry: which models this machine can reach, as declared data.
//!
//! Parity with the Python runtime's `model_config` (the Python
//! Lattice's registry), read-only: the native Lattice reads the SAME
//! `model_endpoints.json`, so an endpoint added in one Lattice appears in the
//! other. `tests/parity/registry/*.json` are recorded from the real Python
//! functions by `tools/lattice_native_parity.py`, and `tests/parity.rs` holds
//! this port to them; a change to either side's rules turns that check red.
//!
//! What is ported, rule for rule:
//! - the built-in catalogue (`_builtins()`), embedded here;
//! - `ModelEndpoint.__post_init__`'s validation (bounded text, a closed set of
//!   kinds, the key-name shape, an OpenAI-compatible endpoint needs a base
//!   URL, the managed llama.cpp server takes neither a URL nor a key, http/https
//!   with a host, no user-info, query or fragment, and a keyed endpoint that is
//!   not on this machine must use HTTPS). A row that fails is skipped and the
//!   report says so;
//! - `load_with_report`: start from the built-ins keyed by id; a missing file is
//!   the valid built-ins-only state; a bounded read (1 MiB); a root that is
//!   `{"version": 1, "endpoints": [...]}` (a bare list is the legacy form and
//!   is accepted but incomplete); at most 10,000 rows; each row MERGES over
//!   the entry with the same id (its own fields override, `builtin` is
//!   recomputed as "is a built-in id"); unknown fields do not stop a row from
//!   merging but mark the report incomplete; order is the catalogue's, then new
//!   ids in file order;
//! - `local` (decided by the address, never by a label), `needs_key`,
//!   `has_key`, `ready` and `status`, with Python's exact status sentences.
//!
//! Ollama is retired (ADR-0041): a saved Ollama row still loads and is judged
//! for locality as before, but it is never ready. The local model is the
//! platform's own llama.cpp server ([`EndpointKind::Llamacpp`]), which the
//! platform starts on loopback with a per-launch token.
//!
//! Writing (`save`, `upsert`, `remove`), as Python writes: a write begins only
//! from a complete observation (`_entries_for_mutation`), every row passes the
//! constructor's validation again, the file is `json.dumps(..., indent=2)` (its
//! default `ensure_ascii`: every non-ASCII character escaped, as a surrogate
//! pair past the BMP) with Python's line end (`save` writes in text mode, so
//! CRLF on Windows), written to Python's own temporary name
//! (`model_endpoints.json.tmp`) and replaced into place, as `os.replace` does.
//! `tests/parity/registry/writes.json` holds each step's bytes as Python wrote
//! them. A key's VALUE is never written here: the registry holds only names.
//!
//! Invariants: loading never fails (a problem is an issue in the report and
//! every endpoint that could be recovered is kept); no key value is read while
//! loading; `local` never resolves a name.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::env::{self, Env};
use crate::keys::{KeyStore, valid_key_name};
use crate::local_model;
use crate::py;
use crate::pyurl;
use crate::state::StateRoot;

pub const CONFIG_SCHEMA_VERSION: i64 = 1;
pub const MAX_CONFIG_BYTES: usize = 1_048_576;
pub const MAX_ENDPOINT_ROWS: usize = 10_000;
pub const MAX_ENDPOINT_FIELD_CHARS: usize = 4_096;

/// What kind of server an endpoint talks to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EndpointKind {
    /// Retired (ADR-0041): a saved row loads, is never ready.
    Ollama,
    OpenaiCompatible,
    Anthropic,
    /// The platform's own llama.cpp server (ADR-0041): started by the platform
    /// on loopback with a per-launch token, so a row names neither an address
    /// nor a key.
    Llamacpp,
}

impl EndpointKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EndpointKind::Ollama => "ollama",
            EndpointKind::OpenaiCompatible => "openai-compatible",
            EndpointKind::Anthropic => "anthropic",
            EndpointKind::Llamacpp => "llamacpp",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "ollama" => Some(EndpointKind::Ollama),
            "openai-compatible" => Some(EndpointKind::OpenaiCompatible),
            "anthropic" => Some(EndpointKind::Anthropic),
            "llamacpp" => Some(EndpointKind::Llamacpp),
            _ => None,
        }
    }
}

/// The status of a saved Ollama row (Python's exact sentence).
pub const OLLAMA_RETIRED_STATUS: &str = "Ollama is retired (ADR-0041); use llama.cpp";
/// The status of a managed llama.cpp row with no model selected.
pub const LLAMACPP_NO_MODEL_STATUS: &str =
    "no model selected (put a GGUF file in ~/.alelyon/models)";

/// One reachable model. Holds the NAME of its key, never a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelEndpoint {
    pub id: String,
    pub label: String,
    pub kind: EndpointKind,
    pub base_url: String,
    pub model: String,
    pub api_key_name: String,
    pub enabled: bool,
    pub builtin: bool,
    pub note: String,
}

impl ModelEndpoint {
    /// Whether the endpoint is on this machine, decided by the address its
    /// client will actually call (and, for a saved Ollama row, the model it
    /// would ask for), never by a label: Anthropic never is; the managed
    /// llama.cpp server always is (the platform starts it on loopback); a saved
    /// Ollama row by [`ollama_is_local`], which reads `OLLAMA_BASE_URL` from
    /// `env` when the row has no address of its own; anything else by
    /// [`is_local_url`].
    pub fn local(&self, env: &dyn Env) -> bool {
        match self.kind {
            EndpointKind::Anthropic => false,
            EndpointKind::Llamacpp => true,
            EndpointKind::Ollama => ollama_is_local(&self.base_url, &self.model, env),
            EndpointKind::OpenaiCompatible => is_local_url(&self.base_url),
        }
    }

    pub fn needs_key(&self) -> bool {
        !self.api_key_name.is_empty()
    }

    /// Is the key this endpoint names present? An endpoint that needs none has it.
    pub fn has_key(&self, keys: &KeyStore) -> bool {
        self.api_key_name.is_empty() || keys.has(&self.api_key_name)
    }

    /// Python's `ready()`: a saved Ollama row never is; the managed llama.cpp
    /// row is whenever it is enabled (with no model selected its provider says
    /// so on every call, rather than the row vanishing without a word).
    pub fn ready(&self, keys: &KeyStore) -> bool {
        match self.kind {
            EndpointKind::Ollama => return false,
            EndpointKind::Llamacpp => return self.enabled,
            _ => {}
        }
        let has_target =
            self.kind != EndpointKind::OpenaiCompatible || !py::strip(&self.base_url).is_empty();
        self.enabled && !self.model.is_empty() && has_target && self.has_key(keys)
    }

    /// One sentence a person can act on; Python's exact strings.
    pub fn status(&self, keys: &KeyStore) -> String {
        if self.kind == EndpointKind::Ollama {
            return OLLAMA_RETIRED_STATUS.to_owned();
        }
        if !self.enabled {
            return "disabled".to_owned();
        }
        if self.kind == EndpointKind::OpenaiCompatible && py::strip(&self.base_url).is_empty() {
            return "no server URL set".to_owned();
        }
        if self.kind == EndpointKind::Llamacpp && self.model.is_empty() {
            return LLAMACPP_NO_MODEL_STATUS.to_owned();
        }
        if self.model.is_empty() {
            return "no model name set".to_owned();
        }
        if self.needs_key() && !self.has_key(keys) {
            return format!("needs {}", self.api_key_name);
        }
        "ready".to_owned()
    }
}

/// `is_local_url`: does this URL address THIS machine? Loopback only.
pub fn is_local_url(url: &str) -> bool {
    pyurl::is_local_url(url)
}

/// `is_ollama_cloud_model`: is this Ollama model served by a remote service
/// through the local daemon? Recent Ollama versions forward a model whose tag is
/// `cloud` or ends in `-cloud` (`gpt-oss:120b-cloud`, `qwen3-coder:480b-cloud`,
/// `foo:cloud`) from the daemon on this machine to a hosted service, so the
/// address a client calls is loopback and the prompt still leaves the machine.
/// Such a name is never local, whatever the address. Matching ignores case. The
/// tag is what follows the last `:` of the last path segment; a name with no tag
/// (`llama3`, `my-cloud`) is not a cloud model.
pub fn is_ollama_cloud_model(model: &str) -> bool {
    let lowered = py::strip(model).to_lowercase();
    let name = lowered.rsplit('/').next().unwrap_or_default();
    match name.rsplit_once(':') {
        Some((_, tag)) => tag == "cloud" || tag.ends_with("-cloud"),
        None => false,
    }
}

/// `ollama_is_local`: would an Ollama client using this address and model stay
/// on this machine? The URL the client would actually call is loopback AND the
/// model is not one the daemon serves remotely ([`is_ollama_cloud_model`]). An
/// empty `base_url` means the server the client fell back to,
/// `OLLAMA_BASE_URL` (read from `env`) or the default loopback server, and that
/// is the address judged. Ollama is retired (ADR-0041), so such a row is never
/// ready; the rule still decides what the window says about where it pointed.
pub fn ollama_is_local(base_url: &str, model: &str, env: &dyn Env) -> bool {
    if is_ollama_cloud_model(model) {
        return false;
    }
    let address = py::strip(base_url);
    if address.is_empty() {
        is_local_url(&local_model::base_url(env))
    } else {
        is_local_url(address)
    }
}

/// Closed, content-free reasons a registry observation is incomplete
/// (`LoadIssue`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoadIssue {
    Unreadable,
    Corrupt,
    UnsupportedRoot,
    UnsupportedSchema,
    InvalidRow,
    DuplicateEndpointId,
}

impl LoadIssue {
    pub fn as_str(self) -> &'static str {
        match self {
            LoadIssue::Unreadable => "unreadable",
            LoadIssue::Corrupt => "corrupt",
            LoadIssue::UnsupportedRoot => "unsupported-root",
            LoadIssue::UnsupportedSchema => "unsupported-schema",
            LoadIssue::InvalidRow => "invalid-row",
            LoadIssue::DuplicateEndpointId => "duplicate-endpoint-id",
        }
    }
}

/// A bounded observation of the registry (`ModelConfigLoadReport`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadReport {
    pub endpoints: Vec<ModelEndpoint>,
    /// Each class once, in the order first seen.
    pub issues: Vec<LoadIssue>,
}

impl LoadReport {
    pub fn complete(&self) -> bool {
        self.issues.is_empty()
    }
}

/// A built-in row: keyless, disabled, no address or model, until told otherwise.
fn ep(id: &str, label: &str, kind: EndpointKind) -> ModelEndpoint {
    ModelEndpoint {
        id: id.to_owned(),
        label: label.to_owned(),
        kind,
        base_url: String::new(),
        model: String::new(),
        api_key_name: String::new(),
        enabled: false,
        builtin: true,
        note: String::new(),
    }
}

impl ModelEndpoint {
    fn at(mut self, base_url: &str) -> Self {
        self.base_url = base_url.to_owned();
        self
    }

    fn model(mut self, model: &str) -> Self {
        self.model = model.to_owned();
        self
    }

    fn key(mut self, name: &str) -> Self {
        self.api_key_name = name.to_owned();
        self
    }

    fn on(mut self) -> Self {
        self.enabled = true;
        self
    }

    fn note(mut self, note: &str) -> Self {
        self.note = note.to_owned();
        self
    }
}

/// The catalogue the registry starts from (`_builtins()`), in its order.
pub fn builtins() -> Vec<ModelEndpoint> {
    use EndpointKind::{Anthropic, Llamacpp, OpenaiCompatible as Openai};
    vec![
        ep("llamacpp-local", "llama.cpp (this machine, managed)", Llamacpp)
            .on()
            .note(
                "Keyless. Runs entirely on this computer: Alelyon starts llama.cpp's server when it is needed, with a GGUF model from ~/.alelyon/models.",
            ),
        ep("local-openai", "Local server (vLLM / TGI / llama.cpp / LM Studio)", Openai)
            .at("http://localhost:8000/v1")
            .note(
                "Point this at your own served weights. Keyless by default; set a key name if your server requires one.",
            ),
        ep("lmstudio", "LM Studio (this machine)", Openai)
            .at("http://localhost:1234/v1")
            .note(
                "LM Studio's local server, on its default port. Keyless. Leave the model blank and pick from what the server lists.",
            ),
        ep("llamacpp", "llama.cpp server you run yourself (this machine)", Openai)
            .at("http://localhost:8080/v1")
            .note(
                "A llama-server you started yourself, on its default port. Most builds serve one model and ignore the model field. The managed entry above needs no server of your own.",
            ),
        ep("anthropic", "Anthropic", Anthropic)
            .model("claude-sonnet-5")
            .key("ANTHROPIC_API_KEY"),
        ep("huggingface", "Hugging Face (Inference Providers)", Openai)
            .at("https://router.huggingface.co/v1")
            .key("HF_TOKEN")
            .note(
                "One token in front of the models hosted through Hugging Face's inference partners. Use a full model id such as openai/gpt-oss-120b; an optional :provider suffix pins which partner serves it.",
            ),
        ep("cohere", "Cohere", Openai)
            .at("https://api.cohere.ai/compatibility/v1")
            .model("command-a-03-2025")
            .key("COHERE_API_KEY"),
        ep("openai", "OpenAI", Openai)
            .at("https://api.openai.com/v1")
            .model("gpt-4o")
            .key("OPENAI_API_KEY"),
        ep("google", "Google Gemini", Openai)
            .at("https://generativelanguage.googleapis.com/v1beta/openai")
            .model("gemini-2.0-flash")
            .key("GEMINI_API_KEY"),
        ep("xai", "xAI Grok", Openai)
            .at("https://api.x.ai/v1")
            .model("grok-2-latest")
            .key("XAI_API_KEY"),
        ep("deepseek", "DeepSeek", Openai)
            .at("https://api.deepseek.com/v1")
            .model("deepseek-chat")
            .key("DEEPSEEK_API_KEY"),
        ep("mistral", "Mistral", Openai)
            .at("https://api.mistral.ai/v1")
            .model("mistral-large-latest")
            .key("MISTRAL_API_KEY"),
        ep("groq", "Groq", Openai)
            .at("https://api.groq.com/openai/v1")
            .model("llama-3.3-70b-versatile")
            .key("GROQ_API_KEY"),
        ep("together", "Together AI", Openai)
            .at("https://api.together.xyz/v1")
            .key("TOGETHER_API_KEY"),
        ep("fireworks", "Fireworks AI", Openai)
            .at("https://api.fireworks.ai/inference/v1")
            .key("FIREWORKS_API_KEY"),
        ep("openrouter", "OpenRouter", Openai)
            .at("https://openrouter.ai/api/v1")
            .key("OPENROUTER_API_KEY"),
    ]
}

/// `config_path`: `ALELYON_MODEL_CONFIG` when set (and not blank), else
/// `<globals>/model_endpoints.json`.
pub fn config_path(env: &dyn Env, state: &StateRoot) -> PathBuf {
    match env::text(env, "ALELYON_MODEL_CONFIG") {
        Some(value) if !py::strip(&value).is_empty() => PathBuf::from(py::strip(&value)),
        _ => state.globals.join("model_endpoints.json"),
    }
}

/// `_bounded_text`.
fn bounded(value: &Value, nonempty: bool) -> Option<&str> {
    let text = value.as_str()?;
    if (nonempty && text.is_empty()) || text.chars().count() > MAX_ENDPOINT_FIELD_CHARS {
        return None;
    }
    Some(text)
}

/// The fields of one row, as JSON values, before validation. A field that is
/// absent takes the dataclass default; one that is present with the wrong type
/// makes the row invalid.
#[derive(Default)]
struct Fields {
    id: Option<Value>,
    label: Option<Value>,
    kind: Option<Value>,
    base_url: Option<Value>,
    model: Option<Value>,
    api_key_name: Option<Value>,
    enabled: Option<Value>,
    note: Option<Value>,
}

impl Fields {
    fn of(endpoint: &ModelEndpoint) -> Self {
        Self {
            id: Some(endpoint.id.clone().into()),
            label: Some(endpoint.label.clone().into()),
            kind: Some(endpoint.kind.as_str().into()),
            base_url: Some(endpoint.base_url.clone().into()),
            model: Some(endpoint.model.clone().into()),
            api_key_name: Some(endpoint.api_key_name.clone().into()),
            enabled: Some(endpoint.enabled.into()),
            note: Some(endpoint.note.clone().into()),
        }
    }

    /// `{**existing, **supplied}` for the known fields except `builtin`.
    fn overlay(&mut self, row: &Map<String, Value>) {
        for (key, value) in row {
            let slot = match key.as_str() {
                "id" => &mut self.id,
                "label" => &mut self.label,
                "kind" => &mut self.kind,
                "base_url" => &mut self.base_url,
                "model" => &mut self.model,
                "api_key_name" => &mut self.api_key_name,
                "enabled" => &mut self.enabled,
                "note" => &mut self.note,
                _ => continue,
            };
            *slot = Some(value.clone());
        }
    }

    /// `ModelEndpoint(**merged)`: the constructor and its `__post_init__`.
    fn build(self, builtin: bool) -> Result<ModelEndpoint, ()> {
        let id = bounded(self.id.as_ref().ok_or(())?, true).ok_or(())?;
        let label = bounded(self.label.as_ref().ok_or(())?, true).ok_or(())?;
        let text = |value: &Option<Value>| -> Result<String, ()> {
            match value {
                None => Ok(String::new()),
                Some(value) => bounded(value, false).map(str::to_owned).ok_or(()),
            }
        };
        let base_url = text(&self.base_url)?;
        let model = text(&self.model)?;
        let api_key_name = text(&self.api_key_name)?;
        let note = text(&self.note)?;
        let kind = match &self.kind {
            None => EndpointKind::OpenaiCompatible,
            Some(value) => EndpointKind::parse(value.as_str().ok_or(())?).ok_or(())?,
        };
        let enabled = match &self.enabled {
            None => true,
            Some(Value::Bool(flag)) => *flag,
            Some(_) => return Err(()),
        };
        if !api_key_name.is_empty() && !valid_key_name(&api_key_name) {
            return Err(());
        }
        if kind == EndpointKind::OpenaiCompatible && base_url.is_empty() {
            return Err(());
        }
        // The platform starts the managed server and holds its launch token.
        if kind == EndpointKind::Llamacpp && (!base_url.is_empty() || !api_key_name.is_empty()) {
            return Err(());
        }
        if !base_url.is_empty() {
            check_base_url(&base_url, !api_key_name.is_empty())?;
        }
        Ok(ModelEndpoint {
            id: id.to_owned(),
            label: label.to_owned(),
            kind,
            base_url,
            model,
            api_key_name,
            enabled,
            builtin,
            note,
        })
    }
}

/// The address rules of `__post_init__`. Also what the address named by
/// `OLLAMA_BASE_URL` must satisfy (see `choices`), so that an address the
/// registry would refuse is refused wherever it comes from.
pub(crate) fn check_base_url(base_url: &str, keyed: bool) -> Result<(), ()> {
    let raw = py::strip(base_url);
    let split = pyurl::urlsplit(raw)?;
    let host = split.hostname().unwrap_or_default();
    split.check_port()?;
    if !matches!(split.scheme.as_str(), "http" | "https") || host.is_empty() {
        return Err(());
    }
    if split.has_userinfo() || raw.contains('?') || raw.contains('#') {
        return Err(());
    }
    if keyed && split.scheme != "https" && !is_local_url(base_url) {
        return Err(());
    }
    Ok(())
}

fn add_issue(issues: &mut Vec<LoadIssue>, issue: LoadIssue) {
    if !issues.contains(&issue) {
        issues.push(issue);
    }
}

/// What a registry file held, as `load_with_report` reads it.
enum Source {
    Missing,
    Unreadable,
    Bytes(Vec<u8>),
}

fn read_source(path: &Path) -> Source {
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Source::Missing,
        Err(_) => return Source::Unreadable,
    };
    let mut bytes = Vec::new();
    match (&mut file)
        .take(MAX_CONFIG_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
    {
        Ok(_) => Source::Bytes(bytes),
        Err(_) => Source::Unreadable,
    }
}

/// `load_with_report(path)`: the built-ins with the file's rows merged over
/// them, and what was wrong with the file.
pub fn load_with_report(path: &Path) -> LoadReport {
    match read_source(path) {
        Source::Missing => report(builtins(), Vec::new()),
        Source::Unreadable => report(builtins(), vec![LoadIssue::Unreadable]),
        Source::Bytes(bytes) => load_bytes_with_report(&bytes),
    }
}

/// [`load_with_report`] over the bytes of a file already read.
pub fn load_bytes_with_report(bytes: &[u8]) -> LoadReport {
    let mut issues = Vec::new();
    if bytes.len() > MAX_CONFIG_BYTES {
        return report(builtins(), vec![LoadIssue::Corrupt]);
    }
    let Some(root) = py::loads_bytes(bytes) else {
        return report(builtins(), vec![LoadIssue::Corrupt]);
    };

    let rows: Vec<Value> = match root {
        Value::Object(object) => {
            let keys_ok = object.len() == 2
                && object.contains_key("version")
                && object.contains_key("endpoints");
            let version_ok = object
                .get("version")
                .and_then(Value::as_i64)
                .is_some_and(|version| version == CONFIG_SCHEMA_VERSION);
            if !keys_ok || !version_ok {
                add_issue(&mut issues, LoadIssue::UnsupportedSchema);
            }
            match object.get("endpoints") {
                Some(Value::Array(rows)) => rows.clone(),
                _ => {
                    add_issue(&mut issues, LoadIssue::UnsupportedSchema);
                    Vec::new()
                }
            }
        }
        Value::Array(rows) => {
            add_issue(&mut issues, LoadIssue::UnsupportedRoot);
            rows
        }
        _ => {
            add_issue(&mut issues, LoadIssue::UnsupportedRoot);
            Vec::new()
        }
    };
    let mut rows = rows;
    if rows.len() > MAX_ENDPOINT_ROWS {
        add_issue(&mut issues, LoadIssue::InvalidRow);
        rows.truncate(MAX_ENDPOINT_ROWS);
    }

    let mut endpoints = builtins();
    let builtin_ids: Vec<String> = endpoints.iter().map(|e| e.id.clone()).collect();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for row in &rows {
        let Some(row) = row.as_object() else {
            add_issue(&mut issues, LoadIssue::InvalidRow);
            continue;
        };
        let Some(row_id) = row.get("id").and_then(|id| bounded(id, true)) else {
            add_issue(&mut issues, LoadIssue::InvalidRow);
            continue;
        };
        if !seen.insert(row_id.to_owned()) {
            add_issue(&mut issues, LoadIssue::DuplicateEndpointId);
        }
        const KNOWN: [&str; 9] = [
            "id",
            "label",
            "kind",
            "base_url",
            "model",
            "api_key_name",
            "enabled",
            "builtin",
            "note",
        ];
        if !row.keys().all(|key| KNOWN.contains(&key.as_str())) {
            add_issue(&mut issues, LoadIssue::InvalidRow);
        }
        if row.get("builtin").is_some_and(|flag| !flag.is_boolean()) {
            add_issue(&mut issues, LoadIssue::InvalidRow);
        }

        let position = endpoints.iter().position(|e| e.id == row_id);
        let mut fields = match position {
            Some(at) => Fields::of(&endpoints[at]),
            None => Fields::default(),
        };
        fields.overlay(row);
        match fields.build(builtin_ids.iter().any(|id| id == row_id)) {
            Ok(entry) => match position {
                Some(at) => endpoints[at] = entry,
                None => endpoints.push(entry),
            },
            Err(()) => add_issue(&mut issues, LoadIssue::InvalidRow),
        }
    }
    report(endpoints, issues)
}

fn report(endpoints: Vec<ModelEndpoint>, issues: Vec<LoadIssue>) -> LoadReport {
    LoadReport { endpoints, issues }
}

// ------------------------------------------------------------------ writing

/// Why a registry write did not begin, or did not finish.
#[derive(Debug)]
pub enum WriteError {
    /// `ModelConfigMutationRefused(INCOMPLETE_REGISTRY)`: the file could not be
    /// read whole, so writing it back would drop what could not be read.
    IncompleteRegistry,
    /// An endpoint `ModelEndpoint.__post_init__` refuses (its id, when it has one).
    Invalid(String),
    Io(std::io::Error),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncompleteRegistry => f.write_str(
                "the model registry could not be read whole, so it was not written (nothing in it was lost)",
            ),
            Self::Invalid(id) => write!(
                f,
                "the endpoint {id:?} is not valid: it needs an id and a label, an http or https address with a host \
                 and no user, query or fragment (https when it uses a key and is not on this computer), and a key \
                 name like MY_SERVICE_KEY"
            ),
            Self::Io(error) => write!(f, "the model registry could not be written: {error}"),
        }
    }
}

/// The line end `save` writes: Python's text mode turns `\n` into this.
pub const LINE_END: &str = if cfg!(windows) { "\r\n" } else { "\n" };

/// The registry file's text, as `save` composes it before its line ends are
/// translated: `json.dumps({"version": 1, "endpoints": rows}, indent=2) + "\n"`,
/// each row the dataclass's fields in their declared order.
pub fn registry_text(endpoints: &[ModelEndpoint]) -> String {
    let mut out = String::from("{\n  \"version\": 1,\n  \"endpoints\": ");
    if endpoints.is_empty() {
        out.push_str("[]");
    } else {
        out.push_str("[\n");
        for (i, e) in endpoints.iter().enumerate() {
            if i > 0 {
                out.push_str(",\n");
            }
            out.push_str("    {\n");
            let fields: [(&str, String); 9] = [
                ("id", ascii_json(&e.id)),
                ("label", ascii_json(&e.label)),
                ("kind", ascii_json(e.kind.as_str())),
                ("base_url", ascii_json(&e.base_url)),
                ("model", ascii_json(&e.model)),
                ("api_key_name", ascii_json(&e.api_key_name)),
                ("enabled", e.enabled.to_string()),
                ("builtin", e.builtin.to_string()),
                ("note", ascii_json(&e.note)),
            ];
            for (j, (name, value)) in fields.iter().enumerate() {
                out.push_str(&format!("      \"{name}\": {value}"));
                out.push_str(if j + 1 < fields.len() { ",\n" } else { "\n" });
            }
            out.push_str("    }");
        }
        out.push_str("\n  ]");
    }
    out.push_str("\n}\n");
    out
}

/// A string as `json.dumps` spells it with `ensure_ascii` (its default).
fn ascii_json(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

/// `save(endpoints)`: every row validated again, then the whole file written to
/// Python's temporary name and replaced into place. A key's value is never
/// among the fields, by construction.
pub fn save(path: &Path, endpoints: &[ModelEndpoint]) -> Result<(), WriteError> {
    for endpoint in endpoints {
        Fields::of(endpoint)
            .build(endpoint.builtin)
            .map_err(|()| WriteError::Invalid(endpoint.id.clone()))?;
    }
    let text = registry_text(endpoints).replace('\n', LINE_END);
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(WriteError::Io)?;
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = PathBuf::from(temporary);
    std::fs::write(&temporary, text.as_bytes()).map_err(WriteError::Io)?;
    crate::fsx::replace_shared(&temporary, path).map_err(WriteError::Io)
}

/// `_entries_for_mutation`: a complete observation, or no write at all.
fn entries_for_mutation(path: &Path) -> Result<Vec<ModelEndpoint>, WriteError> {
    let report = load_with_report(path);
    if !report.complete() {
        return Err(WriteError::IncompleteRegistry);
    }
    Ok(report.endpoints)
}

/// `upsert(endpoint)`: the row with its id replaced (moved to the end), or added.
pub fn upsert(path: &Path, endpoint: ModelEndpoint) -> Result<(), WriteError> {
    let mut entries: Vec<ModelEndpoint> = entries_for_mutation(path)?
        .into_iter()
        .filter(|e| e.id != endpoint.id)
        .collect();
    entries.push(endpoint);
    save(path, &entries)
}

/// `remove(id)`: a user row is deleted from the file; a built-in is disabled
/// instead (it would come back from the catalogue on the next load).
pub fn remove(path: &Path, id: &str) -> Result<(), WriteError> {
    let mut entries = Vec::new();
    for mut e in entries_for_mutation(path)? {
        if e.id != id {
            entries.push(e);
        } else if e.builtin {
            e.enabled = false;
            entries.push(e);
        }
    }
    save(path, &entries)
}

/// `runtime_endpoint`: the built-in managed llama.cpp row takes the model the
/// local model preference selects (`local_model.selected_model`); every other
/// row is returned unchanged.
pub fn runtime_endpoint(
    mut endpoint: ModelEndpoint,
    env: &dyn Env,
    state: &StateRoot,
) -> ModelEndpoint {
    if endpoint.id == "llamacpp-local"
        && endpoint.kind == EndpointKind::Llamacpp
        && endpoint.builtin
    {
        endpoint.model = local_model::selected_model(env, state);
    }
    endpoint
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::env::MapEnv;
    use crate::testkit::TempDir;

    fn keys(env: MapEnv, root: &Path) -> KeyStore {
        KeyStore::new(Arc::new(env), &StateRoot::at(root))
    }

    fn load(json: &str) -> LoadReport {
        load_bytes_with_report(json.as_bytes())
    }

    /// A saved Ollama row with no address of its own (retired, ADR-0041).
    fn saved_ollama() -> ModelEndpoint {
        ModelEndpoint {
            id: "o".into(),
            label: "O".into(),
            kind: EndpointKind::Ollama,
            base_url: String::new(),
            model: "m".into(),
            api_key_name: String::new(),
            enabled: true,
            builtin: false,
            note: String::new(),
        }
    }

    fn find<'a>(report: &'a LoadReport, id: &str) -> &'a ModelEndpoint {
        report
            .endpoints
            .iter()
            .find(|e| e.id == id)
            .unwrap_or_else(|| panic!("no {id}"))
    }

    #[test]
    fn the_catalogue_has_sixteen_builtins_in_order() {
        let ids: Vec<String> = builtins().into_iter().map(|e| e.id).collect();
        assert_eq!(
            ids,
            [
                "llamacpp-local",
                "local-openai",
                "lmstudio",
                "llamacpp",
                "anthropic",
                "huggingface",
                "cohere",
                "openai",
                "google",
                "xai",
                "deepseek",
                "mistral",
                "groq",
                "together",
                "fireworks",
                "openrouter",
            ]
        );
        assert!(builtins().iter().all(|e| e.builtin));
        // Every built-in passes the constructor's own validation.
        for endpoint in builtins() {
            let rebuilt = Fields::of(&endpoint).build(true).unwrap();
            assert_eq!(rebuilt, endpoint, "{}", endpoint.id);
        }
    }

    #[test]
    fn a_missing_file_is_the_builtins_only_state() {
        let dir = TempDir::new("registry-missing");
        let report = load_with_report(&dir.path().join("model_endpoints.json"));
        assert!(report.complete());
        assert_eq!(report.endpoints, builtins());
    }

    #[test]
    fn a_directory_or_an_oversize_file_is_incomplete_but_keeps_the_builtins() {
        let dir = TempDir::new("registry-bad");
        let report = load_with_report(dir.path());
        assert_eq!(report.issues, [LoadIssue::Unreadable]);
        assert_eq!(report.endpoints, builtins());
        let big = dir.path().join("big.json");
        let mut text = br#"{"version": 1, "endpoints": []}"#.to_vec();
        text.resize(MAX_CONFIG_BYTES + 1, b' ');
        std::fs::write(&big, &text).unwrap();
        assert_eq!(load_with_report(&big).issues, [LoadIssue::Corrupt]);
        text.truncate(MAX_CONFIG_BYTES);
        std::fs::write(&big, &text).unwrap();
        assert!(
            load_with_report(&big).complete(),
            "exactly the bound is allowed"
        );
    }

    #[test]
    fn corrupt_json_and_unsupported_roots_are_named() {
        assert_eq!(load("{not json").issues, [LoadIssue::Corrupt]);
        assert_eq!(load("").issues, [LoadIssue::Corrupt]);
        assert_eq!(load("42").issues, [LoadIssue::UnsupportedRoot]);
        assert_eq!(load("null").issues, [LoadIssue::UnsupportedRoot]);
        assert_eq!(load("\"text\"").issues, [LoadIssue::UnsupportedRoot]);
        assert_eq!(
            load("[]").issues,
            [LoadIssue::UnsupportedRoot],
            "the legacy list form is incomplete"
        );
        assert!(load(r#"{"version": 1, "endpoints": []}"#).complete());
        for schema in [
            r#"{"endpoints": []}"#,
            r#"{"version": 2, "endpoints": []}"#,
            r#"{"version": 1.0, "endpoints": []}"#,
            r#"{"version": true, "endpoints": []}"#,
            r#"{"version": "1", "endpoints": []}"#,
            r#"{"version": 1, "endpoints": [], "extra": 1}"#,
            r#"{"version": 1, "endpoints": {}}"#,
            r#"{"version": 1}"#,
        ] {
            assert_eq!(
                load(schema).issues,
                [LoadIssue::UnsupportedSchema],
                "{schema}"
            );
            assert_eq!(load(schema).endpoints, builtins());
        }
    }

    #[test]
    fn a_row_merges_over_the_builtin_with_its_id() {
        let report = load(
            r#"{"version": 1, "endpoints": [
                {"id": "openai", "enabled": true, "model": "gpt-4.1"},
                {"id": "llamacpp-local", "note": "mine"}
            ]}"#,
        );
        assert!(report.complete());
        let openai = find(&report, "openai");
        assert!(openai.enabled && openai.builtin);
        assert_eq!(openai.model, "gpt-4.1");
        assert_eq!(
            openai.label, "OpenAI",
            "fields the row does not give are kept"
        );
        assert_eq!(openai.base_url, "https://api.openai.com/v1");
        assert_eq!(find(&report, "llamacpp-local").note, "mine");
        assert_eq!(report.endpoints.len(), 16);
        let ids: Vec<&str> = report.endpoints.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            ids[..2],
            ["llamacpp-local", "local-openai"],
            "the catalogue's order is kept"
        );
    }

    #[test]
    fn new_ids_follow_the_catalogue_in_file_order_and_are_not_builtin() {
        let report = load(
            r#"[{"id": "zeta", "label": "Zeta", "base_url": "http://127.0.0.1:9000/v1", "model": "z"},
                {"id": "alpha", "label": "Alpha", "base_url": "http://localhost:9001/v1"}]"#,
        );
        let ids: Vec<&str> = report.endpoints.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids[16..], ["zeta", "alpha"]);
        assert!(!find(&report, "zeta").builtin);
        assert_eq!(report.issues, [LoadIssue::UnsupportedRoot]);
    }

    #[test]
    fn a_builtin_flag_in_a_row_cannot_change_what_is_builtin() {
        let report = load(
            r#"{"version": 1, "endpoints": [
                {"id": "openai", "builtin": false},
                {"id": "mine", "label": "Mine", "base_url": "http://localhost:1/v1", "builtin": true}
            ]}"#,
        );
        assert!(report.complete());
        assert!(find(&report, "openai").builtin);
        assert!(!find(&report, "mine").builtin);
        let bad = load(r#"{"version": 1, "endpoints": [{"id": "openai", "builtin": "yes"}]}"#);
        assert_eq!(bad.issues, [LoadIssue::InvalidRow]);
        assert!(find(&bad, "openai").builtin, "the row still merged");
    }

    #[test]
    fn an_unknown_field_merges_the_row_and_marks_the_report_incomplete() {
        let report = load(
            r#"{"version": 1, "endpoints": [{"id": "xai", "enabled": true, "colour": "red"}]}"#,
        );
        assert_eq!(report.issues, [LoadIssue::InvalidRow]);
        assert!(find(&report, "xai").enabled);
    }

    #[test]
    fn a_duplicate_id_merges_over_the_earlier_row_and_is_named() {
        let report = load(
            r#"{"version": 1, "endpoints": [
                {"id": "dup", "label": "First", "base_url": "http://localhost:1/v1", "model": "one"},
                {"id": "dup", "model": "two"}
            ]}"#,
        );
        assert_eq!(report.issues, [LoadIssue::DuplicateEndpointId]);
        let dup = find(&report, "dup");
        assert_eq!((dup.label.as_str(), dup.model.as_str()), ("First", "two"));
        assert_eq!(report.endpoints.len(), 17);
    }

    #[test]
    fn every_kind_of_invalid_row_is_skipped_and_named() {
        let long = "x".repeat(MAX_ENDPOINT_FIELD_CHARS + 1);
        let ok_long = "x".repeat(MAX_ENDPOINT_FIELD_CHARS);
        let rows = [
            "42".to_owned(),
            "null".to_owned(),
            r#"[1]"#.to_owned(),
            r#"{"label": "no id"}"#.to_owned(),
            r#"{"id": "", "label": "empty id"}"#.to_owned(),
            r#"{"id": 7, "label": "numeric id"}"#.to_owned(),
            format!(r#"{{"id": "{long}", "label": "long id", "base_url": "http://localhost/v1"}}"#),
            r#"{"id": "n1", "base_url": "http://localhost/v1"}"#.to_owned(),
            r#"{"id": "n2", "label": "", "base_url": "http://localhost/v1"}"#.to_owned(),
            r#"{"id": "n3", "label": "L", "kind": "gemini", "base_url": "http://localhost/v1"}"#.to_owned(),
            r#"{"id": "n4", "label": "L", "kind": 5, "base_url": "http://localhost/v1"}"#.to_owned(),
            r#"{"id": "n5", "label": "L", "base_url": "http://localhost/v1", "api_key_name": "lower"}"#.to_owned(),
            r#"{"id": "n6", "label": "L", "base_url": ""}"#.to_owned(),
            r#"{"id": "n7", "label": "L"}"#.to_owned(),
            r#"{"id": "n8", "label": "L", "base_url": "ftp://localhost/v1"}"#.to_owned(),
            r#"{"id": "n9", "label": "L", "base_url": "http:///v1"}"#.to_owned(),
            r#"{"id": "n10", "label": "L", "base_url": "http://user:pw@localhost/v1"}"#.to_owned(),
            r#"{"id": "n11", "label": "L", "base_url": "http://localhost/v1?x=1"}"#.to_owned(),
            r#"{"id": "n12", "label": "L", "base_url": "http://localhost/v1#frag"}"#.to_owned(),
            r#"{"id": "n13", "label": "L", "base_url": "http://models.example.test/v1", "api_key_name": "SOME_KEY"}"#.to_owned(),
            r#"{"id": "n14", "label": "L", "base_url": "http://localhost:99999/v1"}"#.to_owned(),
            r#"{"id": "n15", "label": "L", "base_url": "http://localhost/v1", "enabled": "yes"}"#.to_owned(),
            r#"{"id": "n16", "label": "L", "base_url": "http://localhost/v1", "enabled": 1}"#.to_owned(),
            r#"{"id": "n17", "label": "L", "base_url": "http://localhost/v1", "model": null}"#.to_owned(),
            r#"{"id": "n18", "label": "L", "base_url": "http://localhost/v1", "note": 3}"#.to_owned(),
            format!(r#"{{"id": "n19", "label": "L", "base_url": "http://localhost/v1", "note": "{long}"}}"#),
            r#"{"id": "n20", "label": "L", "base_url": "[::1"}"#.to_owned(),
            r#"{"id": "n21", "label": "L", "base_url": "localhost:8000/v1"}"#.to_owned(),
            r#"{"id": "n22", "label": "L", "base_url": "  "}"#.to_owned(),
            r#"{"id": "openai", "base_url": "http://models.example.test/v1"}"#.to_owned(),
        ];
        for row in &rows {
            let report = load(&format!(r#"{{"version": 1, "endpoints": [{row}]}}"#));
            assert_eq!(
                report.issues,
                [LoadIssue::InvalidRow],
                "{}",
                &row[..row.len().min(90)]
            );
            assert_eq!(
                report.endpoints,
                builtins(),
                "an invalid row changes nothing: {}",
                &row[..row.len().min(90)]
            );
        }
        let fine = load(&format!(
            r#"{{"version": 1, "endpoints": [{{"id": "ok", "label": "L", "base_url": "http://localhost/v1", "note": "{ok_long}"}}]}}"#
        ));
        assert!(fine.complete(), "exactly the bound is allowed");
    }

    #[test]
    fn a_keyed_endpoint_off_this_machine_must_use_https() {
        let ok = load(
            r#"{"version": 1, "endpoints": [
                {"id": "a", "label": "A", "base_url": "https://models.example.test/v1", "api_key_name": "SOME_KEY"},
                {"id": "b", "label": "B", "base_url": "http://localhost:8000/v1", "api_key_name": "SOME_KEY"},
                {"id": "c", "label": "C", "base_url": "http://[::1]:8000/v1", "api_key_name": "SOME_KEY"},
                {"id": "d", "label": "D", "base_url": "http://models.example.test/v1"}
            ]}"#,
        );
        assert!(ok.complete(), "{:?}", ok.issues);
        let bad = load(
            r#"{"version": 1, "endpoints": [{"id": "e", "label": "E", "base_url": "http://models.example.test/v1", "api_key_name": "SOME_KEY"}]}"#,
        );
        assert_eq!(bad.issues, [LoadIssue::InvalidRow]);
    }

    #[test]
    fn more_than_ten_thousand_rows_are_cut_and_named() {
        let row = r#"{"id": "openai", "note": "x"}"#;
        let rows = vec![row; MAX_ENDPOINT_ROWS + 5].join(",");
        let report = load(&format!(r#"{{"version": 1, "endpoints": [{rows}]}}"#));
        assert!(report.issues.contains(&LoadIssue::InvalidRow));
        assert!(report.issues.contains(&LoadIssue::DuplicateEndpointId));
        assert_eq!(report.endpoints.len(), 16);
    }

    #[test]
    fn a_registry_saved_with_a_byte_order_mark_or_as_utf16_still_loads() {
        let doc = r#"{"version": 1, "endpoints": [{"id": "xai", "enabled": true}]}"#;
        let mut bom = vec![0xef, 0xbb, 0xbf];
        bom.extend_from_slice(doc.as_bytes());
        assert!(find(&load_bytes_with_report(&bom), "xai").enabled);
        let mut utf16 = vec![0xff, 0xfe];
        utf16.extend(doc.encode_utf16().flat_map(|u| u.to_le_bytes()));
        let report = load_bytes_with_report(&utf16);
        assert!(report.complete() && find(&report, "xai").enabled);
    }

    #[test]
    fn locality_is_decided_by_the_address_and_never_by_the_label() {
        let env = MapEnv::new();
        let managed = builtins().remove(0);
        assert_eq!(managed.kind, EndpointKind::Llamacpp);
        assert!(
            managed.local(&env),
            "the managed server is on loopback by construction"
        );
        let mut endpoint = saved_ollama();
        assert!(
            endpoint.local(&env),
            "Ollama with no address and no variable is the default loopback server"
        );
        endpoint.base_url = "http://192.168.1.5:11434".into();
        assert!(!endpoint.local(&env));
        endpoint.base_url = "http://localhost:11434".into();
        assert!(endpoint.local(&env));
        endpoint.label = "Not local at all".into();
        assert!(endpoint.local(&env), "a label changes nothing");
        let anthropic = builtins()
            .into_iter()
            .find(|e| e.id == "anthropic")
            .unwrap();
        assert!(!anthropic.local(&env));
        let mut hosted = builtins().into_iter().find(|e| e.id == "openai").unwrap();
        assert!(!hosted.local(&env));
        hosted.label = "Local Qwen".into();
        hosted.base_url = "http://127.0.0.1:8000/v1".into();
        assert!(
            hosted.local(&env),
            "loopback address, whatever it is called"
        );
    }

    #[test]
    fn an_ollama_endpoint_without_an_address_is_local_when_the_address_it_will_call_is() {
        let ollama = saved_ollama();
        assert!(ollama.base_url.is_empty());
        for (variable, expected) in [
            (None, true),
            (Some(""), true),
            (Some("http://localhost:11434"), true),
            (Some("http://127.0.0.1:9999/"), true),
            (Some("http://[::1]:11434"), true),
            (Some("http://192.168.1.50:11434"), false),
            (Some("https://ollama.example.test"), false),
            (Some("http://gpu-box:11434"), false),
            (Some("not a url"), false),
            (Some("   "), false),
            (Some("  http://localhost:11434  "), true),
        ] {
            let env = match variable {
                Some(value) => MapEnv::new().with("OLLAMA_BASE_URL", value),
                None => MapEnv::new(),
            };
            assert_eq!(
                ollama.local(&env),
                expected,
                "OLLAMA_BASE_URL = {variable:?}"
            );
        }
        // An address of its own beats the variable, in both directions.
        let far = MapEnv::new().with("OLLAMA_BASE_URL", "http://192.168.1.50:11434");
        let near = MapEnv::new().with("OLLAMA_BASE_URL", "http://localhost:11434");
        let mut own = ollama.clone();
        own.base_url = "http://localhost:11434".into();
        assert!(own.local(&far));
        own.base_url = "http://192.168.1.50:11434".into();
        assert!(!own.local(&near));
    }

    #[test]
    fn a_cloud_model_is_remote_behind_any_address() {
        let env = MapEnv::new();
        for (model, cloud) in [
            ("gpt-oss:120b-cloud", true),
            ("qwen3-coder:480b-cloud", true),
            ("foo:cloud", true),
            ("FOO:CLOUD", true),
            ("Foo:120B-Cloud", true),
            ("  foo:cloud  ", true),
            ("registry.example.test/ns/foo:cloud", true),
            ("registry.example.test:5000/ns/foo:7b-cloud", true),
            ("qwen3-coder:30b", false),
            ("llama3", false),
            ("cloud", false),
            ("my-cloud", false),
            ("foo:cloudy", false),
            ("cloud:7b", false),
            ("foo:120b-cloud-x", false),
            ("cloud/foo:7b", false),
            ("registry.example.test:5000/ns/foo", false),
            ("", false),
        ] {
            assert_eq!(is_ollama_cloud_model(model), cloud, "{model:?}");
            for address in ["", "http://localhost:11434", "http://127.0.0.1:11434"] {
                assert_eq!(
                    ollama_is_local(address, model, &env),
                    !cloud,
                    "{model:?} at {address:?}"
                );
            }
        }
        let mut endpoint = saved_ollama();
        endpoint.model = "gpt-oss:120b-cloud".into();
        assert!(!endpoint.local(&env));
    }

    #[test]
    fn ready_and_status_use_pythons_sentences_in_pythons_order() {
        let dir = TempDir::new("registry-status");
        let none = keys(MapEnv::new(), dir.path());
        let with_key = keys(
            MapEnv::new().with("OPENAI_API_KEY", "fixture-value"),
            dir.path(),
        );
        let mut openai = builtins().into_iter().find(|e| e.id == "openai").unwrap();
        assert_eq!(openai.status(&with_key), "disabled");
        assert!(!openai.ready(&with_key));
        openai.enabled = true;
        assert_eq!(openai.status(&none), "needs OPENAI_API_KEY");
        assert!(!openai.ready(&none));
        assert_eq!(openai.status(&with_key), "ready");
        assert!(openai.ready(&with_key));
        openai.model.clear();
        assert_eq!(openai.status(&with_key), "no model name set");
        assert!(!openai.ready(&with_key));
        openai.model = "gpt-4o".into();
        openai.base_url = "  ".into();
        assert_eq!(openai.status(&with_key), "no server URL set");
        assert!(!openai.ready(&with_key));
        // The managed llama.cpp row: ready whenever it is enabled; with no model
        // its status says what to do.
        let mut managed = builtins().remove(0);
        assert!(
            managed.ready(&none),
            "a keyless local endpoint needs nothing"
        );
        assert_eq!(managed.status(&none), LLAMACPP_NO_MODEL_STATUS);
        assert!(!managed.needs_key());
        managed.model = "qwen3-4b".into();
        assert_eq!(managed.status(&none), "ready");
        managed.enabled = false;
        assert!(!managed.ready(&none));
        assert_eq!(managed.status(&none), "disabled");
        // A saved Ollama row is retired: never ready, and it says so first.
        let mut ollama = saved_ollama();
        assert!(!ollama.ready(&none));
        assert_eq!(ollama.status(&none), OLLAMA_RETIRED_STATUS);
        ollama.enabled = false;
        assert_eq!(ollama.status(&none), OLLAMA_RETIRED_STATUS);
    }

    #[test]
    fn a_managed_row_names_neither_an_address_nor_a_key() {
        let report = load(
            r#"{"version": 1, "endpoints": [
                {"id": "m1", "label": "With a URL", "kind": "llamacpp", "base_url": "http://127.0.0.1:1/v1", "model": "m"},
                {"id": "m2", "label": "With a key", "kind": "llamacpp", "model": "m", "api_key_name": "SOME_KEY"},
                {"id": "m3", "label": "Plain", "kind": "llamacpp", "model": "m"}
            ]}"#,
        );
        assert_eq!(report.issues, [LoadIssue::InvalidRow]);
        assert!(
            report
                .endpoints
                .iter()
                .all(|e| e.id != "m1" && e.id != "m2")
        );
        let plain = find(&report, "m3");
        assert_eq!(plain.kind, EndpointKind::Llamacpp);
        assert!(plain.local(&MapEnv::new()) && !plain.builtin);
    }

    #[test]
    fn the_config_path_prefers_the_override() {
        let state = StateRoot::at("/state");
        assert_eq!(
            config_path(&MapEnv::new(), &state),
            PathBuf::from("/state")
                .join("globals")
                .join("model_endpoints.json")
        );
        let env = MapEnv::new().with("ALELYON_MODEL_CONFIG", "  /elsewhere/endpoints.json ");
        assert_eq!(
            config_path(&env, &state),
            PathBuf::from("/elsewhere/endpoints.json")
        );
        let blank = MapEnv::new().with("ALELYON_MODEL_CONFIG", "   ");
        assert_eq!(
            config_path(&blank, &state),
            state.globals.join("model_endpoints.json")
        );
    }

    #[test]
    fn only_the_builtin_managed_row_takes_the_selected_model() {
        let dir = TempDir::new("registry-runtime");
        let state = StateRoot::at(dir.path());
        std::fs::create_dir_all(&state.globals).unwrap();
        std::fs::write(
            state.globals.join("analyst_model.json"),
            br#"{"model": "qwen3-4b"}"#,
        )
        .unwrap();
        // `OLLAMA_MODEL` decides nothing any more (ADR-0041).
        let env = MapEnv::new().with("OLLAMA_MODEL", "llama3:8b");
        let local = runtime_endpoint(builtins().remove(0), &env, &state);
        assert_eq!(local.model, "qwen3-4b");
        let mut saved = builtins().remove(0);
        saved.builtin = false;
        assert_eq!(runtime_endpoint(saved.clone(), &env, &state), saved);
        let mut custom = builtins().remove(0);
        custom.kind = EndpointKind::OpenaiCompatible;
        custom.base_url = "http://localhost:1/v1".into();
        assert_eq!(runtime_endpoint(custom.clone(), &env, &state), custom);
        let other = builtins().remove(1);
        assert_eq!(runtime_endpoint(other.clone(), &env, &state), other);
    }
}
