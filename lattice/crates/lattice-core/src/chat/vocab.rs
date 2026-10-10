//! The chat's model choices and what each resolves to (the native chat's spec
//! §2.4, §3.3.4; the chat core's spec row C4, as amended by §22 for
//! ADR-0041).
//!
//! **The vocabulary is the web's** (`lattice_service/chat.py`
//! `valid_provider`): `auto`, `local`, `cloud`, `endpoint:<id>` with the id
//! matching `[A-Za-z0-9_.-]{1,80}`, and `dev:echo` when the development model
//! is on. Nothing else is ever written as a pin. One deviation, stricter:
//! Python's `$` also matches before a final newline, so the web accepts
//! `endpoint:a\n`; this port does not. `chat/vocabulary.json`, recorded from
//! the web's Python, pins the rest.
//!
//! **What each choice is (LR1).**
//! - `local` and `auto` are the core's own managed llama.cpp server
//!   ([`Target::Managed`]), and nothing else. Auto has no other candidate:
//!   no registry endpoint, not even one on a loopback address, is ever used
//!   by Auto. Both are [`Locality::Local`] by construction (LR2). When the
//!   server cannot run (no binary, no model chosen, the chosen model is no
//!   file), they resolve to [`Target::Refused`] with an offline sentence,
//!   still Local, and nothing is sent anywhere.
//! - `cloud` is listed and refused until the Cloud client exists (N1).
//! - `endpoint:<id>` is the registry endpoint, used only when picked by name.
//!   A named endpoint is **never Local**, even on a loopback address: it is
//!   shown with its address and treated as off this machine, so the secret
//!   tripwire covers it and a send shown as Local can never resolve to it
//!   (N10). An Ollama endpoint is retired (ADR-0041): listed, never ready, and
//!   refused; it is never treated as local. Anthropic endpoints are refused
//!   until 2b.
//! - A registry row of the **managed kind** (`llamacpp`; the built-in
//!   `endpoint:llamacpp-local`) names no address and no key: it is the core's
//!   own managed server under the row's name, as Python's `endpoint_provider`
//!   builds `llamacpp_provider(row.model)` for it. So it resolves to
//!   [`Target::Managed`] and is Local by construction (LR2), exactly as Local
//!   is; it is not an endpoint a person points somewhere. The built-in row runs
//!   the selected model (`registry::runtime_endpoint`), any other managed row
//!   the model it names, matched exactly (LR4). Auto never uses it: Auto is
//!   [`managed_entry`] and nothing else.
//! - `dev:echo` is the development echo, Local, offered only in development.
//!
//! **Order.** Auto, Local, Cloud; then the enabled registry endpoints, those
//! on this machine first (a managed row, or a loopback address), each group
//! by label without regard to case (the web's `ready_endpoints` order); then
//! the development echo. Endpoints that are turned off are counted, not
//! listed.
//!
//! Nothing here reads an `OLLAMA_*` variable: an Ollama row is judged by its
//! kind alone, and the registry's Ollama projection (`runtime_endpoint`) is
//! not used.
//!
//! The sentences in [`words`] that describe the managed server are
//! PROVISIONAL (spec §22.3, row C4): a wording golden from Python follows
//! ADR-0041 phase 1 once it lands on main.

use std::fs;
use std::path::PathBuf;

use lattice_agents::sanitize_url_for_trace;
use lattice_protocol::Locality;
use lattice_protocol::chat::{ChatChoice, LocalRuntime, RuntimeState, Shown};

use crate::choices;
use crate::env::{self, Env};
use crate::keys::KeyStore;
use crate::llama::files::{
    self, BinaryProblem, LlamaPaths, LocalModel, ModelProblem, find_binary, list_models,
};
use crate::py;
use crate::registry::{self, EndpointKind, ModelEndpoint};
use crate::state::StateRoot;

use super::pyjson::{self, PyValue};

/// The base choices, in the picker's order.
pub const AUTO: &str = "auto";
pub const LOCAL: &str = "local";
pub const CLOUD: &str = "cloud";
pub const BASE_CHOICES: [&str; 3] = [AUTO, LOCAL, CLOUD];
/// The development echo's id.
pub const DEV_ECHO: &str = "dev:echo";
/// Every registry endpoint's choice id starts with this.
pub const ENDPOINT_PREFIX: &str = "endpoint:";
/// The longest endpoint id a choice may name.
pub const ENDPOINT_ID_CHARS: usize = 80;
/// The web's `MAX_MESSAGE_CHARS`.
pub const MAX_MESSAGE_CHARS: usize = 32_000;

pub const AUTO_LABEL: &str = "Auto";
pub const LOCAL_LABEL: &str = "Local";
pub const CLOUD_LABEL: &str = "Cloud";
pub const DEV_ECHO_LABEL: &str = "Development echo";

/// The sentences the picker and the refusals use. Those marked "web" or
/// "chat spec" are the web's or Appendix A's; the rest describe the managed
/// llama.cpp server and are PROVISIONAL (spec §22.3, row C4).
pub mod words {
    /// PROVISIONAL.
    pub const AUTO_DETAIL: &str = "The model on this machine, run by Lattice with llama.cpp. In this window Auto never leaves the machine.";
    /// PROVISIONAL.
    pub const LOCAL_DETAIL: &str = "llama.cpp on this machine, run by Lattice.";
    /// PROVISIONAL. Appended to Auto's and Local's detail when the web
    /// Lattice is configured to load a Hugging Face model in process.
    pub const HF_NOTE: &str = "If the web Lattice can load the Hugging Face model configured for it, its Auto and Local answer with that model; this window uses llama.cpp on this machine.";
    /// Chat spec §2.4.
    pub const CLOUD_DETAIL: &str = "Anthropic. Not available in the native Lattice yet.";
    /// Chat spec Appendix A.
    pub const CLOUD_REFUSAL: &str = "Cloud (Anthropic) is not in the native Lattice yet. Pick a model on this machine, or a configured endpoint.";
    /// Web (`model_choices`).
    pub const ECHO_DETAIL: &str = "Streams a fixed reply. Not a language model.";
    /// Web (`model_choices`), for an endpoint off this machine.
    pub const REMOTE_ENDPOINT_DETAIL: &str = "Choosing it sends this conversation off the machine.";
    /// Web (`model_choices`), for an endpoint on this machine; here only a
    /// managed row's, which is the core's own server.
    pub const MANAGED_ROW_DETAIL: &str = "Runs on this machine.";
    /// PROVISIONAL. An Ollama row's detail.
    pub const OLLAMA_DETAIL: &str = "Ollama, which Lattice no longer uses.";
    /// PROVISIONAL. Why an Ollama row is refused (ADR-0041).
    pub const OLLAMA_RETIRED: &str =
        "Ollama is no longer used by Lattice; pick Local, which runs llama.cpp on this machine.";
    /// Chat spec Appendix A.
    pub const ANTHROPIC_ENDPOINT: &str = "Anthropic models are not in the native Lattice yet.";
    /// Chat spec Appendix A (ledger row 35).
    pub const MISSING_KEY: &str = "No API key for it is set where this window looks (the environment, or FAMEnvironment.env in the Lattice folder).";
    /// `choices.rs`'s sentences for the other registry statuses.
    pub const TURNED_OFF: &str = "It is turned off in the model registry.";
    pub const NO_ADDRESS: &str = "No server address is set for it.";
    pub const NO_MODEL_NAME: &str = "No model name is set for it.";
    pub const NOT_READY: &str = "It is not ready.";
    /// PROVISIONAL.
    pub const NOT_CONFIGURED: &str = "That model is no longer configured.";
    /// PROVISIONAL.
    pub const UNKNOWN_CHOICE: &str = "That is not a model choice this window knows.";
    /// Chat spec Appendix A, first sentence; PROVISIONAL in what follows it.
    pub const NO_LOCAL_MODEL: &str = "No model on this machine is ready.";

    /// PROVISIONAL. A named endpoint on a loopback address: never Local (LR1).
    pub fn named_loopback_detail(address: &str) -> String {
        if address.is_empty() {
            "A server you named, on this machine. Lattice does not run it, so it counts as off this machine.".to_owned()
        } else {
            format!(
                "A server you named, at {address}. Lattice does not run it, so it counts as off this machine."
            )
        }
    }

    /// PROVISIONAL. Local's detail with the chosen model.
    pub fn local_detail_with(model: &str) -> String {
        format!("llama.cpp on this machine, run by Lattice, with {model}.")
    }
}

/// True when `id` is a choice this window may route, as the web's
/// `valid_provider` decides, less its trailing-newline quirk.
pub fn is_valid_choice(id: &str, development: bool) -> bool {
    if BASE_CHOICES.contains(&id) {
        return true;
    }
    if id == DEV_ECHO {
        return development;
    }
    if crate::acp::Agent::from_choice(id).is_some() {
        return true;
    }
    id.strip_prefix(ENDPOINT_PREFIX).is_some_and(|rest| {
        (1..=ENDPOINT_ID_CHARS).contains(&rest.len())
            && rest
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
    })
}

/// Why the managed server cannot answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LocalProblem {
    Binary(BinaryProblem),
    Model(ModelProblem),
}

impl LocalProblem {
    /// One explicit offline sentence. PROVISIONAL.
    pub fn sentence(self) -> &'static str {
        match self {
            Self::Binary(problem) => crate::llama::binary_sentence(problem),
            Self::Model(problem) => crate::llama::model_sentence(problem),
        }
    }
}

/// The managed server as the disk shows it, and, once the runtime fills
/// them in, whether it runs and how its last start ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalView {
    pub paths: LlamaPaths,
    pub binary: Result<PathBuf, BinaryProblem>,
    pub models: Vec<LocalModel>,
    /// `analyst_model.json`'s `model`, or `""`.
    pub selected: String,
    /// The model the managed server runs now, if one runs.
    pub running: Option<String>,
    /// The last start's failure, in one sentence, when it failed.
    pub failed: Option<String>,
}

impl LocalView {
    /// Read the binary, the models and the selection from the disk now.
    pub fn read(env: &dyn Env, state: &StateRoot) -> Self {
        Self::read_at(LlamaPaths::from_env(env), state)
    }

    /// [`LocalView::read`] for paths the caller decided.
    pub fn read_at(paths: LlamaPaths, state: &StateRoot) -> Self {
        Self {
            binary: find_binary(&paths),
            models: list_models(&paths.models_dir),
            selected: files::selected_model(state),
            paths,
            running: None,
            failed: None,
        }
    }

    /// The model Local would run, or why it cannot run one.
    pub fn model(&self) -> Result<LocalModel, LocalProblem> {
        self.binary
            .as_ref()
            .map_err(|problem| LocalProblem::Binary(*problem))?;
        files::resolve_in(&self.selected, &self.models).map_err(LocalProblem::Model)
    }
}

/// What a choice resolves to.
#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    /// The core's managed llama.cpp server, running this model (Local, Auto).
    Managed(LocalModel),
    /// A registry endpoint picked by name (never Local).
    Endpoint(Box<ModelEndpoint>),
    /// The development echo, which sends nothing.
    Echo,
    /// Nothing can answer; the sentence says why.
    Refused(String),
    /// One of the labs' agents, on the reader's own subscription (`crate::acp`).
    Agent(crate::acp::Agent),
}

/// One choice: what the picker shows, what it resolves to, and the provider
/// name an answer from it is recorded with (Python's naming).
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub choice: ChatChoice,
    pub target: Target,
    pub provider: String,
}

/// The picker's list, and how many registry endpoints are turned off.
#[derive(Clone, Debug, PartialEq)]
pub struct Choices {
    pub entries: Vec<Entry>,
    pub turned_off: usize,
}

/// What Prepare resolves a send's choice to, once (N10).
#[derive(Clone, Debug, PartialEq)]
pub struct Resolution {
    pub id: String,
    pub shown: Shown,
    pub target: Target,
    pub provider: String,
}

impl Resolution {
    /// True only when nothing this choice could send leaves the machine: the
    /// managed server and the echo are Local by construction. A named
    /// endpoint never is, whatever its address.
    pub fn affirmatively_local(&self) -> bool {
        self.shown.locality == Locality::Local && !matches!(self.target, Target::Endpoint(_))
    }
}

impl From<Entry> for Resolution {
    fn from(entry: Entry) -> Self {
        Self {
            id: entry.choice.id,
            shown: Shown {
                locality: entry.choice.locality,
                label: entry.choice.label,
            },
            target: entry.target,
            provider: entry.provider,
        }
    }
}

/// Python's provider name for the managed server (`llamacpp_provider`).
pub fn managed_provider(model: &str) -> String {
    if model.is_empty() {
        "llamacpp:(no model selected)".to_owned()
    } else {
        format!("llamacpp:{model}")
    }
}

/// `providers._configured_hf_model_dir`, reduced to "is one configured": the
/// first of its five variables that is set and not empty decides (blank after
/// `strip()` means no); otherwise `<globals>/hf_model_dir.json`'s
/// `model_dir`. Used only for the detail lines of Auto and Local.
pub fn hf_model_configured(env: &dyn Env, state: &StateRoot) -> bool {
    for name in [
        "ALELYON_HF_MODEL_DIR",
        "ALELYON_HF_MODEL_PATH",
        "HF_MODEL_DIR",
        "HF_MODEL_PATH",
        "MODEL_DIRECTORY",
    ] {
        if let Some(value) = env::text(env, name).filter(|value| !value.is_empty()) {
            return !py::strip(&value).is_empty();
        }
    }
    let Ok(bytes) = fs::read(state.globals.join("hf_model_dir.json")) else {
        return false;
    };
    let Ok(text) = String::from_utf8(bytes) else {
        return false;
    };
    let Ok(raw) = pyjson::loads(&text) else {
        return false;
    };
    if !pyjson::py_truthy(&raw) {
        return false;
    }
    let PyValue::Object(_) = raw else {
        return false;
    };
    match raw.get("model_dir") {
        Some(value) if pyjson::py_truthy(value) => !py::strip(&pyjson::py_str(value)).is_empty(),
        _ => false,
    }
}

fn base_choice(
    id: &str,
    label: &str,
    detail: String,
    locality: Locality,
    refusal: Option<String>,
) -> ChatChoice {
    ChatChoice {
        id: id.to_owned(),
        label: label.to_owned(),
        detail,
        locality,
        ready: refusal.is_none(),
        refusal,
    }
}

/// Auto or Local: the managed server, Local by construction.
fn managed_entry(id: &str, local: &LocalView, hf: bool) -> Entry {
    let model = local.model();
    let mut detail = match (id, &model) {
        (AUTO, _) => words::AUTO_DETAIL.to_owned(),
        (_, Ok(model)) => words::local_detail_with(&model.name),
        _ => words::LOCAL_DETAIL.to_owned(),
    };
    if hf {
        detail.push(' ');
        detail.push_str(words::HF_NOTE);
    }
    let (refusal, target) = match model {
        Ok(model) => (None, Target::Managed(model)),
        Err(problem) => {
            let sentence = if id == AUTO {
                format!("{} {}", words::NO_LOCAL_MODEL, problem.sentence())
            } else {
                problem.sentence().to_owned()
            };
            (Some(sentence.clone()), Target::Refused(sentence))
        }
    };
    let label = if id == AUTO { AUTO_LABEL } else { LOCAL_LABEL };
    Entry {
        choice: base_choice(id, label, detail, Locality::Local, refusal),
        target,
        provider: managed_provider(&local.selected),
    }
}

fn cloud_entry() -> Entry {
    let refusal = words::CLOUD_REFUSAL.to_owned();
    Entry {
        choice: base_choice(
            CLOUD,
            CLOUD_LABEL,
            words::CLOUD_DETAIL.to_owned(),
            Locality::Remote,
            Some(refusal.clone()),
        ),
        target: Target::Refused(refusal),
        provider: String::new(),
    }
}

fn echo_entry() -> Entry {
    Entry {
        choice: base_choice(
            DEV_ECHO,
            DEV_ECHO_LABEL,
            words::ECHO_DETAIL.to_owned(),
            Locality::Local,
            None,
        ),
        target: Target::Echo,
        provider: DEV_ECHO.to_owned(),
    }
}

/// A choice nothing answers: not affirmatively local, no label.
fn unknown(id: &str, sentence: &str) -> Entry {
    Entry {
        choice: base_choice(
            id,
            "",
            String::new(),
            Locality::Remote,
            Some(sentence.to_owned()),
        ),
        target: Target::Refused(sentence.to_owned()),
        provider: String::new(),
    }
}

/// Why a registry endpoint is not ready, in Chat's words; never a key name.
fn endpoint_refusal(endpoint: &ModelEndpoint, keys: &KeyStore) -> Option<String> {
    match endpoint.kind {
        EndpointKind::Ollama => return Some(words::OLLAMA_RETIRED.to_owned()),
        EndpointKind::Anthropic => return Some(words::ANTHROPIC_ENDPOINT.to_owned()),
        // A managed row is [`managed_row_entry`]'s; it never reaches here.
        EndpointKind::Llamacpp => return Some(words::NOT_READY.to_owned()),
        EndpointKind::OpenaiCompatible => {}
    }
    if endpoint.ready(keys) {
        return None;
    }
    let status = endpoint.status(keys);
    Some(
        match status.as_str() {
            "disabled" => words::TURNED_OFF,
            "no server URL set" => words::NO_ADDRESS,
            "no model name set" => words::NO_MODEL_NAME,
            _ if status.starts_with("needs ") => words::MISSING_KEY,
            _ => words::NOT_READY,
        }
        .to_owned(),
    )
}

/// Is a registry endpoint on a loopback address? Only an OpenAI-compatible
/// row's own address counts; no variable is read.
fn on_loopback(endpoint: &ModelEndpoint) -> bool {
    endpoint.kind == EndpointKind::OpenaiCompatible && registry::is_local_url(&endpoint.base_url)
}

/// Does a registry endpoint sort with the ones on this machine? A managed
/// row, and an OpenAI-compatible row on a loopback address (which is still
/// never Local here, LR1; only its place in the list follows the web's).
fn sorts_first(endpoint: &ModelEndpoint) -> bool {
    endpoint.kind == EndpointKind::Llamacpp || on_loopback(endpoint)
}

/// A managed row (`llamacpp` kind): the core's own server under the row's
/// name, Local by construction (LR2), with Python's provider name for it
/// (`endpoint_provider`: `<id>:<model>` after `runtime_endpoint`). It cannot
/// run when the binary or the model it names is missing: refused with the
/// offline sentence, still Local, and nothing is sent.
fn managed_row_entry(
    endpoint: ModelEndpoint,
    env: &dyn Env,
    state: &StateRoot,
    local: &LocalView,
) -> Entry {
    let endpoint = registry::runtime_endpoint(endpoint, env, state);
    let model = match &local.binary {
        Err(problem) => Err(LocalProblem::Binary(*problem)),
        Ok(_) => files::resolve_in(&endpoint.model, &local.models).map_err(LocalProblem::Model),
    };
    let (refusal, target) = match model {
        Ok(model) => (None, Target::Managed(model)),
        Err(problem) => {
            let sentence = problem.sentence().to_owned();
            (Some(sentence.clone()), Target::Refused(sentence))
        }
    };
    Entry {
        choice: base_choice(
            &format!("{ENDPOINT_PREFIX}{}", endpoint.id),
            &endpoint.label,
            words::MANAGED_ROW_DETAIL.to_owned(),
            Locality::Local,
            refusal,
        ),
        target,
        provider: format!("{}:{}", endpoint.id, endpoint.model),
    }
}

fn endpoint_entry(
    endpoint: ModelEndpoint,
    env: &dyn Env,
    state: &StateRoot,
    keys: &KeyStore,
    local: &LocalView,
) -> Entry {
    if endpoint.kind == EndpointKind::Llamacpp {
        return managed_row_entry(endpoint, env, state, local);
    }
    let detail = match endpoint.kind {
        EndpointKind::Ollama => words::OLLAMA_DETAIL.to_owned(),
        _ if on_loopback(&endpoint) => words::named_loopback_detail(&sanitize_url_for_trace(
            &choices::api_base(&endpoint, env),
        )),
        _ => words::REMOTE_ENDPOINT_DETAIL.to_owned(),
    };
    let refusal = endpoint_refusal(&endpoint, keys);
    let provider = format!("{}:{}", endpoint.id, endpoint.model);
    let choice = base_choice(
        &format!("{ENDPOINT_PREFIX}{}", endpoint.id),
        &endpoint.label,
        detail,
        // LR1: a named endpoint is never Local, whatever its address.
        Locality::Remote,
        refusal.clone(),
    );
    let target = match refusal {
        Some(sentence) => Target::Refused(sentence),
        None => Target::Endpoint(Box::new(endpoint)),
    };
    Entry {
        choice,
        target,
        provider,
    }
}

/// The registry's endpoints, in the picker's order. The Ollama projection
/// (`runtime_endpoint`) is not applied: it reads `OLLAMA_*`, and an Ollama row
/// is refused here whatever it names.
fn registry_endpoints(env: &dyn Env, state: &StateRoot) -> Vec<ModelEndpoint> {
    let mut endpoints = registry::load_with_report(&registry::config_path(env, state)).endpoints;
    endpoints
        .sort_by_cached_key(|endpoint| (!sorts_first(endpoint), endpoint.label.to_lowercase()));
    endpoints
}

/// One of the labs' agents (`crate::acp`): it runs off this machine, on the
/// reader's own subscription, and is ready once its adapter is installed.
fn agent_entry(agent: crate::acp::Agent, state: &StateRoot) -> Entry {
    let installed = agent.installed(&crate::acp::agents_dir(state));
    let refusal = (!installed).then(|| format!("{} is not installed: add it in Tools.", agent.label()));
    Entry {
        choice: ChatChoice {
            id: agent.choice().to_owned(),
            label: agent.label().to_owned(),
            detail: "Runs off this machine, on your own account, with its own tools.".to_owned(),
            locality: Locality::Remote,
            ready: installed,
            refusal: refusal.clone(),
        },
        target: match refusal {
            Some(sentence) => Target::Refused(sentence),
            None => Target::Agent(agent),
        },
        provider: agent.choice().to_owned(),
    }
}

/// Every choice the picker lists, in order, each with its target.
pub fn chat_choices(
    env: &dyn Env,
    state: &StateRoot,
    keys: &KeyStore,
    development: bool,
    local: &LocalView,
) -> Choices {
    let hf = hf_model_configured(env, state);
    let mut entries = vec![
        managed_entry(AUTO, local, hf),
        managed_entry(LOCAL, local, hf),
        cloud_entry(),
    ];
    let mut turned_off = 0;
    for endpoint in registry_endpoints(env, state) {
        if endpoint.enabled {
            entries.push(endpoint_entry(endpoint, env, state, keys, local));
        } else {
            turned_off += 1;
        }
    }
    for agent in crate::acp::Agent::ALL {
        entries.push(agent_entry(agent, state));
    }
    if development {
        entries.push(echo_entry());
    }
    Choices {
        entries,
        turned_off,
    }
}

/// What `choice` resolves to now: the one call Prepare makes per send (N10).
/// A choice this window does not route, an endpoint that is gone and the echo
/// outside development resolve to [`Target::Refused`], not Local.
pub fn resolve(
    choice: &str,
    env: &dyn Env,
    state: &StateRoot,
    keys: &KeyStore,
    development: bool,
    local: &LocalView,
) -> Resolution {
    if !is_valid_choice(choice, development) {
        return unknown(choice, words::UNKNOWN_CHOICE).into();
    }
    match choice {
        AUTO | LOCAL => managed_entry(choice, local, hf_model_configured(env, state)).into(),
        CLOUD => cloud_entry().into(),
        DEV_ECHO => echo_entry().into(),
        _ if let Some(agent) = crate::acp::Agent::from_choice(choice) => agent_entry(agent, state).into(),
        _ => {
            let id = &choice[ENDPOINT_PREFIX.len()..];
            match registry_endpoints(env, state)
                .into_iter()
                .find(|endpoint| endpoint.id == id)
            {
                Some(endpoint) => endpoint_entry(endpoint, env, state, keys, local).into(),
                None => unknown(choice, words::NOT_CONFIGURED).into(),
            }
        }
    }
}

/// The local runtime as the status line shows it. The headlines are
/// Python's `ModelState.headline()` where Python has one, and
/// the native ones otherwise; all PROVISIONAL.
pub fn describe_local(local: &LocalView) -> LocalRuntime {
    let installed: Vec<String> = local
        .models
        .iter()
        .map(|model| model.name.clone())
        .collect();
    let models_dir = local.paths.models_dir.display().to_string();
    let model = local.selected.clone();
    let folder = if installed.is_empty() {
        format!("no GGUF files in {models_dir}")
    } else {
        format!("{} model(s) in {models_dir}", installed.len())
    };
    let (state, headline, detail) = match (&local.binary, local.model()) {
        (Err(BinaryProblem::Missing), _) => (
            RuntimeState::Offline,
            "llama.cpp's server is not installed on this machine".to_owned(),
            format!(
                "no llama-server: set {} or install a llama.cpp build into {}",
                files::BINARY_ENV,
                local.paths.llama_dir.display()
            ),
        ),
        (Err(problem), _) => (
            RuntimeState::Offline,
            LocalProblem::Binary(*problem).sentence().to_owned(),
            String::new(),
        ),
        (Ok(_), _) if local.failed.is_some() => {
            let failed = local.failed.clone().unwrap_or_default();
            (
                RuntimeState::Error,
                format!("the local model server failed: {failed}"),
                failed,
            )
        }
        (Ok(_), Err(LocalProblem::Model(ModelProblem::NoneChosen))) => (
            RuntimeState::NoModel,
            format!("no model chosen: put a GGUF file in {models_dir}"),
            folder,
        ),
        (Ok(_), Err(_)) => (
            RuntimeState::NoModel,
            format!("{model} is not in {models_dir}"),
            folder,
        ),
        (Ok(_), Ok(found)) => (
            RuntimeState::Ready,
            format!("{} ready", found.name),
            if local.running.as_deref() == Some(found.name.as_str()) {
                "Running on this machine now.".to_owned()
            } else {
                "Starts on this machine when a question needs it.".to_owned()
            },
        ),
    };
    LocalRuntime {
        state,
        model,
        installed,
        headline,
        detail,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::env::MapEnv;
    use crate::testkit::TempDir;

    pub(crate) struct Fixture {
        pub _dir: TempDir,
        pub env: Arc<MapEnv>,
        pub state: StateRoot,
        pub keys: KeyStore,
    }

    impl Fixture {
        pub(crate) fn new(tag: &str, env: MapEnv) -> Self {
            let dir = TempDir::new(tag);
            let home = dir.path().join("home");
            let env = Arc::new(
                env.with("USERPROFILE", home.as_os_str())
                    .with("HOME", home.as_os_str()),
            );
            let state = StateRoot::at(dir.path().join("root"));
            let keys = KeyStore::new(env.clone(), &state);
            Self {
                _dir: dir,
                env,
                state,
                keys,
            }
        }

        pub(crate) fn paths(&self) -> LlamaPaths {
            LlamaPaths::from_env(self.env.as_ref())
        }

        /// A binary file where the paths say, and a model chosen.
        pub(crate) fn install(&self, model: &str) {
            let paths = self.paths();
            let file = paths.models_dir.join(format!("{model}.gguf"));
            self.within(&paths.binary);
            self.within(&file);
            std::fs::create_dir_all(&paths.llama_dir).unwrap();
            std::fs::write(&paths.binary, b"MZ").unwrap();
            crate::llama::files::tests::gguf(&file);
            self.choose(model);
        }

        /// Refuse a stand-in outside the fixture's own directory.
        #[track_caller]
        pub(crate) fn within(&self, path: &std::path::Path) {
            crate::testkit::assert_within(self._dir.path(), path);
        }

        pub(crate) fn choose(&self, model: &str) {
            std::fs::create_dir_all(&self.state.globals).unwrap();
            std::fs::write(
                self.state.globals.join("analyst_model.json"),
                format!("{{\"model\": \"{model}\"}}"),
            )
            .unwrap();
        }

        pub(crate) fn registry(&self, json: &str) {
            std::fs::create_dir_all(&self.state.globals).unwrap();
            std::fs::write(self.state.globals.join("model_endpoints.json"), json).unwrap();
        }

        pub(crate) fn view(&self) -> LocalView {
            LocalView::read(self.env.as_ref(), &self.state)
        }

        pub(crate) fn choices(&self, development: bool) -> Choices {
            chat_choices(
                self.env.as_ref(),
                &self.state,
                &self.keys,
                development,
                &self.view(),
            )
        }

        pub(crate) fn resolve(&self, choice: &str) -> Resolution {
            resolve(
                choice,
                self.env.as_ref(),
                &self.state,
                &self.keys,
                true,
                &self.view(),
            )
        }
    }

    fn ids(choices: &Choices) -> Vec<&str> {
        choices
            .entries
            .iter()
            .map(|e| e.choice.id.as_str())
            .collect()
    }

    fn entry<'a>(choices: &'a Choices, id: &str) -> &'a Entry {
        choices
            .entries
            .iter()
            .find(|e| e.choice.id == id)
            .unwrap_or_else(|| panic!("no {id}"))
    }

    #[test]
    fn the_vocabulary_is_the_webs() {
        for id in [
            "auto",
            "local",
            "cloud",
            "endpoint:a",
            "endpoint:A-b_c.9",
            &format!("endpoint:{}", "x".repeat(80)),
        ] {
            assert!(is_valid_choice(id, false), "{id}");
        }
        assert!(is_valid_choice("dev:echo", true));
        assert!(!is_valid_choice("dev:echo", false));
        for id in [
            "",
            "AUTO",
            " auto",
            "auto ",
            "dev:scripted",
            "dev:",
            "endpoint:",
            "endpoint:a b",
            "endpoint:a/b",
            "endpoint:é",
            "endpoint:a\n",
            "ollama-local",
            &format!("endpoint:{}", "x".repeat(81)),
        ] {
            assert!(!is_valid_choice(id, true), "{id:?}");
        }
    }

    /// The labs' agents: off this machine, refused until their adapter is
    /// installed in Lattice's agents folder, then the agent itself.
    #[test]
    fn an_agent_is_offered_off_this_machine_once_its_adapter_is_installed() {
        use crate::acp::{Agent, agents_dir};
        let f = Fixture::new("vocab-agents", MapEnv::new());
        let r = f.resolve("agent:codex");
        assert!(matches!(&r.target, Target::Refused(why) if why.contains("add it in Tools")), "{r:?}");
        assert_eq!(r.shown.locality, Locality::Remote);
        assert!(!r.affirmatively_local());
        let exe = agents_dir(&f.state)
            .join("node_modules")
            .join("@agentclientprotocol")
            .join("codex-acp")
            .join("dist");
        std::fs::create_dir_all(&exe).unwrap();
        std::fs::write(exe.join("index.js"), b"// codex-acp").unwrap();
        assert_eq!(f.resolve("agent:codex").target, Target::Agent(Agent::Codex));
        assert!(matches!(f.resolve("agent:claude-code").target, Target::Refused(_)));
        assert!(!is_valid_choice("agent:gemini", false));
    }

    #[test]
    fn a_fresh_machine_lists_auto_local_cloud_and_the_enabled_endpoints() {
        let f = Fixture::new("vocab-fresh", MapEnv::new());
        let choices = f.choices(true);
        let listed = ids(&choices);
        assert_eq!(&listed[..3], ["auto", "local", "cloud"]);
        assert_eq!(*listed.last().unwrap(), "dev:echo");
        // The built-in catalogue enables only the managed llama.cpp row;
        // the rest are off.
        // Then the labs' agents (`crate::acp`), not installed here.
        assert_eq!(
            &listed[3..listed.len() - 1],
            ["endpoint:llamacpp-local", "agent:claude-code", "agent:codex"]
        );
        assert_eq!(choices.turned_off, registry::builtins().len() - 1);
        let without_dev = f.choices(false);
        assert!(!ids(&without_dev).contains(&"dev:echo"));
        for entry in &choices.entries {
            assert_eq!(
                entry.choice.ready,
                entry.choice.refusal.is_none(),
                "{:?}",
                entry.choice
            );
            assert_eq!(
                entry.choice.ready,
                !matches!(entry.target, Target::Refused(_)),
                "{:?}",
                entry.choice
            );
        }
    }

    #[test]
    fn local_and_auto_are_the_managed_server_when_it_can_run() {
        let f = Fixture::new("vocab-ready", MapEnv::new());
        f.install("qwen3-8b");
        let choices = f.choices(false);
        for id in ["auto", "local"] {
            let e = entry(&choices, id);
            assert!(e.choice.ready, "{id}");
            assert_eq!(e.choice.locality, Locality::Local);
            assert_eq!(e.provider, "llamacpp:qwen3-8b");
            match &e.target {
                Target::Managed(model) => assert_eq!(model.name, "qwen3-8b"),
                other => panic!("{id}: {other:?}"),
            }
        }
        assert_eq!(entry(&choices, "local").choice.label, "Local");
        assert_eq!(entry(&choices, "auto").choice.label, "Auto");
        assert_eq!(
            entry(&choices, "local").choice.detail,
            "llama.cpp on this machine, run by Lattice, with qwen3-8b."
        );
        let resolution = f.resolve("local");
        assert!(resolution.affirmatively_local());
        assert_eq!(
            resolution.shown,
            Shown {
                locality: Locality::Local,
                label: "Local".into()
            }
        );
    }

    #[test]
    fn local_and_auto_are_offline_with_a_sentence_when_the_server_cannot_run() {
        let f = Fixture::new("vocab-offline", MapEnv::new());
        let missing = f.resolve("local");
        assert_eq!(
            missing.target,
            Target::Refused(
                LocalProblem::Binary(BinaryProblem::Missing)
                    .sentence()
                    .into()
            )
        );
        assert_eq!(
            missing.shown.locality,
            Locality::Local,
            "still Local: nothing is sent"
        );
        let paths = f.paths();
        f.within(&paths.binary);
        std::fs::create_dir_all(&paths.llama_dir).unwrap();
        std::fs::write(&paths.binary, b"MZ").unwrap();
        assert_eq!(
            f.resolve("local").target,
            Target::Refused(
                LocalProblem::Model(ModelProblem::NoneChosen)
                    .sentence()
                    .into()
            )
        );
        f.choose("not-there");
        let auto = f.resolve("auto");
        assert_eq!(
            auto.target,
            Target::Refused(format!(
                "No model on this machine is ready. {}",
                LocalProblem::Model(ModelProblem::NotFound).sentence()
            ))
        );
        for problem in [
            LocalProblem::Binary(BinaryProblem::Missing),
            LocalProblem::Binary(BinaryProblem::Ollama),
            LocalProblem::Binary(BinaryProblem::NotAbsolute),
            LocalProblem::Binary(BinaryProblem::NotExe),
            LocalProblem::Binary(BinaryProblem::NotLocal),
            LocalProblem::Model(ModelProblem::NoneChosen),
            LocalProblem::Model(ModelProblem::NotFound),
        ] {
            assert!(
                problem
                    .sentence()
                    .starts_with("The local model is offline: "),
                "{problem:?}"
            );
        }
    }

    #[test]
    fn a_named_endpoint_is_never_local_even_on_a_loopback_address() {
        let f = Fixture::new("vocab-named", MapEnv::new());
        f.install("m");
        f.registry(
            r#"{"version": 1, "endpoints": [
                {"id": "mine", "label": "My vLLM", "base_url": "http://127.0.0.1:8000/v1", "model": "q", "enabled": true},
                {"id": "lab", "label": "Lab", "base_url": "https://lab.example.test/v1", "model": "q", "enabled": true}
            ]}"#,
        );
        let choices = f.choices(false);
        let mine = entry(&choices, "endpoint:mine");
        assert_eq!(mine.choice.locality, Locality::Remote);
        assert!(mine.choice.ready);
        assert_eq!(
            mine.choice.detail,
            "A server you named, at http://127.0.0.1:8000/v1. Lattice does not run it, so it counts as off this machine."
        );
        assert_eq!(mine.provider, "mine:q");
        let lab = entry(&choices, "endpoint:lab");
        assert_eq!(lab.choice.detail, words::REMOTE_ENDPOINT_DETAIL);
        assert!(
            ids(&choices).iter().position(|id| *id == "endpoint:mine")
                < ids(&choices).iter().position(|id| *id == "endpoint:lab"),
            "the loopback group first, as the web orders them"
        );
        let resolution = f.resolve("endpoint:mine");
        assert!(!resolution.affirmatively_local());
        assert!(matches!(resolution.target, Target::Endpoint(_)));
    }

    #[test]
    fn an_ollama_row_is_listed_refused_and_never_local() {
        let f = Fixture::new(
            "vocab-ollama",
            MapEnv::new().with("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        );
        // Two saved Ollama rows: the old built-in's id with no address of its
        // own (the built-in itself is gone since the managed server landed), and one of a person's.
        f.registry(
            r#"{"version": 1, "endpoints": [
                {"id": "ollama-local", "label": "Ollama (this machine)", "kind": "ollama", "model": "qwen3", "enabled": true},
                {"id": "mine", "label": "Mine", "kind": "ollama", "base_url": "http://localhost:11434", "model": "m", "enabled": true}
            ]}"#,
        );
        let choices = f.choices(false);
        for id in ["endpoint:ollama-local", "endpoint:mine"] {
            let e = entry(&choices, id);
            assert_eq!(e.choice.locality, Locality::Remote, "{id}");
            assert!(!e.choice.ready, "{id}");
            assert_eq!(
                e.choice.refusal.as_deref(),
                Some(words::OLLAMA_RETIRED),
                "{id}"
            );
            assert_eq!(e.target, Target::Refused(words::OLLAMA_RETIRED.into()));
            let resolution = f.resolve(id);
            assert!(!resolution.affirmatively_local(), "{id}");
        }
    }

    /// A managed row (`llamacpp` kind) is the core's own server under
    /// the row's name: Local by construction (LR2), the built-in row with the
    /// selected model and any other with the model it names, offline with
    /// the sentence when that model cannot run, sorted with the rows on this
    /// machine, and never what Auto or Local resolve to.
    /// Mutants: a managed row taken as a named endpoint (Remote, refused); the
    /// built-in row without `runtime_endpoint` (no model).
    #[test]
    fn a_managed_row_is_the_managed_server_and_local_by_construction() {
        let f = Fixture::new("vocab-managed", MapEnv::new());
        let offline = f.choices(false);
        let builtin = entry(&offline, "endpoint:llamacpp-local");
        assert_eq!(builtin.choice.locality, Locality::Local);
        assert_eq!(
            builtin.target,
            Target::Refused(
                LocalProblem::Binary(BinaryProblem::Missing)
                    .sentence()
                    .into()
            ),
            "no binary: offline, still Local, nothing sent"
        );

        f.install("qwen3-8b");
        crate::llama::files::tests::gguf(&f.paths().models_dir.join("small.gguf"));
        f.registry(
            r#"{"version": 1, "endpoints": [
                {"id": "lab", "label": "A lab", "base_url": "https://lab.example.test/v1", "model": "q", "enabled": true},
                {"id": "mine", "label": "Mine, managed", "kind": "llamacpp", "model": "small", "enabled": true},
                {"id": "gone", "label": "Gone", "kind": "llamacpp", "model": "absent", "enabled": true}
            ]}"#,
        );
        let choices = f.choices(false);
        let builtin = entry(&choices, "endpoint:llamacpp-local");
        assert_eq!(builtin.choice.label, "llama.cpp (this machine, managed)");
        assert_eq!(builtin.choice.locality, Locality::Local);
        assert_eq!(builtin.choice.detail, words::MANAGED_ROW_DETAIL);
        assert!(builtin.choice.ready);
        assert_eq!(builtin.provider, "llamacpp-local:qwen3-8b");
        match &builtin.target {
            Target::Managed(model) => assert_eq!(model.name, "qwen3-8b"),
            other => panic!("the built-in managed row: {other:?}"),
        }
        let mine = entry(&choices, "endpoint:mine");
        assert_eq!(mine.choice.locality, Locality::Local);
        assert_eq!(mine.provider, "mine:small");
        match &mine.target {
            Target::Managed(model) => assert_eq!(model.name, "small"),
            other => panic!("a managed row: {other:?}"),
        }
        let gone = entry(&choices, "endpoint:gone");
        assert_eq!(gone.choice.locality, Locality::Local);
        assert!(!gone.choice.ready);
        assert_eq!(
            gone.target,
            Target::Refused(
                LocalProblem::Model(ModelProblem::NotFound)
                    .sentence()
                    .into()
            )
        );
        let listed = ids(&choices);
        let at = |id: &str| listed.iter().position(|x| *x == id).unwrap();
        assert!(
            at("endpoint:llamacpp-local") < at("endpoint:lab")
                && at("endpoint:mine") < at("endpoint:lab"),
            "managed rows sort with the ones on this machine: {listed:?}"
        );
        let resolution = f.resolve("endpoint:llamacpp-local");
        assert!(resolution.affirmatively_local());
        assert!(matches!(resolution.target, Target::Managed(_)));
        for choice in ["auto", "local"] {
            assert_eq!(
                f.resolve(choice).provider,
                "llamacpp:qwen3-8b",
                "{choice} is the managed entry itself, not a registry row"
            );
        }
    }

    #[test]
    fn refusals_use_chats_words_and_never_name_a_key() {
        let f = Fixture::new("vocab-refusals", MapEnv::new());
        f.registry(
            r#"{"version": 1, "endpoints": [
                {"id": "openai", "enabled": true},
                {"id": "anthropic", "enabled": true},
                {"id": "nomodel", "label": "No model", "base_url": "https://x.example.test/v1", "enabled": true}
            ]}"#,
        );
        let choices = f.choices(false);
        let refusal = |id: &str| entry(&choices, id).choice.refusal.clone().unwrap();
        assert_eq!(refusal("endpoint:openai"), words::MISSING_KEY);
        assert_eq!(refusal("endpoint:anthropic"), words::ANTHROPIC_ENDPOINT);
        assert_eq!(refusal("endpoint:nomodel"), words::NO_MODEL_NAME);
        assert_eq!(refusal("cloud"), words::CLOUD_REFUSAL);
        let everything = serde_json::to_string(
            &choices
                .entries
                .iter()
                .map(|e| &e.choice)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for name in [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "_API_KEY",
            "HF_TOKEN",
        ] {
            assert!(!everything.contains(name), "{name}");
        }
        assert_eq!(
            f.resolve("endpoint:gone").target,
            Target::Refused(words::NOT_CONFIGURED.into())
        );
        assert_eq!(f.resolve("endpoint:gone").shown.locality, Locality::Remote);
        let disabled = f.resolve("endpoint:xai");
        assert_eq!(disabled.target, Target::Refused(words::TURNED_OFF.into()));
        assert_eq!(
            resolve(
                "dev:echo",
                f.env.as_ref(),
                &f.state,
                &f.keys,
                false,
                &f.view()
            )
            .target,
            Target::Refused(words::UNKNOWN_CHOICE.into()),
            "the echo outside development"
        );
    }

    #[test]
    fn the_hugging_face_note_follows_pythons_configuration_rule() {
        let f = Fixture::new("vocab-hf", MapEnv::new());
        assert!(!hf_model_configured(f.env.as_ref(), &f.state));
        for (value, expected) in [("D:\\hf", true), ("   ", false)] {
            let env = MapEnv::new()
                .with("HF_MODEL_DIR", value)
                .with("MODEL_DIRECTORY", "D:\\x");
            assert_eq!(
                hf_model_configured(&env, &f.state),
                expected,
                "{value:?}: the first set variable decides"
            );
        }
        std::fs::create_dir_all(&f.state.globals).unwrap();
        let file = f.state.globals.join("hf_model_dir.json");
        for (text, expected) in [
            (&br#"{"model_dir": "D:\\hf"}"#[..], true),
            (br#"{"model_dir": "  "}"#, false),
            (br#"{"model_dir": 5}"#, true),
            (br#"{"model_dir": 0}"#, false),
            (br#"["D:\\hf"]"#, false),
            (b"null", false),
            (b"\xef\xbb\xbf{\"model_dir\": \"D:\\\\hf\"}", false),
        ] {
            std::fs::write(&file, text).unwrap();
            assert_eq!(
                hf_model_configured(f.env.as_ref(), &f.state),
                expected,
                "{}",
                String::from_utf8_lossy(text)
            );
        }
        std::fs::write(&file, br#"{"model_dir": "D:\\hf"}"#).unwrap();
        let choices = f.choices(false);
        assert!(
            entry(&choices, "auto")
                .choice
                .detail
                .ends_with(words::HF_NOTE)
        );
        assert!(
            entry(&choices, "local")
                .choice
                .detail
                .ends_with(words::HF_NOTE)
        );
    }

    #[test]
    fn the_status_line_describes_the_managed_server() {
        let f = Fixture::new("vocab-status", MapEnv::new());
        let offline = describe_local(&f.view());
        assert_eq!(offline.state, RuntimeState::Offline);
        assert_eq!(
            offline.headline,
            "llama.cpp's server is not installed on this machine"
        );
        f.install("qwen3-8b");
        let mut view = f.view();
        let ready = describe_local(&view);
        assert_eq!(ready.state, RuntimeState::Ready);
        assert_eq!(ready.headline, "qwen3-8b ready");
        assert_eq!(ready.installed, ["qwen3-8b"]);
        assert_eq!(
            ready.detail,
            "Starts on this machine when a question needs it."
        );
        view.running = Some("qwen3-8b".into());
        assert_eq!(describe_local(&view).detail, "Running on this machine now.");
        view.failed = Some("the server stopped while it loaded the model".into());
        assert_eq!(describe_local(&view).state, RuntimeState::Error);
        f.choose("");
        let none = describe_local(&f.view());
        assert_eq!(none.state, RuntimeState::NoModel);
        assert!(
            none.headline
                .starts_with("no model chosen: put a GGUF file in ")
        );
        f.choose("other");
        let missing = describe_local(&f.view());
        assert_eq!(missing.state, RuntimeState::NoModel);
        assert!(missing.headline.starts_with("other is not in "));
    }
}
