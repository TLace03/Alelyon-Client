//! Hosted open-weight models: the services that run open models for the reader,
//! such as OpenRouter and Hugging Face.
//!
//! Every provider here speaks the OpenAI-compatible API the chat already uses
//! for an endpoint of the registry, so a provider is a row of the registry with
//! its address filled in, and nothing new answers a turn:
//! - [`PROVIDERS`]: OpenRouter, Hugging Face (Inference Providers), Together,
//!   Groq, Fireworks, DeepInfra and Cerebras, each with where its key is made
//!   and how the reader connects ([`SignIn`]): OpenRouter's own sign-in (a key
//!   the reader approves in the browser and can revoke there, [`openrouter`]),
//!   else a key pasted once;
//! - [`list_models`]: the provider's `/models`, read with the reader's key when
//!   there is one, kept to the open-weight chat models ([`open_weight_chat`]);
//! - [`connect`]: the key to Windows Credential Manager (`keys::vault_store`,
//!   where lattice-core keeps keys typed into Alelyon) and the provider's
//!   registry row enabled; [`choose`]: the row's model set.
//!
//! The registry's catalogue (`registry::builtins`) is the Python registry's, and
//! its parity fixtures pin it, so the two providers it lacks (DeepInfra,
//! Cerebras) are written as rows of the reader's own when first connected.
//!
//! Lattice reads no other program's stored login: a key is the reader's paste
//! or OpenRouter's answer to their approval. A key is never logged, shown or
//! put in an error.

pub mod openrouter;
#[cfg(test)]
mod tests;

use std::path::Path;
use std::time::Duration;

use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::Value;

use crate::keys::{self, SecretString};
use crate::registry::{self, EndpointKind, ModelEndpoint};

/// How the reader connects a provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignIn {
    /// OpenRouter's sign-in: approved in the browser, a key comes back ([`openrouter`]).
    OpenRouter,
    /// A key made on the provider's page and pasted once.
    PasteKey,
}

/// A service that hosts open models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provider {
    /// Its registry row's id (a built-in's where the catalogue has one).
    pub id: &'static str,
    pub label: &'static str,
    /// The OpenAI-compatible base address.
    pub base_url: &'static str,
    /// The name its key is kept under.
    pub key_name: &'static str,
    /// Where the reader makes a key.
    pub key_page: &'static str,
    pub sign_in: SignIn,
    /// Whether its model list is read without a key.
    pub lists_without_key: bool,
    /// One line about it, for the reader.
    pub about: &'static str,
}

/// The providers, in the order they are offered. Addresses and pages are as
/// each provider documents them (2026-10-09).
pub const PROVIDERS: &[Provider] = &[
    Provider {
        id: "openrouter",
        label: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        key_name: "OPENROUTER_API_KEY",
        key_page: "https://openrouter.ai/settings/keys",
        sign_in: SignIn::OpenRouter,
        lists_without_key: true,
        about: "Hundreds of models from many hosts behind one account; sign in and approve a key in the browser.",
    },
    Provider {
        id: "huggingface",
        label: "Hugging Face",
        base_url: "https://router.huggingface.co/v1",
        key_name: "HF_TOKEN",
        key_page: "https://huggingface.co/settings/tokens",
        sign_in: SignIn::PasteKey,
        lists_without_key: true,
        about: "Models on the Hugging Face Hub, served by its inference partners; a token with \"Make calls to Inference Providers\".",
    },
    Provider {
        id: "together",
        label: "Together AI",
        base_url: "https://api.together.xyz/v1",
        key_name: "TOGETHER_API_KEY",
        key_page: "https://api.together.ai/settings/api-keys",
        sign_in: SignIn::PasteKey,
        lists_without_key: false,
        about: "Open models served on Together's own GPUs.",
    },
    Provider {
        id: "groq",
        label: "Groq",
        base_url: "https://api.groq.com/openai/v1",
        key_name: "GROQ_API_KEY",
        key_page: "https://console.groq.com/keys",
        sign_in: SignIn::PasteKey,
        lists_without_key: false,
        about: "Open models on Groq's own chips, known for speed.",
    },
    Provider {
        id: "fireworks",
        label: "Fireworks AI",
        base_url: "https://api.fireworks.ai/inference/v1",
        key_name: "FIREWORKS_API_KEY",
        key_page: "https://fireworks.ai/account/api-keys",
        sign_in: SignIn::PasteKey,
        lists_without_key: false,
        about: "Open models served by Fireworks.",
    },
    Provider {
        id: "deepinfra",
        label: "DeepInfra",
        base_url: "https://api.deepinfra.com/v1/openai",
        key_name: "DEEPINFRA_API_KEY",
        key_page: "https://deepinfra.com/dash/api_keys",
        sign_in: SignIn::PasteKey,
        lists_without_key: false,
        about: "Open models served by DeepInfra, priced per token.",
    },
    Provider {
        id: "cerebras",
        label: "Cerebras",
        base_url: "https://api.cerebras.ai/v1",
        key_name: "CEREBRAS_API_KEY",
        key_page: "https://cloud.cerebras.ai/",
        sign_in: SignIn::PasteKey,
        lists_without_key: false,
        about: "Open models on Cerebras' wafer-scale chips, known for speed.",
    },
];

/// The provider with this id.
pub fn provider(id: &str) -> Option<&'static Provider> {
    PROVIDERS.iter().find(|p| p.id == id)
}

/// A model a provider hosts.
#[derive(Clone, Debug, PartialEq)]
pub struct HostedModel {
    /// The id a request names.
    pub id: String,
    /// Its display name, when the provider gives one, else the id.
    pub name: String,
    /// Its context window in tokens, when said.
    pub context: Option<u64>,
    /// US dollars per million input and output tokens, when said (OpenRouter).
    pub price: Option<(f64, f64)>,
}

// ------------------------------------------------------------------- HTTP

/// The most a provider's answer may hold.
pub const MAX_BODY: usize = 16 * 1024 * 1024;
const CONNECT_WAIT: Duration = Duration::from_secs(15);
const READ_WAIT: Duration = Duration::from_secs(60);

/// What a request came back with: its status and body.
pub type Reply = Result<(u16, Vec<u8>), String>;

/// The seam to the provider: the real one is [`Web`], a test's records.
pub trait Http: Send + Sync {
    fn get(&self, url: String, key: Option<SecretString>) -> BoxFuture<'static, Reply>;
    fn post_json(&self, url: String, body: Value) -> BoxFuture<'static, Reply>;
}

/// Requests off this machine, as the chat's own model client makes them: no
/// redirect, bounded waits, the system's proxy, a bounded body. An error is
/// one sentence, never the transport's text.
pub struct Web {
    client: reqwest::Client,
}

impl Web {
    pub fn new() -> Result<Web, String> {
        reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_WAIT)
            .read_timeout(READ_WAIT)
            .pool_max_idle_per_host(0)
            .user_agent(concat!("Lattice/", env!("CARGO_PKG_VERSION")))
            .build()
            .map(|client| Web { client })
            .map_err(|_| "Lattice could not set up its connection to the provider.".to_owned())
    }
}

async fn read(response: Result<reqwest::Response, reqwest::Error>) -> Reply {
    let mut response = response.map_err(|_| "The provider could not be reached.".to_owned())?;
    let status = response.status().as_u16();
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| "The provider's answer was cut off.".to_owned())?
    {
        if body.len() + chunk.len() > MAX_BODY {
            return Err("The provider's answer was too large.".to_owned());
        }
        body.extend_from_slice(&chunk);
    }
    Ok((status, body))
}

impl Http for Web {
    fn get(&self, url: String, key: Option<SecretString>) -> BoxFuture<'static, Reply> {
        let mut request = self.client.get(url).header("Accept", "application/json");
        if let Some(key) = key {
            request = request.bearer_auth(key.expose());
        }
        async move { read(request.send().await).await }.boxed()
    }

    fn post_json(&self, url: String, body: Value) -> BoxFuture<'static, Reply> {
        let request = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .body(body.to_string());
        async move { read(request.send().await).await }.boxed()
    }
}

/// What a status says, in the reader's words.
pub(crate) fn status_sentence(provider: &str, status: u16) -> String {
    match status {
        401 | 403 => {
            format!("{provider} did not accept the key: make a new one and connect again.")
        }
        402 => format!("{provider} says the account has no credit left."),
        429 => format!("{provider} is limiting requests: try again in a minute."),
        _ => format!("{provider} answered with an error ({status})."),
    }
}

// ------------------------------------------------------------------- models

/// The provider's models, open-weight chat models only, by name.
pub async fn list_models(
    http: &dyn Http,
    provider: &Provider,
    key: Option<SecretString>,
) -> Result<Vec<HostedModel>, String> {
    if key.is_none() && !provider.lists_without_key {
        return Err(format!(
            "Connect {} first: its model list needs your key.",
            provider.label
        ));
    }
    let (status, body) = http
        .get(format!("{}/models", provider.base_url), key)
        .await?;
    if status != 200 {
        return Err(status_sentence(provider.label, status));
    }
    let value: Value = serde_json::from_slice(&body)
        .map_err(|_| format!("{}'s model list could not be read.", provider.label))?;
    parse_models(provider, &value)
        .ok_or_else(|| format!("{}'s model list could not be read.", provider.label))
}

/// The models of a `/models` answer (`{data: [...]}`, or a bare list as
/// Together answers), kept to open-weight chat models and sorted by name.
pub fn parse_models(provider: &Provider, value: &Value) -> Option<Vec<HostedModel>> {
    let rows = match value {
        Value::Array(rows) => rows,
        Value::Object(map) => map.get("data")?.as_array()?,
        _ => return None,
    };
    let mut models: Vec<HostedModel> = rows
        .iter()
        .filter(|row| open_weight_chat(provider, row))
        .filter_map(|row| {
            let id = row.get("id")?.as_str()?.trim();
            if id.is_empty() {
                return None;
            }
            let name = row
                .get("name")
                .or_else(|| row.get("display_name"))
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .unwrap_or(id);
            let context = ["context_length", "context_window", "max_context_length"]
                .iter()
                .find_map(|k| row.get(*k).and_then(Value::as_u64));
            Some(HostedModel {
                id: id.to_owned(),
                name: name.to_owned(),
                context,
                price: price(row),
            })
        })
        .collect();
    models.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then(a.id.cmp(&b.id))
    });
    models.dedup_by(|a, b| a.id == b.id);
    Some(models)
}

/// OpenRouter's prices (US dollars per token, as text) per million tokens.
fn price(row: &Value) -> Option<(f64, f64)> {
    let pricing = row.get("pricing")?;
    let per = |k: &str| -> Option<f64> {
        let v = pricing.get(k)?;
        let n = v.as_f64().or_else(|| v.as_str()?.trim().parse().ok())?;
        (n.is_finite() && n >= 0.0).then_some(n * 1_000_000.0)
    };
    Some((per("prompt")?, per("completion")?))
}

/// Words in an id that name a model that is not a chat model.
const NOT_CHAT: &[&str] = &[
    "whisper",
    "tts",
    "embed",
    "rerank",
    "flux",
    "stable-diffusion",
    "sdxl",
    "orpheus",
    "playai",
    "moderation",
    "image",
    "vision-embed",
];

/// Whether a row is an open-weight chat model:
/// - OpenRouter lists closed models too, and names an open one's weights on the
///   Hub (`hugging_face_id`), so only those are kept;
/// - Together says each row's `type`, so only `chat` ones are kept;
/// - every other provider here hosts open models only; what is not a chat model
///   (speech, embeddings, images) is left out by its id.
pub fn open_weight_chat(provider: &Provider, row: &Value) -> bool {
    let id = row
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();
    if NOT_CHAT.iter().any(|w| id.contains(w)) {
        return false;
    }
    if provider.id == "openrouter" {
        return row
            .get("hugging_face_id")
            .and_then(Value::as_str)
            .is_some_and(|h| !h.trim().is_empty());
    }
    match row.get("type").and_then(Value::as_str) {
        Some(kind) => kind.eq_ignore_ascii_case("chat") || kind.eq_ignore_ascii_case("language"),
        None => true,
    }
}

// ------------------------------------------------------------------- connecting

/// The provider's registry row: the reader's, if the file has one, else the
/// catalogue's, else a new row of the reader's own.
fn row(entries: &[ModelEndpoint], provider: &Provider) -> ModelEndpoint {
    if let Some(found) = entries.iter().find(|e| e.id == provider.id) {
        return found.clone();
    }
    if let Some(builtin) = registry::builtins()
        .into_iter()
        .find(|e| e.id == provider.id)
    {
        return builtin;
    }
    ModelEndpoint {
        id: provider.id.to_owned(),
        label: provider.label.to_owned(),
        kind: EndpointKind::OpenaiCompatible,
        base_url: provider.base_url.to_owned(),
        model: String::new(),
        api_key_name: provider.key_name.to_owned(),
        enabled: false,
        builtin: false,
        note: provider.about.to_owned(),
    }
}

/// The registry's rows as they are now, or why they cannot be changed.
fn entries(path: &Path) -> Result<Vec<ModelEndpoint>, String> {
    let report = registry::load_with_report(path);
    if !report.complete() {
        return Err(
            "The model list file could not be read whole, so it was not changed.".to_owned(),
        );
    }
    Ok(report.endpoints)
}

fn write(path: &Path, endpoint: ModelEndpoint) -> Result<(), String> {
    registry::upsert(path, endpoint)
        .map_err(|_| "The model list file could not be written.".to_owned())
}

/// The reader's key kept in Credential Manager under the provider's key name,
/// and the provider's row enabled at its own address.
pub fn connect(path: &Path, provider: &Provider, key: &SecretString) -> Result<(), String> {
    if key.expose().trim().is_empty() {
        return Err("Paste the key first.".to_owned());
    }
    keys::vault_store(provider.key_name, key)?;
    enable(path, provider)
}

/// The provider's row enabled at its own address, under its key name.
pub fn enable(path: &Path, provider: &Provider) -> Result<(), String> {
    let mut endpoint = row(&entries(path)?, provider);
    endpoint.base_url = provider.base_url.to_owned();
    endpoint.api_key_name = provider.key_name.to_owned();
    endpoint.enabled = true;
    write(path, endpoint)
}

/// The provider's row set to answer with this model (and enabled).
pub fn choose(path: &Path, provider: &Provider, model: &str) -> Result<(), String> {
    let model = model.trim();
    if model.is_empty() {
        return Err("Choose a model first.".to_owned());
    }
    let mut endpoint = row(&entries(path)?, provider);
    endpoint.base_url = provider.base_url.to_owned();
    if endpoint.api_key_name.is_empty() {
        endpoint.api_key_name = provider.key_name.to_owned();
    }
    endpoint.model = model.to_owned();
    endpoint.enabled = true;
    write(path, endpoint)
}

/// Whether a key for the provider is found (kept by Alelyon, or in the
/// environment or an env file, as every key of the chat is found).
pub fn connected(provider: &Provider) -> bool {
    keys::get_key(provider.key_name).is_some()
}
