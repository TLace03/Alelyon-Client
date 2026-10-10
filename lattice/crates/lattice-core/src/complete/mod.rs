//! Tab completions in the editor (tool parity with Cursor's Tab and Copilot's
//! inline suggestions): as the reader types, a code model writes what comes
//! next at the cursor, which the editor shows in grey and Tab accepts.
//!
//! - **Which model** is the reader's choice ([`settings`]), from every model
//!   that can do it ([`catalog`]): a GGUF file on this PC, run by the core's
//!   own llama.cpp server, or a fill-in-the-middle model of a connected
//!   provider. Each is shown with what a thousand completions cost, and
//!   grouped by it.
//! - **A request** is the text before the cursor (at most [`PREFIX_CHARS`])
//!   and after it (at most [`SUFFIX_CHARS`]), asked for at most
//!   [`MAX_TOKENS`] tokens, greedily, waited for at most [`WAIT`]. The answer
//!   is cut at a stop token, at [`MAX_LINES`] lines, and where it starts to
//!   repeat the text after the cursor ([`clean`]).
//! - **Locally**, through llama.cpp's `/infill`, on the loopback address with
//!   the server's own token. A completion never switches the server's model
//!   ([`ManagedRuntime::open_unswitched`]): while a chat's Local model runs,
//!   a completion with another model is refused rather than stopping that
//!   answer.
//! - **From a provider**, through its `/completions` with the family's own
//!   prompt ([`fim`]) and the reader's key. Text that looks like a key is
//!   never sent: such a request is refused before anything leaves.

pub mod catalog;
pub mod fim;
pub mod settings;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::{Value, json};

use crate::env::Env;
use crate::hosted::{self, Reply};
use crate::keys::{KeyStore, SecretString};
use crate::llama::ManagedRuntime;
use crate::llama::files::{self, LlamaPaths};
use crate::net::{self, LoopbackHttp};
use crate::secrets;
use crate::state::StateRoot;

use catalog::Offer;
use settings::{Choice, Settings};

/// The most text before the cursor a request sends, in characters.
pub const PREFIX_CHARS: usize = 6_000;
/// The most text after the cursor a request sends, in characters.
pub const SUFFIX_CHARS: usize = 2_000;
/// The most tokens a completion asks for.
pub const MAX_TOKENS: u32 = 64;
/// The most lines a completion keeps.
pub const MAX_LINES: usize = 12;
/// The longest a completion is waited for.
pub const WAIT: Duration = Duration::from_secs(10);

/// What the editor asks about: the text before and after the cursor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Request {
    pub prefix: String,
    pub suffix: String,
}

impl Request {
    /// The end of the prefix and the start of the suffix that a request sends:
    /// whole lines where the limit allows.
    pub fn clipped(&self) -> (&str, &str) {
        let prefix = tail(&self.prefix, PREFIX_CHARS);
        let prefix = match prefix.len() < self.prefix.len() {
            true => prefix.find('\n').map_or(prefix, |at| &prefix[at + 1..]),
            false => prefix,
        };
        let suffix = head(&self.suffix, SUFFIX_CHARS);
        let suffix = match suffix.len() < self.suffix.len() {
            true => suffix.rfind('\n').map_or(suffix, |at| &suffix[..=at]),
            false => suffix,
        };
        (prefix, suffix)
    }
}

/// The last `chars` characters of `text`.
fn tail(text: &str, chars: usize) -> &str {
    match text.char_indices().rev().nth(chars.saturating_sub(1)) {
        Some((at, _)) if chars > 0 => &text[at..],
        _ if chars == 0 => "",
        _ => text,
    }
}

/// The first `chars` characters of `text`.
fn head(text: &str, chars: usize) -> &str {
    match text.char_indices().nth(chars) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

/// A completion as the editor shows it, from what the model answered: cut at
/// any stop token, at [`MAX_LINES`] lines, before a line that repeats the
/// first line after the cursor, and without the rest of the cursor's line
/// when it ends with it. `None` when nothing is left.
pub fn clean(raw: &str, suffix: &str) -> Option<String> {
    let mut text = raw.to_owned();
    for stop in fim::all_stops() {
        if let Some(at) = text.find(stop) {
            text.truncate(at);
        }
    }
    let (rest, after) = suffix.split_once('\n').unwrap_or((suffix, ""));
    if let Some(next) = after.lines().map(str::trim).find(|l| !l.is_empty()) {
        let mut offset = 0;
        for (i, line) in text.split_inclusive('\n').enumerate() {
            if i > 0 && line.trim() == next {
                text.truncate(offset);
                break;
            }
            offset += line.len();
        }
    }
    if let Some(at) = text.match_indices('\n').nth(MAX_LINES - 1).map(|(at, _)| at) {
        text.truncate(at);
    }
    let mut text = text.trim_end().to_owned();
    let rest = rest.trim_end();
    if !rest.trim().is_empty() && !text.contains('\n') && text.ends_with(rest) {
        text.truncate(text.len() - rest.len());
    }
    (!text.trim().is_empty()).then_some(text)
}

/// The seam to the network: [`Net`], or a test's records.
pub trait Wire: Send + Sync {
    /// GET with the reader's key, when there is one (a provider's model list).
    fn get(&self, url: String, key: Option<SecretString>) -> BoxFuture<'static, Reply>;
    /// POST JSON with a key: the local server's token or the reader's key.
    fn post(&self, url: String, key: SecretString, body: Value) -> BoxFuture<'static, Reply>;
}

/// The real network: the local server through the loopback client (no proxy,
/// 127.0.0.1 only), a provider as the hosted models' client reaches it.
pub struct Net {
    web: hosted::Web,
    loopback: Arc<LoopbackHttp>,
}

impl Net {
    pub fn new() -> Result<Net, String> {
        let loopback = LoopbackHttp::new().map_err(|_| "Lattice could not set up its local connection.".to_owned())?;
        Ok(Net { web: hosted::Web::new()?, loopback: Arc::new(loopback) })
    }
}

impl Wire for Net {
    fn get(&self, url: String, key: Option<SecretString>) -> BoxFuture<'static, Reply> {
        hosted::Http::get(&self.web, url, key)
    }

    fn post(&self, url: String, key: SecretString, body: Value) -> BoxFuture<'static, Reply> {
        if net::is_loopback_url(&url) {
            let loopback = self.loopback.clone();
            return async move {
                loopback
                    .post_json(&url, Some(&key), &body, WAIT)
                    .await
                    .map(|r| (r.status, r.body))
                    .map_err(|_| "The local model server did not answer.".to_owned())
            }
            .boxed();
        }
        self.web.post_keyed(url, key, body, WAIT)
    }
}

/// The editor's completions.
pub struct Completer {
    env: Arc<dyn Env>,
    state: StateRoot,
    local: Arc<dyn ManagedRuntime>,
    wire: Arc<dyn Wire>,
}

impl Completer {
    pub fn new(env: Arc<dyn Env>, state: StateRoot, local: Arc<dyn ManagedRuntime>, wire: Arc<dyn Wire>) -> Self {
        Completer { env, state, local, wire }
    }

    /// The reader's settings.
    pub fn settings(&self) -> Settings {
        settings::load(&settings::file(&self.state))
    }

    pub fn save(&self, settings: &Settings) -> Result<(), String> {
        settings::save(&settings::file(&self.state), settings)
    }

    fn key(&self, name: &str) -> Option<SecretString> {
        KeyStore::new(self.env.clone(), &self.state).get(name)
    }

    /// Every model that can write completions, and a sentence for each
    /// provider whose list could not be read.
    pub async fn offers(&self) -> (Vec<Offer>, Vec<String>) {
        let paths = LlamaPaths::from_env(self.env.as_ref());
        let mut offers = catalog::local_offers(&files::list_models(&paths.models_dir));
        let mut notes = Vec::new();
        let asks = catalog::PROVIDERS.iter().filter_map(|id| hosted::provider(id)).filter_map(|provider| {
            let key = self.key(provider.key_name);
            if key.is_none() && !provider.lists_without_key {
                return None;
            }
            let ready = key.is_some();
            let reply = self.wire.get(format!("{}/models", provider.base_url), key);
            Some(async move { (provider, ready, reply.await) })
        });
        for (provider, ready, reply) in futures::future::join_all(asks).await {
            let unreadable = || format!("{}'s model list could not be read.", provider.label);
            match reply {
                Ok((200, body)) => match serde_json::from_slice::<Value>(&body)
                    .ok()
                    .and_then(|value| catalog::hosted_offers(provider, &value, ready))
                {
                    Some(found) => offers.extend(found),
                    None => notes.push(unreadable()),
                },
                Ok((status, _)) => notes.push(hosted::status_sentence(provider.label, status)),
                Err(why) => notes.push(format!("{}: {why}", provider.label)),
            }
        }
        (offers, notes)
    }

    /// What `choice` writes at the cursor, or why nothing was asked or
    /// answered. `Ok(None)` when the model had nothing to add.
    pub async fn complete(&self, choice: &Choice, request: &Request) -> Result<Option<String>, String> {
        let (prefix, suffix) = request.clipped();
        match choice {
            Choice::Local { model } => {
                let paths = LlamaPaths::from_env(self.env.as_ref());
                let found = files::resolve_model(model, &paths.models_dir)
                    .map_err(|_| format!("{model} is not in the models folder."))?;
                let opened = self.local.open_unswitched(found).await.map_err(|e| e.sentence().to_owned())?;
                let body = json!({
                    "input_prefix": prefix,
                    "input_suffix": suffix,
                    "n_predict": MAX_TOKENS,
                    "temperature": 0.0,
                    "cache_prompt": true,
                });
                let url = format!("{}/infill", opened.endpoint.base_url);
                let (status, body) = self.wire.post(url, opened.endpoint.token.clone(), body).await?;
                drop(opened);
                if status != 200 {
                    return Err(format!(
                        "The local model could not fill in the middle ({status}): choose a code model made for it."
                    ));
                }
                let value: Value = serde_json::from_slice(&body)
                    .map_err(|_| "The local model's answer could not be read.".to_owned())?;
                Ok(value.get("content").and_then(Value::as_str).and_then(|text| clean(text, suffix)))
            }
            Choice::Hosted { provider, model } => {
                let provider = hosted::provider(provider)
                    .filter(|p| catalog::PROVIDERS.contains(&p.id))
                    .ok_or_else(|| "That provider does not offer completions.".to_owned())?;
                let family =
                    fim::family(model).ok_or_else(|| format!("{model} is not a fill-in-the-middle model."))?;
                if secrets::looks_like_secret(prefix) || secrets::looks_like_secret(suffix) {
                    return Err(format!(
                        "The code near the cursor looks like it holds a key, so nothing was sent to {}.",
                        provider.label
                    ));
                }
                let key = self.key(provider.key_name).ok_or_else(|| format!("Connect {} first.", provider.label))?;
                let stops: Vec<&str> = family.stops().iter().copied().take(4).collect();
                let body = json!({
                    "model": model,
                    "prompt": family.prompt(prefix, suffix),
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "stop": stops,
                });
                let (status, body) = self.wire.post(format!("{}/completions", provider.base_url), key, body).await?;
                if status != 200 {
                    return Err(hosted::status_sentence(provider.label, status));
                }
                let value: Value = serde_json::from_slice(&body)
                    .map_err(|_| format!("{}'s answer could not be read.", provider.label))?;
                let text = value.pointer("/choices/0/text").and_then(Value::as_str);
                Ok(text.and_then(|text| clean(text, suffix)))
            }
        }
    }
}
