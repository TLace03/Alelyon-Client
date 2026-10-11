//! Routes for Sinai's parts on the person's own free provider accounts: a
//! mind (chat), eyes (an image with a question), ears (speech to text), a
//! voice (text to speech) and embeddings (text to vectors).
//!
//! Each provider is a [`Cloud`]: OpenRouter, Groq, Cloudflare Workers AI and
//! Gemini. Each person connects their own account; Alelyon holds no key of its
//! own. A [`Reach`] is one provider with the person's key (and, for
//! Cloudflare, their account id) filled in. The routes are plain functions over
//! a [`Client`]:
//! - [`chat`]: OpenAI-compatible `POST <base>/chat/completions`, not streamed;
//! - [`look`]: the same with an image (a `data:` URL in an `image_url` part),
//!   or Cloudflare's own model route with the image's bytes;
//! - [`transcribe`]: OpenAI-compatible `POST <base>/audio/transcriptions`,
//!   multipart (`file`, `model`, `response_format=verbose_json`, word and
//!   segment times);
//! - [`speak`]: Cloudflare's own model route (`{"prompt", "lang"}`; MP3 back,
//!   raw or base64 in its JSON envelope), else OpenAI-compatible
//!   `POST <base>/audio/speech`;
//! - [`embed`]: OpenAI-compatible `POST <base>/embeddings`;
//! - [`models`]: the provider's live model list for a part, kept to the free
//!   ones where the provider says which are free.
//!
//! **Free only.** This version calls free models only: an OpenRouter model
//! must be a free variant (its id ends in `:free`), and [`free_model`] is
//! checked before every call. A provider's free limit is never worked around:
//! a `429` (or Cloudflare's daily-allocation error) is [`Failure::Limited`]
//! with the time the provider gives for trying again, which the caller turns
//! into a paused part. Nothing here retries, switches provider or pays.
//!
//! **Cloudflare's meter.** Every Cloudflare call is counted against the account's free daily allocation by
//! [`super::meter`] before it goes (refused as [`Failure::Limited`] until 00:00 UTC when it could pass 9,000 of the
//! 10,000 Neurons) and after it answers; a Cloudflare model the meter has no rate for is neither listed nor called,
//! and a Cloudflare reach without a meter calls nothing.
//!
//! **Keys.** A key is read where lattice-core finds keys (Credential Manager,
//! the environment, the env files), sent only as the `Authorization` header to
//! its own provider's address, and never logged, shown or put in an error.
//!
//! **Stubs.** A debug build honours [`STUB_ENV`] (an `http://` address on this
//! machine only): every provider's origin is replaced by it, path kept, so
//! tests and screenshots reach a stub server and never a provider. A release
//! build ignores it.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use super::HostedModel;
use super::meter::{self, Meter};
use crate::keys::SecretString;

// ------------------------------------------------------------------- providers

/// How a person connects a provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Connect {
    /// OpenRouter's sign-in in the browser (`hosted::openrouter`), or a pasted key.
    OpenRouterSignIn,
    /// A key made on the provider's page, pasted once.
    PasteKey,
    /// An API token and the account id it belongs to (Cloudflare).
    TokenAndAccount,
}

/// A provider a part can run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cloud {
    pub id: &'static str,
    pub label: &'static str,
    /// The OpenAI-compatible base address; `{account}` stands for the account id.
    pub base: &'static str,
    /// The provider's own model routes (`<run>/<model>`), where it has them.
    pub run: Option<&'static str>,
    /// Where its model list is read.
    pub list: &'static str,
    /// The name its key is kept under.
    pub key_name: &'static str,
    /// The name its account id is kept under, where its addresses need one.
    pub account_name: Option<&'static str>,
    /// Where the person makes a key.
    pub key_page: &'static str,
    pub connect: Connect,
    /// Its data policy in one line, as its own pages say it (read 2026-10-10).
    pub policy: &'static str,
    /// Its free tier uses what is sent to improve its products.
    pub trains: bool,
}

pub const OPENROUTER: Cloud = Cloud {
    id: "openrouter",
    label: "OpenRouter",
    base: "https://openrouter.ai/api/v1",
    run: None,
    list: "https://openrouter.ai/api/v1/models",
    key_name: "OPENROUTER_API_KEY",
    account_name: None,
    key_page: "https://openrouter.ai/settings/keys",
    connect: Connect::OpenRouterSignIn,
    policy: "OpenRouter does not store prompts unless your account opts in; each free model's own host has its own policy.",
    trains: false,
};

pub const GROQ: Cloud = Cloud {
    id: "groq",
    label: "Groq",
    base: "https://api.groq.com/openai/v1",
    run: None,
    list: "https://api.groq.com/openai/v1/models",
    key_name: "GROQ_API_KEY",
    account_name: None,
    key_page: "https://console.groq.com/keys",
    connect: Connect::PasteKey,
    policy: "Groq does not keep what you send by default (at most 30 days for abuse checks) and does not train on it without your permission.",
    trains: false,
};

pub const CLOUDFLARE: Cloud = Cloud {
    id: "cloudflare",
    label: "Cloudflare Workers AI",
    base: "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
    run: Some("https://api.cloudflare.com/client/v4/accounts/{account}/ai/run"),
    list: "https://api.cloudflare.com/client/v4/accounts/{account}/ai/models/search",
    key_name: "CLOUDFLARE_API_TOKEN",
    account_name: Some("CLOUDFLARE_ACCOUNT_ID"),
    key_page: "https://dash.cloudflare.com/profile/api-tokens",
    connect: Connect::TokenAndAccount,
    policy: "Cloudflare does not train models on what you send to Workers AI, and keeps it only if you also use one of its storage services.",
    trains: false,
};

pub const GEMINI: Cloud = Cloud {
    id: "gemini",
    label: "Gemini",
    base: "https://generativelanguage.googleapis.com/v1beta/openai",
    run: None,
    list: "https://generativelanguage.googleapis.com/v1beta/openai/models",
    key_name: "GEMINI_API_KEY",
    account_name: None,
    key_page: "https://aistudio.google.com/apikey",
    connect: Connect::PasteKey,
    policy: "Gemini's free tier uses what you send to improve Google's products, with human review, and is not offered for apps in the EEA, the UK or Switzerland.",
    trains: true,
};

/// The providers parts can run on, in the order they are offered.
pub const CLOUDS: &[Cloud] = &[OPENROUTER, GROQ, CLOUDFLARE, GEMINI];

/// The provider with this id.
pub fn cloud(id: &str) -> Option<&'static Cloud> {
    CLOUDS.iter().find(|c| c.id == id)
}

/// What a part asks a model to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Task {
    /// Think and answer (the mind).
    Chat,
    /// Answer about an image (the eyes).
    Vision,
    /// Speech to text (the ears).
    Transcribe,
    /// Text to speech (the voice).
    Speak,
    /// Text to vectors (the embeddings).
    Embed,
}

/// Whether this version may call `model` of `cloud`: free models only. An
/// OpenRouter model must be a free variant; the other providers here are
/// called on the person's free plan, which their own pages say cannot be
/// charged (Cloudflare bills past its free allocation only on Workers Paid).
pub fn free_model(cloud: &Cloud, model: &str) -> bool {
    let model = model.trim();
    if model.is_empty() {
        return false;
    }
    match cloud.id {
        "openrouter" => model.ends_with(":free"),
        _ => true,
    }
}

// ------------------------------------------------------------------- reaching one

/// In a debug build, an `http://` address on this machine that stands in for
/// every provider (tests and screenshots).
pub const STUB_ENV: &str = "ALELYON_CLOUD_STUB";

/// One provider with the person's key and account filled in.
#[derive(Clone)]
pub struct Reach {
    pub cloud: &'static Cloud,
    base: String,
    run: Option<String>,
    list: String,
    key: SecretString,
    /// A stub on this machine: no proxy.
    direct: bool,
    /// The account id (Cloudflare's), as the meter counts it.
    account: String,
    /// Cloudflare's free-allocation meter: no Cloudflare call goes without one.
    meter: Option<Arc<Meter>>,
}

impl std::fmt::Debug for Reach {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reach")
            .field("cloud", &self.cloud.id)
            .field("base", &self.base)
            .finish_non_exhaustive()
    }
}

/// `origin` (`http://127.0.0.1:<port>`) when it is an `http://` address on
/// this machine with no path, credentials, query or fragment.
pub fn loopback_origin(origin: &str) -> Option<String> {
    let url = url::Url::parse(origin.trim()).ok()?;
    let host = url.host_str()?;
    let loopback = matches!(host, "127.0.0.1" | "localhost" | "[::1]" | "::1");
    let bare = url.path() == "/" && url.query().is_none() && url.fragment().is_none();
    (url.scheme() == "http" && loopback && bare && url.username().is_empty() && url.password().is_none())
        .then(|| url.origin().ascii_serialization())
}

fn restub(address: &str, stub: Option<&str>) -> String {
    match stub {
        Some(origin) => match url::Url::parse(address) {
            Ok(url) => format!("{origin}{}", url.path()),
            Err(_) => address.to_owned(),
        },
        None => address.to_owned(),
    }
}

impl Reach {
    /// `cloud` with `key` (and `account`, where its addresses need one), at its
    /// own addresses, or at `stub`'s origin (see [`loopback_origin`]).
    pub fn new(
        cloud: &'static Cloud,
        key: SecretString,
        account: Option<&str>,
        stub: Option<&str>,
    ) -> Result<Reach, String> {
        if key.expose().trim().is_empty() {
            return Err(format!("Connect {} first: no key is kept for it.", cloud.label));
        }
        let account = match cloud.account_name {
            Some(_) => {
                let account = account.map(str::trim).unwrap_or("");
                // an id is a path segment: letters and digits only, so it cannot change the address
                if account.is_empty() || account.len() > 64 || !account.chars().all(|c| c.is_ascii_alphanumeric()) {
                    return Err(format!(
                        "Connect {} first: its account id is missing or not an account id.",
                        cloud.label
                    ));
                }
                account
            }
            None => "",
        };
        let stub = match stub {
            Some(s) => Some(
                loopback_origin(s)
                    .ok_or_else(|| format!("{STUB_ENV} must be an http:// address on this machine."))?,
            ),
            None => None,
        };
        let fill = |t: &str| restub(&t.replace("{account}", account), stub.as_deref());
        Ok(Reach {
            cloud,
            base: fill(cloud.base),
            run: cloud.run.map(fill),
            list: fill(cloud.list),
            key: SecretString::new(key.expose().trim()),
            direct: stub.is_some(),
            account: account.to_owned(),
            meter: None,
        })
    }

    /// `cloud` with the key and account found by `get` (a key store's lookup),
    /// and, in a debug build, [`STUB_ENV`].
    pub fn found(cloud: &'static Cloud, get: &dyn Fn(&str) -> Option<SecretString>) -> Result<Reach, String> {
        let key = get(cloud.key_name).ok_or_else(|| format!("Connect {} first: no key is kept for it.", cloud.label))?;
        let account = cloud.account_name.and_then(get);
        let stub = if cfg!(debug_assertions) {
            std::env::var(STUB_ENV).ok().filter(|s| !s.trim().is_empty())
        } else {
            None
        };
        let reach = Reach::new(cloud, key, account.as_ref().map(SecretString::expose), stub.as_deref())?;
        // Cloudflare's calls are metered in this user's own file
        Ok(match (cloud.id == CLOUDFLARE.id, meter::default_path()) {
            (true, Some(path)) => reach.with_meter(Arc::new(Meter::at(path))),
            _ => reach,
        })
    }

    /// The same, counted by `meter` (Cloudflare's free allocation).
    pub fn with_meter(mut self, meter: Arc<Meter>) -> Reach {
        self.meter = Some(meter);
        self
    }

    /// The meter that counts this reach's calls, where it has one.
    pub fn meter(&self) -> Option<&Meter> {
        self.meter.as_deref()
    }

    /// The account id the meter counts under.
    pub fn account(&self) -> &str {
        &self.account
    }

    /// The OpenAI-compatible base address.
    pub fn base(&self) -> &str {
        &self.base
    }

    /// Whether it is a stub on this machine.
    pub fn is_stub(&self) -> bool {
        self.direct
    }
}

/// Whether a key for `cloud` (and its account id) is found by `get`.
pub fn connected(cloud: &Cloud, get: &dyn Fn(&str) -> Option<SecretString>) -> bool {
    get(cloud.key_name).is_some() && cloud.account_name.is_none_or(|n| get(n).is_some())
}

// ------------------------------------------------------------------- answers

/// Why a route has no answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    /// The free limit is reached: try again at this time (seconds since 1970).
    Limited { back_at: i64 },
    /// The provider did not accept the key, in one sentence.
    Key(String),
    /// Anything else, in one sentence.
    Failed(String),
}

impl Failure {
    pub fn sentence(&self) -> String {
        match self {
            Failure::Limited { .. } => "The free limit is reached.".to_owned(),
            Failure::Key(s) | Failure::Failed(s) => s.clone(),
        }
    }
}

/// What the provider said about its limits, where it said it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Limits {
    /// Requests left in the provider's window (Groq's is a day).
    pub remaining: Option<u64>,
    /// The window's size in requests.
    pub limit: Option<u64>,
    /// When the window starts again (seconds since 1970).
    pub resets_at: Option<i64>,
}

/// An answer and what the provider said about its limits with it.
#[derive(Clone, Debug, PartialEq)]
pub struct Answer<T> {
    pub value: T,
    pub limits: Limits,
}

/// A reply as the routes read it: status, the headers they use, the body.
#[derive(Clone, Debug, Default)]
pub struct Got {
    pub status: u16,
    /// Lower-case names: `content-type`, `retry-after` and `x-ratelimit-*` only.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Got {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.trim())
    }
}

/// A duration as Groq writes it (`2m59.56s`, `7.66s`, `1h2m`, `120ms`), in seconds.
pub fn duration_text(text: &str) -> Option<f64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if let Ok(n) = text.parse::<f64>() {
        return (n.is_finite() && n >= 0.0).then_some(n);
    }
    let mut total = 0.0;
    let mut number = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() || c == '.' {
            number.push(c);
            continue;
        }
        let n: f64 = number.parse().ok()?;
        number.clear();
        let unit = match (c, chars.peek()) {
            ('m', Some('s')) => {
                chars.next();
                0.001
            }
            ('h', _) => 3600.0,
            ('m', _) => 60.0,
            ('s', _) => 1.0,
            _ => return None,
        };
        total += n * unit;
    }
    number.is_empty().then_some(total)
}

/// An HTTP date (`Sun, 06 Nov 1994 08:49:37 GMT`), in seconds since 1970.
fn http_date(text: &str) -> Option<i64> {
    time::OffsetDateTime::parse(text.trim(), &time::format_description::well_known::Rfc2822)
        .ok()
        .map(|t| t.unix_timestamp())
}

/// A reset written as a number: milliseconds since 1970 (OpenRouter), seconds
/// since 1970, or seconds from now.
fn reset_number(n: f64, now: i64) -> i64 {
    if n > 1e12 {
        (n / 1000.0).ceil() as i64
    } else if n > 1e9 {
        n.ceil() as i64
    } else {
        now + n.ceil() as i64
    }
}

/// The next 00:00 UTC after `now` (Cloudflare's free allocation starts again then).
pub fn next_utc_midnight(now: i64) -> i64 {
    (now.div_euclid(86_400) + 1) * 86_400
}

/// When to try again after a refusal for a limit: `Retry-After` (seconds or a
/// date), else the provider's reset header, else Cloudflare's next midnight
/// (UTC), else a minute from `now`. Never earlier than `now`.
pub fn back_at(cloud: &Cloud, got: &Got, now: i64) -> i64 {
    let after = got.header("retry-after").and_then(|v| {
        v.parse::<f64>()
            .ok()
            .filter(|n| n.is_finite() && *n >= 0.0)
            .map(|n| now + n.ceil() as i64)
            .or_else(|| http_date(v))
    });
    let reset = || {
        ["x-ratelimit-reset-requests", "x-ratelimit-reset"]
            .iter()
            .find_map(|name| got.header(name))
            .and_then(|v| match v.parse::<f64>() {
                Ok(n) if n.is_finite() && n >= 0.0 => Some(reset_number(n, now)),
                _ => duration_text(v).map(|s| now + s.ceil() as i64),
            })
    };
    let at = after.or_else(reset).unwrap_or_else(|| {
        if cloud.id == CLOUDFLARE.id { next_utc_midnight(now) } else { now + 60 }
    });
    at.max(now)
}

/// The limits a reply's headers say (Groq on every reply, OpenRouter on a refusal).
pub fn limits(got: &Got, now: i64) -> Limits {
    let number = |names: &[&str]| names.iter().find_map(|n| got.header(n)).and_then(|v| v.parse::<u64>().ok());
    let resets_at = ["x-ratelimit-reset-requests", "x-ratelimit-reset"]
        .iter()
        .find_map(|name| got.header(name))
        .and_then(|v| match v.parse::<f64>() {
            Ok(n) if n.is_finite() && n >= 0.0 => Some(reset_number(n, now)),
            _ => duration_text(v).map(|s| now + s.ceil() as i64),
        });
    Limits {
        remaining: number(&["x-ratelimit-remaining-requests", "x-ratelimit-remaining"]),
        limit: number(&["x-ratelimit-limit-requests", "x-ratelimit-limit"]),
        resets_at,
    }
}

/// Cloudflare's error for a used-up daily free allocation (its code 4006).
fn allocation_spent(cloud: &Cloud, body: &[u8]) -> bool {
    if cloud.id != CLOUDFLARE.id {
        return false;
    }
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|v| v.get("errors")?.as_array().cloned())
        .is_some_and(|errors| errors.iter().any(|e| e.get("code").and_then(Value::as_u64) == Some(4006)))
}

/// A reply checked: success as it is, a limit as [`Failure::Limited`], the
/// rest in one sentence (never the reply's text, the key or the address).
pub fn checked(cloud: &Cloud, got: Got, now: i64) -> Result<Got, Failure> {
    let label = cloud.label;
    match got.status {
        200..=299 => Ok(got),
        429 => Err(Failure::Limited { back_at: back_at(cloud, &got, now) }),
        _ if allocation_spent(cloud, &got.body) => {
            Err(Failure::Limited { back_at: back_at(cloud, &Got { headers: got.headers, ..Got::default() }, now) })
        }
        401 | 403 => Err(Failure::Key(format!(
            "{label} did not accept the key: make a new one and connect again."
        ))),
        402 => Err(Failure::Failed(format!(
            "{label} asks for payment for this: Alelyon uses free models only, so nothing was paid."
        ))),
        404 => Err(Failure::Failed(format!("{label} does not offer that model at that address."))),
        status => Err(Failure::Failed(format!("{label} answered with an error ({status})."))),
    }
}

// ------------------------------------------------------------------- the client

/// The longest a route waits for a whole answer.
pub const WAIT: Duration = Duration::from_secs(90);
const CONNECT_WAIT: Duration = Duration::from_secs(15);
/// The most audio, or the largest image, sent in one request.
pub const MAX_UPLOAD: usize = 20 * 1024 * 1024;

/// What a request carries.
pub enum Payload {
    Json(Value),
    /// A multipart form: its fields, then one file (`field`, file name, type, bytes).
    Form { fields: Vec<(String, String)>, file: (String, String, String, Vec<u8>) },
}

/// Requests to a provider: no redirect, bounded waits, the system's proxy
/// (none for a stub on this machine), a bounded body.
pub struct Client {
    http: reqwest::Client,
}

impl Client {
    pub fn new(reach: &Reach) -> Result<Client, String> {
        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_WAIT)
            .pool_max_idle_per_host(0)
            .user_agent(concat!("Alelyon/", env!("CARGO_PKG_VERSION")));
        if reach.direct {
            builder = builder.no_proxy();
        }
        builder
            .build()
            .map(|http| Client { http })
            .map_err(|_| "Alelyon could not set up its connection to the provider.".to_owned())
    }

    async fn read(cloud: &Cloud, response: Result<reqwest::Response, reqwest::Error>) -> Result<Got, Failure> {
        let label = cloud.label;
        let mut response = response.map_err(|e| {
            Failure::Failed(if e.is_timeout() {
                format!("{label} did not answer in time.")
            } else {
                format!("{label} could not be reached.")
            })
        })?;
        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .filter(|(n, _)| {
                let n = n.as_str();
                n == "content-type" || n == "retry-after" || n.starts_with("x-ratelimit-")
            })
            .filter_map(|(n, v)| Some((n.as_str().to_owned(), v.to_str().ok()?.to_owned())))
            .collect();
        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| Failure::Failed(format!("{label}'s answer was cut off.")))?
        {
            if body.len() + chunk.len() > super::MAX_BODY {
                return Err(Failure::Failed(format!("{label}'s answer was too large.")));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(Got { status, headers, body })
    }

    /// POST `payload` to `url` with the reach's key.
    pub async fn post(&self, reach: &Reach, url: &str, payload: Payload, wait: Duration) -> Result<Got, Failure> {
        let request = self.http.post(url).timeout(wait).bearer_auth(reach.key.expose());
        let request = match payload {
            Payload::Json(body) => request.header("Content-Type", "application/json").body(body.to_string()),
            Payload::Form { fields, file } => {
                let (boundary, body) = multipart(&fields, &file);
                request.header("Content-Type", format!("multipart/form-data; boundary={boundary}")).body(body)
            }
        };
        Client::read(reach.cloud, request.send().await).await
    }

    /// GET `url` with the reach's key.
    pub async fn get(&self, reach: &Reach, url: &str, wait: Duration) -> Result<Got, Failure> {
        let request = self
            .http
            .get(url)
            .timeout(wait)
            .bearer_auth(reach.key.expose())
            .header("Accept", "application/json");
        Client::read(reach.cloud, request.send().await).await
    }
}

/// A multipart form body and its boundary.
pub fn multipart(fields: &[(String, String)], file: &(String, String, String, Vec<u8>)) -> (String, Vec<u8>) {
    let boundary = format!("alelyon-{}", uuid::Uuid::new_v4().simple());
    let mut body = Vec::with_capacity(file.3.len() + 1024);
    // a value cannot end the part early: CR and LF are dropped from names and values
    let clean = |s: &str| s.replace(['\r', '\n', '"'], " ");
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{}\"\r\n\r\n{}\r\n", clean(name), value.replace(['\r', '\n'], " "))
                .as_bytes(),
        );
    }
    let (field, file_name, mime, bytes) = file;
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\nContent-Type: {}\r\n\r\n",
            clean(field),
            clean(file_name),
            clean(mime)
        )
        .as_bytes(),
    );
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (boundary, body)
}

// ------------------------------------------------------------------- base64

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64, padded.
pub fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Standard base64 (padding and whitespace optional) back to bytes.
pub fn unbase64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0;
    for c in text.bytes().filter(|c| !c.is_ascii_whitespace() && *c != b'=') {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        };
        acc = (acc << 6) | u32::from(v);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Some(out)
}

// ------------------------------------------------------------------- the routes

/// Before a Cloudflare call: its worst case, `most`, reserved under the meter's lock (refused when it could pass the
/// stop today, or when the meter cannot be locked, read or written). Other providers are not metered (None).
fn meter_before(reach: &Reach, model: &str, most: meter::Use, now: i64) -> Result<Option<meter::Reservation>, Failure> {
    if reach.cloud.id != CLOUDFLARE.id {
        return Ok(None);
    }
    let Some(m) = &reach.meter else {
        return Err(Failure::Failed(
            "Cloudflare is called only with its free-allocation meter, and this PC has no place for it.".to_owned(),
        ));
    };
    let Some(cost) = meter::neurons(model, most) else {
        return Err(Failure::Failed(format!("{model} has no published rate the meter knows, so it is not called.")));
    };
    m.reserve(&reach.account, cost, now).map(Some).map_err(|r| match r {
        meter::Refusal::Spent { back_at } => Failure::Limited { back_at },
        meter::Refusal::Meter(why) => Failure::Failed(why),
    })
}

/// After a Cloudflare call that answered: its reservation settled to what it used. A call that failed is never
/// settled, so its worst case stays counted; a settle that cannot be written does the same.
fn meter_after(reach: &Reach, model: &str, reserved: Option<meter::Reservation>, used: meter::Use) {
    if let (Some(r), Some(m), Some(cost)) = (reserved, &reach.meter, meter::neurons(model, used)) {
        let _ = m.settle(&r, cost);
    }
}

/// The token counts an answer reports (`usage`, or Cloudflare's `result.usage`): prompt and completion.
fn usage(value: &Value) -> Option<(u64, u64)> {
    let u = value.get("usage").or_else(|| value.get("result")?.get("usage"))?;
    let prompt = u.get("prompt_tokens").or_else(|| u.get("total_tokens")).and_then(Value::as_u64)?;
    Some((prompt, u.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0)))
}

fn json_of(cloud: &Cloud, got: &Got) -> Result<Value, Failure> {
    serde_json::from_slice(&got.body).map_err(|_| Failure::Failed(format!("{}'s answer could not be read.", cloud.label)))
}

fn unreadable(cloud: &Cloud) -> Failure {
    Failure::Failed(format!("{}'s answer could not be read.", cloud.label))
}

fn refuse_paid(reach: &Reach, model: &str) -> Result<(), Failure> {
    if free_model(reach.cloud, model) {
        Ok(())
    } else {
        Err(Failure::Failed(format!(
            "{model} is not a free model of {}: Alelyon calls free models only.",
            reach.cloud.label
        )))
    }
}

/// A chat completion's text: `choices[0].message.content` (a string, or a list of text parts).
fn completion_text(value: &Value) -> Option<String> {
    let content = value.get("choices")?.get(0)?.get("message")?.get("content")?;
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => Some(
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
        ),
        _ => None,
    }
}

/// The mind: `messages` (OpenAI's chat shape) to `model`, the whole answer back.
pub async fn chat(
    client: &Client,
    reach: &Reach,
    model: &str,
    messages: Vec<Value>,
    max_tokens: u32,
    now: i64,
) -> Result<Answer<String>, Failure> {
    refuse_paid(reach, model)?;
    let tokens_in = meter::tokens_for(Value::Array(messages.clone()).to_string().len());
    let reserved = meter_before(reach, model, meter::Use { input_tokens: tokens_in, output_tokens: u64::from(max_tokens), ..meter::Use::default() }, now)?;
    let body = json!({"model": model, "messages": messages, "max_tokens": max_tokens, "stream": false});
    let got = checked(reach.cloud, client.post(reach, &format!("{}/chat/completions", reach.base), Payload::Json(body), WAIT).await?, now)?;
    let limits = limits(&got, now);
    let answer = json_of(reach.cloud, &got)?;
    let value = completion_text(&answer).ok_or_else(|| unreadable(reach.cloud))?;
    let (used_in, used_out) = usage(&answer).unwrap_or((tokens_in, meter::tokens_for(value.len())));
    meter_after(reach, model, reserved, meter::Use { input_tokens: used_in, output_tokens: used_out, ..meter::Use::default() });
    Ok(Answer { value, limits })
}

/// The eyes: `prompt` about one image (`mime`, such as `image/png`), the answer back.
#[allow(clippy::too_many_arguments)]
pub async fn look(
    client: &Client,
    reach: &Reach,
    model: &str,
    prompt: &str,
    image: &[u8],
    mime: &str,
    max_tokens: u32,
    now: i64,
) -> Result<Answer<String>, Failure> {
    refuse_paid(reach, model)?;
    if image.is_empty() || image.len() > MAX_UPLOAD {
        return Err(Failure::Failed("The image is empty or too large to send.".to_owned()));
    }
    if !mime.starts_with("image/") {
        return Err(Failure::Failed("That is not an image.".to_owned()));
    }
    match &reach.run {
        // Cloudflare's own route takes the image as its bytes
        Some(run) => {
            let tokens_in = meter::tokens_for(prompt.len());
            let most = meter::Use { input_tokens: tokens_in, output_tokens: u64::from(max_tokens), images: 1, ..meter::Use::default() };
            let reserved = meter_before(reach, model, most, now)?;
            let body = json!({
                "messages": [{"role": "user", "content": prompt}],
                "image": image,
                "max_tokens": max_tokens,
            });
            let got = checked(reach.cloud, client.post(reach, &format!("{run}/{model}"), Payload::Json(body), WAIT).await?, now)?;
            let limits = limits(&got, now);
            let answer = json_of(reach.cloud, &got)?;
            let value = answer
                .get("result")
                .and_then(|r| r.get("response"))
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| unreadable(reach.cloud))?;
            // the provider's own counts include the image where it gives them
            let used = match usage(&answer) {
                Some((i, o)) => meter::Use { input_tokens: i, output_tokens: o, ..meter::Use::default() },
                None => meter::Use { input_tokens: tokens_in, output_tokens: meter::tokens_for(value.len()), images: 1, ..meter::Use::default() },
            };
            meter_after(reach, model, reserved, used);
            Ok(Answer { value, limits })
        }
        None => {
            let messages = vec![json!({"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": format!("data:{mime};base64,{}", base64(image))}},
            ]})];
            chat(client, reach, model, messages, max_tokens, now).await
        }
    }
}

/// Accept a model's licence for the person's account where the provider asks
/// for it first (Cloudflare's Llama 3.2 Vision: a request whose prompt is
/// `agree`). Only after the person agreed in the app.
pub async fn accept_licence(client: &Client, reach: &Reach, model: &str, now: i64) -> Result<(), Failure> {
    let Some(run) = &reach.run else { return Ok(()) };
    let most = meter::Use { input_tokens: meter::tokens_for(5), output_tokens: 256, ..meter::Use::default() };
    let reserved = meter_before(reach, model, most, now)?;
    let body = json!({"prompt": "agree"});
    let got = checked(reach.cloud, client.post(reach, &format!("{run}/{model}"), Payload::Json(body), WAIT).await?, now)?;
    let used = serde_json::from_slice::<Value>(&got.body).ok().as_ref().and_then(usage);
    let used = used.map_or(most, |(i, o)| meter::Use { input_tokens: i, output_tokens: o, ..meter::Use::default() });
    meter_after(reach, model, reserved, used);
    Ok(())
}

/// A word heard, with its time in seconds from the start of the audio.
#[derive(Clone, Debug, PartialEq)]
pub struct HeardWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

/// A stretch heard, as Whisper's `verbose_json` gives it.
#[derive(Clone, Debug, PartialEq)]
pub struct HeardSegment {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub no_speech_prob: f32,
    pub avg_logprob: f32,
}

/// What a transcription heard.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Heard {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<HeardSegment>,
    pub words: Vec<HeardWord>,
}

/// OpenAI's `verbose_json` transcription (top-level `text`, `language`,
/// `segments` and `words`), or a plain `{"text"}`.
pub fn parse_heard(value: &Value) -> Option<Heard> {
    let text = value.get("text")?.as_str()?.trim().to_owned();
    let num = |v: &Value, k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0);
    let segments = value
        .get("segments")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|s| HeardSegment {
            start: num(s, "start"),
            end: num(s, "end"),
            text: s.get("text").and_then(Value::as_str).unwrap_or("").to_owned(),
            no_speech_prob: num(s, "no_speech_prob") as f32,
            avg_logprob: num(s, "avg_logprob") as f32,
        })
        .collect();
    let words = value
        .get("words")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|w| {
            let word = w.get("word")?.as_str()?.trim();
            (!word.is_empty()).then(|| HeardWord { word: word.to_owned(), start: num(w, "start"), end: num(w, "end") })
        })
        .collect();
    Some(Heard {
        text,
        language: value.get("language").and_then(Value::as_str).map(str::to_owned),
        segments,
        words,
    })
}

/// The ears: `audio` (a file's bytes, such as a WAV) to text, with word and
/// segment times. `language` None or `auto` lets the model tell.
#[allow(clippy::too_many_arguments)]
pub async fn transcribe(
    client: &Client,
    reach: &Reach,
    model: &str,
    audio: &[u8],
    file_name: &str,
    mime: &str,
    language: Option<&str>,
    prompt: Option<&str>,
    now: i64,
) -> Result<Answer<Heard>, Failure> {
    refuse_paid(reach, model)?;
    if reach.cloud.id == CLOUDFLARE.id {
        return Err(Failure::Failed("Speech to text on Cloudflare is not offered in this version.".to_owned()));
    }
    if audio.is_empty() || audio.len() > MAX_UPLOAD {
        return Err(Failure::Failed("The audio is empty or too large to send.".to_owned()));
    }
    let mut fields: Vec<(String, String)> = vec![
        ("model".into(), model.to_owned()),
        ("response_format".into(), "verbose_json".into()),
        ("temperature".into(), "0".into()),
        ("timestamp_granularities[]".into(), "word".into()),
        ("timestamp_granularities[]".into(), "segment".into()),
    ];
    if let Some(lang) = language.map(str::trim).filter(|l| !l.is_empty() && *l != "auto") {
        fields.push(("language".into(), lang.to_owned()));
    }
    if let Some(p) = prompt.map(str::trim).filter(|p| !p.is_empty()) {
        fields.push(("prompt".into(), p.to_owned()));
    }
    let payload = Payload::Form { fields, file: ("file".into(), file_name.to_owned(), mime.to_owned(), audio.to_vec()) };
    let got = checked(reach.cloud, client.post(reach, &format!("{}/audio/transcriptions", reach.base), payload, WAIT).await?, now)?;
    let limits = limits(&got, now);
    let value = parse_heard(&json_of(reach.cloud, &got)?).ok_or_else(|| unreadable(reach.cloud))?;
    Ok(Answer { value, limits })
}

/// Speech made: its bytes and their type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Speech {
    pub bytes: Vec<u8>,
    pub mime: String,
}

/// The voice: `text` spoken by `model`. Cloudflare's own route takes
/// `{"prompt", "lang"}` and answers MP3 (raw, or base64 as `result.audio`);
/// elsewhere OpenAI's `audio/speech` with `voice` and a WAV answer.
pub async fn speak(
    client: &Client,
    reach: &Reach,
    model: &str,
    text: &str,
    voice: Option<&str>,
    lang: Option<&str>,
    now: i64,
) -> Result<Answer<Speech>, Failure> {
    refuse_paid(reach, model)?;
    if text.trim().is_empty() {
        return Err(Failure::Failed("There is nothing to say.".to_owned()));
    }
    let (url, body) = match &reach.run {
        Some(run) => (format!("{run}/{model}"), json!({"prompt": text, "lang": lang.unwrap_or("en")})),
        None => {
            let mut body = json!({"model": model, "input": text, "response_format": "wav"});
            if let Some(v) = voice {
                body["voice"] = json!(v);
            }
            (format!("{}/audio/speech", reach.base), body)
        }
    };
    let spoken = meter::Use { audio_seconds: meter::speech_seconds(text), ..meter::Use::default() };
    let reserved = meter_before(reach, model, spoken, now)?;
    let got = checked(reach.cloud, client.post(reach, &url, Payload::Json(body), WAIT).await?, now)?;
    meter_after(reach, model, reserved, spoken);
    let limits = limits(&got, now);
    let kind = got.header("content-type").unwrap_or("").to_ascii_lowercase();
    let speech = if kind.starts_with("audio/") {
        Speech { mime: kind.split(';').next().unwrap_or("audio/mpeg").trim().to_owned(), bytes: got.body }
    } else {
        let value = json_of(reach.cloud, &got)?;
        let audio = value
            .get("result")
            .and_then(|r| r.get("audio"))
            .or_else(|| value.get("audio"))
            .and_then(Value::as_str)
            .and_then(unbase64)
            .filter(|b| !b.is_empty())
            .ok_or_else(|| unreadable(reach.cloud))?;
        Speech { bytes: audio, mime: "audio/mpeg".into() }
    };
    Ok(Answer { value: speech, limits })
}

/// The embeddings: one vector per input, in the inputs' order.
pub async fn embed(
    client: &Client,
    reach: &Reach,
    model: &str,
    inputs: &[String],
    now: i64,
) -> Result<Answer<Vec<Vec<f32>>>, Failure> {
    refuse_paid(reach, model)?;
    if inputs.is_empty() {
        return Ok(Answer { value: Vec::new(), limits: Limits::default() });
    }
    let tokens_in: u64 = inputs.iter().map(|i| meter::tokens_for(i.len())).sum();
    let reserved = meter_before(reach, model, meter::Use { input_tokens: tokens_in, ..meter::Use::default() }, now)?;
    let body = json!({"model": model, "input": inputs});
    let got = checked(reach.cloud, client.post(reach, &format!("{}/embeddings", reach.base), Payload::Json(body), WAIT).await?, now)?;
    let limits = limits(&got, now);
    let value = json_of(reach.cloud, &got)?;
    let used_in = usage(&value).map_or(tokens_in, |(i, _)| i);
    meter_after(reach, model, reserved, meter::Use { input_tokens: used_in, ..meter::Use::default() });
    let mut rows: Vec<(usize, Vec<f32>)> = value
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| unreadable(reach.cloud))?
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let index = row.get("index").and_then(Value::as_u64).map_or(i, |n| n as usize);
            let vector = row
                .get("embedding")
                .and_then(Value::as_array)
                .map(|v| v.iter().filter_map(Value::as_f64).map(|x| x as f32).collect::<Vec<_>>())
                .unwrap_or_default();
            (index, vector)
        })
        .collect();
    rows.sort_by_key(|(i, _)| *i);
    if rows.len() != inputs.len() || rows.iter().any(|(_, v)| v.is_empty()) {
        return Err(unreadable(reach.cloud));
    }
    Ok(Answer { value: rows.into_iter().map(|(_, v)| v).collect(), limits })
}

// ------------------------------------------------------------------- live lists

/// Cloudflare's task names for its model search.
fn cloudflare_task(task: Task) -> &'static str {
    match task {
        Task::Chat | Task::Vision => "Text Generation",
        Task::Transcribe => "Automatic Speech Recognition",
        Task::Speak => "Text-to-Speech",
        Task::Embed => "Text Embeddings",
    }
}

fn zero_price(row: &Value) -> bool {
    let Some(pricing) = row.get("pricing") else { return false };
    ["prompt", "completion"].iter().all(|k| {
        pricing
            .get(*k)
            .and_then(|v| v.as_f64().or_else(|| v.as_str()?.trim().parse().ok()))
            .is_some_and(|n: f64| n == 0.0)
    })
}

/// Whether a list row is a model `cloud` offers free for `task`.
pub fn offers(cloud: &Cloud, task: Task, row: &Value) -> bool {
    // Cloudflare names a model by `name` (its `id` is an opaque one); the others by `id`
    let named = if cloud.id == CLOUDFLARE.id { row.get("name") } else { row.get("id") };
    let id = named
        .or_else(|| row.get("id"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();
    if id.is_empty() {
        return false;
    }
    let has = |words: &[&str]| words.iter().any(|w| id.contains(w));
    match cloud.id {
        "openrouter" => {
            let open = row.get("hugging_face_id").and_then(Value::as_str).is_some_and(|h| !h.trim().is_empty());
            let sees = row
                .get("architecture")
                .and_then(|a| a.get("input_modalities"))
                .and_then(Value::as_array)
                .is_some_and(|m| m.iter().any(|x| x.as_str() == Some("image")));
            let free = id.ends_with(":free") && zero_price(row);
            free && open && match task {
                Task::Chat => true,
                Task::Vision => sees,
                _ => false,
            }
        }
        "groq" => match task {
            Task::Transcribe => has(&["whisper"]),
            Task::Speak => has(&["tts", "orpheus", "playai"]),
            Task::Chat => super::open_weight_chat(super::provider("groq").expect("a provider of the catalogue"), row),
            Task::Vision => has(&["vision", "scout", "maverick"]),
            Task::Embed => false,
        },
        "cloudflare" => {
            let task_name = row.get("task").and_then(|t| t.get("name")).and_then(Value::as_str).unwrap_or("");
            let fits = task_name.eq_ignore_ascii_case(cloudflare_task(task));
            fits && match task {
                Task::Vision => has(&["vision"]),
                Task::Chat => !has(&["vision"]),
                _ => true,
            }
        }
        "gemini" => match task {
            Task::Embed => has(&["embedding"]),
            Task::Chat | Task::Vision => has(&["gemini"]) && !has(&["embedding", "tts", "image", "live"]),
            Task::Speak => has(&["tts"]),
            Task::Transcribe => false,
        },
        _ => false,
    }
}

/// The models `reach` offers free for `task`, read live, by name. Cloudflare
/// is searched by its task name; the others' lists are filtered.
pub async fn models(client: &Client, reach: &Reach, task: Task, now: i64) -> Result<Vec<HostedModel>, Failure> {
    let url = if reach.cloud.id == CLOUDFLARE.id {
        let mut url = url::Url::parse(&reach.list).map_err(|_| unreadable(reach.cloud))?;
        url.query_pairs_mut().append_pair("task", cloudflare_task(task)).append_pair("per_page", "100");
        url.to_string()
    } else {
        reach.list.clone()
    };
    let got = checked(reach.cloud, client.get(reach, &url, WAIT).await?, now)?;
    let value = json_of(reach.cloud, &got)?;
    let rows = match &value {
        Value::Array(rows) => rows.clone(),
        Value::Object(map) => map
            .get("data")
            .or_else(|| map.get("result"))
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| unreadable(reach.cloud))?,
        _ => return Err(unreadable(reach.cloud)),
    };
    let mut found: Vec<HostedModel> = rows
        .iter()
        .filter(|row| offers(reach.cloud, task, row))
        .filter_map(|row| {
            // Cloudflare names a model by `name` (its id is a uuid); Gemini's ids may start with `models/`
            let id = if reach.cloud.id == CLOUDFLARE.id { row.get("name") } else { row.get("id") }?.as_str()?.trim();
            let id = id.strip_prefix("models/").unwrap_or(id);
            if id.is_empty() || !free_model(reach.cloud, id) {
                return None;
            }
            // a Cloudflare model the meter has no rate for is not offered
            if reach.cloud.id == CLOUDFLARE.id && meter::rate(id).is_none() {
                return None;
            }
            let name = row
                .get("display_name")
                .or_else(|| if reach.cloud.id == CLOUDFLARE.id { None } else { row.get("name") })
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|n| !n.is_empty())
                .unwrap_or(id);
            Some(HostedModel {
                id: id.to_owned(),
                name: name.to_owned(),
                context: row.get("context_length").and_then(Value::as_u64),
                price: None,
            })
        })
        .collect();
    found.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()).then(a.id.cmp(&b.id)));
    found.dedup_by(|a, b| a.id == b.id);
    Ok(found)
}

#[cfg(test)]
#[path = "parts_tests.rs"]
mod tests;
