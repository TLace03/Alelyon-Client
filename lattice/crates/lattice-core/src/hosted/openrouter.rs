//! OpenRouter's sign-in: the reader approves a key for Lattice in the browser,
//! and OpenRouter hands it back (its OAuth PKCE flow, as documented on
//! 2026-10-09: a localhost callback on any port is accepted).
//!
//! 1. [`begin`] listens on this machine only (`127.0.0.1`, and `::1` on the
//!    same port where it can) and makes the page to open: `openrouter.ai/auth`
//!    with the callback `http://localhost:<port>/callback`, an S256 challenge
//!    of a random verifier, and a random `state`;
//! 2. the caller opens that page; the reader signs in and approves;
//! 3. [`Pending::finish`] waits for the browser to come back with a `code`
//!    (a callback whose `state` is not ours, or another path, is answered and
//!    ignored), tells the browser it may close the tab, and trades the code
//!    for the key (`POST /api/v1/auth/keys` with the verifier).
//!
//! There is no timer: the wait ends when the browser comes back or the caller
//! drops it (Cancel). OpenRouter's code lasts 10 minutes. The key is the
//! reader's, listed and revocable on OpenRouter's keys page.

use std::net::{Ipv4Addr, Ipv6Addr, SocketAddr};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::net::{TcpListener, TcpStream};

use super::{Http, status_sentence};
use crate::keys::SecretString;

/// The page the reader approves on.
pub const AUTH_PAGE: &str = "https://openrouter.ai/auth";
/// Where a code is traded for the key.
pub const KEYS_URL: &str = "https://openrouter.ai/api/v1/auth/keys";
/// What the key is called on OpenRouter's keys page.
pub const KEY_LABEL: &str = "Alelyon Lattice";
/// The most of a callback request read.
const MAX_REQUEST: usize = 8 * 1024;

/// A sign-in under way: the page to open, and the wait for its answer.
pub struct Pending {
    url: String,
    verifier: String,
    state: String,
    v4: TcpListener,
    v6: Option<TcpListener>,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending")
            .field("url", &self.url)
            .finish_non_exhaustive()
    }
}

/// Unpadded base64url, as PKCE's S256 challenge is written.
pub fn base64url(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        let take = chunk.len() + 1;
        for i in 0..take {
            out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
        }
    }
    out
}

/// The S256 challenge of a verifier.
pub fn challenge(verifier: &str) -> String {
    base64url(&Sha256::digest(verifier.as_bytes()))
}

/// The approval page for a callback port, a challenge and a state.
pub fn auth_url(port: u16, challenge: &str, state: &str) -> String {
    let mut url = url::Url::parse(AUTH_PAGE).expect("a fixed address");
    url.query_pairs_mut()
        .append_pair("callback_url", &format!("http://localhost:{port}/callback"))
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("key_label", KEY_LABEL)
        .append_pair("state", state);
    url.into()
}

/// Two random UUIDs' hex: 64 unreserved characters (PKCE wants 43 to 128).
fn random() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

/// Listen for the callback and make the page to open.
pub async fn begin() -> Result<Pending, String> {
    let v4 = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .map_err(|_| {
            "Lattice could not listen for OpenRouter's answer on this computer.".to_owned()
        })?;
    let port = v4
        .local_addr()
        .map_err(|_| {
            "Lattice could not listen for OpenRouter's answer on this computer.".to_owned()
        })?
        .port();
    // A browser may try `localhost` as ::1 first; listen there too where the port is free.
    let v6 = TcpListener::bind(SocketAddr::from((Ipv6Addr::LOCALHOST, port)))
        .await
        .ok();
    let verifier = random();
    let state = uuid::Uuid::new_v4().simple().to_string();
    let url = auth_url(port, &challenge(&verifier), &state);
    Ok(Pending {
        url,
        verifier,
        state,
        v4,
        v6,
    })
}

/// What a request to the callback listener asked.
#[derive(Debug, PartialEq, Eq)]
pub enum Callback {
    /// The approval came back with its code.
    Code(String),
    /// The reader did not approve (or OpenRouter said an error).
    Refused,
    /// Not the callback, or not ours: answered and ignored.
    Other,
}

/// What a request's first line asks, against the state we sent.
pub fn parse_callback(request: &str, state: &str) -> Callback {
    let mut parts = request.lines().next().unwrap_or("").split(' ');
    let (Some("GET"), Some(target)) = (parts.next(), parts.next()) else {
        return Callback::Other;
    };
    let Ok(url) = url::Url::parse(&format!("http://localhost{target}")) else {
        return Callback::Other;
    };
    if url.path() != "/callback" {
        return Callback::Other;
    }
    let pairs: Vec<(String, String)> = url.query_pairs().into_owned().collect();
    let get = |k: &str| pairs.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
    match (get("code"), get("state")) {
        (Some(code), Some(s)) if s == state && !code.trim().is_empty() => {
            Callback::Code(code.to_owned())
        }
        // A denial comes back without our state (OpenRouter's documentation).
        (None, _) if get("error").is_some() => Callback::Refused,
        _ => Callback::Other,
    }
}

/// The page the browser shows after the callback.
fn page(status: &str, words: &str) -> String {
    let body = format!(
        "<!doctype html><meta charset=utf-8><title>Alelyon</title><body style=\"font-family:sans-serif;background:#0b0b0c;color:#e8e2d0;padding:48px\"><h2>{words}</h2></body>"
    );
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n{body}",
        body.len()
    )
}

async fn read_request(stream: &TcpStream) -> Option<String> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") && buf.len() < MAX_REQUEST {
        stream.readable().await.ok()?;
        match stream.try_read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(_) => return None,
        }
    }
    String::from_utf8(buf).ok()
}

async fn write_all(stream: &TcpStream, mut bytes: &[u8]) {
    while !bytes.is_empty() {
        if stream.writable().await.is_err() {
            return;
        }
        match stream.try_write(bytes) {
            Ok(n) => bytes = &bytes[n..],
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(_) => return,
        }
    }
}

impl Pending {
    /// The page to open in the reader's browser.
    pub fn url(&self) -> &str {
        &self.url
    }

    async fn accept(&self) -> Option<TcpStream> {
        match &self.v6 {
            Some(v6) => tokio::select! {
                r = self.v4.accept() => r.ok().map(|(s, _)| s),
                r = v6.accept() => r.ok().map(|(s, _)| s),
            },
            None => self.v4.accept().await.ok().map(|(s, _)| s),
        }
    }

    /// Wait for the approval, then trade its code for the reader's key.
    pub async fn finish(self, http: &dyn Http) -> Result<SecretString, String> {
        let code = loop {
            let Some(stream) = self.accept().await else {
                return Err(
                    "Lattice stopped listening for OpenRouter's answer: try again.".to_owned(),
                );
            };
            let Some(request) = read_request(&stream).await else {
                continue;
            };
            match parse_callback(&request, &self.state) {
                Callback::Code(code) => {
                    write_all(&stream, page("200 OK", "Signed in to OpenRouter. You can close this tab and return to Alelyon.").as_bytes()).await;
                    break code;
                }
                Callback::Refused => {
                    write_all(
                        &stream,
                        page(
                            "200 OK",
                            "OpenRouter was not connected. You can close this tab.",
                        )
                        .as_bytes(),
                    )
                    .await;
                    return Err(
                        "OpenRouter was not connected: the key was not approved.".to_owned()
                    );
                }
                Callback::Other => {
                    write_all(&stream, page("404 Not Found", "Not found.").as_bytes()).await
                }
            }
        };
        exchange(http, &code, &self.verifier).await
    }
}

/// Trade an approval's code for the key.
pub async fn exchange(http: &dyn Http, code: &str, verifier: &str) -> Result<SecretString, String> {
    let body = json!({"code": code, "code_verifier": verifier, "code_challenge_method": "S256"});
    let (status, reply) = http.post_json(KEYS_URL.to_owned(), body).await?;
    if status == 403 {
        return Err("OpenRouter refused the sign-in (it lasts 10 minutes): try again.".to_owned());
    }
    if status != 200 {
        return Err(status_sentence("OpenRouter", status));
    }
    serde_json::from_slice::<Value>(&reply)
        .ok()
        .and_then(|v| v.get("key")?.as_str().map(str::to_owned))
        .filter(|k| !k.trim().is_empty())
        .map(SecretString::new)
        .ok_or_else(|| "OpenRouter's answer held no key.".to_owned())
}
