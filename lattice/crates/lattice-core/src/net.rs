//! The core's own HTTP client, for the managed llama.cpp server only
//! (the chat core's spec §2.2 `net.rs`, §22 LR2; ADR-0041 decision 3).
//!
//! Every request it makes goes to `http://127.0.0.1:<port>/…`, and goes there
//! directly:
//! - a URL whose host is not the literal `127.0.0.1` (no name, no other
//!   address, no user-info) is refused before anything connects, so a
//!   configuration mistake cannot become a request off the machine;
//! - no proxy, ever (`no_proxy()`): an inherited `HTTP_PROXY` must never see
//!   a loopback request or its launch token;
//! - no redirect (`Policy::none()`);
//! - connect and whole-request timeouts, and a bounded body;
//! - no idle connection is kept, so no pool timer runs while the core is
//!   idle (CB1).
//!
//! [`HttpGet`] is the seam: the local runtime asks through it, and a test can
//! hand it a recorder that sees every request. The model client of a turn is
//! not this one: it is `lattice-agents`' Chat Completions client, built
//! through `choices` (N6).
//!
//! Nothing here logs, and an error never carries the transport's text.

use std::io;
use std::time::Duration;

use futures::FutureExt;
use futures::future::BoxFuture;
use lattice_agents::SecretString;

/// The loopback address the managed server binds and the client calls.
pub const LOOPBACK: &str = "127.0.0.1";
/// The most body a response may carry.
pub const MAX_BODY: usize = 4 * 1024 * 1024;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// One GET.
#[derive(Clone, Debug)]
pub struct HttpRequest {
    pub url: String,
    /// Sent as `Authorization: Bearer <token>`.
    pub bearer: Option<SecretString>,
    /// The whole request, connect included.
    pub timeout: Duration,
}

/// What came back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// Why a request has no response. Never the transport's text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NetError {
    /// The URL is not `http://127.0.0.1…`: nothing was sent.
    NotLoopback,
    /// The client could not be built.
    Client,
    /// No connection, or it broke.
    Connect,
    Timeout,
    /// The body was longer than [`MAX_BODY`].
    TooLarge,
}

/// Where the local runtime's requests go through.
pub trait HttpGet: Send + Sync {
    fn get(&self, request: HttpRequest) -> BoxFuture<'static, Result<HttpResponse, NetError>>;
}

/// Is `url` `http://127.0.0.1[:port]/…`, with nothing else in its authority?
pub fn is_loopback_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://") else {
        return false;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    let host = match authority.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => host,
        Some(_) => return false,
        None => authority,
    };
    host == LOOPBACK
}

/// The real client.
pub struct LoopbackHttp {
    client: reqwest::Client,
}

impl LoopbackHttp {
    pub fn new() -> Result<Self, NetError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(CONNECT_TIMEOUT)
            .pool_max_idle_per_host(0)
            .user_agent(concat!("Lattice/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|_| NetError::Client)?;
        Ok(Self { client })
    }
}

impl HttpGet for LoopbackHttp {
    fn get(&self, request: HttpRequest) -> BoxFuture<'static, Result<HttpResponse, NetError>> {
        let client = self.client.clone();
        async move {
            if !is_loopback_url(&request.url) {
                return Err(NetError::NotLoopback);
            }
            let mut builder = client.get(&request.url).timeout(request.timeout);
            if let Some(token) = &request.bearer {
                builder = builder.bearer_auth(token.expose());
            }
            let mut response = builder.send().await.map_err(classify)?;
            let status = response.status().as_u16();
            let mut body = Vec::new();
            while let Some(chunk) = response.chunk().await.map_err(classify)? {
                if body.len() + chunk.len() > MAX_BODY {
                    return Err(NetError::TooLarge);
                }
                body.extend_from_slice(&chunk);
            }
            Ok(HttpResponse { status, body })
        }
        .boxed()
    }
}

impl LoopbackHttp {
    /// POST `body` as JSON to `url` with the same rules as [`HttpGet::get`]: loopback only (nothing is sent
    /// elsewhere), no proxy, no redirect, the token only in the header, the answer bounded by [`MAX_BODY`]. For the
    /// benchmark's one generation (`llama::bench`).
    pub async fn post_json(
        &self,
        url: &str,
        bearer: Option<&SecretString>,
        body: &serde_json::Value,
        timeout: Duration,
    ) -> Result<HttpResponse, NetError> {
        if !is_loopback_url(url) {
            return Err(NetError::NotLoopback);
        }
        let bytes = serde_json::to_vec(body).map_err(|_| NetError::Client)?;
        let mut builder =
            self.client.post(url).timeout(timeout).header("Content-Type", "application/json").body(bytes);
        if let Some(token) = bearer {
            builder = builder.bearer_auth(token.expose());
        }
        let mut response = builder.send().await.map_err(classify)?;
        let status = response.status().as_u16();
        let mut answer = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(classify)? {
            if answer.len() + chunk.len() > MAX_BODY {
                return Err(NetError::TooLarge);
            }
            answer.extend_from_slice(&chunk);
        }
        Ok(HttpResponse { status, body: answer })
    }
}

fn classify(error: reqwest::Error) -> NetError {
    if error.is_timeout() {
        NetError::Timeout
    } else {
        NetError::Connect
    }
}

/// A free port on the loopback address, chosen now (`_free_port`). The
/// listener is closed again at once, as Python's is, so another process could
/// take the port before the server binds it, and could answer `/health`. The
/// readiness check therefore sends nothing until the port's listener belongs
/// to the server's own process (spec §22.6 LR7a, `llama::server`).
pub fn free_loopback_port() -> io::Result<u16> {
    let listener = std::net::TcpListener::bind((LOOPBACK, 0))?;
    Ok(listener.local_addr()?.port())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_loopback_literal_is_a_loopback_url() {
        for url in [
            "http://127.0.0.1:8080/health",
            "http://127.0.0.1/props",
            "http://127.0.0.1:1",
        ] {
            assert!(is_loopback_url(url), "{url}");
        }
        for url in [
            "https://127.0.0.1:8080/health",
            "http://localhost:8080/health",
            "http://127.0.0.2:8080/",
            "http://[::1]:8080/",
            "http://user:pw@127.0.0.1:8080/",
            "http://127.0.0.1.example.test/",
            "http://198.51.100.7:8080/",
            "http://127.0.0.1:x/",
            "http://127.0.0.1:/",
            "127.0.0.1:8080",
        ] {
            assert!(!is_loopback_url(url), "{url}");
        }
    }

    #[test]
    fn a_url_off_the_loopback_address_is_refused_before_anything_connects() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let client = LoopbackHttp::new().unwrap();
        let outcome = runtime.block_on(client.get(HttpRequest {
            url: "http://198.51.100.7:9/props".into(),
            bearer: Some("token".into()),
            timeout: Duration::from_secs(1),
        }));
        assert_eq!(outcome, Err(NetError::NotLoopback));
    }

    #[test]
    fn a_free_port_is_on_the_loopback_address_and_free() {
        let port = free_loopback_port().unwrap();
        assert!(port > 0);
        std::net::TcpListener::bind((LOOPBACK, port)).expect("the port is free again");
    }
}
