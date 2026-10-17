//! The identity service's Pages routes, called with a developer token.
//!
//! Every signed-in Pages route is `POST /identity/native/pages/<route>` with a JSON body of the route's own fields.
//! A developer token stands in for a session: `Authorization: Bearer alyd_...`. The service checks the token's
//! scopes, logs the call for its owner, and labels a post made with it with the token's name.
//!
//! Nothing here prints or logs the token. A transport failure is reported in a few words, never with the request.

use std::time::Duration;

use serde_json::{Value, json};

/// Where the Pages routes live under the service's address.
pub const PAGES_PREFIX: &str = "/identity/native/pages";
/// A developer token's shape: the prefix and 43 URL-safe characters.
pub const TOKEN_PREFIX: &str = "alyd_";

/// Whether `value` has a developer token's shape.
pub fn looks_like_token(value: &str) -> bool {
    value.len() == TOKEN_PREFIX.len() + 43
        && value.starts_with(TOKEN_PREFIX)
        && value[TOKEN_PREFIX.len()..].bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// What went wrong with a call.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    /// The HTTP status, 0 when the service was not reached.
    pub status: u16,
    /// The service's error code (`scope-missing`, `invalid-token`, ...), or a local one.
    pub code: String,
    /// The service's own words.
    pub message: String,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.status == 0 {
            write!(f, "{} ({})", self.message, self.code)
        } else {
            write!(f, "{} ({}, HTTP {})", self.message, self.code, self.status)
        }
    }
}

/// Something that answers a Pages route: the service, or a fake in tests.
pub trait Pages {
    fn call(&self, route: &str, body: &Value) -> Result<Value, Failure>;
}

/// The service, over HTTPS.
pub struct Service {
    base: String,
    token: String,
    http: reqwest::blocking::Client,
}

impl std::fmt::Debug for Service {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Service").field("base", &self.base).finish_non_exhaustive()
    }
}

/// An address the token may be sent to: HTTPS, or plain HTTP to this computer (a local test service).
pub fn allowed_base(base: &str) -> bool {
    let loopback = ["http://127.0.0.1", "http://localhost", "http://[::1]"]
        .iter()
        .any(|p| base == *p || base.starts_with(&format!("{p}:")) || base.starts_with(&format!("{p}/")));
    (base.starts_with("https://") || loopback) && !base.contains('@') && !base.contains('?') && !base.contains('#')
}

impl Service {
    pub fn new(base: &str, token: &str) -> Result<Service, Failure> {
        let base = base.trim_end_matches('/').to_string();
        if !allowed_base(&base) {
            return Err(local("bad-address", "The service address must be https:// (or http:// to this computer)."));
        }
        if !looks_like_token(token) {
            return Err(local("bad-token", "That is not a developer token (alyd_ and 43 characters)."));
        }
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(60))
            .connect_timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .user_agent(concat!("alelyon-cli/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|_| local("no-client", "Could not start HTTPS."))?;
        Ok(Service { base, token: token.to_string(), http })
    }
}

fn local(code: &str, message: &str) -> Failure {
    Failure { status: 0, code: code.into(), message: message.into() }
}

/// A route name as the service spells them: lowercase words joined by hyphens.
pub fn route_ok(route: &str) -> bool {
    !route.is_empty()
        && route.len() <= 40
        && route.bytes().all(|b| b.is_ascii_lowercase() || b == b'-')
        && !route.starts_with('-')
        && !route.ends_with('-')
}

impl Pages for Service {
    fn call(&self, route: &str, body: &Value) -> Result<Value, Failure> {
        if !route_ok(route) {
            return Err(local("bad-route", "That is not a Pages route name."));
        }
        if !body.is_object() {
            return Err(local("bad-body", "A route's fields are one JSON object."));
        }
        let url = format!("{}{}/{}", self.base, PAGES_PREFIX, route);
        let sent = self
            .http
            .post(url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .body(serde_json::to_vec(body).unwrap_or_default())
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    local("timeout", "Alelyon did not answer in time.")
                } else {
                    local("unreachable", "Could not reach Alelyon.")
                }
            })?;
        let status = sent.status().as_u16();
        let text = sent.text().map_err(|_| local("unreadable", "Alelyon's answer could not be read."))?;
        if status == 204 {
            return Ok(json!({"ok": true}));
        }
        let parsed: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
        if (200..300).contains(&status) {
            return Ok(parsed);
        }
        Err(Failure {
            status,
            code: parsed.get("error").and_then(Value::as_str).unwrap_or("error").to_string(),
            message: parsed.get("message").and_then(Value::as_str).unwrap_or("Alelyon refused the request.").to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_token_has_its_shape() {
        assert!(looks_like_token(&format!("alyd_{}", "A".repeat(43))));
        assert!(!looks_like_token(&format!("alyn_{}", "A".repeat(43))));
        assert!(!looks_like_token(&format!("alyd_{}", "A".repeat(42))));
        assert!(!looks_like_token(&format!("alyd_{}!", "A".repeat(42))));
    }

    #[test]
    fn the_token_goes_only_to_https_or_this_computer() {
        assert!(allowed_base("https://id.api.alelyon.com"));
        assert!(allowed_base("http://127.0.0.1:8080"));
        assert!(allowed_base("http://localhost"));
        assert!(!allowed_base("http://id.api.alelyon.com"));
        assert!(!allowed_base("http://localhost.evil.test"));
        assert!(!allowed_base("https://user@id.api.alelyon.com"));
    }

    #[test]
    fn route_names_are_the_services_shape() {
        assert!(route_ok("feed") && route_ok("notifications-read"));
        assert!(!route_ok("") && !route_ok("../me") && !route_ok("Feed") && !route_ok("-x"));
    }

    #[test]
    fn debug_never_shows_the_token() {
        let token = format!("alyd_{}", "S".repeat(43));
        let s = Service::new("https://id.example", &token).unwrap();
        assert!(!format!("{s:?}").contains(&token));
    }
}
