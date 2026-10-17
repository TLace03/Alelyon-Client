//! Credits: the account's credits balance, read from the credits service with a short-lived access token.
//!
//! Two calls, in this order:
//! 1. [`access_token`]: `POST {identity}/identity/native/access-token` with `{client_id, refresh_token, audience}`
//!    answers `{token, token_type: "Bearer", expires_at, audience}`. The refresh token is checked WITHOUT being
//!    rotated, as the social and billing routes check it. The token is for one service only (its audience,
//!    [`AUDIENCE`] here), and that service verifies it offline.
//! 2. [`balance`]: `GET {credits}/credits/balance` with `Authorization: Bearer <token>` answers
//!    `{"balance": {available_credits, bought_credits, promotional_credits, held_credits, blocked}}`, in whole
//!    credits. Whose balance it is comes from the token, never from the request.
//!
//! Buying credits happens on the website, in the person's own browser; nothing here takes a payment. Neither call
//! logs a token, and an access token's `Debug` leaves it out.

use serde_json::{Value, json};

use crate::client::{CLIENT_ID, Client};

/// The audience of the credits service's access tokens.
pub const AUDIENCE: &str = "alelyon-credits";

/// The identity service's route that issues access tokens.
pub const ACCESS_TOKEN_PATH: &str = "/identity/native/access-token";

/// The credits service's balance route.
pub const BALANCE_PATH: &str = "/credits/balance";

/// A short-lived token for one service.
#[derive(Clone, PartialEq)]
pub struct AccessToken {
    pub token: String,
    /// `Bearer`.
    pub token_type: String,
    /// When it stops being accepted (as the service wrote it), or empty when it did not say.
    pub expires_at: String,
    /// The service it is for.
    pub audience: String,
}

impl std::fmt::Debug for AccessToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccessToken")
            .field("token", &"<hidden>")
            .field("token_type", &self.token_type)
            .field("expires_at", &self.expires_at)
            .field("audience", &self.audience)
            .finish()
    }
}

/// The account's credits, in whole credits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Balance {
    /// What can be spent now (bought plus promotional, less what is held). Negative after a disputed payment.
    pub available_credits: i64,
    pub bought_credits: i64,
    pub promotional_credits: i64,
    /// Held for model requests under way.
    pub held_credits: i64,
    /// Spending is blocked until a disputed payment is made good.
    pub blocked: bool,
}

/// Why a credits call did not give an answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Problem {
    Unreachable(String),
    /// The session ended: signing in again is the way back.
    Expired,
    /// Too many at once: try again after this many seconds.
    RateLimited(u64),
    /// The sign-in service is down for maintenance, in its words.
    Maintenance(String),
    /// The service refused or answered something unreadable, with its code (`access-tokens-off`,
    /// `credits-unavailable`, `bad-token`, `bad-balance`, ...) and words to show.
    Refused(String, String),
}

impl Problem {
    pub fn words(&self) -> String {
        match self {
            Problem::Unreachable(why) => format!("Credits could not be reached ({why})."),
            Problem::Expired => "Your session has ended. Sign in again.".into(),
            Problem::RateLimited(s) => format!("Too many attempts. Try again in {s} seconds."),
            Problem::Maintenance(words) => words.clone(),
            Problem::Refused(_, words) => words.clone(),
        }
    }

    /// The service's code, when it refused.
    pub fn code(&self) -> Option<&str> {
        match self {
            Problem::Refused(code, _) => Some(code),
            _ => None,
        }
    }
}

/// Reads an answer of either route (status, Retry-After, body).
pub fn interpret(status: u16, retry_after: Option<u64>, body: &str) -> Result<Value, Problem> {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    if (200..300).contains(&status) {
        return Ok(v);
    }
    let code = v.get("error").and_then(Value::as_str).unwrap_or("").to_string();
    let words = v.get("message").and_then(Value::as_str).filter(|m| !m.is_empty()).map(str::to_string);
    Err(match (status, code.as_str()) {
        (401, _) => Problem::Expired,
        (429, _) => Problem::RateLimited(retry_after.unwrap_or(30)),
        (503, "maintenance") => Problem::Maintenance(words.unwrap_or_else(|| "Alelyon sign-in is down for maintenance.".into())),
        _ => Problem::Refused(code.clone(), words.unwrap_or_else(|| format!("Credits answered {status} ({code})."))),
    })
}

fn unreadable(code: &str) -> Problem {
    Problem::Refused(code.into(), "The credits service's answer could not be read.".into())
}

/// The access token of an answer: `token` (only; the field is not called anything else), a `Bearer` type, and the
/// audience that was asked for.
pub fn access_token_from_json(v: &Value, audience: &str) -> Result<AccessToken, Problem> {
    let text = |k: &str| v.get(k).and_then(Value::as_str).map(str::trim).unwrap_or("").to_string();
    let token = text("token");
    let token_type = text("token_type");
    let said_audience = text("audience");
    if token.is_empty() || !token_type.eq_ignore_ascii_case("bearer") || said_audience != audience {
        return Err(unreadable("bad-token"));
    }
    Ok(AccessToken { token, token_type, expires_at: text("expires_at"), audience: said_audience })
}

/// The balance of a `/credits/balance` answer. `available_credits` must be there; the others read as 0 (and
/// `blocked` as false) when absent.
pub fn balance_from_json(v: &Value) -> Result<Balance, Problem> {
    let b = v.get("balance").filter(|b| b.is_object()).ok_or_else(|| unreadable("bad-balance"))?;
    let int = |k: &str| b.get(k).and_then(Value::as_i64);
    Ok(Balance {
        available_credits: int("available_credits").ok_or_else(|| unreadable("bad-balance"))?,
        bought_credits: int("bought_credits").unwrap_or(0),
        promotional_credits: int("promotional_credits").unwrap_or(0),
        held_credits: int("held_credits").unwrap_or(0),
        blocked: b.get("blocked").and_then(Value::as_bool).unwrap_or(false),
    })
}

/// Whether `origin` may be sent a token: `https://`, or plain `http://` to this machine only.
pub fn origin_allowed(origin: &str) -> bool {
    let o = origin.trim();
    o.starts_with("https://")
        || o.strip_prefix("http://").is_some_and(|rest| {
            let host = rest.split(['/', ':']).next().unwrap_or("");
            host == "127.0.0.1" || host == "localhost"
        })
}

/// An access token for `audience`, from the identity service, with the session's refresh token (not rotated).
pub async fn access_token(identity: &Client, refresh_token: &str, audience: &str) -> Result<AccessToken, Problem> {
    let body = json!({"client_id": CLIENT_ID, "refresh_token": refresh_token, "audience": audience});
    let (status, retry, text) = identity.post_raw(ACCESS_TOKEN_PATH, body).await.map_err(Problem::Unreachable)?;
    access_token_from_json(&interpret(status, retry, &text)?, audience)
}

/// The balance at the credits service at `credits_origin` (no trailing slash), read with an access token for
/// [`AUDIENCE`]. An empty origin means the service is not deployed: nothing is asked of anyone.
pub async fn balance(identity: &Client, refresh_token: &str, credits_origin: &str) -> Result<Balance, Problem> {
    let origin = credits_origin.trim().trim_end_matches('/');
    if origin.is_empty() {
        return Err(Problem::Refused("not-deployed".into(), "Credits are not available yet.".into()));
    }
    if !origin_allowed(origin) {
        return Err(Problem::Refused("bad-origin".into(), "The credits service's address is not a secure one.".into()));
    }
    let token = access_token(identity, refresh_token, AUDIENCE).await?;
    let credits = Client::new(origin.to_string()).map_err(Problem::Unreachable)?;
    let (status, retry, text) = credits.get_bearer_raw(BALANCE_PATH, &token.token).await.map_err(Problem::Unreachable)?;
    balance_from_json(&interpret(status, retry, &text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_access_token_is_read_from_its_token_field_only() {
        let answer = json!({"token": "tok.abc", "token_type": "Bearer", "expires_at": "2026-10-15T04:00:00Z", "audience": AUDIENCE});
        let t = access_token_from_json(&answer, AUDIENCE).unwrap();
        assert_eq!((t.token.as_str(), t.expires_at.as_str(), t.audience.as_str()), ("tok.abc", "2026-10-15T04:00:00Z", AUDIENCE));
        assert!(!format!("{t:?}").contains("tok.abc"), "Debug leaves the token out");
        // the old field name is not read
        let old = json!({"access_token": "tok.abc", "token_type": "Bearer", "expires_at": "x", "audience": AUDIENCE});
        assert_eq!(access_token_from_json(&old, AUDIENCE).unwrap_err().code(), Some("bad-token"));
        // another audience, another type, or no token: refused
        let other = json!({"token": "t", "token_type": "Bearer", "audience": "alelyon-models"});
        assert!(access_token_from_json(&other, AUDIENCE).is_err());
        assert!(access_token_from_json(&json!({"token": "t", "token_type": "MAC", "audience": AUDIENCE}), AUDIENCE).is_err());
        assert!(access_token_from_json(&json!({"token": " ", "token_type": "Bearer", "audience": AUDIENCE}), AUDIENCE).is_err());
        assert!(access_token_from_json(&Value::Null, AUDIENCE).is_err());
    }

    #[test]
    fn a_balance_is_read_in_whole_credits() {
        let answer = json!({"balance": {"available_credits": 1250, "bought_credits": 1000, "promotional_credits": 300, "held_credits": 50, "blocked": false}});
        assert_eq!(
            balance_from_json(&answer),
            Ok(Balance { available_credits: 1250, bought_credits: 1000, promotional_credits: 300, held_credits: 50, blocked: false })
        );
        // after a disputed payment: negative and blocked
        let disputed = json!({"balance": {"available_credits": -40, "bought_credits": 0, "promotional_credits": 0, "held_credits": 0, "blocked": true}});
        let b = balance_from_json(&disputed).unwrap();
        assert!(b.blocked && b.available_credits == -40);
        // only available_credits is required
        assert_eq!(balance_from_json(&json!({"balance": {"available_credits": 7}})).unwrap(), Balance { available_credits: 7, ..Default::default() });
        for bad in [json!({}), json!({"balance": 5}), json!({"balance": {}}), json!({"balance": {"available_credits": "12"}}), json!({"balance": {"available_credits": 1.5}})] {
            assert_eq!(balance_from_json(&bad).unwrap_err().code(), Some("bad-balance"), "{bad}");
        }
    }

    #[test]
    fn each_refusal_is_read_from_its_status_and_code() {
        assert_eq!(interpret(401, None, r#"{"error": "invalid-token", "message": "x"}"#), Err(Problem::Expired));
        assert_eq!(interpret(429, Some(4), "{}"), Err(Problem::RateLimited(4)));
        assert_eq!(interpret(429, None, "{}"), Err(Problem::RateLimited(30)));
        assert_eq!(
            interpret(503, None, r#"{"error": "maintenance", "message": "Alelyon sign-in is down for maintenance."}"#),
            Err(Problem::Maintenance("Alelyon sign-in is down for maintenance.".into()))
        );
        let off = interpret(503, None, r#"{"error": "access-tokens-off", "message": "Access tokens are not offered yet."}"#).unwrap_err();
        assert_eq!((off.code(), off.words().as_str()), (Some("access-tokens-off"), "Access tokens are not offered yet."));
        let unavailable = interpret(503, None, r#"{"error": "credits-unavailable", "message": "Buying credits is not available right now. Try again later."}"#);
        assert_eq!(unavailable.unwrap_err().code(), Some("credits-unavailable"));
        assert!(matches!(interpret(500, None, "not json"), Err(Problem::Refused(c, w)) if c.is_empty() && w.contains("500")));
        assert!(interpret(200, None, r#"{"balance": {"available_credits": 1}}"#).is_ok());
    }

    #[test]
    fn a_token_goes_only_to_a_secure_origin_and_an_empty_one_asks_nothing() {
        assert!(origin_allowed("https://credits.example.test"));
        assert!(origin_allowed("http://127.0.0.1:8123") && origin_allowed("http://localhost/x"));
        for bad in ["http://credits.example.test", "http://127.0.0.1.example.test", "ftp://x", "", "credits.example.test"] {
            assert!(!origin_allowed(bad), "{bad}");
        }
        // no origin, or an insecure one: refused before any request (the identity client points nowhere reachable)
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let identity = Client::new("http://127.0.0.1:9".into()).unwrap();
        let (none, insecure) = rt.block_on(async {
            (balance(&identity, "tok", "  ").await, balance(&identity, "tok", "http://credits.example.test").await)
        });
        assert_eq!(none.unwrap_err().code(), Some("not-deployed"));
        assert_eq!(insecure.unwrap_err().code(), Some("bad-origin"));
    }
}
