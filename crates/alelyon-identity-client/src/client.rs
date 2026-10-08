//! The identity service's native routes, as the contract names them (`/identity/native/*`, `/identity/status`; the
//! contract is `docs/native-sign-in-contract.md`). Every call is HTTPS through the system's TLS (reqwest with
//! native-tls); nothing here logs a token, and an error carries the service's words, never the request.

use std::time::Duration;

use serde_json::{Value, json};

/// The desktop's registered native client: a public client (no secret), whose only redirect is the loopback.
pub const CLIENT_ID: &str = "alelyon-desktop";

/// The deployed identity service.
pub const DEFAULT_URL: &str = "https://id.api.alelyon.com";

/// Where the identity service is: the deployed one, unless `ALELYON_IDENTITY_URL` names another (a local stand-in) or
/// is `off` (no service: an application should say so and offer to carry on offline).
pub fn base_url() -> Option<String> {
    pick_url(std::env::var("ALELYON_IDENTITY_URL").ok().as_deref())
}

/// The service a setting names: `None` (or blank) is the deployed one, `off` (any case) is none, anything else is
/// that address without a trailing slash.
pub fn pick_url(set: Option<&str>) -> Option<String> {
    match set.map(str::trim) {
        Some(u) if u.eq_ignore_ascii_case("off") => None,
        Some(u) if !u.is_empty() => Some(u.trim_end_matches('/').to_string()),
        _ => Some(DEFAULT_URL.to_string()),
    }
}

/// An account as the service describes it.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Account {
    pub id: String,
    pub email: String,
    pub display_name: String,
    pub username: Option<String>,
    pub providers: Vec<String>,
}

impl Account {
    pub fn from_json(v: &Value) -> Option<Account> {
        let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
        Some(Account {
            id: s("id")?,
            email: s("email").unwrap_or_default(),
            display_name: s("display_name").unwrap_or_default(),
            username: s("username"),
            providers: v
                .get("providers")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(|p| p.as_str().map(str::to_string)).collect())
                .unwrap_or_default(),
        })
    }

    /// The name a person sees: the display name, else the username, else the email.
    pub fn name(&self) -> &str {
        [self.display_name.as_str(), self.username.as_deref().unwrap_or(""), self.email.as_str()]
            .into_iter()
            .find(|s| !s.is_empty())
            .unwrap_or("you")
    }
}

/// A signed-in session: the refresh token (the one secret), when it lapses, and who it is. Its `Debug` output leaves
/// the token out, so a session can be logged or asserted on without leaking it.
#[derive(Clone, PartialEq)]
pub struct Session {
    pub refresh_token: String,
    pub expires_at: String,
    pub account: Account,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // never the token
        f.debug_struct("Session").field("expires_at", &self.expires_at).field("account", &self.account).finish_non_exhaustive()
    }
}

impl Session {
    pub fn from_json(v: &Value) -> Option<Session> {
        Some(Session {
            refresh_token: v.get("refresh_token")?.as_str()?.to_string(),
            expires_at: v.get("expires_at").and_then(Value::as_str).unwrap_or_default().to_string(),
            account: Account::from_json(v.get("account")?)?,
        })
    }
}

/// A notice the service publishes: scheduled maintenance, an incident, or news.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Notice {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub body: String,
    pub starts_at: String,
    pub ends_at: String,
}

/// A sign-in provider and whether the service has it switched on.
#[derive(Clone, Debug, PartialEq)]
pub struct Provider {
    pub id: String,
    pub name: String,
    pub enabled: bool,
}

/// What `/identity/status` says.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Status {
    pub service: String,
    pub notices: Vec<Notice>,
    pub providers: Vec<Provider>,
    pub sign_up_url: String,
    pub reset_url: String,
    pub account_url: String,
    pub security_url: String,
}

impl Status {
    pub fn from_json(v: &Value) -> Status {
        let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        Status {
            service: s(v, "service"),
            notices: v
                .get("notices")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|n| Notice {
                            id: s(n, "id"),
                            kind: s(n, "kind"),
                            title: s(n, "title"),
                            body: s(n, "body"),
                            starts_at: s(n, "starts_at"),
                            ends_at: s(n, "ends_at"),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            providers: v
                .get("providers")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|p| {
                            Some(Provider { id: s(p, "id"), name: s(p, "name"), enabled: p.get("enabled").and_then(Value::as_bool)? })
                        })
                        .collect()
                })
                .unwrap_or_default(),
            sign_up_url: s(v, "sign_up_url"),
            reset_url: s(v, "reset_url"),
            account_url: s(v, "account_url"),
            security_url: s(v, "security_url"),
        }
    }

    /// Whether the service says it is down for maintenance now.
    pub fn in_maintenance(&self) -> bool {
        self.service == "maintenance"
    }
}

/// Why a call did not give what was asked.
#[derive(Clone, Debug, PartialEq)]
pub enum Failure {
    /// The email/username and password do not match an account.
    Credentials,
    /// The account's email is not verified yet.
    Unverified,
    /// Too many attempts: try again after this many seconds.
    RateLimited(u64),
    /// The service is down for maintenance; its notice, when it gave one.
    Maintenance(Option<Notice>),
    /// The saved session is no longer valid: sign in again.
    Expired,
    /// The service could not be reached.
    Unreachable(String),
    /// The provider's email already has an account: sign in to it with its password to link the provider (the ticket
    /// goes with that sign-in).
    LinkNeeded(String),
    /// Anything else, in the service's words.
    Other(String),
}

impl Failure {
    pub fn words(&self) -> String {
        match self {
            Failure::Credentials => "Your email or username and password do not match.".into(),
            Failure::Unverified => "Your email address is not verified yet: open the link we sent you, then sign in.".into(),
            Failure::RateLimited(s) => format!("Too many attempts. Try again in {s} s."),
            Failure::Maintenance(Some(n)) if !n.title.is_empty() => format!("{} The service is down for maintenance.", n.title),
            Failure::Maintenance(_) => {
                "Alelyon's accounts are down for scheduled maintenance. Try again later, or use Alelyon offline.".into()
            }
            Failure::Expired => "Your saved sign-in has lapsed. Please sign in again.".into(),
            Failure::Unreachable(why) => format!("Alelyon's sign-in service could not be reached ({why}). You can use Alelyon offline."),
            Failure::LinkNeeded(_) => {
                "An Alelyon account already uses that email. Sign in to it with your password here, and that way of signing in is linked to it.".into()
            }
            Failure::Other(why) => why.clone(),
        }
    }
}

/// Read a reply into its JSON, or the failure its status and `error` code name.
pub fn interpret(status: u16, retry_after: Option<u64>, body: &str) -> Result<Value, Failure> {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    if (200..300).contains(&status) {
        return Ok(v);
    }
    let code = v.get("error").and_then(Value::as_str).unwrap_or("");
    let words = v.get("message").and_then(Value::as_str).filter(|m| !m.is_empty()).map(str::to_string);
    Err(match (status, code) {
        (401, "invalid-credentials") => Failure::Credentials,
        (401, _) => Failure::Expired,
        (403, "unverified-email") => Failure::Unverified,
        (429, _) => Failure::RateLimited(retry_after.unwrap_or(30)),
        (503, _) => {
            let n = v.get("notice").map(|n| {
                let s = |k: &str| n.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
                Notice {
                    id: s("id"),
                    kind: s("kind"),
                    title: s("title"),
                    body: s("body"),
                    starts_at: s("starts_at"),
                    ends_at: s("ends_at"),
                }
            });
            Failure::Maintenance(n)
        }
        _ => Failure::Other(words.unwrap_or_else(|| {
            format!("The sign-in service answered {status} ({}).", if code.is_empty() { "no reason given" } else { code })
        })),
    })
}

/// A client of one identity service.
#[derive(Clone)]
pub struct Client {
    base: String,
    http: reqwest::Client,
}

impl Client {
    /// A client of the service at `base` (no trailing slash; see [`base_url`]). It only builds the HTTPS client:
    /// nothing is sent until a call is made.
    pub fn new(base: String) -> Result<Client, String> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .connect_timeout(Duration::from_secs(8))
            .user_agent(concat!("Alelyon/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| format!("could not make an HTTPS client: {e}"))?;
        Ok(Client { base, http })
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value, Failure> {
        let reply = self
            .http
            .post(format!("{}{path}", self.base))
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| Failure::Unreachable(short(&e)))?;
        read(reply).await
    }

    /// The service this client talks to.
    pub fn base(&self) -> &str {
        &self.base
    }

    /// A native call whose refusals the caller reads itself (the social routes): the HTTP status, Retry-After (whole
    /// seconds) and the body, or why nothing came back.
    pub async fn post_raw(&self, path: &str, body: Value) -> Result<(u16, Option<u64>, String), String> {
        let reply = self
            .http
            .post(format!("{}{path}", self.base))
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(|e| short(&e))?;
        let status = reply.status().as_u16();
        let retry = reply.headers().get("retry-after").and_then(|h| h.to_str().ok()).and_then(|s| s.trim().parse().ok());
        let text = reply.text().await.map_err(|e| short(&e))?;
        Ok((status, retry, text))
    }

    /// `GET /identity/status`: whether the service is up, its notices, which providers are on, and its web pages.
    pub async fn status(&self) -> Result<Status, Failure> {
        let reply = self
            .http
            .get(format!("{}/identity/status", self.base))
            .header("accept", "application/json")
            .send()
            .await
            .map_err(|e| Failure::Unreachable(short(&e)))?;
        read(reply).await.map(|v| Status::from_json(&v))
    }

    /// Password sign-in (`login` is an email or a username). The password is sent once, over TLS, and not kept.
    /// `link_ticket` is the ticket a provider sign-in gave ([`Failure::LinkNeeded`]): with it, this sign-in links
    /// that provider to the account.
    pub async fn sign_in(&self, login: &str, password: &str, stay: bool, link_ticket: Option<&str>) -> Result<Session, Failure> {
        let mut body = json!({"client_id": CLIENT_ID, "login": login, "password": password, "stay_signed_in": stay});
        if let Some(t) = link_ticket {
            body["link_ticket"] = json!(t);
        }
        let v = self.post("/identity/native/sign-in", body).await?;
        Session::from_json(&v).ok_or_else(|| Failure::Other("The sign-in service gave no session.".into()))
    }

    /// Renew a session. The service rotates the token: the one passed in stops working, so keep the new one.
    pub async fn refresh(&self, refresh_token: &str) -> Result<Session, Failure> {
        let v = self.post("/identity/native/refresh", json!({"client_id": CLIENT_ID, "refresh_token": refresh_token})).await?;
        Session::from_json(&v).ok_or_else(|| Failure::Other("The sign-in service gave no session.".into()))
    }

    /// Revoke the token (and its family) at the service.
    pub async fn sign_out(&self, refresh_token: &str) -> Result<(), Failure> {
        self.post("/identity/native/sign-out", json!({"client_id": CLIENT_ID, "refresh_token": refresh_token})).await.map(|_| ())
    }

    /// Trade a provider sign-in's one-time code, with the PKCE verifier only this client holds, for a session.
    pub async fn exchange(&self, redirect_uri: &str, code: &str, verifier: &str) -> Result<Session, Failure> {
        let v = self
            .post(
                "/identity/native/exchange",
                json!({"client_id": CLIENT_ID, "redirect_uri": redirect_uri, "code": code, "verifier": verifier}),
            )
            .await?;
        Session::from_json(&v).ok_or_else(|| Failure::Other("The sign-in service gave no session.".into()))
    }

    /// The browser page that starts a provider's sign-in.
    pub fn provider_start_url(&self, provider: &str, redirect_uri: &str, state: &str, challenge: &str, stay: bool) -> String {
        let q = |s: &str| percent_encoding::utf8_percent_encode(s, percent_encoding::NON_ALPHANUMERIC).to_string();
        format!(
            "{}/identity/native/provider/{}/start?client_id={}&redirect_uri={}&state={}&code_challenge={}&code_challenge_method=S256&stay_signed_in={}",
            self.base,
            q(provider),
            q(CLIENT_ID),
            q(redirect_uri),
            q(state),
            q(challenge),
            stay
        )
    }

    /// Start a QR pairing: a code to show (and its approve page, for a QR code) that a signed-in phone approves.
    pub async fn pair_start(&self, stay: bool) -> Result<Pairing, Failure> {
        let v = self.post("/identity/native/pair/start", json!({"client_id": CLIENT_ID, "stay_signed_in": stay})).await?;
        Pairing::from_json(&v).ok_or_else(|| Failure::Other("The sign-in service gave no pairing code.".into()))
    }

    /// Ask whether the pairing was approved: None while it waits.
    pub async fn pair_poll(&self, pair_id: &str) -> Result<Option<Session>, Failure> {
        let reply = self
            .http
            .post(format!("{}/identity/native/pair/poll", self.base))
            .header("content-type", "application/json")
            .body(json!({"client_id": CLIENT_ID, "pair_id": pair_id}).to_string())
            .send()
            .await
            .map_err(|e| Failure::Unreachable(short(&e)))?;
        if reply.status().as_u16() == 202 {
            return Ok(None);
        }
        let status = reply.status().as_u16();
        let v = read(reply).await.map_err(|f| match (status, f) {
            (410, _) => Failure::Other("The code expired. Get a new one.".into()),
            (403, _) => Failure::Other("The sign-in was refused on your other device.".into()),
            (_, f) => f,
        })?;
        Session::from_json(&v).map(Some).ok_or_else(|| Failure::Other("The sign-in service gave no session.".into()))
    }
}

/// A QR pairing on the way.
#[derive(Clone, Debug, PartialEq)]
pub struct Pairing {
    pub pair_id: String,
    pub user_code: String,
    pub approve_url: String,
    pub expires_at: String,
    pub interval_s: u64,
}

impl Pairing {
    pub fn from_json(v: &Value) -> Option<Pairing> {
        let s = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
        Some(Pairing {
            pair_id: s("pair_id")?,
            user_code: s("user_code")?,
            approve_url: s("approve_url")?,
            expires_at: s("expires_at").unwrap_or_default(),
            interval_s: v.get("interval").and_then(Value::as_u64).unwrap_or(2).clamp(1, 30),
        })
    }
}

async fn read(reply: reqwest::Response) -> Result<Value, Failure> {
    let status = reply.status().as_u16();
    let retry = reply.headers().get("retry-after").and_then(|h| h.to_str().ok()).and_then(|s| s.trim().parse().ok());
    let body = reply.text().await.map_err(|e| Failure::Unreachable(short(&e)))?;
    interpret(status, retry, &body)
}

/// A transport error in a few words, never the URL (it can carry a code).
fn short(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        "it did not answer in time".into()
    } else if e.is_connect() {
        "no connection".into()
    } else {
        "the connection failed".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_deployed_service_is_the_default_and_the_setting_can_name_another_or_none() {
        assert_eq!(pick_url(None).as_deref(), Some(DEFAULT_URL));
        assert_eq!(pick_url(Some("  ")).as_deref(), Some(DEFAULT_URL));
        assert_eq!(pick_url(Some("http://127.0.0.1:8765/")).as_deref(), Some("http://127.0.0.1:8765"));
        assert_eq!(pick_url(Some("OFF")), None);
    }

    const SESSION: &str = r#"{"refresh_token": "tok", "expires_at": "2026-11-07T00:00:00Z", "token_type": "opaque-refresh",
        "account": {"id": "a1", "email": "t@example.com", "display_name": "Ada", "username": null, "providers": ["password", "github"]}}"#;

    #[test]
    fn a_session_is_read_and_its_token_never_printed() {
        let s = Session::from_json(&serde_json::from_str(SESSION).unwrap()).unwrap();
        assert_eq!((s.account.name(), s.account.providers.len()), ("Ada", 2));
        assert!(!format!("{s:?}").contains("tok"), "the token stays out of debug output");
        assert!(Session::from_json(&json!({"account": {"id": "a"}})).is_none(), "no token, no session");
    }

    #[test]
    fn each_failure_is_read_from_its_status_and_code() {
        assert_eq!(interpret(401, None, r#"{"error":"invalid-credentials"}"#), Err(Failure::Credentials));
        assert_eq!(interpret(401, None, r#"{"error":"invalid-token"}"#), Err(Failure::Expired));
        assert_eq!(interpret(403, None, r#"{"error":"unverified-email"}"#), Err(Failure::Unverified));
        assert_eq!(interpret(429, Some(12), "{}"), Err(Failure::RateLimited(12)));
        let m = interpret(503, None, r#"{"error":"maintenance","notice":{"title":"Scheduled update.","ends_at":"2026-10-08T02:00:00Z"}}"#);
        assert!(matches!(&m, Err(Failure::Maintenance(Some(n))) if n.title == "Scheduled update."));
        assert!(matches!(interpret(500, None, "not json"), Err(Failure::Other(w)) if w.contains("500")));
        assert!(interpret(200, None, SESSION).is_ok());
    }

    #[test]
    fn status_reads_notices_and_which_providers_are_on() {
        let st = Status::from_json(
            &json!({"service": "maintenance", "notices": [{"id": "n1", "kind": "maintenance", "title": "Update", "starts_at": "a", "ends_at": "b"}],
            "providers": [{"id": "github", "name": "GitHub", "enabled": true}, {"id": "orcid", "name": "ORCID", "enabled": false}, {"id": "x"}],
            "sign_up_url": "https://www.alelyon.com/account"}),
        );
        assert!(st.in_maintenance());
        assert_eq!(st.notices[0].title, "Update");
        assert_eq!(st.providers.len(), 2, "a provider without its enabled flag is left out");
        assert!(!st.providers[1].enabled);
    }

    #[test]
    fn a_providers_start_address_carries_the_pkce_challenge_encoded() {
        let c = Client::new("https://id.example".into()).unwrap();
        let u = c.provider_start_url("github", "http://127.0.0.1:5123/callback", "st&ate", "chal", true);
        assert!(u.starts_with("https://id.example/identity/native/provider/github/start?client_id=alelyon%2Ddesktop"));
        assert!(u.contains("redirect_uri=http%3A%2F%2F127%2E0%2E0%2E1%3A5123%2Fcallback") && u.contains("state=st%26ate"));
        assert!(u.contains("code_challenge=chal&code_challenge_method=S256&stay_signed_in=true"));
    }

    #[test]
    fn a_pairing_is_read_with_its_poll_interval_bounded() {
        let p = Pairing::from_json(
            &json!({"pair_id": "p", "user_code": "ABCD-1234", "approve_url": "https://x/identity/pair/p", "interval": 0}),
        )
        .unwrap();
        assert_eq!((p.user_code.as_str(), p.interval_s), ("ABCD-1234", 1));
    }
}
