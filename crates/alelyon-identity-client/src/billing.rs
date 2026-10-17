//! Plans and payment: the identity service's billing routes. Every route is a POST under `/identity/native/billing/`
//! (`plan`, `checkout`, `portal`) whose body carries the client id and the session's refresh token, which these routes
//! check WITHOUT rotating it, as the social routes do.
//!
//! An application shows the account's plan and sends the person to a web page to pay: checkout and the billing portal
//! answer a Stripe address to open in the person's own browser. No card number ever passes through an application.
//! The account a checkout or a portal is for is the caller's own, taken from its token.

use serde_json::{Value, json};

use crate::client::{CLIENT_ID, Client};

/// The account's plan as the service describes it.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Plan {
    /// `free`, `pro`, `team`, `max` or `ultra`.
    pub key: String,
    /// The plan's name to show (`Pro`).
    pub name: String,
    /// Whether the plan unlocks the hosted services (hosted model chat).
    pub hosted_access: bool,
    /// The plan's hosted chat allowance, in the service's units.
    pub chat_allowance: u64,
    /// Seats on the plan (more than one only for Team).
    pub seats: u64,
    /// When the paid period ends (RFC 3339), or None when the plan does not lapse or the service did not say.
    pub paid_through: Option<String>,
    /// Where the plan came from, in the service's own word (a subscription, ...), when it says.
    pub source: Option<String>,
    /// The account has a billing record, so the billing portal can open.
    pub manage_available: bool,
    /// The service can take a payment now.
    pub purchase_available: bool,
    /// What the plan allows of the harness on this machine.
    pub harness: Harness,
    /// On a seat of somebody else's Team (`source` `team-seat`): the username of the Team's admin, when the service says.
    pub team_owner: Option<String>,
}

/// What a plan allows of the harness on the person's own machine (the plan catalog's `background_tasks`,
/// `fleet_agents`, `schedules` and `project_manager`): tasks running at once, helper agents working at once, and
/// whether schedules and the project manager are on. Free runs 2 tasks and 4 helpers, with neither; a count of
/// [`Harness::UNLIMITED`] has no limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Harness {
    pub background_tasks: u32,
    pub helper_agents: u32,
    pub schedules: bool,
    pub project_manager: bool,
}

impl Harness {
    /// A count with no limit (the service's `null`).
    pub const UNLIMITED: u32 = u32::MAX;

    /// Free's.
    pub const FREE: Harness = Harness { background_tasks: 2, helper_agents: 4, schedules: false, project_manager: false };

    /// The catalog's numbers for a plan key, for a service that does not send them; Free's for a key this client does
    /// not know.
    pub fn of_key(key: &str) -> Harness {
        let paid = |background_tasks, helper_agents| Harness { background_tasks, helper_agents, schedules: true, project_manager: true };
        match key {
            "pro" | "team" => paid(6, 12),
            "max" | "ultra" => paid(Harness::UNLIMITED, Harness::UNLIMITED),
            _ => Harness::FREE,
        }
    }

    /// The service's `allowances` object, each number or flag it lacks taken from the catalog's for `key`.
    fn from_allowances(key: &str, a: Option<&Value>) -> Harness {
        let known = Harness::of_key(key);
        let Some(a) = a.filter(|a| a.is_object()) else {
            return known;
        };
        let count = |k: &str, or: u32| match a.get(k) {
            Some(Value::Null) => Harness::UNLIMITED,
            Some(n) => n.as_u64().and_then(|n| u32::try_from(n).ok()).unwrap_or(or),
            None => or,
        };
        let flag = |k: &str, or: bool| a.get(k).and_then(Value::as_bool).unwrap_or(or);
        Harness {
            background_tasks: count("background_tasks", known.background_tasks),
            helper_agents: count("fleet_agents", known.helper_agents),
            schedules: flag("schedules", known.schedules),
            project_manager: flag("project_manager", known.project_manager),
        }
    }
}

impl Default for Harness {
    fn default() -> Self {
        Harness::FREE
    }
}

impl Plan {
    pub fn from_json(v: &Value) -> Plan {
        let opt = |k: &str| v.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
        let flag = |k: &str| v.get(k).and_then(Value::as_bool).unwrap_or(false);
        Plan {
            key: opt("key").unwrap_or_else(|| "free".into()),
            name: opt("name").unwrap_or_else(|| "Free".into()),
            hosted_access: flag("hosted_access"),
            chat_allowance: v.get("chat_allowance").and_then(Value::as_u64).unwrap_or(0),
            seats: v.get("seats").and_then(Value::as_u64).unwrap_or(1).max(1),
            paid_through: opt("paid_through"),
            source: opt("source"),
            manage_available: flag("manage_available"),
            purchase_available: flag("purchase_available"),
            harness: Harness::from_allowances(opt("key").as_deref().unwrap_or("free"), v.get("allowances")),
            team_owner: opt("team_owner"),
        }
    }

    /// The plans that are paid for.
    pub const PAID: [&'static str; 4] = ["pro", "team", "max", "ultra"];

    /// A plan that is not paid for (Free, or any key this client does not know): the one to offer an upgrade from.
    pub fn upgradable(&self) -> bool {
        !Plan::PAID.contains(&self.key.as_str())
    }

    /// The paid-through day (`2026-11-10`), when the service gave one.
    pub fn paid_through_day(&self) -> Option<&str> {
        self.paid_through.as_deref().map(|t| t.get(..10).unwrap_or(t))
    }
}

/// How often a plan is billed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interval {
    Month,
    Year,
}

impl Interval {
    /// The word the contract uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Interval::Month => "month",
            Interval::Year => "year",
        }
    }
}

/// Why a billing call did not do what was asked.
#[derive(Clone, Debug, PartialEq)]
pub enum Problem {
    Unreachable(String),
    /// The session ended: signing in again is the way back.
    Expired,
    /// Too many at once: try again after this many seconds.
    RateLimited(u64),
    /// The sign-in service is down for maintenance, in its words.
    Maintenance(String),
    /// Payments cannot be taken now (the service has no payment provider, or the provider failed).
    Unavailable,
    /// The service refused, with its code (`already-subscribed`, `no-billing`, `unknown-plan`, `invalid-seats`, ...)
    /// and its words.
    Refused(String, String),
}

impl Problem {
    pub fn words(&self) -> String {
        match self {
            Problem::Unreachable(why) => format!("Billing could not be reached ({why})."),
            Problem::Expired => "Your session has ended. Sign in again.".into(),
            Problem::RateLimited(s) => format!("Too many attempts. Try again in {s} seconds."),
            Problem::Maintenance(words) => words.clone(),
            Problem::Unavailable => "Payments are not available right now.".into(),
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

/// Reads a billing route's answer (status, Retry-After, body).
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
        (503, _) => Problem::Unavailable,
        _ => Problem::Refused(code.clone(), words.unwrap_or_else(|| format!("Billing answered {status} ({code})."))),
    })
}

/// One billing route: `path` is the full route, `body` its own fields; the client id and the token are added here.
pub async fn call(client: &Client, token: &str, path: &str, mut body: Value) -> Result<Value, Problem> {
    body["client_id"] = json!(CLIENT_ID);
    body["refresh_token"] = json!(token);
    let (status, retry, text) = client.post_raw(path, body).await.map_err(Problem::Unreachable)?;
    interpret(status, retry, &text)
}

/// The `url` of a checkout or portal answer: an `https` address only, since an application opens it in a browser and
/// the system's URL handler would hand any other scheme to whatever program claims it.
pub fn url_from_json(v: &Value) -> Result<String, Problem> {
    match v.get("url").and_then(Value::as_str) {
        Some(u) if u.starts_with("https://") => Ok(u.to_string()),
        _ => Err(Problem::Refused("bad-url".into(), "The billing service did not give a web address to open.".into())),
    }
}

/// The signed-in account's plan.
pub async fn plan(client: &Client, token: &str) -> Result<Plan, Problem> {
    call(client, token, "/identity/native/billing/plan", json!({}))
        .await
        .map(|v| Plan::from_json(v.get("plan").unwrap_or(&Value::Null)))
}

/// A Stripe Checkout address for `plan` (`pro`, `team`, `max`, `ultra`), billed each `interval`, with `seats` (1 except
/// for Team). An account that already pays is refused (`already-subscribed`): changing plan is the portal's job.
pub async fn checkout(client: &Client, token: &str, plan: &str, interval: Interval, seats: u32) -> Result<String, Problem> {
    let v = call(
        client,
        token,
        "/identity/native/billing/checkout",
        json!({"plan": plan, "interval": interval.as_str(), "seats": seats}),
    )
    .await?;
    url_from_json(&v)
}

/// A Stripe customer-portal address (change plan, update the card, cancel, invoices). An account that never paid is
/// refused (`no-billing`).
pub async fn portal(client: &Client, token: &str) -> Result<String, Problem> {
    let v = call(client, token, "/identity/native/billing/portal", json!({})).await?;
    url_from_json(&v)
}

/// One person on a Team, as its admin sees them: their public handle, never an address.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Member {
    /// The account id the service names them by: what removing them takes.
    pub id: String,
    pub username: Option<String>,
    pub display_name: String,
    /// `pending` (invited, not yet accepted) or `active`.
    pub status: String,
    pub invite_id: Option<i64>,
    /// When an invitation lapses (RFC 3339), while it is pending.
    pub expires_at: Option<String>,
    /// Whether the seat gives them the plan now.
    pub has_access: bool,
}

impl Member {
    fn from_json(v: &Value) -> Member {
        let text = |k: &str| v.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
        Member {
            id: text("id").unwrap_or_default(),
            username: text("username"),
            display_name: text("display_name").unwrap_or_default(),
            status: text("status").unwrap_or_else(|| "pending".into()),
            invite_id: v.get("invite_id").and_then(Value::as_i64),
            expires_at: text("expires_at"),
            has_access: v.get("has_access").and_then(Value::as_bool).unwrap_or(false),
        }
    }

    /// How the person is named: `@username`, else their display name, else "someone".
    pub fn name(&self) -> String {
        match (&self.username, self.display_name.is_empty()) {
            (Some(u), _) => format!("@{u}"),
            (None, false) => self.display_name.clone(),
            (None, true) => "someone".into(),
        }
    }

    pub fn pending(&self) -> bool {
        self.status != "active"
    }
}

/// The caller's Team: its seats, how many are taken, and the people on it.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Team {
    pub seats: u32,
    pub used: u32,
    /// More people hold seats than the plan pays for (after the admin lowered the seats).
    pub over_limit: bool,
    pub members: Vec<Member>,
}

impl Team {
    fn from_json(v: &Value) -> Team {
        let count = |k: &str| v.get(k).and_then(Value::as_u64).and_then(|n| u32::try_from(n).ok()).unwrap_or(0);
        Team {
            seats: count("seats"),
            used: count("used"),
            over_limit: v.get("over_limit").and_then(Value::as_bool).unwrap_or(false),
            members: v.get("members").and_then(Value::as_array).map(|m| m.iter().map(Member::from_json).collect()).unwrap_or_default(),
        }
    }
}

/// An invitation to somebody's Team, waiting for the caller.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Invite {
    pub invite_id: i64,
    /// The username of who invited them (the Team's admin).
    pub from: Option<String>,
    pub expires_at: Option<String>,
}

/// The caller's Team. An account without a Team plan is refused (`not-team-owner`).
pub async fn team(client: &Client, token: &str) -> Result<Team, Problem> {
    call(client, token, "/identity/native/billing/team", json!({}))
        .await
        .map(|v| Team::from_json(v.get("team").unwrap_or(&Value::Null)))
}

/// Invite a person to a seat by their exact username; they have a pending seat until they accept.
pub async fn invite(client: &Client, token: &str, username: &str) -> Result<Member, Problem> {
    call(client, token, "/identity/native/billing/team/invite", json!({"username": username.trim().trim_start_matches('@')}))
        .await
        .map(|v| Member::from_json(v.get("member").unwrap_or(&Value::Null)))
}

/// End a person's seat, or cancel their invitation, by the id the Team names them by.
pub async fn remove(client: &Client, token: &str, member_id: &str) -> Result<(), Problem> {
    call(client, token, "/identity/native/billing/team/remove", json!({"member_id": member_id})).await.map(|_| ())
}

/// Give up the caller's own seat on somebody else's Team.
pub async fn leave(client: &Client, token: &str) -> Result<(), Problem> {
    call(client, token, "/identity/native/billing/team/leave", json!({})).await.map(|_| ())
}

/// The invitations waiting for the caller.
pub async fn invites(client: &Client, token: &str) -> Result<Vec<Invite>, Problem> {
    let v = call(client, token, "/identity/native/billing/team/invites", json!({})).await?;
    Ok(v.get("invites")
        .and_then(Value::as_array)
        .map(|list| {
            list.iter()
                .filter_map(|i| {
                    Some(Invite {
                        invite_id: i.get("invite_id").and_then(Value::as_i64)?,
                        from: i.get("from").and_then(Value::as_str).map(str::to_string),
                        expires_at: i.get("expires_at").and_then(Value::as_str).map(str::to_string),
                    })
                })
                .collect()
        })
        .unwrap_or_default())
}

/// Take the seat an invitation offers.
pub async fn accept(client: &Client, token: &str, invite_id: i64) -> Result<(), Problem> {
    call(client, token, "/identity/native/billing/team/accept", json!({"invite_id": invite_id})).await.map(|_| ())
}

/// Turn an invitation down.
pub async fn decline(client: &Client, token: &str, invite_id: i64) -> Result<(), Problem> {
    call(client, token, "/identity/native/billing/team/decline", json!({"invite_id": invite_id})).await.map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_team_and_its_people_are_read_as_the_service_shows_them() {
        let team = Team::from_json(&json!({"seats": 3, "used": 2, "over_limit": false, "members": [
            {"id": "acct_a", "username": "ada", "display_name": "Ada", "status": "active", "invite_id": 4, "has_access": true},
            {"id": "acct_b", "username": null, "display_name": "Bob", "status": "pending", "invite_id": 5,
             "expires_at": "2026-10-30T00:00:00Z", "has_access": false}
        ]}));
        assert_eq!((team.seats, team.used, team.over_limit), (3, 2, false));
        assert_eq!(team.members[0].name(), "@ada");
        assert!(!team.members[0].pending() && team.members[1].pending());
        assert_eq!((team.members[1].name(), team.members[1].invite_id), ("Bob".to_string(), Some(5)));
        assert_eq!(Member::default().name(), "someone");
        assert_eq!(Team::from_json(&Value::Null), Team::default());
        let refused = interpret(409, None, r#"{"error": "seats-full", "message": "Every seat on the team is taken."}"#).unwrap_err();
        assert_eq!((refused.code(), refused.words().as_str()), (Some("seats-full"), "Every seat on the team is taken."));
    }

    #[test]
    fn a_plans_harness_allowances_come_from_the_service_or_else_the_catalog() {
        let free = Plan::from_json(&json!({"key": "free"}));
        assert_eq!(free.harness, Harness::FREE);
        assert_eq!((free.harness.background_tasks, free.harness.helper_agents), (2, 4));
        assert!(!free.harness.schedules && !free.harness.project_manager);
        let pro = Plan::from_json(&json!({"key": "pro"}));
        assert_eq!(pro.harness, Harness { background_tasks: 6, helper_agents: 12, schedules: true, project_manager: true });
        let max = Plan::from_json(&json!({"key": "max", "allowances": {"background_tasks": null, "fleet_agents": null}}));
        assert_eq!((max.harness.background_tasks, max.harness.helper_agents), (Harness::UNLIMITED, Harness::UNLIMITED), "null: unlimited");
        assert_eq!(Plan::from_json(&json!({"key": "max"})).harness.background_tasks, Harness::UNLIMITED);
        let told = Plan::from_json(&json!({"key": "max", "allowances": {"background_tasks": 12, "fleet_agents": 30, "schedules": true}}));
        assert_eq!((told.harness.background_tasks, told.harness.helper_agents), (12, 30), "the service's numbers win");
        assert!(told.harness.project_manager, "a flag it lacks comes from the catalog");
        assert_eq!(Plan::from_json(&json!({"key": "enterprise-x"})).harness, Harness::FREE, "an unknown plan is Free's");
        assert_eq!(Plan::default().harness, Harness::FREE);
    }

    #[test]
    fn a_plan_is_read_with_its_seats_and_paid_through_day() {
        let p = Plan::from_json(&json!({
            "key": "team", "name": "Team", "hosted_access": true, "chat_allowance": 2000, "seats": 4,
            "paid_through": "2026-11-10T00:00:00Z", "source": "stripe", "manage_available": true, "purchase_available": true
        }));
        assert_eq!((p.name.as_str(), p.seats, p.paid_through_day()), ("Team", 4, Some("2026-11-10")));
        assert!(p.hosted_access && p.manage_available && p.purchase_available && !p.upgradable());
        assert_eq!(p.source.as_deref(), Some("stripe"));
    }

    #[test]
    fn free_and_any_unpaid_key_are_upgradable_and_a_missing_field_reads_as_free() {
        let free = Plan::from_json(&json!({"key": "free", "name": "Free", "seats": 1, "paid_through": null}));
        assert!(free.upgradable() && free.paid_through_day().is_none() && !free.manage_available);
        assert!(Plan::from_json(&json!({"key": "retired", "name": "Retired"})).upgradable(), "a key that is not a paid plan");
        for paid in Plan::PAID {
            assert!(!Plan::from_json(&json!({"key": paid})).upgradable(), "{paid}");
        }
        let empty = Plan::from_json(&Value::Null);
        assert_eq!((empty.key.as_str(), empty.name.as_str(), empty.seats), ("free", "Free", 1));
        assert_eq!(Plan::from_json(&json!({"seats": 0})).seats, 1, "never zero seats");
    }

    #[test]
    fn each_billing_refusal_is_read_from_its_status_and_code() {
        assert_eq!(interpret(401, None, r#"{"error": "invalid-token", "message": "x"}"#), Err(Problem::Expired));
        assert_eq!(interpret(429, Some(9), r#"{"error": "rate-limited"}"#), Err(Problem::RateLimited(9)));
        assert_eq!(interpret(429, None, "{}"), Err(Problem::RateLimited(30)), "no Retry-After: a default wait");
        let unavailable = interpret(503, None, r#"{"error": "billing-unavailable", "message": "Payments are not available right now. Try again later."}"#);
        assert_eq!(unavailable, Err(Problem::Unavailable));
        assert_eq!(Problem::Unavailable.words(), "Payments are not available right now.");
        assert_eq!(
            interpret(503, None, r#"{"error": "maintenance", "message": "Alelyon sign-in is down for maintenance."}"#),
            Err(Problem::Maintenance("Alelyon sign-in is down for maintenance.".into()))
        );
        let already = interpret(409, None, r#"{"error": "already-subscribed", "message": "You already have a plan. Change it from Manage billing."}"#);
        assert_eq!(already.as_ref().err().and_then(Problem::code), Some("already-subscribed"));
        assert_eq!(already.unwrap_err().words(), "You already have a plan. Change it from Manage billing.");
        let none = interpret(404, None, r#"{"error": "no-billing", "message": "This account has no plan to manage yet."}"#);
        assert_eq!(none, Err(Problem::Refused("no-billing".into(), "This account has no plan to manage yet.".into())));
        assert!(matches!(interpret(500, None, "not json"), Err(Problem::Refused(c, w)) if c.is_empty() && w.contains("500")));
        assert!(interpret(200, None, r#"{"url": "https://checkout.stripe.com/c/x"}"#).is_ok());
    }

    #[test]
    fn only_an_https_address_is_given_to_open() {
        assert_eq!(url_from_json(&json!({"url": "https://billing.stripe.com/p/s"})).as_deref(), Ok("https://billing.stripe.com/p/s"));
        for bad in [json!({"url": "file:///C:/x"}), json!({"url": "http://example.com"}), json!({"url": 3}), json!({})] {
            assert!(matches!(url_from_json(&bad), Err(Problem::Refused(c, _)) if c == "bad-url"), "{bad}");
        }
    }

    /// The calls themselves, against a stand-in service on a loopback port: each POSTs to its route with the client id
    /// and the token in its body, and reads the answer.
    #[test]
    fn each_call_posts_its_route_with_the_client_id_and_token() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let answers = [
            (200, r#"{"plan": {"key": "pro", "name": "Pro", "hosted_access": true, "seats": 1, "paid_through": "2026-11-10T00:00:00Z", "manage_available": true, "purchase_available": true}}"#),
            (200, r#"{"url": "https://checkout.stripe.com/c/abc"}"#),
            (200, r#"{"url": "https://billing.stripe.com/p/session/xyz"}"#),
            (503, r#"{"error": "billing-unavailable", "message": "Payments are not available right now. Try again later."}"#),
            (409, r#"{"error": "already-subscribed", "message": "You already have a plan. Change it from Manage billing."}"#),
        ];
        let server = std::thread::spawn(move || {
            let mut seen = Vec::new();
            for (status, body) in answers {
                let (mut conn, _) = listener.accept().unwrap();
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                // read the head, then as many body bytes as Content-Length says
                loop {
                    let n = conn.read(&mut chunk).unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    if let Some(end) = text.find("\r\n\r\n") {
                        let len = text[..end]
                            .lines()
                            .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                            .unwrap_or(0);
                        if buf.len() >= end + 4 + len {
                            seen.push(text);
                            break;
                        }
                    }
                    if n == 0 {
                        break;
                    }
                }
                let reason = if status == 200 { "OK" } else { "Refused" };
                let reply = format!(
                    "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                conn.write_all(reply.as_bytes()).unwrap();
            }
            seen
        });

        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let client = Client::new(base).unwrap();
        let (plan_answer, checkout_answer, portal_answer, unavailable, already) = rt.block_on(async {
            (
                plan(&client, "tok-1").await,
                checkout(&client, "tok-1", "team", Interval::Year, 3).await,
                portal(&client, "tok-1").await,
                portal(&client, "tok-1").await,
                checkout(&client, "tok-1", "pro", Interval::Month, 1).await,
            )
        });
        let seen = server.join().unwrap();

        let p = plan_answer.unwrap();
        assert_eq!((p.key.as_str(), p.paid_through_day(), p.manage_available), ("pro", Some("2026-11-10"), true));
        assert_eq!(checkout_answer.as_deref(), Ok("https://checkout.stripe.com/c/abc"));
        assert_eq!(portal_answer.as_deref(), Ok("https://billing.stripe.com/p/session/xyz"));
        assert_eq!(unavailable, Err(Problem::Unavailable));
        assert_eq!(already.as_ref().err().and_then(Problem::code), Some("already-subscribed"));

        let body = |i: usize| -> Value { serde_json::from_str(seen[i].split("\r\n\r\n").nth(1).unwrap()).unwrap() };
        assert!(seen[0].starts_with("POST /identity/native/billing/plan "), "{}", seen[0]);
        assert!(seen[1].starts_with("POST /identity/native/billing/checkout "));
        assert!(seen[2].starts_with("POST /identity/native/billing/portal "));
        assert_eq!(body(0), json!({"client_id": CLIENT_ID, "refresh_token": "tok-1"}));
        assert_eq!(body(1), json!({"client_id": CLIENT_ID, "refresh_token": "tok-1", "plan": "team", "interval": "year", "seats": 3}));
        assert_eq!(body(2), json!({"client_id": CLIENT_ID, "refresh_token": "tok-1"}));
    }
}
