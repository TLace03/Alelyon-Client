//! Friends, presence and one-to-one chat: the identity service's social routes (contract v1,
//! `docs/social-contract.md`). Every route is a POST under `/identity/native/social/` whose body carries the client id
//! and the session's refresh token, which these routes check WITHOUT rotating it.
//!
//! The service runs nothing on a timer: it ages presence when a friend list is read. The application polls, and only
//! while it is open and signed in. The contract's suggested pace: presence every 60 s, the friend list every 30 s
//! (10 s while a friends panel is open), and an open conversation every 5 s.
//!
//! A person here is an id, a username and a display name, never an email address. Messages are stored on the server
//! in plain text: they are not end-to-end encrypted in v1.

use serde_json::{Value, json};

use crate::client::{CLIENT_ID, Client};

/// Someone as the social routes show them.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Person {
    pub id: String,
    pub username: Option<String>,
    pub display_name: String,
}

impl Person {
    pub fn from_json(v: &Value) -> Person {
        Person {
            id: str_of(v, "id"),
            username: v.get("username").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string),
            display_name: str_of(v, "display_name"),
        }
    }

    /// The name a person sees: the display name, else the username.
    pub fn name(&self) -> &str {
        if !self.display_name.is_empty() { &self.display_name } else { self.username.as_deref().unwrap_or("someone") }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Presence {
    Online,
    Away,
    Offline,
}

impl Presence {
    /// The word the contract uses.
    pub fn as_str(self) -> &'static str {
        match self {
            Presence::Online => "online",
            Presence::Away => "away",
            Presence::Offline => "offline",
        }
    }

    /// The contract's word read back; anything unknown reads as offline.
    pub fn from_word(word: Option<&str>) -> Presence {
        match word {
            Some("online") => Presence::Online,
            Some("away") => Presence::Away,
            _ => Presence::Offline,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Friend {
    pub person: Person,
    pub presence: Presence,
    /// Messages from this friend not read yet.
    pub unread: u64,
}

/// A friend request: its id, and who it is from (incoming) or to (outgoing).
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub id: String,
    pub person: Person,
}

/// What `friends` answers: the friends (online, then away, then offline), and the requests waiting both ways.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Friends {
    pub friends: Vec<Friend>,
    pub incoming: Vec<Request>,
    pub outgoing: Vec<Request>,
}

impl Friends {
    pub fn from_json(v: &Value) -> Friends {
        let list = |k: &str| v.get(k).and_then(Value::as_array).cloned().unwrap_or_default();
        Friends {
            friends: list("friends")
                .iter()
                .map(|f| Friend {
                    person: Person::from_json(f.get("person").unwrap_or(&Value::Null)),
                    presence: Presence::from_word(f.get("presence").and_then(Value::as_str)),
                    unread: f.get("unread").and_then(Value::as_u64).unwrap_or(0),
                })
                .collect(),
            incoming: list("incoming")
                .iter()
                .map(|r| Request { id: str_of(r, "id"), person: Person::from_json(r.get("from").unwrap_or(&Value::Null)) })
                .collect(),
            outgoing: list("outgoing")
                .iter()
                .map(|r| Request { id: str_of(r, "id"), person: Person::from_json(r.get("to").unwrap_or(&Value::Null)) })
                .collect(),
        }
    }

    /// Friends online or away.
    pub fn online(&self) -> usize {
        self.friends.iter().filter(|f| f.presence != Presence::Offline).count()
    }

    /// Unread messages from every friend.
    pub fn unread(&self) -> u64 {
        self.friends.iter().map(|f| f.unread).sum()
    }
}

/// A chat message: `from` is the sender's account id.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub id: String,
    pub from: String,
    pub text: String,
    pub sent_at: String,
}

impl Message {
    pub fn from_json(v: &Value) -> Message {
        Message { id: str_of(v, "id"), from: str_of(v, "from"), text: str_of(v, "text"), sent_at: str_of(v, "sent_at") }
    }
}

/// One page of a conversation: up to 50 messages, oldest first, and whether older ones remain.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub more: bool,
}

impl Conversation {
    pub fn from_json(v: &Value) -> Conversation {
        Conversation {
            messages: v.get("messages").and_then(Value::as_array).map(|a| a.iter().map(Message::from_json).collect()).unwrap_or_default(),
            more: v.get("more").and_then(Value::as_bool).unwrap_or(false),
        }
    }
}

/// The people a search found (zero or one: the match is an exact username).
pub fn people_from_json(v: &Value) -> Vec<Person> {
    v.get("people").and_then(Value::as_array).map(|a| a.iter().map(Person::from_json).collect()).unwrap_or_default()
}

fn str_of(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or_default().to_string()
}

/// Why a social call did not do what was asked.
#[derive(Clone, Debug, PartialEq)]
pub enum Problem {
    Unreachable(String),
    /// The session ended: signing in again is the way back.
    Expired,
    /// The account has no username yet; social features need one.
    UsernameNeeded,
    /// Too many at once: try again after this many seconds.
    RateLimited(u64),
    Maintenance,
    /// The service refused, with its code (kept for telling refusals apart) and its words.
    Refused(String, String),
}

impl Problem {
    pub fn words(&self) -> String {
        match self {
            Problem::Unreachable(why) => format!("Friends could not be reached ({why})."),
            Problem::Expired => "Your session has ended. Sign in again to see your friends.".into(),
            Problem::UsernameNeeded => "Choose a username so friends can find you.".into(),
            Problem::RateLimited(s) => format!("That was a lot at once. Try again in {s} seconds."),
            Problem::Maintenance => "Friends are down for scheduled maintenance.".into(),
            Problem::Refused(_, words) => words.clone(),
        }
    }
}

/// Reads a social route's answer (status, Retry-After, body). A 204 has no body and reads as `Value::Null`.
pub fn interpret(status: u16, retry_after: Option<u64>, body: &str) -> Result<Value, Problem> {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    if (200..300).contains(&status) {
        return Ok(v);
    }
    let code = v.get("error").and_then(Value::as_str).unwrap_or("").to_string();
    let words = v.get("message").and_then(Value::as_str).filter(|m| !m.is_empty()).map(str::to_string);
    Err(match (status, code.as_str()) {
        (401, _) => Problem::Expired,
        (409, "username-needed") => Problem::UsernameNeeded,
        (429, _) => Problem::RateLimited(retry_after.unwrap_or(30)),
        (503, _) => Problem::Maintenance,
        _ => Problem::Refused(code.clone(), words.unwrap_or_else(|| format!("Friends answered {status} ({code})."))),
    })
}

/// One social route: `path` is the full route (`/identity/native/social/<name>`), `body` its own fields; the client
/// id and the token are added here.
pub async fn call(client: &Client, token: &str, path: &str, mut body: Value) -> Result<Value, Problem> {
    body["client_id"] = json!(CLIENT_ID);
    body["refresh_token"] = json!(token);
    let (status, retry, text) = client.post_raw(path, body).await.map_err(Problem::Unreachable)?;
    interpret(status, retry, &text)
}

fn me(v: &Value) -> Person {
    Person::from_json(v.get("me").unwrap_or(&Value::Null))
}

/// Who the signed-in account is to its friends. The one route an account without a username may call.
pub async fn profile(client: &Client, token: &str) -> Result<Person, Problem> {
    call(client, token, "/identity/native/social/profile", json!({})).await.map(|v| me(&v))
}

/// Give the account its FIRST username (2-39 lowercase letters, digits, `.`, `_` or `-`); changing one is not in v1.
pub async fn set_username(client: &Client, token: &str, username: &str) -> Result<Person, Problem> {
    call(client, token, "/identity/native/social/profile", json!({"username": username})).await.map(|v| me(&v))
}

/// Find someone by their exact username (any case). Never a prefix, never an email, never the caller: a query that
/// is not a valid username finds nobody.
pub async fn search(client: &Client, token: &str, query: &str) -> Result<Vec<Person>, Problem> {
    call(client, token, "/identity/native/social/search", json!({"query": query})).await.map(|v| people_from_json(&v))
}

pub async fn friends(client: &Client, token: &str) -> Result<Friends, Problem> {
    call(client, token, "/identity/native/social/friends", json!({})).await.map(|v| Friends::from_json(&v))
}

/// Send a friend request by username. The answer is `"requested"`, or `"friends"` when that person had already asked.
pub async fn request(client: &Client, token: &str, username: &str) -> Result<String, Problem> {
    call(client, token, "/identity/native/social/request", json!({"username": username}))
        .await
        .map(|v| v.get("state").and_then(Value::as_str).unwrap_or("requested").to_string())
}

/// Accept or decline a request sent to this account (declining tells nobody).
pub async fn respond(client: &Client, token: &str, request_id: &str, accept: bool) -> Result<(), Problem> {
    call(client, token, "/identity/native/social/respond", json!({"request_id": request_id, "accept": accept})).await.map(|_| ())
}

/// Withdraw a request this account sent.
pub async fn cancel(client: &Client, token: &str, request_id: &str) -> Result<(), Problem> {
    call(client, token, "/identity/native/social/cancel", json!({"request_id": request_id})).await.map(|_| ())
}

/// End a friendship, for both people; their messages are deleted with it.
pub async fn remove(client: &Client, token: &str, friend_id: &str) -> Result<(), Problem> {
    call(client, token, "/identity/native/social/remove", json!({"friend_id": friend_id})).await.map(|_| ())
}

/// Report this app's presence. A friend whose last report is older than 150 s is shown offline.
pub async fn presence(client: &Client, token: &str, state: Presence) -> Result<(), Problem> {
    call(client, token, "/identity/native/social/presence", json!({"state": state.as_str()})).await.map(|_| ())
}

/// The newest 50 messages with a friend (older than the message `before`, when given), oldest first. Reading marks
/// the friend's messages up to the newest returned as read.
pub async fn messages(client: &Client, token: &str, friend_id: &str, before: Option<&str>) -> Result<Conversation, Problem> {
    let mut body = json!({"friend_id": friend_id});
    if let Some(b) = before {
        body["before"] = json!(b);
    }
    call(client, token, "/identity/native/social/messages", body).await.map(|v| Conversation::from_json(&v))
}

/// Send a message: 1-2000 characters after trimming, with no control character except newline (a tab or a carriage
/// return is refused).
pub async fn send(client: &Client, token: &str, friend_id: &str, text: &str) -> Result<Message, Problem> {
    call(client, token, "/identity/native/social/send", json!({"friend_id": friend_id, "text": text}))
        .await
        .map(|v| Message::from_json(v.get("message").unwrap_or(&Value::Null)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_friends_answer_reads_people_presence_and_requests_and_never_needs_an_email() {
        let v = json!({
            "friends": [
                {"person": {"id": "a1", "username": "ada", "display_name": "Ada"}, "presence": "online", "last_seen": null, "unread": 2},
                {"person": {"id": "a2", "username": "bo", "display_name": ""}, "presence": "away", "unread": 0},
                {"person": {"id": "a3", "username": null, "display_name": "Cy"}, "presence": "offline", "unread": 1}
            ],
            "incoming": [{"id": "r1", "from": {"id": "a4", "username": "dee", "display_name": "Dee"}, "sent_at": "2026-10-08T00:00:00Z"}],
            "outgoing": [{"id": "r2", "to": {"id": "a5", "username": "eve", "display_name": "Eve"}, "sent_at": "2026-10-08T00:00:00Z"}]
        });
        let f = Friends::from_json(&v);
        assert_eq!(f.friends.len(), 3);
        assert_eq!((f.online(), f.unread()), (2, 3), "away counts as online; unread adds up");
        assert_eq!(f.friends[1].person.name(), "bo", "no display name: the username");
        assert_eq!(f.friends[2].presence, Presence::Offline);
        assert_eq!((f.incoming[0].id.as_str(), f.incoming[0].person.name()), ("r1", "Dee"));
        assert_eq!(f.outgoing[0].person.name(), "Eve");
    }

    #[test]
    fn refusals_become_words_and_a_missing_username_is_its_own_case() {
        assert_eq!(interpret(409, None, r#"{"error": "username-needed", "message": "x"}"#), Err(Problem::UsernameNeeded));
        assert_eq!(interpret(401, None, "{}"), Err(Problem::Expired));
        assert_eq!(interpret(429, Some(12), "{}"), Err(Problem::RateLimited(12)));
        assert_eq!(interpret(503, None, "{}"), Err(Problem::Maintenance));
        assert_eq!(
            interpret(404, None, r#"{"error": "no-such-person", "message": "Nobody has that username."}"#),
            Err(Problem::Refused("no-such-person".into(), "Nobody has that username.".into()))
        );
        assert_eq!(interpret(200, None, r#"{"state": "friends"}"#), Ok(json!({"state": "friends"})));
    }

    #[test]
    fn the_services_other_refusals_keep_their_codes() {
        // 409 too-many-requests (50 pending outgoing) is a refusal of its own, not the username case
        assert_eq!(
            interpret(409, None, r#"{"error": "too-many-requests", "message": "You have too many requests waiting."}"#),
            Err(Problem::Refused("too-many-requests".into(), "You have too many requests waiting.".into()))
        );
        assert!(matches!(interpret(400, None, r#"{"error": "unknown-client", "message": ""}"#),
            Err(Problem::Refused(c, w)) if c == "unknown-client" && w.contains("400")));
        assert!(matches!(interpret(400, None, r#"{"error": "invalid-request"}"#), Err(Problem::Refused(c, _)) if c == "invalid-request"));
        assert_eq!(interpret(429, None, "{}"), Err(Problem::RateLimited(30)), "no Retry-After: a default wait");
    }

    #[test]
    fn a_no_content_answer_is_success() {
        // respond, cancel, remove and presence answer 204 with no body
        assert_eq!(interpret(204, None, ""), Ok(Value::Null));
    }

    #[test]
    fn a_search_finds_one_person_or_nobody() {
        let one = people_from_json(&json!({"people": [{"id": "a1", "username": "ada", "display_name": "Ada"}]}));
        assert_eq!(one, vec![Person { id: "a1".into(), username: Some("ada".into()), display_name: "Ada".into() }]);
        assert!(people_from_json(&json!({"people": []})).is_empty(), "a query that is not a username finds nobody");
        assert!(people_from_json(&Value::Null).is_empty());
    }

    #[test]
    fn a_conversation_is_read_oldest_first_with_whether_more_remain() {
        let c = Conversation::from_json(&json!({"messages": [
            {"id": "m1", "from": "a1", "text": "hi", "sent_at": "2026-10-08T00:00:00Z"},
            {"id": "m2", "from": "me", "text": "hello\nthere", "sent_at": "2026-10-08T00:00:05Z"}
        ], "more": true}));
        assert_eq!((c.messages.len(), c.messages[1].text.as_str(), c.more), (2, "hello\nthere", true));
        assert_eq!(Conversation::from_json(&json!({})), Conversation::default());
    }

    #[test]
    fn presence_words_round_trip() {
        for p in [Presence::Online, Presence::Away, Presence::Offline] {
            assert_eq!(Presence::from_word(Some(p.as_str())), p);
        }
        assert_eq!(Presence::from_word(Some("busy")), Presence::Offline);
    }
}
