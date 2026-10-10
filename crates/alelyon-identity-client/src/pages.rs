//! Pages, organizations and titles: the identity service's pages routes (`docs/pages-contract.md`). Every route is a
//! POST under `/identity/native/pages/` whose body carries the client id and the session's refresh token, which these
//! routes check WITHOUT rotating it, as the social routes do. An account needs a username first.
//!
//! A page is public: once a person opens theirs, anyone can read their name, title, biography and posts, signed in or
//! not. A person here is never an email address.
//!
//! The name line is the service's: `[ Founder | CEO ] Ada Lovelace (ALEL)`, the bracket the person's title and each
//! parenthesis an organization they chose to show. An application draws the verified mark from `title_verified`
//! only, never by reading the line, because a display name is free text.

use serde_json::{Value, json};

use crate::client::{CLIENT_ID, Client};
pub use super::social::Problem;
use super::social;

/// An organization as a page shows it.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Org {
    pub id: String,
    /// 2-5 letters, upper case, unique.
    pub ticker: String,
    pub name: String,
}

impl Org {
    pub fn from_json(v: &Value) -> Option<Org> {
        v.as_object()?;
        Some(Org { id: str_of(v, "id"), ticker: str_of(v, "ticker"), name: str_of(v, "name") })
    }
}

/// Someone as the pages show them.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Person {
    pub id: String,
    pub username: Option<String>,
    pub display_name: String,
    /// The person's own title (`Founder | CEO`), or empty.
    pub title: String,
    /// The organization the title names, when it names one.
    pub title_org: Option<Org>,
    /// An owner or admin of `title_org` confirmed this exact title.
    pub title_verified: bool,
    /// The organizations the person chose to show, in their order.
    pub tickers: Vec<Org>,
    /// The service's composed line: `[ title ] Display Name (TICK)`.
    pub name_line: String,
    /// The person opened a public page.
    pub has_page: bool,
}

impl Person {
    pub fn from_json(v: &Value) -> Person {
        Person {
            id: str_of(v, "id"),
            username: v.get("username").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string),
            display_name: str_of(v, "display_name"),
            title: str_of(v, "title"),
            title_org: v.get("title_org").and_then(Org::from_json),
            title_verified: v.get("title_verified").and_then(Value::as_bool).unwrap_or(false),
            tickers: list_of(v, "tickers").iter().filter_map(Org::from_json).collect(),
            name_line: str_of(v, "name_line"),
            has_page: v.get("has_page").and_then(Value::as_bool).unwrap_or(false),
        }
    }

    /// The name a person sees: the display name, else the username.
    pub fn name(&self) -> &str {
        if !self.display_name.is_empty() { &self.display_name } else { self.username.as_deref().unwrap_or("someone") }
    }

    /// `@username`, or empty without one.
    pub fn handle(&self) -> String {
        self.username.as_deref().map(|u| format!("@{u}")).unwrap_or_default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    /// 1-500 characters.
    #[default]
    Post,
    /// A title of 1-150 characters and a body of up to 20000.
    Article,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Post => "post",
            Kind::Article => "article",
        }
    }

    fn from_word(word: Option<&str>) -> Kind {
        if word == Some("article") { Kind::Article } else { Kind::Post }
    }
}

/// The longest post, and an article's longest title and body, in characters after trimming.
pub const POST_MAX_CHARS: usize = 500;
pub const ARTICLE_TITLE_MAX_CHARS: usize = 150;
pub const ARTICLE_MAX_CHARS: usize = 20000;
/// The longest title and biography.
pub const TITLE_MAX_CHARS: usize = 60;
pub const BIO_MAX_CHARS: usize = 1000;

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Post {
    pub id: String,
    pub kind: Kind,
    pub author: Person,
    /// The organization it was posted as, when it was.
    pub org: Option<Org>,
    /// An article's title.
    pub title: Option<String>,
    pub text: String,
    /// The post this one answers.
    pub reply_to: Option<String>,
    /// What this one reposts or quotes, as the reader may see it (None when it is gone or hidden from them).
    pub repost_of: Option<Box<Post>>,
    pub repost_of_id: Option<String>,
    pub created_at: String,
    pub edited_at: Option<String>,
    /// Deleted with replies under it: kept so the thread keeps its shape, with its text erased.
    pub deleted: bool,
    pub likes: u64,
    pub replies: u64,
    pub reposts: u64,
    /// Whether the reader liked it.
    pub liked: bool,
}

impl Post {
    pub fn from_json(v: &Value) -> Post {
        Post {
            id: str_of(v, "id"),
            kind: Kind::from_word(v.get("kind").and_then(Value::as_str)),
            author: Person::from_json(v.get("author").unwrap_or(&Value::Null)),
            org: v.get("org").and_then(Org::from_json),
            title: opt_str(v, "title"),
            text: str_of(v, "text"),
            reply_to: opt_str(v, "reply_to"),
            repost_of: v.get("repost_of").filter(|r| r.is_object()).map(|r| Box::new(Post::from_json(r))),
            repost_of_id: opt_str(v, "repost_of_id"),
            created_at: str_of(v, "created_at"),
            edited_at: opt_str(v, "edited_at"),
            deleted: v.get("deleted").and_then(Value::as_bool).unwrap_or(false),
            likes: v.get("likes").and_then(Value::as_u64).unwrap_or(0),
            replies: v.get("replies").and_then(Value::as_u64).unwrap_or(0),
            reposts: v.get("reposts").and_then(Value::as_u64).unwrap_or(0),
            liked: v.get("liked").and_then(Value::as_bool).unwrap_or(false),
        }
    }

    /// A repost with no words of its own: it is drawn as what it carries.
    pub fn is_plain_repost(&self) -> bool {
        self.repost_of_id.is_some() && self.text.is_empty() && !self.deleted
    }
}

/// A page of posts, newest first (a thread's replies oldest first), and whether more remain.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Posts {
    pub posts: Vec<Post>,
    pub more: bool,
}

impl Posts {
    pub fn from_json(v: &Value) -> Posts {
        Posts {
            posts: list_of(v, "posts").iter().map(Post::from_json).collect(),
            more: v.get("more").and_then(Value::as_bool).unwrap_or(false),
        }
    }

    /// The id to ask the next page `before`, when more remain.
    pub fn next(&self) -> Option<String> {
        self.more.then(|| self.posts.last().map(|p| p.id.clone())).flatten()
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct PersonPage {
    pub person: Person,
    pub bio: String,
    pub followers: u64,
    pub following: u64,
    pub posts: Posts,
    pub you_follow: bool,
    pub you_block: bool,
}

impl PersonPage {
    pub fn from_json(v: &Value) -> PersonPage {
        PersonPage {
            person: Person::from_json(v.get("person").unwrap_or(&Value::Null)),
            bio: str_of(v, "bio"),
            followers: v.get("followers").and_then(Value::as_u64).unwrap_or(0),
            following: v.get("following").and_then(Value::as_u64).unwrap_or(0),
            posts: Posts::from_json(v),
            you_follow: v.get("you_follow").and_then(Value::as_bool).unwrap_or(false),
            you_block: v.get("you_block").and_then(Value::as_bool).unwrap_or(false),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct OrgPage {
    pub org: Org,
    pub about: String,
    pub created_at: String,
    pub followers: u64,
    /// Members who have a page and show this organization's ticker.
    pub members: Vec<Person>,
    pub posts: Posts,
    pub you_follow: bool,
    /// The reader's role (`owner`, `admin`, `member`) and state (`active`, `requested`, `invited`), when they have one.
    pub your_role: Option<String>,
    pub your_state: Option<String>,
}

impl OrgPage {
    pub fn from_json(v: &Value) -> OrgPage {
        OrgPage {
            org: v.get("org").and_then(Org::from_json).unwrap_or_default(),
            about: str_of(v, "about"),
            created_at: str_of(v, "created_at"),
            followers: v.get("followers").and_then(Value::as_u64).unwrap_or(0),
            members: list_of(v, "members").iter().map(Person::from_json).collect(),
            posts: Posts::from_json(v),
            you_follow: v.get("you_follow").and_then(Value::as_bool).unwrap_or(false),
            your_role: opt_str(v, "your_role"),
            your_state: opt_str(v, "your_state"),
        }
    }

    /// The reader is an active owner or admin.
    pub fn you_admin(&self) -> bool {
        self.your_state.as_deref() == Some("active") && matches!(self.your_role.as_deref(), Some("owner" | "admin"))
    }
}

/// A post, the post it answers (a deleted one as its tombstone), and a page of its replies, oldest first.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Thread {
    pub post: Post,
    pub parent: Option<Post>,
    pub replies: Posts,
}

impl Thread {
    pub fn from_json(v: &Value) -> Thread {
        Thread {
            post: Post::from_json(v.get("post").unwrap_or(&Value::Null)),
            parent: v.get("parent").filter(|p| p.is_object()).map(Post::from_json),
            replies: Posts {
                posts: list_of(v, "replies").iter().map(Post::from_json).collect(),
                more: v.get("more").and_then(Value::as_bool).unwrap_or(false),
            },
        }
    }
}

/// One of the reader's memberships, in any state.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Membership {
    pub org: Org,
    pub role: String,
    /// `active`, `requested` (the reader asked) or `invited` (the organization asked).
    pub state: String,
    /// The reader shows this ticker after their name.
    pub shown: bool,
    pub confirmed_title: Option<String>,
}

/// The reader's own page: their card, biography and every membership.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Me {
    pub me: Person,
    pub bio: String,
    pub memberships: Vec<Membership>,
}

impl Me {
    pub fn from_json(v: &Value) -> Me {
        Me {
            me: Person::from_json(v.get("me").unwrap_or(&Value::Null)),
            bio: str_of(v, "bio"),
            memberships: list_of(v, "memberships")
                .iter()
                .map(|m| Membership {
                    org: m.get("org").and_then(Org::from_json).unwrap_or_default(),
                    role: str_of(m, "role"),
                    state: str_of(m, "state"),
                    shown: m.get("shown").and_then(Value::as_bool).unwrap_or(false),
                    confirmed_title: opt_str(m, "confirmed_title"),
                })
                .collect(),
        }
    }

    /// The organizations the reader is an active member of.
    pub fn active(&self) -> impl Iterator<Item = &Membership> {
        self.memberships.iter().filter(|m| m.state == "active")
    }

    /// The tickers the reader may post as (active owner or admin).
    pub fn admin_of(&self) -> Vec<String> {
        self.active().filter(|m| m.role == "owner" || m.role == "admin").map(|m| m.org.ticker.clone()).collect()
    }
}

/// A row of an organization's member list (owners and admins only).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Member {
    pub person: Person,
    pub role: String,
    pub state: String,
    pub confirmed_title: Option<String>,
}

pub fn members_from_json(v: &Value) -> Vec<Member> {
    list_of(v, "members")
        .iter()
        .map(|m| Member {
            person: Person::from_json(m.get("person").unwrap_or(&Value::Null)),
            role: str_of(m, "role"),
            state: str_of(m, "state"),
            confirmed_title: opt_str(m, "confirmed_title"),
        })
        .collect()
}

/// Why a report is filed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Spam,
    Harassment,
    Impersonation,
    Illegal,
    Other,
}

impl Reason {
    pub const ALL: [Reason; 5] = [Reason::Spam, Reason::Harassment, Reason::Impersonation, Reason::Illegal, Reason::Other];

    pub fn as_str(self) -> &'static str {
        match self {
            Reason::Spam => "spam",
            Reason::Harassment => "harassment",
            Reason::Impersonation => "impersonation",
            Reason::Illegal => "illegal",
            Reason::Other => "other",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Reason::Spam => "Spam",
            Reason::Harassment => "Harassment",
            Reason::Impersonation => "Impersonation",
            Reason::Illegal => "Illegal",
            Reason::Other => "Something else",
        }
    }
}

/// What a report is about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Post(String),
    Person(String),
    Org(String),
}

/// A post to send.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NewPost {
    pub kind: Kind,
    pub text: String,
    /// An article's title.
    pub title: Option<String>,
    pub reply_to: Option<String>,
    /// Repost (with no text) or quote (with text) this post.
    pub repost_of: Option<String>,
    /// Post as this organization (its ticker); the reader must be an owner or admin.
    pub as_org: Option<String>,
}

impl NewPost {
    /// The route's body, without the credential.
    pub fn body(&self) -> Value {
        let mut body = json!({"kind": self.kind.as_str(), "text": self.text});
        for (key, value) in [("title", &self.title), ("reply_to", &self.reply_to), ("repost_of", &self.repost_of), ("as_org", &self.as_org)] {
            if let Some(v) = value {
                body[key] = json!(v);
            }
        }
        body
    }
}

fn str_of(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or_default().to_string()
}

fn opt_str(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_string)
}

fn list_of(v: &Value, k: &str) -> Vec<Value> {
    v.get(k).and_then(Value::as_array).cloned().unwrap_or_default()
}

/// Words for a person about a refusal, in the pages' terms.
pub fn words(problem: &Problem) -> String {
    match problem {
        Problem::Unreachable(why) => format!("Pages could not be reached ({why})."),
        Problem::Expired => "Your session has ended. Sign in again to see pages.".into(),
        Problem::UsernameNeeded => "Choose a username first (Friends, in the rail).".into(),
        Problem::RateLimited(s) => format!("That was a lot at once. Try again in {s} seconds."),
        Problem::Maintenance => "Pages are down for scheduled maintenance.".into(),
        p if not_offered(p) => "This service does not offer pages yet.".into(),
        Problem::Refused(_, words) => words.clone(),
    }
}

/// The service answered that it has no such route: an older service, without pages.
pub fn not_offered(problem: &Problem) -> bool {
    matches!(problem, Problem::Refused(code, _) if code.is_empty())
}

/// One pages route by name; the client id and the token are added here.
pub async fn call(client: &Client, token: &str, name: &str, mut body: Value) -> Result<Value, Problem> {
    body["client_id"] = json!(CLIENT_ID);
    body["refresh_token"] = json!(token);
    let (status, retry, text) = client.post_raw(&format!("/identity/native/pages/{name}"), body).await.map_err(Problem::Unreachable)?;
    social::interpret(status, retry, &text)
}

/// The reader's own page (whether or not it is open), with every membership.
pub async fn me(client: &Client, token: &str) -> Result<Me, Problem> {
    call(client, token, "me", json!({})).await.map(|v| Me::from_json(&v))
}

/// Open the reader's public page. From then on anyone can read their name, title, biography and posts.
pub async fn open(client: &Client, token: &str) -> Result<Me, Problem> {
    call(client, token, "open", json!({})).await.map(|v| Me::from_json(&v))
}

/// Close the reader's page: nothing of it is shown, and its title and biography are erased. Its posts are kept,
/// unseen, and show again if the page is opened again.
pub async fn close(client: &Client, token: &str) -> Result<Me, Problem> {
    call(client, token, "close", json!({})).await.map(|v| Me::from_json(&v))
}

/// Change the title, the organization it names (a ticker; empty for none) or the biography. `None` leaves one alone.
pub async fn update(client: &Client, token: &str, title: Option<&str>, title_org: Option<&str>, bio: Option<&str>) -> Result<Me, Problem> {
    let mut body = json!({});
    for (key, value) in [("title", title), ("title_org", title_org), ("bio", bio)] {
        if let Some(v) = value {
            body[key] = json!(v);
        }
    }
    call(client, token, "update", body).await.map(|v| Me::from_json(&v))
}

/// Show exactly these tickers after the reader's name, in this order: none, one or several.
pub async fn set_tickers(client: &Client, token: &str, tickers: &[String]) -> Result<Me, Problem> {
    call(client, token, "tickers", json!({"tickers": tickers})).await.map(|v| Me::from_json(&v))
}

fn paged(mut body: Value, before: Option<&str>) -> Value {
    if let Some(b) = before {
        body["before"] = json!(b);
    }
    body
}

pub async fn person(client: &Client, token: &str, username: &str, before: Option<&str>) -> Result<PersonPage, Problem> {
    call(client, token, "person", paged(json!({"username": username}), before)).await.map(|v| PersonPage::from_json(&v))
}

pub async fn org(client: &Client, token: &str, ticker: &str, before: Option<&str>) -> Result<OrgPage, Problem> {
    call(client, token, "org", paged(json!({"ticker": ticker}), before)).await.map(|v| OrgPage::from_json(&v))
}

/// The reader's own posts and those of the people and organizations they follow, newest first.
pub async fn feed(client: &Client, token: &str, before: Option<&str>) -> Result<Posts, Problem> {
    call(client, token, "feed", paged(json!({}), before)).await.map(|v| Posts::from_json(&v))
}

/// Every post on a page, newest first.
pub async fn explore(client: &Client, token: &str, before: Option<&str>) -> Result<Posts, Problem> {
    call(client, token, "explore", paged(json!({}), before)).await.map(|v| Posts::from_json(&v))
}

pub async fn thread(client: &Client, token: &str, post_id: &str, before: Option<&str>) -> Result<Thread, Problem> {
    call(client, token, "thread", paged(json!({"post_id": post_id}), before)).await.map(|v| Thread::from_json(&v))
}

fn post_of(v: &Value) -> Post {
    Post::from_json(v.get("post").unwrap_or(&Value::Null))
}

pub async fn post(client: &Client, token: &str, new: &NewPost) -> Result<Post, Problem> {
    call(client, token, "post", new.body()).await.map(|v| post_of(&v))
}

/// Change a post's text (and an article's title).
pub async fn edit(client: &Client, token: &str, post_id: &str, text: &str, title: Option<&str>) -> Result<Post, Problem> {
    let mut body = json!({"post_id": post_id, "text": text});
    if let Some(t) = title {
        body["title"] = json!(t);
    }
    call(client, token, "edit", body).await.map(|v| post_of(&v))
}

pub async fn delete(client: &Client, token: &str, post_id: &str) -> Result<(), Problem> {
    call(client, token, "delete", json!({"post_id": post_id})).await.map(|_| ())
}

pub async fn like(client: &Client, token: &str, post_id: &str, like: bool) -> Result<(), Problem> {
    call(client, token, "like", json!({"post_id": post_id, "like": like})).await.map(|_| ())
}

pub async fn follow_person(client: &Client, token: &str, username: &str, follow: bool) -> Result<(), Problem> {
    call(client, token, "follow", json!({"username": username, "follow": follow})).await.map(|_| ())
}

pub async fn follow_org(client: &Client, token: &str, ticker: &str, follow: bool) -> Result<(), Problem> {
    call(client, token, "follow", json!({"ticker": ticker, "follow": follow})).await.map(|_| ())
}

/// Block (or unblock) a person: neither sees the other's posts while signed in, and follows end both ways.
pub async fn block(client: &Client, token: &str, username: &str, block: bool) -> Result<(), Problem> {
    call(client, token, "block", json!({"username": username, "block": block})).await.map(|_| ())
}

pub async fn report(client: &Client, token: &str, reason: Reason, note: &str, target: &Target) -> Result<String, Problem> {
    let mut body = json!({"reason": reason.as_str(), "note": note});
    match target {
        Target::Post(id) => body["post_id"] = json!(id),
        Target::Person(username) => body["username"] = json!(username),
        Target::Org(ticker) => body["ticker"] = json!(ticker),
    }
    call(client, token, "report", body).await.map(|v| str_of(&v, "report_id"))
}

/// A new organization, owned by the reader, who shows its ticker. The ticker is 2-5 letters and unique.
pub async fn org_create(client: &Client, token: &str, ticker: &str, name: &str, about: &str) -> Result<Org, Problem> {
    call(client, token, "org-create", json!({"ticker": ticker, "name": name, "about": about}))
        .await
        .map(|v| v.get("org").and_then(Org::from_json).unwrap_or_default())
}

pub async fn org_update(client: &Client, token: &str, ticker: &str, name: Option<&str>, about: Option<&str>) -> Result<Org, Problem> {
    let mut body = json!({"ticker": ticker});
    if let Some(n) = name {
        body["name"] = json!(n);
    }
    if let Some(a) = about {
        body["about"] = json!(a);
    }
    call(client, token, "org-update", body).await.map(|v| v.get("org").and_then(Org::from_json).unwrap_or_default())
}

/// Every member, request and invitation (owners and admins only).
pub async fn org_members(client: &Client, token: &str, ticker: &str) -> Result<Vec<Member>, Problem> {
    call(client, token, "org-members", json!({"ticker": ticker})).await.map(|v| members_from_json(&v))
}

fn state_of(v: &Value) -> String {
    str_of(v, "state")
}

/// Ask to join, or accept an invitation: `"requested"` or `"member"`.
pub async fn org_join(client: &Client, token: &str, ticker: &str) -> Result<String, Problem> {
    call(client, token, "org-join", json!({"ticker": ticker})).await.map(|v| state_of(&v))
}

/// Invite a person, or approve their request: `"invited"` or `"member"`.
pub async fn org_invite(client: &Client, token: &str, ticker: &str, username: &str) -> Result<String, Problem> {
    call(client, token, "org-invite", json!({"ticker": ticker, "username": username})).await.map(|v| state_of(&v))
}

/// Leave, withdraw a request, or decline an invitation.
pub async fn org_leave(client: &Client, token: &str, ticker: &str) -> Result<(), Problem> {
    call(client, token, "org-leave", json!({"ticker": ticker})).await.map(|_| ())
}

/// Remove a member, refuse a request, or withdraw an invitation.
pub async fn org_remove(client: &Client, token: &str, ticker: &str, member_id: &str) -> Result<(), Problem> {
    call(client, token, "org-remove", json!({"ticker": ticker, "member_id": member_id})).await.map(|_| ())
}

/// Owners only: `owner`, `admin` or `member`.
pub async fn org_role(client: &Client, token: &str, ticker: &str, member_id: &str, role: &str) -> Result<(), Problem> {
    call(client, token, "org-role", json!({"ticker": ticker, "member_id": member_id, "role": role})).await.map(|_| ())
}

/// Confirm a member's current title (which must name this organization), or withdraw a confirmation.
pub async fn confirm_title(client: &Client, token: &str, ticker: &str, member_id: &str, confirm: bool) -> Result<Option<String>, Problem> {
    call(client, token, "org-confirm-title", json!({"ticker": ticker, "member_id": member_id, "confirm": confirm}))
        .await
        .map(|v| opt_str(&v, "confirmed_title"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(username: &str) -> Value {
        json!({"id": format!("acct_{username}"), "username": username, "display_name": "Ada Lovelace",
               "title": "Founder | CEO", "title_org": {"id": "org_1", "ticker": "LACE", "name": "Lace Labs"},
               "title_verified": true, "tickers": [{"id": "org_1", "ticker": "LACE", "name": "Lace Labs"}],
               "name_line": "[ Founder | CEO ] Ada Lovelace (LACE)", "has_page": true})
    }

    #[test]
    fn a_person_reads_title_tickers_and_the_name_line_and_never_needs_an_email() {
        let p = Person::from_json(&card("ada"));
        assert_eq!(p.name_line, "[ Founder | CEO ] Ada Lovelace (LACE)");
        assert!(p.title_verified && p.has_page);
        assert_eq!(p.title_org.as_ref().map(|o| o.ticker.as_str()), Some("LACE"));
        assert_eq!(p.tickers.len(), 1);
        assert_eq!(p.handle(), "@ada");
        let bare = Person::from_json(&json!({"id": "a", "username": null, "display_name": ""}));
        assert_eq!((bare.name(), bare.handle().as_str(), bare.title_verified), ("someone", "", false));
        assert!(bare.title_org.is_none() && bare.tickers.is_empty() && !bare.has_page);
    }

    #[test]
    fn a_post_reads_its_repost_and_counts_and_a_plain_repost_is_told_apart() {
        let inner = json!({"id": "pst_1", "kind": "article", "author": card("ada"), "org": null, "title": "On Engines",
                           "text": "Body", "reply_to": null, "repost_of": null, "repost_of_id": null,
                           "created_at": "2026-10-09T00:00:00Z", "edited_at": null, "deleted": false,
                           "likes": 2, "replies": 1, "reposts": 1, "liked": true});
        let plain = Post::from_json(&json!({"id": "pst_2", "kind": "post", "author": card("bob"), "text": "",
                                            "repost_of": inner, "repost_of_id": "pst_1", "created_at": "t",
                                            "liked": null}));
        assert!(plain.is_plain_repost());
        let carried = plain.repost_of.as_deref().expect("carries the original");
        assert_eq!((carried.kind, carried.title.as_deref(), carried.likes, carried.liked), (Kind::Article, Some("On Engines"), 2, true));
        assert!(!plain.liked, "an anonymous null reads as not liked");
        let quote = Post::from_json(&json!({"id": "pst_3", "text": "Read this", "repost_of_id": "pst_1"}));
        assert!(!quote.is_plain_repost() && quote.repost_of.is_none(), "a quote whose original is gone still shows its words");
    }

    #[test]
    fn pages_of_posts_say_where_the_next_one_starts() {
        let page = Posts::from_json(&json!({"posts": [{"id": "pst_a"}, {"id": "pst_b"}], "more": true}));
        assert_eq!(page.next().as_deref(), Some("pst_b"));
        assert_eq!(Posts::from_json(&json!({"posts": [{"id": "pst_a"}], "more": false})).next(), None);
        assert_eq!(Posts::from_json(&Value::Null), Posts::default());
    }

    #[test]
    fn the_pages_of_people_orgs_and_threads_read() {
        let person = PersonPage::from_json(&json!({"person": card("ada"), "bio": "Notes.", "followers": 3, "following": 1,
                                                   "posts": [], "more": false, "you_follow": true, "you_block": null}));
        assert_eq!((person.bio.as_str(), person.followers, person.you_follow, person.you_block), ("Notes.", 3, true, false));
        let org = OrgPage::from_json(&json!({"org": {"id": "org_1", "ticker": "LACE", "name": "Lace Labs"}, "about": "We build.",
                                             "created_at": "t", "followers": 1, "members": [card("ada")], "posts": [],
                                             "more": false, "you_follow": false, "your_role": "admin", "your_state": "active"}));
        assert!(org.you_admin());
        assert_eq!((org.org.ticker.as_str(), org.members.len()), ("LACE", 1));
        let requested = OrgPage::from_json(&json!({"your_role": "member", "your_state": "requested"}));
        assert!(!requested.you_admin());
        let thread = Thread::from_json(&json!({"post": {"id": "pst_2", "reply_to": "pst_1"},
                                               "parent": {"id": "pst_1", "deleted": true, "text": ""},
                                               "replies": [{"id": "pst_3"}], "more": false}));
        assert!(thread.parent.as_ref().is_some_and(|p| p.deleted));
        assert_eq!(thread.replies.posts.len(), 1);
    }

    #[test]
    fn my_page_lists_memberships_and_where_i_may_post() {
        let me = Me::from_json(&json!({"me": card("ada"), "bio": "", "memberships": [
            {"org": {"id": "o1", "ticker": "LACE", "name": "Lace"}, "role": "owner", "state": "active", "shown": true},
            {"org": {"id": "o2", "ticker": "MITX", "name": "Mit"}, "role": "member", "state": "active", "shown": false},
            {"org": {"id": "o3", "ticker": "ZZZ", "name": "Zed"}, "role": "member", "state": "invited", "shown": false}
        ]}));
        assert_eq!(me.active().count(), 2);
        assert_eq!(me.admin_of(), vec!["LACE".to_string()]);
    }

    #[test]
    fn a_new_post_sends_only_what_it_has() {
        let plain = NewPost { text: "hi".into(), ..NewPost::default() }.body();
        assert_eq!(plain, json!({"kind": "post", "text": "hi"}));
        let article = NewPost { kind: Kind::Article, text: "Body".into(), title: Some("T".into()), as_org: Some("LACE".into()), ..NewPost::default() }.body();
        assert_eq!(article, json!({"kind": "article", "text": "Body", "title": "T", "as_org": "LACE"}));
        let repost = NewPost { repost_of: Some("pst_1".into()), ..NewPost::default() }.body();
        assert_eq!(repost, json!({"kind": "post", "text": "", "repost_of": "pst_1"}));
    }

    #[test]
    fn refusals_read_in_the_pages_terms() {
        let page_needed = social::interpret(409, None, r#"{"error": "page-needed", "message": "Open your page first."}"#);
        assert_eq!(page_needed, Err(Problem::Refused("page-needed".into(), "Open your page first.".into())));
        assert_eq!(words(&page_needed.unwrap_err()), "Open your page first.");
        assert!(words(&Problem::Maintenance).starts_with("Pages"));
        // An older service has no pages routes: its 404 carries no code.
        let older = social::interpret(404, None, r#"{"detail": "Not Found"}"#).unwrap_err();
        assert!(not_offered(&older));
        assert_eq!(words(&older), "This service does not offer pages yet.");
        assert!(!not_offered(&Problem::Refused("no-such-post".into(), "That post is not there.".into())));
        let reason_words: Vec<_> = Reason::ALL.iter().map(|r| r.as_str()).collect();
        assert_eq!(reason_words, ["spam", "harassment", "impersonation", "illegal", "other"]);
    }
}
