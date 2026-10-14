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

/// Why a post made as an organization shows no author. Only that organization's active owners and admins are told.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthorFlag {
    /// The author's account is no longer active: deleted, or disabled by the service.
    Deleted,
    /// A moderator suspended the author's page.
    Suspended,
}

impl AuthorFlag {
    pub fn from_word(word: Option<&str>) -> Option<AuthorFlag> {
        match word {
            Some("deleted") => Some(AuthorFlag::Deleted),
            Some("suspended") => Some(AuthorFlag::Suspended),
            _ => None,
        }
    }

    /// What an organization's admin reads on the post.
    pub fn words(self) -> &'static str {
        match self {
            AuthorFlag::Deleted => "Posted by an account that no longer exists.",
            AuthorFlag::Suspended => "Posted by someone whose page is suspended.",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Post {
    pub id: String,
    pub kind: Kind,
    /// Who posted it. None for a post made as an organization whose author is gone: it stays the organization's.
    pub author: Option<Person>,
    /// Why `author` is None, for the organization's owners and admins only.
    pub author_flag: Option<AuthorFlag>,
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
    /// Whether the reader saved it (`bookmark`).
    pub bookmarked: bool,
    /// The poll it carries; boxed, since few posts carry one and a post is often held by value.
    pub poll: Option<Box<Poll>>,
    /// Who may reply to it.
    pub replies_from: ReplyRule,
    /// Whether the reader may reply.
    pub you_can_reply: bool,
    /// The job opening it is (an organization's article); boxed, as the poll is.
    pub job: Option<Box<Job>>,
}

impl Post {
    pub fn from_json(v: &Value) -> Post {
        Post {
            id: str_of(v, "id"),
            kind: Kind::from_word(v.get("kind").and_then(Value::as_str)),
            author: v.get("author").filter(|a| a.is_object()).map(Person::from_json),
            author_flag: AuthorFlag::from_word(v.get("author_flag").and_then(Value::as_str)),
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
            bookmarked: v.get("bookmarked").and_then(Value::as_bool).unwrap_or(false),
            poll: v.get("poll").filter(|p| p.is_object()).map(|p| Box::new(Poll::from_json(p))),
            replies_from: ReplyRule::from_word(v.get("replies_from").and_then(Value::as_str).unwrap_or("")),
            you_can_reply: v.get("you_can_reply").and_then(Value::as_bool).unwrap_or(true),
            job: v.get("job").filter(|j| j.is_object()).map(|j| Box::new(Job::from_json(j))),
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

/// Who may reply in a conversation, as its first post set it; each reply under that post carries the same rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReplyRule {
    #[default]
    Everyone,
    /// The first post's author's followers (or its organization's, for a post made as one).
    Followers,
    /// The people the first post mentions.
    Mentioned,
}

impl ReplyRule {
    pub const ALL: [ReplyRule; 3] = [ReplyRule::Everyone, ReplyRule::Followers, ReplyRule::Mentioned];

    pub fn as_str(self) -> &'static str {
        match self {
            ReplyRule::Everyone => "everyone",
            ReplyRule::Followers => "followers",
            ReplyRule::Mentioned => "mentioned",
        }
    }

    /// An unknown rule reads as the widest, as the service would never send one.
    pub fn from_word(word: &str) -> ReplyRule {
        match word {
            "followers" => ReplyRule::Followers,
            "mentioned" => ReplyRule::Mentioned,
            _ => ReplyRule::Everyone,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReplyRule::Everyone => "Everyone can reply",
            ReplyRule::Followers => "Your followers can reply",
            ReplyRule::Mentioned => "People you mention can reply",
        }
    }
}

impl std::fmt::Display for ReplyRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A poll as a post carries it. `votes` is None until the reader voted or it closed (its author always sees them).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Poll {
    pub options: Vec<String>,
    pub votes: Option<Vec<u64>>,
    pub total: u64,
    pub closes_at: String,
    pub closed: bool,
    /// The option the reader chose.
    pub you_voted: Option<usize>,
}

impl Poll {
    pub fn from_json(v: &Value) -> Poll {
        Poll {
            options: list_of(v, "options").iter().filter_map(|o| o.as_str().map(str::to_string)).collect(),
            votes: v.get("votes").and_then(Value::as_array).map(|a| a.iter().map(|n| n.as_u64().unwrap_or(0)).collect()),
            total: v.get("total").and_then(Value::as_u64).unwrap_or(0),
            closes_at: str_of(v, "closes_at"),
            closed: v.get("closed").and_then(Value::as_bool).unwrap_or(false),
            you_voted: v.get("you_voted").and_then(Value::as_u64).map(|n| n as usize),
        }
    }

    /// Each option's share of the votes, in whole percent, when the votes are shown.
    pub fn shares(&self) -> Option<Vec<u64>> {
        let votes = self.votes.as_ref()?;
        let total: u64 = votes.iter().sum();
        Some(votes.iter().map(|v| (v * 100 + total / 2).checked_div(total).unwrap_or(0)).collect())
    }
}

/// A poll a new post carries: 2-4 options of 1-25 characters, open 1-168 hours.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NewPoll {
    pub options: Vec<String>,
    pub hours: u32,
}

/// Where a job is done.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Workplace {
    #[default]
    OnSite,
    Hybrid,
    Remote,
}

impl Workplace {
    pub const ALL: [Workplace; 3] = [Workplace::OnSite, Workplace::Hybrid, Workplace::Remote];

    pub fn as_str(self) -> &'static str {
        match self {
            Workplace::OnSite => "on-site",
            Workplace::Hybrid => "hybrid",
            Workplace::Remote => "remote",
        }
    }

    pub fn from_word(word: &str) -> Workplace {
        match word {
            "hybrid" => Workplace::Hybrid,
            "remote" => Workplace::Remote,
            _ => Workplace::OnSite,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Workplace::OnSite => "On-site",
            Workplace::Hybrid => "Hybrid",
            Workplace::Remote => "Remote",
        }
    }
}

impl std::fmt::Display for Workplace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// The kind of employment a job offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Employment {
    #[default]
    FullTime,
    PartTime,
    Contract,
    Temporary,
    Internship,
}

impl Employment {
    pub const ALL: [Employment; 5] =
        [Employment::FullTime, Employment::PartTime, Employment::Contract, Employment::Temporary, Employment::Internship];

    pub fn as_str(self) -> &'static str {
        match self {
            Employment::FullTime => "full-time",
            Employment::PartTime => "part-time",
            Employment::Contract => "contract",
            Employment::Temporary => "temporary",
            Employment::Internship => "internship",
        }
    }

    pub fn from_word(word: &str) -> Employment {
        match word {
            "part-time" => Employment::PartTime,
            "contract" => Employment::Contract,
            "temporary" => Employment::Temporary,
            "internship" => Employment::Internship,
            _ => Employment::FullTime,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Employment::FullTime => "Full-time",
            Employment::PartTime => "Part-time",
            Employment::Contract => "Contract",
            Employment::Temporary => "Temporary",
            Employment::Internship => "Internship",
        }
    }
}

impl std::fmt::Display for Employment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A job opening: an organization's article that people apply for.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Job {
    pub location: String,
    pub workplace: Workplace,
    pub employment: Employment,
    /// As the organization wrote it; empty for none.
    pub pay: String,
    pub closes_at: String,
    pub closed: bool,
    /// Whether the reader applied.
    pub applied: bool,
    /// How many applied: for the organization's owners and admins only.
    pub applicants: Option<u64>,
}

impl Job {
    pub fn from_json(v: &Value) -> Job {
        Job {
            location: str_of(v, "location"),
            workplace: Workplace::from_word(v.get("workplace").and_then(Value::as_str).unwrap_or("")),
            employment: Employment::from_word(v.get("employment").and_then(Value::as_str).unwrap_or("")),
            pay: str_of(v, "pay"),
            closes_at: str_of(v, "closes_at"),
            closed: v.get("closed").and_then(Value::as_bool).unwrap_or(false),
            applied: v.get("applied").and_then(Value::as_bool).unwrap_or(false),
            applicants: v.get("applicants").and_then(Value::as_u64),
        }
    }
}

/// The job a new article opens; it is posted as an organization.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NewJob {
    /// Where; may be empty for a remote job.
    pub location: String,
    pub workplace: Workplace,
    pub employment: Employment,
    /// As written; may be empty.
    pub pay: String,
    /// Open for this many days, 1-90.
    pub days: u32,
}

/// Someone who applied for a job, as its organization's owners and admins read them.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Applicant {
    pub person: Person,
    pub note: String,
    pub applied_at: String,
}

impl Applicant {
    pub fn from_json(v: &Value) -> Applicant {
        Applicant {
            person: Person::from_json(v.get("person").unwrap_or(&Value::Null)),
            note: str_of(v, "note"),
            applied_at: str_of(v, "applied_at"),
        }
    }
}

/// What `jobs` lists: open jobs with these words, at this workplace, of this kind and at this organization.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct JobsQuery {
    pub words: String,
    pub workplace: Option<Workplace>,
    pub employment: Option<Employment>,
    pub ticker: Option<String>,
}

/// A position on a person's page, as LinkedIn lists one.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Experience {
    pub id: String,
    pub title: String,
    /// The Alelyon organization it names, while that is active.
    pub org: Option<Org>,
    /// The employer's name: the organization's, or as written.
    pub org_name: String,
    /// YYYY-MM.
    pub started: String,
    /// YYYY-MM; None while it is the person's position now.
    pub ended: Option<String>,
    pub about: String,
    /// An owner or admin of `org` confirmed it, and it was not edited since.
    pub confirmed: bool,
}

impl Experience {
    pub fn from_json(v: &Value) -> Experience {
        Experience {
            id: str_of(v, "id"),
            title: str_of(v, "title"),
            org: v.get("org").and_then(Org::from_json),
            org_name: str_of(v, "org_name"),
            started: str_of(v, "started"),
            ended: opt_str(v, "ended"),
            about: str_of(v, "about"),
            confirmed: v.get("confirmed").and_then(Value::as_bool).unwrap_or(false),
        }
    }

    /// `2019-03 to 2023-12`, or `2024-01 to now`.
    pub fn span(&self) -> String {
        format!("{} to {}", self.started, self.ended.as_deref().unwrap_or("now"))
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Education {
    pub id: String,
    pub school: String,
    pub degree: String,
    pub field: String,
    /// YYYY, when given.
    pub started: Option<String>,
    pub ended: Option<String>,
}

impl Education {
    pub fn from_json(v: &Value) -> Education {
        Education {
            id: str_of(v, "id"),
            school: str_of(v, "school"),
            degree: str_of(v, "degree"),
            field: str_of(v, "field"),
            started: opt_str(v, "started"),
            ended: opt_str(v, "ended"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Skill {
    pub skill: String,
    /// How many people endorsed it; who they are is not shown.
    pub endorsements: u64,
    /// Whether the reader endorsed it; None for the person themselves or a reader signed out.
    pub you_endorsed: Option<bool>,
}

impl Skill {
    pub fn from_json(v: &Value) -> Skill {
        Skill {
            skill: str_of(v, "skill"),
            endorsements: v.get("endorsements").and_then(Value::as_u64).unwrap_or(0),
            you_endorsed: v.get("you_endorsed").and_then(Value::as_bool),
        }
    }
}

/// A person's experience, education and skills.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Profile {
    pub experience: Vec<Experience>,
    pub education: Vec<Education>,
    pub skills: Vec<Skill>,
}

impl Profile {
    pub fn from_json(v: &Value) -> Profile {
        Profile {
            experience: list_of(v, "experience").iter().map(Experience::from_json).collect(),
            education: list_of(v, "education").iter().map(Education::from_json).collect(),
            skills: list_of(v, "skills").iter().map(Skill::from_json).collect(),
        }
    }
}

/// An experience entry to add (`id` empty) or to replace (`id` set).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NewExperience {
    pub id: String,
    pub title: String,
    /// An Alelyon organization's ticker; empty for an employer named in `org_name`.
    pub ticker: String,
    pub org_name: String,
    /// YYYY-MM.
    pub started: String,
    /// YYYY-MM, or empty while it is the position now.
    pub ended: String,
    pub about: String,
}

impl NewExperience {
    pub fn body(&self) -> Value {
        let mut body = json!({"title": self.title, "started": self.started, "about": self.about});
        for (key, value) in [("entry_id", &self.id), ("ticker", &self.ticker), ("org_name", &self.org_name), ("ended", &self.ended)] {
            if !value.is_empty() {
                body[key] = json!(value);
            }
        }
        body
    }
}

/// An education entry to add (`id` empty) or to replace (`id` set). Years are YYYY or empty.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NewEducation {
    pub id: String,
    pub school: String,
    pub degree: String,
    pub field: String,
    pub started: String,
    pub ended: String,
}

impl NewEducation {
    pub fn body(&self) -> Value {
        let mut body = json!({"school": self.school, "degree": self.degree, "field": self.field});
        for (key, value) in [("entry_id", &self.id), ("started", &self.started), ("ended", &self.ended)] {
            if !value.is_empty() {
                body[key] = json!(value);
            }
        }
        body
    }
}

/// What a search found: the people and organizations that match (on an `All` search or their own) and a page of
/// posts (on an `All` or a `Posts` search).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Found {
    pub people: Vec<Person>,
    pub orgs: Vec<Org>,
    pub posts: Posts,
}

impl Found {
    pub fn from_json(v: &Value) -> Found {
        Found {
            people: list_of(v, "people").iter().map(Person::from_json).collect(),
            orgs: list_of(v, "orgs").iter().filter_map(Org::from_json).collect(),
            posts: Posts::from_json(v),
        }
    }
}

/// Which part of Pages a search reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SearchKind {
    /// A few people and organizations beside the first page of posts.
    #[default]
    All,
    Posts,
    People,
    Orgs,
}

impl SearchKind {
    pub const ALL: [SearchKind; 4] = [SearchKind::All, SearchKind::Posts, SearchKind::People, SearchKind::Orgs];

    pub fn as_str(self) -> &'static str {
        match self {
            SearchKind::All => "all",
            SearchKind::Posts => "posts",
            SearchKind::People => "people",
            SearchKind::Orgs => "orgs",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SearchKind::All => "Top",
            SearchKind::Posts => "Posts",
            SearchKind::People => "People",
            SearchKind::Orgs => "Organizations",
        }
    }
}

/// The posts carrying one hashtag, as the service spelled it (lower case, no `#`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Tagged {
    pub tag: String,
    pub posts: Posts,
}

impl Tagged {
    pub fn from_json(v: &Value) -> Tagged {
        Tagged { tag: str_of(v, "tag"), posts: Posts::from_json(v) }
    }
}

/// A trending hashtag: the posts that carry it, and how many different people posted them.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct TrendingTag {
    pub tag: String,
    pub posts: u64,
    pub people: u64,
}

/// The hashtags the most people used over the last `hours`: 24, or 168 when none qualified in a day.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Trending {
    pub hours: u64,
    pub tags: Vec<TrendingTag>,
}

impl Trending {
    pub fn from_json(v: &Value) -> Trending {
        Trending {
            hours: v.get("hours").and_then(Value::as_u64).unwrap_or(24),
            tags: list_of(v, "tags")
                .iter()
                .map(|t| TrendingTag {
                    tag: str_of(t, "tag"),
                    posts: t.get("posts").and_then(Value::as_u64).unwrap_or(0),
                    people: t.get("people").and_then(Value::as_u64).unwrap_or(0),
                })
                .collect(),
        }
    }
}

/// Why someone is suggested. Only public signals: never whom anyone follows or what anyone liked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Why {
    /// They show an organization the reader belongs to or follows (`Suggestion::org`).
    Org,
    /// They answered or quoted the reader in the last 30 days.
    Replied,
    /// The reader answered or quoted them in the last 30 days.
    YouReplied,
    /// Among the most followed.
    Popular,
    /// They have a page; nothing more.
    #[default]
    Page,
}

impl Why {
    pub fn from_word(word: &str) -> Why {
        match word {
            "org" => Why::Org,
            "replied" => Why::Replied,
            "you-replied" => Why::YouReplied,
            "popular" => Why::Popular,
            _ => Why::Page,
        }
    }

    /// The reason in words, naming the organization for `Org`.
    pub fn words(self, org: Option<&Org>) -> String {
        match (self, org) {
            (Why::Org, Some(o)) => format!("Shows ({})", o.ticker),
            (Why::Org, None) => "Shares an organization".into(),
            (Why::Replied, _) => "Replied to you".into(),
            (Why::YouReplied, _) => "You replied".into(),
            (Why::Popular, _) => "Popular on Pages".into(),
            (Why::Page, _) => "On Pages".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Suggestion {
    pub person: Person,
    pub why: Why,
    pub org: Option<Org>,
}

/// Who to follow, and organizations to follow.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Suggestions {
    pub people: Vec<Suggestion>,
    pub orgs: Vec<Org>,
}

impl Suggestions {
    pub fn from_json(v: &Value) -> Suggestions {
        Suggestions {
            people: list_of(v, "people")
                .iter()
                .map(|s| Suggestion {
                    person: Person::from_json(s.get("person").unwrap_or(&Value::Null)),
                    why: Why::from_word(s.get("reason").and_then(Value::as_str).unwrap_or("")),
                    org: s.get("org").and_then(Org::from_json),
                })
                .collect(),
            orgs: list_of(v, "orgs").iter().filter_map(Org::from_json).collect(),
        }
    }
}

/// Something another person did that concerns the reader (`notifications`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Notification {
    pub id: String,
    /// `like`, `reply`, `repost`, `quote`, `mention`, `follow`, `org-invite`, `org-request`, `org-joined` or
    /// `title-confirmed`.
    pub kind: String,
    /// Who did it, while they can still be shown.
    pub actor: Option<Person>,
    /// The post it concerns, as the reader may see it now: the new post for a reply, a quote or a mention; the
    /// reader's own for a like or a repost.
    pub post: Option<Post>,
    pub org: Option<Org>,
    pub at: String,
    pub read: bool,
}

impl Notification {
    pub fn from_json(v: &Value) -> Notification {
        Notification {
            id: str_of(v, "id"),
            kind: str_of(v, "kind"),
            actor: v.get("actor").filter(|a| a.is_object()).map(Person::from_json),
            post: v.get("post").filter(|p| p.is_object()).map(Post::from_json),
            org: v.get("org").and_then(Org::from_json),
            at: str_of(v, "at"),
            read: v.get("read").and_then(Value::as_bool).unwrap_or(false),
        }
    }
}

/// A page of notifications, newest first, whether more remain, and how many are unread in all.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Notifications {
    pub items: Vec<Notification>,
    pub more: bool,
    pub unread: u64,
}

impl Notifications {
    pub fn from_json(v: &Value) -> Notifications {
        Notifications {
            items: list_of(v, "notifications").iter().map(Notification::from_json).collect(),
            more: v.get("more").and_then(Value::as_bool).unwrap_or(false),
            unread: v.get("unread").and_then(Value::as_u64).unwrap_or(0),
        }
    }

    /// The id to ask the next page `before`, when more remain.
    pub fn next(&self) -> Option<String> {
        self.more.then(|| self.items.last().map(|n| n.id.clone())).flatten()
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
    /// The reader muted this person.
    pub you_mute: bool,
    /// The post the person pinned to the top of the page (on the first page of posts only).
    pub pinned: Option<Post>,
    /// Experience, education and skills (on the first page of posts only).
    pub profile: Option<Profile>,
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
            you_mute: v.get("you_mute").and_then(Value::as_bool).unwrap_or(false),
            pinned: v.get("pinned").filter(|p| p.is_object()).map(Post::from_json),
            profile: v.get("profile").filter(|p| p.is_object()).map(Profile::from_json),
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
    /// The service names this account a moderator: the moderators' calls are open to it.
    pub moderator: bool,
    /// A moderator suspended this account's page: it shows nothing, and posting, liking and following are refused.
    pub suspended: bool,
    /// Alelyon may use this account's public posts to improve its services, AI models included: true until it opts out.
    pub ai_training: bool,
    /// How many of the reader's notifications are unread.
    pub unread_notifications: u64,
    /// The reader's own experience, education and skills.
    pub profile: Profile,
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
            moderator: v.get("moderator").and_then(Value::as_bool).unwrap_or(false),
            suspended: v.get("suspended").and_then(Value::as_bool).unwrap_or(false),
            ai_training: v.get("ai_training").and_then(Value::as_bool).unwrap_or(true),
            unread_notifications: v.get("unread_notifications").and_then(Value::as_u64).unwrap_or(0),
            profile: Profile::from_json(v.get("profile").unwrap_or(&Value::Null)),
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

// ------------------------------------------------------------------- moderation (moderators only)

/// What a report or a moderator's act names, as a moderator sees it: hidden and suspended things included.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ModTarget {
    /// `post`, `person`, `org`, or `gone` when it no longer exists.
    pub kind: String,
    pub id: String,
    pub post: Option<Post>,
    /// The reported person, or a post's author.
    pub person: Option<Person>,
    pub org: Option<Org>,
    /// A person's biography or an organization's about text.
    pub text: String,
    /// The post is hidden.
    pub hidden: bool,
    /// The page (or the post's author's) or the organization is suspended.
    pub suspended: bool,
}

impl ModTarget {
    pub fn from_json(v: &Value) -> ModTarget {
        ModTarget {
            kind: str_of(v, "kind"),
            id: str_of(v, "id"),
            post: v.get("post").filter(|p| p.is_object()).map(Post::from_json),
            person: v.get("person").filter(|p| p.is_object()).map(Person::from_json),
            org: v.get("org").and_then(Org::from_json),
            text: str_of(v, "text"),
            hidden: v.get("hidden").and_then(Value::as_bool).unwrap_or(false),
            suspended: v.get("suspended").and_then(Value::as_bool).unwrap_or(false),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct ModReport {
    pub id: String,
    pub reason: String,
    pub note: String,
    pub at: String,
    /// `open`, `actioned` or `dismissed`.
    pub state: String,
    pub reporter: Option<Person>,
    pub target: ModTarget,
    /// Reports still open on the same target.
    pub open_on_target: u64,
    pub resolved_at: Option<String>,
}

impl ModReport {
    pub fn from_json(v: &Value) -> ModReport {
        ModReport {
            id: str_of(v, "id"),
            reason: str_of(v, "reason"),
            note: str_of(v, "note"),
            at: str_of(v, "at"),
            state: str_of(v, "state"),
            reporter: v.get("reporter").filter(|p| p.is_object()).map(Person::from_json),
            target: ModTarget::from_json(v.get("target").unwrap_or(&Value::Null)),
            open_on_target: v.get("open_on_target").and_then(Value::as_u64).unwrap_or(0),
            resolved_at: opt_str(v, "resolved_at"),
        }
    }

    pub fn open(&self) -> bool {
        self.state == "open"
    }
}

/// One entry of the moderation log.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ModAction {
    pub id: String,
    pub at: String,
    /// `hide-post`, `restore-post`, `suspend-page`, `restore-page`, `suspend-org`, `restore-org` or `dismiss`.
    pub action: String,
    /// The moderator, or None for an act taken on the identity host itself (`by_host`).
    pub moderator: Option<Person>,
    pub by_host: bool,
    /// `post`, `person`, `org` or `report`.
    pub target_kind: String,
    pub target_id: String,
    pub report_id: Option<String>,
    pub note: String,
}

impl ModAction {
    pub fn from_json(v: &Value) -> ModAction {
        ModAction {
            id: str_of(v, "id"),
            at: str_of(v, "at"),
            action: str_of(v, "action"),
            moderator: v.get("moderator").filter(|p| p.is_object()).map(Person::from_json),
            by_host: v.get("by_host").and_then(Value::as_bool).unwrap_or(false),
            target_kind: str_of(v, "target_kind"),
            target_id: str_of(v, "target_id"),
            report_id: opt_str(v, "report_id"),
            note: str_of(v, "note"),
        }
    }

    /// The act in a few words, as a log line says it.
    pub fn words(&self) -> &'static str {
        ModAct::from_word(&self.action).map(ModAct::past).unwrap_or(if self.action == "dismiss" { "dismissed a report" } else { "acted" })
    }
}

/// What a moderator can do to what a report names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModAct {
    HidePost,
    RestorePost,
    SuspendPage,
    RestorePage,
    SuspendOrg,
    RestoreOrg,
}

impl ModAct {
    pub const ALL: [ModAct; 6] =
        [ModAct::HidePost, ModAct::RestorePost, ModAct::SuspendPage, ModAct::RestorePage, ModAct::SuspendOrg, ModAct::RestoreOrg];

    pub fn as_str(self) -> &'static str {
        match self {
            ModAct::HidePost => "hide-post",
            ModAct::RestorePost => "restore-post",
            ModAct::SuspendPage => "suspend-page",
            ModAct::RestorePage => "restore-page",
            ModAct::SuspendOrg => "suspend-org",
            ModAct::RestoreOrg => "restore-org",
        }
    }

    pub fn from_word(word: &str) -> Option<ModAct> {
        ModAct::ALL.into_iter().find(|a| a.as_str() == word)
    }

    /// The button's words.
    pub fn label(self) -> &'static str {
        match self {
            ModAct::HidePost => "Hide post",
            ModAct::RestorePost => "Restore post",
            ModAct::SuspendPage => "Suspend page",
            ModAct::RestorePage => "Restore page",
            ModAct::SuspendOrg => "Suspend organization",
            ModAct::RestoreOrg => "Restore organization",
        }
    }

    /// What was done, for the log.
    pub fn past(self) -> &'static str {
        match self {
            ModAct::HidePost => "hid a post",
            ModAct::RestorePost => "restored a post",
            ModAct::SuspendPage => "suspended a page",
            ModAct::RestorePage => "restored a page",
            ModAct::SuspendOrg => "suspended an organization",
            ModAct::RestoreOrg => "restored an organization",
        }
    }

    /// The kind of target it acts on: `post`, `person` or `org`.
    pub fn kind(self) -> &'static str {
        match self {
            ModAct::HidePost | ModAct::RestorePost => "post",
            ModAct::SuspendPage | ModAct::RestorePage => "person",
            ModAct::SuspendOrg | ModAct::RestoreOrg => "org",
        }
    }

    /// The act that fits a target as it stands: hide or suspend it, or restore it when it already is.
    pub fn for_target(t: &ModTarget) -> Option<ModAct> {
        Some(match (t.kind.as_str(), t.hidden || t.suspended) {
            ("post", false) => ModAct::HidePost,
            ("post", true) if t.hidden => ModAct::RestorePost,
            ("post", true) => return None, // a post shown nowhere because its author's page is suspended
            ("person", false) => ModAct::SuspendPage,
            ("person", true) => ModAct::RestorePage,
            ("org", false) => ModAct::SuspendOrg,
            ("org", true) => ModAct::RestoreOrg,
            _ => return None,
        })
    }
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
    /// Who may reply.
    pub replies: ReplyRule,
    /// A poll, on a top-level post only.
    pub poll: Option<NewPoll>,
    /// A job opening, on an article posted as an organization only.
    pub job: Option<NewJob>,
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
        if self.replies != ReplyRule::Everyone {
            body["replies"] = json!(self.replies.as_str());
        }
        if let Some(poll) = &self.poll {
            body["poll"] = json!({"options": poll.options, "hours": poll.hours});
        }
        if let Some(job) = &self.job {
            body["job"] = json!({"location": job.location, "workplace": job.workplace.as_str(),
                                 "employment": job.employment.as_str(), "pay": job.pay, "days": job.days});
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

/// Reports, with what each names: the open ones oldest first, or the closed ones newest first (moderators only).
pub async fn mod_queue(client: &Client, token: &str, closed: bool, before: Option<&str>) -> Result<(Vec<ModReport>, bool), Problem> {
    let body = paged(json!({"state": if closed { "closed" } else { "open" }}), before);
    call(client, token, "mod-queue", body).await.map(|v| {
        (list_of(&v, "reports").iter().map(ModReport::from_json).collect(), v.get("more").and_then(Value::as_bool).unwrap_or(false))
    })
}

/// Act on a target, named directly (`target`), or through a report (`report_id`), or both (they must agree). Hiding
/// or suspending closes every open report on the target (moderators only).
pub async fn mod_act(client: &Client, token: &str, act: ModAct, target: Option<&Target>, report_id: Option<&str>, note: &str) -> Result<ModAction, Problem> {
    let mut body = json!({"action": act.as_str(), "note": note});
    match target {
        Some(Target::Post(id)) => body["post_id"] = json!(id),
        Some(Target::Person(username)) => body["username"] = json!(username),
        Some(Target::Org(ticker)) => body["ticker"] = json!(ticker),
        None => {}
    }
    if let Some(r) = report_id {
        body["report_id"] = json!(r);
    }
    call(client, token, "mod-act", body).await.map(|v| ModAction::from_json(v.get("action").unwrap_or(&Value::Null)))
}

/// Close an open report without acting on what it names (moderators only).
pub async fn mod_dismiss(client: &Client, token: &str, report_id: &str, note: &str) -> Result<ModAction, Problem> {
    call(client, token, "mod-dismiss", json!({"report_id": report_id, "note": note}))
        .await
        .map(|v| ModAction::from_json(v.get("action").unwrap_or(&Value::Null)))
}

/// The moderation log, newest first (moderators only).
pub async fn mod_history(client: &Client, token: &str, before: Option<&str>) -> Result<(Vec<ModAction>, bool), Problem> {
    call(client, token, "mod-history", paged(json!({}), before)).await.map(|v| {
        (list_of(&v, "actions").iter().map(ModAction::from_json).collect(), v.get("more").and_then(Value::as_bool).unwrap_or(false))
    })
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

/// The reader's notifications, newest first, a page after `before`.
pub async fn notifications(client: &Client, token: &str, before: Option<&str>) -> Result<Notifications, Problem> {
    call(client, token, "notifications", paged(json!({}), before)).await.map(|v| Notifications::from_json(&v))
}

/// Mark the reader's notifications read: every one, or those up to and including `up_to` in the list's order.
/// Returns how many stay unread.
pub async fn notifications_read(client: &Client, token: &str, up_to: Option<&str>) -> Result<u64, Problem> {
    let body = match up_to {
        Some(id) => json!({"up_to": id}),
        None => json!({}),
    };
    call(client, token, "notifications-read", body).await.map(|v| v.get("unread").and_then(Value::as_u64).unwrap_or(0))
}

/// Allow, or opt out of, Alelyon using the account's public posts to improve its services, AI models included.
pub async fn ai_training(client: &Client, token: &str, allow: bool) -> Result<Me, Problem> {
    call(client, token, "ai-training", json!({"allow": allow})).await.map(|v| Me::from_json(&v))
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

/// Save a post for oneself, or forget it. Only the reader sees their bookmarks.
pub async fn bookmark(client: &Client, token: &str, post_id: &str, bookmark: bool) -> Result<(), Problem> {
    call(client, token, "bookmark", json!({"post_id": post_id, "bookmark": bookmark})).await.map(|_| ())
}

/// The reader's bookmarks, the most recently saved first.
pub async fn bookmarks(client: &Client, token: &str, before: Option<&str>) -> Result<Posts, Problem> {
    call(client, token, "bookmarks", paged(json!({}), before)).await.map(|v| Posts::from_json(&v))
}

/// Pin one's own top-level post, quote or article to the top of one's page; it replaces any other.
pub async fn pin(client: &Client, token: &str, post_id: &str) -> Result<(), Problem> {
    call(client, token, "pin", json!({"post_id": post_id})).await.map(|_| ())
}

pub async fn unpin(client: &Client, token: &str) -> Result<(), Problem> {
    call(client, token, "unpin", json!({})).await.map(|_| ())
}

/// Search Pages: people, organizations and posts (`All`), or one of them. `before` pages on a `Posts` search only.
/// One `#tag` alone reads that hashtag.
pub async fn search(client: &Client, token: &str, query: &str, kind: SearchKind, before: Option<&str>) -> Result<Found, Problem> {
    call(client, token, "search", paged(json!({"q": query, "kind": kind.as_str()}), before)).await.map(|v| Found::from_json(&v))
}

/// The posts carrying a hashtag (with or without its `#`), newest first.
pub async fn tag(client: &Client, token: &str, tag: &str, before: Option<&str>) -> Result<Tagged, Problem> {
    call(client, token, "tag", paged(json!({"tag": tag}), before)).await.map(|v| Tagged::from_json(&v))
}

pub async fn trending(client: &Client, token: &str) -> Result<Trending, Problem> {
    call(client, token, "trending", json!({})).await.map(|v| Trending::from_json(&v))
}

/// Who to follow, and organizations to follow, from public signals only.
pub async fn suggestions(client: &Client, token: &str) -> Result<Suggestions, Problem> {
    call(client, token, "suggestions", json!({})).await.map(|v| Suggestions::from_json(&v))
}

/// One page (0 first, at most 10) of the ranked For you feed: others' recent top-level posts.
pub async fn for_you(client: &Client, token: &str, page: u32) -> Result<Posts, Problem> {
    call(client, token, "for-you", json!({"page": page})).await.map(|v| Posts::from_json(&v))
}

fn experience_list(v: &Value) -> Vec<Experience> {
    list_of(v, "experience").iter().map(Experience::from_json).collect()
}

fn education_list(v: &Value) -> Vec<Education> {
    list_of(v, "education").iter().map(Education::from_json).collect()
}

/// Add an experience entry, or replace one of the reader's own. A change clears an organization's confirmation.
pub async fn set_experience(client: &Client, token: &str, entry: &NewExperience) -> Result<Vec<Experience>, Problem> {
    call(client, token, "experience", entry.body()).await.map(|v| experience_list(&v))
}

pub async fn delete_experience(client: &Client, token: &str, entry_id: &str) -> Result<Vec<Experience>, Problem> {
    call(client, token, "experience-delete", json!({"entry_id": entry_id})).await.map(|v| experience_list(&v))
}

/// An owner or admin of the organization an entry names confirms it, or takes the confirmation back.
pub async fn confirm_experience(client: &Client, token: &str, ticker: &str, entry_id: &str, confirm: bool) -> Result<(), Problem> {
    call(client, token, "experience-confirm", json!({"ticker": ticker, "entry_id": entry_id, "confirm": confirm})).await.map(|_| ())
}

/// For an organization's owners and admins: the experience entries that name it, the unconfirmed first.
pub async fn org_experience(client: &Client, token: &str, ticker: &str) -> Result<Vec<(Person, Experience)>, Problem> {
    call(client, token, "org-experience", json!({"ticker": ticker})).await.map(|v| {
        list_of(&v, "entries")
            .iter()
            .map(|e| {
                (Person::from_json(e.get("person").unwrap_or(&Value::Null)), Experience::from_json(e.get("entry").unwrap_or(&Value::Null)))
            })
            .collect()
    })
}

pub async fn set_education(client: &Client, token: &str, entry: &NewEducation) -> Result<Vec<Education>, Problem> {
    call(client, token, "education", entry.body()).await.map(|v| education_list(&v))
}

pub async fn delete_education(client: &Client, token: &str, entry_id: &str) -> Result<Vec<Education>, Problem> {
    call(client, token, "education-delete", json!({"entry_id": entry_id})).await.map(|v| education_list(&v))
}

/// The reader's skills, in its order (at most 50). Endorsements of a skill it drops go.
pub async fn set_skills(client: &Client, token: &str, skills: &[String]) -> Result<Vec<Skill>, Problem> {
    call(client, token, "skills", json!({"skills": skills})).await.map(|v| list_of(&v, "skills").iter().map(Skill::from_json).collect())
}

/// Mute someone, silently, or unmute them: their posts leave the reader's feeds and their acts its notifications.
pub async fn mute(client: &Client, token: &str, username: &str, mute: bool) -> Result<(), Problem> {
    call(client, token, "mute", json!({"username": username, "mute": mute})).await.map(|_| ())
}

/// Whom the reader muted, the most recent first.
pub async fn mutes(client: &Client, token: &str) -> Result<Vec<Person>, Problem> {
    call(client, token, "mutes", json!({})).await.map(|v| list_of(&v, "people").iter().map(Person::from_json).collect())
}

/// Open jobs, newest first, as `query` narrows them.
pub async fn jobs(client: &Client, token: &str, query: &JobsQuery, before: Option<&str>) -> Result<Posts, Problem> {
    let mut body = json!({});
    if !query.words.trim().is_empty() {
        body["q"] = json!(query.words.trim());
    }
    if let Some(workplace) = query.workplace {
        body["workplace"] = json!(workplace.as_str());
    }
    if let Some(employment) = query.employment {
        body["employment"] = json!(employment.as_str());
    }
    if let Some(ticker) = &query.ticker {
        body["ticker"] = json!(ticker);
    }
    call(client, token, "jobs", paged(body, before)).await.map(|v| Posts::from_json(&v))
}

/// Apply for an open job with a note its organization's owners and admins read; again, to change the note.
pub async fn apply_job(client: &Client, token: &str, post_id: &str, note: &str) -> Result<(), Problem> {
    call(client, token, "job-apply", json!({"post_id": post_id, "note": note})).await.map(|_| ())
}

/// Withdraw an application, which deletes it.
pub async fn withdraw_application(client: &Client, token: &str, post_id: &str) -> Result<(), Problem> {
    call(client, token, "job-apply", json!({"post_id": post_id, "apply": false})).await.map(|_| ())
}

/// For an organization's owners and admins: who applied for its job, the earliest first, with their notes.
pub async fn job_applicants(client: &Client, token: &str, post_id: &str) -> Result<Vec<Applicant>, Problem> {
    call(client, token, "job-applicants", json!({"post_id": post_id}))
        .await
        .map(|v| list_of(&v, "applicants").iter().map(Applicant::from_json).collect())
}

/// For an organization's owners and admins: close its job now.
pub async fn close_job(client: &Client, token: &str, post_id: &str) -> Result<(), Problem> {
    call(client, token, "job-close", json!({"post_id": post_id})).await.map(|_| ())
}

/// The jobs the reader applied for, the latest application first.
pub async fn applications(client: &Client, token: &str) -> Result<Vec<Post>, Problem> {
    call(client, token, "applications", json!({})).await.map(|v| list_of(&v, "posts").iter().map(Post::from_json).collect())
}

/// Vote in a poll: one vote, never changed, while it is open; never in one's own.
pub async fn vote(client: &Client, token: &str, post_id: &str, option: usize) -> Result<(), Problem> {
    call(client, token, "vote", json!({"post_id": post_id, "option": option})).await.map(|_| ())
}

/// Endorse a skill another person lists, or take the endorsement back. Only the totals are shown.
pub async fn endorse(client: &Client, token: &str, username: &str, skill: &str, endorse: bool) -> Result<(), Problem> {
    call(client, token, "endorse", json!({"username": username, "skill": skill, "endorse": endorse})).await.map(|_| ())
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

/// The organization's posts whose author is gone, each with its `author_flag`, newest first: for its owners and
/// admins to keep or delete.
pub async fn org_flagged(client: &Client, token: &str, ticker: &str, before: Option<&str>) -> Result<Posts, Problem> {
    call(client, token, "org-flagged", paged(json!({"ticker": ticker}), before)).await.map(|v| Posts::from_json(&v))
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
    fn an_organizations_post_whose_author_is_gone_reads_without_them_and_with_the_flag() {
        let gone = Post::from_json(&json!({"id": "pst_4", "author": null, "author_flag": "suspended", "text": "Hello",
                                           "org": {"id": "org_1", "ticker": "LACE", "name": "Lace Labs"}}));
        assert!(gone.author.is_none());
        assert_eq!(gone.author_flag, Some(AuthorFlag::Suspended));
        assert_eq!(gone.org.as_ref().map(|o| o.ticker.as_str()), Some("LACE"));
        let unknown = Post::from_json(&json!({"id": "pst_5", "author": card("ada"), "author_flag": "later"}));
        assert_eq!((unknown.author.as_ref().map(|a| a.name()), unknown.author_flag), (Some("Ada Lovelace"), None));
        assert_eq!(AuthorFlag::from_word(Some("deleted")), Some(AuthorFlag::Deleted));
    }

    #[test]
    fn the_ai_training_choice_reads_as_allowed_unless_the_service_says_otherwise() {
        assert!(!Me::from_json(&json!({"me": card("ada"), "ai_training": false})).ai_training);
        assert!(Me::from_json(&json!({"me": card("ada"), "ai_training": true})).ai_training);
        assert!(Me::from_json(&json!({"me": card("ada")})).ai_training, "allowed until an account opts out");
    }

    #[test]
    fn notifications_read_with_what_they_concern_and_page_on() {
        let page = Notifications::from_json(&json!({"notifications": [
            {"id": "ntf_1", "kind": "like", "actor": card("ada"), "post": {"id": "pst_1", "text": "Mine"}, "org": null,
             "at": "t", "read": false},
            {"id": "ntf_2", "kind": "org-invite", "actor": card("bob"), "post": null,
             "org": {"id": "org_1", "ticker": "LACE", "name": "Lace Labs"}, "at": "t", "read": true}],
            "more": true, "unread": 4}));
        assert_eq!((page.items.len(), page.unread, page.next().as_deref()), (2, 4, Some("ntf_2")));
        let like = &page.items[0];
        assert_eq!((like.kind.as_str(), like.read), ("like", false));
        assert_eq!(like.post.as_ref().map(|p| p.id.as_str()), Some("pst_1"));
        let invite = &page.items[1];
        assert!(invite.post.is_none() && invite.read);
        assert_eq!(invite.org.as_ref().map(|o| o.ticker.as_str()), Some("LACE"));
        assert_eq!(Me::from_json(&json!({"me": card("ada"), "unread_notifications": 3})).unread_notifications, 3);
    }

    #[test]
    fn a_post_says_whether_the_reader_saved_it_and_a_page_carries_its_pin() {
        assert!(Post::from_json(&json!({"id": "pst_1", "bookmarked": true})).bookmarked);
        assert!(!Post::from_json(&json!({"id": "pst_1", "bookmarked": null})).bookmarked);
        let page = PersonPage::from_json(&json!({"person": card("ada"), "posts": [], "more": false,
                                                 "pinned": {"id": "pst_9", "text": "Read me first"}}));
        assert_eq!(page.pinned.as_ref().map(|p| p.id.as_str()), Some("pst_9"));
        assert!(PersonPage::from_json(&json!({"person": card("ada"), "pinned": null})).pinned.is_none());
    }

    #[test]
    fn a_job_reads_from_a_post_and_a_new_one_sends_it() {
        let post = Post::from_json(&json!({"id": "pst_1", "kind": "article", "job": {"location": "London",
            "workplace": "hybrid", "employment": "part-time", "pay": "", "closes_at": "2026-11-13T00:00:00Z",
            "closed": false, "applied": true, "applicants": null}}));
        let job = post.job.expect("a job");
        assert_eq!((job.workplace, job.employment, job.applied, job.applicants), (Workplace::Hybrid, Employment::PartTime, true, None));
        assert_eq!(Post::from_json(&json!({"id": "pst_2"})).job, None);
        let new = NewPost {
            kind: Kind::Article,
            text: "Build".into(),
            title: Some("Engineer".into()),
            as_org: Some("BABG".into()),
            job: Some(NewJob { workplace: Workplace::Remote, employment: Employment::Contract, days: 30, ..NewJob::default() }),
            ..NewPost::default()
        };
        assert_eq!(new.body()["job"], json!({"location": "", "workplace": "remote", "employment": "contract", "pay": "", "days": 30}));
        assert_eq!(Workplace::ALL.map(|w| w.to_string()), ["On-site", "Hybrid", "Remote"]);
        assert_eq!(Employment::ALL.map(Employment::as_str).map(Employment::from_word), Employment::ALL);
    }

    #[test]
    fn a_post_reads_its_poll_and_who_may_reply_and_a_new_one_sends_them() {
        let post = Post::from_json(&json!({"id": "pst_1", "replies_from": "followers", "you_can_reply": false,
            "poll": {"options": ["A", "B", "C"], "votes": [1, 2, 0], "total": 3, "closes_at": "2026-10-15T00:00:00Z",
                     "closed": false, "you_voted": 1}}));
        assert_eq!((post.replies_from, post.you_can_reply), (ReplyRule::Followers, false));
        let poll = post.poll.expect("a poll");
        assert_eq!((poll.you_voted, poll.shares()), (Some(1), Some(vec![33, 67, 0])));
        let hidden = Poll::from_json(&json!({"options": ["A", "B"], "votes": null, "total": 4}));
        assert_eq!((hidden.shares(), hidden.total), (None, 4));
        let plain = Post::from_json(&json!({"id": "pst_2"}));
        assert_eq!((plain.replies_from, plain.you_can_reply, plain.poll), (ReplyRule::Everyone, true, None));
        let new = NewPost {
            text: "Which?".into(),
            replies: ReplyRule::Mentioned,
            poll: Some(NewPoll { options: vec!["A".into(), "B".into()], hours: 24 }),
            ..NewPost::default()
        };
        assert_eq!(new.body(), json!({"kind": "post", "text": "Which?", "replies": "mentioned",
                                     "poll": {"options": ["A", "B"], "hours": 24}}));
        assert!(PersonPage::from_json(&json!({"person": card("ada"), "you_mute": true})).you_mute);
    }

    #[test]
    fn a_page_carries_its_profile_and_me_mine() {
        let page = PersonPage::from_json(&json!({"person": card("ada"), "profile": {
            "experience": [{"id": "exp_1", "title": "Founder", "org": {"id": "org_1", "ticker": "LACE", "name": "Lace Labs"},
                            "org_name": "Lace Labs", "started": "2024-01", "ended": null, "about": "", "confirmed": true}],
            "education": [{"id": "edu_1", "school": "University of London", "degree": "BSc", "field": "Mathematics",
                           "started": "2015", "ended": "2019"}],
            "skills": [{"skill": "Mathematics", "endorsements": 2, "you_endorsed": true},
                       {"skill": "Engines", "endorsements": 0, "you_endorsed": null}]}}));
        let profile = page.profile.expect("the first page carries it");
        assert_eq!((profile.experience[0].span(), profile.experience[0].confirmed), ("2024-01 to now".to_string(), true));
        assert_eq!(profile.education[0].ended.as_deref(), Some("2019"));
        let skills: Vec<_> = profile.skills.iter().map(|s| (s.endorsements, s.you_endorsed)).collect();
        assert_eq!(skills, [(2, Some(true)), (0, None)]);
        assert!(PersonPage::from_json(&json!({"person": card("ada"), "profile": null})).profile.is_none());
        assert_eq!(Me::from_json(&json!({"me": card("ada")})).profile, Profile::default());
        let new = NewExperience { title: "Analyst".into(), org_name: "Babbage & Co".into(), started: "2019-03".into(), ..NewExperience::default() };
        assert_eq!(new.body(), json!({"title": "Analyst", "org_name": "Babbage & Co", "started": "2019-03", "about": ""}));
        let school = NewEducation { id: "edu_1".into(), school: "UCL".into(), ended: "2019".into(), ..NewEducation::default() };
        assert_eq!(school.body(), json!({"entry_id": "edu_1", "school": "UCL", "degree": "", "field": "", "ended": "2019"}));
    }

    #[test]
    fn a_search_reads_people_organizations_and_posts() {
        let found = Found::from_json(&json!({"people": [card("ada")],
                                             "orgs": [{"id": "org_1", "ticker": "LACE", "name": "Lace Labs"}],
                                             "posts": [{"id": "pst_1"}], "more": true}));
        assert_eq!(found.people[0].username.as_deref(), Some("ada"));
        assert_eq!(found.orgs[0].ticker, "LACE");
        assert_eq!(found.posts.next().as_deref(), Some("pst_1"));
        let people_only = Found::from_json(&json!({"people": []}));
        assert!(people_only.orgs.is_empty() && people_only.posts.posts.is_empty());
        assert_eq!(SearchKind::ALL.map(SearchKind::as_str), ["all", "posts", "people", "orgs"]);
        let tagged = Tagged::from_json(&json!({"tag": "looms", "posts": [{"id": "pst_1"}], "more": false}));
        assert_eq!((tagged.tag.as_str(), tagged.posts.posts.len()), ("looms", 1));
    }

    #[test]
    fn trending_tags_and_suggestions_read_with_their_reasons() {
        let t = Trending::from_json(&json!({"hours": 168, "tags": [{"tag": "looms", "posts": 3, "people": 2}]}));
        assert_eq!((t.hours, t.tags[0].tag.as_str(), t.tags[0].posts, t.tags[0].people), (168, "looms", 3, 2));
        let s = Suggestions::from_json(&json!({"people": [
            {"person": card("ada"), "reason": "org", "org": {"id": "org_1", "ticker": "LACE", "name": "Lace Labs"}},
            {"person": card("bob"), "reason": "replied", "org": null},
            {"person": card("cy"), "reason": "something-new", "org": null}],
            "orgs": [{"id": "org_2", "ticker": "MILL", "name": "Mills"}]}));
        assert_eq!(s.people[0].why.words(s.people[0].org.as_ref()), "Shows (LACE)");
        assert_eq!(s.people[1].why, Why::Replied);
        assert_eq!(s.people[2].why, Why::Page, "an unknown reason reads as the plainest one");
        assert_eq!(s.orgs[0].ticker, "MILL");
        assert_eq!(Suggestions::from_json(&Value::Null), Suggestions::default());
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
    fn a_moderators_queue_and_log_read_with_their_targets() {
        let me = Me::from_json(&json!({"me": card("mona"), "memberships": [], "moderator": true, "suspended": false}));
        assert!(me.moderator && !me.suspended);
        assert!(!Me::from_json(&json!({"me": card("ada")})).moderator, "absent reads as not a moderator");
        let report = ModReport::from_json(&json!({
            "id": "rpt_1", "reason": "spam", "note": "Sells things.", "at": "t", "state": "open",
            "reporter": card("bob"), "open_on_target": 2, "resolved_at": null, "resolved_by": null,
            "target": {"kind": "post", "id": "pst_1", "post": {"id": "pst_1", "text": "Buy now"},
                       "person": card("ada"), "org": null, "text": "", "hidden": false, "suspended": false}
        }));
        assert!(report.open());
        assert_eq!((report.target.kind.as_str(), report.open_on_target), ("post", 2));
        assert_eq!(report.target.post.as_ref().map(|p| p.text.as_str()), Some("Buy now"));
        assert_eq!(ModAct::for_target(&report.target), Some(ModAct::HidePost));
        let hidden = ModTarget { kind: "post".into(), hidden: true, ..ModTarget::default() };
        assert_eq!(ModAct::for_target(&hidden), Some(ModAct::RestorePost));
        let behind_suspension = ModTarget { kind: "post".into(), suspended: true, ..ModTarget::default() };
        assert_eq!(ModAct::for_target(&behind_suspension), None, "restored through the page, not the post");
        let page = ModTarget { kind: "person".into(), suspended: true, ..ModTarget::default() };
        assert_eq!(ModAct::for_target(&page), Some(ModAct::RestorePage));
        assert_eq!(ModAct::for_target(&ModTarget { kind: "gone".into(), ..ModTarget::default() }), None);
        let log = ModAction::from_json(&json!({"id": "mod_1", "at": "t", "action": "suspend-page", "moderator": card("mona"),
                                               "by_host": false, "target_kind": "person", "target_id": "acct_ada",
                                               "report_id": "rpt_1", "note": "Impersonation."}));
        assert_eq!((log.words(), log.report_id.as_deref()), ("suspended a page", Some("rpt_1")));
        let host = ModAction::from_json(&json!({"action": "dismiss", "moderator": null, "by_host": true}));
        assert_eq!((host.words(), host.by_host, host.moderator), ("dismissed a report", true, None));
        for act in ModAct::ALL {
            assert_eq!(ModAct::from_word(act.as_str()), Some(act));
        }
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
