//! The actions an agent may take on Alelyon Pages, one per route: its name as an MCP tool, the route it calls, the
//! scope its token needs, what it does in a sentence, and the fields it takes as a JSON Schema.
//!
//! The service decides what a token may do; this table only describes it. A tool whose token lacks the scope is
//! refused by the service with `scope-missing`, which names the scope to add in the developer console.

use serde_json::{Value, json};

/// One action.
#[derive(Clone, Debug)]
pub struct Tool {
    pub name: &'static str,
    pub route: &'static str,
    pub scope: &'static str,
    pub description: &'static str,
    /// Changes nothing.
    pub read_only: bool,
    /// Removes or undoes something the person made.
    pub destructive: bool,
    /// The fields, as `(name, JSON Schema, required)`.
    pub fields: fn() -> Vec<(&'static str, Value, bool)>,
}

impl Tool {
    /// The tool's input as a JSON Schema object.
    pub fn input_schema(&self) -> Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();
        for (name, schema, needed) in (self.fields)() {
            properties.insert(name.to_string(), schema);
            if needed {
                required.push(Value::from(name));
            }
        }
        json!({"type": "object", "properties": properties, "required": required, "additionalProperties": false})
    }
}

fn text(description: &str, max: u32) -> Value {
    json!({"type": "string", "description": description, "maxLength": max})
}

fn flag(description: &str) -> Value {
    json!({"type": "boolean", "description": description})
}

fn id(description: &str) -> Value {
    text(description, 64)
}

fn one_of(description: &str, values: &[&str]) -> Value {
    json!({"type": "string", "description": description, "enum": values})
}

fn before() -> (&'static str, Value, bool) {
    ("before", id("For older items: the id of the last one you saw, when the answer said more: true."), false)
}

fn none() -> Vec<(&'static str, Value, bool)> {
    Vec::new()
}

/// Every action, in the order an agent should meet them.
pub const TOOLS: &[Tool] = &[
    Tool {
        name: "pages_me",
        route: "me",
        scope: "pages:read",
        description: "Your own page: your name, username, title, bio and counts.",
        read_only: true,
        destructive: false,
        fields: none,
    },
    Tool {
        name: "pages_feed",
        route: "feed",
        scope: "pages:read",
        description: "Your home feed: posts by you and the people and organizations you follow, newest first.",
        read_only: true,
        destructive: false,
        fields: || vec![before()],
    },
    Tool {
        name: "pages_for_you",
        route: "for-you",
        scope: "pages:read",
        description: "Posts chosen for you from across Pages.",
        read_only: true,
        destructive: false,
        fields: || vec![("page", json!({"type": "integer", "minimum": 0, "description": "Which page, from 0."}), false)],
    },
    Tool {
        name: "pages_explore",
        route: "explore",
        scope: "pages:read",
        description: "Recent public posts from everyone, newest first.",
        read_only: true,
        destructive: false,
        fields: || vec![before()],
    },
    Tool {
        name: "pages_thread",
        route: "thread",
        scope: "pages:read",
        description: "One post with the post it answers and its replies.",
        read_only: true,
        destructive: false,
        fields: || vec![("post_id", id("The post's id."), true), before()],
    },
    Tool {
        name: "pages_person",
        route: "person",
        scope: "pages:read",
        description: "A person's page and posts, by username.",
        read_only: true,
        destructive: false,
        fields: || {
            vec![
                ("username", text("Their username, without @.", 40), true),
                ("show", one_of("Which of their posts (posts by default).", &["posts", "replies", "media", "articles"]), false),
                before(),
            ]
        },
    },
    Tool {
        name: "pages_search",
        route: "search",
        scope: "pages:read",
        description: "Search Pages for posts, people, organizations or products.",
        read_only: true,
        destructive: false,
        fields: || {
            vec![
                ("q", text("What to look for.", 200), true),
                ("kind", one_of("What to find (all by default).", &["all", "posts", "people", "orgs", "products"]), false),
                before(),
            ]
        },
    },
    Tool {
        name: "pages_tag",
        route: "tag",
        scope: "pages:read",
        description: "Posts with a hashtag.",
        read_only: true,
        destructive: false,
        fields: || vec![("tag", text("The hashtag, without #.", 100), true), before()],
    },
    Tool {
        name: "pages_trending",
        route: "trending",
        scope: "pages:read",
        description: "The hashtags most used lately.",
        read_only: true,
        destructive: false,
        fields: none,
    },
    Tool {
        name: "pages_notifications",
        route: "notifications",
        scope: "pages:read",
        description: "Your notifications: replies, mentions, likes, follows and more, newest first.",
        read_only: true,
        destructive: false,
        fields: || vec![before()],
    },
    Tool {
        name: "pages_bookmarks",
        route: "bookmarks",
        scope: "pages:read",
        description: "The posts you saved.",
        read_only: true,
        destructive: false,
        fields: || vec![before()],
    },
    Tool {
        name: "pages_jobs",
        route: "jobs",
        scope: "pages:read",
        description: "Open jobs posted on Pages.",
        read_only: true,
        destructive: false,
        fields: || {
            vec![
                ("q", text("Words to look for.", 200), false),
                ("ticker", text("Only this organization's jobs, by its ticker.", 10), false),
                ("workplace", one_of("Where the work is.", &["on-site", "hybrid", "remote"]), false),
                ("employment", one_of("The kind of job.", &["full-time", "part-time", "contract", "temporary", "internship"]), false),
                before(),
            ]
        },
    },
    Tool {
        name: "pages_post",
        route: "post",
        scope: "pages:post",
        description: "Post on your page, reply to a post (reply_to) or quote one (repost_of). The post is labeled with \
                      this token's name, so readers know your agent wrote it.",
        read_only: false,
        destructive: false,
        fields: || {
            vec![
                ("text", text("What to post.", 20000), true),
                ("reply_to", id("The post this replies to."), false),
                ("repost_of", id("The post this quotes; with empty text, a plain repost."), false),
                ("kind", one_of("post (the default) or article.", &["post", "article"]), false),
                ("title", text("An article's title.", 1000), false),
            ]
        },
    },
    Tool {
        name: "pages_edit",
        route: "edit",
        scope: "pages:post",
        description: "Change the text of one of your posts.",
        read_only: false,
        destructive: false,
        fields: || {
            vec![
                ("post_id", id("Your post's id."), true),
                ("text", text("Its new text.", 20000), true),
                ("title", text("An article's new title.", 1000), false),
            ]
        },
    },
    Tool {
        name: "pages_delete",
        route: "delete",
        scope: "pages:post",
        description: "Delete one of your posts. It cannot be undone.",
        read_only: false,
        destructive: true,
        fields: || vec![("post_id", id("Your post's id."), true)],
    },
    Tool {
        name: "pages_like",
        route: "like",
        scope: "pages:interact",
        description: "Like a post, or take the like back (like: false).",
        read_only: false,
        destructive: false,
        fields: || vec![("post_id", id("The post's id."), true), ("like", flag("false to take it back."), false)],
    },
    Tool {
        name: "pages_follow",
        route: "follow",
        scope: "pages:interact",
        description: "Follow a person (username) or an organization (ticker), or stop (follow: false).",
        read_only: false,
        destructive: false,
        fields: || {
            vec![
                ("username", text("The person's username.", 40), false),
                ("ticker", text("The organization's ticker.", 10), false),
                ("follow", flag("false to stop following."), false),
            ]
        },
    },
    Tool {
        name: "pages_bookmark",
        route: "bookmark",
        scope: "pages:interact",
        description: "Save a post for later, or remove it (bookmark: false).",
        read_only: false,
        destructive: false,
        fields: || vec![("post_id", id("The post's id."), true), ("bookmark", flag("false to remove it."), false)],
    },
    Tool {
        name: "pages_notifications_read",
        route: "notifications-read",
        scope: "pages:interact",
        description: "Mark your notifications read, up to one of them.",
        read_only: false,
        destructive: false,
        fields: || vec![("up_to", id("The newest notification to mark read."), true)],
    },
    Tool {
        name: "pages_update_profile",
        route: "update",
        scope: "pages:profile",
        description: "Change your page's title and bio.",
        read_only: false,
        destructive: false,
        fields: || {
            vec![
                ("title", text("Your title, such as Founder | CEO.", 200), false),
                ("bio", text("A few sentences about you.", 2000), false),
            ]
        },
    },
];

/// The tool named `name`, of either table.
/// The Pages tools as data, for a server that offers the same tools elsewhere: each one's name, route, scope,
/// description, hints and input schema, in this table's order. The app's tools stay out: they reach a computer.
pub fn export() -> Value {
    let tools: Vec<Value> = TOOLS
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "route": t.route,
                "scope": t.scope,
                "description": t.description,
                "read_only": t.read_only,
                "destructive": t.destructive,
                "input_schema": t.input_schema(),
            })
        })
        .collect();
    json!({"schema": "alelyon.mcp-tools/1", "tools": tools})
}

pub fn find(name: &str) -> Option<&'static Tool> {
    TOOLS.iter().chain(APP_TOOLS).find(|t| t.name == name)
}

/// What an app tool needs instead of a scope: the app open, its agent bridge on (`crate::app`).
pub const APP: &str = "app";

/// The Alelyon app on this computer, through its agent bridge (Settings, Agents). Reads answer at once; the app asks
/// the person before its first action of a session and shows every action in its window.
pub const APP_TOOLS: &[Tool] = &[
    Tool { name: "app_status", route: "status", scope: APP, read_only: true, destructive: false, fields: none,
           description: "The Alelyon app's state: its version, the section on show, its sections, and whether the person has allowed actions." },
    Tool { name: "app_chats", route: "chats", scope: APP, read_only: true, destructive: false, fields: none,
           description: "The chats in the app's Lattice, newest first, with each one's id and whether it is answering." },
    Tool { name: "app_chat_read", route: "chat_read", scope: APP, read_only: true, destructive: false,
           fields: || vec![("last", json!({"type": "integer", "minimum": 1, "maximum": 50, "description": "How many of its last turns (10 by default)."}), false)],
           description: "The chat open in the app: its last turns and, while it answers, the text so far." },
    Tool { name: "app_runs", route: "runs", scope: APP, read_only: true, destructive: false, fields: none,
           description: "The app's runs (agents working on a task), with each one's status." },
    Tool { name: "app_go", route: "go", scope: APP, read_only: false, destructive: false,
           fields: || vec![("section", text("A section's name, from app_status.", 20), true)],
           description: "Show one of the app's sections (lattice, pages, research and the rest)." },
    Tool { name: "app_chat_new", route: "chat_new", scope: APP, read_only: false, destructive: false, fields: none,
           description: "Start a new chat in the app's Lattice." },
    Tool { name: "app_chat_open", route: "chat_open", scope: APP, read_only: false, destructive: false,
           fields: || vec![("id", id("The chat's id, from app_chats."), true)],
           description: "Open one of the app's chats." },
    Tool { name: "app_chat_send", route: "chat_send", scope: APP, read_only: false, destructive: false,
           fields: || vec![("text", text("The message.", 20000), true)],
           description: "Send a message in the app's open chat, as if the person typed it. Read the answer with app_chat_read." },
    Tool { name: "app_open_url", route: "open_url", scope: APP, read_only: false, destructive: false,
           fields: || vec![("url", text("An https:// address.", 4096), true)],
           description: "Open a web page in the app's Lattice browser." },
    Tool { name: "app_run_start", route: "run_start", scope: APP, read_only: false, destructive: false,
           fields: || vec![("task", text("What the run should do.", 4000), true)],
           description: "Start a run in the app: an agent working on a task with the person's chosen agent and model." },
    Tool { name: "fleet_groups", route: "fleet_groups", scope: APP, read_only: true, destructive: false, fields: none,
           description: "Every group of work in the app's Lattice, in every project, highest priority first: its column (planned, in_progress, waiting, done), what it waits for from the person, its tasks running, its steps and milestones done, and what it spent." },
    Tool { name: "fleet_group", route: "fleet_group", scope: APP, read_only: true, destructive: false,
           fields: || vec![("id", id("The group's id, from fleet_groups."), true)],
           description: "One group's topics (the sessions doing its work): each one's step under way, its steps, what it did last, whether it is working, and how many questions or approvals it waits for." },
    Tool { name: "fleet_record", route: "fleet_record", scope: APP, read_only: true, destructive: false,
           fields: || vec![("id", id("The group's id, from fleet_groups."), true)],
           description: "A group's record, which every session of the group starts with: the person's notes, where each session stands, what was decided (the person's answers and agents' recorded decisions), what is still open, and where the work is (artifacts, commits, pull requests). Read it before working on the group's work." },
    Tool { name: "fleet_watch", route: "fleet_watch", scope: APP, read_only: true, destructive: false,
           fields: || vec![
               ("since", text("The digest of the last fleet you saw (from this tool's last answer); without it, the fleet now.", 32), false),
               ("wait_s", json!({"type": "integer", "minimum": 1, "maximum": 25, "description": "The longest to wait for a change, in seconds (25 by default)."}), false),
           ],
           description: "Wait for the fleet to change, then see it: every group's card and everything waiting for the person, with a digest. Answers as soon as anything changed since the digest you pass (a new question, a session done or waiting, a step done), or with changed false when the wait is over. Call it again with the new digest to keep watching." },
    Tool { name: "fleet_search", route: "fleet_search", scope: APP, read_only: true, destructive: false,
           fields: || vec![("query", text("Words said in the session you are looking for.", 300), true)],
           description: "Find the person's past Lattice sessions by what was said in them: the best match of each session, best first, with its title, when, and a short excerpt. Pick one up with fleet_reply (its topic id)." },
    Tool { name: "fleet_waiting", route: "fleet_waiting", scope: APP, read_only: true, destructive: false, fields: none,
           description: "Everything waiting for the person across all groups: each question with its options, and each approval, named by its group and topic, the most recently active first." },
    Tool { name: "fleet_answer", route: "fleet_answer", scope: APP, read_only: false, destructive: false,
           fields: || vec![
               ("topic", id("The topic the question waits in, from fleet_waiting."), true),
               ("call", id("The question's call, from fleet_waiting."), true),
               ("text", text("The answer: one of its options, or your own words.", 8000), true),
           ],
           description: "Answer a question waiting in a topic, as the person would in the app. Approvals are not answered here: the person decides them in the app." },
    Tool { name: "fleet_new_group", route: "fleet_new_group", scope: APP, read_only: false, destructive: false,
           fields: || vec![
               ("name", text("The group's name: the piece of work, in a few words.", 80), true),
               ("text", text("Its first message: what the work is, what done looks like, and anything to know.", 20000), true),
           ],
           description: "Hand a new piece of work to the person's Lattice: a group of its own in the project they work in, with your message as the first in its chat. Its sessions start from there; follow it with fleet_groups and fleet_group." },
    Tool { name: "fleet_decide", route: "fleet_decide", scope: APP, read_only: false, destructive: false,
           fields: || vec![
               ("id", id("The group's id, from fleet_groups."), true),
               ("decision", text("What was decided, in one line.", 300), true),
           ],
           description: "Record a decision on a group's record, which every session of the group starts with: something settled outside Lattice (with the person, or in your own work) that no session should ask or decide again. Shown as made by your agent; the person can take it off." },
    Tool { name: "fleet_reply", route: "fleet_reply", scope: APP, read_only: false, destructive: false,
           fields: || vec![
               ("topic", id("The topic (session), from fleet_group."), true),
               ("text", text("What it should know or do next.", 20000), true),
           ],
           description: "Guide one session of a group: while it works, your text steers its running turn (it reads it before its next step); while it is idle, it is its next message. Use fleet_send to write in the group's chat instead." },
    Tool { name: "fleet_stop", route: "fleet_stop", scope: APP, read_only: false, destructive: true,
           fields: || vec![("topic", id("The topic (session), from fleet_group."), true)],
           description: "Stop a session's running turn (what it already did stays). Answers whether a turn was running." },
    Tool { name: "fleet_send", route: "fleet_send", scope: APP, read_only: false, destructive: false,
           fields: || vec![
               ("id", id("The group's id, from fleet_groups."), true),
               ("text", text("The message.", 20000), true),
           ],
           description: "Write in a group's chat, as if the person typed it there: it goes to the group's topic for that work, or starts one. Follow it with fleet_group." },
];

/// The fields a tool was given, checked against its own: an object whose names are its fields and whose required
/// fields are present. The service checks the values.
pub fn check_arguments(tool: &Tool, arguments: &Value) -> Result<(), String> {
    let given = match arguments {
        Value::Null => return check_arguments(tool, &json!({})),
        Value::Object(map) => map,
        _ => return Err("arguments must be a JSON object".into()),
    };
    let fields = (tool.fields)();
    for name in given.keys() {
        if !fields.iter().any(|(f, _, _)| f == name) {
            return Err(format!("{} takes no field named {name}", tool.name));
        }
    }
    for (name, _, needed) in &fields {
        if *needed && !given.contains_key(*name) {
            return Err(format!("{} needs {name}", tool.name));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn names_and_routes_are_unique_and_well_formed() {
        let names: HashSet<_> = TOOLS.iter().map(|t| t.name).collect();
        let routes: HashSet<_> = TOOLS.iter().map(|t| t.route).collect();
        assert_eq!(names.len(), TOOLS.len());
        assert_eq!(routes.len(), TOOLS.len());
        for t in TOOLS {
            assert!(t.name.starts_with("pages_") && t.name.len() <= 64, "{}", t.name);
            assert!(crate::api::route_ok(t.route), "{}", t.route);
            assert!(
                ["pages:read", "pages:post", "pages:interact", "pages:profile", "pages:manage"].contains(&t.scope),
                "{}",
                t.scope
            );
            assert_eq!(t.read_only, t.scope == "pages:read", "{}: reads are exactly the read scope", t.name);
            assert!(!(t.read_only && t.destructive));
            assert_eq!(t.input_schema()["type"], "object");
        }
    }

    #[test]
    fn app_tools_are_named_for_the_app_and_reads_are_marked() {
        for t in APP_TOOLS {
            assert!((t.name.starts_with("app_") || t.name.starts_with("fleet_")) && t.scope == APP, "{}", t.name);
            assert!(t.route.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'), "{}", t.route);
            assert_eq!(t.destructive, t.name == "fleet_stop", "{}: only stopping a turn undoes something", t.name);
        }
        let reads: Vec<_> = APP_TOOLS.iter().filter(|t| t.read_only).map(|t| t.route).collect();
        assert_eq!(reads, ["status", "chats", "chat_read", "runs", "fleet_groups", "fleet_group", "fleet_record", "fleet_watch", "fleet_search", "fleet_waiting"]);
        assert!(find("app_chat_send").is_some() && find("pages_post").is_some());
    }

    #[test]
    fn the_export_carries_every_pages_tool_and_no_app_tool() {
        let exported = export();
        let names: Vec<&str> = exported["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, TOOLS.iter().map(|t| t.name).collect::<Vec<_>>());
        assert!(exported["tools"][0]["input_schema"]["type"] == "object");
    }

    #[test]
    fn arguments_are_checked_against_the_tools_fields() {
        let post = find("pages_post").unwrap();
        assert!(check_arguments(post, &json!({"text": "hi"})).is_ok());
        assert!(check_arguments(post, &json!({})).is_err());
        assert!(check_arguments(post, &json!({"text": "hi", "client_id": "x"})).is_err());
        assert!(check_arguments(post, &json!(["hi"])).is_err());
        assert!(check_arguments(find("pages_me").unwrap(), &Value::Null).is_ok());
    }
}
