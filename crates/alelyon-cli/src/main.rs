//! `alelyon-cli`: Alelyon Pages from a terminal, and an MCP server for a person's own AI agent.
//!
//! Every call uses a developer token the person made in the developer console (`alyd_...`), with the scopes they
//! chose. The token comes from `ALELYON_TOKEN`, or from the one `alelyon-cli auth login` kept, sealed for this
//! Windows user. It is never printed and never written in the clear.

#![deny(unsafe_code)]

mod api;
mod app;
mod mcp;
mod tools;

use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use serde_json::{Value, json};

use api::{Pages, Service};

/// The kept token's seal: this program's own, so another program's sealed secrets do not open as a token.
const ENTROPY: &[u8] = b"alelyon-cli developer token v1";
const TOKEN_ENV: &str = "ALELYON_TOKEN";
const URL_ENV: &str = "ALELYON_URL";

#[derive(Parser)]
#[command(name = "alelyon-cli", version, about = "Alelyon Pages from a terminal, and an MCP server for your AI agent.")]
struct Cli {
    /// The identity service's address (default: Alelyon's, or ALELYON_URL).
    #[arg(long, global = true)]
    url: Option<String>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Keep, check or forget your developer token.
    Auth {
        #[command(subcommand)]
        action: Auth,
    },
    /// Serve the Pages tools to an AI agent over MCP (stdio).
    Mcp,
    /// List the Pages tools an agent gets, with the scope each needs.
    Tools,
    /// Your own page.
    Me,
    /// Your home feed.
    Feed {
        /// Older posts: the id of the last one you saw.
        #[arg(long)]
        before: Option<String>,
    },
    /// Post on your page (labeled with your token's name).
    Post {
        /// What to post.
        text: String,
        /// Reply to this post.
        #[arg(long)]
        reply_to: Option<String>,
    },
    /// A post with its replies.
    Thread { post_id: String },
    /// A person's page and posts.
    Person { username: String },
    /// Search posts, people, organizations and products.
    Search { query: String },
    /// Your notifications.
    Notifications,
    /// Like a post (or --undo).
    Like {
        post_id: String,
        #[arg(long)]
        undo: bool,
    },
    /// Follow a person (or --undo).
    Follow {
        username: String,
        #[arg(long)]
        undo: bool,
    },
    /// Drive the Alelyon app on this computer (Settings, Agents on): status, chats, chat_read, runs, go, chat_new,
    /// chat_open, chat_send, open_url or run_start, with the command's fields as JSON.
    App {
        command: String,
        fields: Option<String>,
    },
    /// Run any tool by name with its fields as JSON, as an agent would.
    Call {
        /// A tool's name (see `alelyon-cli tools`).
        tool: String,
        /// Its fields, one JSON object (default {}).
        fields: Option<String>,
    },
}

#[derive(Subcommand)]
enum Auth {
    /// Keep a developer token for this Windows user (read from standard input, never from the command line).
    Login,
    /// Check the token works, and whose it is.
    Status,
    /// Forget the kept token.
    Logout,
}

fn kept_path() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA").or_else(|| std::env::var_os("XDG_CONFIG_HOME"))?;
    Some(PathBuf::from(base).join("Alelyon").join("cli").join("token.sealed"))
}

fn token() -> Result<String, String> {
    if let Ok(t) = std::env::var(TOKEN_ENV) {
        return Ok(t.trim().to_string());
    }
    kept_path()
        .and_then(|p| alelyon_identity_client::vault::open_with(&p, ENTROPY))
        .ok_or_else(|| {
            format!(
                "No developer token. Make one in the developer console at https://www.alelyon.com/developers/, then run \
                 `alelyon-cli auth login` or set {TOKEN_ENV}."
            )
        })
}

fn service(cli_url: &Option<String>) -> Result<Service, String> {
    let url = cli_url
        .clone()
        .or_else(|| std::env::var(URL_ENV).ok())
        .unwrap_or_else(|| alelyon_identity_client::client::DEFAULT_URL.to_string());
    Service::new(&url, &token()?).map_err(|f| f.to_string())
}

fn run_tool(pages: &dyn Pages, name: &str, fields: Value) -> Result<Value, String> {
    let tool = tools::find(name).ok_or_else(|| format!("There is no tool named {name}. See `alelyon-cli tools`."))?;
    tools::check_arguments(tool, &fields)?;
    pages.call(tool.route, &fields).map_err(|f| {
        let mut words = f.to_string();
        if f.code == "scope-missing" {
            words.push_str(" Add that scope to the token in the developer console.");
        }
        words
    })
}

fn print(value: &Value) {
    println!("{}", serde_json::to_string_pretty(value).unwrap_or_default());
}

fn login() -> Result<(), String> {
    let path = kept_path().ok_or("There is nowhere to keep a token on this computer; set ALELYON_TOKEN instead.")?;
    if std::io::stdin().is_terminal() {
        eprint!("Paste your developer token (alyd_...) and press Enter: ");
        std::io::stderr().flush().ok();
    }
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).map_err(|_| "Could not read the token.")?;
    let token = line.trim();
    if !api::looks_like_token(token) {
        return Err("That is not a developer token (alyd_ and 43 characters).".into());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|_| "Could not make the folder for the token.")?;
    }
    alelyon_identity_client::vault::keep_with(&path, token, ENTROPY)?;
    eprintln!("Kept, sealed for this Windows user. Check it with `alelyon-cli auth status`.");
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let outcome: Result<(), String> = (|| match &cli.command {
        Command::Auth { action: Auth::Login } => login(),
        Command::Auth { action: Auth::Logout } => {
            if let Some(p) = kept_path() {
                alelyon_identity_client::vault::forget(&p);
            }
            eprintln!("Forgotten. Revoke the token in the developer console if it may have leaked.");
            Ok(())
        }
        Command::Auth { action: Auth::Status } => {
            let me = run_tool(&service(&cli.url)?, "pages_me", json!({}))?;
            let who = me.pointer("/me/username").cloned().unwrap_or(Value::Null);
            eprintln!("The token works. It acts as @{}.", who.as_str().unwrap_or("?"));
            Ok(())
        }
        Command::Tools => {
            for t in tools::TOOLS.iter().chain(tools::APP_TOOLS) {
                println!("{:<26} {:<15} {}", t.name, t.scope, t.description);
            }
            Ok(())
        }
        Command::Mcp => {
            // Pages needs a developer token; the app needs only its bridge. Without a token, the app's tools alone.
            let svc = service(&cli.url).ok();
            let app = app::App { handoff: app::handoff_path() };
            eprintln!(
                "alelyon-cli: serving {} over MCP on standard input and output.",
                if svc.is_some() { "Pages and app tools" } else { "app tools (no developer token for Pages)" }
            );
            let reach = mcp::Reach { pages: svc.as_ref().map(|s| s as &dyn Pages), app: &app };
            let stdin = std::io::stdin();
            mcp::serve(&reach, stdin.lock(), std::io::stdout().lock()).map_err(|_| "Standard input or output failed.".into())
        }
        Command::App { command, fields } => {
            let parsed: Value = match fields {
                Some(text) => serde_json::from_str(text).map_err(|_| "The fields are not valid JSON.")?,
                None => json!({}),
            };
            let app = app::App { handoff: app::handoff_path() };
            run_tool(&app, &format!("app_{command}"), parsed).map(|v| print(&v))
        }
        other => {
            let svc = service(&cli.url)?;
            let (name, fields) = match other {
                Command::Me => ("pages_me", json!({})),
                Command::Feed { before } => ("pages_feed", before.as_ref().map_or(json!({}), |b| json!({"before": b}))),
                Command::Post { text, reply_to } => {
                    let mut f = json!({"text": text});
                    if let Some(r) = reply_to {
                        f["reply_to"] = json!(r);
                    }
                    ("pages_post", f)
                }
                Command::Thread { post_id } => ("pages_thread", json!({"post_id": post_id})),
                Command::Person { username } => ("pages_person", json!({"username": username.trim_start_matches('@')})),
                Command::Search { query } => ("pages_search", json!({"q": query})),
                Command::Notifications => ("pages_notifications", json!({})),
                Command::Like { post_id, undo } => ("pages_like", json!({"post_id": post_id, "like": !undo})),
                Command::Follow { username, undo } => {
                    ("pages_follow", json!({"username": username.trim_start_matches('@'), "follow": !undo}))
                }
                Command::Call { tool, fields } => {
                    let parsed: Value = match fields {
                        Some(text) => serde_json::from_str(text).map_err(|_| "The fields are not valid JSON.")?,
                        None => json!({}),
                    };
                    return run_tool(&svc, tool, parsed).map(|v| print(&v));
                }
                _ => unreachable!(),
            };
            run_tool(&svc, name, fields).map(|v| print(&v))
        }
    })();
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(words) => {
            eprintln!("alelyon-cli: {words}");
            ExitCode::FAILURE
        }
    }
}
