//! Finding connectors (tool parity with the Claude desktop app's registry
//! tools): `find_connectors` searches a catalog of MCP servers Lattice knows,
//! and `suggest_connector` offers the user one of them for their MCP settings.
//!
//! - **The catalog** ([`CATALOG`]) is built in and read here, with no network:
//!   servers that run on this PC over stdio (Lattice refuses remote servers,
//!   T7), each with its command as its project documents it. Its commands
//!   fetch their package (from npm or PyPI) when they first start.
//! - **Asked first:** a suggestion adds the server to the user's own MCP
//!   settings (`crate::mcp::config::put_user_server`) only after their yes in
//!   the core's dialog ([`crate::ports::ConfirmRequest::AddConnector`]), which
//!   shows its command. Declaring a server grants nothing: it starts only when
//!   the user allows it, as any server they declare (FT4).
//! - A name already in the user's settings is never replaced.

use serde_json::{Value, json};

use super::turn::TurnTools;
use crate::mcp::config;
use crate::ports::{ConfirmRequest, Initiated};
use crate::tools::read::ToolError;

/// One server of the catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Connector {
    /// Its id, and the name it is declared under.
    pub id: &'static str,
    pub what: &'static str,
    /// Words it is found by, besides its id and what it does.
    pub words: &'static str,
    pub command: &'static str,
    pub args: &'static [&'static str],
    /// Where its command is documented.
    pub source: &'static str,
}

/// The servers Lattice knows: the Model Context Protocol project's reference
/// servers and Microsoft's Playwright server, as their projects document them.
pub const CATALOG: [Connector; 6] = [
    Connector {
        id: "fetch",
        what: "Fetch a web page and read it as Markdown.",
        words: "web url http page download read markdown",
        command: "uvx",
        args: &["mcp-server-fetch"],
        source: "github.com/modelcontextprotocol/servers (src/fetch)",
    },
    Connector {
        id: "git",
        what: "Read and work with Git repositories: status, diff, log, commit, branches.",
        words: "git repository commit diff branch log version control",
        command: "uvx",
        args: &["mcp-server-git"],
        source: "github.com/modelcontextprotocol/servers (src/git)",
    },
    Connector {
        id: "memory",
        what: "A knowledge graph the model keeps between conversations: entities, relations, observations.",
        words: "memory knowledge graph remember notes persistent",
        command: "npx",
        args: &["-y", "@modelcontextprotocol/server-memory"],
        source: "github.com/modelcontextprotocol/servers (src/memory)",
    },
    Connector {
        id: "time",
        what: "The current time and conversions between time zones.",
        words: "time zone clock date timezone convert",
        command: "uvx",
        args: &["mcp-server-time"],
        source: "github.com/modelcontextprotocol/servers (src/time)",
    },
    Connector {
        id: "sequential-thinking",
        what: "A tool for thinking a problem through in revisable steps.",
        words: "thinking reasoning steps plan problem solving",
        command: "npx",
        args: &["-y", "@modelcontextprotocol/server-sequential-thinking"],
        source: "github.com/modelcontextprotocol/servers (src/sequentialthinking)",
    },
    Connector {
        id: "playwright",
        what: "Drive a browser of its own with Playwright: open pages, click, type, read the page's accessibility tree.",
        words: "browser playwright automation test web click page e2e",
        command: "npx",
        args: &["@playwright/mcp@latest"],
        source: "github.com/microsoft/playwright-mcp",
    },
];

impl Connector {
    pub fn command_line(&self) -> String {
        std::iter::once(self.command).chain(self.args.iter().copied()).collect::<Vec<_>>().join(" ")
    }

    fn entry(&self) -> Value {
        json!({"command": self.command, "args": self.args})
    }

    fn line(&self) -> String {
        format!("{}: {} Runs `{}` ({}).", self.id, self.what, self.command_line(), self.source)
    }
}

/// The catalog's servers that `query`'s words find (every one for no words),
/// best first.
pub fn find(query: &str) -> Vec<&'static Connector> {
    let words: Vec<String> = query.to_lowercase().split_whitespace().map(str::to_owned).collect();
    let mut found: Vec<(usize, &Connector)> = CATALOG
        .iter()
        .map(|c| {
            let text = format!("{} {} {}", c.id, c.what, c.words).to_lowercase();
            (words.iter().filter(|w| text.contains(w.as_str())).count(), c)
        })
        .filter(|(hits, _)| words.is_empty() || *hits > 0)
        .collect();
    found.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.id.cmp(b.1.id)));
    found.into_iter().map(|(_, c)| c).collect()
}

/// `find_connectors {query?}`.
pub(crate) fn find_tool(tools: &TurnTools, args: &Value) -> Result<String, ToolError> {
    let query = args.get("query").and_then(Value::as_str).unwrap_or("");
    let found = find(query);
    let declared: Vec<String> = config::load_user(&tools.inner.config.state)
        .servers
        .into_iter()
        .map(|s| s.key.name)
        .collect();
    if found.is_empty() {
        return Ok(format!(
            "No connector in Lattice's catalog matches. It knows: {}.",
            CATALOG.iter().map(|c| c.id).collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(found
        .iter()
        .map(|c| {
            let mut line = c.line();
            if declared.iter().any(|name| name == c.id) {
                line.push_str(" Already in the user's MCP settings.");
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

/// `suggest_connector {id, why}`: add it to the user's MCP settings after their yes.
pub(crate) async fn suggest_tool(tools: &TurnTools, call: &str, args: &Value) -> Result<String, ToolError> {
    let id = args.get("id").and_then(Value::as_str).unwrap_or("").trim();
    let why = args.get("why").and_then(Value::as_str).unwrap_or("").trim();
    let connector = CATALOG.iter().find(|c| c.id == id).ok_or_else(|| {
        ToolError::new(format!(
            "There is no connector {id} in Lattice's catalog; find_connectors lists them."
        ))
    })?;
    if why.is_empty() || why.chars().count() > 300 {
        return Err(ToolError::new("Say in one or two sentences (at most 300 characters) why the user would want it."));
    }
    let state = &tools.inner.config.state;
    if config::load_user(state).servers.iter().any(|s| s.key.name == connector.id) {
        return Err(ToolError::new(format!(
            "{} is already in the user's MCP settings; it starts when they allow it.",
            connector.id
        )));
    }
    let allowed = tools
        .inner
        .confirmer
        .ask(
            &format!("add-connector:{}:{call}", tools.convo.id),
            ConfirmRequest::AddConnector {
                name: connector.id.to_owned(),
                command: connector.command_line(),
                why: why.to_owned(),
            },
            Initiated::Page,
        )
        .await;
    if !allowed {
        return Err(ToolError::new("The user did not add that connector."));
    }
    config::put_user_server(state, connector.id, &connector.entry()).map_err(ToolError)?;
    Ok(format!(
        "Added {} to the user's MCP settings. It starts when they allow it, and its tools are offered from then on.",
        connector.id
    ))
}
