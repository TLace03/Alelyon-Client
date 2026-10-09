//! The labs' own coding agents, run on the reader's own subscriptions over the
//! Agent Client Protocol, through one ACP client and two adapters. Not a
//! port.
//!
//! - **Claude Code**, through the adapter `@agentclientprotocol/claude-agent-acp`
//!   (`node dist/index.js`), signed in as the reader's `claude` CLI is (its own
//!   `/login`), on its own default model.
//! - **Codex**, through `@agentclientprotocol/codex-acp` (`node
//!   dist/index.js`), signed in with the reader's ChatGPT plan
//!   (`authenticate {methodId: "chat-gpt"}`), its model chosen by Lattice
//!   (a model a ChatGPT plan does not offer is refused).
//!
//! Gemini CLI is not here: on 2026-10-09 Google refused its "Log in with
//! Google" for individuals ("no longer supported ... migrate to the
//! Antigravity suite"), so it could run only on an API key, not a
//! subscription.
//!
//! Each agent's login stays its CLI's own: Lattice never reads, copies or sends
//! a token. The adapters live in Lattice's own folder ([`agents_dir`]),
//! installed there only by the reader's action. An agent runs in a Job Object
//! with the safe environment (`crate::exec::spawn`) plus what a CLI needs to
//! find its own login and the network ([`AGENT_ENV_NAMES`]), and never
//! `CLAUDECODE` (Claude Code refuses to start inside another Claude Code).

pub mod connection;
pub mod pool;
pub mod session;
#[cfg(test)]
mod tests;

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use lattice_sys::process::{Child, JobLimits, SpawnRequest, spawn_with_input};

use crate::env::Env;
use crate::exec::spawn::child_environment;
use crate::state::StateRoot;
pub use session::Model;

/// An agent's Job: a Node runtime and the CLI's own processes.
pub const LIMITS: JobLimits = JobLimits {
    active_processes: 128,
    job_memory: 8 * 1024 * 1024 * 1024,
};

/// What an agent may read of Lattice's environment beyond the safe names:
/// where Windows keeps a user's settings (a CLI's login lives under them), a
/// CLI's own config folder when one is set, and the proxy settings.
pub const AGENT_ENV_NAMES: [&str; 8] = [
    "APPDATA",
    "LOCALAPPDATA",
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "NODE_EXTRA_CA_CERTS",
];

/// One of the labs' agents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Agent {
    ClaudeCode,
    Codex,
}

impl Agent {
    pub const ALL: [Agent; 2] = [Agent::ClaudeCode, Agent::Codex];

    /// Its name in the model list.
    pub fn label(self) -> &'static str {
        match self {
            Agent::ClaudeCode => "Claude Code (your subscription)",
            Agent::Codex => "Codex (your ChatGPT plan)",
        }
    }

    /// The choice's id (`crate::chat::vocab`).
    pub fn choice(self) -> &'static str {
        match self {
            Agent::ClaudeCode => "agent:claude-code",
            Agent::Codex => "agent:codex",
        }
    }

    /// The agent a choice names.
    pub fn from_choice(choice: &str) -> Option<Agent> {
        Agent::ALL.into_iter().find(|a| a.choice() == choice)
    }

    /// The npm package of its adapter, with the version measured.
    pub fn package(self) -> &'static str {
        match self {
            Agent::ClaudeCode => "@agentclientprotocol/claude-agent-acp@0.88.0",
            Agent::Codex => "@agentclientprotocol/codex-acp@2.1.1",
        }
    }

    /// How it signs in, when it needs the client to say.
    pub fn auth(self) -> Option<&'static str> {
        match self {
            Agent::ClaudeCode => None,
            Agent::Codex => Some("chat-gpt"),
        }
    }

    /// The model Lattice chooses for it.
    pub fn model(self) -> Model {
        match self {
            // The renamed adapter answers on its own default (measured).
            Agent::ClaudeCode => Model::Agents,
            // The newest a ChatGPT plan offers, reasoning high (measured).
            Agent::Codex => Model::SetModel("gpt-6-luna[high]".to_owned()),
        }
    }

    /// How it starts, given the adapters' folder and Node: (program, argv).
    pub fn command(self, dir: &Path, node: &Path) -> (PathBuf, Vec<OsString>) {
        let script = self.script(dir);
        (
            node.to_path_buf(),
            vec![node.as_os_str().to_owned(), script.into_os_string()],
        )
    }

    /// Its adapter's script (each runs on Node).
    fn script(self, dir: &Path) -> PathBuf {
        let package = match self {
            Agent::ClaudeCode => "claude-agent-acp",
            Agent::Codex => "codex-acp",
        };
        dir.join("node_modules")
            .join("@agentclientprotocol")
            .join(package)
            .join("dist")
            .join("index.js")
    }

    /// Whether its adapter is installed in `dir`.
    pub fn installed(self, dir: &Path) -> bool {
        self.script(dir).is_file()
    }
}

/// Node, found on `PATH` as Windows finds a program (else where its installer
/// puts it): Claude Code's adapter runs on it.
pub fn find_node(env: &dyn Env) -> PathBuf {
    let path = env.var("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .map(|dir| dir.join("node.exe"))
        .find(|candidate| candidate.is_absolute() && candidate.is_file())
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files\nodejs\node.exe"))
}

/// `<globals>/lattice_native/agents`: the adapters, installed by the reader.
pub fn agents_dir(state: &StateRoot) -> PathBuf {
    state.globals.join("lattice_native").join("agents")
}

/// An agent's environment: the safe block, then [`AGENT_ENV_NAMES`] as this
/// process has them; never `CLAUDECODE` or `CLAUDE_CODE_ENTRYPOINT`.
pub fn environment(
    env: &dyn Env,
    workspace: Option<&Path>,
    globals: &Path,
) -> Vec<(OsString, OsString)> {
    let mut block = child_environment(env, workspace, globals);
    for name in AGENT_ENV_NAMES {
        // The safe block may already name it: each name once.
        let named = block
            .iter()
            .any(|(have, _)| have.to_string_lossy().eq_ignore_ascii_case(name));
        if named {
            continue;
        }
        if let Some(value) = env.var(name) {
            block.push((OsString::from(name), value));
        }
    }
    block.retain(|(name, _)| {
        let name = name.to_string_lossy();
        !name.eq_ignore_ascii_case("CLAUDECODE")
            && !name.eq_ignore_ascii_case("CLAUDE_CODE_ENTRYPOINT")
    });
    block
}

/// Start `agent` in `cwd`, with its stdin and stdout piped.
pub fn start(
    agent: Agent,
    dir: &Path,
    node: &Path,
    cwd: &Path,
    env: &dyn Env,
    globals: &Path,
) -> std::io::Result<Child> {
    let (program, argv) = agent.command(dir, node);
    let block = environment(env, Some(cwd), globals);
    spawn_with_input(&SpawnRequest {
        program: &program,
        argv: &argv,
        cwd,
        env: &block,
        limits: LIMITS,
    })
}

/// Read an agent's stderr on a thread of its own until it ends, handing each
/// line (at most 2,000 characters) to `log`: an agent that writes much to
/// stderr (Codex does) would otherwise block once the pipe is full.
pub fn drain(stderr: std::fs::File, log: std::sync::Arc<dyn Fn(String) + Send + Sync>) {
    use std::io::BufRead;
    let _ = std::thread::Builder::new()
        .name("lattice-acp-err".to_owned())
        .spawn(move || {
            for line in std::io::BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                log(connection::cut(&line, 2000));
            }
        });
}

/// How long an install may take (a download of about 300 MB on a slow line).
pub const INSTALL_WAIT: std::time::Duration = std::time::Duration::from_secs(20 * 60);

/// Install `agent`'s adapter into [`agents_dir`] with npm, beside Node: the
/// reader's own action, which downloads these two packages. Blocking; the
/// last lines npm wrote, when it failed.
pub fn install(agent: Agent, state: &StateRoot, env: &dyn Env) -> Result<(), String> {
    use lattice_sys::process::{BatchRequest, spawn_batch_with_input};
    let dir = agents_dir(state);
    std::fs::create_dir_all(&dir)
        .map_err(|_| "Lattice's agents folder could not be made.".to_owned())?;
    let node = find_node(env);
    let npm = node.with_file_name("npm.cmd");
    if !npm.is_file() {
        return Err(
            "npm was not found beside Node: install Node.js (it brings npm), then try again."
                .to_owned(),
        );
    }
    let system = env
        .var("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    let cmd = system.join("System32").join("cmd.exe");
    let args: Vec<OsString> = ["install", "--no-audit", "--no-fund", "--prefix"]
        .into_iter()
        .map(OsString::from)
        .chain([
            dir.clone().into_os_string(),
            OsString::from(agent.package()),
        ])
        .collect();
    let block = environment(env, None, &state.globals);
    let mut child = spawn_batch_with_input(&BatchRequest {
        cmd: &cmd,
        script: &npm,
        args: &args,
        cwd: &dir,
        env: &block,
        limits: LIMITS,
    })
    .map_err(|error| format!("npm could not start: {error}."))?;
    drop(child.take_stdin());
    let said = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    for stream in [child.take_stdout(), child.take_stderr()]
        .into_iter()
        .flatten()
    {
        let kept = said.clone();
        drain(
            stream,
            std::sync::Arc::new(move |line: String| {
                let mut kept = kept.lock().unwrap_or_else(|p| p.into_inner());
                if kept.len() >= 20 {
                    kept.remove(0);
                }
                kept.push(line);
            }),
        );
    }
    match child.wait(Some(INSTALL_WAIT)) {
        Ok(Some(0)) if agent.installed(&dir) => Ok(()),
        Ok(Some(code)) => Err(format!(
            "npm ended with code {code}: {}",
            said.lock().unwrap_or_else(|p| p.into_inner()).join(" / ")
        )),
        Ok(None) => Err("npm did not finish in 20 minutes.".to_owned()),
        Err(error) => Err(format!("npm could not be waited for: {error}.")),
    }
}
