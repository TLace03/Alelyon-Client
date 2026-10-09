//! The labs' agents as Lattice's models: each conversation that uses one has
//! a session of its own with it, kept while Lattice runs, so the agent keeps
//! its own history; a turn is one prompt to it ([`AgentModel`]).
//!
//! A turn of an agent is a plain turn of Lattice's chat: its answer streams
//! into the transcript as any model's does. The agent works with its own tools
//! in the conversation's folder (else a folder of its own under Lattice's
//! state), and each step it takes is a line of the answer. Whatever it asks
//! permission for is asked of the reader in the core's own dialog
//! ([`Permissions`]): yes takes the agent's allow-once option, no its reject
//! option, and nothing is allowed always. It reads files only inside its
//! folder, and Lattice writes none for it: it writes with its own tools, after
//! asking.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use lattice_agents::model::{
    GenerationTrace, InputItem, Model, ModelError, ModelEvent, ModelRequest, ModelResponse,
    OutputItem,
};
use lattice_sys::process::Child;
use serde_json::{Value, json};
use tokio::runtime::Handle;
use tokio::sync::mpsc;

use super::connection::Connection;
use super::session::{self, AgentSession, Client, Permission, Update};
use super::{Agent, agents_dir, drain, start};
use crate::env::Env;
use crate::state::StateRoot;

/// Asks the reader, in the core's dialog, whether the agent may do what it
/// asks; `true` is yes.
pub type Permissions = Arc<dyn Fn(Agent, Permission) -> BoxFuture<'static, bool> + Send + Sync>;

/// The folder a conversation works in, when it has one.
pub type Folders = Arc<dyn Fn(&str) -> Option<PathBuf> + Send + Sync>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What one session's agent sends, while a turn listens.
struct Sink {
    agent: Agent,
    folder: PathBuf,
    turn: Mutex<Option<mpsc::UnboundedSender<Update>>>,
    permissions: Permissions,
}

impl Client for Sink {
    fn update(&self, update: Update) {
        if let Some(turn) = lock(&self.turn).as_ref() {
            let _ = turn.send(update);
        }
    }

    fn permission(&self, permission: Permission) -> BoxFuture<'static, Option<String>> {
        let ask = (self.permissions)(self.agent, permission.clone());
        Box::pin(async move {
            let yes = ask.await;
            let wanted = if yes { "allow_once" } else { "reject_once" };
            permission
                .options
                .iter()
                .find(|(_, _, kind)| kind == wanted)
                .or_else(|| {
                    // A reject of any kind when no; never an always.
                    (!yes)
                        .then(|| {
                            permission
                                .options
                                .iter()
                                .find(|(_, _, kind)| kind.starts_with("reject"))
                        })
                        .flatten()
                })
                .map(|(id, ..)| id.clone())
        })
    }

    fn read(&self, path: &str, line: Option<u64>, limit: Option<u64>) -> Result<String, String> {
        let wanted = Path::new(path);
        let full = if wanted.is_absolute() {
            wanted.to_path_buf()
        } else {
            self.folder.join(wanted)
        };
        let (Ok(full), Ok(root)) = (full.canonicalize(), self.folder.canonicalize()) else {
            return Err("That file is not there.".to_owned());
        };
        if !full.starts_with(&root) {
            return Err("Lattice reads only files inside the agent's folder.".to_owned());
        }
        let text = std::fs::read_to_string(&full)
            .map_err(|_| "That file could not be read as text.".to_owned())?;
        let start = line.unwrap_or(1).saturating_sub(1) as usize;
        let lines = text.lines().skip(start);
        Ok(match limit {
            Some(limit) => lines.take(limit as usize).collect::<Vec<_>>().join("\n"),
            None => lines.collect::<Vec<_>>().join("\n"),
        })
    }

    fn write(&self, _path: &str, _content: &str) -> Result<(), String> {
        Err(
            "Lattice writes no file for the agent; it writes with its own tools, after asking."
                .to_owned(),
        )
    }
}

struct Live {
    session: AgentSession,
    sink: Arc<Sink>,
    _child: Child,
}

/// The agents' sessions, by (agent, conversation).
pub struct Pool {
    state: StateRoot,
    env: Arc<dyn Env>,
    handle: Handle,
    node: PathBuf,
    permissions: Permissions,
    folders: Folders,
    live: tokio::sync::Mutex<HashMap<(Agent, String), Arc<Live>>>,
}

impl Pool {
    pub fn new(
        state: StateRoot,
        env: Arc<dyn Env>,
        handle: Handle,
        node: PathBuf,
        permissions: Permissions,
        folders: Folders,
    ) -> Self {
        Self {
            state,
            env,
            handle,
            node,
            permissions,
            folders,
            live: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    /// The model a turn of `agent` in conversation `thread` runs.
    pub fn model(self: &Arc<Self>, agent: Agent, thread: &str) -> Arc<dyn Model> {
        Arc::new(AgentModel {
            pool: self.clone(),
            agent,
            thread: thread.to_owned(),
        })
    }

    /// The folder an agent works in for `thread`: the conversation's, else one
    /// of its own under Lattice's state.
    fn folder(&self, thread: &str) -> PathBuf {
        (self.folders)(thread).unwrap_or_else(|| agents_dir(&self.state).join("work").join(thread))
    }

    async fn session(&self, agent: Agent, thread: &str) -> Result<Arc<Live>, String> {
        let mut live = self.live.lock().await;
        let key = (agent, thread.to_owned());
        if let Some(found) = live.get(&key) {
            return Ok(found.clone());
        }
        let dir = agents_dir(&self.state);
        if !agent.installed(&dir) {
            return Err(format!(
                "{} is not installed: add it in Tools.",
                agent.label()
            ));
        }
        let folder = self.folder(thread);
        std::fs::create_dir_all(&folder)
            .map_err(|_| "The agent's folder could not be made.".to_owned())?;
        let mut child = start(
            agent,
            &dir,
            &self.node,
            &folder,
            self.env.as_ref(),
            &self.state.globals,
        )
        .map_err(|error| format!("{} could not start: {error}.", agent.label()))?;
        let (Some(stdout), Some(stdin), Some(stderr)) =
            (child.take_stdout(), child.take_stdin(), child.take_stderr())
        else {
            return Err(format!("{} started without its pipes.", agent.label()));
        };
        drain(stderr, Arc::new(|_| {}));
        let sink = Arc::new(Sink {
            agent,
            folder: folder.clone(),
            turn: Mutex::new(None),
            permissions: self.permissions.clone(),
        });
        let client: Arc<dyn Client> = sink.clone();
        let conn = Connection::start(
            agent.label(),
            Box::new(stdout),
            Box::new(stdin),
            session::events(client.clone(), Arc::new(|_| {})),
            session::asks(client, self.handle.clone()),
        )
        .map_err(|_| format!("{} could not be connected to.", agent.label()))?;
        let session = AgentSession::start(conn, &folder, agent.auth(), &agent.model()).await?;
        let made = Arc::new(Live {
            session,
            sink,
            _child: child,
        });
        live.insert(key, made.clone());
        Ok(made)
    }

    /// Start `agent`'s session for `thread` if it is not running; why not, in
    /// the agent's own words (a model error would hide them).
    pub async fn ready(&self, agent: Agent, thread: &str) -> Result<(), String> {
        self.session(agent, thread).await.map(|_| ())
    }

    /// End every session (Lattice closing).
    pub async fn stop(&self) {
        for (_, live) in self.live.lock().await.drain() {
            live.session.close();
        }
    }
}

/// One turn of an agent: the last thing the reader wrote, as a prompt to the
/// conversation's session.
struct AgentModel {
    pool: Arc<Pool>,
    agent: Agent,
    thread: String,
}

/// A step the agent took, as a line of its answer.
fn step_line(title: &str, status: &str) -> String {
    let status = if status.is_empty() {
        String::new()
    } else {
        format!(" ({status})")
    };
    format!("\n\n*{title}{status}*\n\n")
}

impl Model for AgentModel {
    fn name(&self) -> &str {
        self.agent.choice()
    }

    fn config_for_trace(&self) -> Value {
        json!({"agent": self.agent.label()})
    }

    fn generation_trace(&self) -> GenerationTrace {
        GenerationTrace::Bare
    }

    fn stream(&self, request: ModelRequest) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
        let text = request
            .input
            .iter()
            .rev()
            .find_map(|item| match item {
                InputItem::User(text) | InputItem::UserImages { text, .. } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let (pool, agent, thread) = (self.pool.clone(), self.agent, self.thread.clone());
        let (events, out) = mpsc::unbounded_channel::<Result<ModelEvent, ModelError>>();
        let handle = pool.handle.clone();
        handle.spawn(async move {
            let live = match pool.session(agent, &thread).await {
                Ok(live) => live,
                Err(why) => {
                    let _ = events.send(Err(ModelError::Connection(why)));
                    return;
                }
            };
            let (updates, mut heard) = mpsc::unbounded_channel();
            *lock(&live.sink.turn) = Some(updates);
            let prompt = live.session.prompt(&text);
            tokio::pin!(prompt);
            let mut answer = String::new();
            let ended = loop {
                tokio::select! {
                    ended = &mut prompt => break ended,
                    Some(update) = heard.recv() => {
                        let piece = match update {
                            Update::Text(text) => Some(ModelEvent::TextDelta(text)),
                            Update::Thought(text) => Some(ModelEvent::ReasoningDelta(text)),
                            Update::ToolCall { title, status, .. } => Some(ModelEvent::TextDelta(step_line(&title, &status))),
                            _ => None,
                        };
                        if let Some(piece) = piece {
                            if let ModelEvent::TextDelta(text) = &piece {
                                answer.push_str(text);
                            }
                            if events.send(Ok(piece)).is_err() {
                                // The turn was stopped: ask the agent to stop too.
                                live.session.cancel();
                            }
                        }
                    }
                }
            };
            *lock(&live.sink.turn) = None;
            while let Ok(update) = heard.try_recv() {
                if let Update::Text(text) = update {
                    answer.push_str(&text);
                    let _ = events.send(Ok(ModelEvent::TextDelta(text)));
                }
            }
            match ended {
                Ok(_) => {
                    let _ = events.send(Ok(ModelEvent::Done(ModelResponse {
                        output: vec![OutputItem::Message { text: answer }],
                        usage: None,
                    })));
                }
                Err(why) => {
                    let _ = events.send(Err(ModelError::Connection(why)));
                }
            }
        });
        tokio_stream_from(out)
    }
}

fn tokio_stream_from(
    mut out: mpsc::UnboundedReceiver<Result<ModelEvent, ModelError>>,
) -> BoxStream<'static, Result<ModelEvent, ModelError>> {
    futures::stream::poll_fn(move |cx| out.poll_recv(cx)).boxed()
}
