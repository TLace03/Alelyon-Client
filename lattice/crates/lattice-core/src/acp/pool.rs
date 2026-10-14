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
//! option, and nothing is allowed always.
//!
//! **Its client reads and writes** (`fs/read_text_file`, `fs/write_text_file`)
//! in a conversation's folder go through the conversation ([`ChatFiles`]): a
//! read sees the folder as the conversation does, its staged changes
//! included, under the same path rules as Lattice's own read tools; a write
//! becomes a staged change in the conversation's review, recorded under the
//! turn that made it, and reaches the disk only when the reader keeps it. A
//! path outside the folder is refused. Lattice offers the write
//! (`writeTextFile`) only to a session in a conversation's folder; in a
//! folder of its own the agent reads there and writes with its own tools.
//!
//! What an agent writes **with its own tools** (its shell, or an edit tool
//! that does not ask its client) does not pass through Lattice and is not
//! staged: Lattice cannot intercept it. Such writes are on disk at once and
//! are listed after the turn as changes already made (`crate::convo::lab_turns`).

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

/// A conversation's folder as its agent's client reads and writes reach it.
/// Paths are relative to the folder, with forward slashes; the answers are
/// one sentence when refused.
pub trait ChatFiles: Send + Sync {
    /// The whole text of `path` as conversation `thread` sees it (its staged
    /// changes included).
    fn read(&self, thread: &str, path: &str) -> Result<String, String>;
    /// Stage `content` as the new text of `path` in `thread`'s review, as
    /// call `call` of the turn running there; what was staged, in words.
    /// Nothing is written to the folder.
    fn write(&self, thread: &str, path: &str, content: &str, call: &str) -> Result<String, String>;
}

/// Refused: the path is not inside the conversation's folder.
pub const OUTSIDE: &str = "Lattice reads and writes only files inside this chat's folder.";
/// Refused: a write in a session with no conversation folder (and so no review).
pub const NO_REVIEW: &str = "This chat has no folder, so Lattice has no review to stage the change in; nothing was written.";

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
    /// The changes the agent's tool calls announced, by call, until asked about.
    diffs: Mutex<std::collections::HashMap<String, Vec<super::session::Diff>>>,
    /// The conversation this session serves.
    thread: String,
    /// Its folder's reads and writes, when the session works in a
    /// conversation's folder (its writes are then staged for review);
    /// `None` in a folder of the agent's own.
    files: Option<Arc<dyn ChatFiles>>,
    /// The agent's latest tool call, which a client write is recorded under.
    last_call: Mutex<Option<String>>,
    /// Client writes so far, numbering those made outside a known tool call.
    writes: std::sync::atomic::AtomicU64,
}

impl Sink {
    /// `path` as the conversation's tools name it: relative to the folder,
    /// with forward slashes. An absolute path (ACP's own form) must be inside
    /// the folder; a relative one is taken as relative to it, and the path
    /// rules then refuse any `..` or other escape.
    fn relative(&self, path: &str) -> Result<String, String> {
        let absolute =
            Path::new(path).is_absolute() || path.starts_with(['/', '\\']) || path.contains(':');
        if absolute {
            inside(&self.folder, path).ok_or_else(|| OUTSIDE.to_owned())
        } else {
            Ok(path.replace('\\', "/"))
        }
    }

    /// The call id a client write is recorded under: the agent's latest
    /// tool call, else a numbered one of its own.
    fn write_call(&self) -> String {
        let n = self
            .writes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        lock(&self.last_call)
            .as_ref()
            .map(|id| format!("agent_{id}"))
            .filter(|id| lattice_protocol::conversation::is_call_id(id))
            .unwrap_or_else(|| format!("agent_write_{n}"))
    }

    /// A file in the agent's own folder (no conversation folder): its whole
    /// text, when it is inside that folder.
    fn read_own(&self, path: &str) -> Result<String, String> {
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
        std::fs::read_to_string(&full)
            .map_err(|_| "That file could not be read as text.".to_owned())
    }
}

/// `text` from `line` (1-based) for `limit` lines, each with its line end;
/// the whole text, exactly, when neither is given.
fn slice(text: &str, line: Option<u64>, limit: Option<u64>) -> String {
    if line.is_none() && limit.is_none() {
        return text.to_owned();
    }
    let start = usize::try_from(line.unwrap_or(1).saturating_sub(1)).unwrap_or(usize::MAX);
    let lines = text.split_inclusive('\n').skip(start);
    match limit {
        Some(limit) => lines
            .take(usize::try_from(limit).unwrap_or(usize::MAX))
            .collect(),
        None => lines.collect(),
    }
}

impl Client for Sink {
    fn update(&self, update: Update) {
        // A tool call's proposed changes, for its permission question.
        if let Update::ToolCall { id, diffs, .. } = &update
            && !diffs.is_empty()
        {
            let mut known = lock(&self.diffs);
            if known.len() >= 64 {
                known.clear();
            }
            known.insert(id.clone(), diffs.clone());
        }
        if let Update::ToolCall { id, .. } = &update
            && !id.is_empty()
        {
            *lock(&self.last_call) = Some(id.clone());
        }
        if let Some(turn) = lock(&self.turn).as_ref() {
            let _ = turn.send(update);
        }
    }

    fn permission(&self, mut permission: Permission) -> BoxFuture<'static, Option<String>> {
        // Its changes: in the question itself, else as its tool call announced them.
        if permission.diffs.is_empty() {
            permission.diffs = lock(&self.diffs)
                .remove(&permission.call_id)
                .unwrap_or_default();
        }
        // Shown by their place in the chat's folder.
        for diff in &mut permission.diffs {
            if let Some(inside) = inside(&self.folder, &diff.path) {
                diff.path = inside;
            }
        }
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
        let text = match &self.files {
            Some(files) => files.read(&self.thread, &self.relative(path)?)?,
            None => self.read_own(path)?,
        };
        Ok(slice(&text, line, limit))
    }

    /// Never a write to the folder: a staged change in the conversation's
    /// review, or a refusal.
    fn write(&self, path: &str, content: &str) -> Result<(), String> {
        let Some(files) = &self.files else {
            return Err(NO_REVIEW.to_owned());
        };
        let relative = self.relative(path)?;
        files
            .write(&self.thread, &relative, content, &self.write_call())
            .map(|_| ())
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
    /// The MCP servers every session is offered (the checklist server).
    servers: Vec<Value>,
    permissions: Permissions,
    folders: Folders,
    /// A conversation folder's reads and staged writes.
    files: Arc<dyn ChatFiles>,
    live: tokio::sync::Mutex<HashMap<(Agent, String), Arc<Live>>>,
    /// The folder of a new chat about to be sent to `agent`, before its
    /// conversation is known by id (`Pool::next_folder`).
    next: Mutex<HashMap<Agent, PathBuf>>,
}

impl Pool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        state: StateRoot,
        env: Arc<dyn Env>,
        handle: Handle,
        node: PathBuf,
        servers: Vec<Value>,
        permissions: Permissions,
        folders: Folders,
        files: Arc<dyn ChatFiles>,
    ) -> Self {
        Self {
            state,
            env,
            handle,
            node,
            servers,
            permissions,
            folders,
            files,
            live: tokio::sync::Mutex::new(HashMap::new()),
            next: Mutex::default(),
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
    /// of its own under Lattice's state; `true` for the conversation's.
    fn folder(&self, agent: Agent, thread: &str) -> (PathBuf, bool) {
        match (self.folders)(thread).or_else(|| lock(&self.next).get(&agent).cloned()) {
            Some(folder) => (folder, true),
            None => (agents_dir(&self.state).join("work").join(thread), false),
        }
    }

    /// A new chat with a folder is about to be sent to `agent`: its first
    /// session works there even if it opens before the conversation is bound
    /// to the folder. `None` clears it once the conversation is bound.
    pub fn next_folder(&self, agent: Agent, folder: Option<PathBuf>) {
        let mut next = lock(&self.next);
        match folder {
            Some(folder) => {
                next.insert(agent, folder);
            }
            None => {
                next.remove(&agent);
            }
        }
    }

    async fn session(&self, agent: Agent, thread: &str) -> Result<Arc<Live>, String> {
        let mut live = self.live.lock().await;
        let key = (agent, thread.to_owned());
        if let Some(found) = live.get(&key) {
            return Ok(found.clone());
        }
        let (folder, chats) = self.folder(agent, thread);
        let reviewed = chats.then_some(thread);
        let made = Arc::new(self.open(agent, folder, reviewed).await?);
        live.insert(key, made.clone());
        Ok(made)
    }

    /// Open a session of `agent` in `folder`, on the reader's pick of model
    /// (`super::models`), and keep what it offered. `reviewed` names the
    /// conversation whose folder it is: its client reads and writes then go
    /// through that conversation, and the write is offered.
    async fn open(
        &self,
        agent: Agent,
        folder: PathBuf,
        reviewed: Option<&str>,
    ) -> Result<Live, String> {
        let dir = agents_dir(&self.state);
        if !agent.installed(&dir) {
            return Err(format!(
                "{} is not installed: add it in Tools.",
                agent.label()
            ));
        }
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
            diffs: Mutex::default(),
            thread: reviewed.unwrap_or_default().to_owned(),
            files: reviewed.map(|_| self.files.clone()),
            last_call: Mutex::default(),
            writes: Default::default(),
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
        let model = super::models::model_for(&dir, agent);
        let session = AgentSession::start(
            conn,
            &folder,
            agent.auth(),
            &model,
            agent.mode(),
            reviewed.is_some(),
            &self.servers,
        )
        .await?;
        // Kept for the reader's model picker; a failure to keep it changes nothing here.
        let _ = super::models::remember_offered(&dir, agent, &session.offered);
        Ok(Live {
            session,
            sink,
            _child: child,
        })
    }

    /// Check models: open a session of `agent` in a folder of its own only to
    /// read the models it offers (kept as any session's are), then end it.
    pub async fn check_models(&self, agent: Agent) -> Result<super::models::Offered, String> {
        let folder = agents_dir(&self.state)
            .join("check")
            .join(agent.choice().replace(':', "-"));
        let live = self.open(agent, folder, None).await?;
        let offered = live.session.offered.clone();
        live.session.close();
        if offered.models.is_empty() {
            return Err(format!("{} offered no model to choose.", agent.label()));
        }
        Ok(offered)
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

/// `path`'s place inside `folder` (forward slashes), when it is inside it: as
/// the folder is named, or as its real path (an agent may report a folder
/// reached through a link by where it really is), without regard to case or a
/// verbatim prefix.
fn inside(folder: &Path, path: &str) -> Option<String> {
    let plain = |p: &str| {
        let p = p.replace('/', "\\");
        p.strip_prefix(r"\\?\").map(str::to_owned).unwrap_or(p)
    };
    let wanted = plain(path);
    let mut names = vec![folder.to_string_lossy().into_owned()];
    if let Ok(real) = std::fs::canonicalize(folder) {
        names.push(real.to_string_lossy().into_owned());
    }
    names.into_iter().find_map(|name| {
        let base = plain(name.trim_end_matches(['\\', '/']));
        let n = base.len();
        let head = wanted.get(..n)?;
        let rest = wanted[n..].strip_prefix('\\')?;
        (head.eq_ignore_ascii_case(&base) && !rest.is_empty()).then(|| rest.replace('\\', "/"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acp::session::Diff;

    fn sink(folder: PathBuf, permissions: Permissions, files: Option<Arc<dyn ChatFiles>>) -> Sink {
        Sink {
            agent: Agent::Codex,
            folder,
            turn: Mutex::new(None),
            permissions,
            diffs: Mutex::default(),
            thread: "c1".to_owned(),
            files,
            last_call: Mutex::default(),
            writes: Default::default(),
        }
    }

    fn no_permissions() -> Permissions {
        Arc::new(|_, _| Box::pin(async { false }))
    }

    /// A conversation's files that record what reached them.
    #[derive(Default)]
    struct Files {
        reads: Mutex<Vec<(String, String)>>,
        writes: Mutex<Vec<(String, String, String, String)>>,
    }

    impl ChatFiles for Files {
        fn read(&self, thread: &str, path: &str) -> Result<String, String> {
            self.reads
                .lock()
                .unwrap()
                .push((thread.into(), path.into()));
            Ok("one\r\ntwo\nthree\n".to_owned())
        }
        fn write(
            &self,
            thread: &str,
            path: &str,
            content: &str,
            call: &str,
        ) -> Result<String, String> {
            self.writes.lock().unwrap().push((
                thread.into(),
                path.into(),
                content.into(),
                call.into(),
            ));
            Ok(format!("Staged `{path}`."))
        }
    }

    /// In a conversation's folder, a client read and write reach the
    /// conversation by their place in the folder, the write under the
    /// agent's latest tool call; a path outside the folder (or one beside
    /// it) is refused before it reaches the conversation. A whole read is
    /// the text exactly, line ends kept; a ranged one keeps them too.
    /// Mutant: a write sent on to the conversation without the folder check.
    #[test]
    fn client_reads_and_writes_go_through_the_conversation_inside_its_folder() {
        let dir = crate::testkit::TempDir::new("acp-chat-files");
        let folder = dir.path().join("work");
        std::fs::create_dir_all(&folder).unwrap();
        let files = Arc::new(Files::default());
        let sink = sink(folder.clone(), no_permissions(), Some(files.clone()));
        let at = |rest: &str| folder.join(rest).to_string_lossy().into_owned();

        assert_eq!(
            sink.read(&at("src/a.txt"), None, None).unwrap(),
            "one\r\ntwo\nthree\n"
        );
        assert_eq!(
            sink.read(&at("src/a.txt"), Some(2), Some(1)).unwrap(),
            "two\n"
        );
        assert_eq!(
            files.reads.lock().unwrap()[0],
            ("c1".into(), "src/a.txt".into())
        );

        sink.update(Update::ToolCall {
            id: "toolu_1".into(),
            title: "Write".into(),
            kind: "edit".into(),
            status: "pending".into(),
            diffs: Vec::new(),
        });
        sink.write(&at("src/b.txt"), "new\n").unwrap();
        assert_eq!(
            files.writes.lock().unwrap()[0],
            (
                "c1".into(),
                "src/b.txt".into(),
                "new\n".into(),
                "agent_toolu_1".into()
            )
        );

        let beside = dir.path().join("workshop").join("x.txt");
        let outside = dir.path().join("x.txt");
        for path in [beside, outside] {
            let path = path.to_string_lossy().into_owned();
            assert_eq!(sink.write(&path, "x").unwrap_err(), OUTSIDE);
            assert_eq!(sink.read(&path, None, None).unwrap_err(), OUTSIDE);
        }
        assert_eq!(
            files.writes.lock().unwrap().len(),
            1,
            "nothing outside reached it"
        );
        assert_eq!(files.reads.lock().unwrap().len(), 2);
    }

    /// In a folder of the agent's own (no conversation folder), a client
    /// write is refused and nothing is written there.
    #[test]
    fn a_client_write_without_a_conversation_folder_is_refused() {
        let dir = crate::testkit::TempDir::new("acp-own-folder");
        let sink = sink(dir.path().to_path_buf(), no_permissions(), None);
        let path = dir.path().join("a.txt");
        assert_eq!(
            sink.write(&path.to_string_lossy(), "x").unwrap_err(),
            NO_REVIEW
        );
        assert!(!path.exists());
        std::fs::write(&path, "kept\r\n").unwrap();
        assert_eq!(
            sink.read(&path.to_string_lossy(), None, None).unwrap(),
            "kept\r\n"
        );
    }

    /// A permission question without its own diff takes the one its tool call
    /// announced (Codex's way), with paths shown inside the chat's folder; one
    /// that carries its diff (Claude Code's way) keeps it; yes takes the
    /// allow-once option, no a reject.
    #[test]
    fn a_question_shows_the_change_its_tool_call_announced() {
        let seen: Arc<Mutex<Vec<Permission>>> = Arc::default();
        let answer = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let permissions: Permissions = {
            let (seen, answer) = (seen.clone(), answer.clone());
            Arc::new(move |_, permission| {
                seen.lock().unwrap().push(permission);
                let yes = answer.load(std::sync::atomic::Ordering::SeqCst);
                Box::pin(async move { yes })
            })
        };
        let folder = PathBuf::from(r"C:\work");
        let sink = sink(folder.clone(), permissions, None);
        let diff = |path: &str| Diff {
            path: path.into(),
            old: Some("a\n".into()),
            new: "b\n".into(),
        };
        sink.update(Update::ToolCall {
            id: "t1".into(),
            title: "Editing files".into(),
            kind: "edit".into(),
            status: "in_progress".into(),
            diffs: vec![diff(r"C:\work\src\a.txt")],
        });
        let question = |call: &str, diffs: Vec<Diff>| Permission {
            title: "Edit files".into(),
            detail: String::new(),
            options: vec![
                ("ok".into(), "Allow".into(), "allow_once".into()),
                ("always".into(), "Always".into(), "allow_always".into()),
                ("no".into(), "Reject".into(), "reject_once".into()),
            ],
            call_id: call.into(),
            diffs,
        };
        let pick = futures::executor::block_on(sink.permission(question("t1", Vec::new())));
        assert_eq!(pick.as_deref(), Some("ok"));
        assert_eq!(seen.lock().unwrap()[0].diffs, [diff("src/a.txt")]);
        // Its own diff is kept; a no is a reject, never an always.
        answer.store(false, std::sync::atomic::Ordering::SeqCst);
        let pick = futures::executor::block_on(
            sink.permission(question("t2", vec![diff(r"C:\work\b.txt")])),
        );
        assert_eq!(pick.as_deref(), Some("no"));
        assert_eq!(seen.lock().unwrap()[1].diffs, [diff("b.txt")]);
        // Another spelling of the same folder still counts; a folder beside it does not.
        assert_eq!(
            inside(&folder, r"c:/WORK/x/y.txt").as_deref(),
            Some("x/y.txt")
        );
        assert_eq!(
            inside(&folder, r"\\?\C:\work\z.txt").as_deref(),
            Some("z.txt")
        );
        assert_eq!(inside(&folder, r"C:\workshop\a.txt"), None);
        assert_eq!(inside(&folder, r"C:\work"), None);
        // An announced diff is used once.
        futures::executor::block_on(sink.permission(question("t1", Vec::new())));
        assert!(seen.lock().unwrap()[2].diffs.is_empty());
    }
}
