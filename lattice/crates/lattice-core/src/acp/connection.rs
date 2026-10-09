//! One agent's connection over the Agent Client Protocol: JSON-RPC 2.0, one
//! message per line, over the agent's stdin and stdout. Not a port; the
//! protocol is ACP's, and the peer is Lattice's own, modelled on the MCP
//! client (`crate::mcp::client`) and sharing its message code
//! (`crate::mcp::jsonrpc`), so no crate is added.
//!
//! Unlike an MCP server, an ACP agent asks its client things: whether it may
//! act (`session/request_permission`), and to read or write a file
//! (`fs/read_text_file`, `fs/write_text_file`). Each such request goes to the
//! holder's [`Asks`] as an [`Ask`], which must be answered once, at any time,
//! from any thread; an [`Ask`] dropped unanswered is answered with an error,
//! so the agent never waits on a request nobody holds.
//!
//! Two threads, each blocked on its pipe or channel when nothing happens (so
//! an idle agent costs no wake-ups): the **reader** reads one line at a time
//! (at most [`MAX_LINE`]), hands responses to the request waiting for them,
//! requests to [`Asks`] and notifications to [`Events`], and logs output that
//! is not a message without obeying it; the **writer** writes each line to the
//! agent's stdin. Closing the connection ends the writer, which closes the
//! agent's stdin. When the agent's output ends, every waiting request ends.
//!
//! Everything an agent sends is untrusted: its words are shown, never obeyed.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::oneshot;

use crate::mcp::jsonrpc::{self, Incoming, MAX_LINE, RpcError};

/// The error code of a request Lattice refused or could not answer.
pub const REFUSED: i64 = -32000;

/// What the connection tells its owner, from its reader thread.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// A notification from the agent (`session/update`, for one).
    Notification { method: String, params: Value },
    /// Output that is not a message, logged, never obeyed.
    Log(String),
    /// The output ended or broke; no event comes after this.
    Closed(String),
}

/// Where a connection's events go.
pub type Events = Arc<dyn Fn(Event) + Send + Sync>;

/// Where the agent's requests go.
pub type Asks = Arc<dyn Fn(Ask) + Send + Sync>;

/// Why a request of Lattice's has no result.
#[derive(Clone, Debug, PartialEq)]
pub enum CallError {
    Rpc(RpcError),
    TimedOut,
    Closed(String),
}

impl CallError {
    /// One sentence for the reader.
    pub fn sentence(&self) -> String {
        match self {
            CallError::Rpc(error) => format!("The agent answered: {}", cut(&error.message, 400)),
            CallError::TimedOut => "The agent did not answer in time.".to_owned(),
            CallError::Closed(why) => why.clone(),
        }
    }
}

/// `text` cut to at most `chars` characters.
pub fn cut(text: &str, chars: usize) -> String {
    text.chars().take(chars).collect()
}

enum Out {
    Line(String),
    Close,
}

type Answer = oneshot::Sender<Result<Value, RpcError>>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct Shared {
    pending: Mutex<HashMap<u64, Answer>>,
    closed: Mutex<Option<String>>,
    out: mpsc::Sender<Out>,
}

impl Shared {
    fn close(&self, why: &str) {
        {
            let mut closed = lock(&self.closed);
            if closed.is_none() {
                *closed = Some(why.to_owned());
            }
        }
        let waiting: Vec<Answer> = lock(&self.pending)
            .drain()
            .map(|(_, sender)| sender)
            .collect();
        drop(waiting);
        let _ = self.out.send(Out::Close);
    }
}

/// A request the agent made of Lattice, answered once with [`Ask::answer`] or
/// [`Ask::refuse`]; dropped unanswered, it is refused.
pub struct Ask {
    pub method: String,
    pub params: Value,
    id: Value,
    out: mpsc::Sender<Out>,
    answered: bool,
}

impl Ask {
    /// Answer it with `result`.
    pub fn answer(mut self, result: Value) {
        self.answered = true;
        let _ = self.out.send(Out::Line(jsonrpc::result(&self.id, result)));
    }

    /// Refuse it, saying why in one sentence.
    pub fn refuse(mut self, why: &str) {
        self.answered = true;
        let _ = self
            .out
            .send(Out::Line(jsonrpc::error(&self.id, REFUSED, why)));
    }
}

impl Drop for Ask {
    fn drop(&mut self) {
        if !self.answered {
            let _ = self.out.send(Out::Line(jsonrpc::error(
                &self.id,
                REFUSED,
                "Lattice did not answer that.",
            )));
        }
    }
}

impl std::fmt::Debug for Ask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ask").field("method", &self.method).finish()
    }
}

/// Read up to the next `\n` (not kept) into `line`; `None` at the end, and
/// `Some(false)` for a line past [`MAX_LINE`].
fn read_line(reader: &mut impl BufRead, line: &mut Vec<u8>) -> std::io::Result<Option<bool>> {
    loop {
        let (done, used) = {
            let available = match reader.fill_buf() {
                Ok(available) => available,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            };
            if available.is_empty() {
                return Ok(if line.is_empty() { None } else { Some(true) });
            }
            match available.iter().position(|byte| *byte == b'\n') {
                Some(at) => {
                    line.extend_from_slice(&available[..at]);
                    (true, at + 1)
                }
                None => {
                    line.extend_from_slice(available);
                    (false, available.len())
                }
            }
        };
        reader.consume(used);
        if line.len() > MAX_LINE {
            return Ok(Some(false));
        }
        if done {
            return Ok(Some(true));
        }
    }
}

fn read_loop(reader: Box<dyn Read + Send>, shared: Arc<Shared>, events: Events, asks: Asks) {
    let mut reader = BufReader::with_capacity(64 * 1024, reader);
    let mut line = Vec::new();
    let why = loop {
        line.clear();
        match read_line(&mut reader, &mut line) {
            Ok(Some(true)) => {}
            Ok(None) => break "The agent stopped.".to_owned(),
            Ok(Some(false)) => {
                break "The agent sent a message larger than 8 MiB, so Lattice stopped reading it."
                    .to_owned();
            }
            Err(_) => break "The agent's output could not be read.".to_owned(),
        }
        let text = String::from_utf8_lossy(&line);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        for incoming in jsonrpc::parse(text) {
            match incoming {
                Incoming::Response { id, outcome } => {
                    let sender = id.as_u64().and_then(|id| lock(&shared.pending).remove(&id));
                    if let Some(sender) = sender {
                        let _ = sender.send(outcome);
                    }
                }
                Incoming::Request { id, method, params } => asks(Ask {
                    method,
                    params,
                    id,
                    out: shared.out.clone(),
                    answered: false,
                }),
                Incoming::Notification { method, params } => {
                    events(Event::Notification { method, params })
                }
                Incoming::Invalid(_) => events(Event::Log(format!(
                    "(not an ACP message) {}",
                    cut(text, 300)
                ))),
            }
        }
    };
    shared.close(&why);
    events(Event::Closed(why));
}

fn write_loop(mut writer: Box<dyn Write + Send>, lines: mpsc::Receiver<Out>) {
    for out in lines {
        match out {
            Out::Line(line) => {
                if writer
                    .write_all(line.as_bytes())
                    .and_then(|()| writer.flush())
                    .is_err()
                {
                    break;
                }
            }
            Out::Close => break,
        }
    }
}

/// One agent's connection.
pub struct Connection {
    shared: Arc<Shared>,
    next: AtomicU64,
}

impl Connection {
    /// Start the reader and the writer over the agent's stdout (`reader`) and
    /// stdin (`writer`); `name` names their threads.
    pub fn start(
        name: &str,
        reader: Box<dyn Read + Send>,
        writer: Box<dyn Write + Send>,
        events: Events,
        asks: Asks,
    ) -> std::io::Result<Arc<Self>> {
        let (out, lines) = mpsc::channel();
        let shared = Arc::new(Shared {
            pending: Mutex::default(),
            closed: Mutex::default(),
            out,
        });
        let short = cut(name, 24);
        std::thread::Builder::new()
            .name(format!("lattice-acp-in {short}"))
            .spawn(move || write_loop(writer, lines))?;
        let reading = shared.clone();
        std::thread::Builder::new()
            .name(format!("lattice-acp-out {short}"))
            .spawn(move || read_loop(reader, reading, events, asks))?;
        Ok(Arc::new(Self {
            shared,
            next: AtomicU64::new(1),
        }))
    }

    /// Why the connection is closed, once it is.
    pub fn closed(&self) -> Option<String> {
        lock(&self.shared.closed).clone()
    }

    /// Close it: waiting requests end, and the agent's stdin ends.
    pub fn close(&self) {
        self.shared.close("Lattice closed the connection.");
    }

    /// Send a notification (nothing comes back).
    pub fn notify(&self, method: &str, params: Value) {
        let _ = self
            .shared
            .out
            .send(Out::Line(jsonrpc::notification(method, Some(params))));
    }

    /// Send a request and wait at most `timeout` for its answer. A request
    /// given up is forgotten here; ACP withdraws work with `session/cancel`.
    pub async fn request(
        &self,
        method: &str,
        params: Value,
        timeout: Duration,
    ) -> Result<Value, CallError> {
        let gone = || {
            self.closed()
                .unwrap_or_else(|| "The agent stopped.".to_owned())
        };
        if self.closed().is_some() {
            return Err(CallError::Closed(gone()));
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (sender, answer) = oneshot::channel();
        lock(&self.shared.pending).insert(id, sender);
        if self
            .shared
            .out
            .send(Out::Line(jsonrpc::request(id, method, params)))
            .is_err()
        {
            lock(&self.shared.pending).remove(&id);
            return Err(CallError::Closed(gone()));
        }
        let result = match tokio::time::timeout(timeout, answer).await {
            Ok(Ok(outcome)) => outcome.map_err(CallError::Rpc),
            Ok(Err(_)) => Err(CallError::Closed(gone())),
            Err(_) => Err(CallError::TimedOut),
        };
        lock(&self.shared.pending).remove(&id);
        result
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        let _ = self.shared.out.send(Out::Close);
    }
}
