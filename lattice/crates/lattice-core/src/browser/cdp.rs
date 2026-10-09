//! The Chrome DevTools Protocol over either transport: [`super::pipe`], the
//! two pipes of the browser Lattice starts (`--remote-debugging-pipe`), or
//! [`super::ws`], a WebSocket to a browser whose DevTools listen on a
//! loopback port. One connection, its requests answered by id and its events
//! handed on. Not a port; the protocol is Chromium's (Edge and Chrome speak
//! it), the same messages on either transport.
//!
//! A request is `{"id", "method", "params", "sessionId"?}`; its answer has
//! the same `id` and a `result` or an `error`. An event has a `method` and no
//! `id`. A request waits at most its timeout; the connection's end ends every
//! waiting request with the reason.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use lattice_sys::process::FdPipes;
use serde_json::{Value, json};
use tokio::sync::oneshot;

use super::pipe::{self, Pipe};
use super::ws::{self, Socket};

type Answer = oneshot::Sender<Result<Value, String>>;

/// An event from the browser: its method, params and session.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub method: String,
    pub params: Value,
    pub session: Option<String>,
}

/// Where the connection's events go (on its reader thread).
pub type Events = Arc<dyn Fn(Event) + Send + Sync>;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct Shared {
    pending: Mutex<HashMap<u64, Answer>>,
    closed: Mutex<Option<String>>,
}

/// Where a browser's DevTools are.
#[derive(Debug)]
pub enum Endpoint {
    /// This process's ends of the browser's descriptors 3 and 4: a browser
    /// started with `--remote-debugging-pipe` by
    /// `lattice_sys::process::spawn_with_fd_pipes`, as Lattice's own is.
    Pipe(FdPipes),
    /// `ws://127.0.0.1:<port><path>`: a browser whose DevTools listen on a
    /// loopback port (`--remote-debugging-port`).
    WebSocket { port: u16, path: String },
}

/// The open transport.
enum Link {
    Pipe(Pipe),
    Socket(Socket),
}

impl Link {
    fn send(&self, text: String) -> bool {
        match self {
            Self::Pipe(pipe) => pipe.send(text),
            Self::Socket(socket) => socket.send(text),
        }
    }

    fn close(&self) {
        match self {
            Self::Pipe(pipe) => pipe.close(),
            Self::Socket(socket) => socket.close(),
        }
    }
}

/// One DevTools connection.
pub struct Cdp {
    link: Link,
    shared: Arc<Shared>,
    next: AtomicU64,
}

fn dispatch(shared: &Shared, events: &Events, text: String) {
    let Ok(message) = serde_json::from_str::<Value>(&text) else {
        return;
    };
    if let Some(id) = message.get("id").and_then(Value::as_u64) {
        let sender = lock(&shared.pending).remove(&id);
        if let Some(sender) = sender {
            let outcome = match message.get("error") {
                Some(error) => Err(error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("the browser refused that")
                    .to_owned()),
                None => Ok(message.get("result").cloned().unwrap_or(Value::Null)),
            };
            let _ = sender.send(outcome);
        }
        return;
    }
    if let Some(method) = message.get("method").and_then(Value::as_str) {
        events(Event {
            method: method.to_owned(),
            params: message.get("params").cloned().unwrap_or(Value::Null),
            session: message
                .get("sessionId")
                .and_then(Value::as_str)
                .map(str::to_owned),
        });
    }
}

impl Cdp {
    /// Connect to a browser's DevTools at `endpoint`.
    pub fn connect(endpoint: Endpoint, events: Events) -> Result<Arc<Self>, String> {
        let shared = Arc::new(Shared {
            pending: Mutex::default(),
            closed: Mutex::default(),
        });
        let (reading, ending) = (shared.clone(), shared.clone());
        let ended_events = events.clone();
        let on_text: Box<dyn Fn(String) + Send> =
            Box::new(move |text| dispatch(&reading, &events, text));
        let on_end: Box<dyn FnOnce(String) + Send> = Box::new(move |why: String| {
            *lock(&ending.closed) = Some(why.clone());
            // Dropping the senders ends every waiting request.
            lock(&ending.pending).clear();
            ended_events(Event {
                method: "Lattice.closed".to_owned(),
                params: json!({"reason": why}),
                session: None,
            });
        });
        let link = match endpoint {
            Endpoint::Pipe(pipes) => Link::Pipe(pipe::connect(
                pipes.to_child,
                pipes.from_child,
                on_text,
                on_end,
            )?),
            Endpoint::WebSocket { port, path } => {
                Link::Socket(ws::connect(port, &path, on_text, on_end)?)
            }
        };
        Ok(Arc::new(Self {
            link,
            shared,
            next: AtomicU64::new(1),
        }))
    }

    /// Why the connection ended, once it has.
    pub fn closed(&self) -> Option<String> {
        lock(&self.shared.closed).clone()
    }

    /// Close it. Over the pipes, this ends the browser's DevTools, and
    /// Chromium then closes the browser.
    pub fn close(&self) {
        self.link.close();
    }

    /// Send `method` (to the page `session`, or the browser) and wait at most
    /// `timeout` for its result.
    pub async fn call(
        &self,
        method: &str,
        params: Value,
        session: Option<&str>,
        timeout: Duration,
    ) -> Result<Value, String> {
        if let Some(why) = self.closed() {
            return Err(why);
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (sender, answer) = oneshot::channel();
        lock(&self.shared.pending).insert(id, sender);
        let mut message = json!({"id": id, "method": method, "params": params});
        if let Some(session) = session {
            message["sessionId"] = json!(session);
        }
        if !self.link.send(message.to_string()) {
            lock(&self.shared.pending).remove(&id);
            return Err(self
                .closed()
                .unwrap_or_else(|| "The browser's connection is closed.".to_owned()));
        }
        match tokio::time::timeout(timeout, answer).await {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(_)) => Err(self
                .closed()
                .unwrap_or_else(|| "The browser's connection is closed.".to_owned())),
            Err(_) => {
                lock(&self.shared.pending).remove(&id);
                Err(format!("The browser did not answer {method} in time."))
            }
        }
    }
}

impl Drop for Cdp {
    fn drop(&mut self) {
        self.link.close();
    }
}
