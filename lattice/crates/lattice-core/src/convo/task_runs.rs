//! Tasks run beside the chat: on one chat surface, a task the reader starts
//! from a chip runs as a topic of its own in the background while the chat
//! goes on, and the reader sees it on the activity chart.
//!
//! - **How many at once:** at most the reader's limit of tasks run at once
//!   (default [`DEFAULT_PARALLEL`]); the reader may lift it ("No limit"). Each
//!   task runs its own model turns, so the limit bounds how much computing (or
//!   a provider's charges) tasks take at once. A Start past the limit is
//!   refused with a sentence that says so, and the task still waits on its
//!   chip. The topic the task was started from does not count.
//! - **Where it came from:** each task's topic is recorded with the topic it
//!   was started from and the task's title, so the activity chart can draw
//!   which topic each running task belongs to.
//! - **The store** is `<native>/chat/tasks.json`, written atomically; one
//!   Lattice cannot read is never written over.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::state::StateRoot;

/// The tasks that run at once until the reader picks another limit.
pub const DEFAULT_PARALLEL: u32 = 3;
/// The highest limit the reader may pick short of "No limit".
pub const MAX_PARALLEL: u32 = 16;
/// The most origins kept; the oldest go first.
const MAX_ORIGINS: usize = 200;

/// Where a task's topic came from.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Origin {
    /// The topic it was started from.
    pub from: String,
    /// The task's title.
    pub title: String,
    pub at: f64,
    /// The agent moved the work here from `from`, with the reader's word (`super::switch`): the window follows it.
    #[serde(default)]
    pub switch: bool,
    /// The project it began in, when it was moved to one.
    #[serde(default)]
    pub project: Option<String>,
}

/// How many tasks may run at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Limit {
    At(u32),
    None,
}

impl Default for Limit {
    fn default() -> Self {
        Limit::At(DEFAULT_PARALLEL)
    }
}

impl Limit {
    /// Whether one more task may start while `running` run.
    pub fn allows(self, running: usize) -> bool {
        match self {
            Limit::At(n) => running < n as usize,
            Limit::None => true,
        }
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Stored {
    #[serde(default)]
    v: u32,
    #[serde(default)]
    limit: Limit,
    /// Each task's topic, by its id.
    #[serde(default)]
    origins: HashMap<String, Origin>,
}

/// The store.
pub struct TaskRuns {
    path: PathBuf,
    lock: Mutex<()>,
}

impl TaskRuns {
    pub fn new(state: &StateRoot) -> Self {
        Self { path: state.native_chat_dir().join("tasks.json"), lock: Mutex::new(()) }
    }

    fn read(&self) -> Result<Stored, String> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|_| "The tasks' settings could not be read.".to_owned()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Stored::default()),
            Err(_) => Err("The tasks' settings could not be read.".to_owned()),
        }
    }

    fn change(&self, edit: impl FnOnce(&mut Stored)) -> Result<(), String> {
        let _held = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let mut stored = self.read()?;
        edit(&mut stored);
        stored.v = 1;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|_| "Lattice could not make its settings folder.".to_owned())?;
        }
        let bytes = serde_json::to_vec_pretty(&stored).unwrap_or_default();
        crate::fsx::atomic_write(&self.path, &bytes).map_err(|_| "The tasks' settings could not be saved.".to_owned())
    }

    /// The limit; the default when the store cannot be read.
    pub fn limit(&self) -> Limit {
        self.read().map(|s| s.limit).unwrap_or_default()
    }

    pub fn set_limit(&self, limit: Limit) -> Result<(), String> {
        if let Limit::At(n) = limit
            && !(1..=MAX_PARALLEL).contains(&n)
        {
            return Err(format!("Pick from 1 to {MAX_PARALLEL} tasks at once, or no limit."));
        }
        self.change(|s| s.limit = limit)
    }

    pub fn origins(&self) -> HashMap<String, Origin> {
        self.read().map(|s| s.origins).unwrap_or_default()
    }

    /// Record that the task `title`, started from `from`, runs as `topic`.
    pub fn started(&self, topic: &str, origin: Origin) -> Result<(), String> {
        self.change(|s| {
            s.origins.insert(topic.to_owned(), origin);
            while s.origins.len() > MAX_ORIGINS {
                let oldest = s.origins.iter().min_by(|a, b| a.1.at.total_cmp(&b.1.at)).map(|(k, _)| k.clone());
                match oldest {
                    Some(k) => s.origins.remove(&k),
                    None => break,
                };
            }
        })
    }
}

/// The sentence for a Start past `inner`'s limit now (none without one).
pub(crate) fn full_of(inner: &super::agent::Inner) -> String {
    match inner.task_limit() {
        Limit::At(n) if super::plan_limits::plan_holds(inner.task_runs.limit(), inner.plan()) => full_plan(n),
        Limit::At(n) => full(n),
        Limit::None => String::new(),
    }
}

/// The reader's sentence for a Start past what their plan allows.
pub fn full_plan(limit: u32) -> String {
    let running = if limit == 1 { "1 task is running".to_owned() } else { format!("{limit} tasks are running") };
    format!("{running}, the most your plan allows at once, so this one is queued: it starts when one ends, the highest-priority group's first. A paid plan runs more at once (Settings, under Account).")
}

/// The reader's sentence for a Start past the limit.
pub fn full(limit: u32) -> String {
    let running = if limit == 1 { "1 task is running".to_owned() } else { format!("{limit} tasks are running") };
    format!("{running}, the most your limit allows, so this one is queued: it starts when one ends, the highest-priority group's first. You can raise the limit on the Tools page.")
}
