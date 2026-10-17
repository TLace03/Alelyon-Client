//! Schedules: a prompt the agent asks to run later, once or again and
//! again, and the wake-up of a conversation after a while. Not a port.
//!
//! - **Asked first:** `schedule_prompt` keeps running prompts after the turn
//!   that made it, so the reader says yes in the core's own dialog
//!   ([`crate::ports::ConfirmRequest::Schedule`]) before it is kept.
//! - **What runs:** at its time the prompt is sent with the model and the
//!   shown label of the turn that made it, either to that conversation
//!   (`same_topic`: a wake-up, which waits its turn when the conversation
//!   works) or to a new topic in the same folder and project, a task of the
//!   conversation's (it joins its group, and waits for a slot as a task's
//!   Start does).
//! - **While Lattice is open only:** a schedule due while it was closed does
//!   not run late; its conversation is told it was missed, and a repeating one
//!   goes on from its next time.
//! - **No timer at idle:** with nothing scheduled the runner waits for a
//!   schedule to be made, arming no timer; otherwise it sleeps until the
//!   earliest is due (woken early when one is made or cancelled), and while
//!   one is due but waits for a task slot it looks again every [`TICK`].
//! - **Bounds:** at most [`MAX_SCHEDULES`]; a first run 1 minute to 7 days
//!   away; a repeat every 5 minutes to 7 days. The store is
//!   `<native>/chat/schedules.json`, written atomically.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use lattice_protocol::Shown;
use lattice_protocol::conversation::{Accepted, ConversationEventKind, Mode, SendRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::agent::{Ask, Inner, lock};
use super::item::Item;
use super::turn::TurnTools;
use crate::state::StateRoot;
use crate::tools::read::{ToolError, parse_args};

/// The most schedules kept.
pub const MAX_SCHEDULES: usize = 20;
/// The longest wait, in minutes (7 days).
pub const MAX_MINUTES: u32 = 7 * 24 * 60;
/// The shortest repeat, in minutes.
pub const MIN_EVERY: u32 = 5;
/// How often the runner looks again while a due schedule waits for a slot.
pub const TICK: Duration = Duration::from_secs(20);
/// How late a run may start and still be on time (Lattice was busy).
const GRACE: f64 = 120.0;

/// One schedule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Schedule {
    pub id: String,
    /// The conversation that made it.
    pub conversation: String,
    pub title: String,
    pub prompt: String,
    pub choice: String,
    pub shown: Shown,
    pub mode: Mode,
    /// Again every this many minutes; `None`: once.
    #[serde(default)]
    pub every_minutes: Option<u32>,
    /// When it runs next, seconds since the epoch.
    pub next: f64,
    /// Sent to its own conversation (a wake-up), not to a new topic.
    #[serde(default)]
    pub same_topic: bool,
    pub created: f64,
    #[serde(default)]
    pub runs: u32,
}

#[derive(Default, Serialize, Deserialize)]
struct Stored {
    #[serde(default)]
    v: u32,
    #[serde(default)]
    schedules: Vec<Schedule>,
}

/// The store.
pub struct Schedules {
    path: PathBuf,
    lock: Mutex<()>,
    /// When this process started looking: what was due before it was missed.
    pub(crate) opened: Mutex<Option<f64>>,
    /// Told when a schedule is made or cancelled, so the runner sleeps anew.
    changed: Arc<tokio::sync::Notify>,
    /// Held while due schedules run: one look at a time, so none runs twice.
    looking: tokio::sync::Mutex<()>,
}

impl Schedules {
    pub fn new(state: &StateRoot) -> Self {
        Self {
            path: state.native_chat_dir().join("schedules.json"),
            lock: Mutex::new(()),
            opened: Mutex::default(),
            changed: Arc::default(),
            looking: tokio::sync::Mutex::new(()),
        }
    }

    fn read(&self) -> Result<Stored, String> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|_| "schedules.json is not a record Lattice can read, so it was left as it is.".to_owned()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Stored::default()),
            Err(_) => Err("Lattice could not read schedules.json.".to_owned()),
        }
    }

    fn change<T>(&self, edit: impl FnOnce(&mut Stored) -> Result<T, String>) -> Result<T, String> {
        let _held = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let mut stored = self.read()?;
        let out = edit(&mut stored)?;
        stored.v = 1;
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|_| "Lattice could not make its settings folder.".to_owned())?;
        }
        let bytes = serde_json::to_vec_pretty(&stored).unwrap_or_default();
        crate::fsx::atomic_write(&self.path, &bytes).map_err(|_| "Lattice could not save schedules.json.".to_owned())?;
        Ok(out)
    }

    /// Every schedule, soonest first.
    pub fn list(&self) -> Vec<Schedule> {
        let mut all = self.read().map(|s| s.schedules).unwrap_or_default();
        all.sort_by(|a, b| a.next.total_cmp(&b.next));
        all
    }

    pub fn add(&self, schedule: Schedule) -> Result<(), String> {
        self.change(|s| {
            if s.schedules.len() >= MAX_SCHEDULES {
                return Err(format!("{MAX_SCHEDULES} schedules are kept already: cancel one first."));
            }
            s.schedules.push(schedule);
            Ok(())
        })?;
        self.changed.notify_one();
        Ok(())
    }

    /// Cancel one; `false` when none has that id.
    pub fn cancel(&self, id: &str) -> Result<bool, String> {
        let gone = self.change(|s| {
            let before = s.schedules.len();
            s.schedules.retain(|x| x.id != id);
            Ok(s.schedules.len() != before)
        })?;
        self.changed.notify_one();
        Ok(gone)
    }

    /// After a run (or a miss) at `now`: a repeating schedule moves to its
    /// next time after `now`, a single one goes.
    fn advance(&self, id: &str, now: f64, ran: bool) -> Result<(), String> {
        self.change(|s| {
            if let Some(x) = s.schedules.iter_mut().find(|x| x.id == id) {
                if ran {
                    x.runs += 1;
                }
                match x.every_minutes {
                    Some(every) => {
                        let step = f64::from(every) * 60.0;
                        while x.next <= now {
                            x.next += step;
                        }
                    }
                    None => x.next = f64::NAN,
                }
            }
            s.schedules.retain(|x| !x.next.is_nan());
            Ok(())
        })
    }
}

/// When a schedule runs, in words.
pub fn when_words(first_minutes: u32, every: Option<u32>) -> String {
    let span = |m: u32| match m {
        m if m % (24 * 60) == 0 => format!("{} day{}", m / (24 * 60), if m == 24 * 60 { "" } else { "s" }),
        m if m % 60 == 0 => format!("{} hour{}", m / 60, if m == 60 { "" } else { "s" }),
        m => format!("{m} minute{}", if m == 1 { "" } else { "s" }),
    };
    match every {
        Some(e) => format!("in {}, then every {}", span(first_minutes), span(e)),
        None => format!("once, in {}", span(first_minutes)),
    }
}

#[derive(Deserialize)]
struct ScheduleArgs {
    title: String,
    prompt: String,
    #[serde(default)]
    in_minutes: Option<u32>,
    #[serde(default)]
    every_minutes: Option<u32>,
    #[serde(default)]
    same_topic: bool,
}

/// `schedule_prompt`, `list_schedules` and `cancel_schedule`.
pub(crate) async fn tool(tools: Arc<TurnTools>, name: &str, args: Value) -> Result<String, ToolError> {
    let inner = tools.inner.clone();
    if name == "schedule_prompt" && inner.plan().is_some_and(|plan| !plan.schedules) {
        return Err(ToolError::new(super::plan_limits::paid_only("Schedules")));
    }
    match name {
        "list_schedules" => {
            let all = inner.schedules.list();
            if all.is_empty() {
                return Ok("No schedules.".to_owned());
            }
            let now = inner.now();
            let mut out = String::from("Schedules, soonest first:");
            for s in all {
                let mine = if s.conversation == tools.convo.id { " (this conversation's)" } else { "" };
                let every = s.every_minutes.map(|e| format!(", every {e} minutes")).unwrap_or_default();
                out.push_str(&format!(
                    "\n- {} \u{201c}{}\u{201d}{mine}: next in {} minutes{every}{}",
                    s.id,
                    s.title,
                    ((s.next - now) / 60.0).max(0.0).round(),
                    if s.same_topic { ", here" } else { ", as a new topic" }
                ));
            }
            Ok(out)
        }
        "cancel_schedule" => {
            let id = args.get("id").and_then(Value::as_str).unwrap_or("").trim().to_owned();
            match inner.schedules.cancel(&id).map_err(ToolError::new)? {
                true => Ok(format!("{id} is cancelled.")),
                false => Err(ToolError::new("No schedule has that id: list_schedules names them.")),
            }
        }
        _ => {
            let args: ScheduleArgs = parse_args(&args)?;
            let (title, prompt) = (args.title.trim().to_owned(), args.prompt.trim().to_owned());
            if title.is_empty() || title.chars().count() > 80 || title.chars().any(char::is_control) {
                return Err(ToolError::new("A schedule's title is one line of 1 to 80 characters."));
            }
            if prompt.is_empty() || prompt.chars().count() > 2_000 {
                return Err(ToolError::new("A schedule's prompt is 1 to 2,000 characters."));
            }
            if let Some(e) = args.every_minutes
                && !(MIN_EVERY..=MAX_MINUTES).contains(&e)
            {
                return Err(ToolError::new(format!("A repeat is every {MIN_EVERY} minutes to 7 days.")));
            }
            let first = args.in_minutes.or(args.every_minutes).ok_or_else(|| ToolError::new("Say when: in_minutes, every_minutes, or both."))?;
            if !(1..=MAX_MINUTES).contains(&first) {
                return Err(ToolError::new("A first run is 1 minute to 7 days away."));
            }
            let Some(shown) = tools.shown.clone() else {
                return Err(ToolError::new("This turn cannot make a schedule."));
            };
            let choice = tools.convo.state().last_choice.clone().unwrap_or_default();
            let when = when_words(first, args.every_minutes);
            let allowed = inner
                .confirmer
                .ask(
                    &format!("schedule:{}:{}", tools.convo.id, tools.turn),
                    crate::ports::ConfirmRequest::Schedule {
                        title: title.clone(),
                        when: when.clone(),
                        prompt: prompt.chars().take(600).collect(),
                        here: args.same_topic,
                    },
                    crate::ports::Initiated::Page,
                )
                .await;
            if !allowed {
                return Err(ToolError::new("The user did not allow that schedule; nothing will run."));
            }
            let id = format!("sch_{}", &uuid::Uuid::new_v4().simple().to_string()[..10]);
            let now = inner.now();
            inner
                .schedules
                .add(Schedule {
                    id: id.clone(),
                    conversation: tools.convo.id.clone(),
                    title: crate::secrets::redact(&title),
                    prompt: crate::secrets::redact(&prompt),
                    choice,
                    shown,
                    mode: tools.mode,
                    every_minutes: args.every_minutes,
                    next: now + f64::from(first) * 60.0,
                    same_topic: args.same_topic,
                    created: now,
                    runs: 0,
                })
                .map_err(ToolError::new)?;
            Ok(format!(
                "Scheduled as {id}: \u{201c}{title}\u{201d}, {when}{}. It runs only while Alelyon is open; cancel_schedule ends it.",
                if args.same_topic { ", in this conversation" } else { ", as a new topic" }
            ))
        }
    }
}

/// Run what is due while the chat service lives: no timer while nothing is
/// scheduled; a sleep until the earliest is due otherwise.
pub(crate) fn start_runner(inner: &Arc<Inner>) {
    let weak: Weak<Inner> = Arc::downgrade(inner);
    *lock(&inner.schedules.opened) = Some(inner.now());
    let changed = inner.schedules.changed.clone();
    inner.handle.spawn(async move {
        loop {
            let wait = {
                let Some(inner) = weak.upgrade() else { return };
                tick(&inner).await;
                let now = inner.now();
                inner.schedules.list().first().map(|s| s.next - now)
            };
            match wait {
                // Nothing scheduled: wait for a schedule to be made.
                None => changed.notified().await,
                Some(secs) => {
                    // Due already, waiting for a task slot: look again after a while.
                    let pause = if secs > 0.0 { Duration::from_secs_f64(secs.min(f64::from(MAX_MINUTES) * 60.0)) } else { TICK };
                    tokio::select! {
                        () = tokio::time::sleep(pause) => {}
                        () = changed.notified() => {}
                    }
                }
            }
        }
    });
}

/// Run what is due now; tell each conversation of what was due while
/// Lattice was closed.
pub(crate) async fn tick(inner: &Arc<Inner>) {
    let _one_at_a_time = inner.schedules.looking.lock().await;
    let now = inner.now();
    let opened = lock(&inner.schedules.opened).unwrap_or(now);
    for s in inner.schedules.list().into_iter().filter(|s| s.next <= now) {
        if s.next < opened - GRACE {
            note(inner, &s.conversation, &format!(
                "The schedule \u{201c}{}\u{201d} was due while Alelyon was closed, so it did not run{}.",
                s.title,
                if s.every_minutes.is_some() { "; it goes on from its next time" } else { "" }
            ));
            let _ = inner.schedules.advance(&s.id, now, false);
            continue;
        }
        // Without a plan that has schedules, a due schedule waits: it is told, and goes on from its next time.
        if inner.plan().is_some_and(|plan| !plan.schedules) {
            note(inner, &s.conversation, &format!(
                "The schedule \u{201c}{}\u{201d} did not run: {}",
                s.title,
                super::plan_limits::paid_only("Schedules")
            ));
            let _ = inner.schedules.advance(&s.id, now, false);
            continue;
        }
        if run(inner, &s).await {
            let _ = inner.schedules.advance(&s.id, now, true);
        }
    }
}

/// Write a notice in a conversation.
fn note(inner: &Arc<Inner>, conversation: &str, text: &str) {
    if let Ok(convo) = inner.load(conversation) {
        convo.record(&Item::Notice { text: text.to_owned(), at: inner.now() });
        convo.log.push(ConversationEventKind::Notice { text: text.to_owned() });
    }
}

/// Send a schedule's prompt; `false` when it waits for a slot (a new topic
/// past the limit) and is tried again at the next tick.
async fn run(inner: &Arc<Inner>, s: &Schedule) -> bool {
    let Ok(origin) = inner.load(&s.conversation) else {
        // Its conversation is gone: the schedule goes with it.
        let _ = inner.schedules.cancel(&s.id);
        return false;
    };
    let workspace = origin.state().workspace.as_ref().map(|w| w.id.clone());
    if !s.same_topic {
        let running = lock(&inner.convos).values().filter(|c| c.state().running.is_some()).count();
        if !inner.task_limit().allows(running) {
            return false;
        }
    }
    let request = SendRequest {
        conversation: s.same_topic.then(|| s.conversation.clone()),
        text: s.prompt.clone(),
        choice: s.choice.clone(),
        shown: s.shown.clone(),
        mode: s.mode,
        workspace,
        edit_of: None,
        project: if s.same_topic { None } else { inner.projects.of_chat(&s.conversation).map(|p| p.id) },
        images: Vec::new(),
    };
    match super::agent::start(inner.clone(), Ask::Send(request)).await {
        Ok(Accepted::Started { conversation, .. }) if !s.same_topic => {
            let _ = inner.task_runs.started(
                &conversation.id,
                super::task_runs::Origin { from: s.conversation.clone(), title: format!("Scheduled: {}", s.title), at: inner.now(), ..Default::default() },
            );
            let _ = inner.groups.follow(&conversation.id, &s.conversation, inner.now());
            true
        }
        Ok(_) => true,
        Err(refusal) => {
            note(inner, &s.conversation, &format!("The schedule \u{201c}{}\u{201d} could not run: {}", s.title, refusal.message));
            true
        }
    }
}
