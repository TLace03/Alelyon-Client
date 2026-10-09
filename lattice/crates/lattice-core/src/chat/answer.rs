//! One plain turn's answer (the native chat's spec §3.3.5 "The answer task", §2.7;
//! row C8 of the chat core's spec, with Local and Auto on the managed
//! llama.cpp server, §22).
//!
//! Every path ends with a saved turn (or the D8 refusal to save) and its
//! events; `Done` follows when the job is marked finished:
//! 1. `Stage(Thinking)`.
//! 2. The target Prepare resolved, never resolved again (N10):
//!    - **Refused**: the error turn "That model is not ready. <reason> Pick
//!      another, or change it in the Lattice desktop window under System ›
//!      Models.";
//!    - **the managed server** (Local, Auto): started or reused through the
//!      runtime; a server that cannot start gives the error turn with its
//!      explicit offline sentence, and nothing is sent anywhere (LF1). A lease
//!      holds it in use while the answer streams, so no idle stop interrupts
//!      it (LR7). Its client is built by `models::build_managed`: local, no
//!      proxy, its launch token (N6, LR2);
//!    - **an endpoint** picked by name: first the egress backstop (N4): every
//!      non-system text of the request is checked again, and on a match
//!      nothing is sent and the N4 error turn is saved. Then
//!      `models::build`;
//!    - **the echo**: the web's development reply, in pieces 12 ms apart.
//! 3. The request is `prompt::messages(question, recent)`, temperature 0.2
//!    and no tools.
//! 4. Streaming, biased toward Stop: text goes through `ThinkFilter`, so no
//!    scratchpad is ever shown; reasoning deltas are dropped; `Done` gives
//!    the usage and any refusal.
//! 5. The outcome follows §2.7 and the chat spec's table; the prose is
//!    `strip_think(strip(raw))`, and figures no tool backs are listed
//!    (`grounding`), except for the echo, as the web does.
//! 6. `touch`, then `append_answer`: a thread deleted or archived meanwhile
//!    is not recreated (D8).
//!
//! Nothing here logs, and no error carries transport text.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use lattice_agents::ModelError;
use lattice_agents::model::{
    InputItem, Model, ModelEvent, ModelRequest, ModelSettings, OutputItem,
};
use lattice_protocol::chat::{ChatEventKind, Stage};
use tokio::runtime::Handle;

use super::echo;
use super::grounding;
use super::jobs::Job;
use super::prompt;
use super::store::StoredTurn;
use super::think::{ThinkFilter, strip_think};
use super::transcript::{NewTurn, StoreError, TranscriptStore};
use super::vocab::{Resolution, Target};
use crate::clock::Clock;
use crate::env::Env;
use crate::keys::KeyStore;
use crate::llama::{Lease, ManagedRuntime};
use crate::models::{self, ModelFactory};
use crate::py;
use crate::secrets::looks_like_secret;

/// Python's `temperature` for every chat provider.
pub const TEMPERATURE: f64 = 0.2;
pub const NOTHING_WRITTEN: &str = "stopped before the model had written anything";
pub const EMPTY_ANSWER: &str = "the model returned an empty answer";
pub const NOT_SAVED: &str = "The answer arrived but could not be saved to this conversation.";
pub const GONE: &str =
    "This chat was deleted while the answer was being written; the answer was not saved.";
pub const STOPPED_UNEXPECTEDLY: &str = "The answer stopped unexpectedly.";

/// The N4 sentence for a target off this machine.
pub fn secret_sentence(label: &str) -> String {
    format!(
        "Not sent: this conversation contains text that looks like a key or password, and {label} is off this machine. Pick a model on this machine."
    )
}

/// The error turn for a choice that cannot answer.
pub fn not_ready_sentence(reason: &str) -> String {
    format!(
        "That model is not ready. {reason} Pick another, or change it in the Lattice desktop window under System \u{203a} Models."
    )
}

/// What an answer needs from the core.
pub(crate) struct Context {
    pub store: Arc<dyn TranscriptStore>,
    pub local: Arc<dyn ManagedRuntime>,
    pub env: Arc<dyn Env>,
    pub keys: KeyStore,
    pub factory: Option<ModelFactory>,
    pub handle: Handle,
    pub clock: Clock,
    pub echo_gap: Duration,
    /// The labs' agents' sessions (`crate::acp`), when this window runs them.
    pub agents: Option<Arc<crate::acp::pool::Pool>>,
}

/// What Prepare decided for one answer.
pub(crate) struct Turn {
    pub thread: String,
    pub question: String,
    /// The last visible turns, ending with the question (§2.3).
    pub recent: Vec<StoredTurn>,
    pub resolution: Resolution,
}

/// How an answer ended, before it is saved.
#[derive(Default)]
struct Outcome {
    text: String,
    error: String,
    truncated: bool,
    cancelled: bool,
    prompt_tokens: Option<i64>,
    completion_tokens: Option<i64>,
    /// Ground the prose (not for the echo, as the web does not).
    ground: bool,
}

impl Outcome {
    fn error(sentence: impl Into<String>) -> Self {
        Self {
            error: sentence.into(),
            ..Self::default()
        }
    }
}

/// Text that would leave the machine with this request.
fn request_texts(request: &ModelRequest) -> impl Iterator<Item = &str> {
    request.input.iter().filter_map(|item| match item {
        InputItem::User(text) | InputItem::UserImages { text, .. } => Some(text.as_str()),
        InputItem::Assistant { text, .. } => text.as_deref(),
        InputItem::ToolResult { output, .. } => Some(output.as_str()),
    })
}

/// Run one answer and save it. Never panics on its own account; a panic in
/// it is the supervisor's (`chat::mod`).
pub(crate) async fn answer(ctx: Arc<Context>, job: Arc<Job>, turn: Turn) {
    job.push(ChatEventKind::Stage {
        stage: Stage::Thinking,
        detail: "thinking".into(),
    });
    let outcome = produce(&ctx, &job, &turn).await;
    save(&ctx, &job, &turn, outcome).await;
}

async fn produce(ctx: &Context, job: &Job, turn: &Turn) -> Outcome {
    let messages = prompt::messages(&turn.question, &turn.recent);
    let request = ModelRequest {
        system: messages.system,
        input: messages.input,
        tools: Vec::new(),
        settings: ModelSettings {
            temperature: Some(TEMPERATURE),
            ..ModelSettings::default()
        },
    };
    match &turn.resolution.target {
        Target::Refused(reason) => Outcome::error(not_ready_sentence(reason)),
        Target::Echo => echo_answer(ctx, job, &turn.question).await,
        // One of the labs' agents: a prompt to the conversation's own session
        // with it; it runs off this machine, as an endpoint does.
        Target::Agent(agent) => {
            if request_texts(&request).any(looks_like_secret) {
                return Outcome::error(secret_sentence(&turn.resolution.shown.label));
            }
            match &ctx.agents {
                Some(pool) => {
                    let started = tokio::select! {
                        biased;
                        () = job.stopped() => return stopped_before_anything(),
                        started = pool.ready(*agent, &turn.thread) => started,
                    };
                    if let Err(why) = started {
                        return Outcome::error(why);
                    }
                    stream(job, pool.model(*agent, &turn.thread), request, None, Lease::detached()).await
                }
                None => Outcome::error(format!("{} is not available in this window.", agent.label())),
            }
        }
        Target::Managed(model) => {
            let opened = tokio::select! {
                biased;
                () = job.stopped() => return stopped_before_anything(),
                opened = ctx.local.open(model.clone()) => opened,
            };
            let opened = match opened {
                Ok(opened) => opened,
                Err(error) => return Outcome::error(error.sentence()),
            };
            match models::build_managed(&opened.endpoint, ctx.factory.as_ref()) {
                Ok(built) => {
                    stream(
                        job,
                        built.model,
                        request,
                        Some(built.base_url),
                        opened.lease,
                    )
                    .await
                }
                Err(refusal) => Outcome::error(refusal.message),
            }
        }
        Target::Endpoint(endpoint) => {
            // N4's backstop: another process may have appended a turn since
            // Prepare checked; nothing leaves when any of it looks like a key.
            if !turn.resolution.affirmatively_local()
                && request_texts(&request).any(looks_like_secret)
            {
                return Outcome::error(secret_sentence(&turn.resolution.shown.label));
            }
            match models::build(endpoint, ctx.env.as_ref(), &ctx.keys, ctx.factory.as_ref()) {
                Ok(built) => {
                    stream(
                        job,
                        built.model,
                        request,
                        Some(built.base_url),
                        Lease::detached(),
                    )
                    .await
                }
                Err(refusal) => Outcome::error(refusal.message),
            }
        }
    }
}

fn stopped_before_anything() -> Outcome {
    Outcome {
        error: NOTHING_WRITTEN.into(),
        cancelled: true,
        ..Outcome::default()
    }
}

/// Stream the model, filtered, until it ends, fails or is stopped. `lease`
/// keeps the managed server in use until the stream is done.
async fn stream(
    job: &Job,
    model: Arc<dyn Model>,
    request: ModelRequest,
    base_url: Option<String>,
    lease: Lease,
) -> Outcome {
    let mut events = model.stream(request);
    let mut filter = ThinkFilter::new();
    let mut raw = String::new();
    let mut writing = false;
    let mut usage = None;
    let mut refusal: Option<String> = None;
    let mut error: Option<ModelError> = None;
    let mut cancelled = false;
    let mut emit = |visible: String| {
        if visible.is_empty() {
            return;
        }
        if !writing {
            writing = true;
            job.push(ChatEventKind::Stage {
                stage: Stage::Writing,
                detail: "writing the answer".into(),
            });
        }
        job.push(ChatEventKind::Delta { text: visible });
    };
    let mut stopped = Box::pin(job.stopped());
    loop {
        tokio::select! {
            biased;
            () = &mut stopped => {
                cancelled = true;
                break;
            }
            event = events.next() => match event {
                Some(Ok(ModelEvent::TextDelta(piece))) => {
                    raw.push_str(&piece);
                    emit(filter.feed(&piece));
                }
                Some(Ok(ModelEvent::ReasoningDelta(_))) => {}
                Some(Ok(ModelEvent::Done(response))) => {
                    usage = response.usage;
                    for item in response.output {
                        match item {
                            OutputItem::Refusal { text } if refusal.is_none() => refusal = Some(text),
                            // A model that sent its text only whole.
                            OutputItem::Message { text } if raw.is_empty() => {
                                raw.push_str(&text);
                                emit(filter.feed(&text));
                            }
                            _ => {}
                        }
                    }
                    break;
                }
                Some(Err(failure)) => {
                    error = Some(failure);
                    break;
                }
                None => {
                    error = Some(ModelError::Protocol("the stream ended without its end".into()));
                    break;
                }
            }
        }
    }
    // Dropping the stream closes the connection; then the server is free.
    drop(events);
    drop(lease);
    emit(filter.flush());
    let prose = strip_think(py::strip(&raw));
    let tokens = usage.map(|usage| {
        (
            i64::try_from(usage.input_tokens).ok(),
            i64::try_from(usage.output_tokens).ok(),
        )
    });
    let (prompt_tokens, completion_tokens) = tokens.unwrap_or((None, None));
    if cancelled {
        if prose.is_empty() {
            return stopped_before_anything();
        }
        return Outcome {
            text: prose,
            cancelled: true,
            prompt_tokens,
            completion_tokens,
            ground: true,
            ..Outcome::default()
        };
    }
    if let Some(failure) = error {
        if prose.is_empty() {
            return Outcome::error(models::error_sentence(&failure, base_url.as_deref()));
        }
        return Outcome {
            text: prose,
            truncated: true,
            prompt_tokens,
            completion_tokens,
            ground: true,
            ..Outcome::default()
        };
    }
    if prose.is_empty() {
        let sentence = match refusal {
            Some(words) => models::refusal_sentence(&words),
            None => EMPTY_ANSWER.into(),
        };
        return Outcome {
            prompt_tokens,
            completion_tokens,
            ..Outcome::error(sentence)
        };
    }
    Outcome {
        text: prose,
        prompt_tokens,
        completion_tokens,
        ground: true,
        ..Outcome::default()
    }
}

/// The web's development echo: its reply in pieces, 12 ms apart, Stop honoured.
async fn echo_answer(ctx: &Context, job: &Job, question: &str) -> Outcome {
    job.push(ChatEventKind::Stage {
        stage: Stage::Writing,
        detail: "writing\u{2026}".into(),
    });
    let text = echo::echo_text(question);
    let mut written = String::new();
    for piece in echo::echo_pieces(&text) {
        if job.stop_requested() {
            if written.is_empty() {
                return stopped_before_anything();
            }
            return Outcome {
                text: written,
                cancelled: true,
                ..Outcome::default()
            };
        }
        written.push_str(&piece);
        job.push(ChatEventKind::Delta { text: piece });
        let gap = ctx.echo_gap;
        // On the core's runtime, which has the timer.
        let _ = ctx
            .handle
            .spawn(async move { tokio::time::sleep(gap).await })
            .await;
    }
    Outcome {
        text,
        ..Outcome::default()
    }
}

/// Ground, save and tell: `Stage(Done)`, the saved turn (or why it was not),
/// then `Done`.
async fn save(ctx: &Context, job: &Job, turn: &Turn, outcome: Outcome) {
    let unsupported = if outcome.ground && !outcome.text.is_empty() {
        job.push(ChatEventKind::Stage {
            stage: Stage::Checking,
            detail: "checking where each figure came from".into(),
        });
        grounding::unsupported_without_facts(&outcome.text, &turn.question)
    } else {
        Vec::new()
    };
    job.push(ChatEventKind::Stage {
        stage: Stage::Done,
        detail: "done".into(),
    });
    let new_turn = NewTurn {
        unsupported,
        error: outcome.error.clone(),
        truncated: outcome.truncated,
        cancelled: outcome.cancelled,
        prompt_tokens: outcome.prompt_tokens,
        completion_tokens: outcome.completion_tokens,
        ..NewTurn::assistant(outcome.text, turn.resolution.provider.clone())
    };
    record(ctx, job, &turn.thread, new_turn).await;
}

/// `touch` and `append_answer` the turn, then emit what happened. `Done`
/// follows when the job finishes (`Job::finish`).
pub(crate) async fn record(ctx: &Context, job: &Job, thread: &str, new_turn: NewTurn) {
    let store = ctx.store.clone();
    let id = thread.to_owned();
    let pending = new_turn.clone();
    let saved = ctx
        .handle
        .spawn_blocking(move || {
            // An answer in progress is not the least recently used thread (S25).
            let _ = store.touch(&id);
            store.append_answer(&id, pending)
        })
        .await
        .unwrap_or(Err(StoreError::NotSaved));
    let error = new_turn.error.clone();
    let unsaved = || {
        let mut id = uuid::Uuid::new_v4().simple().to_string();
        id.truncate(12);
        Box::new(new_turn.clone().stored(id, (ctx.clock)()).to_chat_turn())
    };
    let kind = match saved {
        Ok(stored) if error.is_empty() => ChatEventKind::Turn {
            turn: Box::new(stored.to_chat_turn()),
            saved: true,
        },
        Ok(stored) => ChatEventKind::Error {
            message: error,
            turn: Box::new(stored.to_chat_turn()),
            saved: true,
        },
        Err(StoreError::ThreadGone) => ChatEventKind::Error {
            message: GONE.into(),
            turn: unsaved(),
            saved: false,
        },
        Err(_) => ChatEventKind::Error {
            message: NOT_SAVED.into(),
            turn: unsaved(),
            saved: false,
        },
    };
    job.push(kind);
}
