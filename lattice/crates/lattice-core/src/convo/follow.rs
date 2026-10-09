//! The follow coalescer: a conversation's events in batches, by the
//! frame-gap contract with the interface (the chat core's spec §11.2,
//! §11.3 FG1–FG8, CR2–CR4). Not a port; the plain chat's follower is its
//! model, plus urgency, joining and the size bound.
//!
//! - **FG1.** Nothing at idle: an idle follower awaits the log's `watch` and
//!   schedules nothing (no timer, CB1).
//! - **FG2.** At least [`MIN_GAP`] (80 ms) between batches, measured at send.
//!   The wait is a sleep made on the core's runtime (the follower's executor
//!   may have no timer), and only while a batch is pending (CR2).
//! - **FG3.** A batch holding an urgent event (`ApprovalRequested`,
//!   `Question`, `Withheld`, `Error`, `TurnSaved`, `TurnEnded`) goes at once,
//!   and the gap starts again from it.
//! - **FG4.** The named fallback, off unless the interface's measurement
//!   fails: once the running turn's visible text passes 16 KiB, the gap
//!   becomes 250 ms ([`Gap::slow_after`]).
//! - **FG5.** No bulk: adjacent `Delta` events are joined into one (it keeps
//!   the `seq` of the last it holds); `CommandProgress` keeps only the last
//!   one of each call in a batch (its counters are cumulative); a batch is cut
//!   at 256 KiB serialised, and what is left goes in the next one.
//! - **FG6.** Contiguous and idempotent: a batch covers every event after the
//!   previous batch's last `seq`, in order (joined and coalesced events keep
//!   their last `seq`), so a re-follow from a batch's last `seq` replays
//!   nothing twice.
//! - **FG7.** Hidden means silent: dropping the stream ends it; no task of
//!   the core stays alive for it beyond a pending gap's sleep.
//! - **FG8.** The list has no stream of its own: [`Changed`] carries the ids
//!   of conversations whose summary changed, at most once per gap.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::stream::BoxStream;
use lattice_protocol::conversation::{ConversationEvent, ConversationEventKind};
use tokio::runtime::Handle;
use tokio::sync::watch;

use super::log::ConversationLog;

/// The shortest time between two batches (FG2).
pub const MIN_GAP: Duration = Duration::from_millis(80);
/// FG4's gap, once the running turn's visible text passes [`SLOW_AFTER`].
pub const SLOW_GAP: Duration = Duration::from_millis(250);
/// FG4's threshold, in characters of visible text.
pub const SLOW_AFTER: usize = 16 * 1024;
/// The most bytes one batch takes, serialised (FG5).
pub const MAX_BATCH_BYTES: usize = 256 * 1024;

/// The gap a follower keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gap {
    pub min: Duration,
    /// FG4: `Some((threshold, gap))` once the interface's per-batch
    /// measurement failed; `None` by default.
    pub slow_after: Option<(usize, Duration)>,
}

impl Default for Gap {
    fn default() -> Self {
        Self {
            min: MIN_GAP,
            slow_after: None,
        }
    }
}

fn serialized_len(event: &ConversationEvent) -> usize {
    serde_json::to_vec(event).map_or(0, |bytes| bytes.len())
}

/// FG5: join adjacent deltas, keep the last progress of each call, and cut
/// at [`MAX_BATCH_BYTES`]. Returns the batch and how many raw events it
/// covers (the rest wait for the next batch).
pub fn coalesce(raw: &[ConversationEvent]) -> (Vec<ConversationEvent>, usize) {
    // The last CommandProgress of each call within what this batch covers is
    // decided after the cut, so first cut on the raw events' sizes.
    let mut covered = 0;
    let mut bytes = 0;
    for event in raw {
        let size = serialized_len(event);
        if covered > 0 && bytes + size > MAX_BATCH_BYTES {
            break;
        }
        bytes += size;
        covered += 1;
    }
    let raw = &raw[..covered];
    let mut last_progress: HashMap<&str, u64> = HashMap::new();
    for event in raw {
        if let ConversationEventKind::CommandProgress { call_id, .. } = &event.kind {
            last_progress.insert(call_id.as_str(), event.seq);
        }
    }
    let mut out: Vec<ConversationEvent> = Vec::with_capacity(raw.len());
    for event in raw {
        match &event.kind {
            ConversationEventKind::CommandProgress { call_id, .. }
                if last_progress.get(call_id.as_str()) != Some(&event.seq) => {}
            ConversationEventKind::Delta { text } => {
                if let Some(ConversationEvent {
                    seq,
                    kind: ConversationEventKind::Delta { text: joined },
                    ..
                }) = out.last_mut()
                {
                    joined.push_str(text);
                    *seq = event.seq;
                } else {
                    out.push(event.clone());
                }
            }
            _ => out.push(event.clone()),
        }
    }
    (out, covered)
}

/// The visible characters a batch adds (FG4); a new turn starts the count.
fn visible_after(mut visible: usize, batch: &[ConversationEvent]) -> usize {
    for event in batch {
        match &event.kind {
            ConversationEventKind::TurnStarted { .. } => visible = 0,
            ConversationEventKind::Delta { text } => visible += text.chars().count(),
            _ => {}
        }
    }
    visible
}

struct Follower {
    log: Arc<ConversationLog>,
    receiver: watch::Receiver<u64>,
    last: u64,
    sent_at: Option<Instant>,
    visible: usize,
    gap: Gap,
    handle: Handle,
}

impl Follower {
    fn gap(&self) -> Duration {
        match self.gap.slow_after {
            Some((threshold, slow)) if self.visible > threshold => slow.max(self.gap.min),
            _ => self.gap.min,
        }
    }

    async fn next(&mut self) -> Option<Vec<ConversationEvent>> {
        loop {
            // Mark the wake seen before reading, so nothing pushed while
            // reading is missed.
            self.receiver.borrow_and_update();
            let raw = self.log.after(self.last);
            if raw.is_empty() {
                // FG1: nothing changed, nothing sent, nothing scheduled.
                if self.receiver.changed().await.is_err() {
                    return None;
                }
                continue;
            }
            let urgent = raw.iter().any(|event| event.kind.is_urgent());
            if let Some(at) = self.sent_at
                && !urgent
            {
                let since = at.elapsed();
                let gap = self.gap();
                if since < gap {
                    let rest = gap - since;
                    // FG2: the sleep is made on the core's runtime, inside
                    // the task, and only while a batch is pending.
                    let mut receiver = self.receiver.clone();
                    let log = self.log.clone();
                    let last = self.last;
                    let _ = self
                        .handle
                        .spawn(async move {
                            let sleep = tokio::time::sleep(rest);
                            tokio::pin!(sleep);
                            // An urgent event cuts the wait short (FG3).
                            loop {
                                tokio::select! {
                                    () = &mut sleep => return,
                                    changed = receiver.changed() => {
                                        if changed.is_err() {
                                            return;
                                        }
                                        if log.after(last).iter().any(|e| e.kind.is_urgent()) {
                                            return;
                                        }
                                    }
                                }
                            }
                        })
                        .await;
                    continue;
                }
            }
            let (batch, covered) = coalesce(&raw);
            self.last = raw[covered - 1].seq;
            self.sent_at = Some(Instant::now());
            self.visible = visible_after(self.visible, &batch);
            return Some(batch);
        }
    }
}

/// Follow `log` from `after`: batches by FG1–FG7, for as long as the stream
/// is held.
pub fn follow(
    log: Arc<ConversationLog>,
    after: u64,
    gap: Gap,
    handle: Handle,
) -> BoxStream<'static, Vec<ConversationEvent>> {
    let follower = Follower {
        receiver: log.subscribe(),
        log,
        last: after,
        sent_at: None,
        visible: 0,
        gap,
        handle,
    };
    futures::stream::unfold(follower, |mut follower| async move {
        let batch = follower.next().await?;
        Some((batch, follower))
    })
    .boxed()
}

/// FG8: the conversations whose summary changed, as ids only, at most once
/// per gap, for the sidebar.
#[derive(Clone)]
pub struct Changed {
    ids: Arc<Mutex<BTreeSet<String>>>,
    wake: Arc<watch::Sender<u64>>,
}

impl Default for Changed {
    fn default() -> Self {
        Self {
            ids: Arc::default(),
            wake: Arc::new(watch::Sender::new(0)),
        }
    }
}

impl Changed {
    /// Note that `id`'s summary changed.
    pub fn note(&self, id: &str) {
        self.ids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(id.to_owned());
        self.wake
            .send_modify(|count| *count = count.wrapping_add(1));
    }

    /// The notices: each a set of ids, at least `gap` apart; nothing at idle.
    pub fn stream(&self, gap: Duration, handle: Handle) -> BoxStream<'static, Vec<String>> {
        let state = (self.clone(), self.wake.subscribe(), None::<Instant>, handle);
        futures::stream::unfold(
            state,
            move |(changed, mut receiver, sent_at, handle)| async move {
                loop {
                    receiver.borrow_and_update();
                    let pending = !changed
                        .ids
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .is_empty();
                    if !pending {
                        receiver.changed().await.ok()?;
                        continue;
                    }
                    if let Some(at) = sent_at {
                        let since = at.elapsed();
                        if since < gap {
                            let rest = gap - since;
                            let _ = handle
                                .spawn(async move { tokio::time::sleep(rest).await })
                                .await;
                        }
                    }
                    let ids: Vec<String> = std::mem::take(
                        &mut *changed
                            .ids
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner()),
                    )
                    .into_iter()
                    .collect();
                    if ids.is_empty() {
                        continue;
                    }
                    return Some((ids, (changed, receiver, Some(Instant::now()), handle)));
                }
            },
        )
        .boxed()
    }
}
