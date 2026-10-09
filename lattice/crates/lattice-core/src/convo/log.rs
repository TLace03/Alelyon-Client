//! A conversation's events in memory (the chat core's spec §11.2, CR3,
//! CR5, CR7, FG6). Not a port: the run manager's run log and the plain
//! chat's job log are its models.
//!
//! - Every event gets the next `seq` (from 1, one higher each time, never
//!   reused) and the clock's time, under one lock, and wakes the
//!   conversation's one `tokio::sync::watch`, which is all a follower waits
//!   on (CR1, CR3: no timer, no polling).
//! - **Bounds (CR7).** While a turn runs, at most [`MAX_TURN_DELTAS`] `Delta`
//!   events and [`MAX_TURN_DELTA_CHARS`] characters of their text are kept;
//!   past either, deltas are counted and dropped, and the count is reported
//!   as one `DeltasDropped{n}` when the turn finishes (FG6). `CommandProgress`
//!   counts against the same bound: its counters are cumulative, so a dropped
//!   one loses nothing the next does not say.
//! - **A finished turn releases its deltas** ([`ConversationLog::finish_turn`]):
//!   its `Delta` and `CommandProgress` events leave the log when the next
//!   turn finishes (so a follower a little behind still reads the whole
//!   text), because the saved turn holds the text and the command's output is
//!   read by window. A re-follow from an earlier `seq` then skips them;
//!   nothing is replayed twice (FG6). At most two turns' deltas are held.
//! - Reading what came after a `seq` is a binary search plus a copy of what
//!   is new, so a follower's cost does not grow with the conversation's
//!   length (CB6).
//!
//! Nothing here prints, logs or sleeps.

use std::sync::{Mutex, MutexGuard};

use lattice_protocol::conversation::{ConversationEvent, ConversationEventKind};
use tokio::sync::watch;

use crate::clock::Clock;

/// Live `Delta` (and `CommandProgress`) events kept per turn.
pub const MAX_TURN_DELTAS: usize = 50_000;
/// Characters of delta text kept per turn.
pub const MAX_TURN_DELTA_CHARS: usize = 8 * 1024 * 1024;

#[derive(Default)]
struct State {
    events: Vec<ConversationEvent>,
    next_seq: u64,
    deltas: usize,
    delta_chars: usize,
    dropped: u64,
    /// The last `seq` of the previous finished turn: its live-only events go
    /// when the next turn finishes.
    released_through: u64,
}

/// One conversation's event log.
pub struct ConversationLog {
    state: Mutex<State>,
    wake: watch::Sender<u64>,
    clock: Clock,
}

fn is_live_only(kind: &ConversationEventKind) -> bool {
    matches!(
        kind,
        ConversationEventKind::Delta { .. } | ConversationEventKind::CommandProgress { .. }
    )
}

impl ConversationLog {
    pub fn new(clock: Clock) -> Self {
        Self {
            state: Mutex::new(State {
                next_seq: 1,
                ..State::default()
            }),
            wake: watch::Sender::new(0),
            clock,
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record an event; its `seq`, or `None` when it was a delta past the
    /// bounds (counted, and reported when the turn finishes).
    pub fn push(&self, kind: ConversationEventKind) -> Option<u64> {
        let seq = {
            let mut state = self.state();
            if is_live_only(&kind) {
                let chars = match &kind {
                    ConversationEventKind::Delta { text } => text.chars().count(),
                    _ => 0,
                };
                if state.deltas >= MAX_TURN_DELTAS
                    || state.delta_chars + chars > MAX_TURN_DELTA_CHARS
                {
                    state.dropped += 1;
                    return None;
                }
                state.deltas += 1;
                state.delta_chars += chars;
            }
            let seq = state.next_seq;
            state.next_seq += 1;
            let at = (self.clock)();
            state.events.push(ConversationEvent { seq, at, kind });
            seq
        };
        self.wake.send_replace(seq);
        Some(seq)
    }

    /// The running turn has finished: report the deltas dropped past the
    /// bounds (`DeltasDropped{n}`), release the previous finished turn's
    /// `Delta` and `CommandProgress` events, and start the next turn's bounds
    /// afresh.
    pub fn finish_turn(&self) {
        let dropped = {
            let mut state = self.state();
            let through = state.released_through;
            state
                .events
                .retain(|event| event.seq > through || !is_live_only(&event.kind));
            state.released_through = state.next_seq - 1;
            state.deltas = 0;
            state.delta_chars = 0;
            std::mem::take(&mut state.dropped)
        };
        if dropped > 0 {
            self.push(ConversationEventKind::DeltasDropped { n: dropped });
        }
    }

    /// Events with `seq > after`, in order.
    pub fn after(&self, after: u64) -> Vec<ConversationEvent> {
        let state = self.state();
        let from = state.events.partition_point(|event| event.seq <= after);
        state.events[from..].to_vec()
    }

    /// Every event kept now, without the live-only ones (an open's snapshot).
    pub fn recorded(&self) -> (Vec<ConversationEvent>, u64) {
        let state = self.state();
        let events = state
            .events
            .iter()
            .filter(|event| !is_live_only(&event.kind))
            .cloned()
            .collect();
        (events, state.next_seq - 1)
    }

    /// The latest `seq` given (0 before any).
    pub fn last_seq(&self) -> u64 {
        self.state().next_seq - 1
    }

    /// Deltas dropped in the running turn so far.
    pub fn dropped(&self) -> u64 {
        self.state().dropped
    }

    /// What a follower waits on: the latest `seq`.
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.wake.subscribe()
    }

    /// Wake the followers without a new event (they find nothing and wait
    /// again); used when a follower must look at a flag.
    pub fn poke(&self) {
        self.wake.send_modify(|_| {});
    }
}
