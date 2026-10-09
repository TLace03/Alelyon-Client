//! Answer jobs: their events, their bounds, their followers (the native chat's spec
//! §3.3.5 "The jobs table" and "Following"; row C8 of the chat core's spec).
//!
//! - A job's events are kept in order with `seq` from 1, and a
//!   `tokio::sync::watch` wakes its followers (the run manager's pattern).
//! - **Bounds while running** (the run manager's caps): at most
//!   [`MAX_LIVE_DELTAS`] `Delta` events and [`MAX_LIVE_DELTA_CHARS`]
//!   characters of delta text; past either, deltas are counted and dropped
//!   from the live log. The saved turn is unaffected.
//! - **After the job finishes** its `Delta` events are released: the `Turn`
//!   or `Error` event holds the whole text.
//! - A finished job is kept [`FINISHED_RETENTION_S`] seconds, at most
//!   [`FINISHED_KEEP`] of them (`LS/jobs.py`). Pruning is lazy: it runs when a
//!   send, a regenerate, an open or a follow starts, never from a timer.
//! - **Following** coalesces: a batch is delivered at most once per
//!   [`MIN_GAP`] (80 ms), and a batch holding `Done` at once. The wait for the
//!   gap is a sleep made on the core's runtime (the follower's executor may
//!   have no timer), and only while a batch is pending; an idle follower
//!   awaits the watch and schedules nothing (CB1).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use futures::StreamExt;
use futures::stream::BoxStream;
use lattice_protocol::chat::{ChatEvent, ChatEventKind, JobId, ThreadId};
use tokio::runtime::Handle;
use tokio::sync::watch;

/// Live `Delta` events kept per job.
pub const MAX_LIVE_DELTAS: usize = 50_000;
/// Characters of delta text kept per job.
pub const MAX_LIVE_DELTA_CHARS: usize = 8 * 1024 * 1024;
/// How long a finished job is kept, in seconds.
pub const FINISHED_RETENTION_S: f64 = 300.0;
/// How many finished jobs are kept.
pub const FINISHED_KEEP: usize = 32;
/// The shortest time between two batches of one follower.
pub const MIN_GAP: Duration = Duration::from_millis(80);

#[derive(Default)]
struct Log {
    events: Vec<ChatEvent>,
    next_seq: u64,
    deltas: usize,
    delta_chars: usize,
    dropped: usize,
    finished_at: Option<f64>,
}

/// One answer job.
pub struct Job {
    pub id: JobId,
    pub thread: ThreadId,
    log: Mutex<Log>,
    wake: watch::Sender<u64>,
    cancel: watch::Sender<bool>,
}

impl Job {
    pub fn new(thread: ThreadId) -> Arc<Self> {
        Arc::new(Self {
            id: uuid::Uuid::new_v4().simple().to_string(),
            thread,
            log: Mutex::new(Log {
                next_seq: 1,
                ..Log::default()
            }),
            wake: watch::Sender::new(0),
            cancel: watch::Sender::new(false),
        })
    }

    fn log(&self) -> MutexGuard<'_, Log> {
        self.log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record an event (a delta past the bounds is counted and dropped).
    pub fn push(&self, kind: ChatEventKind) {
        let seq = {
            let mut log = self.log();
            if let ChatEventKind::Delta { text } = &kind {
                let chars = text.chars().count();
                if log.deltas >= MAX_LIVE_DELTAS || log.delta_chars + chars > MAX_LIVE_DELTA_CHARS {
                    log.dropped += 1;
                    return;
                }
                log.deltas += 1;
                log.delta_chars += chars;
            }
            let seq = log.next_seq;
            log.next_seq += 1;
            log.events.push(ChatEvent { seq, kind });
            seq
        };
        self.wake.send_replace(seq);
    }

    /// The job ended: its deltas are released (its saved turn holds the
    /// text), and `Done` is its last event. One step under the log's lock, so
    /// a follower sees either a running job or a finished one ending in
    /// `Done`, never a finished one without it.
    pub fn finish(&self, now: f64) {
        let seq = {
            let mut log = self.log();
            if log.finished_at.is_some() {
                return;
            }
            log.events
                .retain(|event| !matches!(event.kind, ChatEventKind::Delta { .. }));
            log.finished_at = Some(now);
            let seq = log.next_seq;
            log.next_seq += 1;
            log.events.push(ChatEvent {
                seq,
                kind: ChatEventKind::Done,
            });
            seq
        };
        self.wake.send_replace(seq);
    }

    pub fn finished(&self) -> bool {
        self.log().finished_at.is_some()
    }

    /// Ask the answer task to stop. True when the job was still running.
    pub fn stop(&self) -> bool {
        if self.finished() {
            return false;
        }
        self.cancel.send_replace(true);
        true
    }

    /// Resolves when a stop is asked for.
    pub fn stopped(&self) -> impl std::future::Future<Output = ()> + Send + 'static {
        let mut receiver = self.cancel.subscribe();
        async move {
            let _ = receiver.wait_for(|stop| *stop).await;
        }
    }

    pub fn stop_requested(&self) -> bool {
        *self.cancel.borrow()
    }

    /// Events with `seq > after`, and whether the job has finished.
    pub fn events_after(&self, after: u64) -> (Vec<ChatEvent>, bool) {
        let log = self.log();
        let events = log
            .events
            .iter()
            .filter(|event| event.seq > after)
            .cloned()
            .collect();
        (events, log.finished_at.is_some())
    }

    /// Deltas dropped past the bounds.
    pub fn dropped(&self) -> usize {
        self.log().dropped
    }

    /// Resolves when the job has finished.
    pub async fn wait_finished(&self) {
        let mut receiver = self.wake.subscribe();
        while !self.finished() {
            if receiver.changed().await.is_err() {
                return;
            }
        }
    }

    /// Every event kept now (tests).
    #[cfg(test)]
    pub fn events(&self) -> Vec<ChatEvent> {
        self.log().events.clone()
    }
}

/// The jobs of this process.
#[derive(Default)]
pub struct Jobs {
    jobs: Mutex<HashMap<JobId, Arc<Job>>>,
}

impl Jobs {
    fn lock(&self) -> MutexGuard<'_, HashMap<JobId, Arc<Job>>> {
        self.jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn insert(&self, job: Arc<Job>) {
        self.lock().insert(job.id.clone(), job);
    }

    pub fn get(&self, id: &str) -> Option<Arc<Job>> {
        self.lock().get(id).cloned()
    }

    /// The running job of `thread`, if any.
    pub fn running_for(&self, thread: &str) -> Option<Arc<Job>> {
        self.lock()
            .values()
            .find(|job| job.thread == thread && !job.finished())
            .cloned()
    }

    pub fn running(&self) -> Vec<Arc<Job>> {
        self.lock()
            .values()
            .filter(|job| !job.finished())
            .cloned()
            .collect()
    }

    /// `_prune_locked`: finished jobs older than the retention, and all but
    /// the newest [`FINISHED_KEEP`], are forgotten.
    pub fn prune(&self, now: f64) {
        let mut jobs = self.lock();
        let mut finished: Vec<(f64, JobId)> = jobs
            .values()
            .filter_map(|job| job.log().finished_at.map(|at| (at, job.id.clone())))
            .collect();
        finished.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (index, (at, id)) in finished.into_iter().enumerate() {
            if index >= FINISHED_KEEP || now - at > FINISHED_RETENTION_S {
                jobs.remove(&id);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }
}

/// A follower's stream: batches at least `gap` apart, the batch holding
/// `Done` at once; it ends after that batch, or when the job is gone.
pub fn follow(
    job: Arc<Job>,
    after: u64,
    gap: Duration,
    handle: Handle,
) -> BoxStream<'static, Vec<ChatEvent>> {
    let receiver = job.wake.subscribe();
    let state = (job, receiver, after, None::<Instant>, false, handle);
    futures::stream::unfold(
        state,
        move |(job, mut receiver, mut last, mut sent_at, done, handle)| async move {
            if done {
                return None;
            }
            loop {
                // Mark the wake seen before reading, so nothing pushed while
                // reading is missed.
                receiver.borrow_and_update();
                let (batch, finished) = job.events_after(last);
                if batch.is_empty() {
                    if finished || receiver.changed().await.is_err() {
                        return None;
                    }
                    continue;
                }
                let urgent = batch
                    .iter()
                    .any(|event| matches!(event.kind, ChatEventKind::Done));
                if let Some(at) = sent_at
                    && !urgent
                {
                    let since = at.elapsed();
                    if since < gap {
                        // The sleep is made on the core's runtime, inside the
                        // task: making it needs a timer.
                        let rest = gap - since;
                        let _ = handle
                            .spawn(async move { tokio::time::sleep(rest).await })
                            .await;
                        continue;
                    }
                }
                last = batch.last().map(|event| event.seq).unwrap_or(last);
                sent_at = Some(Instant::now());
                return Some((batch, (job, receiver, last, sent_at, urgent, handle)));
            }
        },
    )
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta(text: &str) -> ChatEventKind {
        ChatEventKind::Delta { text: text.into() }
    }

    #[test]
    fn events_are_numbered_from_one_and_deltas_are_bounded_then_released() {
        let job = Job::new("t".into());
        job.push(delta("ab"));
        job.push(ChatEventKind::Stage {
            stage: lattice_protocol::chat::Stage::Done,
            detail: "done".into(),
        });
        job.finish(1.0);
        let seqs: Vec<u64> = job.events().iter().map(|e| e.seq).collect();
        assert_eq!(seqs, [2, 3], "the delta is released; Done is last");
        assert!(matches!(job.events()[1].kind, ChatEventKind::Done));
        job.finish(2.0);
        assert_eq!(job.events().len(), 2, "a job finishes once");
        let big = Job::new("t".into());
        for _ in 0..MAX_LIVE_DELTAS + 5 {
            big.push(delta("x"));
        }
        assert_eq!(big.events().len(), MAX_LIVE_DELTAS);
        assert_eq!(big.dropped(), 5);
        let long = Job::new("t".into());
        long.push(delta(&"y".repeat(MAX_LIVE_DELTA_CHARS)));
        long.push(delta("z"));
        assert_eq!(
            long.events().len(),
            1,
            "past 8 Mi characters, deltas are dropped"
        );
        long.finish(1.0);
        assert_eq!(long.events().len(), 1, "a finished job holds no delta");
        assert!(matches!(long.events()[0].kind, ChatEventKind::Done));
    }

    #[test]
    fn pruning_forgets_old_and_surplus_finished_jobs_only() {
        let jobs = Jobs::default();
        let running = Job::new("r".into());
        jobs.insert(running.clone());
        let old = Job::new("o".into());
        old.finish(0.0);
        jobs.insert(old.clone());
        for at in 0..40 {
            let job = Job::new(format!("t{at}"));
            job.finish(1_000.0 + f64::from(at));
            jobs.insert(job);
        }
        jobs.prune(1_100.0);
        assert!(
            jobs.get(&running.id).is_some(),
            "a running job is never pruned"
        );
        assert!(jobs.get(&old.id).is_none(), "older than 300 s");
        assert_eq!(jobs.len(), 1 + FINISHED_KEEP);
    }

    #[test]
    fn a_stop_is_asked_once_and_only_while_running() {
        let job = Job::new("t".into());
        assert!(!job.stop_requested());
        assert!(job.stop());
        assert!(job.stop_requested());
        job.finish(1.0);
        assert!(!job.stop(), "a finished job is not stopped");
    }
}
