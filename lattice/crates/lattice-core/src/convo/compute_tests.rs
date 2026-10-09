//! The core's compute budgets at the agent chat's level (the chat core's spec
//! §11.4; row E12): CB1 in the idle states E9 and E11 left owed, CB4 and CB5.
//! CB2, CB3 and CB6 are the coalescer's (`follow_tests`, row E11a); CB1's
//! waiting question at the tool's level is E9's (`tools::ask_tests`).
//!
//! Every test drives `AgentChat` over the E11 harness ([`H`]): a scratch
//! repository, the memory transcript store, a scripted model, and a 2-worker
//! runtime whose `on_thread_unpark` hook counts its workers' wakes. Nothing
//! reaches the network, a GPU or the real `globals/`; the one program started
//! (CB4) is Windows PowerShell, through the core's own approved-command path,
//! in the scratch repository.
//!
//! **Under load.** The gates run these beside every other test. Each wait is
//! on a condition with a generous deadline, never a fixed sleep standing in
//! for one: CB1's 200 ms settle is "200 ms in which the runtime did not wake"
//! (a turn's tail of disk writes may run late under load), and only then does
//! the 2 s window count; CB4 bounds batches per any 2 s of arrival, which is
//! what the gap guarantees however long the flood is stretched; CB5 waits for
//! the task count to come back.

use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use lattice_protocol::conversation::{
    AgentChatService, ConversationEvent, ConversationEventKind, Decision, ReviewOp, ReviewResult,
};
use serde_json::json;

use super::agent_tests::{H, call, say};
use super::follow::Gap;
use super::item::Item;
use super::sidecar::SidecarStore;

/// CB1: the settle, as time with no wake.
const SETTLE: Duration = Duration::from_millis(200);
/// CB1: the window that must count no wake.
const WINDOW: Duration = Duration::from_secs(2);
/// The longest any condition is waited for.
const DEADLINE: Duration = Duration::from_secs(120);

fn unparks(h: &H) -> u64 {
    h.unparks.load(Ordering::SeqCst)
}

/// Wait until the runtime has not woken for [`SETTLE`].
fn settle(h: &H, state: &str) {
    let deadline = Instant::now() + DEADLINE;
    let mut last = unparks(h);
    let mut quiet_since = Instant::now();
    while quiet_since.elapsed() < SETTLE {
        std::thread::sleep(Duration::from_millis(10));
        let now = unparks(h);
        if now != last {
            last = now;
            quiet_since = Instant::now();
        }
        assert!(
            Instant::now() < deadline,
            "{state}: the runtime never stayed quiet for {SETTLE:?} within {DEADLINE:?} ({last} wakes)"
        );
    }
}

/// A positive control: the runtime wakes after `before` (a wake may land
/// just after the call that caused it returns, so it is waited for).
fn woke(h: &H, before: u64, what: &str) {
    let deadline = Instant::now() + DEADLINE;
    while unparks(h) <= before {
        assert!(
            Instant::now() < deadline,
            "positive control: {what} did not wake the runtime"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// CB1's measurement: settle, then count the wakes in [`WINDOW`].
fn idle_window(h: &H, state: &str) -> u64 {
    settle(h, state);
    let before = unparks(h);
    std::thread::sleep(WINDOW);
    let during = unparks(h) - before;
    println!("cb1 {state}: {during} wakes in {WINDOW:?}");
    during
}

/// The interface's side of a follow: a task on the core's runtime that holds
/// the stream and hands each batch over with the time it arrived. Dropping
/// `stop` (or sending on it) drops the stream.
struct Follower {
    batches: mpsc::Receiver<(Instant, Vec<ConversationEvent>)>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}

impl Follower {
    fn new(h: &H, id: &str, after: u64) -> Self {
        let mut stream = h.chat.follow(id, after).unwrap();
        let (sender, batches) = mpsc::channel();
        let (stop, mut stopped) = tokio::sync::oneshot::channel::<()>();
        let task = h.runtime.spawn(async move {
            loop {
                tokio::select! {
                    batch = stream.next() => match batch {
                        Some(batch) => {
                            if sender.send((Instant::now(), batch)).is_err() {
                                return;
                            }
                        }
                        None => return,
                    },
                    _ = &mut stopped => return,
                }
            }
        });
        Self {
            batches,
            stop: Some(stop),
            task,
        }
    }

    /// Batches until one holds an event `done` accepts.
    fn until(
        &self,
        done: impl Fn(&ConversationEventKind) -> bool,
    ) -> Vec<(Instant, Vec<ConversationEvent>)> {
        let deadline = Instant::now() + DEADLINE;
        let mut all = Vec::new();
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let (at, batch) = self
                .batches
                .recv_timeout(left)
                .unwrap_or_else(|_| panic!("timed out; {} batches so far", all.len()));
            let found = batch.iter().any(|event| done(&event.kind));
            all.push((at, batch));
            if found {
                return all;
            }
        }
    }

    /// Drop the stream and wait for the interface's task to end.
    fn drop_stream(mut self, h: &H) {
        drop(self.stop.take());
        h.runtime.block_on(self.task).unwrap();
    }
}

fn is_approval(kind: &ConversationEventKind) -> bool {
    matches!(kind, ConversationEventKind::ApprovalRequested { .. })
}

fn is_turn_end(kind: &ConversationEventKind) -> bool {
    matches!(kind, ConversationEventKind::TurnEnded { .. })
}

// ------------------------------------------------------------------- CB1

/// CB1: a workspace attached, then a conversation open (a finished turn,
/// opened), each wakes nothing in 2 s after a 200 ms settle. Positive
/// control: a send then wakes the runtime and its turn ends.
/// Mutant: CR9/CB1's 250 ms interval in the follow (it is caught by the
/// follow state below; these two hold no follow).
#[test]
fn cb1_a_workspace_attached_and_a_conversation_open_wake_nothing() {
    let h = H::new("cb1-open");
    let ws = h.workspace();
    assert_eq!(idle_window(&h, "a workspace attached"), 0);
    h.script(vec![say("first"), say("second")]);
    let id = h.agent(None, "hello", &ws);
    h.turns_end(&id, 1);
    h.runtime.block_on(h.chat.open(&id)).unwrap();
    assert_eq!(idle_window(&h, "a conversation open"), 0);
    let before = unparks(&h);
    h.agent(Some(&id), "again", &ws);
    h.turns_end(&id, 2);
    woke(&h, before, "a send");
}

/// CB1: an approval waiting, with the conversation followed (as the
/// interface shows it), wakes nothing. Positive control: Stop wakes it and
/// ends the turn.
#[test]
fn cb1_an_approval_waiting_wakes_nothing() {
    let h = H::new("cb1-approval");
    let ws = h.workspace();
    h.script(vec![
        call(
            "run_command",
            json!({"command": "Write-Output never-runs", "timeout_s": 60}),
            "c1",
        ),
        say("unused"),
    ]);
    let id = h.agent(None, "run it", &ws);
    let follower = Follower::new(&h, &id, 0);
    follower.until(is_approval);
    assert_eq!(idle_window(&h, "an approval waiting"), 0);
    let before = unparks(&h);
    assert!(h.chat.stop(&id));
    follower.until(is_turn_end);
    woke(&h, before, "Stop");
    follower.drop_stream(&h);
}

/// CB1: a question waiting, through the agent chat and followed, wakes
/// nothing (E9 pinned the tool alone). Positive control: the answer wakes it
/// and the turn ends.
#[test]
fn cb1_a_question_waiting_wakes_nothing() {
    let h = H::new("cb1-question");
    let ws = h.workspace();
    h.script(vec![
        call("ask_question", json!({"question": "Proceed?"}), "q1"),
        say("done"),
    ]);
    let id = h.agent(None, "ask me", &ws);
    let follower = Follower::new(&h, &id, 0);
    follower.until(|kind| matches!(kind, ConversationEventKind::Question { .. }));
    assert_eq!(idle_window(&h, "a question waiting"), 0);
    let before = unparks(&h);
    h.chat.answer(&id, "q1", "yes".into()).unwrap();
    follower.until(is_turn_end);
    woke(&h, before, "the answer");
    follower.drop_stream(&h);
}

/// CB1: staged changes waiting for review, followed, wake nothing. Positive
/// control: an Undo wakes it, and the folder is unchanged.
#[test]
fn cb1_staged_changes_waiting_wake_nothing() {
    let h = H::new("cb1-staged");
    let ws = h.workspace();
    h.script(vec![
        call(
            "edit_file",
            json!({"path": "a.txt", "old_string": "a", "new_string": "b"}),
            "e1",
        ),
        say("staged"),
    ]);
    let id = h.agent(None, "change a.txt", &ws);
    let follower = Follower::new(&h, &id, 0);
    let batches = follower.until(is_turn_end);
    let change = batches
        .iter()
        .flat_map(|(_, batch)| batch)
        .find_map(|event| match &event.kind {
            ConversationEventKind::Staged { change, .. } => Some(change.clone()),
            _ => None,
        })
        .expect("a Staged event");
    let waiting = h.runtime.block_on(h.chat.changes(&id)).unwrap();
    assert_eq!(waiting.changes.len(), 1, "{waiting:?}");
    assert_eq!(idle_window(&h, "staged changes waiting"), 0);
    let before = unparks(&h);
    h.runtime
        .block_on(h.chat.review(
            &id,
            vec![ReviewOp::Undo {
                change,
                hunks: None,
                note: None,
            }],
        ))
        .unwrap();
    woke(&h, before, "the review");
    assert_eq!(std::fs::read(h.folder.join("a.txt")).unwrap(), b"a\n");
    follower.drop_stream(&h);
}

/// CB1: a follow subscribed with nothing happening wakes nothing (FG1).
/// Positive control: an event then reaches the follower as a batch.
/// Mutant (spec §11.4): a 250 ms interval in the follow.
#[test]
fn cb1_a_follow_with_nothing_happening_wakes_nothing() {
    let h = H::new("cb1-follow");
    let ws = h.workspace();
    h.script(vec![say("first")]);
    let id = h.agent(None, "hello", &ws);
    let follower = Follower::new(&h, &id, 0);
    follower.until(is_turn_end);
    assert_eq!(idle_window(&h, "a follow with nothing happening"), 0);
    let before = unparks(&h);
    let log = h.chat.inner.loaded(&id).expect("loaded").log.clone();
    log.push(ConversationEventKind::Delta {
        text: "wake".into(),
    });
    let batches = follower
        .until(|kind| matches!(kind, ConversationEventKind::Delta { text } if text == "wake"));
    assert_eq!(batches.len(), 1, "one batch for one event");
    woke(&h, before, "the event");
    follower.drop_stream(&h);
}

// ------------------------------------------------------------------- CB4

/// Lines a second, and for how long, as CB4 names them.
const FLOOD_LINES_PER_SECOND: u64 = 50_000;
const FLOOD_SECONDS: u64 = 2;
/// Each line's bytes with its LF: 100,000 lines are 20,000,000 bytes, past
/// the command's 8 MiB buffer.
const FLOOD_LINE_BYTES: u64 = 200;

/// The flood, as the approved command: 5,000 lines every 100 ms for 2 s,
/// paced by the child's own clock.
fn flood_command() -> String {
    let per_tick = FLOOD_LINES_PER_SECOND / 10;
    let ticks = FLOOD_SECONDS * 10;
    let width = FLOOD_LINE_BYTES - 1;
    format!(
        "$line = ('x' * {width}) + \"`n\"; $block = $line * {per_tick}; $out = [Console]::Out; \
         $clock = [Diagnostics.Stopwatch]::StartNew(); \
         for ($i = 1; $i -le {ticks}; $i++) {{ $out.Write($block); \
         $rest = $i * 100 - $clock.ElapsedMilliseconds; \
         if ($rest -gt 0) {{ Start-Sleep -Milliseconds $rest }} }}; $out.Flush()"
    )
}

/// CB4: an approved command writes 50,000 lines a second for 2 s (20 MB).
/// The follow sends at most 27 batches in any 2 s of their arrival, each at
/// most 4 KiB serialised; the counters reach every byte and line (stdout
/// and stderr together); the
/// command's kept output (its blob) is at most 8 MiB.
/// Mutants: every CommandProgress kept in a batch (not the last per call);
/// the ring raised past 8 MiB.
#[test]
fn cb4_an_output_flood_is_few_small_batches_and_a_bounded_buffer() {
    let h = H::new("cb4-flood");
    let ws = h.workspace();
    h.script(vec![
        call(
            "run_command",
            json!({"command": flood_command(), "timeout_s": 600}),
            "c1",
        ),
        say("flooded"),
    ]);
    let id = h.agent(None, "flood", &ws);
    let follower = Follower::new(&h, &id, 0);
    follower.until(is_approval);
    h.runtime
        .block_on(h.chat.decide(&id, "c1", Decision::Approve))
        .unwrap();
    let batches = follower.until(is_turn_end);
    follower.drop_stream(&h);
    let progress: Vec<(Instant, usize, u64, u64)> = batches
        .iter()
        .filter_map(|(at, batch)| {
            let size = serde_json::to_vec(batch).unwrap().len();
            batch.iter().rev().find_map(|event| match &event.kind {
                ConversationEventKind::CommandProgress {
                    call_id,
                    bytes,
                    lines,
                    ..
                } if call_id == "c1" => Some((*at, size, *bytes, *lines)),
                _ => None,
            })
        })
        .collect();
    assert!(!progress.is_empty(), "no CommandProgress batch");
    let span = progress.last().unwrap().0 - progress[0].0;
    let busiest = (0..progress.len())
        .map(|i| {
            progress[i..]
                .iter()
                .take_while(|(at, ..)| *at - progress[i].0 < Duration::from_secs(2))
                .count()
        })
        .max()
        .unwrap();
    let largest = progress.iter().map(|(_, size, ..)| *size).max().unwrap();
    let (_, _, bytes, lines) = *progress.last().unwrap();
    let blob = h
        .sidecar_items(&id)
        .iter()
        .find_map(|item| match item {
            Item::ToolResult {
                call_id,
                output_blob: Some(sha),
                ..
            } if call_id == "c1" => Some(sha.clone()),
            _ => None,
        })
        .expect("the command's output blob");
    let kept = SidecarStore::new(h.state.native_chat_dir())
        .read_blob(&id, &blob)
        .unwrap()
        .len();
    println!(
        "cb4: {} progress batches over {span:?}, at most {busiest} in any 2 s, largest {largest} bytes; \
         {bytes} bytes and {lines} lines written, {kept} bytes kept",
        progress.len()
    );
    let total_lines = FLOOD_LINES_PER_SECOND * FLOOD_SECONDS;
    // Both pipes count: PowerShell adds a few hundred bytes of its own on
    // stderr (observed: 392 bytes, one line).
    assert!(lines >= total_lines, "every line counted: {lines}");
    assert!(
        bytes >= total_lines * FLOOD_LINE_BYTES,
        "every byte counted: {bytes}"
    );
    assert!(busiest <= 27, "{busiest} batches in 2 s");
    assert!(largest <= 4 * 1024, "a batch of {largest} bytes");
    assert!(
        bytes > 8 * 1024 * 1024,
        "positive control: the flood passed the buffer"
    );
    assert!(kept <= 8 * 1024 * 1024, "{kept} bytes kept");
}

// ------------------------------------------------------------------- CB5

/// CB5: dropping a follow's stream brings the runtime's alive tasks back to
/// the baseline, even when it is dropped while a batch waits out its gap
/// (the gap made 2 s here, so the wait is certain to be pending). Positive
/// control: while the batch waits, a task of the core's is alive beside the
/// interface's.
/// Mutant: the gap's task does not end when its sleep does.
#[test]
fn cb5_dropping_the_stream_returns_the_tasks_to_the_baseline() {
    let gap = Duration::from_secs(2);
    let h = H::with("cb5-unfollow", &[], |config| {
        config.gap = Gap {
            min: gap,
            slow_after: None,
        };
    });
    let ws = h.workspace();
    h.script(vec![say("first")]);
    let id = h.agent(None, "hello", &ws);
    h.turns_end(&id, 1);
    let log = h.chat.inner.loaded(&id).expect("loaded").log.clone();
    let alive = || h.runtime.metrics().num_alive_tasks();
    let wait_for = |what: &str, done: &dyn Fn(usize) -> bool| {
        let deadline = Instant::now() + DEADLINE;
        loop {
            let now = alive();
            if done(now) {
                return now;
            }
            assert!(Instant::now() < deadline, "{what}: {now} tasks alive");
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    settle(&h, "before the follow");
    let baseline = alive();
    let follower = Follower::new(&h, &id, log.last_seq());
    log.push(ConversationEventKind::Delta {
        text: "first".into(),
    });
    follower.until(|kind| matches!(kind, ConversationEventKind::Delta { .. }));
    // The next event waits out the gap in a task of the core's.
    log.push(ConversationEventKind::Delta {
        text: "second".into(),
    });
    let following = wait_for("a pending gap", &|now| now >= baseline + 2);
    follower.drop_stream(&h);
    let after = wait_for("back to the baseline", &|now| now <= baseline);
    println!("cb5: baseline {baseline}, {following} while a batch waited, {after} after the drop");
    assert_eq!(after, baseline);
}

// ------------------------------------------------------- review off-runtime

/// A review's disk work runs on the blocking pool, not on an async worker
/// (CR2's intent; the verifier's "review blocks the async runtime"). On
/// the harness's 2-worker runtime, two Keeps of one conversation are
/// started while its checkpoints are held (a checkpoint as slow as the
/// test makes it); a follow of the conversation still gets an event
/// pushed meanwhile, within 10 s. Then the checkpoints are released and
/// both Keeps finish.
/// Falsifier: the review run on the worker (fails on the code before this
/// commit: both workers held, the event never arrives).
#[test]
fn a_keep_with_a_slow_checkpoint_never_holds_an_async_worker() {
    let h = H::new("review-off-runtime");
    let ws = h.workspace();
    h.script(vec![
        call(
            "edit_file",
            json!({"path": "a.txt", "old_string": "a", "new_string": "b"}),
            "e1",
        ),
        call(
            "write_file",
            json!({"path": "new.txt", "content": "fresh\n"}),
            "w1",
        ),
        say("staged"),
    ]);
    let id = h.agent(None, "change two files", &ws);
    h.turns_end(&id, 1);
    let changes = h.runtime.block_on(h.chat.changes(&id)).unwrap().changes;
    assert_eq!(changes.len(), 2, "{changes:?}");
    let convo = h.chat.inner.loaded(&id).expect("loaded");
    let checkpoints = convo.state().checkpoints.clone().expect("checkpoints");
    let (began, release) = checkpoints.hold_takes();
    let log = convo.log.clone();
    let follower = Follower::new(&h, &id, log.last_seq());
    let keeps: Vec<_> = changes
        .iter()
        .map(|change| {
            h.runtime.spawn(h.chat.review(
                &id,
                vec![ReviewOp::Keep {
                    change: change.id.clone(),
                    hunks: None,
                }],
            ))
        })
        .collect();
    began
        .recv_timeout(DEADLINE)
        .expect("a Keep reached its checkpoint");
    // Give the second Keep time to reach the point where it waits (the
    // checkpoint, or the staging lock the first holds).
    std::thread::sleep(Duration::from_millis(500));
    log.push(ConversationEventKind::Delta {
        text: "still here".into(),
    });
    let delivered = follower
        .batches
        .recv_timeout(Duration::from_secs(10))
        .map(|(_, batch)| batch);
    // Release both before any assertion, so a failure ends cleanly.
    let _ = release.send(());
    let _ = release.send(());
    let delivered = delivered.expect("the follow delivered while the Keeps waited");
    assert!(
        delivered
            .iter()
            .any(|event| matches!(&event.kind, ConversationEventKind::Delta { text } if text == "still here")),
        "{delivered:?}"
    );
    for keep in keeps {
        let outcome = h.runtime.block_on(keep).unwrap().unwrap();
        assert!(
            outcome
                .results
                .iter()
                .all(|result| matches!(result.result, ReviewResult::Kept { .. })),
            "{outcome:?}"
        );
    }
    assert_eq!(std::fs::read(h.folder.join("a.txt")).unwrap(), b"b\n");
    assert_eq!(std::fs::read(h.folder.join("new.txt")).unwrap(), b"fresh\n");
    follower.drop_stream(&h);
}
