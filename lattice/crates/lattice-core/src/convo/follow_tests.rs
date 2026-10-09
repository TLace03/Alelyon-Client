//! The follow coalescer and the conversation log (spec §11.2, §11.3 FG1–FG8,
//! CR7; CB2, CB3, CB6).

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use lattice_protocol::Locality;
use lattice_protocol::conversation::{
    ApprovalDetail, ApprovalKind, CommandMode, ConversationEvent, ConversationEventKind, Mode,
    TurnKind, TurnStatus,
};

use super::follow::{Changed, Gap, MAX_BATCH_BYTES, MIN_GAP, coalesce, follow};
use super::log::{ConversationLog, MAX_TURN_DELTAS};
use crate::clock::system_clock;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

fn new_log() -> Arc<ConversationLog> {
    Arc::new(ConversationLog::new(system_clock()))
}

fn delta(text: &str) -> ConversationEventKind {
    ConversationEventKind::Delta { text: text.into() }
}

fn approval(call: &str) -> ConversationEventKind {
    ConversationEventKind::ApprovalRequested {
        call_id: call.into(),
        kind: ApprovalKind::Command,
        detail: ApprovalDetail::Command {
            text: "cargo test".into(),
            cwd: String::new(),
            mode: CommandMode::PowerShell,
            timeout_s: 600,
            remote: None,
            staged_waiting: 0,
            background: false,
        },
        allow_always_offer: false,
    }
}

fn turn_started(turn: &str) -> ConversationEventKind {
    ConversationEventKind::TurnStarted {
        turn: turn.into(),
        kind: TurnKind::Agent,
        mode: Mode::Agent,
        label: "Local".into(),
        locality: Locality::Local,
    }
}

fn turn_ended(turn: &str) -> ConversationEventKind {
    ConversationEventKind::TurnEnded {
        turn: turn.into(),
        status: TurnStatus::Completed,
    }
}

fn joined_text(batches: &[Vec<ConversationEvent>]) -> String {
    batches
        .iter()
        .flatten()
        .filter_map(|event| match &event.kind {
            ConversationEventKind::Delta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

/// CR7 and FG6: deltas past the bound are dropped and counted, reported as
/// one `DeltasDropped{n}` when the turn finishes, and a finished turn's
/// deltas are released when the next turn finishes; `seq`s stay contiguous
/// and are never reused.
#[test]
fn deltas_are_bounded_counted_and_released_after_the_next_turn() {
    let log = new_log();
    log.push(turn_started("t1"));
    for _ in 0..MAX_TURN_DELTAS + 3 {
        log.push(delta("x"));
    }
    assert_eq!(log.dropped(), 3);
    assert_eq!(log.after(0).len(), 1 + MAX_TURN_DELTAS);
    log.finish_turn();
    log.push(turn_ended("t1"));
    let n = MAX_TURN_DELTAS as u64;
    assert_eq!(
        log.after(0).len(),
        3 + MAX_TURN_DELTAS,
        "a follower a little behind still reads the finished turn's text"
    );
    let (recorded, last) = log.recorded();
    assert_eq!(recorded.len(), 3, "a snapshot holds no delta");
    assert_eq!(last, 3 + n);
    log.push(turn_started("t2"));
    assert_eq!(log.dropped(), 0, "a new turn's bounds start afresh");
    log.push(delta("y"));
    log.finish_turn();
    log.push(turn_ended("t2"));
    let kept = log.after(0);
    let kinds: Vec<&ConversationEventKind> = kept.iter().map(|event| &event.kind).collect();
    assert!(matches!(
        kinds[0],
        ConversationEventKind::TurnStarted { .. }
    ));
    assert_eq!(kinds[1], &ConversationEventKind::DeltasDropped { n: 3 });
    assert!(matches!(kinds[2], ConversationEventKind::TurnEnded { .. }));
    assert!(matches!(
        kinds[3],
        ConversationEventKind::TurnStarted { .. }
    ));
    assert_eq!(
        kinds[4],
        &delta("y"),
        "the latest finished turn keeps its text"
    );
    assert_eq!(kept.len(), 6, "the first turn's deltas were released");
    let seqs: Vec<u64> = kept.iter().map(|event| event.seq).collect();
    assert_eq!(seqs, [1, 2 + n, 3 + n, 4 + n, 5 + n, 6 + n]);
}

/// FG5: adjacent deltas are joined (keeping the last `seq`), a command's
/// progress is coalesced to its last report, other events keep their place,
/// and a batch is cut at 256 KiB serialised.
#[test]
fn a_batch_joins_deltas_coalesces_progress_and_is_bounded() {
    let log = new_log();
    for kind in [
        delta("a"),
        delta("b"),
        ConversationEventKind::CommandProgress {
            call_id: "c1".into(),
            bytes: 10,
            lines: 1,
            tail_preview: "one".into(),
        },
        delta("c"),
        ConversationEventKind::CommandProgress {
            call_id: "c1".into(),
            bytes: 20,
            lines: 2,
            tail_preview: "two".into(),
        },
        ConversationEventKind::ToolCall {
            call_id: "t1".into(),
            tool: "read_file".into(),
            summary: "a.txt".into(),
            target: None,
        },
        delta("d"),
    ] {
        log.push(kind);
    }
    let raw = log.after(0);
    let (batch, covered) = coalesce(&raw);
    assert_eq!(covered, 7);
    let shown: Vec<(u64, String)> = batch
        .iter()
        .map(|event| (event.seq, format!("{:?}", event.kind)))
        .collect();
    println!("{shown:#?}");
    // The first progress is superseded by the second, so "c" joins "ab".
    assert_eq!(batch.len(), 4);
    assert_eq!(batch[0].kind, delta("abc"));
    assert_eq!(batch[0].seq, 4);
    assert!(matches!(
        &batch[1].kind,
        ConversationEventKind::CommandProgress { bytes: 20, .. }
    ));
    assert_eq!(batch[1].seq, 5);
    assert!(matches!(
        batch[2].kind,
        ConversationEventKind::ToolCall { .. }
    ));
    assert_eq!(batch[3].kind, delta("d"));
    // The bound: 100 KiB notices, never joined, cut into batches.
    let big = new_log();
    for _ in 0..8 {
        big.push(ConversationEventKind::Notice {
            text: "n".repeat(100 * 1024),
        });
    }
    let raw = big.after(0);
    let (first, covered) = coalesce(&raw);
    let size = serde_json::to_vec(&first).unwrap().len();
    assert!(size <= MAX_BATCH_BYTES, "{size}");
    assert_eq!(covered, 2);
    let (second, _) = coalesce(&raw[covered..]);
    assert_eq!(second[0].seq, 3, "the rest follows, contiguous");
}

/// FG1, FG6: an idle follower sends nothing; a re-follow from a batch's last
/// `seq` replays nothing twice, and together the batches cover every event.
#[test]
fn fg1_fg6_nothing_at_idle_and_nothing_twice() {
    let rt = runtime();
    let log = new_log();
    let mut stream = follow(log.clone(), 0, Gap::default(), rt.handle().clone());
    let idle = rt
        .block_on(async { tokio::time::timeout(Duration::from_millis(300), stream.next()).await });
    assert!(idle.is_err(), "no batch at idle");
    log.push(turn_started("t1"));
    log.push(delta("hello"));
    let first = rt.block_on(stream.next()).unwrap();
    let last = first.last().unwrap().seq;
    log.push(ConversationEventKind::Notice { text: "n".into() });
    log.push(turn_ended("t1"));
    let second = rt.block_on(stream.next()).unwrap();
    assert!(second.iter().all(|event| event.seq > last));
    let mut again = follow(log.clone(), last, Gap::default(), rt.handle().clone());
    let replayed = rt.block_on(again.next()).unwrap();
    assert_eq!(
        replayed, second,
        "a re-follow from the last seq replays nothing twice"
    );
    let covered: Vec<u64> = first.iter().chain(&second).map(|e| e.seq).collect();
    assert_eq!(covered, [1, 2, 3, 4], "every event, in order, once");
    drop(stream);
}

/// CB2: a model writing one character every 5 ms for 2 s gives at most 27
/// batches, the joined text exactly, and every gap between batches at least
/// 80 ms (measured at receipt, 1 ms allowed for the hand-over).
/// Mutant: no gap (every wake sends).
#[test]
fn cb2_the_gap_holds_while_text_streams() {
    let rt = runtime();
    let log = new_log();
    let stream = follow(log.clone(), 0, Gap::default(), rt.handle().clone());
    let producer = log.clone();
    let written = rt.spawn(async move {
        producer.push(turn_started("t1"));
        let start = Instant::now();
        let length = Duration::from_secs(2);
        let mut sent = 0usize;
        let mut text = String::new();
        loop {
            let due = (start.elapsed().min(length).as_micros() / 5_000) as usize;
            while sent < due {
                let piece = char::from(b'a' + (sent % 26) as u8).to_string();
                text.push_str(&piece);
                producer.push(delta(&piece));
                sent += 1;
            }
            if start.elapsed() >= length {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        producer.finish_turn();
        producer.push(turn_ended("t1"));
        text
    });
    let received = rt.block_on(async {
        let mut stream = stream;
        let mut out = Vec::new();
        while let Some(batch) = stream.next().await {
            let ended = batch
                .iter()
                .any(|e| matches!(e.kind, ConversationEventKind::TurnEnded { .. }));
            out.push((Instant::now(), batch));
            if ended {
                break;
            }
        }
        out
    });
    let text = rt.block_on(written).unwrap();
    let batches: Vec<Vec<ConversationEvent>> = received.iter().map(|(_, b)| b.clone()).collect();
    let gaps: Vec<u128> = received
        .windows(2)
        .filter(|pair| !pair[1].1.iter().any(|e| e.kind.is_urgent()))
        .map(|pair| (pair[1].0 - pair[0].0).as_millis())
        .collect();
    println!(
        "cb2: {} characters, {} batches, smallest non-urgent gap {:?} ms",
        text.len(),
        batches.len(),
        gaps.iter().min()
    );
    assert!(text.len() > 300, "the producer ran");
    assert!(batches.len() <= 27, "{} batches", batches.len());
    assert_eq!(joined_text(&batches), text, "the joined text is exact");
    assert!(
        gaps.iter().all(|gap| *gap >= MIN_GAP.as_millis() - 1),
        "{gaps:?}"
    );
}

/// CB3: an urgent event 5 ms after a batch goes at once; a delta at the
/// same moment waits for the gap (positive control).
/// Mutant: urgency ignored.
#[test]
fn cb3_an_urgent_event_does_not_wait_for_the_gap() {
    let rt = runtime();
    let measure = |urgent: bool| {
        let log = new_log();
        let mut stream = follow(log.clone(), 0, Gap::default(), rt.handle().clone());
        log.push(delta("first"));
        let _ = rt.block_on(stream.next()).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        let pushed = Instant::now();
        log.push(if urgent {
            approval("call_1")
        } else {
            delta("second")
        });
        let batch = rt.block_on(stream.next()).unwrap();
        assert_eq!(batch.len(), 1);
        pushed.elapsed()
    };
    let urgent = measure(true);
    let plain = measure(false);
    println!("cb3: urgent after {urgent:?}; a delta after {plain:?}");
    assert!(urgent < Duration::from_millis(40), "{urgent:?}");
    assert!(
        plain >= Duration::from_millis(60),
        "positive control: {plain:?}"
    );
}

/// FG3 while a gap is pending: an urgent event arriving during the wait cuts
/// it short, and goes in the same batch as the pending deltas.
#[test]
fn an_urgent_event_cuts_a_pending_gap_short() {
    let rt = runtime();
    let log = new_log();
    let mut stream = follow(log.clone(), 0, Gap::default(), rt.handle().clone());
    log.push(delta("a"));
    let _ = rt.block_on(stream.next()).unwrap();
    log.push(delta("b"));
    let next = rt.spawn(async move { stream.next().await });
    std::thread::sleep(Duration::from_millis(10));
    let pushed = Instant::now();
    log.push(ConversationEventKind::Question {
        call_id: "q1".into(),
        text: "Which?".into(),
        options: vec![],
    });
    let batch = rt.block_on(next).unwrap().unwrap();
    println!(
        "fg3: the question arrived {:?} after it was pushed",
        pushed.elapsed()
    );
    assert!(pushed.elapsed() < Duration::from_millis(50));
    assert_eq!(batch.len(), 2);
}

/// FG4: the named fallback, when enabled, lengthens the gap once the turn's
/// visible text passes the threshold.
#[test]
fn fg4_the_fallback_gap_after_long_text() {
    let rt = runtime();
    let log = new_log();
    let gap = Gap {
        min: MIN_GAP,
        slow_after: Some((10, Duration::from_millis(250))),
    };
    let mut stream = follow(log.clone(), 0, gap, rt.handle().clone());
    log.push(delta(&"x".repeat(20)));
    let _ = rt.block_on(stream.next()).unwrap();
    let pushed = Instant::now();
    log.push(delta("y"));
    let _ = rt.block_on(stream.next()).unwrap();
    println!(
        "fg4: after long text the next batch came in {:?}",
        pushed.elapsed()
    );
    assert!(pushed.elapsed() >= Duration::from_millis(200));
}

/// CB6: serialising a batch with a 400-turn conversation open costs at most
/// 1.5 times what it costs with an empty one (the fastest of many runs).
/// Mutant: the log scanned from its start for every batch.
#[test]
fn cb6_a_long_conversation_does_not_slow_a_batch() {
    let open = |turns: usize| {
        let log = new_log();
        for turn in 0..turns {
            let id = format!("t{turn}");
            log.push(turn_started(&id));
            for call in 0..3 {
                log.push(ConversationEventKind::ToolCall {
                    call_id: format!("{id}-{call}"),
                    tool: "read_file".into(),
                    summary: "src/lib.rs".into(),
                    target: Some("src/lib.rs".into()),
                });
                log.push(ConversationEventKind::ToolOutput {
                    call_id: format!("{id}-{call}"),
                    preview: "fn main() {}".into(),
                    withheld: false,
                    truncated: false,
                });
            }
            log.push(turn_ended(&id));
        }
        log.push(turn_started("now"));
        let after = log.last_seq();
        log.push(delta("a few words"));
        (log, after)
    };
    let once = |(log, after): &(Arc<ConversationLog>, u64)| {
        let start = Instant::now();
        let raw = log.after(*after);
        let (batch, _) = coalesce(&raw);
        let bytes = serde_json::to_vec(&batch).unwrap();
        std::hint::black_box(bytes);
        start.elapsed()
    };
    let (empty_log, long_log) = (open(0), open(400));
    let (mut empty, mut long) = (Duration::MAX, Duration::MAX);
    // Alternate, so a change of clock speed meets both alike; keep the
    // fastest of each.
    for _ in 0..20_000 {
        empty = empty.min(once(&empty_log));
        long = long.min(once(&long_log));
    }
    println!("cb6: empty {empty:?}, 400 turns {long:?}");
    assert!(
        long.as_nanos() * 2 <= empty.as_nanos() * 3,
        "{long:?} against {empty:?}"
    );
}

/// FG8: summary changes reach the sidebar as ids, coalesced, at least a gap
/// apart, and nothing at idle.
#[test]
fn fg8_list_changes_are_ids_only_and_coalesced() {
    let rt = runtime();
    let changed = Changed::default();
    let mut stream = changed.stream(MIN_GAP, rt.handle().clone());
    let idle = rt
        .block_on(async { tokio::time::timeout(Duration::from_millis(200), stream.next()).await });
    assert!(idle.is_err());
    changed.note("b");
    changed.note("a");
    changed.note("b");
    let first = rt.block_on(stream.next()).unwrap();
    assert_eq!(first, ["a", "b"]);
    let sent = Instant::now();
    changed.note("c");
    let second = rt.block_on(stream.next()).unwrap();
    assert_eq!(second, ["c"]);
    assert!(sent.elapsed() >= Duration::from_millis(70));
}
