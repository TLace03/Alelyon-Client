//! CR9: sleeps and timers appear in the chat core only where CR2 allows a
//! timer (the chat core's spec §11.1, CR2 and CR9; row E12).
//!
//! What the guard reads as a timer, in `lattice-core/src` (code only: comments,
//! literals and test modules are skipped by the shared scanner, §5.8 ND4):
//! - the words CR9 names: `sleep` (std's and tokio's), `sleep_until`,
//!   `interval` and `interval_at`;
//! - an `Instant::now` (std's or tokio's) inside a `loop`, or a `while`'s
//!   condition or body:
//!   CR9's "`Instant::now()` loops", a wait that polls the clock;
//! - and, closest to CR2's intent ("no timers at idle"; a deviation that
//!   widens CR9's list): tokio's `timeout` and `timeout_at` (by path, or
//!   imported from `tokio::time`), which arm a timer, and std's timed waits
//!   (`recv_timeout`, `recv_deadline`, `wait_timeout`, `wait_timeout_ms`,
//!   `wait_timeout_while`, `park_timeout`, `park_timeout_ms`, `sleep_ms`),
//!   which wake a thread when they expire.
//!
//! They may appear:
//! - anywhere in the files CR9 names: `fsx.rs` (the replace retries),
//!   `chat/store.rs` (the lock's 20 ms steps), `convo/follow.rs` (the gap,
//!   only while a batch is pending), `exec/run.rs` (a command's timeout),
//!   `mcp/` (a request's timeout while a server is asked something: the
//!   handshake, a tool list, a call), `browser/` (the browser's start, a
//!   DevTools request's timeout, and an action's settle while the agent
//!   acts in it), `desktop/` (an action's settle while the agent acts on the
//!   desktop in auto mode), the run manager's `manager.rs` and
//!   `manager/` (its tests), and the run store's `store.rs`;
//! - elsewhere only at the sites [`LISTED`] names by their exact text, each
//!   with the number of times it occurs and why CR2 allows it. A listed site
//!   that moves, multiplies or disappears fails the guard, so the list stays
//!   true.
//!
//! The scanner and the guard are each shown to catch a fixture before they
//! are trusted; the mutant fixture is CB1's 250 ms interval in the agent chat.

mod scanner;

use scanner::{Source, code_only, sources, without_test_modules, word_lines};

/// The files and folders CR9 names, where any timer may appear.
const ALLOWED: [&str; 10] = [
    "lattice-core/src/browser/",
    "lattice-core/src/desktop/",
    "lattice-core/src/fsx.rs",
    "lattice-core/src/chat/store.rs",
    "lattice-core/src/convo/follow.rs",
    "lattice-core/src/exec/run.rs",
    "lattice-core/src/mcp/",
    "lattice-core/src/manager.rs",
    "lattice-core/src/manager/",
    "lattice-core/src/store.rs",
];

/// Words that are timers wherever they are used as an identifier.
const WORDS: [&str; 12] = [
    "sleep",
    "sleep_until",
    "sleep_ms",
    "interval",
    "interval_at",
    "recv_timeout",
    "recv_deadline",
    "wait_timeout",
    "wait_timeout_ms",
    "wait_timeout_while",
    "park_timeout",
    "park_timeout_ms",
];

/// tokio's timeouts, read by their path (a builder's `.timeout(…)` is not one).
const TIMEOUTS: [&str; 2] = ["timeout", "timeout_at"];

/// The sites outside [`ALLOWED`]: (file, exact trimmed line, times, why).
const LISTED: [(&str, &str, usize, &str); 15] = [
    (
        "lattice-core/src/llama/bench.rs",
        "tokio::time::sleep(Duration::from_millis(200)).await;",
        1,
        "a benchmark's server that ended while loading: a moment for its log's last lines, read once for the refusal",
    ),
    (
        "lattice-core/src/llama/bench.rs",
        "if Instant::now() >= deadline {",
        1,
        "a benchmark's readiness poll, only while its own server starts (the managed server's LR7 limit, 180 s)",
    ),
    (
        "lattice-core/src/llama/bench.rs",
        "tokio::time::sleep(config.poll_gap).await;",
        1,
        "a benchmark's readiness poll's pause, only while its own server starts (LR7's poll gap)",
    ),
    (
        "lattice-core/src/chat/answer.rs",
        ".spawn(async move { tokio::time::sleep(gap).await })",
        1,
        "the development echo's pace between pieces, only while its turn runs",
    ),
    (
        "lattice-core/src/chat/jobs.rs",
        ".spawn(async move { tokio::time::sleep(rest).await })",
        1,
        "the plain chat's follow gap (convo/follow.rs's model), only while a batch is pending",
    ),
    (
        "lattice-core/src/chat/jobs.rs",
        "sent_at = Some(Instant::now());",
        1,
        "the plain chat's follower records when a batch went; it does not poll the clock",
    ),
    (
        "lattice-core/src/chat/mod.rs",
        ".spawn(async move { tokio::time::timeout(wait, all).await })",
        1,
        "shutdown's bounded wait for running jobs, once, at exit",
    ),
    (
        "lattice-core/src/convo/agent.rs",
        ".spawn(async move { tokio::time::timeout(wait, all).await })",
        1,
        "shutdown's bounded wait for running turns, once, at exit",
    ),
    (
        "lattice-core/src/llama/server.rs",
        "tokio::time::sleep(idle).await;",
        1,
        "LR7: the managed server's single idle timer, armed when the last request ends",
    ),
    (
        "lattice-core/src/llama/server.rs",
        "if tokio::time::Instant::now() >= deadline {",
        1,
        "LR7: readiness polls /health only while the server starts (180 s at most)",
    ),
    (
        "lattice-core/src/llama/server.rs",
        "tokio::time::sleep(inner.config.poll_gap).await;",
        1,
        "LR7: the readiness poll's pause, only while the server starts",
    ),
    (
        "lattice-core/src/recorder.rs",
        ".wait_timeout_while(guard, INSTALL_WAIT, |slot| slot.is_none())",
        1,
        "a run's start: its first callback waits briefly for the channel to be handed over",
    ),
    (
        "lattice-core/src/staging/review.rs",
        "std::thread::sleep(RETRY_PAUSE);",
        1,
        "CR2: a sharing-violation retry during a write (20 ms steps, as Python's _replace)",
    ),
    (
        "lattice-core/src/testkit.rs",
        ".recv_timeout(Duration::from_secs(seconds))",
        1,
        "test kit: lib.rs declares it under #[cfg(test)], so no shipped build has it",
    ),
    (
        "lattice-core/src/acp/connection.rs",
        "let result = match tokio::time::timeout(timeout, answer).await {",
        1,
        "one of the labs' agents' requests (crate::acp), only while it waits for its answer, as an MCP request's (mcp/)",
    ),
];

fn is_identifier_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `code` with the white space around every `::` removed, lines kept (as the
/// never-delete guard reads paths).
fn squeezed(code: &str) -> String {
    let chars: Vec<char> = code.chars().collect();
    let mut out = String::with_capacity(code.len());
    let mut owed = 0usize;
    let mut at = 0;
    while at < chars.len() {
        let c = chars[at];
        if c == ':' && chars.get(at + 1) == Some(&':') {
            while out.ends_with(|c: char| c.is_whitespace()) {
                if out.pop() == Some('\n') {
                    owed += 1;
                }
            }
            out.push_str("::");
            at += 2;
            while at < chars.len() && chars[at].is_whitespace() {
                if chars[at] == '\n' {
                    owed += 1;
                }
                at += 1;
            }
            continue;
        }
        out.push(c);
        if c == '\n' {
            out.extend(std::iter::repeat_n('\n', owed));
            owed = 0;
        }
        at += 1;
    }
    out.extend(std::iter::repeat_n('\n', owed));
    out
}

fn line_of(code: &str, offset: usize) -> usize {
    code[..offset].matches('\n').count() + 1
}

/// Byte offsets where `word` is used as a whole identifier.
fn word_offsets<'a>(code: &'a str, word: &'a str) -> impl Iterator<Item = usize> + 'a {
    code.match_indices(word).filter_map(move |(at, _)| {
        let before = code[..at].chars().next_back();
        let after = code[at + word.len()..].chars().next();
        (!before.is_some_and(is_identifier_char) && !after.is_some_and(is_identifier_char))
            .then_some(at)
    })
}

/// The offset of the bracket that closes the one at `open`.
fn closing(code: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (at, c) in code[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + at);
                }
            }
            _ => {}
        }
    }
    None
}

/// Every `loop` and `while`, from its keyword to the end of its body (a
/// `while` condition that reads the clock polls it too), as byte ranges.
fn loop_bodies(code: &str) -> Vec<(usize, usize)> {
    let mut bodies = Vec::new();
    for keyword in ["loop", "while"] {
        for at in word_offsets(code, keyword) {
            // The body is the first `{` outside the condition's brackets.
            let mut depth = 0i32;
            let mut open = None;
            for (k, c) in code[at + keyword.len()..].char_indices() {
                match c {
                    '(' | '[' => depth += 1,
                    ')' | ']' => depth -= 1,
                    '{' if depth <= 0 => {
                        open = Some(at + keyword.len() + k);
                        break;
                    }
                    ';' if depth <= 0 => break,
                    _ => {}
                }
            }
            if let Some(open) = open
                && let Some(close) = closing(code, open)
            {
                bodies.push((at, close));
            }
        }
    }
    bodies
}

/// Every timer in `code` (already through the scanner): (line, what).
fn timers(code: &str) -> Vec<(usize, String)> {
    let code = squeezed(code);
    let mut found: Vec<(usize, String)> = Vec::new();
    for word in WORDS {
        for line in word_lines(&code, word) {
            found.push((line, word.to_owned()));
        }
    }
    for word in TIMEOUTS {
        let path = format!("time::{word}");
        for at in word_offsets(&code, word) {
            if code[..at + word.len()].ends_with(&path) {
                found.push((line_of(&code, at), format!("tokio::{path}")));
            }
        }
    }
    // `use tokio::time::{…, timeout, …}`.
    for (at, _) in code.match_indices("tokio::time::{") {
        let open = at + "tokio::time::".len();
        if let Some(close) = closing(&code, open) {
            let names = &code[open..close];
            for word in TIMEOUTS {
                if word_offsets(names, word).next().is_some() {
                    found.push((line_of(&code, at), format!("use tokio::time::{word}")));
                }
            }
        }
    }
    let bodies = loop_bodies(&code);
    for (at, _) in code.match_indices("Instant::now") {
        let after = code[at + "Instant::now".len()..].chars().next();
        if after.is_some_and(is_identifier_char) {
            continue;
        }
        if bodies.iter().any(|(open, close)| *open < at && at < *close) {
            found.push((line_of(&code, at), "Instant::now in a loop".to_owned()));
        }
    }
    found.sort();
    found.dedup();
    found
}

fn allowed(relative: &str) -> bool {
    ALLOWED.iter().any(|allowed| {
        if allowed.ends_with('/') {
            relative.starts_with(allowed)
        } else {
            relative == *allowed
        }
    })
}

/// The timers no rule allows, and every listed site whose count is not met.
fn violations(sources: &[Source]) -> Vec<String> {
    let mut found = Vec::new();
    let mut seen = vec![0usize; LISTED.len()];
    for source in sources {
        if allowed(&source.relative) {
            continue;
        }
        let mut lines: Vec<usize> = Vec::new();
        for (line, what) in timers(&source.code) {
            let text = source.line(line);
            match LISTED
                .iter()
                .position(|(file, listed, _, _)| *file == source.relative && *listed == text)
            {
                Some(index) => {
                    if !lines.contains(&line) {
                        lines.push(line);
                        seen[index] += 1;
                    }
                }
                None => found.push(format!("{}:{line}: {what}: {text}", source.relative)),
            }
        }
    }
    for ((file, text, times, _), seen) in LISTED.iter().zip(seen) {
        if seen != *times {
            found.push(format!(
                "{file}: the listed site `{text}` occurs {seen} times, listed {times}"
            ));
        }
    }
    found
}

fn fixture(relative: &str, text: &str) -> Source {
    Source {
        relative: relative.to_owned(),
        original: text.to_owned(),
        code: without_test_modules(&code_only(text)),
    }
}

/// CR9 on the tree as it is: no timer outside the rule.
#[test]
fn cr9_no_sleep_or_timer_outside_the_allowed_places() {
    let sources = sources(&["lattice-core"]);
    let found = violations(&sources);
    println!("cr9: {} sources, {} findings", sources.len(), found.len());
    for (file, text, times, why) in LISTED {
        println!("  listed {file} x{times}: {text} ({why})");
    }
    for finding in &found {
        println!("  {finding}");
    }
    assert!(found.is_empty(), "{found:#?}");
}

/// The scanner's reading of timers: each kind is found in code, and none in
/// a comment, a string or a test module; `Instant::now` counts only inside a
/// `loop` or `while`; a builder's `.timeout(…)` is not tokio's.
#[test]
fn the_guard_reads_each_kind_of_timer_and_nothing_else() {
    let text = "\
//! A header may say std::thread::sleep and tokio::time::interval.
use tokio::time::{sleep, timeout};
pub async fn outside(rx: std::sync::mpsc::Receiver<()>) {
    let started = std::time::Instant::now();
    let _ = \"thread::sleep in a message\";
    let mut tick = tokio::time::interval(PERIOD);
    loop {
        if std::time::Instant::now() > started {
            break;
        }
    }
    while tokio::time::Instant::now() < deadline {}
    let _ = tokio::time :: timeout(wait, work).await;
    let _ = client.get(url).timeout(wait);
    let _ = rx.recv_timeout(wait);
    tokio::time::sleep_until(at).await;
}

#[cfg(test)]
mod tests {
    fn inside() {
        std::thread::sleep(PAUSE);
        loop { let _ = std::time::Instant::now(); }
    }
}
";
    let found = timers(&fixture("lattice-core/src/x.rs", text).code);
    println!("fixture findings: {found:?}");
    let expected: Vec<(usize, String)> = [
        (2, "sleep"),
        (2, "use tokio::time::timeout"),
        (6, "interval"),
        (8, "Instant::now in a loop"),
        (12, "Instant::now in a loop"),
        (13, "tokio::time::timeout"),
        (15, "recv_timeout"),
        (16, "sleep_until"),
    ]
    .into_iter()
    .map(|(line, what)| (line, what.to_owned()))
    .collect();
    assert_eq!(found, expected);
}

/// The mutant fixture (CB1's): a 250 ms interval in the agent chat is a
/// violation. A sleep in a file CR9 names is not; a listed site copied
/// within its file, or moved to another, is.
#[test]
fn the_guard_catches_a_planted_interval_and_a_moved_or_copied_site() {
    let planted = fixture(
        "lattice-core/src/convo/agent.rs",
        "async fn wake(handle: Handle) {\n    let mut tick = tokio::time::interval(Duration::from_millis(250));\n    loop { tick.tick().await; }\n}\n",
    );
    let found = violations(&[planted]);
    println!("planted fixture: {found:?}");
    assert!(
        found
            .iter()
            .any(|finding| finding.starts_with("lattice-core/src/convo/agent.rs:2: interval")),
        "{found:?}"
    );
    let allowed = fixture(
        "lattice-core/src/convo/follow.rs",
        "async fn gap() { tokio::time::sleep(rest).await; }\n",
    );
    assert!(
        violations(&[allowed])
            .iter()
            .all(|finding| !finding.contains("follow.rs")),
        "a file CR9 names may sleep"
    );
    let listed = "    std::thread::sleep(RETRY_PAUSE);\n";
    let copied = fixture(
        "lattice-core/src/staging/review.rs",
        &format!("fn a() {{\n{listed}}}\nfn b() {{\n{listed}}}\n"),
    );
    let found = violations(&[copied]);
    assert!(
        found
            .iter()
            .any(|finding| finding.contains("occurs 2 times, listed 1")),
        "{found:?}"
    );
    let moved = fixture(
        "lattice-core/src/convo/turn.rs",
        &format!("fn a() {{\n{listed}}}\n"),
    );
    let found = violations(&[moved]);
    assert!(
        found
            .iter()
            .any(|finding| finding.starts_with("lattice-core/src/convo/turn.rs:2: sleep")),
        "{found:?}"
    );
}
