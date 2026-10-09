//! The agent's browser: its policy, its WebSocket and pipe pieces, and a real
//! browser (headless Edge or Chrome) driven over a page this test serves on
//! loopback, its DevTools on its pipes and on no port.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::Arc;

use super::pipe::{self, MAX_MESSAGE};
use super::policy::{
    Effect, Target, check_url, click_looks, label_looks, misstated, parse_keys, press_looks,
    site_of, words_of,
};
use super::ws::{Incoming, accept_for, base64, frame, read_message, sha1};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ------------------------------------------------------------------ policy

#[test]
fn the_browser_opens_the_public_web_and_nothing_on_this_pc_or_its_network() {
    for good in [
        "https://www.google.com/",
        "http://example.com/a?b=c",
        "x.com/home",
        "https://203.0.113.9/",
        "about:blank",
    ] {
        assert!(check_url(good).is_ok(), "{good}: {:?}", check_url(good));
    }
    assert_eq!(check_url("x.com/home").unwrap(), "https://x.com/home");
    for bad in [
        "http://localhost:8080/",
        "http://127.0.0.1/",
        "http://127.1.2.3/",
        "http://10.0.0.5/",
        "http://172.16.4.4/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.64.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fd00::1]/",
        "http://[::ffff:192.168.1.1]/",
        "http://printer/",
        "http://nas.local/",
        "http://db.internal/",
        "http://app.localhost/",
        "https://user:pass@example.com/",
        "file:///C:/Windows/win.ini",
        "javascript:alert(1)",
        "edge://settings",
        "chrome://settings",
        "data:text/html,hi",
        "ftp://example.com/",
        "",
    ] {
        assert!(check_url(bad).is_err(), "{bad} was allowed");
    }
    assert_eq!(
        site_of("https://www.instagram.com/p/x").as_deref(),
        Some("instagram.com")
    );
    assert_eq!(
        site_of("https://mail.google.com/").as_deref(),
        Some("mail.google.com")
    );
}

#[test]
fn effects_that_others_see_money_deletion_and_accounts_ask_first() {
    for (name, asks) in [
        ("view", false),
        ("edit", false),
        ("share", true),
        ("buy", true),
        ("delete", true),
        ("account", true),
    ] {
        let effect = Effect::parse(name).unwrap();
        assert_eq!(effect.asks(), asks, "{name}");
        assert_eq!(effect.name(), name);
    }
    assert_eq!(Effect::parse(" SHARE "), Some(Effect::Share));
    assert_eq!(Effect::parse("post"), None);
}

#[test]
fn a_key_press_names_one_key_with_its_modifiers() {
    let enter = parse_keys("Enter").unwrap();
    assert_eq!(
        (enter.key.as_str(), enter.virtual_key, enter.modifiers),
        ("Enter", 13, 0)
    );
    assert_eq!(enter.text.as_deref(), Some("\r"));
    let select_all = parse_keys("Ctrl+A").unwrap();
    assert_eq!(
        (select_all.code.as_str(), select_all.modifiers),
        ("KeyA", 2)
    );
    assert_eq!(select_all.text, None, "a Ctrl chord types nothing");
    let back_tab = parse_keys("shift+tab").unwrap();
    assert_eq!((back_tab.key.as_str(), back_tab.modifiers), ("Tab", 8));
    assert_eq!(parse_keys("Shift+z").unwrap().text.as_deref(), Some("Z"));
    assert_eq!(parse_keys("7").unwrap().code, "Digit7");
    for bad in ["", "Ctrl+", "Hyper+A", "F13", "ab", "+"] {
        assert!(parse_keys(bad).is_err(), "{bad}");
    }
}

/// The second guard: a short label on a button, a link or a control that
/// reads like an action that asks; a long one is content; text the click
/// landed in is not a control.
#[test]
fn a_label_that_reads_like_posting_paying_deleting_or_an_account_change_is_told_apart() {
    assert_eq!(words_of("tweetButtonInline"), ["tweet", "button", "inline"]);
    assert_eq!(words_of("5 Likes. Like"), ["5", "likes", "like"]);
    assert_eq!(words_of("Sign-out"), ["sign", "out"]);
    let looks = |label: &str| label_looks(&words_of(label));
    for (label, effect) in [
        ("Post", Effect::Share),
        ("Reply", Effect::Share),
        ("5 Likes. Like", Effect::Share),
        ("Buy now", Effect::Buy),
        ("Place your order", Effect::Buy),
        ("Proceed to checkout", Effect::Buy),
        ("Delete post", Effect::Delete),
        ("Log out", Effect::Account),
        ("Accept all cookies", Effect::Account),
    ] {
        assert_eq!(looks(label), Some(effect), "{label}");
    }
    for label in [
        "Press",
        "Search",
        "Next",
        "Reject all",
        "Likes",
        "Comments",
        "Sign in",
        "Orders",
    ] {
        assert_eq!(looks(label), None, "{label}");
    }
    let button = |label: &str| Target {
        kind: "ok".into(),
        tag: "button".into(),
        actionable: true,
        label: label.into(),
        ..Target::default()
    };
    assert_eq!(click_looks(&button("Post")), Some(Effect::Share));
    assert_eq!(
        click_looks(&Target {
            testid: "tweetButtonInline".into(),
            ..button("")
        }),
        Some(Effect::Share),
        "a site's test id names the button"
    );
    let headline = "How to delete a repository from your account settings";
    assert_eq!(
        click_looks(&button(headline)),
        None,
        "a long label is content"
    );
    assert_eq!(
        click_looks(&Target {
            actionable: false,
            ..button("Post")
        }),
        None,
        "text the click landed in is not a control"
    );
    assert_eq!(
        click_looks(&Target {
            frame: true,
            ..button("Pay")
        }),
        None
    );
    let sentence = misstated(Effect::Share, &button("Post"));
    assert!(
        sentence.contains("(\"Post\")") && sentence.contains("as share"),
        "{sentence}"
    );
    // Keys: Enter in a message box sends; Shift+Enter is a new line; a search
    // field is looked up; Enter sends a form whose button posts.
    let enter = parse_keys("Enter").unwrap();
    let editor = Target {
        editor: true,
        label: "Write a message".into(),
        ..button("")
    };
    assert_eq!(press_looks(&enter, &editor), Some(Effect::Share));
    assert_eq!(
        press_looks(&parse_keys("Ctrl+Enter").unwrap(), &editor),
        Some(Effect::Share)
    );
    assert_eq!(
        press_looks(&parse_keys("Shift+Enter").unwrap(), &editor),
        None
    );
    assert_eq!(press_looks(&parse_keys("Tab").unwrap(), &editor), None);
    let search = Target {
        search: true,
        editor: true,
        ..button("Search")
    };
    assert_eq!(press_looks(&enter, &search), None);
    let field = Target {
        tag: "input".into(),
        submit: "Post".into(),
        ..button("Add a comment")
    };
    assert_eq!(press_looks(&enter, &field), Some(Effect::Share));
    assert_eq!(
        press_looks(&parse_keys("Space").unwrap(), &button("Delete")),
        Some(Effect::Delete)
    );
    assert_eq!(press_looks(&enter, &button("Press")), None);
}

// ------------------------------------------------------------- WebSocket

#[test]
fn sha1_base64_and_the_accept_key_match_their_published_vectors() {
    assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    assert_eq!(
        hex(&sha1(b"abc")),
        "a9993e364706816aba3e25717850c26c9cd0d89d"
    );
    assert_eq!(
        hex(&sha1(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
    );
    for (plain, coded) in [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foobar", "Zm9vYmFy"),
    ] {
        assert_eq!(base64(plain.as_bytes()), coded);
    }
    // RFC 6455 §1.3's example.
    assert_eq!(
        accept_for("dGhlIHNhbXBsZSBub25jZQ=="),
        "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    );
}

#[test]
fn a_frame_is_masked_and_a_message_is_read_whole() {
    let sent = frame(0x1, b"Hi", [1, 2, 3, 4]);
    assert_eq!(sent, vec![0x81, 0x82, 1, 2, 3, 4, b'H' ^ 1, b'i' ^ 2]);
    let long = frame(0x1, &[0u8; 300], [0; 4]);
    assert_eq!(&long[..4], &[0x81, 0x80 | 126, 0x01, 0x2c]);
    // A server's text in two fragments, a ping, a close.
    let mut bytes = vec![0x01, 3, b'a', b'b', b'c', 0x80, 2, b'd', b'e'];
    bytes.extend([0x89, 1, b'p', 0x88, 0]);
    let mut reader = &bytes[..];
    assert_eq!(
        read_message(&mut reader).unwrap(),
        Incoming::Text("abcde".to_owned())
    );
    assert_eq!(
        read_message(&mut reader).unwrap(),
        Incoming::Ping(vec![b'p'])
    );
    assert_eq!(read_message(&mut reader).unwrap(), Incoming::Close);
    assert!(read_message(&mut reader).is_err(), "the end");
    let huge = [0x81u8, 127, 0, 0, 0, 1, 0, 0, 0, 0];
    assert!(read_message(&mut &huge[..]).unwrap_err().contains("64 MiB"));
}

// ------------------------------------------------------------------ pipes

/// Chromium's pipe frames (`--remote-debugging-pipe`, JSON mode): a message
/// and a NUL, each way.
#[test]
fn a_pipe_frame_is_a_message_and_a_nul_and_a_message_is_read_whole() {
    assert_eq!(pipe::frame("{\"id\":1}").unwrap(), b"{\"id\":1}\0");
    assert!(
        pipe::frame("a\0b").is_err(),
        "a NUL would end the frame early"
    );
    // serde_json writes a NUL inside a string as \u0000: every request frames.
    let request = serde_json::json!({"id": 1, "params": {"text": "a\u{0}b"}}).to_string();
    assert_eq!(pipe::frame(&request).unwrap().last(), Some(&0), "{request}");
    // Two messages, then the end between messages.
    let bytes = b"{\"id\":1}\0{\"method\":\"Page.loadEventFired\"}\0";
    let mut reader = &bytes[..];
    assert_eq!(
        pipe::read_message(&mut reader).unwrap().as_deref(),
        Some("{\"id\":1}")
    );
    assert_eq!(
        pipe::read_message(&mut reader).unwrap().as_deref(),
        Some("{\"method\":\"Page.loadEventFired\"}")
    );
    assert_eq!(pipe::read_message(&mut reader), Ok(None));
    // A message across many reads: a buffer of one byte.
    let mut slow = std::io::BufReader::with_capacity(1, &bytes[..]);
    assert_eq!(
        pipe::read_message(&mut slow).unwrap().as_deref(),
        Some("{\"id\":1}")
    );
    assert_eq!(
        pipe::read_message(&mut slow).unwrap().as_deref(),
        Some("{\"method\":\"Page.loadEventFired\"}")
    );
    assert_eq!(pipe::read_message(&mut slow), Ok(None));
    // An empty message is one; an end inside a message and bytes that are
    // not UTF-8 are not.
    assert_eq!(pipe::read_message(&mut &b"\0"[..]), Ok(Some(String::new())));
    assert!(
        pipe::read_message(&mut &b"{\"id\""[..])
            .unwrap_err()
            .contains("inside a message")
    );
    assert!(
        pipe::read_message(&mut &b"\xff\xfe\0"[..])
            .unwrap_err()
            .contains("UTF-8")
    );
}

/// The WebSocket client's cap: a message of exactly 64 MiB is read; one byte
/// more is refused at the cap, though its NUL comes right after.
#[test]
fn a_pipe_message_is_read_up_to_64_mib_and_no_further() {
    assert_eq!(MAX_MESSAGE, 64 * 1024 * 1024);
    assert_eq!(MAX_MESSAGE, super::ws::MAX_MESSAGE);
    let at_the_cap = std::io::repeat(b'a')
        .take(MAX_MESSAGE as u64)
        .chain(&b"\0"[..]);
    let read = pipe::read_message(&mut std::io::BufReader::new(at_the_cap))
        .unwrap()
        .unwrap();
    assert_eq!(read.len(), MAX_MESSAGE);
    drop(read);
    let over = std::io::repeat(b'a')
        .take(MAX_MESSAGE as u64 + 1)
        .chain(&b"\0"[..]);
    assert!(
        pipe::read_message(&mut std::io::BufReader::new(over))
            .unwrap_err()
            .contains("64 MiB")
    );
}

/// This process's ends of two in-process pipes, as a started browser's, and
/// the stand-in browser's ends.
#[cfg(windows)]
fn stand_in_pipes() -> (
    lattice_sys::process::FdPipes,
    std::io::PipeReader,
    std::io::PipeWriter,
) {
    use std::fs::File;
    use std::os::windows::io::OwnedHandle;
    let (browser_reads, to_child) = std::io::pipe().unwrap();
    let (from_child, browser_writes) = std::io::pipe().unwrap();
    let pipes = lattice_sys::process::FdPipes {
        to_child: File::from(OwnedHandle::from(to_child)),
        from_child: File::from(OwnedHandle::from(from_child)),
    };
    (pipes, browser_reads, browser_writes)
}

/// One NUL-ended request as the stand-in reads it, NUL included.
#[cfg(windows)]
fn next_request(reader: &mut impl std::io::BufRead) -> Vec<u8> {
    let mut bytes = Vec::new();
    reader.read_until(0, &mut bytes).unwrap();
    bytes
}

/// The DevTools protocol over pipes, against a stand-in browser on two
/// in-process pipes (no browser needed): a request goes out as its JSON and
/// a NUL; an event reaches the events with its session; an answer comes back
/// by id, an error as its message; closing the connection closes what the
/// stand-in reads, and the stand-in's end then says `Lattice.closed`. On a
/// second connection, the stand-in's end ends a waiting request at once with
/// the reason, not at its timeout.
#[cfg(windows)]
#[test]
fn devtools_over_pipes_answers_by_id_hands_on_events_and_ends_with_the_pipe() {
    use std::io::BufReader;
    use std::sync::{Mutex, mpsc};
    use std::time::{Duration, Instant};

    use serde_json::{Value, json};

    use super::cdp::{Cdp, Endpoint, Event};

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let wait = Duration::from_secs(10);
    let id_of = |bytes: &[u8]| {
        serde_json::from_slice::<Value>(&bytes[..bytes.len() - 1]).unwrap()["id"]
            .as_u64()
            .unwrap()
    };
    let (pipes, browser_reads, mut browser_writes) = stand_in_pipes();
    let (seen, requests) = mpsc::channel::<Vec<u8>>();
    let stand_in = std::thread::spawn(move || {
        let mut reader = BufReader::new(browser_reads);
        let first = next_request(&mut reader);
        let event = b"{\"method\":\"Target.targetCreated\",\"params\":{\"targetInfo\":{\"type\":\"page\"}},\"sessionId\":\"S1\"}\0";
        browser_writes.write_all(event).unwrap();
        let answer = format!(
            "{{\"id\":{},\"result\":{{\"product\":\"Stand-in/1\"}}}}\0",
            id_of(&first)
        );
        browser_writes.write_all(answer.as_bytes()).unwrap();
        seen.send(first).unwrap();
        let second = next_request(&mut reader);
        let refusal = format!(
            "{{\"id\":{},\"error\":{{\"code\":-32601,\"message\":\"'Nope.method' wasn't found\"}}}}\0",
            id_of(&second)
        );
        browser_writes.write_all(refusal.as_bytes()).unwrap();
        seen.send(second).unwrap();
        // Then whatever comes, until Lattice closes the pipe.
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        seen.send(rest).unwrap();
        // The stand-in's own end closes last.
        drop(browser_writes);
    });
    let events: Arc<Mutex<Vec<Event>>> = Arc::default();
    let sink = events.clone();
    let cdp = Cdp::connect(
        Endpoint::Pipe(pipes),
        Arc::new(move |event| sink.lock().unwrap().push(event)),
    )
    .unwrap();
    let version = runtime
        .block_on(cdp.call("Browser.getVersion", json!({}), None, wait))
        .unwrap();
    assert_eq!(version, json!({"product": "Stand-in/1"}));
    assert_eq!(
        requests.recv_timeout(wait).unwrap(),
        b"{\"id\":1,\"method\":\"Browser.getVersion\",\"params\":{}}\0"
    );
    // The event came before the answer, on the same reader: it is in.
    {
        let events = events.lock().unwrap();
        assert_eq!(events.len(), 1, "{events:?}");
        assert_eq!(events[0].method, "Target.targetCreated");
        assert_eq!(events[0].session.as_deref(), Some("S1"));
    }
    let refused = runtime.block_on(cdp.call("Nope.method", json!({"a": 1}), Some("S1"), wait));
    assert_eq!(refused, Err("'Nope.method' wasn't found".to_owned()));
    assert_eq!(
        requests.recv_timeout(wait).unwrap(),
        b"{\"id\":2,\"method\":\"Nope.method\",\"params\":{\"a\":1},\"sessionId\":\"S1\"}\0"
    );
    assert_eq!(cdp.closed(), None);
    // Closing ends what the stand-in reads; it sent nothing more.
    cdp.close();
    assert_eq!(requests.recv_timeout(wait).unwrap(), Vec::<u8>::new());
    stand_in.join().unwrap();
    let started = Instant::now();
    while cdp.closed().is_none() && started.elapsed() < wait {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(cdp.closed().as_deref(), Some(pipe::CLOSED));
    assert_eq!(
        events
            .lock()
            .unwrap()
            .last()
            .map(|event| event.method.as_str()),
        Some("Lattice.closed")
    );
    assert_eq!(
        runtime.block_on(cdp.call("Browser.getVersion", json!({}), None, wait)),
        Err(pipe::CLOSED.to_owned())
    );
    // A second connection: the stand-in ends while a request waits.
    let (pipes, browser_reads, browser_writes) = stand_in_pipes();
    let stand_in = std::thread::spawn(move || {
        let mut reader = BufReader::new(browser_reads);
        next_request(&mut reader);
        drop(browser_writes);
        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
    });
    let cdp = Cdp::connect(Endpoint::Pipe(pipes), Arc::new(|_| {})).unwrap();
    let started = Instant::now();
    let outcome = runtime.block_on(cdp.call(
        "Page.navigate",
        json!({"url": "about:blank"}),
        None,
        Duration::from_secs(60),
    ));
    assert_eq!(outcome, Err(pipe::CLOSED.to_owned()));
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    // The reader's end closed the writer too: the stand-in read to its end.
    stand_in.join().unwrap();
}

/// The browser's DevTools are its pipes: its arguments name
/// `--remote-debugging-pipe` and no port, no address and no other pipes.
#[test]
fn the_browser_is_started_with_its_devtools_on_a_pipe_and_no_port() {
    let argv = super::launch::arguments(
        Path::new(r"C:\Edge\msedge.exe"),
        Path::new(r"C:\state\browser\profile"),
        true,
        super::session::VIEWPORT,
    );
    let args: Vec<String> = argv
        .iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    assert!(
        args.contains(&"--remote-debugging-pipe".to_owned()),
        "{args:?}"
    );
    assert!(
        !args
            .iter()
            .any(|arg| arg.starts_with("--remote-debugging-") && arg != "--remote-debugging-pipe"),
        "{args:?}"
    );
}

/// The port and path a browser started with `--remote-debugging-port` writes
/// in its profile (`DevToolsActivePort`): what an earlier version read, and the control
/// below still does.
#[cfg(windows)]
fn active_port(profile: &Path) -> Option<(u16, String)> {
    let text = std::fs::read_to_string(profile.join("DevToolsActivePort")).ok()?;
    let mut lines = text.lines();
    let port: u16 = lines.next()?.trim().parse().ok()?;
    let path = lines.next()?.trim().to_owned();
    (port > 0 && path.starts_with("/devtools/browser/")).then_some((port, path))
}

/// Every TCP listener, IPv4 and IPv6, that one of `pids` owns.
#[cfg(windows)]
fn listeners_of(pids: &[u32]) -> Vec<String> {
    let mut found: Vec<String> = lattice_sys::net::tcp_listeners_v4()
        .unwrap()
        .into_iter()
        .filter(|row| pids.contains(&row.pid))
        .map(|row| format!("{:?} port {} pid {}", row.address, row.port, row.pid))
        .collect();
    found.extend(
        lattice_sys::net::tcp_listeners_v6()
            .unwrap()
            .into_iter()
            .filter(|row| pids.contains(&row.pid))
            .map(|row| format!("{:?} port {} pid {}", row.address, row.port, row.pid)),
    );
    found
}

/// Each installed Edge and Chrome, for real (headless, on scratch profiles).
/// The control, started with `--remote-debugging-port=0` as an earlier version started it,
/// writes `DevToolsActivePort`, a process of its tree listens on that port,
/// and the WebSocket transport speaks DevTools to it: the checks below can
/// see a port. Started as Lattice starts it now, the browser answers DevTools
/// on its pipes, writes no `DevToolsActivePort`, and no process of its tree
/// holds a TCP listener, IPv4 or IPv6; closing the pipe ends it.
#[cfg(windows)]
#[test]
fn the_browser_speaks_devtools_on_its_pipes_and_listens_on_no_port() {
    use std::ffi::OsString;
    use std::time::{Duration, Instant};

    use lattice_sys::process::{SpawnRequest, spawn};
    use serde_json::json;

    use super::cdp::{Cdp, Endpoint};
    use super::launch;
    use super::session::VIEWPORT;
    use crate::env::ProcessEnv;
    use crate::testkit::TempDir;

    let _one = crate::testkit::one_real_browser();
    let env = ProcessEnv;
    let browsers = launch::installed_browsers(&env);
    if browsers.is_empty() {
        eprintln!("no Edge or Chrome installed: skipped");
        return;
    }
    let dir = TempDir::new("browser-pipe");
    let globals = dir.path().join("globals");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let call = Duration::from_secs(30);
    for (index, program) in browsers.iter().enumerate() {
        // The control.
        let profile = dir.path().join(format!("control-{index}"));
        std::fs::create_dir_all(&profile).unwrap();
        let mut argv = launch::arguments(program, &profile, true, VIEWPORT);
        let at = argv
            .iter()
            .position(|arg| arg == "--remote-debugging-pipe")
            .unwrap();
        argv[at] = OsString::from("--remote-debugging-port=0");
        let control = spawn(&SpawnRequest {
            program,
            argv: &argv,
            cwd: program.parent().unwrap(),
            env: &launch::environment(&env, &globals),
            limits: launch::LIMITS,
        })
        .unwrap();
        let waited = Instant::now();
        let (port, path) = loop {
            if let Some(found) = active_port(&profile) {
                break found;
            }
            assert!(
                waited.elapsed() < Duration::from_secs(30),
                "{}: no DevToolsActivePort",
                program.display()
            );
            std::thread::sleep(Duration::from_millis(100));
        };
        let listening = listeners_of(&control.process_ids().unwrap());
        assert!(
            listening
                .iter()
                .any(|row| row.contains(&format!(" port {port} "))),
            "{}: the control's port {port} is not in {listening:?}",
            program.display()
        );
        let cdp = Cdp::connect(Endpoint::WebSocket { port, path }, Arc::new(|_| {})).unwrap();
        let version = runtime
            .block_on(cdp.call("Browser.getVersion", json!({}), None, call))
            .unwrap();
        println!(
            "{} over a port: {} {}",
            program.display(),
            version["product"],
            version["protocolVersion"]
        );
        drop(cdp);
        drop(control);
        // As Lattice starts it.
        let profile = dir.path().join(format!("pipe-{index}"));
        let launch::Started { child, pipes, .. } =
            launch::start(program, &profile, true, VIEWPORT, &env, &globals).unwrap();
        let cdp = Cdp::connect(Endpoint::Pipe(pipes), Arc::new(|_| {})).unwrap();
        let version = runtime
            .block_on(cdp.call(
                "Browser.getVersion",
                json!({}),
                None,
                crate::testkit::BROWSER_START_WAIT,
            ))
            .unwrap();
        println!(
            "{} over its pipes: {} {}",
            program.display(),
            version["product"],
            version["protocolVersion"]
        );
        // A page, so the tree has its renderer too.
        runtime
            .block_on(cdp.call(
                "Target.createTarget",
                json!({"url": "about:blank"}),
                None,
                call,
            ))
            .unwrap();
        assert!(
            !profile.join("DevToolsActivePort").exists(),
            "{}: a DevToolsActivePort was written",
            program.display()
        );
        let pids = child.process_ids().unwrap();
        assert!(pids.contains(&child.pid()), "{pids:?}");
        assert_eq!(
            listeners_of(&pids),
            Vec::<String>::new(),
            "{}: its tree ({} processes) listens",
            program.display(),
            pids.len()
        );
        // Closing the pipe ends the browser: Chromium closes when its
        // DevTools pipe does.
        // 60 s bounds a browser that never ends; a loaded machine has taken
        // over 20 s to end Edge's 17 processes.
        let closed = Instant::now();
        cdp.close();
        let ended = child.wait(Some(Duration::from_secs(60))).unwrap();
        println!(
            "{}: {} processes; ended {:?} after its pipe closed, exit {ended:?}",
            program.display(),
            pids.len(),
            closed.elapsed()
        );
        assert!(
            ended.is_some(),
            "{}: still running 60 s after its pipe closed",
            program.display()
        );
    }
}

// ------------------------------------------------------------- a browser

/// Serve `/` (a button, a text field, a password field, a link, a Post
/// button, a message box and a search field) and `/two` on loopback, each
/// connection on a thread of its own, until the test ends. A browser opens
/// connections it may never send on (measured: Edge and Chrome both do), so
/// one at a time, a silent one held every page behind it.
pub(crate) fn serve() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            std::thread::spawn(move || answer(stream));
        }
    });
    port
}

/// Answer one connection of [`serve`]'s: its request, read for at most
/// 30 s, then the page it asked for.
fn answer(mut stream: std::net::TcpStream) {
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
    let mut request = [0u8; 4096];
    let n = stream.read(&mut request).unwrap_or(0);
    let head = String::from_utf8_lossy(&request[..n]);
    let page = if head.starts_with("GET /two") {
        "<html><head><title>Page two</title></head><body>two</body></html>".to_owned()
    } else if head.starts_with("GET /search") {
        // Bing's results page, as `web_search` reads it: an ad (not a web
        // result), a redirect link, a plain link, and a link that is not a web
        // address. `aHR0cHM6Ly93d3cucnVzdC1sYW5nLm9yZy9sZWFybg` is
        // https://www.rust-lang.org/learn.
        let query = head.split_whitespace().nth(1).unwrap_or("").to_owned();
        format!(
            r#"<html><head><title>{query}</title></head><body><ol id="b_results">
<li class="b_ad"><h2><a href="https://ads.example/">An ad</a></h2></li>
<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&amp;&amp;p=x&amp;u=a1aHR0cHM6Ly93d3cucnVzdC1sYW5nLm9yZy9sZWFybg&amp;ntb=1">Learn Rust</a></h2><div class="b_caption"><p>The Rust book and more.</p></div></li>
<li class="b_algo"><h2><a href="https://doc.rust-lang.org/std/">std - Rust</a></h2></li>
<li class="b_algo"><h2><a href="javascript:alert(1)">Not a page</a></h2></li>
</ol></body></html>"#
        )
    } else {
        r#"<html><head><title>Lattice browser test</title>
<style>body{margin:0} #b{position:absolute;left:100px;top:100px;width:200px;height:50px}
#t{position:absolute;left:100px;top:200px;width:300px;height:30px}
#p{position:absolute;left:100px;top:260px;width:300px;height:30px}
#a{position:absolute;left:100px;top:320px}
#post{position:absolute;left:400px;top:100px;width:100px;height:50px}
#c{position:absolute;left:100px;top:380px;width:300px;height:60px}
#s{position:absolute;left:100px;top:470px;width:300px;height:30px}</style></head>
<body><button id="b" onclick="document.title='clicked'">Press</button>
<input id="t" oninput="document.title='typed '+this.value">
<input id="p" type="password">
<a id="a" href="/two">two</a>
<button id="post" onclick="document.title='posted'">Post</button>
<textarea id="c" placeholder="Write a message" onkeydown="if(event.key==='Enter'&&!event.shiftKey){document.title='sent'}"></textarea>
<input id="s" type="search" placeholder="Search" onkeydown="if(event.key==='Enter'){document.title='searched'}"></body></html>"#
            .to_owned()
    };
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        page.len(),
        page
    );
    let _ = stream.write_all(response.as_bytes());
}

/// A connection opened and never sent on (as a browser opens them) does not
/// hold up the page: with one open and silent, a request on a second
/// connection is answered. No browser needed.
#[test]
fn a_silent_connection_does_not_hold_up_the_page() {
    use std::io::{BufRead, BufReader};
    use std::net::TcpStream;
    use std::time::Duration;

    let port = serve();
    let _silent = TcpStream::connect(("127.0.0.1", port)).unwrap();
    let mut page = TcpStream::connect(("127.0.0.1", port)).unwrap();
    page.set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    page.write_all(b"GET /two HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        .unwrap();
    let mut status = String::new();
    BufReader::new(&page)
        .read_line(&mut status)
        .expect("the page is answered while another connection is silent");
    assert_eq!(status.trim_end(), "HTTP/1.1 200 OK");
}

/// A tab the reader opens (a link of the account panel) is a tab of its own,
/// in front, and the agent keeps its page. A tab opened while nothing runs
/// starts the browser, whose first page stays the agent's (blank); once the
/// agent has a page, a tab opened beside it leaves that page where it was, and
/// the agent's next look is still of it. The address passes the agent's
/// policy: a local one is refused when local addresses are not allowed,
/// before anything starts.
#[cfg(windows)]
#[test]
fn a_tab_the_reader_opens_is_its_own_and_the_agent_keeps_its_page() {
    use std::time::{Duration, Instant};

    use crate::env::ProcessEnv;
    use crate::state::StateRoot;
    use crate::testkit::TempDir;

    let _one = crate::testkit::one_real_browser();
    let env: Arc<dyn crate::env::Env> = Arc::new(ProcessEnv);
    if super::launch::find_browser(env.as_ref()).is_none() {
        eprintln!("no Edge or Chrome installed: skipped");
        return;
    }
    let dir = TempDir::new("agent-browser-tab");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let port = serve();
    let (page, two) = (
        format!("http://127.0.0.1:{port}/"),
        format!("http://127.0.0.1:{port}/two"),
    );
    let strict = super::AgentBrowser::new(
        StateRoot::at(dir.path()),
        env.clone(),
        runtime.handle().clone(),
    );
    assert!(
        runtime
            .block_on(strict.open_tab(&two))
            .unwrap_err()
            .contains("this PC")
    );
    assert_eq!(
        strict.status(),
        super::BrowserStatus::Stopped,
        "nothing started"
    );
    let browser =
        super::AgentBrowser::new(StateRoot::at(dir.path()), env, runtime.handle().clone())
            .with_config(super::BrowserConfig {
                headless: true,
                allow_local: true,
                profile: Some(dir.path().join("profile")),
                program: None,
                start_wait: crate::testkit::BROWSER_START_WAIT,
                load_wait: crate::testkit::BROWSER_LOAD_WAIT,
            });
    // The pages' addresses, once `wanted` is among them (a new tab loads on
    // its own time).
    let pages_with = |wanted: &str| {
        let waited = Instant::now();
        loop {
            let urls = runtime.block_on(browser.page_urls());
            if urls.iter().any(|url| url == wanted) || waited.elapsed() > Duration::from_secs(15) {
                return urls;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    };
    // Nothing runs: the tab starts the browser, and the agent's page stays
    // its first, blank one.
    runtime.block_on(browser.open_tab(&two)).unwrap();
    let urls = pages_with(&two);
    assert!(urls.contains(&two), "{urls:?}");
    let shot = runtime.block_on(browser.look()).unwrap();
    assert_eq!(shot.url, "about:blank", "the agent's page");
    // The agent opens its page; a tab the reader opens beside it leaves it.
    let shot = runtime.block_on(browser.open(&page)).unwrap();
    assert_eq!(shot.title, "Lattice browser test");
    runtime.block_on(browser.open_tab(&two)).unwrap();
    let urls = pages_with(&two);
    assert_eq!(
        urls.iter().filter(|url| **url == two).count(),
        2,
        "{urls:?}"
    );
    assert!(urls.contains(&page), "{urls:?}");
    let covered = runtime.block_on(browser.agent_page_visibility());
    let shot = runtime.block_on(browser.look()).unwrap();
    let after = runtime.block_on(browser.agent_page_visibility());
    println!("the agent's page: {covered} under the reader's tab, {after} after its look");
    assert_eq!(covered, "hidden", "the reader's tab is in front of it");
    assert_eq!(after, "visible", "the agent's page is in front again");
    assert_eq!(
        (shot.url.as_str(), shot.title.as_str()),
        (page.as_str(), "Lattice browser test")
    );
    // Its next action acts on its own page, not on the tab in front.
    let shot = runtime
        .block_on(browser.click(200.0, 125.0, false, Effect::View))
        .unwrap();
    assert_eq!(shot.title, "clicked");
    runtime.block_on(browser.stop());
}

/// A real browser end to end, headless, on a scratch profile: open (its
/// DevTools on its pipes: no `DevToolsActivePort`, and no TCP listener in its
/// tree), look, click, type, the password guard, a link, back, the label guard
/// (a Post button and Enter in a message box stated as view or edit are
/// refused; stated as share they go ahead; Enter in a search field is a
/// view), and Stop ending it. The address guard refuses the same page when
/// local addresses are not allowed.
#[cfg(windows)]
#[test]
fn a_real_browser_opens_clicks_types_refuses_a_password_and_goes_back() {
    use crate::env::ProcessEnv;
    use crate::state::StateRoot;
    use crate::testkit::TempDir;

    let _one = crate::testkit::one_real_browser();
    let env: Arc<dyn crate::env::Env> = Arc::new(ProcessEnv);
    if super::launch::find_browser(env.as_ref()).is_none() {
        eprintln!("no Edge or Chrome installed: skipped");
        return;
    }
    let dir = TempDir::new("agent-browser");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let port = serve();
    let page = format!("http://127.0.0.1:{port}/");
    // Local addresses are refused unless a test allows them.
    let strict = super::AgentBrowser::new(
        StateRoot::at(dir.path()),
        env.clone(),
        runtime.handle().clone(),
    );
    assert!(
        runtime
            .block_on(strict.open(&page))
            .unwrap_err()
            .contains("this PC")
    );
    assert_eq!(
        strict.status(),
        super::BrowserStatus::Stopped,
        "nothing started"
    );
    let browser =
        super::AgentBrowser::new(StateRoot::at(dir.path()), env, runtime.handle().clone())
            .with_config(super::BrowserConfig {
                headless: true,
                allow_local: true,
                profile: Some(dir.path().join("profile")),
                program: None,
                start_wait: crate::testkit::BROWSER_START_WAIT,
                load_wait: crate::testkit::BROWSER_LOAD_WAIT,
            });
    let shot = runtime.block_on(browser.open(&page)).unwrap();
    assert_eq!(shot.title, "Lattice browser test");
    assert_eq!(shot.image.media_type, "image/png");
    assert!(shot.image.base64.starts_with("iVBORw0KGgo"), "a PNG");
    assert!(shot.summary().contains("1280 x 800"));
    // Its DevTools are its pipes: no port file, and nothing in its tree
    // listens (the checks' control is in the test above).
    assert!(
        !dir.path()
            .join("profile")
            .join("DevToolsActivePort")
            .exists()
    );
    let pids = browser.process_ids();
    assert!(pids.len() > 1, "{pids:?}");
    assert_eq!(listeners_of(&pids), Vec::<String>::new());
    let view = Effect::View;
    let shot = runtime
        .block_on(browser.click(200.0, 125.0, false, view))
        .unwrap();
    assert_eq!(shot.title, "clicked", "the click landed on the button");
    runtime
        .block_on(browser.click(250.0, 215.0, false, Effect::Edit))
        .unwrap();
    let shot = runtime.block_on(browser.type_text("hello")).unwrap();
    assert_eq!(shot.title, "typed hello");
    runtime
        .block_on(browser.click(250.0, 275.0, false, Effect::Edit))
        .unwrap();
    let refused = runtime.block_on(browser.type_text("hunter2")).unwrap_err();
    assert!(refused.contains("password"), "{refused}");
    // The label guard: Post stated as view is refused and not clicked.
    let refused = runtime
        .block_on(browser.click(450.0, 125.0, false, view))
        .unwrap_err();
    assert!(
        refused.contains("(\"Post\")") && refused.contains("as share"),
        "{refused}"
    );
    assert_eq!(
        runtime.block_on(browser.look()).unwrap().title,
        "typed hello"
    );
    let shot = runtime
        .block_on(browser.click(450.0, 125.0, false, Effect::Share))
        .unwrap();
    assert_eq!(shot.title, "posted", "stated as share, it goes ahead");
    runtime
        .block_on(browser.click(250.0, 410.0, false, Effect::Edit))
        .unwrap();
    runtime.block_on(browser.type_text("hi there")).unwrap();
    let enter = parse_keys("Enter").unwrap();
    let refused = runtime
        .block_on(browser.press(&enter, Effect::Edit))
        .unwrap_err();
    assert!(
        refused.contains("Write a message") && refused.contains("as share"),
        "{refused}"
    );
    let shot = runtime
        .block_on(browser.press(&parse_keys("Shift+Enter").unwrap(), Effect::Edit))
        .unwrap();
    assert_eq!(shot.title, "posted", "a new line sends nothing");
    let shot = runtime
        .block_on(browser.press(&enter, Effect::Share))
        .unwrap();
    assert_eq!(shot.title, "sent");
    runtime
        .block_on(browser.click(250.0, 485.0, false, view))
        .unwrap();
    runtime.block_on(browser.type_text("cats")).unwrap();
    let shot = runtime.block_on(browser.press(&enter, view)).unwrap();
    assert_eq!(shot.title, "searched", "Enter in a search field is a view");
    let shot = runtime
        .block_on(browser.click(110.0, 330.0, false, view))
        .unwrap();
    assert_eq!(shot.title, "Page two");
    let shot = runtime.block_on(browser.back()).unwrap();
    // Back may restore the page as its script left it (the back-forward
    // cache keeps "typed hello"), or load it again: either way, the first page.
    assert_eq!(shot.url, page);
    assert!(
        matches!(shot.title.as_str(), "searched" | "Lattice browser test"),
        "{}",
        shot.title
    );
    let shot = runtime
        .block_on(browser.press(&parse_keys("Tab").unwrap(), view))
        .unwrap();
    assert!(shot.url.starts_with("http://127.0.0.1:"));
    runtime.block_on(browser.stop());
    assert_eq!(browser.status(), super::BrowserStatus::Stopped);
}

/// Listen on loopback, TCP and UDP, and record what arrives: each TCP
/// request's first line, and how many UDP datagrams.
#[cfg(windows)]
struct Listeners {
    tcp: u16,
    udp: u16,
    lines: Arc<std::sync::Mutex<Vec<String>>>,
    datagrams: Arc<std::sync::atomic::AtomicUsize>,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(windows)]
impl Listeners {
    fn new() -> Self {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::time::Duration;
        let tcp = TcpListener::bind("127.0.0.1:0").unwrap();
        tcp.set_nonblocking(true).unwrap();
        let udp = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        udp.set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        let (tcp_port, udp_port) = (
            tcp.local_addr().unwrap().port(),
            udp.local_addr().unwrap().port(),
        );
        let lines = Arc::new(std::sync::Mutex::new(Vec::new()));
        let datagrams = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen, ending) = (lines.clone(), stop.clone());
        std::thread::spawn(move || {
            while !ending.load(Ordering::SeqCst) {
                match tcp.accept() {
                    Ok((mut stream, _)) => {
                        // Each connection is read on a thread of its own: a
                        // browser opens several at once, and some send nothing.
                        let seen = seen.clone();
                        std::thread::spawn(move || {
                            let _ = stream.set_nonblocking(false);
                            let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
                            let mut head = [0u8; 2048];
                            let n = stream.read(&mut head).unwrap_or(0);
                            let first = String::from_utf8_lossy(&head[..n])
                                .lines()
                                .next()
                                .unwrap_or("(a connection that sent nothing)")
                                .to_owned();
                            seen.lock().unwrap().push(first);
                            let _ = stream
                                .write_all(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n");
                        });
                    }
                    Err(_) => std::thread::sleep(Duration::from_millis(20)),
                }
            }
        });
        let (count, ending) = (datagrams.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut buffer = [0u8; 2048];
            while !ending.load(Ordering::SeqCst) {
                if udp.recv_from(&mut buffer).is_ok() {
                    count.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        Self {
            tcp: tcp_port,
            udp: udp_port,
            lines,
            datagrams,
            stop,
        }
    }

    fn seen(&self) -> (Vec<String>, usize) {
        (
            self.lines.lock().unwrap().clone(),
            self.datagrams.load(std::sync::atomic::Ordering::SeqCst),
        )
    }
}

#[cfg(windows)]
impl Drop for Listeners {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

/// A page that tries every way out it can toward the listeners, as soon as
/// it loads: a fetch, an image, a beacon, a WebSocket, a popup, WebRTC to a
/// STUN server, and, last, a navigation. No Content Security Policy: what
/// is measured is the browser's own flags.
#[cfg(windows)]
fn probe(tcp: u16, udp: u16) -> String {
    let script = r#"
const T = "http://127.0.0.1:TCP";
fetch(T + "/fetch").catch(() => {});
new Image().src = T + "/img";
try { navigator.sendBeacon(T + "/beacon", "x"); } catch (e) {}
try { new WebSocket("ws://127.0.0.1:TCP/ws"); } catch (e) {}
try { window.open(T + "/popup"); } catch (e) {}
try {
  const pc = new RTCPeerConnection({iceServers: [{urls: "stun:127.0.0.1:UDP"}]});
  pc.createDataChannel("x");
  pc.createOffer().then(offer => pc.setLocalDescription(offer));
} catch (e) {}
setTimeout(() => { location.href = T + "/navigate"; }, 1500);
"#
    .replace("TCP", &tcp.to_string())
    .replace("UDP", &udp.to_string());
    format!("<!doctype html><title>probe</title><script>{script}</script>")
}

/// The preview browser reaches nothing, for each installed Edge and Chrome
/// (headless, scratch profiles). The control, started with the agent's
/// browser's own arguments, opens the probe page and the loopback listener
/// sees its requests; the preview browser, with [`super::preview::NO_NETWORK`],
/// opens the same page and nothing arrives, TCP or UDP. The page carries no
/// policy of its own, so this is the flags alone; the policy `page` adds is
/// a second wall, checked here by its text.
#[cfg(windows)]
#[test]
fn the_preview_browser_reaches_nothing_and_the_control_does() {
    use std::time::Duration;

    use lattice_protocol::conversation::ArtifactKind;
    use serde_json::json;

    use super::cdp::{Cdp, Endpoint};
    use super::launch;
    use super::preview::{self, PreviewBrowser};
    use super::session::VIEWPORT;
    use crate::env::{Env, ProcessEnv};
    use crate::state::StateRoot;
    use crate::testkit::TempDir;

    // It starts real browsers (the preview and a control): one at a time,
    // as every such test does.
    let _one = crate::testkit::one_real_browser();

    let wrapped = preview::page(ArtifactKind::Svg, "a <b>", "<svg></svg>");
    assert!(
        wrapped.contains(preview::POLICY) && wrapped.contains("a &lt;b&gt;"),
        "{wrapped}"
    );
    let env = ProcessEnv;
    let browsers = launch::installed_browsers(&env);
    if browsers.is_empty() {
        eprintln!("no Edge or Chrome installed: skipped");
        return;
    }
    let dir = TempDir::new("browser-preview");
    let globals = dir.path().join("globals");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let settle = Duration::from_secs(6);
    for (index, program) in browsers.iter().enumerate() {
        // The control: the agent's browser's arguments.
        let control_seen = {
            let listeners = Listeners::new();
            let profile = dir.path().join(format!("control-{index}"));
            std::fs::create_dir_all(&profile).unwrap();
            let started = launch::start_with(
                program,
                &launch::arguments(program, &profile, true, VIEWPORT),
                &env,
                &globals,
            )
            .unwrap();
            let launch::Started { child, pipes, .. } = started;
            let cdp = Cdp::connect(Endpoint::Pipe(pipes), Arc::new(|_| {})).unwrap();
            runtime
                .block_on(cdp.call(
                    "Target.createTarget",
                    json!({"url": preview::data_url(&probe(listeners.tcp, listeners.udp))}),
                    None,
                    launch::START_WAIT,
                ))
                .unwrap();
            std::thread::sleep(settle);
            let seen = listeners.seen();
            drop(cdp);
            drop(child);
            seen
        };
        assert!(
            !control_seen.0.is_empty(),
            "{}: the control reached nothing, so this test shows nothing",
            program.display()
        );
        // The preview browser.
        let listeners = Listeners::new();
        let browser = PreviewBrowser::new(
            StateRoot::at(dir.path().join("state")),
            Arc::new(ProcessEnv) as Arc<dyn Env>,
            runtime.handle().clone(),
        )
        .with_test_config(
            true,
            Some(program.clone()),
            Some(dir.path().join(format!("preview-{index}"))),
        );
        runtime
            .block_on(browser.show(&probe(listeners.tcp, listeners.udp)))
            .unwrap();
        assert!(
            !runtime.block_on(browser.process_ids()).is_empty(),
            "it runs"
        );
        std::thread::sleep(settle);
        let preview_seen = listeners.seen();
        runtime.block_on(browser.stop());
        eprintln!(
            "{}: control reached {:?} and {} datagram(s); preview reached {:?} and {} datagram(s)",
            program.display(),
            control_seen.0,
            control_seen.1,
            preview_seen.0,
            preview_seen.1
        );
        assert_eq!(
            preview_seen,
            (Vec::new(), 0),
            "{}: the preview browser reached the listeners",
            program.display()
        );
    }
}

// ------------------------------------------------ page text and web search

#[test]
fn a_results_link_is_unwrapped_to_its_own_web_address() {
    use super::session::landing;
    // https://example.com/a?b=c
    assert_eq!(
        landing("https://www.bing.com/ck/a?!&&p=1&u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9hP2I9Yw&ntb=1").as_deref(),
        Some("https://example.com/a?b=c")
    );
    assert_eq!(landing("https://example.com/").as_deref(), Some("https://example.com/"));
    assert_eq!(landing("javascript:alert(1)"), None);
    // file:///C:/x
    assert_eq!(landing("https://www.bing.com/ck/a?u=a1ZmlsZTovLy9DOi94"), None);
    assert_eq!(landing("https://www.bing.com/ck/a?p=1"), None);
    assert_eq!(landing("https://www.bing.com/ck/a?u=a1***"), None);
}

/// A real browser: the page's text with its address and title, and a search
/// on a page of Bing's shape: the query sent, the ad and a non-web link left
/// out, a redirect unwrapped.
#[cfg(windows)]
#[test]
fn the_agent_reads_a_pages_text_and_searches_in_its_browser() {
    use crate::env::ProcessEnv;
    use crate::state::StateRoot;
    use crate::testkit::TempDir;

    let _one = crate::testkit::one_real_browser();
    let env: Arc<dyn crate::env::Env> = Arc::new(ProcessEnv);
    if super::launch::find_browser(env.as_ref()).is_none() {
        eprintln!("no Edge or Chrome installed: skipped");
        return;
    }
    let dir = TempDir::new("agent-browser-read");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let port = serve();
    let browser = super::AgentBrowser::new(StateRoot::at(dir.path()), env, runtime.handle().clone())
        .with_config(super::BrowserConfig {
            headless: true,
            allow_local: true,
            profile: Some(dir.path().join("profile")),
            program: None,
            start_wait: crate::testkit::BROWSER_START_WAIT,
            load_wait: crate::testkit::BROWSER_LOAD_WAIT,
        });
    runtime
        .block_on(browser.open(&format!("http://127.0.0.1:{port}/two")))
        .unwrap();
    let page = runtime.block_on(browser.page_text()).unwrap();
    assert_eq!(page.url, format!("http://127.0.0.1:{port}/two"));
    assert_eq!(page.title, "Page two");
    assert_eq!(page.text.trim(), "two");

    let results = format!("http://127.0.0.1:{port}/search");
    let hits = runtime.block_on(browser.search_at(&results, "rust lang")).unwrap();
    let urls: Vec<&str> = hits.iter().map(|h| h.url.as_str()).collect();
    assert_eq!(urls, ["https://www.rust-lang.org/learn", "https://doc.rust-lang.org/std/"]);
    assert_eq!(hits[0].title, "Learn Rust");
    assert_eq!(hits[0].snippet, "The Rust book and more.");
    // The query went in the address.
    let title = runtime.block_on(browser.page_text()).unwrap().title;
    assert_eq!(title, "/search?q=rust+lang");
    assert_eq!(
        runtime.block_on(browser.search_at(&results, "  ")).unwrap_err(),
        "Say what to search for."
    );
    runtime.block_on(browser.stop());
}

/// By hand (`--ignored`): a real search on Bing's own page.
#[cfg(windows)]
#[test]
#[ignore = "reaches www.bing.com"]
fn a_real_search_on_bing_gives_results() {
    use crate::env::ProcessEnv;
    use crate::state::StateRoot;
    use crate::testkit::TempDir;

    let _one = crate::testkit::one_real_browser();
    let env: Arc<dyn crate::env::Env> = Arc::new(ProcessEnv);
    let dir = TempDir::new("agent-browser-search");
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
    let browser = super::AgentBrowser::new(StateRoot::at(dir.path()), env, runtime.handle().clone()).with_config(
        super::BrowserConfig {
            headless: true,
            allow_local: false,
            profile: Some(dir.path().join("profile")),
            program: None,
            start_wait: crate::testkit::BROWSER_START_WAIT,
            load_wait: crate::testkit::BROWSER_LOAD_WAIT,
        },
    );
    let hits = runtime.block_on(browser.search("rust programming language")).unwrap();
    for hit in &hits {
        eprintln!("{} | {} | {}", hit.title, hit.url, hit.snippet.chars().take(60).collect::<String>());
    }
    if hits.len() < 5 {
        let page = runtime.block_on(browser.page_text()).unwrap();
        eprintln!("PAGE {} | {} | {}", page.url, page.title, page.text.chars().take(600).collect::<String>());
    }
    assert!(hits.len() >= 5, "{hits:?}");
    assert!(hits.iter().all(|h| h.url.starts_with("http") && !h.url.contains("bing.com/ck/")));
    runtime.block_on(browser.stop());
}

