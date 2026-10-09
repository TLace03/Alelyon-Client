//! The chat's privacy falsifiers (the native chat's spec §5.3, as amended by
//! the chat core's spec §22 for ADR-0041).
//!
//! Row C4 holds the choices and their resolution to them: PF1, PF2, PF4 and
//! PF15, amended so that Local and Auto are the managed llama.cpp server and
//! nothing else (LR1), and LR1's own rule that a named endpoint is never
//! Local. Row C8 runs them again through a send, where "zero model calls"
//! and "the store unchanged" can be seen, and adds PF3, PF5–PF8, PF13 and
//! PF14.
//!
//! Each test names the mutant it must catch. The mutants are applied to the
//! source by the row's mutant runs (`run_mutant_c.sh`), which record the red
//! half in their logs; no mutant is compiled into any build.

use lattice_protocol::Locality;

use super::vocab::tests::Fixture;
use super::vocab::{LocalProblem, Target, is_valid_choice};
use crate::env::MapEnv;
use crate::llama::files::BinaryProblem;

/// A hosted endpoint, ready, with its key in the environment.
const HOSTED: &str = r#"{"version": 1, "endpoints": [
    {"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1",
     "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true}
]}"#;

fn never_an_endpoint(f: &Fixture, what: &str) {
    for choice in ["local", "auto"] {
        let resolution = f.resolve(choice);
        assert!(
            matches!(resolution.target, Target::Managed(_) | Target::Refused(_)),
            "{what}: {choice} resolved to {:?}",
            resolution.target
        );
        assert_eq!(
            resolution.shown.locality,
            Locality::Local,
            "{what}: {choice}"
        );
        assert!(resolution.affirmatively_local(), "{what}: {choice}");
        assert!(
            resolution.provider.starts_with("llamacpp:"),
            "{what}: {choice}: {}",
            resolution.provider
        );
    }
}

/// PF1, amended: Local never reaches a remote address. `OLLAMA_BASE_URL`
/// points off the machine and a hosted endpoint is ready; Local and Auto
/// resolve to the managed server (or refuse), never to an endpoint.
/// Mutant: Local resolved through the registry's `ollama-local` row.
#[test]
fn local_with_a_remote_ollama_address_never_leaves_the_machine() {
    let env = MapEnv::new()
        .with("OLLAMA_BASE_URL", "http://198.51.100.7:11434")
        .with("HOSTED_API_KEY", "fixture-hosted");
    let f = Fixture::new("pf1", env);
    f.registry(HOSTED);
    never_an_endpoint(&f, "nothing installed");
    f.install("qwen3-8b");
    never_an_endpoint(&f, "installed");
    assert!(matches!(f.resolve("local").target, Target::Managed(_)));
}

/// PF2, amended: Auto never calls a remote endpoint. The only ready choice
/// besides the managed server is hosted, and the server cannot run: Auto
/// refuses with "No model on this machine is ready…".
/// Mutant: Auto taking the registry's ready endpoints first (Python's
/// `available()` order).
#[test]
fn auto_never_calls_a_remote_endpoint() {
    let f = Fixture::new(
        "pf2",
        MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
    );
    f.registry(HOSTED);
    assert!(
        f.choices(false)
            .entries
            .iter()
            .any(|e| e.choice.id == "endpoint:hosted" && e.choice.ready)
    );
    let auto = f.resolve("auto");
    assert_eq!(
        auto.target,
        Target::Refused(format!(
            "No model on this machine is ready. {}",
            LocalProblem::Binary(BinaryProblem::Missing).sentence()
        ))
    );
}

/// PF4: ids the web does not route, and native-only ids, are never valid.
/// Mutant: a grammar that allows `dev:*`.
#[test]
fn unknown_and_native_only_ids_are_refused() {
    for id in [
        "dev:scripted",
        "dev:anything",
        "endpoint:",
        "endpoint:a b",
        "AUTO",
        "Local",
        &format!("endpoint:{}", "a".repeat(81)),
    ] {
        assert!(!is_valid_choice(id, true), "{id:?}");
        assert!(!is_valid_choice(id, false), "{id:?}");
    }
}

/// A saved Ollama row under the old built-in's id, with no address of its own
/// (the built-in itself is gone since the managed server landed, so only a saved row carries it).
const SAVED_OLD_OLLAMA: &str = r#"{"version": 1, "endpoints": [
    {"id": "ollama-local", "label": "Ollama (this machine)", "kind": "ollama", "model": "qwen3", "enabled": true}
]}"#;

/// PF15, amended: whatever the old Ollama settings say, Local and Auto never
/// leave the machine, and the old Ollama Local (`endpoint:ollama-local`, now
/// a saved row) is listed Remote and refused in every state.
/// Mutant: the Ollama row's locality taken from the registry's address rule.
#[test]
fn old_ollama_settings_never_move_local_or_auto_off_the_machine() {
    let states: [(&str, MapEnv); 4] = [
        // The one state the old address rule called Local: it is what the
        // mutant would show as "on this machine".
        (
            "a loopback daemon and a local tag",
            MapEnv::new().with("OLLAMA_MODEL", "qwen3-coder:30b"),
        ),
        (
            "remote OLLAMA_BASE_URL",
            MapEnv::new().with("OLLAMA_BASE_URL", "http://198.51.100.7:11434"),
        ),
        (
            "a cloud model on a loopback daemon",
            MapEnv::new()
                .with("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                .with("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        ),
        ("a cloud model in analyst_model.json", MapEnv::new()),
    ];
    for (what, env) in states {
        let f = Fixture::new("pf15", env);
        f.install("qwen3-8b");
        f.registry(SAVED_OLD_OLLAMA);
        if what.ends_with("analyst_model.json") {
            f.choose("foo:cloud");
        }
        never_an_endpoint(&f, what);
        let choices = f.choices(false);
        let old = choices
            .entries
            .iter()
            .find(|e| e.choice.id == "endpoint:ollama-local")
            .expect("the saved Ollama row is listed");
        assert_eq!(old.choice.locality, Locality::Remote, "{what}");
        assert!(!old.choice.ready, "{what}");
        let resolution = f.resolve("endpoint:ollama-local");
        assert!(matches!(resolution.target, Target::Refused(_)), "{what}");
        assert!(!resolution.affirmatively_local(), "{what}");
    }
}

/// LR1: an endpoint a person names is never Local, even on a loopback
/// address, and a send shown as Local can never resolve to it.
/// Mutant: a named endpoint's locality taken from its address.
#[test]
fn a_named_loopback_endpoint_is_never_local() {
    let f = Fixture::new("lr1", MapEnv::new());
    f.registry(
        r#"{"version": 1, "endpoints": [
            {"id": "llamacpp", "enabled": true, "model": "x"},
            {"id": "lmstudio", "enabled": true, "model": "x"}
        ]}"#,
    );
    for id in ["endpoint:llamacpp", "endpoint:lmstudio"] {
        let resolution = f.resolve(id);
        assert!(matches!(resolution.target, Target::Endpoint(_)), "{id}");
        assert_eq!(resolution.shown.locality, Locality::Remote, "{id}");
        assert!(!resolution.affirmatively_local(), "{id}");
    }
}

// ------------------------------------------------------------------ row C8
//
// The same rules, and the rest of §5.3, through a send: the plain turn's
// pipeline (`ChatCore`) over the transcript store in memory, a model factory
// that records every client it is asked for, and (LF1) the real managed
// runtime against no server, a missing model and the stub that exits.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::BoxFuture;
use lattice_protocol::RefusalKind;
use lattice_protocol::chat::{ChatEventKind, ChatService, RegenerateRequest};

use super::answer;
use super::core_tests::{Harness, local_shown, shown};
use super::refusals;
use super::store::IndexState;
use super::transcript::TranscriptStore;
use super::vocab::{self, words};
use crate::llama::server::{LlamaRuntime, RuntimeConfig};
use crate::net::{HttpGet, HttpRequest, HttpResponse, LoopbackHttp, NetError, is_loopback_url};

/// A string the detector takes for a key, assembled so no source holds one.
fn secret() -> String {
    format!("sk-{}", "q7".repeat(12))
}

fn hosted(h: &Harness) {
    h.fixture.registry(HOSTED);
}

fn remote_shown() -> lattice_protocol::chat::Shown {
    shown(Locality::Remote, "Hosted")
}

/// Every event of every job, as text.
fn all_events(h: &Harness, jobs: &[String]) -> String {
    jobs.iter()
        .map(|job| format!("{:?}", h.events(job)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn error_message(kind: &ChatEventKind) -> String {
    match kind {
        ChatEventKind::Error { message, .. } => message.clone(),
        other => panic!("not an error: {other:?}"),
    }
}

/// PF1, through a send: with `OLLAMA_BASE_URL` remote and a hosted endpoint
/// ready, Local with no server installed writes the question and saves the
/// offline error turn; no client is built and nothing is started.
/// Mutant: Local resolved through the registry's `ollama-local` row.
#[test]
fn local_with_a_remote_ollama_address_calls_no_model() {
    let env = MapEnv::new()
        .with("OLLAMA_BASE_URL", "http://198.51.100.7:11434")
        .with("HOSTED_API_KEY", "fixture-hosted");
    let h = Harness::new("pf1-send", env);
    hosted(&h);
    h.text("never asked");
    let accepted = h.send(None, "q", "local", local_shown()).unwrap();
    let message = error_message(&h.outcome(&accepted.job));
    assert_eq!(
        message,
        answer::not_ready_sentence(LocalProblem::Binary(BinaryProblem::Missing).sentence())
    );
    assert!(h.asked().is_empty(), "zero model calls");
    assert_eq!(h.local.opens.load(std::sync::atomic::Ordering::Relaxed), 0);
}

/// PF2, through a send: Auto with only a hosted endpoint ready saves "No
/// model on this machine is ready…" and calls nothing.
/// Mutant: Auto taking the registry's ready endpoints first.
#[test]
fn auto_never_calls_a_remote_endpoint_through_a_send() {
    let h = Harness::new(
        "pf2-send",
        MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
    );
    hosted(&h);
    h.text("never asked");
    let accepted = h
        .send(None, "q", "auto", shown(Locality::Local, "Auto"))
        .unwrap();
    let message = error_message(&h.outcome(&accepted.job));
    assert!(
        message.contains("No model on this machine is ready."),
        "{message}"
    );
    assert!(h.asked().is_empty());
}

/// PF3: Cloud is refused before anything is written, and so is every id the
/// web does not route (PF4 through a send).
/// Mutant: a refusal saved as an error turn.
#[test]
fn cloud_and_unknown_ids_are_refused_before_anything_is_written() {
    let h = Harness::new("pf3", MapEnv::new());
    h.fixture.install("m");
    let before = h.snapshot();
    let files_before = files(h.fixture._dir.path());
    for choice in ["cloud", "dev:scripted", "endpoint:", "endpoint:a b", "AUTO"] {
        let refusal = h.send(None, "q", choice, local_shown()).unwrap_err();
        assert_eq!(refusal.kind, RefusalKind::Invalid, "{choice}");
    }
    assert_eq!(h.snapshot(), before);
    assert_eq!(
        files(h.fixture._dir.path()),
        files_before,
        "no file was made"
    );
    assert!(h.asked().is_empty());
}

fn files(root: &std::path::Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            out.push(path.to_string_lossy().into_owned());
            if path.is_dir() {
                stack.push(path);
            }
        }
    }
    out.sort();
    out
}

/// PF5: every pin a send, a regenerate or a pin writes is in the web's
/// vocabulary.
/// Mutant: Auto pinned as the provider it resolved to.
#[test]
fn every_pin_written_is_in_the_shared_vocabulary() {
    let h = Harness::new(
        "pf5",
        MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
    );
    h.fixture.install("m");
    hosted(&h);
    h.text("a");
    let first = h
        .send(None, "q1", "auto", shown(Locality::Local, "Auto"))
        .unwrap();
    h.events(&first.job);
    let id = first.thread.id.clone();
    h.text("b");
    let second = h
        .send(Some(&id), "q2", "auto", shown(Locality::Local, "Auto"))
        .unwrap();
    h.events(&second.job);
    h.text("c");
    let third = h
        .send(Some(&id), "q3", "endpoint:hosted", remote_shown())
        .unwrap();
    h.events(&third.job);
    h.text("d");
    let again = h
        .runtime
        .block_on(h.core.regenerate(RegenerateRequest {
            thread: id.clone(),
            choice: "local".into(),
            shown: local_shown(),
        }))
        .unwrap();
    h.events(&again.job);
    h.runtime.block_on(h.core.pin(&id, "dev:echo")).unwrap();
    let pins: Vec<String> = match h.store.list() {
        IndexState::Rows(rows) => rows.into_iter().map(|row| row.pinned_provider).collect(),
        other => panic!("{other:?}"),
    };
    assert_eq!(pins, ["dev:echo"]);
    // Each pin on the way, as the index held it after each write.
    let seen = [
        first.thread.pinned.clone(),
        second.thread.pinned,
        third.thread.pinned,
        again.thread.pinned,
    ];
    println!("pins written: {seen:?}");
    for pin in seen {
        assert!(vocab::is_valid_choice(&pin, true), "{pin}");
    }
}

/// PF6: a secret anywhere in what a remote model would receive is refused
/// before anything is written: (a) in the new question; (b) in the
/// third-previous user turn. Nothing is asked, the store is unchanged, and
/// the refusal never repeats the secret.
/// Mutants: (a) no tripwire before the writes (the answer's backstop alone);
/// (b) a tripwire over the latest question only.
#[test]
fn a_secret_anywhere_in_the_context_never_reaches_a_remote_model() {
    let h = Harness::new(
        "pf6",
        MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
    );
    h.fixture.install("m");
    hosted(&h);
    let key = secret();
    // (a)
    let before = h.snapshot();
    let refusal = h
        .send(
            None,
            &format!("my key is {key}"),
            "endpoint:hosted",
            remote_shown(),
        )
        .unwrap_err();
    assert_eq!(refusal.kind, RefusalKind::Invalid);
    assert_eq!(refusal.message, answer::secret_sentence("Hosted"));
    assert!(!refusal.message.contains(&key));
    assert_eq!(h.snapshot(), before);
    assert!(h.asked().is_empty());
    // (b) A Local thread may hold it: Local never leaves the machine.
    let mut jobs = Vec::new();
    h.text("noted");
    let first = h
        .send(None, &format!("keep {key} safe"), "local", local_shown())
        .unwrap();
    jobs.push(first.job.clone());
    h.events(&first.job);
    let id = first.thread.id.clone();
    for question in ["second", "third"] {
        h.text("ok");
        let next = h.send(Some(&id), question, "local", local_shown()).unwrap();
        jobs.push(next.job.clone());
        h.events(&next.job);
    }
    let before = h.snapshot();
    let asked = h.asked().len();
    let refusal = h
        .send(
            Some(&id),
            "now ask the hosted model",
            "endpoint:hosted",
            remote_shown(),
        )
        .unwrap_err();
    assert_eq!(refusal.message, answer::secret_sentence("Hosted"));
    assert_eq!(h.snapshot(), before, "nothing was written");
    assert_eq!(h.asked().len(), asked, "nothing was asked");
    assert!(!all_events(&h, &jobs).contains("now ask the hosted"));
}

/// PF7: no key value and no key name appears in the store, an event or a
/// refusal.
/// Mutant: the registry's status ("needs <NAME>") as a refusal.
#[test]
fn keys_never_appear_in_the_store_events_or_refusals() {
    let value = "lattice-pf7-sentinel-value";
    let h = Harness::new("pf7", MapEnv::new().with("HOSTED_API_KEY", value));
    h.fixture.registry(
        r#"{"version": 1, "endpoints": [
            {"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1", "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true},
            {"id": "keyless", "label": "Keyless", "base_url": "https://other.example.test/v1", "model": "m", "api_key_name": "MISSING_PF7_KEY", "enabled": true}
        ]}"#,
    );
    h.text("an answer");
    let one = h
        .send(None, "q", "endpoint:hosted", remote_shown())
        .unwrap();
    let two = h
        .send(
            None,
            "q",
            "endpoint:keyless",
            shown(Locality::Remote, "Keyless"),
        )
        .unwrap();
    let events = all_events(&h, &[one.job.clone(), two.job.clone()]);
    let message = error_message(&h.outcome(&two.job));
    assert_eq!(message, answer::not_ready_sentence(words::MISSING_KEY));
    let choices = h.runtime.block_on(h.core.choices());
    let everything = format!("{events}\n{}\n{choices:?}\n{message}", h.snapshot());
    for leak in [
        value,
        "HOSTED_API_KEY",
        "MISSING_PF7_KEY",
        "_API_KEY",
        "PF7_KEY",
    ] {
        assert!(!everything.contains(leak), "{leak}");
    }
}

/// A one-shot loopback server: it answers 500 with `body` to every request.
fn failing_server(body: &'static str) -> (u16, std::thread::JoinHandle<()>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buffer = [0u8; 65536];
            let _ = stream.read(&mut buffer);
            let _ = write!(
                stream,
                "HTTP/1.1 500 Internal Server Error\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
        }
    });
    (port, server)
}

/// PF8: a model error is one fixed sentence: the server's 500 body never
/// reaches the turn, an event or the store.
/// Mutant: the error worded from the transport's own text.
#[test]
fn model_errors_never_carry_transport_text() {
    let sentinel = "lattice-pf8-sentinel-body";
    let (port, server) = failing_server(sentinel);
    let h = Harness::new("pf8", MapEnv::new());
    h.fixture.registry(&format!(
        r#"{{"version": 1, "endpoints": [{{"id": "lab", "label": "Lab", "base_url": "http://127.0.0.1:{port}/v1", "model": "m", "enabled": true}}]}}"#
    ));
    let accepted = h
        .send(None, "q", "endpoint:lab", shown(Locality::Remote, "Lab"))
        .unwrap();
    let message = error_message(&h.outcome(&accepted.job));
    server.join().unwrap();
    assert_eq!(message, "The model server answered 500.");
    let everything = format!("{}\n{}", all_events(&h, &[accepted.job]), h.snapshot());
    assert!(!everything.contains(sentinel));
}

/// PF13: the managed server's client is local (no proxy); a remote
/// endpoint's is not.
/// Mutant: the managed client built without `local`.
#[test]
fn local_targets_build_proxy_free_clients() {
    let h = Harness::new(
        "pf13",
        MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
    );
    h.fixture.install("m");
    hosted(&h);
    for (choice, shown) in [
        ("local", local_shown()),
        ("auto", shown(Locality::Local, "Auto")),
        ("endpoint:hosted", remote_shown()),
    ] {
        h.text("ok");
        let accepted = h.send(None, "q", choice, shown).unwrap();
        h.events(&accepted.job);
    }
    let asked = h.asked();
    println!("{asked:?}");
    assert_eq!(
        asked.iter().map(|a| a.local).collect::<Vec<_>>(),
        [true, true, false]
    );
    assert!(
        asked[..2]
            .iter()
            .all(|a| a.base_url == "http://127.0.0.1:9/v1")
    );
}

/// PF14: a choice that moved is refused before anything is written: the old
/// Ollama Local shown as Local, a named loopback endpoint shown as Local (a
/// picker from before ADR-0041), and an endpoint whose label changed.
/// Mutant: a core that ignores what was shown.
#[test]
fn a_choice_that_moved_off_the_machine_is_refused() {
    let h = Harness::new(
        "pf14",
        MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
    );
    h.fixture.install("m");
    h.fixture.registry(
        r#"{"version": 1, "endpoints": [
            {"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1", "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true},
            {"id": "mine", "label": "My server", "base_url": "http://127.0.0.1:8000/v1", "model": "q", "enabled": true},
            {"id": "ollama-local", "label": "Ollama (this machine)", "kind": "ollama", "model": "qwen3", "enabled": true}
        ]}"#,
    );
    h.text("never asked");
    let before = h.snapshot();
    for (choice, shown) in [
        (
            "endpoint:ollama-local",
            shown(Locality::Local, "Ollama (this machine)"),
        ),
        ("endpoint:mine", shown(Locality::Local, "My server")),
        (
            "endpoint:hosted",
            shown(Locality::Remote, "Hosted (renamed since)"),
        ),
    ] {
        let refusal = h.send(None, "q", choice, shown).unwrap_err();
        assert_eq!(
            (refusal.kind, refusal.message.as_str()),
            (RefusalKind::Conflict, refusals::MOVED),
            "{choice}"
        );
    }
    assert_eq!(h.snapshot(), before);
    assert!(h.asked().is_empty());
}

/// PF15, through a send: whatever the old Ollama settings say, Local and
/// Auto ask only the managed server, whose client is local.
/// Mutant: the Ollama row's locality taken from the registry's rule.
#[test]
fn cloud_served_and_remote_ollama_never_leave_by_local_or_auto() {
    for (what, env) in [
        (
            "remote base",
            MapEnv::new().with("OLLAMA_BASE_URL", "http://198.51.100.7:11434"),
        ),
        (
            "cloud model",
            MapEnv::new().with("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        ),
    ] {
        let h = Harness::new("pf15-send", env);
        h.fixture.install("m");
        h.fixture.registry(SAVED_OLD_OLLAMA);
        for (choice, label) in [("local", "Local"), ("auto", "Auto")] {
            h.text("ok");
            let accepted = h
                .send(None, "q", choice, shown(Locality::Local, label))
                .unwrap();
            h.events(&accepted.job);
        }
        let asked = h.asked();
        assert_eq!(asked.len(), 2, "{what}");
        assert!(
            asked
                .iter()
                .all(|a| a.local && is_loopback_url(&a.base_url) && a.model == "m"),
            "{what}: {asked:?}"
        );
        let old = h
            .send(
                None,
                "q",
                "endpoint:ollama-local",
                shown(Locality::Remote, "Ollama (this machine)"),
            )
            .unwrap();
        let message = error_message(&h.outcome(&old.job));
        assert_eq!(
            message,
            answer::not_ready_sentence(words::OLLAMA_RETIRED),
            "{what}"
        );
        assert_eq!(h.asked().len(), 2, "{what}: the retired row asks nothing");
    }
}

/// Every request the real runtime makes, seen before it is made.
struct Recording {
    inner: LoopbackHttp,
    urls: Arc<Mutex<Vec<String>>>,
}

impl HttpGet for Recording {
    fn get(&self, request: HttpRequest) -> BoxFuture<'static, Result<HttpResponse, NetError>> {
        self.urls.lock().unwrap().push(request.url.clone());
        self.inner.get(request)
    }
}

/// The stub server this crate's tests build (`lattice-llama-stub`), beside
/// the test binary's folder.
fn stub_binary() -> std::path::PathBuf {
    let exe = std::env::current_exe().unwrap();
    let dir = exe.parent().and_then(|deps| deps.parent()).unwrap();
    let stub = dir.join(format!(
        "lattice-llama-stub{}",
        std::env::consts::EXE_SUFFIX
    ));
    assert!(
        stub.is_file(),
        "{} is missing: `cargo test` builds it with this crate's test targets",
        stub.display()
    );
    stub
}

/// LF1, end to end: a hosted endpoint and its key are configured and no
/// local server can start (no binary, no model file, a server that exits
/// while it loads). Local and Auto answer with an explicit offline sentence;
/// no client is built; the runtime asks nothing off the loopback address.
#[test]
fn with_no_local_server_local_and_auto_refuse_offline_through_a_send() {
    for case in ["no binary", "no model file", "exits while loading"] {
        let mut env = MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted");
        for name in ["SystemRoot", "PATH", "windir"] {
            if let Some(value) = std::env::var_os(name) {
                env.set(name, value);
            }
        }
        if case != "no binary" {
            env.set(crate::llama::files::BINARY_ENV, stub_binary());
        }
        let urls = Arc::new(Mutex::new(Vec::new()));
        let seen = urls.clone();
        let h = Harness::with("lf1-send", env, move |config, fixture, handle| {
            let temp = fixture._dir.path().join("tmp");
            std::fs::create_dir_all(&temp).unwrap();
            let mut runtime = RuntimeConfig::new(fixture.env.clone(), &fixture.state).unwrap();
            runtime.temp_roots = vec![temp];
            runtime.http = Arc::new(Recording {
                inner: LoopbackHttp::new().unwrap(),
                urls: seen,
            });
            runtime.start_timeout = Duration::from_secs(20);
            runtime.poll_gap = Duration::from_millis(50);
            config.local = Arc::new(LlamaRuntime::new(runtime, handle.clone()));
        });
        hosted(&h);
        let model = h.fixture.paths().models_dir.join("m.gguf");
        match case {
            "no binary" => {
                crate::llama::files::tests::gguf(&model);
                h.fixture.choose("m");
            }
            "no model file" => {
                h.fixture.choose("m");
            }
            _ => {
                crate::llama::files::tests::gguf(&model);
                std::fs::write(
                    model.with_file_name("m.gguf.stub.json"),
                    r#"{"exit_during_load": true, "load_ms": 300}"#,
                )
                .unwrap();
                h.fixture.choose("m");
            }
        }
        h.text("never asked");
        for (choice, label) in [("local", "Local"), ("auto", "Auto")] {
            let accepted = h
                .send(None, "q", choice, shown(Locality::Local, label))
                .unwrap();
            let message = error_message(&h.outcome(&accepted.job));
            println!("{case}: {choice}: {message}");
            assert!(
                message.contains("The local model is offline: "),
                "{case}: {choice}: {message}"
            );
        }
        assert!(h.asked().is_empty(), "{case}: no client was built");
        let urls = urls.lock().unwrap().clone();
        assert!(
            urls.iter().all(|url| is_loopback_url(url)),
            "{case}: {urls:?}"
        );
    }
}

#[test]
fn a_send_shown_local_never_reaches_a_named_endpoint_even_when_it_is_ready() {
    let h = Harness::new("lr1-send", MapEnv::new());
    h.fixture.registry(
        r#"{"version": 1, "endpoints": [{"id": "mine", "label": "Local", "base_url": "http://127.0.0.1:8000/v1", "model": "q", "enabled": true}]}"#,
    );
    h.text("never asked");
    // An endpoint someone labelled "Local", sent as the picker would show
    // the real Local: refused, because a named endpoint is never Local.
    let refusal = h
        .send(None, "q", "endpoint:mine", local_shown())
        .unwrap_err();
    assert_eq!(refusal.message, refusals::MOVED);
    assert!(h.asked().is_empty());
}
