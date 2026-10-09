//! The plain chat on the production store (rows G4 and G5 of
//! the chat core's spec §15.3): `ChatCore` composed as a shipped build
//! composes it ([`ChatConfig::new`]: `SharedThreadStore` on
//! `<globals>/lattice_chat`, gate G-WEB open since G5), and the privacy
//! falsifiers PF3, PF6 and PF14 (the native chat's spec §5.3) re-run on that
//! store.
//!
//! - **The production path writes the shared store.** A message to a web
//!   chat and a new chat are saved with their answers, and a rename and a pin
//!   are recorded, as `history.py` reads them. Mutant: `SHARED_WRITES =
//!   false`.
//! - **A closed gate writes nothing.** With the gate closed (the test
//!   constructor), every write a plain chat makes (a new chat, a message to a
//!   listed chat, an edit, a regenerate, a rename, a pin) refuses with the
//!   gate's sentence, and the store's folder is byte for byte what it was.
//! - **The production path is the shared store.** A shipped composition lists
//!   and opens what is on disk in the shared store; a store in memory would
//!   list nothing. Mutant: `ChatConfig::new` building the memory store.
//! - **PF3, PF6, PF14 on the shared store.** "Nothing is written" is the
//!   store's folder, byte for byte, with the store's writes open, so a write
//!   that slipped past Prepare would reach the disk; a positive control shows
//!   that a valid send does change those bytes. The same refusals come first
//!   with the gate closed too: the privacy checks run before any write is
//!   tried.
//! - **The archived list** through `ChatCore::archived`, with the reasons the
//!   reader's own record gives (S21, §5.6).
//!
//! Nothing reaches the network, a GPU or the real `globals/`: every store is
//! under the test's temporary root.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use lattice_protocol::chat::{ChatService, RegenerateRequest, SendRequest};
use lattice_protocol::conversation::ArchiveReason;
use lattice_protocol::{Locality, RefusalKind};

use super::archive;
use super::archived_by_reader;
use super::core_tests::{Harness, local_shown, shown};
use super::store::{IndexState, SharedThreadStore, uuid_ids};
use super::transcript::{NewTurn, StoreError, TranscriptStore};
use super::{ChatConfig, answer, refusals, store_gate};
use crate::env::MapEnv;

/// A hosted endpoint, ready, with its key in the environment.
const HOSTED: &str = r#"{"version": 1, "endpoints": [
    {"id": "hosted", "label": "Hosted", "base_url": "https://hosted.example.test/v1",
     "model": "big", "api_key_name": "HOSTED_API_KEY", "enabled": true}
]}"#;

/// A string the detector takes for a key, assembled so no source holds one.
fn secret() -> String {
    format!("sk-{}", "q7".repeat(12))
}

/// How the test's core records.
#[derive(Clone, Copy, PartialEq)]
enum Store {
    /// As a shipped build: `ChatConfig::new`'s own store, untouched.
    Production,
    /// The shared store with its writes open (the test constructor).
    WritesOpen,
    /// The shared store with its writes closed (the test constructor), as
    /// every shipped build was before G5.
    WritesClosed,
}

/// A core on the shared store under the fixture's own `<globals>`.
fn harness(tag: &str, env: MapEnv, store: Store) -> (Harness, PathBuf) {
    let h = Harness::with(tag, env, move |config: &mut ChatConfig, _, _| {
        // The shipped composition, exactly: what `ChatConfig::new` builds.
        let shipped = ChatConfig::new(
            config.state.clone(),
            config.env.clone(),
            config.local.clone(),
        );
        config.store_dir = shipped.store_dir.clone();
        config.store = match store {
            Store::Production => shipped.store,
            Store::WritesOpen => Arc::new(SharedThreadStore::with_gate(
                config.state.chat_dir(),
                uuid_ids(),
                true,
            )),
            Store::WritesClosed => Arc::new(SharedThreadStore::with_gate(
                config.state.chat_dir(),
                uuid_ids(),
                false,
            )),
        };
    });
    let dir = h.fixture.state.chat_dir();
    (h, dir)
}

/// A writer on `dir` with its writes open, to lay out what a web chat left.
fn seeder(dir: &Path) -> SharedThreadStore {
    SharedThreadStore::with_gate(dir, uuid_ids(), true)
}

/// Every file under the store's folder, with its bytes.
fn files(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(folder) = pending.pop() {
        let Ok(entries) = fs::read_dir(&folder) else {
            continue;
        };
        for entry in entries {
            let path = entry.unwrap().path();
            let name = path
                .strip_prefix(dir)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            if path.is_dir() {
                out.insert(format!("{name}/"), Vec::new());
                pending.push(path);
            } else {
                out.insert(name, fs::read(&path).unwrap());
            }
        }
    }
    out
}

fn send(
    h: &Harness,
    thread: Option<&str>,
    text: &str,
    choice: &str,
    shown: lattice_protocol::chat::Shown,
    edit_of: Option<&str>,
) -> Result<lattice_protocol::chat::Accepted, lattice_protocol::Refusal> {
    h.runtime.block_on(h.core.send(SendRequest {
        thread: thread.map(str::to_owned),
        text: text.into(),
        choice: choice.into(),
        edit_of: edit_of.map(str::to_owned),
        shown,
    }))
}

fn gate_refusal() -> lattice_protocol::Refusal {
    StoreError::SharedWritesOff.refusal()
}

// ------------------------------------------------------------ the production path

/// Row G4, with the gate closed (the test constructor; a shipped build's
/// store has been open since G5): a web chat is listed and opened from the
/// shared store, and every plain-chat write refuses at the gate with the
/// store's folder unchanged byte for byte. Mutant: a store built with its
/// writes open (the writes reach the disk).
#[test]
fn a_closed_gate_refuses_every_plain_chat_write_and_changes_no_byte() {
    let (h, dir) = harness("g4-closed", MapEnv::new(), Store::WritesClosed);
    assert_eq!(
        h.runtime.block_on(h.core.threads()).unwrap().store_dir,
        dir.display().to_string()
    );
    h.fixture.install("m");
    h.text("never asked");
    // What a web chat left: two turns and a pin.
    let web = seeder(&dir);
    let (row, question) = web
        .first_message("a question from the web", "auto")
        .unwrap();
    web.append_answer(
        &row.id,
        NewTurn::assistant("an answer from the web", "llamacpp:m"),
    )
    .unwrap();
    let before = files(&dir);
    // Listed and opened as it is on disk.
    let list = h.runtime.block_on(h.core.threads()).unwrap();
    assert_eq!(list.threads.len(), 1, "the web chat is listed");
    assert_eq!(list.threads[0].id, row.id);
    assert_eq!(list.threads[0].turns, 2);
    assert!(!list.index_unreadable);
    let opened = h.runtime.block_on(h.core.open(&row.id)).unwrap();
    let texts: Vec<&str> = opened.turns.iter().map(|turn| turn.text.as_str()).collect();
    assert_eq!(texts, ["a question from the web", "an answer from the web"]);
    // Every write refuses at the gate.
    let refusals = [
        (
            "new chat",
            send(&h, None, "q", "local", local_shown(), None).map(|_| ()),
        ),
        (
            "message",
            send(&h, Some(&row.id), "q", "local", local_shown(), None).map(|_| ()),
        ),
        (
            "edit",
            send(
                &h,
                Some(&row.id),
                "q2",
                "local",
                local_shown(),
                Some(&question.id),
            )
            .map(|_| ()),
        ),
        (
            "regenerate",
            h.runtime
                .block_on(h.core.regenerate(RegenerateRequest {
                    thread: row.id.clone(),
                    choice: "local".into(),
                    shown: local_shown(),
                }))
                .map(|_| ()),
        ),
        (
            "rename",
            h.runtime
                .block_on(h.core.rename(&row.id, "renamed"))
                .map(|_| ()),
        ),
        ("pin", h.runtime.block_on(h.core.pin(&row.id, "local"))),
    ];
    for (what, result) in refusals {
        println!("{what}: {result:?}");
        assert_eq!(result.unwrap_err(), gate_refusal(), "{what}");
    }
    assert_eq!(
        files(&dir),
        before,
        "the shared store is byte for byte unchanged"
    );
    assert!(h.asked().is_empty(), "no model was asked");
    assert_eq!(store_gate::REFUSAL, gate_refusal().message);
}

/// Rows G4 and G5: a shipped composition lists and opens a web chat from the
/// shared store, and its writes reach that store as `history.py` reads them:
/// a message to the web chat is saved with its answer, a new chat is listed
/// beside it, and a rename and a pin are recorded on the web chat's row.
/// Mutants: `SHARED_WRITES = false` (every write refuses at the gate);
/// `ChatConfig::new` building the memory store (the web chat is not listed).
#[test]
fn a_shipped_composition_reads_and_writes_the_shared_store() {
    let (h, dir) = harness("g4-shipped", MapEnv::new(), Store::Production);
    assert_eq!(
        h.runtime.block_on(h.core.threads()).unwrap().store_dir,
        dir.display().to_string()
    );
    h.fixture.install("m");
    h.text("a native answer");
    // What a web chat left: two turns.
    let web = seeder(&dir);
    let (row, _) = web
        .first_message("a question from the web", "auto")
        .unwrap();
    web.append_answer(
        &row.id,
        NewTurn::assistant("an answer from the web", "llamacpp:m"),
    )
    .unwrap();
    // Listed and opened as it is on disk.
    let list = h.runtime.block_on(h.core.threads()).unwrap();
    assert_eq!(list.threads.len(), 1, "the web chat is listed");
    assert_eq!(list.threads[0].id, row.id);
    assert_eq!(list.threads[0].turns, 2);
    assert!(!list.index_unreadable);
    // A message to the web chat is saved with its answer.
    let accepted = send(
        &h,
        Some(&row.id),
        "a native question",
        "local",
        local_shown(),
        None,
    )
    .unwrap();
    h.events(&accepted.job);
    let texts: Vec<String> = web
        .load(&row.id)
        .iter()
        .map(|turn| turn.text.clone())
        .collect();
    assert_eq!(
        texts,
        [
            "a question from the web",
            "an answer from the web",
            "a native question",
            "a native answer"
        ]
    );
    // A new chat is listed beside it.
    let accepted = send(&h, None, "a new chat", "local", local_shown(), None).unwrap();
    h.events(&accepted.job);
    // A rename and a pin are recorded on the web chat's row.
    h.runtime
        .block_on(h.core.rename(&row.id, "renamed"))
        .unwrap();
    h.runtime.block_on(h.core.pin(&row.id, "local")).unwrap();
    let IndexState::Rows(rows) = web.list() else {
        panic!("the shared index lists both chats")
    };
    assert_eq!(rows.len(), 2);
    let web_row = rows.iter().find(|r| r.id == row.id).unwrap();
    assert_eq!(
        (
            web_row.title.as_str(),
            web_row.pinned_provider.as_str(),
            web_row.turns
        ),
        ("renamed", "local", 4)
    );
    assert_eq!(h.asked().len(), 2, "the model answered both sends");
}

/// The visible last 400 (S7): a web chat of 405 turns, two of them
/// superseded, opens with the last 400 visible ones in order.
#[test]
fn a_long_web_chat_opens_with_its_visible_last_400_turns() {
    let (h, dir) = harness("g4-400", MapEnv::new(), Store::Production);
    let web = seeder(&dir);
    let (row, _) = web.first_message("turn 0", "auto").unwrap();
    for n in 1..405 {
        let turn = if n % 2 == 0 {
            NewTurn::user(format!("turn {n}"))
        } else {
            NewTurn::assistant(format!("turn {n}"), "llamacpp:m")
        };
        web.append(&row.id, turn).unwrap();
    }
    // Hide the last two (and keep their bytes in the file).
    let all = web.load(&row.id);
    web.supersede(&row.id, &all[all.len() - 2].id).unwrap();
    let opened = h.runtime.block_on(h.core.open(&row.id)).unwrap();
    let texts: Vec<String> = opened.turns.iter().map(|turn| turn.text.clone()).collect();
    let expected: Vec<String> = (3..403).map(|n| format!("turn {n}")).collect();
    assert_eq!(texts.len(), 400);
    assert_eq!(texts, expected);
}

// ------------------------------------------------------------ PF3, PF6, PF14

/// A valid send writes the shared store: the positive control for "the
/// store's bytes are unchanged" in PF3, PF6 and PF14 below.
#[test]
fn the_positive_control_a_valid_send_changes_the_shared_store() {
    let (h, dir) = harness("g4-control", MapEnv::new(), Store::WritesOpen);
    h.fixture.install("m");
    h.text("an answer");
    let before = files(&dir);
    let accepted = send(&h, None, "q", "local", local_shown(), None).unwrap();
    h.events(&accepted.job);
    let after = files(&dir);
    assert_ne!(after, before, "a send that is not refused reaches the disk");
    assert!(after.contains_key("index.json"));
    assert_eq!(h.store_rows(), 1);
}

impl Harness {
    /// How many rows the core's store lists.
    fn store_rows(&self) -> usize {
        match self.core_store().list() {
            IndexState::Rows(rows) => rows.len(),
            _ => 0,
        }
    }

    fn core_store(&self) -> SharedThreadStore {
        SharedThreadStore::new(self.fixture.state.chat_dir())
    }
}

/// PF3 on the shared store: Cloud, and every id the web does not route, is
/// refused before anything is written, with the store's writes open.
/// Mutant (the chat spec's): a refusal saved as an error turn.
#[test]
fn pf3_cloud_is_refused_before_anything_is_written_to_the_shared_store() {
    for store in [Store::WritesOpen, Store::WritesClosed, Store::Production] {
        let (h, dir) = harness("g4-pf3", MapEnv::new(), store);
        h.fixture.install("m");
        let web = seeder(&dir);
        let (row, _) = web.first_message("an earlier question", "local").unwrap();
        let before = files(&dir);
        for choice in ["cloud", "dev:scripted", "endpoint:", "endpoint:a b", "AUTO"] {
            for thread in [None, Some(row.id.as_str())] {
                let refusal = send(&h, thread, "q", choice, local_shown(), None).unwrap_err();
                assert_eq!(refusal.kind, RefusalKind::Invalid, "{choice}");
            }
        }
        assert_eq!(files(&dir), before, "the shared store is unchanged");
        assert!(h.asked().is_empty());
    }
}

/// PF6 on the shared store: a secret anywhere in what a remote model would
/// receive is refused before anything is written: (a) in the new question;
/// (b) in the third-previous user turn of a chat on disk. Nothing is asked,
/// the store's bytes are unchanged, and no refusal repeats the secret.
/// Mutants (the chat spec's): (a) no tripwire before the writes; (b) a
/// tripwire over the latest question only.
#[test]
fn pf6_a_secret_anywhere_never_reaches_a_remote_model_or_the_shared_store() {
    let key = secret();
    for store in [Store::WritesOpen, Store::WritesClosed, Store::Production] {
        let (h, dir) = harness(
            "g4-pf6",
            MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
            store,
        );
        h.fixture.install("m");
        h.fixture.registry(HOSTED);
        let remote = || shown(Locality::Remote, "Hosted");
        // (a)
        let before = files(&dir);
        let refusal = send(
            &h,
            None,
            &format!("my key is {key}"),
            "endpoint:hosted",
            remote(),
            None,
        )
        .unwrap_err();
        assert_eq!(refusal.kind, RefusalKind::Invalid);
        assert_eq!(refusal.message, answer::secret_sentence("Hosted"));
        assert!(!refusal.message.contains(&key));
        assert_eq!(files(&dir), before, "(a): nothing was written");
        assert!(h.asked().is_empty());
        // (b) A Local chat may hold it (Local never leaves the machine); it is
        // on disk, three user turns back.
        let web = seeder(&dir);
        let (row, _) = web
            .first_message(&format!("keep {key} safe"), "local")
            .unwrap();
        web.append_answer(&row.id, NewTurn::assistant("noted", "llamacpp:m"))
            .unwrap();
        for question in ["second", "third"] {
            web.append(&row.id, NewTurn::user(question)).unwrap();
            web.append_answer(&row.id, NewTurn::assistant("ok", "llamacpp:m"))
                .unwrap();
        }
        let before = files(&dir);
        let refusal = send(
            &h,
            Some(&row.id),
            "now ask the hosted model",
            "endpoint:hosted",
            remote(),
            None,
        )
        .unwrap_err();
        assert_eq!(refusal.message, answer::secret_sentence("Hosted"));
        assert!(!refusal.message.contains(&key));
        assert_eq!(files(&dir), before, "(b): nothing was written");
        assert!(h.asked().is_empty(), "nothing was asked");
        // A regenerate to the remote model is held to the same rule.
        let refusal = h
            .runtime
            .block_on(h.core.regenerate(RegenerateRequest {
                thread: row.id.clone(),
                choice: "endpoint:hosted".into(),
                shown: remote(),
            }))
            .unwrap_err();
        assert_eq!(refusal.message, answer::secret_sentence("Hosted"));
        assert_eq!(files(&dir), before);
    }
}

/// PF14 on the shared store: a choice that moved is refused before anything
/// is written: the old Ollama Local shown as Local, a named loopback
/// endpoint shown as Local, and an endpoint whose label changed; for a new
/// chat and for one on disk.
/// Mutant (the chat spec's): a core that ignores what was shown.
#[test]
fn pf14_a_choice_that_moved_is_refused_before_the_shared_store_is_written() {
    for store in [Store::WritesOpen, Store::WritesClosed, Store::Production] {
        let (h, dir) = harness(
            "g4-pf14",
            MapEnv::new().with("HOSTED_API_KEY", "fixture-hosted"),
            store,
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
        let web = seeder(&dir);
        let (row, _) = web.first_message("an earlier question", "local").unwrap();
        let before = files(&dir);
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
            for thread in [None, Some(row.id.as_str())] {
                let refusal = send(&h, thread, "q", choice, shown.clone(), None).unwrap_err();
                assert_eq!(
                    (refusal.kind, refusal.message.as_str()),
                    (RefusalKind::Conflict, refusals::MOVED),
                    "{choice}"
                );
            }
        }
        assert_eq!(files(&dir), before, "the shared store is unchanged");
        assert!(h.asked().is_empty());
    }
}

// ------------------------------------------------------------ the archived list

/// §5.6 through `ChatCore::archived`: the 61st thread archives the oldest
/// (`Cap`); the reader's Archive (S21) through the production store (open
/// since G5) archives another and records why (`Reader`); the list is newest
/// `evicted_at` first, 60 to a page. With the gate closed (the test
/// constructor) the reader's Archive refuses, and neither the shared store
/// nor the native record changes.
#[test]
fn the_archived_list_says_why_each_chat_is_there() {
    let (h, dir) = harness("g4-archived", MapEnv::new(), Store::Production);
    let native = h.fixture.state.native_chat_dir();
    let none = h.runtime.block_on(h.core.archived(0)).unwrap();
    assert_eq!(
        (none.total, none.archived.len(), none.archive_unreadable),
        (0, 0, false)
    );
    let web = seeder(&dir);
    let mut ids = Vec::new();
    for n in 0..61 {
        ids.push(
            web.first_message(&format!("question {n}"), "local")
                .unwrap()
                .0
                .id,
        );
    }
    // A store with its gate closed refuses the reader's Archive and records
    // nothing.
    let closed = SharedThreadStore::with_gate(&dir, uuid_ids(), false);
    let before = files(&dir);
    assert_eq!(
        archived_by_reader::archive_by_reader(&closed, &native, &ids[30], 5.0),
        Err(StoreError::SharedWritesOff)
    );
    assert_eq!(files(&dir), before);
    assert!(
        !archived_by_reader::path(&native).exists(),
        "no reason recorded"
    );
    // The production store archives it and records the reason.
    let production = SharedThreadStore::new(&dir);
    let done = archived_by_reader::archive_by_reader(&production, &native, &ids[30], 5.0).unwrap();
    assert!(done.recorded);
    let page = h.runtime.block_on(h.core.archived(0)).unwrap();
    println!("archived: {:?}", page.archived);
    assert_eq!(page.total, 2);
    let shown: Vec<(&str, ArchiveReason)> = page
        .archived
        .iter()
        .map(|row| (row.key.id.as_str(), row.reason))
        .collect();
    assert_eq!(
        shown,
        [
            (ids[30].as_str(), ArchiveReason::Reader),
            (ids[0].as_str(), ArchiveReason::Cap)
        ],
        "newest first, each with its reason"
    );
    assert!(page.archived.iter().all(|row| !row.transcript_missing));
    assert!(
        h.runtime
            .block_on(h.core.archived(1))
            .unwrap()
            .archived
            .is_empty()
    );
    // The listed chats are the other 59.
    assert_eq!(
        h.runtime.block_on(h.core.threads()).unwrap().threads.len(),
        59
    );
}

/// S20, D7 through `ChatCore::archived`: an archive index only Python can
/// read lists nothing and says so; one Python would set aside lists nothing.
#[test]
fn an_archive_index_that_cannot_be_held_lists_nothing() {
    let (h, dir) = harness("g4-archived-d7", MapEnv::new(), Store::Production);
    let evicted = dir.join(archive::EVICTED_DIR);
    fs::create_dir_all(&evicted).unwrap();
    fs::write(
        evicted.join("index.json"),
        b"[{\"id\": \"a1\", \"created\": NaN}]",
    )
    .unwrap();
    let page = h.runtime.block_on(h.core.archived(0)).unwrap();
    assert_eq!((page.total, page.archive_unreadable), (0, true));
    fs::write(evicted.join("index.json"), b"{\"not\": \"a list\"}").unwrap();
    let page = h.runtime.block_on(h.core.archived(0)).unwrap();
    assert_eq!((page.total, page.archive_unreadable), (0, false));
}
